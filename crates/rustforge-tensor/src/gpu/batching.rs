//! Thread-local, bounded queue submission groups. Leases outlive deferred consumers.
use super::{scratch::BufferLease, GpuContext, GpuProfileCounters, GpuTensor};
use std::{cell::RefCell, collections::HashMap, marker::PhantomData, rc::Rc, sync::Arc};

#[derive(Default)]
struct Pending {
    depth: usize,
    commands: Vec<wgpu::CommandBuffer>,
    uniforms: Vec<BufferLease>,
    storage: Vec<Arc<BufferLease>>,
}
thread_local! {
    static PENDING: RefCell<HashMap<usize, Pending>> = RefCell::new(HashMap::new());
}

/// Queues compute command buffers together on this thread and ownership scope.
/// Nested scopes share a batch. Downloads, explicit waits and index uploads flush
/// pending work; bounded batches also flush after 32 command buffers. Drop flushes
/// during unwinding. The guard cannot move to another thread.
/// End the producing batch before passing its outputs to another thread; a
/// consumer on another thread cannot flush this thread's pending commands.
pub struct GpuCommandBatch {
    context: GpuContext,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for GpuCommandBatch {
    fn drop(&mut self) {
        let pending = PENDING.with(|b| {
            let mut b = b.borrow_mut();
            let entry = b.get_mut(&self.context.batch_key()).expect("active batch");
            entry.depth -= 1;
            (entry.depth == 0).then(|| b.remove(&self.context.batch_key()).unwrap())
        });
        if let Some(pending) = pending {
            self.context.submit_pending(pending);
        }
    }
}
impl GpuContext {
    /// Groups compute queue submissions without adding a host wait. Serialize
    /// profiler collectors for useful attribution; actual submissions are counted
    /// by the context which opens or flushes the scope. Other threads remain independent.
    pub fn command_batch(&self) -> GpuCommandBatch {
        PENDING.with(|b| b.borrow_mut().entry(self.batch_key()).or_default().depth += 1);
        GpuCommandBatch {
            context: self.clone(),
            _thread: PhantomData,
        }
    }
    fn batch_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
    pub(super) fn flush_batch(&self) {
        let pending = PENDING.with(|b| {
            b.borrow_mut().get_mut(&self.batch_key()).map(|p| Pending {
                depth: 0,
                commands: std::mem::take(&mut p.commands),
                uniforms: std::mem::take(&mut p.uniforms),
                storage: std::mem::take(&mut p.storage),
            })
        });
        if let Some(pending) = pending {
            self.submit_pending(pending);
        }
    }
    fn submit_pending(&self, pending: Pending) {
        if pending.commands.is_empty() {
            return;
        }
        self.inner.queue.submit(pending.commands);
        self.record_profile(GpuProfileCounters {
            submissions: 1,
            ..Default::default()
        });
        // Uniforms/storage cannot return to caches until the consumer is submitted.
        drop(pending.uniforms);
        drop(pending.storage);
    }
    pub(super) fn submit_compute(
        &self,
        encoder: wgpu::CommandEncoder,
        dispatches: u64,
        parameters: Option<BufferLease>,
        tensors: &[&GpuTensor],
    ) {
        let mut command = Some(encoder.finish());
        let mut parameters = parameters;
        let deferred = PENDING.with(|b| {
            let mut b = b.borrow_mut();
            if let Some(pending) = b.get_mut(&self.batch_key()) {
                pending.commands.push(command.take().unwrap());
                pending.uniforms.extend(parameters.take());
                pending
                    .storage
                    .extend(tensors.iter().map(|t| Arc::clone(&t.buffer)));
                true
            } else {
                false
            }
        });
        self.record_profile(GpuProfileCounters {
            compute_dispatches: dispatches,
            command_buffers: 1,
            ..Default::default()
        });
        if !deferred {
            self.inner.queue.submit(command);
            self.record_profile(GpuProfileCounters {
                submissions: 1,
                ..Default::default()
            });
        } else {
            let full = PENDING.with(|b| b.borrow()[&self.batch_key()].commands.len() >= 32);
            if full {
                self.flush_batch();
            }
        }
    }
}
