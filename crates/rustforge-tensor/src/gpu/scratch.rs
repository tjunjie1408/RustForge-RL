//! Bounded scratch-buffer leases. A buffer cannot be borrowed twice concurrently.
use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

struct Cached {
    buffer: wgpu::Buffer,
    bytes: u64,
}

pub(super) struct BufferPool {
    cached: Mutex<Vec<Cached>>,
    max_buffers: usize,
    max_bytes: u64,
}

impl BufferPool {
    pub(super) fn new(max_buffers: usize, max_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            cached: Mutex::new(Vec::new()),
            max_buffers,
            max_bytes,
        })
    }

    pub(super) fn take(self: &Arc<Self>, bytes: u64) -> Option<BufferLease> {
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        let index = cached
            .iter()
            .enumerate()
            .filter(|(_, b)| b.bytes >= bytes)
            .min_by_key(|(_, b)| b.bytes)
            .map(|(i, _)| i)?;
        let buffer = cached.swap_remove(index);
        Some(self.lease(buffer.buffer, buffer.bytes))
    }

    pub(super) fn lease(self: &Arc<Self>, buffer: wgpu::Buffer, bytes: u64) -> BufferLease {
        BufferLease {
            pool: Arc::clone(self),
            buffer: Some(buffer),
            bytes,
            reusable: true,
        }
    }
}

pub(super) struct BufferLease {
    pool: Arc<BufferPool>,
    buffer: Option<wgpu::Buffer>,
    bytes: u64,
    // Readbacks enable reuse only after successful map/copy/unmap completion.
    pub(super) reusable: bool,
}

impl Deref for BufferLease {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().expect("live buffer lease")
    }
}

impl Drop for BufferLease {
    fn drop(&mut self) {
        if !self.reusable || self.bytes > self.pool.max_bytes {
            return;
        }
        let mut cached = self.pool.cached.lock().unwrap_or_else(|e| e.into_inner());
        let retained: u64 = cached.iter().map(|b| b.bytes).sum();
        if cached.len() < self.pool.max_buffers && retained + self.bytes <= self.pool.max_bytes {
            cached.push(Cached {
                buffer: self.buffer.take().expect("live buffer lease"),
                bytes: self.bytes,
            });
        }
    }
}
