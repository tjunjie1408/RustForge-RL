//! Opt-in host-side accounting. No timestamp queries or extra GPU waits.
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

/// Cumulative host-observed work. Bytes describe logical transfers and physical
/// tensor storage. Scratch allocations/reuses are counted separately; bind group
/// and encoder allocations are excluded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GpuProfileCounters {
    pub submissions: u64,
    pub command_buffers: u64,
    pub tensor_reuses: u64,
    pub compute_dispatches: u64,
    pub readback_submissions: u64,
    pub uploads: u64,
    pub upload_bytes: u64,
    pub readbacks: u64,
    pub readback_bytes: u64,
    pub host_waits: u64,
    /// Wall time inside blocking device polls, not GPU kernel execution time.
    pub host_wait_ns: u64,
    pub tensor_allocations: u64,
    pub tensor_bytes: u64,
    pub parameter_allocations: u64,
    pub parameter_reuses: u64,
    pub readback_allocations: u64,
    pub readback_reuses: u64,
}

impl GpuProfileCounters {
    fn since(&self, before: &Self) -> Self {
        Self {
            submissions: self.submissions.saturating_sub(before.submissions),
            command_buffers: self.command_buffers.saturating_sub(before.command_buffers),
            tensor_reuses: self.tensor_reuses.saturating_sub(before.tensor_reuses),
            compute_dispatches: self
                .compute_dispatches
                .saturating_sub(before.compute_dispatches),
            readback_submissions: self
                .readback_submissions
                .saturating_sub(before.readback_submissions),
            uploads: self.uploads.saturating_sub(before.uploads),
            upload_bytes: self.upload_bytes.saturating_sub(before.upload_bytes),
            readbacks: self.readbacks.saturating_sub(before.readbacks),
            readback_bytes: self.readback_bytes.saturating_sub(before.readback_bytes),
            host_waits: self.host_waits.saturating_sub(before.host_waits),
            host_wait_ns: self.host_wait_ns.saturating_sub(before.host_wait_ns),
            tensor_allocations: self
                .tensor_allocations
                .saturating_sub(before.tensor_allocations),
            tensor_bytes: self.tensor_bytes.saturating_sub(before.tensor_bytes),
            parameter_allocations: self
                .parameter_allocations
                .saturating_sub(before.parameter_allocations),
            parameter_reuses: self
                .parameter_reuses
                .saturating_sub(before.parameter_reuses),
            readback_allocations: self
                .readback_allocations
                .saturating_sub(before.readback_allocations),
            readback_reuses: self.readback_reuses.saturating_sub(before.readback_reuses),
        }
    }

    fn add(&mut self, other: &Self) {
        macro_rules! add { ($($field:ident),*) => { $(self.$field = self.$field.saturating_add(other.$field);)* }; }
        add!(
            submissions,
            command_buffers,
            tensor_reuses,
            compute_dispatches,
            readback_submissions,
            uploads,
            upload_bytes,
            readbacks,
            readback_bytes,
            host_waits,
            host_wait_ns,
            tensor_allocations,
            tensor_bytes,
            parameter_allocations,
            parameter_reuses,
            readback_allocations,
            readback_reuses
        );
    }
}

/// Inclusive host-side phase accounting. Nested phase totals overlap and must
/// not be summed as an exclusive breakdown. Calls include failed/unwound scopes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GpuProfilePhase {
    pub calls: u64,
    pub host_elapsed_ns: u64,
    pub counters: GpuProfileCounters,
}

/// A consistent snapshot of one opt-in collector shared by context clones.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GpuProfile {
    pub counters: GpuProfileCounters,
    pub phases: BTreeMap<&'static str, GpuProfilePhase>,
}

#[derive(Default)]
pub(super) struct Profiler(Mutex<GpuProfile>);

impl Profiler {
    pub(super) fn record(&self, counters: GpuProfileCounters) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .counters
            .add(&counters);
    }

    pub(super) fn snapshot(&self) -> GpuProfile {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub(super) fn scope(self: &Arc<Self>, name: &'static str) -> GpuProfileScope {
        GpuProfileScope {
            profiler: Arc::clone(self),
            name,
            before: self
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .counters
                .clone(),
            started: Instant::now(),
        }
    }
}

/// Records a phase on drop. Scope attribution requires sequential use of the
/// collector: simultaneous operations by other context clones are included.
#[must_use = "hold the scope until the measured phase finishes"]
pub struct GpuProfileScope {
    profiler: Arc<Profiler>,
    name: &'static str,
    before: GpuProfileCounters,
    started: Instant,
}

impl Drop for GpuProfileScope {
    fn drop(&mut self) {
        let elapsed = elapsed_ns(self.started);
        let mut profile = self.profiler.0.lock().unwrap_or_else(|e| e.into_inner());
        let delta = profile.counters.since(&self.before);
        let phase = profile.phases.entry(self.name).or_default();
        phase.calls = phase.calls.saturating_add(1);
        phase.host_elapsed_ns = phase.host_elapsed_ns.saturating_add(elapsed);
        phase.counters.add(&delta);
    }
}

pub(super) fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_and_repeated_scopes_record_inclusive_deltas() {
        let p = Arc::new(Profiler::default());
        p.record(GpuProfileCounters {
            uploads: 9,
            ..Default::default()
        });
        {
            let _outer = p.scope("training");
            for _ in 0..2 {
                let _inner = p.scope("backward");
                p.record(GpuProfileCounters {
                    submissions: 3,
                    compute_dispatches: 3,
                    ..Default::default()
                });
            }
        }
        let snapshot = p.snapshot();
        assert_eq!(snapshot.counters.submissions, 6);
        assert_eq!(snapshot.phases["training"].calls, 1);
        assert_eq!(snapshot.phases["backward"].calls, 2);
        assert_eq!(
            snapshot.phases["training"].counters,
            snapshot.phases["backward"].counters
        );
        assert_eq!(snapshot.phases["training"].counters.uploads, 0);
        assert!(
            snapshot.phases["training"].host_elapsed_ns
                >= snapshot.phases["backward"].host_elapsed_ns
        );
    }

    #[test]
    fn unwinding_records_a_phase_without_poisoning_the_collector() {
        let p = Arc::new(Profiler::default());
        let _ = std::panic::catch_unwind({
            let p = p.clone();
            move || {
                let _scope = p.scope("failed");
                panic!("test unwind");
            }
        });
        assert_eq!(p.snapshot().phases["failed"].calls, 1);
        p.record(GpuProfileCounters {
            host_waits: 1,
            ..Default::default()
        });
        assert_eq!(p.snapshot().counters.host_waits, 1);
    }
}
