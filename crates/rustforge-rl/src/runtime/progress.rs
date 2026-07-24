//! Coalesced latest-value progress snapshots.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use smallvec::SmallVec;

use super::trainer::{MetricId, TrainerStatus};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProgressScalar {
    pub metric: MetricId,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProgressUpdate {
    pub status: TrainerStatus,
    pub global_step: u64,
    pub episode: u64,
    pub episode_step: u64,
    pub elapsed: Duration,
    pub scalars: SmallVec<[ProgressScalar; 8]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProgressSnapshot {
    pub revision: u64,
    pub status: TrainerStatus,
    pub global_step: u64,
    pub episode: u64,
    pub episode_step: u64,
    pub elapsed: Duration,
    pub scalars: SmallVec<[ProgressScalar; 8]>,
}

struct ProgressCell {
    latest: Mutex<Arc<ProgressSnapshot>>,
    next_revision: AtomicU64,
}

pub struct ProgressPublisher {
    cell: Arc<ProgressCell>,
}

#[derive(Clone)]
pub struct ProgressReader {
    cell: Arc<ProgressCell>,
}

pub fn progress_channel() -> (ProgressPublisher, ProgressReader) {
    let initial = Arc::new(ProgressSnapshot {
        revision: 0,
        status: TrainerStatus::Running,
        global_step: 0,
        episode: 0,
        episode_step: 0,
        elapsed: Duration::ZERO,
        scalars: SmallVec::new(),
    });
    let cell = Arc::new(ProgressCell {
        latest: Mutex::new(initial),
        next_revision: AtomicU64::new(0),
    });
    (
        ProgressPublisher { cell: cell.clone() },
        ProgressReader { cell },
    )
}

impl ProgressPublisher {
    pub fn publish(&self, update: ProgressUpdate) -> u64 {
        let revision = self.cell.next_revision.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot = Arc::new(ProgressSnapshot {
            revision,
            status: update.status,
            global_step: update.global_step,
            episode: update.episode,
            episode_step: update.episode_step,
            elapsed: update.elapsed,
            scalars: update.scalars,
        });
        *lock_recover(&self.cell.latest) = snapshot;
        revision
    }
}

impl ProgressReader {
    pub fn snapshot(&self) -> Arc<ProgressSnapshot> {
        lock_recover(&self.cell.latest).clone()
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
