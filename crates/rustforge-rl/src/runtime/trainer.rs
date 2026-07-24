//! Generic trainer boundary and authoritative terminal outcome.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::control::TrainerControl;
use super::event::{TrainingEvent, TrainingEventPublisher};
use super::persistence::{MetricSink, PersistenceSummary};
use super::progress::ProgressPublisher;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetricId(u16);

impl MetricId {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricKind {
    Gauge,
    Counter,
    Rate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub id: MetricId,
    pub name: String,
    pub label: String,
    pub unit: Option<String>,
    pub kind: MetricKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainerCapabilities {
    pub pause_resume: bool,
    pub graceful_stop: bool,
    pub force_stop: bool,
    pub checkpoint: bool,
}

#[derive(Clone, Debug)]
pub struct TrainerMetadata {
    pub algorithm: String,
    pub environment: String,
    pub run_id: String,
    pub capabilities: TrainerCapabilities,
    pub metrics: Vec<MetricDescriptor>,
}

pub trait Trainer: Send + 'static {
    fn metadata(&self) -> TrainerMetadata;
    fn run(self: Box<Self>, context: TrainerContext) -> Result<TrainingSummary, TrainerError>;
}

pub struct TrainerContext {
    pub events: Box<dyn TrainingEventPublisher>,
    pub progress: ProgressPublisher,
    pub control: TrainerControl,
    pub metrics: Box<dyn MetricSink>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainerStatus {
    Running,
    Paused,
    Stopping,
    Stopped,
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Completed,
    GracefulStop,
    ForceStop,
    Failed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingSummary {
    pub total_steps: u64,
    pub total_episodes: u64,
    pub elapsed: Duration,
    pub stop_reason: StopReason,
}

impl TrainingSummary {
    pub fn stopped(
        total_steps: u64,
        total_episodes: u64,
        elapsed: Duration,
        reason: StopReason,
    ) -> Self {
        Self {
            total_steps,
            total_episodes,
            elapsed,
            stop_reason: reason,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingOutcome {
    pub status: TrainerStatus,
    pub summary: TrainingSummary,
    pub persistence: PersistenceSummary,
    pub event_delivery_complete: bool,
    pub error: Option<String>,
}

impl TrainingOutcome {
    pub fn completed(total_steps: u64, total_episodes: u64, elapsed: Duration) -> Self {
        Self {
            status: TrainerStatus::Completed,
            summary: TrainingSummary {
                total_steps,
                total_episodes,
                elapsed,
                stop_reason: StopReason::Completed,
            },
            persistence: PersistenceSummary::complete(),
            event_delivery_complete: true,
            error: None,
        }
    }

    pub fn failed(message: impl Into<String>, summary: TrainingSummary) -> Self {
        Self {
            status: TrainerStatus::Failed,
            summary,
            persistence: PersistenceSummary::complete(),
            event_delivery_complete: true,
            error: Some(message.into()),
        }
    }
}

#[derive(Clone, Default)]
pub struct OutcomeSlot {
    inner: Arc<Mutex<Option<TrainingOutcome>>>,
}

impl OutcomeSlot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store(&self, outcome: TrainingOutcome) {
        *lock_recover(&self.inner) = Some(outcome);
    }

    pub fn load(&self) -> Option<TrainingOutcome> {
        lock_recover(&self.inner).clone()
    }
}

pub fn finalize_outcome<P: TrainingEventPublisher + ?Sized>(
    slot: &OutcomeSlot,
    publisher: &P,
    mut outcome: TrainingOutcome,
) -> TrainingOutcome {
    slot.store(outcome.clone());
    let terminal_event = TrainingEvent::from_terminal(
        outcome.status,
        outcome.summary.clone(),
        outcome.error.clone(),
    );
    if publisher.publish(terminal_event).is_err() {
        outcome.event_delivery_complete = false;
        slot.store(outcome.clone());
    }
    outcome
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainerError {
    pub message: String,
}

impl fmt::Display for TrainerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for TrainerError {}
