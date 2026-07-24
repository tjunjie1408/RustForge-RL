//! Ordered, bounded, semantically reliable training facts.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crossbeam_channel::{Receiver, SendTimeoutError, Sender};
use smallvec::SmallVec;

use super::control::ControlResolution;
use super::persistence::{PersistenceFailure, PersistenceRecovery};
use super::trainer::{MetricId, TrainerStatus, TrainingSummary};

pub const DEFAULT_EVENT_CAPACITY: usize = 1024;
pub const DEFAULT_EVENT_PUBLISH_WAIT: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventSequence(u64);

impl EventSequence {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct EventEnvelope {
    pub sequence: EventSequence,
    pub emitted_at: SystemTime,
    pub event: TrainingEvent,
}

#[derive(Clone, Debug)]
pub enum TrainingEvent {
    Started(TrainingStarted),
    ControlApplied(ControlResolution),
    StatusChanged(StatusChanged),
    EpisodeCompleted(EpisodeSummary),
    PersistenceError(PersistenceFailure),
    PersistenceRecovered(PersistenceRecovery),
    Finished(TrainingFinished),
    Failed(TrainingFailure),
}

#[derive(Clone, Debug)]
pub struct TrainingStarted {
    pub run_id: String,
    pub algorithm: String,
    pub environment: String,
}

#[derive(Clone, Debug)]
pub struct StatusChanged {
    pub status: TrainerStatus,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricValue {
    pub metric: MetricId,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeSummary {
    pub episode: u64,
    pub global_step: u64,
    pub length: u64,
    pub metrics: SmallVec<[MetricValue; 8]>,
}

#[derive(Clone, Debug)]
pub struct TrainingFinished {
    pub summary: TrainingSummary,
}

#[derive(Clone, Debug)]
pub struct TrainingFailure {
    pub message: String,
    pub summary: TrainingSummary,
}

pub trait TrainingEventPublisher: Send {
    fn publish(&self, event: TrainingEvent) -> Result<EventSequence, EventDeliveryError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventDeliveryErrorKind {
    Saturated,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventDeliveryError {
    pub sequence: EventSequence,
    pub kind: EventDeliveryErrorKind,
}

impl fmt::Display for EventDeliveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "training event {} delivery {:?}",
            self.sequence.get(),
            self.kind
        )
    }
}

impl std::error::Error for EventDeliveryError {}

#[derive(Clone)]
pub struct EventDeliveryState {
    inner: Arc<EventDeliveryStateInner>,
}

struct EventDeliveryStateInner {
    complete: AtomicBool,
    failed_count: AtomicU64,
}

impl EventDeliveryState {
    fn new() -> Self {
        Self {
            inner: Arc::new(EventDeliveryStateInner {
                complete: AtomicBool::new(true),
                failed_count: AtomicU64::new(0),
            }),
        }
    }

    fn record_failure(&self) {
        self.inner.complete.store(false, Ordering::Release);
        self.inner.failed_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn is_complete(&self) -> bool {
        self.inner.complete.load(Ordering::Acquire)
    }

    pub fn failed_count(&self) -> u64 {
        self.inner.failed_count.load(Ordering::Relaxed)
    }
}

pub struct BoundedEventPublisher {
    sender: Sender<EventEnvelope>,
    next_sequence: AtomicU64,
    wait: Duration,
    delivery: EventDeliveryState,
}

impl TrainingEventPublisher for BoundedEventPublisher {
    fn publish(&self, event: TrainingEvent) -> Result<EventSequence, EventDeliveryError> {
        let sequence = EventSequence::new(self.next_sequence.fetch_add(1, Ordering::Relaxed) + 1);
        let envelope = EventEnvelope {
            sequence,
            emitted_at: SystemTime::now(),
            event,
        };
        match self.sender.send_timeout(envelope, self.wait) {
            Ok(()) => Ok(sequence),
            Err(SendTimeoutError::Timeout(_)) => {
                self.delivery.record_failure();
                Err(EventDeliveryError {
                    sequence,
                    kind: EventDeliveryErrorKind::Saturated,
                })
            }
            Err(SendTimeoutError::Disconnected(_)) => {
                self.delivery.record_failure();
                Err(EventDeliveryError {
                    sequence,
                    kind: EventDeliveryErrorKind::Closed,
                })
            }
        }
    }
}

pub fn bounded_event_channel(
    capacity: usize,
    wait: Duration,
) -> (
    BoundedEventPublisher,
    Receiver<EventEnvelope>,
    EventDeliveryState,
) {
    let (sender, receiver) = crossbeam_channel::bounded(capacity);
    let delivery = EventDeliveryState::new();
    (
        BoundedEventPublisher {
            sender,
            next_sequence: AtomicU64::new(0),
            wait,
            delivery: delivery.clone(),
        },
        receiver,
        delivery,
    )
}

impl TrainingEvent {
    pub(crate) fn from_terminal(
        status: TrainerStatus,
        summary: TrainingSummary,
        error: Option<String>,
    ) -> Self {
        if status == TrainerStatus::Failed {
            Self::Failed(TrainingFailure {
                message: error.unwrap_or_else(|| "training failed".into()),
                summary,
            })
        } else {
            Self::Finished(TrainingFinished { summary })
        }
    }
}
