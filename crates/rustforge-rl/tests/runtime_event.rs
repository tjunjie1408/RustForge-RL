use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rustforge_rl::runtime::event::{
    bounded_event_channel, EventDeliveryError, EventDeliveryErrorKind, StatusChanged,
    TrainingEvent, TrainingEventPublisher,
};
use rustforge_rl::runtime::trainer::{
    finalize_outcome, OutcomeSlot, TrainerStatus, TrainingOutcome,
};

#[test]
fn publisher_assigns_monotonic_sequences_and_receiver_preserves_order() {
    let (publisher, receiver, delivery) = bounded_event_channel(4, Duration::from_millis(5));
    let first = publisher
        .publish(TrainingEvent::StatusChanged(StatusChanged {
            status: TrainerStatus::Running,
        }))
        .unwrap();
    let second = publisher
        .publish(TrainingEvent::StatusChanged(StatusChanged {
            status: TrainerStatus::Paused,
        }))
        .unwrap();

    assert_eq!(first.get(), 1);
    assert_eq!(second.get(), 2);
    assert_eq!(receiver.recv().unwrap().sequence, first);
    assert_eq!(receiver.recv().unwrap().sequence, second);
    assert!(delivery.is_complete());
}

#[test]
fn saturation_and_receiver_closure_are_detectable_without_blocking_forever() {
    let (publisher, receiver, delivery) = bounded_event_channel(1, Duration::from_millis(1));
    publisher
        .publish(TrainingEvent::StatusChanged(StatusChanged {
            status: TrainerStatus::Running,
        }))
        .unwrap();
    let saturated = publisher
        .publish(TrainingEvent::StatusChanged(StatusChanged {
            status: TrainerStatus::Paused,
        }))
        .unwrap_err();
    assert_eq!(saturated.kind, EventDeliveryErrorKind::Saturated);
    assert!(!delivery.is_complete());
    assert_eq!(delivery.failed_count(), 1);

    drop(receiver);
    let closed = publisher
        .publish(TrainingEvent::StatusChanged(StatusChanged {
            status: TrainerStatus::Stopped,
        }))
        .unwrap_err();
    assert_eq!(closed.kind, EventDeliveryErrorKind::Closed);
    assert_eq!(delivery.failed_count(), 2);
}

struct InspectingFailurePublisher {
    slot: OutcomeSlot,
    observed_stored_outcome: Arc<AtomicBool>,
}

impl TrainingEventPublisher for InspectingFailurePublisher {
    fn publish(
        &self,
        _event: TrainingEvent,
    ) -> Result<rustforge_rl::runtime::event::EventSequence, EventDeliveryError> {
        self.observed_stored_outcome
            .store(self.slot.load().is_some(), Ordering::SeqCst);
        Err(EventDeliveryError {
            sequence: rustforge_rl::runtime::event::EventSequence::new(1),
            kind: EventDeliveryErrorKind::Closed,
        })
    }
}

#[test]
fn terminal_outcome_is_stored_before_notification_and_send_failure_is_auxiliary() {
    let slot = OutcomeSlot::new();
    let observed = Arc::new(AtomicBool::new(false));
    let publisher = InspectingFailurePublisher {
        slot: slot.clone(),
        observed_stored_outcome: observed.clone(),
    };
    let outcome = TrainingOutcome::completed(100, 5, Duration::from_secs(2));

    let returned = finalize_outcome(&slot, &publisher, outcome);
    assert!(observed.load(Ordering::SeqCst));
    assert_eq!(returned.status, TrainerStatus::Completed);
    assert!(!returned.event_delivery_complete);
    let stored = slot.load().unwrap();
    assert_eq!(stored.status, TrainerStatus::Completed);
    assert!(!stored.event_delivery_complete);
}
