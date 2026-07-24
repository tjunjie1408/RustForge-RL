use rustforge_rl::runtime::persistence::{
    PersistenceHealth, PersistenceStatus, PersistenceTracker,
};

#[test]
fn persistence_failures_are_reported_on_transitions_not_every_write() {
    let mut tracker = PersistenceTracker::new();
    let first = tracker.record_failure("disk full");
    assert!(first.is_some());
    assert_eq!(tracker.health(), PersistenceHealth::Degraded);
    assert!(tracker.record_failure("still full").is_none());

    let summary = tracker.summary();
    assert!(!summary.complete);
    assert_eq!(summary.failures, 2);
    assert_eq!(summary.first_error.as_deref(), Some("disk full"));
    assert_eq!(summary.last_error.as_deref(), Some("still full"));

    assert!(tracker.record_recovered().is_some());
    assert_eq!(tracker.health(), PersistenceHealth::Healthy);
    assert!(tracker.record_recovered().is_none());
}

#[test]
fn shared_persistence_status_retains_the_authoritative_summary() {
    let status = PersistenceStatus::new();
    let mut tracker = PersistenceTracker::new();
    tracker.record_failure("disk full");
    tracker.record_failure("still full");
    status.store(tracker.summary());

    let summary = status.load();
    assert!(!summary.complete);
    assert_eq!(summary.failures, 2);
    assert_eq!(summary.first_error.as_deref(), Some("disk full"));
    assert_eq!(summary.last_error.as_deref(), Some("still full"));
}
