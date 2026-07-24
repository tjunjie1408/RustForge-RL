use std::time::Duration;

use rustforge_dashboard::analytics::{
    estimate_eta, is_stalled, observed_rates, reward_alerts, RewardAlertKind,
};

#[test]
fn observed_rates_use_only_progress_since_attach() {
    let rates = observed_rates(100, 500, 4, 14, Duration::from_secs(20)).unwrap();
    assert_eq!(rates.steps_per_second, 20.0);
    assert_eq!(rates.episodes_per_minute, 30.0);
}

#[test]
fn eta_requires_a_known_target_and_positive_finite_rate() {
    assert_eq!(
        estimate_eta(400, Some(1_000), 20.0),
        Some(Duration::from_secs(30))
    );
    assert_eq!(estimate_eta(400, None, 20.0), None);
    assert_eq!(estimate_eta(400, Some(1_000), 0.0), None);
    assert_eq!(estimate_eta(400, Some(1_000), f64::NAN), None);
    assert_eq!(estimate_eta(1_000, Some(1_000), 20.0), Some(Duration::ZERO));
}

#[test]
fn stall_detection_only_applies_to_running_sources() {
    assert!(is_stalled(
        true,
        Duration::from_secs(31),
        Duration::from_secs(30)
    ));
    assert!(!is_stalled(
        false,
        Duration::from_secs(31),
        Duration::from_secs(30)
    ));
}

#[test]
fn reward_alerts_are_sample_bounded_and_detect_target_and_drop() {
    let rewards = [100.0, 110.0, 120.0, 115.0, 40.0, 35.0, 30.0];
    let alerts = reward_alerts(&rewards, Some(110.0), 3, 0.5);

    assert!(alerts
        .iter()
        .any(|alert| alert.kind == RewardAlertKind::TargetReached));
    assert!(alerts
        .iter()
        .any(|alert| alert.kind == RewardAlertKind::Divergence));
    assert!(reward_alerts(&rewards[..2], Some(110.0), 3, 0.5).is_empty());
}
