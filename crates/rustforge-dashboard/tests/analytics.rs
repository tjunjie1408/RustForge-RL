use std::time::Duration;

use rustforge_dashboard::analytics::{
    estimate_eta, is_stalled, observed_rates, reward_alerts, RewardAlertKind,
};
use rustforge_dashboard::app::{AppMode, AppState};
use rustforge_dashboard::metrics::parse_line;
use rustforge_dashboard::monitor::MonitorTracker;
use rustforge_dashboard::source::csv::{CsvSourcePoll, MonitorSourceState};
use std::time::Instant;

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

#[test]
fn monitor_tracker_uses_only_progress_observed_after_attach() {
    let start = Instant::now();
    let mut app = AppState::new(AppMode::Monitor, 32, 8);
    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![parse_line("10,100,0.5,0.5,1000").unwrap()],
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: vec![],
    });
    app.set_total_episodes(Some(20));
    let mut tracker = MonitorTracker::new(&app, start);

    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![parse_line("11,110,0.4,0.4,1200").unwrap()],
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: vec![],
    });
    let insight = tracker.update(&app, start + Duration::from_secs(10));
    assert_eq!(insight.steps_per_second, Some(20.0));
    assert_eq!(insight.episodes_per_minute, Some(6.0));
    assert_eq!(insight.progress_fraction, Some(0.6));
    assert_eq!(insight.eta, Some(Duration::from_secs(80)));
    assert!(!insight.stalled);
}
