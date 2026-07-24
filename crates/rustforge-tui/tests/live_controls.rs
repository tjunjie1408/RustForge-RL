use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rustforge_rl::runtime::control::TrainerControl;
use rustforge_rl::runtime::event::bounded_event_channel;
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::{
    OutcomeSlot, TrainerCapabilities, TrainerMetadata, TrainerStatus, TrainingOutcome,
};
use rustforge_tui::live::{map_live_key, run_live, LiveInput, LiveOptions, LiveSession};

#[test]
fn live_keys_pause_resume_and_escalate_quit_without_exiting_early() {
    let p = KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE);
    assert_eq!(
        map_live_key(p, TrainerStatus::Running, false, false),
        LiveInput::Pause
    );
    assert_eq!(
        map_live_key(p, TrainerStatus::Paused, false, false),
        LiveInput::Resume
    );
    let q = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
    assert_eq!(
        map_live_key(q, TrainerStatus::Running, false, false),
        LiveInput::GracefulStop
    );
    assert_eq!(
        map_live_key(q, TrainerStatus::Stopping, true, false),
        LiveInput::ForceStop
    );
    assert_eq!(
        map_live_key(q, TrainerStatus::Completed, true, true),
        LiveInput::Acknowledge
    );
}

#[tokio::test]
async fn setup_failure_requests_shutdown_and_joins_the_trainer() {
    let joined = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let joined_by_thread = joined.clone();
    let trainer = std::thread::spawn(move || {
        joined_by_thread.store(true, std::sync::atomic::Ordering::Release);
        TrainingOutcome::completed(0, 0, std::time::Duration::ZERO)
    });
    let (_publisher, events, _) = bounded_event_channel(4, std::time::Duration::from_millis(1));
    let (_progress, progress) = progress_channel();
    let result = run_live(
        LiveOptions {
            no_color: true,
            ascii: true,
            target_reward: None,
            total_episodes: 1,
            metrics_path: "unused.csv".into(),
            manifest_path: "unused.json".into(),
            seed: None,
            device: None,
            configuration: Vec::new(),
        },
        LiveSession {
            events,
            progress,
            control: TrainerControl::new(),
            metadata: TrainerMetadata {
                algorithm: "test".into(),
                environment: "test".into(),
                run_id: "test".into(),
                capabilities: TrainerCapabilities {
                    pause_resume: true,
                    graceful_stop: true,
                    force_stop: true,
                    checkpoint: false,
                },
                metrics: Vec::new(),
            },
            outcome: OutcomeSlot::new(),
            trainer,
        },
    )
    .await;
    assert!(result.is_err());
    assert!(joined.load(std::sync::atomic::Ordering::Acquire));
}
