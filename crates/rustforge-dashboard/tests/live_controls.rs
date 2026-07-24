use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rustforge_dashboard::live::{map_live_key, LiveInput};
use rustforge_rl::runtime::trainer::TrainerStatus;

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
