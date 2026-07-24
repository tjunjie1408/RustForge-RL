use std::sync::mpsc;
use std::time::Duration;

use rustforge_rl::runtime::control::{ControlApplyResult, ControlKind, StopMode, TrainerControl};

#[test]
fn superseded_pause_requests_keep_request_id_correlation() {
    let control = TrainerControl::new();
    let pause_1 = control.request_pause();
    let resume = control.request_resume();
    let pause_2 = control.request_pause();
    let observation = control.observe(17, false);

    assert_eq!(observation.resolutions.len(), 3);
    assert_eq!(observation.resolutions[0].request_id, pause_1);
    assert_eq!(
        observation.resolutions[0].result,
        ControlApplyResult::Superseded
    );
    assert_eq!(observation.resolutions[1].request_id, resume);
    assert_eq!(
        observation.resolutions[1].result,
        ControlApplyResult::Superseded
    );
    assert_eq!(observation.resolutions[2].request_id, pause_2);
    assert_eq!(
        observation.resolutions[2].result,
        ControlApplyResult::Applied
    );
    assert!(observation.effective_paused);
}

#[test]
fn graceful_stop_overrides_pause_and_checkpoint_is_capability_gated() {
    let control = TrainerControl::new();
    control.request_pause();
    assert!(control.observe(4, false).effective_paused);

    let stop = control.request_graceful_stop();
    let checkpoint = control.request_checkpoint();
    let observation = control.observe(5, false);
    assert_eq!(observation.stop_mode, StopMode::Graceful);
    assert!(!observation.effective_paused);
    assert!(observation.resolutions.iter().any(|resolution| {
        resolution.request_id == stop && resolution.result == ControlApplyResult::Applied
    }));
    assert!(observation.resolutions.iter().any(|resolution| {
        resolution.request_id == checkpoint
            && resolution.control == ControlKind::Checkpoint
            && resolution.result == ControlApplyResult::Unsupported
    }));

    let late_pause = control.request_pause();
    let rejected = control.observe(6, false);
    assert!(rejected.resolutions.iter().any(|resolution| {
        resolution.request_id == late_pause
            && matches!(resolution.result, ControlApplyResult::Rejected(_))
    }));
}

#[test]
fn force_stop_wakes_a_paused_trainer_without_waiting_for_another_step() {
    let control = TrainerControl::new();
    control.request_pause();
    assert!(control.observe(10, false).effective_paused);

    let worker_control = control.clone();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let observation = worker_control.wait_while_paused(10, false);
        done_tx.send(observation).unwrap();
    });
    started_rx.recv().unwrap();
    control.request_force_stop();

    let observation = done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(observation.stop_mode, StopMode::Force);
    assert!(!observation.effective_paused);
}
