use std::time::Duration;

use rustforge_tui::event_loop::EventCadence;
use rustforge_tui::terminal::{
    validate_terminal_environment, validate_terminal_size, TerminalPreflightError,
    MIN_TERMINAL_HEIGHT, MIN_TERMINAL_WIDTH,
};

#[test]
fn terminal_preflight_rejects_non_tty_before_raw_mode() {
    assert_eq!(
        validate_terminal_environment(false, true),
        Err(TerminalPreflightError::InputNotTerminal)
    );
    assert_eq!(
        validate_terminal_environment(true, false),
        Err(TerminalPreflightError::OutputNotTerminal)
    );
    assert_eq!(validate_terminal_environment(true, true), Ok(()));
}

#[test]
fn live_terminal_size_is_validated_before_raw_mode() {
    assert!(validate_terminal_size(MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT).is_ok());
    assert!(validate_terminal_size(MIN_TERMINAL_WIDTH - 1, MIN_TERMINAL_HEIGHT).is_err());
    assert!(validate_terminal_size(MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT - 1).is_err());
}

#[test]
fn minimum_size_matches_the_stable_resize_help_contract() {
    assert_eq!(MIN_TERMINAL_WIDTH, 60);
    assert_eq!(MIN_TERMINAL_HEIGHT, 18);
}

#[test]
fn default_event_cadences_are_independent_and_bounded() {
    let cadence = EventCadence::default();
    assert_eq!(cadence.progress_sample, Duration::from_millis(250));
    assert!(cadence.render >= Duration::from_millis(50));
    assert!(cadence.render <= Duration::from_millis(100));
    assert!(cadence.source_poll >= Duration::from_millis(100));
    assert!(cadence.outcome_poll >= Duration::from_millis(500));
}
