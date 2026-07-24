use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rustforge_tui::action::Action;
use rustforge_tui::monitor::{map_monitor_key, MonitorInput};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

#[test]
fn keyboard_contract_maps_navigation_and_monitor_quit() {
    assert_eq!(
        map_monitor_key(key(KeyCode::Tab)),
        MonitorInput::Action(Action::NextView)
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::BackTab)),
        MonitorInput::Action(Action::PreviousView)
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::Left)),
        MonitorInput::Action(Action::PreviousRange)
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::Right)),
        MonitorInput::Action(Action::NextRange)
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::PageUp)),
        MonitorInput::Action(Action::ScrollUp(10))
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::Char('f'))),
        MonitorInput::Action(Action::ToggleFollow)
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::Char('g'))),
        MonitorInput::Action(Action::ToggleAlertSettings)
    );
    assert_eq!(map_monitor_key(key(KeyCode::Char('q'))), MonitorInput::Quit);
}

#[test]
fn monitor_never_maps_live_training_controls() {
    assert_eq!(
        map_monitor_key(key(KeyCode::Char('p'))),
        MonitorInput::Ignored
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::Char('s'))),
        MonitorInput::Ignored
    );
}

#[test]
fn ctrl_c_quits_and_release_events_are_ignored() {
    assert_eq!(
        map_monitor_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        MonitorInput::Quit
    );
    let mut released = key(KeyCode::Char('q'));
    released.kind = crossterm::event::KeyEventKind::Release;
    assert_eq!(map_monitor_key(released), MonitorInput::Ignored);
}

#[test]
fn alert_dialog_keys_map_to_session_edit_actions() {
    assert_eq!(
        map_monitor_key(key(KeyCode::Char('5'))),
        MonitorInput::Action(Action::AlertTargetChar('5'))
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::Backspace)),
        MonitorInput::Action(Action::AlertTargetBackspace)
    );
    assert_eq!(
        map_monitor_key(key(KeyCode::Enter)),
        MonitorInput::Action(Action::ApplyAlertTarget)
    );
}
