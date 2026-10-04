//! Shared keyboard navigation for live training and file monitoring.
use crate::action::Action;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
pub(crate) fn plain_key(key: KeyEvent) -> bool {
    !key.modifiers.intersects(
        KeyModifiers::CONTROL
            | KeyModifiers::ALT
            | KeyModifiers::SUPER
            | KeyModifiers::HYPER
            | KeyModifiers::META,
    )
}
pub(crate) fn navigation(key: KeyEvent) -> Option<Action> {
    if !plain_key(key) {
        return None;
    }
    Some(match key.code {
        KeyCode::Tab => Action::NextView,
        KeyCode::BackTab => Action::PreviousView,
        KeyCode::Left => Action::PreviousRange,
        KeyCode::Right => Action::NextRange,
        KeyCode::Up => Action::ScrollUp(1),
        KeyCode::Down => Action::ScrollDown(1),
        KeyCode::PageUp => Action::ScrollUp(10),
        KeyCode::PageDown => Action::ScrollDown(10),
        KeyCode::Home => Action::JumpToFirst,
        KeyCode::End => Action::JumpToLatest,
        KeyCode::Char('f') => Action::ToggleFollow,
        KeyCode::Char('t') => Action::CyclePalette,
        KeyCode::Char('g') => Action::ToggleAlertSettings,
        KeyCode::Char('?') | KeyCode::F(1) => Action::ToggleHelp,
        KeyCode::Esc => Action::DismissDialog,
        KeyCode::Backspace => Action::AlertTargetBackspace,
        KeyCode::Enter => Action::ApplyAlertTarget,
        KeyCode::Char(character) if character.is_ascii_digit() || ".eE+-".contains(character) => {
            Action::AlertTargetChar(character)
        }
        _ => return None,
    })
}
