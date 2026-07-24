//! Renderer-independent user actions.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    NextView,
    PreviousView,
    NextRange,
    PreviousRange,
    ScrollUp(usize),
    ScrollDown(usize),
    JumpToFirst,
    JumpToLatest,
    ToggleFollow,
    CyclePalette,
    ToggleHelp,
    ToggleAlertSettings,
    DismissDialog,
    AlertTargetChar(char),
    AlertTargetBackspace,
    ApplyAlertTarget,
}
