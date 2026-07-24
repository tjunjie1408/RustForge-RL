//! Renderer-independent user actions.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    NextView,
    PreviousView,
    ScrollUp(usize),
    ScrollDown(usize),
    JumpToFirst,
    JumpToLatest,
    ToggleFollow,
    CyclePalette,
    ToggleHelp,
}
