//! Reducer-owned application state shared by CSV and live sources.

use crate::action::Action;
use crate::history::BoundedHistory;
use crate::metrics::MetricRow;
use crate::source::csv::{CsvDiagnostic, CsvSourcePoll, MonitorSourceState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppMode {
    Monitor,
    Live,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Overview,
    Charts,
    RunDetails,
    Events,
}

impl View {
    fn next(self) -> Self {
        match self {
            Self::Overview => Self::Charts,
            Self::Charts => Self::RunDetails,
            Self::RunDetails => Self::Events,
            Self::Events => Self::Overview,
        }
    }

    fn previous(self) -> Self {
        match self {
            Self::Overview => Self::Events,
            Self::Charts => Self::Overview,
            Self::RunDetails => Self::Charts,
            Self::Events => Self::RunDetails,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Palette {
    Default,
    HighContrast,
    Monochrome,
}

impl Palette {
    fn next(self) -> Self {
        match self {
            Self::Default => Self::HighContrast,
            Self::HighContrast => Self::Monochrome,
            Self::Monochrome => Self::Default,
        }
    }
}

pub struct AppState {
    mode: AppMode,
    view: View,
    source_state: MonitorSourceState,
    episodes: BoundedHistory<MetricRow>,
    activity: BoundedHistory<CsvDiagnostic>,
    follow_live: bool,
    scroll_offset: usize,
    palette: Palette,
    help_visible: bool,
    live_controls_available: bool,
}

impl AppState {
    pub fn new(mode: AppMode, episode_capacity: usize, activity_capacity: usize) -> Self {
        Self {
            mode,
            view: View::Overview,
            source_state: MonitorSourceState::Waiting,
            episodes: BoundedHistory::new(episode_capacity),
            activity: BoundedHistory::new(activity_capacity),
            follow_live: true,
            scroll_offset: 0,
            palette: Palette::Default,
            help_visible: false,
            live_controls_available: false,
        }
    }

    pub fn apply_csv_poll(&mut self, poll: CsvSourcePoll) {
        if poll.reset {
            self.episodes.clear();
            self.scroll_offset = 0;
            self.follow_live = true;
        }
        for row in poll.rows {
            self.episodes.push(row);
        }
        for diagnostic in poll.diagnostics {
            self.activity.push(diagnostic);
        }
        self.source_state = poll.state;
    }

    pub fn apply(&mut self, action: Action) {
        match action {
            Action::NextView => self.view = self.view.next(),
            Action::PreviousView => self.view = self.view.previous(),
            Action::ScrollUp(amount) => {
                self.scroll_offset = self.scroll_offset.saturating_add(amount);
                self.follow_live = false;
            }
            Action::ScrollDown(amount) => {
                self.scroll_offset = self.scroll_offset.saturating_sub(amount);
            }
            Action::JumpToFirst => {
                self.scroll_offset = self.episodes.len().saturating_sub(1);
                self.follow_live = false;
            }
            Action::JumpToLatest => {
                self.scroll_offset = 0;
                self.follow_live = true;
            }
            Action::ToggleFollow => self.follow_live = !self.follow_live,
            Action::CyclePalette => self.palette = self.palette.next(),
            Action::ToggleHelp => self.help_visible = !self.help_visible,
        }
    }

    pub fn mode(&self) -> AppMode {
        self.mode
    }

    pub fn view(&self) -> View {
        self.view
    }

    pub fn source_state(&self) -> MonitorSourceState {
        self.source_state
    }

    pub fn episodes(&self) -> &BoundedHistory<MetricRow> {
        &self.episodes
    }

    pub fn latest_episode(&self) -> Option<&MetricRow> {
        self.episodes.back()
    }

    pub fn activity(&self) -> &BoundedHistory<CsvDiagnostic> {
        &self.activity
    }

    pub fn follow_live(&self) -> bool {
        self.follow_live
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }

    pub fn palette(&self) -> Palette {
        self.palette
    }

    pub fn help_visible(&self) -> bool {
        self.help_visible
    }

    pub fn live_controls_visible(&self) -> bool {
        self.mode == AppMode::Live && self.live_controls_available
    }

    pub fn set_live_controls_available(&mut self, available: bool) {
        self.live_controls_available = available;
    }
}
