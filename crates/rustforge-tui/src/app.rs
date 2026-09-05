//! Reducer-owned application state shared by CSV and live sources.

use crate::action::Action;
use crate::analytics::RewardAlert;
use crate::history::BoundedHistory;
use crate::metrics::{MetricLabels, MetricRow};
use crate::source::csv::{CsvDiagnostic, CsvSourcePoll, MonitorSourceState};
use crate::system::SystemSnapshot;
use std::path::PathBuf;
use std::time::Duration;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChartRange {
    Last100,
    Last500,
    Last2000,
    All,
}

impl ChartRange {
    fn next(self) -> Self {
        match self {
            Self::Last100 => Self::Last500,
            Self::Last500 => Self::Last2000,
            Self::Last2000 => Self::All,
            Self::All => Self::Last100,
        }
    }

    fn previous(self) -> Self {
        match self {
            Self::Last100 => Self::All,
            Self::Last500 => Self::Last100,
            Self::Last2000 => Self::Last500,
            Self::All => Self::Last2000,
        }
    }

    pub fn limit(self) -> Option<usize> {
        match self {
            Self::Last100 => Some(100),
            Self::Last500 => Some(500),
            Self::Last2000 => Some(2000),
            Self::All => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialog {
    Help,
    AlertSettings,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunMetadata {
    pub run_id: Option<String>,
    pub algorithm: Option<String>,
    pub environment: Option<String>,
    pub seed: Option<u64>,
    pub device: Option<String>,
    pub metrics_path: Option<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub schema_version: Option<String>,
    /// Stable, display-only run settings supplied by the launching command.
    pub configuration: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MonitorInsights {
    pub elapsed: Duration,
    pub steps_per_second: Option<f64>,
    pub episodes_per_minute: Option<f64>,
    pub progress_fraction: Option<f64>,
    pub eta: Option<Duration>,
    pub stalled: bool,
    pub alerts: Vec<RewardAlert>,
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
    dialog: Option<Dialog>,
    chart_range: ChartRange,
    metric_labels: MetricLabels,
    live_controls_available: bool,
    run_metadata: RunMetadata,
    no_color: bool,
    ascii: bool,
    system_snapshot: Option<SystemSnapshot>,
    target_reward: Option<f64>,
    alert_target_input: String,
    alert_target_error: Option<String>,
    total_episodes: Option<u64>,
    monitor_insights: MonitorInsights,
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
            dialog: None,
            chart_range: ChartRange::Last100,
            metric_labels: MetricLabels::dqn_monitor_defaults(),
            live_controls_available: false,
            run_metadata: RunMetadata::default(),
            no_color: false,
            ascii: false,
            system_snapshot: None,
            target_reward: None,
            alert_target_input: String::new(),
            alert_target_error: None,
            total_episodes: None,
            monitor_insights: MonitorInsights::default(),
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
            Action::NextRange => self.chart_range = self.chart_range.next(),
            Action::PreviousRange => self.chart_range = self.chart_range.previous(),
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
            Action::ToggleHelp => {
                self.dialog = if self.dialog == Some(Dialog::Help) {
                    None
                } else {
                    Some(Dialog::Help)
                }
            }
            Action::ToggleAlertSettings => {
                self.dialog = if self.dialog == Some(Dialog::AlertSettings) {
                    None
                } else {
                    self.alert_target_input = self
                        .target_reward
                        .map(|value| value.to_string())
                        .unwrap_or_default();
                    self.alert_target_error = None;
                    Some(Dialog::AlertSettings)
                }
            }
            Action::DismissDialog => self.dialog = None,
            Action::AlertTargetChar(character) => {
                if self.dialog == Some(Dialog::AlertSettings)
                    && (character.is_ascii_digit() || ".eE+-".contains(character))
                {
                    self.alert_target_input.push(character);
                    self.alert_target_error = None;
                }
            }
            Action::AlertTargetBackspace => {
                if self.dialog == Some(Dialog::AlertSettings) {
                    self.alert_target_input.pop();
                    self.alert_target_error = None;
                }
            }
            Action::ApplyAlertTarget => {
                if self.dialog == Some(Dialog::AlertSettings) {
                    let value = self.alert_target_input.trim();
                    if value.is_empty() {
                        self.target_reward = None;
                        self.dialog = None;
                    } else {
                        match value.parse::<f64>() {
                            Ok(value) if value.is_finite() => {
                                self.target_reward = Some(value);
                                self.dialog = None;
                            }
                            _ => {
                                self.alert_target_error =
                                    Some("Target must be empty or a finite number".into());
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn mode(&self) -> AppMode {
        self.mode
    }

    pub fn view(&self) -> View {
        self.view
    }

    pub fn set_view(&mut self, view: View) {
        self.view = view;
    }

    pub fn source_state(&self) -> MonitorSourceState {
        self.source_state
    }

    pub fn set_source_state(&mut self, state: MonitorSourceState) {
        self.source_state = state;
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
        self.dialog == Some(Dialog::Help)
    }

    pub fn dialog(&self) -> Option<Dialog> {
        self.dialog
    }

    pub fn chart_range(&self) -> ChartRange {
        self.chart_range
    }

    pub fn chart_rows(&self) -> Vec<&MetricRow> {
        let skip = self
            .chart_range
            .limit()
            .map(|limit| self.episodes.len().saturating_sub(limit))
            .unwrap_or(0);
        self.episodes.iter().skip(skip).collect()
    }

    pub fn metric_labels(&self) -> &MetricLabels {
        &self.metric_labels
    }

    pub fn set_metric_labels(&mut self, labels: MetricLabels) {
        self.metric_labels = labels;
    }

    pub fn live_controls_visible(&self) -> bool {
        self.mode == AppMode::Live && self.live_controls_available
    }

    pub fn set_live_controls_available(&mut self, available: bool) {
        self.live_controls_available = available;
    }

    pub fn run_metadata(&self) -> &RunMetadata {
        &self.run_metadata
    }

    pub fn set_run_metadata(&mut self, metadata: RunMetadata) {
        self.run_metadata = metadata;
    }

    pub fn no_color(&self) -> bool {
        self.no_color
    }

    pub fn set_no_color(&mut self, no_color: bool) {
        self.no_color = no_color;
    }

    pub fn ascii(&self) -> bool {
        self.ascii
    }

    pub fn set_ascii(&mut self, ascii: bool) {
        self.ascii = ascii;
    }

    pub fn system_snapshot(&self) -> Option<&SystemSnapshot> {
        self.system_snapshot.as_ref()
    }

    pub fn set_system_snapshot(&mut self, snapshot: SystemSnapshot) {
        self.system_snapshot = Some(snapshot);
    }

    pub fn target_reward(&self) -> Option<f64> {
        self.target_reward
    }

    pub fn set_target_reward(&mut self, target_reward: Option<f64>) {
        self.target_reward = target_reward.filter(|value| value.is_finite());
    }

    pub fn alert_target_input(&self) -> &str {
        &self.alert_target_input
    }

    pub fn alert_target_error(&self) -> Option<&str> {
        self.alert_target_error.as_deref()
    }

    pub fn total_episodes(&self) -> Option<u64> {
        self.total_episodes
    }

    pub fn set_total_episodes(&mut self, total_episodes: Option<u64>) {
        self.total_episodes = total_episodes.filter(|episodes| *episodes > 0);
    }

    pub fn monitor_insights(&self) -> &MonitorInsights {
        &self.monitor_insights
    }

    pub fn set_monitor_insights(&mut self, insights: MonitorInsights) {
        self.monitor_insights = insights;
    }
}
