//! Read-only terminal monitor runner and keyboard contract.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::action::Action;
use crate::analytics::{estimate_eta, is_stalled, observed_rates, reward_alerts};
use crate::app::{AppMode, AppState, MonitorInsights, RunMetadata};
use crate::event_loop::{AppEvent, EventCadence, TerminalEventLoop};
use crate::source::csv::CsvSource;
use crate::system::SystemSampler;
use crate::terminal::{install_terminal_panic_hook, preflight_current_terminal, TerminalGuard};
use crate::ui;

const EPISODE_HISTORY_CAPACITY: usize = 100_000;
const ACTIVITY_HISTORY_CAPACITY: usize = 2_048;

#[derive(Clone, Debug)]
pub struct MonitorOptions {
    pub metrics_path: PathBuf,
    pub no_color: bool,
    pub ascii: bool,
    pub target_reward: Option<f64>,
    pub total_episodes: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonitorInput {
    Action(Action),
    Quit,
    Ignored,
}

pub fn map_monitor_key(key: KeyEvent) -> MonitorInput {
    if key.kind == KeyEventKind::Release {
        return MonitorInput::Ignored;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return MonitorInput::Quit;
    }
    let action = match key.code {
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
        KeyCode::Char('?') => Action::ToggleHelp,
        KeyCode::Esc => Action::DismissDialog,
        KeyCode::Backspace => Action::AlertTargetBackspace,
        KeyCode::Enter => Action::ApplyAlertTarget,
        KeyCode::Char(character) if character.is_ascii_digit() || ".eE+-".contains(character) => {
            Action::AlertTargetChar(character)
        }
        KeyCode::Char('q') => return MonitorInput::Quit,
        _ => return MonitorInput::Ignored,
    };
    MonitorInput::Action(action)
}

pub async fn run_monitor(options: MonitorOptions) -> anyhow::Result<()> {
    validate_options(&options)?;
    preflight_current_terminal().context("terminal monitor requires an interactive terminal")?;
    install_terminal_panic_hook();

    let mut source = CsvSource::new(&options.metrics_path);
    let mut app = AppState::new(
        AppMode::Monitor,
        EPISODE_HISTORY_CAPACITY,
        ACTIVITY_HISTORY_CAPACITY,
    );
    app.set_no_color(options.no_color);
    app.set_ascii(options.ascii);
    app.set_target_reward(options.target_reward);
    app.set_total_episodes(options.total_episodes);
    app.set_run_metadata(RunMetadata {
        metrics_path: Some(options.metrics_path.clone()),
        schema_version: Some("dqn-csv-v1".into()),
        ..RunMetadata::default()
    });
    app.apply_csv_poll(source.poll());
    let mut tracker = MonitorTracker::new(&app, Instant::now());
    app.set_monitor_insights(tracker.update(&app, Instant::now()));

    let mut system = SystemSampler::new();
    app.set_system_snapshot(system.sample());
    let mut terminal = TerminalGuard::enter().context("enter terminal user interface")?;
    let mut events = TerminalEventLoop::new(EventCadence::default());
    terminal
        .terminal_mut()
        .draw(|frame| ui::render(frame, &app))?;

    loop {
        match events.next().await? {
            AppEvent::Terminal(Event::Key(key)) => match map_monitor_key(key) {
                MonitorInput::Action(action) => app.apply(action),
                MonitorInput::Quit => break,
                MonitorInput::Ignored => {}
            },
            AppEvent::Terminal(_) => {}
            AppEvent::SourcePoll => {
                app.apply_csv_poll(source.poll());
                app.set_monitor_insights(tracker.update(&app, Instant::now()));
            }
            AppEvent::ProgressSample => {
                app.set_monitor_insights(tracker.update(&app, Instant::now()));
            }
            AppEvent::OutcomePoll => app.set_system_snapshot(system.sample()),
            AppEvent::Render => {
                terminal
                    .terminal_mut()
                    .draw(|frame| ui::render(frame, &app))?;
            }
        }
    }

    terminal.restore()?;
    Ok(())
}

pub struct MonitorTracker {
    started_at: Instant,
    baseline_at: Instant,
    baseline_step: Option<u64>,
    baseline_episode: Option<u64>,
    last_step: Option<u64>,
    last_progress_at: Instant,
}

impl MonitorTracker {
    pub fn new(app: &AppState, now: Instant) -> Self {
        let latest = app.latest_episode();
        Self {
            started_at: now,
            baseline_at: now,
            baseline_step: latest.map(|row| row.global_step),
            baseline_episode: latest.map(|row| row.episode),
            last_step: latest.map(|row| row.global_step),
            last_progress_at: now,
        }
    }

    pub fn update(&mut self, app: &AppState, now: Instant) -> MonitorInsights {
        let Some(latest) = app.latest_episode() else {
            return MonitorInsights {
                elapsed: now.saturating_duration_since(self.started_at),
                ..MonitorInsights::default()
            };
        };

        if self
            .baseline_step
            .is_some_and(|baseline| latest.global_step < baseline)
            || self
                .baseline_episode
                .is_some_and(|baseline| latest.episode < baseline)
        {
            self.baseline_step = Some(latest.global_step);
            self.baseline_episode = Some(latest.episode);
            self.baseline_at = now;
            self.last_step = Some(latest.global_step);
            self.last_progress_at = now;
        }
        if self.last_step != Some(latest.global_step) {
            self.last_step = Some(latest.global_step);
            self.last_progress_at = now;
        }

        let rates = match (self.baseline_step, self.baseline_episode) {
            (Some(step), Some(episode)) => observed_rates(
                step,
                latest.global_step,
                episode,
                latest.episode,
                now.saturating_duration_since(self.baseline_at),
            ),
            _ => None,
        };
        let completed_episodes = latest.episode.saturating_add(1);
        let progress_fraction = app
            .total_episodes()
            .map(|total| (completed_episodes as f64 / total as f64).clamp(0.0, 1.0));
        let episode_rate = rates.map(|rates| rates.episodes_per_minute / 60.0);
        let eta = episode_rate
            .and_then(|rate| estimate_eta(completed_episodes, app.total_episodes(), rate));
        let rewards: Vec<f64> = app
            .episodes()
            .iter()
            .map(|row| f64::from(row.reward))
            .collect();

        MonitorInsights {
            elapsed: now.saturating_duration_since(self.started_at),
            steps_per_second: rates.map(|rates| rates.steps_per_second),
            episodes_per_minute: rates.map(|rates| rates.episodes_per_minute),
            progress_fraction,
            eta,
            stalled: is_stalled(
                app.source_state() == crate::source::csv::MonitorSourceState::Idle,
                now.saturating_duration_since(self.last_progress_at),
                Duration::from_secs(30),
            ),
            alerts: reward_alerts(&rewards, app.target_reward(), 10, 0.5),
        }
    }
}

fn validate_options(options: &MonitorOptions) -> anyhow::Result<()> {
    if options
        .target_reward
        .is_some_and(|value| !value.is_finite())
    {
        anyhow::bail!("--target-reward must be finite");
    }
    if options.total_episodes == Some(0) {
        anyhow::bail!("--total-episodes must be greater than zero");
    }
    Ok(())
}
