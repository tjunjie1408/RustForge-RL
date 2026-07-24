//! In-process DQN terminal runner with explicit trainer-thread ownership.

use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{anyhow, Context};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use rustforge_rl::runtime::control::TrainerControl;
use rustforge_rl::runtime::event::EventEnvelope;
use rustforge_rl::runtime::progress::ProgressReader;
use rustforge_rl::runtime::trainer::{
    OutcomeSlot, TrainerMetadata, TrainerStatus, TrainingOutcome,
};

use crate::action::Action;
use crate::analytics::reward_alerts;
use crate::app::{AppMode, AppState, RunMetadata};
use crate::event_loop::{AppEvent, EventCadence, TerminalEventLoop};
use crate::source::csv::MonitorSourceState;
use crate::source::live::LiveSource;
use crate::system::SystemSampler;
use crate::terminal::{install_terminal_panic_hook, preflight_current_terminal, TerminalGuard};
use crate::ui;

const EPISODE_HISTORY_CAPACITY: usize = 100_000;
const ACTIVITY_HISTORY_CAPACITY: usize = 2_048;

#[derive(Clone, Debug)]
pub struct LiveOptions {
    pub no_color: bool,
    pub ascii: bool,
    pub target_reward: Option<f64>,
    pub total_episodes: u64,
    pub metrics_path: PathBuf,
    pub manifest_path: PathBuf,
}

pub struct LiveSession {
    pub events: crossbeam_channel::Receiver<EventEnvelope>,
    pub progress: ProgressReader,
    pub control: TrainerControl,
    pub metadata: TrainerMetadata,
    pub outcome: OutcomeSlot,
    pub trainer: JoinHandle<TrainingOutcome>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveInput {
    Action(Action),
    Pause,
    Resume,
    GracefulStop,
    ForceStop,
    Acknowledge,
    Ignored,
}

pub fn map_live_key(
    key: KeyEvent,
    status: TrainerStatus,
    stop_already_requested: bool,
    final_state: bool,
) -> LiveInput {
    if key.kind == KeyEventKind::Release {
        return LiveInput::Ignored;
    }
    let quit = key.code == KeyCode::Char('q')
        || (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c'));
    if quit {
        return if final_state {
            LiveInput::Acknowledge
        } else if stop_already_requested {
            LiveInput::ForceStop
        } else {
            LiveInput::GracefulStop
        };
    }
    if final_state && key.code == KeyCode::Enter {
        return LiveInput::Acknowledge;
    }
    if key.code == KeyCode::Char('p') && !stop_already_requested && !final_state {
        return if status == TrainerStatus::Paused {
            LiveInput::Resume
        } else {
            LiveInput::Pause
        };
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
        _ => return LiveInput::Ignored,
    };
    LiveInput::Action(action)
}

pub async fn run_live(
    options: LiveOptions,
    session: LiveSession,
) -> anyhow::Result<TrainingOutcome> {
    preflight_current_terminal().context("live training requires an interactive terminal")?;
    install_terminal_panic_hook();

    let LiveSession {
        events,
        progress,
        control,
        metadata,
        outcome,
        trainer,
    } = session;
    let mut source = LiveSource::new(events, progress, &metadata).map_err(anyhow::Error::msg)?;
    let mut app = AppState::new(
        AppMode::Live,
        EPISODE_HISTORY_CAPACITY,
        ACTIVITY_HISTORY_CAPACITY,
    );
    app.set_live_controls_available(
        metadata.capabilities.pause_resume || metadata.capabilities.graceful_stop,
    );
    app.set_no_color(options.no_color);
    app.set_ascii(options.ascii);
    app.set_target_reward(options.target_reward);
    app.set_total_episodes(Some(options.total_episodes));
    app.set_run_metadata(RunMetadata {
        run_id: Some(metadata.run_id.clone()),
        algorithm: Some(metadata.algorithm.clone()),
        environment: Some(metadata.environment.clone()),
        metrics_path: Some(options.metrics_path),
        manifest_path: Some(options.manifest_path),
        schema_version: Some("dqn-csv-v1".into()),
        ..RunMetadata::default()
    });

    let mut system = SystemSampler::new();
    app.set_system_snapshot(system.sample());
    let mut terminal = match TerminalGuard::enter() {
        Ok(terminal) => terminal,
        Err(error) => {
            control.request_graceful_stop();
            let _ = trainer.join();
            return Err(error).context("enter terminal user interface");
        }
    };
    let mut terminal_events = TerminalEventLoop::new(EventCadence::default());
    let mut stop_requested = false;
    let mut final_state = outcome.load().is_some();
    let mut source_open = true;
    let ui_result: anyhow::Result<()> = async {
        terminal
            .terminal_mut()
            .draw(|frame| ui::render(frame, &app))?;
        loop {
            tokio::select! {
            live = source.next(), if source_open => match live {
                Some(poll) => app.apply_csv_poll(poll),
                None => {
                    source_open = false;
                    if !final_state && !stop_requested {
                        control.request_graceful_stop();
                        stop_requested = true;
                    }
                }
            },
            event = terminal_events.next() => match event? {
                AppEvent::Terminal(Event::Key(key)) => {
                    let status = source.progress_snapshot().status;
                    match map_live_key(key, status, stop_requested, final_state) {
                        LiveInput::Action(action) => app.apply(action),
                        LiveInput::Pause => { control.request_pause(); }
                        LiveInput::Resume => { control.request_resume(); }
                        LiveInput::GracefulStop => {
                            control.request_graceful_stop();
                            stop_requested = true;
                        }
                        LiveInput::ForceStop => {
                            control.request_force_stop();
                            stop_requested = true;
                        }
                        LiveInput::Acknowledge => break,
                        LiveInput::Ignored => {}
                    }
                }
                AppEvent::Terminal(_) | AppEvent::SourcePoll => {}
                AppEvent::ProgressSample => {
                    let mut insights = source.progress(Some(options.total_episodes), Instant::now());
                    let rewards: Vec<_> = app.episodes().iter().map(|row| f64::from(row.reward)).collect();
                    insights.alerts = reward_alerts(&rewards, app.target_reward(), 20, 0.25);
                    app.set_monitor_insights(insights);
                }
                AppEvent::OutcomePoll => {
                    app.set_system_snapshot(system.sample());
                    if let Some(final_outcome) = outcome.load() {
                        final_state = true;
                        app.set_source_state(match final_outcome.status {
                            TrainerStatus::Failed => MonitorSourceState::SourceError,
                            TrainerStatus::Completed | TrainerStatus::Stopped => MonitorSourceState::Completed,
                            TrainerStatus::Running | TrainerStatus::Paused | TrainerStatus::Stopping => MonitorSourceState::Following,
                        });
                    }
                }
                AppEvent::Render => {
                    terminal.terminal_mut().draw(|frame| ui::render(frame, &app))?;
                }
            }
            }
        }
        Ok(())
    }
    .await;

    if ui_result.is_err() && outcome.load().is_none() {
        control.request_graceful_stop();
    }
    let restore_result = terminal.restore();
    let training_outcome = trainer
        .join()
        .map_err(|_| anyhow!("trainer wrapper panicked before returning an outcome"));
    ui_result?;
    restore_result?;
    training_outcome
}
