use std::path::PathBuf;

use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::Color;
use ratatui::Terminal;
use std::time::Duration;

use rustforge_rl::runtime::trainer::TrainerStatus;
use rustforge_tui::app::{AppMode, AppState, MonitorInsights, RunMetadata, View};
use rustforge_tui::metrics::{parse_line, MetricLabels};
use rustforge_tui::source::csv::{
    CsvDiagnostic, CsvDiagnosticKind, CsvSourcePoll, MonitorSourceState,
};
use rustforge_tui::system::SystemSnapshot;
use rustforge_tui::ui::render;

fn sample_app() -> AppState {
    let mut app = AppState::new(AppMode::Monitor, 128, 32);
    app.set_run_metadata(RunMetadata {
        run_id: Some("historical-cartpole".into()),
        algorithm: Some("DQN".into()),
        environment: Some("CartPole-v1".into()),
        seed: Some(42),
        device: Some("CPU".into()),
        metrics_path: Some(PathBuf::from("target/runs/demo/metrics.csv")),
        manifest_path: None,
        schema_version: Some("dqn-csv-v1".into()),
        configuration: vec![
            ("Episodes".into(), "100".into()),
            ("Learning rate".into(), "0.001".into()),
        ],
    });
    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![
            parse_line("0,10,0.8,1.0,10").unwrap(),
            parse_line("1,30,0.4,0.5,30").unwrap(),
            parse_line("2,20,NaN,0.25,50").unwrap(),
        ],
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: vec![CsvDiagnostic {
            kind: CsvDiagnosticKind::Attached,
            line: None,
            message: "attached to metrics file".into(),
        }],
    });
    app.set_system_snapshot(SystemSnapshot {
        os: Some("TestOS".into()),
        architecture: "x86_64".into(),
        logical_cpus: 8,
        process_cpu_percent: Some(12.5),
        process_memory_bytes: Some(128 * 1024 * 1024),
        used_memory_bytes: Some(4 * 1024 * 1024 * 1024),
        total_memory_bytes: Some(16 * 1024 * 1024 * 1024),
    });
    app
}

fn rendered(app: &AppState, width: u16, height: u16) -> Buffer {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, app)).unwrap();
    terminal.backend().buffer().clone()
}

fn text(buffer: &Buffer) -> String {
    buffer.content.iter().map(|cell| cell.symbol()).collect()
}

#[test]
fn overview_contains_web_parity_kpis_and_source_health() {
    let app = sample_app();
    let output = text(&rendered(&app, 110, 34));

    assert!(output.contains("OVERVIEW"));
    assert!(output.contains("FOLLOWING"));
    assert!(output.contains("Episode"));
    assert!(output.contains("Latest reward"));
    assert!(output.contains("Best reward"));
    assert!(output.contains("Recent avg"));
    assert!(output.contains("Global step"));
    assert!(output.contains("30.00"));
    assert!(output.contains("CPU 12.5%"));
    assert!(output.contains("RSS 128.0 MiB"));
}

#[test]
fn charts_view_contains_reward_average_loss_and_exploration() {
    let mut app = sample_app();
    app.set_view(View::Charts);
    let output = text(&rendered(&app, 110, 34));

    assert!(output.contains("CHARTS"));
    assert!(output.contains("Reward"));
    assert!(output.contains("rolling avg 100"));
    assert!(output.contains("Loss"));
    assert!(output.contains("Exploration / epsilon"));
}

#[test]
fn live_charts_use_descriptor_labels_and_skip_unassigned_optional_panels() {
    let mut app = sample_app();
    app.set_view(View::Charts);
    app.set_metric_labels(MetricLabels {
        episode_reward: "Episode reward".into(),
        primary_loss: Some("PPO policy loss".into()),
        policy_signal: Some("PPO policy entropy".into()),
        throughput: "Steps per second".into(),
    });
    let output = text(&rendered(&app, 110, 34));
    assert!(output.contains("Episode reward"));
    assert!(output.contains("PPO policy loss"));
    assert!(output.contains("PPO policy entropy"));

    app.set_metric_labels(MetricLabels {
        episode_reward: "Episode reward".into(),
        primary_loss: None,
        policy_signal: None,
        throughput: "Steps per second".into(),
    });
    let without_optional = text(&rendered(&app, 110, 34));
    assert!(without_optional.contains("Episode reward"));
    assert!(!without_optional.contains("PPO policy loss"));
    assert!(!without_optional.contains("PPO policy entropy"));
}

#[test]
fn details_and_events_views_render_only_known_facts() {
    let mut app = sample_app();
    app.set_view(View::RunDetails);
    let details = text(&rendered(&app, 100, 28));
    assert!(details.contains("RUN DETAILS"));
    assert!(details.contains("historical-cartpole"));
    assert!(details.contains("dqn-csv-v1"));
    assert!(details.contains("read-only"));
    assert!(details.contains("Learning rate"));
    assert!(details.contains("0.001"));

    app.set_view(View::Events);
    let events = text(&rendered(&app, 100, 28));
    assert!(events.contains("EVENTS"));
    assert!(events.contains("source activity"));
    assert!(events.contains("Attached"));
    assert!(events.contains("attached to metrics file"));
    assert!(!events.contains("TrainingStarted"));
}

#[test]
fn below_minimum_size_renders_stable_resize_help() {
    let app = sample_app();
    let output = text(&rendered(&app, 59, 17));
    assert!(output.contains("Terminal too small"));
    assert!(output.contains("60x18"));
}

#[test]
fn exact_minimum_size_uses_compact_layout_without_panicking() {
    let app = sample_app();
    let output = text(&rendered(&app, 60, 18));
    assert!(output.contains("OVERVIEW"));
    assert!(output.contains("Episode"));
    assert!(!output.contains("Terminal too small"));
}

#[test]
fn no_color_and_ascii_modes_do_not_depend_on_terminal_color_or_unicode() {
    let mut app = sample_app();
    app.set_no_color(true);
    app.set_ascii(true);
    app.set_view(View::RunDetails);
    let backend = rendered(&app, 90, 26);
    let output = text(&backend);

    assert!(output.contains('+'));
    assert!(output.contains('|'));
    assert!(!output.contains('┌'));
    assert!(!output.contains('—'));
    assert!(backend
        .content
        .iter()
        .all(|cell| cell.fg == Color::Reset && cell.bg == Color::Reset));
}

fn live_app() -> AppState {
    let mut app = AppState::new(AppMode::Live, 128, 32);
    app.set_live_controls_available(true);
    app.set_run_metadata(RunMetadata {
        algorithm: Some("dqn".into()),
        environment: Some("cartpole".into()),
        ..RunMetadata::default()
    });
    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![
            parse_line("0,10,0.8,1.0,10").unwrap(),
            parse_line("1,30,0.4,0.5,30").unwrap(),
        ],
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: Vec::new(),
    });
    app.set_trainer_status(Some(TrainerStatus::Running));
    app
}

#[test]
fn live_header_badge_and_footer_hints_follow_trainer_state() {
    let mut app = live_app();
    let running = text(&rendered(&app, 110, 30));
    assert!(running.contains("dqn · cartpole"));
    assert!(running.contains("RUNNING"));
    assert!(running.contains("p pause"));
    assert!(running.contains("q stop"));

    app.set_trainer_status(Some(TrainerStatus::Paused));
    let paused = text(&rendered(&app, 110, 30));
    assert!(paused.contains("PAUSED"));
    assert!(paused.contains("p resume"));
    assert!(!paused.contains("IDLE"));

    app.set_trainer_status(Some(TrainerStatus::Running));
    app.set_stop_requested(true);
    let stopping = text(&rendered(&app, 110, 30));
    assert!(stopping.contains("STOPPING"));
    assert!(stopping.contains("q force stop"));
    assert!(!stopping.contains("p pause"));

    app.set_trainer_status(Some(TrainerStatus::Stopped));
    app.set_finished(true);
    let stopped = text(&rendered(&app, 110, 30));
    assert!(stopped.contains("STOPPED"));
    assert!(!stopped.contains("COMPLETED"));
    assert!(stopped.contains("Enter exit"));

    app.set_trainer_status(Some(TrainerStatus::Failed));
    assert!(text(&rendered(&app, 110, 30)).contains("FAILED"));
}

#[test]
fn live_controls_are_not_advertised_without_the_capability() {
    let mut app = live_app();
    app.set_live_controls_available(false);
    let output = text(&rendered(&app, 110, 30));
    assert!(!output.contains("p pause"));
    assert!(output.contains("q stop"));
}

#[test]
fn progress_line_shows_bar_counts_and_eta_until_finished() {
    let mut app = live_app();
    app.set_total_episodes(Some(200));
    app.set_monitor_insights(MonitorInsights {
        elapsed: Duration::from_secs(65),
        steps_per_second: Some(1_834.0),
        episodes_per_minute: Some(12.5),
        progress_fraction: Some(0.5),
        eta: Some(Duration::from_secs(90)),
        ..MonitorInsights::default()
    });
    let running = text(&rendered(&app, 120, 30));
    assert!(running.contains("50.0%"));
    assert!(running.contains("ep 100/200"));
    assert!(running.contains("ETA 00:01:30"));
    assert!(running.contains("1834 steps/s"));
    assert!(running.contains("00:01:05 elapsed"));

    app.set_finished(true);
    assert!(!text(&rendered(&app, 120, 30)).contains("ETA 00"));
}

#[test]
fn charts_label_axes_with_episode_and_value_ticks() {
    let mut app = sample_app();
    app.set_view(View::Charts);
    let output = text(&rendered(&app, 110, 34));
    assert!(output.contains("ep 0"));
    assert!(output.contains("latest"));
    // Rewards span 10..=30, so the padded value axis reaches beyond both ends.
    assert!(output.contains("31.0"));
    assert!(output.contains("9.0"));
}

#[test]
fn overview_includes_reward_chart_and_monitor_identity() {
    let app = sample_app();
    let output = text(&rendered(&app, 110, 34));
    assert!(output.contains("rolling avg 100"));
    assert!(output.contains("DQN · CartPole-v1"));
}

#[test]
fn narrow_footer_drops_hints_instead_of_overlapping_range() {
    let app = sample_app();
    let buffer = rendered(&app, 60, 18);
    let last_row: String = (0..60).map(|x| buffer[(x, 17)].symbol()).collect();
    assert!(last_row.contains("q quit"));
    assert!(last_row.trim_end().ends_with("range: last 100"));
    assert!(last_row.contains("  range: last 100"));
}

#[test]
fn frozen_view_is_announced_in_footer_and_hides_newer_events() {
    let mut app = sample_app();
    app.apply(rustforge_tui::action::Action::ToggleFollow);
    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![parse_line("3,99,0.1,0.1,70").unwrap()],
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: vec![CsvDiagnostic {
            kind: CsvDiagnosticKind::Truncated,
            line: None,
            message: "arrived after freeze".into(),
        }],
    });
    let output = text(&rendered(&app, 110, 34));
    assert!(output.contains("FROZEN"));
    assert!(output.contains("f follow"));
    assert!(output.contains("attached to metrics file"));
    assert!(!output.contains("arrived after freeze"));

    app.apply(rustforge_tui::action::Action::ToggleFollow);
    let following = text(&rendered(&app, 110, 34));
    assert!(!following.contains("FROZEN"));
    assert!(following.contains("arrived after freeze"));
}
