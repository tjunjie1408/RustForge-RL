use std::path::PathBuf;

use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::Color;
use ratatui::Terminal;
use rustforge_tui::app::{AppMode, AppState, RunMetadata, View};
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
    assert!(output.contains("Reward + rolling avg"));
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
    assert!(output.contains("Episode reward + rolling avg"));
    assert!(output.contains("PPO policy loss"));
    assert!(output.contains("PPO policy entropy"));

    app.set_metric_labels(MetricLabels {
        episode_reward: "Episode reward".into(),
        primary_loss: None,
        policy_signal: None,
        throughput: "Steps per second".into(),
    });
    let without_optional = text(&rendered(&app, 110, 34));
    assert!(without_optional.contains("Episode reward + rolling avg"));
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
    assert!(events.contains("EVENTS / ACTIVITY"));
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
