use rustforge_dashboard::action::Action;
use rustforge_dashboard::app::{AppMode, AppState, Palette, View};
use rustforge_dashboard::history::BoundedHistory;
use rustforge_dashboard::metrics::parse_line;
use rustforge_dashboard::source::csv::{
    CsvDiagnostic, CsvDiagnosticKind, CsvSourcePoll, MonitorSourceState,
};

fn row(episode: u64) -> rustforge_dashboard::metrics::MetricRow {
    parse_line(&format!(
        "{episode},{},0.5,0.9,{}",
        episode as f32 * 10.0,
        episode * 20
    ))
    .unwrap()
}

#[test]
fn bounded_history_evicts_oldest_items_and_counts_evictions() {
    let mut history = BoundedHistory::new(2);
    history.push(1);
    history.push(2);
    history.push(3);

    assert_eq!(history.iter().copied().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(history.evicted(), 1);
}

#[test]
fn csv_poll_updates_source_rows_and_activity() {
    let mut app = AppState::new(AppMode::Monitor, 8, 8);
    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![row(0), row(1)],
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: vec![CsvDiagnostic {
            kind: CsvDiagnosticKind::Attached,
            line: None,
            message: "attached".into(),
        }],
    });

    assert_eq!(app.source_state(), MonitorSourceState::Following);
    assert_eq!(app.episodes().len(), 2);
    assert_eq!(app.latest_episode().unwrap().episode, 1);
    assert_eq!(app.activity().len(), 1);
    assert!(!app.live_controls_visible());
}

#[test]
fn source_reset_clears_old_rows_before_accepting_replacement() {
    let mut app = AppState::new(AppMode::Monitor, 8, 8);
    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![row(5)],
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: vec![],
    });
    app.apply_csv_poll(CsvSourcePoll {
        rows: vec![row(0)],
        state: MonitorSourceState::Following,
        reset: true,
        diagnostics: vec![],
    });

    assert_eq!(app.episodes().len(), 1);
    assert_eq!(app.latest_episode().unwrap().episode, 0);
}

#[test]
fn navigation_cycles_views_and_freezes_when_scrolling_history() {
    let mut app = AppState::new(AppMode::Monitor, 8, 8);
    assert_eq!(app.view(), View::Overview);

    app.apply(Action::NextView);
    assert_eq!(app.view(), View::Charts);
    app.apply(Action::PreviousView);
    assert_eq!(app.view(), View::Overview);

    assert!(app.follow_live());
    app.apply(Action::ScrollUp(1));
    assert!(!app.follow_live());
    assert_eq!(app.scroll_offset(), 1);
    app.apply(Action::JumpToLatest);
    assert!(app.follow_live());
    assert_eq!(app.scroll_offset(), 0);
}

#[test]
fn palette_and_help_are_state_not_widget_concerns() {
    let mut app = AppState::new(AppMode::Monitor, 8, 8);
    assert_eq!(app.palette(), Palette::Default);
    assert!(!app.help_visible());

    app.apply(Action::CyclePalette);
    app.apply(Action::ToggleHelp);
    assert_eq!(app.palette(), Palette::HighContrast);
    assert!(app.help_visible());
}
