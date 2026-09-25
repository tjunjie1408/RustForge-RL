use rustforge_tui::action::Action;
use rustforge_tui::app::{AppMode, AppState, ChartRange, Dialog, Palette, View};
use rustforge_tui::history::BoundedHistory;
use rustforge_tui::metrics::parse_line;
use rustforge_tui::source::csv::{
    CsvDiagnostic, CsvDiagnosticKind, CsvSourcePoll, MonitorSourceState,
};

fn row(episode: u64) -> rustforge_tui::metrics::MetricRow {
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

#[test]
fn chart_range_and_dialog_navigation_are_reducer_owned() {
    let mut app = AppState::new(AppMode::Monitor, 8, 8);
    assert_eq!(app.chart_range(), ChartRange::Last100);
    app.apply(Action::NextRange);
    assert_eq!(app.chart_range(), ChartRange::Last500);
    app.apply(Action::PreviousRange);
    assert_eq!(app.chart_range(), ChartRange::Last100);

    app.apply(Action::ToggleAlertSettings);
    assert_eq!(app.dialog(), Some(Dialog::AlertSettings));
    app.apply(Action::DismissDialog);
    assert_eq!(app.dialog(), None);
}

#[test]
fn alert_target_can_be_edited_for_the_current_session() {
    let mut app = AppState::new(AppMode::Monitor, 8, 8);
    app.apply(Action::ToggleAlertSettings);
    for character in ['1', '9', '5', '.', '5'] {
        app.apply(Action::AlertTargetChar(character));
    }
    app.apply(Action::ApplyAlertTarget);
    assert_eq!(app.target_reward(), Some(195.5));
    assert_eq!(app.dialog(), None);

    app.apply(Action::ToggleAlertSettings);
    app.apply(Action::AlertTargetBackspace);
    assert_eq!(app.alert_target_input(), "195.");
}

fn poll(rows: std::ops::Range<u64>, messages: &[&str]) -> CsvSourcePoll {
    CsvSourcePoll {
        rows: rows.map(row).collect(),
        state: MonitorSourceState::Following,
        reset: false,
        diagnostics: messages
            .iter()
            .map(|message| CsvDiagnostic {
                kind: CsvDiagnosticKind::Lifecycle,
                line: None,
                message: (*message).into(),
            })
            .collect(),
    }
}

fn last_charted_episode(app: &AppState) -> Option<u64> {
    app.chart_rows().last().map(|row| row.episode)
}

fn newest_activity(app: &AppState) -> Option<String> {
    app.activity_newest_first()
        .next()
        .map(|item| item.message.clone())
}

#[test]
fn toggle_follow_freezes_chart_window_and_activity_until_resumed() {
    let mut app = AppState::new(AppMode::Monitor, 64, 16);
    app.apply_csv_poll(poll(0..5, &["first"]));

    app.apply(Action::ToggleFollow);
    assert!(!app.follow_live());
    app.apply_csv_poll(poll(5..10, &["second"]));
    assert_eq!(app.latest_episode().unwrap().episode, 9);
    assert_eq!(last_charted_episode(&app), Some(4));
    assert_eq!(newest_activity(&app).as_deref(), Some("first"));

    app.apply(Action::ToggleFollow);
    assert!(app.follow_live());
    assert_eq!(last_charted_episode(&app), Some(9));
    assert_eq!(newest_activity(&app).as_deref(), Some("second"));
}

#[test]
fn scrolling_freezes_and_jump_to_latest_resumes_following() {
    let mut app = AppState::new(AppMode::Monitor, 64, 16);
    app.apply_csv_poll(poll(0..3, &[]));
    app.apply(Action::ScrollUp(1));
    app.apply_csv_poll(poll(3..6, &[]));
    assert_eq!(last_charted_episode(&app), Some(2));

    app.apply(Action::JumpToLatest);
    assert_eq!(last_charted_episode(&app), Some(5));
}

#[test]
fn frozen_chart_window_respects_range_and_eviction() {
    let mut app = AppState::new(AppMode::Monitor, 8, 4);
    app.apply_csv_poll(poll(0..6, &[]));
    app.apply(Action::ToggleFollow);
    app.apply_csv_poll(poll(6..10, &[]));
    // Capacity 8 evicted episodes 0 and 1; the frozen window still ends at episode 5.
    let charted: Vec<u64> = app.chart_rows().iter().map(|row| row.episode).collect();
    assert_eq!(charted, vec![2, 3, 4, 5]);

    app.apply_csv_poll(poll(10..20, &[]));
    // The frozen window has been fully evicted.
    assert!(app.chart_rows().is_empty());
}

#[test]
fn source_reset_clears_freeze() {
    let mut app = AppState::new(AppMode::Monitor, 64, 16);
    app.apply_csv_poll(poll(0..5, &[]));
    app.apply(Action::ToggleFollow);
    app.apply_csv_poll(CsvSourcePoll {
        reset: true,
        ..poll(0..2, &[])
    });
    assert!(app.follow_live());
    assert_eq!(last_charted_episode(&app), Some(1));
}
