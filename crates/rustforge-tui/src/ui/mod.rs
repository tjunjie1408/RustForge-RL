//! Ratatui rendering for the native training console.

mod charts;
mod details;
mod events;
mod format;
mod overview;
pub mod theme;

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};
use ratatui::Frame;
use rustforge_rl::runtime::trainer::TrainerStatus;

use crate::app::{AppMode, AppState, Dialog, View};
use crate::source::csv::MonitorSourceState;
use crate::terminal::{MIN_TERMINAL_HEIGHT, MIN_TERMINAL_WIDTH};
use theme::Theme;

const VIEWS: [View; 4] = [View::Overview, View::Charts, View::RunDetails, View::Events];

pub fn render(frame: &mut Frame<'_>, app: &AppState) {
    let area = frame.area();
    let theme = Theme::for_app(app);
    if area.width < MIN_TERMINAL_WIDTH || area.height < MIN_TERMINAL_HEIGHT {
        render_resize_help(frame, area, app, theme);
        return;
    }

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, sections[0], app, theme);
    render_progress(frame, sections[1], app, theme);
    match app.view() {
        View::Overview => overview::render(frame, sections[2], app, theme),
        View::Charts => charts::render(frame, sections[2], app, theme),
        View::RunDetails => details::render(frame, sections[2], app, theme),
        View::Events => events::render(frame, sections[2], app, theme),
    }
    render_footer(frame, sections[3], app, theme);
    match app.dialog() {
        Some(Dialog::Help) => render_help(frame, area, app, theme),
        Some(Dialog::AlertSettings) => render_alert_settings(frame, area, app, theme),
        None => {}
    }
}

fn view_name(view: View) -> &'static str {
    match view {
        View::Overview => "OVERVIEW",
        View::Charts => "CHARTS",
        View::RunDetails => "RUN DETAILS",
        View::Events => "EVENTS",
    }
}

fn render_header(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let (status, status_color) = status_badge(app, theme);
    let mut left = vec![
        Span::styled(" RustForge ", theme.badge(theme.accent)),
        Span::raw(" "),
    ];
    if let Some(identity) = run_identity(app, theme) {
        left.push(Span::styled(identity, theme.text_style()));
        left.push(Span::raw("  "));
    }
    left.push(Span::styled(
        format!(" {status} "),
        theme.badge(status_color),
    ));
    let left = Line::from(left);
    let left_width = left.width() as u16;

    let full_tabs = tabs_line(app, theme, false);
    let tabs = if full_tabs.width() as u16 + left_width < area.width {
        full_tabs
    } else {
        tabs_line(app, theme, true)
    };
    let tabs_width = (tabs.width() as u16).min(area.width.saturating_sub(left_width));
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(tabs_width)])
        .split(area);
    frame.render_widget(Paragraph::new(left), columns[0]);
    frame.render_widget(Paragraph::new(tabs).alignment(Alignment::Right), columns[1]);
}

fn tabs_line(app: &AppState, theme: Theme, selected_only: bool) -> Line<'static> {
    let mut spans = Vec::new();
    for view in VIEWS {
        let selected = view == app.view();
        if selected_only && !selected {
            continue;
        }
        if !spans.is_empty() {
            spans.push(Span::raw(" "));
        }
        let style = if selected {
            theme.title().add_modifier(Modifier::UNDERLINED)
        } else {
            theme.muted_style()
        };
        spans.push(Span::styled(format!(" {} ", view_name(view)), style));
    }
    Line::from(spans)
}

fn run_identity(app: &AppState, theme: Theme) -> Option<String> {
    let metadata = app.run_metadata();
    let separator = if theme.ascii { " / " } else { " · " };
    match (&metadata.algorithm, &metadata.environment) {
        (Some(algorithm), Some(environment)) => {
            Some(format!("{algorithm}{separator}{environment}"))
        }
        (Some(name), None) | (None, Some(name)) => Some(name.clone()),
        (None, None) => metadata
            .metrics_path
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|name| format!("monitor{separator}{}", name.to_string_lossy())),
    }
}

fn status_badge(app: &AppState, theme: Theme) -> (&'static str, Color) {
    match app.trainer_status() {
        Some(TrainerStatus::Running) if app.stop_requested() => ("STOPPING", theme.warning),
        Some(TrainerStatus::Running) => ("RUNNING", theme.success),
        Some(TrainerStatus::Paused) => ("PAUSED", theme.warning),
        Some(TrainerStatus::Stopping) => ("STOPPING", theme.warning),
        Some(TrainerStatus::Stopped) => ("STOPPED", theme.warning),
        Some(TrainerStatus::Completed) => ("COMPLETED", theme.success),
        Some(TrainerStatus::Failed) => ("FAILED", theme.error),
        None => (
            source_state_label(app.source_state()),
            source_state_color(app.source_state(), theme),
        ),
    }
}

fn render_progress(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let insights = app.monitor_insights();
    let separator = || {
        Span::styled(
            if theme.ascii { "  |  " } else { "  │  " },
            Style::default().fg(theme.border),
        )
    };
    // Groups are added in priority order and dropped whole when the line is full.
    let mut groups: Vec<Vec<Span<'static>>> = Vec::new();
    match (insights.progress_fraction, app.total_episodes()) {
        (Some(fraction), Some(total)) => {
            let width = usize::from(area.width / 4).clamp(10, 32);
            let (filled, empty) = format::progress_bar(fraction, width, theme.ascii);
            groups.push(vec![
                Span::raw(" "),
                Span::styled(filled, Style::default().fg(theme.accent)),
                Span::styled(empty, Style::default().fg(theme.border)),
                Span::styled(format!(" {:>5.1}%", fraction * 100.0), theme.title()),
            ]);
            let completed = (fraction * total as f64).round() as u64;
            groups.push(vec![
                Span::styled("  ep ", theme.muted_style()),
                Span::styled(
                    format!("{}/{}", format::grouped(completed), format::grouped(total)),
                    theme.text_style(),
                ),
            ]);
            if !app.finished() {
                let eta = insights
                    .eta
                    .map(format::duration)
                    .unwrap_or_else(|| theme.dash().into());
                groups.push(vec![
                    Span::styled("  ETA ", theme.muted_style()),
                    Span::styled(eta, theme.text_style()),
                ]);
            }
        }
        _ => {
            let (episode, step) = app
                .latest_episode()
                .map(|row| {
                    (
                        format::grouped(row.episode),
                        format::grouped(row.global_step),
                    )
                })
                .unwrap_or_else(|| (theme.dash().into(), theme.dash().into()));
            groups.push(vec![
                Span::styled(" ep ", theme.muted_style()),
                Span::styled(episode, theme.text_style()),
            ]);
            groups.push(vec![
                Span::styled("  step ", theme.muted_style()),
                Span::styled(step, theme.text_style()),
            ]);
        }
    }
    if insights.stalled {
        groups.push(vec![
            Span::raw("  "),
            Span::styled(" STALLED ", theme.badge(theme.warning)),
        ]);
    }
    let rate = |value: Option<f64>| {
        value
            .map(format::compact)
            .unwrap_or_else(|| theme.dash().into())
    };
    groups.push(vec![
        separator(),
        Span::styled(rate(insights.steps_per_second), theme.text_style()),
        Span::styled(" steps/s", theme.muted_style()),
    ]);
    groups.push(vec![
        separator(),
        Span::styled(rate(insights.episodes_per_minute), theme.text_style()),
        Span::styled(" ep/min", theme.muted_style()),
    ]);
    groups.push(vec![
        separator(),
        Span::styled(format::duration(insights.elapsed), theme.text_style()),
        Span::styled(" elapsed", theme.muted_style()),
    ]);
    frame.render_widget(
        Paragraph::new(fit_groups(groups, usize::from(area.width))),
        area,
    );
}

/// Concatenate span groups in order, stopping before the first group that does not fit.
fn fit_groups(groups: Vec<Vec<Span<'static>>>, width: usize) -> Line<'static> {
    let mut spans = Vec::new();
    let mut used = 0;
    for group in groups {
        let group_width: usize = group.iter().map(Span::width).sum();
        if used + group_width > width {
            break;
        }
        used += group_width;
        spans.extend(group);
    }
    Line::from(spans)
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let mut hints: Vec<(&str, &str)> = Vec::new();
    match app.mode() {
        AppMode::Live if app.finished() => hints.push(("Enter", "exit")),
        AppMode::Live if app.stop_requested() => hints.push(("q", "force stop")),
        AppMode::Live => {
            if app.live_controls_visible() {
                if app.trainer_status() == Some(TrainerStatus::Paused) {
                    hints.push(("p", "resume"));
                } else {
                    hints.push(("p", "pause"));
                }
            }
            hints.push(("q", "stop"));
        }
        AppMode::Monitor => hints.push(("q", "quit")),
    }
    let arrows = if theme.ascii {
        ("Left/Right", "Up/Down")
    } else {
        ("←/→", "↑/↓")
    };
    hints.extend([
        ("?", "help"),
        ("Tab", "view"),
        (arrows.0, "range"),
        (arrows.1, "scroll"),
        ("g", "alerts"),
        ("t", "theme"),
    ]);

    let range = match app.chart_range().limit() {
        Some(limit) => format!("range: last {limit} "),
        None => "range: all ".into(),
    };
    let groups = hints
        .into_iter()
        .enumerate()
        .map(|(index, (key, description))| {
            vec![
                Span::raw(if index == 0 { " " } else { "  " }),
                Span::styled(key, theme.title()),
                Span::styled(format!(" {description}"), theme.muted_style()),
            ]
        })
        .collect();
    let hints_width = usize::from(area.width).saturating_sub(range.len() + 1);
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(range.len() as u16)])
        .split(area);
    frame.render_widget(Paragraph::new(fit_groups(groups, hints_width)), columns[0]);
    frame.render_widget(
        Paragraph::new(Span::styled(range, theme.muted_style())).alignment(Alignment::Right),
        columns[1],
    );
}

fn render_alert_settings(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let popup = centered_rect(64, 40, area);
    frame.render_widget(Clear, popup);
    let mut lines = vec![
        Line::default(),
        Line::from(vec![
            Span::styled(" Target reward  ", theme.muted_style()),
            Span::styled(format!("{}_", app.alert_target_input()), theme.title()),
        ]),
    ];
    if let Some(error) = app.alert_target_error() {
        lines.push(Line::styled(
            format!(" {error}"),
            Style::default().fg(theme.error),
        ));
    }
    lines.extend([
        Line::default(),
        Line::styled(" Changes affect only this session.", theme.muted_style()),
        Line::styled(
            " Enter applies; Backspace edits; Esc cancels.",
            theme.muted_style(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(theme.block(" Alerts ", app.ascii())),
        popup,
    );
}

fn render_resize_help(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let message = format!(
        "Terminal too small\nCurrent: {}x{}\nRequired: {}x{}",
        area.width, area.height, MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT
    );
    frame.render_widget(
        Paragraph::new(message)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(theme.block(" Resize terminal ", app.ascii())),
        area,
    );
}

fn render_help(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let mut keys: Vec<(&str, &str)> = vec![
        ("Tab / Shift-Tab", "change view"),
        ("Left / Right", "change chart range"),
        ("Up / Down", "scroll one line"),
        ("PgUp / PgDn", "scroll ten lines"),
        ("Home / End", "first / latest"),
        ("f", "follow / freeze"),
        ("g", "alert settings"),
        ("t", "cycle palette"),
        ("? / Esc", "close dialog"),
    ];
    if app.mode() == AppMode::Live {
        if app.live_controls_visible() {
            keys.push(("p", "pause / resume training"));
        }
        keys.push(("q", "graceful stop (finish episode)"));
        keys.push(("q again", "force stop (after current step)"));
        keys.push(("Enter", "exit after training ends"));
    } else {
        keys.push(("q / Ctrl-C", "quit monitor"));
    }
    let lines: Vec<Line<'_>> = keys
        .into_iter()
        .map(|(key, description)| {
            Line::from(vec![
                Span::styled(format!(" {key:<17}"), theme.title()),
                Span::styled(description, theme.text_style()),
            ])
        })
        .collect();
    let height = (lines.len() as u16 + 2).min(area.height);
    let width = 56.min(area.width);
    let popup = Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).block(theme.block(" Keyboard ", app.ascii())),
        popup,
    );
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

pub(crate) fn source_state_label(state: MonitorSourceState) -> &'static str {
    match state {
        MonitorSourceState::Waiting => "WAITING",
        MonitorSourceState::Following => "FOLLOWING",
        MonitorSourceState::Idle => "IDLE",
        MonitorSourceState::Completed => "COMPLETED",
        MonitorSourceState::SourceError => "SOURCE ERROR",
    }
}

fn source_state_color(state: MonitorSourceState, theme: Theme) -> Color {
    match state {
        MonitorSourceState::Following | MonitorSourceState::Completed => theme.success,
        MonitorSourceState::Waiting | MonitorSourceState::Idle => theme.warning,
        MonitorSourceState::SourceError => theme.error,
    }
}
