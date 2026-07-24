//! Ratatui rendering for the native training console.

mod charts;
mod details;
mod events;
mod overview;
pub mod theme;

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{AppState, Dialog, View};
use crate::source::csv::MonitorSourceState;
use crate::terminal::{MIN_TERMINAL_HEIGHT, MIN_TERMINAL_WIDTH};
use theme::Theme;

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
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(2),
        ])
        .split(area);
    render_header(frame, sections[0], app, theme);
    match app.view() {
        View::Overview => overview::render(frame, sections[1], app, theme),
        View::Charts => charts::render(frame, sections[1], app, theme),
        View::RunDetails => details::render(frame, sections[1], app, theme),
        View::Events => events::render(frame, sections[1], app, theme),
    }
    render_footer(frame, sections[2], app, theme);
    match app.dialog() {
        Some(Dialog::Help) => render_help(frame, area, app, theme),
        Some(Dialog::AlertSettings) => render_alert_settings(frame, area, app, theme),
        None => {}
    }
}

fn render_header(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let view = match app.view() {
        View::Overview => "OVERVIEW",
        View::Charts => "CHARTS",
        View::RunDetails => "RUN DETAILS",
        View::Events => "EVENTS / ACTIVITY",
    };
    let status = source_state_label(app.source_state());
    let title = Line::from(vec![
        Span::styled(" RustForge ", theme.title()),
        Span::styled(format!("{view}  "), Style::default().fg(theme.text)),
        Span::styled(status, source_state_style(app.source_state(), theme)),
    ]);
    frame.render_widget(
        Paragraph::new(title).block(theme.block(" Native training console ", app.ascii())),
        area,
    );
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let range = format!("{:?}", app.chart_range());
    let controls = if app.live_controls_visible() {
        "Tab views  arrows navigate  f follow  p pause/resume  q stop  ? help"
    } else {
        "Tab views  arrows navigate  f follow  t palette  q quit  ? help"
    };
    frame.render_widget(
        Paragraph::new(format!("{controls}  range:{range}"))
            .style(Style::default().fg(theme.muted))
            .alignment(Alignment::Center),
        area,
    );
}

fn render_alert_settings(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let popup = centered_rect(64, 40, area);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(format!(
            "Alert settings\n\nTarget reward: {}_\n{}\nChanges affect only this monitor session.\n\nEnter applies; Backspace edits; Esc cancels.",
            app.alert_target_input(),
            app.alert_target_error().unwrap_or("")
        ))
        .wrap(Wrap { trim: true })
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
    let popup = centered_rect(70, 70, area);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(
            "Tab / Shift-Tab  change view\n\
             Left / Right     change chart range\n\
             Up / Down        scroll or move focus\n\
             Home / End       first / latest\n\
             f                follow / freeze\n\
             t                cycle palette\n\
             ?                close help\n\
             q                quit or graceful stop",
        )
        .style(Style::default().fg(theme.text))
        .block(theme.block(" Keyboard help ", app.ascii())),
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

fn source_state_style(state: MonitorSourceState, theme: Theme) -> Style {
    let color = match state {
        MonitorSourceState::Following | MonitorSourceState::Completed => theme.success,
        MonitorSourceState::Waiting | MonitorSourceState::Idle => theme.warning,
        MonitorSourceState::SourceError => theme.error,
    };
    Style::default().fg(color)
}
