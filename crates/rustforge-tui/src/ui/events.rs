use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::analytics::RewardAlertKind;
use crate::app::{AppMode, AppState};
use crate::source::csv::{CsvDiagnostic, CsvDiagnosticKind};
use crate::ui::theme::Theme;

const TAG_WIDTH: usize = 12;

pub fn render(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let visible = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line<'_>> = app
        .activity()
        .iter()
        .rev()
        .skip(app.scroll_offset())
        .take(visible)
        .map(|item| activity_line(item, theme))
        .collect();
    let lines = if lines.is_empty() {
        vec![empty_line(app, theme)]
    } else {
        lines
    };
    let subtitle = if app.mode() == AppMode::Live {
        "reliable training events and runtime health"
    } else {
        "source activity, not synthesized training events"
    };
    let title = Line::from(vec![
        Span::styled(" Events ", theme.title()),
        Span::styled(
            format!("{} {subtitle} ", if theme.ascii { "-" } else { "·" }),
            theme.muted_style(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(theme.block(title, theme.ascii)),
        area,
    );
}

/// Alerts, stall warnings, then the newest activity, capped at `limit` lines.
pub(super) fn recent_lines(app: &AppState, theme: Theme, limit: usize) -> Vec<Line<'_>> {
    let insights = app.monitor_insights();
    let mut lines = Vec::with_capacity(limit);
    for alert in &insights.alerts {
        let (message, color) = match alert.kind {
            RewardAlertKind::TargetReached => (
                format!("target reward reached: {:.2}", alert.value),
                theme.success,
            ),
            RewardAlertKind::Divergence => (
                format!("reward divergence: recent avg {:.2}", alert.value),
                theme.warning,
            ),
        };
        lines.push(tagged(" ALERT ", theme.badge(color), message, theme));
    }
    if insights.stalled {
        lines.push(tagged(
            " STALL ",
            theme.badge(theme.warning),
            "no new progress for at least 30 seconds".into(),
            theme,
        ));
    }
    lines.extend(
        app.activity()
            .iter()
            .rev()
            .map(|item| activity_line(item, theme)),
    );
    lines.truncate(limit);
    if lines.is_empty() {
        lines.push(empty_line(app, theme));
    }
    lines
}

fn tagged(tag: &str, style: Style, message: String, theme: Theme) -> Line<'static> {
    Line::from(vec![
        Span::raw(" "),
        Span::styled(tag.to_owned(), style),
        Span::raw(" ".repeat(TAG_WIDTH.saturating_sub(tag.len()) + 1)),
        Span::styled(message, theme.text_style()),
    ])
}

fn activity_line(item: &CsvDiagnostic, theme: Theme) -> Line<'_> {
    let (tag, color) = kind_tag(item.kind, theme);
    let mut spans = vec![
        Span::raw(" "),
        Span::styled(format!("{tag:<TAG_WIDTH$}"), Style::default().fg(color)),
        Span::raw(" "),
    ];
    if let Some(line) = item.line {
        spans.push(Span::styled(format!("#{line} "), theme.muted_style()));
    }
    spans.push(Span::styled(item.message.as_str(), theme.text_style()));
    Line::from(spans)
}

fn empty_line(app: &AppState, theme: Theme) -> Line<'static> {
    Line::styled(
        if app.mode() == AppMode::Live {
            " No live training event has been observed yet"
        } else {
            " No persisted-source activity has been observed yet"
        },
        theme.muted_style(),
    )
}

fn kind_tag(kind: CsvDiagnosticKind, theme: Theme) -> (&'static str, Color) {
    match kind {
        CsvDiagnosticKind::Lifecycle => ("Lifecycle", theme.accent),
        CsvDiagnosticKind::Control => ("Control", theme.success),
        CsvDiagnosticKind::Persistence => ("Persistence", theme.warning),
        CsvDiagnosticKind::Attached => ("Attached", theme.success),
        CsvDiagnosticKind::Reappeared => ("Reappeared", theme.success),
        CsvDiagnosticKind::Disappeared => ("Disappeared", theme.warning),
        CsvDiagnosticKind::Truncated => ("Truncated", theme.warning),
        CsvDiagnosticKind::Replaced => ("Replaced", theme.warning),
        CsvDiagnosticKind::HeaderMismatch => ("Bad header", theme.error),
        CsvDiagnosticKind::MalformedRow => ("Bad row", theme.error),
        CsvDiagnosticKind::InvalidUtf8 => ("Bad UTF-8", theme.error),
        CsvDiagnosticKind::IoError => ("I/O error", theme.error),
    }
}
