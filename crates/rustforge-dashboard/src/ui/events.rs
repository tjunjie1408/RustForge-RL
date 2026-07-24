use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::app::AppState;
use crate::ui::theme::Theme;

pub fn render(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let visible = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line<'_>> = app
        .activity()
        .iter()
        .rev()
        .skip(app.scroll_offset())
        .take(visible)
        .map(|item| {
            let location = item
                .line
                .map(|line| format!(" line {line}"))
                .unwrap_or_default();
            Line::from(vec![
                Span::styled(format!("{:?}{location}: ", item.kind), theme.title()),
                Span::styled(item.message.as_str(), Style::default().fg(theme.text)),
            ])
        })
        .collect();
    let lines = if lines.is_empty() {
        vec![Line::styled(
            "No persisted-source activity has been observed",
            Style::default().fg(theme.muted),
        )]
    } else {
        lines
    };
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(theme.block(
                " Source activity (not synthesized training events) ",
                app.ascii(),
            )),
        area,
    );
}
