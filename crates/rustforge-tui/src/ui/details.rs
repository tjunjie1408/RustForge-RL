use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{AppMode, AppState};
use crate::ui::theme::Theme;

pub fn render(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let metadata = app.run_metadata();
    let mode = match app.mode() {
        AppMode::Monitor => "read-only persisted metrics monitor",
        AppMode::Live => "integrated live training",
    };
    let lines = vec![
        field("Mode", Some(mode.to_owned()), theme, app.ascii()),
        field("Run ID", metadata.run_id.clone(), theme, app.ascii()),
        field("Algorithm", metadata.algorithm.clone(), theme, app.ascii()),
        field(
            "Environment",
            metadata.environment.clone(),
            theme,
            app.ascii(),
        ),
        field(
            "Seed",
            metadata.seed.map(|seed| seed.to_string()),
            theme,
            app.ascii(),
        ),
        field("Device", metadata.device.clone(), theme, app.ascii()),
        field(
            "Metrics",
            metadata
                .metrics_path
                .as_ref()
                .map(|path| path.display().to_string()),
            theme,
            app.ascii(),
        ),
        field(
            "Manifest",
            metadata
                .manifest_path
                .as_ref()
                .map(|path| path.display().to_string()),
            theme,
            app.ascii(),
        ),
        field(
            "Schema",
            metadata.schema_version.clone(),
            theme,
            app.ascii(),
        ),
        field(
            "Pause / resume",
            capability(app.live_controls_visible()),
            theme,
            app.ascii(),
        ),
        field(
            "Graceful / force stop",
            capability(app.live_controls_visible()),
            theme,
            app.ascii(),
        ),
        field("Checkpoint", Some("unavailable".into()), theme, app.ascii()),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(theme.block(" Run configuration and capabilities ", app.ascii())),
        area,
    );
}

fn capability(available: bool) -> Option<String> {
    Some(
        if available {
            "available"
        } else {
            "unavailable"
        }
        .into(),
    )
}

fn field<'a>(label: &'a str, value: Option<String>, theme: Theme, ascii: bool) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{label:<24}"), Style::default().fg(theme.muted)),
        Span::styled(
            value.unwrap_or_else(|| if ascii { "-".into() } else { "—".into() }),
            Style::default().fg(theme.text),
        ),
    ])
}
