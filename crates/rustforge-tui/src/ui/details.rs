use ratatui::layout::Rect;
use ratatui::style::Modifier;
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
    let path =
        |path: &Option<std::path::PathBuf>| path.as_ref().map(|path| path.display().to_string());
    let mut lines = vec![
        section("Run", theme),
        field("Mode", Some(mode.to_owned()), theme),
        field("Run ID", metadata.run_id.clone(), theme),
        field("Algorithm", metadata.algorithm.clone(), theme),
        field("Environment", metadata.environment.clone(), theme),
        field("Seed", metadata.seed.map(|seed| seed.to_string()), theme),
        field("Device", metadata.device.clone(), theme),
        Line::default(),
        section("Artifacts", theme),
        field("Metrics", path(&metadata.metrics_path), theme),
        field("Manifest", path(&metadata.manifest_path), theme),
        field("Schema", metadata.schema_version.clone(), theme),
        Line::default(),
        section("Controls", theme),
        capability("Pause / resume", app.live_controls_visible(), theme),
        capability("Graceful / force stop", app.live_controls_visible(), theme),
        capability("Checkpoint", false, theme),
    ];
    if !metadata.configuration.is_empty() {
        lines.push(Line::default());
        lines.push(section("Training configuration", theme));
        lines.extend(
            metadata
                .configuration
                .iter()
                .map(|(label, value)| field(label, Some(value.clone()), theme)),
        );
    }
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((app.scroll_offset().min(u16::MAX as usize) as u16, 0))
            .block(theme.block(" Run details ", app.ascii())),
        area,
    );
}

fn section(title: &str, theme: Theme) -> Line<'_> {
    Line::styled(
        format!(" {title}"),
        theme.title().add_modifier(Modifier::UNDERLINED),
    )
}

fn capability(label: &str, available: bool, theme: Theme) -> Line<'_> {
    let (text, color) = if available {
        ("available", theme.success)
    } else {
        ("unavailable", theme.muted)
    };
    let mut line = field(label, Some(text.into()), theme);
    if let Some(value) = line.spans.last_mut() {
        value.style = value.style.fg(color);
    }
    line
}

fn field<'a>(label: &'a str, value: Option<String>, theme: Theme) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("   {label:<24}"), theme.muted_style()),
        Span::styled(
            value.unwrap_or_else(|| theme.dash().into()),
            theme.text_style(),
        ),
    ])
}
