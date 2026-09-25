use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::AppState;
use crate::metrics::MetricRow;
use crate::system::format_bytes;
use crate::ui::theme::Theme;
use crate::ui::{charts, events, format};

const LABEL_WIDTH: usize = 15;
const RECENT_WINDOW: usize = 100;
const SPARK_HISTORY: usize = 64;

/// Extracts one optional per-episode series from a metric row.
type SeriesValue = fn(&MetricRow) -> Option<f32>;

pub fn render(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let events_height = match area.height {
        26.. => 8,
        14.. => 5,
        _ => 0,
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(events_height)])
        .split(area);
    let side_width = match area.width {
        100.. => 36,
        80.. => 32,
        _ => 30,
    };
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(side_width)])
        .split(rows[0]);

    charts::render_reward(frame, top[0], app, theme);

    let training = training_lines(app, theme, usize::from(side_width.saturating_sub(2)));
    let side = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(training.len() as u16 + 2),
            Constraint::Min(0),
        ])
        .split(top[1]);
    frame.render_widget(
        Paragraph::new(training).block(theme.block(" Training ", theme.ascii)),
        side[0],
    );
    if side[1].height >= 3 {
        frame.render_widget(
            Paragraph::new(system_lines(app, theme)).block(theme.block(" System ", theme.ascii)),
            side[1],
        );
    }

    if events_height > 0 {
        let visible = usize::from(events_height.saturating_sub(2));
        frame.render_widget(
            Paragraph::new(events::recent_lines(app, theme, visible))
                .block(theme.block(" Events ", theme.ascii)),
            rows[1],
        );
    }
}

fn training_lines(app: &AppState, theme: Theme, inner_width: usize) -> Vec<Line<'static>> {
    let rows = app.episodes();
    let latest = rows.back();
    let best = rows
        .iter()
        .map(|row| f64::from(row.reward))
        .filter(|value| value.is_finite())
        .fold(None, |best: Option<f64>, value| {
            Some(best.map_or(value, |current| current.max(value)))
        });
    let recent: Vec<f64> = rows
        .iter()
        .rev()
        .take(RECENT_WINDOW)
        .map(|row| f64::from(row.reward))
        .filter(|value| value.is_finite())
        .collect();
    let recent_average =
        (!recent.is_empty()).then(|| recent.iter().sum::<f64>() / recent.len() as f64);

    let mut lines = vec![
        kpi(
            "Episode",
            latest.map(|row| format::grouped(row.episode)),
            theme,
        ),
        kpi(
            "Global step",
            latest.map(|row| format::grouped(row.global_step)),
            theme,
        ),
        kpi(
            "Latest reward",
            latest.map(|row| format!("{:.2}", row.reward)),
            theme,
        ),
        highlighted_kpi(
            "Best reward",
            best.map(|value| format!("{value:.2}")),
            theme,
        ),
        kpi(
            "Recent avg",
            recent_average.map(|value| format!("{value:.2}")),
            theme,
        ),
    ];
    if let Some(target) = app.target_reward() {
        let reached = best.is_some_and(|best| best >= target);
        let mut line = kpi("Target", Some(format!("{target:.2}")), theme);
        if reached {
            line.spans.push(Span::styled(
                if theme.ascii {
                    "  reached"
                } else {
                    "  ✓ reached"
                },
                Style::default().fg(theme.success),
            ));
        }
        lines.push(line);
    }

    let labels = app.metric_labels();
    let series: [(Option<&str>, SeriesValue, Color); 2] = [
        (
            labels.primary_loss.as_deref(),
            |row| row.primary_loss,
            theme.series_loss,
        ),
        (
            labels.policy_signal.as_deref(),
            |row| row.policy_signal,
            theme.series_policy,
        ),
    ];
    for (label, value, color) in series {
        let Some(label) = label else {
            continue;
        };
        let history: Vec<f64> = rows
            .iter()
            .rev()
            .take(SPARK_HISTORY)
            .rev()
            .filter_map(|row| value(row).map(f64::from))
            .collect();
        lines.push(spark_line(label, &history, color, theme, inner_width));
    }
    lines
}

fn kpi(label: &str, value: Option<String>, theme: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!(
                " {:<LABEL_WIDTH$}",
                format::truncate(label, LABEL_WIDTH - 1, theme.ascii)
            ),
            theme.muted_style(),
        ),
        Span::styled(
            value.unwrap_or_else(|| theme.dash().into()),
            theme.text_style(),
        ),
    ])
}

fn highlighted_kpi(label: &str, value: Option<String>, theme: Theme) -> Line<'static> {
    let mut line = kpi(label, value, theme);
    if let Some(value) = line.spans.last_mut() {
        value.style = Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD);
    }
    line
}

fn spark_line(
    label: &str,
    history: &[f64],
    color: Color,
    theme: Theme,
    inner_width: usize,
) -> Line<'static> {
    let latest = history
        .iter()
        .rev()
        .find(|value| value.is_finite())
        .map(|value| format::compact(*value))
        .unwrap_or_else(|| theme.dash().into());
    let spark_width = inner_width.saturating_sub(1 + LABEL_WIDTH + 1 + latest.chars().count());
    let spark = format::sparkline(history, spark_width, theme.ascii);
    let spark_width = if spark.is_empty() { 0 } else { spark_width };
    Line::from(vec![
        Span::styled(
            format!(
                " {:<LABEL_WIDTH$}",
                format::truncate(label, LABEL_WIDTH - 1, theme.ascii)
            ),
            theme.muted_style(),
        ),
        Span::styled(format!("{spark:<spark_width$}"), Style::default().fg(color)),
        Span::raw(" "),
        Span::styled(latest, theme.text_style()),
    ])
}

fn system_lines(app: &AppState, theme: Theme) -> Vec<Line<'static>> {
    let Some(snapshot) = app.system_snapshot() else {
        return vec![Line::styled(
            format!(" {}", theme.dash()),
            theme.muted_style(),
        )];
    };
    let dash = theme.dash();
    let cpu = snapshot
        .process_cpu_percent
        .map(|value| format!("{value:.1}%"))
        .unwrap_or_else(|| dash.into());
    let rss = snapshot
        .process_memory_bytes
        .map(format_bytes)
        .unwrap_or_else(|| dash.into());
    let memory = match (snapshot.used_memory_bytes, snapshot.total_memory_bytes) {
        (Some(used), Some(total)) => format!("{} / {}", format_bytes(used), format_bytes(total)),
        _ => dash.into(),
    };
    let separator = if theme.ascii { " / " } else { " · " };
    vec![
        Line::from(vec![
            Span::styled(" CPU ", theme.muted_style()),
            Span::styled(cpu, theme.text_style()),
            Span::styled("  RSS ", theme.muted_style()),
            Span::styled(rss, theme.text_style()),
        ]),
        Line::from(vec![
            Span::styled(" Mem ", theme.muted_style()),
            Span::styled(memory, theme.text_style()),
        ]),
        Line::styled(
            format!(
                " {} {}{separator}{} CPU",
                snapshot.os.as_deref().unwrap_or("unknown OS"),
                snapshot.architecture,
                snapshot.logical_cpus
            ),
            theme.muted_style(),
        ),
    ]
}
