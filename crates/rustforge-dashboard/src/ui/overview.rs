use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::app::AppState;
use crate::system::format_bytes;
use crate::ui::theme::Theme;

pub fn render(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
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
        .take(100)
        .map(|row| f64::from(row.reward))
        .filter(|value| value.is_finite())
        .collect();
    let recent_average = if recent.is_empty() {
        None
    } else {
        Some(recent.iter().sum::<f64>() / recent.len() as f64)
    };

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(5)])
        .split(area);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(sections[0]);

    let kpis = vec![
        metric_line(
            "Episode",
            latest.map(|row| row.episode.to_string()),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Latest reward",
            latest.map(|row| format!("{:.2}", row.reward)),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Best reward",
            best.map(|value| format!("{value:.2}")),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Recent avg (100)",
            recent_average.map(|value| format!("{value:.2}")),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Global step",
            latest.map(|row| row.global_step.to_string()),
            theme,
            app.ascii(),
        ),
    ];
    frame.render_widget(
        Paragraph::new(kpis).block(theme.block(" Training KPIs ", app.ascii())),
        top[0],
    );

    let snapshot = app.system_snapshot();
    let insights = app.monitor_insights();
    let system = vec![
        metric_line(
            "Elapsed",
            Some(format_duration(insights.elapsed)),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Steps/sec",
            insights.steps_per_second.map(|value| format!("{value:.1}")),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Episodes/min",
            insights
                .episodes_per_minute
                .map(|value| format!("{value:.2}")),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Progress / ETA",
            insights.progress_fraction.map(|progress| {
                let eta = insights.eta.map(format_duration).unwrap_or_else(|| {
                    if app.ascii() {
                        "-".into()
                    } else {
                        "—".into()
                    }
                });
                format!("{:.1}% / {eta}", progress * 100.0)
            }),
            theme,
            app.ascii(),
        ),
        metric_line(
            "System",
            snapshot.map(|snapshot| {
                format!(
                    "{} {} / {} logical CPU",
                    snapshot.os.as_deref().unwrap_or("unknown OS"),
                    snapshot.architecture,
                    snapshot.logical_cpus
                )
            }),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Process",
            snapshot.map(|snapshot| {
                let cpu = snapshot
                    .process_cpu_percent
                    .map(|value| format!("CPU {value:.1}%"))
                    .unwrap_or_else(|| "CPU —".into());
                let rss = snapshot
                    .process_memory_bytes
                    .map(|value| format!("RSS {}", format_bytes(value)))
                    .unwrap_or_else(|| "RSS —".into());
                format!("{cpu} / {rss}")
            }),
            theme,
            app.ascii(),
        ),
        metric_line(
            "Memory",
            snapshot.and_then(|snapshot| {
                Some(format!(
                    "{} / {}",
                    format_bytes(snapshot.used_memory_bytes?),
                    format_bytes(snapshot.total_memory_bytes?)
                ))
            }),
            theme,
            app.ascii(),
        ),
    ];
    frame.render_widget(
        Paragraph::new(system).block(theme.block(" Live insights ", app.ascii())),
        top[1],
    );

    let mut activity: Vec<Line<'_>> = app
        .activity()
        .iter()
        .rev()
        .take(sections[1].height.saturating_sub(2) as usize)
        .map(|item| {
            Line::from(vec![
                Span::styled(format!("{:?}: ", item.kind), theme.title()),
                Span::styled(item.message.as_str(), Style::default().fg(theme.text)),
            ])
        })
        .collect();
    if insights.stalled {
        activity.insert(
            0,
            Line::styled(
                "STALL: no new progress for at least 30 seconds",
                Style::default().fg(theme.warning),
            ),
        );
    }
    for alert in insights.alerts.iter().rev() {
        activity.insert(
            0,
            Line::styled(
                format!("ALERT {:?}: {:.2}", alert.kind, alert.value),
                Style::default().fg(theme.warning),
            ),
        );
    }
    let activity = if activity.is_empty() {
        vec![Line::styled(
            "No source activity yet",
            Style::default().fg(theme.muted),
        )]
    } else {
        activity
    };
    frame.render_widget(
        Paragraph::new(activity)
            .wrap(Wrap { trim: true })
            .block(theme.block(" Recent activity ", app.ascii())),
        sections[1],
    );
}

fn format_duration(duration: std::time::Duration) -> String {
    let seconds = duration.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

fn metric_line<'a>(label: &'a str, value: Option<String>, theme: Theme, ascii: bool) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{label:<20}"), Style::default().fg(theme.muted)),
        Span::styled(
            value.unwrap_or_else(|| if ascii { "-".into() } else { "—".into() }),
            Style::default().fg(theme.text),
        ),
    ])
}
