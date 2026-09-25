use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Chart, Dataset, GraphType, Paragraph};
use ratatui::Frame;

use crate::analytics::{downsample_min_max, rolling_average};
use crate::app::AppState;
use crate::ui::format;
use crate::ui::theme::Theme;

const TREND_WINDOW: usize = 100;

pub fn render(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let labels = app.metric_labels();
    let panel_count = 1
        + usize::from(labels.primary_loss.is_some())
        + usize::from(labels.policy_signal.is_some());
    let constraints = vec![Constraint::Ratio(1, panel_count as u32); panel_count];
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);
    render_reward(frame, sections[0], app, theme);
    let mut panel = 1;
    if let Some(label) = labels.primary_loss.as_deref() {
        render_single(
            frame,
            sections[panel],
            app,
            theme,
            label,
            |row| row.primary_loss.map(f64::from),
            theme.series_loss,
        );
        panel += 1;
    }
    if let Some(label) = labels.policy_signal.as_deref() {
        render_single(
            frame,
            sections[panel],
            app,
            theme,
            label,
            |row| row.policy_signal.map(f64::from),
            theme.series_policy,
        );
    }
}

/// Reward series plus its trailing average; shared by the overview and charts views.
pub(super) fn render_reward(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let label = app.metric_labels().episode_reward.as_str();
    let title = format!(" {label} ");
    let raw: Vec<(f64, Option<f64>)> = app
        .chart_rows()
        .into_iter()
        .map(|row| (row.episode as f64, finite(f64::from(row.reward))))
        .collect();
    let finite_points: Vec<(f64, f64)> =
        raw.iter().filter_map(|(x, y)| y.map(|y| (*x, y))).collect();
    if finite_points.is_empty() {
        render_no_data(frame, area, app, theme, &title);
        return;
    }
    let values: Vec<f64> = finite_points.iter().map(|(_, y)| *y).collect();
    let trend_points: Vec<(f64, Option<f64>)> = finite_points
        .iter()
        .zip(rolling_average(&values, TREND_WINDOW))
        .map(|((x, _), average)| (*x, Some(average)))
        .collect();
    let cap = chart_point_cap(area);
    let reward = flatten(downsample_min_max(&raw, cap));
    let trend = flatten(downsample_min_max(&trend_points, cap));
    let x_bounds = x_bounds(finite_points.iter().map(|(x, _)| *x));
    let y_bounds = bounds(reward.iter().chain(trend.iter()).map(|(_, y)| *y));

    let (dot, line) = if theme.ascii {
        ("*", "--")
    } else {
        ("•", "━━")
    };
    let trend_label = if area.width >= 60 {
        format!(" rolling avg {TREND_WINDOW} ")
    } else {
        format!(" avg {TREND_WINDOW} ")
    };
    let legend = Line::from(vec![
        Span::styled(dot, Style::default().fg(theme.series_raw)),
        Span::styled(" raw  ", theme.muted_style()),
        Span::styled(line, Style::default().fg(theme.series_trend)),
        Span::styled(trend_label, theme.muted_style()),
    ])
    .alignment(Alignment::Right);
    // Raw rewards are noisy, so draw them as points and let the trend carry the line.
    let datasets = vec![
        dataset(&reward, theme.series_raw, theme).graph_type(GraphType::Scatter),
        dataset(&trend, theme.series_trend, theme),
    ];
    frame.render_widget(
        Chart::new(datasets)
            .block(theme.block(title, theme.ascii).title_top(legend))
            .x_axis(x_axis(x_bounds, theme))
            .y_axis(y_axis(y_bounds, theme))
            .legend_position(None),
        area,
    );
}

#[allow(clippy::too_many_arguments)]
fn render_single<F>(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &AppState,
    theme: Theme,
    label: &str,
    value: F,
    color: Color,
) where
    F: Fn(&crate::metrics::MetricRow) -> Option<f64>,
{
    let title = format!(" {label} ");
    let raw: Vec<(f64, Option<f64>)> = app
        .chart_rows()
        .into_iter()
        .map(|row| (row.episode as f64, value(row).and_then(finite)))
        .collect();
    let segments = sampled_segments(&raw, chart_point_cap(area));
    if segments.is_empty() {
        render_no_data(frame, area, app, theme, &title);
        return;
    }
    let latest = segments
        .last()
        .and_then(|segment| segment.last())
        .map(|(_, y)| *y);
    let x_bounds = x_bounds(segments.iter().flatten().map(|(x, _)| *x));
    let y_bounds = bounds(segments.iter().flatten().map(|(_, y)| *y));
    let datasets: Vec<Dataset<'_>> = segments
        .iter()
        .map(|points| dataset(points, color, theme))
        .collect();
    let mut block = theme.block(title, theme.ascii);
    if let Some(latest) = latest {
        block = block.title_top(
            Line::from(vec![
                Span::styled("latest ", theme.muted_style()),
                Span::styled(
                    format!("{} ", format::compact(latest)),
                    Style::default().fg(color),
                ),
            ])
            .alignment(Alignment::Right),
        );
    }
    frame.render_widget(
        Chart::new(datasets)
            .block(block)
            .x_axis(x_axis(x_bounds, theme))
            .y_axis(y_axis(y_bounds, theme))
            .legend_position(None),
        area,
    );
}

fn dataset<'a>(points: &'a [(f64, f64)], color: Color, theme: Theme) -> Dataset<'a> {
    Dataset::default()
        .marker(if theme.ascii {
            Marker::Dot
        } else {
            Marker::Braille
        })
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(points)
}

fn render_no_data(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme, title: &str) {
    let message = if app.episodes().is_empty() {
        "Waiting for the first completed episode…"
    } else {
        "No finite data in this range"
    };
    let message = if theme.ascii {
        message.replace('…', "...")
    } else {
        message.to_owned()
    };
    frame.render_widget(
        Paragraph::new(message)
            .style(theme.muted_style())
            .alignment(Alignment::Center)
            .block(theme.block(title.to_owned(), app.ascii())),
        area,
    );
}

fn x_axis<'a>(bounds: [f64; 2], theme: Theme) -> Axis<'a> {
    Axis::default()
        .style(Style::default().fg(theme.border))
        .bounds(bounds)
        .labels([
            Span::styled(format!("ep {:.0}", bounds[0]), theme.muted_style()),
            Span::styled(format!("{:.0}", bounds[1]), theme.muted_style()),
        ])
}

fn y_axis<'a>(bounds: [f64; 2], theme: Theme) -> Axis<'a> {
    let middle = (bounds[0] + bounds[1]) / 2.0;
    Axis::default()
        .style(Style::default().fg(theme.border))
        .bounds(bounds)
        .labels_alignment(Alignment::Right)
        .labels(
            [bounds[0], middle, bounds[1]]
                .map(|value| Span::styled(format::compact(value), theme.muted_style())),
        )
}

fn chart_point_cap(area: Rect) -> usize {
    usize::from(area.width.saturating_sub(4))
        .saturating_mul(2)
        .max(2)
}

fn flatten(points: Vec<(f64, Option<f64>)>) -> Vec<(f64, f64)> {
    points
        .into_iter()
        .filter_map(|(x, y)| y.filter(|value| value.is_finite()).map(|y| (x, y)))
        .collect()
}

fn sampled_segments(points: &[(f64, Option<f64>)], max_points: usize) -> Vec<Vec<(f64, f64)>> {
    if max_points == 0 {
        return Vec::new();
    }
    let mut raw_segments: Vec<Vec<(f64, Option<f64>)>> = Vec::new();
    let mut current = Vec::new();
    for point in points {
        if point.1.filter(|value| value.is_finite()).is_some() {
            current.push(*point);
        } else if !current.is_empty() {
            raw_segments.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        raw_segments.push(current);
    }
    raw_segments.truncate(max_points);
    if raw_segments.is_empty() {
        return Vec::new();
    }

    let per_segment = (max_points / raw_segments.len()).max(1);
    raw_segments
        .into_iter()
        .map(|segment| flatten(downsample_min_max(&segment, per_segment)))
        .filter(|segment| !segment.is_empty())
        .collect()
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// Episode axis spans exactly the data so the edge labels are real episode numbers.
fn x_bounds(values: impl Iterator<Item = f64>) -> [f64; 2] {
    let [minimum, maximum] = extent(values).unwrap_or([0.0, 1.0]);
    if minimum == maximum {
        [minimum - 1.0, maximum + 1.0]
    } else {
        [minimum, maximum]
    }
}

fn bounds(values: impl Iterator<Item = f64>) -> [f64; 2] {
    let Some([minimum, maximum]) = extent(values) else {
        return [0.0, 1.0];
    };
    if minimum == maximum {
        let padding = minimum.abs().max(1.0) * 0.05;
        return [minimum - padding, maximum + padding];
    }
    let padding = (maximum - minimum) * 0.05;
    [minimum - padding, maximum + padding]
}

fn extent(values: impl Iterator<Item = f64>) -> Option<[f64; 2]> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for value in values.filter(|value| value.is_finite()) {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    (minimum.is_finite() && maximum.is_finite()).then_some([minimum, maximum])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_split_line_datasets_and_total_points_stay_bounded() {
        let points = vec![
            (0.0, Some(1.0)),
            (1.0, Some(0.9)),
            (2.0, None),
            (3.0, Some(0.7)),
            (4.0, Some(0.6)),
        ];
        let segments = sampled_segments(&points, 4);
        assert_eq!(segments.len(), 2);
        assert!(segments.iter().map(Vec::len).sum::<usize>() <= 4);
        assert_eq!(segments[0].last().unwrap().0, 1.0);
        assert_eq!(segments[1].first().unwrap().0, 3.0);
    }

    #[test]
    fn chart_bounds_handle_empty_constant_non_finite_and_extreme_values() {
        assert_eq!(bounds(std::iter::empty()), [0.0, 1.0]);
        let constant = bounds([5.0, 5.0].into_iter());
        assert!(constant[0] < 5.0 && constant[1] > 5.0);
        assert_eq!(bounds([f64::NAN, f64::INFINITY].into_iter()), [0.0, 1.0]);
        let extreme = bounds([-1.0e12, 1.0e12].into_iter());
        assert!(extreme[0] < -1.0e12 && extreme[1] > 1.0e12);
    }

    #[test]
    fn episode_axis_is_unpadded_and_widens_single_points() {
        assert_eq!(x_bounds([3.0, 10.0].into_iter()), [3.0, 10.0]);
        assert_eq!(x_bounds([4.0].into_iter()), [3.0, 5.0]);
        assert_eq!(x_bounds(std::iter::empty()), [0.0, 1.0]);
    }
}
