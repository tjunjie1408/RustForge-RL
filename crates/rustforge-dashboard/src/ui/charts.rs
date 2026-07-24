use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::symbols::Marker;
use ratatui::text::Span;
use ratatui::widgets::{Axis, Chart, Dataset, GraphType, Paragraph};
use ratatui::Frame;

use crate::analytics::{downsample_min_max, rolling_average};
use crate::app::AppState;
use crate::ui::theme::Theme;

pub fn render(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(area);
    render_reward(frame, sections[0], app, theme);
    render_single(
        frame,
        sections[1],
        app,
        theme,
        " Loss ",
        "Loss",
        |row| row.avg_loss.map(f64::from),
        theme.warning,
    );
    render_single(
        frame,
        sections[2],
        app,
        theme,
        " Exploration / epsilon ",
        "epsilon",
        |row| Some(f64::from(row.epsilon)),
        theme.success,
    );
}

fn render_reward(frame: &mut Frame<'_>, area: Rect, app: &AppState, theme: Theme) {
    let raw: Vec<(f64, Option<f64>)> = app
        .episodes()
        .iter()
        .map(|row| (row.episode as f64, finite(f64::from(row.reward))))
        .collect();
    if raw.is_empty() {
        render_no_data(frame, area, app, theme, " Reward + rolling avg ");
        return;
    }
    let rewards: Vec<f64> = raw.iter().map(|(_, value)| value.unwrap_or(0.0)).collect();
    let averages = rolling_average(&rewards, 100);
    let average_points: Vec<(f64, Option<f64>)> = raw
        .iter()
        .zip(averages)
        .map(|((x, _), average)| (*x, Some(average)))
        .collect();
    let cap = chart_point_cap(area);
    let reward = flatten(downsample_min_max(&raw, cap));
    let average = flatten(downsample_min_max(&average_points, cap));
    let all_values = reward.iter().chain(average.iter()).map(|(_, value)| *value);
    let x_bounds = bounds(reward.iter().map(|(x, _)| *x));
    let y_bounds = bounds(all_values);
    let datasets = vec![
        Dataset::default()
            .name("reward")
            .marker(if app.ascii() {
                Marker::Dot
            } else {
                Marker::Braille
            })
            .graph_type(GraphType::Line)
            .style(Style::default().fg(theme.warning))
            .data(&reward),
        Dataset::default()
            .name("rolling avg (100)")
            .marker(if app.ascii() {
                Marker::Dot
            } else {
                Marker::Braille
            })
            .graph_type(GraphType::Line)
            .style(Style::default().fg(theme.accent))
            .data(&average),
    ];
    frame.render_widget(
        Chart::new(datasets)
            .block(theme.block(" Reward + rolling avg ", app.ascii()))
            .x_axis(axis("episode", x_bounds, theme))
            .y_axis(axis("reward", y_bounds, theme)),
        area,
    );
}

#[allow(clippy::too_many_arguments)]
fn render_single<F>(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &AppState,
    theme: Theme,
    title: &'static str,
    label: &'static str,
    value: F,
    color: ratatui::style::Color,
) where
    F: Fn(&crate::metrics::MetricRow) -> Option<f64>,
{
    let raw: Vec<(f64, Option<f64>)> = app
        .episodes()
        .iter()
        .map(|row| (row.episode as f64, value(row).and_then(finite)))
        .collect();
    let segments = sampled_segments(&raw, chart_point_cap(area));
    if segments.is_empty() {
        render_no_data(frame, area, app, theme, title);
        return;
    }
    let x_bounds = bounds(segments.iter().flatten().map(|(x, _)| *x));
    let y_bounds = bounds(segments.iter().flatten().map(|(_, y)| *y));
    let datasets: Vec<Dataset<'_>> = segments
        .iter()
        .enumerate()
        .map(|(index, points)| {
            Dataset::default()
                .name(if index == 0 { label } else { "" })
                .marker(if app.ascii() {
                    Marker::Dot
                } else {
                    Marker::Braille
                })
                .graph_type(GraphType::Line)
                .style(Style::default().fg(color))
                .data(points)
        })
        .collect();
    frame.render_widget(
        Chart::new(datasets)
            .block(theme.block(title, app.ascii()))
            .x_axis(axis("episode", x_bounds, theme))
            .y_axis(axis(label, y_bounds, theme)),
        area,
    );
}

fn render_no_data(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &AppState,
    theme: Theme,
    title: &'static str,
) {
    frame.render_widget(
        Paragraph::new("No finite data")
            .style(Style::default().fg(theme.muted))
            .block(theme.block(title, app.ascii())),
        area,
    );
}

fn axis(title: &'static str, bounds: [f64; 2], theme: Theme) -> Axis<'static> {
    Axis::default()
        .title(Span::styled(title, Style::default().fg(theme.muted)))
        .style(Style::default().fg(theme.muted))
        .bounds(bounds)
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

fn bounds(values: impl Iterator<Item = f64>) -> [f64; 2] {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for value in values.filter(|value| value.is_finite()) {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return [0.0, 1.0];
    }
    if minimum == maximum {
        let padding = minimum.abs().max(1.0) * 0.05;
        return [minimum - padding, maximum + padding];
    }
    let padding = (maximum - minimum) * 0.05;
    [minimum - padding, maximum + padding]
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
}
