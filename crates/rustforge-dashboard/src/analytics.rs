//! Renderer-independent dashboard calculations.

use crate::metrics::MetricRow;

/// KPI values shown by both the Web dashboard and its terminal replacement.
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardStats {
    pub episode: u64,
    pub global_step: u64,
    pub latest_reward: f64,
    pub best_reward: f64,
    pub recent_average_reward: f64,
}

/// Calculate the latest Web-compatible KPI snapshot.
pub fn dashboard_stats(rows: &[MetricRow], recent_window: usize) -> Option<DashboardStats> {
    let latest = rows.last()?;
    let window = recent_window.max(1);
    let recent = &rows[rows.len().saturating_sub(window)..];
    let recent_average_reward =
        recent.iter().map(|row| f64::from(row.reward)).sum::<f64>() / recent.len() as f64;
    let best_reward = rows
        .iter()
        .map(|row| f64::from(row.reward))
        .fold(f64::NEG_INFINITY, f64::max);

    Some(DashboardStats {
        episode: latest.episode,
        global_step: latest.global_step,
        latest_reward: f64::from(latest.reward),
        best_reward,
        recent_average_reward,
    })
}

/// Return the trailing-window average at every input position.
pub fn rolling_average(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut result = Vec::with_capacity(values.len());
    let mut sum = 0.0;

    for (index, value) in values.iter().copied().enumerate() {
        sum += value;
        if index >= window {
            sum -= values[index - window];
        }
        result.push(sum / (index + 1).min(window) as f64);
    }

    result
}

/// Downsample ordered points while retaining each bucket's finite extrema.
///
/// Missing values are retained unchanged when no sampling is needed. When a
/// bucket contains only gaps, one gap is emitted so the renderer can preserve
/// the discontinuity.
pub fn downsample_min_max(
    points: &[(f64, Option<f64>)],
    max_points: usize,
) -> Vec<(f64, Option<f64>)> {
    if points.len() <= max_points {
        return points.to_vec();
    }
    if max_points == 0 {
        return Vec::new();
    }
    if max_points == 1 {
        return vec![points[0]];
    }

    let bucket_count = max_points / 2;
    let bucket_size = points.len().div_ceil(bucket_count);
    let mut sampled = Vec::with_capacity(max_points);

    for bucket in points.chunks(bucket_size) {
        let mut minimum: Option<(usize, f64)> = None;
        let mut maximum: Option<(usize, f64)> = None;

        for (index, (_, value)) in bucket.iter().enumerate() {
            let Some(value) = value.filter(|value| value.is_finite()) else {
                continue;
            };
            if minimum.map_or(true, |(_, current)| value < current) {
                minimum = Some((index, value));
            }
            if maximum.map_or(true, |(_, current)| value > current) {
                maximum = Some((index, value));
            }
        }

        match (minimum, maximum) {
            (Some(minimum), Some(maximum)) if minimum.0 != maximum.0 => {
                let (first, second) = if minimum.0 < maximum.0 {
                    (minimum, maximum)
                } else {
                    (maximum, minimum)
                };
                sampled.push((bucket[first.0].0, Some(first.1)));
                sampled.push((bucket[second.0].0, Some(second.1)));
            }
            (Some(point), _) => sampled.push((bucket[point.0].0, Some(point.1))),
            _ => sampled.push((bucket[0].0, None)),
        }
    }

    sampled.truncate(max_points);
    sampled
}
