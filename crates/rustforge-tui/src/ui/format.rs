//! Small, renderer-independent text formatting helpers.

use std::time::Duration;

const SPARK_UNICODE: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const SPARK_ASCII: [char; 5] = ['_', '.', '-', '=', '#'];

/// Short numeric label for axes and rates, e.g. `1.8k`, `-0.031`, `2.4M`.
pub fn compact(value: f64) -> String {
    if !value.is_finite() {
        return "NaN".into();
    }
    let magnitude = value.abs();
    if magnitude >= 1.0e9 {
        format!("{:.1}G", value / 1.0e9)
    } else if magnitude >= 1.0e6 {
        format!("{:.1}M", value / 1.0e6)
    } else if magnitude >= 1.0e4 {
        format!("{:.1}k", value / 1.0e3)
    } else if magnitude >= 100.0 {
        format!("{value:.0}")
    } else if magnitude >= 1.0 {
        format!("{value:.1}")
    } else if magnitude >= 0.001 || magnitude == 0.0 {
        let text = format!("{value:.3}");
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        format!("{value:.1e}")
    }
}

/// Integer with thousands separators, e.g. `14,950`.
pub fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            output.push(',');
        }
        output.push(digit);
    }
    output
}

pub fn duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

/// Render the trailing `width` values as a one-line sparkline.
pub fn sparkline(values: &[f64], width: usize, ascii: bool) -> String {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let tail = &finite[finite.len().saturating_sub(width)..];
    if tail.is_empty() {
        return String::new();
    }
    let (minimum, maximum) = tail
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(*value), hi.max(*value))
        });
    let levels: &[char] = if ascii { &SPARK_ASCII } else { &SPARK_UNICODE };
    let top = (levels.len() - 1) as f64;
    tail.iter()
        .map(|value| {
            let level = if maximum > minimum {
                ((value - minimum) / (maximum - minimum) * top).round() as usize
            } else {
                levels.len() / 2
            };
            levels[level.min(levels.len() - 1)]
        })
        .collect()
}

/// Split a progress bar of `width` cells into its filled and empty parts.
pub fn progress_bar(fraction: f64, width: usize, ascii: bool) -> (String, String) {
    let fraction = if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = ((fraction * width as f64).round() as usize).min(width);
    let (full, empty) = if ascii { ('#', '-') } else { ('█', '░') };
    (
        std::iter::repeat(full).take(filled).collect(),
        std::iter::repeat(empty).take(width - filled).collect(),
    )
}

/// Truncate to at most `width` characters, marking the cut.
pub fn truncate(text: &str, width: usize, ascii: bool) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    let mut output: String = text.chars().take(width - 1).collect();
    output.push(if ascii { '~' } else { '…' });
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_covers_magnitudes_and_non_finite() {
        assert_eq!(compact(1_834.0), "1834");
        assert_eq!(compact(18_340.0), "18.3k");
        assert_eq!(compact(2_400_000.0), "2.4M");
        assert_eq!(compact(12.34), "12.3");
        assert_eq!(compact(-0.0312), "-0.031");
        assert_eq!(compact(0.0), "0");
        assert_eq!(compact(0.00002), "2.0e-5");
        assert_eq!(compact(f64::NAN), "NaN");
    }

    #[test]
    fn grouped_inserts_separators() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(950), "950");
        assert_eq!(grouped(14_950), "14,950");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn sparkline_scales_to_range_and_skips_non_finite() {
        assert_eq!(sparkline(&[0.0, f64::NAN, 1.0], 8, false), "▁█");
        assert_eq!(sparkline(&[2.0, 2.0], 8, true), "--");
        assert_eq!(sparkline(&[1.0, 2.0, 3.0], 2, true), "_#");
        assert_eq!(sparkline(&[], 4, false), "");
    }

    #[test]
    fn progress_bar_clamps_fraction() {
        assert_eq!(progress_bar(0.5, 4, true), ("##".into(), "--".into()));
        assert_eq!(progress_bar(2.0, 3, true), ("###".into(), String::new()));
        assert_eq!(
            progress_bar(f64::NAN, 2, true),
            (String::new(), "--".into())
        );
    }

    #[test]
    fn truncate_marks_cut() {
        assert_eq!(truncate("abcdef", 4, true), "abc~");
        assert_eq!(truncate("abc", 4, false), "abc");
    }
}
