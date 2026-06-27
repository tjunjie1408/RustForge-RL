//! The metric row parsed from a training CSV line.
use serde::Serialize;

/// One row of `episode,reward,avg_loss,epsilon,global_step`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MetricRow {
    pub episode: u64,
    pub reward: f32,
    /// `None` when no training occurred yet (the trainer writes NaN before warmup).
    pub avg_loss: Option<f32>,
    pub epsilon: f32,
    pub global_step: u64,
}

/// Parse one CSV line into a `MetricRow`.
///
/// Returns `None` for the header, blank lines, and malformed/partial lines.
/// The `avg_loss` field maps to `None` when empty, `NaN`, or non-finite.
pub fn parse_line(line: &str) -> Option<MetricRow> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let mut f = line.split(',');
    let episode = f.next()?.trim().parse::<u64>().ok()?; // header "episode" -> None
    let reward = f.next()?.trim().parse::<f32>().ok()?;
    let avg_loss = match f.next()?.trim().parse::<f32>() {
        Ok(v) if v.is_finite() => Some(v),
        _ => None, // empty / NaN / inf -> None
    };
    let epsilon = f.next()?.trim().parse::<f32>().ok()?;
    let global_step = f.next()?.trim().parse::<u64>().ok()?;
    Some(MetricRow { episode, reward, avg_loss, epsilon, global_step })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_row() {
        let r = parse_line("5,12.5,0.3,0.8,140").unwrap();
        assert_eq!(
            r,
            MetricRow { episode: 5, reward: 12.5, avg_loss: Some(0.3), epsilon: 0.8, global_step: 140 }
        );
    }

    #[test]
    fn header_is_none() {
        assert!(parse_line("episode,reward,avg_loss,epsilon,global_step").is_none());
    }

    #[test]
    fn blank_is_none() {
        assert!(parse_line("").is_none());
        assert!(parse_line("   ").is_none());
    }

    #[test]
    fn malformed_or_partial_is_none() {
        assert!(parse_line("5,12.5,0.3").is_none()); // too few fields
        assert!(parse_line("abc,1,2,3,4").is_none()); // non-numeric episode
    }

    #[test]
    fn nan_or_empty_avg_loss_maps_to_none() {
        assert_eq!(parse_line("1,9.0,NaN,1.0,10").unwrap().avg_loss, None);
        assert_eq!(parse_line("1,9.0,,1.0,10").unwrap().avg_loss, None);
        assert_eq!(parse_line("1,9.0,inf,1.0,10").unwrap().avg_loss, None);
    }
}
