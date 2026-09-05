//! The metric row parsed from a training CSV line.
use serde::Serialize;

/// Stable header used by the existing DQN CSV metrics format.
pub const DQN_CSV_V1_HEADER: &str = "episode,reward,avg_loss,epsilon,global_step";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricLabels {
    pub episode_reward: String,
    pub primary_loss: Option<String>,
    pub policy_signal: Option<String>,
    pub throughput: String,
}

impl MetricLabels {
    pub fn dqn_monitor_defaults() -> Self {
        Self {
            episode_reward: "Reward".into(),
            primary_loss: Some("Loss".into()),
            policy_signal: Some("Exploration / epsilon".into()),
            throughput: "Steps/sec".into(),
        }
    }
}

/// One reducer-facing row mapped from either persisted or live metrics.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MetricRow {
    pub episode: u64,
    pub reward: f32,
    /// `None` when no training occurred yet (the trainer writes NaN before warmup).
    pub primary_loss: Option<f32>,
    pub policy_signal: Option<f32>,
    pub global_step: u64,
}

/// Parse one CSV line into a `MetricRow`.
///
/// Returns `None` for the header, blank lines, and malformed/partial lines.
/// Legacy `avg_loss` maps to an optional semantic field; DQN CSV v1 requires a
/// present finite `epsilon`, which is stored in the optional semantic field.
pub fn parse_line(line: &str) -> Option<MetricRow> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let mut f = line.split(',');
    let episode = f.next()?.trim().parse::<u64>().ok()?; // header "episode" -> None
    let reward = f.next()?.trim().parse::<f32>().ok()?;
    let primary_loss = match f.next()?.trim().parse::<f32>() {
        Ok(v) if v.is_finite() => Some(v),
        _ => None, // empty / NaN / inf -> None
    };
    let policy_signal = f
        .next()?
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite())
        .map(Some)?;
    let global_step = f.next()?.trim().parse::<u64>().ok()?;
    Some(MetricRow {
        episode,
        reward,
        primary_loss,
        policy_signal,
        global_step,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_row() {
        let r = parse_line("5,12.5,0.3,0.8,140").unwrap();
        assert_eq!(
            r,
            MetricRow {
                episode: 5,
                reward: 12.5,
                primary_loss: Some(0.3),
                policy_signal: Some(0.8),
                global_step: 140
            }
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
    fn non_finite_or_empty_loss_maps_to_none() {
        assert_eq!(parse_line("1,9.0,NaN,1.0,10").unwrap().primary_loss, None);
        assert_eq!(parse_line("1,9.0,,1.0,10").unwrap().primary_loss, None);
        assert_eq!(parse_line("1,9.0,inf,1.0,10").unwrap().primary_loss, None);
    }

    #[test]
    fn missing_or_non_finite_dqn_epsilon_rejects_the_row() {
        for epsilon in ["", "NaN", "inf", "-inf"] {
            assert!(parse_line(&format!("1,9.0,0.5,{epsilon},10")).is_none());
        }
    }
}
