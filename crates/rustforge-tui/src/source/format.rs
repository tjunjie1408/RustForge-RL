//! Metric CSV formats the monitor recognizes, detected from the header line.
//!
//! - RustForge DQN CSV v1: `episode,reward,avg_loss,epsilon,global_step`.
//! - Stable-Baselines3 `Monitor` files (`monitor.csv`): a `#{json}` metadata
//!   line, then `r,l,t[,info keys...]` with one row per finished episode.
//! - Stable-Baselines3 CSV logger files (`progress.csv`): one row per log
//!   dump with `/`-separated keys such as `rollout/ep_rew_mean`.

use std::borrow::Cow;

use crate::metrics::{parse_line, MetricLabels, MetricRow, DQN_CSV_V1_HEADER};

/// A recognized metrics CSV format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsvFormat {
    RustforgeDqnV1,
    Sb3Monitor,
    Sb3Progress,
}

impl CsvFormat {
    /// Stable identifier shown as the run's schema.
    pub fn schema_name(self) -> &'static str {
        match self {
            Self::RustforgeDqnV1 => "dqn-csv-v1",
            Self::Sb3Monitor => "sb3-monitor",
            Self::Sb3Progress => "sb3-progress",
        }
    }
}

/// Header diagnostic for files that match no supported format.
pub(crate) const UNSUPPORTED_HEADER: &str =
    "unrecognized CSV header; expected RustForge DQN CSV v1 \
     (`episode,reward,avg_loss,epsilon,global_step`), an SB3 monitor.csv (`r,l,t`), \
     or an SB3 progress.csv (`time/total_timesteps`, ...)";

/// Outcome of parsing one data line.
#[derive(Debug, PartialEq)]
pub(crate) enum ParsedLine {
    Row(MetricRow),
    /// A well-formed line that carries no episode metric yet.
    Skip,
    Malformed(&'static str),
}

/// Stateful row parser bound to the header it was detected from.
#[derive(Clone, Debug)]
pub(crate) struct RowParser {
    layout: Layout,
    rows: u64,
    steps: u64,
}

#[derive(Clone, Debug)]
enum Layout {
    DqnV1,
    Sb3Monitor {
        fields: usize,
        reward: usize,
        length: usize,
    },
    Sb3Progress(ProgressColumns),
}

#[derive(Clone, Debug)]
struct ProgressColumns {
    fields: usize,
    reward: Option<usize>,
    total_steps: usize,
    episodes: Option<usize>,
    iterations: Option<usize>,
    loss: Option<(usize, &'static str)>,
    signal: Option<(usize, &'static str)>,
}

/// First present key wins; the label is what the TUI shows for it.
const LOSS_KEYS: &[(&str, &str)] = &[
    ("train/loss", "Loss"),
    ("train/critic_loss", "Critic loss"),
    ("train/value_loss", "Value loss"),
];
const SIGNAL_KEYS: &[(&str, &str)] = &[
    ("rollout/exploration_rate", "Exploration / epsilon"),
    ("train/ent_coef", "Entropy coefficient"),
    ("train/entropy_loss", "Entropy loss"),
];

impl RowParser {
    /// Detects the format from a header line, or `None` if it is unsupported.
    pub(crate) fn detect(header: &str) -> Option<Self> {
        let header = header.trim();
        let layout = if header == DQN_CSV_V1_HEADER {
            Layout::DqnV1
        } else {
            let names = split_fields(header);
            let find = |key: &str| names.iter().position(|name| name.trim() == key);
            if let (Some(reward), Some(length), Some(_)) = (find("r"), find("l"), find("t")) {
                Layout::Sb3Monitor {
                    fields: names.len(),
                    reward,
                    length,
                }
            } else if let Some(total_steps) = find("time/total_timesteps") {
                let first = |keys: &[(&str, &'static str)]| {
                    keys.iter()
                        .find_map(|(key, label)| find(key).map(|index| (index, *label)))
                };
                Layout::Sb3Progress(ProgressColumns {
                    fields: names.len(),
                    reward: find("rollout/ep_rew_mean"),
                    total_steps,
                    episodes: find("time/episodes"),
                    iterations: find("time/iterations"),
                    loss: first(LOSS_KEYS),
                    signal: first(SIGNAL_KEYS),
                })
            } else {
                return None;
            }
        };
        Some(Self {
            layout,
            rows: 0,
            steps: 0,
        })
    }

    pub(crate) fn format(&self) -> CsvFormat {
        match self.layout {
            Layout::DqnV1 => CsvFormat::RustforgeDqnV1,
            Layout::Sb3Monitor { .. } => CsvFormat::Sb3Monitor,
            Layout::Sb3Progress(_) => CsvFormat::Sb3Progress,
        }
    }

    pub(crate) fn labels(&self) -> MetricLabels {
        match &self.layout {
            Layout::DqnV1 => MetricLabels::dqn_monitor_defaults(),
            Layout::Sb3Monitor { .. } => MetricLabels {
                episode: "Episode".into(),
                episode_reward: "Episode reward".into(),
                primary_loss: None,
                policy_signal: None,
                throughput: "Steps/sec".into(),
            },
            Layout::Sb3Progress(columns) => MetricLabels {
                episode: if columns.episodes.is_some() {
                    "Episode".into()
                } else {
                    "Update".into()
                },
                episode_reward: "Mean episode reward".into(),
                primary_loss: columns.loss.map(|(_, label)| label.into()),
                policy_signal: columns.signal.map(|(_, label)| label.into()),
                throughput: "Steps/sec".into(),
            },
        }
    }

    pub(crate) fn parse(&mut self, line: &str) -> ParsedLine {
        let parsed = match &self.layout {
            Layout::DqnV1 => parse_dqn_v1(line),
            Layout::Sb3Monitor {
                fields,
                reward,
                length,
            } => {
                let values = split_fields(line);
                if values.len() != *fields {
                    return ParsedLine::Malformed("field count does not match the header");
                }
                let (Some(reward), Some(length)) = (
                    finite(&values[*reward]),
                    finite(&values[*length]).filter(|length| *length >= 0.0),
                ) else {
                    return ParsedLine::Malformed("invalid SB3 monitor row");
                };
                self.steps = self.steps.saturating_add(length as u64);
                ParsedLine::Row(MetricRow {
                    episode: self.rows,
                    reward: reward as f32,
                    primary_loss: None,
                    policy_signal: None,
                    global_step: self.steps,
                })
            }
            Layout::Sb3Progress(columns) => {
                let values = split_fields(line);
                if values.len() != columns.fields {
                    return ParsedLine::Malformed("field count does not match the header");
                }
                let Some(total_steps) =
                    finite(&values[columns.total_steps]).filter(|steps| *steps >= 0.0)
                else {
                    return ParsedLine::Malformed("invalid time/total_timesteps");
                };
                // Rows logged before the first episode finishes have no reward.
                let Some(reward) = columns.reward.and_then(|index| finite(&values[index])) else {
                    return ParsedLine::Skip;
                };
                let counter = |index: Option<usize>| {
                    index
                        .and_then(|index| finite(&values[index]))
                        .filter(|count| *count >= 1.0)
                        .map(|count| count as u64 - 1)
                };
                let optional =
                    |column: Option<(usize, &str)>| column.and_then(|(i, _)| finite(&values[i]));
                ParsedLine::Row(MetricRow {
                    episode: counter(columns.episodes)
                        .or_else(|| counter(columns.iterations))
                        .unwrap_or(self.rows),
                    reward: reward as f32,
                    primary_loss: optional(columns.loss).map(|value| value as f32),
                    policy_signal: optional(columns.signal).map(|value| value as f32),
                    global_step: total_steps as u64,
                })
            }
        };
        if matches!(parsed, ParsedLine::Row(_)) {
            self.rows += 1;
        }
        parsed
    }
}

fn parse_dqn_v1(line: &str) -> ParsedLine {
    if line.split(',').count() != 5 {
        return ParsedLine::Malformed("expected exactly five CSV fields");
    }
    match parse_line(line) {
        Some(row) if row.reward.is_finite() => ParsedLine::Row(row),
        Some(_) => ParsedLine::Malformed("reward must be finite"),
        None => ParsedLine::Malformed("invalid DQN CSV v1 metric row"),
    }
}

fn finite(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

/// Splits one CSV line, honoring double-quoted fields and `""` escapes.
fn split_fields(line: &str) -> Vec<Cow<'_, str>> {
    if !line.contains('"') {
        return line.split(',').map(Cow::Borrowed).collect();
    }
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(Cow::Owned(std::mem::take(&mut field))),
            _ => field.push(character),
        }
    }
    fields.push(Cow::Owned(field));
    fields
}

/// Reads `env_id` from an SB3 monitor metadata line such as
/// `#{"t_start": 1712345678.9, "env_id": "CartPole-v1"}`.
pub(crate) fn sb3_monitor_env_id(comment: &str) -> Option<String> {
    let json = comment.strip_prefix('#')?;
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    let env_id = value.get("env_id")?.as_str()?.trim();
    (!env_id.is_empty()).then(|| env_id.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(parser: &mut RowParser, lines: &[&str]) -> Vec<ParsedLine> {
        lines.iter().map(|line| parser.parse(line)).collect()
    }

    fn row(parsed: &ParsedLine) -> &MetricRow {
        match parsed {
            ParsedLine::Row(row) => row,
            other => panic!("expected a row, got {other:?}"),
        }
    }

    #[test]
    fn detects_each_supported_header() {
        let detect = |header| RowParser::detect(header).map(|parser| parser.format());
        assert_eq!(detect(DQN_CSV_V1_HEADER), Some(CsvFormat::RustforgeDqnV1));
        assert_eq!(detect("r,l,t"), Some(CsvFormat::Sb3Monitor));
        assert_eq!(detect("r,l,t,is_success"), Some(CsvFormat::Sb3Monitor));
        assert_eq!(
            detect("rollout/ep_rew_mean,time/total_timesteps"),
            Some(CsvFormat::Sb3Progress)
        );
        assert_eq!(detect("episode,reward,epsilon"), None);
        assert_eq!(detect("r,l"), None);
    }

    #[test]
    fn sb3_monitor_rows_count_episodes_and_accumulate_steps() {
        let mut parser = RowParser::detect("r,l,t").unwrap();
        let parsed = rows(&mut parser, &["18.0,18,5.08", "-3.5,40,5.20"]);
        assert_eq!(
            row(&parsed[0]),
            &MetricRow {
                episode: 0,
                reward: 18.0,
                primary_loss: None,
                policy_signal: None,
                global_step: 18,
            }
        );
        assert_eq!(row(&parsed[1]).episode, 1);
        assert_eq!(row(&parsed[1]).reward, -3.5);
        assert_eq!(row(&parsed[1]).global_step, 58);
    }

    #[test]
    fn sb3_monitor_rejects_bad_rows_without_advancing() {
        let mut parser = RowParser::detect("r,l,t").unwrap();
        let parsed = rows(&mut parser, &["18.0,18", "nan,18,1", "1,-2,1", "5,10,1"]);
        assert!(matches!(parsed[0], ParsedLine::Malformed(_)));
        assert!(matches!(parsed[1], ParsedLine::Malformed(_)));
        assert!(matches!(parsed[2], ParsedLine::Malformed(_)));
        assert_eq!(row(&parsed[3]).episode, 0);
        assert_eq!(row(&parsed[3]).global_step, 10);
    }

    #[test]
    fn sb3_monitor_handles_quoted_info_columns() {
        let mut parser = RowParser::detect("r,l,t,note").unwrap();
        let parsed = parser.parse(r#"2.5,7,0.1,"left, then ""right""""#);
        assert_eq!(row(&parsed).reward, 2.5);
    }

    #[test]
    fn sb3_dqn_progress_maps_episodes_loss_and_exploration() {
        let header = "rollout/ep_rew_mean,time/total_timesteps,rollout/ep_len_mean,time/fps,\
                      time/time_elapsed,rollout/exploration_rate,time/episodes,train/n_updates,\
                      train/loss,train/learning_rate";
        let mut parser = RowParser::detect(header).unwrap();
        let labels = parser.labels();
        assert_eq!(labels.episode, "Episode");
        assert_eq!(labels.primary_loss.as_deref(), Some("Loss"));
        assert_eq!(
            labels.policy_signal.as_deref(),
            Some("Exploration / epsilon")
        );

        let parsed = rows(
            &mut parser,
            &[
                "18.0,18,18.0,956,0,0.943,1,,,",
                "22.5,1200,22.5,900,1,0.05,53,50,0.125,0.0001",
            ],
        );
        assert_eq!(
            row(&parsed[0]),
            &MetricRow {
                episode: 0,
                reward: 18.0,
                primary_loss: None,
                policy_signal: Some(0.943),
                global_step: 18,
            }
        );
        assert_eq!(row(&parsed[1]).episode, 52);
        assert_eq!(row(&parsed[1]).primary_loss, Some(0.125));
        assert_eq!(row(&parsed[1]).global_step, 1200);
    }

    #[test]
    fn sb3_ppo_progress_counts_updates() {
        let header = "rollout/ep_rew_mean,time/total_timesteps,time/iterations,\
                      train/entropy_loss,train/loss,train/value_loss";
        let mut parser = RowParser::detect(header).unwrap();
        let labels = parser.labels();
        assert_eq!(labels.episode, "Update");
        assert_eq!(labels.primary_loss.as_deref(), Some("Loss"));
        assert_eq!(labels.policy_signal.as_deref(), Some("Entropy loss"));

        let parsed = rows(
            &mut parser,
            &["25.5,512,1,,,", "25.6,1024,2,-0.68,47.7,99.3"],
        );
        assert_eq!(row(&parsed[0]).episode, 0);
        assert_eq!(row(&parsed[1]).episode, 1);
        assert_eq!(row(&parsed[1]).policy_signal, Some(-0.68));
    }

    #[test]
    fn sb3_sac_progress_prefers_critic_loss_and_entropy_coefficient() {
        let header = "rollout/ep_rew_mean,time/total_timesteps,time/episodes,\
                      train/actor_loss,train/critic_loss,train/ent_coef";
        let labels = RowParser::detect(header).unwrap().labels();
        assert_eq!(labels.primary_loss.as_deref(), Some("Critic loss"));
        assert_eq!(labels.policy_signal.as_deref(), Some("Entropy coefficient"));
    }

    #[test]
    fn sb3_progress_skips_rows_without_a_reward_yet() {
        let mut parser = RowParser::detect("time/total_timesteps,time/fps").unwrap();
        assert_eq!(parser.parse("512,900"), ParsedLine::Skip);

        let mut parser = RowParser::detect("rollout/ep_rew_mean,time/total_timesteps").unwrap();
        assert_eq!(parser.parse(",512"), ParsedLine::Skip);
        assert!(matches!(parser.parse("1.0,"), ParsedLine::Malformed(_)));
        assert!(matches!(parser.parse("1.0"), ParsedLine::Malformed(_)));
        assert_eq!(row(&parser.parse("1.0,1024")).episode, 0);
    }

    #[test]
    fn dqn_v1_rows_keep_their_messages() {
        let mut parser = RowParser::detect(DQN_CSV_V1_HEADER).unwrap();
        assert_eq!(
            parser.parse("1,2,0.4,0.9,20,extra"),
            ParsedLine::Malformed("expected exactly five CSV fields")
        );
        assert_eq!(
            parser.parse("2,NaN,0.3,0.8,30"),
            ParsedLine::Malformed("reward must be finite")
        );
        assert_eq!(row(&parser.parse("0,1,0.5,1,10")).global_step, 10);
    }

    #[test]
    fn reads_env_id_from_sb3_metadata() {
        assert_eq!(
            sb3_monitor_env_id(r#"#{"t_start": 1790491100.4, "env_id": "CartPole-v1"}"#).as_deref(),
            Some("CartPole-v1")
        );
        assert_eq!(sb3_monitor_env_id(r#"#{"t_start": 1.0}"#), None);
        assert_eq!(sb3_monitor_env_id("# not json"), None);
    }
}
