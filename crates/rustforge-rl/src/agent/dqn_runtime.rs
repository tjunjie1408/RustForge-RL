//! DQN adapter for the generic live-training runtime and its shared core loop.

use std::convert::TryFrom;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rustforge_autograd::no_grad;
use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_tensor::Tensor;
use smallvec::{smallvec, SmallVec};

use super::live_runtime::{throughput, LiveHooks, RewardWindow, StepDecision, StepPosition};
use super::{DQNConfig, EpsilonGreedy, DQN};
use crate::buffer::{PrioritizedReplayBuffer, ReplayBuffer, TransitionBatch};
use crate::env::{Environment, IntoTensorBuffer};
use crate::metrics::{AgentLogger, CsvLogger, EpisodeMetrics};
use crate::runtime::event::MetricValue;
use crate::runtime::progress::ProgressScalar;
use crate::runtime::trainer::{
    MetricDescriptor, MetricId, MetricKind, MetricRole, StopReason, Trainer, TrainerCapabilities,
    TrainerContext, TrainerError, TrainerMetadata, TrainingSummary,
};
use crate::training::{episode_done, replay_done};

pub const REWARD_EPISODE: MetricId = MetricId::new(1);
pub const REWARD_MOVING_AVERAGE: MetricId = MetricId::new(2);
pub const LOSS_TD: MetricId = MetricId::new(3);
pub const EXPLORATION_EPSILON: MetricId = MetricId::new(4);
pub const REPLAY_BUFFER_SIZE: MetricId = MetricId::new(5);
pub const STEPS_PER_SECOND: MetricId = MetricId::new(6);

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(0);

pub struct DqnTrainerAdapter<E> {
    env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    environment: String,
    run_id: String,
}

impl<E> DqnTrainerAdapter<E> {
    pub fn new(
        env: E,
        config: DQNConfig,
        episodes: usize,
        max_steps_per_episode: usize,
        environment: impl Into<String>,
    ) -> Self {
        let sequence = NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            env,
            config,
            episodes,
            max_steps_per_episode,
            environment: environment.into(),
            run_id: format!("dqn-{sequence}"),
        }
    }
}

impl<E> Trainer for DqnTrainerAdapter<E>
where
    E: Environment + Send + 'static,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    fn metadata(&self) -> TrainerMetadata {
        TrainerMetadata {
            algorithm: "dqn".into(),
            environment: self.environment.clone(),
            run_id: self.run_id.clone(),
            capabilities: TrainerCapabilities {
                pause_resume: true,
                graceful_stop: true,
                force_stop: true,
                checkpoint: false,
            },
            metrics: metric_descriptors(),
        }
    }

    fn run(self: Box<Self>, context: TrainerContext) -> Result<TrainingSummary, TrainerError> {
        let metadata = self.metadata();
        let mut hooks = LiveHooks::new(context, metadata);
        let result = train_dqn_core(
            self.env,
            self.config,
            self.episodes,
            self.max_steps_per_episode,
            &mut hooks,
        );
        hooks.flush();
        result.map(|result| result.summary)
    }
}

pub(crate) fn train_dqn_headless<E>(
    env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    log_path: Option<&str>,
) -> Result<DQN, TrainerError>
where
    E: Environment,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    let logger = log_path
        .map(|path| {
            CsvLogger::new(path).map_err(|error| TrainerError {
                message: format!("failed to create DQN CSV log at {path}: {error}"),
            })
        })
        .transpose()?;
    let mut hooks = HeadlessHooks { logger };
    let result = train_dqn_core(env, config, episodes, max_steps_per_episode, &mut hooks);
    hooks.flush();
    result.map(|result| result.agent)
}

struct DqnRunResult {
    agent: DQN,
    summary: TrainingSummary,
}

#[derive(Clone, Copy)]
struct StepState {
    global_step: u64,
    episode: u64,
    episode_step: u64,
    episode_reward: f32,
    latest_loss: Option<f32>,
    epsilon: f32,
    replay_size: usize,
    elapsed: Duration,
}

impl StepState {
    fn position(&self) -> StepPosition {
        StepPosition {
            global_step: self.global_step,
            episode: self.episode,
            episode_step: self.episode_step,
            elapsed: self.elapsed,
        }
    }
}

trait DqnHooks {
    fn started(&mut self) {}
    fn after_step(&mut self, _state: StepState) -> StepDecision {
        StepDecision::Continue
    }
    fn episode_completed(
        &mut self,
        _metrics: &EpisodeMetrics,
        _rolling_average: f32,
        _episode_length: u64,
        _replay_size: usize,
        _elapsed: Duration,
    ) {
    }
    fn flush(&mut self) {}
}

struct HeadlessHooks {
    logger: Option<CsvLogger>,
}

impl DqnHooks for HeadlessHooks {
    fn episode_completed(
        &mut self,
        metrics: &EpisodeMetrics,
        rolling_average: f32,
        _episode_length: u64,
        _replay_size: usize,
        _elapsed: Duration,
    ) {
        println!(
            "Episode {:4} | Reward: {:6.1} | Rolling: {:6.1} | Epsilon: {:.3} | Loss: {:.4}",
            metrics.episode,
            metrics.reward,
            rolling_average,
            metrics.epsilon,
            if metrics.avg_loss.is_nan() {
                0.0
            } else {
                metrics.avg_loss
            }
        );
        if let Some(logger) = &self.logger {
            logger.log(metrics);
        }
    }

    fn flush(&mut self) {
        if let Some(logger) = &self.logger {
            logger.flush();
        }
    }
}

impl DqnHooks for LiveHooks {
    fn started(&mut self) {
        self.publish_started();
    }

    fn after_step(&mut self, state: StepState) -> StepDecision {
        self.observe_controls(state.position(), step_scalars(state))
    }

    fn episode_completed(
        &mut self,
        metrics: &EpisodeMetrics,
        rolling_average: f32,
        episode_length: u64,
        replay_size: usize,
        elapsed: Duration,
    ) {
        let values = episode_values(metrics, rolling_average, replay_size, elapsed);
        let position = StepPosition {
            global_step: metrics.global_step as u64,
            episode: metrics.episode as u64,
            episode_step: episode_length,
            elapsed,
        };
        self.publish_episode(position, values);
    }

    fn flush(&mut self) {
        self.flush_metrics();
    }
}

fn step_scalars(state: StepState) -> SmallVec<[ProgressScalar; 8]> {
    let mut scalars = smallvec![
        ProgressScalar {
            metric: REWARD_EPISODE,
            value: f64::from(state.episode_reward),
        },
        ProgressScalar {
            metric: EXPLORATION_EPSILON,
            value: f64::from(state.epsilon),
        },
        ProgressScalar {
            metric: REPLAY_BUFFER_SIZE,
            value: state.replay_size as f64,
        },
        ProgressScalar {
            metric: STEPS_PER_SECOND,
            value: throughput(state.global_step, state.elapsed),
        },
    ];
    if let Some(loss) = state.latest_loss.filter(|loss| loss.is_finite()) {
        scalars.push(ProgressScalar {
            metric: LOSS_TD,
            value: f64::from(loss),
        });
    }
    scalars
}

fn train_dqn_core<E, H>(
    mut env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    hooks: &mut H,
) -> Result<DqnRunResult, TrainerError>
where
    E: Environment,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
    H: DqnHooks,
{
    let started = Instant::now();
    let obs_dim = E::Obs::DIM;
    let num_actions = config.num_actions;
    let batch_size = 32usize;
    let warmup_steps = 128usize;
    let mut agent = DQN::new(config);
    let mut explorer = EpsilonGreedy::new(1.0, 0.05, 2_000);
    let use_per = agent.config().use_per;
    let mut replay = ReplayBuffer::new(10_000, obs_dim);
    let mut batch = TransitionBatch::new(batch_size, obs_dim);
    let mut per_replay = PrioritizedReplayBuffer::new(10_000, obs_dim, 0.6);
    let mut per_weights = Tensor::zeros(&[batch_size, 1]);
    let mut per_tree_indices = vec![0; batch_size];
    let mut global_step = 0usize;
    let mut completed_episodes = 0usize;
    let mut rewards_window = RewardWindow::new(100);
    let mut stop_reason = StopReason::Completed;
    hooks.started();

    'episodes: for episode in 0..episodes {
        let (state, _) = env.reset(Some(2026 + episode as u64));
        let mut state_buf = vec![0.0f32; obs_dim];
        state.write_to_buffer(&mut state_buf);
        let mut episode_reward = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0usize;
        let mut latest_loss = None;
        let mut episode_length = 0usize;
        let mut graceful_requested = false;
        let mut force_after_episode = false;

        for step_index in 0..max_steps_per_episode {
            let input = Tensor::from_vec(state_buf.clone(), &[1, obs_dim]);
            let output = no_grad(|| agent.q_net().forward(&Variable::from_tensor(input)));
            let q_values = output.data();
            ensure_finite_q_values(&q_values, episode, global_step)?;
            let action_idx = explorer.select_action(&q_values, global_step, num_actions);
            let env_action = E::Act::try_from(action_idx).map_err(|error| TrainerError {
                message: format!(
                    "DQN action index {action_idx} was rejected by the environment: {error:?}"
                ),
            })?;
            let (next_state, reward, terminated, truncated, _) = env.step(env_action);
            episode_reward += reward;
            episode_length += 1;
            let mut next_state_buf = vec![0.0f32; obs_dim];
            next_state.write_to_buffer(&mut next_state_buf);
            if use_per {
                per_replay.push(
                    &state_buf,
                    action_idx,
                    reward,
                    &next_state_buf,
                    replay_done(terminated, truncated),
                );
            } else {
                replay.push(
                    &state_buf,
                    action_idx,
                    reward,
                    &next_state_buf,
                    replay_done(terminated, truncated),
                );
            }
            state_buf = next_state_buf;
            let replay_size = if use_per {
                per_replay.len()
            } else {
                replay.len()
            };
            if replay_size >= warmup_steps {
                let loss = if use_per {
                    let beta_steps = agent.config().per_beta_annealing_steps as f32;
                    let beta = (0.4 + 0.6 * (global_step as f32 / beta_steps)).min(1.0);
                    per_replay.sample(
                        batch_size,
                        beta,
                        &mut batch,
                        &mut per_weights,
                        &mut per_tree_indices,
                    );
                    let (loss, td) = agent.train_step(&batch, Some(&per_weights));
                    if let Some(errors) = &td {
                        per_replay.update_priorities(&per_tree_indices[..batch.size], errors);
                    }
                    loss
                } else {
                    replay.sample(batch_size, &mut batch);
                    agent.train_step(&batch, None).0
                };
                latest_loss = Some(loss);
                if loss.is_finite() {
                    loss_sum += loss;
                    loss_count += 1;
                }
            }
            global_step += 1;
            let boundary =
                episode_done(terminated, truncated) || step_index + 1 == max_steps_per_episode;
            match hooks.after_step(StepState {
                global_step: global_step as u64,
                episode: episode as u64,
                episode_step: episode_length as u64,
                episode_reward,
                latest_loss,
                epsilon: explorer.epsilon(global_step),
                replay_size,
                elapsed: started.elapsed(),
            }) {
                StepDecision::Continue => {}
                StepDecision::GracefulStop => graceful_requested = true,
                StepDecision::ForceStop => {
                    if boundary {
                        force_after_episode = true;
                    } else {
                        stop_reason = StopReason::ForceStop;
                        break 'episodes;
                    }
                }
            }
            if boundary {
                break;
            }
        }

        let rolling_average = rewards_window.push(episode_reward);
        let metrics = EpisodeMetrics {
            episode,
            reward: episode_reward,
            avg_loss: if loss_count > 0 {
                loss_sum / loss_count as f32
            } else {
                f32::NAN
            },
            epsilon: explorer.epsilon(global_step),
            global_step,
        };
        hooks.episode_completed(
            &metrics,
            rolling_average,
            episode_length as u64,
            if use_per {
                per_replay.len()
            } else {
                replay.len()
            },
            started.elapsed(),
        );
        completed_episodes += 1;
        if force_after_episode {
            stop_reason = StopReason::ForceStop;
            break;
        }
        if graceful_requested {
            stop_reason = StopReason::GracefulStop;
            break;
        }
    }

    Ok(DqnRunResult {
        agent,
        summary: TrainingSummary::stopped(
            global_step as u64,
            completed_episodes as u64,
            started.elapsed(),
            stop_reason,
        ),
    })
}

/// Fails the run when the online network has diverged to NaN or infinite Q-values.
fn ensure_finite_q_values(
    q_values: &Tensor,
    episode: usize,
    global_step: usize,
) -> Result<(), TrainerError> {
    if q_values.data().iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TrainerError {
            message: format!(
                "DQN Q-values became non-finite at episode {episode}, global step {global_step};                  training diverged"
            ),
        })
    }
}

fn metric_descriptors() -> Vec<MetricDescriptor> {
    vec![
        metric(
            REWARD_EPISODE,
            "reward.episode",
            "Episode reward",
            None,
            MetricKind::Gauge,
        ),
        metric(
            REWARD_MOVING_AVERAGE,
            "reward.moving_average",
            "Moving average reward",
            None,
            MetricKind::Gauge,
        ),
        metric(LOSS_TD, "loss.td", "TD loss", None, MetricKind::Gauge),
        metric(
            EXPLORATION_EPSILON,
            "exploration.epsilon",
            "Epsilon",
            None,
            MetricKind::Gauge,
        ),
        metric(
            REPLAY_BUFFER_SIZE,
            "replay_buffer.size",
            "Replay buffer",
            Some("transitions"),
            MetricKind::Gauge,
        ),
        metric(
            STEPS_PER_SECOND,
            "performance.steps_per_second",
            "Steps per second",
            Some("steps/s"),
            MetricKind::Rate,
        ),
    ]
}

fn metric(
    id: MetricId,
    name: &str,
    label: &str,
    unit: Option<&str>,
    kind: MetricKind,
) -> MetricDescriptor {
    MetricDescriptor {
        id,
        name: name.into(),
        label: label.into(),
        unit: unit.map(str::to_owned),
        kind,
        role: match name {
            "reward.episode" => Some(MetricRole::EpisodeReward),
            "loss.td" => Some(MetricRole::PrimaryLoss),
            "exploration.epsilon" => Some(MetricRole::PolicySignal),
            "performance.steps_per_second" => Some(MetricRole::Throughput),
            _ => None,
        },
    }
}

fn episode_values(
    metrics: &EpisodeMetrics,
    rolling_average: f32,
    replay_size: usize,
    elapsed: Duration,
) -> SmallVec<[MetricValue; 8]> {
    let mut values = smallvec![
        MetricValue {
            metric: REWARD_EPISODE,
            value: f64::from(metrics.reward)
        },
        MetricValue {
            metric: REWARD_MOVING_AVERAGE,
            value: f64::from(rolling_average)
        },
        MetricValue {
            metric: EXPLORATION_EPSILON,
            value: f64::from(metrics.epsilon)
        },
        MetricValue {
            metric: REPLAY_BUFFER_SIZE,
            value: replay_size as f64
        },
        MetricValue {
            metric: STEPS_PER_SECOND,
            value: throughput(metrics.global_step as u64, elapsed),
        },
    ];
    if metrics.avg_loss.is_finite() {
        values.push(MetricValue {
            metric: LOSS_TD,
            value: f64::from(metrics.avg_loss),
        });
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_q_values_fail_the_run_with_context() {
        let finite = Tensor::from_vec(vec![0.5, -1.0], &[1, 2]);
        assert!(ensure_finite_q_values(&finite, 3, 40).is_ok());

        for bad in [f32::NAN, f32::INFINITY] {
            let diverged = Tensor::from_vec(vec![0.5, bad], &[1, 2]);
            let error = ensure_finite_q_values(&diverged, 3, 40).unwrap_err();
            assert!(error.message.contains("episode 3"));
            assert!(error.message.contains("global step 40"));
        }
    }
}
