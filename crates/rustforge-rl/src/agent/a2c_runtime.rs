//! A2C adapter for the generic live-training runtime.

use rustforge_autograd::no_grad;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use smallvec::{smallvec, SmallVec};

use super::live_runtime::{
    progress_scalars, throughput, LiveHooks, RewardWindow, StepDecision, StepPosition,
};
use super::on_policy_runtime::{derive_seed, finite_values, EpisodeBoundary};
use super::{A2CConfig, A2C};
use crate::buffer::RolloutBuffer;
use crate::env::{Environment, IntoTensorBuffer};
use crate::runtime::event::MetricValue;
use crate::runtime::progress::ProgressScalar;
use crate::runtime::trainer::{
    MetricDescriptor, MetricId, MetricKind, MetricRole, StopReason, Trainer, TrainerCapabilities,
    TrainerContext, TrainerError, TrainerMetadata, TrainingSummary,
};

const REWARD_EPISODE: MetricId = MetricId::new(201);
const REWARD_MOVING_AVERAGE: MetricId = MetricId::new(202);
const LOSS_TOTAL: MetricId = MetricId::new(203);
const LOSS_ACTOR: MetricId = MetricId::new(204);
const LOSS_CRITIC: MetricId = MetricId::new(205);
const POLICY_ENTROPY: MetricId = MetricId::new(206);
const ROLLOUT_SIZE: MetricId = MetricId::new(207);
const STEPS_PER_SECOND: MetricId = MetricId::new(208);

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(0);

/// Returns the approved A2C integration profile for CartPole.
pub fn cartpole_a2c_config() -> A2CConfig {
    A2CConfig {
        obs_dim: 4,
        num_actions: 2,
        ..A2CConfig::default()
    }
}

pub struct A2cTrainerAdapter<E> {
    env: E,
    config: A2CConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    environment: String,
    seed: Option<u64>,
    run_id: String,
}

impl<E> A2cTrainerAdapter<E> {
    pub fn new(
        env: E,
        config: A2CConfig,
        episodes: usize,
        max_steps_per_episode: usize,
        environment: impl Into<String>,
        seed: Option<u64>,
    ) -> Self {
        let sequence = NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            env,
            config,
            episodes,
            max_steps_per_episode,
            environment: environment.into(),
            seed,
            run_id: format!("a2c-{sequence}"),
        }
    }
}

impl<E> Trainer for A2cTrainerAdapter<E>
where
    E: Environment + Send + 'static,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    fn metadata(&self) -> TrainerMetadata {
        TrainerMetadata {
            algorithm: "a2c".into(),
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
        let result = train_a2c_core(
            self.env,
            self.config,
            self.episodes,
            self.max_steps_per_episode,
            self.seed,
            &mut hooks,
        );
        hooks.flush_metrics();
        result
    }
}

#[derive(Clone, Copy)]
struct StepState {
    global_step: u64,
    episode: u64,
    episode_step: u64,
    episode_reward: f32,
    rollout_size: usize,
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

#[derive(Clone, Copy)]
struct EpisodeState {
    step: StepState,
    moving_average: f32,
    total_loss: f32,
    actor_loss: f32,
    critic_loss: f32,
    entropy: f32,
}

fn train_a2c_core<E>(
    mut env: E,
    config: A2CConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    seed: Option<u64>,
    hooks: &mut LiveHooks,
) -> Result<TrainingSummary, TrainerError>
where
    E: Environment,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    let started = Instant::now();
    if episodes > 0 && max_steps_per_episode == 0 {
        return Err(TrainerError {
            message: "max_steps_per_episode must be greater than zero when episodes is positive"
                .into(),
        });
    }
    let obs_dim = E::Obs::DIM;
    let gamma = config.gamma;
    let lambda = config.lambda;
    let (mut agent, environment_seed, mut action_rng) = match seed {
        Some(base_seed) => (
            A2C::new_seeded(config, derive_seed(base_seed, 0)),
            Some(derive_seed(base_seed, 1)),
            Some(StdRng::seed_from_u64(derive_seed(base_seed, 2))),
        ),
        None => (A2C::new(config), None, None),
    };
    let mut rollout = RolloutBuffer::new(max_steps_per_episode, obs_dim);
    let mut global_step = 0usize;
    let mut completed_episodes = 0usize;
    let mut rewards_window = RewardWindow::new(100);
    let mut stop_reason = StopReason::Completed;
    hooks.publish_started();

    'episodes: for episode in 0..episodes {
        rollout.clear();
        let episode_seed = if episode == 0 { environment_seed } else { None };
        let (state, _) = env.reset(episode_seed);
        let mut state_buf = vec![0.0; obs_dim];
        state.write_to_buffer(&mut state_buf);
        let mut episode_reward = 0.0;
        let mut episode_length = 0usize;
        let mut last_value = 0.0;
        let mut graceful_requested = false;
        let mut force_after_episode = false;

        for step_index in 0..max_steps_per_episode {
            let (logits, value) = no_grad(|| agent.forward(&state_buf));
            let action_index = match action_rng.as_mut() {
                Some(rng) => A2C::sample_action(&logits.data(), rng),
                None => A2C::sample_action_default(&logits.data()),
            };
            let action = E::Act::try_from(action_index).map_err(|error| TrainerError {
                message: format!(
                    "A2C action index {action_index} was rejected by the environment: {error:?}"
                ),
            })?;
            let step_limit = step_index + 1 == max_steps_per_episode;
            let (next_state, reward, terminated, truncated, _) = env.step(action);
            let boundary = EpisodeBoundary::classify(terminated, truncated, step_limit);
            let mut next_state_buf = vec![0.0; obs_dim];
            next_state.write_to_buffer(&mut next_state_buf);
            rollout.push(
                &state_buf,
                action_index,
                reward,
                value.data().item(),
                boundary.done_mask(),
            );
            state_buf = next_state_buf;
            episode_reward += reward;
            episode_length += 1;
            global_step += 1;
            let step_state = StepState {
                global_step: global_step as u64,
                episode: episode as u64,
                episode_step: episode_length as u64,
                episode_reward,
                rollout_size: rollout.len(),
                elapsed: started.elapsed(),
            };

            match hooks.observe_controls(step_state.position(), step_scalars(step_state)) {
                StepDecision::Continue => {}
                StepDecision::GracefulStop => graceful_requested = true,
                StepDecision::ForceStop => {
                    if boundary != EpisodeBoundary::None {
                        force_after_episode = true;
                    } else {
                        stop_reason = StopReason::ForceStop;
                        break 'episodes;
                    }
                }
            }
            if let Some(bootstrap_value) = boundary.bootstrap_value(|| agent.value_of(&state_buf)) {
                last_value = bootstrap_value;
                break;
            }
        }

        if rollout.is_empty() {
            continue;
        }
        rollout.compute_returns_and_advantages(gamma, lambda, last_value);
        let losses = agent.train_on_rollout(&rollout.to_batch());
        let completed_state = StepState {
            global_step: global_step as u64,
            episode: episode as u64,
            episode_step: episode_length as u64,
            episode_reward,
            rollout_size: rollout.len(),
            elapsed: started.elapsed(),
        };
        match hooks.observe_controls(completed_state.position(), step_scalars(completed_state)) {
            StepDecision::Continue => {}
            StepDecision::GracefulStop => graceful_requested = true,
            StepDecision::ForceStop => force_after_episode = true,
        }
        let moving_average = rewards_window.push(episode_reward);
        let values = episode_values(EpisodeState {
            step: completed_state,
            moving_average,
            total_loss: losses.0,
            actor_loss: losses.1,
            critic_loss: losses.2,
            entropy: losses.3,
        });
        hooks.publish_episode(completed_state.position(), values.clone());
        hooks.publish_progress(completed_state.position(), progress_scalars(&values));
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

    Ok(TrainingSummary::stopped(
        global_step as u64,
        completed_episodes as u64,
        started.elapsed(),
        stop_reason,
    ))
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
            "Reward average over the latest 100 episodes",
            None,
            MetricKind::Gauge,
        ),
        metric(
            LOSS_TOTAL,
            "loss.total",
            "A2C total loss",
            None,
            MetricKind::Gauge,
        ),
        metric(
            LOSS_ACTOR,
            "loss.actor",
            "A2C actor loss",
            None,
            MetricKind::Gauge,
        ),
        metric(
            LOSS_CRITIC,
            "loss.critic",
            "A2C critic loss",
            None,
            MetricKind::Gauge,
        ),
        metric(
            POLICY_ENTROPY,
            "policy.entropy",
            "A2C policy entropy",
            None,
            MetricKind::Gauge,
        ),
        metric(
            ROLLOUT_SIZE,
            "rollout.size",
            "Completed rollout length",
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
            "loss.actor" => Some(MetricRole::PrimaryLoss),
            "policy.entropy" => Some(MetricRole::PolicySignal),
            "performance.steps_per_second" => Some(MetricRole::Throughput),
            _ => None,
        },
    }
}

fn step_scalars(state: StepState) -> SmallVec<[ProgressScalar; 8]> {
    smallvec![
        ProgressScalar {
            metric: REWARD_EPISODE,
            value: f64::from(state.episode_reward)
        },
        ProgressScalar {
            metric: ROLLOUT_SIZE,
            value: state.rollout_size as f64
        },
        ProgressScalar {
            metric: STEPS_PER_SECOND,
            value: throughput(state.global_step, state.elapsed)
        },
    ]
}

fn episode_values(state: EpisodeState) -> SmallVec<[MetricValue; 8]> {
    finite_values(smallvec![
        MetricValue {
            metric: REWARD_EPISODE,
            value: f64::from(state.step.episode_reward)
        },
        MetricValue {
            metric: REWARD_MOVING_AVERAGE,
            value: f64::from(state.moving_average)
        },
        MetricValue {
            metric: LOSS_TOTAL,
            value: f64::from(state.total_loss)
        },
        MetricValue {
            metric: LOSS_ACTOR,
            value: f64::from(state.actor_loss)
        },
        MetricValue {
            metric: LOSS_CRITIC,
            value: f64::from(state.critic_loss)
        },
        MetricValue {
            metric: POLICY_ENTROPY,
            value: f64::from(state.entropy)
        },
        MetricValue {
            metric: ROLLOUT_SIZE,
            value: state.step.rollout_size as f64
        },
        MetricValue {
            metric: STEPS_PER_SECOND,
            value: throughput(state.step.global_step, state.step.elapsed)
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_losses_are_omitted_without_dropping_the_reward() {
        let values = episode_values(EpisodeState {
            step: StepState {
                global_step: 10,
                episode: 0,
                episode_step: 10,
                episode_reward: 12.0,
                rollout_size: 10,
                elapsed: Duration::from_secs(1),
            },
            moving_average: 12.0,
            total_loss: f32::NAN,
            actor_loss: f32::NEG_INFINITY,
            critic_loss: 0.5,
            entropy: 0.1,
        });
        assert!(values.iter().all(|value| value.value.is_finite()));
        assert!(values
            .iter()
            .any(|value| value.metric == REWARD_EPISODE && value.value == 12.0));
        assert!(!values.iter().any(|value| value.metric == LOSS_TOTAL));
        assert!(!values.iter().any(|value| value.metric == LOSS_ACTOR));
    }
}
