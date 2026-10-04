//! PPO Continuous adapter for the generic live-training runtime.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use smallvec::{smallvec, SmallVec};

use super::live_runtime::{
    progress_scalars, throughput, LiveHooks, RewardWindow, StepDecision, StepPosition,
};
use super::on_policy_runtime::{derive_seed, finite_values, EpisodeBoundary};
use super::ppo_continuous_backend::PpoContinuousBackend;
use super::{PPOConfig, PPOContinuousConfig, PpoRuntimeOptions};
use crate::buffer::ContinuousRolloutBuffer;
use crate::env::{Environment, IntoTensorBuffer};
use crate::runtime::event::MetricValue;
use crate::runtime::progress::ProgressScalar;
use crate::runtime::trainer::{
    MetricDescriptor, MetricId, MetricKind, MetricRole, StopReason, Trainer, TrainerCapabilities,
    TrainerContext, TrainerError, TrainerMetadata, TrainingSummary,
};

const REWARD_EPISODE: MetricId = MetricId::new(101);
const REWARD_MOVING_AVERAGE: MetricId = MetricId::new(102);
const LOSS_POLICY: MetricId = MetricId::new(103);
const LOSS_VALUE: MetricId = MetricId::new(104);
const ROLLOUT_SIZE: MetricId = MetricId::new(106);
const STEPS_PER_SECOND: MetricId = MetricId::new(107);

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(0);

/// Returns the continuous PPO integration profile for Pendulum.
pub fn pendulum_ppo_config() -> PPOContinuousConfig {
    PPOContinuousConfig {
        base: PPOConfig {
            obs_dim: 3,
            lr: 1e-3,
            ..PPOConfig::default()
        },
        act_dim: 1,
        action_low: vec![-2.],
        action_high: vec![2.],
    }
}

pub struct PpoContinuousTrainerAdapter<E, F> {
    env: E,
    action: F,
    config: PPOContinuousConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    environment: String,
    seed: Option<u64>,
    run_id: String,
    options: PpoRuntimeOptions,
}

impl<E, F> PpoContinuousTrainerAdapter<E, F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        env: E,
        action: F,
        config: PPOContinuousConfig,
        episodes: usize,
        max_steps_per_episode: usize,
        environment: impl Into<String>,
        seed: Option<u64>,
    ) -> Self {
        let sequence = NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            env,
            action,
            config,
            episodes,
            max_steps_per_episode,
            environment: environment.into(),
            seed,
            run_id: format!("ppo-continuous-{sequence}"),
            options: PpoRuntimeOptions::default(),
        }
    }
    /// Construction and checkpoint I/O occur inside `run`, on the owning worker.
    pub fn with_options(mut self, options: PpoRuntimeOptions) -> Self {
        self.options = options;
        self
    }
}

impl<E, F> Trainer for PpoContinuousTrainerAdapter<E, F>
where
    E: Environment + Send + 'static,
    F: FnMut(&[f32]) -> Result<E::Act, TrainerError> + Send + 'static,
{
    fn metadata(&self) -> TrainerMetadata {
        TrainerMetadata {
            algorithm: "ppo-continuous".into(),
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
        let result = train_ppo_continuous_core(
            self.env,
            self.action,
            self.config,
            self.episodes,
            self.max_steps_per_episode,
            self.seed,
            &self.options,
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
    policy_loss: f32,
    value_loss: f32,
}

#[allow(clippy::too_many_arguments)]
fn train_ppo_continuous_core<E, F>(
    mut env: E,
    mut action: F,
    config: PPOContinuousConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    seed: Option<u64>,
    options: &PpoRuntimeOptions,
    hooks: &mut LiveHooks,
) -> Result<TrainingSummary, TrainerError>
where
    E: Environment,
    F: FnMut(&[f32]) -> Result<E::Act, TrainerError>,
{
    let started = Instant::now();
    if episodes > 0 && max_steps_per_episode == 0 {
        return Err(TrainerError {
            message: "max_steps_per_episode must be greater than zero when episodes is positive"
                .into(),
        });
    }
    let obs_dim = E::Obs::DIM;
    options.validate()?;
    let (environment_seed, mut action_rng, mut shuffle_rng) = match seed {
        Some(base_seed) => (
            Some(derive_seed(base_seed, 1)),
            Some(StdRng::seed_from_u64(derive_seed(base_seed, 2))),
            Some(StdRng::seed_from_u64(derive_seed(base_seed, 3))),
        ),
        None => (None, None, None),
    };
    let mut agent = PpoContinuousBackend::new(config, seed.map(|s| derive_seed(s, 0)), options)?;
    if agent.config().base.obs_dim != obs_dim
        || env.action_space()
            != crate::env::Space::continuous(
                agent.config().action_low.clone(),
                agent.config().action_high.clone(),
            )
    {
        return Err(TrainerError {
            message: "PPO agent dimensions do not match the environment".into(),
        });
    }
    let gamma = agent.config().base.gamma;
    let gae_lambda = agent.config().base.gae_lambda;
    let act_dim = agent.config().act_dim;
    if max_steps_per_episode.checked_mul(obs_dim).is_none()
        || max_steps_per_episode.checked_mul(act_dim).is_none()
    {
        return Err(TrainerError {
            message: "continuous rollout capacity overflow".into(),
        });
    }
    let mut rollout = ContinuousRolloutBuffer::new(max_steps_per_episode, obs_dim, act_dim);
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
        if state_buf.iter().any(|v| !v.is_finite()) {
            return Err(TrainerError {
                message: "continuous PPO observation must be finite".into(),
            });
        }
        let mut episode_reward = 0.0;
        let mut episode_length = 0usize;
        let mut last_value = 0.0;
        let mut graceful_requested = false;
        let mut force_after_episode = false;

        for step_index in 0..max_steps_per_episode {
            let (action_vector, old_log_probability, value) = match action_rng.as_mut() {
                Some(rng) => agent.select_action_with_rng(&state_buf, rng),
                None => agent.select_action(&state_buf),
            }?;
            if !old_log_probability.is_finite()
                || !value.is_finite()
                || action_vector.len() != act_dim
                || action_vector.iter().enumerate().any(|(i, a)| {
                    !a.is_finite()
                        || *a < agent.config().action_low[i]
                        || *a > agent.config().action_high[i]
                })
            {
                return Err(TrainerError {
                    message:
                        "continuous PPO action/value must be finite and within configured bounds"
                            .into(),
                });
            }
            let converted_action = action(&action_vector)?;
            let step_limit = step_index + 1 == max_steps_per_episode;
            let (next_state, reward, terminated, truncated, _) = env.step(converted_action);
            let boundary = EpisodeBoundary::classify(terminated, truncated, step_limit);
            let mut next_state_buf = vec![0.0; obs_dim];
            next_state.write_to_buffer(&mut next_state_buf);
            rollout.push(
                &state_buf,
                &action_vector,
                reward,
                value,
                boundary.done_mask(),
                old_log_probability,
            );
            if !reward.is_finite() || next_state_buf.iter().any(|v| !v.is_finite()) {
                return Err(TrainerError {
                    message: "continuous PPO environment returned nonfinite values".into(),
                });
            }
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
            if boundary != EpisodeBoundary::None {
                last_value = if boundary == EpisodeBoundary::Terminated {
                    0.
                } else {
                    agent.value_of(&state_buf)?
                };
                break;
            }
        }

        if rollout.is_empty() {
            continue;
        }
        rollout.compute_returns_and_advantages(gamma, gae_lambda, last_value);
        let mut batch = crate::buffer::ContinuousRolloutBatch::new(rollout.len(), obs_dim, act_dim);
        rollout.fill_batch(&mut batch);
        let losses = match shuffle_rng.as_mut() {
            Some(rng) => agent.train_on_batch_with_rng(&batch, rng),
            None => agent.train_on_batch(&batch),
        }?;
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
            policy_loss: losses.0,
            value_loss: losses.1,
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

    agent.save(options)?;
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
            LOSS_POLICY,
            "loss.policy",
            "PPO policy loss",
            None,
            MetricKind::Gauge,
        ),
        metric(
            LOSS_VALUE,
            "loss.value",
            "PPO value loss",
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
            "loss.policy" => Some(MetricRole::PrimaryLoss),
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
            value: f64::from(state.episode_reward),
        },
        ProgressScalar {
            metric: ROLLOUT_SIZE,
            value: state.rollout_size as f64,
        },
        ProgressScalar {
            metric: STEPS_PER_SECOND,
            value: throughput(state.global_step, state.elapsed),
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
            metric: LOSS_POLICY,
            value: f64::from(state.policy_loss)
        },
        MetricValue {
            metric: LOSS_VALUE,
            value: f64::from(state.value_loss)
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
