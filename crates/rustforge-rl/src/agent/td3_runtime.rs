//! Continuous TD3 collection/replay loop for the shared worker and console runtime.
use super::{
    live_runtime::{
        progress_scalars, throughput, LiveHooks, RewardWindow, StepDecision, StepPosition,
    },
    on_policy_runtime::derive_seed,
    td3_backend::Td3Backend,
    TD3Config, Td3RuntimeOptions,
};
use crate::{
    buffer::{ContinuousReplayBuffer, ContinuousTransitionBatch},
    env::{Environment, IntoTensorBuffer, Space},
    runtime::{
        event::MetricValue,
        trainer::{
            MetricDescriptor, MetricId, MetricKind, MetricRole, StopReason, Trainer,
            TrainerCapabilities, TrainerContext, TrainerError, TrainerMetadata, TrainingSummary,
        },
    },
    training::{episode_done, replay_done},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use smallvec::{smallvec, SmallVec};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};
const REWARD: MetricId = MetricId::new(401);
const AVERAGE: MetricId = MetricId::new(402);
const CRITIC: MetricId = MetricId::new(403);
const ACTOR: MetricId = MetricId::new(404);
const REPLAY: MetricId = MetricId::new(405);
const SPEED: MetricId = MetricId::new(406);
static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(0);
pub fn pendulum_td3_config() -> TD3Config {
    let mut c = TD3Config::new(3, 1, vec![-2.], vec![2.]);
    c.hidden_dim = 64;
    c
}
pub struct Td3TrainerAdapter<E, F> {
    env: E,
    action: F,
    config: TD3Config,
    episodes: usize,
    max_steps: usize,
    environment: String,
    seed: Option<u64>,
    run_id: String,
    options: Td3RuntimeOptions,
}
impl<E, F> Td3TrainerAdapter<E, F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        env: E,
        action: F,
        config: TD3Config,
        episodes: usize,
        max_steps: usize,
        environment: impl Into<String>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            env,
            action,
            config,
            episodes,
            max_steps,
            environment: environment.into(),
            seed,
            run_id: format!("td3-{}", NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed) + 1),
            options: Td3RuntimeOptions::default(),
        }
    }
    /// GPU contexts, networks and checkpoint I/O are constructed inside `run`.
    pub fn with_options(mut self, options: Td3RuntimeOptions) -> Self {
        self.options = options;
        self
    }
}
impl<E, F> Trainer for Td3TrainerAdapter<E, F>
where
    E: Environment + Send + 'static,
    F: FnMut(&[f32]) -> Result<E::Act, TrainerError> + Send + 'static,
{
    fn metadata(&self) -> TrainerMetadata {
        TrainerMetadata {
            algorithm: "td3".into(),
            environment: self.environment.clone(),
            run_id: self.run_id.clone(),
            capabilities: TrainerCapabilities {
                pause_resume: true,
                graceful_stop: true,
                force_stop: true,
                checkpoint: false,
            },
            metrics: descriptors(),
        }
    }
    fn run(self: Box<Self>, context: TrainerContext) -> Result<TrainingSummary, TrainerError> {
        let mut hooks = LiveHooks::new(context, self.metadata());
        let result = train(
            self.env,
            self.action,
            self.config,
            self.episodes,
            self.max_steps,
            self.seed,
            &self.options,
            &mut hooks,
        );
        hooks.flush_metrics();
        result
    }
}
#[allow(clippy::too_many_arguments)]
fn train<E, F>(
    mut env: E,
    mut action: F,
    config: TD3Config,
    episodes: usize,
    max_steps: usize,
    seed: Option<u64>,
    options: &Td3RuntimeOptions,
    hooks: &mut LiveHooks,
) -> Result<TrainingSummary, TrainerError>
where
    E: Environment,
    F: FnMut(&[f32]) -> Result<E::Act, TrainerError>,
{
    options.validate()?;
    if episodes > 0 && max_steps == 0 {
        return Err(error("max_steps must be positive"));
    }
    let mut agent = Td3Backend::new(config, seed.map(|s| derive_seed(s, 0)), options)?;
    let c = agent.config();
    let obs = c.obs_dim;
    let act = c.act_dim;
    if obs != E::Obs::DIM
        || env.action_space() != Space::continuous(c.action_low.clone(), c.action_high.clone())
    {
        return Err(error(
            "TD3 agent dimensions/bounds do not match the environment",
        ));
    }
    for size in [options.replay_capacity, options.batch_size] {
        for width in [obs, act] {
            size.checked_mul(width)
                .ok_or_else(|| error("TD3 replay/batch capacity overflow"))?;
        }
    }
    let bounds = (c.action_low.clone(), c.action_high.clone());
    let stream = |offset| {
        seed.map_or_else(StdRng::from_entropy, |s| {
            StdRng::seed_from_u64(derive_seed(s, offset))
        })
    };
    let mut exploration = stream(2);
    let mut sampling = stream(3);
    let mut smoothing = stream(4);
    let mut replay = ContinuousReplayBuffer::new(options.replay_capacity, obs, act);
    let mut batch = ContinuousTransitionBatch::new(options.batch_size, obs, act);
    let started = Instant::now();
    let mut steps = 0u64;
    let mut completed = 0u64;
    let mut window = RewardWindow::new(100);
    let mut moving_average = 0.;
    let mut reason = StopReason::Completed;
    let mut critic_loss = 0.;
    let mut actor_loss = 0.;
    hooks.publish_started();
    'episodes: for episode in 0..episodes {
        let (state, _) = env.reset(if episode == 0 {
            seed.map(|s| derive_seed(s, 1))
        } else {
            None
        });
        let mut state_buf = vec![0.; obs];
        state.write_to_buffer(&mut state_buf);
        finite(&state_buf)?;
        let mut reward_sum = 0.;
        let mut length = 0u64;
        let mut graceful = false;
        for index in 0..max_steps {
            let selected = if steps < options.start_steps as u64 {
                bounds
                    .0
                    .iter()
                    .zip(&bounds.1)
                    .map(|(&l, &h)| exploration.gen_range(l..=h))
                    .collect()
            } else {
                agent.select_action(&state_buf, options.exploration_std, &mut exploration)?
            };
            if selected.len() != act
                || selected
                    .iter()
                    .enumerate()
                    .any(|(i, v)| !v.is_finite() || *v < bounds.0[i] || *v > bounds.1[i])
            {
                return Err(error("TD3 action must be finite and bounded"));
            }
            let (next, reward, terminated, truncated, _) = env.step(action(&selected)?);
            let mut next_buf = vec![0.; obs];
            next.write_to_buffer(&mut next_buf);
            finite(&next_buf)?;
            reward_sum += reward;
            finite(&[reward, reward_sum])?;
            steps += 1;
            length += 1;
            let position = StepPosition {
                global_step: steps,
                episode: episode as u64,
                episode_step: length,
                elapsed: started.elapsed(),
            };
            let progress = progress_scalars(&values(
                reward_sum,
                moving_average,
                critic_loss,
                actor_loss,
                replay.len(),
                position,
            ));
            // A force stop discards this in-flight transition/update, keeping earlier successful updates.
            match hooks.observe_controls(position, progress) {
                StepDecision::ForceStop => {
                    reason = StopReason::ForceStop;
                    break 'episodes;
                }
                StepDecision::GracefulStop => graceful = true,
                StepDecision::Continue => {}
            }
            replay.push(
                &state_buf,
                &selected,
                reward,
                &next_buf,
                replay_done(terminated, truncated),
            );
            state_buf = next_buf;
            if steps >= options.learning_starts as u64 {
                replay.sample_with_rng(options.batch_size, &mut batch, &mut sampling);
                let losses = agent.train(&batch, &mut smoothing)?;
                critic_loss = losses.0;
                if let Some(loss) = losses.1 {
                    actor_loss = loss;
                }
            }
            let metrics = values(
                reward_sum,
                moving_average,
                critic_loss,
                actor_loss,
                replay.len(),
                position,
            );
            hooks.publish_progress(position, progress_scalars(&metrics));
            if episode_done(terminated, truncated) || index + 1 == max_steps {
                break;
            }
        }
        let position = StepPosition {
            global_step: steps,
            episode: episode as u64,
            episode_step: length,
            elapsed: started.elapsed(),
        };
        let average = window.push(reward_sum);
        finite(&[average])?;
        moving_average = average;
        hooks.publish_episode(
            position,
            values(
                reward_sum,
                average,
                critic_loss,
                actor_loss,
                replay.len(),
                position,
            ),
        );
        completed += 1;
        match hooks.observe_controls(
            position,
            progress_scalars(&values(
                reward_sum,
                average,
                critic_loss,
                actor_loss,
                replay.len(),
                position,
            )),
        ) {
            StepDecision::ForceStop => {
                reason = StopReason::ForceStop;
                break;
            }
            StepDecision::GracefulStop => graceful = true,
            StepDecision::Continue => {}
        }
        if graceful {
            reason = StopReason::GracefulStop;
            break;
        }
    }
    agent.save(options)?;
    Ok(TrainingSummary::stopped(
        steps,
        completed,
        started.elapsed(),
        reason,
    ))
}
fn finite(values: &[f32]) -> Result<(), TrainerError> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(error("TD3 environment/action/reward values must be finite"))
    } else {
        Ok(())
    }
}
fn error(message: &str) -> TrainerError {
    TrainerError {
        message: message.into(),
    }
}
fn descriptors() -> Vec<MetricDescriptor> {
    [
        (REWARD, "reward.episode", Some(MetricRole::EpisodeReward)),
        (AVERAGE, "reward.moving_average", None),
        (CRITIC, "loss.critic", Some(MetricRole::PrimaryLoss)),
        (ACTOR, "loss.policy", Some(MetricRole::PolicySignal)),
        (REPLAY, "replay.size", None),
        (
            SPEED,
            "performance.steps_per_second",
            Some(MetricRole::Throughput),
        ),
    ]
    .into_iter()
    .map(|(id, name, role)| MetricDescriptor {
        id,
        name: name.into(),
        label: match name {
            "reward.episode" => "Episode reward",
            "reward.moving_average" => "Reward average over the latest 100 episodes",
            "loss.critic" => "Latest TD3 critic loss",
            "loss.policy" => "Latest delayed actor loss",
            "replay.size" => "Replay buffer size",
            _ => "Steps per second",
        }
        .into(),
        unit: match name {
            "replay.size" => Some("transitions".into()),
            "performance.steps_per_second" => Some("steps/s".into()),
            _ => None,
        },
        kind: if id == SPEED {
            MetricKind::Rate
        } else {
            MetricKind::Gauge
        },
        role,
    })
    .collect()
}
fn values(
    reward: f32,
    average: f32,
    critic: f32,
    actor: f32,
    replay: usize,
    position: StepPosition,
) -> SmallVec<[MetricValue; 8]> {
    smallvec![
        MetricValue {
            metric: REWARD,
            value: reward as f64
        },
        MetricValue {
            metric: AVERAGE,
            value: average as f64
        },
        MetricValue {
            metric: CRITIC,
            value: critic as f64
        },
        MetricValue {
            metric: ACTOR,
            value: actor as f64
        },
        MetricValue {
            metric: REPLAY,
            value: replay as f64
        },
        MetricValue {
            metric: SPEED,
            value: throughput(position.global_step, position.elapsed)
        }
    ]
}
