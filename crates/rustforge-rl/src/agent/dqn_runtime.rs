//! DQN adapter for the generic live-training runtime and its shared core loop.

use std::convert::TryFrom;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_tensor::Tensor;
use smallvec::{smallvec, SmallVec};

use super::{DQNConfig, EpsilonGreedy, DQN};
use crate::buffer::{PrioritizedReplayBuffer, ReplayBuffer, TransitionBatch};
use crate::env::{Environment, IntoTensorBuffer};
use crate::metrics::{AgentLogger, CsvLogger, EpisodeMetrics};
use crate::runtime::control::{ControlObservation, StopMode, TrainerControl};
use crate::runtime::event::{
    EpisodeSummary, MetricValue, StatusChanged, TrainingEvent, TrainingStarted,
};
use crate::runtime::persistence::{
    MetricRecord, MetricSink, PersistenceEvent, PersistenceStatus, PersistenceTracker,
};
use crate::runtime::progress::{ProgressPublisher, ProgressScalar, ProgressUpdate};
use crate::runtime::trainer::{
    MetricDescriptor, MetricId, MetricKind, StopReason, Trainer, TrainerCapabilities,
    TrainerContext, TrainerError, TrainerMetadata, TrainerStatus, TrainingSummary,
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
        Ok(result.summary)
    }
}

pub(crate) fn train_dqn_headless<E>(
    env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    log_path: Option<&str>,
) -> DQN
where
    E: Environment,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    let logger = log_path.map(|path| CsvLogger::new(path).expect("Failed to create CSV logger"));
    let mut hooks = HeadlessHooks { logger };
    let result = train_dqn_core(env, config, episodes, max_steps_per_episode, &mut hooks);
    hooks.flush();
    result.agent
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

enum StepDecision {
    Continue,
    GracefulStop,
    ForceStop,
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

struct LiveHooks {
    events: Box<dyn crate::runtime::event::TrainingEventPublisher>,
    progress: ProgressPublisher,
    control: TrainerControl,
    metrics: Box<dyn MetricSink>,
    metadata: TrainerMetadata,
    status: TrainerStatus,
    persistence: PersistenceTracker,
    persistence_status: PersistenceStatus,
}

impl LiveHooks {
    fn new(context: TrainerContext, metadata: TrainerMetadata) -> Self {
        Self {
            events: context.events,
            progress: context.progress,
            control: context.control,
            metrics: context.metrics,
            persistence_status: context.persistence,
            metadata,
            status: TrainerStatus::Running,
            persistence: PersistenceTracker::new(),
        }
    }

    fn publish(&self, event: TrainingEvent) {
        let _ = self.events.publish(event);
    }

    fn publish_status(&mut self, status: TrainerStatus) {
        if self.status != status {
            self.status = status;
            self.publish(TrainingEvent::StatusChanged(StatusChanged { status }));
        }
    }

    fn publish_resolutions(&self, observation: &ControlObservation) {
        for resolution in &observation.resolutions {
            self.publish(TrainingEvent::ControlApplied(*resolution));
        }
    }

    fn progress_scalars(state: StepState) -> SmallVec<[ProgressScalar; 8]> {
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

    fn publish_progress(&self, state: StepState) {
        self.progress.publish(ProgressUpdate {
            status: self.status,
            global_step: state.global_step,
            episode: state.episode,
            episode_step: state.episode_step,
            elapsed: state.elapsed,
            scalars: Self::progress_scalars(state),
        });
    }

    fn observe_controls(&mut self, state: StepState) -> StepDecision {
        let observation = self.control.observe(state.global_step, false);
        self.publish_resolutions(&observation);
        match observation.stop_mode {
            StopMode::Force => {
                self.publish_status(TrainerStatus::Stopping);
                self.publish_progress(state);
                return StepDecision::ForceStop;
            }
            StopMode::Graceful => {
                self.publish_status(TrainerStatus::Stopping);
                self.publish_progress(state);
                return StepDecision::GracefulStop;
            }
            StopMode::None => {}
        }

        if observation.effective_paused {
            self.publish_status(TrainerStatus::Paused);
            self.publish_progress(state);
            let resumed = self.control.wait_while_paused(state.global_step, false);
            self.publish_resolutions(&resumed);
            match resumed.stop_mode {
                StopMode::Force => {
                    self.publish_status(TrainerStatus::Stopping);
                    self.publish_progress(state);
                    return StepDecision::ForceStop;
                }
                StopMode::Graceful => {
                    self.publish_status(TrainerStatus::Stopping);
                    self.publish_progress(state);
                    return StepDecision::GracefulStop;
                }
                StopMode::None => self.publish_status(TrainerStatus::Running),
            }
        }
        self.publish_progress(state);
        StepDecision::Continue
    }

    fn record_persistence(&mut self, result: Result<(), crate::runtime::persistence::MetricError>) {
        let transition = match result {
            Ok(()) => self.persistence.record_recovered(),
            Err(error) => self.persistence.record_failure(error.message),
        };
        match transition {
            Some(PersistenceEvent::Failed(failure)) => {
                self.publish(TrainingEvent::PersistenceError(failure));
            }
            Some(PersistenceEvent::Recovered(recovery)) => {
                self.publish(TrainingEvent::PersistenceRecovered(recovery));
            }
            None => {}
        }
        self.persistence_status.store(self.persistence.summary());
    }
}

impl DqnHooks for LiveHooks {
    fn started(&mut self) {
        self.publish(TrainingEvent::Started(TrainingStarted {
            run_id: self.metadata.run_id.clone(),
            algorithm: self.metadata.algorithm.clone(),
            environment: self.metadata.environment.clone(),
        }));
    }

    fn after_step(&mut self, state: StepState) -> StepDecision {
        self.observe_controls(state)
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
        self.publish(TrainingEvent::EpisodeCompleted(EpisodeSummary {
            episode: metrics.episode as u64,
            global_step: metrics.global_step as u64,
            length: episode_length,
            metrics: values.clone(),
        }));
        let result = self.metrics.emit(&MetricRecord {
            episode: metrics.episode as u64,
            global_step: metrics.global_step as u64,
            values,
        });
        self.record_persistence(result);
    }

    fn flush(&mut self) {
        let result = self.metrics.flush();
        self.record_persistence(result);
    }
}

fn train_dqn_core<E, H>(
    mut env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    hooks: &mut H,
) -> DqnRunResult
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
    let mut rewards_window = Vec::with_capacity(100);
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
            let output = agent.q_net().forward(&Variable::from_tensor(input));
            let action_idx = explorer.select_action(&output.data(), global_step, num_actions);
            let env_action = E::Act::try_from(action_idx)
                .unwrap_or_else(|_| unreachable!("DQN produced invalid action index"));
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

        if rewards_window.len() == 100 {
            rewards_window.remove(0);
        }
        rewards_window.push(episode_reward);
        let rolling_average = rewards_window.iter().sum::<f32>() / rewards_window.len() as f32;
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

    DqnRunResult {
        agent,
        summary: TrainingSummary::stopped(
            global_step as u64,
            completed_episodes as u64,
            started.elapsed(),
            stop_reason,
        ),
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

fn throughput(global_step: u64, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        global_step as f64 / seconds
    } else {
        0.0
    }
}
