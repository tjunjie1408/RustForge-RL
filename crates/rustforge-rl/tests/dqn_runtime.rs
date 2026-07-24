use std::time::{Duration, Instant};

use rustforge_rl::agent::{DQNConfig, DqnTrainerAdapter};
use rustforge_rl::env::{Environment, Space};
use rustforge_rl::runtime::control::{ControlApplyResult, ControlKind, TrainerControl};
use rustforge_rl::runtime::event::{
    bounded_event_channel, TrainingEvent, DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT,
};
use rustforge_rl::runtime::persistence::{NullMetricSink, PersistenceStatus};
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::{StopReason, Trainer, TrainerContext};

#[derive(Clone, Copy)]
struct OnlyAction;

impl TryFrom<usize> for OnlyAction {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value == 0 {
            Ok(Self)
        } else {
            Err("invalid action")
        }
    }
}

struct ThreeStepEnv {
    step: usize,
}

struct ForceAtTerminalEnv {
    inner: ThreeStepEnv,
    control: TrainerControl,
}

impl Environment for ForceAtTerminalEnv {
    type Obs = [f32; 1];
    type Act = OnlyAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.inner.reset(seed)
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        let result = self.inner.step(action);
        if result.2 {
            self.control.request_force_stop();
        }
        result
    }

    fn action_space(&self) -> Space {
        self.inner.action_space()
    }

    fn observation_space(&self) -> Space {
        self.inner.observation_space()
    }
}

impl ThreeStepEnv {
    fn new() -> Self {
        Self { step: 0 }
    }
}

impl Environment for ThreeStepEnv {
    type Obs = [f32; 1];
    type Act = OnlyAction;
    type Info = ();

    fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.step = 0;
        ([0.0], ())
    }

    fn step(&mut self, _action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        self.step += 1;
        ([self.step as f32], 1.0, self.step == 3, false, ())
    }

    fn action_space(&self) -> Space {
        Space::discrete(1)
    }

    fn observation_space(&self) -> Space {
        Space::continuous(vec![-10.0], vec![10.0])
    }
}

fn config() -> DQNConfig {
    DQNConfig {
        obs_dim: 1,
        num_actions: 1,
        hidden_dim: 4,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 10,
        double_dqn: false,
        use_per: false,
        per_beta_annealing_steps: 100,
    }
}

fn runtime() -> (
    TrainerContext,
    crossbeam_channel::Receiver<rustforge_rl::runtime::event::EventEnvelope>,
    rustforge_rl::runtime::progress::ProgressReader,
    TrainerControl,
) {
    let (events, receiver, _) =
        bounded_event_channel(DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT);
    let (progress, reader) = progress_channel();
    let control = TrainerControl::new();
    (
        TrainerContext {
            events: Box::new(events),
            progress,
            control: control.clone(),
            metrics: Box::new(NullMetricSink),
            persistence: PersistenceStatus::new(),
        },
        receiver,
        reader,
        control,
    )
}

#[test]
fn dqn_adapter_exposes_generic_metadata_and_no_checkpoint_capability() {
    let adapter = DqnTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "three-step");
    let metadata = adapter.metadata();

    assert_eq!(metadata.algorithm, "dqn");
    assert_eq!(metadata.environment, "three-step");
    assert!(metadata.capabilities.pause_resume);
    assert!(metadata.capabilities.graceful_stop);
    assert!(metadata.capabilities.force_stop);
    assert!(!metadata.capabilities.checkpoint);
    let names: Vec<_> = metadata
        .metrics
        .iter()
        .map(|metric| metric.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "reward.episode",
            "reward.moving_average",
            "loss.td",
            "exploration.epsilon",
            "replay_buffer.size",
            "performance.steps_per_second",
        ]
    );
}

#[test]
fn dqn_adapter_publishes_generic_events_and_latest_progress() {
    let adapter = DqnTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "three-step");
    let (context, receiver, progress, _) = runtime();

    let summary = Box::new(adapter).run(context).expect("training succeeds");

    assert_eq!(summary.stop_reason, StopReason::Completed);
    assert_eq!(summary.total_steps, 6);
    assert_eq!(summary.total_episodes, 2);
    let events: Vec<_> = receiver.try_iter().map(|envelope| envelope.event).collect();
    assert!(matches!(events.first(), Some(TrainingEvent::Started(_))));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, TrainingEvent::EpisodeCompleted(_)))
            .count(),
        2
    );
    let latest = progress.snapshot();
    assert_eq!(latest.global_step, 6);
    assert_eq!(latest.episode, 1);
    assert_eq!(latest.episode_step, 3);
    assert!(!latest.scalars.is_empty());
}

#[test]
fn graceful_stop_finishes_episode_but_force_stop_discards_partial_episode() {
    let graceful = DqnTrainerAdapter::new(ThreeStepEnv::new(), config(), 5, 10, "three-step");
    let (graceful_context, graceful_events, _, graceful_control) = runtime();
    graceful_control.request_graceful_stop();
    let graceful_summary = Box::new(graceful)
        .run(graceful_context)
        .expect("graceful stop succeeds");
    assert_eq!(graceful_summary.stop_reason, StopReason::GracefulStop);
    assert_eq!(graceful_summary.total_steps, 3);
    assert_eq!(graceful_summary.total_episodes, 1);
    assert_eq!(
        graceful_events
            .try_iter()
            .filter(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_)))
            .count(),
        1
    );

    let force = DqnTrainerAdapter::new(ThreeStepEnv::new(), config(), 5, 10, "three-step");
    let (force_context, force_events, _, force_control) = runtime();
    force_control.request_force_stop();
    let force_summary = Box::new(force)
        .run(force_context)
        .expect("force stop succeeds");
    assert_eq!(force_summary.stop_reason, StopReason::ForceStop);
    assert_eq!(force_summary.total_steps, 1);
    assert_eq!(force_summary.total_episodes, 0);
    assert!(!force_events
        .try_iter()
        .any(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_))));
}

#[test]
fn elapsed_time_is_process_monotonic_duration() {
    let adapter = DqnTrainerAdapter::new(ThreeStepEnv::new(), config(), 1, 10, "three-step");
    let (context, _, _, _) = runtime();
    let summary = Box::new(adapter).run(context).expect("training succeeds");
    assert!(summary.elapsed >= Duration::ZERO);
}

#[test]
fn pause_resume_and_unsupported_checkpoint_are_correlated_live_controls() {
    let adapter = DqnTrainerAdapter::new(ThreeStepEnv::new(), config(), 1, 10, "three-step");
    let (context, receiver, _, control) = runtime();
    let checkpoint_id = control.request_checkpoint();
    let pause_id = control.request_pause();
    let handle = std::thread::spawn(move || Box::new(adapter).run(context));

    let deadline = Instant::now() + Duration::from_secs(2);
    let mut saw_paused = false;
    let mut resolutions = Vec::new();
    while Instant::now() < deadline && !saw_paused {
        let envelope = receiver
            .recv_timeout(Duration::from_millis(50))
            .expect("pause acknowledgement arrives");
        match envelope.event {
            TrainingEvent::ControlApplied(resolution) => resolutions.push(resolution),
            TrainingEvent::StatusChanged(change)
                if change.status == rustforge_rl::runtime::trainer::TrainerStatus::Paused =>
            {
                saw_paused = true;
            }
            _ => {}
        }
    }
    assert!(saw_paused);
    assert!(resolutions.iter().any(|resolution| {
        resolution.request_id == checkpoint_id
            && resolution.control == ControlKind::Checkpoint
            && resolution.result == ControlApplyResult::Unsupported
    }));
    assert!(resolutions.iter().any(|resolution| {
        resolution.request_id == pause_id
            && resolution.control == ControlKind::Pause
            && resolution.result == ControlApplyResult::Applied
    }));

    let resume_id = control.request_resume();
    let summary = handle
        .join()
        .expect("trainer thread does not panic")
        .expect("training succeeds");
    assert_eq!(summary.stop_reason, StopReason::Completed);
    assert!(receiver.try_iter().any(|envelope| {
        matches!(
            envelope.event,
            TrainingEvent::ControlApplied(resolution)
                if resolution.request_id == resume_id
                    && resolution.result == ControlApplyResult::Applied
        )
    }));
}

#[test]
fn force_stop_on_a_terminal_step_preserves_the_completed_episode() {
    let (context, receiver, _, control) = runtime();
    let env = ForceAtTerminalEnv {
        inner: ThreeStepEnv::new(),
        control,
    };
    let adapter = DqnTrainerAdapter::new(env, config(), 2, 10, "force-at-terminal");
    let summary = Box::new(adapter).run(context).expect("force stop succeeds");

    assert_eq!(summary.stop_reason, StopReason::ForceStop);
    assert_eq!(summary.total_steps, 3);
    assert_eq!(summary.total_episodes, 1);
    assert!(receiver
        .try_iter()
        .any(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_))));
}
