use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rustforge_rl::agent::{cartpole_a2c_config, A2CConfig, A2cTrainerAdapter};
use rustforge_rl::env::{CartPole, Environment, Space};
use rustforge_rl::runtime::control::{ControlApplyResult, ControlKind, TrainerControl};
use rustforge_rl::runtime::event::{
    bounded_event_channel, EventEnvelope, TrainingEvent, DEFAULT_EVENT_CAPACITY,
    DEFAULT_EVENT_PUBLISH_WAIT,
};
use rustforge_rl::runtime::persistence::{
    MetricError, MetricRecord, MetricSink, PersistenceStatus,
};
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::{MetricId, StopReason, Trainer, TrainerContext};

#[derive(Clone, Copy)]
struct OnlyAction;

impl TryFrom<usize> for OnlyAction {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        (value == 0).then_some(Self).ok_or("invalid action")
    }
}

struct ThreeStepEnv {
    step: usize,
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

#[derive(Clone, Copy)]
enum StopRequest {
    Graceful,
    Force,
}

struct StopAtStepEnv {
    inner: ThreeStepEnv,
    control: TrainerControl,
    request: StopRequest,
    trigger_step: usize,
}

struct CountingEnv {
    inner: ThreeStepEnv,
    steps: Arc<AtomicUsize>,
}

impl Environment for CountingEnv {
    type Obs = [f32; 1];
    type Act = OnlyAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.inner.reset(seed)
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        self.steps.fetch_add(1, Ordering::SeqCst);
        self.inner.step(action)
    }

    fn action_space(&self) -> Space {
        self.inner.action_space()
    }

    fn observation_space(&self) -> Space {
        self.inner.observation_space()
    }
}

impl Environment for StopAtStepEnv {
    type Obs = [f32; 1];
    type Act = OnlyAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.inner.reset(seed)
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        let result = self.inner.step(action);
        if self.inner.step == self.trigger_step {
            match self.request {
                StopRequest::Graceful => self.control.request_graceful_stop(),
                StopRequest::Force => self.control.request_force_stop(),
            };
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

#[derive(Clone, Copy)]
struct RejectingAction;

impl TryFrom<usize> for RejectingAction {
    type Error = &'static str;

    fn try_from(_value: usize) -> Result<Self, Self::Error> {
        Err("rejected")
    }
}

struct RejectingEnv {
    steps: Arc<AtomicUsize>,
}

impl Environment for RejectingEnv {
    type Obs = [f32; 1];
    type Act = RejectingAction;
    type Info = ();

    fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Self::Info) {
        ([0.0], ())
    }

    fn step(&mut self, _action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        self.steps.fetch_add(1, Ordering::SeqCst);
        ([0.0], 0.0, true, false, ())
    }

    fn action_space(&self) -> Space {
        Space::discrete(1)
    }

    fn observation_space(&self) -> Space {
        Space::continuous(vec![-1.0], vec![1.0])
    }
}

fn config() -> A2CConfig {
    A2CConfig {
        obs_dim: 1,
        num_actions: 1,
        hidden_dim: 4,
        lr: 1e-3,
        gamma: 0.99,
        lambda: 0.95,
        c_value: 0.5,
        c_entropy: 0.01,
    }
}

#[derive(Clone, Default)]
struct RecordingSink {
    records: Arc<Mutex<Vec<MetricRecord>>>,
}

impl MetricSink for RecordingSink {
    fn emit(&mut self, record: &MetricRecord) -> Result<(), MetricError> {
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MetricError> {
        Ok(())
    }
}

struct ScriptedSink {
    calls: usize,
}

impl MetricSink for ScriptedSink {
    fn emit(&mut self, _record: &MetricRecord) -> Result<(), MetricError> {
        self.calls += 1;
        if self.calls == 1 {
            Err(MetricError {
                message: "emit failed".into(),
            })
        } else {
            Ok(())
        }
    }

    fn flush(&mut self) -> Result<(), MetricError> {
        Err(MetricError {
            message: "flush failed".into(),
        })
    }
}

fn runtime(
    sink: Box<dyn MetricSink>,
) -> (
    TrainerContext,
    crossbeam_channel::Receiver<rustforge_rl::runtime::event::EventEnvelope>,
    rustforge_rl::runtime::progress::ProgressReader,
    TrainerControl,
    PersistenceStatus,
) {
    let (events, receiver, _) =
        bounded_event_channel(DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT);
    let (progress, reader) = progress_channel();
    let control = TrainerControl::new();
    let persistence = PersistenceStatus::new();
    (
        TrainerContext {
            events: Box::new(events),
            progress,
            control: control.clone(),
            metrics: sink,
            persistence: persistence.clone(),
        },
        receiver,
        reader,
        control,
        persistence,
    )
}

type RetainedEpisode = (u64, u64, Vec<(MetricId, u64)>);

fn seeded_records(seed: u64) -> Vec<RetainedEpisode> {
    let sink = RecordingSink::default();
    let records = sink.records.clone();
    let (context, _, _, _, _) = runtime(Box::new(sink));
    let adapter = A2cTrainerAdapter::new(
        CartPole::with_max_steps(30),
        A2CConfig {
            hidden_dim: 16,
            ..cartpole_a2c_config()
        },
        6,
        30,
        "cartpole",
        Some(seed),
    );
    Box::new(adapter).run(context).expect("training succeeds");
    let retained = records
        .lock()
        .unwrap()
        .iter()
        .map(|record| {
            let values = record
                .values
                .iter()
                .filter(|value| value.metric != MetricId::new(208))
                .map(|value| (value.metric, value.value.to_bits()))
                .collect();
            (record.episode, record.global_step, values)
        })
        .collect();
    retained
}

#[test]
fn a2c_profile_and_metadata_pin_the_contract() {
    let profile = cartpole_a2c_config();
    assert_eq!(profile.obs_dim, 4);
    assert_eq!(profile.num_actions, 2);

    let adapter = A2cTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "three-step", None);
    let metadata = adapter.metadata();
    assert_eq!(metadata.algorithm, "a2c");
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
            "loss.total",
            "loss.actor",
            "loss.critic",
            "policy.entropy",
            "rollout.size",
            "performance.steps_per_second",
        ]
    );
}

#[test]
fn a2c_runtime_is_seed_reproducible_with_a_different_seed_negative_control() {
    assert_eq!(seeded_records(2026), seeded_records(2026));
    assert_ne!(seeded_records(2026), seeded_records(2027));
}

#[test]
fn completed_episodes_publish_all_metrics_and_progress() {
    let sink = RecordingSink::default();
    let records = sink.records.clone();
    let (context, receiver, progress, _, _) = runtime(Box::new(sink));
    let adapter = A2cTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "three-step", None);

    let summary = Box::new(adapter).run(context).expect("training succeeds");

    assert_eq!(summary.total_steps, 6);
    assert_eq!(summary.total_episodes, 2);
    assert_eq!(records.lock().unwrap().len(), 2);
    assert!(records
        .lock()
        .unwrap()
        .iter()
        .all(|record| record.values.len() == 8));
    assert_eq!(progress.snapshot().scalars.len(), 8);
    assert_eq!(
        receiver
            .try_iter()
            .filter(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_)))
            .count(),
        2
    );
}

#[test]
fn zero_step_limit_and_action_conversion_fail_without_environment_steps() {
    let (context, _, _, _, _) = runtime(Box::new(RecordingSink::default()));
    let invalid = A2cTrainerAdapter::new(ThreeStepEnv::new(), config(), 1, 0, "invalid", None);
    assert!(Box::new(invalid)
        .run(context)
        .unwrap_err()
        .message
        .contains("max_steps_per_episode"));

    let steps = Arc::new(AtomicUsize::new(0));
    let (context, _, _, _, _) = runtime(Box::new(RecordingSink::default()));
    let rejecting = A2cTrainerAdapter::new(
        RejectingEnv {
            steps: steps.clone(),
        },
        config(),
        1,
        3,
        "reject",
        None,
    );
    assert!(Box::new(rejecting)
        .run(context)
        .unwrap_err()
        .message
        .contains("A2C action index"));
    assert_eq!(steps.load(Ordering::SeqCst), 0);
}

#[test]
fn graceful_stop_records_current_episode_and_force_stop_discards_partial_episode() {
    let graceful_sink = RecordingSink::default();
    let graceful_records = graceful_sink.records.clone();
    let (context, _, _, control, _) = runtime(Box::new(graceful_sink));
    let graceful = A2cTrainerAdapter::new(
        StopAtStepEnv {
            inner: ThreeStepEnv::new(),
            control,
            request: StopRequest::Graceful,
            trigger_step: 1,
        },
        config(),
        5,
        10,
        "graceful",
        None,
    );
    let summary = Box::new(graceful).run(context).unwrap();
    assert_eq!(summary.stop_reason, StopReason::GracefulStop);
    assert_eq!(summary.total_episodes, 1);
    assert_eq!(graceful_records.lock().unwrap().len(), 1);

    let force_sink = RecordingSink::default();
    let force_records = force_sink.records.clone();
    let (context, _, _, control, _) = runtime(Box::new(force_sink));
    let force = A2cTrainerAdapter::new(
        StopAtStepEnv {
            inner: ThreeStepEnv::new(),
            control,
            request: StopRequest::Force,
            trigger_step: 1,
        },
        config(),
        5,
        10,
        "force",
        None,
    );
    let summary = Box::new(force).run(context).unwrap();
    assert_eq!(summary.stop_reason, StopReason::ForceStop);
    assert_eq!(summary.total_episodes, 0);
    assert!(force_records.lock().unwrap().is_empty());

    let boundary_sink = RecordingSink::default();
    let boundary_records = boundary_sink.records.clone();
    let (context, _, _, control, _) = runtime(Box::new(boundary_sink));
    let boundary_force = A2cTrainerAdapter::new(
        StopAtStepEnv {
            inner: ThreeStepEnv::new(),
            control,
            request: StopRequest::Force,
            trigger_step: 3,
        },
        config(),
        5,
        10,
        "force-boundary",
        None,
    );
    let summary = Box::new(boundary_force).run(context).unwrap();
    assert_eq!(summary.stop_reason, StopReason::ForceStop);
    assert_eq!(summary.total_episodes, 1);
    assert_eq!(boundary_records.lock().unwrap().len(), 1);
}

fn receive_until_paused(
    receiver: &crossbeam_channel::Receiver<EventEnvelope>,
    resolutions: &mut Vec<rustforge_rl::runtime::control::ControlResolution>,
) -> bool {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(envelope) => match envelope.event {
                TrainingEvent::ControlApplied(resolution) => resolutions.push(resolution),
                TrainingEvent::StatusChanged(change)
                    if change.status == rustforge_rl::runtime::trainer::TrainerStatus::Paused =>
                {
                    return true;
                }
                _ => {}
            },
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return false,
        }
    }
    false
}

#[test]
fn pause_blocks_steps_and_resume_and_checkpoint_resolutions_are_published() {
    let steps = Arc::new(AtomicUsize::new(0));
    let adapter = A2cTrainerAdapter::new(
        CountingEnv {
            inner: ThreeStepEnv::new(),
            steps: steps.clone(),
        },
        config(),
        1,
        10,
        "pause",
        None,
    );
    let (context, receiver, _, control, _) = runtime(Box::new(RecordingSink::default()));
    let checkpoint_id = control.request_checkpoint();
    let pause_id = control.request_pause();
    let handle = std::thread::spawn(move || Box::new(adapter).run(context));

    let mut resolutions = Vec::new();
    assert!(receive_until_paused(&receiver, &mut resolutions));
    let paused_at = steps.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(steps.load(Ordering::SeqCst), paused_at);
    let resume_id = control.request_resume();
    let summary = handle.join().unwrap().unwrap();
    resolutions.extend(
        receiver
            .try_iter()
            .filter_map(|envelope| match envelope.event {
                TrainingEvent::ControlApplied(resolution) => Some(resolution),
                _ => None,
            }),
    );

    assert_eq!(summary.stop_reason, StopReason::Completed);
    assert!(resolutions
        .iter()
        .any(|resolution| resolution.request_id == checkpoint_id
            && resolution.control == ControlKind::Checkpoint
            && resolution.result == ControlApplyResult::Unsupported));
    assert!(resolutions
        .iter()
        .any(|resolution| resolution.request_id == pause_id
            && resolution.control == ControlKind::Pause
            && resolution.result == ControlApplyResult::Applied));
    assert!(resolutions
        .iter()
        .any(|resolution| resolution.request_id == resume_id
            && resolution.control == ControlKind::Resume
            && resolution.result == ControlApplyResult::Applied));
}

#[test]
fn persistence_failure_recovery_and_flush_are_authoritative() {
    let (context, receiver, _, _, persistence) = runtime(Box::new(ScriptedSink { calls: 0 }));
    let adapter = A2cTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "persistence", None);
    Box::new(adapter).run(context).unwrap();
    let events: Vec<_> = receiver.try_iter().map(|event| event.event).collect();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, TrainingEvent::PersistenceError(_)))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, TrainingEvent::PersistenceRecovered(_)))
            .count(),
        1
    );
    let status = persistence.load();
    assert!(!status.complete);
    assert_eq!(status.failures, 2);
}
