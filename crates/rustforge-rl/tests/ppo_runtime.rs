use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rustforge_rl::agent::{
    cartpole_ppo_config, PPOConfig, PPODiscreteConfig, PpoDiscreteTrainerAdapter,
};
use rustforge_rl::env::{CartPole, Environment, Space};
use rustforge_rl::runtime::control::{ControlApplyResult, ControlKind, TrainerControl};
use rustforge_rl::runtime::event::{
    bounded_event_channel, EventEnvelope, EventSequence, StatusChanged, TrainingEvent,
    DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT,
};
use rustforge_rl::runtime::persistence::{
    MetricError, MetricRecord, MetricSink, PersistenceStatus,
};
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::{
    MetricId, StopReason, Trainer, TrainerContext, TrainerStatus,
};

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

#[derive(Clone, Copy)]
struct RejectingAction;

struct RejectedActionError {
    index: usize,
}

impl std::fmt::Debug for RejectedActionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RejectedActionError")
            .field("index", &self.index)
            .finish()
    }
}

impl TryFrom<usize> for RejectingAction {
    type Error = RejectedActionError;

    fn try_from(index: usize) -> Result<Self, Self::Error> {
        Err(RejectedActionError { index })
    }
}

struct RejectingActionEnv {
    steps: Arc<AtomicUsize>,
}

impl Environment for RejectingActionEnv {
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

struct ThreeStepEnv {
    step: usize,
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
                StopRequest::Graceful => {
                    self.control.request_graceful_stop();
                }
                StopRequest::Force => {
                    self.control.request_force_stop();
                }
            }
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

fn config() -> PPODiscreteConfig {
    PPODiscreteConfig {
        base: PPOConfig {
            obs_dim: 1,
            hidden_dim: 4,
            lr: 1e-3,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            ppo_epochs: 1,
            mini_batch_size: 3,
        },
        num_actions: 1,
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

struct FlushRecordingSink {
    emits: Arc<AtomicUsize>,
    flushes: Arc<AtomicUsize>,
}

impl MetricSink for FlushRecordingSink {
    fn emit(&mut self, _record: &MetricRecord) -> Result<(), MetricError> {
        self.emits.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MetricError> {
        self.flushes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn runtime(
    sink: Box<dyn MetricSink>,
) -> (
    TrainerContext,
    crossbeam_channel::Receiver<rustforge_rl::runtime::event::EventEnvelope>,
    rustforge_rl::runtime::progress::ProgressReader,
    TrainerControl,
) {
    let (context, receiver, reader, control, _) = runtime_with_status(sink);
    (context, receiver, reader, control)
}

fn runtime_with_status(
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

type RetainedMetricBits = (MetricId, u64);
type RetainedEpisodeBits = (u64, u64, Vec<RetainedMetricBits>);

fn seeded_cartpole_records(seed: u64) -> Vec<RetainedEpisodeBits> {
    let config = PPODiscreteConfig {
        base: PPOConfig {
            obs_dim: 4,
            hidden_dim: 16,
            lr: 1e-3,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            ppo_epochs: 2,
            mini_batch_size: 16,
        },
        num_actions: 2,
    };
    let adapter = PpoDiscreteTrainerAdapter::new(
        CartPole::with_max_steps(50),
        config,
        8,
        50,
        "cartpole",
        Some(seed),
    );
    let sink = RecordingSink::default();
    let records = sink.records.clone();
    let (context, _, _, _) = runtime(Box::new(sink));

    Box::new(adapter).run(context).expect("training succeeds");

    let retained = records
        .lock()
        .unwrap()
        .iter()
        .map(|record| {
            let values = record
                .values
                .iter()
                .filter(|value| value.metric != MetricId::new(107))
                .map(|value| (value.metric, value.value.to_bits()))
                .collect();
            (record.episode, record.global_step, values)
        })
        .collect();
    retained
}

#[test]
fn cartpole_ppo_profile_pins_the_approved_learning_rate() {
    let config = cartpole_ppo_config();

    assert_eq!(config.base.obs_dim, 4);
    assert_eq!(config.num_actions, 2);
    assert_eq!(config.base.lr, 1e-3);
    assert_eq!(PPOConfig::default().lr, 3e-4);
}

#[test]
fn ppo_runtime_same_seed_reproduces_completed_episode_metrics() {
    assert_eq!(seeded_cartpole_records(2026), seeded_cartpole_records(2026));
}

#[test]
fn ppo_runtime_different_seed_changes_completed_episode_metrics() {
    assert_ne!(seeded_cartpole_records(2026), seeded_cartpole_records(2027));
}

#[test]
fn ppo_adapter_exposes_generic_metadata_and_no_checkpoint_capability() {
    let adapter =
        PpoDiscreteTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "three-step", None);
    let metadata = adapter.metadata();

    assert_eq!(metadata.algorithm, "ppo-discrete");
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
            "loss.policy",
            "loss.value",
            "policy.entropy",
            "rollout.size",
            "performance.steps_per_second",
        ]
    );
}

#[test]
fn rejected_action_returns_contextual_error_without_stepping_and_still_flushes() {
    let steps = Arc::new(AtomicUsize::new(0));
    let emits = Arc::new(AtomicUsize::new(0));
    let flushes = Arc::new(AtomicUsize::new(0));
    let env = RejectingActionEnv {
        steps: steps.clone(),
    };
    let adapter = PpoDiscreteTrainerAdapter::new(env, config(), 1, 3, "rejecting", None);
    let sink = FlushRecordingSink {
        emits: emits.clone(),
        flushes: flushes.clone(),
    };
    let (context, _, _, _) = runtime(Box::new(sink));

    let error = Box::new(adapter)
        .run(context)
        .expect_err("action conversion must be reported as a trainer error");

    assert!(error.message.contains("action index 0"));
    assert!(error.message.contains("RejectedActionError"));
    assert_eq!(steps.load(Ordering::SeqCst), 0);
    assert_eq!(emits.load(Ordering::SeqCst), 0);
    assert_eq!(flushes.load(Ordering::SeqCst), 1);
}

#[test]
fn zero_step_limit_is_rejected_for_positive_episodes_but_zero_episodes_is_a_noop() {
    let invalid_emits = Arc::new(AtomicUsize::new(0));
    let invalid_flushes = Arc::new(AtomicUsize::new(0));
    let invalid =
        PpoDiscreteTrainerAdapter::new(ThreeStepEnv::new(), config(), 1, 0, "invalid", None);
    let invalid_sink = FlushRecordingSink {
        emits: invalid_emits.clone(),
        flushes: invalid_flushes.clone(),
    };
    let (invalid_context, invalid_events, _, _) = runtime(Box::new(invalid_sink));

    let error = Box::new(invalid)
        .run(invalid_context)
        .expect_err("positive episodes require a positive step limit");

    assert!(error.message.contains("max_steps_per_episode"));
    assert!(invalid_events.try_iter().next().is_none());
    assert_eq!(invalid_emits.load(Ordering::SeqCst), 0);
    assert_eq!(invalid_flushes.load(Ordering::SeqCst), 1);

    let noop_emits = Arc::new(AtomicUsize::new(0));
    let noop_flushes = Arc::new(AtomicUsize::new(0));
    let noop = PpoDiscreteTrainerAdapter::new(ThreeStepEnv::new(), config(), 0, 0, "noop", None);
    let noop_sink = FlushRecordingSink {
        emits: noop_emits.clone(),
        flushes: noop_flushes.clone(),
    };
    let (noop_context, noop_events, _, _) = runtime(Box::new(noop_sink));

    let summary = Box::new(noop)
        .run(noop_context)
        .expect("zero episodes is a completed no-op");

    assert_eq!(summary.stop_reason, StopReason::Completed);
    assert_eq!(summary.total_steps, 0);
    assert_eq!(summary.total_episodes, 0);
    assert!(noop_events
        .try_iter()
        .any(|event| matches!(event.event, TrainingEvent::Started(_))));
    assert_eq!(noop_emits.load(Ordering::SeqCst), 0);
    assert_eq!(noop_flushes.load(Ordering::SeqCst), 1);
}

#[test]
fn ppo_adapter_publishes_completed_episodes_metrics_and_latest_progress() {
    let adapter =
        PpoDiscreteTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "three-step", None);
    let sink = RecordingSink::default();
    let records = sink.records.clone();
    let (context, receiver, progress, _) = runtime(Box::new(sink));

    let summary = Box::new(adapter).run(context).expect("training succeeds");

    assert_eq!(summary.stop_reason, StopReason::Completed);
    assert_eq!(summary.total_steps, 6);
    assert_eq!(summary.total_episodes, 2);
    let events: Vec<_> = receiver.try_iter().map(|envelope| envelope.event).collect();
    assert!(matches!(events.first(), Some(TrainingEvent::Started(_))));
    let completed: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            TrainingEvent::EpisodeCompleted(summary) => Some(summary),
            _ => None,
        })
        .collect();
    assert_eq!(completed.len(), 2);
    assert!(completed.iter().all(|summary| {
        summary.metrics.len() == 7
            && summary
                .metrics
                .iter()
                .all(|metric| metric.value.is_finite())
    }));
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 2);
    assert!(records.iter().all(|record| {
        record.values.len() == 7 && record.values.iter().all(|metric| metric.value.is_finite())
    }));
    let latest = progress.snapshot();
    assert_eq!(latest.global_step, 6);
    assert_eq!(latest.episode, 1);
    assert_eq!(latest.episode_step, 3);
    assert_eq!(latest.scalars.len(), 7);
}

#[test]
fn graceful_stop_finishes_and_records_the_current_episode() {
    let sink = RecordingSink::default();
    let records = sink.records.clone();
    let (context, receiver, _, control) = runtime(Box::new(sink));
    let env = StopAtStepEnv {
        inner: ThreeStepEnv::new(),
        control,
        request: StopRequest::Graceful,
        trigger_step: 1,
    };
    let adapter = PpoDiscreteTrainerAdapter::new(env, config(), 5, 10, "graceful", None);

    let summary = Box::new(adapter)
        .run(context)
        .expect("graceful stop succeeds");

    assert_eq!(summary.stop_reason, StopReason::GracefulStop);
    assert_eq!(summary.total_steps, 3);
    assert_eq!(summary.total_episodes, 1);
    assert_eq!(records.lock().unwrap().len(), 1);
    assert_eq!(
        receiver
            .try_iter()
            .filter(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_)))
            .count(),
        1
    );
}

#[test]
fn force_stop_after_a_non_terminal_step_discards_the_partial_rollout() {
    let sink = RecordingSink::default();
    let records = sink.records.clone();
    let (context, receiver, _, control) = runtime(Box::new(sink));
    let env = StopAtStepEnv {
        inner: ThreeStepEnv::new(),
        control,
        request: StopRequest::Force,
        trigger_step: 1,
    };
    let adapter = PpoDiscreteTrainerAdapter::new(env, config(), 5, 10, "force-partial", None);

    let summary = Box::new(adapter).run(context).expect("force stop succeeds");

    assert_eq!(summary.stop_reason, StopReason::ForceStop);
    assert_eq!(summary.total_steps, 1);
    assert_eq!(summary.total_episodes, 0);
    assert!(records.lock().unwrap().is_empty());
    assert!(!receiver
        .try_iter()
        .any(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_))));
}

#[test]
fn force_stop_on_a_terminal_step_preserves_the_completed_episode() {
    let sink = RecordingSink::default();
    let records = sink.records.clone();
    let (context, receiver, _, control) = runtime(Box::new(sink));
    let env = StopAtStepEnv {
        inner: ThreeStepEnv::new(),
        control,
        request: StopRequest::Force,
        trigger_step: 3,
    };
    let adapter = PpoDiscreteTrainerAdapter::new(env, config(), 2, 10, "force-terminal", None);

    let summary = Box::new(adapter).run(context).expect("force stop succeeds");

    assert_eq!(summary.stop_reason, StopReason::ForceStop);
    assert_eq!(summary.total_steps, 3);
    assert_eq!(summary.total_episodes, 1);
    assert_eq!(records.lock().unwrap().len(), 1);
    assert!(receiver
        .try_iter()
        .any(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_))));
}

fn receive_until_paused(
    receiver: &crossbeam_channel::Receiver<EventEnvelope>,
    deadline: Instant,
    resolutions: &mut Vec<rustforge_rl::runtime::control::ControlResolution>,
) -> bool {
    while Instant::now() < deadline {
        let wait = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(50));
        match receiver.recv_timeout(wait) {
            Ok(envelope) => match envelope.event {
                TrainingEvent::ControlApplied(resolution) => resolutions.push(resolution),
                TrainingEvent::StatusChanged(change) if change.status == TrainerStatus::Paused => {
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
fn pause_event_polling_survives_a_transient_receive_timeout() {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let producer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(75));
        sender
            .send(EventEnvelope {
                sequence: EventSequence::new(1),
                emitted_at: std::time::SystemTime::now(),
                event: TrainingEvent::StatusChanged(StatusChanged {
                    status: TrainerStatus::Paused,
                }),
            })
            .expect("test receiver remains connected");
    });
    let mut resolutions = Vec::new();

    let saw_paused = receive_until_paused(
        &receiver,
        Instant::now() + Duration::from_millis(250),
        &mut resolutions,
    );

    producer.join().expect("producer does not panic");
    assert!(saw_paused);
}

#[test]
fn pause_blocks_environment_steps_and_checkpoint_is_unsupported_until_resume() {
    let steps = Arc::new(AtomicUsize::new(0));
    let env = CountingEnv {
        inner: ThreeStepEnv::new(),
        steps: steps.clone(),
    };
    let adapter = PpoDiscreteTrainerAdapter::new(env, config(), 1, 10, "pause", None);
    let (context, receiver, _, control) = runtime(Box::new(RecordingSink::default()));
    let checkpoint_id = control.request_checkpoint();
    let pause_id = control.request_pause();
    let handle = std::thread::spawn(move || Box::new(adapter).run(context));

    let mut resolutions = Vec::new();
    let saw_paused = receive_until_paused(
        &receiver,
        Instant::now() + Duration::from_secs(2),
        &mut resolutions,
    );

    let paused_at = steps.load(Ordering::SeqCst);
    if saw_paused {
        std::thread::sleep(Duration::from_millis(30));
    }
    let still_paused_at = steps.load(Ordering::SeqCst);
    let resume_id = control.request_resume();
    let summary = handle
        .join()
        .expect("trainer thread does not panic")
        .expect("training succeeds");
    for envelope in receiver.try_iter() {
        if let TrainingEvent::ControlApplied(resolution) = envelope.event {
            resolutions.push(resolution);
        }
    }

    assert!(saw_paused, "pause acknowledgement arrives before deadline");
    assert_eq!(paused_at, 1);
    assert_eq!(still_paused_at, paused_at);
    assert_eq!(summary.stop_reason, StopReason::Completed);
    assert!(resolutions.iter().any(|resolution| {
        resolution.request_id == resume_id
            && resolution.control == ControlKind::Resume
            && resolution.result == ControlApplyResult::Applied
    }));
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
}

struct ScriptedSink {
    emit_calls: usize,
}

impl MetricSink for ScriptedSink {
    fn emit(&mut self, _record: &MetricRecord) -> Result<(), MetricError> {
        self.emit_calls += 1;
        if self.emit_calls == 1 {
            Err(MetricError {
                message: "first emit failed".into(),
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

#[test]
fn persistence_failure_recovery_and_flush_failure_are_tracked() {
    let adapter =
        PpoDiscreteTrainerAdapter::new(ThreeStepEnv::new(), config(), 2, 10, "persistence", None);
    let sink = ScriptedSink { emit_calls: 0 };
    let (context, receiver, _, _, persistence) = runtime_with_status(Box::new(sink));

    let summary = Box::new(adapter).run(context).expect("training succeeds");

    assert_eq!(summary.stop_reason, StopReason::Completed);
    let events: Vec<_> = receiver.try_iter().map(|envelope| envelope.event).collect();
    let failures: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            TrainingEvent::PersistenceError(failure) => Some(failure),
            _ => None,
        })
        .collect();
    let recoveries: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            TrainingEvent::PersistenceRecovered(recovery) => Some(recovery),
            _ => None,
        })
        .collect();
    assert_eq!(failures.len(), 2);
    assert_eq!(failures[0].message, "first emit failed");
    assert_eq!(failures[0].failures, 1);
    assert_eq!(failures[1].message, "flush failed");
    assert_eq!(failures[1].failures, 2);
    assert_eq!(recoveries.len(), 1);
    assert_eq!(recoveries[0].failures, 1);
    let status = persistence.load();
    assert!(!status.complete);
    assert_eq!(status.failures, 2);
    assert_eq!(status.first_error.as_deref(), Some("first emit failed"));
    assert_eq!(status.last_error.as_deref(), Some("flush failed"));
}
