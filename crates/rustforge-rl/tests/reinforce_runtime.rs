use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rustforge_rl::agent::{cartpole_reinforce_config, REINFORCEConfig, ReinforceTrainerAdapter};
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
use rustforge_rl::runtime::trainer::{MetricId, MetricRole, StopReason, Trainer, TrainerContext};

#[derive(Clone, Copy)]
struct OnlyAction;

impl TryFrom<usize> for OnlyAction {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        (value == 0).then_some(Self).ok_or("invalid action")
    }
}

#[derive(Clone, Copy)]
enum Boundary {
    Terminated,
    Truncated,
    StepCap,
}

struct BoundaryEnv {
    boundary: Boundary,
    step: usize,
}

impl BoundaryEnv {
    fn new(boundary: Boundary) -> Self {
        Self { boundary, step: 0 }
    }
}

impl Environment for BoundaryEnv {
    type Obs = [f32; 1];
    type Act = OnlyAction;
    type Info = ();

    fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.step = 0;
        ([0.0], ())
    }

    fn step(&mut self, _action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        self.step += 1;
        let at_boundary = self.step == 2;
        (
            [self.step as f32],
            1.0,
            at_boundary && matches!(self.boundary, Boundary::Terminated),
            at_boundary && matches!(self.boundary, Boundary::Truncated),
            (),
        )
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
    inner: BoundaryEnv,
    control: TrainerControl,
    request: StopRequest,
    trigger_step: usize,
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

struct CountingEnv {
    inner: BoundaryEnv,
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

struct ObservedEnv {
    resets: Arc<AtomicUsize>,
    steps: Arc<AtomicUsize>,
}

impl Environment for ObservedEnv {
    type Obs = [f32; 1];
    type Act = OnlyAction;
    type Info = ();

    fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.resets.fetch_add(1, Ordering::SeqCst);
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

fn config() -> REINFORCEConfig {
    REINFORCEConfig {
        obs_dim: 1,
        num_actions: 1,
        hidden_dim: 4,
        lr: 1e-3,
        gamma: 0.99,
        use_baseline: true,
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
    crossbeam_channel::Receiver<EventEnvelope>,
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
    let adapter = ReinforceTrainerAdapter::new(
        CartPole::with_max_steps(30),
        REINFORCEConfig {
            hidden_dim: 16,
            ..cartpole_reinforce_config()
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
                .filter(|value| value.metric != MetricId::new(305))
                .map(|value| (value.metric, value.value.to_bits()))
                .collect();
            (record.episode, record.global_step, values)
        })
        .collect();
    retained
}

#[test]
fn reinforce_profile_and_metadata_pin_the_contract() {
    let profile = cartpole_reinforce_config();
    assert_eq!(profile.obs_dim, 4);
    assert_eq!(profile.num_actions, 2);

    let adapter = ReinforceTrainerAdapter::new(
        BoundaryEnv::new(Boundary::Terminated),
        config(),
        2,
        10,
        "two-step",
        None,
    );
    let metadata = adapter.metadata();
    assert_eq!(metadata.algorithm, "reinforce");
    assert!(!metadata.capabilities.checkpoint);
    let contract: Vec<_> = metadata
        .metrics
        .iter()
        .map(|metric| (metric.id.get(), metric.name.as_str(), metric.role))
        .collect();
    assert_eq!(
        contract,
        [
            (301, "reward.episode", Some(MetricRole::EpisodeReward)),
            (302, "reward.moving_average", None),
            (303, "loss.policy", Some(MetricRole::PrimaryLoss)),
            (304, "rollout.size", None),
            (
                305,
                "performance.steps_per_second",
                Some(MetricRole::Throughput),
            ),
        ]
    );
    assert!(!metadata
        .metrics
        .iter()
        .any(|metric| metric.role == Some(MetricRole::PolicySignal)));
}

#[test]
fn reinforce_runtime_is_seed_reproducible_with_a_different_seed_negative_control() {
    assert_eq!(seeded_records(2026), seeded_records(2026));
    assert_ne!(seeded_records(2026), seeded_records(2027));
}

#[test]
fn completed_episodes_publish_metrics_progress_and_all_boundary_kinds() {
    for boundary in [Boundary::Terminated, Boundary::Truncated, Boundary::StepCap] {
        let sink = RecordingSink::default();
        let records = sink.records.clone();
        let (context, receiver, progress, _, _) = runtime(Box::new(sink));
        let adapter = ReinforceTrainerAdapter::new(
            BoundaryEnv::new(boundary),
            config(),
            2,
            2,
            "boundary",
            None,
        );

        let summary = Box::new(adapter).run(context).expect("training succeeds");

        assert_eq!(summary.total_steps, 4);
        assert_eq!(summary.total_episodes, 2);
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|record| {
            record.values.len() == 5
                && record.values.iter().all(|value| value.value.is_finite())
                && record
                    .values
                    .iter()
                    .any(|value| value.metric == MetricId::new(304) && value.value == 2.0)
        }));
        assert_eq!(progress.snapshot().scalars.len(), 5);
        assert_eq!(
            receiver
                .try_iter()
                .filter(|event| matches!(event.event, TrainingEvent::EpisodeCompleted(_)))
                .count(),
            2
        );
    }
}

#[test]
fn zero_step_limit_and_action_conversion_fail_without_environment_steps() {
    let (context, _, _, _, _) = runtime(Box::new(RecordingSink::default()));
    let invalid = ReinforceTrainerAdapter::new(
        BoundaryEnv::new(Boundary::Terminated),
        config(),
        1,
        0,
        "invalid",
        None,
    );
    assert!(Box::new(invalid)
        .run(context)
        .unwrap_err()
        .message
        .contains("max_steps_per_episode"));

    let steps = Arc::new(AtomicUsize::new(0));
    let (context, _, _, _, _) = runtime(Box::new(RecordingSink::default()));
    let rejecting = ReinforceTrainerAdapter::new(
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
        .contains("REINFORCE action index"));
    assert_eq!(steps.load(Ordering::SeqCst), 0);
}

#[test]
fn observation_dimension_mismatch_returns_error_before_environment_or_agent_effects() {
    let resets = Arc::new(AtomicUsize::new(0));
    let steps = Arc::new(AtomicUsize::new(0));
    let (context, _, _, _, _) = runtime(Box::new(RecordingSink::default()));
    let adapter = ReinforceTrainerAdapter::new(
        ObservedEnv {
            resets: resets.clone(),
            steps: steps.clone(),
        },
        REINFORCEConfig {
            obs_dim: 4,
            ..config()
        },
        1,
        3,
        "mismatched-observation",
        Some(2026),
    );

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Box::new(adapter).run(context)
    }));

    assert_eq!(resets.load(Ordering::SeqCst), 0);
    assert_eq!(steps.load(Ordering::SeqCst), 0);
    let error = outcome
        .expect("observation mismatch must return an error instead of panicking")
        .expect_err("observation mismatch must fail");
    assert!(error.message.contains("REINFORCE"));
    assert!(error.message.contains("observation dimension"));
    assert!(error.message.contains("4"));
    assert!(error.message.contains("1"));
}

#[test]
fn graceful_stop_records_current_episode_and_force_stop_handles_partial_and_boundary() {
    let run = |request, trigger_step| {
        let sink = RecordingSink::default();
        let records = sink.records.clone();
        let (context, _, _, control, _) = runtime(Box::new(sink));
        let adapter = ReinforceTrainerAdapter::new(
            StopAtStepEnv {
                inner: BoundaryEnv::new(Boundary::Terminated),
                control,
                request,
                trigger_step,
            },
            config(),
            5,
            10,
            "stop",
            None,
        );
        let summary = Box::new(adapter).run(context).unwrap();
        let record_count = records.lock().unwrap().len();
        (summary, record_count)
    };

    let (graceful, graceful_records) = run(StopRequest::Graceful, 1);
    assert_eq!(graceful.stop_reason, StopReason::GracefulStop);
    assert_eq!(graceful.total_episodes, 1);
    assert_eq!(graceful_records, 1);

    let (partial_force, partial_records) = run(StopRequest::Force, 1);
    assert_eq!(partial_force.stop_reason, StopReason::ForceStop);
    assert_eq!(partial_force.total_episodes, 0);
    assert_eq!(partial_records, 0);

    let (boundary_force, boundary_records) = run(StopRequest::Force, 2);
    assert_eq!(boundary_force.stop_reason, StopReason::ForceStop);
    assert_eq!(boundary_force.total_episodes, 1);
    assert_eq!(boundary_records, 1);
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
    let adapter = ReinforceTrainerAdapter::new(
        CountingEnv {
            inner: BoundaryEnv::new(Boundary::Terminated),
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
    assert!(resolutions.iter().any(|resolution| {
        resolution.request_id == resume_id
            && resolution.control == ControlKind::Resume
            && resolution.result == ControlApplyResult::Applied
    }));
}

#[test]
fn persistence_failure_recovery_and_flush_are_authoritative() {
    let (context, receiver, _, _, persistence) = runtime(Box::new(ScriptedSink { calls: 0 }));
    let adapter = ReinforceTrainerAdapter::new(
        BoundaryEnv::new(Boundary::Terminated),
        config(),
        2,
        10,
        "persistence",
        None,
    );
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
