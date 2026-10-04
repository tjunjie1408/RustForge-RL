#![cfg(feature = "gpu")]
use rustforge_rl::{
    agent::{
        gpu_ppo::GpuPpoContinuous, PPOConfig, PPOContinuousConfig, PpoContinuousTrainerAdapter,
        PpoDevice, PpoRuntimeOptions,
    },
    env::{Environment, Space},
    runtime::{
        control::TrainerControl,
        event::{
            bounded_event_channel, TrainingEvent, DEFAULT_EVENT_CAPACITY,
            DEFAULT_EVENT_PUBLISH_WAIT,
        },
        persistence::{JsonlMetricSink, NullMetricSink, PersistenceStatus},
        progress::progress_channel,
        trainer::{StopReason, Trainer, TrainerContext, TrainerStatus},
    },
};
use rustforge_tensor::gpu::GpuContext;
use std::{
    fs,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
fn config() -> PPOContinuousConfig {
    PPOContinuousConfig {
        base: PPOConfig {
            obs_dim: 1,
            hidden_dim: 3,
            lr: 0.03,
            gamma: 0.5,
            ppo_epochs: 2,
            mini_batch_size: 2,
            ..PPOConfig::default()
        },
        act_dim: 1,
        action_low: vec![-1.],
        action_high: vec![1.],
    }
}
struct Env {
    step: usize,
    steps: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
    control: TrainerControl,
    stop: Option<(usize, bool)>,
    invalid: bool,
}
impl Environment for Env {
    type Obs = [f32; 1];
    type Act = f32;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        self.step = 0;
        self.resets.fetch_add(1, Ordering::SeqCst);
        ([if self.invalid { f32::NAN } else { 1. }], ())
    }
    fn step(&mut self, a: f32) -> (Self::Obs, f32, bool, bool, ()) {
        assert!((-1. ..=1.).contains(&a));
        self.step += 1;
        self.steps.fetch_add(1, Ordering::SeqCst);
        if let Some((trigger, force)) = self.stop {
            if self.step == trigger {
                if force {
                    self.control.request_force_stop();
                } else {
                    self.control.request_graceful_stop();
                }
            }
        }
        ([1.], 1., self.step == 3, false, ())
    }
    fn action_space(&self) -> Space {
        Space::continuous(vec![-1.], vec![1.])
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.], vec![1.])
    }
}
fn env(control: TrainerControl) -> Env {
    Env {
        step: 0,
        steps: Arc::new(AtomicUsize::new(0)),
        resets: Arc::new(AtomicUsize::new(0)),
        control,
        stop: None,
        invalid: false,
    }
}
fn context(
    control: TrainerControl,
) -> (
    TrainerContext,
    crossbeam_channel::Receiver<rustforge_rl::runtime::event::EventEnvelope>,
) {
    let (events, receiver, _) =
        bounded_event_channel(DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT);
    let (progress, _) = progress_channel();
    (
        TrainerContext {
            events: Box::new(events),
            progress,
            control,
            metrics: Box::new(NullMetricSink),
            persistence: PersistenceStatus::new(),
        },
        receiver,
    )
}
fn options() -> PpoRuntimeOptions {
    PpoRuntimeOptions {
        device: PpoDevice::Gpu,
        ..Default::default()
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn runtime_resumes_saved_configuration_and_emits_finite_jsonl_metrics() {
    let device = GpuContext::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.chk");
    let target = directory.path().join("target.chk");
    let metrics = directory.path().join("metrics.jsonl");
    let original = GpuPpoContinuous::new_seeded(&device, config(), 42).unwrap();
    original.save_checkpoint(&source).unwrap();
    let control = TrainerControl::new();
    let e = env(control.clone());
    let resets = e.resets.clone();
    let mut requested = config();
    requested.base.hidden_dim = 9;
    requested.base.gamma = 0.1;
    requested.base.ppo_epochs = 7;
    requested.base.lr = 0.001;
    let adapter = PpoContinuousTrainerAdapter::new(
        e,
        |a: &[f32]| Ok(a[0]),
        requested,
        2,
        5,
        "constant",
        Some(2026),
    )
    .with_options(PpoRuntimeOptions {
        resume: Some(source),
        checkpoint: Some(target.clone()),
        ..options()
    });
    let (mut context, receiver) = context(control);
    context.metrics =
        Box::new(JsonlMetricSink::create(&metrics, &adapter.metadata().metrics).unwrap());
    let summary = Box::new(adapter).run(context).unwrap();
    assert_eq!(summary.total_steps, 6);
    assert_eq!(summary.total_episodes, 2);
    assert_eq!(resets.load(Ordering::SeqCst), 2);
    let restored = GpuPpoContinuous::load_checkpoint(&device, &target).unwrap();
    assert_eq!(restored.config(), original.config());
    assert_eq!(
        (restored.actor_updates(), restored.critic_updates()),
        (8, 8)
    );
    let text = fs::read_to_string(metrics).unwrap();
    assert_eq!(text.lines().count(), 2);
    for line in text.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        let m = record["metrics"].as_object().unwrap();
        assert!(m.contains_key("loss.policy") && m.contains_key("loss.value"));
        assert!(m.values().all(|v| v.as_f64().unwrap().is_finite()));
    }
    assert!(receiver
        .try_iter()
        .any(|e| matches!(e.event, TrainingEvent::EpisodeCompleted(_))));
}
#[test]
#[ignore = "requires a GPU adapter"]
fn controlled_stops_save_only_completed_updates_and_drop_forced_partial_rollout() {
    let device = GpuContext::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    for (force, trigger, episodes, updates) in [(false, 1, 1, 4), (true, 1, 0, 0), (true, 3, 1, 4)]
    {
        let control = TrainerControl::new();
        let mut e = env(control.clone());
        e.stop = Some((trigger, force));
        let path = directory.path().join(format!("{force}-{trigger}.chk"));
        let adapter = PpoContinuousTrainerAdapter::new(
            e,
            |a: &[f32]| Ok(a[0]),
            config(),
            10,
            5,
            "stop",
            Some(2026),
        )
        .with_options(PpoRuntimeOptions {
            checkpoint: Some(path.clone()),
            ..options()
        });
        let (context, _receiver) = context(control);
        let summary = Box::new(adapter).run(context).unwrap();
        assert_eq!(summary.total_episodes, episodes);
        assert_eq!(
            summary.stop_reason,
            if force {
                StopReason::ForceStop
            } else {
                StopReason::GracefulStop
            }
        );
        let agent = GpuPpoContinuous::load_checkpoint(&device, &path).unwrap();
        assert_eq!(
            (agent.actor_updates(), agent.critic_updates()),
            (updates, updates)
        );
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn failed_resume_dimension_mismatch_and_nonfinite_observation_preserve_checkpoint() {
    let device = GpuContext::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ppo.chk");
    GpuPpoContinuous::new_seeded(&device, config(), 42)
        .unwrap()
        .save_checkpoint(&path)
        .unwrap();
    let saved = fs::read(&path).unwrap();
    for kind in 0..4 {
        let control = TrainerControl::new();
        let mut e = env(control.clone());
        let steps = e.steps.clone();
        let resets = e.resets.clone();
        e.invalid = kind == 2;
        let mut c = config();
        if kind == 3 {
            c.action_high[0] = 2.;
        }
        if kind == 1 {
            c.base.obs_dim = 2;
        }
        let adapter = PpoContinuousTrainerAdapter::new(
            e,
            |a: &[f32]| Ok(a[0]),
            c,
            1,
            5,
            "invalid",
            Some(2026),
        )
        .with_options(PpoRuntimeOptions {
            resume: if kind == 0 {
                Some(directory.path().join("missing"))
            } else {
                None
            },
            checkpoint: Some(path.clone()),
            ..options()
        });
        let (context, _receiver) = context(control);
        assert!(Box::new(adapter).run(context).is_err());
        assert_eq!(steps.load(Ordering::SeqCst), 0);
        assert_eq!(resets.load(Ordering::SeqCst), usize::from(kind == 2));
        assert_eq!(fs::read(&path).unwrap(), saved);
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn pause_resume_blocks_steps_and_worker_owns_gpu_agent() {
    let control = TrainerControl::new();
    let e = env(control.clone());
    let steps = e.steps.clone();
    let adapter = PpoContinuousTrainerAdapter::new(
        e,
        |a: &[f32]| Ok(a[0]),
        config(),
        1,
        5,
        "pause",
        Some(2026),
    )
    .with_options(options());
    let (context, receiver) = context(control.clone());
    control.request_pause();
    control.request_checkpoint();
    let worker = std::thread::spawn(move || Box::new(adapter).run(context));
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut paused = false;
    while Instant::now() < deadline {
        if let Ok(envelope) = receiver.recv_timeout(Duration::from_millis(50)) {
            if let TrainingEvent::StatusChanged(status) = envelope.event {
                if status.status == TrainerStatus::Paused {
                    paused = true;
                    break;
                }
            }
        }
    }
    let before = steps.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(30));
    let after = steps.load(Ordering::SeqCst);
    control.request_resume();
    let summary = worker.join().unwrap().unwrap();
    assert!(paused);
    assert_eq!(before, 1);
    assert_eq!(before, after);
    assert_eq!(summary.total_steps, 3);
}
