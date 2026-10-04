#![cfg(feature = "gpu")]
use rustforge_rl::{
    agent::{gpu_sac::GpuSac, SACConfig, SacDevice, SacRuntimeOptions, SacTrainerAdapter},
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
fn config() -> SACConfig {
    let mut c = SACConfig::new(1, 1, vec![-1.], vec![1.]);
    c.hidden_dim = 3;
    c.tau = 0.2;
    c
}
fn options() -> SacRuntimeOptions {
    SacRuntimeOptions {
        device: SacDevice::Gpu,
        batch_size: 2,
        replay_capacity: 8,
        start_steps: 0,
        learning_starts: 1,
        ..Default::default()
    }
}
struct Env {
    step: usize,
    length: usize,
    steps: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
    control: TrainerControl,
    stop: Option<(usize, bool)>,
    invalid: u8,
    truncated: bool,
}
impl Environment for Env {
    type Obs = [f32; 1];
    type Act = f32;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        self.step = 0;
        self.resets.fetch_add(1, Ordering::SeqCst);
        ([if self.invalid == 1 { f32::NAN } else { 1. }], ())
    }
    fn step(&mut self, action: f32) -> (Self::Obs, f32, bool, bool, ()) {
        assert!((-1. ..=1.).contains(&action));
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
        (
            [if self.invalid == 3 { f32::NAN } else { 1. }],
            if self.invalid == 2 || (self.invalid == 6 && self.step == 2) {
                f32::INFINITY
            } else {
                1.
            },
            self.step == self.length && !self.truncated,
            self.step == self.length && self.truncated,
            (),
        )
    }
    fn action_space(&self) -> Space {
        Space::continuous(vec![-1.], vec![if self.invalid == 4 { 2. } else { 1. }])
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.], vec![1.])
    }
}
fn env(control: TrainerControl) -> Env {
    Env {
        step: 0,
        length: 3,
        steps: Arc::new(AtomicUsize::new(0)),
        resets: Arc::new(AtomicUsize::new(0)),
        control,
        stop: None,
        invalid: 0,
        truncated: false,
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
#[test]
#[ignore = "requires a GPU adapter"]
fn runtime_resumes_saved_configuration_temperature_and_emits_eight_finite_jsonl_metrics() {
    let device = GpuContext::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.chk");
    let target = dir.path().join("target.chk");
    let metrics = dir.path().join("metrics.jsonl");
    let mut original = GpuSac::new_seeded(&device, config(), 42).unwrap();
    let mut prior = rustforge_rl::buffer::ContinuousTransitionBatch::new(2, 1, 1);
    prior.size = 2;
    prior.states = rustforge_tensor::Tensor::full(&[2, 1], 1.);
    prior.next_states = prior.states.clone();
    prior.rewards = rustforge_tensor::Tensor::full(&[2, 1], 0.5);
    let noise = rustforge_tensor::Tensor::zeros(&[2, 1]);
    for _ in 0..2 {
        original
            .train_step_with_noise(&prior, &noise, &noise)
            .unwrap();
    }

    original.save_checkpoint(&source).unwrap();
    let control = TrainerControl::new();
    let e = env(control.clone());
    let resets = e.resets.clone();
    let mut requested = config();
    requested.hidden_dim = 9;
    requested.obs_dim = 9;
    requested.gamma = 0.1;
    requested.init_alpha = 0.7;
    requested.alpha_lr = 0.01;
    let adapter = SacTrainerAdapter::new(
        e,
        |a: &[f32]| Ok(a[0]),
        requested,
        2,
        5,
        "constant",
        Some(2026),
    )
    .with_options(SacRuntimeOptions {
        resume: Some(source),
        checkpoint: Some(target.clone()),
        ..options()
    });
    let (mut context, receiver) = context(control);
    context.metrics =
        Box::new(JsonlMetricSink::create(&metrics, &adapter.metadata().metrics).unwrap());
    let summary = Box::new(adapter).run(context).unwrap();
    assert_eq!((summary.total_steps, summary.total_episodes), (6, 2));
    assert_eq!(resets.load(Ordering::SeqCst), 2);
    let saved = GpuSac::load_checkpoint(&device, &target).unwrap();
    assert_eq!(saved.config(), original.config());
    assert_eq!(saved.updates(), 8);
    assert_ne!(
        saved.alpha().unwrap().to_bits(),
        original.alpha().unwrap().to_bits()
    );
    let text = fs::read_to_string(metrics).unwrap();
    assert_eq!(text.lines().count(), 2);
    for line in text.lines() {
        let r: serde_json::Value = serde_json::from_str(line).unwrap();
        let m = r["metrics"].as_object().unwrap();
        assert_eq!(m.len(), 8);
        assert!(
            m.contains_key("loss.critic")
                && m.contains_key("loss.policy")
                && m.contains_key("replay.size")
                && m.contains_key("loss.temperature")
                && m.contains_key("temperature.alpha")
        );
        assert!(m.values().all(|v| v.as_f64().unwrap().is_finite()));
    }
    assert!(receiver
        .try_iter()
        .any(|e| matches!(e.event, TrainingEvent::EpisodeCompleted(_))));
}
#[test]
#[ignore = "requires a GPU adapter"]
fn controlled_stops_save_completed_atomic_updates_and_force_drops_inflight_transition() {
    let device = GpuContext::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    for (force, trigger, episodes, updates) in [(false, 1, 1, 3), (true, 1, 0, 0), (true, 3, 0, 2)]
    {
        let control = TrainerControl::new();
        let mut e = env(control.clone());
        e.stop = Some((trigger, force));
        let path = dir.path().join(format!("{force}-{trigger}.chk"));
        let adapter =
            SacTrainerAdapter::new(e, |a: &[f32]| Ok(a[0]), config(), 10, 5, "stop", Some(2026))
                .with_options(SacRuntimeOptions {
                    checkpoint: Some(path.clone()),
                    ..options()
                });
        let (context, _) = context(control);
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
        let saved = GpuSac::load_checkpoint(&device, path).unwrap();
        assert_eq!(saved.updates(), updates);
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn runtime_failures_keep_prior_checkpoint_for_bad_environment_action_conversion_and_save_path() {
    let device = GpuContext::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("keep.chk");
    GpuSac::new_seeded(&device, config(), 42)
        .unwrap()
        .save_checkpoint(&path)
        .unwrap();
    let before = fs::read(&path).unwrap();
    for invalid in 1..=6 {
        let control = TrainerControl::new();
        let mut e = env(control.clone());
        e.invalid = invalid;
        let adapter = SacTrainerAdapter::new(
            e,
            move |a: &[f32]| {
                if invalid == 5 {
                    Err(rustforge_rl::runtime::trainer::TrainerError {
                        message: "conversion failed".into(),
                    })
                } else {
                    Ok(a[0])
                }
            },
            config(),
            1,
            3,
            "bad",
            Some(2026),
        )
        .with_options(SacRuntimeOptions {
            checkpoint: Some(path.clone()),
            ..options()
        });
        let (context, _) = context(control);
        assert!(Box::new(adapter).run(context).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
    }
    let control = TrainerControl::new();
    let adapter = SacTrainerAdapter::new(
        env(control.clone()),
        |a: &[f32]| Ok(a[0]),
        config(),
        0,
        0,
        "save-error",
        None,
    )
    .with_options(SacRuntimeOptions {
        checkpoint: Some(dir.path().to_path_buf()),
        ..options()
    });
    let (context, _) = context(control);
    assert!(Box::new(adapter).run(context).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn pause_retains_inflight_transition_and_interactive_checkpoint_is_unsupported() {
    let control = TrainerControl::new();
    let e = env(control.clone());
    let steps = e.steps.clone();
    let adapter =
        SacTrainerAdapter::new(e, |a: &[f32]| Ok(a[0]), config(), 1, 3, "pause", Some(2026))
            .with_options(options());
    assert!(!adapter.metadata().capabilities.checkpoint);
    let (context, receiver) = context(control.clone());
    control.request_pause();
    control.request_checkpoint();
    let worker = std::thread::spawn(move || Box::new(adapter).run(context));
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut paused = false;
    while Instant::now() < deadline {
        if let Ok(e) = receiver.recv_timeout(Duration::from_millis(50)) {
            if let TrainingEvent::StatusChanged(s) = e.event {
                if s.status == TrainerStatus::Paused {
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
    assert_eq!((before, after, summary.total_steps), (1, 1, 3));
}

#[test]
#[ignore = "requires a GPU adapter"]
fn truncation_and_runtime_step_limit_keep_bootstrap_while_termination_disables_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for (name, length, truncated) in [
        ("terminal", 1, false),
        ("truncated", 1, true),
        ("step-limit", 3, false),
    ] {
        let control = TrainerControl::new();
        let mut e = env(control.clone());
        e.length = length;
        e.truncated = truncated;
        let mut c = config();
        c.hidden_dim = 8;
        let path = dir.path().join(format!("{name}.chk"));
        let adapter = SacTrainerAdapter::new(e, |a: &[f32]| Ok(a[0]), c, 1, 1, name, Some(2026))
            .with_options(SacRuntimeOptions {
                checkpoint: Some(path.clone()),
                ..options()
            });
        let (context, _) = context(control);
        assert_eq!(Box::new(adapter).run(context).unwrap().total_steps, 1);
        paths.push(path);
    }
    assert_eq!(fs::read(&paths[1]).unwrap(), fs::read(&paths[2]).unwrap());
    assert_ne!(fs::read(&paths[0]).unwrap(), fs::read(&paths[1]).unwrap());
}
