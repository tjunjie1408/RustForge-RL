#![cfg(feature = "gpu")]

use rustforge_nn::gpu::GpuModule;
use rustforge_rl::{
    agent::{DQNConfig, DqnDevice, DqnRuntimeOptions, DqnTrainerAdapter, GpuDqn},
    buffer::TransitionBatch,
    env::{Environment, Space},
    runtime::{
        control::TrainerControl,
        event::{
            bounded_event_channel, TrainingEvent, DEFAULT_EVENT_CAPACITY,
            DEFAULT_EVENT_PUBLISH_WAIT,
        },
        persistence::{NullMetricSink, PersistenceStatus},
        progress::progress_channel,
        trainer::{StopReason, Trainer, TrainerContext},
    },
};
use rustforge_tensor::gpu::GpuContext;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct ConstantEnv {
    resets: Arc<AtomicUsize>,
}
impl Environment for ConstantEnv {
    type Obs = [f32; 1];
    type Act = usize;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        self.resets.fetch_add(1, Ordering::Relaxed);
        ([1.], ())
    }
    fn step(&mut self, action: usize) -> (Self::Obs, f32, bool, bool, ()) {
        assert_eq!(action, 0);
        ([1.], 1., false, false, ())
    }
    fn action_space(&self) -> Space {
        Space::discrete(1)
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.], vec![1.])
    }
}
fn config() -> DQNConfig {
    DQNConfig {
        obs_dim: 1,
        num_actions: 1,
        hidden_dim: 4,
        lr: 0.01,
        gamma: 0.9,
        target_update_freq: 5,
        ..DQNConfig::default()
    }
}
fn batch() -> TransitionBatch {
    let mut batch = TransitionBatch::new(32, 1);
    batch.states.data_mut().fill(1.);
    batch.next_states.data_mut().fill(1.);
    batch.rewards.data_mut().fill(1.);
    batch.size = 32;
    batch
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
fn bits(agent: &GpuDqn, target: bool) -> Vec<Vec<u32>> {
    let net = if target {
        agent.target_net()
    } else {
        agent.q_net()
    };
    net.parameters()
        .iter()
        .map(|p| {
            p.to_cpu()
                .unwrap()
                .to_vec()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect()
}
#[test]
#[ignore = "requires a GPU adapter"]
fn worker_resume_preserves_optimizer_target_cadence_and_uses_checkpoint_config() {
    let device = GpuContext::new().expect("GPU runtime tests require an adapter");
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.chk");
    let saved = directory.path().join("saved.chk");
    let mut original = GpuDqn::new_seeded(&device, config(), 42).unwrap();
    for _ in 0..7 {
        original.train_step(&batch()).unwrap();
    }
    assert_ne!(bits(&original, false), bits(&original, true));
    original.save_checkpoint(&source).unwrap();
    let resets = Arc::new(AtomicUsize::new(0));
    // Caller defaults deliberately differ; saved hyperparameters are authoritative.
    let mut requested = config();
    requested.hidden_dim = 8;
    requested.target_update_freq = 100;
    requested.gamma = 0.1;
    let adapter = DqnTrainerAdapter::new(
        ConstantEnv {
            resets: resets.clone(),
        },
        requested,
        1,
        130,
        "constant",
    )
    .with_options(DqnRuntimeOptions {
        device: DqnDevice::Gpu,
        resume: Some(source),
        checkpoint: Some(saved.clone()),
    });
    let (context, events) = context(TrainerControl::new());
    let summary = std::thread::spawn(move || Box::new(adapter).run(context))
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(summary.total_steps, 130);
    assert_eq!(summary.total_episodes, 1);
    assert_eq!(resets.load(Ordering::Relaxed), 1);
    assert!(events
        .try_iter()
        .any(|e| matches!(e.event, TrainingEvent::EpisodeCompleted(_))));
    let mut restored = GpuDqn::load_checkpoint(&device, &saved).unwrap();
    assert_eq!(restored.train_steps(), 10); // Three updates after replay warmup.
    assert_eq!(restored.config().hidden_dim, 4);
    assert_eq!(restored.config().target_update_freq, 5);
    for _ in 0..3 {
        original.train_step(&batch()).unwrap();
    }
    assert_eq!(bits(&original, false), bits(&restored, false));
    assert_eq!(bits(&restored, false), bits(&restored, true)); // Sync at step 10.
    let left = original.train_step(&batch()).unwrap();
    let right = restored.train_step(&batch()).unwrap();
    assert_eq!(left.to_bits(), right.to_bits());
    assert_eq!(bits(&original, false), bits(&restored, false));
}
#[test]
#[ignore = "requires a GPU adapter"]
fn controlled_stops_save_without_resetting_target_lag_and_pause_works_on_worker() {
    let device = GpuContext::new().expect("GPU runtime tests require an adapter");
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.chk");
    let mut original = GpuDqn::new_seeded(&device, config(), 42).unwrap();
    for _ in 0..7 {
        original.train_step(&batch()).unwrap();
    }
    original.save_checkpoint(&source).unwrap();
    for force in [true, false] {
        let output = directory.path().join(format!("stopped-{force}.chk"));
        let control = TrainerControl::new();
        if force {
            control.request_force_stop();
        } else {
            control.request_graceful_stop();
        }
        let (context, _) = context(control);
        let adapter = DqnTrainerAdapter::new(
            ConstantEnv {
                resets: Default::default(),
            },
            config(),
            2,
            3,
            "constant",
        )
        .with_options(DqnRuntimeOptions {
            device: DqnDevice::Gpu,
            resume: Some(source.clone()),
            checkpoint: Some(output.clone()),
        });
        let summary = std::thread::spawn(move || Box::new(adapter).run(context))
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(
            summary.stop_reason,
            if force {
                StopReason::ForceStop
            } else {
                StopReason::GracefulStop
            }
        );
        assert_eq!(summary.total_steps, if force { 1 } else { 3 });
        assert_eq!(
            std::fs::read(&source).unwrap(),
            std::fs::read(output).unwrap()
        );
    }
    let control = TrainerControl::new();
    control.request_pause();
    let (context, events) = context(control.clone());
    let adapter = DqnTrainerAdapter::new(
        ConstantEnv {
            resets: Default::default(),
        },
        config(),
        1,
        3,
        "constant",
    )
    .with_options(DqnRuntimeOptions {
        device: DqnDevice::Gpu,
        ..Default::default()
    });
    let worker = std::thread::spawn(move || Box::new(adapter).run(context));
    loop {
        let event = events
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .event;
        if matches!(event, TrainingEvent::StatusChanged(change) if change.status == rustforge_rl::runtime::trainer::TrainerStatus::Paused)
        {
            break;
        }
    }
    control.request_resume();
    assert_eq!(worker.join().unwrap().unwrap().total_steps, 3);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn invalid_resume_and_save_fail_explicitly_without_checkpoint_corruption() {
    let device = GpuContext::new().expect("GPU runtime tests require an adapter");
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.chk");
    let mut incompatible = config();
    incompatible.num_actions = 2;
    GpuDqn::new_seeded(&device, incompatible, 42)
        .unwrap()
        .save_checkpoint(&source)
        .unwrap();
    let before = std::fs::read(&source).unwrap();
    let resets = Arc::new(AtomicUsize::new(0));
    let adapter = DqnTrainerAdapter::new(
        ConstantEnv {
            resets: resets.clone(),
        },
        config(),
        1,
        3,
        "constant",
    )
    .with_options(DqnRuntimeOptions {
        device: DqnDevice::Gpu,
        resume: Some(source.clone()),
        checkpoint: Some(source.clone()),
    });
    let (ctx, _) = context(TrainerControl::new());
    let error = Box::new(adapter).run(ctx).unwrap_err();
    assert!(error.message.contains("dimensions"));
    assert_eq!(resets.load(Ordering::Relaxed), 0);
    assert_eq!(std::fs::read(&source).unwrap(), before);
    std::fs::write(&source, b"RFPARAMS legacy CPU checkpoint").unwrap();
    let adapter = DqnTrainerAdapter::new(
        ConstantEnv {
            resets: resets.clone(),
        },
        config(),
        1,
        3,
        "constant",
    )
    .with_options(DqnRuntimeOptions {
        device: DqnDevice::Gpu,
        resume: Some(source.clone()),
        checkpoint: Some(source.clone()),
    });
    let (ctx, _) = context(TrainerControl::new());
    assert!(Box::new(adapter)
        .run(ctx)
        .unwrap_err()
        .message
        .contains("checkpoint"));
    assert_eq!(resets.load(Ordering::Relaxed), 0);
    assert_eq!(
        std::fs::read(source).unwrap(),
        b"RFPARAMS legacy CPU checkpoint"
    );
    let adapter = DqnTrainerAdapter::new(ConstantEnv { resets }, config(), 1, 3, "constant")
        .with_options(DqnRuntimeOptions {
            device: DqnDevice::Gpu,
            checkpoint: Some(directory.path().to_path_buf()),
            ..Default::default()
        });
    let (ctx, _) = context(TrainerControl::new());
    assert!(Box::new(adapter).run(ctx).is_err());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn prioritized_runtime_resumes_saved_replay_mode_and_beta_schedule() {
    let device = GpuContext::new().expect("GPU runtime tests require an adapter");
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("per.chk");
    let output = directory.path().join("updated.chk");
    let mut cfg = config();
    cfg.use_per = true;
    cfg.per_beta_annealing_steps = 150;
    let mut original = GpuDqn::new_seeded(&device, cfg, 42).unwrap();
    let weights = rustforge_tensor::Tensor::ones(&[32, 1]);
    for _ in 0..7 {
        original
            .train_step_with_weights(&batch(), Some(&weights))
            .unwrap();
    }
    original.save_checkpoint(&source).unwrap();
    // Resume without --use-per must use the saved PER setting.
    let adapter = DqnTrainerAdapter::new(
        ConstantEnv {
            resets: Default::default(),
        },
        config(),
        1,
        130,
        "constant",
    )
    .with_options(DqnRuntimeOptions {
        device: DqnDevice::Gpu,
        resume: Some(source),
        checkpoint: Some(output.clone()),
    });
    let (ctx, _) = context(TrainerControl::new());
    let summary = std::thread::spawn(move || Box::new(adapter).run(ctx))
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(summary.total_steps, 130);
    let restored = GpuDqn::load_checkpoint(&device, output).unwrap();
    assert!(restored.config().use_per);
    assert_eq!(restored.config().per_beta_annealing_steps, 150);
    assert_eq!(restored.train_steps(), 10);
    assert_eq!(bits(&restored, false), bits(&restored, true));
    // Priority feedback changes weights as sampled rows are updated and new
    // transitions retain maximum priority. The runtime sampler is stochastic;
    // exact update equivalence is tested separately with seeded replay buffers.
    for p in restored.q_net().parameters() {
        assert!(p
            .to_cpu()
            .unwrap()
            .to_vec()
            .iter()
            .all(|value| value.is_finite()));
        assert!(p.requires_grad());
    }
    assert!(restored
        .target_net()
        .parameters()
        .iter()
        .all(|p| !p.requires_grad()));
}
