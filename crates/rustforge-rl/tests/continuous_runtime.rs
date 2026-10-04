use rustforge_rl::{
    agent::{PPOConfig, PPOContinuousConfig, PpoContinuousTrainerAdapter, PpoRuntimeOptions},
    env::{Environment, Space},
    runtime::{
        control::TrainerControl,
        event::{bounded_event_channel, DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT},
        persistence::{JsonlMetricSink, NullMetricSink, PersistenceStatus},
        progress::progress_channel,
        trainer::{StopReason, Trainer, TrainerContext},
    },
};
use std::{
    fs,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
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
    PpoRuntimeOptions::default()
}
#[test]
fn seeded_continuous_cpu_runtime_repeats_episode_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let mut all = Vec::new();
    for run in 0..2 {
        let control = TrainerControl::new();
        let adapter = PpoContinuousTrainerAdapter::new(
            env(control.clone()),
            |a: &[f32]| Ok(a[0]),
            config(),
            2,
            5,
            "constant",
            Some(2026),
        );
        assert_eq!(adapter.metadata().algorithm, "ppo-continuous");
        let path = dir.path().join(format!("{run}.jsonl"));
        let (mut ctx, _) = context(control);
        ctx.metrics =
            Box::new(JsonlMetricSink::create(&path, &adapter.metadata().metrics).unwrap());
        let result = Box::new(adapter).run(ctx).unwrap();
        assert_eq!((result.total_steps, result.total_episodes), (6, 2));
        let records: Vec<serde_json::Value> = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        all.push(
            records
                .iter()
                .map(|r| {
                    [
                        r["metrics"]["reward.episode"].as_f64().unwrap(),
                        r["metrics"]["loss.policy"].as_f64().unwrap(),
                        r["metrics"]["loss.value"].as_f64().unwrap(),
                    ]
                })
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(all[0], all[1]);
    assert!(all[0].iter().flatten().all(|v| v.is_finite()));
}
#[test]
fn cpu_continuous_controls_finish_gracefully_and_drop_forced_partial_episodes() {
    for (force, trigger, episodes, steps) in [(false, 1, 1, 3), (true, 1, 0, 1), (true, 3, 1, 3)] {
        let control = TrainerControl::new();
        let mut e = env(control.clone());
        e.stop = Some((trigger, force));
        let adapter = PpoContinuousTrainerAdapter::new(
            e,
            |a: &[f32]| Ok(a[0]),
            config(),
            10,
            5,
            "controls",
            Some(2026),
        )
        .with_options(options());
        let (ctx, _) = context(control);
        let result = Box::new(adapter).run(ctx).unwrap();
        assert_eq!(
            (result.total_episodes, result.total_steps),
            (episodes, steps)
        );
        assert_eq!(
            result.stop_reason,
            if force {
                StopReason::ForceStop
            } else {
                StopReason::GracefulStop
            }
        );
    }
}
#[test]
fn cpu_continuous_invalid_config_observation_and_action_conversion_fail_before_step() {
    for kind in 0..4 {
        let control = TrainerControl::new();
        let mut e = env(control.clone());
        e.invalid = kind == 2;
        let steps = e.steps.clone();
        let resets = e.resets.clone();
        let mut c = config();
        if kind == 0 {
            c.base.ppo_epochs = 0;
        }
        if kind == 1 {
            c.action_high[0] = 2.;
        }
        let adapter = PpoContinuousTrainerAdapter::new(
            e,
            move |a: &[f32]| {
                if kind == 3 {
                    Err(rustforge_rl::runtime::trainer::TrainerError {
                        message: "conversion rejected".into(),
                    })
                } else {
                    Ok(a[0])
                }
            },
            c,
            1,
            5,
            "invalid",
            Some(2026),
        );
        let (ctx, _) = context(control);
        assert!(Box::new(adapter).run(ctx).is_err());
        assert_eq!(steps.load(Ordering::SeqCst), 0);
        assert_eq!(resets.load(Ordering::SeqCst), usize::from(kind >= 2));
    }
}
