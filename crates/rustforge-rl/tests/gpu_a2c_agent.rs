#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustforge_autograd::{gpu::GpuVariable, no_grad};
use rustforge_rl::{
    agent::{
        gpu_a2c::{GpuA2c, GpuA2cError, GpuA2cLossError, GpuA2cRolloutOptions},
        A2CConfig, A2C,
    },
    buffer::RolloutBatch,
    env::{Environment, Space},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn config() -> A2CConfig {
    A2CConfig {
        obs_dim: 2,
        hidden_dim: 8,
        num_actions: 2,
        lr: 0.01,
        ..Default::default()
    }
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
fn params(agent: &GpuA2c) -> Vec<Vec<f32>> {
    agent
        .net()
        .parameters()
        .iter()
        .map(|p| p.to_cpu().unwrap().to_vec())
        .collect()
}
fn options(episodes: usize, max_steps: usize) -> GpuA2cRolloutOptions {
    GpuA2cRolloutOptions {
        episodes,
        max_steps,
        seed: Some(99),
    }
}
struct Bandit;
impl Environment for Bandit {
    type Obs = [f32; 2];
    type Act = usize;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        ([1., 0.], ())
    }
    fn step(&mut self, a: usize) -> (Self::Obs, f32, bool, bool, ()) {
        assert!(a < 2);
        ([1., 0.], if a == 0 { 1. } else { -1. }, true, false, ())
    }
    fn action_space(&self) -> Space {
        Space::discrete(2)
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.; 2], vec![1.; 2])
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_sampling_and_active_rollout_adam_updates_match_actual_cpu_a2c() {
    let context = GpuContext::new().unwrap();
    let mut gpu = GpuA2c::new_seeded(&context, config(), 42).unwrap();
    let mut cpu = A2C::new_seeded(config(), 42);
    for (g, c) in gpu.net().parameters().iter().zip(cpu.net().parameters()) {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
    }
    let mut r1 = StdRng::seed_from_u64(9);
    let mut r2 = r1.clone();
    for _ in 0..30 {
        let (a, lp, v) = gpu.select_action_with_rng(&[0.2, 0.7], &mut r1).unwrap();
        let (x, cv) = cpu.forward(&[0.2, 0.7]);
        let data = x.data().to_vec();
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = data.iter().map(|l| (l - max).exp()).sum::<f32>();
        assert_eq!(a, A2C::sample_action(&x.data(), &mut r2));
        close(
            &[lp, v],
            &[data[a] - max - sum.ln(), cv.data().item()],
            1e-5,
        );
        let expected: Vec<_> = data.iter().map(|l| (l - max).exp() / sum).collect();
        close(
            &gpu.action_probabilities(&[0.2, 0.7]).unwrap(),
            &expected,
            1e-6,
        );
    }
    assert_eq!(r1.gen::<u64>(), r2.gen::<u64>());
    let batch = RolloutBatch {
        states: Tensor::from_vec(
            vec![
                1.,
                0.,
                0.,
                1.,
                0.2,
                0.7,
                -0.3,
                0.8,
                0.9,
                0.2,
                f32::NAN,
                f32::NAN,
            ],
            &[6, 2],
        ),
        actions: vec![0, 1, 0, 1, 1, 999],
        returns: Tensor::from_vec(vec![1., -0.2, 0.7, 0.1, -0.4, f32::NAN], &[6, 1]),
        advantages: Tensor::from_vec(vec![2., -1., 0.3, -0.7, 0.2, f32::NAN], &[6, 1]),
        old_log_probs: Tensor::zeros(&[0]),
        size: 5,
    };
    let active = RolloutBatch {
        states: batch.states.slice_axis(0, 0, 5).unwrap(),
        actions: batch.actions[..5].to_vec(),
        returns: batch.returns.slice_axis(0, 0, 5).unwrap(),
        advantages: batch.advantages.slice_axis(0, 0, 5).unwrap(),
        old_log_probs: Tensor::full(&[5, 1], f32::NAN),
        size: 5,
    };
    for _ in 0..4 {
        let m = gpu.train_on_rollout(&batch).unwrap();
        let (total, actor, value, entropy) = cpu.train_on_rollout(&active);
        close(
            &[m.total_loss, m.actor_loss, m.value_loss, m.entropy],
            &[total, actor, value, entropy],
            1e-5,
        );
        for (g, c) in gpu.net().parameters().iter().zip(cpu.net().parameters()) {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                1e-5,
            );
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 1e-5);
        }
    }
    assert_eq!(gpu.updates(), 4);
}
struct Episodes {
    step: usize,
    terminal: bool,
    truncated: bool,
    seeds: Vec<Option<u64>>,
    invalid: bool,
    actions: usize,
}
impl Environment for Episodes {
    type Obs = [f32; 2];
    type Act = usize;
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, ()) {
        self.seeds.push(seed);
        self.step = 0;
        ([1., 0.], ())
    }
    fn step(&mut self, _: usize) -> (Self::Obs, f32, bool, bool, ()) {
        self.step += 1;
        (
            [if self.invalid { f32::NAN } else { 1. }, 0.],
            1.,
            self.terminal && self.step == 2,
            self.truncated && self.step == 2,
            (),
        )
    }
    fn action_space(&self) -> Space {
        Space::discrete(self.actions)
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.; 2], vec![1.; 2])
    }
}
fn episodes(terminal: bool, truncated: bool) -> Episodes {
    Episodes {
        step: 0,
        terminal,
        truncated,
        seeds: Vec::new(),
        invalid: false,
        actions: 2,
    }
}
#[derive(Clone)]
struct RejectedAction;
impl TryFrom<usize> for RejectedAction {
    type Error = ();
    fn try_from(_: usize) -> Result<Self, ()> {
        Err(())
    }
}
struct Rejecting {
    steps: usize,
}
impl Environment for Rejecting {
    type Obs = [f32; 2];
    type Act = RejectedAction;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        ([1., 0.], ())
    }
    fn step(&mut self, _: RejectedAction) -> (Self::Obs, f32, bool, bool, ()) {
        self.steps += 1;
        ([1., 0.], 1., true, false, ())
    }
    fn action_space(&self) -> Space {
        Space::discrete(2)
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.; 2], vec![1.; 2])
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn multi_step_gae_bootstraps_truncation_limits_and_isolates_episode_resets() {
    let context = GpuContext::new().unwrap();
    let mut c = config();
    c.gamma = 0.5;
    c.lambda = 0.8;
    let agent = GpuA2c::new_seeded(&context, c, 42).unwrap();
    for (i, p) in agent.net().parameters().iter().enumerate() {
        let mut t = Tensor::zeros(p.data().shape());
        if i == 5 {
            t.data_mut().fill(2.);
        }
        p.copy_data_from(&GpuVariable::new(&context, &t, false).unwrap())
            .unwrap();
    }
    let opts = GpuA2cRolloutOptions {
        seed: Some(u64::MAX),
        ..options(3, 3)
    };
    let mut rng = StdRng::seed_from_u64(9);
    let mut terminal = episodes(true, true);
    let a = agent
        .collect_rollout_with_rng(&mut terminal, opts, &mut rng.clone())
        .unwrap();
    let b = agent
        .collect_rollout_with_rng(&mut episodes(false, true), opts, &mut rng.clone())
        .unwrap();
    let c = agent
        .collect_rollout_with_rng(
            &mut episodes(false, false),
            GpuA2cRolloutOptions {
                max_steps: 2,
                ..opts
            },
            &mut rng,
        )
        .unwrap();
    assert_eq!(a.size, 6);
    assert_eq!(b.size, 6);
    assert_eq!(c.size, 6);
    assert_eq!(a.actions, b.actions);
    assert_eq!(b.actions, c.actions);
    close(&a.returns.to_vec(), &[1.6, 1., 1.6, 1., 1.6, 1.], 1e-6);
    close(
        &a.advantages.to_vec(),
        &[-0.4, -1., -0.4, -1., -0.4, -1.],
        1e-6,
    );
    close(&b.returns.to_vec(), &[2.; 6], 1e-6);
    close(&c.returns.to_vec(), &b.returns.to_vec(), 0.);
    assert_eq!(terminal.seeds, vec![Some(u64::MAX), Some(0), Some(1)]);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn fresh_environment_rollouts_learn_rewarding_action_with_one_update_per_batch() {
    let context = GpuContext::new().unwrap();
    let mut c = config();
    c.lr = 0.03;
    let mut agent = GpuA2c::new_seeded(&context, c, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let initial = agent.action_probabilities(&[1., 0.]).unwrap()[0];
    for _ in 0..60 {
        let batch = agent
            .collect_rollout_with_rng(&mut Bandit, options(32, 1), &mut rng)
            .unwrap();
        let m = agent.train_on_rollout(&batch).unwrap();
        assert!([m.total_loss, m.actor_loss, m.value_loss, m.entropy]
            .iter()
            .all(|v| v.is_finite()));
    }
    let final_p = agent.action_probabilities(&[1., 0.]).unwrap()[0];
    println!("GPU A2C bandit: probability {initial:.6} -> {final_p:.6}");
    assert!(final_p > 0.95 && final_p > initial + 0.2);
    assert_eq!(agent.updates(), 60);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn invalid_rollouts_and_nonfinite_gradients_preserve_parameters_and_update_clock() {
    let context = GpuContext::new().unwrap();
    let mut agent = GpuA2c::new_seeded(&context, config(), 42).unwrap();
    let before = params(&agent);
    let mut rng = StdRng::seed_from_u64(1);
    let mut batch = agent
        .collect_rollout_with_rng(&mut Bandit, options(4, 1), &mut rng)
        .unwrap();
    batch.actions[0] = 2;
    assert!(agent.train_on_rollout(&batch).is_err());
    batch.actions[0] = 0;
    batch.advantages.data_mut().fill(f32::NAN);
    assert!(agent.train_on_rollout(&batch).is_err());
    batch.advantages.data_mut().fill(1.);
    batch.size = 5;
    assert!(agent.train_on_rollout(&batch).is_err());
    batch.size = 4;
    assert!(no_grad(|| agent.train_on_rollout(&batch)).is_err());
    batch.size = 0;
    assert_eq!(agent.train_on_rollout(&batch).unwrap().total_loss, 0.);
    batch.size = 4;
    assert!(agent
        .select_action_with_rng(&[f32::NAN, 0.], &mut rng)
        .is_err());
    assert!(agent.value_of(&[1.]).is_err());
    let mut e = episodes(true, false);
    e.actions = 3;
    assert!(agent
        .collect_rollout_with_rng(&mut e, options(1, 3), &mut rng)
        .is_err());
    assert!(e.seeds.is_empty());
    e.actions = 2;
    e.invalid = true;
    assert!(agent
        .collect_rollout_with_rng(&mut e, options(1, 3), &mut rng)
        .is_err());
    for opts in [options(0, 1), options(1, 0), options(usize::MAX, 2)] {
        assert!(agent
            .collect_rollout_with_rng(&mut Bandit, opts, &mut rng)
            .is_err());
    }
    let mut rejecting = Rejecting { steps: 0 };
    assert!(agent
        .collect_rollout_with_rng(&mut rejecting, options(1, 1), &mut rng)
        .is_err());
    assert_eq!(rejecting.steps, 0);
    assert_eq!(params(&agent), before);
    assert_eq!(agent.updates(), 0);
    // Finite forward loss but overflow in critic-weight gradient; Adam must not run.
    let mut c = config();
    c.c_value = 1e20;
    let mut guarded = GpuA2c::new_seeded(&context, c, 42).unwrap();
    for (i, p) in guarded.net().parameters().iter().enumerate() {
        let mut t = Tensor::zeros(p.data().shape());
        if i == 1 {
            t.data_mut().fill(1e20);
        }
        p.copy_data_from(&GpuVariable::new(&context, &t, false).unwrap())
            .unwrap();
    }
    let before = params(&guarded);
    batch.returns.data_mut().fill(1.);
    assert!(matches!(
        guarded.train_on_rollout(&batch),
        Err(GpuA2cError::Loss(GpuA2cLossError::NonFinite))
    ));
    assert_eq!(guarded.updates(), 0);
    assert_eq!(params(&guarded), before);
    // Finite 2e20 gradients overflow Adam's square even with ordinary features.
    for p in guarded.net().parameters() {
        p.copy_data_from(
            &GpuVariable::new(&context, &Tensor::zeros(p.data().shape()), false).unwrap(),
        )
        .unwrap();
    }
    let before = params(&guarded);
    assert!(matches!(
        guarded.train_on_rollout(&batch),
        Err(GpuA2cError::Loss(GpuA2cLossError::NonFinite))
    ));
    assert_eq!(guarded.updates(), 0);
    assert_eq!(params(&guarded), before);
    // Recovered training must behave like a fresh optimizer with identical data.
    let mut reference = GpuA2c::new_seeded(&context, guarded.config().clone(), 42).unwrap();
    for (p, old) in reference
        .net()
        .parameters()
        .iter()
        .zip(guarded.net().parameters())
    {
        p.copy_data_from(&old).unwrap();
    }
    batch.returns.data_mut().fill(1e-20);
    let m = guarded.train_on_rollout(&batch).unwrap();
    let r = reference.train_on_rollout(&batch).unwrap();
    close(
        &[m.total_loss, m.actor_loss, m.value_loss, m.entropy],
        &[r.total_loss, r.actor_loss, r.value_loss, r.entropy],
        0.,
    );
    assert_eq!(params(&guarded), params(&reference));
    assert_eq!(guarded.updates(), 1);
}
