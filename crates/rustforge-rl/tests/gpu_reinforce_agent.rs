#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustforge_autograd::{gpu::GpuVariable, no_grad};
use rustforge_rl::{
    agent::{
        gpu_reinforce::{GpuReinforce, GpuReinforceError, GpuReinforceRolloutOptions},
        REINFORCEConfig, REINFORCE,
    },
    buffer::RolloutBatch,
    env::{Environment, Space},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn config() -> REINFORCEConfig {
    REINFORCEConfig {
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
fn params(agent: &GpuReinforce) -> Vec<Vec<f32>> {
    agent
        .net()
        .parameters()
        .iter()
        .map(|p| p.to_cpu().unwrap().to_vec())
        .collect()
}
fn options(episodes: usize, max_steps: usize) -> GpuReinforceRolloutOptions {
    GpuReinforceRolloutOptions {
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
struct Episodes {
    step: usize,
    terminal: bool,
    truncated: bool,
    seeds: Vec<Option<u64>>,
    invalid: bool,
    invalid_reset: bool,
    reward: f32,
    actions: usize,
}
impl Environment for Episodes {
    type Obs = [f32; 2];
    type Act = usize;
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, ()) {
        self.seeds.push(seed);
        self.step = 0;
        ([if self.invalid_reset { f32::NAN } else { 1. }, 0.], ())
    }
    fn step(&mut self, _: usize) -> (Self::Obs, f32, bool, bool, ()) {
        self.step += 1;
        (
            [if self.invalid { f32::NAN } else { 1. }, 0.],
            self.reward,
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
        invalid_reset: false,
        reward: 1.,
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
fn sampling_and_active_batch_updates_match_cpu_in_both_baseline_modes() {
    use rustforge_nn::Module;
    let context = GpuContext::new().unwrap();
    for baseline in [false, true] {
        let mut c = config();
        c.use_baseline = baseline;
        let mut gpu = GpuReinforce::new_seeded(&context, c.clone(), 42).unwrap();
        let mut cpu = REINFORCE::new_seeded(c, 42);
        let mut r1 = StdRng::seed_from_u64(9);
        let mut r2 = r1.clone();
        for _ in 0..30 {
            let (action, lp) = gpu.select_action_with_rng(&[0.2, 0.7], &mut r1).unwrap();
            let logits = cpu.forward(&[0.2, 0.7]);
            let data = logits.data().to_vec();
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = data.iter().map(|v| (v - max).exp()).sum::<f32>();
            assert_eq!(action, REINFORCE::sample_action(&logits.data(), &mut r2));
            close(&[lp], &[data[action] - max - sum.ln()], 1e-6);
            let probabilities: Vec<_> = data.iter().map(|v| (v - max).exp() / sum).collect();
            close(
                &gpu.action_probabilities(&[0.2, 0.7]).unwrap(),
                &probabilities,
                1e-6,
            );
        }
        assert_eq!(r1.gen::<u64>(), r2.gen::<u64>());
        // NaN capacity must not influence active baseline, loss or finite checks.
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
            advantages: Tensor::from_vec(vec![2., -1., 0.3, -0.7, 0.2, f32::NAN], &[6, 1]),
            returns: Tensor::zeros(&[0]),
            old_log_probs: Tensor::zeros(&[0]),
            size: 5,
        };
        let active = RolloutBatch {
            states: batch.states.slice_axis(0, 0, 5).unwrap(),
            actions: batch.actions[..5].to_vec(),
            advantages: batch.advantages.slice_axis(0, 0, 5).unwrap(),
            returns: Tensor::full(&[5, 1], f32::NAN),
            old_log_probs: Tensor::full(&[5, 1], f32::NAN),
            size: 5,
        };
        for _ in 0..4 {
            let loss = gpu.train_on_rollout(&batch).unwrap();
            close(&[loss], &[cpu.train_on_rollout(&active)], 1e-5);
            for (g, c) in gpu
                .net()
                .parameters()
                .iter()
                .zip(cpu.policy_net().parameters())
            {
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
}
#[test]
#[ignore = "requires a GPU adapter"]
fn monte_carlo_returns_zero_bootstrap_and_isolate_episodes() {
    let context = GpuContext::new().unwrap();
    let mut c = config();
    c.gamma = 0.5;
    let agent = GpuReinforce::new_seeded(&context, c, 42).unwrap();
    let opts = GpuReinforceRolloutOptions {
        seed: Some(u64::MAX),
        ..options(3, 3)
    };
    let rng = StdRng::seed_from_u64(9);
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
            GpuReinforceRolloutOptions {
                max_steps: 2,
                ..opts
            },
            &mut rng.clone(),
        )
        .unwrap();
    assert_eq!(a.size, 6);
    assert_eq!(a.actions, b.actions);
    assert_eq!(b.actions, c.actions);
    for batch in [&a, &b, &c] {
        close(&batch.returns.to_vec(), &[1.5, 1., 1.5, 1., 1.5, 1.], 0.);
        close(&batch.advantages.to_vec(), &batch.returns.to_vec(), 0.);
        assert!(batch.old_log_probs.to_vec().iter().all(|v| v.is_finite()));
    }
    assert_eq!(terminal.seeds, vec![Some(u64::MAX), Some(0), Some(1)]);
    let limited = agent
        .collect_rollout_with_rng(&mut episodes(false, false), options(2, 3), &mut rng.clone())
        .unwrap();
    close(
        &limited.returns.to_vec(),
        &[1.75, 1.5, 1., 1.75, 1.5, 1.],
        0.,
    );
    assert_eq!(agent.updates(), 0);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn fresh_rollouts_learn_rewarding_action_with_one_update_per_batch() {
    let context = GpuContext::new().unwrap();
    let mut c = config();
    c.lr = 0.03;
    let mut agent = GpuReinforce::new_seeded(&context, c, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let initial = agent.action_probabilities(&[1., 0.]).unwrap()[0];
    for _ in 0..60 {
        let batch = agent
            .collect_rollout_with_rng(&mut Bandit, options(32, 1), &mut rng)
            .unwrap();
        assert!(agent.train_on_rollout(&batch).unwrap().is_finite());
    }
    let final_p = agent.action_probabilities(&[1., 0.]).unwrap()[0];
    println!("GPU REINFORCE bandit: probability {initial:.6} -> {final_p:.6}");
    assert!(final_p > 0.95 && final_p > initial + 0.2);
    assert_eq!(agent.updates(), 60);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn invalid_inputs_and_gradient_square_overflow_preserve_parameters_and_recover() {
    let context = GpuContext::new().unwrap();
    let mut c = config();
    c.use_baseline = false;
    let mut agent = GpuReinforce::new_seeded(&context, c.clone(), 42).unwrap();
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
    assert_eq!(agent.train_on_rollout(&batch).unwrap(), 0.);
    batch.size = 4;
    assert!(agent
        .select_action_with_rng(&[f32::NAN, 0.], &mut rng)
        .is_err());
    assert!(agent.action_probabilities(&[1.]).is_err());
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
    e.invalid = false;
    e.invalid_reset = true;
    assert!(agent
        .collect_rollout_with_rng(&mut e, options(1, 3), &mut rng)
        .is_err());
    assert_eq!(e.step, 0);
    e.invalid_reset = false;
    e.reward = f32::NAN;
    assert!(agent
        .collect_rollout_with_rng(&mut e, options(1, 3), &mut rng)
        .is_err());
    e.reward = f32::MAX;
    assert!(agent
        .collect_rollout_with_rng(&mut e, options(1, 3), &mut rng)
        .is_err());
    batch.size = usize::MAX;
    assert!(agent.train_on_rollout(&batch).is_err());
    batch.size = 4;
    batch.states.data_mut()[[0, 0]] = f32::NAN;
    assert!(agent.train_on_rollout(&batch).is_err());
    batch.states.data_mut()[[0, 0]] = 1.;
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
    // Forward loss stays finite, but hidden activations overflow the weight gradient.
    let mut guarded = GpuReinforce::new_seeded(&context, c, 42).unwrap();
    batch.actions.fill(0);
    batch.advantages.data_mut().fill(1e20);
    for (i, p) in guarded.net().parameters().iter().enumerate() {
        let mut tensor = Tensor::zeros(p.data().shape());
        if i == 1 {
            tensor.data_mut().fill(1e20);
        }
        p.copy_data_from(&GpuVariable::new(&context, &tensor, false).unwrap())
            .unwrap();
    }
    let before = params(&guarded);
    assert!(matches!(
        guarded.train_on_rollout(&batch),
        Err(GpuReinforceError::NonFinite)
    ));
    assert_eq!(params(&guarded), before);
    assert_eq!(guarded.updates(), 0);
    // Ordinary hidden activation, finite gradients, but their squares overflow Adam.
    for p in guarded.net().parameters() {
        p.copy_data_from(
            &GpuVariable::new(&context, &Tensor::zeros(p.data().shape()), false).unwrap(),
        )
        .unwrap();
    }
    let before = params(&guarded);
    assert!(matches!(
        guarded.train_on_rollout(&batch),
        Err(GpuReinforceError::NonFinite)
    ));
    assert_eq!(params(&guarded), before);
    assert_eq!(guarded.updates(), 0);
    let mut reference = GpuReinforce::new_seeded(&context, guarded.config().clone(), 42).unwrap();
    for (p, old) in reference
        .net()
        .parameters()
        .iter()
        .zip(guarded.net().parameters())
    {
        p.copy_data_from(&old).unwrap();
    }
    batch.advantages.data_mut().fill(1.);
    close(
        &[guarded.train_on_rollout(&batch).unwrap()],
        &[reference.train_on_rollout(&batch).unwrap()],
        0.,
    );
    assert_eq!(params(&guarded), params(&reference));
    assert_eq!(guarded.updates(), 1);
}
