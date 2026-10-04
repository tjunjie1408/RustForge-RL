#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, SeedableRng};
use rustforge_autograd::{gpu::GpuVariable, no_grad, Variable};
use rustforge_rl::{
    agent::{
        a2c::ActorCriticNet,
        gpu_ppo::{GpuActorCriticNet, GpuPpoDiscrete},
        PPOConfig, PPODiscrete, PPODiscreteConfig,
    },
    buffer::RolloutBatch,
    env::{Environment, Space},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn config() -> PPODiscreteConfig {
    PPODiscreteConfig {
        base: PPOConfig {
            obs_dim: 2,
            hidden_dim: 8,
            lr: 0.01,
            ppo_epochs: 3,
            mini_batch_size: 3,
            ..PPOConfig::default()
        },
        num_actions: 2,
    }
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_network_sampling_and_shuffled_partial_minibatches_match_cpu() {
    let context = GpuContext::new().unwrap();
    let net = GpuActorCriticNet::new_seeded(&context, 2, 8, 2, 42).unwrap();
    let reference = ActorCriticNet::new_seeded(2, 8, 2, 42);
    for (g, c) in net.parameters().iter().zip(reference.parameters()) {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
    }
    let input = Tensor::from_vec(vec![1., 0., 0., 1., 0.2, 0.7], &[3, 2]);
    let (g, v) = net
        .forward(&GpuVariable::new(&context, &input, false).unwrap())
        .unwrap();
    let (c, cv) = reference.forward(&Variable::from_tensor(input));
    close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 1e-5);
    close(&v.to_cpu().unwrap().to_vec(), &cv.data().to_vec(), 1e-5);
    let mut gpu = GpuPpoDiscrete::new_seeded(&context, config(), 42).unwrap();
    let mut cpu = PPODiscrete::new_seeded(config(), 42);
    let mut r1 = StdRng::seed_from_u64(9);
    let mut r2 = StdRng::seed_from_u64(9);
    for _ in 0..30 {
        let a = gpu.select_action_with_rng(&[0.2, 0.7], &mut r1).unwrap();
        let b = cpu.select_action_with_rng(&[0.2, 0.7], &mut r2);
        assert_eq!(a.0, b.0);
        close(&[a.1, a.2], &[b.1, b.2], 1e-5);
    }
    // Unused tail is deliberately invalid: normalization/gather consume active rows only.
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
        advantages: Tensor::from_vec(vec![1., -1., 0.3, -0.7, 0.2, f32::NAN], &[6, 1]),
        old_log_probs: Tensor::from_vec(vec![-0.7, -0.6, -0.8, -0.5, -0.9, f32::NAN], &[6, 1]),
        size: 5,
    };
    for _ in 0..3 {
        let a = gpu.train_on_batch_with_rng(&batch, &mut r1).unwrap();
        let b = cpu.train_on_batch_with_rng(&batch, &mut r2);
        close(
            &[a.policy_loss, a.value_loss, a.entropy],
            &[b.0, b.1, b.2],
            2e-4,
        );
        let a = gpu.select_action_with_rng(&[0.2, 0.7], &mut r1).unwrap();
        let b = cpu.select_action_with_rng(&[0.2, 0.7], &mut r2);
        assert_eq!(a.0, b.0);
        close(&[a.1, a.2], &[b.1, b.2], 3e-4);
    }
    assert_eq!(gpu.updates(), 18);
}
struct Bandit {
    terminal: bool,
}
impl Environment for Bandit {
    type Obs = [f32; 2];
    type Act = usize;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        ([1., 0.], ())
    }
    fn step(&mut self, a: usize) -> (Self::Obs, f32, bool, bool, ()) {
        (
            [1., 0.],
            if a == 0 { 1. } else { -1. },
            self.terminal,
            !self.terminal,
            (),
        )
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
fn rollout_gae_distinguishes_terminal_truncation_and_step_limit() {
    let context = GpuContext::new().unwrap();
    let agent = GpuPpoDiscrete::new_seeded(&context, config(), 42).unwrap();
    for (i, p) in agent.net().parameters().iter().enumerate() {
        let mut t = Tensor::zeros(p.data().shape());
        if i == 5 {
            t.data_mut().fill(2.);
        }
        p.copy_data_from(&GpuVariable::new(&context, &t, true).unwrap())
            .unwrap();
    }
    let mut terminal = Bandit { terminal: true };
    let mut truncated = Bandit { terminal: false };
    let mut r1 = StdRng::seed_from_u64(9);
    let mut r2 = r1.clone();
    let a = agent
        .collect_rollout_with_rng(&mut terminal, 3, 5, Some(2), &mut r1)
        .unwrap();
    let b = agent
        .collect_rollout_with_rng(&mut truncated, 3, 5, Some(2), &mut r2)
        .unwrap();
    assert_eq!(a.size, 3);
    assert_eq!(b.size, 3);
    assert_eq!(a.actions, b.actions);
    for (a, b) in a.returns.to_vec().iter().zip(b.returns.to_vec()) {
        approx::assert_abs_diff_eq!(b - a, 1.98, epsilon = 1e-5);
    }
    struct Continuing;
    impl Environment for Continuing {
        type Obs = [f32; 2];
        type Act = usize;
        type Info = ();
        fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
            ([1., 0.], ())
        }
        fn step(&mut self, _: usize) -> (Self::Obs, f32, bool, bool, ()) {
            ([1., 0.], 1., false, false, ())
        }
        fn action_space(&self) -> Space {
            Space::discrete(2)
        }
        fn observation_space(&self) -> Space {
            Space::continuous(vec![0.; 2], vec![1.; 2])
        }
    }
    let c = agent
        .collect_rollout_with_rng(&mut Continuing, 2, 1, None, &mut r1)
        .unwrap();
    close(&c.returns.to_vec(), &[2.98, 2.98], 1e-5);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_environment_rollouts_learn_rewarding_action() {
    let context = GpuContext::new().unwrap();
    let mut c = config();
    c.base.mini_batch_size = 32;
    c.base.lr = 0.03;
    let mut agent = GpuPpoDiscrete::new_seeded(&context, c, 42).unwrap();
    let mut actions = StdRng::seed_from_u64(7);
    let mut shuffle = StdRng::seed_from_u64(8);
    let initial = agent
        .net()
        .forward(
            &GpuVariable::new(&context, &Tensor::from_vec(vec![1., 0.], &[1, 2]), false).unwrap(),
        )
        .unwrap()
        .0
        .softmax()
        .unwrap()
        .to_cpu()
        .unwrap()
        .to_vec()[0];
    for _ in 0..20 {
        let batch = agent
            .collect_rollout_with_rng(
                &mut Bandit { terminal: true },
                32,
                1,
                Some(99),
                &mut actions,
            )
            .unwrap();
        let m = agent.train_on_batch_with_rng(&batch, &mut shuffle).unwrap();
        assert!(m.total_loss.is_finite());
    }
    let final_p = agent
        .net()
        .forward(
            &GpuVariable::new(&context, &Tensor::from_vec(vec![1., 0.], &[1, 2]), false).unwrap(),
        )
        .unwrap()
        .0
        .softmax()
        .unwrap()
        .to_cpu()
        .unwrap()
        .to_vec()[0];
    println!("GPU PPO bandit: probability {initial:.6} -> {final_p:.6}");
    assert!(final_p > 0.95 && final_p > initial + 0.2);
    assert_eq!(agent.updates(), 60);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn malformed_rollouts_and_disabled_gradients_do_not_update_parameters() {
    let context = GpuContext::new().unwrap();
    let mut agent = GpuPpoDiscrete::new_seeded(&context, config(), 42).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    let mut batch = agent
        .collect_rollout_with_rng(&mut Bandit { terminal: true }, 4, 1, None, &mut rng)
        .unwrap();
    let before: Vec<_> = agent
        .net()
        .parameters()
        .iter()
        .map(|p| p.to_cpu().unwrap().to_vec())
        .collect();
    batch.actions[3] = 2;
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    batch.actions[3] = 1;
    batch.advantages.data_mut().fill(f32::NAN);
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    batch.advantages = Tensor::from_vec(vec![1e20, -1e20, 1e20, -1e20], &[4, 1]);
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    batch.advantages.data_mut().fill(1.);
    assert!(no_grad(|| agent.train_on_batch_with_rng(&batch, &mut rng)).is_err());
    batch.size = 0;
    assert_eq!(
        agent
            .train_on_batch_with_rng(&batch, &mut rng)
            .unwrap()
            .total_loss,
        0.
    );
    assert!(agent
        .select_action_with_rng(&[f32::NAN, 0.], &mut rng)
        .is_err());
    assert!(agent
        .collect_rollout_with_rng(&mut Bandit { terminal: true }, 0, 1, None, &mut rng)
        .is_err());
    assert_eq!(agent.updates(), 0);
    for (p, b) in agent.net().parameters().iter().zip(before) {
        close(&p.to_cpu().unwrap().to_vec(), &b, 0.);
    }
    let mut c = config();
    c.base.mini_batch_size = 0;
    assert!(GpuPpoDiscrete::new_seeded(&context, c, 42).is_err());
    // Finite objective with an overflowing critic weight gradient must not reach Adam.
    let mut c = config();
    c.base.value_coef = 1e20;
    let mut guarded = GpuPpoDiscrete::new_seeded(&context, c, 42).unwrap();
    for (i, p) in guarded.net().parameters().iter().enumerate() {
        let mut t = Tensor::zeros(p.data().shape());
        if i == 1 {
            t.data_mut().fill(1e20);
        }
        p.copy_data_from(&GpuVariable::new(&context, &t, true).unwrap())
            .unwrap();
    }
    let before: Vec<_> = guarded
        .net()
        .parameters()
        .iter()
        .map(|p| p.to_cpu().unwrap().to_vec())
        .collect();
    batch.size = 4;
    batch.returns.data_mut().fill(1.);
    batch.old_log_probs.data_mut().fill(-std::f32::consts::LN_2);
    assert!(matches!(
        guarded.train_on_batch_with_rng(&batch, &mut rng),
        Err(rustforge_rl::agent::gpu_ppo::GpuPpoError::Loss(
            rustforge_rl::agent::gpu_ppo::GpuPpoLossError::NonFinite
        ))
    ));
    assert_eq!(guarded.updates(), 0);
    for (p, b) in guarded.net().parameters().iter().zip(before) {
        close(&p.to_cpu().unwrap().to_vec(), &b, 0.);
    }
}
