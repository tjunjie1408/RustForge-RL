#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, SeedableRng};
use rustforge_autograd::{gpu::GpuVariable, no_grad};
use rustforge_nn::gpu::GpuModule;
use rustforge_rl::{
    agent::{
        gpu_ppo::{GpuContinuousRolloutOptions, GpuPpoContinuous, GpuPpoError},
        PPOConfig, PPOContinuous, PPOContinuousConfig,
    },
    buffer::ContinuousRolloutBatch,
    env::{Environment, Space},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn config() -> PPOContinuousConfig {
    PPOContinuousConfig {
        base: PPOConfig {
            obs_dim: 2,
            hidden_dim: 8,
            lr: 0.01,
            ppo_epochs: 3,
            mini_batch_size: 3,
            ..PPOConfig::default()
        },
        act_dim: 1,
        action_low: vec![-1.],
        action_high: vec![1.],
    }
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
fn parameters(agent: &GpuPpoContinuous) -> Vec<Vec<f32>> {
    agent
        .actor()
        .parameters()
        .into_iter()
        .chain(agent.critic().parameters())
        .map(|p| p.to_cpu().unwrap().to_vec())
        .collect()
}
struct Target {
    terminal: bool,
    truncated: bool,
    steps: usize,
}
impl Environment for Target {
    type Obs = [f32; 2];
    type Act = f32;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        ([1., 0.], ())
    }
    fn step(&mut self, a: f32) -> (Self::Obs, f32, bool, bool, ()) {
        assert!((-1. ..=1.).contains(&a));
        self.steps += 1;
        (
            [1., 0.],
            -(a - 0.5).powi(2),
            self.terminal,
            self.truncated,
            (),
        )
    }
    fn action_space(&self) -> Space {
        Space::continuous(vec![-1.], vec![1.])
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.; 2], vec![1.; 2])
    }
}
fn target(terminal: bool, truncated: bool) -> Target {
    Target {
        terminal,
        truncated,
        steps: 0,
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_actor_critic_sampling_and_partial_minibatch_updates_match_cpu() {
    let context = GpuContext::new().unwrap();
    let mut gpu = GpuPpoContinuous::new_seeded(&context, config(), 42).unwrap();
    let mut cpu = PPOContinuous::new_seeded(config(), 42);
    for (g, c) in gpu.actor().parameters().iter().zip(cpu.actor.parameters()) {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
    }
    for (g, c) in gpu
        .critic()
        .parameters()
        .iter()
        .zip(rustforge_nn::Module::parameters(cpu.critic()))
    {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
    }
    let mut r1 = StdRng::seed_from_u64(9);
    let mut r2 = r1.clone();
    for _ in 0..10 {
        let (a, lp, v) = gpu.select_action_with_rng(&[0.2, 0.7], &mut r1).unwrap();
        let (b, cl, cv) = cpu.select_action_with_rng(&[0.2, 0.7], &mut r2);
        close(&a, &b, 1e-5);
        close(&[lp, v], &[cl, cv], 1e-4);
    }
    let mut batch = ContinuousRolloutBatch::new(6, 2, 1);
    batch.size = 5;
    batch.states = Tensor::from_vec(
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
    );
    batch.returns = Tensor::from_vec(vec![1., -0.2, 0.7, 0.1, -0.4, f32::NAN], &[6, 1]);
    batch.advantages = Tensor::from_vec(vec![1., -1., 0.3, -0.7, 0.2, f32::NAN], &[6, 1]);
    // Capture on-policy actions: the seeded policy has narrow state-dependent
    // variance, so arbitrary actions would create extreme importance ratios.
    let mut action_rng = StdRng::seed_from_u64(17);
    let state_values = batch.states.to_vec();
    let mut action_values: Vec<f32> = (0..5)
        .flat_map(|i| {
            cpu.select_action_with_rng(&state_values[i * 2..(i + 1) * 2], &mut action_rng)
                .0
        })
        .collect();
    action_values.push(f32::NAN);
    batch.actions = Tensor::from_vec(action_values, &[6, 1]);
    let states =
        rustforge_autograd::Variable::from_tensor(batch.states.slice_axis(0, 0, 5).unwrap());
    let actions =
        rustforge_autograd::Variable::from_tensor(batch.actions.slice_axis(0, 0, 5).unwrap());
    let mut lp = cpu
        .actor
        .log_prob_from_action(&states, &actions)
        .data()
        .to_vec();
    lp.push(f32::NAN);
    batch.old_log_probs = Tensor::from_vec(lp, &[6, 1]);
    for _ in 0..3 {
        let m = gpu.train_on_batch_with_rng(&batch, &mut r1).unwrap();
        let c = cpu.train_on_batch_with_rng(&batch, &mut r2);
        close(&[m.policy_loss, m.value_loss], &[c.0, c.1], 5e-4);
        for (g, c) in gpu.actor().parameters().iter().zip(cpu.actor.parameters()) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 5e-4);
        }
        for (g, c) in gpu
            .critic()
            .parameters()
            .iter()
            .zip(rustforge_nn::Module::parameters(cpu.critic()))
        {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 5e-4);
        }
    }
    assert_eq!((gpu.actor_updates(), gpu.critic_updates()), (18, 18));
    let mut c = config();
    c.act_dim = 2;
    c.action_low = vec![-1., -2.];
    c.action_high = vec![1., 3.];
    let gpu = GpuPpoContinuous::new_seeded(&context, c.clone(), 42).unwrap();
    let cpu = PPOContinuous::new_seeded(c, 42);
    let mut r1 = StdRng::seed_from_u64(29);
    let mut r2 = r1.clone();
    for _ in 0..5 {
        let (a, lp, v) = gpu.select_action_with_rng(&[0.2, 0.7], &mut r1).unwrap();
        let (b, cl, cv) = cpu.select_action_with_rng(&[0.2, 0.7], &mut r2);
        close(&a, &b, 1e-5);
        close(&[lp, v], &[cl, cv], 1e-4);
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn continuous_rollout_gae_bootstraps_truncation_and_limits_without_reset_leakage() {
    let context = GpuContext::new().unwrap();
    let agent = GpuPpoContinuous::new_seeded(&context, config(), 42).unwrap();
    for (i, p) in agent.critic().parameters().iter().enumerate() {
        let mut t = Tensor::zeros(p.data().shape());
        if i == 5 {
            t.data_mut().fill(2.);
        }
        p.copy_data_from(&GpuVariable::new(&context, &t, true).unwrap())
            .unwrap();
    }
    let options = GpuContinuousRolloutOptions {
        episodes: 3,
        max_steps: 5,
        seed: Some(1),
    };
    let mut a = StdRng::seed_from_u64(9);
    let mut b = a.clone();
    let terminal = agent
        .collect_rollout_with_rng(&mut target(true, false), options, &mut a, |x| Ok(x[0]))
        .unwrap();
    let truncated = agent
        .collect_rollout_with_rng(&mut target(false, true), options, &mut b, |x| Ok(x[0]))
        .unwrap();
    assert_eq!(terminal.size, 3);
    assert_eq!(truncated.size, 3);
    close(&terminal.actions.to_vec(), &truncated.actions.to_vec(), 0.);
    for (t, u) in terminal
        .returns
        .to_vec()
        .iter()
        .zip(truncated.returns.to_vec())
    {
        approx::assert_abs_diff_eq!(u - t, 1.98, epsilon = 1e-5);
    }
    let mut a = StdRng::seed_from_u64(9);
    let continuing = agent
        .collect_rollout_with_rng(
            &mut target(false, false),
            GpuContinuousRolloutOptions {
                max_steps: 1,
                ..options
            },
            &mut a,
            |x| Ok(x[0]),
        )
        .unwrap();
    close(
        &continuing.returns.to_vec(),
        &truncated.returns.to_vec(),
        1e-5,
    );
}
#[test]
#[ignore = "requires a GPU adapter"]
fn fresh_seeded_continuous_environment_rollouts_learn_target_action() {
    let context = GpuContext::new().unwrap();
    let mut c = config();
    c.base.lr = 0.01;
    c.base.mini_batch_size = 64;
    let mut agent = GpuPpoContinuous::new_seeded(&context, c, 42).unwrap();
    let evaluate = |agent: &GpuPpoContinuous| {
        let mut rng = StdRng::seed_from_u64(99);
        let batch = agent
            .collect_rollout_with_rng(
                &mut target(true, false),
                GpuContinuousRolloutOptions {
                    episodes: 128,
                    max_steps: 1,
                    seed: Some(1),
                },
                &mut rng,
                |x| Ok(x[0]),
            )
            .unwrap();
        -batch.returns.to_vec().iter().sum::<f32>() / 128.
    };
    let initial = evaluate(&agent);
    let mut sampling = StdRng::seed_from_u64(7);
    let mut shuffle = StdRng::seed_from_u64(8);
    for _ in 0..30 {
        let batch = agent
            .collect_rollout_with_rng(
                &mut target(true, false),
                GpuContinuousRolloutOptions {
                    episodes: 64,
                    max_steps: 1,
                    seed: Some(1),
                },
                &mut sampling,
                |x| Ok(x[0]),
            )
            .unwrap();
        let m = agent.train_on_batch_with_rng(&batch, &mut shuffle).unwrap();
        assert!(m.policy_loss.is_finite() && m.value_loss.is_finite());
    }
    let final_cost = evaluate(&agent);
    let action = agent.deterministic_action(&[1., 0.]).unwrap()[0];
    println!("Continuous GPU PPO: sampled MSE {initial:.6} -> {final_cost:.6}; deterministic action {action:.6}");
    assert!(final_cost < 0.02 && final_cost < initial * 0.1);
    assert!((action - 0.5).abs() < 0.1);
    assert_eq!((agent.actor_updates(), agent.critic_updates()), (90, 90));
}
#[test]
#[ignore = "requires a GPU adapter"]
fn invalid_rollouts_conversion_and_nonfinite_gradients_fail_before_updates() {
    let context = GpuContext::new().unwrap();
    let mut agent = GpuPpoContinuous::new_seeded(&context, config(), 42).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    let before = parameters(&agent);
    let options = GpuContinuousRolloutOptions {
        episodes: 4,
        max_steps: 1,
        seed: None,
    };
    let mut env = target(true, false);
    assert!(agent
        .collect_rollout_with_rng(&mut env, options, &mut rng, |_| Err(
            GpuPpoError::InvalidInput("rejected action")
        ))
        .is_err());
    assert_eq!(env.steps, 0);
    let mut batch = agent
        .collect_rollout_with_rng(&mut env, options, &mut rng, |a| Ok(a[0]))
        .unwrap();
    let actions = batch.actions.to_vec();
    batch.actions.data_mut().fill(2.);
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    batch.actions = Tensor::from_vec(actions, &[4, 1]);
    batch.advantages.data_mut().fill(f32::NAN);
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    batch.advantages = Tensor::from_vec(vec![1e20, -1e20, 1e20, -1e20], &[4, 1]);
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    batch.advantages.data_mut().fill(1.);
    assert!(no_grad(|| agent.train_on_batch_with_rng(&batch, &mut rng)).is_err());
    batch.size = 5;
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    batch.size = 0;
    assert_eq!(
        agent
            .train_on_batch_with_rng(&batch, &mut rng)
            .unwrap()
            .policy_loss,
        0.
    );
    assert_eq!(parameters(&agent), before);
    assert_eq!((agent.actor_updates(), agent.critic_updates()), (0, 0));
    assert!(agent.value_of(&[f32::NAN, 0.]).is_err());
    assert!(agent.select_action_with_rng(&[1.], &mut rng).is_err());
    let mut c = config();
    c.act_dim = 0;
    assert!(GpuPpoContinuous::new_seeded(&context, c, 42).is_err());
    // Finite forward value/loss with an overflowing final critic weight gradient.
    for (i, p) in agent.critic().parameters().iter().enumerate() {
        let mut t = Tensor::zeros(p.data().shape());
        if i == 3 {
            t.data_mut().fill(1e38);
        }
        p.copy_data_from(&GpuVariable::new(&context, &t, false).unwrap())
            .unwrap();
    }
    let before = parameters(&agent);
    batch.size = 4;
    batch.returns.data_mut().fill(2.);
    assert!(agent.train_on_batch_with_rng(&batch, &mut rng).is_err());
    assert_eq!(parameters(&agent), before);
    assert_eq!((agent.actor_updates(), agent.critic_updates()), (0, 0));
}
