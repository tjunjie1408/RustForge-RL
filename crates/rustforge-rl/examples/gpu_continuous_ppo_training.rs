use rand::{rngs::StdRng, SeedableRng};
use rustforge_rl::{
    agent::{
        gpu_ppo::{GpuContinuousRolloutOptions, GpuPpoContinuous},
        PPOConfig, PPOContinuousConfig,
    },
    env::{Environment, Space},
};
use rustforge_tensor::gpu::GpuContext;
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
fn main() {
    let context = GpuContext::new().unwrap();
    let mut c = PPOContinuousConfig {
        base: PPOConfig {
            obs_dim: 2,
            hidden_dim: 8,
            ppo_epochs: 3,
            ..PPOConfig::default()
        },
        act_dim: 1,
        action_low: vec![-1.],
        action_high: vec![1.],
    };
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
