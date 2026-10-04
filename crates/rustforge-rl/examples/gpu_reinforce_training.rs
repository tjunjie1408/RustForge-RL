//! Learns the rewarding categorical action from fresh environment rollouts.
use rand::{rngs::StdRng, SeedableRng};
use rustforge_rl::{
    agent::{
        gpu_reinforce::{GpuReinforce, GpuReinforceRolloutOptions},
        REINFORCEConfig,
    },
    env::{Environment, Space},
};
use rustforge_tensor::gpu::GpuContext;
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
fn main() {
    let context = GpuContext::new().unwrap();
    let mut c = REINFORCEConfig {
        obs_dim: 2,
        hidden_dim: 8,
        num_actions: 2,
        ..Default::default()
    };
    c.lr = 0.03;
    let mut agent = GpuReinforce::new_seeded(&context, c, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let initial = agent.action_probabilities(&[1., 0.]).unwrap()[0];
    for _ in 0..60 {
        let batch = agent
            .collect_rollout_with_rng(
                &mut Bandit,
                GpuReinforceRolloutOptions {
                    episodes: 32,
                    max_steps: 1,
                    seed: Some(99),
                },
                &mut rng,
            )
            .unwrap();
        assert!(agent.train_on_rollout(&batch).unwrap().is_finite());
    }
    let final_p = agent.action_probabilities(&[1., 0.]).unwrap()[0];
    println!("GPU REINFORCE bandit: probability {initial:.6} -> {final_p:.6}");
    assert!(final_p > 0.95 && final_p > initial + 0.2);
    assert_eq!(agent.updates(), 60);
}
