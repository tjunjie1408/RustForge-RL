//! Seeded environment rollouts and GPU PPO learning on a one-step bandit.
use rand::{rngs::StdRng, SeedableRng};
use rustforge_autograd::{gpu::GpuVariable, no_grad};
use rustforge_rl::{
    agent::{gpu_ppo::GpuPpoDiscrete, PPOConfig, PPODiscreteConfig},
    env::{Environment, Space},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
struct Bandit;
impl Environment for Bandit {
    type Obs = [f32; 2];
    type Act = usize;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        ([1., 0.], ())
    }
    fn step(&mut self, action: usize) -> (Self::Obs, f32, bool, bool, ()) {
        (
            [1., 0.],
            if action == 0 { 1. } else { -1. },
            true,
            false,
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let config = PPODiscreteConfig {
        base: PPOConfig {
            obs_dim: 2,
            hidden_dim: 8,
            lr: 0.03,
            ppo_epochs: 3,
            mini_batch_size: 32,
            ..PPOConfig::default()
        },
        num_actions: 2,
    };
    let mut agent = GpuPpoDiscrete::new_seeded(&context, config, 42)?;
    let state = GpuVariable::new(&context, &Tensor::from_vec(vec![1., 0.], &[1, 2]), false)?;
    let probability = |agent: &GpuPpoDiscrete| -> Result<f32, Box<dyn std::error::Error>> {
        Ok(no_grad(|| agent.net().forward(&state))?
            .0
            .softmax()?
            .to_cpu()?
            .to_vec()[0])
    };
    let initial = probability(&agent)?;
    let mut sampling = StdRng::seed_from_u64(7);
    let mut shuffle = StdRng::seed_from_u64(8);
    for _ in 0..20 {
        let batch = agent.collect_rollout_with_rng(&mut Bandit, 32, 1, Some(99), &mut sampling)?;
        agent.train_on_batch_with_rng(&batch, &mut shuffle)?;
    }
    let final_p = probability(&agent)?;
    assert!(final_p > 0.95 && final_p > initial + 0.2);
    println!("GPU PPO bandit: rewarding action probability {initial:.6} -> {final_p:.6}; {} Adam updates",agent.updates());
    Ok(())
}
