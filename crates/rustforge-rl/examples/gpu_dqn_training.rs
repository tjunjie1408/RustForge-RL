//! Seeded Double DQN on a deterministic two-state environment with bootstrapping.
#[path = "support/gpu_chain.rs"]
mod gpu_chain;
use gpu_chain::TinyChain;
use rustforge_rl::{
    agent::{DQNConfig, GpuDqn},
    env::Environment,
};
use rustforge_tensor::gpu::GpuContext;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let mut agent = GpuDqn::new_seeded(
        &context,
        DQNConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 8,
            lr: 0.03,
            gamma: 0.9,
            target_update_freq: 10,
            double_dqn: true,
            ..DQNConfig::default()
        },
        42,
    )?;
    let mut env = TinyChain::default();
    let replay = env.full_replay();
    let batch = agent.upload_batch(&replay)?;
    let initial = agent.train_device_batch(&batch)?;
    let mut final_loss = initial;
    for _ in 0..299 {
        final_loss = agent.train_device_batch(&batch)?;
    }
    let mut actions = Vec::new();
    let (mut state, _) = env.reset(Some(0));
    let mut episode_return = 0.;
    loop {
        let action = agent.select_greedy_action(&state)?;
        actions.push(action);
        let (next, reward, terminated, truncated, _) = env.step(action);
        episode_return += reward;
        state = next;
        if terminated || truncated {
            break;
        }
        assert!(actions.len() < 3);
    }
    println!("GPU Double DQN MSE {initial:.6} -> {final_loss:.8}; greedy actions {actions:?}; return {episode_return}");
    assert!(final_loss < 0.001 && final_loss < initial * 0.01);
    assert_eq!(actions, vec![0, 1]);
    assert_eq!(episode_return, 1.);
    Ok(())
}
