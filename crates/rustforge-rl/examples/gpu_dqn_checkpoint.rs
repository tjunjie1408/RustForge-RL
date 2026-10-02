//! Save/resume demonstration, including delayed target synchronization.
#[path = "support/gpu_chain.rs"]
mod gpu_chain;
use rustforge_rl::agent::{DQNConfig, GpuDqn};
use rustforge_tensor::gpu::GpuContext;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: gpu_dqn_checkpoint <checkpoint-path>")?;
    let context = GpuContext::new()?;
    let mut agent = GpuDqn::new_seeded(
        &context,
        DQNConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 8,
            lr: 0.03,
            gamma: 0.9,
            target_update_freq: 5,
            double_dqn: true,
            ..DQNConfig::default()
        },
        42,
    )?;
    let replay = gpu_chain::TinyChain::default().full_replay();
    let batch = agent.upload_batch(&replay)?;
    for _ in 0..7 {
        agent.train_device_batch(&batch)?;
    }
    agent.save_checkpoint(&path)?;
    let mut resumed = GpuDqn::load_checkpoint(&context, &path)?;
    let mut loss = 0.;
    for _ in 0..8 {
        let uninterrupted = agent.train_device_batch(&batch)?;
        loss = resumed.train_device_batch(&batch)?;
        assert_eq!(loss.to_bits(), uninterrupted.to_bits());
    }
    assert_eq!(resumed.train_steps(), agent.train_steps());
    println!(
        "Saved at step 7; resumed updates match through step {}, loss {loss:.6}",
        resumed.train_steps()
    );
    Ok(())
}
