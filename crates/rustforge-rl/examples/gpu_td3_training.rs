//! Fresh environment collection, seeded continuous replay and GPU TD3 learning.
#[path = "support/gpu_td3_target.rs"]
mod target;
use rustforge_tensor::gpu::GpuContext;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let r = target::learn(&context)?;
    println!("GPU TD3 fresh continuous episodes: cost {:.6} -> {:.6}; action {:.6}; {} critic / {} actor updates", r.initial_cost, r.final_cost, r.action, r.critic_updates, r.actor_updates);
    assert!(r.final_cost < 0.01 && r.final_cost < r.initial_cost * 0.1);
    assert!((r.action - 0.5).abs() < 0.1);
    assert_eq!((r.critic_updates, r.actor_updates), (536, 268));
    Ok(())
}
