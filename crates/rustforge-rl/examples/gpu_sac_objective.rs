//! Fixed SAC twin-critic optimization; replay/agent/environment learning follows separately.
use rustforge_autograd::gpu::{GpuAdam, GpuVariable};
use rustforge_nn::gpu::{GpuLinear, GpuModule};
use rustforge_rl::agent::gpu_sac::{
    sac_critic_loss, GpuSacCriticInputs, GpuSacError, GpuSacLossConfig,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let first = GpuLinear::new_seeded(&context, 2, 1, 42)?;
    let second = GpuLinear::new_seeded(&context, 2, 1, 43)?;
    let states = GpuVariable::new(&context, &Tensor::eye(2), false)?;
    let upload = |v| GpuVariable::new(&context, &Tensor::from_vec(v, &[2, 1]), false);
    let rewards = upload(vec![1., -1.])?;
    let dones = upload(vec![0., 1.])?;
    let t1 = upload(vec![0.5, 2.])?;
    let t2 = upload(vec![1., 3.])?;
    let next_log_probs = upload(vec![-0.5, -1.])?;
    let mut parameters = first.parameters();
    parameters.extend(second.parameters());
    let mut adam = GpuAdam::new(parameters.clone(), 0.03)?;
    let objective = || -> Result<_, Box<dyn std::error::Error>> {
        Ok(sac_critic_loss(
            GpuSacCriticInputs {
                q1: &first.forward(&states)?,
                q2: &second.forward(&states)?,
                target_q1: &t1,
                target_q2: &t2,
                next_log_probs: &next_log_probs,
                rewards: &rewards,
                dones: &dones,
            },
            GpuSacLossConfig {
                gamma: 0.9,
                alpha: 0.2,
            },
        )?)
    };
    let initial = objective()?.checked_metrics()?.total_loss;
    for _ in 0..200 {
        let objective = objective()?;
        objective.checked_metrics()?;
        adam.zero_grad();
        objective.total_loss.backward()?;
        for p in &parameters {
            let g = p.grad().ok_or(GpuSacError::InvalidBatch)?;
            let square = context.mul_device(&g, &g)?;
            for tensor in [&*g, &square] {
                if context
                    .download(&context.nonfinite_count_device(tensor)?)?
                    .item()
                    != 0.
                {
                    return Err(GpuSacError::NonFinite.into());
                }
            }
        }
        adam.step()?;
    }
    let final_loss = objective()?.checked_metrics()?.total_loss;
    println!("Fixed GPU SAC twin-critic objective: {initial:.6} -> {final_loss:.9e}; 200 Adam updates; targets [1.54,-1]");
    assert!(final_loss < initial && final_loss < 0.001);
    Ok(())
}
