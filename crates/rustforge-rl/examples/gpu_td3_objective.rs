//! Fixed twin-critic TD3 optimization, without replay or environment learning.
use rustforge_autograd::gpu::{GpuAdam, GpuVariable};
use rustforge_nn::gpu::{GpuLinear, GpuModule};
use rustforge_rl::agent::gpu_td3::{td3_critic_loss, GpuTd3Error, GpuTd3LossConfig};
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
    let mut parameters = first.parameters();
    parameters.extend(second.parameters());
    let mut adam = GpuAdam::new(parameters.clone(), 0.03)?;
    let objective = || -> Result<_, Box<dyn std::error::Error>> {
        Ok(td3_critic_loss(
            &first.forward(&states)?,
            &second.forward(&states)?,
            &t1,
            &t2,
            &rewards,
            &dones,
            GpuTd3LossConfig { gamma: 0.9 },
        )?)
    };
    let initial = objective()?.checked_metrics()?.total_loss;
    for _ in 0..200 {
        let objective = objective()?;
        objective.checked_metrics()?;
        adam.zero_grad();
        objective.total_loss.backward()?;
        for p in &parameters {
            if let Some(g) = p.grad() {
                let square = context.mul_device(&g, &g)?;
                for tensor in [&*g, &square] {
                    if context
                        .download(&context.nonfinite_count_device(tensor)?)?
                        .item()
                        != 0.
                    {
                        return Err(GpuTd3Error::NonFinite.into());
                    }
                }
            }
        }
        adam.step()?;
    }
    let final_loss = objective()?.checked_metrics()?.total_loss;
    println!(
        "Fixed GPU TD3 twin-critic objective: {initial:.6} -> {final_loss:.9e}; 200 Adam updates"
    );
    assert!(final_loss < initial && final_loss < 0.001);
    Ok(())
}
