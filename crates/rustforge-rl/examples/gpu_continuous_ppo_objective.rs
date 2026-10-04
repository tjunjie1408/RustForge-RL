//! Fixed continuous PPO objective optimization; no environment or rollout trainer.
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad,
};
use rustforge_nn::gpu::{GpuLinear, GpuModule};
use rustforge_rl::agent::{
    gpu_gaussian::GpuGaussianTransform,
    gpu_ppo::{continuous_ppo_loss, GpuContinuousPpoInputs},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let actor = GpuLinear::new_seeded(&context, 2, 1, 42)?;
    let std = GpuLinear::new_seeded(&context, 2, 1, 43)?;
    let critic = GpuLinear::new_seeded(&context, 2, 1, 44)?;
    let transform = GpuGaussianTransform::new(&context, &[-1.], &[1.])?;
    let input = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![1., 0., 0., 1., 1., 0., 0., 1.], &[4, 2]),
        false,
    )?;
    let upload = |data| GpuVariable::new(&context, &Tensor::from_vec(data, &[4, 1]), false);
    let actions = upload(vec![0.5, -0.5, -0.5, 0.5])?;
    let advantages = upload(vec![1., -1., 1., -1.])?;
    let returns = upload(vec![1., -1., 1., -1.])?;
    let old = no_grad(|| -> Result<_, Box<dyn std::error::Error>> {
        Ok(transform
            .log_prob_from_action(&actor.forward(&input)?, &std.forward(&input)?, &actions)?
            .log_probs)
    })?;
    let mut params = actor.parameters();
    params.extend(std.parameters());
    let mut actor_optimizer = GpuAdam::new(params, 0.03)?;
    let mut critic_optimizer = GpuAdam::new(critic.parameters(), 0.03)?;
    let objective = || -> Result<_, Box<dyn std::error::Error>> {
        Ok(continuous_ppo_loss(
            GpuContinuousPpoInputs {
                mean: &actor.forward(&input)?,
                raw_log_std: &std.forward(&input)?,
                values: &critic.forward(&input)?,
                actions: &actions,
                old_log_probs: &old,
                advantages: &advantages,
                returns: &returns,
            },
            &transform,
            0.2,
        )?)
    };
    let initial = objective()?.checked_metrics()?;
    for _ in 0..80 {
        let loss = objective()?;
        loss.checked_metrics()?;
        actor_optimizer.zero_grad();
        critic_optimizer.zero_grad();
        loss.policy_loss.backward()?;
        loss.value_loss.backward()?;
        actor_optimizer.step()?;
        critic_optimizer.step()?;
    }
    let final_metrics = objective()?.checked_metrics()?;
    assert!(final_metrics.policy_loss < initial.policy_loss - 0.1);
    assert!(final_metrics.value_loss < 0.01);
    println!(
        "Fixed continuous GPU PPO: policy {:.6} -> {:.6}, value {:.6} -> {:.6}",
        initial.policy_loss,
        final_metrics.policy_loss,
        initial.value_loss,
        final_metrics.value_loss
    );
    Ok(())
}
