//! Optimizes a fixed categorical PPO minibatch on device; not a rollout trainer.
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad,
};
use rustforge_nn::gpu::{GpuLinear, GpuModule};
use rustforge_rl::agent::gpu_ppo::{discrete_ppo_loss, GpuPpoLossConfig};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::rc::Rc;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let actor = GpuLinear::new_seeded(&context, 4, 2, 42)?;
    let critic = GpuLinear::new_seeded(&context, 4, 1, 43)?;
    let states = GpuVariable::new(
        &context,
        &Tensor::from_vec(
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            &[4, 4],
        ),
        false,
    )?;
    let actions = Rc::new(context.upload_indices(&[0, 0, 1, 1], 2)?);
    let initial_logits = no_grad(|| actor.forward(&states))?;
    let old = initial_logits.log_softmax()?.gather_actions(&actions)?;
    let advantages = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![1., -1., 1., -1.], &[4, 1]),
        false,
    )?;
    let returns = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![1., -1., 0.5, -0.5], &[4, 1]),
        false,
    )?;
    let mut parameters = actor.parameters();
    parameters.extend(critic.parameters());
    let mut optimizer = GpuAdam::new(parameters, 0.03)?;
    let objective = || -> Result<_, Box<dyn std::error::Error>> {
        Ok(discrete_ppo_loss(
            &actor.forward(&states)?,
            &critic.forward(&states)?,
            &actions,
            &old,
            &advantages,
            &returns,
            GpuPpoLossConfig::default(),
        )?)
    };
    let initial = objective()?.checked_metrics()?;
    for _ in 0..80 {
        let loss = objective()?;
        loss.checked_metrics()?;
        optimizer.zero_grad();
        loss.total_loss.backward()?;
        optimizer.step()?;
    }
    let final_metrics = objective()?.checked_metrics()?;
    assert!(final_metrics.total_loss < initial.total_loss);
    assert!(final_metrics.value_loss < 0.01);
    println!(
        "Fixed GPU PPO minibatch: total loss {:.6} -> {:.6}, value loss {:.6}",
        initial.total_loss, final_metrics.total_loss, final_metrics.value_loss
    );
    Ok(())
}
