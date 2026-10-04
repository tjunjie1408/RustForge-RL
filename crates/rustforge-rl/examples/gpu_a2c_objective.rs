//! Fixed supervised A2C objective optimization; environment rollouts follow later.
use rustforge_autograd::gpu::{GpuAdam, GpuVariable};
use rustforge_rl::agent::gpu_a2c::{a2c_loss, GpuA2cLossConfig, GpuA2cLossError, GpuA2cNet};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::rc::Rc;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let net = GpuA2cNet::new_seeded(&context, 4, 8, 2, 42)?;
    let states = GpuVariable::new(&context, &Tensor::eye(4), false)?;
    let actions = Rc::new(context.upload_indices(&[0, 0, 1, 1], 2)?);
    let advantages = GpuVariable::new(&context, &Tensor::ones(&[4, 1]), false)?;
    let returns = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![1., -1., 0.5, -0.5], &[4, 1]),
        false,
    )?;
    let mut optimizer = GpuAdam::new(net.parameters(), 0.03)?;
    let objective = || -> Result<_, Box<dyn std::error::Error>> {
        let (logits, values) = net.forward(&states)?;
        Ok(a2c_loss(
            &logits,
            &values,
            &actions,
            &advantages,
            &returns,
            GpuA2cLossConfig::default(),
        )?)
    };
    let initial = objective()?.checked_metrics()?;
    for _ in 0..80 {
        let loss = objective()?;
        loss.checked_metrics()?;
        optimizer.zero_grad();
        loss.total_loss.backward()?;
        for p in net.parameters() {
            if let Some(gradient) = p.grad() {
                if context
                    .download(&context.nonfinite_count_device(&gradient)?)?
                    .item()
                    != 0.
                {
                    return Err(GpuA2cLossError::NonFinite.into());
                }
            }
        }
        optimizer.step()?;
    }
    let final_metrics = objective()?.checked_metrics()?;
    assert!(final_metrics.total_loss < initial.total_loss);
    assert!(final_metrics.actor_loss < 0.05 && final_metrics.value_loss < 0.01);
    println!(
        "Fixed GPU A2C minibatch: total {:.6} -> {:.6}; actor {:.6}; value {:.6}",
        initial.total_loss,
        final_metrics.total_loss,
        final_metrics.actor_loss,
        final_metrics.value_loss
    );
    Ok(())
}
