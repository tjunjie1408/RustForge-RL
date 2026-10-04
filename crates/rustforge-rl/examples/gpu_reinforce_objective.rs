//! Fixed policy-gradient optimization, without environment learning claims.
use rustforge_autograd::gpu::{GpuAdam, GpuVariable};
use rustforge_rl::agent::gpu_reinforce::{reinforce_loss, GpuReinforceError, GpuReinforceNet};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::rc::Rc;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let net = GpuReinforceNet::new_seeded(&context, 4, 8, 2, 42)?;
    let states = GpuVariable::new(&context, &Tensor::eye(4), false)?;
    let actions = Rc::new(context.upload_indices(&[0, 0, 1, 1], 2)?);
    let advantages = GpuVariable::new(&context, &Tensor::ones(&[4, 1]), false)?;
    let mut adam = GpuAdam::new(net.parameters(), 0.03)?;
    let objective = || -> Result<_, GpuReinforceError> {
        reinforce_loss(&net.forward(&states)?, &actions, &advantages, false)
    };
    let initial = objective()?.checked_loss()?;
    for _ in 0..80 {
        let loss = objective()?;
        loss.checked_loss()?;
        adam.zero_grad();
        loss.loss.backward()?;
        for p in net.parameters() {
            if let Some(g) = p.grad() {
                let square = context.mul_device(&g, &g)?;
                for tensor in [&*g, &square] {
                    if context
                        .download(&context.nonfinite_count_device(tensor)?)?
                        .item()
                        != 0.
                    {
                        return Err(GpuReinforceError::NonFinite.into());
                    }
                }
            }
        }
        adam.step()?;
    }
    let final_loss = objective()?.checked_loss()?;
    assert!(final_loss < initial && final_loss < 0.03);
    println!("Fixed GPU REINFORCE objective: {initial:.6} -> {final_loss:.6}; 80 Adam updates");
    Ok(())
}
