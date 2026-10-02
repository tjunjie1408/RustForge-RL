//! Seeded nonlinear regression with device-resident modules and Adam state.
use rustforge_autograd::gpu::{GpuAdam, GpuVariable};
use rustforge_nn::gpu::{GpuLinear, GpuModule, GpuReLU, GpuSequential};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let x = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![-2., -1., 0., 1., 2.], &[5, 1]),
        false,
    )?;
    let target = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![3., 2., 1., 2., 3.], &[5, 1]),
        false,
    )?;
    let model = GpuSequential::new(vec![
        Box::new(GpuLinear::new_seeded(&context, 1, 8, 42)?),
        Box::new(GpuReLU),
        Box::new(GpuLinear::new_seeded(&context, 8, 1, 43)?),
    ]);
    let mut optimizer = GpuAdam::new(model.parameters(), 0.03)?;
    let initial = model.forward(&x)?.mse_loss(&target)?.to_cpu()?.item();
    for _ in 0..250 {
        optimizer.zero_grad();
        model.forward(&x)?.mse_loss(&target)?.backward()?;
        optimizer.step()?;
    }
    let final_loss = model.forward(&x)?.mse_loss(&target)?.to_cpu()?.item();
    println!("Seeded MLP MSE {initial:.6} -> {final_loss:.8}");
    assert!(final_loss.is_finite() && final_loss < 0.001 && final_loss < initial * 0.01);
    Ok(())
}
