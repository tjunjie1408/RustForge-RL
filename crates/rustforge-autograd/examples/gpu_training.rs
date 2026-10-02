//! Deterministic linear regression with resident data, gradients and SGD state.
use rustforge_autograd::gpu::{GpuSgd, GpuVariable};
use rustforge_tensor::{gpu::GpuContext, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    let x = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![1., 0., 0., 1., 1., 1., -1., 1.], &[4, 2]),
        false,
    )?;
    let y = GpuVariable::new(
        &context,
        &Tensor::from_vec(vec![2., -3., -1., -5.], &[4, 1]),
        false,
    )?;
    let w = GpuVariable::new(&context, &Tensor::zeros(&[2, 1]), true)?;
    let mut optimizer = GpuSgd::new(vec![w.clone()], 0.1, 0.8)?;
    let initial = x.matmul(&w)?.mse_loss(&y)?.to_cpu()?.to_vec()[0];
    for _ in 0..100 {
        optimizer.zero_grad();
        x.matmul(&w)?.mse_loss(&y)?.backward()?;
        optimizer.step()?;
    }
    let final_loss = x.matmul(&w)?.mse_loss(&y)?.to_cpu()?.to_vec()[0];
    println!(
        "MSE {initial:.6} -> {final_loss:.8}; weights {:?}",
        w.to_cpu()?.to_vec()
    );
    assert!(final_loss.is_finite() && final_loss < initial * 0.0001);
    Ok(())
}
