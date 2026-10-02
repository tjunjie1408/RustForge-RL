use rustforge_tensor::{gpu::GpuContext, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new()?;
    println!("Adapter: {:?}", context.adapter_info());
    let left = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let right = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], &[3, 2]);
    let result = context.matmul(&left, &right)?;
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![58., 64., 139., 154.]);
    println!("GPU result: {result}");
    let left_device = context.upload(&left)?;
    let right_device = context.upload(&right)?;
    let identity_device = context.upload(&Tensor::eye(2))?;
    let intermediate = context.matmul_device(&left_device, &right_device)?;
    let mut output = context.zeros(&[2, 2])?;
    context.matmul_into(&intermediate, &identity_device, &mut output)?;
    // Only the final result is downloaded; the intermediate stays on device.
    let chained = context.download(&output)?;
    assert_eq!(chained.to_vec(), result.to_vec());
    println!("Chained device result: {chained}");
    let activated = context.relu_device(&output)?;
    let sum = context.sum_device(&activated)?;
    assert_eq!(context.download(&sum)?.item(), 415.);
    println!("Device ReLU → sum: 415");
    Ok(())
}
