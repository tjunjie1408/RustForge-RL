//! Reproducible CPU/reference/tiled comparison, including adapter metadata.
//! GPU timings include command submission and completion, excluding transfers.

use std::{hint::black_box, time::Instant};

use rustforge_tensor::{
    gpu::{GpuContext, GpuTensor, MatmulKernel},
    Tensor,
};

fn gpu_batch(
    context: &GpuContext,
    left: &GpuTensor,
    right: &GpuTensor,
    output: &mut GpuTensor,
    iterations: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    context.synchronize();
    let start = Instant::now();
    for _ in 0..iterations {
        context.matmul_into(left, right, output)?;
    }
    context.synchronize();
    Ok(start.elapsed().as_secs_f64() * 1000. / f64::from(iterations))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(10);
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let sizes = std::env::args()
        .nth(2)
        .map(|value| {
            value
                .split(',')
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![32, 64, 128, 256]);
    if sizes.is_empty() || sizes.contains(&0) {
        return Err("matrix sizes must be positive".into());
    }
    let context = GpuContext::new()?;
    let naive = context.with_matmul_kernel(MatmulKernel::Naive);
    let tiled = context.with_matmul_kernel(MatmulKernel::Tiled);
    println!("# adapter: {:?}", context.adapter_info());
    println!("# GPU buffers reused; upload/download and pipeline initialization excluded");
    println!(
        "# CPU Tensor::matmul includes output allocation; three trials, alternating GPU order"
    );
    println!("n,iterations,trial,cpu_ms,naive_ms,tiled_ms");
    for n in sizes {
        let left = Tensor::rand_uniform(&[n, n], -1., 1., Some(123));
        let right = Tensor::rand_uniform(&[n, n], -1., 1., Some(124));
        let expected = left.matmul(&right);
        let left_device = context.upload(&left)?;
        let right_device = context.upload(&right)?;
        let mut naive_output = context.zeros(&[n, n])?;
        let mut tiled_output = context.zeros(&[n, n])?;
        for _ in 0..3 {
            naive.matmul_into(&left_device, &right_device, &mut naive_output)?;
            tiled.matmul_into(&left_device, &right_device, &mut tiled_output)?;
        }
        context.synchronize();
        for trial in 0..3 {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(left.matmul(&right));
            }
            let cpu_ms = start.elapsed().as_secs_f64() * 1000. / f64::from(iterations);
            let (naive_ms, tiled_ms) = if trial % 2 == 0 {
                let naive_ms = gpu_batch(
                    &naive,
                    &left_device,
                    &right_device,
                    &mut naive_output,
                    iterations,
                )?;
                let tiled_ms = gpu_batch(
                    &tiled,
                    &left_device,
                    &right_device,
                    &mut tiled_output,
                    iterations,
                )?;
                (naive_ms, tiled_ms)
            } else {
                let tiled_ms = gpu_batch(
                    &tiled,
                    &left_device,
                    &right_device,
                    &mut tiled_output,
                    iterations,
                )?;
                let naive_ms = gpu_batch(
                    &naive,
                    &left_device,
                    &right_device,
                    &mut naive_output,
                    iterations,
                )?;
                (naive_ms, tiled_ms)
            };
            // Verify measured outputs, outside the timed region.
            for output in [&naive_output, &tiled_output] {
                let actual = context.download(output)?;
                if actual.shape() != expected.shape() {
                    return Err(format!("output shape mismatch at n={n}").into());
                }
                for (actual, expected) in actual.to_vec().iter().zip(expected.to_vec()) {
                    if !actual.is_finite()
                        || !expected.is_finite()
                        || (*actual - expected).abs() > 1e-4 + 1e-4 * expected.abs()
                    {
                        return Err(
                            format!("numerical mismatch at n={n}: {actual} vs {expected}").into(),
                        );
                    }
                }
            }
            println!("{n},{iterations},{trial},{cpu_ms:.6},{naive_ms:.6},{tiled_ms:.6}");
        }
    }
    Ok(())
}
