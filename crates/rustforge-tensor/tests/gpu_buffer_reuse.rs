#![cfg(feature = "gpu")]
use rustforge_tensor::{gpu::GpuContext, Tensor};

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn queued_uniform_rewrites_preserve_operations_and_saved_tensors() {
    let context = GpuContext::new().unwrap().with_profiling();
    let original = context
        .upload(&Tensor::from_vec(vec![-2., 1., 3., 4.], &[2, 2]))
        .unwrap();
    let identity = context
        .upload(&Tensor::from_vec(vec![1., 0., 0., 1.], &[2, 2]))
        .unwrap();
    let mut value = context.scale_device(&original, 2.).unwrap();
    // Alternate kernels and uniform sizes without any intermediate host wait.
    for _ in 0..32 {
        value = context.scale_device(&value, 0.5).unwrap();
        value = context.matmul_device(&value, &identity).unwrap();
        value = context.scale_device(&value, 2.).unwrap();
    }
    assert_eq!(context.profile_snapshot().unwrap().counters.host_waits, 0);
    assert_eq!(
        context.download(&value).unwrap().to_vec(),
        vec![-4., 2., 6., 8.]
    );
    assert_eq!(
        context.download(&original).unwrap().to_vec(),
        vec![-2., 1., 3., 4.]
    );
    let c = context.profile_snapshot().unwrap().counters;
    assert!(c.parameter_allocations <= 2);
    assert_eq!(
        c.parameter_allocations + c.parameter_reuses,
        c.compute_dispatches
    );
    assert_eq!(c.readback_allocations, 1);
    assert_eq!(c.readback_reuses, 1);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn staging_reuse_handles_sizes_scalar_packs_and_integer_indices() {
    let context = GpuContext::new().unwrap().with_profiling();
    let small = context.upload(&Tensor::full(&[], -0.0)).unwrap();
    let large = context
        .upload(&Tensor::from_vec(
            (0..128).map(|i| i as f32).collect(),
            &[128],
        ))
        .unwrap();
    let indices = context.upload_indices(&[127, 3, 0], 128).unwrap();
    assert_eq!(
        context.download(&small).unwrap().item().to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(
        context.download(&large).unwrap().to_vec(),
        (0..128).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(context.download_indices(&indices).unwrap(), vec![127, 3, 0]);
    let pack = vec![&small; 24]; // Fits the larger cached allocation, maps only 96 bytes.
    assert_eq!(context.download_scalars(&pack).unwrap().len(), 24);
    assert_eq!(
        context.download_scalars(&[&small]).unwrap()[0].to_bits(),
        (-0.0f32).to_bits()
    );
    let c = context.profile_snapshot().unwrap().counters;
    assert_eq!(c.readbacks, 5);
    assert_eq!(c.host_waits, 5);
    assert_eq!(c.readback_allocations, 2);
    assert_eq!(c.readback_reuses, 3);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn concurrent_context_clones_keep_uniform_and_readback_leases_exclusive() {
    let context = GpuContext::new().unwrap().with_profiling();
    std::thread::scope(|scope| {
        for worker in 1..=4 {
            let context = context.clone();
            scope.spawn(move || {
                let input = context.upload(&Tensor::full(&[32], worker as f32)).unwrap();
                for scale in 1..=24 {
                    let output = context.scale_device(&input, scale as f32).unwrap();
                    assert_eq!(
                        context.download(&output).unwrap().to_vec(),
                        vec![(worker * scale) as f32; 32]
                    );
                }
            });
        }
    });
    let c = context.profile_snapshot().unwrap().counters;
    assert_eq!(c.compute_dispatches, 96);
    assert_eq!(c.readbacks, 96);
    assert_eq!(c.parameter_allocations + c.parameter_reuses, 96);
    assert_eq!(c.readback_allocations + c.readback_reuses, 96);
    assert!(c.parameter_reuses > 0 && c.readback_reuses > 0);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn oversized_staging_buffers_are_discarded_after_readback() {
    let context = GpuContext::new().unwrap().with_profiling();
    let large = context.zeros(&[2_100_000]).unwrap(); // Above the 8 MiB idle-cache budget.
    for _ in 0..2 {
        let result = context.download(&large).unwrap();
        assert_eq!(result.numel(), 2_100_000);
        assert!(result.to_vec().iter().all(|v| *v == 0.));
    }
    let c = context.profile_snapshot().unwrap().counters;
    assert_eq!(c.readback_allocations, 2);
    assert_eq!(c.readback_reuses, 0);
}
