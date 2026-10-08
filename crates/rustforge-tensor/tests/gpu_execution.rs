#![cfg(feature = "gpu")]
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn nested_batches_flush_at_readbacks_and_keep_parameters_distinct() {
    let c = GpuContext::new().unwrap().with_profiling();
    let x = c.upload(&Tensor::ones(&[4])).unwrap();
    let batch = c.command_batch();
    let a = c.scale_device(&x, 2.).unwrap();
    {
        let _nested = c.clone().command_batch();
        let b = c.scale_device(&a, 3.).unwrap();
        assert_eq!(c.profile_snapshot().unwrap().counters.submissions, 0);
        assert_eq!(c.download(&b).unwrap().to_vec(), vec![6.; 4]);
    }
    let d = c.scale_device(&a, -4.).unwrap();
    drop(batch);
    assert_eq!(c.download(&d).unwrap().to_vec(), vec![-8.; 4]);
    let counts = c.profile_snapshot().unwrap().counters;
    assert_eq!(
        (
            counts.compute_dispatches,
            counts.command_buffers,
            counts.submissions
        ),
        (3, 5, 4)
    );
    assert_eq!(counts.host_waits, 2);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn bounded_batches_reuse_uniforms_without_waits() {
    let c = GpuContext::new().unwrap().with_profiling();
    let mut x = c.upload(&Tensor::ones(&[8])).unwrap();
    let scope = c.command_batch();
    for _ in 0..96 {
        x = c.scale_device(&x, 1.).unwrap();
    }
    drop(scope);
    let p = c.profile_snapshot().unwrap().counters;
    assert_eq!(p.submissions, 3);
    assert_eq!(p.host_waits, 0);
    assert_eq!(p.parameter_allocations, 32);
    assert_eq!(p.parameter_reuses, 64);
    assert_eq!(c.download(&x).unwrap().to_vec(), vec![1.; 8]);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn failed_and_unwound_scopes_flush_prior_work_and_leave_no_active_batch() {
    let c = GpuContext::new().unwrap().with_profiling();
    let x = c.upload(&Tensor::ones(&[2])).unwrap();
    let result: Result<_, GpuError> = (|| {
        let _batch = c.command_batch();
        let value = c.scale_device(&x, 7.)?;
        let invalid = c.zeros(&[3])?;
        c.add_device(&value, &invalid)
    })();
    assert!(result.is_err());
    assert_eq!(c.profile_snapshot().unwrap().counters.submissions, 1);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _batch = c.command_batch();
        let _ = c.scale_device(&x, 9.).unwrap();
        panic!("exercise batch cleanup");
    }));
    assert!(unwound.is_err());
    assert_eq!(c.profile_snapshot().unwrap().counters.submissions, 2);
    let value = c.scale_device(&x, 11.).unwrap();
    assert_eq!(c.profile_snapshot().unwrap().counters.submissions, 3);
    assert_eq!(c.download(&value).unwrap().to_vec(), vec![11.; 2]);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn pending_storage_is_not_recycled_by_concurrent_clones() {
    let c = GpuContext::new().unwrap();
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|s| {
        for worker in 1..=4 {
            let c = c.clone();
            let barrier = &barrier;
            s.spawn(move || {
                for _ in 0..12 {
                    let _batch = c.command_batch();
                    let x = c.full(&[64], worker as f32).unwrap();
                    let a = c.scale_device(&x, 2.).unwrap();
                    let b = c.scale_device(&a, 3.).unwrap();
                    drop(x);
                    drop(a); // Deferred command buffers must retain these leases.
                    barrier.wait();
                    assert_eq!(
                        c.download(&b).unwrap().to_vec(),
                        vec![(worker * 6) as f32; 64]
                    );
                }
            });
        }
    });
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn workspace_reuse_preserves_live_snapshots_and_empty_reductions() {
    let c = GpuContext::new().unwrap().with_profiling();
    let original = c.upload(&Tensor::from_vec(vec![1., 2., 3.], &[3])).unwrap();
    let saved = c.scale_device(&original, 2.).unwrap();
    for factor in 1..=32 {
        let output = c.scale_device(&original, factor as f32).unwrap();
        assert_eq!(
            c.download(&output).unwrap().to_vec(),
            vec![factor as f32, (factor * 2) as f32, (factor * 3) as f32]
        );
    }
    assert_eq!(c.download(&saved).unwrap().to_vec(), vec![2., 4., 6.]);
    let empty = c.zeros(&[0, 3]).unwrap();
    assert_eq!(
        c.download(&c.sum_rows_device(&empty).unwrap())
            .unwrap()
            .to_vec(),
        vec![0.; 3]
    );
    let p = c.profile_snapshot().unwrap().counters;
    assert!(p.tensor_reuses >= 30);
    assert!(p.tensor_allocations < 10);
    // Larger recycled buffers may contain NaN tails outside the logical shape.
    let tail_context = GpuContext::new().unwrap();
    drop(tail_context.full(&[64], f32::NAN).unwrap());
    let small = tail_context.full(&[8], 2.).unwrap();
    assert_eq!(
        tail_context
            .download(&tail_context.sum_device(&small).unwrap())
            .unwrap()
            .item(),
        16.
    );
    assert_eq!(
        tail_context
            .download(&tail_context.nonfinite_count_device(&small).unwrap())
            .unwrap()
            .item(),
        0.
    );
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn arbitrary_tensor_batch_preserves_shapes_bits_and_validates_before_work() {
    let c = GpuContext::new().unwrap().with_profiling();
    let a = c
        .upload(&Tensor::from_vec(
            vec![1., -0., f32::INFINITY, f32::NAN],
            &[2, 2],
        ))
        .unwrap();
    let b = c.full(&[], 3.).unwrap();
    let empty = c.zeros(&[0, 4]).unwrap();
    let before = c.profile_snapshot().unwrap().counters;
    let values = c.download_tensors(&[&a, &empty, &b, &a]).unwrap();
    assert_eq!(
        values
            .iter()
            .map(|t| t.shape().to_vec())
            .collect::<Vec<_>>(),
        vec![vec![2, 2], vec![0, 4], vec![], vec![2, 2]]
    );
    let bits = values[0]
        .to_vec()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bits, [1., -0., f32::INFINITY, f32::NAN].map(f32::to_bits));
    assert_eq!(values[2].item(), 3.);
    let after = c.profile_snapshot().unwrap().counters;
    assert_eq!(after.host_waits - before.host_waits, 1);
    assert_eq!(after.readback_bytes - before.readback_bytes, 36);
    let foreign = GpuContext::new().unwrap().zeros(&[]).unwrap();
    assert!(matches!(
        c.download_tensors(&[&a, &foreign]),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(c.download_tensors(&[]).unwrap().is_empty());
    assert_eq!(c.download_tensors(&[&empty]).unwrap()[0].shape(), &[0, 4]);
    assert_eq!(c.profile_snapshot().unwrap().counters, after);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn fused_optimizer_primitives_match_unfused_formulas_and_reduce_dispatches() {
    let c = GpuContext::new().unwrap().with_profiling();
    let a = c
        .upload(&Tensor::from_vec(vec![0., 0.125, 2., 100.], &[4]))
        .unwrap();
    let b = c
        .upload(&Tensor::from_vec(vec![1., 2., 4., 8.], &[4]))
        .unwrap();
    let reference = [
        c.add_device(&a, &c.scale_device(&b, 0.5).unwrap()).unwrap(),
        c.scale_device(&c.mul_device(&a, &a).unwrap(), 0.25)
            .unwrap(),
        c.sqrt_add_device(&c.scale_device(&a, 0.5).unwrap(), 0.01)
            .unwrap(),
        c.scale_device(&c.div_device(&a, &b).unwrap(), -0.2)
            .unwrap(),
    ];
    let before = c.profile_snapshot().unwrap().counters;
    let fused = [
        c.scale_add_device(&a, &b, 0.5).unwrap(),
        c.square_scale_device(&a, 0.25).unwrap(),
        c.scale_sqrt_add_device(&a, 0.5, 0.01).unwrap(),
        c.div_scale_device(&a, &b, -0.2).unwrap(),
    ];
    assert_eq!(
        c.profile_snapshot().unwrap().counters.compute_dispatches - before.compute_dispatches,
        4
    );
    for (expected, actual) in reference.iter().zip(&fused) {
        for (e, a) in c
            .download(expected)
            .unwrap()
            .to_vec()
            .iter()
            .zip(c.download(actual).unwrap().to_vec())
        {
            approx::assert_abs_diff_eq!(*e, a, epsilon = 1e-6);
        }
    }
    let invalid = c.zeros(&[3]).unwrap();
    assert!(matches!(
        c.scale_add_device(&a, &invalid, 1.),
        Err(GpuError::ElementwiseShapeMismatch { .. })
    ));
}
