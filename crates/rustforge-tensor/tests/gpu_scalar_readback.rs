#![cfg(feature = "gpu")]
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn scalar_batch_preserves_order_bits_and_uses_one_wait() {
    let context = GpuContext::new().unwrap().with_profiling();
    let tensors: Vec<_> = [1.25, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]
        .into_iter()
        .map(|v| context.upload(&Tensor::full(&[], v)).unwrap())
        .collect();
    let references: Vec<_> = tensors.iter().collect();
    let before = context.profile_snapshot().unwrap();
    let values = context.download_scalars(&references).unwrap();
    for (value, tensor) in values.iter().zip(&tensors) {
        let expected = context.download(tensor).unwrap().item();
        assert_eq!(value.to_bits(), expected.to_bits());
    }
    let after = context.profile_snapshot().unwrap();
    // Five reference downloads plus exactly one packed download; no compute or
    // tensor allocation is needed to copy the scalars into a shared staging buffer.
    assert_eq!(after.counters.readbacks - before.counters.readbacks, 6);
    assert_eq!(after.counters.host_waits - before.counters.host_waits, 6);
    assert_eq!(after.counters.submissions - before.counters.submissions, 6);
    assert_eq!(
        after.counters.readback_bytes - before.counters.readback_bytes,
        40
    );
    assert_eq!(
        after.counters.compute_dispatches,
        before.counters.compute_dispatches
    );
    assert_eq!(
        after.counters.tensor_allocations,
        before.counters.tensor_allocations
    );
    let repeated = context
        .download_scalars(&[&tensors[1], &tensors[0], &tensors[1]])
        .unwrap();
    assert_eq!(
        repeated.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![
            (-0.0_f32).to_bits(),
            1.25_f32.to_bits(),
            (-0.0_f32).to_bits()
        ]
    );
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn empty_and_invalid_batches_do_not_submit_or_wait() {
    let context = GpuContext::new().unwrap().with_profiling();
    let scalar = context.zeros(&[1, 1]).unwrap();
    let vector = context.zeros(&[2]).unwrap();
    let empty = context.zeros(&[0]).unwrap();
    let foreign = GpuContext::new().unwrap().zeros(&[]).unwrap();
    let before = context.profile_snapshot().unwrap();
    assert!(context.download_scalars(&[]).unwrap().is_empty());
    for invalid in [&vector, &empty] {
        assert!(matches!(
            context.download_scalars(&[&scalar, invalid]),
            Err(GpuError::ExpectedScalar { .. })
        ));
    }
    assert!(matches!(
        context.download_scalars(&[&scalar, &foreign]),
        Err(GpuError::DeviceMismatch)
    ));
    assert_eq!(context.profile_snapshot().unwrap(), before);
    assert_eq!(context.download_scalars(&[&scalar]).unwrap(), vec![0.]);
}
