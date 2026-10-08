#![cfg(feature = "gpu")]
use rustforge_tensor::{
    gpu::{GpuContext, GpuError, MatmulKernel},
    Tensor,
};

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn counters_cover_tensor_and_index_transfers_without_extra_work() {
    let plain = GpuContext::new().unwrap();
    assert!(plain.profile_snapshot().is_none());
    assert!(plain.profile_scope("disabled").is_none());
    let context = plain.with_profiling();
    let clone = context.clone().with_matmul_kernel(MatmulKernel::Naive);
    {
        let _scope = context.profile_scope("chain");
        let a = clone
            .upload(&Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]))
            .unwrap();
        let b = clone
            .upload(&Tensor::from_vec(vec![1., 0., 0., 1.], &[2, 2]))
            .unwrap();
        let out = clone.matmul_device(&a, &b).unwrap();
        let indices = clone.upload_indices(&[1, 0], 2).unwrap();
        let empty = clone.upload(&Tensor::zeros(&[0, 2])).unwrap();
        assert_eq!(clone.download(&out).unwrap().to_vec(), vec![1., 2., 3., 4.]);
        assert_eq!(clone.download_indices(&indices).unwrap(), vec![1, 0]);
        assert!(clone.download(&empty).unwrap().to_vec().is_empty());
        clone.synchronize();
    }
    let snapshot = context.profile_snapshot().unwrap();
    let c = &snapshot.counters;
    assert_eq!(c.submissions, 3);
    assert_eq!(c.compute_dispatches, 1);
    assert_eq!(c.readback_submissions, 2);
    assert_eq!((c.uploads, c.upload_bytes), (3, 40));
    assert_eq!((c.readbacks, c.readback_bytes), (2, 24));
    assert_eq!(c.host_waits, 3);
    assert_eq!((c.parameter_allocations, c.parameter_reuses), (1, 0));
    assert_eq!((c.readback_allocations, c.readback_reuses), (1, 1));
    assert_eq!((c.tensor_allocations, c.tensor_bytes), (5, 60));
    assert_eq!(snapshot.phases["chain"].counters, *c);
    assert_eq!(snapshot.phases["chain"].calls, 1);
    assert_eq!(snapshot, clone.profile_snapshot().unwrap());

    // Invalid operations and empty downloads perform no accounted transfer.
    assert!(clone.upload_indices(&[2], 2).is_err());
    let foreign = GpuContext::new().unwrap().zeros(&[2, 2]).unwrap();
    assert!(matches!(
        clone.download(&foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert_eq!(context.profile_snapshot().unwrap(), snapshot);
    // Zero-inner-dimension products enqueue a clear, not a compute dispatch.
    let left = clone.zeros(&[2, 0]).unwrap();
    let right = clone.zeros(&[0, 2]).unwrap();
    let cleared = clone.matmul_device(&left, &right).unwrap();
    assert_eq!(clone.download(&cleared).unwrap().to_vec(), vec![0.; 4]);
    let after_clear = clone.profile_snapshot().unwrap().counters;
    assert_eq!(after_clear.submissions, c.submissions + 2);
    assert_eq!(after_clear.compute_dispatches, c.compute_dispatches);
    assert_eq!(after_clear.readbacks, c.readbacks + 1);
    // A fresh collector is compatible with existing tensors but has no history.
    let independent = context.with_profiling();
    let compatible = plain.zeros(&[1]).unwrap();
    assert!(independent.is_compatible(&compatible));
    assert_eq!(
        independent.profile_snapshot().unwrap().counters.submissions,
        0
    );
}
