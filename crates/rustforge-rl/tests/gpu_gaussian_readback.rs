#![cfg(feature = "gpu")]
use rustforge_autograd::gpu::GpuVariable;
use rustforge_rl::agent::gpu_gaussian::{GpuGaussianError, GpuGaussianTransform};
use rustforge_tensor::{gpu::GpuContext, Tensor};

fn variable(context: &GpuContext, values: Vec<f32>, shape: &[usize], grad: bool) -> GpuVariable {
    GpuVariable::new(context, &Tensor::from_vec(values, shape), grad).unwrap()
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn sampled_and_stored_metrics_match_reference_with_one_readback() {
    let context = GpuContext::new().unwrap().with_profiling();
    let transform = GpuGaussianTransform::new(&context, &[-2., -1.], &[2., 3.]).unwrap();
    for rows in [0, 1, 7] {
        let shape = [rows, 2];
        let mean = variable(&context, vec![0.2; rows * 2], &shape, true);
        let std = variable(&context, vec![-0.3; rows * 2], &shape, true);
        let noise = variable(&context, vec![0.5; rows * 2], &shape, false);
        let sample = transform.sample_with_noise(&mean, &std, &noise).unwrap();
        let stored = transform
            .log_prob_from_action(&mean, &std, &sample.actions)
            .unwrap();
        for sampled in [true, false] {
            let distribution = if sampled {
                &sample.distribution
            } else {
                &stored
            };
            let expected_log_prob = distribution
                .log_probs
                .mean()
                .unwrap()
                .to_cpu()
                .unwrap()
                .item();
            let expected_entropy = distribution
                .base_entropy
                .mean()
                .unwrap()
                .to_cpu()
                .unwrap()
                .item();
            let before = context.profile_snapshot().unwrap();
            let metrics = if sampled {
                sample.checked_metrics().unwrap()
            } else {
                distribution.checked_metrics().unwrap()
            };
            let after = context.profile_snapshot().unwrap();
            assert_eq!(metrics.mean_log_prob.to_bits(), expected_log_prob.to_bits());
            assert_eq!(metrics.base_entropy.to_bits(), expected_entropy.to_bits());
            assert_eq!(after.counters.readbacks - before.counters.readbacks, 1);
            assert_eq!(after.counters.host_waits - before.counters.host_waits, 1);
            assert!(mean.grad().is_none() && std.grad().is_none());
        }
        if rows > 0 {
            let checked = sample.checked_metrics().unwrap();
            assert!(checked.mean_log_prob.is_finite());
            sample
                .distribution
                .log_probs
                .mean()
                .unwrap()
                .backward()
                .unwrap();
            assert!(mean.grad().is_some() && std.grad().is_some());
        }
    }
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn batched_checks_reject_masked_inputs_outputs_and_mean_overflow() {
    let context = GpuContext::new().unwrap();
    let transform = GpuGaussianTransform::new(&context, &[-1.], &[1.]).unwrap();
    for input in 0..3 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = [0., 0., 0.];
            values[input] = bad;
            let mean = variable(&context, vec![values[0]], &[1, 1], true);
            let std = variable(&context, vec![values[1]], &[1, 1], true);
            let other = variable(&context, vec![values[2]], &[1, 1], false);
            let sampled = transform.sample_with_noise(&mean, &std, &other).unwrap();
            let stored = transform.log_prob_from_action(&mean, &std, &other).unwrap();
            assert!(matches!(
                sampled.checked_metrics(),
                Err(GpuGaussianError::NonFinite)
            ));
            assert!(matches!(
                stored.checked_metrics(),
                Err(GpuGaussianError::NonFinite)
            ));
            assert!(mean.grad().is_none() && std.grad().is_none());
        }
    }
    let mean = variable(&context, vec![0.; 2], &[2, 1], true);
    let std = variable(&context, vec![0.; 2], &[2, 1], true);
    let noise = variable(&context, vec![0.; 2], &[2, 1], false);
    for output in 0..3 {
        let mut sample = transform.sample_with_noise(&mean, &std, &noise).unwrap();
        let bad = variable(&context, vec![f32::NAN; 2], &[2, 1], false);
        match output {
            0 => sample.actions = bad,
            1 => sample.distribution.log_probs = bad,
            _ => sample.distribution.base_entropy = bad,
        }
        assert!(matches!(
            sample.checked_metrics(),
            Err(GpuGaussianError::NonFinite)
        ));
    }
    // Individually finite statistics can still overflow their mean reduction.
    for entropy in [false, true] {
        let mut sample = transform.sample_with_noise(&mean, &std, &noise).unwrap();
        let huge = variable(&context, vec![f32::MAX; 2], &[2, 1], false);
        if entropy {
            sample.distribution.base_entropy = huge;
        } else {
            sample.distribution.log_probs = huge;
        }
        assert!(matches!(
            sample.checked_metrics(),
            Err(GpuGaussianError::NonFinite)
        ));
    }
    assert!(mean.grad().is_none() && std.grad().is_none());
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn replaced_fields_in_other_contexts_keep_previous_validation_behavior() {
    let context = GpuContext::new().unwrap();
    let other = GpuContext::new().unwrap();
    let transform = GpuGaussianTransform::new(&context, &[-1.], &[1.]).unwrap();
    let zero = variable(&context, vec![0.], &[1, 1], false);
    for field in 0..3 {
        let mut sample = transform.sample_with_noise(&zero, &zero, &zero).unwrap();
        let replacement = variable(&other, vec![3.], &[1, 1], false);
        match field {
            0 => sample.actions = replacement,
            1 => sample.distribution.log_probs = replacement,
            _ => sample.distribution.base_entropy = replacement,
        }
        let expected_log = sample
            .distribution
            .log_probs
            .mean()
            .unwrap()
            .to_cpu()
            .unwrap()
            .item();
        let expected_entropy = sample
            .distribution
            .base_entropy
            .mean()
            .unwrap()
            .to_cpu()
            .unwrap()
            .item();
        let metrics = sample.checked_metrics().unwrap();
        assert_eq!(metrics.mean_log_prob.to_bits(), expected_log.to_bits());
        assert_eq!(metrics.base_entropy.to_bits(), expected_entropy.to_bits());
        let invalid = variable(&other, vec![f32::NAN], &[1, 1], false);
        match field {
            0 => sample.actions = invalid,
            1 => sample.distribution.log_probs = invalid,
            _ => sample.distribution.base_entropy = invalid,
        }
        assert!(matches!(
            sample.checked_metrics(),
            Err(GpuGaussianError::NonFinite)
        ));
    }
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn checked_action_readback_preserves_bits_and_rejects_hidden_invalid_noise() {
    let c = GpuContext::new().unwrap().with_profiling();
    let transform = GpuGaussianTransform::new(&c, &[-2., -1.], &[2., 3.]).unwrap();
    let mean = variable(&c, vec![0.2; 6], &[3, 2], true);
    let std = variable(&c, vec![-0.3; 6], &[3, 2], true);
    let noise = variable(&c, vec![0.5; 6], &[3, 2], false);
    let sample = transform.sample_with_noise(&mean, &std, &noise).unwrap();
    let expected = sample.actions.to_cpu().unwrap().to_vec();
    let before = c.profile_snapshot().unwrap().counters;
    let actual = sample.checked_actions().unwrap().to_vec();
    let after = c.profile_snapshot().unwrap().counters;
    assert_eq!(
        actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
    );
    assert_eq!(
        (
            after.readbacks - before.readbacks,
            after.host_waits - before.host_waits
        ),
        (1, 1)
    );
    sample
        .distribution
        .log_probs
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert!(mean
        .grad_cpu()
        .unwrap()
        .unwrap()
        .to_vec()
        .iter()
        .all(|v| v.is_finite()));
    let bad_noise = variable(&c, vec![f32::INFINITY; 6], &[3, 2], false);
    let invalid = transform
        .sample_with_noise(&mean, &std, &bad_noise)
        .unwrap();
    assert!(matches!(
        invalid.checked_actions(),
        Err(GpuGaussianError::NonFinite)
    ));
}
