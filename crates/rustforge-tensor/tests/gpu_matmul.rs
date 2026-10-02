#![cfg(feature = "gpu")]

use approx::assert_abs_diff_eq;
use rustforge_tensor::{
    gpu::{GpuContext, GpuError, MatmulKernel},
    Tensor,
};
use std::sync::OnceLock;

fn context() -> &'static GpuContext {
    static CONTEXT: OnceLock<GpuContext> = OnceLock::new();
    CONTEXT.get_or_init(|| GpuContext::new().expect("GPU adapter required for this test"))
}

fn assert_parity(context: &GpuContext, left: &Tensor, right: &Tensor) {
    let expected = left.matmul(right);
    let actual = context
        .matmul(left, right)
        .expect("GPU multiplication failed");
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.to_vec().iter().zip(expected.to_vec()) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-4);
    }
}

// Keep normal CPU CI usable without graphics drivers. Run this suite explicitly
// with --include-ignored: unavailable adapters are errors, never silent passes.
#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn gpu_matches_cpu_for_rectangular_and_partial_workgroups() {
    let context = context();
    eprintln!("Testing adapter: {:?}", context.adapter_info());
    for (m, k, n) in [(1, 1, 1), (2, 3, 4), (8, 8, 8), (9, 17, 11), (17, 3, 9)] {
        let left = Tensor::rand_uniform(&[m, k], -1., 1., Some(42));
        let right = Tensor::rand_uniform(&[k, n], -1., 1., Some(43));
        let left_before = left.to_vec();
        let right_before = right.to_vec();
        assert_parity(context, &left, &right);
        assert_eq!(left.to_vec(), left_before);
        assert_eq!(right.to_vec(), right_before);
    }
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn gpu_packs_noncontiguous_inputs_in_logical_order() {
    let context = context();
    let left = Tensor::from_ndarray(
        Tensor::rand_uniform(&[7, 9], -1., 1., Some(1))
            .data()
            .clone()
            .reversed_axes(),
    );
    let right = Tensor::from_ndarray(
        Tensor::rand_uniform(&[11, 7], -1., 1., Some(2))
            .data()
            .clone()
            .reversed_axes(),
    );
    assert!(!left.data().is_standard_layout());
    assert!(!right.data().is_standard_layout());
    assert_parity(context, &left, &right);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn gpu_handles_empty_products_and_rejects_bad_shapes() {
    let context = context();
    for (left, right, output) in [
        ([0, 3], [3, 4], [0, 4]),
        ([2, 0], [0, 3], [2, 3]),
        ([2, 3], [3, 0], [2, 0]),
    ] {
        let result = context
            .matmul(&Tensor::zeros(&left), &Tensor::zeros(&right))
            .unwrap();
        assert_eq!(result.shape(), &output);
        assert!(result.to_vec().iter().all(|value| *value == 0.));
    }
    assert!(matches!(
        context.matmul(&Tensor::zeros(&[2, 3]), &Tensor::zeros(&[4, 2])),
        Err(GpuError::InvalidShape { .. })
    ));
    assert!(matches!(
        context.matmul(&Tensor::zeros(&[3]), &Tensor::zeros(&[3, 2])),
        Err(GpuError::InvalidShape { .. })
    ));
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn gpu_preserves_nonfinite_results() {
    let right = Tensor::ones(&[1, 1]);
    for kernel in [MatmulKernel::Naive, MatmulKernel::Tiled] {
        let context = context().with_matmul_kernel(kernel);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result = context
                .matmul(&Tensor::from_vec(vec![value], &[1, 1]), &right)
                .unwrap()
                .item();
            if value.is_nan() {
                assert!(result.is_nan());
            } else {
                assert_eq!(result, value);
            }
        }
    }
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn persistent_tensors_roundtrip_scalars_ranks_and_empty_shapes() {
    let context = context();
    for tensor in [
        Tensor::scalar(7.5),
        Tensor::rand_uniform(&[2, 3, 4], -1., 1., Some(12)),
        Tensor::zeros(&[0, 3]),
        Tensor::zeros(&[2, 0, 3]),
    ] {
        let device = context.upload(&tensor).unwrap();
        assert_eq!(device.shape(), tensor.shape());
        assert_eq!(device.numel(), tensor.numel());
        assert_eq!(device.is_empty(), tensor.is_empty());
        let downloaded = context.download(&device).unwrap();
        assert_eq!(downloaded.shape(), tensor.shape());
        assert_eq!(downloaded.to_vec(), tensor.to_vec());
    }
    let zeros = context.zeros(&[2, 3, 4]).unwrap();
    assert_eq!(context.download(&zeros).unwrap().to_vec(), vec![0.; 24]);
    assert!(matches!(
        context.zeros(&[65536, 65536]),
        Err(GpuError::LimitExceeded)
    ));
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn persistent_products_chain_and_reuse_inputs() {
    let context = context();
    let left = Tensor::rand_uniform(&[9, 17], -1., 1., Some(10));
    let middle = Tensor::rand_uniform(&[17, 11], -1., 1., Some(11));
    let right = Tensor::rand_uniform(&[11, 7], -1., 1., Some(12));
    let expected = left.matmul(&middle).matmul(&right);
    let left_device = context.upload(&left).unwrap();
    let middle_device = context.upload(&middle).unwrap();
    let right_device = context.upload(&right).unwrap();
    for _ in 0..3 {
        let intermediate = context.matmul_device(&left_device, &middle_device).unwrap();
        let output = context.matmul_device(&intermediate, &right_device).unwrap();
        drop(intermediate);
        // No intermediate is downloaded, including across queued dispatches.
        let actual = context.download(&output).unwrap();
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.to_vec().iter().zip(expected.to_vec()) {
            assert_abs_diff_eq!(*actual, expected, epsilon = 1e-4);
        }
    }
    assert_eq!(
        context.download(&left_device).unwrap().to_vec(),
        left.to_vec()
    );
    assert_eq!(
        context.download(&middle_device).unwrap().to_vec(),
        middle.to_vec()
    );
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn reused_output_is_overwritten_and_zero_inner_dimension_clears_it() {
    let context = context();
    let left = context.upload(&Tensor::ones(&[2, 3])).unwrap();
    let right = context.upload(&Tensor::ones(&[3, 2])).unwrap();
    let mut output = context.upload(&Tensor::full(&[2, 2], 99.)).unwrap();
    context.matmul_into(&left, &right, &mut output).unwrap();
    assert_eq!(context.download(&output).unwrap().to_vec(), vec![3.; 4]);
    let left = context.zeros(&[2, 0]).unwrap();
    let right = context.zeros(&[0, 2]).unwrap();
    context.matmul_into(&left, &right, &mut output).unwrap();
    assert_eq!(context.download(&output).unwrap().to_vec(), vec![0.; 4]);
    let mut wrong_shape = context.upload(&Tensor::full(&[4], 99.)).unwrap();
    assert!(matches!(
        context.matmul_into(&left, &right, &mut wrong_shape),
        Err(GpuError::InvalidOutputShape { .. })
    ));
    assert_eq!(
        context.download(&wrong_shape).unwrap().to_vec(),
        vec![99.; 4]
    );
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn persistent_tensors_reject_foreign_devices_and_accept_context_clones() {
    // Retain both devices for this process: older EGL drivers can conflict
    // when separate contexts are created/destroyed concurrently by test threads.
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let context = context();
    let other = OTHER.get_or_init(|| GpuContext::new().expect("second device required"));
    let local = context.upload(&Tensor::ones(&[2, 2])).unwrap();
    let foreign = other.upload(&Tensor::ones(&[2, 2])).unwrap();
    let mut output = context.zeros(&[2, 2]).unwrap();
    let mut foreign_output = other.zeros(&[2, 2]).unwrap();
    assert!(matches!(
        context.download(&foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.matmul_device(&local, &foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.matmul_device(&foreign, &local),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.matmul_into(&foreign, &local, &mut output),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.matmul_into(&local, &local, &mut foreign_output),
        Err(GpuError::DeviceMismatch)
    ));
    let clone = context.clone();
    clone.matmul_into(&local, &local, &mut output).unwrap();
    assert_eq!(clone.download(&output).unwrap().to_vec(), vec![2.; 4]);
    let empty_foreign = other.zeros(&[0, 2]).unwrap();
    assert!(matches!(
        context.download(&empty_foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.add_device(&local, &foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.mul_device(&foreign, &local),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.relu_device(&foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.sum_device(&empty_foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.mean_device(&foreign),
        Err(GpuError::DeviceMismatch)
    ));
    assert!(matches!(
        context.matmul_t(&local, &foreign),
        Err(GpuError::DeviceMismatch)
    ));
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.to_vec().iter().zip(expected.to_vec()) {
        assert!(
            (*actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
            "actual {actual}, expected {expected}"
        );
    }
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn tiled_and_naive_kernels_match_cpu_across_tile_edges() {
    for kernel in [MatmulKernel::Naive, MatmulKernel::Tiled] {
        let context = context().with_matmul_kernel(kernel);
        for (m, k, n) in [(1, 7, 9), (7, 8, 15), (8, 9, 8), (17, 31, 33), (32, 32, 32)] {
            let a = Tensor::rand_uniform(&[m, k], -1., 1., Some(91));
            let b = Tensor::rand_uniform(&[k, n], -1., 1., Some(92));
            assert_tensor_close(&context.matmul(&a, &b).unwrap(), &a.matmul(&b));
        }
    }
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn transposed_products_match_cpu_and_reuse_outputs() {
    for kernel in [MatmulKernel::Naive, MatmulKernel::Tiled] {
        let context = context().with_matmul_kernel(kernel);
        let a = Tensor::rand_uniform(&[9, 17], -1., 1., Some(17));
        let b = Tensor::rand_uniform(&[11, 17], -1., 1., Some(18));
        let a_device = context.upload(&a).unwrap();
        let b_device = context.upload(&b).unwrap();
        let result = context.matmul_t(&a_device, &b_device).unwrap();
        assert_tensor_close(&context.download(&result).unwrap(), &a.matmul_t(&b));
        let c = Tensor::rand_uniform(&[9, 11], -1., 1., Some(19));
        let c_device = context.upload(&c).unwrap();
        let result = context.t_matmul(&a_device, &c_device).unwrap();
        assert_tensor_close(&context.download(&result).unwrap(), &a.t_matmul(&c));
        let mut output = context.zeros(&[17, 11]).unwrap();
        context
            .t_matmul_into(&a_device, &c_device, &mut output)
            .unwrap();
        assert_tensor_close(&context.download(&output).unwrap(), &a.t_matmul(&c));
        let mut output = context.zeros(&[9, 11]).unwrap();
        context
            .matmul_t_into(&a_device, &b_device, &mut output)
            .unwrap();
        assert_tensor_close(&context.download(&output).unwrap(), &a.matmul_t(&b));
        assert!(matches!(
            context.matmul_t(&a_device, &c_device),
            Err(GpuError::InvalidShape { .. })
        ));
        let empty_a = context.zeros(&[0, 2]).unwrap();
        let empty_b = context.zeros(&[0, 3]).unwrap();
        let result = context.t_matmul(&empty_a, &empty_b).unwrap();
        assert_eq!(context.download(&result).unwrap().to_vec(), vec![0.; 6]);
    }
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn device_elementwise_operations_match_cpu_without_broadcasting() {
    let context = context();
    for shape in [vec![], vec![257], vec![2, 3, 17], vec![0, 3]] {
        let a = Tensor::rand_uniform(&shape, -2., 2., Some(30));
        let b = Tensor::rand_uniform(&shape, -2., 2., Some(31));
        let a_device = context.upload(&a).unwrap();
        let b_device = context.upload(&b).unwrap();
        let result = context.add_device(&a_device, &b_device).unwrap();
        assert_tensor_close(&context.download(&result).unwrap(), &(&a + &b));
        let result = context.mul_device(&a_device, &b_device).unwrap();
        assert_tensor_close(&context.download(&result).unwrap(), &(&a * &b));
        let result = context.relu_device(&a_device).unwrap();
        assert_tensor_close(&context.download(&result).unwrap(), &a.relu());
    }
    let a = context.zeros(&[2, 2]).unwrap();
    let b = context.zeros(&[4]).unwrap();
    assert!(matches!(
        context.add_device(&a, &b),
        Err(GpuError::ElementwiseShapeMismatch { .. })
    ));
    assert!(matches!(
        context.mul_device(&a, &b),
        Err(GpuError::ElementwiseShapeMismatch { .. })
    ));
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn hierarchical_reductions_match_cpu_including_empty_and_multiple_passes() {
    let context = context();
    for length in [0, 1, 255, 256, 257, 65537] {
        let input = Tensor::rand_uniform(&[length], -1., 1., Some(777));
        let device = context.upload(&input).unwrap();
        let sum = context.sum_device(&device).unwrap();
        let mean = context.mean_device(&device).unwrap();
        assert_eq!(sum.shape(), &[]);
        assert_eq!(mean.shape(), &[]);
        assert_tensor_close(&context.download(&sum).unwrap(), &input.sum());
        assert_tensor_close(&context.download(&mean).unwrap(), &input.mean());
    }
    let scalar = context.upload(&Tensor::scalar(7.)).unwrap();
    assert_eq!(
        context
            .download(&context.mean_device(&scalar).unwrap())
            .unwrap()
            .item(),
        7.
    );
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn elementwise_and_reductions_preserve_cpu_nonfinite_semantics() {
    let context = context();
    let input = Tensor::from_vec(
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1., 2.],
        &[5],
    );
    let device = context.upload(&input).unwrap();
    assert_eq!(
        context
            .download(&context.relu_device(&device).unwrap())
            .unwrap()
            .to_vec(),
        input.relu().to_vec()
    );
    assert!(context
        .download(&context.sum_device(&device).unwrap())
        .unwrap()
        .item()
        .is_nan());
    assert!(context
        .download(&context.mean_device(&device).unwrap())
        .unwrap()
        .item()
        .is_nan());
    let inf = context
        .upload(&Tensor::from_vec(vec![f32::INFINITY; 257], &[257]))
        .unwrap();
    assert_eq!(
        context
            .download(&context.sum_device(&inf).unwrap())
            .unwrap()
            .item(),
        f32::INFINITY
    );
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn operations_cover_the_second_dispatch_grid_row() {
    let context = context();
    // More than 65,535 workgroups: exercises the 2D dispatch and padding guard.
    let input = context.upload(&Tensor::ones(&[16_777_217])).unwrap();
    let activated = context.relu_device(&input).unwrap();
    assert!(context
        .download(&activated)
        .unwrap()
        .to_vec()
        .iter()
        .all(|x| *x == 1.));
    let mean = context.mean_device(&activated).unwrap();
    assert_tensor_close(&context.download(&mean).unwrap(), &Tensor::scalar(1.));
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn row_bias_and_reduction_match_cpu_across_workgroup_edges() {
    let context = context();
    for (rows, cols) in [(0, 3), (3, 0), (1, 1), (3, 257), (257, 3)] {
        let matrix = Tensor::rand_uniform(&[rows, cols], -1., 1., Some(12));
        let bias = Tensor::rand_uniform(&[cols], -1., 1., Some(13));
        let dm = context.upload(&matrix).unwrap();
        let db = context.upload(&bias).unwrap();
        let actual = context
            .download(&context.add_bias_device(&dm, &db).unwrap())
            .unwrap();
        let expected = &matrix + &bias;
        assert_eq!(actual.shape(), &[rows, cols]);
        for (&a, &b) in actual.to_vec().iter().zip(expected.to_vec().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        let sums = context
            .download(&context.sum_rows_device(&dm).unwrap())
            .unwrap();
        assert_eq!(sums.shape(), &[cols]);
        let data = matrix.to_vec();
        for (col, &actual) in sums.to_vec().iter().enumerate() {
            let expected: f32 = (0..rows).map(|row| data[row * cols + col]).sum();
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
    }
    let matrix = context.zeros(&[2, 3]).unwrap();
    for shape in [vec![2], vec![1, 3], vec![]] {
        assert!(context
            .add_bias_device(&matrix, &context.zeros(&shape).unwrap())
            .is_err());
    }
    assert!(context
        .sum_rows_device(&context.zeros(&[3]).unwrap())
        .is_err());
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    assert!(context
        .add_bias_device(&matrix, &other.zeros(&[3]).unwrap())
        .is_err());
    assert!(context
        .sum_rows_device(&other.zeros(&[2, 3]).unwrap())
        .is_err());
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn adam_primitives_match_cpu_and_reject_shape_device_mismatch() {
    let context = context();
    for shape in [vec![], vec![0, 3], vec![257]] {
        let values = Tensor::rand_uniform(&shape, 0., 4., Some(9));
        let input = context.upload(&values).unwrap();
        let denominator = context.sqrt_add_device(&input, 0.01).unwrap();
        let ratio = context.div_device(&input, &denominator).unwrap();
        let actual = context.download(&ratio).unwrap();
        assert_eq!(actual.shape(), shape);
        for (&a, &v) in actual.to_vec().iter().zip(values.to_vec().iter()) {
            assert_abs_diff_eq!(a, v / (v.sqrt() + 0.01), epsilon = 1e-6);
        }
    }
    let input = context.zeros(&[2]).unwrap();
    assert!(context
        .div_device(&input, &context.zeros(&[1]).unwrap())
        .is_err());
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    assert!(context
        .div_device(&input, &other.zeros(&[2]).unwrap())
        .is_err());
    assert!(context
        .sqrt_add_device(&other.zeros(&[2]).unwrap(), 0.01)
        .is_err());
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn typed_action_indices_argmax_gather_and_scatter_match_cpu() {
    let context = context();
    let indices = context
        .upload_indices(&[0, 16_777_217], 16_777_218)
        .unwrap();
    assert_eq!(
        context.download_indices(&indices).unwrap(),
        vec![0, 16_777_217]
    );
    for rows in [0, 1, 257] {
        let matrix = Tensor::rand_uniform(&[rows, 5], -1., 1., Some(44));
        let device = context.upload(&matrix).unwrap();
        let indices = context.argmax_rows_device(&device).unwrap();
        let choices = matrix.argmax_axis(1).unwrap();
        assert_eq!(context.download_indices(&indices).unwrap(), choices);
        let gathered = context
            .download(&context.gather_rows_device(&device, &indices).unwrap())
            .unwrap();
        assert_eq!(gathered.shape(), &[rows, 1]);
        let expected = matrix.gather(1, &choices).unwrap();
        assert_eq!(gathered.to_vec(), expected.to_vec());
        let gradient = context.full(&[rows, 1], 2.).unwrap();
        let scattered = context
            .download(&context.scatter_rows_device(&gradient, &indices).unwrap())
            .unwrap();
        let mut expected = vec![0.; rows * 5];
        for (row, &col) in choices.iter().enumerate() {
            expected[row * 5 + col] = 2.;
        }
        assert_eq!(scattered.to_vec(), expected);
    }
    let ties = Tensor::from_vec(vec![1., 2., 2., -0., 0., -0., 0., -0., 0.], &[3, 3]);
    assert_eq!(
        context
            .download_indices(
                &context
                    .argmax_rows_device(&context.upload(&ties).unwrap())
                    .unwrap()
            )
            .unwrap(),
        ties.argmax_axis(1).unwrap()
    );
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn action_indices_reject_invalid_shapes_devices_and_nonfinite_rows() {
    let context = context();
    assert!(context.upload_indices(&[2], 2).is_err());
    assert!(context.upload_indices(&[], 0).is_err());
    assert!(context
        .argmax_rows_device(&context.zeros(&[2, 0]).unwrap())
        .is_err());
    assert!(context
        .argmax_rows_device(&context.zeros(&[2]).unwrap())
        .is_err());
    let indices = context.upload_indices(&[1, 0], 2).unwrap();
    assert!(context
        .gather_rows_device(&context.zeros(&[1, 2]).unwrap(), &indices)
        .is_err());
    assert!(context
        .gather_rows_device(&context.zeros(&[2, 3]).unwrap(), &indices)
        .is_err());
    assert!(context
        .scatter_rows_device(&context.zeros(&[2]).unwrap(), &indices)
        .is_err());
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    let foreign = other.upload_indices(&[0, 1], 2).unwrap();
    assert!(context
        .gather_rows_device(&context.zeros(&[2, 2]).unwrap(), &foreign)
        .is_err());
    assert!(context
        .scatter_rows_device(&context.zeros(&[2, 1]).unwrap(), &foreign)
        .is_err());
    assert!(context.download_indices(&foreign).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let matrix = context
            .upload(&Tensor::from_vec(vec![1., value], &[1, 2]))
            .unwrap();
        let choice = context.argmax_rows_device(&matrix).unwrap();
        assert!(matches!(
            context.download_indices(&choice),
            Err(GpuError::NonFinite)
        ));
        let output = context
            .download(&context.gather_rows_device(&matrix, &choice).unwrap())
            .unwrap();
        assert!(output.item().is_nan());
    }
}
