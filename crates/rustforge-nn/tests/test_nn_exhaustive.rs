//! Exhaustive integration tests for rustforge-nn.
//!
//! Covers: Linear layer, activation modules, loss functions, Sequential container,
//! Dropout, and LayerNorm — including edge cases, gradient flow, and numerical checks.

use approx::assert_abs_diff_eq;
use rustforge_autograd::Variable;
use rustforge_nn::*;
use rustforge_tensor::Tensor;

// Module: Linear Layer

mod linear_layer {
    use super::*;

    #[test]
    fn output_shape_batch_1() {
        let layer = Linear::new(4, 3);
        let x = Variable::new(Tensor::ones(&[1, 4]), false);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), vec![1, 3]);
    }

    #[test]
    fn output_shape_batch_32() {
        let layer = Linear::new(10, 5);
        let x = Variable::new(Tensor::ones(&[32, 10]), false);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), vec![32, 5]);
    }

    #[test]
    fn output_shape_batch_128() {
        let layer = Linear::new(64, 16);
        let x = Variable::new(Tensor::ones(&[128, 64]), false);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), vec![128, 16]);
    }

    #[test]
    fn no_bias_parameter_count() {
        let layer = Linear::no_bias(10, 5);
        assert_eq!(layer.parameters().len(), 1);
        assert_eq!(layer.parameters()[0].shape(), vec![5, 10]);
    }

    #[test]
    fn with_bias_parameter_count() {
        let layer = Linear::new(10, 5);
        assert_eq!(layer.parameters().len(), 2);
        assert_eq!(layer.parameters()[0].shape(), vec![5, 10]);
        assert_eq!(layer.parameters()[1].shape(), vec![5]);
    }

    #[test]
    fn known_values_computation() {
        let layer = Linear::new(2, 3);
        // W = [[1, 0], [0, 1], [1, 1]], b = [0.1, 0.2, 0.3]
        layer.parameters()[0].set_data(Tensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[3, 2],
        ));
        layer.parameters()[1].set_data(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]));

        // x = [[1, 2]]
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
        let y = layer.forward(&x);
        let data = y.data().to_vec();
        // y = x @ W^T + b = [[1,2]] @ [[1,0,1],[0,1,1]] + [0.1,0.2,0.3]
        // = [[1, 2, 3]] + [0.1, 0.2, 0.3] = [[1.1, 2.2, 3.3]]
        assert_abs_diff_eq!(data[0], 1.1, epsilon = 1e-5);
        assert_abs_diff_eq!(data[1], 2.2, epsilon = 1e-5);
        assert_abs_diff_eq!(data[2], 3.3, epsilon = 1e-5);
    }

    #[test]
    fn no_bias_computation() {
        let layer = Linear::no_bias(2, 2);
        layer.parameters()[0].set_data(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
        let x = Variable::new(Tensor::from_vec(vec![3.0, 7.0], &[1, 2]), false);
        let y = layer.forward(&x);
        let data = y.data().to_vec();
        assert_abs_diff_eq!(data[0], 3.0, epsilon = 1e-5);
        assert_abs_diff_eq!(data[1], 7.0, epsilon = 1e-5);
    }

    #[test]
    fn gradient_flows_to_weight() {
        let layer = Linear::new(3, 2);
        let x = Variable::new(Tensor::ones(&[2, 3]), false);
        let loss = layer.forward(&x).sum();
        loss.backward();
        assert!(
            layer.parameters()[0].grad().is_some(),
            "Weight should have gradient"
        );
    }

    #[test]
    fn gradient_flows_to_bias() {
        let layer = Linear::new(3, 2);
        let x = Variable::new(Tensor::ones(&[2, 3]), false);
        let loss = layer.forward(&x).sum();
        loss.backward();
        assert!(
            layer.parameters()[1].grad().is_some(),
            "Bias should have gradient"
        );
    }

    #[test]
    fn gradient_flows_to_input() {
        let layer = Linear::new(3, 2);
        let x = Variable::new(Tensor::ones(&[2, 3]), true);
        let loss = layer.forward(&x).sum();
        loss.backward();
        assert!(x.grad().is_some(), "Input should have gradient");
        assert_eq!(x.grad().unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn in_out_features() {
        let layer = Linear::new(784, 256);
        assert_eq!(layer.in_features(), 784);
        assert_eq!(layer.out_features(), 256);
    }

    #[test]
    fn large_layer_no_nan() {
        let layer = Linear::new(512, 256);
        let x = Variable::new(Tensor::rand_uniform(&[8, 512], -1.0, 1.0, Some(42)), false);
        let y = layer.forward(&x);
        for &v in y.data().to_vec().iter() {
            assert!(!v.is_nan(), "Large layer output should not contain NaN");
        }
    }

    #[test]
    fn kaiming_init_bounds() {
        // Kaiming uniform: values should be in [-a, a] where a = sqrt(6/fan_in)
        let layer = Linear::new(100, 50);
        let w = layer.parameters()[0].data().to_vec();
        let a = (6.0_f32 / 100.0).sqrt();
        for &v in &w {
            assert!(
                v >= -a * 1.1 && v <= a * 1.1,
                "Kaiming value {} out of approximate bounds ±{:.4}",
                v,
                a
            );
        }
    }

    #[test]
    fn bias_initialized_to_zero() {
        let layer = Linear::new(10, 5);
        let bias = layer.parameters()[1].data().to_vec();
        for &v in &bias {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn multi_layer_gradient_chain() {
        let l1 = Linear::new(4, 8);
        let l2 = Linear::new(8, 2);
        let x = Variable::new(Tensor::rand_uniform(&[3, 4], -1.0, 1.0, Some(42)), false);
        let h = l1.forward(&x);
        let y = l2.forward(&h);
        let loss = y.sum();
        loss.backward();

        for (i, p) in l1.parameters().iter().enumerate() {
            assert!(p.grad().is_some(), "L1 param {} should have gradient", i);
        }
        for (i, p) in l2.parameters().iter().enumerate() {
            assert!(p.grad().is_some(), "L2 param {} should have gradient", i);
        }
    }
}

// Module: Activations

mod activations {
    use super::*;

    #[test]
    fn relu_preserves_shape() {
        let relu = ReLU;
        let x = Variable::new(Tensor::ones(&[3, 4]), false);
        assert_eq!(relu.forward(&x).shape(), vec![3, 4]);
    }

    #[test]
    fn relu_gradient_positive_is_one() {
        let relu = ReLU;
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 5.0], &[1, 3]), true);
        let loss = relu.forward(&x).sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        assert_eq!(grad, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn relu_gradient_negative_is_zero() {
        let relu = ReLU;
        let x = Variable::new(Tensor::from_vec(vec![-1.0, -2.0, -5.0], &[1, 3]), true);
        let loss = relu.forward(&x).sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        assert_eq!(grad, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_at_zero() {
        let relu = ReLU;
        let x = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), true);
        let y = relu.forward(&x);
        assert_eq!(y.data().to_vec(), vec![0.0]);
    }

    #[test]
    fn sigmoid_output_range() {
        let sigmoid = Sigmoid;
        let x = Variable::new(Tensor::rand_uniform(&[100], -10.0, 10.0, Some(42)), false);
        let y = sigmoid.forward(&x);
        for &v in y.data().to_vec().iter() {
            assert!(
                (0.0..=1.0).contains(&v),
                "Sigmoid output {} must be in [0,1]",
                v
            );
        }
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let sigmoid = Sigmoid;
        let x = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        assert_abs_diff_eq!(sigmoid.forward(&x).data().item(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn sigmoid_monotonicity() {
        let sigmoid = Sigmoid;
        let x = Variable::new(
            Tensor::from_vec(vec![-5.0, -2.0, 0.0, 2.0, 5.0], &[1, 5]),
            false,
        );
        let y = sigmoid.forward(&x).data().to_vec();
        for i in 0..4 {
            assert!(y[i] < y[i + 1], "Sigmoid must be monotonically increasing");
        }
    }

    #[test]
    fn sigmoid_gradient_peak_at_zero() {
        let sigmoid = Sigmoid;
        let x = Variable::new(Tensor::from_vec(vec![-2.0, 0.0, 2.0], &[1, 3]), true);
        let loss = sigmoid.forward(&x).sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        // sigmoid'(0) = 0.25, sigmoid'(±2) < 0.25
        assert!(
            grad[1] > grad[0],
            "Gradient at 0 should be larger than at -2"
        );
        assert!(
            grad[1] > grad[2],
            "Gradient at 0 should be larger than at 2"
        );
        assert_abs_diff_eq!(grad[1], 0.25, epsilon = 1e-4);
    }

    #[test]
    fn tanh_output_range() {
        let tanh = Tanh;
        let x = Variable::new(Tensor::rand_uniform(&[100], -10.0, 10.0, Some(42)), false);
        let y = tanh.forward(&x);
        for &v in y.data().to_vec().iter() {
            assert!(
                (-1.0..=1.0).contains(&v),
                "Tanh output {} must be in [-1,1]",
                v
            );
        }
    }

    #[test]
    fn tanh_at_zero_is_zero() {
        let tanh = Tanh;
        let x = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        assert_abs_diff_eq!(tanh.forward(&x).data().item(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn tanh_is_odd_function() {
        let tanh = Tanh;
        let x_pos = Variable::new(Tensor::from_vec(vec![1.5], &[1, 1]), false);
        let x_neg = Variable::new(Tensor::from_vec(vec![-1.5], &[1, 1]), false);
        let y_pos = tanh.forward(&x_pos).data().item();
        let y_neg = tanh.forward(&x_neg).data().item();
        assert_abs_diff_eq!(y_pos, -y_neg, epsilon = 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let softmax = Softmax;
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            false,
        );
        let y = softmax.forward(&x).data().to_vec();
        for row in 0..2 {
            let sum: f32 = (0..3).map(|c| y[row * 3 + c]).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn softmax_all_positive() {
        let softmax = Softmax;
        let x = Variable::new(Tensor::from_vec(vec![-10.0, 0.0, 10.0], &[1, 3]), false);
        let y = softmax.forward(&x).data().to_vec();
        for &v in &y {
            assert!(v > 0.0, "Softmax output must be positive");
        }
    }

    #[test]
    fn softmax_numerical_stability_extreme() {
        let softmax = Softmax;
        let x = Variable::new(
            Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3]),
            false,
        );
        let y = softmax.forward(&x).data().to_vec();
        for &v in &y {
            assert!(!v.is_nan());
            assert!(!v.is_infinite());
        }
        let sum: f32 = y.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn all_activations_no_params() {
        assert!(ReLU.parameters().is_empty());
        assert!(Sigmoid.parameters().is_empty());
        assert!(Tanh.parameters().is_empty());
        assert!(Softmax.parameters().is_empty());
    }
}

// Module: Loss Functions

mod loss_functions {
    use super::*;

    // --- MSE Loss ---

    #[test]
    fn mse_zero_loss() {
        let p = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let t = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let loss = mse_loss(&p, &t);
        assert_abs_diff_eq!(loss.data().item(), 0.0, epsilon = 1e-7);
    }

    #[test]
    fn mse_known_value() {
        // MSE = mean((1-0)^2 + (2-0)^2 + (3-0)^2) = (1+4+9)/3 = 14/3
        let p = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let t = Variable::new(Tensor::from_vec(vec![0.0, 0.0, 0.0], &[1, 3]), false);
        let loss = mse_loss(&p, &t);
        assert_abs_diff_eq!(loss.data().item(), 14.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn mse_symmetry() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 5.0], &[1, 2]), false);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 7.0], &[1, 2]), false);
        let l1 = mse_loss(&a, &b).data().item();
        let l2 = mse_loss(&b, &a).data().item();
        assert_abs_diff_eq!(l1, l2, epsilon = 1e-6);
    }

    #[test]
    fn mse_gradient_known() {
        // MSE grad = 2*(pred-target)/N
        let p = Variable::new(Tensor::from_vec(vec![2.0, 4.0], &[1, 2]), true);
        let t = Variable::new(Tensor::from_vec(vec![1.0, 3.0], &[1, 2]), false);
        let loss = mse_loss(&p, &t);
        loss.backward();
        let grad = p.grad().unwrap().to_vec();
        // 2*(2-1)/2 = 1, 2*(4-3)/2 = 1
        assert_abs_diff_eq!(grad[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(grad[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn mse_scaling() {
        // If we double the difference, MSE should quadruple
        let p1 = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
        let t1 = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        let l1 = mse_loss(&p1, &t1).data().item();

        let p2 = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1]), false);
        let t2 = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        let l2 = mse_loss(&p2, &t2).data().item();

        assert_abs_diff_eq!(l2, 4.0 * l1, epsilon = 1e-5);
    }

    #[test]
    fn mse_batch_sizes() {
        for batch_size in [1, 2, 8, 32] {
            let p = Variable::new(Tensor::ones(&[batch_size, 3]), false);
            let t = Variable::new(Tensor::zeros(&[batch_size, 3]), false);
            let loss = mse_loss(&p, &t);
            // MSE of ones vs zeros = mean(1^2) = 1
            assert_abs_diff_eq!(loss.data().item(), 1.0, epsilon = 1e-5);
        }
    }

    // --- Cross-Entropy Loss ---

    #[test]
    fn ce_perfect_prediction_low_loss() {
        let logits = Variable::new(
            Tensor::from_vec(vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0], &[2, 3]),
            false,
        );
        let targets = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]),
            false,
        );
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(
            loss.data().item() < 0.01,
            "Perfect prediction loss should be near 0"
        );
    }

    #[test]
    fn ce_uniform_logits_equals_log_c() {
        // When logits are uniform, CE = -log(1/C) = log(C)
        let c = 4;
        let logits = Variable::new(Tensor::zeros(&[1, c]), false);
        let targets = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], &[1, c]), false);
        let loss = cross_entropy_loss(&logits, &targets);
        let expected = (c as f32).ln();
        assert_abs_diff_eq!(loss.data().item(), expected, epsilon = 1e-4);
    }

    #[test]
    fn ce_gradient_flow() {
        let logits = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), true);
        let targets = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]), false);
        let loss = cross_entropy_loss(&logits, &targets);
        loss.backward();
        assert!(logits.grad().is_some());
    }

    #[test]
    fn ce_numerical_stability_large_logits() {
        let logits = Variable::new(
            Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3]),
            true,
        );
        let targets = Variable::new(Tensor::from_vec(vec![0.0, 0.0, 1.0], &[1, 3]), false);
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(!loss.data().item().is_nan());
        assert!(!loss.data().item().is_infinite());
    }

    #[test]
    fn ce_batch_sizes() {
        for batch_size in [1, 2, 8] {
            let logits = Variable::new(
                Tensor::rand_uniform(&[batch_size, 5], -1.0, 1.0, Some(42)),
                true,
            );
            let mut target_data = vec![0.0; batch_size * 5];
            for i in 0..batch_size {
                target_data[i * 5] = 1.0; // class 0 for all
            }
            let targets = Variable::new(Tensor::from_vec(target_data, &[batch_size, 5]), false);
            let loss = cross_entropy_loss(&logits, &targets);
            assert!(!loss.data().item().is_nan());
            loss.backward();
            assert!(logits.grad().is_some());
        }
    }

    // --- Huber Loss ---

    #[test]
    fn huber_small_error_quadratic() {
        // |err| = 0.5 < delta=1.0 → loss = 0.5 * 0.5^2 = 0.125
        let p = Variable::new(Tensor::from_vec(vec![1.5], &[1, 1]), false);
        let t = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
        let loss = huber_loss(&p, &t, 1.0);
        assert_abs_diff_eq!(loss.data().item(), 0.125, epsilon = 1e-4);
    }

    #[test]
    fn huber_large_error_linear() {
        // |err| = 5.0 > delta=1.0 → loss = 1.0 * (5.0 - 0.5) = 4.5
        let p = Variable::new(Tensor::from_vec(vec![6.0], &[1, 1]), false);
        let t = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
        let loss = huber_loss(&p, &t, 1.0);
        assert_abs_diff_eq!(loss.data().item(), 4.5, epsilon = 1e-3);
    }

    #[test]
    fn huber_zero_error() {
        let p = Variable::new(Tensor::from_vec(vec![3.0], &[1, 1]), false);
        let t = Variable::new(Tensor::from_vec(vec![3.0], &[1, 1]), false);
        let loss = huber_loss(&p, &t, 1.0);
        assert_abs_diff_eq!(loss.data().item(), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn huber_at_boundary() {
        // |err| = delta=1.0 exact boundary
        // Quadratic: 0.5 * 1.0^2 = 0.5
        // Linear: 1.0 * (1.0 - 0.5) = 0.5
        // Both should give same value at boundary
        let p = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1]), false);
        let t = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
        let loss = huber_loss(&p, &t, 1.0);
        assert_abs_diff_eq!(loss.data().item(), 0.5, epsilon = 1e-3);
    }

    #[test]
    fn huber_gradient_flow() {
        let p = Variable::new(Tensor::from_vec(vec![3.0, 0.5], &[1, 2]), true);
        let t = Variable::new(Tensor::from_vec(vec![1.0, 1.0], &[1, 2]), false);
        let loss = huber_loss(&p, &t, 1.0);
        loss.backward();
        assert!(p.grad().is_some());
    }

    #[test]
    fn huber_different_deltas() {
        let p = Variable::new(Tensor::from_vec(vec![3.0], &[1, 1]), false);
        let t = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        let l1 = huber_loss(&p, &t, 0.5).data().item();
        let l2 = huber_loss(&p, &t, 1.0).data().item();
        let l3 = huber_loss(&p, &t, 2.0).data().item();
        // Larger delta → more quadratic → higher loss at same error
        assert!(
            l1 < l2 && l2 < l3,
            "Larger delta should give more loss: {} {} {}",
            l1,
            l2,
            l3
        );
    }
}

// Module: Sequential Container

mod sequential {
    use super::*;

    #[test]
    fn empty_sequential_is_identity() {
        let model = Sequential::new(vec![]);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let y = model.forward(&x);
        assert_eq!(y.data().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn single_layer_sequential() {
        let model = Sequential::new(vec![Box::new(Linear::new(4, 2))]);
        let x = Variable::new(Tensor::ones(&[3, 4]), false);
        let y = model.forward(&x);
        assert_eq!(y.shape(), vec![3, 2]);
    }

    #[test]
    fn deep_sequential_10_layers() {
        let mut layers: Vec<Box<dyn Module>> = Vec::new();
        let dim = 8;
        for _ in 0..5 {
            layers.push(Box::new(Linear::new(dim, dim)));
            layers.push(Box::new(ReLU));
        }
        let model = Sequential::new(layers);
        let x = Variable::new(Tensor::rand_uniform(&[2, dim], -1.0, 1.0, Some(42)), false);
        let y = model.forward(&x);
        assert_eq!(y.shape(), vec![2, dim]);
    }

    #[test]
    fn parameters_collected_from_all_layers() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)), // 2 params
            Box::new(ReLU),              // 0 params
            Box::new(Linear::new(8, 4)), // 2 params
            Box::new(ReLU),              // 0 params
            Box::new(Linear::new(4, 2)), // 2 params
        ]);
        assert_eq!(model.parameters().len(), 6);
    }

    #[test]
    fn len_and_is_empty() {
        let empty = Sequential::new(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let model = Sequential::new(vec![Box::new(ReLU), Box::new(Sigmoid)]);
        assert!(!model.is_empty());
        assert_eq!(model.len(), 2);
    }

    #[test]
    fn forward_shape_chain() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(10, 20)),
            Box::new(ReLU),
            Box::new(Linear::new(20, 5)),
        ]);
        let x = Variable::new(Tensor::ones(&[4, 10]), false);
        let y = model.forward(&x);
        assert_eq!(y.shape(), vec![4, 5]);
    }

    #[test]
    fn gradient_flows_through_all_layers() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 5)),
            Box::new(ReLU),
            Box::new(Linear::new(5, 2)),
        ]);
        let x = Variable::new(Tensor::ones(&[2, 3]), false);
        let loss = model.forward(&x).sum();
        loss.backward();

        for (i, p) in model.parameters().iter().enumerate() {
            assert!(p.grad().is_some(), "Param {} should have gradient", i);
        }
    }

    #[test]
    fn training_mode_propagation() {
        let mut model = Sequential::new(vec![
            Box::new(Linear::new(4, 4)),
            Box::new(Dropout::new(0.5)),
        ]);
        assert!(model.is_training());
        model.set_training(false);
        assert!(!model.is_training());
    }
}

// Module: Dropout

mod dropout {
    use super::*;

    #[test]
    fn eval_mode_is_identity() {
        let mut d = Dropout::new(0.5);
        d.set_training(false);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]), false);
        let y = d.forward(&x);
        assert_eq!(y.data().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn training_mode_masks_elements() {
        let d = Dropout::new(0.5);
        let x = Variable::new(Tensor::ones(&[100, 100]), false);
        let y = d.forward(&x);
        let data = y.data().to_vec();
        let num_zeros = data.iter().filter(|&&v| v == 0.0).count();
        let zero_ratio = num_zeros as f32 / data.len() as f32;
        assert!(
            (0.35..0.65).contains(&zero_ratio),
            "Expected ~50% zeros, got {:.1}%",
            zero_ratio * 100.0
        );
    }

    #[test]
    fn inverted_scaling_correct() {
        let d = Dropout::new(0.5);
        let x = Variable::new(Tensor::ones(&[100, 100]), false);
        let y = d.forward(&x);
        let data = y.data().to_vec();
        let non_zero: Vec<f32> = data.into_iter().filter(|&v| v != 0.0).collect();
        // Non-zero values should be scaled by 1/(1-0.5) = 2.0
        for &v in &non_zero {
            assert_abs_diff_eq!(v, 2.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn p_zero_no_dropout() {
        let d = Dropout::new(0.0);
        let x = Variable::new(Tensor::ones(&[10, 10]), false);
        let y = d.forward(&x);
        assert_eq!(y.data().to_vec(), vec![1.0; 100]);
    }

    #[test]
    fn high_dropout_rate() {
        let d = Dropout::new(0.99);
        let x = Variable::new(Tensor::ones(&[1000]), false);
        let y = d.forward(&x);
        let data = y.data().to_vec();
        let num_zeros = data.iter().filter(|&&v| v == 0.0).count();
        let zero_ratio = num_zeros as f32 / data.len() as f32;
        assert!(
            zero_ratio > 0.9,
            "With p=0.99, expected >90% zeros, got {:.1}%",
            zero_ratio * 100.0
        );
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn invalid_p_1_panics() {
        Dropout::new(1.0);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn invalid_p_negative_panics() {
        Dropout::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn invalid_p_over_1_panics() {
        Dropout::new(1.5);
    }

    #[test]
    fn expected_value_preservation() {
        // Mean of many dropout runs should approximate the original
        let d = Dropout::new(0.3);
        let x = Variable::new(Tensor::full(&[1000], 5.0), false);
        let n_trials = 50;
        let mut total_mean = 0.0;
        for _ in 0..n_trials {
            let y = d.forward(&x);
            total_mean += y.data().mean().item();
        }
        total_mean /= n_trials as f32;
        // Should be close to 5.0 due to inverted scaling
        assert_abs_diff_eq!(total_mean, 5.0, epsilon = 0.5);
    }

    #[test]
    fn dropout_variance_with_inverted_scaling() {
        // With inverted dropout, variance of output = var(x) * (1/(1-p))
        // For constant input x=1: output is 0 or 1/(1-p) with probabilities p and (1-p)
        // E[Y] = 1, E[Y^2] = (1-p)*(1/(1-p))^2 = 1/(1-p), Var[Y] = p/(1-p)
        let p = 0.5_f32;
        let d = Dropout::new(p);
        let x = Variable::new(Tensor::ones(&[10000]), false);
        let y = d.forward(&x);
        let data = y.data().to_vec();

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;

        let expected_var = p / (1.0 - p); // = 1.0 for p=0.5
        assert_abs_diff_eq!(var, expected_var, epsilon = 0.15);
    }

    #[test]
    fn training_flag_toggling() {
        let mut d = Dropout::new(0.5);
        assert!(d.is_training());
        d.set_training(false);
        assert!(!d.is_training());
        d.set_training(true);
        assert!(d.is_training());
    }
}

// Module: LayerNorm

mod layer_norm {
    use super::*;

    #[test]
    fn output_shape_preserved() {
        let ln = LayerNorm::new(4);
        let x = Variable::new(Tensor::rand_uniform(&[3, 4], -1.0, 1.0, Some(42)), false);
        let y = ln.forward(&x);
        assert_eq!(y.shape(), vec![3, 4]);
    }

    #[test]
    fn output_mean_approximately_zero() {
        let ln = LayerNorm::new(8);
        let x = Variable::new(Tensor::rand_uniform(&[5, 8], -10.0, 10.0, Some(42)), false);
        let y = ln.forward(&x);
        let data = y.data().to_vec();
        for row in 0..5 {
            let mean: f32 = (0..8).map(|c| data[row * 8 + c]).sum::<f32>() / 8.0;
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn output_std_approximately_one() {
        let ln = LayerNorm::new(8);
        let x = Variable::new(
            Tensor::from_vec((0..40).map(|i| i as f32).collect(), &[5, 8]),
            false,
        );
        let y = ln.forward(&x);
        let data = y.data().to_vec();
        for row in 0..5 {
            let mean: f32 = (0..8).map(|c| data[row * 8 + c]).sum::<f32>() / 8.0;
            let var: f32 = (0..8)
                .map(|c| (data[row * 8 + c] - mean).powi(2))
                .sum::<f32>()
                / 8.0;
            let std = var.sqrt();
            assert_abs_diff_eq!(std, 1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn parameters_gamma_beta() {
        let ln = LayerNorm::new(16);
        let params = ln.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), vec![16]); // gamma
        assert_eq!(params[1].shape(), vec![16]); // beta
    }

    #[test]
    fn gradient_flows_through_all() {
        let ln = LayerNorm::new(4);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let loss = ln.forward(&x).sum();
        loss.backward();
        assert!(x.grad().is_some(), "Input should have gradient");
        let params = ln.parameters();
        assert!(params[0].grad().is_some(), "Gamma should have gradient");
        assert!(params[1].grad().is_some(), "Beta should have gradient");
    }

    #[test]
    fn constant_input_stable() {
        // All same values → variance = 0 → eps prevents NaN
        let ln = LayerNorm::new(4);
        let x = Variable::new(Tensor::full(&[2, 4], 5.0), false);
        let y = ln.forward(&x);
        for &v in y.data().to_vec().iter() {
            assert!(!v.is_nan(), "Constant input should not produce NaN");
            assert!(!v.is_infinite(), "Constant input should not produce Inf");
        }
    }

    #[test]
    fn different_batch_sizes() {
        let ln = LayerNorm::new(4);
        for batch in [1, 2, 8, 32] {
            let x = Variable::new(
                Tensor::rand_uniform(&[batch, 4], -1.0, 1.0, Some(42)),
                false,
            );
            let y = ln.forward(&x);
            assert_eq!(y.shape(), vec![batch, 4]);
        }
    }

    #[test]
    fn custom_eps() {
        let ln = LayerNorm::with_eps(4, 1e-3);
        let x = Variable::new(Tensor::ones(&[2, 4]), false);
        let y = ln.forward(&x);
        for &v in y.data().to_vec().iter() {
            assert!(!v.is_nan());
        }
    }
}
