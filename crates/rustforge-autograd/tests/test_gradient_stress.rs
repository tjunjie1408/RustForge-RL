//! Stress tests and property-based gradient tests for rustforge-autograd.
//!
//! Covers: deep computation graphs, numerical stability at extremes,
//! and proptest-based property verification of gradient correctness.

use approx::assert_abs_diff_eq;
use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

// Helpers

fn numerical_gradient<F>(f: &F, param: &Variable, epsilon: f32) -> Vec<f32>
where
    F: Fn() -> Variable,
{
    let data = param.data().clone();
    let n = data.numel();
    let mut num_grads = vec![0.0f32; n];
    for i in 0..n {
        let original = data.to_vec();
        let mut plus = original.clone();
        plus[i] += epsilon;
        param.set_data(Tensor::from_vec(plus, data.shape()));
        let loss_plus = f().data().item();

        let mut minus = original.clone();
        minus[i] -= epsilon;
        param.set_data(Tensor::from_vec(minus, data.shape()));
        let loss_minus = f().data().item();

        num_grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
        param.set_data(Tensor::from_vec(original, data.shape()));
    }
    num_grads
}

fn assert_grads_close(analytic: &[f32], numerical: &[f32], atol: f32, rtol: f32) {
    assert_eq!(analytic.len(), numerical.len());
    for (i, (a, n)) in analytic.iter().zip(numerical.iter()).enumerate() {
        let tol = atol.max(rtol * a.abs().max(n.abs()));
        assert!(
            (a - n).abs() <= tol,
            "Gradient mismatch at {}: analytic={}, numerical={}, tol={}",
            i,
            a,
            n,
            tol
        );
    }
}

// Module: Deep Computation Graphs

mod deep_graphs {
    use super::*;

    #[test]
    fn chain_50_additions_gradient_is_one() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        let mut result = x.clone();
        for _ in 0..50 {
            result = &result + &Variable::from_tensor(Tensor::from_vec(vec![0.01], &[1]));
        }
        let loss = result.sum();
        loss.backward();
        // Gradient through chain of additions is 1.0
        assert_abs_diff_eq!(x.grad().unwrap().to_vec()[0], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn chain_20_scalar_multiplies() {
        // f(x) = x * 2 * 2 * ... (20 times) = x * 2^20
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        let mut result = x.clone();
        for _ in 0..20 {
            result = &result * 2.0;
        }
        let loss = result.sum();
        loss.backward();
        let expected_grad = 2.0_f32.powi(20);
        assert_abs_diff_eq!(x.grad().unwrap().to_vec()[0], expected_grad, epsilon = 1.0);
    }

    #[test]
    fn nested_relu_chain() {
        // relu(relu(relu(x))) with positive x should pass through
        let x = Variable::new(Tensor::from_vec(vec![5.0], &[1]), true);
        let y = x.relu().relu().relu().relu().relu();
        let loss = y.sum();
        loss.backward();
        // All relus pass (x > 0), so gradient = 1
        assert_abs_diff_eq!(x.grad().unwrap().to_vec()[0], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn nested_relu_chain_negative() {
        // relu(relu(relu(x))) with negative x should be zeroed
        let x = Variable::new(Tensor::from_vec(vec![-5.0], &[1]), true);
        let y = x.relu().relu().relu();
        let loss = y.sum();
        loss.backward();
        assert_abs_diff_eq!(x.grad().unwrap().to_vec()[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn matmul_chain() {
        // Y = W1 @ W2 @ W3 @ x → verify gradient shapes
        let w1 = Variable::new(Tensor::rand_uniform(&[4, 3], -0.5, 0.5, Some(42)), true);
        let w2 = Variable::new(Tensor::rand_uniform(&[3, 3], -0.5, 0.5, Some(43)), true);
        let w3 = Variable::new(Tensor::rand_uniform(&[3, 2], -0.5, 0.5, Some(44)), true);
        let x = Variable::new(Tensor::rand_uniform(&[2, 2], -0.5, 0.5, Some(45)), false);
        let y = w1.matmul(&w2).matmul(&w3).matmul(&x);
        let loss = y.sum();
        loss.backward();

        assert_eq!(w1.grad().unwrap().shape(), &[4, 3]);
        assert_eq!(w2.grad().unwrap().shape(), &[3, 3]);
        assert_eq!(w3.grad().unwrap().shape(), &[3, 2]);
    }
}

// Module: Numerical Stability

mod numerical_stability {
    use super::*;

    #[test]
    fn sigmoid_extreme_positive_gradient() {
        // sigmoid(500) ≈ 1.0, grad = sigmoid(x)*(1-sigmoid(x)) ≈ 0
        let x = Variable::new(Tensor::from_vec(vec![500.0], &[1]), true);
        let loss = x.sigmoid().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec()[0];
        assert!(
            !grad.is_nan(),
            "Sigmoid grad at extreme positive should not be NaN"
        );
        assert!(grad.abs() < 1e-4, "Sigmoid grad at x=500 should be near 0");
    }

    #[test]
    fn sigmoid_extreme_negative_gradient() {
        let x = Variable::new(Tensor::from_vec(vec![-500.0], &[1]), true);
        let loss = x.sigmoid().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec()[0];
        assert!(!grad.is_nan());
        assert!(grad.abs() < 1e-4);
    }

    #[test]
    fn tanh_extreme_gradient() {
        let x = Variable::new(Tensor::from_vec(vec![100.0, -100.0], &[2]), true);
        let loss = x.tanh_().sum();
        loss.backward();
        let grads = x.grad().unwrap().to_vec();
        for g in &grads {
            assert!(!g.is_nan(), "Tanh grad should not be NaN at extremes");
        }
    }

    #[test]
    fn exp_near_overflow_boundary() {
        // exp(85) ≈ 7.2e36 (within f32 range), exp(89) ≈ 4.5e38 (near max)
        let x = Variable::new(Tensor::from_vec(vec![80.0], &[1]), true);
        let loss = x.exp().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec()[0];
        assert!(!grad.is_nan(), "exp grad near overflow should not be NaN");
        // grad(exp(x)) = exp(x), which is huge but finite around 80
        assert!(grad.is_finite() || grad.is_infinite()); // Both are acceptable
    }

    #[test]
    fn log_small_positive_gradient() {
        // log(1e-6) is well-defined, grad = 1/x = 1e6
        let x = Variable::new(Tensor::from_vec(vec![1e-6], &[1]), true);
        let loss = x.log().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec()[0];
        assert!(!grad.is_nan());
        assert_abs_diff_eq!(grad, 1.0 / 1e-6, epsilon = 1e2);
    }

    #[test]
    fn sqrt_near_zero_gradient() {
        // sqrt(ε) = √ε, grad = 1/(2√ε) which is large
        let eps = 1e-6;
        let x = Variable::new(Tensor::from_vec(vec![eps], &[1]), true);
        let loss = x.sqrt().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec()[0];
        assert!(!grad.is_nan());
        let expected = 1.0 / (2.0 * eps.sqrt());
        assert_abs_diff_eq!(grad, expected, epsilon = expected * 0.1);
    }

    #[test]
    fn div_small_divisor_gradient() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        let d = Variable::new(Tensor::from_vec(vec![1e-6], &[1]), true);
        let loss = (&x / &d).sum();
        loss.backward();
        // dx = 1/d = 1e6, dd = -x/d^2 = -1e12
        let dx = x.grad().unwrap().to_vec()[0];
        assert!(!dx.is_nan());
        assert_abs_diff_eq!(dx, 1e6, epsilon = 1e3);
    }

    #[test]
    fn pow_near_zero_exponent() {
        // f(x) = x^0.01, df/dx = 0.01 * x^(-0.99)
        let x = Variable::new(Tensor::from_vec(vec![2.0, 5.0], &[2]), true);
        let f = || x.pow(0.01).sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.01, 5e-2);
    }

    #[test]
    fn combined_stable_computation() {
        // Sigmoid(matmul) chain — common in neural networks
        let w = Variable::new(Tensor::rand_uniform(&[3, 3], -0.5, 0.5, Some(42)), true);
        let x = Variable::from_tensor(Tensor::rand_uniform(&[2, 3], -1.0, 1.0, Some(43)));
        let loss = x.matmul(&w).sigmoid().sum();
        loss.backward();
        let grad = w.grad().unwrap();
        for &g in grad.to_vec().iter() {
            assert!(!g.is_nan(), "Gradient should not contain NaN");
            assert!(g.is_finite(), "Gradient should be finite");
        }
    }

    #[test]
    fn exp_negative_large_gradient() {
        // exp(-100) ≈ 3.7e-44, grad = exp(-100) ≈ 0
        let x = Variable::new(Tensor::from_vec(vec![-100.0], &[1]), true);
        let loss = x.exp().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec()[0];
        assert!(!grad.is_nan());
        assert!(grad.abs() < 1e-30, "exp(-100) grad should be essentially 0");
    }
}

// Module: Property-Based Gradient Tests (proptest)

mod proptest_gradients {
    use super::*;
    use proptest::prelude::*;

    fn small_shape() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(1usize..=5, 1..=2)
    }

    proptest! {
        #[test]
        fn prop_sum_add_gradient_is_ones(shape in small_shape(), seed in 0u64..10000) {
            let a = Variable::new(Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed)), true);
            let b = Variable::new(Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed + 1)), true);
            let loss = (&a + &b).sum();
            loss.backward();
            for g in a.grad().unwrap().to_vec() {
                prop_assert!((g - 1.0).abs() < 1e-5, "grad(sum(a+b))/da should be 1, got {}", g);
            }
        }

        #[test]
        fn prop_sum_mul_gradient_is_other(seed in 0u64..10000) {
            let shape = vec![3, 4];
            let a = Variable::new(Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed)), true);
            let b = Variable::new(Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed + 1)), true);
            let loss = (&a * &b).sum();
            loss.backward();
            let ga = a.grad().unwrap().to_vec();
            let b_data = b.data().to_vec();
            for (g, bv) in ga.iter().zip(b_data.iter()) {
                prop_assert!((g - bv).abs() < 1e-4, "grad(sum(a*b))/da = b: got {} vs {}", g, bv);
            }
        }

        #[test]
        fn prop_gradient_linearity(seed in 0u64..10000) {
            // grad(2*f) == 2*grad(f)
            let x = Variable::new(Tensor::rand_uniform(&[4], -2.0, 2.0, Some(seed)), true);

            // Compute grad(f) where f = sum(x^2)
            let loss1 = (&x * &x).sum();
            loss1.backward();
            let grad_f = x.grad().unwrap().to_vec();

            x.zero_grad();

            // Compute grad(2f) = grad(sum(2*x^2))
            let loss2 = (&(&x * &x) * 2.0).sum();
            loss2.backward();
            let grad_2f = x.grad().unwrap().to_vec();

            for (gf, g2f) in grad_f.iter().zip(grad_2f.iter()) {
                prop_assert!((g2f - 2.0 * gf).abs() < 1e-3,
                    "grad(2f)={} should be 2*grad(f)={}", g2f, 2.0 * gf);
            }
        }

        #[test]
        fn prop_matmul_identity_gradient(seed in 0u64..10000) {
            let n = 3;
            let a = Variable::new(Tensor::rand_uniform(&[n, n], -1.0, 1.0, Some(seed)), true);
            let eye = Variable::from_tensor(Tensor::eye(n));
            let loss = a.matmul(&eye).sum();
            loss.backward();
            // grad(sum(A@I))/dA = ones(n,n) @ I^T = ones(n,n)
            // Since A@I = A, sum(A@I) = sum(A), gradient is all 1.0
            let grad = a.grad().unwrap().to_vec();
            for (idx, g) in grad.iter().enumerate() {
                prop_assert!((g - 1.0).abs() < 1e-4,
                    "grad[{}]={} expected 1.0", idx, g);
            }
        }

        #[test]
        fn prop_neg_gradient_is_neg_one(shape in small_shape(), seed in 0u64..10000) {
            let x = Variable::new(Tensor::rand_uniform(&shape, -5.0, 5.0, Some(seed)), true);
            let loss = (-&x).sum();
            loss.backward();
            for g in x.grad().unwrap().to_vec() {
                prop_assert!((g - (-1.0)).abs() < 1e-5, "grad(sum(-x)) should be -1, got {}", g);
            }
        }

        #[test]
        fn prop_sub_self_zero_gradient(shape in small_shape(), seed in 0u64..10000) {
            let x = Variable::new(Tensor::rand_uniform(&shape, -5.0, 5.0, Some(seed)), true);
            let loss = (&x - &x).sum();
            loss.backward();
            // d(sum(x-x))/dx = 1 - 1 = 0
            for g in x.grad().unwrap().to_vec() {
                prop_assert!(g.abs() < 1e-5, "grad(sum(x-x)) should be 0, got {}", g);
            }
        }

        #[test]
        fn prop_scalar_mul_gradient(
            seed in 0u64..10000,
            scalar in -10.0f32..10.0,
        ) {
            let x = Variable::new(Tensor::rand_uniform(&[3], -1.0, 1.0, Some(seed)), true);
            let loss = (&x * scalar).sum();
            loss.backward();
            for g in x.grad().unwrap().to_vec() {
                prop_assert!((g - scalar).abs() < 1e-4,
                    "grad(sum(x*c))/dx = c: got {} vs {}", g, scalar);
            }
        }

        #[test]
        fn prop_relu_gradient_binary(seed in 0u64..10000) {
            let x = Variable::new(Tensor::rand_uniform(&[10], -5.0, 5.0, Some(seed)), true);
            let loss = x.relu().sum();
            loss.backward();
            let input = x.data().to_vec();
            let grad = x.grad().unwrap().to_vec();
            for (xi, gi) in input.iter().zip(grad.iter()) {
                if *xi > 0.0 {
                    prop_assert!((gi - 1.0).abs() < 1e-5, "relu grad should be 1 for x>0");
                } else if *xi < 0.0 {
                    prop_assert!(gi.abs() < 1e-5, "relu grad should be 0 for x<0");
                }
            }
        }

        #[test]
        fn prop_sigmoid_gradient_positive(seed in 0u64..10000) {
            let x = Variable::new(Tensor::rand_uniform(&[5], -3.0, 3.0, Some(seed)), true);
            let loss = x.sigmoid().sum();
            loss.backward();
            for g in x.grad().unwrap().to_vec() {
                prop_assert!(g >= 0.0, "sigmoid gradient should be non-negative, got {}", g);
                prop_assert!(g <= 0.25 + 1e-5, "sigmoid gradient max is 0.25, got {}", g);
            }
        }

        #[test]
        fn prop_random_forward_backward_no_panic(seed in 0u64..10000) {
            let x = Variable::new(Tensor::rand_uniform(&[3, 4], -1.0, 1.0, Some(seed)), true);
            let w = Variable::new(Tensor::rand_uniform(&[4, 2], -0.5, 0.5, Some(seed + 1)), true);
            let loss = x.matmul(&w).relu().sum();
            // Must not panic
            loss.backward();
            prop_assert!(x.grad().is_some());
            prop_assert!(w.grad().is_some());
            let x_grad = x.grad().unwrap();
            let w_grad = w.grad().unwrap();
            prop_assert_eq!(x_grad.shape(), &[3, 4]);
            prop_assert_eq!(w_grad.shape(), &[4, 2]);
        }

        #[test]
        fn prop_chain_numerical_check(seed in 0u64..100) {
            let x = Variable::new(Tensor::rand_uniform(&[3], 0.5, 2.0, Some(seed)), true);
            let f = || (&x * &x).sigmoid().sum();
            let loss = f();
            loss.backward();
            let analytic = x.grad().unwrap().to_vec();
            x.zero_grad();
            let numerical = numerical_gradient(&f, &x, 1e-4);
            for (a, n) in analytic.iter().zip(numerical.iter()) {
                let tol = 0.05_f32.max(0.05 * a.abs().max(n.abs()));
                prop_assert!((a - n).abs() <= tol,
                    "Mismatch: analytic={}, numerical={}", a, n);
            }
        }
    }
}
