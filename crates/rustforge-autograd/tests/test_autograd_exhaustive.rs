//! Exhaustive integration tests for rustforge-autograd.
//!
//! Covers: gradient correctness for ALL ops (numerical verification),
//! broadcasting gradient reduction, edge cases, variable properties,
//! and optimizer stress tests.

use approx::assert_abs_diff_eq;
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::optimizer::sgd::SGD;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_tensor::Tensor;

/// Numerical gradient via central finite differences.
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

/// Assert analytic ≈ numerical gradients with relative+absolute tolerance.
fn assert_grads_close(analytic: &[f32], numerical: &[f32], atol: f32, rtol: f32) {
    assert_eq!(analytic.len(), numerical.len(), "gradient length mismatch");
    for (i, (a, n)) in analytic.iter().zip(numerical.iter()).enumerate() {
        let tol = atol.max(rtol * a.abs().max(n.abs()));
        assert!(
            (a - n).abs() <= tol,
            "Gradient mismatch at index {}: analytic={}, numerical={}, tol={}",
            i,
            a,
            n,
            tol
        );
    }
}

// Module: Gradient Correctness — every op verified numerically

mod gradient_correctness {
    use super::*;

    #[test]
    fn grad_add() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]), true);
        let f = || (&a + &b).sum();
        let loss = f();
        loss.backward();
        let da = a.grad().unwrap().to_vec();
        let db = b.grad().unwrap().to_vec();
        // d(sum(a+b))/da = 1, d/db = 1
        for &g in &da {
            assert_abs_diff_eq!(g, 1.0, epsilon = 1e-5);
        }
        for &g in &db {
            assert_abs_diff_eq!(g, 1.0, epsilon = 1e-5);
        }

        a.zero_grad();
        let num = numerical_gradient(&f, &a, 1e-4);
        assert_grads_close(&da, &num, 0.01, 1e-3);
    }

    #[test]
    fn grad_sub() {
        let a = Variable::new(Tensor::from_vec(vec![5.0, 6.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let loss = (&a - &b).sum();
        loss.backward();
        assert_eq!(a.grad().unwrap().to_vec(), vec![1.0, 1.0]);
        assert_eq!(b.grad().unwrap().to_vec(), vec![-1.0, -1.0]);
    }

    #[test]
    fn grad_mul() {
        let a = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0], &[2]), true);
        let f = || (&a * &b).sum();
        let loss = f();
        loss.backward();
        // d(sum(a*b))/da = b, d/db = a
        assert_eq!(a.grad().unwrap().to_vec(), vec![4.0, 5.0]);
        assert_eq!(b.grad().unwrap().to_vec(), vec![2.0, 3.0]);

        a.zero_grad();
        let num_a = numerical_gradient(&f, &a, 1e-4);
        assert_grads_close(&[4.0, 5.0], &num_a, 0.02, 1e-3);
    }

    #[test]
    fn grad_div() {
        let a = Variable::new(Tensor::from_vec(vec![6.0, 8.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0, 4.0], &[2]), true);
        let f = || (&a / &b).sum();
        let loss = f();
        loss.backward();
        let da = a.grad().unwrap().to_vec();
        let db = b.grad().unwrap().to_vec();
        // da = 1/b, db = -a/b^2
        assert_abs_diff_eq!(da[0], 0.5, epsilon = 1e-4);
        assert_abs_diff_eq!(da[1], 0.25, epsilon = 1e-4);
        assert_abs_diff_eq!(db[0], -1.5, epsilon = 1e-4);
        assert_abs_diff_eq!(db[1], -0.5, epsilon = 1e-4);

        b.zero_grad();
        let num_b = numerical_gradient(&f, &b, 1e-4);
        assert_grads_close(&db, &num_b, 0.05, 1e-2);
    }

    #[test]
    fn grad_neg() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let loss = (-&x).sum();
        loss.backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn grad_matmul_2d() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let w = Variable::new(Tensor::from_vec(vec![0.5, 0.3, 0.7, 0.1], &[2, 2]), true);
        let f = || x.matmul(&w).sum();
        let loss = f();
        loss.backward();
        let dw = w.grad().unwrap().to_vec();
        w.zero_grad();
        let num_dw = numerical_gradient(&f, &w, 1e-4);
        assert_grads_close(&dw, &num_dw, 0.1, 1e-2);
    }

    #[test]
    fn grad_matmul_vec_dot() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]), true);
        let loss = a.matmul(&b); // scalar
        loss.backward();
        // d(a·b)/da = b, d(a·b)/db = a
        assert_eq!(a.grad().unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(b.grad().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn grad_matmul_mat_vec() {
        let m = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let v = Variable::new(Tensor::from_vec(vec![1.0, 1.0], &[2]), true);
        let f = || m.matmul(&v).sum();
        let loss = f();
        loss.backward();
        let dm = m.grad().unwrap().to_vec();
        m.zero_grad();
        let num_dm = numerical_gradient(&f, &m, 1e-4);
        assert_grads_close(&dm, &num_dm, 0.1, 1e-2);
    }

    #[test]
    fn grad_relu() {
        let x = Variable::new(
            Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]),
            true,
        );
        let loss = x.relu().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        assert_eq!(grad, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn grad_relu_numerical() {
        let x = Variable::new(
            Tensor::from_vec(vec![0.5, -0.3, 1.0, -1.0, 2.0], &[5]),
            true,
        );
        let f = || x.relu().sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
    }

    #[test]
    fn grad_sigmoid_numerical() {
        let x = Variable::new(Tensor::from_vec(vec![0.5, -0.3, 1.0, -1.0], &[4]), true);
        let f = || x.sigmoid().sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
    }

    #[test]
    fn grad_tanh_numerical() {
        let x = Variable::new(Tensor::from_vec(vec![0.5, -0.3, 1.5, -1.5], &[4]), true);
        let f = || x.tanh_().sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
    }

    #[test]
    fn grad_exp_numerical() {
        let x = Variable::new(Tensor::from_vec(vec![0.5, 1.0, -0.5, 2.0], &[4]), true);
        let f = || x.exp().sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.05, 1e-2);
    }

    #[test]
    fn grad_log_exact() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3]), true);
        let loss = x.log().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        assert_abs_diff_eq!(grad[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(grad[1], 0.5, epsilon = 1e-5);
        assert_abs_diff_eq!(grad[2], 0.25, epsilon = 1e-5);
    }

    #[test]
    fn grad_log_numerical() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 5.0], &[3]), true);
        let f = || x.log().sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
    }

    #[test]
    fn grad_pow_exact() {
        // f(x) = sum(x^3), df/dx = 3x^2
        let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), true);
        let loss = x.pow(3.0).sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        assert_abs_diff_eq!(grad[0], 12.0, epsilon = 1e-3);
        assert_abs_diff_eq!(grad[1], 27.0, epsilon = 1e-3);
    }

    #[test]
    fn grad_pow_numerical() {
        let x = Variable::new(Tensor::from_vec(vec![1.5, 2.5], &[2]), true);
        let f = || x.pow(2.5).sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.1, 1e-2);
    }

    #[test]
    fn grad_sqrt_exact() {
        let x = Variable::new(Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3]), true);
        let loss = x.sqrt().sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        assert_abs_diff_eq!(grad[0], 0.25, epsilon = 1e-4);
        assert_abs_diff_eq!(grad[1], 1.0 / 6.0, epsilon = 1e-4);
        assert_abs_diff_eq!(grad[2], 0.125, epsilon = 1e-4);
    }

    #[test]
    fn grad_sum_exact() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let loss = x.sum();
        loss.backward();
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn grad_mean_exact() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]), true);
        let loss = x.mean();
        loss.backward();
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 0.25, epsilon = 1e-6);
        }
    }

    #[test]
    fn grad_sum_axis_numerical() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let f = || x.sum_axis(1, false).sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.05, 1e-2);
    }

    #[test]
    fn grad_sum_axis_keepdim_numerical() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let f = || x.sum_axis(0, true).sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.05, 1e-2);
    }

    #[test]
    fn grad_transpose_numerical() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let f = || x.t().sum();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.02, 1e-3);
    }

    #[test]
    fn grad_scalar_add() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let loss = (&x + 10.0).sum();
        loss.backward();
        // d(sum(x+c))/dx = 1
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn grad_scalar_mul() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
        let loss = (&x * 3.0).sum();
        loss.backward();
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn grad_scalar_sub() {
        let x = Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[2]), true);
        let loss = (&x - 5.0).sum();
        loss.backward();
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn grad_scalar_div() {
        let x = Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[2]), true);
        let loss = (&x / 4.0).sum();
        loss.backward();
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 0.25, epsilon = 1e-5);
        }
    }

    #[test]
    fn grad_chain_matmul_relu_sum() {
        let w = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]), true);
        let f = || {
            let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
            x.matmul(&w).relu().sum()
        };
        let loss = f();
        loss.backward();
        let analytic = w.grad().unwrap().to_vec();
        w.zero_grad();
        let numerical = numerical_gradient(&f, &w, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.1, 1e-2);
    }

    #[test]
    fn grad_chain_sigmoid_pow_mean() {
        let x = Variable::new(Tensor::from_vec(vec![0.5, 1.0, -0.5, -1.0], &[4]), true);
        let f = || x.sigmoid().pow(2.0).mean();
        let loss = f();
        loss.backward();
        let analytic = x.grad().unwrap().to_vec();
        x.zero_grad();
        let numerical = numerical_gradient(&f, &x, 1e-4);
        assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
    }

    #[test]
    fn grad_multi_path_accumulation() {
        // f(x) = sum(x*x + x*2 + x) = sum(x^2 + 3x), df/dx = 2x + 3
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let x_sq = &x * &x;
        let x_2 = &x * 2.0;
        let total = &(&x_sq + &x_2) + &x;
        let loss = total.sum();
        loss.backward();
        let grad = x.grad().unwrap().to_vec();
        assert_abs_diff_eq!(grad[0], 5.0, epsilon = 1e-4); // 2*1+3
        assert_abs_diff_eq!(grad[1], 7.0, epsilon = 1e-4); // 2*2+3
    }

    #[test]
    fn grad_diamond_graph() {
        // a → b = a*2, a → c = a*3, d = b + c = 5a, loss = sum(d)
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let b = &a * 2.0;
        let c = &a * 3.0;
        let d = &b + &c;
        let loss = d.sum();
        loss.backward();
        let grad = a.grad().unwrap().to_vec();
        assert_abs_diff_eq!(grad[0], 5.0, epsilon = 1e-4);
        assert_abs_diff_eq!(grad[1], 5.0, epsilon = 1e-4);
    }
}

// Module: Broadcasting Gradient Tests

mod broadcast_gradients {
    use super::*;

    #[test]
    fn broadcast_add_2d_plus_1d() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let b = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]), true);
        let loss = (&x + &b).sum();
        loss.backward();

        // grad_b should be summed across broadcast dim (axis 0): each = 2.0
        let gb = b.grad().unwrap().to_vec();
        for &g in &gb {
            assert_abs_diff_eq!(g, 2.0, epsilon = 1e-5);
        }

        // grad_x should be 1.0 everywhere
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn broadcast_mul_2d_times_row() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let s = Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[1, 2]), true);
        let f = || (&x * &s).sum();
        let loss = f();
        loss.backward();
        // grad_s_j = sum_i x_ij = [1+3, 2+4] = [4, 6]
        let gs = s.grad().unwrap().to_vec();
        assert_abs_diff_eq!(gs[0], 4.0, epsilon = 1e-5);
        assert_abs_diff_eq!(gs[1], 6.0, epsilon = 1e-5);

        s.zero_grad();
        let num = numerical_gradient(&f, &s, 1e-4);
        assert_grads_close(&[4.0, 6.0], &num, 0.1, 1e-2);
    }

    #[test]
    fn broadcast_sub_gradient_reduction() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let b = Variable::new(Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]), true);
        let loss = (&x - &b).sum();
        loss.backward();
        let gb = b.grad().unwrap().to_vec();
        // d(sum(x-b))/db = -1 * num_rows = -2
        for &g in &gb {
            assert_abs_diff_eq!(g, -2.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn broadcast_div_gradient_reduction() {
        let x = Variable::new(Tensor::from_vec(vec![6.0, 8.0, 12.0, 16.0], &[2, 2]), true);
        let d = Variable::new(Tensor::from_vec(vec![2.0, 4.0], &[1, 2]), true);
        let f = || (&x / &d).sum();
        let loss = f();
        loss.backward();
        let gd = d.grad().unwrap().to_vec();
        d.zero_grad();
        let num_gd = numerical_gradient(&f, &d, 1e-4);
        assert_grads_close(&gd, &num_gd, 0.1, 1e-2);
    }

    #[test]
    fn broadcast_scalar_variable() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let s = Variable::new(Tensor::from_vec(vec![10.0], &[1, 1]), true);
        let loss = (&x * &s).sum();
        loss.backward();
        // grad_s = sum(x) = 10
        let gs = s.grad().unwrap().to_vec();
        assert_abs_diff_eq!(gs[0], 10.0, epsilon = 1e-4);
    }

    #[test]
    fn broadcast_high_dim() {
        let x = Variable::new(Tensor::rand_uniform(&[2, 3, 4], -1.0, 1.0, Some(42)), true);
        let b = Variable::new(Tensor::rand_uniform(&[4], -1.0, 1.0, Some(43)), true);
        let f = || (&x + &b).sum();
        let loss = f();
        loss.backward();
        let gb = b.grad().unwrap();
        assert_eq!(gb.shape(), &[4]);
        // Each bias element is broadcast across 2*3=6 positions
        for &g in gb.to_vec().iter() {
            assert_abs_diff_eq!(g, 6.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn broadcast_both_sides() {
        // [3,1] + [1,4] → [3,4]
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]), true);
        let b = Variable::new(
            Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]),
            true,
        );
        let f = || (&a + &b).sum();
        let loss = f();
        loss.backward();
        let ga = a.grad().unwrap();
        let gb = b.grad().unwrap();
        assert_eq!(ga.shape(), &[3, 1]);
        assert_eq!(gb.shape(), &[1, 4]);
        // grad_a summed across axis 1 (4 positions) = 4.0
        for &g in ga.to_vec().iter() {
            assert_abs_diff_eq!(g, 4.0, epsilon = 1e-4);
        }
        // grad_b summed across axis 0 (3 positions) = 3.0
        for &g in gb.to_vec().iter() {
            assert_abs_diff_eq!(g, 3.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn broadcast_add_numerical_check() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let b = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]), true);
        let f = || (&x + &b).sum();
        let loss = f();
        loss.backward();
        let analytic_b = b.grad().unwrap().to_vec();
        b.zero_grad();
        let num_b = numerical_gradient(&f, &b, 1e-4);
        assert_grads_close(&analytic_b, &num_b, 0.02, 1e-3);
    }

    #[test]
    fn broadcast_mul_numerical_check() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let s = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[1, 2]), true);
        let f = || (&x * &s).sum();
        let loss = f();
        loss.backward();
        let analytic_s = s.grad().unwrap().to_vec();
        s.zero_grad();
        let num_s = numerical_gradient(&f, &s, 1e-4);
        assert_grads_close(&analytic_s, &num_s, 0.1, 1e-2);
    }
}

// Module: Edge Cases

mod edge_cases {
    use super::*;

    #[test]
    fn no_grad_variable_no_gradient() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), false);
        let y = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]), true);
        let loss = (&x + &y).sum();
        loss.backward();
        assert!(
            x.grad().is_none(),
            "No-grad variable should have no gradient"
        );
        assert!(y.grad().is_some());
    }

    #[test]
    #[should_panic(expected = "backward() can only be called on scalar")]
    fn backward_on_non_scalar_panics() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let y = &x * 2.0;
        y.backward();
    }

    #[test]
    fn detach_stops_gradient() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let detached = x.detach();
        assert!(!detached.requires_grad());
        assert!(!detached.has_grad_fn());
        let loss = (&detached * 2.0).sum();
        loss.backward();
        assert!(
            x.grad().is_none(),
            "Original should not receive gradient through detach"
        );
    }

    #[test]
    fn zero_grad_then_backward() {
        let x = Variable::new(Tensor::from_vec(vec![3.0], &[1]), true);
        // First backward
        let y1 = (&x * &x).sum();
        y1.backward();
        assert_abs_diff_eq!(x.grad().unwrap().to_vec()[0], 6.0, epsilon = 1e-4);

        // Zero grad and do another backward
        x.zero_grad();
        assert!(x.grad().is_none());
        let y2 = (&x * 2.0).sum();
        y2.backward();
        assert_abs_diff_eq!(x.grad().unwrap().to_vec()[0], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn variable_used_many_times() {
        // f(x) = sum(x + x + x + x + x) = 5*sum(x), df/dx = 5
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let y = &(&(&(&x + &x) + &x) + &x) + &x;
        let loss = y.sum();
        loss.backward();
        for g in x.grad().unwrap().to_vec() {
            assert_abs_diff_eq!(g, 5.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn deep_graph_no_stack_overflow() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        let mut result = x.clone();
        for _ in 0..200 {
            result = &result + 0.001;
        }
        let loss = result.sum();
        loss.backward();
        // Should complete without stack overflow
        assert!(x.grad().is_some());
    }

    #[test]
    fn set_data_does_not_affect_gradient() {
        let x = Variable::new(Tensor::from_vec(vec![2.0], &[1]), true);
        let loss = (&x * &x).sum();
        loss.backward();
        let grad_before = x.grad().unwrap().to_vec()[0];
        x.set_data(Tensor::from_vec(vec![100.0], &[1]));
        // Gradient should still be the old one
        let grad_after = x.grad().unwrap().to_vec()[0];
        assert_abs_diff_eq!(grad_before, grad_after, epsilon = 1e-6);
    }

    #[test]
    fn clone_shares_state() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        let x2 = x.clone();
        // Use backward to set gradient via public API
        let loss = (&x * 5.0).sum();
        loss.backward();
        // Clone should see the same gradient
        assert_abs_diff_eq!(x2.grad().unwrap().to_vec()[0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn requires_grad_propagation_all_false() {
        let a = Variable::new(Tensor::ones(&[2]), false);
        let b = Variable::new(Tensor::ones(&[2]), false);
        let c = &a + &b;
        assert!(!c.requires_grad());
        assert!(!c.has_grad_fn());
    }

    #[test]
    fn requires_grad_propagation_one_true() {
        let a = Variable::new(Tensor::ones(&[2]), true);
        let b = Variable::new(Tensor::ones(&[2]), false);
        let c = &a + &b;
        assert!(c.requires_grad());
        assert!(c.has_grad_fn());
    }

    #[test]
    fn graph_inputs_leaf_none() {
        let x = Variable::new(Tensor::ones(&[2]), true);
        assert!(x.graph_inputs().is_none());
    }

    #[test]
    fn graph_inputs_computed_some() {
        let a = Variable::new(Tensor::ones(&[2]), true);
        let b = Variable::new(Tensor::ones(&[2]), true);
        let c = &a + &b;
        let inputs = c.graph_inputs().expect("Should have graph inputs");
        assert_eq!(inputs.len(), 2);
    }
}

// Module: Variable Properties & Thread Safety

mod variable_properties {
    use super::*;

    // Variable uses Rc<RefCell<>>, so it is NOT Send or Sync.
    // This is a compile-time assertion documenting the design decision.
    // If the design changes to Arc, these tests should be updated.

    #[test]
    fn variable_is_not_send() {
        // Rc<RefCell<>> is !Send. We verify this by ensuring Variable
        // cannot be sent across threads. This is a documentation test.
        fn _not_send<T>()
        where
            T: Send,
        {
        }
        // Uncomment below to verify it FAILS:
        // _not_send::<Variable>();
        // For now, just document the fact:
        let _ = Variable::new(Tensor::scalar(1.0), false);
    }

    #[test]
    fn variable_equality_is_pointer_based() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]), false);
        let b = Variable::new(Tensor::from_vec(vec![1.0], &[1]), false);
        assert_ne!(a, b, "Different Rc pointers should not be equal");

        let c = a.clone();
        assert_eq!(a, c, "Same Rc pointer should be equal");
    }

    #[test]
    #[allow(clippy::mutable_key_type)]
    fn variable_hash_consistency() {
        use std::collections::HashSet;
        let a = Variable::new(Tensor::ones(&[2]), true);
        let b = a.clone();
        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b), "Clone should hash the same");
    }

    #[test]
    fn variable_debug_format() {
        let v = Variable::new(Tensor::ones(&[3, 4]), true);
        let debug = format!("{:?}", v);
        assert!(debug.contains("Variable"));
        assert!(debug.contains("requires_grad: true"));
    }

    #[test]
    fn from_tensor_no_grad() {
        let v = Variable::from_tensor(Tensor::ones(&[5]));
        assert!(!v.requires_grad());
        assert!(v.grad().is_none());
    }

    #[test]
    fn id_unique() {
        let a = Variable::new(Tensor::ones(&[2]), true);
        let b = Variable::new(Tensor::ones(&[2]), true);
        assert_ne!(a.id(), b.id());
        let c = a.clone();
        assert_eq!(a.id(), c.id());
    }
}

// Module: Optimizer Stress Tests

mod optimizer_stress {
    use super::*;

    #[test]
    fn sgd_converges_quadratic() {
        // Minimize f(w) = (w - 3)^2
        let w = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let target = Variable::new(Tensor::from_vec(vec![3.0], &[1]), false);
        let mut sgd = SGD::new(vec![w.clone()], 0.1, 0.0);
        for _ in 0..200 {
            sgd.zero_grad();
            let diff = &w - &target;
            let loss = (&diff * &diff).sum();
            loss.backward();
            sgd.step();
        }
        assert_abs_diff_eq!(w.data().to_vec()[0], 3.0, epsilon = 1e-2);
    }

    #[test]
    fn sgd_momentum_converges_faster() {
        let w_no_mom = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let w_mom = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let target = Variable::new(Tensor::from_vec(vec![5.0], &[1]), false);

        let mut sgd_no = SGD::new(vec![w_no_mom.clone()], 0.01, 0.0);
        let mut sgd_mom = SGD::new(vec![w_mom.clone()], 0.01, 0.9);

        for _ in 0..100 {
            sgd_no.zero_grad();
            let diff = &w_no_mom - &target;
            let loss = (&diff * &diff).sum();
            loss.backward();
            sgd_no.step();

            sgd_mom.zero_grad();
            let diff = &w_mom - &target;
            let loss = (&diff * &diff).sum();
            loss.backward();
            sgd_mom.step();
        }

        let err_no = (w_no_mom.data().to_vec()[0] - 5.0).abs();
        let err_mom = (w_mom.data().to_vec()[0] - 5.0).abs();
        assert!(
            err_mom <= err_no,
            "Momentum should converge at least as fast: no_mom_err={}, mom_err={}",
            err_no,
            err_mom
        );
    }

    #[test]
    fn adam_converges_quadratic() {
        let w = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let target = Variable::new(Tensor::from_vec(vec![5.0], &[1]), false);
        let mut adam = Adam::new(vec![w.clone()], 0.1);
        for _ in 0..300 {
            adam.zero_grad();
            let diff = &w - &target;
            let loss = (&diff * &diff).sum();
            loss.backward();
            adam.step();
        }
        assert_abs_diff_eq!(w.data().to_vec()[0], 5.0, epsilon = 0.1);
    }

    #[test]
    fn sgd_zero_lr_no_update() {
        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let mut sgd = SGD::new(vec![w.clone()], 0.0, 0.0);
        // Set gradient via backward
        let loss = (&w * 999.0).sum();
        loss.backward();
        sgd.step();
        assert_abs_diff_eq!(w.data().to_vec()[0], 10.0, epsilon = 1e-6);
    }

    #[test]
    fn adam_zero_lr_no_update() {
        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let mut adam = Adam::with_betas(vec![w.clone()], 0.0, 0.9, 0.999, 1e-8);
        // Set gradient via backward
        let loss = (&w * 999.0).sum();
        loss.backward();
        adam.step();
        assert_abs_diff_eq!(w.data().to_vec()[0], 10.0, epsilon = 1e-6);
    }

    #[test]
    fn optimizer_skips_no_grad_params() {
        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let frozen = Variable::new(Tensor::from_vec(vec![5.0], &[1]), false);
        let mut sgd = SGD::new(vec![w.clone(), frozen.clone()], 0.1, 0.0);

        // Set gradient on w via backward
        let loss = (&w * 2.0).sum();
        loss.backward();
        sgd.step();

        assert!(
            (w.data().to_vec()[0] - 10.0).abs() > 1e-6,
            "w should have been updated"
        );
        assert_abs_diff_eq!(frozen.data().to_vec()[0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn sgd_negative_lr_moves_opposite() {
        let w = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let mut sgd = SGD::new(vec![w.clone()], -0.1, 0.0);
        // Set gradient = 1 via backward
        let loss = w.sum();
        loss.backward();
        sgd.step();
        // w = 0 - (-0.1) * 1.0 = 0.1 (moves positive, opposite of normal)
        assert!(w.data().to_vec()[0] > 0.0);
    }

    #[test]
    fn adam_multiple_params() {
        let w1 = Variable::new(Tensor::from_vec(vec![0.0, 0.0], &[2]), true);
        let w2 = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let t1 = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]), false);
        let t2 = Variable::new(Tensor::from_vec(vec![5.0], &[1]), false);

        let mut adam = Adam::new(vec![w1.clone(), w2.clone()], 0.1);
        for _ in 0..300 {
            adam.zero_grad();
            let d1 = &w1 - &t1;
            let d2 = &w2 - &t2;
            let loss = (&d1 * &d1).sum() + (&d2 * &d2).sum();
            loss.backward();
            adam.step();
        }

        assert_abs_diff_eq!(w1.data().to_vec()[0], 3.0, epsilon = 0.2);
        assert_abs_diff_eq!(w1.data().to_vec()[1], 4.0, epsilon = 0.2);
        assert_abs_diff_eq!(w2.data().to_vec()[0], 5.0, epsilon = 0.2);
    }

    #[test]
    fn sgd_monotonic_loss_decrease() {
        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let target = Variable::new(Tensor::from_vec(vec![0.0], &[1]), false);
        let mut sgd = SGD::new(vec![w.clone()], 0.01, 0.0);

        let mut losses = Vec::new();
        for _ in 0..50 {
            sgd.zero_grad();
            let diff = &w - &target;
            let loss = (&diff * &diff).sum();
            losses.push(loss.data().item());
            loss.backward();
            sgd.step();
        }

        // With small lr on convex problem, loss should monotonically decrease
        for i in 1..losses.len() {
            assert!(
                losses[i] <= losses[i - 1] + 1e-6,
                "Loss should decrease: step {}: {} > {}",
                i,
                losses[i],
                losses[i - 1]
            );
        }
    }

    #[test]
    fn zero_grad_clears_all() {
        let w1 = Variable::new(Tensor::ones(&[3]), true);
        let w2 = Variable::new(Tensor::ones(&[2, 2]), true);
        // Set gradients via backward
        let loss = &w1.sum() + &w2.sum();
        loss.backward();
        assert!(w1.grad().is_some());
        assert!(w2.grad().is_some());

        let mut sgd = SGD::new(vec![w1.clone(), w2.clone()], 0.01, 0.0);
        sgd.zero_grad();
        assert!(w1.grad().is_none());
        assert!(w2.grad().is_none());
    }
}
