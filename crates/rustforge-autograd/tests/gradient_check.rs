//! Integration tests for autograd — numerical gradient checking.
//!
//! These tests verify the correctness of the automatic differentiation engine
//! by comparing analytic gradients (from backward()) with numerical gradients
//! computed via the finite difference method:
//!
//!   ∂f/∂x ≈ [f(x + ε) - f(x - ε)] / (2ε)
//!
//! If the analytic and numerical gradients match (within tolerance), the
//! backward implementations are correct.
//!
//! Note: We use relative tolerance or wider absolute tolerance for numerical
//! checks because f32 finite differences with ε=1e-4 introduce O(ε²) truncation
//! error and O(ε⁻¹) rounding error.

use approx::assert_abs_diff_eq;
use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

/// Numerical gradient checking helper.
///
/// For each element of `param`, perturbs it by ±ε and computes the finite
/// difference approximation of the gradient.
fn numerical_gradient<F>(f: &F, param: &Variable, epsilon: f32) -> Vec<f32>
where
    F: Fn() -> Variable,
{
    let data = param.data();
    let n = data.numel();
    let mut num_grads = vec![0.0f32; n];

    for i in 0..n {
        let original = data.to_vec();

        // f(x + ε)
        let mut plus = original.clone();
        plus[i] += epsilon;
        param.set_data(Tensor::from_vec(plus, data.shape()));
        let loss_plus = f().data().item();

        // f(x - ε)
        let mut minus = original.clone();
        minus[i] -= epsilon;
        param.set_data(Tensor::from_vec(minus, data.shape()));
        let loss_minus = f().data().item();

        // Finite difference
        num_grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon);

        // Restore original data
        param.set_data(Tensor::from_vec(original, data.shape()));
    }

    num_grads
}

/// Asserts that analytic and numerical gradients match within relative tolerance.
/// Uses max(atol, rtol * max(|a|, |n|)) as the effective tolerance.
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

// Basic gradient tests (exact, hand-computed)

#[test]
fn test_grad_x_squared() {
    // f(x) = x², df/dx = 2x
    // x = [3.0], expected grad = [6.0]
    let x = Variable::new(Tensor::from_vec(vec![3.0], &[1]), true);
    let y = &x * &x;
    y.backward();

    let grad = x.grad().unwrap();
    assert_abs_diff_eq!(grad.to_vec()[0], 6.0, epsilon = 1e-5);
}

#[test]
fn test_grad_x_squared_numerical() {
    let x = Variable::new(Tensor::from_vec(vec![3.0], &[1]), true);

    let f = || {
        let y = &x * &x;
        y.sum()
    };

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();

    x.zero_grad();
    let numerical = numerical_gradient(&f, &x, 1e-4);

    assert_grads_close(&analytic, &numerical, 1e-2, 1e-3);
}

#[test]
fn test_grad_linear() {
    // f(x) = sum(x * w), df/dw = x, df/dx = w
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
    let w = Variable::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]), true);
    let y = (&x * &w).sum();
    y.backward();

    let dx = x.grad().unwrap().to_vec();
    let dw = w.grad().unwrap().to_vec();
    assert_eq!(dx, vec![4.0, 5.0, 6.0]);
    assert_eq!(dw, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_grad_matmul() {
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let w = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]), true);

    let y = x.matmul(&w).sum();
    y.backward();

    // Verify dx: for y = sum(X @ W), dy/dX = ones @ W.T
    // W.T = [[0.1, 0.3], [0.2, 0.4]]
    // ones = [[1,1],[1,1]]
    // dx = [[0.1+0.3, 0.2+0.4], [0.1+0.3, 0.2+0.4]] = [[0.4, 0.6], [0.4, 0.6]]
    // Wait, actually dy/dX_ij = sum_k W_jk = row_sum of W
    // But matmul grad: grad @ W.T, grad = ones([2,2])
    // [[1,1],[1,1]] @ [[0.1,0.3],[0.2,0.4]] = [[0.3,0.7],[0.3,0.7]]
    let dx = x.grad().unwrap().to_vec();
    assert_abs_diff_eq!(dx[0], 0.3, epsilon = 1e-5);
    assert_abs_diff_eq!(dx[1], 0.7, epsilon = 1e-5);

    // Numerical check for w
    let f = || {
        let xc = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        xc.matmul(&w).sum()
    };

    w.zero_grad();
    let numerical = numerical_gradient(&f, &w, 1e-4);
    let loss = f();
    loss.backward();
    let analytic = w.grad().unwrap().to_vec();

    assert_grads_close(&analytic, &numerical, 0.05, 1e-2);
}

#[test]
fn test_grad_relu() {
    // f(x) = sum(relu(x)), df/dx = (x > 0)
    let x = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]), true);
    let y = x.relu().sum();
    y.backward();

    let grad = x.grad().unwrap().to_vec();
    assert_eq!(grad, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_grad_sigmoid() {
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
fn test_grad_tanh() {
    let x = Variable::new(Tensor::from_vec(vec![0.5, -0.3, 1.0, -1.0], &[4]), true);

    let f = || x.tanh_().sum();

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();

    x.zero_grad();
    let numerical = numerical_gradient(&f, &x, 1e-4);

    assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
}

#[test]
fn test_grad_exp() {
    let x = Variable::new(Tensor::from_vec(vec![0.5, 1.0, -0.5], &[3]), true);

    let f = || x.exp().sum();

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();

    x.zero_grad();
    let numerical = numerical_gradient(&f, &x, 1e-4);

    assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
}

#[test]
fn test_grad_log() {
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);

    let f = || x.log().sum();

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();

    // Hand-computed: d(ln(x))/dx = 1/x = [1.0, 0.5, 0.333...]
    assert_abs_diff_eq!(analytic[0], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(analytic[1], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(analytic[2], 1.0 / 3.0, epsilon = 1e-5);

    x.zero_grad();
    let numerical = numerical_gradient(&f, &x, 1e-4);
    assert_grads_close(&analytic, &numerical, 0.02, 1e-2);
}

#[test]
fn test_grad_pow() {
    // f(x) = sum(x^3), df/dx = 3*x^2
    let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), true);

    let f = || x.pow(3.0).sum();

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();

    // Hand-computed: df/dx = 3*x^2 = [12.0, 27.0]
    assert_abs_diff_eq!(analytic[0], 12.0, epsilon = 1e-4);
    assert_abs_diff_eq!(analytic[1], 27.0, epsilon = 1e-4);

    x.zero_grad();
    let numerical = numerical_gradient(&f, &x, 1e-4);
    assert_grads_close(&analytic, &numerical, 0.1, 1e-2);
}

#[test]
fn test_grad_sqrt() {
    let x = Variable::new(Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3]), true);

    let f = || x.sqrt().sum();

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();

    // Hand-computed: d(√x)/dx = 1/(2√x) = [0.25, 1/6, 0.125]
    assert_abs_diff_eq!(analytic[0], 0.25, epsilon = 1e-4);
    assert_abs_diff_eq!(analytic[1], 1.0 / 6.0, epsilon = 1e-4);
    assert_abs_diff_eq!(analytic[2], 0.125, epsilon = 1e-4);
}

#[test]
fn test_grad_mean() {
    // f(x) = mean(x), df/dx_i = 1/n
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]), true);
    let y = x.mean();
    y.backward();

    let grad = x.grad().unwrap().to_vec();
    for g in &grad {
        assert_abs_diff_eq!(*g, 0.25, epsilon = 1e-6);
    }
}

#[test]
fn test_grad_neg() {
    // f(x) = sum(-x), df/dx = -1
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
    let y = (-&x).sum();
    y.backward();

    let grad = x.grad().unwrap().to_vec();
    assert_eq!(grad, vec![-1.0, -1.0, -1.0]);
}

#[test]
fn test_grad_div() {
    let x = Variable::new(Tensor::from_vec(vec![6.0, 8.0], &[2]), true);
    let y = Variable::new(Tensor::from_vec(vec![2.0, 4.0], &[2]), true);

    // Analytic: dx = 1/y = [0.5, 0.25], dy = -x/y² = [-1.5, -0.5]
    let loss = (&x / &y).sum();
    loss.backward();

    let dx = x.grad().unwrap().to_vec();
    assert_abs_diff_eq!(dx[0], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(dx[1], 0.25, epsilon = 1e-5);

    let dy = y.grad().unwrap().to_vec();
    assert_abs_diff_eq!(dy[0], -1.5, epsilon = 1e-5);
    assert_abs_diff_eq!(dy[1], -0.5, epsilon = 1e-5);

    // Numerical check
    x.zero_grad();
    y.zero_grad();
    let f_x = || (&x / &y).sum();
    let numerical_dx = numerical_gradient(&f_x, &x, 1e-4);
    assert_grads_close(&dx, &numerical_dx, 0.02, 1e-2);

    let f_y = || (&x / &y).sum();
    let numerical_dy = numerical_gradient(&f_y, &y, 1e-4);
    assert_grads_close(&dy, &numerical_dy, 0.02, 1e-2);
}

// Chain tests (multiple operations composed)

#[test]
fn test_grad_chain_matmul_relu_sum() {
    // f(w) = sum(relu(x @ w))
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let w = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]), true);

    let f = || x.matmul(&w).relu().sum();

    let loss = f();
    loss.backward();
    let analytic_dw = w.grad().unwrap().to_vec();

    w.zero_grad();
    x.zero_grad();
    let numerical_dw = numerical_gradient(&f, &w, 1e-4);

    assert_grads_close(&analytic_dw, &numerical_dw, 0.1, 1e-2);
}

#[test]
fn test_grad_chain_sigmoid_mse() {
    // f(w) = sum((sigmoid(x * w) - target)²)
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
    let w = Variable::new(Tensor::from_vec(vec![0.5, -0.3, 0.1], &[3]), true);
    let target = Variable::new(Tensor::from_vec(vec![0.8, 0.2, 0.6], &[3]), false);

    let f = || {
        let pred = (&x * &w).sigmoid();
        let diff = &pred - &target;
        (&diff * &diff).sum()
    };

    let loss = f();
    loss.backward();
    let analytic_dw = w.grad().unwrap().to_vec();

    w.zero_grad();
    x.zero_grad();
    let numerical_dw = numerical_gradient(&f, &w, 1e-4);

    assert_grads_close(&analytic_dw, &numerical_dw, 0.02, 1e-2);
}

#[test]
fn test_grad_variable_reuse() {
    // f(x) = sum(x*x + x) → df/dx = 2x + 1
    let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), true);
    let y = (&(&x * &x) + &x).sum();
    y.backward();

    let grad = x.grad().unwrap().to_vec();
    assert_abs_diff_eq!(grad[0], 5.0, epsilon = 1e-4);
    assert_abs_diff_eq!(grad[1], 7.0, epsilon = 1e-4);
}

#[test]
fn test_grad_broadcast_add() {
    // f(x, b) = sum(x + b) where x:[2,3], b:[3]
    let x = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        true,
    );
    let b = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]), true);

    let y = (&x + &b).sum();
    y.backward();

    // df/db: summed along broadcast dim → each = 2.0
    let grad_b = b.grad().unwrap().to_vec();
    assert_abs_diff_eq!(grad_b[0], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_b[1], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_b[2], 2.0, epsilon = 1e-6);

    // df/dx = 1.0 for all
    let grad_x = x.grad().unwrap().to_vec();
    for g in &grad_x {
        assert_abs_diff_eq!(*g, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_grad_broadcast_mul() {
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let s = Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[1, 2]), true);

    let f = || (&x * &s).sum();

    let loss = f();
    loss.backward();
    let analytic_ds = s.grad().unwrap().to_vec();

    // Hand-computed: ds_j = sum_i x_ij = [1+3, 2+4] = [4, 6]
    assert_abs_diff_eq!(analytic_ds[0], 4.0, epsilon = 1e-5);
    assert_abs_diff_eq!(analytic_ds[1], 6.0, epsilon = 1e-5);

    s.zero_grad();
    x.zero_grad();
    let numerical_ds = numerical_gradient(&f, &s, 1e-4);
    assert_grads_close(&analytic_ds, &numerical_ds, 0.1, 1e-2);
}

#[test]
fn test_grad_sum_axis() {
    let x = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        true,
    );

    // sum_axis(0) then sum: same as sum_all → grad = 1.0 everywhere
    let f = || x.sum_axis(0, false).sum();

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();
    for g in &analytic {
        assert_abs_diff_eq!(*g, 1.0, epsilon = 1e-5);
    }

    x.zero_grad();
    let numerical = numerical_gradient(&f, &x, 1e-4);
    assert_grads_close(&analytic, &numerical, 0.05, 1e-2);
}

#[test]
fn test_grad_scalar_ops() {
    // f(x) = sum((x*3 + 1)²)
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);

    let f = || (&x * 3.0 + 1.0).pow(2.0).sum();

    let loss = f();
    loss.backward();
    let analytic = x.grad().unwrap().to_vec();

    // d/dx[(3x+1)²] = 6(3x+1), x=1→24, x=2→42
    assert_abs_diff_eq!(analytic[0], 24.0, epsilon = 1e-3);
    assert_abs_diff_eq!(analytic[1], 42.0, epsilon = 1e-3);

    x.zero_grad();
    let numerical = numerical_gradient(&f, &x, 1e-4);
    assert_grads_close(&analytic, &numerical, 0.2, 1e-2);
}

// Milestone test — the exact code from AGENT.md

#[test]
fn test_milestone_matmul_relu_sum_backward() {
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let w = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]), true);

    let y = x.matmul(&w).relu().sum();
    y.backward();

    let dx = x.grad().expect("x should have gradient");
    let _dw = w.grad().expect("w should have gradient");
    assert_eq!(dx.shape(), &[2, 2]);
    assert_eq!(_dw.shape(), &[2, 2]);

    // Numerical verification for w
    let f = || {
        let xc = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        xc.matmul(&w).relu().sum()
    };

    w.zero_grad();
    let numerical_dw = numerical_gradient(&f, &w, 1e-4);
    let loss = f();
    loss.backward();
    let analytic_dw = w.grad().unwrap().to_vec();

    assert_grads_close(&analytic_dw, &numerical_dw, 0.1, 1e-2);

    println!("✅ Milestone verified: x.matmul(&w).relu().sum().backward() works!");
    println!("   dw = {:?}", analytic_dw);
}

// Optimizer integration tests

#[test]
fn test_sgd_training_loop() {
    use rustforge_autograd::optimizer::sgd::SGD;
    use rustforge_autograd::Optimizer;

    // Minimize f(w) = (w - 3)²
    let w = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
    let target = Variable::new(Tensor::from_vec(vec![3.0], &[1]), false);
    let mut sgd = SGD::new(vec![w.clone()], 0.1, 0.0);

    for _ in 0..100 {
        sgd.zero_grad();
        let diff = &w - &target;
        let loss = (&diff * &diff).sum();
        loss.backward();
        sgd.step();
    }

    assert_abs_diff_eq!(w.data().to_vec()[0], 3.0, epsilon = 1e-3);
}

#[test]
fn test_adam_training_loop() {
    use rustforge_autograd::optimizer::adam::Adam;
    use rustforge_autograd::Optimizer;

    // Minimize f(w) = (w - 5)²
    let w = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
    let target = Variable::new(Tensor::from_vec(vec![5.0], &[1]), false);
    let mut adam = Adam::new(vec![w.clone()], 0.1);

    for _ in 0..200 {
        adam.zero_grad();
        let diff = &w - &target;
        let loss = (&diff * &diff).sum();
        loss.backward();
        adam.step();
    }

    assert_abs_diff_eq!(w.data().to_vec()[0], 5.0, epsilon = 0.1);
}
