//! Behaviour of `no_grad` and the fused transposed matmul.

use rustforge_autograd::{is_grad_enabled, no_grad, Variable};
use rustforge_tensor::Tensor;

fn param() -> Variable {
    Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true)
}

#[test]
fn no_grad_builds_no_graph_but_computes_the_same_values() {
    let w = param();
    let x = Variable::new(Tensor::from_vec(vec![0.5, -1.0], &[1, 2]), false);

    let tracked = x.matmul(&w).relu().sum();
    let untracked = no_grad(|| x.matmul(&w).relu().sum());

    assert!(tracked.requires_grad() && tracked.has_grad_fn());
    assert!(!untracked.requires_grad() && !untracked.has_grad_fn());
    assert_eq!(tracked.data().to_vec(), untracked.data().to_vec());
}

#[test]
fn no_grad_restores_mode_after_return_nesting_and_panic() {
    assert!(is_grad_enabled());
    no_grad(|| {
        assert!(!is_grad_enabled());
        no_grad(|| assert!(!is_grad_enabled()));
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());

    let result = std::panic::catch_unwind(|| no_grad(|| panic!("inside no_grad")));
    assert!(result.is_err());
    assert!(is_grad_enabled());
}

#[test]
fn parameters_used_under_no_grad_receive_no_gradient() {
    let w = param();
    let out = no_grad(|| w.sum());
    assert!(!out.requires_grad());
    // A tracked use afterwards still works normally.
    w.sum().backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![1.0; 4]);
}

#[test]
fn matmul_t_matches_matmul_with_explicit_transpose_in_value_and_gradient() {
    let x_data = Tensor::from_vec(vec![0.5, -1.0, 2.0, 0.25, 1.5, -0.5], &[2, 3]);
    let w_data = Tensor::from_vec(
        vec![
            0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8, -0.9, 1.0, 1.1, 1.2,
        ],
        &[4, 3],
    );

    let x1 = Variable::new(x_data.clone(), true);
    let w1 = Variable::new(w_data.clone(), true);
    let y1 = x1.matmul(&w1.t());
    y1.pow(2.0).sum().backward();

    let x2 = Variable::new(x_data, true);
    let w2 = Variable::new(w_data, true);
    let y2 = x2.matmul_t(&w2);
    y2.pow(2.0).sum().backward();

    let close = |a: Vec<f32>, b: Vec<f32>| {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    };
    assert_eq!(y2.shape(), vec![2, 4]);
    close(y1.data().to_vec(), y2.data().to_vec());
    close(x1.grad().unwrap().to_vec(), x2.grad().unwrap().to_vec());
    close(w1.grad().unwrap().to_vec(), w2.grad().unwrap().to_vec());
}

#[test]
fn matmul_t_skips_gradient_for_inputs_that_do_not_need_it() {
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
    let w = param();
    x.matmul_t(&w).sum().backward();
    assert!(x.grad().is_none());
    // d/dW sum(x Wᵀ) = broadcast x across output rows.
    assert_eq!(w.grad().unwrap().to_vec(), vec![1.0, 2.0, 1.0, 2.0]);
}
