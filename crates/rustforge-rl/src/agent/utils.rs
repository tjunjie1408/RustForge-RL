//! Parameter copy utilities and differentiable math helpers for RL agents.
//!
//! Provides building blocks shared across off-policy algorithms (TD3, SAC, DQN)
//! and on-policy algorithms (PPO) that need:
//!
//! - **`hard_update`**: Full parameter copy from source to target network.
//! - **`soft_update`**: Polyak averaging for smooth target network tracking.
//! - **`elementwise_min_var`**: Differentiable element-wise minimum (clipped double-Q).
//! - **`clamp_var`**: Differentiable clamping for action bounds (TD3 noise clipping).
//!
//! ## Design
//!
//! All differentiable helpers are composed from existing autograd primitives
//! (`relu`, `+`, `-`) — no new `GradFn` implementations are required. Gradients
//! flow correctly through the standard computation graph.

use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

/// Copies each source parameter's data into the corresponding target parameter.
///
/// This performs a full (hard) copy: after calling `hard_update`, every target
/// parameter will hold an identical clone of the corresponding source parameter's
/// tensor data.
///
/// Typically used for periodic target network synchronization in DQN.
///
/// ## Panics
///
/// Panics if `source` and `target` have different lengths.
///
/// ## Example
/// ```rust,ignore
/// hard_update(&online_net.parameters(), &target_net.parameters());
/// ```
pub fn hard_update(source: &[Variable], target: &[Variable]) {
    assert_eq!(
        source.len(),
        target.len(),
        "hard_update: source ({}) and target ({}) parameter counts must match",
        source.len(),
        target.len()
    );
    for (src, tgt) in source.iter().zip(target.iter()) {
        tgt.set_data(src.data().clone());
    }
}

/// Polyak-averages source parameters into target parameters.
///
/// For each parameter pair, computes:
/// ```text
/// target = τ * source + (1 - τ) * target
/// ```
///
/// With `τ = 1.0` this is equivalent to `hard_update`. With small `τ` (e.g. 0.005)
/// the target network tracks the source slowly, which stabilises training in
/// TD3 and SAC.
///
/// ## Panics
///
/// - If `source` and `target` have different lengths.
/// - If `tau` is outside `[0.0, 1.0]`.
///
/// ## Example
/// ```rust,ignore
/// soft_update(&online_net.parameters(), &target_net.parameters(), 0.005);
/// ```
pub fn soft_update(source: &[Variable], target: &[Variable], tau: f32) {
    assert_eq!(
        source.len(),
        target.len(),
        "soft_update: source ({}) and target ({}) parameter counts must match",
        source.len(),
        target.len()
    );
    assert!(
        (0.0..=1.0).contains(&tau),
        "soft_update: tau must be in [0.0, 1.0], got {}",
        tau
    );
    for (src, tgt) in source.iter().zip(target.iter()) {
        let src_data = src.data();
        let tgt_data = tgt.data();
        // target = τ * source + (1 - τ) * target
        let interpolated = &(&*src_data * tau) + &(&*tgt_data * (1.0 - tau));
        drop(src_data);
        drop(tgt_data);
        tgt.set_data(interpolated);
    }
}

/// Differentiable element-wise minimum of two Variables.
///
/// Computes `min(x, y)` using the identity:
/// ```text
/// min(x, y) = y - relu(y - x)
/// ```
///
/// This preserves gradient flow through both `x` and `y` via the existing
/// `relu` and subtraction autograd ops — no custom `GradFn` is needed.
///
/// Used in clipped double-Q learning (TD3/SAC) to take the minimum of two
/// critic predictions.
///
/// ## Example
/// ```rust,ignore
/// let q_min = elementwise_min_var(&q1, &q2);
/// ```
pub fn elementwise_min_var(x: &Variable, y: &Variable) -> Variable {
    // min(x, y) = y - relu(y - x)
    y - &(y - x).relu()
}

/// Differentiable clamp that restricts Variable values to `[min_val, max_val]`.
///
/// Uses relu composition to achieve differentiable clamping:
/// ```text
/// max(x, min_val) = x + relu(min_val - x)
/// min(x, max_val) = max_val - relu(max_val - x)
/// ```
///
/// The gradient is `1.0` when the input is within the range, and `0.0` when
/// clipped — matching the standard clamp gradient behaviour.
///
/// Used in TD3 for clamping exploration noise and squashed actions.
///
/// ## Panics
///
/// Panics if `min_val > max_val`.
///
/// ## Example
/// ```rust,ignore
/// let clipped = clamp_var(&noisy_action, -1.0, 1.0);
/// ```
pub fn clamp_var(x: &Variable, min_val: f32, max_val: f32) -> Variable {
    assert!(
        min_val <= max_val,
        "clamp_var: min_val ({}) must be <= max_val ({})",
        min_val,
        max_val
    );

    let shape = x.shape();
    let numel: usize = shape.iter().product();

    // max(x, min_val): clamp from below
    let min_const = Variable::from_tensor(Tensor::from_vec(vec![min_val; numel], &shape));
    let low_clipped = x + &(&min_const - x).relu();

    // min(low_clipped, max_val): clamp from above
    let max_const = Variable::from_tensor(Tensor::from_vec(vec![max_val; numel], &shape));
    

    &max_const - &(&max_const - &low_clipped).relu()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── hard_update ──

    #[test]
    fn hard_update_copies_params() {
        let src = vec![
            Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), false),
            Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[2]), false),
        ];
        let tgt = vec![
            Variable::new(Tensor::zeros(&[3]), false),
            Variable::new(Tensor::zeros(&[2]), false),
        ];

        hard_update(&src, &tgt);

        assert_eq!(tgt[0].data().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(tgt[1].data().to_vec(), vec![10.0, 20.0]);
    }

    // ── soft_update ──

    #[test]
    fn soft_update_interpolates() {
        let src = vec![Variable::new(
            Tensor::from_vec(vec![10.0, 20.0], &[2]),
            false,
        )];
        let tgt_data = vec![0.0, 0.0];

        // tau = 0 → target unchanged
        {
            let tgt = vec![Variable::new(
                Tensor::from_vec(tgt_data.clone(), &[2]),
                false,
            )];
            soft_update(&src, &tgt, 0.0);
            let result = tgt[0].data().to_vec();
            assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-6);
        }

        // tau = 1 → target becomes source
        {
            let tgt = vec![Variable::new(
                Tensor::from_vec(tgt_data.clone(), &[2]),
                false,
            )];
            soft_update(&src, &tgt, 1.0);
            let result = tgt[0].data().to_vec();
            assert_abs_diff_eq!(result[0], 10.0, epsilon = 1e-6);
            assert_abs_diff_eq!(result[1], 20.0, epsilon = 1e-6);
        }

        // tau = 0.5 → average
        {
            let tgt = vec![Variable::new(
                Tensor::from_vec(tgt_data.clone(), &[2]),
                false,
            )];
            soft_update(&src, &tgt, 0.5);
            let result = tgt[0].data().to_vec();
            assert_abs_diff_eq!(result[0], 5.0, epsilon = 1e-6);
            assert_abs_diff_eq!(result[1], 10.0, epsilon = 1e-6);
        }
    }

    // ── elementwise_min_var ──

    #[test]
    fn elementwise_min_selects_smaller() {
        let x = Variable::from_tensor(Tensor::from_vec(vec![1.0, 5.0, 3.0, 7.0], &[4]));
        let y = Variable::from_tensor(Tensor::from_vec(vec![4.0, 2.0, 3.0, 8.0], &[4]));
        let result = elementwise_min_var(&x, &y);
        let data = result.data().to_vec();
        assert_abs_diff_eq!(data[0], 1.0, epsilon = 1e-6); // min(1, 4) = 1
        assert_abs_diff_eq!(data[1], 2.0, epsilon = 1e-6); // min(5, 2) = 2
        assert_abs_diff_eq!(data[2], 3.0, epsilon = 1e-6); // min(3, 3) = 3
        assert_abs_diff_eq!(data[3], 7.0, epsilon = 1e-6); // min(7, 8) = 7
    }

    #[test]
    fn elementwise_min_gradient_flow() {
        // x = [1, 5], y = [4, 2]
        // min = [1, 2] → sum = 3
        let x = Variable::new(Tensor::from_vec(vec![1.0, 5.0], &[2]), true);
        let y = Variable::new(Tensor::from_vec(vec![4.0, 2.0], &[2]), true);

        let result = elementwise_min_var(&x, &y);
        let loss = result.sum();
        loss.backward();

        // For element 0: min selects x (1 < 4) → dx=1, dy=0
        // For element 1: min selects y (2 < 5) → dx=0, dy=1
        let x_grad = x.grad().expect("x should have gradient");
        let y_grad = y.grad().expect("y should have gradient");

        assert_abs_diff_eq!(x_grad.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x_grad.to_vec()[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y_grad.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y_grad.to_vec()[1], 1.0, epsilon = 1e-6);
    }

    // ── clamp_var ──

    #[test]
    fn clamp_var_within_range() {
        // Values already within [-1, 1] should pass through unchanged
        let x = Variable::from_tensor(Tensor::from_vec(vec![-0.5, 0.0, 0.5], &[3]));
        let result = clamp_var(&x, -1.0, 1.0);
        let data = result.data().to_vec();
        assert_abs_diff_eq!(data[0], -0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(data[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[2], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn clamp_var_clips_extremes() {
        // Values outside [-1, 1] should be clipped
        let x = Variable::from_tensor(Tensor::from_vec(vec![-3.0, -1.0, 0.0, 1.0, 3.0], &[5]));
        let result = clamp_var(&x, -1.0, 1.0);
        let data = result.data().to_vec();
        assert_abs_diff_eq!(data[0], -1.0, epsilon = 1e-6); // clipped low
        assert_abs_diff_eq!(data[1], -1.0, epsilon = 1e-6); // at boundary
        assert_abs_diff_eq!(data[2], 0.0, epsilon = 1e-6); // pass through
        assert_abs_diff_eq!(data[3], 1.0, epsilon = 1e-6); // at boundary
        assert_abs_diff_eq!(data[4], 1.0, epsilon = 1e-6); // clipped high
    }

    #[test]
    fn clamp_var_gradient_flow() {
        // x = [-3, 0, 3], clamp to [-1, 1]
        // output = [-1, 0, 1]
        // grad should be: [0, 1, 0] (1 inside range, 0 outside)
        let x = Variable::new(Tensor::from_vec(vec![-3.0, 0.0, 3.0], &[3]), true);
        let result = clamp_var(&x, -1.0, 1.0);
        let loss = result.sum();
        loss.backward();

        let grad = x.grad().expect("x should have gradient");
        let g = grad.to_vec();
        assert_abs_diff_eq!(g[0], 0.0, epsilon = 1e-6); // clipped below → no grad
        assert_abs_diff_eq!(g[1], 1.0, epsilon = 1e-6); // inside range → grad = 1
        assert_abs_diff_eq!(g[2], 0.0, epsilon = 1e-6); // clipped above → no grad
    }

    #[test]
    fn clamp_var_boundary_nan_safety() {
        // Verify no NaN/Inf with values well outside the clamp range.
        // We use values that are large but not so extreme that f32 arithmetic
        // loses precision in the relu-based intermediates.
        let x = Variable::from_tensor(Tensor::from_vec(
            vec![-1000.0, 1000.0, -100.0, 100.0, 0.0],
            &[5],
        ));
        let result = clamp_var(&x, -1.0, 1.0);
        let data = result.data().to_vec();

        for (i, val) in data.iter().enumerate() {
            assert!(val.is_finite(), "element {} is not finite: {}", i, val);
        }

        assert_abs_diff_eq!(data[0], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[2], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[3], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[4], 0.0, epsilon = 1e-6);
    }
}
