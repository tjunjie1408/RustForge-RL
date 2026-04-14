//! Loss functions for training neural networks.
//!
//! Loss functions measure the discrepancy between the model's predictions and
//! the ground truth targets. They return a scalar `Variable` suitable for
//! calling `backward()` to compute gradients.
//!
//! ## Available Loss Functions
//!
//! | Function | Formula | Use Case |
//! |----------|---------|----------|
//! | `mse_loss` | `mean((pred - target)²)` | Regression |
//! | `cross_entropy_loss` | `-mean(Σ t·log_softmax(x))` | Classification |
//! | `huber_loss` | Smooth L1 variant | Robust regression (DQN) |

use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

/// Mean Squared Error loss.
///
/// ## Formula
/// `L = (1/N) Σ (pred_i - target_i)²`
///
/// ## Arguments
/// - `pred`: Model predictions.
/// - `target`: Ground truth values (same shape as `pred`).
///
/// ## Returns
/// Scalar variable (the mean squared error).
///
/// ## Example
/// ```rust,ignore
/// let loss = mse_loss(&predictions, &targets);
/// loss.backward();
/// ```
pub fn mse_loss(pred: &Variable, target: &Variable) -> Variable {
    let diff = pred - target;
    let sq = diff.pow(2.0);
    sq.mean()
}

/// Cross-entropy loss for classification tasks.
///
/// Computes the negative log-likelihood of the correct class predictions
/// using numerically stable log-softmax.
///
/// ## Formula
/// ```text
/// L = -(1/N) Σ_i Σ_c  t_{i,c} · log(softmax(x_{i,c}))
/// ```
///
/// ## Arguments
/// - `logits`: Raw (unnormalized) model outputs, shape `[batch, num_classes]`.
/// - `targets`: One-hot encoded target labels, shape `[batch, num_classes]`.
///
/// ## Numerical Stability
/// Uses the log-sum-exp trick: subtracts the per-row maximum before
/// exponentiation to prevent overflow.
///
/// ## Returns
/// Scalar variable (the mean cross-entropy loss).
///
/// ## Example
/// ```rust,ignore
/// // Binary classification
/// let logits = model.forward(&x);  // [batch, 2]
/// let targets = one_hot_encode(&labels, 2);  // [batch, 2]
/// let loss = cross_entropy_loss(&logits, &targets);
/// loss.backward();
/// ```
pub fn cross_entropy_loss(logits: &Variable, targets: &Variable) -> Variable {
    // Numerical stability: shift by max per row (detached — no gradient needed)
    let max_val = Variable::from_tensor(logits.data().max_axis(1, true).unwrap());
    let shifted = logits - &max_val;

    // log_softmax = shifted - log(sum(exp(shifted)))
    let exp_vals = shifted.exp();
    let sum_exp = exp_vals.sum_axis(1, true);
    let log_sum_exp = sum_exp.log();
    let log_probs = &shifted - &log_sum_exp;

    // CE = -mean(sum(targets * log_probs, axis=1))
    let elementwise = targets * &log_probs;
    let per_sample = elementwise.sum_axis(1, false);
    -(per_sample.mean())
}

/// Huber loss (Smooth L1 loss).
///
/// A robust loss function that is less sensitive to outliers than MSE.
/// Behaves like MSE for small errors and like MAE for large errors.
///
/// ## Formula
/// ```text
/// L_δ(a) = 0.5 · a²           if |a| ≤ δ
///        = δ · (|a| - 0.5·δ)   otherwise
/// where a = pred - target
/// ```
///
/// ## Arguments
/// - `pred`: Model predictions.
/// - `target`: Ground truth values.
/// - `delta`: Threshold where the loss transitions from quadratic to linear.
///   Default is typically 1.0.
///
/// ## Returns
/// Scalar variable (the mean Huber loss).
///
/// ## Use in RL
/// Commonly used in DQN to stabilize training by reducing the impact of
/// large TD-error outliers.
pub fn huber_loss(pred: &Variable, target: &Variable, delta: f32) -> Variable {
    let diff = pred - target;
    let abs_diff = diff.pow(2.0).sqrt(); // |pred - target| via sqrt(x²)
    let abs_diff_data = abs_diff.data();

    // Build condition mask: 1.0 where |diff| <= delta, 0.0 otherwise
    let mask_data = Tensor::from_ndarray(
        abs_diff_data
            .data()
            .mapv(|x| if x <= delta { 1.0 } else { 0.0 }),
    );
    let mask = Variable::from_tensor(mask_data.clone());
    let inv_mask = Variable::from_tensor(Tensor::from_ndarray(
        mask_data.data().mapv(|x| 1.0 - x),
    ));

    // Quadratic part: 0.5 * diff²
    let quadratic = diff.pow(2.0) * 0.5;

    // Linear part: delta * (|diff| - 0.5 * delta)
    let linear = (&abs_diff - 0.5 * delta) * delta;

    // Combine: mask * quadratic + inv_mask * linear
    let combined = &(&quadratic * &mask) + &(&linear * &inv_mask);
    combined.mean()
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mse_loss_zero() {
        // pred == target → loss == 0
        let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let loss = mse_loss(&pred, &target);
        assert_abs_diff_eq!(loss.data().item(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_known_value() {
        // pred = [1, 2, 3], target = [0, 0, 0]
        // MSE = mean(1² + 2² + 3²) = mean(1 + 4 + 9) = 14/3 ≈ 4.667
        let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), true);
        let target = Variable::new(Tensor::from_vec(vec![0.0, 0.0, 0.0], &[1, 3]), false);
        let loss = mse_loss(&pred, &target);
        assert_abs_diff_eq!(loss.data().item(), 14.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mse_loss_gradient() {
        let pred = Variable::new(Tensor::from_vec(vec![2.0, 4.0], &[1, 2]), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 3.0], &[1, 2]), false);
        let loss = mse_loss(&pred, &target);
        loss.backward();

        // d(MSE)/d(pred) = 2 * (pred - target) / N = 2 * [1, 1] / 2 = [1, 1]
        let grad = pred.grad().unwrap().to_vec();
        assert_abs_diff_eq!(grad[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(grad[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        // logits strongly favor the correct class → loss should be very small
        let logits = Variable::new(
            Tensor::from_vec(vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0], &[2, 3]),
            true,
        );
        let targets = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]),
            false,
        );
        let loss = cross_entropy_loss(&logits, &targets);
        // With strong logits, loss should be near 0
        assert!(
            loss.data().item() < 0.001,
            "Loss {} should be near 0 for perfect predictions",
            loss.data().item()
        );
    }

    #[test]
    fn test_cross_entropy_gradient_flow() {
        let logits = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]),
            true,
        );
        let targets = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]),
            false,
        );
        let loss = cross_entropy_loss(&logits, &targets);
        loss.backward();
        assert!(logits.grad().is_some(), "Logits should have gradients");
    }

    #[test]
    fn test_cross_entropy_numerical_stability() {
        // Large logits that would cause overflow without max subtraction
        let logits = Variable::new(
            Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3]),
            true,
        );
        let targets = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 1.0], &[1, 3]),
            false,
        );
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(
            !loss.data().item().is_nan(),
            "Loss should not be NaN with large logits"
        );
        assert!(
            !loss.data().item().is_infinite(),
            "Loss should not be Inf with large logits"
        );
    }

    #[test]
    fn test_huber_loss_small_error() {
        // |pred - target| = 0.5 < delta=1.0 → quadratic region: 0.5 * 0.25 = 0.125
        let pred = Variable::new(Tensor::from_vec(vec![1.5], &[1, 1]), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
        let loss = huber_loss(&pred, &target, 1.0);
        assert_abs_diff_eq!(loss.data().item(), 0.125, epsilon = 1e-4);
    }

    #[test]
    fn test_huber_loss_large_error() {
        // |pred - target| = 5.0 > delta=1.0 → linear region: 1.0 * (5.0 - 0.5) = 4.5
        let pred = Variable::new(Tensor::from_vec(vec![6.0], &[1, 1]), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]), false);
        let loss = huber_loss(&pred, &target, 1.0);
        assert_abs_diff_eq!(loss.data().item(), 4.5, epsilon = 1e-4);
    }
}
