//! Activation function modules.
//!
//! Stateless activation layers that implement the `Module` trait.
//! These have no learnable parameters — they simply apply element-wise
//! nonlinearities to their inputs.
//!
//! ## Available Activations
//!
//! | Activation | Range | Typical Use |
//! |-----------|-------|-------------|
//! | `ReLU` | [0, ∞) | Hidden layers (most common) |
//! | `Sigmoid` | (0, 1) | Binary classification output |
//! | `Tanh` | (-1, 1) | Hidden layers, bounded output |
//! | `Softmax` | (0, 1), sums to 1 | Multi-class classification output |

use rustforge_autograd::Variable;

use crate::module::Module;

// ============================================================================
// ReLU: max(0, x)
// ============================================================================

/// Rectified Linear Unit activation function.
///
/// `ReLU(x) = max(0, x)`
///
/// The most widely used activation function in deep learning.
/// Simple, computationally efficient, and mitigates the vanishing gradient problem
/// for positive inputs. However, can suffer from "dying ReLU" where neurons
/// permanently output 0 for all inputs.
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Variable {
        input.relu()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

// ============================================================================
// Sigmoid: σ(x) = 1 / (1 + exp(-x))
// ============================================================================

/// Sigmoid activation function.
///
/// `σ(x) = 1 / (1 + exp(-x))`
///
/// Maps input to the (0, 1) range. Commonly used as the output activation
/// for binary classification tasks. Also used in LSTM gates.
///
/// Note: For multi-class classification, prefer Softmax.
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Variable {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

// ============================================================================
// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// ============================================================================

/// Hyperbolic tangent activation function.
///
/// `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
///
/// Maps input to the (-1, 1) range. Zero-centered output makes it preferred
/// over Sigmoid for hidden layers in some architectures (e.g., RNNs).
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Variable {
        input.tanh_()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

// ============================================================================
// Softmax: exp(x_i) / Σ exp(x_j)  (along axis 1)
// ============================================================================

/// Softmax activation function (applied along axis 1 — the class dimension).
///
/// `softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))`
///
/// Converts logits into a probability distribution (all outputs > 0, sum to 1).
/// Used as the output layer for multi-class classification.
///
/// ## Numerical Stability
/// Subtracts the maximum value (detached — no gradient) before exponentiation
/// to prevent overflow.
pub struct Softmax;

impl Module for Softmax {
    fn forward(&self, input: &Variable) -> Variable {
        // Detach max for numerical stability (max doesn't need gradients)
        let max_val = Variable::from_tensor(input.data().max_axis(1, true).unwrap());
        let shifted = input - &max_val;
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum_axis(1, true);
        &exp_vals / &sum_exp
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rustforge_tensor::Tensor;

    #[test]
    fn test_relu_forward() {
        let relu = ReLU;
        let x = Variable::new(
            Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5]),
            true,
        );
        let y = relu.forward(&x);
        assert_eq!(y.data().to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid;
        let x = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        let y = sigmoid.forward(&x);
        assert_abs_diff_eq!(y.data().to_vec()[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh;
        let x = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]), false);
        let y = tanh.forward(&x);
        assert_abs_diff_eq!(y.data().to_vec()[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let softmax = Softmax;
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            false,
        );
        let y = softmax.forward(&x);
        let data = y.data().to_vec();

        // Each row should sum to 1.0
        let row1_sum: f32 = data[0..3].iter().sum();
        let row2_sum: f32 = data[3..6].iter().sum();
        assert_abs_diff_eq!(row1_sum, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(row2_sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let softmax = Softmax;
        // Large values that would overflow without max subtraction
        let x = Variable::new(
            Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3]),
            false,
        );
        let y = softmax.forward(&x);
        let data = y.data().to_vec();

        // Should not be NaN or Inf
        for &v in &data {
            assert!(!v.is_nan(), "Softmax produced NaN");
            assert!(!v.is_infinite(), "Softmax produced Inf");
        }

        // Should still sum to 1
        let sum: f32 = data.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_activations_no_parameters() {
        assert!(ReLU.parameters().is_empty());
        assert!(Sigmoid.parameters().is_empty());
        assert!(Tanh.parameters().is_empty());
        assert!(Softmax.parameters().is_empty());
    }
}
