//! Layer normalization.
//!
//! Normalizes the input across the last dimension for each sample independently,
//! then applies a learnable affine transformation.
//!
//! ## Formula
//! ```text
//! y = (x - mean) / sqrt(var + eps) * gamma + beta
//! ```
//!
//! where `mean` and `var` are computed per-sample across the normalized dimension,
//! and `gamma` (scale) and `beta` (shift) are learnable parameters.
//!
//! ## Why LayerNorm?
//!
//! Unlike BatchNorm, LayerNorm normalizes across features (not batch),
//! making it suitable for:
//! - Small batch sizes or batch size 1
//! - Recurrent neural networks (RNNs)
//! - Transformer architectures
//!
//! ## Reference
//! Ba, Kiros & Hinton, "Layer Normalization", 2016.

use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

use crate::module::Module;

/// Layer normalization module.
///
/// ## Example
/// ```rust,ignore
/// let ln = LayerNorm::new(256);  // normalize across last dim of size 256
/// let x = Variable::new(Tensor::randn(&[32, 256], None), false);
/// let y = ln.forward(&x);
/// // y has mean ≈ 0 and std ≈ 1 across the last dimension
/// ```
pub struct LayerNorm {
    /// Normalized shape (size of last dimension).
    normalized_shape: usize,
    /// Learnable scale parameter (gamma), initialized to 1.
    gamma: Variable,
    /// Learnable shift parameter (beta), initialized to 0.
    beta: Variable,
    /// Small constant for numerical stability.
    eps: f32,
}

impl LayerNorm {
    /// Creates a new LayerNorm module.
    ///
    /// ## Arguments
    /// - `normalized_shape`: Size of the last dimension to normalize across.
    pub fn new(normalized_shape: usize) -> Self {
        Self::with_eps(normalized_shape, 1e-5)
    }

    /// Creates a new LayerNorm module with a custom epsilon value.
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        let gamma = Variable::new(Tensor::ones(&[normalized_shape]), true);
        let beta = Variable::new(Tensor::zeros(&[normalized_shape]), true);
        LayerNorm {
            normalized_shape,
            gamma,
            beta,
            eps,
        }
    }
}

impl Module for LayerNorm {
    /// Forward pass: normalizes across the last dimension.
    ///
    /// Input shape: `[batch, ..., normalized_shape]`
    /// Output shape: same as input
    fn forward(&self, input: &Variable) -> Variable {
        let ndim = input.shape().len();
        let last_axis = ndim - 1;

        // mean(x, axis=-1, keepdim=true)
        let mean = input.sum_axis(last_axis, true) / self.normalized_shape as f32;

        // var(x, axis=-1, keepdim=true) = mean((x - mean)², axis=-1)
        let centered = input - &mean;
        let var = centered.pow(2.0).sum_axis(last_axis, true) / self.normalized_shape as f32;

        // normalize: (x - mean) / sqrt(var + eps)
        let std_inv = (var + self.eps).pow(-0.5);
        let normalized = &centered * &std_inv;

        // affine: gamma * normalized + beta
        &(&normalized * &self.gamma) + &self.beta
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_layernorm_output_shape() {
        let ln = LayerNorm::new(4);
        let x = Variable::new(Tensor::from_vec(vec![1.0; 12], &[3, 4]), false);
        let y = ln.forward(&x);
        assert_eq!(y.shape(), vec![3, 4]);
    }

    #[test]
    fn test_layernorm_normalizes() {
        let ln = LayerNorm::new(4);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
            false,
        );
        let y = ln.forward(&x);
        let data = y.data().to_vec();

        // Each row should have mean ≈ 0 (since gamma=1, beta=0)
        let row1_mean = (data[0] + data[1] + data[2] + data[3]) / 4.0;
        let row2_mean = (data[4] + data[5] + data[6] + data[7]) / 4.0;
        assert_abs_diff_eq!(row1_mean, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(row2_mean, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_layernorm_parameters() {
        let ln = LayerNorm::new(8);
        let params = ln.parameters();
        assert_eq!(params.len(), 2); // gamma + beta
        assert_eq!(params[0].shape(), vec![8]); // gamma
        assert_eq!(params[1].shape(), vec![8]); // beta
    }

    #[test]
    fn test_layernorm_gradient_flow() {
        let ln = LayerNorm::new(3);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), true);
        let y = ln.forward(&x);
        y.sum().backward();

        assert!(x.grad().is_some(), "Input should have gradient");
        let params = ln.parameters();
        assert!(params[0].grad().is_some(), "Gamma should have gradient");
        assert!(params[1].grad().is_some(), "Beta should have gradient");
    }
}
