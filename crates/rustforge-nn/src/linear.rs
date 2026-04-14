//! Fully-connected (dense) linear layer.
//!
//! Implements the affine transformation:
//!
//! ```text
//! y = x @ W^T + b
//! ```
//!
//! where:
//! - `x`: input tensor of shape `[batch, in_features]`
//! - `W`: weight matrix of shape `[out_features, in_features]`
//! - `b`: bias vector of shape `[out_features]`
//! - `y`: output tensor of shape `[batch, out_features]`
//!
//! ## Weight Initialization
//!
//! Follows PyTorch's convention:
//! - Weight: Kaiming uniform initialization (optimal for ReLU activations)
//! - Bias: zeros
//!
//! ## Example
//! ```rust,ignore
//! use rustforge_nn::{Module, Linear};
//!
//! let layer = Linear::new(784, 256);
//! let x = Variable::new(Tensor::randn(&[32, 784], None), false);
//! let y = layer.forward(&x);
//! assert_eq!(y.shape(), vec![32, 256]);
//! ```

use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

use crate::module::Module;

/// Fully-connected linear layer: y = x @ W^T + b
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    weight: Variable,
    /// Bias vector [out_features], None if bias is disabled
    bias: Option<Variable>,
}

impl Linear {
    /// Creates a new linear layer with Kaiming uniform weight initialization.
    ///
    /// ## Arguments
    /// - `in_features`: Size of each input sample.
    /// - `out_features`: Size of each output sample.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // W ~ Kaiming uniform, shape [out_features, in_features]
        let weight_data = Tensor::kaiming_uniform(&[out_features, in_features], None);
        let weight = Variable::new(weight_data, true);

        // b = zeros, shape [out_features]
        let bias_data = Tensor::zeros(&[out_features]);
        let bias = Variable::new(bias_data, true);

        Linear {
            weight,
            bias: Some(bias),
        }
    }

    /// Creates a linear layer without bias.
    ///
    /// Useful when the subsequent layer (e.g., BatchNorm) already has a bias term.
    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        let weight_data = Tensor::kaiming_uniform(&[out_features, in_features], None);
        let weight = Variable::new(weight_data, true);

        Linear { weight, bias: None }
    }

    /// Returns the input dimension size.
    pub fn in_features(&self) -> usize {
        self.weight.shape()[1]
    }

    /// Returns the output dimension size.
    pub fn out_features(&self) -> usize {
        self.weight.shape()[0]
    }
}

impl Module for Linear {
    /// Forward pass: y = x @ W^T + b
    ///
    /// ## Input
    /// - `input`: tensor of shape `[batch, in_features]`
    ///
    /// ## Output
    /// - tensor of shape `[batch, out_features]`
    fn forward(&self, input: &Variable) -> Variable {
        // x [batch, in] @ W^T [in, out] → [batch, out]
        let out = input.matmul(&self.weight.t());
        match &self.bias {
            Some(b) => &out + b,
            None => out,
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linear_shapes() {
        let layer = Linear::new(4, 3);
        assert_eq!(layer.in_features(), 4);
        assert_eq!(layer.out_features(), 3);

        // Forward with batch of 2
        let x = Variable::new(Tensor::ones(&[2, 4]), false);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), vec![2, 3]);
    }

    #[test]
    fn test_linear_no_bias() {
        let layer = Linear::no_bias(3, 2);
        assert_eq!(layer.parameters().len(), 1); // weight only

        let x = Variable::new(Tensor::ones(&[1, 3]), false);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_linear_parameters_count() {
        let layer = Linear::new(10, 5);
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].shape(), vec![5, 10]); // weight
        assert_eq!(params[1].shape(), vec![5]); // bias
    }

    #[test]
    fn test_linear_gradient_flow() {
        let layer = Linear::new(2, 1);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
        let y = layer.forward(&x);
        let loss = y.sum();
        loss.backward();

        // Weight and bias should have gradients
        let params = layer.parameters();
        assert!(params[0].grad().is_some(), "Weight should have gradient");
        assert!(params[1].grad().is_some(), "Bias should have gradient");
    }

    #[test]
    fn test_linear_known_values() {
        // Manual test: W = [[1, 0], [0, 1], [1, 1]], b = [0.1, 0.2, 0.3]
        // x = [[1, 2]]
        // y = x @ W^T + b = [[1, 2]] @ [[1, 0, 1], [0, 1, 1]] + [0.1, 0.2, 0.3]
        //   = [[1, 2, 3]] + [0.1, 0.2, 0.3] = [[1.1, 2.2, 3.3]]
        let layer = Linear::new(2, 3);
        // Override with known values
        layer.parameters()[0].set_data(Tensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[3, 2],
        ));
        layer.parameters()[1].set_data(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]));

        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
        let y = layer.forward(&x);
        let data = y.data().to_vec();
        assert_abs_diff_eq!(data[0], 1.1, epsilon = 1e-5);
        assert_abs_diff_eq!(data[1], 2.2, epsilon = 1e-5);
        assert_abs_diff_eq!(data[2], 3.3, epsilon = 1e-5);
    }
}
