//! Dropout regularization layer.
//!
//! During training, randomly zeroes some elements of the input tensor with
//! probability `p`, and scales the remaining elements by `1/(1-p)` to maintain
//! the expected value (inverted dropout).
//!
//! During evaluation, Dropout acts as an identity function (no-op).
//!
//! ## Why Dropout?
//!
//! Dropout prevents co-adaptation of neurons by forcing the network to learn
//! redundant representations, effectively acting as an ensemble method.
//!
//! ## Reference
//! Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks
//! from Overfitting", JMLR 2014.

use rand::Rng;
use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

use crate::module::Module;

/// Dropout layer with inverted dropout scaling.
///
/// ## Example
/// ```rust,ignore
/// let mut dropout = Dropout::new(0.5);
///
/// // Training mode: randomly masks ~50% of elements
/// dropout.set_training(true);
/// let y = dropout.forward(&x);
///
/// // Evaluation mode: identity pass-through
/// dropout.set_training(false);
/// let y = dropout.forward(&x);
/// ```
pub struct Dropout {
    /// Probability of an element being zeroed (0.0 to 1.0).
    p: f32,
    /// Whether the layer is in training mode.
    training: bool,
}

impl Dropout {
    /// Creates a new Dropout layer.
    ///
    /// ## Arguments
    /// - `p`: Dropout probability (e.g., 0.5). Must be in [0, 1).
    ///
    /// ## Panics
    /// Panics if `p` is outside [0, 1).
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Dropout { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Variable) -> Variable {
        if !self.training || self.p == 0.0 {
            // Evaluation mode: identity
            return input.clone();
        }

        // Generate Bernoulli mask: 1.0 with probability (1 - p), 0.0 with probability p
        let shape = input.shape();
        let mut rng = rand::thread_rng();
        let mask_data: Vec<f32> = (0..shape.iter().product())
            .map(|_| {
                if rng.gen::<f32>() > self.p {
                    1.0 / (1.0 - self.p) // Scale by 1/(1-p) for inverted dropout
                } else {
                    0.0
                }
            })
            .collect();

        let mask = Variable::from_tensor(Tensor::from_vec(mask_data, &shape));

        // Apply mask (gradient flows through the mask via element-wise multiply)
        input * &mask
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![] // Dropout has no learnable parameters
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dropout_eval_mode() {
        let mut dropout = Dropout::new(0.5);
        dropout.set_training(false);

        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let y = dropout.forward(&x);
        assert_eq!(y.data().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dropout_training_mode() {
        let dropout = Dropout::new(0.5);

        // With p=0.5, roughly half the elements should be zeroed
        let x = Variable::new(Tensor::ones(&[100, 100]), false);
        let y = dropout.forward(&x);

        let data = y.data().to_vec();
        let num_zeros = data.iter().filter(|&&v| v == 0.0).count();
        let total = data.len();

        // Should be roughly 50% zeros (with tolerance for randomness)
        let zero_ratio = num_zeros as f32 / total as f32;
        assert!(
            (0.35..0.65).contains(&zero_ratio),
            "Expected ~50% zeros, got {:.1}%",
            zero_ratio * 100.0
        );

        // Non-zero values should be scaled by 1/(1-p) = 2.0
        let non_zero_vals: Vec<f32> = data.into_iter().filter(|&v| v != 0.0).collect();
        if !non_zero_vals.is_empty() {
            assert_abs_diff_eq!(non_zero_vals[0], 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout_zero_prob() {
        let dropout = Dropout::new(0.0);

        let x = Variable::new(Tensor::ones(&[5, 5]), false);
        let y = dropout.forward(&x);
        assert_eq!(y.data().to_vec(), vec![1.0; 25]);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn test_dropout_invalid_prob() {
        Dropout::new(1.0);
    }
}
