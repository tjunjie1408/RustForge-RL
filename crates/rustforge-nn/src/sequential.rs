//! Sequential container — chains multiple modules into a pipeline.
//!
//! `Sequential` applies modules in order, passing the output of each
//! as the input to the next. This is the simplest way to build a
//! feed-forward neural network.
//!
//! ## Example
//! ```rust,ignore
//! use rustforge_nn::*;
//!
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(2, 16)),
//!     Box::new(ReLU),
//!     Box::new(Linear::new(16, 1)),
//!     Box::new(Sigmoid),
//! ]);
//!
//! let output = model.forward(&input);
//! let params = model.parameters();
//! ```

use rustforge_autograd::Variable;

use crate::module::Module;

/// Sequential container that chains modules in order.
///
/// Each module's output becomes the next module's input:
/// `output = layers[n]( ... layers[1]( layers[0](input) ) )`
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    /// Creates a new Sequential container from a list of modules.
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential {
            layers,
            training: true,
        }
    }

    /// Returns the number of layers in the container.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns whether the container is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    /// Forward pass: chains all layers sequentially.
    fn forward(&self, input: &Variable) -> Variable {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Returns parameters from all child layers (in layer order).
    fn parameters(&self) -> Vec<Variable> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    /// Propagates training mode to all child layers.
    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    /// Returns whether the container is in training mode.
    fn is_training(&self) -> bool {
        self.training
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::ReLU;
    use crate::linear::Linear;
    use rustforge_tensor::Tensor;

    #[test]
    fn test_sequential_forward() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 2)),
        ]);

        let x = Variable::new(Tensor::ones(&[3, 4]), false);
        let y = model.forward(&x);
        assert_eq!(y.shape(), vec![3, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 2)),
        ]);

        let params = model.parameters();
        // Linear(4,8): weight + bias = 2 params
        // ReLU: 0 params
        // Linear(8,2): weight + bias = 2 params
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_sequential_empty() {
        let model = Sequential::new(vec![]);
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);

        let x = Variable::new(Tensor::ones(&[2, 3]), false);
        let y = model.forward(&x);
        assert_eq!(y.shape(), vec![2, 3]);
    }

    #[test]
    fn test_sequential_gradient_flow() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 1)),
        ]);

        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
        let y = model.forward(&x);
        y.sum().backward();

        // All parameters should have gradients
        for (i, param) in model.parameters().iter().enumerate() {
            assert!(
                param.grad().is_some(),
                "Parameter {} should have gradient",
                i
            );
        }
    }
}
