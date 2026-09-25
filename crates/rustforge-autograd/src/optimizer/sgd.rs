//! Stochastic Gradient Descent optimizer with optional momentum.
//!
//! ## Update Rules
//!
//! **Vanilla SGD** (momentum = 0):
//! ```text
//! θ ← θ - lr · ∇θ
//! ```
//!
//! **SGD with momentum**:
//! ```text
//! v ← μ · v + ∇θ
//! θ ← θ - lr · v
//! ```
//!
//! Momentum helps accelerate SGD in the relevant direction and dampens oscillations.

use ndarray::Zip;
use rustforge_tensor::Tensor;

use crate::variable::Variable;

use super::Optimizer;

/// SGD optimizer with optional momentum.
///
/// ## Example
/// ```rust
/// use rustforge_tensor::Tensor;
/// use rustforge_autograd::{Variable, optimizer::{Optimizer, sgd::SGD}};
///
/// let weight = Variable::new(Tensor::zeros(&[2, 2]), true);
/// let mut sgd = SGD::new(vec![weight.clone()], 0.01, 0.9);
///
/// // Simulate a backward pass
/// let loss = weight.sum();
/// loss.backward();
/// sgd.step();
/// ```
pub struct SGD {
    /// Parameters to optimize.
    params: Vec<Variable>,
    /// Learning rate.
    lr: f32,
    /// Momentum coefficient (0.0 = no momentum).
    momentum: f32,
    /// Velocity buffers for momentum (one per parameter).
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    /// Creates a new SGD optimizer.
    ///
    /// ## Arguments
    /// - `params`: Variables to optimize (must have `requires_grad = true`).
    /// - `lr`: Learning rate (e.g. 0.01).
    /// - `momentum`: Momentum coefficient (0.0 for vanilla SGD, typically 0.9).
    pub fn new(params: Vec<Variable>, lr: f32, momentum: f32) -> Self {
        let n = params.len();
        SGD {
            params,
            lr,
            momentum,
            velocities: vec![None; n],
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        let (lr, momentum) = (self.lr, self.momentum);
        for (param, velocity) in self.params.iter().zip(&mut self.velocities) {
            param.update_with_grad(|data, grad| {
                if momentum > 0.0 {
                    // v = μ·v + grad ;  θ = θ - lr·v
                    let v = match velocity {
                        Some(v) => {
                            Zip::from(v.data_mut())
                                .and(grad.data())
                                .for_each(|v, &g| *v = *v * momentum + g);
                            v
                        }
                        None => velocity.insert(grad.clone()),
                    };
                    Zip::from(data.data_mut())
                        .and(v.data())
                        .for_each(|p, &v| *p -= v * lr);
                } else {
                    // θ = θ - lr·∇θ
                    Zip::from(data.data_mut())
                        .and(grad.data())
                        .for_each(|p, &g| *p -= g * lr);
                }
            });
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn lr(&self) -> f32 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_vanilla() {
        // θ = 10.0, grad = 2.0, lr = 0.1
        // After step: θ = 10.0 - 0.1 * 2.0 = 9.8
        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        w.accumulate_grad(Tensor::from_vec(vec![2.0], &[1]));

        let mut sgd = SGD::new(vec![w.clone()], 0.1, 0.0);
        sgd.step();

        let data = w.data().to_vec();
        assert!((data[0] - 9.8).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum() {
        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        w.accumulate_grad(Tensor::from_vec(vec![2.0], &[1]));

        let mut sgd = SGD::new(vec![w.clone()], 0.1, 0.9);

        // Step 1: v = 0.9*0 + 2.0 = 2.0, θ = 10.0 - 0.1*2.0 = 9.8
        sgd.step();
        assert!((w.data().to_vec()[0] - 9.8).abs() < 1e-6);

        // Step 2: grad = 2.0 again, v = 0.9*2.0 + 2.0 = 3.8, θ = 9.8 - 0.1*3.8 = 9.42
        w.zero_grad();
        w.accumulate_grad(Tensor::from_vec(vec![2.0], &[1]));
        sgd.step();
        assert!((w.data().to_vec()[0] - 9.42).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_zero_grad() {
        let w = Variable::new(Tensor::from_vec(vec![5.0], &[1]), true);
        w.accumulate_grad(Tensor::from_vec(vec![1.0], &[1]));
        assert!(w.grad().is_some());

        let mut sgd = SGD::new(vec![w.clone()], 0.01, 0.0);
        sgd.zero_grad();
        assert!(w.grad().is_none());
    }
}
