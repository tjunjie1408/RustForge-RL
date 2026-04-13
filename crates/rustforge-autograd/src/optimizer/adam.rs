//! Adam optimizer (Adaptive Moment Estimation).
//!
//! ## Update Rules
//!
//! ```text
//! m ← β₁·m + (1-β₁)·g           (first moment / mean)
//! v ← β₂·v + (1-β₂)·g²          (second moment / uncentered variance)
//! m̂ ← m / (1 - β₁ᵗ)             (bias-corrected first moment)
//! v̂ ← v / (1 - β₂ᵗ)             (bias-corrected second moment)
//! θ ← θ - lr · m̂ / (√v̂ + ε)    (parameter update)
//! ```
//!
//! Default hyperparameters follow the original paper (Kingma & Ba, 2015):
//! β₁ = 0.9, β₂ = 0.999, ε = 1e-8

use rustforge_tensor::Tensor;

use crate::variable::Variable;

use super::Optimizer;

/// Adam optimizer with bias correction.
///
/// ## Example
/// ```rust,ignore
/// let mut adam = Adam::new(vec![weight.clone(), bias.clone()], 1e-3);
/// loss.backward();
/// adam.step();
/// ```
pub struct Adam {
    /// Parameters to optimize.
    params: Vec<Variable>,
    /// Learning rate.
    lr: f32,
    /// Exponential decay rate for the first moment estimates.
    beta1: f32,
    /// Exponential decay rate for the second moment estimates.
    beta2: f32,
    /// Small constant for numerical stability (prevents division by zero).
    epsilon: f32,
    /// Current timestep (incremented each step for bias correction).
    t: usize,
    /// First moment vectors (one per parameter).
    m: Vec<Tensor>,
    /// Second moment vectors (one per parameter).
    v: Vec<Tensor>,
}

impl Adam {
    /// Creates a new Adam optimizer with default betas (β₁=0.9, β₂=0.999, ε=1e-8).
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        Self::with_betas(params, lr, 0.9, 0.999, 1e-8)
    }

    /// Creates a new Adam optimizer with custom hyperparameters.
    ///
    /// ## Arguments
    /// - `params`: Variables to optimize.
    /// - `lr`: Learning rate (e.g. 1e-3).
    /// - `beta1`: First moment decay rate (default: 0.9).
    /// - `beta2`: Second moment decay rate (default: 0.999).
    /// - `epsilon`: Numerical stability constant (default: 1e-8).
    pub fn with_betas(
        params: Vec<Variable>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        let m: Vec<Tensor> = params.iter().map(|p| Tensor::zeros(&p.shape())).collect();
        let v: Vec<Tensor> = params.iter().map(|p| Tensor::zeros(&p.shape())).collect();
        Adam {
            params,
            lr,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m,
            v,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                // m = β₁·m + (1-β₁)·g
                self.m[i] = &(&self.m[i] * self.beta1) + &(&grad * (1.0 - self.beta1));

                // v = β₂·v + (1-β₂)·g²
                let grad_sq = &grad * &grad;
                self.v[i] = &(&self.v[i] * self.beta2) + &(&grad_sq * (1.0 - self.beta2));

                // Bias-corrected estimates
                let m_hat = &self.m[i] / (1.0 - self.beta1.powf(t));
                let v_hat = &self.v[i] / (1.0 - self.beta2.powf(t));

                // θ = θ - lr · m̂ / (√v̂ + ε)
                let denom = &v_hat.sqrt() + self.epsilon;
                let update = &m_hat / &denom;
                let new_data = &param.data() - &(&update * self.lr);
                param.set_data(new_data);
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_basic() {
        // Verify that Adam makes a reasonable update
        let w = Variable::new(Tensor::from_vec(vec![5.0], &[1]), true);
        w.accumulate_grad(&Tensor::from_vec(vec![2.0], &[1]));

        let mut adam = Adam::new(vec![w.clone()], 0.1);
        let before = w.data().to_vec()[0];
        adam.step();
        let after = w.data().to_vec()[0];

        // Parameter should decrease since gradient is positive
        assert!(after < before, "Adam should decrease w when grad > 0");
    }

    #[test]
    fn test_adam_multiple_steps() {
        // Verify Adam converges toward zero when gradient always points away
        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let mut adam = Adam::new(vec![w.clone()], 0.5);

        for _ in 0..20 {
            w.zero_grad();
            // gradient = w (gradient points toward higher values of w²/2)
            let current = w.data().to_vec()[0];
            w.accumulate_grad(&Tensor::from_vec(vec![current], &[1]));
            adam.step();
        }

        // Should have moved significantly toward 0
        let final_val = w.data().to_vec()[0].abs();
        assert!(
            final_val < 5.0,
            "Adam should converge toward 0, got {}",
            final_val
        );
    }

    #[test]
    fn test_adam_zero_grad() {
        let w = Variable::new(Tensor::from_vec(vec![5.0], &[1]), true);
        w.accumulate_grad(&Tensor::from_vec(vec![1.0], &[1]));
        assert!(w.grad().is_some());

        let mut adam = Adam::new(vec![w.clone()], 0.01);
        adam.zero_grad();
        assert!(w.grad().is_none());
    }
}
