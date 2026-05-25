//! RMSprop optimizer.
//!
//! ## Update Rules
//!
//! ```text
//! v ← α·v + (1-α)·g²
//! if momentum:
//!     m ← momentum·m + g / (√v + ε)
//!     θ ← θ - lr · m
//! else:
//!     θ ← θ - lr · g / (√v + ε)
//! ```

use super::Optimizer;
use crate::variable::Variable;
use rustforge_tensor::Tensor;

/// RMSprop optimizer.
///
/// ## Example
/// ```rust
/// use rustforge_tensor::Tensor;
/// use rustforge_autograd::{Variable, optimizer::{Optimizer, rmsprop::RMSprop}};
///
/// let w = Variable::new(Tensor::zeros(&[2, 2]), true);
/// // lr=0.01, alpha=0.99, eps=1e-8, momentum=None
/// let mut opt = RMSprop::new(vec![w.clone()], 0.01, 0.99, 1e-8, None);
///
/// let loss = w.sum();
/// loss.backward();
/// opt.step();
/// ```
pub struct RMSprop {
    params: Vec<Variable>,
    lr: f32,
    alpha: f32,
    eps: f32,
    momentum: Option<f32>,
    /// Moving average of squared gradients
    v: Vec<Tensor>,
    /// Momentum buffer (empty if momentum is None)
    m: Vec<Tensor>,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer.
    ///
    /// ## Arguments
    /// - `params`: Variables to optimize.
    /// - `lr`: Learning rate.
    /// - `alpha`: Smoothing constant (default in PyTorch is 0.99).
    /// - `eps`: Term added to denominator to improve numerical stability (default 1e-8).
    /// - `momentum`: Optional momentum factor.
    pub fn new(
        params: Vec<Variable>,
        lr: f32,
        alpha: f32,
        eps: f32,
        momentum: Option<f32>,
    ) -> Self {
        let v = params.iter().map(|p| Tensor::zeros(&p.shape())).collect();
        let m = if momentum.is_some() {
            params.iter().map(|p| Tensor::zeros(&p.shape())).collect()
        } else {
            Vec::new()
        };

        RMSprop {
            params,
            lr,
            alpha,
            eps,
            momentum,
            v,
            m,
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                // v ← α·v + (1-α)·g²
                let grad_sq = &grad * &grad;
                self.v[i] = &(&self.v[i] * self.alpha) + &(&grad_sq * (1.0 - self.alpha));

                let denom = &self.v[i].sqrt() + self.eps;
                let step_val = &grad / &denom;

                let update = if let Some(mom) = self.momentum {
                    // m ← momentum·m + g / (√v + ε)
                    self.m[i] = &(&self.m[i] * mom) + &step_val;
                    &self.m[i] * self.lr
                } else {
                    &step_val * self.lr
                };

                let new_data = {
                    let param_d = param.data();
                    &*param_d - &update
                };
                param.set_data(new_data);
            }
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
    fn test_rmsprop_no_momentum() {
        let w = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        w.accumulate_grad(&Tensor::from_vec(vec![2.0], &[1]));

        let mut opt = RMSprop::new(vec![w.clone()], 0.1, 0.5, 0.0, None);
        // alpha=0.5, grad=2.0
        // v = 0.5 * 0 + 0.5 * 4 = 2.0
        // step = grad / sqrt(v) = 2.0 / sqrt(2.0) = sqrt(2.0) = 1.4142135
        // w = 1.0 - 0.1 * 1.4142135 = 0.8585786
        opt.step();

        let new_w = w.data().to_vec()[0];
        assert!((new_w - 0.8585786).abs() < 1e-5);
    }

    #[test]
    fn test_rmsprop_momentum() {
        let w = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        w.accumulate_grad(&Tensor::from_vec(vec![2.0], &[1]));

        let mut opt = RMSprop::new(vec![w.clone()], 0.1, 0.5, 0.0, Some(0.9));
        // step 1:
        // v = 2.0
        // step_val = sqrt(2.0) = 1.4142135
        // m = 0.9 * 0 + 1.4142135 = 1.4142135
        // update = m * 0.1 = 0.14142135
        // w = 1.0 - 0.14142135 = 0.8585786
        opt.step();
        assert!((w.data().to_vec()[0] - 0.8585786).abs() < 1e-5);

        // step 2:
        w.zero_grad();
        w.accumulate_grad(&Tensor::from_vec(vec![2.0], &[1]));
        // v = 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        // step_val = 2.0 / sqrt(3.0) = 1.1547005
        // m = 0.9 * 1.4142135 + 1.1547005 = 1.2727921 + 1.1547005 = 2.4274926
        // update = m * 0.1 = 0.24274926
        // w = 0.8585786 - 0.24274926 = 0.6158293
        opt.step();
        assert!((w.data().to_vec()[0] - 0.6158293).abs() < 1e-5);
    }

    #[test]
    fn test_rmsprop_convergence() {
        // Fit y = a * x^2 + b on synthetic data
        let true_a = 3.0;
        let true_b = -1.5;

        let a = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let b = Variable::new(Tensor::from_vec(vec![0.0], &[1]), true);
        let mut opt = RMSprop::new(vec![a.clone(), b.clone()], 0.05, 0.99, 1e-8, Some(0.9));

        for _ in 0..1000 {
            opt.zero_grad();

            // We optimize expected MSE loss on an arbitrary set of points
            // loss = (a*x^2 + b - y)^2
            // To simplify, let's just use gradients for expected loss directly,
            // or just compute loss manually since we don't have full NN API in this test scope easily.
            // Actually, we can use autograd API!
            // Let's use two points: x=1, x=2
            // x=1: y = 3 - 1.5 = 1.5
            // x=2: y = 12 - 1.5 = 10.5

            let x1 = Variable::new(Tensor::from_vec(vec![1.0], &[1]), false);
            let y1_true = Variable::new(Tensor::from_vec(vec![1.5], &[1]), false);
            let y1_pred = &(&a * &x1.pow(2.0)) + &b;
            let err1 = &y1_pred - &y1_true;
            let loss1 = &err1 * &err1;

            let x2 = Variable::new(Tensor::from_vec(vec![2.0], &[1]), false);
            let y2_true = Variable::new(Tensor::from_vec(vec![10.5], &[1]), false);
            let y2_pred = &(&a * &x2.pow(2.0)) + &b;
            let err2 = &y2_pred - &y2_true;
            let loss2 = &err2 * &err2;

            let total_loss = &loss1 + &loss2;
            total_loss.backward();

            opt.step();
        }

        let a_final = a.data().to_vec()[0];
        let b_final = b.data().to_vec()[0];

        assert!((a_final - true_a).abs() < 1e-3, "a: {}", a_final);
        assert!((b_final - true_b).abs() < 1e-3, "b: {}", b_final);
    }

    #[test]
    fn test_rmsprop_set_lr() {
        let w = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        w.accumulate_grad(&Tensor::from_vec(vec![2.0], &[1]));

        let mut opt = RMSprop::new(vec![w.clone()], 0.1, 0.5, 0.0, None);
        assert_eq!(opt.lr(), 0.1, "Initial learning rate should be 0.1");

        opt.set_lr(0.2);
        assert_eq!(opt.lr(), 0.2, "Updated learning rate should be 0.2");

        // With lr=0.2, alpha=0.5, grad=2.0
        // v = 0.5 * 0 + 0.5 * 4 = 2.0
        // step = grad / sqrt(v) = 2.0 / sqrt(2.0) = sqrt(2.0) = 1.4142135
        // w = 1.0 - 0.2 * 1.4142135 = 0.7171573
        opt.step();

        let new_w = w.data().to_vec()[0];
        assert!((new_w - 0.7171573).abs() < 1e-5, "Expected parameter value after step to be 0.7171573, got {}", new_w);
    }

    #[test]
    fn test_rmsprop_zero_grad() {
        let w1 = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        let w2 = Variable::new(Tensor::from_vec(vec![2.0], &[1]), true);
        w1.accumulate_grad(&Tensor::from_vec(vec![2.0], &[1]));
        w2.accumulate_grad(&Tensor::from_vec(vec![3.0], &[1]));

        let mut opt = RMSprop::new(vec![w1.clone(), w2.clone()], 0.1, 0.5, 0.0, None);

        assert!(w1.grad().is_some(), "w1 should have a gradient before zero_grad");
        assert!(w2.grad().is_some(), "w2 should have a gradient before zero_grad");

        opt.zero_grad();

        assert!(w1.grad().is_none(), "w1 gradient should be cleared after zero_grad");
        assert!(w2.grad().is_none(), "w2 gradient should be cleared after zero_grad");
    }

    #[test]
    fn test_rmsprop_no_grad_safety() {
        let w = Variable::new(Tensor::from_vec(vec![1.5], &[1]), true);
        let mut opt = RMSprop::new(vec![w.clone()], 0.1, 0.5, 0.0, None);

        assert!(w.grad().is_none(), "w should not have any gradient initially");

        // Step should be a no-op for w
        opt.step();

        let w_val = w.data().to_vec()[0];
        assert_eq!(w_val, 1.5, "w data should remain unchanged when there is no gradient");
    }
}
