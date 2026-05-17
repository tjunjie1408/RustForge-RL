//! Learning rate scheduling strategies.
//!
//! Provides time-dependent learning rate functions for use with any `Optimizer`
//! via `optimizer.set_lr(schedule.lr_at(step))`.
//!
//! ## Available Schedules
//!
//! | Schedule | Formula | Use Case |
//! |----------|---------|----------|
//! | `Constant` | `lr(t) = lr` | Baseline / debugging |
//! | `LinearDecay` | `lr(t) = start + (end - start) * t / total` | PPO, A2C |
//! | `CosineAnnealing` | `lr(t) = min + (max - min) * (1 + cos(πt/T)) / 2` | SAC, fine-tuning |
//!
//! ## Example
//!
//! ```rust,ignore
//! let schedule = LRSchedule::linear_decay(1e-3, 1e-5, 10_000);
//! for step in 0..10_000 {
//!     optimizer.set_lr(schedule.lr_at(step));
//!     // ... train_step ...
//! }
//! ```

/// Learning rate schedule.
///
/// Computes the learning rate as a function of the current training step.
pub enum LRSchedule {
    /// Constant learning rate (no decay).
    Constant(f32),
    /// Linear decay from `start` to `end` over `total_steps`.
    /// After `total_steps`, clamped at `end`.
    LinearDecay {
        start: f32,
        end: f32,
        total_steps: usize,
    },
    /// Cosine annealing between `max_lr` and `min_lr` with period `t_max`.
    /// After `t_max`, clamped at `min_lr`.
    CosineAnnealing {
        max_lr: f32,
        min_lr: f32,
        t_max: usize,
    },
}

impl LRSchedule {
    /// Creates a constant schedule.
    pub fn constant(lr: f32) -> Self {
        LRSchedule::Constant(lr)
    }

    /// Creates a linear decay schedule.
    pub fn linear_decay(start: f32, end: f32, total_steps: usize) -> Self {
        LRSchedule::LinearDecay {
            start,
            end,
            total_steps,
        }
    }

    /// Creates a cosine annealing schedule.
    pub fn cosine_annealing(max_lr: f32, min_lr: f32, t_max: usize) -> Self {
        LRSchedule::CosineAnnealing {
            max_lr,
            min_lr,
            t_max,
        }
    }

    /// Returns the learning rate at the given step.
    pub fn lr_at(&self, step: usize) -> f32 {
        match self {
            LRSchedule::Constant(lr) => *lr,
            LRSchedule::LinearDecay {
                start,
                end,
                total_steps,
            } => {
                if step >= *total_steps {
                    return *end;
                }
                let fraction = step as f32 / *total_steps as f32;
                start + (end - start) * fraction
            }
            LRSchedule::CosineAnnealing {
                max_lr,
                min_lr,
                t_max,
            } => {
                if step >= *t_max {
                    return *min_lr;
                }
                let fraction = step as f32 / *t_max as f32;
                let cosine = (std::f32::consts::PI * fraction).cos();
                min_lr + (max_lr - min_lr) * (1.0 + cosine) / 2.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn constant_always_same() {
        let s = LRSchedule::constant(0.001);
        assert_abs_diff_eq!(s.lr_at(0), 0.001, epsilon = 1e-8);
        assert_abs_diff_eq!(s.lr_at(100), 0.001, epsilon = 1e-8);
        assert_abs_diff_eq!(s.lr_at(100_000), 0.001, epsilon = 1e-8);
    }

    #[test]
    fn linear_decay_boundaries() {
        let s = LRSchedule::linear_decay(1.0, 0.0, 100);

        // Start
        assert_abs_diff_eq!(s.lr_at(0), 1.0, epsilon = 1e-6);
        // Mid
        assert_abs_diff_eq!(s.lr_at(50), 0.5, epsilon = 1e-6);
        // End
        assert_abs_diff_eq!(s.lr_at(100), 0.0, epsilon = 1e-6);
        // Beyond: clamped
        assert_abs_diff_eq!(s.lr_at(200), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn linear_decay_non_zero_end() {
        let s = LRSchedule::linear_decay(1e-3, 1e-5, 1000);

        assert_abs_diff_eq!(s.lr_at(0), 1e-3, epsilon = 1e-8);
        assert_abs_diff_eq!(s.lr_at(1000), 1e-5, epsilon = 1e-8);
        assert_abs_diff_eq!(s.lr_at(2000), 1e-5, epsilon = 1e-8);

        // Monotonically decreasing
        let lr_100 = s.lr_at(100);
        let lr_500 = s.lr_at(500);
        assert!(lr_100 > lr_500);
    }

    #[test]
    fn cosine_annealing_boundaries() {
        let s = LRSchedule::cosine_annealing(1.0, 0.0, 100);

        // Start: max_lr
        assert_abs_diff_eq!(s.lr_at(0), 1.0, epsilon = 1e-6);
        // Mid: (max+min)/2 = 0.5
        assert_abs_diff_eq!(s.lr_at(50), 0.5, epsilon = 1e-5);
        // End: min_lr
        assert_abs_diff_eq!(s.lr_at(100), 0.0, epsilon = 1e-5);
        // Beyond: clamped at min
        assert_abs_diff_eq!(s.lr_at(200), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_annealing_symmetry() {
        // lr at step t and step (t_max - t) should sum to max_lr + min_lr
        let s = LRSchedule::cosine_annealing(1.0, 0.1, 200);

        for t in 0..=100 {
            let lr_t = s.lr_at(t);
            let lr_mirror = s.lr_at(200 - t);
            assert_abs_diff_eq!(lr_t + lr_mirror, 1.0 + 0.1, epsilon = 1e-5);
        }
    }

    #[test]
    fn set_lr_integration_with_adam() {
        // Verify set_lr actually changes the lr that Adam uses
        use rustforge_autograd::optimizer::adam::Adam;
        use rustforge_autograd::variable::Variable;
        use rustforge_autograd::Optimizer;
        use rustforge_tensor::Tensor;

        let w = Variable::new(Tensor::from_vec(vec![5.0], &[1]), true);
        let mut adam = Adam::new(vec![w.clone()], 0.1);
        assert_abs_diff_eq!(adam.lr(), 0.1, epsilon = 1e-8);

        adam.set_lr(0.01);
        assert_abs_diff_eq!(adam.lr(), 0.01, epsilon = 1e-8);

        // Use backward to populate gradient: loss = w → dloss/dw = 1
        let loss = w.sum();
        loss.backward();
        let before = w.data().to_vec()[0];
        adam.step();
        let after = w.data().to_vec()[0];

        // Adam step direction is correct (decreases w since grad > 0)
        assert!(after < before);
    }

    #[test]
    fn set_lr_integration_with_sgd() {
        use rustforge_autograd::optimizer::sgd::SGD;
        use rustforge_autograd::variable::Variable;
        use rustforge_autograd::Optimizer;
        use rustforge_tensor::Tensor;

        let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let mut sgd = SGD::new(vec![w.clone()], 1.0, 0.0);
        assert_abs_diff_eq!(sgd.lr(), 1.0, epsilon = 1e-8);

        sgd.set_lr(0.1);
        assert_abs_diff_eq!(sgd.lr(), 0.1, epsilon = 1e-8);

        // Use backward to populate gradient: loss = w → dloss/dw = 1
        // Step with lr=0.1, grad=1.0: w = 10 - 0.1*1 = 9.9
        let loss = w.sum();
        loss.backward();
        sgd.step();
        assert_abs_diff_eq!(w.data().to_vec()[0], 9.9, epsilon = 1e-6);
    }
}
