//! Epsilon-greedy exploration strategy for DQN.
//!
//! Balances exploration (random action) and exploitation (greedy action)
//! via a linearly decaying epsilon parameter.
//!
//! ## Decay Schedule
//!
//! ```text
//! ε(t) = max(ε_end, ε_start - (ε_start - ε_end) * t / decay_steps)
//! ```
//!
//! - At `t=0`: `ε = ε_start` (high exploration)
//! - At `t=decay_steps`: `ε = ε_end` (mostly exploitation)
//! - After `t > decay_steps`: `ε = ε_end` (clamped)

use rand::Rng;
use rustforge_tensor::Tensor;

/// Linear-decay epsilon-greedy exploration strategy.
///
/// ## Example
/// ```rust,ignore
/// let mut eps = EpsilonGreedy::new(1.0, 0.01, 10_000);
///
/// // Early training: mostly random
/// let action = eps.select_action(&q_values, 0, num_actions);
///
/// // Late training: mostly greedy
/// let action = eps.select_action(&q_values, 15_000, num_actions);
/// ```
pub struct EpsilonGreedy {
    /// Starting epsilon (high exploration).
    epsilon_start: f32,
    /// Final epsilon (low exploration).
    epsilon_end: f32,
    /// Number of steps over which epsilon decays linearly.
    decay_steps: usize,
}

impl EpsilonGreedy {
    /// Creates a new epsilon-greedy strategy.
    ///
    /// ## Arguments
    /// - `epsilon_start`: Initial epsilon (typically 1.0 = fully random).
    /// - `epsilon_end`: Final epsilon (typically 0.01 = 1% random).
    /// - `decay_steps`: Number of steps to linearly decay from start to end.
    pub fn new(epsilon_start: f32, epsilon_end: f32, decay_steps: usize) -> Self {
        EpsilonGreedy {
            epsilon_start,
            epsilon_end,
            decay_steps,
        }
    }

    /// Returns the current epsilon value given the step count.
    ///
    /// Linearly interpolates between `start` and `end`, clamped at `end`.
    pub fn epsilon(&self, step: usize) -> f32 {
        if step >= self.decay_steps {
            return self.epsilon_end;
        }
        let fraction = step as f32 / self.decay_steps as f32;
        self.epsilon_start + (self.epsilon_end - self.epsilon_start) * fraction
    }

    /// Selects an action using epsilon-greedy strategy.
    ///
    /// With probability `ε`: returns a uniformly random action (exploration).
    /// With probability `1-ε`: returns `argmax(q_values)` (exploitation).
    ///
    /// ## Arguments
    /// - `q_values`: Q-values for each action, shape `[num_actions]` or `[1, num_actions]`.
    /// - `step`: Current training step (for epsilon decay).
    /// - `num_actions`: Total number of discrete actions.
    ///
    /// ## Returns
    /// The selected action index.
    pub fn select_action(&self, q_values: &Tensor, step: usize, num_actions: usize) -> usize {
        let eps = self.epsilon(step);
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < eps {
            // Explore: random action
            rng.gen_range(0..num_actions)
        } else {
            // Exploit: argmax of Q-values
            let flat = q_values.to_vec();
            flat.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsilon_decay() {
        let eps = EpsilonGreedy::new(1.0, 0.01, 1000);

        // At step 0: epsilon = 1.0
        assert!((eps.epsilon(0) - 1.0).abs() < 1e-6);

        // At step 500: epsilon = 0.505
        assert!((eps.epsilon(500) - 0.505).abs() < 1e-3);

        // At step 1000: epsilon = 0.01
        assert!((eps.epsilon(1000) - 0.01).abs() < 1e-6);

        // After decay: clamped at 0.01
        assert!((eps.epsilon(5000) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_greedy_action() {
        let eps = EpsilonGreedy::new(0.0, 0.0, 1); // zero epsilon = always greedy

        let q = Tensor::from_vec(vec![1.0, 5.0, 3.0], &[3]);
        let action = eps.select_action(&q, 100, 3);
        assert_eq!(action, 1); // argmax is index 1 (value 5.0)
    }

    #[test]
    fn test_random_action_distribution() {
        let eps = EpsilonGreedy::new(1.0, 1.0, 1); // epsilon=1.0 = always random

        let q = Tensor::from_vec(vec![100.0, 0.0, 0.0], &[3]); // would be greedy=0
        let mut counts = [0u32; 3];

        for _ in 0..3000 {
            let action = eps.select_action(&q, 0, 3);
            counts[action] += 1;
        }

        // Each action should get roughly 1/3 of the picks
        for count in &counts {
            assert!(
                *count > 500,
                "With epsilon=1.0, each action should be selected frequently, got {:?}",
                counts
            );
        }
    }
}
