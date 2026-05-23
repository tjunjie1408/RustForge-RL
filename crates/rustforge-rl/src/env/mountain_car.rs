//! MountainCar-Continuous environment — classic continuous-control hill-climbing task.
//!
//! # Physics
//!
//! A car is placed in a valley between two hills. The engine is too weak to climb the
//! right hill directly; the agent must learn to rock back and forth, building momentum
//! to reach the goal position on the right hilltop.
//!
//! # State Space
//!
//! `[f32; 2]` = `[position, velocity]`
//! - `position`: Car position along the 1D track (-1.2 to 0.6)
//! - `velocity`: Car velocity (-0.07 to 0.07)
//!
//! # Action Space
//!
//! `[f32; 1]` — a single continuous force in [-1.0, 1.0].
//! Actions outside this range are clamped.
//!
//! # Reward
//!
//! - `100.0` when the car reaches the goal (position >= 0.45).
//! - `-0.1 * action² ` otherwise (penalizes large forces to encourage efficient solutions).
//!
//! # NaN Defense
//!
//! After each physics step, a health check validates that no state element is NaN or Inf.
//! If floating-point poisoning is detected, the episode is immediately terminated with
//! a safe default observation returned.
//!
//! # Reproducibility
//!
//! Internal `StdRng` instance. `reset(Some(seed))` re-seeds; `reset(None)` continues
//! the existing PRNG stream for deterministic multi-episode trajectories.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use super::spaces::Space;
use super::traits::Environment;

/// Type alias for the continuous mountain car action: a single force value.
pub type MountainCarAction = [f32; 1];

// Physics constants matching Gymnasium MountainCarContinuous-v0
const POWER: f32 = 0.0015;
const MIN_POS: f32 = -1.2;
const MAX_POS: f32 = 0.6;
const MAX_SPEED: f32 = 0.07;
const GOAL_POS: f32 = 0.45;

/// MountainCar-Continuous environment following Gymnasium specification.
///
/// # Constants (matching Gymnasium source)
/// - Power: 0.0015
/// - Position bounds: [-1.2, 0.6]
/// - Velocity bounds: [-0.07, 0.07]
/// - Goal position: 0.45
/// - Max steps: 999 (default)
pub struct MountainCarContinuous {
    /// Current state: [position, velocity]
    state: [f32; 2],
    /// Internal PRNG for reproducibility
    rng: StdRng,
    /// Current step count within the episode
    steps: usize,
    /// Maximum steps before truncation
    max_steps: usize,
}

impl MountainCarContinuous {
    /// Create a new MountainCarContinuous environment with default settings.
    ///
    /// The environment is created with an unseeded RNG. Call `reset(Some(seed))`
    /// before use to ensure reproducibility. Default `max_steps` is 999.
    pub fn new() -> Self {
        MountainCarContinuous {
            state: [0.0; 2],
            rng: StdRng::from_entropy(),
            steps: 0,
            max_steps: 999,
        }
    }

    /// Create a MountainCarContinuous with a custom maximum step limit.
    pub fn with_max_steps(max_steps: usize) -> Self {
        MountainCarContinuous {
            state: [0.0; 2],
            rng: StdRng::from_entropy(),
            steps: 0,
            max_steps,
        }
    }

    /// Check if the current state contains NaN or Inf values (floating-point poisoning).
    ///
    /// Returns `true` if any state element is non-finite, indicating the physics
    /// simulation has been corrupted.
    #[inline]
    fn is_state_poisoned(&self) -> bool {
        self.state.iter().any(|x| !x.is_finite())
    }

    /// Returns a safe default observation within valid bounds.
    ///
    /// Used when floating-point poisoning is detected to prevent NaN propagation
    /// to downstream neural networks.
    #[inline]
    fn safe_default_obs() -> [f32; 2] {
        [0.0, 0.0]
    }

    /// Get mutable reference to internal state.
    ///
    /// Intended for testing and debugging (e.g., positioning car near goal).
    /// Not recommended for use in production training loops.
    pub fn state_mut(&mut self) -> &mut [f32; 2] {
        &mut self.state
    }
}

impl Default for MountainCarContinuous {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for MountainCarContinuous {
    type Obs = [f32; 2];
    type Act = [f32; 1];
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        // Reseed or continue PRNG
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        // Position uniformly sampled from [-0.6, -0.4], velocity = 0.0
        let position = self.rng.gen_range(-0.6f32..=-0.4f32);
        self.state = [position, 0.0];
        self.steps = 0;

        (self.state, ())
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        // Clamp action to valid range
        let force = action[0].clamp(-1.0, 1.0);

        let [position, velocity] = self.state;

        // Physics update (matches Gymnasium MountainCarContinuous)
        let new_velocity = (velocity + force * POWER - 0.0025 * (3.0 * position).cos())
            .clamp(-MAX_SPEED, MAX_SPEED);
        let new_position = (position + new_velocity).clamp(MIN_POS, MAX_POS);

        // Left wall: reset velocity to zero when hitting the boundary
        let new_velocity = if new_position == MIN_POS {
            0.0
        } else {
            new_velocity
        };

        self.state = [new_position, new_velocity];
        self.steps += 1;

        // NaN defense: health check after physics update
        if self.is_state_poisoned() {
            return (Self::safe_default_obs(), 0.0, true, false, ());
        }

        // Check termination: reached the goal
        let terminated = new_position >= GOAL_POS;
        let truncated = !terminated && self.steps >= self.max_steps;

        // Reward: +100 at goal, otherwise penalty proportional to action squared
        let reward = if terminated {
            100.0
        } else {
            -0.1 * force * force
        };

        (self.state, reward, terminated, truncated, ())
    }

    fn action_space(&self) -> Space {
        Space::continuous(vec![-1.0], vec![1.0])
    }

    fn observation_space(&self) -> Space {
        Space::continuous(vec![MIN_POS, -MAX_SPEED], vec![MAX_POS, MAX_SPEED])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn reset_position_range() {
        let mut env = MountainCarContinuous::new();
        for seed in 0..100 {
            let (obs, _) = env.reset(Some(seed));
            assert!(
                obs[0] >= -0.6 && obs[0] <= -0.4,
                "seed {}: position {} not in [-0.6, -0.4]",
                seed,
                obs[0]
            );
            assert_abs_diff_eq!(obs[1], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn physics_gravity_pull() {
        // From the valley bottom with zero action, gravity should pull
        // the car toward the left (negative velocity due to cos(3 * -0.5) > 0).
        let mut env = MountainCarContinuous::new();
        env.reset(Some(42));

        // Manually set state to valley bottom
        *env.state_mut() = [-0.5, 0.0];

        let (obs, _reward, _terminated, _truncated, _) = env.step([0.0]);

        // Gravity term: -0.0025 * cos(3 * -0.5) = -0.0025 * cos(-1.5)
        // cos(-1.5) ≈ 0.0707, so gravity contribution ≈ -0.000177
        // Velocity should be negative (car pulled downhill)
        assert!(
            obs[1] < 0.0,
            "velocity should be negative due to gravity, got {}",
            obs[1]
        );
    }

    #[test]
    fn goal_terminates() {
        let mut env = MountainCarContinuous::new();
        env.reset(Some(0));

        // Place car just below the goal with some positive velocity
        *env.state_mut() = [0.44, 0.05];

        let (_obs, reward, terminated, _truncated, _) = env.step([1.0]);

        assert!(terminated, "should terminate when position >= GOAL_POS");
        assert_abs_diff_eq!(reward, 100.0, epsilon = 1e-6);
    }

    #[test]
    fn left_wall_velocity_reset() {
        let mut env = MountainCarContinuous::new();
        env.reset(Some(0));

        // Place car near left wall with negative velocity
        *env.state_mut() = [-1.19, -0.05];

        let (obs, _reward, _terminated, _truncated, _) = env.step([0.0]);

        // Position should be clipped to MIN_POS
        assert_abs_diff_eq!(obs[0], MIN_POS, epsilon = 1e-6);
        // Velocity should be reset to 0 when hitting the left wall
        assert_abs_diff_eq!(obs[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn action_clipping() {
        let mut env = MountainCarContinuous::new();
        env.reset(Some(42));
        *env.state_mut() = [-0.5, 0.0];

        // Step with action=5.0 (should be clamped to 1.0)
        let (obs_clamped, _, _, _, _) = env.step([5.0]);

        // Reset and step with action=1.0 for comparison
        env.reset(Some(42));
        *env.state_mut() = [-0.5, 0.0];
        let (obs_normal, _, _, _, _) = env.step([1.0]);

        // Both should produce identical results
        assert_abs_diff_eq!(obs_clamped[0], obs_normal[0], epsilon = 1e-9);
        assert_abs_diff_eq!(obs_clamped[1], obs_normal[1], epsilon = 1e-9);
    }

    #[test]
    fn truncation_at_max_steps() {
        let max_steps = 10;
        let mut env = MountainCarContinuous::with_max_steps(max_steps);
        env.reset(Some(0));

        // Run for max_steps with zero action (should not reach goal)
        let mut truncated = false;
        for _ in 0..max_steps {
            let result = env.step([0.0]);
            truncated = result.3;
        }

        assert!(truncated, "should be truncated after max_steps");
    }

    #[test]
    fn reward_penalty_proportional_to_action() {
        let mut env = MountainCarContinuous::new();

        // Small action
        env.reset(Some(42));
        *env.state_mut() = [-0.5, 0.0];
        let (_, reward_small, terminated_small, _, _) = env.step([0.1]);

        // Large action
        env.reset(Some(42));
        *env.state_mut() = [-0.5, 0.0];
        let (_, reward_large, terminated_large, _, _) = env.step([0.9]);

        // Neither should terminate from the valley
        assert!(!terminated_small);
        assert!(!terminated_large);

        // Expected: reward = -0.1 * action^2
        let expected_small = -0.1 * 0.1 * 0.1;
        let expected_large = -0.1 * 0.9 * 0.9;

        assert_abs_diff_eq!(reward_small, expected_small, epsilon = 1e-6);
        assert_abs_diff_eq!(reward_large, expected_large, epsilon = 1e-6);

        // Larger action should incur a larger (more negative) penalty
        assert!(
            reward_large < reward_small,
            "larger action should have more negative reward: {} vs {}",
            reward_large,
            reward_small
        );
    }
}
