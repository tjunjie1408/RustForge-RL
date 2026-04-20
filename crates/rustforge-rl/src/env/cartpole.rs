//! CartPole-v1 environment — classic inverted pendulum control task.
//!
//! # Physics
//!
//! A pole is attached by an un-actuated joint to a cart moving along a frictionless track.
//! The system is controlled by applying a force of +10N or -10N to the cart.
//! The pendulum starts upright, and the goal is to prevent it from falling over.
//!
//! # State Space
//!
//! `[f32; 4]` = `[x, x_dot, theta, theta_dot]`
//! - `x`: Cart position (-4.8 to 4.8, terminated at ±2.4)
//! - `x_dot`: Cart velocity
//! - `theta`: Pole angle in radians (terminated at ±0.2095 ≈ ±12°)
//! - `theta_dot`: Pole angular velocity
//!
//! # Action Space
//!
//! `CartPoleAction` enum: `Left` (force -10N) or `Right` (force +10N).
//! Compile-time exhaustive — no invalid actions possible.
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

/// CartPole action: discrete binary choice.
///
/// Using an enum instead of `usize` guarantees compile-time action validity —
/// no out-of-bounds checks needed in `step()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartPoleAction {
    /// Apply force to the left (-10N).
    Left,
    /// Apply force to the right (+10N).
    Right,
}

/// CartPole-v1 environment following Gymnasium specification.
///
/// # Constants (matching Gymnasium source)
/// - Gravity: 9.8 m/s²
/// - Cart mass: 1.0 kg
/// - Pole mass: 0.1 kg
/// - Pole half-length: 0.5 m
/// - Force magnitude: 10.0 N
/// - Time step (tau): 0.02 s
/// - Max steps: 500 (Gymnasium v1.0)
pub struct CartPole {
    /// Current state: [x, x_dot, theta, theta_dot]
    state: [f32; 4],
    /// Internal PRNG for reproducibility
    rng: StdRng,
    /// Current step count within the episode
    steps: usize,
    /// Maximum steps before truncation
    max_steps: usize,
}

// Physics constants matching Gymnasium CartPole-v1
const GRAVITY: f32 = 9.8;
const CART_MASS: f32 = 1.0;
const POLE_MASS: f32 = 0.1;
const TOTAL_MASS: f32 = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH: f32 = 0.5;
const POLE_MASS_LENGTH: f32 = POLE_MASS * POLE_HALF_LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;

// Termination thresholds
const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD: f32 = 12.0 * std::f32::consts::PI / 180.0; // 12 degrees in radians

impl CartPole {
    /// Create a new CartPole environment with default settings.
    ///
    /// The environment is created with an unseeded RNG. Call `reset(Some(seed))`
    /// before use to ensure reproducibility.
    pub fn new() -> Self {
        CartPole {
            state: [0.0; 4],
            rng: StdRng::from_entropy(),
            steps: 0,
            max_steps: 500,
        }
    }

    /// Create a CartPole with a custom maximum step limit.
    pub fn with_max_steps(max_steps: usize) -> Self {
        CartPole {
            state: [0.0; 4],
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
    fn safe_default_obs() -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    /// Returns whether the current state satisfies termination conditions.
    ///
    /// Termination occurs when:
    /// - Cart position `|x| > 2.4`
    /// - Pole angle `|θ| > 12°` (0.2095 rad)
    #[inline]
    fn is_terminated(&self) -> bool {
        let [x, _, theta, _] = self.state;
        x.abs() > X_THRESHOLD || theta.abs() > THETA_THRESHOLD
    }

    /// Perform Euler integration physics step.
    ///
    /// Updates `self.state` in-place using the equations of motion for
    /// the cart-pole system. Matches the Gymnasium CartPole-v1 source.
    fn physics_step(&mut self, force: f32) {
        let [x, x_dot, theta, theta_dot] = self.state;

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Acceleration calculations (Euler semi-implicit integration)
        let temp = (force + POLE_MASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        // Euler integration
        self.state = [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ];
    }

    /// Get mutable reference to internal state.
    ///
    /// Intended for testing and debugging (e.g., NaN injection fuzzing).
    /// Not recommended for use in production training loops.
    pub fn state_mut(&mut self) -> &mut [f32; 4] {
        &mut self.state
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CartPole {
    type Obs = [f32; 4];
    type Act = CartPoleAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        // Reseed or continue PRNG
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        // Initialize state with small random values in [-0.05, 0.05]
        // (matches Gymnasium default)
        self.state = [
            self.rng.gen_range(-0.05..0.05),
            self.rng.gen_range(-0.05..0.05),
            self.rng.gen_range(-0.05..0.05),
            self.rng.gen_range(-0.05..0.05),
        ];
        self.steps = 0;

        (self.state, ())
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        let force = match action {
            CartPoleAction::Left => -FORCE_MAG,
            CartPoleAction::Right => FORCE_MAG,
        };

        // Physics integration
        self.physics_step(force);
        self.steps += 1;

        // NaN defense: health check after physics update
        if self.is_state_poisoned() {
            return (Self::safe_default_obs(), 0.0, true, false, ());
        }

        // Check termination conditions
        let terminated = self.is_terminated();
        let truncated = !terminated && self.steps >= self.max_steps;

        // Reward: +1.0 for every step where the pole is still balanced
        let reward = if terminated { 0.0 } else { 1.0 };

        (self.state, reward, terminated, truncated, ())
    }

    fn action_space(&self) -> Space {
        Space::discrete(2)
    }

    fn observation_space(&self) -> Space {
        Space::continuous(
            vec![-4.8, -f32::MAX, -THETA_THRESHOLD * 2.0, -f32::MAX],
            vec![4.8, f32::MAX, THETA_THRESHOLD * 2.0, f32::MAX],
        )
    }
}
