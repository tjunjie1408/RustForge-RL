//! Pendulum-v1 environment — classic continuous-control inverted pendulum task.
//!
//! # Physics
//!
//! A pendulum is attached to a pivot. The goal is to apply torque to swing the pendulum up
//! and keep it upright.
//!
//! Note that `theta = 0` corresponds to vertical up (unstable equilibrium), and `theta = ±pi`
//! corresponds to natural hanging down (stable equilibrium). This is the standard convention
//! for inverted pendulum swing-up tasks, which is opposite to standard physics textbooks.
//!
//! # State Space
//!
//! The state represents the angle and angular velocity of the pendulum:
//! - `state: [theta, theta_dot]`
//! - `theta`: Angle (not normalized, raw cumulative angle)
//! - `theta_dot`: Angular velocity
//!
//! # Observation Space
//!
//! `[f32; 3]` = `[cos(theta), sin(theta), theta_dot]`
//! - `cos(theta) ∈ [-1.0, 1.0]`
//! - `sin(theta) ∈ [-1.0, 1.0]`
//! - `theta_dot`: Angular velocity, clamped to `[-8.0, 8.0]`
//!
//! # Action Space
//!
//! `PendulumAction` wrapping torque `f32`, clamped to `[-2.0, 2.0]`.
//!
//! # Reward
//!
//! `reward = -(angle_normalize(theta)^2 + 0.1 * theta_dot^2 + 0.001 * u^2)`
//!
//! # NaN Defense
//!
//! After each physics step, a health check validates that no state element is NaN or Inf.
//! If floating-point poisoning is detected, the episode is immediately terminated with
//! `terminated = true` and a safe default observation `[1.0, 0.0, 0.0]` is returned.
//!
//! # Reproducibility
//!
//! Internal `StdRng` instance. `reset(Some(seed))` re-seeds; `reset(None)` continues
//! the existing PRNG stream.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use super::spaces::Space;
use super::traits::Environment;

/// Pendulum continuous action: torque value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PendulumAction(pub f32);

impl PendulumAction {
    /// Create a new PendulumAction.
    pub fn new(torque: f32) -> Self {
        PendulumAction(torque)
    }
}

// Constants matching Gymnasium Pendulum-v1
const G: f32 = 10.0;
const M: f32 = 1.0;
const L: f32 = 1.0;
const DT: f32 = 0.05;
const MAX_SPEED: f32 = 8.0;
const MAX_TORQUE: f32 = 2.0;
const DEFAULT_MAX_STEPS: usize = 200;
const RESET_THETA_BOUND: f32 = std::f32::consts::PI;
const RESET_THETA_DOT_BOUND: f32 = 1.0;

/// Pendulum-v1 environment following Gymnasium specification.
pub struct Pendulum {
    /// Current state: [theta, theta_dot]
    state: [f32; 2],
    /// Internal PRNG for reproducibility
    rng: StdRng,
    /// Current step count within the episode
    steps: usize,
    /// Maximum steps before truncation
    max_steps: usize,
}

impl Pendulum {
    /// Create a new Pendulum environment with default settings.
    pub fn new() -> Self {
        Pendulum {
            state: [0.0, 0.0],
            rng: StdRng::from_entropy(),
            steps: 0,
            max_steps: DEFAULT_MAX_STEPS,
        }
    }

    /// Create a Pendulum environment with a custom maximum step limit.
    pub fn with_max_steps(max_steps: usize) -> Self {
        Pendulum {
            state: [0.0, 0.0],
            rng: StdRng::from_entropy(),
            steps: 0,
            max_steps,
        }
    }

    #[doc(hidden)]
    /// Test-only state access. NOT part of the stable API.
    /// Do not call from production code.
    pub fn set_state(&mut self, theta: f32, theta_dot: f32) {
        self.state = [theta, theta_dot];
    }

    #[doc(hidden)]
    /// Test-only state access. NOT part of the stable API.
    /// Do not call from production code.
    pub fn get_state(&self) -> [f32; 2] {
        self.state
    }

    /// Check if the current state contains NaN or Inf values.
    #[inline]
    fn is_state_poisoned(&self) -> bool {
        self.state.iter().any(|x| !x.is_finite())
    }

    /// Returns a safe default observation corresponding to theta=0, theta_dot=0.
    #[inline]
    fn safe_default_obs() -> [f32; 3] {
        [1.0, 0.0, 0.0]
    }
}

impl Default for Pendulum {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize an angle `x` to `[-pi, pi]`.
pub fn angle_normalize(x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let two_pi = 2.0 * pi;
    ((x + pi).rem_euclid(two_pi)) - pi
}

/// Compute the reward for a given state and action.
pub fn compute_reward(theta: f32, theta_dot: f32, u: f32) -> f32 {
    -(angle_normalize(theta).powi(2) + 0.1 * theta_dot.powi(2) + 0.001 * u.powi(2))
}

impl Environment for Pendulum {
    type Obs = [f32; 3];
    type Act = PendulumAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        // Initialize state randomly:
        // theta ~ Uniform(-pi, pi)
        // theta_dot ~ Uniform(-1, 1)
        self.state = [
            self.rng.gen_range(-RESET_THETA_BOUND..RESET_THETA_BOUND),
            self.rng
                .gen_range(-RESET_THETA_DOT_BOUND..RESET_THETA_DOT_BOUND),
        ];
        self.steps = 0;

        let [theta, theta_dot] = self.state;
        ([theta.cos(), theta.sin(), theta_dot], ())
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        let u = action.0;
        let u = if u.is_nan() {
            tracing::warn!("Pendulum received NaN action. Defaulting to 0.0.");
            0.0
        } else {
            u.clamp(-MAX_TORQUE, MAX_TORQUE)
        };

        let [theta, theta_dot] = self.state;

        // Compute reward using pre-step state
        let reward = compute_reward(theta, theta_dot, u);

        // Physics update (Gymnasium v1 semi-implicit Euler integration)
        let new_theta_dot =
            theta_dot + (3.0 * G / (2.0 * L) * theta.sin() + 3.0 / (M * L * L) * u) * DT;
        let new_theta_dot = new_theta_dot.clamp(-MAX_SPEED, MAX_SPEED);
        let new_theta = theta + new_theta_dot * DT;

        self.state = [new_theta, new_theta_dot];
        self.steps += 1;

        // NaN defense
        if self.is_state_poisoned() {
            return (Self::safe_default_obs(), 0.0, true, false, ());
        }

        let terminated = false;
        let truncated = self.steps >= self.max_steps;

        let obs = [new_theta.cos(), new_theta.sin(), new_theta_dot];
        (obs, reward, terminated, truncated, ())
    }

    fn action_space(&self) -> Space {
        Space::continuous(vec![-MAX_TORQUE], vec![MAX_TORQUE])
    }

    fn observation_space(&self) -> Space {
        Space::continuous(vec![-1.0, -1.0, -MAX_SPEED], vec![1.0, 1.0, MAX_SPEED])
    }
}
