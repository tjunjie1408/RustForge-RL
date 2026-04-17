//! Zero-cost generic environment wrappers.
//!
//! Wrappers implement the decorator pattern via generics (`<E: Environment>`)
//! to leverage Rust's monomorphization for zero dynamic dispatch overhead.
//!
//! Unlike `Box<dyn Environment>`, these wrappers produce specialized machine code
//! for each concrete environment type at compile time.

use super::spaces::Space;
use super::traits::Environment;

/// Truncates episodes after a maximum number of steps.
///
/// If the inner environment hasn't terminated naturally by `max_steps`,
/// the episode is truncated (returns `truncated = true`).
///
/// # Zero-Cost Abstraction
///
/// `TimeLimit<CartPole>` is monomorphized — the wrapper overhead is a single
/// counter increment and comparison per step, with no dynamic dispatch.
pub struct TimeLimit<E: Environment> {
    /// Wrapped environment
    env: E,
    /// Maximum steps before truncation
    max_steps: usize,
    /// Current step counter
    current_step: usize,
}

impl<E: Environment> TimeLimit<E> {
    /// Create a new TimeLimit wrapper.
    ///
    /// # Arguments
    /// - `env`: The environment to wrap.
    /// - `max_steps`: Maximum steps before truncation.
    pub fn new(env: E, max_steps: usize) -> Self {
        TimeLimit {
            env,
            max_steps,
            current_step: 0,
        }
    }

    /// Get a reference to the inner environment.
    pub fn inner(&self) -> &E {
        &self.env
    }

    /// Get a mutable reference to the inner environment.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.env
    }

    /// Get the current step count.
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

impl<E: Environment> Environment for TimeLimit<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type Info = E::Info;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.current_step = 0;
        self.env.reset(seed)
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        let (obs, reward, terminated, truncated, info) = self.env.step(action);
        self.current_step += 1;

        // If the inner environment already terminated, respect that
        if terminated {
            return (obs, reward, terminated, truncated, info);
        }

        // Check time limit truncation
        let truncated = truncated || self.current_step >= self.max_steps;
        (obs, reward, terminated, truncated, info)
    }

    fn action_space(&self) -> Space {
        self.env.action_space()
    }

    fn observation_space(&self) -> Space {
        self.env.observation_space()
    }
}

/// Scales rewards by a constant factor.
///
/// Useful for normalizing reward magnitudes across different environments
/// or for reward shaping experiments.
///
/// # Zero-Cost Abstraction
///
/// `RewardScale<CartPole>` is monomorphized — the wrapper overhead is a single
/// f32 multiplication per step.
pub struct RewardScale<E: Environment> {
    /// Wrapped environment
    env: E,
    /// Reward scaling factor
    scale: f32,
}

impl<E: Environment> RewardScale<E> {
    /// Create a new RewardScale wrapper.
    ///
    /// # Arguments
    /// - `env`: The environment to wrap.
    /// - `scale`: Multiplicative factor applied to all rewards.
    pub fn new(env: E, scale: f32) -> Self {
        RewardScale { env, scale }
    }

    /// Get a reference to the inner environment.
    pub fn inner(&self) -> &E {
        &self.env
    }

    /// Get the scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

impl<E: Environment> Environment for RewardScale<E> {
    type Obs = E::Obs;
    type Act = E::Act;
    type Info = E::Info;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        self.env.reset(seed)
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        let (obs, reward, terminated, truncated, info) = self.env.step(action);
        (obs, reward * self.scale, terminated, truncated, info)
    }

    fn action_space(&self) -> Space {
        self.env.action_space()
    }

    fn observation_space(&self) -> Space {
        self.env.observation_space()
    }
}
