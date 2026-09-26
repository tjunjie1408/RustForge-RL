//! Backward-compatible headless DQN training entry point.

use crate::agent::dqn_runtime::train_dqn_headless;
use crate::agent::{DQNConfig, DQN};
use crate::env::Environment;
use crate::runtime::trainer::TrainerError;
use std::convert::TryFrom;
use std::fmt::Debug;

/// Generic DQN training loop for discrete action environments.
///
/// ## Panics
/// Panics if the CSV log cannot be created or training diverges (non-finite
/// Q-values). Use [`try_train_dqn`] to handle those cases as errors.
pub fn train_dqn<E>(
    env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    log_path: Option<&str>,
) -> DQN
where
    E: Environment,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    try_train_dqn(env, config, episodes, max_steps_per_episode, log_path)
        .unwrap_or_else(|error| panic!("DQN training failed: {error}"))
}

/// Fallible form of [`train_dqn`].
///
/// Returns an error instead of panicking when the CSV log cannot be created
/// or training diverges.
pub fn try_train_dqn<E>(
    env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    log_path: Option<&str>,
) -> Result<DQN, TrainerError>
where
    E: Environment,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    train_dqn_headless(env, config, episodes, max_steps_per_episode, log_path)
}
