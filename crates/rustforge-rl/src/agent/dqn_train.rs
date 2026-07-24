//! Backward-compatible headless DQN training entry point.

use crate::agent::dqn_runtime::train_dqn_headless;
use crate::agent::{DQNConfig, DQN};
use crate::env::Environment;
use std::convert::TryFrom;
use std::fmt::Debug;

/// Generic DQN training loop for discrete action environments.
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
    train_dqn_headless(env, config, episodes, max_steps_per_episode, log_path)
}
