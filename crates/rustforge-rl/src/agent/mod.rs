//! Agent module — RL algorithm implementations.
//!
//! Provides exploration strategies and learning algorithms (DQN, etc.).

pub mod dqn;
pub mod epsilon_greedy;

pub use dqn::{DQNConfig, DQN};
pub use epsilon_greedy::EpsilonGreedy;
