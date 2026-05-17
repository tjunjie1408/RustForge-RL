//! Agent module — RL algorithm implementations.
//!
//! Provides exploration strategies, learning algorithms (DQN, REINFORCE, A2C),
//! and shared utilities (returns computation, LR scheduling).

pub mod a2c;
pub mod dqn;
pub mod epsilon_greedy;
pub mod reinforce;
pub mod returns;
pub mod schedule;

pub use a2c::{A2CConfig, ActorCriticNet, A2C};
pub use dqn::{DQNConfig, DQN};
pub use epsilon_greedy::EpsilonGreedy;
pub use reinforce::{REINFORCEConfig, REINFORCE};
pub use returns::{compute_discounted_returns, compute_gae};
pub use schedule::LRSchedule;
