//! Agent module — RL algorithm implementations.
//!
//! Provides exploration strategies, learning algorithms (DQN, REINFORCE, A2C,
//! PPO, TD3, SAC), and shared utilities (returns, LR scheduling, Gaussian policy).

pub mod a2c;
pub mod dqn;
pub mod epsilon_greedy;
pub mod gaussian_policy;
pub mod ppo;
pub mod reinforce;
pub mod returns;
pub mod sac;
pub mod schedule;
pub mod td3;
pub mod utils;

pub use a2c::{A2CConfig, ActorCriticNet, A2C};
pub use dqn::{DQNConfig, DQN};
pub use epsilon_greedy::EpsilonGreedy;
pub use gaussian_policy::{GaussianPolicy, GaussianPolicyNet};
pub use ppo::{PPOConfig, PPOContinuous, PPOContinuousConfig, PPODiscrete, PPODiscreteConfig};
pub use reinforce::{REINFORCEConfig, REINFORCE};
pub use returns::{compute_discounted_returns, compute_gae};
pub use sac::{SACConfig, SAC};
pub use schedule::LRSchedule;
pub use td3::{TD3Config, TD3};
pub use utils::{clamp_var, elementwise_min_var, hard_update, soft_update};
