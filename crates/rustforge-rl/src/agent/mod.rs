//! Agent module — RL algorithm implementations.
//!
//! Provides exploration strategies, learning algorithms (DQN, REINFORCE, A2C,
//! PPO, TD3, SAC), and shared utilities (returns, LR scheduling, Gaussian policy).

pub mod a2c;
mod a2c_runtime;
pub mod dqn;
mod dqn_runtime;
pub mod dqn_train;
pub mod epsilon_greedy;
pub mod gaussian_policy;
mod on_policy_runtime;
pub mod ppo;
mod ppo_runtime;
pub mod reinforce;
mod reinforce_runtime;
pub mod returns;
pub mod sac;
pub mod schedule;
pub mod td3;
pub mod utils;

pub use a2c::{A2CConfig, ActorCriticNet, A2C};
pub use a2c_runtime::{cartpole_a2c_config, A2cTrainerAdapter};
pub use dqn::{DQNConfig, DQN};
pub use dqn_runtime::DqnTrainerAdapter;
pub use dqn_train::train_dqn;
pub use epsilon_greedy::EpsilonGreedy;
pub use gaussian_policy::{GaussianPolicy, GaussianPolicyNet};
pub use ppo::{PPOConfig, PPOContinuous, PPOContinuousConfig, PPODiscrete, PPODiscreteConfig};
pub use ppo_runtime::{cartpole_ppo_config, PpoDiscreteTrainerAdapter};
pub use reinforce::{REINFORCEConfig, REINFORCE};
pub use reinforce_runtime::{cartpole_reinforce_config, ReinforceTrainerAdapter};
pub use returns::{compute_discounted_returns, compute_gae};
pub use sac::{SACConfig, SAC};
pub use schedule::LRSchedule;
pub use td3::{TD3Config, TD3};
pub use utils::{clamp_var, elementwise_min_var, hard_update, soft_update};
