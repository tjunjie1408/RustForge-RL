//! Agent module — RL algorithm implementations.
//!
//! Provides exploration strategies, learning algorithms (DQN, REINFORCE, A2C,
//! PPO, TD3, SAC), and shared utilities (returns, LR scheduling, Gaussian policy).

pub mod a2c;
mod a2c_backend;
mod a2c_runtime;
pub mod dqn;
mod dqn_backend;
mod dqn_runtime;
pub mod dqn_train;
pub mod epsilon_greedy;
pub mod gaussian_policy;
mod live_runtime;
mod on_policy_runtime;
pub mod ppo;
mod ppo_backend;
mod ppo_continuous_backend;
mod ppo_continuous_runtime;
mod ppo_runtime;
pub mod reinforce;
mod reinforce_backend;
mod reinforce_runtime;
pub mod returns;
pub mod sac;
pub mod schedule;
pub mod td3;
mod td3_backend;
mod td3_runtime;
pub mod utils;

pub use a2c::{A2CConfig, ActorCriticNet, A2C};
pub use a2c_backend::{A2cDevice, A2cRuntimeOptions};
pub use a2c_runtime::{cartpole_a2c_config, A2cTrainerAdapter};
pub use dqn::{DQNConfig, DQN};
pub use dqn_backend::{DqnDevice, DqnRuntimeOptions};
pub use dqn_runtime::DqnTrainerAdapter;
pub use dqn_train::{train_dqn, try_train_dqn};
pub use epsilon_greedy::EpsilonGreedy;
pub use gaussian_policy::{GaussianPolicy, GaussianPolicyNet};
pub use ppo::{PPOConfig, PPOContinuous, PPOContinuousConfig, PPODiscrete, PPODiscreteConfig};
pub use ppo_backend::{PpoDevice, PpoRuntimeOptions};
pub use ppo_continuous_runtime::{pendulum_ppo_config, PpoContinuousTrainerAdapter};
pub use ppo_runtime::{cartpole_ppo_config, PpoDiscreteTrainerAdapter};
pub use reinforce::{REINFORCEConfig, REINFORCE};
pub use reinforce_backend::{ReinforceDevice, ReinforceRuntimeOptions};
pub use reinforce_runtime::{cartpole_reinforce_config, ReinforceTrainerAdapter};
pub use returns::{compute_discounted_returns, compute_gae};
pub use sac::{SACConfig, SAC};
pub use schedule::LRSchedule;
pub use td3::{TD3Config, TD3};
pub use td3_backend::{Td3Device, Td3RuntimeOptions};
pub use td3_runtime::{pendulum_td3_config, Td3TrainerAdapter};
pub use utils::{clamp_var, elementwise_min_var, hard_update, soft_update};

#[cfg(feature = "gpu")]
pub mod gpu_dqn;
#[cfg(feature = "gpu")]
pub use gpu_dqn::{GpuDqn, GpuDqnBatch, GpuDqnError};

#[cfg(feature = "gpu")]
pub mod gpu_ppo;

#[cfg(feature = "gpu")]
pub mod gpu_gaussian;

#[cfg(feature = "gpu")]
pub mod gpu_a2c;

#[cfg(feature = "gpu")]
pub mod gpu_reinforce;

#[cfg(feature = "gpu")]
pub mod gpu_td3;

#[cfg(feature = "gpu")]
pub mod gpu_sac;
