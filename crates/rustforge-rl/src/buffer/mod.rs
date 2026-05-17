//! Experience buffer modules for RL training.
//!
//! - `ReplayBuffer`: Off-policy uniform sampling (DQN). SoA layout, zero-alloc hot path.
//! - `RolloutBuffer`: On-policy trajectory collection (REINFORCE, A2C). Collect→Compute→Consume→Clear.

pub mod replay;
pub mod rollout;

pub use replay::{ReplayBuffer, TransitionBatch};
pub use rollout::{RolloutBatch, RolloutBuffer};
