//! Experience buffer modules for RL training.
//!
//! - `ReplayBuffer`: Off-policy uniform sampling (DQN). SoA layout, zero-alloc hot path.
//! - `RolloutBuffer`: On-policy trajectory collection (REINFORCE, A2C). Collect→Compute→Consume→Clear.
//! - `ContinuousReplayBuffer`: Off-policy for continuous actions (TD3, SAC).
//! - `ContinuousRolloutBuffer`: On-policy for continuous actions (PPO Continuous).

pub mod replay;
pub mod rollout;

pub use replay::{
    ContinuousReplayBuffer, ContinuousTransitionBatch, ReplayBuffer, TransitionBatch,
};
pub use rollout::{ContinuousRolloutBatch, ContinuousRolloutBuffer, RolloutBatch, RolloutBuffer};
