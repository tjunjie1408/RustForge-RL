//! Experience replay buffer module.
//!
//! Provides `ReplayBuffer` for DQN off-policy training with uniform random sampling.
//! Uses Structure-of-Arrays (SoA) layout for cache-friendly, zero-allocation hot paths.

pub mod replay;

pub use replay::{ReplayBuffer, TransitionBatch};
