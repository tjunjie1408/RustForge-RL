//! RustForge RL — Reinforcement Learning Algorithms (Phase 3-4 Implementation)
//!
//! ## Architecture
//!
//! ```text
//! rustforge-rl
//! ├── env/               (Environment trait, spaces, concrete envs, wrappers, vectorization)
//! │   ├── traits.rs      (Environment trait, IntoTensorBuffer bridge)
//! │   ├── spaces.rs      (Space enum: Discrete, Box, MultiDiscrete)
//! │   ├── cartpole.rs    (CartPole-v1 classic control)
//! │   ├── gridworld.rs   (Discrete 2D grid maze)
//! │   ├── wrappers.rs    (TimeLimit, RewardScale — zero-cost generic wrappers)
//! │   └── vector.rs      (SyncVectorEnv — batched env with pre-allocated buffers)
//! └── (future: agents, buffers, trainer)
//! ```

pub mod env;
