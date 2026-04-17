//! RL Environment module — environments, action/observation spaces, and wrappers.
//!
//! ## Architecture
//!
//! ```text
//! env/
//! ├── traits.rs        (Environment trait, IntoTensorBuffer bridge)
//! ├── spaces.rs        (Space enum: Discrete, Box, MultiDiscrete)
//! ├── cartpole.rs      (CartPole-v1 classic control)
//! ├── gridworld.rs     (Discrete 2D grid maze)
//! ├── wrappers.rs      (TimeLimit, RewardScale — zero-cost generic wrappers)
//! └── vector.rs        (SyncVectorEnv — batched environment with pre-allocated buffers)
//! ```

pub mod cartpole;
pub mod gridworld;
pub mod spaces;
pub mod traits;
pub mod vector;
pub mod wrappers;

// Re-export core types
pub use cartpole::{CartPole, CartPoleAction};
pub use gridworld::{GridAction, GridWorld};
pub use spaces::Space;
pub use traits::{Environment, IntoTensorBuffer};
pub use vector::{BatchStepResult, SyncVectorEnv};
pub use wrappers::{RewardScale, TimeLimit};
