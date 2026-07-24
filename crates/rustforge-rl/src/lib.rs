//! RustForge RL — Reinforcement Learning Algorithms (Phase 3 Implementation)
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
//! ├── buffer/            (Experience buffers)
//! │   ├── replay.rs      (ReplayBuffer — off-policy, SoA layout, zero-alloc sample)
//! │   └── rollout.rs     (RolloutBuffer — on-policy, collect→compute→consume→clear)
//! ├── agent/             (RL algorithm implementations)
//! │   ├── epsilon_greedy.rs  (ε-greedy exploration with linear decay)
//! │   ├── dqn.rs             (DQN + Double DQN with target network)
//! │   ├── reinforce.rs       (REINFORCE — Monte Carlo policy gradient)
//! │   ├── a2c.rs             (A2C — Advantage Actor-Critic, shared trunk)
//! │   ├── returns.rs         (Discounted returns + GAE computation)
//! │   └── schedule.rs        (LR scheduling: Constant, LinearDecay, CosineAnnealing)
//! ├── metrics.rs         (AgentLogger trait, CsvLogger, NullLogger)
//! └── training.rs        (episode_done / replay_done helpers)
//! ```

pub mod agent;
pub mod buffer;
pub mod env;
pub mod metrics;
pub mod runtime;
pub mod training;
