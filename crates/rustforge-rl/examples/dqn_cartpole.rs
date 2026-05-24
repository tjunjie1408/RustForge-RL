//! Minimal DQN + CartPole training loop.
//!
//! Run with:
//!
//! ```text
//! cargo run -p rustforge-rl --example dqn_cartpole
//! ```
//!
//! This example intentionally keeps the loop explicit. It shows how the strong
//! `CartPoleAction` enum at the environment boundary maps to the `usize` action
//! indices used by the Q-network and replay buffer.

use rustforge_rl::agent::{train_dqn, DQNConfig};
use rustforge_rl::env::CartPole;

fn main() {
    let env = CartPole::with_max_steps(500);
    let config = DQNConfig {
        obs_dim: 4,
        num_actions: 2,
        hidden_dim: 64,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 100,
        double_dqn: true,
    };

    train_dqn(env, config, 50, 500, Some("dqn_cartpole_metrics.csv"));
}
