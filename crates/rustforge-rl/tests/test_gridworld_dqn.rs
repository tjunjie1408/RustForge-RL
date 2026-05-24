use rustforge_rl::agent::{train_dqn, DQNConfig};
use rustforge_rl::env::GridWorld;

#[test]
fn test_gridworld_dqn_smoke() {
    let env = GridWorld::new();
    let config = DQNConfig {
        obs_dim: 2,
        num_actions: 4,
        hidden_dim: 32,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 10,
        double_dqn: false,
        use_per: false,
        per_beta_annealing_steps: 20000,
    };

    // Train for 2 episodes, max 10 steps each, no logging
    let _agent = train_dqn(env, config, 2, 10, None);
}
