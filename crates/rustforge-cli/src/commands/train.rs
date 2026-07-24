use rustforge_rl::agent::{train_dqn, DQNConfig};
use rustforge_rl::env::{CartPole, GridWorld};

use crate::cli::{Algorithm, Environment, TrainArgs};

pub fn execute(args: TrainArgs) -> anyhow::Result<()> {
    let log_path = (!args.no_log).then(|| args.output.to_string_lossy().into_owned());
    let config = dqn_config(args.env, args.use_per);
    match (args.algorithm, args.env) {
        (Algorithm::Dqn, Environment::Cartpole) => {
            println!("Training DQN on CartPole for {} episodes...", args.episodes);
            train_dqn(
                CartPole::with_max_steps(500),
                config,
                args.episodes,
                500,
                log_path.as_deref(),
            );
        }
        (Algorithm::Dqn, Environment::Gridworld) => {
            println!(
                "Training DQN on GridWorld for {} episodes...",
                args.episodes
            );
            train_dqn(
                GridWorld::new(),
                config,
                args.episodes,
                100,
                log_path.as_deref(),
            );
        }
    }
    Ok(())
}

pub(crate) fn dqn_config(env: Environment, use_per: bool) -> DQNConfig {
    DQNConfig {
        obs_dim: match env {
            Environment::Cartpole => 4,
            Environment::Gridworld => 2,
        },
        num_actions: match env {
            Environment::Cartpole => 2,
            Environment::Gridworld => 4,
        },
        hidden_dim: 64,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 100,
        double_dqn: true,
        use_per,
        per_beta_annealing_steps: 20_000,
    }
}
