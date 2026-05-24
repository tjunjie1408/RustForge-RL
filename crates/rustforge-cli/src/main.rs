use clap::{Parser, Subcommand, ValueEnum};
use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_rl::agent::{train_dqn, DQNConfig};
use rustforge_rl::env::{CartPole, GridWorld};
use rustforge_tensor::Tensor;

#[derive(Parser)]
#[command(
    name = "rustforge-cli",
    version,
    about = "RustForge Command Line Interface"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train an RL agent
    Train {
        /// The algorithm to train
        #[arg(value_enum)]
        algo: Algo,

        /// The environment to train on
        #[arg(long, default_value = "cartpole")]
        env: EnvType,

        /// Number of episodes to train
        #[arg(long, default_value_t = 500)]
        episodes: usize,

        /// Disable logging
        #[arg(long)]
        no_log: bool,

        /// Custom path for the output CSV log file
        #[arg(long, default_value = "target/cli_train_dqn.csv")]
        output: String,
    },
    /// Export the computation graph to Graphviz DOT format
    ExportGraph,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Algo {
    Dqn,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum EnvType {
    Cartpole,
    Gridworld,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            algo,
            env,
            episodes,
            no_log,
            output,
        } => {
            let log_path = if no_log { None } else { Some(output.as_str()) };

            match algo {
                Algo::Dqn => {
                    let config = DQNConfig {
                        obs_dim: match env {
                            EnvType::Cartpole => 4,
                            EnvType::Gridworld => 2,
                        },
                        num_actions: match env {
                            EnvType::Cartpole => 2,
                            EnvType::Gridworld => 4,
                        },
                        hidden_dim: 64,
                        lr: 1e-3,
                        gamma: 0.99,
                        target_update_freq: 100,
                        double_dqn: true,
                    };

                    match env {
                        EnvType::Cartpole => {
                            println!("Training DQN on CartPole for {} episodes...", episodes);
                            let cartpole = CartPole::with_max_steps(500);
                            train_dqn(cartpole, config, episodes, 500, log_path);
                        }
                        EnvType::Gridworld => {
                            println!("Training DQN on GridWorld for {} episodes...", episodes);
                            let gridworld = GridWorld::new();
                            train_dqn(gridworld, config, episodes, 100, log_path);
                        }
                    }
                }
            }
        }
        Commands::ExportGraph => {
            // Instantiate a real DQN model, run a forward pass on a dummy tensor, and call `.export_graphviz()`
            let config = DQNConfig {
                obs_dim: 4,
                num_actions: 2,
                hidden_dim: 64,
                lr: 1e-3,
                gamma: 0.99,
                target_update_freq: 100,
                double_dqn: true,
            };
            let agent = rustforge_rl::agent::DQN::new(config);
            let input = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[1, 4]);
            let output = agent.q_net().forward(&Variable::new(input, true));
            let dot = output.export_graphviz();
            println!("{}", dot);
        }
    }

    Ok(())
}
