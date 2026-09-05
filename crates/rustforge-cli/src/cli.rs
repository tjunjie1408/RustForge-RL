use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "rustforge",
    version,
    about = "RustForge native RL training console"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Train an agent without an interactive terminal.
    Train(TrainArgs),
    /// Inspect a completed or actively written DQN CSV v1 file.
    Monitor(MonitorArgs),
    /// Train an agent with the native live terminal console.
    Run(RunArgs),
    /// Export a DQN computation graph as Graphviz DOT.
    ExportGraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Algorithm {
    Dqn,
    Ppo,
    A2c,
    Reinforce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Environment {
    Cartpole,
    Gridworld,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(value_enum)]
    pub algorithm: Algorithm,
    #[arg(long, value_enum, default_value_t = Environment::Cartpole)]
    pub env: Environment,
    #[arg(long, default_value_t = 100, value_parser = parse_positive_usize)]
    pub episodes: usize,
    #[arg(long)]
    pub no_log: bool,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long, requires = "output")]
    pub overwrite: bool,
    #[arg(long)]
    pub use_per: bool,
}

#[derive(Debug, Args)]
pub struct MonitorArgs {
    /// DQN CSV v1 file to load and follow.
    pub metrics: PathBuf,
    #[arg(long)]
    pub no_color: bool,
    #[arg(long)]
    pub ascii: bool,
    #[arg(long, value_parser = parse_finite_f64)]
    pub target_reward: Option<f64>,
    #[arg(long, value_parser = parse_positive_u64)]
    pub total_episodes: Option<u64>,
}

#[derive(Debug, Args)]
pub struct RunArgs {
    #[arg(value_enum)]
    pub algorithm: Algorithm,
    #[arg(long, value_enum, default_value_t = Environment::Cartpole)]
    pub env: Environment,
    #[arg(long, default_value_t = 100, value_parser = parse_positive_usize)]
    pub episodes: usize,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long, requires = "output")]
    pub overwrite: bool,
    #[arg(long)]
    pub use_per: bool,
    #[arg(long)]
    pub no_color: bool,
    #[arg(long)]
    pub ascii: bool,
    #[arg(long, value_parser = parse_finite_f64)]
    pub target_reward: Option<f64>,
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| "must be a positive integer".to_owned())
        .and_then(|value| {
            (value > 0)
                .then_some(value)
                .ok_or_else(|| "must be greater than zero".to_owned())
        })
}

fn parse_positive_u64(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| "must be a positive integer".to_owned())
        .and_then(|value| {
            (value > 0)
                .then_some(value)
                .ok_or_else(|| "must be greater than zero".to_owned())
        })
}

fn parse_finite_f64(value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| "must be a number".to_owned())
        .and_then(|value| {
            value
                .is_finite()
                .then_some(value)
                .ok_or_else(|| "must be finite".to_owned())
        })
}
