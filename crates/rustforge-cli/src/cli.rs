use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "rustforge",
    version,
    about = "Train, inspect and monitor RustForge RL experiments",
    after_help = "Examples:\n  rustforge train sac -n 10\n  rustforge run ppo -e pendulum\n  rustforge plan sac\n\nDefault environments: Pendulum for SAC/TD3; CartPole otherwise."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Train an agent without an interactive terminal.
    #[command(visible_alias = "fit")]
    Train(TrainArgs),
    /// Inspect a completed or actively written metrics CSV file.
    #[command(visible_alias = "watch")]
    Monitor(MonitorArgs),
    /// Train an agent with the native live terminal console.
    #[command(visible_alias = "live")]
    Run(RunArgs),
    /// Print a validated JSON training plan without starting training or writing files.
    Plan(PlanArgs),
    /// Export a DQN computation graph as Graphviz DOT.
    ExportGraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Algorithm {
    Dqn,
    Ppo,
    A2c,
    Reinforce,
    Td3,
    Sac,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Environment {
    /// Choose Pendulum for SAC/TD3, CartPole for the other algorithms.
    Auto,
    #[value(alias = "cart-pole", alias = "CartPole-v1")]
    Cartpole,
    #[value(alias = "grid-world")]
    Gridworld,
    #[value(alias = "Pendulum-v1")]
    Pendulum,
}

impl Environment {
    pub fn resolve(self, algorithm: Algorithm) -> Self {
        match self {
            Self::Auto => match algorithm {
                Algorithm::Sac | Algorithm::Td3 => Self::Pendulum,
                _ => Self::Cartpole,
            },
            environment => environment,
        }
    }
}

/// Training backend requested explicitly by the user.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
}

impl Device {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU (wgpu)",
        }
    }
}

#[derive(Clone, Debug, Default, Args)]
pub struct ExecutionArgs {
    /// GPU supports DQN, PPO, A2C, REINFORCE, TD3 and SAC; requires the gpu feature.
    #[arg(short = 'd', long, value_enum, ignore_case = true, default_value_t = Device::Cpu)]
    pub device: Device,
    /// Restore GPU training state; environment, rollout/replay and random streams restart.
    #[arg(long)]
    pub resume: Option<PathBuf>,
    /// Atomically save GPU training state on completion or controlled stop; replaces this file.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
}

impl ExecutionArgs {
    pub(crate) fn runtime_options(
        &self,
        algorithm: Algorithm,
        use_per: bool,
    ) -> anyhow::Result<rustforge_rl::agent::DqnRuntimeOptions> {
        let options = rustforge_rl::agent::DqnRuntimeOptions {
            device: match self.device {
                Device::Cpu => rustforge_rl::agent::DqnDevice::Cpu,
                Device::Gpu => rustforge_rl::agent::DqnDevice::Gpu,
            },
            resume: self.resume.clone(),
            checkpoint: self.checkpoint.clone(),
        };
        if algorithm == Algorithm::Ppo {
            rustforge_rl::agent::PpoRuntimeOptions::from(options.clone()).validate()
        } else if algorithm == Algorithm::A2c {
            rustforge_rl::agent::A2cRuntimeOptions::from(options.clone()).validate()
        } else if algorithm == Algorithm::Reinforce {
            rustforge_rl::agent::ReinforceRuntimeOptions::from(options.clone()).validate()
        } else if algorithm == Algorithm::Td3 {
            rustforge_rl::agent::Td3RuntimeOptions::from(options.clone()).validate()
        } else if algorithm == Algorithm::Sac {
            rustforge_rl::agent::SacRuntimeOptions::from(options.clone()).validate()
        } else {
            options.validate(use_per)
        }
        .map_err(|error| anyhow::anyhow!(error))?;
        Ok(options)
    }
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[command(flatten)]
    pub execution: ExecutionArgs,
    #[arg(value_enum, ignore_case = true)]
    pub algorithm: Algorithm,
    #[arg(short = 'e', long, value_enum, ignore_case = true, default_value_t = Environment::Auto)]
    pub env: Environment,
    #[arg(short = 'n', long, default_value_t = 100, value_parser = parse_positive_usize)]
    pub episodes: usize,
    #[arg(long)]
    pub no_log: bool,
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,
    #[arg(long, requires = "output")]
    pub overwrite: bool,
    #[arg(long)]
    pub use_per: bool,
}

#[derive(Debug, Args)]
pub struct MonitorArgs {
    /// Metrics CSV to load and follow: RustForge DQN CSV v1, or a
    /// Stable-Baselines3 monitor.csv or progress.csv (detected from the header).
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
    #[command(flatten)]
    pub execution: ExecutionArgs,
    #[arg(value_enum, ignore_case = true)]
    pub algorithm: Algorithm,
    #[arg(short = 'e', long, value_enum, ignore_case = true, default_value_t = Environment::Auto)]
    pub env: Environment,
    #[arg(short = 'n', long, default_value_t = 100, value_parser = parse_positive_usize)]
    pub episodes: usize,
    #[arg(short = 'o', long)]
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

/// Noninteractive discovery for scripts and coding agents; stdout is JSON only.
#[derive(Debug, Args)]
pub struct PlanArgs {
    #[command(flatten)]
    pub execution: ExecutionArgs,
    #[arg(value_enum, ignore_case = true)]
    pub algorithm: Algorithm,
    #[arg(short = 'e', long, value_enum, ignore_case = true, default_value_t = Environment::Auto)]
    pub env: Environment,
    #[arg(short = 'n', long, default_value_t = 100, value_parser = parse_positive_usize)]
    pub episodes: usize,
    #[arg(long)]
    pub use_per: bool,
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
