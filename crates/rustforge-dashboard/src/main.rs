//! Transitional `rustforge-dashboard` binary: read-only Ratatui CSV monitor.

use std::path::PathBuf;

use clap::Parser;
use rustforge_dashboard::monitor::{run_monitor, MonitorOptions};

#[derive(Parser)]
#[command(
    name = "rustforge-dashboard",
    about = "Native terminal monitor for RustForge training metrics"
)]
struct Args {
    /// DQN CSV v1 metrics file to load and follow.
    #[arg(long, default_value = "target/cli_train_dqn.csv")]
    log: PathBuf,

    /// Disable semantic colors (also enabled by the NO_COLOR environment variable).
    #[arg(long)]
    no_color: bool,

    /// Use ASCII borders and markers instead of Unicode symbols.
    #[arg(long)]
    ascii: bool,

    /// Optional finite reward threshold used by monitor alerts.
    #[arg(long)]
    target_reward: Option<f64>,

    /// Optional trusted total episode count for progress and ETA.
    #[arg(long)]
    total_episodes: Option<u64>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_monitor(MonitorOptions {
        metrics_path: args.log,
        no_color: args.no_color || std::env::var_os("NO_COLOR").is_some(),
        ascii: args.ascii,
        target_reward: args.target_reward,
        total_episodes: args.total_episodes,
    })
    .await
}
