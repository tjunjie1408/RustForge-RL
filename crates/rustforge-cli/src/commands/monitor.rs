use rustforge_dashboard::monitor::{run_monitor, MonitorOptions};

use crate::cli::MonitorArgs;

pub async fn execute(args: MonitorArgs) -> anyhow::Result<()> {
    run_monitor(MonitorOptions {
        metrics_path: args.metrics,
        no_color: args.no_color || std::env::var_os("NO_COLOR").is_some(),
        ascii: args.ascii,
        target_reward: args.target_reward,
        total_episodes: args.total_episodes,
    })
    .await
}
