pub mod cli;
pub mod commands;

use cli::{Cli, Commands};

pub async fn dispatch(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Train(args) => commands::train::execute(args),
        Commands::Monitor(args) => commands::monitor::execute(args).await,
        Commands::Run(args) => commands::run::execute(args).await,
        Commands::Plan(args) => {
            println!("{}", commands::run::describe(args)?);
            Ok(())
        }
        Commands::ExportGraph => commands::export_graph(),
    }
}
