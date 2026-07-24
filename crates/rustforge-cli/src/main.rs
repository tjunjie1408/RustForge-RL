use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rustforge_cli::dispatch(rustforge_cli::cli::Cli::parse()).await
}
