//! `rustforge-dashboard` — live web dashboard for a RustForge training CSV.
use std::path::PathBuf;

use clap::Parser;

use rustforge_dashboard::server::router;
use rustforge_dashboard::state::{spawn_tail_task, AppState};

#[derive(Parser)]
#[command(name = "rustforge-dashboard", about = "Live web dashboard for RustForge training runs")]
struct Args {
    /// Path to the training CSV log (episode,reward,avg_loss,epsilon,global_step).
    #[arg(long)]
    log: PathBuf,
    /// Port to serve on.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Host/interface to bind.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let state = AppState::new(1024);
    spawn_tail_task(state.clone(), args.log.clone());

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;
    println!("RustForge dashboard: http://{addr}  (watching {})", args.log.display());

    axum::serve(listener, router(state)).await?;
    Ok(())
}
