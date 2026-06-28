//! `rustforge-dashboard` — live web dashboard for a RustForge training CSV.
//!
//! Watch an existing run, or with `--train` start one and watch it in a single
//! command. Auto-opens the browser unless `--no-open` is given.
use std::path::PathBuf;

use clap::Parser;

use rustforge_dashboard::launch::{browser_url, open_browser, spawn_trainer, Algo, Env};
use rustforge_dashboard::server::router;
use rustforge_dashboard::state::{spawn_tail_task, AppState};

#[derive(Parser)]
#[command(
    name = "rustforge-dashboard",
    about = "Live web dashboard for RustForge training runs"
)]
struct Args {
    /// Training CSV to watch (also the trainer's --output in --train mode).
    #[arg(long, default_value = "target/cli_train_dqn.csv")]
    log: PathBuf,
    /// Train first, then watch: spawn `rustforge-cli train <ALGO>`. Bare `--train` => dqn.
    #[arg(long, value_enum, num_args = 0..=1, default_missing_value = "dqn")]
    train: Option<Algo>,
    /// Episodes to train (only used with --train).
    #[arg(long, default_value_t = 200)]
    episodes: usize,
    /// Environment to train on (only used with --train).
    #[arg(long, value_enum, default_value = "cartpole")]
    env: Env,
    /// Do not auto-open the browser on startup.
    #[arg(long)]
    no_open: bool,
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

    // Bind FIRST: a bind failure (e.g. port already in use) must error out
    // BEFORE we spawn a trainer that would otherwise be orphaned.
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;
    println!(
        "RustForge dashboard: http://{addr}  (watching {})",
        args.log.display()
    );

    // Port is secured — now optionally spawn the trainer.
    let mut trainer = match args.train {
        Some(algo) => {
            println!(
                "Starting trainer: {} on {} for {} episodes -> {}",
                algo.as_arg(),
                args.env.as_arg(),
                args.episodes,
                args.log.display()
            );
            Some(
                spawn_trainer(algo, args.env, args.episodes, &args.log)
                    .map_err(|e| anyhow::anyhow!("failed to start trainer (rustforge-cli): {e}"))?,
            )
        }
        None => None,
    };

    if !args.no_open {
        open_browser(&browser_url(&args.host, args.port));
    }

    // Serve until Ctrl+C; reap the trainer on EVERY exit path (success or error).
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let serve_result = axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await;
    if let Some(child) = trainer.as_mut() {
        let _ = child.kill();
    }
    serve_result?;
    Ok(())
}
