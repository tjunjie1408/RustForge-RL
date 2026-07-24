use std::path::PathBuf;

use rustforge_cli::cli::{Algorithm, Environment, RunArgs};

#[tokio::test]
async fn non_tty_run_fails_before_creating_output_or_starting_training() {
    let output = std::env::temp_dir().join(format!("rustforge-nontty-{}", std::process::id()));
    if output.exists() {
        std::fs::remove_dir_all(&output).unwrap();
    }
    let result = rustforge_cli::commands::run::execute(RunArgs {
        algorithm: Algorithm::Dqn,
        env: Environment::Cartpole,
        episodes: 1,
        output: Some(PathBuf::from(&output)),
        overwrite: false,
        use_per: false,
        no_color: true,
        ascii: true,
        target_reward: None,
    })
    .await;
    assert!(result.is_err());
    assert!(!output.exists());
}
