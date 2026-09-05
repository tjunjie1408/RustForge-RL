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

#[tokio::test]
async fn invalid_ppo_combinations_fail_before_terminal_or_filesystem_side_effects() {
    let cases = [
        (Environment::Gridworld, false, "PPO supports only CartPole"),
        (
            Environment::Cartpole,
            true,
            "--use-per is supported only by DQN",
        ),
    ];

    for (index, (env, use_per, expected)) in cases.into_iter().enumerate() {
        let output = std::env::temp_dir().join(format!(
            "rustforge-invalid-ppo-{}-{index}",
            std::process::id()
        ));
        if output.exists() {
            std::fs::remove_dir_all(&output).unwrap();
        }
        let error = rustforge_cli::commands::run::execute(RunArgs {
            algorithm: Algorithm::Ppo,
            env,
            episodes: 1,
            output: Some(output.clone()),
            overwrite: false,
            use_per,
            no_color: true,
            ascii: true,
            target_reward: None,
        })
        .await
        .unwrap_err();

        assert!(error.to_string().contains(expected), "{error:#}");
        assert!(!output.exists());
    }
}

#[tokio::test]
async fn valid_ppo_reaches_terminal_preflight_without_creating_artifacts() {
    let output =
        std::env::temp_dir().join(format!("rustforge-valid-ppo-nontty-{}", std::process::id()));
    if output.exists() {
        std::fs::remove_dir_all(&output).unwrap();
    }

    let error = rustforge_cli::commands::run::execute(RunArgs {
        algorithm: Algorithm::Ppo,
        env: Environment::Cartpole,
        episodes: 1,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
        no_color: true,
        ascii: true,
        target_reward: None,
    })
    .await
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("rustforge run requires an interactive terminal"),
        "{error:#}"
    );
    assert!(!output.exists());
}

#[tokio::test]
async fn invalid_reinforce_combinations_fail_before_terminal_or_filesystem_side_effects() {
    let cases = [
        (
            Environment::Gridworld,
            false,
            "REINFORCE supports only CartPole",
        ),
        (
            Environment::Cartpole,
            true,
            "--use-per is supported only by DQN",
        ),
    ];

    for (index, (env, use_per, expected)) in cases.into_iter().enumerate() {
        let output = std::env::temp_dir().join(format!(
            "rustforge-invalid-reinforce-{}-{index}",
            std::process::id()
        ));
        if output.exists() {
            std::fs::remove_dir_all(&output).unwrap();
        }
        let error = rustforge_cli::commands::run::execute(RunArgs {
            algorithm: Algorithm::Reinforce,
            env,
            episodes: 1,
            output: Some(output.clone()),
            overwrite: false,
            use_per,
            no_color: true,
            ascii: true,
            target_reward: None,
        })
        .await
        .unwrap_err();

        assert!(error.to_string().contains(expected), "{error:#}");
        assert!(!output.exists());
    }
}
