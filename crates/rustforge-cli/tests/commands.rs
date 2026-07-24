use clap::Parser;
use rustforge_cli::cli::{Algorithm, Cli, Commands, Environment};

#[test]
fn final_command_surface_has_train_monitor_and_dqn_run() {
    let train = Cli::try_parse_from([
        "rustforge",
        "train",
        "dqn",
        "--env",
        "gridworld",
        "--episodes",
        "2",
    ])
    .unwrap();
    assert!(
        matches!(train.command, Commands::Train(args) if args.algorithm == Algorithm::Dqn && args.env == Environment::Gridworld)
    );

    let monitor = Cli::try_parse_from(["rustforge", "monitor", "metrics.csv"]).unwrap();
    assert!(matches!(monitor.command, Commands::Monitor(_)));

    let run = Cli::try_parse_from(["rustforge", "run", "dqn", "--episodes", "2"]).unwrap();
    assert!(matches!(run.command, Commands::Run(args) if args.algorithm == Algorithm::Dqn));
}

#[test]
fn unsupported_algorithms_and_invalid_inputs_fail_before_execution() {
    assert!(Cli::try_parse_from(["rustforge", "run", "ppo"]).is_err());
    assert!(Cli::try_parse_from(["rustforge", "run", "dqn", "--episodes", "0"]).is_err());
    assert!(Cli::try_parse_from(["rustforge", "run", "dqn", "--target-reward", "NaN"]).is_err());
    assert!(Cli::try_parse_from(["rustforge", "run", "dqn", "--overwrite"]).is_err());
    assert!(Cli::try_parse_from([
        "rustforge",
        "monitor",
        "metrics.csv",
        "--total-episodes",
        "0"
    ])
    .is_err());
}
