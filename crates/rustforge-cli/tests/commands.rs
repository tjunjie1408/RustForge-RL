use clap::Parser;
use rustforge_cli::cli::{Algorithm, Cli, Commands, Environment};

#[test]
fn final_command_surface_has_train_monitor_and_multi_algorithm_run() {
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

    let ppo_train = Cli::try_parse_from(["rustforge", "train", "ppo", "--episodes", "2"]).unwrap();
    assert!(
        matches!(ppo_train.command, Commands::Train(args) if args.algorithm == Algorithm::Ppo && args.output.is_none())
    );

    let ppo_run = Cli::try_parse_from(["rustforge", "run", "ppo", "--episodes", "2"]).unwrap();
    assert!(matches!(ppo_run.command, Commands::Run(args) if args.algorithm == Algorithm::Ppo));

    let a2c_train = Cli::try_parse_from(["rustforge", "train", "a2c", "--episodes", "2"]).unwrap();
    assert!(matches!(a2c_train.command, Commands::Train(args) if args.algorithm == Algorithm::A2c));

    let a2c_run = Cli::try_parse_from(["rustforge", "run", "a2c", "--episodes", "2"]).unwrap();
    assert!(matches!(a2c_run.command, Commands::Run(args) if args.algorithm == Algorithm::A2c));

    let reinforce_train =
        Cli::try_parse_from(["rustforge", "train", "reinforce", "--episodes", "2"]).unwrap();
    assert!(
        matches!(reinforce_train.command, Commands::Train(args) if args.algorithm == Algorithm::Reinforce)
    );

    let reinforce_run =
        Cli::try_parse_from(["rustforge", "run", "reinforce", "--episodes", "2"]).unwrap();
    assert!(
        matches!(reinforce_run.command, Commands::Run(args) if args.algorithm == Algorithm::Reinforce)
    );
}

#[test]
fn unsupported_algorithms_and_invalid_inputs_fail_before_execution() {
    assert!(Cli::try_parse_from(["rustforge", "run", "dqn", "--episodes", "0"]).is_err());
    assert!(Cli::try_parse_from(["rustforge", "run", "dqn", "--target-reward", "NaN"]).is_err());
    assert!(Cli::try_parse_from(["rustforge", "run", "dqn", "--overwrite"]).is_err());
    assert!(Cli::try_parse_from(["rustforge", "train", "dqn", "--overwrite"]).is_err());
    assert!(Cli::try_parse_from([
        "rustforge",
        "monitor",
        "metrics.csv",
        "--total-episodes",
        "0"
    ])
    .is_err());
}

#[test]
fn device_and_checkpoint_flags_parse_for_headless_and_live_commands() {
    use rustforge_cli::cli::Device;
    for command in ["train", "run"] {
        let parsed = Cli::try_parse_from([
            "rustforge",
            command,
            "dqn",
            "--device",
            "gpu",
            "--resume",
            "input.chk",
            "--checkpoint",
            "output.chk",
        ])
        .unwrap();
        let execution = match parsed.command {
            Commands::Train(args) => args.execution,
            Commands::Run(args) => args.execution,
            _ => unreachable!(),
        };
        assert_eq!(execution.device, Device::Gpu);
        assert_eq!(execution.resume.unwrap().to_str().unwrap(), "input.chk");
        assert_eq!(
            execution.checkpoint.unwrap().to_str().unwrap(),
            "output.chk"
        );
        let default = Cli::try_parse_from(["rustforge", command, "dqn"]).unwrap();
        match default.command {
            Commands::Train(args) => assert_eq!(args.execution.device, Device::Cpu),
            Commands::Run(args) => assert_eq!(args.execution.device, Device::Cpu),
            _ => unreachable!(),
        }
    }
    assert!(Cli::try_parse_from(["rustforge", "train", "dqn", "--device", "cuda"]).is_err());
}
