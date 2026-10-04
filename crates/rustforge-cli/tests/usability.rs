use clap::Parser;
use rustforge_cli::cli::{Cli, Commands, Device, Environment};
#[test]
fn short_flags_aliases_and_algorithm_defaults_resolve_consistently() {
    for (command, algorithm, environment) in [
        ("fit", "SAC", Environment::Pendulum),
        ("train", "TD3", Environment::Pendulum),
        ("live", "PPO", Environment::Cartpole),
        ("run", "dqn", Environment::Cartpole),
    ] {
        let parsed =
            Cli::try_parse_from(["rustforge", command, algorithm, "-n", "3", "-d", "CPU"]).unwrap();
        let (a, e, n, d) = match parsed.command {
            Commands::Train(a) => (a.algorithm, a.env, a.episodes, a.execution.device),
            Commands::Run(a) => (a.algorithm, a.env, a.episodes, a.execution.device),
            _ => panic!("wrong command"),
        };
        assert_eq!(e.resolve(a), environment);
        assert_eq!(n, 3);
        assert_eq!(d, Device::Cpu);
    }
    let args = Cli::try_parse_from([
        "rustforge",
        "train",
        "ppo",
        "-e",
        "Pendulum-v1",
        "-o",
        "trial.jsonl",
    ])
    .unwrap();
    assert!(
        matches!(args.command,Commands::Train(a) if a.env==Environment::Pendulum && a.output.as_ref().unwrap().to_str()==Some("trial.jsonl"))
    );
}
#[test]
fn plan_json_reports_effective_defaults_without_a_terminal_or_output_side_effects() {
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_rustforge"))
        .args(["plan", "sac", "-n", "7"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(plan["schema"], "rustforge-training-plan-v1");
    assert_eq!(plan["environment"], "pendulum");
    assert_eq!(plan["episodes"], 7);
    assert_eq!(plan["device"], "cpu");
    assert_eq!(plan["metrics"].as_array().unwrap().len(), 8);
    assert_eq!(plan["configuration_source"], "built-in profile");
    assert!(plan["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["name"] == "temperature.alpha"));
}
#[test]
fn explicit_invalid_plan_combinations_fail_without_starting_training() {
    for args in [
        vec!["plan", "sac", "-e", "cartpole"],
        vec!["plan", "ppo", "--use-per"],
        vec!["plan", "dqn", "--checkpoint", "missing.chk"],
    ] {
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_rustforge"))
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
}
#[cfg(feature = "gpu")]
#[test]
fn gpu_resume_plan_is_read_only_and_does_not_claim_constructor_config_is_effective() {
    let parsed = Cli::try_parse_from([
        "rustforge",
        "plan",
        "sac",
        "-d",
        "gpu",
        "--resume",
        "nonexistent.chk",
    ])
    .unwrap();
    let args = match parsed.command {
        Commands::Plan(a) => a,
        _ => unreachable!(),
    };
    assert_eq!(args.algorithm, rustforge_cli::cli::Algorithm::Sac);
    let value: serde_json::Value =
        serde_json::from_str(&rustforge_cli::commands::run::describe(args).unwrap()).unwrap();
    assert_eq!(value["configuration_source"], "checkpoint");
    assert_eq!(
        value["configuration"]["Agent configuration"],
        "Restored from checkpoint"
    );
    assert_eq!(value["resume"], "nonexistent.chk");
}
