use std::path::PathBuf;

use rustforge_cli::cli::{Algorithm, Environment, TrainArgs};

fn temporary_output(suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "rustforge-headless-ppo-{}-{suffix}.jsonl",
        std::process::id()
    ))
}

#[test]
fn headless_a2c_cartpole_writes_generic_jsonl_metrics() {
    let output = temporary_output("a2c-metrics");
    let _ = std::fs::remove_file(&output);

    rustforge_cli::commands::train::execute(TrainArgs {
        execution: Default::default(),
        algorithm: Algorithm::A2c,
        env: Environment::Cartpole,
        episodes: 1,
        no_log: false,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
    })
    .unwrap();

    let content = std::fs::read_to_string(&output).unwrap();
    let records: Vec<_> = content.lines().collect();
    assert_eq!(records.len(), 1);
    assert!(records[0].contains("\"reward.episode\":"));
    assert!(records[0].contains("\"loss.actor\":"));
    assert!(records[0].contains("\"policy.entropy\":"));
    std::fs::remove_file(output).unwrap();
}

#[test]
fn headless_reinforce_cartpole_writes_five_finite_generic_jsonl_metrics() {
    let output = temporary_output("reinforce-metrics");
    let _ = std::fs::remove_file(&output);

    rustforge_cli::commands::train::execute(TrainArgs {
        execution: Default::default(),
        algorithm: Algorithm::Reinforce,
        env: Environment::Cartpole,
        episodes: 2,
        no_log: false,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
    })
    .unwrap();

    let content = std::fs::read_to_string(&output).unwrap();
    let records: Vec<_> = content.lines().collect();
    assert_eq!(records.len(), 2);
    for record in records {
        let metrics = record
            .strip_prefix("{\"episode\":")
            .and_then(|record| record.split_once(",\"global_step\":"))
            .and_then(|(_, record)| record.split_once(",\"metrics\":{"))
            .map(|(_, metrics)| metrics.strip_suffix("}}").unwrap())
            .expect("generic JSONL record shape");
        let metrics: Vec<_> = metrics
            .split(',')
            .map(|entry| entry.split_once(':').expect("metric name and value"))
            .collect();
        assert_eq!(metrics.len(), 5);
        let mut names = metrics.iter().map(|(name, _)| *name).collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "\"loss.policy\"",
                "\"performance.steps_per_second\"",
                "\"reward.episode\"",
                "\"reward.moving_average\"",
                "\"rollout.size\"",
            ]
        );
        assert!(metrics
            .iter()
            .all(|(_, value)| value.parse::<f64>().is_ok_and(f64::is_finite)));
    }
    std::fs::remove_file(output).unwrap();
}

#[test]
fn invalid_reinforce_combinations_fail_before_creating_output() {
    for (suffix, env, use_per) in [
        ("reinforce-gridworld", Environment::Gridworld, false),
        ("reinforce-per", Environment::Cartpole, true),
    ] {
        let output = temporary_output(suffix);
        let _ = std::fs::remove_file(&output);
        let error = rustforge_cli::commands::train::execute(TrainArgs {
            execution: Default::default(),
            algorithm: Algorithm::Reinforce,
            env,
            episodes: 1,
            no_log: false,
            output: Some(output.clone()),
            overwrite: false,
            use_per,
        })
        .unwrap_err();
        assert!(error.to_string().contains(if use_per {
            "--use-per"
        } else {
            "REINFORCE supports only CartPole"
        }));
        assert!(!output.exists());
    }
}

#[test]
fn invalid_a2c_combinations_fail_before_creating_output() {
    for (suffix, env, use_per) in [
        ("a2c-gridworld", Environment::Gridworld, false),
        ("a2c-per", Environment::Cartpole, true),
    ] {
        let output = temporary_output(suffix);
        let _ = std::fs::remove_file(&output);
        let error = rustforge_cli::commands::train::execute(TrainArgs {
            execution: Default::default(),
            algorithm: Algorithm::A2c,
            env,
            episodes: 1,
            no_log: false,
            output: Some(output.clone()),
            overwrite: false,
            use_per,
        })
        .unwrap_err();
        assert!(error.to_string().contains(if use_per {
            "--use-per"
        } else {
            "only CartPole"
        }));
        assert!(!output.exists());
    }
}

#[test]
fn headless_ppo_cartpole_writes_generic_jsonl_metrics() {
    let output = temporary_output("metrics");
    if output.exists() {
        std::fs::remove_file(&output).unwrap();
    }

    rustforge_cli::commands::train::execute(TrainArgs {
        execution: Default::default(),
        algorithm: Algorithm::Ppo,
        env: Environment::Cartpole,
        episodes: 1,
        no_log: false,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
    })
    .unwrap();

    let lines = std::fs::read_to_string(&output).unwrap();
    let records: Vec<_> = lines.lines().collect();
    assert_eq!(records.len(), 1);
    assert!(records[0].starts_with("{\"episode\":0,\"global_step\":"));
    assert!(records[0].contains("\"metrics\":{"));
    assert!(records[0].contains("\"reward.episode\":"));
    assert!(records[0].contains("\"loss.policy\":"));

    std::fs::remove_file(output).unwrap();
}

#[test]
fn explicit_headless_output_is_not_overwritten_without_permission() {
    for algorithm in [
        Algorithm::Dqn,
        Algorithm::Ppo,
        Algorithm::A2c,
        Algorithm::Reinforce,
    ] {
        let output = temporary_output(match algorithm {
            Algorithm::Dqn => "existing-dqn",
            Algorithm::Ppo => "existing-ppo",
            Algorithm::A2c => "existing-a2c",
            Algorithm::Reinforce => "existing-reinforce",
            Algorithm::Td3 => "existing-td3",
            Algorithm::Sac => "existing-sac",
        });
        std::fs::write(&output, "keep me").unwrap();

        let error = rustforge_cli::commands::train::execute(TrainArgs {
            execution: Default::default(),
            algorithm,
            env: Environment::Cartpole,
            episodes: 1,
            no_log: false,
            output: Some(output.clone()),
            overwrite: false,
            use_per: false,
        })
        .unwrap_err();

        assert!(error.to_string().contains("already exists"), "{error:#}");
        assert_eq!(std::fs::read_to_string(&output).unwrap(), "keep me");
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn headless_dqn_preserves_the_exact_csv_v1_boundary() {
    let output = temporary_output("dqn-v1");
    if output.exists() {
        std::fs::remove_file(&output).unwrap();
    }

    rustforge_cli::commands::train::execute(TrainArgs {
        execution: Default::default(),
        algorithm: Algorithm::Dqn,
        env: Environment::Cartpole,
        episodes: 1,
        no_log: false,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
    })
    .unwrap();

    let content = std::fs::read_to_string(&output).unwrap();
    let lines: Vec<_> = content.lines().collect();
    assert_eq!(lines[0], "episode,reward,avg_loss,epsilon,global_step");
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1].split(',').count(), 5);

    std::fs::remove_file(output).unwrap();
}

#[test]
fn headless_continuous_ppo_pendulum_writes_generic_jsonl_metrics() {
    let output = temporary_output("continuous-metrics");
    let _ = std::fs::remove_file(&output);

    rustforge_cli::commands::train::execute(TrainArgs {
        execution: Default::default(),
        algorithm: Algorithm::Ppo,
        env: Environment::Pendulum,
        episodes: 1,
        no_log: false,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
    })
    .unwrap();

    let content = std::fs::read_to_string(&output).unwrap();
    let records: Vec<_> = content.lines().collect();
    assert_eq!(records.len(), 1);
    assert!(records[0].contains("\"reward.episode\":"));
    assert!(records[0].contains("\"loss.policy\":"));
    assert!(records[0].contains("\"loss.value\":"));
    std::fs::remove_file(output).unwrap();
}

#[test]
fn headless_cpu_td3_pendulum_writes_six_finite_jsonl_metrics() {
    let output = temporary_output("td3-metrics");
    let _ = std::fs::remove_file(&output);
    rustforge_cli::commands::train::execute(TrainArgs {
        execution: Default::default(),
        algorithm: Algorithm::Td3,
        env: Environment::Pendulum,
        episodes: 1,
        no_log: false,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
    })
    .unwrap();
    let text = std::fs::read_to_string(&output).unwrap();
    assert_eq!(text.lines().count(), 1);
    let metrics = text
        .split_once("\"metrics\":{")
        .unwrap()
        .1
        .strip_suffix("}}\n")
        .unwrap();
    assert_eq!(metrics.split(',').count(), 6);
    assert!(metrics.contains("\"loss.critic\":"));
    assert!(metrics.split(',').all(|m| m
        .split_once(':')
        .unwrap()
        .1
        .parse::<f64>()
        .unwrap()
        .is_finite()));
    std::fs::remove_file(output).unwrap();
}

#[test]
fn headless_cpu_sac_pendulum_writes_eight_finite_jsonl_metrics() {
    let output = temporary_output("sac-metrics");
    let _ = std::fs::remove_file(&output);
    rustforge_cli::commands::train::execute(TrainArgs {
        execution: Default::default(),
        algorithm: Algorithm::Sac,
        env: Environment::Pendulum,
        episodes: 1,
        no_log: false,
        output: Some(output.clone()),
        overwrite: false,
        use_per: false,
    })
    .unwrap();
    let text = std::fs::read_to_string(&output).unwrap();
    assert_eq!(text.lines().count(), 1);
    let metrics = text
        .split_once("\"metrics\":{")
        .unwrap()
        .1
        .strip_suffix("}}\n")
        .unwrap();
    assert_eq!(metrics.split(',').count(), 8);
    assert!(metrics.contains("\"loss.critic\":"));
    assert!(metrics.split(',').all(|m| m
        .split_once(':')
        .unwrap()
        .1
        .parse::<f64>()
        .unwrap()
        .is_finite()));
    std::fs::remove_file(output).unwrap();
}
