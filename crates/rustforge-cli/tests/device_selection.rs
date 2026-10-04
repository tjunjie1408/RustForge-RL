use clap::Parser;
use rustforge_cli::cli::{Cli, Commands};
use std::{fs, path::PathBuf};

fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("rustforge-device-{}-{name}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
#[test]
fn unsupported_device_combinations_preserve_existing_output() {
    let directory = directory("invalid");
    let output = directory.join("output.csv");
    fs::write(&output, "existing metrics").unwrap();
    for (algorithm, flags, expected) in [
        (
            "reinforce",
            vec!["--device", "gpu"],
            "only by DQN, PPO and A2C",
        ),
        (
            "dqn",
            vec!["--resume", "missing.chk"],
            "require --device gpu",
        ),
        (
            "dqn",
            vec!["--checkpoint", "output.chk"],
            "require --device gpu",
        ),
        (
            "ppo",
            vec!["--checkpoint", "output.chk"],
            "require --device gpu",
        ),
        (
            "ppo",
            vec!["--resume", "missing.chk"],
            "require --device gpu",
        ),
        (
            "ppo",
            vec!["--env", "pendulum", "--resume", "missing.chk"],
            "require --device gpu",
        ),
        (
            "dqn",
            vec!["--env", "pendulum"],
            "Pendulum supports only PPO",
        ),
        (
            "a2c",
            vec!["--resume", "missing.chk"],
            "require --device gpu",
        ),
        (
            "a2c",
            vec!["--checkpoint", "out.chk"],
            "require --device gpu",
        ),
    ] {
        let mut arguments = vec![
            "rustforge",
            "train",
            algorithm,
            "--output",
            output.to_str().unwrap(),
            "--overwrite",
        ];
        arguments.extend(flags);
        let args = match Cli::try_parse_from(arguments).unwrap().command {
            Commands::Train(args) => args,
            _ => unreachable!(),
        };
        let error = rustforge_cli::commands::train::execute(args).unwrap_err();
        assert!(error.to_string().contains(expected), "{error:#}");
        assert_eq!(fs::read_to_string(&output).unwrap(), "existing metrics");
    }
    fs::remove_dir_all(directory).unwrap();
}
#[cfg(not(feature = "gpu"))]
#[test]
fn unavailable_gpu_feature_fails_before_creating_metrics() {
    let directory = directory("feature");
    let output = directory.join("output.csv");
    for algorithm in ["dqn", "ppo", "a2c"] {
        let args = match Cli::try_parse_from([
            "rustforge",
            "train",
            algorithm,
            "--device",
            "gpu",
            "--output",
            output.to_str().unwrap(),
        ])
        .unwrap()
        .command
        {
            Commands::Train(args) => args,
            _ => unreachable!(),
        };
        let error = rustforge_cli::commands::train::execute(args).unwrap_err();
        assert!(error.to_string().contains("--features gpu"), "{error:#}");
        assert!(!output.exists());
    }
    fs::remove_dir_all(directory).unwrap();
}
#[cfg(feature = "gpu")]
#[test]
fn checkpoint_metrics_alias_is_rejected_before_truncation_or_device_creation() {
    let directory = directory("alias");
    let output = directory.join("agent.chk");
    fs::write(&output, "existing checkpoint").unwrap();
    for (algorithm, flag) in ["dqn", "ppo", "a2c"]
        .into_iter()
        .flat_map(|algorithm| ["--resume", "--checkpoint"].map(|flag| (algorithm, flag)))
    {
        let args = match Cli::try_parse_from([
            "rustforge",
            "train",
            algorithm,
            "--device",
            "gpu",
            "--output",
            output.to_str().unwrap(),
            "--overwrite",
            flag,
            directory.join(".").join("agent.chk").to_str().unwrap(),
        ])
        .unwrap()
        .command
        {
            Commands::Train(args) => args,
            _ => unreachable!(),
        };
        assert!(rustforge_cli::commands::train::execute(args)
            .unwrap_err()
            .to_string()
            .contains("must differ"));
        assert_eq!(fs::read_to_string(&output).unwrap(), "existing checkpoint");
    }
    fs::remove_dir_all(directory).unwrap();
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_cli_checkpoint_resume_routes_full_state_and_keeps_csv_format() {
    use rustforge_rl::agent::{DQNConfig, GpuDqn};
    use rustforge_tensor::gpu::GpuContext;
    let directory = directory("resume");
    let checkpoint = directory.join("agent.chk");
    let output = directory.join("metrics.csv");
    let context = GpuContext::new().expect("GPU CLI tests require an adapter");
    let mut agent = GpuDqn::new_seeded(
        &context,
        DQNConfig {
            obs_dim: 2,
            num_actions: 4,
            hidden_dim: 8,
            gamma: 0.9,
            target_update_freq: 7,
            ..DQNConfig::default()
        },
        42,
    )
    .unwrap();
    let mut batch = rustforge_rl::buffer::TransitionBatch::new(1, 2);
    batch.states.data_mut().fill(0.5);
    batch.next_states.data_mut().fill(0.5);
    batch.rewards.data_mut().fill(1.);
    batch.size = 1;
    for _ in 0..3 {
        agent.train_step(&batch).unwrap();
    }
    agent.save_checkpoint(&checkpoint).unwrap();
    let before = fs::read(&checkpoint).unwrap();
    let args = match Cli::try_parse_from([
        "rustforge",
        "train",
        "dqn",
        "--device",
        "gpu",
        "--env",
        "gridworld",
        "--episodes",
        "1",
        "--output",
        output.to_str().unwrap(),
        "--resume",
        checkpoint.to_str().unwrap(),
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ])
    .unwrap()
    .command
    {
        Commands::Train(args) => args,
        _ => unreachable!(),
    };
    rustforge_cli::commands::train::execute(args).unwrap();
    // A single GridWorld episode has fewer than 128 transitions: no optimizer update.
    assert_eq!(fs::read(&checkpoint).unwrap(), before);
    let restored = GpuDqn::load_checkpoint(&context, checkpoint).unwrap();
    assert_eq!(restored.train_steps(), 3);
    assert_eq!(restored.config().target_update_freq, 7);
    let csv = fs::read_to_string(output).unwrap();
    assert!(csv.starts_with("episode,reward,avg_loss,epsilon,global_step\n"));
    assert_eq!(csv.lines().count(), 2);
    fs::remove_dir_all(directory).unwrap();
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_cli_accepts_prioritized_replay_and_saves_mode_for_resume() {
    use rustforge_rl::agent::GpuDqn;
    use rustforge_tensor::gpu::GpuContext;
    let directory = directory("per");
    let checkpoint = directory.join("agent.chk");
    let args = match Cli::try_parse_from([
        "rustforge",
        "train",
        "dqn",
        "--device",
        "gpu",
        "--use-per",
        "--env",
        "gridworld",
        "--episodes",
        "1",
        "--no-log",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ])
    .unwrap()
    .command
    {
        Commands::Train(args) => args,
        _ => unreachable!(),
    };
    rustforge_cli::commands::train::execute(args).unwrap();
    let context = GpuContext::new().expect("GPU CLI tests require an adapter");
    let original = GpuDqn::load_checkpoint(&context, &checkpoint).unwrap();
    assert!(original.config().use_per);
    let before = fs::read(&checkpoint).unwrap();
    let args = match Cli::try_parse_from([
        "rustforge",
        "train",
        "dqn",
        "--device",
        "gpu",
        "--env",
        "gridworld",
        "--episodes",
        "1",
        "--no-log",
        "--resume",
        checkpoint.to_str().unwrap(),
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ])
    .unwrap()
    .command
    {
        Commands::Train(args) => args,
        _ => unreachable!(),
    };
    rustforge_cli::commands::train::execute(args).unwrap();
    assert_eq!(fs::read(&checkpoint).unwrap(), before);
    fs::remove_dir_all(directory).unwrap();
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_ppo_cli_trains_and_resumes_saved_config_with_generic_jsonl_metrics() {
    use rustforge_cli::cli::{Algorithm, Device, Environment, ExecutionArgs, TrainArgs};
    use rustforge_rl::agent::{gpu_ppo::GpuPpoDiscrete, PPOConfig, PPODiscreteConfig};
    use rustforge_tensor::gpu::GpuContext;
    let directory = directory("ppo");
    let checkpoint = directory.join("ppo.chk");
    let metrics = directory.join("metrics.jsonl");
    let context = GpuContext::new().unwrap();
    // Custom saved epochs/width/lr must override CartPole CLI defaults.
    let config = PPODiscreteConfig {
        base: PPOConfig {
            obs_dim: 4,
            hidden_dim: 8,
            ppo_epochs: 2,
            mini_batch_size: 64,
            lr: 0.01,
            ..PPOConfig::default()
        },
        num_actions: 2,
    };
    GpuPpoDiscrete::new_seeded(&context, config.clone(), 42)
        .unwrap()
        .save_checkpoint(&checkpoint)
        .unwrap();
    let args = TrainArgs {
        execution: ExecutionArgs {
            device: Device::Gpu,
            resume: Some(checkpoint.clone()),
            checkpoint: Some(checkpoint.clone()),
        },
        algorithm: Algorithm::Ppo,
        env: Environment::Cartpole,
        episodes: 1,
        no_log: false,
        output: Some(metrics.clone()),
        overwrite: false,
        use_per: false,
    };
    rustforge_cli::commands::train::execute(args).unwrap();
    let resumed = GpuPpoDiscrete::load_checkpoint(&context, &checkpoint).unwrap();
    assert_eq!(resumed.config(), &config);
    assert!(resumed.updates() >= 2);
    let first = resumed.updates();
    let content = fs::read_to_string(&metrics).unwrap();
    assert_eq!(content.lines().count(), 1);
    assert!(
        content.contains("\"loss.policy\":")
            && content.contains("\"loss.value\":")
            && content.contains("\"policy.entropy\":")
    );
    assert!(!content.contains("NaN"));
    let args = TrainArgs {
        execution: ExecutionArgs {
            device: Device::Gpu,
            resume: Some(checkpoint.clone()),
            checkpoint: Some(checkpoint.clone()),
        },
        algorithm: Algorithm::Ppo,
        env: Environment::Cartpole,
        episodes: 1,
        no_log: true,
        output: None,
        overwrite: false,
        use_per: false,
    };
    rustforge_cli::commands::train::execute(args).unwrap();
    assert!(
        GpuPpoDiscrete::load_checkpoint(&context, &checkpoint)
            .unwrap()
            .updates()
            > first
    );
    fs::remove_dir_all(directory).unwrap();
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn continuous_gpu_ppo_cli_resumes_both_optimizers_and_writes_jsonl() {
    use rustforge_cli::cli::{Algorithm, Device, Environment, ExecutionArgs, TrainArgs};
    use rustforge_rl::agent::{gpu_ppo::GpuPpoContinuous, pendulum_ppo_config};
    use rustforge_tensor::gpu::GpuContext;
    let dir = directory("continuous-ppo");
    let checkpoint = dir.join("agent.chk");
    let metrics = dir.join("metrics.jsonl");
    let context = GpuContext::new().unwrap();
    let mut config = pendulum_ppo_config();
    config.base.hidden_dim = 8;
    config.base.ppo_epochs = 1;
    config.base.mini_batch_size = 64;
    config.base.lr = 1e-4;
    GpuPpoContinuous::new_seeded(&context, config.clone(), 42)
        .unwrap()
        .save_checkpoint(&checkpoint)
        .unwrap();
    for run in 0..2 {
        rustforge_cli::commands::train::execute(TrainArgs {
            execution: ExecutionArgs {
                device: Device::Gpu,
                resume: Some(checkpoint.clone()),
                checkpoint: Some(checkpoint.clone()),
            },
            algorithm: Algorithm::Ppo,
            env: Environment::Pendulum,
            episodes: 1,
            no_log: run == 1,
            output: if run == 0 {
                Some(metrics.clone())
            } else {
                None
            },
            overwrite: false,
            use_per: false,
        })
        .unwrap();
        let agent = GpuPpoContinuous::load_checkpoint(&context, &checkpoint).unwrap();
        assert_eq!(agent.config(), &config);
        assert_eq!(
            (agent.actor_updates(), agent.critic_updates()),
            (4 * (run + 1), 4 * (run + 1))
        );
    }
    let text = fs::read_to_string(metrics).unwrap();
    assert_eq!(text.lines().count(), 1);
    assert!(text.contains("\"loss.policy\":") && text.contains("\"loss.value\":"));
    let metrics = text
        .split_once("\"metrics\":{")
        .unwrap()
        .1
        .strip_suffix("}}\n")
        .unwrap();
    assert!(metrics.split(',').all(|m| m
        .split_once(':')
        .unwrap()
        .1
        .parse::<f64>()
        .unwrap()
        .is_finite()));
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_a2c_cli_trains_and_resumes_saved_configuration_and_adam_with_jsonl() {
    use rustforge_cli::cli::{Algorithm, Device, Environment, ExecutionArgs, TrainArgs};
    use rustforge_rl::agent::{cartpole_a2c_config, gpu_a2c::GpuA2c};
    use rustforge_tensor::gpu::GpuContext;
    let dir = directory("a2c");
    let checkpoint = dir.join("agent.chk");
    let metrics = dir.join("metrics.jsonl");
    let context = GpuContext::new().unwrap();
    let mut config = cartpole_a2c_config();
    config.hidden_dim = 8;
    config.lr = 0.003;
    config.gamma = 0.9;
    config.lambda = 0.8;
    GpuA2c::new_seeded(&context, config.clone(), 42)
        .unwrap()
        .save_checkpoint(&checkpoint)
        .unwrap();
    for run in 0..2 {
        rustforge_cli::commands::train::execute(TrainArgs {
            execution: ExecutionArgs {
                device: Device::Gpu,
                resume: Some(checkpoint.clone()),
                checkpoint: Some(checkpoint.clone()),
            },
            algorithm: Algorithm::A2c,
            env: Environment::Cartpole,
            episodes: 1,
            no_log: run == 1,
            output: if run == 0 {
                Some(metrics.clone())
            } else {
                None
            },
            overwrite: false,
            use_per: false,
        })
        .unwrap();
        let restored = GpuA2c::load_checkpoint(&context, &checkpoint).unwrap();
        assert_eq!(restored.config(), &config);
        assert_eq!(restored.updates(), run + 1);
    }
    let text = fs::read_to_string(metrics).unwrap();
    assert_eq!(text.lines().count(), 1);
    for name in ["loss.total", "loss.actor", "loss.critic", "policy.entropy"] {
        assert!(text.contains(&format!("\"{name}\":")));
    }
    let metrics = text
        .split_once("\"metrics\":{")
        .unwrap()
        .1
        .strip_suffix("}}\n")
        .unwrap();
    assert_eq!(metrics.split(',').count(), 8);
    assert!(metrics.split(',').all(|m| m
        .split_once(':')
        .unwrap()
        .1
        .parse::<f64>()
        .unwrap()
        .is_finite()));
    fs::remove_dir_all(dir).unwrap();
}
