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
        ("ppo", vec!["--device", "gpu"], "only by DQN"),
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
            "dqn",
            vec!["--device", "gpu", "--use-per"],
            "prioritized replay",
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
    let args = match Cli::try_parse_from([
        "rustforge",
        "train",
        "dqn",
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
    fs::remove_dir_all(directory).unwrap();
}
#[cfg(feature = "gpu")]
#[test]
fn checkpoint_metrics_alias_is_rejected_before_truncation_or_device_creation() {
    let directory = directory("alias");
    let output = directory.join("agent.chk");
    fs::write(&output, "existing checkpoint").unwrap();
    for flag in ["--resume", "--checkpoint"] {
        let args = match Cli::try_parse_from([
            "rustforge",
            "train",
            "dqn",
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
