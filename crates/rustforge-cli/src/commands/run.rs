use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::thread;

use anyhow::Context;
use rustforge_rl::agent::{
    cartpole_a2c_config, cartpole_ppo_config, cartpole_reinforce_config, pendulum_ppo_config,
    A2cTrainerAdapter, DQNConfig, DqnTrainerAdapter, PpoContinuousTrainerAdapter,
    PpoDiscreteTrainerAdapter, ReinforceTrainerAdapter,
};
use rustforge_rl::env::{CartPole, GridWorld, Pendulum, PendulumAction};
use rustforge_rl::metrics::DqnCsvMetricSink;
use rustforge_rl::runtime::control::TrainerControl;
use rustforge_rl::runtime::event::{
    bounded_event_channel, DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT,
};
use rustforge_rl::runtime::persistence::{
    JsonlMetricSink, MetricSink, PersistenceStatus, RunArtifacts, RunManifest, DQN_CSV_V1_SCHEMA,
    GENERIC_JSONL_V1_SCHEMA,
};
use rustforge_rl::runtime::progress::{progress_channel, ProgressReader};
use rustforge_rl::runtime::trainer::{
    finalize_outcome, OutcomeSlot, StopReason, Trainer, TrainerContext, TrainerStatus,
    TrainingOutcome, TrainingSummary,
};
use rustforge_tui::live::{run_live, LiveOptions, LiveSession};
use rustforge_tui::terminal::{preflight_current_terminal, preflight_current_terminal_size};

use crate::cli::{Algorithm, Environment, ExecutionArgs, RunArgs};
use crate::commands::train::{dqn_config, validate_algorithm_environment};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetricFormat {
    DqnCsvV1,
    GenericJsonlV1,
}

impl MetricFormat {
    fn schema(self) -> &'static str {
        match self {
            Self::DqnCsvV1 => DQN_CSV_V1_SCHEMA,
            Self::GenericJsonlV1 => GENERIC_JSONL_V1_SCHEMA,
        }
    }

    fn create_sink(
        self,
        path: &Path,
        descriptors: &[rustforge_rl::runtime::trainer::MetricDescriptor],
    ) -> anyhow::Result<Box<dyn MetricSink>> {
        match self {
            Self::DqnCsvV1 => Ok(Box::new(
                DqnCsvMetricSink::create(path, descriptors)
                    .context("create DQN CSV v1 persistence sink")?,
            )),
            Self::GenericJsonlV1 => Ok(Box::new(
                JsonlMetricSink::create(path, descriptors)
                    .context("create generic JSONL v1 persistence sink")?,
            )),
        }
    }
}

struct TrainingPlan {
    trainer: Box<dyn Trainer>,
    display_config: Vec<(String, String)>,
    metrics: MetricFormat,
}

pub async fn execute(args: RunArgs) -> anyhow::Result<()> {
    validate_algorithm_environment(args.algorithm, args.env, args.use_per)?;
    args.execution
        .runtime_options(args.algorithm, args.use_per)?;
    preflight_current_terminal().context("rustforge run requires an interactive terminal")?;
    preflight_current_terminal_size().context("terminal is too small for rustforge run")?;

    let TrainingPlan {
        trainer,
        display_config,
        metrics,
    } = training_plan(
        args.algorithm,
        args.env,
        args.episodes,
        args.use_per,
        &args.execution,
    )?;
    let metadata = trainer.metadata();
    if let Some(output) = &args.output {
        crate::commands::train::validate_checkpoint_paths(
            &args.execution,
            Some(&output.join("metrics.csv")),
        )?;
        crate::commands::train::validate_checkpoint_paths(
            &args.execution,
            Some(&output.join("manifest.json")),
        )?;
        let directory = crate::commands::train::resolve_path(output)?;
        for path in [&args.execution.resume, &args.execution.checkpoint]
            .into_iter()
            .flatten()
        {
            if crate::commands::train::resolve_path(path)?.starts_with(&directory) {
                anyhow::bail!("checkpoint paths must be outside the live run output directory");
            }
        }
    }
    let mut source_config = BTreeMap::from([
        ("episodes".into(), args.episodes.to_string()),
        ("environment".into(), metadata.environment.clone()),
    ]);
    source_config.insert("device".into(), args.execution.device.label().into());
    if let Some(path) = &args.execution.resume {
        source_config.insert("resume".into(), path.display().to_string());
    }
    if let Some(path) = &args.execution.checkpoint {
        source_config.insert("checkpoint".into(), path.display().to_string());
    }
    if args.algorithm == Algorithm::Dqn {
        source_config.insert(
            if args.execution.resume.is_some() {
                "requested_use_per"
            } else {
                "use_per"
            }
            .into(),
            args.use_per.to_string(),
        );
    }
    let manifest = RunManifest::started_with_metrics_schema(
        &metadata,
        metrics.schema(),
        Some(2026),
        args.target_reward,
        source_config,
    );
    let artifacts = match &args.output {
        Some(path) => RunArtifacts::create_at(path, args.overwrite, manifest),
        None => RunArtifacts::create_default(Path::new("target/runs"), manifest),
    }
    .context("create run artifacts")?;
    let sink = metrics.create_sink(artifacts.metrics_path(), &metadata.metrics)?;

    let (publisher, event_receiver, delivery) =
        bounded_event_channel(DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT);
    let (progress, progress_reader) = progress_channel();
    let final_progress = progress_reader.clone();
    let control = TrainerControl::new();
    let persistence = PersistenceStatus::new();
    let persistence_for_thread = persistence.clone();
    let outcome = OutcomeSlot::new();
    let outcome_for_thread = outcome.clone();
    let final_publisher = publisher.clone();
    let context = TrainerContext {
        events: Box::new(publisher),
        progress,
        control: control.clone(),
        metrics: sink,
        persistence,
    };
    let trainer_thread = thread::spawn(move || {
        let result = catch_unwind(AssertUnwindSafe(|| trainer.run(context)));
        let mut training_outcome = match result {
            Ok(Ok(summary)) => TrainingOutcome {
                status: if summary.stop_reason == StopReason::Completed {
                    TrainerStatus::Completed
                } else {
                    TrainerStatus::Stopped
                },
                summary,
                persistence: persistence_for_thread.load(),
                event_delivery_complete: delivery.is_complete(),
                error: None,
            },
            Ok(Err(error)) => failed_outcome(error.to_string(), &final_progress),
            Err(payload) => failed_outcome(panic_message(payload), &final_progress),
        };
        training_outcome.persistence = persistence_for_thread.load();
        training_outcome.event_delivery_complete &= delivery.is_complete();
        finalize_outcome(&outcome_for_thread, &final_publisher, training_outcome)
    });

    let live_options = LiveOptions {
        no_color: args.no_color || std::env::var_os("NO_COLOR").is_some(),
        ascii: args.ascii,
        target_reward: args.target_reward,
        total_episodes: args.episodes as u64,
        metrics_path: artifacts.metrics_path().to_path_buf(),
        manifest_path: artifacts.manifest_path().to_path_buf(),
        metrics_schema: metrics.schema().into(),
        seed: Some(2026),
        device: Some(args.execution.device.label().into()),
        configuration: display_config,
    };
    let result = run_live(
        live_options,
        LiveSession {
            events: event_receiver,
            progress: progress_reader,
            control,
            metadata,
            outcome: outcome.clone(),
            trainer: trainer_thread,
        },
    )
    .await;

    let authoritative = result.as_ref().ok().cloned().or_else(|| outcome.load());
    if let Some(authoritative) = &authoritative {
        artifacts
            .finalize(authoritative)
            .context("finalize run manifest")?;
    }
    let training_outcome = result?;
    println!("run artifacts: {}", artifacts.directory().display());
    if training_outcome.status == TrainerStatus::Failed {
        anyhow::bail!(
            "training failed: {}",
            training_outcome.error.as_deref().unwrap_or("unknown error")
        );
    }
    Ok(())
}

fn training_plan(
    algorithm: Algorithm,
    env: Environment,
    episodes: usize,
    use_per: bool,
    execution: &ExecutionArgs,
) -> anyhow::Result<TrainingPlan> {
    validate_algorithm_environment(algorithm, env, use_per)?;
    let runtime_options = execution.runtime_options(algorithm, use_per)?;
    match (algorithm, env) {
        (Algorithm::Dqn, Environment::Cartpole) => {
            let max_steps = 500;
            let config = dqn_config(env, use_per);
            let display_config = if execution.resume.is_some() {
                vec![
                    (
                        "Agent configuration".into(),
                        "Restored from checkpoint".into(),
                    ),
                    ("Replay / exploration".into(), "New run".into()),
                ]
            } else {
                dqn_display_config(episodes, max_steps, &config)
            };
            Ok(TrainingPlan {
                trainer: Box::new(
                    DqnTrainerAdapter::new(
                        CartPole::with_max_steps(max_steps),
                        config,
                        episodes,
                        max_steps,
                        "cartpole",
                    )
                    .with_options(runtime_options.clone()),
                ),
                display_config,
                metrics: MetricFormat::DqnCsvV1,
            })
        }
        (Algorithm::Dqn, Environment::Gridworld) => {
            let max_steps = 100;
            let config = dqn_config(env, use_per);
            let display_config = if execution.resume.is_some() {
                vec![
                    (
                        "Agent configuration".into(),
                        "Restored from checkpoint".into(),
                    ),
                    ("Replay / exploration".into(), "New run".into()),
                ]
            } else {
                dqn_display_config(episodes, max_steps, &config)
            };
            Ok(TrainingPlan {
                trainer: Box::new(
                    DqnTrainerAdapter::new(
                        GridWorld::new(),
                        config,
                        episodes,
                        max_steps,
                        "gridworld",
                    )
                    .with_options(runtime_options.clone()),
                ),
                display_config,
                metrics: MetricFormat::DqnCsvV1,
            })
        }
        (Algorithm::Ppo, Environment::Cartpole) => {
            let max_steps = 500;
            let config = cartpole_ppo_config();
            let display_config = vec![
                ("Episodes".into(), episodes.to_string()),
                ("Max steps / episode".into(), max_steps.to_string()),
                (
                    "Observation dimensions".into(),
                    config.base.obs_dim.to_string(),
                ),
                ("Actions".into(), config.num_actions.to_string()),
                (
                    "Hidden dimensions".into(),
                    config.base.hidden_dim.to_string(),
                ),
                ("Learning rate".into(), config.base.lr.to_string()),
                ("Discount gamma".into(), config.base.gamma.to_string()),
                ("GAE lambda".into(), config.base.gae_lambda.to_string()),
                ("PPO epochs".into(), config.base.ppo_epochs.to_string()),
            ];
            let display_config = if execution.resume.is_some() {
                vec![
                    (
                        "Agent configuration".into(),
                        "Restored from checkpoint".into(),
                    ),
                    ("Rollout / random streams".into(), "New run".into()),
                ]
            } else {
                display_config
            };
            Ok(TrainingPlan {
                trainer: Box::new(
                    PpoDiscreteTrainerAdapter::new(
                        CartPole::with_max_steps(max_steps),
                        config,
                        episodes,
                        max_steps,
                        "cartpole",
                        Some(2026),
                    )
                    .with_options(runtime_options.clone().into()),
                ),
                display_config,
                metrics: MetricFormat::GenericJsonlV1,
            })
        }
        (Algorithm::Ppo, Environment::Pendulum) => {
            let config = pendulum_ppo_config();
            let display_config = if execution.resume.is_some() {
                vec![
                    (
                        "Agent configuration".into(),
                        "Restored from checkpoint".into(),
                    ),
                    ("Rollout / random streams".into(), "New run".into()),
                ]
            } else {
                vec![
                    ("Episodes".into(), episodes.to_string()),
                    ("Max steps / episode".into(), "200".into()),
                    ("Action bounds".into(), "[-2, 2]".into()),
                    (
                        "Hidden dimensions".into(),
                        config.base.hidden_dim.to_string(),
                    ),
                    ("Learning rate".into(), config.base.lr.to_string()),
                    ("PPO epochs".into(), config.base.ppo_epochs.to_string()),
                ]
            };
            Ok(TrainingPlan {
                trainer: Box::new(
                    PpoContinuousTrainerAdapter::new(
                        Pendulum::with_max_steps(200),
                        |a: &[f32]| Ok(PendulumAction::new(a[0])),
                        config,
                        episodes,
                        200,
                        "pendulum",
                        Some(2026),
                    )
                    .with_options(runtime_options.clone().into()),
                ),
                display_config,
                metrics: MetricFormat::GenericJsonlV1,
            })
        }
        (Algorithm::Ppo, Environment::Gridworld) => unreachable!("validated above"),
        (Algorithm::A2c, Environment::Cartpole) => {
            let max_steps = 500;
            let config = cartpole_a2c_config();
            let display_config = vec![
                ("Episodes".into(), episodes.to_string()),
                ("Max steps / episode".into(), max_steps.to_string()),
                ("Observation dimensions".into(), config.obs_dim.to_string()),
                ("Actions".into(), config.num_actions.to_string()),
                ("Hidden dimensions".into(), config.hidden_dim.to_string()),
                ("Learning rate".into(), config.lr.to_string()),
                ("Discount gamma".into(), config.gamma.to_string()),
                ("GAE lambda".into(), config.lambda.to_string()),
            ];
            let display_config = if execution.resume.is_some() {
                vec![
                    (
                        "Agent configuration".into(),
                        "Restored from checkpoint".into(),
                    ),
                    ("Rollout / random streams".into(), "New run".into()),
                ]
            } else {
                display_config
            };
            Ok(TrainingPlan {
                trainer: Box::new(
                    A2cTrainerAdapter::new(
                        CartPole::with_max_steps(max_steps),
                        config,
                        episodes,
                        max_steps,
                        "cartpole",
                        Some(2026),
                    )
                    .with_options(runtime_options.clone().into()),
                ),
                display_config,
                metrics: MetricFormat::GenericJsonlV1,
            })
        }
        (Algorithm::A2c, Environment::Gridworld) => unreachable!("validated above"),
        (Algorithm::Reinforce, Environment::Cartpole) => {
            let max_steps = 500;
            let config = cartpole_reinforce_config();
            let display_config = vec![
                ("Episodes".into(), episodes.to_string()),
                ("Max steps / episode".into(), max_steps.to_string()),
                ("Observation dimensions".into(), config.obs_dim.to_string()),
                ("Actions".into(), config.num_actions.to_string()),
                ("Hidden dimensions".into(), config.hidden_dim.to_string()),
                ("Learning rate".into(), config.lr.to_string()),
                ("Discount gamma".into(), config.gamma.to_string()),
                ("Mean baseline".into(), config.use_baseline.to_string()),
            ];
            let display_config = if execution.resume.is_some() {
                vec![
                    (
                        "Agent configuration".into(),
                        "Restored from checkpoint".into(),
                    ),
                    ("Rollout / random streams".into(), "New run".into()),
                ]
            } else {
                display_config
            };
            Ok(TrainingPlan {
                trainer: Box::new(
                    ReinforceTrainerAdapter::new(
                        CartPole::with_max_steps(max_steps),
                        config,
                        episodes,
                        max_steps,
                        "cartpole",
                        Some(2026),
                    )
                    .with_options(runtime_options.clone().into()),
                ),
                display_config,
                metrics: MetricFormat::GenericJsonlV1,
            })
        }
        (Algorithm::Reinforce, Environment::Gridworld) => unreachable!("validated above"),
        (_, Environment::Pendulum) => unreachable!("validated above"),
    }
}

fn dqn_display_config(
    episodes: usize,
    max_steps: usize,
    config: &DQNConfig,
) -> Vec<(String, String)> {
    vec![
        ("Episodes".into(), episodes.to_string()),
        ("Max steps / episode".into(), max_steps.to_string()),
        ("Observation dimensions".into(), config.obs_dim.to_string()),
        ("Actions".into(), config.num_actions.to_string()),
        ("Hidden dimensions".into(), config.hidden_dim.to_string()),
        ("Learning rate".into(), config.lr.to_string()),
        ("Discount gamma".into(), config.gamma.to_string()),
        (
            "Target update frequency".into(),
            config.target_update_freq.to_string(),
        ),
        ("Double DQN".into(), config.double_dqn.to_string()),
        ("Prioritized replay".into(), config.use_per.to_string()),
        (
            "PER beta annealing steps".into(),
            config.per_beta_annealing_steps.to_string(),
        ),
    ]
}

fn failed_outcome(message: String, progress: &ProgressReader) -> TrainingOutcome {
    let snapshot = progress.snapshot();
    TrainingOutcome {
        status: TrainerStatus::Failed,
        summary: TrainingSummary::stopped(
            snapshot.global_step,
            snapshot.episode,
            snapshot.elapsed,
            StopReason::Failed,
        ),
        persistence: rustforge_rl::runtime::persistence::PersistenceSummary::complete(),
        event_delivery_complete: true,
        error: Some(message),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("trainer panicked: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("trainer panicked: {message}")
    } else {
        "trainer panicked with a non-string payload".into()
    }
}

#[cfg(test)]
mod tests {
    use super::training_plan;
    use crate::cli::{Algorithm, Environment};

    #[cfg(feature = "gpu")]
    #[test]
    fn a2c_gpu_plan_routes_options_and_labels_restored_configuration() {
        let options = crate::cli::ExecutionArgs {
            device: crate::cli::Device::Gpu,
            resume: Some("a2c.chk".into()),
            checkpoint: Some("a2c.chk".into()),
        };
        let plan =
            training_plan(Algorithm::A2c, Environment::Cartpole, 1, false, &options).unwrap();
        assert_eq!(plan.trainer.metadata().algorithm, "a2c");
        assert_eq!(plan.metrics.schema(), "rustforge-metrics-jsonl-v1");
        assert!(plan
            .display_config
            .iter()
            .any(|(name, value)| name == "Agent configuration"
                && value == "Restored from checkpoint"));
        assert!(!plan.trainer.metadata().capabilities.checkpoint);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn reinforce_gpu_plan_routes_options_and_labels_restored_configuration() {
        let options = crate::cli::ExecutionArgs {
            device: crate::cli::Device::Gpu,
            resume: Some("reinforce.chk".into()),
            checkpoint: Some("reinforce.chk".into()),
        };
        let plan = training_plan(
            Algorithm::Reinforce,
            Environment::Cartpole,
            1,
            false,
            &options,
        )
        .unwrap();
        assert_eq!(plan.trainer.metadata().algorithm, "reinforce");
        assert_eq!(plan.metrics.schema(), "rustforge-metrics-jsonl-v1");
        assert!(plan
            .display_config
            .iter()
            .any(|(name, value)| name == "Agent configuration"
                && value == "Restored from checkpoint"));
        assert!(!plan.trainer.metadata().capabilities.checkpoint);
    }

    #[test]
    fn continuous_training_plan_binds_pendulum_and_rejects_unsupported_algorithms() {
        let plan = training_plan(
            Algorithm::Ppo,
            Environment::Pendulum,
            1,
            false,
            &crate::cli::ExecutionArgs::default(),
        )
        .unwrap();
        assert_eq!(plan.trainer.metadata().algorithm, "ppo-continuous");
        assert_eq!(plan.trainer.metadata().environment, "pendulum");
        assert!(plan
            .display_config
            .iter()
            .any(|(name, _)| name == "Action bounds"));
        for algorithm in [Algorithm::Dqn, Algorithm::A2c, Algorithm::Reinforce] {
            assert!(training_plan(
                algorithm,
                Environment::Pendulum,
                1,
                false,
                &crate::cli::ExecutionArgs::default()
            )
            .is_err());
        }
    }

    #[test]
    fn gpu_training_plan_builds_worker_configuration_without_initializing_adapter() {
        let execution = crate::cli::ExecutionArgs {
            device: crate::cli::Device::Gpu,
            resume: Some("not-opened-on-caller.chk".into()),
            checkpoint: Some("output.chk".into()),
        };
        let plan =
            training_plan(Algorithm::Dqn, Environment::Gridworld, 1, false, &execution).unwrap();
        assert_eq!(plan.metrics.schema(), "dqn-csv-v1");
        assert!(plan.display_config.contains(&(
            "Agent configuration".into(),
            "Restored from checkpoint".into()
        )));
        assert!(!plan.trainer.metadata().capabilities.checkpoint); // Interactive checkpoint control remains unsupported.
        assert!(training_plan(Algorithm::Dqn, Environment::Gridworld, 1, true, &execution).is_ok());
        let plan =
            training_plan(Algorithm::Ppo, Environment::Cartpole, 1, false, &execution).unwrap();
        assert_eq!(plan.metrics.schema(), "rustforge-metrics-jsonl-v1");
        assert!(plan.display_config.contains(&(
            "Agent configuration".into(),
            "Restored from checkpoint".into()
        )));
        assert!(plan
            .display_config
            .contains(&("Rollout / random streams".into(), "New run".into())));
        assert!(!plan.trainer.metadata().capabilities.checkpoint);
    }

    #[test]
    fn training_plan_binds_ppo_runtime_and_jsonl_schema() {
        let plan = training_plan(
            Algorithm::Ppo,
            Environment::Cartpole,
            1,
            false,
            &Default::default(),
        )
        .unwrap();
        let metadata = plan.trainer.metadata();

        assert_eq!(metadata.algorithm, "ppo-discrete");
        assert_eq!(metadata.environment, "cartpole");
        assert!(plan
            .display_config
            .contains(&("Max steps / episode".into(), "500".into())));
        assert!(plan
            .display_config
            .contains(&("Learning rate".into(), "0.001".into())));
        assert_eq!(plan.metrics.schema(), "rustforge-metrics-jsonl-v1");
    }

    #[test]
    fn training_plan_binds_a2c_runtime_and_jsonl_schema() {
        let plan = training_plan(
            Algorithm::A2c,
            Environment::Cartpole,
            1,
            false,
            &Default::default(),
        )
        .unwrap();
        let metadata = plan.trainer.metadata();

        assert_eq!(metadata.algorithm, "a2c");
        assert_eq!(metadata.environment, "cartpole");
        assert!(plan
            .display_config
            .contains(&("Max steps / episode".into(), "500".into())));
        assert_eq!(plan.metrics.schema(), "rustforge-metrics-jsonl-v1");
    }

    #[test]
    fn training_plan_rejects_unsupported_a2c_combinations() {
        let gridworld = training_plan(
            Algorithm::A2c,
            Environment::Gridworld,
            1,
            false,
            &Default::default(),
        )
        .err()
        .expect("A2C GridWorld is rejected");
        assert!(gridworld.to_string().contains("only CartPole"));

        let per = training_plan(
            Algorithm::A2c,
            Environment::Cartpole,
            1,
            true,
            &Default::default(),
        )
        .err()
        .expect("A2C prioritized replay is rejected");
        assert!(per.to_string().contains("--use-per"));
    }

    #[test]
    fn training_plan_binds_reinforce_runtime_roles_and_jsonl_schema() {
        let plan = training_plan(
            Algorithm::Reinforce,
            Environment::Cartpole,
            1,
            false,
            &Default::default(),
        )
        .unwrap();
        let metadata = plan.trainer.metadata();

        assert_eq!(metadata.algorithm, "reinforce");
        assert_eq!(metadata.environment, "cartpole");
        assert!(plan
            .display_config
            .contains(&("Max steps / episode".into(), "500".into())));
        assert_eq!(plan.metrics.schema(), "rustforge-metrics-jsonl-v1");
        assert!(metadata.metrics.iter().any(|metric| {
            metric.role == Some(rustforge_rl::runtime::trainer::MetricRole::EpisodeReward)
        }));
        assert!(metadata.metrics.iter().any(|metric| {
            metric.role == Some(rustforge_rl::runtime::trainer::MetricRole::PrimaryLoss)
                && metric.name == "loss.policy"
        }));
        assert!(metadata.metrics.iter().any(|metric| {
            metric.role == Some(rustforge_rl::runtime::trainer::MetricRole::Throughput)
        }));
        assert!(!metadata.metrics.iter().any(|metric| {
            metric.role == Some(rustforge_rl::runtime::trainer::MetricRole::PolicySignal)
        }));
    }
}
