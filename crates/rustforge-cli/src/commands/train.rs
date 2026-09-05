use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context;
use rustforge_rl::agent::{
    cartpole_a2c_config, cartpole_ppo_config, cartpole_reinforce_config, A2cTrainerAdapter,
    DQNConfig, DqnTrainerAdapter, PpoDiscreteTrainerAdapter, ReinforceTrainerAdapter,
};
use rustforge_rl::env::{CartPole, GridWorld};
use rustforge_rl::metrics::DqnCsvMetricSink;
use rustforge_rl::runtime::control::TrainerControl;
use rustforge_rl::runtime::event::{
    EventDeliveryError, EventSequence, TrainingEvent, TrainingEventPublisher,
};
use rustforge_rl::runtime::persistence::{
    JsonlMetricSink, MetricSink, NullMetricSink, PersistenceStatus,
};
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::{Trainer, TrainerContext, TrainingSummary};

use crate::cli::{Algorithm, Environment, TrainArgs};

pub fn execute(args: TrainArgs) -> anyhow::Result<()> {
    validate_algorithm_environment(args.algorithm, args.env, args.use_per)?;
    let explicit_output = args.output.is_some();
    let output = args.output.unwrap_or_else(|| match args.algorithm {
        Algorithm::Dqn => "target/cli_train_dqn.csv".into(),
        Algorithm::Ppo => "target/cli_train_ppo.jsonl".into(),
        Algorithm::A2c => "target/cli_train_a2c.jsonl".into(),
        Algorithm::Reinforce => "target/cli_train_reinforce.jsonl".into(),
    });
    let (trainer, metric_format): (Box<dyn Trainer>, HeadlessMetricFormat) =
        match (args.algorithm, args.env) {
            (Algorithm::Dqn, Environment::Cartpole) => {
                println!("Training DQN on CartPole for {} episodes...", args.episodes);
                (
                    Box::new(DqnTrainerAdapter::new(
                        CartPole::with_max_steps(500),
                        dqn_config(args.env, args.use_per),
                        args.episodes,
                        500,
                        "cartpole",
                    )),
                    HeadlessMetricFormat::DqnCsvV1,
                )
            }
            (Algorithm::Dqn, Environment::Gridworld) => {
                println!(
                    "Training DQN on GridWorld for {} episodes...",
                    args.episodes
                );
                (
                    Box::new(DqnTrainerAdapter::new(
                        GridWorld::new(),
                        dqn_config(args.env, args.use_per),
                        args.episodes,
                        100,
                        "gridworld",
                    )),
                    HeadlessMetricFormat::DqnCsvV1,
                )
            }
            (Algorithm::Ppo, Environment::Cartpole) => {
                println!("Training PPO on CartPole for {} episodes...", args.episodes);
                (
                    Box::new(PpoDiscreteTrainerAdapter::new(
                        CartPole::with_max_steps(500),
                        cartpole_ppo_config(),
                        args.episodes,
                        500,
                        "cartpole",
                        Some(2026),
                    )),
                    HeadlessMetricFormat::GenericJsonlV1,
                )
            }
            (Algorithm::Ppo, Environment::Gridworld) => unreachable!("validated above"),
            (Algorithm::A2c, Environment::Cartpole) => {
                println!("Training A2C on CartPole for {} episodes...", args.episodes);
                (
                    Box::new(A2cTrainerAdapter::new(
                        CartPole::with_max_steps(500),
                        cartpole_a2c_config(),
                        args.episodes,
                        500,
                        "cartpole",
                        Some(2026),
                    )),
                    HeadlessMetricFormat::GenericJsonlV1,
                )
            }
            (Algorithm::A2c, Environment::Gridworld) => unreachable!("validated above"),
            (Algorithm::Reinforce, Environment::Cartpole) => {
                println!(
                    "Training REINFORCE on CartPole for {} episodes...",
                    args.episodes
                );
                (
                    Box::new(ReinforceTrainerAdapter::new(
                        CartPole::with_max_steps(500),
                        cartpole_reinforce_config(),
                        args.episodes,
                        500,
                        "cartpole",
                        Some(2026),
                    )),
                    HeadlessMetricFormat::GenericJsonlV1,
                )
            }
            (Algorithm::Reinforce, Environment::Gridworld) => unreachable!("validated above"),
        };

    let metadata = trainer.metadata();
    let metrics: Box<dyn MetricSink> = match args.no_log {
        false => {
            let file = open_output_file(&output, explicit_output, args.overwrite)?;
            metric_format.create_sink(file, &metadata.metrics)?
        }
        true => Box::new(NullMetricSink),
    };
    let summary = run_headless(trainer, metrics, args.algorithm)?;
    if args.algorithm != Algorithm::Dqn {
        println!(
            "{:?} training completed: {} episode(s), {} step(s)",
            args.algorithm, summary.total_episodes, summary.total_steps
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum HeadlessMetricFormat {
    DqnCsvV1,
    GenericJsonlV1,
}

impl HeadlessMetricFormat {
    fn create_sink(
        self,
        file: File,
        descriptors: &[rustforge_rl::runtime::trainer::MetricDescriptor],
    ) -> anyhow::Result<Box<dyn MetricSink>> {
        match self {
            Self::DqnCsvV1 => Ok(Box::new(
                DqnCsvMetricSink::from_file(file, descriptors)
                    .context("create DQN CSV v1 persistence sink")?,
            )),
            Self::GenericJsonlV1 => Ok(Box::new(
                JsonlMetricSink::from_file(file, descriptors)
                    .context("create generic JSONL v1 persistence sink")?,
            )),
        }
    }
}

fn open_output_file(path: &Path, explicit: bool, overwrite: bool) -> anyhow::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true);
    if explicit && !overwrite {
        options.create_new(true);
    } else {
        options.create(true).truncate(true);
    }
    options.open(path).map_err(|error| {
        anyhow::anyhow!(
            "training output already exists or could not be opened: {}: {error}",
            path.display()
        )
    })
}

fn run_headless(
    trainer: Box<dyn Trainer>,
    metrics: Box<dyn MetricSink>,
    algorithm: Algorithm,
) -> anyhow::Result<TrainingSummary> {
    let (progress, _progress_reader) = progress_channel();
    let persistence = PersistenceStatus::new();
    let summary = trainer
        .run(TrainerContext {
            events: Box::new(DiscardEventPublisher::default()),
            progress,
            control: TrainerControl::new(),
            metrics,
            persistence: persistence.clone(),
        })
        .map_err(|error| anyhow::anyhow!("{algorithm:?} training failed: {error}"))?;
    let persistence = persistence.load();
    if !persistence.complete {
        anyhow::bail!(
            "{algorithm:?} metrics persistence incomplete after {} failure(s): {}",
            persistence.failures,
            persistence
                .last_error
                .as_deref()
                .unwrap_or("unknown persistence error")
        );
    }
    Ok(summary)
}

#[derive(Default)]
struct DiscardEventPublisher {
    sequence: AtomicU64,
}

impl TrainingEventPublisher for DiscardEventPublisher {
    fn publish(&self, _event: TrainingEvent) -> Result<EventSequence, EventDeliveryError> {
        Ok(EventSequence::new(
            self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
        ))
    }
}

pub(crate) fn validate_algorithm_environment(
    algorithm: Algorithm,
    env: Environment,
    use_per: bool,
) -> anyhow::Result<()> {
    if algorithm != Algorithm::Dqn && env != Environment::Cartpole {
        let algorithm_name = match algorithm {
            Algorithm::Ppo => "PPO",
            Algorithm::A2c => "A2C",
            Algorithm::Reinforce => "REINFORCE",
            Algorithm::Dqn => unreachable!("DQN supports GridWorld"),
        };
        anyhow::bail!("{algorithm_name} supports only CartPole");
    }
    if algorithm != Algorithm::Dqn && use_per {
        anyhow::bail!("--use-per is supported only by DQN");
    }
    Ok(())
}

pub(crate) fn dqn_config(env: Environment, use_per: bool) -> DQNConfig {
    DQNConfig {
        obs_dim: match env {
            Environment::Cartpole => 4,
            Environment::Gridworld => 2,
        },
        num_actions: match env {
            Environment::Cartpole => 2,
            Environment::Gridworld => 4,
        },
        hidden_dim: 64,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 100,
        double_dqn: true,
        use_per,
        per_beta_annealing_steps: 20_000,
    }
}
