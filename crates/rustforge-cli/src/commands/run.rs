use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::thread;

use anyhow::Context;
use rustforge_rl::agent::DqnTrainerAdapter;
use rustforge_rl::env::{CartPole, GridWorld};
use rustforge_rl::metrics::DqnCsvMetricSink;
use rustforge_rl::runtime::control::TrainerControl;
use rustforge_rl::runtime::event::{
    bounded_event_channel, DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT,
};
use rustforge_rl::runtime::persistence::{PersistenceStatus, RunArtifacts, RunManifest};
use rustforge_rl::runtime::progress::{progress_channel, ProgressReader};
use rustforge_rl::runtime::trainer::{
    finalize_outcome, OutcomeSlot, StopReason, Trainer, TrainerContext, TrainerStatus,
    TrainingOutcome, TrainingSummary,
};
use rustforge_tui::live::{run_live, LiveOptions, LiveSession};
use rustforge_tui::terminal::{preflight_current_terminal, preflight_current_terminal_size};

use crate::cli::{Algorithm, Environment, RunArgs};
use crate::commands::train::dqn_config;

pub async fn execute(args: RunArgs) -> anyhow::Result<()> {
    preflight_current_terminal().context("rustforge run requires an interactive terminal")?;
    preflight_current_terminal_size().context("terminal is too small for rustforge run")?;

    let trainer: Box<dyn Trainer> = match (args.algorithm, args.env) {
        (Algorithm::Dqn, Environment::Cartpole) => Box::new(DqnTrainerAdapter::new(
            CartPole::with_max_steps(500),
            dqn_config(args.env, args.use_per),
            args.episodes,
            500,
            "cartpole",
        )),
        (Algorithm::Dqn, Environment::Gridworld) => Box::new(DqnTrainerAdapter::new(
            GridWorld::new(),
            dqn_config(args.env, args.use_per),
            args.episodes,
            100,
            "gridworld",
        )),
    };
    let metadata = trainer.metadata();
    let source_config = BTreeMap::from([
        ("episodes".into(), args.episodes.to_string()),
        ("environment".into(), metadata.environment.clone()),
        ("use_per".into(), args.use_per.to_string()),
    ]);
    let manifest = RunManifest::started(&metadata, Some(2026), args.target_reward, source_config);
    let artifacts = match &args.output {
        Some(path) => RunArtifacts::create_at(path, args.overwrite, manifest),
        None => RunArtifacts::create_default(Path::new("target/runs"), manifest),
    }
    .context("create run artifacts")?;
    let sink = DqnCsvMetricSink::create(artifacts.metrics_path(), &metadata.metrics)
        .context("create DQN CSV v1 persistence sink")?;

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
        metrics: Box::new(sink),
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
