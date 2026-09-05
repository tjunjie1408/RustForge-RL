use std::collections::BTreeMap;
use std::fs;
use std::time::Duration;

use rustforge_rl::agent::{DQNConfig, DqnTrainerAdapter};
use rustforge_rl::env::CartPole;
use rustforge_rl::runtime::persistence::{RunArtifacts, RunManifest, GENERIC_JSONL_V1_SCHEMA};
use rustforge_rl::runtime::trainer::{Trainer, TrainingOutcome};

#[test]
fn explicit_run_directory_is_never_overwritten_without_permission() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("chosen-run");
    fs::create_dir(&target).unwrap();
    fs::write(target.join("keep.txt"), "user data").unwrap();
    let metadata = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata();
    let manifest = RunManifest::started(&metadata, Some(42), Some(195.0), BTreeMap::new());

    let error = RunArtifacts::create_at(&target, false, manifest).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(
        fs::read_to_string(target.join("keep.txt")).unwrap(),
        "user data"
    );
}

#[test]
fn default_run_directories_are_collision_safe_and_manifest_is_finalized() {
    let directory = tempfile::tempdir().unwrap();
    let metadata = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata();
    let manifest = RunManifest::started(
        &metadata,
        Some(42),
        Some(195.0),
        BTreeMap::from([("episodes".into(), "1".into())]),
    );
    let first = RunArtifacts::create_default(directory.path(), manifest.clone()).unwrap();
    let second = RunArtifacts::create_default(directory.path(), manifest).unwrap();
    assert_ne!(first.directory(), second.directory());
    assert!(first.metrics_path().ends_with("metrics.csv"));

    let outcome = TrainingOutcome::completed(3, 1, Duration::from_millis(25));
    first.finalize(&outcome).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(first.manifest_path()).unwrap()).unwrap();
    assert_eq!(json["run_id"], metadata.run_id);
    assert_eq!(json["metrics_schema"], "dqn-csv-v1");
    assert_eq!(json["outcome"]["summary"]["total_steps"], 3);
    assert_eq!(json["outcome"]["event_delivery_complete"], true);
}

#[test]
fn manifest_can_select_dqn_or_generic_metrics_schema_explicitly() {
    let metadata = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata();

    let dqn = RunManifest::started_with_metrics_schema(
        &metadata,
        "dqn-csv-v1",
        Some(42),
        Some(195.0),
        BTreeMap::new(),
    );
    assert_eq!(dqn.schema_version, 1);
    assert_eq!(dqn.metrics_schema, "dqn-csv-v1");

    let generic = RunManifest::started_with_metrics_schema(
        &metadata,
        GENERIC_JSONL_V1_SCHEMA,
        Some(42),
        Some(195.0),
        BTreeMap::new(),
    );
    assert_eq!(generic.schema_version, 1);
    assert_eq!(generic.metrics_schema, GENERIC_JSONL_V1_SCHEMA);

    let json = serde_json::to_value(generic).unwrap();
    assert!(json["metrics"][0].get("role").is_none());
}

#[test]
fn artifact_metric_filename_matches_the_selected_schema() {
    let directory = tempfile::tempdir().unwrap();
    let metadata = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata();

    let dqn = RunArtifacts::create_at(
        directory.path().join("dqn"),
        false,
        RunManifest::started(&metadata, Some(42), None, BTreeMap::new()),
    )
    .unwrap();
    assert!(dqn.metrics_path().ends_with("metrics.csv"));

    let ppo = RunArtifacts::create_at(
        directory.path().join("ppo"),
        false,
        RunManifest::started_with_metrics_schema(
            &metadata,
            GENERIC_JSONL_V1_SCHEMA,
            Some(42),
            None,
            BTreeMap::new(),
        ),
    )
    .unwrap();
    assert!(ppo.metrics_path().ends_with("metrics.jsonl"));
}

#[test]
fn artifact_creation_rejects_unknown_metrics_schema() {
    let directory = tempfile::tempdir().unwrap();
    let metadata = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata();
    let manifest = RunManifest::started_with_metrics_schema(
        &metadata,
        "unknown-metrics-v9",
        Some(42),
        None,
        BTreeMap::new(),
    );

    let target = directory.path().join("unknown");
    let error = RunArtifacts::create_at(&target, false, manifest).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("unknown-metrics-v9"));
    assert!(!target.exists());
}
