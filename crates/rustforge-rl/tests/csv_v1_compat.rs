use std::fs::{self, OpenOptions};

use rustforge_rl::agent::{DQNConfig, DqnTrainerAdapter};
use rustforge_rl::env::CartPole;
use rustforge_rl::metrics::{DqnCsvMetricSink, DQN_CSV_V1_HEADER};
use rustforge_rl::runtime::event::MetricValue;
use rustforge_rl::runtime::persistence::{MetricRecord, MetricSink};
use rustforge_rl::runtime::trainer::Trainer;
use smallvec::smallvec;

#[test]
fn dqn_csv_sink_preserves_the_exact_v1_schema_and_field_order() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metrics.csv");
    let adapter = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    );
    let metadata = adapter.metadata();
    let id = |name: &str| {
        metadata
            .metrics
            .iter()
            .find(|metric| metric.name == name)
            .unwrap()
            .id
    };
    let mut sink = DqnCsvMetricSink::create(&path, &metadata.metrics).unwrap();

    sink.emit(&MetricRecord {
        episode: 7,
        global_step: 123,
        values: smallvec![
            MetricValue {
                metric: id("reward.episode"),
                value: 42.5
            },
            MetricValue {
                metric: id("reward.moving_average"),
                value: 40.0
            },
            MetricValue {
                metric: id("loss.td"),
                value: 0.125
            },
            MetricValue {
                metric: id("exploration.epsilon"),
                value: 0.2
            },
        ],
    })
    .unwrap();
    sink.flush().unwrap();

    let content = fs::read_to_string(path).unwrap();
    let lines: Vec<_> = content.lines().collect();
    assert_eq!(lines[0], DQN_CSV_V1_HEADER);
    assert_eq!(lines[1], "7,42.5,0.125,0.2,123");
}

#[test]
fn dqn_csv_sink_preserves_legacy_f32_text_after_runtime_widening() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("f32-text.csv");
    let metadata = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata();
    let id = |name: &str| {
        metadata
            .metrics
            .iter()
            .find(|metric| metric.name == name)
            .unwrap()
            .id
    };
    let mut sink = DqnCsvMetricSink::create(&path, &metadata.metrics).unwrap();

    sink.emit(&MetricRecord {
        episode: 3,
        global_step: 77,
        values: smallvec![
            MetricValue {
                metric: id("reward.episode"),
                value: f64::from(42.2_f32),
            },
            MetricValue {
                metric: id("loss.td"),
                value: f64::from(0.125_f32),
            },
            MetricValue {
                metric: id("exploration.epsilon"),
                value: f64::from(0.2_f32),
            },
        ],
    })
    .unwrap();
    sink.flush().unwrap();

    assert_eq!(
        fs::read_to_string(path).unwrap(),
        concat!(
            "episode,reward,avg_loss,epsilon,global_step\n",
            "3,42.2,0.125,0.2,77\n"
        )
    );
}

#[test]
fn missing_td_loss_is_written_as_nan_for_legacy_readers() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metrics.csv");
    let adapter = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    );
    let metadata = adapter.metadata();
    let id = |name: &str| {
        metadata
            .metrics
            .iter()
            .find(|metric| metric.name == name)
            .unwrap()
            .id
    };
    let mut sink = DqnCsvMetricSink::create(&path, &metadata.metrics).unwrap();
    sink.emit(&MetricRecord {
        episode: 0,
        global_step: 3,
        values: smallvec![
            MetricValue {
                metric: id("reward.episode"),
                value: 3.0
            },
            MetricValue {
                metric: id("exploration.epsilon"),
                value: 0.9
            },
        ],
    })
    .unwrap();
    sink.flush().unwrap();
    assert!(fs::read_to_string(path).unwrap().contains("0,3,NaN,0.9,3"));
}

#[test]
fn dqn_csv_sink_accepts_an_exclusively_opened_file_without_reopening_the_path() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("exclusive.csv");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let metadata = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        0,
        10,
        "cartpole",
    )
    .metadata();

    let mut sink = DqnCsvMetricSink::from_file(file, &metadata.metrics).unwrap();
    sink.flush().unwrap();

    assert_eq!(
        fs::read_to_string(path).unwrap(),
        format!("{DQN_CSV_V1_HEADER}\n")
    );
}

#[test]
fn invalid_dqn_descriptors_do_not_truncate_an_existing_path() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preserve.csv");
    fs::write(&path, "keep me").unwrap();

    let error = match DqnCsvMetricSink::create(&path, &[]) {
        Ok(_) => panic!("invalid descriptors must be rejected"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(fs::read_to_string(path).unwrap(), "keep me");
}
