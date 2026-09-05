use rustforge_rl::runtime::event::MetricValue;
use rustforge_rl::runtime::persistence::{
    JsonlMetricSink, MetricRecord, MetricSink, GENERIC_JSONL_V1_SCHEMA,
};
use rustforge_rl::runtime::trainer::{
    validate_metric_descriptors, MetricDescriptor, MetricId, MetricKind, MetricRole, Trainer,
};
use smallvec::smallvec;

fn descriptor(id: u16, name: &str, role: Option<MetricRole>) -> MetricDescriptor {
    MetricDescriptor {
        id: MetricId::new(id),
        name: name.into(),
        label: name.into(),
        unit: None,
        kind: MetricKind::Gauge,
        role,
    }
}

#[test]
fn metric_descriptors_accept_unique_ids_names_and_roles() {
    let descriptors = [
        descriptor(1, "reward.episode", Some(MetricRole::EpisodeReward)),
        descriptor(2, "loss.policy", Some(MetricRole::PrimaryLoss)),
        descriptor(3, "loss.value", None),
        descriptor(4, "policy.entropy", Some(MetricRole::PolicySignal)),
        descriptor(
            5,
            "performance.steps_per_second",
            Some(MetricRole::Throughput),
        ),
    ];

    validate_metric_descriptors(&descriptors).expect("unique descriptor schema is valid");
}

#[test]
fn metric_descriptors_reject_duplicate_ids() {
    let descriptors = [
        descriptor(7, "reward.episode", Some(MetricRole::EpisodeReward)),
        descriptor(7, "loss.policy", Some(MetricRole::PrimaryLoss)),
    ];

    let error = validate_metric_descriptors(&descriptors).unwrap_err();
    assert!(error.to_string().contains("duplicate metric id 7"));
}

#[test]
fn metric_descriptors_reject_duplicate_names() {
    let descriptors = [
        descriptor(1, "reward.episode", Some(MetricRole::EpisodeReward)),
        descriptor(2, "reward.episode", Some(MetricRole::PrimaryLoss)),
    ];

    let error = validate_metric_descriptors(&descriptors).unwrap_err();
    assert!(error
        .to_string()
        .contains("duplicate metric name reward.episode"));
}

#[test]
fn metric_descriptors_reject_empty_names() {
    let error = validate_metric_descriptors(&[descriptor(1, "", None)]).unwrap_err();
    assert!(error.to_string().contains("metric name must not be empty"));
}

#[test]
fn invalid_jsonl_descriptors_do_not_truncate_an_existing_path() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("preserve.jsonl");
    std::fs::write(&path, "keep me").unwrap();

    let error = match JsonlMetricSink::create(&path, &[descriptor(1, "", None)]) {
        Ok(_) => panic!("invalid descriptors must be rejected"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(std::fs::read_to_string(path).unwrap(), "keep me");
}

#[test]
fn metric_descriptors_reject_duplicate_assigned_roles() {
    let descriptors = [
        descriptor(1, "loss.policy", Some(MetricRole::PrimaryLoss)),
        descriptor(2, "loss.value", Some(MetricRole::PrimaryLoss)),
        descriptor(3, "rollout.size", None),
    ];

    let error = validate_metric_descriptors(&descriptors).unwrap_err();
    assert!(error
        .to_string()
        .contains("duplicate metric role PrimaryLoss"));
}

#[test]
fn generic_jsonl_v1_writes_exact_object_shape_one_per_line() {
    assert_eq!(GENERIC_JSONL_V1_SCHEMA, "rustforge-metrics-jsonl-v1");
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("metrics.jsonl");
    let descriptors = [
        descriptor(10, "reward.episode", Some(MetricRole::EpisodeReward)),
        descriptor(20, "loss.policy", Some(MetricRole::PrimaryLoss)),
    ];
    let mut sink = JsonlMetricSink::create(&path, &descriptors).unwrap();

    sink.emit(&MetricRecord {
        episode: 0,
        global_step: 128,
        values: smallvec![
            MetricValue {
                metric: MetricId::new(20),
                value: 0.04,
            },
            MetricValue {
                metric: MetricId::new(10),
                value: 12.5,
            },
        ],
    })
    .unwrap();
    sink.emit(&MetricRecord {
        episode: 1,
        global_step: 256,
        values: smallvec![MetricValue {
            metric: MetricId::new(10),
            value: 9.0,
        }],
    })
    .unwrap();
    sink.flush().unwrap();

    let content = std::fs::read_to_string(path).unwrap();
    let lines: Vec<_> = content.lines().collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        lines[0],
        r#"{"episode":0,"global_step":128,"metrics":{"reward.episode":12.5,"loss.policy":0.04}}"#
    );
    assert_eq!(
        lines[1],
        r#"{"episode":1,"global_step":256,"metrics":{"reward.episode":9.0}}"#
    );
}

#[test]
fn generic_jsonl_v1_emit_is_immediately_visible_to_an_independent_reader() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("durable.jsonl");
    let mut sink =
        JsonlMetricSink::create(&path, &[descriptor(1, "reward.episode", None)]).unwrap();

    sink.emit(&MetricRecord {
        episode: 0,
        global_step: 3,
        values: smallvec![MetricValue {
            metric: MetricId::new(1),
            value: 3.0,
        }],
    })
    .unwrap();

    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        "{\"episode\":0,\"global_step\":3,\"metrics\":{\"reward.episode\":3.0}}\n"
    );
}

struct PartialThenErrorWriter {
    bytes: Arc<Mutex<Vec<u8>>>,
    remaining: usize,
}

impl Write for PartialThenErrorWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("injected write failure"));
        }
        let written = self.remaining.min(buffer.len());
        self.bytes
            .lock()
            .unwrap()
            .extend_from_slice(&buffer[..written]);
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn generic_jsonl_v1_poisoned_sink_never_appends_after_a_partial_write() {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = PartialThenErrorWriter {
        bytes: bytes.clone(),
        remaining: 12,
    };
    let mut sink =
        JsonlMetricSink::from_writer(writer, &[descriptor(1, "reward.episode", None)]).unwrap();
    let record = MetricRecord {
        episode: 0,
        global_step: 3,
        values: smallvec![MetricValue {
            metric: MetricId::new(1),
            value: 3.0,
        }],
    };

    let first = sink.emit(&record).unwrap_err();
    assert!(first.to_string().contains("injected write failure"));
    let partial = bytes.lock().unwrap().clone();
    assert!(!partial.is_empty());

    let second = sink.emit(&record).unwrap_err();
    assert!(second.to_string().contains("poisoned"));
    assert!(sink.flush().unwrap_err().to_string().contains("poisoned"));
    assert_eq!(*bytes.lock().unwrap(), partial);
}

#[test]
fn generic_jsonl_v1_escapes_nonempty_metric_names() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("metrics.jsonl");
    let unusual_name = "comma,name\n\"quoted\"";
    let mut sink = JsonlMetricSink::create(&path, &[descriptor(1, unusual_name, None)]).unwrap();
    sink.emit(&MetricRecord {
        episode: 0,
        global_step: 1,
        values: smallvec![MetricValue {
            metric: MetricId::new(1),
            value: 2.0,
        }],
    })
    .unwrap();
    sink.flush().unwrap();

    let value: serde_json::Value =
        serde_json::from_str(std::fs::read_to_string(path).unwrap().trim()).unwrap();
    assert_eq!(value["metrics"][unusual_name], 2.0);
}

#[test]
fn generic_jsonl_v1_rejects_duplicate_and_unknown_record_ids_before_writing() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("metrics.jsonl");
    let mut sink =
        JsonlMetricSink::create(&path, &[descriptor(1, "reward.episode", None)]).unwrap();

    let duplicate = sink
        .emit(&MetricRecord {
            episode: 0,
            global_step: 1,
            values: smallvec![
                MetricValue {
                    metric: MetricId::new(1),
                    value: 1.0,
                },
                MetricValue {
                    metric: MetricId::new(1),
                    value: 2.0,
                },
            ],
        })
        .unwrap_err();
    assert!(duplicate.to_string().contains("duplicate metric id 1"));

    let unknown = sink
        .emit(&MetricRecord {
            episode: 0,
            global_step: 1,
            values: smallvec![MetricValue {
                metric: MetricId::new(99),
                value: 1.0,
            }],
        })
        .unwrap_err();
    assert!(unknown.to_string().contains("unknown metric id 99"));
    sink.flush().unwrap();
    assert_eq!(std::fs::read_to_string(path).unwrap(), "");
}

#[test]
fn generic_jsonl_v1_rejects_non_finite_values_before_writing() {
    let temp = tempfile::tempdir().unwrap();

    for (index, value) in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        .into_iter()
        .enumerate()
    {
        let path = temp.path().join(format!("non-finite-{index}.jsonl"));
        let mut sink =
            JsonlMetricSink::create(&path, &[descriptor(1, "reward.episode", None)]).unwrap();
        let error = sink
            .emit(&MetricRecord {
                episode: 0,
                global_step: 1,
                values: smallvec![MetricValue {
                    metric: MetricId::new(1),
                    value,
                }],
            })
            .unwrap_err();
        assert!(error.to_string().contains("non-finite metric id 1"));
        sink.flush().unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "");
    }
}

#[test]
fn dqn_and_ppo_metadata_assign_only_dashboard_semantic_roles() {
    let dqn = DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata();
    let ppo = PpoDiscreteTrainerAdapter::new(
        CartPole::with_max_steps(10),
        PPODiscreteConfig {
            base: PPOConfig::default(),
            num_actions: 2,
        },
        1,
        10,
        "cartpole",
        None,
    )
    .metadata();

    let roles = |metadata: &rustforge_rl::runtime::trainer::TrainerMetadata| {
        metadata
            .metrics
            .iter()
            .map(|metric| (metric.name.clone(), metric.role))
            .collect::<BTreeMap<_, _>>()
    };
    let dqn_roles = roles(&dqn);
    assert_eq!(dqn_roles["reward.episode"], Some(MetricRole::EpisodeReward));
    assert_eq!(dqn_roles["loss.td"], Some(MetricRole::PrimaryLoss));
    assert_eq!(
        dqn_roles["exploration.epsilon"],
        Some(MetricRole::PolicySignal)
    );
    assert_eq!(
        dqn_roles["performance.steps_per_second"],
        Some(MetricRole::Throughput)
    );
    assert_eq!(dqn_roles["reward.moving_average"], None);
    assert_eq!(dqn_roles["replay_buffer.size"], None);

    let ppo_roles = roles(&ppo);
    assert_eq!(ppo_roles["reward.episode"], Some(MetricRole::EpisodeReward));
    assert_eq!(ppo_roles["loss.policy"], Some(MetricRole::PrimaryLoss));
    assert_eq!(ppo_roles["policy.entropy"], Some(MetricRole::PolicySignal));
    assert_eq!(
        ppo_roles["performance.steps_per_second"],
        Some(MetricRole::Throughput)
    );
    assert_eq!(ppo_roles["reward.moving_average"], None);
    assert_eq!(ppo_roles["loss.value"], None);
    assert_eq!(ppo_roles["rollout.size"], None);

    validate_metric_descriptors(&dqn.metrics).unwrap();
    validate_metric_descriptors(&ppo.metrics).unwrap();
}
use std::collections::BTreeMap;

use rustforge_rl::agent::{
    DQNConfig, DqnTrainerAdapter, PPOConfig, PPODiscreteConfig, PpoDiscreteTrainerAdapter,
};
use rustforge_rl::env::CartPole;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
