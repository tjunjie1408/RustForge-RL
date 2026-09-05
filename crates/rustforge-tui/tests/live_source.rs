use std::time::{Duration, Instant};

use rustforge_rl::agent::{
    cartpole_a2c_config, A2cTrainerAdapter, DQNConfig, DqnTrainerAdapter, PPOConfig,
    PPODiscreteConfig, PpoDiscreteTrainerAdapter,
};
use rustforge_rl::env::CartPole;
use rustforge_rl::runtime::event::{
    bounded_event_channel, BoundedEventPublisher, EpisodeSummary, MetricValue, TrainingEvent,
    TrainingEventPublisher,
};
use rustforge_rl::runtime::progress::{
    progress_channel, ProgressPublisher, ProgressScalar, ProgressUpdate,
};
use rustforge_rl::runtime::trainer::{
    MetricId, MetricRole, Trainer, TrainerMetadata, TrainerStatus,
};
use rustforge_tui::source::csv::CsvDiagnosticKind;
use rustforge_tui::source::live::LiveSource;

fn dqn_metadata() -> TrainerMetadata {
    DqnTrainerAdapter::new(
        CartPole::with_max_steps(10),
        DQNConfig::default(),
        1,
        10,
        "cartpole",
    )
    .metadata()
}

fn ppo_metadata() -> TrainerMetadata {
    PpoDiscreteTrainerAdapter::new(
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
    .metadata()
}

fn a2c_metadata() -> TrainerMetadata {
    A2cTrainerAdapter::new(
        CartPole::with_max_steps(10),
        cartpole_a2c_config(),
        1,
        10,
        "cartpole",
        None,
    )
    .metadata()
}

fn role_id(metadata: &TrainerMetadata, role: MetricRole) -> MetricId {
    metadata
        .metrics
        .iter()
        .find(|metric| metric.role == Some(role))
        .unwrap()
        .id
}

fn runtime(metadata: &TrainerMetadata) -> (BoundedEventPublisher, ProgressPublisher, LiveSource) {
    let (events, receiver, _) = bounded_event_channel(16, Duration::from_millis(1));
    let (progress, reader) = progress_channel();
    let source = LiveSource::new(receiver, reader, metadata).unwrap();
    (events, progress, source)
}

fn publish_episode(publisher: &BoundedEventPublisher, episode: u64, metrics: Vec<MetricValue>) {
    publisher
        .publish(TrainingEvent::EpisodeCompleted(EpisodeSummary {
            episode,
            global_step: 99,
            length: 20,
            metrics: metrics.into(),
        }))
        .unwrap();
}

#[tokio::test]
async fn live_source_resolves_dqn_metrics_by_role_even_when_names_change() {
    let mut metadata = dqn_metadata();
    for metric in &mut metadata.metrics {
        if metric.role.is_some() {
            metric.name = format!("renamed.{}", metric.id.get());
        }
    }
    let reward = role_id(&metadata, MetricRole::EpisodeReward);
    let loss = role_id(&metadata, MetricRole::PrimaryLoss);
    let policy = role_id(&metadata, MetricRole::PolicySignal);
    let throughput = role_id(&metadata, MetricRole::Throughput);
    let (events, progress, mut source) = runtime(&metadata);

    publish_episode(
        &events,
        4,
        vec![
            MetricValue {
                metric: policy,
                value: 0.1,
            },
            MetricValue {
                metric: reward,
                value: 18.5,
            },
            MetricValue {
                metric: loss,
                value: 0.25,
            },
        ],
    );
    progress.publish(ProgressUpdate {
        status: TrainerStatus::Running,
        global_step: 99,
        episode: 4,
        episode_step: 20,
        elapsed: Duration::from_secs(2),
        scalars: vec![ProgressScalar {
            metric: throughput,
            value: 49.5,
        }]
        .into(),
    });

    let poll = source.next().await.unwrap();
    assert_eq!(poll.rows.len(), 1);
    assert_eq!(poll.rows[0].episode, 4);
    assert_eq!(poll.rows[0].reward, 18.5);
    assert_eq!(poll.rows[0].primary_loss, Some(0.25));
    assert_eq!(poll.rows[0].policy_signal, Some(0.1));
    assert_eq!(poll.rows[0].global_step, 99);
    assert_eq!(
        source.progress(None, Instant::now()).steps_per_second,
        Some(49.5)
    );
    assert_eq!(
        source.metric_labels().primary_loss.as_deref(),
        Some("TD loss")
    );
    assert_eq!(
        source.metric_labels().policy_signal.as_deref(),
        Some("Epsilon")
    );
}

#[tokio::test]
async fn ppo_metadata_initializes_without_dqn_metric_names_and_uses_ppo_labels() {
    let metadata = ppo_metadata();
    assert!(!metadata
        .metrics
        .iter()
        .any(|metric| metric.name == "loss.td"));
    assert!(!metadata
        .metrics
        .iter()
        .any(|metric| metric.name == "exploration.epsilon"));
    let reward = role_id(&metadata, MetricRole::EpisodeReward);
    let loss = role_id(&metadata, MetricRole::PrimaryLoss);
    let policy = role_id(&metadata, MetricRole::PolicySignal);
    let (events, _progress, mut source) = runtime(&metadata);

    publish_episode(
        &events,
        0,
        vec![
            MetricValue {
                metric: reward,
                value: 12.5,
            },
            MetricValue {
                metric: loss,
                value: 0.04,
            },
            MetricValue {
                metric: policy,
                value: 0.72,
            },
        ],
    );

    let poll = source.next().await.unwrap();
    assert_eq!(poll.rows[0].primary_loss, Some(0.04));
    assert_eq!(poll.rows[0].policy_signal, Some(0.72));
    assert_eq!(
        source.metric_labels().primary_loss.as_deref(),
        Some("PPO policy loss")
    );
    assert_eq!(
        source.metric_labels().policy_signal.as_deref(),
        Some("PPO policy entropy")
    );
}

#[tokio::test]
async fn a2c_metadata_uses_role_driven_actor_loss_and_entropy_labels() {
    let metadata = a2c_metadata();
    let reward = role_id(&metadata, MetricRole::EpisodeReward);
    let loss = role_id(&metadata, MetricRole::PrimaryLoss);
    let policy = role_id(&metadata, MetricRole::PolicySignal);
    let (events, _progress, mut source) = runtime(&metadata);

    publish_episode(
        &events,
        0,
        vec![
            MetricValue {
                metric: reward,
                value: 8.0,
            },
            MetricValue {
                metric: loss,
                value: 0.03,
            },
            MetricValue {
                metric: policy,
                value: 0.68,
            },
        ],
    );
    let poll = source.next().await.unwrap();
    assert_eq!(poll.rows[0].primary_loss, Some(0.03));
    assert_eq!(poll.rows[0].policy_signal, Some(0.68));
    assert_eq!(
        source.metric_labels().primary_loss.as_deref(),
        Some("A2C actor loss")
    );
    assert_eq!(
        source.metric_labels().policy_signal.as_deref(),
        Some("A2C policy entropy")
    );
}

#[test]
fn live_source_rejects_missing_or_duplicate_required_roles() {
    for required in [MetricRole::EpisodeReward, MetricRole::Throughput] {
        let mut metadata = dqn_metadata();
        metadata
            .metrics
            .iter_mut()
            .find(|metric| metric.role == Some(required))
            .unwrap()
            .role = None;
        let (_events, receiver, _) = bounded_event_channel(4, Duration::from_millis(1));
        let (_progress, reader) = progress_channel();
        let error = LiveSource::new(receiver, reader, &metadata).err().unwrap();
        assert!(error.contains(&format!("missing required metric role {required:?}")));
    }

    let mut duplicate = dqn_metadata();
    duplicate
        .metrics
        .iter_mut()
        .find(|metric| metric.role.is_none())
        .unwrap()
        .role = Some(MetricRole::EpisodeReward);
    let (_events, receiver, _) = bounded_event_channel(4, Duration::from_millis(1));
    let (_progress, reader) = progress_channel();
    let error = LiveSource::new(receiver, reader, &duplicate).err().unwrap();
    assert!(error.contains("duplicate metric role EpisodeReward"));
}

#[tokio::test]
async fn malformed_required_reward_is_diagnosed_without_fabricating_a_row() {
    let metadata = dqn_metadata();
    let reward = role_id(&metadata, MetricRole::EpisodeReward);
    let cases = [
        ("missing", vec![]),
        (
            "duplicate",
            vec![
                MetricValue {
                    metric: reward,
                    value: 1.0,
                },
                MetricValue {
                    metric: reward,
                    value: 2.0,
                },
            ],
        ),
        (
            "NaN",
            vec![MetricValue {
                metric: reward,
                value: f64::NAN,
            }],
        ),
        (
            "infinite",
            vec![MetricValue {
                metric: reward,
                value: f64::INFINITY,
            }],
        ),
        (
            "outside f32 range",
            vec![MetricValue {
                metric: reward,
                value: f64::MAX,
            }],
        ),
    ];

    for (case, metrics) in cases {
        let (events, _progress, mut source) = runtime(&metadata);
        publish_episode(&events, 0, metrics);
        let poll = source.next().await.unwrap();
        assert!(poll.rows.is_empty(), "{case} reward fabricated a row");
        assert!(poll.diagnostics.iter().any(|item| {
            item.kind == CsvDiagnosticKind::MalformedRow && item.message.contains("episode reward")
        }));
    }
}

#[tokio::test]
async fn missing_or_non_finite_optional_metrics_map_to_none() {
    let metadata = ppo_metadata();
    let reward = role_id(&metadata, MetricRole::EpisodeReward);
    let loss = role_id(&metadata, MetricRole::PrimaryLoss);
    let policy = role_id(&metadata, MetricRole::PolicySignal);
    let (events, _progress, mut source) = runtime(&metadata);
    publish_episode(
        &events,
        0,
        vec![
            MetricValue {
                metric: reward,
                value: 3.0,
            },
            MetricValue {
                metric: loss,
                value: f64::MAX,
            },
            MetricValue {
                metric: policy,
                value: f64::NEG_INFINITY,
            },
        ],
    );

    let poll = source.next().await.unwrap();
    assert_eq!(poll.rows.len(), 1);
    assert_eq!(poll.rows[0].primary_loss, None);
    assert_eq!(poll.rows[0].policy_signal, None);
}

#[tokio::test]
async fn metadata_may_omit_optional_roles_entirely() {
    let mut metadata = ppo_metadata();
    for metric in &mut metadata.metrics {
        if matches!(
            metric.role,
            Some(MetricRole::PrimaryLoss | MetricRole::PolicySignal)
        ) {
            metric.role = None;
        }
    }
    let reward = role_id(&metadata, MetricRole::EpisodeReward);
    let (events, _progress, mut source) = runtime(&metadata);
    assert_eq!(source.metric_labels().primary_loss, None);
    assert_eq!(source.metric_labels().policy_signal, None);
    publish_episode(
        &events,
        0,
        vec![MetricValue {
            metric: reward,
            value: 3.0,
        }],
    );

    let poll = source.next().await.unwrap();
    assert_eq!(poll.rows.len(), 1);
    assert_eq!(poll.rows[0].primary_loss, None);
    assert_eq!(poll.rows[0].policy_signal, None);
}
