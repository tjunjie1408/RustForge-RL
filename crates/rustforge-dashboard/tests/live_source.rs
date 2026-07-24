use rustforge_dashboard::source::live::LiveSource;
use rustforge_rl::agent::{DQNConfig, DqnTrainerAdapter};
use rustforge_rl::env::CartPole;
use rustforge_rl::runtime::event::{
    bounded_event_channel, EpisodeSummary, MetricValue, TrainingEvent,
};
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::Trainer;

#[tokio::test]
async fn live_source_maps_generic_episode_metrics_without_importing_dqn_fields() {
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
    let (publisher, receiver, _) = bounded_event_channel(16, std::time::Duration::from_millis(1));
    let (_progress, reader) = progress_channel();
    let mut source = LiveSource::new(receiver, reader, &metadata).unwrap();
    rustforge_rl::runtime::event::TrainingEventPublisher::publish(
        &publisher,
        TrainingEvent::EpisodeCompleted(EpisodeSummary {
            episode: 4,
            global_step: 99,
            length: 20,
            metrics: vec![
                MetricValue {
                    metric: id("reward.episode"),
                    value: 18.5,
                },
                MetricValue {
                    metric: id("loss.td"),
                    value: 0.25,
                },
                MetricValue {
                    metric: id("exploration.epsilon"),
                    value: 0.1,
                },
            ]
            .into(),
        }),
    )
    .unwrap();

    let poll = source.next().await.unwrap();
    assert_eq!(poll.rows.len(), 1);
    assert_eq!(poll.rows[0].episode, 4);
    assert_eq!(poll.rows[0].reward, 18.5);
    assert_eq!(poll.rows[0].avg_loss, Some(0.25));
    assert_eq!(poll.rows[0].epsilon, 0.1);
    assert_eq!(poll.rows[0].global_step, 99);
}
