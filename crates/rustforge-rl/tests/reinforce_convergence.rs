use std::sync::{Arc, Mutex};

use rustforge_rl::agent::{cartpole_reinforce_config, ReinforceTrainerAdapter};
use rustforge_rl::env::CartPole;
use rustforge_rl::runtime::control::TrainerControl;
use rustforge_rl::runtime::event::{
    bounded_event_channel, DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT,
};
use rustforge_rl::runtime::persistence::{
    MetricError, MetricRecord, MetricSink, PersistenceStatus,
};
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::{MetricId, Trainer, TrainerContext};

#[derive(Clone, Default)]
struct RewardSink {
    rewards: Arc<Mutex<Vec<f64>>>,
}

impl MetricSink for RewardSink {
    fn emit(&mut self, record: &MetricRecord) -> Result<(), MetricError> {
        let reward = record
            .values
            .iter()
            .find(|value| value.metric == MetricId::new(301))
            .expect("episode reward metric")
            .value;
        self.rewards.lock().unwrap().push(reward);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MetricError> {
        Ok(())
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[test]
#[ignore = "deterministic 1,000-episode learning acceptance gate"]
fn reinforce_cartpole_seed_2026_meets_learning_gate() {
    let sink = RewardSink::default();
    let rewards = sink.rewards.clone();
    let (events, _, _) = bounded_event_channel(DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_PUBLISH_WAIT);
    let (progress, _) = progress_channel();
    let adapter = ReinforceTrainerAdapter::new(
        CartPole::with_max_steps(500),
        cartpole_reinforce_config(),
        1_000,
        500,
        "cartpole",
        Some(2026),
    );
    let summary = Box::new(adapter)
        .run(TrainerContext {
            events: Box::new(events),
            progress,
            control: TrainerControl::new(),
            metrics: Box::new(sink),
            persistence: PersistenceStatus::new(),
        })
        .expect("training succeeds");

    let rewards = rewards.lock().unwrap();
    assert_eq!(summary.total_episodes, 1_000);
    assert_eq!(rewards.len(), 1_000);
    let first_100 = mean(&rewards[..100]);
    let final_100 = mean(&rewards[900..]);
    let improvement = final_100 - first_100;
    println!(
        "REINFORCE seed=2026 first_100={first_100:.6} final_100={final_100:.6} improvement={improvement:.6}"
    );
    assert!(
        final_100 >= 100.0,
        "final-100 mean {final_100:.6} is below 100"
    );
    assert!(
        improvement >= 50.0,
        "improvement {improvement:.6} is below 50 (first={first_100:.6}, final={final_100:.6})"
    );
}
