//! Slow deterministic A2C CartPole learning acceptance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rustforge_rl::agent::{cartpole_a2c_config, A2cTrainerAdapter};
use rustforge_rl::env::CartPole;
use rustforge_rl::runtime::control::TrainerControl;
use rustforge_rl::runtime::event::{
    EventDeliveryError, EventSequence, TrainingEvent, TrainingEventPublisher,
};
use rustforge_rl::runtime::persistence::{
    MetricError, MetricRecord, MetricSink, PersistenceStatus,
};
use rustforge_rl::runtime::progress::progress_channel;
use rustforge_rl::runtime::trainer::{MetricId, Trainer, TrainerContext};

const SEED: u64 = 2026;
const EPISODES: usize = 500;
const MAX_STEPS_PER_EPISODE: usize = 500;
const WINDOW: usize = 100;
const FINAL_MEAN_THRESHOLD: f64 = 50.0;
const IMPROVEMENT_THRESHOLD: f64 = 25.0;

#[derive(Default)]
struct DiscardEvents {
    sequence: AtomicU64,
}

impl TrainingEventPublisher for DiscardEvents {
    fn publish(&self, _event: TrainingEvent) -> Result<EventSequence, EventDeliveryError> {
        Ok(EventSequence::new(
            self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
        ))
    }
}

struct RewardSink {
    reward_id: MetricId,
    rewards: Arc<Mutex<Vec<f64>>>,
}

impl MetricSink for RewardSink {
    fn emit(&mut self, record: &MetricRecord) -> Result<(), MetricError> {
        let mut matching = record
            .values
            .iter()
            .filter(|value| value.metric == self.reward_id);
        let reward = matching.next().ok_or_else(|| MetricError {
            message: "completed episode omitted reward.episode".into(),
        })?;
        if matching.next().is_some() {
            return Err(MetricError {
                message: "completed episode duplicated reward.episode".into(),
            });
        }
        self.rewards.lock().unwrap().push(reward.value);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MetricError> {
        Ok(())
    }
}

#[test]
#[ignore = "slow deterministic learning acceptance"]
fn a2c_converges_on_cartpole_seed_2026() {
    let adapter = A2cTrainerAdapter::new(
        CartPole::with_max_steps(MAX_STEPS_PER_EPISODE),
        cartpole_a2c_config(),
        EPISODES,
        MAX_STEPS_PER_EPISODE,
        "cartpole",
        Some(SEED),
    );
    let reward_id = adapter
        .metadata()
        .metrics
        .iter()
        .find(|descriptor| descriptor.name == "reward.episode")
        .expect("A2C metadata declares reward.episode")
        .id;
    let rewards = Arc::new(Mutex::new(Vec::with_capacity(EPISODES)));
    let (progress, _) = progress_channel();
    let context = TrainerContext {
        events: Box::new(DiscardEvents::default()),
        progress,
        control: TrainerControl::new(),
        metrics: Box::new(RewardSink {
            reward_id,
            rewards: rewards.clone(),
        }),
        persistence: PersistenceStatus::new(),
    };

    let started = Instant::now();
    let summary = Box::new(adapter)
        .run(context)
        .expect("A2C training succeeds");
    let elapsed = started.elapsed();
    let rewards = rewards.lock().unwrap();

    assert_eq!(summary.total_episodes, EPISODES as u64);
    assert_eq!(rewards.len(), EPISODES);
    let first_mean = rewards[..WINDOW].iter().sum::<f64>() / WINDOW as f64;
    let final_mean = rewards[EPISODES - WINDOW..].iter().sum::<f64>() / WINDOW as f64;
    let improvement = final_mean - first_mean;

    eprintln!(
        "A2C seed={SEED} episodes={EPISODES} first_100_mean={first_mean:.2} \
         final_100_mean={final_mean:.2} improvement={improvement:.2} elapsed={elapsed:?}"
    );

    assert!(
        final_mean >= FINAL_MEAN_THRESHOLD,
        "final-100 mean {final_mean:.2} is below {FINAL_MEAN_THRESHOLD:.2}"
    );
    assert!(
        improvement >= IMPROVEMENT_THRESHOLD,
        "improvement {improvement:.2} is below {IMPROVEMENT_THRESHOLD:.2}; \
         first-100={first_mean:.2}, final-100={final_mean:.2}"
    );
}
