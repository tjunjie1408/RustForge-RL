use std::time::Duration;

use rustforge_rl::runtime::progress::{progress_channel, ProgressScalar, ProgressUpdate};
use rustforge_rl::runtime::trainer::{MetricId, TrainerStatus};
use smallvec::smallvec;

#[test]
fn unread_step_progress_is_coalesced_to_the_latest_snapshot() {
    let (publisher, reader) = progress_channel();
    for step in 1..=10_000 {
        publisher.publish(ProgressUpdate {
            status: TrainerStatus::Running,
            global_step: step,
            episode: step / 100,
            episode_step: step % 100,
            elapsed: Duration::from_millis(step),
            scalars: smallvec![ProgressScalar {
                metric: MetricId::new(1),
                value: step as f64,
            }],
        });
    }

    let latest = reader.snapshot();
    assert_eq!(latest.revision, 10_000);
    assert_eq!(latest.global_step, 10_000);
    assert_eq!(latest.scalars.len(), 1);
}
