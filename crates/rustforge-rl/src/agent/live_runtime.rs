//! Live-training plumbing shared by the algorithm runtimes.
//!
//! Each runtime owns its training loop, metric descriptors, and metric
//! values. This module owns everything that must behave identically across
//! algorithms: lifecycle events, pause/stop control handling, progress
//! publication, and metric persistence tracking.

use std::collections::VecDeque;
use std::time::Duration;

use smallvec::SmallVec;

use crate::runtime::control::{ControlObservation, StopMode, TrainerControl};
use crate::runtime::event::{
    EpisodeSummary, MetricValue, StatusChanged, TrainingEvent, TrainingEventPublisher,
    TrainingStarted,
};
use crate::runtime::persistence::{
    MetricError, MetricRecord, MetricSink, PersistenceEvent, PersistenceStatus, PersistenceTracker,
};
use crate::runtime::progress::{ProgressPublisher, ProgressScalar, ProgressUpdate};
use crate::runtime::trainer::{TrainerContext, TrainerMetadata, TrainerStatus};

/// What the training loop should do after a control observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StepDecision {
    Continue,
    GracefulStop,
    ForceStop,
}

/// Counters that locate a progress update or episode record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StepPosition {
    pub(crate) global_step: u64,
    pub(crate) episode: u64,
    pub(crate) episode_step: u64,
    pub(crate) elapsed: Duration,
}

/// Publishes lifecycle, control, progress, and persistence state for one run.
pub(crate) struct LiveHooks {
    events: Box<dyn TrainingEventPublisher>,
    progress: ProgressPublisher,
    control: TrainerControl,
    metrics: Box<dyn MetricSink>,
    metadata: TrainerMetadata,
    status: TrainerStatus,
    persistence: PersistenceTracker,
    persistence_status: PersistenceStatus,
}

impl LiveHooks {
    pub(crate) fn new(context: TrainerContext, metadata: TrainerMetadata) -> Self {
        Self {
            events: context.events,
            progress: context.progress,
            control: context.control,
            metrics: context.metrics,
            persistence_status: context.persistence,
            metadata,
            status: TrainerStatus::Running,
            persistence: PersistenceTracker::new(),
        }
    }

    pub(crate) fn publish_started(&self) {
        self.publish(TrainingEvent::Started(TrainingStarted {
            run_id: self.metadata.run_id.clone(),
            algorithm: self.metadata.algorithm.clone(),
            environment: self.metadata.environment.clone(),
        }));
    }

    /// Applies pending controls at a step boundary and publishes progress.
    ///
    /// Blocks while paused. A stop request is reported to the caller, which
    /// decides whether it takes effect now or at the episode boundary.
    pub(crate) fn observe_controls(
        &mut self,
        position: StepPosition,
        scalars: SmallVec<[ProgressScalar; 8]>,
    ) -> StepDecision {
        let observation = self.control.observe(position.global_step, false);
        self.publish_resolutions(&observation);
        if let Some(decision) = self.stop_decision(observation.stop_mode, position, &scalars) {
            return decision;
        }

        if observation.effective_paused {
            self.publish_status(TrainerStatus::Paused);
            self.publish_progress(position, scalars.clone());
            let resumed = self.control.wait_while_paused(position.global_step, false);
            self.publish_resolutions(&resumed);
            if let Some(decision) = self.stop_decision(resumed.stop_mode, position, &scalars) {
                return decision;
            }
            self.publish_status(TrainerStatus::Running);
        }
        self.publish_progress(position, scalars);
        StepDecision::Continue
    }

    /// Publishes a completed episode and persists its metric record.
    pub(crate) fn publish_episode(
        &mut self,
        position: StepPosition,
        values: SmallVec<[MetricValue; 8]>,
    ) {
        self.publish(TrainingEvent::EpisodeCompleted(EpisodeSummary {
            episode: position.episode,
            global_step: position.global_step,
            length: position.episode_step,
            metrics: values.clone(),
        }));
        let result = self.metrics.emit(&MetricRecord {
            episode: position.episode,
            global_step: position.global_step,
            values,
        });
        self.record_persistence(result);
    }

    pub(crate) fn publish_progress(
        &self,
        position: StepPosition,
        scalars: SmallVec<[ProgressScalar; 8]>,
    ) {
        self.progress.publish(ProgressUpdate {
            status: self.status,
            global_step: position.global_step,
            episode: position.episode,
            episode_step: position.episode_step,
            elapsed: position.elapsed,
            scalars,
        });
    }

    pub(crate) fn flush_metrics(&mut self) {
        let result = self.metrics.flush();
        self.record_persistence(result);
    }

    fn stop_decision(
        &mut self,
        mode: StopMode,
        position: StepPosition,
        scalars: &SmallVec<[ProgressScalar; 8]>,
    ) -> Option<StepDecision> {
        let decision = match mode {
            StopMode::Force => StepDecision::ForceStop,
            StopMode::Graceful => StepDecision::GracefulStop,
            StopMode::None => return None,
        };
        self.publish_status(TrainerStatus::Stopping);
        self.publish_progress(position, scalars.clone());
        Some(decision)
    }

    fn publish(&self, event: TrainingEvent) {
        let _ = self.events.publish(event);
    }

    fn publish_status(&mut self, status: TrainerStatus) {
        if self.status != status {
            self.status = status;
            self.publish(TrainingEvent::StatusChanged(StatusChanged { status }));
        }
    }

    fn publish_resolutions(&self, observation: &ControlObservation) {
        for resolution in &observation.resolutions {
            self.publish(TrainingEvent::ControlApplied(*resolution));
        }
    }

    fn record_persistence(&mut self, result: Result<(), MetricError>) {
        let transition = match result {
            Ok(()) => self.persistence.record_recovered(),
            Err(error) => self.persistence.record_failure(error.message),
        };
        match transition {
            Some(PersistenceEvent::Failed(failure)) => {
                self.publish(TrainingEvent::PersistenceError(failure));
            }
            Some(PersistenceEvent::Recovered(recovery)) => {
                self.publish(TrainingEvent::PersistenceRecovered(recovery));
            }
            None => {}
        }
        self.persistence_status.store(self.persistence.summary());
    }
}

/// Converts an episode record into progress scalars.
pub(crate) fn progress_scalars(values: &[MetricValue]) -> SmallVec<[ProgressScalar; 8]> {
    values
        .iter()
        .map(|value| ProgressScalar {
            metric: value.metric,
            value: value.value,
        })
        .collect()
}

/// Environment steps per second since the run started.
pub(crate) fn throughput(global_step: u64, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        global_step as f64 / seconds
    } else {
        0.0
    }
}

/// Trailing average over the most recent episode rewards.
pub(crate) struct RewardWindow {
    rewards: VecDeque<f32>,
    capacity: usize,
}

impl RewardWindow {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            rewards: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Records a reward and returns the average of the retained window.
    pub(crate) fn push(&mut self, reward: f32) -> f32 {
        if self.rewards.len() == self.capacity {
            self.rewards.pop_front();
        }
        self.rewards.push_back(reward);
        // Sum oldest to newest, matching the original Vec-based window exactly.
        self.rewards.iter().sum::<f32>() / self.rewards.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_window_averages_only_the_most_recent_rewards() {
        let mut window = RewardWindow::new(3);
        assert_eq!(window.push(3.0), 3.0);
        assert_eq!(window.push(6.0), 4.5);
        assert_eq!(window.push(9.0), 6.0);
        assert_eq!(window.push(12.0), 9.0);
    }

    #[test]
    fn throughput_is_zero_before_time_elapses() {
        assert_eq!(throughput(10, Duration::ZERO), 0.0);
        assert_eq!(throughput(10, Duration::from_secs(2)), 5.0);
    }
}
