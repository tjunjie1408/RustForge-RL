//! Adapter from the generic in-process training protocol to dashboard observations.

use std::time::Instant;

use crossbeam_channel::Receiver;
use rustforge_rl::runtime::event::{EventEnvelope, MetricValue, TrainingEvent};
use rustforge_rl::runtime::progress::{ProgressReader, ProgressSnapshot};
use rustforge_rl::runtime::trainer::{
    validate_metric_descriptors, MetricDescriptor, MetricId, MetricRole, TrainerMetadata,
    TrainerStatus,
};

use crate::app::MonitorInsights;
use crate::metrics::{MetricLabels, MetricRow};
use crate::source::csv::{CsvDiagnostic, CsvDiagnosticKind, CsvSourcePoll, MonitorSourceState};

pub struct LiveSource {
    events: tokio::sync::mpsc::UnboundedReceiver<EventEnvelope>,
    progress: ProgressReader,
    metrics: LiveMetricRoles,
    state: MonitorSourceState,
    last_revision: u64,
    last_progress_at: Instant,
    disconnected: bool,
}

#[derive(Clone, Debug)]
struct ResolvedMetric {
    id: MetricId,
    label: String,
}

#[derive(Clone, Debug)]
struct LiveMetricRoles {
    episode_reward: ResolvedMetric,
    primary_loss: Option<ResolvedMetric>,
    policy_signal: Option<ResolvedMetric>,
    throughput: ResolvedMetric,
}

impl LiveMetricRoles {
    fn resolve(metadata: &TrainerMetadata) -> Result<Self, String> {
        validate_metric_descriptors(&metadata.metrics)
            .map_err(|error| format!("invalid live metric descriptors: {error}"))?;
        let required = |role| {
            descriptor_for_role(metadata, role)
                .map(ResolvedMetric::from)
                .ok_or_else(|| format!("live source is missing required metric role {role:?}"))
        };
        Ok(Self {
            episode_reward: required(MetricRole::EpisodeReward)?,
            primary_loss: descriptor_for_role(metadata, MetricRole::PrimaryLoss)
                .map(ResolvedMetric::from),
            policy_signal: descriptor_for_role(metadata, MetricRole::PolicySignal)
                .map(ResolvedMetric::from),
            throughput: required(MetricRole::Throughput)?,
        })
    }

    fn labels(&self) -> MetricLabels {
        MetricLabels {
            episode_reward: self.episode_reward.label.clone(),
            primary_loss: self
                .primary_loss
                .as_ref()
                .map(|metric| metric.label.clone()),
            policy_signal: self
                .policy_signal
                .as_ref()
                .map(|metric| metric.label.clone()),
            throughput: self.throughput.label.clone(),
        }
    }
}

impl From<&MetricDescriptor> for ResolvedMetric {
    fn from(descriptor: &MetricDescriptor) -> Self {
        Self {
            id: descriptor.id,
            label: descriptor.label.clone(),
        }
    }
}

fn descriptor_for_role(metadata: &TrainerMetadata, role: MetricRole) -> Option<&MetricDescriptor> {
    metadata
        .metrics
        .iter()
        .find(|descriptor| descriptor.role == Some(role))
}

impl LiveSource {
    pub fn new(
        events: Receiver<EventEnvelope>,
        progress: ProgressReader,
        metadata: &TrainerMetadata,
    ) -> Result<Self, String> {
        let metrics = LiveMetricRoles::resolve(metadata)?;
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        std::thread::spawn(move || {
            while let Ok(envelope) = events.recv() {
                if event_tx.send(envelope).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            events: event_rx,
            progress,
            metrics,
            state: MonitorSourceState::Waiting,
            last_revision: 0,
            last_progress_at: Instant::now(),
            disconnected: false,
        })
    }

    pub fn drain(&mut self) -> CsvSourcePoll {
        let mut poll = CsvSourcePoll {
            rows: Vec::new(),
            state: self.state,
            reset: false,
            diagnostics: Vec::new(),
        };
        loop {
            match self.events.try_recv() {
                Ok(envelope) => self.apply(envelope, &mut poll),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    self.disconnected = true;
                    break;
                }
            }
        }
        poll.state = self.state;
        poll
    }

    pub async fn next(&mut self) -> Option<CsvSourcePoll> {
        let envelope = self.events.recv().await;
        match envelope {
            Some(envelope) => {
                let mut poll = CsvSourcePoll {
                    rows: Vec::new(),
                    state: self.state,
                    reset: false,
                    diagnostics: Vec::new(),
                };
                self.apply(envelope, &mut poll);
                while let Ok(envelope) = self.events.try_recv() {
                    self.apply(envelope, &mut poll);
                }
                poll.state = self.state;
                Some(poll)
            }
            None => {
                self.disconnected = true;
                None
            }
        }
    }

    pub fn progress(&mut self, total_episodes: Option<u64>, now: Instant) -> MonitorInsights {
        let snapshot = self.progress.snapshot();
        if snapshot.revision != self.last_revision {
            self.last_revision = snapshot.revision;
            self.last_progress_at = now;
        }
        progress_insights(
            &snapshot,
            self.metrics.throughput.id,
            total_episodes,
            now,
            self.last_progress_at,
        )
    }

    pub fn progress_snapshot(&self) -> std::sync::Arc<ProgressSnapshot> {
        self.progress.snapshot()
    }

    pub fn disconnected(&self) -> bool {
        self.disconnected
    }

    pub fn metric_labels(&self) -> MetricLabels {
        self.metrics.labels()
    }

    fn apply(&mut self, envelope: EventEnvelope, poll: &mut CsvSourcePoll) {
        let sequence = envelope.sequence.get();
        match envelope.event {
            TrainingEvent::Started(started) => {
                self.state = MonitorSourceState::Following;
                diagnostic(
                    poll,
                    CsvDiagnosticKind::Lifecycle,
                    sequence,
                    format!(
                        "training started: {} on {} ({})",
                        started.algorithm, started.environment, started.run_id
                    ),
                );
            }
            TrainingEvent::StatusChanged(change) => {
                self.state = state_for_status(change.status);
                diagnostic(
                    poll,
                    CsvDiagnosticKind::Lifecycle,
                    sequence,
                    format!("status changed to {:?}", change.status),
                );
            }
            TrainingEvent::ControlApplied(resolution) => diagnostic(
                poll,
                CsvDiagnosticKind::Control,
                sequence,
                format!(
                    "{:?} request {}: {:?} at step {}",
                    resolution.control,
                    resolution.request_id.get(),
                    resolution.result,
                    resolution.applied_at_step
                ),
            ),
            TrainingEvent::EpisodeCompleted(summary) => {
                let reward = match required_finite_value(
                    &summary.metrics,
                    &self.metrics.episode_reward,
                    "episode reward",
                ) {
                    Ok(reward) => reward,
                    Err(message) => {
                        diagnostic(poll, CsvDiagnosticKind::MalformedRow, sequence, message);
                        self.state = MonitorSourceState::Following;
                        return;
                    }
                };
                let primary_loss =
                    optional_finite_value(&summary.metrics, self.metrics.primary_loss.as_ref());
                let policy_signal =
                    optional_finite_value(&summary.metrics, self.metrics.policy_signal.as_ref());
                poll.rows.push(MetricRow {
                    episode: summary.episode,
                    reward,
                    primary_loss,
                    policy_signal,
                    global_step: summary.global_step,
                });
                self.state = MonitorSourceState::Following;
            }
            TrainingEvent::PersistenceError(failure) => diagnostic(
                poll,
                CsvDiagnosticKind::Persistence,
                sequence,
                format!(
                    "persistence degraded after {} failure(s): {}",
                    failure.failures, failure.message
                ),
            ),
            TrainingEvent::PersistenceRecovered(recovery) => diagnostic(
                poll,
                CsvDiagnosticKind::Persistence,
                sequence,
                format!(
                    "persistence recovered after {} failure(s)",
                    recovery.failures
                ),
            ),
            TrainingEvent::Finished(finished) => {
                self.state = MonitorSourceState::Completed;
                diagnostic(
                    poll,
                    CsvDiagnosticKind::Lifecycle,
                    sequence,
                    format!("training finished: {:?}", finished.summary.stop_reason),
                );
            }
            TrainingEvent::Failed(failure) => {
                self.state = MonitorSourceState::SourceError;
                diagnostic(
                    poll,
                    CsvDiagnosticKind::Lifecycle,
                    sequence,
                    format!("training failed: {}", failure.message),
                );
            }
        }
    }
}

fn required_finite_value(
    values: &[MetricValue],
    metric: &ResolvedMetric,
    role_name: &str,
) -> Result<f32, String> {
    let mut matches = values.iter().filter(|value| value.metric == metric.id);
    let Some(value) = matches.next() else {
        return Err(format!("{role_name} metric is missing"));
    };
    if matches.next().is_some() {
        return Err(format!("{role_name} metric is duplicated"));
    }
    let converted = value.value as f32;
    if !value.value.is_finite() || !converted.is_finite() {
        return Err(format!("{role_name} metric must be finite"));
    }
    Ok(converted)
}

fn optional_finite_value(values: &[MetricValue], metric: Option<&ResolvedMetric>) -> Option<f32> {
    let metric = metric?;
    let mut matches = values.iter().filter(|value| value.metric == metric.id);
    let value = matches.next()?;
    let converted = value.value as f32;
    if matches.next().is_some() || !value.value.is_finite() || !converted.is_finite() {
        return None;
    }
    Some(converted)
}

fn state_for_status(status: TrainerStatus) -> MonitorSourceState {
    match status {
        TrainerStatus::Running | TrainerStatus::Stopping => MonitorSourceState::Following,
        TrainerStatus::Paused => MonitorSourceState::Idle,
        TrainerStatus::Stopped | TrainerStatus::Completed => MonitorSourceState::Completed,
        TrainerStatus::Failed => MonitorSourceState::SourceError,
    }
}

fn diagnostic(poll: &mut CsvSourcePoll, kind: CsvDiagnosticKind, sequence: u64, message: String) {
    poll.diagnostics.push(CsvDiagnostic {
        kind,
        line: Some(sequence),
        message,
    });
}

fn progress_insights(
    snapshot: &ProgressSnapshot,
    throughput_metric: MetricId,
    total_episodes: Option<u64>,
    now: Instant,
    last_progress_at: Instant,
) -> MonitorInsights {
    let steps_per_second = snapshot
        .scalars
        .iter()
        .find(|scalar| scalar.metric == throughput_metric)
        .map(|scalar| scalar.value)
        .filter(|value| value.is_finite());
    let completed = snapshot.episode.saturating_add(1);
    let progress_fraction = total_episodes
        .filter(|total| *total > 0)
        .map(|total| (completed as f64 / total as f64).clamp(0.0, 1.0));
    let episodes_per_minute = (snapshot.elapsed.as_secs_f64() > 0.0)
        .then(|| completed as f64 / snapshot.elapsed.as_secs_f64() * 60.0);
    let eta = match (total_episodes, episodes_per_minute) {
        (Some(total), Some(rate)) if rate > 0.001 && completed < total => Some(
            std::time::Duration::from_secs_f64((total - completed) as f64 / (rate / 60.0)),
        ),
        _ => None,
    };
    MonitorInsights {
        elapsed: snapshot.elapsed,
        steps_per_second,
        episodes_per_minute,
        progress_fraction,
        eta,
        stalled: snapshot.status == TrainerStatus::Running
            && now.saturating_duration_since(last_progress_at)
                >= std::time::Duration::from_secs(30),
        alerts: Vec::new(),
    }
}
