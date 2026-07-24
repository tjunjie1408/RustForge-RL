//! Metric persistence boundary and deduplicated health tracking.

use std::fmt;

use smallvec::SmallVec;

use super::event::MetricValue;

#[derive(Clone, Debug, PartialEq)]
pub struct MetricRecord {
    pub episode: u64,
    pub global_step: u64,
    pub values: SmallVec<[MetricValue; 8]>,
}

pub trait MetricSink: Send {
    fn emit(&mut self, record: &MetricRecord) -> Result<(), MetricError>;
    fn flush(&mut self) -> Result<(), MetricError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricError {
    pub message: String,
}

impl fmt::Display for MetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MetricError {}

pub struct NullMetricSink;

impl MetricSink for NullMetricSink {
    fn emit(&mut self, _record: &MetricRecord) -> Result<(), MetricError> {
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MetricError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistenceHealth {
    Healthy,
    Degraded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistenceFailure {
    pub message: String,
    pub failures: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistenceRecovery {
    pub failures: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistenceEvent {
    Failed(PersistenceFailure),
    Recovered(PersistenceRecovery),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistenceSummary {
    pub complete: bool,
    pub failures: u64,
    pub first_error: Option<String>,
    pub last_error: Option<String>,
}

impl PersistenceSummary {
    pub fn complete() -> Self {
        Self {
            complete: true,
            failures: 0,
            first_error: None,
            last_error: None,
        }
    }
}

pub struct PersistenceTracker {
    health: PersistenceHealth,
    failures: u64,
    first_error: Option<String>,
    last_error: Option<String>,
}

impl PersistenceTracker {
    pub fn new() -> Self {
        Self {
            health: PersistenceHealth::Healthy,
            failures: 0,
            first_error: None,
            last_error: None,
        }
    }

    pub fn health(&self) -> PersistenceHealth {
        self.health
    }

    pub fn record_failure(&mut self, message: impl Into<String>) -> Option<PersistenceEvent> {
        let message = message.into();
        self.failures += 1;
        self.first_error.get_or_insert_with(|| message.clone());
        self.last_error = Some(message.clone());
        if self.health == PersistenceHealth::Degraded {
            return None;
        }
        self.health = PersistenceHealth::Degraded;
        Some(PersistenceEvent::Failed(PersistenceFailure {
            message,
            failures: self.failures,
        }))
    }

    pub fn record_recovered(&mut self) -> Option<PersistenceEvent> {
        if self.health == PersistenceHealth::Healthy {
            return None;
        }
        self.health = PersistenceHealth::Healthy;
        Some(PersistenceEvent::Recovered(PersistenceRecovery {
            failures: self.failures,
        }))
    }

    pub fn summary(&self) -> PersistenceSummary {
        PersistenceSummary {
            complete: self.failures == 0,
            failures: self.failures,
            first_error: self.first_error.clone(),
            last_error: self.last_error.clone(),
        }
    }
}

impl Default for PersistenceTracker {
    fn default() -> Self {
        Self::new()
    }
}
