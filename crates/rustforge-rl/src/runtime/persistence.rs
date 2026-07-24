//! Metric persistence boundary and deduplicated health tracking.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use smallvec::SmallVec;

use super::event::MetricValue;
use super::trainer::{MetricDescriptor, TrainerMetadata, TrainingOutcome};

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

#[derive(Clone)]
pub struct PersistenceStatus {
    inner: Arc<Mutex<PersistenceSummary>>,
}

impl Default for PersistenceStatus {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistenceStatus {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(PersistenceSummary::complete())),
        }
    }

    pub fn store(&self, summary: PersistenceSummary) {
        *self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = summary;
    }

    pub fn load(&self) -> PersistenceSummary {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
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

#[derive(Clone, Debug, Serialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub metrics_schema: String,
    pub run_id: String,
    pub algorithm: String,
    pub environment: String,
    pub seed: Option<u64>,
    pub target_reward: Option<f64>,
    pub source_config: BTreeMap<String, String>,
    pub metrics: Vec<ManifestMetric>,
    pub started_at_unix_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at_unix_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<ManifestOutcome>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManifestMetric {
    pub id: u16,
    pub name: String,
    pub label: String,
    pub unit: Option<String>,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManifestOutcome {
    pub status: String,
    pub summary: ManifestSummary,
    pub persistence: ManifestPersistenceSummary,
    pub event_delivery_complete: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManifestSummary {
    pub total_steps: u64,
    pub total_episodes: u64,
    pub elapsed_ms: u128,
    pub stop_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManifestPersistenceSummary {
    pub complete: bool,
    pub failures: u64,
    pub first_error: Option<String>,
    pub last_error: Option<String>,
}

impl RunManifest {
    pub fn started(
        metadata: &TrainerMetadata,
        seed: Option<u64>,
        target_reward: Option<f64>,
        source_config: BTreeMap<String, String>,
    ) -> Self {
        Self {
            schema_version: 1,
            metrics_schema: "dqn-csv-v1".into(),
            run_id: metadata.run_id.clone(),
            algorithm: metadata.algorithm.clone(),
            environment: metadata.environment.clone(),
            seed,
            target_reward,
            source_config,
            metrics: metadata.metrics.iter().map(ManifestMetric::from).collect(),
            started_at_unix_ms: unix_time_ms(),
            finished_at_unix_ms: None,
            outcome: None,
        }
    }

    fn finish(&mut self, outcome: &TrainingOutcome) {
        self.finished_at_unix_ms = Some(unix_time_ms());
        self.outcome = Some(ManifestOutcome::from(outcome));
    }
}

impl From<&MetricDescriptor> for ManifestMetric {
    fn from(descriptor: &MetricDescriptor) -> Self {
        Self {
            id: descriptor.id.get(),
            name: descriptor.name.clone(),
            label: descriptor.label.clone(),
            unit: descriptor.unit.clone(),
            kind: format!("{:?}", descriptor.kind).to_lowercase(),
        }
    }
}

impl From<&TrainingOutcome> for ManifestOutcome {
    fn from(outcome: &TrainingOutcome) -> Self {
        Self {
            status: format!("{:?}", outcome.status).to_lowercase(),
            summary: ManifestSummary {
                total_steps: outcome.summary.total_steps,
                total_episodes: outcome.summary.total_episodes,
                elapsed_ms: outcome.summary.elapsed.as_millis(),
                stop_reason: format!("{:?}", outcome.summary.stop_reason).to_lowercase(),
            },
            persistence: ManifestPersistenceSummary {
                complete: outcome.persistence.complete,
                failures: outcome.persistence.failures,
                first_error: outcome.persistence.first_error.clone(),
                last_error: outcome.persistence.last_error.clone(),
            },
            event_delivery_complete: outcome.event_delivery_complete,
            error: outcome.error.clone(),
        }
    }
}

#[derive(Debug)]
pub struct RunArtifacts {
    directory: PathBuf,
    metrics_path: PathBuf,
    manifest_path: PathBuf,
    manifest: RunManifest,
}

impl RunArtifacts {
    pub fn create_default(base: impl AsRef<Path>, manifest: RunManifest) -> io::Result<Self> {
        fs::create_dir_all(base.as_ref())?;
        let prefix = format!(
            "{}-{}",
            unix_time_ms(),
            sanitize_component(&manifest.run_id)
        );
        for suffix in 0..1000_u16 {
            let name = if suffix == 0 {
                prefix.clone()
            } else {
                format!("{prefix}-{suffix}")
            };
            let directory = base.as_ref().join(name);
            match fs::create_dir(&directory) {
                Ok(()) => return Self::initialize(directory, manifest),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a collision-safe run directory",
        ))
    }

    pub fn create_at(
        directory: impl AsRef<Path>,
        overwrite: bool,
        manifest: RunManifest,
    ) -> io::Result<Self> {
        let directory = directory.as_ref();
        if directory.exists() && !overwrite {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("run output already exists: {}", directory.display()),
            ));
        }
        fs::create_dir_all(directory)?;
        Self::initialize(directory.to_path_buf(), manifest)
    }

    fn initialize(directory: PathBuf, manifest: RunManifest) -> io::Result<Self> {
        let artifacts = Self {
            metrics_path: directory.join("metrics.csv"),
            manifest_path: directory.join("manifest.json"),
            directory,
            manifest,
        };
        artifacts.write_manifest(&artifacts.manifest)?;
        Ok(artifacts)
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn metrics_path(&self) -> &Path {
        &self.metrics_path
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn finalize(&self, outcome: &TrainingOutcome) -> io::Result<()> {
        let mut manifest = self.manifest.clone();
        manifest.finish(outcome);
        self.write_manifest(&manifest)
    }

    fn write_manifest(&self, manifest: &RunManifest) -> io::Result<()> {
        let temporary = self.directory.join("manifest.json.tmp");
        let mut file = fs::File::create(&temporary)?;
        serde_json::to_writer_pretty(&mut file, manifest).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        match fs::rename(&temporary, &self.manifest_path) {
            Ok(()) => Ok(()),
            Err(error)
                if self.manifest_path.exists()
                    && matches!(
                        error.kind(),
                        io::ErrorKind::AlreadyExists | io::ErrorKind::PermissionDenied
                    ) =>
            {
                fs::remove_file(&self.manifest_path)?;
                fs::rename(temporary, &self.manifest_path)
            }
            Err(error) => Err(error),
        }
    }
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn sanitize_component(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "run".into()
    } else {
        sanitized
    }
}
