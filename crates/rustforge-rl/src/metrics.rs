//! Training metrics logging — `AgentLogger` trait with CSV backend.
//!
//! Provides a trait-based interface for recording per-episode training metrics
//! (episode index, reward, loss, epsilon, steps) to various backends.
//! The included `CsvLogger` writes metrics incrementally to a CSV file,
//! protected by a `Mutex` for thread safety.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use crate::runtime::persistence::{MetricError, MetricRecord, MetricSink};
use crate::runtime::trainer::{MetricDescriptor, MetricId};

pub const DQN_CSV_V1_HEADER: &str = "episode,reward,avg_loss,epsilon,global_step";

/// A single record of per-episode training metrics.
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    /// Episode index (0-based).
    pub episode: usize,
    /// Total reward accumulated during the episode.
    pub reward: f32,
    /// Average training loss during the episode (NaN if no training occurred).
    pub avg_loss: f32,
    /// Exploration parameter (e.g., epsilon for ε-greedy).
    pub epsilon: f32,
    /// Total environment steps taken so far.
    pub global_step: usize,
}

/// Trait for logging training metrics.
///
/// Implementors must be `Send + Sync` to support future multi-threaded training loops.
pub trait AgentLogger: Send + Sync {
    /// Log a single episode's metrics.
    fn log(&self, metrics: &EpisodeMetrics);

    /// Flush any buffered data to the underlying storage.
    fn flush(&self);
}

/// CSV-based logger that writes metrics incrementally to a file.
///
/// Each call to `log()` appends a single CSV row. The file handle is
/// protected by a `Mutex` to support concurrent logging from multiple threads.
///
/// # CSV Format
///
/// ```text
/// episode,reward,avg_loss,epsilon,global_step
/// 0,15.0,0.12345,1.000,15
/// 1,22.0,0.09876,0.950,37
/// ```
pub struct CsvLogger {
    writer: Mutex<BufWriter<File>>,
}

impl CsvLogger {
    /// Create a new `CsvLogger` that writes to the given path.
    ///
    /// Creates the file (or truncates if it exists) and writes the CSV header.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the file cannot be created.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{DQN_CSV_V1_HEADER}")?;
        writer.flush()?;
        Ok(CsvLogger {
            writer: Mutex::new(writer),
        })
    }
}

/// Non-panicking DQN CSV v1 persistence sink used by integrated runs.
pub struct DqnCsvMetricSink {
    writer: BufWriter<File>,
    reward: MetricId,
    loss: MetricId,
    epsilon: MetricId,
}

impl DqnCsvMetricSink {
    pub fn create(
        path: impl AsRef<Path>,
        descriptors: &[MetricDescriptor],
    ) -> std::io::Result<Self> {
        let metric = |name: &str| {
            descriptors
                .iter()
                .find(|descriptor| descriptor.name == name)
                .map(|descriptor| descriptor.id)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("missing DQN metric descriptor: {name}"),
                    )
                })
        };
        let reward = metric("reward.episode")?;
        let loss = metric("loss.td")?;
        let epsilon = metric("exploration.epsilon")?;
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "{DQN_CSV_V1_HEADER}")?;
        writer.flush()?;
        Ok(Self {
            writer,
            reward,
            loss,
            epsilon,
        })
    }

    fn value(record: &MetricRecord, metric: MetricId) -> Option<f64> {
        record
            .values
            .iter()
            .find(|value| value.metric == metric)
            .map(|value| value.value)
    }
}

impl MetricSink for DqnCsvMetricSink {
    fn emit(&mut self, record: &MetricRecord) -> Result<(), MetricError> {
        let reward = Self::value(record, self.reward).ok_or_else(|| MetricError {
            message: "DQN CSV record is missing reward.episode".into(),
        })?;
        let loss = Self::value(record, self.loss).unwrap_or(f64::NAN);
        let epsilon = Self::value(record, self.epsilon).ok_or_else(|| MetricError {
            message: "DQN CSV record is missing exploration.epsilon".into(),
        })?;
        writeln!(
            self.writer,
            "{},{reward},{loss},{epsilon},{}",
            record.episode, record.global_step
        )
        .map_err(|error| MetricError {
            message: error.to_string(),
        })
    }

    fn flush(&mut self) -> Result<(), MetricError> {
        self.writer.flush().map_err(|error| MetricError {
            message: error.to_string(),
        })
    }
}

impl AgentLogger for CsvLogger {
    fn log(&self, m: &EpisodeMetrics) {
        let mut w = self.writer.lock().expect("CsvLogger mutex poisoned");
        writeln!(
            w,
            "{},{},{},{},{}",
            m.episode, m.reward, m.avg_loss, m.epsilon, m.global_step
        )
        .expect("CsvLogger write failed");
    }

    fn flush(&self) {
        let mut w = self.writer.lock().expect("CsvLogger mutex poisoned");
        w.flush().expect("CsvLogger flush failed");
    }
}

/// A no-op logger that discards all metrics. Useful for testing or when
/// logging is not needed.
pub struct NullLogger;

impl AgentLogger for NullLogger {
    fn log(&self, _metrics: &EpisodeMetrics) {}
    fn flush(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn csv_logger_writes_header_and_rows() {
        let dir = std::env::temp_dir().join("rustforge_test_csv_logger");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_metrics.csv");

        let logger = CsvLogger::new(&path).unwrap();
        logger.log(&EpisodeMetrics {
            episode: 0,
            reward: 15.0,
            avg_loss: 0.123,
            epsilon: 1.0,
            global_step: 15,
        });
        logger.log(&EpisodeMetrics {
            episode: 1,
            reward: 22.5,
            avg_loss: 0.098,
            epsilon: 0.95,
            global_step: 37,
        });
        logger.flush();

        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 data rows");
        assert_eq!(lines[0], "episode,reward,avg_loss,epsilon,global_step");
        assert!(lines[1].starts_with("0,15,"));
        assert!(lines[2].starts_with("1,22.5,"));

        // Cleanup
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn null_logger_does_not_panic() {
        let logger = NullLogger;
        logger.log(&EpisodeMetrics {
            episode: 0,
            reward: 0.0,
            avg_loss: f32::NAN,
            epsilon: 1.0,
            global_step: 0,
        });
        logger.flush();
    }
}
