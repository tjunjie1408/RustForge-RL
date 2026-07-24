//! Robust incremental reader for the persisted DQN CSV v1 format.

use std::fs::{self, File, Metadata};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::metrics::{parse_line, MetricRow, DQN_CSV_V1_HEADER};

const MAX_READ_BYTES_PER_POLL: u64 = 1024 * 1024;
const MAX_CSV_LINE_BYTES: usize = 64 * 1024;

/// Observable state of a persisted-metrics source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonitorSourceState {
    Waiting,
    Following,
    Idle,
    Completed,
    SourceError,
}

/// A typed source activity or problem reported to the monitor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsvDiagnosticKind {
    Attached,
    Disappeared,
    Reappeared,
    Truncated,
    Replaced,
    HeaderMismatch,
    MalformedRow,
    InvalidUtf8,
    IoError,
    Lifecycle,
    Control,
    Persistence,
}

/// One diagnostic emitted by a CSV poll.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsvDiagnostic {
    pub kind: CsvDiagnosticKind,
    pub line: Option<u64>,
    pub message: String,
}

/// Result of one incremental source poll.
#[derive(Clone, Debug, PartialEq)]
pub struct CsvSourcePoll {
    pub rows: Vec<MetricRow>,
    pub state: MonitorSourceState,
    pub reset: bool,
    pub diagnostics: Vec<CsvDiagnostic>,
}

/// Incrementally follows one DQN CSV v1 file.
pub struct CsvSource {
    path: PathBuf,
    offset: u64,
    pending: Vec<u8>,
    next_line: u64,
    header_valid: Option<bool>,
    identity: Option<FileIdentity>,
    anchor_offset: u64,
    anchor: Vec<u8>,
    present: bool,
    seen_file: bool,
    last_io_error: Option<String>,
    discarding_oversized_line: bool,
}

impl CsvSource {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            offset: 0,
            pending: Vec::new(),
            next_line: 1,
            header_valid: None,
            identity: None,
            anchor_offset: 0,
            anchor: Vec::new(),
            present: false,
            seen_file: false,
            last_io_error: None,
            discarding_oversized_line: false,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn poll(&mut self) -> CsvSourcePoll {
        let mut poll = CsvSourcePoll {
            rows: Vec::new(),
            state: MonitorSourceState::Idle,
            reset: false,
            diagnostics: Vec::new(),
        };

        let metadata = match fs::metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if self.present {
                    poll.diagnostics.push(diagnostic(
                        CsvDiagnosticKind::Disappeared,
                        None,
                        "metrics file disappeared",
                    ));
                    self.reset_reader();
                }
                self.present = false;
                self.identity = None;
                self.last_io_error = None;
                poll.state = MonitorSourceState::Waiting;
                return poll;
            }
            Err(error) => {
                self.report_io_error(&mut poll, "read metadata", &error);
                return poll;
            }
        };

        let identity = FileIdentity::from_metadata(&metadata);
        if !self.present {
            let kind = if self.seen_file {
                poll.reset = true;
                CsvDiagnosticKind::Reappeared
            } else {
                CsvDiagnosticKind::Attached
            };
            self.reset_reader();
            poll.diagnostics.push(diagnostic(
                kind,
                None,
                if kind == CsvDiagnosticKind::Attached {
                    "attached to metrics file"
                } else {
                    "metrics file reappeared"
                },
            ));
            self.present = true;
            self.seen_file = true;
            self.identity = Some(identity);
        } else if self.identity.as_ref() != Some(&identity) {
            self.reset_reader();
            self.identity = Some(identity);
            poll.reset = true;
            poll.diagnostics.push(diagnostic(
                CsvDiagnosticKind::Replaced,
                None,
                "metrics file was replaced",
            ));
        } else if metadata.len() < self.offset {
            self.reset_reader();
            poll.reset = true;
            poll.diagnostics.push(diagnostic(
                CsvDiagnosticKind::Truncated,
                None,
                "metrics file was truncated",
            ));
        } else {
            match self.anchor_changed() {
                Ok(true) => {
                    self.reset_reader();
                    poll.reset = true;
                    poll.diagnostics.push(diagnostic(
                        CsvDiagnosticKind::Replaced,
                        None,
                        "previously read metrics content changed",
                    ));
                }
                Ok(false) => {}
                Err(error) => {
                    self.report_io_error(&mut poll, "verify metrics file", &error);
                    return poll;
                }
            }
        }

        let size = metadata.len();
        if size == self.offset {
            self.last_io_error = None;
            poll.state = if self.header_valid == Some(false) {
                MonitorSourceState::SourceError
            } else {
                MonitorSourceState::Idle
            };
            return poll;
        }

        let read_length = (size - self.offset).min(MAX_READ_BYTES_PER_POLL);
        let chunk = match read_range(&self.path, self.offset, read_length) {
            Ok(chunk) => chunk,
            Err(error) => {
                self.report_io_error(&mut poll, "read metrics", &error);
                return poll;
            }
        };
        self.offset += chunk.len() as u64;
        self.pending.extend_from_slice(&chunk);
        self.last_io_error = None;
        if let Err(error) = self.refresh_anchor() {
            self.report_io_error(&mut poll, "record metrics file position", &error);
            return poll;
        }

        let mut severe_error = false;
        if self.discarding_oversized_line {
            if let Some(newline) = self.pending.iter().position(|byte| *byte == b'\n') {
                self.pending.drain(..=newline);
                self.discarding_oversized_line = false;
            } else {
                self.pending.clear();
                poll.state = MonitorSourceState::Idle;
                return poll;
            }
        }
        if !self.pending.contains(&b'\n') && self.pending.len() > MAX_CSV_LINE_BYTES {
            let line_number = self.next_line;
            self.next_line += 1;
            self.pending.clear();
            self.discarding_oversized_line = true;
            if self.header_valid.is_none() {
                self.header_valid = Some(false);
                severe_error = true;
            }
            poll.diagnostics.push(diagnostic(
                CsvDiagnosticKind::MalformedRow,
                Some(line_number),
                "CSV line exceeds the 64 KiB safety limit",
            ));
        }
        if let Some(last_newline) = self.pending.iter().rposition(|byte| *byte == b'\n') {
            let complete: Vec<u8> = self.pending.drain(..=last_newline).collect();
            for raw_line in complete.split(|byte| *byte == b'\n') {
                if raw_line.is_empty() {
                    continue;
                }
                let line_number = self.next_line;
                self.next_line += 1;
                let raw_line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);

                if raw_line.len() > MAX_CSV_LINE_BYTES {
                    if self.header_valid.is_none() {
                        self.header_valid = Some(false);
                        severe_error = true;
                    }
                    poll.diagnostics.push(diagnostic(
                        CsvDiagnosticKind::MalformedRow,
                        Some(line_number),
                        "CSV line exceeds the 64 KiB safety limit",
                    ));
                    continue;
                }

                let line = match std::str::from_utf8(raw_line) {
                    Ok(line) => line,
                    Err(error) => {
                        severe_error = true;
                        poll.diagnostics.push(diagnostic(
                            CsvDiagnosticKind::InvalidUtf8,
                            Some(line_number),
                            format!("invalid UTF-8: {error}"),
                        ));
                        continue;
                    }
                };
                let line = if line_number == 1 {
                    line.strip_prefix('\u{feff}').unwrap_or(line)
                } else {
                    line
                };
                if line.trim().is_empty() {
                    continue;
                }

                match self.header_valid {
                    None => {
                        let valid = line.trim() == DQN_CSV_V1_HEADER;
                        self.header_valid = Some(valid);
                        if !valid {
                            severe_error = true;
                            poll.diagnostics.push(diagnostic(
                                CsvDiagnosticKind::HeaderMismatch,
                                Some(line_number),
                                format!("expected CSV header `{DQN_CSV_V1_HEADER}`"),
                            ));
                        }
                    }
                    Some(false) => {}
                    Some(true) => {
                        if line.split(',').count() != 5 {
                            poll.diagnostics.push(diagnostic(
                                CsvDiagnosticKind::MalformedRow,
                                Some(line_number),
                                "expected exactly five CSV fields",
                            ));
                        } else if let Some(row) = parse_line(line) {
                            if row.reward.is_finite() && row.epsilon.is_finite() {
                                poll.rows.push(row);
                            } else {
                                poll.diagnostics.push(diagnostic(
                                    CsvDiagnosticKind::MalformedRow,
                                    Some(line_number),
                                    "reward and epsilon must be finite",
                                ));
                            }
                        } else {
                            poll.diagnostics.push(diagnostic(
                                CsvDiagnosticKind::MalformedRow,
                                Some(line_number),
                                "invalid DQN CSV v1 metric row",
                            ));
                        }
                    }
                }
            }
        }

        poll.state = if severe_error || self.header_valid == Some(false) {
            MonitorSourceState::SourceError
        } else if poll.rows.is_empty() {
            MonitorSourceState::Idle
        } else {
            MonitorSourceState::Following
        };
        poll
    }

    fn reset_reader(&mut self) {
        self.offset = 0;
        self.pending.clear();
        self.next_line = 1;
        self.header_valid = None;
        self.anchor_offset = 0;
        self.anchor.clear();
        self.discarding_oversized_line = false;
    }

    fn anchor_changed(&self) -> io::Result<bool> {
        if self.anchor.is_empty() {
            return Ok(false);
        }
        let current = read_range(&self.path, self.anchor_offset, self.anchor.len() as u64)?;
        Ok(current != self.anchor)
    }

    fn refresh_anchor(&mut self) -> io::Result<()> {
        const ANCHOR_BYTES: u64 = 256;
        let length = self.offset.min(ANCHOR_BYTES);
        self.anchor_offset = self.offset - length;
        self.anchor = read_range(&self.path, self.anchor_offset, length)?;
        Ok(())
    }

    fn report_io_error(&mut self, poll: &mut CsvSourcePoll, operation: &str, error: &io::Error) {
        let message = format!("failed to {operation}: {error}");
        if self.last_io_error.as_deref() != Some(&message) {
            poll.diagnostics.push(diagnostic(
                CsvDiagnosticKind::IoError,
                None,
                message.clone(),
            ));
            self.last_io_error = Some(message);
        }
        poll.state = MonitorSourceState::SourceError;
    }
}

fn diagnostic(
    kind: CsvDiagnosticKind,
    line: Option<u64>,
    message: impl Into<String>,
) -> CsvDiagnostic {
    CsvDiagnostic {
        kind,
        line,
        message: message.into(),
    }
}

fn read_range(path: &Path, offset: u64, length: u64) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut chunk = Vec::with_capacity(length.min(usize::MAX as u64) as usize);
    file.take(length).read_to_end(&mut chunk)?;
    Ok(chunk)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileIdentity(FileIdentityInner);

#[cfg(windows)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct FileIdentityInner(u64);

#[cfg(unix)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct FileIdentityInner(u64, u64);

#[cfg(not(any(windows, unix)))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct FileIdentityInner(Option<std::time::SystemTime>);

impl FileIdentity {
    #[cfg(windows)]
    fn from_metadata(metadata: &Metadata) -> Self {
        use std::os::windows::fs::MetadataExt;
        Self(FileIdentityInner(metadata.creation_time()))
    }

    #[cfg(unix)]
    fn from_metadata(metadata: &Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;
        Self(FileIdentityInner(metadata.dev(), metadata.ino()))
    }

    #[cfg(not(any(windows, unix)))]
    fn from_metadata(metadata: &Metadata) -> Self {
        Self(FileIdentityInner(metadata.created().ok()))
    }
}
