//! Incremental CSV reader: yields rows appended since the last `poll()`.
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use crate::metrics::{parse_line, MetricRow};

/// Tracks a byte offset into a CSV file and a buffered partial trailing line.
pub struct CsvTailer {
    path: PathBuf,
    offset: u64,
    pending: String,
}

/// Result of one `poll()`: new complete rows, and whether a truncation reset happened.
pub struct PollResult {
    pub rows: Vec<MetricRow>,
    pub reset: bool,
}

impl CsvTailer {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            offset: 0,
            pending: String::new(),
        }
    }

    /// Read newly appended bytes (read-only, shared access), return complete rows.
    /// Detects truncation by size shrink and re-reads from the start.
    pub fn poll(&mut self) -> PollResult {
        let mut reset = false;

        let size = match fs::metadata(&self.path) {
            Ok(m) => m.len(),
            Err(_) => {
                return PollResult {
                    rows: Vec::new(),
                    reset: false,
                }
            } // file absent
        };

        if size < self.offset {
            self.offset = 0;
            self.pending.clear();
            reset = true;
        }
        if size == self.offset {
            return PollResult {
                rows: Vec::new(),
                reset,
            };
        }

        let mut file = match fs::File::open(&self.path) {
            Ok(f) => f,
            Err(_) => {
                return PollResult {
                    rows: Vec::new(),
                    reset,
                }
            }
        };
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return PollResult {
                rows: Vec::new(),
                reset,
            };
        }

        let mut chunk = String::new();
        let n = (&mut file)
            .take(size - self.offset)
            .read_to_string(&mut chunk)
            .unwrap_or(0);
        self.offset += n as u64;
        self.pending.push_str(&chunk);

        let mut rows = Vec::new();
        if let Some(last_nl) = self.pending.rfind('\n') {
            let complete = self.pending[..=last_nl].to_string();
            let rest = self.pending[last_nl + 1..].to_string();
            for line in complete.lines() {
                if let Some(row) = parse_line(line) {
                    rows.push(row);
                }
            }
            self.pending = rest;
        }
        PollResult { rows, reset }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_path(tag: &str) -> PathBuf {
        static C: AtomicU64 = AtomicU64::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("rf_tail_{}_{}_{}.csv", tag, std::process::id(), n))
    }

    fn append(path: &PathBuf, s: &str) {
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    #[test]
    fn yields_new_complete_rows_and_withholds_partial() {
        let p = unique_path("yields");
        let _ = fs::remove_file(&p);
        let mut t = CsvTailer::new(&p);

        append(
            &p,
            "episode,reward,avg_loss,epsilon,global_step\n0,1.0,0.5,0.9,10\n",
        );
        let r1 = t.poll();
        assert_eq!(r1.rows.len(), 1);
        assert_eq!(r1.rows[0].episode, 0);
        assert!(!r1.reset);

        append(&p, "1,2.0,0.4,0.8,2"); // no newline yet
        let r2 = t.poll();
        assert!(r2.rows.is_empty());

        append(&p, "0\n"); // completes the line
        let r3 = t.poll();
        assert_eq!(r3.rows.len(), 1);
        assert_eq!(r3.rows[0].episode, 1);

        fs::remove_file(&p).ok();
    }

    #[test]
    fn truncation_resets_and_rereads() {
        let p = unique_path("trunc");
        let _ = fs::remove_file(&p);
        let mut t = CsvTailer::new(&p);
        append(&p, "0,1.0,0.5,0.9,10\n1,2.0,0.4,0.8,20\n");
        assert_eq!(t.poll().rows.len(), 2);

        fs::write(&p, "0,9.0,0.1,0.5,5\n").unwrap(); // smaller -> truncation
        let r = t.poll();
        assert!(r.reset);
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0].reward, 9.0);

        fs::remove_file(&p).ok();
    }

    #[test]
    fn missing_file_yields_empty() {
        let p = unique_path("missing");
        let _ = fs::remove_file(&p);
        let mut t = CsvTailer::new(&p);
        let r = t.poll();
        assert!(r.rows.is_empty() && !r.reset);
    }
}
