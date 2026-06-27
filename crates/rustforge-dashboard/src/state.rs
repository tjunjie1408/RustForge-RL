//! Shared in-memory history + broadcast, and the background tail task.
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::broadcast;

use crate::metrics::MetricRow;
use crate::tail::CsvTailer;

/// Authoritative history (for snapshots) + a broadcast stream (for live appends).
#[derive(Clone)]
pub struct AppState {
    history: Arc<RwLock<Vec<MetricRow>>>,
    tx: broadcast::Sender<MetricRow>,
}

impl AppState {
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { history: Arc::new(RwLock::new(Vec::new())), tx }
    }

    /// Append to history AND broadcast while holding the write lock, so a
    /// concurrent `snapshot_and_subscribe` cannot interleave (no gap/duplicate).
    pub fn push(&self, row: MetricRow) {
        let mut h = self.history.write().expect("history lock poisoned");
        h.push(row.clone());
        let _ = self.tx.send(row); // Err only if there are no receivers
    }

    /// Reset history after a truncation/replace.
    pub fn clear(&self) {
        self.history.write().expect("history lock poisoned").clear();
    }

    /// Clone the current history AND subscribe under the read lock, so the
    /// returned receiver sees exactly the rows pushed AFTER this snapshot.
    pub fn snapshot_and_subscribe(&self) -> (Vec<MetricRow>, broadcast::Receiver<MetricRow>) {
        let guard = self.history.read().expect("history lock poisoned");
        let rx = self.tx.subscribe();
        (guard.clone(), rx)
    }
}

/// Background task: poll the CSV every 250ms, reset on truncation, push new rows.
pub fn spawn_tail_task(state: AppState, log_path: PathBuf) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tailer = CsvTailer::new(log_path);
        let mut interval = tokio::time::interval(Duration::from_millis(250));
        loop {
            interval.tick().await;
            let result = tailer.poll(); // small sync fs read on a 250ms cadence
            if result.reset {
                state.clear();
            }
            for r in result.rows {
                state.push(r);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(ep: u64) -> MetricRow {
        MetricRow { episode: ep, reward: ep as f32, avg_loss: Some(0.1), epsilon: 0.5, global_step: ep * 10 }
    }

    #[tokio::test]
    async fn push_appends_history_and_broadcasts() {
        let s = AppState::new(16);
        let (snap, mut rx) = s.snapshot_and_subscribe();
        assert!(snap.is_empty());
        s.push(row(1));
        assert_eq!(rx.recv().await.unwrap().episode, 1);
        let (snap2, _) = s.snapshot_and_subscribe();
        assert_eq!(snap2.len(), 1);
    }

    #[tokio::test]
    async fn snapshot_then_streams_subsequent() {
        let s = AppState::new(16);
        s.push(row(1));
        s.push(row(2));
        let (snap, mut rx) = s.snapshot_and_subscribe();
        assert_eq!(snap.len(), 2);
        s.push(row(3));
        assert_eq!(rx.recv().await.unwrap().episode, 3);
    }

    #[test]
    fn clear_empties_history() {
        let s = AppState::new(16);
        s.push(row(1));
        s.clear();
        let (snap, _) = s.snapshot_and_subscribe();
        assert!(snap.is_empty());
    }

    #[tokio::test]
    async fn tail_task_streams_appended_rows() {
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static C: AtomicU64 = AtomicU64::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("rf_state_{}_{}.csv", std::process::id(), n));
        std::fs::write(&p, "episode,reward,avg_loss,epsilon,global_step\n").unwrap();

        let state = AppState::new(64);
        let (_snap, mut rx) = state.snapshot_and_subscribe();
        let handle = spawn_tail_task(state.clone(), p.clone());

        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
            f.write_all(b"0,1.0,0.5,0.9,10\n").unwrap();
        }

        let got = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("timed out waiting for row")
            .expect("recv");
        assert_eq!(got.episode, 0);

        handle.abort();
        std::fs::remove_file(&p).ok();
    }
}
