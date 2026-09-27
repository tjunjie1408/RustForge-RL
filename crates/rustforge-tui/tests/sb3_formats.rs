//! `rustforge monitor` following Stable-Baselines3 log files.
//!
//! Fixture lines are trimmed from logs written by SB3 2.9 (`Monitor` wrapper and
//! the `csv` logger format).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rustforge_tui::app::{AppMode, AppState};
use rustforge_tui::monitor::apply_source_poll;
use rustforge_tui::source::csv::{CsvDiagnosticKind, CsvSource, MonitorSourceState};
use rustforge_tui::source::format::CsvFormat;

const MONITOR_HEAD: &str =
    "#{\"t_start\": 1790491100.402073, \"env_id\": \"CartPole-v1\"}\nr,l,t\n";

fn unique_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rustforge_sb3_{tag}_{}_{}.csv",
        std::process::id(),
        n
    ))
}

fn append(path: &PathBuf, text: &str) {
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open fixture");
    file.write_all(text.as_bytes()).expect("append fixture");
}

fn monitor_app() -> AppState {
    AppState::new(AppMode::Monitor, 1_000, 100)
}

#[test]
fn follows_an_sb3_monitor_file_episode_by_episode() {
    let path = unique_path("monitor");
    fs::write(&path, format!("{MONITOR_HEAD}18.0,18,5.085702\n")).unwrap();
    let mut source = CsvSource::new(&path);
    let mut app = monitor_app();

    apply_source_poll(&mut app, &mut source);
    assert_eq!(source.format(), Some(CsvFormat::Sb3Monitor));
    assert_eq!(app.source_state(), MonitorSourceState::Following);
    assert_eq!(app.episodes().len(), 1);
    assert_eq!(app.metric_labels().episode_reward, "Episode reward");
    assert_eq!(app.metric_labels().primary_loss, None);
    assert_eq!(app.metric_labels().policy_signal, None);
    let metadata = app.run_metadata();
    assert_eq!(metadata.schema_version.as_deref(), Some("sb3-monitor"));
    assert_eq!(metadata.environment.as_deref(), Some("CartPole-v1"));

    append(&path, "31.0,31,5.101\n12.0,12,5.11");
    apply_source_poll(&mut app, &mut source);
    assert_eq!(app.episodes().len(), 2, "the partial last line waits");

    append(&path, "\n");
    apply_source_poll(&mut app, &mut source);
    let latest = app.latest_episode().unwrap();
    assert_eq!(latest.episode, 2);
    assert_eq!(latest.reward, 12.0);
    assert_eq!(latest.global_step, 18 + 31 + 12);

    fs::remove_file(path).ok();
}

#[test]
fn follows_sb3_progress_through_a_header_rewrite() {
    let path = unique_path("progress");
    // Before `learning_starts`, SB3's DQN has not logged any `train/` keys.
    fs::write(
        &path,
        "rollout/ep_rew_mean,time/total_timesteps,rollout/exploration_rate,time/episodes\n\
         18.0,18,0.943,1\n\
         17.0,34,0.892,2\n",
    )
    .unwrap();
    let mut source = CsvSource::new(&path);
    let mut app = monitor_app();

    apply_source_poll(&mut app, &mut source);
    assert_eq!(source.format(), Some(CsvFormat::Sb3Progress));
    assert_eq!(app.episodes().len(), 2);
    assert_eq!(app.metric_labels().primary_loss, None);
    assert_eq!(
        app.metric_labels().policy_signal.as_deref(),
        Some("Exploration / epsilon")
    );

    // New keys make SB3 rewrite the whole file with a wider header.
    fs::write(
        &path,
        "rollout/ep_rew_mean,time/total_timesteps,rollout/exploration_rate,time/episodes,train/loss\n\
         18.0,18,0.943,1,\n\
         17.0,34,0.892,2,\n\
         21.5,1200,0.05,53,0.125\n",
    )
    .unwrap();
    apply_source_poll(&mut app, &mut source);
    assert!(app
        .activity()
        .iter()
        .any(|item| item.kind == CsvDiagnosticKind::Replaced));
    assert_eq!(app.episodes().len(), 3, "rows are re-read, not duplicated");
    assert_eq!(app.metric_labels().primary_loss.as_deref(), Some("Loss"));
    let latest = app.latest_episode().unwrap();
    assert_eq!(latest.episode, 52);
    assert_eq!(latest.primary_loss, Some(0.125));
    assert_eq!(latest.global_step, 1200);
    assert_eq!(
        app.run_metadata().schema_version.as_deref(),
        Some("sb3-progress")
    );

    fs::remove_file(path).ok();
}

#[test]
fn unsupported_headers_name_the_supported_formats() {
    let path = unique_path("unsupported");
    fs::write(&path, "step,value\n1,2\n").unwrap();
    let mut source = CsvSource::new(&path);

    let poll = source.poll();
    assert_eq!(poll.state, MonitorSourceState::SourceError);
    let mismatch = poll
        .diagnostics
        .iter()
        .find(|item| item.kind == CsvDiagnosticKind::HeaderMismatch)
        .expect("header mismatch");
    assert!(mismatch.message.contains("monitor.csv"));
    assert!(mismatch.message.contains("progress.csv"));
    assert_eq!(source.format(), None);

    fs::remove_file(path).ok();
}
