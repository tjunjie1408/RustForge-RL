use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rustforge_tui::source::csv::{CsvDiagnosticKind, CsvSource, MonitorSourceState};

fn unique_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rustforge_csv_source_{tag}_{}_{}.csv",
        std::process::id(),
        n
    ))
}

fn append(path: &PathBuf, bytes: &[u8]) {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open source fixture");
    file.write_all(bytes).expect("append source fixture");
}

fn has_diagnostic(
    poll: &rustforge_tui::source::csv::CsvSourcePoll,
    kind: CsvDiagnosticKind,
) -> bool {
    poll.diagnostics.iter().any(|item| item.kind == kind)
}

#[test]
fn missing_file_waits_then_attaches_and_becomes_idle_at_clean_eof() {
    let path = unique_path("waiting");
    fs::remove_file(&path).ok();
    let mut source = CsvSource::new(&path);

    let waiting = source.poll();
    assert_eq!(waiting.state, MonitorSourceState::Waiting);
    assert!(waiting.rows.is_empty());

    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n0,1,0.5,1,10\n",
    )
    .unwrap();
    let following = source.poll();
    assert_eq!(following.state, MonitorSourceState::Following);
    assert_eq!(following.rows.len(), 1);
    assert!(has_diagnostic(&following, CsvDiagnosticKind::Attached));

    let idle = source.poll();
    assert_eq!(idle.state, MonitorSourceState::Idle);
    assert!(idle.rows.is_empty());

    fs::remove_file(path).ok();
}

#[test]
fn accepts_utf8_bom_crlf_blank_lines_and_partial_final_lines() {
    let path = unique_path("format");
    fs::write(
        &path,
        b"\xEF\xBB\xBFepisode,reward,avg_loss,epsilon,global_step\r\n\r\n0,1,NaN,1,10\r\n1,2,0.4,0.9,",
    )
    .unwrap();
    let mut source = CsvSource::new(&path);

    let first = source.poll();
    assert_eq!(first.rows.len(), 1);
    assert_eq!(first.rows[0].avg_loss, None);
    append(&path, b"20\r\n");
    let second = source.poll();
    assert_eq!(second.rows.len(), 1);
    assert_eq!(second.rows[0].episode, 1);
    assert_eq!(second.rows[0].global_step, 20);

    fs::remove_file(path).ok();
}

#[test]
fn reports_header_mismatch_without_inventing_rows() {
    let path = unique_path("header");
    fs::write(&path, b"episode,reward,epsilon\n0,1,1\n").unwrap();
    let mut source = CsvSource::new(&path);

    let poll = source.poll();
    assert_eq!(poll.state, MonitorSourceState::SourceError);
    assert!(poll.rows.is_empty());
    assert!(has_diagnostic(&poll, CsvDiagnosticKind::HeaderMismatch));

    let still_invalid = source.poll();
    assert_eq!(still_invalid.state, MonitorSourceState::SourceError);
    assert!(!has_diagnostic(
        &still_invalid,
        CsvDiagnosticKind::HeaderMismatch
    ));

    fs::remove_file(path).ok();
}

#[test]
fn reports_malformed_rows_but_keeps_valid_rows() {
    let path = unique_path("malformed");
    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\nnot,a,metric,row,here\n0,1,0.5,1,10\n1,2,0.4,0.9,20,extra\n2,NaN,0.3,0.8,30\n",
    )
    .unwrap();
    let mut source = CsvSource::new(&path);

    let poll = source.poll();
    assert_eq!(poll.rows.len(), 1);
    assert_eq!(poll.rows[0].episode, 0);
    assert_eq!(poll.state, MonitorSourceState::Following);
    assert!(has_diagnostic(&poll, CsvDiagnosticKind::MalformedRow));

    fs::remove_file(path).ok();
}

#[test]
fn same_path_replacement_resets_and_rereads_from_the_start() {
    let path = unique_path("replace");
    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n0,1,0.5,1,10\n",
    )
    .unwrap();
    let mut source = CsvSource::new(&path);
    assert_eq!(source.poll().rows.len(), 1);

    fs::remove_file(&path).unwrap();
    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n0,7,0.2,0.5,5\n1,8,0.1,0.4,10\n",
    )
    .unwrap();
    let replaced = source.poll();
    assert!(replaced.reset);
    assert_eq!(replaced.rows.len(), 2);
    assert_eq!(replaced.rows[0].reward, 7.0);
    assert!(has_diagnostic(&replaced, CsvDiagnosticKind::Replaced));

    fs::remove_file(path).ok();
}

#[test]
fn disappearance_and_reappearance_are_reported_once_per_transition() {
    let path = unique_path("reappear");
    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n0,1,0.5,1,10\n",
    )
    .unwrap();
    let mut source = CsvSource::new(&path);
    assert_eq!(source.poll().rows.len(), 1);

    fs::remove_file(&path).unwrap();
    let disappeared = source.poll();
    assert_eq!(disappeared.state, MonitorSourceState::Waiting);
    assert!(has_diagnostic(&disappeared, CsvDiagnosticKind::Disappeared));
    let still_missing = source.poll();
    assert!(!has_diagnostic(
        &still_missing,
        CsvDiagnosticKind::Disappeared
    ));

    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n0,9,0.1,0.5,5\n",
    )
    .unwrap();
    let reappeared = source.poll();
    assert!(reappeared.reset);
    assert_eq!(reappeared.rows.len(), 1);
    assert_eq!(reappeared.rows[0].reward, 9.0);
    assert!(has_diagnostic(&reappeared, CsvDiagnosticKind::Reappeared));

    fs::remove_file(path).ok();
}

#[test]
fn truncation_resets_the_source_and_rereads_the_header() {
    let path = unique_path("truncate");
    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n0,1,0.5,1,10\n1,2,0.4,0.9,20\n",
    )
    .unwrap();
    let mut source = CsvSource::new(&path);
    assert_eq!(source.poll().rows.len(), 2);

    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n0,8,0.2,0.4,5\n",
    )
    .unwrap();
    let reset = source.poll();
    assert!(reset.reset);
    assert_eq!(reset.rows.len(), 1);
    assert_eq!(reset.rows[0].reward, 8.0);
    assert!(has_diagnostic(&reset, CsvDiagnosticKind::Truncated));

    fs::remove_file(path).ok();
}

#[test]
fn invalid_utf8_is_consumed_and_reported_without_retry_flooding() {
    let path = unique_path("utf8");
    fs::write(
        &path,
        b"episode,reward,avg_loss,epsilon,global_step\n\xFF\xFE\n0,1,0.5,1,10\n",
    )
    .unwrap();
    let mut source = CsvSource::new(&path);

    let first = source.poll();
    assert_eq!(first.state, MonitorSourceState::SourceError);
    assert_eq!(first.rows.len(), 1);
    assert!(has_diagnostic(&first, CsvDiagnosticKind::InvalidUtf8));

    let second = source.poll();
    assert_eq!(second.state, MonitorSourceState::Idle);
    assert!(!has_diagnostic(&second, CsvDiagnosticKind::InvalidUtf8));

    fs::remove_file(path).ok();
}
