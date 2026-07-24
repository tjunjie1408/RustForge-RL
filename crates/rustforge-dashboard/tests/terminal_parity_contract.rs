use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rustforge_dashboard::analytics::{dashboard_stats, downsample_min_max, rolling_average};
use rustforge_dashboard::metrics::{parse_line, DQN_CSV_V1_HEADER};
use rustforge_dashboard::source::csv::CsvSource;

fn unique_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rustforge_terminal_parity_{tag}_{}_{}.csv",
        std::process::id(),
        n
    ))
}

fn append(path: &PathBuf, contents: &str) {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open parity fixture");
    file.write_all(contents.as_bytes())
        .expect("append parity fixture");
}

#[test]
fn dqn_csv_v1_schema_and_snapshot_are_preserved() {
    assert_eq!(
        DQN_CSV_V1_HEADER,
        "episode,reward,avg_loss,epsilon,global_step"
    );

    let path = unique_path("snapshot");
    append(
        &path,
        concat!(
            "episode,reward,avg_loss,epsilon,global_step\n",
            "0,10.0,0.5,1.0,10\n",
            "1,20.0,NaN,0.9,30\n"
        ),
    );

    let mut source = CsvSource::new(&path);
    let snapshot = source.poll();
    assert_eq!(snapshot.rows.len(), 2);
    assert_eq!(snapshot.rows[0].episode, 0);
    assert_eq!(snapshot.rows[1].avg_loss, None);
    assert!(!snapshot.reset);

    fs::remove_file(path).ok();
}

#[test]
fn append_updates_are_incremental_and_partial_lines_are_withheld() {
    let path = unique_path("append");
    append(
        &path,
        "episode,reward,avg_loss,epsilon,global_step\n0,1,0.5,1,5\n",
    );
    let mut source = CsvSource::new(&path);
    assert_eq!(source.poll().rows.len(), 1);

    append(&path, "1,2,0.4,0.9,");
    assert!(source.poll().rows.is_empty());
    append(&path, "10\n");
    let appended = source.poll();
    assert_eq!(appended.rows.len(), 1);
    assert_eq!(appended.rows[0].episode, 1);
    assert_eq!(appended.rows[0].global_step, 10);

    fs::remove_file(path).ok();
}

#[test]
fn terminal_kpis_match_the_replacement_contract() {
    let rows = [
        parse_line("0,10,0.5,1.0,10").unwrap(),
        parse_line("1,30,0.4,0.9,25").unwrap(),
        parse_line("2,20,0.3,0.8,45").unwrap(),
    ];

    let stats = dashboard_stats(&rows, 100).expect("stats for non-empty history");
    assert_eq!(stats.episode, 2);
    assert_eq!(stats.global_step, 45);
    assert_eq!(stats.latest_reward, 20.0);
    assert_eq!(stats.best_reward, 30.0);
    assert_eq!(stats.recent_average_reward, 20.0);
}

#[test]
fn rolling_average_uses_the_current_hundred_episode_default_semantics() {
    let rewards: Vec<f64> = (1..=105).map(f64::from).collect();
    let average = rolling_average(&rewards, 100);

    assert_eq!(average.len(), rewards.len());
    assert_eq!(average[0], 1.0);
    assert_eq!(average[99], 50.5);
    assert_eq!(average[104], 55.5);
}

#[test]
fn min_max_downsampling_preserves_spikes_and_x_order() {
    let points: Vec<(f64, Option<f64>)> = (0..100)
        .map(|x| {
            let y = match x {
                21 => -500.0,
                24 => 1_000.0,
                _ => x as f64,
            };
            (x as f64, Some(y))
        })
        .collect();

    let sampled = downsample_min_max(&points, 20);
    assert!(sampled.len() <= 20);
    assert!(sampled.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    assert!(sampled.iter().any(|point| point.1 == Some(-500.0)));
    assert!(sampled.iter().any(|point| point.1 == Some(1_000.0)));
}

#[test]
fn loss_gaps_remain_gaps() {
    let points = vec![(0.0, Some(1.0)), (1.0, None), (2.0, Some(0.5))];

    assert_eq!(downsample_min_max(&points, 10), points);
}
