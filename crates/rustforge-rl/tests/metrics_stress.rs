//! Stress / high-contention integration test for `CsvLogger` thread safety.
//!
//! 16 threads × 100 episodes = 1 600 rows written concurrently.

use rustforge_rl::metrics::{AgentLogger, CsvLogger, EpisodeMetrics};
use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;
use std::thread;

const NUM_THREADS: usize = 16;
const EPISODES_PER_THREAD: usize = 100;
const TOTAL_ROWS: usize = NUM_THREADS * EPISODES_PER_THREAD; // 1600
const EXPECTED_FIELDS: usize = 5; // episode,reward,avg_loss,epsilon,global_step

#[test]
fn test_csv_logger_high_contention() {
    // -- setup: unique file in temp dir -----------------------------------
    let dir = std::env::temp_dir().join("rustforge_metrics_stress");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!(
        "stress_{}.csv",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    // Wrap cleanup in a guard so the file is removed even on panic.
    struct Cleanup(std::path::PathBuf, std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).ok();
            std::fs::remove_dir(&self.1).ok();
        }
    }
    let _cleanup = Cleanup(path.clone(), dir.clone());

    let logger = Arc::new(CsvLogger::new(&path).unwrap());
    let mut handles = Vec::with_capacity(NUM_THREADS);

    // -- spawn 16 threads, each writing 100 episodes ----------------------
    // Thread `t` writes episodes `t*100 .. (t+1)*100`.
    // This gives every (thread, episode) pair a globally unique episode number,
    // making it straightforward to verify no rows are lost or duplicated.
    for t in 0..NUM_THREADS {
        let logger = Arc::clone(&logger);
        handles.push(thread::spawn(move || {
            let base = t * EPISODES_PER_THREAD;
            for i in 0..EPISODES_PER_THREAD {
                let episode = base + i;
                logger.log(&EpisodeMetrics {
                    episode,
                    reward: episode as f32 * 0.1,
                    avg_loss: 1.0 / (episode as f32 + 1.0),
                    epsilon: 1.0 - (t as f32 / NUM_THREADS as f32),
                    global_step: episode * 10,
                });
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread panicked");
    }
    logger.flush();

    // -- read back the CSV ------------------------------------------------
    let mut content = String::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_string(&mut content)
        .unwrap();

    let lines: Vec<&str> = content.trim().lines().collect();

    // -- assertion 1: exactly 1600 data rows + 1 header = 1601 lines ------
    assert_eq!(
        lines.len(),
        TOTAL_ROWS + 1,
        "expected {} lines (1 header + {} data rows), got {}",
        TOTAL_ROWS + 1,
        TOTAL_ROWS,
        lines.len()
    );

    // -- assertion 2: header is well-formed --------------------------------
    assert_eq!(
        lines[0], "episode,reward,avg_loss,epsilon,global_step",
        "unexpected CSV header"
    );

    // -- assertion 3: no malformed rows & every expected episode appears ----
    let mut seen_episodes = HashSet::with_capacity(TOTAL_ROWS);

    for (line_idx, line) in lines.iter().enumerate().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            EXPECTED_FIELDS,
            "malformed row at line {} (expected {} fields, got {}): {:?}",
            line_idx + 1,
            EXPECTED_FIELDS,
            fields.len(),
            line
        );

        let episode: usize = fields[0].parse().unwrap_or_else(|e| {
            panic!(
                "non-integer episode at line {}: {:?} ({})",
                line_idx + 1,
                fields[0],
                e
            )
        });

        // Validate that all remaining fields parse as expected types.
        for (col, field) in fields.iter().enumerate().skip(1) {
            match col {
                1..=3 => {
                    field.parse::<f32>().unwrap_or_else(|e| {
                        panic!(
                            "non-float field at line {} col {}: {:?} ({})",
                            line_idx + 1,
                            col,
                            field,
                            e
                        )
                    });
                }
                4 => {
                    field.parse::<usize>().unwrap_or_else(|e| {
                        panic!(
                            "non-integer global_step at line {} col {}: {:?} ({})",
                            line_idx + 1,
                            col,
                            field,
                            e
                        )
                    });
                }
                _ => unreachable!(),
            }
        }

        let is_new = seen_episodes.insert(episode);
        assert!(
            is_new,
            "duplicate episode {} found at line {}",
            episode,
            line_idx + 1
        );
    }

    // -- assertion 4: every expected episode number is present -------------
    for expected in 0..TOTAL_ROWS {
        assert!(
            seen_episodes.contains(&expected),
            "missing episode {}",
            expected
        );
    }
}
