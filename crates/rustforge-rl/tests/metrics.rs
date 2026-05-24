//! Integration tests for the metrics logging system.

use rustforge_rl::metrics::{AgentLogger, CsvLogger, EpisodeMetrics};
use std::io::Read;

#[test]
fn csv_logger_roundtrip() {
    let dir = std::env::temp_dir().join("rustforge_metrics_integration");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("integration_test.csv");

    let logger = CsvLogger::new(&path).unwrap();

    // Log 5 episodes
    for i in 0..5 {
        logger.log(&EpisodeMetrics {
            episode: i,
            reward: i as f32 * 10.0,
            avg_loss: 1.0 / (i as f32 + 1.0),
            epsilon: 1.0 - i as f32 * 0.2,
            global_step: i * 100,
        });
    }
    logger.flush();

    // Read back and verify
    let mut content = String::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_string(&mut content)
        .unwrap();

    let lines: Vec<&str> = content.trim().lines().collect();
    assert_eq!(lines.len(), 6, "1 header + 5 data rows");
    assert_eq!(lines[0], "episode,reward,avg_loss,epsilon,global_step");

    // Verify first data row
    let fields: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(fields[0], "0");
    assert_eq!(fields[4], "0");

    // Verify last data row
    let fields: Vec<&str> = lines[5].split(',').collect();
    assert_eq!(fields[0], "4");
    assert_eq!(fields[4], "400");

    // Cleanup
    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&dir).ok();
}

#[test]
fn csv_logger_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let dir = std::env::temp_dir().join("rustforge_metrics_thread");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thread_test.csv");

    let logger = Arc::new(CsvLogger::new(&path).unwrap());
    let mut handles = vec![];

    // Spawn 4 threads, each logging 10 episodes
    for t in 0..4 {
        let logger = Arc::clone(&logger);
        handles.push(thread::spawn(move || {
            for i in 0..10 {
                logger.log(&EpisodeMetrics {
                    episode: t * 10 + i,
                    reward: i as f32,
                    avg_loss: 0.1,
                    epsilon: 0.5,
                    global_step: i,
                });
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
    logger.flush();

    // Read back and verify all 40 rows present
    let mut content = String::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_string(&mut content)
        .unwrap();

    let lines: Vec<&str> = content.trim().lines().collect();
    assert_eq!(lines.len(), 41, "1 header + 40 data rows from 4 threads");

    // Cleanup
    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&dir).ok();
}
