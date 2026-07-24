use rustforge_tui::system::{format_bytes, SystemSampler};

#[test]
fn byte_formatting_is_stable_for_terminal_display() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(1024), "1.0 KiB");
    assert_eq!(format_bytes(128 * 1024 * 1024), "128.0 MiB");
}

#[test]
fn sampler_returns_available_fields_without_panicking() {
    let mut sampler = SystemSampler::new();
    let snapshot = sampler.sample();
    assert!(!snapshot.architecture.is_empty());
    assert!(snapshot.logical_cpus >= 1);
    if let (Some(used), Some(total)) = (snapshot.used_memory_bytes, snapshot.total_memory_bytes) {
        assert!(used <= total);
    }
}
