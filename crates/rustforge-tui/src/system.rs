//! Bounded-cadence system and current-process sampling.

use sysinfo::{get_current_pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SystemSnapshot {
    pub os: Option<String>,
    pub architecture: String,
    pub logical_cpus: usize,
    pub process_cpu_percent: Option<f32>,
    pub process_memory_bytes: Option<u64>,
    pub used_memory_bytes: Option<u64>,
    pub total_memory_bytes: Option<u64>,
}

pub struct SystemSampler {
    system: System,
    pid: Option<sysinfo::Pid>,
}

impl SystemSampler {
    pub fn new() -> Self {
        Self {
            system: System::new_all(),
            pid: get_current_pid().ok(),
        }
    }

    pub fn sample(&mut self) -> SystemSnapshot {
        self.system.refresh_memory();
        self.system.refresh_cpu_usage();
        if let Some(pid) = self.pid {
            self.system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                false,
                ProcessRefreshKind::nothing().with_cpu().with_memory(),
            );
        }
        let process = self.pid.and_then(|pid| self.system.process(pid));
        SystemSnapshot {
            os: System::long_os_version().or_else(System::name),
            architecture: std::env::consts::ARCH.to_owned(),
            logical_cpus: self.system.cpus().len().max(1),
            process_cpu_percent: process.map(|process| process.cpu_usage()),
            process_memory_bytes: process.map(|process| process.memory()),
            used_memory_bytes: Some(self.system.used_memory()),
            total_memory_bytes: Some(self.system.total_memory()),
        }
    }
}

impl Default for SystemSampler {
    fn default() -> Self {
        Self::new()
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes / KIB)
    } else {
        format!("{} B", bytes as u64)
    }
}
