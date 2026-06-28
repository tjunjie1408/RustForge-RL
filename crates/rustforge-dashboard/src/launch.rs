//! Helpers for the one-command launch: building the trainer subprocess
//! invocation and computing the browser URL. The pure functions are
//! unit-tested; the side-effecting spawn/open wrappers are exercised by the
//! CLI wiring and manual verification.
use std::path::Path;
use std::process::{Child, Command, Stdio};

use clap::ValueEnum;

/// RL algorithm to train. Mirrors `rustforge-cli`'s positional algo; forwarded
/// as a string so the dashboard stays decoupled from the training crates.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Algo {
    Dqn,
}

impl Algo {
    /// The token the CLI's `train` subcommand expects as its positional algo.
    pub fn as_arg(self) -> &'static str {
        match self {
            Algo::Dqn => "dqn",
        }
    }
}

/// Environment to train on. Mirrors `rustforge-cli`'s `--env`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Env {
    Cartpole,
    Gridworld,
}

impl Env {
    pub fn as_arg(self) -> &'static str {
        match self {
            Env::Cartpole => "cartpole",
            Env::Gridworld => "gridworld",
        }
    }
}

/// Build the argument vector for `rustforge-cli` (after the program and any
/// leading `run -p …` args): `train <algo> --env <env> --episodes <N> --output <log>`.
pub fn train_child_args(algo: Algo, env: Env, episodes: usize, log: &Path) -> Vec<String> {
    vec![
        "train".to_string(),
        algo.as_arg().to_string(),
        "--env".to_string(),
        env.as_arg().to_string(),
        "--episodes".to_string(),
        episodes.to_string(),
        "--output".to_string(),
        log.to_string_lossy().into_owned(),
    ]
}

/// Compute the URL to open in a browser. A wildcard bind address
/// (`0.0.0.0`, `::`, or empty) is not connectable, so fall back to loopback.
pub fn browser_url(host: &str, port: u16) -> String {
    let connect_host = match host {
        "0.0.0.0" | "::" | "" => "127.0.0.1",
        h => h,
    };
    format!("http://{connect_host}:{port}")
}

/// Resolve how to launch the trainer: prefer a `rustforge-cli` binary sitting
/// next to this executable; otherwise fall back to `cargo run -p rustforge-cli --`
/// (reliable in a dev checkout — cargo builds the CLI as needed). Returns the
/// program to run and the leading args that precede the `train …` args.
pub fn resolve_trainer() -> (String, Vec<String>) {
    let exe_name = if cfg!(windows) {
        "rustforge-cli.exe"
    } else {
        "rustforge-cli"
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(exe_name);
            if sibling.is_file() {
                return (sibling.to_string_lossy().into_owned(), Vec::new());
            }
        }
    }
    (
        "cargo".to_string(),
        vec![
            "run".to_string(),
            "-p".to_string(),
            "rustforge-cli".to_string(),
            "--".to_string(),
        ],
    )
}

/// Spawn the trainer as a child process, inheriting stdio so its progress is
/// visible in the same terminal. Returns the child handle so the caller can
/// kill it on shutdown.
pub fn spawn_trainer(algo: Algo, env: Env, episodes: usize, log: &Path) -> std::io::Result<Child> {
    let (program, mut args) = resolve_trainer();
    args.extend(train_child_args(algo, env, episodes, log));
    Command::new(program)
        .args(args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
}

/// Open `url` in the default browser. Failure is non-fatal — warn and continue.
pub fn open_browser(url: &str) {
    if let Err(e) = webbrowser::open(url) {
        eprintln!("warning: could not open a browser ({e}); open {url} manually");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn train_args_for_dqn_cartpole() {
        let args = train_child_args(
            Algo::Dqn,
            Env::Cartpole,
            200,
            &PathBuf::from("target/run.csv"),
        );
        let expected: Vec<&str> = vec![
            "train",
            "dqn",
            "--env",
            "cartpole",
            "--episodes",
            "200",
            "--output",
            "target/run.csv",
        ];
        assert_eq!(args, expected);
    }

    #[test]
    fn train_args_forward_env_and_episodes() {
        let args = train_child_args(Algo::Dqn, Env::Gridworld, 50, &PathBuf::from("a.csv"));
        assert_eq!(args[1], "dqn");
        assert_eq!(args[3], "gridworld");
        assert_eq!(args[5], "50");
        assert_eq!(args.last().unwrap(), "a.csv");
    }

    #[test]
    fn browser_url_loopback_passthrough() {
        assert_eq!(browser_url("127.0.0.1", 8080), "http://127.0.0.1:8080");
    }

    #[test]
    fn browser_url_wildcard_maps_to_loopback() {
        assert_eq!(browser_url("0.0.0.0", 8080), "http://127.0.0.1:8080");
        assert_eq!(browser_url("::", 9000), "http://127.0.0.1:9000");
        assert_eq!(browser_url("", 3000), "http://127.0.0.1:3000");
    }

    #[test]
    fn browser_url_custom_host_passthrough() {
        assert_eq!(browser_url("192.168.1.5", 8080), "http://192.168.1.5:8080");
    }
}
