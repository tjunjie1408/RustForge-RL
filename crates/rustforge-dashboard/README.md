# rustforge-dashboard

Live web dashboard for a RustForge training run. It tails a training CSV
(`episode,reward,avg_loss,epsilon,global_step`) and streams reward / loss /
epsilon curves to the browser over WebSocket. The trainer is unchanged — the
dashboard only reads the CSV it already writes.

## Run

```bash
# one command: train, serve, and open the browser
cargo run -p rustforge-dashboard -- --train dqn --episodes 200

# or just watch a CSV (default: target/cli_train_dqn.csv) — browser opens automatically
cargo run -p rustforge-dashboard
cargo run -p rustforge-dashboard -- --log target/run.csv

# watch a run produced by the CLI directly (algo positional; CSV path is --output):
cargo run -p rustforge-cli --release -- train dqn --env cartpole --episodes 200 --output target/run.csv
```

Flags: `--train [<ALGO>]` (spawn the trainer first; bare `--train` ⇒ `dqn`), `--episodes` (default 200), `--env` (default `cartpole`), `--log <PATH>` (default `target/cli_train_dqn.csv`), `--no-open` (don't open a browser), `--port` (default 8080), `--host` (default 127.0.0.1).

## How it works

- A background task polls the CSV every 250ms (`std::fs`, read-only/shared so the
  trainer keeps writing) and appends parsed rows to an in-memory history +
  `tokio::sync::broadcast`.
- On connect, the WebSocket sends a full history snapshot, then live appends; a
  ~20s Ping heartbeat reaps dead connections; the client auto-reconnects.
- The frontend (vanilla JS + vendored Chart.js, embedded in the binary) renders
  the three charts. Long runs are downsampled to ≤2000 points **preserving each
  bucket's min/max**, so reward spikes / loss explosions stay visible.

## Notes / limitations

- Single run (one `--log` file). Multi-run comparison is future work.
- Truncation is detected by file-size shrink; a same-path restart that grows past
  the old size within one 250ms poll may briefly mis-read (no crash — malformed
  lines are skipped). Restart the dashboard to recover. See the design spec.
- CPU/local by default; bind `--host 0.0.0.0` only on a trusted network (no auth).
