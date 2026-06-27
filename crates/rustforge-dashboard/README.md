# rustforge-dashboard

Live web dashboard for a RustForge training run. It tails a training CSV
(`episode,reward,avg_loss,epsilon,global_step`) and streams reward / loss /
epsilon curves to the browser over WebSocket. The trainer is unchanged — the
dashboard only reads the CSV it already writes.

## Run

```bash
# produce a (growing) CSV, e.g. via the CLI:
cargo run -p rustforge-cli --release -- train --algo dqn --env cartpole --episodes 200 --log target/run.csv

# watch it live:
cargo run -p rustforge-dashboard -- --log target/run.csv
# open http://127.0.0.1:8080
```

Flags: `--log <PATH>` (required), `--port` (default 8080), `--host` (default 127.0.0.1).

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
