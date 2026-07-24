# rustforge-dashboard

Transitional native terminal monitor for a RustForge training run. It loads and
follows the existing DQN CSV v1 format:

```text
episode,reward,avg_loss,epsilon,global_step
```

This crate is being migrated to `rustforge-tui`. The current binary is a
read-only Ratatui monitor; it no longer launches a browser or binds a server
port. Live pause/resume/stop controls belong to the later integrated
`rustforge run` path, not CSV attach mode.

## Run

```bash
# Start headless training in one terminal.
cargo run -p rustforge-cli --release -- \
  train dqn --env cartpole --episodes 200 --output target/run.csv

# Attach the native terminal monitor in another terminal.
cargo run -p rustforge-dashboard -- --log target/run.csv

# Accessibility and optional monitoring metadata.
cargo run -p rustforge-dashboard -- \
  --log target/run.csv --no-color --ascii \
  --target-reward 195 --total-episodes 200
```

The monitor can attach before the file exists, follow a growing file, or open a
completed file. EOF is shown as `IDLE`; it is not treated as proof that training
completed.

## Keys

```text
Tab / Shift-Tab  change view
Left / Right     change chart range
Up / Down        scroll
PageUp/PageDown  scroll by page
Home / End       first / latest
f                follow / freeze
t                cycle palette
g                alert settings
?                help
q / Ctrl+C       quit monitor
```

Monitor mode intentionally has no pause, resume, stop, or checkpoint controls.
