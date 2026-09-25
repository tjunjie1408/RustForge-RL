# RustForge native terminal console

Ratatui monitor and live training console for RustForge. Monitor mode loads and
follows the existing DQN CSV v1 format:

```text
episode,reward,avg_loss,epsilon,global_step
```

The user-facing entry point is the `rustforge` binary. This crate is a library
consumed by `rustforge-cli`; it does not expose a separate dashboard binary.

## Run

```bash
# Start headless training in one terminal.
cargo run -p rustforge-cli --bin rustforge --release -- \
  train dqn --env cartpole --episodes 200 --output target/run.csv

# Attach the native terminal monitor in another terminal.
cargo run -p rustforge-cli --bin rustforge -- monitor target/run.csv

# Accessibility and optional monitoring metadata.
cargo run -p rustforge-cli --bin rustforge -- monitor target/run.csv \
  --no-color --ascii \
  --target-reward 195 --total-episodes 200
```

The monitor can attach before the file exists, follow a growing file, or open a
completed file. EOF is shown as `IDLE`; it is not treated as proof that training
completed.

## Layout

- **Header:** run identity (`algorithm · environment`, or the monitored file),
  a status badge, and view tabs. Live sessions show the trainer status
  (`RUNNING`, `PAUSED`, `STOPPING`, `STOPPED`, `COMPLETED`, `FAILED`); monitor
  sessions show the source state (`WAITING`, `FOLLOWING`, `IDLE`, ...).
- **Progress line:** progress bar and ETA when the episode total is known,
  followed by steps/s, episodes/min, and elapsed time.
- **Overview:** reward chart (raw points plus a 100-episode rolling average),
  training KPIs with loss and policy-signal sparklines, system resources, and
  the newest events with alerts first.
- **Footer:** key hints that follow the live state (`p pause`/`p resume`,
  `q force stop` after a stop request, `Enter exit` once training ends) and the
  active chart range.

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

`f`, scrolling up, or `Home` freezes the view: charts keep the episode window
that was current at that moment and the event lists hide newer entries, while
KPIs and new data keep updating in the background. The footer shows `FROZEN`
until `f` or `End` resumes following.

Monitor mode intentionally has no pause, resume, stop, or checkpoint controls.

For integrated DQN training with live controls:

```bash
cargo run -p rustforge-cli --bin rustforge -- run dqn \
  --env cartpole --episodes 200 --target-reward 195
```

In live mode, `p` pauses/resumes at a step boundary. The first `q` requests a
graceful episode-boundary stop; a second `q` escalates to force stop after the
current step. Checkpoint controls remain disabled because recoverable DQN
checkpointing is not implemented.
