# RustForge vs Stable-Baselines3 — DQN on CartPole

Head-to-head benchmark of RustForge's DQN (via its PyO3 bindings) against
Stable-Baselines3's DQN on CartPole, reporting **speed** (throughput) and
**seed-averaged learning curves**.

## What is actually compared

This is an **end-to-end system comparison**, not an algorithm-kernel comparison.
RustForge runs the *entire* loop — environment, replay, and gradient steps — in
native Rust with zero per-step Python crossings. SB3 steps a Python/Gymnasium
`CartPole-v1` environment through the interpreter and trains a PyTorch model. So
the speed result reflects "native Rust system" vs "Python orchestration + PyTorch",
which is the meaningful real-world comparison.

## Setup

```bash
cd ../../crates/rustforge-python
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Unix: source .venv/bin/activate
pip install -r ../../benchmarks/sb3_comparison/requirements.txt
maturin develop --release    # RELEASE BUILD REQUIRED — debug invalidates speed numbers
```

## Run

```bash
# from the repo root, with the venv active:
RUSTFORGE_RELEASE_ACK=1 python -m benchmarks.sb3_comparison.benchmark         # full: 10 runs, 50k steps
python -m benchmarks.sb3_comparison.benchmark --quick                          # smoke: 1 run, tiny budget
```

Artifacts are written to `results/`: `results.json` (raw), `summary.md` (tables),
`learning_curve.png`.

## Matched configuration

Both frameworks use the SAME algorithm and hyperparameters (pinned from the
RustForge source):

| Knob | Value |
|---|---|
| Environment | CartPole (`"cartpole"` ↔ `CartPole-v1`) |
| Network | single hidden layer, 64 units, ReLU (`net_arch=[64]`) |
| Double DQN | off (RustForge `double_dqn=False`; SB3 DQN is vanilla) |
| Train cadence | 1 gradient step / env step (`train_freq=1`) |
| lr / γ / target-update | 1e-3 / 0.99 / 100 |
| batch / buffer / warmup | 32 / 10,000 / 128 |
| ε schedule | linear 1.0 → 0.05 over 2,000 steps |
| Device | CPU (both) |
| Budget / runs | 50,000 env steps, 10 runs |

## Fairness caveats

- **Time-limit truncation is bootstrapped on both sides.** RustForge stores
  `replay_done = terminated` (not `terminated || truncated`), so the 500-step
  truncation bootstraps the TD target rather than zeroing it; SB3 does the same
  via `handle_timeout_termination`.
- **Seeding asymmetry.** SB3 runs are fully seeded. RustForge's `DQN.train` has no
  seed parameter and uses an unseeded explorer, so its runs vary by inherent RNG —
  the 10 runs capture that variance (shown as the ±1 std band).
- **End-to-end measurement.** The speed number includes each stack's environment
  stepping (native Rust vs Python/Gymnasium), not just the gradient kernels.
- **Minor:** SB3's `target_update_interval` counts env steps while RustForge's
  counts train steps; with `train_freq=1` these differ only by the 128-step warmup.

## Hardware

Results in `results/` were produced on: **Windows 11, AMD Ryzen (AMD64 Family 25), 16 cores, CPU-only (no GPU)**, Python 3.14, PyTorch CPU build.
