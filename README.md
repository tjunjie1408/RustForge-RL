# RustForge RL

Reinforcement learning in Rust, with native Python bindings and a terminal training
console. CPU and optional wgpu backends support DQN/Double DQN, PPO, A2C,
REINFORCE, TD3 and SAC. Python currently exposes native environments and CPU DQN.

[Python](#python) · [CLI](#cli) · [Development](#development) · [Status](#status)

## CLI

Build/install from this checkout with Rust 1.75 or newer:

```bash
cargo install --path crates/rustforge-cli --locked
rustforge train dqn -n 10
rustforge train sac -n 10 -o sac.jsonl
rustforge run ppo -e pendulum -n 10
```

`train` runs without a terminal; `run` opens the live console. Aliases are `fit`
and `live`. `-e`, `-n`, `-o`, `-d` mean environment, episodes, output and device.
Names are case-insensitive. With no environment flag, SAC/TD3 choose Pendulum;
other algorithms choose CartPole. Explicit unsupported combinations fail early.

| Algorithm | CLI environments | Metrics |
| --- | --- | --- |
| DQN / Double DQN | CartPole, GridWorld | CSV |
| PPO | CartPole, Pendulum | JSONL |
| A2C, REINFORCE | CartPole | JSONL |
| TD3, SAC | Pendulum | JSONL |

For coding agents and scripts, inspect a validated plan before running:

```bash
rustforge plan sac
rustforge plan ppo -e pendulum -n 10
rustforge train sac -n 10 -o experiments/sac.jsonl
```

`plan` writes only JSON to stdout, reports resolved defaults and metric names,
and does not start training, require a terminal, open checkpoints or write files.
Create output parent directories yourself. Existing explicit outputs require
`--overwrite`; incompatible flags are rejected before output mutation.

Install with `--features gpu` to enable GPU execution:

```bash
cargo install --path crates/rustforge-cli --features gpu --locked
rustforge train sac -d gpu -n 10 --checkpoint sac.chk
rustforge train sac -d gpu -n 10 --resume sac.chk --checkpoint sac.chk
```

GPU checkpoints restore agent/optimizer state. Environment, replay/rollout and
random streams restart. CPU checkpoint flags are unsupported.

In the console: **Space** pauses/resumes, **q** requests graceful stop, a second
**q** forces stop, **? / F1** opens help, **Tab** changes view, and arrows navigate.
`p` remains a pause alias. In a completed run, Enter exits. Use `--ascii` and
`--no-color` for accessible output.

```bash
rustforge monitor metrics.csv
rustforge watch path/to/sb3/progress.csv
```

The file monitor accepts RustForge DQN CSV and SB3 monitor/progress CSV; it does
not read the generic JSONL training logs.

## Python

The distribution is `rustforge-rl`; the import is `rustforge`. Published wheels
support CPython 3.9+; the convenience API below is available from this checkout.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install "maturin>=1.9,<2.0" "gymnasium>=0.29" pytest
cd crates/rustforge-python
maturin develop
```

```python
import rustforge as rf

print(rf.available_envs())
env = rf.make_env("CartPole-v1")  # native; no Gymnasium dependency
obs = env.reset(42)
agent = rf.train("cartpole", episodes=10, output="dqn.csv")
action = agent.predict(obs)

# Optional Gymnasium API (NumPy observations, standard reset/step tuples):
gym_env = rf.make_env("pendulum", api="gymnasium")
obs, info = gym_env.reset(seed=42)
```

Factory names accept case, hyphens/underscores and documented Gym ids. Native
`DQN.train`, environment classes and Gymnasium `make` remain available. Python
training supports CartPole, GridWorld and MountainCar with DQN; use the CLI for
other algorithms or GPU training. Environment seeding is supported; the Python
DQN training workflow does not promise fully seeded training.

## Development

The dependency direction is `tensor → autograd → nn → rl`, with CLI, TUI and
Python interfaces above RL. Tensor/autograd APIs support Rust experiments;
[examples](docs/tensor-examples.md) show the basic operations.

```bash
cargo test --locked --workspace --all-features --exclude rustforge-python
cargo clippy --locked --workspace --all-targets --all-features --exclude rustforge-python -- -D warnings
cargo fmt --all -- --check
# After maturin develop, from crates/rustforge-python:
pytest -q
```

Adapter-required GPU tests are ignored by default. Run the explicit commands in
[the GPU guide](docs/gpu-development.md) on an adapter-equipped machine.

## Status

GPU implementation stages through 11c are complete. Correctness checks run on
Mesa llvmpipe; physical GPU validation and performance profiling remain pending.
Full experiment-state resume is separate from current agent checkpoints.

The [CPU DQN/SB3 comparison](docs/performance.md) reports throughput and learning
curves under its stated setup; it does not establish GPU performance.

- [Workflow details and compatibility](docs/usability.md)
- [Python bindings](crates/rustforge-python/README.md)
- [GPU implementation and verification](docs/gpu-development.md)
- [Changelog](CHANGELOG.md)
- [Contribution guide](CONTRIBUTING.md)

Licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
