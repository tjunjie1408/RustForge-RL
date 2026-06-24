# rustforge (Python bindings)

PyO3 bindings to the RustForge RL framework: native environments and a DQN agent,
plus a Gymnasium-compatible adapter.

## Install (development)

```bash
cd crates/rustforge-python
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Unix: source .venv/bin/activate
pip install "maturin>=1.7,<2.0" pytest "gymnasium>=0.29" "numpy>=1.21"
maturin develop
```

## Usage

```python
import rustforge

# Gymnasium-style env
env = rustforge.make("CartPole")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# Train + run a DQN
agent = rustforge.DQN.train("cartpole", episodes=200)
action = agent.predict([float(x) for x in obs])
```

Available env ids: `CartPole`, `GridWorld`, `MountainCar`, `MountainCarContinuous`, `Pendulum`.
`DQN.train` supports the discrete envs: `"cartpole"`, `"gridworld"`, `"mountaincar"`.

## Error handling

Discrete environments (`CartPole`, `GridWorld`, `MountainCar`) validate the
`action` passed to `step`:

- An **out-of-range** action (e.g. `5` when only `0`, `1` are valid) raises
  `ValueError`.
- A **negative** action (e.g. `-1`) raises `OverflowError`, *not* `ValueError`.
  The action is a Rust `usize`, so PyO3 rejects negative integers during
  argument conversion, before the range check runs.

Continuous environments (`Pendulum`, `MountainCarContinuous`) raise
`ValueError` when the action list has the wrong length (they expect length 1).
