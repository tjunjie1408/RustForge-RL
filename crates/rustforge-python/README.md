# rustforge (Python bindings)

PyO3 bindings to the RustForge RL framework: native environments and a DQN agent,
plus a Gymnasium-compatible adapter.

## Install (development)

```bash
cd crates/rustforge-python
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Unix: source .venv/bin/activate
pip install "maturin>=1.7,<2.0" "gymnasium>=0.29" "numpy>=1.21"
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
