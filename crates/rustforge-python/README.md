# RustForge Python

Native environments and CPU DQN, with an optional Gymnasium bridge. Import
`rustforge`; the package distribution is `rustforge-rl` (CPython 3.9+).

```bash
pip install rustforge-rl
pip install "rustforge-rl[gym]"  # optional Gymnasium support
```

To use the new convenience functions from this checkout, activate a virtualenv,
install maturin and run `maturin develop` in this directory.

```python
import rustforge as rf

print(rf.available_envs())
env = rf.make_env("CartPole-v1")
agent = rf.train("cartpole", episodes=10, output="dqn.csv")
action = agent.predict(env.reset(42))

# Optional Gymnasium: float32 NumPy observations and standard tuples.
env = rf.make_env("pendulum", api="gymnasium")
obs, info = env.reset(seed=42)
```

`make_env` defaults to native reset/step tuples. Environment names accept case,
hyphens/underscores and supported Gym ids. Native classes, `DQN.train` and
Gymnasium `make` remain available. Python DQN supports CartPole, GridWorld and
MountainCar; use the CLI for other algorithms and GPU training. Episode-limit
defaults in `train` are 500, 100 and 200 respectively. Training randomness is not
fully seed-controlled.

`output` accepts strings or pathlib paths. Invalid training settings raise
`ValueError` before opening output; training/log I/O failures raise `RuntimeError`.
Native negative discrete actions raise `OverflowError`; out-of-range actions and
invalid action dimensions raise `ValueError`.

[Workflow details](../../docs/usability.md) · [Project README](../../README.md)
