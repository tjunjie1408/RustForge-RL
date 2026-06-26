"""Framework-driving runners: train RustForge DQN and SB3 DQN, time the call,
return a RunResult. Heavy deps (rustforge, stable_baselines3) are imported lazily
so analysis/config stay importable without them."""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

from benchmarks.sb3_comparison import analysis, config


@dataclass
class RunResult:
    framework: str
    seed: int
    train_seconds: float
    total_steps: int
    curve: list  # list[tuple[int, float]]


def _ramdisk_tmp_csv() -> str:
    """Create a temp .csv on a RAM-backed dir when available (Linux /dev/shm),
    else the system temp dir. Returns the path; caller deletes it."""
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, path = tempfile.mkstemp(suffix=".csv", dir=tmpdir)
    os.close(fd)
    return path


def run_rustforge(run_idx: int) -> RunResult:
    """Train RustForge DQN on CartPole natively; parse its per-episode CSV.

    NOTE: DQN.train has no seed parameter and unseeded exploration, so `run_idx`
    is only a run counter — runs vary by RustForge's inherent (unseeded) RNG.
    """
    import rustforge

    path = _ramdisk_tmp_csv()
    try:
        t0 = time.perf_counter()
        rustforge.DQN.train(
            "cartpole",
            episodes=config.RUSTFORGE_EPISODES,
            max_steps=config.MAX_STEPS,
            hidden_dim=config.HIDDEN_DIM,
            lr=config.LR,
            gamma=config.GAMMA,
            double_dqn=False,
            log_path=path,
        )
        train_seconds = time.perf_counter() - t0
        curve = analysis.parse_rustforge_csv(path)
    finally:
        os.remove(path)

    total_steps = curve[-1][0] if curve else 0
    return RunResult("rustforge", run_idx, train_seconds, total_steps, curve)


def _make_sb3_callback():
    """Build an SB3 BaseCallback that records (cumulative_timesteps,
    episode_reward) when a Monitor-wrapped episode finishes. Defined as a
    factory to keep the SB3 import lazy."""
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardCurveCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.curve: list = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    self.curve.append((int(self.num_timesteps), float(ep["r"])))
            return True

    return RewardCurveCallback()


def run_sb3(seed: int) -> RunResult:
    """Train Stable-Baselines3 DQN on CartPole-v1 (CPU) with matched config."""
    import gymnasium as gym
    from stable_baselines3 import DQN as SB3DQN
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(gym.make("CartPole-v1"))
    model = SB3DQN("MlpPolicy", env, **config.sb3_kwargs(seed))
    callback = _make_sb3_callback()

    t0 = time.perf_counter()
    model.learn(total_timesteps=config.STEP_BUDGET, callback=callback, progress_bar=False)
    train_seconds = time.perf_counter() - t0

    total_steps = int(model.num_timesteps)
    return RunResult("sb3", seed, train_seconds, total_steps, callback.curve)
