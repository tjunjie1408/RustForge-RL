"""Gymnasium-compatible wrappers around RustForge native environments."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Gymnasium bridge requires the 'gymnasium' package. "
        "Install it with: pip install 'rustforge[gym]'"
    ) from exc

from . import _core

# Friendly id -> native class.
_REGISTRY = {
    "CartPole": _core.CartPole,
    "GridWorld": _core.GridWorld,
    "MountainCar": _core.MountainCar,
    "MountainCarContinuous": _core.MountainCarContinuous,
    "Pendulum": _core.Pendulum,
}


def _to_gym_space(space: Any) -> "gym.Space":
    if space.kind == "discrete":
        return spaces.Discrete(space.n)
    if space.kind == "box":
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    raise ValueError(f"unsupported space kind: {space.kind!r}")


class RustForgeEnv(gym.Env):
    """Adapts a native RustForge environment to the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(self, native_env: Any):
        self._env = native_env
        self.action_space = _to_gym_space(native_env.action_space())
        self.observation_space = _to_gym_space(native_env.observation_space())
        self._discrete = isinstance(self.action_space, spaces.Discrete)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self._env.reset(seed)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        if self._discrete:
            native_action = int(action)
        else:
            native_action = [float(x) for x in np.asarray(action).reshape(-1)]
        obs, reward, terminated, truncated = self._env.step(native_action)
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            {},
        )


def make(env_id: str, **kwargs: Any) -> RustForgeEnv:
    """Create a Gymnasium-wrapped RustForge environment by id.

    Valid ids: 'CartPole', 'GridWorld', 'MountainCar',
    'MountainCarContinuous', 'Pendulum'.
    """
    if env_id not in _REGISTRY:
        raise ValueError(f"unknown env_id {env_id!r}; valid ids: {sorted(_REGISTRY)}")
    native = _REGISTRY[env_id](**kwargs)
    return RustForgeEnv(native)
