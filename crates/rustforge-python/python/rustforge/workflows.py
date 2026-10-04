"""Small, typed entry points for native experiments and optional Gymnasium use."""
from __future__ import annotations

import os
from typing import Any, Literal, Optional, Union, TYPE_CHECKING, overload

from . import _core

if TYPE_CHECKING:
    from .gym import RustForgeEnv

NativeEnv = Union[_core.CartPole, _core.GridWorld, _core.MountainCar,
                  _core.MountainCarContinuous, _core.Pendulum]

_ENVIRONMENTS = {
    "cartpole": _core.CartPole,
    "gridworld": _core.GridWorld,
    "mountaincar": _core.MountainCar,
    "mountaincarcontinuous": _core.MountainCarContinuous,
    "pendulum": _core.Pendulum,
}
_ALIASES = {
    "cartpolev1": "cartpole",
    "mountaincarv0": "mountaincar",
    "mountaincarcontinuousv0": "mountaincarcontinuous",
    "pendulumv1": "pendulum",
}


def environment_name(name: str) -> str:
    """Resolve documented aliases; reject typos and unsupported version suffixes."""
    if not isinstance(name, str):
        raise TypeError("environment name must be a string")
    key = name.strip().lower().replace("-", "").replace("_", "")
    key = _ALIASES.get(key, key)
    if key not in _ENVIRONMENTS:
        raise ValueError(f"unknown environment {name!r}; choose from {', '.join(_ENVIRONMENTS)}")
    return key


def available_envs() -> tuple[str, ...]:
    """Canonical factory names, usable without the optional Gymnasium dependency."""
    return tuple(_ENVIRONMENTS)


@overload
def make_env(name: str = "cartpole", *, api: Literal["native"] = "native", **kwargs: Any) -> NativeEnv: ...

@overload
def make_env(name: str = "cartpole", *, api: Literal["gymnasium"], **kwargs: Any) -> "RustForgeEnv": ...

def make_env(name: str = "cartpole", *, api: Literal["native", "gymnasium"] = "native", **kwargs: Any) -> Any:
    """Create a native env, or explicitly request the optional Gymnasium API.

    Native reset/step return observation / (observation, reward, terminated,
    truncated). Gymnasium uses its standard two-/five-element return values.
    Names are case-insensitive; hyphens, underscores and listed Gym ids work.
    """
    key = environment_name(name)
    if api not in ("native", "gymnasium"):
        raise ValueError("api must be 'native' or 'gymnasium'")
    if api == "gymnasium":
        from .gym import RustForgeEnv
        return RustForgeEnv(_ENVIRONMENTS[key](**kwargs))
    return _ENVIRONMENTS[key](**kwargs)


def train(
    env: str = "cartpole",
    *,
    algorithm: str = "dqn",
    episodes: int = 100,
    max_steps: Optional[int] = None,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    double_dqn: bool = True,
    output: Optional[Union[str, os.PathLike[str]]] = None,
) -> _core.DQN:
    """Train native CPU DQN and return an agent with predict(observation).

    Python currently exposes DQN training; the CLI supports the other algorithms
    and GPU execution. Default episode limits are 500 / 100 / 200 for CartPole,
    GridWorld / MountainCar. Training randomness is not fully seed-controlled.
    Existing DQN.train remains available for lower-level callers.
    """
    if not isinstance(algorithm, str) or algorithm.strip().lower() != "dqn":
        raise ValueError("Python training supports only 'dqn'; use the CLI for PPO/A2C/REINFORCE/TD3/SAC")
    key = environment_name(env)
    if key not in ("cartpole", "gridworld", "mountaincar"):
        raise ValueError("DQN requires cartpole, gridworld or mountaincar; continuous environments need another algorithm")
    max_steps = {"cartpole": 500, "gridworld": 100, "mountaincar": 200}[key] if max_steps is None else max_steps
    for name, value in (("episodes", episodes), ("max_steps", max_steps), ("hidden_dim", hidden_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    path = os.fsdecode(os.fspath(output)) if output is not None else None
    return _core.DQN.train(key, episodes=episodes, max_steps=max_steps, hidden_dim=hidden_dim,
                           lr=lr, gamma=gamma, double_dqn=double_dqn, log_path=path)
