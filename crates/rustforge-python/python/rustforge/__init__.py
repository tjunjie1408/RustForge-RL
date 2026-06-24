"""RustForge RL — Python bindings."""
from . import _core
from ._core import (
    DQN,
    CartPole,
    GridWorld,
    MountainCar,
    MountainCarContinuous,
    Pendulum,
    Space,
)

__all__ = [
    "_core",
    "CartPole",
    "GridWorld",
    "MountainCar",
    "MountainCarContinuous",
    "Pendulum",
    "Space",
    "DQN",
]

try:
    from .gym import RustForgeEnv, make  # noqa: F401

    __all__ += ["RustForgeEnv", "make"]
except ImportError:
    # gymnasium not installed; the native bindings still work without the bridge.
    pass
