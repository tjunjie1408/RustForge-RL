import pytest

from rustforge import _core


def test_mountaincar_reset_and_spaces():
    env = _core.MountainCar()
    obs = env.reset(seed=1)
    assert len(obs) == 2
    assert env.action_space().n == 3


def test_mountaincar_step():
    env = _core.MountainCar(max_steps=50)
    env.reset(seed=1)
    obs, reward, terminated, truncated = env.step(2)  # Right
    assert len(obs) == 2
    assert reward == -1.0


def test_mountaincar_invalid_action_raises():
    env = _core.MountainCar()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(7)
