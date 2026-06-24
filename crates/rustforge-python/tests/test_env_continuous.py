import pytest

from rustforge import _core


def test_pendulum_reset_and_spaces():
    env = _core.Pendulum()
    obs = env.reset(seed=3)
    assert len(obs) == 3
    a = env.action_space()
    assert a.kind == "box"
    assert len(a.low) == 1


def test_pendulum_step_takes_list_action():
    env = _core.Pendulum(max_steps=50)
    env.reset(seed=3)
    obs, reward, terminated, truncated = env.step([0.5])
    assert len(obs) == 3
    assert isinstance(reward, float)


def test_pendulum_wrong_action_len_raises():
    env = _core.Pendulum()
    env.reset(seed=3)
    with pytest.raises(ValueError):
        env.step([0.1, 0.2])


def test_mountaincar_continuous_step():
    env = _core.MountainCarContinuous(max_steps=50)
    obs = env.reset(seed=3)
    assert len(obs) == 2
    obs, reward, terminated, truncated = env.step([1.0])
    assert len(obs) == 2
