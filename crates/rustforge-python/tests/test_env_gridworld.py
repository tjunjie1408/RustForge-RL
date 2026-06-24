import pytest

from rustforge import _core


def test_gridworld_reset_and_spaces():
    env = _core.GridWorld()
    obs = env.reset(seed=1)
    assert len(obs) == 2
    assert env.action_space().kind == "discrete"
    assert env.action_space().n == 4
    assert env.observation_space().kind == "box"


def test_gridworld_step():
    env = _core.GridWorld()
    env.reset(seed=1)
    obs, reward, terminated, truncated = env.step(1)  # Down
    assert len(obs) == 2
    assert isinstance(reward, float)


def test_gridworld_invalid_action_raises():
    env = _core.GridWorld()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(9)
