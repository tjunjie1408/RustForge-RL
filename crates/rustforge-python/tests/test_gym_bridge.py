import numpy as np
import pytest

pytest.importorskip("gymnasium")

import rustforge


def test_make_cartpole_spaces():
    env = rustforge.make("CartPole")
    assert int(env.action_space.n) == 2
    assert env.observation_space.shape == (4,)


def test_cartpole_reset_returns_array_and_info():
    env = rustforge.make("CartPole")
    obs, info = env.reset(seed=7)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert info == {}


def test_cartpole_step_five_tuple():
    env = rustforge.make("CartPole")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info == {}


def test_pendulum_continuous_step():
    env = rustforge.make("Pendulum")
    assert env.action_space.shape == (1,)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert obs.shape == (3,)


def test_unknown_env_id_raises():
    with pytest.raises(ValueError):
        rustforge.make("Breakout")
