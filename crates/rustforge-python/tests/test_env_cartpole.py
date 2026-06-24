import pytest

from rustforge import _core


def test_cartpole_reset_shape():
    env = _core.CartPole()
    obs = env.reset(seed=42)
    assert len(obs) == 4
    assert all(isinstance(x, float) for x in obs)


def test_cartpole_reset_is_deterministic():
    env = _core.CartPole()
    a = env.reset(seed=123)
    b = env.reset(seed=123)
    assert a == b


def test_cartpole_step_returns_4_tuple_without_info():
    env = _core.CartPole()
    env.reset(seed=0)
    obs, reward, terminated, truncated = env.step(1)
    assert len(obs) == 4
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_cartpole_invalid_action_raises():
    env = _core.CartPole()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(5)


def test_cartpole_spaces():
    env = _core.CartPole()
    a = env.action_space()
    o = env.observation_space()
    assert a.kind == "discrete"
    assert a.n == 2
    assert o.kind == "box"
    assert len(o.low) == 4
    assert len(o.high) == 4


def test_cartpole_negative_action_raises_overflow():
    # Discrete actions are a Rust usize; PyO3 rejects negative ints during
    # argument conversion, so a negative action is OverflowError, not ValueError.
    env = _core.CartPole()
    env.reset(seed=0)
    with pytest.raises(OverflowError):
        env.step(-1)
