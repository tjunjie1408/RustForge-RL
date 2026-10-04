import subprocess
import sys

import pytest
import rustforge as rf


@pytest.mark.parametrize("name", rf.available_envs())
def test_native_factory_and_discovery(name):
    env = rf.make_env(name)
    assert env.reset(42)
    assert env.action_space().kind in ("discrete", "box")


@pytest.mark.parametrize("name", ["CartPole-v1", "CART_POLE", "cart-pole"])
def test_factory_and_training_accept_consistent_names(name):
    env = rf.make_env(name)
    agent = rf.train(name, episodes=1, max_steps=5)
    assert agent.predict(env.reset(42)) in (0, 1)
    native = rf.DQN.train(name, episodes=1, max_steps=5)
    assert native.predict(env.reset(42)) in (0, 1)


def test_pathlike_training_output(tmp_path):
    output = tmp_path / "metrics.csv"
    agent = rf.train("grid-world", episodes=1, max_steps=3, output=output)
    assert agent.train_steps() >= 0
    assert output.read_text().startswith("episode,")


@pytest.mark.parametrize("kwargs", [{"env": "pendulum"}, {"algorithm": "sac"}, {"episodes": 0}, {"max_steps": True}, {"lr": float("nan")}, {"gamma": 1.1}])
def test_invalid_training_request_preserves_existing_output(tmp_path, kwargs):
    output = tmp_path / "keep.csv"
    output.write_text("keep")
    with pytest.raises(ValueError):
        rf.train(output=output, **kwargs)
    assert output.read_text() == "keep"


def test_gym_factory_aliases_and_bad_api():
    env = rf.make("Pendulum-v1")
    assert env.reset(seed=42)[0].shape == (3,)
    with pytest.raises(ValueError, match="api"):
        rf.make_env(api="wrong")
    with pytest.raises(ValueError, match="unknown"):
        rf.make_env("CartPole-v99")


def test_native_discovery_does_not_require_gymnasium():
    code = '''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "gymnasium" or name.startswith("gymnasium."):
        raise ImportError("blocked optional dependency")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import rustforge as rf
assert "cartpole" in rf.available_envs()
assert len(rf.make_env().reset(42)) == 4
try:
    rf.make_env(api="gymnasium")
except ImportError as error:
    assert "rustforge-rl[gym]" in str(error)
else:
    raise AssertionError("optional dependency was silently ignored")
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_nonfinite_observation_is_a_value_error():
    agent = rf.train(episodes=1, max_steps=5)
    with pytest.raises(ValueError, match="finite"):
        agent.predict([float("nan"), 0.0, 0.0, 0.0])
