import pytest

from rustforge import _core


def test_dqn_train_and_predict_cartpole():
    agent = _core.DQN.train("cartpole", episodes=3, max_steps=50)
    action = agent.predict([0.0, 0.0, 0.0, 0.0])
    assert action in (0, 1)
    assert agent.train_steps() >= 0


def test_dqn_train_gridworld():
    agent = _core.DQN.train("gridworld", episodes=2, max_steps=30)
    action = agent.predict([0.0, 0.0])
    assert action in (0, 1, 2, 3)


def test_dqn_unknown_env_raises():
    with pytest.raises(ValueError):
        _core.DQN.train("pong", episodes=1)


def test_dqn_predict_wrong_obs_len_raises():
    agent = _core.DQN.train("cartpole", episodes=1, max_steps=10)
    with pytest.raises(ValueError):
        agent.predict([0.0, 0.0])
