import numpy as np
import pytest

pytest.importorskip("gymnasium")

import rustforge
from rustforge import _core


def test_gym_env_conformance_cartpole():
    env = rustforge.make("CartPole")
    obs, info = env.reset(seed=11)
    assert env.observation_space.contains(obs), "reset obs not in observation_space"
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert info == {}
        if terminated or truncated:
            obs, info = env.reset(seed=11)


def test_train_dqn_then_drive_gym_env():
    # Train a DQN natively, then drive a Gymnasium-wrapped env with its policy.
    agent = _core.DQN.train("cartpole", episodes=5, max_steps=100)

    env = rustforge.make("CartPole")
    obs, _ = env.reset(seed=2026)
    total_reward = 0.0
    for _ in range(100):
        action = agent.predict([float(x) for x in obs])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    # CartPole pays +1 per surviving step, so any successful rollout is positive.
    assert total_reward > 0.0
