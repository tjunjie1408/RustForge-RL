"""Matched configuration for the RustForge-vs-SB3 DQN benchmark.

All values are pinned from the RustForge source so both frameworks train the
SAME algorithm under the SAME hyperparameters:
- batch_size / buffer_size / warmup: crates/rustforge-rl/src/agent/dqn_train.rs
- epsilon schedule (linear 1.0->0.05 over 2000 steps): crates/rustforge-rl/src/agent/epsilon_greedy.rs
"""
from __future__ import annotations

import os

# --- Shared training budget -------------------------------------------------
STEP_BUDGET: int = 50_000          # env steps SB3 trains for; RustForge curves truncated here
MAX_STEPS: int = 500               # per-episode cap (CartPole-v1 style)
SEEDS: list[int] = list(range(10)) # 10 runs

# RustForge DQN.train is EPISODE-budgeted (no step budget / no resume). This count is
# calibrated in Task 7 so a run reaches >= STEP_BUDGET env steps; curves are then
# truncated to STEP_BUDGET. 300 is the starting estimate.
RUSTFORGE_EPISODES: int = 300

# --- Matched hyperparameters ------------------------------------------------
HIDDEN_DIM: int = 64
LR: float = 1e-3
GAMMA: float = 0.99
BATCH_SIZE: int = 32
BUFFER_SIZE: int = 10_000
LEARNING_STARTS: int = 128
TARGET_UPDATE: int = 100
EPS_INITIAL: float = 1.0
EPS_FINAL: float = 0.05
EPS_DECAY_STEPS: int = 2_000

# --- Evaluation -------------------------------------------------------------
SOLVED_THRESHOLD: float = 475.0    # CartPole-v1 reward threshold
SOLVED_WINDOW: int = 100           # episodes

# --- Result paths -----------------------------------------------------------
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(_PKG_DIR, "results")
RESULTS_JSON: str = os.path.join(RESULTS_DIR, "results.json")
SUMMARY_MD: str = os.path.join(RESULTS_DIR, "summary.md")
PLOT_PNG: str = os.path.join(RESULTS_DIR, "learning_curve.png")


def sb3_kwargs(seed: int) -> dict:
    """Stable-Baselines3 DQN constructor kwargs matched to RustForge's DQN."""
    return {
        "learning_rate": LR,
        "buffer_size": BUFFER_SIZE,
        "learning_starts": LEARNING_STARTS,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": TARGET_UPDATE,
        "exploration_initial_eps": EPS_INITIAL,
        "exploration_final_eps": EPS_FINAL,
        "exploration_fraction": EPS_DECAY_STEPS / STEP_BUDGET,
        "policy_kwargs": {"net_arch": [HIDDEN_DIM]},
        "device": "cpu",
        "seed": seed,
        "verbose": 0,
    }
