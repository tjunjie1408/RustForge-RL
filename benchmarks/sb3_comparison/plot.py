"""Render the seed-averaged learning curve from a benchmark results.json."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_COLORS = {"rustforge": "#d35400", "sb3": "#2980b9"}


def plot_from_results(results_json: str, out_png: str) -> None:
    with open(results_json) as f:
        data = json.load(f)

    grid = np.asarray(data["step_grid"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for fw in ("rustforge", "sb3"):
        agg = data["aggregate"][fw]
        mean = np.asarray(agg["mean"], dtype=float)
        std = np.asarray(agg["std"], dtype=float)
        color = _COLORS.get(fw)
        ax.plot(grid, mean, label=fw, color=color)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("DQN on CartPole — RustForge vs Stable-Baselines3")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
