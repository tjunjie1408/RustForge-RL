"""Pure analysis helpers for the SB3 benchmark (no framework imports)."""
from __future__ import annotations

import csv

Curve = list  # list[tuple[int, float]] — (cumulative_env_step, episode_reward)


def parse_rustforge_csv(path: str) -> Curve:
    """Read a RustForge training CSV into a (global_step, reward) curve."""
    curve: Curve = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            curve.append((int(row["global_step"]), float(row["reward"])))
    return curve


def truncate_curve(curve: Curve, max_step: int) -> Curve:
    """Keep only points whose cumulative step is <= max_step."""
    return [(s, r) for (s, r) in curve if s <= max_step]
