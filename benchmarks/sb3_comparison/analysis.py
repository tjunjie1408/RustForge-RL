"""Pure analysis helpers for the SB3 benchmark (no framework imports)."""
from __future__ import annotations

import csv
import numpy as np

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


def steps_to_solved(curve: Curve, threshold: float, window: int):
    """First cumulative step where the trailing `window`-episode mean reward
    reaches `threshold`. Returns None if never reached (or fewer than `window`
    episodes)."""
    rewards = [r for _, r in curve]
    steps = [s for s, _ in curve]
    for i in range(len(rewards)):
        if i + 1 < window:
            continue
        if sum(rewards[i - window + 1 : i + 1]) / window >= threshold:
            return steps[i]
    return None


def make_step_grid(max_step: int, n: int = 200) -> list:
    """Evenly spaced integer step grid from 0 to max_step (inclusive)."""
    return [int(x) for x in np.linspace(0, max_step, n)]


def aggregate(curves: list, step_grid: list):
    """Interpolate each curve onto step_grid, return (mean, std) per grid point.
    Empty curves are skipped."""
    grid = np.asarray(step_grid, dtype=float)
    rows = []
    for curve in curves:
        if not curve:
            continue
        xs = np.asarray([s for s, _ in curve], dtype=float)
        ys = np.asarray([r for _, r in curve], dtype=float)
        rows.append(np.interp(grid, xs, ys))
    if not rows:
        zeros = [0.0] * len(step_grid)
        return zeros, zeros
    arr = np.vstack(rows)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


def format_summary(speed: dict, solved: dict, step_budget: int, n_runs: int) -> str:
    """Render the speed + steps-to-solved tables as markdown."""
    lines = [
        f"# SB3 Benchmark — DQN on CartPole ({n_runs} runs, {step_budget:,} env-step budget)",
        "",
        "## Speed (training call, CPU)",
        "",
        "| Framework | Train time (s) | Throughput (steps/sec) |",
        "|---|---|---|",
    ]
    for fw in ("rustforge", "sb3"):
        s = speed[fw]
        lines.append(
            f"| {fw} | {s['time_mean']:.2f} ± {s['time_std']:.2f} "
            f"| {s['thru_mean']:,.0f} ± {s['thru_std']:,.0f} |"
        )
    rf, sb = speed["rustforge"]["thru_mean"], speed["sb3"]["thru_mean"]
    if sb > 0:
        lines += ["", f"**RustForge throughput speedup:** {rf / sb:.1f}x"]
    lines += [
        "",
        "## Learning (steps to reach CartPole-v1 solved threshold = 475)",
        "",
        "| Framework | Mean steps to solved | Runs solved |",
        "|---|---|---|",
    ]
    for fw in ("rustforge", "sb3"):
        v = solved[fw]
        mean = "n/a" if v["mean_steps"] is None else f"{v['mean_steps']:,.0f}"
        lines.append(f"| {fw} | {mean} | {v['n_solved']}/{v['n']} |")
    return "\n".join(lines) + "\n"
