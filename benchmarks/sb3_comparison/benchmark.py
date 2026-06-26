"""Orchestrate the RustForge-vs-SB3 DQN benchmark and write artifacts."""
from __future__ import annotations

import argparse
import json
import os
import statistics

from benchmarks.sb3_comparison import analysis, config, runners


def _speed_stats(results: list) -> dict:
    out = {}
    for fw in ("rustforge", "sb3"):
        rs = [r for r in results if r.framework == fw]
        times = [r.train_seconds for r in rs]
        thru = [r.total_steps / r.train_seconds for r in rs if r.train_seconds > 0]
        out[fw] = {
            "time_mean": statistics.fmean(times) if times else 0.0,
            "time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "thru_mean": statistics.fmean(thru) if thru else 0.0,
            "thru_std": statistics.pstdev(thru) if len(thru) > 1 else 0.0,
        }
    return out


def _solved_stats(results: list, step_budget: int) -> dict:
    out = {}
    for fw in ("rustforge", "sb3"):
        rs = [r for r in results if r.framework == fw]
        hits = [
            analysis.steps_to_solved(
                analysis.truncate_curve(r.curve, step_budget),
                config.SOLVED_THRESHOLD,
                config.SOLVED_WINDOW,
            )
            for r in rs
        ]
        solved = [h for h in hits if h is not None]
        out[fw] = {
            "mean_steps": statistics.fmean(solved) if solved else None,
            "n_solved": len(solved),
            "n": len(rs),
        }
    return out


def run_benchmark(seeds=None, step_budget=None, results_json=None, summary_md=None,
                  plot_png=None) -> dict:
    seeds = config.SEEDS if seeds is None else seeds
    step_budget = config.STEP_BUDGET if step_budget is None else step_budget
    results_json = config.RESULTS_JSON if results_json is None else results_json
    summary_md = config.SUMMARY_MD if summary_md is None else summary_md

    if not _is_release_build_acknowledged():
        print("[benchmark] WARNING: ensure rustforge was built with "
              "`maturin develop --release` — a debug build invalidates the speed numbers.")

    results = []
    for seed in seeds:
        print(f"[benchmark] rustforge run {seed} ...")
        results.append(runners.run_rustforge(seed))
        print(f"[benchmark] sb3 run {seed} ...")
        results.append(runners.run_sb3(seed))

    grid = analysis.make_step_grid(step_budget)
    aggregate = {}
    for fw in ("rustforge", "sb3"):
        curves = [analysis.truncate_curve(r.curve, step_budget)
                  for r in results if r.framework == fw]
        mean, std = analysis.aggregate(curves, grid)
        aggregate[fw] = {"mean": mean, "std": std}

    speed = _speed_stats(results)
    solved = _solved_stats(results, step_budget)

    payload = {
        "step_budget": step_budget,
        "seeds": list(seeds),
        "step_grid": grid,
        "runs": [
            {"framework": r.framework, "seed": r.seed,
             "train_seconds": r.train_seconds, "total_steps": r.total_steps,
             "curve": [[s, rw] for s, rw in r.curve]}
            for r in results
        ],
        "aggregate": aggregate,
        "speed": speed,
        "solved": solved,
    }

    for _path in (results_json, summary_md):
        _parent = os.path.dirname(_path)
        if _parent:
            os.makedirs(_parent, exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(analysis.format_summary(speed, solved, step_budget, len(seeds)))

    # Plot is best-effort here; Task 6 owns plot.py. Import lazily so this module
    # works before plot.py exists.
    try:
        from benchmarks.sb3_comparison import plot
        plot.plot_from_results(results_json, plot_png or config.PLOT_PNG)
    except Exception as exc:  # pragma: no cover
        print(f"[benchmark] plot skipped: {exc}")

    return payload


def _is_release_build_acknowledged() -> bool:
    # Release/debug is not introspectable from Python; treat env opt-in as ack.
    return os.environ.get("RUSTFORGE_RELEASE_ACK") == "1"


def main() -> None:
    ap = argparse.ArgumentParser(description="RustForge vs SB3 DQN benchmark")
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, tiny budget — smoke run, not for reporting")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds/runs")
    ap.add_argument("--budget", type=int, default=None, help="env-step budget")
    args = ap.parse_args()

    if args.quick:
        run_benchmark(seeds=[0], step_budget=2_000)
        return
    seeds = list(range(args.seeds)) if args.seeds else None
    run_benchmark(seeds=seeds, step_budget=args.budget)


if __name__ == "__main__":
    main()
