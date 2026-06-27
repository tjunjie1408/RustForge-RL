import csv
import json as _json

from benchmarks.sb3_comparison import config, analysis


def test_matched_config_values():
    assert config.STEP_BUDGET == 50_000
    assert config.SEEDS == list(range(10))
    assert config.BATCH_SIZE == 32
    assert config.BUFFER_SIZE == 10_000
    assert config.LEARNING_STARTS == 128
    assert config.GAMMA == 0.99
    assert config.LR == 1e-3
    assert config.HIDDEN_DIM == 64
    assert config.TARGET_UPDATE == 100
    assert config.EPS_INITIAL == 1.0 and config.EPS_FINAL == 0.05
    assert config.EPS_DECAY_STEPS == 2_000
    assert config.SOLVED_THRESHOLD == 475.0
    assert config.SOLVED_WINDOW == 100


def test_sb3_kwargs_match_rustforge():
    kw = config.sb3_kwargs(seed=7)
    assert kw["batch_size"] == 32
    assert kw["buffer_size"] == 10_000
    assert kw["learning_starts"] == 128
    assert kw["train_freq"] == 1
    assert kw["gradient_steps"] == 1
    assert kw["target_update_interval"] == 100
    assert kw["policy_kwargs"] == {"net_arch": [64]}
    assert kw["device"] == "cpu"
    assert kw["seed"] == 7
    # ε linear schedule maps exactly onto SB3's exploration_fraction
    assert kw["exploration_initial_eps"] == 1.0
    assert kw["exploration_final_eps"] == 0.05
    assert abs(kw["exploration_fraction"] - 2_000 / 50_000) < 1e-12


def test_parse_rustforge_csv(tmp_path):
    p = tmp_path / "log.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "avg_loss", "epsilon", "global_step"])
        w.writerow([0, 12.0, 0.5, 0.9, 12])
        w.writerow([1, 30.0, 0.4, 0.8, 42])
    assert analysis.parse_rustforge_csv(str(p)) == [(12, 12.0), (42, 30.0)]


def test_truncate_curve_keeps_points_within_budget():
    curve = [(10, 1.0), (50, 2.0), (90, 3.0)]
    assert analysis.truncate_curve(curve, 60) == [(10, 1.0), (50, 2.0)]


def test_truncate_curve_empty_when_all_beyond():
    assert analysis.truncate_curve([(100, 1.0)], 50) == []


def test_steps_to_solved_returns_step_at_first_window_crossing():
    # window=3, threshold=2.0. First full window (idx2: 1,1,3 -> avg 1.67) fails;
    # the window ending at idx3 (1,3,3 -> avg 2.33) is the first to reach 2.0.
    curve = [(10, 1.0), (20, 1.0), (30, 3.0), (40, 3.0)]
    assert analysis.steps_to_solved(curve, threshold=2.0, window=3) == 40


def test_steps_to_solved_solves_at_first_full_window():
    # A run already at threshold across its first `window` episodes is solved at
    # that window's last step (index window-1); it must NOT be skipped.
    curve = [(10, 3.0), (20, 3.0), (30, 3.0)]
    assert analysis.steps_to_solved(curve, threshold=2.0, window=3) == 30


def test_steps_to_solved_none_when_never_reached():
    curve = [(10, 1.0), (20, 1.0), (30, 1.0)]
    assert analysis.steps_to_solved(curve, threshold=2.0, window=3) is None


def test_make_step_grid_spans_zero_to_max():
    grid = analysis.make_step_grid(100, n=5)
    assert grid[0] == 0 and grid[-1] == 100 and len(grid) == 5


def test_aggregate_means_and_stds_on_grid():
    grid = [0, 10, 20]
    c1 = [(0, 0.0), (20, 20.0)]   # interp -> [0, 10, 20]
    c2 = [(0, 0.0), (20, 40.0)]   # interp -> [0, 20, 40]
    mean, std = analysis.aggregate([c1, c2], grid)
    assert mean == [0.0, 15.0, 30.0]
    assert std[0] == 0.0 and std[2] == 10.0


def test_format_summary_is_markdown_with_both_frameworks():
    speed = {
        "rustforge": {"time_mean": 1.0, "time_std": 0.1, "thru_mean": 50000.0, "thru_std": 100.0},
        "sb3": {"time_mean": 10.0, "time_std": 0.5, "thru_mean": 5000.0, "thru_std": 50.0},
    }
    solved = {
        "rustforge": {"mean_steps": 20000.0, "n_solved": 9, "n": 10},
        "sb3": {"mean_steps": 25000.0, "n_solved": 8, "n": 10},
    }
    md = analysis.format_summary(speed, solved, step_budget=50_000, n_runs=10)
    assert "rustforge" in md and "sb3" in md
    assert "steps/sec" in md
    assert "|" in md  # a markdown table


def test_plot_from_results_writes_nonempty_png(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")

    from benchmarks.sb3_comparison import plot

    results = {
        "step_budget": 100,
        "step_grid": [0, 50, 100],
        "aggregate": {
            "rustforge": {"mean": [0.0, 10.0, 20.0], "std": [0.0, 1.0, 2.0]},
            "sb3": {"mean": [0.0, 8.0, 16.0], "std": [0.0, 1.0, 2.0]},
        },
    }
    rj = tmp_path / "results.json"
    rj.write_text(_json.dumps(results))
    out = tmp_path / "curve.png"
    plot.plot_from_results(str(rj), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_solved_stats_truncates_to_budget(monkeypatch):
    from benchmarks.sb3_comparison import benchmark, config
    from benchmarks.sb3_comparison.runners import RunResult

    monkeypatch.setattr(config, "SOLVED_THRESHOLD", 2.0)
    monkeypatch.setattr(config, "SOLVED_WINDOW", 2)
    # Crosses the (window=2) threshold only at step 60/80 — beyond a 50-step budget.
    curve = [(10, 1.0), (40, 1.0), (60, 3.0), (80, 3.0)]
    rs = [RunResult("rustforge", 0, 1.0, 80, curve),
          RunResult("sb3", 0, 1.0, 50, [(10, 1.0), (40, 1.0)])]
    out = benchmark._solved_stats(rs, step_budget=50)
    # Truncated to 50 steps, the trailing-2 mean never reaches 2.0 -> unsolved.
    assert out["rustforge"]["n_solved"] == 0
    assert out["rustforge"]["mean_steps"] is None
    assert out["sb3"]["n_solved"] == 0
