import json

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("rustforge")

from benchmarks.sb3_comparison import runners


def test_run_rustforge_tiny(monkeypatch):
    monkeypatch.setattr(runners.config, "RUSTFORGE_EPISODES", 3)
    monkeypatch.setattr(runners.config, "MAX_STEPS", 50)
    r = runners.run_rustforge(run_idx=0)
    assert r.framework == "rustforge"
    assert r.train_seconds > 0.0
    assert r.total_steps > 0
    assert len(r.curve) >= 1
    assert all(isinstance(s, int) and isinstance(rw, float) for s, rw in r.curve)


def test_run_sb3_tiny(monkeypatch):
    monkeypatch.setattr(runners.config, "STEP_BUDGET", 400)
    r = runners.run_sb3(seed=0)
    assert r.framework == "sb3"
    assert r.train_seconds > 0.0
    assert r.total_steps >= 400
    assert len(r.curve) >= 1
    assert all(isinstance(s, int) and isinstance(rw, float) for s, rw in r.curve)


def test_run_benchmark_quick_writes_artifacts(tmp_path, monkeypatch):
    from benchmarks.sb3_comparison import benchmark

    monkeypatch.setattr(benchmark.config, "RUSTFORGE_EPISODES", 3)
    monkeypatch.setattr(benchmark.config, "MAX_STEPS", 50)
    monkeypatch.setattr(benchmark.runners.config, "RUSTFORGE_EPISODES", 3)
    monkeypatch.setattr(benchmark.runners.config, "MAX_STEPS", 50)

    results_json = tmp_path / "results.json"
    summary_md = tmp_path / "summary.md"
    data = benchmark.run_benchmark(
        seeds=[0], step_budget=400,
        results_json=str(results_json), summary_md=str(summary_md),
    )

    assert results_json.exists() and summary_md.exists()
    saved = json.loads(results_json.read_text())
    frameworks = {run["framework"] for run in saved["runs"]}
    assert frameworks == {"rustforge", "sb3"}
    assert saved["aggregate"]["rustforge"]["mean"]
    assert saved["aggregate"]["sb3"]["mean"]
    assert data["speed"]["rustforge"]["thru_mean"] > 0
