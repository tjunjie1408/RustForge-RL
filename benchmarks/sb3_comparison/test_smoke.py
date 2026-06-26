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
