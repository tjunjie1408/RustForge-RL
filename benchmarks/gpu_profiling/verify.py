"""Validate the profiling runner on a hardware or software adapter (Python 3.9+)."""
import argparse
import json
import math
from pathlib import Path
import subprocess


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=root / "target/release/examples/gpu_training_profile")
    parser.add_argument("--steps", type=int, default=70)
    args = parser.parse_args()
    if args.steps < 64:
        parser.error("--steps must be at least 64 to exercise learning")
    binary = args.binary.resolve()
    for argv in [["unknown"], ["td3", "0"], ["sac", "bad"], ["dqn", "1", "extra"]]:
        result = subprocess.run([str(binary), *argv], capture_output=True, text=True)
        assert result.returncode != 0 and not result.stdout
    for algorithm in ("dqn", "td3", "sac"):
        result = subprocess.run([str(binary), algorithm, str(args.steps)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        data = json.loads(result.stdout)
        assert data["schema"] == "rustforge-gpu-training-profile-v1"
        assert data["algorithm"] == algorithm and data["steps"] == args.steps
        assert data["timing_kind"] == "host_wall_inclusive" and data["phase_times_overlap"]
        updates = args.steps - 63
        assert data["training"]["updates"] == updates
        phases = data["profile"]["phases"]
        counters = data["profile"]["counters"]
        assert phases["inference"]["calls"] == args.steps
        assert phases["training"]["calls"] == updates
        backwards = updates
        if algorithm == "td3":
            backwards += updates // data["training"]["config"]["policy_delay"]
        elif algorithm == "sac":
            backwards *= 3
        assert phases["backward"]["calls"] == backwards
        assert phases["optimizer"]["calls"] == backwards
        assert counters["submissions"] == counters["compute_dispatches"] + counters["readback_submissions"]
        assert counters["readbacks"] == counters["readback_submissions"]
        assert counters["host_waits"] == counters["readbacks"] + 1
        assert counters["uploads"] > 0 and counters["upload_bytes"] > 0
        for phase in phases.values():
            assert phase["calls"] > 0 and phase["host_elapsed_ns"] > 0
            assert phase["counters"]["host_waits"] <= counters["host_waits"]
        if algorithm != "dqn":
            assert phases["finite_validation"]["counters"]["readbacks"] > 0
        if algorithm == "sac":
            gaussian = phases["gaussian_validation_metrics"]
            assert gaussian["counters"]["readbacks"] == gaussian["calls"]
        finite(data)
        print(f"{algorithm}: {updates} updates, {counters['submissions']} submissions, "
              f"{counters['readbacks']} readbacks; adapter {data['adapter']['name']}")
    print("Three profiling workflows and four invalid-argument cases passed.")


if __name__ == "__main__":
    main()
