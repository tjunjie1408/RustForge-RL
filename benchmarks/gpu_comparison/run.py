"""Collect release matmul and CLI CPU/GPU measurements without plotting dependencies."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILES = (
    ("dqn", "cartpole", 30),
    ("ppo", "cartpole", 10),
    ("ppo", "pendulum", 10),
    ("a2c", "cartpole", 10),
    ("reinforce", "cartpole", 10),
    ("td3", "pendulum", 10),
    ("sac", "pendulum", 10),
)


def capture(command, timeout=600):
    return subprocess.run(
        [str(arg) for arg in command],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def save_text(path, contents):
    path.write_text(contents, encoding="utf-8", newline="\n")


def save_json(path, data):
    save_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_metrics(path, algorithm, expected_episodes):
    if algorithm == "dqn":
        with path.open(encoding="utf-8") as source:
            records = list(csv.DictReader(source))
        rewards = [float(row["reward"]) for row in records]
        losses = [float(row["avg_loss"]) for row in records]
        if not any(loss > 0 for loss in losses):
            raise ValueError("DQN benchmark did not record a training loss")
        training_started = False
        for row in records:
            loss = float(row["avg_loss"])
            if math.isfinite(loss):
                training_started = True
            elif not math.isnan(loss) or training_started:
                raise ValueError("invalid DQN loss after training began")
            if not all(math.isfinite(float(value)) for key, value in row.items() if key != "avg_loss"):
                raise ValueError("non-finite CSV metric")
    else:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rewards = [row["metrics"]["reward.episode"] for row in records]
        for row in records:
            if not all(math.isfinite(value) for value in row["metrics"].values()):
                raise ValueError("non-finite JSONL metric")
    if len(records) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episodes, got {len(records)}")
    steps = [int(row["global_step"]) for row in records]
    if steps[0] <= 0 or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("global steps must increase")
    return steps[-1], sum(rewards) / len(rewards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sizes", default="32,64,128,256,512,1024")
    args = parser.parse_args()
    if args.trials < 1 or args.iterations < 1:
        parser.error("trials and iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    suffix = ".exe" if os.name == "nt" else ""
    cli = ROOT / "target" / "release" / f"rustforge{suffix}"
    matrix = ROOT / "target" / "release" / "examples" / f"gpu_benchmark{suffix}"
    metadata = {
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "source_commit": capture(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "source_status": capture(["git", "status", "--short", "--untracked-files=no"]).stdout.splitlines(),
        "platform": platform.platform(),
        "cpu": capture(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).stdout.strip()
        if os.name == "nt" else platform.processor(),
        "rustc": capture(["rustc", "-Vv"]).stdout.strip(),
        "gpu_inventory": capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]).stdout.strip(),
        "build": "release, locked dependencies, gpu feature",
        "binary_sha256": {binary.name: hashlib.sha256(binary.read_bytes()).hexdigest() for binary in (cli, matrix)},
        "trials": args.trials,
        "matrix_iterations": args.iterations,
        "matrix_sizes": args.sizes,
        "profiles": PROFILES,
        "timing": "CLI process wall time including initialization, training and metrics persistence; compilation excluded",
    }
    save_json(args.output / "metadata.json", metadata)
    save_text(args.output / "source-diff.patch", capture(["git", "diff", "--", "crates/rustforge-tensor/examples/gpu_benchmark.rs"]).stdout)
    print("Matrix benchmark", flush=True)
    measured = capture([matrix, args.iterations, args.sizes])
    save_text(args.output / "matmul.csv", measured.stdout)
    save_text(args.output / "matmul.log", measured.stderr)
    if measured.returncode:
        raise RuntimeError(f"matmul failed: {measured.stderr}")
    samples = []
    for algorithm, environment, episodes in PROFILES:
        for trial in range(args.trials):
            devices = ("cpu", "gpu") if trial % 2 == 0 else ("gpu", "cpu")
            for device in devices:
                run_id = f"{algorithm}-{environment}-{device}-{trial}"
                extension = "csv" if algorithm == "dqn" else "jsonl"
                metrics = args.output / f"{run_id}.{extension}"
                command = [cli, "train", algorithm, "--env", environment, "--device", device,
                           "--episodes", episodes, "--output", metrics.resolve()]
                print(f"Running {run_id}: {episodes} episodes", flush=True)
                started = time.perf_counter()
                measured = capture(command)
                elapsed = time.perf_counter() - started
                save_text(args.output / f"{run_id}.log", measured.stdout + measured.stderr)
                if measured.returncode:
                    raise RuntimeError(f"{run_id} failed: {measured.stderr}")
                steps, reward = read_metrics(metrics, algorithm, episodes)
                samples.append({"algorithm": algorithm, "environment": environment, "device": device,
                                "trial": trial, "episodes": episodes, "steps": steps,
                                "wall_seconds": elapsed, "steps_per_second": steps / elapsed,
                                "mean_reward": reward, "metrics_file": metrics.name})
                save_json(args.output / "training.json", samples)
                print(f"  {steps} steps, {elapsed:.3f}s, {steps / elapsed:.1f} steps/s", flush=True)
    metadata["completed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    save_json(args.output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
