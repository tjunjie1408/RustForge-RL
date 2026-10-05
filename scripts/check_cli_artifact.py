"""Check the shipped CLI without requiring a GPU adapter or a terminal."""

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


def main():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    tag = os.environ.get("GITHUB_REF", "")
    if tag.startswith("refs/tags/v") and tag != f"refs/tags/v{version}":
        raise RuntimeError(f"Tag {tag} does not match package version {version}")
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="rustforge-artifact-") as directory:
        def run(*args):
            return subprocess.run([binary, *args], cwd=directory, check=True,
                                  capture_output=True, text=True).stdout

        if run("--version").strip() != f"rustforge {version}":
            raise RuntimeError("Unexpected CLI version")
        plan = json.loads(run("plan", "dqn", "--device", "gpu"))
        if plan["device"] != "gpu" or plan["schema"] != "rustforge-training-plan-v1":
            raise RuntimeError("GPU plan discovery failed")
        output = Path(directory) / "metrics.csv"
        run("train", "dqn", "--device", "cpu", "--episodes", "2", "--output", str(output))
        with output.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["episode", "reward", "avg_loss", "epsilon", "global_step"]:
                raise RuntimeError("Unexpected DQN CSV schema")
            rows = list(reader)
        if len(rows) != 2 or not all(
            math.isfinite(float(row[key]))
            for row in rows for key in ("episode", "reward", "epsilon", "global_step")
        ):
            raise RuntimeError("CPU training did not produce two valid episodes")
        # CSV v1 uses NaN for absent loss during replay warm-up.
        if any(math.isinf(float(row["avg_loss"])) for row in rows):
            raise RuntimeError("CPU training produced infinite loss")
    print(f"CLI {version}: version, GPU discovery and CPU training passed")


if __name__ == "__main__":
    main()
