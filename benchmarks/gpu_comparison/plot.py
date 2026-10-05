"""Validate saved measurements and generate the figures and GPU performance report."""

import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"cpu": "#475569", "naive": "#0072B2", "tiled": "#D55E00", "gpu": "#0072B2"}


def distribution(values):
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("timings and throughputs must be finite and positive")
    return statistics.median(values), min(values), max(values)


def save_figure(figure, stem):
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    output = io.StringIO()
    figure.savefig(output, format="svg", bbox_inches="tight")
    svg = "\n".join(line.rstrip() for line in output.getvalue().splitlines()) + "\n"
    stem.with_suffix(".svg").write_text(svg, encoding="utf-8", newline="\n")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    if "completed_at" not in metadata:
        raise ValueError("benchmark run is incomplete")
    raw = (results / "matmul.csv").read_text(encoding="utf-8").splitlines()
    matrix = list(csv.DictReader(line for line in raw if not line.startswith("#")))
    training = json.loads((results / "training.json").read_text(encoding="utf-8"))
    if len(training) != len(metadata["profiles"]) * metadata["trials"] * 2:
        raise ValueError("missing training samples")
    sizes = [int(size) for size in metadata["matrix_sizes"].split(",")]
    if len(matrix) != len(sizes) * 3:
        raise ValueError("missing matrix trials")
    matrix_stats = {}
    for size in sizes:
        rows = [row for row in matrix if int(row["n"]) == size]
        if sorted(int(row["trial"]) for row in rows) != [0, 1, 2]:
            raise ValueError("missing or duplicate matrix trial")
        matrix_stats[size] = {device: distribution([float(row[f"{device}_ms"]) for row in rows])
                              for device in ("cpu", "naive", "tiled")}
    grouped = defaultdict(list)
    for sample in training:
        if sample["steps"] <= 0 or not math.isfinite(sample["mean_reward"]):
            raise ValueError("invalid training metrics")
        if not math.isclose(sample["steps_per_second"], sample["steps"] / sample["wall_seconds"]):
            raise ValueError("throughput does not match measured steps/time")
        grouped[(sample["algorithm"], sample["environment"], sample["device"])].append(sample)
    profiles = [(algorithm, environment, episodes) for algorithm, environment, episodes in metadata["profiles"]]
    for algorithm, environment, episodes in profiles:
        for device in ("cpu", "gpu"):
            samples = grouped[(algorithm, environment, device)]
            if sorted(row["trial"] for row in samples) != list(range(metadata["trials"])):
                raise ValueError("missing or duplicate training trial")
            if any(row["episodes"] != episodes for row in samples):
                raise ValueError("episode budget mismatch")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.1), layout="constrained")
    for device, label in (("cpu", "CPU"), ("naive", "GPU direct"), ("tiled", "GPU tiled")):
        medians, lows, highs = np.array([matrix_stats[size][device] for size in sizes]).T
        axes[0].plot(sizes, medians, "o-", color=COLORS[device], label=label)
        axes[0].fill_between(sizes, lows, highs, color=COLORS[device], alpha=0.12)
    axes[0].set(xscale="log", yscale="log", xlabel="Square matrix dimension N", ylabel="Milliseconds / product",
                title="Resident matmul latency · lower is better")
    axes[0].set_xticks(sizes, labels=[str(size) for size in sizes])
    axes[0].legend(frameon=False)
    for device, label in (("naive", "Direct"), ("tiled", "Tiled")):
        ratios = [matrix_stats[size]["cpu"][0] / matrix_stats[size][device][0] for size in sizes]
        axes[1].plot(sizes, ratios, "o-", color=COLORS[device], label=label)
    axes[1].axhline(1, color="#475569", linestyle="--", linewidth=1)
    axes[1].set(xscale="log", yscale="log", xlabel="Square matrix dimension N", ylabel="CPU latency / GPU latency",
                title="Matmul speed ratio · above 1 means GPU faster")
    axes[1].set_xticks(sizes, labels=[str(size) for size in sizes])
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    adapter = next(line for line in raw if line.startswith("# adapter:"))
    figure.suptitle("Resident CPU/GPU matrix products · release · FP32\nMedians and observed trial ranges; GPU transfers excluded", fontsize=12)
    save_figure(figure, figures / "matmul-comparison")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    labels = [f"{algorithm.upper()} / {environment.title()}" for algorithm, environment, _ in profiles]
    y = np.arange(len(labels))
    rates = {}
    for device, offset in (("cpu", -0.1), ("gpu", 0.1)):
        stats = [distribution([sample["steps_per_second"] for sample in grouped[(algorithm, environment, device)]])
                 for algorithm, environment, _ in profiles]
        medians, lows, highs = np.array(stats).T
        rates[device] = medians
        axes[0].errorbar(medians, y + offset, xerr=[medians - lows, highs - medians], fmt="o",
                         color=COLORS[device], label=device.upper(), capsize=3)
    axes[0].set(xscale="log", xlabel="Environment steps / wall-clock second",
                title="CLI throughput · higher is better")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False)
    ratios = rates["gpu"] / rates["cpu"]
    axes[1].scatter(ratios, y, color=COLORS["gpu"], s=45)
    for position, ratio in enumerate(ratios):
        axes[1].annotate(f"{ratio:.4f}×", (ratio, position), xytext=(8, -3), textcoords="offset points")
    axes[1].axvline(1, color="#475569", linestyle="--", linewidth=1)
    axes[1].set(xscale="log", xlabel="GPU throughput / CPU throughput",
                title="Training speed ratio · above 1 means GPU faster")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(ratios) / 2, max(1.6, max(ratios) * 2))
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(f"Fresh CLI processes · same default configuration · {metadata['trials']} trials\nIncludes initialization and metric persistence; episode budgets vary by profile", fontsize=12)
    save_figure(figure, figures / "training-comparison")

    asset = "../" + results.relative_to(ROOT).as_posix()
    best_size = max(sizes, key=lambda size: matrix_stats[size]["cpu"][0] / matrix_stats[size]["tiled"][0])
    best_ratio = matrix_stats[best_size]["cpu"][0] / matrix_stats[best_size]["tiled"][0]
    report = ["# GPU performance benchmark", "", f"Measured on **{metadata['started_at'][:10]}** from source commit `{metadata['source_commit'][:12]}`.", "",
              f"The tiled matmul kernel's best measured CPU/GPU latency ratio is **{best_ratio:.2f}× at {best_size}×{best_size}**. "
              f"GPU CLI throughput is lower than CPU in {sum(ratio < 1 for ratio in ratios)} of {len(ratios)} measured profiles. "
              "These are separate workloads: fast resident matrix products do not establish fast end-to-end RL training.", "",
              "## Environment and artifacts", "", f"- CPU: {metadata['cpu']}",
              "- GPU and driver:", "", "```text", metadata["gpu_inventory"], "```", "",
              "- Selected matmul adapter:", "", "```text", adapter.removeprefix("# adapter: "), "```", "",
              f"- OS: `{metadata['platform']}`; `{metadata['rustc'].splitlines()[0]}`; release build with `gpu` and locked dependencies.",
              f"- [Run metadata]({asset}/metadata.json), [matmul CSV]({asset}/matmul.csv), [training samples]({asset}/training.json).",
              f"- The matrix harness adds an optional size list; [the exact measured source patch]({asset}/source-diff.patch) is saved with binary hashes.",
              "- One laptop session; clock, power and thermal conditions were not locked. Three trials show observed spread, not confidence intervals.", "",
              "## Resident matrix multiplication", "", f"![CPU and GPU matmul comparison]({asset}/figures/matmul-comparison.png)", "",
              f"[Vector SVG]({asset}/figures/matmul-comparison.svg)", "",
              f"Each size uses deterministic inputs (seeds 123/124), three warmup dispatches per GPU kernel, three trials and {metadata['matrix_iterations']} products per trial. "
              "GPU order alternates. Timings include submission, dispatch parameter allocation and completion; upload, download and pipeline initialization are excluded. "
              "GPU output buffers are reused; CPU output allocation is included. After each trial, each GPU kernel's final output is checked against the CPU result using `1e-4 + 1e-4 * abs(expected)` tolerance, outside timing.", "",
              "Median milliseconds per product; brackets show the observed minimum–maximum:", "",
              "| N × N | CPU ms | GPU direct ms | GPU tiled ms | CPU / tiled |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for size in sizes:
        cells = [f"{median:.4f} [{low:.4f}–{high:.4f}]" for median, low, high in matrix_stats[size].values()]
        report.append(f"| {size} × {size} | " + " | ".join(cells) + f" | {matrix_stats[size]['cpu'][0] / matrix_stats[size]['tiled'][0]:.2f}× |")
    report += ["", "A ratio above 1 means GPU is faster. This measures square FP32 products, not arbitrary network shapes or transfer-inclusive latency. "
               "The previous [llvmpipe measurements](gpu-development.md#performance-evidence) remain a separate historical dataset from a different machine.", "",
               "## End-to-end CLI training", "", f"![CPU and GPU training throughput]({asset}/figures/training-comparison.png)", "",
               f"[Vector SVG]({asset}/figures/training-comparison.svg)", "",
               "All six algorithms are covered, including both PPO environment routes. Each trial launches a fresh release CLI process; CPU/GPU order alternates. "
               "Wall time includes process startup, GPU adapter/pipeline initialization, training, metrics writes and shutdown. "
               "No checkpoint is saved and no TUI is run. Actual completed steps are divided by process wall time. Medians aggregate per-trial rates, not ratios of pooled totals.", "",
               "| Algorithm / environment | Episodes | CPU steps/s | GPU steps/s | GPU / CPU |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for index, (algorithm, environment, episodes) in enumerate(profiles):
        report.append(f"| {algorithm.upper()} / {environment} | {episodes} | {rates['cpu'][index]:,.1f} | {rates['gpu'][index]:,.1f} | {ratios[index]:.4f}× |")
    report += ["", "Completed steps and total wall time, both medians:", "",
               "| Algorithm / environment | CPU steps | GPU steps | CPU seconds | GPU seconds |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for algorithm, environment, _ in profiles:
        samples = [grouped[(algorithm, environment, device)] for device in ("cpu", "gpu")]
        steps = [statistics.median(row["steps"] for row in group) for group in samples]
        seconds = [statistics.median(row["wall_seconds"] for row in group) for group in samples]
        report.append(f"| {algorithm.upper()} / {environment} | {steps[0]:,.0f} | {steps[1]:,.0f} | {seconds[0]:.3f} | {seconds[1]:.3f} |")
    report += ["", "### Interpretation and limits", "",
               "- Default networks use 64 hidden units. TD3/SAC use batch 64, learn from step 64 and use random actions for the first 1,000 steps; all their runs complete 2,000 steps. "
               "The 30-episode DQN budget is checked for a nonzero recorded training loss. Metrics must be finite, except DQN's documented no-update NaN loss before training starts; step counters must increase.",
               "- CLI training is episode-budgeted. CartPole policies can finish different numbers of steps, so raw total seconds are not equal-work comparisons. "
               "Throughput normalizes actual steps, but different trajectories still produce different update workloads.",
               "- The CLI supplies seed 2026 to the on-policy and continuous trainers; DQN has no CLI seed parameter. "
               "Three trials repeat the CLI defaults rather than sampling three independent learning seeds. Cross-backend numerical trajectories need not match.",
               "- These short runs do not establish convergence, reward superiority, checkpoint recovery, device-loss robustness, or full GPU correctness. "
               "Kernel dispatch, synchronization, readback and startup are plausible overheads; no profiler was run to apportion their cost.",
               "- The historical [RustForge vs SB3 CPU benchmark](performance.md#historical-cpu-dqn-benchmark) was not rerun. "
               "It uses a different workload and timing boundary and must not be combined with this dataset to claim a GPU-vs-SB3 speedup.", "",
               "## Reproduce", "", "From the workspace root:", "", "```bash",
               "cargo build --release --locked -p rustforge-cli --features gpu --bin rustforge",
               "cargo build --release --locked -p rustforge-tensor --features gpu --example gpu_benchmark",
               "python benchmarks/gpu_comparison/run.py --output benchmarks/gpu_comparison/results/NEW-RUN --trials 3 --iterations 100",
               "python benchmarks/gpu_comparison/plot.py benchmarks/gpu_comparison/results/NEW-RUN", "```", "",
               "Use the standalone Python environment described in [the benchmark guide](../benchmarks/gpu_comparison/README.md). "
               "The collector refuses to overwrite an existing run directory. Plots require Matplotlib; collection uses only Python's standard library.", ""]
    repair_path = results / "softmax-fix-validation.json"
    repair = json.loads(repair_path.read_text(encoding="utf-8")) if repair_path.exists() else None
    report[6:6] = ["For the implementation paths and timing differences, see "
                   "[why CLI training differs from a matrix benchmark](gpu-cli-performance.md).", ""]
    validation_path = results / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        report += ["", "## Verification history", "",
                   f"[Verification record]({asset}/validation.json). This is separate from the successful performance run.", "",
                   "| Check | Result |", "| --- | --- |"]
        for check in validation["checks"]:
            report.append(f"| `{check['command']}` | {check['result']} |")
        failure = validation.get("failure")
        if failure:
            report += ["", f"**Original pre-fix numerical failure:** `{failure['test']}` failed at "
                       f"[{failure['file']}:{failure['line']}](../{failure['file']}#L{failure['line']}). "
                       f"Actual probability `{failure['actual']}` vs reference `{failure['reference']}` differs by "
                       f"`{abs(failure['actual'] - failure['reference']):.3g}`, exceeding tolerance `{failure['absolute_tolerance']}`. "
                       "The isolated pre-fix rerun failed identically. These records describe the original benchmark snapshot, "
                       "and are retained without overwriting the measured binary hashes or timing samples.", ""]
            if not repair:
                report[6:6] = ["**Validation caveat:** the GPU tensor suite has one reproducible numerical failure "
                               "(22/23 tests pass); see [the verification section](#verification-history).", ""]
    if repair:
        report[6:6] = ["**Follow-up correctness fix:** the softmax precision regression is fixed in the current working tree. "
                       "The GPU tensor suite passes 24/24 tests and related policy loss suites pass 11/11. "
                       "The performance tables above remain pre-fix measurements; the 42-process timing suite was not rerun.", ""]
        report += ["", "### Softmax fix validation", "", f"[Post-fix record]({asset}/softmax-fix-validation.json). "
                   "See [the diagnosis and implementation explanation](gpu-cli-performance.md#softmax-precision-fix).", "",
                   "| Check | Result |", "| --- | --- |"]
        for check in repair["checks"]:
            report.append(f"| `{check['command']}` | {check['result']} |")
    (ROOT / "docs" / "gpu-performance.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    print(f"Validated {len(matrix)} matrix samples and {len(training)} training samples; wrote figures and docs/gpu-performance.md")


if __name__ == "__main__":
    main()
