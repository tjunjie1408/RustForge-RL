# CPU/GPU benchmark suite

This suite measures resident square FP32 matrix products and fresh-process CLI
training on CPU and GPU. The collector saves raw metrics, process logs, binary
hashes, source commit, measured source patch, adapter information and timing
boundaries. It never overwrites an existing result directory.

Build from the workspace root:

```bash
cargo build --release --locked -p rustforge-cli --features gpu --bin rustforge
cargo build --release --locked -p rustforge-tensor --features gpu --example gpu_benchmark
```

Collection uses Python's standard library. Create a standalone plotting
environment (activate it using your platform's usual command):

```bash
python -m venv benchmarks/gpu_comparison/.venv
python -m pip install -r benchmarks/gpu_comparison/requirements.txt
python benchmarks/gpu_comparison/run.py --output benchmarks/gpu_comparison/results/NEW-RUN --trials 3 --iterations 100
python benchmarks/gpu_comparison/plot.py benchmarks/gpu_comparison/results/NEW-RUN
```

The collector uses `nvidia-smi` for device inventory, so this run setup expects
an NVIDIA driver installation. The matrix benchmark records the actual selected
wgpu adapter separately. Run serially without other compute-intensive work.
CLI subprocesses have a 600-second timeout; any failed process, invalid metrics
or missing sample prevents completion. DQN permits its documented NaN loss
sentinel before the first update; later losses and other metrics must be finite.

Matmul always has three trials; `--trials` controls CLI repetitions. CLI defaults
are used, with 30 DQN episodes and 10 episodes for each other profile. CPU/GPU
order alternates. These short runs measure performance rather than learning
quality. See [the full report](../../docs/gpu-performance.md) for methodology,
limits and [the recorded run](results/2026-10-05/metadata.json).

Plotting validates sample counts and timing consistency, produces PNG/SVG files
beside the raw results, and updates `docs/gpu-performance.md`. It deliberately
refuses incomplete runs. Error bars represent observed ranges, not confidence
intervals.

The dated run also includes `validation.json`, recording additional checks and
the original GPU numerical failure. A separate `softmax-fix-validation.json`
records the subsequent correction. Plotting preserves both records in the report;
benchmark completion alone does not imply that the correctness suite passed.
