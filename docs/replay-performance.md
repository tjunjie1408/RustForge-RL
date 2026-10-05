# Replay sampling performance

Measured on 2026-10-05 after the v0.2.0 release. Replay sampling now extracts
contiguous destination slices once per batch, instead of checking ndarray layout
for every sampled row. Prioritized replay also removes its temporary vector:
it writes transitions and unnormalized weights immediately, then normalizes only
the active weight prefix. No new buffer storage or dependencies are needed.

## Measurements

Baseline: release commit `a1e9b19`. Both binaries used the same benchmark source,
Rust 1.99.0, the default Cargo release profile, and this managed Linux environment
(AMD EPYC 9V74, five exposed logical CPUs). No build or test ran during the paired
measurements. The virtualized machine was not CPU-pinned; these are local
microbenchmark results, not a hardware-independent throughput promise.

| Sampler | Observation / batch dimensions | Before (µs/batch) | After (µs/batch) | Time reduction |
| --- | --- | ---: | ---: | ---: |
| Uniform | 4 / 64 | 2.700 | 1.884 | 30.2% |
| Uniform | 32 / 256 | 12.751 | 9.072 | 28.9% |
| Prioritized | 4 / 64 | 8.783 | 7.390 | 15.9% |
| Prioritized | 32 / 256 | 37.546 | 28.850 | 23.2% |
| Continuous | 4 / 64 | 2.818 | 1.710 | 39.3% |
| Continuous | 32 / 256 | 13.545 | 7.540 | 44.3% |

Each process warms up for 1,000 samples and measures seven trials of 20,000
samples. Each cell above is the median of three process medians. Process order
alternates baseline/optimized, optimized/baseline, baseline/optimized. The replay
capacity is 16,384, continuous actions have two components, and PER uses alpha
0.6, beta 0.4 and unequal priorities. PER and continuous RNGs use seed 42;
uniform replay retains its existing thread RNG. All storage is allocated before
sampling. `black_box` retains sampled outputs.

Allocation counting runs separately from timing, so allocation-counter increments
do not bias the timed baseline. The counted 20,000 samples allocate **20,000 times
before and zero times after** for PER. Uniform and continuous replay allocate zero
in both versions. Setup, insertion, parameter updates and destruction are excluded.
See the [raw measurements](replay-sampling-benchmark.csv).

These changes affect CPU replay feeding DQN, TD3 and SAC, including their GPU
agents. They do not measure whole-training speedup, GPU dispatch, transfers or
synchronization. The existing [GPU CLI comparison](gpu-cli-performance.md)
identifies separate overhead that remains to be profiled and optimized.

## Reproduce

```bash
cargo run --locked --release -p rustforge-rl --example replay_benchmark -- 20000
```

Output is CSV with nanoseconds per batch and total allocation count for the
separate allocation trial. The iteration argument must be positive. To compare
with v0.2.0, use the same example source in a checkout at `a1e9b19`, compile with
the same toolchain/profile, and alternate the two saved binaries on an otherwise
idle machine. The baseline sampler implementations are unchanged in that build;
the benchmark itself is new in this change.

## Architecture decisions

- Keep the existing APIs, stratified PER distribution and RNG draw order.
- Reuse the supplied weight tensor for temporary values. A second pass divides
  by the same maximum-weight expression as before; do not replace it with an
  algebraically equivalent formula that could change rounding.
- Keep slice bounds checks and Rust memory safety in production sampling.
  Allocation instrumentation uses a system-delegating allocator only in the
  example and a dedicated integration-test executable.
- Preserve inactive rows, tree indices and actions when a batch is partial.

## Validation

- Default native workspace: **1,074 passed, 0 failed, 27 ignored**.
- GPU-enabled native workspace: **1,116 passed, 0 failed, 191 ignored**.
  Adapter-required tests remain ignored; no physical GPU checks were run.
- Workspace all-target/all-feature Clippy (excluding the Python extension):
  passed with warnings denied.
- Rust 1.75 `cargo check -p rustforge-rl --all-targets`: passed, including the
  new benchmark and allocation test.
- Rebuilt v0.2.0 Python extension with `maturin develop --locked --offline`:
  **50 passed, 0 failed** in pytest.
- Workspace formatting and Git whitespace checks: passed.
- The two new tests were rerun successfully after the Clippy-only test-loop edit.

Two new tests provide the delta: a reference-equivalence unit test and an
allocation-regression integration test. The reference test covers two capacities
(including a non-power-of-two tree), partial/full wrapped buffers, three seeds,
four requested sizes including zero, and three beta values: 144 combinations.
It compares every active transition field, exact floating-point weight bits,
indices, the continuing RNG stream and unchanged inactive rows. Existing seeded
continuous replay, PER priority-churn and agent-training tests remain in place.

## Deviations from Plan

Physical GPU profiling and end-to-end speedup claims are deferred. This iteration
isolates shared replay overhead and verifies numerical equivalence before moving
to GPU execution changes.

## Issue Resolution Progress

| Change | Issue | Status |
| --- | --- | --- |
| Hoist contiguous slice extraction for all replay samplers | Unassigned | Implemented |
| Remove per-batch PER allocation | Unassigned | Implemented |
| Seed/weight equivalence and allocation regression coverage | Unassigned | Implemented |
| Reproducible paired CPU microbenchmark | Unassigned | Measured |
