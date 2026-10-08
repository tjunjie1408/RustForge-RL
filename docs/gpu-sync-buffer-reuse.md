# GPU synchronization and scratch buffer reuse

TD3/SAC critic, actor and temperature objective checks now download their
aggregate nonfinite count, mask/temperature validation and scalar metrics in one
batch. Each generated objective performs one readback and one host wait. The
original reductions, loss arithmetic, validation order and autograd graphs are
preserved. Network, action and gradient checks still guard their existing update
boundaries.

GPU contexts also cache dispatch-uniform and readback staging buffers. These
caches work with profiling enabled or disabled and apply to every GPU algorithm.
Four additive profiling counters expose explicit buffer allocations and reuse:
`parameter_allocations`, `parameter_reuses`, `readback_allocations` and
`readback_reuses`. See the [profiling contract](gpu-profiling.md).

## Architecture Decisions

- Context clones share caches; independent tensor ownership scopes have separate
  caches. Pools retain buffers without retaining their owning `DeviceState`,
  avoiding an ownership cycle.
- Uniform leases remain exclusive until the consuming encoder is submitted.
  Later `queue.write_buffer` calls follow previously submitted consumers on the
  shared queue. Reuse adds no host wait and remains safe across context clones.
- Readback leases remain exclusive through map, copy and unmap. Only the requested
  byte range is mapped, preserving lengths and scalar bits when reusing larger
  buffers. Failed mapping or an unwinding copy discards the lease.
- Best-fit caches retain at most 32 uniform buffers / 4 KiB and four readback
  buffers / 8 MiB per ownership scope. Readback capacities grow geometrically
  from 64 bytes. Buffers above the idle-cache budget are discarded after use.
  These limits bound retained idle memory, not simultaneous active operations.
  No pool mutex is held across submission, mapping or a device wait.
- Public loss fields replaced with non-scalar or foreign-context variables use
  the previous validation path. This preserves legacy ownership/error behavior.
- Tensor outputs, autograd snapshots and optimizer state keep their current
  storage and lifetimes. Bind groups and encoders are still created per dispatch;
  wgpu's internal upload staging is outside the explicit allocation counters.

## Paired profiling evidence

The before binary is commit `fdfb225`. Three before/after pairs per algorithm use
release builds of the same runner, 70 environment steps, batch size 64 and seven
learning updates on Mesa llvmpipe (LLVM 19.1.7). TD3 performs three actor updates;
SAC performs seven actor and temperature updates.

| Counter | TD3 before | TD3 after | SAC before | SAC after |
| --- | ---: | ---: | ---: | ---: |
| Readback batches | 331 | 300 | 429 | 380 |
| Blocking waits, including final completion | 332 | 301 | 430 | 381 |
| Queue submissions | 6,666 | 6,635 | 14,257 | 14,208 |
| Compute dispatches | 6,335 | 6,335 | 13,828 | 13,828 |
| Readback bytes | 1,324 | 1,324 | 5,076 | 5,076 |
| Explicit uniform allocations | 6,335 | 2 | 13,828 | 1 |
| Explicit readback allocations | 331 | 1 | 429 | 1 |

Before allocation counts are inferred from the old implementation's one uniform
per compute dispatch and one staging buffer per readback; the baseline profiler
had no scratch counters. After counts are measured by the new counters. Every
pair has identical seeded training diagnostics, compute dispatch counts, tensor
allocations/bytes, upload counts/bytes and logical readback bytes.

Objective readbacks fall from 41 to 10 for TD3 and from 70 to 21 for SAC. Overall
waits fall 9.3% and 11.4%, respectively. DQN retains 77 readbacks / 78 waits in the
same runner and benefits from scratch reuse. Its replay sampler is unseeded, so
no cross-binary exact-loss comparison is claimed.

[Raw counters and host timings](gpu-sync-buffer-reuse.csv) include three runs per
version/algorithm. Measurements overlapped software-adapter regression tests;
these host timings are noisy and do not establish a latency improvement. Physical
GPU measurements remain a separate local validation step.

To reproduce the current profile on a physical GPU:

```bash
cargo build --release -p rustforge-rl --example gpu_training_profile --features gpu
python3 benchmarks/gpu_profiling/verify.py
cargo run --release -p rustforge-rl --example gpu_training_profile --features gpu -- td3 70
cargo run --release -p rustforge-rl --example gpu_training_profile --features gpu -- sac 70
```

To run adapter-required unit/integration checks (excluding the six unrelated CPU
convergence gates):

```bash
cargo test --workspace --exclude rustforge-python --all-features --lib --tests -- \
  --ignored --skip a2c_cartpole_convergence \
  --skip a2c_converges_on_cartpole_seed_2026 \
  --skip ppo_converges_on_cartpole_seed_2026 \
  --skip reinforce_cartpole_seed_2026_meets_learning_gate \
  --skip dqn_converges_on_cartpole --skip dqn_per_converges_on_cartpole
```

Plain workspace testing leaves adapter-required cases ignored.

For a paired comparison, retain a release runner built at `fdfb225` before
building the current runner. Run each binary with the same algorithm/step count,
compare the JSON `training` values and the counters above, and collect timings
without concurrent compilation or testing. Repeat with your research workload;
these fixed small-network profiles are diagnostic cases.

## Validation and test delta

- Normal GPU-enabled workspace: **1,118 passed, 0 failed, 205 ignored**. The seven
  new adapter tests increase the baseline ignored count from 198 to 205; no
  adapter-independent tests were added or removed.
- Explicit software-adapter unit/integration run: **178 passed, 0 failed**,
  including **171 existing + seven new** tests across the GPU crates and CLI.
  This includes learning acceptance, checkpoint continuation, rollback, CPU/f64
  numerical parity and concurrent scratch-buffer reuse.
- The broad `--ignored` invocation also forced ignored illustrative NN doctests
  to compile: one passed and eight failed on missing example variables/imports
  or obsolete example signatures. These unchanged snippets remain ignored in
  the normal passing suite. Run adapter checks with `--lib --tests` to exclude
  ignored documentation examples; they are outside this implementation.
- New coverage: four scratch-reuse tests cover queued uniform rewrites and saved
  tensors, variable-size scalar/f32/u32 staging, concurrent context clones and
  oversized-buffer discard. Three objective tests cover exact scalar bits and
  one-wait behavior for all five objectives, error precedence and the legacy
  foreign-scalar fallback.
- The existing profiling test now verifies allocation/reuse counters and their
  scope propagation. The runner verifier checks both cache accounting identities
  and one objective readback/wait per backward pass.
- Three paired TD3/SAC comparisons preserve exact seeded diagnostics. The runner
  verifier passes three algorithm workflows and four invalid-argument cases.
- Rustfmt, warning-denied Clippy and Rust 1.75 GPU all-target checks pass.

## Deviations from Plan

No requested task was dropped. Buffer reuse targets dispatch uniforms and staging
storage first; tensor/optimizer arenas and bind-group caching remain future
optimizations. Synchronization is reduced at objective diagnostic boundaries;
existing earlier safety checks are retained. Physical GPU timing is deferred to
local hardware, consistent with the prior implementation phases.

## Issue Resolution Progress

| Feature | Issue | Status |
| --- | --- | --- |
| Pack objective checks and metrics into one wait | Unassigned | Implemented and profiled |
| Bounded dispatch-uniform and staging-buffer reuse | Unassigned | Implemented and tested |
| Allocation/reuse profiling and regression coverage | Unassigned | Implemented |
| Physical GPU performance measurements | Unassigned | Local hardware validation pending |
