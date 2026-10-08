# GPU training instrumentation

GPU contexts offer opt-in host accounting. Normal contexts remain unprofiled.
Enable profiling **before constructing agents or GPU variables** so their cloned
contexts share the collector:

```rust,ignore
let context = GpuContext::new()?.with_profiling();
let agent = GpuDqn::new_seeded(&context, config, 42)?;
{
    let _phase = context.profile_scope("inference");
    let action = agent.select_greedy_action(&state)?;
}
let profile = context.profile_snapshot().unwrap();
```

A fresh `with_profiling()` handle shares tensor ownership/device resources but
starts a new collector. Its clones and `with_matmul_kernel()` handles share the
collector. Previously created handles/agents remain unchanged. No global setting,
metric-schema change or automatic output file is introduced.

## Run on your GPU

```bash
cargo build --locked --release -p rustforge-rl --features gpu --example gpu_training_profile
cargo run --locked --release -p rustforge-rl --features gpu --example gpu_training_profile -- td3 128 > td3-profile.json
cargo run --locked --release -p rustforge-rl --features gpu --example gpu_training_profile -- sac 128 > sac-profile.json
cargo run --locked --release -p rustforge-rl --features gpu --example gpu_training_profile -- dqn 128 > dqn-profile.json
python benchmarks/gpu_profiling/verify.py --steps 70
```

On Windows, the verifier accepts
`--binary target/release/examples/gpu_training_profile.exe`. It requires only
Python 3.9+ and checks all three algorithms plus four invalid-argument cases.
Driver stdout contains one JSON report with schema
`rustforge-gpu-training-profile-v1`; failures return a nonzero status and no report.
Redirecting stdout does not capture adapter diagnostics emitted on stderr.

The driver uses CartPole for DQN and Pendulum for TD3/SAC. It collects exactly the
requested environment-step budget, starts learning at 64 stored transitions and
then updates once per step. It uses the default DQN config and the existing
Pendulum runtime TD3/SAC configs. The report includes the actual config, adapter,
batch size, update count, completed episodes and last training diagnostics.
A budget below 64 intentionally has no backward/optimizer/training phase.

This is a profiling workload, **not the exact CLI learning protocol**: inference
runs on every collection step rather than using a random-action warmup; DQN uses
greedy collection. Network/environment seeds are 42; continuous collection,
replay, target and actor noise have separate seeded streams. Uniform DQN replay
uses its existing unseeded thread RNG, indicated by a null `replay_seed`.

## Counter and timing contract

| Field / phase | Meaning |
| --- | --- |
| `submissions` | Explicit queue submissions, including compute, readback copies and zero-inner-dimension matmul clears |
| `command_buffers` | Encoded command buffers; scoped batches submit several together, so this can exceed `submissions`. Zero-inner-dimension clears count as buffers without compute dispatches |
| `compute_dispatches` | Compute dispatch calls, excluding empty operations and clears |
| `readback_submissions` | Explicit device-to-readback-buffer copy submissions |
| `uploads`, `upload_bytes` | Nonempty tensor/index uploads and logical bytes; includes initialized parameter tensors, excludes dispatch uniforms |
| `readbacks`, `readback_bytes` | Nonempty readback batches requested and logical bytes, even if subsequent mapping fails; packed scalars share one batch |
| `host_waits`, `host_wait_ns` | Blocking device poll calls and wall time inside those calls |
| `tensor_allocations`, `tensor_bytes` | Newly allocated tensor storage and physical bytes; empty tensors reserve four bytes |
| `tensor_reuses` | Recycled device storage for fully overwritten kernel outputs; live graph snapshots and optimizer state remain immutable |
| `parameter_allocations`, `parameter_reuses` | Explicit dispatch-uniform buffers created or leased from the shared context cache; excludes wgpu internal upload staging |
| `readback_allocations`, `readback_reuses` | Explicit readback staging buffers created or leased; buffer capacity may exceed the logical download length |
| `initialization` | Agent/network initialization in the driver; process device initialization is reported separately |
| `inference`, `environment`, `replay`, `training` | Driver host phases, including CPU work and any waits inside them |
| `batch_upload` | DQN's separate batch upload; TD3/SAC uploads remain inside their training methods |
| `linear_forward` | Shared GPU affine-layer forward calls; excludes subsequent activations and network-level checks |
| `backward`, `optimizer` | Shared GPU autograd and SGD/Adam calls, including failed/no-op calls |
| `finite_validation` | TD3/SAC finite-check helpers, including their reductions and scalar readbacks |
| `objective_diagnostics` | TD3/SAC objective finite-count reductions and packed scalar diagnostics, with one readback/wait per call |
| `gaussian_validation_metrics` | Gaussian density validation and its diagnostic readback; sampled actions now share that batch |
| `inference_readback` | TD3 action validation flags packed with physical actions |
| `temperature_readback` | SAC log-temperature validation flag packed with its exponentiated value |
| `final_completion` | One explicit completion wait at the end of the profiling driver |

Profiling itself adds **no GPU commands, transfers, timestamp queries or waits**.
The driver's final completion wait is deliberate and appears in its own phase.
Disabled contexts do not allocate a collector, read timing clocks or lock a
profiler. Enabled accounting uses a mutex and scope snapshots, so it adds host
overhead and is intended for diagnosis rather than benchmarking peak throughput.

Phase timings are **inclusive host wall time**, not GPU kernel time. Asynchronous
work can execute after its submitting phase ends and contribute to a later wait.
Nested phase totals overlap: do not add `training`, `backward` and `optimizer`
times or counters. Snapshots include completed scopes; live scopes appear on drop.
Calls include failed/unwound scopes rather than only successful updates.

Serialize operations on clones of a profiled context for phase attribution.
Counters are thread-safe, but concurrent work in the same collector is included
in a scope's delta. The process-wide device/queue is also shared with other tensor
ownership scopes, so a blocking poll can wait on unrelated work. Use an otherwise
idle process for interpretable results. No GPU timestamp-query feature is needed.

## Verified software-adapter observation

On Mesa llvmpipe (LLVM 19.1.7), the 70-step release-mode workflows each completed
seven training updates. These counts are diagnostic smoke evidence, not physical
GPU latency measurements or learning-quality results:

| Algorithm | Submissions | Readbacks | Backward / optimizer calls |
| --- | ---: | ---: | ---: |
| DQN | 1,090 | 77 | 7 / 7 |
| TD3 | 6,666 | 331 | 10 / 10 |
| SAC | 15,097 | 1,269 | 21 / 21 |

These counts precede the batched Gaussian readback change. In that baseline,
Gaussian validation/metrics recorded 840 readbacks across 84 calls, with physical
action validation outside the phase. See the [batched readback follow-up](gpu-batched-readback.md)
for the resulting counts and software-adapter validation. Hardware profiles are
still needed to assess latency improvements.

## Instrumentation-phase validation and test delta

- GPU-enabled native workspace: **1,118 passed, 0 failed, 193 ignored**.
- Two new adapter-independent unit tests exercise nested/repeated scopes and
  unwind behavior: **+2 passing tests** compared with the replay phase's 1,116.
- Two new adapter-required integration tests: **+2 ignored tests** in normal
  workspace execution. Both were explicitly run and passed on llvmpipe. They
  verify exact submission/transfer/allocation counts, empty/error/clear paths,
  clone/collector isolation, and identical seeded DQN losses and parameters with
  profiling on/off. DQN also verifies automatic backward/optimizer phase counts.
- Existing adapter-required tensor regression suite: **24 passed, 0 failed** on
  llvmpipe. Together with the two new integration tests, **26 adapter-required
  tests** were explicitly exercised separately from the normal workspace run.
- Three release profiling workflows and four invalid-argument cases passed the
  JSON verifier, including phase-call counts, finite diagnostics and work/wait
  consistency. Budgets are fixed; unavailable adapters fail explicitly.
- Workspace all-target/all-feature Clippy with warnings denied, formatting, Git
  whitespace checks, and Rust 1.75 GPU-enabled RL all-target compilation: passed.

## Architecture decisions

Keep instrumentation at shared tensor/autograd/affine paths so it works across
GPU agents without modifying their numerical algorithms. A caller-supplied phase
guard supports more detailed boundaries. Structured output is a developer example
for now; existing CLI metrics and training flags are preserved.

## Deviations from Plan

Actual GPU kernel durations require timestamp queries or an external GPU profiler
and are deferred. This phase supplies host timing and exact work counts. The
profiling runner covers the three replay-based algorithms; common library hooks
also support on-policy agents when enabled by their callers. CLI integration and
GPU buffer/dispatch/check consolidation remain follow-up work.

## Issue Resolution Progress

| Change | Issue | Status |
| --- | --- | --- |
| Opt-in context counters and inclusive phase scopes | Unassigned | Implemented |
| Shared backward/optimizer/affine and validation hooks | Unassigned | Implemented |
| DQN/TD3/SAC structured profiling runner | Unassigned | Verified on llvmpipe |
| Counter and numerical-equivalence regression tests | Unassigned | Passed |
| Physical GPU latency profiling | Unassigned | Deferred |

The [synchronization and scratch reuse follow-up](gpu-sync-buffer-reuse.md)
measures the new caches and batched objective checks. Scratch counters are
additive fields in the existing version-1 profiling JSON; fresh collectors may
reuse buffers already retained by their shared context.

The [GPU execution follow-up](gpu-execution-optimization.md) adds bounded queue
submission batches, recycled kernel-output storage and fused optimizer arithmetic.
`command_buffers` and `tensor_reuses` are additive version-1 JSON fields.
Compute dispatches are counted when encoded; actual queue submissions are counted
when a batch flushes. A live batch may therefore show dispatches without a matching
submission yet. Opening/flush contexts account for submissions; serialize contexts
and use one collector when comparing inclusive phase counters.
