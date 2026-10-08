# GPU command batching, training storage reuse and inference readbacks

All four performance tasks are implemented in their priority order:

1. `GpuContext::command_batch` groups compute command buffers into fewer queue
   submissions. GPU linear layers and SGD/Adam open scopes automatically.
2. Fully overwritten kernel outputs reuse device storage after the last live
   owner releases it, reducing training temporary allocations.
3. `download_tensors` packs arbitrary f32 shapes into one readback. TD3/SAC
   inference copies final action-validation flags and actions together. SAC's
   log-temperature check and exponentiated temperature also share a readback.
4. SGD/Adam use fused scale/add, square/scale, scale/sqrt/add and divide/scale
   kernels, reducing dispatches and intermediate tensors.

## Architecture Decisions

- Batches are thread-local and keyed by tensor ownership scope. Clones on the
  same thread share the pending group; other threads remain independent. Guards
  cannot move between threads. Finish a producing batch before passing outputs
  to another thread: a consumer cannot flush another thread's pending work.
- Nested guards flush at the outermost drop. Downloads, explicit waits and typed
  index uploads flush pending work first. Batches also flush after 32 command
  buffers. Successful/error returns and unwinding remove their scopes normally.
  Batching adds no host wait and keeps GPU commands in their encoded order.
- Deferred commands retain exclusive uniform leases and references to every
  input/output storage lease until submission. This prevents cached storage from
  being overwritten by a concurrent context clone before a pending consumer runs.
  Pools hold no device ownership references and create no ownership cycle.
- Uniform capacities are normalized to 32 bytes so matmul and elementwise layouts
  share a cache efficiently. Retention remains bounded at 32 buffers / 4 KiB;
  batching needs more simultaneous uniforms than immediate submission.
- The new tensor cache retains at most 64 idle buffers / 64 MiB per ownership
  scope. Best-fit leases may have larger physical capacity than their logical
  shape; kernels and transfers continue to address only logical elements.
  Idle-cache limits do not bound concurrent active tensors or autograd graphs.
- Public `zeros`, empty reductions requiring defined scalar/vector zeros, and
  partially written scatter outputs remain initialized. Only fully overwritten
  kernel outputs use recycled storage. Immutable forward snapshots, target leaves
  and optimizer moments stay alive until their existing owners release them.
- Optimizers still prepare all replacements before committing parameters/moments
  and clocks. Existing failed-update rollback, checkpoint continuation and
  missing-gradient behavior are unchanged.
- Inference retains the earlier network-logit checks. TD3 still validates before
  drawing CPU exploration noise; SAC keeps its existing draw order. Raw Gaussian
  inputs/intermediates, diagnostic mean overflow and physical actions are still
  checked. Foreign public diagnostic fields use the existing fallback behavior.
- Kernel fusion preserves the intended operation order and is verified at existing
  CPU/f64 tolerances. Exact seeded diagnostics match on the tested software
  adapter; hardware drivers may round fused expressions differently.

The profiling JSON retains its version-1 schema and adds `command_buffers` and
`tensor_reuses`. `submissions` counts actual queue submissions; it no longer acts
as a proxy for dispatch count. Pending batches account for encoded dispatches
before their actual submission. The opening/flush context accounts for the
submission; use one collector and serialized calls for useful phase attribution.

## Paired profiling evidence

The baseline is the working tree with the previous
[synchronization and scratch-buffer optimizations](gpu-sync-buffer-reuse.md),
captured before these four changes. It is not unmodified release v0.2.0 or plain
HEAD `fdfb225`. Both binaries use the same release runner and configuration:
70 environment steps, batch size 64, seven critic updates, three TD3 actor updates,
and seven SAC actor/temperature updates on Mesa llvmpipe (LLVM 19.1.7).

| Counter | TD3 before | TD3 after | SAC before | SAC after |
| --- | ---: | ---: | ---: | ---: |
| Queue submissions | 6,635 | 5,005 | 14,208 | 11,876 |
| Compute dispatches | 6,335 | 5,861 | 13,828 | 13,135 |
| Readback batches | 300 | 230 | 380 | 295 |
| Blocking waits, including final completion | 301 | 231 | 381 | 296 |
| Fresh tensor allocations | 6,604 | 794 | 14,849 | 2,153 |
| Recycled tensor-storage leases | 0 | 5,336 | 0 | 12,003 |
| Fresh tensor-allocation bytes | 13,154,712 | 3,700,596 | 19,114,844 | 6,264,720 |
| Logical readback bytes | 1,324 | 1,324 | 5,076 | 5,076 |
| Dispatch-uniform allocations | 2 | 32 | 1 | 32 |

Fresh tensor allocations fall **88.0% for TD3 and 85.5% for SAC**. Submissions
fall **24.6% and 16.4%**. Inference readbacks fall from three to two per action;
the earlier network guard remains one of those waits. SAC also removes one wait
per temperature readback. Neither optimization reduces required transfer bytes.
DQN retains 77 readbacks while its submissions fall from 1,090 to 581.

Three pairs per algorithm alternate before/after, after/before, before/after.
All pairs retain exact seeded training diagnostics and logical upload/readback
counts and bytes. Baseline reuse counts are inferred as zero because the earlier
implementation did not recycle tensor storage. Raw counts and host timings are
in [the CSV](gpu-execution-optimization.csv). Final measurements ran without
concurrent compilation or regression tests. Software host timing on a shared
virtualized CPU does not establish physical GPU latency gains.

## Validation and test delta

- Normal all-features workspace: **1,118 passed, 0 failed, 215 ignored**.
  Ten new adapter tests increase the previous phase's ignored count from 205
  to 215; no adapter-independent test count changes.
- New tests: seven tensor tests cover nested/readback flushing, the bounded
  batch limit, errors/unwinding, concurrent deferred storage, live snapshots and
  zero/NaN-tail reductions, mixed-shape packed reads and fused arithmetic.
  One NN test checks affine/Adam batching and frozen snapshots; one Gaussian
  test checks exact action bits, gradients and hidden invalid noise; one agent
  test checks TD3 and deterministic/stochastic SAC inference waits.
- Full adapter-required unit/integration run: **188 passed, 0 failed**
  (**178 existing + ten new** tests).
- Existing CPU/f64 gradient/optimizer parity, snapshot isolation and GPU
  tensor/profiling tests pass. Full learning, rollback, checkpoint and runtime
  verification uses adapter-required unit/integration tests, excluding ignored
  illustrative doctests and six unrelated CPU convergence gates.
- Three paired TD3/SAC profiles preserve exact seeded diagnostics. The runner
  verifier passes all three algorithms and four invalid-argument cases and
  checks command-buffer/submission accounting and storage reuse.
- Rustfmt, warning-denied all-target Clippy, default-feature workspace checks
  and Rust 1.75 GPU all-target checks pass.

## Local hardware validation

```bash
cargo build --release -p rustforge-rl --example gpu_training_profile --features gpu
python3 benchmarks/gpu_profiling/verify.py
cargo test -p rustforge-tensor -p rustforge-autograd -p rustforge-nn --features gpu \
  --lib --tests -- --ignored
cargo test -p rustforge-rl --features gpu --test gpu_inference_readback \
  --test gpu_gaussian_readback --test gpu_td3_loss --test gpu_sac_loss -- --ignored
cargo run --release -p rustforge-rl --example gpu_training_profile --features gpu -- td3 70
cargo run --release -p rustforge-rl --example gpu_training_profile --features gpu -- sac 70
```

Confirm the JSON reports your physical adapter, then repeat with your research
network sizes/batches and without concurrent testing. Compare independent
inference and training workloads: these short profiles include initialization,
profiling overhead and repeated small-network inference. Inclusive host phase
timings are not GPU kernel execution times.

## Deviations from Plan

All four tasks are complete. Command batching groups existing command buffers
rather than introducing one shared encoder; bind groups/encoders are still created
per dispatch. Training storage is recycled by lifetime rather than by mutable
optimizer arenas, preserving shared immutable snapshots. Fusion targets common
optimizer expressions; matmul/bias/activation fusion remains a separate future
optimization. Earlier inference safety boundaries remain in place. Physical GPU
measurements are deferred to local hardware.

## Issue Resolution Progress

| Feature | Issue | Status |
| --- | --- | --- |
| Batch linear-layer and optimizer commands | Unassigned | Implemented and profiled |
| Reuse released training tensor storage | Unassigned | Implemented and tested |
| Pack inference/temperature checks and readbacks | Unassigned | Implemented and tested |
| Fuse common optimizer arithmetic | Unassigned | Implemented and parity checked |
| Physical GPU performance measurements | Unassigned | Local hardware validation pending |
