# GPU finite reductions and backward execution

A subagent reviewed `fdfb225..3810158` after those commits were pushed to
`perf/replay-sampling`. The source review found no confirmed blocking defect in
buffer ownership, deferred submission, logical-length indexing or diagnostic
error ordering. It identified repeated finite-check kernels and individually
submitted backward operations as the next practical optimization targets.
The subagent also reviewed the implementation below without finding a blocking
issue. Runtime validation is reported separately from that source review.

## Implementation and Architecture Decisions

- Nonfinite detection maps exponent bits to zero/one within the first
  hierarchical reduction. Later passes sum those partial counts normally.
  Every pass respects the logical element count, excluding recycled capacity
  tails. Empty inputs still return a defined scalar zero. This removes one
  dispatch and the full-size indicator tensor for each nonempty finite check.
- `GpuVariable::backward` opens a command batch after scalar-loss and
  requires-grad validation, inside its existing profiling scope. Existing
  32-command bounds and pending storage/uniform leases apply. Leaf gradients
  are still prepared before assignment; graph traversal, accumulation order
  and existing readback boundaries remain unchanged.
- `tanh_backward_device` uses the immutable saved forward output to compute
  the existing square, negative scale, addition and gradient multiplication
  in one kernel. Autograd uses it instead of five kernels and their temporaries.
  Numerical validation uses existing CPU/f64 tolerances; hardware drivers may
  round fused expressions differently.
- TD3/SAC finite-validation, objective-diagnostic and final inference loops
  batch their reduction/addition commands up to the existing readback boundary.
  Gaussian validation batches only its compatible-context path; replaced foreign
  fields retain the existing fallback. SAC temperature checks also batch their
  count and exponentiation. Readbacks flush each group before copying results,
  preserving all validation boundaries and error ordering.

These changes add no GPU synchronization. Diagnostic validation retains
the same checked snapshots and error precedence. Profiling fields and the
version-1 JSON schema are unchanged.

## Paired profiling evidence

The baseline binary represents committed `3810158`, including the previous
GPU optimizations. Both variants use 70 environment steps, batch size 64,
seven critic updates, three TD3 actor updates and seven SAC actor/temperature
updates on Mesa llvmpipe (LLVM 19.1.7). Three pairs per algorithm alternate
before/after, after/before, before/after. All six pairs preserve exact seeded
training diagnostics, transfer counts/bytes and host wait counts.

| Counter | TD3 before | TD3 after | SAC before | SAC after |
| --- | ---: | ---: | ---: | ---: |
| Compute dispatches | 5,861 | 4,836 | 13,135 | 11,174 |
| Queue submissions | 5,005 | 2,023 | 11,876 | 6,288 |
| Backward submissions | 278 | 10 | 728 | 35 |
| Finite-validation submissions | 2,395 | 307 | 3,129 | 364 |
| Objective-diagnostic submissions | 156 | 20 | 546 | 42 |
| Readback batches | 230 | 230 | 295 | 295 |
| Host waits, including final completion | 231 | 231 | 296 | 296 |
| Fresh tensor allocations | 794 | 812 | 2,153 | 2,217 |
| Fresh tensor-allocation bytes | 3,700,596 | 4,010,448 | 6,264,720 | 6,535,968 |
| Logical readback bytes | 1,324 | 1,324 | 5,076 | 5,076 |

Queue submissions fall **59.6% for TD3 and 47.1% for SAC**; compute dispatches
fall **17.5% and 14.9%**. Deferred backward groups keep more temporaries leased
until submission, increasing cumulative fresh tensor-allocation bytes by
**8.4% and 4.3%** in these workloads. Those counters measure cumulative
allocations, not peak memory. The cache and batch limits remain unchanged.

Raw structural counters are in [the CSV](gpu-reduction-backward-optimization.csv).
This comparison reports work counts, not latency: profiling runs overlapped
regression testing/compilation. Physical GPU timing remains a local follow-up.
The DQN verifier reports 504 submissions and 77 readbacks; its replay sampling
is unseeded, so it is excluded from the exact paired diagnostic comparison.

## Validation and test delta

- Normal native all-features workspace: **1,118 passed, 0 failed, 218 ignored**.
  Three new adapter-required tests increase the previous ignored count from
  215 to 218. Existing normal passing tests are unchanged.
- Focused tensor/autograd adapter suite: **29 passed, 0 failed**
  (**26 existing + three new** tests).
- Full native adapter-required unit/integration run: **191 passed, 0 failed**
  (**188 existing + three new** tests), including learning, failed-update rollback,
  checkpoint continuation and CLI/runtime routes. As in the preceding phase,
  this run excludes ignored illustrative doctests and six unrelated CPU
  convergence gates. Python bindings are excluded from the native suites.
- Focused RL objective/Gaussian/inference suite: **8 passed, 0 failed**.
  Existing objective checks now assert one compute submission plus one readback
  for each small diagnostic, including invalid-input error paths; these assertion
  changes do not add tests. The foreign-field fallback test still passes.
- Two new tensor tests cover first-pass-only mapping, empty and multi-pass
  counts through 65,537 elements, finite extremes, NaN/infinities, cached tails,
  fused tanh arithmetic/special values and shape/ownership errors. One new
  autograd test covers shared tanh graphs, repeated backward accumulation,
  immutable saved values, submission counts and early validation/no-grad paths.
- Six paired TD3/SAC profiles preserve exact seeded diagnostics. The profiling
  verifier passes DQN, TD3 and SAC plus four invalid-argument cases.
- Rustfmt, whitespace checks, warning-denied native all-target Clippy and
  Rust 1.75 tensor/autograd/NN/RL GPU all-target checks pass.

Run the new regressions on local hardware with:

```bash
cargo test -p rustforge-tensor --features gpu --test gpu_execution -- --ignored
cargo test -p rustforge-autograd --features gpu --test gpu_autograd -- --ignored
cargo build --release -p rustforge-rl --example gpu_training_profile --features gpu
python3 benchmarks/gpu_profiling/verify.py
```

## Deviations from Plan

All four selected optimizations are implemented. Whole validation-loop batching
was added after the initial three changes passed their first regression run.
Affine matmul+bias fusion remains a follow-up requiring explicit autograd snapshot
and transpose-gradient coverage. No GPU timing claim is made from these shared
software-adapter runs.

## Issue Resolution Progress

| Feature | Issue | Status |
| --- | --- | --- |
| Review pushed GPU execution changes | Unassigned | Completed; no confirmed blocking defect |
| Fuse first nonfinite reduction pass | Unassigned | Implemented and tested |
| Batch backward traversal | Unassigned | Implemented and tested |
| Fuse tanh backward | Unassigned | Implemented and parity checked |
| Batch complete validation loops | Unassigned | Implemented and tested |
| Fuse affine matmul+bias | Unassigned | Next implementation candidate |
| Physical GPU latency/memory measurements | Unassigned | Local hardware follow-up |
