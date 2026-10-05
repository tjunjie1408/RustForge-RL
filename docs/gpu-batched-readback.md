# Batched Gaussian validation readbacks

Gaussian validation now copies its nonfinite-check results and two diagnostic
means into one readback buffer, submits those copies together and waits once.
The reduction kernels, raw-input snapshots, mean arithmetic and host finite
checks remain unchanged. Sampled physical actions share the same validation
batch. This affects GPU SAC and continuous PPO's shared Gaussian operators.

`GpuContext::download_scalars(&[&GpuTensor])` is also available to callers that
need several scalar diagnostics. It accepts any single-element shape, preserves
input order and exact bits, and performs one submission/map/wait for a nonempty
batch. It validates all shapes and ownership before allocating or submitting.
An empty list does no work. Repeated scalar inputs are allowed.

The original single-tensor and typed-index downloads use the same mapping helper
as before. Profiling counts mapped readback **batches**, not individual scalar
copies inside a batch; byte totals retain the logical transfer volume.

## Preserved behavior

- Checks still cover raw means, unclipped log standard deviations, caller-supplied
  noise/actions, intermediate snapshots, output densities, entropy and physical
  sampled actions. Clipping or nonlinearities cannot hide nonfinite inputs.
- Individually finite densities or entropy can still overflow their mean
  reduction; the host check continues to reject this as `NonFinite`.
- Diagnostic reductions do not add an autograd graph or mutate leaf gradients.
  The existing action/log-probability graph remains differentiable afterward.
- Empty observation batches keep their existing zero-mean diagnostic behavior.
- Public diagnostic fields may be replaced with variables from other ownership
  scopes. That unusual case falls back to the original per-context validation
  and mean downloads, preserving prior results and `NonFinite` errors. Generated
  distributions use the single-context fast path.

Validation errors can still enqueue diagnostic work before returning, but no
optimizer update or training clock changes are introduced by these checks.

## Paired profiling evidence

Three pairs of release-mode runs used the same profiling driver, configuration,
seeds and Mesa llvmpipe (LLVM 19.1.7) adapter. Each run collected 70 environment
steps and completed seven learning updates. Order alternated before/after,
after/before, before/after. No compilation/test ran during measurement. The
before binary contains the instrumentation phase immediately preceding this
change, not the uninstrumented v0.2.0 release.

| SAC counter | Before | After |
| --- | ---: | ---: |
| Readback batches | 1,269 | 429 |
| Blocking waits, including final completion | 1,270 | 430 |
| Queue submissions | 15,097 | 14,257 |
| Compute dispatches | 13,828 | 13,828 |
| Readback bytes | 5,076 | 5,076 |
| Gaussian validation/metrics phase readbacks | 840 | 84 |

Overall SAC readback batches fell **66.2%**. Each generated Gaussian diagnostic
call now maps once; the phase has 84 calls. Before this change, physical-action
validation happened outside that profiling phase. The new phase includes it;
overall counts show 840 fewer waits rather than just the 756-readback phase delta.

TD3 is the control: all three pairs retained 6,666 submissions, 331 readbacks,
332 waits and 1,324 readback bytes. TD3 and SAC returned **identical seeded final
training diagnostics** in every pair. Compute dispatches, uploads/upload bytes,
tensor allocations/allocation bytes and readback bytes matched their baselines.
The optimization reduces transfer setup and synchronization, not computation or
bandwidth. Raw counts and host timings: [paired runs](gpu-batched-readback.csv).

Software-adapter host timings are reported in the CSV. These short runs include
profiling overhead and execute on a shared virtualized CPU; they do not establish
physical GPU latency improvement. SAC's median host time was 1,382.4 → 1,309.6 ms;
the unchanged TD3 control varied from 657.5 → 684.3 ms. Treat these as local
observations with substantial timing noise. Follow the [profiling guide](gpu-profiling.md)
to collect hardware results. Do not infer kernel time from inclusive host phases.

## Validation and test delta

- GPU-enabled native workspace: **1,118 passed, 0 failed, 198 ignored**.
  The five new tests increase the prior phase's ignored count from 193 to 198;
  no adapter-independent test count changes.
- Explicit software-adapter checks: **41 passed, 0 failed**: 2 new scalar tests,
  3 new Gaussian tests, 11 existing continuous-PPO/SAC objective tests, and 25
  tensor/profiling regression tests. These are separate from the normal run.
- Three paired TD3/SAC comparisons preserved exact seeded training diagnostics
  and verified the expected work/byte counts. The JSON verifier also passed all
  three algorithms and four invalid-argument cases with the updated one-readback
  Gaussian phase assertion.
- Workspace all-target/all-feature Clippy with warnings denied, Rust 1.75
  GPU-enabled RL all-target compilation, formatting and whitespace checks: passed.

Five new adapter-required tests are added: two scalar-copy API tests and three
Gaussian regression tests. They cover scalar bit patterns/order/repetition,
no-work empty/error cases, exact sampled/stored-action metrics, batch sizes 0/1/7,
NaN/±Inf in each raw input, invalid output fields, mean overflow, gradient behavior,
and the cross-context compatibility fallback. They are ignored during ordinary
CPU CI and explicitly exercised on the software adapter.

## Architecture decisions

Pack existing scalar results with buffer copies instead of adding an aggregate
floating-point reduction or new shader. This preserves exact means and per-input
checks, adds no compute dispatches and generalizes to other scalar diagnostics.
Readback packing is independent of command batching or kernel fusion.

## Deviations from Plan

Physical GPU speedup remains unmeasured. A compatibility fallback retains the
older path for externally replaced fields in different tensor ownership scopes.

## Issue Resolution Progress

| Change | Issue | Status |
| --- | --- | --- |
| One-submission/one-wait scalar readback API | Unassigned | Implemented |
| Pack Gaussian checks and diagnostic means | Unassigned | Implemented |
| Numerical, gradient and error regression coverage | Unassigned | Passed on llvmpipe |
| Paired profiling and seeded diagnostic equivalence | Unassigned | Verified |
| Physical GPU latency comparison | Unassigned | Deferred |
