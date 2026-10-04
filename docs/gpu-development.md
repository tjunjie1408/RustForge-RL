# Phase 5: GPU implementation

The README roadmap tracks GPU support with wgpu as a Phase 5 milestone.
Planned algorithm integration is complete; physical GPU validation remains
deferred. This plan delivered the software incrementally; stages 1–3
stage 4a/4b/4c autograd, module and DQN integration, and stages 5a/5b/5c checkpoints, runtime integration and prioritized replay
are complete. Stages 6a–6c add categorical PPO losses, rollout training and
runtime/checkpoint integration. Stage 7a adds continuous Gaussian/PPO objectives;
stage 7b adds continuous agent sampling, rollout/GAE and minibatch training.
Stage 7c adds continuous runtime, Pendulum CLI routing and dual-optimizer checkpoints.
Stage 8a adds categorical A2C objective and shared-network update parity; stage 8b
adds owned agent sampling, CPU rollout/GAE and fresh-environment learning.
Stage 8c adds A2C worker/CLI routing, checkpoints and runtime controls.
Stage 9a adds the REINFORCE categorical objective, optional mean baseline and
seeded policy network. Stage 9b adds an owned REINFORCE agent, CPU episode-local
Monte Carlo returns, guarded updates and environment learning. Stage 9c adds
REINFORCE worker/CLI routing, checkpoints and runtime controls. Stage 10a adds
TD3 loss/target foundations and deterministic action scaling/smoothing. Stage 10b
adds owned TD3 replay training, caller-owned noise streams, delayed updates and
Polyak targets. Stage 10c adds Pendulum worker/CLI integration and resumable
six-network/two-optimizer checkpoints. Stage 11a adds supplied-noise squashed
Gaussian sampling, soft twin-critic targets and actor/temperature objectives.
Stage 11b adds seeded SAC replay training with three resident optimizers,
transactional updates, learned temperature and fresh continuous learning.
Stage 11c completes SAC Pendulum runtime/CLI and five-network/temperature/
three-optimizer checkpoint integration.

## Implementation stages

| Stage | Deliverable | Status and acceptance |
| --- | --- | --- |
| 1 | Explicit GPU matrix multiplication | Complete: optional feature, reusable compute context, WGSL kernel, validated inputs, readback, example, numerical tests, and CI configuration |
| 2 | Persistent device tensors | Complete: explicit upload/download, ownership checks, shared context handles, output buffer reuse, and verified chaining without intermediate transfers |
| 3 | Optimized kernels and tensor operations | Complete: tiled/direct kernels, transposed products, add/multiply/ReLU, hierarchical full sum/mean, and a release benchmark with verified outputs |
| 4a | Device autograd and optimizer foundation | Complete: reverse-mode GPU variables, CPU/finite-difference gradient parity, resident momentum SGD, and deterministic supervised convergence |
| 4b | Neural-network modules and optimizer coverage | Complete: seeded GPU Linear/ReLU/Sequential, feature-bias gradients, resident Adam, CPU model/update parity and nonlinear supervised convergence |
| 4c | RL training integration | Complete: GPU DQN/Double DQN, typed resident action selection/gather, frozen target synchronization, CPU update parity and deterministic environment convergence |
| 5a | GPU checkpoint and resume | Complete: versioned online/target/Adam state, validated atomic file replacement and bit-identical resumed updates with target cadence preserved |
| 5b | Runtime and CLI integration | Complete: explicit CPU/GPU DQN device selection, worker-owned agents, checkpoint routing, controls, CSV compatibility and selected-backend display; physical GPU benchmarking remains pending |
| 5c | GPU prioritized experience replay | Complete: resident weighted TD loss, CPU priority feedback, CLI support and version-1 checkpoint compatibility |
| 6a | GPU PPO loss foundation | Complete: stable categorical probability operators and gradients, clipped policy/value/entropy loss, CPU parity and fixed minibatch optimization |
| 6b | GPU PPO agent training | Complete: actor/critic network, rollout collection/GAE, seeded categorical sampling, shuffled partial minibatches, resident Adam and environment validation |
| 6c | GPU PPO runtime and checkpoints | Complete: worker-owned CPU/GPU selection, headless/live CartPole CLI, resumable model/Adam/configuration/clock, metrics and controls |
| 7a | Continuous-policy GPU foundations | Complete: Gaussian densities/base entropy, supplied-noise reparameterization, continuous PPO loss/gradient/Adam parity and fixed objective optimization |
| 7b | Continuous GPU PPO agent training | Complete: seeded Gaussian actor/value, caller-controlled RNGs, CPU rollout/GAE, shuffled partial minibatches and separate GPU Adam; hardware validation deferred |
| 7c | Continuous GPU PPO runtime and checkpoints | Complete: worker-owned CPU/GPU agents, Pendulum headless/live CLI, distinct dual-Adam checkpoints and controls |
| 8a | GPU A2C objective foundation | Complete: categorical actor/value/entropy objective, CPU/f64 gradient and actual seeded CPU A2C Adam parity, fixed-batch optimization |
| 8b | GPU A2C agent training | Complete: owned network/Adam/clock, seeded sampling, CPU episode-local rollout/GAE, active-row CPU parity and environment learning |
| 8c | GPU A2C runtime and checkpoints | Complete: worker-owned CPU/GPU agents, CartPole headless/live CLI, distinct shared-network/Adam checkpoints, metrics and controls |
| 9a | GPU REINFORCE objective foundation | Complete: categorical Monte Carlo policy loss, resident optional mean baseline, seeded policy and actual CPU gradient/Adam parity |
| 9b | GPU REINFORCE agent training | Complete: owned policy/Adam/clock, seeded action sampling, CPU Monte Carlo rollouts, active-row CPU parity and environment learning |
| 9c | GPU REINFORCE runtime and checkpoints | Complete: worker-owned CPU/GPU agents, CartPole headless/live CLI, distinct policy/Adam checkpoints, metrics and controls |
| 10a | GPU TD3 objective foundation | Complete: detached twin-critic targets/loss, actor objective, supplied-noise smoothing/scaling and CPU/f64 gradient/Adam parity |
| 10b | GPU TD3 agent training | Complete: owned actor/twin critics/targets, seeded continuous replay/noise, transactional delayed updates, Polyak synchronization and fresh continuous learning |
| 10c | GPU TD3 runtime and checkpoints | Complete: Pendulum worker/CLI, six-network/two-Adam snapshots, delayed-clock resume, controls and finite JSONL metrics |
| 11a | GPU SAC objective foundation | Complete: squashed Gaussian sampling, detached soft targets, actor/temperature objectives and CPU/f64 gradient/Adam parity |
| 11b | GPU SAC agent training | Complete: seeded actor/twin critics/targets, transactional three-Adam updates, replay/noise streams, CPU parity and fresh learning |
| 11c | GPU SAC runtime and checkpoints | Complete: Pendulum worker/CLI, five networks/log-alpha/three-Adam snapshots, exact resumed updates, controls and eight finite metrics |

## Architecture decisions

- The `gpu` feature belongs to `rustforge-tensor`, at the bottom of the
  dependency graph. Default builds do not enable wgpu.
- `GpuContext` establishes a tensor ownership scope over the process compute
  device, queue, cached pipelines and adapter metadata. Device resources are
  initialized once and retained until process exit; buffers and ownership
  scopes are released with their owners.
- `GpuContext::matmul` accepts CPU tensors and returns a CPU tensor. It
  explicitly uploads, dispatches, and downloads; existing tensor methods and
  autograd behavior are preserved. This is an independently usable compute
  primitive, not an accelerated training backend.
- `GpuTensor` owns contiguous device storage and immutable shape metadata.
  Explicit `upload` and `download` support scalars and arbitrary ranks;
  multiplication still requires rank two. Empty tensors retain their logical
  shape with a minimal nonzero physical buffer.
- `matmul_device` queues a product and returns its device output without host
  synchronization. Subsequent products use queue ordering to consume that
  output. `matmul_into` reuses an exact-shape output buffer; tensor data stays
  on device, although small dispatch parameter buffers and bindings are still
  allocated per operation. Zero inner dimensions clear reused output storage.
- Context clones share the same device state through `Arc`. Tensor ownership
  uses identity of that scope, not adapter metadata, so independently created
  contexts cannot mix tensors even though they share compute resources.
  Tensors retain their ownership scope until dropped.
  No reference cycle links device state back to tensors.
- Device tensors do not implement `Clone` and expose no raw buffers. Exclusive
  mutable borrowing of outputs prevents safe callers from aliasing them with
  immutable inputs during a dispatch.
- The tiled shader caches two 8×8 tiles in workgroup memory; edge invocations
  participate in barriers and output bounds checks prevent out-of-range
  writes. Transpose flags change input indexing without allocating copies.
  The original direct dot-product kernel remains available. CPU/software
  adapters default to the direct kernel following local measurements; other
  adapters default to tiling. `with_matmul_kernel` overrides that choice on a
  shared device. Noncontiguous uploads are packed in logical row-major order.
- Device elementwise operations preserve logical shape and use 256 lanes per
  workgroup. Binary operations require identical shapes, with no broadcasting.
  ReLU's comparison maps NaN to zero, matching the existing CPU `f32::max`.
- Full sum and mean use a 256-lane tree reduction, recursively reducing device
  partials to a scalar. Mean divides only on the final pass. Empty reductions
  produce zero, matching CPU behavior. Large elementwise/reduction workloads
  use a checked 2D dispatch grid with guards for extra workgroups and lanes.
- `rustforge-autograd/gpu` enables a separate fallible `GpuVariable` API,
  preserving the CPU variable API. Derivatives cover exact-shape
  add/multiply/subtract, scalar scaling, all three matrix product variants,
  ReLU, full sum/mean, and MSE. Scalar broadcasting is an explicit device
  primitive for reduction gradients. Explicit matrix + feature-vector bias
  broadcasting is also supported; general binary broadcasting, axis
  reductions remain pending; categorical softmax/log-softmax over matrix columns
  are available in stage 6a.
- Forward graphs hold immutable `Rc<GpuTensor>` snapshots. Optimizer steps
  replace leaf buffers, preserving earlier graphs and detached snapshots.
  No output backlinks create graph cycles. Backward walks the graph
  iteratively, uses fresh adjoints on every invocation, and accumulates leaf
  gradients. Intermediate gradients are not retained. Only single-element
  losses may seed backward. `no_grad` disables recording for GPU operations.
- `GpuSgd` supports vanilla and momentum SGD with device-resident state.
  `GpuAdam` maintains first/second moments on the device and uses bias
  correction matching the CPU Adam optimizer. Each successful step advances
  its global timestep, including steps without gradients; parameters with
  missing gradients retain their moments. Both optimizers validate finite
  hyperparameters and distinct trainable leaves before use.
  Trainable leaves must be unique and share a context. Parameters and
  gradients are updated only after all fallible device operations for a step
  or backward pass succeed. Forward, backward and optimizers require no tensor
  readback; factories upload data and explicit inspection downloads it.
  State and operation outputs allocate new buffers. CPU APIs still use CPU
  variables; the explicit GPU DQN API selects the new backend.
- `rustforge-nn/gpu` enables a separate fallible `GpuModule` trait with
  `GpuLinear`, `GpuReLU` and `GpuSequential`. Linear uses [out, in] weights
  and optional [out] bias, matching CPU layout and parameter order. Seeded
  Kaiming initialization happens on the CPU once and is uploaded; subsequent
  forward, backward and optimizer operations remain on device. Empty batches
  produce shaped empty outputs and zero parameter gradients. Module errors
  belong to the nn crate and wrap lower-layer errors without reversing the
  workspace dependency direction.
- Bias addition broadcasts only [features] across [batch, features]. Its
  backward sums rows into the bias shape. The correctness-oriented row-sum
  kernel assigns one feature column per lane and adds rows in order; long
  batches are not reduced hierarchically. Adam adds device square-root plus
  epsilon and exact-shape division primitives; their differentiation is not
  exposed in this stage. Dropout, normalization, serialization and other GPU
  modules are not implemented yet.
- `GpuIndices` owns typed u32 action storage with a declared feature bound.
  Uploads validate host usize actions before conversion. Shader argmax selects
  the last greatest finite value using CPU-compatible total-order keys,
  including signed-zero ties. Indices never pass through floating-point
  arithmetic/conversion. Gather produces [batch, 1]; its derivative scatters
  into [batch, actions] with one writer per element and no atomics.
- Nonfinite argmax rows produce an integer sentinel. Explicit index download
  returns an error; device gather propagates NaN instead of indexing outside
  storage. GPU action selection rejects both NaN and infinities, whereas the
  CPU argmax API rejects NaN only. Finite-value CPU parity is tested. Argmax
  itself is nondifferentiable and used for detached targets/action selection.
- `GpuVariable::copy_data_from` assigns a same-shape leaf's immutable snapshot
  and clears gradients, preserving existing forward graphs. GPU Linear frozen
  snapshots use independent nontrainable leaf handles. Sequential parameter
  synchronization validates all counts/shapes/owners before assignment and
  shares immutable buffers; later optimizer updates cannot mutate target data.
- `rustforge-rl/gpu` enables a separate fallible `GpuDqn` using the existing
  `DQNConfig` and CPU `TransitionBatch`. It supports uniform/prioritized-replay vanilla
  and Double DQN, a seeded two-layer Q-network and resident Adam. Target
  parameters are frozen and all Bellman target computation runs under
  `no_grad`. Hard synchronization follows successful training-step count;
  frequency zero disables automatic synchronization, matching CPU behavior.
- Uploaded `GpuDqnBatch` values are reusable immutable device batches. Only
  active rows are validated/uploaded; unused replay capacity is ignored.
  Shape/action bounds, finite active data and binary terminal flags are checked
  before uploads. Environment truncations must use `replay_done` so they
  continue bootstrapping. Training downloads one scalar loss before backward,
  rejects nonfinite loss without updating parameters/counters, and keeps
  predictions, target selection, TD targets, gradients and Adam state resident.
  Greedy environment interaction explicitly uploads one observation and
  downloads one integer action. Weighted batches additionally return absolute
  pre-update TD errors to CPU priority storage. The CLI/runtime explicitly
  select CPU or GPU; Python APIs continue to use CPU agents.
- `GpuAdam::state` downloads validated host moment snapshots including all
  hyperparameters and its global timestep. `restore_state` validates counts,
  shapes, finite first/second moments, nonnegative second moments and clock
  consistency before uploading any state. Every upload finishes preparation
  before the optimizer fields change; successful restoration clears gradients.
  Parameter values are separate and must be restored by the caller.
- GPU DQN checkpoint version 1 uses `RFGPUDQN` (8 bytes), a little-endian
  u32 version and a fixed-width little-endian bincode payload. Explicit
  row-major tensor shapes/values avoid deserializing ndarray storage before
  validating model dimensions. The payload records all DQN configuration,
  online and independently delayed target parameters, Adam moments and
  hyperparameters, and completed training steps. Adam/DQN clocks and learning
  rates must agree; trained DQN parameters must each have moment state.
- Checkpoint reads are bounded to 256 MiB, including the 12-byte header.
  Wrong/truncated headers, unsupported versions, truncated/trailing payloads,
  invalid counts/shapes/data lengths, nonfinite values, negative variances,
  invalid hyperparameters and inconsistent clocks are rejected before device
  allocation. `load_checkpoint` builds a new agent on the supplied context;
  `restore_checkpoint` assigns that candidate only after all device uploads
  and optimizer restoration succeed. It does not synchronize target weights.
- Saves validate and serialize before touching the destination, use an
  exclusively created temporary file in the same directory, sync and close
  it, and then rename it over the destination. A failed save preserves the
  previous file and attempts temporary-file cleanup. Parent-directory fsync
  is not performed; process-level atomic replacement is tested, not power-loss
  durability. CPU `RFPARAMS` parameter-only files and their readers stay unchanged.
- Checkpoints omit replay contents, environment state, exploration/sampling RNG
  state and live gradients. Exact resume is verified with the same batches,
  context and matrix kernel; it does not promise an identical experiment with
  a different adapter or newly sampled replay. Successful in-place restore
  replaces model handles; callers must reacquire cached parameter handles.
  Training under `no_grad` now returns an error before updating clocks/state,
  preserving the trained-parameter moment invariant.
- Rank and inner-dimension checks, checked host-to-shader dimension
  conversion, storage buffer limits, index limits, and dispatch limits are
  enforced before allocation. Empty products return correctly shaped zeros.
- wgpu 0.19 is used to retain the repository's Rust 1.75 support. The lockfile
  remains format v3.
- Driver-independent validation tests run with the GPU feature. Tests that
  execute shaders are explicitly ignored in ordinary runs and required by a
  separate CI job with a software adapter; they fail if no adapter is present.
- CPU and GPU accumulation can differ slightly. Finite comparisons use
  absolute tolerance 1e-4, with additional relative tolerance 1e-4 on larger
  products and tree reductions; separate cases check NaN and infinity behavior.

## Run locally

From the repository root, with Rust and a working compute driver installed:

```bash
cargo run --locked -p rustforge-tensor --features gpu --example gpu_matmul
cargo test --locked -p rustforge-tensor --features gpu
cargo test --locked -p rustforge-tensor --features gpu --test gpu_matmul -- --include-ignored --nocapture
cargo test --locked -p rustforge-tensor --features gpu --lib gpu::tests::persistent_operations_avoid_transfers_and_reuse_storage -- --include-ignored
cargo run --locked --release -p rustforge-tensor --features gpu --example gpu_benchmark -- 10
```

The example prints the actual adapter and verifies the result
`[[58, 64], [139, 154]]` using both the convenience CPU-to-CPU API and a chain
with a persistent intermediate and reusable output. A software adapter is
valid for shader correctness testing but does not establish hardware
acceleration.

The example also chains device ReLU into a full sum and verifies the scalar
415. The benchmark checks every measured result against CPU multiplication.

In this cloud workspace, activate the installed tools first:

```bash
export CARGO_HOME=/workspace/.rustforge-env/cargo
export RUSTUP_HOME=/workspace/.rustforge-env/rustup
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_BUILD_JOBS=4
export XDG_RUNTIME_DIR=/workspace/.rustforge-env/xdg-runtime
export MESA_SHADER_CACHE_DIR=/workspace/.rustforge-env/mesa-cache
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
cd /workspace/RustForge-RL
```

## Stages 1–3 walkthrough and validation

Added `gpu.rs`, the WGSL shader, an example, and an integration test suite.
Enabled the backend only through the tensor crate's optional feature. Added
an Ubuntu software Vulkan CI job and extended the MSRV job to check all
features. Updated the README and changelog.

Stage 2 adds persistent tensor ownership, upload/download and zero-initialized
storage APIs, shared context handles, and device-only product operations. The original
convenience API now uses these primitives. Added transfer/allocation counters
only in test builds to verify chaining avoids tensor transfers and output
buffer replacement; production builds have no counter overhead.

Stage 3 adds the tiled kernel, direct-kernel selection, copy-free left/right
transposed products and output-reuse variants, elementwise operations, and
hierarchical full reductions. Added a release benchmark with verified outputs,
warmup, synchronization, repeated trials, and adapter metadata.

Verified in the cloud workspace:

| Check | Verified result |
| --- | --- |
| Workspace tests with all features, excluding Python | 1,040 passed, 0 failed, 42 ignored |
| Explicit shader integration suite | 14 passed, 0 failed, 0 ignored on Mesa llvmpipe using the GL backend |
| Explicit storage/transfer unit test | 1 passed: stable output buffer identity, no intermediate transfers across products/elementwise/reductions, no tensor allocations during output reuse, retained device lifetime, and no ownership cycle |
| GPU example | Convenience/chained products produce the expected 2×2 result; chained ReLU/sum produces 415; adapter metadata verified |
| Release matrix benchmark | All 12 measured rows verified against CPU results; three trials at each of four matrix sizes |
| Python binding rebuild and tests | 32 passed, 0 failed |
| Workspace Clippy, all targets and features, excluding Python | Passed with warnings denied |
| Default-feature workspace check, all targets, excluding Python | Passed |
| Rust 1.75 workspace check, all targets and features, excluding Python | Passed |
| Formatting and diff whitespace checks | Passed |

Test delta from stage 2: two new driver-independent unit tests raise ordinary
workspace passes from 1,038 to 1,040. Six new adapter-required integration
tests raise ordinary ignored counts from 36 to 42. All 15 adapter-required
tests were explicitly executed and passed. Three existing GPU tests were
expanded to check both kernels' nonfinite results, foreign-device rejection
for the new operations, and transfer-free elementwise/reduction chains.
No tests were removed.

Cumulative delta from onboarding: six driver-independent unit tests and
15 adapter-required tests were added. Stage 3 checks tile edges, both matrix
kernels, transposed numerical parity/output reuse, scalar/rank/empty elementwise
operations, shape rejection, reductions around workgroup boundaries and across
multiple passes, CPU-compatible nonfinite behavior, and a 16,777,217-element
input that exercises the second row of the dispatch grid and its padding guard.

The new GitHub Actions job has been configured, not run remotely. Local shader
validation used Mesa's GL software adapter; Vulkan, physical GPUs, macOS and
Windows runtime execution remain unverified. No throughput or training speedup
is claimed.

## Performance evidence

The release benchmark compares CPU multiplication with direct and tiled
matrix kernels on the same Mesa llvmpipe (LLVM 19.1.7) GL software adapter.
The [raw CSV](gpu-benchmark-llvmpipe.csv) contains adapter metadata and all
three trials for each square-matrix size.

Inputs are deterministically seeded. GPU inputs and outputs are resident and
reused. Three warmup dispatches per kernel run before timing. Each trial times
10 products and waits for completion, then reports milliseconds per product.
GPU trial order alternates. Timings include host command submission, dispatch
parameter allocation, and queue completion; they exclude uploads, downloads,
and pipeline initialization. CPU timings include allocation of each output.
Every timed GPU result is checked against CPU values outside the timed region,
including explicit shape and finite-value checks.

Median milliseconds per product across the three trials:

| Matrix size | Native CPU | Direct device kernel | Tiled device kernel |
| --- | ---: | ---: | ---: |
| 32×32 | 0.0012 | 0.1044 | 0.1359 |
| 64×64 | 0.0079 | 0.3092 | 0.6384 |
| 128×128 | 0.0543 | 0.9103 | 2.4570 |
| 256×256 | 0.4331 | 7.8771 | 18.9476 |

This is one cloud-instance run, with visible trial variability in the raw
CSV; the table is not a cross-machine or physical-GPU performance estimate.

These software-adapter measurements show that tiling is slower than the
direct kernel here, and both are slower than native CPU multiplication. This
is why CPU adapters retain the direct kernel by default. The tiled path is
available for hardware adapters and explicit benchmarking; physical GPU
performance and end-to-end training gains remain unverified.

## Deviations from plan

- Stage 2 has no scope cuts. Per-dispatch parameter allocations remain;
  tensor storage is reused, and no allocation-free claim is made.
- Stage 4 is split into 4a (autograd/SGD), 4b (modules/Adam), and 4c
  (RL integration) to validate gradients and parameter state before adapting
  agents. Stages 4a/4b examples are supervised; stage 4c adds a deterministic
  two-state RL environment. No hardware speedup is established.
- Stage 4b uses a simple per-column row reduction for bias gradients rather
  than extending the hierarchical full-reduction kernel to arbitrary axes.
  This implements the required feature-bias contract; arbitrary broadcasting
  and axis reductions remain separate work. Adam uses composed device kernels
  and allocates new state/output buffers, with no performance claim.
- Stage 4c selects uniform-replay DQN/Double DQN as the first GPU agent.
  Other algorithms, prioritized replay, stochastic replay sampling, checkpoint
  serialization and the existing CLI training runtime are not ported in this
  stage. The deterministic example enumerates full replay through an actual
  two-state Environment to isolate bootstrapping and policy correctness.
  Network and action reductions use correctness-oriented kernels; no
  zero-allocation or training-speedup claim is made.
- Stage 5 is split into 5a (state checkpoint/resume) and 5b (CLI/runtime
  integration). The GPU format is separate from existing CPU parameter files;
  it restores training state rather than only online weights. Replay,
  environment and RNG state remain external. File replacement does not
  include parent-directory fsync, so no power-loss durability claim is made.
- Stage 3 retains the direct kernel on CPU/software adapters instead of
  universally switching to tiling: measured barrier overhead made the tiled
  shader slower in this environment. Binary broadcasting, axis reductions,
  and training-specific operations remain outside this milestone.

- Fixed the pre-existing `clippy::question_mark` warning in the SB3 format
  detector so strict workspace linting succeeds.
- Corrected the pre-existing MSRV-incompatible `backtrace` lockfile entry
  from 0.3.76 to 0.3.74, which also selects compatible `addr2line`, `gimli`,
  and `object` dependencies. Full Rust 1.75 validation then succeeded.
- The development guide references six local safety harness files that are
  absent from this checkout. Available numerical tests, deterministically
  seeded parity cases, and documented CI checks were used. The GPU paths
  make no allocation-free claim.

## Issue resolution progress

No issue IDs or issue tracker plan were supplied in this task; IDs are not
inferred or invented.

| Feature | Issue ID | Progress |
| --- | --- | --- |
| Phase 5 GPU matrix multiplication foundation | Unassigned | Implemented and locally validated |
| GPU numerical parity CI | Unassigned | Configured; remote execution pending |
| Persistent device tensors | Unassigned | Implemented and locally validated |
| Tiled/transposed products and device tensor operations | Unassigned | Implemented and locally validated |
| GPU performance comparison | Unassigned | Release benchmark and software-adapter evidence saved |
| Device autograd and momentum SGD | Unassigned | Implemented and locally validated |
| GPU neural-network modules and Adam | Unassigned | Implemented and locally validated (4b) |
| GPU DQN/Double DQN and target synchronization | Unassigned | Implemented and locally validated (4c) |
| Typed GPU action indices and gather gradients | Unassigned | Implemented and locally validated |
| GPU Adam snapshot and restore | Unassigned | Implemented and locally validated |
| GPU DQN checkpoint/resume | Unassigned | Implemented and locally validated (5a) |
| GPU CLI/runtime integration | Unassigned | Complete (5b): shared DQN loop, worker construction, explicit backend selection, checkpoint routing and control verification |


## Stage 4a usage and validation

Enable `rustforge-autograd/gpu` and create `GpuVariable` leaves on one
`GpuContext`. Construct a scalar loss with fallible GPU operations, call
`GpuSgd::zero_grad`, `loss.backward`, and `GpuSgd::step`. Device gradients
accumulate across backward calls until explicitly cleared. Frozen inputs and
no-grad operations do not retain gradients or graph history. Detached values
share the forward snapshot and do not change when an optimizer updates a leaf.

```bash
cargo test --locked -p rustforge-autograd --features gpu --test gpu_autograd -- --include-ignored
cargo test --locked -p rustforge-autograd --features gpu --lib gpu::tests -- --include-ignored
cargo run --locked -p rustforge-autograd --features gpu --example gpu_training
```

The deterministic example uploads four two-feature samples, targets and
initial weights once. It runs 100 momentum SGD steps without inspecting
intermediate losses or gradients. MSE falls from 9.75 to below 1e-6 and
weights converge to approximately `[2.000009, -3.0000134]`. Only the initial
loss, final loss and final weights are downloaded for reporting.

Seven new adapter-required tests cover CPU analytic gradient parity,
central finite differences for both operands of rectangular normal and
transposed products composed with ReLU/MSE, shared subgraphs and repeated
backward, scalar/empty reduction gradients, ReLU zero/NaN masks, graph-free
inference, saved forward snapshots, vanilla/momentum SGD numerical updates,
invalid optimizer configurations, foreign-device rejection, deterministic
training convergence, deep iterative graph traversal and graph ownership
release. Finite-difference inputs stay away from ReLU's nondifferentiable zero
boundary. The existing transfer-counter test also verifies device-only fill,
scaling, scalar expansion and ReLU backward without tensor transfers.
No tests were removed.

| Check | Verified result |
| --- | --- |
| Native workspace tests, all features excluding Python | 1,040 passed, 0 failed, 49 ignored |
| Explicit GPU autograd integration tests | 6 passed, 0 failed |
| Explicit GPU graph lifetime test | 1 passed, 0 failed |
| Existing explicit GPU tensor suite and expanded transfer test | 15 passed, 0 failed |
| Deterministic supervised training example | Converged in 100 steps; final weights within 0.001 of target |
| Python binding rebuild and tests | 32 passed, 0 failed |
| Strict workspace Clippy, all targets/features excluding Python | Passed |
| Default-feature workspace check, all targets excluding Python | Passed |
| Rust 1.75 workspace check, all targets/features excluding Python | Passed |
| Formatting and diff whitespace checks | Passed |

Delta from stage 3: ordinary passes remain 1,040, while seven new
adapter-required tests increase ordinary ignored counts from 42 to 49.
All 22 adapter-required GPU tests were explicitly executed and passed.
The GPU CI job now also runs autograd, graph lifetime and training checks;
remote CI execution remains pending. Local execution used Mesa llvmpipe GL.
Physical GPU/Vulkan execution and training speedups remain unverified.


## Stage 4b usage and validation

Enable `rustforge-nn/gpu`. Construct a `GpuSequential` from seeded
`GpuLinear` layers and `GpuReLU`, then pass `model.parameters()` to
`GpuAdam`. Calling `parameters()` returns shared handles to trainable leaves;
optimizer steps are visible to subsequent model forward calls. CPU modules
and RL agents continue to use their existing APIs. Stage 4c below adds a selected GPU RL agent with target synchronization and
checks detached estimates and deterministic training parity.

```bash
cargo test --locked -p rustforge-tensor --features gpu --test gpu_matmul -- --include-ignored
cargo test --locked -p rustforge-autograd --features gpu --test gpu_autograd -- --include-ignored
cargo test --locked -p rustforge-nn --features gpu --test gpu_modules -- --include-ignored
cargo run --locked -p rustforge-nn --features gpu --example gpu_mlp_training
```

The seeded 1→8→1 ReLU model learns `abs(x) + 1` from five fixed samples.
The example uploads samples, targets and initial parameters once, trains
for 250 Adam steps without intermediate tensor readbacks, and downloads only
initial and final loss scalars. MSE falls from 19.430431 to approximately
0.00013221. This is a supervised correctness check on a software adapter,
not evidence of physical GPU throughput or RL convergence.

Nine adapter-required tests were added: two tensor tests check bias addition,
row sums, square-root/epsilon and division across empty/scalar shapes and
workgroup boundaries, plus invalid shapes and foreign devices; three autograd
tests check bias gradients against CPU and finite differences, Adam parity
with default/custom betas and missing/zero gradients, and invalid optimizer
configurations; four module tests check supplied/seeded CPU model parity,
input and parameter gradients, multiple Adam updates, no-grad inference,
seed reproducibility, no-bias and empty-container behavior, empty batches,
invalid parameter dimensions, foreign contexts and nonlinear convergence.
The existing transfer-counter test now also checks bias addition/row sums
and Adam primitives without tensor transfers. No tests were removed.


| Check | Verified result |
| --- | --- |
| Native workspace tests, all features excluding Python | 1,040 passed, 0 failed, 58 ignored |
| Explicit GPU tensor integration suite | 16 passed, 0 failed |
| Expanded device transfer/storage test | 1 passed, 0 failed |
| Explicit GPU autograd integration suite | 9 passed, 0 failed |
| GPU graph lifetime test | 1 passed, 0 failed |
| Explicit GPU module integration suite | 4 passed, 0 failed |
| Seeded nonlinear GPU MLP example | MSE 19.430431 → 0.00013221 in 250 Adam steps |
| Python binding rebuild and tests | 32 passed, 0 failed |
| Strict workspace Clippy, all targets/features excluding Python | Passed |
| Default-feature workspace check, all targets excluding Python | Passed |
| Rust 1.75 workspace check, all targets/features excluding Python | Passed |
| Formatting and diff whitespace checks | Passed |

Delta from stage 4a: ordinary passes remain 1,040. Nine new
adapter-required tests raise ordinary ignored counts from 49 to 58.
All 31 adapter-required GPU tests were explicitly run and passed. The CI GPU
job now executes module parity and the MLP example in addition to the earlier
checks; remote execution remains pending. Local GPU execution used Mesa
llvmpipe GL; physical GPU/Vulkan training execution remains unverified.

Accumulated debug artifacts filled the workspace disk during the initial
regression build and caused linker failures before tests ran. Removed only
Cargo's generated debug cache, then rebuilt with `CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=0` and `CARGO_PROFILE_TEST_DEBUG=0`. All checks above
then passed. Source files and the saved release benchmark were preserved.
These environment options reduce disk use without enabling optimization or
removing test assertions.


## Stage 4c usage and validation

Enable `rustforge-rl/gpu`. Create `GpuDqn::new_seeded` on a shared context,
using `DQNConfig` with `use_per=false`. The external environment loop may call
`select_greedy_action`, choose exploration actions with a seeded caller policy,
and collect transitions in existing CPU replay buffers. Call `train_step`
with a CPU `TransitionBatch`, or upload once with `upload_batch` and reuse
`train_device_batch`. Each successful update returns the loss before the
optimizer update. `td_targets` exposes a detached device result for inspection;
`update_target` explicitly synchronizes frozen parameter snapshots.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_dqn -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_dqn_training
```

The example collects all four state/action transitions through the
`Environment` trait. State 0 transitions to state 1, with rewards 0/-1;
state 1 terminates with rewards -1/+1. This exercises bootstrapping rather
than treating every transition as terminal. With gamma=0.9, optimal Q-values
are `[0.9, -0.1]` and `[-1, 1]`. Seeded Double DQN trains for 300 updates;
MSE falls from 3.003548 to approximately 0.00013010. Its greedy rollout
chooses `[0, 1]` and earns return 1. The replay batch uploads once;
training downloads a scalar loss per step, and evaluation downloads actions.

Ten adapter-required tests were added: two tensor tests cover typed integer
roundtrips beyond f32 precision, argmax/gather/scatter parity across empty
batches and workgroup edges, signed-zero/tie ordering, invalid bounds/shapes,
foreign contexts and nonfinite sentinels; two autograd tests cover gather
CPU/finite-difference gradients, repeated/empty backward and validated leaf
snapshot assignment; one module test covers frozen snapshot isolation and
validated synchronization; five agent tests cover vanilla/Double DQN target,
loss, gradient and update parity, scheduled/disabled/manual target sync,
terminal/truncation behavior, deliberately different online/target action
choices, frozen target gradients, partial replay capacity, rejected malformed
inputs/foreign batches, divergence before optimizer/counter changes, seed
reproducibility, single-transition overfitting, two-state bootstrapping,
optimal Q-values and actual environment policy return. The existing
transfer-counter test now also verifies device argmax/gather/scatter without
intermediate tensor or index transfers. No tests were removed.


| Check | Verified result |
| --- | --- |
| Native workspace tests, all features excluding Python | 1,040 passed, 0 failed, 68 ignored |
| Explicit GPU tensor integration suite | 18 passed, 0 failed |
| Expanded device transfer/storage test | 1 passed, 0 failed |
| Explicit GPU autograd integration suite | 11 passed, 0 failed |
| GPU graph lifetime test | 1 passed, 0 failed |
| Explicit GPU module integration suite | 5 passed, 0 failed |
| Explicit GPU DQN integration suite | 5 passed, 0 failed |
| Seeded Double DQN environment example | MSE 3.003548 → 0.00013010 in 300 steps; greedy actions [0, 1], episode return 1 |
| Python binding rebuild and tests | 32 passed, 0 failed |
| Strict workspace Clippy, all targets/features excluding Python | Passed |
| Default-feature workspace check, all targets excluding Python | Passed |
| Rust 1.75 workspace check, all targets/features excluding Python | Passed |
| Formatting and diff whitespace checks | Passed |

Delta from stage 4b: ordinary passes remain 1,040. Ten new
adapter-required tests raise ordinary ignored counts from 58 to 68.
All 41 adapter-required GPU tests were explicitly executed and passed.
CI now also runs GPU DQN parity/convergence and the environment example;
remote execution remains pending. Local GPU execution used Mesa llvmpipe GL.
Physical GPU/Vulkan execution and end-to-end throughput gains remain unverified.
Validation used the disk-saving debug/incremental settings documented in stage 4b.

## Stage 5a checkpoint usage and validation

GPU DQN now exposes `save_checkpoint(path)`,
`GpuDqn::load_checkpoint(&context, path)`, and
`restore_checkpoint(path)`. Save/restore explicitly transfer parameter and
moment values at the checkpoint boundary. Normal training remains resident,
with its existing scalar-loss readback. Loading keeps the checkpoint's
configuration, Adam bias-correction clock, completed update count, delayed
frozen target values and synchronization cadence. It rejects CPU parameter
files rather than silently treating them as a GPU training checkpoint.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_checkpoint -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_dqn_checkpoint -- /tmp/gpu-dqn-resume.chk
```

The example saves after update 7 with a target synchronization frequency of 5.
The loaded agent then matches eight uninterrupted updates through step 15,
including synchronization at steps 10 and 15, with bit-identical loss values.
Integration tests also compare all online/target tensor bits and the entire
serialized file after continuation, covering Adam moments as well as weights.
Both vanilla/Double DQN and disabled automatic target updates are checked.

Four new driver-independent tests verify exact binary float roundtrips
(including negative zero and subnormals), malformed/version/truncation/trailing
payload rejection, parameter/config/optimizer validation before ndarray
construction, bounded reads of sparse oversized files, and atomic replacement
with cleanup after a forced rename failure. Five adapter-required tests were
added: two test custom-beta Adam snapshots, missing-gradient moments, exact
resumed updates, and validation failures preserving live gradients/state;
three test trained/untrained DQN save/load, delayed frozen targets, exact
continued updates and clocks, successful replacement semantics, malformed
header/version/shape rollback, I/O errors, and failed saves preserving the
old file. The existing DQN validation test now also verifies that training
under no-grad fails without advancing its counter. No tests were removed.

| Check | Verified result |
| --- | --- |
| Native workspace tests, all features excluding Python | 1,044 passed, 0 failed, 73 ignored |
| Driver-independent checkpoint format/I/O tests | 4 passed, 0 failed (also included in workspace total) |
| Explicit GPU tensor integration suite | 18 passed, 0 failed |
| Device transfer/storage test | 1 passed, 0 failed |
| Explicit GPU autograd integration suite | 13 passed, 0 failed |
| GPU graph lifetime test | 1 passed, 0 failed |
| Explicit GPU module integration suite | 5 passed, 0 failed |
| Explicit GPU DQN integration suite | 5 passed, 0 failed |
| Explicit GPU checkpoint/resume integration suite | 3 passed, 0 failed |
| Checkpoint resume example | Step 7 save; bit-identical continued losses through step 15 (final loss 0.145052) |
| Python binding rebuild and tests | 32 passed, 0 failed |
| Strict workspace Clippy, all targets/features excluding Python | Passed |
| Default-feature workspace check, all targets excluding Python | Passed |
| Rust 1.75 workspace check, all targets/features excluding Python | Passed |
| Formatting and diff whitespace checks | Passed |

Delta from stage 4c: four driver-independent tests raise ordinary passes
from 1,040 to 1,044. Five new adapter-required tests raise ordinary ignored
counts from 68 to 73. All 46 adapter-required GPU tests were explicitly run
and passed. One existing DQN test was expanded to cover disabled gradient
recording. No tests were removed. The CI GPU job now runs checkpoint/resume
rollback tests and the resume example; remote execution remains pending.
Local execution used Mesa llvmpipe GL. Other adapter/backend execution and
cross-device numerical resume parity remain unverified. Validation used the
disk-saving debug/incremental settings documented in stage 4b.

The RL crate now directly declares the already-locked bincode workspace
package as an optional GPU dependency. No new package versions were resolved;
CPU parameter-file behavior stays unchanged.

## Stage 5b walkthrough: runtime and CLI integration

`DqnTrainerAdapter::with_options(DqnRuntimeOptions)` selects `DqnDevice::Cpu`
(default) or `Gpu`. The adapter contains environment/configuration/paths, which
are safe to send to the training worker. The shared loop constructs the agent
and its ownership scope inside that worker; no `Rc` GPU model crosses threads.
The existing CPU `train_dqn` and `try_train_dqn` APIs still return CPU DQN agents.
CPU parameter persistence and file formats are unchanged.

Both `rustforge train dqn` and `rustforge run dqn` accept `--device cpu|gpu`.
Enable `rustforge-cli/gpu` when building the GPU CLI. GPU algorithms other than
DQN and missing compile-time support return explicit errors
before opening metrics files. There is no implicit CPU-agent fallback. The wgpu
adapter can be software; selected-backend display does not imply physical GPU
hardware. Live configuration and manifests record the requested backend and
checkpoint paths. Resumed configuration is labeled as coming from the checkpoint
instead of displaying fresh-run defaults as if they were restored settings.

`--resume` routes through `GpuDqn::load_checkpoint`, with the saved configuration,
online/target parameters, Adam state and update clock intact. The selected
environment must match the observation and action dimensions. Replay, environment,
exploration RNG, episode counters and environment-step counters restart. The
original warmup (128 transitions), batch size (32), epsilon schedule, bootstrapping,
CSV schema and runtime metric IDs remain shared across backends. GPU exploration
uses a lazy greedy callback: random action decisions stay on CPU; greedy decisions
read back one typed index. Sampled replay batches are uploaded once per update.

`--checkpoint` saves full GPU state after normal completion or either controlled
stop. Saves atomically replace the requested path, which may be the resume path.
Failed initialization, resume or training does not save partial state; checkpoint
write failures fail the run. Pause/resume and graceful/force stop use existing live
controls. The interactive checkpoint key is still unsupported, so metadata keeps
that capability disabled. Checkpoint paths must differ from headless metrics and
stay outside explicit live output directories to avoid artifact replacement.

```bash
cargo run --locked -p rustforge-cli --features gpu -- train dqn --device gpu --env gridworld --episodes 20 --no-log --checkpoint /tmp/gpu-dqn.chk
cargo run --locked -p rustforge-cli --features gpu -- train dqn --device gpu --env gridworld --episodes 20 --no-log --resume /tmp/gpu-dqn.chk --checkpoint /tmp/gpu-dqn.chk
# Requires an interactive terminal:
cargo run --locked -p rustforge-cli --features gpu -- run dqn --device gpu --checkpoint /tmp/gpu-live.chk

cargo test --locked -p rustforge-rl --features gpu --test gpu_runtime -- --include-ignored
cargo test --locked -p rustforge-cli --features gpu --test device_selection -- --include-ignored
```

### Deviations from Plan (stage 5b)

- Simultaneous caller-side checkpoint inspection and worker construction exposed
  wgpu 0.19 EGL display teardown and independent context-lock failures. Sharing
  one adapter also exposed queue-ID conflicts when requesting multiple devices.
  `GpuContext::new` now initializes one process compute device and cached pipelines
  under a lock, retaining those resources until exit. Each new context still has
  an independent ownership scope and rejects foreign tensors; clones share their
  scope. Buffers, graphs and optimizer state are released with their owners. This
  intentionally changes compute-device lifetime and prevents repeated independent
  device creation. Concurrent worker/inspection tests exercise this behavior.
- CPU runtime checkpoint flags remain unavailable; the existing CPU library
  parameter persistence API is preserved. GPU resume uses the complete GPU format
  and rejects legacy CPU files. Complete experiment persistence and interactive
  checkpoint commands remain separate work.
- Hardware adapter selection, device-loss recovery and physical-GPU throughput
  tuning are not added. All adapter-required checks ran on Mesa llvmpipe GL.

### Issue Resolution Progress (stage 5b)

| Work | Issue ID | Result |
| --- | --- | --- |
| Explicit device selection and worker construction | Unassigned | Complete, shared headless/live DQN loop with CPU default |
| GPU checkpoint CLI routing and target cadence | Unassigned | Complete, resumed updates match uninterrupted updates; controlled stops preserve lag |
| GPU lifecycle across workers and inspection | Unassigned | Complete, process compute resources with isolated ownership scopes |
| Unsupported option and output-alias handling | Unassigned | Complete, errors precede output truncation/device construction |
| Hardware performance and full experiment continuation | Unassigned | Pending |

### Verification (stage 5b)

| Check | Passing result | Delta from stage 5a |
| --- | --- | --- |
| Native workspace, all features, excluding Python bindings | 1,050 passed, 77 ignored, 0 failed | +6 ordinary tests; +4 adapter-required ignored tests |
| Adapter-required GPU checks executed explicitly | 50 passed, 0 failed | Previous 46 +3 runtime worker tests +1 CLI checkpoint test |
| Default-feature device/CLI/runtime checks | 13 passed, 0 failed | Includes 2 feature-unavailable rejection tests excluded by all-feature builds |
| Python bindings | 32 passed, 0 failed | No API changes |
| Workspace Clippy, all targets/all features, warnings denied | Passed | Runtime/CLI and test additions included |
| Rust 1.75, workspace all targets/all features | Passed | Optional CLI GPU feature included |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The six ordinary additions cover seeded lazy epsilon selection/error propagation,
invalid runtime options before environment interaction, CLI flag parsing, rejected
CLI device combinations before output truncation, checkpoint/metrics alias
rejection, and live training-plan construction without initializing an adapter.
Existing CLI fixtures add default execution options. The three adapter-required
runtime tests cover actual worker ownership, exact resumed updates with restored
hyperparameters, synchronization at the saved cadence, controlled stops, pause,
invalid resume before environment interaction and checkpoint-save failure cleanup.
The CLI GPU test checks full-state routing and unchanged CSV persistence. Remote
CI and physical hardware results are not claimed.

## Stage 5c walkthrough: GPU prioritized replay

`GpuDqn::train_step_with_weights(batch, Option<&Tensor>)` adds CPU-compatible
weighted updates without changing the existing scalar-returning `train_step`
API. `upload_batch_with_weights` retains immutable frozen importance weights in
a reusable `GpuDqnBatch`; `train_device_batch_with_td_errors` returns the scalar
loss and optional priority errors. With no weights, the old uniform path still
reads back only the loss scalar.

The weighted path computes `mean(w * (Q(s,a) - target)^2)` on the GPU. It divides
by active batch size, matching CPU DQN; it does not normalize by the sum of
weights. Priority errors are absolute, unweighted TD differences from before the
optimizer update. Weights have shape `[capacity, 1]`, with enough active rows;
only active values must be finite and nonnegative. Zero weights are allowed.
Weight validation precedes device uploads; nonfinite TD differences or losses
fail before gradients, optimizer state, parameters or training clocks change.
Weights are frozen and Bellman targets remain detached.

The shared runtime already implements stratified CPU PER, priority updates and
beta annealing. Its GPU backend now uploads the sampled batch and weights and
returns TD errors through that loop. Alpha remains 0.6; beta increases from 0.4
to 1.0 using the saved `per_beta_annealing_steps` configuration and this run's
environment-step counter. GPU PER configurations require a positive annealing
length. Uniform configurations retain their previous validity rules.

Both CLI modes accept `--device gpu --use-per`. GPU resume restores the checkpoint's
PER setting and beta schedule even when the flag is omitted. Saved settings are
authoritative; the live manifest labels a resumed command's flag as
`requested_use_per`. Checkpoint version 1 already includes both fields, so no
wire-format change is necessary. Existing uniform files remain readable; older
binaries that reject PER configuration cannot load new PER files. Replay
contents, priorities, maximum priority and sampler RNG remain outside the agent
checkpoint and restart on resume.

```bash
cargo run --release --locked -p rustforge-cli --features gpu -- train dqn --device gpu --use-per --env gridworld --episodes 20 --no-log --checkpoint gpu-per.chk
cargo run --release --locked -p rustforge-cli --features gpu -- train dqn --device gpu --env gridworld --episodes 20 --no-log --resume gpu-per.chk --checkpoint gpu-per.chk
# In an interactive terminal:
cargo run --release --locked -p rustforge-cli --features gpu -- run dqn --device gpu --use-per --checkpoint gpu-per-live.chk

cargo test --locked -p rustforge-rl --features gpu --test gpu_dqn --test gpu_checkpoint --test gpu_runtime -- --include-ignored
cargo test --locked -p rustforge-cli --features gpu --test device_selection -- --include-ignored
```

### Deviations from Plan (stage 5c)

- Weighted loss composes existing device subtract/multiply/mean operations;
  no new shader or autograd operator is needed. Absolute errors are calculated
  from the explicit TD readback on CPU, where the priority tree lives.
- Replay data and sampler state are not added to checkpoints. Bit-identical
  continuation uses identical externally supplied batches/weights; fresh runtime
  replay remains stochastic. Physical GPU performance testing stays deferred.

### Issue Resolution Progress (stage 5c)

| Work | Issue ID | Result |
| --- | --- | --- |
| Weighted GPU TD loss and reusable weights | Unassigned | Complete, CPU loss/gradient/update parity for vanilla and Double DQN |
| CPU priority feedback from GPU errors | Unassigned | Complete, seeded stratified replay agrees with CPU across priority updates |
| Headless/live CLI PER and resume | Unassigned | Complete, saved mode and beta settings restored through shared runtime |
| Checkpoint compatibility and validation | Unassigned | Complete, version-1 codec accepts valid PER; zero annealing length and invalid weights rejected |
| Physical GPU profiling and complete replay persistence | Unassigned | Deferred/separate milestones |

### Verification (stage 5c)

| Check | Result | Delta from stage 5b |
| --- | --- | --- |
| Native workspace, all features, excluding Python bindings | 1,051 passed, 83 ignored, 0 failed | +1 ordinary codec test; +6 adapter-required ignored tests |
| GPU DQN, checkpoint and runtime suites executed | 16 passed, 0 failed | 8 DQN +4 checkpoint +4 runtime tests |
| CLI device suite executed with ignored tests included | 4 passed, 0 failed | 2 driver-independent +2 adapter-required tests |
| Default-feature device/CLI/runtime checks | 13 passed, 0 failed | Feature-unavailable rejection preserved |
| Workspace Clippy, all targets/all features, warnings denied | Passed | Includes new weighted paths and fixtures |
| Rust 1.75, workspace all targets/all features | Passed | Locked dependency versions unchanged |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The ordinary addition round-trips PER configuration in the existing codec and
checks uniform-file compatibility. Six adapter-required additions cover weighted
CPU parity, seeded priority feedback, invalid/zero weights and TD overflow rollback,
exact weighted checkpoint continuation, runtime restoration of saved PER settings,
and CLI PER save/resume. Existing fixtures that rejected all PER requests now
reject zero annealing lengths or check supported configurations. Previous tensor,
autograd and NN GPU suites have no implementation changes in this stage; Python
bindings have no API changes. GPU execution used Mesa llvmpipe GL; remote CI and
physical GPU performance results are not claimed.

## Stage 6a walkthrough: categorical PPO loss foundation

Tensor device operations now support exponential, stable log-softmax over the
columns of `[batch, actions]`, softmax, the log-softmax vector-Jacobian product,
and a scalar nonfinite-value count. Log-softmax subtracts the row maximum before
summing exponentials, avoiding exp/log cancellation for very unlikely actions.
Empty batches retain their shape; an empty action axis is rejected. Nonfinite
logits produce NaN across their entire row. Large finite differences outside the
f32 representable range can still overflow; checked loss diagnostics reject them.

`GpuVariable::exp` and `log_softmax` save immutable forward outputs for backward.
Their derivatives are `g * exp(x)` and `g - exp(log_probs) * sum(g, actions)`.
`softmax` composes these operators. `minimum` and `clamp` compose existing ReLU
operations, matching CPU RL utilities exactly: minimum ties select the right
operand; clamp passes gradient at the lower bound and stops it at the upper
bound. General binary broadcasting and arbitrary axis reductions are unchanged.

`agent::gpu_ppo::categorical_policy_loss` computes the gathered action log
probabilities, importance ratios, clipped surrogate and mean categorical entropy.
`discrete_ppo_loss` combines that objective with value MSE and entropy coefficients
from `GpuPpoLossConfig`. Old log probabilities, advantages and returns are
explicitly detached even when passed as trainable variables. The caller prepares
advantages and typed action indices; reference shapes are `[batch, 1]` and must
match a nonempty logits batch. All tensors must share the same ownership scope.
Clip epsilon is finite in `[0, 1)`; value and entropy coefficients are finite and
nonnegative.

Constructing the loss graph performs no host readback. `GpuPpoLoss::checked_metrics`
is an explicit validation/diagnostic boundary before applying optimizer updates.
It downloads four scalar losses/entropy plus a scalar nonfinite-ratio indicator.
Clipping can hide infinity in a finite surrogate, but the exp derivative can then
produce NaN through `infinity * 0`; the ratio guard rejects this case. Callers must
run the check before backward and optimizer updates. Low-level autograd operations
still expose their ordinary shader floating-point behavior.

The example uses seeded GPU Linear actor and critic heads, a fixed four-state
minibatch, captured old policy probabilities and resident Adam. After 80 updates,
verified software-adapter output is:

```text
Fixed GPU PPO minibatch: total loss 0.366946 -> -0.191857, value loss 0.000173
```

```bash
cargo run --locked -p rustforge-rl --features gpu --example gpu_ppo_objective
cargo test --locked -p rustforge-rl --features gpu --test gpu_ppo_loss -- --include-ignored
cargo test --locked -p rustforge-tensor --features gpu --test gpu_matmul -- --include-ignored
cargo test --locked -p rustforge-autograd --features gpu --test gpu_autograd -- --include-ignored
```

### Deviations from Plan (stage 6a)

- PPO is split into loss foundations (6a), agent/rollout training (6b), and later
  runtime/checkpoint integration. Existing GPU CLI device selection remains DQN
  only. The fixed minibatch example validates loss optimization; environment
  learning, sampling, GAE integration and end-to-end PPO remain stage 6b work.
- Categorical kernels use a correctness-oriented sequential row scan per output
  element, with quadratic work in the action count. Existing shared shader
  bindings and dispatch bounds are reused. Parallel row reductions and fused
  PPO kernels need later measurement; no speedup is claimed.
- No separate clamp/minimum shader is added: matching CPU's composed boundary
  gradient contract takes precedence. A device scalar finite-value guard is added
  because clipping alone can mask invalid ratios.

### Issue Resolution Progress (stage 6a)

| Work | Issue ID | Result |
| --- | --- | --- |
| Stable categorical probabilities and device derivatives | Unassigned | Complete, extreme logits, singleton/empty batches and CPU/finite-difference validation |
| Clipped surrogate, value loss and categorical entropy | Unassigned | Complete, CPU losses/gradients and four Adam updates agree |
| Rollout reference detachment and invalid objective rejection | Unassigned | Complete, frozen reference gradients and overflow checks |
| Fixed GPU actor/critic minibatch optimization | Unassigned | Complete, checked loss decreases with resident Adam |
| PPO rollout, sampling, mini-batches and runtime | Unassigned | Next (6b and later integration) |

### Verification (stage 6a)

| Check | Passing result | Delta from stage 5c |
| --- | --- | --- |
| Native workspace, all features, excluding Python bindings | 1,052 passed, 90 ignored, 0 failed | +1 ordinary configuration test; +7 adapter-required tests |
| Adapter-required GPU checks executed explicitly | 63 passed, 0 failed | 20 tensor +1 storage +15 autograd +1 graph +5 NN +19 RL +2 CLI |
| Fixed GPU PPO minibatch example | Passed | Total loss 0.366946 to -0.191857; final value MSE 0.000173 |
| Default-feature workspace/all-targets check | Passed | GPU example gated by required features; CPU APIs preserved |
| Workspace Clippy, all targets/all features, warnings denied | Passed | Includes new operators, helper module and example |
| Rust 1.75, workspace all targets/all features | Passed | No new dependencies or lockfile changes |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The ordinary addition checks clip/coefficient configuration without a driver.
Seven adapter-required additions cover stable tensor probabilities and backward
shape/ownership errors (2), CPU/finite-difference autograd and immutable snapshots
with no-grad/empty handling (2), complete PPO losses and optimizer parity,
clipping boundary/tie gradients, and invalid objective validation (3). The existing
storage/transfer unit now includes probability and finite-check operations before
asserting no intermediate transfers. The existing graph-lifetime unit covers saved
categorical outputs without graph cycles. DQN/PER, checkpoint, runtime and CLI
GPU suites were rerun because the shared shader and autograd implementation changed.
GPU execution used Mesa llvmpipe GL. Python APIs have no changes; physical hardware
and remote CI results are not claimed.

## Stage 6b: discrete PPO agent and environment rollouts

`GpuPpoDiscrete` uses the CPU PPO shared-trunk architecture, parameter ordering
and seeded initialization: Linear/ReLU trunk, categorical actor and scalar critic.
Model parameters, gradients and Adam moments stay on device. Sampling downloads
log probabilities and value at the environment boundary and consumes a caller-owned
RNG. Training uses an independently supplied shuffle RNG, the CPU active-row
advantage normalization, in-place epoch shuffles, exact partial minibatches and
mean metrics over updates. `updates()` counts completed Adam steps.

`collect_rollout_with_rng` accepts an environment, episode count, step limit,
optional reset seed and sampling RNG. It calls the shared CPU `RolloutBuffer`/GAE
for each episode before concatenating the results. True termination bootstraps
zero; truncation and step limits bootstrap the critic at the final observation.
There is no GAE propagation across resets. A reset seed is advanced per episode;
callers control the seed for each subsequent rollout.

Configuration, active input shapes, action bounds, finite values and normalized
advantage overflow are checked before training mutations. Inactive capacity tails
are ignored. Each minibatch checks losses/ratios before backward, then checks device
gradients with scalar readbacks before Adam. Runtime failure retains earlier
completed minibatch updates; the whole multi-epoch call is not transactional.
Invalid observations, mismatched environment dimensions and action conversions
return errors. Collection advances the environment and sampling RNG even if a
later transition fails. CPU policy/GAE and host minibatch uploads are deliberate
boundaries; device-native collection and fused training kernels are later work.

```bash
cargo run --locked -p rustforge-rl --features gpu --example gpu_ppo_training
cargo test --locked -p rustforge-rl --features gpu --test gpu_ppo_agent -- --include-ignored
```

Verified software-adapter example output:

```text
GPU PPO bandit: rewarding action probability 0.046995 -> 0.995741; 60 Adam updates
```

### Deviations from Plan (stage 6b)

- Multi-episode collection computes GAE in separate episode buffers and concatenates
  batches, preserving bootstrap semantics across truncations. The environment
  demonstration uses a one-step bandit; it verifies learning from fresh rollouts,
  rather than claiming CartPole convergence or physical GPU performance.
- The library agent is complete for this stage. GPU PPO live/headless runtime
  selection and checkpoint persistence remain stage 6c; the CLI still routes GPU
  requests only to DQN. No checkpoint/RNG persistence contract is added here.
- Objective and gradient safety checks use explicit scalar readbacks per minibatch.
  These checks prioritize numerical validity; profiling is deferred.

### Issue Resolution Progress (stage 6b)

| Work | Issue ID | Result |
| --- | --- | --- |
| Seeded shared actor/critic and categorical sampling | Unassigned | Complete, CPU parameters/forward and identical sampling streams agree |
| CPU rollout and per-episode GAE integration | Unassigned | Complete, terminal/truncation/step-limit bootstrap tests |
| Shuffled epochs and partial minibatches with resident Adam | Unassigned | Complete, CPU losses and updated policy/value agree over 18 updates |
| Invalid inputs and numerical validation before updates | Unassigned | Complete, zero updates and unchanged parameters for rejected batches |
| Deterministic environment learning | Unassigned | Complete, rewarding probability 4.7% to 99.6% in 60 updates |
| GPU PPO runtime and checkpoint integration | Unassigned | Next (6c) |

### Verification (stage 6b)

| Check | Passing result | Delta from stage 6a |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,052 passed, 94 ignored, 0 failed | +4 adapter-required agent tests; ordinary count unchanged |
| PPO adapter-required tests explicitly executed | 7 passed, 0 failed | 4 new agent tests +3 existing loss tests rerun |
| Seeded GPU PPO environment learning example | Passed | New example, 0.046995 to 0.995741 rewarding probability |
| Workspace Clippy/all targets/all features, warnings denied | Passed | New agent, tests and example |
| Rust 1.75 workspace/all targets/all features | Passed | No dependency or lockfile changes |
| Default-feature workspace/all-targets check | Passed | GPU examples remain gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The four new tests cover architecture/initialization and seeded sampling plus CPU
training parity with inactive NaN tails and partial minibatches; terminal,
truncation and step-limit GAE; environment learning; and invalid configuration,
observations, active data, overflow, empty batches and disabled gradients. GPU
execution used Mesa llvmpipe GL. The lower tensor/autograd/NN kernels did not
change during stage 6b; their prior 63-check validation remains recorded under 6a.
GPU CI now explicitly runs the four agent tests and environment-learning example.
Python APIs are unchanged; physical GPU testing remains deferred.

## Stage 6c: PPO runtime, CLI and training-state checkpoints

`PpoDiscreteTrainerAdapter::with_options(PpoRuntimeOptions)` selects CPU (default)
or GPU. Options are plain values; the runtime constructs the agent, context and
checkpoint inside the owning worker. The shared PPO loop retains seeded action/
shuffle streams, per-episode GAE, metrics, pause/resume, graceful stop and force
stop. GPU model and loaded configuration must match the environment observation
and discrete action dimensions before any reset. Saved gamma/lambda, epochs,
minibatch size and learning rate are authoritative when resuming.

CartPole supports GPU PPO in headless `train` and live `run`. The live plan uses
the existing generic JSONL metrics schema and displays the requested device.
Resume displays configuration as restored, rather than reporting fresh defaults.
Unsupported algorithms, CPU checkpoint options, missing GPU builds, PER for PPO,
unsupported environments and checkpoint/metrics path aliases are rejected through
the existing validation paths. Interactive checkpoint requests remain unsupported;
checkpoint files are written on successful completion or controlled stops.

```bash
cargo run --locked -p rustforge-cli --features gpu -- train ppo --device gpu --episodes 10 --checkpoint target/ppo.chk
cargo run --locked -p rustforge-cli --features gpu -- train ppo --device gpu --episodes 10 --resume target/ppo.chk --checkpoint target/ppo.chk
cargo run --locked -p rustforge-cli --features gpu -- run ppo --device gpu --episodes 10 --resume target/ppo.chk --checkpoint target/ppo.chk
cargo test --locked -p rustforge-rl --features gpu --test gpu_ppo_checkpoint --test gpu_ppo_runtime -- --include-ignored
```

PPO checkpoint v1 begins with `RFGPUPPO` and a little-endian u32 version. The
fixed-integer little-endian bincode payload includes discrete PPO configuration,
six actor/critic tensors, Adam configuration and optional moments, and the update
clock. DQN's `RFGPUDQN` format is unchanged. The complete file is bounded to 256
MiB; oversized files, truncated/wrong-algorithm headers, unknown versions,
trailing bytes, inconsistent shapes/counts, nonfinite values, negative variances,
invalid hyperparameters, missing moments after updates, and clock/lr mismatches
are rejected. Host validation precedes GPU tensor allocation and ndarray
construction. A load creates a new agent; restore replaces the live agent only
after success, so handles acquired before restore still reference the old model.

Saves capture and validate all device state before creating a temporary file,
sync and close it, and rename it in the destination directory. Validation errors
preserve an existing checkpoint; failed I/O cleans up temporary files. Parent
directories must exist. Resume restores model/Adam/configuration/update state,
not experiment state: environment, rollout, RNG streams, moving reward window,
run counters and metrics restart. Bit-identical continuation is verified only
for identical input batches and caller-supplied shuffle streams. A force-stop
mid-episode discards its untrained partial rollout and saves completed updates;
a force-stop at the episode boundary completes that episode's training before
saving. Failed runs do not replace the requested checkpoint.

### Deviations from Plan (stage 6c)

- PPO uses a separate checkpoint signature and validates its shared actor/critic
  architecture. DQN checkpoint bytes and public APIs are preserved. PPO config
  gains Clone/Debug/PartialEq/Serde implementations to support snapshots without
  changing CPU training behavior.
- Interactive checkpoint control is still unsupported, matching DQN runtime;
  save-on-completion/controlled-stop is the concrete supported contract.
- The runtime collects one episode per training batch to retain CPU PPO behavior.
  Vectorized/multi-episode runtime collection and complete RNG/environment
  persistence remain separate work.
- Workspace validation exposed the existing unseeded CPU XOR test failing at
  loss 0.2502. It now uses seeds 42/43 with unchanged convergence assertions;
  verified loss is 0.000998. This is a test reproducibility repair.

### Issue Resolution Progress (stage 6c)

| Work | Issue ID | Result |
| --- | --- | --- |
| Worker-owned GPU PPO backend and saved configuration | Unassigned | Complete, resume/shape validation and CPU runtime regression checks |
| Versioned PPO model/Adam checkpoint and atomic restore | Unassigned | Complete, bit-identical continuation and rejected malformed/failed saves |
| Headless/live CLI routing and restored-config display | Unassigned | Complete, headless GPU CartPole save/resume and live-plan schema/display checks |
| Metrics, pause/resume and controlled-stop semantics | Unassigned | Complete, finite JSONL, worker ownership and partial-rollout discard tests |
| CPU XOR convergence reproducibility | Unassigned | Complete, fixed initialization with original assertions retained |
| Continuous-policy GPU foundations | Unassigned | Next (7a) |

### Verification (stage 6c)

| Check | Passing result | Delta from stage 6b |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,056 passed, 101 ignored, 0 failed | +3 checkpoint codec/validation/I/O tests; +1 runtime-options test; +7 adapter-required tests |
| Adapter-required GPU RL regression tests executed | 21 passed, 0 failed | PPO loss 3 +agent 4 +checkpoint 2 +runtime 4; existing DQN checkpoint 4 +runtime 4 |
| GPU CLI device-selection suite | 5 passed, 0 failed | 3 adapter-required (including new PPO save/resume), 2 ordinary validation tests |
| Seeded CPU XOR convergence | Passed | Existing test modified, loss 0.000998; no test-count delta |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Includes checkpoint codec, runtime, CLI and tests |
| Rust 1.75 workspace/all targets/all features | Passed | No new dependencies or lockfile changes |
| Default-feature CLI/runtime tests and workspace check | 23 passed, 0 failed; check passed | CLI device validation 2 +headless 7 +PPO runtime 14 |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

Seven new adapter-required tests cover PPO exact checkpoint continuation,
untrained/failed restore and save preservation (2); saved configuration/finite
metrics, graceful/force stops, invalid resume/dimensions/observations and worker
pause/resume (4); and CLI CartPole save/resume/JSONL output (1). Existing live-plan
and device validation tests now cover PPO. The three new codec tests run without
an adapter and exercise malformed metadata, file limits and atomic write cleanup.
GPU CI explicitly runs the codec, PPO runtime/checkpoint and CLI suites. Execution
used Mesa llvmpipe GL; physical hardware and interactive terminal rendering were
not tested. Python APIs have no changes.

## Stage 7a: continuous Gaussian operators and PPO objectives

The tensor backend adds natural logarithm, stable tanh, numeric clipping for
frozen inputs, action-column sums `[batch, actions] -> [batch, 1]`, and their
column broadcast. GPU autograd adds log/tanh/exact-shape division/column-sum
backward. Saved inputs/outputs are immutable snapshots; repeated backward,
no-grad, empty tensors and graph lifetime retain their existing contracts.
Column reduction uses a correctness-oriented sequential scan per row.
Tanh uses a stable negative-exponential expression, saturating at infinities
and preserving signed zero. Log returns -infinity for zero and NaN for negative
or NaN inputs. Numeric detached clipping differs from the composed differentiable
clamp: distribution parameter clipping retains CPU's boundary gradients.

`GpuGaussianTransform::new(context, low, high)` validates finite strictly ordered
bounds, finite positive scale and finite bias, and uploads frozen per-action
constants once. The input/output probability calculations stay on device:

- `log_prob_from_action(mean, raw_log_std, actions)` detaches stored actions,
  reverses scaling, clamps normalized actions to `[-1+1e-6, 1-1e-6]`, applies
  the inverse tanh, and evaluates diagonal Gaussian density plus tanh/scaling
  Jacobian corrections. Finite out-of-bounds actions follow CPU's clipping rule.
- `sample_with_noise(mean, raw_log_std, noise)` detaches supplied standard-normal
  noise, reparameterizes, squashes/scales actions and retains gradients to both
  distribution outputs. RNG ownership and actual noise generation are later
  agent responsibilities.
- Raw log std is differentiably clamped to `[-20, 2]`, matching CPU. The
  `base_entropy` output is the analytic entropy of the unsquashed diagonal
  Gaussian. It is not the exact entropy of the transformed action distribution.

`continuous_ppo_loss(GpuContinuousPpoInputs, transform, clip_eps)` builds clipped
policy and value MSE graphs with frozen actions, old log probabilities, advantages
and returns. It intentionally returns separate losses: current CPU continuous
PPO uses separate actor/critic Adam updates and does not apply the configured
value/entropy coefficients. Checked metrics validate immutable snapshots of raw
inputs/references, log densities, ratios and objective values using explicit
scalar readbacks. This catches invalid inputs hidden by clipping and overflowing
ratios hidden by a finite clipped loss. Call these diagnostics before backward;
any future trainer must also validate gradients before optimizer updates.

```bash
cargo run --locked -p rustforge-rl --features gpu --example gpu_continuous_ppo_objective
cargo test --locked -p rustforge-rl --features gpu --test gpu_continuous_ppo -- --include-ignored
```

Verified software-adapter fixed minibatch output after 80 actor/critic updates:

```text
Fixed continuous GPU PPO: policy -0.000000 -> -0.200000, value 0.979212 -> 0.000122
```

### Deviations from Plan (stage 7a)

- CPU continuous PPO already uses tanh squashing and action scaling, so matching
  its Gaussian/Jacobian/inverse-action behavior is included. Supplied-noise
  reparameterization is also exposed to validate the new tanh gradients. There is
  no full GPU Gaussian network/agent, environment trainer, RNG owner or runtime
  routing in this stage; the example optimizes a fixed minibatch.
- Base Gaussian entropy is diagnostic, with no entropy bonus or shared weighted
  total objective, preserving current CPU continuous PPO's separate updates.
- Boundary parity exposed a CPU precision problem: this Rust build's f32 atanh
  evaluated the negative clamped endpoint as -7.219154 versus f64's -7.247733.
  CPU stored-action inversion now uses f64 before converting to f32, making
  endpoint densities symmetric and aligning the GPU's stable inverse. This
  deliberately corrects near-boundary likelihoods for CPU Gaussian consumers
  (continuous PPO/SAC); sample generation and action clipping conventions stay
  the same. A new non-GPU test checks symmetry and the f64 density reference.
- The inverse action is detached, so its device numeric clip/log composition
  needs no atanh derivative. General inverse-hyperbolic autograd and fused
  Gaussian kernels remain later work.

### Issue Resolution Progress (stage 7a)

| Work | Issue ID | Result |
| --- | --- | --- |
| Device log/tanh, division gradients and action-column reductions | Unassigned | Complete, CPU/f64 gradients, shapes, owners, empty tensors and snapshots |
| Squashed/scaled Gaussian densities and base entropy | Unassigned | Complete, actual CPU GaussianPolicy parity including endpoints/outside bounds |
| Reparameterized supplied-noise action/density gradients | Unassigned | Complete, f64 finite differences and frozen-noise/no-grad checks |
| Continuous PPO policy/value losses and separate Adam updates | Unassigned | Complete, detached references and four CPU/GPU updates agree |
| CPU endpoint inverse precision | Unassigned | Complete, symmetric f64-reference density regression |
| Fixed continuous objective optimization | Unassigned | Complete, clipped policy improvement and value MSE below 0.001 |
| Continuous GPU agent, rollout and runtime | Unassigned | Next (7b and subsequent integration) |

### Verification (stage 7a)

| Check | Passing result | Delta from stage 6c |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,057 passed, 109 ignored, 0 failed | +1 CPU endpoint test; +8 adapter-required tests |
| Adapter-required GPU checks executed | 82 passed, 0 failed | Tensor 22 +storage 1 +autograd 17 +graph 1 +NN 5 +RL 33 +CLI 3 |
| Continuous objective example | Passed | Policy 0 to -0.2; value MSE 0.979212 to 0.000122 |
| CPU Gaussian policy unit suite | 10 passed, 0 failed | Includes new endpoint regression |
| Python bindings rebuilt and pytest | 32 passed, 0 failed | Rerun because shared CPU Gaussian inverse changed |
| Workspace Clippy/all targets/all features, warnings denied | Passed | New operators, Gaussian module, objective, tests and example |
| Rust 1.75 workspace/all targets/all features | Passed | No dependencies or lockfile changes |
| Default-feature workspace/all-targets check | Passed | GPU example remains gated; CPU APIs retain their signatures |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The eight new adapter-required tests cover tensor log/tanh/clipping and column
shape/ownership behavior (2), CPU/finite-difference gradients and immutable
snapshots/no-grad/empty handling (2), and actual CPU Gaussian density/gradient,
supplied-noise f64 reference, PPO loss/gradient/four Adam updates, and invalid
input/hidden overflow checks (4). The existing transfer unit covers all new
operations without intermediate transfers; the lifetime unit covers the saved
continuous operator graph. All GPU tensor/autograd/NN/agent/CLI suites were rerun
because shader and reverse-mode paths changed. Two ordinary tests in the explicit
CLI suite are excluded from the 82 adapter-required count. GPU execution used
Mesa llvmpipe GL. GPU CI runs the continuous objective suite and example;
physical hardware performance is not claimed.

## Stage 7b: continuous GPU PPO agent training

`GpuPpoContinuous` owns a seeded Gaussian actor and value network, frozen action
transform, separate device Adam optimizers and separate successful-update clocks.
Both networks use two ReLU hidden layers. Actor trunk layers use seeds `s` and
`s+1`, mean/std heads use `s+2` and `s+3`, and critic layers use `s+4..s+6`, with
wrapping addition. Additive CPU `new_seeded`, `sample_with_rng`,
`select_action_with_rng`, `train_on_batch_with_rng` and critic access support actual
CPU/GPU parity without changing existing convenience APIs.

Sampling uses the shared host Box–Muller implementation (two RNG draws per action
dimension), uploads noise and applies the resident Gaussian transform. Actions,
log density, value and finite-check scalars are explicit readbacks for environment
interaction. Training uploads each shuffled host minibatch; actor, critic,
backward graph, gradients and Adam state stay on device. The final partial
minibatch is included and unused batch capacity is ignored. Loss/value/base-entropy
metrics average minibatches, matching CPU continuous PPO. Base entropy describes
the unsquashed Gaussian and is diagnostic; the continuous CPU objective does not
apply the discrete entropy/value coefficients.

`collect_rollout_with_rng` accepts an `Environment` plus a fallible action-vector
conversion callback. Observation dimensions and exact continuous action bounds
must match configuration before reset. Each episode uses its own CPU continuous
rollout buffer and GAE computation before concatenation. True terminals bootstrap
zero; truncation and imposed step limits bootstrap the final observation. Reset
seeds increment with wrapping addition. Sampling and shuffle streams are supplied
separately by the caller.

Active tensor shapes, finite inputs, action bounds, advantage normalization and
update-clock overflow are validated before training. Each minibatch checks finite
objective metrics and both networks' gradients before either Adam step. Previously
completed steps remain committed if a later minibatch fails. The two optimizers
commit separately and their clocks record each successful step; this is not a
transaction over the whole batch. Failed collection may consume RNG/environment
state. Continuous runtime, checkpoints and persistence of these RNGs are not part
of this stage.

```bash
cargo test --locked -p rustforge-rl --test continuous_seeded
cargo test --locked -p rustforge-rl --features gpu --test gpu_continuous_agent -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_continuous_ppo_training
```

The target-action environment rewards `-(action - 0.5)^2` within `[-1,1]`.
Model seed 42, sampling seed 7 and shuffle seed 8 train 30 iterations of 64 fresh
one-step episodes, with three PPO epochs per iteration. Evaluation uses 128
samples with seed 99 before and after training. Sampled MSE fell from **0.235857**
to **0.000182**, and the deterministic action reached **0.505693**. Actor and critic
each completed 90 updates. This verifies bounded continuous environment learning
on Mesa llvmpipe GL; it does not measure physical GPU performance.

### Deviations from Plan (stage 7b)

- CPU continuous PPO lacked seeded initialization and supplied-RNG APIs. Added
  backward-compatible constructors and RNG variants to validate the actual shared
  Gaussian sampler, actor/value parameters and minibatch optimizer behavior.
- Continuous environment action types do not share a vector conversion trait.
  Rollout collection accepts a fallible callback rather than changing environment
  interfaces. Box-space bounds are checked against agent configuration.
- An initial parity fixture used arbitrary actions under a narrow seeded policy,
  producing extreme importance ratios after repeated updates. The finite guard
  correctly rejected it. The fixture now records on-policy actions and old
  densities, retaining five active rows, a three-row minibatch and NaN padding.
- Hardware validation remains deferred by user instruction. Continuous worker,
  CLI and checkpoint integration is stage 7c.

### Issue Resolution Progress (stage 7b)

| Work | Issue ID | Result |
| --- | --- | --- |
| CPU seeded continuous policy/critic and supplied RNG APIs | Unassigned | Complete, wrapping seeds, multidimensional bounds and exact repeatability |
| GPU Gaussian actor/value sampling and training | Unassigned | Complete, seeded CPU parameter/sample/loss and repeated Adam parity |
| CPU rollout and episode-local GAE | Unassigned | Complete, true-terminal, truncation and imposed-limit bootstrap checks |
| Partial minibatches, NaN padding and separate update clocks | Unassigned | Complete, five active rows, three-row minibatches and 18 updates per optimizer |
| Invalid conversion/input/normalization and gradient rejection | Unassigned | Complete, no optimizer update on rejected minibatch, including critic overflow |
| Fresh continuous environment learning and runnable example | Unassigned | Complete, seeded target-action convergence |
| Continuous GPU runtime/checkpoint integration | Unassigned | Next (7c) |

### Verification (stage 7b)

| Check | Passing result | Delta from stage 7a |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,059 passed, 113 ignored, 0 failed | +2 ordinary CPU tests; +4 adapter-required tests |
| Adapter-required PPO checks executed | 18 passed, 0 failed | Continuous agent 4 +continuous objective 4 +discrete agent 4 +discrete checkpoint 2 +discrete runtime 4 |
| Continuous environment training example | Passed | Sampled MSE 0.235857 to 0.000182; deterministic action 0.505693; 90 updates per optimizer |
| Seeded CPU continuous API checks | 2 passed, 0 failed | Wrapping model seeds, asymmetric two-action bounds, supplied sampling/shuffle RNGs, partial minibatches and NaN tail |
| Extended two-action CPU/GPU sampling parity | Passed | Focused rerun of existing continuous agent parity test after adding multidimensional sampling |
| Python bindings rebuilt and pytest | 32 passed, 0 failed | Shared CPU sampling/training convenience wrappers now delegate to RNG variants |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Includes new agent, two test files and environment example |
| Rust 1.75 workspace/all targets/all features | Passed | No new dependencies or lockfile changes |
| Default-feature workspace/all-targets check | Passed | GPU example remains feature-gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The four new adapter-required tests cover (1) seeded CPU/GPU actor/value
initialization, single/two-action sampling and repeated partial-minibatch Adam
parity, (2) episode-local terminal/truncation/step-limit GAE, (3) fresh environment
learning and update counts, and (4) conversion/input/normalization and nonfinite
critic-gradient rejection before optimizer updates. The two ordinary CPU tests
run in the workspace suite; the focused PPO run also ran them, so its 20 passing
tests comprise 18 adapter-required checks plus those two ordinary tests. Tensor,
autograd and NN kernels did not change in stage 7b; their full adapter suites were
verified in stage 7a. GPU CI includes the new agent suite and training example.

## Stage 7c: continuous runtime, CLI and dual-optimizer checkpoints

`PpoContinuousTrainerAdapter<E, F>` stores a Send environment and Send fallible
continuous-action callback. It constructs `PpoContinuousBackend` inside the
owning worker, so device variables never cross threads. CPU/GPU selection uses
the existing `PpoRuntimeOptions`. Seeded runs derive separate model, environment,
action-noise and minibatch-shuffle seeds, as in the discrete PPO runtime. Config
validation is shared between CPU runtime and GPU construction. Runtime validates
observation/action dimensions and exact Box bounds before environment reset,
including when saved configuration overrides requested configuration.

The runtime collects an episode-local continuous rollout and computes CPU GAE.
True terminals bootstrap zero; truncation and step limits use the final value.
Pause/resume retains the rollout. Graceful stop completes and trains the current
episode. Force stop during a partial episode discards it; force stop on the last
step permits that complete episode's update. Final checkpoint saves happen on
normal completion and controlled stops; an error does not overwrite the final
checkpoint. Interactive checkpoint control remains unsupported (capability false).

`train ppo --env pendulum` and `run ppo --env pendulum` select continuous PPO;
CartPole continues to select discrete PPO. Default device is CPU. Pendulum's
profile uses three observations, one torque action in `[-2,2]`, two 64-unit hidden
layers, learning rate 0.001 and 200-step episodes. GPU requires the `gpu` feature.
DQN/A2C/REINFORCE on Pendulum, PPO on GridWorld, PPO prioritized replay and CPU
checkpoint flags fail validation before metrics output is created or overwritten.
Live display identifies the selected device and shows restored configuration as
such. Existing metrics/checkpoint alias protection applies to both PPO variants.

Continuous metadata uses algorithm `ppo-continuous` and generic JSONL v1 metrics:
`reward.episode`, `reward.moving_average`, `loss.policy`, `loss.value`,
`rollout.size` and `performance.steps_per_second`. The runtime does not expose a
categorical entropy metric for a squashed continuous distribution.

### Continuous checkpoint contract

`RFGPUPC0` version 1 is distinct from discrete `RFGPUPPO` and DQN `RFGPUDQN`.
A fixed little-endian header/body is bounded to 256 MiB, including the header.
The body contains serialized continuous configuration/action bounds, eight actor
parameter tensors, six critic parameter tensors, separate Adam hyperparameters,
moments and timesteps, and separate actor/critic successful-update counters.
Each clock must match its own optimizer; the two clocks may differ if a previous
step committed only one optimizer. Every shape, tensor length, finite value,
nonnegative variance, moment-presence/progress relation and config/rate/clock
contract is checked on the host before allocating a candidate GPU agent.

The continuous format reuses the discrete codec's tensor/moment representations
and atomic sibling-file replacement helper; the discrete wire format is unchanged.
Restore replaces the live agent only after full success, keeping old external
parameter handles attached to their old model. Gradients are not persisted.
Environment, rollout, random streams, episode/step run counters and metrics start
fresh; this is model/optimizer resume rather than complete experiment persistence.

```bash
cargo run --locked -p rustforge-cli --features gpu -- train ppo --env pendulum --device gpu --episodes 1 --checkpoint target/pendulum-ppo.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- train ppo --env pendulum --device gpu --episodes 1 --resume target/pendulum-ppo.chk --checkpoint target/pendulum-ppo.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- run ppo --env pendulum --device gpu --episodes 10
cargo test --locked -p rustforge-rl --features gpu --test gpu_continuous_checkpoint --test gpu_continuous_runtime -- --include-ignored
```

### Deviations from Plan (stage 7c)

- Added a seeded CPU continuous runtime route alongside GPU routing to keep the
  CLI's default-device behavior consistent. Config/bounds validation is shared
  with GPU construction rather than duplicating separate acceptance rules.
- Reused existing private discrete tensor/Adam codec and atomic write helpers
  with visibility limited to the GPU PPO module. Kept the continuous header and
  body separate so old discrete checkpoints retain their version and layout.
- Omitted runtime entropy instead of labeling base-Gaussian entropy as the
  squashed distribution's entropy. Policy/value losses retain CPU continuous PPO
  semantics; base entropy remains available in the lower-level GPU agent metrics.
- Pendulum smoke runs validate finite training and routing, not convergence or
  physical GPU speed. Live plan construction is tested; terminal rendering was
  not exercised interactively in this cloud session. Hardware work is deferred
  by user instruction, and full RNG/environment persistence is a separate stage.

### Issue Resolution Progress (stage 7c)

| Work | Issue ID | Result |
| --- | --- | --- |
| Continuous checkpoint host codec/config validation | Unassigned | Complete, headers, bits, bounds, shapes, dual moments and independent clocks |
| GPU dual-Adam checkpoint resume and live restore | Unassigned | Complete, bit-identical resumed losses/parameters/final file; failed load/save preserves state |
| Worker-owned continuous CPU/GPU runtime | Unassigned | Complete, seed streams, episode rollout/GAE, dimension/bounds contracts and finite JSONL |
| Pause/resume and graceful/force stops | Unassigned | Complete, completed-update counters and forced partial-rollout discard |
| Pendulum headless/live CLI routing and display | Unassigned | Complete, CPU default, GPU fresh/resumed runs and plan/schema checks |
| Reproducible CPU runtime and invalid conversion/config handling | Unassigned | Complete, repeated seeded metrics and failures before step |
| Physical GPU profiling / full experiment persistence | Unassigned | Deferred |

### Verification (stage 7c)

| Check | Passing result | Delta from stage 7b |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,066 passed, 120 ignored, 0 failed | +7 ordinary tests; +7 adapter-required tests |
| Adapter-required PPO checks executed | 28 passed, 0 failed | RL 24 +CLI 4, detailed below |
| Host checkpoint codecs (DQN, discrete and continuous PPO) | 10 passed, 0 failed | +2 continuous wire/metadata tests; no adapter allocated |
| CPU continuous runtime suite | 3 passed, 0 failed | Seeded repeated metrics, graceful/forced controls, invalid config/bounds/observation/conversion |
| CLI unit/headless PPO suite | 14 passed, 0 failed | +1 continuous live plan test, +1 CPU Pendulum headless JSONL test |
| Default-build CLI device validation | 2 passed, 0 failed | Feature-disabled GPU and unsupported/checkpoint flags fail before output mutation |
| Fresh default-profile GPU Pendulum CLI | Passed | 1 episode, 200 steps, final continuous checkpoint written |
| Python bindings rebuilt and pytest | 32 passed, 0 failed | Native module additions remain compatible |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Includes continuous codec, runtime, CLI and tests |
| Rust 1.75 workspace/all targets/all features | Passed | No dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | CPU continuous runtime and Pendulum CLI compile without GPU |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The seven new ordinary tests comprise two host continuous checkpoint tests,
three CPU continuous runtime tests, one live training-plan test and one headless
CPU Pendulum metric test. Existing CLI validation tests gained Pendulum rejection
cases without changing their test count. The seven new adapter-required tests
comprise continuous checkpoint 2, continuous runtime 4 and continuous CLI 1.
The 28 executed adapter-required checks comprise continuous agent 4 +continuous
objective 4 +continuous checkpoint 2 +continuous runtime 4 +discrete agent 4
+discrete checkpoint 2 +discrete runtime 4 +CLI 4. Two ordinary CLI validation
tests also ran in the explicit CLI suite and are excluded from the 28 count.
Existing GPU continuous learning again reached sampled MSE 0.000182 from
0.235857. Tensor/autograd/NN shader paths did not change in this stage. Execution
used Mesa llvmpipe GL; physical hardware validation and performance claims remain
deferred. GPU CI explicitly runs both continuous checkpoint host and adapter suites.

## Stage 8a: GPU A2C objective foundation

`agent::gpu_a2c::a2c_loss` builds the actual CPU A2C objective on device:

```text
actor_loss = -mean(log_softmax(logits)[actions] * detach(advantages))
value_loss = mean((values - detach(returns))^2)
entropy = -mean(sum_actions(exp(log_softmax(logits)) * log_softmax(logits)))
total_loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
```

Advantages retain their original scale; there is no normalization, clipping,
importance ratio or old-policy density. Coefficients default to CPU A2C's 0.5
and 0.01, and must be finite/nonnegative (zero is valid). Losses require nonempty
`[batch,actions]` logits, matching resident typed actions and `[batch,1]` value,
advantage and return tensors. Shape/ownership validation uses existing device
contracts; action bounds are checked at typed index upload/gather.

The API returns actor/value/entropy/total loss variables and selected action log
probabilities, without host readbacks or optimizer mutation. References are
immutable detached snapshots even when callers mark them trainable.
`checked_metrics()` explicitly reads finite-check and metric scalars, checking
all frozen inputs and all four objective components even if a coefficient is
zero. The graph supports no-grad inference. Finite forward metrics do not guarantee
finite backward gradients: callers must validate gradients before optimizer steps,
as the runnable example does.

`GpuA2cNet` re-exports the existing `GpuActorCriticNet`: Linear→ReLU shared trunk,
actor/value heads, six parameter tensors and wrapping seed offsets 0/1/2. This
matches CPU `ActorCriticNet` parameter order and initialization exactly. Constructor
and forward errors retain the shared network's existing `GpuPpoError` type; the
A2C objective exposes its own `GpuA2cLossError`. No shader, autograd or network
implementation changes are required.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_a2c_loss -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_a2c_objective
```

The fixed-batch example uses model seed 42, four one-hot observations, two actions,
eight hidden units, positive advantages and fixed value targets. Eighty combined
Adam steps at learning rate 0.03 reduced total loss from **1.312391** to
**0.001536**, with actor loss **0.001547** and value MSE **0.000208**. Loss/input
metrics and all parameter gradients are checked before every update. This
validates objective optimization on Mesa llvmpipe GL; it does not collect
rollouts or demonstrate environment learning or physical GPU performance.

### Deviations from Plan (stage 8a)

- Re-exported the existing shared GPU actor/value network as `GpuA2cNet` instead
  of duplicating its implementation. Actual seeded CPU A2C training validates
  all six parameters and their gradients across four combined Adam updates.
- CPU computes entropy using softmax by division; GPU computes probabilities
  via exponentiated log-softmax. CPU objective/gradient and f64 finite-difference
  checks verify the equivalent formula within floating-point tolerance.
- Added frozen-input finite checks and zero-coefficient overflow coverage, plus
  an example that checks gradients before Adam. An objective API alone does not
  provide an atomic agent update or guarantee finite gradients.
- GPU agent ownership, sampling, rollout/GAE, CLI/runtime and checkpoints are
  subsequent stages. Hardware testing remains deferred by user instruction.

### Issue Resolution Progress (stage 8a)

| Work | Issue ID | Result |
| --- | --- | --- |
| Device A2C actor/value/entropy/combined objective | Unassigned | Complete, actual CPU loss and unnormalized-advantage/entropy conventions |
| Detached rollout references and immutable snapshots | Unassigned | Complete, no reference gradients and post-forward mutation/no-grad checks |
| CPU/f64 output and gradient parity | Unassigned | Complete, central finite differences for logits and values |
| Seeded shared-network and combined Adam parity | Unassigned | Complete, six initial tensors and four actual CPU A2C gradient/parameter updates |
| Shape, owner, typed-action, config and nonfinite guards | Unassigned | Complete, extreme logits, one-action entropy and masked value overflow |
| Runnable fixed-batch optimization and GPU CI | Unassigned | Complete, loss reduction and explicit gradient checks |
| Fresh GPU A2C rollout learning | Unassigned | Next (8b) |

### Verification (stage 8a)

| Check | Passing result | Delta from stage 7c |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,067 passed, 124 ignored, 0 failed | +1 ordinary A2C coefficient test; +4 adapter-required A2C tests |
| Adapter-required objective checks executed | 7 passed, 0 failed | A2C 4 +existing discrete PPO objective 3 |
| Fixed GPU A2C objective example | Passed | 80 Adam steps; total 1.312391 to 0.001536; actor 0.001547; value 0.000208 |
| Workspace Clippy/all targets/all features, warnings denied | Passed | New objective module, tests and feature-gated example |
| Rust 1.75 workspace/all targets/all features | Passed | No dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | A2C GPU module and example remain gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The four new adapter-required tests cover (1) CPU composite loss/gradient and f64
finite-difference parity with detached targets, (2) actual seeded CPU A2C forward,
combined gradients and four Adam parameter updates, (3) uniform/single-action
entropy, extreme logits and raw advantage scale, and (4) invalid shapes/owners,
typed action bounds, frozen forward snapshots, no-grad and nonfinite/zero-weight
value-overflow detection. The ordinary coefficient test validates defaults, zero,
negative and nonfinite coefficients without adapter allocation. Existing PPO
objective parity was rerun because A2C reuses its categorical/device foundation.
No CPU algorithm, Python binding, tensor shader, autograd or NN module behavior
changed. Execution used Mesa llvmpipe GL, with physical hardware tests deferred.
GPU CI explicitly executes the new A2C suite and example.

## Stage 8b: GPU A2C agent sampling and rollout training

`agent::gpu_a2c::GpuA2c` owns a `GpuA2cNet`, resident Adam, validated `A2CConfig`
and a successful-update counter. Seeded model initialization retains the CPU
architecture, parameter order and wrapping layer seed offsets. `A2CConfig` now
also derives Clone/Debug/PartialEq without changing its fields or defaults.
`GpuA2cError` wraps device/autograd/shared-network/objective errors and input
validation; existing shared-network errors retain their source.

`select_action_with_rng` returns action index, log probability and value, using
one caller-controlled f32 random draw. Stable log probabilities and the scalar
value are explicit inference readbacks; probability exponentiation/normalization
and categorical sampling occur on CPU, matching existing categorical runtime
boundaries. `action_probabilities` provides explicit inference diagnostics.
Forward computation does not retain an autograd graph. CPU `A2C::sample_action`
with a matching seeded stream selects the same tested sequence, and the next RNG
draw matches after 30 actions.

`collect_rollout_with_rng` accepts `GpuA2cRolloutOptions` with episode count,
step limit and optional reset seed. Observation dimensions and the discrete
space must match configuration before reset. Capacity products must not overflow;
episode count and step limit must be positive. Environment actions use fallible
`TryFrom<usize>` conversion before step. Reset seeds increment with wrapping
addition. Each episode computes GAE in its own CPU `RolloutBuffer`, then complete
active batches are concatenated. True terminal—including simultaneous terminal
and truncation—bootstraps zero; truncation and imposed step limits bootstrap the
final observation. Rewards, observations (including final terminal observations),
returns and advantages must be finite. Failed collection can consume environment
or RNG state; it does not train the agent.

`train_on_rollout` performs one combined A2C Adam step over the active rows.
States, advantages/returns and typed actions are uploaded for the batch;
network parameters, graph, gradients and optimizer moments stay resident.
References are detached, advantages are unnormalized, and old log probabilities
are unused (even their shape/content is irrelevant). Unused capacity can contain
NaN or invalid actions. Empty batches return zero metrics without an update.
Shape, active action bounds, finite inputs, gradient recording and update-clock
capacity are validated. Frozen objective/input metrics, all gradients and their
squares are checked before Adam; the clock increments only after a successful
step. Gradient checks explicitly read finite-count scalars rather than gradient
tensors. Earlier successful calls remain committed if a later call fails.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_a2c_agent -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_a2c_training
```

The example rewards action 0 with +1 and action 1 with -1 in a one-step two-action
environment. Model seed 42, action seed 7, width 8 and learning rate 0.03 train
60 fresh 32-episode rollouts. The rewarding action's probability rises from
**0.046995** to **0.998541**, with exactly 60 combined Adam updates. This validates
fresh environment learning on Mesa llvmpipe GL, without a physical GPU speed claim.

### Deviations from Plan (stage 8b)

- Added Clone/Debug/PartialEq to CPU A2C configuration to support GPU ownership
  and parity inspection; CPU training and sampling implementations are unchanged.
- GPU training accepts unused rollout capacity by slicing active rows. CPU A2C
  currently expects its tensors to contain the exact active batch, so the parity
  test supplies the corresponding five-row CPU batch while GPU gets NaN padding.
- Added squared-gradient finite checks: Adam squares gradients before weighting
  its variance, so finite 2e20 gradients can poison its moments. Rejection and
  successful recovery versus a fresh optimizer validate this path, along with
  finite-forward/infinite-backward rejection. Shared Adam/PPO code was unchanged.
- Kept one update per rollout, without PPO epochs or shuffling. Sampling uses a
  supplied RNG, and collection uses the existing discrete action conversion
  contract. Runtime/CLI/checkpoint integration follows in stage 8c.
- Physical hardware testing remains deferred by user instruction.

### Issue Resolution Progress (stage 8b)

| Work | Issue ID | Result |
| --- | --- | --- |
| Seeded owned GPU A2C network/Adam/configuration/clock | Unassigned | Complete, actual CPU parameters, gradients and four combined update parity |
| Reproducible categorical sampling and probability diagnostics | Unassigned | Complete, CPU action sequence/log/value/probability and RNG-consumption checks |
| CPU episode-local multi-step rollout/GAE | Unassigned | Complete, terminal precedence, truncation/limits, reset isolation and seed wrapping |
| Active-row training with unused references/capacity | Unassigned | Complete, five active rows, NaN padding, invalid tail actions and empty old-density tensor |
| Input/action conversion and finite gradient/variance guards | Unassigned | Complete, no parameter/clock update on rejection; recovery matches fresh optimizer |
| Fresh environment learning, runnable example and GPU CI | Unassigned | Complete, rewarding action probability above 0.998 after 60 updates |
| GPU A2C worker/CLI/checkpoint integration | Unassigned | Next (8c) |

### Verification (stage 8b)

| Check | Passing result | Delta from stage 8a |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,068 passed, 128 ignored, 0 failed | +1 ordinary agent-config test; +4 adapter-required agent tests |
| Adapter-required A2C checks executed | 8 passed, 0 failed | Agent 4 +objective 4 |
| Guard/recovery test focused rerun | 1 passed, 0 failed | Extended existing rejection test with optimizer recovery after failed backward/square checks |
| GPU A2C environment example | Passed | Probability 0.046995 to 0.998541; 60 combined updates |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Agent, test suite and gated example |
| Rust 1.75 workspace/all targets/all features | Passed | No dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | CPU configuration derives remain compatible; GPU example gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The four new adapter-required tests cover (1) seeded CPU sampling and four actual
CPU A2C active-batch Adam gradient/parameter updates, (2) analytic multi-step GAE
for terminal/truncation/limits and reset isolation, (3) fresh rollout learning
and one-update-per-batch cadence, and (4) malformed/nonfinite input, action-space
and conversion rejection, no-grad/empty/overflow cases, nonfinite gradients,
finite-gradient square overflow, and successful optimizer recovery. One ordinary
agent-config test checks dimensions, rates, discount/lambda and coefficients
without an adapter. GPU objective suites were rerun; shader/autograd/NN code did
not change. GPU CI explicitly runs the new agent suite and environment example.

## Stage 8c: GPU A2C runtime and checkpoints

`A2cTrainerAdapter::with_options` selects CPU or GPU inside the training worker.
The CPU backend remains the default; its sampling and update math are unchanged.
GPU A2C is available through `train a2c --device gpu` and
`run a2c --device gpu` on CartPole. Configuration and environment dimensions/action
counts are checked before reset, including when the checkpoint configuration
replaces the requested configuration. Nonfinite observations, rewards and values
return errors; failed bootstrap estimates propagate through the shared boundary
helper. True termination still takes precedence and bootstraps zero.

The distinct version-1 `RFGPUA2C` format contains configuration, six shared-network
parameter tensors, Adam hyperparameters/moments and the successful-update counter.
A 256 MiB bound, exact shapes, finite values, nonnegative second moments, matching
learning rates and consistent Adam/update clocks are validated on the host before
candidate model allocation. Unsupported versions, truncated/trailing data and
other algorithms' checkpoint headers are rejected. Saving validates state before
writing a temporary sibling file and atomically replacing the destination. Failed
restore preserves the existing agent; failed save preserves the previous file.

Resume restores model/optimizer state rather than the entire experiment. Gradients,
environment state, rollouts, RNG state and run counters are not persisted. Each run
starts fresh model/environment/action RNG streams; resumed parameters overwrite
initial model parameters. CPU checkpoint flags are rejected. Interactive checkpoint
requests remain unsupported by the adapter capability contract.

Pause retains the in-progress rollout. Graceful stop completes and trains the
current episode; forced stop discards a partial episode, but trains an episode
already completed at the boundary. Normal completion and controlled stops save
when requested. Runtime errors leave the destination checkpoint untouched.
The existing eight-metric A2C schema reports episode/moving-average reward,
total/actor/critic loss, entropy, rollout size and throughput. Both CLI routes
forward runtime options; live display identifies saved configuration and new
rollout/random streams on resume.

```bash
cargo run --locked -p rustforge-cli --features gpu -- train a2c --device gpu --episodes 1 --checkpoint target/a2c.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- train a2c --device gpu --episodes 1 --resume target/a2c.chk --checkpoint target/a2c.chk --no-log
cargo test --locked -p rustforge-rl --features gpu --test gpu_a2c_checkpoint --test gpu_a2c_runtime -- --include-ignored
```

### Deviations from Plan (stage 8c)

- The A2C codec follows the existing bounded/atomic checkpoint pattern with its own
  format and errors, preserving DQN and PPO wire contracts.
- Shared host configuration validation also gives the CPU adapter early errors for
  invalid configuration or environment dimensions; CPU training math is unchanged.
- The shared bootstrap helper became fallible to propagate device failures while
  preserving terminal precedence. Existing boundary and PPO runtime tests pass.
- Interactive checkpoint requests remain unsupported; final save and resume are
  implemented through runtime options.
- Physical GPU testing remains deferred by user instruction. Mesa llvmpipe GL
  exercises compute paths. Live CLI option binding was tested without actual TTY
  rendering; CartPole smoke runs establish routing, not convergence.

### Issue Resolution Progress (stage 8c)

| Work | Issue ID | Result |
| --- | --- | --- |
| Bounded shared-network/Adam checkpoint format | Unassigned | Complete, host corruption validation and atomic failure preservation |
| Bit-identical checkpoint continuation | Unassigned | Complete, three saved updates followed by three matched updates and identical final bytes |
| Worker-owned CPU/GPU A2C selection | Unassigned | Complete, restored configuration, early environment checks and CPU regression coverage |
| Pause, graceful stop and forced partial-rollout stop | Unassigned | Complete, controlled-stop saves and partial/boundary update counts |
| CartPole headless/live CLI and JSONL metrics | Unassigned | Complete, fresh/resumed training, live option binding and eight finite metrics |
| Physical GPU validation | Unassigned | Deferred by user instruction |

### Verification (stage 8c)

| Check | Passing result | Delta from stage 8b |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,075 passed, 135 ignored, 0 failed | +7 ordinary tests; +7 adapter-required tests |
| Adapter-required checks executed | 27 passed, 0 failed | A2C objective/agent/checkpoint/runtime 14; discrete/continuous PPO runtime 8; CLI GPU routes 5 |
| Host A2C checkpoint codec suite | 3 passed, 0 failed | Header/version/truncation/trailing checks, metadata validation and atomic cleanup/bounds |
| CLI device suite with GPU | 7 passed, 0 failed | Five adapter-required routes plus two ordinary validation tests |
| Default-feature CLI device validation | 2 passed, 0 failed | Includes A2C feature rejection and CPU checkpoint flag checks |
| Fresh GPU A2C CartPole CLI smoke | Passed | One episode, 11 steps, checkpoint written; no convergence claim |
| Python extension rebuild and pytest | 32 passed, 0 failed | Existing Python behavior retained |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python; no dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | Excluding Python |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

Seven new ordinary tests cover the codec (three), options validation, fallible
bootstrap boundaries, CPU pre-reset validation and live CLI plan binding. Seven
new adapter-required tests cover checkpoint continuation/restoration (two),
runtime resume/controls/errors/pause (four) and the A2C CLI checkpoint route (one).
GPU CI runs the codec, checkpoint and runtime suites and includes the CLI route in
its existing device-selection suite. Counts above distinguish ignored native
checks from adapter-required checks explicitly executed on llvmpipe.

## Stage 9a: GPU REINFORCE objective foundation

`agent::gpu_reinforce::GpuReinforceNet` uses Linear/ReLU/Linear with four
parameters in CPU REINFORCE order. Initialization uses the supplied seed and
its wrapping successor, matching `REINFORCE::new_seeded` exactly. Dimensions and
parameter-size multiplication are validated before layer construction.

`reinforce_loss` computes `-mean(log pi(action) * advantages)` with stable
categorical log probabilities and typed action indices. Advantages are detached
snapshots. With Monte Carlo rollout collection (zero values, lambda 1, zero final
bootstrap), advantages equal discounted returns. The objective accepts supplied
advantages like CPU REINFORCE; it does not compute returns itself or consume old
log probabilities. There is no critic, entropy term, importance ratio or variance
normalization. The optional baseline subtracts the batch mean from advantages.

Mean reduction and scalar broadcast use existing device operators, so graph
construction and baseline subtraction perform no host readback. `checked_loss`
explicitly checks logits, supplied advantages, effective advantages and scalar
loss for finite values before backward. A finite input can overflow the mean
reduction, which is rejected at this boundary. Gradient and squared-gradient
checks remain the optimizer caller's responsibility; the example performs both
before each Adam step. This stage does not own an optimizer or training clock.

Singleton or constant-advantage batches produce zero centered advantages when the
baseline is enabled. Baseline-disabled batches preserve raw advantages. A single
available action has zero policy loss and gradient. Stable log probabilities
also handle finite extreme logits, including selecting the low-probability action.
Shapes, action length/column metadata and cross-context tensors/indices are rejected
through typed errors. Input advantages do not receive gradients in either mode.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_reinforce_loss -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_reinforce_objective
```

The seeded fixed example optimizes four identity observations with actions
`[0, 0, 1, 1]` and baseline-disabled unit advantages. Loss decreases from
**0.774196** to **0.000944** over 80 Adam updates. This verifies fixed-objective
optimization on Mesa llvmpipe GL; it does not establish environment learning or
physical GPU performance.

### Deviations from Plan (stage 9a)

- Reused existing scalar broadcast for detached mean centering instead of adding
  an autograd operator or modifying CPU REINFORCE. No lower-layer or CPU algorithm
  changes were needed.
- Exposed effective detached advantages for diagnostics, and explicitly checked
  them for reduction overflow before backward.
- The fixed example uses baseline-disabled unit advantages: subtracting their
  mean would correctly make this batch's policy gradient zero. Both baseline
  modes are covered by loss/gradient and actual CPU Adam parity tests.
- Rollout collection, owned optimizer/clock, runtime/CLI and checkpoints remain
  stages 9b/9c. Physical GPU testing remains deferred by user instruction.

### Issue Resolution Progress (stage 9a)

| Work | Issue ID | Result |
| --- | --- | --- |
| Seeded policy network and dimension guards | Unassigned | Complete, four CPU-matching parameters including wrapping seed parity |
| Stable categorical Monte Carlo objective | Unassigned | Complete, detached raw advantages and optional resident mean baseline |
| Independent numerical and actual CPU update parity | Unassigned | Complete, f64 finite differences and four CPU Adam updates in each baseline mode |
| Edge cases, invalid metadata and finite validation | Unassigned | Complete, singleton/constant batches, extreme logits, foreign contexts and baseline overflow |
| Fixed optimization example and GPU CI | Unassigned | Complete, loss below 0.001 after 80 updates |
| Owned Monte Carlo rollout agent | Unassigned | Next (9b) |
| Runtime/CLI and resumable policy/Adam state | Unassigned | Planned (9c) |

### Verification (stage 9a)

| Check | Passing result | Delta from stage 8c |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,076 passed, 139 ignored, 0 failed | +1 ordinary dimension test; +4 adapter-required objective tests |
| Adapter-required REINFORCE objective suite | 4 passed, 0 failed | Both baseline modes, f64 gradients, CPU Adam parity, edge cases and rejection |
| Fixed GPU REINFORCE objective example | Passed | Loss 0.774196 to 0.000944; 80 updates |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python; no dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | Excluding Python; new module/example feature gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The four adapter-required tests cover (1) both baseline losses, detached input
references and independent f64 finite differences, (2) seeded network logits,
parameter gradients and four actual CPU REINFORCE Adam updates per baseline mode,
(3) uniform/single-action/extreme logits and singleton/constant-advantage zero
gradients, and (4) malformed/empty batches, invalid action metadata, foreign tensor
and index ownership, nonfinite inputs and finite-input baseline overflow. One
ordinary test validates dimensions without constructing an adapter. The native
suite leaves adapter-required tests ignored; the focused suite explicitly executes
all four on llvmpipe. GPU CI runs both the suite and fixed objective example.

## Stage 9b: GPU REINFORCE agent training

`GpuReinforce` owns the seeded policy, resident Adam, configuration and successful
update counter. Construction validates dimensions, positive finite learning rate
and discount in [0,1]. Inference records no gradient graph; explicit readback
produces stable categorical log probabilities and normalized probabilities.
`select_action_with_rng` consumes one draw from a caller-owned RNG and returns the
action and log probability. Its seeded action sequence matches CPU REINFORCE.

`collect_rollout_with_rng` validates environment dimensions/actions and rollout
bounds before reset. Each episode gets a wrapping seed offset and its own CPU
rollout buffer. Values are zero, lambda is 1 and final bootstrap is zero, including
truncation and step limits. Returns therefore equal discounted Monte Carlo
advantages and never leak across resets. State/action conversion, reward, next
observation and resulting returns are checked. Sampling logs are retained for
diagnostics but are not consumed during training.

`train_on_rollout` applies one Adam step to the active states/actions/advantages.
Optional mean subtraction uses only active rows; there is no variance normalization.
Unused capacity, returns and old log probabilities are ignored. Empty batches
return zero without updating the clock. Shape, action, finite input and counter
checks precede objective construction. Finite loss is checked before backward;
gradients and their squares are checked before Adam mutates moments/parameters.
Failed backward/variance checks leave parameters and the update clock unchanged;
subsequent valid training clears gradients and matches a fresh optimizer.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_reinforce_agent -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_reinforce_training
```

The example rewards action 0 with +1 and action 1 with -1. Model seed 42, action
seed 7, hidden width 8 and learning rate 0.03 train 60 fresh batches of 32 one-step
episodes with the optional mean baseline enabled. Rewarding action probability
rises from **0.046995** to **0.999581**, with exactly 60 successful Adam updates.
This establishes fresh environment learning on Mesa llvmpipe GL, without a
physical GPU speed or CartPole convergence claim.

### Deviations from Plan (stage 9b)

- Added Clone/Debug/PartialEq to CPU REINFORCE configuration for owned GPU state
  and inspection; CPU training and sampling math remain unchanged.
- GPU training slices active rows, including before mean subtraction. CPU training
  expects exact active tensors, so parity supplies a matching five-row CPU batch
  while the GPU receives NaN padding and an invalid unused action.
- Retained sampled log probabilities as rollout diagnostics, while the objective
  recomputes current-policy logs. Returns and old logs may be absent during updates.
- Finite gradients are insufficient for Adam: squared-gradient checks reject
  second-moment overflow. Recovery tests cover both failed backward and failed
  square checks before a successful update versus a fresh optimizer.
- Runtime/CLI/checkpoints remain stage 9c. Physical GPU testing stays deferred by
  user instruction.

### Issue Resolution Progress (stage 9b)

| Work | Issue ID | Result |
| --- | --- | --- |
| Owned policy/Adam/configuration/update clock | Unassigned | Complete, four actual CPU gradient/parameter update checks per baseline mode |
| Seeded categorical sampling and diagnostics | Unassigned | Complete, CPU action/log/probability sequence and RNG consumption parity |
| CPU episode-local Monte Carlo collection | Unassigned | Complete, analytic terminal/truncation/limit returns, reset isolation and seed wrapping |
| Active-row mean baseline and unused references | Unassigned | Complete, NaN padding and absent returns/old logs do not affect training |
| Input/gradient/square validation and recovery | Unassigned | Complete, parameters/clock preserved on rejection and recovery matches fresh Adam |
| Environment learning, runnable example and GPU CI | Unassigned | Complete, rewarding action probability above 0.999 after 60 updates |
| Worker/CLI/checkpoint integration | Unassigned | Next (9c) |

### Verification (stage 9b)

| Check | Passing result | Delta from stage 9a |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,077 passed, 143 ignored, 0 failed | +1 ordinary config test; +4 adapter-required agent tests |
| Adapter-required REINFORCE suites executed | 8 passed, 0 failed | Agent 4 + objective 4 |
| GPU REINFORCE environment example | Passed | Probability 0.046995 to 0.999581; 60 updates |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python; no dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | Excluding Python; CPU config derives compatible and GPU example gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The four new adapter-required tests cover (1) seeded CPU sampling and four actual
CPU active-batch updates per baseline mode, (2) analytic episode-local discounted
returns with terminal/truncation/limits and reset seed wrapping, (3) fresh rollout
learning and one-update-per-batch cadence, and (4) invalid actions/bounds/capacity,
nonfinite states/rewards/returns, conversion failures, no-grad/empty/overflow cases,
failed gradients/squares and successful optimizer recovery. One ordinary config
test runs without adapter allocation. The existing four objective tests were
rerun explicitly; native workspace execution leaves all eight adapter-required
REINFORCE checks ignored. GPU CI runs the agent suite and environment example.

## Stage 9c: GPU REINFORCE runtime and checkpoints

`ReinforceTrainerAdapter::with_options` chooses CPU or GPU inside the worker;
GPU values do not cross thread boundaries. CPU remains the default, with unchanged
training and sampling math. GPU CartPole training is available through
`train reinforce --device gpu` and `run reinforce --device gpu`. Requested spaces
are checked before fresh agent construction. On resume, saved configuration
replaces requested values and is checked against the environment before reset.
The saved discount and baseline flag govern training. Existing derived model,
environment and action seeds remain separated.

The distinct version-1 `RFGPUREI` checkpoint stores configuration, four policy
parameter tensors, Adam hyperparameters/moments and the successful-update counter.
It follows the existing 256 MiB bounded little-endian codec and atomic sibling-file
replacement pattern. Exact tensor shapes/counts, finite values, nonnegative second
moments, matching learning rates, consistent clocks and moment presence are
validated on the host before candidate model allocation. Wrong algorithm/version,
truncated/trailing data and malformed metadata are rejected. A failed restore
leaves the live agent unchanged; a failed save preserves the destination file.

Resume persists training state rather than the whole experiment: gradients,
environment, rollout, random streams, run counters and metrics are not restored.
New environment/action streams start each run. CPU checkpoint flags are rejected,
and interactive checkpoint requests remain unsupported by the capability contract.
Normal completion and controlled stops save when requested; runtime failures leave
the previous checkpoint untouched.

The worker retains pause/resume, graceful stop and forced stop behavior. Pause
keeps the current rollout; graceful stop finishes and trains the episode; forced
stop discards a partial rollout but trains an already-completed boundary episode.
Monte Carlo returns use zero final bootstrap for all episode boundaries. Runtime
checks finite observations, logits, rewards, accumulated episode reward and returns.
The unchanged five-metric JSONL schema reports episode/moving-average reward,
policy loss, rollout size and throughput. Live plans forward options and identify
restored configuration plus fresh rollout/random streams.

```bash
cargo run --locked -p rustforge-cli --features gpu -- train reinforce --device gpu --episodes 1 --checkpoint target/reinforce.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- train reinforce --device gpu --episodes 1 --resume target/reinforce.chk --checkpoint target/reinforce.chk --no-log
cargo test --locked -p rustforge-rl --features gpu --test gpu_reinforce_checkpoint --test gpu_reinforce_runtime -- --include-ignored
```

### Deviations from Plan (stage 9c)

- Added serde configuration derives and shared host validation for CPU/GPU runtime
  construction. CPU training math is unchanged; invalid configurations now fail early.
- Preserved the detailed observation-dimension error and fresh-run pre-construction
  validation while checking saved dimensions after checkpoint load.
- The REINFORCE codec follows the established bounded/atomic pattern with its own
  magic/error/API, preserving all DQN/PPO/A2C checkpoint formats.
- Interactive checkpoint requests remain unsupported; runtime options implement
  final save/resume. Environment, rollout and RNG persistence remain out of scope.
- Physical GPU tests remain deferred by user instruction. Compute tests use Mesa
  llvmpipe GL; live option binding is tested without actual TTY rendering. The fresh
  CartPole CLI smoke demonstrates routing/finite training, not convergence.

### Issue Resolution Progress (stage 9c)

| Work | Issue ID | Result |
| --- | --- | --- |
| Distinct bounded policy/Adam checkpoint codec | Unassigned | Complete, host metadata validation and atomic failure cleanup |
| Bit-identical checkpoint continuation | Unassigned | Complete, three saved plus three resumed updates in each baseline mode, matching parameters/loss/final bytes |
| Worker-owned CPU/GPU backend and saved configuration | Unassigned | Complete, restored discount/baseline and early dimension/action validation |
| Pause, graceful stop and forced partial-rollout handling | Unassigned | Complete, preserved rollouts and completed-update checkpoint counts |
| Headless/live CLI and five finite JSONL metrics | Unassigned | Complete, fresh/resumed runs, live plan binding, feature/flag/path validation |
| Runtime error and failed checkpoint preservation | Unassigned | Complete, missing resume, restored/fresh dimension errors, nonfinite observation/reward and reward overflow |
| Physical GPU validation | Unassigned | Deferred by user instruction |

### Verification (stage 9c)

| Check | Passing result | Delta from stage 9b |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,083 passed, 150 ignored, 0 failed | +6 ordinary tests; +7 adapter-required tests |
| Adapter-required suites executed | 32 passed, 0 failed | REINFORCE objective/agent/checkpoint/runtime 14; A2C/discrete PPO/continuous PPO runtime 12; CLI GPU routes 6 |
| Host REINFORCE checkpoint codec | 3 passed, 0 failed | Header/version/truncation/trailing, malformed metadata and bounded/atomic file operations |
| GPU CLI device suite | 8 passed, 0 failed | Six adapter-required routes plus two ordinary validation tests |
| Default-feature CLI device validation | 2 passed, 0 failed | Includes REINFORCE feature rejection and CPU checkpoint flags |
| Final runtime suite after dimension-error refinement | 4 passed, 0 failed | Restored/fresh spaces, finite errors, controls and metrics |
| Fresh GPU REINFORCE CartPole CLI smoke | Passed | One episode, 11 steps, checkpoint written; no convergence claim |
| Python extension rebuild and pytest | 32 passed, 0 failed | Existing Python behavior retained |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python; no dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | Excluding Python |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

Six new ordinary tests cover codec validation (three), runtime options, CPU
pre-reset validation and live plan binding. Seven new adapter-required tests cover
checkpoint continuation/restoration (two), runtime resume/controls/errors/pause
(four) and the CLI checkpoint route (one). Existing native CPU controls/seed tests
pass; adapter-required REINFORCE foundation/agent and other on-policy runtime suites
were explicitly rerun. Counts distinguish ignored native tests from llvmpipe checks
executed separately. GPU CI runs the codec and checkpoint/runtime suites, and its
existing CLI device suite includes the REINFORCE route.

## Stage 10a: GPU TD3 objective foundations

`agent::gpu_td3::td3_critic_loss` matches CPU TD3's detached target:
`y = reward + gamma * (1 - done) * min(target_q1, target_q2)`.
The loss sums the two current-critic MSEs. Target estimates, rewards and masks
receive no gradients; only current Q predictions remain differentiable. Matching
nonempty [batch,1] shapes and ownership are checked before graph construction.
Discount is finite in [0,1]; done masks are checked in [0,1], retaining CPU arithmetic
for fractional masks. Terminal rows reduce to immediate reward. Checked metrics
validate raw inputs, targets and both critic losses before backward.

`td3_actor_loss` computes `-mean(Q1)` and leaves the input gradient graph live.
Callers supply Q1 evaluated at scaled deterministic actor actions and control
critic freezing/update cadence. Tests propagate gradients through an actual seeded
Linear actor, tanh, action scaling and a fixed differentiable critic, then compare
four CPU/GPU Adam updates. No owned networks, replay or scheduling are added yet.

`GpuTd3ActionTransform` validates nonempty matching finite ordered action bounds
and finite positive affine scale/bias before uploading constants. The actor path
scales tanh-bounded normalized actions without adding another clamp, matching CPU
TD3. The target path uses supplied standard-normal noise:
`clip(raw + clip(std * noise, -noise_clip, noise_clip), -1, 1)`, followed by affine
action scaling. Noise deviation/clip may be zero and must be finite/nonnegative.
Target outputs are detached; no RNG is consumed. Actor scaling retains gradients.

All objective/action graph construction and bound expansion stay on device with
no host readback. Explicit `checked_metrics`, `checked_loss` and action `checked`
validate before backward. Snapshots include raw noise, scaled noise and the
pre-clamp action sum so clipping cannot hide invalid inputs or intermediate
overflow. Invalid masked target estimates are also rejected. Finite forward values
do not guarantee safe backward/Adam arithmetic; callers check gradients and their
squares before optimizer steps, as the example does.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_td3_loss -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_td3_objective
```

The fixed example trains two seeded affine critics on two identity feature vectors.
Rewards `[1,-1]`, masks `[0,1]`, target estimates `[0.5,2]`/`[1,3]` and discount 0.9
produce fixed targets `[1.45,-1]`. Combined Adam at rate 0.03 reduces twin-critic
loss from **6.055205** to **3.031658480e-9** in 200 updates on Mesa llvmpipe GL.
This validates objective optimization, not replay/environment learning or physical
GPU performance.

### Deviations from Plan (stage 10a)

- Reused existing GPU minimum, clamp, tanh, reduction and bias expansion operators;
  no tensor/autograd/NN or CPU TD3 implementation changes were needed.
- Parity compares the actual CPU TD3 formulas and CPU Adam on seeded affine critics
  and actor paths. Complete CPU TD3 agent/noise/cadence parity belongs to stage 10b,
  since its existing constructor/noise generation are unseeded.
- Supplied noise is explicitly standard-normal and is multiplied by the configured
  deviation before clipping in normalized action space, matching CPU TD3's order.
- Added raw/intermediate finite snapshots and bounded done-mask checks. Fractional
  masks in [0,1] retain CPU arithmetic; invalid/nonfinite inputs are rejected.
- The fixed example uses 200 updates to meet its convergence threshold; an initial
  100-update run did not reach loss below 0.001. The threshold was retained.
- Owned networks, replay, delayed updates and target synchronization remain 10b;
  runtime/CLI/checkpoints remain 10c. Physical GPU tests stay deferred by user
  instruction.

### Issue Resolution Progress (stage 10a)

| Work | Issue ID | Result |
| --- | --- | --- |
| Detached twin-target Bellman values and summed critic MSE | Unassigned | Complete, terminal/fractional masks, gamma endpoints and target detachment |
| Deterministic actor objective and differentiable affine scaling | Unassigned | Complete, tanh/scaling/fixed-critic gradients and four CPU Adam updates |
| Supplied-noise target smoothing | Unassigned | Complete, normalized clipping before scaling, asymmetric bounds, zero noise/clip and detached outputs |
| CPU/f64 loss, gradient and fixed-update parity | Unassigned | Complete, independent finite differences and four twin-critic CPU Adam updates |
| Input/config/ownership/finite guards | Unassigned | Complete, invalid shapes/masks/bounds, foreign contexts and clipping-hidden overflow |
| Fixed objective example and GPU CI | Unassigned | Complete, combined critic loss below 4e-9 after 200 updates |
| Owned actor/twin critics/replay/delayed updates | Unassigned | Next (10b) |

### Verification (stage 10a)

| Check | Passing result | Delta from stage 9c |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,085 passed, 155 ignored, 0 failed | +2 ordinary config/bounds tests; +5 adapter-required objective tests |
| Adapter-required TD3 objective suite | 5 passed, 0 failed | CPU/f64 loss/gradient, critic/actor Adam parity, smoothing and rejection |
| Fixed GPU TD3 twin-critic example | Passed | Loss 6.055205 to 3.031658480e-9; 200 Adam updates |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python; no dependency/lockfile changes |
| Default-feature workspace/all-targets check | Passed | Excluding Python; new module/example gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The five adapter-required tests cover (1) twin targets/losses, detached references,
gamma endpoints, masks and independent f64 finite differences, (2) four actual CPU
Adam updates of seeded affine twin critics, (3) actor gradients through tanh,
affine scaling and a fixed critic with four CPU Adam updates, (4) supplied-noise
smoothing order, asymmetric bounds, zero-noise/clip and detached outputs, and
(5) malformed/empty shapes, masks, ownership, nonfinite raw inputs, masked invalid
targets, reduction overflow and clipping-hidden noise/action overflow. Two ordinary
tests validate loss/smoothing configuration and affine bounds without allocating
an adapter. Native execution leaves adapter checks ignored; all five were run
explicitly on llvmpipe. GPU CI runs the suite and fixed objective example.

## Stage 10b: owned GPU TD3 replay training

`agent::gpu_td3::GpuTd3` owns the seeded three-layer tanh actor, two three-layer
critics, independent frozen target leaves, separate actor/joint-critic Adam state,
and successful critic/actor update clocks. Hidden layers use ReLU. Actor seeds
are `seed`, `seed+1`, `seed+2`; critic seeds are offsets 3–5 and 6–8, with wrapping
arithmetic. CPU `TD3::new_seeded` uses the same architecture and initialization.
Targets initially share immutable online value snapshots; optimizer updates
replace buffers and cannot change those snapshots.

Device `concat_columns_device` joins observation/action feature matrices;
`slice_columns_device` splits the backward adjoint into both input graphs.
Operations validate devices, matrix shapes, row counts, ranges and index/storage
limits, including empty feature axes. WGSL dispatch parameters carry explicit
logical widths instead of inferring them from physical empty-buffer storage.
The existing tensor transfer-counter check now includes concat/slice and verifies
that the operations make no host transfers.

`GpuTd3::train_step_with_rng` validates active replay rows before consuming noise.
Only `batch.size` rows are uploaded; stale capacity values, including NaNs, are
ignored. States/actions/rewards/next states must be finite, matrix widths must
match configuration, and done masks must lie in [0,1]. Fractional masks keep CPU
arithmetic. Empty batches return `(0,None)` without RNG draws or clock changes;
nonempty training inside `no_grad` is rejected. Physical replay actions need only
be finite, matching CPU TD3's accepted inputs.

Exploration noise is added in physical action units and clipped to action bounds.
Target noise is added/clipped in normalized action units before affine scaling.
Both seeded CPU/GPU RNG methods preserve CPU's Box–Muller multiplication order
`std * sqrt(-2*ln(u1)) * cos(u2)`, with two uniforms per action. Zero exploration
deviation consumes no draws; zero target deviation still consumes two draws.
Caller-owned exploration, replay and smoothing streams are independent.
`ContinuousReplayBuffer::sample_with_rng` samples with replacement into existing
batch storage; its default wrapper retains thread RNG behavior.

The agent's explicit `train_step_with_target_noise` accepts **pre-scaled** Gaussian
noise, clips it using the configured normalized-space clip and ignores capacity
rows. In contrast, the stage 10a objective-level `smooth_target_actions` helper
accepts standard-normal samples and multiplies by its smoothing deviation. This
separation preserves CPU arithmetic order for exact caller-controlled streams.

Critic targets detach all target paths. Both critics take one joint Adam step.
On scheduled steps, the actor uses the **updated** Q1 with frozen critic weights,
while state/action concat retains actor gradients. Actor Adam advances only then,
and all three targets use `tau*online + (1-tau)*old`. Delay zero disables actor and
target updates. Tau endpoints 0/1 are supported without altering CPU arithmetic.

Updates are prepared on independent trainable parameter leaves and a device
snapshot of Adam moments/clock (`GpuAdam::fork`). Immutable moment buffers are
shared until optimizer steps replace them; no parameter/moment host snapshot is
needed. Both critic and scheduled actor computations must succeed before live
leaf values and optimizer state are committed. Public parameter handles stay live.
Failures preserve online/target parameters, both optimizer states and counters;
caller RNG draws already used are not rolled back. Temporary graphs drop after
the call. Successful commits clear live gradients.

Scalar checks validate affine logits before nonlinearities can hide invalid values,
actions/targets/losses, gradients and gradient squares before Adam, candidate
parameters after Adam and Polyak targets before commit. Multiple diagnostic counts
are combined on device before one scalar readback. Full tensors stay resident;
selection downloads only final physical actions. These checks provide correctness
coverage and do not establish physical GPU throughput.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_td3_agent -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_td3_training
```

The fresh example collects 600 terminal episodes with reward `-(action-0.5)^2`,
including 64 uniform-action warm-up episodes, followed by noisy policy collection
and seeded replay sampling. On Mesa llvmpipe GL, 536 critic/268 actor updates reduce
deterministic cost from **0.291425** to **0.000515**, with final action **0.477313**.
The acceptance checks retain cost below 0.01, at least 90% reduction and action
within 0.1 of the known optimum. This demonstrates fresh replay learning on a small
continuous environment; Pendulum performance and physical GPU throughput are not
measured here.

### Deviations from Plan (stage 10b)

- Added independent device parameter snapshots and Adam-state rebinding so a
  scheduled actor failure rolls back the prepared critic update as well. These
  are lower-layer GPU APIs; CPU optimizers and default TD3 formulas are preserved.
- Added CPU seeded construction, network/counter accessors and caller-RNG/explicit
  noise hooks for full-agent parity. Default CPU constructors retain their original
  unseeded initialization; exploration, smoothing, delay and Polyak math are kept.
- Added seeded continuous replay sampling to make fresh collection/sampling checks
  reproducible. The default sampling wrapper still uses thread RNG.
- Fresh learning uses a terminal continuous target environment with a known
  quadratic reward, rather than claiming Pendulum performance before runtime
  integration. An initial 16-unit/rate-0.003 actor/rate-0.01 critic run reached cost
  0.012079 and missed the unchanged 0.01 threshold. The final example uses 32 units,
  actor rate 0.001 and critic rate 0.005; the threshold remains unchanged.
- Runtime, Pendulum CLI, multi-optimizer checkpoints and controls remain stage
  10c. Physical GPU profiling stays deferred by user instruction.

### Issue Resolution Progress (stage 10b)

| Work | Issue ID | Result |
| --- | --- | --- |
| Resident state/action concat and gradient splitting | Unassigned | Complete, unequal/empty axes, shared/repeated backward, ownership/range checks and no-transfer verification |
| Owned seeded actor, twin critics and frozen targets | Unassigned | Complete, CPU initialization/network parameter parity and target isolation |
| Continuous replay and caller-owned noise streams | Unassigned | Complete, seeded sampling, active-row validation, zero-noise draw rules and pre-scaled supplied noise |
| Delayed actor updates and all-network Polyak synchronization | Unassigned | Complete, CPU/GPU six-update parity, delays 0/1/2/3 and tau 0/0.2/1 |
| Transactional parameters/Adam/counters and finite guards | Unassigned | Complete, actor failure and gradient-square overflow rollback followed by identical valid updates |
| Fresh continuous-environment learning | Unassigned | Complete, replay collection, seeded sampling and deterministic policy convergence |
| Pendulum runtime, CLI and multi-optimizer checkpoints | Unassigned | Next (10c) |

### Verification (stage 10b)

| Check | Passing result | Delta from stage 10a |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,089 passed, 163 ignored, 0 failed | +4 ordinary tests; +8 adapter-required tests |
| Full GPU tensor matrix/operation suite | 23 passed, 0 failed | +1 concat/slice/empty-axis/ownership/range test |
| Full GPU autograd suite | 19 passed, 0 failed | +1 concat/shared/repeated/frozen/empty backward test; +1 device Adam fork/rebind test |
| Tensor resident-operation transfer counter | 1 passed, 0 failed | Existing check expanded with concat/slice |
| TD3 owned-agent suite | 5 passed, 0 failed | Seeded CPU parity/cadence/Polyak; RNG; active-row/rejection; rollback; fresh learning |
| TD3 objective regression suite | 5 passed, 0 failed | Existing stage 10a coverage retained after diagnostic batching |
| Fresh GPU TD3 environment example | Passed | Cost 0.291425 to 0.000515; action 0.477313; 536 critic/268 actor updates |
| Fixed GPU TD3 objective example | Passed | Loss 6.055205 to 3.031658480e-9; 200 Adam updates |
| Python editable rebuild and regression suite | 32 passed, 0 failed | CPU TD3/RNG changes included in rebuilt extension |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python; array snapshot helper uses compatible API |
| Default-feature workspace/all-targets check | Passed | Excluding Python; GPU agent/example remain gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The four ordinary additions are one GPU-owned configuration validation test, two
CPU seeded TD3/RNG tests and one seeded continuous replay test. The eight new
adapter-required tests are one tensor, two autograd and five TD3-agent tests.
All eight are included in the 53 explicitly passing GPU checks listed above;
the remaining 45 are regression coverage from the existing tensor/autograd/TD3
objective suites and expanded transfer-counter test. Native execution leaves the
adapter tests ignored; explicit checks used Mesa llvmpipe GL. Physical GPU tests
remain deferred. No dependencies or lockfile changes were needed. Logs are saved
under `/tmp/rustforge-gpu-stage10b-*.log`.

## Stage 10c: TD3 worker, Pendulum CLI and checkpoints

`Td3TrainerAdapter` uses the shared `Trainer`/`LiveHooks` lifecycle, progress,
metric persistence and controls. Backend construction, GPU ownership and
checkpoint I/O happen inside `run` on the worker; device variables never cross
threads. The new Pendulum profile uses three observations, one action in [-2,2]
and 64-unit hidden layers. CPU remains the default; GPU requires the feature.
`train td3` and `run td3` accept only Pendulum, reject prioritized replay, and
check execution/checkpoint/output conflicts before truncating metrics or creating
live run artifacts. The live plan displays saved configuration on resume.

`Td3RuntimeOptions` sets replay capacity, mini-batch size, uniform-action warm-up,
minimum training transitions and physical exploration deviation. Defaults use
100,000 replay slots, batches of 64, 1,000 warm-up transitions, learning from
transition 64 and deviation 0.1 after warm-up. Replay training starts during
uniform collection; one critic update runs per accepted transition after the
learning threshold. The GPU agent owns delayed actor/target cadence. Separate
seed streams control initialization, environment reset, exploration, replay
sampling and target smoothing. Unseeded runs use entropy-backed streams.

The runtime validates effective restored dimensions/bounds against the environment,
replay allocation products, finite observations/actions/rewards and episode/window
reward sums. Invalid action conversion or environment output returns an error.
Only true `terminated` flags disable bootstrap; truncation and runtime step limits
end/reset the episode but keep replay done masks zero. Seeded checkpoint comparisons
verify identical truncation/step-limit updates and distinct true-terminal updates.

Pause blocks at the step boundary while retaining the in-flight transition.
Graceful stop completes the current episode. Force stop discards the in-flight
transition/update and saves earlier completed atomic updates; its environment step
is included in the summary, while the interrupted episode is not completed.
Completion/controlled stop saves the final checkpoint. Error returns skip save
and preserve any prior checkpoint, including errors after earlier successful
in-memory updates. Interactive checkpoint requests remain unsupported; capability
metadata reports `checkpoint=false` and shared controls report their resolution.

JSONL keeps schema `rustforge-metrics-jsonl-v1` with six finite named metrics:
`reward.episode`, `reward.moving_average`, `loss.critic`, `loss.policy`, `replay.size`
and `performance.steps_per_second`. Loss metrics report the latest successful
critic/delayed actor update (zero until the first respective update). Metric roles
bind episode reward, primary critic loss, actor signal and throughput to the live
console. Human-readable labels describe these values.

The bounded little-endian version-1 format uses magic **`RFGPUTD3`** and a 12-byte
header, with a maximum complete size of 256 MiB. It saves the actor, Q1, Q2, all
three frozen targets, actor Adam, joint critic Adam, `TD3Config`, critic clock and
actor clock. Host validation checks all six architectures, shapes/value counts,
finite parameters/targets, finite nonnegative Adam variances, optimizer settings,
learning-rate agreement, moment presence, platform clock limits and delayed
cadence (`actor_updates = critic_updates / policy_delay`, or zero for delay zero).
Truncated/foreign/version-mismatched/trailing data and oversized files fail before
replacement allocation. Config validation is shared by CPU runtime and GPU agent.

Saving validates and encodes the complete snapshot before writing a same-directory
exclusive temporary file, synchronizing it, closing it and renaming over the final
path. Failed writes/renames clean up the temporary file. Loading validates the
complete host state, constructs a candidate agent, restores all six networks and
both optimizer states, and assigns it only after success. Targets remain frozen;
live gradients are cleared. Successful restore replaces handles. Fixed supplied-
noise continued updates and saved bytes are bit-identical for delays 0/1/2/3,
including resume at the next actor boundary.

Resume restores **agent training state**. Environment state, replay contents,
exploration/replay/smoothing RNG streams, gradients, episode/run counters, metrics
and warm-up counters restart. The serialized agent config overrides constructor
config; runtime replay/exploration options remain part of the new run. This does
not promise an identical resumed experiment with fresh data/RNG streams.

```bash
cargo test --locked -p rustforge-rl --features gpu --lib gpu_td3::agent::checkpoint
cargo test --locked -p rustforge-rl --features gpu --test gpu_td3_checkpoint --test gpu_td3_runtime -- --include-ignored
cargo test --locked -p rustforge-cli --features gpu --test device_selection gpu_td3_pendulum_cli -- --include-ignored
cargo run --locked -p rustforge-cli --features gpu -- train td3 --env pendulum --device gpu --episodes 1 --checkpoint target/td3.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- train td3 --env pendulum --device gpu --episodes 1 --resume target/td3.chk --checkpoint target/td3.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- run td3 --env pendulum --device gpu --episodes 10
```

### Deviations from Plan (stage 10c)

- Added CPU TD3 runtime/CLI routing alongside GPU selection so CPU remains the
  default. CPU runtime uses the seeded/RNG hooks from 10b; CPU checkpoint flags
  remain unsupported. No CPU TD3 update formulas or lower-layer GPU math changed.
- Shared TD3 configuration validation between CPU runtime and GPU agent and added
  serde support to `TD3Config` for the checkpoint wire format.
- Runtime options explicitly separate uniform warm-up and learning start. Defaults
  train during warm-up after transition 64; these options/replay/RNG/warm-up state
  restart after resume and are not checkpointed.
- Kept the generic JSONL schema with six TD3-specific metrics rather than reusing
  the five REINFORCE metrics. Critic and delayed actor losses are distinct signals.
- Force stop keeps earlier successful per-transition updates and discards the
  in-flight update; TD3 has no on-policy rollout to discard as a unit.
- Reused the established codec/atomic-write pattern in a dedicated TD3 module,
  preserving other algorithm formats and checkpoint APIs. No dependencies or
  lockfile changes were required.
- Physical GPU profiling and complete experiment-state persistence stay deferred.

### Issue Resolution Progress (stage 10c)

| Work | Issue ID | Result |
| --- | --- | --- |
| Worker-owned CPU/GPU TD3 backend and Pendulum profile | Unassigned | Complete, effective saved config and environment/bounds validation |
| Headless/live CLI selection and preflight validation | Unassigned | Complete, parsing, runtime plan, output/alias guards and CPU/GPU Pendulum smoke checks |
| Six-network/two-Adam versioned checkpoints | Unassigned | Complete, bounded host validation, atomic save and candidate restore |
| Resume delayed actor cadence | Unassigned | Complete, bit-identical supplied-noise updates/bytes for delays 0/1/2/3 |
| Replay bootstrap and seeded runtime streams | Unassigned | Complete, terminal versus truncation/step-limit checks and independent streams |
| Pause/graceful/force-stop and file preservation | Unassigned | Complete, retained in-flight pause, prior completed updates, conversion/environment/save errors |
| Generic JSONL and live metric roles | Unassigned | Complete, six finite metrics and restored-configuration display |
| GPU SAC objective foundation | Unassigned | Next (11a) |

### Verification (stage 10c)

| Check | Passing result | Delta from stage 10b |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,097 passed, 171 ignored, 0 failed | +8 ordinary tests; +8 adapter-required tests |
| Host TD3 checkpoint codec/file validation | 3 passed, 0 failed | Wire bits/header/format; malformed six-network/Adam/cadence data; size limit/atomic cleanup |
| GPU TD3 checkpoint continuation | 2 passed, 0 failed | Delays 0/1/2/3, target freezing, bit-identical continuation/bytes and failed I/O preservation |
| GPU TD3 runtime controls/bootstrap | 5 passed, 0 failed | Saved config/JSONL, stops, pause, terminal/truncation/cutoff and failure preservation |
| GPU Pendulum CLI resume | 1 passed, 0 failed | 200-step episode, saved 3-unit hidden profile/delay 3; 137 critic/45 actor updates |
| GPU TD3 agent/objective regressions | 9 passed, 0 failed | Four agent parity/guard/rollback tests plus five objective tests; prior fresh-learning result unchanged |
| Default-feature CLI device guards | 3 passed, 0 failed | TD3 included in missing-feature, invalid environment/PER/CPU checkpoint checks |
| Python editable rebuild and regression suite | 32 passed, 0 failed | Shared CPU config/serde changes included in rebuilt extension |
| Workspace Clippy/all targets/all features, warnings denied | Passed | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python |
| Default-feature workspace/all-targets check | Passed | GPU checkpoints remain feature-gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The eight ordinary additions are three host checkpoint tests, one runtime-option
test and four CLI checks (parse, invalid combinations/output preservation, live
plan/roles and CPU Pendulum metrics). The eight new adapter-required tests are two
checkpoint, five runtime and one CLI tests. Explicit GPU verification runs these
eight plus nine TD3 regressions, for **17 passing adapter-required checks**. The
prior fresh-learning example was not rerun because update math is unchanged;
its stage 10b result remains documented above. Other algorithm/lower-layer GPU
suites were unaffected. Explicit GPU checks used Mesa llvmpipe GL; physical GPU
performance is not asserted. Logs are under `/tmp/rustforge-gpu-stage10c-*.log`.

## Stage 11a: GPU SAC objective foundation

`agent::gpu_sac` supplies independently usable SAC objectives behind the `gpu`
feature. `GpuSacActionTransform` reuses the shared Gaussian transform for
supplied-noise reparameterization, tanh squashing, physical action scaling and
the CPU policy's log-probability Jacobian correction. Mean and log-standard-
deviation gradients stay live; supplied noise is detached. Asymmetric action
bounds, saturation and log-standard-deviation clamp boundaries are covered.

The critic objective computes a detached target
`reward + gamma*(1-done)*(min(target_q1,target_q2)-alpha*next_log_prob)` and sums
the two mean squared errors. The actor objective computes
`mean(alpha*log_prob-min(q1,q2))`, retaining the action path through frozen
critics. Minimum ties follow the CPU implementation's Q2 derivative. The
learned-temperature objective computes
`-log_alpha*mean(detach(log_prob)+target_entropy)`; only log-alpha receives
its gradient. The exposed detached alpha is a snapshot before its optimizer
update and must be recomputed for the next update.

Inputs require matching nonempty `[batch,1]` shapes and context ownership.
Gamma must be finite in `[0,1]`; fixed alpha must be finite and nonnegative.
Fractional done masks in `[0,1]` preserve CPU arithmetic. Checked metrics reject
nonfinite inputs, intermediate values and losses, including invalid values
hidden by terminal masks or tanh saturation. Learned alpha additionally must
remain finite and strictly positive. These helpers build graphs; the owned
agent and transactional optimizer scheduling belong to stage 11b.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_sac_loss -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_sac_objective
```

The fixed-objective example optimizes two seeded affine critics with targets
`[1.54,-1]`. Its summed loss falls from **6.306413 to 8.55e-9** after 200 Adam
updates. This validates objective optimization; fresh-environment learning
follows with the owned agent.

### Deviations from Plan (stage 11a)

- Reused the existing Gaussian transform and policy network instead of adding
  duplicate sampling math. Shared sampling diagnostics now check physical
  actions and pre-tanh intermediates; continuous PPO regressions pass.
- Actual Adam parity uses the seeded CPU Gaussian policy and twin affine
  critics with caller-supplied noise. The full CPU SAC agent's seeded/noise
  hooks and complete update-order parity remain part of 11b.
- Fixed alpha zero is accepted as an entropy-disabled boundary; learned alpha
  rejects exponential overflow and underflow to zero.
- Default-feature Clippy exposed an unused checkpoint path in the stage 10c
  CPU TD3 rejection branch. Including that path in its diagnostic resolves the
  warning without changing checkpoint behavior.
- No dependencies or lockfile changes were needed. Physical GPU profiling,
  runtime/checkpoints and full experiment-state persistence remain deferred.

### Issue Resolution Progress (stage 11a)

| Work | Issue ID | Result |
| --- | --- | --- |
| Supplied-noise Gaussian actions and corrected log probabilities | Unassigned | Complete, asymmetric bounds, saturation and detached noise |
| Detached soft twin-critic targets and summed MSE | Unassigned | Complete, CPU/f64 gradients and four seeded Adam updates |
| Actor objective and action gradients through frozen critics | Unassigned | Complete, CPU/f64 checks and four seeded Gaussian-policy Adam updates |
| Learned-temperature objective and scalar shapes | Unassigned | Complete, detached policy, four Adam updates and alpha representability guards |
| Shape, ownership and finite-value diagnostics | Unassigned | Complete, masked invalid inputs and intermediate/reduction overflow checks |
| Fixed-objective optimization and CI | Unassigned | Complete, 200 updates and dedicated test/example steps |
| Owned GPU SAC agent and fresh learning | Unassigned | Next (11b) |

### Verification (stage 11a)

| Check | Passing result | Delta from stage 10c |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,098 passed, 178 ignored, 0 failed | +1 ordinary configuration test; +7 adapter-required tests |
| GPU SAC objective/sampling/Adam checks | 7 passed, 0 failed | Critic, actor, temperature, seeded critic/policy updates, supplied-noise gradients and guards |
| Shared Gaussian/continuous PPO regressions | 4 passed, 0 failed | Sampling diagnostic changes covered |
| Fixed SAC twin-critic example | Passed | Loss 6.306413 → 8.55e-9 in 200 Adam updates |
| Workspace Clippy/all targets, default and all features | Passed, warnings denied | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python |
| Default-feature workspace/all-targets check | Passed | GPU SAC remains feature-gated |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

Explicit adapter verification totals **11 passing checks** (seven SAC and four
continuous PPO regressions), using Mesa llvmpipe GL. Physical GPU performance is
not asserted. Python was not rebuilt for this GPU-only stage; its 32-test stage
10c result remains historical. Logs are under
`/tmp/rustforge-gpu-stage11a-*.log`.

## Stage 11b: owned GPU SAC replay training

`GpuSac` owns a Gaussian actor, twin Q-critics, two frozen critic targets and
three resident Adam optimizers (actor, joint critics and log temperature). The
actor's four affine layers match CPU GaussianPolicy's parameter/seed order;
critics use two hidden ReLU layers with seed offsets 4 and 7. Collection,
replay selection, next-action sampling and actor-update sampling can use
independent caller RNG streams. Explicit standard-normal matrices also support
reproducible updates. Deterministic evaluation uses the scaled tanh mean.

Continuous replay remains caller-owned on CPU. Training validates active rows
and uploads only those rows; unused capacity can contain stale nonfinite values.
Empty batches return zero losses/current alpha without advancing clocks or
consuming randomness. Nonempty training requires gradients enabled, matching
finite state/action/reward/noise matrices and done masks in `[0,1]`. Construction
validates dimensions, allocation arithmetic, action bounds, positive finite
learning rates/initial temperature and discount/Polyak rates in `[0,1]`.

Each batch caches alpha, updates the critics against detached soft targets,
updates the actor through frozen copies of the **updated** critics, updates
log-alpha from detached pre-actor-step sampled log probabilities, and Polyak-
updates both targets. Candidate network leaves and optimizer forks hold all
changes until loss, gradient, gradient-square, parameter, temperature and target
checks pass. Live parameters, moments, targets and successful-update clock are
preserved on failure. Successful commits retain existing public parameter
handles and clear live gradients. Caller RNG draws are not rolled back after
a device/arithmetic error; malformed host batches draw nothing.

CPU SAC gains seeded construction, caller RNG streams, explicit-noise training
and read-only critic accessors; GaussianPolicy gains a detached supplied-noise
sampling hook. Existing entry points delegate to these hooks with thread RNGs.
CPU equations and update order stay the same. Six complete CPU/GPU updates agree
within 3e-4 for Polyak rates 0, 0.2 and 1, including terminal/fractional masks,
parameters, both targets, learned alpha and continued Adam state. Initial
parameters match exactly; host/device exponential rounding needs a small alpha
tolerance. Three successful updates interleaved with late actor overflow failures
remain bit-identical to an uninterrupted GPU reference, including retained leaf
handles. A separate late temperature failure leaves every network unchanged.

```bash
cargo test --locked -p rustforge-rl --features gpu --test gpu_sac_agent -- --include-ignored
cargo run --locked -p rustforge-rl --features gpu --example gpu_sac_training
```

The fresh-learning example collects 600 terminal continuous-control episodes
with 64 uniform warm-up episodes, seeded replay and independent collection,
target and actor noise streams. Each transition is newly collected from the
environment; later actions come from the stochastic policy. Deterministic
evaluation cost falls from **0.200846 to 0.000684**, with action **0.473848**
for a target of 0.5, after **536 critic/actor/temperature updates**. This is a
small reproducible control task, not a Pendulum or hardware performance claim.

### Deviations from Plan (stage 11b)

- Reused stage 10b's caller-owned continuous replay API rather than coupling
  replay/RNG ownership to the agent. Collection, replay and two update-noise
  streams remain independent.
- Added a dedicated snapshot-capable Gaussian actor and critic wrapper because
  the existing PPO policy uses a heterogeneous sequential trunk without a
  transactional snapshot API. Sampling/loss math reuses stage 11a unchanged.
- One successful-update clock covers all three optimizers and both targets:
  SAC schedules each on every nonempty update. Optimizer moments/steps stay
  resident and follow that clock; checkpoint persistence belongs to 11c.
- Added CPU seeded/supplied-noise hooks to verify the actual complete SAC agent,
  preserving existing unseeded entry points and formulas.
- No dependencies or lockfile changes were needed. Physical GPU profiling and
  full experiment-state persistence remain deferred.

### Issue Resolution Progress (stage 11b)

| Work | Issue ID | Result |
| --- | --- | --- |
| Seeded stochastic actor/twin critics/frozen targets | Unassigned | Complete, matching CPU parameter order and wrapping seed offsets |
| Resident three-Adam update order and temperature tuning | Unassigned | Complete, cached alpha and updated-critic actor gradients |
| Continuous replay, active rows and independent noise streams | Unassigned | Complete, partial/empty batches and deterministic evaluation |
| Atomic candidate updates and finite-gradient/temperature guards | Unassigned | Complete, late failure rollback, bit-identical continued Adam and retained handles |
| Complete seeded CPU SAC parity | Unassigned | Complete, six updates for tau 0/0.2/1 |
| Fresh continuous learning and CI | Unassigned | Complete, seeded terminal control example and dedicated steps |
| Pendulum runtime/CLI and three-optimizer checkpoints | Unassigned | Next (11c) |

### Verification (stage 11b)

| Check | Passing result | Delta from stage 11a |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,100 passed, 182 ignored, 0 failed | +2 ordinary tests; +4 adapter-required tests |
| Complete GPU SAC agent checks | 4 passed, 0 failed | CPU update parity, selection/partial rows, validation/temperature rollback, late actor overflow/continued Adam |
| GPU SAC objective and shared Gaussian/PPO regressions | 11 passed, 0 failed | Seven SAC objectives and four continuous PPO regressions |
| Fresh continuous-environment learning | Passed | Cost 0.200846 → 0.000684; action 0.473848; 536 updates per optimizer |
| Python editable rebuild and regression suite | 32 passed, 0 failed | Shared CPU Gaussian/SAC hooks included in rebuilt extension |
| Workspace Clippy/all targets, default and all features | Passed, warnings denied | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The two ordinary additions are the GPU SAC host configuration test and CPU SAC
seeded/noise/partial-row regression. The four adapter-required additions are
in `gpu_sac_agent`; focused verification totals **15 passing GPU checks**.
Explicit GPU checks use Mesa llvmpipe GL; physical GPU performance is not asserted.
Logs are under `/tmp/rustforge-gpu-stage11b-*.log`.

## Stage 11c: SAC runtime and checkpoints

`SacTrainerAdapter` constructs the backend inside its worker so GPU Rc leaves
stay on their owning thread. `SacRuntimeOptions` defaults to CPU, replay capacity
100,000, batch size 64, 1,000 uniform-action warm-up transitions and learning
start at transition 64. Training can therefore start during warm-up. Collection,
replay selection, target-policy sampling and actor-update sampling have separate
seeded streams. The CLI uses a 3-observation/1-action, 64-hidden-unit Pendulum
profile with bounds `[-2,2]` and a 200-step episode limit. Restored agent
configuration overrides that profile; replay/runtime options remain part of
the new run. CPU and GPU share SAC configuration validation.

`train sac --env pendulum` and `run sac --env pendulum` support explicit device
selection. Environment/PER/device/checkpoint/output-alias guards run before
output mutation. CPU remains the default and CPU checkpoint flags are rejected.
Live plans show restored configuration as such rather than presenting requested
hyperparameters as effective saved values. The runtime validates environment
observations, action bounds, rewards and running sums; failures preserve the
previous checkpoint. Terminal transitions disable bootstrap; truncations and
runtime step limits keep it. Pause retains the in-flight transition; graceful
stop finishes the episode; force stop discards that transition/update and saves
earlier completed updates. Interactive checkpoint requests remain unsupported;
checkpoint flags save on completion or controlled stop.

JSONL v1 emits eight metrics: episode/moving-average reward, latest critic,
policy and temperature losses, current alpha, replay size and throughput. Critic
loss is the primary loss and policy loss is the policy signal; all emitted
values are finite. Alpha is initialized from the effective agent state before
the first update, including on resume.

Version-1 `RFGPUSAC` checkpoints contain the Gaussian actor (eight parameters),
four critics (six parameters each), log-alpha, actor/joint-critic/temperature
Adam states, configuration and the successful-update clock. Target entropy is
derived from the saved action dimension. The complete little-endian file is
bounded at 256 MiB including its 12-byte header. Host validation rejects bad
headers/versions/truncation/trailing bytes, incompatible shapes/counts, nonfinite
parameters/moments, negative variances, nonrepresentable temperatures, invalid
configuration and optimizer clocks/rates/moment presence inconsistent with update
progress. Loading validates host state before replacement network allocation;
restored temperature is also checked on device.

Saving validates/encodes before an exclusive same-directory temporary write,
synchronization and rename. Failed writes/renames clean up temporary files.
Loading constructs a complete candidate, restores all five networks/log-alpha/
three optimizers and assigns it only after success. Targets remain frozen and
live gradients are clear. Successful restore replaces parameter handles; failed
restore preserves them. With fixed supplied noise, four continued updates and
final saved bytes are bit-identical for Polyak rates 0, 0.2 and 1.

Resume restores **agent training state**, not the complete experiment. Replay,
environment state, RNG streams, gradients, episode/run counters and warm-up
restart. Runtime continuation from a previously trained source advances the
saved clock from 2 to 8 across six newly collected transitions; the CLI smoke
check restores a saved 3-hidden-unit profile and custom temperature learning
rate, then performs 137 updates in one 200-step Pendulum episode.

```bash
cargo test --locked -p rustforge-rl --features gpu --lib gpu_sac::agent::checkpoint
cargo test --locked -p rustforge-rl --features gpu --test gpu_sac_checkpoint --test gpu_sac_runtime -- --include-ignored
cargo test --locked -p rustforge-cli --features gpu --test device_selection gpu_sac_pendulum_cli -- --include-ignored
cargo run --locked -p rustforge-cli --features gpu -- train sac --env pendulum --device gpu --episodes 1 --checkpoint target/sac.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- train sac --env pendulum --device gpu --episodes 1 --resume target/sac.chk --checkpoint target/sac.chk --no-log
cargo run --locked -p rustforge-cli --features gpu -- run sac --env pendulum --device gpu --episodes 10
```

### Deviations from Plan (stage 11c)

- Added CPU SAC runtime/CLI routing alongside GPU selection so CPU stays the
  default; CPU checkpoints remain unsupported. Shared config validation and
  serde/PartialEq support enable consistent runtime and checkpoint contracts.
- Reused the existing dedicated checkpoint codec/atomic-write pattern in a SAC
  module, preserving other algorithm formats. No dependencies or lockfile
  changes were needed.
- Recorded eight SAC metrics instead of the six TD3 metrics because temperature
  loss and alpha are separate training signals. The generic JSONL schema and
  console metric roles remain unchanged.
- As with TD3, force stop retains earlier per-transition updates and drops the
  in-flight update; SAC has no on-policy rollout to discard as a unit.
- Complete experiment-state persistence and physical GPU profiling remain
  separate work. Fresh-learning math is unchanged, so the stage 11b example's
  result remains historical rather than being rerun for this integration stage.

### Issue Resolution Progress (stage 11c)

| Work | Issue ID | Result |
| --- | --- | --- |
| Worker-owned CPU/GPU SAC backend and Pendulum profile | Unassigned | Complete, effective saved config and bounds validation |
| Headless/live CLI routing and preflight guards | Unassigned | Complete, parsing, plans, output preservation and CPU/GPU Pendulum smoke checks |
| Five-network/log-alpha/three-Adam checkpoints | Unassigned | Complete, bounded host validation, atomic save and candidate restore |
| Exact temperature/target/optimizer continuation | Unassigned | Complete, supplied-noise updates and saved bytes for tau 0/0.2/1 |
| Replay bootstrap and independent runtime noise streams | Unassigned | Complete, terminal/truncation/cutoff checks and fresh replay on resume |
| Pause, stops and failure preservation | Unassigned | Complete, retained pause transition, prior completed updates and previous file preservation |
| Eight finite JSONL metrics and live roles | Unassigned | Complete, temperature loss/alpha included and restored-config display |
| Physical GPU validation/profiling | Unassigned | Deferred by user |

### Verification (stage 11c)

| Check | Passing result | Delta from stage 11b |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,108 passed, 190 ignored, 0 failed | +8 ordinary tests; +8 adapter-required tests |
| Host SAC checkpoint codec/file validation | 3 passed, 0 failed | Wire bits/header/format, malformed networks/temperature/Adam/clocks and size/atomic cleanup |
| GPU SAC checkpoint continuation | 2 passed, 0 failed | Tau 0/0.2/1, frozen targets, exact resumed updates/bytes and failed I/O preservation |
| GPU SAC runtime controls/bootstrap | 5 passed, 0 failed | Trained resume/JSONL, stops, pause, terminal/truncation/cutoff and failures |
| GPU Pendulum CLI resume | 1 passed, 0 failed | Saved hidden profile/temperature rate; 200 steps and 137 updates |
| SAC agent/objective/shared Gaussian regressions | 15 passed, 0 failed | Four SAC agent, seven objective and four continuous PPO checks |
| Default-feature CLI device/output guards | 4 passed, 0 failed | SAC included in missing-feature/invalid environment/PER/CPU checkpoint guards |
| Python editable rebuild and regression suite | 32 passed, 0 failed | Shared config validation/serde changes included in rebuilt extension |
| Workspace Clippy/all targets, default and all features | Passed, warnings denied | Excluding Python |
| Rust 1.75 workspace/all targets/all features | Passed | Excluding Python |
| Formatting and patch whitespace | Passed | `cargo fmt --all -- --check`, `git diff --check` |

The eight ordinary additions are three host checkpoint tests, one runtime-options
test and four CLI tests (parse, invalid combinations/output preservation, live
plan/roles and CPU Pendulum metrics). The eight adapter-required additions are
two checkpoint, five runtime and one CLI checks. Explicit GPU verification runs
these eight plus 15 regressions, for **23 passing adapter-required checks**.
Checks use Mesa llvmpipe GL; physical GPU performance is not asserted. Logs are
under `/tmp/rustforge-gpu-stage11c-*.log`.

## Remaining work

All planned GPU algorithm implementation stages 1–11c are complete. Physical GPU
correctness/performance validation remains deferred at the user's request. Full
experiment-state persistence, broader environment benchmarks and any further
kernel tuning are separate follow-up work; no additional software stage is
claimed complete by these checks.
