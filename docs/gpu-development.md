# Phase 5: GPU implementation

The README roadmap identifies GPU support with wgpu as the next unfinished
Phase 5 milestone. This plan delivers that work incrementally; stages 1–3
stage 4a/4b/4c autograd, module and DQN integration, and stage 5a checkpoints
are complete.

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
| 5b | Runtime and CLI integration | Next: explicit CPU/GPU DQN device selection and checkpoint routing, preserving CPU defaults and existing file compatibility; physical GPU benchmarking remains pending |

## Architecture decisions

- The `gpu` feature belongs to `rustforge-tensor`, at the bottom of the
  dependency graph. Default builds do not enable wgpu.
- `GpuContext` owns the device, queue, cached matmul pipeline, and adapter
  metadata. Callers should reuse it rather than initialize a device for each
  multiplication.
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
  uses identity of that state, not adapter metadata, so different devices on
  the same adapter cannot mix. Tensors retain their device until dropped.
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
  reductions and softmax remain pending.
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
  `DQNConfig` and CPU `TransitionBatch`. It supports uniform-replay vanilla
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
  downloads one integer action. Configurations requesting prioritized replay
  are rejected; the CLI/runtime and Python APIs continue to use CPU agents.
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
| GPU CLI/runtime integration | Unassigned | Next implementation stage (5b) |


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

## Next implementation: stage 5b

Integrate explicit device selection into the DQN training runtime and CLI,
preserving CPU defaults and existing CPU checkpoint compatibility. GPU resume
must route through the complete training-state checkpoint API and retain target
cadence; backend construction should occur inside the owning training worker.
Replay/environment/exploration persistence and deterministic stochastic replay
need separate design and validation. Physical-GPU measurements are still
needed before making a throughput or end-to-end acceleration claim.
