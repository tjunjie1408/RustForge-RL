# GPU development guide

GPU support is optional and explicit. Implementation covers DQN, categorical
and continuous PPO, A2C, REINFORCE, TD3 and SAC through stages 1–11c. CPU APIs
remain available; Python agents continue to use the CPU backend.

- [Hardware measurements and verification history](gpu-performance.md)
- [Why CLI training differs from a matrix benchmark](gpu-cli-performance.md)
- [GPU training instrumentation and profiling](gpu-profiling.md)
- [Benchmark collection and plotting](../benchmarks/gpu_comparison/README.md)

This page summarizes current contracts. Detailed stage-by-stage implementation
and historical test inventories remain in Git history.

## Supported capabilities

| Layer | Implemented GPU support |
| --- | --- |
| Tensor | Owned device storage, explicit transfers, direct/tiled matrix products, transposed products, output reuse, elementwise operations, reductions, categorical indices/probabilities and Gaussian-policy primitives |
| Autograd | Fallible device variables, immutable forward snapshots, backward propagation, detached/no-grad execution and resident SGD/Adam state |
| Neural networks | Seeded Linear/ReLU/Sequential, feature bias gradients and frozen parameter snapshots |
| DQN | Vanilla/Double DQN, uniform/prioritized CPU replay, resident training and target synchronization |
| PPO | CartPole categorical and Pendulum Gaussian policies, CPU rollout/GAE, minibatch updates and resident optimizers |
| A2C / REINFORCE | CartPole policy learning, CPU rollout/return construction and guarded device updates |
| TD3 / SAC | Pendulum replay training, twin critics and targets; TD3 delayed actor updates; SAC learned entropy temperature |
| Runtime | Explicit CPU/GPU selection, headless/live CLI routes, generic metrics, controls and agent-state checkpoints |

## Architecture and contracts

### Device storage and execution

- `GpuContext` owns a tensor scope over a process-wide device, queue and cached
  pipelines. Clones share ownership; separately created contexts reject each
  other's tensors even when they use the same adapter. Tensors retain their
  scope until dropped.
- `GpuTensor` owns contiguous device storage and immutable shape metadata.
  Uploads pack noncontiguous CPU data in logical row-major order. Empty tensors
  preserve logical shapes using minimal physical storage.
- `matmul_device` queues work without host synchronization. `matmul_into` reuses
  an exact-shape output buffer. Dispatch uniforms use a bounded scratch cache;
  bind groups and command encoders are still created for each dispatch.
  Queue ordering allows device results to feed subsequent operations.
- `command_batch` groups compute command buffers on the calling thread, with
  automatic flushing at downloads/waits and a 32-command limit. Linear layers
  and optimizers, plus backward traversals, use it internally. Finish a producing
  batch before handing its outputs to another thread.
- Fully written kernel outputs recycle storage after their last live owner drops.
  Deferred batches retain storage/uniform leases until submission. Public zeros,
  empty reductions and partially written scatter outputs keep zero initialization.
- The convenience `matmul` uploads CPU inputs and downloads the result. Explicit
  downloads wait for completion; device-only chaining avoids those transfers.
- `download_scalars` packs single-element results into one readback submission
  and wait. Gaussian and TD3/SAC objective validation use it for checks and
  metrics. Readback staging buffers use a bounded cache shared by context clones;
  see the [synchronization/buffer follow-up](gpu-sync-buffer-reuse.md) and the
  [readback comparison](gpu-batched-readback.md).
- `download_tensors` packs arbitrary f32 shapes into one readback; TD3/SAC
  inference combines final action validation and action copies. The
  [execution walkthrough](gpu-execution-optimization.md) describes these changes
  and fused SGD/Adam arithmetic.
- Finite checks map indicators directly into their first hierarchical reduction;
  backward batches gradient commands and uses a fused tanh derivative. TD3/SAC
  and Gaussian validation batch compute up to existing readback boundaries. See the
  [review follow-up](gpu-reduction-backward-optimization.md) for paired work
  counts, allocation tradeoffs and the next implementation candidates.
- CPU/software adapters default to the direct matrix kernel; other adapters use
  the tiled kernel. `with_matmul_kernel` changes that choice on a shared context.
  The tiled shader uses 8 × 8 workgroup tiles and supports edge/transpose cases.
- Binary elementwise operations require matching shapes. Feature-vector bias
  and selected row/column operations have explicit APIs; arbitrary broadcasting
  is not implied. Typed action indices use u32 storage and validated bounds.

### Training and validation

- GPU APIs are fallible and separate from CPU tensor/variable APIs. CPU
  environments, replay sampling and rollout processing remain host-side;
  training uploads active batches and explicitly reads actions/metrics back.
- Immutable graph snapshots keep prior forwards and frozen targets stable when
  optimizers replace leaf buffers. Leaves must be distinct, trainable and owned
  by one context. Losses, gradients and candidate updates are validated before
  guarded updates replace live state.
- Truncation is distinct from termination: use the replay terminal helper so a
  time-limit truncation can continue bootstrapping.
- DQN prioritized replay retains priorities on CPU and downloads absolute TD
  errors for feedback. This is not a fully device-resident replay system.
- Scalar loss/finite checks and action downloads can synchronize with the host.
  Resident model/optimizer state does not imply transfer-free training or
  end-to-end GPU acceleration. See the [performance explanation](gpu-cli-performance.md).

### Agent checkpoints and controls

GPU checkpoint formats retain algorithm configuration, parameters, optimizer
state and update clocks; target networks and SAC temperature state are included
where applicable. Load/restore validates the format and host metadata before
uploading candidates. Saves prepare data before atomically replacing the file.
Restores preserve delayed targets rather than silently synchronizing them.

These are **agent training-state checkpoints**, not full experiment recovery.
Environment state, replay/rollout state, caller RNG streams and all runtime
progress are not captured together. Model/optimizer update continuation does
not prove bit-identical process-restart training.

Pause/resume controls affect the live process. Graceful stop finishes the current
episode; force stop exits after the current step without inventing a partial
completed-episode event. `monitor` reads metrics and has no trainer controls.

## Run locally

From the workspace root, with Rust and a working compute driver:

```bash
cargo build --release --locked -p rustforge-cli --features gpu --bin rustforge
cargo run --locked -p rustforge-cli --features gpu -- train dqn --device gpu --env cartpole --episodes 30
cargo run --locked -p rustforge-cli --features gpu -- train ppo --device gpu --env pendulum --episodes 10
cargo run --locked -p rustforge-cli --features gpu -- run dqn --device gpu
```

`--device cpu` selects the CPU route. Explicit metrics output paths are protected
against accidental overwrite. Use the CLI's `--help` for checkpoint and control
options. A GPU feature build is required for `--device gpu`.

The tensor example prints the actual adapter and checks both CPU-to-CPU and
resident chained operations. The benchmark accepts an iteration count and an
optional comma-separated list of square sizes:

```bash
cargo run --locked -p rustforge-tensor --features gpu --example gpu_matmul
cargo run --release --locked -p rustforge-tensor --features gpu --example gpu_benchmark -- 100 32,64,128,256,512,1024
```

A software adapter can validate shaders; it does not establish hardware speed.
Always record adapter metadata rather than inferring the device from the feature
flag or a successful build.

## Verification

Adapter-required tests are ignored by default. Execute them explicitly:

```bash
cargo test --locked -p rustforge-tensor --features gpu --test gpu_matmul -- --include-ignored --test-threads=1
cargo test --locked -p rustforge-tensor --features gpu --lib gpu::tests::persistent_operations_avoid_transfers_and_reuse_storage -- --include-ignored
cargo test --locked -p rustforge-rl --features gpu --test gpu_ppo_loss --test gpu_a2c_loss --test gpu_reinforce_loss -- --include-ignored --test-threads=1
cargo fmt --all -- --check
cargo clippy --locked -p rustforge-tensor --features gpu --all-targets -- -D warnings
```

Algorithm-specific objective, agent, checkpoint and runtime suites live in
[RL tests](../crates/rustforge-rl/tests); CLI device selection is covered in
[CLI tests](../crates/rustforge-cli/tests/device_selection.rs).
[CI](../.github/workflows/ci.yml) lists the broader explicit GPU test matrix.
A CI configuration is not evidence that a remote run passed.

On **2026-10-05**, the RTX 5060 Laptop GPU/Vulkan post-fix checks passed:
24 GPU tensor tests, 11 PPO/A2C/REINFORCE loss tests, strict tensor Clippy,
formatting, release CLI build and a two-episode GPU PPO smoke run. The softmax
repair adds an offset-invariance regression without relaxing tolerances.
[The validation record](../benchmarks/gpu_comparison/results/2026-10-05/softmax-fix-validation.json)
records the checked scope; it is not a full hardware correctness sweep.

## Performance evidence

The [physical GPU report](gpu-performance.md) contains three trials per profile,
18 matrix records, 42 CPU/GPU CLI processes, PNG/SVG comparisons, raw data and
reproduction commands. Tiled 1024-square matmul was 12.74× faster than native CPU
matmul; all seven default CLI profiles had lower GPU throughput. Matrix timings
exclude transfers and initialization; CLI timings include them.

The performance data and binary hashes describe the **pre-softmax-fix build**.
Post-fix correctness checks are separate; the full timing suite was not rerun.
Do not treat a resident matrix speedup as a training or GPU-vs-SB3 speedup.

Historical Mesa llvmpipe GL measurements remain in
[the software-adapter CSV](gpu-benchmark-llvmpipe.csv). They use a different
machine and timing setup; they are not physical-GPU estimates. The historical
[CPU DQN/SB3 benchmark](performance.md#historical-cpu-dqn-benchmark) is also a
separate workload. No profiler has apportioned current CLI overheads.

## Remaining work

- Full experiment-state persistence and proven process-restart recovery.
- Broader hardware/driver correctness sweeps, larger workloads and profiler data.
- Transfer/submission/allocation tuning without weakening update validation.
- Broader GPU module/operator coverage; existing explicit APIs do not imply
  general broadcasting, every CPU module, or Python GPU agent support.
