# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may
contain breaking changes.

## [Unreleased]

### Added

- Owned GPU REINFORCE agent with seeded categorical sampling, resident policy/Adam
  and successful-update counter. Episode-local CPU Monte Carlo rollouts use zero
  final bootstrap for termination/truncation/limits. One update consumes active
  rows with an optional mean baseline, ignoring unused capacity and old references.
  Finite input/loss/gradient/squared-gradient guards run before Adam. CPU sampling
  and update parity, analytic returns, rejection/recovery and fresh environment
  learning are tested. CPU configuration gains Clone/Debug/PartialEq; training math
  is unchanged. Runtime/CLI and checkpoints follow in stage 9c.

- GPU REINFORCE objective foundation with a seeded CPU-matching policy network,
  stable categorical policy loss, detached advantages and optional resident
  batch-mean baseline. Validates shapes, ownership and finite inputs/centered
  advantages/loss. Tests cover f64 gradient parity, both baseline modes, four
  actual CPU Adam updates, singleton/constant batches, extreme logits and invalid
  inputs. A fixed-objective example and GPU CI steps verify optimization; owned
  rollout training is described above, with runtime/CLI/checkpoints still pending.

- GPU A2C worker/runtime integration and CartPole headless/live CLI device routing.
  Distinct bounded version-1 `RFGPUA2C` checkpoints save configuration, shared
  actor/value parameters, Adam moments and update counter with atomic replacement.
  Host validation precedes candidate model allocation; resume restores training
  state with fresh environment/rollout/random streams. Runtime validates dimensions
  before reset, propagates bootstrap errors, preserves pause/graceful/forced stop
  behavior and reports eight JSONL metrics. Checkpoint continuation matches
  uninterrupted updates bit for bit; CPU math remains unchanged.

- GPU A2C agent with seeded categorical sampling, shared actor/value network,
  resident Adam, successful-update counter and CPU episode-local rollout/GAE.
  One combined update consumes active rows with raw advantages; unused capacity
  and old log probabilities are ignored. Finite loss/gradient/squared-gradient
  checks precede Adam. Actual CPU update parity, multi-step bootstrap boundaries,
  rejected-input/optimizer recovery tests and a seeded environment-learning example
  validate the agent; runtime/CLI and checkpoints are described above.

- GPU A2C objective foundation: resident categorical policy-gradient, value MSE,
  entropy and combined losses with detached rollout references, unnormalized
  advantages, shape/device validation and finite frozen-input checks.
  Reuses the seeded GPU actor/value network. Actual CPU A2C loss/gradient/four
  Adam-update parity, f64 finite differences and a fixed-batch optimization example
  validate the foundation; agent training is described above.

- Continuous PPO CPU/GPU runtime adapter and Pendulum headless/live CLI routing.
  A distinct bounded version-1 `RFGPUPC0` checkpoint stores Gaussian action bounds,
  actor/critic parameters, both Adam states and separate update counters; restore
  validates host metadata before device allocation and saves replace atomically.
  Pause/resume and graceful/forced stop follow the existing on-policy controls.
  Continuous metrics use generic JSONL without categorical entropy; environment,
  rollout and random streams restart on resume.
- Seeded continuous GPU PPO agent with Gaussian actor/value networks, explicit
  sampling and shuffle RNGs, CPU episode-local rollout/GAE, shuffled partial
  minibatches and separate resident actor/critic Adam state and update clocks.
  Finite losses and both networks' gradients are checked before updates. A
  target-action environment example verifies learning from fresh rollouts.
  CPU continuous PPO also exposes seeded construction and caller-controlled RNG
  APIs.

- Optional `rustforge-tensor/gpu` backend: reusable `GpuContext`, rank-two
  `f32` WGSL matrix multiplication, a runnable example, and adapter-required
  CPU parity tests. CPU tensor operations and training stay on their existing
  backend.
- Persistent `GpuTensor` buffers with explicit upload/download, device
  ownership checks, and chained device matrix products. `matmul_into`
  reuses output storage without intermediate tensor transfers; cloned
  contexts share their device.
- Tiled GPU matrix multiplication, copy-free transposed products, device
  addition/multiplication/ReLU, and hierarchical full sum/mean reductions.
  An explicit kernel selector and release benchmark compare the tiled and
  reference paths. CPU/software adapters retain the reference kernel by
  default, following measured software-adapter results.

- Optional `rustforge-autograd/gpu`: device-resident reverse-mode variables
  with matrix/elementwise/reduction gradients, immutable forward snapshots,
  leaf accumulation and no-grad inference. Momentum SGD keeps parameter and
  optimizer state on device. Numerical gradient tests and deterministic linear
  regression validate the foundation; CPU neural-network modules and RL agents
  still use their existing backend.

- Optional `rustforge-nn/gpu`: seeded Linear, ReLU and Sequential modules,
  explicit feature-bias broadcasting with device gradients, and supplied
  parameter uploads. GPU Adam retains bias-corrected moments on the device.
  CPU parity tests cover complete model gradients and optimizer updates;
  a seeded nonlinear MLP example verifies supervised convergence.

- Optional `rustforge-rl/gpu`: seeded uniform-replay DQN and Double DQN,
  reusable device replay batches, detached Bellman targets, frozen target
  snapshots with hard synchronization, and device Adam training. Typed u32
  GPU action selection/gather and scatter gradients avoid index readbacks
  during training. CPU parity tests and a deterministic two-state environment
  verify bootstrapping and learned policy behavior.

- Versioned GPU DQN checkpoints save online/frozen-target parameters,
  Adam moments/hyperparameters and training clocks. Host validation precedes
  device uploads, saves replace completed files atomically, and failed restores
  preserve the live agent. Bit-identical resume tests cover target delay and
  synchronization cadence; the new checkpoint example demonstrates continuation.
  GPU Adam exposes validated snapshot/restore APIs. GPU DQN rejects training
  under no-grad to keep optimizer state consistent with successful updates.

- Optional `rustforge-cli/gpu` adds `--device cpu|gpu` to headless and live
  training, with GPU DQN `--resume` and `--checkpoint` routing through full
  training-state files. Agents are constructed inside their owning worker;
  CPU remains the default. GPU requests reject unsupported algorithms and
  builds without support. Saves occur on completion or controlled stop; replay,
  environment and exploration state restart on resume. The live display and
  manifest record the selected backend.

- GPU DQN supports prioritized replay with finite nonnegative importance
  weights and resident weighted TD loss. Absolute pre-update TD errors return
  to the CPU replay buffer for priority updates. Headless/live CLI modes accept
  `--device gpu --use-per`; checkpoints preserve replay mode and beta schedule
  using the existing version-1 format. Replay contents, priorities and sampler
  RNG still restart on resume. CPU parity covers losses, gradients, updates and
  seeded priority feedback; invalid weights and TD overflow fail before updates.

- GPU categorical policy foundations: stable row log-softmax/softmax,
  exponential gradients, and differentiable clamp/minimum matching CPU clipping
  boundaries. `agent::gpu_ppo` builds clipped PPO policy loss, mean categorical
  entropy and value MSE with detached rollout references. Explicit diagnostic
  validation rejects overflowing importance ratios even when clipping hides them
  in a finite loss. CPU/finite-difference parity, optimizer update tests and a
  fixed device minibatch example validate this stage.

- Discrete GPU PPO actor/critic agent with seeded sampling and shuffled partial
  minibatches. CPU rollout collection computes GAE independently for each episode,
  with value bootstrap on truncation/step limits and zero bootstrap on terminals.
  Resident Adam updates guard objective, ratio and gradient finiteness. CPU parity,
  invalid-input tests and seeded environment learning validate the library agent;
  GPU PPO CLI/runtime selection and checkpoints are available below.

- GPU PPO Discrete live/headless runtime and CartPole CLI support via `--device
  gpu`, with `--resume`/`--checkpoint`. Version-1 `RFGPUPPO` checkpoints persist
  configuration, actor/critic parameters, Adam moments and update count with
  bounded validation and atomic replacement. Resume uses saved configuration;
  environment, rollout, random streams and run counters restart. Pause/resume and
  controlled stops preserve the existing runtime contract; forced partial
  rollouts are discarded before checkpointing. GPU checkpoints reject algorithm,
  version, shape, numerical and clock mismatches.
- Seeded the existing CPU XOR convergence test to remove random-initialization
  failures while retaining its loss and prediction assertions.

- Continuous GPU policy foundations: logarithm, stable tanh, exact-shape division
  gradients, action-column reductions and numeric clipping for detached inputs.
  `GpuGaussianTransform` evaluates scaled/squashed diagonal Gaussian densities,
  base Gaussian entropy and reparameterized sampling from frozen supplied noise.
  Continuous PPO objectives match CPU's separate policy/value updates without
  an entropy bonus. CPU/f64/gradient/Adam parity and a fixed objective example
  validate these APIs; continuous agents/runtime are described above.

- `rustforge monitor` follows Stable-Baselines3 logs: `Monitor` wrapper files
  (`monitor.csv`, one row per episode) and CSV logger files (`progress.csv`,
  with loss and exploration or entropy panels). The format is detected from
  the header.

### Fixed

- CPU Gaussian stored-action inversion evaluates atanh in `f64` before returning
  `f32`, preventing precision loss near the negative action endpoint and restoring
  symmetric densities. A fixed endpoint test validates the f64 reference.

- GPU contexts cache process compute resources while preserving separate tensor
  ownership scopes. This avoids wgpu 0.19 EGL display, context-lock and queue
  conflicts when checkpoint inspection and training workers coexist.


- The locked backtrace dependencies now support the declared Rust 1.75 MSRV.
- The SB3 log format detector passes the current strict Clippy check.

- Prioritized replay no longer samples unfilled buffer slots when the sum
  tree's floating-point totals drift, which could turn DQN training with
  `--use-per` into NaN Q-values.

## [0.1.0] - 2026-09-27

First public release.

### Added

- Tensor engine (`rustforge-tensor`) on `ndarray`, with broadcasting,
  reductions, and `matmul_t` / `t_matmul` for transposed GEMM without copies.
- Reverse-mode autograd (`rustforge-autograd`) with SGD, Adam, and RMSprop,
  plus `no_grad` for inference paths.
- Neural-network modules (`rustforge-nn`) and versioned parameter files
  (`RFPARAMS` header, format version 1; headerless files still load).
- RL environments and agents (`rustforge-rl`): CartPole, GridWorld,
  MountainCar, Pendulum; DQN (with PER and Double DQN), REINFORCE, A2C, PPO,
  SAC, and TD3.
- `rustforge` CLI with headless `train`, read-only `monitor`, and `run`, an
  in-process trainer with a live terminal console (pause/resume, graceful and
  force stop).
- Python package `rustforge-rl` (import name `rustforge`): native
  environments, DQN training and prediction, and a Gymnasium bridge. Ships
  abi3 wheels for CPython ≥ 3.9.
- Fallible APIs `try_train_dqn` and `DQN::try_select_greedy_action`.

### Fixed

- CPU Gaussian stored-action inversion evaluates atanh in `f64` before returning
  `f32`, preventing precision loss near the negative action endpoint and restoring
  symmetric densities. A fixed endpoint test validates the f64 reference.

- Non-finite values no longer crash training: NaN Q-values end a run as a
  reported failure, non-finite losses are dropped from metric records, and
  prioritized replay ignores non-finite TD errors.
- SAC and TD3 handle partially filled replay batches.
- `gather` / `scatter_add` check index bounds in release builds.
- Repeated `backward()` calls no longer compound gradients through shared
  intermediate results.
- Loading parameters validates every tensor before assigning any, and saving
  is atomic.
- Terminal console: divergence alerts for negative rewards, a working
  follow/freeze key, and scrolling bounded to the visible content.

### Changed

- The declared MSRV (Rust 1.75) is enforced in CI; `Cargo.lock` uses format
  v3 and dependency resolution prefers MSRV-compatible versions.

[Unreleased]: https://github.com/tjunjie1408/RustForge-RL/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tjunjie1408/RustForge-RL/releases/tag/v0.1.0
