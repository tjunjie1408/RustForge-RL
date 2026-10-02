# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may
contain breaking changes.

## [Unreleased]

### Added

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
  verify bootstrapping and learned policy behavior. Prioritized replay and
  CLI/runtime integration remain pending.

- Versioned GPU DQN checkpoints save online/frozen-target parameters,
  Adam moments/hyperparameters and training clocks. Host validation precedes
  device uploads, saves replace completed files atomically, and failed restores
  preserve the live agent. Bit-identical resume tests cover target delay and
  synchronization cadence; the new checkpoint example demonstrates continuation.
  GPU Adam exposes validated snapshot/restore APIs. GPU DQN rejects training
  under no-grad to keep optimizer state consistent with successful updates.

- `rustforge monitor` follows Stable-Baselines3 logs: `Monitor` wrapper files
  (`monitor.csv`, one row per episode) and CSV logger files (`progress.csv`,
  with loss and exploration or entropy panels). The format is detected from
  the header.

### Fixed

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
