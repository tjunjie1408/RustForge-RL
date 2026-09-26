# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may
contain breaking changes.

## [Unreleased]

## [0.1.0] - Unreleased

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
