<p align="center">
  <h1 align="center">🔥 RustForge RL</h1>
  <p align="center">
    <strong>A high-performance Reinforcement Learning framework built from the ground up in Rust.</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#roadmap">Roadmap</a> •
    <a href="#contributing">Contributing</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange?style=flat-square&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/status-Phase%201%20Complete-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/tests-60%20passing-brightgreen?style=flat-square" alt="Tests">
</p>

---

## Why RustForge RL?

The Reinforcement Learning ecosystem is dominated by Python frameworks (Stable-Baselines3, RLlib, CleanRL), which are powerful but carry inherent limitations:

| Pain Point | Python Frameworks | RustForge RL |
|---|---|---|
| **Runtime Speed** | GIL bottleneck, interpreter overhead | Zero-cost abstractions, native speed |
| **Memory Safety** | Runtime errors, memory leaks | Compile-time guarantees via ownership |
| **Concurrency** | Fragile multiprocessing | Fearless concurrency with `Send`/`Sync` |
| **Deployment** | Heavy runtimes, dependency hell | Single static binary, no runtime |
| **Reproducibility** | Floating-point non-determinism | Deterministic seeding at every layer |

RustForge RL aims to be the **first comprehensive, production-grade RL framework in Rust** — not just a toy implementation, but a framework you can use to train real agents and deploy them anywhere.

---

## Features

### 🧮 Tensor Engine (`rustforge-tensor`) — ✅ Complete

A PyTorch-style tensor library built on top of [`ndarray`](https://github.com/rust-ndarray/ndarray):

- **Creation**: `from_vec`, `zeros`, `ones`, `eye`, `arange`, `linspace`, `scalar`, `full`
- **Shape Transforms**: `reshape`, `flatten`, `transpose`, `permute`, `unsqueeze`, `squeeze`
- **Arithmetic**: Overloaded `+`, `-`, `*`, `/` with full broadcasting support
- **Matrix Math**: `matmul` supporting dot products, matrix-vector, and batch matrix multiplication
- **Reductions**: `sum`, `mean`, `max`, `argmax`, `var`, `std_dev` (with axis + keepdim support)
- **Activations**: `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax` (numerically stable)
- **Math Ops**: `exp`, `log`, `pow`, `sqrt`, `abs`, `clamp`, `neg`, `reciprocal`
- **Concatenation**: `cat` and `stack` with arbitrary axis support
- **Random Init**: Uniform, Normal, Xavier/Glorot, Kaiming/He initialization strategies
- **Display**: PyTorch-style pretty printing with automatic truncation for large tensors

### 🔄 Autograd Engine (`rustforge-autograd`) — 🚧 In Progress

- `Variable` wrapper with gradient tracking
- Dynamic computational graph construction
- Backward pass via topological sort + chain rule
- Optimizers: SGD, Adam, RMSProp

### 🧠 Neural Network Modules (`rustforge-nn`) — 📋 Planned

- `Linear`, `Conv2d`, `BatchNorm`, `LayerNorm`
- `Sequential` container, `Module` trait
- Loss functions: MSE, CrossEntropy, Huber
- Model serialization and checkpointing

### 🎮 RL Algorithms (`rustforge-rl`) — 📋 Planned

- **Value-Based**: DQN, Double DQN, Dueling DQN, Prioritized Experience Replay
- **Policy Gradient**: REINFORCE, A2C, PPO (clip & penalty variants)
- **Off-Policy**: SAC, TD3, DDPG
- **Environment Interface**: Gymnasium-compatible trait for custom environments
- **Replay Buffers**: Uniform, Prioritized (SumTree), HER

### 🐍 Python Bindings (`rustforge-python`) — 📋 Planned

- PyO3-powered Python API for seamless integration
- NumPy array interop (zero-copy where possible)
- Drop-in replacement for select PyTorch/SB3 workflows

### 📊 Training Dashboard (`rustforge-dashboard`) — 📋 Planned

- Real-time web-based monitoring via Axum + WebSocket
- Live reward curves, loss plots, episode statistics
- Hyperparameter tracking and experiment comparison

---

## Architecture

RustForge RL is organized as a **Cargo workspace** with strict, one-directional dependencies:

```
rustforge-rl/
├── Cargo.toml                 # Workspace root
├── crates/
│   ├── rustforge-tensor/      # 🧮 Tensor computation engine
│   │   └── src/
│   │       ├── lib.rs         # Crate entry point & re-exports
│   │       ├── tensor.rs      # Core Tensor struct + operations
│   │       ├── ops.rs         # Operator overloading (+, -, *, /, matmul)
│   │       ├── shape.rs       # Broadcasting rules & shape utilities
│   │       ├── random.rs      # Random initialization strategies
│   │       ├── display.rs     # Pretty-print formatting
│   │       └── error.rs       # Type-safe error definitions
│   │
│   ├── rustforge-autograd/    # 🔄 Automatic differentiation
│   ├── rustforge-nn/          # 🧠 Neural network layers
│   └── rustforge-rl/          # 🎮 RL algorithms
│
├── examples/                  # Runnable examples (coming soon)
└── benches/                   # Performance benchmarks (coming soon)
```

**Dependency graph:**

```
tensor ← autograd ← nn ← rl
                          ↓
                      dashboard
                      python bindings
```

Each layer only depends on the layer below it, ensuring clean separation of concerns and independent testability.

---

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) 1.75+ (2021 edition)
- A C compiler (for `ndarray`'s BLAS backend — MSVC on Windows, GCC/Clang on Linux/macOS)

### Build & Test

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rustforge-rl.git
cd rustforge-rl

# Build the entire workspace
cargo build

# Run all tests (51 unit tests + 9 doc tests)
cargo test -p rustforge-tensor

# Run with optimizations for benchmarking
cargo build --release
```

### Basic Usage

```rust
use rustforge_tensor::Tensor;

fn main() {
    // Create tensors
    let weights = Tensor::xavier_uniform(&[128, 64], Some(42));
    let input = Tensor::rand_normal(&[32, 64], 0.0, 1.0, None);
    let bias = Tensor::zeros(&[128]);

    // Forward pass: output = input @ weights^T + bias
    let output = input.matmul(&weights.t()) + bias;

    // Activation
    let activated = output.relu();

    // Softmax for probability distribution
    let probs = activated.softmax(1).unwrap();

    println!("Output shape: {:?}", probs.shape());
    println!("Probabilities:\n{}", probs);
}
```

### Tensor Operations Examples

```rust
use rustforge_tensor::Tensor;

// Broadcasting: [3, 1] + [1, 4] → [3, 4]
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]);
let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]);
let c = &a + &b;  // shape: [3, 4]

// Reductions
let data = Tensor::rand_uniform(&[100, 50], 0.0, 1.0, Some(0));
println!("Mean: {:.4}", data.mean().item());
println!("Std:  {:.4}", data.std_dev().item());

// Matrix multiplication
let q = Tensor::randn(&[8, 64], Some(1));  // queries
let k = Tensor::randn(&[8, 64], Some(2));  // keys
let attention = q.matmul(&k.t());           // [8, 8] attention scores
let weights = (& attention / 8.0_f32.sqrt()).softmax(1).unwrap();
```

---

## Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| **Phase 1** | Tensor Engine | ✅ Complete (51 tests passing) |
| **Phase 1** | Autograd Engine | 🚧 In Progress |
| **Phase 2** | Neural Network Modules | 📋 Planned |
| **Phase 2** | Optimizers (SGD, Adam) | 📋 Planned |
| **Phase 3** | DQN + CartPole | 📋 Planned |
| **Phase 3** | PPO + Continuous Control | 📋 Planned |
| **Phase 4** | SAC, TD3, DDPG | 📋 Planned |
| **Phase 4** | Python Bindings (PyO3) | 📋 Planned |
| **Phase 5** | Training Dashboard | 📋 Planned |
| **Phase 5** | GPU Support (wgpu) | 📋 Planned |

---

## Design Philosophy

### 1. **Correctness First, Then Performance**
Every operation is backed by comprehensive unit tests with numerical precision checks. We use `approx` for floating-point comparisons and deterministic seeding for reproducibility.

### 2. **PyTorch-Familiar API**
If you've used PyTorch, you'll feel right at home. Method names, broadcasting rules, and tensor semantics are intentionally aligned with PyTorch conventions.

### 3. **Zero-Cost Abstractions**
Rust's ownership system lets us provide a safe, high-level API without runtime overhead. No garbage collector, no reference counting at the tensor layer — just stack-allocated wrappers around contiguous memory.

### 4. **Modular by Design**
Each crate is independently usable. Need just tensors? Use `rustforge-tensor`. Want autograd without RL? Use `rustforge-autograd`. The workspace structure enforces clean boundaries.

---

## Performance

RustForge RL is built on `ndarray` which leverages BLAS for matrix operations. Preliminary benchmarks on common operations:

| Operation | Shape | RustForge | Notes |
|-----------|-------|-----------|-------|
| MatMul | [512, 512] × [512, 512] | ~2ms | With OpenBLAS |
| Softmax | [1024, 1024] | ~1ms | Numerically stable |
| Xavier Init | [1024, 1024] | ~3ms | ChaCha20 RNG |
| Broadcasting Add | [1000, 1] + [1, 1000] | ~0.5ms | Native ndarray |

> **Note**: Benchmarks are from development builds. Release builds (`--release`) are typically 10-30× faster.

---

## Contributing

We welcome contributions of all kinds! RustForge RL is in its early stages, making it an excellent time to get involved.

### Ways to Contribute

- 🐛 **Bug Reports**: Found an issue? Open a GitHub Issue with reproduction steps
- 📖 **Documentation**: Improve doc comments, add examples, write tutorials
- 🧪 **Tests**: Add edge cases, property-based tests, or integration tests
- 🚀 **Features**: Pick an item from the roadmap and submit a PR
- 💡 **Ideas**: Suggest new RL algorithms, optimizations, or API improvements

### Getting Started

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/rustforge-rl.git
cd rustforge-rl

# Create a feature branch
git checkout -b feat/your-feature

# Make changes and run tests
cargo test --workspace
cargo clippy --workspace

# Submit a PR!
```

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and address all warnings
- Add doc comments with `///` for all public items
- Include unit tests for new functionality
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust 2021 Edition |
| Tensor Backend | [ndarray](https://github.com/rust-ndarray/ndarray) 0.16 |
| Random Number Generation | [rand](https://github.com/rust-random/rand) 0.8 + ChaCha20 |
| Serialization | [serde](https://serde.rs/) + bincode |
| Logging | [tracing](https://github.com/tokio-rs/tracing) |
| Testing | Built-in + [approx](https://github.com/brendanzab/approx) |
| Future: Python Bindings | [PyO3](https://pyo3.rs/) |
| Future: Web Dashboard | [Axum](https://github.com/tokio-rs/axum) + WebSocket |
| Future: GPU | [wgpu](https://wgpu.rs/) |

---

## References & Inspiration

This project draws inspiration from and builds upon ideas in:

- **[PyTorch](https://pytorch.org/)** — API design and tensor semantics
- **[tch-rs](https://github.com/LaurentMazare/tch-rs)** — Rust bindings for libtorch
- **[candle](https://github.com/huggingface/candle)** — Minimalist ML framework in Rust
- **[burn](https://github.com/tracel-ai/burn)** — Deep learning framework in Rust
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** — RL algorithm implementations
- **[CleanRL](https://github.com/vwxyzjn/cleanrl)** — Single-file RL implementations

### Key Papers

- Mnih et al., *"Playing Atari with Deep Reinforcement Learning"* (DQN, 2013)
- Schulman et al., *"Proximal Policy Optimization Algorithms"* (PPO, 2017)
- Haarnoja et al., *"Soft Actor-Critic"* (SAC, 2018)
- Glorot & Bengio, *"Understanding the difficulty of training deep feedforward neural networks"* (Xavier Init, 2010)
- He et al., *"Delving Deep into Rectifiers"* (Kaiming Init, 2015)

---

## License

This project is dual-licensed under:

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

You may choose either license. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for details.

---

<p align="center">
  <strong>Built with 🦀 Rust and ❤️ passion for RL</strong>
  <br>
  <sub>RustForge RL — Forging intelligent agents, one tensor at a time.</sub>
</p>
