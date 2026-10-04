# CPU DQN benchmark

### RL Training Benchmark: RustForge vs Stable-Baselines3

A head-to-head comparison trains **the same DQN** (matched architecture and
hyperparameters) on **CartPole**, on **CPU**, for a **50,000 environment-step
budget**, averaged over **10 runs** — RustForge driven through its Python
bindings, [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
through Gymnasium + PyTorch.

**RustForge trains ~22× faster** in end-to-end throughput. Under this
deliberately parity-matched config (single 64-unit hidden layer, vanilla DQN,
ε decaying to 0.05), neither framework reaches the strict CartPole-v1 "solved"
bar (mean reward ≥ 475 over 100 episodes) within the 50k-step budget — the
learning curves below show comparable reward trajectories for both.

![Training throughput: RustForge vs Stable-Baselines3](../benchmarks/sb3_comparison/results/speed_comparison.png)

| Framework | Train time (s) | Throughput (steps/sec) | Solved ≤ 50k steps |
|-----------|----------------|------------------------|--------------------|
| **RustForge** (native Rust) | 6.3 ± 2.7 | **13,680 ± 5,949** | 0 / 10 |
| Stable-Baselines3 (Python + PyTorch) | 96.9 ± 35.9 | 616 ± 280 | 0 / 10 |

> **On the time column:** RustForge's `DQN.train` is *episode*-budgeted (a fixed
> 300 episodes ≈ 70k env steps on average), while SB3 trains exactly 50k steps.
> The raw train-time column therefore covers *different step counts* —
> **throughput (steps/sec) is the apples-to-apples metric**, since it divides each
> framework's own steps by its own wall-clock time.

![Learning curves — reward vs environment steps](../benchmarks/sb3_comparison/results/learning_curve.png)

This is an **end-to-end system comparison** (native Rust environment + training
loop vs Python/Gymnasium + PyTorch), measured on CPU — the right regime for a
small MLP policy. Full methodology, fairness caveats, and reproduction steps:
[`benchmarks/sb3_comparison/`](../benchmarks/sb3_comparison/README.md).

> Measured on Windows 11, AMD Ryzen (16 cores), CPU-only, Python 3.14 + PyTorch CPU build.

---
