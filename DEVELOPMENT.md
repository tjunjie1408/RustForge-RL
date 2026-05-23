# RustForge-RL Development Guide 🛠️

This document outlines the guidelines and safety harnesses for developers and AI assistants working on the RustForge-RL codebase.

## AI Assistant Safety Harnesses

Before making any modifications, introducing new algorithms, or preparing commits, **AI assistants must read and strictly adhere to the guidelines** specified in the local `skills/` directory:

1. **[Pre-Commit Lint & Formatting Harness](file:///D:/self-learning/Rust/RustForge-RL/skills/pre_commit_harness.md)**: Steps to format code, fix warnings, and run clippy before any commit.
2. **[Gradient & Numerics Integrity Harness](file:///D:/self-learning/Rust/RustForge-RL/skills/gradient_numerics_harness.md)**: Safeguards to verify analytical gradients using numerical finite-differences and to check NaN/Inf propagation.
3. **[RL Convergence Sanity Harness](file:///D:/self-learning/Rust/RustForge-RL/skills/rl_convergence_harness.md)**: Convergence checks using single-transition overfitting and deterministic dummy environment convergence.
4. **[Zero-Allocation Hot-Path Harness](file:///D:/self-learning/Rust/RustForge-RL/skills/zero_allocation_harness.md)**: Zero-memory-allocation assertions in hotspots (steps, rollout fills, batch sampling).
5. **[Graph Lifecycle & Target Sync Harness](file:///D:/self-learning/Rust/RustForge-RL/skills/graph_leak_and_target_sync_harness.md)**: Detection of reference cycles in the autograd computation graph and verification of target network gradient detaching.
6. **[Deterministic & Seed Reproducibility Harness](file:///D:/self-learning/Rust/RustForge-RL/skills/seed_reproducibility_harness.md)**: Guidelines to ensure all stochastic agents (exploration, buffer sampling, weights init) behave 100% deterministically when given a seed.
