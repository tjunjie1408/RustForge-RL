# Contributing to RustForge RL

First off, thank you for considering contributing to **RustForge RL**! 🎉

Whether you're fixing a bug, adding a new RL algorithm, or improving the documentation, your help is greatly appreciated. This document provides guidelines to ensure a smooth contribution process.

## 🧠 Project Architecture Overview

Before diving into the code, it's helpful to understand the workspace structure. RustForge RL enforces a strict one-way dependency flow:

1. `rustforge-tensor` (Foundation: Math & Tensors)
2. `rustforge-autograd` (Automatic Differentiation)
3. `rustforge-nn` (Neural Network Layers & Modules)
4. `rustforge-rl` (Reinforcement Learning Algorithms)

> **Rule of Thumb:** A lower layer cannot depend on a higher layer. For example, `tensor` should know nothing about "gradients", and `nn` should know nothing about "agents".

## 🚀 How to Contribute

### 1. Find an Issue
- Browse the [Issue Tracker](https://github.com/tjunjie1408/RustForge-RL/issues) to find tasks.
- If you have a new idea or found a bug, please **open an issue first** to discuss it before writing code.

### 2. Fork and Branch
1. Fork the repository to your own GitHub account.
2. Clone the repository to your local machine.
3. Create a descriptive branch name from `master`:
   ```bash
   git checkout -b feat/add-ppo-algorithm
   # or
   git checkout -b fix/tensor-broadcasting-bug
   ```

### 3. Development Workflow
RustForge RL prioritizes **correctness and stability**. Please follow these steps iteratively:

- **Write Code**: Ensure your code is modular and placed in the correct crate.
- **Write Tests**: Every new feature or operation *must* have unit tests. If you are doing numerical operations, use the `approx` crate (`assert_abs_diff_eq!`) for floating-point comparisons.
- **Run Tests**: Verify your changes do not break existing functionality.
  ```bash
  cargo test --workspace
  ```

### 4. Code Quality Standards
Before creating a Pull Request, you must run the following commands. The CI pipeline will fail if these checks do not pass:

**Formatting:**
Ensure your code matches the standard Rust style:
```bash
cargo fmt --all
```

**Linting:**
We enforce a warning-free codebase. Run Clippy:
```bash
cargo clippy --workspace -- -D warnings
```

### 5. Commit Standards
We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Please structure your commit messages like this:
- `feat(tensor): add chunk method`
- `fix(autograd): resolve gradient accumulation bug in broadcast`
- `docs: update readme with quickstart`
- `style: apply rustfmt formatting`
- `test(nn): add integration tests for XOR MLP`

### 6. Open a Pull Request (PR)
- Push your branch to your forked repository.
- Open a PR against the `master` branch of the main repository.
- Fill out the PR template describing *why* and *how* you made the changes.

## 🤝 Getting Help

If you are stuck, feel free to ask questions in the issue or PR comments. We are happy to help new contributors get familiar with the codebase!

Happy Hacking! 🦀
