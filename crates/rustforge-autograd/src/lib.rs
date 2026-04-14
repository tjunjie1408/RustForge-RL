//! # RustForge Autograd
//!
//! Reverse-mode automatic differentiation engine — the gradient computation backbone
//! of the RustForge RL framework.
//!
//! This crate provides `Variable`, a differentiable wrapper around `Tensor` that
//! automatically tracks operations in a computation graph and computes gradients
//! via reverse-mode AD (backpropagation).
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rustforge_tensor::Tensor;
//! use rustforge_autograd::{Variable, Optimizer};
//! use rustforge_autograd::optimizer::sgd::SGD;
//!
//! // Create differentiable variables
//! let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
//! let w = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]), true);
//!
//! // Forward pass (computation graph is built implicitly)
//! let y = x.matmul(&w).relu().sum();
//!
//! // Backward pass (gradients are computed for all requires_grad variables)
//! y.backward();
//!
//! // Access gradients
//! println!("dw = {:?}", w.grad());
//! ```
//!
//! ## Supported Operations
//!
//! | Category | Operations |
//! |----------|-----------|
//! | Arithmetic | `+`, `-`, `*`, `/`, negation |
//! | Matrix | `matmul` |
//! | Activations | `relu`, `sigmoid`, `tanh` |
//! | Math | `exp`, `log`, `pow`, `sqrt` |
//! | Reductions | `sum`, `mean`, `sum_axis` |
//! | Scalar | `Variable ± f32`, `Variable × f32` |

pub mod backward;
pub mod graph;
pub mod ops;
pub mod optimizer;
pub mod variable;

// Re-export core types for user convenience
pub use optimizer::Optimizer;
pub use variable::Variable;
