//! # RustForge Tensor
//!
//! High-performance tensor computation engine — the foundational layer of the RustForge RL framework.
//!
//! This crate provides core functionalities such as creation, operations, and reshaping of N-dimensional tensors.
//! Built on top of `ndarray`, it supports broadcasting semantics and various random initialization strategies.
//!
//! ## Quick Start
//!
//! ```rust
//! use rustforge_tensor::Tensor;
//!
//! // Create tensors
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = Tensor::ones(&[2, 2]);
//!
//! // Basic operations
//! let c = &a + &b;
//! let d = a.matmul(&b);
//!
//! // Mathematical functions
//! let e = c.relu();
//! let f = d.softmax(1);
//! ```

pub mod display;
pub mod error;
pub mod ops;
pub mod random;
pub mod shape;
pub mod tensor;

// Re-export core types for user convenience
pub use error::TensorError;
pub use tensor::Tensor;
