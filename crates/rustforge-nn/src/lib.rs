//! # RustForge NN — Neural Network Modules
//!
//! A modular neural network library built on top of `rustforge-autograd`.
//! Provides the `Module` abstraction, common layers, loss functions, and
//! training utilities.
//!
//! ## Architecture
//!
//! ```text
//! rustforge-nn
//! ├── Module trait         (core abstraction)
//! ├── Linear               (fully-connected layer)
//! ├── Activations          (ReLU, Sigmoid, Tanh, Softmax)
//! ├── Loss functions       (MSE, CrossEntropy, Huber)
//! ├── Sequential           (layer container)
//! ├── Dropout              (regularization)
//! ├── LayerNorm            (normalization)
//! └── Serialization        (save/load parameters)
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rustforge_tensor::Tensor;
//! use rustforge_autograd::{Variable, Optimizer};
//! use rustforge_autograd::optimizer::adam::Adam;
//! use rustforge_nn::*;
//!
//! // Build a 2-layer MLP
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(2, 16)),
//!     Box::new(ReLU),
//!     Box::new(Linear::new(16, 1)),
//!     Box::new(Sigmoid),
//! ]);
//!
//! // Setup optimizer
//! let mut optimizer = Adam::new(model.parameters(), 0.01, 0.9, 0.999, 1e-8);
//!
//! // Training loop
//! for epoch in 0..1000 {
//!     optimizer.zero_grad();
//!     let output = model.forward(&input);
//!     let loss = mse_loss(&output, &target);
//!     loss.backward();
//!     optimizer.step();
//! }
//! ```

pub mod activation;
pub mod dropout;
pub mod linear;
pub mod loss;
pub mod module;
pub mod normalization;
pub mod sequential;
pub mod serialization;

// Re-export core types for user convenience
pub use activation::{ReLU, Sigmoid, Softmax, Tanh};
pub use dropout::Dropout;
pub use linear::Linear;
pub use loss::{cross_entropy_loss, huber_loss, mse_loss};
pub use module::Module;
pub use normalization::LayerNorm;
pub use sequential::Sequential;
pub use serialization::{load_parameters, save_parameters};
