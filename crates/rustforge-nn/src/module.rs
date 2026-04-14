//! Core `Module` trait — the fundamental abstraction for all neural network layers.
//!
//! Every neural network component (Linear, ReLU, Sequential, etc.) implements this
//! trait, providing a uniform interface for forward computation and parameter access.
//!
//! ## Design Rationale
//!
//! - `forward(&self, ...)` uses shared reference to allow computation graph building
//!   without mutability concerns. Only `set_training` requires `&mut self`.
//! - `parameters()` returns cloned `Variable` handles (cheap Rc clone), enabling
//!   optimizer construction without lifetime issues.
//! - Default implementations for `set_training`/`is_training` make stateless layers
//!   (ReLU, etc.) zero-boilerplate.

use rustforge_autograd::Variable;

/// Trait for neural network modules.
///
/// ## Example
/// ```rust,ignore
/// use rustforge_nn::{Module, Linear, ReLU};
///
/// let linear = Linear::new(4, 2);
/// let relu = ReLU;
///
/// let x = Variable::new(Tensor::randn(&[8, 4], None), false);
/// let h = linear.forward(&x);
/// let y = relu.forward(&h);
///
/// // Collect parameters for optimizer
/// let params = linear.parameters();
/// ```
pub trait Module {
    /// Performs the forward pass: transforms input into output.
    ///
    /// This builds the computation graph implicitly through `Variable` operations,
    /// enabling automatic gradient computation via `backward()`.
    fn forward(&self, input: &Variable) -> Variable;

    /// Returns all learnable parameters of this module.
    ///
    /// Used to construct optimizers: `SGD::new(model.parameters(), lr, momentum)`.
    /// For composite modules (e.g., Sequential), this recursively collects
    /// parameters from all children.
    fn parameters(&self) -> Vec<Variable>;

    /// Switches between training and evaluation mode.
    ///
    /// Affects behavior of layers like Dropout (random masking vs. identity)
    /// and BatchNorm (running stats vs. batch stats). Default is no-op for
    /// stateless layers.
    fn set_training(&mut self, _training: bool) {}

    /// Returns whether the module is in training mode.
    ///
    /// Default is `true` (training mode).
    fn is_training(&self) -> bool {
        true
    }
}
