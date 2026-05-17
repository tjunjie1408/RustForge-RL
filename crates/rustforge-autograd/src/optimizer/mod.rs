//! Optimizer implementations for parameter updates.
//!
//! Optimizers adjust model parameters using computed gradients to minimize
//! a loss function. This module provides the `Optimizer` trait and concrete
//! implementations: SGD (with optional momentum) and Adam.

pub mod adam;
pub mod sgd;

/// Trait for parameter optimizers.
///
/// An optimizer holds references to a set of `Variable` parameters and updates
/// them based on their accumulated gradients.
///
/// ## Typical Usage
/// ```rust,ignore
/// let mut optimizer = SGD::new(vec![w.clone(), b.clone()], 0.01, 0.0);
///
/// for epoch in 0..100 {
///     optimizer.zero_grad();
///     let loss = compute_loss(&w, &b, &data);
///     loss.backward();
///     optimizer.step();
/// }
/// ```
pub trait Optimizer {
    /// Updates all parameters using their computed gradients.
    ///
    /// Should be called after `backward()` has populated gradients.
    fn step(&mut self);

    /// Resets all parameter gradients to `None`.
    ///
    /// Should be called at the beginning of each training iteration
    /// to prevent gradient accumulation across iterations.
    fn zero_grad(&mut self);

    /// Updates the learning rate.
    ///
    /// Used by `LRSchedule` to adjust the learning rate during training.
    /// Every optimizer must implement this — a default no-op would silently
    /// ignore scheduler calls if an optimizer forgets to override.
    fn set_lr(&mut self, lr: f32);

    /// Returns the current learning rate.
    fn lr(&self) -> f32;
}
