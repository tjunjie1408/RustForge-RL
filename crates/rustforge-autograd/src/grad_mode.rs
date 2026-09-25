//! Thread-local switch for gradient tracking.
//!
//! Inference paths (action selection, target estimates) do not need a
//! computation graph. Running them under [`no_grad`] skips building graph
//! nodes and the tensor copies each node saves for its backward pass.

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether operations on this thread currently record gradients.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Runs `f` with gradient tracking disabled on the current thread.
///
/// Results computed inside have `requires_grad() == false` and no graph
/// history. The previous mode is restored when `f` returns or panics, and
/// calls may be nested.
///
/// ```rust
/// use rustforge_autograd::{no_grad, Variable};
/// use rustforge_tensor::Tensor;
///
/// let w = Variable::new(Tensor::ones(&[2, 2]), true);
/// let y = no_grad(|| w.sum());
/// assert!(!y.requires_grad());
/// ```
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = GradModeGuard::set(false);
    f()
}

struct GradModeGuard {
    previous: bool,
}

impl GradModeGuard {
    fn set(enabled: bool) -> Self {
        let previous = GRAD_ENABLED.with(|mode| mode.replace(enabled));
        Self { previous }
    }
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|mode| mode.set(self.previous));
    }
}
