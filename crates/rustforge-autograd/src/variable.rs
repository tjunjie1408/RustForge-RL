//! Core differentiable variable type.
//!
//! `Variable` wraps a `Tensor` with gradient tracking capabilities.
//! It uses `Rc<RefCell<>>` for shared ownership in the computation graph,
//! enabling automatic differentiation through reverse-mode AD.
//!
//! ## Design
//!
//! - Single-threaded (`Rc`, not `Arc`) — appropriate for eager-mode autograd.
//! - `Clone` is cheap (clones the `Rc` pointer, not the tensor data).
//! - Equality is by pointer identity (same `Rc` = same variable).

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use rustforge_tensor::Tensor;

use crate::backward;
use crate::graph::GradFn;

/// A differentiable variable that tracks gradients through a computation graph.
///
/// `Variable` is a reference-counted wrapper around tensor data with optional
/// gradient storage and a link to the computation graph (`grad_fn`).
///
/// ## Example
/// ```rust,ignore
/// use rustforge_tensor::Tensor;
/// use rustforge_autograd::Variable;
///
/// let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), true);
/// let y = &x * &x;  // y = x², graph tracks this
/// y.sum().backward();
/// // x.grad() is now Some(Tensor([2.0, 4.0, 6.0]))  (dy/dx = 2x)
/// ```
#[derive(Clone)]
pub struct Variable {
    pub(crate) inner: Rc<RefCell<VariableInner>>,
}

pub(crate) struct VariableInner {
    pub(crate) data: Tensor,
    pub(crate) grad: Option<Tensor>,
    pub(crate) requires_grad: bool,
    pub(crate) grad_fn: Option<Box<dyn GradFn>>,
}

impl Variable {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Creates a new variable with the given tensor data.
    ///
    /// ## Arguments
    /// - `data`: The tensor data.
    /// - `requires_grad`: Whether to track gradients for this variable.
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                grad: None,
                requires_grad,
                grad_fn: None,
            })),
        }
    }

    /// Creates a variable that does not require gradients.
    pub fn from_tensor(data: Tensor) -> Self {
        Self::new(data, false)
    }

    /// Internal constructor: creates a variable with a gradient function attached.
    ///
    /// Used by operations in `ops.rs` to build the computation graph.
    pub(crate) fn from_grad_fn(
        data: Tensor,
        requires_grad: bool,
        grad_fn: Option<Box<dyn GradFn>>,
    ) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                grad: None,
                requires_grad,
                grad_fn,
            })),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns a clone of the underlying tensor data.
    pub fn data(&self) -> Tensor {
        self.inner.borrow().data.clone()
    }

    /// Returns a clone of the accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().grad.clone()
    }

    /// Returns whether this variable requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Returns the shape of the underlying tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    /// Returns whether this variable has a gradient function
    /// (i.e. is the result of a differentiable operation).
    pub fn has_grad_fn(&self) -> bool {
        self.inner.borrow().grad_fn.is_some()
    }

    // ========================================================================
    // Gradient Management
    // ========================================================================

    /// Resets the gradient to `None`.
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    /// Replaces the underlying tensor data (used by optimizers for parameter updates).
    pub fn set_data(&self, data: Tensor) {
        self.inner.borrow_mut().data = data;
    }

    /// Accumulates a gradient into this variable's grad field.
    ///
    /// If no gradient exists yet, sets it. Otherwise, adds to the existing gradient.
    /// This is the core mechanism for gradient accumulation during backward pass,
    /// handling the case where a variable is used multiple times in the graph.
    pub(crate) fn accumulate_grad(&self, grad: &Tensor) {
        let mut inner = self.inner.borrow_mut();
        match &inner.grad {
            Some(existing) => {
                inner.grad = Some(existing + grad);
            }
            None => {
                inner.grad = Some(grad.clone());
            }
        }
    }

    /// Sets the gradient directly (used to seed the backward pass with 1.0).
    pub(crate) fn set_grad(&self, grad: Tensor) {
        self.inner.borrow_mut().grad = Some(grad);
    }

    // ========================================================================
    // Backward Pass
    // ========================================================================

    /// Runs reverse-mode automatic differentiation from this variable.
    ///
    /// This should be called on a **scalar** variable (the loss).
    /// After calling `backward()`, all variables with `requires_grad = true`
    /// in the computation graph will have their `.grad()` populated.
    ///
    /// ## Panics
    /// Panics if this variable is not a scalar (single element).
    ///
    /// ## Example
    /// ```rust,ignore
    /// let x = Variable::new(Tensor::from_vec(vec![3.0], &[1]), true);
    /// let y = &x * &x;  // y = x²
    /// y.backward();
    /// // x.grad() ≈ [6.0]  (dy/dx = 2x = 6)
    /// ```
    pub fn backward(&self) {
        backward::backward(self);
    }

    // ========================================================================
    // Math Operations (forward computation + graph tracking)
    // ========================================================================

    /// Matrix multiplication: self @ rhs
    ///
    /// Supports the same shape combinations as `Tensor::matmul`:
    /// - `[m,k] × [k,n]` → `[m,n]`
    /// - `[k] · [k]` → scalar
    /// - `[m,k] × [k]` → `[m]`
    /// - `[k] × [k,n]` → `[n]`
    pub fn matmul(&self, rhs: &Variable) -> Variable {
        crate::ops::var_matmul(self, rhs)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Variable {
        crate::ops::var_relu(self)
    }

    /// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Variable {
        crate::ops::var_sigmoid(self)
    }

    /// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    pub fn tanh_(&self) -> Variable {
        crate::ops::var_tanh(self)
    }

    /// Exponential: exp(x)
    pub fn exp(&self) -> Variable {
        crate::ops::var_exp(self)
    }

    /// Natural logarithm: ln(x)
    pub fn log(&self) -> Variable {
        crate::ops::var_log(self)
    }

    /// Power: x^p
    pub fn pow(&self, p: f32) -> Variable {
        crate::ops::var_pow(self, p)
    }

    /// Square root: √x
    pub fn sqrt(&self) -> Variable {
        crate::ops::var_sqrt(self)
    }

    /// Sum of all elements (returns scalar variable).
    pub fn sum(&self) -> Variable {
        crate::ops::var_sum(self)
    }

    /// Mean of all elements (returns scalar variable).
    pub fn mean(&self) -> Variable {
        crate::ops::var_mean(self)
    }

    /// Sum along a specified axis.
    pub fn sum_axis(&self, axis: usize, keepdim: bool) -> Variable {
        crate::ops::var_sum_axis(self, axis, keepdim)
    }

    /// Transpose: swaps the last two dimensions (with gradient tracking).
    ///
    /// For 2D tensors `[m, n]` → `[n, m]`.
    /// For 1D or 0D tensors, returns a clone.
    ///
    /// ## Example
    /// ```rust,ignore
    /// let w = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    /// let wt = w.t(); // shape [2, 2] transposed
    /// ```
    pub fn t(&self) -> Variable {
        crate::ops::var_transpose(self)
    }

    /// Creates a detached copy of this variable (no gradient tracking).
    ///
    /// The returned variable has the same tensor data but `requires_grad = false`
    /// and no `grad_fn`. Useful for:
    /// - Target values in loss computation
    /// - Numerical stability shifts (e.g., max subtraction in softmax)
    /// - Stopping gradient flow at a specific point
    pub fn detach(&self) -> Variable {
        Variable::from_tensor(self.data())
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

/// Pointer-based equality: two Variables are equal iff they share the same Rc.
impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Variable {}

/// Hash by Rc pointer address, consistent with PartialEq.
impl std::hash::Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.inner) as usize).hash(state);
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        f.debug_struct("Variable")
            .field("shape", &inner.data.shape())
            .field("requires_grad", &inner.requires_grad)
            .field("has_grad", &inner.grad.is_some())
            .field("has_grad_fn", &inner.grad_fn.is_some())
            .finish()
    }
}

impl Variable {
    /// Returns the input variables of this variable's gradient function, if any.
    ///
    /// Used for computation graph inspection and visualization (e.g. CLI graph export).
    /// Returns `None` for leaf variables that have no `grad_fn`.
    pub fn graph_inputs(&self) -> Option<Vec<Variable>> {
        let inner = self.inner.borrow();
        inner.grad_fn.as_ref().map(|gf| gf.inputs())
    }

    /// Returns a unique identifier for this variable based on its `Rc` pointer address.
    ///
    /// Useful for graph traversal visited-set tracking and Mermaid diagram node labeling.
    /// Note: IDs are only unique within a single program execution.
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_creation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let v = Variable::new(t, true);
        assert!(v.requires_grad());
        assert_eq!(v.shape(), vec![3]);
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_variable_from_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let v = Variable::from_tensor(t);
        assert!(!v.requires_grad());
    }

    #[test]
    fn test_grad_accumulation() {
        let v = Variable::new(Tensor::zeros(&[2]), true);
        let g1 = Tensor::ones(&[2]);
        let g2 = Tensor::ones(&[2]);
        v.accumulate_grad(&g1);
        v.accumulate_grad(&g2);
        let grad = v.grad().unwrap();
        assert_eq!(grad.to_vec(), vec![2.0, 2.0]);
    }

    #[test]
    fn test_zero_grad() {
        let v = Variable::new(Tensor::zeros(&[2]), true);
        v.accumulate_grad(&Tensor::ones(&[2]));
        assert!(v.grad().is_some());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_set_data() {
        let v = Variable::new(Tensor::zeros(&[2]), true);
        v.set_data(Tensor::ones(&[2]));
        assert_eq!(v.data().to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_clone_shares_rc() {
        let v1 = Variable::new(Tensor::zeros(&[2]), true);
        let v2 = v1.clone();
        assert_eq!(v1, v2); // same Rc pointer
        v1.accumulate_grad(&Tensor::ones(&[2]));
        assert!(v2.grad().is_some()); // shared state
    }
}
