//! Computation graph nodes (gradient functions).
//!
//! Each differentiable operation creates a `GradFn` implementation that records:
//! 1. The input `Variable`s (for graph traversal and gradient accumulation)
//! 2. Any cached forward-pass values needed for gradient computation
//!
//! ## Chain Rule
//!
//! For each operation y = f(a, b, ...):
//!   ∂L/∂a = ∂L/∂y · ∂y/∂a
//!
//! The `backward()` method receives ∂L/∂y (grad_output) and computes ∂L/∂a, ∂L/∂b, ...

use rustforge_tensor::Tensor;
use smallvec::{smallvec, SmallVec};

use crate::variable::Variable;

/// Stack-allocated small vectors for GradFn results.
/// Capacity 2 covers all unary and binary ops without heap allocation.
pub type GradInputs = SmallVec<[Variable; 2]>;
pub type GradOutputs = SmallVec<[Tensor; 2]>;

/// Trait for computation graph nodes.
///
/// Each operation that produces a `Variable` with `requires_grad = true`
/// stores a `GradFn` that can compute gradients for its inputs.
pub trait GradFn {
    /// Returns the input variables this operation depends on.
    /// Used for topological sort during the backward pass.
    fn inputs(&self) -> GradInputs;

    /// Computes gradients for each input given the output gradient.
    ///
    /// Returns gradients in the same order as `inputs()`.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs;
}

// Helper: Broadcasting gradient reduction

/// Reduces a gradient tensor to match the original shape of a broadcast input.
///
/// When broadcasting occurs during a forward op (e.g. `[3,4] + [4]` → `[3,4]`),
/// the gradient for the smaller input must be summed along the broadcast dimensions
/// to restore its original shape.
///
/// ## Algorithm
/// 1. Pad `original_shape` with leading 1s to match `grad` dimensions
/// 2. For each dimension where padded_shape[i] == 1 but grad_shape[i] > 1,
///    sum along that axis (keepdim=true)
/// 3. Reshape to `original_shape`
///
/// ## Example
/// If `grad` has shape [3, 4] and `original_shape` is [4]:
/// → Sum along axis 0 → [1, 4] → reshape to [4]
pub(crate) fn reduce_grad_for_broadcast(grad: &Tensor, original_shape: &[usize]) -> Tensor {
    let grad_shape = grad.shape();
    let orig_shape = original_shape;

    // Fast path: shapes already match
    if grad_shape == orig_shape {
        return grad.clone();
    }

    let grad_ndim = grad_shape.len();
    let orig_ndim = orig_shape.len();
    let pad_len = grad_ndim.saturating_sub(orig_ndim);

    // Build padded shape on stack for ndim <= 8, heap fallback otherwise
    let padded: smallvec::SmallVec<[usize; 8]> = {
        let mut p = smallvec::SmallVec::new();
        p.extend(std::iter::repeat_n(1usize, pad_len));
        p.extend_from_slice(orig_shape);
        p
    };

    let mut result = grad.clone();

    // Sum along dimensions that were broadcast (padded[i]==1 but result has size > 1).
    // Process from highest axis to lowest so indices remain valid after each reduction.
    for i in (0..grad_ndim).rev() {
        if padded[i] == 1 && result.shape()[i] > 1 {
            result = result.sum_axis(i, true).unwrap();
        }
    }

    // Remove any extra leading dimensions and reshape to original shape
    result.reshape(orig_shape).unwrap()
}

// Addition: y = a + b

pub struct AddGrad {
    pub lhs: Variable,
    pub rhs: Variable,
}

impl GradFn for AddGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.lhs.clone(), self.rhs.clone()]
    }

    /// ∂L/∂a = grad_output, ∂L/∂b = grad_output
    /// (with broadcast reduction if shapes differ)
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let grad_lhs = reduce_grad_for_broadcast(grad_output, &self.lhs.shape());
        let grad_rhs = reduce_grad_for_broadcast(grad_output, &self.rhs.shape());
        smallvec![grad_lhs, grad_rhs]
    }
}

// Subtraction: y = a - b

pub struct SubGrad {
    pub lhs: Variable,
    pub rhs: Variable,
}

impl GradFn for SubGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.lhs.clone(), self.rhs.clone()]
    }

    /// ∂L/∂a = grad_output, ∂L/∂b = -grad_output
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let grad_lhs = reduce_grad_for_broadcast(grad_output, &self.lhs.shape());
        let neg_grad = grad_output.neg();
        let grad_rhs = reduce_grad_for_broadcast(&neg_grad, &self.rhs.shape());
        smallvec![grad_lhs, grad_rhs]
    }
}

// Element-wise multiplication: y = a * b (Hadamard product)

pub struct MulGrad {
    pub lhs: Variable,
    pub rhs: Variable,
    /// Cached lhs forward data (needed for rhs gradient)
    pub lhs_data: Tensor,
    /// Cached rhs forward data (needed for lhs gradient)
    pub rhs_data: Tensor,
}

impl GradFn for MulGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.lhs.clone(), self.rhs.clone()]
    }

    /// ∂L/∂a = grad_output * b, ∂L/∂b = grad_output * a
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let raw_grad_lhs = grad_output * &self.rhs_data;
        let raw_grad_rhs = grad_output * &self.lhs_data;
        let grad_lhs = reduce_grad_for_broadcast(&raw_grad_lhs, &self.lhs.shape());
        let grad_rhs = reduce_grad_for_broadcast(&raw_grad_rhs, &self.rhs.shape());
        smallvec![grad_lhs, grad_rhs]
    }
}

// Division: y = a / b

pub struct DivGrad {
    pub lhs: Variable,
    pub rhs: Variable,
    pub lhs_data: Tensor,
    pub rhs_data: Tensor,
}

impl GradFn for DivGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.lhs.clone(), self.rhs.clone()]
    }

    /// ∂L/∂a = grad_output / b
    /// ∂L/∂b = -grad_output * a / b²
    ///
    /// Numerical safety: clamps b² away from zero to prevent inf gradients.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        debug_assert!(
            self.rhs_data.data().iter().all(|v| *v != 0.0),
            "DivGrad: division by zero detected in rhs_data"
        );

        let raw_grad_lhs = grad_output / &self.rhs_data;

        let numerator = grad_output * &self.lhs_data;
        let b_sq = &self.rhs_data * &self.rhs_data;
        // Clamp b² away from zero to prevent inf/NaN gradient explosion
        let b_sq_safe = Tensor::from_ndarray(b_sq.data().mapv(|x| {
            if x.abs() < 1e-12 {
                1e-12_f32.copysign(x)
            } else {
                x
            }
        }));
        let raw_grad_rhs = (&numerator / &b_sq_safe).neg();

        let grad_lhs = reduce_grad_for_broadcast(&raw_grad_lhs, &self.lhs.shape());
        let grad_rhs = reduce_grad_for_broadcast(&raw_grad_rhs, &self.rhs.shape());
        smallvec![grad_lhs, grad_rhs]
    }
}

// Negation: y = -a

pub struct NegGrad {
    pub input: Variable,
}

impl GradFn for NegGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂a = -grad_output
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        smallvec![grad_output.neg()]
    }
}

// Matrix multiplication: y = a @ b

pub struct MatmulGrad {
    pub lhs: Variable,
    pub rhs: Variable,
    pub lhs_data: Tensor,
    pub rhs_data: Tensor,
}

impl GradFn for MatmulGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.lhs.clone(), self.rhs.clone()]
    }

    /// For y = A @ B:
    ///   ∂L/∂A = grad_output @ Bᵀ
    ///   ∂L/∂B = Aᵀ @ grad_output
    ///
    /// Handles 1D and 2D cases with appropriate reshaping.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let a = &self.lhs_data;
        let b = &self.rhs_data;

        match (a.ndim(), b.ndim()) {
            // Dot product: [k] · [k] → scalar
            (1, 1) => {
                let g = grad_output.item();
                smallvec![b * g, a * g]
            }
            // Matrix-vector: [m,k] × [k] → [m]
            (2, 1) => {
                // ∂L/∂A = grad[:,None] @ b[None,:] (outer product) → [m,k]
                let grad_a = grad_output.unsqueeze(1).matmul(&b.unsqueeze(0));
                // ∂L/∂b = A.T @ grad → [k]
                let grad_b = a.t().matmul(grad_output);
                smallvec![grad_a, grad_b]
            }
            // Vector-matrix: [k] × [k,n] → [n]
            (1, 2) => {
                // ∂L/∂a = grad @ B.T → [k]
                let grad_a = grad_output.matmul(&b.t());
                // ∂L/∂B = a[:,None] @ grad[None,:] (outer product) → [k,n]
                let grad_b = a.unsqueeze(1).matmul(&grad_output.unsqueeze(0));
                smallvec![grad_a, grad_b]
            }
            // Standard 2D: [m,k] × [k,n] → [m,n]
            (2, 2) => {
                let grad_a = grad_output.matmul(&b.t());
                let grad_b = a.t().matmul(grad_output);
                smallvec![grad_a, grad_b]
            }
            // Higher dimensions: treat as batch matmul (simplified)
            _ => {
                let grad_a = grad_output.matmul(&b.t());
                let grad_b = a.t().matmul(grad_output);
                smallvec![grad_a, grad_b]
            }
        }
    }
}

// ReLU: y = max(0, x)

pub struct ReluGrad {
    pub input: Variable,
    pub input_data: Tensor,
}

impl GradFn for ReluGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output * (x > 0)
    /// The gradient passes through where x > 0 and is zeroed where x <= 0.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let mask =
            Tensor::from_ndarray(
                self.input_data
                    .data()
                    .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            );
        smallvec![grad_output * &mask]
    }
}

// Sigmoid: y = σ(x) = 1 / (1 + exp(-x))

pub struct SigmoidGrad {
    pub input: Variable,
    /// Cached forward output σ(x), reused to avoid recomputation.
    pub output_data: Tensor,
}

impl GradFn for SigmoidGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output * σ(x) * (1 - σ(x))
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let one = Tensor::ones(self.output_data.shape());
        let one_minus_sig = &one - &self.output_data;
        let local_grad = &self.output_data * &one_minus_sig;
        smallvec![grad_output * &local_grad]
    }
}

// Tanh: y = tanh(x)

pub struct TanhGrad {
    pub input: Variable,
    /// Cached forward output tanh(x).
    pub output_data: Tensor,
}

impl GradFn for TanhGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output * (1 - tanh²(x))
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let one = Tensor::ones(self.output_data.shape());
        let tanh_sq = &self.output_data * &self.output_data;
        let local_grad = &one - &tanh_sq;
        smallvec![grad_output * &local_grad]
    }
}

// Exp: y = exp(x)

pub struct ExpGrad {
    pub input: Variable,
    /// Cached forward output exp(x).
    pub output_data: Tensor,
}

impl GradFn for ExpGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output * exp(x)
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        smallvec![grad_output * &self.output_data]
    }
}

// Log: y = ln(x)

pub struct LogGrad {
    pub input: Variable,
    pub input_data: Tensor,
}

impl GradFn for LogGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output / x
    ///
    /// Numerical safety: clamps input away from zero to prevent inf gradients.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        // Clamp input_data away from zero to prevent division by zero in gradient
        let safe_input = Tensor::from_ndarray(self.input_data.data().mapv(|x| {
            if x.abs() < 1e-7 {
                1e-7_f32.copysign(x)
            } else {
                x
            }
        }));
        smallvec![grad_output / &safe_input]
    }
}

// Pow: y = x^p

pub struct PowGrad {
    pub input: Variable,
    pub input_data: Tensor,
    pub exponent: f32,
}

impl GradFn for PowGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output * p * x^(p-1)
    ///
    /// Numerical safety: debug-asserts finite gradient output.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let p = self.exponent;
        let local_grad = &self.input_data.pow(p - 1.0) * p;
        let result = grad_output * &local_grad;
        debug_assert!(
            result.data().iter().all(|v| v.is_finite()),
            "PowGrad: non-finite gradient detected (exponent={})",
            p
        );
        smallvec![result]
    }
}

// Sqrt: y = √x

pub struct SqrtGrad {
    pub input: Variable,
    /// Cached forward output √x.
    pub output_data: Tensor,
}

impl GradFn for SqrtGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output / (2√x)
    ///
    /// Numerical safety: clamps √x away from zero to prevent inf gradients.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        // Clamp output away from zero to prevent division by zero
        let safe_output = Tensor::from_ndarray(self.output_data.data().mapv(|x| x.max(1e-7)));
        let two_sqrt = &safe_output * 2.0;
        smallvec![grad_output / &two_sqrt]
    }
}

// Sum: y = Σ x_i (reduces to scalar)

pub struct SumGrad {
    pub input: Variable,
}

impl GradFn for SumGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x_i = grad_output (broadcast to input shape)
    /// Since sum reduces to scalar, gradient is uniform across all elements.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let shape = self.input.shape();
        let grad = Tensor::full(&shape, grad_output.item());
        smallvec![grad]
    }
}

// SumAxis: y = sum(x, axis)

pub struct SumAxisGrad {
    pub input: Variable,
    pub axis: usize,
    pub keepdim: bool,
}

impl GradFn for SumAxisGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output broadcast back to input shape along the summed axis.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let input_shape = self.input.shape();

        // If keepdim=false, re-insert the reduced dimension
        let grad_expanded = if self.keepdim {
            grad_output.clone()
        } else {
            grad_output.unsqueeze(self.axis)
        };

        // Broadcast: multiply ones_like(input) * grad_expanded
        let ones = Tensor::ones(&input_shape);
        smallvec![&ones * &grad_expanded]
    }
}

// Mean: y = mean(x) = sum(x) / numel

pub struct MeanGrad {
    pub input: Variable,
}

impl GradFn for MeanGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x_i = grad_output / numel
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let shape = self.input.shape();
        let numel: usize = shape.iter().product();
        let grad = Tensor::full(&shape, grad_output.item() / numel as f32);
        smallvec![grad]
    }
}

// Scalar operations: y = x + scalar, y = x * scalar

/// Gradient for y = x + scalar (constant addition doesn't change gradient).
pub struct ScalarAddGrad {
    pub input: Variable,
}

impl GradFn for ScalarAddGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output (adding a constant has derivative 1)
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        smallvec![grad_output.clone()]
    }
}

/// Gradient for y = x * scalar.
pub struct ScalarMulGrad {
    pub input: Variable,
    pub scalar: f32,
}

impl GradFn for ScalarMulGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = grad_output * scalar
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        smallvec![grad_output * self.scalar]
    }
}

// Transpose: y = x^T (swap last two dimensions)

/// Gradient for matrix transpose.
///
/// Since transpose is its own inverse, the gradient is simply
/// the transpose of the output gradient.
pub struct TransposeGrad {
    pub input: Variable,
}

impl GradFn for TransposeGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂x = transpose(grad_output)
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        smallvec![grad_output.t()]
    }
}

// Gather along axis: y[i] = x[i, indices[i]]

/// Gradient for gather operation (used in DQN for Q-value extraction).
///
/// Forward: `output[i] = input[i, indices[i]]` for axis=1
/// Backward: scatter-add the output gradient back to the gathered positions.
///
/// ```text
/// grad_input = zeros_like(input)
/// grad_input[i, indices[i]] += grad_output[i]
/// ```
pub struct GatherAxisGrad {
    pub input: Variable,
    /// The indices used during the forward gather.
    pub indices: Vec<usize>,
    /// Original input shape `[B, A]` for scatter_add target.
    pub input_shape: Vec<usize>,
    pub axis: usize,
}

impl GradFn for GatherAxisGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone()]
    }

    /// ∂L/∂input[i, indices[i]] = grad_output[i]
    /// All other positions receive zero gradient.
    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        // grad_output shape: [B, 1] → flatten to [B] for scatter_add values
        let grad_flat = grad_output.flatten();
        let scattered =
            Tensor::scatter_add(&self.input_shape, self.axis, &self.indices, &grad_flat)
                .expect("scatter_add failed in GatherAxisGrad::backward");
        smallvec![scattered]
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_reduce_grad_same_shape() {
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = reduce_grad_for_broadcast(&grad, &[2, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reduce_grad_broadcast_row() {
        // grad [2,3] → original shape [3] (broadcast along axis 0)
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = reduce_grad_for_broadcast(&grad, &[3]);
        assert_eq!(result.shape(), &[3]);
        // sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_grad_broadcast_col() {
        // grad [2,3] → original shape [2,1] (broadcast along axis 1)
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = reduce_grad_for_broadcast(&grad, &[2, 1]);
        assert_eq!(result.shape(), &[2, 1]);
        let data = result.to_vec();
        assert_abs_diff_eq!(data[0], 6.0, epsilon = 1e-6); // 1+2+3
        assert_abs_diff_eq!(data[1], 15.0, epsilon = 1e-6); // 4+5+6
    }

    #[test]
    fn test_reduce_grad_broadcast_scalar() {
        // grad [2,3] → original shape [1,1]
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = reduce_grad_for_broadcast(&grad, &[1, 1]);
        assert_eq!(result.shape(), &[1, 1]);
        assert_abs_diff_eq!(result.to_vec()[0], 21.0, epsilon = 1e-6);
    }
}
