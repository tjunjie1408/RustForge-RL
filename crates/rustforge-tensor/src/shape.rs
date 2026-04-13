//! Shape utility functions.
//!
//! This module provides utility functions for shape inference, broadcasting rules, etc.
//! Broadcasting is a NumPy-style shape alignment mechanism that allows
//! tensors of different shapes to be used in element-wise operations.
//!
//! ## Broadcasting Rules
//!
//! Two tensors can be broadcast if (comparing axes from right to left):
//! 1. The dimensions are equal, or
//! 2. One of the dimensions is 1
//!
//! Examples:
//! - `[3, 4]` and `[4]` → broadcast to `[3, 4]`
//! - `[2, 1, 5]` and `[3, 5]` → broadcast to `[2, 3, 5]`
//! - `[3, 4]` and `[3, 5]` → **Incompatible**

use crate::error::{TensorError, TensorResult};

/// Calculates the resulting shape after broadcasting two shapes.
///
/// ## Algorithm
/// 1. Align the two shapes from the right.
/// 2. Compare dimension by dimension:
///    - Equal → Keep it
///    - One is 1 → Take the larger one
///    - Otherwise → Return error
///
/// ## Arguments
/// - `shape_a`: Shape of the first tensor
/// - `shape_b`: Shape of the second tensor
///
/// ## Returns
/// The broadcasted shape, or an error if shapes are incompatible.
///
/// ## Example
/// ```rust
/// use rustforge_tensor::shape::broadcast_shape;
/// let result = broadcast_shape(&[3, 1, 5], &[4, 5]).unwrap();
/// assert_eq!(result, vec![3, 4, 5]);
/// ```
pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> TensorResult<Vec<usize>> {
    let max_ndim = shape_a.len().max(shape_b.len());
    let mut result = Vec::with_capacity(max_ndim);

    // Align and compare from right to left
    for i in 0..max_ndim {
        // Read from right, assume missing dimensions are 1
        let dim_a = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };
        let dim_b = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b {
            result.push(dim_a);
        } else if dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            return Err(TensorError::ShapeMismatch {
                op: "broadcast".to_string(),
                left: shape_a.to_vec(),
                right: shape_b.to_vec(),
            });
        }
    }

    // Results were built from right to left, so reverse them
    result.reverse();
    Ok(result)
}

/// Checks if shapes for matrix multiplication are compatible and returns the resulting shape.
///
/// For 2D matrix multiplication: `[m, k] × [k, n] → [m, n]`
///
/// Also supports batch matrix multiplication: `[..., m, k] × [..., k, n] → [..., m, n]`
/// where the `...` part follows broadcasting rules.
pub fn matmul_shape(shape_a: &[usize], shape_b: &[usize]) -> TensorResult<Vec<usize>> {
    if shape_a.len() < 2 || shape_b.len() < 2 {
        // For 1D vectors, try to interpret as matrix multiplication
        if shape_a.len() == 1 && shape_b.len() == 1 {
            // Dot product: [k] · [k] → scalar
            if shape_a[0] != shape_b[0] {
                return Err(TensorError::ShapeMismatch {
                    op: "dot".to_string(),
                    left: shape_a.to_vec(),
                    right: shape_b.to_vec(),
                });
            }
            return Ok(vec![1]);
        }
        if shape_a.len() == 1 && shape_b.len() == 2 {
            // [k] × [k, n] → [n]
            if shape_a[0] != shape_b[0] {
                return Err(TensorError::ShapeMismatch {
                    op: "matmul".to_string(),
                    left: shape_a.to_vec(),
                    right: shape_b.to_vec(),
                });
            }
            return Ok(vec![shape_b[1]]);
        }
        if shape_a.len() == 2 && shape_b.len() == 1 {
            // [m, k] × [k] → [m]
            if shape_a[1] != shape_b[0] {
                return Err(TensorError::ShapeMismatch {
                    op: "matmul".to_string(),
                    left: shape_a.to_vec(),
                    right: shape_b.to_vec(),
                });
            }
            return Ok(vec![shape_a[0]]);
        }
    }

    // 2D and above matrix multiplication
    let m = shape_a[shape_a.len() - 2];
    let k1 = shape_a[shape_a.len() - 1];
    let k2 = shape_b[shape_b.len() - 2];
    let n = shape_b[shape_b.len() - 1];

    if k1 != k2 {
        return Err(TensorError::ShapeMismatch {
            op: "matmul".to_string(),
            left: shape_a.to_vec(),
            right: shape_b.to_vec(),
        });
    }

    // Broadcast batch dimensions
    if shape_a.len() > 2 || shape_b.len() > 2 {
        let batch_a = &shape_a[..shape_a.len() - 2];
        let batch_b = &shape_b[..shape_b.len() - 2];
        let mut batch_shape = broadcast_shape(batch_a, batch_b)?;
        batch_shape.push(m);
        batch_shape.push(n);
        Ok(batch_shape)
    } else {
        Ok(vec![m, n])
    }
}

/// Calculates the total number of elements for a given shape.
///
/// This is the product of all dimension sizes. An empty shape (scalar) returns 1.
pub fn shape_numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

/// Computes the default strides for a given shape (row-major / C-order).
///
/// The stride represents the number of elements to skip to move one step along a given axis.
/// In a row-major layout, the stride of the last dimension is 1.
///
/// Example: The stride for shape `[2, 3, 4]` is `[12, 4, 1]`
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_same_shape() {
        let result = broadcast_shape(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_scalar() {
        let result = broadcast_shape(&[3, 4], &[1]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_different_ndim() {
        let result = broadcast_shape(&[3, 1, 5], &[4, 5]).unwrap();
        assert_eq!(result, vec![3, 4, 5]);
    }

    #[test]
    fn test_broadcast_row_col() {
        let result = broadcast_shape(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        let result = broadcast_shape(&[3, 4], &[3, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_shape_2d() {
        let result = matmul_shape(&[2, 3], &[3, 4]).unwrap();
        assert_eq!(result, vec![2, 4]);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let result = matmul_shape(&[2, 3], &[4, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_numel() {
        assert_eq!(shape_numel(&[2, 3, 4]), 24);
        assert_eq!(shape_numel(&[]), 1);
        assert_eq!(shape_numel(&[5]), 5);
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[3, 4]), vec![4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
    }
}
