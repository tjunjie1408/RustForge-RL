//! Tensor error type definitions.
//!
//! This module defines all error types that might occur during tensor operations.
//! It uses Rust's enum types to implement type-safe error handling.

use std::fmt;

/// Error types in tensor operations.
///
/// Each variant represents a specific error condition and contains context information
/// about the error.
#[derive(Debug, Clone)]
pub enum TensorError {
    /// Shape mismatch error.
    ///
    /// Thrown when the shapes of two tensors do not meet the operation requirements.
    /// For example, matrix multiplication requires A's columns to equal B's rows.
    ShapeMismatch {
        op: String,
        left: Vec<usize>,
        right: Vec<usize>,
    },

    /// Invalid shape error.
    ///
    /// Thrown when the target shape of a reshape operation does not match the total
    /// number of elements in the tensor.
    InvalidShape {
        expected_elements: usize,
        got_elements: usize,
        shape: Vec<usize>,
    },

    /// Axis index out of bounds error.
    AxisOutOfBounds {
        axis: usize,
        ndim: usize,
    },

    /// Empty tensor error.
    EmptyTensor,

    /// Data length does not match shape.
    DataShapeMismatch {
        data_len: usize,
        shape_elements: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { op, left, right } => {
                write!(
                    f,
                    "Shape mismatch for operation '{}': left={:?}, right={:?}",
                    op, left, right
                )
            }
            TensorError::InvalidShape {
                expected_elements,
                got_elements,
                shape,
            } => {
                write!(
                    f,
                    "Invalid reshape: tensor has {} elements, but target shape {:?} requires {} elements",
                    expected_elements, shape, got_elements
                )
            }
            TensorError::AxisOutOfBounds { axis, ndim } => {
                write!(
                    f,
                    "Axis {} is out of bounds for tensor with {} dimensions",
                    axis, ndim
                )
            }
            TensorError::EmptyTensor => {
                write!(f, "Operation not supported on empty tensor")
            }
            TensorError::DataShapeMismatch {
                data_len,
                shape_elements,
            } => {
                write!(
                    f,
                    "Data length {} does not match shape product {}",
                    data_len, shape_elements
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Result type alias for tensor operations.
pub type TensorResult<T> = Result<T, TensorError>;
