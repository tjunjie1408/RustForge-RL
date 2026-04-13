//! Tensor display formatting.
//!
//! Provides human-readable tensor string formatting, similar to PyTorch and NumPy
//! output styles. Automatically handles truncation for large tensors.

use std::fmt;

use crate::tensor::Tensor;

/// Maximum number of elements to display per axis (exceeding this will truncate the middle part)
const EDGE_ITEMS: usize = 3;
/// Threshold to trigger truncated display
const THRESHOLD: usize = 1000;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape();
        let data = self.to_vec();

        write!(f, "Tensor(")?;

        if self.is_scalar() {
            write!(f, "{:.4}", data[0])?;
        } else if data.len() <= THRESHOLD {
            format_recursive(f, &data, shape, 0, 0)?;
        } else {
            write!(f, "[...{} elements...]", data.len())?;
        }

        write!(f, ", shape={:?})", shape)
    }
}

/// Recursively format multi-dimensional tensors.
fn format_recursive(
    f: &mut fmt::Formatter<'_>,
    data: &[f32],
    shape: &[usize],
    dim: usize,
    offset: usize,
) -> fmt::Result {
    if dim == shape.len() - 1 {
        // Innermost dimension: print numbers directly
        write!(f, "[")?;
        let n = shape[dim];
        if n <= EDGE_ITEMS * 2 {
            for i in 0..n {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:>8.4}", data[offset + i])?;
            }
        } else {
            for i in 0..EDGE_ITEMS {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:>8.4}", data[offset + i])?;
            }
            write!(f, ", ..., ")?;
            for i in (n - EDGE_ITEMS)..n {
                if i > n - EDGE_ITEMS {
                    write!(f, ", ")?;
                }
                write!(f, "{:>8.4}", data[offset + i])?;
            }
        }
        write!(f, "]")
    } else {
        // Outer dimensions: recurse
        let stride: usize = shape[dim + 1..].iter().product();
        let n = shape[dim];
        let indent = "  ".repeat(dim + 1);

        write!(f, "[")?;
        if n <= EDGE_ITEMS * 2 {
            for i in 0..n {
                if i > 0 {
                    write!(f, ",\n{}", indent)?;
                }
                format_recursive(f, data, shape, dim + 1, offset + i * stride)?;
            }
        } else {
            for i in 0..EDGE_ITEMS {
                if i > 0 {
                    write!(f, ",\n{}", indent)?;
                }
                format_recursive(f, data, shape, dim + 1, offset + i * stride)?;
            }
            write!(f, ",\n{}...", indent)?;
            for i in (n - EDGE_ITEMS)..n {
                write!(f, ",\n{}", indent)?;
                format_recursive(f, data, shape, dim + 1, offset + i * stride)?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_display_1d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let s = format!("{}", t);
        assert!(s.contains("Tensor("));
        assert!(s.contains("shape=[3]"));
    }

    #[test]
    fn test_display_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = format!("{}", t);
        assert!(s.contains("shape=[2, 2]"));
    }

    #[test]
    fn test_display_scalar() {
        let t = Tensor::scalar(3.14);
        let s = format!("{}", t);
        assert!(s.contains("3.14"));
        assert!(s.contains("shape=[]"));
    }
}
