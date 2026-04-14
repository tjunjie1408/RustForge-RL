//! Tensor arithmetic operations.
//!
//! This module implements addition, subtraction, multiplication, division, and matrix
//! multiplication for `Tensor` by overloading Rust's operators (the `std::ops` trait).
//! It supports both tensor-tensor and tensor-scalar operations.
//!
//! ## Broadcasting Semantics
//!
//! All element-wise operations follow NumPy-style broadcasting rules. For example:
//! - `[3, 4] + [4]` → The vector `[4]` is broadcasted to every row.
//! - `[3, 1] * [1, 4]` → The resulting shape is `[3, 4]`.
//!
//! ## Ownership Design
//!
//! Operators are implemented for both `&Tensor` and `Tensor`, allowing flexible usage:
//! ```rust,ignore
//! let c = &a + &b;   // ref + ref
//! let d = a + &b;    // val + ref (a is consumed)
//! let e = &a + b;    // ref + val (b is consumed)
//! ```

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::tensor::Tensor;

// Tensor + Tensor (Element-wise addition with broadcasting)

impl<'b> Add<&'b Tensor> for &Tensor {
    type Output = Tensor;

    /// Element-wise addition, supports broadcasting.
    ///
    /// ## Mathematical Definition
    /// C[i,j,...] = A[i,j,...] + B[i,j,...]
    ///
    /// If shapes differ, they are aligned according to broadcasting rules before operation.
    fn add(self, rhs: &'b Tensor) -> Tensor {
        // ndarray's + operator inherently supports broadcasting
        let result = self.data() + rhs.data();
        Tensor::from_ndarray(result)
    }
}

// Tensor - Tensor (Element-wise subtraction with broadcasting)

impl<'b> Sub<&'b Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &'b Tensor) -> Tensor {
        let result = self.data() - rhs.data();
        Tensor::from_ndarray(result)
    }
}

// Tensor * Tensor (Element-wise multiplication / Hadamard product with broadcasting)

impl<'b> Mul<&'b Tensor> for &Tensor {
    type Output = Tensor;

    /// Element-wise multiplication (Hadamard product), NOT matrix multiplication.
    ///
    /// For matrix multiplication, please use `Tensor::matmul()`.
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        let result = self.data() * rhs.data();
        Tensor::from_ndarray(result)
    }
}

// Tensor / Tensor (Element-wise division with broadcasting)

impl<'b> Div<&'b Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &'b Tensor) -> Tensor {
        let result = self.data() / rhs.data();
        Tensor::from_ndarray(result)
    }
}

// Unary negation -Tensor

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.neg()
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    #[allow(clippy::needless_borrow)]
    fn neg(self) -> Tensor {
        (&self).neg()
    }
}

// Tensor + Scalar / Scalar + Tensor

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Tensor {
        Tensor::from_ndarray(self.data() + rhs)
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        Tensor::from_ndarray(self + rhs.data())
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Tensor {
        Tensor::from_ndarray(self.data() - rhs)
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {
        Tensor::from_ndarray(self.data() * rhs)
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        Tensor::from_ndarray(self * rhs.data())
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Tensor {
        Tensor::from_ndarray(self.data() / rhs)
    }
}

// Operator overloading for value types (Ownership consuming versions)
// These implementations allow users to avoid writing `&` references every time

impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        &self + rhs
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        self + &rhs
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        &self + &rhs
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        &self - rhs
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self - &rhs
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        &self - &rhs
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        &self * rhs
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        self * &rhs
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        &self * &rhs
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        &self / rhs
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self / &rhs
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        &self / &rhs
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Tensor {
        &self + rhs
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Tensor {
        &self - rhs
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        &self * rhs
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f32) -> Tensor {
        &self / rhs
    }
}

// Matrix Multiplication (Not through operator overloading, uses method call)

impl Tensor {
    /// Matrix multiplication.
    ///
    /// ## Mathematical Definition
    /// For 2D tensors: C = A × B, where C[i,j] = Σ_k A[i,k] * B[k,j]
    ///
    /// ## Supported Shape Combinations
    /// - `[m, k] × [k, n]` → `[m, n]` — Standard matrix multiplication
    /// - `[k] · [k]` → `[1]` — Vector dot product
    /// - `[m, k] × [k]` → `[m]` — Matrix-vector multiplication
    /// - `[k] × [k, n]` → `[n]` — Vector-matrix multiplication
    ///
    /// ## Example
    /// ```rust
    /// use rustforge_tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.shape(), &[2, 2]);
    /// ```
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let a = self;
        let b = rhs;

        match (a.ndim(), b.ndim()) {
            // Vector dot product: [k] · [k] → scalar
            (1, 1) => {
                assert_eq!(
                    a.shape()[0],
                    b.shape()[0],
                    "Dot product requires same length vectors: {} vs {}",
                    a.shape()[0],
                    b.shape()[0]
                );
                let dot: f32 = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(x, y)| x * y)
                    .sum();
                Tensor::scalar(dot)
            }

            // Matrix × Vector: [m, k] × [k] → [m]
            (2, 1) => {
                assert_eq!(
                    a.shape()[1],
                    b.shape()[0],
                    "matmul shape mismatch: [{}, {}] × [{}]",
                    a.shape()[0],
                    a.shape()[1],
                    b.shape()[0]
                );
                let a_2d = a
                    .data()
                    .clone()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let b_1d = b
                    .data()
                    .clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();
                let result = a_2d.dot(&b_1d);
                Tensor::from_ndarray(result.into_dyn())
            }

            // Vector × Matrix: [k] × [k, n] → [n]
            (1, 2) => {
                assert_eq!(
                    a.shape()[0],
                    b.shape()[0],
                    "matmul shape mismatch: [{}] × [{}, {}]",
                    a.shape()[0],
                    b.shape()[0],
                    b.shape()[1]
                );
                let a_1d = a
                    .data()
                    .clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();
                let b_2d = b
                    .data()
                    .clone()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let result = a_1d.dot(&b_2d);
                Tensor::from_ndarray(result.into_dyn())
            }

            // Standard 2D Matrix Multiplication: [m, k] × [k, n] → [m, n]
            (2, 2) => {
                assert_eq!(
                    a.shape()[1],
                    b.shape()[0],
                    "matmul shape mismatch: [{}, {}] × [{}, {}]",
                    a.shape()[0],
                    a.shape()[1],
                    b.shape()[0],
                    b.shape()[1]
                );
                let a_2d = a
                    .data()
                    .clone()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let b_2d = b
                    .data()
                    .clone()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let result = a_2d.dot(&b_2d);
                Tensor::from_ndarray(result.into_dyn())
            }

            // High dimension batch matmul: [..., m, k] × [..., k, n] → [..., m, n]
            _ => {
                // Simplified implementation: reshape to 3D, perform matmul per batch
                self.batch_matmul(rhs)
            }
        }
    }

    /// Internal implementation of batch matrix multiplication.
    ///
    /// Handles tensors higher than 2D by flattening the batch dimensions into a single
    /// batch dimension, performing 2D matrix multiplication individually, and then
    /// reshaping back to the target shape.
    fn batch_matmul(&self, rhs: &Tensor) -> Tensor {
        let a_shape = self.shape();
        let b_shape = rhs.shape();

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        assert_eq!(
            k,
            b_shape[b_shape.len() - 2],
            "batch matmul inner dimension mismatch"
        );

        // Calculate batch dimension
        let batch_a: usize = a_shape[..a_shape.len() - 2].iter().product();
        let batch_b: usize = b_shape[..b_shape.len() - 2].iter().product();
        let batch_size = batch_a.max(batch_b);

        // Reshape to [batch, m, k] and [batch, k, n]
        let a_flat = self.reshape(&[batch_a, m, k]).unwrap();
        let b_flat = rhs.reshape(&[batch_b, k, n]).unwrap();

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let a_i = a_flat.select(0, i % batch_a).unwrap();
            let b_i = b_flat.select(0, i % batch_b).unwrap();
            let ab_i = a_i.matmul(&b_i);
            results.push(ab_i);
        }

        // Concatenate results and reshape
        let refs: Vec<&Tensor> = results.iter().collect();
        let stacked = Tensor::stack(&refs, 0).unwrap();

        // Restore batch dimension shape
        let result_shape = crate::shape::matmul_shape(a_shape, b_shape)
            .expect("batch matmul shape computation failed");
        stacked.reshape(&result_shape).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = &a + &b;
        assert_eq!(c.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = &a - &b;
        assert_eq!(c.to_vec(), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_mul_elementwise() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
        let c = &a * &b;
        assert_eq!(c.to_vec(), vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_div() {
        let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let b = Tensor::from_vec(vec![2.0, 5.0, 10.0, 8.0], &[2, 2]);
        let c = &a / &b;
        assert_eq!(c.to_vec(), vec![5.0, 4.0, 3.0, 5.0]);
    }

    #[test]
    fn test_scalar_ops() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = &a + 10.0;
        assert_eq!(b.to_vec(), vec![11.0, 12.0, 13.0]);

        let c = &a * 2.0;
        assert_eq!(c.to_vec(), vec![2.0, 4.0, 6.0]);

        let d = 3.0 * &a;
        assert_eq!(d.to_vec(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3]);
        let b = -&a;
        assert_eq!(b.to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_matmul_2d() {
        // [2, 3] × [3, 2] = [2, 2]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        // c[0,0] = 1*1 + 2*2 + 3*3 = 14
        // c[0,1] = 1*4 + 2*5 + 3*6 = 32
        // c[1,0] = 4*1 + 5*2 + 6*3 = 32
        // c[1,1] = 4*4 + 5*5 + 6*6 = 77
        let data = c.to_vec();
        assert_abs_diff_eq!(data[0], 14.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[1], 32.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[2], 32.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[3], 77.0, epsilon = 1e-6);
    }

    #[test]
    fn test_matmul_vec_dot() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
        let c = a.matmul(&b);
        // 1*4 + 2*5 + 3*6 = 32
        assert_abs_diff_eq!(c.item(), 32.0, epsilon = 1e-6);
    }

    #[test]
    fn test_matmul_mat_vec() {
        // [2, 3] × [3] = [2]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let eye = Tensor::eye(2);
        let result = a.matmul(&eye);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ownership_variants() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2]);

        // ref + ref
        let _c = &a + &b;

        // val + ref
        let a2 = a.clone();
        let _d = a2 + &b;

        // ref + val
        let b2 = b.clone();
        let _e = &a + b2;

        // val + val
        let a3 = a.clone();
        let b3 = b.clone();
        let _f = a3 + b3;
    }
}
