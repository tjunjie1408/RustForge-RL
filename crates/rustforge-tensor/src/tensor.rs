//! Core tensor struct definition.
//!
//! `Tensor` is the foundational data structure of RustForge, encapsulating `ndarray::ArrayD<f32>`,
//! and providing core functionalities such as creation, indexing, and reshaping.
//!
//! ## Design Decisions
//!
//! - **Using `f32` instead of generics**: f32 precision is sufficient for RL scenarios, avoiding
//!   the complexity introduced by generics. Can support f64 via feature flags in the future.
//! - **Encapsulating `ndarray`**: Doesn't expose the ndarray type directly, preserving the flexibility
//!   to switch backends in the future.
//! - **Clone as value semantics**: Tensor implements Clone, operations produce new tensors without
//!   modifying the original data. The autograd layer will add reference counting and gradient tracking
//!   on top of this.

use ndarray::{ArrayD, IxDyn, Axis};
use serde::{Serialize, Deserialize};

use crate::error::{TensorError, TensorResult};
use crate::shape;

/// N-dimensional floating-point tensor.
///
/// Internal storage is based on `ndarray::ArrayD<f32>` (dynamic dimension array),
/// supporting numerical operations for arbitrary dimensions.
///
/// ## Example
/// ```rust
/// use rustforge_tensor::Tensor;
///
/// // Create a 2x3 matrix from a Vec
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.numel(), 6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Underlying ndarray dynamic dimension array
    data: ArrayD<f32>,
}

impl Tensor {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Creates a tensor directly from an internal ndarray array.
    ///
    /// This is a low-level API typically used internally within the crate or for
    /// interoperability with ndarray.
    pub fn from_ndarray(data: ArrayD<f32>) -> Self {
        Tensor { data }
    }

    /// Creates a tensor from a flattened data vector and a specified shape.
    ///
    /// ## Arguments
    /// - `data`: Element data in row-major order.
    /// - `shape`: Target shape.
    ///
    /// ## Panics
    /// If `data.len()` does not equal the product of the dimensions in `shape`.
    ///
    /// ## Example
    /// ```rust
    /// use rustforge_tensor::Tensor;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// ```
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let expected = shape::shape_numel(shape);
        assert_eq!(
            data.len(),
            expected,
            "Data length {} does not match shape {:?} (expected {} elements)",
            data.len(),
            shape,
            expected
        );
        let data = ArrayD::from_shape_vec(IxDyn(shape), data)
            .expect("Failed to create ndarray from shape and data");
        Tensor { data }
    }

    /// Creates a tensor of all zeros with the specified shape.
    ///
    /// ## Example
    /// ```rust
    /// use rustforge_tensor::Tensor;
    /// let t = Tensor::zeros(&[3, 4]);
    /// assert_eq!(t.shape(), &[3, 4]);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Creates a tensor of all ones with the specified shape.
    pub fn ones(shape: &[usize]) -> Self {
        Tensor {
            data: ArrayD::ones(IxDyn(shape)),
        }
    }

    /// Creates a tensor filled with a specified value.
    pub fn full(shape: &[usize], value: f32) -> Self {
        Tensor {
            data: ArrayD::from_elem(IxDyn(shape), value),
        }
    }

    /// Creates a scalar tensor (0-dimensional).
    pub fn scalar(value: f32) -> Self {
        Tensor {
            data: ArrayD::from_elem(IxDyn(&[]), value),
        }
    }

    /// Creates an identity matrix (square matrix).
    ///
    /// ## Arguments
    /// - `n`: Matrix dimension (n x n).
    pub fn eye(n: usize) -> Self {
        let mut data = ArrayD::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            data[[i, i]] = 1.0;
        }
        Tensor { data }
    }

    /// Creates a tensor with evenly spaced values within a given interval (similar to numpy.arange).
    ///
    /// Starts from `start` with a step size of `step`, excluding `end`.
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut values = Vec::new();
        let mut v = start;
        if step > 0.0 {
            while v < end {
                values.push(v);
                v += step;
            }
        } else if step < 0.0 {
            while v > end {
                values.push(v);
                v += step;
            }
        }
        let len = values.len();
        Tensor::from_vec(values, &[len])
    }

    /// Creates a tensor with evenly spaced numbers over a specified interval (similar to numpy.linspace).
    ///
    /// `num` evenly spaced samples, calculated over the interval [`start`, `end`].
    pub fn linspace(start: f32, end: f32, num: usize) -> Self {
        if num == 0 {
            return Tensor::from_vec(vec![], &[0]);
        }
        if num == 1 {
            return Tensor::from_vec(vec![start], &[1]);
        }
        let step = (end - start) / (num - 1) as f32;
        let values: Vec<f32> = (0..num).map(|i| start + step * i as f32).collect();
        Tensor::from_vec(values, &[num])
    }

    // ========================================================================
    // Property Queries
    // ========================================================================

    /// Returns the shape of the tensor (sizes of dimensions).
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the number of dimensions of the tensor (ndim).
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the tensor is a scalar (0-dimensional).
    pub fn is_scalar(&self) -> bool {
        self.data.ndim() == 0
    }

    /// Returns whether the tensor is empty (contains no elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets an immutable reference to the underlying ndarray.
    ///
    /// Intended for efficient internal crate operations; external use is discouraged.
    pub fn data(&self) -> &ArrayD<f32> {
        &self.data
    }

    /// Gets a mutable reference to the underlying ndarray.
    pub fn data_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }

    /// Exports the tensor data as a flattened Vec<f32>.
    ///
    /// Copies zero data if the underlying storage is contiguous; otherwise, cloning is necessary.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().cloned().collect()
    }

    /// Gets the scalar value (only for scalars or single-element tensors).
    ///
    /// ## Panics
    /// If the tensor contains more than one element.
    pub fn item(&self) -> f32 {
        assert!(
            self.numel() == 1,
            "item() requires a single-element tensor, got {} elements",
            self.numel()
        );
        *self.data.iter().next().unwrap()
    }

    // ========================================================================
    // Reshaping & Transformations
    // ========================================================================

    /// Changes the tensor shape (without modifying data).
    ///
    /// The total number of elements in the new shape must equal the original tensor.
    /// Ensures contiguous memory allocation when necessary.
    ///
    /// ## Example
    /// ```rust
    /// use rustforge_tensor::Tensor;
    /// let t = Tensor::arange(0.0, 6.0, 1.0);
    /// let t = t.reshape(&[2, 3]).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> TensorResult<Tensor> {
        let new_numel = shape::shape_numel(new_shape);
        if self.numel() != new_numel {
            return Err(TensorError::InvalidShape {
                expected_elements: self.numel(),
                got_elements: new_numel,
                shape: new_shape.to_vec(),
            });
        }
        // Ensure contiguous storage before reshape
        let contiguous = self.data.as_standard_layout().into_owned();
        let reshaped = contiguous
            .into_shape_with_order(IxDyn(new_shape))
            .map_err(|_| TensorError::InvalidShape {
                expected_elements: self.numel(),
                got_elements: new_numel,
                shape: new_shape.to_vec(),
            })?;
        Ok(Tensor::from_ndarray(reshaped))
    }

    /// Flattens the tensor into one dimension.
    pub fn flatten(&self) -> Tensor {
        self.reshape(&[self.numel()]).unwrap()
    }

    /// Transpose matrix (for 2D tensors).
    ///
    /// For higher-dimensional tensors, swaps the last two dimensions.
    pub fn t(&self) -> Tensor {
        if self.ndim() < 2 {
            return self.clone();
        }
        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        let n = axes.len();
        axes.swap(n - 2, n - 1);
        self.permute(&axes)
    }

    /// Permutes the tensor dimensions according to the specified axis order.
    ///
    /// ## Arguments
    /// - `axes`: The new axis order. E.g. `[2, 0, 1]` means the original 2nd axis becomes the 0th,
    ///   the original 0th axis becomes the 1st, and so on.
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        let permuted = self.data.clone().permuted_axes(IxDyn(axes));
        // Ensure memory is contiguous
        Tensor::from_ndarray(permuted.as_standard_layout().to_owned())
    }

    /// Inserts a new dimension of size 1 at the specified position.
    ///
    /// Similar to PyTorch's `unsqueeze` or NumPy's `expand_dims`.
    ///
    /// ## Example
    /// ```rust
    /// use rustforge_tensor::Tensor;
    /// let t = Tensor::zeros(&[3, 4]);
    /// let t = t.unsqueeze(0);
    /// assert_eq!(t.shape(), &[1, 3, 4]);
    /// ```
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        let mut new_shape: Vec<usize> = self.shape().to_vec();
        new_shape.insert(axis, 1);
        self.reshape(&new_shape).unwrap()
    }

    /// Removes a specified dimension of size 1.
    ///
    /// If no axis is specified, removes all dimensions of size 1.
    pub fn squeeze(&self, axis: Option<usize>) -> Tensor {
        match axis {
            Some(ax) => {
                assert_eq!(self.shape()[ax], 1, "Cannot squeeze axis {} with size {}", ax, self.shape()[ax]);
                let mut new_shape: Vec<usize> = self.shape().to_vec();
                new_shape.remove(ax);
                if new_shape.is_empty() {
                    // If squeezing all dimensions, it becomes a scalar
                    Tensor::scalar(self.item())
                } else {
                    self.reshape(&new_shape).unwrap()
                }
            }
            None => {
                let new_shape: Vec<usize> = self.shape().iter().filter(|&&d| d != 1).cloned().collect();
                if new_shape.is_empty() {
                    Tensor::scalar(self.item())
                } else {
                    self.reshape(&new_shape).unwrap()
                }
            }
        }
    }

    // ========================================================================
    // Indexing and Slicing
    // ========================================================================

    /// Selects a specific index along a given axis, returning a sliced sub-tensor with reduced dimensions.
    ///
    /// Similar to NumPy's `a[3]` (along axis 0) or `np.take(a, 3, axis=1)`.
    pub fn select(&self, axis: usize, index: usize) -> TensorResult<Tensor> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        let view = self.data.index_axis(Axis(axis), index);
        Ok(Tensor::from_ndarray(view.to_owned()))
    }

    /// Slices along the specified axis (similar to Python's `a[start:end]`).
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> TensorResult<Tensor> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        let sliced = self.data.slice_axis(
            Axis(axis),
            ndarray::Slice::from(start..end),
        );
        Ok(Tensor::from_ndarray(sliced.to_owned()))
    }

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    /// Calculates the sum of all elements.
    pub fn sum(&self) -> Tensor {
        Tensor::scalar(self.data.sum())
    }

    /// Calculates the sum along a specified axis.
    ///
    /// ## Arguments
    /// - `axis`: The axis to sum over.
    /// - `keepdim`: Whether the reduced dimensions are retained with length 1.
    pub fn sum_axis(&self, axis: usize, keepdim: bool) -> TensorResult<Tensor> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        let summed = self.data.sum_axis(Axis(axis));
        if keepdim {
            let mut shape: Vec<usize> = self.shape().to_vec();
            shape[axis] = 1;
            Ok(Tensor::from_ndarray(
                summed.into_shape_with_order(IxDyn(&shape)).unwrap(),
            ))
        } else {
            Ok(Tensor::from_ndarray(summed))
        }
    }

    /// Calculates the mean of all elements.
    pub fn mean(&self) -> Tensor {
        Tensor::scalar(self.data.mean().unwrap_or(0.0))
    }

    /// Calculates the mean along a specified axis.
    pub fn mean_axis(&self, axis: usize, keepdim: bool) -> TensorResult<Tensor> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        let meaned = self.data.mean_axis(Axis(axis)).unwrap();
        if keepdim {
            let mut shape: Vec<usize> = self.shape().to_vec();
            shape[axis] = 1;
            Ok(Tensor::from_ndarray(
                meaned.into_shape_with_order(IxDyn(&shape)).unwrap(),
            ))
        } else {
            Ok(Tensor::from_ndarray(meaned))
        }
    }

    /// Returns the maximum value of all elements.
    pub fn max(&self) -> TensorResult<Tensor> {
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_val.is_infinite() && max_val.is_sign_negative() {
            return Err(TensorError::EmptyTensor);
        }
        Ok(Tensor::scalar(max_val))
    }

    /// Calculates the maximum values along a specified axis.
    pub fn max_axis(&self, axis: usize, keepdim: bool) -> TensorResult<Tensor> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }
        let result = self.data.map_axis(Axis(axis), |lane| {
            lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        });
        if keepdim {
            let mut shape: Vec<usize> = self.shape().to_vec();
            shape[axis] = 1;
            Ok(Tensor::from_ndarray(
                result.into_shape_with_order(IxDyn(&shape)).unwrap(),
            ))
        } else {
            Ok(Tensor::from_ndarray(result))
        }
    }

    /// Returns the indices of the maximum values along a specified axis (argmax).
    pub fn argmax_axis(&self, axis: usize) -> TensorResult<Vec<usize>> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }

        // Find argmax along each lane of the axis
        let result = self.data.map_axis(Axis(axis), |lane| {
            lane.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        });

        Ok(result.iter().cloned().collect())
    }

    /// Calculates the variance of all elements.
    pub fn var(&self) -> Tensor {
        let mean = self.data.mean().unwrap_or(0.0);
        let var = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / (self.numel() as f32);
        Tensor::scalar(var)
    }

    /// Calculates the standard deviation of all elements.
    pub fn std_dev(&self) -> Tensor {
        Tensor::scalar(self.var().item().sqrt())
    }

    // ========================================================================
    // Element-wise Mathematical Functions
    // ========================================================================

    /// ReLU activation function: max(0, x)
    ///
    /// This is one of the most commonly used activation functions in deep learning.
    /// The gradient is 1 when x > 0 and 0 when x <= 0.
    pub fn relu(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.max(0.0)))
    }

    /// Sigmoid activation function: 1 / (1 + exp(-x))
    ///
    /// Maps the input to the (0, 1) range, commonly used for binary classification output.
    /// Note: Numerically stable version handles extreme inputs.
    pub fn sigmoid(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| {
            if x >= 0.0 {
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        }))
    }

    /// Tanh activation function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    ///
    /// Maps the input to the (-1, 1) range.
    pub fn tanh_(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.tanh()))
    }

    /// Exponential function: exp(x)
    pub fn exp(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.exp()))
    }

    /// Natural logarithm: ln(x)
    pub fn log(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.ln()))
    }

    /// Safe logarithm: ln(x + eps) to avoid log(0) producing -inf
    pub fn log_safe(&self, eps: f32) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| (x + eps).ln()))
    }

    /// Power operation: x^p
    pub fn pow(&self, p: f32) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.powf(p)))
    }

    /// Square root: sqrt(x)
    pub fn sqrt(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.sqrt()))
    }

    /// Absolute value: |x|
    pub fn abs(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.abs()))
    }

    /// Clamping: clamps all elements to be within the [min, max] range.
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| x.clamp(min, max)))
    }

    /// Negation: -x
    pub fn neg(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| -x))
    }

    /// Reciprocal: 1/x
    pub fn reciprocal(&self) -> Tensor {
        Tensor::from_ndarray(self.data.mapv(|x| 1.0 / x))
    }

    // ========================================================================
    // Advanced Operations
    // ========================================================================

    /// Softmax function (along a specified axis).
    ///
    /// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    ///
    /// Subtracting the maximum value (log-sum-exp trick) ensures numerical stability
    /// by avoiding exp() overflows.
    pub fn softmax(&self, axis: usize) -> TensorResult<Tensor> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }

        // Compute max along axis (keepdim=true) for numerical stability
        let max_vals = self.max_axis(axis, true)?;
        let shifted = self - &max_vals;
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum_axis(axis, true)?;

        Ok(&exp_vals / &sum_exp)
    }

    /// Log-Softmax function (along a specified axis).
    ///
    /// log_softmax(x_i) = x_i - max(x) - log(Σ exp(x_j - max(x)))
    ///
    /// More numerically stable than computing softmax followed by log.
    pub fn log_softmax(&self, axis: usize) -> TensorResult<Tensor> {
        if axis >= self.ndim() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                ndim: self.ndim(),
            });
        }

        let max_vals = self.max_axis(axis, true)?;
        let shifted = self - &max_vals;
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum_axis(axis, true)?;
        let log_sum_exp = sum_exp.log();

        Ok(&shifted - &log_sum_exp)
    }

    // ========================================================================
    // Concatenation and Splitting
    // ========================================================================

    /// Concatenates multiple tensors along a specified axis.
    ///
    /// Similar to NumPy's `np.concatenate` or PyTorch's `torch.cat`.
    pub fn cat(tensors: &[&Tensor], axis: usize) -> TensorResult<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let result = ndarray::concatenate(Axis(axis), &views).map_err(|_| {
            TensorError::ShapeMismatch {
                op: "cat".to_string(),
                left: tensors[0].shape().to_vec(),
                right: tensors.last().unwrap().shape().to_vec(),
            }
        })?;

        Ok(Tensor::from_ndarray(result))
    }

    /// Stacks multiple tensors along an axis (adds a new dimension).
    ///
    /// Similar to NumPy's `np.stack` or PyTorch's `torch.stack`.
    /// All tensors must have the exact same shape.
    pub fn stack(tensors: &[&Tensor], axis: usize) -> TensorResult<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let result = ndarray::stack(Axis(axis), &views).map_err(|_| {
            TensorError::ShapeMismatch {
                op: "stack".to_string(),
                left: tensors[0].shape().to_vec(),
                right: tensors.last().unwrap().shape().to_vec(),
            }
        })?;

        Ok(Tensor::from_ndarray(result))
    }
}

// ============================================================================
// PartialEq Implementation (Used for testing)
// ============================================================================

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_zeros_ones() {
        let z = Tensor::zeros(&[3, 4]);
        assert_eq!(z.shape(), &[3, 4]);
        assert_eq!(z.to_vec(), vec![0.0; 12]);

        let o = Tensor::ones(&[2, 3]);
        assert_eq!(o.to_vec(), vec![1.0; 6]);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_scalar() {
        let s = Tensor::scalar(3.14159);
        assert!(s.is_scalar());
        assert_abs_diff_eq!(s.item(), 3.14159, epsilon = 1e-6);
    }

    #[test]
    fn test_eye() {
        let e = Tensor::eye(3);
        assert_eq!(e.shape(), &[3, 3]);
        let data = e.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_arange() {
        let t = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_linspace() {
        let t = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.shape(), &[5]);
        let data = t.to_vec();
        assert_abs_diff_eq!(data[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[4], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::arange(0.0, 6.0, 1.0);
        let r = t.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_error() {
        let t = Tensor::arange(0.0, 6.0, 1.0);
        assert!(t.reshape(&[2, 4]).is_err());
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.t();
        assert_eq!(tt.shape(), &[3, 2]);
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let t = Tensor::zeros(&[3, 4]);
        let u = t.unsqueeze(0);
        assert_eq!(u.shape(), &[1, 3, 4]);
        let s = u.squeeze(Some(0));
        assert_eq!(s.shape(), &[3, 4]);
    }

    #[test]
    fn test_sum_mean() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_abs_diff_eq!(t.sum().item(), 10.0, epsilon = 1e-6);
        assert_abs_diff_eq!(t.mean().item(), 2.5, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_axis() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = t.sum_axis(0, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        let data = s.to_vec();
        assert_abs_diff_eq!(data[0], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[1], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        let r = t.relu();
        assert_eq!(r.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::from_vec(vec![0.0], &[1]);
        let s = t.sigmoid();
        assert_abs_diff_eq!(s.item(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let s = t.softmax(1).unwrap();
        let data = s.to_vec();
        // sum of softmax outputs should be 1
        let total: f32 = data.iter().sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-5);
        // larger inputs should mapping to larger probabilities
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_clamp() {
        let t = Tensor::from_vec(vec![-2.0, 0.5, 3.0], &[3]);
        let c = t.clamp(0.0, 1.0);
        assert_eq!(c.to_vec(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_select() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let row = t.select(0, 1).unwrap();
        assert_eq!(row.shape(), &[3]);
        assert_eq!(row.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cat() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[1, 2]);
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2]);
        let c = Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_var_std() {
        let t = Tensor::from_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]);
        let var = t.var().item();
        assert_abs_diff_eq!(var, 4.0, epsilon = 1e-5);
        let std = t.std_dev().item();
        assert_abs_diff_eq!(std, 2.0, epsilon = 1e-5);
    }
}
