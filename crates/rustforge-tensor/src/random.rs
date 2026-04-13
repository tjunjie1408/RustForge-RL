//! Random tensor generation.
//!
//! This module provides multiple random initialization strategies, including uniform
//! distribution, normal distribution, and Xavier/He weight initialization schemes
//! commonly used in deep learning.
//!
//! ## Why is Weight Initialization Important?
//!
//! The training of neural networks highly depends on the initial values of the weights:
//! - **Too Large**: Exploding gradients, making training unstable.
//! - **Too Small**: Vanishing gradients, making training extremely slow.
//! - **Xavier Initialization**: Suitable for Sigmoid/Tanh activation functions.
//! - **He Initialization**: Suitable for ReLU activation functions.

use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Uniform, Normal};

use crate::tensor::Tensor;

/// Creates an RNG using a specified seed or the global entropy source.
fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

impl Tensor {
    /// Creates a tensor with uniformly distributed random values U(low, high).
    ///
    /// ## Arguments
    /// - `shape`: The shape of the tensor.
    /// - `low`: The lower bound (inclusive).
    /// - `high`: The upper bound (exclusive).
    /// - `seed`: Optional random seed (for reproducibility).
    ///
    /// ## Example
    /// ```rust
    /// use rustforge_tensor::Tensor;
    /// let t = Tensor::rand_uniform(&[3, 4], 0.0, 1.0, Some(42));
    /// assert_eq!(t.shape(), &[3, 4]);
    /// ```
    pub fn rand_uniform(shape: &[usize], low: f32, high: f32, seed: Option<u64>) -> Self {
        let mut rng = make_rng(seed);
        let dist = Uniform::new(low, high);
        let data = ArrayD::random_using(IxDyn(shape), dist, &mut rng);
        Tensor::from_ndarray(data)
    }

    /// Creates a tensor with normally distributed random values N(mean, std²).
    ///
    /// ## Arguments
    /// - `shape`: The shape of the tensor.
    /// - `mean`: The mean.
    /// - `std`: The standard deviation.
    /// - `seed`: Optional random seed.
    pub fn rand_normal(shape: &[usize], mean: f32, std: f32, seed: Option<u64>) -> Self {
        let mut rng = make_rng(seed);
        let dist = Normal::new(mean, std).expect("Invalid normal distribution parameters");
        let data = ArrayD::random_using(IxDyn(shape), dist, &mut rng);
        Tensor::from_ndarray(data)
    }

    /// Creates a tensor with standard normal distributed random values N(0, 1).
    pub fn randn(shape: &[usize], seed: Option<u64>) -> Self {
        Self::rand_normal(shape, 0.0, 1.0, seed)
    }

    /// Xavier/Glorot uniform initialization.
    ///
    /// ## Mathematical Definition
    /// W ~ U(-a, a), where a = sqrt(6 / (fan_in + fan_out))
    ///
    /// ## Applicable Scenarios
    /// - Sigmoid and Tanh activation functions.
    /// - Maintains consistent variance of signals during forward and backward passes.
    ///
    /// ## Arguments
    /// - `shape`: Weight shape, conventionally shape[0] = fan_out, shape[1] = fan_in.
    /// - `seed`: Optional random seed.
    ///
    /// ## References
    /// Glorot & Bengio, "Understanding the difficulty of training deep feedforward
    /// neural networks", AISTATS 2010.
    pub fn xavier_uniform(shape: &[usize], seed: Option<u64>) -> Self {
        let (fan_in, fan_out) = compute_fans(shape);
        let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
        Self::rand_uniform(shape, -a, a, seed)
    }

    /// Xavier/Glorot normal initialization.
    ///
    /// ## Mathematical Definition
    /// W ~ N(0, σ²), where σ = sqrt(2 / (fan_in + fan_out))
    pub fn xavier_normal(shape: &[usize], seed: Option<u64>) -> Self {
        let (fan_in, fan_out) = compute_fans(shape);
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        Self::rand_normal(shape, 0.0, std, seed)
    }

    /// Kaiming/He uniform initialization.
    ///
    /// ## Mathematical Definition
    /// W ~ U(-a, a), where a = sqrt(6 / fan_in)
    ///
    /// ## Applicable Scenarios
    /// - ReLU activation function and its variants (Leaky ReLU, PReLU).
    /// - ReLU zeroes out output for about half the neurons; He initialization
    ///   compensates for this effect by increasing the initial variance.
    ///
    /// ## References
    /// He et al., "Delving Deep into Rectifiers", ICCV 2015.
    pub fn kaiming_uniform(shape: &[usize], seed: Option<u64>) -> Self {
        let (fan_in, _) = compute_fans(shape);
        let a = (6.0 / fan_in as f32).sqrt();
        Self::rand_uniform(shape, -a, a, seed)
    }

    /// Kaiming/He normal initialization.
    ///
    /// ## Mathematical Definition
    /// W ~ N(0, σ²), where σ = sqrt(2 / fan_in)
    pub fn kaiming_normal(shape: &[usize], seed: Option<u64>) -> Self {
        let (fan_in, _) = compute_fans(shape);
        let std = (2.0 / fan_in as f32).sqrt();
        Self::rand_normal(shape, 0.0, std, seed)
    }
}

/// Computes the fan_in and fan_out of a weight matrix.
///
/// - For 2D weights `[out_features, in_features]`:
///   fan_in = in_features, fan_out = out_features
/// - For convolutional weights `[out_channels, in_channels, kH, kW]`:
///   fan_in = in_channels * kH * kW, fan_out = out_channels * kH * kW
/// - For 1D `[n]`: fan_in = fan_out = n
fn compute_fans(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], shape[0]),
        2 => (shape[1], shape[0]),
        _ => {
            // Convolutional kernels: [out_channels, in_channels, *kernel_size]
            let receptive_field: usize = shape[2..].iter().product();
            let fan_in = shape[1] * receptive_field;
            let fan_out = shape[0] * receptive_field;
            (fan_in, fan_out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_uniform_shape() {
        let t = Tensor::rand_uniform(&[3, 4], 0.0, 1.0, Some(42));
        assert_eq!(t.shape(), &[3, 4]);
        // All values should be within the [0, 1) range
        for &v in t.to_vec().iter() {
            assert!((0.0..1.0).contains(&v), "Value {} out of range [0, 1)", v);
        }
    }

    #[test]
    fn test_rand_normal_shape() {
        let t = Tensor::rand_normal(&[100, 100], 0.0, 1.0, Some(42));
        assert_eq!(t.shape(), &[100, 100]);
        // Mean of a large sample should be close to 0
        let mean = t.mean().item();
        assert!(mean.abs() < 0.1, "Mean {} too far from 0", mean);
    }

    #[test]
    fn test_reproducibility() {
        let t1 = Tensor::rand_uniform(&[5, 5], 0.0, 1.0, Some(123));
        let t2 = Tensor::rand_uniform(&[5, 5], 0.0, 1.0, Some(123));
        assert_eq!(t1.to_vec(), t2.to_vec(), "Same seed should produce same values");
    }

    #[test]
    fn test_xavier_uniform() {
        let t = Tensor::xavier_uniform(&[256, 128], Some(42));
        assert_eq!(t.shape(), &[256, 128]);
        // Xavier's range should be sqrt(6/(128+256)) ≈ 0.125
        let a = (6.0_f32 / (128.0 + 256.0)).sqrt();
        for &v in t.to_vec().iter() {
            assert!(v >= -a && v <= a, "Xavier value {} out of expected range", v);
        }
    }

    #[test]
    fn test_kaiming_normal() {
        let t = Tensor::kaiming_normal(&[256, 128], Some(42));
        assert_eq!(t.shape(), &[256, 128]);
        // Kaiming's std should be sqrt(2/128) ≈ 0.125
        let expected_std = (2.0_f32 / 128.0).sqrt();
        let actual_std = t.std_dev().item();
        assert!(
            (actual_std - expected_std).abs() < 0.02,
            "Kaiming std {} too far from expected {}",
            actual_std,
            expected_std
        );
    }

    #[test]
    fn test_compute_fans_2d() {
        assert_eq!(compute_fans(&[256, 128]), (128, 256));
    }

    #[test]
    fn test_compute_fans_conv() {
        // [out_channels=64, in_channels=3, kH=3, kW=3]
        assert_eq!(compute_fans(&[64, 3, 3, 3]), (27, 576));
    }
}
