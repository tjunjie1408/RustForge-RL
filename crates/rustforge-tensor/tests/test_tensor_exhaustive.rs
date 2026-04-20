//! Exhaustive integration tests for rustforge-tensor.
//!
//! Covers: constructors, property queries, reshaping/transforms, indexing,
//! reductions, element-wise math, softmax, concatenation, arithmetic with
//! broadcasting, numerical edge cases, thread safety, and property-based fuzz.

use approx::assert_abs_diff_eq;
use rustforge_tensor::{Tensor, TensorError};

// Module: Constructors

mod constructors {
    use super::*;

    #[test]
    fn from_vec_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn from_vec_1d() {
        let t = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.to_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn from_vec_3d() {
        let t = Tensor::from_vec((0..24).map(|i| i as f32).collect(), &[2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn from_vec_4d() {
        let t = Tensor::from_vec(vec![0.0; 120], &[2, 3, 4, 5]);
        assert_eq!(t.shape(), &[2, 3, 4, 5]);
        assert_eq!(t.numel(), 120);
    }

    #[test]
    fn from_vec_5d() {
        let t = Tensor::from_vec(vec![1.0; 240], &[2, 3, 4, 5, 2]);
        assert_eq!(t.shape(), &[2, 3, 4, 5, 2]);
        assert_eq!(t.numel(), 240);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn from_vec_shape_mismatch_panics() {
        Tensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 2]);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn from_vec_empty_data_nonzero_shape_panics() {
        Tensor::from_vec(vec![], &[2, 3]);
    }

    #[test]
    fn zeros_various_shapes() {
        for shape in &[vec![0], vec![1], vec![5], vec![3, 4], vec![2, 3, 4]] {
            let t = Tensor::zeros(shape);
            assert_eq!(t.shape(), shape.as_slice());
            for &v in t.to_vec().iter() {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn ones_various_shapes() {
        for shape in &[vec![1], vec![5], vec![3, 4], vec![2, 3, 4]] {
            let t = Tensor::ones(shape);
            assert_eq!(t.shape(), shape.as_slice());
            for &v in t.to_vec().iter() {
                assert_eq!(v, 1.0);
            }
        }
    }

    #[test]
    fn full_custom_value() {
        let t = Tensor::full(&[3, 3], 42.0);
        assert_eq!(t.shape(), &[3, 3]);
        for &v in t.to_vec().iter() {
            assert_eq!(v, 42.0);
        }
    }

    #[test]
    fn full_negative_value() {
        let t = Tensor::full(&[2, 2], -999.0);
        for &v in t.to_vec().iter() {
            assert_eq!(v, -999.0);
        }
    }

    #[test]
    fn scalar_creation() {
        let s = Tensor::scalar(3.15);
        assert!(s.is_scalar());
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert_abs_diff_eq!(s.item(), 3.15, epsilon = 1e-6);
    }

    #[test]
    fn eye_various_sizes() {
        for n in [0, 1, 2, 5, 10] {
            let e = Tensor::eye(n);
            assert_eq!(e.shape(), &[n, n]);
            let data = e.to_vec();
            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_eq!(data[i * n + j], expected);
                }
            }
        }
    }

    #[test]
    fn arange_positive_step() {
        let t = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn arange_fractional_step() {
        let t = Tensor::arange(0.0, 1.0, 0.25);
        assert_eq!(t.shape(), &[4]);
        assert_abs_diff_eq!(t.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(t.to_vec()[3], 0.75, epsilon = 1e-6);
    }

    #[test]
    fn arange_negative_step() {
        let t = Tensor::arange(5.0, 0.0, -1.0);
        assert_eq!(t.to_vec(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn arange_zero_step_empty() {
        let t = Tensor::arange(0.0, 5.0, 0.0);
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn arange_empty_range() {
        let t = Tensor::arange(5.0, 0.0, 1.0);
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn linspace_normal() {
        let t = Tensor::linspace(0.0, 10.0, 5);
        let data = t.to_vec();
        assert_abs_diff_eq!(data[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[2], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[4], 10.0, epsilon = 1e-6);
    }

    #[test]
    fn linspace_zero_points() {
        let t = Tensor::linspace(0.0, 1.0, 0);
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn linspace_single_point() {
        let t = Tensor::linspace(5.0, 10.0, 1);
        assert_eq!(t.numel(), 1);
        assert_abs_diff_eq!(t.item(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn linspace_same_start_end() {
        let t = Tensor::linspace(3.0, 3.0, 5);
        for &v in t.to_vec().iter() {
            assert_abs_diff_eq!(v, 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn linspace_many_points() {
        let t = Tensor::linspace(0.0, 1.0, 1001);
        assert_eq!(t.shape(), &[1001]);
        assert_abs_diff_eq!(t.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(t.to_vec()[1000], 1.0, epsilon = 1e-4);
    }
}

// ============================================================================
// Module: Property Queries
// ============================================================================

mod properties {
    use super::*;

    #[test]
    fn shape_ndim_numel_0d() {
        let t = Tensor::scalar(1.0);
        assert_eq!(t.shape(), &[] as &[usize]);
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.numel(), 1);
        assert!(t.is_scalar());
        assert!(!t.is_empty());
    }

    #[test]
    fn shape_ndim_numel_1d() {
        let t = Tensor::zeros(&[5]);
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.numel(), 5);
        assert!(!t.is_scalar());
    }

    #[test]
    fn shape_ndim_numel_3d() {
        let t = Tensor::zeros(&[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn is_empty_zero_dim() {
        let t = Tensor::zeros(&[0]);
        assert!(t.is_empty());
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn item_scalar() {
        assert_abs_diff_eq!(Tensor::scalar(42.0).item(), 42.0, epsilon = 1e-6);
    }

    #[test]
    fn item_single_element_1d() {
        let t = Tensor::from_vec(vec![7.5], &[1]);
        assert_abs_diff_eq!(t.item(), 7.5, epsilon = 1e-6);
    }

    #[test]
    fn item_single_element_2d() {
        let t = Tensor::from_vec(vec![99.0], &[1, 1]);
        assert_abs_diff_eq!(t.item(), 99.0, epsilon = 1e-6);
    }

    #[test]
    fn item_single_element_3d() {
        let t = Tensor::from_vec(vec![0.5], &[1, 1, 1]);
        assert_abs_diff_eq!(t.item(), 0.5, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "item() requires a single-element tensor")]
    fn item_multi_element_panics() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        t.item();
    }

    #[test]
    fn to_vec_preserves_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_vec(data.clone(), &[2, 3]);
        assert_eq!(t.to_vec(), data);
    }

    #[test]
    fn data_returns_reference() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let arr = t.data();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn data_mut_allows_modification() {
        let mut t = Tensor::zeros(&[3]);
        t.data_mut().fill(5.0);
        assert_eq!(t.to_vec(), vec![5.0, 5.0, 5.0]);
    }
}

// Module: Reshape & Transform

mod reshape_transform {
    use super::*;

    #[test]
    fn reshape_1d_to_2d() {
        let t = Tensor::arange(0.0, 6.0, 1.0);
        let r = t.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.to_vec(), t.to_vec());
    }

    #[test]
    fn reshape_2d_to_3d() {
        let t = Tensor::arange(0.0, 24.0, 1.0).reshape(&[2, 12]).unwrap();
        let r = t.reshape(&[2, 3, 4]).unwrap();
        assert_eq!(r.shape(), &[2, 3, 4]);
    }

    #[test]
    fn reshape_3d_to_1d() {
        let t = Tensor::zeros(&[2, 3, 4]);
        let r = t.reshape(&[24]).unwrap();
        assert_eq!(r.shape(), &[24]);
    }

    #[test]
    fn reshape_preserves_data_order() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_vec(data.clone(), &[6]);
        let r = t.reshape(&[2, 3]).unwrap();
        assert_eq!(r.to_vec(), data);
    }

    #[test]
    fn reshape_error_element_mismatch() {
        let t = Tensor::zeros(&[6]);
        let result = t.reshape(&[2, 4]);
        assert!(result.is_err());
        match result.unwrap_err() {
            TensorError::InvalidShape {
                expected_elements,
                got_elements,
                ..
            } => {
                assert_eq!(expected_elements, 6);
                assert_eq!(got_elements, 8);
            }
            other => panic!("Expected InvalidShape, got {:?}", other),
        }
    }

    #[test]
    fn flatten_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let f = t.flatten();
        assert_eq!(f.shape(), &[4]);
        assert_eq!(f.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn flatten_3d() {
        let t = Tensor::zeros(&[2, 3, 4]);
        let f = t.flatten();
        assert_eq!(f.shape(), &[24]);
    }

    #[test]
    fn flatten_1d_noop() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let f = t.flatten();
        assert_eq!(f.shape(), &[3]);
    }

    #[test]
    fn transpose_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.t();
        assert_eq!(tt.shape(), &[3, 2]);
        // Row-major: original [1,2,3; 4,5,6], transposed [1,4; 2,5; 3,6]
        assert_eq!(tt.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_1d_identity() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let tt = t.t();
        assert_eq!(tt.shape(), &[3]);
        assert_eq!(tt.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn transpose_3d_swaps_last_two() {
        let t = Tensor::zeros(&[2, 3, 4]);
        let tt = t.t();
        assert_eq!(tt.shape(), &[2, 4, 3]);
    }

    #[test]
    fn transpose_square_involution() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let tt = t.t().t();
        assert_eq!(tt.to_vec(), t.to_vec());
    }

    #[test]
    fn permute_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p = t.permute(&[1, 0]);
        assert_eq!(p.shape(), &[3, 2]);
    }

    #[test]
    fn permute_3d() {
        let t = Tensor::zeros(&[2, 3, 4]);
        let p = t.permute(&[2, 0, 1]);
        assert_eq!(p.shape(), &[4, 2, 3]);
    }

    #[test]
    fn unsqueeze_axis_0() {
        let t = Tensor::zeros(&[3, 4]);
        let u = t.unsqueeze(0);
        assert_eq!(u.shape(), &[1, 3, 4]);
    }

    #[test]
    fn unsqueeze_axis_middle() {
        let t = Tensor::zeros(&[3, 4]);
        let u = t.unsqueeze(1);
        assert_eq!(u.shape(), &[3, 1, 4]);
    }

    #[test]
    fn unsqueeze_axis_end() {
        let t = Tensor::zeros(&[3, 4]);
        let u = t.unsqueeze(2);
        assert_eq!(u.shape(), &[3, 4, 1]);
    }

    #[test]
    fn squeeze_specific_axis() {
        let t = Tensor::zeros(&[1, 3, 1, 4]);
        let s = t.squeeze(Some(0));
        assert_eq!(s.shape(), &[3, 1, 4]);
    }

    #[test]
    fn squeeze_all() {
        let t = Tensor::zeros(&[1, 3, 1, 4, 1]);
        let s = t.squeeze(None);
        assert_eq!(s.shape(), &[3, 4]);
    }

    #[test]
    fn squeeze_to_scalar() {
        let t = Tensor::from_vec(vec![42.0], &[1, 1, 1]);
        let s = t.squeeze(None);
        assert!(s.is_scalar());
        assert_abs_diff_eq!(s.item(), 42.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "Cannot squeeze axis")]
    fn squeeze_non_one_panics() {
        let t = Tensor::zeros(&[3, 4]);
        t.squeeze(Some(0));
    }

    #[test]
    fn unsqueeze_then_squeeze_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let u = t.unsqueeze(0).unsqueeze(2);
        assert_eq!(u.shape(), &[1, 3, 1]);
        let s = u.squeeze(None);
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec(), vec![1.0, 2.0, 3.0]);
    }
}

// Module: Indexing & Slicing

mod indexing_slicing {
    use super::*;

    #[test]
    fn select_axis_0_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let row0 = t.select(0, 0).unwrap();
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.to_vec(), vec![1.0, 2.0, 3.0]);

        let row1 = t.select(0, 1).unwrap();
        assert_eq!(row1.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn select_axis_1_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let col1 = t.select(1, 1).unwrap();
        assert_eq!(col1.shape(), &[2]);
        assert_eq!(col1.to_vec(), vec![2.0, 5.0]);
    }

    #[test]
    fn select_3d() {
        let t = Tensor::zeros(&[2, 3, 4]);
        let s = t.select(0, 0).unwrap();
        assert_eq!(s.shape(), &[3, 4]);
    }

    #[test]
    fn select_axis_oob_returns_error() {
        let t = Tensor::zeros(&[2, 3]);
        let result = t.select(2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn slice_axis_normal() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let s = t.slice_axis(0, 1, 4).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn slice_axis_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = t.slice_axis(1, 0, 2).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.to_vec(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn slice_axis_full_range() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let s = t.slice_axis(0, 0, 3).unwrap();
        assert_eq!(s.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn slice_axis_single_element() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let s = t.slice_axis(0, 1, 2).unwrap();
        assert_eq!(s.to_vec(), vec![2.0]);
    }

    #[test]
    fn slice_axis_oob_returns_error() {
        let t = Tensor::zeros(&[3, 4]);
        let result = t.slice_axis(5, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn select_boundary_indices() {
        let t = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
        // First element
        assert_abs_diff_eq!(t.select(0, 0).unwrap().item(), 10.0, epsilon = 1e-6);
        // Last element
        assert_abs_diff_eq!(t.select(0, 2).unwrap().item(), 30.0, epsilon = 1e-6);
    }
}

// Module: Reductions

mod reductions {
    use super::*;

    #[test]
    fn sum_1d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        assert_abs_diff_eq!(t.sum().item(), 6.0, epsilon = 1e-6);
    }

    #[test]
    fn sum_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_abs_diff_eq!(t.sum().item(), 10.0, epsilon = 1e-6);
    }

    #[test]
    fn sum_single_element() {
        let t = Tensor::scalar(42.0);
        assert_abs_diff_eq!(t.sum().item(), 42.0, epsilon = 1e-6);
    }

    #[test]
    fn mean_known_values() {
        let t = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4]);
        assert_abs_diff_eq!(t.mean().item(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn mean_single_element() {
        let t = Tensor::from_vec(vec![7.0], &[1]);
        assert_abs_diff_eq!(t.mean().item(), 7.0, epsilon = 1e-6);
    }

    #[test]
    fn sum_axis_0_keepdim_false() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = t.sum_axis(0, false).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis_0_keepdim_true() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = t.sum_axis(0, true).unwrap();
        assert_eq!(s.shape(), &[1, 3]);
        assert_eq!(s.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis_1() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = t.sum_axis(1, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        assert_eq!(s.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn sum_axis_oob_error() {
        let t = Tensor::zeros(&[2, 3]);
        assert!(t.sum_axis(2, false).is_err());
    }

    #[test]
    fn mean_axis_known_values() {
        let t = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], &[2, 2]);
        let m = t.mean_axis(1, false).unwrap();
        assert_eq!(m.shape(), &[2]);
        assert_abs_diff_eq!(m.to_vec()[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(m.to_vec()[1], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn max_positive() {
        let t = Tensor::from_vec(vec![1.0, 5.0, 3.0], &[3]);
        assert_abs_diff_eq!(t.max().unwrap().item(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn max_negative() {
        let t = Tensor::from_vec(vec![-5.0, -1.0, -3.0], &[3]);
        assert_abs_diff_eq!(t.max().unwrap().item(), -1.0, epsilon = 1e-6);
    }

    #[test]
    fn max_axis_2d() {
        let t = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
        let m = t.max_axis(1, false).unwrap();
        assert_eq!(m.shape(), &[2]);
        assert_eq!(m.to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn argmax_axis_correctness() {
        let t = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
        let indices = t.argmax_axis(1).unwrap();
        assert_eq!(indices, vec![1, 2]); // max at index 1 in first row, index 2 in second
    }

    #[test]
    fn var_known_values() {
        // Values: [2, 4, 4, 4, 5, 5, 7, 9], mean=5, var=4
        let t = Tensor::from_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]);
        assert_abs_diff_eq!(t.var().item(), 4.0, epsilon = 1e-4);
    }

    #[test]
    fn var_constant_tensor_zero() {
        let t = Tensor::full(&[100], 5.0);
        assert_abs_diff_eq!(t.var().item(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn std_dev_known_values() {
        let t = Tensor::from_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]);
        assert_abs_diff_eq!(t.std_dev().item(), 2.0, epsilon = 1e-4);
    }
}

// Module: Element-wise Math

mod elementwise_math {
    use super::*;

    #[test]
    fn relu_mixed_values() {
        let t = Tensor::from_vec(vec![-3.0, -1.0, 0.0, 0.5, 2.0, 100.0], &[6]);
        let r = t.relu();
        assert_eq!(r.to_vec(), vec![0.0, 0.0, 0.0, 0.5, 2.0, 100.0]);
    }

    #[test]
    fn relu_all_negative() {
        let t = Tensor::from_vec(vec![-1.0, -2.0, -3.0], &[3]);
        assert_eq!(t.relu().to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_all_positive() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        assert_eq!(t.relu().to_vec(), t.to_vec());
    }

    #[test]
    fn sigmoid_at_zero() {
        let t = Tensor::from_vec(vec![0.0], &[1]);
        assert_abs_diff_eq!(t.sigmoid().item(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn sigmoid_large_positive() {
        let t = Tensor::from_vec(vec![100.0, 500.0, 1000.0], &[3]);
        let s = t.sigmoid();
        for &v in s.to_vec().iter() {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-4);
            assert!(!v.is_nan());
            assert!(!v.is_infinite());
        }
    }

    #[test]
    fn sigmoid_large_negative() {
        let t = Tensor::from_vec(vec![-100.0, -500.0, -1000.0], &[3]);
        let s = t.sigmoid();
        for &v in s.to_vec().iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-4);
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn sigmoid_monotonicity() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let s = t.sigmoid().to_vec();
        for i in 0..4 {
            assert!(s[i] < s[i + 1], "Sigmoid must be monotonically increasing");
        }
    }

    #[test]
    fn sigmoid_output_range() {
        let t = Tensor::rand_uniform(&[1000], -10.0, 10.0, Some(42));
        let s = t.sigmoid();
        for &v in s.to_vec().iter() {
            assert!(
                v > 0.0 && v < 1.0,
                "Sigmoid output must be in (0,1), got {}",
                v
            );
        }
    }

    #[test]
    fn tanh_at_zero() {
        let t = Tensor::from_vec(vec![0.0], &[1]);
        assert_abs_diff_eq!(t.tanh_().item(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn tanh_output_range() {
        let t = Tensor::rand_uniform(&[1000], -10.0, 10.0, Some(42));
        let th = t.tanh_();
        for &v in th.to_vec().iter() {
            assert!(
                (-1.0..=1.0).contains(&v),
                "Tanh output must be in [-1,1], got {}",
                v
            );
        }
    }

    #[test]
    fn tanh_large_values_saturate() {
        let t = Tensor::from_vec(vec![100.0, -100.0], &[2]);
        let th = t.tanh_();
        assert_abs_diff_eq!(th.to_vec()[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(th.to_vec()[1], -1.0, epsilon = 1e-4);
    }

    #[test]
    fn exp_basic() {
        let t = Tensor::from_vec(vec![0.0, 1.0], &[2]);
        let e = t.exp();
        assert_abs_diff_eq!(e.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(e.to_vec()[1], std::f32::consts::E, epsilon = 1e-4);
    }

    #[test]
    fn exp_large_overflow() {
        let t = Tensor::from_vec(vec![200.0], &[1]);
        let e = t.exp();
        assert!(e.item().is_infinite(), "exp(200) should overflow to Inf");
    }

    #[test]
    fn exp_negative() {
        let t = Tensor::from_vec(vec![-10.0], &[1]);
        let e = t.exp();
        assert!(e.item() > 0.0, "exp(x) is always positive");
        assert!(e.item() < 1e-3);
    }

    #[test]
    fn log_basic() {
        let t = Tensor::from_vec(vec![1.0, std::f32::consts::E], &[2]);
        let l = t.log();
        assert_abs_diff_eq!(l.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(l.to_vec()[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn log_zero_gives_clamped_value() {
        // After numerics hardening, log(0) is clamped to log(1e-7) ≈ -16.12
        let t = Tensor::from_vec(vec![0.0], &[1]);
        let l = t.log();
        assert!(l.item().is_finite(), "log(0) should be clamped to finite value");
        assert!(l.item() < -10.0, "log(0) clamped value should be very negative");
    }

    #[test]
    fn log_negative_gives_clamped_value() {
        // After numerics hardening, log(-1) is clamped to log(1e-7) ≈ -16.12
        let t = Tensor::from_vec(vec![-1.0], &[1]);
        let l = t.log();
        assert!(l.item().is_finite(), "log(negative) should be clamped to finite value");
    }

    #[test]
    fn log_safe_prevents_neg_inf() {
        let t = Tensor::from_vec(vec![0.0], &[1]);
        let l = t.log_safe(1e-7);
        assert!(!l.item().is_infinite(), "log_safe should prevent -Inf");
        assert!(!l.item().is_nan());
    }

    #[test]
    fn pow_integer_exponent() {
        let t = Tensor::from_vec(vec![2.0, 3.0], &[2]);
        let p = t.pow(3.0);
        assert_abs_diff_eq!(p.to_vec()[0], 8.0, epsilon = 1e-5);
        assert_abs_diff_eq!(p.to_vec()[1], 27.0, epsilon = 1e-4);
    }

    #[test]
    fn pow_fractional_exponent() {
        let t = Tensor::from_vec(vec![4.0, 9.0], &[2]);
        let p = t.pow(0.5);
        assert_abs_diff_eq!(p.to_vec()[0], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(p.to_vec()[1], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn sqrt_perfect_squares() {
        let t = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0], &[5]);
        let s = t.sqrt();
        assert_eq!(s.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sqrt_zero() {
        let t = Tensor::from_vec(vec![0.0], &[1]);
        assert_eq!(t.sqrt().item(), 0.0);
    }

    #[test]
    fn sqrt_negative_gives_zero() {
        // After numerics hardening, sqrt(-1) is clamped to sqrt(0) = 0.0
        let t = Tensor::from_vec(vec![-1.0], &[1]);
        assert_abs_diff_eq!(t.sqrt().item(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn abs_mixed() {
        let t = Tensor::from_vec(vec![-3.0, 0.0, 5.0], &[3]);
        assert_eq!(t.abs().to_vec(), vec![3.0, 0.0, 5.0]);
    }

    #[test]
    fn neg_values() {
        let t = Tensor::from_vec(vec![1.0, -2.0, 0.0], &[3]);
        assert_eq!(t.neg().to_vec(), vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn reciprocal_basic() {
        let t = Tensor::from_vec(vec![2.0, 4.0, 0.5], &[3]);
        let r = t.reciprocal();
        assert_abs_diff_eq!(r.to_vec()[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(r.to_vec()[1], 0.25, epsilon = 1e-6);
        assert_abs_diff_eq!(r.to_vec()[2], 2.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "reciprocal() called with zero elements")]
    fn reciprocal_zero_panics_in_debug() {
        // After numerics hardening, reciprocal(0) panics in debug builds
        let t = Tensor::from_vec(vec![0.0], &[1]);
        let _ = t.reciprocal();
    }

    #[test]
    fn clamp_all_in_range() {
        let t = Tensor::from_vec(vec![0.3, 0.5, 0.7], &[3]);
        assert_eq!(t.clamp(0.0, 1.0).to_vec(), vec![0.3, 0.5, 0.7]);
    }

    #[test]
    fn clamp_clips_extremes() {
        let t = Tensor::from_vec(vec![-5.0, 0.5, 10.0], &[3]);
        assert_eq!(t.clamp(0.0, 1.0).to_vec(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn clamp_tight_range() {
        let t = Tensor::from_vec(vec![-100.0, 0.0, 100.0], &[3]);
        let c = t.clamp(0.0, 0.0);
        assert_eq!(c.to_vec(), vec![0.0, 0.0, 0.0]);
    }
}

// Module: Softmax & Log-Softmax

mod softmax {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = t.softmax(1).unwrap();
        for row in 0..2 {
            let row_sum: f32 = (0..3).map(|c| s.to_vec()[row * 3 + c]).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn softmax_all_positive() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let s = t.softmax(1).unwrap();
        for &v in s.to_vec().iter() {
            assert!(v > 0.0, "Softmax outputs must be positive");
        }
    }

    #[test]
    fn softmax_monotonicity() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let s = t.softmax(1).unwrap().to_vec();
        assert!(
            s[0] < s[1] && s[1] < s[2],
            "Softmax should preserve ordering"
        );
    }

    #[test]
    fn softmax_identical_inputs_uniform() {
        let t = Tensor::from_vec(vec![5.0, 5.0, 5.0], &[1, 3]);
        let s = t.softmax(1).unwrap().to_vec();
        for &v in &s {
            assert_abs_diff_eq!(v, 1.0 / 3.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn softmax_numerical_stability_large() {
        let t = Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3]);
        let s = t.softmax(1).unwrap();
        let data = s.to_vec();
        for &v in &data {
            assert!(!v.is_nan(), "Softmax should not produce NaN");
            assert!(!v.is_infinite(), "Softmax should not produce Inf");
        }
        let sum: f32 = data.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn softmax_axis_oob_error() {
        let t = Tensor::zeros(&[2, 3]);
        assert!(t.softmax(5).is_err());
    }

    #[test]
    fn log_softmax_consistency() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let ls = t.log_softmax(1).unwrap();
        let s = t.softmax(1).unwrap().log();
        let ls_data = ls.to_vec();
        let s_data = s.to_vec();
        for (a, b) in ls_data.iter().zip(s_data.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-4);
        }
    }

    #[test]
    fn log_softmax_axis_oob_error() {
        let t = Tensor::zeros(&[2, 3]);
        assert!(t.log_softmax(5).is_err());
    }
}

// Module: Concatenation & Stacking

mod concat_stack {
    use super::*;

    #[test]
    fn cat_axis_0() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[1, 2]);
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cat_axis_1() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0], &[2, 1]);
        let c = Tensor::cat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn cat_multiple_tensors() {
        let a = Tensor::ones(&[1, 3]);
        let b = Tensor::full(&[1, 3], 2.0);
        let c = Tensor::full(&[1, 3], 3.0);
        let result = Tensor::cat(&[&a, &b, &c], 0).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
    }

    #[test]
    fn cat_empty_list_error() {
        let result = Tensor::cat(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn stack_axis_0() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2]);
        let c = Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn stack_axis_1() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2]);
        let c = Tensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn stack_empty_list_error() {
        let result = Tensor::stack(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn cat_mismatched_shapes_error() {
        let a = Tensor::ones(&[2, 3]);
        let b = Tensor::ones(&[2, 4]);
        let result = Tensor::cat(&[&a, &b], 0);
        assert!(result.is_err());
    }
}

// Module: Arithmetic & Broadcasting

mod arithmetic_broadcast {
    use super::*;

    #[test]
    fn add_same_shape() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let c = &a + &b;
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn sub_same_shape() {
        let a = Tensor::from_vec(vec![10.0, 20.0], &[2]);
        let b = Tensor::from_vec(vec![3.0, 7.0], &[2]);
        assert_eq!((&a - &b).to_vec(), vec![7.0, 13.0]);
    }

    #[test]
    fn mul_same_shape() {
        let a = Tensor::from_vec(vec![2.0, 3.0], &[2]);
        let b = Tensor::from_vec(vec![4.0, 5.0], &[2]);
        assert_eq!((&a * &b).to_vec(), vec![8.0, 15.0]);
    }

    #[test]
    fn div_same_shape() {
        let a = Tensor::from_vec(vec![10.0, 20.0], &[2]);
        let b = Tensor::from_vec(vec![2.0, 5.0], &[2]);
        assert_eq!((&a / &b).to_vec(), vec![5.0, 4.0]);
    }

    #[test]
    fn broadcast_2d_plus_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn broadcast_row_times_col() {
        let row = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let col = Tensor::from_vec(vec![10.0, 20.0], &[2, 1]);
        let result = &row * &col;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.to_vec(), vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn broadcast_3d() {
        let a = Tensor::ones(&[2, 1, 5]);
        let b = Tensor::ones(&[3, 5]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3, 5]);
    }

    #[test]
    #[should_panic]
    fn broadcast_incompatible_shapes_panics() {
        let a = Tensor::ones(&[3, 4]);
        let b = Tensor::ones(&[5]);
        let _ = &a + &b;
    }

    #[test]
    #[should_panic]
    fn broadcast_incompatible_inner_dims_panics() {
        let a = Tensor::ones(&[3, 4]);
        let b = Tensor::ones(&[3, 5]);
        let _ = &a * &b;
    }

    #[test]
    fn div_by_zero_gives_inf() {
        let a = Tensor::from_vec(vec![1.0], &[1]);
        let b = Tensor::from_vec(vec![0.0], &[1]);
        let c = &a / &b;
        assert!(c.item().is_infinite());
    }

    #[test]
    fn scalar_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = &a + 10.0;
        assert_eq!(b.to_vec(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn scalar_sub() {
        let a = Tensor::from_vec(vec![10.0, 20.0], &[2]);
        let b = &a - 5.0;
        assert_eq!(b.to_vec(), vec![5.0, 15.0]);
    }

    #[test]
    fn scalar_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = &a * 3.0;
        assert_eq!(b.to_vec(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn f32_mul_tensor() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = 5.0 * &a;
        assert_eq!(b.to_vec(), vec![5.0, 10.0]);
    }

    #[test]
    fn f32_add_tensor() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = 100.0 + &a;
        assert_eq!(b.to_vec(), vec![101.0, 102.0]);
    }

    #[test]
    fn scalar_div() {
        let a = Tensor::from_vec(vec![10.0, 20.0], &[2]);
        let b = &a / 2.0;
        assert_eq!(b.to_vec(), vec![5.0, 10.0]);
    }

    #[test]
    fn neg_operator() {
        let a = Tensor::from_vec(vec![1.0, -2.0, 0.0], &[3]);
        let b = -&a;
        assert_eq!(b.to_vec(), vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn neg_value_operator() {
        let a = Tensor::from_vec(vec![1.0, -2.0, 0.0], &[3]);
        let b = -a;
        assert_eq!(b.to_vec(), vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn all_ownership_variants_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2]);
        let expected = vec![4.0, 6.0];

        assert_eq!((&a + &b).to_vec(), expected);
        assert_eq!((a.clone() + &b).to_vec(), expected);
        assert_eq!((&a + b.clone()).to_vec(), expected);
        assert_eq!((a.clone() + b.clone()).to_vec(), expected);
    }

    #[test]
    fn all_ownership_variants_sub() {
        let a = Tensor::from_vec(vec![5.0, 6.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let expected = vec![4.0, 4.0];

        assert_eq!((&a - &b).to_vec(), expected);
        assert_eq!((a.clone() - &b).to_vec(), expected);
        assert_eq!((&a - b.clone()).to_vec(), expected);
        assert_eq!((a.clone() - b.clone()).to_vec(), expected);
    }

    #[test]
    fn all_ownership_variants_mul() {
        let a = Tensor::from_vec(vec![2.0, 3.0], &[2]);
        let b = Tensor::from_vec(vec![4.0, 5.0], &[2]);
        let expected = vec![8.0, 15.0];

        assert_eq!((&a * &b).to_vec(), expected);
        assert_eq!((a.clone() * &b).to_vec(), expected);
        assert_eq!((&a * b.clone()).to_vec(), expected);
        assert_eq!((a.clone() * b.clone()).to_vec(), expected);
    }

    #[test]
    fn all_ownership_variants_div() {
        let a = Tensor::from_vec(vec![10.0, 20.0], &[2]);
        let b = Tensor::from_vec(vec![2.0, 5.0], &[2]);
        let expected = vec![5.0, 4.0];

        assert_eq!((&a / &b).to_vec(), expected);
        assert_eq!((a.clone() / &b).to_vec(), expected);
        assert_eq!((&a / b.clone()).to_vec(), expected);
        assert_eq!((a.clone() / b.clone()).to_vec(), expected);
    }
}

// Module: Matmul

mod matmul {
    use super::*;

    #[test]
    fn matmul_2d_known_values() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_abs_diff_eq!(c.to_vec()[0], 14.0, epsilon = 1e-5);
        assert_abs_diff_eq!(c.to_vec()[1], 32.0, epsilon = 1e-5);
    }

    #[test]
    fn matmul_vec_dot() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
        assert_abs_diff_eq!(a.matmul(&b).item(), 32.0, epsilon = 1e-5);
    }

    #[test]
    fn matmul_mat_vec() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn matmul_vec_mat() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.to_vec(), vec![3.0, 2.0, 2.0]);
    }

    #[test]
    fn matmul_identity() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let eye = Tensor::eye(2);
        let result = a.matmul(&eye);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "matmul shape mismatch")]
    fn matmul_incompatible_panics() {
        let a = Tensor::ones(&[2, 3]);
        let b = Tensor::ones(&[4, 2]);
        a.matmul(&b);
    }

    #[test]
    fn matmul_large() {
        let a = Tensor::rand_uniform(&[64, 128], 0.0, 1.0, Some(42));
        let b = Tensor::rand_uniform(&[128, 32], 0.0, 1.0, Some(43));
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[64, 32]);
        // Verify no NaN
        for &v in c.to_vec().iter() {
            assert!(!v.is_nan());
        }
    }
}

// Module: Numerical Edge Cases

mod numerical_edge_cases {
    use super::*;

    #[test]
    fn nan_propagation_add() {
        let a = Tensor::from_vec(vec![1.0, f32::NAN], &[2]);
        let b = Tensor::from_vec(vec![2.0, 3.0], &[2]);
        let c = &a + &b;
        assert!(!c.to_vec()[0].is_nan());
        assert!(c.to_vec()[1].is_nan());
    }

    #[test]
    fn nan_propagation_mul() {
        let a = Tensor::from_vec(vec![f32::NAN], &[1]);
        let b = Tensor::from_vec(vec![0.0], &[1]);
        let c = &a * &b;
        assert!(c.item().is_nan(), "NaN * 0 should be NaN");
    }

    #[test]
    fn inf_arithmetic() {
        let a = Tensor::from_vec(vec![f32::INFINITY], &[1]);
        let b = Tensor::from_vec(vec![1.0], &[1]);
        assert!((&a + &b).item().is_infinite());
        assert!((&a * &b).item().is_infinite());
    }

    #[test]
    fn inf_minus_inf_is_nan() {
        let a = Tensor::from_vec(vec![f32::INFINITY], &[1]);
        let b = Tensor::from_vec(vec![f32::INFINITY], &[1]);
        let c = &a - &b;
        assert!(c.item().is_nan(), "Inf - Inf should be NaN");
    }

    #[test]
    fn large_tensor_creation() {
        let t = Tensor::zeros(&[1000, 1000]);
        assert_eq!(t.numel(), 1_000_000);
        assert_abs_diff_eq!(t.sum().item(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn large_tensor_operations() {
        let a = Tensor::ones(&[500, 500]);
        let b = Tensor::ones(&[500, 500]);
        let c = &a + &b;
        assert_abs_diff_eq!(c.mean().item(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn subnormal_values() {
        let tiny = f32::MIN_POSITIVE * 0.5;
        let t = Tensor::from_vec(vec![tiny], &[1]);
        assert!(t.item() > 0.0, "Subnormal should be positive");
    }

    #[test]
    fn display_does_not_panic() {
        let tensors = vec![
            Tensor::scalar(1.0),
            Tensor::zeros(&[3]),
            Tensor::ones(&[2, 3]),
            Tensor::zeros(&[2, 3, 4]),
        ];
        for t in &tensors {
            let s = format!("{}", t);
            assert!(s.contains("Tensor("));
        }
    }
}

// Module: Thread Safety (Compile-time assertions)

mod thread_safety {
    use super::*;

    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}

    #[test]
    fn tensor_is_send() {
        _assert_send::<Tensor>();
    }

    #[test]
    fn tensor_is_sync() {
        _assert_sync::<Tensor>();
    }
}

// Module: Random Tensor Tests

mod random_tensors {
    use super::*;

    #[test]
    fn rand_uniform_in_range() {
        let t = Tensor::rand_uniform(&[1000], 0.0, 1.0, Some(42));
        for &v in t.to_vec().iter() {
            assert!((0.0..1.0).contains(&v), "Value {} out of [0, 1)", v);
        }
    }

    #[test]
    fn rand_normal_statistics() {
        let t = Tensor::rand_normal(&[10000], 0.0, 1.0, Some(42));
        let mean = t.mean().item();
        let std = t.std_dev().item();
        assert!(mean.abs() < 0.1, "Mean {} too far from 0", mean);
        assert!((std - 1.0).abs() < 0.1, "Std {} too far from 1", std);
    }

    #[test]
    fn rand_reproducibility() {
        let t1 = Tensor::rand_uniform(&[10, 10], 0.0, 1.0, Some(999));
        let t2 = Tensor::rand_uniform(&[10, 10], 0.0, 1.0, Some(999));
        assert_eq!(t1.to_vec(), t2.to_vec());
    }

    #[test]
    fn rand_different_seeds_differ() {
        let t1 = Tensor::rand_uniform(&[100], 0.0, 1.0, Some(1));
        let t2 = Tensor::rand_uniform(&[100], 0.0, 1.0, Some(2));
        assert_ne!(t1.to_vec(), t2.to_vec());
    }

    #[test]
    fn xavier_uniform_bounds() {
        let shape = &[256, 128];
        let t = Tensor::xavier_uniform(shape, Some(42));
        let a = (6.0_f32 / (128.0 + 256.0)).sqrt();
        for &v in t.to_vec().iter() {
            assert!(v >= -a && v <= a, "Xavier value {} out of bounds ±{}", v, a);
        }
    }

    #[test]
    fn kaiming_normal_statistics() {
        let t = Tensor::kaiming_normal(&[500, 200], Some(42));
        let expected_std = (2.0_f32 / 200.0).sqrt();
        let actual_std = t.std_dev().item();
        assert!(
            (actual_std - expected_std).abs() < 0.03,
            "Kaiming std {} too far from {}",
            actual_std,
            expected_std
        );
    }
}

// Module: Property-Based (proptest)

mod proptest_fuzz {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid shapes (1-4 dims, 1-20 per dim)
    fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(1usize..=20, 1..=4)
    }

    proptest! {
        #[test]
        fn zeros_shape_matches(shape in shape_strategy()) {
            let t = Tensor::zeros(&shape);
            prop_assert_eq!(t.shape(), shape.as_slice());
            prop_assert_eq!(t.numel(), shape.iter().product::<usize>());
        }

        #[test]
        fn ones_all_ones(shape in shape_strategy()) {
            let t = Tensor::ones(&shape);
            for v in t.to_vec() {
                prop_assert_eq!(v, 1.0);
            }
        }

        #[test]
        fn add_commutative(
            shape in shape_strategy(),
            seed1 in 0u64..10000,
            seed2 in 0u64..10000,
        ) {
            let a = Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed1));
            let b = Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed2));
            let ab = &a + &b;
            let ba = &b + &a;
            for (x, y) in ab.to_vec().iter().zip(ba.to_vec().iter()) {
                prop_assert!((x - y).abs() < 1e-6);
            }
        }

        #[test]
        fn mul_commutative(
            shape in shape_strategy(),
            seed1 in 0u64..10000,
            seed2 in 0u64..10000,
        ) {
            let a = Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed1));
            let b = Tensor::rand_uniform(&shape, -1.0, 1.0, Some(seed2));
            let ab = &a * &b;
            let ba = &b * &a;
            for (x, y) in ab.to_vec().iter().zip(ba.to_vec().iter()) {
                prop_assert!((x - y).abs() < 1e-6);
            }
        }

        #[test]
        fn add_zero_identity(shape in shape_strategy(), seed in 0u64..10000) {
            let a = Tensor::rand_uniform(&shape, -10.0, 10.0, Some(seed));
            let zero = Tensor::zeros(&shape);
            let result = &a + &zero;
            for (x, y) in result.to_vec().iter().zip(a.to_vec().iter()) {
                prop_assert!((x - y).abs() < 1e-6);
            }
        }

        #[test]
        fn mul_one_identity(shape in shape_strategy(), seed in 0u64..10000) {
            let a = Tensor::rand_uniform(&shape, -10.0, 10.0, Some(seed));
            let one = Tensor::ones(&shape);
            let result = &a * &one;
            for (x, y) in result.to_vec().iter().zip(a.to_vec().iter()) {
                prop_assert!((x - y).abs() < 1e-6);
            }
        }

        #[test]
        fn reshape_preserves_numel(shape in shape_strategy(), seed in 0u64..10000) {
            let t = Tensor::rand_uniform(&shape, 0.0, 1.0, Some(seed));
            let numel = t.numel();
            let flat = t.flatten();
            prop_assert_eq!(flat.numel(), numel);
            prop_assert_eq!(flat.shape(), &[numel]);
        }

        #[test]
        fn neg_involution(shape in shape_strategy(), seed in 0u64..10000) {
            let a = Tensor::rand_uniform(&shape, -10.0, 10.0, Some(seed));
            let double_neg = a.neg().neg();
            for (x, y) in double_neg.to_vec().iter().zip(a.to_vec().iter()) {
                prop_assert!((x - y).abs() < 1e-5);
            }
        }

        #[test]
        fn sub_self_is_zero(shape in shape_strategy(), seed in 0u64..10000) {
            let a = Tensor::rand_uniform(&shape, -10.0, 10.0, Some(seed));
            let result = &a - &a;
            for v in result.to_vec() {
                prop_assert!((v).abs() < 1e-6, "a - a should be 0, got {}", v);
            }
        }
    }
}

// Module: PartialEq

mod equality {
    use super::*;

    #[test]
    fn equal_tensors() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        assert_eq!(a, b);
    }

    #[test]
    fn unequal_values() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 3.0], &[2]);
        assert_ne!(a, b);
    }

    #[test]
    fn unequal_shapes() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        assert_ne!(a, b);
    }

    #[test]
    fn clone_preserves_equality() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = a.clone();
        assert_eq!(a, b);
    }
}
