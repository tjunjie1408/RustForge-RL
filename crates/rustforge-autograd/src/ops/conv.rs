//! 2D Convolution operation and gradient tracking.

use crate::graph::{GradFn, GradInputs, GradOutputs};
use crate::variable::Variable;
use rustforge_tensor::Tensor;
use smallvec::smallvec;

/// Gradient function for 2D Convolution.
pub struct Conv2dGrad {
    pub input: Variable,
    pub weight: Variable,
}

impl GradFn for Conv2dGrad {
    fn inputs(&self) -> GradInputs {
        smallvec![self.input.clone(), self.weight.clone()]
    }

    fn backward(&self, grad_output: &Tensor) -> GradOutputs {
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();
        let grad_out_shape = grad_output.shape();

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let out_channels = weight_shape[0];
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];

        let out_h = grad_out_shape[2];
        let out_w = grad_out_shape[3];

        let mut grad_w = Tensor::zeros(&weight_shape);
        let mut grad_input = Tensor::zeros(&input_shape);

        let input_ref = self.input.data();
        let weight_ref = self.weight.data();

        // `Tensor::to_vec` makes a flat copy which is easier to index and prevents
        // borrowing issues with `Ref`. It's slightly slower but safer for this MVP.
        let input_vec = input_ref.to_vec();
        let weight_vec = weight_ref.to_vec();
        let grad_out_vec = grad_output.to_vec();

        {
            // We can safely use `data_mut().as_slice_mut()` on our owned Tensors `grad_w` and `grad_input`.
            // Let's use `to_vec` to collect results and then construct Tensors. Actually, we can
            // get `as_slice_mut` without `data_mut()` if we just use a flat vec.
            let mut gw_vec = vec![0.0; grad_w.numel()];
            let mut gi_vec = vec![0.0; grad_input.numel()];

            for b in 0..batch_size {
                for oc in 0..out_channels {
                    for ic in 0..in_channels {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let go = grad_out_vec
                                    [((b * out_channels + oc) * out_h + oh) * out_w + ow];
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let ih = oh + kh;
                                        let iw = ow + kw;

                                        let i_val = input_vec
                                            [((b * in_channels + ic) * in_h + ih) * in_w + iw];
                                        let w_val =
                                            weight_vec[((oc * in_channels + ic) * kernel_h + kh)
                                                * kernel_w
                                                + kw];

                                        let w_idx = ((oc * in_channels + ic) * kernel_h + kh)
                                            * kernel_w
                                            + kw;
                                        let i_idx =
                                            ((b * in_channels + ic) * in_h + ih) * in_w + iw;

                                        gw_vec[w_idx] += i_val * go;
                                        gi_vec[i_idx] += go * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Re-assign to Tensors
            grad_w = Tensor::from_vec(gw_vec, &weight_shape);
            grad_input = Tensor::from_vec(gi_vec, &input_shape);
        }

        smallvec![grad_input, grad_w]
    }
}

/// Computes 2D convolution for stride=1, padding=0.
///
/// - `input` shape: `[batch_size, in_channels, in_h, in_w]`
/// - `weight` shape: `[out_channels, in_channels, kernel_h, kernel_w]`
pub fn var_conv2d(input: &Variable, weight: &Variable) -> Variable {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    assert_eq!(
        input_shape.len(),
        4,
        "Input must be 4D [batch, channels, h, w]"
    );
    assert_eq!(
        weight_shape.len(),
        4,
        "Weight must be 4D [out_c, in_c, kh, kw]"
    );

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    let out_channels = weight_shape[0];
    assert_eq!(weight_shape[1], in_channels, "Conv2d in_channels mismatch");
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    assert!(in_h >= kernel_h, "Input height < kernel height");
    assert!(in_w >= kernel_w, "Input width < kernel width");

    let out_h = in_h - kernel_h + 1;
    let out_w = in_w - kernel_w + 1;

    let input_ref = input.data();
    let weight_ref = weight.data();
    let input_vec = input_ref.to_vec();
    let weight_vec = weight_ref.to_vec();

    let mut out_vec = vec![0.0; batch_size * out_channels * out_h * out_w];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh + kh;
                                let iw = ow + kw;

                                let i_val =
                                    input_vec[((b * in_channels + ic) * in_h + ih) * in_w + iw];
                                let w_val = weight_vec
                                    [((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];

                                sum += i_val * w_val;
                            }
                        }
                    }
                    out_vec[((b * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    }

    let out = Tensor::from_vec(out_vec, &[batch_size, out_channels, out_h, out_w]);

    let requires_grad = input.requires_grad() || weight.requires_grad();

    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(Conv2dGrad {
            input: input.clone(),
            weight: weight.clone(),
        }))
    } else {
        None
    };

    Variable::from_grad_fn(out, requires_grad, grad_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_forward() {
        let input = Variable::new(Tensor::ones(&[1, 1, 3, 3]), false);
        let weight = Variable::new(Tensor::ones(&[1, 1, 2, 2]), false);

        let out = var_conv2d(&input, &weight);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);

        let out_data = out.data().to_vec();
        for &val in &out_data {
            assert!((val - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_conv2d_backward() {
        let input = Variable::new(Tensor::ones(&[1, 1, 3, 3]), true);
        let weight = Variable::new(Tensor::ones(&[1, 1, 2, 2]), true);

        let out = var_conv2d(&input, &weight);
        let loss = out.sum();
        loss.backward();

        let grad_w = weight.grad().unwrap().to_vec();
        for &gw in &grad_w {
            assert!((gw - 4.0).abs() < 1e-5);
        }
    }
}
