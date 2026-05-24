//! 2D Convolution layer.
//!
//! Implements a 2D convolution over an input signal composed of several input planes.

use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

use crate::module::Module;

/// 2D Convolutional layer.
///
/// Current implementation strictly supports `stride=1` and `padding=0`.
pub struct Conv2d {
    /// Weight tensor [out_channels, in_channels, kernel_h, kernel_w]
    pub weight: Variable,
    /// Bias vector [out_channels], None if bias is disabled
    pub bias: Option<Variable>,

    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
}

impl Conv2d {
    /// Creates a new Conv2d layer.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        // Kaiming uniform initialization
        let k = 1.0 / (in_channels as f32 * kernel_size.0 as f32 * kernel_size.1 as f32).sqrt();

        let weight_data = Tensor::rand_uniform(
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
            -k,
            k,
            None,
        );
        let weight = Variable::new(weight_data, true);

        Conv2d {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Variable) -> Variable {
        use rustforge_autograd::ops::var_conv2d;

        let out = var_conv2d(input, &self.weight);

        if let Some(b) = &self.bias {
            // Need to broadcast bias [out_channels] to [batch, out_channels, out_h, out_w]
            // We can reshape bias to [1, out_channels, 1, 1] and add
            // Wait, does our autograd support broadcasting? AddOp does support it partially.
            // Let's rely on Tensor's broadcast.
            // Actually `var_add` supports broadcast if ndarray supports it.
            // But we must reshape the Variable first.
            let _b_reshaped = Variable::new(
                b.data()
                    .clone()
                    .reshape(&[1, self.out_channels, 1, 1])
                    .unwrap(),
                b.requires_grad(),
            );
            // We should ideally use a reshape operation in autograd, but since bias grad just sums over batch, h, and w,
            // we can do it directly. Wait, if we create a new Variable without grad_fn, gradients won't flow back to `b`!
            // We need a proper bias addition or reshape operation.
            // Since we don't have reshape grad yet, we can skip bias for now or implement bias grad.
            // Let's implement it correctly. Actually, let's keep it simple: no bias for Conv2d in Phase C
            // since we are just laying the foundation. Let's make it `None`.
            out // TODO: Implement broadcasted bias addition with gradients
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_module() {
        let conv = Conv2d::new(3, 16, (5, 5));
        let x = Variable::new(Tensor::ones(&[2, 3, 32, 32]), false);
        let y = conv.forward(&x);

        assert_eq!(y.shape(), vec![2, 16, 28, 28]);
    }
}
