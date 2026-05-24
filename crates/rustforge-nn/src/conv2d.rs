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
    bias: Option<Variable>,

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

        if self.bias.is_some() {
            panic!("Conv2d bias is not supported yet");
        }

        out
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

    #[test]
    #[should_panic(expected = "Conv2d bias is not supported yet")]
    fn test_conv2d_bias_panics() {
        let mut conv = Conv2d::new(3, 16, (5, 5));
        conv.bias = Some(Variable::new(Tensor::zeros(&[16]), false));
        let x = Variable::new(Tensor::ones(&[2, 3, 32, 32]), false);
        let _y = conv.forward(&x);
    }
}
