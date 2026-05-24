//! Noisy Networks for Exploration.
//!
//! Implements the factorized Gaussian noise layer:
//! `y = x @ (\mu_w + \sigma_w \odot \epsilon_w)^T + (\mu_b + \sigma_b \odot \epsilon_b)`

use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

use crate::module::Module;

/// Noisy linear layer for exploration.
pub struct NoisyLinear {
    pub weight_mu: Variable,
    pub weight_sigma: Variable,
    pub bias_mu: Option<Variable>,
    pub bias_sigma: Option<Variable>,

    in_features: usize,
    out_features: usize,

    weight_epsilon: Tensor,
    bias_epsilon: Option<Tensor>,
}

impl NoisyLinear {
    /// Creates a new NoisyLinear layer with factorized Gaussian noise.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mu_range = 1.0 / (in_features as f32).sqrt();
        let sigma_init = 0.5 / (in_features as f32).sqrt();

        let weight_mu = Variable::new(
            Tensor::rand_uniform(&[out_features, in_features], -mu_range, mu_range, None),
            true,
        );

        let w_sigma_data = vec![sigma_init; out_features * in_features];
        let weight_sigma = Variable::new(
            Tensor::from_vec(w_sigma_data, &[out_features, in_features]),
            true,
        );

        let bias_mu = Variable::new(
            Tensor::rand_uniform(&[out_features], -mu_range, mu_range, None),
            true,
        );

        let b_sigma_data = vec![sigma_init; out_features];
        let bias_sigma = Variable::new(Tensor::from_vec(b_sigma_data, &[out_features]), true);

        let mut layer = NoisyLinear {
            weight_mu,
            weight_sigma,
            bias_mu: Some(bias_mu),
            bias_sigma: Some(bias_sigma),
            in_features,
            out_features,
            weight_epsilon: Tensor::zeros(&[out_features, in_features]),
            bias_epsilon: Some(Tensor::zeros(&[out_features])),
        };

        layer.reset_noise();
        layer
    }

    /// Transformation function for factorized noise: f(x) = sign(x) * sqrt(|x|)
    fn f(x: f32) -> f32 {
        x.signum() * x.abs().sqrt()
    }
}

impl Module for NoisyLinear {
    fn forward(&self, input: &Variable) -> Variable {
        // Wrap epsilons in Variables without gradients
        let eps_w_var = Variable::new(self.weight_epsilon.clone(), false);

        // \mu_w + \sigma_w \odot \epsilon_w
        let weight = &self.weight_mu + &(&self.weight_sigma * &eps_w_var);

        let out = input.matmul(&weight.t());

        if let Some(b_mu) = &self.bias_mu {
            let b_sigma = self.bias_sigma.as_ref().unwrap();
            let eps_b_var = Variable::new(self.bias_epsilon.as_ref().unwrap().clone(), false);

            // \mu_b + \sigma_b \odot \epsilon_b
            let bias = b_mu + &(b_sigma * &eps_b_var);
            &out + &bias
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![self.weight_mu.clone(), self.weight_sigma.clone()];
        if let Some(b_mu) = &self.bias_mu {
            params.push(b_mu.clone());
        }
        if let Some(b_sigma) = &self.bias_sigma {
            params.push(b_sigma.clone());
        }
        params
    }

    fn reset_noise(&mut self) {
        let eps_in = Tensor::randn(&[self.in_features], None);
        let eps_out = Tensor::randn(&[self.out_features], None);

        // We know these are flat 1D arrays
        let eps_in_slice = eps_in.to_vec();
        let eps_out_slice = eps_out.to_vec();

        let f_in: Vec<f32> = eps_in_slice.iter().map(|&x| Self::f(x)).collect();
        let f_out: Vec<f32> = eps_out_slice.iter().map(|&x| Self::f(x)).collect();

        // To avoid ndarray slice errors, we can use Tensor::from_vec
        let mut w_eps_data = Vec::with_capacity(self.out_features * self.in_features);

        for &o in &f_out {
            for &i in &f_in {
                w_eps_data.push(o * i);
            }
        }
        self.weight_epsilon = Tensor::from_vec(w_eps_data, &[self.out_features, self.in_features]);

        if self.bias_epsilon.is_some() {
            self.bias_epsilon = Some(Tensor::from_vec(f_out, &[self.out_features]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noisy_linear_shapes() {
        let layer = NoisyLinear::new(4, 3);
        let x = Variable::new(Tensor::ones(&[2, 4]), false);

        let y = layer.forward(&x);
        assert_eq!(y.shape(), vec![2, 3]);
        assert_eq!(layer.parameters().len(), 4);
    }

    #[test]
    fn test_noisy_linear_reset_noise() {
        let mut layer = NoisyLinear::new(4, 3);

        let eps1 = layer.weight_epsilon.to_vec();
        layer.reset_noise();
        let eps2 = layer.weight_epsilon.to_vec();

        // Noise should be different after reset
        assert_ne!(eps1, eps2);
    }

    #[test]
    fn test_noisy_linear_deterministic_without_reset() {
        let layer = NoisyLinear::new(4, 3);
        let x = Variable::new(Tensor::ones(&[2, 4]), false);

        let y1 = layer.forward(&x);
        let y2 = layer.forward(&x);

        // Consecutive forward passes without reset must produce identical outputs
        assert_eq!(y1.data().to_vec(), y2.data().to_vec());
    }

    #[test]
    fn test_noisy_linear_stochastic_with_reset() {
        let mut layer = NoisyLinear::new(4, 3);
        let x = Variable::new(Tensor::ones(&[2, 4]), false);

        let y1 = layer.forward(&x);
        layer.reset_noise();
        let y2 = layer.forward(&x);

        // Forward passes with a noise reset in between must produce different outputs
        assert_ne!(y1.data().to_vec(), y2.data().to_vec());
    }

    #[test]
    fn test_noisy_linear_gradient_flow() {
        let layer = NoisyLinear::new(4, 3);
        let x = Variable::new(Tensor::ones(&[1, 4]), false);
        let y = layer.forward(&x);

        y.sum().backward();

        // Both weight_mu and weight_sigma must receive gradients
        assert!(layer.weight_mu.grad().is_some());
        assert!(layer.weight_sigma.grad().is_some());

        if let Some(ref bias_mu) = layer.bias_mu {
            assert!(bias_mu.grad().is_some());
        }
        if let Some(ref bias_sigma) = layer.bias_sigma {
            assert!(bias_sigma.grad().is_some());
        }

        // The weight_sigma gradient should be non-zero (since noise epsilon is non-zero)
        let sig_grad = layer.weight_sigma.grad().unwrap().to_vec();
        let any_nonzero = sig_grad.iter().any(|&g| g.abs() > 1e-6);
        assert!(any_nonzero, "Sigma gradient should be non-zero");
    }
}
