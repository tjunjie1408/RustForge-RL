//! Gaussian (Normal) policy for continuous action spaces.
//!
//! ## Overview
//!
//! `GaussianPolicy` wraps a neural network that outputs the mean and log-std of
//! a diagonal Gaussian distribution over actions. It supports:
//!
//! - **`sample`**: Reparameterized sampling with tanh squashing → `(action, log_prob)`.
//! - **`log_prob_from_action`**: Recomputes log π(a|s) for actions stored in a buffer.
//! - **`deterministic_action`**: Returns the mean action (no noise), squashed through tanh.
//!
//! ## Architecture
//!
//! ```text
//! Linear(obs_dim, hidden) → ReLU → Linear(hidden, hidden) → ReLU    [trunk]
//!     ├── Linear(hidden, act_dim)    [mean_head → μ]
//!     └── Linear(hidden, act_dim)    [log_std_head → clamp to [-20, 2]]
//!
//! u = μ + ε · exp(log_std),    ε ~ N(0, I)     (reparameterization trick)
//! a = tanh(u) * scale + bias                     (squash + rescale to action bounds)
//! ```
//!
//! ## Log-Probability Correction
//!
//! The tanh squashing requires a Jacobian correction to the log-probability:
//!
//! ```text
//! log π(a|s) = Σ_i [log N(u_i | μ_i, σ_i) - log(1 - tanh²(u_i)) - log(scale_i)]
//! ```
//!
//! ## Design Decisions
//!
//! - **Separate trunk/heads**: Allows shared features while keeping mean and std
//!   independent (important for SAC where the std head learns exploration).
//! - **Clamped log_std**: [-20, 2] prevents numerical issues (σ too small or too large).
//! - **`log_prob_from_action`**: For PPO, buffer stores squashed/scaled actions.
//!   This method inverts the transform to recover `u`, then evaluates log π under
//!   the current policy. The inversion uses `atanh(clamp(normalized, -1+ε, 1-ε))`
//!   to avoid NaN at boundaries.

use rand::Rng;
use rustforge_autograd::Variable;
use rustforge_nn::{Linear, Module, ReLU, Sequential};
use rustforge_tensor::Tensor;

use crate::agent::utils::clamp_var;

/// Minimum log standard deviation (prevents σ from collapsing to zero).
const LOG_STD_MIN: f32 = -20.0;
/// Maximum log standard deviation (prevents σ from exploding).
const LOG_STD_MAX: f32 = 2.0;
/// Small epsilon for numerical safety in atanh inversion.
const ATANH_EPS: f32 = 1e-6;

/// Neural network that outputs mean and log_std for a Gaussian policy.
///
/// ```text
/// trunk: Linear → ReLU → Linear → ReLU
/// mean_head:    Linear(hidden, act_dim) → μ
/// log_std_head: Linear(hidden, act_dim) → clamp(log σ, -20, 2)
/// ```
pub struct GaussianPolicyNet {
    /// Shared feature extraction trunk.
    trunk: Sequential,
    /// Mean output head.
    mean_head: Linear,
    /// Log standard deviation output head.
    log_std_head: Linear,
}

impl GaussianPolicyNet {
    /// Creates a new Gaussian policy network.
    ///
    /// ## Arguments
    /// - `obs_dim`: Observation space dimensionality.
    /// - `hidden_dim`: Hidden layer size.
    /// - `act_dim`: Action space dimensionality.
    pub fn new(obs_dim: usize, hidden_dim: usize, act_dim: usize) -> Self {
        let trunk = Sequential::new(vec![
            Box::new(Linear::new(obs_dim, hidden_dim)),
            Box::new(ReLU),
            Box::new(Linear::new(hidden_dim, hidden_dim)),
            Box::new(ReLU),
        ]);
        let mean_head = Linear::new(hidden_dim, act_dim);
        let log_std_head = Linear::new(hidden_dim, act_dim);

        GaussianPolicyNet {
            trunk,
            mean_head,
            log_std_head,
        }
    }

    /// Forward pass: returns (mean, clamped_log_std).
    ///
    /// - `mean`: `[batch, act_dim]` — unbounded mean.
    /// - `log_std`: `[batch, act_dim]` — clamped to `[LOG_STD_MIN, LOG_STD_MAX]`.
    pub fn forward(&self, state: &Variable) -> (Variable, Variable) {
        let features = self.trunk.forward(state);
        let mean = self.mean_head.forward(&features);
        let raw_log_std = self.log_std_head.forward(&features);
        let log_std = clamp_var(&raw_log_std, LOG_STD_MIN, LOG_STD_MAX);
        (mean, log_std)
    }

    /// Returns all trainable parameters.
    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.trunk.parameters();
        params.extend(self.mean_head.parameters());
        params.extend(self.log_std_head.parameters());
        params
    }
}

/// Gaussian policy with tanh squashing and action rescaling.
///
/// Wraps `GaussianPolicyNet` and handles:
/// - Reparameterized sampling with tanh squashing
/// - Action rescaling to arbitrary `[low, high]` bounds
/// - Log-probability computation with Jacobian correction
///
/// ## Action Transform
///
/// ```text
/// action = tanh(u) * scale + bias
///   where scale = (high - low) / 2
///         bias  = (high + low) / 2
/// ```
pub struct GaussianPolicy {
    /// The underlying neural network.
    pub net: GaussianPolicyNet,
    /// Per-dimension action scale: `(high - low) / 2`.
    scale: Vec<f32>,
    /// Per-dimension action bias: `(high + low) / 2`.
    bias: Vec<f32>,
    /// Action dimensionality.
    act_dim: usize,
}

impl GaussianPolicy {
    /// Creates a new Gaussian policy.
    ///
    /// ## Arguments
    /// - `obs_dim`: Observation space dimensionality.
    /// - `hidden_dim`: Hidden layer size.
    /// - `act_dim`: Action space dimensionality.
    /// - `action_low`: Per-dimension action lower bounds.
    /// - `action_high`: Per-dimension action upper bounds.
    pub fn new(
        obs_dim: usize,
        hidden_dim: usize,
        act_dim: usize,
        action_low: &[f32],
        action_high: &[f32],
    ) -> Self {
        assert_eq!(action_low.len(), act_dim);
        assert_eq!(action_high.len(), act_dim);

        let scale: Vec<f32> = action_low
            .iter()
            .zip(action_high.iter())
            .map(|(l, h)| (h - l) / 2.0)
            .collect();
        let bias: Vec<f32> = action_low
            .iter()
            .zip(action_high.iter())
            .map(|(l, h)| (h + l) / 2.0)
            .collect();

        GaussianPolicy {
            net: GaussianPolicyNet::new(obs_dim, hidden_dim, act_dim),
            scale,
            bias,
            act_dim,
        }
    }

    /// Samples an action using reparameterization trick + tanh squashing.
    ///
    /// Returns `(action_variable, log_prob_variable)` where:
    /// - `action`: `[batch, act_dim]` — squashed and scaled action.
    /// - `log_prob`: `[batch, 1]` — log π(a|s) with Jacobian correction.
    ///
    /// Gradient flows through both outputs (reparameterization trick).
    pub fn sample(&self, state: &Variable) -> (Variable, Variable) {
        let (mean, log_std) = self.net.forward(state);
        let std = log_std.exp();

        let shape = mean.shape();
        let numel: usize = shape.iter().product();

        // ε ~ N(0, I)
        let mut rng = rand::thread_rng();
        let noise_data: Vec<f32> = (0..numel)
            .map(|_| sample_standard_normal(&mut rng))
            .collect();
        let noise = Variable::from_tensor(Tensor::from_vec(noise_data, &shape));

        // u = μ + ε · σ  (reparameterization)
        let u = &mean + &(&noise * &std);

        // tanh squashing
        let tanh_u = u.tanh_();

        // log π(a|s) = Σ_i [log N(u_i|μ_i,σ_i) - log(1 - tanh²(u_i)) - log(scale_i)]
        let log_prob = self.compute_log_prob(&u, &mean, &log_std, &tanh_u);

        // action = tanh(u) * scale + bias
        let action = self.apply_action_scaling(&tanh_u);

        (action, log_prob)
    }

    /// Computes log π(a|s) for actions stored in a buffer.
    ///
    /// This inverts the tanh+scale transform to recover `u`, then evaluates the
    /// log-probability under the current policy parameters.
    ///
    /// ## Arguments
    /// - `state`: `[batch, obs_dim]` — observations.
    /// - `action`: `[batch, act_dim]` — squashed/scaled actions from the buffer.
    ///   These are treated as constants (no gradient flows through them).
    ///
    /// ## Returns
    /// `log_prob`: `[batch, 1]` — log π(a|s) under the *current* policy.
    pub fn log_prob_from_action(&self, state: &Variable, action: &Variable) -> Variable {
        let (mean, log_std) = self.net.forward(state);

        let batch = mean.shape()[0];

        // Invert scaling: normalized = (action - bias) / scale
        let bias_tensor = Variable::from_tensor(Tensor::from_vec(
            self.bias.repeat(batch),
            &[batch, self.act_dim],
        ));
        let scale_tensor = Variable::from_tensor(Tensor::from_vec(
            self.scale.repeat(batch),
            &[batch, self.act_dim],
        ));

        // action is from buffer — detach to prevent gradient flow through stored actions
        let action_detached = action.detach();
        let normalized = &(&action_detached - &bias_tensor) / &scale_tensor;

        // clamp to (-1+ε, 1-ε) for atanh safety, then recover u
        let normalized_data = normalized.data();
        let clamped: Vec<f32> = normalized_data
            .to_vec()
            .iter()
            .map(|x| x.clamp(-1.0 + ATANH_EPS, 1.0 - ATANH_EPS))
            .collect();
        drop(normalized_data);

        let u_data: Vec<f32> = clamped.iter().map(|x| x.atanh()).collect();
        let u = Variable::from_tensor(Tensor::from_vec(u_data, &[batch, self.act_dim]));

        let tanh_u_data: Vec<f32> = clamped; // tanh(atanh(x)) = x (clamped normalized)
        let tanh_u = Variable::from_tensor(Tensor::from_vec(tanh_u_data, &[batch, self.act_dim]));

        self.compute_log_prob(&u, &mean, &log_std, &tanh_u)
    }

    /// Returns the deterministic action: `tanh(μ) * scale + bias`.
    ///
    /// No noise, no gradient. Used for evaluation/deployment.
    pub fn deterministic_action(&self, state: &Variable) -> Vec<f32> {
        let (mean, _log_std) = self.net.forward(state);
        let mean_data = mean.data();
        let values = mean_data.to_vec();
        let batch = mean.shape()[0];

        let mut actions = Vec::with_capacity(batch * self.act_dim);
        for b in 0..batch {
            for d in 0..self.act_dim {
                let u = values[b * self.act_dim + d];
                let a = u.tanh() * self.scale[d] + self.bias[d];
                actions.push(a);
            }
        }
        actions
    }

    /// Returns all trainable parameters.
    pub fn parameters(&self) -> Vec<Variable> {
        self.net.parameters()
    }

    /// Computes log π(a|s) given pre-squash values and distribution params.
    ///
    /// ```text
    /// log π = Σ_i [log N(u_i | μ_i, σ_i) - log(1 - tanh²(u_i)) - log(scale_i)]
    /// ```
    fn compute_log_prob(
        &self,
        u: &Variable,
        mean: &Variable,
        log_std: &Variable,
        tanh_u: &Variable,
    ) -> Variable {
        let std = log_std.exp();

        // log N(u | μ, σ) = -0.5 * ((u - μ)/σ)² - log(σ) - 0.5*log(2π)
        let diff = u - mean;
        let z = &diff / &std; // (u - μ) / σ
        let log_normal = &(&z.pow(2.0) * (-0.5)) - log_std;
        // Subtract 0.5*log(2π) as a constant
        let shape = u.shape();
        let numel: usize = shape.iter().product();
        let half_log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
        let half_log_2pi_const =
            Variable::from_tensor(Tensor::from_vec(vec![half_log_2pi; numel], &shape));
        let log_normal = &log_normal - &half_log_2pi_const;

        // Jacobian correction: -log(1 - tanh²(u)) = -log(1 - tanh_u²)
        let tanh_sq = tanh_u * tanh_u;
        let numel_ones = Variable::from_tensor(Tensor::from_vec(vec![1.0; numel], &shape));
        let one_minus_tanh_sq = &numel_ones - &tanh_sq;
        // Add small epsilon for numerical safety
        let eps_const = Variable::from_tensor(Tensor::from_vec(vec![1e-6_f32; numel], &shape));
        let safe_one_minus = &one_minus_tanh_sq + &eps_const;
        let log_jacobian = safe_one_minus.log(); // log(1 - tanh²(u) + ε)

        // log(scale) correction
        let batch = shape[0];
        let log_scale_data: Vec<f32> = self
            .scale
            .iter()
            .map(|s| s.ln())
            .cycle()
            .take(numel)
            .collect();
        let log_scale =
            Variable::from_tensor(Tensor::from_vec(log_scale_data, &[batch, self.act_dim]));

        // Per-dimension log prob: log_normal - log_jacobian - log_scale
        let per_dim = &(&log_normal - &log_jacobian) - &log_scale;

        // Sum over action dimensions → [batch, 1]
        per_dim.sum_axis(1, true)
    }

    /// Applies tanh squashing and action scaling: `tanh(u) * scale + bias`.
    fn apply_action_scaling(&self, tanh_u: &Variable) -> Variable {
        let shape = tanh_u.shape();
        let batch = shape[0];

        let scale_tensor = Variable::from_tensor(Tensor::from_vec(
            self.scale.repeat(batch),
            &[batch, self.act_dim],
        ));
        let bias_tensor = Variable::from_tensor(Tensor::from_vec(
            self.bias.repeat(batch),
            &[batch, self.act_dim],
        ));

        &(tanh_u * &scale_tensor) + &bias_tensor
    }
}

/// Samples from N(0,1) using Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f32 {
    let u1: f32 = rng.gen_range(1e-7..1.0);
    let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy() -> GaussianPolicy {
        GaussianPolicy::new(4, 64, 2, &[-1.0, -2.0], &[1.0, 2.0])
    }

    #[test]
    fn construction_and_forward() {
        let policy = make_policy();
        let state = Variable::from_tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]));
        let (mean, log_std) = policy.net.forward(&state);
        assert_eq!(mean.shape(), vec![1, 2]);
        assert_eq!(log_std.shape(), vec![1, 2]);

        // log_std should be clamped
        let ls = log_std.data().to_vec();
        for v in &ls {
            assert!(
                *v >= LOG_STD_MIN - 1e-5 && *v <= LOG_STD_MAX + 1e-5,
                "log_std {} out of range [{}, {}]",
                v,
                LOG_STD_MIN,
                LOG_STD_MAX
            );
        }
    }

    #[test]
    fn sample_returns_correct_shapes() {
        let policy = make_policy();
        let state = Variable::from_tensor(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        ));
        let (action, log_prob) = policy.sample(&state);
        assert_eq!(action.shape(), vec![2, 2]); // [batch=2, act_dim=2]
        assert_eq!(log_prob.shape(), vec![2, 1]); // [batch=2, 1]
    }

    #[test]
    fn sample_actions_within_bounds() {
        let policy = make_policy();
        let state = Variable::from_tensor(Tensor::from_vec(vec![0.1; 4 * 50], &[50, 4]));
        let (action, _) = policy.sample(&state);
        let action_data = action.data().to_vec();

        for (i, v) in action_data.iter().enumerate() {
            let dim = i % 2;
            let (lo, hi) = if dim == 0 {
                (-1.0_f32, 1.0_f32)
            } else {
                (-2.0_f32, 2.0_f32)
            };
            assert!(
                *v > lo - 0.01 && *v < hi + 0.01,
                "action[{}] = {} outside [{}, {}]",
                i,
                v,
                lo,
                hi
            );
        }
    }

    #[test]
    fn log_prob_finite() {
        let policy = make_policy();
        let state = Variable::from_tensor(Tensor::from_vec(vec![0.5; 4], &[1, 4]));
        let (_, log_prob) = policy.sample(&state);
        let lp = log_prob.data().to_vec();
        for v in &lp {
            assert!(v.is_finite(), "log_prob is not finite: {}", v);
        }
    }

    #[test]
    fn deterministic_action_is_deterministic() {
        let policy = make_policy();
        let state = Variable::from_tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]));
        let a1 = policy.deterministic_action(&state);
        let a2 = policy.deterministic_action(&state);
        assert_eq!(a1, a2);
    }

    #[test]
    fn log_prob_from_action_finite_and_gradient_flows() {
        let policy = make_policy();
        let state = Variable::new(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[1, 4]), false);

        // Sample an action to use as buffer data
        let (action, _) = policy.sample(&state);
        let action_data = action.data().clone();

        // Now compute log_prob_from_action — gradient should flow to policy params
        let action_buffer = Variable::from_tensor(action_data);
        let log_prob = policy.log_prob_from_action(&state, &action_buffer);

        let lp = log_prob.data().to_vec();
        for v in &lp {
            assert!(v.is_finite(), "log_prob_from_action is not finite: {}", v);
        }

        // Verify gradient flows to policy parameters
        let loss = log_prob.sum();
        loss.backward();

        let params = policy.parameters();
        let has_grad = params.iter().any(|p| p.grad().is_some());
        assert!(
            has_grad,
            "No gradients flowed to policy parameters via log_prob_from_action"
        );
    }

    #[test]
    fn log_prob_from_action_boundary_nan_safety() {
        let policy = make_policy();
        let state = Variable::from_tensor(Tensor::from_vec(vec![0.0; 4], &[1, 4]));

        // Actions exactly at bounds (scale=[1,2], bias=[0,0])
        // For dim 0: low=-1, high=1 → boundary action = 1.0 or -1.0
        // For dim 1: low=-2, high=2 → boundary action = 2.0 or -2.0
        let boundary_action = Variable::from_tensor(Tensor::from_vec(vec![0.999, 1.999], &[1, 2]));
        let log_prob = policy.log_prob_from_action(&state, &boundary_action);
        let lp = log_prob.data().to_vec();
        for v in &lp {
            assert!(v.is_finite(), "log_prob at boundary is not finite: {}", v);
            assert!(!v.is_nan(), "log_prob at boundary is NaN");
        }
    }

    #[test]
    fn parameter_count() {
        let policy = make_policy();
        let params = policy.parameters();
        // trunk: Linear(4,64) → 4*64+64=320, Linear(64,64) → 64*64+64=4160
        // mean_head: Linear(64,2) → 64*2+2=130
        // log_std_head: Linear(64,2) → 64*2+2=130
        // Total params count = 8 variables (4 weight, 4 bias)
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn sample_gradient_flow_through_reparameterization() {
        let policy = make_policy();
        let state = Variable::from_tensor(Tensor::from_vec(vec![1.0; 4], &[1, 4]));
        let (action, _log_prob) = policy.sample(&state);

        // Loss on action — gradient should flow to policy params via reparam trick
        let action_loss = action.sum();
        action_loss.backward();

        let params = policy.parameters();
        let has_grad = params.iter().any(|p| p.grad().is_some());
        assert!(
            has_grad,
            "No gradients flowed through reparameterized sampling"
        );
    }
}
