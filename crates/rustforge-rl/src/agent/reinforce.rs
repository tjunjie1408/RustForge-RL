//! REINFORCE (Monte Carlo Policy Gradient) algorithm.
//!
//! ## Algorithm
//!
//! REINFORCE learns a stochastic policy π(a|s; θ) via the policy gradient theorem:
//!
//! ```text
//! ∇J(θ) = E[Σ_t ∇log π(a_t|s_t; θ) · G_t]
//! ```
//!
//! where G_t is the discounted return from timestep t.
//!
//! ## Key Design Decisions
//!
//! - **No epsilon-greedy**: Policy gradient algorithms explore via stochastic sampling
//!   from the policy distribution itself, not via epsilon-greedy.
//! - **No stored log_probs**: The rollout buffer stores only states/actions/rewards.
//!   At training time, the current policy re-forwards to produce differentiable
//!   `log π(a|s)` with a live computation graph.
//! - **Baseline**: Optional mean-subtraction of advantages (`G_t -= mean(G_t)`)
//!   to reduce variance. Enabled by default via `REINFORCEConfig::use_baseline`.
//! - **Log-softmax**: Uses the numerically stable composite pattern from
//!   `cross_entropy_loss` (shift by max, exp, sum, log) rather than a dedicated op.
//!
//! ## Usage with RolloutBuffer
//!
//! REINFORCE has no critic, so push `value=0.0` to the buffer and call
//! `compute_returns_and_advantages(gamma, lambda=1.0, last_value=0.0)`.
//! Result: `advantages == returns` (Monte Carlo discounted returns).

use rand::Rng;
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::{Linear, Module, ReLU, Sequential};
use rustforge_tensor::Tensor;

use crate::buffer::RolloutBatch;

/// Numerically stable log-softmax via composite autograd ops.
///
/// Replicates the pattern from `cross_entropy_loss`:
/// ```text
/// log_softmax(x) = (x - max(x)) - log(Σ exp(x - max(x)))
/// ```
///
/// `max(x)` is detached (constant shift, no gradient needed).
fn log_softmax_var(logits: &Variable) -> Variable {
    // Detach max for numerical stability (same pattern as cross_entropy_loss)
    let max_val = Variable::from_tensor(logits.data().max_axis(1, true).unwrap());
    let shifted = logits - &max_val;
    let exp_vals = shifted.exp();
    let sum_exp = exp_vals.sum_axis(1, true);
    let log_sum_exp = sum_exp.log();
    &shifted - &log_sum_exp
}

/// Configuration for the REINFORCE agent.
pub struct REINFORCEConfig {
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Number of discrete actions.
    pub num_actions: usize,
    /// Hidden layer size for the policy MLP.
    pub hidden_dim: usize,
    /// Learning rate for Adam optimizer.
    pub lr: f32,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Whether to subtract mean(advantages) as a baseline to reduce variance.
    pub use_baseline: bool,
}

impl Default for REINFORCEConfig {
    fn default() -> Self {
        REINFORCEConfig {
            obs_dim: 4,
            num_actions: 2,
            hidden_dim: 64,
            lr: 1e-3,
            gamma: 0.99,
            use_baseline: true,
        }
    }
}

/// REINFORCE (Monte Carlo Policy Gradient) agent.
///
/// Owns a policy network and Adam optimizer.
/// The policy outputs raw logits; softmax is applied for sampling and log-softmax
/// for the policy gradient loss.
pub struct REINFORCE {
    /// Policy network: outputs raw logits [batch, num_actions].
    policy_net: Sequential,
    /// Adam optimizer for the policy network.
    optimizer: Adam,
    /// Configuration.
    config: REINFORCEConfig,
}

impl REINFORCE {
    /// Creates a new REINFORCE agent.
    ///
    /// Builds a 2-layer MLP: `Linear(obs_dim, hidden) → ReLU → Linear(hidden, num_actions)`.
    pub fn new(config: REINFORCEConfig) -> Self {
        let policy_net = Sequential::new(vec![
            Box::new(Linear::new(config.obs_dim, config.hidden_dim)),
            Box::new(ReLU),
            Box::new(Linear::new(config.hidden_dim, config.num_actions)),
        ]);

        let optimizer = Adam::new(policy_net.parameters(), config.lr);

        REINFORCE {
            policy_net,
            optimizer,
            config,
        }
    }

    /// Forward pass: returns raw logits for the given state.
    ///
    /// The output is policy logits, NOT Q-values. For action selection,
    /// apply softmax and sample from the resulting categorical distribution.
    ///
    /// ## Arguments
    /// - `state`: Observation as a flat f32 slice, length = `obs_dim`.
    ///
    /// ## Returns
    /// `Variable` of shape `[1, num_actions]` — raw logits.
    pub fn forward(&self, state: &[f32]) -> Variable {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        self.policy_net.forward(&state_var)
    }

    /// Samples an action from the policy's categorical distribution.
    ///
    /// Applies softmax to logits and samples proportionally.
    ///
    /// ## Arguments
    /// - `logits`: Raw logits, shape `[num_actions]` or `[1, num_actions]`.
    /// - `rng`: External RNG for deterministic testing.
    ///
    /// ## Returns
    /// Sampled action index.
    pub fn sample_action(logits: &Tensor, rng: &mut impl Rng) -> usize {
        let flat = logits.to_vec();

        // Stable softmax
        let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = flat.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|e| e / sum_exp).collect();

        // Sample from categorical distribution
        let u: f32 = rng.gen();
        let mut cumulative = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return i;
            }
        }
        probs.len() - 1 // fallback for floating-point edge case
    }

    /// Convenience wrapper using thread_rng.
    pub fn sample_action_default(logits: &Tensor) -> usize {
        let mut rng = rand::thread_rng();
        Self::sample_action(logits, &mut rng)
    }

    /// Trains on a rollout batch using the REINFORCE policy gradient.
    ///
    /// ## Algorithm
    ///
    /// 1. Forward current policy on batch states → logits `[B, A]`
    /// 2. `log_softmax(logits)` → `log_probs [B, A]`
    /// 3. `gather(log_probs, 1, actions)` → `selected_log_probs [B, 1]`
    /// 4. If baseline: `advantages -= mean(advantages)`
    /// 5. `loss = -mean(selected_log_probs * detach(advantages))`
    /// 6. backward + step
    ///
    /// ## Returns
    /// The scalar loss value for logging.
    pub fn train_on_rollout(&mut self, batch: &RolloutBatch) -> f32 {
        // ── 1. Forward through current policy ──
        let states_var = Variable::from_tensor(batch.states.clone());
        let logits = self.policy_net.forward(&states_var); // [B, num_actions]

        // ── 2. Log-softmax (numerically stable composite) ──
        let log_probs = log_softmax_var(&logits); // [B, num_actions]

        // ── 3. Gather log-probs for taken actions ──
        let selected_log_probs = log_probs.gather(1, &batch.actions[..batch.size]); // [B, 1]

        // ── 4. Prepare advantages (detached, with optional baseline) ──
        let mut adv_vec = batch.advantages.to_vec();
        if self.config.use_baseline {
            let mean_adv: f32 = adv_vec.iter().sum::<f32>() / adv_vec.len() as f32;
            for a in adv_vec.iter_mut() {
                *a -= mean_adv;
            }
        }
        let advantages = Variable::from_tensor(Tensor::from_vec(adv_vec, &[batch.size, 1]));

        // ── 5. Policy gradient loss: -mean(log_prob * advantage) ──
        let weighted = &selected_log_probs * &advantages;
        let loss = -(weighted.mean());
        let loss_val = loss.data().item();

        // ── 6. Backward + optimizer step ──
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        loss_val
    }

    /// Returns a reference to the policy network.
    pub fn policy_net(&self) -> &Sequential {
        &self.policy_net
    }

    /// Returns the discount factor.
    pub fn gamma(&self) -> f32 {
        self.config.gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::RolloutBuffer;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;

    fn make_reinforce() -> REINFORCE {
        REINFORCE::new(REINFORCEConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 8,
            lr: 1e-2,
            gamma: 0.99,
            use_baseline: true,
        })
    }

    #[test]
    fn construction_and_forward() {
        let agent = make_reinforce();
        assert_eq!(agent.policy_net.parameters().len(), 4); // 2 layers × (W + b)

        let logits = agent.forward(&[1.0, 0.0]);
        assert_eq!(logits.shape(), vec![1, 2]);
    }

    #[test]
    fn sample_action_deterministic() {
        // With logits strongly favoring action 1, sampling should almost always pick 1
        let logits = Tensor::from_vec(vec![-100.0, 100.0], &[2]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..10 {
            let action = REINFORCE::sample_action(&logits, &mut rng);
            assert_eq!(action, 1, "Strong logits should always pick action 1");
        }
    }

    #[test]
    fn sample_action_distribution() {
        // Uniform logits → roughly uniform sampling
        let logits = Tensor::from_vec(vec![0.0, 0.0, 0.0], &[3]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut counts = [0u32; 3];

        for _ in 0..3000 {
            let action = REINFORCE::sample_action(&logits, &mut rng);
            counts[action] += 1;
        }

        for count in &counts {
            assert!(
                *count > 700,
                "Uniform logits should give roughly equal distribution, got {:?}",
                counts
            );
        }
    }

    #[test]
    fn log_softmax_var_numerical_stability() {
        // Large logits that would overflow without max subtraction
        let logits = Variable::new(
            Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3]),
            true,
        );
        let log_probs = log_softmax_var(&logits);
        let data = log_probs.data().to_vec();

        for val in &data {
            assert!(!val.is_nan(), "log_softmax should not produce NaN");
            assert!(!val.is_infinite(), "log_softmax should not produce Inf");
        }

        // log_softmax should be negative (log of probabilities < 1)
        for val in &data {
            assert!(*val <= 0.0, "log probabilities should be <= 0, got {}", val);
        }

        // Sum of softmax should be 1 → log_sum_exp should cancel correctly
        // Verify: log_softmax values should sum to proper log-probs
        let exp_sum: f32 = data.iter().map(|x| x.exp()).sum();
        assert_abs_diff_eq!(exp_sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn log_softmax_var_gradient_flows() {
        let logits = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), true);
        let log_probs = log_softmax_var(&logits);
        let loss = log_probs.sum();
        loss.backward();

        assert!(logits.grad().is_some(), "Gradient should flow through log_softmax");
    }

    #[test]
    fn train_on_rollout_produces_finite_loss() {
        let mut agent = make_reinforce();

        let mut buf = RolloutBuffer::new(10, 2);
        buf.push(&[1.0, 0.0], 0, 1.0, 0.0, 0.0);
        buf.push(&[0.0, 1.0], 1, -1.0, 0.0, 0.0);
        buf.push(&[0.5, 0.5], 0, 0.5, 0.0, 1.0);
        buf.compute_returns_and_advantages(0.99, 1.0, 0.0);

        let batch = buf.to_batch();
        let loss = agent.train_on_rollout(&batch);

        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
    }

    #[test]
    fn simple_mdp_learns_correct_action() {
        // Simple 2-action MDP: action 0 always gives reward=1, action 1 gives reward=0.
        // After training, policy should favor action 0.
        let mut agent = REINFORCE::new(REINFORCEConfig {
            obs_dim: 1,
            num_actions: 2,
            hidden_dim: 8,
            lr: 5e-3,
            gamma: 0.0, // gamma=0 → each step's return is just its reward
            use_baseline: true,
        });

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let state = [1.0_f32]; // constant state

        for _ in 0..50 {
            let mut buf = RolloutBuffer::new(20, 1);

            // Collect a mini-episode
            for _ in 0..20 {
                let logits = agent.forward(&state);
                let action = REINFORCE::sample_action(&logits.data(), &mut rng);
                let reward = if action == 0 { 1.0 } else { 0.0 };
                buf.push(&state, action, reward, 0.0, 1.0); // each step is terminal
            }

            buf.compute_returns_and_advantages(0.0, 1.0, 0.0);
            let batch = buf.to_batch();
            let loss = agent.train_on_rollout(&batch);
            assert!(loss.is_finite());
        }

        // After training, check policy preference
        let logits = agent.forward(&state);
        let logits_data = logits.data().to_vec();

        // Action 0 logit should be higher than action 1
        assert!(
            logits_data[0] > logits_data[1],
            "After training, action 0 (reward=1) should have higher logit than action 1 (reward=0). Got: {:?}",
            logits_data
        );
    }
}
