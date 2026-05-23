//! Proximal Policy Optimization (PPO) with clipped surrogate objective.
//!
//! ## Algorithm
//!
//! PPO constrains policy updates using a clipped importance ratio:
//!
//! ```text
//! r(θ) = π(a|s; θ) / π_old(a|s; θ_old)
//!
//! L_clip = -min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)
//! L_value = 0.5 · (V(s) - G_t)²
//! L_entropy = -c_ent · H[π]   (discrete only; continuous uses log_prob as proxy)
//! L_total = L_clip + c_vf · L_value + L_entropy
//! ```
//!
//! ## Architecture
//!
//! Two variants are provided:
//!
//! - **`PPODiscrete`**: Shared actor-critic trunk with discrete action head.
//!   Uses `ActorCriticNet` and `RolloutBuffer`/`RolloutBatch`.
//!
//! - **`PPOContinuous`**: Separate GaussianPolicy actor and value critic.
//!   Uses `ContinuousRolloutBuffer`/`ContinuousRolloutBatch`.
//!
//! ## Mini-Batch Training
//!
//! PPO typically performs multiple epochs of mini-batch updates on the same
//! rollout data. Mini-batches are filled from shuffled row indices into a
//! pre-allocated structure to minimize allocation.
//!
//! ## Design Decisions
//!
//! - **Advantage standardization**: Advantages are normalized (zero-mean, unit-var)
//!   before computing the surrogate loss to stabilize training.
//! - **Separate PPODiscrete / PPOContinuous**: Avoids generic complexity and keeps
//!   each variant's code straightforward.
//! - **`push_with_log_prob`**: The rollout buffer stores old log-probs at collection
//!   time, avoiding a separate forward pass during training.

use rand::seq::SliceRandom;
use rand::Rng;
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::loss::mse_loss;
use rustforge_nn::{Linear, Module, ReLU, Sequential};
use rustforge_tensor::Tensor;

use crate::agent::a2c::ActorCriticNet;
use crate::agent::gaussian_policy::GaussianPolicy;
use crate::agent::utils::clamp_var;
use crate::buffer::{ContinuousRolloutBatch, RolloutBatch};

/// Numerically stable log-softmax via composite autograd ops.
fn log_softmax_var(logits: &Variable) -> Variable {
    let max_val = Variable::from_tensor(logits.data().max_axis(1, true).unwrap());
    let shifted = logits - &max_val;
    let exp_vals = shifted.exp();
    let sum_exp = exp_vals.sum_axis(1, true);
    let log_sum_exp = sum_exp.log();
    &shifted - &log_sum_exp
}

/// Shared configuration for PPO.
pub struct PPOConfig {
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Hidden layer size.
    pub hidden_dim: usize,
    /// Learning rate for Adam.
    pub lr: f32,
    /// Discount factor γ.
    pub gamma: f32,
    /// GAE λ.
    pub gae_lambda: f32,
    /// Clipping epsilon ε for surrogate objective.
    pub clip_eps: f32,
    /// Value loss coefficient.
    pub value_coef: f32,
    /// Entropy bonus coefficient (used in discrete PPO).
    pub entropy_coef: f32,
    /// Number of PPO epochs per rollout.
    pub ppo_epochs: usize,
    /// Mini-batch size for PPO updates.
    pub mini_batch_size: usize,
}

impl Default for PPOConfig {
    fn default() -> Self {
        PPOConfig {
            obs_dim: 4,
            hidden_dim: 64,
            lr: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            ppo_epochs: 4,
            mini_batch_size: 64,
        }
    }
}

fn standardize_active_advantages(advantages: &Tensor, size: usize) -> Vec<f32> {
    let adv_all = advantages.to_vec();
    let adv_data = &adv_all[..size];
    let mean_adv: f32 = adv_data.iter().sum::<f32>() / size as f32;
    let var_adv: f32 = adv_data.iter().map(|a| (a - mean_adv).powi(2)).sum::<f32>() / size as f32;
    let std_adv = (var_adv + 1e-8).sqrt();
    adv_data.iter().map(|a| (a - mean_adv) / std_adv).collect()
}

struct PPODiscreteMiniBatch {
    states: Tensor,
    returns: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
}

impl PPODiscreteMiniBatch {
    fn new(capacity: usize, obs_dim: usize) -> Self {
        Self {
            states: Tensor::zeros(&[capacity, obs_dim]),
            returns: Tensor::zeros(&[capacity, 1]),
            advantages: Tensor::zeros(&[capacity, 1]),
            old_log_probs: Tensor::zeros(&[capacity, 1]),
        }
    }
}

struct PPOContinuousMiniBatch {
    states: Tensor,
    actions: Tensor,
    returns: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
}

impl PPOContinuousMiniBatch {
    fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            states: Tensor::zeros(&[capacity, obs_dim]),
            actions: Tensor::zeros(&[capacity, act_dim]),
            returns: Tensor::zeros(&[capacity, 1]),
            advantages: Tensor::zeros(&[capacity, 1]),
            old_log_probs: Tensor::zeros(&[capacity, 1]),
        }
    }
}

// ─────────────────────────────────────────────────────────
//  PPODiscrete — discrete action spaces
// ─────────────────────────────────────────────────────────

/// PPO configuration specific to discrete action spaces.
pub struct PPODiscreteConfig {
    /// Base PPO config.
    pub base: PPOConfig,
    /// Number of discrete actions.
    pub num_actions: usize,
}

/// PPO agent for discrete action spaces.
///
/// Uses `ActorCriticNet` (shared trunk) and `RolloutBuffer`/`RolloutBatch`.
pub struct PPODiscrete {
    /// Shared actor-critic network.
    net: ActorCriticNet,
    /// Adam optimizer.
    optimizer: Adam,
    /// Configuration.
    config: PPODiscreteConfig,
}

impl PPODiscrete {
    /// Creates a new PPO discrete agent.
    pub fn new(config: PPODiscreteConfig) -> Self {
        let net = ActorCriticNet::new(
            config.base.obs_dim,
            config.base.hidden_dim,
            config.num_actions,
        );
        let optimizer = Adam::new(net.parameters(), config.base.lr);
        PPODiscrete {
            net,
            optimizer,
            config,
        }
    }

    /// Selects an action by sampling from the policy distribution.
    ///
    /// Returns `(action_index, log_prob, value)`.
    pub fn select_action(&self, state: &[f32]) -> (usize, f32, f32) {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.base.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        let (logits, value) = self.net.forward(&state_var);

        // Softmax for sampling
        let log_probs = log_softmax_var(&logits);
        let probs_data = log_probs.data().to_vec();

        // Sample from categorical distribution
        let mut rng = rand::thread_rng();
        let probs: Vec<f32> = probs_data.iter().map(|lp| lp.exp()).collect();
        let total: f32 = probs.iter().sum();
        let sample: f32 = rng.gen::<f32>() * total;
        let mut cumsum = 0.0;
        let mut action = 0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if sample <= cumsum {
                action = i;
                break;
            }
        }

        let log_prob = probs_data[action];
        let value_scalar = value.data().to_vec()[0];

        (action, log_prob, value_scalar)
    }

    /// Returns the value estimate V(s) for a state.
    pub fn value_of(&self, state: &[f32]) -> f32 {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.base.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        let (_logits, value) = self.net.forward(&state_var);
        let result = value.data().to_vec()[0];
        result
    }

    /// Trains on a rollout batch for one PPO epoch.
    ///
    /// Returns `(policy_loss, value_loss, entropy)` averaged over mini-batches.
    pub fn train_on_batch(&mut self, batch: &RolloutBatch) -> (f32, f32, f32) {
        let n = batch.size;
        if n == 0 {
            return (0.0, 0.0, 0.0);
        }

        let mini_batch_size = self.config.base.mini_batch_size.min(n);
        let obs_dim = batch.states.shape()[1];

        // Standardize advantages
        let norm_adv = standardize_active_advantages(&batch.advantages, n);

        // Cache batch values to flat vectors once at the beginning
        let states_flat = batch.states.to_vec();
        let returns_flat = batch.returns.to_vec();
        let old_lp_flat = batch.old_log_probs.to_vec();

        // Preallocate mini-batch structures
        let mut mb = PPODiscreteMiniBatch::new(mini_batch_size, obs_dim);
        let mut mb_actions = vec![0; mini_batch_size];

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();

        let mut total_policy_loss = 0.0_f32;
        let mut total_value_loss = 0.0_f32;
        let mut total_entropy = 0.0_f32;
        let mut num_updates = 0;

        for _epoch in 0..self.config.base.ppo_epochs {
            indices.shuffle(&mut rng);

            for start in (0..n).step_by(mini_batch_size) {
                let end = (start + mini_batch_size).min(n);
                let mb_size = end - start;
                if mb_size == 0 {
                    continue;
                }

                let mb_indices = &indices[start..end];

                // Write into pre-allocated mini-batch in-place
                {
                    let mb_states_mut = mb.states.data_mut();
                    let mb_returns_mut = mb.returns.data_mut();
                    let mb_advantages_mut = mb.advantages.data_mut();
                    let mb_old_log_probs_mut = mb.old_log_probs.data_mut();

                    let mb_states_slice = mb_states_mut.as_slice_mut().unwrap();
                    let mb_returns_slice = mb_returns_mut.as_slice_mut().unwrap();
                    let mb_advantages_slice = mb_advantages_mut.as_slice_mut().unwrap();
                    let mb_old_log_probs_slice = mb_old_log_probs_mut.as_slice_mut().unwrap();

                    for (b, &idx) in mb_indices.iter().enumerate() {
                        let src_offset = idx * obs_dim;
                        let dst_offset = b * obs_dim;
                        mb_states_slice[dst_offset..dst_offset + obs_dim]
                            .copy_from_slice(&states_flat[src_offset..src_offset + obs_dim]);

                        mb_returns_slice[b] = returns_flat[idx];
                        mb_advantages_slice[b] = norm_adv[idx];
                        mb_old_log_probs_slice[b] = old_lp_flat[idx];

                        mb_actions[b] = batch.actions[idx];
                    }
                }

                // Slice the preallocated tensors and wrap in Variables
                let mb_states = Variable::from_tensor(mb.states.slice_axis(0, 0, mb_size).unwrap());
                let mb_returns =
                    Variable::from_tensor(mb.returns.slice_axis(0, 0, mb_size).unwrap());
                let mb_advantages =
                    Variable::from_tensor(mb.advantages.slice_axis(0, 0, mb_size).unwrap());
                let mb_old_log_probs =
                    Variable::from_tensor(mb.old_log_probs.slice_axis(0, 0, mb_size).unwrap());
                let mb_actions_slice = &mb_actions[..mb_size];

                // Forward pass
                self.optimizer.zero_grad();
                let (logits, values) = self.net.forward(&mb_states);
                let log_probs = log_softmax_var(&logits);

                // Gather log π(a_taken|s) for each sample
                let new_log_probs = log_probs.gather(1, mb_actions_slice);

                // Importance ratio r(θ)
                let ratio = (&new_log_probs - &mb_old_log_probs).exp();

                // Clipped surrogate loss
                let surr1 = &ratio * &mb_advantages;
                let clip_ratio = clamp_var(
                    &ratio,
                    1.0 - self.config.base.clip_eps,
                    1.0 + self.config.base.clip_eps,
                );
                let surr2 = &clip_ratio * &mb_advantages;

                // L_clip = -min(surr1, surr2) → we want to maximize, so negate
                use crate::agent::utils::elementwise_min_var;
                let min_surr = elementwise_min_var(&surr1, &surr2);
                let policy_loss = &min_surr.mean() * (-1.0);

                // Value loss
                let value_loss = mse_loss(&values, &mb_returns);

                // Entropy bonus (H = -Σ π·log π)
                let probs = log_probs.exp();
                let entropy_terms = &probs * &log_probs;
                let entropy = &entropy_terms.sum() * (-1.0 / mb_size as f32);

                // Total loss
                let total_loss = &(&policy_loss + &(&value_loss * self.config.base.value_coef))
                    - &(&entropy * self.config.base.entropy_coef);

                total_loss.backward();
                self.optimizer.step();

                total_policy_loss += policy_loss.data().to_vec()[0];
                total_value_loss += value_loss.data().to_vec()[0];
                total_entropy += entropy.data().to_vec()[0];
                num_updates += 1;
            }
        }

        if num_updates > 0 {
            (
                total_policy_loss / num_updates as f32,
                total_value_loss / num_updates as f32,
                total_entropy / num_updates as f32,
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

// ─────────────────────────────────────────────────────────
//  PPOContinuous — continuous action spaces
// ─────────────────────────────────────────────────────────

/// PPO configuration specific to continuous action spaces.
pub struct PPOContinuousConfig {
    /// Base PPO config.
    pub base: PPOConfig,
    /// Action dimensionality.
    pub act_dim: usize,
    /// Action lower bounds (per dimension).
    pub action_low: Vec<f32>,
    /// Action upper bounds (per dimension).
    pub action_high: Vec<f32>,
}

/// PPO agent for continuous action spaces.
///
/// Uses `GaussianPolicy` as the actor and a separate MLP value critic.
/// Training consumes `ContinuousRolloutBatch` data.
pub struct PPOContinuous {
    /// Gaussian policy actor.
    pub actor: GaussianPolicy,
    /// Value critic: Linear → ReLU → Linear → ReLU → Linear(1).
    critic: Sequential,
    /// Adam optimizer for actor parameters.
    actor_optimizer: Adam,
    /// Adam optimizer for critic parameters.
    critic_optimizer: Adam,
    /// Configuration.
    config: PPOContinuousConfig,
}

impl PPOContinuous {
    /// Creates a new PPO continuous agent.
    pub fn new(config: PPOContinuousConfig) -> Self {
        let actor = GaussianPolicy::new(
            config.base.obs_dim,
            config.base.hidden_dim,
            config.act_dim,
            &config.action_low,
            &config.action_high,
        );

        let critic = Sequential::new(vec![
            Box::new(Linear::new(config.base.obs_dim, config.base.hidden_dim)),
            Box::new(ReLU),
            Box::new(Linear::new(config.base.hidden_dim, config.base.hidden_dim)),
            Box::new(ReLU),
            Box::new(Linear::new(config.base.hidden_dim, 1)),
        ]);

        let actor_optimizer = Adam::new(actor.parameters(), config.base.lr);
        let critic_optimizer = Adam::new(critic.parameters(), config.base.lr);

        PPOContinuous {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            config,
        }
    }

    /// Samples an action from the policy.
    ///
    /// Returns `(action_vec, log_prob_scalar, value_scalar)`.
    pub fn select_action(&self, state: &[f32]) -> (Vec<f32>, f32, f32) {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.base.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);

        let (action, log_prob) = self.actor.sample(&state_var);
        let value = self.critic.forward(&state_var);

        let action_data = action.data().to_vec();
        let log_prob_scalar = log_prob.data().to_vec()[0];
        let value_scalar = value.data().to_vec()[0];

        (action_data, log_prob_scalar, value_scalar)
    }

    /// Returns the value estimate V(s) for a state.
    pub fn value_of(&self, state: &[f32]) -> f32 {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.base.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        let value = self.critic.forward(&state_var);
        let result = value.data().to_vec()[0];
        result
    }

    /// Trains on a continuous rollout batch for one PPO epoch.
    ///
    /// Returns `(policy_loss, value_loss)` averaged over mini-batches.
    pub fn train_on_batch(&mut self, batch: &ContinuousRolloutBatch) -> (f32, f32) {
        let n = batch.size;
        if n == 0 {
            return (0.0, 0.0);
        }

        let mini_batch_size = self.config.base.mini_batch_size.min(n);
        let obs_dim = batch.states.shape()[1];
        let act_dim = batch.actions.shape()[1];

        // Standardize advantages
        let norm_adv = standardize_active_advantages(&batch.advantages, n);

        // Cache batch values to flat vectors once at the beginning
        let states_flat = batch.states.to_vec();
        let actions_flat = batch.actions.to_vec();
        let returns_flat = batch.returns.to_vec();
        let old_lp_flat = batch.old_log_probs.to_vec();

        // Preallocate mini-batch structures
        let mut mb = PPOContinuousMiniBatch::new(mini_batch_size, obs_dim, act_dim);

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();

        let mut total_policy_loss = 0.0_f32;
        let mut total_value_loss = 0.0_f32;
        let mut num_updates = 0;

        for _epoch in 0..self.config.base.ppo_epochs {
            indices.shuffle(&mut rng);

            for start in (0..n).step_by(mini_batch_size) {
                let end = (start + mini_batch_size).min(n);
                let mb_size = end - start;
                if mb_size == 0 {
                    continue;
                }

                let mb_indices = &indices[start..end];

                // Write into pre-allocated mini-batch in-place
                {
                    let mb_states_mut = mb.states.data_mut();
                    let mb_actions_mut = mb.actions.data_mut();
                    let mb_returns_mut = mb.returns.data_mut();
                    let mb_advantages_mut = mb.advantages.data_mut();
                    let mb_old_log_probs_mut = mb.old_log_probs.data_mut();

                    let mb_states_slice = mb_states_mut.as_slice_mut().unwrap();
                    let mb_actions_slice = mb_actions_mut.as_slice_mut().unwrap();
                    let mb_returns_slice = mb_returns_mut.as_slice_mut().unwrap();
                    let mb_advantages_slice = mb_advantages_mut.as_slice_mut().unwrap();
                    let mb_old_log_probs_slice = mb_old_log_probs_mut.as_slice_mut().unwrap();

                    for (b, &idx) in mb_indices.iter().enumerate() {
                        let src_s_offset = idx * obs_dim;
                        let dst_s_offset = b * obs_dim;
                        mb_states_slice[dst_s_offset..dst_s_offset + obs_dim]
                            .copy_from_slice(&states_flat[src_s_offset..src_s_offset + obs_dim]);

                        let src_a_offset = idx * act_dim;
                        let dst_a_offset = b * act_dim;
                        mb_actions_slice[dst_a_offset..dst_a_offset + act_dim]
                            .copy_from_slice(&actions_flat[src_a_offset..src_a_offset + act_dim]);

                        mb_returns_slice[b] = returns_flat[idx];
                        mb_advantages_slice[b] = norm_adv[idx];
                        mb_old_log_probs_slice[b] = old_lp_flat[idx];
                    }
                }

                // Slice the preallocated tensors and wrap in Variables
                let mb_states = Variable::from_tensor(mb.states.slice_axis(0, 0, mb_size).unwrap());
                let mb_actions =
                    Variable::from_tensor(mb.actions.slice_axis(0, 0, mb_size).unwrap());
                let mb_returns =
                    Variable::from_tensor(mb.returns.slice_axis(0, 0, mb_size).unwrap());
                let mb_advantages =
                    Variable::from_tensor(mb.advantages.slice_axis(0, 0, mb_size).unwrap());
                let mb_old_log_probs =
                    Variable::from_tensor(mb.old_log_probs.slice_axis(0, 0, mb_size).unwrap());

                // ── Actor update ──
                self.actor_optimizer.zero_grad();
                let new_log_probs = self.actor.log_prob_from_action(&mb_states, &mb_actions);

                let ratio = (&new_log_probs - &mb_old_log_probs).exp();

                let surr1 = &ratio * &mb_advantages;
                let clip_ratio = clamp_var(
                    &ratio,
                    1.0 - self.config.base.clip_eps,
                    1.0 + self.config.base.clip_eps,
                );
                let surr2 = &clip_ratio * &mb_advantages;

                use crate::agent::utils::elementwise_min_var;
                let min_surr = elementwise_min_var(&surr1, &surr2);
                let policy_loss = &min_surr.mean() * (-1.0);

                policy_loss.backward();
                self.actor_optimizer.step();

                // ── Critic update ──
                self.critic_optimizer.zero_grad();
                let values = self.critic.forward(&mb_states);
                let value_loss = mse_loss(&values, &mb_returns);

                value_loss.backward();
                self.critic_optimizer.step();

                total_policy_loss += policy_loss.data().to_vec()[0];
                total_value_loss += value_loss.data().to_vec()[0];
                num_updates += 1;
            }
        }

        if num_updates > 0 {
            (
                total_policy_loss / num_updates as f32,
                total_value_loss / num_updates as f32,
            )
        } else {
            (0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppo_discrete_construction() {
        let config = PPODiscreteConfig {
            base: PPOConfig {
                obs_dim: 4,
                hidden_dim: 32,
                ..PPOConfig::default()
            },
            num_actions: 2,
        };
        let agent = PPODiscrete::new(config);
        let state = [0.1_f32, 0.2, 0.3, 0.4];
        let (action, log_prob, value) = agent.select_action(&state);
        assert!(action < 2);
        assert!(log_prob.is_finite());
        assert!(value.is_finite());
    }

    #[test]
    fn ppo_discrete_train_produces_finite_losses() {
        let config = PPODiscreteConfig {
            base: PPOConfig {
                obs_dim: 4,
                hidden_dim: 32,
                mini_batch_size: 4,
                ppo_epochs: 2,
                ..PPOConfig::default()
            },
            num_actions: 2,
        };
        let mut agent = PPODiscrete::new(config);

        // Create a small rollout batch
        let mut buf = crate::buffer::RolloutBuffer::new(8, 4);
        for i in 0..8 {
            let state = [i as f32 * 0.1; 4];
            let (action, log_prob, value) = agent.select_action(&state);
            buf.push_with_log_prob(&state, action, 1.0, value, 0.0, log_prob);
        }
        buf.compute_returns_and_advantages(0.99, 0.95, 0.0);
        let batch = buf.to_batch();

        let (pl, vl, ent) = agent.train_on_batch(&batch);
        assert!(pl.is_finite(), "policy_loss is not finite: {}", pl);
        assert!(vl.is_finite(), "value_loss is not finite: {}", vl);
        assert!(ent.is_finite(), "entropy is not finite: {}", ent);
    }

    #[test]
    fn ppo_continuous_construction() {
        let config = PPOContinuousConfig {
            base: PPOConfig {
                obs_dim: 4,
                hidden_dim: 32,
                ..PPOConfig::default()
            },
            act_dim: 2,
            action_low: vec![-1.0, -1.0],
            action_high: vec![1.0, 1.0],
        };
        let agent = PPOContinuous::new(config);
        let state = [0.1_f32, 0.2, 0.3, 0.4];
        let (action, log_prob, value) = agent.select_action(&state);
        assert_eq!(action.len(), 2);
        assert!(log_prob.is_finite());
        assert!(value.is_finite());
    }

    #[test]
    fn ppo_continuous_train_produces_finite_losses() {
        let config = PPOContinuousConfig {
            base: PPOConfig {
                obs_dim: 4,
                hidden_dim: 32,
                mini_batch_size: 4,
                ppo_epochs: 2,
                ..PPOConfig::default()
            },
            act_dim: 2,
            action_low: vec![-1.0, -1.0],
            action_high: vec![1.0, 1.0],
        };
        let mut agent = PPOContinuous::new(config);

        let mut buf = crate::buffer::ContinuousRolloutBuffer::new(8, 4, 2);
        let mut batch = ContinuousRolloutBatch::new(8, 4, 2);
        for i in 0..8 {
            let state = [i as f32 * 0.1; 4];
            let (action, log_prob, value) = agent.select_action(&state);
            buf.push(&state, &action, 1.0, value, 0.0, log_prob);
        }
        buf.compute_returns_and_advantages(0.99, 0.95, 0.0);
        buf.fill_batch(&mut batch);

        let (pl, vl) = agent.train_on_batch(&batch);
        assert!(pl.is_finite(), "policy_loss is not finite: {}", pl);
        assert!(vl.is_finite(), "value_loss is not finite: {}", vl);
    }

    #[test]
    fn ppo_discrete_value_of() {
        let config = PPODiscreteConfig {
            base: PPOConfig {
                obs_dim: 4,
                hidden_dim: 32,
                ..PPOConfig::default()
            },
            num_actions: 2,
        };
        let agent = PPODiscrete::new(config);
        let v = agent.value_of(&[0.1, 0.2, 0.3, 0.4]);
        assert!(v.is_finite());
    }

    #[test]
    fn ppo_continuous_value_of() {
        let config = PPOContinuousConfig {
            base: PPOConfig {
                obs_dim: 4,
                hidden_dim: 32,
                ..PPOConfig::default()
            },
            act_dim: 2,
            action_low: vec![-1.0, -1.0],
            action_high: vec![1.0, 1.0],
        };
        let agent = PPOContinuous::new(config);
        let v = agent.value_of(&[0.1, 0.2, 0.3, 0.4]);
        assert!(v.is_finite());
    }

    #[test]
    fn ppo_continuous_ignores_inactive_tail_rows() {
        // Create preallocated batch with capacity = 10
        let mut batch = ContinuousRolloutBatch::new(10, 4, 2);

        // Populate first 3 elements of batch.advantages with values [1.0, 2.0, 3.0]
        {
            let adv_mut = batch.advantages.data_mut();
            let slice = adv_mut.as_slice_mut().unwrap();
            slice[0] = 1.0;
            slice[1] = 2.0;
            slice[2] = 3.0;

            // Populate indices 3..10 with massive dummy values (e.g. 1_000_000.0)
            for val in &mut slice[3..10] {
                *val = 1_000_000.0;
            }
        }

        // size = 3
        batch.size = 3;

        // Call standardize_active_advantages
        let norm_adv = standardize_active_advantages(&batch.advantages, batch.size);

        // For size = 3, data is [1.0, 2.0, 3.0]
        // mean = 2.0
        // var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3 = 0.66666667
        // std = (0.66666667 + 1e-8)^0.5 ≈ 0.81649658
        // norm = [(1-2)/std, (2-2)/std, (3-2)/std] = [-1.22474487, 0.0, 1.22474487]

        assert_eq!(norm_adv.len(), 3);
        approx::assert_relative_eq!(norm_adv[0], -1.224_744_9, epsilon = 1e-5);
        approx::assert_relative_eq!(norm_adv[1], 0.0, epsilon = 1e-5);
        approx::assert_relative_eq!(norm_adv[2], 1.224_744_9, epsilon = 1e-5);
    }
}
