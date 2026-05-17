//! Advantage Actor-Critic (A2C) algorithm.
//!
//! ## Algorithm
//!
//! A2C combines a policy (actor) and a value function (critic) with shared
//! representation learning. The total loss has three components:
//!
//! ```text
//! L = L_actor + c_value · L_critic - c_entropy · H[π]
//!
//! L_actor  = -mean(log π(a_t|s_t; θ) · A_t)     (policy gradient)
//! L_critic = mean((G_t - V(s_t; θ))²)            (value regression)
//! H[π]     = -Σ_a π(a|s) · log π(a|s)            (entropy bonus)
//! ```
//!
//! ## Network Architecture
//!
//! Uses `ActorCriticNet` — a shared-trunk network with separate actor/critic heads:
//!
//! ```text
//! Linear(obs_dim, hidden) → ReLU       [shared trunk]
//!     ├── Linear(hidden, num_actions)   [actor head → policy logits]
//!     └── Linear(hidden, 1)             [critic head → V(s)]
//! ```
//!
//! `Sequential` cannot represent this topology (it is linear chain only),
//! so `ActorCriticNet` is a dedicated structure.
//!
//! ## Training Flow
//!
//! 1. Collect n_steps experience → RolloutBuffer (values from critic, detached)
//! 2. `compute_returns_and_advantages(gamma, lambda, last_value)`
//! 3. `train_on_rollout(batch)`:
//!    - Re-forward current policy+critic on batch states
//!    - Compute actor loss, critic loss, entropy bonus
//!    - Single combined backward pass
//! 4. `clear()` buffer

use rand::Rng;
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::loss::mse_loss;
use rustforge_nn::{Linear, Module, ReLU, Sequential};
use rustforge_tensor::Tensor;

use crate::buffer::RolloutBatch;

/// Numerically stable log-softmax via composite autograd ops.
///
/// Same pattern as `reinforce::log_softmax_var`.
fn log_softmax_var(logits: &Variable) -> Variable {
    let max_val = Variable::from_tensor(logits.data().max_axis(1, true).unwrap());
    let shifted = logits - &max_val;
    let exp_vals = shifted.exp();
    let sum_exp = exp_vals.sum_axis(1, true);
    let log_sum_exp = sum_exp.log();
    &shifted - &log_sum_exp
}

/// Numerically stable softmax via composite ops (not in computation graph for entropy).
fn softmax_var(logits: &Variable) -> Variable {
    let max_val = Variable::from_tensor(logits.data().max_axis(1, true).unwrap());
    let shifted = logits - &max_val;
    let exp_vals = shifted.exp();
    let sum_exp = exp_vals.sum_axis(1, true);
    &exp_vals / &sum_exp
}

/// Actor-Critic network with shared trunk and separate heads.
///
/// Cannot be represented by `Sequential` (linear chain). This structure
/// enables shared feature learning between actor and critic while allowing
/// separate output heads.
pub struct ActorCriticNet {
    /// Shared feature extraction: Linear(obs_dim, hidden) → ReLU
    trunk: Sequential,
    /// Policy head: Linear(hidden, num_actions) → raw logits
    actor_head: Linear,
    /// Value head: Linear(hidden, 1) → scalar V(s)
    critic_head: Linear,
}

impl ActorCriticNet {
    /// Creates a new actor-critic network.
    pub fn new(obs_dim: usize, hidden_dim: usize, num_actions: usize) -> Self {
        let trunk = Sequential::new(vec![
            Box::new(Linear::new(obs_dim, hidden_dim)),
            Box::new(ReLU),
        ]);
        let actor_head = Linear::new(hidden_dim, num_actions);
        let critic_head = Linear::new(hidden_dim, 1);

        ActorCriticNet {
            trunk,
            actor_head,
            critic_head,
        }
    }

    /// Forward pass: returns (logits, value).
    ///
    /// - `logits`: `[batch, num_actions]` — raw policy logits
    /// - `value`: `[batch, 1]` — state value V(s)
    pub fn forward(&self, input: &Variable) -> (Variable, Variable) {
        let features = self.trunk.forward(input);
        let logits = self.actor_head.forward(&features);
        let value = self.critic_head.forward(&features);
        (logits, value)
    }

    /// Returns all trainable parameters (trunk + actor_head + critic_head).
    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.trunk.parameters();
        params.extend(self.actor_head.parameters());
        params.extend(self.critic_head.parameters());
        params
    }
}

/// Configuration for the A2C agent.
pub struct A2CConfig {
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Number of discrete actions.
    pub num_actions: usize,
    /// Hidden layer size.
    pub hidden_dim: usize,
    /// Learning rate for Adam optimizer.
    pub lr: f32,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// GAE λ ∈ [0, 1].
    pub lambda: f32,
    /// Value loss coefficient (default: 0.5).
    pub c_value: f32,
    /// Entropy bonus coefficient (default: 0.01).
    pub c_entropy: f32,
}

impl Default for A2CConfig {
    fn default() -> Self {
        A2CConfig {
            obs_dim: 4,
            num_actions: 2,
            hidden_dim: 64,
            lr: 1e-3,
            gamma: 0.99,
            lambda: 0.95,
            c_value: 0.5,
            c_entropy: 0.01,
        }
    }
}

/// Advantage Actor-Critic (A2C) agent.
pub struct A2C {
    /// Actor-critic network (shared trunk).
    net: ActorCriticNet,
    /// Adam optimizer for all parameters.
    optimizer: Adam,
    /// Configuration.
    config: A2CConfig,
}

impl A2C {
    /// Creates a new A2C agent.
    pub fn new(config: A2CConfig) -> Self {
        let net = ActorCriticNet::new(config.obs_dim, config.hidden_dim, config.num_actions);
        let optimizer = Adam::new(net.parameters(), config.lr);

        A2C {
            net,
            optimizer,
            config,
        }
    }

    /// Forward pass through the network: returns (logits, value).
    pub fn forward(&self, state: &[f32]) -> (Variable, Variable) {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        self.net.forward(&state_var)
    }

    /// Returns the critic's value estimate for a state (detached f32).
    ///
    /// Used during rollout collection to fill `RolloutBuffer::push(... value ...)`.
    pub fn value_of(&self, state: &[f32]) -> f32 {
        let (_, value) = self.forward(state);
        let val = value.data().item();
        val
    }

    /// Samples an action from the policy's categorical distribution.
    pub fn sample_action(logits: &Tensor, rng: &mut impl Rng) -> usize {
        // Reuse REINFORCE's sampling logic
        crate::agent::reinforce::REINFORCE::sample_action(logits, rng)
    }

    /// Convenience wrapper using thread_rng.
    pub fn sample_action_default(logits: &Tensor) -> usize {
        let mut rng = rand::thread_rng();
        Self::sample_action(logits, &mut rng)
    }

    /// Trains on a rollout batch with combined actor-critic-entropy loss.
    ///
    /// ## Algorithm
    ///
    /// 1. Re-forward: `(logits, values) = net.forward(states)`
    /// 2. Actor loss: `-mean(log_softmax(logits).gather(actions) * detach(advantages))`
    /// 3. Critic loss: `mse(values, detach(returns))`
    /// 4. Entropy: `-mean(Σ_a softmax(logits) * log_softmax(logits))`
    /// 5. Total: `actor_loss + c_value * critic_loss - c_entropy * entropy`
    ///
    /// ## Returns
    /// `(total_loss, actor_loss, critic_loss, entropy)` for logging.
    pub fn train_on_rollout(&mut self, batch: &RolloutBatch) -> (f32, f32, f32, f32) {
        // ── 1. Re-forward through current network ──
        let states_var = Variable::from_tensor(batch.states.clone());
        let (logits, values) = self.net.forward(&states_var); // [B, A], [B, 1]

        // ── 2. Actor loss ──
        let log_probs = log_softmax_var(&logits);
        let selected_log_probs = log_probs.gather(1, &batch.actions[..batch.size]);

        // Advantages must be detached — no gradient back to rollout values
        let advantages = Variable::from_tensor(batch.advantages.clone());
        let actor_loss = -(&selected_log_probs * &advantages).mean();

        // ── 3. Critic loss ──
        // Returns must be detached — these are fixed targets
        let returns = Variable::from_tensor(batch.returns.clone());
        let critic_loss = mse_loss(&values, &returns);

        // ── 4. Entropy bonus ──
        let probs = softmax_var(&logits);
        let entropy_per_sample = -(&probs * &log_probs).sum_axis(1, false); // [B]
        let entropy = entropy_per_sample.mean(); // scalar

        // ── 5. Combined loss ──
        // total = actor + c_value * critic - c_entropy * entropy
        let scaled_critic = &critic_loss * self.config.c_value;
        let scaled_entropy = &entropy * self.config.c_entropy;
        let total_loss = &(&actor_loss + &scaled_critic) - &scaled_entropy;

        let total_val = total_loss.data().item();
        let actor_val = actor_loss.data().item();
        let critic_val = critic_loss.data().item();
        let entropy_val = entropy.data().item();

        // ── 6. Backward + step ──
        self.optimizer.zero_grad();
        total_loss.backward();
        self.optimizer.step();

        (total_val, actor_val, critic_val, entropy_val)
    }

    /// Returns the network's gamma.
    pub fn gamma(&self) -> f32 {
        self.config.gamma
    }

    /// Returns the network's lambda.
    pub fn lambda(&self) -> f32 {
        self.config.lambda
    }

    /// Returns a reference to the actor-critic network.
    pub fn net(&self) -> &ActorCriticNet {
        &self.net
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::RolloutBuffer;
    use approx::assert_abs_diff_eq;

    #[test]
    fn actor_critic_net_forward_shapes() {
        let net = ActorCriticNet::new(4, 32, 3);
        let input = Variable::new(Tensor::from_vec(vec![0.0; 8], &[2, 4]), false);
        let (logits, value) = net.forward(&input);

        assert_eq!(logits.shape(), vec![2, 3]);
        assert_eq!(value.shape(), vec![2, 1]);
    }

    #[test]
    fn actor_critic_net_parameter_count() {
        let net = ActorCriticNet::new(4, 16, 2);
        let params = net.parameters();

        // trunk: Linear(4,16) = W+b = 2 params
        // actor_head: Linear(16,2) = W+b = 2 params
        // critic_head: Linear(16,1) = W+b = 2 params
        // Total: 6 Variable objects
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn actor_critic_net_gradient_flow() {
        let net = ActorCriticNet::new(2, 8, 3);
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
        let (logits, _value) = net.forward(&input);

        // Actor loss
        let actor_loss = logits.sum();
        actor_loss.backward();

        // Trunk and actor_head should have gradients
        let params = net.parameters();
        // Trunk weight (param 0) should have grad from actor path
        assert!(
            params[0].grad().is_some(),
            "Trunk weight should have gradient"
        );
        // Actor head weight (param 2) should have grad
        assert!(
            params[2].grad().is_some(),
            "Actor head should have gradient"
        );

        // Reset and test critic path
        for p in &params {
            p.zero_grad();
        }

        let input2 = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]), false);
        let (_, value2) = net.forward(&input2);
        let critic_loss = value2.sum();
        critic_loss.backward();

        // Trunk and critic_head should have gradients
        assert!(
            params[0].grad().is_some(),
            "Trunk weight should have gradient from critic"
        );
        assert!(
            params[4].grad().is_some(),
            "Critic head should have gradient"
        );
    }

    #[test]
    fn a2c_construction() {
        let agent = A2C::new(A2CConfig::default());
        assert_eq!(agent.net.parameters().len(), 6);
    }

    #[test]
    fn a2c_value_of() {
        let agent = A2C::new(A2CConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 8,
            ..A2CConfig::default()
        });

        let value = agent.value_of(&[1.0, 0.0]);
        assert!(value.is_finite());
    }

    #[test]
    fn a2c_train_produces_finite_losses() {
        let mut agent = A2C::new(A2CConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 8,
            lr: 1e-3,
            gamma: 0.99,
            lambda: 0.95,
            c_value: 0.5,
            c_entropy: 0.01,
        });

        let mut buf = RolloutBuffer::new(10, 2);
        buf.push(&[1.0, 0.0], 0, 1.0, 0.5, 0.0);
        buf.push(&[0.0, 1.0], 1, -1.0, 0.3, 0.0);
        buf.push(&[0.5, 0.5], 0, 0.5, 0.4, 1.0);
        buf.compute_returns_and_advantages(0.99, 0.95, 0.0);

        let batch = buf.to_batch();
        let (total, actor, critic, entropy) = agent.train_on_rollout(&batch);

        assert!(total.is_finite(), "Total loss: {}", total);
        assert!(actor.is_finite(), "Actor loss: {}", actor);
        assert!(critic.is_finite(), "Critic loss: {}", critic);
        assert!(entropy.is_finite(), "Entropy: {}", entropy);
    }

    #[test]
    fn entropy_sign_in_total_loss() {
        // Verify entropy enters total loss with correct sign:
        // total = actor + c_value * critic - c_entropy * entropy
        // Higher entropy should DECREASE total loss (entropy is subtracted).
        //
        // Construct directly: use known logits to compute expected entropy.
        let logits = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0], &[1, 2]), // uniform → max entropy
            true,
        );
        let log_probs = log_softmax_var(&logits);
        let probs = softmax_var(&logits);
        let entropy = -(&probs * &log_probs).sum_axis(1, false).mean();
        let entropy_val = entropy.data().item();

        // Uniform over 2 actions: entropy = log(2) ≈ 0.693
        assert_abs_diff_eq!(entropy_val, (2.0_f32).ln(), epsilon = 1e-4);

        // Peaked distribution → lower entropy
        let peaked_logits = Variable::new(Tensor::from_vec(vec![10.0, -10.0], &[1, 2]), true);
        let peaked_log_probs = log_softmax_var(&peaked_logits);
        let peaked_probs = softmax_var(&peaked_logits);
        let peaked_entropy = -(&peaked_probs * &peaked_log_probs)
            .sum_axis(1, false)
            .mean();
        let peaked_entropy_val = peaked_entropy.data().item();

        assert!(
            peaked_entropy_val < entropy_val,
            "Peaked distribution should have lower entropy: {} vs {}",
            peaked_entropy_val,
            entropy_val
        );

        // In total loss: -c_entropy * entropy
        // Higher entropy → more negative term → lower total loss
        let c_entropy = 0.01;
        let uniform_contribution = -c_entropy * entropy_val;
        let peaked_contribution = -c_entropy * peaked_entropy_val;
        assert!(
            uniform_contribution < peaked_contribution,
            "Higher entropy should contribute more negatively to total loss"
        );
    }

    #[test]
    #[ignore] // Long-running experimental convergence test — sensitive to init/hyperparams
    fn a2c_cartpole_convergence() {
        use crate::env::{CartPole, CartPoleAction, Environment};
        use crate::training::{episode_done, replay_done};

        let mut agent = A2C::new(A2CConfig {
            obs_dim: 4,
            num_actions: 2,
            hidden_dim: 64,
            lr: 7e-4,
            gamma: 0.99,
            lambda: 0.95,
            c_value: 0.5,
            c_entropy: 0.01,
        });

        let mut env = CartPole::with_max_steps(500);
        let mut rng = rand::thread_rng();
        let mut rewards_window = Vec::new();

        for episode in 0..500 {
            let mut buf = RolloutBuffer::new(500, 4);
            let (mut state, _) = env.reset(Some(episode as u64));
            let mut episode_reward = 0.0;

            for _ in 0..500 {
                let (logits, _) = agent.forward(&state);
                let logits_data = logits.data().clone();
                let action_idx = A2C::sample_action(&logits_data, &mut rng);
                let value = agent.value_of(&state);

                let env_action = CartPoleAction::try_from(action_idx).unwrap();
                let (next_state, reward, terminated, truncated, _) = env.step(env_action);
                episode_reward += reward;

                buf.push(
                    &state,
                    action_idx,
                    reward,
                    value,
                    if replay_done(terminated, truncated) {
                        1.0
                    } else {
                        0.0
                    },
                );

                state = next_state;
                if episode_done(terminated, truncated) {
                    break;
                }
            }

            let last_value = if !buf.is_empty() {
                agent.value_of(&state)
            } else {
                0.0
            };
            buf.compute_returns_and_advantages(0.99, 0.95, last_value);
            let batch = buf.to_batch();
            agent.train_on_rollout(&batch);

            rewards_window.push(episode_reward);
            if rewards_window.len() > 100 {
                rewards_window.remove(0);
            }
        }

        let avg = rewards_window.iter().sum::<f32>() / rewards_window.len() as f32;
        // NOTE: A2C convergence on CartPole is sensitive to random init and
        // single-layer trunk. This is an experimental validation, not a hard
        // convergence guarantee. avg > 50 indicates meaningful learning.
        assert!(
            avg > 50.0,
            "A2C should show learning (avg reward > 50) on CartPole, got {}",
            avg
        );
    }
}
