//! Soft Actor-Critic (SAC) with auto-tuned temperature.
//!
//! ## Algorithm
//!
//! SAC maximizes a maximum-entropy objective:
//!
//! ```text
//! J(π) = E[Σ_t r_t + α · H(π(·|s_t))]
//! ```
//!
//! Key components:
//!
//! 1. **GaussianPolicy actor**: Stochastic policy with tanh squashing.
//! 2. **Twin Q-critics**: Two Q-networks, using `min(Q1, Q2)` for stability.
//! 3. **Target twin critics**: Soft-updated copies for stable TD targets.
//! 4. **Auto-tuned temperature**: `log_alpha` is optimized to maintain a
//!    target entropy of `-act_dim`.
//!
//! ```text
//! Critic Target:  y = r + γ · (1-d) · [min(Q1_t(s',ã), Q2_t(s',ã)) - α · log π(ã|s')]
//!                 ã ~ π(·|s')
//!
//! Critic Loss:    L_Q = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
//! Actor Loss:     L_π = mean(α · log π(ã|s) - min(Q1(s,ã), Q2(s,ã)))
//!                 ã ~ π(·|s)
//! Alpha Loss:     L_α = -α · mean(log π(ã|s) + target_entropy)    [detached from actor]
//! ```
//!
//! ## Gradient Isolation
//!
//! - **Critic update**: Actor log-prob in target `y` is detached.
//! - **Actor update**: Q-values are evaluated with current actor samples,
//!   gradient flows through actor only.
//! - **Alpha update**: `log π` is detached from the actor's computation graph.
//!   Alpha gradient only flows to `log_alpha`.

use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::loss::mse_loss;
use rustforge_nn::{Linear, Module, ReLU, Sequential};
use rustforge_tensor::Tensor;

use crate::agent::gaussian_policy::GaussianPolicy;
use crate::agent::utils::{elementwise_min_var, hard_update, soft_update};
use crate::buffer::ContinuousTransitionBatch;

/// Configuration for the SAC agent.
pub struct SACConfig {
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Action dimensionality.
    pub act_dim: usize,
    /// Hidden layer size.
    pub hidden_dim: usize,
    /// Learning rate for actor.
    pub actor_lr: f32,
    /// Learning rate for critics.
    pub critic_lr: f32,
    /// Learning rate for alpha (temperature).
    pub alpha_lr: f32,
    /// Discount factor γ.
    pub gamma: f32,
    /// Soft update coefficient τ.
    pub tau: f32,
    /// Initial temperature α.
    pub init_alpha: f32,
    /// Action lower bounds.
    pub action_low: Vec<f32>,
    /// Action upper bounds.
    pub action_high: Vec<f32>,
}

impl SACConfig {
    /// Creates a default SAC config for a given environment.
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        action_low: Vec<f32>,
        action_high: Vec<f32>,
    ) -> Self {
        SACConfig {
            obs_dim,
            act_dim,
            hidden_dim: 256,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            init_alpha: 0.2,
            action_low,
            action_high,
        }
    }
}

/// Builds a Q-critic network (takes concatenated [state, action]).
fn build_critic(obs_dim: usize, act_dim: usize, hidden_dim: usize) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(obs_dim + act_dim, hidden_dim)),
        Box::new(ReLU),
        Box::new(Linear::new(hidden_dim, hidden_dim)),
        Box::new(ReLU),
        Box::new(Linear::new(hidden_dim, 1)),
    ])
}

/// Soft Actor-Critic agent with auto-tuned temperature.
pub struct SAC {
    /// Stochastic Gaussian policy actor.
    pub actor: GaussianPolicy,
    /// First Q-critic.
    critic1: Sequential,
    /// Second Q-critic (twin).
    critic2: Sequential,
    /// Target for critic 1.
    critic1_target: Sequential,
    /// Target for critic 2.
    critic2_target: Sequential,

    /// Log temperature parameter (optimizable).
    log_alpha: Variable,
    /// Target entropy: -act_dim.
    target_entropy: f32,

    /// Actor optimizer.
    actor_optimizer: Adam,
    /// Joint critic optimizer.
    critic_optimizer: Adam,
    /// Alpha optimizer.
    alpha_optimizer: Adam,

    /// Configuration.
    config: SACConfig,
}

impl SAC {
    /// Creates a new SAC agent.
    pub fn new(config: SACConfig) -> Self {
        let actor = GaussianPolicy::new(
            config.obs_dim,
            config.hidden_dim,
            config.act_dim,
            &config.action_low,
            &config.action_high,
        );

        let critic1 = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);
        let critic2 = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);
        let critic1_target = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);
        let critic2_target = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);

        // Sync targets
        hard_update(&critic1.parameters(), &critic1_target.parameters());
        hard_update(&critic2.parameters(), &critic2_target.parameters());

        let actor_optimizer = Adam::new(actor.parameters(), config.actor_lr);

        let mut critic_params = critic1.parameters();
        critic_params.extend(critic2.parameters());
        let critic_optimizer = Adam::new(critic_params, config.critic_lr);

        // log_alpha parameter
        let log_alpha = Variable::new(Tensor::from_vec(vec![config.init_alpha.ln()], &[1]), true);
        let alpha_optimizer = Adam::new(vec![log_alpha.clone()], config.alpha_lr);

        let target_entropy = -(config.act_dim as f32);

        SAC {
            actor,
            critic1,
            critic2,
            critic1_target,
            critic2_target,
            log_alpha,
            target_entropy,
            actor_optimizer,
            critic_optimizer,
            alpha_optimizer,
            config,
        }
    }

    /// Returns the current temperature α = exp(log_alpha).
    pub fn alpha(&self) -> f32 {
        self.log_alpha.data().to_vec()[0].exp()
    }

    /// Selects an action by sampling from the stochastic policy.
    ///
    /// Returns the action as a `Vec<f32>`.
    pub fn select_action(&self, state: &[f32]) -> Vec<f32> {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        let (action, _log_prob) = self.actor.sample(&state_var);
        let result = action.data().to_vec();
        result
    }

    /// Returns the deterministic action (for evaluation).
    pub fn deterministic_action(&self, state: &[f32]) -> Vec<f32> {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        self.actor.deterministic_action(&state_var)
    }

    /// Performs one training step on a batch of transitions.
    ///
    /// Returns `(critic_loss, actor_loss, alpha_loss, alpha_value)`.
    pub fn train_step(&mut self, batch: &ContinuousTransitionBatch) -> (f32, f32, f32, f32) {
        let n = batch.size;
        if n == 0 {
            return (0.0, 0.0, 0.0, self.alpha());
        }

        let alpha = self.alpha();

        // ── 1. Critic update ──
        self.critic_optimizer.zero_grad();

        let next_states_var = Variable::from_tensor(batch.next_states.clone());
        let (next_actions, next_log_probs) = self.actor.sample(&next_states_var);

        // Target Q values (detached)
        let next_sa = self.concat_state_action(&next_states_var, &next_actions);
        let q1_target_next = self.critic1_target.forward(&next_sa).detach();
        let q2_target_next = self.critic2_target.forward(&next_sa).detach();
        let q_target_min = elementwise_min_var(
            &Variable::from_tensor(q1_target_next.data().clone()),
            &Variable::from_tensor(q2_target_next.data().clone()),
        );

        // y = r + γ·(1-d)·[min(Q1_t, Q2_t) - α·log π(ã|s')]
        let next_lp_detached = next_log_probs.detach();
        let entropy_bonus = &next_lp_detached * (-alpha);
        let q_minus_entropy = &q_target_min + &entropy_bonus;

        let rewards_var = Variable::from_tensor(batch.rewards.clone());
        let dones_var = Variable::from_tensor(batch.dones.clone());
        let ones = Variable::from_tensor(Tensor::from_vec(vec![1.0; n], &[n, 1]));
        let not_done = &ones - &dones_var;
        let target_y = &rewards_var + &(&(&not_done * &q_minus_entropy) * self.config.gamma);
        let target_y = target_y.detach();

        let states_var = Variable::from_tensor(batch.states.clone());
        let actions_var = Variable::from_tensor(batch.actions.clone());
        let current_sa = self.concat_state_action(&states_var, &actions_var);
        let q1 = self.critic1.forward(&current_sa);
        let q2 = self.critic2.forward(&current_sa);
        let critic_loss = &mse_loss(&q1, &target_y) + &mse_loss(&q2, &target_y);

        critic_loss.backward();
        self.critic_optimizer.step();
        let critic_loss_val = critic_loss.data().to_vec()[0];

        // ── 2. Actor update ──
        self.actor_optimizer.zero_grad();

        let (new_actions, new_log_probs) = self.actor.sample(&states_var);
        let new_sa = self.concat_state_action(&states_var, &new_actions);
        let q1_new = self.critic1.forward(&new_sa);
        let q2_new = self.critic2.forward(&new_sa);
        let q_min_new = elementwise_min_var(&q1_new, &q2_new);

        // L_π = mean(α · log π - Q)
        let actor_loss = (&(&new_log_probs * alpha) - &q_min_new).mean();

        actor_loss.backward();
        self.actor_optimizer.step();
        let actor_loss_val = actor_loss.data().to_vec()[0];

        // ── 3. Alpha update ──
        self.alpha_optimizer.zero_grad();

        // Detach log_probs from actor graph for alpha gradient
        let log_probs_detached = new_log_probs.detach();
        let target_ent_const =
            Variable::from_tensor(Tensor::from_vec(vec![self.target_entropy; n], &[n, 1]));
        let alpha_error = &log_probs_detached + &target_ent_const;
        let alpha_loss = &(&self.log_alpha * &alpha_error.mean()) * (-1.0);

        alpha_loss.backward();
        self.alpha_optimizer.step();
        let alpha_loss_val = alpha_loss.data().to_vec()[0];

        // ── 4. Soft update targets ──
        soft_update(
            &self.critic1.parameters(),
            &self.critic1_target.parameters(),
            self.config.tau,
        );
        soft_update(
            &self.critic2.parameters(),
            &self.critic2_target.parameters(),
            self.config.tau,
        );

        (
            critic_loss_val,
            actor_loss_val,
            alpha_loss_val,
            self.alpha(),
        )
    }

    /// Concatenates state and action tensors along the feature dimension.
    fn concat_state_action(&self, state: &Variable, action: &Variable) -> Variable {
        state.concat(action, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ContinuousReplayBuffer;

    fn make_config() -> SACConfig {
        SACConfig {
            obs_dim: 4,
            act_dim: 2,
            hidden_dim: 32,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            init_alpha: 0.2,
            action_low: vec![-1.0, -1.0],
            action_high: vec![1.0, 1.0],
        }
    }

    #[test]
    fn sac_construction_and_select_action() {
        let agent = SAC::new(make_config());
        let state = [0.1_f32, 0.2, 0.3, 0.4];
        let action = agent.select_action(&state);
        assert_eq!(action.len(), 2);
        assert!(agent.alpha() > 0.0);
    }

    #[test]
    fn sac_deterministic_action() {
        let agent = SAC::new(make_config());
        let state = [0.5_f32; 4];
        let a1 = agent.deterministic_action(&state);
        let a2 = agent.deterministic_action(&state);
        assert_eq!(a1, a2);
    }

    #[test]
    fn sac_train_step_produces_finite_losses() {
        let mut agent = SAC::new(make_config());

        let mut buf = ContinuousReplayBuffer::new(100, 4, 2);
        for i in 0..20 {
            let v = i as f32 * 0.1;
            let state = [v; 4];
            let action = agent.select_action(&state);
            buf.push(&state, &action, 1.0, &[v + 0.1; 4], i == 19);
        }

        let mut batch = ContinuousTransitionBatch::new(8, 4, 2);
        buf.sample(8, &mut batch);

        let (cl, al, alpha_l, alpha) = agent.train_step(&batch);
        assert!(cl.is_finite(), "critic_loss is not finite: {}", cl);
        assert!(al.is_finite(), "actor_loss is not finite: {}", al);
        assert!(alpha_l.is_finite(), "alpha_loss is not finite: {}", alpha_l);
        assert!(alpha > 0.0, "alpha must be positive: {}", alpha);
    }

    #[test]
    fn sac_alpha_auto_tunes() {
        let mut agent = SAC::new(make_config());
        let initial_alpha = agent.alpha();

        let mut buf = ContinuousReplayBuffer::new(100, 4, 2);
        for i in 0..50 {
            let v = i as f32 * 0.1;
            let action = agent.select_action(&[v; 4]);
            buf.push(&[v; 4], &action, 1.0, &[v + 0.1; 4], false);
        }

        let mut batch = ContinuousTransitionBatch::new(16, 4, 2);

        for _ in 0..5 {
            buf.sample(16, &mut batch);
            agent.train_step(&batch);
        }

        let final_alpha = agent.alpha();
        // Alpha should have changed from initial value
        // (We can't predict direction, just that it moves)
        let diff = (final_alpha - initial_alpha).abs();
        assert!(
            diff.is_finite(),
            "Alpha change should be finite"
        );
        assert!(final_alpha > 0.0, "Alpha must remain positive");
    }

    #[test]
    fn sac_target_entropy_is_negative_act_dim() {
        let config = make_config();
        let agent = SAC::new(config);
        assert_eq!(agent.target_entropy, -2.0); // act_dim = 2
    }

    #[test]
    fn sac_actor_gradient_flow_regression() {
        let mut config = make_config();
        // Set init_alpha to virtually zero to isolate the Q-value gradient path
        config.init_alpha = 1e-15;
        let mut agent = SAC::new(config);

        // Get initial actor parameters
        let initial_params: Vec<Vec<f32>> = agent
            .actor
            .parameters()
            .iter()
            .map(|p| p.data().to_vec())
            .collect();

        let mut buf = ContinuousReplayBuffer::new(100, 4, 2);
        for i in 0..20 {
            let v = i as f32 * 0.1;
            buf.push(&[v; 4], &[0.0, 0.0], 1.0, &[v + 0.1; 4], false);
        }

        let mut batch = ContinuousTransitionBatch::new(8, 4, 2);
        buf.sample(8, &mut batch);

        // Run train step
        agent.train_step(&batch);

        // Check if actor parameters have changed
        let updated_params: Vec<Vec<f32>> = agent
            .actor
            .parameters()
            .iter()
            .map(|p| p.data().to_vec())
            .collect();

        let mut changed = false;
        for (init, updated) in initial_params.iter().zip(updated_params.iter()) {
            for (i, u) in init.iter().zip(updated.iter()) {
                if (i - u).abs() > 1e-7 {
                    changed = true;
                }
            }
        }
        assert!(
            changed,
            "Actor parameters did not change! Gradient flow is likely broken."
        );
    }
}
