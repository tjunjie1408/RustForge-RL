//! Twin Delayed DDPG (TD3) algorithm.
//!
//! ## Algorithm
//!
//! TD3 extends DDPG with three key techniques:
//!
//! 1. **Clipped Double-Q**: Takes `min(Q1, Q2)` to reduce overestimation.
//! 2. **Delayed Policy Updates**: Updates actor and target networks less
//!    frequently than the critics (every `policy_delay` steps).
//! 3. **Target Action Smoothing**: Adds clipped noise to target actions to
//!    make the value estimate robust to function approximation errors.
//!
//! ```text
//! Target:  y = r + γ · min(Q1_target(s', ã), Q2_target(s', ã))
//!          ã = clip(μ_target(s') + clip(ε, -c, c), a_low, a_high)
//!          ε ~ N(0, σ_target)
//!
//! Critic Loss:  L_Q = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
//! Actor Loss:   L_μ = -mean(Q1(s, μ(s)))
//! ```
//!
//! ## Design Decisions
//!
//! - **Deterministic actor**: Unlike SAC's stochastic policy, TD3 uses a
//!   deterministic policy `μ(s)` with separate exploration noise at collection.
//! - **Separate actor/critic optimizers**: Delayed policy updates mean the
//!   actor optimizer steps less frequently.
//! - **Target networks**: Both critics and actor have target copies, updated
//!   via `soft_update` with small τ.

use rand::Rng;
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::loss::mse_loss;
use rustforge_nn::{Linear, Module, ReLU, Sequential, Tanh};
use rustforge_tensor::Tensor;

use crate::agent::utils::{clamp_var, elementwise_min_var, hard_update, soft_update};
use crate::buffer::ContinuousTransitionBatch;

/// Configuration for the TD3 agent.
pub struct TD3Config {
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
    /// Discount factor γ.
    pub gamma: f32,
    /// Soft update coefficient τ.
    pub tau: f32,
    /// Standard deviation of target smoothing noise.
    pub target_noise_std: f32,
    /// Clipping bound for target smoothing noise.
    pub target_noise_clip: f32,
    /// How often to update actor (every `policy_delay` critic updates).
    pub policy_delay: usize,
    /// Action lower bounds (per dimension).
    pub action_low: Vec<f32>,
    /// Action upper bounds (per dimension).
    pub action_high: Vec<f32>,
}

impl TD3Config {
    /// Creates a default TD3 config for a given environment.
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        action_low: Vec<f32>,
        action_high: Vec<f32>,
    ) -> Self {
        TD3Config {
            obs_dim,
            act_dim,
            hidden_dim: 256,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            target_noise_std: 0.2,
            target_noise_clip: 0.5,
            policy_delay: 2,
            action_low,
            action_high,
        }
    }
}

/// Builds a deterministic actor network: Linear → ReLU → Linear → ReLU → Linear → Tanh.
fn build_actor(obs_dim: usize, hidden_dim: usize, act_dim: usize) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(obs_dim, hidden_dim)),
        Box::new(ReLU),
        Box::new(Linear::new(hidden_dim, hidden_dim)),
        Box::new(ReLU),
        Box::new(Linear::new(hidden_dim, act_dim)),
        Box::new(Tanh),
    ])
}

/// Builds a Q-critic network (takes concatenated [state, action]):
/// Linear → ReLU → Linear → ReLU → Linear(1).
fn build_critic(obs_dim: usize, act_dim: usize, hidden_dim: usize) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(obs_dim + act_dim, hidden_dim)),
        Box::new(ReLU),
        Box::new(Linear::new(hidden_dim, hidden_dim)),
        Box::new(ReLU),
        Box::new(Linear::new(hidden_dim, 1)),
    ])
}

/// Twin Delayed DDPG agent.
pub struct TD3 {
    /// Deterministic actor: μ(s) → tanh-bounded action in [-1, 1].
    actor: Sequential,
    /// Target actor.
    actor_target: Sequential,
    /// First Q-critic.
    critic1: Sequential,
    /// Second Q-critic (twin).
    critic2: Sequential,
    /// First target critic.
    critic1_target: Sequential,
    /// Second target critic.
    critic2_target: Sequential,

    /// Actor optimizer.
    actor_optimizer: Adam,
    /// Joint critic optimizer (both critics share one optimizer).
    critic_optimizer: Adam,

    /// Configuration.
    config: TD3Config,
    /// Per-dimension action scale (for rescaling tanh output).
    scale: Vec<f32>,
    /// Per-dimension action bias.
    bias: Vec<f32>,
    /// Total critic train steps (for delayed actor updates).
    train_steps: usize,
}

impl TD3 {
    /// Creates a new TD3 agent.
    pub fn new(config: TD3Config) -> Self {
        let actor = build_actor(config.obs_dim, config.hidden_dim, config.act_dim);
        let actor_target = build_actor(config.obs_dim, config.hidden_dim, config.act_dim);
        let critic1 = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);
        let critic2 = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);
        let critic1_target = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);
        let critic2_target = build_critic(config.obs_dim, config.act_dim, config.hidden_dim);

        // Sync targets
        hard_update(&actor.parameters(), &actor_target.parameters());
        hard_update(&critic1.parameters(), &critic1_target.parameters());
        hard_update(&critic2.parameters(), &critic2_target.parameters());

        let actor_optimizer = Adam::new(actor.parameters(), config.actor_lr);
        let mut critic_params = critic1.parameters();
        critic_params.extend(critic2.parameters());
        let critic_optimizer = Adam::new(critic_params, config.critic_lr);

        let scale: Vec<f32> = config
            .action_low
            .iter()
            .zip(config.action_high.iter())
            .map(|(l, h)| (h - l) / 2.0)
            .collect();
        let bias: Vec<f32> = config
            .action_low
            .iter()
            .zip(config.action_high.iter())
            .map(|(l, h)| (h + l) / 2.0)
            .collect();

        TD3 {
            actor,
            actor_target,
            critic1,
            critic2,
            critic1_target,
            critic2_target,
            actor_optimizer,
            critic_optimizer,
            config,
            scale,
            bias,
            train_steps: 0,
        }
    }

    /// Selects a deterministic action (with optional exploration noise).
    ///
    /// Returns the scaled action as a `Vec<f32>`.
    pub fn select_action(&self, state: &[f32], noise_std: f32) -> Vec<f32> {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        let raw_action = self.actor.forward(&state_var);
        let raw_data = raw_action.data().to_vec();

        let mut rng = rand::thread_rng();
        let mut action = Vec::with_capacity(self.config.act_dim);
        for (i, &v) in raw_data.iter().enumerate() {
            // raw_action is in [-1, 1] (tanh output). Scale to action bounds.
            let scaled = v * self.scale[i] + self.bias[i];
            let noise = if noise_std > 0.0 {
                let u1: f32 = rng.gen_range(1e-7..1.0);
                let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                noise_std * (-2.0 * u1.ln()).sqrt() * u2.cos()
            } else {
                0.0
            };
            let noisy =
                (scaled + noise).clamp(self.config.action_low[i], self.config.action_high[i]);
            action.push(noisy);
        }
        action
    }

    /// Performs one training step on a batch of transitions.
    ///
    /// Returns `(critic_loss, actor_loss_or_none)`.
    /// Actor loss is None when this step is not a policy-update step.
    pub fn train_step(&mut self, batch: &ContinuousTransitionBatch) -> (f32, Option<f32>) {
        let n = batch.size;
        if n == 0 {
            return (0.0, None);
        }

        let _obs_dim = self.config.obs_dim;
        let act_dim = self.config.act_dim;

        // ── Critic update ──
        self.critic_optimizer.zero_grad();

        // Target action: ã = clip(μ_target(s') + clip(ε, -c, c), a_low, a_high)
        let next_states_var = Variable::from_tensor(batch.next_states.clone());
        let target_actions_raw = self.actor_target.forward(&next_states_var); // [-1, 1]

        // Target smoothing noise
        let mut rng = rand::thread_rng();
        let noise_data: Vec<f32> = (0..n * act_dim)
            .map(|_| {
                let u1: f32 = rng.gen_range(1e-7..1.0);
                let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                let noise = self.config.target_noise_std * (-2.0 * u1.ln()).sqrt() * u2.cos();
                noise.clamp(
                    -self.config.target_noise_clip,
                    self.config.target_noise_clip,
                )
            })
            .collect();
        let noise_var = Variable::from_tensor(Tensor::from_vec(noise_data, &[n, act_dim]));
        let noisy_target = &target_actions_raw + &noise_var;
        let clamped_target = clamp_var(&noisy_target, -1.0, 1.0);

        // Scale target actions for critic input
        let target_actions_scaled = self.scale_actions(&clamped_target, n);

        // Q_target(s', ã)
        let target_sa = self.concat_state_action(&next_states_var, &target_actions_scaled);
        let q1_target = self.critic1_target.forward(&target_sa).detach();
        let q2_target = self.critic2_target.forward(&target_sa).detach();
        let q_target = elementwise_min_var(
            &Variable::from_tensor(q1_target.data().clone()),
            &Variable::from_tensor(q2_target.data().clone()),
        );

        // y = r + γ · (1 - done) · min(Q1_t, Q2_t)
        let rewards_var = Variable::from_tensor(batch.rewards.clone());
        let dones_var = Variable::from_tensor(batch.dones.clone());
        let ones = Variable::from_tensor(Tensor::from_vec(vec![1.0; n], &[n, 1]));
        let not_done = &ones - &dones_var;
        let target_y = &rewards_var + &(&(&not_done * &q_target) * self.config.gamma);
        let target_y = target_y.detach();

        // Current Q values
        let states_var = Variable::from_tensor(batch.states.clone());
        let actions_var = Variable::from_tensor(batch.actions.clone());
        let current_sa = self.concat_state_action(&states_var, &actions_var);
        let q1 = self.critic1.forward(&current_sa);
        let q2 = self.critic2.forward(&current_sa);

        let critic_loss = &mse_loss(&q1, &target_y) + &mse_loss(&q2, &target_y);
        critic_loss.backward();
        self.critic_optimizer.step();

        let critic_loss_val = critic_loss.data().to_vec()[0];
        self.train_steps += 1;

        // ── Delayed actor update ──
        let actor_loss_val = if self.train_steps.is_multiple_of(self.config.policy_delay) {
            self.actor_optimizer.zero_grad();

            let actor_actions_raw = self.actor.forward(&states_var); // [-1, 1]
            let actor_actions_scaled = self.scale_actions(&actor_actions_raw, n);
            let actor_sa = self.concat_state_action(&states_var, &actor_actions_scaled);
            let q1_val = self.critic1.forward(&actor_sa);
            let actor_loss = &q1_val.mean() * (-1.0); // maximize Q

            actor_loss.backward();
            self.actor_optimizer.step();

            // Soft update targets
            soft_update(
                &self.actor.parameters(),
                &self.actor_target.parameters(),
                self.config.tau,
            );
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

            let actor_loss_val = actor_loss.data().to_vec()[0];
            Some(actor_loss_val)
        } else {
            None
        };

        (critic_loss_val, actor_loss_val)
    }

    /// Concatenates state and action tensors along the feature dimension.
    fn concat_state_action(&self, state: &Variable, action: &Variable) -> Variable {
        state.concat(action, 1)
    }

    /// Scales tanh-bounded actions [-1,1] to actual action bounds.
    fn scale_actions(&self, raw: &Variable, batch: usize) -> Variable {
        let act_dim = self.config.act_dim;
        let scale_tensor = Variable::from_tensor(Tensor::from_vec(
            self.scale.repeat(batch),
            &[batch, act_dim],
        ));
        let bias_tensor =
            Variable::from_tensor(Tensor::from_vec(self.bias.repeat(batch), &[batch, act_dim]));
        &(raw * &scale_tensor) + &bias_tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ContinuousReplayBuffer;

    fn make_config() -> TD3Config {
        TD3Config {
            obs_dim: 4,
            act_dim: 2,
            hidden_dim: 32,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            target_noise_std: 0.2,
            target_noise_clip: 0.5,
            policy_delay: 2,
            action_low: vec![-1.0, -1.0],
            action_high: vec![1.0, 1.0],
        }
    }

    #[test]
    fn td3_construction_and_select_action() {
        let agent = TD3::new(make_config());
        let state = [0.1_f32, 0.2, 0.3, 0.4];
        let action = agent.select_action(&state, 0.1);
        assert_eq!(action.len(), 2);
        for (i, a) in action.iter().enumerate() {
            assert!(
                *a >= -1.0 && *a <= 1.0,
                "action[{}] = {} out of bounds",
                i,
                a
            );
        }
    }

    #[test]
    fn td3_train_step_produces_finite_losses() {
        let mut agent = TD3::new(make_config());

        let mut buf = ContinuousReplayBuffer::new(100, 4, 2);
        for i in 0..20 {
            let v = i as f32 * 0.1;
            let state = [v; 4];
            let action = agent.select_action(&state, 0.1);
            buf.push(&state, &action, 1.0, &[v + 0.1; 4], i == 19);
        }

        let mut batch = ContinuousTransitionBatch::new(8, 4, 2);
        buf.sample(8, &mut batch);

        let (critic_loss, actor_loss) = agent.train_step(&batch);
        assert!(
            critic_loss.is_finite(),
            "critic_loss is not finite: {}",
            critic_loss
        );
        if let Some(al) = actor_loss {
            assert!(al.is_finite(), "actor_loss is not finite: {}", al);
        }
    }

    #[test]
    fn td3_delayed_policy_update() {
        let mut agent = TD3::new(make_config());

        let mut buf = ContinuousReplayBuffer::new(100, 4, 2);
        for i in 0..20 {
            let v = i as f32 * 0.1;
            buf.push(&[v; 4], &[0.0, 0.0], 1.0, &[v + 0.1; 4], false);
        }

        let mut batch = ContinuousTransitionBatch::new(8, 4, 2);
        buf.sample(8, &mut batch);

        // Step 1: critic only (policy_delay=2)
        let (_, actor_loss1) = agent.train_step(&batch);
        assert!(actor_loss1.is_none(), "Actor should not update on step 1");

        // Step 2: both critic and actor
        let (_, actor_loss2) = agent.train_step(&batch);
        assert!(
            actor_loss2.is_some(),
            "Actor should update on step 2 (policy_delay=2)"
        );
    }

    #[test]
    fn td3_deterministic_action_no_noise() {
        let agent = TD3::new(make_config());
        let state = [0.5_f32; 4];
        let a1 = agent.select_action(&state, 0.0);
        let a2 = agent.select_action(&state, 0.0);
        assert_eq!(a1, a2, "Deterministic actions should match with no noise");
    }

    #[test]
    fn td3_actor_gradient_flow_regression() {
        let mut agent = TD3::new(make_config());

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

        // Run 2 steps to trigger delayed policy update (policy_delay=2)
        agent.train_step(&batch);
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
