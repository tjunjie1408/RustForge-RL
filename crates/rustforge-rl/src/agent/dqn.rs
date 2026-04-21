//! Deep Q-Network (DQN) algorithm with Target Network and optional Double DQN.
//!
//! ## Algorithm Overview
//!
//! DQN learns an action-value function Q(s, a) via temporal-difference (TD) learning:
//!
//! ```text
//! TD Target:   y = r + γ · max_a' Q_target(s', a') · (1 - done)
//! Loss:        L(θ) = MSE(Q(s, a; θ), y)
//! ```
//!
//! Key techniques:
//! 1. **Experience Replay**: Breaks temporal correlations by sampling uniformly from a buffer.
//! 2. **Target Network**: Stabilizes training by using a frozen copy of Q for TD targets.
//!    Updated via hard copy every `target_update_freq` steps.
//! 3. **Double DQN** (optional): Reduces overestimation by decoupling action selection
//!    (online net) from evaluation (target net):
//!    ```text
//!    a* = argmax_a Q(s', a; θ)           // online selects
//!    y  = r + γ · Q_target(s', a*; θ⁻)   // target evaluates
//!    ```
//!
//! ## Architecture Decisions
//!
//! - **Target Network has `requires_grad=false`**: All parameters of the target network
//!   are non-differentiable. No computation graph is built during target forward pass,
//!   saving memory and preventing gradient pollution.
//! - **Hard copy only** (not soft/Polyak update): Appropriate for DQN.
//!   Soft update (`τ·θ + (1-τ)·θ⁻`) is left for SAC/TD3 in Phase 4.
//! - **Agent-Env boundary**: DQN owns networks+optimizer. The training loop is external —
//!   the caller drives `env.step()` and feeds transitions to `buffer.push()` + `train_step()`.

use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::loss::mse_loss;
use rustforge_nn::{Linear, Module, ReLU, Sequential};
use rustforge_tensor::Tensor;

use crate::buffer::TransitionBatch;

/// Configuration for the DQN agent.
///
/// ## Example
/// ```rust,ignore
/// let config = DQNConfig {
///     obs_dim: 4,
///     num_actions: 2,
///     hidden_dim: 64,
///     lr: 1e-3,
///     gamma: 0.99,
///     target_update_freq: 100,
///     double_dqn: false,
/// };
/// ```
pub struct DQNConfig {
    /// Dimensionality of the observation space.
    pub obs_dim: usize,
    /// Number of discrete actions.
    pub num_actions: usize,
    /// Hidden layer size for the Q-network MLP.
    pub hidden_dim: usize,
    /// Learning rate for Adam optimizer.
    pub lr: f32,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// How often to hard-copy online → target network (in train steps).
    pub target_update_freq: usize,
    /// Whether to use Double DQN (decoupled action selection/evaluation).
    pub double_dqn: bool,
}

impl Default for DQNConfig {
    fn default() -> Self {
        DQNConfig {
            obs_dim: 4,
            num_actions: 2,
            hidden_dim: 64,
            lr: 1e-3,
            gamma: 0.99,
            target_update_freq: 100,
            double_dqn: false,
        }
    }
}

/// Deep Q-Network agent.
///
/// Owns the online Q-network, target Q-network (frozen), and Adam optimizer.
/// The target network is a structural clone with `requires_grad=false` on all parameters.
pub struct DQN {
    /// Online Q-network: Q(s, ·; θ). Trainable.
    q_net: Sequential,
    /// Target Q-network: Q(s, ·; θ⁻). Frozen (no grad).
    target_net: Sequential,
    /// Adam optimizer for the online Q-network parameters.
    optimizer: Adam,
    /// Configuration.
    config: DQNConfig,
    /// Number of `train_step` calls so far (for target update scheduling).
    train_steps: usize,
}

impl DQN {
    /// Creates a new DQN agent with the given configuration.
    ///
    /// Builds a 2-layer MLP: `Linear(obs_dim, hidden) → ReLU → Linear(hidden, num_actions)`.
    /// The target network is initialized as a parameter-matched copy with gradients disabled.
    pub fn new(config: DQNConfig) -> Self {
        // Build online Q-network
        let q_net = Sequential::new(vec![
            Box::new(Linear::new(config.obs_dim, config.hidden_dim)),
            Box::new(ReLU),
            Box::new(Linear::new(config.hidden_dim, config.num_actions)),
        ]);

        // Build target Q-network (same architecture, independent parameters)
        let target_net = Sequential::new(vec![
            Box::new(Linear::new(config.obs_dim, config.hidden_dim)),
            Box::new(ReLU),
            Box::new(Linear::new(config.hidden_dim, config.num_actions)),
        ]);

        // Sync target to match online network
        let q_params = q_net.parameters();
        let t_params = target_net.parameters();
        for (tp, qp) in t_params.iter().zip(q_params.iter()) {
            tp.set_data(qp.data().clone());
        }

        // Create optimizer for online network only
        let optimizer = Adam::new(q_params, config.lr);

        DQN {
            q_net,
            target_net,
            optimizer,
            config,
            train_steps: 0,
        }
    }

    /// Selects an action given an observation (greedy, no exploration).
    ///
    /// Performs a forward pass through the online Q-network and returns `argmax_a Q(s, a)`.
    /// The input should be a single observation vector of length `obs_dim`.
    ///
    /// ## Arguments
    /// - `state`: Observation as a flat f32 slice, length = `obs_dim`.
    ///
    /// ## Returns
    /// The greedy action index.
    pub fn select_greedy_action(&self, state: &[f32]) -> usize {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]);
        let state_var = Variable::from_tensor(state_tensor);
        let q_values = self.q_net.forward(&state_var);
        let q_data = q_values.data();
        q_data
            .argmax_axis(1)
            .expect("argmax failed in select_greedy_action")[0]
    }

    /// Performs a single training step on a batch of transitions.
    ///
    /// ## Algorithm
    ///
    /// 1. Forward online network: `q_values = Q(states; θ)` → `[B, A]`
    /// 2. Gather taken Q-values: `q_taken = q_values.gather(1, actions)` → `[B, 1]`
    /// 3. Compute TD target (no grad):
    ///    - Vanilla DQN: `y = r + γ · max_a' Q_target(s', a') · (1 - done)`
    ///    - Double DQN:  `a* = argmax Q(s', ·; θ)`, `y = r + γ · Q_target(s', a*; θ⁻) · (1 - done)`
    /// 4. Loss = MSE(q_taken, y)
    /// 5. Backward + optimizer step
    /// 6. Periodically hard-copy online → target
    ///
    /// ## Returns
    /// The scalar loss value for logging.
    pub fn train_step(&mut self, batch: &TransitionBatch) -> f32 {
        // ── 1. Forward pass through online Q-network ──
        let states_var = Variable::new(batch.states.clone(), false);
        let q_values = self.q_net.forward(&states_var); // [B, num_actions]

        // ── 2. Gather Q-values for taken actions ──
        let q_taken = q_values.gather(1, &batch.actions[..batch.size]); // [B, 1]

        // ── 3. Compute TD target (DETACHED — no gradient through target net) ──
        let next_states_var = Variable::from_tensor(batch.next_states.clone());
        let td_target = if self.config.double_dqn {
            // Double DQN: online net selects action, target net evaluates
            // a* = argmax_a Q(s', a; θ)  (online network, but detached)
            let next_q_online = self.q_net.forward(&next_states_var);
            let next_q_online_data = next_q_online.data();
            let best_actions = next_q_online_data
                .argmax_axis(1)
                .expect("argmax failed in double DQN");

            // Q_target(s', a*; θ⁻)  (target network, detached)
            let next_q_target = self.target_net.forward(&next_states_var);
            let next_q_target_data = next_q_target.data();
            let next_q_selected = next_q_target_data
                .gather(1, &best_actions)
                .expect("gather failed in double DQN");

            // y = r + γ * Q_target(s', a*) * (1 - done)
            let ones = Tensor::ones(batch.dones.shape());
            let not_done = &ones - &batch.dones;
            let discounted = &(&next_q_selected * self.config.gamma) * &not_done;
            let target = &batch.rewards + &discounted;
            Variable::from_tensor(target) // DETACHED — from_tensor = no grad
        } else {
            // Vanilla DQN: target = r + γ * max_a' Q_target(s', a') * (1 - done)
            let next_q_target = self.target_net.forward(&next_states_var);
            let next_q_data = next_q_target.data();
            let max_next_q = next_q_data
                .max_axis(1, true)
                .expect("max_axis failed in DQN");

            let ones = Tensor::ones(batch.dones.shape());
            let not_done = &ones - &batch.dones;
            let discounted = &(&max_next_q * self.config.gamma) * &not_done;
            let target = &batch.rewards + &discounted;
            Variable::from_tensor(target) // DETACHED — from_tensor = no grad
        };

        // ── 4. Compute MSE loss ──
        let loss = mse_loss(&q_taken, &td_target);
        let loss_val = loss.data().item();

        // ── 5. Backward pass + optimizer step ──
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        // ── 6. Periodically update target network (hard copy) ──
        self.train_steps += 1;
        if self.train_steps.is_multiple_of(self.config.target_update_freq) {
            self.update_target();
        }

        loss_val
    }

    /// Hard-copies all parameters from online → target network.
    ///
    /// `target_param.set_data(online_param.data().clone())`
    ///
    /// This is the DQN-standard synchronization. Soft/Polyak update is reserved
    /// for SAC/TD3 in Phase 4 Week 9-10.
    pub fn update_target(&self) {
        let q_params = self.q_net.parameters();
        let t_params = self.target_net.parameters();
        for (tp, qp) in t_params.iter().zip(q_params.iter()) {
            tp.set_data(qp.data().clone());
        }
    }

    /// Returns the number of training steps completed.
    pub fn train_steps(&self) -> usize {
        self.train_steps
    }

    /// Returns a reference to the online Q-network.
    pub fn q_net(&self) -> &Sequential {
        &self.q_net
    }

    /// Returns a reference to the target Q-network.
    pub fn target_net(&self) -> &Sequential {
        &self.target_net
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ReplayBuffer;
    use approx::assert_abs_diff_eq;

    fn make_dqn() -> DQN {
        DQN::new(DQNConfig {
            obs_dim: 4,
            num_actions: 2,
            hidden_dim: 16,
            lr: 1e-3,
            gamma: 0.99,
            target_update_freq: 10,
            double_dqn: false,
        })
    }

    #[test]
    fn test_dqn_construction() {
        let dqn = make_dqn();
        // Online net: Linear(4,16) has 2 params, ReLU has 0, Linear(16,2) has 2 → total 4
        assert_eq!(dqn.q_net.parameters().len(), 4);
        assert_eq!(dqn.target_net.parameters().len(), 4);
    }

    #[test]
    fn test_target_sync_on_construction() {
        let dqn = make_dqn();
        let q_params = dqn.q_net.parameters();
        let t_params = dqn.target_net.parameters();

        for (qp, tp) in q_params.iter().zip(t_params.iter()) {
            let q_data = qp.data().to_vec();
            let t_data = tp.data().to_vec();
            for (qv, tv) in q_data.iter().zip(t_data.iter()) {
                assert_abs_diff_eq!(qv, tv, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_select_greedy_action() {
        let dqn = make_dqn();
        let state = [0.1, 0.2, 0.3, 0.4];
        let action = dqn.select_greedy_action(&state);
        // Action should be 0 or 1 (valid discrete action)
        assert!(action < 2);
    }

    #[test]
    fn test_gradient_flow_q_net_only() {
        // RL Convergence Harness: q_net gets gradients, target_net does NOT
        let mut dqn = make_dqn();

        // Fill buffer with a single transition
        let mut buffer = ReplayBuffer::new(100, 4);
        buffer.push(&[0.1, 0.2, 0.3, 0.4], 0, 1.0, &[0.5, 0.6, 0.7, 0.8], false);

        let mut batch = crate::buffer::TransitionBatch::new(1, 4);
        buffer.sample(1, &mut batch);

        // Train one step
        let _loss = dqn.train_step(&batch);

        // CRITICAL: online network params should have gradients
        for (i, p) in dqn.q_net.parameters().iter().enumerate() {
            assert!(
                p.grad().is_some(),
                "q_net parameter {} should have gradient after train_step",
                i
            );
        }

        // CRITICAL: target network params should NOT have gradients
        // (they were created independently and never participated in backward)
        for (i, p) in dqn.target_net.parameters().iter().enumerate() {
            assert!(
                p.grad().is_none(),
                "target_net parameter {} should NOT have gradient, but got {:?}",
                i,
                p.grad()
            );
        }
    }

    #[test]
    fn test_train_step_decreases_loss() {
        // Overfit test: repeated train_step on same batch should decrease loss
        let mut dqn = DQN::new(DQNConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 8,
            lr: 1e-2,
            gamma: 0.0, // gamma=0 → target = reward (simplest case)
            target_update_freq: 1000,
            double_dqn: false,
        });

        // Single transition: state=[1,0], action=0, reward=1.0, done=true
        let mut buffer = ReplayBuffer::new(100, 2);
        buffer.push(&[1.0, 0.0], 0, 1.0, &[0.0, 0.0], true);

        let mut batch = crate::buffer::TransitionBatch::new(1, 2);
        buffer.sample(1, &mut batch);

        let first_loss = dqn.train_step(&batch);
        let mut last_loss = first_loss;
        for _ in 0..100 {
            // Re-sample same transition (buffer has only 1)
            buffer.sample(1, &mut batch);
            last_loss = dqn.train_step(&batch);
        }

        assert!(
            last_loss < first_loss * 0.5,
            "Loss should decrease significantly: first={}, last={}",
            first_loss,
            last_loss
        );
    }

    #[test]
    fn test_hard_copy_target_update() {
        let mut dqn = make_dqn();

        // Train a few steps to diverge online from target
        let mut buffer = ReplayBuffer::new(100, 4);
        for i in 0..10 {
            let v = i as f32 * 0.1;
            buffer.push(
                &[v, v, v, v],
                i % 2,
                1.0,
                &[v + 0.1, v + 0.1, v + 0.1, v + 0.1],
                false,
            );
        }

        let mut batch = crate::buffer::TransitionBatch::new(4, 4);
        buffer.sample(4, &mut batch);

        // Train a few steps (without hitting target_update_freq)
        for _ in 0..5 {
            buffer.sample(4, &mut batch);
            dqn.train_step(&batch);
        }

        // After training, online params should differ from target
        let q_p = dqn.q_net.parameters();
        let t_p = dqn.target_net.parameters();
        let q_w = q_p[0].data().to_vec();
        let t_w = t_p[0].data().to_vec();
        let max_diff: f32 = q_w
            .iter()
            .zip(t_w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 1e-6,
            "After training, online and target should differ"
        );

        // Force update
        dqn.update_target();

        // After update, they should match exactly
        let q_p2 = dqn.q_net.parameters();
        let t_p2 = dqn.target_net.parameters();
        for (qp, tp) in q_p2.iter().zip(t_p2.iter()) {
            let qd = qp.data().to_vec();
            let td = tp.data().to_vec();
            for (qv, tv) in qd.iter().zip(td.iter()) {
                assert_abs_diff_eq!(qv, tv, epsilon = 1e-8);
            }
        }
    }
}
