//! On-policy rollout buffer with SoA (Structure-of-Arrays) layout.
//!
//! ## Design: Collect → Compute → Consume → Clear
//!
//! Unlike `ReplayBuffer` (circular, off-policy), `RolloutBuffer` is designed for
//! on-policy algorithms (REINFORCE, A2C, PPO). Each rollout cycle:
//!
//! 1. **Collect**: `push()` transitions from environment interaction.
//! 2. **Compute**: `compute_returns_and_advantages()` fills returns/advantages.
//! 3. **Consume**: `to_batch()` produces a `RolloutBatch` for training.
//! 4. **Clear**: `clear()` resets the buffer for the next rollout.
//!
//! ## `dones` Semantics
//!
//! The `done` parameter represents **true terminal states only** (`terminated`),
//! NOT time-limit truncation. Use `training::replay_done(terminated, truncated)`
//! to produce the correct value. This ensures GAE does not underestimate
//! truncated episode tails.
//!
//! ## REINFORCE vs A2C
//!
//! - **REINFORCE** (no critic): pass `value=0.0` in `push()`, then call
//!   `compute_returns_and_advantages(gamma, 1.0, 0.0)`. Result: `advantages == returns`.
//! - **A2C**: pass critic's detached V(s_t) as `value` in `push()`, then call
//!   `compute_returns_and_advantages(gamma, lambda, last_value)`.
//!
//! ## No `log_probs` Storage
//!
//! This buffer intentionally does **not** store log-probabilities. REINFORCE/A2C
//! must re-forward through the current policy at training time to obtain
//! differentiable `log π(a|s)` with a live computation graph. Stored f32 log-probs
//! would have no gradient path back to policy parameters.
//!
//! (PPO will add `old_log_probs` for importance ratio computation in Phase 4.)

use rustforge_tensor::Tensor;

use crate::agent::returns;

/// A batch of rollout data ready for training.
///
/// Produced by `RolloutBuffer::to_batch()` after `compute_returns_and_advantages()`.
///
/// ## Field Types
///
/// - `states`: `Tensor` shape `[len, obs_dim]`
/// - `actions`: `Vec<usize>` — type-safe integer indices (same as `TransitionBatch`)
/// - `returns`: `Tensor` shape `[len, 1]` — value regression targets
/// - `advantages`: `Tensor` shape `[len, 1]` — policy loss weights
pub struct RolloutBatch {
    /// Observation states, shape `[len, obs_dim]`.
    pub states: Tensor,
    /// Action indices, length `len`.
    pub actions: Vec<usize>,
    /// Returns G_t (= advantages + values), shape `[len, 1]`.
    pub returns: Tensor,
    /// Advantages A_t (GAE), shape `[len, 1]`.
    pub advantages: Tensor,
    /// Number of valid entries.
    pub size: usize,
}

/// On-policy rollout buffer with SoA layout and pre-allocated storage.
///
/// ## Capacity Behavior
///
/// Unlike `ReplayBuffer` (circular overwrite), `RolloutBuffer` **panics** if
/// `push()` is called when the buffer is full. On-policy algorithms should
/// `clear()` after each training update. Exceeding capacity is a logic error.
pub struct RolloutBuffer {
    /// Flattened states, length = capacity * obs_dim.
    states: Vec<f32>,
    /// Action indices, length = capacity.
    actions: Vec<usize>,
    /// Per-step rewards, length = capacity.
    rewards: Vec<f32>,
    /// Per-step value estimates V(s_t), length = capacity.
    values: Vec<f32>,
    /// Done flags (f32: 1.0 = true terminal, 0.0 = not), length = capacity.
    dones: Vec<f32>,

    /// Computed returns G_t, length = capacity. Filled by `compute_returns_and_advantages`.
    returns: Vec<f32>,
    /// Computed advantages A_t, length = capacity. Filled by `compute_returns_and_advantages`.
    advantages: Vec<f32>,

    /// Maximum number of transitions.
    capacity: usize,
    /// Observation dimensionality.
    obs_dim: usize,
    /// Current number of stored transitions.
    len: usize,
    /// Whether `compute_returns_and_advantages` has been called since last `clear`.
    computed: bool,
}

impl RolloutBuffer {
    /// Creates a new rollout buffer with pre-allocated storage.
    ///
    /// ## Arguments
    /// - `capacity`: Maximum transitions per rollout.
    /// - `obs_dim`: Dimensionality of a single observation.
    pub fn new(capacity: usize, obs_dim: usize) -> Self {
        RolloutBuffer {
            states: vec![0.0; capacity * obs_dim],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            values: vec![0.0; capacity],
            dones: vec![0.0; capacity],
            returns: vec![0.0; capacity],
            advantages: vec![0.0; capacity],
            capacity,
            obs_dim,
            len: 0,
            computed: false,
        }
    }

    /// Pushes a single transition into the buffer.
    ///
    /// ## Arguments
    /// - `state`: Observation, length must be `obs_dim`.
    /// - `action`: Action index.
    /// - `reward`: Scalar reward.
    /// - `value`: Critic's V(s_t) estimate (detached f32). Use 0.0 for REINFORCE.
    /// - `done`: Terminal flag (f32: 1.0 = true terminal only, not time-limit truncation).
    ///
    /// ## Panics
    /// If the buffer is already at capacity. On-policy buffers should `clear()` between rollouts.
    pub fn push(&mut self, state: &[f32], action: usize, reward: f32, value: f32, done: f32) {
        assert!(
            self.len < self.capacity,
            "RolloutBuffer overflow: len={} >= capacity={}. Call clear() between rollouts.",
            self.len,
            self.capacity
        );
        debug_assert_eq!(
            state.len(),
            self.obs_dim,
            "state.len()={} != obs_dim={}",
            state.len(),
            self.obs_dim
        );

        let offset = self.len * self.obs_dim;
        self.states[offset..offset + self.obs_dim].copy_from_slice(state);
        self.actions[self.len] = action;
        self.rewards[self.len] = reward;
        self.values[self.len] = value;
        self.dones[self.len] = done;

        self.len += 1;
        self.computed = false; // invalidate any previous computation
    }

    /// Computes returns and advantages using GAE.
    ///
    /// Delegates to `returns::compute_gae()`. After this call, `to_batch()` becomes valid.
    ///
    /// ## Arguments
    /// - `gamma`: Discount factor γ ∈ [0, 1].
    /// - `lambda`: GAE λ ∈ [0, 1]. Use 1.0 for REINFORCE (MC returns).
    /// - `last_value`: V(s_{T+1}). Use 0.0 if the episode terminated, or
    ///   the critic's estimate if the episode was truncated by a time limit.
    pub fn compute_returns_and_advantages(&mut self, gamma: f32, lambda: f32, last_value: f32) {
        let (advantages, rets) = returns::compute_gae(
            &self.rewards[..self.len],
            &self.values[..self.len],
            &self.dones[..self.len],
            last_value,
            gamma,
            lambda,
        );

        self.advantages[..self.len].copy_from_slice(&advantages);
        self.returns[..self.len].copy_from_slice(&rets);
        self.computed = true;
    }

    /// Converts the buffer contents into a `RolloutBatch` of tensors.
    ///
    /// ## Panics
    /// If `compute_returns_and_advantages()` has not been called since the last
    /// `clear()` or `push()`. This prevents consuming stale/zeroed returns.
    pub fn to_batch(&self) -> RolloutBatch {
        assert!(
            self.computed,
            "must call compute_returns_and_advantages() before to_batch()"
        );
        assert!(self.len > 0, "cannot create batch from empty RolloutBuffer");

        let states = Tensor::from_vec(
            self.states[..self.len * self.obs_dim].to_vec(),
            &[self.len, self.obs_dim],
        );
        let actions = self.actions[..self.len].to_vec();
        let returns = Tensor::from_vec(self.returns[..self.len].to_vec(), &[self.len, 1]);
        let advantages = Tensor::from_vec(self.advantages[..self.len].to_vec(), &[self.len, 1]);

        RolloutBatch {
            states,
            actions,
            returns,
            advantages,
            size: self.len,
        }
    }

    /// Resets the buffer for the next rollout. Does not deallocate memory.
    pub fn clear(&mut self) {
        self.len = 0;
        self.computed = false;
    }

    /// Returns the number of transitions currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_buffer() -> RolloutBuffer {
        RolloutBuffer::new(10, 2)
    }

    #[test]
    fn push_and_len() {
        let mut buf = make_buffer();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        buf.push(&[1.0, 2.0], 0, 1.0, 0.5, 0.0);
        assert_eq!(buf.len(), 1);

        buf.push(&[3.0, 4.0], 1, -1.0, 0.3, 1.0);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    #[should_panic(expected = "RolloutBuffer overflow")]
    fn push_overflow_panics() {
        let mut buf = RolloutBuffer::new(2, 1);
        buf.push(&[1.0], 0, 1.0, 0.0, 0.0);
        buf.push(&[2.0], 0, 1.0, 0.0, 0.0);
        buf.push(&[3.0], 0, 1.0, 0.0, 0.0); // panics
    }

    #[test]
    fn clear_resets() {
        let mut buf = make_buffer();
        buf.push(&[1.0, 2.0], 0, 1.0, 0.0, 0.0);
        buf.push(&[3.0, 4.0], 1, 2.0, 0.0, 0.0);
        buf.compute_returns_and_advantages(0.99, 0.95, 0.0);
        assert_eq!(buf.len(), 2);

        buf.clear();
        assert!(buf.is_empty());
        assert!(!buf.computed);
    }

    #[test]
    #[should_panic(expected = "must call compute_returns_and_advantages")]
    fn to_batch_before_compute_panics() {
        let mut buf = make_buffer();
        buf.push(&[1.0, 2.0], 0, 1.0, 0.0, 0.0);
        buf.to_batch(); // panics
    }

    #[test]
    #[should_panic(expected = "must call compute_returns_and_advantages")]
    fn to_batch_after_push_invalidates() {
        let mut buf = make_buffer();
        buf.push(&[1.0, 2.0], 0, 1.0, 0.0, 0.0);
        buf.compute_returns_and_advantages(0.99, 0.95, 0.0);

        // New push invalidates computed flag
        buf.push(&[3.0, 4.0], 1, 2.0, 0.0, 0.0);
        buf.to_batch(); // panics
    }

    #[test]
    fn to_batch_shapes() {
        let mut buf = RolloutBuffer::new(10, 4);
        for i in 0..5 {
            let v = i as f32;
            buf.push(&[v, v, v, v], i % 2, v, 0.0, 0.0);
        }
        buf.compute_returns_and_advantages(0.99, 0.95, 0.0);

        let batch = buf.to_batch();
        assert_eq!(batch.size, 5);
        assert_eq!(batch.states.shape(), &[5, 4]);
        assert_eq!(batch.returns.shape(), &[5, 1]);
        assert_eq!(batch.advantages.shape(), &[5, 1]);
        assert_eq!(batch.actions.len(), 5);
    }

    #[test]
    fn reinforce_pattern_values_zero() {
        // REINFORCE: values=0, λ=1 → advantages == returns == MC returns
        let mut buf = RolloutBuffer::new(10, 1);
        buf.push(&[0.0], 0, 1.0, 0.0, 0.0); // value=0
        buf.push(&[0.0], 1, 2.0, 0.0, 0.0);
        buf.push(&[0.0], 0, 3.0, 0.0, 0.0);
        buf.compute_returns_and_advantages(0.9, 1.0, 0.0);

        let batch = buf.to_batch();
        let returns = batch.returns.to_vec();
        let advantages = batch.advantages.to_vec();

        // MC: G_2=3, G_1=2+0.9*3=4.7, G_0=1+0.9*4.7=5.23
        let g2 = 3.0_f32;
        let g1 = 2.0 + 0.9 * g2;
        let g0 = 1.0 + 0.9 * g1;

        assert_abs_diff_eq!(returns[0], g0, epsilon = 1e-5);
        assert_abs_diff_eq!(returns[1], g1, epsilon = 1e-5);
        assert_abs_diff_eq!(returns[2], g2, epsilon = 1e-5);

        // advantages == returns when values=0
        assert_abs_diff_eq!(advantages[0], returns[0], epsilon = 1e-5);
        assert_abs_diff_eq!(advantages[1], returns[1], epsilon = 1e-5);
        assert_abs_diff_eq!(advantages[2], returns[2], epsilon = 1e-5);
    }

    #[test]
    fn a2c_pattern_with_values() {
        let mut buf = RolloutBuffer::new(10, 1);
        buf.push(&[0.0], 0, 1.0, 2.0, 0.0); // value=2.0
        buf.push(&[0.0], 1, 3.0, 4.0, 0.0); // value=4.0
        buf.compute_returns_and_advantages(0.9, 0.0, 5.0);

        let batch = buf.to_batch();
        let returns = batch.returns.to_vec();
        let advantages = batch.advantages.to_vec();

        // λ=0 → advantages = TD errors
        // δ_1 = 3 + 0.9*5*1 - 4 = 3.5
        // δ_0 = 1 + 0.9*4*1 - 2 = 2.6
        assert_abs_diff_eq!(advantages[0], 2.6, epsilon = 1e-5);
        assert_abs_diff_eq!(advantages[1], 3.5, epsilon = 1e-5);

        // returns = advantages + values
        assert_abs_diff_eq!(returns[0], 2.6 + 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(returns[1], 3.5 + 4.0, epsilon = 1e-5);
    }

    #[test]
    fn reuse_after_clear() {
        let mut buf = RolloutBuffer::new(5, 1);

        // First rollout
        buf.push(&[1.0], 0, 1.0, 0.0, 0.0);
        buf.compute_returns_and_advantages(0.99, 1.0, 0.0);
        let batch1 = buf.to_batch();
        assert_eq!(batch1.size, 1);

        // Clear and second rollout
        buf.clear();
        buf.push(&[2.0], 1, 5.0, 0.0, 1.0);
        buf.push(&[3.0], 0, 3.0, 0.0, 0.0);
        buf.compute_returns_and_advantages(0.99, 1.0, 0.0);
        let batch2 = buf.to_batch();
        assert_eq!(batch2.size, 2);
    }
}
