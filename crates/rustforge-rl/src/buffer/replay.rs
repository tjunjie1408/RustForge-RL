//! Uniform-sampling experience replay buffer with SoA (Structure-of-Arrays) layout.
//!
//! ## Design: Zero-Allocation Hot Path
//!
//! The buffer stores transitions in flat, pre-allocated arrays (one per field).
//! This avoids the catastrophic cache-miss pattern of `Vec<Transition>` (AoS layout)
//! and enables `sample()` to fill a pre-allocated `TransitionBatch` without any
//! heap allocation on the hot path.
//!
//! ## Memory Layout
//!
//! ```text
//! states:      [f32; capacity * obs_dim]    — contiguous row-major
//! actions:     [usize; capacity]            — integer indices (type-safe)
//! rewards:     [f32; capacity]
//! next_states: [f32; capacity * obs_dim]    — contiguous row-major
//! dones:       [bool; capacity]
//! ```
//!
//! ## Usage in DQN
//!
//! ```rust,ignore
//! let mut buffer = ReplayBuffer::new(10_000, 4); // capacity=10000, obs_dim=4
//! let mut batch = TransitionBatch::new(64, 4);   // batch_size=64, obs_dim=4
//!
//! // Collect experience
//! buffer.push(&[0.1, 0.2, 0.3, 0.4], 1, 1.0, &[0.5, 0.6, 0.7, 0.8], false);
//!
//! // Sample (zero allocation — fills existing batch)
//! buffer.sample(64, &mut batch);
//! ```

use rand::Rng;
use rustforge_tensor::Tensor;

/// A pre-allocated batch of transitions for zero-allocation sampling.
///
/// The `sample()` method writes directly into these fields, reusing the
/// underlying memory across calls.
///
/// ## Field Types
///
/// - `states`, `next_states`, `rewards`, `dones`: `Tensor` (f32)
/// - `actions`: `Vec<usize>` — type-safe integer indices, never f32
///
/// `dones` uses `f32` encoding: `1.0` for done, `0.0` for not done.
/// This allows direct use in the TD target formula: `reward + gamma * (1 - done) * max_q_next`.
pub struct TransitionBatch {
    /// Observation states, shape `[batch_size, obs_dim]`.
    pub states: Tensor,
    /// Action indices, length `batch_size`. Stored as `usize` for type safety.
    pub actions: Vec<usize>,
    /// Rewards, shape `[batch_size, 1]`.
    pub rewards: Tensor,
    /// Next observation states, shape `[batch_size, obs_dim]`.
    pub next_states: Tensor,
    /// Done flags as f32 (1.0 = done, 0.0 = not done), shape `[batch_size, 1]`.
    pub dones: Tensor,
    /// The actual number of valid entries (may be < batch_size if buffer not full).
    pub size: usize,
}

impl TransitionBatch {
    /// Creates a new pre-allocated batch with the given dimensions.
    ///
    /// All tensors are initialized to zeros and will be overwritten by `sample()`.
    pub fn new(batch_size: usize, obs_dim: usize) -> Self {
        TransitionBatch {
            states: Tensor::zeros(&[batch_size, obs_dim]),
            actions: vec![0; batch_size],
            rewards: Tensor::zeros(&[batch_size, 1]),
            next_states: Tensor::zeros(&[batch_size, obs_dim]),
            dones: Tensor::zeros(&[batch_size, 1]),
            size: 0,
        }
    }
}

/// Uniform-sampling experience replay buffer with SoA memory layout.
///
/// Stores transitions in flat, pre-allocated arrays for cache-friendly access.
/// Uses circular buffer semantics: when full, new transitions overwrite the oldest.
///
/// ## Performance Characteristics
///
/// - `push()`: O(1) — direct index write, no allocation
/// - `sample()`: O(batch_size) — random index generation + flat copy
/// - Memory: `capacity * (2 * obs_dim + 3) * 4 bytes` (approximate)
pub struct ReplayBuffer {
    /// Flattened states storage, length = capacity * obs_dim.
    states: Vec<f32>,
    /// Action indices, length = capacity.
    actions: Vec<usize>,
    /// Rewards, length = capacity.
    rewards: Vec<f32>,
    /// Flattened next-states storage, length = capacity * obs_dim.
    next_states: Vec<f32>,
    /// Done flags, length = capacity.
    dones: Vec<bool>,

    /// Maximum number of transitions stored.
    capacity: usize,
    /// Observation dimensionality.
    obs_dim: usize,
    /// Current write position (circular).
    cursor: usize,
    /// Number of transitions currently stored (min(total_pushed, capacity)).
    len: usize,
}

impl ReplayBuffer {
    /// Creates a new replay buffer with pre-allocated storage.
    ///
    /// ## Arguments
    /// - `capacity`: Maximum number of transitions to store.
    /// - `obs_dim`: Dimensionality of a single observation (e.g., 4 for CartPole).
    pub fn new(capacity: usize, obs_dim: usize) -> Self {
        ReplayBuffer {
            states: vec![0.0; capacity * obs_dim],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            next_states: vec![0.0; capacity * obs_dim],
            dones: vec![false; capacity],
            capacity,
            obs_dim,
            cursor: 0,
            len: 0,
        }
    }

    /// Pushes a single transition into the buffer.
    ///
    /// Uses circular addressing: when full, overwrites the oldest transition.
    /// This is O(1) with zero allocation — just index-based writes.
    ///
    /// ## Arguments
    /// - `state`: Observation, length must be `obs_dim`.
    /// - `action`: Action index (usize).
    /// - `reward`: Scalar reward.
    /// - `next_state`: Next observation, length must be `obs_dim`.
    /// - `done`: Whether the episode terminated after this transition.
    pub fn push(
        &mut self,
        state: &[f32],
        action: usize,
        reward: f32,
        next_state: &[f32],
        done: bool,
    ) {
        debug_assert_eq!(state.len(), self.obs_dim);
        debug_assert_eq!(next_state.len(), self.obs_dim);

        let offset = self.cursor * self.obs_dim;
        self.states[offset..offset + self.obs_dim].copy_from_slice(state);
        self.next_states[offset..offset + self.obs_dim].copy_from_slice(next_state);
        self.actions[self.cursor] = action;
        self.rewards[self.cursor] = reward;
        self.dones[self.cursor] = done;

        self.cursor = (self.cursor + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Returns the number of transitions stored.
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

    /// Samples a random batch of transitions and writes into the provided batch.
    ///
    /// **Zero allocation on the hot path**: the method writes directly into the
    /// pre-allocated `TransitionBatch` fields using `Tensor::data_mut()` and
    /// slice writes. No `Vec::new()`, `.clone()`, or `Box::new()` calls.
    ///
    /// ## Arguments
    /// - `batch_size`: Number of transitions to sample. Clamped to `self.len()`.
    /// - `batch`: Pre-allocated batch to fill. Must have been created with the
    ///   same `obs_dim` and at least `batch_size` capacity.
    ///
    /// ## Panics
    /// If the buffer is empty.
    pub fn sample(&self, batch_size: usize, batch: &mut TransitionBatch) {
        assert!(!self.is_empty(), "Cannot sample from empty ReplayBuffer");

        let actual_batch = batch_size.min(self.len);
        let mut rng = rand::thread_rng();

        // Build the batch data directly into flat f32 arrays
        let states_flat = batch.states.data_mut();
        let next_states_flat = batch.next_states.data_mut();
        let rewards_flat = batch.rewards.data_mut();
        let dones_flat = batch.dones.data_mut();

        for b in 0..actual_batch {
            let idx = rng.gen_range(0..self.len);

            // Copy state: source [idx * obs_dim .. (idx+1) * obs_dim] → dest [b * obs_dim ..]
            let src_offset = idx * self.obs_dim;
            let dst_offset = b * self.obs_dim;

            // Use raw indexed access via ndarray for zero-copy writes
            let src_state = &self.states[src_offset..src_offset + self.obs_dim];
            let dst_state =
                &mut states_flat.as_slice_mut().unwrap()[dst_offset..dst_offset + self.obs_dim];
            dst_state.copy_from_slice(src_state);

            let src_ns = &self.next_states[src_offset..src_offset + self.obs_dim];
            let dst_ns = &mut next_states_flat.as_slice_mut().unwrap()
                [dst_offset..dst_offset + self.obs_dim];
            dst_ns.copy_from_slice(src_ns);

            // Rewards and dones: shape [batch_size, 1], so index is just [b]
            rewards_flat.as_slice_mut().unwrap()[b] = self.rewards[idx];
            dones_flat.as_slice_mut().unwrap()[b] = if self.dones[idx] { 1.0 } else { 0.0 };

            // Actions: direct usize write
            batch.actions[b] = self.actions[idx];
        }

        batch.size = actual_batch;
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_push_and_len() {
        let mut buf = ReplayBuffer::new(100, 4);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        buf.push(&[1.0, 2.0, 3.0, 4.0], 0, 1.0, &[5.0, 6.0, 7.0, 8.0], false);
        assert_eq!(buf.len(), 1);

        buf.push(&[0.1, 0.2, 0.3, 0.4], 1, -1.0, &[0.5, 0.6, 0.7, 0.8], true);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_buffer_circular_overwrite() {
        let mut buf = ReplayBuffer::new(3, 2);
        buf.push(&[1.0, 1.0], 0, 1.0, &[2.0, 2.0], false);
        buf.push(&[3.0, 3.0], 1, 2.0, &[4.0, 4.0], false);
        buf.push(&[5.0, 5.0], 0, 3.0, &[6.0, 6.0], false);
        assert_eq!(buf.len(), 3);

        // This should overwrite the first transition
        buf.push(&[7.0, 7.0], 1, 4.0, &[8.0, 8.0], true);
        assert_eq!(buf.len(), 3); // still 3 (capacity)

        // Verify the oldest was overwritten: states[0] should now be [7.0, 7.0]
        assert_eq!(buf.states[0], 7.0);
        assert_eq!(buf.states[1], 7.0);
    }

    #[test]
    fn test_transition_batch_creation() {
        let batch = TransitionBatch::new(32, 4);
        assert_eq!(batch.states.shape(), &[32, 4]);
        assert_eq!(batch.rewards.shape(), &[32, 1]);
        assert_eq!(batch.next_states.shape(), &[32, 4]);
        assert_eq!(batch.dones.shape(), &[32, 1]);
        assert_eq!(batch.actions.len(), 32);
    }

    #[test]
    fn test_sample_fills_batch() {
        let mut buf = ReplayBuffer::new(100, 4);
        for i in 0..50 {
            let v = i as f32;
            buf.push(
                &[v, v, v, v],
                i % 2,
                v,
                &[v + 1.0, v + 1.0, v + 1.0, v + 1.0],
                i % 5 == 0,
            );
        }

        let mut batch = TransitionBatch::new(16, 4);
        buf.sample(16, &mut batch);
        assert_eq!(batch.size, 16);

        // Verify all actions are valid (0 or 1)
        for a in &batch.actions[..batch.size] {
            assert!(*a < 2);
        }

        // Verify dones are 0.0 or 1.0
        let dones = batch.dones.to_vec();
        for d in &dones[..batch.size] {
            assert!(*d == 0.0 || *d == 1.0);
        }
    }

    #[test]
    fn test_sample_with_small_buffer() {
        let mut buf = ReplayBuffer::new(100, 2);
        buf.push(&[1.0, 2.0], 0, 1.0, &[3.0, 4.0], false);

        let mut batch = TransitionBatch::new(10, 2);
        buf.sample(10, &mut batch);
        // Should clamp to buffer size
        assert_eq!(batch.size, 1);

        // The single sample should be our single transition
        let states = batch.states.to_vec();
        assert_eq!(states[0], 1.0);
        assert_eq!(states[1], 2.0);
        assert_eq!(batch.actions[0], 0);
    }

    #[test]
    #[should_panic(expected = "Cannot sample from empty")]
    fn test_sample_empty_panics() {
        let buf = ReplayBuffer::new(100, 4);
        let mut batch = TransitionBatch::new(8, 4);
        buf.sample(8, &mut batch);
    }
}
