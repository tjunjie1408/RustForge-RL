//! Prioritized Experience Replay (PER) buffer.

use crate::buffer::sum_tree::SumTree;
use crate::buffer::TransitionBatch;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustforge_tensor::Tensor;

/// Prioritized Experience Replay Buffer.
pub struct PrioritizedReplayBuffer {
    states: Vec<f32>,
    actions: Vec<usize>,
    rewards: Vec<f32>,
    next_states: Vec<f32>,
    dones: Vec<bool>,

    tree: SumTree,

    obs_dim: usize,
    alpha: f32,

    /// The maximum priority seen so far, assigned to new transitions.
    max_priority: f32,

    /// Seedable PRNG for stratified sampling, enabling reproducible runs.
    rng: StdRng,
}

impl PrioritizedReplayBuffer {
    /// Creates a new Prioritized Replay Buffer.
    ///
    /// - `capacity`: Max number of transitions.
    /// - `obs_dim`: Dimension of states.
    /// - `alpha`: Determines how much prioritization is used (0.0 = uniform, 1.0 = full prioritization).
    pub fn new(capacity: usize, obs_dim: usize, alpha: f32) -> Self {
        Self::with_rng(capacity, obs_dim, alpha, StdRng::from_entropy())
    }

    /// Creates a new Prioritized Replay Buffer with a fixed seed for reproducible sampling.
    ///
    /// Same parameters as [`new`](Self::new), plus `seed` to make stratified sampling
    /// deterministic across runs (useful for tests and reproducible experiments).
    pub fn with_seed(capacity: usize, obs_dim: usize, alpha: f32, seed: u64) -> Self {
        Self::with_rng(capacity, obs_dim, alpha, StdRng::seed_from_u64(seed))
    }

    fn with_rng(capacity: usize, obs_dim: usize, alpha: f32, rng: StdRng) -> Self {
        PrioritizedReplayBuffer {
            states: vec![0.0; capacity * obs_dim],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            next_states: vec![0.0; capacity * obs_dim],
            dones: vec![false; capacity],
            tree: SumTree::new(capacity),
            obs_dim,
            alpha,
            max_priority: 1.0,
            rng,
        }
    }

    /// Pushes a transition with the maximum known priority to guarantee it is sampled at least once.
    pub fn push(
        &mut self,
        state: &[f32],
        action: usize,
        reward: f32,
        next_state: &[f32],
        done: bool,
    ) {
        let data_idx = self.tree.add(self.max_priority);

        let offset = data_idx * self.obs_dim;
        self.states[offset..offset + self.obs_dim].copy_from_slice(state);
        self.next_states[offset..offset + self.obs_dim].copy_from_slice(next_state);
        self.actions[data_idx] = action;
        self.rewards[data_idx] = reward;
        self.dones[data_idx] = done;
    }

    /// Updates the priorities of recently sampled transitions.
    pub fn update_priorities(&mut self, tree_indices: &[usize], td_errors: &[f32]) {
        for (&tree_idx, &err) in tree_indices.iter().zip(td_errors.iter()) {
            let p = (err.abs() + 1e-5).powf(self.alpha);
            if !p.is_finite() {
                continue;
            }
            self.tree.update(tree_idx, p);
            if p > self.max_priority {
                self.max_priority = p;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tree.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Samples a batch using prioritization without allocating.
    ///
    /// Only the active prefix is written; rows beyond `batch.size` remain unchanged.
    ///
    /// - `beta`: IS weight annealing parameter.
    /// - `batch`: The pre-allocated TransitionBatch to sample into.
    /// - `weights`: The pre-allocated Tensor to write Importance Sampling weights into (shape `[batch_size, 1]`).
    /// - `tree_indices`: Slice to store tree indices for updating priorities.
    pub fn sample(
        &mut self,
        batch_size: usize,
        beta: f32,
        batch: &mut TransitionBatch,
        weights: &mut Tensor,
        tree_indices: &mut [usize],
    ) {
        assert!(!self.is_empty(), "Cannot sample from empty buffer");

        let actual_batch = batch_size.min(self.len());

        let total_p = self.tree.total_priority();
        let segment = total_p / actual_batch as f32;

        let mut min_prob = f32::MAX;

        // Extract contiguous destinations once, outside the sampling loop.
        let states = batch.states.data_mut().as_slice_mut().unwrap();
        let next_states = batch.next_states.data_mut().as_slice_mut().unwrap();
        let rewards = batch.rewards.data_mut().as_slice_mut().unwrap();
        let dones = batch.dones.data_mut().as_slice_mut().unwrap();
        let weights = weights.data_mut().as_slice_mut().unwrap();

        for b in 0..actual_batch {
            let lower = segment * (b as f32);
            let upper = segment * ((b + 1) as f32);
            let s = self.rng.gen_range(lower..upper);
            let (tree_idx, p, data_idx) = self.tree.get(s);
            let prob = p / total_p;
            if prob < min_prob {
                min_prob = prob;
            }

            let src_offset = data_idx * self.obs_dim;
            let dst_offset = b * self.obs_dim;
            states[dst_offset..dst_offset + self.obs_dim]
                .copy_from_slice(&self.states[src_offset..src_offset + self.obs_dim]);
            next_states[dst_offset..dst_offset + self.obs_dim]
                .copy_from_slice(&self.next_states[src_offset..src_offset + self.obs_dim]);
            rewards[b] = self.rewards[data_idx];
            dones[b] = if self.dones[data_idx] { 1.0 } else { 0.0 };
            batch.actions[b] = self.actions[data_idx];
            tree_indices[b] = tree_idx;

            // Reuse caller-owned weights as scratch space. Preserve the original
            // arithmetic and seeded draw order; only normalization needs a second pass.
            weights[b] = (prob * self.len() as f32).powf(-beta);
        }

        let max_weight = (min_prob * self.len() as f32).powf(-beta);
        for weight in &mut weights[..actual_batch] {
            *weight /= max_weight;
        }

        batch.size = actual_batch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_sampling_matches_reference_draws_weights_and_active_rows() {
        for capacity in [7, 8] {
            for pushes in [3, capacity + 5] {
                for seed in [0, 42, 99] {
                    let mut buf = PrioritizedReplayBuffer::with_seed(capacity, 2, 0.6, seed);
                    for i in 0..pushes {
                        buf.push(&[i as f32, -1.0], i, i as f32, &[2.0, i as f32], i % 2 == 0);
                    }
                    let indices: Vec<_> = (capacity - 1..capacity - 1 + buf.len()).collect();
                    let errors: Vec<_> = (0..buf.len()).map(|i| (i + 1) as f32 * 3.0).collect();
                    buf.update_priorities(&indices, &errors);
                    let mut reference_rng = buf.rng.clone();

                    for requested in [0, 1, 4, 12] {
                        for beta in [0.0, 0.4, 1.0] {
                            let mut batch = TransitionBatch::new(12, 2);
                            for tensor in [
                                &mut batch.states,
                                &mut batch.next_states,
                                &mut batch.rewards,
                                &mut batch.dones,
                            ] {
                                tensor.data_mut().fill(-99.0);
                            }
                            batch.actions.fill(usize::MAX);
                            let mut weights = Tensor::full(&[12, 1], -99.0);
                            let mut tree_indices = [usize::MAX; 12];

                            // Original two-pass sampler: keep draws before copying and weighting.
                            let size = requested.min(buf.len());
                            let total = buf.tree.total_priority();
                            let segment = total / size as f32;
                            let draws: Vec<_> = (0..size)
                                .map(|b| {
                                    let s = reference_rng
                                        .gen_range(segment * b as f32..segment * (b + 1) as f32);
                                    let (index, priority, slot) = buf.tree.get(s);
                                    (index, priority / total, slot)
                                })
                                .collect();
                            let min_prob = draws.iter().map(|draw| draw.1).fold(f32::MAX, f32::min);
                            let max_weight = (min_prob * buf.len() as f32).powf(-beta);

                            buf.sample(
                                requested,
                                beta,
                                &mut batch,
                                &mut weights,
                                &mut tree_indices,
                            );
                            assert_eq!(batch.size, size);
                            for (b, &(index, prob, slot)) in draws.iter().enumerate() {
                                assert_eq!(tree_indices[b], index);
                                assert_eq!(batch.actions[b], buf.actions[slot]);
                                assert_eq!(
                                    &batch.states.data().as_slice().unwrap()[b * 2..b * 2 + 2],
                                    &buf.states[slot * 2..slot * 2 + 2]
                                );
                                assert_eq!(
                                    &batch.next_states.data().as_slice().unwrap()[b * 2..b * 2 + 2],
                                    &buf.next_states[slot * 2..slot * 2 + 2]
                                );
                                assert_eq!(batch.rewards.data()[[b, 0]], buf.rewards[slot]);
                                assert_eq!(
                                    batch.dones.data()[[b, 0]],
                                    if buf.dones[slot] { 1.0 } else { 0.0 }
                                );
                                let expected = (prob * buf.len() as f32).powf(-beta) / max_weight;
                                assert_eq!(weights.data()[[b, 0]].to_bits(), expected.to_bits());
                            }
                            for (b, &tree_index) in tree_indices.iter().enumerate().skip(size) {
                                assert_eq!(tree_index, usize::MAX);
                                assert_eq!(batch.actions[b], usize::MAX);
                                assert_eq!(weights.data()[[b, 0]], -99.0);
                                assert_eq!(batch.rewards.data()[[b, 0]], -99.0);
                                assert_eq!(batch.dones.data()[[b, 0]], -99.0);
                                for col in 0..2 {
                                    assert_eq!(batch.states.data()[[b, col]], -99.0);
                                    assert_eq!(batch.next_states.data()[[b, col]], -99.0);
                                }
                            }
                            assert_eq!(
                                buf.rng.clone().gen::<u64>(),
                                reference_rng.clone().gen::<u64>()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_per_push_and_sample() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 0.6);
        buf.push(&[1.0, 1.0, 1.0, 1.0], 0, 1.0, &[2.0, 2.0, 2.0, 2.0], false);
        buf.push(&[3.0, 3.0, 3.0, 3.0], 1, -1.0, &[4.0, 4.0, 4.0, 4.0], true);

        assert_eq!(buf.len(), 2);

        let mut batch = TransitionBatch::new(10, 4);
        let mut weights = Tensor::zeros(&[10, 1]);
        let mut tree_indices = vec![0; 10];
        buf.sample(10, 0.4, &mut batch, &mut weights, &mut tree_indices);

        assert_eq!(batch.size, 2);

        let w_vec = weights.to_vec();
        // With only 2 elements equal priority (initial max_priority=1.0), weights should be 1.0 after normalization.
        assert!((w_vec[0] - 1.0).abs() < 1e-4);
        assert!((w_vec[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn non_finite_td_errors_keep_previous_priorities() {
        let mut buf = PrioritizedReplayBuffer::new(8, 2, 0.6);
        buf.push(&[1.0, 1.0], 0, 1.0, &[2.0, 2.0], false);
        buf.push(&[3.0, 3.0], 1, 1.0, &[4.0, 4.0], false);

        let mut batch = TransitionBatch::new(2, 2);
        let mut weights = Tensor::zeros(&[2, 1]);
        let mut tree_indices = vec![0; 2];
        buf.sample(2, 0.4, &mut batch, &mut weights, &mut tree_indices);
        buf.update_priorities(&tree_indices, &[f32::NAN, f32::INFINITY]);

        assert!(buf.tree.total_priority().is_finite());
        assert!(buf.max_priority.is_finite());
        // Sampling must keep working instead of panicking on a NaN range.
        buf.sample(2, 0.4, &mut batch, &mut weights, &mut tree_indices);
        assert!(weights.to_vec().iter().all(|weight| weight.is_finite()));
    }

    #[test]
    fn test_per_priority_updates() {
        let mut buf = PrioritizedReplayBuffer::new(100, 2, 1.0);
        buf.push(&[1.0, 1.0], 0, 1.0, &[2.0, 2.0], false); // idx 0
        buf.push(&[3.0, 3.0], 1, 1.0, &[4.0, 4.0], false); // idx 1

        let mut batch = TransitionBatch::new(2, 2);
        let mut weights = Tensor::zeros(&[2, 1]);
        let mut tree_indices = vec![0; 2];
        buf.sample(2, 1.0, &mut batch, &mut weights, &mut tree_indices);

        // Update priorities: highly prioritize the first transition
        buf.update_priorities(&tree_indices[..batch.size], &[100.0, 0.0]); // td_error 100 vs 0 (actually 1e-5 due to safety)

        // Next sample should mostly pick the 100.0 error transition.
        buf.sample(2, 1.0, &mut batch, &mut weights, &mut tree_indices);
        // Due to stratification, the high-priority item will definitely be picked in its segment.
        // The test is mostly to ensure it runs without panic.
    }

    #[test]
    fn sampling_under_priority_churn_never_returns_unfilled_slots() {
        use rand::{Rng, SeedableRng};

        let (capacity, batch_size) = (4_096usize, 64usize);
        for seed in [0u64, 1, 2, 5] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut buf = PrioritizedReplayBuffer::with_seed(capacity, 1, 0.6, seed);
            let mut batch = TransitionBatch::new(batch_size, 1);
            let mut weights = Tensor::zeros(&[batch_size, 1]);
            let mut tree_indices = vec![0; batch_size];
            for step in 0..2_000usize {
                buf.push(&[0.0], 0, 1.0, &[0.0], false);
                if step < batch_size {
                    continue;
                }
                buf.sample(batch_size, 0.4, &mut batch, &mut weights, &mut tree_indices);
                for &tree_idx in &tree_indices {
                    let data_idx = tree_idx - (capacity - 1);
                    assert!(
                        data_idx < buf.len(),
                        "seed {seed} step {step}: slot {data_idx}"
                    );
                }
                assert!(weights.to_vec().iter().all(|w| w.is_finite()));
                let td: Vec<f32> = (0..batch_size)
                    .map(|_| rng.gen_range(-50.0..50.0))
                    .collect();
                buf.update_priorities(&tree_indices, &td);
            }
        }
    }
}
