//! Prioritized Experience Replay (PER) buffer.

use crate::buffer::sum_tree::SumTree;
use crate::buffer::TransitionBatch;
use rand::Rng;
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
}

impl PrioritizedReplayBuffer {
    /// Creates a new Prioritized Replay Buffer.
    ///
    /// - `capacity`: Max number of transitions.
    /// - `obs_dim`: Dimension of states.
    /// - `alpha`: Determines how much prioritization is used (0.0 = uniform, 1.0 = full prioritization).
    pub fn new(capacity: usize, obs_dim: usize, alpha: f32) -> Self {
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

    /// Samples a batch using prioritization.
    ///
    /// - `beta`: IS weight annealing parameter.
    /// - `batch`: The pre-allocated TransitionBatch to sample into.
    /// - `weights`: The pre-allocated Tensor to write Importance Sampling weights into (shape `[batch_size, 1]`).
    /// - `tree_indices`: Slice to store tree indices for updating priorities.
    pub fn sample(
        &self,
        batch_size: usize,
        beta: f32,
        batch: &mut TransitionBatch,
        weights: &mut Tensor,
        tree_indices: &mut [usize],
    ) {
        assert!(!self.is_empty(), "Cannot sample from empty buffer");

        let actual_batch = batch_size.min(self.len());
        let mut rng = rand::thread_rng();

        let total_p = self.tree.total_priority();
        let segment = total_p / actual_batch as f32;

        let mut min_prob = f32::MAX;

        // Temporary storage for sampled data
        let mut sampled_data = Vec::with_capacity(actual_batch);

        for b in 0..actual_batch {
            let lower = segment * (b as f32);
            let upper = segment * ((b + 1) as f32);
            let s = rng.gen_range(lower..upper);

            let (tree_idx, p, data_idx) = self.tree.get(s);
            let prob = p / total_p;
            if prob < min_prob {
                min_prob = prob;
            }
            sampled_data.push((tree_idx, prob, data_idx));
        }

        let max_weight = (min_prob * self.len() as f32).powf(-beta);

        let states_flat = batch.states.data_mut();
        let next_states_flat = batch.next_states.data_mut();
        let rewards_flat = batch.rewards.data_mut();
        let dones_flat = batch.dones.data_mut();
        let weights_flat = weights.data_mut();

        for (b, &(tree_idx, prob, data_idx)) in sampled_data.iter().enumerate() {
            let src_offset = data_idx * self.obs_dim;
            let dst_offset = b * self.obs_dim;

            let src_state = &self.states[src_offset..src_offset + self.obs_dim];
            let dst_state =
                &mut states_flat.as_slice_mut().unwrap()[dst_offset..dst_offset + self.obs_dim];
            dst_state.copy_from_slice(src_state);

            let src_ns = &self.next_states[src_offset..src_offset + self.obs_dim];
            let dst_ns = &mut next_states_flat.as_slice_mut().unwrap()
                [dst_offset..dst_offset + self.obs_dim];
            dst_ns.copy_from_slice(src_ns);

            rewards_flat.as_slice_mut().unwrap()[b] = self.rewards[data_idx];
            dones_flat.as_slice_mut().unwrap()[b] = if self.dones[data_idx] { 1.0 } else { 0.0 };

            let weight = (prob * self.len() as f32).powf(-beta) / max_weight;
            weights_flat.as_slice_mut().unwrap()[b] = weight;

            batch.actions[b] = self.actions[data_idx];
            tree_indices[b] = tree_idx;
        }

        batch.size = actual_batch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
