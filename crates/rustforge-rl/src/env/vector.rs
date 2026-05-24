//! Synchronous vectorized environment with pre-allocated SoA buffers.
//!
//! # Architecture
//!
//! `SyncVectorEnv` wraps N independent environments and manages:
//! 1. **Pre-allocated contiguous memory** — a single `Vec<f32>` buffer in row-major
//!    `[N, obs_dim]` layout, directly compatible with `ndarray::ArrayView2`.
//! 2. **Auto-reset protocol** — when any sub-environment terminates, it is immediately
//!    reset within the same `step_batch` call. The terminal observation is preserved
//!    in a stack-allocated `Option<Obs>` (zero heap cost).
//! 3. **SoA batch returns** — `BatchStepResult` contains borrowed slices into
//!    pre-allocated internal buffers, zero allocation per step.
//!
//! # Borrow-Split Pattern
//!
//! The implementation destructures `self` into individual fields to satisfy the
//! borrow checker when simultaneously iterating over `&mut envs` and writing to
//! `&mut obs_buffer`. This is the ONLY safe pattern without `unsafe`.
//!
//! # Memory Layout
//!
//! Row-major AoS: `[env0_obs0, env0_obs1, ..., env0_obsD, env1_obs0, ...]`
//! This is the natural C-order layout for `ndarray::Array2<f32>` with shape `(N, D)`,
//! enabling true zero-copy tensor construction.

use super::traits::{Environment, IntoTensorBuffer};

/// SoA batch result — all fields are borrowed slices into pre-allocated buffers.
///
/// Zero allocation per `step_batch()` call. Lifetimes tie results to the
/// `SyncVectorEnv` that produced them, preventing use-after-mutation.
///
/// # Memory Layout
/// - `obs`: Contiguous `[N × DIM]` f32 buffer, row-major.
///   Convert to ndarray: `ArrayView2::from_shape((n, dim), obs).unwrap()`
/// - `rewards`, `terminated`, `truncated`: Length-N contiguous slices.
/// - `terminal_obs`: Stack-allocated `Option<O>` per environment.
///   `Some(obs)` only when `terminated[i] || truncated[i]`.
pub struct BatchStepResult<'a, O: Clone> {
    /// Contiguous f32 observation buffer, row-major [N, obs_dim].
    pub obs: &'a [f32],
    /// Per-environment rewards, length N.
    pub rewards: &'a [f32],
    /// Per-environment terminated flags, length N.
    pub terminated: &'a [bool],
    /// Per-environment truncated flags, length N.
    pub truncated: &'a [bool],
    /// Terminal observations for auto-reset environments.
    /// `Some(obs)` contains the true final observation before the environment
    /// was auto-reset. For CartPole: `Option<[f32; 4]>` = 17 bytes on stack.
    pub terminal_obs: &'a [Option<O>],
}

/// Synchronous vectorized environment with pre-allocated SoA buffers.
///
/// Wraps N independent environments and executes them sequentially in a single thread.
/// Pre-allocates all output buffers during construction — `step_batch` never allocates.
///
/// # Type Constraints
/// - `E::Obs: IntoTensorBuffer` — required for writing observations into the f32 buffer.
/// - `E::Act: Clone` — required for distributing actions to sub-environments.
pub struct SyncVectorEnv<E: Environment>
where
    E::Obs: IntoTensorBuffer,
{
    /// Sub-environments
    envs: Vec<E>,
    /// Pre-allocated observation buffer: [N × DIM] f32, row-major
    obs_buffer: Vec<f32>,
    /// Pre-allocated reward buffer: [N]
    reward_buffer: Vec<f32>,
    /// Pre-allocated terminated flags: [N]
    terminated_buffer: Vec<bool>,
    /// Pre-allocated truncated flags: [N]
    truncated_buffer: Vec<bool>,
    /// Pre-allocated terminal observation storage: [N]
    terminal_obs_buffer: Vec<Option<E::Obs>>,
    /// Observation dimensionality (cached from IntoTensorBuffer::DIM)
    obs_dim: usize,
}

impl<E: Environment> SyncVectorEnv<E>
where
    E::Obs: IntoTensorBuffer,
{
    /// Create a new SyncVectorEnv from a vector of pre-constructed environments.
    ///
    /// All buffers are pre-allocated during construction. No allocations occur
    /// during `step_batch` or `reset_all`.
    pub fn new(envs: Vec<E>) -> Self {
        let n = envs.len();
        let obs_dim = E::Obs::DIM;

        SyncVectorEnv {
            envs,
            obs_buffer: vec![0.0; n * obs_dim],
            reward_buffer: vec![0.0; n],
            terminated_buffer: vec![false; n],
            truncated_buffer: vec![false; n],
            terminal_obs_buffer: (0..n).map(|_| None).collect(),
            obs_dim,
        }
    }

    /// Number of sub-environments.
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Observation dimensionality per environment.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Reset all environments.
    ///
    /// Each environment is reset with its own seed (if provided) to ensure
    /// independent PRNG streams.
    ///
    /// Returns a reference to the contiguous observation buffer.
    pub fn reset_all(&mut self, seeds: Option<&[u64]>) -> &[f32] {
        let obs_dim = self.obs_dim;

        // Borrow-split: destructure self into independent fields
        let envs = &mut self.envs;
        let obs_buf = &mut self.obs_buffer;

        for (i, (env, obs_chunk)) in envs
            .iter_mut()
            .zip(obs_buf.chunks_exact_mut(obs_dim))
            .enumerate()
        {
            let seed = seeds.map(|s| s[i]);
            let (obs, _info) = env.reset(seed);
            obs.write_to_buffer(obs_chunk);
        }

        &self.obs_buffer
    }

    /// Execute one step across all environments.
    ///
    /// # Auto-Reset Protocol
    ///
    /// When a sub-environment terminates or is truncated:
    /// 1. The terminal observation is stashed in `terminal_obs_buffer[i]`.
    /// 2. The environment is immediately reset via `reset(None)` (continues PRNG).
    /// 3. The NEW initial observation is written into the obs buffer.
    ///
    /// This ensures the obs buffer always contains valid initial states for
    /// the next forward pass, while terminal observations are preserved for
    /// value function bootstrapping.
    ///
    /// # Borrow-Split Pattern
    ///
    /// Self is destructured into individual fields to allow simultaneous
    /// `&mut envs` and `&mut obs_buffer` access without `unsafe`.
    pub fn step_batch(&mut self, actions: &[E::Act]) -> BatchStepResult<'_, E::Obs> {
        assert_eq!(
            actions.len(),
            self.envs.len(),
            "Expected {} actions, got {}",
            self.envs.len(),
            actions.len()
        );

        let obs_dim = self.obs_dim;

        // Borrow-split: destructure self to satisfy borrow checker
        let envs = &mut self.envs;
        let obs_buf = &mut self.obs_buffer;
        let rewards = &mut self.reward_buffer;
        let terms = &mut self.terminated_buffer;
        let truncs = &mut self.truncated_buffer;
        let term_obs = &mut self.terminal_obs_buffer;

        for (i, ((env, obs_chunk), action)) in envs
            .iter_mut()
            .zip(obs_buf.chunks_exact_mut(obs_dim))
            .zip(actions.iter())
            .enumerate()
        {
            let (obs, r, terminated, truncated, _info) = env.step(action.clone());
            rewards[i] = r;
            terms[i] = terminated;
            truncs[i] = truncated;

            if terminated || truncated {
                // Stash terminal obs (stack-allocated clone for fixed-size arrays)
                term_obs[i] = Some(obs);
                // Auto-reset: continue PRNG stream (no reseeding)
                let (new_obs, _) = env.reset(None);
                new_obs.write_to_buffer(obs_chunk);
            } else {
                term_obs[i] = None;
                obs.write_to_buffer(obs_chunk);
            }
        }

        BatchStepResult {
            obs: &self.obs_buffer,
            rewards: &self.reward_buffer,
            terminated: &self.terminated_buffer,
            truncated: &self.truncated_buffer,
            terminal_obs: &self.terminal_obs_buffer,
        }
    }

    /// Get a reference to the raw observation buffer.
    pub fn obs_buffer(&self) -> &[f32] {
        &self.obs_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::spaces::Space;

    #[derive(Clone, Copy)]
    struct MockAction;

    struct MockEnv {
        step_count: usize,
        max_steps: usize,
        reset_count: usize,
    }

    impl MockEnv {
        fn new(max_steps: usize) -> Self {
            Self {
                step_count: 0,
                max_steps,
                reset_count: 0,
            }
        }
    }

    impl Environment for MockEnv {
        type Obs = [f32; 1];
        type Act = MockAction;
        type Info = ();

        fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Self::Info) {
            self.reset_count += 1;
            self.step_count = 0;
            ([1.0], ())
        }

        fn step(&mut self, _action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
            self.step_count += 1;
            let terminated = self.step_count >= self.max_steps;

            ([2.0], 1.5, terminated, false, ())
        }

        fn action_space(&self) -> Space {
            Space::discrete(1)
        }

        fn observation_space(&self) -> Space {
            Space::continuous(vec![0.0], vec![10.0])
        }
    }

    #[test]
    fn test_sync_vector_env_new() {
        let envs = vec![MockEnv::new(2), MockEnv::new(2), MockEnv::new(2)];
        let vec_env = SyncVectorEnv::new(envs);

        assert_eq!(vec_env.num_envs(), 3);
        assert_eq!(vec_env.obs_dim(), 1);
        assert_eq!(vec_env.obs_buffer.len(), 3); // 3 envs * 1 dim
        assert_eq!(vec_env.reward_buffer.len(), 3);
        assert_eq!(vec_env.terminated_buffer.len(), 3);
        assert_eq!(vec_env.truncated_buffer.len(), 3);
        assert_eq!(vec_env.terminal_obs_buffer.len(), 3);
    }

    #[test]
    fn test_sync_vector_env_reset_all() {
        let envs = vec![MockEnv::new(2), MockEnv::new(2)];
        let mut vec_env = SyncVectorEnv::new(envs);

        let obs = vec_env.reset_all(Some(&[42, 43]));

        // Check if observations are correctly written to the buffer
        // MockEnv returns [1.0] on reset
        assert_eq!(obs, &[1.0, 1.0]);
        assert_eq!(vec_env.obs_buffer, &[1.0, 1.0]);

        // Verify reset_count incremented in inner envs
        assert_eq!(vec_env.envs[0].reset_count, 1);
        assert_eq!(vec_env.envs[1].reset_count, 1);
    }

    #[test]
    fn test_sync_vector_env_step_batch() {
        // Create 2 mock envs: Env 0 terminates at step 1, Env 1 terminates at step 2
        let envs = vec![MockEnv::new(1), MockEnv::new(2)];
        let mut vec_env = SyncVectorEnv::new(envs);

        vec_env.reset_all(None);

        let actions = vec![MockAction, MockAction];

        // Step 1
        let result = vec_env.step_batch(&actions);

        // Env 0: step_count=1 (max=1) -> terminated
        // Env 1: step_count=1 (max=2) -> not terminated
        assert_eq!(result.terminated, &[true, false]);
        assert_eq!(result.rewards, &[1.5, 1.5]);

        // Since Env 0 terminated, its terminal obs should be Some([2.0])
        // It then auto-resets, so the active obs buffer should have the new reset obs ([1.0])
        assert_eq!(result.terminal_obs[0], Some([2.0]));
        assert_eq!(result.obs[0], 1.0); // Env 0 reset obs

        // Env 1 didn't terminate, so it gets None and the stepped obs ([2.0])
        assert_eq!(result.terminal_obs[1], None);
        assert_eq!(result.obs[1], 2.0); // Env 1 step obs

        // Verify env reset counts
        assert_eq!(vec_env.envs[0].reset_count, 2); // Initial reset + Auto-reset
        assert_eq!(vec_env.envs[1].reset_count, 1); // Initial reset only

        // Step 2
        let result = vec_env.step_batch(&actions);

        // Env 0: step_count=1 (since it was reset) -> terminated (max=1)
        // Env 1: step_count=2 -> terminated (max=2)
        assert_eq!(result.terminated, &[true, true]);

        // Both terminated, both should have terminal obs [2.0]
        assert_eq!(result.terminal_obs[0], Some([2.0]));
        assert_eq!(result.terminal_obs[1], Some([2.0]));

        // Both should auto-reset and show [1.0] in the active buffer
        assert_eq!(result.obs, &[1.0, 1.0]);

        // Verify env reset counts
        assert_eq!(vec_env.envs[0].reset_count, 3);
        assert_eq!(vec_env.envs[1].reset_count, 2);
    }
}
