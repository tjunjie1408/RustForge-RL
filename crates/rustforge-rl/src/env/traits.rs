//! Core `Environment` trait and `IntoTensorBuffer` bridge for zero-cost type conversion.
//!
//! ## Design Decisions
//!
//! - **Associated types** (`Obs`, `Act`, `Info`) instead of enums — compile-time type safety,
//!   zero dynamic dispatch overhead, and monomorphization for maximum performance.
//! - **`IntoTensorBuffer`** bridge trait — enables heterogeneous `Obs` types (e.g., `[f32; 4]`,
//!   `[usize; 2]`) to be written into a contiguous `f32` neural network buffer at zero cost.
//!   `[f32; N]::write_to_buffer` compiles down to a single `memcpy`.
//! - **`reset(seed: Option<u64>)`** semantics:
//!   - `Some(seed)`: re-instantiate internal PRNG from this seed (full deterministic replay).
//!   - `None`: continue existing PRNG stream, reset physics state only. This preserves
//!     the pseudorandom sequence across episode boundaries for absolute reproducibility.
//! - **Gymnasium v1.0 step return**: `(obs, reward, terminated, truncated, info)` with explicit
//!   separation of terminal states (goal/failure) vs truncation (time limit).

use super::spaces::Space;

/// Bridge trait: enables zero-cost conversion of heterogeneous `Obs` types
/// into a contiguous `f32` neural network buffer.
///
/// # Monomorphization Guarantees
///
/// - `[f32; 4]::write_to_buffer()` → inlined `memcpy` (16 bytes)
/// - `[usize; 2]::write_to_buffer()` → inlined `as f32` cast loop (2 iterations)
/// - `const DIM` enables compile-time buffer size calculation in `SyncVectorEnv`
///
/// # Example
/// ```rust,ignore
/// use rustforge_rl::env::IntoTensorBuffer;
///
/// let obs: [f32; 4] = [1.0, 0.5, -0.3, 0.1];
/// let mut buf = [0.0f32; 4];
/// obs.write_to_buffer(&mut buf);
/// assert_eq!(buf, obs);
/// ```
pub trait IntoTensorBuffer {
    /// Dimensionality of this observation when flattened to f32.
    const DIM: usize;

    /// Write this observation into a pre-allocated f32 slice.
    ///
    /// Caller guarantees `buffer.len() == Self::DIM`.
    fn write_to_buffer(&self, buffer: &mut [f32]);

    /// Read an observation back from a f32 slice.
    ///
    /// Caller guarantees `buffer.len() == Self::DIM`.
    fn read_from_buffer(buffer: &[f32]) -> Self;
}

// Zero-cost implementation for [f32; N] — compiles to memcpy
impl<const N: usize> IntoTensorBuffer for [f32; N] {
    const DIM: usize = N;

    #[inline]
    fn write_to_buffer(&self, buffer: &mut [f32]) {
        buffer.copy_from_slice(self);
    }

    #[inline]
    fn read_from_buffer(buffer: &[f32]) -> Self {
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(buffer);
        arr
    }
}

// Type-converting implementation for [usize; N] — element-wise as f32
impl<const N: usize> IntoTensorBuffer for [usize; N] {
    const DIM: usize = N;

    #[inline]
    fn write_to_buffer(&self, buffer: &mut [f32]) {
        for (dst, &src) in buffer.iter_mut().zip(self.iter()) {
            *dst = src as f32;
        }
    }

    #[inline]
    fn read_from_buffer(buffer: &[f32]) -> Self {
        let mut arr = [0usize; N];
        for (dst, &src) in arr.iter_mut().zip(buffer.iter()) {
            *dst = src as usize;
        }
        arr
    }
}

/// Core environment trait with full associated-type safety.
///
/// # Associated Types
///
/// - `Obs`: Observation type returned by `reset` and `step`. Must implement
///   `IntoTensorBuffer` for integration with the vectorized environment layer.
/// - `Act`: Action type accepted by `step`. Uses exhaustive enums for compile-time
///   validity (e.g., `CartPoleAction::Left`, `GridAction::Up`).
/// - `Info`: Auxiliary information type. Use `()` (zero-size type) for lightweight
///   environments — compiled away entirely. Use custom structs for diagnostic-rich envs.
///
/// # Example
/// ```rust,ignore
/// use rustforge_rl::env::{Environment, CartPole, CartPoleAction};
///
/// let mut env = CartPole::new();
/// let (obs, _info) = env.reset(Some(42));
/// let (next_obs, reward, terminated, truncated, _info) = env.step(CartPoleAction::Right);
/// ```
pub trait Environment {
    /// Observation type (e.g., `[f32; 4]` for CartPole).
    type Obs: Clone + IntoTensorBuffer;

    /// Action type (e.g., `CartPoleAction` enum for CartPole).
    type Act: Clone;

    /// Auxiliary info type (e.g., `()` for lightweight environments).
    type Info;

    /// Reset the environment to an initial state.
    ///
    /// - `Some(seed)`: re-instantiate internal PRNG from this seed.
    /// - `None`: continue existing PRNG stream (reset physics only).
    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info);

    /// Advance one timestep with the given action.
    ///
    /// Returns `(observation, reward, terminated, truncated, info)`:
    /// - `terminated`: episode ended due to reaching a terminal state (goal/failure).
    /// - `truncated`: episode ended due to a time limit or boundary condition.
    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info);

    /// Returns the action space descriptor (for neural network output sizing).
    fn action_space(&self) -> Space;

    /// Returns the observation space descriptor (for neural network input sizing).
    fn observation_space(&self) -> Space;
}
