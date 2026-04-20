//! Action and observation space descriptors.
//!
//! `Space` provides metadata about the shape and bounds of action/observation spaces.
//! This is used by downstream neural network layers to automatically size their
//! input/output dimensions, and by fuzzing tests to validate sampling.
//!
//! ## Design Notes
//!
//! - `Space` is a metadata descriptor, NOT the runtime action/observation type.
//!   The actual types are defined as associated types on `Environment`.
//! - `sample()` is provided for exploration strategies (e.g., ε-greedy).
//! - `contains()` is provided for fuzzing/correctness assertions.

use rand::Rng;

/// Descriptor for action or observation spaces.
///
/// This enum captures the shape and bounds of a space for metadata purposes
/// (e.g., neural network sizing). The actual runtime types are governed by
/// the `Environment` trait's associated types.
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    /// Discrete space with `n` possible values: {0, 1, ..., n-1}.
    ///
    /// Used for environments with a finite number of actions (e.g., CartPole: Left/Right).
    Discrete(usize),

    /// Continuous box space with element-wise lower and upper bounds.
    ///
    /// Each element `i` of an observation/action must satisfy `low[i] <= x[i] <= high[i]`.
    /// `shape` describes the dimensionality.
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
        shape: Vec<usize>,
    },

    /// Multi-discrete space: multiple independent discrete sub-spaces.
    ///
    /// `nvec[i]` is the number of possible values for the i-th sub-action.
    MultiDiscrete(Vec<usize>),
}

impl Space {
    /// Create a new Discrete space.
    ///
    /// # Panics
    /// If `n == 0` (a discrete space must have at least one action).
    pub fn discrete(n: usize) -> Self {
        assert!(n > 0, "Discrete space must have at least 1 action, got 0");
        Space::Discrete(n)
    }

    /// Create a new Box (continuous) space.
    ///
    /// # Panics
    /// - If `low` and `high` have different lengths.
    /// - If any `low[i] > high[i]`.
    pub fn continuous(low: Vec<f32>, high: Vec<f32>) -> Self {
        assert_eq!(
            low.len(),
            high.len(),
            "Box space: low and high must have same length, got {} vs {}",
            low.len(),
            high.len()
        );
        for (i, (&l, &h)) in low.iter().zip(high.iter()).enumerate() {
            assert!(
                l <= h,
                "Box space: low[{}]={} must be <= high[{}]={}",
                i,
                l,
                i,
                h
            );
        }
        let shape = vec![low.len()];
        Space::Box { low, high, shape }
    }

    /// Sample a random value from this space as a `Vec<f32>`.
    ///
    /// For `Discrete(n)`: returns a single-element vec with a random integer in [0, n).
    /// For `Box`: returns a vec with each element uniformly sampled within bounds.
    /// For `MultiDiscrete`: returns a vec with each element sampled from its sub-space.
    pub fn sample(&self, rng: &mut impl Rng) -> Vec<f32> {
        match self {
            Space::Discrete(n) => {
                vec![rng.gen_range(0..*n) as f32]
            }
            Space::Box { low, high, .. } => low
                .iter()
                .zip(high.iter())
                .map(|(&l, &h)| rng.gen_range(l..=h))
                .collect(),
            Space::MultiDiscrete(nvec) => {
                nvec.iter().map(|&n| rng.gen_range(0..n) as f32).collect()
            }
        }
    }

    /// Check if a given value (as `&[f32]`) is contained within this space.
    ///
    /// For `Discrete(n)`: checks value is a single integer in [0, n).
    /// For `Box`: checks each element is within [low, high].
    /// For `MultiDiscrete`: checks each element is a valid integer in its sub-space.
    pub fn contains(&self, value: &[f32]) -> bool {
        match self {
            Space::Discrete(n) => {
                if value.len() != 1 {
                    return false;
                }
                let v = value[0];
                v >= 0.0 && v < *n as f32 && v == v.floor()
            }
            Space::Box { low, high, .. } => {
                if value.len() != low.len() {
                    return false;
                }
                value
                    .iter()
                    .zip(low.iter().zip(high.iter()))
                    .all(|(&v, (&l, &h))| v >= l && v <= h)
            }
            Space::MultiDiscrete(nvec) => {
                if value.len() != nvec.len() {
                    return false;
                }
                value
                    .iter()
                    .zip(nvec.iter())
                    .all(|(&v, &n)| v >= 0.0 && v < n as f32 && v == v.floor())
            }
        }
    }

    /// Returns the total dimensionality of the space (flattened).
    pub fn dim(&self) -> usize {
        match self {
            Space::Discrete(_) => 1,
            Space::Box { low, .. } => low.len(),
            Space::MultiDiscrete(nvec) => nvec.len(),
        }
    }
}
