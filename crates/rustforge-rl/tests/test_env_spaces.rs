//! Exhaustive tests for `Space` — Discrete, Box, MultiDiscrete.
//!
//! Covers: sampling validity, boundary checks, edge cases, and proptest fuzzing.

use proptest::prelude::*;
use rustforge_rl::env::Space;

mod discrete_space {
    use super::*;

    #[test]
    fn discrete_1_sample_always_zero() {
        let space = Space::discrete(1);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let sample = space.sample(&mut rng);
            assert_eq!(sample.len(), 1);
            assert_eq!(sample[0], 0.0, "Discrete(1) sample must always be 0");
        }
    }

    #[test]
    #[should_panic(expected = "Discrete space must have at least 1 action")]
    fn discrete_zero_panics() {
        Space::discrete(0);
    }

    #[test]
    fn discrete_contains_valid() {
        let space = Space::discrete(4);
        assert!(space.contains(&[0.0]));
        assert!(space.contains(&[1.0]));
        assert!(space.contains(&[3.0]));
    }

    #[test]
    fn discrete_contains_invalid() {
        let space = Space::discrete(4);
        assert!(!space.contains(&[4.0]), "4 is out of range for Discrete(4)");
        assert!(!space.contains(&[-1.0]), "Negative is invalid");
        assert!(!space.contains(&[1.5]), "Non-integer is invalid");
        assert!(!space.contains(&[0.0, 1.0]), "Wrong length");
    }

    #[test]
    fn discrete_dim_is_one() {
        let space = Space::discrete(10);
        assert_eq!(space.dim(), 1);
    }
}

mod box_space {
    use super::*;

    #[test]
    fn box_contains_boundary_exact() {
        let space = Space::continuous(vec![0.0, -1.0], vec![1.0, 1.0]);
        // Exact boundary values should be contained
        assert!(space.contains(&[0.0, -1.0]), "Lower bound inclusive");
        assert!(space.contains(&[1.0, 1.0]), "Upper bound inclusive");
        assert!(space.contains(&[0.5, 0.0]), "Interior point");
    }

    #[test]
    fn box_contains_outside() {
        let space = Space::continuous(vec![0.0, -1.0], vec![1.0, 1.0]);
        assert!(!space.contains(&[1.01, 0.0]), "Above upper bound");
        assert!(!space.contains(&[-0.01, 0.0]), "Below lower bound");
        assert!(!space.contains(&[0.5]), "Wrong dimensionality");
    }

    #[test]
    #[should_panic(expected = "low and high must have same length")]
    fn box_mismatched_lengths_panics() {
        Space::continuous(vec![0.0, 0.0], vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "must be <=")]
    fn box_inverted_bounds_panics() {
        Space::continuous(vec![1.0], vec![0.0]);
    }

    #[test]
    fn box_degenerate_point_space() {
        // low == high: only a single point is valid
        let space = Space::continuous(vec![0.5], vec![0.5]);
        assert!(space.contains(&[0.5]));
        assert!(!space.contains(&[0.6]));
    }

    #[test]
    fn box_dim() {
        let space = Space::continuous(vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(space.dim(), 4);
    }
}

mod multi_discrete_space {
    use super::*;

    #[test]
    fn multi_discrete_contains_valid() {
        let space = Space::MultiDiscrete(vec![3, 5, 2]);
        assert!(space.contains(&[0.0, 0.0, 0.0]));
        assert!(space.contains(&[2.0, 4.0, 1.0]));
    }

    #[test]
    fn multi_discrete_contains_invalid() {
        let space = Space::MultiDiscrete(vec![3, 5, 2]);
        assert!(!space.contains(&[3.0, 0.0, 0.0]), "Out of range");
        assert!(!space.contains(&[0.0, 5.0, 0.0]), "Out of range");
        assert!(!space.contains(&[0.0, 0.0]), "Wrong length");
    }

    #[test]
    fn multi_discrete_dim() {
        let space = Space::MultiDiscrete(vec![3, 5, 2]);
        assert_eq!(space.dim(), 3);
    }
}

proptest! {
    /// Fuzz test: random Box space bounds → sample always within bounds.
    #[test]
    fn proptest_box_sample_within_bounds(
        low0 in -100.0f32..0.0f32,
        low1 in -100.0f32..0.0f32,
        range0 in 0.01f32..200.0f32,
        range1 in 0.01f32..200.0f32,
    ) {
        let high0 = low0 + range0;
        let high1 = low1 + range1;
        let space = Space::continuous(vec![low0, low1], vec![high0, high1]);
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let sample = space.sample(&mut rng);
            prop_assert!(
                space.contains(&sample),
                "Sample {:?} should be within bounds [{}, {}] x [{}, {}]",
                sample, low0, high0, low1, high1
            );
        }
    }

    /// Fuzz test: random Discrete(n) → sample always valid integer in [0, n).
    #[test]
    fn proptest_discrete_sample_valid(n in 1usize..1000) {
        let space = Space::discrete(n);
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let sample = space.sample(&mut rng);
            prop_assert!(
                space.contains(&sample),
                "Discrete({}) sample {:?} should be valid",
                n, sample
            );
        }
    }

    /// Fuzz test: random MultiDiscrete → sample always valid.
    #[test]
    fn proptest_multi_discrete_sample_valid(
        n0 in 1usize..100,
        n1 in 1usize..100,
        n2 in 1usize..100,
    ) {
        let space = Space::MultiDiscrete(vec![n0, n1, n2]);
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let sample = space.sample(&mut rng);
            prop_assert!(
                space.contains(&sample),
                "MultiDiscrete({:?}) sample {:?} should be valid",
                vec![n0, n1, n2], sample
            );
        }
    }
}
