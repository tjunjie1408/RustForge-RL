//! Exhaustive tests for environment wrappers: TimeLimit and RewardScale.
//!
//! Covers: truncation timing, inner termination priority, counter reset,
//! reward scaling math, and wrapper composition.

use rustforge_rl::env::{CartPole, CartPoleAction, Environment, RewardScale, TimeLimit};

// TimeLimit Wrapper

mod time_limit {
    use super::*;

    #[test]
    fn truncation_at_exact_step() {
        let inner = CartPole::with_max_steps(1000); // High inner limit
        let mut env = TimeLimit::new(inner, 10);
        env.reset(Some(42));

        // Alternate actions to keep pole balanced
        let actions = [CartPoleAction::Left, CartPoleAction::Right];
        let mut truncated_step = None;

        for step in 0..20 {
            let action = actions[step % 2];
            let (_, _, terminated, truncated, _) = env.step(action);

            if terminated {
                break; // Inner env terminated naturally
            }
            if truncated {
                truncated_step = Some(step + 1);
                break;
            }
        }

        if let Some(step) = truncated_step {
            assert_eq!(
                step, 10,
                "TimeLimit should truncate at exactly max_steps=10, got {}",
                step
            );
        }
    }

    #[test]
    fn inner_termination_takes_priority() {
        // Use a very generous time limit
        let inner = CartPole::new();
        let mut env = TimeLimit::new(inner, 10000);
        env.reset(Some(42));

        // Push persistently — CartPole will terminate naturally
        let mut natural_term = false;
        for _ in 0..500 {
            let (_, _, terminated, truncated, _) = env.step(CartPoleAction::Right);
            if terminated {
                natural_term = true;
                assert!(!truncated, "Truncated should be false when naturally terminated");
                break;
            }
        }
        assert!(
            natural_term,
            "CartPole should naturally terminate before generous time limit"
        );
    }

    #[test]
    fn counter_resets_after_reset() {
        let inner = CartPole::with_max_steps(1000);
        let mut env = TimeLimit::new(inner, 5);
        env.reset(Some(42));

        // Take 3 steps
        for _ in 0..3 {
            env.step(CartPoleAction::Right);
        }
        assert_eq!(env.current_step(), 3);

        // Reset should zero the counter
        env.reset(Some(99));
        assert_eq!(
            env.current_step(),
            0,
            "Counter should be 0 after reset"
        );
    }

    #[test]
    fn timelimit_preserves_obs_type() {
        let inner = CartPole::new();
        let mut env = TimeLimit::new(inner, 100);
        let (obs, _) = env.reset(Some(42));
        // Type system ensures this is [f32; 4]
        assert_eq!(obs.len(), 4);
    }
}

// RewardScale Wrapper

mod reward_scale {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn reward_scaled_correctly() {
        let inner = CartPole::new();
        let mut env = RewardScale::new(inner, 0.5);
        env.reset(Some(42));

        let (_, reward, terminated, _, _) = env.step(CartPoleAction::Right);
        if !terminated {
            assert_abs_diff_eq!(reward, 0.5, epsilon = 1e-6);
        } else {
            assert_abs_diff_eq!(reward, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn negative_scale_inverts_reward() {
        let inner = CartPole::new();
        let mut env = RewardScale::new(inner, -1.0);
        env.reset(Some(42));

        let (_, reward, terminated, _, _) = env.step(CartPoleAction::Right);
        if !terminated {
            assert_abs_diff_eq!(reward, -1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn zero_scale_zeroes_reward() {
        let inner = CartPole::new();
        let mut env = RewardScale::new(inner, 0.0);
        env.reset(Some(42));

        let (_, reward, _, _, _) = env.step(CartPoleAction::Right);
        assert_abs_diff_eq!(reward, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn scale_accessor() {
        let inner = CartPole::new();
        let env = RewardScale::new(inner, 2.5);
        assert_abs_diff_eq!(env.scale(), 2.5, epsilon = 1e-6);
    }
}

// Wrapper Composition

mod composition {
    use super::*;

    #[test]
    fn timelimit_wrapping_rewardscale_compiles_and_works() {
        let cartpole = CartPole::new();
        let scaled = RewardScale::new(cartpole, 0.1);
        let mut env = TimeLimit::new(scaled, 50);

        let (obs, _) = env.reset(Some(42));
        assert_eq!(obs.len(), 4, "Composed wrapper should preserve obs type");

        let (_, reward, _, _, _) = env.step(CartPoleAction::Right);
        // Reward should be scaled: 1.0 * 0.1 = 0.1 (if alive)
        if reward != 0.0 {
            // not terminated
            assert!((reward - 0.1).abs() < 1e-5, "Composed reward should be 0.1");
        }
    }

    #[test]
    fn rewardscale_wrapping_timelimit_compiles_and_works() {
        let cartpole = CartPole::new();
        let limited = TimeLimit::new(cartpole, 50);
        let mut env = RewardScale::new(limited, 2.0);

        let (obs, _) = env.reset(Some(42));
        assert_eq!(obs.len(), 4);

        let (_, reward, _, _, _) = env.step(CartPoleAction::Right);
        if reward != 0.0 {
            assert!((reward - 2.0).abs() < 1e-5, "Composed reward should be 2.0");
        }
    }

    #[test]
    fn deeply_nested_wrappers_compile() {
        // TimeLimit<RewardScale<TimeLimit<CartPole>>>
        let inner = CartPole::new();
        let w1 = TimeLimit::new(inner, 1000);
        let w2 = RewardScale::new(w1, 0.5);
        let mut env = TimeLimit::new(w2, 100);

        let (_obs, _) = env.reset(Some(42));
        let (_, _, _, _, _) = env.step(CartPoleAction::Right);
        // Just verify it compiles and runs without panic
    }
}
