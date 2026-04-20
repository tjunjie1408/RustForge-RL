//! Exhaustive tests for CartPole-v1 environment.
//!
//! Covers: determinism, PRNG continuity, reward invariant, termination conditions,
//! NaN/Inf injection, extreme values, and action enum exhaustiveness.

use rustforge_rl::env::{CartPole, CartPoleAction, Environment};

// Determinism & Reproducibility

mod determinism {
    use super::*;

    #[test]
    fn seeded_environments_produce_identical_trajectories() {
        let mut env1 = CartPole::new();
        let mut env2 = CartPole::new();

        env1.reset(Some(42));
        env2.reset(Some(42));

        let actions = [
            CartPoleAction::Right,
            CartPoleAction::Left,
            CartPoleAction::Right,
            CartPoleAction::Right,
            CartPoleAction::Left,
        ];

        // Run 1000 steps with deterministic action sequence
        for step in 0..1000 {
            let action = actions[step % actions.len()];
            let (obs1, r1, t1, tr1, _) = env1.step(action);
            let (obs2, r2, t2, tr2, _) = env2.step(action);

            assert_eq!(obs1, obs2, "Observations diverged at step {}", step);
            assert_eq!(r1, r2, "Rewards diverged at step {}", step);
            assert_eq!(t1, t2, "Terminated diverged at step {}", step);
            assert_eq!(tr1, tr2, "Truncated diverged at step {}", step);

            if t1 || tr1 {
                env1.reset(None);
                env2.reset(None);
            }
        }
    }

    #[test]
    fn prng_continuity_across_reset_none() {
        // Scenario: env_a does reset(42) → episode → reset(None) → episode
        // env_b does the same, should produce identical results
        let mut env_a = CartPole::new();
        let mut env_b = CartPole::new();

        let (obs_a1, _) = env_a.reset(Some(42));
        let (obs_b1, _) = env_b.reset(Some(42));
        assert_eq!(obs_a1, obs_b1, "Initial obs should match");

        // Run a few steps
        for _ in 0..5 {
            env_a.step(CartPoleAction::Right);
            env_b.step(CartPoleAction::Right);
        }

        // Both reset with None — should continue same PRNG stream
        let (obs_a2, _) = env_a.reset(None);
        let (obs_b2, _) = env_b.reset(None);
        assert_eq!(obs_a2, obs_b2, "Post-reset(None) obs should match");

        // Continue stepping — should remain identical
        for step in 0..100 {
            let (obs_a, _, _, _, _) = env_a.step(CartPoleAction::Left);
            let (obs_b, _, _, _, _) = env_b.step(CartPoleAction::Left);
            assert_eq!(
                obs_a, obs_b,
                "Obs diverged at step {} after reset(None)",
                step
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_trajectories() {
        let mut env1 = CartPole::new();
        let mut env2 = CartPole::new();

        let (obs1, _) = env1.reset(Some(42));
        let (obs2, _) = env2.reset(Some(99));

        // With different seeds, initial observations should differ
        assert_ne!(
            obs1, obs2,
            "Different seeds should produce different initial states"
        );
    }
}

// Reward Invariant

mod reward {
    use super::*;

    #[test]
    fn reward_is_one_while_alive() {
        let mut env = CartPole::new();
        env.reset(Some(42));

        for _ in 0..100 {
            let (_, reward, terminated, _, _) = env.step(CartPoleAction::Right);
            if terminated {
                // When terminated, reward should be 0.0
                assert_eq!(reward, 0.0, "Reward should be 0.0 on termination");
                break;
            } else {
                assert_eq!(reward, 1.0, "Reward should be 1.0 while alive");
            }
        }
    }

    #[test]
    fn reward_zero_on_termination() {
        let mut env = CartPole::new();
        env.reset(Some(42));

        // Keep pushing right until termination
        loop {
            let (_, reward, terminated, truncated, _) = env.step(CartPoleAction::Right);
            if terminated {
                assert_eq!(reward, 0.0, "Terminated reward must be 0.0");
                break;
            }
            if truncated {
                break; // Max steps reached without natural termination
            }
        }
    }
}

// Termination Conditions

mod termination {
    use super::*;

    #[test]
    fn terminates_when_cart_out_of_bounds() {
        let mut env = CartPole::new();
        env.reset(Some(42));

        // Push right persistently — cart will eventually go out of bounds
        let mut terminated = false;
        for _ in 0..500 {
            let result = env.step(CartPoleAction::Right);
            if result.2 {
                // terminated
                terminated = true;
                break;
            }
        }
        assert!(
            terminated,
            "CartPole should terminate when pushed persistently in one direction"
        );
    }

    #[test]
    fn truncation_at_max_steps() {
        // Use a very short max_steps to trigger truncation quickly
        let mut env = CartPole::with_max_steps(10);
        env.reset(Some(42));

        let actions = [CartPoleAction::Left, CartPoleAction::Right];
        let mut truncated_at = None;

        for step in 0..20 {
            let action = actions[step % 2];
            let (_, _, terminated, truncated, _) = env.step(action);

            if terminated {
                break; // Natural termination
            }
            if truncated {
                truncated_at = Some(step + 1); // 1-indexed step count
                break;
            }
        }

        if let Some(step) = truncated_at {
            assert_eq!(
                step, 10,
                "Episode should be truncated at exactly max_steps=10, got {}",
                step
            );
        }
        // If naturally terminated before 10 steps, that's also acceptable
    }
}

// Floating-Point Defense (NaN / Inf Injection)

mod fp_defense {
    use super::*;

    #[test]
    fn nan_injection_terminates_gracefully() {
        let mut env = CartPole::new();
        env.reset(Some(42));

        // Inject NaN into internal state via test-only accessor
        // We need to step once first to establish the env, then corrupt state
        env.step(CartPoleAction::Right);

        // Force NaN into state
        *env.state_mut() = [f32::NAN, 0.0, 0.0, 0.0];

        let (obs, _reward, terminated, _, _) = env.step(CartPoleAction::Right);

        assert!(
            terminated,
            "NaN-poisoned state should trigger immediate termination"
        );
        // Returned obs should be finite (safe default)
        for (i, &v) in obs.iter().enumerate() {
            assert!(
                v.is_finite(),
                "obs[{}] = {} should be finite after NaN poisoning",
                i,
                v
            );
        }
    }

    #[test]
    fn inf_injection_terminates_gracefully() {
        let mut env = CartPole::new();
        env.reset(Some(42));
        env.step(CartPoleAction::Right);

        *env.state_mut() = [f32::INFINITY, 0.0, 0.0, 0.0];

        let (obs, _reward, terminated, _, _) = env.step(CartPoleAction::Right);

        assert!(
            terminated,
            "Inf-poisoned state should trigger immediate termination"
        );
        for (i, &v) in obs.iter().enumerate() {
            assert!(
                v.is_finite(),
                "obs[{}] = {} should be finite after Inf poisoning",
                i,
                v
            );
        }
    }

    #[test]
    fn neg_inf_injection_terminates_gracefully() {
        let mut env = CartPole::new();
        env.reset(Some(42));
        env.step(CartPoleAction::Right);

        *env.state_mut() = [0.0, f32::NEG_INFINITY, 0.0, 0.0];

        let (obs, _reward, terminated, _, _) = env.step(CartPoleAction::Right);

        assert!(terminated, "NEG_INFINITY should trigger termination");
        for (i, &v) in obs.iter().enumerate() {
            assert!(v.is_finite(), "obs[{}] = {} should be finite", i, v);
        }
    }

    #[test]
    fn extreme_value_no_nan_propagation() {
        let mut env = CartPole::new();
        env.reset(Some(42));
        env.step(CartPoleAction::Right);

        // Inject extreme but finite value — physics might overflow to NaN
        *env.state_mut() = [1e35, 0.0, 0.0, 0.0];

        let (obs, _, terminated, _, _) = env.step(CartPoleAction::Right);

        // Should either terminate cleanly or produce finite output
        if !terminated {
            for (i, &v) in obs.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "obs[{}] = {} should be finite with extreme input",
                    i,
                    v
                );
            }
        }
    }
}

// Action Enum & Space

mod action_space {
    use super::*;

    #[test]
    fn action_enum_is_exhaustive() {
        // This test verifies at compile time that Left and Right are the only variants.
        // If someone adds a third variant, the match will fail to compile.
        let action = CartPoleAction::Left;
        let _force = match action {
            CartPoleAction::Left => -10.0f32,
            CartPoleAction::Right => 10.0f32,
        };
    }

    #[test]
    fn action_space_is_discrete_2() {
        let env = CartPole::new();
        assert_eq!(env.action_space(), rustforge_rl::env::Space::discrete(2));
    }

    #[test]
    fn observation_space_dimensionality() {
        let env = CartPole::new();
        assert_eq!(env.observation_space().dim(), 4);
    }
}

// Reset Behavior

mod reset_behavior {
    use super::*;

    #[test]
    fn reset_returns_finite_obs() {
        let mut env = CartPole::new();
        for seed in 0..100 {
            let (obs, _) = env.reset(Some(seed));
            for (i, &v) in obs.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Reset obs[{}] = {} should be finite for seed {}",
                    i,
                    v,
                    seed
                );
            }
        }
    }

    #[test]
    fn reset_initial_state_within_bounds() {
        let mut env = CartPole::new();
        for seed in 0..100 {
            let (obs, _) = env.reset(Some(seed));
            for (i, &v) in obs.iter().enumerate() {
                assert!(
                    (-0.05..=0.05).contains(&v),
                    "Reset obs[{}] = {} should be in [-0.05, 0.05] for seed {}",
                    i,
                    v,
                    seed
                );
            }
        }
    }
}
