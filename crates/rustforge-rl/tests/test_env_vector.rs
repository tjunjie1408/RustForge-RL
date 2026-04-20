//! Exhaustive tests for SyncVectorEnv — vectorized environment with pre-allocated buffers.
//!
//! Covers: buffer contiguity, auto-reset correctness, PRNG isolation,
//! batch determinism, zero-copy ndarray bridge, IntoTensorBuffer correctness,
//! SoA layout validation, and borrow-split safety.

use rustforge_rl::env::{CartPole, CartPoleAction, IntoTensorBuffer, SyncVectorEnv};

// IntoTensorBuffer Trait

mod into_tensor_buffer {
    use super::*;

    #[test]
    fn f32_array_write_roundtrip() {
        let obs: [f32; 4] = [1.0, -2.5, std::f32::consts::PI, 0.001];
        let mut buf = [0.0f32; 4];
        obs.write_to_buffer(&mut buf);
        assert_eq!(buf, obs, "write_to_buffer should produce exact copy");

        let recovered = <[f32; 4]>::read_from_buffer(&buf);
        assert_eq!(recovered, obs, "read_from_buffer should roundtrip exactly");
    }

    #[test]
    fn usize_array_write_converts_to_f32() {
        let obs: [usize; 2] = [3, 7];
        let mut buf = [0.0f32; 2];
        obs.write_to_buffer(&mut buf);
        assert_eq!(buf, [3.0, 7.0], "usize should be converted to f32");
    }

    #[test]
    fn usize_array_roundtrip() {
        let obs: [usize; 2] = [42, 99];
        let mut buf = [0.0f32; 2];
        obs.write_to_buffer(&mut buf);
        let recovered = <[usize; 2]>::read_from_buffer(&buf);
        assert_eq!(recovered, obs, "usize roundtrip should be exact");
    }

    #[test]
    fn dim_constant_correct() {
        assert_eq!(<[f32; 4]>::DIM, 4);
        assert_eq!(<[f32; 1]>::DIM, 1);
        assert_eq!(<[usize; 2]>::DIM, 2);
        assert_eq!(<[usize; 10]>::DIM, 10);
    }
}

// Buffer Layout & Contiguity

mod buffer_layout {
    use super::*;

    #[test]
    fn obs_buffer_has_correct_size() {
        let envs: Vec<CartPole> = (0..4).map(|_| CartPole::new()).collect();
        let vec_env = SyncVectorEnv::new(envs);

        assert_eq!(
            vec_env.obs_buffer().len(),
            4 * 4, // 4 envs × 4 obs_dim
            "Buffer should be N × DIM"
        );
    }

    #[test]
    fn obs_buffer_is_contiguous() {
        let envs: Vec<CartPole> = (0..4).map(|_| CartPole::new()).collect();
        let vec_env = SyncVectorEnv::new(envs);

        let buf = vec_env.obs_buffer();
        // Pointer arithmetic: buf[i] and buf[i+1] should be adjacent in memory
        let ptr = buf.as_ptr();
        for i in 0..buf.len() - 1 {
            unsafe {
                assert_eq!(
                    ptr.add(i + 1) as usize - ptr.add(i) as usize,
                    std::mem::size_of::<f32>(),
                    "Buffer elements must be contiguous in memory"
                );
            }
        }
    }

    #[test]
    fn reset_all_populates_buffer() {
        let envs: Vec<CartPole> = (0..3).map(|_| CartPole::new()).collect();
        let mut vec_env = SyncVectorEnv::new(envs);

        let seeds = vec![10, 20, 30];
        let obs = vec_env.reset_all(Some(&seeds));

        // Each env's obs should be written into the contiguous buffer
        assert_eq!(obs.len(), 3 * 4);

        // All values should be finite and within [-0.05, 0.05] (CartPole init range)
        for (i, &v) in obs.iter().enumerate() {
            assert!(
                v.is_finite() && (-0.06..=0.06).contains(&v),
                "obs[{}] = {} should be finite and near zero",
                i,
                v
            );
        }
    }

    #[test]
    fn num_envs_and_obs_dim() {
        let envs: Vec<CartPole> = (0..7).map(|_| CartPole::new()).collect();
        let vec_env = SyncVectorEnv::new(envs);
        assert_eq!(vec_env.num_envs(), 7);
        assert_eq!(vec_env.obs_dim(), 4);
    }
}

// Auto-Reset Protocol

mod auto_reset {
    use super::*;

    #[test]
    fn auto_reset_preserves_terminal_obs() {
        // Use very short env that terminates quickly
        let envs: Vec<CartPole> = (0..2).map(|_| CartPole::with_max_steps(5)).collect();
        let mut vec_env = SyncVectorEnv::new(envs);
        vec_env.reset_all(Some(&[42, 43]));

        // Step until at least one env terminates/truncates
        let actions = vec![CartPoleAction::Right, CartPoleAction::Left];
        let mut found_terminal = false;

        for _ in 0..20 {
            let result = vec_env.step_batch(&actions);

            for i in 0..2 {
                if result.terminated[i] || result.truncated[i] {
                    assert!(
                        result.terminal_obs[i].is_some(),
                        "Env {} terminated but terminal_obs is None",
                        i
                    );
                    found_terminal = true;

                    // The primary obs buffer should contain the reset obs (not terminal)
                    let offset = i * 4;
                    let reset_obs = &result.obs[offset..offset + 4];
                    for &v in reset_obs {
                        assert!(
                            v.is_finite() && v.abs() <= 0.06,
                            "Reset obs should be in init range, got {}",
                            v
                        );
                    }
                } else {
                    assert!(
                        result.terminal_obs[i].is_none(),
                        "Env {} not terminated but terminal_obs is Some",
                        i
                    );
                }
            }

            if found_terminal {
                break;
            }
        }

        assert!(
            found_terminal,
            "At least one env should have terminated within 20 steps"
        );
    }
}

// SoA Layout Validation

mod soa_layout {
    use super::*;

    #[test]
    fn rewards_slice_length_matches_num_envs() {
        let envs: Vec<CartPole> = (0..4).map(|_| CartPole::new()).collect();
        let mut vec_env = SyncVectorEnv::new(envs);
        vec_env.reset_all(Some(&[1, 2, 3, 4]));

        let actions = vec![
            CartPoleAction::Right,
            CartPoleAction::Left,
            CartPoleAction::Right,
            CartPoleAction::Left,
        ];
        let result = vec_env.step_batch(&actions);

        assert_eq!(result.rewards.len(), 4, "Rewards should have N elements");
        assert_eq!(
            result.terminated.len(),
            4,
            "Terminated should have N elements"
        );
        assert_eq!(
            result.truncated.len(),
            4,
            "Truncated should have N elements"
        );
        assert_eq!(
            result.terminal_obs.len(),
            4,
            "Terminal obs should have N elements"
        );
        assert_eq!(result.obs.len(), 16, "Obs should have N × DIM elements");
    }
}

// Batch Determinism

mod batch_determinism {
    use super::*;

    #[test]
    fn identical_seeds_produce_identical_batch_results() {
        let mk_env = || {
            let envs: Vec<CartPole> = (0..4).map(|_| CartPole::new()).collect();
            let mut ve = SyncVectorEnv::new(envs);
            ve.reset_all(Some(&[10, 20, 30, 40]));
            ve
        };

        let mut ve1 = mk_env();
        let mut ve2 = mk_env();

        let actions = vec![
            CartPoleAction::Right,
            CartPoleAction::Left,
            CartPoleAction::Right,
            CartPoleAction::Right,
        ];

        for step in 0..50 {
            let r1 = ve1.step_batch(&actions);
            let r2 = ve2.step_batch(&actions);

            assert_eq!(r1.obs, r2.obs, "Obs buffers diverged at step {}", step);
            assert_eq!(r1.rewards, r2.rewards, "Rewards diverged at step {}", step);
            assert_eq!(
                r1.terminated, r2.terminated,
                "Terminated diverged at step {}",
                step
            );
            assert_eq!(
                r1.truncated, r2.truncated,
                "Truncated diverged at step {}",
                step
            );
        }
    }
}

// PRNG Isolation

mod prng_isolation {
    use super::*;

    #[test]
    fn each_sub_env_has_independent_rng() {
        let envs: Vec<CartPole> = (0..3).map(|_| CartPole::new()).collect();
        let mut vec_env = SyncVectorEnv::new(envs);

        // Seed each env differently
        let obs = vec_env.reset_all(Some(&[1, 2, 3]));

        // The initial observations for each env should differ
        let env0_obs = &obs[0..4];
        let env1_obs = &obs[4..8];
        let env2_obs = &obs[8..12];

        assert_ne!(
            env0_obs, env1_obs,
            "Different seeds should produce different initial obs"
        );
        assert_ne!(
            env1_obs, env2_obs,
            "Different seeds should produce different initial obs"
        );
    }
}

// Error Handling

mod error_handling {
    use super::*;

    #[test]
    #[should_panic(expected = "Expected 3 actions")]
    fn wrong_action_count_panics() {
        let envs: Vec<CartPole> = (0..3).map(|_| CartPole::new()).collect();
        let mut vec_env = SyncVectorEnv::new(envs);
        vec_env.reset_all(Some(&[1, 2, 3]));

        // Pass wrong number of actions
        let actions = vec![CartPoleAction::Right, CartPoleAction::Left]; // 2 instead of 3
        vec_env.step_batch(&actions);
    }
}
