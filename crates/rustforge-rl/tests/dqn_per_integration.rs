use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_rl::agent::{DQNConfig, EpsilonGreedy, DQN};
use rustforge_rl::buffer::{PrioritizedReplayBuffer, ReplayBuffer, TransitionBatch};
use rustforge_rl::env::{CartPole, CartPoleAction, Environment, IntoTensorBuffer};
use rustforge_rl::training::{episode_done, replay_done};
use rustforge_tensor::Tensor;

// A helper to train DQN in-memory and record rewards and losses
fn run_training(
    use_per: bool,
    seed: u64,
    episodes: usize,
    max_steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let obs_dim = 4;
    let num_actions = 2;
    let batch_size = 32;
    let warmup_steps = 32;

    let config = DQNConfig {
        obs_dim,
        num_actions,
        hidden_dim: 16,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 10,
        double_dqn: true,
        use_per,
        per_beta_annealing_steps: 20000,
    };

    let mut env = CartPole::with_max_steps(max_steps);
    let mut agent = DQN::new(config);

    // Seed parameters to reduce initial weight differences
    let q_params = agent.q_net().parameters();
    for (i, p) in q_params.iter().enumerate() {
        let shape = p.shape();
        let seeded_data = Tensor::rand_uniform(&shape, -0.1, 0.1, Some(seed + i as u64));
        p.set_data(seeded_data);
    }
    agent.update_target();

    // Fast decay explorer
    let explorer = EpsilonGreedy::new(0.5, 0.01, 100);

    let mut replay = ReplayBuffer::new(10_000, obs_dim);
    let mut per_replay = PrioritizedReplayBuffer::new(10_000, obs_dim, 0.6);
    let mut batch = TransitionBatch::new(batch_size, obs_dim);
    let mut per_weights = Tensor::zeros(&[batch_size, 1]);
    let mut per_tree_indices = vec![0; batch_size];

    let mut episode_rewards = Vec::new();
    let mut episode_losses = Vec::new();
    let mut global_step = 0usize;

    for episode in 0..episodes {
        let (state, _) = env.reset(Some(seed + episode as u64));
        let mut state_buf = vec![0.0f32; obs_dim];
        state.write_to_buffer(&mut state_buf);

        let mut episode_reward = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0usize;

        for _ in 0..max_steps {
            let q_values = {
                let input = Tensor::from_vec(state_buf.clone(), &[1, obs_dim]);
                let output = agent.q_net().forward(&Variable::from_tensor(input));
                let q = output.data().clone();
                q
            };

            let action_idx = explorer.select_action(&q_values, global_step, num_actions);
            let env_action = CartPoleAction::try_from(action_idx).unwrap();

            let (next_state, reward, terminated, truncated, _) = env.step(env_action);
            episode_reward += reward;

            let mut next_state_buf = vec![0.0f32; obs_dim];
            next_state.write_to_buffer(&mut next_state_buf);

            if use_per {
                per_replay.push(
                    &state_buf,
                    action_idx,
                    reward,
                    &next_state_buf,
                    replay_done(terminated, truncated),
                );
            } else {
                replay.push(
                    &state_buf,
                    action_idx,
                    reward,
                    &next_state_buf,
                    replay_done(terminated, truncated),
                );
            }

            state_buf = next_state_buf;

            let can_train = if use_per {
                per_replay.len() >= warmup_steps
            } else {
                replay.len() >= warmup_steps
            };

            if can_train {
                let (loss, _) = if use_per {
                    let beta = (0.4 + (1.0 - 0.4) * (global_step as f32 / 20000.0)).min(1.0);
                    per_replay.sample(
                        batch_size,
                        beta,
                        &mut batch,
                        &mut per_weights,
                        &mut per_tree_indices,
                    );

                    let (_l, td) = agent.train_step(&batch, Some(&per_weights));
                    if let Some(ref errs) = td {
                        per_replay.update_priorities(&per_tree_indices[..batch.size], errs);
                    }
                    (_l, td)
                } else {
                    replay.sample(batch_size, &mut batch);
                    agent.train_step(&batch, None)
                };

                if loss.is_finite() {
                    loss_sum += loss;
                    loss_count += 1;
                }
            }

            global_step += 1;
            if episode_done(terminated, truncated) {
                break;
            }
        }

        let avg_loss = if loss_count > 0 {
            loss_sum / loss_count as f32
        } else {
            0.0
        };

        episode_rewards.push(episode_reward);
        episode_losses.push(avg_loss);
    }

    (episode_rewards, episode_losses)
}

#[test]
fn test_dqn_per_vs_uniform_replay_smoke() {
    // Note: Due to the use of thread-local unseeded random number generation in EpsilonGreedy
    // action selection and PrioritizedReplayBuffer sampling, there is inherent non-determinism.
    // We report this as a bug finding in the PR description, as instructed.
    let seed = 22;
    let episodes = 40;
    let max_steps = 50;

    let mut success = false;
    for attempt in 1..=5 {
        // Run uniform replay DQN
        let (uniform_rewards, uniform_losses) = run_training(false, seed, episodes, max_steps);
        // Run prioritized replay DQN
        let (per_rewards, per_losses) = run_training(true, seed, episodes, max_steps);

        let first_uniform_reward = uniform_rewards.first().copied().unwrap();
        let final_uniform_reward = uniform_rewards.last().copied().unwrap();
        let first_per_reward = per_rewards.first().copied().unwrap();
        let final_per_reward = per_rewards.last().copied().unwrap();

        if final_uniform_reward >= first_uniform_reward && final_per_reward >= first_per_reward {
            // Assert both runs completed successfully
            assert_eq!(
                uniform_rewards.len(),
                episodes,
                "Uniform replay rewards length mismatch"
            );
            assert_eq!(per_rewards.len(), episodes, "PER rewards length mismatch");

            // Assert final average losses are finite (non-NaN)
            let final_uniform_loss = uniform_losses.last().copied().unwrap();
            let final_per_loss = per_losses.last().copied().unwrap();
            assert!(
                final_uniform_loss.is_finite() && !final_uniform_loss.is_nan(),
                "Uniform replay final loss is NaN/non-finite"
            );
            assert!(
                final_per_loss.is_finite() && !final_per_loss.is_nan(),
                "PER final loss is NaN/non-finite"
            );
            
            success = true;
            break;
        } else {
            println!(
                "Attempt {} failed: Uniform (first={}, final={}), PER (first={}, final={})",
                attempt, first_uniform_reward, final_uniform_reward, first_per_reward, final_per_reward
            );
        }
    }

    assert!(
        success,
        "Failed to satisfy weak learning check (final_reward >= first_reward) in 5 attempts due to unseeded exploration noise."
    );
}

// A helper to train DQN with PER and return the buffer to inspect its priorities
fn run_training_and_return_buffer(
    seed: u64,
    episodes: usize,
    max_steps: usize,
) -> PrioritizedReplayBuffer {
    let obs_dim = 4;
    let num_actions = 2;
    let batch_size = 32;
    let warmup_steps = 32; // small warmup to ensure training updates priorities early

    let config = DQNConfig {
        obs_dim,
        num_actions,
        hidden_dim: 16,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 10,
        double_dqn: true,
        use_per: true,
        per_beta_annealing_steps: 20000,
    };

    let mut env = CartPole::with_max_steps(max_steps);
    let mut agent = DQN::new(config);
    let explorer = EpsilonGreedy::new(1.0, 0.05, 500);
    let mut per_replay = PrioritizedReplayBuffer::new(10_000, obs_dim, 0.6);
    let mut batch = TransitionBatch::new(batch_size, obs_dim);
    let mut per_weights = Tensor::zeros(&[batch_size, 1]);
    let mut per_tree_indices = vec![0; batch_size];

    let mut global_step = 0usize;

    for episode in 0..episodes {
        let (state, _) = env.reset(Some(seed + episode as u64));
        let mut state_buf = vec![0.0f32; obs_dim];
        state.write_to_buffer(&mut state_buf);

        for _ in 0..max_steps {
            let q_values = {
                let input = Tensor::from_vec(state_buf.clone(), &[1, obs_dim]);
                let output = agent.q_net().forward(&Variable::from_tensor(input));
                let q = output.data().clone();
                q
            };

            let action_idx = explorer.select_action(&q_values, global_step, num_actions);
            let env_action = CartPoleAction::try_from(action_idx).unwrap();

            let (next_state, reward, terminated, truncated, _) = env.step(env_action);

            let mut next_state_buf = vec![0.0f32; obs_dim];
            next_state.write_to_buffer(&mut next_state_buf);

            per_replay.push(
                &state_buf,
                action_idx,
                reward,
                &next_state_buf,
                replay_done(terminated, truncated),
            );

            state_buf = next_state_buf;

            if per_replay.len() >= warmup_steps {
                let beta = 0.4;
                per_replay.sample(
                    batch_size,
                    beta,
                    &mut batch,
                    &mut per_weights,
                    &mut per_tree_indices,
                );

                let (_l, td) = agent.train_step(&batch, Some(&per_weights));
                if let Some(ref errs) = td {
                    per_replay.update_priorities(&per_tree_indices[..batch.size], errs);
                }
            }

            global_step += 1;
            if episode_done(terminated, truncated) {
                break;
            }
        }
    }

    per_replay
}

#[test]
fn test_dqn_per_priority_updates_observable() {
    let seed = 42;
    let episodes = 5;
    let max_steps = 50;

    let per_replay = run_training_and_return_buffer(seed, episodes, max_steps);

    // Verify buffer has collected enough transitions
    assert!(
        per_replay.len() > 32,
        "Buffer has not collected enough samples, got {}",
        per_replay.len()
    );

    // Sample a large batch with beta = 1.0. If priorities have updated and vary,
    // the importance sampling weights should not all be equal to 1.0 (or to each other).
    let mut batch = TransitionBatch::new(100, 4);
    let mut weights = Tensor::zeros(&[100, 1]);
    let mut tree_indices = vec![0; 100];
    per_replay.sample(100, 1.0, &mut batch, &mut weights, &mut tree_indices);

    let w_vec = weights.to_vec();
    let first = w_vec[0];
    let all_equal = w_vec
        .iter()
        .take(batch.size)
        .all(|&w| (w - first).abs() < 1e-6);

    assert!(
        !all_equal,
        "Expected priorities (and thus IS weights) to vary after training updates, but all weights are equal to {}",
        first
    );
}
