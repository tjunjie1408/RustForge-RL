//! DQN CartPole end-to-end convergence test.
//!
//! This test verifies that the DQN agent can learn to solve CartPole-v1 within
//! a reasonable number of episodes. It is marked `#[ignore]` because convergence
//! tests are slow (~30–60 seconds) and should only be run locally, not in CI.
//!
//! # Running
//!
//! ```text
//! cargo test -p rustforge-rl --test dqn_convergence -- --ignored
//! ```
//!
//! # Convergence Criteria
//!
//! - **Seed**: `12345` (deterministic PRNG for environment + weight init)
//! - **Max episodes**: 150
//! - **Target**: Rolling average reward over the last 10 episodes > 100
//! - **Threshold relaxation**: Different CPUs, SIMD settings, or BLAS backends
//!   may cause minor convergence speed variation. Adjust `MAX_EPISODES` or
//!   `TARGET_REWARD` if the test consistently fails on custom hardware.
//!
//! **Margin Note:** Currently converges at episode 93/150 with avg 108.6/100
//! on standard targets. Future refactors may shift these margins; if it fails
//! by a small margin, investigate before relaxing thresholds.

use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_rl::agent::{DQNConfig, EpsilonGreedy, DQN};
use rustforge_rl::buffer::{ReplayBuffer, TransitionBatch};
use rustforge_rl::env::{CartPole, CartPoleAction, Environment};
use rustforge_rl::metrics::{AgentLogger, CsvLogger, EpisodeMetrics};
use rustforge_rl::training::{episode_done, replay_done};
use rustforge_tensor::Tensor;

const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 2;
const SEED: u64 = 12345;
const MAX_EPISODES: usize = 150;
const TARGET_REWARD: f32 = 100.0;
const ROLLING_WINDOW: usize = 10;
const MAX_STEPS_PER_EPISODE: usize = 500;
const BATCH_SIZE: usize = 64;
const WARMUP_STEPS: usize = 256;
const BUFFER_SIZE: usize = 10_000;

fn q_values_for_state(agent: &DQN, state: &[f32; OBS_DIM]) -> Tensor {
    let input = Tensor::from_vec(state.to_vec(), &[1, OBS_DIM]);
    let output = agent.q_net().forward(&Variable::from_tensor(input));
    let data = output.data().clone();
    data
}

#[test]
#[ignore] // Run locally only: cargo test --test dqn_convergence -- --ignored
fn dqn_converges_on_cartpole() {
    let mut env = CartPole::with_max_steps(MAX_STEPS_PER_EPISODE);
    let mut agent = DQN::new(DQNConfig {
        obs_dim: OBS_DIM,
        num_actions: NUM_ACTIONS,
        hidden_dim: 128,
        lr: 5e-4,
        gamma: 0.99,
        target_update_freq: 200,
        double_dqn: true,
        use_per: false,
        per_beta_annealing_steps: 20000,
    });
    let explorer = EpsilonGreedy::new(1.0, 0.02, 5_000);
    let mut replay = ReplayBuffer::new(BUFFER_SIZE, OBS_DIM);
    let mut batch = TransitionBatch::new(BATCH_SIZE, OBS_DIM);

    let logger =
        CsvLogger::new("target/dqn_convergence_metrics.csv").expect("Failed to create CSV logger");

    let mut global_step = 0usize;
    let mut rewards: Vec<f32> = Vec::with_capacity(MAX_EPISODES);
    let mut converged = false;

    for episode in 0..MAX_EPISODES {
        let (mut state, _) = env.reset(Some(SEED + episode as u64));
        let mut episode_reward = 0.0f32;
        let mut episode_loss = 0.0f32;
        let mut train_steps = 0usize;

        for _ in 0..MAX_STEPS_PER_EPISODE {
            let q_values = q_values_for_state(&agent, &state);
            let action_idx = explorer.select_action(&q_values, global_step, NUM_ACTIONS);
            let env_action = CartPoleAction::try_from(action_idx)
                .expect("DQN produced invalid CartPole action index");

            let (next_state, reward, terminated, truncated, _) = env.step(env_action);
            episode_reward += reward;

            replay.push(
                &state,
                action_idx,
                reward,
                &next_state,
                replay_done(terminated, truncated),
            );

            if replay.len() >= WARMUP_STEPS {
                replay.sample(BATCH_SIZE, &mut batch);
                let (loss, _td_errors) = agent.train_step(&batch, None);
                episode_loss += loss;
                train_steps += 1;
            }

            global_step += 1;
            state = next_state;

            if episode_done(terminated, truncated) {
                break;
            }
        }

        let avg_loss = if train_steps > 0 {
            episode_loss / train_steps as f32
        } else {
            f32::NAN
        };

        logger.log(&EpisodeMetrics {
            episode,
            reward: episode_reward,
            avg_loss,
            epsilon: explorer.epsilon(global_step),
            global_step,
        });

        rewards.push(episode_reward);

        // Check rolling average convergence
        if rewards.len() >= ROLLING_WINDOW {
            let window_start = rewards.len() - ROLLING_WINDOW;
            let rolling_avg: f32 =
                rewards[window_start..].iter().sum::<f32>() / ROLLING_WINDOW as f32;

            eprintln!(
                "episode={:03} reward={:6.1} rolling_avg={:6.2} epsilon={:.3} steps={}",
                episode,
                episode_reward,
                rolling_avg,
                explorer.epsilon(global_step),
                global_step,
            );

            if rolling_avg >= TARGET_REWARD {
                eprintln!(
                    "✅ Converged at episode {} with rolling avg {:.2} (target: {:.0})",
                    episode, rolling_avg, TARGET_REWARD
                );
                converged = true;
                break;
            }
        } else {
            eprintln!(
                "episode={:03} reward={:6.1} epsilon={:.3} steps={}",
                episode,
                episode_reward,
                explorer.epsilon(global_step),
                global_step,
            );
        }
    }

    assert!(
        converged,
        "DQN did not converge on CartPole within {} episodes. \
         Last 10 episode rewards: {:?}",
        MAX_EPISODES,
        &rewards[rewards.len().saturating_sub(ROLLING_WINDOW)..],
    );
}

#[test]
#[ignore] // Run locally only: cargo test --test dqn_convergence -- --ignored
fn dqn_per_converges_on_cartpole() {
    use rustforge_rl::buffer::PrioritizedReplayBuffer;

    let mut env = CartPole::with_max_steps(MAX_STEPS_PER_EPISODE);
    let mut agent = DQN::new(DQNConfig {
        obs_dim: OBS_DIM,
        num_actions: NUM_ACTIONS,
        hidden_dim: 128,
        lr: 5e-4,
        gamma: 0.99,
        target_update_freq: 200,
        double_dqn: true,
        use_per: true,
        per_beta_annealing_steps: 20000,
    });
    let explorer = EpsilonGreedy::new(1.0, 0.02, 5_000);
    let mut replay = PrioritizedReplayBuffer::new(BUFFER_SIZE, OBS_DIM, 0.6);
    let mut batch = TransitionBatch::new(BATCH_SIZE, OBS_DIM);
    let mut per_weights = Tensor::zeros(&[BATCH_SIZE, 1]);
    let mut per_tree_indices = vec![0; BATCH_SIZE];

    let logger = CsvLogger::new("target/dqn_per_convergence_metrics.csv")
        .expect("Failed to create CSV logger");

    let mut global_step = 0usize;
    let mut rewards: Vec<f32> = Vec::with_capacity(MAX_EPISODES);
    let mut converged = false;

    for episode in 0..MAX_EPISODES {
        let (mut state, _) = env.reset(Some(SEED + episode as u64));
        let mut episode_reward = 0.0f32;
        let mut episode_loss = 0.0f32;
        let mut train_steps = 0usize;

        for _ in 0..MAX_STEPS_PER_EPISODE {
            let q_values = q_values_for_state(&agent, &state);
            let action_idx = explorer.select_action(&q_values, global_step, NUM_ACTIONS);
            let env_action = CartPoleAction::try_from(action_idx)
                .expect("DQN produced invalid CartPole action index");

            let (next_state, reward, terminated, truncated, _) = env.step(env_action);
            episode_reward += reward;

            replay.push(
                &state,
                action_idx,
                reward,
                &next_state,
                replay_done(terminated, truncated),
            );

            if replay.len() >= WARMUP_STEPS {
                let beta_steps = agent.config().per_beta_annealing_steps as f32;
                let beta = (0.4 + (1.0 - 0.4) * (global_step as f32 / beta_steps)).min(1.0);
                replay.sample(
                    BATCH_SIZE,
                    beta,
                    &mut batch,
                    &mut per_weights,
                    &mut per_tree_indices,
                );
                let (loss, td_errors) = agent.train_step(&batch, Some(&per_weights));
                if let Some(errs) = td_errors {
                    replay.update_priorities(&per_tree_indices[..batch.size], &errs);
                }
                episode_loss += loss;
                train_steps += 1;
            }

            global_step += 1;
            state = next_state;

            if episode_done(terminated, truncated) {
                break;
            }
        }

        let avg_loss = if train_steps > 0 {
            episode_loss / train_steps as f32
        } else {
            f32::NAN
        };

        logger.log(&EpisodeMetrics {
            episode,
            reward: episode_reward,
            avg_loss,
            epsilon: explorer.epsilon(global_step),
            global_step,
        });

        rewards.push(episode_reward);

        // Check rolling average convergence
        if rewards.len() >= ROLLING_WINDOW {
            let window_start = rewards.len() - ROLLING_WINDOW;
            let rolling_avg: f32 =
                rewards[window_start..].iter().sum::<f32>() / ROLLING_WINDOW as f32;

            eprintln!(
                "episode={:03} reward={:6.1} rolling_avg={:6.2} epsilon={:.3} steps={} (PER)",
                episode,
                episode_reward,
                rolling_avg,
                explorer.epsilon(global_step),
                global_step,
            );

            if rolling_avg >= TARGET_REWARD {
                eprintln!(
                    "✅ Converged with PER at episode {} with rolling avg {:.2} (target: {:.0})",
                    episode, rolling_avg, TARGET_REWARD
                );
                converged = true;
                break;
            }
        } else {
            eprintln!(
                "episode={:03} reward={:6.1} epsilon={:.3} steps={} (PER)",
                episode,
                episode_reward,
                explorer.epsilon(global_step),
                global_step,
            );
        }
    }

    assert!(
        converged,
        "DQN with PER did not converge on CartPole within {} episodes. \
         Last 10 episode rewards: {:?}",
        MAX_EPISODES,
        &rewards[rewards.len().saturating_sub(ROLLING_WINDOW)..],
    );
}
