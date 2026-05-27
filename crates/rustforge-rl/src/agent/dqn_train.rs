//! Generic DQN training loop.

use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_tensor::Tensor;

use crate::agent::{DQNConfig, EpsilonGreedy, DQN};
use crate::buffer::{PrioritizedReplayBuffer, ReplayBuffer, TransitionBatch};
use crate::env::{Environment, IntoTensorBuffer};
use crate::metrics::{AgentLogger, CsvLogger, EpisodeMetrics};
use crate::training::{episode_done, replay_done};
use std::convert::TryFrom;
use std::fmt::Debug;

/// Generic DQN training loop for discrete action environments.
pub fn train_dqn<E>(
    mut env: E,
    config: DQNConfig,
    episodes: usize,
    max_steps_per_episode: usize,
    log_path: Option<&str>,
) -> DQN
where
    E: Environment,
    E::Act: TryFrom<usize>,
    <E::Act as TryFrom<usize>>::Error: Debug,
{
    let obs_dim = E::Obs::DIM;
    let num_actions = config.num_actions;
    let batch_size = 32usize;
    let warmup_steps = 128usize;

    let mut agent = DQN::new(config);
    let mut explorer = EpsilonGreedy::new(1.0, 0.05, 2_000);

    let use_per = agent.config().use_per;
    let mut replay = ReplayBuffer::new(10_000, obs_dim);
    let mut batch = TransitionBatch::new(batch_size, obs_dim);

    let mut per_replay = PrioritizedReplayBuffer::new(10_000, obs_dim, 0.6);
    let mut per_weights = Tensor::zeros(&[batch_size, 1]);
    let mut per_tree_indices = vec![0; batch_size];

    let mut logger =
        log_path.map(|path| CsvLogger::new(path).expect("Failed to create CSV logger"));

    let mut global_step = 0usize;
    let mut rewards_window = Vec::with_capacity(100);

    for episode in 0..episodes {
        let (state, _) = env.reset(Some(2026 + episode as u64));
        let mut state_buf = vec![0.0f32; obs_dim];
        state.write_to_buffer(&mut state_buf);

        let mut episode_reward = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0usize;

        for _ in 0..max_steps_per_episode {
            let q_values = {
                let input = Tensor::from_vec(state_buf.clone(), &[1, obs_dim]);
                let output = agent.q_net().forward(&Variable::from_tensor(input));
                let q = output.data().clone();
                q
            };

            let action_idx = explorer.select_action(&q_values, global_step, num_actions);

            let env_action = match E::Act::try_from(action_idx) {
                Ok(action) => action,
                Err(_) => unreachable!("DQN produced invalid action index"),
            };

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
                    let beta_steps = agent.config().per_beta_annealing_steps as f32;
                    let beta = (0.4 + (1.0 - 0.4) * (global_step as f32 / beta_steps)).min(1.0);
                    per_replay.sample(
                        batch_size,
                        beta,
                        &mut batch,
                        &mut per_weights,
                        &mut per_tree_indices,
                    );

                    let (l, td) = agent.train_step(&batch, Some(&per_weights));
                    if let Some(ref errs) = td {
                        per_replay.update_priorities(&per_tree_indices[..batch.size], errs);
                    }
                    (l, td)
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

        if rewards_window.len() == 100 {
            rewards_window.remove(0);
        }
        rewards_window.push(episode_reward);
        let rolling_avg = rewards_window.iter().sum::<f32>() / rewards_window.len() as f32;

        let epsilon = explorer.epsilon(global_step);
        let avg_loss = if loss_count > 0 {
            loss_sum / loss_count as f32
        } else {
            f32::NAN
        };

        println!(
            "Episode {:4} | Reward: {:6.1} | Rolling: {:6.1} | Epsilon: {:.3} | Loss: {:.4}",
            episode,
            episode_reward,
            rolling_avg,
            epsilon,
            if avg_loss.is_nan() { 0.0 } else { avg_loss }
        );

        if let Some(logger) = logger.as_mut() {
            logger.log(&EpisodeMetrics {
                episode,
                reward: episode_reward,
                avg_loss,
                epsilon,
                global_step,
            });
        }
    }

    if let Some(logger) = logger {
        logger.flush();
    }

    agent
}
