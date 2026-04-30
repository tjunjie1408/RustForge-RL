//! Minimal DQN + CartPole training loop.
//!
//! Run with:
//!
//! ```text
//! cargo run -p rustforge-rl --example dqn_cartpole
//! ```
//!
//! This example intentionally keeps the loop explicit. It shows how the strong
//! `CartPoleAction` enum at the environment boundary maps to the `usize` action
//! indices used by the Q-network and replay buffer.

use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_rl::agent::{DQNConfig, EpsilonGreedy, DQN};
use rustforge_rl::buffer::{ReplayBuffer, TransitionBatch};
use rustforge_rl::env::{CartPole, CartPoleAction, Environment};
use rustforge_rl::training::{episode_done, replay_done};
use rustforge_tensor::Tensor;

const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 2;

fn q_values_for_state(agent: &DQN, state: &[f32; OBS_DIM]) -> Tensor {
    let input = Tensor::from_vec(state.to_vec(), &[1, OBS_DIM]);
    let output = agent.q_net().forward(&Variable::from_tensor(input));
    let q_values = output.data().clone();
    q_values
}

fn main() {
    let episodes = 50usize;
    let max_steps_per_episode = 500usize;
    let batch_size = 32usize;
    let warmup_steps = 128usize;

    // DQN::train_step already performs the hard target-network sync according
    // to this configuration. The example keeps the frequency visible here and
    // does not call update_target() again, avoiding duplicate hard updates.
    let target_update_freq = 100usize;

    let mut env = CartPole::with_max_steps(max_steps_per_episode);
    let mut agent = DQN::new(DQNConfig {
        obs_dim: OBS_DIM,
        num_actions: NUM_ACTIONS,
        hidden_dim: 64,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq,
        double_dqn: true,
    });
    let explorer = EpsilonGreedy::new(1.0, 0.05, 2_000);
    let mut replay = ReplayBuffer::new(10_000, OBS_DIM);
    let mut batch = TransitionBatch::new(batch_size, OBS_DIM);

    let mut global_step = 0usize;
    let mut rewards_window = Vec::with_capacity(100);

    for episode in 0..episodes {
        let (mut state, _) = env.reset(Some(2026 + episode as u64));
        let mut episode_reward = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0usize;

        for _ in 0..max_steps_per_episode {
            let q_values = q_values_for_state(&agent, &state);
            let action_idx = explorer.select_action(&q_values, global_step, NUM_ACTIONS);

            let env_action = match CartPoleAction::try_from(action_idx) {
                Ok(action) => action,
                Err(_) => unreachable!("DQN produced invalid CartPole action index"),
            };

            let (next_state, reward, terminated, truncated, _) = env.step(env_action);
            episode_reward += reward;

            // Reset the environment on either flag, but only true terminal
            // states should disable TD bootstrapping in replay.
            replay.push(
                &state,
                action_idx,
                reward,
                &next_state,
                replay_done(terminated, truncated),
            );

            if replay.len() >= warmup_steps {
                replay.sample(batch_size, &mut batch);
                let loss = agent.train_step(&batch);
                if loss.is_finite() {
                    loss_sum += loss;
                    loss_count += 1;
                }
            }

            global_step += 1;
            state = next_state;

            if episode_done(terminated, truncated) {
                break;
            }
        }

        rewards_window.push(episode_reward);
        if rewards_window.len() > 100 {
            rewards_window.remove(0);
        }

        let moving_avg = rewards_window.iter().sum::<f32>() / rewards_window.len() as f32;
        let avg_loss = if loss_count == 0 {
            0.0
        } else {
            loss_sum / loss_count as f32
        };
        let epsilon = explorer.epsilon(global_step);

        println!(
            "episode={episode:03} reward={episode_reward:6.1} avg_reward={moving_avg:6.2} loss={avg_loss:9.5} epsilon={epsilon:.3} steps={global_step}"
        );
    }

    println!(
        "Finished. For a longer local convergence check, increase `episodes` and watch the moving average approach CartPole's solved threshold."
    );
}
