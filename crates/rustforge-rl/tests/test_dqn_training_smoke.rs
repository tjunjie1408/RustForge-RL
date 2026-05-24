use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_rl::agent::{DQNConfig, EpsilonGreedy, DQN};
use rustforge_rl::buffer::{ReplayBuffer, TransitionBatch};
use rustforge_rl::env::{CartPole, CartPoleAction, Environment};
use rustforge_rl::training::{episode_done, replay_done};
use rustforge_tensor::Tensor;

fn q_values_for_state(agent: &DQN, state: &[f32; 4]) -> Tensor {
    let input = Tensor::from_vec(state.to_vec(), &[1, 4]);
    let output = agent.q_net().forward(&Variable::from_tensor(input));
    let q_values = output.data().clone();
    q_values
}

#[test]
fn cartpole_action_index_mapping_is_explicit() {
    assert_eq!(CartPoleAction::try_from(0), Ok(CartPoleAction::Left));
    assert_eq!(CartPoleAction::try_from(1), Ok(CartPoleAction::Right));
    assert!(CartPoleAction::try_from(2).is_err());
    assert_eq!(usize::from(CartPoleAction::Left), 0);
    assert_eq!(usize::from(CartPoleAction::Right), 1);

    let dqn = DQN::new(DQNConfig::default());
    let action = dqn.select_greedy_action(&[0.0, 0.0, 0.0, 0.0]);
    assert!(action < 2);

    let explorer = EpsilonGreedy::new(0.0, 0.0, 1);
    let q_values = Tensor::from_vec(vec![0.1, 0.2], &[1, 2]);
    let action = explorer.select_action(&q_values, 0, 2);
    assert!(action < 2);
}

#[test]
fn replay_done_distinguishes_terminated_from_truncated() {
    assert!(episode_done(true, false));
    assert!(episode_done(false, true));
    assert!(replay_done(true, false));
    assert!(!replay_done(false, true));

    let mut terminated_buffer = ReplayBuffer::new(1, 4);
    terminated_buffer.push(
        &[0.0, 0.0, 0.0, 0.0],
        0,
        1.0,
        &[0.1, 0.0, 0.0, 0.0],
        replay_done(true, false),
    );
    let mut terminated_batch = TransitionBatch::new(1, 4);
    terminated_buffer.sample(1, &mut terminated_batch);
    assert_eq!(terminated_batch.dones.to_vec()[0], 1.0);

    let mut truncated_buffer = ReplayBuffer::new(1, 4);
    truncated_buffer.push(
        &[0.0, 0.0, 0.0, 0.0],
        0,
        1.0,
        &[0.1, 0.0, 0.0, 0.0],
        replay_done(false, true),
    );
    let mut truncated_batch = TransitionBatch::new(1, 4);
    truncated_buffer.sample(1, &mut truncated_batch);
    assert_eq!(truncated_batch.dones.to_vec()[0], 0.0);
}

#[test]
fn dqn_cartpole_training_smoke_runs_and_produces_finite_loss() {
    let mut env = CartPole::with_max_steps(25);
    let mut agent = DQN::new(DQNConfig {
        obs_dim: 4,
        num_actions: 2,
        hidden_dim: 16,
        lr: 1e-3,
        gamma: 0.99,
        target_update_freq: 10,
        double_dqn: false,
    });
    let explorer = EpsilonGreedy::new(0.2, 0.05, 100);
    let mut buffer = ReplayBuffer::new(256, 4);
    let mut batch = TransitionBatch::new(8, 4);

    let warmup_steps = 8;
    let mut global_step = 0usize;
    let mut train_steps = 0usize;
    let mut last_loss = None;
    let mut total_reward = 0.0f32;

    for episode in 0..4 {
        let (mut state, _) = env.reset(Some(42 + episode));

        for _ in 0..25 {
            let q_values = q_values_for_state(&agent, &state);
            let action_idx = explorer.select_action(&q_values, global_step, 2);
            assert!(action_idx < 2);
            let action = CartPoleAction::try_from(action_idx).expect("valid CartPole action");

            let (next_state, reward, terminated, truncated, _) = env.step(action);
            total_reward += reward;

            buffer.push(
                &state,
                action_idx,
                reward,
                &next_state,
                replay_done(terminated, truncated),
            );

            if buffer.len() >= warmup_steps {
                buffer.sample(8, &mut batch);
                let (loss, _) = agent.train_step(&batch, None);
                assert!(loss.is_finite());
                last_loss = Some(loss);
                train_steps += 1;
            }

            global_step += 1;
            state = next_state;

            if episode_done(terminated, truncated) {
                break;
            }
        }
    }

    assert!(train_steps > 0);
    assert!(last_loss.is_some());
    assert!(total_reward.is_finite());
}
