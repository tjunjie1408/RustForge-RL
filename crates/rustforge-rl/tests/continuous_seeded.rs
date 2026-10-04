use rand::{rngs::StdRng, SeedableRng};
use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_rl::{
    agent::{PPOConfig, PPOContinuous, PPOContinuousConfig},
    buffer::ContinuousRolloutBatch,
};
use rustforge_tensor::Tensor;

fn config() -> PPOContinuousConfig {
    PPOContinuousConfig {
        base: PPOConfig {
            obs_dim: 2,
            hidden_dim: 8,
            ppo_epochs: 2,
            mini_batch_size: 3,
            ..PPOConfig::default()
        },
        act_dim: 2,
        action_low: vec![-1., -2.],
        action_high: vec![1., 3.],
    }
}
fn parameters(agent: &PPOContinuous) -> Vec<Vec<f32>> {
    agent
        .actor
        .parameters()
        .into_iter()
        .chain(agent.critic().parameters())
        .map(|p| p.data().to_vec())
        .collect()
}
#[test]
fn seeded_continuous_initialization_and_sampling_repeat_with_wrapping_seeds() {
    let a = PPOContinuous::new_seeded(config(), u64::MAX);
    let b = PPOContinuous::new_seeded(config(), u64::MAX);
    let c = PPOContinuous::new_seeded(config(), 42);
    assert_eq!(parameters(&a), parameters(&b));
    assert_ne!(parameters(&a), parameters(&c));
    let mut r1 = StdRng::seed_from_u64(7);
    let mut r2 = r1.clone();
    for _ in 0..10 {
        let first = a.select_action_with_rng(&[0.2, 0.7], &mut r1);
        assert_eq!(first, b.select_action_with_rng(&[0.2, 0.7], &mut r2));
        assert!((-1. ..=1.).contains(&first.0[0]));
        assert!((-2. ..=3.).contains(&first.0[1]));
        assert!(first.1.is_finite() && first.2.is_finite());
    }
}
#[test]
fn seeded_continuous_shuffle_repeats_partial_minibatches_and_ignores_unused_tail() {
    let mut a = PPOContinuous::new_seeded(config(), 42);
    let mut b = PPOContinuous::new_seeded(config(), 42);
    let mut batch = ContinuousRolloutBatch::new(6, 2, 2);
    batch.size = 5;
    let states = [0.2, 0.7].repeat(5);
    let mut sampling = StdRng::seed_from_u64(17);
    let actions: Vec<f32> = (0..5)
        .flat_map(|_| a.select_action_with_rng(&[0.2, 0.7], &mut sampling).0)
        .collect();
    let mut lp = a
        .actor
        .log_prob_from_action(
            &Variable::from_tensor(Tensor::from_vec(states.clone(), &[5, 2])),
            &Variable::from_tensor(Tensor::from_vec(actions.clone(), &[5, 2])),
        )
        .data()
        .to_vec();
    let mut states = states;
    states.extend([f32::NAN; 2]);
    let mut actions = actions;
    actions.extend([f32::NAN; 2]);
    lp.push(f32::NAN);
    batch.states = Tensor::from_vec(states, &[6, 2]);
    batch.actions = Tensor::from_vec(actions, &[6, 2]);
    batch.old_log_probs = Tensor::from_vec(lp, &[6, 1]);
    batch.returns = Tensor::from_vec(vec![1., -0.2, 0.7, 0.1, -0.4, f32::NAN], &[6, 1]);
    batch.advantages = Tensor::from_vec(vec![1., -1., 0.3, -0.7, 0.2, f32::NAN], &[6, 1]);
    let mut r1 = StdRng::seed_from_u64(9);
    let mut r2 = r1.clone();
    for _ in 0..2 {
        let metrics = a.train_on_batch_with_rng(&batch, &mut r1);
        assert!(metrics.0.is_finite() && metrics.1.is_finite());
        assert_eq!(metrics, b.train_on_batch_with_rng(&batch, &mut r2));
        assert_eq!(parameters(&a), parameters(&b));
    }
}
