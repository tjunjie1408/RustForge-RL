//! Fresh terminal continuous-control episodes for a reproducible TD3 learning check.
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustforge_rl::{
    agent::{gpu_td3::GpuTd3, td3::TD3Config},
    buffer::{ContinuousReplayBuffer, ContinuousTransitionBatch},
    env::{Environment, Space},
};
use rustforge_tensor::gpu::GpuContext;
struct Target;
impl Environment for Target {
    type Obs = [f32; 2];
    type Act = f32;
    type Info = ();
    fn reset(&mut self, _: Option<u64>) -> (Self::Obs, ()) {
        ([1., 0.], ())
    }
    fn step(&mut self, action: f32) -> (Self::Obs, f32, bool, bool, ()) {
        assert!((-1. ..=1.).contains(&action));
        ([1., 0.], -(action - 0.5).powi(2), true, false, ())
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.; 2], vec![1.; 2])
    }
    fn action_space(&self) -> Space {
        Space::continuous(vec![-1.], vec![1.])
    }
}
pub struct LearningReport {
    pub initial_cost: f32,
    pub final_cost: f32,
    pub action: f32,
    pub critic_updates: usize,
    pub actor_updates: usize,
}
pub fn learn(context: &GpuContext) -> Result<LearningReport, Box<dyn std::error::Error>> {
    let mut config = TD3Config::new(2, 1, vec![-1.], vec![1.]);
    config.hidden_dim = 32;
    config.actor_lr = 0.001;
    config.critic_lr = 0.005;
    config.tau = 0.05;
    let mut agent = GpuTd3::new_seeded(context, config, 42)?;
    let initial = agent.select_action(&[1., 0.], 0.)?[0];
    let mut replay = ContinuousReplayBuffer::new(2048, 2, 1);
    let mut env = Target;
    let mut exploration = StdRng::seed_from_u64(7);
    let mut sampling = StdRng::seed_from_u64(8);
    let mut smoothing = StdRng::seed_from_u64(9);
    let mut batch = ContinuousTransitionBatch::new(64, 2, 1);
    for episode in 0..600 {
        let (state, _) = env.reset(Some(episode));
        let action = if episode < 64 {
            exploration.gen_range(-1. ..=1.)
        } else {
            agent.select_action_with_rng(&state, 0.3, &mut exploration)?[0]
        };
        let (next, reward, terminated, _, _) = env.step(action);
        replay.push(&state, &[action], reward, &next, terminated);
        if episode >= 64 {
            replay.sample_with_rng(64, &mut batch, &mut sampling);
            let (critic, actor) = agent.train_step_with_rng(&batch, &mut smoothing)?;
            assert!(critic.is_finite() && actor.map_or(true, |v| v.is_finite()));
        }
    }
    let action = agent.select_action(&[1., 0.], 0.)?[0];
    Ok(LearningReport {
        initial_cost: (initial - 0.5).powi(2),
        final_cost: (action - 0.5).powi(2),
        action,
        critic_updates: agent.updates(),
        actor_updates: agent.actor_updates(),
    })
}
