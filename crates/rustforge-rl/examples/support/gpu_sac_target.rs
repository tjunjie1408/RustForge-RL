//! Fresh terminal continuous-control episodes for a reproducible SAC learning check.
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustforge_rl::{
    agent::{gpu_sac::GpuSac, sac::SACConfig},
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
    let mut config = SACConfig::new(2, 1, vec![-1.], vec![1.]);
    config.hidden_dim = 16;
    config.actor_lr = 0.003;
    config.critic_lr = 0.005;
    config.tau = 0.05;
    config.init_alpha = 0.02;
    config.alpha_lr = 0.003;
    let mut agent = GpuSac::new_seeded(context, config, 42)?;
    let initial = agent.deterministic_action(&[1., 0.])?[0];
    let mut replay = ContinuousReplayBuffer::new(2048, 2, 1);
    let mut env = Target;
    let mut exploration = StdRng::seed_from_u64(7);
    let mut sampling = StdRng::seed_from_u64(8);
    let mut target_noise = StdRng::seed_from_u64(9);
    let mut actor_noise = StdRng::seed_from_u64(10);
    let mut batch = ContinuousTransitionBatch::new(64, 2, 1);
    for episode in 0..600 {
        let (state, _) = env.reset(Some(episode));
        let action = if episode < 64 {
            exploration.gen_range(-1. ..=1.)
        } else {
            agent.select_action_with_rng(&state, &mut exploration)?[0]
        };
        let (next, reward, terminated, _, _) = env.step(action);
        replay.push(&state, &[action], reward, &next, terminated);
        if episode >= 64 {
            replay.sample_with_rng(64, &mut batch, &mut sampling);
            let (critic, actor, temperature, alpha) =
                agent.train_step_with_rngs(&batch, &mut target_noise, &mut actor_noise)?;
            assert!(
                [critic, actor, temperature, alpha]
                    .iter()
                    .all(|v| v.is_finite())
                    && alpha > 0.
            );
        }
    }
    let action = agent.deterministic_action(&[1., 0.])?[0];
    Ok(LearningReport {
        initial_cost: (initial - 0.5).powi(2),
        final_cost: (action - 0.5).powi(2),
        action,
        critic_updates: agent.updates(),
        actor_updates: agent.updates(),
    })
}
