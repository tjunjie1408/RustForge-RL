#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustforge_autograd::no_grad;
use rustforge_nn::{gpu::GpuModule, Module};
use rustforge_rl::{
    agent::{
        gpu_td3::{GpuTd3, GpuTd3Error},
        td3::{TD3Config, TD3},
    },
    buffer::{ContinuousReplayBuffer, ContinuousTransitionBatch},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::sync::OnceLock;
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().expect("GPU TD3 tests require an adapter"))
}
fn config() -> TD3Config {
    let mut c = TD3Config::new(2, 2, vec![-2., 1.], vec![4., 5.]);
    c.hidden_dim = 8;
    c.actor_lr = 0.002;
    c.critic_lr = 0.003;
    c.tau = 0.2;
    c
}
fn batch() -> ContinuousTransitionBatch {
    let mut replay = ContinuousReplayBuffer::new(16, 2, 2);
    for (s, a, r, next, done) in [
        ([0.2, -0.3], [0.5, 2.], 0.7, [0.3, -0.2], false),
        ([0.7, 0.2], [-1., 4.], -0.3, [0.1, 0.4], true),
        ([-0.1, 0.8], [3., 1.5], 0.2, [0.2, 0.5], false),
    ] {
        replay.push(&s, &a, r, &next, done);
    }
    let mut b = ContinuousTransitionBatch::new(8, 2, 2);
    replay.sample_with_rng(8, &mut b, &mut StdRng::seed_from_u64(8));
    // The rows returned by the replay buffer are fixed for all compared updates.
    b.dones.data_mut()[[1, 0]] = 0.25;
    b
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        assert!(x.is_finite() && y.is_finite());
        approx::assert_abs_diff_eq!(x, y, epsilon = eps);
    }
}
fn values(agent: &GpuTd3) -> Vec<Vec<f32>> {
    [
        agent.actor(),
        agent.critic1(),
        agent.critic2(),
        agent.actor_target(),
        agent.critic1_target(),
        agent.critic2_target(),
    ]
    .into_iter()
    .flat_map(GpuModule::parameters)
    .map(|p| p.to_cpu().unwrap().to_vec())
    .collect()
}
fn compare(cpu: &TD3, gpu: &GpuTd3, eps: f32) {
    for (c, g) in [
        cpu.actor(),
        cpu.critic1(),
        cpu.critic2(),
        cpu.actor_target(),
        cpu.critic1_target(),
        cpu.critic2_target(),
    ]
    .into_iter()
    .zip([
        gpu.actor(),
        gpu.critic1(),
        gpu.critic2(),
        gpu.actor_target(),
        gpu.critic1_target(),
        gpu.critic2_target(),
    ]) {
        for (c, g) in c.parameters().iter().zip(g.parameters()) {
            close(&c.data().to_vec(), &g.to_cpu().unwrap().to_vec(), eps);
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_twin_critics_actor_delay_and_polyak_match_complete_cpu_td3() {
    let b = batch();
    for (delay, tau) in [(0, 0.2), (1, 0.), (1, 1.), (2, 0.2), (3, 0.2)] {
        let mut c = config();
        c.policy_delay = delay;
        c.tau = tau;
        let mut cpu = TD3::new_seeded(c.clone(), u64::MAX - 2);
        let mut gpu = GpuTd3::new_seeded(context(), c, u64::MAX - 2).unwrap();
        compare(&cpu, &gpu, 0.);
        let mut cr = StdRng::seed_from_u64(67);
        let mut gr = cr.clone();
        let actor_before = values(&gpu)[..6].to_vec();
        let target_before = values(&gpu)[18..].to_vec();
        for step in 1..=6 {
            let cm = cpu.train_step_with_rng(&b, &mut cr);
            let gm = gpu.train_step_with_rng(&b, &mut gr).unwrap();
            approx::assert_abs_diff_eq!(cm.0, gm.0, epsilon = 2e-4);
            assert_eq!(cm.1.is_some(), gm.1.is_some());
            if let (Some(c), Some(g)) = (cm.1, gm.1) {
                approx::assert_abs_diff_eq!(c, g, epsilon = 2e-4);
            }
            compare(&cpu, &gpu, 2e-4);
            assert_eq!(gpu.updates(), step);
            assert_eq!(gpu.actor_updates(), step.checked_div(delay).unwrap_or(0));
            assert_eq!(cpu.updates(), step);
            for target in [
                gpu.actor_target(),
                gpu.critic1_target(),
                gpu.critic2_target(),
            ] {
                assert!(target
                    .parameters()
                    .iter()
                    .all(|p| !p.requires_grad() && p.grad().is_none()));
            }
            assert!(gpu
                .critic1()
                .parameters()
                .iter()
                .all(|p| p.grad().is_none()));
        }
        assert_eq!(cr.gen::<u64>(), gr.gen::<u64>());
        if delay == 0 {
            assert_eq!(&values(&gpu)[..6], &actor_before);
        }
        if delay == 0 || tau == 0. {
            assert_eq!(&values(&gpu)[18..], &target_before);
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn exploration_and_target_noise_streams_match_cpu_and_zero_noise_draw_rules() {
    let cpu = TD3::new_seeded(config(), 42);
    let gpu = GpuTd3::new_seeded(context(), config(), 42).unwrap();
    for std in [0., 0.2, 100.] {
        let mut cr = StdRng::seed_from_u64(7);
        let mut gr = cr.clone();
        close(
            &cpu.select_action_with_rng(&[0.1, 0.5], std, &mut cr),
            &gpu.select_action_with_rng(&[0.1, 0.5], std, &mut gr)
                .unwrap(),
            2e-6,
        );
        let first = cr.gen::<u64>();
        assert_eq!(first, gr.gen::<u64>());
        if std == 0. {
            assert_eq!(first, StdRng::seed_from_u64(7).gen::<u64>());
        }
    }
    let mut c = config();
    c.target_noise_std = 0.;
    c.target_noise_clip = 0.;
    let mut cpu = TD3::new_seeded(c.clone(), 1);
    let mut gpu = GpuTd3::new_seeded(context(), c, 1).unwrap();
    let b = batch();
    let mut cr = StdRng::seed_from_u64(2);
    let mut gr = cr.clone();
    let mut expected = cr.clone();
    for _ in 0..b.size * 2 {
        let _: f32 = expected.gen_range(1e-7..1.);
        let _: f32 = expected.gen_range(0. ..std::f32::consts::TAU);
    }
    cpu.train_step_with_rng(&b, &mut cr);
    gpu.train_step_with_rng(&b, &mut gr).unwrap();
    let next = cr.gen::<u64>();
    assert_eq!(next, expected.gen::<u64>());
    assert_eq!(gr.gen::<u64>(), next);
    compare(&cpu, &gpu, 1e-4);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn active_rows_ignore_stale_capacity_and_reject_invalid_batches_before_rng_or_mutation() {
    let mut gpu = GpuTd3::new_seeded(context(), config(), 42).unwrap();
    let mut exact = GpuTd3::new_seeded(context(), config(), 42).unwrap();
    let mut b = batch();
    let active = b.valid_rows();
    for tensor in [
        &mut b.states,
        &mut b.next_states,
        &mut b.actions,
        &mut b.rewards,
        &mut b.dones,
    ] {
        let width = tensor.shape()[1];
        for row in b.size..tensor.shape()[0] {
            for col in 0..width {
                tensor.data_mut()[[row, col]] = f32::NAN;
            }
        }
    }
    let mut noise = Tensor::full(&[8, 2], f32::NAN);
    for row in 0..b.size {
        for col in 0..2 {
            noise.data_mut()[[row, col]] = (row as f32 - 1.) * 0.7;
        }
    }
    let gm = gpu.train_step_with_target_noise(&b, &noise).unwrap();
    let em = exact
        .train_step_with_target_noise(&active, &noise.slice_axis(0, 0, b.size).unwrap())
        .unwrap();
    assert_eq!(gm, em);
    assert_eq!(values(&gpu), values(&exact));
    let before = values(&gpu);
    let mut rng = StdRng::seed_from_u64(6);
    let mut expected = rng.clone();
    for kind in 0..6 {
        let mut bad = active.valid_rows();
        match kind {
            0 => bad.states = Tensor::zeros(&[bad.size, 1]),
            1 => bad.size = 99,
            2 => bad.actions.data_mut()[[0, 0]] = f32::INFINITY,
            3 => bad.dones.data_mut()[[0, 0]] = 1.1,
            4 => bad.rewards.data_mut()[[0, 0]] = f32::NAN,
            _ => bad.next_states = Tensor::zeros(&[bad.size * 2]),
        }
        assert!(gpu.train_step_with_rng(&bad, &mut rng).is_err());
        assert_eq!(values(&gpu), before);
        assert_eq!((gpu.updates(), gpu.actor_updates()), (1, 0));
    }
    assert!(no_grad(|| gpu.train_step_with_rng(&active, &mut rng)).is_err());
    assert!(gpu
        .train_step_with_target_noise(&active, &Tensor::zeros(&[1, 2]))
        .is_err());
    assert!(gpu
        .train_step_with_target_noise(&active, &Tensor::full(&[3, 2], f32::NAN))
        .is_err());
    let empty = ContinuousTransitionBatch::new(0, 2, 2);
    assert_eq!(
        gpu.train_step_with_rng(&empty, &mut rng).unwrap(),
        (0., None)
    );
    for (state, std) in [
        (&[0.][..], 0.),
        (&[0., f32::NAN][..], 0.),
        (&[0., 0.][..], -1.),
        (&[0., 0.][..], f32::INFINITY),
    ] {
        assert!(gpu.select_action_with_rng(state, std, &mut rng).is_err());
    }
    assert_eq!(rng.gen::<u64>(), expected.gen::<u64>());
    assert_eq!(values(&gpu), before);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn scheduled_actor_failure_rolls_back_prepared_critic_and_adam_updates() {
    let mut c = config();
    c.policy_delay = 1;
    let mut gpu = GpuTd3::new_seeded(context(), c.clone(), 42).unwrap();
    let mut control = GpuTd3::new_seeded(context(), c, 42).unwrap();
    let actor_weight = gpu.actor().parameters()[0].clone();
    let clean = actor_weight.detach();
    let corrupt = rustforge_autograd::gpu::GpuVariable::new(
        context(),
        &Tensor::full(actor_weight.data().shape(), f32::NAN),
        true,
    )
    .unwrap();
    actor_weight.copy_data_from(&corrupt).unwrap();
    let before = values(&gpu);
    let b = batch();
    let noise = Tensor::zeros(&[b.size, 2]);
    assert!(matches!(
        gpu.train_step_with_target_noise(&b, &noise),
        Err(GpuTd3Error::NonFinite)
    ));
    assert_eq!((gpu.updates(), gpu.actor_updates()), (0, 0));
    assert_eq!(&values(&gpu)[6..], &before[6..]);
    actor_weight.copy_data_from(&clean).unwrap();
    for _ in 0..3 {
        assert_eq!(
            gpu.train_step_with_target_noise(&b, &noise).unwrap(),
            control.train_step_with_target_noise(&b, &noise).unwrap()
        );
        assert_eq!(values(&gpu), values(&control));
    }
    // Finite forward loss can still produce gradient-square overflow before Adam.
    let before = values(&gpu);
    let mut overflow = b.valid_rows();
    overflow.size = 1;
    overflow.rewards.data_mut()[[0, 0]] = 1e19;
    assert!(matches!(
        gpu.train_step_with_target_noise(&overflow, &Tensor::zeros(&[1, 2])),
        Err(GpuTd3Error::NonFinite)
    ));
    assert_eq!(values(&gpu), before);
    assert_eq!((gpu.updates(), gpu.actor_updates()), (3, 3));
    assert_eq!(
        gpu.train_step_with_target_noise(&b, &noise).unwrap(),
        control.train_step_with_target_noise(&b, &noise).unwrap()
    );
    assert_eq!(values(&gpu), values(&control));
    // Public parameter handles remain live across successful transactional updates.
    assert_eq!(
        actor_weight.to_cpu().unwrap().to_vec(),
        gpu.actor().parameters()[0].to_cpu().unwrap().to_vec()
    );
}

#[path = "../examples/support/gpu_td3_target.rs"]
mod target;
#[test]
#[ignore = "requires a GPU adapter"]
fn fresh_continuous_environment_replay_improves_deterministic_policy() {
    let r = target::learn(context()).unwrap();
    assert!(r.final_cost < 0.01 && r.final_cost < r.initial_cost * 0.1);
    assert!((r.action - 0.5).abs() < 0.1);
    assert_eq!((r.critic_updates, r.actor_updates), (536, 268));
}
