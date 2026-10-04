#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustforge_autograd::no_grad;
use rustforge_nn::{gpu::GpuModule, Module};
use rustforge_rl::{
    agent::{
        gpu_sac::{GpuSac, GpuSacError},
        sac::{SACConfig, SAC},
    },
    buffer::{ContinuousReplayBuffer, ContinuousTransitionBatch},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::sync::OnceLock;
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().expect("GPU SAC tests require an adapter"))
}
fn config() -> SACConfig {
    let mut c = SACConfig::new(2, 2, vec![-2., 1.], vec![4., 5.]);
    c.hidden_dim = 8;
    c.actor_lr = 0.002;
    c.critic_lr = 0.003;
    c.alpha_lr = 0.004;
    c.tau = 0.2;
    c
}
fn batch() -> ContinuousTransitionBatch {
    let mut replay = ContinuousReplayBuffer::new(8, 2, 2);
    for (s, a, r, n, d) in [
        ([0.2, -0.3], [0.5, 2.], 0.7, [0.3, -0.2], false),
        ([0.7, 0.2], [-1., 4.], -0.3, [0.1, 0.4], true),
        ([-0.1, 0.8], [3., 1.5], 0.2, [0.2, 0.5], false),
    ] {
        replay.push(&s, &a, r, &n, d);
    }
    let mut b = ContinuousTransitionBatch::new(8, 2, 2);
    replay.sample_with_rng(8, &mut b, &mut StdRng::seed_from_u64(8));
    b.dones.data_mut()[[1, 0]] = 0.25;
    b
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
fn values(g: &GpuSac) -> Vec<Vec<f32>> {
    g.actor()
        .parameters()
        .into_iter()
        .chain(
            [
                g.critic1(),
                g.critic2(),
                g.critic1_target(),
                g.critic2_target(),
            ]
            .into_iter()
            .flat_map(GpuModule::parameters),
        )
        .chain([g.log_alpha().clone()])
        .map(|p| p.to_cpu().unwrap().to_vec())
        .collect()
}
fn compare(c: &SAC, g: &GpuSac, eps: f32) {
    for (c, g) in c.actor.parameters().iter().zip(g.actor().parameters()) {
        close(&c.data().to_vec(), &g.to_cpu().unwrap().to_vec(), eps);
    }
    for (c, g) in [
        c.critic1(),
        c.critic2(),
        c.critic1_target(),
        c.critic2_target(),
    ]
    .into_iter()
    .zip([
        g.critic1(),
        g.critic2(),
        g.critic1_target(),
        g.critic2_target(),
    ]) {
        for (c, g) in c.parameters().iter().zip(g.parameters()) {
            close(&c.data().to_vec(), &g.to_cpu().unwrap().to_vec(), eps);
        }
    }
    approx::assert_abs_diff_eq!(c.alpha(), g.alpha().unwrap(), epsilon = eps.max(2e-7));
}
#[test]
#[ignore = "requires a GPU adapter"]
fn complete_seeded_cpu_sac_update_order_adam_and_polyak_parity() {
    let b = batch();
    for tau in [0., 0.2, 1.] {
        let mut cfg = config();
        cfg.tau = tau;
        let mut c = SAC::new_seeded(cfg.clone(), u64::MAX - 2);
        let mut g = GpuSac::new_seeded(context(), cfg, u64::MAX - 2).unwrap();
        compare(&c, &g, 0.);
        let mut ct = StdRng::seed_from_u64(67);
        let mut gt = ct.clone();
        let mut ca = StdRng::seed_from_u64(68);
        let mut ga = ca.clone();
        for step in 1..=6 {
            let cm = c.train_step_with_rngs(&b, &mut ct, &mut ca);
            let gm = g.train_step_with_rngs(&b, &mut gt, &mut ga).unwrap();
            close(&[cm.0, cm.1, cm.2, cm.3], &[gm.0, gm.1, gm.2, gm.3], 3e-4);
            compare(&c, &g, 3e-4);
            assert_eq!(g.updates(), step);
            for t in [g.critic1_target(), g.critic2_target()] {
                assert!(t
                    .parameters()
                    .iter()
                    .all(|p| !p.requires_grad() && p.grad().is_none()));
            }
            assert!(g
                .actor()
                .parameters()
                .iter()
                .chain(g.critic1().parameters().iter())
                .all(|p| p.grad().is_none()));
        }
        assert_eq!(ct.gen::<u64>(), gt.gen::<u64>());
        assert_eq!(ca.gen::<u64>(), ga.gen::<u64>());
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_selection_deterministic_bounds_empty_batches_and_partial_rows() {
    let cfg = config();
    let c = SAC::new_seeded(cfg.clone(), 42);
    let mut g = GpuSac::new_seeded(context(), cfg, 42).unwrap();
    let mut cr = StdRng::seed_from_u64(9);
    let mut gr = cr.clone();
    for _ in 0..4 {
        let a = g.select_action_with_rng(&[0.2, -0.3], &mut gr).unwrap();
        close(&a, &c.select_action_with_rng(&[0.2, -0.3], &mut cr), 2e-5);
        assert!((-2. ..=4.).contains(&a[0]) && (1. ..=5.).contains(&a[1]));
    }
    let mut untouched = gr.clone();
    let deterministic = g.deterministic_action(&[0.2, -0.3]).unwrap();
    close(&deterministic, &c.deterministic_action(&[0.2, -0.3]), 2e-5);
    assert!(g.select_action_with_rng(&[f32::NAN, 0.], &mut gr).is_err());
    assert_eq!(gr.gen::<u64>(), untouched.gen::<u64>());
    let empty = ContinuousTransitionBatch::new(0, 2, 2);
    let before = values(&g);
    let mut tr = StdRng::seed_from_u64(5);
    let mut ar = StdRng::seed_from_u64(6);
    let mut tc = tr.clone();
    let mut ac = ar.clone();
    let m = g.train_step_with_rngs(&empty, &mut tr, &mut ar).unwrap();
    assert_eq!((m.0, m.1, m.2), (0., 0., 0.));
    assert_eq!(g.updates(), 0);
    assert_eq!(values(&g), before);
    assert_eq!(tr.gen::<u64>(), tc.gen::<u64>());
    assert_eq!(ar.gen::<u64>(), ac.gen::<u64>());
    let mut b = batch();
    for t in [
        &mut b.states,
        &mut b.next_states,
        &mut b.actions,
        &mut b.rewards,
        &mut b.dones,
    ] {
        let active = b.size * t.shape()[1];
        for v in t.data_mut().iter_mut().skip(active) {
            *v = f32::NAN;
        }
    }
    let mut other = GpuSac::new_seeded(context(), config(), 42).unwrap();
    let clean = b.valid_rows();
    let noise = Tensor::from_vec(vec![0.3; 16], &[8, 2]);
    g.train_step_with_noise(&b, &noise, &noise).unwrap();
    other.train_step_with_noise(&clean, &noise, &noise).unwrap();
    assert_eq!(values(&g), values(&other));
}
#[test]
#[ignore = "requires a GPU adapter"]
fn validation_and_late_temperature_failure_preserve_all_training_state() {
    let b = batch();
    let noise = Tensor::from_vec(vec![0.1; 6], &[3, 2]);
    let mut g = GpuSac::new_seeded(context(), config(), 42).unwrap();
    let before = values(&g);
    let mut bad = b.valid_rows();
    bad.size = 9;
    assert!(g.train_step_with_noise(&bad, &noise, &noise).is_err());
    bad = b.valid_rows();
    bad.dones.data_mut()[[0, 0]] = 1.1;
    assert!(g.train_step_with_noise(&bad, &noise, &noise).is_err());
    bad = b.valid_rows();
    bad.states.data_mut()[[0, 0]] = f32::NAN;
    let mut tr = StdRng::seed_from_u64(7);
    let mut ar = StdRng::seed_from_u64(8);
    let mut tc = tr.clone();
    let mut ac = ar.clone();
    assert!(g.train_step_with_rngs(&bad, &mut tr, &mut ar).is_err());
    assert_eq!(tr.gen::<u64>(), tc.gen::<u64>());
    assert_eq!(ar.gen::<u64>(), ac.gen::<u64>());
    assert!(no_grad(|| g.train_step_with_noise(&b, &noise, &noise)).is_err());
    assert!(g
        .train_step_with_noise(&b, &Tensor::zeros(&[2, 2]), &noise)
        .is_err());
    let mut nan_noise = noise.clone();
    nan_noise.data_mut()[[0, 0]] = f32::NAN;
    assert!(g.train_step_with_noise(&b, &noise, &nan_noise).is_err());
    assert_eq!(values(&g), before);
    assert_eq!(g.updates(), 0);
    let mut c = config();
    c.alpha_lr = 1e38;
    let mut failing = GpuSac::new_seeded(context(), c, 42).unwrap();
    let before = values(&failing);
    for _ in 0..2 {
        assert!(matches!(
            failing.train_step_with_noise(&b, &noise, &noise),
            Err(GpuSacError::InvalidTemperature) | Err(GpuSacError::NonFinite)
        ));
        assert_eq!(values(&failing), before);
        assert_eq!(failing.updates(), 0);
    }
    // Host rejection before a subsequent valid step cannot alter resident Adam moments.
    let mut reference = GpuSac::new_seeded(context(), config(), 42).unwrap();
    g.train_step_with_noise(&b, &noise, &noise).unwrap();
    reference.train_step_with_noise(&b, &noise, &noise).unwrap();
    assert_eq!(values(&g), values(&reference));
}

#[test]
#[ignore = "requires a GPU adapter"]
fn late_actor_overflow_preserves_adam_continuation_and_live_handles() {
    let b = batch();
    let noise = Tensor::from_vec(vec![0.1; 6], &[3, 2]);
    let bad = Tensor::from_vec(vec![1e30; 6], &[3, 2]);
    let mut a = GpuSac::new_seeded(context(), config(), 42).unwrap();
    let mut reference = GpuSac::new_seeded(context(), config(), 42).unwrap();
    let retained = a.actor().parameters()[0].clone();
    for step in 1..=3 {
        a.train_step_with_noise(&b, &noise, &noise).unwrap();
        reference.train_step_with_noise(&b, &noise, &noise).unwrap();
        let before = values(&a);
        assert!(a.train_step_with_noise(&b, &noise, &bad).is_err());
        assert_eq!(a.updates(), step);
        assert_eq!(values(&a), before);
        assert_eq!(values(&a), values(&reference));
        assert_eq!(
            retained.to_cpu().unwrap().to_vec(),
            a.actor().parameters()[0].to_cpu().unwrap().to_vec()
        );
    }
}
