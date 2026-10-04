#![cfg(feature = "gpu")]
use rustforge_nn::gpu::GpuModule;
use rustforge_rl::{
    agent::{gpu_sac::GpuSac, SACConfig},
    buffer::ContinuousTransitionBatch,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::{fs, sync::OnceLock};
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().unwrap())
}
fn config(tau: f32) -> SACConfig {
    let mut c = SACConfig::new(2, 1, vec![-2.], vec![3.]);
    c.hidden_dim = 4;
    c.tau = tau;
    c
}
fn batch() -> ContinuousTransitionBatch {
    let mut b = ContinuousTransitionBatch::new(3, 2, 1);
    b.size = 3;
    b.states = Tensor::from_vec(vec![0.2, -0.3, 0.8, 0.2, -0.5, 0.9], &[3, 2]);
    b.next_states = b.states.clone();
    b.actions = Tensor::from_vec(vec![0.5, -1., 2.], &[3, 1]);
    b.rewards = Tensor::from_vec(vec![0.5, -0.3, 1.], &[3, 1]);
    b.dones = Tensor::from_vec(vec![0., 1., 0.25], &[3, 1]);
    b
}
fn bits(a: &GpuSac) -> Vec<Vec<u32>> {
    a.actor()
        .parameters()
        .into_iter()
        .chain(
            [
                a.critic1(),
                a.critic2(),
                a.critic1_target(),
                a.critic2_target(),
            ]
            .into_iter()
            .flat_map(GpuModule::parameters),
        )
        .chain([a.log_alpha().clone()])
        .map(|p| {
            p.to_cpu()
                .unwrap()
                .to_vec()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}
#[test]
#[ignore = "requires a GPU adapter"]
fn checkpoints_resume_bit_identical_three_adam_temperature_and_target_updates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("original.chk");
    let resumed_path = dir.path().join("resumed.chk");
    for tau in [0., 0.2, 1.] {
        let mut a = GpuSac::new_seeded(context(), config(tau), 42).unwrap();
        let b = batch();
        let noise = Tensor::from_vec(vec![0.1, -0.3, 0.7], &[3, 1]);
        for _ in 0..3 {
            a.train_step_with_noise(&b, &noise, &noise).unwrap();
        }
        a.save_checkpoint(&path).unwrap();
        let mut r = GpuSac::load_checkpoint(context(), &path).unwrap();
        assert_eq!(a.config(), r.config());
        assert_eq!(bits(&a), bits(&r));
        assert!(r.actor().parameters().iter().all(|p| p.grad().is_none()));
        assert!(r
            .critic1_target()
            .parameters()
            .iter()
            .chain(&r.critic2_target().parameters())
            .all(|p| !p.requires_grad() && p.grad().is_none()));
        for _ in 0..4 {
            let am = a.train_step_with_noise(&b, &noise, &noise).unwrap();
            let rm = r.train_step_with_noise(&b, &noise, &noise).unwrap();
            assert_eq!(
                [am.0, am.1, am.2, am.3].map(f32::to_bits),
                [rm.0, rm.1, rm.2, rm.3].map(f32::to_bits)
            );
            assert_eq!(bits(&a), bits(&r));
            assert_eq!(a.updates(), r.updates());
        }
        a.save_checkpoint(&path).unwrap();
        r.save_checkpoint(&resumed_path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), fs::read(&resumed_path).unwrap());
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn untrained_restore_and_failed_load_or_save_preserve_live_state_and_previous_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.chk");
    let mut a = GpuSac::new_seeded(context(), config(0.2), 42).unwrap();
    a.save_checkpoint(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let b = GpuSac::load_checkpoint(context(), &path).unwrap();
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(b.updates(), 0);
    a.train_step_with_noise(&batch(), &Tensor::zeros(&[3, 1]), &Tensor::zeros(&[3, 1]))
        .unwrap();
    let before = bits(&a);
    let malformed = dir.path().join("bad.chk");
    fs::write(&malformed, b"not a checkpoint").unwrap();
    assert!(a.restore_checkpoint(&malformed).is_err());
    assert!(a.restore_checkpoint(dir.path().join("missing")).is_err());
    assert_eq!(bits(&a), before);
    assert_eq!(a.updates(), 1);
    assert!(a.save_checkpoint(dir.path()).is_err());
    assert!(a.save_checkpoint(dir.path().join("missing/state")).is_err());
    let p = a.critic1_target().parameters()[0].clone();
    let clean = p.detach();
    let invalid = rustforge_autograd::gpu::GpuVariable::new(
        context(),
        &Tensor::full(p.data().shape(), f32::NAN),
        false,
    )
    .unwrap();
    p.copy_data_from(&invalid).unwrap();
    assert!(a.save_checkpoint(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    p.copy_data_from(&clean).unwrap();
    let alpha = a.log_alpha().clone();
    let alpha_clean = alpha.detach();
    alpha
        .copy_data_from(
            &rustforge_autograd::gpu::GpuVariable::new(
                context(),
                &Tensor::from_vec(vec![100.], &[1]),
                false,
            )
            .unwrap(),
        )
        .unwrap();
    assert!(a.save_checkpoint(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    alpha.copy_data_from(&alpha_clean).unwrap();
    a.restore_checkpoint(&path).unwrap();
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(a.updates(), 0);
}
