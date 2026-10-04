#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, SeedableRng};
use rustforge_rl::{
    agent::{gpu_ppo::GpuPpoDiscrete, PPOConfig, PPODiscreteConfig},
    buffer::RolloutBatch,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::fs;
fn config() -> PPODiscreteConfig {
    PPODiscreteConfig {
        base: PPOConfig {
            obs_dim: 2,
            hidden_dim: 8,
            lr: 0.01,
            ppo_epochs: 2,
            mini_batch_size: 3,
            ..PPOConfig::default()
        },
        num_actions: 2,
    }
}
fn batch() -> RolloutBatch {
    RolloutBatch {
        states: Tensor::from_vec(vec![1., 0., 0., 1., 0.2, 0.7, -0.3, 0.8, 0.9, 0.2], &[5, 2]),
        actions: vec![0, 1, 0, 1, 1],
        returns: Tensor::from_vec(vec![1., -0.2, 0.7, 0.1, -0.4], &[5, 1]),
        advantages: Tensor::from_vec(vec![1., -1., 0.3, -0.7, 0.2], &[5, 1]),
        old_log_probs: Tensor::from_vec(vec![-0.7, -0.6, -0.8, -0.5, -0.9], &[5, 1]),
        size: 5,
    }
}
fn bits(t: Tensor) -> Vec<u32> {
    t.to_vec().iter().map(|x| x.to_bits()).collect()
}
fn equal(a: &GpuPpoDiscrete, b: &GpuPpoDiscrete) {
    assert_eq!(a.config(), b.config());
    assert_eq!(a.updates(), b.updates());
    for (a, b) in a.net().parameters().iter().zip(b.net().parameters()) {
        assert_eq!(bits(a.to_cpu().unwrap()), bits(b.to_cpu().unwrap()));
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn checkpoint_resume_matches_fixed_minibatch_training_with_identical_shuffle_stream() {
    let context = GpuContext::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ppo.chk");
    let mut original = GpuPpoDiscrete::new_seeded(&context, config(), 42).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..3 {
        original
            .train_on_batch_with_rng(&batch(), &mut rng)
            .unwrap();
    }
    original.save_checkpoint(&path).unwrap();
    let mut resumed = GpuPpoDiscrete::load_checkpoint(&context, &path).unwrap();
    let mut other = rng.clone();
    equal(&original, &resumed);
    assert!(resumed
        .net()
        .parameters()
        .iter()
        .all(|p| p.grad().is_none()));
    for _ in 0..3 {
        let a = original
            .train_on_batch_with_rng(&batch(), &mut rng)
            .unwrap();
        let b = resumed
            .train_on_batch_with_rng(&batch(), &mut other)
            .unwrap();
        assert_eq!(
            [a.policy_loss, a.value_loss, a.entropy, a.total_loss].map(f32::to_bits),
            [b.policy_loss, b.value_loss, b.entropy, b.total_loss].map(f32::to_bits)
        );
        equal(&original, &resumed);
    }
    original.save_checkpoint(&path).unwrap();
    let second = directory.path().join("resumed.chk");
    resumed.save_checkpoint(&second).unwrap();
    assert_eq!(fs::read(&path).unwrap(), fs::read(&second).unwrap());
    assert_eq!(original.updates(), 24);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn untrained_restore_and_failed_io_preserve_live_parameters_and_saved_file() {
    let context = GpuContext::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ppo.chk");
    let mut original = GpuPpoDiscrete::new_seeded(&context, config(), 42).unwrap();
    original.save_checkpoint(&path).unwrap();
    let saved = fs::read(&path).unwrap();
    let old = original.net().parameters()[0].clone();
    let untrained = GpuPpoDiscrete::load_checkpoint(&context, &path).unwrap();
    equal(&original, &untrained);
    original
        .train_on_batch_with_rng(&batch(), &mut StdRng::seed_from_u64(1))
        .unwrap();
    let changed = bits(old.to_cpu().unwrap());
    original.restore_checkpoint(&path).unwrap();
    equal(&original, &untrained);
    assert_eq!(bits(old.to_cpu().unwrap()), changed);
    for data in [
        vec![],
        b"RFGPUDQN\x01\0\0\0".to_vec(),
        saved[..saved.len() - 1].to_vec(),
    ] {
        let bad = directory.path().join("bad.chk");
        fs::write(&bad, data).unwrap();
        assert!(original.restore_checkpoint(&bad).is_err());
        equal(&original, &untrained);
    }
    assert!(original
        .restore_checkpoint(directory.path().join("missing"))
        .is_err());
    assert!(original
        .save_checkpoint(directory.path().join("missing-parent").join("out.chk"))
        .is_err());
    assert_eq!(fs::read(&path).unwrap(), saved);
    let p = original.net().parameters()[0].clone();
    let mut invalid = p.to_cpu().unwrap();
    invalid.data_mut().fill(f32::NAN);
    p.copy_data_from(
        &rustforge_autograd::gpu::GpuVariable::new(&context, &invalid, false).unwrap(),
    )
    .unwrap();
    assert!(original.save_checkpoint(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), saved);
}
