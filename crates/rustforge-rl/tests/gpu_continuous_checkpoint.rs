#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, SeedableRng};
use rustforge_nn::gpu::GpuModule;
use rustforge_rl::{
    agent::{gpu_ppo::GpuPpoContinuous, PPOConfig, PPOContinuousConfig},
    buffer::ContinuousRolloutBatch,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::fs;
fn config() -> PPOContinuousConfig {
    PPOContinuousConfig {
        base: PPOConfig {
            obs_dim: 2,
            hidden_dim: 8,
            lr: 0.001,
            ppo_epochs: 2,
            mini_batch_size: 3,
            ..PPOConfig::default()
        },
        act_dim: 1,
        action_low: vec![-1.],
        action_high: vec![1.],
    }
}
fn batch(agent: &GpuPpoContinuous) -> ContinuousRolloutBatch {
    let mut rng = StdRng::seed_from_u64(17);
    let mut actions = Vec::new();
    let mut logs = Vec::new();
    for _ in 0..5 {
        let (a, lp, _) = agent.select_action_with_rng(&[1., 0.], &mut rng).unwrap();
        actions.extend(a);
        logs.push(lp);
    }
    ContinuousRolloutBatch {
        states: Tensor::from_vec([1., 0.].repeat(5), &[5, 2]),
        actions: Tensor::from_vec(actions, &[5, 1]),
        returns: Tensor::from_vec(vec![1., -0.2, 0.7, 0.1, -0.4], &[5, 1]),
        advantages: Tensor::from_vec(vec![1., -1., 0.3, -0.7, 0.2], &[5, 1]),
        old_log_probs: Tensor::from_vec(logs, &[5, 1]),
        size: 5,
    }
}
fn bits(t: Tensor) -> Vec<u32> {
    t.to_vec().iter().map(|x| x.to_bits()).collect()
}
fn equal(a: &GpuPpoContinuous, b: &GpuPpoContinuous) {
    assert_eq!(a.config(), b.config());
    assert_eq!(
        (a.actor_updates(), a.critic_updates()),
        (b.actor_updates(), b.critic_updates())
    );
    for (a, b) in a
        .actor()
        .parameters()
        .into_iter()
        .chain(a.critic().parameters())
        .zip(
            b.actor()
                .parameters()
                .into_iter()
                .chain(b.critic().parameters()),
        )
    {
        assert_eq!(bits(a.to_cpu().unwrap()), bits(b.to_cpu().unwrap()));
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn checkpoint_resume_matches_fixed_minibatch_training_with_identical_shuffle_stream() {
    let context = GpuContext::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ppo.chk");
    let mut original = GpuPpoContinuous::new_seeded(&context, config(), 42).unwrap();
    let batch = batch(&original);
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..3 {
        original.train_on_batch_with_rng(&batch, &mut rng).unwrap();
    }
    original.save_checkpoint(&path).unwrap();
    let mut resumed = GpuPpoContinuous::load_checkpoint(&context, &path).unwrap();
    let mut other = rng.clone();
    equal(&original, &resumed);
    assert!(resumed
        .actor()
        .parameters()
        .into_iter()
        .chain(resumed.critic().parameters())
        .all(|p| p.grad().is_none()));
    for _ in 0..3 {
        let a = original.train_on_batch_with_rng(&batch, &mut rng).unwrap();
        let b = resumed.train_on_batch_with_rng(&batch, &mut other).unwrap();
        assert_eq!(
            [a.policy_loss, a.value_loss, a.base_entropy].map(f32::to_bits),
            [b.policy_loss, b.value_loss, b.base_entropy].map(f32::to_bits)
        );
        equal(&original, &resumed);
    }
    original.save_checkpoint(&path).unwrap();
    let second = directory.path().join("resumed.chk");
    resumed.save_checkpoint(&second).unwrap();
    assert_eq!(fs::read(&path).unwrap(), fs::read(&second).unwrap());
    assert_eq!(
        (original.actor_updates(), original.critic_updates()),
        (24, 24)
    );
}
#[test]
#[ignore = "requires a GPU adapter"]
fn untrained_restore_and_failed_io_preserve_live_parameters_and_saved_file() {
    let context = GpuContext::new().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ppo.chk");
    let mut original = GpuPpoContinuous::new_seeded(&context, config(), 42).unwrap();
    original.save_checkpoint(&path).unwrap();
    let saved = fs::read(&path).unwrap();
    let batch = batch(&original);
    let old = original.actor().parameters()[0].clone();
    let untrained = GpuPpoContinuous::load_checkpoint(&context, &path).unwrap();
    equal(&original, &untrained);
    original
        .train_on_batch_with_rng(&batch, &mut StdRng::seed_from_u64(1))
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
    let p = original.actor().parameters()[0].clone();
    let mut invalid = p.to_cpu().unwrap();
    invalid.data_mut().fill(f32::NAN);
    p.copy_data_from(
        &rustforge_autograd::gpu::GpuVariable::new(&context, &invalid, false).unwrap(),
    )
    .unwrap();
    assert!(original.save_checkpoint(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), saved);
}
