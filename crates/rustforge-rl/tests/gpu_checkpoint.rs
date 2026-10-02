#![cfg(feature = "gpu")]
#[path = "../examples/support/gpu_chain.rs"]
mod gpu_chain;
use rustforge_autograd::gpu::GpuVariable;
use rustforge_nn::gpu::GpuModule;
use rustforge_rl::agent::{DQNConfig, GpuDqn};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::{fs, rc::Rc, sync::OnceLock};
fn context() -> &'static GpuContext {
    static CONTEXT: OnceLock<GpuContext> = OnceLock::new();
    CONTEXT.get_or_init(|| GpuContext::new().expect("GPU tests require an adapter"))
}
fn agent(double: bool, frequency: usize) -> GpuDqn {
    GpuDqn::new_seeded(
        context(),
        DQNConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 8,
            lr: 0.03,
            gamma: 0.9,
            target_update_freq: frequency,
            double_dqn: double,
            ..DQNConfig::default()
        },
        42,
    )
    .unwrap()
}
fn bits(t: &Tensor) -> Vec<u32> {
    t.to_vec().iter().map(|v| v.to_bits()).collect()
}
fn assert_agents_equal(a: &GpuDqn, b: &GpuDqn) {
    assert_eq!(a.train_steps(), b.train_steps());
    for (left, right) in [(a.q_net(), b.q_net()), (a.target_net(), b.target_net())] {
        for (a, b) in left.parameters().iter().zip(right.parameters()) {
            assert_eq!(bits(&a.to_cpu().unwrap()), bits(&b.to_cpu().unwrap()));
        }
    }
    assert!(b
        .target_net()
        .parameters()
        .iter()
        .all(|p| !p.requires_grad() && p.grad().is_none() && !p.has_grad_fn()));
}
#[test]
#[ignore = "requires a GPU adapter"]
fn checkpoint_resume_matches_uninterrupted_updates_and_preserves_target_lag() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("agent.chk");
    let replay = gpu_chain::TinyChain::default().full_replay();
    for (double, frequency) in [(false, 5), (true, 5), (true, 0)] {
        let mut original = agent(double, frequency);
        let batch = original.upload_batch(&replay).unwrap();
        for _ in 0..7 {
            original.train_device_batch(&batch).unwrap();
        }
        assert!(original
            .q_net()
            .parameters()
            .iter()
            .zip(original.target_net().parameters())
            .any(|(q, t)| bits(&q.to_cpu().unwrap()) != bits(&t.to_cpu().unwrap())));
        original.save_checkpoint(&path).unwrap();
        let mut resumed = GpuDqn::load_checkpoint(context(), &path).unwrap();
        assert_agents_equal(&original, &resumed);
        assert_eq!(resumed.config().target_update_freq, frequency);
        assert_eq!(resumed.config().double_dqn, double);
        assert!(resumed
            .q_net()
            .parameters()
            .iter()
            .all(|p| p.grad().is_none()));
        for _ in 0..8 {
            let uninterrupted = original.train_device_batch(&batch).unwrap();
            let continuation = resumed.train_device_batch(&batch).unwrap();
            assert_eq!(uninterrupted.to_bits(), continuation.to_bits());
            assert_agents_equal(&original, &resumed);
        }
        original.save_checkpoint(&path).unwrap();
        let saved = fs::read(&path).unwrap();
        resumed.save_checkpoint(&path).unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            saved,
            "all serialized moments/clocks must match after resumed training"
        );
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn untrained_checkpoint_and_successful_restore_preserve_config_and_frozen_parameters() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("agent.chk");
    let original = agent(true, 4);
    original.save_checkpoint(&path).unwrap();
    let mut restored = agent(false, 0);
    let old_handle = restored.q_net().parameters()[0].clone();
    restored.restore_checkpoint(&path).unwrap();
    assert_agents_equal(&original, &restored);
    assert!(!Rc::ptr_eq(
        &old_handle.data(),
        &restored.q_net().parameters()[0].data()
    ));
    assert!(restored.config().double_dqn);
    assert_eq!(restored.config().target_update_freq, 4);
    let replay = gpu_chain::TinyChain::default().full_replay();
    let mut original = original;
    assert_eq!(
        original.train_step(&replay).unwrap().to_bits(),
        restored.train_step(&replay).unwrap().to_bits()
    );
    assert_agents_equal(&original, &restored);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn failed_restore_or_save_preserves_live_state_and_existing_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let good = directory.path().join("good.chk");
    let bad = directory.path().join("bad.chk");
    let mut original = agent(true, 5);
    let replay = gpu_chain::TinyChain::default().full_replay();
    for _ in 0..2 {
        original.train_step(&replay).unwrap();
    }
    original.save_checkpoint(&good).unwrap();
    let saved = fs::read(&good).unwrap();
    let mut control = GpuDqn::load_checkpoint(context(), &good).unwrap();
    let handles: Vec<_> = original
        .q_net()
        .parameters()
        .iter()
        .map(|p| p.data())
        .collect();
    for kind in 0..5 {
        let mut bytes = saved.clone();
        match kind {
            0 => bytes.truncate(11),
            1 => bytes[8..12].copy_from_slice(&99u32.to_le_bytes()),
            2 => bytes.push(0),
            3 => {
                bytes[0] = b'X';
            }
            _ => {
                // V1's first fixed-width payload field is obs_dim (u64).
                // Changing it makes the stored weight shapes incompatible.
                bytes[12..20].copy_from_slice(&3u64.to_le_bytes());
            }
        }
        fs::write(&bad, bytes).unwrap();
        assert!(original.restore_checkpoint(&bad).is_err());
        for (p, data) in original.q_net().parameters().iter().zip(&handles) {
            assert!(Rc::ptr_eq(&p.data(), data));
        }
        assert_agents_equal(&original, &control);
    }
    assert!(original
        .restore_checkpoint(directory.path().join("missing.chk"))
        .is_err());
    assert_eq!(
        original.train_step(&replay).unwrap().to_bits(),
        control.train_step(&replay).unwrap().to_bits()
    );
    assert_agents_equal(&original, &control);
    original.q_net().parameters()[3]
        .copy_data_from(
            &GpuVariable::new(
                context(),
                &Tensor::from_vec(vec![f32::NAN, 0.], &[2]),
                false,
            )
            .unwrap(),
        )
        .unwrap();
    assert!(original.save_checkpoint(&good).is_err());
    assert_eq!(fs::read(&good).unwrap(), saved);
    assert!(fs::read_dir(directory.path()).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains(".tmp-")));
}
