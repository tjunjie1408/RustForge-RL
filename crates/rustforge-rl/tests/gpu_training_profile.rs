#![cfg(feature = "gpu")]
use rustforge_nn::gpu::GpuModule;
use rustforge_rl::{
    agent::{DQNConfig, GpuDqn},
    buffer::TransitionBatch,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn profiling_preserves_seeded_training_and_records_backward_optimizer_phases() {
    let plain = GpuContext::new().unwrap();
    let profiled = plain.with_profiling();
    let mut a = GpuDqn::new_seeded(&plain, DQNConfig::default(), 42).unwrap();
    let mut b = GpuDqn::new_seeded(&profiled, DQNConfig::default(), 42).unwrap();
    let mut batch = TransitionBatch::new(4, 4);
    batch.states = Tensor::full(&[4, 4], 0.2);
    batch.next_states = Tensor::full(&[4, 4], 0.3);
    batch.rewards = Tensor::ones(&[4, 1]);
    batch.size = 4;
    for _ in 0..3 {
        assert_eq!(
            a.train_step(&batch).unwrap().to_bits(),
            b.train_step(&batch).unwrap().to_bits()
        );
    }
    let snapshot = profiled.profile_snapshot().unwrap();
    assert_eq!(snapshot.phases["backward"].calls, 3);
    assert_eq!(snapshot.phases["optimizer"].calls, 3);
    assert!(snapshot.phases["backward"].counters.compute_dispatches > 0);
    assert!(snapshot.phases["optimizer"].counters.compute_dispatches > 0);
    assert!(snapshot.phases["linear_forward"].calls > 0);
    assert_eq!(snapshot.counters.readbacks, 3);
    assert_eq!(snapshot.counters.host_waits, 3);
    assert_eq!(a.train_steps(), b.train_steps());
    for (x, y) in a.q_net().parameters().iter().zip(b.q_net().parameters()) {
        assert_eq!(x.to_cpu().unwrap().to_vec(), y.to_cpu().unwrap().to_vec());
    }
    assert!(plain.profile_snapshot().is_none());
}
