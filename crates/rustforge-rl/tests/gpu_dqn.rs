#![cfg(feature = "gpu")]
use rustforge_autograd::{gpu::GpuVariable, no_grad, Variable};
use rustforge_nn::{gpu::GpuModule, Module};
use rustforge_rl::{
    agent::{DQNConfig, GpuDqn, GpuDqnError, DQN},
    buffer::TransitionBatch,
    training::replay_done,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
#[path = "../examples/support/gpu_chain.rs"]
mod gpu_chain;
use rustforge_rl::env::Environment;
use std::sync::OnceLock;
fn context() -> &'static GpuContext {
    static CONTEXT: OnceLock<GpuContext> = OnceLock::new();
    CONTEXT.get_or_init(|| GpuContext::new().expect("GPU tests require an adapter"))
}
fn config(double: bool, frequency: usize) -> DQNConfig {
    DQNConfig {
        obs_dim: 2,
        num_actions: 2,
        hidden_dim: 8,
        lr: 0.02,
        gamma: 0.9,
        target_update_freq: frequency,
        double_dqn: double,
        ..DQNConfig::default()
    }
}
fn batch() -> TransitionBatch {
    let mut b = TransitionBatch::new(4, 2);
    b.states = Tensor::from_vec(vec![0.2, -0.6, 1., 0.3, -0.2, 0.8, 2., 1.], &[4, 2]);
    b.next_states = Tensor::from_vec(vec![0.4, 0.1, -0.7, 0.2, 0.8, -0.3, 1., -1.], &[4, 2]);
    b.actions = vec![0, 1, 1, 0];
    b.rewards = Tensor::from_vec(vec![0.3, -0.4, 0.7, 1.], &[4, 1]);
    // True terminals stop bootstrap; time-limit truncation does not.
    b.dones = Tensor::from_vec(
        vec![1., 0., if replay_done(false, true) { 1. } else { 0. }, 0.],
        &[4, 1],
    );
    b.size = 4;
    b
}
fn close(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = epsilon);
    }
}
fn cpu_targets(cpu: &DQN, b: &TransitionBatch) -> Tensor {
    let next = Variable::from_tensor(b.next_states.clone());
    let tq = no_grad(|| cpu.target_net().forward(&next)).data().clone();
    let values = if cpu.config().double_dqn {
        let oq = no_grad(|| cpu.q_net().forward(&next));
        let actions = oq.data().argmax_axis(1).unwrap();
        tq.gather(1, &actions).unwrap()
    } else {
        tq.max_axis(1, true).unwrap()
    };
    &b.rewards + &(&(&values * cpu.config().gamma) * &(&Tensor::ones(b.dones.shape()) - &b.dones))
}

#[test]
#[ignore = "requires a GPU adapter"]
fn vanilla_and_double_dqn_targets_gradients_updates_and_sync_match_cpu() {
    for double in [false, true] {
        let mut gpu = GpuDqn::new_seeded(context(), config(double, 2), 42).unwrap();
        let mut cpu = DQN::new(config(double, 2));
        for (c, g) in cpu
            .q_net()
            .parameters()
            .iter()
            .zip(gpu.q_net().parameters())
        {
            c.set_data(g.to_cpu().unwrap());
        }
        cpu.update_target();
        let b = batch();
        let resident = gpu.upload_batch(&b).unwrap();
        for step in 1..=4 {
            let expected = cpu_targets(&cpu, &b);
            let target = gpu.td_targets(&resident).unwrap();
            assert!(!target.requires_grad() && !target.has_grad_fn());
            close(&target.to_cpu().unwrap().to_vec(), &expected.to_vec(), 2e-6);
            let gl = gpu.train_device_batch(&resident).unwrap();
            let cl = cpu.train_step(&b, None).0;
            close(&[gl], &[cl], 3e-6);
            assert_eq!(gpu.train_steps(), step);
            for (g, c) in gpu
                .q_net()
                .parameters()
                .iter()
                .zip(cpu.q_net().parameters())
            {
                close(
                    &g.grad_cpu().unwrap().unwrap().to_vec(),
                    &c.grad().unwrap().to_vec(),
                    3e-6,
                );
                close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 3e-6);
            }
            for (g, c) in gpu
                .target_net()
                .parameters()
                .iter()
                .zip(cpu.target_net().parameters())
            {
                close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 3e-6);
                assert!(!g.requires_grad() && g.grad().is_none() && !g.has_grad_fn());
            }
        }
        assert_eq!(
            gpu.select_greedy_action(&[0.2, -0.6]).unwrap(),
            cpu.select_greedy_action(&[0.2, -0.6])
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn targets_stay_frozen_until_sync_and_double_dqn_selects_online_actions() {
    for double in [false, true] {
        let mut gpu = GpuDqn::new_seeded(context(), config(double, 0), 7).unwrap();
        let tp = gpu.target_net().parameters();
        let originals: Vec<_> = tp.iter().map(|p| p.to_cpu().unwrap().to_vec()).collect();
        let b = batch();
        let resident = gpu.upload_batch(&b).unwrap();
        gpu.train_device_batch(&resident).unwrap();
        for (p, original) in tp.iter().zip(&originals) {
            close(&p.to_cpu().unwrap().to_vec(), original, 0.);
        }
        // Force online action 0 and target action 1 to differ. This directly
        // distinguishes Double DQN evaluation from vanilla max evaluation.
        for p in gpu.q_net().parameters().iter().chain(&tp) {
            p.copy_data_from(
                &GpuVariable::from_device(
                    context(),
                    context().zeros(p.data().shape()).unwrap(),
                    false,
                )
                .unwrap(),
            )
            .unwrap();
        }
        gpu.q_net().parameters()[3]
            .copy_data_from(
                &GpuVariable::new(context(), &Tensor::from_vec(vec![10., 0.], &[2]), false)
                    .unwrap(),
            )
            .unwrap();
        tp[3]
            .copy_data_from(
                &GpuVariable::new(context(), &Tensor::from_vec(vec![2., 5.], &[2]), false).unwrap(),
            )
            .unwrap();
        let targets = gpu
            .td_targets(&resident)
            .unwrap()
            .to_cpu()
            .unwrap()
            .to_vec();
        let next = if double { 2. } else { 5. };
        close(
            &targets,
            &[0.3, -0.4 + 0.9 * next, 0.7 + 0.9 * next, 1. + 0.9 * next],
            1e-6,
        );
        gpu.update_target().unwrap();
        for (t, q) in tp.iter().zip(gpu.q_net().parameters()) {
            close(
                &t.to_cpu().unwrap().to_vec(),
                &q.to_cpu().unwrap().to_vec(),
                0.,
            );
            assert!(!t.requires_grad() && t.grad().is_none());
        }
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn partial_batches_invalid_input_and_divergence_fail_without_training() {
    let mut gpu = GpuDqn::new_seeded(context(), config(false, 3), 12).unwrap();
    let mut b = batch();
    b.size = 2;
    // Active rows upload successfully even with invalid stale capacity.
    b.states.data_mut()[[3, 0]] = f32::NAN;
    b.actions[3] = usize::MAX;
    b.rewards.data_mut()[[3, 0]] = f32::NAN;
    let resident = gpu.upload_batch(&b).unwrap();
    assert_eq!(resident.len(), 2);
    assert!(!resident.is_empty());
    gpu.train_device_batch(&resident).unwrap();
    let before = gpu.train_steps();
    assert!(matches!(
        no_grad(|| gpu.train_device_batch(&resident)),
        Err(GpuDqnError::GradientsDisabled)
    ));
    assert_eq!(gpu.train_steps(), before);
    b.actions[0] = 2;
    assert!(gpu.upload_batch(&b).is_err());
    b.actions[0] = 0;
    b.dones.data_mut()[[0, 0]] = 0.5;
    assert!(gpu.upload_batch(&b).is_err());
    b.dones.data_mut()[[0, 0]] = 1.;
    b.states.data_mut()[[0, 0]] = f32::INFINITY;
    assert!(gpu.upload_batch(&b).is_err());
    b.size = 0;
    assert!(gpu.upload_batch(&b).is_err());
    b = batch();
    b.size = 5;
    assert!(gpu.upload_batch(&b).is_err());
    b = batch();
    b.next_states = Tensor::zeros(&[4, 3]);
    assert!(gpu.upload_batch(&b).is_err());
    assert!(gpu.select_greedy_action(&[1.]).is_err());
    assert!(gpu.select_greedy_action(&[f32::NAN, 0.]).is_err());
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    let foreign = GpuDqn::new_seeded(other, config(false, 3), 12)
        .unwrap()
        .upload_batch(&batch())
        .unwrap();
    assert!(gpu.train_device_batch(&foreign).is_err());
    // Loss validation occurs before backward, optimizer, counter or target sync.
    let bias = gpu.q_net().parameters()[3].clone();
    bias.copy_data_from(
        &GpuVariable::new(
            context(),
            &Tensor::from_vec(vec![f32::NAN, 0.], &[2]),
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let snapshots: Vec<_> = gpu.q_net().parameters().iter().map(|p| p.data()).collect();
    assert!(matches!(
        gpu.train_step(&batch()),
        Err(GpuDqnError::NonFiniteLoss)
    ));
    assert_eq!(gpu.train_steps(), before);
    for (p, data) in gpu.q_net().parameters().iter().zip(snapshots) {
        assert!(std::rc::Rc::ptr_eq(&p.data(), &data));
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn configuration_validation_and_seed_reproducibility() {
    for kind in 0..6 {
        let mut c = config(false, 2);
        match kind {
            0 => c.obs_dim = 0,
            1 => c.num_actions = 0,
            2 => c.hidden_dim = 0,
            3 => c.lr = f32::NAN,
            4 => c.gamma = 1.1,
            _ => c.use_per = true,
        }
        assert!(GpuDqn::new_seeded(context(), c, 1).is_err());
    }
    let a = GpuDqn::new_seeded(context(), config(true, 0), 44).unwrap();
    let b = GpuDqn::new_seeded(context(), config(true, 0), 44).unwrap();
    for (a, b) in a.q_net().parameters().iter().zip(b.q_net().parameters()) {
        close(
            &a.to_cpu().unwrap().to_vec(),
            &b.to_cpu().unwrap().to_vec(),
            0.,
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn single_transition_overfit_and_two_state_policy_converge() {
    let mut gpu = GpuDqn::new_seeded(context(), config(true, 10), 42).unwrap();
    let mut one = TransitionBatch::new(1, 2);
    one.states = Tensor::from_vec(vec![1., 0.], &[1, 2]);
    one.actions = vec![0];
    one.rewards = Tensor::from_vec(vec![1.], &[1, 1]);
    one.dones = Tensor::ones(&[1, 1]);
    one.size = 1;
    let resident = gpu.upload_batch(&one).unwrap();
    let initial = gpu.train_device_batch(&resident).unwrap();
    let mut loss = initial;
    for _ in 0..79 {
        loss = gpu.train_device_batch(&resident).unwrap();
    }
    assert!(
        loss < 0.001 && loss < initial * 0.01,
        "single-transition MSE {initial} -> {loss}"
    );
    // Replay is collected through a deterministic Environment and includes
    // nonterminal bootstrapping followed by a rewarding terminal state.
    let mut env = gpu_chain::TinyChain::default();
    let full = env.full_replay();
    let resident = gpu.upload_batch(&full).unwrap();
    for _ in 0..220 {
        loss = gpu.train_device_batch(&resident).unwrap();
    }
    assert!(loss < 0.001, "two-state MSE {loss}");
    assert_eq!(gpu.select_greedy_action(&[1., 0.]).unwrap(), 0);
    assert_eq!(gpu.select_greedy_action(&[0., 1.]).unwrap(), 1);
    let (state, _) = env.reset(Some(0));
    let action = gpu.select_greedy_action(&state).unwrap();
    let (state, reward, terminal, _, _) = env.step(action);
    assert!(!terminal);
    let action = gpu.select_greedy_action(&state).unwrap();
    let (_, final_reward, terminal, _, _) = env.step(action);
    assert!(terminal);
    assert_eq!(reward + final_reward, 1.);
    let input = GpuVariable::new(
        context(),
        &Tensor::from_vec(vec![1., 0., 0., 1.], &[2, 2]),
        false,
    )
    .unwrap();
    close(
        &no_grad(|| gpu.q_net().forward(&input).unwrap())
            .to_cpu()
            .unwrap()
            .to_vec(),
        &[0.9, -0.1, -1., 1.],
        0.03,
    );

    assert!(gpu
        .target_net()
        .parameters()
        .iter()
        .all(|p| p.grad().is_none()));
}
