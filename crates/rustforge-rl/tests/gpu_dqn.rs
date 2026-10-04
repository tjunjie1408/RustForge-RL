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
            _ => {
                c.use_per = true;
                c.per_beta_annealing_steps = 0;
            }
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

#[test]
#[ignore = "requires a GPU adapter"]
fn weighted_td_loss_gradients_errors_and_updates_match_cpu() {
    for double in [false, true] {
        let mut cfg = config(double, 2);
        cfg.use_per = true;
        let mut gpu = GpuDqn::new_seeded(context(), cfg, 42).unwrap();
        let mut cfg = config(double, 2);
        cfg.use_per = true;
        let mut cpu = DQN::new(cfg);
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
        // Extra inactive capacity is ignored, including its invalid values.
        let weights = Tensor::from_vec(vec![0., 0.25, 0.7, 1.5, f32::NAN, -1.], &[6, 1]);
        let resident = gpu.upload_batch_with_weights(&b, Some(&weights)).unwrap();
        for step in 1..=4 {
            let prediction = no_grad(|| {
                cpu.q_net()
                    .forward(&Variable::from_tensor(b.states.clone()))
            })
            .data()
            .gather(1, &b.actions)
            .unwrap();
            let expected_errors: Vec<f32> = (&prediction - &cpu_targets(&cpu, &b))
                .to_vec()
                .into_iter()
                .map(f32::abs)
                .collect();
            let (gl, ge) = gpu.train_device_batch_with_td_errors(&resident).unwrap();
            let (cl, ce) = cpu.train_step(&b, Some(&weights));
            close(&[gl], &[cl], 4e-6);
            close(ge.as_ref().unwrap(), &expected_errors, 3e-6);
            close(ge.as_ref().unwrap(), ce.as_ref().unwrap(), 3e-6);
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
                    4e-6,
                );
                close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 4e-6);
            }
            for (g, c) in gpu
                .target_net()
                .parameters()
                .iter()
                .zip(cpu.target_net().parameters())
            {
                close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 4e-6);
                assert!(!g.requires_grad() && g.grad().is_none());
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_prioritized_sampling_and_priority_feedback_match_cpu() {
    use rustforge_rl::buffer::PrioritizedReplayBuffer;
    let mut cfg = config(true, 3);
    cfg.use_per = true;
    let mut gpu = GpuDqn::new_seeded(context(), cfg, 7).unwrap();
    let mut cfg = config(true, 3);
    cfg.use_per = true;
    let mut cpu = DQN::new(cfg);
    for (c, g) in cpu
        .q_net()
        .parameters()
        .iter()
        .zip(gpu.q_net().parameters())
    {
        c.set_data(g.to_cpu().unwrap());
    }
    cpu.update_target();
    let mut gr = PrioritizedReplayBuffer::with_seed(16, 2, 0.6, 77);
    let mut cr = PrioritizedReplayBuffer::with_seed(16, 2, 0.6, 77);
    for i in 0..12 {
        let state = [i as f32 / 12., 0.5];
        for replay in [&mut gr, &mut cr] {
            replay.push(&state, i % 2, i as f32 / 6. - 1., &[0.5, 0.2], i % 3 == 0);
        }
    }
    let mut gb = TransitionBatch::new(4, 2);
    let mut cb = TransitionBatch::new(4, 2);
    let mut gw = Tensor::zeros(&[4, 1]);
    let mut cw = Tensor::zeros(&[4, 1]);
    let mut gi = [0; 4];
    let mut ci = [0; 4];
    let mut saw_nonuniform = false;
    for step in 0..12 {
        let beta = 0.4 + step as f32 * 0.05;
        gr.sample(4, beta, &mut gb, &mut gw, &mut gi);
        cr.sample(4, beta, &mut cb, &mut cw, &mut ci);
        assert_eq!(gi, ci);
        assert_eq!(gb.actions, cb.actions);
        close(&gw.to_vec(), &cw.to_vec(), 5e-5);
        saw_nonuniform |= gw.to_vec().iter().any(|v| *v < 0.99);
        let (gl, ge) = gpu.train_step_with_weights(&gb, Some(&gw)).unwrap();
        let (cl, ce) = cpu.train_step(&cb, Some(&cw));
        close(&[gl], &[cl], 5e-5);
        close(ge.as_ref().unwrap(), ce.as_ref().unwrap(), 5e-5);
        gr.update_priorities(&gi[..gb.size], ge.as_ref().unwrap());
        cr.update_priorities(&ci[..cb.size], ce.as_ref().unwrap());
    }
    assert!(
        saw_nonuniform,
        "priority feedback must produce nonuniform importance weights"
    );
    for (g, c) in gpu
        .q_net()
        .parameters()
        .iter()
        .zip(cpu.q_net().parameters())
    {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 5e-5);
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn invalid_weights_preserve_state_and_zero_weights_keep_unweighted_priorities() {
    use std::rc::Rc;
    let directory = tempfile::tempdir().unwrap();
    let before = directory.path().join("before.chk");
    let after = directory.path().join("after.chk");
    let mut gpu = GpuDqn::new_seeded(context(), config(true, 5), 42).unwrap();
    let b = batch();
    gpu.train_step(&b).unwrap();
    gpu.save_checkpoint(&before).unwrap();
    let params = gpu.q_net().parameters();
    let data: Vec<_> = params.iter().map(GpuVariable::data).collect();
    let gradients: Vec<_> = params.iter().map(|p| p.grad().unwrap()).collect();
    for weights in [
        Tensor::zeros(&[3, 1]),
        Tensor::zeros(&[4]),
        Tensor::zeros(&[4, 2]),
        Tensor::from_vec(vec![-1., 1., 1., 1.], &[4, 1]),
        Tensor::from_vec(vec![f32::NAN, 1., 1., 1.], &[4, 1]),
        Tensor::from_vec(vec![f32::INFINITY, 1., 1., 1.], &[4, 1]),
    ] {
        assert!(matches!(
            gpu.train_step_with_weights(&b, Some(&weights)),
            Err(GpuDqnError::InvalidBatch(_))
        ));
        for ((p, d), g) in params.iter().zip(&data).zip(&gradients) {
            assert!(Rc::ptr_eq(&p.data(), d));
            assert!(Rc::ptr_eq(&p.grad().unwrap(), g));
        }
        gpu.save_checkpoint(&after).unwrap();
        assert_eq!(
            std::fs::read(&before).unwrap(),
            std::fs::read(&after).unwrap()
        );
    }
    let mut fresh = GpuDqn::new_seeded(context(), config(true, 5), 42).unwrap();
    let original: Vec<_> = fresh
        .q_net()
        .parameters()
        .iter()
        .map(|p| p.to_cpu().unwrap().to_vec())
        .collect();
    let (loss, errors) = fresh
        .train_step_with_weights(&b, Some(&Tensor::zeros(&[4, 1])))
        .unwrap();
    assert_eq!(loss, 0.);
    assert!(errors.unwrap().iter().any(|v| *v > 0.));
    assert_eq!(fresh.train_steps(), 1);
    for (p, expected) in fresh.q_net().parameters().iter().zip(original) {
        close(&p.to_cpu().unwrap().to_vec(), &expected, 0.);
        assert!(p
            .grad_cpu()
            .unwrap()
            .unwrap()
            .to_vec()
            .iter()
            .all(|v| *v == 0.));
    }
    assert!(fresh.train_step_with_weights(&b, None).unwrap().1.is_none());
    // Finite parameters can still overflow the raw TD difference. Reject it
    // before returning priorities or modifying the optimizer/target clock.
    for p in fresh
        .q_net()
        .parameters()
        .iter()
        .chain(fresh.target_net().parameters().iter())
    {
        p.copy_data_from(
            &GpuVariable::from_device(context(), context().zeros(p.data().shape()).unwrap(), false)
                .unwrap(),
        )
        .unwrap();
    }
    fresh.q_net().parameters()[3]
        .copy_data_from(
            &GpuVariable::new(context(), &Tensor::from_vec(vec![f32::MAX; 2], &[2]), false)
                .unwrap(),
        )
        .unwrap();
    fresh.target_net().parameters()[3]
        .copy_data_from(
            &GpuVariable::new(
                context(),
                &Tensor::from_vec(vec![-f32::MAX; 2], &[2]),
                false,
            )
            .unwrap(),
        )
        .unwrap();
    let step = fresh.train_steps();
    let data = fresh.q_net().parameters()[3].data();
    assert!(matches!(
        fresh.train_step_with_weights(&b, Some(&Tensor::ones(&[4, 1]))),
        Err(GpuDqnError::NonFiniteTdErrors)
    ));
    assert_eq!(fresh.train_steps(), step);
    assert!(Rc::ptr_eq(&fresh.q_net().parameters()[3].data(), &data));
}
