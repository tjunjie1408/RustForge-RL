#![cfg(feature = "gpu")]
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    Variable,
};
use rustforge_nn::Module;
use rustforge_rl::{
    agent::{
        gpu_reinforce::{reinforce_loss, GpuReinforceError, GpuReinforceNet},
        REINFORCEConfig, REINFORCE,
    },
    buffer::RolloutBatch,
};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};
use std::{rc::Rc, sync::OnceLock};
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().expect("GPU REINFORCE tests require an adapter"))
}
fn gpu(data: &[f32], shape: &[usize], grad: bool) -> GpuVariable {
    GpuVariable::new(context(), &Tensor::from_vec(data.to_vec(), shape), grad).unwrap()
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
fn oracle(x: &[f64], actions: &[usize], adv: &[f64], baseline: bool) -> f64 {
    let mean = if baseline {
        adv.iter().sum::<f64>() / adv.len() as f64
    } else {
        0.
    };
    let cols = x.len() / actions.len();
    actions
        .iter()
        .enumerate()
        .map(|(r, a)| {
            let row = &x[r * cols..(r + 1) * cols];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let logsum = row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            -(row[*a] - max - logsum) * (adv[r] - mean)
        })
        .sum::<f64>()
        / actions.len() as f64
}
#[test]
#[ignore = "requires a GPU adapter"]
fn loss_gradients_and_detachment_match_f64_for_both_baselines() {
    let data = [0.2, -0.4, 0.9, -0.3, 0.8, 0.1, 0.6, -0.1, 0.2];
    let adv = [2., -0.7, 0.4];
    let actions = [2, 0, 1];
    let xd: Vec<f64> = data.iter().map(|v| *v as f64).collect();
    let ad: Vec<f64> = adv.iter().map(|v| *v as f64).collect();
    for baseline in [false, true] {
        let x = gpu(&data, &[3, 3], true);
        let a = gpu(&adv, &[3, 1], true);
        let indices = Rc::new(context().upload_indices(&actions, 3).unwrap());
        let objective = reinforce_loss(&x, &indices, &a, baseline).unwrap();
        approx::assert_abs_diff_eq!(
            objective.checked_loss().unwrap() as f64,
            oracle(&xd, &actions, &ad, baseline),
            epsilon = 2e-6
        );
        objective.loss.backward().unwrap();
        assert!(a.grad().is_none());
        assert!(!objective.effective_advantages.requires_grad());
        let grads = x.grad_cpu().unwrap().unwrap().to_vec();
        for i in 0..data.len() {
            let mut plus = xd.clone();
            let mut minus = xd.clone();
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            approx::assert_abs_diff_eq!(
                grads[i] as f64,
                (oracle(&plus, &actions, &ad, baseline) - oracle(&minus, &actions, &ad, baseline))
                    / 2e-5,
                epsilon = 2e-6
            );
        }
        let mean = if baseline {
            adv.iter().sum::<f32>() / 3.
        } else {
            0.
        };
        close(
            &objective.effective_advantages.to_cpu().unwrap().to_vec(),
            &adv.map(|v| v - mean),
            1e-6,
        );
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_policy_and_four_adam_updates_match_actual_cpu_reinforce() {
    for baseline in [false, true] {
        let config = REINFORCEConfig {
            obs_dim: 2,
            hidden_dim: 8,
            num_actions: 3,
            lr: 0.003,
            gamma: 0.9,
            use_baseline: baseline,
        };
        // Exercise wrapping of the second layer seed as well.
        let mut cpu = REINFORCE::new_seeded(config, u64::MAX);
        let net = GpuReinforceNet::new_seeded(context(), 2, 8, 3, u64::MAX).unwrap();
        let batch = RolloutBatch {
            states: Tensor::from_vec(vec![1., 0., 0., 1., 0.2, 0.7, -0.3, 0.8, 0.9, 0.2], &[5, 2]),
            actions: vec![0, 1, 2, 0, 1],
            advantages: Tensor::from_vec(vec![2., -1., 0.3, -0.7, 0.2], &[5, 1]),
            returns: Tensor::full(&[5, 1], f32::NAN),
            old_log_probs: Tensor::full(&[5, 1], f32::NAN),
            size: 5,
        };
        let states = GpuVariable::new(context(), &batch.states, false).unwrap();
        let advantages = GpuVariable::new(context(), &batch.advantages, true).unwrap();
        let actions = Rc::new(context().upload_indices(&batch.actions, 3).unwrap());
        let mut adam = GpuAdam::new(net.parameters(), 0.003).unwrap();
        assert_eq!(net.parameters().len(), 4);
        for (g, c) in net.parameters().iter().zip(cpu.policy_net().parameters()) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
        }
        for _ in 0..4 {
            adam.zero_grad();
            let logits = net.forward(&states).unwrap();
            let cpu_logits = cpu
                .policy_net()
                .forward(&Variable::from_tensor(batch.states.clone()));
            close(
                &logits.to_cpu().unwrap().to_vec(),
                &cpu_logits.data().to_vec(),
                2e-6,
            );
            let objective = reinforce_loss(&logits, &actions, &advantages, baseline).unwrap();
            let loss = objective.checked_loss().unwrap();
            objective.loss.backward().unwrap();
            approx::assert_abs_diff_eq!(loss, cpu.train_on_rollout(&batch), epsilon = 3e-6);
            for (g, c) in net.parameters().iter().zip(cpu.policy_net().parameters()) {
                close(
                    &g.grad_cpu().unwrap().unwrap().to_vec(),
                    &c.grad().unwrap().to_vec(),
                    5e-6,
                );
            }
            adam.step().unwrap();
            for (g, c) in net.parameters().iter().zip(cpu.policy_net().parameters()) {
                close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 5e-6);
            }
        }
        assert_eq!(adam.state().unwrap().timestep, 4);
        assert!(advantages.grad().is_none());
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn baseline_singleton_constant_advantages_and_extreme_logits() {
    for (data, cols, action, baseline, expected) in [
        (vec![0., 0.], 2, 0, false, 2. * 2f32.ln()),
        (vec![1000., -1000.], 2, 1, false, 4000.),
        (vec![1000.], 1, 0, false, 0.),
        (vec![1., -1.], 2, 1, true, 0.),
    ] {
        let x = gpu(&data, &[1, cols], true);
        let a = gpu(&[2.], &[1, 1], true);
        let actions = Rc::new(context().upload_indices(&[action], cols).unwrap());
        let objective = reinforce_loss(&x, &actions, &a, baseline).unwrap();
        approx::assert_abs_diff_eq!(objective.checked_loss().unwrap(), expected, epsilon = 1e-5);
        objective.loss.backward().unwrap();
        assert!(a.grad().is_none());
        if baseline || cols == 1 {
            close(
                &x.grad_cpu().unwrap().unwrap().to_vec(),
                &vec![0.; cols],
                0.,
            );
        }
    }
    let x = gpu(&[1., -1., -1., 1.], &[2, 2], true);
    let a = gpu(&[3., 3.], &[2, 1], true);
    let actions = Rc::new(context().upload_indices(&[0, 1], 2).unwrap());
    let objective = reinforce_loss(&x, &actions, &a, true).unwrap();
    assert_eq!(objective.checked_loss().unwrap(), 0.);
    objective.loss.backward().unwrap();
    close(&x.grad_cpu().unwrap().unwrap().to_vec(), &[0.; 4], 0.);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn malformed_nonfinite_and_foreign_inputs_are_rejected() {
    let x = gpu(&[0., 0., 0., 0.], &[2, 2], true);
    let a = gpu(&[1., 2.], &[2, 1], false);
    let actions = Rc::new(context().upload_indices(&[0, 1], 2).unwrap());
    for shape in [&[2][..], &[1, 2][..], &[2, 2][..]] {
        let bad = GpuVariable::new(context(), &Tensor::zeros(shape), false).unwrap();
        assert!(matches!(
            reinforce_loss(&x, &actions, &bad, true),
            Err(GpuReinforceError::InvalidBatch)
        ));
    }
    let empty = gpu(&[], &[0, 2], true);
    let empty_adv = gpu(&[], &[0, 1], false);
    let empty_actions = Rc::new(context().upload_indices(&[], 2).unwrap());
    assert!(matches!(
        reinforce_loss(&empty, &empty_actions, &empty_adv, false),
        Err(GpuReinforceError::InvalidBatch)
    ));
    let wrong = Rc::new(context().upload_indices(&[0], 2).unwrap());
    assert!(matches!(
        reinforce_loss(&x, &wrong, &a, false),
        Err(GpuReinforceError::InvalidBatch)
    ));
    let wrong_cols = Rc::new(context().upload_indices(&[0, 1], 3).unwrap());
    assert!(matches!(
        reinforce_loss(&x, &wrong_cols, &a, false),
        Err(GpuReinforceError::InvalidBatch)
    ));
    assert!(context().upload_indices(&[2], 2).is_err());
    let foreign = GpuContext::new().unwrap();
    let foreign_adv = GpuVariable::new(&foreign, &Tensor::ones(&[2, 1]), false).unwrap();
    assert!(matches!(
        reinforce_loss(&x, &actions, &foreign_adv, false),
        Err(GpuReinforceError::Device(GpuError::DeviceMismatch))
    ));
    let foreign_actions = Rc::new(foreign.upload_indices(&[0, 1], 2).unwrap());
    assert!(reinforce_loss(&x, &foreign_actions, &a, false).is_err());
    for baseline in [false, true] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bad = gpu(&[value, 1.], &[2, 1], false);
            assert!(matches!(
                reinforce_loss(&x, &actions, &bad, baseline)
                    .unwrap()
                    .checked_loss(),
                Err(GpuReinforceError::NonFinite)
            ));
            let bad_logits = gpu(&[value, 0., 0., 0.], &[2, 2], true);
            assert!(matches!(
                reinforce_loss(&bad_logits, &actions, &a, baseline)
                    .unwrap()
                    .checked_loss(),
                Err(GpuReinforceError::NonFinite)
            ));
        }
    }
    // Finite input can overflow the baseline reduction; checked_loss must reject it.
    let huge = gpu(&[f32::MAX, f32::MAX], &[2, 1], false);
    assert!(matches!(
        reinforce_loss(&x, &actions, &huge, true)
            .unwrap()
            .checked_loss(),
        Err(GpuReinforceError::NonFinite)
    ));
    assert!(x.grad().is_none());
}
