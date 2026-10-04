#![cfg(feature = "gpu")]
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad, Variable,
};
use rustforge_rl::{
    agent::{
        gpu_a2c::{a2c_loss, GpuA2cLossConfig, GpuA2cLossError, GpuA2cNet},
        A2CConfig, A2C,
    },
    buffer::RolloutBatch,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::{rc::Rc, sync::OnceLock};
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().expect("GPU A2C tests require an adapter"))
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
fn cpu_loss(
    x: &Variable,
    v: &Variable,
    actions: &[usize],
    adv: &Variable,
    ret: &Variable,
    c: GpuA2cLossConfig,
) -> (Variable, Variable, Variable, Variable) {
    let max = Variable::from_tensor(x.data().max_axis(1, true).unwrap());
    let shifted = x - &max;
    let exp = shifted.exp();
    let sum = exp.sum_axis(1, true);
    let logs = &shifted - &sum.log();
    let actor = -(&logs.gather(1, actions) * &adv.detach()).mean();
    let diff = v - &ret.detach();
    let value = (&diff * &diff).mean();
    let entropy = -(&(&exp / &sum) * &logs).sum_axis(1, false).mean();
    let total = &(&actor + &(&value * c.value_coef)) - &(&entropy * c.entropy_coef);
    (actor, value, entropy, total)
}
fn oracle(
    x: &[f64],
    v: &[f64],
    actions: &[usize],
    adv: &[f64],
    ret: &[f64],
    c: GpuA2cLossConfig,
) -> f64 {
    let n = actions.len();
    let columns = x.len() / n;
    let mut result = 0.;
    for row in 0..n {
        let xs = &x[row * columns..(row + 1) * columns];
        let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = xs.iter().map(|a| (a - max).exp()).sum::<f64>().ln();
        let logs: Vec<_> = xs.iter().map(|a| a - max - log_sum).collect();
        let entropy = -logs.iter().map(|l| l.exp() * l).sum::<f64>();
        result += -logs[actions[row]] * adv[row]
            + c.value_coef as f64 * (v[row] - ret[row]).powi(2)
            - c.entropy_coef as f64 * entropy;
    }
    result / n as f64
}
#[test]
#[ignore = "requires a GPU adapter"]
fn a2c_losses_and_detached_gradients_match_cpu_and_f64_finite_differences() {
    let data = [0.2, -0.4, 0.9, -0.3, 0.8, 0.1];
    let values = [0.1, -0.2];
    let adv = [2., -0.7];
    let ret = [0.8, -0.5];
    let actions = [2, 0];
    let c = GpuA2cLossConfig {
        value_coef: 0.7,
        entropy_coef: 0.13,
    };
    let x = gpu(&data, &[2, 3], true);
    let v = gpu(&values, &[2, 1], true);
    let a = gpu(&adv, &[2, 1], true);
    let r = gpu(&ret, &[2, 1], true);
    let indices = Rc::new(context().upload_indices(&actions, 3).unwrap());
    let cx = Variable::new(Tensor::from_vec(data.to_vec(), &[2, 3]), true);
    let cv = Variable::new(Tensor::from_vec(values.to_vec(), &[2, 1]), true);
    let ca = Variable::new(Tensor::from_vec(adv.to_vec(), &[2, 1]), true);
    let cr = Variable::new(Tensor::from_vec(ret.to_vec(), &[2, 1]), true);
    let loss = a2c_loss(&x, &v, &indices, &a, &r, c).unwrap();
    let m = loss.checked_metrics().unwrap();
    let (actor, value, entropy, total) = cpu_loss(&cx, &cv, &actions, &ca, &cr, c);
    close(
        &[m.actor_loss, m.value_loss, m.entropy, m.total_loss],
        &[
            actor.data().item(),
            value.data().item(),
            entropy.data().item(),
            total.data().item(),
        ],
        2e-6,
    );
    loss.total_loss.backward().unwrap();
    total.backward();
    close(
        &x.grad_cpu().unwrap().unwrap().to_vec(),
        &cx.grad().unwrap().to_vec(),
        2e-6,
    );
    close(
        &v.grad_cpu().unwrap().unwrap().to_vec(),
        &cv.grad().unwrap().to_vec(),
        2e-6,
    );
    assert!(a.grad().is_none() && r.grad().is_none() && ca.grad().is_none() && cr.grad().is_none());
    let xd: Vec<f64> = data.iter().map(|x| *x as f64).collect();
    let vd: Vec<f64> = values.iter().map(|x| *x as f64).collect();
    let ad: Vec<f64> = adv.iter().map(|x| *x as f64).collect();
    let rd: Vec<f64> = ret.iter().map(|x| *x as f64).collect();
    for (data, gradient, is_logits) in [
        (&xd, x.grad_cpu().unwrap().unwrap().to_vec(), true),
        (&vd, v.grad_cpu().unwrap().unwrap().to_vec(), false),
    ] {
        for i in 0..data.len() {
            let mut plus = data.clone();
            let mut minus = data.clone();
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            let f = |d: &[f64]| {
                if is_logits {
                    oracle(d, &vd, &actions, &ad, &rd, c)
                } else {
                    oracle(&xd, d, &actions, &ad, &rd, c)
                }
            };
            approx::assert_abs_diff_eq!(
                gradient[i] as f64,
                (f(&plus) - f(&minus)) / 2e-5,
                epsilon = 2e-6
            );
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_shared_network_gradients_and_repeated_adam_updates_match_actual_cpu_a2c() {
    let config = || A2CConfig {
        obs_dim: 2,
        num_actions: 3,
        hidden_dim: 8,
        lr: 0.003,
        c_value: 0.7,
        c_entropy: 0.13,
        ..Default::default()
    };
    let mut cpu = A2C::new_seeded(config(), 42);
    let net = GpuA2cNet::new_seeded(context(), 2, 8, 3, 42).unwrap();
    let batch = RolloutBatch {
        states: Tensor::from_vec(vec![1., 0., 0., 1., 0.2, 0.7, -0.3, 0.8, 0.9, 0.2], &[5, 2]),
        actions: vec![0, 1, 2, 0, 1],
        returns: Tensor::from_vec(vec![1., -0.2, 0.7, 0.1, -0.4], &[5, 1]),
        advantages: Tensor::from_vec(vec![2., -1., 0.3, -0.7, 0.2], &[5, 1]),
        old_log_probs: Tensor::full(&[5, 1], f32::NAN),
        size: 5,
    };
    let states = GpuVariable::new(context(), &batch.states, false).unwrap();
    let a = GpuVariable::new(context(), &batch.advantages, true).unwrap();
    let r = GpuVariable::new(context(), &batch.returns, true).unwrap();
    let indices = Rc::new(context().upload_indices(&batch.actions, 3).unwrap());
    let mut adam = GpuAdam::new(net.parameters(), 0.003).unwrap();
    for (g, c) in net.parameters().iter().zip(cpu.net().parameters()) {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
    }
    for _ in 0..4 {
        adam.zero_grad();
        let (x, v) = net.forward(&states).unwrap();
        let (cx, cv) = cpu
            .net()
            .forward(&Variable::from_tensor(batch.states.clone()));
        close(&x.to_cpu().unwrap().to_vec(), &cx.data().to_vec(), 2e-6);
        close(&v.to_cpu().unwrap().to_vec(), &cv.data().to_vec(), 2e-6);
        let loss = a2c_loss(
            &x,
            &v,
            &indices,
            &a,
            &r,
            GpuA2cLossConfig {
                value_coef: 0.7,
                entropy_coef: 0.13,
            },
        )
        .unwrap();
        let m = loss.checked_metrics().unwrap();
        loss.total_loss.backward().unwrap();
        let (total, actor, value, entropy) = cpu.train_on_rollout(&batch);
        close(
            &[m.total_loss, m.actor_loss, m.value_loss, m.entropy],
            &[total, actor, value, entropy],
            3e-6,
        );
        for (g, c) in net.parameters().iter().zip(cpu.net().parameters()) {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                5e-6,
            );
        }
        adam.step().unwrap();
        for (g, c) in net.parameters().iter().zip(cpu.net().parameters()) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 5e-6);
        }
    }
    assert_eq!(adam.state().unwrap().timestep, 4);
    assert!(a.grad().is_none() && r.grad().is_none());
}
#[test]
#[ignore = "requires a GPU adapter"]
fn a2c_uniform_single_action_and_extreme_logits_preserve_entropy_and_raw_advantages() {
    for (data, columns, action, expected_entropy) in [
        (vec![0., 0.], 2, 0, 2f32.ln()),
        (vec![1000., -1000.], 2, 1, 0.),
        (vec![1000.], 1, 0, 0.),
    ] {
        let x = gpu(&data, &[1, columns], true);
        let v = gpu(&[0.], &[1, 1], true);
        let a = gpu(&[2.], &[1, 1], false);
        let indices = Rc::new(context().upload_indices(&[action], columns).unwrap());
        let loss = a2c_loss(
            &x,
            &v,
            &indices,
            &a,
            &v,
            GpuA2cLossConfig {
                value_coef: 0.,
                entropy_coef: 0.2,
            },
        )
        .unwrap();
        let m = loss.checked_metrics().unwrap();
        approx::assert_abs_diff_eq!(m.entropy, expected_entropy, epsilon = 1e-6);
        let expected_actor = if columns == 1 {
            0.
        } else if action == 1 {
            4000.
        } else {
            2. * 2f32.ln()
        };
        approx::assert_abs_diff_eq!(m.actor_loss, expected_actor, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(m.total_loss, m.actor_loss - 0.2 * m.entropy, epsilon = 1e-6);
        loss.total_loss.backward().unwrap();
        assert!(x
            .grad_cpu()
            .unwrap()
            .unwrap()
            .to_vec()
            .iter()
            .all(|v| v.is_finite()));
        if columns == 1 {
            close(&x.grad_cpu().unwrap().unwrap().to_vec(), &[0.], 0.);
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn a2c_rejects_invalid_inputs_and_checks_frozen_forward_snapshots_with_no_grad_support() {
    let x = gpu(&[0.2, -0.1], &[1, 2], true);
    let v = gpu(&[0.1], &[1, 1], true);
    let a = gpu(&[2.], &[1, 1], true);
    let r = gpu(&[0.5], &[1, 1], true);
    let indices = Rc::new(context().upload_indices(&[0], 2).unwrap());
    let loss = a2c_loss(&x, &v, &indices, &a, &r, Default::default()).unwrap();
    let before = loss.checked_metrics().unwrap();
    for p in [&x, &v, &a, &r] {
        p.copy_data_from(&gpu(
            &vec![f32::NAN; p.data().numel()],
            p.data().shape(),
            false,
        ))
        .unwrap();
    }
    let after = loss.checked_metrics().unwrap();
    close(
        &[
            before.total_loss,
            before.actor_loss,
            before.value_loss,
            before.entropy,
        ],
        &[
            after.total_loss,
            after.actor_loss,
            after.value_loss,
            after.entropy,
        ],
        0.,
    );
    loss.total_loss.backward().unwrap();
    assert!(x
        .grad_cpu()
        .unwrap()
        .unwrap()
        .to_vec()
        .iter()
        .all(|v| v.is_finite()));
    assert!(a.grad().is_none() && r.grad().is_none());
    assert!(matches!(
        a2c_loss(&x, &v, &indices, &a, &r, Default::default())
            .unwrap()
            .checked_metrics(),
        Err(GpuA2cLossError::NonFinite)
    ));
    let x = gpu(&[0., 0.], &[1, 2], true);
    let v = gpu(&[0.], &[1, 1], true);
    let a = gpu(&[1.], &[1, 1], true);
    let frozen = no_grad(|| a2c_loss(&x, &v, &indices, &a, &v, Default::default())).unwrap();
    assert!(!frozen.total_loss.requires_grad());
    assert!(frozen.checked_metrics().is_ok());
    assert!(a2c_loss(
        &x,
        &v,
        &indices,
        &a,
        &v,
        GpuA2cLossConfig {
            value_coef: -1.,
            ..Default::default()
        }
    )
    .is_err());
    assert!(a2c_loss(
        &x,
        &gpu(&[0., 0.], &[2, 1], false),
        &indices,
        &a,
        &v,
        Default::default()
    )
    .is_err());
    assert!(a2c_loss(
        &x,
        &v,
        &indices,
        &gpu(&[1.], &[1], false),
        &v,
        Default::default()
    )
    .is_err());
    assert!(a2c_loss(
        &gpu(&[], &[0, 2], true),
        &gpu(&[], &[0, 1], false),
        &Rc::new(context().upload_indices(&[], 2).unwrap()),
        &gpu(&[], &[0, 1], false),
        &gpu(&[], &[0, 1], false),
        Default::default()
    )
    .is_err());
    assert!(a2c_loss(
        &x,
        &v,
        &Rc::new(context().upload_indices(&[0, 1], 2).unwrap()),
        &a,
        &v,
        Default::default()
    )
    .is_err());
    assert!(a2c_loss(
        &x,
        &v,
        &Rc::new(context().upload_indices(&[0], 3).unwrap()),
        &a,
        &v,
        Default::default()
    )
    .is_err());
    assert!(context().upload_indices(&[2], 2).is_err());
    let foreign = GpuContext::new().unwrap();
    let foreign_v = GpuVariable::new(&foreign, &Tensor::zeros(&[1, 1]), false).unwrap();
    assert!(a2c_loss(&x, &foreign_v, &indices, &a, &v, Default::default()).is_err());
    assert!(a2c_loss(
        &x,
        &v,
        &Rc::new(foreign.upload_indices(&[0], 2).unwrap()),
        &a,
        &v,
        Default::default()
    )
    .is_err());
    let overflow = gpu(&[1e20], &[1, 1], true);
    assert!(matches!(
        a2c_loss(
            &x,
            &overflow,
            &indices,
            &a,
            &v,
            GpuA2cLossConfig {
                value_coef: 0.,
                entropy_coef: 0.
            }
        )
        .unwrap()
        .checked_metrics(),
        Err(GpuA2cLossError::NonFinite)
    ));
}
