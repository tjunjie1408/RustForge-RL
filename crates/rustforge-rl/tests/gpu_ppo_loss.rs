#![cfg(feature = "gpu")]
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    Optimizer, Variable,
};
use rustforge_rl::agent::{
    gpu_ppo::{categorical_policy_loss, discrete_ppo_loss, GpuPpoLossConfig, GpuPpoLossError},
    utils::{clamp_var, elementwise_min_var},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::{rc::Rc, sync::OnceLock};
fn context() -> &'static GpuContext {
    static CONTEXT: OnceLock<GpuContext> = OnceLock::new();
    CONTEXT.get_or_init(|| GpuContext::new().expect("GPU PPO tests require an adapter"))
}
fn gpu(data: &[f32], shape: &[usize], grad: bool) -> GpuVariable {
    GpuVariable::new(context(), &Tensor::from_vec(data.to_vec(), shape), grad).unwrap()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = tolerance);
    }
}
fn cpu_loss(
    logits: &Variable,
    values: &Variable,
    actions: &[usize],
    old: &Variable,
    adv: &Variable,
    returns: &Variable,
) -> (Variable, Variable, Variable, Variable) {
    let max = Variable::from_tensor(logits.data().max_axis(1, true).unwrap());
    let shifted = logits - &max;
    let exp = shifted.exp();
    let logs = &shifted - &exp.sum_axis(1, true).log();
    let ratio = (&logs.gather(1, actions) - &old.detach()).exp();
    let first = &ratio * &adv.detach();
    let second = &clamp_var(&ratio, 0.8, 1.2) * &adv.detach();
    let policy = &elementwise_min_var(&first, &second).mean() * -1.;
    let entropy = &(&logs.exp() * &logs).sum() * (-1. / logits.shape()[0] as f32);
    let diff = values - &returns.detach();
    let value = (&diff * &diff).mean();
    let total = &(&policy + &(&value * 0.5)) - &(&entropy * 0.01);
    (policy, value, entropy, total)
}
#[test]
#[ignore = "requires a GPU adapter"]
fn complete_discrete_ppo_losses_gradients_and_adam_updates_match_cpu() {
    let n = 6;
    let data: Vec<f32> = (0..n * 3).map(|i| (i as f32 - 7.) * 0.11).collect();
    let value_data = [0.1, 0.2, -0.4, 0.8, -0.6, 0.3];
    let actions = [0, 1, 2, 0, 1, 2];
    let adv = [2., -2., 1., -1., 0., 0.5];
    let returns = [0.7, -0.4, 1.1, 0.1, -0.8, 0.3];
    let x = gpu(&data, &[n, 3], true);
    let v = gpu(&value_data, &[n, 1], true);
    let current = x
        .log_softmax()
        .unwrap()
        .gather_actions(&Rc::new(context().upload_indices(&actions, 3).unwrap()))
        .unwrap()
        .to_cpu()
        .unwrap()
        .to_vec();
    let desired = [1.5f32, 0.6, 0.9, 1.1, 1.4, 1.0];
    let old: Vec<f32> = current
        .iter()
        .zip(desired)
        .map(|(p, r)| p - r.ln())
        .collect();
    let g_old = gpu(&old, &[n, 1], true);
    let g_adv = gpu(&adv, &[n, 1], true);
    let g_return = gpu(&returns, &[n, 1], true);
    let indices = Rc::new(context().upload_indices(&actions, 3).unwrap());
    let cx = Variable::new(Tensor::from_vec(data, &[n, 3]), true);
    let cv = Variable::new(Tensor::from_vec(value_data.to_vec(), &[n, 1]), true);
    let cold = Variable::new(Tensor::from_vec(old, &[n, 1]), true);
    let cadv = Variable::new(Tensor::from_vec(adv.to_vec(), &[n, 1]), true);
    let cret = Variable::new(Tensor::from_vec(returns.to_vec(), &[n, 1]), true);
    let mut optimizer = GpuAdam::new(vec![x.clone(), v.clone()], 0.001).unwrap();
    let mut cpu_optimizer = Adam::new(vec![cx.clone(), cv.clone()], 0.001);
    for _ in 0..4 {
        optimizer.zero_grad();
        cpu_optimizer.zero_grad();
        let loss = discrete_ppo_loss(
            &x,
            &v,
            &indices,
            &g_old,
            &g_adv,
            &g_return,
            GpuPpoLossConfig::default(),
        )
        .unwrap();
        let metrics = loss.checked_metrics().unwrap();
        let (p, val, ent, total) = cpu_loss(&cx, &cv, &actions, &cold, &cadv, &cret);
        close(
            &[
                metrics.policy_loss,
                metrics.value_loss,
                metrics.entropy,
                metrics.total_loss,
            ],
            &[
                p.data().item(),
                val.data().item(),
                ent.data().item(),
                total.data().item(),
            ],
            3e-6,
        );
        loss.total_loss.backward().unwrap();
        total.backward();
        for (g, c) in [(&x, &cx), (&v, &cv)] {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                3e-6,
            );
        }
        assert!(g_old.grad().is_none() && g_adv.grad().is_none() && g_return.grad().is_none());
        assert!(cold.grad().is_none() && cadv.grad().is_none() && cret.grad().is_none());
        optimizer.step().unwrap();
        cpu_optimizer.step();
        close(&x.to_cpu().unwrap().to_vec(), &cx.data().to_vec(), 5e-6);
        close(&v.to_cpu().unwrap().to_vec(), &cv.data().to_vec(), 5e-6);
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn clipping_boundaries_and_minimum_ties_match_cpu_gradient_conventions() {
    let data = [0.7, 0.8, 1., 1.2, 1.3];
    let x = gpu(&data, &[5], true);
    let cx = Variable::new(Tensor::from_vec(data.to_vec(), &[5]), true);
    let gx = x.clamp(0.8, 1.2).unwrap();
    let cpu = clamp_var(&cx, 0.8, 1.2);
    close(&gx.to_cpu().unwrap().to_vec(), &cpu.data().to_vec(), 1e-7);
    gx.sum().unwrap().backward().unwrap();
    cpu.sum().backward();
    close(
        &x.grad_cpu().unwrap().unwrap().to_vec(),
        &cx.grad().unwrap().to_vec(),
        0.,
    );
    for (lower, upper) in [(2., 1.), (f32::NAN, 1.), (0., f32::INFINITY)] {
        assert!(x.clamp(lower, upper).is_err());
    }
    let a = gpu(&[1., 2., 3.], &[3], true);
    let b = gpu(&[1., 3., 2.], &[3], true);
    a.minimum(&b).unwrap().sum().unwrap().backward().unwrap();
    close(&a.grad_cpu().unwrap().unwrap().to_vec(), &[0., 1., 0.], 0.);
    close(&b.grad_cpu().unwrap().unwrap().to_vec(), &[1., 0., 1.], 0.);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn objective_validates_batch_owners_config_and_nonfinite_metrics() {
    let x = gpu(&[1000., -1000., 999.], &[1, 3], true);
    let v = gpu(&[0.], &[1, 1], true);
    let a = Rc::new(context().upload_indices(&[0], 3).unwrap());
    let old = gpu(&[-0.4], &[1, 1], false);
    let adv = gpu(&[1.], &[1, 1], false);
    let loss = discrete_ppo_loss(&x, &v, &a, &old, &adv, &v, GpuPpoLossConfig::default()).unwrap();
    assert!(loss.checked_metrics().unwrap().entropy.is_finite());
    for eps in [-0.1, 1., f32::NAN] {
        assert!(matches!(
            categorical_policy_loss(&x, &a, &old, &adv, eps),
            Err(GpuPpoLossError::InvalidConfig)
        ));
    }
    assert!(categorical_policy_loss(&x, &a, &old, &gpu(&[1., 2.], &[2, 1], false), 0.2).is_err());
    assert!(categorical_policy_loss(
        &gpu(&[], &[0, 3], false),
        &Rc::new(context().upload_indices(&[], 3).unwrap()),
        &gpu(&[], &[0, 1], false),
        &gpu(&[], &[0, 1], false),
        0.2
    )
    .is_err());
    let other = GpuContext::new().unwrap();
    let foreign = GpuVariable::new(&other, &Tensor::zeros(&[1, 1]), false).unwrap();
    assert!(categorical_policy_loss(&x, &a, &foreign, &adv, 0.2).is_err());
    let foreign_actions = Rc::new(other.upload_indices(&[0], 3).unwrap());
    assert!(categorical_policy_loss(&x, &foreign_actions, &old, &adv, 0.2).is_err());
    let invalid_old = gpu(&[-1000.], &[1, 1], false);
    let overflow = discrete_ppo_loss(
        &x,
        &v,
        &a,
        &invalid_old,
        &adv,
        &v,
        GpuPpoLossConfig::default(),
    )
    .unwrap();
    assert!(matches!(
        overflow.checked_metrics(),
        Err(GpuPpoLossError::NonFinite)
    ));
    assert!(x.grad().is_none() && v.grad().is_none());
}
