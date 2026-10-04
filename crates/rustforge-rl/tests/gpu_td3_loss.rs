#![cfg(feature = "gpu")]
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    optimizer::adam::Adam,
    Optimizer, Variable,
};
use rustforge_nn::{
    gpu::{GpuLinear, GpuModule},
    mse_loss, Linear, Module,
};
use rustforge_rl::agent::{
    clamp_var, elementwise_min_var,
    gpu_td3::{
        td3_actor_loss, td3_critic_loss, GpuTd3ActionTransform, GpuTd3Error, GpuTd3LossConfig,
        GpuTd3SmoothingConfig,
    },
};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};
use std::sync::OnceLock;
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().expect("GPU TD3 tests require an adapter"))
}
fn gpu(data: &[f32], shape: &[usize], grad: bool) -> GpuVariable {
    GpuVariable::new(context(), &Tensor::from_vec(data.to_vec(), shape), grad).unwrap()
}
fn cpu(data: &[f32], shape: &[usize], grad: bool) -> Variable {
    Variable::new(Tensor::from_vec(data.to_vec(), shape), grad)
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
fn cpu_target(t1: &Variable, t2: &Variable, r: &Variable, d: &Variable, gamma: f32) -> Variable {
    let ones = Variable::from_tensor(Tensor::ones(d.data().shape()));
    let minimum = elementwise_min_var(&t1.detach(), &t2.detach());
    (&r.detach() + &(&(&(&ones - &d.detach()) * &minimum) * gamma)).detach()
}
#[test]
#[ignore = "requires a GPU adapter"]
fn twin_targets_losses_detachment_and_gradients_match_cpu_and_f64() {
    let first = [0.1, -0.4, 0.7, -1.];
    let second = [-0.2, 0.3, -0.1, 0.5];
    let t1 = [2., 3., 1., -2.];
    let t2 = [1., 2., 1., -3.];
    let r = [0.5, -1., 0.2, 2.];
    let d = [0., 1., 0.25, 0.];
    for gamma in [0., 0.9, 1.] {
        let q1 = gpu(&first, &[4, 1], true);
        let q2 = gpu(&second, &[4, 1], true);
        let gt1 = gpu(&t1, &[4, 1], true);
        let gt2 = gpu(&t2, &[4, 1], true);
        let gr = gpu(&r, &[4, 1], true);
        let gd = gpu(&d, &[4, 1], true);
        let c1 = cpu(&first, &[4, 1], true);
        let c2 = cpu(&second, &[4, 1], true);
        let ct1 = cpu(&t1, &[4, 1], true);
        let ct2 = cpu(&t2, &[4, 1], true);
        let cr = cpu(&r, &[4, 1], true);
        let cd = cpu(&d, &[4, 1], true);
        let target = cpu_target(&ct1, &ct2, &cr, &cd, gamma);
        let l1 = mse_loss(&c1, &target);
        let l2 = mse_loss(&c2, &target);
        let total = &l1 + &l2;
        let objective =
            td3_critic_loss(&q1, &q2, &gt1, &gt2, &gr, &gd, GpuTd3LossConfig { gamma }).unwrap();
        let metrics = objective.checked_metrics().unwrap();
        close(
            &objective.target_values.to_cpu().unwrap().to_vec(),
            &target.data().to_vec(),
            1e-6,
        );
        close(
            &[
                metrics.critic1_loss,
                metrics.critic2_loss,
                metrics.total_loss,
            ],
            &[l1.data().item(), l2.data().item(), total.data().item()],
            2e-6,
        );
        objective.total_loss.backward().unwrap();
        total.backward();
        close(
            &q1.grad_cpu().unwrap().unwrap().to_vec(),
            &c1.grad().unwrap().to_vec(),
            1e-6,
        );
        close(
            &q2.grad_cpu().unwrap().unwrap().to_vec(),
            &c2.grad().unwrap().to_vec(),
            1e-6,
        );
        assert!([&gt1, &gt2, &gr, &gd].iter().all(|v| v.grad().is_none()));
        assert!([&ct1, &ct2, &cr, &cd].iter().all(|v| v.grad().is_none()));
        assert!(!objective.target_values.requires_grad());
        let targets: Vec<f64> = (0..4)
            .map(|i| r[i] as f64 + gamma as f64 * (1. - d[i] as f64) * (t1[i].min(t2[i]) as f64))
            .collect();
        let oracle = |x: &[f64], y: &[f64]| {
            x.iter()
                .zip(y)
                .zip(&targets)
                .map(|((x, y), t)| (x - t).powi(2) + (y - t).powi(2))
                .sum::<f64>()
                / 4.
        };
        let x: Vec<f64> = first.iter().map(|v| *v as f64).collect();
        let y: Vec<f64> = second.iter().map(|v| *v as f64).collect();
        approx::assert_abs_diff_eq!(metrics.total_loss as f64, oracle(&x, &y), epsilon = 2e-6);
        for (which, gradient) in [
            (true, q1.grad_cpu().unwrap().unwrap().to_vec()),
            (false, q2.grad_cpu().unwrap().unwrap().to_vec()),
        ] {
            for i in 0..4 {
                let mut plus = if which { x.clone() } else { y.clone() };
                let mut minus = plus.clone();
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                let evaluate = |v: &[f64]| if which { oracle(v, &y) } else { oracle(&x, v) };
                approx::assert_abs_diff_eq!(
                    gradient[i] as f64,
                    (evaluate(&plus) - evaluate(&minus)) / 2e-5,
                    epsilon = 2e-6
                );
            }
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn twin_critic_four_adam_updates_match_cpu_objective() {
    let q1 = GpuLinear::new_seeded(context(), 2, 1, 42).unwrap();
    let q2 = GpuLinear::new_seeded(context(), 2, 1, 43).unwrap();
    let c1 = Linear::new_seeded(2, 1, 42);
    let c2 = Linear::new_seeded(2, 1, 43);
    let input = Tensor::from_vec(vec![1., 0., 0., 1., 0.2, 0.7], &[3, 2]);
    let x = GpuVariable::new(context(), &input, false).unwrap();
    let cx = Variable::from_tensor(input);
    let t1 = gpu(&[1., 2., 3.], &[3, 1], true);
    let t2 = gpu(&[2., 1., 2.], &[3, 1], true);
    let r = gpu(&[0.3, -0.2, 0.5], &[3, 1], false);
    let d = gpu(&[0., 1., 0.], &[3, 1], false);
    let target = cpu_target(
        &cpu(&[1., 2., 3.], &[3, 1], false),
        &cpu(&[2., 1., 2.], &[3, 1], false),
        &cpu(&[0.3, -0.2, 0.5], &[3, 1], false),
        &cpu(&[0., 1., 0.], &[3, 1], false),
        0.9,
    );
    let mut gp = q1.parameters();
    gp.extend(q2.parameters());
    let mut cp = c1.parameters();
    cp.extend(c2.parameters());
    let mut ga = GpuAdam::new(gp.clone(), 0.003).unwrap();
    let mut ca = Adam::new(cp.clone(), 0.003);
    for (g, c) in gp.iter().zip(&cp) {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
    }
    for _ in 0..4 {
        ga.zero_grad();
        ca.zero_grad();
        let objective = td3_critic_loss(
            &q1.forward(&x).unwrap(),
            &q2.forward(&x).unwrap(),
            &t1,
            &t2,
            &r,
            &d,
            GpuTd3LossConfig { gamma: 0.9 },
        )
        .unwrap();
        let loss = &mse_loss(&c1.forward(&cx), &target) + &mse_loss(&c2.forward(&cx), &target);
        close(
            &[objective.checked_metrics().unwrap().total_loss],
            &[loss.data().item()],
            3e-6,
        );
        objective.total_loss.backward().unwrap();
        loss.backward();
        for (g, c) in gp.iter().zip(&cp) {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                3e-6,
            );
        }
        ga.step().unwrap();
        ca.step();
        for (g, c) in gp.iter().zip(&cp) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 5e-6);
        }
    }
    assert!(t1.grad().is_none() && t2.grad().is_none());
    assert_eq!(ga.state().unwrap().timestep, 4);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn actor_gradients_through_tanh_scaling_and_frozen_critic_match_cpu_updates() {
    let actor = GpuLinear::new_seeded(context(), 2, 2, 42).unwrap();
    let cpu_actor = Linear::new_seeded(2, 2, 42);
    let transform = GpuTd3ActionTransform::new(context(), &[-2., 1.], &[4., 5.]).unwrap();
    let data = [1., 0., 0., 1., 0.2, 0.7];
    let x = gpu(&data, &[3, 2], false);
    let cx = cpu(&data, &[3, 2], false);
    let weight = gpu(&[0.7, -0.3], &[2, 1], false);
    let cw = cpu(&[0.7, -0.3], &[2, 1], false);
    let scale = cpu(&[3., 2., 3., 2., 3., 2.], &[3, 2], false);
    let bias = cpu(&[1., 3., 1., 3., 1., 3.], &[3, 2], false);
    let mut ga = GpuAdam::new(actor.parameters(), 0.003).unwrap();
    let mut ca = Adam::new(cpu_actor.parameters(), 0.003);
    for _ in 0..4 {
        ga.zero_grad();
        ca.zero_grad();
        let raw = actor.forward(&x).unwrap().tanh().unwrap();
        let actions = transform.scale_actor_actions(&raw).unwrap();
        actions.checked().unwrap();
        let q = actions.actions.matmul(&weight).unwrap();
        let objective = td3_actor_loss(&q).unwrap();
        let c_raw = cpu_actor.forward(&cx).tanh_();
        let c_actions = &(&c_raw * &scale) + &bias;
        let c_q = c_actions.matmul(&cw);
        let loss = -c_q.mean();
        close(
            &actions.actions.to_cpu().unwrap().to_vec(),
            &c_actions.data().to_vec(),
            2e-6,
        );
        close(
            &[objective.checked_loss().unwrap()],
            &[loss.data().item()],
            2e-6,
        );
        objective.loss.backward().unwrap();
        loss.backward();
        for (g, c) in actor.parameters().iter().zip(cpu_actor.parameters()) {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                3e-6,
            );
        }
        ga.step().unwrap();
        ca.step();
        for (g, c) in actor.parameters().iter().zip(cpu_actor.parameters()) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 5e-6);
        }
    }
    assert!(weight.grad().is_none() && cw.grad().is_none());
    assert_eq!(ga.state().unwrap().timestep, 4);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn supplied_noise_is_clipped_before_normalized_clamp_and_scaling_without_gradients() {
    let transform = GpuTd3ActionTransform::new(context(), &[-2., 1.], &[4., 5.]).unwrap();
    let raw_data = [0.9, -0.9, -0.9, 0.9, 0.2, -0.2];
    let noise_data = [3., -3., -3., 3., -0.5, 0.5];
    let raw = gpu(&raw_data, &[3, 2], true);
    let noise = gpu(&noise_data, &[3, 2], true);
    for config in [
        GpuTd3SmoothingConfig {
            noise_std: 0.2,
            noise_clip: 0.3,
        },
        GpuTd3SmoothingConfig {
            noise_std: 0.,
            noise_clip: 0.3,
        },
        GpuTd3SmoothingConfig {
            noise_std: 0.2,
            noise_clip: 0.,
        },
    ] {
        let output = transform
            .smooth_target_actions(&raw, &noise, config)
            .unwrap();
        output.checked().unwrap();
        let cr = cpu(&raw_data, &[3, 2], false);
        let cn = cpu(&noise_data, &[3, 2], false);
        let clipped = clamp_var(
            &(&cn * config.noise_std),
            -config.noise_clip,
            config.noise_clip,
        );
        let normalized = clamp_var(&(&cr + &clipped), -1., 1.);
        let scaled = &(&normalized * &cpu(&[3., 2., 3., 2., 3., 2.], &[3, 2], false))
            + &cpu(&[1., 3., 1., 3., 1., 3.], &[3, 2], false);
        close(
            &output.normalized_actions.to_cpu().unwrap().to_vec(),
            &normalized.data().to_vec(),
            1e-6,
        );
        close(
            &output.actions.to_cpu().unwrap().to_vec(),
            &scaled.data().to_vec(),
            1e-6,
        );
        for (i, v) in output.actions.to_cpu().unwrap().to_vec().iter().enumerate() {
            let n = (raw_data[i] as f64
                + (noise_data[i] as f64 * config.noise_std as f64)
                    .clamp(-config.noise_clip as f64, config.noise_clip as f64))
            .clamp(-1., 1.);
            let expected = n * if i % 2 == 0 { 3. } else { 2. } + if i % 2 == 0 { 1. } else { 3. };
            approx::assert_abs_diff_eq!(*v as f64, expected, epsilon = 1e-6);
        }
        assert!(
            !output.actions.requires_grad()
                && !output.actions.has_grad_fn()
                && !output.normalized_actions.requires_grad()
        );
    }
    assert!(raw.grad().is_none() && noise.grad().is_none());
}
#[test]
#[ignore = "requires a GPU adapter"]
fn malformed_masks_shapes_contexts_and_nonfinite_intermediates_are_rejected() {
    let q = gpu(&[0., 0.], &[2, 1], true);
    let r = gpu(&[1., 1.], &[2, 1], false);
    let d = gpu(&[0., 1.], &[2, 1], false);
    for shape in [&[2][..], &[1, 2][..], &[0, 1][..], &[2, 2][..]] {
        let bad = GpuVariable::new(context(), &Tensor::zeros(shape), false).unwrap();
        assert!(matches!(
            td3_critic_loss(&q, &bad, &q, &q, &r, &d, Default::default()),
            Err(GpuTd3Error::InvalidBatch)
        ));
        assert!(td3_actor_loss(&bad).is_err());
    }
    for value in [-0.1, 1.1] {
        let invalid = gpu(&[value, 0.], &[2, 1], false);
        assert!(matches!(
            td3_critic_loss(&q, &q, &q, &q, &r, &invalid, Default::default())
                .unwrap()
                .checked_metrics(),
            Err(GpuTd3Error::InvalidBatch)
        ));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let invalid = gpu(&[value, 0.], &[2, 1], true);
        assert!(matches!(
            td3_critic_loss(&q, &q, &invalid, &q, &r, &d, Default::default())
                .unwrap()
                .checked_metrics(),
            Err(GpuTd3Error::NonFinite)
        ));
        assert!(matches!(
            td3_actor_loss(&invalid).unwrap().checked_loss(),
            Err(GpuTd3Error::NonFinite)
        ));
    }
    let foreign = GpuContext::new().unwrap();
    let bad = GpuVariable::new(&foreign, &Tensor::zeros(&[2, 1]), false).unwrap();
    assert!(matches!(
        td3_critic_loss(&q, &bad, &q, &q, &r, &d, Default::default()),
        Err(GpuTd3Error::Device(GpuError::DeviceMismatch))
    ));
    let transform = GpuTd3ActionTransform::new(context(), &[-1.], &[1.]).unwrap();
    assert!(transform.scale_actor_actions(&bad).is_err());
    let wrong = gpu(&[0., 0.], &[1, 2], false);
    assert!(transform.scale_actor_actions(&wrong).is_err());
    assert!(transform
        .smooth_target_actions(&q, &wrong, Default::default())
        .is_err());
    // Clipping would hide nonfinite source noise or overflow in std multiplication.
    for (noise, std) in [(f32::INFINITY, 0.2), (f32::MAX, f32::MAX)] {
        let noise = gpu(&[noise, 0.], &[2, 1], false);
        let output = transform
            .smooth_target_actions(
                &q,
                &noise,
                GpuTd3SmoothingConfig {
                    noise_std: std,
                    noise_clip: 0.5,
                },
            )
            .unwrap();
        assert!(matches!(output.checked(), Err(GpuTd3Error::NonFinite)));
    }
    let masked_invalid = gpu(&[0., f32::NAN], &[2, 1], false);
    assert!(matches!(
        td3_critic_loss(&q, &q, &masked_invalid, &q, &r, &d, Default::default())
            .unwrap()
            .checked_metrics(),
        Err(GpuTd3Error::NonFinite)
    ));
    let huge_raw = gpu(&[f32::MAX, 0.], &[2, 1], false);
    let huge_noise = gpu(&[f32::MAX, 0.], &[2, 1], false);
    assert!(matches!(
        transform
            .smooth_target_actions(
                &huge_raw,
                &huge_noise,
                GpuTd3SmoothingConfig {
                    noise_std: 1.,
                    noise_clip: f32::MAX
                }
            )
            .unwrap()
            .checked(),
        Err(GpuTd3Error::NonFinite)
    ));
    let huge = gpu(&[f32::MAX, f32::MAX], &[2, 1], false);
    assert!(
        td3_critic_loss(&q, &q, &huge, &huge, &r, &d, GpuTd3LossConfig { gamma: 1. })
            .unwrap()
            .checked_metrics()
            .is_err()
    );
    assert!(matches!(
        td3_actor_loss(&huge).unwrap().checked_loss(),
        Err(GpuTd3Error::NonFinite)
    ));
    assert!(q.grad().is_none());
}
