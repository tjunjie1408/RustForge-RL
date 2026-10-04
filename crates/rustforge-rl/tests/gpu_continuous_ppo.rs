#![cfg(feature = "gpu")]
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad,
    optimizer::adam::Adam,
    Optimizer, Variable,
};
use rustforge_rl::agent::{
    gpu_gaussian::{GpuGaussianError, GpuGaussianTransform},
    gpu_ppo::{continuous_ppo_loss, GpuContinuousPpoInputs},
    utils::{clamp_var, elementwise_min_var},
    GaussianPolicy,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::sync::OnceLock;
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().expect("continuous PPO tests require an adapter"))
}
fn gpu(v: &[f32], shape: &[usize], train: bool) -> GpuVariable {
    GpuVariable::new(context(), &Tensor::from_vec(v.to_vec(), shape), train).unwrap()
}
fn cpu(v: &[f32], shape: &[usize], train: bool) -> Variable {
    Variable::new(Tensor::from_vec(v.to_vec(), shape), train)
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
// The CPU GaussianPolicy stored-action calculation, evaluated on explicit outputs
// so independent per-row leaves can exercise full loss/Adam parity.
fn cpu_log_prob(
    mean: &Variable,
    raw: &Variable,
    actions: &Variable,
    low: &[f32],
    high: &[f32],
) -> Variable {
    let shape = mean.shape();
    let dims = shape[1];
    let batch = shape[0];
    let scale: Vec<_> = low.iter().zip(high).map(|(l, h)| (h - l) / 2.).collect();
    let bias: Vec<_> = low.iter().zip(high).map(|(l, h)| (h + l) / 2.).collect();
    let action = actions.data().to_vec();
    let clamped: Vec<_> = action
        .iter()
        .enumerate()
        .map(|(i, a)| ((a - bias[i % dims]) / scale[i % dims]).clamp(-1. + 1e-6, 1. - 1e-6))
        .collect();
    let u = cpu(
        &clamped
            .iter()
            .map(|x| (*x as f64).atanh() as f32)
            .collect::<Vec<_>>(),
        &shape,
        false,
    );
    let t = cpu(&clamped, &shape, false);
    let logstd = clamp_var(raw, -20., 2.);
    let z = &(&u - mean) / &logstd.exp();
    let normal = &(&(&z.pow(2.) * -0.5) - &logstd)
        - &cpu(
            &vec![0.5 * (2. * std::f32::consts::PI).ln(); batch * dims],
            &shape,
            false,
        );
    let jacobian = (&(&cpu(&vec![1.; batch * dims], &shape, false) - &(&t * &t))
        + &cpu(&vec![1e-6; batch * dims], &shape, false))
        .log();
    let logscale = cpu(
        &scale
            .iter()
            .map(|s| s.ln())
            .cycle()
            .take(batch * dims)
            .collect::<Vec<_>>(),
        &shape,
        false,
    );
    (&(&normal - &jacobian) - &logscale).sum_axis(1, true)
}
#[test]
#[ignore = "requires a GPU adapter"]
fn stored_action_density_and_gradients_match_actual_cpu_policy_at_bounds() {
    let low = [-2., 0.5];
    let high = [1., 4.];
    let transform = GpuGaussianTransform::new(context(), &low, &high).unwrap();
    let policy = GaussianPolicy::new(1, 2, 2, &low, &high);
    for p in policy.parameters() {
        p.set_data(Tensor::zeros(&p.shape()));
    }
    policy.parameters()[5].set_data(Tensor::from_vec(vec![0.2, -0.3], &[2]));
    policy.parameters()[7].set_data(Tensor::from_vec(vec![-0.5, 0.25], &[2]));
    let actions = [-0.8, 2., -2., 4., 1., 0.5, -20., 30.];
    let gactions = gpu(&actions, &[4, 2], true);
    let mean = gpu(&[0.2, -0.3].repeat(4), &[4, 2], true);
    let raw = gpu(&[-0.5, 0.25].repeat(4), &[4, 2], true);
    let stats = transform
        .log_prob_from_action(&mean, &raw, &gactions)
        .unwrap();
    let metrics = stats.checked_metrics().unwrap();
    let ca = cpu(&actions, &[4, 2], true);
    let cl = policy.log_prob_from_action(&cpu(&[1.; 4], &[4, 1], false), &ca);
    close(
        &stats.log_probs.to_cpu().unwrap().to_vec(),
        &cl.data().to_vec(),
        2e-4,
    );
    stats.log_probs.sum().unwrap().backward().unwrap();
    cl.sum().backward();
    let sum_rows = |x: Vec<f32>| {
        vec![
            x.iter().step_by(2).sum::<f32>(),
            x.iter().skip(1).step_by(2).sum::<f32>(),
        ]
    };
    close(
        &sum_rows(mean.grad_cpu().unwrap().unwrap().to_vec()),
        &policy.parameters()[5].grad().unwrap().to_vec(),
        3e-4,
    );
    close(
        &sum_rows(raw.grad_cpu().unwrap().unwrap().to_vec()),
        &policy.parameters()[7].grad().unwrap().to_vec(),
        5e-4,
    );
    assert!(gactions.grad().is_none() && ca.grad().is_none());
    let expected = -0.5 + 0.25 + 2. * (0.5 * (2. * std::f32::consts::PI).ln() + 0.5);
    approx::assert_abs_diff_eq!(metrics.base_entropy, expected, epsilon = 1e-5);
    // CPU clamp endpoint conventions: lower passes gradient, upper stops it.
    let raw = gpu(&[-20., 2., -21., 3.], &[2, 2], true);
    let mean = gpu(&[0.; 4], &[2, 2], true);
    let zero = gpu(&[-0.5, 2.25].repeat(2), &[2, 2], false);
    let stats = transform.log_prob_from_action(&mean, &raw, &zero).unwrap();
    stats.checked_metrics().unwrap();
    stats.base_entropy.sum().unwrap().backward().unwrap();
    close(
        &raw.grad_cpu().unwrap().unwrap().to_vec(),
        &[1., 0., 0., 0.],
        0.,
    );
}
#[test]
#[ignore = "requires a GPU adapter"]
fn reparameterized_sample_matches_f64_density_and_gradient_with_frozen_noise() {
    let low = [-2., 0.5];
    let high = [1., 4.];
    let transform = GpuGaussianTransform::new(context(), &low, &high).unwrap();
    let m = [0.2, -0.3, 1.2, -1.1];
    let s = [-0.5, 0.25, -0.7, 0.1];
    let n = [0.4, -0.8, 1.1, -0.2];
    let mean = gpu(&m, &[2, 2], true);
    let std = gpu(&s, &[2, 2], true);
    let noise = gpu(&n, &[2, 2], true);
    let sampled = transform.sample_with_noise(&mean, &std, &noise).unwrap();
    sampled.distribution.checked_metrics().unwrap();
    let reference = |m: &[f32], s: &[f32]| -> (Vec<f32>, Vec<f32>, f64) {
        let mut actions = Vec::new();
        let mut logs = vec![0f64; 2];
        for i in 0..4 {
            let scale = (high[i % 2] - low[i % 2]) as f64 / 2.;
            let bias = (high[i % 2] + low[i % 2]) as f64 / 2.;
            let z = n[i] as f64;
            let u = m[i] as f64 + z * (s[i] as f64).exp();
            let t = u.tanh();
            actions.push((t * scale + bias) as f32);
            logs[i / 2] += -0.5 * z * z
                - s[i] as f64
                - 0.5 * (2. * std::f64::consts::PI).ln()
                - (1. - t * t + 1e-6).ln()
                - scale.ln();
        }
        let value = logs.iter().sum::<f64>() + 0.1 * actions.iter().map(|x| *x as f64).sum::<f64>();
        (actions, logs.into_iter().map(|x| x as f32).collect(), value)
    };
    let (actions, logs, _) = reference(&m, &s);
    close(&sampled.actions.to_cpu().unwrap().to_vec(), &actions, 1e-5);
    close(
        &sampled.distribution.log_probs.to_cpu().unwrap().to_vec(),
        &logs,
        1e-5,
    );
    let loss = sampled
        .distribution
        .log_probs
        .sum()
        .unwrap()
        .add(&sampled.actions.sum().unwrap().scale(0.1).unwrap())
        .unwrap();
    loss.backward().unwrap();
    assert!(noise.grad().is_none());
    for i in 0..4 {
        for which in 0..2 {
            let mut mp = m;
            let mut mm = m;
            let mut sp = s;
            let mut sm = s;
            if which == 0 {
                mp[i] += 0.001;
                mm[i] -= 0.001;
            } else {
                sp[i] += 0.001;
                sm[i] -= 0.001;
            }
            let denominator = if which == 0 {
                mp[i] as f64 - mm[i] as f64
            } else {
                sp[i] as f64 - sm[i] as f64
            };
            let expected = (reference(&mp, &sp).2 - reference(&mm, &sm).2) / denominator;
            let actual = if which == 0 {
                mean.grad_cpu().unwrap().unwrap().to_vec()[i]
            } else {
                std.grad_cpu().unwrap().unwrap().to_vec()[i]
            };
            approx::assert_abs_diff_eq!(actual as f64, expected, epsilon = 8e-5);
        }
    }
    let no_grad_sample = no_grad(|| transform.sample_with_noise(&mean, &std, &noise)).unwrap();
    assert!(
        !no_grad_sample.actions.requires_grad()
            && !no_grad_sample.distribution.log_probs.has_grad_fn()
    );
}
#[test]
#[ignore = "requires a GPU adapter"]
fn continuous_ppo_losses_gradients_and_separate_adam_updates_match_cpu() {
    let shape = [6, 2];
    let low = [-2., 0.5];
    let high = [1., 4.];
    let transform = GpuGaussianTransform::new(context(), &low, &high).unwrap();
    let m: Vec<_> = (0..12).map(|i| i as f32 * 0.03 - 0.2).collect();
    let s = vec![-0.4; 12];
    let act = [
        -0.8, 2., -0.4, 2.3, -0.1, 1.9, -0.7, 2.6, 0.2, 2.8, -1., 1.7,
    ];
    let ret = [0.7, -0.4, 1.1, 0.1, -0.8, 0.3];
    let adv = [2., -2., 1., -1., 0., 0.5];
    let val = [0.1, 0.2, -0.4, 0.8, -0.6, 0.3];
    let gm = gpu(&m, &shape, true);
    let gs = gpu(&s, &shape, true);
    let gv = gpu(&val, &[6, 1], true);
    let ca = cpu(&act, &shape, true);
    let cm = cpu(&m, &shape, true);
    let cs = cpu(&s, &shape, true);
    let cv = cpu(&val, &[6, 1], true);
    let initial = cpu_log_prob(&cm, &cs, &ca, &low, &high).data().to_vec();
    let old: Vec<_> = initial
        .iter()
        .zip([0.7f32, 1., 1.4, 0.6, 1.3, 1.])
        .map(|(x, r)| x - r.ln())
        .collect();
    let ga = gpu(&act, &shape, true);
    let go = gpu(&old, &[6, 1], true);
    let gd = gpu(&adv, &[6, 1], true);
    let gr = gpu(&ret, &[6, 1], true);
    let co = cpu(&old, &[6, 1], true);
    let cd = cpu(&adv, &[6, 1], true);
    let cr = cpu(&ret, &[6, 1], true);
    let mut gactor = GpuAdam::new(vec![gm.clone(), gs.clone()], 0.01).unwrap();
    let mut gcritic = GpuAdam::new(vec![gv.clone()], 0.01).unwrap();
    let mut actor = Adam::new(vec![cm.clone(), cs.clone()], 0.01);
    let mut critic = Adam::new(vec![cv.clone()], 0.01);
    for _ in 0..4 {
        let objective = continuous_ppo_loss(
            GpuContinuousPpoInputs {
                mean: &gm,
                raw_log_std: &gs,
                values: &gv,
                actions: &ga,
                old_log_probs: &go,
                advantages: &gd,
                returns: &gr,
            },
            &transform,
            0.2,
        )
        .unwrap();
        let metrics = objective.checked_metrics().unwrap();
        let new = cpu_log_prob(&cm, &cs, &ca, &low, &high);
        let ratio = (&new - &co.detach()).exp();
        let first = &ratio * &cd.detach();
        let second = &clamp_var(&ratio, 0.8, 1.2) * &cd.detach();
        let policy = &elementwise_min_var(&first, &second).mean() * -1.;
        let diff = &cv - &cr.detach();
        let value = (&diff * &diff).mean();
        close(
            &[metrics.policy_loss, metrics.value_loss],
            &[policy.data().item(), value.data().item()],
            3e-5,
        );
        gactor.zero_grad();
        gcritic.zero_grad();
        actor.zero_grad();
        critic.zero_grad();
        objective.policy_loss.backward().unwrap();
        objective.value_loss.backward().unwrap();
        policy.backward();
        value.backward();
        for (g, c) in [(&gm, &cm), (&gs, &cs), (&gv, &cv)] {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                4e-5,
            );
        }
        for reference in [&ga, &go, &gd, &gr] {
            assert!(reference.grad().is_none());
        }
        for reference in [&ca, &co, &cd, &cr] {
            assert!(reference.grad().is_none());
        }
        gactor.step().unwrap();
        gcritic.step().unwrap();
        actor.step();
        critic.step();
        for (g, c) in [(&gm, &cm), (&gs, &cs), (&gv, &cv)] {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 3e-5);
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn continuous_objective_rejects_invalid_shapes_owners_inputs_and_hidden_overflow() {
    let transform = GpuGaussianTransform::new(context(), &[-1.], &[1.]).unwrap();
    let mean = gpu(&[0.], &[1, 1], true);
    let std = gpu(&[0.], &[1, 1], true);
    let actions = gpu(&[0.], &[1, 1], true);
    let values = gpu(&[0.], &[1, 1], true);
    let old = gpu(&[-1.], &[1, 1], true);
    let adv = gpu(&[-1.], &[1, 1], true);
    let returns = gpu(&[1.], &[1, 1], true);
    let objective = || {
        continuous_ppo_loss(
            GpuContinuousPpoInputs {
                mean: &mean,
                raw_log_std: &std,
                actions: &actions,
                values: &values,
                old_log_probs: &old,
                advantages: &adv,
                returns: &returns,
            },
            &transform,
            0.2,
        )
        .unwrap()
    };
    assert!(objective().checked_metrics().is_ok());
    for target in [&mean, &std, &actions, &old, &adv, &returns] {
        for bad in [f32::NAN, f32::INFINITY] {
            target.copy_data_from(&gpu(&[bad], &[1, 1], false)).unwrap();
            assert!(objective().checked_metrics().is_err());
            target
                .copy_data_from(&gpu(
                    &[if std::ptr::eq(target, &old) { -1. } else { 0. }],
                    &[1, 1],
                    false,
                ))
                .unwrap();
        }
    }
    old.copy_data_from(&gpu(&[-1000.], &[1, 1], false)).unwrap();
    adv.copy_data_from(&gpu(&[1.], &[1, 1], false)).unwrap();
    let loss = objective();
    assert!(loss.policy_loss.to_cpu().unwrap().item().is_finite());
    assert!(loss.checked_metrics().is_err());
    assert!(mean.grad().is_none());
    let wrong = gpu(&[0., 0.], &[1, 2], false);
    assert!(matches!(
        transform.log_prob_from_action(&mean, &std, &wrong),
        Err(GpuGaussianError::InvalidShape)
    ));
    let other = GpuContext::new().unwrap();
    let foreign = GpuVariable::new(&other, &Tensor::zeros(&[1, 1]), false).unwrap();
    assert!(transform.sample_with_noise(&mean, &std, &foreign).is_err());
    let empty = gpu(&[], &[0, 1], false);
    let stats = transform
        .log_prob_from_action(&empty, &empty, &empty)
        .unwrap();
    assert_eq!(stats.log_probs.data().shape(), [0, 1]);
    assert_eq!(stats.checked_metrics().unwrap().mean_log_prob, 0.);
    assert!(continuous_ppo_loss(
        GpuContinuousPpoInputs {
            mean: &empty,
            raw_log_std: &empty,
            actions: &empty,
            values: &empty,
            old_log_probs: &empty,
            advantages: &empty,
            returns: &empty
        },
        &transform,
        0.2
    )
    .is_err());
    assert!(continuous_ppo_loss(
        GpuContinuousPpoInputs {
            mean: &mean,
            raw_log_std: &std,
            actions: &actions,
            values: &wrong,
            old_log_probs: &old,
            advantages: &adv,
            returns: &returns
        },
        &transform,
        0.2
    )
    .is_err());
    for (low, high) in [
        (vec![], vec![]),
        (vec![0.], vec![0.]),
        (vec![1.], vec![-1.]),
        (vec![0.], vec![f32::NAN]),
        (vec![-f32::MAX], vec![f32::MAX]),
    ] {
        assert!(GpuGaussianTransform::new(context(), &low, &high).is_err());
    }
}
