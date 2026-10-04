#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad,
    optimizer::adam::Adam,
    Optimizer, Variable,
};
use rustforge_nn::{
    gpu::{GpuLinear, GpuModule},
    mse_loss, Linear, Module,
};
use rustforge_rl::agent::{
    gpu_ppo::GpuGaussianPolicyNet,
    gpu_sac::{
        sac_actor_loss, sac_critic_loss, sac_temperature_loss, GpuSacActionTransform,
        GpuSacCriticInputs, GpuSacError, GpuSacLossConfig,
    },
    utils::elementwise_min_var,
    GaussianPolicy,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::sync::OnceLock;
fn context() -> &'static GpuContext {
    static C: OnceLock<GpuContext> = OnceLock::new();
    C.get_or_init(|| GpuContext::new().expect("GPU SAC tests require an adapter"))
}
fn gpu(v: &[f32], shape: &[usize], grad: bool) -> GpuVariable {
    GpuVariable::new(context(), &Tensor::from_vec(v.to_vec(), shape), grad).unwrap()
}
fn cpu(v: &[f32], shape: &[usize], grad: bool) -> Variable {
    Variable::new(Tensor::from_vec(v.to_vec(), shape), grad)
}
fn close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = eps);
    }
}
fn target(
    t1: &Variable,
    t2: &Variable,
    lp: &Variable,
    r: &Variable,
    d: &Variable,
    c: GpuSacLossConfig,
) -> Variable {
    let one = Variable::from_tensor(Tensor::ones(d.data().shape()));
    let soft = &elementwise_min_var(&t1.detach(), &t2.detach()) + &(&lp.detach() * -c.alpha);
    (&r.detach() + &(&(&(&one - &d.detach()) * &soft) * c.gamma)).detach()
}
fn critic<'a>(
    q1: &'a GpuVariable,
    q2: &'a GpuVariable,
    t1: &'a GpuVariable,
    t2: &'a GpuVariable,
    lp: &'a GpuVariable,
    r: &'a GpuVariable,
    d: &'a GpuVariable,
) -> GpuSacCriticInputs<'a> {
    GpuSacCriticInputs {
        q1,
        q2,
        target_q1: t1,
        target_q2: t2,
        next_log_probs: lp,
        rewards: r,
        dones: d,
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn soft_targets_losses_gradients_and_detach_boundaries_match_cpu_and_f64() {
    let av = [0.1, -0.4, 0.7, -1.];
    let bv = [-0.2, 0.3, -0.1, 0.5];
    let t1 = [2., 3., 1., -2.];
    let t2 = [1., 2., 1., -3.];
    let lp = [-0.5, -1., 0.2, -2.];
    let r = [0.5, -1., 0.2, 2.];
    let d = [0., 1., 0.25, 0.];
    for gamma in [0., 0.9, 1.] {
        for alpha in [0., 0.2, 1.] {
            let c = GpuSacLossConfig { gamma, alpha };
            let q1 = gpu(&av, &[4, 1], true);
            let q2 = gpu(&bv, &[4, 1], true);
            let gt1 = gpu(&t1, &[4, 1], true);
            let gt2 = gpu(&t2, &[4, 1], true);
            let gl = gpu(&lp, &[4, 1], true);
            let gr = gpu(&r, &[4, 1], true);
            let gd = gpu(&d, &[4, 1], true);
            let loss = sac_critic_loss(critic(&q1, &q2, &gt1, &gt2, &gl, &gr, &gd), c).unwrap();
            let metrics = loss.checked_metrics().unwrap();
            let cq1 = cpu(&av, &[4, 1], true);
            let cq2 = cpu(&bv, &[4, 1], true);
            let y = target(
                &cpu(&t1, &[4, 1], true),
                &cpu(&t2, &[4, 1], true),
                &cpu(&lp, &[4, 1], true),
                &cpu(&r, &[4, 1], true),
                &cpu(&d, &[4, 1], true),
                c,
            );
            let cl = &mse_loss(&cq1, &y) + &mse_loss(&cq2, &y);
            close(&[metrics.total_loss], &[cl.data().item()], 2e-6);
            close(
                &loss.target_values.to_cpu().unwrap().to_vec(),
                &y.data().to_vec(),
                1e-6,
            );
            loss.total_loss.backward().unwrap();
            cl.backward();
            close(
                &q1.grad_cpu().unwrap().unwrap().to_vec(),
                &cq1.grad().unwrap().to_vec(),
                1e-6,
            );
            close(
                &q2.grad_cpu().unwrap().unwrap().to_vec(),
                &cq2.grad().unwrap().to_vec(),
                1e-6,
            );
            assert!(!loss.target_values.requires_grad());
            for v in [&gt1, &gt2, &gl, &gr, &gd] {
                assert!(v.grad().is_none());
            }
            let evaluate = |a: &[f64], b: &[f64]| -> f64 {
                (0..4)
                    .map(|i| {
                        let y = r[i] as f64
                            + gamma as f64
                                * (1. - d[i] as f64)
                                * ((t1[i] as f64).min(t2[i] as f64) - alpha as f64 * lp[i] as f64);
                        ((a[i] - y).powi(2) + (b[i] - y).powi(2)) / 4.
                    })
                    .sum()
            };
            for (which, g) in [(0, &q1), (1, &q2)] {
                for i in 0..4 {
                    let mut a = av.map(|x| x as f64);
                    let mut b = bv.map(|x| x as f64);
                    let eps = 1e-5;
                    if which == 0 {
                        a[i] += eps;
                    } else {
                        b[i] += eps;
                    }
                    let plus = evaluate(&a, &b);
                    if which == 0 {
                        a[i] -= 2. * eps;
                    } else {
                        b[i] -= 2. * eps;
                    }
                    let minus = evaluate(&a, &b);
                    approx::assert_abs_diff_eq!(
                        g.grad_cpu().unwrap().unwrap().to_vec()[i] as f64,
                        (plus - minus) / (2. * eps),
                        epsilon = 2e-6
                    );
                }
            }
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn actor_objective_matches_cpu_gradients_f64_and_cpu_tie_convention() {
    for alpha in [0., 0.2, 1.] {
        let a = [0.3, -0.4, 0.8];
        let b = [-0.2, 0.7, -0.1];
        let l = [-0.5, 0.2, -1.];
        let q1 = gpu(&a, &[3, 1], true);
        let q2 = gpu(&b, &[3, 1], true);
        let lp = gpu(&l, &[3, 1], true);
        let c1 = cpu(&a, &[3, 1], true);
        let c2 = cpu(&b, &[3, 1], true);
        let cl = cpu(&l, &[3, 1], true);
        let loss = sac_actor_loss(&q1, &q2, &lp, alpha).unwrap();
        let expected = (&(&cl * alpha) - &elementwise_min_var(&c1, &c2)).mean();
        close(
            &[loss.checked_loss().unwrap()],
            &[expected.data().item()],
            1e-6,
        );
        loss.loss.backward().unwrap();
        expected.backward();
        for (g, c) in [(&q1, &c1), (&q2, &c2), (&lp, &cl)] {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                1e-6,
            );
        }
        let value = |a: [f64; 3], b: [f64; 3], l: [f64; 3]| {
            (0..3)
                .map(|i| (alpha as f64 * l[i] - a[i].min(b[i])) / 3.)
                .sum::<f64>()
        };
        for (which, g) in [(0, &q1), (1, &q2), (2, &lp)] {
            for i in 0..3 {
                let mut ap = a.map(|v| v as f64);
                let mut bp = b.map(|v| v as f64);
                let mut lp = l.map(|v| v as f64);
                let mut am = ap;
                let mut bm = bp;
                let mut lm = lp;
                let eps = 1e-5;
                match which {
                    0 => {
                        ap[i] += eps;
                        am[i] -= eps;
                    }
                    1 => {
                        bp[i] += eps;
                        bm[i] -= eps;
                    }
                    _ => {
                        lp[i] += eps;
                        lm[i] -= eps;
                    }
                }
                approx::assert_abs_diff_eq!(
                    g.grad_cpu().unwrap().unwrap().to_vec()[i] as f64,
                    (value(ap, bp, lp) - value(am, bm, lm)) / (2. * eps),
                    epsilon = 1e-6
                );
            }
        }
    }
    let a = gpu(&[0.4, 0.4], &[2, 1], true);
    let b = gpu(&[0.4, 0.4], &[2, 1], true);
    let l = gpu(&[0., 0.], &[2, 1], false);
    sac_actor_loss(&a, &b, &l, 0.2)
        .unwrap()
        .loss
        .backward()
        .unwrap();
    close(&a.grad_cpu().unwrap().unwrap().to_vec(), &[0., 0.], 0.);
    close(&b.grad_cpu().unwrap().unwrap().to_vec(), &[-0.5, -0.5], 0.);
}
#[test]
#[ignore = "requires a GPU adapter"]
fn learned_temperature_detaches_policy_and_matches_cpu_f64_and_four_adam_updates() {
    for shape in [&[][..], &[1][..], &[1, 1][..]] {
        let log = gpu(&[0.2f32.ln()], shape, true);
        let clog = cpu(&[0.2f32.ln()], shape, true);
        let lp = gpu(&[-0.3, -0.8, 0.2], &[3, 1], true);
        let clp = cpu(&[-0.3, -0.8, 0.2], &[3, 1], true);
        let mut ga = GpuAdam::new(vec![log.clone()], 0.003).unwrap();
        let mut ca = Adam::new(vec![clog.clone()], 0.003);
        for _ in 0..4 {
            ga.zero_grad();
            ca.zero_grad();
            let loss = sac_temperature_loss(&log, &lp, -2.).unwrap();
            let error = &clp.detach() + &cpu(&[-2.; 3], &[3, 1], false);
            let shaped = cpu(&[error.mean().data().item()], shape, false);
            let expected = &(&clog * &shaped) * -1.;
            let m = loss.checked_metrics().unwrap();
            close(
                &[m.loss, m.alpha],
                &[expected.data().item(), clog.data().item().exp()],
                2e-6,
            );
            assert!(!loss.alpha.requires_grad());
            loss.loss.backward().unwrap();
            expected.backward();
            close(
                &log.grad_cpu().unwrap().unwrap().to_vec(),
                &clog.grad().unwrap().to_vec(),
                1e-6,
            );
            let finite_difference =
                |x: f64| -x * ((-0.3f32 as f64 - 0.8f32 as f64 + 0.2f32 as f64) / 3. - 2.);
            let x = log.to_cpu().unwrap().item() as f64;
            approx::assert_abs_diff_eq!(
                log.grad_cpu().unwrap().unwrap().item() as f64,
                (finite_difference(x + 1e-5) - finite_difference(x - 1e-5)) / 2e-5,
                epsilon = 1e-6
            );
            assert!(lp.grad().is_none() && clp.grad().is_none());
            ga.step().unwrap();
            ca.step();
            close(&log.to_cpu().unwrap().to_vec(), &clog.data().to_vec(), 1e-6);
        }
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_affine_twin_critics_match_cpu_sac_formula_and_four_adam_updates() {
    let q1 = GpuLinear::new_seeded(context(), 2, 1, 42).unwrap();
    let q2 = GpuLinear::new_seeded(context(), 2, 1, 43).unwrap();
    let c1 = Linear::new_seeded(2, 1, 42);
    let c2 = Linear::new_seeded(2, 1, 43);
    let x = gpu(&[1., 0., 0., 1., 0.2, 0.7], &[3, 2], false);
    let cx = cpu(&[1., 0., 0., 1., 0.2, 0.7], &[3, 2], false);
    let t1 = gpu(&[1., -0.2, 0.5], &[3, 1], true);
    let t2 = gpu(&[0.5, 0.3, 0.4], &[3, 1], true);
    let lp = gpu(&[-0.5, -1., 0.2], &[3, 1], true);
    let r = gpu(&[0.3, -0.2, 0.5], &[3, 1], false);
    let d = gpu(&[0., 1., 0.], &[3, 1], false);
    let c = GpuSacLossConfig {
        gamma: 0.9,
        alpha: 0.2,
    };
    let y = target(
        &cpu(&[1., -0.2, 0.5], &[3, 1], false),
        &cpu(&[0.5, 0.3, 0.4], &[3, 1], false),
        &cpu(&[-0.5, -1., 0.2], &[3, 1], false),
        &cpu(&[0.3, -0.2, 0.5], &[3, 1], false),
        &cpu(&[0., 1., 0.], &[3, 1], false),
        c,
    );
    let mut gp = q1.parameters();
    gp.extend(q2.parameters());
    let mut cp = c1.parameters();
    cp.extend(c2.parameters());
    let mut ga = GpuAdam::new(gp.clone(), 0.003).unwrap();
    let mut ca = Adam::new(cp.clone(), 0.003);
    for _ in 0..4 {
        ga.zero_grad();
        ca.zero_grad();
        let loss = sac_critic_loss(
            critic(
                &q1.forward(&x).unwrap(),
                &q2.forward(&x).unwrap(),
                &t1,
                &t2,
                &lp,
                &r,
                &d,
            ),
            c,
        )
        .unwrap();
        let expected = &mse_loss(&c1.forward(&cx), &y) + &mse_loss(&c2.forward(&cx), &y);
        close(
            &[loss.checked_metrics().unwrap().total_loss],
            &[expected.data().item()],
            2e-6,
        );
        loss.total_loss.backward().unwrap();
        expected.backward();
        for (g, c) in gp.iter().zip(&cp) {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                2e-6,
            );
        }
        ga.step().unwrap();
        ca.step();
        for (g, c) in gp.iter().zip(&cp) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 3e-6);
        }
    }
    assert!(t1.grad().is_none() && t2.grad().is_none() && lp.grad().is_none());
}
#[test]
#[ignore = "requires a GPU adapter"]
fn actual_seeded_gaussian_actor_sampling_and_frozen_twins_match_cpu_adam_updates() {
    let low = [-2., 0.5];
    let high = [1., 4.];
    let actor = GpuGaussianPolicyNet::new_seeded(context(), 2, 8, 2, 42).unwrap();
    let policy = GaussianPolicy::new_seeded(2, 8, 2, &low, &high, 42);
    let transform = GpuSacActionTransform::new(context(), &low, &high).unwrap();
    let x = gpu(&[1., 0., 0., 1., 0.2, 0.7], &[3, 2], false);
    let cx = cpu(&[1., 0., 0., 1., 0.2, 0.7], &[3, 2], false);
    let w1 = gpu(&[0.1, 0.2, 0.7, -0.3], &[4, 1], false);
    let w2 = gpu(&[-0.2, 0.1, 0.2, 0.6], &[4, 1], false);
    let cw1 = cpu(&[0.1, 0.2, 0.7, -0.3], &[4, 1], false);
    let cw2 = cpu(&[-0.2, 0.1, 0.2, 0.6], &[4, 1], false);
    let gp = actor.parameters();
    let cp = policy.parameters();
    let mut ga = GpuAdam::new(gp.clone(), 0.003).unwrap();
    let mut ca = Adam::new(cp.clone(), 0.003);
    for (g, c) in gp.iter().zip(&cp) {
        close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 0.);
    }
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..4 {
        ga.zero_grad();
        ca.zero_grad();
        let mut noise_rng = rng.clone();
        let noise: Vec<_> = (0..6)
            .map(|_| {
                let u1: f32 = noise_rng.gen_range(1e-7..1.);
                let u2: f32 = noise_rng.gen_range(0. ..std::f32::consts::TAU);
                (-2. * u1.ln()).sqrt() * u2.cos()
            })
            .collect();
        let noise = gpu(&noise, &[3, 2], true);
        let (mean, std) = actor.forward_raw(&x).unwrap();
        let sample = transform.sample_with_noise(&mean, &std, &noise).unwrap();
        sample.checked_metrics().unwrap();
        let (actions, lp) = policy.sample_with_rng(&cx, &mut rng);
        close(
            &sample.actions.to_cpu().unwrap().to_vec(),
            &actions.data().to_vec(),
            2e-5,
        );
        close(
            &sample.distribution.log_probs.to_cpu().unwrap().to_vec(),
            &lp.data().to_vec(),
            3e-5,
        );
        let sa = x.concat_columns(&sample.actions).unwrap();
        let csa = cx.concat(&actions, 1);
        let q1 = sa.matmul(&w1).unwrap();
        let q2 = sa.matmul(&w2).unwrap();
        let cq1 = csa.matmul(&cw1);
        let cq2 = csa.matmul(&cw2);
        let loss = sac_actor_loss(&q1, &q2, &sample.distribution.log_probs, 0.2).unwrap();
        let expected = (&(&lp * 0.2) - &elementwise_min_var(&cq1, &cq2)).mean();
        close(
            &[loss.checked_loss().unwrap()],
            &[expected.data().item()],
            3e-5,
        );
        loss.loss.backward().unwrap();
        expected.backward();
        assert!(noise.grad().is_none() && w1.grad().is_none() && w2.grad().is_none());
        for (g, c) in gp.iter().zip(&cp) {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                3e-5,
            );
        }
        ga.step().unwrap();
        ca.step();
        for (g, c) in gp.iter().zip(&cp) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 3e-5);
        }
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn supplied_noise_squashed_actor_gradient_matches_independent_f64_and_saturation_contract() {
    let low = [-2., 0.5];
    let high = [1., 4.];
    let transform = GpuSacActionTransform::new(context(), &low, &high).unwrap();
    let m = [0.2, -0.3, 1.2, -1.1];
    let s = [-0.5, 0.25, -0.7, 0.1];
    let n = [0.4, -0.8, 1.1, -0.2];
    let mean = gpu(&m, &[2, 2], true);
    let std = gpu(&s, &[2, 2], true);
    let noise = gpu(&n, &[2, 2], true);
    let sample = transform.sample_with_noise(&mean, &std, &noise).unwrap();
    sample.checked_metrics().unwrap();
    let q1 = sample
        .actions
        .matmul(&gpu(&[1., 0.3], &[2, 1], false))
        .unwrap();
    let q2 = sample
        .actions
        .matmul(&gpu(&[-0.4, 0.7], &[2, 1], false))
        .unwrap();
    let loss = sac_actor_loss(&q1, &q2, &sample.distribution.log_probs, 0.2).unwrap();
    loss.checked_loss().unwrap();
    loss.loss.backward().unwrap();
    assert!(noise.grad().is_none());
    let value = |m: [f64; 4], s: [f64; 4]| {
        let mut a = [0f64; 4];
        let mut lp = [0f64; 2];
        for i in 0..4 {
            let scale = (high[i % 2] - low[i % 2]) as f64 / 2.;
            let bias = (high[i % 2] + low[i % 2]) as f64 / 2.;
            let u = m[i] + n[i] as f64 * s[i].exp();
            let t = u.tanh();
            a[i] = t * scale + bias;
            lp[i / 2] += -0.5 * (n[i] as f64).powi(2)
                - s[i]
                - 0.5 * (2. * std::f64::consts::PI).ln()
                - (1. - t * t + 1e-6).ln()
                - scale.ln();
        }
        (0..2)
            .map(|i| {
                (0.2 * lp[i]
                    - (a[2 * i] + 0.3 * a[2 * i + 1]).min(-0.4 * a[2 * i] + 0.7 * a[2 * i + 1]))
                    / 2.
            })
            .sum::<f64>()
    };
    approx::assert_abs_diff_eq!(
        loss.checked_loss().unwrap() as f64,
        value(m.map(|v| v as f64), s.map(|v| v as f64)),
        epsilon = 2e-5
    );
    for (which, g) in [(0, &mean), (1, &std)] {
        for i in 0..4 {
            let mut mp = m.map(|v| v as f64);
            let mut mm = mp;
            let mut sp = s.map(|v| v as f64);
            let mut sm = sp;
            let eps = 1e-5;
            if which == 0 {
                mp[i] += eps;
                mm[i] -= eps;
            } else {
                sp[i] += eps;
                sm[i] -= eps;
            }
            approx::assert_abs_diff_eq!(
                g.grad_cpu().unwrap().unwrap().to_vec()[i] as f64,
                (value(mp, sp) - value(mm, sm)) / (2. * eps),
                epsilon = 2e-4
            );
        }
    }
    let mean = gpu(&[20., -20.], &[1, 2], true);
    let raw = gpu(&[-21., 3.], &[1, 2], true);
    let zero = gpu(&[0., 0.], &[1, 2], true);
    let saturated = transform.sample_with_noise(&mean, &raw, &zero).unwrap();
    saturated.checked_metrics().unwrap();
    close(
        &saturated.actions.to_cpu().unwrap().to_vec(),
        &[high[0], low[1]],
        1e-6,
    );
    saturated
        .distribution
        .log_probs
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    close(&raw.grad_cpu().unwrap().unwrap().to_vec(), &[0., 0.], 0.);
    assert!(zero.grad().is_none());
    let detached = no_grad(|| transform.sample_with_noise(&mean, &raw, &zero)).unwrap();
    assert!(!detached.actions.requires_grad() && !detached.distribution.log_probs.requires_grad());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn invalid_batches_devices_masks_config_and_hidden_nonfinite_values_are_rejected() {
    let v = gpu(&[0.2, 0.4], &[2, 1], true);
    let d = gpu(&[0., 1.], &[2, 1], false);
    let c = GpuSacLossConfig::default();
    for bad in [
        gpu(&[], &[0, 1], false),
        gpu(&[0.], &[1, 1], false),
        gpu(&[0., 0.], &[2], false),
    ] {
        assert!(sac_critic_loss(critic(&v, &bad, &v, &v, &v, &v, &d), c).is_err());
        assert!(sac_actor_loss(&v, &bad, &v, 0.2).is_err());
    }
    assert!(sac_actor_loss(
        &gpu(&[], &[0, 1], true),
        &gpu(&[], &[0, 1], true),
        &gpu(&[], &[0, 1], true),
        0.2
    )
    .is_err());
    let foreign =
        GpuVariable::new(&GpuContext::new().unwrap(), &Tensor::ones(&[2, 1]), false).unwrap();
    assert!(sac_critic_loss(critic(&v, &v, &foreign, &v, &v, &v, &d), c).is_err());
    assert!(sac_actor_loss(&v, &foreign, &v, 0.2).is_err());
    assert!(sac_temperature_loss(&gpu(&[0., 0.], &[2], true), &v, -1.).is_err());
    assert!(sac_temperature_loss(&gpu(&[0.], &[1], true), &foreign, -1.).is_err());
    for invalid in [-1., f32::NAN, f32::INFINITY] {
        assert!(sac_actor_loss(&v, &v, &v, invalid).is_err());
    }
    assert!(sac_temperature_loss(&gpu(&[0.], &[1], true), &v, f32::NAN).is_err());
    for mask in [-0.1, 1.1, f32::NAN] {
        let invalid = gpu(&[mask, 1.], &[2, 1], false);
        assert!(sac_critic_loss(critic(&v, &v, &v, &v, &v, &v, &invalid), c)
            .unwrap()
            .checked_metrics()
            .is_err());
    }
    for position in 0..7 {
        let invalid = gpu(&[f32::NAN, 0.], &[2, 1], true);
        let mut inputs = [&v, &v, &v, &v, &v, &v, &d];
        inputs[position] = &invalid;
        let loss = sac_critic_loss(
            critic(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
            ),
            GpuSacLossConfig {
                gamma: 0.,
                alpha: 0.,
            },
        )
        .unwrap();
        assert!(loss.checked_metrics().is_err());
    }
    let nan = gpu(&[f32::NAN, 0.], &[2, 1], true);
    assert!(sac_actor_loss(&nan, &v, &v, 0.2)
        .unwrap()
        .checked_loss()
        .is_err());
    assert!(sac_actor_loss(&v, &v, &nan, 0.)
        .unwrap()
        .checked_loss()
        .is_err());
    // Finite raw values can overflow the entropy term before masking it out.
    let huge = gpu(&[f32::MAX, f32::MAX], &[2, 1], false);
    let terminal = gpu(&[1., 1.], &[2, 1], false);
    assert!(sac_critic_loss(
        critic(&v, &v, &v, &v, &huge, &v, &terminal),
        GpuSacLossConfig {
            gamma: 0.,
            alpha: 2.
        }
    )
    .unwrap()
    .checked_metrics()
    .is_err());
    assert!(sac_actor_loss(&huge, &huge, &v, 0.2)
        .unwrap()
        .checked_loss()
        .is_err());
    for log in [f32::NAN, 100., -200.] {
        assert!(matches!(
            sac_temperature_loss(&gpu(&[log], &[1], true), &v, -1.)
                .unwrap()
                .checked_metrics(),
            Err(GpuSacError::NonFinite | GpuSacError::InvalidTemperature)
        ));
    }
    assert!(sac_temperature_loss(&gpu(&[0.], &[1], true), &huge, 0.)
        .unwrap()
        .checked_metrics()
        .is_err());
    let transform = GpuSacActionTransform::new(context(), &[-1.], &[1.]).unwrap();
    let raw = gpu(&[0., 0.], &[2, 1], true);
    assert!(transform
        .sample_with_noise(
            &gpu(&[], &[0, 1], true),
            &gpu(&[], &[0, 1], true),
            &gpu(&[], &[0, 1], true)
        )
        .is_err());
    assert!(transform.sample_with_noise(&v, &raw, &foreign).is_err());
    assert!(transform
        .sample_with_noise(&v, &gpu(&[0.], &[1, 1], false), &v)
        .is_err());
    for (mean, std, noise) in [(&nan, &raw, &v), (&v, &nan, &v), (&v, &raw, &nan)] {
        assert!(transform
            .sample_with_noise(mean, std, noise)
            .unwrap()
            .checked_metrics()
            .is_err());
    }
    let mean = gpu(&[f32::MAX, f32::MAX], &[2, 1], true);
    let std = gpu(&[2., 2.], &[2, 1], true);
    assert!(transform
        .sample_with_noise(&mean, &std, &huge)
        .unwrap()
        .checked_metrics()
        .is_err());
    assert!(GpuSacActionTransform::new(context(), &[], &[]).is_err());
    assert!(GpuSacActionTransform::new(context(), &[1.], &[1.]).is_err());
}
