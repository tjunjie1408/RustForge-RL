#![cfg(feature = "gpu")]
use rustforge_autograd::{
    gpu::{GpuSgd, GpuVariable},
    no_grad, Variable,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::sync::OnceLock;
fn context() -> &'static GpuContext {
    static CONTEXT: OnceLock<GpuContext> = OnceLock::new();
    CONTEXT.get_or_init(|| GpuContext::new().expect("GPU tests require an adapter"))
}
fn variable(values: &[f32], shape: &[usize], grad: bool) -> GpuVariable {
    GpuVariable::new(context(), &Tensor::from_vec(values.to_vec(), shape), grad).unwrap()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "nonfinite gradient or parameter: {a}");
        approx::assert_abs_diff_eq!(a, b, epsilon = tolerance);
    }
}
fn gradient(v: &GpuVariable) -> Vec<f32> {
    v.grad_cpu().unwrap().unwrap().to_vec()
}

#[test]
#[ignore = "requires a GPU adapter"]
fn shared_graph_repeated_backward_and_cpu_parity() {
    let values = vec![-2., 0., 3.];
    let x = variable(&values, &[3], true);
    let square = x.mul(&x).unwrap();
    let loss = square
        .add(&square)
        .unwrap()
        .scale(0.5)
        .unwrap()
        .mean()
        .unwrap();
    loss.backward().unwrap();
    let cpu = Variable::new(Tensor::from_vec(values, &[3]), true);
    let cpu_loss = (&cpu * &cpu).mean();
    cpu_loss.backward();
    close(&gradient(&x), &cpu.grad().unwrap().to_vec(), 1e-6);
    assert!(square.grad().is_none());
    loss.backward().unwrap();
    close(&gradient(&x), &[-8. / 3., 0., 4.], 1e-6);
    x.zero_grad();
    square.sum().unwrap().backward().unwrap();
    close(&gradient(&x), &[-4., 0., 6.], 1e-6);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn matrix_gradients_match_finite_differences() {
    // Rectangular matrices catch swapped dimensions in each transpose rule.
    for kind in 0..3 {
        let (ashape, bshape): (&[usize], &[usize]) = match kind {
            0 => (&[2, 3], &[3, 2]),
            1 => (&[2, 3], &[2, 3]),
            _ => (&[3, 2], &[3, 2]),
        };
        let av = vec![0.2, -0.4, 0.8, 1.1, 0.3, -0.7];
        let bv = vec![-0.6, 0.5, 0.9, -0.2, 0.4, 0.73];
        let evaluate = |a: &[f32], b: &[f32], grad: bool| {
            let a = variable(a, ashape, grad);
            let b = variable(b, bshape, grad);
            let product = match kind {
                0 => a.matmul(&b),
                1 => a.matmul_t(&b),
                _ => a.t_matmul(&b),
            }
            .unwrap();
            let prediction = product.relu().unwrap();
            let target = variable(&[0.1, -0.3, 0.2, 0.5], &[2, 2], false);
            (prediction.mse_loss(&target).unwrap(), a, b)
        };
        let (loss, a, b) = evaluate(&av, &bv, true);
        loss.backward().unwrap();
        for (side, analytic) in [(0, gradient(&a)), (1, gradient(&b))] {
            for (i, &expected) in analytic.iter().enumerate() {
                let (mut ap, mut bp) = (av.clone(), bv.clone());
                let selected = if side == 0 { &mut ap } else { &mut bp };
                selected[i] += 0.001;
                let plus = evaluate(&ap, &bp, false).0.to_cpu().unwrap().to_vec()[0];
                let selected = if side == 0 { &mut ap } else { &mut bp };
                selected[i] -= 0.002;
                let minus = evaluate(&ap, &bp, false).0.to_cpu().unwrap().to_vec()[0];
                close(&[expected], &[(plus - minus) / 0.002], 0.0002);
            }
        }
        if kind == 0 {
            let ca = Variable::new(Tensor::from_vec(av.clone(), ashape), true);
            let cb = Variable::new(Tensor::from_vec(bv.clone(), bshape), true);
            let target = Variable::new(Tensor::from_vec(vec![0.1, -0.3, 0.2, 0.5], &[2, 2]), false);
            let diff = &ca.matmul(&cb).relu() - &target;
            (&diff * &diff).mean().backward();
            close(&gradient(&a), &ca.grad().unwrap().to_vec(), 1e-6);
            close(&gradient(&b), &cb.grad().unwrap().to_vec(), 1e-6);
        }
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn reductions_relu_empty_and_scalar_contracts() {
    let x = variable(&[-1., 0., 2., f32::NAN], &[2, 2], true);
    x.relu().unwrap().sum().unwrap().backward().unwrap();
    close(&gradient(&x), &[0., 0., 1., 0.], 0.);
    assert!(x.backward().is_err());
    let empty = variable(&[], &[0, 3], true);
    let loss = empty.mean().unwrap();
    close(&loss.to_cpu().unwrap().to_vec(), &[0.], 0.);
    loss.backward().unwrap();
    assert_eq!(empty.grad().unwrap().shape(), &[0, 3]);
    let scalar = variable(&[4.], &[], true);
    scalar.scale(3.).unwrap().backward().unwrap();
    close(&gradient(&scalar), &[3.], 0.);
    assert!(context()
        .broadcast_scalar_device(&x.data(), &[2], 1.)
        .is_err());
    assert!(x.add(&scalar).is_err());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn sgd_momentum_and_forward_snapshots() {
    for momentum in [0., 0.8] {
        let x = variable(&[2., -1.], &[2], true);
        let snapshot = x.detach();
        let old_loss = x.mul(&x).unwrap().sum().unwrap();
        let mut opt = GpuSgd::new(vec![x.clone()], 0.1, momentum).unwrap();
        let mut expected = [2., -1.];
        let mut velocity = [0., 0.];
        for _ in 0..4 {
            opt.zero_grad();
            x.mul(&x).unwrap().sum().unwrap().backward().unwrap();
            for i in 0..2 {
                velocity[i] = momentum * velocity[i] + 2. * expected[i];
                expected[i] -= 0.1 * velocity[i];
            }
            opt.step().unwrap();
            close(&x.to_cpu().unwrap().to_vec(), &expected, 1e-6);
            assert!(!x.has_grad_fn());
        }
        opt.zero_grad();
        old_loss.backward().unwrap();
        close(&gradient(&x), &[4., -2.], 0.);
        close(&snapshot.to_cpu().unwrap().to_vec(), &[2., -1.], 0.);
        assert!(!snapshot.requires_grad() && !snapshot.has_grad_fn());
        let untracked = no_grad(|| x.mul(&x).unwrap());
        assert!(!untracked.requires_grad() && !untracked.has_grad_fn());
        opt.zero_grad();
        untracked.sum().unwrap().backward().unwrap();
        assert!(x.grad().is_none());
        opt.step().unwrap();
        close(&x.to_cpu().unwrap().to_vec(), &expected, 1e-6);
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn optimizer_and_device_validation() {
    let x = variable(&[1.], &[1], true);
    for (lr, m) in [
        (0., 0.),
        (-1., 0.),
        (f32::NAN, 0.),
        (0.1, -0.1),
        (0.1, 1.),
        (0.1, f32::INFINITY),
    ] {
        assert!(GpuSgd::new(vec![x.clone()], lr, m).is_err());
    }
    assert!(GpuSgd::new(vec![x.clone(), x.clone()], 0.1, 0.).is_err());
    assert!(GpuSgd::new(vec![x.detach()], 0.1, 0.).is_err());
    assert!(GpuSgd::new(vec![x.scale(2.).unwrap()], 0.1, 0.).is_err());
    // Retain devices until process exit: Mesa EGL teardown can conflict with
    // other tests concurrently using the shared software context.
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    assert!(GpuVariable::from_device(context(), other.zeros(&[1]).unwrap(), true).is_err());
    let foreign = GpuVariable::from_device(other, other.zeros(&[1]).unwrap(), true).unwrap();
    assert!(x.mul(&foreign).is_err());
    assert!(GpuSgd::new(vec![x, foreign], 0.1, 0.).is_err());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn deterministic_training_converges() {
    let x = variable(&[1., 0., 0., 1., 1., 1., -1., 1.], &[4, 2], false);
    let y = variable(&[2., -3., -1., -5.], &[4, 1], false);
    let w = variable(&[0., 0.], &[2, 1], true);
    let mut opt = GpuSgd::new(vec![w.clone()], 0.1, 0.8).unwrap();
    for _ in 0..100 {
        opt.zero_grad();
        x.matmul(&w)
            .unwrap()
            .mse_loss(&y)
            .unwrap()
            .backward()
            .unwrap();
        opt.step().unwrap();
    }
    close(&w.to_cpu().unwrap().to_vec(), &[2., -3.], 0.001);
    let loss = x
        .matmul(&w)
        .unwrap()
        .mse_loss(&y)
        .unwrap()
        .to_cpu()
        .unwrap()
        .to_vec()[0];
    assert!(loss.is_finite() && loss < 1e-6);
    assert!(x.grad().is_none() && y.grad().is_none());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn bias_broadcast_gradients_match_cpu_and_finite_differences() {
    let values = vec![0.1, 0.3, -0.4, 0.7, 0.2, -0.8];
    let bias_values = vec![0.2, -0.1, 0.5];
    let x = variable(&values, &[2, 3], true);
    let b = variable(&bias_values, &[3], true);
    let out = x.add_bias(&b).unwrap();
    out.mul(&out).unwrap().mean().unwrap().backward().unwrap();
    let cx = Variable::new(Tensor::from_vec(values.clone(), &[2, 3]), true);
    let cb = Variable::new(Tensor::from_vec(bias_values.clone(), &[3]), true);
    let cout = &cx + &cb;
    (&cout * &cout).mean().backward();
    close(&gradient(&x), &cx.grad().unwrap().to_vec(), 1e-6);
    close(&gradient(&b), &cb.grad().unwrap().to_vec(), 1e-6);
    for i in 0..3 {
        let evaluate = |offset| {
            let mut bs = bias_values.clone();
            bs[i] += offset;
            let y = variable(&values, &[2, 3], false)
                .add_bias(&variable(&bs, &[3], false))
                .unwrap();
            y.mul(&y).unwrap().mean().unwrap().to_cpu().unwrap().item()
        };
        close(
            &[gradient(&b)[i]],
            &[(evaluate(0.001) - evaluate(-0.001)) / 0.002],
            0.0001,
        );
    }
    let empty = variable(&[], &[0, 3], true);
    b.zero_grad();
    empty
        .add_bias(&b)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    close(&gradient(&b), &[0.; 3], 0.);
    assert_eq!(empty.grad().unwrap().shape(), &[0, 3]);
    assert!(x.add_bias(&variable(&[1., 2.], &[2], true)).is_err());
    assert!(x.add_bias(&variable(&[1., 2., 3.], &[1, 3], true)).is_err());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn adam_matches_cpu_with_missing_and_zero_gradients() {
    use rustforge_autograd::{
        gpu::GpuAdam,
        optimizer::{adam::Adam, Optimizer},
    };
    for (beta1, beta2, epsilon) in [(0.9, 0.999, 1e-8), (0., 0., 0.001), (0.5, 0.8, 0.01)] {
        let a = variable(&[1., -2., 0.], &[3], true);
        let b = variable(&[3., -0.5, 0.], &[3], true);
        let ca = Variable::new(Tensor::from_vec(vec![1., -2., 0.], &[3]), true);
        let cb = Variable::new(Tensor::from_vec(vec![3., -0.5, 0.], &[3]), true);
        let snapshot = a.detach();
        let mut gpu =
            GpuAdam::with_betas(vec![a.clone(), b.clone()], 0.03, beta1, beta2, epsilon).unwrap();
        let mut cpu = Adam::with_betas(vec![ca.clone(), cb.clone()], 0.03, beta1, beta2, epsilon);
        for step in 0..6 {
            gpu.zero_grad();
            cpu.zero_grad();
            // First iteration has no gradients: both optimizers advance the
            // global clock. Later steps alternate parameters and then share one.
            if step > 0 && step != 2 {
                a.mul(&a).unwrap().sum().unwrap().backward().unwrap();
                (&ca * &ca).sum().backward();
            }
            if step >= 2 {
                b.mul(&b).unwrap().mean().unwrap().backward().unwrap();
                (&cb * &cb).mean().backward();
            }
            gpu.step().unwrap();
            cpu.step();
            close(&a.to_cpu().unwrap().to_vec(), &ca.data().to_vec(), 2e-6);
            close(&b.to_cpu().unwrap().to_vec(), &cb.data().to_vec(), 2e-6);
        }
        close(&snapshot.to_cpu().unwrap().to_vec(), &[1., -2., 0.], 0.);
        assert!(!a.has_grad_fn() && !b.has_grad_fn());
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn adam_rejects_invalid_configuration_and_parameters() {
    use rustforge_autograd::gpu::GpuAdam;
    let p = variable(&[1.], &[1], true);
    for (lr, b1, b2, eps) in [
        (0., 0.9, 0.99, 1e-8),
        (f32::INFINITY, 0.9, 0.99, 1e-8),
        (0.1, 1., 0.99, 1e-8),
        (0.1, -0.1, 0.99, 1e-8),
        (0.1, 0.9, f32::NAN, 1e-8),
        (0.1, 0.9, 1., 1e-8),
        (0.1, 0.9, 0.99, 0.),
        (0.1, 0.9, 0.99, f32::INFINITY),
    ] {
        assert!(GpuAdam::with_betas(vec![p.clone()], lr, b1, b2, eps).is_err());
    }
    assert!(GpuAdam::new(vec![p.clone(), p.clone()], 0.1).is_err());
    assert!(GpuAdam::new(vec![p.detach()], 0.1).is_err());
    assert!(GpuAdam::new(vec![p.scale(2.).unwrap()], 0.1).is_err());
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    let foreign = GpuVariable::from_device(other, other.zeros(&[1]).unwrap(), true).unwrap();
    assert!(GpuAdam::new(vec![p.clone(), foreign], 0.1).is_err());
    close(&p.to_cpu().unwrap().to_vec(), &[1.], 0.);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn gather_gradients_match_cpu_and_finite_differences() {
    use std::rc::Rc;
    let values = vec![0.2, -0.3, 0.8, 0.4, -0.5, 1.2];
    let x = variable(&values, &[2, 3], true);
    let indices = Rc::new(context().upload_indices(&[1, 1], 3).unwrap());
    let selected = x.gather_actions(&indices).unwrap();
    selected
        .mul(&selected)
        .unwrap()
        .mean()
        .unwrap()
        .backward()
        .unwrap();
    let cpu = Variable::new(Tensor::from_vec(values.clone(), &[2, 3]), true);
    let choice = cpu.gather(1, &[1, 1]);
    (&choice * &choice).mean().backward();
    close(&gradient(&x), &cpu.grad().unwrap().to_vec(), 1e-6);
    for i in 0..6 {
        let eval = |offset| {
            let mut data = values.clone();
            data[i] += offset;
            let y = variable(&data, &[2, 3], false)
                .gather_actions(&indices)
                .unwrap();
            y.mul(&y).unwrap().mean().unwrap().to_cpu().unwrap().item()
        };
        close(
            &[gradient(&x)[i]],
            &[(eval(0.001) - eval(-0.001)) / 0.002],
            0.0001,
        );
    }
    selected.sum().unwrap().backward().unwrap();
    close(&gradient(&x), &[0., 0.7, 0., 0., 0.5, 0.], 1e-6);
    let empty = variable(&[], &[0, 3], true);
    empty
        .gather_actions(&Rc::new(context().upload_indices(&[], 3).unwrap()))
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(empty.grad().unwrap().shape(), &[0, 3]);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn leaf_snapshot_assignment_preserves_old_graphs_and_validates_before_change() {
    let source = variable(&[2., 3.], &[2], true);
    let destination = variable(&[1., -1.], &[2], true);
    let old_loss = destination.mul(&destination).unwrap().sum().unwrap();
    old_loss.backward().unwrap();
    destination.copy_data_from(&source).unwrap();
    assert!(destination.grad().is_none());
    old_loss.backward().unwrap();
    close(&gradient(&destination), &[2., -2.], 0.);
    close(&destination.to_cpu().unwrap().to_vec(), &[2., 3.], 0.);
    assert!(destination
        .copy_data_from(&variable(&[1.], &[1], false))
        .is_err());
    assert!(destination
        .scale(2.)
        .unwrap()
        .copy_data_from(&source)
        .is_err());
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    let foreign = GpuVariable::from_device(other, other.zeros(&[2]).unwrap(), false).unwrap();
    assert!(destination.copy_data_from(&foreign).is_err());
    close(&destination.to_cpu().unwrap().to_vec(), &[2., 3.], 0.);
    let frozen = destination.detach();
    frozen.copy_data_from(&source).unwrap();
    assert!(!frozen.requires_grad());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn adam_state_resume_matches_updates_with_missing_gradients_and_custom_betas() {
    use rustforge_autograd::gpu::GpuAdam;
    let a = variable(&[2., -1.], &[2], true);
    let untouched = variable(&[3.], &[1], true);
    let mut optimizer =
        GpuAdam::with_betas(vec![a.clone(), untouched.clone()], 0.03, 0.7, 0.8, 0.001).unwrap();
    for _ in 0..3 {
        optimizer.zero_grad();
        a.mul(&a).unwrap().mean().unwrap().backward().unwrap();
        optimizer.step().unwrap();
    }
    let state = optimizer.state().unwrap();
    assert_eq!(state.timestep, 3);
    assert!(state.moments[1].is_none());
    let b = GpuVariable::new(context(), &a.to_cpu().unwrap(), true).unwrap();
    let b_untouched = GpuVariable::new(context(), &untouched.to_cpu().unwrap(), true).unwrap();
    let mut resumed = GpuAdam::new(vec![b.clone(), b_untouched], 0.5).unwrap();
    b.sum().unwrap().backward().unwrap();
    resumed.restore_state(&state).unwrap();
    assert!(b.grad().is_none());
    for _ in 0..5 {
        optimizer.zero_grad();
        resumed.zero_grad();
        a.mul(&a).unwrap().mean().unwrap().backward().unwrap();
        b.mul(&b).unwrap().mean().unwrap().backward().unwrap();
        optimizer.step().unwrap();
        resumed.step().unwrap();
        close(
            &a.to_cpu().unwrap().to_vec(),
            &b.to_cpu().unwrap().to_vec(),
            0.,
        );
    }
    let actual = resumed.state().unwrap();
    let expected = optimizer.state().unwrap();
    assert_eq!(actual.timestep, expected.timestep);
    close(
        &actual.moments[0].as_ref().unwrap().first.to_vec(),
        &expected.moments[0].as_ref().unwrap().first.to_vec(),
        0.,
    );
    close(
        &actual.moments[0].as_ref().unwrap().second.to_vec(),
        &expected.moments[0].as_ref().unwrap().second.to_vec(),
        0.,
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn invalid_adam_state_preserves_optimizer_and_live_gradients() {
    use rustforge_autograd::gpu::GpuAdam;
    let p = variable(&[1., 2.], &[2], true);
    let mut optimizer = GpuAdam::new(vec![p.clone()], 0.1).unwrap();
    p.sum().unwrap().backward().unwrap();
    optimizer.step().unwrap();
    let state = optimizer.state().unwrap();
    let before = gradient(&p);
    for kind in 0..8 {
        let mut bad = state.clone();
        match kind {
            0 => bad.lr = f32::NAN,
            1 => bad.moments.clear(),
            2 => bad.timestep = 0,
            3 => bad.timestep = usize::MAX,
            4 => bad.moments[0].as_mut().unwrap().first = Tensor::zeros(&[1]),
            5 => bad.moments[0].as_mut().unwrap().second = Tensor::from_vec(vec![-1., 0.], &[2]),
            6 => {
                bad.moments[0].as_mut().unwrap().first =
                    Tensor::from_vec(vec![f32::INFINITY, 0.], &[2])
            }
            _ => bad.beta2 = 1.,
        }
        assert!(optimizer.restore_state(&bad).is_err());
        assert_eq!(optimizer.state().unwrap().timestep, 1);
        close(&gradient(&p), &before, 0.);
        close(
            &optimizer.state().unwrap().moments[0]
                .as_ref()
                .unwrap()
                .first
                .to_vec(),
            &state.moments[0].as_ref().unwrap().first.to_vec(),
            0.,
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn categorical_autograd_matches_cpu_and_finite_differences() {
    let data = [0.2, -0.7, 1.1, -0.3, 0.5, 0.8];
    let weights = [0.4, -0.5, 0.7, 1.2, 0.1, -0.9];
    let x = variable(&data, &[2, 3], true);
    let w = variable(&weights, &[2, 3], false);
    let objective = x.log_softmax().unwrap().mul(&w).unwrap().sum().unwrap();
    objective.backward().unwrap();
    let cx = Variable::new(Tensor::from_vec(data.to_vec(), &[2, 3]), true);
    let max = Variable::from_tensor(cx.data().max_axis(1, true).unwrap());
    let shifted = &cx - &max;
    let exps = shifted.exp();
    let logs = &shifted - &exps.sum_axis(1, true).log();
    let cpu_loss =
        (&logs * &Variable::from_tensor(Tensor::from_vec(weights.to_vec(), &[2, 3]))).sum();
    cpu_loss.backward();
    close(&gradient(&x), &cx.grad().unwrap().to_vec(), 2e-6);
    let evaluate = |values: &[f32]| -> f64 {
        (0..2)
            .map(|row| {
                let maximum = values[row * 3..row * 3 + 3]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max) as f64;
                let total: f64 = values[row * 3..row * 3 + 3]
                    .iter()
                    .map(|v| (*v as f64 - maximum).exp())
                    .sum();
                (0..3)
                    .map(|col| {
                        let i = row * 3 + col;
                        weights[i] as f64 * (values[i] as f64 - maximum - total.ln())
                    })
                    .sum::<f64>()
            })
            .sum()
    };
    let grads = gradient(&x);
    for i in 0..6 {
        let mut plus = data;
        let mut minus = data;
        plus[i] += 0.001;
        minus[i] -= 0.001;
        approx::assert_abs_diff_eq!(
            grads[i],
            ((evaluate(&plus) - evaluate(&minus)) / (plus[i] - minus[i]) as f64) as f32,
            epsilon = 2e-4
        );
    }
    x.zero_grad();
    x.softmax()
        .unwrap()
        .mul(&w)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    let probabilities = x.softmax().unwrap().to_cpu().unwrap().to_vec();
    let expected: Vec<f32> = (0..6)
        .map(|i| {
            let row = i / 3;
            let sum: f32 = (0..3)
                .map(|c| probabilities[row * 3 + c] * weights[row * 3 + c])
                .sum();
            probabilities[i] * (weights[i] - sum)
        })
        .collect();
    close(&gradient(&x), &expected, 2e-6);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn new_unary_ops_preserve_forward_snapshots_no_grad_and_empty_batches() {
    let x = variable(&[0.1, -0.2, 0.5], &[1, 3], true);
    let exp = x.exp().unwrap();
    let logs = x.log_softmax().unwrap();
    let loss = exp.sum().unwrap().add(&logs.sum().unwrap()).unwrap();
    x.copy_data_from(&variable(&[9., 8., 7.], &[1, 3], false))
        .unwrap();
    loss.backward().unwrap();
    let sum = 0.1f32.exp() + (-0.2f32).exp() + 0.5f32.exp();
    let expected: Vec<f32> = [0.1f32, -0.2, 0.5]
        .iter()
        .map(|v| v.exp() + 1. - 3. * v.exp() / sum)
        .collect();
    close(&gradient(&x), &expected, 2e-6);
    loss.backward().unwrap();
    close(
        &gradient(&x),
        &expected.iter().map(|v| 2. * v).collect::<Vec<_>>(),
        3e-6,
    );
    let frozen = no_grad(|| x.log_softmax().unwrap().exp().unwrap());
    assert!(!frozen.requires_grad() && !frozen.has_grad_fn());
    let empty = variable(&[], &[0, 3], true);
    empty
        .log_softmax()
        .unwrap()
        .exp()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(empty.grad_cpu().unwrap().unwrap().shape(), &[0, 3]);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn log_tanh_division_and_column_sum_match_cpu_and_finite_differences() {
    let a = [0.2, 0.5, 1., 2., 3., 4.];
    let b = [0.7, 1.1, 1.3, 1.5, 1.7, 2.];
    let weights = [0.3, -0.7];
    let x = variable(&a, &[2, 3], true);
    let y = variable(&b, &[2, 3], true);
    let w = variable(&weights, &[2, 1], false);
    let loss = x
        .div(&y)
        .unwrap()
        .log()
        .unwrap()
        .tanh()
        .unwrap()
        .sum_columns()
        .unwrap()
        .mul(&w)
        .unwrap()
        .sum()
        .unwrap();
    loss.backward().unwrap();
    let cx = Variable::new(Tensor::from_vec(a.to_vec(), &[2, 3]), true);
    let cy = Variable::new(Tensor::from_vec(b.to_vec(), &[2, 3]), true);
    let cw = Variable::from_tensor(Tensor::from_vec(weights.to_vec(), &[2, 1]));
    let cl = (&(&cx / &cy).log().tanh_().sum_axis(1, true) * &cw).sum();
    cl.backward();
    close(&gradient(&x), &cx.grad().unwrap().to_vec(), 2e-5);
    close(&gradient(&y), &cy.grad().unwrap().to_vec(), 2e-5);
    let reference = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .enumerate()
            .map(|(i, (a, b))| ((*a as f64) / (*b as f64)).ln().tanh() * (weights[i / 3] as f64))
            .sum()
    };
    for i in 0..6 {
        let mut plus = a;
        let mut minus = a;
        plus[i] += 0.001;
        minus[i] -= 0.001;
        let dx =
            (reference(&plus, &b) - reference(&minus, &b)) / (plus[i] as f64 - minus[i] as f64);
        approx::assert_abs_diff_eq!(gradient(&x)[i] as f64, dx, epsilon = 2e-5);
    }
}
#[test]
#[ignore = "requires a GPU adapter"]
fn continuous_operators_preserve_forward_snapshots_and_empty_no_grad_contracts() {
    let a = variable(&[0.5, 2.], &[1, 2], true);
    let b = variable(&[2., 3.], &[1, 2], true);
    let loss = a
        .div(&b)
        .unwrap()
        .log()
        .unwrap()
        .tanh()
        .unwrap()
        .sum_columns()
        .unwrap()
        .sum()
        .unwrap();
    a.copy_data_from(&variable(&[8., 9.], &[1, 2], false))
        .unwrap();
    b.copy_data_from(&variable(&[5., 6.], &[1, 2], false))
        .unwrap();
    loss.backward().unwrap();
    let first = gradient(&a);
    loss.backward().unwrap();
    close(
        &gradient(&a),
        &first.iter().map(|x| 2. * x).collect::<Vec<_>>(),
        2e-5,
    );
    for (i, (av, bv)) in [(0.5f32, 2f32), (2., 3.)].iter().enumerate() {
        let t = (av / bv).ln().tanh();
        approx::assert_abs_diff_eq!(first[i], (1. - t * t) / av, epsilon = 2e-5);
        approx::assert_abs_diff_eq!(gradient(&b)[i], -2. * (1. - t * t) / bv, epsilon = 2e-5);
    }
    let inference = no_grad(|| {
        a.div(&b)
            .unwrap()
            .log()
            .unwrap()
            .tanh()
            .unwrap()
            .sum_columns()
            .unwrap()
    });
    assert!(!inference.requires_grad() && !inference.has_grad_fn());
    let empty = variable(&[], &[0, 3], true);
    let other = variable(&[], &[0, 3], false);
    empty
        .div(&other)
        .unwrap()
        .log()
        .unwrap()
        .tanh()
        .unwrap()
        .sum_columns()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(gradient(&empty), Vec::<f32>::new());
    let empty_columns = variable(&[], &[2, 0], true);
    empty_columns
        .sum_columns()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert!(gradient(&empty_columns).is_empty());
    assert!(a.div(&variable(&[1.], &[1], false)).is_err());
    assert!(variable(&[1.], &[1], true).sum_columns().is_err());
}
