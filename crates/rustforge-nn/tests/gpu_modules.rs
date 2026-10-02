#![cfg(feature = "gpu")]
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad,
    optimizer::{adam::Adam, Optimizer},
    Variable,
};
use rustforge_nn::{
    gpu::{GpuLinear, GpuModule, GpuReLU, GpuSequential},
    Linear, Module, ReLU, Sequential,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::{rc::Rc, sync::OnceLock};
fn context() -> &'static GpuContext {
    static CONTEXT: OnceLock<GpuContext> = OnceLock::new();
    CONTEXT.get_or_init(|| GpuContext::new().expect("GPU tests require an adapter"))
}
fn close(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite());
        approx::assert_abs_diff_eq!(a, b, epsilon = epsilon);
    }
}
fn gpu_model() -> GpuSequential {
    GpuSequential::new(vec![
        Box::new(GpuLinear::new_seeded(context(), 2, 4, 42).unwrap()),
        Box::new(GpuReLU),
        Box::new(GpuLinear::new_seeded(context(), 4, 2, 43).unwrap()),
    ])
}

#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_mlp_forward_gradients_and_adam_match_cpu() {
    let gpu = gpu_model();
    assert_eq!(gpu.len(), 3);
    let cpu = Sequential::new(vec![
        Box::new(Linear::new_seeded(2, 4, 42)),
        Box::new(ReLU),
        Box::new(Linear::new_seeded(4, 2, 43)),
    ]);
    let data = Tensor::from_vec(vec![0.2, -0.7, 1.3, 0.6, -0.5, 0.4], &[3, 2]);
    let target = Tensor::from_vec(vec![0.3, -0.1, 0.2, 0.9, -0.6, 0.7], &[3, 2]);
    let gx = GpuVariable::new(context(), &data, true).unwrap();
    let cx = Variable::new(data, true);
    let gy = GpuVariable::new(context(), &target, false).unwrap();
    let cy = Variable::new(target, false);
    let (gp, cp) = (gpu.parameters(), cpu.parameters());
    let mut go = GpuAdam::new(gp.clone(), 0.01).unwrap();
    let mut co = Adam::new(cp.clone(), 0.01);
    for _ in 0..4 {
        go.zero_grad();
        co.zero_grad();
        gx.zero_grad();
        cx.zero_grad();
        let gout = gpu.forward(&gx).unwrap();
        let cout = cpu.forward(&cx);
        close(
            &gout.to_cpu().unwrap().to_vec(),
            &cout.data().to_vec(),
            1e-6,
        );
        gout.mse_loss(&gy).unwrap().backward().unwrap();
        let diff = &cout - &cy;
        (&diff * &diff).mean().backward();
        close(
            &gx.grad_cpu().unwrap().unwrap().to_vec(),
            &cx.grad().unwrap().to_vec(),
            2e-6,
        );
        for (g, c) in gp.iter().zip(&cp) {
            close(
                &g.grad_cpu().unwrap().unwrap().to_vec(),
                &c.grad().unwrap().to_vec(),
                2e-6,
            );
        }
        go.step().unwrap();
        co.step();
        for (g, c) in gp.iter().zip(&cp) {
            close(&g.to_cpu().unwrap().to_vec(), &c.data().to_vec(), 2e-6);
        }
    }
    go.zero_grad();
    let inference = no_grad(|| gpu.forward(&gx).unwrap());
    assert!(!inference.requires_grad() && !inference.has_grad_fn());
    inference.sum().unwrap().backward().unwrap();
    assert!(gp.iter().all(|p| p.grad().is_none()));
}

#[test]
#[ignore = "requires a GPU adapter"]
fn module_initialization_and_shape_contracts() {
    let a = GpuLinear::new_seeded(context(), 2, 3, 11).unwrap();
    let b = GpuLinear::new_seeded(context(), 2, 3, 11).unwrap();
    let different = GpuLinear::new_seeded(context(), 2, 3, 12).unwrap();
    close(
        &a.parameters()[0].to_cpu().unwrap().to_vec(),
        &b.parameters()[0].to_cpu().unwrap().to_vec(),
        0.,
    );
    assert_ne!(
        a.parameters()[0].to_cpu().unwrap().to_vec(),
        different.parameters()[0].to_cpu().unwrap().to_vec()
    );
    assert_eq!((a.in_features(), a.out_features()), (2, 3));
    assert!(GpuLinear::new_seeded(context(), 0, 3, 1).is_err());
    assert!(GpuLinear::new_seeded(context(), usize::MAX, 3, 1).is_err());
    assert!(GpuLinear::from_tensors(context(), &Tensor::ones(&[2]), None).is_err());
    assert!(GpuLinear::from_tensors(
        context(),
        &Tensor::ones(&[3, 2]),
        Some(&Tensor::zeros(&[1, 3]))
    )
    .is_err());
    let no_bias = GpuLinear::no_bias_seeded(context(), 2, 3, 11).unwrap();
    assert_eq!(no_bias.parameters().len(), 1);
    let x = GpuVariable::new(context(), &Tensor::ones(&[2, 2]), false).unwrap();
    close(
        &a.forward(&x).unwrap().to_cpu().unwrap().to_vec(),
        &no_bias.forward(&x).unwrap().to_cpu().unwrap().to_vec(),
        0.,
    );
    let empty = GpuVariable::new(context(), &Tensor::zeros(&[0, 2]), true).unwrap();
    let out = a.forward(&empty).unwrap();
    assert_eq!(out.data().shape(), &[0, 3]);
    out.sum().unwrap().backward().unwrap();
    for p in a.parameters() {
        close(
            &p.grad_cpu().unwrap().unwrap().to_vec(),
            &vec![0.; p.data().numel()],
            0.,
        );
    }
    let invalid = GpuVariable::new(context(), &Tensor::ones(&[2, 1]), false).unwrap();
    assert!(a.forward(&invalid).is_err());
    let sequential = GpuSequential::new(vec![]);
    assert!(sequential.is_empty() && sequential.parameters().is_empty());
    assert!(Rc::ptr_eq(
        &sequential.forward(&x).unwrap().data(),
        &x.data()
    ));
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    let foreign = GpuVariable::new(other, &Tensor::ones(&[1, 2]), false).unwrap();
    assert!(a.forward(&foreign).is_err());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn supplied_linear_parameters_match_cpu_and_propagate_input_gradients() {
    let weight = Tensor::from_vec(vec![1., 2., -3., 0.5], &[2, 2]);
    let bias = Tensor::from_vec(vec![0.7, -1.], &[2]);
    let gpu = GpuLinear::from_tensors(context(), &weight, Some(&bias)).unwrap();
    let cpu = Linear::new_seeded(2, 2, 1);
    let cp = cpu.parameters();
    cp[0].set_data(weight);
    cp[1].set_data(bias);
    let data = Tensor::from_vec(vec![1., -2., 0.5, 3.], &[2, 2]);
    let gx = GpuVariable::new(context(), &data, true).unwrap();
    let cx = Variable::new(data, true);
    let gy = gpu.forward(&gx).unwrap();
    let cy = cpu.forward(&cx);
    close(&gy.to_cpu().unwrap().to_vec(), &cy.data().to_vec(), 0.);
    gy.sum().unwrap().backward().unwrap();
    cy.sum().backward();
    close(
        &gx.grad_cpu().unwrap().unwrap().to_vec(),
        &cx.grad().unwrap().to_vec(),
        0.,
    );
    for (g, c) in gpu.parameters().iter().zip(cp) {
        close(
            &g.grad_cpu().unwrap().unwrap().to_vec(),
            &c.grad().unwrap().to_vec(),
            0.,
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn seeded_nonlinear_training_converges() {
    let x = GpuVariable::new(
        context(),
        &Tensor::from_vec(vec![-2., -1., 0., 1., 2.], &[5, 1]),
        false,
    )
    .unwrap();
    let y = GpuVariable::new(
        context(),
        &Tensor::from_vec(vec![3., 2., 1., 2., 3.], &[5, 1]),
        false,
    )
    .unwrap();
    let model = GpuSequential::new(vec![
        Box::new(GpuLinear::new_seeded(context(), 1, 8, 42).unwrap()),
        Box::new(GpuReLU),
        Box::new(GpuLinear::new_seeded(context(), 8, 1, 43).unwrap()),
    ]);
    let mut opt = GpuAdam::new(model.parameters(), 0.03).unwrap();
    let initial = model
        .forward(&x)
        .unwrap()
        .mse_loss(&y)
        .unwrap()
        .to_cpu()
        .unwrap()
        .item();
    for _ in 0..250 {
        opt.zero_grad();
        model
            .forward(&x)
            .unwrap()
            .mse_loss(&y)
            .unwrap()
            .backward()
            .unwrap();
        opt.step().unwrap();
    }
    let loss = model
        .forward(&x)
        .unwrap()
        .mse_loss(&y)
        .unwrap()
        .to_cpu()
        .unwrap()
        .item();
    assert!(
        loss.is_finite() && loss < 0.001 && loss < initial * 0.01,
        "MSE {initial} -> {loss}"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn frozen_module_snapshots_sync_without_online_aliasing_or_partial_updates() {
    let layer = GpuLinear::new_seeded(context(), 2, 3, 7).unwrap();
    let target = GpuSequential::new(vec![Box::new(layer.frozen_snapshot())]);
    let online = GpuSequential::new(vec![Box::new(layer)]);
    let original = online.parameters()[0].to_cpu().unwrap().to_vec();
    assert!(target
        .parameters()
        .iter()
        .all(|p| !p.requires_grad() && !p.has_grad_fn()));
    assert!(Rc::ptr_eq(
        &online.parameters()[0].data(),
        &target.parameters()[0].data()
    ));
    let mut opt = GpuAdam::new(online.parameters(), 0.1).unwrap();
    online.parameters()[0].sum().unwrap().backward().unwrap();
    opt.step().unwrap();
    close(
        &target.parameters()[0].to_cpu().unwrap().to_vec(),
        &original,
        0.,
    );
    assert!(!Rc::ptr_eq(
        &online.parameters()[0].data(),
        &target.parameters()[0].data()
    ));
    target.copy_parameters_from(&online).unwrap();
    for (a, b) in target.parameters().iter().zip(online.parameters()) {
        assert!(Rc::ptr_eq(&a.data(), &b.data()));
        assert!(!a.requires_grad() && a.grad().is_none());
    }
    let before = target.parameters()[0].to_cpu().unwrap().to_vec();
    let no_bias = GpuSequential::new(vec![Box::new(
        GpuLinear::no_bias_seeded(context(), 2, 3, 9).unwrap(),
    )]);
    assert!(target.copy_parameters_from(&no_bias).is_err());
    let wrong_shape = GpuSequential::new(vec![Box::new(
        GpuLinear::new_seeded(context(), 2, 4, 9).unwrap(),
    )]);
    assert!(target.copy_parameters_from(&wrong_shape).is_err());
    static OTHER: OnceLock<GpuContext> = OnceLock::new();
    let other = OTHER.get_or_init(|| GpuContext::new().unwrap());
    let foreign = GpuSequential::new(vec![Box::new(
        GpuLinear::new_seeded(other, 2, 3, 9).unwrap(),
    )]);
    assert!(target.copy_parameters_from(&foreign).is_err());
    close(
        &target.parameters()[0].to_cpu().unwrap().to_vec(),
        &before,
        0.,
    );
}
