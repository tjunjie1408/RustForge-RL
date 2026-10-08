#![cfg(feature = "gpu")]
use rustforge_autograd::gpu::GpuVariable;
use rustforge_rl::agent::{
    gpu_sac::{
        sac_actor_loss, sac_critic_loss, sac_temperature_loss, GpuSacCriticInputs, GpuSacError,
        GpuSacLossConfig,
    },
    gpu_td3::{td3_actor_loss, td3_critic_loss, GpuTd3Error, GpuTd3LossConfig},
};
use rustforge_tensor::{gpu::GpuContext, Tensor};

fn variable(context: &GpuContext, values: &[f32]) -> GpuVariable {
    GpuVariable::new(
        context,
        &Tensor::from_vec(values.to_vec(), &[values.len(), 1]),
        false,
    )
    .unwrap()
}

fn one_wait<T>(context: &GpuContext, read: impl FnOnce() -> T) -> T {
    let before = context.profile_snapshot().unwrap().counters;
    let result = read();
    let after = context.profile_snapshot().unwrap().counters;
    assert_eq!(after.host_waits - before.host_waits, 1);
    assert_eq!(after.readbacks - before.readbacks, 1);
    // These small objectives fit one bounded compute group and one readback.
    assert_eq!(after.submissions - before.submissions, 2);
    result
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn all_five_objectives_read_exact_metrics_with_one_wait_each() {
    let context = GpuContext::new().unwrap().with_profiling();
    let q = variable(&context, &[1., 2.]);
    let r = variable(&context, &[0.25, -0.5]);
    let d = variable(&context, &[0., 1.]);
    let lp = variable(&context, &[-0.5, -1.]);
    let la = variable(&context, &[-1.]);
    let td3 = td3_critic_loss(&q, &q, &q, &q, &r, &d, GpuTd3LossConfig::default()).unwrap();
    let actor = td3_actor_loss(&q).unwrap();
    let sac = sac_critic_loss(
        GpuSacCriticInputs {
            q1: &q,
            q2: &q,
            target_q1: &q,
            target_q2: &q,
            next_log_probs: &lp,
            rewards: &r,
            dones: &d,
        },
        GpuSacLossConfig::default(),
    )
    .unwrap();
    let sac_actor = sac_actor_loss(&q, &q, &lp, 0.2).unwrap();
    let temperature = sac_temperature_loss(&la, &lp, -1.).unwrap();
    // Reference reads use the original scalar tensors; batching must preserve bits.
    let expected_td3 = [
        td3.critic1_loss.to_cpu().unwrap().item(),
        td3.critic2_loss.to_cpu().unwrap().item(),
        td3.total_loss.to_cpu().unwrap().item(),
    ];
    let expected_sac = [
        sac.critic1_loss.to_cpu().unwrap().item(),
        sac.critic2_loss.to_cpu().unwrap().item(),
        sac.total_loss.to_cpu().unwrap().item(),
    ];
    let expected_actor = actor.loss.to_cpu().unwrap().item();
    let expected_sac_actor = sac_actor.loss.to_cpu().unwrap().item();
    let expected_temperature = [
        temperature.alpha.to_cpu().unwrap().item(),
        temperature.loss.to_cpu().unwrap().item(),
    ];
    for _ in 0..2 {
        let m = one_wait(&context, || td3.checked_metrics().unwrap());
        assert_eq!(
            [m.critic1_loss, m.critic2_loss, m.total_loss].map(f32::to_bits),
            expected_td3.map(f32::to_bits)
        );
        assert_eq!(
            one_wait(&context, || actor.checked_loss().unwrap()).to_bits(),
            expected_actor.to_bits()
        );
        let m = one_wait(&context, || sac.checked_metrics().unwrap());
        assert_eq!(
            [m.critic1_loss, m.critic2_loss, m.total_loss].map(f32::to_bits),
            expected_sac.map(f32::to_bits)
        );
        assert_eq!(
            one_wait(&context, || sac_actor.checked_loss().unwrap()).to_bits(),
            expected_sac_actor.to_bits()
        );
        let m = one_wait(&context, || temperature.checked_metrics().unwrap());
        assert_eq!(
            [m.alpha, m.loss].map(f32::to_bits),
            expected_temperature.map(f32::to_bits)
        );
    }
    let phase = &context.profile_snapshot().unwrap().phases["objective_diagnostics"];
    assert_eq!(phase.calls, 10);
    assert_eq!(phase.counters.readbacks, 10);
    assert_eq!(phase.counters.submissions, 20);
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn packed_diagnostics_preserve_validation_error_precedence() {
    let context = GpuContext::new().unwrap().with_profiling();
    let q = variable(&context, &[1.]);
    let invalid = variable(&context, &[f32::NAN]);
    let d = variable(&context, &[2.]);
    for (r, nonfinite) in [(&q, false), (&invalid, true)] {
        let td3 = td3_critic_loss(&q, &q, &q, &q, r, &d, GpuTd3LossConfig::default()).unwrap();
        let error = one_wait(&context, || td3.checked_metrics()).unwrap_err();
        if nonfinite {
            assert!(matches!(error, GpuTd3Error::NonFinite));
        } else {
            assert!(matches!(error, GpuTd3Error::InvalidBatch));
        }
        let sac = sac_critic_loss(
            GpuSacCriticInputs {
                q1: &q,
                q2: &q,
                target_q1: &q,
                target_q2: &q,
                next_log_probs: &q,
                rewards: r,
                dones: &d,
            },
            GpuSacLossConfig::default(),
        )
        .unwrap();
        let error = one_wait(&context, || sac.checked_metrics()).unwrap_err();
        if nonfinite {
            assert!(matches!(error, GpuSacError::NonFinite));
        } else {
            assert!(matches!(error, GpuSacError::InvalidBatch));
        }
    }
    let underflow = variable(&context, &[-1000.]);
    let temperature = sac_temperature_loss(&underflow, &q, -1.).unwrap();
    assert!(matches!(
        one_wait(&context, || temperature.checked_metrics()),
        Err(GpuSacError::InvalidTemperature)
    ));
    let temperature = sac_temperature_loss(&underflow, &invalid, -1.).unwrap();
    assert!(matches!(
        one_wait(&context, || temperature.checked_metrics()),
        Err(GpuSacError::NonFinite)
    ));
}

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn externally_replaced_foreign_scalar_keeps_existing_td3_behavior() {
    let context = GpuContext::new().unwrap().with_profiling();
    let q = variable(&context, &[1.]);
    let mut objective = td3_actor_loss(&q).unwrap();
    // Public diagnostic fields historically allow independent-context scalar reads.
    let foreign = GpuContext::new().unwrap();
    objective.loss = variable(&foreign, &[7.]);
    assert_eq!(objective.checked_loss().unwrap(), 7.);
    assert!(!context
        .profile_snapshot()
        .unwrap()
        .phases
        .contains_key("objective_diagnostics"));
}
