//! Continuous PPO objective matching CPU's separate policy and value updates.
use super::GpuPpoLossError;
use crate::agent::gpu_gaussian::{GpuGaussianLogProb, GpuGaussianTransform};
use rustforge_autograd::gpu::GpuVariable;
use rustforge_tensor::gpu::GpuError;
type Result<T> = std::result::Result<T, GpuPpoLossError>;
pub struct GpuContinuousPpoInputs<'a> {
    pub mean: &'a GpuVariable,
    pub raw_log_std: &'a GpuVariable,
    pub values: &'a GpuVariable,
    pub actions: &'a GpuVariable,
    pub old_log_probs: &'a GpuVariable,
    pub advantages: &'a GpuVariable,
    pub returns: &'a GpuVariable,
}
pub struct GpuContinuousPpoLoss {
    pub policy_loss: GpuVariable,
    pub value_loss: GpuVariable,
    pub gaussian: GpuGaussianLogProb,
    pub ratios: GpuVariable,
    references: Vec<GpuVariable>,
}
#[derive(Clone, Copy, Debug)]
pub struct GpuContinuousPpoMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub base_entropy: f32,
}
impl GpuContinuousPpoLoss {
    pub fn checked_metrics(&self) -> Result<GpuContinuousPpoMetrics> {
        let gaussian = self.gaussian.checked_metrics()?;
        let context = self.ratios.context();
        for v in self
            .references
            .iter()
            .chain([&self.ratios, &self.policy_loss, &self.value_loss])
        {
            let invalid = context.nonfinite_count_device(&v.data())?;
            if context.download(&invalid)?.item() != 0. {
                return Err(GpuPpoLossError::NonFinite);
            }
        }
        Ok(GpuContinuousPpoMetrics {
            policy_loss: self.policy_loss.to_cpu()?.item(),
            value_loss: self.value_loss.to_cpu()?.item(),
            base_entropy: gaussian.base_entropy,
        })
    }
}
/// Builds device policy/value graphs, detaching all rollout references. There is
/// no entropy bonus or value coefficient, matching current CPU PPOContinuous.
pub fn continuous_ppo_loss(
    inputs: GpuContinuousPpoInputs<'_>,
    transform: &GpuGaussianTransform,
    clip_eps: f32,
) -> Result<GpuContinuousPpoLoss> {
    if !clip_eps.is_finite() || !(0. ..1.).contains(&clip_eps) {
        return Err(GpuPpoLossError::InvalidConfig);
    }
    let shape = inputs.mean.data().shape().to_vec();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(GpuPpoLossError::InvalidContinuousBatch);
    }
    for variable in [
        inputs.values,
        inputs.old_log_probs,
        inputs.advantages,
        inputs.returns,
    ] {
        if variable.data().shape() != [shape[0], 1] {
            return Err(GpuPpoLossError::InvalidContinuousBatch);
        }
        if !inputs.mean.context().is_compatible(&variable.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
    }
    let gaussian =
        transform.log_prob_from_action(inputs.mean, inputs.raw_log_std, inputs.actions)?;
    let ratios = gaussian
        .log_probs
        .sub(&inputs.old_log_probs.detach())?
        .exp()?;
    let advantages = inputs.advantages.detach();
    let first = ratios.mul(&advantages)?;
    let second = ratios
        .clamp(1. - clip_eps, 1. + clip_eps)?
        .mul(&advantages)?;
    let policy_loss = first.minimum(&second)?.mean()?.scale(-1.)?;
    let value_loss = inputs.values.mse_loss(&inputs.returns.detach())?;
    Ok(GpuContinuousPpoLoss {
        policy_loss,
        value_loss,
        gaussian,
        ratios,
        references: [
            inputs.values,
            inputs.old_log_probs,
            inputs.advantages,
            inputs.returns,
        ]
        .iter()
        .map(|v| v.detach())
        .collect(),
    })
}
