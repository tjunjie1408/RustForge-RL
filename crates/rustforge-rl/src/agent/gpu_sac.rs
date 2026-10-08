//! GPU SAC objectives with caller-supplied policy samples and critic estimates.
//! Owned replay training is provided by GpuSac; runtime integration follows separately.
mod agent;
/// Reparameterized tanh-squashed Gaussian sampling, including action-scale density correction.
pub use super::gpu_gaussian::GpuGaussianSample;
use super::gpu_gaussian::{GpuGaussianError, GpuGaussianTransform};
pub use agent::{
    GpuSac, GpuSacCheckpointError, GpuSacCritic, GpuSacPolicy, CHECKPOINT_MAGIC,
    CHECKPOINT_VERSION, MAX_CHECKPOINT_BYTES,
};
use rustforge_autograd::gpu::{GpuAutogradError, GpuVariable};
use rustforge_tensor::gpu::{GpuContext, GpuError};
use std::{error::Error, fmt};
/// SAC's nonempty-batch contract around the shared Gaussian sampling formula.
pub struct GpuSacActionTransform {
    transform: GpuGaussianTransform,
}
impl GpuSacActionTransform {
    pub fn new(context: &GpuContext, low: &[f32], high: &[f32]) -> Result<Self> {
        Ok(Self {
            transform: GpuGaussianTransform::new(context, low, high)?,
        })
    }
    /// Standard-normal noise is detached; action/density gradients reach mean and std.
    /// Call checked_metrics on the returned sample before backward.
    pub fn sample_with_noise(
        &self,
        mean: &GpuVariable,
        raw_log_std: &GpuVariable,
        noise: &GpuVariable,
    ) -> Result<GpuGaussianSample> {
        let shape = mean.data().shape().to_vec();
        if shape.len() != 2 || shape[0] == 0 {
            return Err(GpuSacError::InvalidBatch);
        }
        Ok(self.transform.sample_with_noise(mean, raw_log_std, noise)?)
    }
}

#[derive(Debug)]
pub enum GpuSacError {
    Checkpoint(GpuSacCheckpointError),
    Autograd(GpuAutogradError),
    Gaussian(GpuGaussianError),
    Module(rustforge_nn::gpu::GpuModuleError),
    Device(GpuError),
    InvalidConfig,
    InvalidBatch,
    InvalidTemperature,
    NonFinite,
}
impl fmt::Display for GpuSacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(e) => e.fmt(f),
            Self::Module(e) => e.fmt(f),
            Self::Gaussian(e) => e.fmt(f),
            Self::Autograd(e) => e.fmt(f), Self::Device(e) => e.fmt(f),
            Self::InvalidConfig => f.write_str("SAC requires compatible positive dimensions, finite positive learning rates/initial temperature, valid bounds, discount/tau in [0,1] and finite entropy configuration"),
            Self::InvalidBatch => f.write_str("SAC requires matching nonempty [batch,1] matrices, masks in [0,1] and a scalar log temperature"),
            Self::InvalidTemperature => f.write_str("SAC exp(log_alpha) must be finite and strictly positive"),
            Self::NonFinite => f.write_str("GPU SAC inputs, intermediates or objectives are nonfinite"),
        }
    }
}
impl Error for GpuSacError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Checkpoint(e) => Some(e),
            Self::Module(e) => Some(e),
            Self::Gaussian(e) => Some(e),
            Self::Autograd(e) => Some(e),
            Self::Device(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuAutogradError> for GpuSacError {
    fn from(e: GpuAutogradError) -> Self {
        Self::Autograd(e)
    }
}
impl From<GpuError> for GpuSacError {
    fn from(e: GpuError) -> Self {
        Self::Device(e)
    }
}
impl From<GpuGaussianError> for GpuSacError {
    fn from(e: GpuGaussianError) -> Self {
        Self::Gaussian(e)
    }
}
pub type Result<T> = std::result::Result<T, GpuSacError>;
#[derive(Clone, Copy, Debug)]
pub struct GpuSacLossConfig {
    pub gamma: f32,
    pub alpha: f32,
}
impl Default for GpuSacLossConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            alpha: 0.2,
        }
    }
}
impl GpuSacLossConfig {
    pub fn validate(&self) -> Result<()> {
        validate_alpha(self.alpha)?;
        if !self.gamma.is_finite() || !(0. ..=1.).contains(&self.gamma) {
            return Err(GpuSacError::InvalidConfig);
        }
        Ok(())
    }
}
fn validate_alpha(alpha: f32) -> Result<()> {
    if !alpha.is_finite() || alpha < 0. {
        Err(GpuSacError::InvalidConfig)
    } else {
        Ok(())
    }
}
fn matrix_group(first: &GpuVariable, others: &[&GpuVariable]) -> Result<()> {
    let data = first.data();
    let shape = data.shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] != 1 {
        return Err(GpuSacError::InvalidBatch);
    }
    for v in others {
        if v.data().shape() != shape {
            return Err(GpuSacError::InvalidBatch);
        }
        if !first.context().is_compatible(&v.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
    }
    Ok(())
}
fn finite(context: &GpuContext, variables: impl IntoIterator<Item = GpuVariable>) -> Result<()> {
    let _profile = context.profile_scope("finite_validation");
    let _batch = context.command_batch();
    let mut count = None;
    for v in variables {
        let next = context.nonfinite_count_device(&v.data())?;
        count = Some(match count {
            Some(old) => context.add_device(&old, &next)?,
            None => next,
        });
    }
    if let Some(count) = count {
        if context.download(&count)?.item() != 0. {
            return Err(GpuSacError::NonFinite);
        }
    }
    Ok(())
}
/// Live current critics, and detached next-state target estimates/log probabilities.
pub struct GpuSacCriticInputs<'a> {
    pub q1: &'a GpuVariable,
    pub q2: &'a GpuVariable,
    pub target_q1: &'a GpuVariable,
    pub target_q2: &'a GpuVariable,
    pub next_log_probs: &'a GpuVariable,
    pub rewards: &'a GpuVariable,
    pub dones: &'a GpuVariable,
}
pub struct GpuSacCriticLoss {
    pub target_values: GpuVariable,
    pub critic1_loss: GpuVariable,
    pub critic2_loss: GpuVariable,
    pub total_loss: GpuVariable,
    mask_violation: GpuVariable,
    inputs: Vec<GpuVariable>,
}
#[derive(Clone, Copy, Debug)]
pub struct GpuSacCriticMetrics {
    pub critic1_loss: f32,
    pub critic2_loss: f32,
    pub total_loss: f32,
}
impl GpuSacCriticLoss {
    /// Explicit scalar validation/readback before backward. Check gradients/squares before Adam.
    pub fn checked_metrics(&self) -> Result<GpuSacCriticMetrics> {
        let checks: Vec<_> = self
            .inputs
            .iter()
            .chain([
                &self.target_values,
                &self.critic1_loss,
                &self.critic2_loss,
                &self.total_loss,
            ])
            .cloned()
            .collect();
        if let Some(v) = super::gpu_diagnostics::checked_scalars(
            self.total_loss.context(),
            &checks,
            &[
                &self.mask_violation,
                &self.critic1_loss,
                &self.critic2_loss,
                &self.total_loss,
            ],
        )? {
            if v[0] != 0. {
                return Err(GpuSacError::NonFinite);
            }
            if v[1] != 0. {
                return Err(GpuSacError::InvalidBatch);
            }
            return Ok(GpuSacCriticMetrics {
                critic1_loss: v[2],
                critic2_loss: v[3],
                total_loss: v[4],
            });
        }
        finite(
            self.total_loss.context(),
            self.inputs
                .iter()
                .chain([
                    &self.target_values,
                    &self.critic1_loss,
                    &self.critic2_loss,
                    &self.total_loss,
                ])
                .cloned(),
        )?;
        if self.mask_violation.to_cpu()?.item() != 0. {
            return Err(GpuSacError::InvalidBatch);
        }
        Ok(GpuSacCriticMetrics {
            critic1_loss: self.critic1_loss.to_cpu()?.item(),
            critic2_loss: self.critic2_loss.to_cpu()?.item(),
            total_loss: self.total_loss.to_cpu()?.item(),
        })
    }
}
/// y = detach(r + gamma*(1-done)*(min(target_q1,target_q2) - alpha*next_log_prob));
/// loss = MSE(q1,y) + MSE(q2,y). No gradient reaches any target/reward/mask.
/// Fractional masks in [0,1] preserve CPU arithmetic. Alpha is a cached scalar.
pub fn sac_critic_loss(
    inputs: GpuSacCriticInputs<'_>,
    config: GpuSacLossConfig,
) -> Result<GpuSacCriticLoss> {
    config.validate()?;
    let GpuSacCriticInputs {
        q1,
        q2,
        target_q1,
        target_q2,
        next_log_probs,
        rewards,
        dones,
    } = inputs;
    matrix_group(
        q1,
        &[q2, target_q1, target_q2, next_log_probs, rewards, dones],
    )?;
    let context = q1.context();
    let ones = GpuVariable::from_device(context, context.full(q1.data().shape(), 1.)?, false)?;
    let mask = dones.detach();
    let mask_violation = mask
        .scale(-1.)?
        .relu()?
        .add(&mask.sub(&ones)?.relu()?)?
        .sum()?;
    let minimum = target_q1.detach().minimum(&target_q2.detach())?;
    // Preserve CPU SAC's negative-alpha multiplication followed by addition.
    let entropy_bonus = next_log_probs.detach().scale(-config.alpha)?;
    let soft_values = minimum.add(&entropy_bonus)?;
    let target_values = rewards
        .detach()
        .add(&ones.sub(&mask)?.mul(&soft_values)?.scale(config.gamma)?)?
        .detach();
    let critic1_loss = q1.mse_loss(&target_values)?;
    let critic2_loss = q2.mse_loss(&target_values)?;
    let total_loss = critic1_loss.add(&critic2_loss)?;
    let mut snapshots: Vec<_> = [q1, q2, target_q1, target_q2, next_log_probs, rewards, dones]
        .into_iter()
        .map(GpuVariable::detach)
        .collect();
    snapshots.extend([
        minimum.detach(),
        entropy_bonus.detach(),
        soft_values.detach(),
    ]);
    Ok(GpuSacCriticLoss {
        target_values,
        critic1_loss,
        critic2_loss,
        total_loss,
        mask_violation,
        inputs: snapshots,
    })
}
pub struct GpuSacActorLoss {
    pub loss: GpuVariable,
    inputs: Vec<GpuVariable>,
}
impl GpuSacActorLoss {
    pub fn checked_loss(&self) -> Result<f32> {
        let checks: Vec<_> = self.inputs.iter().chain([&self.loss]).cloned().collect();
        if let Some(v) =
            super::gpu_diagnostics::checked_scalars(self.loss.context(), &checks, &[&self.loss])?
        {
            if v[0] != 0. {
                return Err(GpuSacError::NonFinite);
            }
            return Ok(v[1]);
        }
        finite(
            self.loss.context(),
            self.inputs.iter().chain([&self.loss]).cloned(),
        )?;
        Ok(self.loss.to_cpu()?.item())
    }
}
/// mean(alpha*log_prob - min(Q1,Q2)); actor sampling and both critic graphs stay live.
/// Caller freezes critic weights while retaining their action derivatives. Equal
/// critic values use CPU's Q2 derivative convention, not an averaged subgradient.
pub fn sac_actor_loss(
    q1: &GpuVariable,
    q2: &GpuVariable,
    log_probs: &GpuVariable,
    alpha: f32,
) -> Result<GpuSacActorLoss> {
    validate_alpha(alpha)?;
    matrix_group(q1, &[q2, log_probs])?;
    let minimum = q1.minimum(q2)?;
    let entropy_term = log_probs.scale(alpha)?;
    let per_row = entropy_term.sub(&minimum)?;
    let loss = per_row.mean()?;
    Ok(GpuSacActorLoss {
        loss,
        inputs: [q1, q2, log_probs, &minimum, &entropy_term, &per_row]
            .into_iter()
            .map(GpuVariable::detach)
            .collect(),
    })
}
pub struct GpuSacTemperatureLoss {
    pub loss: GpuVariable,
    /// Cached detached temperature for the current critic/actor update; recompute after Adam.
    pub alpha: GpuVariable,
    inputs: Vec<GpuVariable>,
}
#[derive(Clone, Copy, Debug)]
pub struct GpuSacTemperatureMetrics {
    pub loss: f32,
    pub alpha: f32,
}
impl GpuSacTemperatureLoss {
    pub fn checked_metrics(&self) -> Result<GpuSacTemperatureMetrics> {
        let checks: Vec<_> = self.inputs.iter().chain([&self.loss]).cloned().collect();
        if let Some(v) = super::gpu_diagnostics::checked_scalars(
            self.loss.context(),
            &checks,
            &[&self.alpha, &self.loss],
        )? {
            if v[0] != 0. {
                return Err(GpuSacError::NonFinite);
            }
            if !v[1].is_finite() || v[1] <= 0. {
                return Err(GpuSacError::InvalidTemperature);
            }
            return Ok(GpuSacTemperatureMetrics {
                loss: v[2],
                alpha: v[1],
            });
        }
        finite(
            self.loss.context(),
            self.inputs.iter().chain([&self.loss]).cloned(),
        )?;
        let alpha = self.alpha.to_cpu()?.item();
        if !alpha.is_finite() || alpha <= 0. {
            return Err(GpuSacError::InvalidTemperature);
        }
        Ok(GpuSacTemperatureMetrics {
            loss: self.loss.to_cpu()?.item(),
            alpha,
        })
    }
}
/// -log_alpha * mean(detach(log_prob) + target_entropy). Only log_alpha gets gradients.
/// CPU SAC uses target_entropy=-act_dim; callers may supply any finite target.
pub fn sac_temperature_loss(
    log_alpha: &GpuVariable,
    log_probs: &GpuVariable,
    target_entropy: f32,
) -> Result<GpuSacTemperatureLoss> {
    if !target_entropy.is_finite() {
        return Err(GpuSacError::InvalidConfig);
    }
    matrix_group(log_probs, &[])?;
    if log_alpha.data().numel() != 1 {
        return Err(GpuSacError::InvalidBatch);
    }
    if !log_alpha.context().is_compatible(&log_probs.data()) {
        return Err(GpuError::DeviceMismatch.into());
    }
    let context = log_probs.context();
    let constant = GpuVariable::from_device(
        context,
        context.full(log_probs.data().shape(), target_entropy)?,
        false,
    )?;
    let error = log_probs.detach().add(&constant)?;
    let mean_error = error.mean()?;
    // Broadcast the single-element mean to the scalar parameter's logical shape.
    let shaped_error = GpuVariable::from_device(
        context,
        context.broadcast_scalar_device(&mean_error.data(), log_alpha.data().shape(), 1.)?,
        false,
    )?;
    let loss = log_alpha.mul(&shaped_error)?.scale(-1.)?;
    let alpha = log_alpha.detach().exp()?;
    Ok(GpuSacTemperatureLoss {
        loss,
        alpha,
        inputs: [log_alpha, log_probs, &error, &mean_error]
            .into_iter()
            .map(GpuVariable::detach)
            .collect(),
    })
}

impl From<rustforge_nn::gpu::GpuModuleError> for GpuSacError {
    fn from(e: rustforge_nn::gpu::GpuModuleError) -> Self {
        Self::Module(e)
    }
}

impl From<GpuSacCheckpointError> for GpuSacError {
    fn from(e: GpuSacCheckpointError) -> Self {
        Self::Checkpoint(e)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loss_config_rejects_invalid_discounts_and_temperature_without_adapter() {
        GpuSacLossConfig::default().validate().unwrap();
        for gamma in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            assert!(GpuSacLossConfig { gamma, alpha: 0.2 }.validate().is_err());
        }
        for alpha in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(GpuSacLossConfig { gamma: 0.99, alpha }.validate().is_err());
        }
        for gamma in [0., 1.] {
            GpuSacLossConfig { gamma, alpha: 0. }.validate().unwrap();
        }
    }
}
