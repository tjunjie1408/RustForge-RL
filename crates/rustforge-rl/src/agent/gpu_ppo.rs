//! Device loss foundation for discrete PPO. Rollout references are detached;
//! CPU rollout/GAE and the discrete actor/critic trainer are provided by `agent`.
mod agent;
mod continuous;
mod continuous_agent;
pub use agent::{
    GpuActorCriticNet, GpuPpoCheckpointError, GpuPpoDiscrete, GpuPpoError, CHECKPOINT_MAGIC,
    CHECKPOINT_VERSION, MAX_CHECKPOINT_BYTES,
};
pub use continuous::{
    continuous_ppo_loss, GpuContinuousPpoInputs, GpuContinuousPpoLoss, GpuContinuousPpoMetrics,
};
pub use continuous_agent::{
    GpuContinuousRolloutOptions, GpuGaussianPolicyNet, GpuPpoContinuous,
    CONTINUOUS_CHECKPOINT_MAGIC, CONTINUOUS_CHECKPOINT_VERSION,
};
use rustforge_autograd::gpu::{GpuAutogradError, GpuVariable};
use rustforge_tensor::gpu::{GpuError, GpuIndices};
use std::{error::Error, fmt, rc::Rc};

#[derive(Clone, Copy, Debug)]
pub struct GpuPpoLossConfig {
    pub clip_eps: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
}
impl Default for GpuPpoLossConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
        }
    }
}
impl GpuPpoLossConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.clip_eps.is_finite()
            || !(0.0..1.0).contains(&self.clip_eps)
            || !self.value_coef.is_finite()
            || self.value_coef < 0.
            || !self.entropy_coef.is_finite()
            || self.entropy_coef < 0.
        {
            return Err(GpuPpoLossError::InvalidConfig);
        }
        Ok(())
    }
}
#[derive(Debug)]
pub enum GpuPpoLossError {
    Autograd(GpuAutogradError),
    Device(GpuError),
    InvalidConfig,
    InvalidBatch,
    NonFinite,
    Gaussian(crate::agent::gpu_gaussian::GpuGaussianError),
    InvalidContinuousBatch,
}
impl fmt::Display for GpuPpoLossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Autograd(error) => error.fmt(f),
            Self::Device(error) => error.fmt(f),
            Self::InvalidConfig => f.write_str("PPO clip epsilon must be finite in [0, 1); loss coefficients must be finite and nonnegative"),
            Self::InvalidBatch => f.write_str("PPO losses require nonempty [batch, actions] logits, matching typed actions and [batch, 1] values/references"),
            Self::Gaussian(e) => e.fmt(f),
            Self::InvalidContinuousBatch => f.write_str("continuous PPO requires nonempty [batch,actions] means/actions/std and [batch,1] values/references"),
            Self::NonFinite => f.write_str("GPU PPO objective is nonfinite"),
        }
    }
}
impl Error for GpuPpoLossError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Autograd(e) => Some(e),
            Self::Device(e) => Some(e),
            Self::Gaussian(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuAutogradError> for GpuPpoLossError {
    fn from(error: GpuAutogradError) -> Self {
        Self::Autograd(error)
    }
}
impl From<GpuError> for GpuPpoLossError {
    fn from(error: GpuError) -> Self {
        Self::Device(error)
    }
}
impl From<crate::agent::gpu_gaussian::GpuGaussianError> for GpuPpoLossError {
    fn from(error: crate::agent::gpu_gaussian::GpuGaussianError) -> Self {
        Self::Gaussian(error)
    }
}
type Result<T> = std::result::Result<T, GpuPpoLossError>;

pub struct GpuPpoPolicyLoss {
    pub policy_loss: GpuVariable,
    pub entropy: GpuVariable,
    pub new_log_probs: GpuVariable,
    pub ratios: GpuVariable,
}
pub struct GpuPpoLoss {
    pub policy: GpuPpoPolicyLoss,
    pub value_loss: GpuVariable,
    pub total_loss: GpuVariable,
}
#[derive(Clone, Copy, Debug)]
pub struct GpuPpoMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub total_loss: f32,
}
impl GpuPpoLoss {
    /// Explicit diagnostic readback. Call before backward/optimizer updates to
    /// reject nonfinite objectives; constructing the losses performs no readback.
    pub fn checked_metrics(&self) -> Result<GpuPpoMetrics> {
        // Clipping can hide an infinite importance ratio in a finite loss, but
        // its exp derivative would still contaminate gradients (infinity * 0).
        let context = self.policy.ratios.context();
        let invalid = context.nonfinite_count_device(&self.policy.ratios.data())?;
        if context.download(&invalid)?.item() != 0. {
            return Err(GpuPpoLossError::NonFinite);
        }
        let metrics = GpuPpoMetrics {
            policy_loss: self.policy.policy_loss.to_cpu()?.item(),
            value_loss: self.value_loss.to_cpu()?.item(),
            entropy: self.policy.entropy.to_cpu()?.item(),
            total_loss: self.total_loss.to_cpu()?.item(),
        };
        if [
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.total_loss,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err(GpuPpoLossError::NonFinite);
        }
        Ok(metrics)
    }
}

/// Clipped policy surrogate and mean categorical entropy on device. Reference
/// old log probabilities and advantages are detached even if supplied as trainable
/// variables. Action bounds and ownership use the typed gather contract.
pub fn categorical_policy_loss(
    logits: &GpuVariable,
    actions: &Rc<GpuIndices>,
    old_log_probs: &GpuVariable,
    advantages: &GpuVariable,
    clip_eps: f32,
) -> Result<GpuPpoPolicyLoss> {
    if !clip_eps.is_finite() || !(0.0..1.0).contains(&clip_eps) {
        return Err(GpuPpoLossError::InvalidConfig);
    }
    let shape = logits.data();
    let shape = shape.shape();
    if shape.len() != 2
        || shape[0] == 0
        || shape[1] == 0
        || actions.len() != shape[0]
        || actions.columns() != shape[1]
    {
        return Err(GpuPpoLossError::InvalidBatch);
    }
    for variable in [old_log_probs, advantages] {
        if variable.data().shape() != [shape[0], 1] {
            return Err(GpuPpoLossError::InvalidBatch);
        }
        if !logits.context().is_compatible(&variable.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
    }
    let log_probs = logits.log_softmax()?;
    let new_log_probs = log_probs.gather_actions(actions)?;
    let ratios = new_log_probs.sub(&old_log_probs.detach())?.exp()?;
    let advantages = advantages.detach();
    let first = ratios.mul(&advantages)?;
    let second = ratios
        .clamp(1.0 - clip_eps, 1.0 + clip_eps)?
        .mul(&advantages)?;
    let policy_loss = first.minimum(&second)?.mean()?.scale(-1.)?;
    let entropy = log_probs
        .exp()?
        .mul(&log_probs)?
        .sum()?
        .scale(-1. / shape[0] as f32)?;
    Ok(GpuPpoPolicyLoss {
        policy_loss,
        entropy,
        new_log_probs,
        ratios,
    })
}

/// CPU PPO's total loss: clipped policy + value_coef * MSE - entropy_coef * H.
/// Returns are detached. This function builds a device graph; it does not update
/// parameters or collect rollouts. Use checked_metrics before applying updates.
pub fn discrete_ppo_loss(
    logits: &GpuVariable,
    values: &GpuVariable,
    actions: &Rc<GpuIndices>,
    old_log_probs: &GpuVariable,
    advantages: &GpuVariable,
    returns: &GpuVariable,
    config: GpuPpoLossConfig,
) -> Result<GpuPpoLoss> {
    config.validate()?;
    let shape = logits.data();
    let shape = shape.shape();
    if shape.len() != 2 {
        return Err(GpuPpoLossError::InvalidBatch);
    }
    for variable in [values, returns] {
        if variable.data().shape() != [shape[0], 1] {
            return Err(GpuPpoLossError::InvalidBatch);
        }
        if !logits.context().is_compatible(&variable.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
    }
    let policy =
        categorical_policy_loss(logits, actions, old_log_probs, advantages, config.clip_eps)?;
    let value_loss = values.mse_loss(&returns.detach())?;
    let total_loss = policy
        .policy_loss
        .add(&value_loss.scale(config.value_coef)?)?
        .sub(&policy.entropy.scale(config.entropy_coef)?)?;
    Ok(GpuPpoLoss {
        policy,
        value_loss,
        total_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loss_configuration_rejects_invalid_clipping_and_coefficients_without_adapter() {
        assert!(GpuPpoLossConfig::default().validate().is_ok());
        assert!(GpuPpoLossConfig {
            clip_eps: 0.,
            value_coef: 0.,
            entropy_coef: 0.
        }
        .validate()
        .is_ok());
        for value in [-1., f32::NAN, f32::INFINITY] {
            for field in 0..3 {
                let mut config = GpuPpoLossConfig::default();
                match field {
                    0 => config.clip_eps = value,
                    1 => config.value_coef = value,
                    _ => config.entropy_coef = value,
                }
                assert!(config.validate().is_err());
            }
        }
        assert!(GpuPpoLossConfig {
            clip_eps: 1.,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
}
