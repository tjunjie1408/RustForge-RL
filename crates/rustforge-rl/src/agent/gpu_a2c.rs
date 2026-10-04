//! GPU A2C objectives and agent; environment sampling/rollout/GAE stay on CPU.
mod agent;
pub use agent::{
    GpuA2c, GpuA2cCheckpointError, GpuA2cError, GpuA2cRolloutOptions, CHECKPOINT_MAGIC,
    CHECKPOINT_VERSION, MAX_CHECKPOINT_BYTES,
};
use rustforge_autograd::gpu::{GpuAutogradError, GpuVariable};
use rustforge_tensor::gpu::{GpuError, GpuIndices};
use std::{error::Error, fmt, rc::Rc};

/// Shared trunk and actor/value heads, matching CPU A2C's architecture and seeds.
pub use super::gpu_ppo::GpuActorCriticNet as GpuA2cNet;

#[derive(Clone, Copy, Debug)]
pub struct GpuA2cLossConfig {
    pub value_coef: f32,
    pub entropy_coef: f32,
}
impl Default for GpuA2cLossConfig {
    fn default() -> Self {
        Self {
            value_coef: 0.5,
            entropy_coef: 0.01,
        }
    }
}
impl GpuA2cLossConfig {
    pub fn validate(&self) -> Result<()> {
        if [self.value_coef, self.entropy_coef]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
        {
            return Err(GpuA2cLossError::InvalidConfig);
        }
        Ok(())
    }
}
#[derive(Debug)]
pub enum GpuA2cLossError {
    Autograd(GpuAutogradError),
    Device(GpuError),
    InvalidConfig,
    InvalidBatch,
    NonFinite,
}
impl fmt::Display for GpuA2cLossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Autograd(e) => e.fmt(f),
            Self::Device(e) => e.fmt(f),
            Self::InvalidConfig => f.write_str("A2C loss coefficients must be finite and nonnegative"),
            Self::InvalidBatch => f.write_str(
                "A2C requires nonempty [batch,actions] logits, matching typed actions and [batch,1] values/advantages/returns",
            ),
            Self::NonFinite => f.write_str("GPU A2C input or objective is nonfinite"),
        }
    }
}
impl Error for GpuA2cLossError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Autograd(e) => Some(e),
            Self::Device(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuAutogradError> for GpuA2cLossError {
    fn from(e: GpuAutogradError) -> Self {
        Self::Autograd(e)
    }
}
impl From<GpuError> for GpuA2cLossError {
    fn from(e: GpuError) -> Self {
        Self::Device(e)
    }
}
type Result<T> = std::result::Result<T, GpuA2cLossError>;

pub struct GpuA2cLoss {
    pub actor_loss: GpuVariable,
    pub value_loss: GpuVariable,
    pub entropy: GpuVariable,
    pub total_loss: GpuVariable,
    pub selected_log_probs: GpuVariable,
    inputs: Vec<GpuVariable>,
}
#[derive(Clone, Copy, Debug)]
pub struct GpuA2cMetrics {
    pub actor_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub total_loss: f32,
}
impl GpuA2cLoss {
    /// Explicit scalar readbacks. Check before backward and check resulting
    /// gradients before Adam; finite forward metrics do not guarantee finite gradients.
    pub fn checked_metrics(&self) -> Result<GpuA2cMetrics> {
        let context = self.total_loss.context();
        for v in &self.inputs {
            let invalid = context.nonfinite_count_device(&v.data())?;
            if context.download(&invalid)?.item() != 0. {
                return Err(GpuA2cLossError::NonFinite);
            }
        }
        let metrics = GpuA2cMetrics {
            actor_loss: self.actor_loss.to_cpu()?.item(),
            value_loss: self.value_loss.to_cpu()?.item(),
            entropy: self.entropy.to_cpu()?.item(),
            total_loss: self.total_loss.to_cpu()?.item(),
        };
        if [
            metrics.actor_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.total_loss,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err(GpuA2cLossError::NonFinite);
        }
        Ok(metrics)
    }
}
/// CPU A2C: -mean(log pi(a)*advantages) + value_coef*MSE - entropy_coef*H.
/// Advantages/returns are detached snapshots, without normalization or importance
/// ratios. This builds a resident graph and performs no host readback or update.
pub fn a2c_loss(
    logits: &GpuVariable,
    values: &GpuVariable,
    actions: &Rc<GpuIndices>,
    advantages: &GpuVariable,
    returns: &GpuVariable,
    config: GpuA2cLossConfig,
) -> Result<GpuA2cLoss> {
    config.validate()?;
    let shape = logits.data().shape().to_vec();
    if shape.len() != 2
        || shape[0] == 0
        || shape[1] == 0
        || actions.len() != shape[0]
        || actions.columns() != shape[1]
    {
        return Err(GpuA2cLossError::InvalidBatch);
    }
    for variable in [values, advantages, returns] {
        if variable.data().shape() != [shape[0], 1] {
            return Err(GpuA2cLossError::InvalidBatch);
        }
        if !logits.context().is_compatible(&variable.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
    }
    let logs = logits.log_softmax()?;
    let selected_log_probs = logs.gather_actions(actions)?;
    let actor_loss = selected_log_probs
        .mul(&advantages.detach())?
        .mean()?
        .scale(-1.)?;
    let value_loss = values.mse_loss(&returns.detach())?;
    let entropy = logs.exp()?.mul(&logs)?.sum_columns()?.mean()?.scale(-1.)?;
    let total_loss = actor_loss
        .add(&value_loss.scale(config.value_coef)?)?
        .sub(&entropy.scale(config.entropy_coef)?)?;
    Ok(GpuA2cLoss {
        actor_loss,
        value_loss,
        entropy,
        total_loss,
        selected_log_probs,
        inputs: [logits, values, advantages, returns]
            .iter()
            .map(|v| v.detach())
            .collect(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a2c_coefficients_validate_without_adapter_allocation() {
        assert!(GpuA2cLossConfig::default().validate().is_ok());
        assert!(GpuA2cLossConfig {
            value_coef: 0.,
            entropy_coef: 0.
        }
        .validate()
        .is_ok());
        for value in [-1., f32::NAN, f32::INFINITY] {
            assert!(GpuA2cLossConfig {
                value_coef: value,
                ..Default::default()
            }
            .validate()
            .is_err());
            assert!(GpuA2cLossConfig {
                entropy_coef: value,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
    }
}
