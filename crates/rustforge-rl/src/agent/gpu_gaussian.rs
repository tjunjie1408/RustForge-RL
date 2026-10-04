//! Tanh-squashed, scaled diagonal Gaussian operators. Distribution inputs and
//! supplied noise/actions live on device; this module does not own an RNG or agent.
use rustforge_autograd::gpu::{GpuAutogradError, GpuVariable};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};
use std::{error::Error, fmt};
#[derive(Debug)]
pub enum GpuGaussianError {
    Autograd(GpuAutogradError),
    Device(GpuError),
    InvalidShape,
    InvalidBounds,
    NonFinite,
}
impl fmt::Display for GpuGaussianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Autograd(e) => e.fmt(f),
            Self::Device(e) => e.fmt(f),
            Self::InvalidShape => f.write_str("Gaussian inputs must have matching [batch,actions] shapes and configured action dimensions"),
            Self::InvalidBounds => f.write_str("Gaussian action bounds must be nonempty, matching, finite and strictly ordered with finite positive scale and finite bias"),
            Self::NonFinite => f.write_str("GPU Gaussian inputs or distribution statistics are nonfinite"),
        }
    }
}
impl Error for GpuGaussianError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Autograd(e) => Some(e),
            Self::Device(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuError> for GpuGaussianError {
    fn from(e: GpuError) -> Self {
        Self::Device(e)
    }
}
impl From<GpuAutogradError> for GpuGaussianError {
    fn from(e: GpuAutogradError) -> Self {
        Self::Autograd(e)
    }
}
type Result<T> = std::result::Result<T, GpuGaussianError>;

pub struct GpuGaussianTransform {
    context: GpuContext,
    scale: GpuVariable,
    bias: GpuVariable,
    log_scale: GpuVariable,
    actions: usize,
}
pub struct GpuGaussianLogProb {
    /// Squashed/scaled action log density [batch,1], including the Jacobian.
    pub log_probs: GpuVariable,
    /// Analytic entropy of the base Gaussian [batch,1]; not squashed entropy.
    pub base_entropy: GpuVariable,
    // Immutable snapshots ensure clipping cannot hide invalid raw inputs.
    inputs: Vec<GpuVariable>,
}
pub struct GpuGaussianSample {
    pub actions: GpuVariable,
    pub distribution: GpuGaussianLogProb,
}
impl GpuGaussianSample {
    /// Validates the physical actions as well as density inputs/intermediates.
    pub fn checked_metrics(&self) -> Result<GpuGaussianMetrics> {
        let context = self.actions.context();
        if context
            .download(&context.nonfinite_count_device(&self.actions.data())?)?
            .item()
            != 0.
        {
            return Err(GpuGaussianError::NonFinite);
        }
        self.distribution.checked_metrics()
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GpuGaussianMetrics {
    pub mean_log_prob: f32,
    pub base_entropy: f32,
}
impl GpuGaussianLogProb {
    /// Explicit scalar validation/readback before backward or optimizer updates.
    pub fn checked_metrics(&self) -> Result<GpuGaussianMetrics> {
        for variable in self
            .inputs
            .iter()
            .chain([&self.log_probs, &self.base_entropy])
        {
            let c = variable.context();
            let invalid = c.nonfinite_count_device(&variable.data())?;
            if c.download(&invalid)?.item() != 0. {
                return Err(GpuGaussianError::NonFinite);
            }
        }
        let metrics = GpuGaussianMetrics {
            mean_log_prob: self.log_probs.mean()?.to_cpu()?.item(),
            base_entropy: self.base_entropy.mean()?.to_cpu()?.item(),
        };
        if !metrics.mean_log_prob.is_finite() || !metrics.base_entropy.is_finite() {
            return Err(GpuGaussianError::NonFinite);
        }
        Ok(metrics)
    }
}
impl GpuGaussianTransform {
    pub fn new(context: &GpuContext, low: &[f32], high: &[f32]) -> Result<Self> {
        if low.is_empty()
            || low.len() != high.len()
            || low
                .iter()
                .zip(high)
                .any(|(l, h)| !l.is_finite() || !h.is_finite() || l >= h)
        {
            return Err(GpuGaussianError::InvalidBounds);
        }
        let scale: Vec<_> = low.iter().zip(high).map(|(l, h)| (h - l) / 2.).collect();
        let bias: Vec<_> = low.iter().zip(high).map(|(l, h)| (h + l) / 2.).collect();
        if scale.iter().any(|s| !s.is_finite() || *s <= 0.) || bias.iter().any(|b| !b.is_finite()) {
            return Err(GpuGaussianError::InvalidBounds);
        }
        let log_scale = scale.iter().map(|s| s.ln()).collect();
        let upload = |v| GpuVariable::new(context, &Tensor::from_vec(v, &[low.len()]), false);
        Ok(Self {
            context: context.clone(),
            scale: upload(scale)?,
            bias: upload(bias)?,
            log_scale: upload(log_scale)?,
            actions: low.len(),
        })
    }
    fn validate(
        &self,
        mean: &GpuVariable,
        log_std: &GpuVariable,
        other: &GpuVariable,
    ) -> Result<()> {
        let data = mean.data();
        let shape = data.shape();
        if shape.len() != 2
            || shape[1] != self.actions
            || log_std.data().shape() != shape
            || other.data().shape() != shape
        {
            return Err(GpuGaussianError::InvalidShape);
        }
        for v in [mean, log_std, other] {
            if !self.context.is_compatible(&v.data()) {
                return Err(GpuError::DeviceMismatch.into());
            }
        }
        Ok(())
    }
    fn expand(&self, feature: &GpuVariable, shape: &[usize]) -> Result<GpuVariable> {
        let zeros = self.context.zeros(shape)?;
        Ok(GpuVariable::from_device(
            &self.context,
            self.context.add_bias_device(&zeros, &feature.data())?,
            false,
        )?)
    }
    fn constant(&self, shape: &[usize], value: f32) -> Result<GpuVariable> {
        Ok(GpuVariable::from_device(
            &self.context,
            self.context.full(shape, value)?,
            false,
        )?)
    }
    fn distribution(
        &self,
        u: &GpuVariable,
        mean: &GpuVariable,
        log_std: &GpuVariable,
        tanh_u: &GpuVariable,
        inputs: Vec<GpuVariable>,
    ) -> Result<GpuGaussianLogProb> {
        let shape = mean.data().shape().to_vec();
        let z = u.sub(mean)?.div(&log_std.exp()?)?;
        let normal = z
            .mul(&z)?
            .scale(-0.5)?
            .sub(log_std)?
            .sub(&self.constant(&shape, 0.5 * (2. * std::f32::consts::PI).ln())?)?;
        let jacobian = self
            .constant(&shape, 1.)?
            .sub(&tanh_u.mul(tanh_u)?)?
            .add(&self.constant(&shape, 1e-6)?)?
            .log()?;
        let log_probs = normal
            .sub(&jacobian)?
            .sub(&self.expand(&self.log_scale, &shape)?)?
            .sum_columns()?;
        let base_entropy = log_std
            .add(&self.constant(&shape, 0.5 * (2. * std::f32::consts::PI).ln() + 0.5)?)?
            .sum_columns()?;
        Ok(GpuGaussianLogProb {
            log_probs,
            base_entropy,
            inputs,
        })
    }
    /// Stored actions are detached, normalized and clamped to (-1+1e-6,1-1e-6).
    /// Actions outside bounds follow CPU's clipping convention; nonfinite inputs
    /// are rejected by checked_metrics even if clipping masks them.
    pub fn log_prob_from_action(
        &self,
        mean: &GpuVariable,
        raw_log_std: &GpuVariable,
        actions: &GpuVariable,
    ) -> Result<GpuGaussianLogProb> {
        self.validate(mean, raw_log_std, actions)?;
        let shape = mean.data().shape().to_vec();
        let detached = actions.detach();
        let normalized = detached
            .sub(&self.expand(&self.bias, &shape)?)?
            .div(&self.expand(&self.scale, &shape)?)?;
        let clamped = GpuVariable::from_device(
            &self.context,
            self.context
                .clamp_device(&normalized.data(), -1. + 1e-6, 1. - 1e-6)?,
            false,
        )?;
        let one = self.constant(&shape, 1.)?;
        let u = one
            .add(&clamped)?
            .log()?
            .sub(&one.sub(&clamped)?.log()?)?
            .scale(0.5)?;
        self.distribution(
            &u,
            mean,
            &raw_log_std.clamp(-20., 2.)?,
            &clamped,
            vec![mean.detach(), raw_log_std.detach(), detached],
        )
    }
    /// Reparameterized sampling with caller-supplied standard-normal noise.
    /// Noise is detached. Gradients through action and log density reach mean/std.
    pub fn sample_with_noise(
        &self,
        mean: &GpuVariable,
        raw_log_std: &GpuVariable,
        noise: &GpuVariable,
    ) -> Result<GpuGaussianSample> {
        self.validate(mean, raw_log_std, noise)?;
        let shape = mean.data().shape().to_vec();
        let std = raw_log_std.clamp(-20., 2.)?;
        let u = mean.add(&noise.detach().mul(&std.exp()?)?)?;
        let tanh_u = u.tanh()?;
        let actions = tanh_u
            .mul(&self.expand(&self.scale, &shape)?)?
            .add(&self.expand(&self.bias, &shape)?)?;
        let distribution = self.distribution(
            &u,
            mean,
            &std,
            &tanh_u,
            vec![
                mean.detach(),
                raw_log_std.detach(),
                noise.detach(),
                std.detach(),
                u.detach(),
                tanh_u.detach(),
            ],
        )?;
        Ok(GpuGaussianSample {
            actions,
            distribution,
        })
    }
}
