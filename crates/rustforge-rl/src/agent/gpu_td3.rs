//! Device-resident TD3 objectives, owned networks, replay updates and target synchronization.
mod agent;
pub use agent::{
    GpuTd3, GpuTd3CheckpointError, GpuTd3Net, CHECKPOINT_MAGIC, CHECKPOINT_VERSION,
    MAX_CHECKPOINT_BYTES,
};
use rustforge_autograd::gpu::{GpuAutogradError, GpuVariable};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};
use std::{error::Error, fmt};

#[derive(Debug)]
pub enum GpuTd3Error {
    Autograd(GpuAutogradError),
    Checkpoint(GpuTd3CheckpointError),
    Module(rustforge_nn::gpu::GpuModuleError),
    InvalidInput(&'static str),
    Device(GpuError),
    InvalidConfig,
    InvalidBounds,
    InvalidBatch,
    NonFinite,
}
impl fmt::Display for GpuTd3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(e)=>e.fmt(f), Self::Module(e)=>e.fmt(f), Self::InvalidInput(message)=>f.write_str(message),
            Self::Autograd(e)=>e.fmt(f),Self::Device(e)=>e.fmt(f),
            Self::InvalidConfig=>f.write_str("TD3 requires positive compatible dimensions and finite learning rates, discount/tau in [0,1], and nonnegative finite smoothing deviation/clip"),
            Self::InvalidBounds=>f.write_str("TD3 bounds must be nonempty, finite, matching and strictly ordered with positive finite scale and finite bias"),
            Self::InvalidBatch=>f.write_str("TD3 requires matching nonempty matrices and done masks in [0,1]"),
            Self::NonFinite=>f.write_str("GPU TD3 inputs, intermediates or objective are nonfinite"),
        }
    }
}
impl Error for GpuTd3Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Autograd(e) => Some(e),
            Self::Module(e) => Some(e),
            Self::Checkpoint(e) => Some(e),
            Self::Device(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuAutogradError> for GpuTd3Error {
    fn from(e: GpuAutogradError) -> Self {
        Self::Autograd(e)
    }
}
impl From<GpuError> for GpuTd3Error {
    fn from(e: GpuError) -> Self {
        Self::Device(e)
    }
}
impl From<rustforge_nn::gpu::GpuModuleError> for GpuTd3Error {
    fn from(e: rustforge_nn::gpu::GpuModuleError) -> Self {
        Self::Module(e)
    }
}
pub type Result<T> = std::result::Result<T, GpuTd3Error>;
#[derive(Clone, Copy, Debug)]
pub struct GpuTd3LossConfig {
    pub gamma: f32,
}
impl Default for GpuTd3LossConfig {
    fn default() -> Self {
        Self { gamma: 0.99 }
    }
}
impl GpuTd3LossConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.gamma.is_finite() || !(0. ..=1.).contains(&self.gamma) {
            return Err(GpuTd3Error::InvalidConfig);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GpuTd3SmoothingConfig {
    pub noise_std: f32,
    pub noise_clip: f32,
}
impl Default for GpuTd3SmoothingConfig {
    fn default() -> Self {
        Self {
            noise_std: 0.2,
            noise_clip: 0.5,
        }
    }
}
impl GpuTd3SmoothingConfig {
    pub fn validate(&self) -> Result<()> {
        if [self.noise_std, self.noise_clip]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
        {
            return Err(GpuTd3Error::InvalidConfig);
        }
        Ok(())
    }
}
fn finite(inputs: impl IntoIterator<Item = GpuVariable>) -> Result<()> {
    let mut inputs = inputs.into_iter();
    let Some(first) = inputs.next() else {
        return Ok(());
    };
    finite_data(
        first.context(),
        std::iter::once(first.data()).chain(inputs.map(|v| v.data())),
    )
}
fn finite_data(
    context: &GpuContext,
    inputs: impl IntoIterator<Item = std::rc::Rc<rustforge_tensor::gpu::GpuTensor>>,
) -> Result<()> {
    let _profile = context.profile_scope("finite_validation");
    let mut count = None;
    for data in inputs {
        let next = context.nonfinite_count_device(&data)?;
        count = Some(match count {
            Some(old) => context.add_device(&old, &next)?,
            None => next,
        });
    }
    if let Some(count) = count {
        if context.download(&count)?.item() != 0. {
            return Err(GpuTd3Error::NonFinite);
        }
    }
    Ok(())
}
fn matrix_group(first: &GpuVariable, others: &[&GpuVariable], cols: usize) -> Result<()> {
    let data = first.data();
    let shape = data.shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] != cols {
        return Err(GpuTd3Error::InvalidBatch);
    }
    for v in others {
        if v.data().shape() != shape {
            return Err(GpuTd3Error::InvalidBatch);
        }
        if !first.context().is_compatible(&v.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
    }
    Ok(())
}
fn affine_bounds(low: &[f32], high: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    if low.is_empty()
        || low.len() != high.len()
        || low
            .iter()
            .zip(high)
            .any(|(l, h)| !l.is_finite() || !h.is_finite() || l >= h)
    {
        return Err(GpuTd3Error::InvalidBounds);
    }
    let scale: Vec<_> = low.iter().zip(high).map(|(l, h)| (h - l) / 2.).collect();
    let bias: Vec<_> = low.iter().zip(high).map(|(l, h)| (h + l) / 2.).collect();
    if scale.iter().any(|v| !v.is_finite() || *v <= 0.) || bias.iter().any(|v| !v.is_finite()) {
        return Err(GpuTd3Error::InvalidBounds);
    }
    Ok((scale, bias))
}
/// Affine normalized-action scaling, shared by deterministic actor and target smoothing.
pub struct GpuTd3ActionTransform {
    context: GpuContext,
    scale: GpuVariable,
    bias: GpuVariable,
    actions: usize,
}
pub struct GpuTd3Actions {
    pub normalized_actions: GpuVariable,
    pub actions: GpuVariable,
    inputs: Vec<GpuVariable>,
}
impl GpuTd3Actions {
    /// Explicit validation; includes raw inputs/intermediates so clipping cannot hide overflow.
    pub fn checked(&self) -> Result<()> {
        finite(
            self.inputs
                .iter()
                .chain([&self.normalized_actions, &self.actions])
                .cloned(),
        )
    }
}
impl GpuTd3ActionTransform {
    pub fn new(context: &GpuContext, low: &[f32], high: &[f32]) -> Result<Self> {
        let (scale, bias) = affine_bounds(low, high)?;
        let upload = |v| GpuVariable::new(context, &Tensor::from_vec(v, &[low.len()]), false);
        Ok(Self {
            context: context.clone(),
            scale: upload(scale)?,
            bias: upload(bias)?,
            actions: low.len(),
        })
    }
    fn validate(&self, raw: &GpuVariable) -> Result<()> {
        matrix_group(raw, &[], self.actions)?;
        if !self.context.is_compatible(&raw.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
        Ok(())
    }
    /// Preserves actor gradients. Caller supplies tanh-bounded actions, like CPU TD3.
    /// No extra clamp is applied on the actor path.
    pub fn scale_actor_actions(&self, raw: &GpuVariable) -> Result<GpuTd3Actions> {
        self.validate(raw)?;
        let zeros = self.context.zeros(raw.data().shape())?;
        let scale = GpuVariable::from_device(
            &self.context,
            self.context.add_bias_device(&zeros, &self.scale.data())?,
            false,
        )?;
        let actions = raw.mul(&scale)?.add_bias(&self.bias)?;
        Ok(GpuTd3Actions {
            normalized_actions: raw.clone(),
            actions,
            inputs: vec![raw.detach()],
        })
    }
    /// clip(raw + clip(std * supplied standard-normal noise, -clip, clip), -1,1),
    /// then affine scaling. All target outputs are detached; no RNG is consumed.
    pub fn smooth_target_actions(
        &self,
        raw: &GpuVariable,
        standard_normal: &GpuVariable,
        config: GpuTd3SmoothingConfig,
    ) -> Result<GpuTd3Actions> {
        config.validate()?;
        self.validate(raw)?;
        matrix_group(raw, &[standard_normal], self.actions)?;
        let noise = standard_normal.detach().scale(config.noise_std)?;
        let unclipped = raw
            .detach()
            .add(&noise.clamp(-config.noise_clip, config.noise_clip)?)?;
        let normalized = unclipped.clamp(-1., 1.)?;
        let mut output = self.scale_actor_actions(&normalized)?;
        output
            .inputs
            .extend([raw.detach(), standard_normal.detach(), noise, unclipped]);
        Ok(output)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GpuTd3CriticMetrics {
    pub critic1_loss: f32,
    pub critic2_loss: f32,
    pub total_loss: f32,
}
pub struct GpuTd3CriticLoss {
    pub target_values: GpuVariable,
    pub critic1_loss: GpuVariable,
    pub critic2_loss: GpuVariable,
    pub total_loss: GpuVariable,
    mask_violation: GpuVariable,
    inputs: Vec<GpuVariable>,
}
impl GpuTd3CriticLoss {
    /// Explicit readbacks before backward; callers must check gradients/squares before Adam.
    pub fn checked_metrics(&self) -> Result<GpuTd3CriticMetrics> {
        finite(self.inputs.iter().chain([&self.target_values]).cloned())?;
        if self.mask_violation.to_cpu()?.item() != 0. {
            return Err(GpuTd3Error::InvalidBatch);
        }
        let m = GpuTd3CriticMetrics {
            critic1_loss: self.critic1_loss.to_cpu()?.item(),
            critic2_loss: self.critic2_loss.to_cpu()?.item(),
            total_loss: self.total_loss.to_cpu()?.item(),
        };
        if [m.critic1_loss, m.critic2_loss, m.total_loss]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(GpuTd3Error::NonFinite);
        }
        Ok(m)
    }
}
/// y = detach(reward + gamma*(1-done)*min(target_q1,target_q2));
/// loss = MSE(q1,y) + MSE(q2,y). No gradient reaches targets/rewards/done masks.
/// Construction stays on device. Masks must be in [0,1]; fractional masks retain CPU arithmetic.
pub fn td3_critic_loss(
    q1: &GpuVariable,
    q2: &GpuVariable,
    target_q1: &GpuVariable,
    target_q2: &GpuVariable,
    rewards: &GpuVariable,
    dones: &GpuVariable,
    config: GpuTd3LossConfig,
) -> Result<GpuTd3CriticLoss> {
    config.validate()?;
    matrix_group(q1, &[q2, target_q1, target_q2, rewards, dones], 1)?;
    let context = q1.context();
    let ones = GpuVariable::from_device(context, context.full(q1.data().shape(), 1.)?, false)?;
    let mask = dones.detach();
    let mask_violation = mask
        .scale(-1.)?
        .relu()?
        .add(&mask.sub(&ones)?.relu()?)?
        .sum()?;
    let minimum = target_q1.detach().minimum(&target_q2.detach())?;
    let target_values = rewards
        .detach()
        .add(&ones.sub(&mask)?.mul(&minimum)?.scale(config.gamma)?)?
        .detach();
    let critic1_loss = q1.mse_loss(&target_values)?;
    let critic2_loss = q2.mse_loss(&target_values)?;
    let total_loss = critic1_loss.add(&critic2_loss)?;
    Ok(GpuTd3CriticLoss {
        target_values,
        critic1_loss,
        critic2_loss,
        total_loss,
        mask_violation,
        inputs: [q1, q2, target_q1, target_q2, rewards, dones]
            .iter()
            .map(|v| v.detach())
            .collect(),
    })
}
pub struct GpuTd3ActorLoss {
    pub loss: GpuVariable,
    input: GpuVariable,
}
impl GpuTd3ActorLoss {
    pub fn checked_loss(&self) -> Result<f32> {
        finite([self.input.clone()])?;
        let loss = self.loss.to_cpu()?.item();
        if !loss.is_finite() {
            return Err(GpuTd3Error::NonFinite);
        }
        Ok(loss)
    }
}
/// -mean(Q1(state, scaled_actor_action)). Caller controls critic freezing and update cadence.
pub fn td3_actor_loss(q1_for_actor: &GpuVariable) -> Result<GpuTd3ActorLoss> {
    matrix_group(q1_for_actor, &[], 1)?;
    Ok(GpuTd3ActorLoss {
        loss: q1_for_actor.mean()?.scale(-1.)?,
        input: q1_for_actor.detach(),
    })
}
impl From<GpuTd3CheckpointError> for GpuTd3Error {
    fn from(e: GpuTd3CheckpointError) -> Self {
        Self::Checkpoint(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loss_and_smoothing_config_validate_without_adapter() {
        for gamma in [0., 0.99, 1.] {
            assert!(GpuTd3LossConfig { gamma }.validate().is_ok());
        }
        for gamma in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            assert!(GpuTd3LossConfig { gamma }.validate().is_err());
        }
        assert!(GpuTd3SmoothingConfig::default().validate().is_ok());
        assert!(GpuTd3SmoothingConfig {
            noise_std: 0.,
            noise_clip: 0.
        }
        .validate()
        .is_ok());
        for value in [-1., f32::NAN, f32::INFINITY] {
            assert!(GpuTd3SmoothingConfig {
                noise_std: value,
                ..Default::default()
            }
            .validate()
            .is_err());
            assert!(GpuTd3SmoothingConfig {
                noise_clip: value,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
    }
    #[test]
    fn affine_bounds_validate_before_device_allocation() {
        assert_eq!(
            affine_bounds(&[-2., 1.], &[4., 5.]).unwrap(),
            (vec![3., 2.], vec![1., 3.])
        );
        for (low, high) in [
            (vec![], vec![]),
            (vec![0.], vec![1., 2.]),
            (vec![1.], vec![1.]),
            (vec![2.], vec![1.]),
            (vec![f32::NAN], vec![1.]),
            (vec![0.], vec![f32::INFINITY]),
            (vec![-f32::MAX], vec![f32::MAX]),
            (vec![f32::MAX / 2.], vec![f32::MAX]),
            (vec![0.], vec![f32::from_bits(1)]),
        ] {
            assert!(affine_bounds(&low, &high).is_err());
        }
    }
}
