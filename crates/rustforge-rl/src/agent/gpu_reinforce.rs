//! GPU REINFORCE objective and agent; sampling and Monte Carlo rollouts stay on CPU.
mod agent;
pub use agent::{GpuReinforce, GpuReinforceRolloutOptions};
use rustforge_autograd::gpu::{GpuAutogradError, GpuVariable};
use rustforge_nn::gpu::{GpuLinear, GpuModule, GpuModuleError, GpuReLU, GpuSequential};
use rustforge_tensor::gpu::{GpuContext, GpuError, GpuIndices};
use std::{error::Error, fmt, rc::Rc};

#[derive(Debug)]
pub enum GpuReinforceError {
    Autograd(GpuAutogradError),
    Module(GpuModuleError),
    Device(GpuError),
    InvalidInput(&'static str),
    InvalidDimensions,
    InvalidBatch,
    NonFinite,
}
impl fmt::Display for GpuReinforceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Autograd(e) => e.fmt(f),
            Self::Module(e) => e.fmt(f),
            Self::Device(e) => e.fmt(f),
            Self::InvalidInput(message) => f.write_str(message),
            Self::InvalidDimensions => f.write_str("REINFORCE dimensions must be positive with representable parameter sizes"),
            Self::InvalidBatch => f.write_str("REINFORCE requires nonempty [batch,actions] logits, matching typed actions and [batch,1] advantages"),
            Self::NonFinite => f.write_str("GPU REINFORCE input, centered advantages or loss is nonfinite"),
        }
    }
}
impl Error for GpuReinforceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Autograd(e) => Some(e),
            Self::Module(e) => Some(e),
            Self::Device(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuAutogradError> for GpuReinforceError {
    fn from(e: GpuAutogradError) -> Self {
        Self::Autograd(e)
    }
}
impl From<GpuModuleError> for GpuReinforceError {
    fn from(e: GpuModuleError) -> Self {
        Self::Module(e)
    }
}
impl From<GpuError> for GpuReinforceError {
    fn from(e: GpuError) -> Self {
        Self::Device(e)
    }
}
pub type Result<T> = std::result::Result<T, GpuReinforceError>;

/// Linear/ReLU/Linear policy matching CPU REINFORCE initialization and parameter order.
pub struct GpuReinforceNet {
    policy: GpuSequential,
}
fn validate_dimensions(obs: usize, hidden: usize, actions: usize) -> Result<()> {
    if obs == 0
        || hidden == 0
        || actions == 0
        || hidden.checked_mul(obs).is_none()
        || actions.checked_mul(hidden).is_none()
    {
        return Err(GpuReinforceError::InvalidDimensions);
    }
    Ok(())
}
impl GpuReinforceNet {
    pub fn new_seeded(
        context: &GpuContext,
        obs: usize,
        hidden: usize,
        actions: usize,
        seed: u64,
    ) -> Result<Self> {
        validate_dimensions(obs, hidden, actions)?;
        Ok(Self {
            policy: GpuSequential::new(vec![
                Box::new(GpuLinear::new_seeded(context, obs, hidden, seed)?),
                Box::new(GpuReLU),
                Box::new(GpuLinear::new_seeded(
                    context,
                    hidden,
                    actions,
                    seed.wrapping_add(1),
                )?),
            ]),
        })
    }
    pub fn forward(&self, states: &GpuVariable) -> Result<GpuVariable> {
        Ok(self.policy.forward(states)?)
    }
    pub fn parameters(&self) -> Vec<GpuVariable> {
        self.policy.parameters()
    }
}

pub struct GpuReinforceLoss {
    pub loss: GpuVariable,
    pub selected_log_probs: GpuVariable,
    /// Detached raw or mean-centered advantages used by the objective.
    pub effective_advantages: GpuVariable,
    inputs: Vec<GpuVariable>,
}
impl GpuReinforceLoss {
    /// Explicit finite checks/readback before backward. Callers must also check
    /// gradients and their squares before Adam; finite loss alone is insufficient.
    pub fn checked_loss(&self) -> Result<f32> {
        let context = self.loss.context();
        for input in self
            .inputs
            .iter()
            .chain(std::iter::once(&self.effective_advantages))
        {
            if context
                .download(&context.nonfinite_count_device(&input.data())?)?
                .item()
                != 0.
            {
                return Err(GpuReinforceError::NonFinite);
            }
        }
        let loss = self.loss.to_cpu()?.item();
        if !loss.is_finite() {
            return Err(GpuReinforceError::NonFinite);
        }
        Ok(loss)
    }
}

/// -mean(log pi(action) * detached advantages), optionally subtracting their mean.
/// With Monte Carlo rollouts, advantages equal discounted returns (value=0,
/// lambda=1, last_value=0). No normalization, critic, entropy or importance ratio.
/// Graph construction and mean centering stay on device without host readback.
pub fn reinforce_loss(
    logits: &GpuVariable,
    actions: &Rc<GpuIndices>,
    advantages: &GpuVariable,
    use_baseline: bool,
) -> Result<GpuReinforceLoss> {
    let shape = logits.data().shape().to_vec();
    if shape.len() != 2
        || shape[0] == 0
        || shape[1] == 0
        || actions.len() != shape[0]
        || actions.columns() != shape[1]
        || advantages.data().shape() != [shape[0], 1]
    {
        return Err(GpuReinforceError::InvalidBatch);
    }
    let context = logits.context();
    if !context.is_compatible(&advantages.data()) {
        return Err(GpuError::DeviceMismatch.into());
    }
    let detached = advantages.detach();
    let effective_advantages = if use_baseline {
        let mean = detached.mean()?;
        let broadcast = GpuVariable::from_device(
            context,
            context.broadcast_scalar_device(&mean.data(), &[shape[0], 1], 1.)?,
            false,
        )?;
        detached.sub(&broadcast)?
    } else {
        detached
    };
    let selected_log_probs = logits.log_softmax()?.gather_actions(actions)?;
    let loss = selected_log_probs
        .mul(&effective_advantages)?
        .mean()?
        .scale(-1.)?;
    Ok(GpuReinforceLoss {
        loss,
        selected_log_probs,
        effective_advantages,
        inputs: vec![logits.detach(), advantages.detach()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_dimensions_validate_without_adapter() {
        assert!(validate_dimensions(4, 8, 2).is_ok());
        for (o, h, a) in [
            (0, 8, 2),
            (4, 0, 2),
            (4, 8, 0),
            (usize::MAX, 2, 1),
            (1, 2, usize::MAX),
        ] {
            assert!(matches!(
                validate_dimensions(o, h, a),
                Err(GpuReinforceError::InvalidDimensions)
            ));
        }
    }
}
