//! Optional uniform-replay DQN/Double DQN with resident network training.
//! Environment interaction and replay storage remain on CPU. Upload batches
//! explicitly to reuse them; each training step downloads only its loss scalar.
//! Versioned checkpoints explicitly transfer state. Prioritized replay and CLI
//! runtime support are not enabled here.
mod checkpoint;
use crate::{agent::DQNConfig, buffer::TransitionBatch};
pub use checkpoint::{
    GpuDqnCheckpointError, CHECKPOINT_MAGIC, CHECKPOINT_VERSION, MAX_CHECKPOINT_BYTES,
};
use rustforge_autograd::{
    gpu::{GpuAdam, GpuAutogradError, GpuVariable},
    no_grad,
};
use rustforge_nn::gpu::{GpuLinear, GpuModule, GpuModuleError, GpuReLU, GpuSequential};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError, GpuIndices},
    Tensor,
};
use std::{error::Error, fmt, rc::Rc};

#[derive(Debug)]
pub enum GpuDqnError {
    Device(GpuError),
    Autograd(GpuAutogradError),
    Module(GpuModuleError),
    Checkpoint(GpuDqnCheckpointError),
    InvalidConfig(&'static str),
    InvalidBatch(&'static str),
    NonFiniteLoss,
    GradientsDisabled,
    StepOverflow,
}
impl fmt::Display for GpuDqnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(e) => e.fmt(f),
            Self::Autograd(e) => e.fmt(f),
            Self::Module(e) => e.fmt(f),
            Self::Checkpoint(e) => e.fmt(f),
            Self::InvalidConfig(message) | Self::InvalidBatch(message) => f.write_str(message),
            Self::NonFiniteLoss => {
                f.write_str("GPU DQN loss is nonfinite; optimizer step was not applied")
            }
            Self::GradientsDisabled => f.write_str("GPU DQN training requires gradient recording"),
            Self::StepOverflow => f.write_str("GPU DQN training step overflow"),
        }
    }
}
impl Error for GpuDqnError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Device(e) => Some(e),
            Self::Autograd(e) => Some(e),
            Self::Module(e) => Some(e),
            Self::Checkpoint(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuError> for GpuDqnError {
    fn from(e: GpuError) -> Self {
        Self::Device(e)
    }
}
impl From<GpuAutogradError> for GpuDqnError {
    fn from(e: GpuAutogradError) -> Self {
        Self::Autograd(e)
    }
}
impl From<GpuModuleError> for GpuDqnError {
    fn from(e: GpuModuleError) -> Self {
        Self::Module(e)
    }
}
pub type Result<T> = std::result::Result<T, GpuDqnError>;

/// Immutable uploaded replay batch. Construct through GpuDqn::upload_batch;
/// active rows alone are validated/uploaded, ignoring unused replay capacity.
pub struct GpuDqnBatch {
    states: GpuVariable,
    next_states: GpuVariable,
    rewards: GpuVariable,
    dones: GpuVariable,
    actions: Rc<GpuIndices>,
}
impl GpuDqnBatch {
    pub fn len(&self) -> usize {
        self.actions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

pub struct GpuDqn {
    context: GpuContext,
    q_net: GpuSequential,
    target_net: GpuSequential,
    optimizer: GpuAdam,
    config: DQNConfig,
    train_steps: usize,
}
impl GpuDqn {
    /// Builds seeded Linear→ReLU→Linear networks. Target handles are distinct,
    /// frozen leaves; their immutable storage snapshots initially match online.
    pub fn new_seeded(context: &GpuContext, config: DQNConfig, seed: u64) -> Result<Self> {
        validate_config(&config)?;
        let first = GpuLinear::new_seeded(context, config.obs_dim, config.hidden_dim, seed)?;
        let last = GpuLinear::new_seeded(
            context,
            config.hidden_dim,
            config.num_actions,
            seed.wrapping_add(1),
        )?;
        let target_net = GpuSequential::new(vec![
            Box::new(first.frozen_snapshot()),
            Box::new(GpuReLU),
            Box::new(last.frozen_snapshot()),
        ]);
        let q_net = GpuSequential::new(vec![Box::new(first), Box::new(GpuReLU), Box::new(last)]);
        let optimizer = GpuAdam::new(q_net.parameters(), config.lr)?;
        Ok(Self {
            context: context.clone(),
            q_net,
            target_net,
            optimizer,
            config,
            train_steps: 0,
        })
    }
    pub fn upload_batch(&self, batch: &TransitionBatch) -> Result<GpuDqnBatch> {
        let n = batch.size;
        let matrix_valid =
            |t: &Tensor, cols| t.shape().len() == 2 && t.shape()[0] >= n && t.shape()[1] == cols;
        if n == 0
            || !matrix_valid(&batch.states, self.config.obs_dim)
            || !matrix_valid(&batch.next_states, self.config.obs_dim)
            || !matrix_valid(&batch.rewards, 1)
            || !matrix_valid(&batch.dones, 1)
            || batch.actions.len() < n
        {
            return Err(GpuDqnError::InvalidBatch("DQN batch must have nonempty active rows and matching observation/reward/done/action dimensions"));
        }
        let slice = |t: &Tensor| {
            t.slice_axis(0, 0, n)
                .map_err(|_| GpuDqnError::InvalidBatch("invalid active batch slice"))
        };
        let (states, next_states, rewards, dones) = (
            slice(&batch.states)?,
            slice(&batch.next_states)?,
            slice(&batch.rewards)?,
            slice(&batch.dones)?,
        );
        if [&states, &next_states, &rewards, &dones]
            .iter()
            .any(|t| t.to_vec().iter().any(|v| !v.is_finite()))
            || dones.to_vec().iter().any(|&v| v != 0.0 && v != 1.0)
        {
            return Err(GpuDqnError::InvalidBatch(
                "active batch data must be finite and done flags must be zero or one",
            ));
        }
        // Validate action values before uploading any batch tensor.
        if let Some(&action) = batch.actions[..n]
            .iter()
            .find(|&&a| a >= self.config.num_actions)
        {
            return Err(GpuError::ActionOutOfBounds {
                action,
                columns: self.config.num_actions,
            }
            .into());
        }
        Ok(GpuDqnBatch {
            states: GpuVariable::new(&self.context, &states, false)?,
            next_states: GpuVariable::new(&self.context, &next_states, false)?,
            rewards: GpuVariable::new(&self.context, &rewards, false)?,
            dones: GpuVariable::new(&self.context, &dones, false)?,
            actions: Rc::new(
                self.context
                    .upload_indices(&batch.actions[..n], self.config.num_actions)?,
            ),
        })
    }
    fn validate_batch(&self, batch: &GpuDqnBatch) -> Result<()> {
        if batch.is_empty()
            || batch.states.data().shape()[1] != self.config.obs_dim
            || batch.actions.columns() != self.config.num_actions
        {
            return Err(GpuDqnError::InvalidBatch(
                "uploaded batch does not match this agent",
            ));
        }
        if !self.context.is_compatible(&batch.states.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
        Ok(())
    }
    /// Computes detached Bellman targets on device. Terminal flags suppress
    /// bootstrapping; time-limit truncations should use replay_done's false flag.
    pub fn td_targets(&self, batch: &GpuDqnBatch) -> Result<GpuVariable> {
        self.validate_batch(batch)?;
        no_grad(|| {
            let next_target = self.target_net.forward(&batch.next_states)?;
            let selections = if self.config.double_dqn {
                self.context
                    .argmax_rows_device(&self.q_net.forward(&batch.next_states)?.data())?
            } else {
                self.context.argmax_rows_device(&next_target.data())?
            };
            let next_value = next_target.gather_actions(&Rc::new(selections))?;
            let ones = GpuVariable::from_device(
                &self.context,
                self.context.full(batch.dones.data().shape(), 1.0)?,
                false,
            )?;
            let not_done = ones.sub(&batch.dones)?;
            Ok(batch
                .rewards
                .add(&next_value.scale(self.config.gamma)?.mul(&not_done)?)?)
        })
    }
    /// Uploads active CPU replay rows, then trains. Use train_device_batch to
    /// reuse an already uploaded batch without repeating those transfers.
    pub fn train_step(&mut self, batch: &TransitionBatch) -> Result<f32> {
        let batch = self.upload_batch(batch)?;
        self.train_device_batch(&batch)
    }
    pub fn train_device_batch(&mut self, batch: &GpuDqnBatch) -> Result<f32> {
        if !rustforge_autograd::is_grad_enabled() {
            return Err(GpuDqnError::GradientsDisabled);
        }
        self.validate_batch(batch)?;
        let next_step = self
            .train_steps
            .checked_add(1)
            .ok_or(GpuDqnError::StepOverflow)?;
        let prediction = self
            .q_net
            .forward(&batch.states)?
            .gather_actions(&batch.actions)?;
        let loss = prediction.mse_loss(&self.td_targets(batch)?)?;
        let value = loss.to_cpu()?.item();
        if !value.is_finite() {
            return Err(GpuDqnError::NonFiniteLoss);
        }
        self.optimizer.zero_grad();
        loss.backward()?;
        self.optimizer.step()?;
        self.train_steps = next_step;
        if self.config.target_update_freq != 0 && next_step % self.config.target_update_freq == 0 {
            self.update_target()?;
        }
        Ok(value)
    }
    /// Hard synchronization replaces frozen parameter snapshots without copies.
    /// Online updates allocate new buffers, so target values remain fixed between syncs.
    pub fn update_target(&self) -> Result<()> {
        Ok(self.target_net.copy_parameters_from(&self.q_net)?)
    }
    /// Greedy action selection downloads only one typed index. Exploration is
    /// controlled by the caller's seeded policy; no graph is recorded here.
    pub fn select_greedy_action(&self, state: &[f32]) -> Result<usize> {
        if state.len() != self.config.obs_dim || state.iter().any(|v| !v.is_finite()) {
            return Err(GpuDqnError::InvalidBatch(
                "state must be finite and match observation dimensions",
            ));
        }
        no_grad(|| {
            let input = GpuVariable::new(
                &self.context,
                &Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]),
                false,
            )?;
            let values = self.q_net.forward(&input)?;
            Ok(self
                .context
                .download_indices(&self.context.argmax_rows_device(&values.data())?)?[0])
        })
    }
    pub fn q_net(&self) -> &GpuSequential {
        &self.q_net
    }
    pub fn target_net(&self) -> &GpuSequential {
        &self.target_net
    }
    pub fn config(&self) -> &DQNConfig {
        &self.config
    }
    pub fn train_steps(&self) -> usize {
        self.train_steps
    }
}

impl From<GpuDqnCheckpointError> for GpuDqnError {
    fn from(e: GpuDqnCheckpointError) -> Self {
        Self::Checkpoint(e)
    }
}

fn validate_config(config: &DQNConfig) -> Result<()> {
    if config.obs_dim == 0
        || config.num_actions == 0
        || config.hidden_dim == 0
        || !config.lr.is_finite()
        || config.lr <= 0.0
        || !config.gamma.is_finite()
        || !(0.0..=1.0).contains(&config.gamma)
    {
        return Err(GpuDqnError::InvalidConfig(
            "DQN dimensions/lr must be positive; gamma must be finite and in [0, 1]",
        ));
    }
    if config.use_per {
        return Err(GpuDqnError::InvalidConfig(
            "GPU DQN currently supports uniform replay; prioritized replay is not enabled",
        ));
    }
    Ok(())
}
