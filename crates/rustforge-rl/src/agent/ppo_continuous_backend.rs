//! Continuous agents are constructed inside the owning runtime worker.
use super::{DqnDevice, PPOContinuous, PPOContinuousConfig, PpoRuntimeOptions};
use crate::{buffer::ContinuousRolloutBatch, runtime::trainer::TrainerError};
use rand::Rng;
pub(super) enum PpoContinuousBackend {
    Cpu {
        agent: Box<PPOContinuous>,
        config: PPOContinuousConfig,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<super::gpu_ppo::GpuPpoContinuous>),
}
impl PpoContinuousBackend {
    pub(super) fn new(
        config: PPOContinuousConfig,
        seed: Option<u64>,
        options: &PpoRuntimeOptions,
    ) -> Result<Self, TrainerError> {
        options.validate()?;
        match options.device {
            DqnDevice::Cpu => {
                validate_continuous_config(&config).map_err(|message| TrainerError {
                    message: message.into(),
                })?;
                let agent = match seed {
                    Some(seed) => PPOContinuous::new_seeded(config.clone(), seed),
                    None => PPOContinuous::new(config.clone()),
                };
                Ok(Self::Cpu {
                    agent: Box::new(agent),
                    config,
                })
            }
            DqnDevice::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let context = rustforge_tensor::gpu::GpuContext::new().map_err(gpu_error)?;
                    let agent = match &options.resume {
                        Some(path) => {
                            super::gpu_ppo::GpuPpoContinuous::load_checkpoint(&context, path)
                                .map_err(gpu_error)?
                        }
                        None => super::gpu_ppo::GpuPpoContinuous::new_seeded(
                            &context,
                            config,
                            seed.unwrap_or_else(rand::random),
                        )
                        .map_err(gpu_error)?,
                    };
                    Ok(Self::Gpu(Box::new(agent)))
                }
                #[cfg(not(feature = "gpu"))]
                Err(TrainerError {
                    message: "GPU support is not compiled in".into(),
                })
            }
        }
    }
    pub(super) fn config(&self) -> &PPOContinuousConfig {
        match self {
            Self::Cpu { config, .. } => config,
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.config(),
        }
    }
    pub(super) fn select_action_with_rng<R: Rng + ?Sized>(
        &self,
        state: &[f32],
        rng: &mut R,
    ) -> Result<(Vec<f32>, f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.select_action_with_rng(state, rng)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.select_action_with_rng(state, rng).map_err(gpu_error),
        }
    }
    pub(super) fn select_action(
        &self,
        state: &[f32],
    ) -> Result<(Vec<f32>, f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.select_action(state)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent
                .select_action_with_rng(state, &mut rand::thread_rng())
                .map_err(gpu_error),
        }
    }
    pub(super) fn value_of(&self, state: &[f32]) -> Result<f32, TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.value_of(state)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.value_of(state).map_err(gpu_error),
        }
    }
    pub(super) fn train_on_batch_with_rng<R: Rng + ?Sized>(
        &mut self,
        batch: &ContinuousRolloutBatch,
        rng: &mut R,
    ) -> Result<(f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => {
                let (policy, value) = agent.train_on_batch_with_rng(batch, rng);
                Ok((policy, value))
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => {
                let m = agent
                    .train_on_batch_with_rng(batch, rng)
                    .map_err(gpu_error)?;
                Ok((m.policy_loss, m.value_loss))
            }
        }
    }
    pub(super) fn train_on_batch(
        &mut self,
        batch: &ContinuousRolloutBatch,
    ) -> Result<(f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => {
                let (policy, value) = agent.train_on_batch(batch);
                Ok((policy, value))
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => {
                let m = agent
                    .train_on_batch_with_rng(batch, &mut rand::thread_rng())
                    .map_err(gpu_error)?;
                Ok((m.policy_loss, m.value_loss))
            }
        }
    }
    pub(super) fn save(&self, options: &PpoRuntimeOptions) -> Result<(), TrainerError> {
        if let Some(path) = &options.checkpoint {
            match self {
                #[cfg(feature = "gpu")]
                Self::Gpu(agent) => agent.save_checkpoint(path).map_err(gpu_error)?,
                Self::Cpu { .. } => {
                    return Err(TrainerError {
                        message: format!(
                            "CPU PPO runtime cannot write GPU checkpoint {}",
                            path.display()
                        ),
                    })
                }
            }
        }
        Ok(())
    }
}
#[cfg(feature = "gpu")]
fn gpu_error(error: impl std::fmt::Display) -> TrainerError {
    TrainerError {
        message: format!("GPU PPO: {error}"),
    }
}

pub(crate) fn validate_continuous_config(config: &PPOContinuousConfig) -> Result<(), &'static str> {
    let c = &config.base;
    if c.obs_dim == 0
        || c.hidden_dim == 0
        || config.act_dim == 0
        || c.ppo_epochs == 0
        || c.mini_batch_size == 0
        || !c.lr.is_finite()
        || c.lr <= 0.
        || !c.gamma.is_finite()
        || !(0. ..=1.).contains(&c.gamma)
        || !c.gae_lambda.is_finite()
        || !(0. ..=1.).contains(&c.gae_lambda)
        || !c.clip_eps.is_finite()
        || !(0. ..1.).contains(&c.clip_eps)
        || !c.value_coef.is_finite()
        || c.value_coef < 0.
        || !c.entropy_coef.is_finite()
        || c.entropy_coef < 0.
        || c.hidden_dim.checked_mul(c.obs_dim).is_none()
        || c.hidden_dim.checked_mul(c.hidden_dim).is_none()
        || c.hidden_dim.checked_mul(config.act_dim).is_none()
    {
        return Err("invalid continuous PPO dimensions or hyperparameters");
    }
    if config.action_low.len() != config.act_dim
        || config.action_high.len() != config.act_dim
        || config
            .action_low
            .iter()
            .zip(&config.action_high)
            .any(|(&l, &h)| {
                !l.is_finite()
                    || !h.is_finite()
                    || l >= h
                    || !((h - l) / 2.).is_finite()
                    || (h - l) / 2. <= 0.
                    || !((h + l) / 2.).is_finite()
            })
    {
        return Err("invalid continuous PPO action bounds");
    }
    Ok(())
}
