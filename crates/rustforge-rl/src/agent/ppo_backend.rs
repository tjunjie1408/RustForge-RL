//! Backend values are constructed inside the worker; GPU Rc values never cross threads.
use super::{DqnDevice, DqnRuntimeOptions, PPODiscrete, PPODiscreteConfig};
use crate::{buffer::RolloutBatch, runtime::trainer::TrainerError};
use rand::Rng;
use std::path::PathBuf;

pub use super::DqnDevice as PpoDevice;
/// GPU checkpoints restore model/Adam/configuration/updates. Environment,
/// rollout, run counters, metrics and action/shuffle RNG start a fresh run.
#[derive(Clone, Debug, Default)]
pub struct PpoRuntimeOptions {
    pub device: PpoDevice,
    pub resume: Option<PathBuf>,
    pub checkpoint: Option<PathBuf>,
}
impl From<DqnRuntimeOptions> for PpoRuntimeOptions {
    fn from(options: DqnRuntimeOptions) -> Self {
        Self {
            device: options.device,
            resume: options.resume,
            checkpoint: options.checkpoint,
        }
    }
}
impl PpoRuntimeOptions {
    pub fn validate(&self) -> Result<(), TrainerError> {
        let message = match self.device {
            PpoDevice::Cpu if self.resume.is_some() || self.checkpoint.is_some() => {
                Some("PPO runtime --resume and --checkpoint require --device gpu")
            }
            PpoDevice::Gpu if !cfg!(feature = "gpu") => {
                Some("GPU support is not compiled in; rebuild with --features gpu")
            }
            _ => None,
        };
        match message {
            Some(message) => Err(TrainerError {
                message: message.into(),
            }),
            None => Ok(()),
        }
    }
}
pub(super) enum PpoBackend {
    Cpu {
        agent: Box<PPODiscrete>,
        config: PPODiscreteConfig,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<super::gpu_ppo::GpuPpoDiscrete>),
}
impl PpoBackend {
    pub(super) fn new(
        config: PPODiscreteConfig,
        seed: Option<u64>,
        options: &PpoRuntimeOptions,
    ) -> Result<Self, TrainerError> {
        options.validate()?;
        match options.device {
            DqnDevice::Cpu => {
                let agent = match seed {
                    Some(seed) => PPODiscrete::new_seeded(config.clone(), seed),
                    None => PPODiscrete::new(config.clone()),
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
                            super::gpu_ppo::GpuPpoDiscrete::load_checkpoint(&context, path)
                                .map_err(gpu_error)?
                        }
                        None => super::gpu_ppo::GpuPpoDiscrete::new_seeded(
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
    pub(super) fn config(&self) -> &PPODiscreteConfig {
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
    ) -> Result<(usize, f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.select_action_with_rng(state, rng)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.select_action_with_rng(state, rng).map_err(gpu_error),
        }
    }
    pub(super) fn select_action(&self, state: &[f32]) -> Result<(usize, f32, f32), TrainerError> {
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
        batch: &RolloutBatch,
        rng: &mut R,
    ) -> Result<(f32, f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.train_on_batch_with_rng(batch, rng)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => {
                let m = agent
                    .train_on_batch_with_rng(batch, rng)
                    .map_err(gpu_error)?;
                Ok((m.policy_loss, m.value_loss, m.entropy))
            }
        }
    }
    pub(super) fn train_on_batch(
        &mut self,
        batch: &RolloutBatch,
    ) -> Result<(f32, f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.train_on_batch(batch)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => {
                let m = agent
                    .train_on_batch_with_rng(batch, &mut rand::thread_rng())
                    .map_err(gpu_error)?;
                Ok((m.policy_loss, m.value_loss, m.entropy))
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
