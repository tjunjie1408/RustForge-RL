//! Backend values are constructed inside the worker; GPU Rc values never cross threads.
use super::{A2CConfig, DqnDevice, DqnRuntimeOptions, A2C};
use crate::{buffer::RolloutBatch, runtime::trainer::TrainerError};
use rand::Rng;
use std::path::PathBuf;

pub use super::DqnDevice as A2cDevice;
/// GPU checkpoints restore model/Adam/configuration/updates. Environment,
/// rollout, run counters, metrics and action RNG start a fresh run.
#[derive(Clone, Debug, Default)]
pub struct A2cRuntimeOptions {
    pub device: A2cDevice,
    pub resume: Option<PathBuf>,
    pub checkpoint: Option<PathBuf>,
}
impl From<DqnRuntimeOptions> for A2cRuntimeOptions {
    fn from(options: DqnRuntimeOptions) -> Self {
        Self {
            device: options.device,
            resume: options.resume,
            checkpoint: options.checkpoint,
        }
    }
}
impl A2cRuntimeOptions {
    pub fn validate(&self) -> Result<(), TrainerError> {
        let message = match self.device {
            A2cDevice::Cpu if self.resume.is_some() || self.checkpoint.is_some() => {
                Some("A2C runtime --resume and --checkpoint require --device gpu")
            }
            A2cDevice::Gpu if !cfg!(feature = "gpu") => {
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
pub(super) enum A2cBackend {
    Cpu {
        agent: Box<A2C>,
        config: A2CConfig,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<super::gpu_a2c::GpuA2c>),
}
impl A2cBackend {
    pub(super) fn new(
        config: A2CConfig,
        seed: Option<u64>,
        options: &A2cRuntimeOptions,
    ) -> Result<Self, TrainerError> {
        options.validate()?;
        match options.device {
            DqnDevice::Cpu => {
                super::a2c::validate_a2c_config(&config).map_err(|message| TrainerError {
                    message: message.into(),
                })?;
                let agent = match seed {
                    Some(seed) => A2C::new_seeded(config.clone(), seed),
                    None => A2C::new(config.clone()),
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
                        Some(path) => super::gpu_a2c::GpuA2c::load_checkpoint(&context, path)
                            .map_err(gpu_error)?,
                        None => super::gpu_a2c::GpuA2c::new_seeded(
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
    pub(super) fn config(&self) -> &A2CConfig {
        match self {
            Self::Cpu { config, .. } => config,
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.config(),
        }
    }
    pub(super) fn select_action_with_rng<R: Rng>(
        &self,
        state: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => {
                let (logits, value) = rustforge_autograd::no_grad(|| agent.forward(state));
                let action = A2C::sample_action(&logits.data(), rng);
                let value = value.data().item();
                Ok((action, value))
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => {
                let (action, _, value) = agent
                    .select_action_with_rng(state, rng)
                    .map_err(gpu_error)?;
                Ok((action, value))
            }
        }
    }
    pub(super) fn select_action(&self, state: &[f32]) -> Result<(usize, f32), TrainerError> {
        self.select_action_with_rng(state, &mut rand::thread_rng())
    }
    pub(super) fn value_of(&self, state: &[f32]) -> Result<f32, TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.value_of(state)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.value_of(state).map_err(gpu_error),
        }
    }
    pub(super) fn train_on_rollout(
        &mut self,
        batch: &RolloutBatch,
    ) -> Result<(f32, f32, f32, f32), TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.train_on_rollout(batch)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => {
                let m = agent.train_on_rollout(batch).map_err(gpu_error)?;
                Ok((m.total_loss, m.actor_loss, m.value_loss, m.entropy))
            }
        }
    }
    pub(super) fn save(&self, options: &A2cRuntimeOptions) -> Result<(), TrainerError> {
        if let Some(path) = &options.checkpoint {
            match self {
                #[cfg(feature = "gpu")]
                Self::Gpu(agent) => agent.save_checkpoint(path).map_err(gpu_error)?,
                Self::Cpu { .. } => {
                    return Err(TrainerError {
                        message: format!(
                            "CPU A2C runtime cannot write GPU checkpoint {}",
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
        message: format!("GPU A2C: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a2c_runtime_options_reject_cpu_checkpoint_flags_and_check_feature_without_adapter() {
        assert!(A2cRuntimeOptions::default().validate().is_ok());
        for resume in [false, true] {
            let options = A2cRuntimeOptions {
                resume: resume.then(|| PathBuf::from("agent.chk")),
                checkpoint: (!resume).then(|| PathBuf::from("agent.chk")),
                ..Default::default()
            };
            assert!(options.validate().is_err());
        }
        assert_eq!(
            A2cRuntimeOptions {
                device: A2cDevice::Gpu,
                ..Default::default()
            }
            .validate()
            .is_ok(),
            cfg!(feature = "gpu")
        );
    }
}
