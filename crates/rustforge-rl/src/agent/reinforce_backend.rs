//! Backend values are constructed inside the worker; GPU Rc values never cross threads.
use super::{DqnDevice, DqnRuntimeOptions, REINFORCEConfig, REINFORCE};
use crate::{buffer::RolloutBatch, runtime::trainer::TrainerError};
use rand::Rng;
use std::path::PathBuf;

pub use super::DqnDevice as ReinforceDevice;
/// GPU checkpoints restore model/Adam/configuration/updates. Environment,
/// rollout, run counters, metrics and action RNG start a fresh run.
#[derive(Clone, Debug, Default)]
pub struct ReinforceRuntimeOptions {
    pub device: ReinforceDevice,
    pub resume: Option<PathBuf>,
    pub checkpoint: Option<PathBuf>,
}
impl From<DqnRuntimeOptions> for ReinforceRuntimeOptions {
    fn from(options: DqnRuntimeOptions) -> Self {
        Self {
            device: options.device,
            resume: options.resume,
            checkpoint: options.checkpoint,
        }
    }
}
impl ReinforceRuntimeOptions {
    pub fn validate(&self) -> Result<(), TrainerError> {
        let message = match self.device {
            ReinforceDevice::Cpu if self.resume.is_some() || self.checkpoint.is_some() => {
                Some("REINFORCE runtime --resume and --checkpoint require --device gpu")
            }
            ReinforceDevice::Gpu if !cfg!(feature = "gpu") => {
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
pub(super) enum ReinforceBackend {
    Cpu {
        agent: Box<REINFORCE>,
        config: REINFORCEConfig,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<super::gpu_reinforce::GpuReinforce>),
}
impl ReinforceBackend {
    pub(super) fn new(
        config: REINFORCEConfig,
        seed: Option<u64>,
        options: &ReinforceRuntimeOptions,
    ) -> Result<Self, TrainerError> {
        options.validate()?;
        match options.device {
            DqnDevice::Cpu => {
                super::reinforce::validate_reinforce_config(&config).map_err(|message| {
                    TrainerError {
                        message: message.into(),
                    }
                })?;
                let agent = match seed {
                    Some(seed) => REINFORCE::new_seeded(config.clone(), seed),
                    None => REINFORCE::new(config.clone()),
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
                            super::gpu_reinforce::GpuReinforce::load_checkpoint(&context, path)
                                .map_err(gpu_error)?
                        }
                        None => super::gpu_reinforce::GpuReinforce::new_seeded(
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
    pub(super) fn config(&self) -> &REINFORCEConfig {
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
    ) -> Result<usize, TrainerError> {
        match self {
            Self::Cpu { agent, .. } => {
                let logits = rustforge_autograd::no_grad(|| agent.forward(state));
                if logits.data().to_vec().iter().any(|v| !v.is_finite()) {
                    return Err(TrainerError {
                        message: "REINFORCE logits must be finite".into(),
                    });
                }
                let action = REINFORCE::sample_action(&logits.data(), rng);
                Ok(action)
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent
                .select_action_with_rng(state, rng)
                .map(|v| v.0)
                .map_err(gpu_error),
        }
    }
    pub(super) fn select_action(&self, state: &[f32]) -> Result<usize, TrainerError> {
        self.select_action_with_rng(state, &mut rand::thread_rng())
    }
    pub(super) fn train_on_rollout(&mut self, batch: &RolloutBatch) -> Result<f32, TrainerError> {
        let loss = match self {
            Self::Cpu { agent, .. } => agent.train_on_rollout(batch),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.train_on_rollout(batch).map_err(gpu_error)?,
        };
        if !loss.is_finite() {
            return Err(TrainerError {
                message: "REINFORCE loss must be finite".into(),
            });
        }
        Ok(loss)
    }
    pub(super) fn save(&self, options: &ReinforceRuntimeOptions) -> Result<(), TrainerError> {
        if let Some(path) = &options.checkpoint {
            match self {
                #[cfg(feature = "gpu")]
                Self::Gpu(agent) => agent.save_checkpoint(path).map_err(gpu_error)?,
                Self::Cpu { .. } => {
                    return Err(TrainerError {
                        message: format!(
                            "CPU REINFORCE runtime cannot write GPU checkpoint {}",
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
        message: format!("GPU REINFORCE: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reinforce_runtime_options_reject_cpu_checkpoint_flags_and_check_feature_without_adapter() {
        assert!(ReinforceRuntimeOptions::default().validate().is_ok());
        for resume in [false, true] {
            let options = ReinforceRuntimeOptions {
                resume: resume.then(|| PathBuf::from("agent.chk")),
                checkpoint: (!resume).then(|| PathBuf::from("agent.chk")),
                ..Default::default()
            };
            assert!(options.validate().is_err());
        }
        assert_eq!(
            ReinforceRuntimeOptions {
                device: ReinforceDevice::Gpu,
                ..Default::default()
            }
            .validate()
            .is_ok(),
            cfg!(feature = "gpu")
        );
    }
}
