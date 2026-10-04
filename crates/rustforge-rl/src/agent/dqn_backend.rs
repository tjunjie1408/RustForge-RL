//! Backend construction stays inside the thread that owns the training loop.

use std::path::PathBuf;

use rustforge_autograd::{no_grad, Variable};
use rustforge_nn::Module;
use rustforge_tensor::Tensor;

use super::{DQNConfig, EpsilonGreedy, DQN};
use crate::buffer::TransitionBatch;
use crate::runtime::trainer::TrainerError;

/// Explicit backend selection. GPU requests never silently use the CPU DQN.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DqnDevice {
    #[default]
    Cpu,
    Gpu,
}

/// Execution configuration passed to the owning training worker.
/// GPU resume restores the agent, optimizer and target cadence, while replay,
/// exploration and environment state start a new run. Checkpoints are saved on
/// successful completion or controlled stop, using atomic file replacement.
#[derive(Clone, Debug, Default)]
pub struct DqnRuntimeOptions {
    pub device: DqnDevice,
    pub resume: Option<PathBuf>,
    pub checkpoint: Option<PathBuf>,
}

impl DqnRuntimeOptions {
    /// Checks unsupported combinations without constructing an agent or device.
    pub fn validate(&self, _use_per: bool) -> Result<(), TrainerError> {
        let message = match self.device {
            DqnDevice::Cpu if self.resume.is_some() || self.checkpoint.is_some() =>
                Some("runtime --resume and --checkpoint require --device gpu; CPU parameter files remain available through the existing library API"),
            DqnDevice::Gpu if !cfg!(feature = "gpu") =>
                Some("GPU support is not compiled in; rebuild with --features gpu"),
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

pub(super) enum DqnBackend {
    Cpu(Box<DQN>),
    #[cfg(feature = "gpu")]
    Gpu(Box<super::GpuDqn>),
}

impl DqnBackend {
    pub(super) fn new(
        config: DQNConfig,
        options: &DqnRuntimeOptions,
    ) -> Result<Self, TrainerError> {
        options.validate(config.use_per)?;
        match options.device {
            DqnDevice::Cpu => Ok(Self::Cpu(Box::new(DQN::new(config)))),
            DqnDevice::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let context = rustforge_tensor::gpu::GpuContext::new().map_err(gpu_error)?;
                    let agent = match &options.resume {
                        Some(path) => {
                            super::GpuDqn::load_checkpoint(&context, path).map_err(gpu_error)?
                        }
                        None => {
                            super::GpuDqn::new_seeded(&context, config, 2026).map_err(gpu_error)?
                        }
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

    pub(super) fn config(&self) -> &DQNConfig {
        match self {
            Self::Cpu(agent) => agent.config(),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent.config(),
        }
    }

    pub(super) fn select_action(
        &self,
        state: &[f32],
        explorer: &mut EpsilonGreedy,
        episode: usize,
        step: usize,
    ) -> Result<usize, TrainerError> {
        match self {
            Self::Cpu(agent) => {
                let input = Tensor::from_vec(state.to_vec(), &[1, self.config().obs_dim]);
                let output = no_grad(|| agent.q_net().forward(&Variable::from_tensor(input)));
                let q_values = output.data();
                super::dqn_runtime::ensure_finite_q_values(&q_values, episode, step)?;
                Ok(explorer.select_action(&q_values, step, self.config().num_actions))
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => {
                if state.iter().any(|value| !value.is_finite()) {
                    return Err(TrainerError { message: format!("GPU DQN observation is non-finite at episode {episode}, global step {step}") });
                }
                explorer.select_action_with(step, self.config().num_actions, || {
                    agent.select_greedy_action(state).map_err(gpu_error)
                })
            }
        }
    }

    pub(super) fn train_step(
        &mut self,
        batch: &TransitionBatch,
        weights: Option<&Tensor>,
    ) -> Result<(f32, Option<Vec<f32>>), TrainerError> {
        match self {
            Self::Cpu(agent) => Ok(agent.train_step(batch, weights)),
            #[cfg(feature = "gpu")]
            Self::Gpu(agent) => agent
                .train_step_with_weights(batch, weights)
                .map_err(gpu_error),
        }
    }

    pub(super) fn save(&self, options: &DqnRuntimeOptions) -> Result<(), TrainerError> {
        if let Some(path) = &options.checkpoint {
            match self {
                #[cfg(feature = "gpu")]
                Self::Gpu(agent) => agent.save_checkpoint(path).map_err(gpu_error)?,
                Self::Cpu(_) => {
                    return Err(TrainerError {
                        message: format!(
                            "CPU runtime cannot write GPU checkpoint {}",
                            path.display()
                        ),
                    })
                }
            }
        }
        Ok(())
    }

    pub(super) fn into_cpu(self) -> Result<DQN, TrainerError> {
        match self {
            Self::Cpu(agent) => Ok(*agent),
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => Err(TrainerError {
                message: "expected CPU DQN".into(),
            }),
        }
    }
}

#[cfg(feature = "gpu")]
fn gpu_error(error: impl std::fmt::Display) -> TrainerError {
    TrainerError {
        message: format!("GPU DQN: {error}"),
    }
}
