//! Backend values are constructed inside the owning worker; GPU Rc values stay there.
pub use super::DqnDevice as Td3Device;
use super::{DqnDevice, DqnRuntimeOptions, TD3Config, TD3};
use crate::{buffer::ContinuousTransitionBatch, runtime::trainer::TrainerError};
use rand::Rng;
use std::path::PathBuf;
/// Resume restores model/Adam/configuration/clocks, with fresh replay and random streams.
#[derive(Clone, Debug)]
pub struct Td3RuntimeOptions {
    pub device: Td3Device,
    pub resume: Option<PathBuf>,
    pub checkpoint: Option<PathBuf>,
    pub replay_capacity: usize,
    pub batch_size: usize,
    /// Uniform-action warm-up transitions at the start of each run.
    pub start_steps: usize,
    /// Minimum collected transitions before replay training starts.
    pub learning_starts: usize,
    /// Exploration deviation in physical action units after warm-up.
    pub exploration_std: f32,
}
impl Default for Td3RuntimeOptions {
    fn default() -> Self {
        Self {
            device: Td3Device::Cpu,
            resume: None,
            checkpoint: None,
            replay_capacity: 100_000,
            batch_size: 64,
            start_steps: 1_000,
            learning_starts: 64,
            exploration_std: 0.1,
        }
    }
}
impl From<DqnRuntimeOptions> for Td3RuntimeOptions {
    fn from(o: DqnRuntimeOptions) -> Self {
        Self {
            device: o.device,
            resume: o.resume,
            checkpoint: o.checkpoint,
            ..Self::default()
        }
    }
}
impl Td3RuntimeOptions {
    pub fn validate(&self) -> Result<(), TrainerError> {
        let message = if self.device == Td3Device::Cpu
            && (self.resume.is_some() || self.checkpoint.is_some())
        {
            Some("TD3 runtime --resume and --checkpoint require --device gpu")
        } else if self.device == Td3Device::Gpu && !cfg!(feature = "gpu") {
            Some("GPU support is not compiled in; rebuild with --features gpu")
        } else if self.replay_capacity == 0
            || self.batch_size == 0
            || self.batch_size > self.replay_capacity
            || self.learning_starts == 0
            || self.learning_starts > self.replay_capacity
            || !self.exploration_std.is_finite()
            || self.exploration_std < 0.
        {
            Some("TD3 requires positive compatible replay/batch/learning-start sizes and finite nonnegative exploration deviation")
        } else {
            None
        };
        message.map_or(Ok(()), |message| {
            Err(TrainerError {
                message: message.into(),
            })
        })
    }
}
pub(super) enum Td3Backend {
    Cpu {
        agent: Box<TD3>,
        config: TD3Config,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<super::gpu_td3::GpuTd3>),
}
impl Td3Backend {
    pub(super) fn new(
        config: TD3Config,
        seed: Option<u64>,
        options: &Td3RuntimeOptions,
    ) -> Result<Self, TrainerError> {
        options.validate()?;
        match options.device {
            DqnDevice::Cpu => {
                super::td3::validate_td3_config(&config).map_err(error)?;
                let agent = match seed {
                    Some(seed) => TD3::new_seeded(config.clone(), seed),
                    None => TD3::new(config.clone()),
                };
                Ok(Self::Cpu {
                    agent: Box::new(agent),
                    config,
                })
            }
            DqnDevice::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let context = rustforge_tensor::gpu::GpuContext::new().map_err(error)?;
                    let agent = match &options.resume {
                        Some(path) => super::gpu_td3::GpuTd3::load_checkpoint(&context, path)
                            .map_err(error)?,
                        None => super::gpu_td3::GpuTd3::new_seeded(
                            &context,
                            config,
                            seed.unwrap_or_else(rand::random),
                        )
                        .map_err(error)?,
                    };
                    Ok(Self::Gpu(Box::new(agent)))
                }
                #[cfg(not(feature = "gpu"))]
                Err(error("GPU support is not compiled in"))
            }
        }
    }
    pub(super) fn config(&self) -> &TD3Config {
        match self {
            Self::Cpu { config, .. } => config,
            #[cfg(feature = "gpu")]
            Self::Gpu(a) => a.config(),
        }
    }
    pub(super) fn select_action(
        &self,
        state: &[f32],
        std: f32,
        rng: &mut impl Rng,
    ) -> Result<Vec<f32>, TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.select_action_with_rng(state, std, rng)),
            #[cfg(feature = "gpu")]
            Self::Gpu(a) => a.select_action_with_rng(state, std, rng).map_err(error),
        }
    }
    pub(super) fn train(
        &mut self,
        batch: &ContinuousTransitionBatch,
        rng: &mut impl Rng,
    ) -> Result<(f32, Option<f32>), TrainerError> {
        let metrics = match self {
            Self::Cpu { agent, .. } => agent.train_step_with_rng(batch, rng),
            #[cfg(feature = "gpu")]
            Self::Gpu(a) => a.train_step_with_rng(batch, rng).map_err(error)?,
        };
        if !metrics.0.is_finite() || metrics.1.is_some_and(|v| !v.is_finite()) {
            return Err(error("nonfinite TD3 update loss"));
        }
        Ok(metrics)
    }
    pub(super) fn save(&self, options: &Td3RuntimeOptions) -> Result<(), TrainerError> {
        if let Some(path) = &options.checkpoint {
            match self {
                #[cfg(feature = "gpu")]
                Self::Gpu(a) => a.save_checkpoint(path).map_err(error)?,
                Self::Cpu { .. } => {
                    return Err(error(format!(
                        "CPU TD3 cannot write GPU checkpoint {}",
                        path.display()
                    )))
                }
            }
        }
        Ok(())
    }
}
fn error(e: impl std::fmt::Display) -> TrainerError {
    TrainerError {
        message: format!("TD3: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_options_validate_cpu_checkpoint_and_replay_noise_constraints() {
        let c = Td3RuntimeOptions::default();
        c.validate().unwrap();
        for kind in 0..8 {
            let mut bad = c.clone();
            match kind {
                0 => bad.resume = Some("in.chk".into()),
                1 => bad.checkpoint = Some("out.chk".into()),
                2 => bad.replay_capacity = 0,
                3 => bad.batch_size = 0,
                4 => bad.batch_size = bad.replay_capacity + 1,
                5 => bad.learning_starts = 0,
                6 => bad.learning_starts = bad.replay_capacity + 1,
                _ => bad.exploration_std = f32::NAN,
            };
            assert!(bad.validate().is_err());
        }
        let gpu = Td3RuntimeOptions {
            device: Td3Device::Gpu,
            ..c
        };
        assert_eq!(gpu.validate().is_ok(), cfg!(feature = "gpu"));
    }
}
