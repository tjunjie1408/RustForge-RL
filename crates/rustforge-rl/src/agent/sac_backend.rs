//! Backend values are constructed inside the owning worker; GPU Rc values stay there.
pub use super::DqnDevice as SacDevice;
use super::{DqnDevice, DqnRuntimeOptions, SACConfig, SAC};
use crate::{buffer::ContinuousTransitionBatch, runtime::trainer::TrainerError};
use rand::Rng;
use std::path::PathBuf;
/// Resume restores model/Adam/configuration/clocks, with fresh replay and random streams.
#[derive(Clone, Debug)]
pub struct SacRuntimeOptions {
    pub device: SacDevice,
    pub resume: Option<PathBuf>,
    pub checkpoint: Option<PathBuf>,
    pub replay_capacity: usize,
    pub batch_size: usize,
    /// Uniform-action warm-up transitions at the start of each run.
    pub start_steps: usize,
    /// Minimum collected transitions before replay training starts.
    pub learning_starts: usize,
}
impl Default for SacRuntimeOptions {
    fn default() -> Self {
        Self {
            device: SacDevice::Cpu,
            resume: None,
            checkpoint: None,
            replay_capacity: 100_000,
            batch_size: 64,
            start_steps: 1_000,
            learning_starts: 64,
        }
    }
}
impl From<DqnRuntimeOptions> for SacRuntimeOptions {
    fn from(o: DqnRuntimeOptions) -> Self {
        Self {
            device: o.device,
            resume: o.resume,
            checkpoint: o.checkpoint,
            ..Self::default()
        }
    }
}
impl SacRuntimeOptions {
    pub fn validate(&self) -> Result<(), TrainerError> {
        let message = if self.device == SacDevice::Cpu
            && (self.resume.is_some() || self.checkpoint.is_some())
        {
            Some("SAC runtime --resume and --checkpoint require --device gpu")
        } else if self.device == SacDevice::Gpu && !cfg!(feature = "gpu") {
            Some("GPU support is not compiled in; rebuild with --features gpu")
        } else if self.replay_capacity == 0
            || self.batch_size == 0
            || self.batch_size > self.replay_capacity
            || self.learning_starts == 0
            || self.learning_starts > self.replay_capacity
        {
            Some("SAC requires positive compatible replay/batch/learning-start sizes")
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
pub(super) enum SacBackend {
    Cpu {
        agent: Box<SAC>,
        config: SACConfig,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<super::gpu_sac::GpuSac>),
}
impl SacBackend {
    pub(super) fn new(
        config: SACConfig,
        seed: Option<u64>,
        options: &SacRuntimeOptions,
    ) -> Result<Self, TrainerError> {
        options.validate()?;
        match options.device {
            DqnDevice::Cpu => {
                super::sac::validate_sac_config(&config).map_err(error)?;
                let agent = match seed {
                    Some(seed) => SAC::new_seeded(config.clone(), seed),
                    None => SAC::new(config.clone()),
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
                        Some(path) => super::gpu_sac::GpuSac::load_checkpoint(&context, path)
                            .map_err(error)?,
                        None => super::gpu_sac::GpuSac::new_seeded(
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
    pub(super) fn config(&self) -> &SACConfig {
        match self {
            Self::Cpu { config, .. } => config,
            #[cfg(feature = "gpu")]
            Self::Gpu(a) => a.config(),
        }
    }
    pub(super) fn select_action(
        &self,
        state: &[f32],
        rng: &mut impl Rng,
    ) -> Result<Vec<f32>, TrainerError> {
        match self {
            Self::Cpu { agent, .. } => Ok(agent.select_action_with_rng(state, rng)),
            #[cfg(feature = "gpu")]
            Self::Gpu(a) => a.select_action_with_rng(state, rng).map_err(error),
        }
    }
    pub(super) fn train(
        &mut self,
        batch: &ContinuousTransitionBatch,
        target_rng: &mut impl Rng,
        actor_rng: &mut impl Rng,
    ) -> Result<(f32, f32, f32, f32), TrainerError> {
        let metrics = match self {
            Self::Cpu { agent, .. } => agent.train_step_with_rngs(batch, target_rng, actor_rng),
            #[cfg(feature = "gpu")]
            Self::Gpu(a) => a
                .train_step_with_rngs(batch, target_rng, actor_rng)
                .map_err(error)?,
        };
        if [metrics.0, metrics.1, metrics.2, metrics.3]
            .iter()
            .any(|v| !v.is_finite())
            || metrics.3 <= 0.
        {
            return Err(error("nonfinite SAC update loss"));
        }
        Ok(metrics)
    }
    pub(super) fn alpha(&self) -> Result<f32, TrainerError> {
        let alpha = match self {
            Self::Cpu { agent, .. } => agent.alpha(),
            #[cfg(feature = "gpu")]
            Self::Gpu(a) => a.alpha().map_err(error)?,
        };
        if !alpha.is_finite() || alpha <= 0. {
            return Err(error("SAC temperature must be finite and positive"));
        }
        Ok(alpha)
    }
    pub(super) fn save(&self, options: &SacRuntimeOptions) -> Result<(), TrainerError> {
        if let Some(path) = &options.checkpoint {
            match self {
                #[cfg(feature = "gpu")]
                Self::Gpu(a) => a.save_checkpoint(path).map_err(error)?,
                Self::Cpu { .. } => {
                    return Err(error(format!(
                        "CPU SAC cannot write GPU checkpoint {}",
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
        message: format!("SAC: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_options_validate_cpu_checkpoint_and_replay_constraints() {
        let c = SacRuntimeOptions::default();
        c.validate().unwrap();
        for kind in 0..7 {
            let mut bad = c.clone();
            match kind {
                0 => bad.resume = Some("in.chk".into()),
                1 => bad.checkpoint = Some("out.chk".into()),
                2 => bad.replay_capacity = 0,
                3 => bad.batch_size = 0,
                4 => bad.batch_size = bad.replay_capacity + 1,
                5 => bad.learning_starts = 0,
                _ => bad.learning_starts = bad.replay_capacity + 1,
            };
            assert!(bad.validate().is_err());
        }
        let gpu = SacRuntimeOptions {
            device: SacDevice::Gpu,
            ..c
        };
        assert_eq!(gpu.validate().is_ok(), cfg!(feature = "gpu"));
    }
}
