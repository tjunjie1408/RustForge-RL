//! Continuous PPO v1. Environment, rollout, RNG and gradients restart on load.
use super::{GpuPpoContinuous, Result};
use crate::agent::{
    gpu_ppo::{
        agent::checkpoint::{codec, invalid, write_atomic, SavedAdam, SavedTensor},
        GpuPpoCheckpointError, MAX_CHECKPOINT_BYTES,
    },
    PPOContinuousConfig,
};
use bincode::Options;
use rustforge_autograd::gpu::GpuVariable;
use rustforge_nn::gpu::GpuModule;
use rustforge_tensor::gpu::GpuContext;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, path::Path};
pub const CONTINUOUS_CHECKPOINT_MAGIC: &[u8; 8] = b"RFGPUPC0";
pub const CONTINUOUS_CHECKPOINT_VERSION: u32 = 1;
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SavedContinuous {
    config: PPOContinuousConfig,
    actor: Vec<SavedTensor>,
    critic: Vec<SavedTensor>,
    actor_adam: SavedAdam,
    critic_adam: SavedAdam,
    actor_updates: u64,
    critic_updates: u64,
}
fn shapes(c: &PPOContinuousConfig) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let h = c.base.hidden_dim;
    let mut trunk = vec![vec![h, c.base.obs_dim], vec![h], vec![h, h], vec![h]];
    let mut actor = trunk.clone();
    actor.extend([
        vec![c.act_dim, h],
        vec![c.act_dim],
        vec![c.act_dim, h],
        vec![c.act_dim],
    ]);
    trunk.extend([vec![1, h], vec![1]]);
    (actor, trunk)
}
fn validate_network(
    parameters: &[SavedTensor],
    adam: &SavedAdam,
    clock: u64,
    shapes: &[Vec<usize>],
    lr: f32,
) -> Result<()> {
    if parameters.len() != shapes.len() || adam.moments.len() != shapes.len() {
        return Err(invalid(
            "continuous checkpoint parameter/moment count mismatch",
        ));
    }
    let step = usize::try_from(clock)
        .map_err(|_| invalid("continuous PPO clock exceeds this platform"))?;
    if step == usize::MAX || clock != adam.timestep || lr != adam.lr {
        return Err(invalid(
            "continuous PPO/Adam clocks or learning rates disagree",
        ));
    }
    for (p, shape) in parameters.iter().zip(shapes) {
        p.validate(shape, false)?;
    }
    for (moment, shape) in adam.moments.iter().zip(shapes) {
        if moment.is_some() != (step > 0) {
            return Err(invalid("continuous checkpoint moment/progress mismatch"));
        }
        if let Some(moment) = moment {
            moment.first.validate(shape, false)?;
            moment.second.validate(shape, true)?;
        }
    }
    adam.state()?.validate_for_shapes(shapes)?;
    Ok(())
}
impl SavedContinuous {
    fn validate(&self) -> Result<()> {
        GpuPpoContinuous::validate_config(&self.config)?;
        let (actor, critic) = shapes(&self.config);
        validate_network(
            &self.actor,
            &self.actor_adam,
            self.actor_updates,
            &actor,
            self.config.base.lr,
        )?;
        validate_network(
            &self.critic,
            &self.critic_adam,
            self.critic_updates,
            &critic,
            self.config.base.lr,
        )
    }
    fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        if codec()
            .serialized_size(self)
            .map_err(GpuPpoCheckpointError::from)?
            > MAX_CHECKPOINT_BYTES - 12
        {
            return Err(GpuPpoCheckpointError::TooLarge.into());
        }
        let mut data = CONTINUOUS_CHECKPOINT_MAGIC.to_vec();
        data.extend(CONTINUOUS_CHECKPOINT_VERSION.to_le_bytes());
        data.extend(
            codec()
                .serialize(self)
                .map_err(GpuPpoCheckpointError::from)?,
        );
        Ok(data)
    }
}
fn decode(data: &[u8]) -> Result<SavedContinuous> {
    if data.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(GpuPpoCheckpointError::TooLarge.into());
    }
    if data.len() < 12 || &data[..8] != CONTINUOUS_CHECKPOINT_MAGIC {
        return Err(invalid(
            "invalid or truncated continuous GPU PPO checkpoint header",
        ));
    }
    let version = u32::from_le_bytes(data[8..12].try_into().expect("validated header"));
    if version != CONTINUOUS_CHECKPOINT_VERSION {
        return Err(GpuPpoCheckpointError::UnsupportedVersion(version).into());
    }
    let saved: SavedContinuous = codec()
        .deserialize(&data[12..])
        .map_err(GpuPpoCheckpointError::from)?;
    saved.validate()?;
    Ok(saved)
}
impl GpuPpoContinuous {
    /// Atomically saves actor/critic parameters, both Adam states and individual clocks.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let capture = |p: Vec<GpuVariable>| {
            p.iter()
                .map(|p| Ok(SavedTensor::capture(&p.to_cpu()?)))
                .collect::<Result<Vec<_>>>()
        };
        let saved = SavedContinuous {
            config: self.config.clone(),
            actor: capture(self.actor.parameters())?,
            critic: capture(self.critic.parameters())?,
            actor_adam: SavedAdam::capture(&self.actor_optimizer.state()?),
            critic_adam: SavedAdam::capture(&self.critic_optimizer.state()?),
            actor_updates: self.actor_updates,
            critic_updates: self.critic_updates,
        };
        write_atomic(path.as_ref(), &saved.encode()?).map_err(GpuPpoCheckpointError::from)?;
        Ok(())
    }
    /// Validates the entire host state before allocating a replacement agent.
    pub fn load_checkpoint(context: &GpuContext, path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(GpuPpoCheckpointError::from)?;
        if file.metadata().map_err(GpuPpoCheckpointError::from)?.len() > MAX_CHECKPOINT_BYTES {
            return Err(GpuPpoCheckpointError::TooLarge.into());
        }
        let mut data = Vec::new();
        file.take(MAX_CHECKPOINT_BYTES + 1)
            .read_to_end(&mut data)
            .map_err(GpuPpoCheckpointError::from)?;
        let saved = decode(&data)?;
        let mut agent = Self::new_seeded(context, saved.config, 0)?;
        for (p, t) in agent
            .actor
            .parameters()
            .iter()
            .zip(&saved.actor)
            .chain(agent.critic.parameters().iter().zip(&saved.critic))
        {
            p.copy_data_from(&GpuVariable::new(context, &t.tensor(), false)?)?;
        }
        agent
            .actor_optimizer
            .restore_state(&saved.actor_adam.state()?)?;
        agent
            .critic_optimizer
            .restore_state(&saved.critic_adam.state()?)?;
        agent.actor_updates = saved.actor_updates;
        agent.critic_updates = saved.critic_updates;
        Ok(agent)
    }
    pub fn restore_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let candidate = Self::load_checkpoint(&self.context, path)?;
        *self = candidate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::PPOConfig;
    use rustforge_autograd::gpu::{GpuAdamMomentState, GpuAdamState};
    use rustforge_tensor::Tensor;
    fn fixture() -> SavedContinuous {
        let config = PPOContinuousConfig {
            base: PPOConfig {
                obs_dim: 2,
                hidden_dim: 3,
                ..Default::default()
            },
            act_dim: 2,
            action_low: vec![-1., -2.],
            action_high: vec![1., 3.],
        };
        let (a, c) = shapes(&config);
        let tensors = |shapes: &[Vec<usize>]| {
            shapes
                .iter()
                .map(|s| SavedTensor::capture(&Tensor::full(s, 0.1)))
                .collect()
        };
        let adam = |shapes: &[Vec<usize>]| {
            SavedAdam::capture(&GpuAdamState {
                lr: config.base.lr,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                timestep: 1,
                moments: shapes
                    .iter()
                    .map(|s| {
                        Some(GpuAdamMomentState {
                            first: Tensor::full(s, 0.1),
                            second: Tensor::full(s, 0.01),
                        })
                    })
                    .collect(),
            })
        };
        SavedContinuous {
            actor: tensors(&a),
            critic: tensors(&c),
            actor_adam: adam(&a),
            critic_adam: adam(&c),
            actor_updates: 1,
            critic_updates: 1,
            config,
        }
    }
    fn unchecked(s: &SavedContinuous) -> Vec<u8> {
        let mut bytes = CONTINUOUS_CHECKPOINT_MAGIC.to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(codec().serialize(s).unwrap());
        bytes
    }
    #[test]
    fn continuous_codec_preserves_bits_and_independent_clocks_and_rejects_other_formats() {
        let mut s = fixture();
        s.actor[0].values[0] = -0.;
        s.actor_updates = 2;
        s.actor_adam.timestep = 2;
        let bytes = s.encode().unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(
            restored.actor[0].tensor().to_vec()[0].to_bits(),
            (-0f32).to_bits()
        );
        assert_eq!(restored.config, s.config);
        assert_eq!((restored.actor_updates, restored.critic_updates), (2, 1));
        for end in [0, 7, 11, 12, bytes.len() - 1] {
            assert!(decode(&bytes[..end]).is_err());
        }
        for magic in [b"RFGPUPPO", b"RFGPUDQN"] {
            let mut bad = bytes.clone();
            bad[..8].copy_from_slice(magic);
            assert!(decode(&bad).is_err());
        }
        let mut bad = bytes.clone();
        bad[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(decode(&bad).is_err());
        let mut bad = bytes;
        bad.push(0);
        assert!(decode(&bad).is_err());
    }
    #[test]
    fn malformed_continuous_metadata_and_moments_fail_before_device_allocation() {
        for kind in 0..15 {
            let mut s = fixture();
            match kind {
                0 => {
                    s.actor.pop();
                }
                1 => {
                    s.critic.pop();
                }
                2 => s.actor[0].shape = vec![6],
                3 => s.config.action_low.pop().map(|_| ()).unwrap(),
                4 => s.config.action_high[0] = s.config.action_low[0],
                5 => s.config.action_low[0] = f32::NAN,
                6 => s.actor_adam.timestep = 2,
                7 => s.critic_adam.lr = 1.,
                8 => s.actor_adam.moments[0] = None,
                9 => s.critic_adam.moments[0].as_mut().unwrap().second.values[0] = -1.,
                10 => s.actor[0].values[0] = f32::INFINITY,
                11 => s.config.base.hidden_dim = usize::MAX,
                12 => {
                    s.critic_updates = u64::MAX;
                    s.critic_adam.timestep = u64::MAX;
                }
                13 => s.config.base.ppo_epochs = 0,
                _ => {
                    s.critic[0].values.pop();
                }
            }
            assert!(decode(&unchecked(&s)).is_err(), "case {kind}");
        }
        let mut s = fixture();
        s.actor_updates = 0;
        s.actor_adam.timestep = 0;
        s.actor_adam.moments.fill(None);
        assert!(decode(&s.encode().unwrap()).is_ok());
    }
}
