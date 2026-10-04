//! GPU SAC checkpoint v1: bounded little-endian tensor/temperature/Adam state.
//! Rollout, environment, RNG and gradients are intentionally not persisted.
use super::{GpuSac, GpuSacError, Result, SACConfig};
use bincode::Options;
use rustforge_autograd::gpu::{GpuAdamMomentState, GpuAdamState, GpuVariable};
use rustforge_nn::gpu::GpuModule;
use rustforge_tensor::{gpu::GpuContext, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub const CHECKPOINT_MAGIC: &[u8; 8] = b"RFGPUSAC";
pub const CHECKPOINT_VERSION: u32 = 1;
/// Maximum complete file size, including its 12-byte header.
pub const MAX_CHECKPOINT_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
pub enum GpuSacCheckpointError {
    Io(io::Error),
    Encoding(bincode::Error),
    UnsupportedVersion(u32),
    InvalidState(&'static str),
    TooLarge,
}
impl std::fmt::Display for GpuSacCheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GPU checkpoint I/O error: {e}"),
            Self::Encoding(e) => write!(f, "GPU checkpoint encoding error: {e}"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GPU SAC checkpoint version {v}")
            }
            Self::InvalidState(message) => f.write_str(message),
            Self::TooLarge => write!(f, "GPU checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes"),
        }
    }
}
impl std::error::Error for GpuSacCheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Encoding(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for GpuSacCheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<bincode::Error> for GpuSacCheckpointError {
    fn from(e: bincode::Error) -> Self {
        Self::Encoding(e)
    }
}
fn invalid(message: &'static str) -> GpuSacError {
    GpuSacCheckpointError::InvalidState(message).into()
}
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
        .with_limit(MAX_CHECKPOINT_BYTES - 12)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}
impl SavedTensor {
    fn capture(t: &Tensor) -> Self {
        Self {
            shape: t.shape().to_vec(),
            values: t.to_vec(),
        }
    }
    fn validate(&self, shape: &[usize], second_moment: bool) -> Result<()> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| invalid("checkpoint tensor shape overflow"))?;
        if self.shape != shape || self.values.len() != count {
            return Err(invalid("checkpoint tensor shape/data length mismatch"));
        }
        if self
            .values
            .iter()
            .any(|&v| !v.is_finite() || (second_moment && v < 0.))
        {
            return Err(invalid(
                "checkpoint tensors must be finite and Adam variances nonnegative",
            ));
        }
        Ok(())
    }
    // All wire shapes/data lengths must be validated before creating ndarrays.
    fn tensor(&self) -> Tensor {
        Tensor::from_vec(self.values.clone(), &self.shape)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedMoments {
    first: SavedTensor,
    second: SavedTensor,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedAdam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: u64,
    moments: Vec<Option<SavedMoments>>,
}
impl SavedAdam {
    fn capture(s: &GpuAdamState) -> Self {
        Self {
            lr: s.lr,
            beta1: s.beta1,
            beta2: s.beta2,
            epsilon: s.epsilon,
            timestep: s.timestep as u64,
            moments: s
                .moments
                .iter()
                .map(|m| {
                    m.as_ref().map(|m| SavedMoments {
                        first: SavedTensor::capture(&m.first),
                        second: SavedTensor::capture(&m.second),
                    })
                })
                .collect(),
        }
    }
    fn state(&self) -> Result<GpuAdamState> {
        Ok(GpuAdamState {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            timestep: usize::try_from(self.timestep)
                .map_err(|_| invalid("Adam timestep exceeds this platform"))?,
            moments: self
                .moments
                .iter()
                .map(|m| {
                    m.as_ref().map(|m| GpuAdamMomentState {
                        first: m.first.tensor(),
                        second: m.second.tensor(),
                    })
                })
                .collect(),
        })
    }
}

fn create_temporary(path: &Path) -> io::Result<(PathBuf, File)> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint path must name a file",
        )
    })?;
    loop {
        let mut name = name.to_os_string();
        name.push(format!(
            ".tmp-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let temporary = path.with_file_name(name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let (temporary, mut file) = create_temporary(path)?;
    // The closure consumes/closes File even on an early write error, so
    // cleanup and rename do not race an open file handle on Windows.
    let written = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedSac {
    config: SACConfig,
    // actor, Q1, Q2, Q1 target, Q2 target
    networks: [Vec<SavedTensor>; 5],
    log_alpha: SavedTensor,
    actor_adam: SavedAdam,
    critic_adam: SavedAdam,
    updates: u64,
    alpha_adam: SavedAdam,
}
fn shapes(c: &SACConfig) -> [Vec<Vec<usize>>; 5] {
    let critic = vec![
        vec![c.hidden_dim, c.obs_dim + c.act_dim],
        vec![c.hidden_dim],
        vec![c.hidden_dim, c.hidden_dim],
        vec![c.hidden_dim],
        vec![1, c.hidden_dim],
        vec![1],
    ];
    let actor = vec![
        vec![c.hidden_dim, c.obs_dim],
        vec![c.hidden_dim],
        vec![c.hidden_dim, c.hidden_dim],
        vec![c.hidden_dim],
        vec![c.act_dim, c.hidden_dim],
        vec![c.act_dim],
        vec![c.act_dim, c.hidden_dim],
        vec![c.act_dim],
    ];
    [
        actor,
        critic.clone(),
        critic.clone(),
        critic.clone(),
        critic,
    ]
}
fn validate_adam(adam: &SavedAdam, shapes: &[Vec<usize>], clock: u64, lr: f32) -> Result<()> {
    if adam.moments.len() != shapes.len() || adam.timestep != clock || adam.lr != lr {
        return Err(invalid(
            "SAC checkpoint Adam shapes, clocks or learning rates disagree",
        ));
    }
    for (m, shape) in adam.moments.iter().zip(shapes) {
        if m.is_some() != (clock > 0) {
            return Err(invalid(
                "SAC checkpoint moments do not match update progress",
            ));
        }
        if let Some(m) = m {
            m.first.validate(shape, false)?;
            m.second.validate(shape, true)?;
        }
    }
    adam.state()?.validate_for_shapes(shapes)?;
    Ok(())
}
impl SavedSac {
    fn validate(&self) -> Result<()> {
        GpuSac::validate_config(&self.config)?;
        let critic = usize::try_from(self.updates)
            .map_err(|_| invalid("SAC critic clock exceeds platform"))?;
        if critic == usize::MAX {
            return Err(invalid("SAC checkpoint clock cannot advance"));
        }
        self.log_alpha.validate(&[1], false)?;
        let alpha = self.log_alpha.values[0].exp();
        if !alpha.is_finite() || alpha <= 0. {
            return Err(invalid(
                "SAC checkpoint temperature must be finite and positive",
            ));
        }
        let shapes = shapes(&self.config);
        for (network, shapes) in self.networks.iter().zip(&shapes) {
            if network.len() != shapes.len() {
                return Err(invalid("SAC checkpoint network parameter count mismatch"));
            }
            for (p, s) in network.iter().zip(shapes) {
                p.validate(s, false)?;
            }
        }
        validate_adam(
            &self.actor_adam,
            &shapes[0],
            self.updates,
            self.config.actor_lr,
        )?;
        let critics: Vec<_> = shapes[1].iter().chain(&shapes[2]).cloned().collect();
        validate_adam(
            &self.critic_adam,
            &critics,
            self.updates,
            self.config.critic_lr,
        )?;
        validate_adam(
            &self.alpha_adam,
            &[vec![1]],
            self.updates,
            self.config.alpha_lr,
        )?;
        Ok(())
    }
    fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        if codec()
            .serialized_size(self)
            .map_err(GpuSacCheckpointError::from)?
            > MAX_CHECKPOINT_BYTES - 12
        {
            return Err(GpuSacCheckpointError::TooLarge.into());
        }
        let mut data = CHECKPOINT_MAGIC.to_vec();
        data.extend(CHECKPOINT_VERSION.to_le_bytes());
        data.extend(
            codec()
                .serialize(self)
                .map_err(GpuSacCheckpointError::from)?,
        );
        Ok(data)
    }
}
fn decode(data: &[u8]) -> Result<SavedSac> {
    if data.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(GpuSacCheckpointError::TooLarge.into());
    }
    if data.len() < 12 || &data[..8] != CHECKPOINT_MAGIC {
        return Err(invalid("invalid or truncated GPU SAC checkpoint header"));
    }
    let version = u32::from_le_bytes(data[8..12].try_into().expect("validated header"));
    if version != CHECKPOINT_VERSION {
        return Err(GpuSacCheckpointError::UnsupportedVersion(version).into());
    }
    let saved: SavedSac = codec()
        .deserialize(&data[12..])
        .map_err(GpuSacCheckpointError::from)?;
    saved.validate()?;
    Ok(saved)
}
fn read_checkpoint(path: &Path) -> Result<SavedSac> {
    let file = File::open(path).map_err(GpuSacCheckpointError::from)?;
    if file.metadata().map_err(GpuSacCheckpointError::from)?.len() > MAX_CHECKPOINT_BYTES {
        return Err(GpuSacCheckpointError::TooLarge.into());
    }
    let mut data = Vec::new();
    file.take(MAX_CHECKPOINT_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(GpuSacCheckpointError::from)?;
    decode(&data)
}
impl GpuSac {
    /// Atomic versioned snapshot of five networks, temperature, three Adam states and update clock.
    /// Environment, replay, gradients, random streams and run counters restart on load.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let parameters = [
            self.actor().parameters(),
            self.critic1().parameters(),
            self.critic2().parameters(),
            self.critic1_target().parameters(),
            self.critic2_target().parameters(),
        ];
        let mut captured: [Vec<SavedTensor>; 5] = std::array::from_fn(|_| Vec::new());
        for (destination, parameters) in captured.iter_mut().zip(parameters) {
            *destination = parameters
                .iter()
                .map(|p| Ok(SavedTensor::capture(&p.to_cpu()?)))
                .collect::<Result<_>>()?;
        }
        super::checked_alpha(&self.log_alpha)?;
        let saved = SavedSac {
            config: self.config.clone(),
            networks: captured,
            log_alpha: SavedTensor::capture(&self.log_alpha.to_cpu()?),
            actor_adam: SavedAdam::capture(&self.actor_optimizer.state()?),
            critic_adam: SavedAdam::capture(&self.critic_optimizer.state()?),
            alpha_adam: SavedAdam::capture(&self.alpha_optimizer.state()?),
            updates: self.updates as u64,
        };
        write_atomic(path.as_ref(), &saved.encode()?).map_err(GpuSacCheckpointError::from)?;
        Ok(())
    }
    /// Validates the complete host state before allocating a replacement agent.
    pub fn load_checkpoint(context: &GpuContext, path: impl AsRef<Path>) -> Result<Self> {
        let saved = read_checkpoint(path.as_ref())?;
        let mut agent = Self::new_seeded(context, saved.config, 0)?;
        for (parameters, saved) in [
            agent.actor().parameters(),
            agent.critic1().parameters(),
            agent.critic2().parameters(),
            agent.critic1_target().parameters(),
            agent.critic2_target().parameters(),
        ]
        .into_iter()
        .zip(&saved.networks)
        {
            for (p, value) in parameters.iter().zip(saved) {
                p.copy_data_from(&GpuVariable::new(context, &value.tensor(), false)?)?;
            }
        }
        agent.log_alpha.copy_data_from(&GpuVariable::new(
            context,
            &saved.log_alpha.tensor(),
            false,
        )?)?;
        super::checked_alpha(&agent.log_alpha)?;
        agent
            .alpha_optimizer
            .restore_state(&saved.alpha_adam.state()?)?;
        agent
            .actor_optimizer
            .restore_state(&saved.actor_adam.state()?)?;
        agent
            .critic_optimizer
            .restore_state(&saved.critic_adam.state()?)?;
        agent.updates = usize::try_from(saved.updates)
            .map_err(|_| invalid("SAC critic clock exceeds platform"))?;
        Ok(agent)
    }
    /// A failed load preserves the complete live agent; a successful load replaces handles.
    pub fn restore_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let candidate = Self::load_checkpoint(&self.context, path)?;
        *self = candidate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> SavedSac {
        let mut c = SACConfig::new(2, 2, vec![-1., -2.], vec![2., 3.]);
        c.hidden_dim = 3;
        let shapes = shapes(&c);
        let adam = |s: &[Vec<usize>], lr, timestep| {
            SavedAdam::capture(&GpuAdamState {
                lr,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                timestep,
                moments: s
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
        SavedSac {
            networks: std::array::from_fn(|i| {
                shapes[i]
                    .iter()
                    .map(|s| SavedTensor::capture(&Tensor::full(s, 0.1)))
                    .collect()
            }),
            actor_adam: adam(&shapes[0], c.actor_lr, 3),
            critic_adam: adam(
                &shapes[1]
                    .iter()
                    .chain(&shapes[2])
                    .cloned()
                    .collect::<Vec<_>>(),
                c.critic_lr,
                3,
            ),
            updates: 3,
            log_alpha: SavedTensor::capture(&Tensor::from_vec(vec![0.2f32.ln()], &[1])),
            alpha_adam: adam(&[vec![1]], c.alpha_lr, 3),
            config: c,
        }
    }
    fn unchecked(s: &SavedSac) -> Vec<u8> {
        let mut b = CHECKPOINT_MAGIC.to_vec();
        b.extend(1u32.to_le_bytes());
        b.extend(codec().serialize(s).unwrap());
        b
    }
    #[test]
    fn codec_preserves_bits_five_networks_temperature_and_three_clocks_and_rejects_other_formats() {
        let mut s = fixture();
        s.networks[3][0].values[0] = -0.;
        let bytes = s.encode().unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(restored.config, s.config);
        assert_eq!(restored.updates, 3);
        assert_eq!(
            restored.log_alpha.values[0].to_bits(),
            s.log_alpha.values[0].to_bits()
        );
        assert_eq!(
            restored.networks[3][0].values[0].to_bits(),
            (-0f32).to_bits()
        );
        for end in [0, 7, 11, 12, bytes.len() - 1] {
            assert!(decode(&bytes[..end]).is_err());
        }
        for magic in [
            b"RFGPUPPO",
            b"RFGPUDQN",
            b"RFGPUREI",
            b"RFGPUPC0",
            b"RFGPUTD3",
        ] {
            let mut b = bytes.clone();
            b[..8].copy_from_slice(magic);
            assert!(decode(&b).is_err());
        }
        let mut b = bytes.clone();
        b[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(decode(&b).is_err());
        let mut b = bytes;
        b.push(0);
        assert!(decode(&b).is_err());
    }
    #[test]
    fn malformed_parameters_targets_clocks_and_moments_fail_on_host() {
        for kind in 0..23 {
            let mut s = fixture();
            match kind {
                0 => {
                    s.networks[0].pop();
                }
                1 => {
                    s.networks[4].pop();
                }
                2 => s.networks[1][0].shape = vec![12],
                3 => {
                    s.networks[4][0].values.pop();
                }
                4 => s.networks[2][0].values[0] = f32::NAN,
                5 => s.networks[3][0].values[0] = f32::INFINITY,
                6 => s.alpha_adam.timestep = 2,
                7 => s.critic_adam.timestep = 4,
                8 => s.actor_adam.lr = 1.,
                9 => s.critic_adam.moments[0] = None,
                10 => s.actor_adam.moments[0].as_mut().unwrap().second.values[0] = -1.,
                11 => s.critic_adam.beta1 = 1.,
                12 => s.config.hidden_dim = usize::MAX,
                13 => s.config.action_high[0] = s.config.action_low[0],
                14 => s.log_alpha.values[0] = 100.,
                15 => {
                    s.updates = u64::MAX;
                    s.critic_adam.timestep = u64::MAX;
                }
                16 => {
                    s.actor_adam.moments.pop();
                }
                17 => s.log_alpha.values[0] = -200.,
                18 => s.actor_adam.moments[0].as_mut().unwrap().first.shape = vec![999],
                19 => s.alpha_adam.moments[0] = None,
                20 => s.alpha_adam.moments[0].as_mut().unwrap().second.values[0] = -1.,
                21 => s.log_alpha.shape = vec![],
                _ => s.alpha_adam.lr = 1.,
            }
            assert!(decode(&unchecked(&s)).is_err(), "case {kind}");
        }
    }
    #[test]
    fn file_size_limit_and_atomic_replace_reject_failed_destinations_without_leftovers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.chk");
        write_atomic(&path, b"old").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        let target = dir.path().join("directory");
        fs::create_dir(&target).unwrap();
        assert!(write_atomic(&target, b"candidate").is_err());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
        let large = dir.path().join("large.chk");
        File::create(&large)
            .unwrap()
            .set_len(MAX_CHECKPOINT_BYTES + 1)
            .unwrap();
        assert!(matches!(
            read_checkpoint(&large),
            Err(GpuSacError::Checkpoint(GpuSacCheckpointError::TooLarge))
        ));
    }
}
