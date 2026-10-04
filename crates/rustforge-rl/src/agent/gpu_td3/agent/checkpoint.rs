//! GPU TD3 checkpoint v1: bounded little-endian tensor/Adam state.
//! Rollout, environment, RNG and gradients are intentionally not persisted.
use super::{GpuTd3, GpuTd3Error, Result, TD3Config};
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

pub const CHECKPOINT_MAGIC: &[u8; 8] = b"RFGPUTD3";
pub const CHECKPOINT_VERSION: u32 = 1;
/// Maximum complete file size, including its 12-byte header.
pub const MAX_CHECKPOINT_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
pub enum GpuTd3CheckpointError {
    Io(io::Error),
    Encoding(bincode::Error),
    UnsupportedVersion(u32),
    InvalidState(&'static str),
    TooLarge,
}
impl std::fmt::Display for GpuTd3CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GPU checkpoint I/O error: {e}"),
            Self::Encoding(e) => write!(f, "GPU checkpoint encoding error: {e}"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GPU TD3 checkpoint version {v}")
            }
            Self::InvalidState(message) => f.write_str(message),
            Self::TooLarge => write!(f, "GPU checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes"),
        }
    }
}
impl std::error::Error for GpuTd3CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Encoding(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for GpuTd3CheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<bincode::Error> for GpuTd3CheckpointError {
    fn from(e: bincode::Error) -> Self {
        Self::Encoding(e)
    }
}
fn invalid(message: &'static str) -> GpuTd3Error {
    GpuTd3CheckpointError::InvalidState(message).into()
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
struct SavedTd3 {
    config: TD3Config,
    // actor, Q1, Q2, actor target, Q1 target, Q2 target
    networks: [Vec<SavedTensor>; 6],
    actor_adam: SavedAdam,
    critic_adam: SavedAdam,
    updates: u64,
    actor_updates: u64,
}
fn shapes(c: &TD3Config) -> [Vec<Vec<usize>>; 6] {
    let net = |input, output| {
        vec![
            vec![c.hidden_dim, input],
            vec![c.hidden_dim],
            vec![c.hidden_dim, c.hidden_dim],
            vec![c.hidden_dim],
            vec![output, c.hidden_dim],
            vec![output],
        ]
    };
    let actor = net(c.obs_dim, c.act_dim);
    let critic = net(c.obs_dim + c.act_dim, 1);
    [
        actor.clone(),
        critic.clone(),
        critic.clone(),
        actor,
        critic.clone(),
        critic,
    ]
}
fn validate_adam(adam: &SavedAdam, shapes: &[Vec<usize>], clock: u64, lr: f32) -> Result<()> {
    if adam.moments.len() != shapes.len() || adam.timestep != clock || adam.lr != lr {
        return Err(invalid(
            "TD3 checkpoint Adam shapes, clocks or learning rates disagree",
        ));
    }
    for (m, shape) in adam.moments.iter().zip(shapes) {
        if m.is_some() != (clock > 0) {
            return Err(invalid(
                "TD3 checkpoint moments do not match update progress",
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
impl SavedTd3 {
    fn validate(&self) -> Result<()> {
        GpuTd3::validate_config(&self.config)?;
        let critic = usize::try_from(self.updates)
            .map_err(|_| invalid("TD3 critic clock exceeds platform"))?;
        let actor = usize::try_from(self.actor_updates)
            .map_err(|_| invalid("TD3 actor clock exceeds platform"))?;
        if critic == usize::MAX
            || actor == usize::MAX
            || actor != critic.checked_div(self.config.policy_delay).unwrap_or(0)
        {
            return Err(invalid(
                "TD3 checkpoint clocks do not match delayed update cadence",
            ));
        }
        let shapes = shapes(&self.config);
        for (network, shapes) in self.networks.iter().zip(&shapes) {
            if network.len() != shapes.len() {
                return Err(invalid("TD3 checkpoint network parameter count mismatch"));
            }
            for (p, s) in network.iter().zip(shapes) {
                p.validate(s, false)?;
            }
        }
        validate_adam(
            &self.actor_adam,
            &shapes[0],
            self.actor_updates,
            self.config.actor_lr,
        )?;
        let critics: Vec<_> = shapes[1].iter().chain(&shapes[2]).cloned().collect();
        validate_adam(
            &self.critic_adam,
            &critics,
            self.updates,
            self.config.critic_lr,
        )?;
        Ok(())
    }
    fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        if codec()
            .serialized_size(self)
            .map_err(GpuTd3CheckpointError::from)?
            > MAX_CHECKPOINT_BYTES - 12
        {
            return Err(GpuTd3CheckpointError::TooLarge.into());
        }
        let mut data = CHECKPOINT_MAGIC.to_vec();
        data.extend(CHECKPOINT_VERSION.to_le_bytes());
        data.extend(
            codec()
                .serialize(self)
                .map_err(GpuTd3CheckpointError::from)?,
        );
        Ok(data)
    }
}
fn decode(data: &[u8]) -> Result<SavedTd3> {
    if data.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(GpuTd3CheckpointError::TooLarge.into());
    }
    if data.len() < 12 || &data[..8] != CHECKPOINT_MAGIC {
        return Err(invalid("invalid or truncated GPU TD3 checkpoint header"));
    }
    let version = u32::from_le_bytes(data[8..12].try_into().expect("validated header"));
    if version != CHECKPOINT_VERSION {
        return Err(GpuTd3CheckpointError::UnsupportedVersion(version).into());
    }
    let saved: SavedTd3 = codec()
        .deserialize(&data[12..])
        .map_err(GpuTd3CheckpointError::from)?;
    saved.validate()?;
    Ok(saved)
}
fn read_checkpoint(path: &Path) -> Result<SavedTd3> {
    let file = File::open(path).map_err(GpuTd3CheckpointError::from)?;
    if file.metadata().map_err(GpuTd3CheckpointError::from)?.len() > MAX_CHECKPOINT_BYTES {
        return Err(GpuTd3CheckpointError::TooLarge.into());
    }
    let mut data = Vec::new();
    file.take(MAX_CHECKPOINT_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(GpuTd3CheckpointError::from)?;
    decode(&data)
}
impl GpuTd3 {
    /// Atomic versioned snapshot of all networks, both Adam states and delayed-update clocks.
    /// Environment, replay, gradients, random streams and run counters restart on load.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let networks = [
            self.actor(),
            self.critic1(),
            self.critic2(),
            self.actor_target(),
            self.critic1_target(),
            self.critic2_target(),
        ];
        let mut captured: [Vec<SavedTensor>; 6] = std::array::from_fn(|_| Vec::new());
        for (destination, network) in captured.iter_mut().zip(networks) {
            *destination = network
                .parameters()
                .iter()
                .map(|p| Ok(SavedTensor::capture(&p.to_cpu()?)))
                .collect::<Result<_>>()?;
        }
        let saved = SavedTd3 {
            config: self.config.clone(),
            networks: captured,
            actor_adam: SavedAdam::capture(&self.actor_optimizer.state()?),
            critic_adam: SavedAdam::capture(&self.critic_optimizer.state()?),
            updates: self.updates as u64,
            actor_updates: self.actor_updates as u64,
        };
        write_atomic(path.as_ref(), &saved.encode()?).map_err(GpuTd3CheckpointError::from)?;
        Ok(())
    }
    /// Validates the complete host state before allocating a replacement agent.
    pub fn load_checkpoint(context: &GpuContext, path: impl AsRef<Path>) -> Result<Self> {
        let saved = read_checkpoint(path.as_ref())?;
        let mut agent = Self::new_seeded(context, saved.config, 0)?;
        for (network, parameters) in [
            agent.actor(),
            agent.critic1(),
            agent.critic2(),
            agent.actor_target(),
            agent.critic1_target(),
            agent.critic2_target(),
        ]
        .into_iter()
        .zip(&saved.networks)
        {
            for (p, value) in network.parameters().iter().zip(parameters) {
                p.copy_data_from(&GpuVariable::new(context, &value.tensor(), false)?)?;
            }
        }
        agent
            .actor_optimizer
            .restore_state(&saved.actor_adam.state()?)?;
        agent
            .critic_optimizer
            .restore_state(&saved.critic_adam.state()?)?;
        agent.updates = usize::try_from(saved.updates)
            .map_err(|_| invalid("TD3 critic clock exceeds platform"))?;
        agent.actor_updates = usize::try_from(saved.actor_updates)
            .map_err(|_| invalid("TD3 actor clock exceeds platform"))?;
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
    fn fixture() -> SavedTd3 {
        let mut c = TD3Config::new(2, 2, vec![-1., -2.], vec![2., 3.]);
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
        SavedTd3 {
            networks: std::array::from_fn(|i| {
                shapes[i]
                    .iter()
                    .map(|s| SavedTensor::capture(&Tensor::full(s, 0.1)))
                    .collect()
            }),
            actor_adam: adam(&shapes[0], c.actor_lr, 1),
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
            actor_updates: 1,
            config: c,
        }
    }
    fn unchecked(s: &SavedTd3) -> Vec<u8> {
        let mut b = CHECKPOINT_MAGIC.to_vec();
        b.extend(1u32.to_le_bytes());
        b.extend(codec().serialize(s).unwrap());
        b
    }
    #[test]
    fn codec_preserves_bits_six_networks_and_delayed_clocks_and_rejects_other_formats() {
        let mut s = fixture();
        s.networks[3][0].values[0] = -0.;
        let bytes = s.encode().unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(restored.config, s.config);
        assert_eq!((restored.updates, restored.actor_updates), (3, 1));
        assert_eq!(
            restored.networks[3][0].values[0].to_bits(),
            (-0f32).to_bits()
        );
        for end in [0, 7, 11, 12, bytes.len() - 1] {
            assert!(decode(&bytes[..end]).is_err());
        }
        for magic in [b"RFGPUPPO", b"RFGPUDQN", b"RFGPUREI", b"RFGPUPC0"] {
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
        let mut s = fixture();
        s.config.policy_delay = 0;
        s.actor_updates = 0;
        s.actor_adam.timestep = 0;
        s.actor_adam.moments.fill(None);
        assert!(decode(&s.encode().unwrap()).is_ok());
    }
    #[test]
    fn malformed_parameters_targets_clocks_and_moments_fail_on_host() {
        for kind in 0..19 {
            let mut s = fixture();
            match kind {
                0 => {
                    s.networks[0].pop();
                }
                1 => {
                    s.networks[5].pop();
                }
                2 => s.networks[1][0].shape = vec![12],
                3 => {
                    s.networks[4][0].values.pop();
                }
                4 => s.networks[2][0].values[0] = f32::NAN,
                5 => s.networks[3][0].values[0] = f32::INFINITY,
                6 => s.actor_updates = 2,
                7 => s.critic_adam.timestep = 4,
                8 => s.actor_adam.lr = 1.,
                9 => s.critic_adam.moments[0] = None,
                10 => s.actor_adam.moments[0].as_mut().unwrap().second.values[0] = -1.,
                11 => s.critic_adam.beta1 = 1.,
                12 => s.config.hidden_dim = usize::MAX,
                13 => s.config.action_high[0] = s.config.action_low[0],
                14 => s.config.target_noise_clip = f32::NAN,
                15 => {
                    s.updates = u64::MAX;
                    s.critic_adam.timestep = u64::MAX;
                }
                16 => {
                    s.actor_adam.moments.pop();
                }
                17 => s.config.policy_delay = 0,
                _ => s.actor_adam.moments[0].as_mut().unwrap().first.shape = vec![999],
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
            Err(GpuTd3Error::Checkpoint(GpuTd3CheckpointError::TooLarge))
        ));
    }
}
