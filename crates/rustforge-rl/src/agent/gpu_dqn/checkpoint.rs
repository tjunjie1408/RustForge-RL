//! GPU DQN checkpoint v1: magic, little-endian version, fixed-integer bincode
//! payload of explicit row-major tensors, configuration, Adam state and clock.
//! This format is separate from CPU parameter-only RFPARAMS files.
use super::{validate_config, DQNConfig, GpuDqn, Result};
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

pub const CHECKPOINT_MAGIC: &[u8; 8] = b"RFGPUDQN";
pub const CHECKPOINT_VERSION: u32 = 1;
/// Maximum complete file size, including its 12-byte header.
pub const MAX_CHECKPOINT_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
pub enum GpuDqnCheckpointError {
    Io(io::Error),
    Encoding(bincode::Error),
    UnsupportedVersion(u32),
    InvalidState(&'static str),
    TooLarge,
}
impl std::fmt::Display for GpuDqnCheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GPU checkpoint I/O error: {e}"),
            Self::Encoding(e) => write!(f, "GPU checkpoint encoding error: {e}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GPU DQN checkpoint version {v}"),
            Self::InvalidState(message) => f.write_str(message),
            Self::TooLarge => write!(f, "GPU checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes"),
        }
    }
}
impl std::error::Error for GpuDqnCheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Encoding(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for GpuDqnCheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<bincode::Error> for GpuDqnCheckpointError {
    fn from(e: bincode::Error) -> Self {
        Self::Encoding(e)
    }
}
fn invalid(message: &'static str) -> super::GpuDqnError {
    GpuDqnCheckpointError::InvalidState(message).into()
}
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
        .with_limit(MAX_CHECKPOINT_BYTES - 12)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedConfig {
    obs_dim: usize,
    num_actions: usize,
    hidden_dim: usize,
    lr: f32,
    gamma: f32,
    target_update_freq: usize,
    double_dqn: bool,
    use_per: bool,
    per_beta_annealing_steps: usize,
}
impl SavedConfig {
    fn capture(c: &DQNConfig) -> Self {
        Self {
            obs_dim: c.obs_dim,
            num_actions: c.num_actions,
            hidden_dim: c.hidden_dim,
            lr: c.lr,
            gamma: c.gamma,
            target_update_freq: c.target_update_freq,
            double_dqn: c.double_dqn,
            use_per: c.use_per,
            per_beta_annealing_steps: c.per_beta_annealing_steps,
        }
    }
    fn config(&self) -> DQNConfig {
        DQNConfig {
            obs_dim: self.obs_dim,
            num_actions: self.num_actions,
            hidden_dim: self.hidden_dim,
            lr: self.lr,
            gamma: self.gamma,
            target_update_freq: self.target_update_freq,
            double_dqn: self.double_dqn,
            use_per: self.use_per,
            per_beta_annealing_steps: self.per_beta_annealing_steps,
        }
    }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedDqn {
    config: SavedConfig,
    online: Vec<SavedTensor>,
    target: Vec<SavedTensor>,
    adam: SavedAdam,
    train_steps: u64,
}
impl SavedDqn {
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            vec![self.config.hidden_dim, self.config.obs_dim],
            vec![self.config.hidden_dim],
            vec![self.config.num_actions, self.config.hidden_dim],
            vec![self.config.num_actions],
        ]
    }
    fn validate(&self) -> Result<()> {
        validate_config(&self.config.config())?;
        let shapes = self.shapes();
        if self.online.len() != shapes.len()
            || self.target.len() != shapes.len()
            || self.adam.moments.len() != shapes.len()
        {
            return Err(invalid(
                "checkpoint parameter/moment counts do not match DQN architecture",
            ));
        }
        let step = usize::try_from(self.train_steps)
            .map_err(|_| invalid("DQN step count exceeds this platform"))?;
        if step == usize::MAX
            || self.train_steps != self.adam.timestep
            || self.config.lr != self.adam.lr
        {
            return Err(invalid(
                "checkpoint DQN/Adam clocks or learning rates disagree",
            ));
        }
        for ((online, target), shape) in self.online.iter().zip(&self.target).zip(&shapes) {
            online.validate(shape, false)?;
            target.validate(shape, false)?;
        }
        for (moment, shape) in self.adam.moments.iter().zip(&shapes) {
            if moment.is_some() != (step > 0) {
                return Err(invalid(
                    "checkpoint Adam moments do not match DQN training progress",
                ));
            }
            if let Some(moment) = moment {
                moment.first.validate(shape, false)?;
                moment.second.validate(shape, true)?;
            }
        }
        self.adam.state()?.validate_for_shapes(&shapes)?;
        Ok(())
    }
    fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let size = codec()
            .serialized_size(self)
            .map_err(GpuDqnCheckpointError::from)?;
        if size > MAX_CHECKPOINT_BYTES - 12 {
            return Err(GpuDqnCheckpointError::TooLarge.into());
        }
        let mut data = CHECKPOINT_MAGIC.to_vec();
        data.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
        data.extend(
            codec()
                .serialize(self)
                .map_err(GpuDqnCheckpointError::from)?,
        );
        Ok(data)
    }
}
fn decode(data: &[u8]) -> Result<SavedDqn> {
    if data.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(GpuDqnCheckpointError::TooLarge.into());
    }
    if data.len() < 12 || &data[..8] != CHECKPOINT_MAGIC {
        return Err(invalid("invalid or truncated GPU DQN checkpoint header"));
    }
    let version = u32::from_le_bytes(data[8..12].try_into().expect("validated header"));
    if version != CHECKPOINT_VERSION {
        return Err(GpuDqnCheckpointError::UnsupportedVersion(version).into());
    }
    let checkpoint: SavedDqn = codec()
        .deserialize(&data[12..])
        .map_err(GpuDqnCheckpointError::from)?;
    checkpoint.validate()?;
    Ok(checkpoint)
}
fn read_checkpoint(path: &Path) -> Result<SavedDqn> {
    let file = File::open(path).map_err(GpuDqnCheckpointError::from)?;
    if file.metadata().map_err(GpuDqnCheckpointError::from)?.len() > MAX_CHECKPOINT_BYTES {
        return Err(GpuDqnCheckpointError::TooLarge.into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_CHECKPOINT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(GpuDqnCheckpointError::from)?;
    decode(&bytes)
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
impl GpuDqn {
    /// Saves configuration, online/target values, Adam moments/clock and step
    /// count. Replay/environment/exploration state and live gradients are omitted.
    /// Validation/serialization precede creation of a temporary file. A synced
    /// file is atomically renamed in the same directory; failed saves preserve
    /// the old checkpoint. Maximum file size is MAX_CHECKPOINT_BYTES.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let capture = |params: Vec<GpuVariable>| -> Result<Vec<SavedTensor>> {
            params
                .iter()
                .map(|p| Ok(SavedTensor::capture(&p.to_cpu()?)))
                .collect()
        };
        let checkpoint = SavedDqn {
            config: SavedConfig::capture(&self.config),
            online: capture(self.q_net.parameters())?,
            target: capture(self.target_net.parameters())?,
            adam: SavedAdam::capture(&self.optimizer.state()?),
            train_steps: self.train_steps as u64,
        };
        let bytes = checkpoint.encode()?;
        write_atomic(path.as_ref(), &bytes).map_err(GpuDqnCheckpointError::from)?;
        Ok(())
    }
    /// Validates the entire host checkpoint before device allocation, then
    /// builds a new agent on the supplied context. Target values are restored
    /// independently of online values; loading does not synchronize them.
    pub fn load_checkpoint(context: &GpuContext, path: impl AsRef<Path>) -> Result<Self> {
        let checkpoint = read_checkpoint(path.as_ref())?;
        let mut agent = Self::new_seeded(context, checkpoint.config.config(), 0)?;
        for (params, values) in [
            (agent.q_net.parameters(), &checkpoint.online),
            (agent.target_net.parameters(), &checkpoint.target),
        ] {
            for (parameter, value) in params.iter().zip(values) {
                let data = GpuVariable::new(context, &value.tensor(), false)?;
                parameter.copy_data_from(&data)?;
            }
        }
        agent.optimizer.restore_state(&checkpoint.adam.state()?)?;
        agent.train_steps = usize::try_from(checkpoint.train_steps)
            .map_err(|_| invalid("DQN step count exceeds this platform"))?;
        Ok(agent)
    }
    /// Builds a complete replacement before assigning it. On error the live
    /// agent remains unchanged. On success, previously acquired parameter
    /// handles still refer to the old model; reacquire them from this agent.
    pub fn restore_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let candidate = Self::load_checkpoint(&self.context, path)?;
        *self = candidate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> SavedDqn {
        let config = SavedConfig::capture(&DQNConfig {
            obs_dim: 2,
            num_actions: 2,
            hidden_dim: 3,
            ..DQNConfig::default()
        });
        let shapes = [vec![3, 2], vec![3], vec![2, 3], vec![2]];
        let online: Vec<_> = shapes
            .iter()
            .map(|shape| SavedTensor {
                shape: shape.clone(),
                values: vec![0.1; shape.iter().product()],
            })
            .collect();
        let target = online.clone();
        let adam = SavedAdam {
            lr: config.lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 1,
            moments: online
                .iter()
                .map(|t| {
                    Some(SavedMoments {
                        first: t.clone(),
                        second: t.clone(),
                    })
                })
                .collect(),
        };
        SavedDqn {
            config,
            online,
            target,
            adam,
            train_steps: 1,
        }
    }
    fn unchecked_bytes(c: &SavedDqn) -> Vec<u8> {
        let mut data = CHECKPOINT_MAGIC.to_vec();
        data.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
        data.extend(codec().serialize(c).unwrap());
        data
    }
    #[test]
    fn binary_codec_preserves_float_bits_and_rejects_headers_truncation_and_trailing_data() {
        let mut checkpoint = fixture();
        checkpoint.online[0].values = vec![
            -0.,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            1.2345679,
            f32::MAX,
            -1.,
        ];
        let bytes = checkpoint.encode().unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(
            restored.online[0]
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            checkpoint.online[0]
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        for length in [0, 7, 8, 11, 12, bytes.len() - 1] {
            assert!(decode(&bytes[..length]).is_err());
        }
        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        assert!(decode(&wrong).is_err());
        let mut version = bytes.clone();
        version[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            decode(&version),
            Err(super::super::GpuDqnError::Checkpoint(
                GpuDqnCheckpointError::UnsupportedVersion(2)
            ))
        ));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        assert!(decode(b"RFPARAMS\x01\x00\x00\x00").is_err());
    }
    #[test]
    fn malformed_checkpoint_metadata_is_rejected_before_tensor_construction() {
        for kind in 0..13 {
            let mut bad = fixture();
            match kind {
                0 => bad.online.pop().map(|_| ()).unwrap(),
                1 => bad.target[3].shape = vec![1, 2],
                2 => bad.online[0].values.pop().map(|_| ()).unwrap(),
                3 => bad.config.use_per = true,
                4 => bad.config.gamma = f32::NAN,
                5 => bad.adam.timestep = 2,
                6 => bad.adam.lr = 0.5,
                7 => bad.adam.moments[0] = None,
                8 => bad.adam.moments[1].as_mut().unwrap().second.values[0] = -1.,
                9 => bad.target[0].values[0] = f32::INFINITY,
                10 => bad.adam.beta1 = 1.,
                11 => {
                    bad.config.obs_dim = usize::MAX;
                    bad.config.hidden_dim = 2;
                    bad.online[0].shape = vec![2, usize::MAX];
                }
                _ => {
                    bad.train_steps = u64::MAX;
                    bad.adam.timestep = u64::MAX;
                }
            }
            assert!(
                decode(&unchecked_bytes(&bad)).is_err(),
                "bad state case {kind}"
            );
        }
    }
    #[test]
    fn atomic_write_replaces_complete_files_and_cleans_failed_renames() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("agent.chk");
        fs::write(&path, b"old checkpoint").unwrap();
        let bytes = fixture().encode().unwrap();
        write_atomic(&path, &bytes).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(read_checkpoint(&path).is_ok());
        let destination = directory.path().join("existing-directory");
        fs::create_dir(&destination).unwrap();
        assert!(write_atomic(&destination, b"incomplete").is_err());
        assert!(destination.is_dir());
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));
    }
    #[test]
    fn bounded_read_rejects_oversized_files_without_loading_them() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("large.chk");
        File::create(&path)
            .unwrap()
            .set_len(MAX_CHECKPOINT_BYTES + 1)
            .unwrap();
        assert!(matches!(
            read_checkpoint(&path),
            Err(super::super::GpuDqnError::Checkpoint(
                GpuDqnCheckpointError::TooLarge
            ))
        ));
    }
}
