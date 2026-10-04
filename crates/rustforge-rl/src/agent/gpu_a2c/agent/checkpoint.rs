//! GPU A2C checkpoint v1: bounded little-endian tensor/Adam state.
//! Rollout, environment, RNG and gradients are intentionally not persisted.
use super::{A2CConfig, GpuA2c, GpuA2cError, Result};
use bincode::Options;
use rustforge_autograd::gpu::{GpuAdamMomentState, GpuAdamState, GpuVariable};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub const CHECKPOINT_MAGIC: &[u8; 8] = b"RFGPUA2C";
pub const CHECKPOINT_VERSION: u32 = 1;
/// Maximum complete file size, including its 12-byte header.
pub const MAX_CHECKPOINT_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
pub enum GpuA2cCheckpointError {
    Io(io::Error),
    Encoding(bincode::Error),
    UnsupportedVersion(u32),
    InvalidState(&'static str),
    TooLarge,
}
impl std::fmt::Display for GpuA2cCheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GPU checkpoint I/O error: {e}"),
            Self::Encoding(e) => write!(f, "GPU checkpoint encoding error: {e}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GPU A2C checkpoint version {v}"),
            Self::InvalidState(message) => f.write_str(message),
            Self::TooLarge => write!(f, "GPU checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes"),
        }
    }
}
impl std::error::Error for GpuA2cCheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Encoding(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for GpuA2cCheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<bincode::Error> for GpuA2cCheckpointError {
    fn from(e: bincode::Error) -> Self {
        Self::Encoding(e)
    }
}
fn invalid(message: &'static str) -> GpuA2cError {
    GpuA2cCheckpointError::InvalidState(message).into()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedA2c {
    config: A2CConfig,
    parameters: Vec<SavedTensor>,
    adam: SavedAdam,
    updates: u64,
}
impl SavedA2c {
    fn shapes(&self) -> Vec<Vec<usize>> {
        let c = &self.config;
        vec![
            vec![c.hidden_dim, c.obs_dim],
            vec![c.hidden_dim],
            vec![c.num_actions, c.hidden_dim],
            vec![c.num_actions],
            vec![1, c.hidden_dim],
            vec![1],
        ]
    }
    fn validate(&self) -> Result<()> {
        GpuA2c::validate_config(&self.config)?;
        let shapes = self.shapes();
        if self.parameters.len() != shapes.len() || self.adam.moments.len() != shapes.len() {
            return Err(invalid(
                "checkpoint parameter/moment count does not match A2C architecture",
            ));
        }
        let step = usize::try_from(self.updates)
            .map_err(|_| invalid("A2C update count exceeds this platform"))?;
        if step == usize::MAX
            || self.updates != self.adam.timestep
            || self.config.lr != self.adam.lr
        {
            return Err(invalid(
                "checkpoint A2C/Adam clocks or learning rates disagree",
            ));
        }
        for (parameter, shape) in self.parameters.iter().zip(&shapes) {
            parameter.validate(shape, false)?;
        }
        for (moment, shape) in self.adam.moments.iter().zip(&shapes) {
            if moment.is_some() != (step > 0) {
                return Err(invalid(
                    "checkpoint Adam moments do not match A2C training progress",
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
        if codec()
            .serialized_size(self)
            .map_err(GpuA2cCheckpointError::from)?
            > MAX_CHECKPOINT_BYTES - 12
        {
            return Err(GpuA2cCheckpointError::TooLarge.into());
        }
        let mut data = CHECKPOINT_MAGIC.to_vec();
        data.extend(CHECKPOINT_VERSION.to_le_bytes());
        data.extend(
            codec()
                .serialize(self)
                .map_err(GpuA2cCheckpointError::from)?,
        );
        Ok(data)
    }
}
fn decode(data: &[u8]) -> Result<SavedA2c> {
    if data.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(GpuA2cCheckpointError::TooLarge.into());
    }
    if data.len() < 12 || &data[..8] != CHECKPOINT_MAGIC {
        return Err(invalid("invalid or truncated GPU A2C checkpoint header"));
    }
    let version = u32::from_le_bytes(data[8..12].try_into().expect("validated header"));
    if version != CHECKPOINT_VERSION {
        return Err(GpuA2cCheckpointError::UnsupportedVersion(version).into());
    }
    let checkpoint: SavedA2c = codec()
        .deserialize(&data[12..])
        .map_err(GpuA2cCheckpointError::from)?;
    checkpoint.validate()?;
    Ok(checkpoint)
}
fn read_checkpoint(path: &Path) -> Result<SavedA2c> {
    let file = File::open(path).map_err(GpuA2cCheckpointError::from)?;
    if file.metadata().map_err(GpuA2cCheckpointError::from)?.len() > MAX_CHECKPOINT_BYTES {
        return Err(GpuA2cCheckpointError::TooLarge.into());
    }
    let mut data = Vec::new();
    file.take(MAX_CHECKPOINT_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(GpuA2cCheckpointError::from)?;
    decode(&data)
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

impl GpuA2c {
    /// Atomically replaces the destination after validating the complete state.
    /// Does not persist gradients, rollout, environment or random streams.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let parameters = self
            .net
            .parameters()
            .iter()
            .map(|p| Ok(SavedTensor::capture(&p.to_cpu()?)))
            .collect::<Result<Vec<_>>>()?;
        let saved = SavedA2c {
            config: self.config.clone(),
            parameters,
            adam: SavedAdam::capture(&self.optimizer.state()?),
            updates: self.updates,
        };
        let bytes = saved.encode()?;
        write_atomic(path.as_ref(), &bytes).map_err(GpuA2cCheckpointError::from)?;
        Ok(())
    }
    /// Validates all host metadata and tensors before allocating a new agent.
    pub fn load_checkpoint(context: &GpuContext, path: impl AsRef<Path>) -> Result<Self> {
        let saved = read_checkpoint(path.as_ref())?;
        let mut agent = Self::new_seeded(context, saved.config, 0)?;
        for (parameter, value) in agent.net.parameters().iter().zip(&saved.parameters) {
            parameter.copy_data_from(&GpuVariable::new(context, &value.tensor(), false)?)?;
        }
        agent.optimizer.restore_state(&saved.adam.state()?)?;
        agent.updates = saved.updates;
        Ok(agent)
    }
    /// Replaces the live agent only after successful load; old handles remain old.
    pub fn restore_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let candidate = Self::load_checkpoint(&self.context, path)?;
        *self = candidate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SavedA2c {
        let config = A2CConfig {
            obs_dim: 2,
            hidden_dim: 3,
            num_actions: 2,
            ..Default::default()
        };
        let shapes = [
            vec![3, 2],
            vec![3],
            vec![2, 3],
            vec![2],
            vec![1, 3],
            vec![1],
        ];
        let parameters: Vec<_> = shapes
            .iter()
            .map(|shape| SavedTensor {
                shape: shape.clone(),
                values: vec![0.1; shape.iter().product()],
            })
            .collect();
        let adam = SavedAdam {
            lr: config.lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 1,
            moments: parameters
                .iter()
                .map(|t| {
                    Some(SavedMoments {
                        first: t.clone(),
                        second: t.clone(),
                    })
                })
                .collect(),
        };
        SavedA2c {
            config,
            parameters,
            adam,
            updates: 1,
        }
    }
    fn unchecked(saved: &SavedA2c) -> Vec<u8> {
        let mut data = CHECKPOINT_MAGIC.to_vec();
        data.extend(CHECKPOINT_VERSION.to_le_bytes());
        data.extend(codec().serialize(saved).unwrap());
        data
    }
    #[test]
    fn codec_preserves_bits_and_rejects_wrong_algorithm_version_truncation_and_trailing_data() {
        let mut saved = fixture();
        saved.parameters[0].values = vec![
            -0.,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            1.2345679,
            f32::MAX,
            -1.,
        ];
        let bytes = saved.encode().unwrap();
        let restored = decode(&bytes).unwrap();
        assert_eq!(saved.config, restored.config);
        assert_eq!(
            saved.parameters[0]
                .values
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            restored.parameters[0]
                .values
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        for end in [0, 7, 8, 11, 12, bytes.len() - 1] {
            assert!(decode(&bytes[..end]).is_err());
        }
        let mut wrong = bytes.clone();
        wrong[..8].copy_from_slice(b"RFGPUPPO");
        assert!(decode(&wrong).is_err());
        wrong = bytes.clone();
        wrong[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            decode(&wrong),
            Err(GpuA2cError::Checkpoint(
                GpuA2cCheckpointError::UnsupportedVersion(2)
            ))
        ));
        wrong = bytes;
        wrong.push(0);
        assert!(decode(&wrong).is_err());
    }
    #[test]
    fn malformed_state_is_rejected_before_tensor_or_device_construction() {
        for kind in 0..16 {
            let mut bad = fixture();
            match kind {
                0 => {
                    bad.parameters.pop();
                }
                1 => bad.parameters[0].shape = vec![1, 6],
                2 => {
                    bad.parameters[0].values.pop();
                }
                3 => bad.config.c_value = -1.,
                4 => bad.config.gamma = f32::NAN,
                5 => bad.adam.timestep = 2,
                6 => bad.adam.lr = 1.,
                7 => bad.adam.moments[0] = None,
                8 => bad.adam.moments[1].as_mut().unwrap().second.values[0] = -1.,
                9 => bad.parameters[0].values[0] = f32::INFINITY,
                10 => bad.adam.beta1 = 1.,
                11 => {
                    bad.config.obs_dim = usize::MAX;
                    bad.parameters[0].shape = vec![3, usize::MAX];
                }
                12 => {
                    bad.updates = u64::MAX;
                    bad.adam.timestep = u64::MAX;
                }
                13 => bad.config.num_actions = 0,
                14 => bad.config.lambda = -1.,
                _ => bad.adam.moments[0].as_mut().unwrap().first.shape = vec![6],
            }
            assert!(decode(&unchecked(&bad)).is_err(), "case {kind}");
        }
        let mut untrained = fixture();
        untrained.updates = 0;
        untrained.adam.timestep = 0;
        untrained.adam.moments.fill(None);
        assert!(decode(&untrained.encode().unwrap()).is_ok());
        untrained.adam.moments[0] = fixture().adam.moments[0].clone();
        assert!(decode(&unchecked(&untrained)).is_err());
    }
    #[test]
    fn atomic_replace_failure_cleanup_and_bounded_file_read() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("a2c.chk");
        fs::write(&path, b"old").unwrap();
        let bytes = fixture().encode().unwrap();
        write_atomic(&path, &bytes).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(read_checkpoint(&path).is_ok());
        let target = directory.path().join("directory");
        fs::create_dir(&target).unwrap();
        assert!(write_atomic(&target, b"partial").is_err());
        assert!(fs::read_dir(directory.path()).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));
        File::create(&path)
            .unwrap()
            .set_len(MAX_CHECKPOINT_BYTES + 1)
            .unwrap();
        assert!(matches!(
            read_checkpoint(&path),
            Err(GpuA2cError::Checkpoint(GpuA2cCheckpointError::TooLarge))
        ));
    }
}
