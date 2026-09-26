//! Model serialization and deserialization.
//!
//! Save and load model parameters using `serde` + `bincode`.
//! Only the tensor data is serialized — the computation graph and
//! optimizer state are not included.
//!
//! ## File format (version 1)
//!
//! ```text
//! magic    8 bytes   b"RFPARAMS"
//! version  u32 LE    1
//! payload  bincode   Vec<Tensor>, in `Module::parameters()` order
//! ```
//!
//! Files written before the header existed (a bare bincode `Vec<Tensor>`)
//! are still accepted as version 0.
//!
//! Parameters are matched by position, so loading requires the same
//! architecture. Loading validates every parameter before assigning any, and
//! saving writes a temporary file that replaces the target only once it is
//! complete.
//!
//! ## Example
//! ```rust,ignore
//! use rustforge_nn::{Linear, Module};
//! use rustforge_nn::serialization::{save_parameters, load_parameters};
//!
//! let model = Linear::new(10, 5);
//! // ... train model ...
//! save_parameters(&model, "model.bin").unwrap();
//!
//! let model2 = Linear::new(10, 5);
//! load_parameters(&model2, "model.bin").unwrap();
//! // model2 now has the same weights as model
//! ```

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rustforge_tensor::Tensor;

use crate::module::Module;

/// Leading bytes identifying a RustForge parameter file.
pub const MAGIC: &[u8; 8] = b"RFPARAMS";
/// Format version written by [`save_parameters`].
pub const FORMAT_VERSION: u32 = 1;

/// Error type for serialization operations.
#[derive(Debug)]
pub enum SerializationError {
    /// I/O error (file not found, permission denied, truncated header, etc.).
    Io(io::Error),
    /// Bincode serialization/deserialization error, including truncated or corrupt payloads.
    Bincode(bincode::Error),
    /// Parameter count mismatch between model and saved file.
    ParameterCountMismatch { expected: usize, got: usize },
    /// Shape mismatch between model parameter and saved tensor.
    ShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// The file declares a format version this build cannot read.
    UnsupportedVersion(u32),
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Bincode(e) => write!(f, "Serialization error: {}", e),
            Self::ParameterCountMismatch { expected, got } => {
                write!(
                    f,
                    "Parameter count mismatch: model has {} parameters, file has {}",
                    expected, got
                )
            }
            Self::ShapeMismatch {
                index,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Shape mismatch at parameter {}: expected {:?}, got {:?}",
                    index, expected, got
                )
            }
            Self::UnsupportedVersion(version) => write!(
                f,
                "Unsupported parameter file version {} (this build reads up to {})",
                version, FORMAT_VERSION
            ),
        }
    }
}

impl std::error::Error for SerializationError {}

impl From<io::Error> for SerializationError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<bincode::Error> for SerializationError {
    fn from(e: bincode::Error) -> Self {
        Self::Bincode(e)
    }
}

/// Saves model parameters to a binary file.
///
/// Writes the versioned format described in the module docs. The data is
/// written to a temporary file in the same directory and then renamed over
/// `path`, so an interrupted save never leaves a partially written model.
///
/// ## Arguments
/// - `module`: The module whose parameters to save.
/// - `path`: File path to write to.
pub fn save_parameters(
    module: &dyn Module,
    path: impl AsRef<Path>,
) -> Result<(), SerializationError> {
    let path = path.as_ref();
    let tensors: Vec<Tensor> = module
        .parameters()
        .iter()
        .map(|p| p.data().clone())
        .collect();
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bincode::serialize_into(&mut bytes, &tensors)?;

    let temporary = temporary_path(path);
    let written = write_synced(&temporary, &bytes).and_then(|()| fs::rename(&temporary, path));
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    Ok(written?)
}

/// Loads model parameters from a binary file.
///
/// Reads tensors from the file and sets them on the module's existing
/// `Variable` parameters. The module must have the same architecture
/// (same number and shapes of parameters) as when it was saved. Every
/// parameter is validated before any is assigned, so on error the module is
/// left unchanged.
///
/// ## Arguments
/// - `module`: The module to load parameters into.
/// - `path`: File path to read from.
///
/// ## Errors
/// - `UnsupportedVersion` if the file was written by a newer format.
/// - `Bincode` if the payload is truncated or corrupt.
/// - `ParameterCountMismatch` if the number of saved tensors doesn't match.
/// - `ShapeMismatch` if any tensor shape differs.
pub fn load_parameters(
    module: &dyn Module,
    path: impl AsRef<Path>,
) -> Result<(), SerializationError> {
    let data = fs::read(path)?;
    let tensors = decode(&data)?;

    let params = module.parameters();
    if tensors.len() != params.len() {
        return Err(SerializationError::ParameterCountMismatch {
            expected: params.len(),
            got: tensors.len(),
        });
    }
    for (index, (param, tensor)) in params.iter().zip(&tensors).enumerate() {
        if param.shape() != tensor.shape() {
            return Err(SerializationError::ShapeMismatch {
                index,
                expected: param.shape(),
                got: tensor.shape().to_vec(),
            });
        }
    }
    for (param, tensor) in params.iter().zip(tensors) {
        param.set_data(tensor);
    }
    Ok(())
}

fn decode(data: &[u8]) -> Result<Vec<Tensor>, SerializationError> {
    let Some(rest) = data.strip_prefix(MAGIC.as_slice()) else {
        // Version 0: files written before the header was introduced.
        return Ok(bincode::deserialize(data)?);
    };
    let (version, payload) = match rest.get(..4) {
        Some(version) => (
            u32::from_le_bytes(version.try_into().expect("slice has four bytes")),
            &rest[4..],
        ),
        None => {
            return Err(SerializationError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "parameter file header is truncated",
            )))
        }
    };
    match version {
        FORMAT_VERSION => Ok(bincode::deserialize(payload)?),
        other => Err(SerializationError::UnsupportedVersion(other)),
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}", std::process::id()));
    path.with_file_name(name)
}

fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::Linear;
    use crate::module::Module;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_save_and_load() {
        let model = Linear::new(3, 2);
        let path = "test_model_save_load.bin";

        // Save
        save_parameters(&model, path).unwrap();

        // Load into a new model with the same architecture
        let model2 = Linear::new(3, 2);
        load_parameters(&model2, path).unwrap();

        // Parameters should match
        let p1 = model.parameters();
        let p2 = model2.parameters();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.data().to_vec(), b.data().to_vec());
        }

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_load_parameter_count_mismatch() {
        let model1 = Linear::new(3, 2);
        let path = "test_model_mismatch.bin";
        save_parameters(&model1, path).unwrap();

        // Different architecture
        let model2 = Linear::no_bias(4, 3);
        let result = load_parameters(&model2, path);
        assert!(result.is_err());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_round_trip_preserves_values() {
        let model = Linear::new(2, 1);
        let path = "test_model_roundtrip.bin";

        // Set known values (using values that don't trigger clippy::approx_constant)
        model.parameters()[0].set_data(rustforge_tensor::Tensor::from_vec(
            vec![3.15, 2.73],
            &[1, 2],
        ));
        model.parameters()[1].set_data(rustforge_tensor::Tensor::from_vec(vec![0.42], &[1]));

        save_parameters(&model, path).unwrap();

        let model2 = Linear::new(2, 1);
        load_parameters(&model2, path).unwrap();

        let w = model2.parameters()[0].data().to_vec();
        let b = model2.parameters()[1].data().to_vec();
        assert_abs_diff_eq!(w[0], 3.15, epsilon = 1e-6);
        assert_abs_diff_eq!(w[1], 2.73, epsilon = 1e-6);
        assert_abs_diff_eq!(b[0], 0.42, epsilon = 1e-6);

        let _ = fs::remove_file(path);
    }

    /// Two free parameters, so tests can control every shape.
    struct TwoParams {
        first: rustforge_autograd::Variable,
        second: rustforge_autograd::Variable,
    }

    impl TwoParams {
        fn new(first: &[f32], second: &[f32]) -> Self {
            let variable = |values: &[f32]| {
                rustforge_autograd::Variable::new(
                    Tensor::from_vec(values.to_vec(), &[values.len()]),
                    true,
                )
            };
            Self {
                first: variable(first),
                second: variable(second),
            }
        }
    }

    impl Module for TwoParams {
        fn forward(&self, input: &rustforge_autograd::Variable) -> rustforge_autograd::Variable {
            input.clone()
        }

        fn parameters(&self) -> Vec<rustforge_autograd::Variable> {
            vec![self.first.clone(), self.second.clone()]
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rustforge-serialization-{}-{name}",
            std::process::id()
        ))
    }

    #[test]
    fn saved_files_start_with_magic_and_version() {
        let path = temp_path("header.bin");
        save_parameters(&TwoParams::new(&[1.0], &[2.0]), &path).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            FORMAT_VERSION
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn failed_load_leaves_every_parameter_untouched() {
        let path = temp_path("partial.bin");
        save_parameters(&TwoParams::new(&[7.0, 7.0], &[7.0, 7.0, 7.0]), &path).unwrap();

        // The first shape matches, the second does not.
        let target = TwoParams::new(&[0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]);
        let error = load_parameters(&target, &path).unwrap_err();
        assert!(matches!(
            error,
            SerializationError::ShapeMismatch { index: 1, .. }
        ));
        assert_eq!(target.first.data().to_vec(), vec![0.0, 0.0]);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn legacy_headerless_files_still_load() {
        let path = temp_path("legacy.bin");
        let tensors = vec![
            Tensor::from_vec(vec![1.5], &[1]),
            Tensor::from_vec(vec![2.5, 3.5], &[2]),
        ];
        fs::write(&path, bincode::serialize(&tensors).unwrap()).unwrap();

        let target = TwoParams::new(&[0.0], &[0.0, 0.0]);
        load_parameters(&target, &path).unwrap();
        assert_eq!(target.second.data().to_vec(), vec![2.5, 3.5]);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn unsupported_versions_and_corrupt_files_are_rejected() {
        let path = temp_path("future.bin");
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        let target = TwoParams::new(&[0.0], &[0.0]);
        assert!(matches!(
            load_parameters(&target, &path),
            Err(SerializationError::UnsupportedVersion(version)) if version == FORMAT_VERSION + 1
        ));

        let mut truncated = MAGIC.to_vec();
        truncated.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        truncated.extend_from_slice(&[1, 2, 3]);
        fs::write(&path, &truncated).unwrap();
        assert!(matches!(
            load_parameters(&target, &path),
            Err(SerializationError::Bincode(_))
        ));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn save_replaces_existing_files_without_leaving_temporaries() {
        let dir = temp_path("atomic-dir");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bin");
        fs::write(&path, b"old contents").unwrap();

        save_parameters(&TwoParams::new(&[4.0], &[5.0]), &path).unwrap();
        let target = TwoParams::new(&[0.0], &[0.0]);
        load_parameters(&target, &path).unwrap();
        assert_eq!(target.first.data().to_vec(), vec![4.0]);

        let entries: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert_eq!(entries.len(), 1, "temporary file left behind");
        let _ = fs::remove_dir_all(&dir);
    }
}
