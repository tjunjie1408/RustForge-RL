//! Model serialization and deserialization.
//!
//! Save and load model parameters using `serde` + `bincode`.
//! Only the tensor data is serialized — the computation graph and
//! optimizer state are not included.
//!
//! ## Design
//!
//! Parameters are extracted from a `Module` as `Vec<Tensor>`, serialized
//! via bincode, and written to a binary file. To load, the file is read
//! back into `Vec<Tensor>` and each tensor is set on the corresponding
//! `Variable` in the module.
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
use std::io;

use rustforge_tensor::Tensor;

use crate::module::Module;

/// Error type for serialization operations.
#[derive(Debug)]
pub enum SerializationError {
    /// I/O error (file not found, permission denied, etc.).
    Io(io::Error),
    /// Bincode serialization/deserialization error.
    Bincode(bincode::Error),
    /// Parameter count mismatch between model and saved file.
    ParameterCountMismatch { expected: usize, got: usize },
    /// Shape mismatch between model parameter and saved tensor.
    ShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
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
/// Extracts all parameter tensors from the module and serializes them
/// using bincode.
///
/// ## Arguments
/// - `module`: The module whose parameters to save.
/// - `path`: File path to write to.
pub fn save_parameters(module: &dyn Module, path: &str) -> Result<(), SerializationError> {
    let params = module.parameters();
    let tensors: Vec<Tensor> = params.iter().map(|p| p.data()).collect();
    let encoded = bincode::serialize(&tensors)?;
    fs::write(path, encoded)?;
    Ok(())
}

/// Loads model parameters from a binary file.
///
/// Reads tensors from the file and sets them on the module's existing
/// `Variable` parameters. The module must have the same architecture
/// (same number and shapes of parameters) as when it was saved.
///
/// ## Arguments
/// - `module`: The module to load parameters into.
/// - `path`: File path to read from.
///
/// ## Errors
/// - `ParameterCountMismatch` if the number of saved tensors doesn't match.
/// - `ShapeMismatch` if any tensor shape differs.
pub fn load_parameters(module: &dyn Module, path: &str) -> Result<(), SerializationError> {
    let data = fs::read(path)?;
    let tensors: Vec<Tensor> = bincode::deserialize(&data)?;

    let params = module.parameters();
    if tensors.len() != params.len() {
        return Err(SerializationError::ParameterCountMismatch {
            expected: params.len(),
            got: tensors.len(),
        });
    }

    for (i, (param, tensor)) in params.iter().zip(tensors.iter()).enumerate() {
        if param.shape() != tensor.shape() {
            return Err(SerializationError::ShapeMismatch {
                index: i,
                expected: param.shape(),
                got: tensor.shape().to_vec(),
            });
        }
        param.set_data(tensor.clone());
    }

    Ok(())
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
}
