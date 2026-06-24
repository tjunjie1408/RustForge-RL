//! Python wrapper for the RustForge `Space` descriptor.

use pyo3::prelude::*;
use rustforge_rl::env::Space;

/// Action/observation space descriptor exposed to Python.
///
/// `kind` is one of `"discrete"`, `"box"`, or `"multidiscrete"`. Only the fields
/// relevant to that kind are populated; the rest are `None`.
#[pyclass(name = "Space", module = "rustforge._core")]
#[derive(Clone)]
pub struct PySpace {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub n: Option<usize>,
    #[pyo3(get)]
    pub low: Option<Vec<f32>>,
    #[pyo3(get)]
    pub high: Option<Vec<f32>>,
    #[pyo3(get)]
    pub shape: Option<Vec<usize>>,
    #[pyo3(get)]
    pub nvec: Option<Vec<usize>>,
}

#[pymethods]
impl PySpace {
    fn __repr__(&self) -> String {
        format!(
            "Space(kind={:?}, n={:?}, shape={:?})",
            self.kind, self.n, self.shape
        )
    }
}

/// Convert a native `Space` into its Python-facing representation.
pub fn space_to_py(space: &Space) -> PySpace {
    match space {
        Space::Discrete(n) => PySpace {
            kind: "discrete".to_string(),
            n: Some(*n),
            low: None,
            high: None,
            shape: None,
            nvec: None,
        },
        Space::Box { low, high, shape } => PySpace {
            kind: "box".to_string(),
            n: None,
            low: Some(low.clone()),
            high: Some(high.clone()),
            shape: Some(shape.clone()),
            nvec: None,
        },
        Space::MultiDiscrete(nvec) => PySpace {
            kind: "multidiscrete".to_string(),
            n: None,
            low: None,
            high: None,
            shape: None,
            nvec: Some(nvec.clone()),
        },
    }
}
