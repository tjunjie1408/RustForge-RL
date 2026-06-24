//! PyO3 Python bindings for the RustForge RL framework.
//!
//! The compiled extension is exposed to Python as `rustforge._core`.

use pyo3::prelude::*;

/// The `rustforge._core` extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes are registered here in later tasks.
    let _ = m;
    Ok(())
}
