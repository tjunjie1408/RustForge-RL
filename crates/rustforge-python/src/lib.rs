//! PyO3 Python bindings for the RustForge RL framework.
//!
//! The compiled extension is exposed to Python as `rustforge._core`.

use pyo3::prelude::*;

mod env;
mod space;

use env::{PyCartPole, PyGridWorld, PyMountainCar};
use space::PySpace;

/// The `rustforge._core` extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpace>()?;
    m.add_class::<PyCartPole>()?;
    m.add_class::<PyGridWorld>()?;
    m.add_class::<PyMountainCar>()?;
    Ok(())
}
