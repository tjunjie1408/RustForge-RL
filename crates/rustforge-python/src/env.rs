//! Python wrappers for RustForge environments.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustforge_rl::env::{CartPole, CartPoleAction, Environment};

use crate::space::{space_to_py, PySpace};

/// CartPole-v1 environment (discrete actions: 0 = Left, 1 = Right).
#[pyclass(name = "CartPole", module = "rustforge._core")]
pub struct PyCartPole {
    inner: CartPole,
}

#[pymethods]
impl PyCartPole {
    #[new]
    #[pyo3(signature = (max_steps = 500))]
    fn new(max_steps: usize) -> Self {
        PyCartPole {
            inner: CartPole::with_max_steps(max_steps),
        }
    }

    /// Reset the environment and return the initial observation.
    #[pyo3(signature = (seed = None))]
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        let (obs, _info) = self.inner.reset(seed);
        obs.to_vec()
    }

    /// Step the environment. Returns `(observation, reward, terminated, truncated)`.
    fn step(&mut self, action: usize) -> PyResult<(Vec<f32>, f32, bool, bool)> {
        let act =
            CartPoleAction::try_from(action).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (obs, reward, terminated, truncated, _info) = self.inner.step(act);
        Ok((obs.to_vec(), reward, terminated, truncated))
    }

    fn action_space(&self) -> PySpace {
        space_to_py(&self.inner.action_space())
    }

    fn observation_space(&self) -> PySpace {
        space_to_py(&self.inner.observation_space())
    }
}
