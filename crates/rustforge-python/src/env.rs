//! Python wrappers for RustForge environments.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustforge_rl::env::{
    CartPole, CartPoleAction, DiscreteMountainCarAction, Environment, GridAction, GridWorld,
    MountainCar, MountainCarAction, MountainCarContinuous, Pendulum, PendulumAction,
};

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

/// GridWorld 2D maze (discrete actions: 0=Up, 1=Down, 2=Left, 3=Right).
#[pyclass(name = "GridWorld", module = "rustforge._core")]
pub struct PyGridWorld {
    inner: GridWorld,
}

#[pymethods]
impl PyGridWorld {
    #[new]
    fn new() -> Self {
        PyGridWorld {
            inner: GridWorld::new(),
        }
    }

    #[pyo3(signature = (seed = None))]
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        let (obs, _info) = self.inner.reset(seed);
        vec![obs[0] as f32, obs[1] as f32]
    }

    fn step(&mut self, action: usize) -> PyResult<(Vec<f32>, f32, bool, bool)> {
        let act = GridAction::try_from(action).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (obs, reward, terminated, truncated, _info) = self.inner.step(act);
        Ok((
            vec![obs[0] as f32, obs[1] as f32],
            reward,
            terminated,
            truncated,
        ))
    }

    fn action_space(&self) -> PySpace {
        space_to_py(&self.inner.action_space())
    }

    fn observation_space(&self) -> PySpace {
        space_to_py(&self.inner.observation_space())
    }
}

/// Discrete MountainCar (actions: 0=Left, 1=Idle, 2=Right).
#[pyclass(name = "MountainCar", module = "rustforge._core")]
pub struct PyMountainCar {
    inner: MountainCar,
}

#[pymethods]
impl PyMountainCar {
    #[new]
    #[pyo3(signature = (max_steps = 200))]
    fn new(max_steps: usize) -> Self {
        PyMountainCar {
            inner: MountainCar::with_max_steps(max_steps),
        }
    }

    #[pyo3(signature = (seed = None))]
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        let (obs, _info) = self.inner.reset(seed);
        obs.to_vec()
    }

    fn step(&mut self, action: usize) -> PyResult<(Vec<f32>, f32, bool, bool)> {
        let act = DiscreteMountainCarAction::try_from(action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
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

/// Pendulum-v1 (continuous torque action, length 1, clamped to [-2, 2]).
#[pyclass(name = "Pendulum", module = "rustforge._core")]
pub struct PyPendulum {
    inner: Pendulum,
}

#[pymethods]
impl PyPendulum {
    #[new]
    #[pyo3(signature = (max_steps = 200))]
    fn new(max_steps: usize) -> Self {
        PyPendulum {
            inner: Pendulum::with_max_steps(max_steps),
        }
    }

    #[pyo3(signature = (seed = None))]
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        let (obs, _info) = self.inner.reset(seed);
        obs.to_vec()
    }

    fn step(&mut self, action: Vec<f32>) -> PyResult<(Vec<f32>, f32, bool, bool)> {
        if action.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "Pendulum expects an action of length 1, got {}",
                action.len()
            )));
        }
        let (obs, reward, terminated, truncated, _info) =
            self.inner.step(PendulumAction::new(action[0]));
        Ok((obs.to_vec(), reward, terminated, truncated))
    }

    fn action_space(&self) -> PySpace {
        space_to_py(&self.inner.action_space())
    }

    fn observation_space(&self) -> PySpace {
        space_to_py(&self.inner.observation_space())
    }
}

/// MountainCarContinuous-v0 (continuous force action, length 1, clamped to [-1, 1]).
#[pyclass(name = "MountainCarContinuous", module = "rustforge._core")]
pub struct PyMountainCarContinuous {
    inner: MountainCarContinuous,
}

#[pymethods]
impl PyMountainCarContinuous {
    #[new]
    #[pyo3(signature = (max_steps = 999))]
    fn new(max_steps: usize) -> Self {
        PyMountainCarContinuous {
            inner: MountainCarContinuous::with_max_steps(max_steps),
        }
    }

    #[pyo3(signature = (seed = None))]
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        let (obs, _info) = self.inner.reset(seed);
        obs.to_vec()
    }

    fn step(&mut self, action: Vec<f32>) -> PyResult<(Vec<f32>, f32, bool, bool)> {
        if action.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "MountainCarContinuous expects an action of length 1, got {}",
                action.len()
            )));
        }
        let act: MountainCarAction = [action[0]];
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
