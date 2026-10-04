//! Python wrapper for the DQN agent.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use rustforge_rl::agent::{try_train_dqn, DQNConfig, DQN};
use rustforge_rl::env::{CartPole, GridWorld, MountainCar};

/// Trained Deep Q-Network agent.
///
/// Construct via `DQN.train(...)`, then call `predict(obs)` for greedy actions.
///
/// Declared `unsendable`: the underlying autograd graph uses `Rc`, so `DQN` is not `Send`.
#[pyclass(name = "DQN", module = "rustforge._core", unsendable)]
pub struct PyDQN {
    inner: DQN,
    obs_dim: usize,
}

#[pymethods]
impl PyDQN {
    /// Train a DQN on a built-in discrete environment and return the trained agent.
    ///
    /// `env_name` must be one of `"cartpole"`, `"gridworld"`, `"mountaincar"`.
    /// Raises `RuntimeError` if the log file cannot be created or training diverges.
    #[staticmethod]
    #[pyo3(signature = (
        env_name,
        episodes = 100,
        max_steps = 500,
        hidden_dim = 64,
        lr = 1e-3,
        gamma = 0.99,
        double_dqn = true,
        log_path = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn train(
        env_name: &str,
        episodes: usize,
        max_steps: usize,
        hidden_dim: usize,
        lr: f32,
        gamma: f32,
        double_dqn: bool,
        log_path: Option<String>,
    ) -> PyResult<PyDQN> {
        if episodes == 0 || max_steps == 0 || hidden_dim == 0 {
            return Err(PyValueError::new_err(
                "episodes, max_steps and hidden_dim must be positive",
            ));
        }
        if !lr.is_finite() || lr <= 0. || !gamma.is_finite() || !(0. ..=1.).contains(&gamma) {
            return Err(PyValueError::new_err(
                "lr must be finite and positive; gamma must be finite in [0,1]",
            ));
        }
        let normalized = env_name.trim().to_ascii_lowercase().replace(['-', '_'], "");
        let env_name = match normalized.as_str() {
            "cartpolev1" => "cartpole",
            "mountaincarv0" => "mountaincar",
            name => name,
        };
        let (obs_dim, num_actions) = match env_name {
            "cartpole" => (4usize, 2usize),
            "gridworld" => (2usize, 4usize),
            "mountaincar" => (2usize, 3usize),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown env_name {other:?}; expected 'cartpole', 'gridworld', or 'mountaincar'"
                )))
            }
        };

        let config = DQNConfig {
            obs_dim,
            num_actions,
            hidden_dim,
            lr,
            gamma,
            target_update_freq: 100,
            double_dqn,
            use_per: false,
            per_beta_annealing_steps: 20_000,
        };

        let path = log_path.as_deref();
        let inner = match env_name {
            "cartpole" => try_train_dqn(
                CartPole::with_max_steps(max_steps),
                config,
                episodes,
                max_steps,
                path,
            ),
            "gridworld" => try_train_dqn(GridWorld::new(), config, episodes, max_steps, path),
            "mountaincar" => try_train_dqn(
                MountainCar::with_max_steps(max_steps),
                config,
                episodes,
                max_steps,
                path,
            ),
            _ => unreachable!("env_name validated above"),
        }
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

        Ok(PyDQN { inner, obs_dim })
    }

    /// Greedy action for an observation (argmax over Q-values).
    ///
    /// Raises `RuntimeError` if the network's Q-values are NaN.
    fn predict(&self, obs: Vec<f32>) -> PyResult<usize> {
        if obs.len() != self.obs_dim {
            return Err(PyValueError::new_err(format!(
                "expected observation of length {}, got {}",
                self.obs_dim,
                obs.len()
            )));
        }
        if obs.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("observation values must be finite"));
        }
        self.inner
            .try_select_greedy_action(&obs)
            .map_err(|error| PyRuntimeError::new_err(format!("cannot select an action: {error}")))
    }

    /// Number of completed training steps.
    fn train_steps(&self) -> usize {
        self.inner.train_steps()
    }
}
