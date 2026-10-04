//! Owned GPU A2C: CPU sampling/rollout/GAE, resident shared network and Adam.
mod checkpoint;
use super::{a2c_loss, GpuA2cLossConfig, GpuA2cLossError, GpuA2cMetrics, GpuA2cNet};
use crate::{
    agent::{gpu_ppo::GpuPpoError, A2CConfig},
    buffer::{RolloutBatch, RolloutBuffer},
    env::{Environment, IntoTensorBuffer},
};
pub use checkpoint::{
    GpuA2cCheckpointError, CHECKPOINT_MAGIC, CHECKPOINT_VERSION, MAX_CHECKPOINT_BYTES,
};
use rand::Rng;
use rustforge_autograd::{
    gpu::{GpuAdam, GpuAutogradError, GpuVariable},
    no_grad,
};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};
use std::{error::Error, fmt, rc::Rc};
#[derive(Debug)]
pub enum GpuA2cError {
    Device(GpuError),
    Autograd(GpuAutogradError),
    Network(GpuPpoError),
    Loss(GpuA2cLossError),
    Checkpoint(GpuA2cCheckpointError),
    InvalidInput(&'static str),
}
impl fmt::Display for GpuA2cError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(e) => e.fmt(f),
            Self::Autograd(e) => e.fmt(f),
            Self::Network(e) => e.fmt(f),
            Self::Loss(e) => e.fmt(f),
            Self::Checkpoint(e) => e.fmt(f),
            Self::InvalidInput(e) => f.write_str(e),
        }
    }
}
impl Error for GpuA2cError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Device(e) => Some(e),
            Self::Autograd(e) => Some(e),
            Self::Network(e) => Some(e),
            Self::Loss(e) => Some(e),
            Self::Checkpoint(e) => Some(e),
            Self::InvalidInput(_) => None,
        }
    }
}
macro_rules! conversion {
    ($source:ty,$variant:ident) => {
        impl From<$source> for GpuA2cError {
            fn from(e: $source) -> Self {
                Self::$variant(e)
            }
        }
    };
}
conversion!(GpuError, Device);
conversion!(GpuAutogradError, Autograd);
conversion!(GpuPpoError, Network);
conversion!(GpuA2cLossError, Loss);
conversion!(GpuA2cCheckpointError, Checkpoint);
type Result<T> = std::result::Result<T, GpuA2cError>;
#[derive(Clone, Copy, Debug)]
pub struct GpuA2cRolloutOptions {
    pub episodes: usize,
    pub max_steps: usize,
    pub seed: Option<u64>,
}
pub struct GpuA2c {
    context: GpuContext,
    net: GpuA2cNet,
    optimizer: GpuAdam,
    config: A2CConfig,
    updates: u64,
}
impl GpuA2c {
    pub fn validate_config(config: &A2CConfig) -> Result<()> {
        GpuA2cLossConfig {
            value_coef: config.c_value,
            entropy_coef: config.c_entropy,
        }
        .validate()?;
        crate::agent::a2c::validate_a2c_config(config).map_err(GpuA2cError::InvalidInput)
    }

    pub fn new_seeded(context: &GpuContext, config: A2CConfig, seed: u64) -> Result<Self> {
        Self::validate_config(&config)?;
        let net = GpuA2cNet::new_seeded(
            context,
            config.obs_dim,
            config.hidden_dim,
            config.num_actions,
            seed,
        )?;
        let optimizer = GpuAdam::new(net.parameters(), config.lr)?;
        Ok(Self {
            context: context.clone(),
            net,
            optimizer,
            config,
            updates: 0,
        })
    }
    pub fn config(&self) -> &A2CConfig {
        &self.config
    }
    pub fn net(&self) -> &GpuA2cNet {
        &self.net
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    fn state(&self, state: &[f32]) -> Result<GpuVariable> {
        if state.len() != self.config.obs_dim || state.iter().any(|v| !v.is_finite()) {
            return Err(GpuA2cError::InvalidInput(
                "observation must have the configured dimension and finite values",
            ));
        }
        Ok(GpuVariable::new(
            &self.context,
            &Tensor::from_vec(state.to_vec(), &[1, state.len()]),
            false,
        )?)
    }
    pub fn value_of(&self, state: &[f32]) -> Result<f32> {
        let input = self.state(state)?;
        let value = no_grad(|| self.net.forward(&input))?.1.to_cpu()?.item();
        if !value.is_finite() {
            return Err(GpuA2cLossError::NonFinite.into());
        }
        Ok(value)
    }
    fn distribution(&self, state: &[f32]) -> Result<(Vec<f32>, Vec<f32>, f32)> {
        let input = self.state(state)?;
        let (logits, value) = no_grad(|| self.net.forward(&input))?;
        let logs = no_grad(|| logits.log_softmax())?.to_cpu()?.to_vec();
        let value = value.to_cpu()?.item();
        if !value.is_finite() || logs.iter().any(|v| !v.is_finite()) {
            return Err(GpuA2cLossError::NonFinite.into());
        }
        let mut probs: Vec<_> = logs.iter().map(|l| l.exp()).collect();
        let total: f32 = probs.iter().sum();
        if !total.is_finite() || total <= 0. {
            return Err(GpuA2cLossError::NonFinite.into());
        }
        for p in &mut probs {
            *p /= total;
        }
        Ok((logs, probs, value))
    }
    /// Explicit inference readback; does not record an autograd graph.
    pub fn action_probabilities(&self, state: &[f32]) -> Result<Vec<f32>> {
        Ok(self.distribution(state)?.1)
    }
    /// Samples using one caller-owned RNG draw; returns action, log density and value.
    pub fn select_action_with_rng<R: Rng + ?Sized>(
        &self,
        state: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32, f32)> {
        let (logs, probs, value) = self.distribution(state)?;
        let sample = rng.gen::<f32>();
        let mut cumulative = 0.;
        let mut selected = probs.len() - 1;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if sample < cumulative {
                selected = i;
                break;
            }
        }
        Ok((selected, logs[selected], value))
    }
    pub fn select_action(&self, state: &[f32]) -> Result<(usize, f32, f32)> {
        self.select_action_with_rng(state, &mut rand::thread_rng())
    }
    /// Collects complete episodes, computing GAE separately before concatenating.
    /// True terminals bootstrap zero; truncation and the step limit bootstrap V(next).
    pub fn collect_rollout_with_rng<E: Environment, R: Rng + ?Sized>(
        &self,
        env: &mut E,
        options: GpuA2cRolloutOptions,
        rng: &mut R,
    ) -> Result<RolloutBatch>
    where
        E::Act: TryFrom<usize>,
    {
        let GpuA2cRolloutOptions {
            episodes,
            max_steps,
            seed,
        } = options;
        if env.action_space() != crate::env::Space::discrete(self.config.num_actions) {
            return Err(GpuA2cError::InvalidInput(
                "environment action space differs from A2C configuration",
            ));
        }
        if E::Obs::DIM != self.config.obs_dim {
            return Err(GpuA2cError::InvalidInput(
                "environment observation dimension differs from A2C configuration",
            ));
        }
        if episodes == 0
            || max_steps == 0
            || episodes
                .checked_mul(max_steps)
                .and_then(|n| n.checked_mul(self.config.obs_dim))
                .is_none()
        {
            return Err(GpuA2cError::InvalidInput(
                "rollout episodes and step limit must be positive and bounded",
            ));
        }
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut returns = Vec::new();
        let mut advantages = Vec::new();
        let mut logs = Vec::new();
        for episode in 0..episodes {
            let (observation, _) = env.reset(seed.map(|s| s.wrapping_add(episode as u64)));
            let mut state = vec![0.; self.config.obs_dim];
            observation.write_to_buffer(&mut state);
            let mut rollout = RolloutBuffer::new(max_steps, state.len());
            let mut bootstrap = 0.;
            for step in 0..max_steps {
                let (action, lp, value) = self.select_action_with_rng(&state, rng)?;
                let act = E::Act::try_from(action).map_err(|_| {
                    GpuA2cError::InvalidInput("environment rejected categorical action")
                })?;
                let (next, reward, terminal, truncated, _) = env.step(act);
                if !reward.is_finite() {
                    return Err(GpuA2cLossError::NonFinite.into());
                }
                rollout.push_with_log_prob(
                    &state,
                    action,
                    reward,
                    value,
                    if terminal { 1. } else { 0. },
                    lp,
                );
                next.write_to_buffer(&mut state);
                if state.iter().any(|v| !v.is_finite()) {
                    return Err(GpuA2cLossError::NonFinite.into());
                }
                if terminal || truncated || step + 1 == max_steps {
                    bootstrap = if terminal { 0. } else { self.value_of(&state)? };
                    break;
                }
            }
            rollout.compute_returns_and_advantages(
                self.config.gamma,
                self.config.lambda,
                bootstrap,
            );
            let batch = rollout.to_batch();
            if batch
                .returns
                .to_vec()
                .iter()
                .chain(batch.advantages.to_vec().iter())
                .any(|v| !v.is_finite())
            {
                return Err(GpuA2cLossError::NonFinite.into());
            }
            states.extend(batch.states.to_vec());
            actions.extend(batch.actions);
            returns.extend(batch.returns.to_vec());
            advantages.extend(batch.advantages.to_vec());
            logs.extend(batch.old_log_probs.to_vec());
        }
        let n = actions.len();
        Ok(RolloutBatch {
            states: Tensor::from_vec(states, &[n, self.config.obs_dim]),
            actions,
            returns: Tensor::from_vec(returns, &[n, 1]),
            advantages: Tensor::from_vec(advantages, &[n, 1]),
            old_log_probs: Tensor::from_vec(logs, &[n, 1]),
            size: n,
        })
    }
    /// One combined Adam step over active rows, matching CPU A2C. Old log
    /// probabilities and unused capacity are ignored; advantages are unnormalized.
    pub fn train_on_rollout(&mut self, batch: &RolloutBatch) -> Result<GpuA2cMetrics> {
        let n = batch.size;
        if n == 0 {
            return Ok(GpuA2cMetrics {
                actor_loss: 0.,
                value_loss: 0.,
                entropy: 0.,
                total_loss: 0.,
            });
        }
        if !rustforge_autograd::is_grad_enabled() {
            return Err(GpuA2cError::InvalidInput(
                "A2C training requires gradient recording",
            ));
        }
        let obs = self.config.obs_dim;
        let state_count = n
            .checked_mul(obs)
            .ok_or(GpuA2cError::InvalidInput("A2C batch size overflow"))?;
        for (tensor, cols) in [
            (&batch.states, obs),
            (&batch.advantages, 1),
            (&batch.returns, 1),
        ] {
            if tensor.shape().len() != 2 || tensor.shape()[0] < n || tensor.shape()[1] != cols {
                return Err(GpuA2cError::InvalidInput(
                    "invalid A2C active rollout tensor shapes",
                ));
            }
        }
        if batch.actions.len() < n
            || batch.actions[..n]
                .iter()
                .any(|&a| a >= self.config.num_actions)
        {
            return Err(GpuA2cError::InvalidInput("invalid A2C rollout actions"));
        }
        let states = batch.states.to_vec();
        let advantages = batch.advantages.to_vec();
        let returns = batch.returns.to_vec();
        if states[..state_count]
            .iter()
            .chain(&advantages[..n])
            .chain(&returns[..n])
            .any(|v| !v.is_finite())
        {
            return Err(GpuA2cLossError::NonFinite.into());
        }
        let next = self
            .updates
            .checked_add(1)
            .and_then(|s| usize::try_from(s).ok())
            .ok_or(GpuA2cError::InvalidInput("A2C update clock overflow"))?;
        if next == usize::MAX {
            return Err(GpuA2cError::InvalidInput("A2C update clock overflow"));
        }
        let upload = |data: Vec<f32>, cols| {
            GpuVariable::new(&self.context, &Tensor::from_vec(data, &[n, cols]), false)
        };
        let input = upload(states[..state_count].to_vec(), obs)?;
        let advantages = upload(advantages[..n].to_vec(), 1)?;
        let returns = upload(returns[..n].to_vec(), 1)?;
        let actions = Rc::new(
            self.context
                .upload_indices(&batch.actions[..n], self.config.num_actions)?,
        );
        let (logits, values) = self.net.forward(&input)?;
        let loss = a2c_loss(
            &logits,
            &values,
            &actions,
            &advantages,
            &returns,
            GpuA2cLossConfig {
                value_coef: self.config.c_value,
                entropy_coef: self.config.c_entropy,
            },
        )?;
        let metrics = loss.checked_metrics()?;
        self.optimizer.zero_grad();
        loss.total_loss.backward()?;
        for parameter in self.net.parameters() {
            if let Some(gradient) = parameter.grad() {
                // Adam squares gradients before weighting its second moment.
                // A finite gradient can still overflow that intermediate.
                let squared = self.context.mul_device(&gradient, &gradient)?;
                if self
                    .context
                    .download(&self.context.nonfinite_count_device(&squared)?)?
                    .item()
                    != 0.
                {
                    return Err(GpuA2cLossError::NonFinite.into());
                }
                if self
                    .context
                    .download(&self.context.nonfinite_count_device(&gradient)?)?
                    .item()
                    != 0.
                {
                    return Err(GpuA2cLossError::NonFinite.into());
                }
            }
        }
        self.optimizer.step()?;
        self.updates += 1;
        Ok(metrics)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a2c_agent_config_validation_requires_no_adapter() {
        assert!(GpuA2c::validate_config(&A2CConfig::default()).is_ok());
        for kind in 0..10 {
            let mut c = A2CConfig::default();
            match kind {
                0 => c.obs_dim = 0,
                1 => c.hidden_dim = 0,
                2 => c.num_actions = 0,
                3 => c.lr = 0.,
                4 => c.lr = f32::NAN,
                5 => c.gamma = 1.1,
                6 => c.lambda = -0.1,
                7 => c.c_value = -1.,
                8 => c.c_entropy = f32::INFINITY,
                _ => c.obs_dim = usize::MAX,
            }
            assert!(GpuA2c::validate_config(&c).is_err(), "case {kind}");
        }
    }
}
