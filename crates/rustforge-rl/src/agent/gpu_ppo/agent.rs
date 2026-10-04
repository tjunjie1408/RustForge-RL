//! Discrete PPO with CPU rollout/GAE and device-resident model and Adam state.
#[path = "checkpoint.rs"]
pub(super) mod checkpoint;
use super::{discrete_ppo_loss, GpuPpoLossConfig, GpuPpoLossError, GpuPpoMetrics};
use crate::{
    agent::ppo::PPODiscreteConfig,
    buffer::{RolloutBatch, RolloutBuffer},
    env::{Environment, IntoTensorBuffer},
};
pub use checkpoint::{
    GpuPpoCheckpointError, CHECKPOINT_MAGIC, CHECKPOINT_VERSION, MAX_CHECKPOINT_BYTES,
};
use rand::{seq::SliceRandom, Rng};
use rustforge_autograd::{
    gpu::{GpuAdam, GpuAutogradError, GpuVariable},
    no_grad,
};
use rustforge_nn::gpu::{GpuLinear, GpuModule, GpuModuleError};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};
use std::{error::Error, fmt, rc::Rc};

#[derive(Debug)]
pub enum GpuPpoError {
    Device(GpuError),
    Autograd(GpuAutogradError),
    Module(GpuModuleError),
    Loss(GpuPpoLossError),
    Gaussian(crate::agent::gpu_gaussian::GpuGaussianError),
    Checkpoint(GpuPpoCheckpointError),
    InvalidInput(&'static str),
}
impl fmt::Display for GpuPpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(e) => e.fmt(f),
            Self::Autograd(e) => e.fmt(f),
            Self::Module(e) => e.fmt(f),
            Self::Loss(e) => e.fmt(f),
            Self::Gaussian(e) => e.fmt(f),
            Self::Checkpoint(e) => e.fmt(f),
            Self::InvalidInput(e) => f.write_str(e),
        }
    }
}
impl Error for GpuPpoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Device(e) => Some(e),
            Self::Autograd(e) => Some(e),
            Self::Module(e) => Some(e),
            Self::Loss(e) => Some(e),
            Self::Gaussian(e) => Some(e),
            Self::Checkpoint(e) => Some(e),
            Self::InvalidInput(_) => None,
        }
    }
}
macro_rules! conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for GpuPpoError {
            fn from(e: $source) -> Self {
                Self::$variant(e)
            }
        }
    };
}
conversion!(GpuError, Device);
conversion!(GpuAutogradError, Autograd);
conversion!(GpuModuleError, Module);
conversion!(GpuPpoLossError, Loss);
conversion!(crate::agent::gpu_gaussian::GpuGaussianError, Gaussian);
conversion!(GpuPpoCheckpointError, Checkpoint);
type Result<T> = std::result::Result<T, GpuPpoError>;

pub struct GpuActorCriticNet {
    trunk: GpuLinear,
    actor: GpuLinear,
    critic: GpuLinear,
}
impl GpuActorCriticNet {
    pub fn new_seeded(
        context: &GpuContext,
        obs: usize,
        hidden: usize,
        actions: usize,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            trunk: GpuLinear::new_seeded(context, obs, hidden, seed)?,
            actor: GpuLinear::new_seeded(context, hidden, actions, seed.wrapping_add(1))?,
            critic: GpuLinear::new_seeded(context, hidden, 1, seed.wrapping_add(2))?,
        })
    }
    pub fn forward(&self, input: &GpuVariable) -> Result<(GpuVariable, GpuVariable)> {
        let features = self.trunk.forward(input)?.relu()?;
        Ok((
            self.actor.forward(&features)?,
            self.critic.forward(&features)?,
        ))
    }
    pub fn parameters(&self) -> Vec<GpuVariable> {
        let mut p = self.trunk.parameters();
        p.extend(self.actor.parameters());
        p.extend(self.critic.parameters());
        p
    }
}

pub struct GpuPpoDiscrete {
    context: GpuContext,
    net: GpuActorCriticNet,
    optimizer: GpuAdam,
    config: PPODiscreteConfig,
    updates: u64,
}
impl GpuPpoDiscrete {
    pub fn new_seeded(context: &GpuContext, config: PPODiscreteConfig, seed: u64) -> Result<Self> {
        Self::validate_config(&config)?;
        let c = &config.base;
        let net = GpuActorCriticNet::new_seeded(
            context,
            c.obs_dim,
            c.hidden_dim,
            config.num_actions,
            seed,
        )?;
        let optimizer = GpuAdam::new(net.parameters(), c.lr)?;
        Ok(Self {
            context: context.clone(),
            net,
            optimizer,
            config,
            updates: 0,
        })
    }
    pub(super) fn validate_config(config: &PPODiscreteConfig) -> Result<()> {
        let c = &config.base;
        Self::loss_config(config).validate()?;
        if c.obs_dim == 0
            || c.hidden_dim == 0
            || config.num_actions == 0
            || !c.lr.is_finite()
            || c.lr <= 0.
            || !c.gamma.is_finite()
            || !(0. ..=1.).contains(&c.gamma)
            || !c.gae_lambda.is_finite()
            || !(0. ..=1.).contains(&c.gae_lambda)
            || c.ppo_epochs == 0
            || c.mini_batch_size == 0
        {
            return Err(GpuPpoError::InvalidInput("PPO dimensions, epochs and minibatch size must be positive; learning rate positive and finite; gamma/lambda in [0,1]"));
        }
        Ok(())
    }
    pub fn config(&self) -> &PPODiscreteConfig {
        &self.config
    }
    fn loss_config(c: &PPODiscreteConfig) -> GpuPpoLossConfig {
        GpuPpoLossConfig {
            clip_eps: c.base.clip_eps,
            value_coef: c.base.value_coef,
            entropy_coef: c.base.entropy_coef,
        }
    }
    pub fn net(&self) -> &GpuActorCriticNet {
        &self.net
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    fn state(&self, state: &[f32]) -> Result<GpuVariable> {
        if state.len() != self.config.base.obs_dim || state.iter().any(|v| !v.is_finite()) {
            return Err(GpuPpoError::InvalidInput(
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
        let value = no_grad(|| self.net.forward(&self.state(state)?))?
            .1
            .to_cpu()?
            .item();
        if !value.is_finite() {
            return Err(GpuPpoLossError::NonFinite.into());
        }
        Ok(value)
    }
    /// Explicit environment-boundary readbacks; caller owns the sampling stream.
    pub fn select_action_with_rng<R: Rng + ?Sized>(
        &self,
        state: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32, f32)> {
        let (logits, value) = no_grad(|| self.net.forward(&self.state(state)?))?;
        let logs = no_grad(|| logits.log_softmax())?.to_cpu()?.to_vec();
        let value = value.to_cpu()?.item();
        if !value.is_finite() || logs.iter().any(|x| !x.is_finite()) {
            return Err(GpuPpoLossError::NonFinite.into());
        }
        let probs: Vec<_> = logs.iter().map(|x| x.exp()).collect();
        let total: f32 = probs.iter().sum();
        if !total.is_finite() || total <= 0. {
            return Err(GpuPpoLossError::NonFinite.into());
        }
        let sample = rng.gen::<f32>() * total;
        let mut cumulative = 0.;
        let mut selected = 0;
        for (i, p) in probs.iter().enumerate() {
            if *p > 0. {
                selected = i;
                cumulative += p;
                if sample < cumulative {
                    break;
                }
            }
        }
        Ok((selected, logs[selected], value))
    }
    /// Collects complete episodes, computing GAE separately before concatenating.
    /// True terminals bootstrap zero; truncation and the step limit bootstrap V(next).
    pub fn collect_rollout_with_rng<E: Environment, R: Rng + ?Sized>(
        &self,
        env: &mut E,
        episodes: usize,
        max_steps: usize,
        seed: Option<u64>,
        rng: &mut R,
    ) -> Result<RolloutBatch>
    where
        E::Act: TryFrom<usize>,
    {
        if E::Obs::DIM != self.config.base.obs_dim {
            return Err(GpuPpoError::InvalidInput(
                "environment observation dimension differs from PPO configuration",
            ));
        }
        if episodes == 0 || max_steps == 0 || episodes.checked_mul(max_steps).is_none() {
            return Err(GpuPpoError::InvalidInput(
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
            let mut state = vec![0.; self.config.base.obs_dim];
            observation.write_to_buffer(&mut state);
            let mut rollout = RolloutBuffer::new(max_steps, state.len());
            let mut bootstrap = 0.;
            for step in 0..max_steps {
                let (action, lp, value) = self.select_action_with_rng(&state, rng)?;
                let act = E::Act::try_from(action).map_err(|_| {
                    GpuPpoError::InvalidInput("environment rejected categorical action")
                })?;
                let (next, reward, terminal, truncated, _) = env.step(act);
                if !reward.is_finite() {
                    return Err(GpuPpoLossError::NonFinite.into());
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
                if terminal || truncated || step + 1 == max_steps {
                    bootstrap = if terminal { 0. } else { self.value_of(&state)? };
                    break;
                }
            }
            rollout.compute_returns_and_advantages(
                self.config.base.gamma,
                self.config.base.gae_lambda,
                bootstrap,
            );
            let batch = rollout.to_batch();
            states.extend(batch.states.to_vec());
            actions.extend(batch.actions);
            returns.extend(batch.returns.to_vec());
            advantages.extend(batch.advantages.to_vec());
            logs.extend(batch.old_log_probs.to_vec());
        }
        let n = actions.len();
        Ok(RolloutBatch {
            states: Tensor::from_vec(states, &[n, self.config.base.obs_dim]),
            actions,
            returns: Tensor::from_vec(returns, &[n, 1]),
            advantages: Tensor::from_vec(advantages, &[n, 1]),
            old_log_probs: Tensor::from_vec(logs, &[n, 1]),
            size: n,
        })
    }
    /// Mean metrics over minibatch updates, matching CPU PPO. Validation precedes
    /// mutation; runtime failures retain earlier completed minibatch updates.
    pub fn train_on_batch_with_rng<R: Rng + ?Sized>(
        &mut self,
        batch: &RolloutBatch,
        rng: &mut R,
    ) -> Result<GpuPpoMetrics> {
        let mut metrics = GpuPpoMetrics {
            policy_loss: 0.,
            value_loss: 0.,
            entropy: 0.,
            total_loss: 0.,
        };
        let n = batch.size;
        if n == 0 {
            return Ok(metrics);
        }
        if !rustforge_autograd::is_grad_enabled() {
            return Err(GpuPpoError::InvalidInput(
                "PPO training requires gradient recording",
            ));
        }
        let obs = self.config.base.obs_dim;
        if batch.states.shape().len() != 2
            || batch.states.shape()[0] < n
            || batch.states.shape()[1] != obs
            || batch.actions.len() < n
            || batch.actions[..n]
                .iter()
                .any(|&a| a >= self.config.num_actions)
        {
            return Err(GpuPpoError::InvalidInput(
                "invalid rollout states or actions",
            ));
        }
        for t in [&batch.returns, &batch.advantages, &batch.old_log_probs] {
            if t.shape().len() != 2 || t.shape()[0] < n || t.shape()[1] != 1 {
                return Err(GpuPpoError::InvalidInput(
                    "rollout references must have active [batch,1] rows",
                ));
            }
        }
        let states = batch.states.to_vec();
        let returns = batch.returns.to_vec();
        let old = batch.old_log_probs.to_vec();
        let adv = batch.advantages.to_vec();
        if states[..n * obs]
            .iter()
            .chain(&returns[..n])
            .chain(&old[..n])
            .chain(&adv[..n])
            .any(|v| !v.is_finite())
        {
            return Err(GpuPpoLossError::NonFinite.into());
        }
        let mean = adv[..n].iter().sum::<f32>() / n as f32;
        let std =
            (adv[..n].iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32 + 1e-8).sqrt();
        let adv: Vec<_> = adv[..n].iter().map(|a| (a - mean) / std).collect();
        if !mean.is_finite() || !std.is_finite() || adv.iter().any(|a| !a.is_finite()) {
            return Err(GpuPpoLossError::NonFinite.into());
        }
        let mb = self.config.base.mini_batch_size.min(n);
        let count = n
            .div_ceil(mb)
            .checked_mul(self.config.base.ppo_epochs)
            .and_then(|c| u64::try_from(c).ok())
            .ok_or(GpuPpoError::InvalidInput("PPO update count overflow"))?;
        self.updates
            .checked_add(count)
            .ok_or(GpuPpoError::InvalidInput("PPO update clock overflow"))?;
        let mut indices: Vec<_> = (0..n).collect();
        for _ in 0..self.config.base.ppo_epochs {
            indices.shuffle(rng);
            for rows in indices.chunks(mb) {
                let upload = |data: Vec<f32>, cols: usize| -> Result<GpuVariable> {
                    Ok(GpuVariable::new(
                        &self.context,
                        &Tensor::from_vec(data, &[rows.len(), cols]),
                        false,
                    )?)
                };
                let input = upload(
                    rows.iter()
                        .flat_map(|&i| states[i * obs..(i + 1) * obs].iter().copied())
                        .collect(),
                    obs,
                )?;
                let ret = upload(rows.iter().map(|&i| returns[i]).collect(), 1)?;
                let advantage = upload(rows.iter().map(|&i| adv[i]).collect(), 1)?;
                let lp = upload(rows.iter().map(|&i| old[i]).collect(), 1)?;
                let actions = Rc::new(self.context.upload_indices(
                    &rows.iter().map(|&i| batch.actions[i]).collect::<Vec<_>>(),
                    self.config.num_actions,
                )?);
                let (logits, values) = self.net.forward(&input)?;
                let loss = discrete_ppo_loss(
                    &logits,
                    &values,
                    &actions,
                    &lp,
                    &advantage,
                    &ret,
                    Self::loss_config(&self.config),
                )?;
                let m = loss.checked_metrics()?;
                self.optimizer.zero_grad();
                loss.total_loss.backward()?;
                // Scalar readbacks guard Adam moments against finite-forward,
                // overflowing-backward objectives without downloading gradients.
                for parameter in self.net.parameters() {
                    if let Some(gradient) = parameter.grad() {
                        let invalid = self.context.nonfinite_count_device(&gradient)?;
                        if self.context.download(&invalid)?.item() != 0. {
                            return Err(GpuPpoLossError::NonFinite.into());
                        }
                    }
                }
                self.optimizer.step()?;
                self.updates += 1;
                metrics.policy_loss += m.policy_loss;
                metrics.value_loss += m.value_loss;
                metrics.entropy += m.entropy;
                metrics.total_loss += m.total_loss;
            }
        }
        metrics.policy_loss /= count as f32;
        metrics.value_loss /= count as f32;
        metrics.entropy /= count as f32;
        metrics.total_loss /= count as f32;
        Ok(metrics)
    }
}
