//! Continuous GPU PPO; environment interaction, GAE and shuffling stay on CPU.
#[path = "continuous_checkpoint.rs"]
mod checkpoint;
use super::{
    continuous_ppo_loss, GpuContinuousPpoInputs, GpuContinuousPpoMetrics, GpuPpoError,
    GpuPpoLossError,
};
use crate::{
    agent::{
        gaussian_policy::sample_standard_normal, gpu_gaussian::GpuGaussianTransform,
        PPOContinuousConfig,
    },
    buffer::{ContinuousRolloutBatch, ContinuousRolloutBuffer},
    env::{Environment, IntoTensorBuffer, Space},
};
pub use checkpoint::{CONTINUOUS_CHECKPOINT_MAGIC, CONTINUOUS_CHECKPOINT_VERSION};
use rand::{seq::SliceRandom, Rng};
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad,
};
use rustforge_nn::gpu::{GpuLinear, GpuModule, GpuReLU, GpuSequential};
use rustforge_tensor::{gpu::GpuContext, Tensor};
type Result<T> = std::result::Result<T, GpuPpoError>;

/// Matches CPU GaussianPolicyNet's architecture/order. Raw std outputs are
/// passed to GpuGaussianTransform, which applies the CPU clipping contract.
pub struct GpuGaussianPolicyNet {
    trunk: GpuSequential,
    mean: GpuLinear,
    log_std: GpuLinear,
}
impl GpuGaussianPolicyNet {
    pub fn new_seeded(
        context: &GpuContext,
        obs: usize,
        hidden: usize,
        actions: usize,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            trunk: GpuSequential::new(vec![
                Box::new(GpuLinear::new_seeded(context, obs, hidden, seed)?),
                Box::new(GpuReLU),
                Box::new(GpuLinear::new_seeded(
                    context,
                    hidden,
                    hidden,
                    seed.wrapping_add(1),
                )?),
                Box::new(GpuReLU),
            ]),
            mean: GpuLinear::new_seeded(context, hidden, actions, seed.wrapping_add(2))?,
            log_std: GpuLinear::new_seeded(context, hidden, actions, seed.wrapping_add(3))?,
        })
    }
    pub fn forward_raw(&self, input: &GpuVariable) -> Result<(GpuVariable, GpuVariable)> {
        let features = self.trunk.forward(input)?;
        Ok((
            self.mean.forward(&features)?,
            self.log_std.forward(&features)?,
        ))
    }
    pub fn parameters(&self) -> Vec<GpuVariable> {
        let mut p = self.trunk.parameters();
        p.extend(self.mean.parameters());
        p.extend(self.log_std.parameters());
        p
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GpuContinuousRolloutOptions {
    pub episodes: usize,
    pub max_steps: usize,
    pub seed: Option<u64>,
}
pub struct GpuPpoContinuous {
    context: GpuContext,
    actor: GpuGaussianPolicyNet,
    critic: GpuSequential,
    transform: GpuGaussianTransform,
    actor_optimizer: GpuAdam,
    critic_optimizer: GpuAdam,
    config: PPOContinuousConfig,
    actor_updates: u64,
    critic_updates: u64,
}
impl GpuPpoContinuous {
    pub(crate) fn validate_config(config: &PPOContinuousConfig) -> Result<()> {
        crate::agent::ppo_continuous_backend::validate_continuous_config(config)
            .map_err(GpuPpoError::InvalidInput)
    }

    pub fn new_seeded(
        context: &GpuContext,
        config: PPOContinuousConfig,
        seed: u64,
    ) -> Result<Self> {
        Self::validate_config(&config)?;
        let transform =
            GpuGaussianTransform::new(context, &config.action_low, &config.action_high)?;
        let c = &config.base;
        let actor = GpuGaussianPolicyNet::new_seeded(
            context,
            c.obs_dim,
            c.hidden_dim,
            config.act_dim,
            seed,
        )?;
        let critic = GpuSequential::new(vec![
            Box::new(GpuLinear::new_seeded(
                context,
                c.obs_dim,
                c.hidden_dim,
                seed.wrapping_add(4),
            )?),
            Box::new(GpuReLU),
            Box::new(GpuLinear::new_seeded(
                context,
                c.hidden_dim,
                c.hidden_dim,
                seed.wrapping_add(5),
            )?),
            Box::new(GpuReLU),
            Box::new(GpuLinear::new_seeded(
                context,
                c.hidden_dim,
                1,
                seed.wrapping_add(6),
            )?),
        ]);
        let actor_optimizer = GpuAdam::new(actor.parameters(), c.lr)?;
        let critic_optimizer = GpuAdam::new(critic.parameters(), c.lr)?;
        Ok(Self {
            context: context.clone(),
            actor,
            critic,
            transform,
            actor_optimizer,
            critic_optimizer,
            config,
            actor_updates: 0,
            critic_updates: 0,
        })
    }
    pub fn actor(&self) -> &GpuGaussianPolicyNet {
        &self.actor
    }
    pub fn critic(&self) -> &GpuSequential {
        &self.critic
    }
    pub fn config(&self) -> &PPOContinuousConfig {
        &self.config
    }
    pub fn actor_updates(&self) -> u64 {
        self.actor_updates
    }
    pub fn critic_updates(&self) -> u64 {
        self.critic_updates
    }
    fn state(&self, state: &[f32]) -> Result<GpuVariable> {
        if state.len() != self.config.base.obs_dim || state.iter().any(|v| !v.is_finite()) {
            return Err(GpuPpoError::InvalidInput(
                "observation must have configured dimension and finite values",
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
        let value = no_grad(|| self.critic.forward(&input))?.to_cpu()?.item();
        if !value.is_finite() {
            return Err(GpuPpoLossError::NonFinite.into());
        }
        Ok(value)
    }
    pub fn select_action_with_rng<R: Rng + ?Sized>(
        &self,
        state: &[f32],
        rng: &mut R,
    ) -> Result<(Vec<f32>, f32, f32)> {
        no_grad(|| {
            let input = self.state(state)?;
            let (mean, std) = self.actor.forward_raw(&input)?;
            let noise: Vec<_> = (0..self.config.act_dim)
                .map(|_| sample_standard_normal(rng))
                .collect();
            let noise = GpuVariable::new(
                &self.context,
                &Tensor::from_vec(noise, &[1, self.config.act_dim]),
                false,
            )?;
            let sample = self.transform.sample_with_noise(&mean, &std, &noise)?;
            let metrics = sample.distribution.checked_metrics()?;
            let action = sample.actions.to_cpu()?.to_vec();
            let value = self.critic.forward(&input)?.to_cpu()?.item();
            if !value.is_finite() || action.iter().any(|a| !a.is_finite()) {
                return Err(GpuPpoLossError::NonFinite.into());
            }
            Ok((action, metrics.mean_log_prob, value))
        })
    }
    pub fn select_action(&self, state: &[f32]) -> Result<(Vec<f32>, f32, f32)> {
        self.select_action_with_rng(state, &mut rand::thread_rng())
    }
    pub fn deterministic_action(&self, state: &[f32]) -> Result<Vec<f32>> {
        no_grad(|| {
            let (mean, std) = self.actor.forward_raw(&self.state(state)?)?;
            let noise = GpuVariable::from_device(
                &self.context,
                self.context.zeros(&[1, self.config.act_dim])?,
                false,
            )?;
            let sample = self.transform.sample_with_noise(&mean, &std, &noise)?;
            sample.distribution.checked_metrics()?;
            Ok(sample.actions.to_cpu()?.to_vec())
        })
    }
    /// Each episode gets independent GAE before batches are concatenated. The
    /// callback converts a validated action vector to the environment's Act type.
    pub fn collect_rollout_with_rng<
        E: Environment,
        R: Rng + ?Sized,
        F: FnMut(&[f32]) -> Result<E::Act>,
    >(
        &self,
        env: &mut E,
        options: GpuContinuousRolloutOptions,
        rng: &mut R,
        mut action: F,
    ) -> Result<ContinuousRolloutBatch> {
        let GpuContinuousRolloutOptions {
            episodes,
            max_steps,
            seed,
        } = options;
        let obs = self.config.base.obs_dim;
        let act = self.config.act_dim;
        if E::Obs::DIM != obs {
            return Err(GpuPpoError::InvalidInput(
                "environment observation dimension differs from PPO configuration",
            ));
        }
        if !matches!(env.action_space(),Space::Box{low,high,shape} if low==self.config.action_low&&high==self.config.action_high&&shape==[act])
        {
            return Err(GpuPpoError::InvalidInput(
                "environment action bounds/dimensions differ from PPO configuration",
            ));
        }
        let capacity = episodes
            .checked_mul(max_steps)
            .ok_or(GpuPpoError::InvalidInput("rollout capacity overflow"))?;
        if episodes == 0
            || max_steps == 0
            || capacity.checked_mul(obs).is_none()
            || capacity.checked_mul(act).is_none()
        {
            return Err(GpuPpoError::InvalidInput(
                "rollout episodes/step limit must be positive and bounded",
            ));
        }
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut returns = Vec::new();
        let mut advantages = Vec::new();
        let mut logs = Vec::new();
        for episode in 0..episodes {
            let (observation, _) = env.reset(seed.map(|s| s.wrapping_add(episode as u64)));
            let mut state = vec![0.; obs];
            observation.write_to_buffer(&mut state);
            let mut rollout = ContinuousRolloutBuffer::new(max_steps, obs, act);
            let mut bootstrap = 0.;
            for step in 0..max_steps {
                let (a, lp, value) = self.select_action_with_rng(&state, rng)?;
                let (next, reward, terminal, truncated, _) = env.step(action(&a)?);
                if !reward.is_finite() {
                    return Err(GpuPpoLossError::NonFinite.into());
                }
                rollout.push(
                    &state,
                    &a,
                    reward,
                    value,
                    if terminal { 1. } else { 0. },
                    lp,
                );
                next.write_to_buffer(&mut state);
                if state.iter().any(|v| !v.is_finite()) {
                    return Err(GpuPpoLossError::NonFinite.into());
                }
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
            let mut batch = ContinuousRolloutBatch::new(rollout.len(), obs, act);
            rollout.fill_batch(&mut batch);
            states.extend(batch.states.to_vec());
            actions.extend(batch.actions.to_vec());
            returns.extend(batch.returns.to_vec());
            advantages.extend(batch.advantages.to_vec());
            logs.extend(batch.old_log_probs.to_vec());
        }
        let n = returns.len();
        Ok(ContinuousRolloutBatch {
            states: Tensor::from_vec(states, &[n, obs]),
            actions: Tensor::from_vec(actions, &[n, act]),
            returns: Tensor::from_vec(returns, &[n, 1]),
            advantages: Tensor::from_vec(advantages, &[n, 1]),
            old_log_probs: Tensor::from_vec(logs, &[n, 1]),
            size: n,
        })
    }
    pub fn train_on_batch(
        &mut self,
        batch: &ContinuousRolloutBatch,
    ) -> Result<GpuContinuousPpoMetrics> {
        self.train_on_batch_with_rng(batch, &mut rand::thread_rng())
    }
    /// Averages metrics per minibatch (CPU parity); errors retain previously
    /// completed optimizer steps. Actor/critic clocks track each successful step.
    pub fn train_on_batch_with_rng<R: Rng + ?Sized>(
        &mut self,
        batch: &ContinuousRolloutBatch,
        rng: &mut R,
    ) -> Result<GpuContinuousPpoMetrics> {
        let mut metrics = GpuContinuousPpoMetrics {
            policy_loss: 0.,
            value_loss: 0.,
            base_entropy: 0.,
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
        let act = self.config.act_dim;
        for (tensor, cols) in [
            (&batch.states, obs),
            (&batch.actions, act),
            (&batch.returns, 1),
            (&batch.advantages, 1),
            (&batch.old_log_probs, 1),
        ] {
            if tensor.shape().len() != 2 || tensor.shape()[0] < n || tensor.shape()[1] != cols {
                return Err(GpuPpoError::InvalidInput(
                    "invalid continuous rollout shapes or active size",
                ));
            }
            if tensor.to_vec()[..n * cols].iter().any(|v| !v.is_finite()) {
                return Err(GpuPpoLossError::NonFinite.into());
            }
        }
        let states = batch.states.to_vec();
        let actions = batch.actions.to_vec();
        let returns = batch.returns.to_vec();
        let old = batch.old_log_probs.to_vec();
        let adv = batch.advantages.to_vec();
        if actions[..n * act].iter().enumerate().any(|(i, a)| {
            *a < self.config.action_low[i % act] || *a > self.config.action_high[i % act]
        }) {
            return Err(GpuPpoError::InvalidInput(
                "stored actions are outside configured bounds",
            ));
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
        for clock in [self.actor_updates, self.critic_updates] {
            let next = clock
                .checked_add(count)
                .and_then(|c| usize::try_from(c).ok())
                .ok_or(GpuPpoError::InvalidInput("PPO update clock overflow"))?;
            if next == usize::MAX {
                return Err(GpuPpoError::InvalidInput("PPO update clock overflow"));
            }
        }
        let mut indices: Vec<_> = (0..n).collect();
        for _ in 0..self.config.base.ppo_epochs {
            indices.shuffle(rng);
            for rows in indices.chunks(mb) {
                let upload = |data: Vec<f32>, cols| -> Result<GpuVariable> {
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
                let action = upload(
                    rows.iter()
                        .flat_map(|&i| actions[i * act..(i + 1) * act].iter().copied())
                        .collect(),
                    act,
                )?;
                let ret = upload(rows.iter().map(|&i| returns[i]).collect(), 1)?;
                let advantage = upload(rows.iter().map(|&i| adv[i]).collect(), 1)?;
                let lp = upload(rows.iter().map(|&i| old[i]).collect(), 1)?;
                let (mean, raw) = self.actor.forward_raw(&input)?;
                let value = self.critic.forward(&input)?;
                let loss = continuous_ppo_loss(
                    GpuContinuousPpoInputs {
                        mean: &mean,
                        raw_log_std: &raw,
                        values: &value,
                        actions: &action,
                        old_log_probs: &lp,
                        advantages: &advantage,
                        returns: &ret,
                    },
                    &self.transform,
                    self.config.base.clip_eps,
                )?;
                let m = loss.checked_metrics()?;
                self.actor_optimizer.zero_grad();
                self.critic_optimizer.zero_grad();
                loss.policy_loss.backward()?;
                loss.value_loss.backward()?;
                for parameter in self
                    .actor
                    .parameters()
                    .into_iter()
                    .chain(self.critic.parameters())
                {
                    if let Some(gradient) = parameter.grad() {
                        let invalid = self.context.nonfinite_count_device(&gradient)?;
                        if self.context.download(&invalid)?.item() != 0. {
                            return Err(GpuPpoLossError::NonFinite.into());
                        }
                    }
                }
                self.actor_optimizer.step()?;
                self.actor_updates += 1;
                self.critic_optimizer.step()?;
                self.critic_updates += 1;
                metrics.policy_loss += m.policy_loss;
                metrics.value_loss += m.value_loss;
                metrics.base_entropy += m.base_entropy;
            }
        }
        metrics.policy_loss /= count as f32;
        metrics.value_loss /= count as f32;
        metrics.base_entropy /= count as f32;
        Ok(metrics)
    }
}
