//! Owned policy/Adam and successful update clock; CPU episode-local Monte Carlo rollouts.
use super::{reinforce_loss, validate_dimensions, GpuReinforceError, GpuReinforceNet, Result};
use crate::{
    agent::REINFORCEConfig,
    buffer::{RolloutBatch, RolloutBuffer},
    env::{Environment, IntoTensorBuffer, Space},
};
use rand::Rng;
use rustforge_autograd::{
    gpu::{GpuAdam, GpuVariable},
    no_grad,
};
use rustforge_tensor::{gpu::GpuContext, Tensor};
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
pub struct GpuReinforceRolloutOptions {
    pub episodes: usize,
    pub max_steps: usize,
    pub seed: Option<u64>,
}
pub struct GpuReinforce {
    context: GpuContext,
    net: GpuReinforceNet,
    optimizer: GpuAdam,
    config: REINFORCEConfig,
    updates: u64,
}
impl GpuReinforce {
    pub fn validate_config(config: &REINFORCEConfig) -> Result<()> {
        validate_dimensions(config.obs_dim, config.hidden_dim, config.num_actions)?;
        if !config.lr.is_finite()
            || config.lr <= 0.
            || !config.gamma.is_finite()
            || !(0. ..=1.).contains(&config.gamma)
        {
            return Err(GpuReinforceError::InvalidInput(
                "REINFORCE requires positive finite learning rate and discount in [0,1]",
            ));
        }
        Ok(())
    }
    pub fn new_seeded(context: &GpuContext, config: REINFORCEConfig, seed: u64) -> Result<Self> {
        Self::validate_config(&config)?;
        let net = GpuReinforceNet::new_seeded(
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
    pub fn config(&self) -> &REINFORCEConfig {
        &self.config
    }
    pub fn net(&self) -> &GpuReinforceNet {
        &self.net
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    fn distribution(&self, state: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if state.len() != self.config.obs_dim || state.iter().any(|v| !v.is_finite()) {
            return Err(GpuReinforceError::InvalidInput(
                "observation must have configured dimension and finite values",
            ));
        }
        let input = GpuVariable::new(
            &self.context,
            &Tensor::from_vec(state.to_vec(), &[1, state.len()]),
            false,
        )?;
        let logs = no_grad(|| {
            self.net
                .forward(&input)?
                .log_softmax()
                .map_err(GpuReinforceError::from)
        })?
        .to_cpu()?
        .to_vec();
        if logs.iter().any(|v| !v.is_finite()) {
            return Err(GpuReinforceError::NonFinite);
        }
        let mut probabilities: Vec<_> = logs.iter().map(|v| v.exp()).collect();
        let total = probabilities.iter().sum::<f32>();
        if !total.is_finite() || total <= 0. {
            return Err(GpuReinforceError::NonFinite);
        }
        for p in &mut probabilities {
            *p /= total;
        }
        Ok((logs, probabilities))
    }
    /// Explicit inference readback without a recorded gradient graph.
    pub fn action_probabilities(&self, state: &[f32]) -> Result<Vec<f32>> {
        Ok(self.distribution(state)?.1)
    }
    /// One caller-owned RNG draw; returns action and its log probability.
    pub fn select_action_with_rng<R: Rng + ?Sized>(
        &self,
        state: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32)> {
        let (logs, probs) = self.distribution(state)?;
        let draw = rng.gen::<f32>();
        let mut cumulative = 0.;
        let mut selected = probs.len() - 1;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if draw < cumulative {
                selected = i;
                break;
            }
        }
        Ok((selected, logs[selected]))
    }
    pub fn select_action(&self, state: &[f32]) -> Result<(usize, f32)> {
        self.select_action_with_rng(state, &mut rand::thread_rng())
    }
    /// Discounted Monte Carlo returns per episode: values=0, lambda=1,
    /// final bootstrap=0 for termination, truncation and step limits alike.
    pub fn collect_rollout_with_rng<E: Environment, R: Rng + ?Sized>(
        &self,
        env: &mut E,
        options: GpuReinforceRolloutOptions,
        rng: &mut R,
    ) -> Result<RolloutBatch>
    where
        E::Act: TryFrom<usize>,
    {
        if E::Obs::DIM != self.config.obs_dim
            || env.action_space() != Space::discrete(self.config.num_actions)
        {
            return Err(GpuReinforceError::InvalidInput(
                "environment spaces differ from REINFORCE configuration",
            ));
        }
        let GpuReinforceRolloutOptions {
            episodes,
            max_steps,
            seed,
        } = options;
        if episodes == 0
            || max_steps == 0
            || episodes
                .checked_mul(max_steps)
                .and_then(|v| v.checked_mul(self.config.obs_dim))
                .is_none()
        {
            return Err(GpuReinforceError::InvalidInput(
                "rollout episodes and step limit must be positive and bounded",
            ));
        }
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut returns = Vec::new();
        let mut logs = Vec::new();
        for episode in 0..episodes {
            let (observation, _) = env.reset(seed.map(|s| s.wrapping_add(episode as u64)));
            let mut state = vec![0.; self.config.obs_dim];
            observation.write_to_buffer(&mut state);
            let mut rollout = RolloutBuffer::new(max_steps, state.len());
            for step in 0..max_steps {
                let (action, lp) = self.select_action_with_rng(&state, rng)?;
                let act = E::Act::try_from(action).map_err(|_| {
                    GpuReinforceError::InvalidInput("environment rejected categorical action")
                })?;
                let (next, reward, terminal, truncated, _) = env.step(act);
                if !reward.is_finite() {
                    return Err(GpuReinforceError::NonFinite);
                }
                rollout.push_with_log_prob(
                    &state,
                    action,
                    reward,
                    0.,
                    if terminal { 1. } else { 0. },
                    lp,
                );
                next.write_to_buffer(&mut state);
                if state.iter().any(|v| !v.is_finite()) {
                    return Err(GpuReinforceError::NonFinite);
                }
                if terminal || truncated || step + 1 == max_steps {
                    break;
                }
            }
            rollout.compute_returns_and_advantages(self.config.gamma, 1., 0.);
            let batch = rollout.to_batch();
            let episode_returns = batch.returns.to_vec();
            if episode_returns.iter().any(|v| !v.is_finite()) {
                return Err(GpuReinforceError::NonFinite);
            }
            states.extend(batch.states.to_vec());
            actions.extend(batch.actions);
            returns.extend(episode_returns);
            logs.extend(batch.old_log_probs.to_vec());
        }
        let n = actions.len();
        Ok(RolloutBatch {
            states: Tensor::from_vec(states, &[n, self.config.obs_dim]),
            actions,
            advantages: Tensor::from_vec(returns.clone(), &[n, 1]),
            returns: Tensor::from_vec(returns, &[n, 1]),
            old_log_probs: Tensor::from_vec(logs, &[n, 1]),
            size: n,
        })
    }
    /// One Adam update over active states/actions/advantages. Baseline uses only
    /// active rows; returns, old log probabilities and unused capacity are ignored.
    /// Empty batches return zero without updating the clock.
    pub fn train_on_rollout(&mut self, batch: &RolloutBatch) -> Result<f32> {
        let n = batch.size;
        if n == 0 {
            return Ok(0.);
        }
        if !rustforge_autograd::is_grad_enabled() {
            return Err(GpuReinforceError::InvalidInput(
                "REINFORCE training requires gradient recording",
            ));
        }
        let obs = self.config.obs_dim;
        let state_count = n.checked_mul(obs).ok_or(GpuReinforceError::InvalidInput(
            "REINFORCE batch size overflow",
        ))?;
        for (tensor, cols) in [(&batch.states, obs), (&batch.advantages, 1)] {
            if tensor.shape().len() != 2 || tensor.shape()[0] < n || tensor.shape()[1] != cols {
                return Err(GpuReinforceError::InvalidInput(
                    "invalid REINFORCE active rollout tensor shapes",
                ));
            }
        }
        if batch.actions.len() < n
            || batch.actions[..n]
                .iter()
                .any(|a| *a >= self.config.num_actions)
        {
            return Err(GpuReinforceError::InvalidInput(
                "invalid REINFORCE rollout actions",
            ));
        }
        let states = batch.states.to_vec();
        let advantages = batch.advantages.to_vec();
        if states[..state_count]
            .iter()
            .chain(&advantages[..n])
            .any(|v| !v.is_finite())
        {
            return Err(GpuReinforceError::NonFinite);
        }
        let next = self
            .updates
            .checked_add(1)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(GpuReinforceError::InvalidInput(
                "REINFORCE update clock overflow",
            ))?;
        if next == usize::MAX {
            return Err(GpuReinforceError::InvalidInput(
                "REINFORCE update clock overflow",
            ));
        }
        let input = GpuVariable::new(
            &self.context,
            &Tensor::from_vec(states[..state_count].to_vec(), &[n, obs]),
            false,
        )?;
        let advantages = GpuVariable::new(
            &self.context,
            &Tensor::from_vec(advantages[..n].to_vec(), &[n, 1]),
            false,
        )?;
        let actions = Rc::new(
            self.context
                .upload_indices(&batch.actions[..n], self.config.num_actions)?,
        );
        let objective = reinforce_loss(
            &self.net.forward(&input)?,
            &actions,
            &advantages,
            self.config.use_baseline,
        )?;
        let loss = objective.checked_loss()?;
        self.optimizer.zero_grad();
        objective.loss.backward()?;
        for p in self.net.parameters() {
            if let Some(g) = p.grad() {
                let square = self.context.mul_device(&g, &g)?;
                for tensor in [&*g, &square] {
                    if self
                        .context
                        .download(&self.context.nonfinite_count_device(tensor)?)?
                        .item()
                        != 0.
                    {
                        return Err(GpuReinforceError::NonFinite);
                    }
                }
            }
        }
        self.optimizer.step()?;
        self.updates += 1;
        Ok(loss)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configuration_validates_without_adapter() {
        assert!(GpuReinforce::validate_config(&REINFORCEConfig::default()).is_ok());
        for kind in 0..9 {
            let mut c = REINFORCEConfig::default();
            match kind {
                0 => c.obs_dim = 0,
                1 => c.hidden_dim = 0,
                2 => c.num_actions = 0,
                3 => c.obs_dim = usize::MAX,
                4 => c.lr = 0.,
                5 => c.lr = f32::NAN,
                6 => c.gamma = -0.1,
                7 => c.gamma = 1.1,
                _ => c.gamma = f32::INFINITY,
            }
            assert!(GpuReinforce::validate_config(&c).is_err(), "case {kind}");
        }
    }
}
