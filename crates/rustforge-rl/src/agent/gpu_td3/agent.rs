//! Owned TD3 networks and transactional device-resident updates.
mod checkpoint;
use super::*;
use crate::{agent::td3::TD3Config, buffer::ContinuousTransitionBatch};
pub use checkpoint::{
    GpuTd3CheckpointError, CHECKPOINT_MAGIC, CHECKPOINT_VERSION, MAX_CHECKPOINT_BYTES,
};
use rand::Rng;
use rustforge_autograd::{gpu::GpuAdam, no_grad};
use rustforge_nn::gpu::{GpuLinear, GpuModule, GpuModuleError};

/// Three affine layers with two ReLUs; actor outputs additionally use tanh.
pub struct GpuTd3Net {
    layers: [GpuLinear; 3],
    actor: bool,
}
impl GpuTd3Net {
    fn new(
        context: &GpuContext,
        input: usize,
        hidden: usize,
        output: usize,
        seed: u64,
        actor: bool,
    ) -> Result<Self> {
        Ok(Self {
            layers: [
                GpuLinear::new_seeded(context, input, hidden, seed)?,
                GpuLinear::new_seeded(context, hidden, hidden, seed.wrapping_add(1))?,
                GpuLinear::new_seeded(context, hidden, output, seed.wrapping_add(2))?,
            ],
            actor,
        })
    }
    fn snapshot(&self, trainable: bool) -> Self {
        Self {
            layers: std::array::from_fn(|i| {
                let l = &self.layers[i];
                if trainable {
                    l.trainable_snapshot()
                } else {
                    l.frozen_snapshot()
                }
            }),
            actor: self.actor,
        }
    }
    fn checked_forward(&self, input: &GpuVariable) -> Result<GpuVariable> {
        let mut value = input.clone();
        let mut logits = Vec::with_capacity(3);
        for (i, layer) in self.layers.iter().enumerate() {
            value = layer.forward(&value)?;
            // Retain pre-activation snapshots so nonlinearities cannot hide invalid logits.
            logits.push(value.clone());
            if i < 2 {
                value = value.relu()?;
            }
        }
        finite(logits)?;
        if self.actor {
            value = value.tanh()?;
        }
        Ok(value)
    }
}
impl GpuModule for GpuTd3Net {
    fn forward(&self, input: &GpuVariable) -> std::result::Result<GpuVariable, GpuModuleError> {
        let first = self.layers[0].forward(input)?.relu()?;
        let second = self.layers[1].forward(&first)?.relu()?;
        let last = self.layers[2].forward(&second)?;
        Ok(if self.actor { last.tanh()? } else { last })
    }
    fn parameters(&self) -> Vec<GpuVariable> {
        self.layers.iter().flat_map(GpuModule::parameters).collect()
    }
}

/// GPU TD3 with caller-owned noise streams and separate successful update clocks.
/// Networks, Adam moments, gradients and target synchronization stay on device.
/// Batches upload only active rows. Validation reads back scalar diagnostics/losses;
/// action selection downloads the final physical action vector.
/// A failed batch preserves all live parameters, targets, optimizer state and clocks.
/// Caller RNG draws are not rolled back on a device/arithmetic failure.
pub struct GpuTd3 {
    context: GpuContext,
    config: TD3Config,
    transform: GpuTd3ActionTransform,
    actor: GpuTd3Net,
    critic1: GpuTd3Net,
    critic2: GpuTd3Net,
    actor_target: GpuTd3Net,
    critic1_target: GpuTd3Net,
    critic2_target: GpuTd3Net,
    actor_optimizer: GpuAdam,
    critic_optimizer: GpuAdam,
    updates: usize,
    actor_updates: usize,
}
impl GpuTd3 {
    pub fn validate_config(config: &TD3Config) -> Result<()> {
        affine_bounds(&config.action_low, &config.action_high)?;
        crate::agent::td3::validate_td3_config(config).map_err(|_| GpuTd3Error::InvalidConfig)
    }
    pub fn new_seeded(context: &GpuContext, config: TD3Config, seed: u64) -> Result<Self> {
        Self::validate_config(&config)?;
        let transform =
            GpuTd3ActionTransform::new(context, &config.action_low, &config.action_high)?;
        let actor = GpuTd3Net::new(
            context,
            config.obs_dim,
            config.hidden_dim,
            config.act_dim,
            seed,
            true,
        )?;
        let input = config.obs_dim + config.act_dim;
        let critic1 = GpuTd3Net::new(
            context,
            input,
            config.hidden_dim,
            1,
            seed.wrapping_add(3),
            false,
        )?;
        let critic2 = GpuTd3Net::new(
            context,
            input,
            config.hidden_dim,
            1,
            seed.wrapping_add(6),
            false,
        )?;
        let actor_optimizer = GpuAdam::new(actor.parameters(), config.actor_lr)?;
        let critic_optimizer =
            GpuAdam::new(critic_parameters(&critic1, &critic2), config.critic_lr)?;
        Ok(Self {
            context: context.clone(),
            config,
            transform,
            actor_target: actor.snapshot(false),
            critic1_target: critic1.snapshot(false),
            critic2_target: critic2.snapshot(false),
            actor,
            critic1,
            critic2,
            actor_optimizer,
            critic_optimizer,
            updates: 0,
            actor_updates: 0,
        })
    }
    pub fn config(&self) -> &TD3Config {
        &self.config
    }
    pub fn actor(&self) -> &GpuTd3Net {
        &self.actor
    }
    pub fn critic1(&self) -> &GpuTd3Net {
        &self.critic1
    }
    pub fn critic2(&self) -> &GpuTd3Net {
        &self.critic2
    }
    pub fn actor_target(&self) -> &GpuTd3Net {
        &self.actor_target
    }
    pub fn critic1_target(&self) -> &GpuTd3Net {
        &self.critic1_target
    }
    pub fn critic2_target(&self) -> &GpuTd3Net {
        &self.critic2_target
    }
    pub fn updates(&self) -> usize {
        self.updates
    }
    pub fn actor_updates(&self) -> usize {
        self.actor_updates
    }

    pub fn select_action(&self, state: &[f32], noise_std: f32) -> Result<Vec<f32>> {
        self.select_action_with_rng(state, noise_std, &mut rand::thread_rng())
    }
    /// Exploration noise is in physical action units; zero deviation draws nothing.
    pub fn select_action_with_rng(
        &self,
        state: &[f32],
        noise_std: f32,
        rng: &mut impl Rng,
    ) -> Result<Vec<f32>> {
        if state.len() != self.config.obs_dim
            || state.iter().any(|v| !v.is_finite())
            || !noise_std.is_finite()
            || noise_std < 0.
        {
            return Err(GpuTd3Error::InvalidInput(
                "finite observation and nonnegative finite exploration deviation required",
            ));
        }
        let actions = no_grad(|| -> Result<_> {
            let input = GpuVariable::new(
                &self.context,
                &Tensor::from_vec(state.to_vec(), &[1, state.len()]),
                false,
            )?;
            let raw = self.actor.checked_forward(&input)?;
            let output = self.transform.scale_actor_actions(&raw)?;
            Ok(output.checked_to_cpu()?.to_vec())
        })?;
        actions
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let noise = if noise_std > 0. {
                    gaussian(rng, noise_std)
                } else {
                    0.
                };
                let value = v + noise;
                if !value.is_finite() {
                    return Err(GpuTd3Error::NonFinite);
                }
                Ok(value.clamp(self.config.action_low[i], self.config.action_high[i]))
            })
            .collect()
    }
    pub fn train_step(&mut self, batch: &ContinuousTransitionBatch) -> Result<(f32, Option<f32>)> {
        self.train_step_with_rng(batch, &mut rand::thread_rng())
    }
    /// Target noise consumes two uniforms per active action, even with zero deviation.
    pub fn train_step_with_rng(
        &mut self,
        batch: &ContinuousTransitionBatch,
        rng: &mut impl Rng,
    ) -> Result<(f32, Option<f32>)> {
        self.validate_batch(batch)?;
        if batch.size == 0 {
            return Ok((0., None));
        }
        let values = (0..batch.size * self.config.act_dim)
            .map(|_| gaussian(rng, self.config.target_noise_std))
            .collect();
        let noise = Tensor::from_vec(values, &[batch.size, self.config.act_dim]);
        self.train_step_with_target_noise(batch, &noise)
    }
    fn validate_batch(&self, batch: &ContinuousTransitionBatch) -> Result<()> {
        if batch.size == 0 {
            return Ok(());
        }
        if !rustforge_autograd::is_grad_enabled() {
            return Err(GpuTd3Error::InvalidInput(
                "TD3 training requires gradients enabled",
            ));
        }
        for (tensor, columns) in [
            (&batch.states, self.config.obs_dim),
            (&batch.next_states, self.config.obs_dim),
            (&batch.actions, self.config.act_dim),
            (&batch.rewards, 1),
            (&batch.dones, 1),
        ] {
            validate_rows(tensor, batch.size, columns)?;
        }
        if batch
            .dones
            .data()
            .iter()
            .take(batch.size)
            .any(|v| !(0. ..=1.).contains(v))
        {
            return Err(GpuTd3Error::InvalidBatch);
        }
        Ok(())
    }
    /// Supplied noise is pre-scaled in normalized action units, then clipped here.
    /// Extra capacity rows (including stale nonfinite values) are ignored.
    pub fn train_step_with_target_noise(
        &mut self,
        batch: &ContinuousTransitionBatch,
        noise: &Tensor,
    ) -> Result<(f32, Option<f32>)> {
        self.validate_batch(batch)?;
        if batch.size == 0 {
            return Ok((0., None));
        }
        validate_rows(noise, batch.size, self.config.act_dim)?;
        let updates = self
            .updates
            .checked_add(1)
            .ok_or(GpuTd3Error::InvalidConfig)?;
        let scheduled = self.config.policy_delay != 0 && updates % self.config.policy_delay == 0;
        let actor_updates = if scheduled {
            self.actor_updates
                .checked_add(1)
                .ok_or(GpuTd3Error::InvalidConfig)?
        } else {
            self.actor_updates
        };
        let batch = batch.valid_rows();
        let upload =
            |t: &Tensor| GpuVariable::new(&self.context, t, false).map_err(GpuTd3Error::from);
        let states = upload(&batch.states)?;
        let actions = upload(&batch.actions)?;
        let next_states = upload(&batch.next_states)?;
        let rewards = upload(&batch.rewards)?;
        let dones = upload(&batch.dones)?;
        let noise = upload(
            &noise
                .slice_axis(0, 0, batch.size)
                .map_err(|_| GpuTd3Error::InvalidBatch)?,
        )?;
        let (t1, t2) = no_grad(|| -> Result<_> {
            let raw = self.actor_target.checked_forward(&next_states)?;
            let smoothed = self.transform.smooth_target_actions(
                &raw,
                &noise,
                GpuTd3SmoothingConfig {
                    noise_std: 1.,
                    noise_clip: self.config.target_noise_clip,
                },
            )?;
            smoothed.checked()?;
            let input = next_states.concat_columns(&smoothed.actions)?;
            Ok((
                self.critic1_target.checked_forward(&input)?,
                self.critic2_target.checked_forward(&input)?,
            ))
        })?;
        // Independent leaves and shared immutable moment snapshots make the update transactional.
        let critic1 = self.critic1.snapshot(true);
        let critic2 = self.critic2.snapshot(true);
        let critic_params = critic_parameters(&critic1, &critic2);
        let mut critic_optimizer = self.critic_optimizer.fork(critic_params.clone())?;
        let input = states.concat_columns(&actions)?;
        let loss = td3_critic_loss(
            &critic1.checked_forward(&input)?,
            &critic2.checked_forward(&input)?,
            &t1,
            &t2,
            &rewards,
            &dones,
            GpuTd3LossConfig {
                gamma: self.config.gamma,
            },
        )?;
        let metrics = loss.checked_metrics()?;
        loss.total_loss.backward()?;
        validate_gradients(&critic_params)?;
        critic_optimizer.step()?;
        finite(critic_params)?;
        let actor = self.actor.snapshot(true);
        let mut actor_optimizer = self.actor_optimizer.fork(actor.parameters())?;
        let mut targets = None;
        let actor_loss = if scheduled {
            let raw = actor.checked_forward(&states)?;
            let scaled = self.transform.scale_actor_actions(&raw)?;
            scaled.checked()?;
            // Frozen updated Q1 still differentiates with respect to actor actions.
            let input = states.concat_columns(&scaled.actions)?;
            let q = critic1.snapshot(false).checked_forward(&input)?;
            let loss = td3_actor_loss(&q)?;
            let metric = loss.checked_loss()?;
            loss.loss.backward()?;
            validate_gradients(&actor.parameters())?;
            actor_optimizer.step()?;
            finite(actor.parameters())?;
            targets = Some([
                polyak(&self.context, &actor, &self.actor_target, self.config.tau)?,
                polyak(
                    &self.context,
                    &critic1,
                    &self.critic1_target,
                    self.config.tau,
                )?,
                polyak(
                    &self.context,
                    &critic2,
                    &self.critic2_target,
                    self.config.tau,
                )?,
            ]);
            Some(metric)
        } else {
            None
        };
        // All fallible computation/validation finishes before copying live leaf handles.
        let new_critic_optimizer =
            critic_optimizer.fork(critic_parameters(&self.critic1, &self.critic2))?;
        let new_actor_optimizer = actor_optimizer.fork(self.actor.parameters())?;
        copy_parameters(&self.critic1, &critic1)?;
        copy_parameters(&self.critic2, &critic2)?;
        if scheduled {
            copy_parameters(&self.actor, &actor)?;
        }
        if let Some([a, c1, c2]) = targets {
            copy_parameters(&self.actor_target, &a)?;
            copy_parameters(&self.critic1_target, &c1)?;
            copy_parameters(&self.critic2_target, &c2)?;
        }
        self.actor_optimizer = new_actor_optimizer;
        self.critic_optimizer = new_critic_optimizer;
        self.updates = updates;
        self.actor_updates = actor_updates;
        self.actor_optimizer.zero_grad();
        self.critic_optimizer.zero_grad();
        Ok((metrics.total_loss, actor_loss))
    }
}
fn gaussian(rng: &mut impl Rng, std: f32) -> f32 {
    let u1: f32 = rng.gen_range(1e-7..1.0);
    let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    std * (-2. * u1.ln()).sqrt() * u2.cos()
}
fn validate_rows(tensor: &Tensor, rows: usize, columns: usize) -> Result<()> {
    let shape = tensor.shape();
    if shape.len() != 2 || shape[0] < rows || shape[1] != columns {
        return Err(GpuTd3Error::InvalidBatch);
    }
    if tensor
        .data()
        .iter()
        .take(rows * columns)
        .any(|v| !v.is_finite())
    {
        return Err(GpuTd3Error::NonFinite);
    }
    Ok(())
}
fn critic_parameters(first: &GpuTd3Net, second: &GpuTd3Net) -> Vec<GpuVariable> {
    let mut p = first.parameters();
    p.extend(second.parameters());
    p
}
fn validate_gradients(parameters: &[GpuVariable]) -> Result<()> {
    let mut values = Vec::with_capacity(parameters.len() * 2);
    for p in parameters {
        let g = p.grad().ok_or(GpuTd3Error::InvalidInput(
            "TD3 parameter gradient is missing",
        ))?;
        let square = p.context().mul_device(&g, &g)?;
        values.extend([g, std::rc::Rc::new(square)]);
    }
    if let Some(first) = parameters.first() {
        finite_data(first.context(), values)?;
    }
    Ok(())
}
fn copy_parameters(destination: &GpuTd3Net, source: &GpuTd3Net) -> Result<()> {
    for (d, s) in destination.parameters().iter().zip(source.parameters()) {
        d.copy_data_from(&s)?;
    }
    Ok(())
}
fn polyak(
    context: &GpuContext,
    online: &GpuTd3Net,
    target: &GpuTd3Net,
    tau: f32,
) -> Result<GpuTd3Net> {
    let output = target.snapshot(false);
    for ((out, live), old) in output
        .parameters()
        .iter()
        .zip(online.parameters())
        .zip(target.parameters())
    {
        let data = context.add_device(
            &context.scale_device(&live.data(), tau)?,
            &context.scale_device(&old.data(), 1. - tau)?,
        )?;
        out.copy_data_from(&GpuVariable::from_device(context, data, false)?)?;
    }
    finite(output.parameters())?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_config_validates_dimensions_rates_bounds_and_noise_without_adapter() {
        let c = TD3Config::new(2, 1, vec![-1.], vec![1.]);
        GpuTd3::validate_config(&c).unwrap();
        for kind in 0..8 {
            let mut bad = c.clone();
            match kind {
                0 => bad.obs_dim = 0,
                1 => bad.hidden_dim = 0,
                2 => bad.act_dim = 2,
                3 => bad.tau = 1.1,
                4 => bad.actor_lr = f32::INFINITY,
                5 => bad.critic_lr = 0.,
                6 => bad.obs_dim = usize::MAX,
                _ => bad.target_noise_std = -1.,
            }
            assert!(GpuTd3::validate_config(&bad).is_err());
        }
        let mut delay_zero = c;
        delay_zero.policy_delay = 0;
        GpuTd3::validate_config(&delay_zero).unwrap();
    }
}
