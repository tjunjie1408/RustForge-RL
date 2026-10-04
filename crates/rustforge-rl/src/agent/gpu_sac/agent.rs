//! Transactional GPU SAC with caller-owned noise and replay sampling streams.
use super::*;
use crate::{
    agent::{gaussian_policy::sample_standard_normal, sac::SACConfig},
    buffer::ContinuousTransitionBatch,
};
use rand::Rng;
use rustforge_autograd::{gpu::GpuAdam, no_grad};
use rustforge_nn::gpu::{GpuLinear, GpuModule, GpuModuleError};
use rustforge_tensor::Tensor;
/// SAC Q network: state/action input and two hidden ReLU layers.
pub struct GpuSacCritic {
    layers: [GpuLinear; 3],
}
impl GpuSacCritic {
    fn new(
        context: &GpuContext,
        input: usize,
        hidden: usize,
        output: usize,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            layers: [
                GpuLinear::new_seeded(context, input, hidden, seed)?,
                GpuLinear::new_seeded(context, hidden, hidden, seed.wrapping_add(1))?,
                GpuLinear::new_seeded(context, hidden, output, seed.wrapping_add(2))?,
            ],
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
        finite(input.context(), logits)?;
        Ok(value)
    }
}
impl GpuModule for GpuSacCritic {
    fn forward(&self, input: &GpuVariable) -> std::result::Result<GpuVariable, GpuModuleError> {
        let first = self.layers[0].forward(input)?.relu()?;
        let second = self.layers[1].forward(&first)?.relu()?;
        let last = self.layers[2].forward(&second)?;
        Ok(last)
    }
    fn parameters(&self) -> Vec<GpuVariable> {
        self.layers.iter().flat_map(GpuModule::parameters).collect()
    }
}

/// Gaussian policy with the CPU parameter and seed order; raw std is clipped by the transform.
pub struct GpuSacPolicy {
    layers: [GpuLinear; 4],
}
impl GpuSacPolicy {
    fn new(context: &GpuContext, c: &SACConfig, seed: u64) -> Result<Self> {
        Ok(Self {
            layers: [
                GpuLinear::new_seeded(context, c.obs_dim, c.hidden_dim, seed)?,
                GpuLinear::new_seeded(context, c.hidden_dim, c.hidden_dim, seed.wrapping_add(1))?,
                GpuLinear::new_seeded(context, c.hidden_dim, c.act_dim, seed.wrapping_add(2))?,
                GpuLinear::new_seeded(context, c.hidden_dim, c.act_dim, seed.wrapping_add(3))?,
            ],
        })
    }
    fn snapshot(&self) -> Self {
        Self {
            layers: std::array::from_fn(|i| self.layers[i].trainable_snapshot()),
        }
    }
    pub fn parameters(&self) -> Vec<GpuVariable> {
        self.layers.iter().flat_map(GpuModule::parameters).collect()
    }
    pub fn forward_raw(&self, states: &GpuVariable) -> Result<(GpuVariable, GpuVariable)> {
        let first = self.layers[0].forward(states)?;
        let second = self.layers[1].forward(&first.relu()?)?;
        let features = second.relu()?;
        let mean = self.layers[2].forward(&features)?;
        let std = self.layers[3].forward(&features)?;
        finite(states.context(), [first, second, mean.clone(), std.clone()])?;
        Ok((mean, std))
    }
}

/// Actor, twin critics/targets and three resident Adam states.
/// Every nonempty successful update advances all optimizers and both targets.
/// Failure preserves live state and parameter handles. Caller RNG draws are not rolled back.
/// Only active replay rows upload; actions and scalar diagnostics download.
pub struct GpuSac {
    context: GpuContext,
    config: SACConfig,
    transform: GpuSacActionTransform,
    actor: GpuSacPolicy,
    critic1: GpuSacCritic,
    critic2: GpuSacCritic,
    critic1_target: GpuSacCritic,
    critic2_target: GpuSacCritic,
    log_alpha: GpuVariable,
    actor_optimizer: GpuAdam,
    critic_optimizer: GpuAdam,
    alpha_optimizer: GpuAdam,
    updates: usize,
}
impl GpuSac {
    pub fn validate_config(c: &SACConfig) -> Result<()> {
        crate::agent::sac::validate_sac_config(c).map_err(|_| GpuSacError::InvalidConfig)
    }
    pub fn new_seeded(context: &GpuContext, config: SACConfig, seed: u64) -> Result<Self> {
        Self::validate_config(&config)?;
        let actor = GpuSacPolicy::new(context, &config, seed)?;
        let critic1 = GpuSacCritic::new(
            context,
            config.obs_dim + config.act_dim,
            config.hidden_dim,
            1,
            seed.wrapping_add(4),
        )?;
        let critic2 = GpuSacCritic::new(
            context,
            config.obs_dim + config.act_dim,
            config.hidden_dim,
            1,
            seed.wrapping_add(7),
        )?;
        let log_alpha = GpuVariable::new(
            context,
            &Tensor::from_vec(vec![config.init_alpha.ln()], &[1]),
            true,
        )?;
        checked_alpha(&log_alpha)?;
        Ok(Self {
            context: context.clone(),
            transform: GpuSacActionTransform::new(
                context,
                &config.action_low,
                &config.action_high,
            )?,
            actor_optimizer: GpuAdam::new(actor.parameters(), config.actor_lr)?,
            critic_optimizer: GpuAdam::new(
                critic_parameters(&critic1, &critic2),
                config.critic_lr,
            )?,
            alpha_optimizer: GpuAdam::new(vec![log_alpha.clone()], config.alpha_lr)?,
            critic1_target: critic1.snapshot(false),
            critic2_target: critic2.snapshot(false),
            config,
            actor,
            critic1,
            critic2,
            log_alpha,
            updates: 0,
        })
    }
    pub fn config(&self) -> &SACConfig {
        &self.config
    }
    pub fn actor(&self) -> &GpuSacPolicy {
        &self.actor
    }
    pub fn critic1(&self) -> &GpuSacCritic {
        &self.critic1
    }
    pub fn critic2(&self) -> &GpuSacCritic {
        &self.critic2
    }
    pub fn critic1_target(&self) -> &GpuSacCritic {
        &self.critic1_target
    }
    pub fn critic2_target(&self) -> &GpuSacCritic {
        &self.critic2_target
    }
    pub fn log_alpha(&self) -> &GpuVariable {
        &self.log_alpha
    }
    pub fn updates(&self) -> usize {
        self.updates
    }
    pub fn alpha(&self) -> Result<f32> {
        checked_alpha(&self.log_alpha)
    }
    fn state(&self, state: &[f32]) -> Result<GpuVariable> {
        if state.len() != self.config.obs_dim || state.iter().any(|v| !v.is_finite()) {
            return Err(GpuSacError::InvalidBatch);
        }
        Ok(GpuVariable::new(
            &self.context,
            &Tensor::from_vec(state.to_vec(), &[1, self.config.obs_dim]),
            false,
        )?)
    }
    pub fn select_action(&self, state: &[f32]) -> Result<Vec<f32>> {
        self.select_action_with_rng(state, &mut rand::thread_rng())
    }
    pub fn select_action_with_rng(&self, state: &[f32], rng: &mut impl Rng) -> Result<Vec<f32>> {
        let state = self.state(state)?;
        let noise = noise(1, self.config.act_dim, rng);
        no_grad(|| -> Result<_> {
            let noise = GpuVariable::new(&self.context, &noise, false)?;
            let (mean, std) = self.actor.forward_raw(&state)?;
            let sample = self.transform.sample_with_noise(&mean, &std, &noise)?;
            sample.checked_metrics()?;
            Ok(sample.actions.to_cpu()?.to_vec())
        })
    }
    pub fn deterministic_action(&self, state: &[f32]) -> Result<Vec<f32>> {
        let state = self.state(state)?;
        no_grad(|| -> Result<_> {
            let (mean, std) = self.actor.forward_raw(&state)?;
            let zeros = GpuVariable::from_device(
                &self.context,
                self.context.full(mean.data().shape(), 0.)?,
                false,
            )?;
            let sample = self.transform.sample_with_noise(&mean, &std, &zeros)?;
            sample.checked_metrics()?;
            Ok(sample.actions.to_cpu()?.to_vec())
        })
    }
    fn validate_batch(&self, b: &ContinuousTransitionBatch) -> Result<()> {
        if b.size == 0 {
            return Ok(());
        }
        if !rustforge_autograd::is_grad_enabled() {
            return Err(GpuSacError::InvalidBatch);
        }
        for (t, cols) in [
            (&b.states, self.config.obs_dim),
            (&b.next_states, self.config.obs_dim),
            (&b.actions, self.config.act_dim),
            (&b.rewards, 1),
            (&b.dones, 1),
        ] {
            validate_rows(t, b.size, cols)?;
        }
        if b.dones
            .data()
            .iter()
            .take(b.size)
            .any(|v| !(0. ..=1.).contains(v))
        {
            return Err(GpuSacError::InvalidBatch);
        }
        Ok(())
    }
    pub fn train_step(&mut self, b: &ContinuousTransitionBatch) -> Result<(f32, f32, f32, f32)> {
        self.train_step_with_rngs(b, &mut rand::thread_rng(), &mut rand::thread_rng())
    }
    /// Streams are separate from collection and replay RNGs; invalid host batches draw nothing.
    pub fn train_step_with_rngs(
        &mut self,
        b: &ContinuousTransitionBatch,
        target_rng: &mut impl Rng,
        actor_rng: &mut impl Rng,
    ) -> Result<(f32, f32, f32, f32)> {
        self.validate_batch(b)?;
        if b.size == 0 {
            return Ok((0., 0., 0., self.alpha()?));
        }
        self.train_step_with_noise(
            b,
            &noise(b.size, self.config.act_dim, target_rng),
            &noise(b.size, self.config.act_dim, actor_rng),
        )
    }
    pub fn train_step_with_noise(
        &mut self,
        b: &ContinuousTransitionBatch,
        target_noise: &Tensor,
        actor_noise: &Tensor,
    ) -> Result<(f32, f32, f32, f32)> {
        self.validate_batch(b)?;
        if b.size == 0 {
            return Ok((0., 0., 0., self.alpha()?));
        }
        validate_rows(target_noise, b.size, self.config.act_dim)?;
        validate_rows(actor_noise, b.size, self.config.act_dim)?;
        let updates = self
            .updates
            .checked_add(1)
            .ok_or(GpuSacError::InvalidConfig)?;
        let alpha = self.alpha()?;
        let b = b.valid_rows();
        let upload = |t: &Tensor| -> Result<_> { Ok(GpuVariable::new(&self.context, t, false)?) };
        let states = upload(&b.states)?;
        let next = upload(&b.next_states)?;
        let actions = upload(&b.actions)?;
        let rewards = upload(&b.rewards)?;
        let dones = upload(&b.dones)?;
        let target_noise = upload(
            &target_noise
                .slice_axis(0, 0, b.size)
                .map_err(|_| GpuSacError::InvalidBatch)?,
        )?;
        let actor_noise = upload(
            &actor_noise
                .slice_axis(0, 0, b.size)
                .map_err(|_| GpuSacError::InvalidBatch)?,
        )?;
        let (target1, target2, next_lp) = no_grad(|| -> Result<_> {
            let (mean, std) = self.actor.forward_raw(&next)?;
            let sample = self
                .transform
                .sample_with_noise(&mean, &std, &target_noise)?;
            sample.checked_metrics()?;
            let sa = next.concat_columns(&sample.actions)?;
            Ok((
                self.critic1_target.checked_forward(&sa)?,
                self.critic2_target.checked_forward(&sa)?,
                sample.distribution.log_probs,
            ))
        })?;
        let critic1 = self.critic1.snapshot(true);
        let critic2 = self.critic2.snapshot(true);
        let cp = critic_parameters(&critic1, &critic2);
        let mut co = self.critic_optimizer.fork(cp.clone())?;
        let sa = states.concat_columns(&actions)?;
        let q1 = critic1.checked_forward(&sa)?;
        let q2 = critic2.checked_forward(&sa)?;
        let loss = sac_critic_loss(
            GpuSacCriticInputs {
                q1: &q1,
                q2: &q2,
                target_q1: &target1,
                target_q2: &target2,
                next_log_probs: &next_lp,
                rewards: &rewards,
                dones: &dones,
            },
            GpuSacLossConfig {
                gamma: self.config.gamma,
                alpha,
            },
        )?;
        let cm = loss.checked_metrics()?.total_loss;
        loss.total_loss.backward()?;
        validate_gradients(&cp)?;
        co.step()?;
        finite(&self.context, cp)?;
        let actor = self.actor.snapshot();
        let ap = actor.parameters();
        let mut ao = self.actor_optimizer.fork(ap.clone())?;
        let (mean, std) = actor.forward_raw(&states)?;
        let sample = self
            .transform
            .sample_with_noise(&mean, &std, &actor_noise)?;
        sample.checked_metrics()?;
        let sa = states.concat_columns(&sample.actions)?;
        let q1 = critic1.snapshot(false).checked_forward(&sa)?;
        let q2 = critic2.snapshot(false).checked_forward(&sa)?;
        let loss = sac_actor_loss(&q1, &q2, &sample.distribution.log_probs, alpha)?;
        let am = loss.checked_loss()?;
        loss.loss.backward()?;
        validate_gradients(&ap)?;
        ao.step()?;
        finite(&self.context, ap)?;
        let log_alpha = self.log_alpha.leaf_snapshot(true);
        let mut to = self.alpha_optimizer.fork(vec![log_alpha.clone()])?;
        let loss = sac_temperature_loss(
            &log_alpha,
            &sample.distribution.log_probs,
            -(self.config.act_dim as f32),
        )?;
        let tm = loss.checked_metrics()?.loss;
        loss.loss.backward()?;
        validate_gradients(std::slice::from_ref(&log_alpha))?;
        to.step()?;
        let alpha = checked_alpha(&log_alpha)?;
        let target1 = polyak(
            &self.context,
            &critic1,
            &self.critic1_target,
            self.config.tau,
        )?;
        let target2 = polyak(
            &self.context,
            &critic2,
            &self.critic2_target,
            self.config.tau,
        )?;
        // Prepare optimizer bindings before committing compatible live leaf handles.
        let ao = ao.fork(self.actor.parameters())?;
        let co = co.fork(critic_parameters(&self.critic1, &self.critic2))?;
        let to = to.fork(vec![self.log_alpha.clone()])?;
        for (d, s) in self.actor.parameters().iter().zip(actor.parameters()) {
            d.copy_data_from(&s)?;
        }
        copy_parameters(&self.critic1, &critic1)?;
        copy_parameters(&self.critic2, &critic2)?;
        copy_parameters(&self.critic1_target, &target1)?;
        copy_parameters(&self.critic2_target, &target2)?;
        self.log_alpha.copy_data_from(&log_alpha)?;
        self.actor_optimizer = ao;
        self.critic_optimizer = co;
        self.alpha_optimizer = to;
        self.actor_optimizer.zero_grad();
        self.critic_optimizer.zero_grad();
        self.alpha_optimizer.zero_grad();
        self.updates = updates;
        Ok((cm, am, tm, alpha))
    }
}
fn noise(rows: usize, columns: usize, rng: &mut impl Rng) -> Tensor {
    Tensor::from_vec(
        (0..rows * columns)
            .map(|_| sample_standard_normal(rng))
            .collect(),
        &[rows, columns],
    )
}
fn checked_alpha(log_alpha: &GpuVariable) -> Result<f32> {
    finite(log_alpha.context(), [log_alpha.clone()])?;
    let alpha = log_alpha.detach().exp()?.to_cpu()?.item();
    if !alpha.is_finite() || alpha <= 0. {
        return Err(GpuSacError::InvalidTemperature);
    }
    Ok(alpha)
}
fn validate_rows(tensor: &Tensor, rows: usize, columns: usize) -> Result<()> {
    let shape = tensor.shape();
    if shape.len() != 2 || shape[0] < rows || shape[1] != columns {
        return Err(GpuSacError::InvalidBatch);
    }
    if tensor
        .data()
        .iter()
        .take(rows * columns)
        .any(|v| !v.is_finite())
    {
        return Err(GpuSacError::NonFinite);
    }
    Ok(())
}
fn critic_parameters(first: &GpuSacCritic, second: &GpuSacCritic) -> Vec<GpuVariable> {
    let mut p = first.parameters();
    p.extend(second.parameters());
    p
}
fn validate_gradients(parameters: &[GpuVariable]) -> Result<()> {
    let mut values = Vec::with_capacity(parameters.len() * 2);
    for p in parameters {
        let g = p.grad().ok_or(GpuSacError::InvalidBatch)?;
        let square = p.context().mul_device(&g, &g)?;
        values.extend([g, std::rc::Rc::new(square)]);
    }
    if let Some(first) = parameters.first() {
        let variables = values
            .into_iter()
            .map(|data| {
                GpuVariable::from_device(
                    first.context(),
                    first.context().scale_device(&data, 1.)?,
                    false,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        finite(first.context(), variables)?;
    }
    Ok(())
}
fn copy_parameters(destination: &GpuSacCritic, source: &GpuSacCritic) -> Result<()> {
    for (d, s) in destination.parameters().iter().zip(source.parameters()) {
        d.copy_data_from(&s)?;
    }
    Ok(())
}
fn polyak(
    context: &GpuContext,
    online: &GpuSacCritic,
    target: &GpuSacCritic,
    tau: f32,
) -> Result<GpuSacCritic> {
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
    finite(context, output.parameters())?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_guards_dimensions_rates_temperature_bounds_and_polyak() {
        let c = SACConfig::new(2, 1, vec![-1.], vec![1.]);
        GpuSac::validate_config(&c).unwrap();
        for kind in 0..12 {
            let mut bad = c.clone();
            match kind {
                0 => bad.obs_dim = 0,
                1 => bad.hidden_dim = 0,
                2 => bad.act_dim = 2,
                3 => bad.tau = 1.1,
                4 => bad.actor_lr = f32::INFINITY,
                5 => bad.critic_lr = 0.,
                6 => bad.alpha_lr = -1.,
                7 => bad.init_alpha = 0.,
                8 => bad.obs_dim = usize::MAX,
                9 => bad.gamma = f32::NAN,
                10 => bad.action_low[0] = bad.action_high[0],
                _ => bad.init_alpha = f32::INFINITY,
            }
            assert!(GpuSac::validate_config(&bad).is_err());
        }
        for boundary in [0., 1.] {
            let mut c = c.clone();
            c.gamma = boundary;
            c.tau = boundary;
            GpuSac::validate_config(&c).unwrap();
        }
    }
}
