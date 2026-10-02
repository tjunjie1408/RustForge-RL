//! Optional reverse-mode autograd with device-resident tensors and optimizers.
//!
//! Variables retain immutable forward snapshots. Backward accumulates leaf
//! gradients; call `zero_grad` between training steps. Binary operations require
//! identical shapes, and backward requires a single-element loss.
use rustforge_tensor::{
    gpu::{GpuContext, GpuError, GpuIndices, GpuTensor},
    Tensor,
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
    rc::Rc,
};

#[derive(Debug)]
pub enum GpuAutogradError {
    Device(GpuError),
    NonScalarLoss(Vec<usize>),
    InvalidOptimizer(&'static str),
    NonLeafAssignment,
}
impl fmt::Display for GpuAutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(e) => e.fmt(f),
            Self::NonScalarLoss(shape) => {
                write!(f, "backward requires a single-element loss, got {shape:?}")
            }
            Self::NonLeafAssignment => {
                f.write_str("device data assignment requires a leaf variable")
            }
            Self::InvalidOptimizer(message) => f.write_str(message),
        }
    }
}
impl Error for GpuAutogradError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Device(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuError> for GpuAutogradError {
    fn from(e: GpuError) -> Self {
        Self::Device(e)
    }
}
type Result<T> = std::result::Result<T, GpuAutogradError>;

#[derive(Clone)]
pub struct GpuVariable {
    context: GpuContext,
    inner: Rc<RefCell<Inner>>,
}
struct Inner {
    data: Rc<GpuTensor>,
    grad: Option<Rc<GpuTensor>>,
    requires_grad: bool,
    op: Option<Op>,
}
#[derive(Clone)]
enum Op {
    Add(GpuVariable, GpuVariable),
    Bias(GpuVariable, GpuVariable),
    Gather(GpuVariable, Rc<GpuIndices>),
    Mul(GpuVariable, GpuVariable, Rc<GpuTensor>, Rc<GpuTensor>),
    Matmul(
        GpuVariable,
        GpuVariable,
        Rc<GpuTensor>,
        Rc<GpuTensor>,
        MatrixKind,
    ),
    Relu(GpuVariable, Rc<GpuTensor>),
    Scale(GpuVariable, f32),
    Reduce(GpuVariable, f32),
}
#[derive(Clone, Copy)]
enum MatrixKind {
    Normal,
    RightTranspose,
    LeftTranspose,
}
impl Op {
    fn parents(&self) -> Vec<GpuVariable> {
        match self {
            Self::Add(a, b) | Self::Bias(a, b) | Self::Mul(a, b, ..) | Self::Matmul(a, b, ..) => {
                vec![a.clone(), b.clone()]
            }
            Self::Relu(a, _) | Self::Scale(a, _) | Self::Reduce(a, _) | Self::Gather(a, _) => {
                vec![a.clone()]
            }
        }
    }
}
impl GpuVariable {
    pub fn new(context: &GpuContext, data: &Tensor, requires_grad: bool) -> Result<Self> {
        Self::from_device(context, context.upload(data)?, requires_grad)
    }
    pub fn from_device(context: &GpuContext, data: GpuTensor, requires_grad: bool) -> Result<Self> {
        if !context.is_compatible(&data) {
            return Err(GpuError::DeviceMismatch.into());
        }
        Ok(Self::build(context, Rc::new(data), requires_grad, None))
    }
    fn build(
        context: &GpuContext,
        data: Rc<GpuTensor>,
        requires_grad: bool,
        op: Option<Op>,
    ) -> Self {
        Self {
            context: context.clone(),
            inner: Rc::new(RefCell::new(Inner {
                data,
                grad: None,
                requires_grad,
                op,
            })),
        }
    }
    fn output(&self, data: GpuTensor, parents: &[&Self], op: impl FnOnce() -> Op) -> Self {
        let record = crate::is_grad_enabled() && parents.iter().any(|p| p.requires_grad());
        Self::build(&self.context, Rc::new(data), record, record.then(op))
    }
    fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }
    pub fn data(&self) -> Rc<GpuTensor> {
        self.inner.borrow().data.clone()
    }
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
    /// Assigns an immutable data snapshot to a same-shape leaf, clearing its
    /// gradient. Sharing storage is safe: optimizer steps replace leaf buffers.
    /// No graph or tensor transfer is retained by the destination.
    pub fn copy_data_from(&self, source: &Self) -> Result<()> {
        if self.has_grad_fn() {
            return Err(GpuAutogradError::NonLeafAssignment);
        }
        let data = source.data();
        if !self.context.is_compatible(&data) {
            return Err(GpuError::DeviceMismatch.into());
        }
        if self.data().shape() != data.shape() {
            return Err(GpuError::InvalidOutputShape {
                expected: self.data().shape().to_vec(),
                actual: data.shape().to_vec(),
            }
            .into());
        }
        let mut inner = self.inner.borrow_mut();
        inner.data = data;
        inner.grad = None;
        Ok(())
    }
    /// Gathers one typed action index per row; backward scatters into the
    /// original action matrix. Index storage is retained without host readback.
    pub fn gather_actions(&self, indices: &Rc<GpuIndices>) -> Result<Self> {
        Ok(self.output(
            self.context.gather_rows_device(&self.data(), indices)?,
            &[self],
            || Op::Gather(self.clone(), indices.clone()),
        ))
    }
    pub fn grad(&self) -> Option<Rc<GpuTensor>> {
        self.inner.borrow().grad.clone()
    }
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }
    pub fn has_grad_fn(&self) -> bool {
        self.inner.borrow().op.is_some()
    }
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }
    pub fn to_cpu(&self) -> Result<Tensor> {
        Ok(self.context.download(&self.data())?)
    }
    pub fn grad_cpu(&self) -> Result<Option<Tensor>> {
        self.grad()
            .map(|g| self.context.download(&g).map_err(Into::into))
            .transpose()
    }
    /// Shares an immutable snapshot without retaining its computation graph.
    pub fn detach(&self) -> Self {
        Self::build(&self.context, self.data(), false, None)
    }
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        Ok(self.output(
            self.context.add_device(&self.data(), &rhs.data())?,
            &[self, rhs],
            || Op::Add(self.clone(), rhs.clone()),
        ))
    }
    /// Explicitly broadcasts a feature-vector bias over matrix rows.
    pub fn add_bias(&self, bias: &Self) -> Result<Self> {
        Ok(self.output(
            self.context.add_bias_device(&self.data(), &bias.data())?,
            &[self, bias],
            || Op::Bias(self.clone(), bias.clone()),
        ))
    }
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        let (a, b) = (self.data(), rhs.data());
        Ok(
            self.output(self.context.mul_device(&a, &b)?, &[self, rhs], || {
                Op::Mul(self.clone(), rhs.clone(), a, b)
            }),
        )
    }
    pub fn scale(&self, factor: f32) -> Result<Self> {
        Ok(self.output(
            self.context.scale_device(&self.data(), factor)?,
            &[self],
            || Op::Scale(self.clone(), factor),
        ))
    }
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.add(&rhs.scale(-1.0)?)
    }
    pub fn relu(&self) -> Result<Self> {
        let a = self.data();
        Ok(self.output(self.context.relu_device(&a)?, &[self], || {
            Op::Relu(self.clone(), a)
        }))
    }
    pub fn sum(&self) -> Result<Self> {
        Ok(
            self.output(self.context.sum_device(&self.data())?, &[self], || {
                Op::Reduce(self.clone(), 1.0)
            }),
        )
    }
    pub fn mean(&self) -> Result<Self> {
        let n = self.data().numel();
        Ok(
            self.output(self.context.mean_device(&self.data())?, &[self], || {
                Op::Reduce(self.clone(), if n == 0 { 0.0 } else { 1.0 / n as f32 })
            }),
        )
    }
    pub fn mse_loss(&self, target: &Self) -> Result<Self> {
        let diff = self.sub(target)?;
        diff.mul(&diff)?.mean()
    }
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        self.matrix(rhs, MatrixKind::Normal)
    }
    pub fn matmul_t(&self, rhs: &Self) -> Result<Self> {
        self.matrix(rhs, MatrixKind::RightTranspose)
    }
    pub fn t_matmul(&self, rhs: &Self) -> Result<Self> {
        self.matrix(rhs, MatrixKind::LeftTranspose)
    }
    fn matrix(&self, rhs: &Self, kind: MatrixKind) -> Result<Self> {
        let (a, b) = (self.data(), rhs.data());
        let out = match kind {
            MatrixKind::Normal => self.context.matmul_device(&a, &b)?,
            MatrixKind::RightTranspose => self.context.matmul_t(&a, &b)?,
            MatrixKind::LeftTranspose => self.context.t_matmul(&a, &b)?,
        };
        Ok(self.output(out, &[self, rhs], || {
            Op::Matmul(self.clone(), rhs.clone(), a, b, kind)
        }))
    }
    /// Seeds a scalar loss with one and accumulates gradients in trainable leaves.
    /// Each call uses fresh intermediate adjoints, including for shared graphs.
    pub fn backward(&self) -> Result<()> {
        if self.data().numel() != 1 {
            return Err(GpuAutogradError::NonScalarLoss(
                self.data().shape().to_vec(),
            ));
        }
        if !self.requires_grad() {
            return Ok(());
        }
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack = vec![(self.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                order.push(node);
                continue;
            }
            if !visited.insert(node.id()) {
                continue;
            }
            stack.push((node.clone(), true));
            if let Some(op) = &node.inner.borrow().op {
                for parent in op.parents() {
                    if parent.requires_grad() {
                        stack.push((parent, false));
                    }
                }
            }
        }
        let mut adjoints = HashMap::new();
        adjoints.insert(
            self.id(),
            Rc::new(self.context.full(self.data().shape(), 1.0)?),
        );
        let mut leaves = Vec::new();
        for node in order.into_iter().rev() {
            let Some(g) = adjoints.remove(&node.id()) else {
                continue;
            };
            let op = node.inner.borrow().op.clone();
            let mut contributions = Vec::new();
            match op {
                None => {
                    leaves.push((node, g));
                    continue;
                }
                Some(Op::Add(a, b)) => {
                    contributions.push((a, g.clone()));
                    contributions.push((b, g));
                }
                Some(Op::Bias(a, b)) => {
                    if b.requires_grad() {
                        contributions.push((b, Rc::new(self.context.sum_rows_device(&g)?)));
                    }
                    contributions.push((a, g));
                }
                Some(Op::Mul(a, b, av, bv)) => {
                    if a.requires_grad() {
                        contributions.push((a, Rc::new(self.context.mul_device(&g, &bv)?)));
                    }
                    if b.requires_grad() {
                        contributions.push((b, Rc::new(self.context.mul_device(&g, &av)?)));
                    }
                }
                Some(Op::Gather(a, indices)) => {
                    contributions
                        .push((a, Rc::new(self.context.scatter_rows_device(&g, &indices)?)));
                }
                Some(Op::Scale(a, f)) => {
                    contributions.push((a, Rc::new(self.context.scale_device(&g, f)?)))
                }
                Some(Op::Relu(a, av)) => {
                    contributions.push((a, Rc::new(self.context.relu_backward_device(&av, &g)?)))
                }
                Some(Op::Reduce(a, f)) => {
                    let out = self
                        .context
                        .broadcast_scalar_device(&g, a.data().shape(), f)?;
                    contributions.push((a, Rc::new(out)));
                }
                Some(Op::Matmul(a, b, av, bv, kind)) => {
                    if a.requires_grad() {
                        let out = match kind {
                            MatrixKind::Normal => self.context.matmul_t(&g, &bv)?,
                            MatrixKind::RightTranspose => self.context.matmul_device(&g, &bv)?,
                            MatrixKind::LeftTranspose => self.context.matmul_t(&bv, &g)?,
                        };
                        contributions.push((a, Rc::new(out)));
                    }
                    if b.requires_grad() {
                        let out = match kind {
                            MatrixKind::Normal => self.context.t_matmul(&av, &g)?,
                            MatrixKind::RightTranspose => self.context.t_matmul(&g, &av)?,
                            MatrixKind::LeftTranspose => self.context.matmul_device(&av, &g)?,
                        };
                        contributions.push((b, Rc::new(out)));
                    }
                }
            }
            for (parent, gradient) in contributions {
                if !parent.requires_grad() {
                    continue;
                }
                let gradient = match adjoints.remove(&parent.id()) {
                    Some(old) => Rc::new(self.context.add_device(&old, &gradient)?),
                    None => gradient,
                };
                adjoints.insert(parent.id(), gradient);
            }
        }
        // Prepare all accumulated leaf buffers before committing any updates.
        let updates: Result<Vec<_>> = leaves
            .into_iter()
            .map(|(node, g)| {
                let g = match node.grad() {
                    Some(old) => Rc::new(self.context.add_device(&old, &g)?),
                    None => g,
                };
                Ok((node, g))
            })
            .collect();
        for (node, g) in updates? {
            node.inner.borrow_mut().grad = Some(g);
        }
        Ok(())
    }
}

/// Momentum SGD: velocity = momentum * velocity + gradient;
/// parameter = parameter - learning_rate * velocity. State stays on device.
pub struct GpuSgd {
    params: Vec<GpuVariable>,
    velocities: Vec<Option<Rc<GpuTensor>>>,
    lr: f32,
    momentum: f32,
}
impl GpuSgd {
    pub fn new(params: Vec<GpuVariable>, lr: f32, momentum: f32) -> Result<Self> {
        if !lr.is_finite() || lr <= 0.0 || !momentum.is_finite() || !(0.0..1.0).contains(&momentum)
        {
            return Err(GpuAutogradError::InvalidOptimizer(
                "learning rate must be finite and positive; momentum must be in [0, 1)",
            ));
        }
        validate_parameters(&params)?;
        let velocities = vec![None; params.len()];
        Ok(Self {
            params,
            velocities,
            lr,
            momentum,
        })
    }
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
    pub fn step(&mut self) -> Result<()> {
        let mut updates = Vec::new();
        for (index, p) in self.params.iter().enumerate() {
            let Some(g) = p.grad() else {
                continue;
            };
            let velocity = if self.momentum == 0.0 {
                g
            } else {
                match &self.velocities[index] {
                    Some(old) => Rc::new(
                        p.context
                            .add_device(&p.context.scale_device(old, self.momentum)?, &g)?,
                    ),
                    None => g,
                }
            };
            let data = p
                .context
                .add_device(&p.data(), &p.context.scale_device(&velocity, -self.lr)?)?;
            updates.push((index, Rc::new(data), velocity));
        }
        for (index, data, velocity) in updates {
            self.params[index].inner.borrow_mut().data = data;
            if self.momentum != 0.0 {
                self.velocities[index] = Some(velocity);
            }
        }
        Ok(())
    }
}

fn validate_parameters(params: &[GpuVariable]) -> Result<()> {
    let mut seen = HashSet::new();
    for p in params {
        if !p.requires_grad() || p.has_grad_fn() || !seen.insert(p.id()) {
            return Err(GpuAutogradError::InvalidOptimizer(
                "parameters must be distinct trainable leaves",
            ));
        }
        if !params[0].context.is_compatible(&p.data()) {
            return Err(GpuError::DeviceMismatch.into());
        }
    }
    Ok(())
}

/// Adam with bias correction, matching the CPU optimizer's global timestep.
/// Missing gradients leave that parameter's moments unchanged, while every
/// successful step advances the global clock. Parameters and moments stay on
/// device; updates replace buffers to preserve saved forward snapshots.
pub struct GpuAdam {
    params: Vec<GpuVariable>,
    moments: Vec<Option<AdamMoments>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
}
#[derive(Clone)]
struct AdamMoments {
    first: Rc<GpuTensor>,
    second: Rc<GpuTensor>,
}
/// Host snapshot of Adam's configuration, clock and optional per-parameter moments.
/// Parameter values and gradients are not part of optimizer state. Export and
/// restore explicitly transfer moment tensors at checkpoint boundaries.
#[derive(Debug, Clone)]
pub struct GpuAdamState {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub timestep: usize,
    pub moments: Vec<Option<GpuAdamMomentState>>,
}
#[derive(Debug, Clone)]
pub struct GpuAdamMomentState {
    pub first: Tensor,
    pub second: Tensor,
}
impl GpuAdamState {
    /// Validates a snapshot without allocating or uploading device storage.
    pub fn validate_for_shapes(&self, shapes: &[Vec<usize>]) -> Result<()> {
        validate_adam_config(self.lr, self.beta1, self.beta2, self.epsilon)?;
        if self.moments.len() != shapes.len()
            || self.timestep == usize::MAX
            || (self.timestep == 0 && self.moments.iter().any(Option::is_some))
        {
            return Err(GpuAutogradError::InvalidOptimizer(
                "invalid Adam moment count or timestep",
            ));
        }
        for (moment, shape) in self.moments.iter().zip(shapes) {
            if let Some(moment) = moment {
                if moment.first.shape() != shape || moment.second.shape() != shape {
                    return Err(GpuAutogradError::InvalidOptimizer(
                        "Adam moment shapes must match parameter shapes",
                    ));
                }
                if moment.first.data().iter().any(|v| !v.is_finite())
                    || moment
                        .second
                        .data()
                        .iter()
                        .any(|v| !v.is_finite() || *v < 0.0)
                {
                    return Err(GpuAutogradError::InvalidOptimizer(
                        "Adam moments must be finite and second moments nonnegative",
                    ));
                }
            }
        }
        Ok(())
    }
}
fn validate_adam_config(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Result<()> {
    if !lr.is_finite()
        || lr <= 0.0
        || !beta1.is_finite()
        || !(0.0..1.0).contains(&beta1)
        || !beta2.is_finite()
        || !(0.0..1.0).contains(&beta2)
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(GpuAutogradError::InvalidOptimizer(
            "Adam requires positive finite learning rate/epsilon and finite betas in [0, 1)",
        ));
    }
    Ok(())
}
impl GpuAdam {
    pub fn new(params: Vec<GpuVariable>, lr: f32) -> Result<Self> {
        Self::with_betas(params, lr, 0.9, 0.999, 1e-8)
    }
    pub fn with_betas(
        params: Vec<GpuVariable>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self> {
        validate_adam_config(lr, beta1, beta2, epsilon)?;
        validate_parameters(&params)?;
        let moments = vec![None; params.len()];
        Ok(Self {
            params,
            moments,
            lr,
            beta1,
            beta2,
            epsilon,
            t: 0,
        })
    }
    /// Downloads a validated host snapshot. Live gradients are not saved.
    pub fn state(&self) -> Result<GpuAdamState> {
        let moments = self
            .params
            .iter()
            .zip(&self.moments)
            .map(|(p, m)| {
                m.as_ref()
                    .map(|m| -> Result<_> {
                        Ok(GpuAdamMomentState {
                            first: p.context.download(&m.first)?,
                            second: p.context.download(&m.second)?,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let state = GpuAdamState {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            timestep: self.t,
            moments,
        };
        state.validate_for_shapes(
            &self
                .params
                .iter()
                .map(|p| p.data().shape().to_vec())
                .collect::<Vec<_>>(),
        )?;
        Ok(state)
    }
    /// Validates and uploads all moments before replacing any optimizer state.
    /// Successful restoration clears live gradients; failures preserve them.
    pub fn restore_state(&mut self, state: &GpuAdamState) -> Result<()> {
        state.validate_for_shapes(
            &self
                .params
                .iter()
                .map(|p| p.data().shape().to_vec())
                .collect::<Vec<_>>(),
        )?;
        let moments = self
            .params
            .iter()
            .zip(&state.moments)
            .map(|(p, m)| {
                m.as_ref()
                    .map(|m| -> Result<_> {
                        Ok(AdamMoments {
                            first: Rc::new(p.context.upload(&m.first)?),
                            second: Rc::new(p.context.upload(&m.second)?),
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        self.moments = moments;
        self.lr = state.lr;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.epsilon = state.epsilon;
        self.t = state.timestep;
        self.zero_grad();
        Ok(())
    }
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
    pub fn step(&mut self) -> Result<()> {
        let t = self
            .t
            .checked_add(1)
            .ok_or(GpuAutogradError::InvalidOptimizer("Adam timestep overflow"))?;
        let bias1 = 1.0 - self.beta1.powf(t as f32);
        let bias2 = 1.0 - self.beta2.powf(t as f32);
        let mut updates = Vec::new();
        for (index, p) in self.params.iter().enumerate() {
            let Some(g) = p.grad() else {
                continue;
            };
            let context = &p.context;
            let gm = context.scale_device(&g, 1.0 - self.beta1)?;
            let gv = context.scale_device(&context.mul_device(&g, &g)?, 1.0 - self.beta2)?;
            let (m, v) = match &self.moments[index] {
                Some(old) => (
                    context.add_device(&context.scale_device(&old.first, self.beta1)?, &gm)?,
                    context.add_device(&context.scale_device(&old.second, self.beta2)?, &gv)?,
                ),
                None => (gm, gv),
            };
            let numerator = context.scale_device(&m, 1.0 / bias1)?;
            let denominator =
                context.sqrt_add_device(&context.scale_device(&v, 1.0 / bias2)?, self.epsilon)?;
            let delta =
                context.scale_device(&context.div_device(&numerator, &denominator)?, -self.lr)?;
            let data = context.add_device(&p.data(), &delta)?;
            updates.push((index, Rc::new(data), Rc::new(m), Rc::new(v)));
        }
        for (index, data, m, v) in updates {
            self.params[index].inner.borrow_mut().data = data;
            self.moments[index] = Some(AdamMoments {
                first: m,
                second: v,
            });
        }
        self.t = t;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn graph_drops_without_cycles_and_backward_handles_deep_chains() {
        let context = GpuContext::new().unwrap();
        let leaf = GpuVariable::new(&context, &Tensor::ones(&[]), true).unwrap();
        let weak = Rc::downgrade(&leaf.inner);
        let mut loss = leaf.clone();
        for _ in 0..512 {
            loss = loss.scale(1.).unwrap();
        }
        loss.backward().unwrap();
        assert_eq!(leaf.grad_cpu().unwrap().unwrap().item(), 1.);
        drop(leaf);
        assert!(weak.upgrade().is_some());
        drop(loss);
        assert!(
            weak.upgrade().is_none(),
            "graph nodes must not retain their outputs"
        );
    }
}
