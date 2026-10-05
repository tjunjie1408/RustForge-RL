//! Optional GPU modules. All forward operations are fallible and device-resident.
//! CPU modules remain separate. This foundation supports Linear, ReLU and
//! Sequential; stochastic layers, normalization and serialization are pending.
use rustforge_autograd::gpu::{GpuAutogradError, GpuVariable};
use rustforge_tensor::{
    gpu::{GpuContext, GpuError},
    Tensor,
};
use std::{error::Error, fmt};
/// Failures from layer configuration or device computation.
#[derive(Debug)]
pub enum GpuModuleError {
    Autograd(GpuAutogradError),
    InvalidParameters(&'static str),
}
impl fmt::Display for GpuModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Autograd(e) => e.fmt(f),
            Self::InvalidParameters(message) => f.write_str(message),
        }
    }
}
impl Error for GpuModuleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Autograd(e) => Some(e),
            _ => None,
        }
    }
}
impl From<GpuAutogradError> for GpuModuleError {
    fn from(e: GpuAutogradError) -> Self {
        Self::Autograd(e)
    }
}
impl From<GpuError> for GpuModuleError {
    fn from(e: GpuError) -> Self {
        Self::Autograd(e.into())
    }
}
pub type Result<T> = std::result::Result<T, GpuModuleError>;

/// Device counterpart of the CPU Module trait for deterministic feedforward layers.
pub trait GpuModule {
    fn forward(&self, input: &GpuVariable) -> Result<GpuVariable>;
    fn parameters(&self) -> Vec<GpuVariable>;
}

/// Affine layer: input [batch, in] @ weight [out, in]^T + bias [out].
pub struct GpuLinear {
    weight: GpuVariable,
    bias: Option<GpuVariable>,
}
impl GpuLinear {
    /// Reproducible Kaiming-uniform initialization with a zero bias.
    /// Random initialization occurs on the CPU once, then is uploaded.
    pub fn new_seeded(
        context: &GpuContext,
        in_features: usize,
        out_features: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::initialize(context, in_features, out_features, seed, true)
    }
    pub fn no_bias_seeded(
        context: &GpuContext,
        in_features: usize,
        out_features: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::initialize(context, in_features, out_features, seed, false)
    }
    fn initialize(
        context: &GpuContext,
        in_features: usize,
        out_features: usize,
        seed: u64,
        bias: bool,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(GpuModuleError::InvalidParameters(
                "Linear dimensions must be positive",
            ));
        }
        // Check device storage/index limits before allocating host initialization.
        drop(context.zeros(&[out_features, in_features])?);
        let weight = Tensor::kaiming_uniform(&[out_features, in_features], Some(seed));
        let bias = bias.then(|| Tensor::zeros(&[out_features]));
        Self::from_tensors(context, &weight, bias.as_ref())
    }
    /// Uploads caller-supplied parameters as independent trainable leaves.
    /// Validates parameter shapes before either upload.
    pub fn from_tensors(
        context: &GpuContext,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Self> {
        let shape = weight.shape();
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(GpuModuleError::InvalidParameters(
                "Linear weight must have positive [out, in] dimensions",
            ));
        }
        if bias.is_some_and(|b| b.shape() != [shape[0]]) {
            return Err(GpuModuleError::InvalidParameters(
                "Linear bias must have shape [out]",
            ));
        }
        Ok(Self {
            weight: GpuVariable::new(context, weight, true)?,
            bias: bias
                .map(|b| GpuVariable::new(context, b, true))
                .transpose()?,
        })
    }
    /// Independent frozen leaf handles sharing immutable parameter snapshots.
    /// Subsequent online optimizer steps cannot change these values.
    pub fn frozen_snapshot(&self) -> Self {
        Self {
            weight: self.weight.detach(),
            bias: self.bias.as_ref().map(GpuVariable::detach),
        }
    }
    /// Independent trainable leaves sharing immutable values, with no live gradients.
    pub fn trainable_snapshot(&self) -> Self {
        Self {
            weight: self.weight.leaf_snapshot(true),
            bias: self.bias.as_ref().map(|b| b.leaf_snapshot(true)),
        }
    }
    pub fn in_features(&self) -> usize {
        self.weight.data().shape()[1]
    }
    pub fn out_features(&self) -> usize {
        self.weight.data().shape()[0]
    }
}
impl GpuModule for GpuLinear {
    fn forward(&self, input: &GpuVariable) -> Result<GpuVariable> {
        let _profile = input.context().profile_scope("linear_forward");
        let output = input.matmul_t(&self.weight)?;
        match &self.bias {
            Some(bias) => Ok(output.add_bias(bias)?),
            None => Ok(output),
        }
    }
    fn parameters(&self) -> Vec<GpuVariable> {
        let mut parameters = vec![self.weight.clone()];
        parameters.extend(self.bias.iter().cloned());
        parameters
    }
}

pub struct GpuReLU;
impl GpuModule for GpuReLU {
    fn forward(&self, input: &GpuVariable) -> Result<GpuVariable> {
        Ok(input.relu()?)
    }
    fn parameters(&self) -> Vec<GpuVariable> {
        Vec::new()
    }
}

/// Applies modules in order; an empty container returns the same variable handle.
pub struct GpuSequential {
    layers: Vec<Box<dyn GpuModule>>,
}
impl GpuSequential {
    pub fn new(layers: Vec<Box<dyn GpuModule>>) -> Self {
        Self { layers }
    }
    /// Copies all parameter snapshots after validating the whole destination.
    /// No tensor transfer or allocation is needed, and frozen leaves stay frozen.
    pub fn copy_parameters_from(&self, source: &Self) -> Result<()> {
        let destination = self.parameters();
        let source = source.parameters();
        if destination.len() != source.len() {
            return Err(GpuModuleError::InvalidParameters(
                "parameter counts must match for synchronization",
            ));
        }
        for (d, s) in destination.iter().zip(&source) {
            if d.has_grad_fn() || d.data().shape() != s.data().shape() {
                return Err(GpuModuleError::InvalidParameters(
                    "synchronization requires same-shape destination leaves",
                ));
            }
            if !d.context().is_compatible(&s.data()) {
                return Err(GpuError::DeviceMismatch.into());
            }
        }
        for (d, s) in destination.iter().zip(&source) {
            d.copy_data_from(s)?;
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}
impl GpuModule for GpuSequential {
    fn forward(&self, input: &GpuVariable) -> Result<GpuVariable> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }
    fn parameters(&self) -> Vec<GpuVariable> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
