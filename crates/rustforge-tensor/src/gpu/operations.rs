//! Device elementwise kernels and hierarchical full-tensor reductions.

use super::{GpuContext, GpuError, GpuTensor};

pub(super) fn create_pipelines(
    device: &wgpu::Device,
) -> (
    wgpu::BindGroupLayout,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
) {
    let entries: Vec<_> = (0..4)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 3 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding != 2,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tensor operations"),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tensor operations"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tensor operations"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../gpu_ops.wgsl").into()),
    });
    let create = |entry_point| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
        })
    };
    (layout, create("elementwise"), create("reduce"))
}

impl GpuContext {
    /// Joins rank-two matrices along their feature axis without host transfers.
    pub fn concat_columns_device(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(left)?;
        self.ensure_owner(right)?;
        let (a, b) = (left.shape(), right.shape());
        if a.len() != 2 || b.len() != 2 || a[0] != b[0] {
            return Err(GpuError::ElementwiseShapeMismatch {
                left: a.to_vec(),
                right: b.to_vec(),
            });
        }
        let columns = a[1].checked_add(b[1]).ok_or(GpuError::LimitExceeded)?;
        let first = u32::try_from(a[1]).map_err(|_| GpuError::LimitExceeded)?;
        let second = u32::try_from(b[1]).map_err(|_| GpuError::LimitExceeded)?;
        u32::try_from(columns).map_err(|_| GpuError::LimitExceeded)?;
        let output = self.output_storage(&[a[0], columns])?;
        if !output.is_empty() {
            self.dispatch_operation_layout(
                left,
                right,
                &output,
                [output.numel() as u32, 20, 0],
                [first, second, 0, 0],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }
    /// Copies a contiguous column range from every row of a rank-two matrix.
    pub fn slice_columns_device(
        &self,
        input: &GpuTensor,
        start: usize,
        len: usize,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(GpuError::ExpectedMatrix {
                shape: shape.to_vec(),
            });
        }
        if start > shape[1] || len > shape[1] - start {
            return Err(GpuError::InvalidColumnRange {
                columns: shape[1],
                start,
                len,
            });
        }
        let stride = u32::try_from(shape[1]).map_err(|_| GpuError::LimitExceeded)?;
        let width = u32::try_from(len).map_err(|_| GpuError::LimitExceeded)?;
        let offset = u32::try_from(start).map_err(|_| GpuError::LimitExceeded)?;
        let output = self.output_storage(&[shape[0], len])?;
        if !output.is_empty() {
            self.dispatch_operation_layout(
                input,
                input,
                &output,
                [output.numel() as u32, 21, 0],
                [stride, width, offset, 0],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }
    /// Elementwise exponential, with shader floating-point semantics.
    pub fn exp_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise(input, input, 11)
    }

    /// Elementwise natural logarithm: zero gives -infinity, negatives/NaN give NaN.
    pub fn log_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise(input, input, 15)
    }
    /// Stable tanh, including saturated values, infinities and signed zero.
    pub fn tanh_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise(input, input, 16)
    }
    /// Numeric clipping for detached data (distinct from autograd clamp composition).
    pub fn clamp_device(
        &self,
        input: &GpuTensor,
        lower: f32,
        upper: f32,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(GpuError::InvalidBounds);
        }
        let bound = self.full(input.shape(), upper)?;
        let output = self.output_storage(input.shape())?;
        if !input.is_empty() {
            self.dispatch_operation(
                input,
                &bound,
                &output,
                [input.numel() as u32, 19, lower.to_bits()],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }
    /// Sum columns of a rank-two matrix into [rows,1], keeping the action axis.
    pub fn sum_columns_device(&self, matrix: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(matrix)?;
        if matrix.shape().len() != 2 {
            return Err(GpuError::ExpectedMatrix {
                shape: matrix.shape().to_vec(),
            });
        }
        let output = self.output_storage(&[matrix.shape()[0], 1])?;
        if !output.is_empty() {
            self.dispatch_operation(
                matrix,
                matrix,
                &output,
                [output.numel() as u32, 17, matrix.shape()[1] as u32],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }
    /// Repeat each [rows,1] element across a specified number of columns.
    pub fn broadcast_columns_device(
        &self,
        column: &GpuTensor,
        columns: usize,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(column)?;
        if column.shape().len() != 2 || column.shape()[1] != 1 {
            return Err(GpuError::ExpectedMatrix {
                shape: column.shape().to_vec(),
            });
        }
        let output = self.output_storage(&[column.shape()[0], columns])?;
        if !output.is_empty() {
            self.dispatch_operation(
                column,
                column,
                &output,
                [output.numel() as u32, 18, columns as u32],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Stable log-softmax over action columns of [batch, actions]. Empty batches
    /// are supported, but an empty action axis is rejected. Nonfinite logits
    /// produce NaN for the entire row. No intermediate data is downloaded.
    pub fn log_softmax_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        if input.shape().len() != 2 || input.shape()[1] == 0 {
            return Err(GpuError::InvalidCategoricalShape {
                shape: input.shape().to_vec(),
            });
        }
        let output = self.output_storage(input.shape())?;
        if !input.is_empty() {
            self.dispatch_operation(
                input,
                input,
                &output,
                [input.numel() as u32, 12, input.shape()[1] as u32],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Stable categorical probabilities, composed from log-softmax and exp.
    pub fn softmax_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.exp_device(&self.log_softmax_device(input)?)
    }

    /// Vector-Jacobian product: g - exp(log_probs) * sum(g, actions).
    pub fn log_softmax_backward_device(
        &self,
        log_probs: &GpuTensor,
        gradient: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(log_probs)?;
        self.ensure_owner(gradient)?;
        if log_probs.shape().len() != 2 || log_probs.shape()[1] == 0 {
            return Err(GpuError::InvalidCategoricalShape {
                shape: log_probs.shape().to_vec(),
            });
        }
        if log_probs.shape() != gradient.shape() {
            return Err(GpuError::ElementwiseShapeMismatch {
                left: log_probs.shape().to_vec(),
                right: gradient.shape().to_vec(),
            });
        }
        let output = self.output_storage(log_probs.shape())?;
        if !log_probs.is_empty() {
            self.dispatch_operation(
                log_probs,
                gradient,
                &output,
                [log_probs.numel() as u32, 13, log_probs.shape()[1] as u32],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Reduces nonfinite-value indicators to a scalar for explicit validation.
    /// The count can round for very large inputs; zero/nonzero remains reliable.
    pub fn nonfinite_count_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, false, true)
    }

    /// Adds identical-shape device tensors; broadcasting is not supported.
    pub fn add_device(&self, left: &GpuTensor, right: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise(left, right, 0)
    }

    /// Multiplies identical-shape device tensors element by element.
    pub fn mul_device(&self, left: &GpuTensor, right: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise(left, right, 1)
    }

    /// Applies ReLU on device. Like Tensor::relu, NaN maps to zero.
    pub fn relu_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise(input, input, 2)
    }

    /// Adds a feature-vector bias to each row of a rank-two matrix.
    /// This is explicit row broadcasting, not general binary broadcasting.
    pub fn add_bias_device(
        &self,
        matrix: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(matrix)?;
        self.ensure_owner(bias)?;
        if matrix.shape().len() != 2
            || bias.shape().len() != 1
            || matrix.shape()[1] != bias.shape()[0]
        {
            return Err(GpuError::InvalidBiasShape {
                matrix: matrix.shape().to_vec(),
                bias: bias.shape().to_vec(),
            });
        }
        let output = self.output_storage(matrix.shape())?;
        if !matrix.is_empty() {
            self.dispatch_operation(
                matrix,
                bias,
                &output,
                [matrix.numel() as u32, 7, bias.numel() as u32],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Sums matrix rows into [features]. Each lane reduces one column in row
    /// order; this correctness-oriented kernel does not parallelize long rows.
    pub fn sum_rows_device(&self, matrix: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(matrix)?;
        if matrix.shape().len() != 2 {
            return Err(GpuError::ExpectedMatrix {
                shape: matrix.shape().to_vec(),
            });
        }
        if matrix.is_empty() {
            return self.zeros(&[matrix.shape()[1]]);
        }
        let output = self.output_storage(&[matrix.shape()[1]])?;
        if !matrix.is_empty() {
            self.dispatch_operation(
                matrix,
                matrix,
                &output,
                [output.numel() as u32, 9, matrix.shape()[0] as u32],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Computes sqrt(input) + epsilon on device (used by Adam).
    /// Negative and nonfinite inputs follow shader floating-point semantics.
    pub fn sqrt_add_device(&self, input: &GpuTensor, epsilon: f32) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        let output = self.output_storage(input.shape())?;
        if !input.is_empty() {
            self.dispatch_operation(
                input,
                input,
                &output,
                [input.numel() as u32, 8, epsilon.to_bits()],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Divides identical-shape tensors element by element (used by Adam).
    pub fn div_device(&self, left: &GpuTensor, right: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise(left, right, 10)
    }

    fn elementwise(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        operation: u32,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(left)?;
        self.ensure_owner(right)?;
        if left.shape() != right.shape() {
            return Err(GpuError::ElementwiseShapeMismatch {
                left: left.shape().to_vec(),
                right: right.shape().to_vec(),
            });
        }
        let output = self.output_storage(left.shape())?;
        if !left.is_empty() {
            self.dispatch_operation(
                left,
                right,
                &output,
                [left.numel() as u32, operation, 1],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Fused a + b * factor, used for optimizer and target updates.
    pub fn scale_add_device(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        factor: f32,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_elementwise(a, b, 22, factor, 0.)
    }
    /// Fused (input * input) * factor, preserving operation order.
    pub fn square_scale_device(
        &self,
        input: &GpuTensor,
        factor: f32,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_elementwise(input, input, 23, factor, 0.)
    }
    /// Fused sqrt(input * factor) + epsilon, used by bias-corrected Adam.
    pub fn scale_sqrt_add_device(
        &self,
        input: &GpuTensor,
        factor: f32,
        epsilon: f32,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_elementwise(input, input, 24, factor, epsilon)
    }
    /// Fused (a / b) * factor, preserving division before scaling.
    pub fn div_scale_device(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        factor: f32,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_elementwise(a, b, 25, factor, 0.)
    }
    fn fused_elementwise(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        operation: u32,
        factor: f32,
        epsilon: f32,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(a)?;
        self.ensure_owner(b)?;
        if a.shape() != b.shape() {
            return Err(GpuError::ElementwiseShapeMismatch {
                left: a.shape().to_vec(),
                right: b.shape().to_vec(),
            });
        }
        let output = self.output_storage(a.shape())?;
        if !output.is_empty() {
            self.dispatch_operation_layout(
                a,
                b,
                &output,
                [a.numel() as u32, operation, factor.to_bits()],
                [epsilon.to_bits(), 0, 0, 0],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Multiplies each element by a host scalar without downloading data.
    pub fn scale_device(&self, input: &GpuTensor, scale: f32) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        let output = self.output_storage(input.shape())?;
        if !input.is_empty() {
            self.dispatch_operation(
                input,
                input,
                &output,
                [input.numel() as u32, 3, scale.to_bits()],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Applies the ReLU derivative to an identical-shape incoming gradient.
    /// The mask is one only for positive inputs, matching CPU autograd.
    pub fn relu_backward_device(
        &self,
        input: &GpuTensor,
        gradient: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.elementwise(input, gradient, 4)
    }

    /// Applies the tanh derivative to an identical-shape incoming gradient.
    /// `output` is the saved forward tanh value, preserving snapshot semantics.
    pub fn tanh_backward_device(
        &self,
        output: &GpuTensor,
        gradient: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.elementwise(output, gradient, 26)
    }

    /// Broadcasts a single-element device tensor to an arbitrary output shape.
    /// This explicit primitive is used to backpropagate full sum and mean.
    pub fn broadcast_scalar_device(
        &self,
        input: &GpuTensor,
        shape: &[usize],
        scale: f32,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        if input.numel() != 1 {
            return Err(GpuError::ExpectedScalar {
                shape: input.shape().to_vec(),
            });
        }
        let output = self.output_storage(shape)?;
        if !output.is_empty() {
            self.dispatch_operation(
                input,
                input,
                &output,
                [output.numel() as u32, 5, scale.to_bits()],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Creates a constant tensor directly on the device, without a tensor upload.
    pub fn full(&self, shape: &[usize], value: f32) -> Result<GpuTensor, GpuError> {
        let output = self.output_storage(shape)?;
        if !output.is_empty() {
            let dummy = self.zeros(&[])?;
            self.dispatch_operation(
                &dummy,
                &dummy,
                &output,
                [output.numel() as u32, 6, value.to_bits()],
                &self.inner.elementwise_pipeline,
            )?;
        }
        Ok(output)
    }

    /// Reduces all elements to a device scalar using a hierarchical sum.
    /// Empty tensors produce zero, matching Tensor::sum.
    pub fn sum_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, false, false)
    }

    /// Reduces all elements to a device scalar mean.
    /// Empty tensors produce zero, matching Tensor::mean.
    pub fn mean_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, true, false)
    }

    fn reduce(
        &self,
        input: &GpuTensor,
        mean: bool,
        nonfinite: bool,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        if input.is_empty() {
            return self.zeros(&[]);
        }
        let mut current = input;
        let mut intermediate: Option<GpuTensor> = None;
        let mut first_pass = true;
        loop {
            let groups = (current.numel() as u32).div_ceil(256);
            let shape = if groups == 1 {
                vec![]
            } else {
                vec![groups as usize]
            };
            let output = self.output_storage(&shape)?;
            let operation = if nonfinite && first_pass {
                14
            } else if mean && groups == 1 {
                4
            } else {
                3
            };
            self.dispatch_operation(
                current,
                current,
                &output,
                [current.numel() as u32, operation, input.numel() as u32],
                &self.inner.reduction_pipeline,
            )?;
            if groups == 1 {
                return Ok(output);
            }
            // Submitted commands retain the previous buffer until they finish.
            intermediate.replace(output);
            first_pass = false;
            current = intermediate
                .as_ref()
                .expect("reduction output just assigned");
        }
    }

    pub(super) fn dispatch_operation(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        output: &GpuTensor,
        parameters: [u32; 3],
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<(), GpuError> {
        self.dispatch_operation_layout(left, right, output, parameters, [0; 4], pipeline)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_operation_layout(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        output: &GpuTensor,
        parameters: [u32; 3],
        layout: [u32; 4],
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<(), GpuError> {
        let [length, operation, auxiliary] = parameters;
        let groups = length.div_ceil(256);
        let grid = dispatch_grid(
            groups,
            self.inner
                .device
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        let parameters = self.dispatch_parameters(bytemuck::cast_slice(&[
            length, operation, groups, auxiliary, layout[0], layout[1], layout[2], layout[3],
        ]));
        let buffers: [&wgpu::Buffer; 4] =
            [&left.buffer, &right.buffer, &output.buffer, &parameters];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tensor operation bindings"),
                layout: &self.inner.ops_layout,
                entries: &entries,
            });
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tensor operation"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tensor operation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        self.submit_compute(
            encoder,
            u64::from(length != 0),
            Some(parameters),
            &[left, right, output],
        );
        Ok(())
    }
}

fn dispatch_grid(groups: u32, limit: u32) -> Result<[u32; 2], GpuError> {
    if groups == 0 {
        return Ok([0, 0]);
    }
    if limit == 0 {
        return Err(GpuError::LimitExceeded);
    }
    let width = groups.min(limit);
    let height = groups.div_ceil(width);
    if height > limit || u64::from(width) * u64::from(height) > u64::from(u32::MAX) {
        return Err(GpuError::LimitExceeded);
    }
    Ok([width, height])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_dispatch_into_rows_and_rejects_overflow() {
        assert_eq!(dispatch_grid(0, 1).unwrap(), [0, 0]);
        assert!(matches!(dispatch_grid(17, 4), Err(GpuError::LimitExceeded)));
        assert_eq!(dispatch_grid(13, 4).unwrap(), [4, 4]);
        assert_eq!(dispatch_grid(65536, 65535).unwrap(), [65535, 2]);
        assert!(dispatch_grid(1, 0).is_err());
    }
}
