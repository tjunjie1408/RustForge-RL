//! Device elementwise kernels and hierarchical full-tensor reductions.

use wgpu::util::DeviceExt;

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
        let output = self.zeros(matrix.shape())?;
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
        let output = self.zeros(&[matrix.shape()[1]])?;
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
        let output = self.zeros(input.shape())?;
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
        let output = self.zeros(left.shape())?;
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

    /// Multiplies each element by a host scalar without downloading data.
    pub fn scale_device(&self, input: &GpuTensor, scale: f32) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        let output = self.zeros(input.shape())?;
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
        let output = self.zeros(shape)?;
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
        let output = self.zeros(shape)?;
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
        self.reduce(input, false)
    }

    /// Reduces all elements to a device scalar mean.
    /// Empty tensors produce zero, matching Tensor::mean.
    pub fn mean_device(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, true)
    }

    fn reduce(&self, input: &GpuTensor, mean: bool) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(input)?;
        if input.is_empty() {
            return self.zeros(&[]);
        }
        let mut current = input;
        let mut intermediate: Option<GpuTensor> = None;
        loop {
            let groups = (current.numel() as u32).div_ceil(256);
            let shape = if groups == 1 {
                vec![]
            } else {
                vec![groups as usize]
            };
            let output = self.zeros(&shape)?;
            let operation = if mean && groups == 1 { 4 } else { 3 };
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
        let [length, operation, auxiliary] = parameters;
        let groups = length.div_ceil(256);
        let grid = dispatch_grid(
            groups,
            self.inner
                .device
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        let parameters = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor operation parameters"),
                contents: bytemuck::cast_slice(&[length, operation, groups, auxiliary]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let buffers = [&left.buffer, &right.buffer, &output.buffer, &parameters];
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
        self.inner.queue.submit(Some(encoder.finish()));
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
