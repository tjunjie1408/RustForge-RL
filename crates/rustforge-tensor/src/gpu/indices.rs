//! Typed u32 action buffers, resident row selection and gather/scatter kernels.
use super::{GpuContext, GpuError, GpuTensor};

/// Integer action indices for one matrix row each. Storage is never exposed as
/// a floating-point tensor. The declared column bound is checked by gather.
pub struct GpuIndices {
    pub(super) storage: GpuTensor,
    columns: usize,
}
impl GpuIndices {
    pub fn len(&self) -> usize {
        self.storage.numel()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
}
pub(super) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("action indices"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("action indices"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../gpu_indices.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("action indices"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    })
}
impl GpuContext {
    /// Uploads typed actions, validating every index before queuing a write.
    pub fn upload_indices(
        &self,
        actions: &[usize],
        columns: usize,
    ) -> Result<GpuIndices, GpuError> {
        if columns == 0 || columns > u32::MAX as usize {
            return Err(GpuError::InvalidIndices {
                shape: vec![],
                count: actions.len(),
                columns,
            });
        }
        for &action in actions {
            if action >= columns {
                return Err(GpuError::ActionOutOfBounds { action, columns });
            }
        }
        let storage = self.zeros(&[actions.len()])?;
        if !actions.is_empty() {
            let values: Vec<u32> = actions.iter().map(|&action| action as u32).collect();
            self.inner
                .queue
                .write_buffer(&storage.buffer, 0, bytemuck::cast_slice(&values));
            self.record_profile(super::GpuProfileCounters {
                uploads: 1,
                upload_bytes: (values.len() as u64) * 4,
                ..Default::default()
            });
            #[cfg(test)]
            self.inner
                .transfers
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(GpuIndices { storage, columns })
    }
    /// Selects the last greatest value per row, matching CPU total_cmp ties
    /// (including signed zero). Nonfinite rows produce an invalid sentinel;
    /// download reports an error and gather produces NaN without indexing it.
    pub fn argmax_rows_device(&self, matrix: &GpuTensor) -> Result<GpuIndices, GpuError> {
        self.ensure_owner(matrix)?;
        if matrix.shape().len() != 2 || matrix.shape()[1] == 0 {
            return Err(GpuError::InvalidIndices {
                shape: matrix.shape().to_vec(),
                count: 0,
                columns: 0,
            });
        }
        let columns = matrix.shape()[1];
        if columns > u32::MAX as usize {
            return Err(GpuError::LimitExceeded);
        }
        let storage = self.zeros(&[matrix.shape()[0]])?;
        if !storage.is_empty() {
            self.dispatch_operation(
                matrix,
                matrix,
                &storage,
                [storage.numel() as u32, 0, columns as u32],
                &self.inner.indices_pipeline,
            )?;
        }
        Ok(GpuIndices { storage, columns })
    }
    pub fn download_indices(&self, indices: &GpuIndices) -> Result<Vec<usize>, GpuError> {
        let values = self.readback_values::<u32>(&indices.storage)?;
        if values
            .iter()
            .any(|&value| value as usize >= indices.columns)
        {
            return Err(GpuError::NonFinite);
        }
        Ok(values.into_iter().map(|value| value as usize).collect())
    }
    fn validate_indices(&self, matrix: &GpuTensor, indices: &GpuIndices) -> Result<(), GpuError> {
        self.ensure_owner(matrix)?;
        self.ensure_owner(&indices.storage)?;
        if matrix.shape() != [indices.len(), indices.columns] {
            return Err(GpuError::InvalidIndices {
                shape: matrix.shape().to_vec(),
                count: indices.len(),
                columns: indices.columns,
            });
        }
        Ok(())
    }
    /// Gathers one action per row into [batch, 1], without index readback.
    pub fn gather_rows_device(
        &self,
        matrix: &GpuTensor,
        indices: &GpuIndices,
    ) -> Result<GpuTensor, GpuError> {
        self.validate_indices(matrix, indices)?;
        let output = self.zeros(&[indices.len(), 1])?;
        if !output.is_empty() {
            self.dispatch_operation(
                matrix,
                &indices.storage,
                &output,
                [output.numel() as u32, 1, indices.columns as u32],
                &self.inner.indices_pipeline,
            )?;
        }
        Ok(output)
    }
    /// Scatters a [batch, 1] incoming gather gradient into [batch, actions].
    /// One invocation writes each output element, so no atomics are required.
    pub fn scatter_rows_device(
        &self,
        gradient: &GpuTensor,
        indices: &GpuIndices,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(gradient)?;
        self.ensure_owner(&indices.storage)?;
        if gradient.shape() != [indices.len(), 1] {
            return Err(GpuError::InvalidIndices {
                shape: gradient.shape().to_vec(),
                count: indices.len(),
                columns: indices.columns,
            });
        }
        let output = self.zeros(&[indices.len(), indices.columns])?;
        if !output.is_empty() {
            self.dispatch_operation(
                gradient,
                &indices.storage,
                &output,
                [output.numel() as u32, 2, indices.columns as u32],
                &self.inner.indices_pipeline,
            )?;
        }
        Ok(output)
    }
}
