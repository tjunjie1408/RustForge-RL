//! Optional, explicit GPU operations. Enable the `gpu` Cargo feature.
//!
//! Upload once and chain operations using persistent `GpuTensor` storage.
//! Downloads are explicit. This backend does not track gradients or accelerate
//! existing `Tensor::matmul` calls automatically.

use std::fmt;
use std::sync::{mpsc, Arc, Mutex, OnceLock};

use wgpu::util::DeviceExt;

use crate::Tensor;

mod indices;
mod operations;
pub use indices::GpuIndices;

/// Matrix kernel selection for measurement and adapter-specific tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulKernel {
    /// Reuses input tiles in workgroup memory.
    Tiled,
    /// Direct dot products, retained as a reference and for small workloads.
    Naive,
}

/// Failures from GPU initialization or tensor operations.
#[derive(Debug)]
pub enum GpuError {
    /// No compatible graphics or software adapter is available.
    AdapterUnavailable,
    /// The adapter could not create a compute device.
    RequestDevice(wgpu::RequestDeviceError),
    /// Operands must be rank-two matrices with matching inner dimensions.
    InvalidShape { left: Vec<usize>, right: Vec<usize> },
    /// Bias addition requires [batch, features] and [features].
    InvalidBiasShape {
        matrix: Vec<usize>,
        bias: Vec<usize>,
    },
    /// A row reduction requires a rank-two tensor.
    ExpectedMatrix { shape: Vec<usize> },
    /// A shape, buffer, index, or dispatch exceeds supported limits.
    LimitExceeded,
    /// A tensor belongs to a different compute device.
    DeviceMismatch,
    /// A scalar-broadcast source must contain exactly one element.
    ExpectedScalar { shape: Vec<usize> },
    /// A reusable output does not have the required shape.
    InvalidOutputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Elementwise binary operations require identical logical shapes.
    ElementwiseShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    /// Action selection/gather requires a nonempty feature axis and compatible indices.
    InvalidIndices {
        shape: Vec<usize>,
        count: usize,
        columns: usize,
    },
    /// A host action index exceeds the declared action count.
    ActionOutOfBounds { action: usize, columns: usize },
    /// Action selection encountered nonfinite Q-values.
    NonFinite,
    /// Reading the computed buffer failed.
    Readback(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterUnavailable => write!(f, "no compatible GPU adapter available"),
            Self::RequestDevice(error) => write!(f, "could not create GPU device: {error}"),
            Self::InvalidShape { left, right } => {
                write!(
                    f,
                    "GPU matmul requires [m, k] and [k, n], got {left:?} and {right:?}"
                )
            }
            Self::InvalidBiasShape { matrix, bias } => write!(f, "GPU bias addition requires [batch, features] and [features], got {matrix:?} and {bias:?}"),
            Self::ExpectedMatrix { shape } => write!(f, "GPU row reduction requires a matrix, got {shape:?}"),
            Self::LimitExceeded => write!(f, "GPU tensor exceeds supported shape or device limits"),
            Self::DeviceMismatch => write!(f, "tensor belongs to a different GPU device"),
            Self::ExpectedScalar { shape } => {
                write!(f, "expected a single-element GPU tensor, got {shape:?}")
            }
            Self::InvalidOutputShape { expected, actual } => {
                write!(f, "GPU output shape must be {expected:?}, got {actual:?}")
            }
            Self::ElementwiseShapeMismatch { left, right } => write!(
                f,
                "GPU elementwise shapes must match, got {left:?} and {right:?}"
            ),
            Self::InvalidIndices { shape, count, columns } => write!(f, "action indices ({count} rows, {columns} columns) do not match matrix {shape:?}"),
            Self::ActionOutOfBounds { action, columns } => write!(f, "action {action} is outside 0..{columns}"),
            Self::NonFinite => write!(f, "GPU action selection encountered nonfinite values"),
            Self::Readback(error) => write!(f, "GPU buffer readback failed: {error}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RequestDevice(error) => Some(error),
            _ => None,
        }
    }
}

/// Reusable compute device with cached `f32` tensor operation pipelines.
///
/// The process compute device is initialized once; each new context establishes
/// a separate tensor ownership scope. Clone a context to share that scope.
/// A software adapter may be selected on machines without hardware graphics.
/// Cloning a context shares the same device and accepts the same tensors.
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<DeviceState>,
    matmul_kernel: MatmulKernel,
}

struct DeviceState {
    resources: Arc<DeviceResources>,
    #[cfg(test)]
    transfers: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    tensor_allocations: std::sync::atomic::AtomicUsize,
}

impl std::ops::Deref for DeviceState {
    type Target = DeviceResources;
    fn deref(&self) -> &Self::Target {
        &self.resources
    }
}

struct DeviceResources {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    naive_pipeline: wgpu::ComputePipeline,
    ops_layout: wgpu::BindGroupLayout,
    elementwise_pipeline: wgpu::ComputePipeline,
    reduction_pipeline: wgpu::ComputePipeline,
    indices_pipeline: wgpu::ComputePipeline,
    adapter_info: wgpu::AdapterInfo,
}

/// Owned, contiguous `f32` tensor storage on a specific compute device.
///
/// Shape is immutable. Storage remains alive until the tensor is dropped, and
/// retains its device even if the original context is dropped. Storage handles
/// cannot be cloned; reuse inputs by reference and outputs by mutable borrow.
pub struct GpuTensor {
    owner: Arc<DeviceState>,
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
}

impl GpuTensor {
    /// Logical tensor dimensions, including empty dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of logical elements; a scalar has one element.
    pub fn numel(&self) -> usize {
        if self.shape.contains(&0) {
            0
        } else {
            self.shape.iter().product()
        }
    }

    /// Whether the logical tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }
}

impl GpuContext {
    /// Creates an isolated tensor ownership scope on the process compute device.
    /// Device/queue/pipelines are initialized once and retained for the process
    /// lifetime. Separate contexts reject each other's tensors; clones share a scope.
    pub fn new() -> Result<Self, GpuError> {
        // wgpu 0.19 EGL instances may share a display, while repeated adapter
        // enumeration creates separate locks for the same GL context. Multiple
        // devices requested from one adapter also alias queue IDs. Initialize
        // one compute device and share its thread-safe resources instead.
        static RESOURCES: OnceLock<Arc<DeviceResources>> = OnceLock::new();
        static INITIALIZE: Mutex<()> = Mutex::new(());
        let resources = if let Some(resources) = RESOURCES.get() {
            Arc::clone(resources)
        } else {
            let _guard = INITIALIZE.lock().unwrap_or_else(|error| error.into_inner());
            if let Some(resources) = RESOURCES.get() {
                Arc::clone(resources)
            } else {
                let resources = Arc::new(pollster::block_on(Self::initialize())?);
                let _ = RESOURCES.set(Arc::clone(&resources));
                resources
            }
        };
        let matmul_kernel = if resources.adapter_info.device_type == wgpu::DeviceType::Cpu {
            MatmulKernel::Naive
        } else {
            MatmulKernel::Tiled
        };
        Ok(Self {
            matmul_kernel,
            inner: Arc::new(DeviceState {
                resources,
                #[cfg(test)]
                transfers: std::sync::atomic::AtomicUsize::new(0),
                #[cfg(test)]
                tensor_allocations: std::sync::atomic::AtomicUsize::new(0),
            }),
        })
    }

    async fn initialize() -> Result<DeviceResources, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::AdapterUnavailable)?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RustForge GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(GpuError::RequestDevice)?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RustForge matmul"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_matmul.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RustForge matmul"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });
        let naive_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RustForge reference matmul"),
            layout: None,
            module: &shader,
            entry_point: "naive",
        });
        let (ops_layout, elementwise_pipeline, reduction_pipeline) =
            operations::create_pipelines(&device);
        let indices_pipeline = indices::create_pipeline(&device, &ops_layout);
        Ok(DeviceResources {
            device,
            queue,
            pipeline,
            naive_pipeline,
            ops_layout,
            elementwise_pipeline,
            reduction_pipeline,
            indices_pipeline,
            adapter_info,
        })
    }

    /// Identifies the selected adapter, including whether it is a software CPU.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.inner.adapter_info
    }

    /// Shares this device with a different matrix kernel selection.
    ///
    /// Tensors are compatible with both handles. This does not reinitialize
    /// the device or transfer data; use it to compare kernels on an adapter.
    pub fn with_matmul_kernel(&self, kernel: MatmulKernel) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            matmul_kernel: kernel,
        }
    }

    /// Waits for previously queued device work without downloading tensor data.
    pub fn synchronize(&self) {
        self.inner.device.poll(wgpu::Maintain::Wait);
    }

    /// Uploads a CPU tensor into owned, reusable device storage.
    ///
    /// Scalars and arbitrary ranks can be transferred; matrix multiplication
    /// still requires rank two. Noncontiguous inputs are packed in logical order.
    pub fn upload(&self, tensor: &Tensor) -> Result<GpuTensor, GpuError> {
        validate_storage(tensor.shape(), &self.inner.device.limits())?;
        #[cfg(test)]
        {
            self.inner
                .transfers
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner
                .tensor_allocations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let values = tensor.to_vec();
        // Empty tensors need a nonzero physical buffer but retain their shape.
        let contents = if values.is_empty() {
            bytemuck::cast_slice(&[0.0f32])
        } else {
            bytemuck::cast_slice(&values)
        };
        let buffer = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uploaded tensor"),
                contents,
                usage: tensor_usage(),
            });
        Ok(GpuTensor {
            owner: Arc::clone(&self.inner),
            buffer,
            shape: tensor.shape().to_vec(),
        })
    }

    /// Allocates zero-initialized device storage, suitable for output reuse.
    pub fn zeros(&self, shape: &[usize]) -> Result<GpuTensor, GpuError> {
        let bytes = validate_storage(shape, &self.inner.device.limits())?;
        #[cfg(test)]
        self.inner
            .tensor_allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("device tensor"),
            size: bytes.max(4),
            usage: tensor_usage(),
            mapped_at_creation: false,
        });
        Ok(GpuTensor {
            owner: Arc::clone(&self.inner),
            buffer,
            shape: shape.to_vec(),
        })
    }

    /// Downloads a tensor from this context's device, waiting for queued work.
    ///
    /// Explicit tensor/index downloads map readback buffers. Empty values require no transfer.
    pub fn download(&self, tensor: &GpuTensor) -> Result<Tensor, GpuError> {
        Ok(Tensor::from_vec(
            self.readback_values::<f32>(tensor)?,
            tensor.shape(),
        ))
    }

    // Shared raw readback keeps integer indices out of the f32 tensor API.
    fn readback_values<T: bytemuck::Pod>(&self, tensor: &GpuTensor) -> Result<Vec<T>, GpuError> {
        self.ensure_owner(tensor)?;
        let output_bytes = validate_storage(tensor.shape(), &self.inner.device.limits())?;
        if output_bytes == 0 {
            return Ok(Vec::new());
        }
        #[cfg(test)]
        self.inner
            .transfers
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let readback = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("download commands"),
                });
        encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &readback, 0, output_bytes);
        self.inner.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.inner.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| GpuError::Readback(error.to_string()))?
            .map_err(|error| GpuError::Readback(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        Ok(values)
    }

    /// Enqueues a matrix product and returns its persistent device output.
    ///
    /// Inputs and outputs stay on the device. Queue ordering allows the result
    /// to feed another operation immediately without a host wait or readback.
    pub fn matmul_device(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.matmul_device_impl(left, right, false, false)
    }

    /// Computes A × Bᵀ on device, without allocating a transposed copy of B.
    pub fn matmul_t(&self, left: &GpuTensor, right: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.matmul_device_impl(left, right, false, true)
    }

    /// Computes Aᵀ × B on device, without allocating a transposed copy of A.
    pub fn t_matmul(&self, left: &GpuTensor, right: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.matmul_device_impl(left, right, true, false)
    }

    fn matmul_device_impl(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Result<GpuTensor, GpuError> {
        self.ensure_owner(left)?;
        self.ensure_owner(right)?;
        let [m, _, n] = validate_product(
            left.shape(),
            right.shape(),
            transpose_left,
            transpose_right,
            &self.inner.device.limits(),
        )?;
        let mut output = self.zeros(&[m as usize, n as usize])?;
        self.matmul_into_impl(left, right, &mut output, transpose_left, transpose_right)?;
        Ok(output)
    }

    /// Enqueues a matrix product into an existing device output buffer.
    ///
    /// The output must have exactly the expected shape and belong to this
    /// device. Exclusive borrowing and non-cloneable tensors prevent aliasing
    /// an input with the output. No tensor buffer is allocated or transferred;
    /// small dispatch parameters and bindings are still allocated per call.
    pub fn matmul_into(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), GpuError> {
        self.matmul_into_impl(left, right, output, false, false)
    }

    /// Writes A × Bᵀ into an exact-shape existing output buffer.
    pub fn matmul_t_into(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), GpuError> {
        self.matmul_into_impl(left, right, output, false, true)
    }

    /// Writes Aᵀ × B into an exact-shape existing output buffer.
    pub fn t_matmul_into(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), GpuError> {
        self.matmul_into_impl(left, right, output, true, false)
    }

    fn matmul_into_impl(
        &self,
        left: &GpuTensor,
        right: &GpuTensor,
        output: &mut GpuTensor,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Result<(), GpuError> {
        for tensor in [left, right, &*output] {
            self.ensure_owner(tensor)?;
        }
        let [m, k, n] = validate_product(
            left.shape(),
            right.shape(),
            transpose_left,
            transpose_right,
            &self.inner.device.limits(),
        )?;
        let expected = vec![m as usize, n as usize];
        if output.shape() != expected {
            return Err(GpuError::InvalidOutputShape {
                expected,
                actual: output.shape().to_vec(),
            });
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("matmul commands"),
                });
        if k == 0 {
            // Clear a reused output instead of leaving a previous result behind.
            encoder.clear_buffer(&output.buffer, 0, None);
        } else {
            let parameters =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("matmul dimensions"),
                        contents: bytemuck::cast_slice(&[
                            m,
                            k,
                            n,
                            u32::from(transpose_left) | (u32::from(transpose_right) << 1),
                        ]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let pipeline = match self.matmul_kernel {
                MatmulKernel::Tiled => &self.inner.pipeline,
                MatmulKernel::Naive => &self.inner.naive_pipeline,
            };
            let layout = pipeline.get_bind_group_layout(0);
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
                    label: Some("matmul bindings"),
                    layout: &layout,
                    entries: &entries,
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(n.div_ceil(8), m.div_ceil(8), 1);
        }
        self.inner.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Convenience CPU-to-CPU matrix multiplication using the device backend.
    ///
    /// For chaining or repeated inputs, use upload, matmul_device/matmul_into,
    /// and download explicitly to avoid intermediate host transfers.
    pub fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, GpuError> {
        validate_dimensions(left.shape(), right.shape(), &self.inner.device.limits())?;
        let left = self.upload(left)?;
        let right = self.upload(right)?;
        self.download(&self.matmul_device(&left, &right)?)
    }

    /// Checks whether a tensor belongs to this device, without a transfer.
    pub fn is_compatible(&self, tensor: &GpuTensor) -> bool {
        Arc::ptr_eq(&self.inner, &tensor.owner)
    }

    fn ensure_owner(&self, tensor: &GpuTensor) -> Result<(), GpuError> {
        if self.is_compatible(tensor) {
            Ok(())
        } else {
            Err(GpuError::DeviceMismatch)
        }
    }
}

fn tensor_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

fn validate_storage(shape: &[usize], limits: &wgpu::Limits) -> Result<u64, GpuError> {
    for &dimension in shape {
        u32::try_from(dimension).map_err(|_| GpuError::LimitExceeded)?;
    }
    // ndarray also bounds the product of nonzero axes for empty tensors.
    // Preserve that invariant so a later download cannot panic on its shape.
    let nonzero_elements = shape
        .iter()
        .try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension.max(1))
        })
        .ok_or(GpuError::LimitExceeded)?;
    if nonzero_elements > isize::MAX as usize {
        return Err(GpuError::LimitExceeded);
    }
    let elements = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1u64, |count, &dimension| {
                count.checked_mul(dimension as u64)
            })
            .ok_or(GpuError::LimitExceeded)?
    };
    if elements > u64::from(u32::MAX) {
        return Err(GpuError::LimitExceeded);
    }
    let bytes = elements * 4;
    if bytes.max(4) > limits.max_buffer_size
        || bytes.max(4) > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(GpuError::LimitExceeded);
    }
    Ok(bytes)
}

fn validate_product(
    left: &[usize],
    right: &[usize],
    transpose_left: bool,
    transpose_right: bool,
    limits: &wgpu::Limits,
) -> Result<[u32; 3], GpuError> {
    if left.len() != 2 || right.len() != 2 {
        return Err(GpuError::InvalidShape {
            left: left.to_vec(),
            right: right.to_vec(),
        });
    }
    let a = if transpose_left {
        [left[1], left[0]]
    } else {
        [left[0], left[1]]
    };
    let b = if transpose_right {
        [right[1], right[0]]
    } else {
        [right[0], right[1]]
    };
    validate_dimensions(&a, &b, limits)
}

fn validate_dimensions(
    left: &[usize],
    right: &[usize],
    limits: &wgpu::Limits,
) -> Result<[u32; 3], GpuError> {
    if left.len() != 2 || right.len() != 2 || left[1] != right[0] {
        return Err(GpuError::InvalidShape {
            left: left.to_vec(),
            right: right.to_vec(),
        });
    }
    let m = u32::try_from(left[0]).map_err(|_| GpuError::LimitExceeded)?;
    let k = u32::try_from(left[1]).map_err(|_| GpuError::LimitExceeded)?;
    let n = u32::try_from(right[1]).map_err(|_| GpuError::LimitExceeded)?;
    for (rows, columns) in [(m, k), (k, n), (m, n)] {
        let elements = u64::from(rows) * u64::from(columns);
        // Shader indexing uses u32, even when the host supports larger buffers.
        if elements > u64::from(u32::MAX)
            || elements * 4 > limits.max_buffer_size
            || elements * 4 > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err(GpuError::LimitExceeded);
        }
    }
    if n.div_ceil(8) > limits.max_compute_workgroups_per_dimension
        || m.div_ceil(8) > limits.max_compute_workgroups_per_dimension
    {
        return Err(GpuError::LimitExceeded);
    }
    Ok([m, k, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_effective_transposed_dimensions() {
        let limits = wgpu::Limits::default();
        assert_eq!(
            validate_product(&[9, 17], &[11, 17], false, true, &limits).unwrap(),
            [9, 17, 11]
        );
        assert_eq!(
            validate_product(&[9, 17], &[9, 11], true, false, &limits).unwrap(),
            [17, 9, 11]
        );
        assert_eq!(
            validate_product(&[9, 17], &[11, 9], true, true, &limits).unwrap(),
            [17, 9, 11]
        );
        assert!(matches!(
            validate_product(&[9, 17], &[9, 11], false, true, &limits),
            Err(GpuError::InvalidShape { .. })
        ));
        assert!(matches!(
            validate_product(&[], &[1, 1], true, false, &limits),
            Err(GpuError::InvalidShape { .. })
        ));
    }

    #[test]
    fn validates_storage_sizes_without_allocating() {
        let limits = wgpu::Limits::default();
        assert_eq!(validate_storage(&[], &limits).unwrap(), 4);
        assert_eq!(validate_storage(&[2, 3, 4], &limits).unwrap(), 96);
        assert_eq!(validate_storage(&[65536, 1024, 0], &limits).unwrap(), 0);
        for shape in [vec![65536, 65536], vec![u32::MAX as usize; 4]] {
            assert!(matches!(
                validate_storage(&shape, &limits),
                Err(GpuError::LimitExceeded)
            ));
        }
        assert!(matches!(
            validate_storage(&[u32::MAX as usize, u32::MAX as usize, 0], &limits),
            Err(GpuError::LimitExceeded)
        ));
        let tiny = wgpu::Limits {
            max_storage_buffer_binding_size: 8,
            ..limits
        };
        assert!(matches!(
            validate_storage(&[3], &tiny),
            Err(GpuError::LimitExceeded)
        ));
    }

    #[test]
    #[ignore = "requires a hardware or software wgpu adapter"]
    fn persistent_operations_avoid_transfers_and_reuse_storage() {
        use std::sync::atomic::Ordering;
        let context = GpuContext::new().expect("GPU adapter required for this test");
        let left = context.upload(&Tensor::ones(&[2, 3])).unwrap();
        let right = context.upload(&Tensor::ones(&[3, 2])).unwrap();
        let identity = context.upload(&Tensor::eye(2)).unwrap();
        let actions = context.upload_indices(&[0, 1], 2).unwrap();
        let transfers = context.inner.transfers.load(Ordering::Relaxed);
        let intermediate = context.matmul_device(&left, &right).unwrap();
        let mut output = context.matmul_device(&intermediate, &identity).unwrap();
        assert_eq!(context.inner.transfers.load(Ordering::Relaxed), transfers);
        let allocation_count = context.inner.tensor_allocations.load(Ordering::Relaxed);
        let buffer_id = output.buffer.global_id();
        for _ in 0..5 {
            context
                .matmul_into(&intermediate, &identity, &mut output)
                .unwrap();
        }
        assert_eq!(output.buffer.global_id(), buffer_id);
        assert_eq!(
            context.inner.tensor_allocations.load(Ordering::Relaxed),
            allocation_count
        );
        assert_eq!(context.inner.transfers.load(Ordering::Relaxed), transfers);
        assert_eq!(context.download(&output).unwrap().to_vec(), vec![3.; 4]);
        assert_eq!(
            context.inner.transfers.load(Ordering::Relaxed),
            transfers + 1
        );
        let before_operations = context.inner.transfers.load(Ordering::Relaxed);
        let doubled = context.add_device(&output, &output).unwrap();
        let activated = context.relu_device(&doubled).unwrap();
        let sum = context.sum_device(&activated).unwrap();
        let mean = context.mean_device(&activated).unwrap();
        let constant = context.full(&[2, 2], -2.).unwrap();
        let scaled = context.scale_device(&constant, -0.5).unwrap();
        let broadcast = context
            .broadcast_scalar_device(&sum, &[2, 2], 0.25)
            .unwrap();
        let relu_gradient = context.relu_backward_device(&constant, &broadcast).unwrap();
        let bias = context.full(&[2], 2.).unwrap();
        let affine = context.add_bias_device(&output, &bias).unwrap();
        let row_sum = context.sum_rows_device(&affine).unwrap();
        let denominator = context.sqrt_add_device(&scaled, 1.).unwrap();
        let ratio = context.div_device(&broadcast, &denominator).unwrap();
        let chosen = context.argmax_rows_device(&affine).unwrap();
        let gathered = context.gather_rows_device(&affine, &chosen).unwrap();
        let scattered = context.scatter_rows_device(&gathered, &actions).unwrap();
        assert_eq!(
            context.inner.transfers.load(Ordering::Relaxed),
            before_operations
        );
        assert_eq!(context.download(&sum).unwrap().item(), 24.);
        assert_eq!(context.download(&mean).unwrap().item(), 6.);
        assert_eq!(context.download(&scaled).unwrap().to_vec(), vec![1.; 4]);
        assert_eq!(context.download(&broadcast).unwrap().to_vec(), vec![6.; 4]);
        assert_eq!(
            context.download(&relu_gradient).unwrap().to_vec(),
            vec![0.; 4]
        );
        assert_eq!(context.download(&row_sum).unwrap().to_vec(), vec![10.; 2]);
        assert_eq!(context.download(&ratio).unwrap().to_vec(), vec![3.; 4]);
        assert_eq!(context.download_indices(&chosen).unwrap(), vec![1, 1]);
        assert_eq!(context.download(&gathered).unwrap().to_vec(), vec![5.; 2]);
        assert_eq!(
            context.download(&scattered).unwrap().to_vec(),
            vec![5., 0., 0., 5.]
        );
        drop(actions);
        drop(chosen);
        drop(gathered);
        drop(scattered);
        drop(bias);
        drop(affine);
        drop(row_sum);
        drop(denominator);
        drop(ratio);
        drop(constant);
        drop(scaled);
        drop(broadcast);
        drop(relu_gradient);
        drop(doubled);
        drop(activated);
        drop(sum);
        drop(mean);
        let device = Arc::downgrade(&context.inner);
        drop(left);
        drop(right);
        drop(identity);
        drop(intermediate);
        drop(context);
        assert!(device.upgrade().is_some(), "the output retains its device");
        let retained = GpuContext {
            inner: Arc::clone(&output.owner),
            matmul_kernel: MatmulKernel::Tiled,
        };
        assert_eq!(retained.download(&output).unwrap().to_vec(), vec![3.; 4]);
        drop(retained);
        drop(output);
        assert!(
            device.upgrade().is_none(),
            "device ownership has no reference cycle"
        );
    }

    #[test]
    fn rejects_invalid_ranks_and_inner_dimensions() {
        for (left, right) in [
            (vec![2], vec![2, 3]),
            (vec![2, 3], vec![4, 2]),
            (vec![1, 2, 3], vec![3, 2]),
        ] {
            assert!(matches!(
                validate_dimensions(&left, &right, &wgpu::Limits::default()),
                Err(GpuError::InvalidShape { .. })
            ));
        }
    }

    #[test]
    fn rejects_buffer_dispatch_and_index_overflow() {
        let limits = wgpu::Limits::default();
        assert!(matches!(
            validate_dimensions(&[65536, 65536], &[65536, 1], &limits),
            Err(GpuError::LimitExceeded)
        ));
        assert!(matches!(
            validate_dimensions(
                &[1, 1],
                &[
                    1,
                    8 * (limits.max_compute_workgroups_per_dimension as usize + 1)
                ],
                &limits
            ),
            Err(GpuError::LimitExceeded)
        ));
        let tiny = wgpu::Limits {
            max_storage_buffer_binding_size: 16,
            ..limits
        };
        assert!(matches!(
            validate_dimensions(&[2, 3], &[3, 2], &tiny),
            Err(GpuError::LimitExceeded)
        ));
    }

    #[test]
    fn accepts_empty_and_rectangular_products() {
        for (left, right, expected) in [
            ([0, 3], [3, 2], [0, 3, 2]),
            ([2, 0], [0, 3], [2, 0, 3]),
            ([7, 9], [9, 11], [7, 9, 11]),
        ] {
            assert_eq!(
                validate_dimensions(&left, &right, &wgpu::Limits::default()).unwrap(),
                expected
            );
        }
    }
}
