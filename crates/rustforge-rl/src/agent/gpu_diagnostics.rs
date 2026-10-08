//! Pack finite checks and scalar diagnostics without changing their arithmetic.
use rustforge_autograd::gpu::GpuVariable;
use rustforge_tensor::gpu::{GpuContext, GpuError};

/// None requests the original path for externally replaced scalar fields that
/// have a different owner or non-scalar shape. The caller retains error ordering.
/// Returns the original aggregate invalid-count followed by diagnostic scalars.
pub(super) fn checked_scalars(
    context: &GpuContext,
    checks: &[GpuVariable],
    scalars: &[&GpuVariable],
) -> Result<Option<Vec<f32>>, GpuError> {
    if checks.is_empty()
        || scalars.iter().any(|v| {
            let data = v.data();
            !context.is_compatible(&data) || data.numel() != 1
        })
    {
        return Ok(None);
    }
    let _profile = context.profile_scope("objective_diagnostics");
    let mut count = None;
    for variable in checks {
        let next = context.nonfinite_count_device(&variable.data())?;
        count = Some(match count {
            Some(old) => context.add_device(&old, &next)?,
            None => next,
        });
    }
    let count = count.expect("nonempty checked variables");
    let data: Vec<_> = scalars.iter().map(|v| v.data()).collect();
    let refs: Vec<_> = std::iter::once(&count)
        .chain(data.iter().map(|v| v.as_ref()))
        .collect();
    context.download_scalars(&refs).map(Some)
}
