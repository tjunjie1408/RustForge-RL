//! Heap allocation regression test for the forward/backward hot path.
//!
//! Uses `dhat` to instrument the global allocator and assert that the
//! matmul + backward pass does not exceed an allocation budget.
//!
//! ## Running
//! ```bash
//! cargo test --test dhat_hotpath --features dhat-heap -- --nocapture --test-threads=1
//! ```
//!
//! This test is gated behind `#[cfg(feature = "dhat-heap")]` so it does not
//! interfere with normal `cargo test` runs.
//!
//! ## IMPORTANT
//! dhat only allows one profiler per process. Tests MUST run sequentially
//! (--test-threads=1) to avoid "profiler already running" panics.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
mod tests {
    use rustforge_autograd::Variable;
    use rustforge_tensor::Tensor;

    #[test]
    fn test_all_dhat_allocation_budgets() {
        run_matmul_backward_allocation_budget();
        run_forward_only_allocation_budget();
        run_training_loop_allocation_consistency();
    }

    /// Performs a matmul forward + backward pass and reports heap allocation stats.
    ///
    /// Phase A baseline: captures current allocation count.
    /// After Phase B+C optimizations, the allocation count should drop dramatically.
    fn run_matmul_backward_allocation_budget() {
        // --- Setup phase (allocations here are expected and not measured) ---
        let a_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b_data = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let a = Variable::new(a_data, true);
        let b = Variable::new(b_data, true);

        // Warm-up pass (primes any lazy initialization)
        {
            let c = a.matmul(&b);
            let loss = c.sum();
            loss.backward();
            a.zero_grad();
            b.zero_grad();
        }

        // --- Hot path measurement starts here ---
        let _profiler = dhat::Profiler::builder().testing().build();

        // Forward pass: matmul
        let c = a.matmul(&b);

        // Reduction to scalar
        let loss = c.sum();

        // Backward pass
        loss.backward();

        // --- Measurement ends ---
        let stats = dhat::HeapStats::get();

        eprintln!("=== DHAT Hot-Path Allocation Report ===");
        eprintln!("  Total blocks allocated: {}", stats.total_blocks);
        eprintln!("  Total bytes allocated:  {}", stats.total_bytes);
        eprintln!("  Max blocks live:        {}", stats.max_blocks);
        eprintln!("  Max bytes live:         {}", stats.max_bytes);
        eprintln!("=======================================");

        // Phase A baseline: ~47 blocks per iteration observed.
        // After Phase C, target: < 20 blocks.
        assert!(
            stats.total_blocks < 200,
            "REGRESSION: Hot-path allocated {} blocks (budget: <200 for baseline).",
            stats.total_blocks
        );
    }

    /// Tests that a simple forward-only pass (no backward) has minimal allocations.
    fn run_forward_only_allocation_budget() {
        let a_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b_data = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // No grad tracking — pure tensor forward
        let a = Variable::new(a_data, false);
        let b = Variable::new(b_data, false);

        // Warm-up
        {
            let _c = a.matmul(&b);
        }

        let _profiler = dhat::Profiler::builder().testing().build();

        let _c = a.matmul(&b);

        let stats = dhat::HeapStats::get();

        eprintln!("=== DHAT Forward-Only Report ===");
        eprintln!("  Total blocks allocated: {}", stats.total_blocks);
        eprintln!("  Total bytes allocated:  {}", stats.total_bytes);
        eprintln!("================================");

        // Forward-only baseline: 17 blocks observed.
        // Target after optimization: < 10 blocks.
        assert!(
            stats.total_blocks < 100,
            "REGRESSION: Forward-only allocated {} blocks (budget: <100).",
            stats.total_blocks
        );
    }

    /// Stress test: repeated forward+backward in a loop, measuring per-iteration cost.
    fn run_training_loop_allocation_consistency() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
        let b = Variable::new(Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]), true);

        // Warm-up
        {
            let c = a.matmul(&b);
            c.sum().backward();
            a.zero_grad();
            b.zero_grad();
        }

        let _profiler = dhat::Profiler::builder().testing().build();

        // 10 iterations of forward+backward
        for _ in 0..10 {
            let c = a.matmul(&b);
            let loss = c.sum();
            loss.backward();
            a.zero_grad();
            b.zero_grad();
        }

        let stats = dhat::HeapStats::get();

        eprintln!("=== DHAT Training Loop Report (10 iters) ===");
        eprintln!("  Total blocks allocated: {}", stats.total_blocks);
        eprintln!("  Total bytes allocated:  {}", stats.total_bytes);
        eprintln!("  Blocks per iteration:   ~{}", stats.total_blocks / 10);
        eprintln!("  Bytes per iteration:    ~{}", stats.total_bytes / 10);
        eprintln!("=============================================");

        // Baseline: 47 blocks/iter, 2353 bytes/iter.
        // Target after optimization: < 20 blocks/iter.
        let blocks_per_iter = stats.total_blocks / 10;
        assert!(
            blocks_per_iter < 200,
            "REGRESSION: {} blocks per iteration (budget: <200).",
            blocks_per_iter
        );
    }
}
