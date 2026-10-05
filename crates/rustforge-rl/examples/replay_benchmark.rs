//! CPU replay microbenchmark. Run with `cargo run --release -p rustforge-rl
//! --example replay_benchmark -- 20000`. Reports the median of seven trials.
use rand::{rngs::StdRng, SeedableRng};
use rustforge_rl::buffer::{
    ContinuousReplayBuffer, ContinuousTransitionBatch, PrioritizedReplayBuffer, ReplayBuffer,
    TransitionBatch,
};
use rustforge_tensor::Tensor;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: All memory operations are delegated unchanged to the system allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure(kind: &str, obs: usize, batch: usize, iterations: usize, mut sample: impl FnMut()) {
    for _ in 0..1000 {
        sample();
    }
    // Count allocations separately so atomic increments do not bias timings.
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..iterations {
        sample();
    }
    COUNTING.store(false, Ordering::Relaxed);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let mut trials = [0.0_f64; 7];
    for trial in &mut trials {
        let start = Instant::now();
        for _ in 0..iterations {
            sample();
        }
        *trial = start.elapsed().as_nanos() as f64 / iterations as f64;
    }
    trials.sort_by(f64::total_cmp);
    println!(
        "{kind},{obs},{batch},{iterations},{:.1},{allocations}",
        trials[3]
    );
}

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>().expect("iterations must be an integer"))
        .unwrap_or(20_000);
    assert!(iterations > 0, "iterations must be positive");
    println!("sampler,obs_dim,batch_size,iterations,median_ns_per_batch,allocations");
    for (obs, batch_size) in [(4, 64), (32, 256)] {
        let capacity = 16_384;
        let mut uniform = ReplayBuffer::new(capacity, obs);
        let mut per = PrioritizedReplayBuffer::with_seed(capacity, obs, 0.6, 42);
        let mut continuous = ContinuousReplayBuffer::new(capacity, obs, 2);
        let mut state = vec![0.0; obs];
        for i in 0..capacity {
            state.fill(i as f32);
            uniform.push(&state, i % 2, i as f32, &state, i % 7 == 0);
            per.push(&state, i % 2, i as f32, &state, i % 7 == 0);
            continuous.push(&state, &[0.25, -0.5], i as f32, &state, i % 7 == 0);
        }
        // Unequal priorities exercise importance weights, beyond the initial uniform tree.
        let indices: Vec<_> = (capacity - 1..2 * capacity - 1).collect();
        let errors: Vec<_> = (0..capacity).map(|i| (i % 100 + 1) as f32).collect();
        per.update_priorities(&indices, &errors);
        let mut batch = TransitionBatch::new(batch_size, obs);
        let mut weights = Tensor::zeros(&[batch_size, 1]);
        let mut tree_indices = vec![0; batch_size];
        measure("uniform", obs, batch_size, iterations, || {
            uniform.sample(batch_size, &mut batch);
            black_box(&batch);
        });
        measure("prioritized", obs, batch_size, iterations, || {
            per.sample(batch_size, 0.4, &mut batch, &mut weights, &mut tree_indices);
            black_box((&batch, &weights, &tree_indices));
        });
        let mut batch = ContinuousTransitionBatch::new(batch_size, obs, 2);
        let mut rng = StdRng::seed_from_u64(42);
        measure("continuous", obs, batch_size, iterations, || {
            continuous.sample_with_rng(batch_size, &mut batch, &mut rng);
            black_box(&batch);
        });
    }
}
