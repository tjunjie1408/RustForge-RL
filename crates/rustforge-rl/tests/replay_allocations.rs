//! Track only this test thread, excluding allocations by the test harness.
use rustforge_rl::buffer::{PrioritizedReplayBuffer, TransitionBatch};
use rustforge_tensor::Tensor;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count_allocation() {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

// SAFETY: Requests and pointers pass unchanged to the system allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn prioritized_sampling_reuses_storage_without_allocations() {
    let mut replay = PrioritizedReplayBuffer::with_seed(128, 4, 0.6, 42);
    for i in 0..128 {
        replay.push(&[i as f32; 4], i % 2, 1.0, &[2.0; 4], false);
    }
    let mut batch = TransitionBatch::new(256, 4);
    let mut weights = Tensor::zeros(&[256, 1]);
    let mut indices = [0; 256];

    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|enabled| enabled.set(true));
    for size in [0, 1, 64, 256] {
        for _ in 0..10 {
            replay.sample(size, 0.4, &mut batch, &mut weights, &mut indices);
        }
    }
    COUNTING.with(|enabled| enabled.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}
