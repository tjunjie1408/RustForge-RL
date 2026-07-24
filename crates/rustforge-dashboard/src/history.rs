//! Fixed-capacity histories used by long-running monitor sessions.

use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct BoundedHistory<T> {
    items: VecDeque<T>,
    capacity: usize,
    evicted: u64,
}

impl<T> BoundedHistory<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
            evicted: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.capacity == 0 {
            self.evicted += 1;
            return;
        }
        if self.items.len() == self.capacity {
            self.items.pop_front();
            self.evicted += 1;
        }
        self.items.push_back(item);
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.evicted = 0;
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn evicted(&self) -> u64 {
        self.evicted
    }

    pub fn back(&self) -> Option<&T> {
        self.items.back()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.items.iter()
    }
}
