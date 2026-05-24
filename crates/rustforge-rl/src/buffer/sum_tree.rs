//! A SumTree implementation for Prioritized Experience Replay.

/// A binary tree where each parent node is the sum of its children.
/// Used for efficiently sampling from a generic discrete probability distribution.
///
/// The tree is implemented using a flat array. For a capacity `N`, the tree requires
/// `2N - 1` nodes. The first `N - 1` nodes are internal nodes, and the last `N` nodes
/// are the leaf nodes storing the priorities.
pub struct SumTree {
    capacity: usize,
    tree: Vec<f32>,
    write_ptr: usize,
    size: usize,
}

impl SumTree {
    /// Creates a new SumTree with the given capacity.
    /// The capacity determines the maximum number of leaves.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be strictly positive");
        SumTree {
            capacity,
            tree: vec![0.0; 2 * capacity - 1],
            write_ptr: 0,
            size: 0,
        }
    }

    /// Adds a new priority to the tree and returns the index of the leaf node.
    pub fn add(&mut self, priority: f32) -> usize {
        let tree_idx = self.write_ptr + self.capacity - 1;
        self.update(tree_idx, priority);

        let current_ptr = self.write_ptr;
        self.write_ptr = (self.write_ptr + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }

        current_ptr
    }

    /// Updates the priority of the leaf node at the given index.
    pub fn update(&mut self, tree_idx: usize, priority: f32) {
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;
        self.propagate_changes(tree_idx, change);
    }

    /// Propagates the change in priority up to the root.
    fn propagate_changes(&mut self, mut tree_idx: usize, change: f32) {
        while tree_idx != 0 {
            tree_idx = (tree_idx - 1) / 2;
            self.tree[tree_idx] += change;
        }
    }

    /// Samples a leaf node based on the given cumulative sum `s`.
    /// Returns a tuple `(tree_idx, priority, data_idx)`.
    pub fn get(&self, mut s: f32) -> (usize, f32, usize) {
        let mut parent_idx = 0;

        loop {
            let left_child_idx = 2 * parent_idx + 1;
            let right_child_idx = left_child_idx + 1;

            if left_child_idx >= self.tree.len() {
                // Reached a leaf node
                break;
            }

            if s <= self.tree[left_child_idx] {
                parent_idx = left_child_idx;
            } else {
                s -= self.tree[left_child_idx];
                parent_idx = right_child_idx;
            }
        }

        let data_idx = parent_idx - (self.capacity - 1);
        (parent_idx, self.tree[parent_idx], data_idx)
    }

    /// Returns the total sum of priorities (the root node).
    pub fn total_priority(&self) -> f32 {
        self.tree[0]
    }

    /// Returns the current number of elements stored.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the capacity of the tree.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_tree_add_and_update() {
        let mut tree = SumTree::new(4);
        tree.add(1.0); // Write ptr 0, tree_idx 3
        tree.add(2.0); // Write ptr 1, tree_idx 4
        tree.add(3.0); // Write ptr 2, tree_idx 5

        assert_eq!(tree.total_priority(), 6.0);

        tree.add(4.0); // Write ptr 3, tree_idx 6
        assert_eq!(tree.total_priority(), 10.0);

        // Overflow capacity
        tree.add(5.0); // Write ptr 0, overwrites 1.0 -> 5.0
        assert_eq!(tree.total_priority(), 14.0); // 5 + 2 + 3 + 4 = 14

        tree.update(4, 6.0); // Update tree_idx 4 from 2.0 to 6.0
        assert_eq!(tree.total_priority(), 18.0); // 5 + 6 + 3 + 4 = 18
    }

    #[test]
    fn test_sum_tree_get() {
        let mut tree = SumTree::new(4);
        tree.add(10.0); // idx 0 (tree 3)
        tree.add(20.0); // idx 1 (tree 4)
        tree.add(30.0); // idx 2 (tree 5)
        tree.add(40.0); // idx 3 (tree 6)

        assert_eq!(tree.total_priority(), 100.0);

        let (_, p, d) = tree.get(5.0);
        assert_eq!(d, 0);
        assert_eq!(p, 10.0);

        let (_, p, d) = tree.get(15.0);
        assert_eq!(d, 1);
        assert_eq!(p, 20.0);

        let (_, p, d) = tree.get(45.0);
        assert_eq!(d, 2);
        assert_eq!(p, 30.0);

        let (_, p, d) = tree.get(80.0);
        assert_eq!(d, 3);
        assert_eq!(p, 40.0);
    }
}
