//! Reverse-mode automatic differentiation backward pass.
//!
//! Computes gradients for all `Variable`s in the computation graph
//! by walking the graph in reverse topological order (from output to inputs).
//!
//! ## Algorithm
//!
//! 1. Seed the output variable's gradient with `ones_like(output)` (typically a scalar 1.0)
//! 2. Compute topological sort via DFS post-order
//! 3. Reverse to get output-first ordering
//! 4. For each node, call `GradFn::backward()` to compute input gradients
//! 5. Accumulate (add) gradients into each input Variable's `.grad` field
//!
//! ## Graph Traversal
//!
//! The computation graph is implicit — each Variable optionally holds a `GradFn`
//! which stores `Rc`-cloned references to its input Variables. We use pointer
//! identity (`Rc::as_ptr`) for the visited set to correctly handle shared nodes.

use std::collections::HashSet;
use std::rc::Rc;

use rustforge_tensor::Tensor;

use crate::variable::Variable;

/// Performs reverse-mode AD from the given output variable.
///
/// After this call, all `Variable`s with `requires_grad = true` that contribute
/// to the output will have their `.grad()` populated with the computed gradient.
///
/// ## Panics
/// Panics if `output` is not a scalar (single-element tensor).
pub fn backward(output: &Variable) {
    let output_data = output.data();
    assert!(
        output_data.numel() == 1,
        "backward() can only be called on scalar (single-element) variables, got shape {:?}",
        output_data.shape()
    );

    // Step 1: Seed the output gradient to 1.0
    output.set_grad(Tensor::ones(output_data.shape()));

    // Step 2: Topological sort (output first, leaves last)
    let sorted = topological_sort(output);

    // Step 3: Walk in order (output → intermediates → leaves), propagate gradients
    for var in sorted.iter() {
        // Extract the gradient and grad_fn info, releasing the borrow before accumulation.
        // This avoids borrowing conflicts since accumulate_grad needs borrow_mut.
        let grad_result = {
            let inner = var.inner.borrow();
            match (&inner.grad, &inner.grad_fn) {
                (Some(grad), Some(grad_fn)) => {
                    let g = grad.clone();
                    let inputs = grad_fn.inputs();
                    let input_grads = grad_fn.backward(&g);
                    Some((inputs, input_grads))
                }
                _ => None,
            }
        }; // immutable borrow released here

        if let Some((inputs, input_grads)) = grad_result {
            for (input_var, input_grad) in inputs.into_iter().zip(input_grads) {
                if input_var.requires_grad() {
                    input_var.accumulate_grad(&input_grad);
                }
            }
        }
    }
}

/// Computes topological sort of the computation graph via DFS post-order.
///
/// Returns variables in reverse post-order (output first, leaves last),
/// which is the correct order for backward gradient propagation.
///
/// ## Why DFS Post-Order?
///
/// DFS post-order visits children before parents. Reversing this gives
/// parents before children — exactly what we need: process the output's
/// gradient first, then propagate to its inputs.
fn topological_sort(output: &Variable) -> Vec<Variable> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut sorted = Vec::new();

    dfs(output, &mut visited, &mut sorted);
    sorted.reverse(); // reverse post-order = output first
    sorted
}

/// DFS helper: visits children first, then pushes the current node (post-order).
fn dfs(
    var: &Variable,
    visited: &mut HashSet<usize>,
    sorted: &mut Vec<Variable>,
) {
    let ptr = Rc::as_ptr(&var.inner) as usize;
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    // Visit children (inputs to this operation)
    let children = {
        let inner = var.inner.borrow();
        inner.grad_fn.as_ref().map(|gf| gf.inputs())
    };

    if let Some(inputs) = children {
        for input in &inputs {
            dfs(input, visited, sorted);
        }
    }

    sorted.push(var.clone());
}
