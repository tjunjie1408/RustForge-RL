//! Graphviz DOT format exporter for computation graphs.

use crate::variable::Variable;
use std::collections::HashSet;
use std::rc::Rc;

/// Exports the computation graph backward from `root` to Graphviz DOT format.
///
/// Nodes represent Variables, colored by their properties:
/// - Light green: Leaf nodes requiring gradient
/// - Light blue: Intermediate nodes with a `GradFn`
/// - Light gray: Nodes that do not require gradient
///
/// Ensures no infinite loops even if cycles were to exist, by using a `HashSet` to
/// track visited nodes.
pub fn export_dot(root: &Variable) -> String {
    let mut dot = String::new();
    dot.push_str("digraph ComputationGraph {\n");
    dot.push_str("    rankdir=BT;\n");
    dot.push_str("    node [shape=box, style=filled, fillcolor=lightgray];\n");

    let mut visited_nodes = HashSet::new();
    let mut visited_edges = HashSet::new();
    let mut queue = vec![root.clone()];

    while let Some(var) = queue.pop() {
        let var_id = Rc::as_ptr(&var.inner) as usize;
        if visited_nodes.contains(&var_id) {
            continue;
        }
        visited_nodes.insert(var_id);

        let shape_str = var
            .shape()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");

        let label = if var.has_grad_fn() {
            format!("Op\\n[{}]", shape_str)
        } else {
            format!("Leaf\\n[{}]", shape_str)
        };

        let color = if var.requires_grad() {
            if var.has_grad_fn() {
                "lightblue"
            } else {
                "lightgreen"
            }
        } else {
            "lightgray"
        };

        dot.push_str(&format!(
            "    v{} [label=\"{}\", fillcolor=\"{}\"];\n",
            var_id, label, color
        ));

        let inputs = {
            let inner = var.inner.borrow();
            inner.grad_fn.as_ref().map(|grad_fn| grad_fn.inputs())
        };

        if let Some(inputs) = inputs {
            for input in inputs {
                let input_id = Rc::as_ptr(&input.inner) as usize;
                let edge = (input_id, var_id);
                if !visited_edges.contains(&edge) {
                    // Arrow points from input to output (forward flow)
                    dot.push_str(&format!("    v{} -> v{};\n", input_id, var_id));
                    visited_edges.insert(edge);
                }
                queue.push(input);
            }
        }
    }

    dot.push_str("}\n");
    dot
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustforge_tensor::Tensor;

    #[test]
    fn test_export_dot_cycle_avoidance() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]), true);
        let b = &a + &a; // Same input used twice -> diamond graph (a -> b, a -> b)

        let dot = export_dot(&b);

        // Assert it exported without infinite loop and contains definitions for both
        assert!(dot.contains("Leaf\\n[1]"));
        assert!(dot.contains("Op\\n[1]"));
        assert!(dot.contains("digraph ComputationGraph"));

        let a_id = Rc::as_ptr(&a.inner) as usize;
        let count = dot.matches(&format!("v{} [label=", a_id)).count();
        assert_eq!(
            count, 1,
            "Node a should only be defined once in the DOT file"
        );
    }
}
