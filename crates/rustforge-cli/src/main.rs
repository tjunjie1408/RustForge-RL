use std::env;
use std::fs;
use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;
use anyhow::Result;
use std::collections::HashSet;

// A simple recursive DFS to print the graph using Mermaid notation.
fn write_graph(var: &Variable, out: &mut String, visited: &mut HashSet<usize>) {
    let ptr = var.id();
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    let has_grad_fn = var.has_grad_fn();
    let shape = var.shape();

    // Output node declaration
    if has_grad_fn {
        out.push_str(&format!("    Var{ptr}[Result {:?}]\n", shape));
    } else {
        out.push_str(&format!("    Var{ptr}[Variable {:?}]\n", shape));
    }

    if let Some(inputs) = var.graph_inputs() {
        for input in inputs.iter() {
            let in_ptr = input.id();
            out.push_str(&format!("    Var{in_ptr} -->|Op| Var{ptr}\n"));
            write_graph(input, out, visited);
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let default_path = "target/graph.md".to_string();
    let output_path = if args.len() > 1 {
        &args[1]
    } else {
        &default_path
    };

    let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
    let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]), true);
    let c = &a + &b;
    let d = &c * &a;
    let e = d.sum();

    let mut markdown = String::from("# Computation Graph\n\n```mermaid\ngraph TD\n");
    let mut visited = HashSet::new();

    write_graph(&e, &mut markdown, &mut visited);

    markdown.push_str("```\n");

    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    fs::write(output_path, markdown)?;
    println!("Generated {}", output_path);

    Ok(())
}
