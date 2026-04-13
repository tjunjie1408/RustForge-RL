use std::fs;
use rustforge_autograd::Variable;
use rustforge_tensor::Tensor;

fn main() {
    let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
    let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]), true);
    let _c = &a + &b;

    let markdown = r#"
# Computation Graph

```mermaid
graph TD
    A[Variable A] -->|Add| C[Result C]
    B[Variable B] -->|Add| C
```
"#;
    fs::write("graph.md", markdown).unwrap();
    println!("Generated graph.md");
}
