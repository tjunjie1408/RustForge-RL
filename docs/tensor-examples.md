# Rust tensor examples

### Basic Usage

```rust
use rustforge_tensor::Tensor;

fn main() {
    // Create tensors
    let weights = Tensor::xavier_uniform(&[128, 64], Some(42));
    let input = Tensor::rand_normal(&[32, 64], 0.0, 1.0, None);
    let bias = Tensor::zeros(&[128]);

    // Forward pass: output = input @ weights^T + bias
    let output = input.matmul(&weights.t()) + bias;

    // Activation
    let activated = output.relu();

    // Softmax for probability distribution
    let probs = activated.softmax(1).unwrap();

    println!("Output shape: {:?}", probs.shape());
    println!("Probabilities:\n{}", probs);
}
```

### Tensor Operations Examples

```rust
use rustforge_tensor::Tensor;

// Broadcasting: [3, 1] + [1, 4] → [3, 4]
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]);
let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]);
let c = &a + &b;  // shape: [3, 4]

// Reductions
let data = Tensor::rand_uniform(&[100, 50], 0.0, 1.0, Some(0));
println!("Mean: {:.4}", data.mean().item());
println!("Std:  {:.4}", data.std_dev().item());

// Matrix multiplication
let q = Tensor::randn(&[8, 64], Some(1));  // queries
let k = Tensor::randn(&[8, 64], Some(2));  // keys
let attention = q.matmul(&k.t());           // [8, 8] attention scores
let weights = (& attention / 8.0_f32.sqrt()).softmax(1).unwrap();
```
