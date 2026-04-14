//! Integration test: Linear regression learns y = 2x + 1.
//!
//! Validates that a single Linear layer with MSE loss and SGD optimizer
//! can learn a simple linear function from data.

use rustforge_autograd::optimizer::sgd::SGD;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::{mse_loss, Linear, Module};
use rustforge_tensor::Tensor;

use approx::assert_abs_diff_eq;

#[test]
fn test_linear_regression() {
    // ================================================================
    // Model: Linear(1, 1) — learns y = wx + b
    // ================================================================
    let model = Linear::new(1, 1);

    // Initialize with poor values to verify learning
    model.parameters()[0].set_data(Tensor::from_vec(vec![0.0], &[1, 1])); // w = 0
    model.parameters()[1].set_data(Tensor::from_vec(vec![0.0], &[1])); // b = 0

    let mut optimizer = SGD::new(model.parameters(), 0.01, 0.0);

    // ================================================================
    // Training data: y = 2x + 1
    // ================================================================
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let inputs = Variable::new(Tensor::from_vec(x_data, &[8, 1]), false);
    let targets = Variable::new(Tensor::from_vec(y_data, &[8, 1]), false);

    // ================================================================
    // Training loop
    // ================================================================
    for _epoch in 0..5000 {
        optimizer.zero_grad();

        let output = model.forward(&inputs);
        let loss = mse_loss(&output, &targets);

        loss.backward();
        optimizer.step();
    }

    // ================================================================
    // Validation: weight ≈ 2.0, bias ≈ 1.0
    // ================================================================
    let weight = model.parameters()[0].data().to_vec()[0];
    let bias = model.parameters()[1].data().to_vec()[0];

    println!("✅ Linear regression result:");
    println!("   Learned: y = {:.4}x + {:.4}", weight, bias);
    println!("   Expected: y = 2.0000x + 1.0000");

    assert_abs_diff_eq!(weight, 2.0, epsilon = 0.1);
    assert_abs_diff_eq!(bias, 1.0, epsilon = 0.1);
}

#[test]
fn test_cross_entropy_classification() {
    // ================================================================
    // Simple classification: learn to separate two classes
    // ================================================================
    use rustforge_autograd::optimizer::adam::Adam;
    use rustforge_nn::{cross_entropy_loss, ReLU, Sequential, Softmax};

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 2)),
    ]);

    let mut optimizer = Adam::new(model.parameters(), 0.01);

    // Class 0: points near (0, 0), Class 1: points near (1, 1)
    let inputs = Variable::new(
        Tensor::from_vec(
            vec![
                0.1, 0.1, // class 0
                0.2, 0.0, // class 0
                0.0, 0.2, // class 0
                0.9, 0.9, // class 1
                0.8, 1.0, // class 1
                1.0, 0.8, // class 1
            ],
            &[6, 2],
        ),
        false,
    );
    // One-hot targets
    let targets = Variable::new(
        Tensor::from_vec(
            vec![
                1.0, 0.0, // class 0
                1.0, 0.0, // class 0
                1.0, 0.0, // class 0
                0.0, 1.0, // class 1
                0.0, 1.0, // class 1
                0.0, 1.0, // class 1
            ],
            &[6, 2],
        ),
        false,
    );

    let mut final_loss = f32::MAX;
    for _epoch in 0..1000 {
        optimizer.zero_grad();
        let logits = model.forward(&inputs);
        let loss = cross_entropy_loss(&logits, &targets);
        final_loss = loss.data().item();
        loss.backward();
        optimizer.step();

        if final_loss < 0.01 {
            break;
        }
    }

    // Apply softmax to get predictions
    let softmax = Softmax;
    let logits = model.forward(&inputs);
    let probs = softmax.forward(&logits);
    let pred_data = probs.data().to_vec();

    println!("✅ Classification result:");
    println!("   Final loss: {:.6}", final_loss);
    for i in 0..6 {
        let p0 = pred_data[i * 2];
        let p1 = pred_data[i * 2 + 1];
        let pred_class = if p0 > p1 { 0 } else { 1 };
        let true_class = if i < 3 { 0 } else { 1 };
        println!(
            "   Sample {}: pred=[{:.3}, {:.3}] → class {} (true: {})",
            i, p0, p1, pred_class, true_class
        );
    }

    assert!(
        final_loss < 0.1,
        "Classification loss {:.4} should be < 0.1",
        final_loss
    );
}
