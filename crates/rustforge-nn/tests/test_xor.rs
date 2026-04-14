//! Phase 2 Milestone Test: 2-layer MLP learns XOR.
//!
//! XOR is a classic non-linearly-separable problem that requires at least
//! one hidden layer with a nonlinear activation to solve. If this test passes,
//! it validates the entire stack: Linear, ReLU, Sigmoid, MSE loss, backward,
//! and optimizer.
//!
//! ## XOR Truth Table
//! | x1 | x2 | y |
//! |----|----|----|
//! | 0  | 0  | 0  |
//! | 0  | 1  | 1  |
//! | 1  | 0  | 1  |
//! | 1  | 1  | 0  |

use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::{mse_loss, Linear, Module, ReLU, Sequential, Sigmoid};
use rustforge_tensor::Tensor;

#[test]
fn test_xor_training() {
    // ================================================================
    // Model: Linear(2, 16) → ReLU → Linear(16, 1) → Sigmoid
    // ================================================================
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 16)),
        Box::new(ReLU),
        Box::new(Linear::new(16, 1)),
        Box::new(Sigmoid),
    ]);

    let mut optimizer = Adam::new(model.parameters(), 0.01);

    // ================================================================
    // XOR dataset
    // ================================================================
    let inputs = Variable::new(
        Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]),
        false,
    );
    let targets = Variable::new(Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[4, 1]), false);

    // ================================================================
    // Training loop
    // ================================================================
    let mut final_loss = f32::MAX;
    for _epoch in 0..2000 {
        optimizer.zero_grad();

        let output = model.forward(&inputs);
        let loss = mse_loss(&output, &targets);
        final_loss = loss.data().item();

        loss.backward();
        optimizer.step();

        // Early stopping if loss is very small
        if final_loss < 0.001 {
            break;
        }
    }

    // ================================================================
    // Validation: check each XOR output
    // ================================================================
    let output = model.forward(&inputs);
    let preds = output.data().to_vec();

    assert!(
        final_loss < 0.01,
        "XOR training failed: final loss {:.4} should be < 0.01",
        final_loss
    );

    let tolerance = 0.15;
    assert!(
        preds[0] < tolerance,
        "XOR(0,0) = {:.3}, expected ≈ 0.0",
        preds[0]
    );
    assert!(
        (preds[1] - 1.0).abs() < tolerance,
        "XOR(0,1) = {:.3}, expected ≈ 1.0",
        preds[1]
    );
    assert!(
        (preds[2] - 1.0).abs() < tolerance,
        "XOR(1,0) = {:.3}, expected ≈ 1.0",
        preds[2]
    );
    assert!(
        preds[3] < tolerance,
        "XOR(1,1) = {:.3}, expected ≈ 0.0",
        preds[3]
    );

    println!("✅ XOR training successful!");
    println!("   Final loss: {:.6}", final_loss);
    println!(
        "   Predictions: [{:.3}, {:.3}, {:.3}, {:.3}]",
        preds[0], preds[1], preds[2], preds[3]
    );
    println!("   Expected:    [0.0,   1.0,   1.0,   0.0  ]");
}
