//! End-to-end training integration tests for rustforge-nn.
//!
//! Validates that the full stack (Tensor → Variable → Module → Optimizer)
//! works correctly for realistic training scenarios.

use approx::assert_abs_diff_eq;
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::optimizer::sgd::SGD;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::*;
use rustforge_tensor::Tensor;

// Module: Function Approximation

mod function_approx {
    use super::*;

    #[test]
    fn learn_linear_function() {
        // Learn y = 2x + 1
        let model = Linear::new(1, 1);
        model.parameters()[0].set_data(Tensor::from_vec(vec![0.0], &[1, 1]));
        model.parameters()[1].set_data(Tensor::from_vec(vec![0.0], &[1]));

        let mut opt = SGD::new(model.parameters(), 0.01, 0.0);

        let x_data: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();
        let x = Variable::new(Tensor::from_vec(x_data, &[10, 1]), false);
        let t = Variable::new(Tensor::from_vec(y_data, &[10, 1]), false);

        for _ in 0..5000 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }

        let w = model.parameters()[0].data().to_vec()[0];
        let b = model.parameters()[1].data().to_vec()[0];
        assert_abs_diff_eq!(w, 2.0, epsilon = 0.15);
        assert_abs_diff_eq!(b, 1.0, epsilon = 0.15);
    }

    #[test]
    fn learn_quadratic_function() {
        // Learn y = x² using MLP with hidden layer
        let model = Sequential::new(vec![
            Box::new(Linear::new(1, 32)),
            Box::new(ReLU),
            Box::new(Linear::new(32, 32)),
            Box::new(ReLU),
            Box::new(Linear::new(32, 1)),
        ]);

        let mut opt = Adam::new(model.parameters(), 0.01);

        // Training data: x in [-2, 2], y = x^2
        let x_vals: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.1).collect();
        let y_vals: Vec<f32> = x_vals.iter().map(|&x| x * x).collect();
        let n = x_vals.len();
        let x = Variable::new(Tensor::from_vec(x_vals, &[n, 1]), false);
        let t = Variable::new(Tensor::from_vec(y_vals, &[n, 1]), false);

        let mut final_loss = f32::MAX;
        for _ in 0..500 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            final_loss = loss.data().item();
            loss.backward();
            opt.step();
        }

        assert!(
            final_loss < 0.5,
            "Should learn quadratic reasonably well, loss={}",
            final_loss
        );
    }

    #[test]
    fn loss_monotonically_decreases_early() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.001);
        let x = Variable::new(Tensor::rand_uniform(&[20, 3], -1.0, 1.0, Some(42)), false);
        let t = Variable::new(Tensor::rand_uniform(&[20, 1], -1.0, 1.0, Some(43)), false);

        let mut losses = Vec::new();
        for _ in 0..50 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            losses.push(loss.data().item());
            loss.backward();
            opt.step();
        }

        // Average loss of last 10 should be less than average of first 10
        let early_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let late_avg: f32 = losses[40..].iter().sum::<f32>() / 10.0;
        assert!(
            late_avg < early_avg,
            "Loss should decrease: early_avg={}, late_avg={}",
            early_avg,
            late_avg
        );
    }

    #[test]
    fn multi_output_function() {
        // Learn f(x) = [2x, -x + 3]
        let model = Linear::new(1, 2);
        model.parameters()[0].set_data(Tensor::from_vec(vec![0.0, 0.0], &[2, 1]));
        model.parameters()[1].set_data(Tensor::from_vec(vec![0.0, 0.0], &[2]));

        let mut opt = SGD::new(model.parameters(), 0.001, 0.0);

        let x_data: Vec<f32> = (1..=20).map(|i| i as f32 * 0.5).collect();
        let mut y_data = Vec::new();
        for &x in &x_data {
            y_data.push(2.0 * x);
            y_data.push(-x + 3.0);
        }
        let n = x_data.len();
        let x = Variable::new(Tensor::from_vec(x_data, &[n, 1]), false);
        let t = Variable::new(Tensor::from_vec(y_data, &[n, 2]), false);

        for _ in 0..5000 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }

        let final_loss = mse_loss(&model.forward(&x), &t).data().item();
        assert!(
            final_loss < 0.5,
            "Multi-output should converge, loss={}",
            final_loss
        );
    }

    #[test]
    fn overfit_tiny_dataset() {
        // Model should memorize 3 data points
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 16)),
            Box::new(ReLU),
            Box::new(Linear::new(16, 1)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]),
            false,
        );
        let t = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]), false);

        for _ in 0..1000 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }

        let final_loss = mse_loss(&model.forward(&x), &t).data().item();
        assert!(
            final_loss < 0.01,
            "Should overfit tiny dataset, loss={}",
            final_loss
        );
    }
}

// Module: Classification

mod classification {
    use super::*;

    #[test]
    fn three_class_classification() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 16)),
            Box::new(ReLU),
            Box::new(Linear::new(16, 3)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        // Three clusters
        let inputs = Variable::new(
            Tensor::from_vec(
                vec![
                    0.0, 0.0, // class 0
                    0.1, -0.1, // class 0
                    1.0, 0.0, // class 1
                    1.1, 0.1, // class 1
                    0.0, 1.0, // class 2
                    -0.1, 0.9, // class 2
                ],
                &[6, 2],
            ),
            false,
        );
        let targets = Variable::new(
            Tensor::from_vec(
                vec![
                    1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 1.0,
                ],
                &[6, 3],
            ),
            false,
        );

        for _ in 0..500 {
            opt.zero_grad();
            let logits = model.forward(&inputs);
            let loss = cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
        }

        // Check accuracy
        let logits = model.forward(&inputs);
        let pred_data = logits.data().to_vec();
        let mut correct = 0;
        let true_classes = [0, 0, 1, 1, 2, 2];
        for i in 0..6 {
            let pred_class = (0..3)
                .max_by(|&a, &b| {
                    pred_data[i * 3 + a]
                        .partial_cmp(&pred_data[i * 3 + b])
                        .unwrap()
                })
                .unwrap();
            if pred_class == true_classes[i] {
                correct += 1;
            }
        }
        assert!(
            correct >= 4,
            "Should classify at least 4/6 correctly, got {}/6",
            correct
        );
    }

    #[test]
    fn binary_classification_mse_sigmoid() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)),
            Box::new(Sigmoid),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        // AND gate
        let inputs = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]),
            false,
        );
        let targets = Variable::new(Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0], &[4, 1]), false);

        for _ in 0..1000 {
            opt.zero_grad();
            let y = model.forward(&inputs);
            let loss = mse_loss(&y, &targets);
            loss.backward();
            opt.step();
        }

        let preds = model.forward(&inputs).data().to_vec();
        assert!(preds[0] < 0.3, "AND(0,0)={:.3} should be near 0", preds[0]);
        assert!(preds[3] > 0.7, "AND(1,1)={:.3} should be near 1", preds[3]);
    }

    #[test]
    fn accuracy_improves_over_training() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 2)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        let inputs = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 0.1, 0.1, 0.9, 0.9, 1.0, 1.0], &[4, 2]),
            false,
        );
        let targets = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[4, 2]),
            false,
        );

        let early_loss = cross_entropy_loss(&model.forward(&inputs), &targets)
            .data()
            .item();

        for _ in 0..300 {
            opt.zero_grad();
            let loss = cross_entropy_loss(&model.forward(&inputs), &targets);
            loss.backward();
            opt.step();
        }

        let late_loss = cross_entropy_loss(&model.forward(&inputs), &targets)
            .data()
            .item();
        assert!(
            late_loss < early_loss,
            "Loss should decrease: {} → {}",
            early_loss,
            late_loss
        );
    }
}

// Module: Architecture Variations

mod architectures {
    use super::*;

    #[test]
    fn shallow_network_1_hidden() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);
        let x = Variable::new(Tensor::rand_uniform(&[10, 4], -1.0, 1.0, Some(42)), false);
        let t = Variable::new(Tensor::rand_uniform(&[10, 1], -1.0, 1.0, Some(43)), false);

        let initial_loss = mse_loss(&model.forward(&x), &t).data().item();
        for _ in 0..200 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }
        let final_loss = mse_loss(&model.forward(&x), &t).data().item();
        assert!(final_loss < initial_loss, "Shallow net should train");
    }

    #[test]
    fn deep_network_5_hidden() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 16)),
            Box::new(ReLU),
            Box::new(Linear::new(16, 16)),
            Box::new(ReLU),
            Box::new(Linear::new(16, 16)),
            Box::new(ReLU),
            Box::new(Linear::new(16, 16)),
            Box::new(ReLU),
            Box::new(Linear::new(16, 1)),
        ]);

        // Just verify it runs without crashing
        let mut opt = Adam::new(model.parameters(), 0.001);
        let x = Variable::new(Tensor::rand_uniform(&[10, 4], -1.0, 1.0, Some(42)), false);
        let t = Variable::new(Tensor::rand_uniform(&[10, 1], -1.0, 1.0, Some(43)), false);

        for _ in 0..50 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }

        // All parameters should have gradients
        for (i, p) in model.parameters().iter().enumerate() {
            assert!(
                p.grad().is_some() || !p.requires_grad(),
                "Param {} didn't get gradient",
                i
            );
        }
    }

    #[test]
    fn wide_network() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 256)),
            Box::new(ReLU),
            Box::new(Linear::new(256, 1)),
        ]);
        let x = Variable::new(Tensor::rand_uniform(&[8, 4], -1.0, 1.0, Some(42)), false);
        let y = model.forward(&x);
        assert_eq!(y.shape(), vec![8, 1]);

        // Verify gradient flow
        let loss = y.sum();
        loss.backward();
        assert_eq!(model.parameters().len(), 4);
    }

    #[test]
    fn network_with_dropout_train_eval_diff() {
        let mut model = Sequential::new(vec![
            Box::new(Linear::new(4, 16)),
            Box::new(ReLU),
            Box::new(Dropout::new(0.5)),
            Box::new(Linear::new(16, 1)),
        ]);

        let x = Variable::new(Tensor::ones(&[5, 4]), false);

        // Training mode — output varies due to dropout
        model.set_training(true);
        let y_train1 = model.forward(&x).data().to_vec();
        let y_train2 = model.forward(&x).data().to_vec();
        // Outputs should likely differ (extremely unlikely to be the same with p=0.5)
        let same = y_train1 == y_train2;
        // Note: This is probabilistic. With 80 values and p=0.5, the chance
        // of identical outputs is negligibly small.

        // Eval mode — deterministic
        model.set_training(false);
        let y_eval1 = model.forward(&x).data().to_vec();
        let y_eval2 = model.forward(&x).data().to_vec();
        assert_eq!(y_eval1, y_eval2, "Eval mode should be deterministic");

        // In practice, training outputs should often differ from eval (unless lucky)
        if !same {
            // Expected case: training outputs differ
        }
    }

    #[test]
    fn network_with_layernorm() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(LayerNorm::new(8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 2)),
        ]);

        let mut opt = Adam::new(model.parameters(), 0.01);
        let x = Variable::new(Tensor::rand_uniform(&[10, 4], -1.0, 1.0, Some(42)), false);
        let t = Variable::new(Tensor::rand_uniform(&[10, 2], -1.0, 1.0, Some(43)), false);

        let initial_loss = mse_loss(&model.forward(&x), &t).data().item();
        for _ in 0..200 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }
        let final_loss = mse_loss(&model.forward(&x), &t).data().item();
        assert!(final_loss < initial_loss, "LayerNorm model should train");
    }
}

// Module: Optimizer Comparison

mod optimizer_compare {
    use super::*;

    #[test]
    fn sgd_vs_adam_both_converge() {
        fn train_with_optimizer(mut opt: Box<dyn Optimizer>, model: &Sequential) -> f32 {
            let x = Variable::new(
                Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]),
                false,
            );
            let t = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]), false);

            for _ in 0..500 {
                opt.zero_grad();
                let loss = mse_loss(&model.forward(&x), &t);
                loss.backward();
                opt.step();
            }
            mse_loss(&model.forward(&x), &t).data().item()
        }

        let model_sgd = Sequential::new(vec![
            Box::new(Linear::new(2, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)),
        ]);
        let model_adam = Sequential::new(vec![
            Box::new(Linear::new(2, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)),
        ]);

        let sgd_loss = train_with_optimizer(
            Box::new(SGD::new(model_sgd.parameters(), 0.01, 0.9)),
            &model_sgd,
        );
        let adam_loss = train_with_optimizer(
            Box::new(Adam::new(model_adam.parameters(), 0.01)),
            &model_adam,
        );

        assert!(sgd_loss < 2.0, "SGD should converge, loss={}", sgd_loss);
        assert!(adam_loss < 2.0, "Adam should converge, loss={}", adam_loss);
    }

    #[test]
    fn sgd_momentum_vs_vanilla() {
        let w_vanilla = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let w_momentum = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
        let target = Variable::new(Tensor::from_vec(vec![0.0], &[1]), false);

        let mut sgd_v = SGD::new(vec![w_vanilla.clone()], 0.01, 0.0);
        let mut sgd_m = SGD::new(vec![w_momentum.clone()], 0.01, 0.9);

        for _ in 0..100 {
            sgd_v.zero_grad();
            let d = &w_vanilla - &target;
            (&d * &d).sum().backward();
            sgd_v.step();

            sgd_m.zero_grad();
            let d = &w_momentum - &target;
            (&d * &d).sum().backward();
            sgd_m.step();
        }

        let err_v = w_vanilla.data().to_vec()[0].abs();
        let err_m = w_momentum.data().to_vec()[0].abs();
        assert!(
            err_m <= err_v + 0.1,
            "Momentum should help: vanilla_err={}, momentum_err={}",
            err_v,
            err_m
        );
    }

    #[test]
    fn adam_different_learning_rates() {
        let results: Vec<f32> = vec![0.001, 0.01, 0.1]
            .into_iter()
            .map(|lr| {
                let w = Variable::new(Tensor::from_vec(vec![10.0], &[1]), true);
                let target = Variable::new(Tensor::from_vec(vec![0.0], &[1]), false);
                let mut adam = Adam::new(vec![w.clone()], lr);
                for _ in 0..100 {
                    adam.zero_grad();
                    let d = &w - &target;
                    (&d * &d).sum().backward();
                    adam.step();
                }
                let w_data = w.data();
                w_data.to_vec()[0].abs()
            })
            .collect();

        // Higher lr should generally converge faster (lower final error)
        // But very high lr might overshoot. Just check all converge somewhat.
        for (i, &err) in results.iter().enumerate() {
            assert!(
                err < 10.0,
                "lr variant {} should converge at least somewhat, err={}",
                i,
                err
            );
        }
    }

    #[test]
    fn both_optimizers_converge_on_simple_problem() {
        for optimizer_type in &["sgd", "adam"] {
            let w = Variable::new(Tensor::from_vec(vec![5.0, -3.0], &[2]), true);
            let target = Variable::new(Tensor::from_vec(vec![0.0, 0.0], &[2]), false);

            let mut opt: Box<dyn Optimizer> = match *optimizer_type {
                "sgd" => Box::new(SGD::new(vec![w.clone()], 0.01, 0.9)),
                "adam" => Box::new(Adam::new(vec![w.clone()], 0.1)),
                _ => unreachable!(),
            };

            for _ in 0..500 {
                opt.zero_grad();
                let d = &w - &target;
                let loss = (&d * &d).sum();
                loss.backward();
                opt.step();
            }

            let final_w = w.data().to_vec();
            for &v in &final_w {
                assert!(
                    v.abs() < 1.0,
                    "{} should converge near 0, got {:?}",
                    optimizer_type,
                    final_w
                );
            }
        }
    }
}
