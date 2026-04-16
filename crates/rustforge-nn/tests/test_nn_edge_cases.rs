//! Edge case and robustness tests for rustforge-nn.
//!
//! Covers: serialization round-trips, training stress tests,
//! and cross-module interaction tests.

use approx::assert_abs_diff_eq;
use rustforge_autograd::optimizer::adam::Adam;
use rustforge_autograd::optimizer::sgd::SGD;
use rustforge_autograd::{Optimizer, Variable};
use rustforge_nn::*;
use rustforge_tensor::Tensor;
use std::fs;

// Module: Serialization Round-Trip

mod serialization {
    use super::*;

    #[test]
    fn save_load_preserves_weights() {
        let model = Linear::new(4, 3);
        let path = "test_ser_weights.bin";

        save_parameters(&model, path).unwrap();
        let model2 = Linear::new(4, 3);
        load_parameters(&model2, path).unwrap();

        let p1 = model.parameters();
        let p2 = model2.parameters();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.data().to_vec(), b.data().to_vec());
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn save_load_sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 2)),
        ]);
        let path = "test_ser_sequential.bin";

        save_parameters(&model, path).unwrap();
        let model2 = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 2)),
        ]);
        load_parameters(&model2, path).unwrap();

        let p1 = model.parameters();
        let p2 = model2.parameters();
        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.data().to_vec(), b.data().to_vec());
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_wrong_param_count_errors() {
        let model1 = Linear::new(4, 3); // 2 params
        let path = "test_ser_count_mismatch.bin";
        save_parameters(&model1, path).unwrap();

        let model2 = Linear::no_bias(5, 3); // 1 param
        let result = load_parameters(&model2, path);
        assert!(result.is_err());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_wrong_shape_errors() {
        let model1 = Linear::new(4, 3);
        let path = "test_ser_shape_mismatch.bin";
        save_parameters(&model1, path).unwrap();

        let model2 = Linear::new(5, 3); // Different in_features
        let result = load_parameters(&model2, path);
        assert!(result.is_err());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_nonexistent_file_errors() {
        let model = Linear::new(4, 3);
        let result = load_parameters(&model, "nonexistent_model_file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn load_corrupted_file_errors() {
        let path = "test_ser_corrupted.bin";
        fs::write(path, b"not valid bincode data!!!").unwrap();

        let model = Linear::new(4, 3);
        let result = load_parameters(&model, path);
        assert!(result.is_err());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn multiple_save_load_cycles() {
        let model = Linear::new(3, 2);
        model.parameters()[0].set_data(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        model.parameters()[1].set_data(Tensor::from_vec(vec![0.1, 0.2], &[2]));

        let path = "test_ser_multi_cycle.bin";

        for _ in 0..5 {
            save_parameters(&model, path).unwrap();
            let model2 = Linear::new(3, 2);
            load_parameters(&model2, path).unwrap();

            let w = model2.parameters()[0].data().to_vec();
            assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-6);
            assert_abs_diff_eq!(w[5], 6.0, epsilon = 1e-6);
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn round_trip_preserves_inference() {
        let model = Linear::new(3, 2);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let y_before = model.forward(&x).data().to_vec();

        let path = "test_ser_inference.bin";
        save_parameters(&model, path).unwrap();

        let model2 = Linear::new(3, 2);
        load_parameters(&model2, path).unwrap();
        let y_after = model2.forward(&x).data().to_vec();

        assert_eq!(y_before, y_after);

        let _ = fs::remove_file(path);
    }
}

// Module: Training Stress Tests

mod training_stress {
    use super::*;

    #[test]
    fn training_100_epochs_no_panic() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);
        let x = Variable::new(Tensor::rand_uniform(&[10, 4], -1.0, 1.0, Some(42)), false);
        let t = Variable::new(Tensor::rand_uniform(&[10, 1], -1.0, 1.0, Some(43)), false);

        for _ in 0..100 {
            opt.zero_grad();
            let y = model.forward(&x);
            let loss = mse_loss(&y, &t);
            loss.backward();
            opt.step();
        }
    }

    #[test]
    fn training_with_all_zero_inputs() {
        let model = Linear::new(3, 2);
        let mut opt = SGD::new(model.parameters(), 0.01, 0.0);
        let x = Variable::new(Tensor::zeros(&[4, 3]), false);
        let t = Variable::new(Tensor::ones(&[4, 2]), false);

        for _ in 0..50 {
            opt.zero_grad();
            let y = model.forward(&x);
            let loss = mse_loss(&y, &t);
            loss.backward();
            opt.step();
        }
        // Should not panic or produce NaN
        let y = model.forward(&x);
        for &v in y.data().to_vec().iter() {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn training_with_same_targets() {
        let model = Linear::new(2, 1);
        let mut opt = SGD::new(model.parameters(), 0.01, 0.0);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], &[4, 2]),
            false,
        );
        // All targets are the same
        let t = Variable::new(Tensor::full(&[4, 1], 3.0), false);

        for _ in 0..200 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }
        // Loss should have decreased
        let final_loss = mse_loss(&model.forward(&x), &t).data().item();
        assert!(
            final_loss < 1.0,
            "Loss should have decreased, got {}",
            final_loss
        );
    }

    #[test]
    fn training_batch_size_1() {
        let model = Linear::new(3, 1);
        let mut opt = Adam::new(model.parameters(), 0.01);
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]), false);
        let t = Variable::new(Tensor::from_vec(vec![5.0], &[1, 1]), false);

        for _ in 0..500 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }
        // Should converge
        let pred = model.forward(&x).data().item();
        assert!(
            (pred - 5.0).abs() < 1.0,
            "Should converge near target, got {}",
            pred
        );
    }

    #[test]
    fn training_very_large_lr_no_crash() {
        let model = Linear::new(2, 1);
        let mut opt = SGD::new(model.parameters(), 100.0, 0.0);
        let x = Variable::new(Tensor::ones(&[4, 2]), false);
        let t = Variable::new(Tensor::ones(&[4, 1]), false);

        for _ in 0..10 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }
        // Just verify it doesn't crash — loss may be huge or NaN
    }

    #[test]
    fn training_very_small_lr_barely_moves() {
        let model = Linear::new(2, 1);
        model.parameters()[0].set_data(Tensor::from_vec(vec![10.0, 10.0], &[1, 2]));
        model.parameters()[1].set_data(Tensor::from_vec(vec![0.0], &[1]));
        let w_before = model.parameters()[0].data().to_vec();

        let mut opt = SGD::new(model.parameters(), 1e-10, 0.0);
        let x = Variable::new(Tensor::ones(&[4, 2]), false);
        let t = Variable::new(Tensor::ones(&[4, 1]), false);

        for _ in 0..10 {
            opt.zero_grad();
            let loss = mse_loss(&model.forward(&x), &t);
            loss.backward();
            opt.step();
        }

        let w_after = model.parameters()[0].data().to_vec();
        for (b, a) in w_before.iter().zip(w_after.iter()) {
            assert!(
                (b - a).abs() < 0.001,
                "With tiny lr, weight should barely change: {} → {}",
                b,
                a
            );
        }
    }

    #[test]
    fn full_pipeline_backward_step() {
        // Complete: forward → loss → backward → step
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 3)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.001);
        let x = Variable::new(Tensor::rand_uniform(&[16, 4], -1.0, 1.0, Some(42)), false);

        let mut target_data = vec![0.0; 16 * 3];
        for i in 0..16 {
            target_data[i * 3 + (i % 3)] = 1.0;
        }
        let targets = Variable::new(Tensor::from_vec(target_data, &[16, 3]), false);

        let initial_loss = cross_entropy_loss(&model.forward(&x), &targets)
            .data()
            .item();
        for _ in 0..100 {
            opt.zero_grad();
            let logits = model.forward(&x);
            let loss = cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
        }
        let final_loss = cross_entropy_loss(&model.forward(&x), &targets)
            .data()
            .item();
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: {} → {}",
            initial_loss,
            final_loss
        );
    }
}

// Module: Cross-Module Interaction

mod cross_module {
    use super::*;

    #[test]
    fn linear_layernorm_relu_pipeline() {
        let linear = Linear::new(4, 8);
        let ln = LayerNorm::new(8);
        let relu = ReLU;

        let x = Variable::new(Tensor::rand_uniform(&[3, 4], -1.0, 1.0, Some(42)), true);
        let h = linear.forward(&x);
        let h_norm = ln.forward(&h);
        let y = relu.forward(&h_norm);

        assert_eq!(y.shape(), vec![3, 8]);

        let loss = y.sum();
        loss.backward();

        assert!(x.grad().is_some());
        for p in linear.parameters() {
            assert!(p.grad().is_some());
        }
        for p in ln.parameters() {
            assert!(p.grad().is_some());
        }
    }

    #[test]
    fn dropout_linear_train_vs_eval() {
        let linear = Linear::new(4, 4);
        let mut dropout = Dropout::new(0.5);

        let x = Variable::new(Tensor::ones(&[10, 4]), false);

        // Training mode — some values should be zeroed
        dropout.set_training(true);
        let mut has_zero = false;
        for _ in 0..10 {
            let h = linear.forward(&x);
            let y = dropout.forward(&h);
            if y.data().to_vec().contains(&0.0) {
                has_zero = true;
                break;
            }
        }
        assert!(has_zero, "Dropout should zero some values in training mode");

        // Eval mode — pass-through
        dropout.set_training(false);
        let h = linear.forward(&x);
        let y_eval = dropout.forward(&h);
        assert_eq!(h.data().to_vec(), y_eval.data().to_vec());
    }

    #[test]
    fn sequential_with_layernorm_dropout() {
        let mut model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(LayerNorm::new(8)),
            Box::new(ReLU),
            Box::new(Dropout::new(0.1)),
            Box::new(Linear::new(8, 2)),
        ]);

        let x = Variable::new(Tensor::rand_uniform(&[5, 4], -1.0, 1.0, Some(42)), false);

        // Training forward
        let y_train = model.forward(&x);
        assert_eq!(y_train.shape(), vec![5, 2]);

        // Eval forward
        model.set_training(false);
        let y_eval = model.forward(&x);
        assert_eq!(y_eval.shape(), vec![5, 2]);
    }

    #[test]
    fn huber_loss_with_sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 4)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 1)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        let x = Variable::new(Tensor::rand_uniform(&[8, 3], -1.0, 1.0, Some(42)), false);
        let t = Variable::new(Tensor::rand_uniform(&[8, 1], -1.0, 1.0, Some(43)), false);

        for _ in 0..50 {
            opt.zero_grad();
            let y = model.forward(&x);
            let loss = huber_loss(&y, &t, 1.0);
            loss.backward();
            opt.step();
        }
        // Should not crash
    }

    #[test]
    fn cross_entropy_with_layernorm_model() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 8)),
            Box::new(LayerNorm::new(8)),
            Box::new(ReLU),
            Box::new(Linear::new(8, 3)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        let x = Variable::new(Tensor::rand_uniform(&[6, 3], -1.0, 1.0, Some(42)), false);
        let targets = Variable::new(
            Tensor::from_vec(
                vec![
                    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 1.0,
                ],
                &[6, 3],
            ),
            false,
        );

        for _ in 0..100 {
            opt.zero_grad();
            let logits = model.forward(&x);
            let loss = cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
        }

        let final_loss = cross_entropy_loss(&model.forward(&x), &targets)
            .data()
            .item();
        assert!(!final_loss.is_nan(), "Loss should not be NaN");
    }

    #[test]
    fn sigmoid_output_for_binary_classification() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 1)),
            Box::new(Sigmoid),
        ]);

        let x = Variable::new(Tensor::rand_uniform(&[8, 2], -1.0, 1.0, Some(42)), false);
        let y = model.forward(&x);
        assert_eq!(y.shape(), vec![8, 1]);

        // All outputs should be in (0, 1)
        for &v in y.data().to_vec().iter() {
            assert!(v > 0.0 && v < 1.0, "Sigmoid output {} must be in (0,1)", v);
        }
    }

    #[test]
    fn mixed_loss_types_same_model() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 4)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 1)),
        ]);

        let x = Variable::new(Tensor::rand_uniform(&[8, 3], -1.0, 1.0, Some(42)), false);
        let t = Variable::new(Tensor::rand_uniform(&[8, 1], -1.0, 1.0, Some(43)), false);

        // MSE loss
        let y1 = model.forward(&x);
        let l1 = mse_loss(&y1, &t);
        assert!(!l1.data().item().is_nan());

        // Huber loss
        let y2 = model.forward(&x);
        let l2 = huber_loss(&y2, &t, 1.0);
        assert!(!l2.data().item().is_nan());
    }
}
