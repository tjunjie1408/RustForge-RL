//! Operator overloading and math operations for `Variable`.
//!
//! Each operator:
//! 1. Extracts `.data()` from input Variables
//! 2. Performs the forward computation using `Tensor` ops
//! 3. Creates the appropriate `GradFn` struct (capturing input Variables)
//! 4. Wraps the result in a new `Variable` with `grad_fn` set
//!
//! ## Ownership Design
//!
//! Operators are implemented for all combinations of `&Variable` and `Variable`:
//! ```rust,ignore
//! let c = &a + &b;   // ref + ref
//! let d = a + &b;    // val + ref (a is consumed)
//! let e = &a + b;    // ref + val (b is consumed)
//! let f = a + b;     // val + val
//! ```

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::graph::{
    AddGrad, DivGrad, ExpGrad, GradFn, LogGrad, MatmulGrad, MeanGrad, MulGrad, NegGrad, PowGrad,
    ReluGrad, ScalarAddGrad, ScalarMulGrad, SigmoidGrad, SqrtGrad, SubGrad, SumAxisGrad, SumGrad,
    TanhGrad,
};
use crate::variable::Variable;

/// Returns true if any of the given variables requires gradient tracking.
fn needs_grad(inputs: &[&Variable]) -> bool {
    inputs.iter().any(|v| v.requires_grad())
}

// ============================================================================
// Variable + Variable
// ============================================================================

impl<'b> Add<&'b Variable> for &Variable {
    type Output = Variable;

    fn add(self, rhs: &'b Variable) -> Variable {
        let result_data = &self.data() + &rhs.data();
        let requires_grad = needs_grad(&[self, rhs]);
        let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
            Some(Box::new(AddGrad {
                lhs: self.clone(),
                rhs: rhs.clone(),
            }))
        } else {
            None
        };
        Variable::from_grad_fn(result_data, requires_grad, grad_fn)
    }
}

// ============================================================================
// Variable - Variable
// ============================================================================

impl<'b> Sub<&'b Variable> for &Variable {
    type Output = Variable;

    fn sub(self, rhs: &'b Variable) -> Variable {
        let result_data = &self.data() - &rhs.data();
        let requires_grad = needs_grad(&[self, rhs]);
        let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
            Some(Box::new(SubGrad {
                lhs: self.clone(),
                rhs: rhs.clone(),
            }))
        } else {
            None
        };
        Variable::from_grad_fn(result_data, requires_grad, grad_fn)
    }
}

// ============================================================================
// Variable * Variable (element-wise)
// ============================================================================

impl<'b> Mul<&'b Variable> for &Variable {
    type Output = Variable;

    fn mul(self, rhs: &'b Variable) -> Variable {
        let lhs_data = self.data();
        let rhs_data = rhs.data();
        let result_data = &lhs_data * &rhs_data;
        let requires_grad = needs_grad(&[self, rhs]);
        let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
            Some(Box::new(MulGrad {
                lhs: self.clone(),
                rhs: rhs.clone(),
                lhs_data,
                rhs_data,
            }))
        } else {
            None
        };
        Variable::from_grad_fn(result_data, requires_grad, grad_fn)
    }
}

// ============================================================================
// Variable / Variable
// ============================================================================

impl<'b> Div<&'b Variable> for &Variable {
    type Output = Variable;

    fn div(self, rhs: &'b Variable) -> Variable {
        let lhs_data = self.data();
        let rhs_data = rhs.data();
        let result_data = &lhs_data / &rhs_data;
        let requires_grad = needs_grad(&[self, rhs]);
        let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
            Some(Box::new(DivGrad {
                lhs: self.clone(),
                rhs: rhs.clone(),
                lhs_data,
                rhs_data,
            }))
        } else {
            None
        };
        Variable::from_grad_fn(result_data, requires_grad, grad_fn)
    }
}

// ============================================================================
// -Variable (negation)
// ============================================================================

impl Neg for &Variable {
    type Output = Variable;

    fn neg(self) -> Variable {
        let result_data = self.data().neg();
        let requires_grad = self.requires_grad();
        let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
            Some(Box::new(NegGrad {
                input: self.clone(),
            }))
        } else {
            None
        };
        Variable::from_grad_fn(result_data, requires_grad, grad_fn)
    }
}

impl Neg for Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        (&self).neg()
    }
}

// ============================================================================
// Ownership consuming variants (delegate to &-& version)
// ============================================================================

impl Add<&Variable> for Variable {
    type Output = Variable;
    fn add(self, rhs: &Variable) -> Variable {
        &self + rhs
    }
}

impl Add<Variable> for &Variable {
    type Output = Variable;
    fn add(self, rhs: Variable) -> Variable {
        self + &rhs
    }
}

impl Add<Variable> for Variable {
    type Output = Variable;
    fn add(self, rhs: Variable) -> Variable {
        &self + &rhs
    }
}

impl Sub<&Variable> for Variable {
    type Output = Variable;
    fn sub(self, rhs: &Variable) -> Variable {
        &self - rhs
    }
}

impl Sub<Variable> for &Variable {
    type Output = Variable;
    fn sub(self, rhs: Variable) -> Variable {
        self - &rhs
    }
}

impl Sub<Variable> for Variable {
    type Output = Variable;
    fn sub(self, rhs: Variable) -> Variable {
        &self - &rhs
    }
}

impl Mul<&Variable> for Variable {
    type Output = Variable;
    fn mul(self, rhs: &Variable) -> Variable {
        &self * rhs
    }
}

impl Mul<Variable> for &Variable {
    type Output = Variable;
    fn mul(self, rhs: Variable) -> Variable {
        self * &rhs
    }
}

impl Mul<Variable> for Variable {
    type Output = Variable;
    fn mul(self, rhs: Variable) -> Variable {
        &self * &rhs
    }
}

impl Div<&Variable> for Variable {
    type Output = Variable;
    fn div(self, rhs: &Variable) -> Variable {
        &self / rhs
    }
}

impl Div<Variable> for &Variable {
    type Output = Variable;
    fn div(self, rhs: Variable) -> Variable {
        self / &rhs
    }
}

impl Div<Variable> for Variable {
    type Output = Variable;
    fn div(self, rhs: Variable) -> Variable {
        &self / &rhs
    }
}

// ============================================================================
// Variable + f32, Variable - f32
// ============================================================================

impl Add<f32> for &Variable {
    type Output = Variable;

    fn add(self, rhs: f32) -> Variable {
        let result_data = &self.data() + rhs;
        let requires_grad = self.requires_grad();
        let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
            Some(Box::new(ScalarAddGrad {
                input: self.clone(),
            }))
        } else {
            None
        };
        Variable::from_grad_fn(result_data, requires_grad, grad_fn)
    }
}

impl Add<f32> for Variable {
    type Output = Variable;
    fn add(self, rhs: f32) -> Variable {
        &self + rhs
    }
}

impl Sub<f32> for &Variable {
    type Output = Variable;

    fn sub(self, rhs: f32) -> Variable {
        self + (-rhs)
    }
}

impl Sub<f32> for Variable {
    type Output = Variable;
    fn sub(self, rhs: f32) -> Variable {
        &self - rhs
    }
}

// ============================================================================
// Variable * f32, f32 * Variable
// ============================================================================

impl Mul<f32> for &Variable {
    type Output = Variable;

    fn mul(self, rhs: f32) -> Variable {
        let result_data = &self.data() * rhs;
        let requires_grad = self.requires_grad();
        let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
            Some(Box::new(ScalarMulGrad {
                input: self.clone(),
                scalar: rhs,
            }))
        } else {
            None
        };
        Variable::from_grad_fn(result_data, requires_grad, grad_fn)
    }
}

impl Mul<f32> for Variable {
    type Output = Variable;
    fn mul(self, rhs: f32) -> Variable {
        &self * rhs
    }
}

impl Mul<&Variable> for f32 {
    type Output = Variable;
    fn mul(self, rhs: &Variable) -> Variable {
        rhs * self
    }
}

impl Mul<Variable> for f32 {
    type Output = Variable;
    fn mul(self, rhs: Variable) -> Variable {
        &rhs * self
    }
}

// ============================================================================
// Variable / f32
// ============================================================================

impl Div<f32> for &Variable {
    type Output = Variable;

    fn div(self, rhs: f32) -> Variable {
        // x / scalar is equivalent to x * (1/scalar)
        self * (1.0 / rhs)
    }
}

impl Div<f32> for Variable {
    type Output = Variable;
    fn div(self, rhs: f32) -> Variable {
        &self / rhs
    }
}

// ============================================================================
// Named operation functions (called from Variable methods)
// ============================================================================

/// Matrix multiplication with gradient tracking.
pub fn var_matmul(lhs: &Variable, rhs: &Variable) -> Variable {
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let result_data = lhs_data.matmul(&rhs_data);
    let requires_grad = needs_grad(&[lhs, rhs]);
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(MatmulGrad {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            lhs_data,
            rhs_data,
        }))
    } else {
        None
    };
    Variable::from_grad_fn(result_data, requires_grad, grad_fn)
}

/// ReLU with gradient tracking.
pub fn var_relu(input: &Variable) -> Variable {
    let input_data = input.data();
    let result_data = input_data.relu();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(ReluGrad {
            input: input.clone(),
            input_data,
        }))
    } else {
        None
    };
    Variable::from_grad_fn(result_data, requires_grad, grad_fn)
}

/// Sigmoid with gradient tracking.
pub fn var_sigmoid(input: &Variable) -> Variable {
    let input_data = input.data();
    let output_data = input_data.sigmoid();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(SigmoidGrad {
            input: input.clone(),
            output_data: output_data.clone(),
        }))
    } else {
        None
    };
    Variable::from_grad_fn(output_data, requires_grad, grad_fn)
}

/// Tanh with gradient tracking.
pub fn var_tanh(input: &Variable) -> Variable {
    let input_data = input.data();
    let output_data = input_data.tanh_();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(TanhGrad {
            input: input.clone(),
            output_data: output_data.clone(),
        }))
    } else {
        None
    };
    Variable::from_grad_fn(output_data, requires_grad, grad_fn)
}

/// Exp with gradient tracking.
pub fn var_exp(input: &Variable) -> Variable {
    let input_data = input.data();
    let output_data = input_data.exp();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(ExpGrad {
            input: input.clone(),
            output_data: output_data.clone(),
        }))
    } else {
        None
    };
    Variable::from_grad_fn(output_data, requires_grad, grad_fn)
}

/// Log with gradient tracking.
pub fn var_log(input: &Variable) -> Variable {
    let input_data = input.data();
    let result_data = input_data.log();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(LogGrad {
            input: input.clone(),
            input_data,
        }))
    } else {
        None
    };
    Variable::from_grad_fn(result_data, requires_grad, grad_fn)
}

/// Pow with gradient tracking.
pub fn var_pow(input: &Variable, p: f32) -> Variable {
    let input_data = input.data();
    let result_data = input_data.pow(p);
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(PowGrad {
            input: input.clone(),
            input_data,
            exponent: p,
        }))
    } else {
        None
    };
    Variable::from_grad_fn(result_data, requires_grad, grad_fn)
}

/// Sqrt with gradient tracking.
pub fn var_sqrt(input: &Variable) -> Variable {
    let input_data = input.data();
    let output_data = input_data.sqrt();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(SqrtGrad {
            input: input.clone(),
            output_data: output_data.clone(),
        }))
    } else {
        None
    };
    Variable::from_grad_fn(output_data, requires_grad, grad_fn)
}

/// Sum (all elements) with gradient tracking.
pub fn var_sum(input: &Variable) -> Variable {
    let result_data = input.data().sum();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(SumGrad {
            input: input.clone(),
        }))
    } else {
        None
    };
    Variable::from_grad_fn(result_data, requires_grad, grad_fn)
}

/// Mean (all elements) with gradient tracking.
pub fn var_mean(input: &Variable) -> Variable {
    let result_data = input.data().mean();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(MeanGrad {
            input: input.clone(),
        }))
    } else {
        None
    };
    Variable::from_grad_fn(result_data, requires_grad, grad_fn)
}

/// Sum along axis with gradient tracking.
pub fn var_sum_axis(input: &Variable, axis: usize, keepdim: bool) -> Variable {
    let result_data = input.data().sum_axis(axis, keepdim).unwrap();
    let requires_grad = input.requires_grad();
    let grad_fn: Option<Box<dyn GradFn>> = if requires_grad {
        Some(Box::new(SumAxisGrad {
            input: input.clone(),
            axis,
            keepdim,
        }))
    } else {
        None
    };
    Variable::from_grad_fn(result_data, requires_grad, grad_fn)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rustforge_tensor::Tensor;

    #[test]
    fn test_add_forward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), false);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]), false);
        let c = &a + &b;
        assert_eq!(c.data().to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_forward() {
        let a = Variable::new(Tensor::from_vec(vec![5.0, 6.0], &[2]), false);
        let b = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), false);
        let c = &a - &b;
        assert_eq!(c.data().to_vec(), vec![4.0, 4.0]);
    }

    #[test]
    fn test_mul_forward() {
        let a = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]), false);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0], &[2]), false);
        let c = &a * &b;
        assert_eq!(c.data().to_vec(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_div_forward() {
        let a = Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[2]), false);
        let b = Variable::new(Tensor::from_vec(vec![2.0, 5.0], &[2]), false);
        let c = &a / &b;
        assert_eq!(c.data().to_vec(), vec![5.0, 4.0]);
    }

    #[test]
    fn test_neg_forward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3]), false);
        let b = -&a;
        assert_eq!(b.data().to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_scalar_ops_forward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]), false);
        let b = &a + 10.0;
        assert_eq!(b.data().to_vec(), vec![11.0, 12.0, 13.0]);

        let c = &a * 2.0;
        assert_eq!(c.data().to_vec(), vec![2.0, 4.0, 6.0]);

        let d = 3.0 * &a;
        assert_eq!(d.data().to_vec(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_matmul_forward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let b = Variable::new(Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]), false);
        let c = a.matmul(&b);
        let data = c.data().to_vec();
        assert_abs_diff_eq!(data[0], 19.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[1], 22.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[2], 43.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data[3], 50.0, epsilon = 1e-6);
    }

    #[test]
    fn test_relu_forward() {
        let a = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]), false);
        let b = a.relu();
        assert_eq!(b.data().to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_requires_grad_propagation() {
        let a = Variable::new(Tensor::ones(&[2]), true);
        let b = Variable::new(Tensor::ones(&[2]), false);
        let c = &a + &b;
        // If either input requires grad, output should too
        assert!(c.requires_grad());
        assert!(c.has_grad_fn());

        let d = Variable::new(Tensor::ones(&[2]), false);
        let e = Variable::new(Tensor::ones(&[2]), false);
        let f = &d + &e;
        assert!(!f.requires_grad());
        assert!(!f.has_grad_fn());
    }

    #[test]
    fn test_mixed_variable_operations() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]), false);

        let c = a.clone() + &b;
        assert!(c.requires_grad());

        let d = &a + b.clone();
        assert!(d.requires_grad());

        let e = a.clone() + b.clone();
        assert!(e.requires_grad());

        let f = a.clone() - &b;
        assert!(f.requires_grad());

        let g = &a - b.clone();
        assert!(g.requires_grad());

        let h = a.clone() - b.clone();
        assert!(h.requires_grad());

        let i = a.clone() * &b;
        assert!(i.requires_grad());

        let j = &a * b.clone();
        assert!(j.requires_grad());

        let k = a.clone() * b.clone();
        assert!(k.requires_grad());

        let l = a.clone() / &b;
        assert!(l.requires_grad());

        let m = &a / b.clone();
        assert!(m.requires_grad());

        let n = a.clone() / b.clone();
        assert!(n.requires_grad());
    }

    #[test]
    fn test_div_by_zero() {
        let a = Variable::new(Tensor::from_vec(vec![10.0], &[1]), false);
        let b = Variable::new(Tensor::from_vec(vec![0.0], &[1]), false);
        let c = &a / &b;
        assert!(c.data().to_vec()[0].is_infinite());
    }

    #[test]
    fn test_div_by_zero_scalar() {
        let a = Variable::new(Tensor::from_vec(vec![10.0], &[1]), false);
        let c = &a / 0.0;
        assert!(c.data().to_vec()[0].is_infinite());

        let d = a.clone() / 0.0;
        assert!(d.data().to_vec()[0].is_infinite());
    }

    #[test]
    fn test_mul_by_zero() {
        let a = Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[2]), true);
        let b = &a * 0.0;
        assert_eq!(b.data().to_vec(), vec![0.0, 0.0]);
        assert!(b.requires_grad());

        let c = 0.0 * &a;
        assert_eq!(c.data().to_vec(), vec![0.0, 0.0]);
        assert!(c.requires_grad());

        let d = a.clone() * 0.0;
        assert_eq!(d.data().to_vec(), vec![0.0, 0.0]);
        assert!(d.requires_grad());

        let e = 0.0 * a.clone();
        assert_eq!(e.data().to_vec(), vec![0.0, 0.0]);
        assert!(e.requires_grad());
    }

    #[test]
    fn test_scalar_sub() {
        let a = Variable::new(Tensor::from_vec(vec![10.0, 20.0], &[2]), true);
        let b = &a - 2.0;
        assert_eq!(b.data().to_vec(), vec![8.0, 18.0]);
        assert!(b.requires_grad());

        let c = a.clone() - 2.0;
        assert_eq!(c.data().to_vec(), vec![8.0, 18.0]);
        assert!(c.requires_grad());
    }

    #[test]
    fn test_neg_value() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3]), false);
        let b = -a.clone();
        assert_eq!(b.data().to_vec(), vec![-1.0, 2.0, -3.0]);
    }
}
