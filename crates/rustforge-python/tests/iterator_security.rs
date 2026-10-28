//! Regression for RUSTSEC-2026-0176: an oversized skip must exhaust the iterator.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

#[test]
fn list_iterator_oversized_skip_does_not_wrap_to_a_previous_item() {
    Python::initialize();
    Python::attach(|py| {
        let sequence = PyList::new(py, [10, 20, 30]).unwrap();
        let mut iterator = sequence.iter();
        assert_eq!(iterator.next().unwrap().extract::<i32>().unwrap(), 10);
        assert!(iterator.nth(usize::MAX).is_none());
    });
}

#[test]
fn tuple_iterator_oversized_skip_does_not_wrap_to_a_previous_item() {
    Python::initialize();
    Python::attach(|py| {
        let sequence = PyTuple::new(py, [10, 20, 30]).unwrap();
        let mut iterator = sequence.iter();
        assert_eq!(iterator.next().unwrap().extract::<i32>().unwrap(), 10);
        assert!(iterator.nth(usize::MAX).is_none());
    });
}

#[test]
fn list_iterator_oversized_backward_skip_does_not_read_past_the_end() {
    Python::initialize();
    Python::attach(|py| {
        let sequence = PyList::new(py, [10, 20, 30]).unwrap();
        let mut iterator = sequence.iter();
        assert_eq!(iterator.next_back().unwrap().extract::<i32>().unwrap(), 30);
        assert!(iterator.nth_back(usize::MAX).is_none());
    });
}

#[test]
fn tuple_iterator_oversized_backward_skip_does_not_read_past_the_end() {
    Python::initialize();
    Python::attach(|py| {
        let sequence = PyTuple::new(py, [10, 20, 30]).unwrap();
        let mut iterator = sequence.iter();
        assert_eq!(iterator.next_back().unwrap().extract::<i32>().unwrap(), 30);
        assert!(iterator.nth_back(usize::MAX).is_none());
    });
}
