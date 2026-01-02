//! RAM Accelerator - High-performance RAM neuron evaluation for Apple Silicon
//!
//! Provides GPU-accelerated evaluation of RAM neuron connectivity patterns
//! using Metal compute shaders on M-series Macs.

use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;

#[path = "ram.rs"]
mod ram;

#[path = "metal_evaluator.rs"]
mod metal_evaluator;

pub use ram::RAMNeuron;
pub use metal_evaluator::MetalEvaluator;

/// Evaluate a single connectivity pattern (CPU, for comparison)
#[pyfunction]
fn evaluate_connectivity_cpu(
    connectivity: Vec<Vec<i64>>,
    word_to_cluster: std::collections::HashMap<String, u64>,
    train_tokens: Vec<String>,
    test_tokens: Vec<String>,
    bits_per_neuron: usize,
    eval_subset: usize,
) -> PyResult<f64> {
    let word_map: FxHashMap<String, u64> = word_to_cluster.into_iter().collect();
    let result = ram::evaluate_single(
        &connectivity,
        &word_map,
        &train_tokens,
        &test_tokens,
        bits_per_neuron,
        eval_subset,
    );
    Ok(result)
}

/// Evaluate multiple connectivity patterns in parallel on CPU (using rayon)
#[pyfunction]
fn evaluate_batch_cpu(
    connectivities: Vec<Vec<Vec<i64>>>,
    word_to_cluster: std::collections::HashMap<String, u64>,
    train_tokens: Vec<String>,
    test_tokens: Vec<String>,
    bits_per_neuron: usize,
    eval_subset: usize,
) -> PyResult<Vec<f64>> {
    let word_map: FxHashMap<String, u64> = word_to_cluster.into_iter().collect();
    let word_map = Arc::new(word_map);
    let train = Arc::new(train_tokens);
    let test = Arc::new(test_tokens);

    let results: Vec<f64> = connectivities
        .par_iter()
        .map(|conn| {
            ram::evaluate_single(
                conn,
                &word_map,
                &train,
                &test,
                bits_per_neuron,
                eval_subset,
            )
        })
        .collect();

    Ok(results)
}

/// Evaluate multiple connectivity patterns on Metal GPU
#[pyfunction]
fn evaluate_batch_metal(
    py: Python<'_>,
    connectivities: Vec<Vec<Vec<i64>>>,
    word_to_cluster: std::collections::HashMap<String, u64>,
    train_tokens: Vec<String>,
    test_tokens: Vec<String>,
    bits_per_neuron: usize,
    eval_subset: usize,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        let evaluator = metal_evaluator::MetalEvaluator::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        evaluator
            .evaluate_batch(
                &connectivities,
                &word_to_cluster,
                &train_tokens,
                &test_tokens,
                bits_per_neuron,
                eval_subset,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
}

/// Check if Metal GPU is available
#[pyfunction]
fn metal_available() -> bool {
    metal_evaluator::MetalEvaluator::is_available()
}

/// Get Metal device info
#[pyfunction]
fn metal_device_info() -> PyResult<String> {
    metal_evaluator::MetalEvaluator::device_info()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Get number of CPU cores available for rayon
#[pyfunction]
fn cpu_cores() -> usize {
    rayon::current_num_threads()
}

/// Python module definition
#[pymodule]
fn ram_accelerator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_connectivity_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch_metal, m)?)?;
    m.add_function(wrap_pyfunction!(metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(metal_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_cores, m)?)?;
    Ok(())
}
