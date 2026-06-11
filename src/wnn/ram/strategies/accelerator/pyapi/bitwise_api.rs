//! Bitwise RAMLM — nudge training + quad forward.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// Bitwise RAMLM — Nudge Training + Quad Forward (PyO3 wrappers)
// =============================================================================

/// Bitwise batch training with 4-state nudging (QUAD modes).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_bitwise_train_batch_nudge_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    target_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> PyResult<(usize, Vec<i64>)> {
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let target_slice = target_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Target bits array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let target_vec: Vec<u8> = target_slice.to_vec();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mut mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        let modified = ramlm::bitwise_train_batch_nudge(
            &input_bools,
            &target_vec,
            &conn_vec,
            &mut mem_vec,
            num_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            words_per_neuron,
            neuron_sample_rate,
            rng_seed,
        );
        Ok((modified, mem_vec))
    })
}

/// Complete train + forward + CE in one Rust call (CPU with optional Metal CE acceleration).
/// Returns (ce, accuracy, per_bit_accuracy_list, modified_count, updated_memory).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_bitwise_train_and_eval_numpy<'py>(
    py: Python<'py>,
    train_input_bits: PyReadonlyArray1<'py, u8>,
    train_target_bits: PyReadonlyArray1<'py, u8>,
    eval_input_bits: PyReadonlyArray1<'py, u8>,
    eval_targets: PyReadonlyArray1<'py, u32>,
    token_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_train: usize,
    num_eval: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    vocab_size: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    empty_value: f32,
) -> PyResult<(f64, f64, Vec<f32>, Vec<i64>)> {
    let train_input = train_input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    })?;
    let train_target = train_target_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    })?;
    let eval_input = eval_input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    })?;
    let eval_tgt = eval_targets.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    })?;
    let tok_bits = token_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    })?;

    let train_bools: Vec<bool> = train_input.iter().map(|&b| b != 0).collect();
    let eval_bools: Vec<bool> = eval_input.iter().map(|&b| b != 0).collect();
    let train_tgt_vec: Vec<u8> = train_target.to_vec();
    let eval_tgt_vec: Vec<u32> = eval_tgt.to_vec();
    let tok_bits_vec: Vec<u8> = tok_bits.to_vec();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mut mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        let (ce, acc, per_bit_acc) = ramlm::bitwise_train_and_eval_full(
            &train_bools, &train_tgt_vec,
            &eval_bools, &eval_tgt_vec, &tok_bits_vec,
            &conn_vec, &mut mem_vec,
            num_train, num_eval, total_input_bits,
            bits_per_neuron, neurons_per_cluster, num_clusters,
            words_per_neuron, vocab_size,
            memory_mode, neuron_sample_rate, rng_seed,
            empty_value,
        );
        Ok((ce, acc, per_bit_acc, mem_vec))
    })
}

/// Forward pass for QUAD_BINARY mode: P = count(cell >= 2) / neurons_per_cluster
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_forward_batch_quad_binary_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
) -> PyResult<Vec<f32>> {
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        Ok(ramlm::forward_batch_quad_binary(
            &input_bools, &conn_vec, &mem_vec,
            num_examples, total_input_bits, num_neurons,
            bits_per_neuron, neurons_per_cluster, num_clusters, words_per_neuron,
        ))
    })
}

/// Forward pass for QUAD_WEIGHTED mode: P = sum(weight[cell]) / neurons_per_cluster
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_forward_batch_quad_weighted_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
) -> PyResult<Vec<f32>> {
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        Ok(ramlm::forward_batch_quad_weighted(
            &input_bools, &conn_vec, &mem_vec,
            num_examples, total_input_bits, num_neurons,
            bits_per_neuron, neurons_per_cluster, num_clusters, words_per_neuron,
        ))
    })
}
