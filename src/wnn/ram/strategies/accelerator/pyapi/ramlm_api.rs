//! RAMLM acceleration (proper RAM WNN architecture).
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// RAMLM ACCELERATION (proper RAM WNN architecture)
// =============================================================================

/// Batch training for RAMClusterLayer using NumPy arrays (FAST - near zero-copy)
///
/// Same as ramlm_train_batch but uses numpy arrays for input, avoiding Python list
/// conversion overhead. This is typically 5-10x faster for large batches.
///
/// Note: Training stays on CPU (rayon) because atomic writes have high contention
/// on GPU. The bottleneck is data transfer, not computation.
///
/// Args:
///   input_bits: [num_examples * total_input_bits] u8 numpy array (0/1 values)
///   true_clusters: [num_examples] i64 numpy array of target cluster indices
///   false_clusters: [num_examples * num_negatives] i64 numpy array
///   connections: [num_neurons * bits_per_neuron] i64 numpy array
///   memory_words: [num_neurons * words_per_neuron] i64 numpy array (will be copied and modified)
///
/// Returns: (num_modified, new_memory_words)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_train_batch_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    true_clusters: PyReadonlyArray1<'py, i64>,
    false_clusters: PyReadonlyArray1<'py, i64>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_negatives: usize,
    words_per_neuron: usize,
    allow_override: bool,
) -> PyResult<(usize, Vec<i64>)> {
    // Extract data from numpy arrays BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let true_slice = true_clusters.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("True clusters array not contiguous: {}", e))
    })?;
    let false_slice = false_clusters.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("False clusters array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits, copy others
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let true_vec: Vec<i64> = true_slice.to_vec();
    let false_vec: Vec<i64> = false_slice.to_vec();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mut mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        let modified = ramlm::train_batch(
            &input_bools,
            &true_vec,
            &false_vec,
            &conn_vec,
            &mut mem_vec,
            num_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_negatives,
            words_per_neuron,
            allow_override,
        );
        Ok((modified, mem_vec))
    })
}

/// Bitwise batch training for BitwiseRAMLM (dense memory)
///
/// Multi-label training: each example trains ALL clusters (one per output bit).
/// target_bits[ex, cluster] = 1 means TRUE, 0 means FALSE.
///
/// Args:
///   input_bits: [num_examples * total_input_bits] u8 numpy array
///   target_bits: [num_examples * num_clusters] u8 numpy array (0/1 per cluster)
///   connections: [num_neurons * bits_per_neuron] i64 numpy array
///   memory_words: [num_neurons * words_per_neuron] i64 numpy array
///
/// Returns: (num_modified, new_memory_words)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_bitwise_train_batch_numpy<'py>(
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
    allow_override: bool,
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
        let modified = ramlm::bitwise_train_batch(
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
            allow_override,
        );
        Ok((modified, mem_vec))
    })
}

/// Tiered batch training - ALL tiers in a single Rust call (eliminates Python loop overhead)
///
/// This is the optimized training function for tiered architectures. Instead of calling
/// Rust separately for each tier (with Python overhead between), this function handles
/// ALL tiers internally with full rayon parallelization.
///
/// Args:
///   input_bits: [num_examples * total_input_bits] u8 numpy array (0/1 values)
///   true_clusters: [num_examples] i64 numpy array of global cluster indices
///   false_clusters: [num_examples * num_negatives] i64 numpy array of global cluster indices
///   connections_flat: All tiers' connections concatenated (tier0..tier1..tier2..)
///   memory_words_flat: All tiers' memory concatenated (tier0..tier1..tier2..)
///   tier_configs: List of (cluster_start, cluster_end, neurons_per_cluster, bits_per_neuron,
///                         words_per_neuron, memory_offset, conn_offset) tuples
///
/// Returns: (num_modified, updated_memory_words_flat)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_train_batch_tiered_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    true_clusters: PyReadonlyArray1<'py, i64>,
    false_clusters: PyReadonlyArray1<'py, i64>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    memory_words_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
    tier_configs: Vec<(usize, usize, usize, usize, usize, usize, usize)>,
    allow_override: bool,
) -> PyResult<(usize, Vec<i64>)> {
    // Extract data from numpy arrays BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let true_slice = true_clusters.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("True clusters array not contiguous: {}", e))
    })?;
    let false_slice = false_clusters.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("False clusters array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert to Rust types
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let true_vec: Vec<i64> = true_slice.to_vec();
    let false_vec: Vec<i64> = false_slice.to_vec();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mut mem_vec: Vec<i64> = mem_slice.to_vec();

    // Convert tier configs to struct
    let tier_structs: Vec<ramlm::TierConfig> = tier_configs.iter().map(|&(cluster_start, cluster_end, neurons_per_cluster, bits_per_neuron, words_per_neuron, memory_offset, conn_offset)| {
        ramlm::TierConfig {
            cluster_start,
            cluster_end,
            neurons_per_cluster,
            bits_per_neuron,
            words_per_neuron,
            memory_offset,
            conn_offset,
        }
    }).collect();

    py.allow_threads(|| {
        let modified = ramlm::train_batch_tiered(
            &input_bools,
            &true_vec,
            &false_vec,
            &conn_vec,
            &mut mem_vec,
            num_examples,
            total_input_bits,
            num_negatives,
            &tier_structs,
            allow_override,
        );
        Ok((modified, mem_vec))
    })
}

/// Check if Metal RAMLM is available
#[pyfunction]
pub(crate) fn ramlm_metal_available() -> bool {
    metal_ramlm::MetalRAMLMEvaluator::is_available()
}
