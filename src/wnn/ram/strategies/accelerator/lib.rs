//! RAM Accelerator - High-performance RAM neuron evaluation for Apple Silicon
//!
//! Provides GPU-accelerated evaluation of RAM neuron connectivity patterns
//! using Metal compute shaders on M-series Macs.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn};

// Global cached Metal evaluator for RAMLM (avoids shader recompilation)
// Using OnceLock + Option to handle initialization errors gracefully
static METAL_RAMLM_EVALUATOR: OnceLock<Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>> = OnceLock::new();

fn get_cached_metal_evaluator() -> Result<&'static Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>, String> {
    Ok(METAL_RAMLM_EVALUATOR.get_or_init(|| {
        Mutex::new(metal_ramlm::MetalRAMLMEvaluator::new().ok())
    }))
}

#[path = "ram.rs"]
mod ram;

#[path = "ramlm.rs"]
mod ramlm;

#[path = "metal_evaluator.rs"]
mod metal_evaluator;

#[path = "metal_ramlm.rs"]
mod metal_ramlm;

#[path = "sparse_memory.rs"]
mod sparse_memory;

pub use ram::RAMNeuron;
pub use metal_evaluator::MetalEvaluator;
pub use metal_ramlm::MetalRAMLMEvaluator;

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

/// Evaluate cascade with all RAMs, optimizing one at a time
/// base_connectivities: [n2_conn, n3_conn, n4_conn, n5_conn, n6_conn]
/// candidates: candidate connectivities for the RAM at target_ram_idx
/// target_ram_idx: 0=n2, 1=n3, 2=n4, 3=n5, 4=n6
#[pyfunction]
fn evaluate_cascade_batch_cpu(
    base_connectivities: Vec<Vec<Vec<i64>>>,  // 5 RAMs: [n2, n3, n4, n5, n6]
    candidates: Vec<Vec<Vec<i64>>>,            // candidate patterns for target RAM
    target_ram_idx: usize,
    word_to_cluster: std::collections::HashMap<String, u64>,
    train_tokens: Vec<String>,
    test_tokens: Vec<String>,
    eval_subset: usize,
) -> PyResult<Vec<f64>> {
    let word_map: FxHashMap<String, u64> = word_to_cluster.into_iter().collect();

    let results = ram::evaluate_cascade_batch(
        &base_connectivities,
        &candidates,
        target_ram_idx,
        &word_map,
        &train_tokens,
        &test_tokens,
        eval_subset,
    );

    Ok(results)
}

/// Evaluate FULL NETWORK (exact + generalized + voting) with pre-computed exact results
/// exact_results: For each test position - None (not covered), Some(true) (correct), Some(false) (incorrect)
/// Only evaluates generalized RAMs for positions not covered by exact
#[pyfunction]
fn evaluate_fullnetwork_batch_cpu(
    base_connectivities: Vec<Vec<Vec<i64>>>,  // 5 RAMs: [n2, n3, n4, n5, n6]
    candidates: Vec<Vec<Vec<i64>>>,            // candidate patterns for target RAM
    target_ram_idx: usize,
    word_to_cluster: std::collections::HashMap<String, u64>,
    train_tokens: Vec<String>,
    test_tokens: Vec<String>,
    exact_results: Vec<Option<bool>>,          // Pre-computed exact RAM results
    eval_subset: usize,
) -> PyResult<Vec<f64>> {
    let word_map: FxHashMap<String, u64> = word_to_cluster.into_iter().collect();

    let results = ram::evaluate_fullnetwork_batch(
        &base_connectivities,
        &candidates,
        target_ram_idx,
        &word_map,
        &train_tokens,
        &test_tokens,
        &exact_results,
        eval_subset,
    );

    Ok(results)
}

/// Evaluate FULL NETWORK using PERPLEXITY as metric (lower is better)
/// exact_probs: For each test position - None (not covered by exact), Some(prob) (probability assigned to target)
/// Returns perplexity = exp(mean cross-entropy) for each candidate
#[pyfunction]
fn evaluate_fullnetwork_perplexity_batch_cpu(
    base_connectivities: Vec<Vec<Vec<i64>>>,
    candidates: Vec<Vec<Vec<i64>>>,
    target_ram_idx: usize,
    word_to_cluster: std::collections::HashMap<String, u64>,
    train_tokens: Vec<String>,
    test_tokens: Vec<String>,
    exact_probs: Vec<Option<f64>>,
    eval_subset: usize,
    vocab_size: usize,
    cascade_threshold: f64,
) -> PyResult<Vec<f64>> {
    let word_map: FxHashMap<String, u64> = word_to_cluster.into_iter().collect();

    let results = ram::evaluate_fullnetwork_perplexity_batch(
        &base_connectivities,
        &candidates,
        target_ram_idx,
        &word_map,
        &train_tokens,
        &test_tokens,
        &exact_probs,
        eval_subset,
        vocab_size,
        cascade_threshold,
    );

    Ok(results)
}

// =============================================================================
// BATCH PREDICTION WITH PRE-TRAINED RAMs (for voting strategies acceleration)
// =============================================================================

/// Batch predict using pre-trained RAMs on CPU (rayon parallel)
///
/// This is the key acceleration for evaluate_voting_strategies():
/// - Takes pre-trained RAM data (connectivity + memory tables)
/// - Predicts for all test positions in parallel
/// - Returns predictions for each position from each RAM
///
/// Args:
///   generalized_rams: List of (n, connectivity, memory) for each RAM
///     - n: context length (2-6)
///     - connectivity: Vec<Vec<i64>> - which bits each neuron observes
///     - memory: Vec<HashMap<u64, HashMap<String, u32>>> - per neuron lookup tables
///   exact_rams: List of (n, contexts) for each exact RAM
///     - n: context length
///     - contexts: HashMap<Vec<u64>, HashMap<String, u32>> - context bits → word counts
///   word_to_bits: HashMap<String, u64> - word encoding
///   test_tokens: Vec<String> - tokens to predict
///
/// Returns: Vec<HashMap<String, (String, f32)>> - per position: method → (word, confidence)
#[pyfunction]
fn predict_all_batch_cpu(
    py: Python<'_>,
    generalized_rams: Vec<(
        usize,  // n
        Vec<Vec<i64>>,  // connectivity per neuron
        Vec<std::collections::HashMap<u64, std::collections::HashMap<String, u32>>>,  // memory per neuron
    )>,
    exact_rams: Vec<(
        usize,  // n
        std::collections::HashMap<Vec<u64>, std::collections::HashMap<String, u32>>,  // context → word counts
    )>,
    word_to_bits: std::collections::HashMap<String, u64>,
    test_tokens: Vec<String>,
) -> PyResult<Vec<std::collections::HashMap<String, (String, f32)>>> {
    py.allow_threads(|| {
        let word_map: FxHashMap<String, u64> = word_to_bits.into_iter().collect();

        let results = ram::predict_all_batch(
            &generalized_rams,
            &exact_rams,
            &word_map,
            &test_tokens,
        );

        Ok(results)
    })
}

/// Batch predict using pre-trained RAMs on Metal GPU
/// Same interface as predict_all_batch_cpu but uses GPU parallelism
#[pyfunction]
fn predict_all_batch_metal(
    py: Python<'_>,
    generalized_rams: Vec<(
        usize,
        Vec<Vec<i64>>,
        Vec<std::collections::HashMap<u64, std::collections::HashMap<String, u32>>>,
    )>,
    exact_rams: Vec<(
        usize,
        std::collections::HashMap<Vec<u64>, std::collections::HashMap<String, u32>>,
    )>,
    word_to_bits: std::collections::HashMap<String, u64>,
    test_tokens: Vec<String>,
) -> PyResult<Vec<std::collections::HashMap<String, (String, f32)>>> {
    // For now, fall back to CPU - Metal implementation would require
    // significant shader work to handle HashMap lookups on GPU
    // The CPU rayon version is already very fast for this workload
    py.allow_threads(|| {
        let word_map: FxHashMap<String, u64> = word_to_bits.into_iter().collect();

        let results = ram::predict_all_batch(
            &generalized_rams,
            &exact_rams,
            &word_map,
            &test_tokens,
        );

        Ok(results)
    })
}

/// Batch predict using hybrid CPU+GPU
/// Uses CPU for HashMap-heavy prediction (better suited than GPU for this)
#[pyfunction]
fn predict_all_batch_hybrid(
    py: Python<'_>,
    generalized_rams: Vec<(
        usize,
        Vec<Vec<i64>>,
        Vec<std::collections::HashMap<u64, std::collections::HashMap<String, u32>>>,
    )>,
    exact_rams: Vec<(
        usize,
        std::collections::HashMap<Vec<u64>, std::collections::HashMap<String, u32>>,
    )>,
    word_to_bits: std::collections::HashMap<String, u64>,
    test_tokens: Vec<String>,
) -> PyResult<Vec<std::collections::HashMap<String, (String, f32)>>> {
    // HashMap lookups don't parallelize well on GPU (memory divergence)
    // Use CPU rayon which is optimal for this workload
    py.allow_threads(|| {
        let word_map: FxHashMap<String, u64> = word_to_bits.into_iter().collect();

        let results = ram::predict_all_batch(
            &generalized_rams,
            &exact_rams,
            &word_map,
            &test_tokens,
        );

        Ok(results)
    })
}

/// Compute exact RAM probabilities in parallel (HUGE speedup over Python)
/// Returns Vec<Option<f64>> where Some(prob) is P(target|context) if exact match found.
/// Python: ~10+ min for 287k tokens. Rust: ~seconds.
#[pyfunction]
fn compute_exact_probs_batch(
    py: Python<'_>,
    exact_rams: Vec<(
        usize,
        std::collections::HashMap<Vec<u64>, std::collections::HashMap<String, u32>>,
    )>,
    word_to_bits: std::collections::HashMap<String, u64>,
    tokens: Vec<String>,
) -> PyResult<Vec<Option<f64>>> {
    py.allow_threads(|| {
        let word_map: FxHashMap<String, u64> = word_to_bits.into_iter().collect();
        let results = ram::compute_exact_probs_batch(&exact_rams, &word_map, &tokens);
        Ok(results)
    })
}

/// Compute exact probabilities using word-based exact RAMs (no bit encoding).
///
/// This is the FAST version - works with String contexts directly, avoiding
/// the expensive Python bit-encoding step that takes 30+ minutes for 5M+ patterns.
///
/// # Arguments
/// * `exact_rams` - Vec of (n, contexts) where contexts is Vec<(context_words, {target: count})>
///   Note: Using Vec of tuples because Python can't have list keys in dicts
/// * `tokens` - Token sequence to evaluate
///
/// # Performance
/// - Python loop: ~10+ min for 287k tokens
/// - Rust parallel: ~seconds for 287k tokens
#[pyfunction]
fn compute_exact_probs_words(
    py: Python<'_>,
    exact_rams: Vec<(
        usize,
        Vec<(Vec<String>, std::collections::HashMap<String, u32>)>,
    )>,
    tokens: Vec<String>,
) -> PyResult<Vec<Option<f64>>> {
    py.allow_threads(|| {
        let results = ram::compute_exact_probs_words(exact_rams, &tokens);
        Ok(results)
    })
}

// =============================================================================
// RAMLM ACCELERATION (proper RAM WNN architecture)
// =============================================================================

/// Batch training for RAMClusterLayer
///
/// This is the core acceleration for RAMLM training. Uses rayon for parallel
/// processing of examples and atomic memory writes for thread safety.
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits] flattened bool array
///   true_clusters: [num_examples] target cluster indices
///   false_clusters_flat: [num_examples * num_negatives] flattened negative indices
///   connections_flat: [num_neurons * bits_per_neuron] flattened connections
///   memory_words: [num_neurons * words_per_neuron] flattened memory (MUTABLE)
///   ... dimension parameters ...
///
/// Returns: number of memory cells modified
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_train_batch(
    py: Python<'_>,
    input_bits_flat: Vec<bool>,
    true_clusters: Vec<i64>,
    false_clusters_flat: Vec<i64>,
    connections_flat: Vec<i64>,
    mut memory_words: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_negatives: usize,
    words_per_neuron: usize,
    allow_override: bool,
) -> PyResult<(usize, Vec<i64>)> {
    py.allow_threads(|| {
        let modified = ramlm::train_batch(
            &input_bits_flat,
            &true_clusters,
            &false_clusters_flat,
            &connections_flat,
            &mut memory_words,
            num_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_negatives,
            words_per_neuron,
            allow_override,
        );
        Ok((modified, memory_words))
    })
}

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
fn ramlm_train_batch_numpy<'py>(
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
fn ramlm_train_batch_tiered_numpy<'py>(
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

/// Batch forward pass for RAMClusterLayer (CPU - rayon parallel)
///
/// Computes probabilities for all clusters for all examples in parallel.
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits] flattened bool array
///   connections_flat: [num_neurons * bits_per_neuron] flattened connections
///   memory_words: [num_neurons * words_per_neuron] flattened memory
///   ... dimension parameters ...
///
/// Returns: [num_examples * num_clusters] flattened probabilities
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch(
    py: Python<'_>,
    input_bits_flat: Vec<bool>,
    connections_flat: Vec<i64>,
    memory_words: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        let probs = ramlm::forward_batch(
            &input_bits_flat,
            &connections_flat,
            &memory_words,
            num_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            words_per_neuron,
        );
        Ok(probs)
    })
}

/// Batch forward pass for RAMClusterLayer (Metal GPU - 40 cores on M4 Max)
///
/// Same interface as ramlm_forward_batch but uses Metal GPU compute shaders.
/// Particularly effective for large vocabularies (50K clusters) where GPU
/// parallelism provides massive speedup.
///
/// Returns: [num_examples * num_clusters] flattened probabilities
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch_metal(
    py: Python<'_>,
    input_bits_flat: Vec<bool>,
    connections_flat: Vec<i64>,
    memory_words: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        evaluator
            .forward_batch(
                &input_bits_flat,
                &connections_flat,
                &memory_words,
                num_examples,
                total_input_bits,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
}

/// Check if Metal RAMLM is available
#[pyfunction]
fn ramlm_metal_available() -> bool {
    metal_ramlm::MetalRAMLMEvaluator::is_available()
}

// =============================================================================
// NUMPY-BASED RAMLM FUNCTIONS (Zero-copy for maximum performance)
// =============================================================================

/// Batch forward pass using numpy arrays (FAST - zero-copy)
///
/// This is the optimized version that avoids Python list conversion overhead.
/// Accepts numpy arrays directly for near-zero-copy access.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,       // numpy bool/uint8 array
    connections: PyReadonlyArray1<'py, i64>,      // numpy int64 array
    memory_words: PyReadonlyArray1<'py, i64>,     // numpy int64 array
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
) -> PyResult<Vec<f32>> {
    // Extract data BEFORE allow_threads (numpy arrays aren't thread-safe)
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    // Copy connections and memory for thread safety
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        let probs = ramlm::forward_batch(
            &input_bools,
            &conn_vec,
            &mem_vec,
            num_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            words_per_neuron,
        );
        Ok(probs)
    })
}

/// Batch forward pass using numpy arrays on Metal GPU (FAST - zero-copy)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch_metal_numpy<'py>(
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
    // Extract data BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        evaluator
            .forward_batch(
                &input_bools,
                &conn_vec,
                &mem_vec,
                num_examples,
                total_input_bits,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
}

/// Initialize and cache the Metal RAMLM evaluator
/// Call this once at startup to avoid first-call latency
#[pyfunction]
fn ramlm_init_metal() -> PyResult<bool> {
    let guard = get_cached_metal_evaluator()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?
        .lock()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(guard.is_some())
}

/// Batch forward pass using CACHED Metal evaluator (no recompilation)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch_metal_cached<'py>(
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
    // Extract data BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        // Get cached evaluator
        let guard = get_cached_metal_evaluator()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let evaluator = guard.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Metal not available"))?;

        evaluator
            .forward_batch(
                &input_bools,
                &conn_vec,
                &mem_vec,
                num_examples,
                total_input_bits,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
}

/// Hybrid CPU+GPU forward pass with CACHED Metal (56 cores on M4 Max)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch_hybrid_cached<'py>(
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
    // Extract data BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        // For small batches, just use CPU
        if num_examples < 10 {
            return Ok(ramlm::forward_batch(
                &input_bools,
                &conn_vec,
                &mem_vec,
                num_examples,
                total_input_bits,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
            ));
        }

        // Split work: GPU gets 70%, CPU gets 30%
        let gpu_examples = (num_examples * 7) / 10;
        let cpu_examples = num_examples - gpu_examples;

        // Run GPU and CPU in parallel
        let gpu_input: Vec<bool> = input_bools[..gpu_examples * total_input_bits].to_vec();
        let cpu_input: Vec<bool> = input_bools[gpu_examples * total_input_bits..].to_vec();
        let conn_vec_cpu = conn_vec.clone();
        let mem_vec_cpu = mem_vec.clone();

        // CPU thread (runs in parallel with main thread doing GPU)
        let cpu_handle = std::thread::spawn(move || {
            ramlm::forward_batch(
                &cpu_input,
                &conn_vec_cpu,
                &mem_vec_cpu,
                cpu_examples,
                total_input_bits,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
            )
        });

        // GPU on main thread (uses cached evaluator)
        let guard = get_cached_metal_evaluator()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let evaluator = guard.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Metal not available"))?;

        let gpu_probs = evaluator
            .forward_batch(
                &gpu_input,
                &conn_vec,
                &mem_vec,
                gpu_examples,
                total_input_bits,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        // Drop the lock before waiting for CPU
        drop(guard);

        // Wait for CPU and combine results
        let cpu_probs = cpu_handle
            .join()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("CPU thread panicked"))?;

        // Combine: GPU results first, then CPU results
        let mut all_probs = gpu_probs;
        all_probs.extend(cpu_probs);
        Ok(all_probs)
    })
}

/// Hybrid CPU+GPU forward pass (uses all 56 cores on M4 Max)
///
/// Splits work between CPU (16 cores via rayon) and GPU (40 cores via Metal).
/// Optimal for large batches where both can work in parallel.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch_hybrid_numpy<'py>(
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
    // Extract data BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        // Split work: GPU gets 70%, CPU gets 30% (GPU is faster for this workload)
        let gpu_examples = (num_examples * 7) / 10;
        let cpu_examples = num_examples - gpu_examples;

        if gpu_examples == 0 || cpu_examples == 0 {
            // Fall back to single-backend for small batches
            if gpu_examples == 0 {
                return Ok(ramlm::forward_batch(
                    &input_bools,
                    &conn_vec,
                    &mem_vec,
                    num_examples,
                    total_input_bits,
                    num_neurons,
                    bits_per_neuron,
                    neurons_per_cluster,
                    num_clusters,
                    words_per_neuron,
                ));
            } else {
                let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
                return evaluator
                    .forward_batch(
                        &input_bools,
                        &conn_vec,
                        &mem_vec,
                        num_examples,
                        total_input_bits,
                        num_neurons,
                        bits_per_neuron,
                        neurons_per_cluster,
                        num_clusters,
                        words_per_neuron,
                    )
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e));
            }
        }

        // Run GPU and CPU in parallel using std::thread
        let gpu_input: Vec<bool> = input_bools[..gpu_examples * total_input_bits].to_vec();
        let cpu_input: Vec<bool> = input_bools[gpu_examples * total_input_bits..].to_vec();
        let conn_vec_gpu = conn_vec.clone();
        let mem_vec_gpu = mem_vec.clone();

        let gpu_handle = std::thread::spawn(move || {
            let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()?;
            evaluator.forward_batch(
                &gpu_input,
                &conn_vec_gpu,
                &mem_vec_gpu,
                gpu_examples,
                total_input_bits,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
            )
        });

        // CPU processes remaining examples
        let cpu_probs = ramlm::forward_batch(
            &cpu_input,
            &conn_vec,
            &mem_vec,
            cpu_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            words_per_neuron,
        );

        // Wait for GPU and combine results
        let gpu_probs = gpu_handle
            .join()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("GPU thread panicked"))?
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        // Combine: GPU results first, then CPU results
        let mut all_probs = gpu_probs;
        all_probs.extend(cpu_probs);
        Ok(all_probs)
    })
}

// =============================================================================
// SPARSE MEMORY BACKEND (for >10 bits per neuron)
// =============================================================================

/// Python wrapper for SparseLayerMemory
/// Provides HashMap-based sparse storage for neurons with >10 bits
#[pyclass]
struct SparseMemory {
    inner: Arc<sparse_memory::SparseLayerMemory>,
    num_neurons: usize,
    bits_per_neuron: usize,
}

#[pymethods]
impl SparseMemory {
    /// Create a new sparse memory layer
    #[new]
    fn new(num_neurons: usize, bits_per_neuron: usize) -> Self {
        Self {
            inner: Arc::new(sparse_memory::SparseLayerMemory::new(num_neurons, bits_per_neuron)),
            num_neurons,
            bits_per_neuron,
        }
    }

    /// Read a single cell value (returns 0=FALSE, 1=TRUE, 2=EMPTY)
    fn read_cell(&self, neuron_idx: usize, address: u64) -> u8 {
        self.inner.read_cell(neuron_idx, address)
    }

    /// Write a single cell value
    /// Returns True if the cell was modified
    fn write_cell(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool {
        self.inner.write_cell(neuron_idx, address, value, allow_override)
    }

    /// Get total number of written cells across all neurons
    fn total_cells(&self) -> usize {
        self.inner.total_cells()
    }

    /// Get per-neuron cell counts
    fn cell_counts(&self) -> Vec<usize> {
        self.inner.cell_counts()
    }

    /// Export to list of (neuron_idx, address, value) tuples
    fn export(&self) -> Vec<(usize, u64, u8)> {
        self.inner.export()
    }

    /// Import from list of (neuron_idx, address, value) tuples
    fn import_cells(&self, cells: Vec<(usize, u64, u8)>) {
        self.inner.import(&cells);
    }

    /// Reset all memory to empty
    fn reset(&self) {
        self.inner.reset();
    }

    /// Get number of neurons
    #[getter]
    fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Get bits per neuron
    #[getter]
    fn bits_per_neuron(&self) -> usize {
        self.bits_per_neuron
    }

    /// Memory size if this were dense (for comparison)
    #[getter]
    fn dense_memory_size(&self) -> u64 {
        1u64 << self.bits_per_neuron
    }
}

/// Batch training for sparse memory backend (parallel)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_train_batch(
    py: Python<'_>,
    memory: &SparseMemory,
    input_bits_flat: Vec<bool>,
    true_clusters: Vec<i64>,
    false_clusters_flat: Vec<i64>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    neurons_per_cluster: usize,
    num_negatives: usize,
    allow_override: bool,
) -> PyResult<usize> {
    py.allow_threads(|| {
        let modified = sparse_memory::train_batch_sparse(
            &memory.inner,
            &input_bits_flat,
            &true_clusters,
            &false_clusters_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            memory.bits_per_neuron,
            neurons_per_cluster,
            num_negatives,
            allow_override,
        );
        Ok(modified)
    })
}

/// Batch forward pass for sparse memory backend (parallel)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_forward_batch(
    py: Python<'_>,
    memory: &SparseMemory,
    input_bits_flat: Vec<bool>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        let probs = sparse_memory::forward_batch_sparse(
            &memory.inner,
            &input_bits_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            memory.bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
        );
        Ok(probs)
    })
}

/// Evaluate multiple connectivity patterns in parallel (for GA/TS optimization)
///
/// This is the KEY function for accelerating GA/TS optimization.
/// Instead of evaluating candidates sequentially in Python, this evaluates
/// all candidates in parallel using rayon, with each candidate getting
/// its own fresh sparse memory.
///
/// Expected speedup: 8-16x on 16-core M4 Max
///
/// Args:
///   candidates_flat: Flattened connectivity patterns [num_candidates * num_neurons * bits_per_neuron]
///   num_candidates: Number of candidate patterns
///   train_input_bits: Training input bits [num_train * total_input_bits]
///   train_true_clusters: Target clusters for training [num_train]
///   train_false_clusters: Negative clusters for training [num_train * num_negatives]
///   eval_input_bits: Evaluation input bits [num_eval * total_input_bits]
///   eval_targets: Target clusters for evaluation [num_eval]
///   num_train_examples: Number of training examples
///   num_eval_examples: Number of evaluation examples
///   total_input_bits: Total input bits (context_size * bits_per_token)
///   bits_per_neuron: Bits per neuron for this layer
///   neurons_per_cluster: Neurons per output cluster
///   num_clusters: Number of output clusters (vocab_size)
///   num_negatives: Number of negative samples per training example
///
/// Returns: Cross-entropy for each candidate [num_candidates]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_candidates_parallel(
    py: Python<'_>,
    candidates_flat: Vec<i64>,
    num_candidates: usize,
    train_input_bits: Vec<bool>,
    train_true_clusters: Vec<i64>,
    train_false_clusters: Vec<i64>,
    eval_input_bits: Vec<bool>,
    eval_targets: Vec<i64>,
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        let results = sparse_memory::evaluate_candidates_parallel(
            &candidates_flat,
            num_candidates,
            &train_input_bits,
            &train_true_clusters,
            &train_false_clusters,
            &eval_input_bits,
            &eval_targets,
            num_train_examples,
            num_eval_examples,
            total_input_bits,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            num_negatives,
        );
        Ok(results)
    })
}

/// Evaluate multiple TIERED connectivity patterns in parallel (for GA/TS optimization)
///
/// This is the KEY function for accelerating GA/TS optimization with tiered architectures.
/// Each candidate gets its own fresh tiered sparse memory, and all candidates are
/// evaluated in parallel using rayon.
///
/// Expected speedup: 8-16x on 16-core M4 Max (evaluating 50 candidates in parallel)
///
/// Args:
///   candidates_flat: Flattened connectivity patterns [num_candidates * conn_size_per_candidate]
///   num_candidates: Number of candidate patterns to evaluate
///   conn_size_per_candidate: Size of connections per candidate (sum of all tier neuron*bits)
///   train_input_bits: Pre-encoded training input bits [num_train * total_input_bits]
///   train_true_clusters: Target cluster indices for training [num_train]
///   train_false_clusters: Negative cluster indices for training [num_train * num_negatives]
///   eval_input_bits: Pre-encoded evaluation input bits [num_eval * total_input_bits]
///   eval_targets: Target cluster indices for evaluation [num_eval]
///   tier_configs_flat: Flattened tier configs [num_tiers * 3]: (end_cluster, neurons_per_cluster, bits_per_neuron)
///   num_tiers: Number of tiers
///   num_train_examples: Number of training examples
///   num_eval_examples: Number of evaluation examples
///   total_input_bits: Total input bits (context_size * bits_per_token)
///   num_clusters: Total number of output clusters (vocab_size)
///   num_negatives: Number of negative samples per training example
///
/// Returns: Cross-entropy for each candidate [num_candidates]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_candidates_parallel_tiered(
    py: Python<'_>,
    candidates_flat: Vec<i64>,
    num_candidates: usize,
    conn_size_per_candidate: usize,
    train_input_bits: Vec<bool>,
    train_true_clusters: Vec<i64>,
    train_false_clusters: Vec<i64>,
    eval_input_bits: Vec<bool>,
    eval_targets: Vec<i64>,
    tier_configs_flat: Vec<i64>,
    num_tiers: usize,
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        // Convert flat tier configs to tuple format
        let tier_configs: Vec<(usize, usize, usize)> = (0..num_tiers)
            .map(|i| {
                let base = i * 3;
                (
                    tier_configs_flat[base] as usize,      // end_cluster
                    tier_configs_flat[base + 1] as usize,  // neurons_per_cluster
                    tier_configs_flat[base + 2] as usize,  // bits_per_neuron
                )
            })
            .collect();

        let results = sparse_memory::evaluate_candidates_parallel_tiered(
            &candidates_flat,
            num_candidates,
            conn_size_per_candidate,
            &train_input_bits,
            &train_true_clusters,
            &train_false_clusters,
            &eval_input_bits,
            &eval_targets,
            &tier_configs,
            num_train_examples,
            num_eval_examples,
            total_input_bits,
            num_clusters,
            num_negatives,
        );
        Ok(results)
    })
}

/// Evaluate multiple TIERED connectivity patterns using HYBRID CPU+GPU (56 cores on M4 Max)
///
/// This is the ULTIMATE acceleration for GA/TS optimization:
/// - Training: CPU with DashMap (optimal for parallel writes)
/// - Evaluation: GPU with Metal binary search (massive parallelism)
/// - Candidates: Processed in parallel via rayon
///
/// Expected speedup: 20-40x on M4 Max (16 CPU + 40 GPU cores)
///
/// Args: Same as evaluate_candidates_parallel_tiered
/// Returns: Cross-entropy for each candidate [num_candidates]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_candidates_parallel_hybrid(
    py: Python<'_>,
    candidates_flat: Vec<i64>,
    num_candidates: usize,
    conn_size_per_candidate: usize,
    train_input_bits: Vec<bool>,
    train_true_clusters: Vec<i64>,
    train_false_clusters: Vec<i64>,
    eval_input_bits: Vec<bool>,
    eval_targets: Vec<i64>,
    tier_configs_flat: Vec<i64>,
    num_tiers: usize,
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        // Convert flat tier configs to tuple format
        let tier_configs: Vec<(usize, usize, usize)> = (0..num_tiers)
            .map(|i| {
                let base = i * 3;
                (
                    tier_configs_flat[base] as usize,      // end_cluster
                    tier_configs_flat[base + 1] as usize,  // neurons_per_cluster
                    tier_configs_flat[base + 2] as usize,  // bits_per_neuron
                )
            })
            .collect();

        let results = sparse_memory::evaluate_candidates_parallel_hybrid(
            &candidates_flat,
            num_candidates,
            conn_size_per_candidate,
            &train_input_bits,
            &train_true_clusters,
            &train_false_clusters,
            &eval_input_bits,
            &eval_targets,
            &tier_configs,
            num_train_examples,
            num_eval_examples,
            total_input_bits,
            num_clusters,
            num_negatives,
        );
        Ok(results)
    })
}

/// Check if sparse Metal GPU is available
#[pyfunction]
fn sparse_metal_available() -> bool {
    metal_ramlm::MetalSparseEvaluator::new().is_ok()
}

/// Python module definition
#[pymodule]
fn ram_accelerator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_connectivity_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch_metal, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_cascade_batch_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_fullnetwork_batch_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_fullnetwork_perplexity_batch_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(metal_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_cores, m)?)?;
    // New batch prediction functions
    m.add_function(wrap_pyfunction!(predict_all_batch_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(predict_all_batch_metal, m)?)?;
    m.add_function(wrap_pyfunction!(predict_all_batch_hybrid, m)?)?;
    // Exact probs acceleration (bit-encoded - deprecated, slow due to export)
    m.add_function(wrap_pyfunction!(compute_exact_probs_batch, m)?)?;
    // Exact probs acceleration (word-based - FAST, no export overhead)
    m.add_function(wrap_pyfunction!(compute_exact_probs_words, m)?)?;
    // RAMLM acceleration (proper RAM WNN architecture)
    m.add_function(wrap_pyfunction!(ramlm_train_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_train_batch_numpy, m)?)?;  // FAST numpy-based training
    m.add_function(wrap_pyfunction!(ramlm_train_batch_tiered_numpy, m)?)?;  // FAST tiered training (all tiers in one call)
    m.add_function(wrap_pyfunction!(ramlm_forward_batch, m)?)?;
    // RAMLM Metal GPU acceleration
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_metal, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_metal_available, m)?)?;
    // RAMLM NumPy-based acceleration (FAST - zero-copy)
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_metal_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_hybrid_numpy, m)?)?;
    // RAMLM Cached Metal (avoids shader recompilation)
    m.add_function(wrap_pyfunction!(ramlm_init_metal, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_metal_cached, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_hybrid_cached, m)?)?;
    // Sparse memory backend (for >10 bits per neuron)
    m.add_class::<SparseMemory>()?;
    m.add_function(wrap_pyfunction!(sparse_train_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_forward_batch, m)?)?;
    // Parallel GA/TS candidate evaluation (KEY optimization)
    m.add_function(wrap_pyfunction!(evaluate_candidates_parallel, m)?)?;
    // Parallel GA/TS for TIERED architectures (16 cores parallel)
    m.add_function(wrap_pyfunction!(evaluate_candidates_parallel_tiered, m)?)?;
    // Parallel GA/TS HYBRID CPU+GPU (56 cores - ULTIMATE optimization)
    m.add_function(wrap_pyfunction!(evaluate_candidates_parallel_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_metal_available, m)?)?;
    Ok(())
}
