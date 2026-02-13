//! RAM Accelerator - High-performance RAM neuron evaluation for Apple Silicon
//!
//! Provides GPU-accelerated evaluation of RAM neuron connectivity patterns
//! using Metal compute shaders on M-series Macs.
#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use numpy::PyReadonlyArray1;

// Global cached Metal evaluator for RAMLM (avoids shader recompilation)
// Using OnceLock + Option to handle initialization errors gracefully
static METAL_RAMLM_EVALUATOR: OnceLock<Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>> = OnceLock::new();

fn get_cached_metal_evaluator() -> Result<&'static Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>, String> {
    Ok(METAL_RAMLM_EVALUATOR.get_or_init(|| {
        Mutex::new(metal_ramlm::MetalRAMLMEvaluator::new().ok())
    }))
}

// Global cached Metal evaluator for Gating (avoids shader recompilation)
static METAL_GATING_EVALUATOR: OnceLock<Mutex<Option<metal_gating::MetalGatingEvaluator>>> = OnceLock::new();

fn get_cached_metal_gating_evaluator() -> Result<&'static Mutex<Option<metal_gating::MetalGatingEvaluator>>, String> {
    Ok(METAL_GATING_EVALUATOR.get_or_init(|| {
        Mutex::new(metal_gating::MetalGatingEvaluator::new().ok())
    }))
}

#[path = "ram.rs"]
mod ram;

#[path = "ramlm.rs"]
mod ramlm;

// Metal evaluator modules: real on macOS, stubs on other platforms
#[cfg(target_os = "macos")]
#[path = "metal_evaluator.rs"]
mod metal_evaluator;

#[cfg(not(target_os = "macos"))]
mod metal_evaluator {
    pub struct MetalEvaluator;
    impl MetalEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn is_available() -> bool { false }
        pub fn device_info() -> Result<String, String> { Err("Metal not available on this platform".into()) }
        pub fn evaluate_batch(
            &self, _: &[Vec<Vec<i64>>], _: &std::collections::HashMap<String, u64>,
            _: &[String], _: &[String], _: usize, _: usize,
        ) -> Result<Vec<f64>, String> { Err("Metal not available on this platform".into()) }
    }
}

#[cfg(target_os = "macos")]
#[path = "metal_ramlm.rs"]
mod metal_ramlm;

#[cfg(not(target_os = "macos"))]
mod metal_ramlm {
    pub struct MetalRAMLMEvaluator;
    impl MetalRAMLMEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn is_available() -> bool { false }
        pub fn device_info() -> Result<String, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch(
            &self, _: &[bool], _: &[i64], _: &[i64],
            _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct MetalSparseEvaluator;
    impl MetalSparseEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch_sparse(
            &self, _: &[bool], _: &[i64], _: &[u64], _: &[u8], _: &[u32], _: &[u32],
            _: usize, _: usize, _: usize, _: usize, _: usize, _: usize,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch_general(
            &self, _: &[bool], _: &[i64], _: &[u64], _: &[u8], _: &[u32], _: &[u32],
            _: &[(u32, u32, u32, u32)], _: usize, _: usize, _: usize,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct MetalGroupEvaluator;
    impl MetalGroupEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct MetalSparseCEEvaluator;
    impl MetalSparseCEEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct MetalCEReduceEvaluator;
    impl MetalCEReduceEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct SparseGroupData<'a> {
        pub connections: &'a [i64],
        pub keys: &'a [u64],
        pub values: &'a [u8],
        pub offsets: &'a [u32],
        pub counts: &'a [u32],
        pub cluster_ids: &'a [usize],
        pub bits_per_neuron: usize,
        pub neurons_per_cluster: usize,
        pub actual_neurons_per_cluster: Option<&'a [u32]>,
    }

    pub fn reset_sparse_buffer_cache() {}
    pub fn get_sparse_cache_generation() -> u64 { 0 }
}

#[path = "sparse_memory.rs"]
mod sparse_memory;

#[path = "per_cluster.rs"]
mod per_cluster;

#[path = "adaptive.rs"]
mod adaptive;

#[path = "token_cache.rs"]
mod token_cache;

#[path = "neighbor_search.rs"]
mod neighbor_search;

#[path = "gating.rs"]
mod gating;

#[cfg(target_os = "macos")]
#[path = "metal_gating.rs"]
mod metal_gating;

#[cfg(not(target_os = "macos"))]
mod metal_gating {
    pub struct MetalGatingEvaluator;
    impl MetalGatingEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn is_available() -> bool { false }
    }
    pub fn invalidate_gating_cache() {}
    pub fn reset_gating_buffer_cache() {}
}

#[path = "eval_worker.rs"]
pub mod eval_worker;

#[path = "bitwise_ramlm.rs"]
mod bitwise_ramlm;

pub use ram::RAMNeuron;
pub use per_cluster::{PerClusterEvaluator, FitnessMode, TierOptConfig, ClusterOptResult, TierOptResult};
#[cfg(target_os = "macos")]
pub use metal_evaluator::MetalEvaluator;
#[cfg(target_os = "macos")]
pub use metal_ramlm::MetalRAMLMEvaluator;

/// Set the EMPTY cell value for probability calculation
/// 0.0 = EMPTY cells don't contribute (no artificial competition) - RECOMMENDED
/// 0.5 = EMPTY cells add 0.5 probability (old default, inflates PPL with 50K classes)
#[pyfunction]
fn set_empty_value(value: f32) {
    ramlm::set_empty_value(value);
    sparse_memory::set_empty_value(value);
}

/// Get the current EMPTY cell value
#[pyfunction]
fn get_empty_value() -> f32 {
    ramlm::get_empty_value()
}

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

/// Reset Metal evaluators to free accumulated driver state.
///
/// Call this periodically (e.g., every 50 generations) during long optimization
/// runs to prevent slowdown from Metal driver state accumulation.
///
/// The evaluators will be lazily re-initialized on next use.
#[pyfunction]
fn reset_metal_evaluators() {
    adaptive::reset_metal_evaluators();
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

/// Batch predict using pre-trained RAMs (CPU rayon parallel)
#[pyfunction]
fn predict_all_batch(
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
fn ramlm_bitwise_train_batch_numpy<'py>(
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

/// Bitwise batch training for sparse memory backend
///
/// Multi-label training: each example trains ALL clusters (one per output bit).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_bitwise_train_batch(
    py: Python<'_>,
    memory: &SparseMemory,
    input_bits_flat: Vec<bool>,
    target_bits_flat: Vec<u8>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    allow_override: bool,
) -> PyResult<usize> {
    py.allow_threads(|| {
        let modified = sparse_memory::bitwise_train_batch_sparse(
            &memory.inner,
            &input_bits_flat,
            &target_bits_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            memory.bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
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

// =============================================================================
// TIERED SPARSE MEMORY (for variable bits-per-tier architectures)
// =============================================================================

/// Python wrapper for TieredSparseMemory
/// Provides tiered sparse storage for architectures with different bits per tier
#[pyclass]
struct TieredSparseMemory {
    inner: Arc<sparse_memory::TieredSparseMemory>,
    num_clusters: usize,
    tier_configs: Vec<(usize, usize, usize)>,  // (end_cluster, neurons_per_cluster, bits_per_neuron)
}

#[pymethods]
impl TieredSparseMemory {
    /// Create a new tiered sparse memory
    /// tier_configs: List of (end_cluster, neurons_per_cluster, bits_per_neuron) tuples
    ///               Tiers must be consecutive starting from 0
    #[new]
    fn new(tier_configs: Vec<(usize, usize, usize)>, num_clusters: usize) -> Self {
        Self {
            inner: Arc::new(sparse_memory::TieredSparseMemory::new(&tier_configs, num_clusters)),
            num_clusters,
            tier_configs,
        }
    }

    /// Get total number of written cells across all tiers
    fn total_cells(&self) -> usize {
        self.inner.total_cells()
    }

    /// Reset all memory to empty
    fn reset(&self) {
        self.inner.reset();
    }

    /// Get number of clusters
    #[getter]
    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get tier configurations
    #[getter]
    fn tier_configs(&self) -> Vec<(usize, usize, usize)> {
        self.tier_configs.clone()
    }
}

/// Batch training for tiered sparse memory backend (parallel)
/// Memory stays in Rust - only returns count of modified cells
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_train_batch_tiered(
    py: Python<'_>,
    memory: &TieredSparseMemory,
    input_bits_flat: Vec<bool>,
    true_clusters: Vec<i64>,
    false_clusters_flat: Vec<i64>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
) -> PyResult<usize> {
    py.allow_threads(|| {
        let modified = sparse_memory::train_batch_tiered(
            &memory.inner,
            &input_bits_flat,
            &true_clusters,
            &false_clusters_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            num_negatives,
        );
        Ok(modified)
    })
}

/// Batch training for tiered sparse memory using NumPy arrays (FAST)
///
/// Uses numpy arrays to avoid Python list conversion overhead.
/// Typically 3-5x faster than sparse_train_batch_tiered for large batches.
///
/// Args:
///   input_bits: [num_examples * total_input_bits] u8 numpy array (0/1 values)
///   true_clusters: [num_examples] i64 numpy array of logical cluster indices
///   false_clusters_flat: [num_examples * num_negatives] i64 numpy array
///   connections_flat: [total_neurons * max_bits_per_neuron] i64 numpy array
///
/// Returns: number of memory cells modified
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_train_batch_tiered_numpy<'py>(
    py: Python<'py>,
    memory: &TieredSparseMemory,
    input_bits: PyReadonlyArray1<'py, u8>,
    true_clusters: PyReadonlyArray1<'py, i64>,
    false_clusters_flat: PyReadonlyArray1<'py, i64>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
) -> PyResult<usize> {
    // Extract slices before allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let true_slice = true_clusters.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("True clusters array not contiguous: {}", e))
    })?;
    let false_slice = false_clusters_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("False clusters array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;

    // Convert to owned vecs (needed for Send across allow_threads)
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let true_vec: Vec<i64> = true_slice.to_vec();
    let false_vec: Vec<i64> = false_slice.to_vec();
    let conn_vec: Vec<i64> = conn_slice.to_vec();

    // Run training in parallel (releases GIL)
    py.allow_threads(|| {
        let modified = sparse_memory::train_batch_tiered(
            &memory.inner,
            &input_bools,
            &true_vec,
            &false_vec,
            &conn_vec,
            num_examples,
            total_input_bits,
            num_negatives,
        );
        Ok(modified)
    })
}

/// Batch forward pass for tiered sparse memory backend (parallel)
/// Legacy version using Vec - prefer sparse_forward_batch_tiered_numpy for speed
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_forward_batch_tiered(
    py: Python<'_>,
    memory: &TieredSparseMemory,
    input_bits_flat: Vec<bool>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        let probs = sparse_memory::forward_batch_tiered(
            &memory.inner,
            &input_bits_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
        );
        Ok(probs)
    })
}

/// Batch forward pass for tiered sparse memory using NumPy arrays (FAST)
///
/// Uses numpy arrays to avoid Python list conversion overhead.
/// Returns probabilities as a flat numpy array [num_examples * num_clusters].
///
/// Args:
///   input_bits: [num_examples * total_input_bits] u8 numpy array (0/1 values)
///   connections_flat: [total_neurons * max_bits_per_neuron] i64 numpy array
///
/// Returns: [num_examples * num_clusters] f32 numpy array of probabilities
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_forward_batch_tiered_numpy<'py>(
    py: Python<'py>,
    memory: &TieredSparseMemory,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
) -> PyResult<Py<numpy::PyArray1<f32>>> {
    // Extract slices before allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;

    // Convert u8 to bool
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();

    // Run forward pass in parallel (releases GIL)
    let probs = py.allow_threads(|| {
        sparse_memory::forward_batch_tiered(
            &memory.inner,
            &input_bools,
            &conn_vec,
            num_examples,
            total_input_bits,
        )
    });

    // Convert to numpy array
    Ok(numpy::PyArray1::from_vec(py, probs).into())
}

// =============================================================================
// SPARSE GPU CACHE - Cached GPU export for Metal forward pass
// =============================================================================

/// Cached GPU export data for sparse forward pass via Metal.
///
/// Holds the exported sparse memory (keys, values, offsets, counts, cluster_infos)
/// and a Metal evaluator. This avoids re-exporting on every forward call.
///
/// Usage:
///   cache = sparse_export_for_gpu(memory)           # For TieredRAMClusterLayer
///   cache = sparse_export_groups_for_gpu(...)        # For AdaptiveClusteredRAM
///   probs = sparse_forward_metal_numpy(cache, ...)   # Shared forward
#[pyclass]
struct SparseGpuCache {
    keys: Vec<u64>,
    values: Vec<u8>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    cluster_infos: Vec<(u32, u32, u32, u32)>,
    num_clusters: usize,
    evaluator: metal_ramlm::MetalSparseEvaluator,
}

#[pymethods]
impl SparseGpuCache {
    /// Get number of clusters
    #[getter]
    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get total entries in sparse memory
    #[getter]
    fn total_entries(&self) -> usize {
        self.keys.len()
    }

    /// Get memory size in bytes (approximate)
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4
            + self.counts.len() * 4 + self.cluster_infos.len() * 16
    }
}

/// Export a TieredSparseMemory to GPU cache for Metal forward pass.
///
/// For TieredRAMClusterLayer: exports single memory with all tiers.
#[pyfunction]
fn sparse_export_for_gpu(
    py: Python<'_>,
    memory: &TieredSparseMemory,
) -> PyResult<SparseGpuCache> {
    py.allow_threads(|| {
        let export = memory.inner.export_for_gpu_general();
        let evaluator = metal_ramlm::MetalSparseEvaluator::new()
            .map_err(|e| format!("Failed to create Metal evaluator: {}", e))?;

        Ok(SparseGpuCache {
            keys: export.keys,
            values: export.values,
            offsets: export.offsets,
            counts: export.counts,
            cluster_infos: export.cluster_infos,
            num_clusters: export.num_clusters,
            evaluator,
        })
    }).map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Export multiple TieredSparseMemory groups to a single GPU cache.
///
/// For AdaptiveClusteredRAM: combines multiple group exports, mapping each
/// group's clusters to their correct global cluster positions.
///
/// Args:
///   memories: List of TieredSparseMemory objects (one per config group)
///   cluster_ids_per_group: List of cluster ID lists (mapping local → global)
///   num_clusters: Total number of clusters across all groups
#[pyfunction]
fn sparse_export_groups_for_gpu(
    py: Python<'_>,
    memories: Vec<pyo3::PyRef<'_, TieredSparseMemory>>,
    cluster_ids_per_group: Vec<Vec<usize>>,
    num_clusters: usize,
) -> PyResult<SparseGpuCache> {
    // Extract Arc references before releasing GIL (PyRef can't cross thread boundary)
    let inner_refs: Vec<Arc<sparse_memory::TieredSparseMemory>> = memories.iter()
        .map(|m| Arc::clone(&m.inner))
        .collect();

    py.allow_threads(|| {
        let mut all_keys: Vec<u64> = Vec::new();
        let mut all_values: Vec<u8> = Vec::new();
        let mut all_offsets: Vec<u32> = Vec::new();
        let mut all_counts: Vec<u32> = Vec::new();
        let mut all_cluster_infos: Vec<(u32, u32, u32, u32)> = vec![(0, 0, 0, 0); num_clusters];

        let mut global_neuron_base: u32 = 0;
        let mut global_conn_offset: u32 = 0;

        for (group_idx, inner) in inner_refs.iter().enumerate() {
            let export = inner.export_for_gpu_general();
            let cluster_ids = &cluster_ids_per_group[group_idx];

            // Adjust offsets for global arrays
            let key_base = all_keys.len() as u32;
            for &off in &export.offsets {
                all_offsets.push(key_base + off);
            }
            all_counts.extend(&export.counts);
            all_keys.extend(&export.keys);
            all_values.extend(&export.values);

            // Map cluster_infos from local to global positions
            for (local_idx, &(neurons, bits, start_neuron, conn_offset)) in export.cluster_infos.iter().enumerate() {
                if local_idx < cluster_ids.len() {
                    let global_cluster = cluster_ids[local_idx];
                    all_cluster_infos[global_cluster] = (
                        neurons,
                        bits,
                        global_neuron_base + start_neuron,
                        global_conn_offset + conn_offset,
                    );
                }
            }

            // Update bases for next group
            let total_neurons: u32 = export.offsets.len() as u32;
            global_neuron_base += total_neurons;

            // Connection offset: sum of all neurons * bits across this group's tiers
            let group_conn_size: u32 = export.cluster_infos.iter()
                .map(|&(n, b, _, _)| n * b)
                .sum();
            global_conn_offset += group_conn_size;
        }

        let evaluator = metal_ramlm::MetalSparseEvaluator::new()
            .map_err(|e| format!("Failed to create Metal evaluator: {}", e))?;

        Ok(SparseGpuCache {
            keys: all_keys,
            values: all_values,
            offsets: all_offsets,
            counts: all_counts,
            cluster_infos: all_cluster_infos,
            num_clusters,
            evaluator,
        })
    }).map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Run Metal GPU forward pass using cached export data.
///
/// Args:
///   cache: SparseGpuCache from sparse_export_for_gpu or sparse_export_groups_for_gpu
///   input_bits: [num_examples * total_input_bits] u8 numpy array
///   connections_flat: [total_connections] i64 numpy array
///   num_examples: batch size
///   total_input_bits: bits per example
///
/// Returns: [num_examples * num_clusters] f32 numpy array
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_forward_metal_numpy<'py>(
    py: Python<'py>,
    cache: &SparseGpuCache,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
) -> PyResult<Py<numpy::PyArray1<f32>>> {
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;

    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();

    let probs = py.allow_threads(|| {
        cache.evaluator.forward_batch_general(
            &input_bools,
            &conn_vec,
            &cache.keys,
            &cache.values,
            &cache.offsets,
            &cache.counts,
            &cache.cluster_infos,
            num_examples,
            total_input_bits,
            cache.num_clusters,
        ).map_err(|e| format!("Metal forward failed: {}", e))
    }).map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(numpy::PyArray1::from_vec(py, probs).into())
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

/// Hybrid CPU+GPU evaluation with explicit memory budget
///
/// Same as evaluate_candidates_parallel_hybrid but with memory_budget_gb parameter
/// to control memory usage. Use this when you want to limit RAM consumption.
///
/// Args:
///   memory_budget_gb: Maximum memory to use for sparse memory pool (in GB).
///                     If None, auto-detects and uses 60% of available RAM.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (candidates_flat, num_candidates, conn_size_per_candidate, train_input_bits, train_true_clusters, train_false_clusters, eval_input_bits, eval_targets, tier_configs_flat, num_tiers, num_train_examples, num_eval_examples, total_input_bits, num_clusters, num_negatives, memory_budget_gb=None))]
fn evaluate_candidates_parallel_hybrid_with_budget(
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
    memory_budget_gb: Option<f64>,
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

        let results = match memory_budget_gb {
            Some(budget) => sparse_memory::evaluate_candidates_parallel_hybrid_with_budget(
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
                budget,
            ),
            None => sparse_memory::evaluate_candidates_parallel_hybrid(
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
            ),
        };
        Ok(results)
    })
}

/// Get estimated memory per sparse memory object (for debugging/tuning)
#[pyfunction]
fn estimate_sparse_memory_gb(
    tier_configs_flat: Vec<i64>,
    num_tiers: usize,
    num_clusters: usize,
    num_train_examples: usize,
    num_negatives: usize,
) -> f64 {
    let tier_configs: Vec<(usize, usize, usize)> = (0..num_tiers)
        .map(|i| {
            let base = i * 3;
            (
                tier_configs_flat[base] as usize,
                tier_configs_flat[base + 1] as usize,
                tier_configs_flat[base + 2] as usize,
            )
        })
        .collect();

    let bytes = sparse_memory::estimate_memory_per_tiered_sparse(
        &tier_configs,
        num_clusters,
        num_train_examples,
        num_negatives,
    );
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Get available system memory in GB
#[pyfunction]
fn get_system_memory_gb() -> f64 {
    sparse_memory::get_available_memory_gb()
}

/// Check if sparse Metal GPU is available
#[pyfunction]
fn sparse_metal_available() -> bool {
    metal_ramlm::MetalSparseEvaluator::new().is_ok()
}

// ============================================================================
// Per-Cluster Optimization (Rust-accelerated discriminative optimization)
// ============================================================================

// Global cache for per-cluster evaluators (shared across all functions)
static PER_CLUSTER_EVALUATORS: OnceLock<Mutex<Vec<per_cluster::PerClusterEvaluator>>> = OnceLock::new();

fn get_per_cluster_evaluators() -> &'static Mutex<Vec<per_cluster::PerClusterEvaluator>> {
    PER_CLUSTER_EVALUATORS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Create a per-cluster evaluator for discriminative optimization
///
/// This is the Rust-accelerated version of the Python IncrementalEvaluator.
/// Use this for per-cluster GA/TS optimization with discriminative fitness.
///
/// Args:
///     train_contexts_flat: Training contexts as flat bools [num_train * context_bits]
///     train_targets: Training target cluster IDs [num_train]
///     eval_contexts_flat: Eval contexts as flat bools [num_eval * context_bits]
///     eval_targets: Eval target cluster IDs [num_eval]
///     context_bits: Total bits per context
///     cluster_neurons: List of (cluster_id, start_neuron, end_neuron)
///     cluster_bits: List of (cluster_id, bits_per_neuron)
///     num_clusters: Total number of clusters (vocab size)
///
/// Returns:
///     Evaluator ID for subsequent calls
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn per_cluster_create_evaluator(
    _py: Python<'_>,
    train_contexts_flat: Vec<bool>,
    train_targets: Vec<usize>,
    eval_contexts_flat: Vec<bool>,
    eval_targets: Vec<usize>,
    context_bits: usize,
    cluster_neurons: Vec<(usize, usize, usize)>,  // (cluster_id, start, end)
    cluster_bits: Vec<(usize, usize)>,            // (cluster_id, bits)
    num_clusters: usize,
) -> PyResult<usize> {
    // Build maps
    let mut cluster_to_neurons = FxHashMap::default();
    let mut cluster_to_bits = FxHashMap::default();

    for (cluster_id, start, end) in cluster_neurons {
        cluster_to_neurons.insert(cluster_id, (start, end));
    }
    for (cluster_id, bits) in cluster_bits {
        cluster_to_bits.insert(cluster_id, bits);
    }

    let evaluator = per_cluster::PerClusterEvaluator::new(
        &train_contexts_flat,
        &train_targets,
        &eval_contexts_flat,
        &eval_targets,
        context_bits,
        cluster_to_neurons,
        cluster_to_bits,
        num_clusters,
    );

    // Store in global cache and return ID
    let mut guard = get_per_cluster_evaluators().lock().unwrap();
    let id = guard.len();
    guard.push(evaluator);

    Ok(id)
}

/// Evaluate multiple connectivity variants for a cluster (batch)
///
/// This is the key acceleration point - evaluates entire GA population
/// or TS neighbor set in parallel using rayon.
///
/// Args:
///     evaluator_id: ID from per_cluster_create_evaluator
///     cluster_id: Cluster being optimized
///     variants: List of connectivity variants [num_variants][neurons * bits]
///     fitness_mode: 1=PositiveOnly, 2=PenalizeWins, 3=PenalizeHighVotes (default)
///
/// Returns:
///     Fitness scores for each variant [num_variants]
#[pyfunction]
fn per_cluster_evaluate_batch(
    py: Python<'_>,
    evaluator_id: usize,
    cluster_id: usize,
    variants: Vec<Vec<i64>>,
    fitness_mode: i32,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        let guard = get_per_cluster_evaluators().lock().unwrap();

        if evaluator_id >= guard.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid evaluator ID"));
        }

        let mode = per_cluster::FitnessMode::from(fitness_mode);
        let results = guard[evaluator_id].evaluate_variants_batch(cluster_id, &variants, mode);
        Ok(results)
    })
}

/// Optimize a single cluster using GA
///
/// Args:
///     evaluator_id: ID from per_cluster_create_evaluator
///     cluster_id: Cluster to optimize
///     initial_connectivity: Starting connectivity [neurons * bits]
///     ga_gens: Number of GA generations
///     ga_population: Population size
///     mutation_rate: Mutation probability per connection
///     seed: Random seed
///
/// Returns:
///     (final_connectivity, initial_fitness, final_fitness, improvement_pct)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn per_cluster_optimize_ga(
    py: Python<'_>,
    evaluator_id: usize,
    cluster_id: usize,
    initial_connectivity: Vec<i64>,
    ga_gens: usize,
    ga_population: usize,
    mutation_rate: f64,
    seed: u64,
) -> PyResult<(Vec<i64>, f64, f64, f64)> {
    py.allow_threads(|| {
        let guard = get_per_cluster_evaluators().lock().unwrap();

        if evaluator_id >= guard.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid evaluator ID"));
        }

        let config = per_cluster::TierOptConfig {
            tier: 0,
            ga_gens,
            ga_population,
            ts_iters: 0,
            ts_neighbors: 0,
            mutation_rate,
            enabled: true,
            fitness_mode: per_cluster::FitnessMode::SimpleDiscriminative,
        };

        let result = guard[evaluator_id].optimize_cluster_ga(
            cluster_id,
            &initial_connectivity,
            &config,
            seed,
        );

        Ok((
            result.final_connectivity,
            result.initial_fitness,
            result.final_fitness,
            result.improvement_pct,
        ))
    })
}

/// Optimize a single cluster using Tabu Search
///
/// Args:
///     evaluator_id: ID from per_cluster_create_evaluator
///     cluster_id: Cluster to optimize
///     initial_connectivity: Starting connectivity [neurons * bits]
///     ts_iters: Number of TS iterations
///     ts_neighbors: Neighbors per iteration
///     mutation_rate: Mutation probability
///     seed: Random seed
///
/// Returns:
///     (final_connectivity, initial_fitness, final_fitness, improvement_pct)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn per_cluster_optimize_ts(
    py: Python<'_>,
    evaluator_id: usize,
    cluster_id: usize,
    initial_connectivity: Vec<i64>,
    ts_iters: usize,
    ts_neighbors: usize,
    mutation_rate: f64,
    seed: u64,
) -> PyResult<(Vec<i64>, f64, f64, f64)> {
    py.allow_threads(|| {
        let guard = get_per_cluster_evaluators().lock().unwrap();

        if evaluator_id >= guard.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid evaluator ID"));
        }

        let config = per_cluster::TierOptConfig {
            tier: 0,
            ga_gens: 0,
            ga_population: 0,
            ts_iters,
            ts_neighbors,
            mutation_rate,
            enabled: true,
            fitness_mode: per_cluster::FitnessMode::SimpleDiscriminative,
        };

        let result = guard[evaluator_id].optimize_cluster_ts(
            cluster_id,
            &initial_connectivity,
            &config,
            seed,
        );

        Ok((
            result.final_connectivity,
            result.initial_fitness,
            result.final_fitness,
            result.improvement_pct,
        ))
    })
}

/// Precompute global baseline votes for ALL clusters (enables true global CE)
///
/// This enables exact global CE computation during optimization.
/// Memory: num_eval * num_clusters * 8 bytes (e.g., 4K * 50K * 8 = 1.6GB)
/// Time: ~3s for 50K clusters (parallelized with rayon)
///
/// Call this ONCE before optimization starts. After this, all CE fitness
/// computations will use true global softmax over all 50K clusters.
///
/// Args:
///     evaluator_id: ID from per_cluster_create_evaluator
///     all_connectivities: Dict mapping cluster_id -> connectivity for ALL clusters
///
/// Returns:
///     True if successful
#[pyfunction]
fn per_cluster_precompute_global_baseline(
    py: Python<'_>,
    evaluator_id: usize,
    all_connectivities: std::collections::HashMap<usize, Vec<i64>>,
) -> PyResult<bool> {
    py.allow_threads(|| {
        let mut guard = get_per_cluster_evaluators().lock().unwrap();

        if evaluator_id >= guard.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid evaluator ID"));
        }

        // Convert to FxHashMap
        let conns: FxHashMap<usize, Vec<i64>> = all_connectivities.into_iter().collect();

        guard[evaluator_id].precompute_global_baseline(&conns);

        Ok(true)
    })
}

/// Update global baseline for a specific cluster after optimization
///
/// Call this after optimizing a cluster to update the cached baseline
/// with the new connectivity's votes. This keeps the cache accurate
/// for subsequent cluster optimizations.
///
/// Args:
///     evaluator_id: ID from per_cluster_create_evaluator
///     cluster_id: Cluster that was just optimized
///     new_connectivity: The optimized connectivity pattern
///
/// Returns:
///     True if successful
#[pyfunction]
fn per_cluster_update_global_baseline(
    py: Python<'_>,
    evaluator_id: usize,
    cluster_id: usize,
    new_connectivity: Vec<i64>,
) -> PyResult<bool> {
    py.allow_threads(|| {
        let mut guard = get_per_cluster_evaluators().lock().unwrap();

        if evaluator_id >= guard.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid evaluator ID"));
        }

        // Compute new votes for this cluster
        let new_votes = guard[evaluator_id].train_and_vote(cluster_id, &new_connectivity);

        // Update the global baseline
        guard[evaluator_id].update_global_baseline(cluster_id, &new_votes);

        Ok(true)
    })
}

/// Optimize all clusters in a tier (parallel)
///
/// This is the main acceleration function - optimizes all clusters
/// in a tier using parallel processing (rayon).
///
/// Args:
///     evaluator_id: ID from per_cluster_create_evaluator
///     tier: Tier index
///     cluster_ids: List of cluster IDs in this tier
///     initial_connectivities: Dict mapping cluster_id -> connectivity
///     ga_gens, ga_population, ts_iters, ts_neighbors, mutation_rate: Config
///     seed: Base random seed
///
/// Returns:
///     List of (cluster_id, final_connectivity, initial_fitness, final_fitness, improvement_pct)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn per_cluster_optimize_tier(
    py: Python<'_>,
    evaluator_id: usize,
    tier: usize,
    cluster_ids: Vec<usize>,
    initial_connectivities: std::collections::HashMap<usize, Vec<i64>>,
    ga_gens: usize,
    ga_population: usize,
    ts_iters: usize,
    ts_neighbors: usize,
    mutation_rate: f64,
    seed: u64,
    fitness_mode: i32,
) -> PyResult<Vec<(usize, Vec<i64>, f64, f64, f64)>> {
    py.allow_threads(|| {
        let mut guard = get_per_cluster_evaluators().lock().unwrap();

        if evaluator_id >= guard.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid evaluator ID"));
        }

        let config = per_cluster::TierOptConfig {
            tier,
            ga_gens,
            ga_population,
            ts_iters,
            ts_neighbors,
            mutation_rate,
            enabled: true,
            fitness_mode: per_cluster::FitnessMode::from(fitness_mode),
        };

        // Convert to FxHashMap
        let conns: FxHashMap<usize, Vec<i64>> = initial_connectivities.into_iter().collect();

        let result = guard[evaluator_id].optimize_tier(
            tier,
            &cluster_ids,
            &conns,
            &config,
            seed,
        );

        // Convert to Python-friendly format
        let py_results: Vec<(usize, Vec<i64>, f64, f64, f64)> = result
            .cluster_results
            .into_iter()
            .map(|r| (
                r.cluster_id,
                r.final_connectivity,
                r.initial_fitness,
                r.final_fitness,
                r.improvement_pct,
            ))
            .collect();

        Ok(py_results)
    })
}

/// Optimize clusters in groups for joint optimization (iterative refinement)
///
/// Clusters are split into groups of `group_size` and each group is optimized
/// jointly. This captures inter-cluster competition (if cluster A gets stronger,
/// cluster B must adapt). Groups are processed in parallel.
///
/// Args:
///     evaluator_id: ID from per_cluster_create_evaluator
///     tier: Tier index
///     cluster_ids: List of cluster IDs in this tier
///     initial_connectivities: Dict mapping cluster_id -> connectivity
///     group_size: Number of clusters per group (e.g., 10 for top tokens)
///     ga_gens, ga_population, ts_iters, ts_neighbors, mutation_rate: Config
///     seed: Base random seed
///
/// Returns:
///     List of (cluster_id, final_connectivity, initial_fitness, final_fitness, improvement_pct)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn per_cluster_optimize_tier_grouped(
    py: Python<'_>,
    evaluator_id: usize,
    tier: usize,
    cluster_ids: Vec<usize>,
    initial_connectivities: std::collections::HashMap<usize, Vec<i64>>,
    group_size: usize,
    ga_gens: usize,
    ga_population: usize,
    ts_iters: usize,
    ts_neighbors: usize,
    mutation_rate: f64,
    seed: u64,
    fitness_mode: i32,
) -> PyResult<Vec<(usize, Vec<i64>, f64, f64, f64)>> {
    py.allow_threads(|| {
        let guard = get_per_cluster_evaluators().lock().unwrap();

        if evaluator_id >= guard.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid evaluator ID"));
        }

        let config = per_cluster::TierOptConfig {
            tier,
            ga_gens,
            ga_population,
            ts_iters,
            ts_neighbors,
            mutation_rate,
            enabled: true,
            fitness_mode: per_cluster::FitnessMode::from(fitness_mode),
        };

        // Convert to FxHashMap
        let conns: FxHashMap<usize, Vec<i64>> = initial_connectivities.into_iter().collect();

        let result = guard[evaluator_id].optimize_tier_grouped(
            &cluster_ids,
            &conns,
            group_size,
            &config,
            seed,
        );

        // Convert to Python-friendly format
        let py_results: Vec<(usize, Vec<i64>, f64, f64, f64)> = result
            .cluster_results
            .into_iter()
            .map(|r| (
                r.cluster_id,
                r.final_connectivity,
                r.initial_fitness,
                r.final_fitness,
                r.improvement_pct,
            ))
            .collect();

        Ok(py_results)
    })
}

// ============================================================================
// Adaptive Architecture (per-cluster variable bits/neurons)
// ============================================================================

/// Python wrapper for ConfigGroup data
#[pyclass]
#[derive(Clone)]
struct AdaptiveConfigGroup {
    #[pyo3(get)]
    neurons: usize,
    #[pyo3(get)]
    bits: usize,
    #[pyo3(get)]
    words_per_neuron: usize,
    #[pyo3(get)]
    cluster_ids: Vec<usize>,
    #[pyo3(get)]
    actual_neurons: Option<Vec<u32>>,  // Per-cluster actual neurons (for coalesced groups)
    #[pyo3(get)]
    memory_offset: usize,
    #[pyo3(get)]
    conn_offset: usize,
}

#[pymethods]
impl AdaptiveConfigGroup {
    #[new]
    fn new(neurons: usize, bits: usize, cluster_ids: Vec<usize>) -> Self {
        let group = adaptive::ConfigGroup::new(neurons, bits, cluster_ids.clone());
        Self {
            neurons: group.neurons,
            bits: group.bits,
            words_per_neuron: group.words_per_neuron,
            cluster_ids,
            actual_neurons: None,  // Uniform group (not coalesced)
            memory_offset: 0,
            conn_offset: 0,
        }
    }

    fn cluster_count(&self) -> usize {
        self.cluster_ids.len()
    }

    fn total_neurons(&self) -> usize {
        self.cluster_ids.len() * self.neurons
    }

    fn memory_size(&self) -> usize {
        self.total_neurons() * self.words_per_neuron
    }

    fn conn_size(&self) -> usize {
        self.total_neurons() * self.bits
    }
}

/// Build config groups from per-cluster configuration
///
/// Groups clusters by their (neurons, bits) to enable efficient batch processing.
///
/// Args:
///     bits_per_cluster: Number of bits for each cluster [num_clusters]
///     neurons_per_cluster: Number of neurons for each cluster [num_clusters]
///
/// Returns: List of AdaptiveConfigGroup objects with computed offsets
#[pyfunction]
fn adaptive_build_config_groups(
    bits_per_cluster: Vec<usize>,
    neurons_per_cluster: Vec<usize>,
) -> Vec<AdaptiveConfigGroup> {
    // Use build_groups which checks WNN_COALESCE_GROUPS env var
    let groups = adaptive::build_groups(&bits_per_cluster, &neurons_per_cluster);
    groups
        .into_iter()
        .map(|g| AdaptiveConfigGroup {
            neurons: g.neurons,
            bits: g.bits,
            words_per_neuron: g.words_per_neuron,
            cluster_ids: g.cluster_ids,
            actual_neurons: g.actual_neurons,
            memory_offset: g.memory_offset,
            conn_offset: g.conn_offset,
        })
        .collect()
}

/// Forward pass for adaptive architecture (CPU, parallel with rayon)
///
/// Processes each config group efficiently, then scatters results to output.
///
/// Args:
///     input_bits: [num_examples * total_input_bits] u8 numpy array (0/1)
///     connections_flat: All groups' connections concatenated [total_conns]
///     memory_words: All groups' memory concatenated [total_memory]
///     group_neurons: Per-group neurons [num_groups]
///     group_bits: Per-group bits [num_groups]
///     group_words_per_neuron: Per-group words_per_neuron [num_groups]
///     group_cluster_ids_flat: Flattened cluster IDs for all groups
///     group_cluster_counts: Number of clusters per group [num_groups]
///     group_memory_offsets: Memory offset per group [num_groups]
///     group_conn_offsets: Connection offset per group [num_groups]
///     num_examples: Number of input examples
///     total_input_bits: Total input bits per example
///     num_clusters: Total number of clusters (vocab size)
///
/// Returns: [num_examples * num_clusters] probabilities
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adaptive_forward_batch<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    group_neurons: Vec<usize>,
    group_bits: Vec<usize>,
    _group_words_per_neuron: Vec<usize>,
    group_cluster_ids_flat: Vec<usize>,
    group_cluster_counts: Vec<usize>,
    group_memory_offsets: Vec<usize>,
    group_conn_offsets: Vec<usize>,
    num_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
) -> PyResult<Vec<f32>> {
    // Extract data before allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert to owned data
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    // Reconstruct ConfigGroups
    let num_groups = group_neurons.len();
    let mut groups = Vec::with_capacity(num_groups);
    let mut cluster_offset = 0;

    for i in 0..num_groups {
        let cluster_count = group_cluster_counts[i];
        let cluster_ids = group_cluster_ids_flat[cluster_offset..cluster_offset + cluster_count].to_vec();
        cluster_offset += cluster_count;

        let mut group = adaptive::ConfigGroup::new(group_neurons[i], group_bits[i], cluster_ids);
        group.memory_offset = group_memory_offsets[i];
        group.conn_offset = group_conn_offsets[i];
        groups.push(group);
    }

    py.allow_threads(|| {
        let probs = adaptive::forward_batch_adaptive(
            &input_bools,
            &conn_vec,
            &mem_vec,
            &groups,
            num_examples,
            total_input_bits,
            num_clusters,
        );
        Ok(probs)
    })
}

/// Training for adaptive architecture (CPU, parallel with rayon)
///
/// Two-phase training: TRUE first, then FALSE (to ensure TRUE priority).
///
/// Args:
///     input_bits: [num_examples * total_input_bits] u8 numpy array
///     true_clusters: [num_examples] target cluster indices
///     false_clusters_flat: [num_examples * num_negatives] negative cluster indices
///     connections_flat: All groups' connections concatenated
///     memory_words: All groups' memory (modified in place)
///     group_neurons, group_bits, etc.: Group configuration (same as forward)
///     num_examples, total_input_bits, num_negatives, num_clusters: Dimensions
///     allow_override: Whether to allow overwriting non-EMPTY cells
///
/// Returns: Number of cells modified
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adaptive_train_batch<'py>(
    _py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    true_clusters: PyReadonlyArray1<'py, i64>,
    false_clusters_flat: PyReadonlyArray1<'py, i64>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    mut memory_words: numpy::PyReadwriteArray1<'py, i64>,
    group_neurons: Vec<usize>,
    group_bits: Vec<usize>,
    _group_words_per_neuron: Vec<usize>,
    group_cluster_ids_flat: Vec<usize>,
    group_cluster_counts: Vec<usize>,
    group_memory_offsets: Vec<usize>,
    group_conn_offsets: Vec<usize>,
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
    num_clusters: usize,
    allow_override: bool,
) -> PyResult<usize> {
    // Extract data before allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let true_slice = true_clusters.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("True clusters not contiguous: {}", e))
    })?;
    let false_slice = false_clusters_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("False clusters not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice_mut().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert to owned data (for bools and connections)
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let true_vec: Vec<i64> = true_slice.to_vec();
    let false_vec: Vec<i64> = false_slice.to_vec();
    let conn_vec: Vec<i64> = conn_slice.to_vec();

    // Reconstruct ConfigGroups
    let num_groups = group_neurons.len();
    let mut groups = Vec::with_capacity(num_groups);
    let mut cluster_offset = 0;

    for i in 0..num_groups {
        let cluster_count = group_cluster_counts[i];
        let cluster_ids = group_cluster_ids_flat[cluster_offset..cluster_offset + cluster_count].to_vec();
        cluster_offset += cluster_count;

        let mut group = adaptive::ConfigGroup::new(group_neurons[i], group_bits[i], cluster_ids);
        group.memory_offset = group_memory_offsets[i];
        group.conn_offset = group_conn_offsets[i];
        groups.push(group);
    }

    // Note: We can't use py.allow_threads here because we need mutable access to mem_slice
    // The Rust function uses atomics internally, so it's thread-safe
    let modified = adaptive::train_batch_adaptive(
        &input_bools,
        &true_vec,
        &false_vec,
        &conn_vec,
        mem_slice,
        &groups,
        num_examples,
        total_input_bits,
        num_negatives,
        num_clusters,
        allow_override,
    );

    Ok(modified)
}

/// Evaluate multiple genomes in parallel using Rust/rayon
///
/// This is the KEY acceleration for GA optimization - evaluates all genomes
/// concurrently using a thread pool (16 threads on M4 Max).
///
/// Memory efficient: ~200MB per active genome, not ~2GB like Python multiprocessing.
///
/// Args:
///   genomes_bits_flat: [num_genomes * num_clusters] bits per cluster
///   genomes_neurons_flat: [num_genomes * num_clusters] neurons per cluster
///   num_genomes: Number of genomes to evaluate
///   num_clusters: Vocabulary size
///   train_input_bits: [num_train * total_input_bits] training contexts
///   train_targets: [num_train] target clusters
///   train_negatives: [num_train * num_negatives] negative samples
///   num_train: Number of training examples
///   num_negatives: Negatives per example
///   eval_input_bits: [num_eval * total_input_bits] eval contexts
///   eval_targets: [num_eval] eval targets
///   num_eval: Number of eval examples
///   total_input_bits: Bits per context
///   empty_value: Value for EMPTY cells (0.0 recommended)
///
/// Returns: [num_genomes] cross-entropy values
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_genomes_parallel<'py>(
    py: Python<'py>,
    genomes_bits_flat: Vec<usize>,
    genomes_neurons_flat: Vec<usize>,
    genomes_connections_flat: Vec<i64>,  // NEW: flattened connections (empty = random)
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: PyReadonlyArray1<'py, u8>,
    train_targets: PyReadonlyArray1<'py, i64>,
    train_negatives: PyReadonlyArray1<'py, i64>,
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: PyReadonlyArray1<'py, u8>,
    eval_targets: PyReadonlyArray1<'py, i64>,
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> PyResult<Vec<(f64, f64)>> {
    // Returns Vec of (cross_entropy, accuracy) tuples - one per genome
    // Extract data before allow_threads
    let train_input_slice = train_input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train input not contiguous: {}", e))
    })?;
    let train_targets_slice = train_targets.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train targets not contiguous: {}", e))
    })?;
    let train_negatives_slice = train_negatives.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train negatives not contiguous: {}", e))
    })?;
    let eval_input_slice = eval_input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Eval input not contiguous: {}", e))
    })?;
    let eval_targets_slice = eval_targets.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Eval targets not contiguous: {}", e))
    })?;

    // Convert to owned data
    let train_input_bools: Vec<bool> = train_input_slice.iter().map(|&b| b != 0).collect();
    let train_targets_vec: Vec<i64> = train_targets_slice.to_vec();
    let train_negatives_vec: Vec<i64> = train_negatives_slice.to_vec();
    let eval_input_bools: Vec<bool> = eval_input_slice.iter().map(|&b| b != 0).collect();
    let eval_targets_vec: Vec<i64> = eval_targets_slice.to_vec();

    // Set empty value for this evaluation
    ramlm::set_empty_value(empty_value);

    py.allow_threads(|| {
        let fitness = adaptive::evaluate_genomes_parallel(
            &genomes_bits_flat,
            &genomes_neurons_flat,
            &genomes_connections_flat,  // Pass connections to Rust
            num_genomes,
            num_clusters,
            &train_input_bools,
            &train_targets_vec,
            &train_negatives_vec,
            num_train,
            num_negatives,
            &eval_input_bools,
            &eval_targets_vec,
            num_eval,
            total_input_bits,
            empty_value,
        );
        Ok(fitness)
    })
}

/// Evaluate genomes with multi-subset rotation support.
///
/// All token subsets are pre-encoded and passed at once. The train_subset_idx
/// and eval_subset_idx select which subset to use for this batch of evaluations.
///
/// This enables per-generation/iteration rotation of data subsets, acting as
/// a regularizer that forces genomes to generalize across all subsets.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_genomes_parallel_multisubset<'py>(
    py: Python<'py>,
    genomes_bits_flat: Vec<usize>,
    genomes_neurons_flat: Vec<usize>,
    genomes_connections_flat: Vec<i64>,
    num_genomes: usize,
    num_clusters: usize,
    // All train subsets concatenated
    train_subsets_flat: PyReadonlyArray1<'py, u8>,
    train_targets_flat: PyReadonlyArray1<'py, i64>,
    train_negatives_flat: PyReadonlyArray1<'py, i64>,
    train_subset_counts: Vec<usize>,
    // All eval subsets concatenated
    eval_subsets_flat: PyReadonlyArray1<'py, u8>,
    eval_targets_flat: PyReadonlyArray1<'py, i64>,
    eval_subset_counts: Vec<usize>,
    // Subset selection
    train_subset_idx: usize,
    eval_subset_idx: usize,
    // Other params
    num_negatives: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> PyResult<Vec<(f64, f64)>> {
    // Extract data before allow_threads
    let train_subsets_slice = train_subsets_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train subsets not contiguous: {}", e))
    })?;
    let train_targets_slice = train_targets_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train targets not contiguous: {}", e))
    })?;
    let train_negatives_slice = train_negatives_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train negatives not contiguous: {}", e))
    })?;
    let eval_subsets_slice = eval_subsets_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Eval subsets not contiguous: {}", e))
    })?;
    let eval_targets_slice = eval_targets_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Eval targets not contiguous: {}", e))
    })?;

    // Convert to owned data
    let train_subsets_bools: Vec<bool> = train_subsets_slice.iter().map(|&b| b != 0).collect();
    let train_targets_vec: Vec<i64> = train_targets_slice.to_vec();
    let train_negatives_vec: Vec<i64> = train_negatives_slice.to_vec();
    let eval_subsets_bools: Vec<bool> = eval_subsets_slice.iter().map(|&b| b != 0).collect();
    let eval_targets_vec: Vec<i64> = eval_targets_slice.to_vec();

    // Set empty value for this evaluation
    ramlm::set_empty_value(empty_value);

    py.allow_threads(|| {
        let fitness = adaptive::evaluate_genomes_parallel_multisubset(
            &genomes_bits_flat,
            &genomes_neurons_flat,
            &genomes_connections_flat,
            num_genomes,
            num_clusters,
            &train_subsets_bools,
            &train_targets_vec,
            &train_negatives_vec,
            &train_subset_counts,
            &eval_subsets_bools,
            &eval_targets_vec,
            &eval_subset_counts,
            train_subset_idx,
            eval_subset_idx,
            num_negatives,
            total_input_bits,
            empty_value,
        );
        Ok(fitness)
    })
}

/// Evaluate genomes with parallel hybrid CPU+GPU evaluation.
///
/// This is the high-performance variant that uses:
/// - Memory pool for parallel genome training (8 parallel)
/// - GPU batch evaluation for multiple genomes
/// - CPU+GPU hybrid split (dense ≤12 bits on CPU, sparse >12 bits on GPU)
/// - Pipelining (CPU trains batch N+1 while GPU evaluates batch N)
///
/// Expected speedup: 4-8x over sequential `evaluate_genomes_parallel`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_genomes_parallel_hybrid<'py>(
    py: Python<'py>,
    genomes_bits_flat: Vec<usize>,
    genomes_neurons_flat: Vec<usize>,
    genomes_connections_flat: Vec<i64>,
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: PyReadonlyArray1<'py, u8>,
    train_targets: PyReadonlyArray1<'py, i64>,
    train_negatives: PyReadonlyArray1<'py, i64>,
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: PyReadonlyArray1<'py, u8>,
    eval_targets: PyReadonlyArray1<'py, i64>,
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> PyResult<Vec<(f64, f64)>> {
    // Extract data before allow_threads
    let train_input_slice = train_input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train input not contiguous: {}", e))
    })?;
    let train_targets_slice = train_targets.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train targets not contiguous: {}", e))
    })?;
    let train_negatives_slice = train_negatives.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Train negatives not contiguous: {}", e))
    })?;
    let eval_input_slice = eval_input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Eval input not contiguous: {}", e))
    })?;
    let eval_targets_slice = eval_targets.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Eval targets not contiguous: {}", e))
    })?;

    // Convert to owned data
    let train_input_bools: Vec<bool> = train_input_slice.iter().map(|&b| b != 0).collect();
    let train_targets_vec: Vec<i64> = train_targets_slice.to_vec();
    let train_negatives_vec: Vec<i64> = train_negatives_slice.to_vec();
    let eval_input_bools: Vec<bool> = eval_input_slice.iter().map(|&b| b != 0).collect();
    let eval_targets_vec: Vec<i64> = eval_targets_slice.to_vec();

    // Set empty value for this evaluation
    ramlm::set_empty_value(empty_value);

    py.allow_threads(|| {
        let fitness = adaptive::evaluate_genomes_parallel_hybrid(
            &genomes_bits_flat,
            &genomes_neurons_flat,
            &genomes_connections_flat,
            num_genomes,
            num_clusters,
            &train_input_bools,
            &train_targets_vec,
            &train_negatives_vec,
            num_train,
            num_negatives,
            &eval_input_bools,
            &eval_targets_vec,
            num_eval,
            total_input_bits,
            empty_value,
        );
        Ok(fitness)
    })
}

// =============================================================================
// TOKEN CACHE - Persistent token storage with subset rotation
// =============================================================================

/// Python-accessible TokenCache for persistent token storage.
///
/// Create once at session start, then use for all evaluations without
/// any data transfer overhead.
#[pyclass]
struct TokenCacheWrapper {
    inner: token_cache::TokenCache,
}

#[pymethods]
impl TokenCacheWrapper {
    /// Create a new token cache with all data pre-encoded and partitioned.
    ///
    /// # Arguments
    /// * `encoding_table` - Optional semantic encoding table (token_id → semantic_bits).
    ///   If provided, tokens are encoded using learned semantic bits instead of raw binary.
    ///   Similar tokens will have similar bit patterns, enabling better generalization.
    ///   Pre-computed in Python using MutualInfoEncoder or similar.
    /// * `encoding_bits` - Number of bits in semantic encoding (required if encoding_table provided).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (train_tokens, eval_tokens, test_tokens, vocab_size, context_size, cluster_order, num_parts, num_negatives, seed, encoding_table=None, encoding_bits=None, num_eval_parts=None))]
    fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        test_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        cluster_order: Vec<usize>,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        encoding_table: Option<Vec<u64>>,
        encoding_bits: Option<usize>,
        num_eval_parts: Option<usize>,
    ) -> Self {
        Self {
            inner: token_cache::TokenCache::new(
                train_tokens,
                eval_tokens,
                test_tokens,
                vocab_size,
                context_size,
                cluster_order,
                num_parts,
                num_negatives,
                seed,
                encoding_table,
                encoding_bits,
                num_eval_parts.unwrap_or(1),
            ),
        }
    }

    /// Get the next train subset index (advances rotator).
    fn next_train_idx(&mut self) -> usize {
        self.inner.next_train_idx()
    }

    /// Get the next eval subset index (advances rotator).
    fn next_eval_idx(&mut self) -> usize {
        self.inner.next_eval_idx()
    }

    /// Reset rotators with optional new seed.
    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) {
        self.inner.reset(seed);
    }

    /// Get number of train subsets.
    fn num_train_subsets(&self) -> usize {
        self.inner.num_train_subsets()
    }

    /// Get vocab size.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get total input bits.
    fn total_input_bits(&self) -> usize {
        self.inner.total_input_bits()
    }

    /// Evaluate genomes using a specific train/eval subset combination.
    ///
    /// This is the main evaluation function - zero data copy, just uses
    /// pre-cached data selected by indices.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        empty_value: f32,
    ) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| {
            Ok(token_cache::evaluate_genomes_cached(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                train_subset_idx,
                eval_subset_idx,
                empty_value,
            ))
        })
    }

    /// Evaluate genomes using full train/eval data (for final evaluation).
    fn evaluate_genomes_full(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        empty_value: f32,
    ) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| {
            Ok(token_cache::evaluate_genomes_cached_full(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                empty_value,
            ))
        })
    }

    /// Evaluate genomes using hybrid CPU+GPU parallel evaluation (4-8x speedup).
    ///
    /// Uses memory pool for parallel training, GPU batch evaluation, and pipelining.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        empty_value: f32,
    ) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| {
            Ok(token_cache::evaluate_genomes_cached_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                train_subset_idx,
                eval_subset_idx,
                empty_value,
            ))
        })
    }

    /// Evaluate genomes using full data with hybrid CPU+GPU (4-8x speedup).
    fn evaluate_genomes_full_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        empty_value: f32,
    ) -> PyResult<Vec<(f64, f64)>> {
        py.allow_threads(|| {
            Ok(token_cache::evaluate_genomes_cached_full_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                empty_value,
            ))
        })
    }

    /// Evaluate a single genome WITH gating, returning both gated and non-gated metrics.
    ///
    /// This function:
    /// 1. Trains base RAM on full training data
    /// 2. Trains gating model on training data (target gate = true only for target cluster)
    /// 3. Evaluates WITHOUT gating → (ce, acc)
    /// 4. Evaluates WITH gating → (gated_ce, gated_acc)
    ///
    /// # Arguments
    /// * `bits_flat` - Bits per cluster [num_clusters]
    /// * `neurons_flat` - Neurons per cluster [num_clusters]
    /// * `connections_flat` - Connections [total_connections]
    /// * `neurons_per_gate` - Number of RAM neurons per gate (default 8)
    /// * `bits_per_gate_neuron` - Address bits per gate neuron (default 12)
    /// * `vote_threshold_frac` - Fraction of neurons that must fire for gate=1 (default 0.5)
    /// * `empty_value` - Value for EMPTY cells (default 0.5)
    /// * `gating_seed` - Random seed for gating connectivity
    ///
    /// # Returns
    /// (ce, accuracy, gated_ce, gated_accuracy)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        bits_flat,
        neurons_flat,
        connections_flat,
        neurons_per_gate = 8,
        bits_per_gate_neuron = 12,
        vote_threshold_frac = 0.5,
        empty_value = 0.5,
        gating_seed = 42
    ))]
    fn evaluate_genome_with_gating(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        neurons_per_gate: usize,
        bits_per_gate_neuron: usize,
        vote_threshold_frac: f32,
        empty_value: f32,
        gating_seed: u64,
    ) -> PyResult<(f64, f64, f64, f64)> {
        py.allow_threads(|| {
            Ok(token_cache::evaluate_genome_with_gating(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                neurons_per_gate,
                bits_per_gate_neuron,
                vote_threshold_frac,
                empty_value,
                gating_seed,
            ))
        })
    }

    /// Search for neighbors above accuracy threshold, all in Rust.
    ///
    /// This eliminates Python↔Rust round trips by doing mutation, evaluation,
    /// and filtering entirely in Rust. Logs progress to file with flush.
    ///
    /// Returns: List of (bits_flat, neurons_flat, connections_flat, CE, accuracy)
    /// for candidates that passed the threshold.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        base_bits,
        base_neurons,
        base_connections,
        target_count,
        max_attempts,
        accuracy_threshold,
        min_bits,
        max_bits,
        min_neurons,
        max_neurons,
        bits_mutation_rate,
        neurons_mutation_rate,
        train_subset_idx,
        eval_subset_idx,
        empty_value,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None
    ))]
    fn search_neighbors(
        &self,
        py: Python<'_>,
        base_bits: Vec<usize>,
        base_neurons: Vec<usize>,
        base_connections: Vec<i64>,
        target_count: usize,
        max_attempts: usize,
        accuracy_threshold: f64,
        min_bits: usize,
        max_bits: usize,
        min_neurons: usize,
        max_neurons: usize,
        bits_mutation_rate: f64,
        neurons_mutation_rate: f64,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        empty_value: f32,
        seed: u64,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64)>> {
        let num_clusters = base_bits.len();
        let total_input_bits = self.inner.total_input_bits();

        let config = neighbor_search::MutationConfig {
            num_clusters,
            mutable_clusters,  // None = all clusters, Some(indices) = only those
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            total_input_bits,
        };

        py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();

            let candidates = if return_best_n {
                neighbor_search::search_neighbors_best_n(
                    &self.inner,
                    &base_bits,
                    &base_neurons,
                    &base_connections,
                    target_count,
                    max_attempts,
                    accuracy_threshold,
                    &config,
                    train_subset_idx,
                    eval_subset_idx,
                    empty_value,
                    seed,
                    log_path_ref,
                    generation,
                    total_generations,
                )
            } else {
                let (passed, _) = neighbor_search::search_neighbors_with_threshold(
                    &self.inner,
                    &base_bits,
                    &base_neurons,
                    &base_connections,
                    target_count,
                    max_attempts,
                    accuracy_threshold,
                    &config,
                    train_subset_idx,
                    eval_subset_idx,
                    empty_value,
                    seed,
                    log_path_ref,
                    generation,
                    total_generations,
                );
                passed
            };

            // Convert to Python-friendly format
            Ok(candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_cluster,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                ))
                .collect())
        })
    }

    /// Search for GA offspring above accuracy threshold, all in Rust.
    ///
    /// Performs tournament selection, crossover, mutation, and evaluation
    /// entirely in Rust. Returns viable offspring (accuracy >= threshold).
    ///
    /// Args:
    ///   - population: List of (bits, neurons, connections, fitness) tuples
    ///   - target_count: Number of viable offspring needed
    ///   - max_attempts: Maximum offspring to generate
    ///   - accuracy_threshold: Minimum accuracy for viable offspring
    ///
    /// Returns: List of (bits, neurons, connections, CE, accuracy) tuples
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        population,
        target_count,
        max_attempts,
        accuracy_threshold,
        min_bits,
        max_bits,
        min_neurons,
        max_neurons,
        bits_mutation_rate,
        neurons_mutation_rate,
        crossover_rate,
        tournament_size,
        train_subset_idx,
        eval_subset_idx,
        empty_value,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None
    ))]
    fn search_offspring(
        &self,
        py: Python<'_>,
        population: Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64)>,
        target_count: usize,
        max_attempts: usize,
        accuracy_threshold: f64,
        min_bits: usize,
        max_bits: usize,
        min_neurons: usize,
        max_neurons: usize,
        bits_mutation_rate: f64,
        neurons_mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        empty_value: f32,
        seed: u64,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
    ) -> PyResult<(Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64)>, usize, usize)> {
        // Returns: (candidates, evaluated, viable)
        let num_clusters = if !population.is_empty() {
            population[0].0.len()
        } else {
            return Ok((Vec::new(), 0, 0));
        };
        let total_input_bits = self.inner.total_input_bits();

        let ga_config = neighbor_search::GAConfig {
            num_clusters,
            mutable_clusters,  // None = all clusters, Some(indices) = only those
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            crossover_rate,
            tournament_size,
            total_input_bits,
        };

        py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();

            let result = neighbor_search::search_offspring(
                &self.inner,
                &population,
                target_count,
                max_attempts,
                accuracy_threshold,
                &ga_config,
                train_subset_idx,
                eval_subset_idx,
                empty_value,
                seed,
                log_path_ref,
                generation,
                total_generations,
                return_best_n,
            );

            // Convert to Python-friendly format: (candidates, evaluated, viable)
            let candidates: Vec<_> = result.candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_cluster,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                ))
                .collect();
            Ok((candidates, result.evaluated, result.viable))
        })
    }
}

// =============================================================================
// RAMGating Python Wrapper
// =============================================================================

/// Python wrapper for RAM-based gating
///
/// Uses dedicated RAM neurons to learn which clusters should be active
/// for each input context. Gate output is binary (0 or 1) via majority voting.
#[pyclass]
struct RAMGatingWrapper {
    inner: gating::RAMGating,
}

#[pymethods]
impl RAMGatingWrapper {
    /// Create a new RAMGating model
    ///
    /// # Arguments
    /// * `num_clusters` - Number of clusters to gate (vocabulary size)
    /// * `neurons_per_gate` - Number of RAM neurons per cluster (default 8)
    /// * `bits_per_neuron` - Address bits per neuron (default 12)
    /// * `total_input_bits` - Total input bits (context_size * bits_per_token)
    /// * `threshold` - Fraction of neurons that must fire (default 0.5)
    /// * `seed` - Random seed for connectivity initialization
    #[new]
    #[pyo3(signature = (num_clusters, neurons_per_gate, bits_per_neuron, total_input_bits, threshold=0.5, seed=None))]
    fn new(
        num_clusters: usize,
        neurons_per_gate: usize,
        bits_per_neuron: usize,
        total_input_bits: usize,
        threshold: f32,
        seed: Option<u64>,
    ) -> Self {
        Self {
            inner: gating::RAMGating::new(
                num_clusters,
                neurons_per_gate,
                bits_per_neuron,
                total_input_bits,
                threshold,
                seed,
            ),
        }
    }

    /// Compute binary gates for a batch of inputs
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples in batch
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters] (0.0 or 1.0)
    fn forward_batch(&self, py: Python<'_>, input_bits_flat: Vec<bool>, batch_size: usize) -> Vec<f32> {
        py.allow_threads(|| self.inner.forward_batch(&input_bits_flat, batch_size))
    }

    /// Train gate neurons for a batch of examples
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `target_gates_flat` - Flattened target gates [batch_size * num_clusters]
    /// * `batch_size` - Number of examples
    /// * `allow_override` - Whether to override non-EMPTY cells
    ///
    /// # Returns
    /// Total cells modified across batch
    fn train_batch(
        &self,
        py: Python<'_>,
        input_bits_flat: Vec<bool>,
        target_gates_flat: Vec<bool>,
        batch_size: usize,
        allow_override: bool,
    ) -> usize {
        py.allow_threads(|| {
            self.inner.train_batch(&input_bits_flat, &target_gates_flat, batch_size, allow_override)
        })
    }

    /// Reset all memory cells to EMPTY
    fn reset(&self) {
        self.inner.reset();
    }

    /// Get statistics about memory usage
    ///
    /// # Returns
    /// (empty_count, false_count, true_count)
    fn memory_stats(&self) -> (usize, usize, usize) {
        self.inner.memory_stats()
    }

    /// Export memory state as bytes (for serialization)
    fn export_memory(&self) -> Vec<u8> {
        self.inner.export_memory()
    }

    /// Import memory state from bytes
    fn import_memory(&self, data: Vec<u8>) -> PyResult<()> {
        self.inner.import_memory(&data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })
    }

    /// Get total number of gate neurons
    fn total_neurons(&self) -> usize {
        self.inner.total_neurons()
    }

    /// Get gating configuration
    fn config(&self) -> (usize, usize, usize, usize, usize) {
        let cfg = self.inner.config();
        (cfg.num_clusters, cfg.neurons_per_gate, cfg.bits_per_neuron, cfg.total_input_bits, cfg.vote_threshold)
    }

    /// Compute binary gates on Metal GPU
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples in batch
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters] (0.0 or 1.0)
    #[cfg(target_os = "macos")]
    fn forward_batch_metal(&self, py: Python<'_>, input_bits_flat: Vec<bool>, batch_size: usize) -> PyResult<Vec<f32>> {
        py.allow_threads(|| {
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;
            evaluator.forward_batch(&self.inner, &input_bits_flat, batch_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Compute binary gates with hybrid CPU+GPU
    ///
    /// Splits the batch between CPU (rayon) and GPU (Metal) for maximum throughput.
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples
    /// * `cpu_fraction` - Fraction of batch to process on CPU (0.0-1.0, default 0.3)
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters] (0.0 or 1.0)
    #[cfg(target_os = "macos")]
    #[pyo3(signature = (input_bits_flat, batch_size, cpu_fraction=0.3))]
    fn forward_batch_hybrid(&self, py: Python<'_>, input_bits_flat: Vec<bool>, batch_size: usize, cpu_fraction: f32) -> PyResult<Vec<f32>> {
        py.allow_threads(|| {
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;
            metal_gating::forward_batch_hybrid(&self.inner, evaluator, &input_bits_flat, batch_size, cpu_fraction)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Train gate neurons on Metal GPU
    ///
    /// Uses Metal compute shaders with atomic memory writes for parallel training.
    /// After training, memory is synced back from GPU.
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `target_gates_flat` - Flattened target gates [batch_size * num_clusters]
    /// * `batch_size` - Number of training examples
    ///
    /// # Returns
    /// Number of training examples processed (batch_size)
    #[cfg(target_os = "macos")]
    fn train_batch_metal(
        &self,
        py: Python<'_>,
        input_bits_flat: Vec<bool>,
        target_gates_flat: Vec<bool>,
        batch_size: usize,
    ) -> PyResult<usize> {
        py.allow_threads(|| {
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;

            // Train on GPU and get updated memory
            let updated_memory = evaluator.train_batch(&self.inner, &input_bits_flat, &target_gates_flat, batch_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Import the updated memory back into the gating model
            self.inner.import_memory(&updated_memory)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Invalidate forward cache since memory changed
            metal_gating::invalidate_gating_cache();

            Ok(batch_size)
        })
    }
}

/// Check if Metal gating is available
#[pyfunction]
fn gating_metal_available() -> bool {
    metal_gating::MetalGatingEvaluator::is_available()
}

/// Invalidate gating cache (call after training to force memory re-upload)
#[pyfunction]
fn invalidate_gating_cache() {
    metal_gating::invalidate_gating_cache();
}

/// Reset and clear all gating buffer caches to free GPU memory
/// Call this when done with gating operations to prevent memory leaks
#[pyfunction]
fn reset_gating_buffer_cache() {
    metal_gating::reset_gating_buffer_cache();
}

/// Apply gates to cluster scores element-wise
///
/// Multiplies each cluster score by its gate value.
///
/// # Arguments
/// * `scores` - [batch_size * num_clusters] cluster scores
/// * `gates` - [batch_size * num_clusters] gate values (0.0 or 1.0)
///
/// # Returns
/// Gated scores [batch_size * num_clusters]
#[pyfunction]
fn apply_gates(scores: Vec<f32>, gates: Vec<f32>) -> Vec<f32> {
    gating::apply_gates(&scores, &gates)
}

/// Compute target gates from target cluster indices
///
/// Creates a boolean vector where only the target cluster is true for each example.
/// Used for training gating from supervised targets.
///
/// # Arguments
/// * `targets` - [batch_size] target cluster indices
/// * `num_clusters` - Number of clusters (vocabulary size)
///
/// # Returns
/// Flattened target gates [batch_size * num_clusters]
#[pyfunction]
fn compute_target_gates(targets: Vec<i64>, num_clusters: usize) -> Vec<bool> {
    gating::compute_target_gates(&targets, num_clusters)
}

/// Compute target gates with neighbor expansion
///
/// Creates target gates where the target cluster and its neighbors are set to true.
/// This allows gating to learn contextual relevance beyond just the exact target.
///
/// # Arguments
/// * `targets` - [batch_size] target cluster indices
/// * `num_clusters` - Number of clusters
/// * `neighbor_clusters` - [batch_size * neighbors_per_example] neighbor indices
/// * `neighbors_per_example` - Number of neighbors per example
///
/// # Returns
/// Flattened target gates [batch_size * num_clusters]
#[pyfunction]
fn compute_target_gates_expanded(
    targets: Vec<i64>,
    num_clusters: usize,
    neighbor_clusters: Vec<i64>,
    neighbors_per_example: usize,
) -> Vec<bool> {
    gating::compute_target_gates_expanded(
        &targets,
        num_clusters,
        Some(&neighbor_clusters),
        neighbors_per_example,
    )
}

// =============================================================================
// Bitwise RAMLM — Nudge Training + Quad Forward (PyO3 wrappers)
// =============================================================================

/// Bitwise batch training with 4-state nudging (QUAD modes).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_bitwise_train_batch_nudge_numpy<'py>(
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

/// Neuron-parallel training — much faster than example-parallel for all modes.
/// Supports TERNARY (0), QUAD_BINARY (1), QUAD_WEIGHTED (2).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_bitwise_train_neuron_parallel_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    target_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    memory_mode: u8,
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
        let modified = ramlm::bitwise_train_neuron_parallel(
            &input_bools,
            &target_vec,
            &conn_vec,
            &mut mem_vec,
            num_examples,
            total_input_bits,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            words_per_neuron,
            memory_mode,
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
fn ramlm_bitwise_train_and_eval_numpy<'py>(
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
        );
        Ok((ce, acc, per_bit_acc, mem_vec))
    })
}

/// Forward pass for QUAD_BINARY mode: P = count(cell >= 2) / neurons_per_cluster
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ramlm_forward_batch_quad_binary_numpy<'py>(
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
fn ramlm_forward_batch_quad_weighted_numpy<'py>(
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

// =============================================================================
// Bitwise RAMLM Cache Wrapper (PyO3)
// =============================================================================

/// Persistent cache for bitwise genome evaluation.
///
/// Holds pre-encoded tokens in Rust memory. Evaluates genomes
/// entirely in Rust+Metal (no Python overhead per genome).
#[pyclass]
struct BitwiseCacheWrapper {
    inner: bitwise_ramlm::BitwiseTokenCache,
}

#[pymethods]
impl BitwiseCacheWrapper {
    #[new]
    fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        num_parts: usize,
        num_eval_parts: usize,
        pad_token_id: u32,
    ) -> Self {
        Self {
            inner: bitwise_ramlm::BitwiseTokenCache::new(
                train_tokens, eval_tokens, vocab_size, context_size,
                num_parts, num_eval_parts, pad_token_id,
            ),
        }
    }

    /// Evaluate genomes with per-cluster heterogeneous configs (subset training + eval).
    ///
    /// bits_per_cluster_flat: [num_genomes * num_clusters]
    /// neurons_per_cluster_flat: [num_genomes * num_clusters]
    /// connections_flat: variable total (sum of all genomes' connections)
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes(
        &self,
        py: Python<'_>,
        bits_per_cluster_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64)>> {
        py.allow_threads(|| {
            Ok(bitwise_ramlm::evaluate_genomes(
                &self.inner, &bits_per_cluster_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes, train_subset_idx, eval_subset_idx,
                memory_mode, neuron_sample_rate, rng_seed,
            ))
        })
    }

    /// Evaluate genomes with per-cluster heterogeneous configs (full training + full eval).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes_full(
        &self,
        py: Python<'_>,
        bits_per_cluster_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64)>> {
        py.allow_threads(|| {
            Ok(bitwise_ramlm::evaluate_genomes_full(
                &self.inner, &bits_per_cluster_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes,
                memory_mode, neuron_sample_rate, rng_seed,
            ))
        })
    }

    /// Get next train subset index (advances rotator).
    fn next_train_idx(&self) -> usize {
        self.inner.next_train_idx()
    }

    /// Get next eval subset index (advances rotator).
    fn next_eval_idx(&self) -> usize {
        self.inner.next_eval_idx()
    }

    /// Reset subset rotation (both train and eval).
    fn reset(&self) {
        self.inner.reset();
    }

    fn vocab_size(&self) -> usize { self.inner.vocab_size }
    fn total_input_bits(&self) -> usize { self.inner.total_input_bits }
    fn num_parts(&self) -> usize { self.inner.num_parts }
    fn num_eval_parts(&self) -> usize { self.inner.num_eval_parts }
    fn num_bits(&self) -> usize { self.inner.num_bits }
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
    m.add_function(wrap_pyfunction!(reset_metal_evaluators, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_cores, m)?)?;
    // New batch prediction functions
    m.add_function(wrap_pyfunction!(predict_all_batch_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(predict_all_batch, m)?)?;
    m.add_function(wrap_pyfunction!(predict_all_batch_hybrid, m)?)?;
    // Exact probs acceleration (bit-encoded - deprecated, slow due to export)
    m.add_function(wrap_pyfunction!(compute_exact_probs_batch, m)?)?;
    // Exact probs acceleration (word-based - FAST, no export overhead)
    m.add_function(wrap_pyfunction!(compute_exact_probs_words, m)?)?;
    // RAMLM acceleration (proper RAM WNN architecture)
    m.add_function(wrap_pyfunction!(ramlm_train_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_train_batch_numpy, m)?)?;  // FAST numpy-based training
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_batch_numpy, m)?)?;  // Bitwise multi-label training (dense)
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
    m.add_function(wrap_pyfunction!(sparse_bitwise_train_batch, m)?)?;  // Bitwise multi-label training (sparse)
    m.add_function(wrap_pyfunction!(sparse_forward_batch, m)?)?;
    // Tiered sparse memory backend (for variable bits per tier)
    m.add_class::<TieredSparseMemory>()?;
    m.add_function(wrap_pyfunction!(sparse_train_batch_tiered, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_train_batch_tiered_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_forward_batch_tiered, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_forward_batch_tiered_numpy, m)?)?;
    // Metal GPU sparse forward (cached export for fast repeated forward)
    m.add_class::<SparseGpuCache>()?;
    m.add_function(wrap_pyfunction!(sparse_export_for_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_export_groups_for_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_forward_metal_numpy, m)?)?;
    // Parallel GA/TS candidate evaluation (KEY optimization)
    m.add_function(wrap_pyfunction!(evaluate_candidates_parallel, m)?)?;
    // Parallel GA/TS for TIERED architectures (16 cores parallel)
    m.add_function(wrap_pyfunction!(evaluate_candidates_parallel_tiered, m)?)?;
    // Parallel GA/TS HYBRID CPU+GPU (memory-adaptive, pipelined)
    m.add_function(wrap_pyfunction!(evaluate_candidates_parallel_hybrid, m)?)?;
    // Hybrid with explicit memory budget
    m.add_function(wrap_pyfunction!(evaluate_candidates_parallel_hybrid_with_budget, m)?)?;
    // Memory estimation utilities
    m.add_function(wrap_pyfunction!(estimate_sparse_memory_gb, m)?)?;
    m.add_function(wrap_pyfunction!(get_system_memory_gb, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_metal_available, m)?)?;
    // Per-cluster optimization (Rust-accelerated discriminative optimization)
    m.add_function(wrap_pyfunction!(per_cluster_create_evaluator, m)?)?;
    m.add_function(wrap_pyfunction!(per_cluster_evaluate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(per_cluster_optimize_ga, m)?)?;
    m.add_function(wrap_pyfunction!(per_cluster_optimize_ts, m)?)?;
    m.add_function(wrap_pyfunction!(per_cluster_optimize_tier, m)?)?;
    // Global CE with caching (true global softmax over all 50K clusters)
    m.add_function(wrap_pyfunction!(per_cluster_precompute_global_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(per_cluster_update_global_baseline, m)?)?;
    // Group-based optimization for iterative refinement
    m.add_function(wrap_pyfunction!(per_cluster_optimize_tier_grouped, m)?)?;
    // EMPTY cell value configuration (affects PPL calculation)
    m.add_function(wrap_pyfunction!(set_empty_value, m)?)?;
    m.add_function(wrap_pyfunction!(get_empty_value, m)?)?;
    // Adaptive architecture (per-cluster variable bits/neurons)
    m.add_class::<AdaptiveConfigGroup>()?;
    m.add_function(wrap_pyfunction!(adaptive_build_config_groups, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_train_batch, m)?)?;
    // Parallel genome evaluation (KEY for GA optimization)
    m.add_function(wrap_pyfunction!(evaluate_genomes_parallel, m)?)?;
    // Multi-subset parallel genome evaluation (for per-iteration rotation)
    m.add_function(wrap_pyfunction!(evaluate_genomes_parallel_multisubset, m)?)?;
    // Parallel hybrid CPU+GPU genome evaluation (4-8x speedup)
    m.add_function(wrap_pyfunction!(evaluate_genomes_parallel_hybrid, m)?)?;
    // Token cache for persistent token storage with subset rotation
    m.add_class::<TokenCacheWrapper>()?;
    // RAM-based gating (weightless per-cluster gating with majority voting)
    m.add_class::<RAMGatingWrapper>()?;
    m.add_function(wrap_pyfunction!(apply_gates, m)?)?;
    m.add_function(wrap_pyfunction!(compute_target_gates, m)?)?;
    m.add_function(wrap_pyfunction!(compute_target_gates_expanded, m)?)?;
    m.add_function(wrap_pyfunction!(gating_metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(invalidate_gating_cache, m)?)?;
    m.add_function(wrap_pyfunction!(reset_gating_buffer_cache, m)?)?;
    // Bitwise RAMLM evaluation (full Rust+Metal pipeline)
    m.add_class::<BitwiseCacheWrapper>()?;
    // Bitwise RAMLM — nudge training + quad forward
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_batch_nudge_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_neuron_parallel_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_and_eval_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_quad_binary_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_quad_weighted_numpy, m)?)?;
    Ok(())
}
