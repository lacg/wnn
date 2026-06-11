//! RAM Accelerator - High-performance RAM neuron evaluation for Apple Silicon
//!
//! Provides GPU-accelerated evaluation of RAM neuron connectivity patterns
//! using Metal compute shaders on M-series Macs.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::SeedableRng;
use std::sync::{Arc, RwLock};
use std::sync::{Mutex, OnceLock};
use numpy::PyReadonlyArray1;

/// Release-mode validation of the flat-genome triple at the PyO3 boundary.
///
/// Wraps `adaptive::validate_flat_genomes` into a `PyValueError` so a
/// misaligned batch fails loudly instead of being scored with silently
/// shifted offsets (the internal `debug_assert`s are compiled out under
/// `--release`).
fn validate_flat_genomes_py(
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
) -> PyResult<()> {
    adaptive::validate_flat_genomes(bits_flat, neurons_flat, connections_flat, num_genomes, num_clusters)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

// Global cached Metal evaluator for RAMLM (avoids shader recompilation)
// Using OnceLock + Option to handle initialization errors gracefully
static METAL_RAMLM_EVALUATOR: OnceLock<Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>> = OnceLock::new();

fn get_cached_metal_evaluator() -> Result<&'static Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>, String> {
    Ok(METAL_RAMLM_EVALUATOR.get_or_init(|| {
        Mutex::new(metal_ramlm::MetalRAMLMEvaluator::new().ok())
    }))
}

// Global cached Metal trainer for GPU address computation during training
static METAL_TRAINER: OnceLock<Mutex<Option<metal_train::MetalTrainer>>> = OnceLock::new();

fn get_cached_metal_trainer() -> Result<&'static Mutex<Option<metal_train::MetalTrainer>>, String> {
    Ok(METAL_TRAINER.get_or_init(|| {
        Mutex::new(metal_train::MetalTrainer::new().ok())
    }))
}

// Global cached Metal evaluator for Gating (avoids shader recompilation)
static METAL_GATING_EVALUATOR: OnceLock<Mutex<Option<metal_gating::MetalGatingEvaluator>>> = OnceLock::new();

fn get_cached_metal_gating_evaluator() -> Result<&'static Mutex<Option<metal_gating::MetalGatingEvaluator>>, String> {
    Ok(METAL_GATING_EVALUATOR.get_or_init(|| {
        Mutex::new(metal_gating::MetalGatingEvaluator::new().ok())
    }))
}

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
            &self, _: &[u64], _: &[i64], _: &[i64],
            _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: u8,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct MetalSparseEvaluator;
    impl MetalSparseEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch_sparse(
            &self, _: &[u64], _: &[i64], _: &[u64], _: &[u8], _: &[u32], _: &[u32],
            _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: u8,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch_general(
            &self, _: &[u64], _: &[i64], _: &[u64], _: &[u8], _: &[u32], _: &[u32],
            _: &[(u32, u32, u32, u32)], _: usize, _: usize, _: usize, _: u8,
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

#[path = "neuron_memory.rs"]
mod neuron_memory;

#[path = "sparse_memory.rs"]
mod sparse_memory;

#[path = "adaptive.rs"]
mod adaptive;

#[path = "token_cache.rs"]
mod token_cache;

#[path = "ids_cache.rs"]
mod ids_cache;
mod ids_streaming;
mod packed_bits;
mod atomic_hashtable;

#[path = "neighbor_search.rs"]
mod neighbor_search;

#[path = "gating.rs"]
mod gating;

#[cfg(target_os = "macos")]
#[path = "metal_gating.rs"]
mod metal_gating;

#[path = "eval_worker.rs"]
pub mod eval_worker;

#[path = "bitwise_ramlm.rs"]
mod bitwise_ramlm;

#[path = "multistage.rs"]
mod multistage;

#[path = "adaptation.rs"]
mod adaptation;

#[cfg(target_os = "macos")]
#[path = "metal_stats.rs"]
mod metal_stats;

#[cfg(target_os = "macos")]
#[path = "metal_train.rs"]
mod metal_train;

#[cfg(not(target_os = "macos"))]
mod metal_train {
    pub struct MetalTrainer;
    impl MetalTrainer {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }
}

#[cfg(target_os = "macos")]
mod metal_atomic_test;

#[cfg(target_os = "macos")]
mod marker_train;

// Drone-controller hot-path (paper #1): attitude sim, controller wrapper,
// Strategy-5 QSR decoder, monotonicity counter, reward.
// See src/wnn/ram/strategies/accelerator/controller.rs and the design
// memory project_drone_controller_paper1.md.
mod controller;
mod controller_training;
mod controller_split;
mod dagger_train;
mod cancel;

// GPU-batched closed-loop controller eval (macOS/Metal only).
#[cfg(target_os = "macos")]
#[path = "metal_controller.rs"]
mod metal_controller;

#[cfg(target_os = "macos")]
pub use metal_evaluator::MetalEvaluator;
#[cfg(target_os = "macos")]
pub use metal_ramlm::MetalRAMLMEvaluator;

/// Set the EMPTY cell value for probability calculation
/// Check if Metal GPU is available
#[pyfunction]
fn metal_available() -> bool {
    metal_evaluator::MetalEvaluator::is_available()
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
/// Find threshold maximizing weighted fitness (F1, FPR, Acc, CE weights).
/// Returns (threshold, f1, fpr, acc, fitness).
#[pyfunction]
fn find_optimal_threshold_fitness_py(
    scores: Vec<f64>,
    labels: Vec<i64>,
    w_ce: f32,
    w_f1: f32,
    w_fpr: f32,
    w_acc: f32,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    Ok(adaptive::find_optimal_threshold_fitness(&scores, &labels, w_ce, w_f1, w_fpr, w_acc))
}

/// Fit Platt scaling. Returns (threshold, a, b).
#[pyfunction]
fn fit_platt_scaling_py(scores: Vec<f64>, labels: Vec<i64>) -> PyResult<(f64, f64, f64)> {
    Ok(adaptive::fit_platt_scaling(&scores, &labels))
}

/// Fit Beta calibration. Returns (threshold, a, b, c).
#[pyfunction]
fn fit_beta_calibration_py(scores: Vec<f64>, labels: Vec<i64>) -> PyResult<(f64, f64, f64, f64)> {
    Ok(adaptive::fit_beta_calibration(&scores, &labels))
}

/// Fit empirical threshold. Returns (threshold, n_bins).
#[pyfunction]
fn fit_empirical_threshold_py(scores: Vec<f64>, labels: Vec<i64>) -> PyResult<(f64, usize)> {
    Ok(adaptive::fit_empirical_threshold(&scores, &labels))
}

/// Compute (CE, accuracy, F1-macro, FPR) for a single-cluster binary classifier
/// from raw scores at a given threshold. CE is binary cross-entropy
/// (threshold-independent); accuracy/F1/FPR depend on `threshold`.
/// `normal_class` is 0 by default, set to 1 when flip_labels is active.
#[pyfunction]
#[pyo3(signature = (scores, labels, threshold, normal_class=0))]
fn compute_binary_metrics_at_threshold_py(
    scores: Vec<f64>,
    labels: Vec<i64>,
    threshold: f64,
    normal_class: usize,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(adaptive::compute_binary_metrics_at_threshold(
        &scores, &labels, threshold, normal_class,
    ))
}

/// Sweep scores for the F1-macro-optimal threshold (binary classification).
/// Returns (threshold, f1_macro, fpr).
#[pyfunction]
fn find_optimal_threshold_f1_py(scores: Vec<f64>, labels: Vec<i64>) -> PyResult<(f64, f64, f64)> {
    Ok(adaptive::find_optimal_threshold_f1(&scores, &labels))
}

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
    empty_value: f32,
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
            empty_value,
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
    empty_value: f32,
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

        let (packed_input, wpe) = crate::neuron_memory::pack_bools_to_u64(
            &input_bools, num_examples, total_input_bits
        );

        evaluator
            .forward_batch(
                &packed_input,
                &conn_vec,
                &mem_vec,
                num_examples,
                wpe,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
                crate::neuron_memory::MODE_TERNARY,
                empty_value,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
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
    empty_value: f32,
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

        let (packed_input, wpe) = crate::neuron_memory::pack_bools_to_u64(
            &input_bools, num_examples, total_input_bits
        );

        evaluator
            .forward_batch(
                &packed_input,
                &conn_vec,
                &mem_vec,
                num_examples,
                wpe,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
                crate::neuron_memory::MODE_TERNARY,
                empty_value,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
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
    empty_value: f32,
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
                    empty_value,
                ));
            } else {
                let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
                let (packed_input, wpe) = crate::neuron_memory::pack_bools_to_u64(
                    &input_bools, num_examples, total_input_bits
                );
                return evaluator
                    .forward_batch(
                        &packed_input,
                        &conn_vec,
                        &mem_vec,
                        num_examples,
                        wpe,
                        num_neurons,
                        bits_per_neuron,
                        neurons_per_cluster,
                        num_clusters,
                        words_per_neuron,
                        crate::neuron_memory::MODE_TERNARY,
                        empty_value,
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
            let (packed_gpu_input, gpu_wpe) = crate::neuron_memory::pack_bools_to_u64(
                &gpu_input, gpu_examples, total_input_bits
            );
            evaluator.forward_batch(
                &packed_gpu_input,
                &conn_vec_gpu,
                &mem_vec_gpu,
                gpu_examples,
                gpu_wpe,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
                crate::neuron_memory::MODE_TERNARY,
                empty_value,
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
            empty_value,
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
fn sparse_bitwise_train_batch<'py>(
    py: Python<'py>,
    memory: &SparseMemory,
    input_bits_flat: PyReadonlyArray1<'py, u8>,
    target_bits_flat: PyReadonlyArray1<'py, u8>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    allow_override: bool,
) -> PyResult<usize> {
    let target_bits_flat: Vec<u8> = target_bits_flat.as_slice()
        .expect("target_bits_flat must be contiguous").to_vec();
    let connections_flat: Vec<i64> = connections_flat.as_slice()
        .expect("connections_flat must be contiguous").to_vec();
    let input_bools: Vec<bool> = input_bits_flat.as_slice()
        .expect("input_bits_flat must be contiguous")
        .iter().map(|&b| b != 0).collect();
    py.allow_threads(|| {
        let modified = sparse_memory::bitwise_train_batch_sparse(
            &memory.inner,
            &input_bools,
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
    empty_value: f32,
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
            empty_value,
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
    empty_value: f32,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        let probs = sparse_memory::forward_batch_tiered(
            &memory.inner,
            &input_bits_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            empty_value,
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
    empty_value: f32,
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
            empty_value,
        )
    });

    // Convert to numpy array
    Ok(numpy::PyArray1::from_vec(py, probs).into())
}

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
    empty_value: f32,
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
        let (packed_input, wpe) = crate::neuron_memory::pack_bools_to_u64(
            &input_bools, num_examples, total_input_bits
        );
        cache.evaluator.forward_batch_general(
            &packed_input,
            &conn_vec,
            &cache.keys,
            &cache.values,
            &cache.offsets,
            &cache.counts,
            &cache.cluster_infos,
            num_examples,
            wpe,
            cache.num_clusters,
            crate::neuron_memory::MODE_TERNARY,
            empty_value,
        ).map_err(|e| format!("Metal forward failed: {}", e))
    }).map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(numpy::PyArray1::from_vec(py, probs).into())
}

/// Option B B2 — parity test for the Metal marker-FSM training kernel.
///
/// Builds a small synthetic training scenario, trains via the CPU
/// MarkerHashTable path AND via the Metal GPU kernel, and asserts the
/// resulting (key, value) snapshots are identical.
#[cfg(target_os = "macos")]
#[pyfunction]
#[pyo3(signature = (num_neurons=128, num_examples=200, bits_per_neuron=16, total_input_bits=96, seed=42))]
fn run_marker_train_parity_test(
    num_neurons: usize,
    num_examples: usize,
    bits_per_neuron: usize,
    total_input_bits: usize,
    seed: u64,
) -> PyResult<Vec<(String, bool, String, f64, f64)>> {
    use atomic_hashtable::MarkerHashTable;
    use marker_train::{MarkerTrainer, NeuronTrainMeta, TrainParams};
    use metal::MTLResourceOptions;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);

    let words_per_example = (total_input_bits + 63) / 64;

    // ---- Synthetic data ----
    // packed_input: each example's bits packed into u64 words
    let mut packed_input: Vec<u64> = vec![0; num_examples * words_per_example];
    for i in 0..(num_examples * words_per_example) {
        packed_input[i] = rng.gen();
        // Mask to total_input_bits
        if (i + 1) % words_per_example == 0 {
            let extra_bits = words_per_example * 64 - total_input_bits;
            if extra_bits > 0 {
                let mask = (1u64 << (64 - extra_bits)) - 1;
                packed_input[i] &= mask;
            }
        }
    }

    // connections: bits_per_neuron per neuron, indices into total_input_bits
    let mut connections: Vec<i32> = Vec::with_capacity(num_neurons * bits_per_neuron);
    for _ in 0..(num_neurons * bits_per_neuron) {
        connections.push(rng.gen_range(0..total_input_bits) as i32);
    }

    // train_targets: 80/20 binary labels
    let train_targets: Vec<i64> = (0..num_examples).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();

    // No negatives for single-cluster binary IDS
    let train_negatives: Vec<i64> = vec![0; 1];

    // class_weights — all 1s for parity test
    let class_weights: Vec<u32> = vec![1, 1];

    // Slot capacity per neuron — pre-sized for the test's worst case
    // (each example is a unique address; we want ≤ 75% load to keep probing
    // efficient). Next power of two above num_examples × 4/3, min 256.
    let target = (num_examples * 4 / 3).max(256);
    let slot_capacity: usize = target.next_power_of_two();
    let total_slots = num_neurons * slot_capacity;

    // Neuron metadata (slot offsets contiguous)
    let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_neurons);
    for n in 0..num_neurons {
        neuron_meta.push(NeuronTrainMeta {
            bits: bits_per_neuron as u32,
            conn_offset: (n * bits_per_neuron) as u32,
            slot_offset: (n * slot_capacity) as u32,
            slot_capacity: slot_capacity as u32,
            cluster_idx: 0,
            _pad: 0,
        });
    }

    // ---- Allocate Metal buffers for GPU side ----
    let trainer = MarkerTrainer::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let device = trainer.device();

    let packed_bytes = (packed_input.len() * 8) as u64;
    let packed_buf = device.new_buffer_with_data(
        packed_input.as_ptr() as *const _,
        packed_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let conn_bytes = (connections.len() * 4) as u64;
    let conn_buf = device.new_buffer_with_data(
        connections.as_ptr() as *const _,
        conn_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let targets_bytes = (train_targets.len() * 8) as u64;
    let targets_buf = device.new_buffer_with_data(
        train_targets.as_ptr() as *const _,
        targets_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let negs_bytes = (train_negatives.len() * 8) as u64;
    let negs_buf = device.new_buffer_with_data(
        train_negatives.as_ptr() as *const _,
        negs_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    // GPU side: one Metal-backed MarkerHashTable with `total_slots` capacity
    // (single flat buffer; per-neuron slot regions inside via offsets)
    let gpu_table = MarkerHashTable::new_metal(device, total_slots, 1);
    let (markers_buf, keys_buf, values_buf) = gpu_table.metal_buffers().unwrap();

    let params = TrainParams {
        num_examples: num_examples as u32,
        num_negatives: 0,
        num_neurons: num_neurons as u32,
        num_genomes: 1,  // single-genome dispatch in this test
        words_per_example: words_per_example as u32,
        num_classes: 2,
        memory_mode: 2,  // QUAD_WEIGHTED
        single_cluster: 1,
        normal_class: 0,
        conn_stride: (num_neurons * bits_per_neuron) as u32,
        neuron_sample_rate: 1.0,  // no sampling in this single-genome test
        rng_seed: 0,
        num_example_chunks: 1,  // parity test: no B10
        oi_mode: 0,
        example_offset: 0,
        examples_in_dispatch: 0,
    };

    let t_gpu_start = std::time::Instant::now();
    let gpu_kernel_ms = trainer.train(
        &packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
        &class_weights, params,
        &markers_buf, &keys_buf, &values_buf,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let gpu_total_ms = t_gpu_start.elapsed().as_secs_f64() * 1000.0;
    let _ = gpu_kernel_ms;

    // ---- CPU reference: train an identical workload via MarkerHashTable
    // per neuron (Heap-backed) and merge into a single sorted snapshot ----
    let t_cpu_start = std::time::Instant::now();
    let cpu_tables: Vec<MarkerHashTable> = (0..num_neurons)
        .map(|_| MarkerHashTable::new(slot_capacity, 1))
        .collect();
    for ex_idx in 0..num_examples {
        let target = train_targets[ex_idx];
        let nudge_dir = target == 1;
        let ex_words = &packed_input[ex_idx * words_per_example..(ex_idx + 1) * words_per_example];
        for n in 0..num_neurons {
            let conn_start = n * bits_per_neuron;
            let mut addr: u64 = 0;
            for i in 0..bits_per_neuron {
                let c = connections[conn_start + i];
                if c < 0 { continue; }
                let cu = c as usize;
                let word_idx = cu / 64;
                let bit_idx = cu % 64;
                let bit = (ex_words[word_idx] >> bit_idx) & 1;
                addr |= bit << (bits_per_neuron - 1 - i);
            }
            cpu_tables[n].nudge(addr, nudge_dir);
        }
    }
    let cpu_ms = t_cpu_start.elapsed().as_secs_f64() * 1000.0;

    // ---- Build per-neuron sorted snapshots from both paths ----
    // GPU: walk the flat buffers' per-neuron sub-regions
    let markers_slice = unsafe {
        std::slice::from_raw_parts(markers_buf.contents() as *const u32, total_slots)
    };
    let keys_slice = unsafe {
        std::slice::from_raw_parts(keys_buf.contents() as *const u64, total_slots)
    };
    let values_slice = unsafe {
        std::slice::from_raw_parts(values_buf.contents() as *const u32, total_slots)
    };

    let mut results: Vec<(String, bool, String, f64, f64)> = Vec::new();
    let mut mismatches = 0usize;
    let mut total_keys_gpu = 0usize;
    let mut total_keys_cpu = 0usize;
    for n in 0..num_neurons {
        let off = n * slot_capacity;
        let mut gpu_entries: Vec<(u64, u8)> = (0..slot_capacity)
            .filter_map(|i| {
                if markers_slice[off + i] == 0xFFFFFFFF {
                    Some((keys_slice[off + i], (values_slice[off + i] & 0xFF) as u8))
                } else { None }
            })
            .collect();
        gpu_entries.sort_by_key(|(k, _)| *k);
        let cpu_entries: Vec<(u64, u8)> = cpu_tables[n].snapshot_sorted();
        total_keys_gpu += gpu_entries.len();
        total_keys_cpu += cpu_entries.len();
        if gpu_entries != cpu_entries {
            mismatches += 1;
        }
    }

    let ok = mismatches == 0 && total_keys_gpu == total_keys_cpu;
    let detail = format!(
        "neurons={}, mismatches={}, gpu_keys={}, cpu_keys={}",
        num_neurons, mismatches, total_keys_gpu, total_keys_cpu,
    );
    results.push((
        "marker_train_gpu_cpu_parity".into(),
        ok,
        detail,
        gpu_total_ms,
        cpu_ms,
    ));

    // B3a: Test export_per_neuron against direct per-neuron snapshot
    // (also confirms the export is consumable by the eval path).
    {
        let slot_offsets: Vec<u32> = (0..num_neurons as u32)
            .map(|n| n * slot_capacity as u32).collect();
        let slot_capacities: Vec<u32> = vec![slot_capacity as u32; num_neurons];
        let (keys, values, offsets, counts) =
            gpu_table.export_per_neuron(&slot_offsets, &slot_capacities);
        // Verify offsets+counts are consistent
        let mut consistent = true;
        for n in 0..num_neurons {
            let start = offsets[n] as usize;
            let cnt = counts[n] as usize;
            if start + cnt > keys.len() {
                consistent = false;
                break;
            }
            // Per-neuron must be sorted by key
            for i in 1..cnt {
                if keys[start + i - 1] > keys[start + i] {
                    consistent = false;
                    break;
                }
            }
            // Compare against direct per-neuron walk
            let mut direct: Vec<(u64, u8)> = (0..slot_capacity)
                .filter_map(|i| {
                    let slot = n * slot_capacity + i;
                    if markers_slice[slot] == 0xFFFFFFFF {
                        Some((keys_slice[slot], (values_slice[slot] & 0xFF) as u8))
                    } else { None }
                })
                .collect();
            direct.sort_by_key(|(k, _)| *k);
            let from_export: Vec<(u64, u8)> = (0..cnt)
                .map(|i| (keys[start + i], values[start + i])).collect();
            if direct != from_export {
                consistent = false;
                break;
            }
        }
        let detail = format!(
            "exported {} keys across {} neurons; per-neuron sorted={}",
            keys.len(), num_neurons, consistent
        );
        results.push(("export_per_neuron".into(), consistent, detail, 0.0, 0.0));
    }

    Ok(results)
}

/// Option B B4-batched — multi-genome Metal kernel parity test.
///
/// Builds `num_genomes` synthetic genomes (different random connections,
/// same shape), trains them all via a SINGLE batched Metal dispatch AND
/// individually via the existing CPU MarkerHashTable path. Validates
/// that the per-genome output is identical and reports GPU vs CPU
/// wall-time.
#[cfg(target_os = "macos")]
#[pyfunction]
#[pyo3(signature = (num_genomes=16, num_neurons=100, num_examples=5000, bits_per_neuron=48, total_input_bits=96, seed=42, neuron_sample_rate=1.0, rng_seed=0))]
fn run_marker_train_batched_parity_test(
    num_genomes: usize,
    num_neurons: usize,
    num_examples: usize,
    bits_per_neuron: usize,
    total_input_bits: usize,
    seed: u64,
    neuron_sample_rate: f32,
    rng_seed: u32,
) -> PyResult<Vec<(String, bool, String, f64, f64)>> {
    use atomic_hashtable::MarkerHashTable;
    use marker_train::{MarkerTrainer, NeuronTrainMeta, TrainParams};
    use metal::MTLResourceOptions;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);
    let words_per_example = (total_input_bits + 63) / 64;
    let conn_per_genome = num_neurons * bits_per_neuron;

    // Shared input + targets
    let mut packed_input: Vec<u64> = vec![0; num_examples * words_per_example];
    for i in 0..(num_examples * words_per_example) { packed_input[i] = rng.gen(); }
    let train_targets: Vec<i64> = (0..num_examples).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();

    // Per-genome connections (different random seed pattern)
    let total_conns = num_genomes * conn_per_genome;
    let mut connections: Vec<i32> = Vec::with_capacity(total_conns);
    for _ in 0..total_conns {
        connections.push(rng.gen_range(0..total_input_bits) as i32);
    }

    // Slot capacity per neuron — pre-sized for worst case
    let target = (num_examples * 4 / 3).max(256);
    let slot_capacity_per_neuron: usize = target.next_power_of_two();
    let slots_per_genome = num_neurons * slot_capacity_per_neuron;
    let total_slots = num_genomes * slots_per_genome;

    // Per-(genome, neuron) metadata — slot_offset is global into the flat
    // buffer; conn_offset is genome-relative (kernel adds genome's base).
    let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_genomes * num_neurons);
    for g in 0..num_genomes {
        for n in 0..num_neurons {
            neuron_meta.push(NeuronTrainMeta {
                bits: bits_per_neuron as u32,
                conn_offset: (n * bits_per_neuron) as u32,
                slot_offset: ((g * num_neurons + n) * slot_capacity_per_neuron) as u32,
                slot_capacity: slot_capacity_per_neuron as u32,
                cluster_idx: 0,
                _pad: 0,
            });
        }
    }

    let class_weights: Vec<u32> = vec![1, 1];
    let train_negatives: Vec<i64> = vec![0; 1];

    let trainer = MarkerTrainer::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let device = trainer.device();

    let packed_buf = device.new_buffer_with_data(
        packed_input.as_ptr() as *const _,
        (packed_input.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let conn_buf = device.new_buffer_with_data(
        connections.as_ptr() as *const _,
        (connections.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let targets_buf = device.new_buffer_with_data(
        train_targets.as_ptr() as *const _,
        (train_targets.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let negs_buf = device.new_buffer_with_data(
        train_negatives.as_ptr() as *const _,
        (train_negatives.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let gpu_table = MarkerHashTable::new_metal(device, total_slots, 1);
    let (markers_buf, keys_buf, values_buf) = gpu_table.metal_buffers().unwrap();

    let params = TrainParams {
        num_examples: num_examples as u32,
        num_negatives: 0,
        num_neurons: num_neurons as u32,
        num_genomes: num_genomes as u32,
        words_per_example: words_per_example as u32,
        num_classes: 2,
        memory_mode: 2,
        single_cluster: 1,
        normal_class: 0,
        conn_stride: conn_per_genome as u32,
        neuron_sample_rate,
        rng_seed,
        num_example_chunks: 1,  // parity test: no B10
        oi_mode: 0,
        example_offset: 0,
        examples_in_dispatch: 0,
    };

    let t_gpu_start = std::time::Instant::now();
    let gpu_kernel_ms = trainer.train(
        &packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
        &class_weights, params,
        &markers_buf, &keys_buf, &values_buf,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let gpu_total_ms = t_gpu_start.elapsed().as_secs_f64() * 1000.0;
    let _ = gpu_kernel_ms;

    // Sampling skip — same xorshift as kernel + production CPU path.
    let should_skip = |neuron_idx: usize, ex_idx: usize| -> bool {
        if neuron_sample_rate >= 1.0 { return false; }
        let mut rng = rng_seed
            .wrapping_add((neuron_idx as u32).wrapping_mul(1000003))
            .wrapping_add((ex_idx as u32).wrapping_mul(2654435761));
        if rng == 0 { rng = 1; }
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        ((rng >> 8) as f32) / 16777216.0 >= neuron_sample_rate
    };

    // CPU reference: for each genome, build per-neuron MarkerHashTables,
    // train sequentially (with matching sampling decisions).
    let t_cpu_start = std::time::Instant::now();
    let mut cpu_per_genome: Vec<Vec<Vec<(u64, u8)>>> = Vec::with_capacity(num_genomes);
    for g in 0..num_genomes {
        let tables: Vec<MarkerHashTable> = (0..num_neurons)
            .map(|_| MarkerHashTable::new(slot_capacity_per_neuron, 1))
            .collect();
        let conn_base = g * conn_per_genome;
        for ex_idx in 0..num_examples {
            let target = train_targets[ex_idx];
            let nudge_dir = target == 1;
            let ex_words = &packed_input[ex_idx * words_per_example..(ex_idx + 1) * words_per_example];
            for n in 0..num_neurons {
                if should_skip(n, ex_idx) { continue; }
                let conn_start = conn_base + n * bits_per_neuron;
                let mut addr: u64 = 0;
                for i in 0..bits_per_neuron {
                    let c = connections[conn_start + i];
                    if c < 0 { continue; }
                    let cu = c as usize;
                    let bit = (ex_words[cu / 64] >> (cu % 64)) & 1;
                    addr |= bit << (bits_per_neuron - 1 - i);
                }
                tables[n].nudge(addr, nudge_dir);
            }
        }
        cpu_per_genome.push(tables.iter().map(|t| t.snapshot_sorted()).collect());
    }
    let cpu_ms = t_cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU per-genome snapshots from the flat buffer
    let markers_slice = unsafe {
        std::slice::from_raw_parts(markers_buf.contents() as *const u32, total_slots)
    };
    let keys_slice = unsafe {
        std::slice::from_raw_parts(keys_buf.contents() as *const u64, total_slots)
    };
    let values_slice = unsafe {
        std::slice::from_raw_parts(values_buf.contents() as *const u32, total_slots)
    };

    let mut results: Vec<(String, bool, String, f64, f64)> = Vec::new();
    let mut total_mismatches = 0usize;
    let mut total_keys_gpu = 0usize;
    let mut total_keys_cpu = 0usize;
    for g in 0..num_genomes {
        for n in 0..num_neurons {
            let off = (g * num_neurons + n) * slot_capacity_per_neuron;
            let mut gpu_entries: Vec<(u64, u8)> = (0..slot_capacity_per_neuron)
                .filter_map(|i| {
                    if markers_slice[off + i] == 0xFFFFFFFF {
                        Some((keys_slice[off + i], (values_slice[off + i] & 0xFF) as u8))
                    } else { None }
                })
                .collect();
            gpu_entries.sort_by_key(|(k, _)| *k);
            let cpu_entries = &cpu_per_genome[g][n];
            total_keys_gpu += gpu_entries.len();
            total_keys_cpu += cpu_entries.len();
            if gpu_entries != *cpu_entries { total_mismatches += 1; }
        }
    }

    let ok = total_mismatches == 0 && total_keys_gpu == total_keys_cpu;
    let speedup = cpu_ms / gpu_total_ms;
    let detail = format!(
        "{} genomes × {} neurons; mismatches={}; gpu_keys={}, cpu_keys={}; speedup={:.2}x",
        num_genomes, num_neurons, total_mismatches, total_keys_gpu, total_keys_cpu, speedup,
    );
    results.push((
        "marker_train_batched_parity".into(),
        ok,
        detail,
        gpu_total_ms,
        cpu_ms,
    ));
    Ok(results)
}

/// Option B B5 — multi-cluster Metal kernel parity test.
///
/// Builds `num_genomes` synthetic genomes, each with K clusters of `neurons_per_cluster`
/// neurons. Targets are random class labels in [0, K). For each example, K-1
/// `train_negatives` are sampled (all other clusters). Trains via the batched
/// Metal kernel AND a CPU MarkerHashTable reference; validates exact per-neuron
/// snapshot match.
#[cfg(target_os = "macos")]
#[pyfunction]
#[pyo3(signature = (num_genomes=4, num_clusters=8, neurons_per_cluster=12, num_examples=2000, bits_per_neuron=24, total_input_bits=512, seed=42, neuron_sample_rate=1.0, rng_seed=0))]
fn run_marker_train_multicluster_parity_test(
    num_genomes: usize,
    num_clusters: usize,
    neurons_per_cluster: usize,
    num_examples: usize,
    bits_per_neuron: usize,
    total_input_bits: usize,
    seed: u64,
    neuron_sample_rate: f32,
    rng_seed: u32,
) -> PyResult<Vec<(String, bool, String, f64, f64)>> {
    use atomic_hashtable::MarkerHashTable;
    use marker_train::{MarkerTrainer, NeuronTrainMeta, TrainParams};
    use metal::MTLResourceOptions;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);
    let num_neurons_per_genome = num_clusters * neurons_per_cluster;
    let words_per_example = (total_input_bits + 63) / 64;
    let conn_per_genome = num_neurons_per_genome * bits_per_neuron;
    let num_negatives = num_clusters.saturating_sub(1).min(8);  // up to 8 negatives

    // Shared inputs
    let mut packed_input: Vec<u64> = vec![0; num_examples * words_per_example];
    for i in 0..(num_examples * words_per_example) { packed_input[i] = rng.gen(); }
    let train_targets: Vec<i64> = (0..num_examples)
        .map(|_| rng.gen_range(0..num_clusters) as i64)
        .collect();
    // For each example, generate `num_negatives` distinct negative cluster ids
    // (all different from target). Layout: flat [num_examples * num_negatives].
    let mut train_negatives: Vec<i64> = Vec::with_capacity(num_examples * num_negatives);
    for ex in 0..num_examples {
        let target = train_targets[ex] as usize;
        let mut neg_pool: Vec<usize> = (0..num_clusters).filter(|&c| c != target).collect();
        // shuffle and take first num_negatives
        for i in (1..neg_pool.len()).rev() {
            let j = rng.gen_range(0..=i);
            neg_pool.swap(i, j);
        }
        for k in 0..num_negatives {
            train_negatives.push(neg_pool[k % neg_pool.len()] as i64);
        }
    }

    let total_conns = num_genomes * conn_per_genome;
    let mut connections: Vec<i32> = Vec::with_capacity(total_conns);
    for _ in 0..total_conns {
        connections.push(rng.gen_range(0..total_input_bits) as i32);
    }

    let target_cap = (num_examples * 4 / 3).max(256);
    let slot_capacity_per_neuron: usize = target_cap.next_power_of_two();
    let slots_per_genome = num_neurons_per_genome * slot_capacity_per_neuron;
    let total_slots = num_genomes * slots_per_genome;

    // Per-(genome, neuron) metadata. Neurons within a genome are grouped by
    // cluster: cluster c's neurons are at [c * neurons_per_cluster, (c+1) * neurons_per_cluster).
    let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_genomes * num_neurons_per_genome);
    for g in 0..num_genomes {
        for c in 0..num_clusters {
            for n_local in 0..neurons_per_cluster {
                let n_global = c * neurons_per_cluster + n_local;
                neuron_meta.push(NeuronTrainMeta {
                    bits: bits_per_neuron as u32,
                    conn_offset: (n_global * bits_per_neuron) as u32,
                    slot_offset: ((g * num_neurons_per_genome + n_global) * slot_capacity_per_neuron) as u32,
                    slot_capacity: slot_capacity_per_neuron as u32,
                    cluster_idx: c as u32,
                    _pad: 0,
                });
            }
        }
    }

    let class_weights: Vec<u32> = vec![1; num_clusters];

    let trainer = MarkerTrainer::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let device = trainer.device();

    let packed_buf = device.new_buffer_with_data(
        packed_input.as_ptr() as *const _,
        (packed_input.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let conn_buf = device.new_buffer_with_data(
        connections.as_ptr() as *const _,
        (connections.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let targets_buf = device.new_buffer_with_data(
        train_targets.as_ptr() as *const _,
        (train_targets.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let negs_buf = device.new_buffer_with_data(
        train_negatives.as_ptr() as *const _,
        (train_negatives.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let gpu_table = MarkerHashTable::new_metal(device, total_slots, 1);
    let (markers_buf, keys_buf, values_buf) = gpu_table.metal_buffers().unwrap();

    let params = TrainParams {
        num_examples: num_examples as u32,
        num_negatives: num_negatives as u32,
        num_neurons: num_neurons_per_genome as u32,
        num_genomes: num_genomes as u32,
        words_per_example: words_per_example as u32,
        num_classes: num_clusters as u32,
        memory_mode: 2,
        single_cluster: 0,   // <-- multi-cluster path
        normal_class: 0,
        conn_stride: conn_per_genome as u32,
        neuron_sample_rate,
        rng_seed,
        num_example_chunks: 1,  // multi-cluster parity test: no B10
        oi_mode: 0,
        example_offset: 0,
        examples_in_dispatch: 0,
    };

    let t_gpu_start = std::time::Instant::now();
    let _ = trainer.train(
        &packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
        &class_weights, params,
        &markers_buf, &keys_buf, &values_buf,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let gpu_total_ms = t_gpu_start.elapsed().as_secs_f64() * 1000.0;

    // Sampling skip — must match the Metal kernel's should_skip_sample()
    // and the production CPU path at adaptive.rs:2867-2880.
    let should_skip = |neuron_idx: usize, ex_idx: usize| -> bool {
        if neuron_sample_rate >= 1.0 { return false; }
        let mut rng = rng_seed
            .wrapping_add((neuron_idx as u32).wrapping_mul(1000003))
            .wrapping_add((ex_idx as u32).wrapping_mul(2654435761));
        if rng == 0 { rng = 1; }
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        ((rng >> 8) as f32) / 16777216.0 >= neuron_sample_rate
    };

    // CPU reference — mirror the GPU semantics exactly (incl. sampling).
    let t_cpu_start = std::time::Instant::now();
    let mut cpu_per_genome: Vec<Vec<Vec<(u64, u8)>>> = Vec::with_capacity(num_genomes);
    for g in 0..num_genomes {
        let tables: Vec<MarkerHashTable> = (0..num_neurons_per_genome)
            .map(|_| MarkerHashTable::new(slot_capacity_per_neuron, 1))
            .collect();
        let conn_base = g * conn_per_genome;
        for ex_idx in 0..num_examples {
            let target = train_targets[ex_idx] as usize;
            let ex_words = &packed_input[ex_idx * words_per_example..(ex_idx + 1) * words_per_example];

            // Positive: nudge TRUE for neurons in cluster `target`
            for n_local in 0..neurons_per_cluster {
                let n_global = target * neurons_per_cluster + n_local;
                if should_skip(n_global, ex_idx) { continue; }
                let conn_start = conn_base + n_global * bits_per_neuron;
                let mut addr: u64 = 0;
                for i in 0..bits_per_neuron {
                    let c = connections[conn_start + i];
                    if c < 0 { continue; }
                    let cu = c as usize;
                    let bit = (ex_words[cu / 64] >> (cu % 64)) & 1;
                    addr |= bit << (bits_per_neuron - 1 - i);
                }
                tables[n_global].nudge(addr, true);
            }

            // Negatives: nudge FALSE for neurons in each negative cluster
            // (mirror kernel's skip-if-equal-target rule)
            let neg_base = ex_idx * num_negatives;
            for k in 0..num_negatives {
                let false_cluster = train_negatives[neg_base + k] as usize;
                if false_cluster == target { continue; }
                for n_local in 0..neurons_per_cluster {
                    let n_global = false_cluster * neurons_per_cluster + n_local;
                    if should_skip(n_global, ex_idx) { continue; }
                    let conn_start = conn_base + n_global * bits_per_neuron;
                    let mut addr: u64 = 0;
                    for i in 0..bits_per_neuron {
                        let c = connections[conn_start + i];
                        if c < 0 { continue; }
                        let cu = c as usize;
                        let bit = (ex_words[cu / 64] >> (cu % 64)) & 1;
                        addr |= bit << (bits_per_neuron - 1 - i);
                    }
                    tables[n_global].nudge(addr, false);
                }
            }
        }
        cpu_per_genome.push(tables.iter().map(|t| t.snapshot_sorted()).collect());
    }
    let cpu_ms = t_cpu_start.elapsed().as_secs_f64() * 1000.0;

    // Compare per-neuron snapshots
    let markers_slice = unsafe { std::slice::from_raw_parts(markers_buf.contents() as *const u32, total_slots) };
    let keys_slice = unsafe { std::slice::from_raw_parts(keys_buf.contents() as *const u64, total_slots) };
    let values_slice = unsafe { std::slice::from_raw_parts(values_buf.contents() as *const u32, total_slots) };

    let mut results: Vec<(String, bool, String, f64, f64)> = Vec::new();
    let mut total_mismatches = 0usize;
    let mut total_keys_gpu = 0usize;
    let mut total_keys_cpu = 0usize;
    let mut first_mismatch: Option<String> = None;
    for g in 0..num_genomes {
        for n_global in 0..num_neurons_per_genome {
            let off = (g * num_neurons_per_genome + n_global) * slot_capacity_per_neuron;
            let mut gpu_entries: Vec<(u64, u8)> = (0..slot_capacity_per_neuron)
                .filter_map(|i| {
                    if markers_slice[off + i] == 0xFFFFFFFF {
                        Some((keys_slice[off + i], (values_slice[off + i] & 0xFF) as u8))
                    } else { None }
                })
                .collect();
            gpu_entries.sort_by_key(|(k, _)| *k);
            let cpu_entries = &cpu_per_genome[g][n_global];
            total_keys_gpu += gpu_entries.len();
            total_keys_cpu += cpu_entries.len();
            if gpu_entries != *cpu_entries {
                total_mismatches += 1;
                if first_mismatch.is_none() {
                    let cluster = n_global / neurons_per_cluster;
                    first_mismatch = Some(format!(
                        "genome={} neuron={} cluster={} gpu_n={} cpu_n={}",
                        g, n_global, cluster, gpu_entries.len(), cpu_entries.len()
                    ));
                }
            }
        }
    }

    let ok = total_mismatches == 0 && total_keys_gpu == total_keys_cpu;
    let speedup = cpu_ms / gpu_total_ms.max(0.001);
    let detail = format!(
        "{} genomes × {} clusters × {} neurons/cluster; mismatches={}; gpu_keys={}, cpu_keys={}; first_mismatch={:?}; speedup={:.2}x",
        num_genomes, num_clusters, neurons_per_cluster,
        total_mismatches, total_keys_gpu, total_keys_cpu, first_mismatch, speedup,
    );
    results.push(("marker_train_multicluster_parity".into(), ok, detail, gpu_total_ms, cpu_ms));
    Ok(results)
}

/// Run MarkerHashTable unit tests and return (name, passed, details) tuples.
/// Used to validate B0 from Python since cargo test --lib can't link with the
/// PyO3 extension-module feature.
#[pyfunction]
fn run_marker_hashtable_tests() -> PyResult<Vec<(String, bool, String)>> {
    use atomic_hashtable::{AtomicHashTable, MarkerHashTable};
    use rayon::prelude::*;
    let mut results: Vec<(String, bool, String)> = Vec::new();

    // basic_write_read
    {
        let t = MarkerHashTable::new(64, 2);
        let mut ok = true;
        let mut why = String::new();
        if !t.write(42, 1, false) { ok = false; why.push_str("write returned false; "); }
        if t.read(42) != 1 { ok = false; why.push_str(&format!("read(42)={} expected 1; ", t.read(42))); }
        if t.read(99) != 2 { ok = false; why.push_str(&format!("read(99)={} expected default 2; ", t.read(99))); }
        if t.len() != 1 { ok = false; why.push_str(&format!("len={} expected 1; ", t.len())); }
        results.push(("basic_write_read".into(), ok, why));
    }
    // true_wins_over_false
    {
        let t = MarkerHashTable::new(64, 2);
        let mut ok = true;
        let mut why = String::new();
        if !t.write(7, 1, false) { ok = false; why.push_str("initial write TRUE failed; "); }
        if t.write(7, 0, false) { ok = false; why.push_str("FALSE-over-TRUE was accepted; "); }
        if t.read(7) != 1 { ok = false; why.push_str(&format!("read(7)={} expected 1; ", t.read(7))); }
        results.push(("true_wins_over_false".into(), ok, why));
    }
    // nudge_clamping
    {
        let t = MarkerHashTable::new(64, 1);
        let mut ok = true;
        let mut why = String::new();
        t.nudge(11, true);
        if t.read(11) != 2 { ok = false; why.push_str(&format!("after 1 nudge_true: {} expected 2; ", t.read(11))); }
        t.nudge(11, true);
        if t.read(11) != 3 { ok = false; why.push_str(&format!("after 2 nudge_true: {} expected 3; ", t.read(11))); }
        if t.nudge(11, true) { ok = false; why.push_str("saturated nudge_true reported true; "); }
        if t.read(11) != 3 { ok = false; why.push_str(&format!("saturated value: {} expected 3; ", t.read(11))); }
        t.nudge(11, false);
        if t.read(11) != 2 { ok = false; why.push_str(&format!("after nudge_false: {} expected 2; ", t.read(11))); }
        results.push(("nudge_clamping".into(), ok, why));
    }
    // resize_under_load
    {
        let t = MarkerHashTable::new(16, 2);
        let mut ok = true;
        let mut why = String::new();
        for k in 0u64..1000 {
            if !t.write(k, 1, false) { ok = false; why.push_str(&format!("write {} failed; ", k)); break; }
        }
        if t.len() != 1000 { ok = false; why.push_str(&format!("len={} expected 1000; ", t.len())); }
        for k in 0u64..1000 {
            if t.read(k) != 1 { ok = false; why.push_str(&format!("read {}={} expected 1; ", k, t.read(k))); break; }
        }
        results.push(("resize_under_load".into(), ok, why));
    }
    // parallel_writes_distinct_keys
    {
        let t = MarkerHashTable::new(64, 2);
        (0u64..10_000).into_par_iter().for_each(|k| { t.write(k, 1, false); });
        let mut ok = t.len() == 10_000;
        let mut why = if ok { String::new() } else { format!("len={} expected 10000; ", t.len()) };
        for k in 0u64..10_000 {
            if t.read(k) != 1 { ok = false; why.push_str(&format!("missing {}; ", k)); break; }
        }
        results.push(("parallel_writes_distinct_keys".into(), ok, why));
    }
    // parallel_nudges_same_keys (the contention stress test)
    {
        let t = MarkerHashTable::new(64, 1);
        let work: Vec<u64> = (0u64..10).cycle().take(10_000).collect();
        work.par_iter().for_each(|&k| { t.nudge(k, true); });
        let mut ok = true;
        let mut why = String::new();
        for k in 0u64..10 {
            if t.read(k) != 3 { ok = false; why.push_str(&format!("key {} read={} expected 3; ", k, t.read(k))); }
        }
        results.push(("parallel_nudges_same_keys".into(), ok, why));
    }
    // snapshot_sorted
    {
        let t = MarkerHashTable::new(64, 2);
        for (k, v) in &[(100u64, 1u8), (50, 0), (75, 1), (25, 0)] {
            t.write(*k, *v, true);
        }
        let snap = t.snapshot_sorted();
        let expected = vec![(25u64, 0u8), (50, 0), (75, 1), (100, 1)];
        let ok = snap == expected;
        let why = if ok { String::new() } else { format!("snap={:?} expected {:?}", snap, expected) };
        results.push(("snapshot_sorted".into(), ok, why));
    }
    // parity_with_atomic_random_workload
    {
        let a = AtomicHashTable::new(64, 1);
        let m = MarkerHashTable::new(64, 1);
        let mut state: u64 = 0xC0FFEE;
        for _ in 0..5000 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let key = state & 0xFFFFFFFFFFFF;
            let nudge_true = (state >> 32) & 1 == 1;
            a.nudge(key, nudge_true);
            m.nudge(key, nudge_true);
        }
        let snap_a = a.snapshot_sorted();
        let snap_m = m.snapshot_sorted();
        let ok = snap_a == snap_m;
        let why = if ok {
            format!("both produced {} entries", snap_a.len())
        } else {
            format!("DIVERGED: atomic {} vs marker {} entries", snap_a.len(), snap_m.len())
        };
        results.push(("parity_with_atomic_random_workload".into(), ok, why));
    }

    // B1: Metal-backed MarkerHashTable parity vs Heap-backed
    #[cfg(target_os = "macos")]
    {
        let device = metal::Device::system_default();
        if device.is_none() {
            results.push(("metal_backed_parity".into(), false,
                "no Metal device available; skipping".into()));
        } else {
            let device = device.unwrap();
            // Capacity sized generously so no resize is needed (Metal-backed
            // doesn't support live resize). 5000 distinct keys at 25% load:
            let metal_table = MarkerHashTable::new_metal(&device, 32768, 1);
            let heap_table = MarkerHashTable::new(32768, 1);
            let mut state: u64 = 0xC0FFEE;
            for _ in 0..5000 {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                let key = state & 0xFFFFFFFFFFFF;
                let nudge_true = (state >> 32) & 1 == 1;
                metal_table.nudge(key, nudge_true);
                heap_table.nudge(key, nudge_true);
            }
            let snap_metal = metal_table.snapshot_sorted();
            let snap_heap = heap_table.snapshot_sorted();
            let ok = snap_metal == snap_heap;
            let buffers = metal_table.metal_buffers();
            let why = if ok {
                format!(
                    "metal & heap MarkerHashTable identical ({} entries); metal_buffers={}",
                    snap_metal.len(),
                    if buffers.is_some() { "Some(markers,keys,values)" } else { "None" }
                )
            } else {
                format!("DIVERGED: metal {} vs heap {} entries", snap_metal.len(), snap_heap.len())
            };
            results.push(("metal_backed_parity".into(), ok, why));
        }
    }

    // B1: Metal-backed parallel writes — stresses the CAS-coherent-within-CPU
    // case where the underlying memory is a Metal shared buffer (no GPU
    // involvement yet).
    #[cfg(target_os = "macos")]
    {
        let device = metal::Device::system_default();
        if let Some(device) = device {
            let t = MarkerHashTable::new_metal(&device, 32768, 1);
            (0u64..10_000).into_par_iter().for_each(|k| { t.write(k, 1, false); });
            let mut ok = t.len() == 10_000;
            let mut why = if ok { String::new() } else { format!("len={} expected 10000; ", t.len()) };
            for k in 0u64..10_000 {
                if t.read(k) != 1 { ok = false; why.push_str(&format!("missing {}; ", k)); break; }
            }
            results.push(("metal_backed_parallel_writes".into(), ok, why));
        }
    }

    Ok(results)
}

/// Atomic CAS microbenchmark for Option C foundation validation.
///
/// Returns a list of (test_name, passed, expected, observed, details, elapsed_ms)
/// tuples. The three tests validate:
///   1. GPU-only atomic CAS produces correct count (sanity)
///   2. Concurrent CPU+GPU CAS on a shared buffer is atomic (the key test)
///   3. CAS-claim-from-EMPTY with contention yields exactly one winner per slot
///
/// On non-macOS platforms returns an error (Metal not available).
#[pyfunction]
#[pyo3(signature = (num_slots=256, gpu_threads=4096, cpu_threads=64, iterations=1000))]
fn run_atomic_cas_microbench(
    num_slots: usize,
    gpu_threads: usize,
    cpu_threads: usize,
    iterations: usize,
) -> PyResult<Vec<(String, bool, u64, u64, String, f64)>> {
    #[cfg(target_os = "macos")]
    {
        let results = metal_atomic_test::run_microbench(
            num_slots, gpu_threads, cpu_threads, iterations,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(results.into_iter().map(|r| (
            r.test_name, r.passed, r.expected, r.observed, r.details, r.elapsed_ms,
        )).collect())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (num_slots, gpu_threads, cpu_threads, iterations);
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Atomic CAS microbench requires Metal (macOS).",
        ))
    }
}

// ============================================================================
// Per-Cluster Optimization (Rust-accelerated discriminative optimization)
// ============================================================================

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
#[pyo3(signature = (input_bits, connections_flat, memory_words, group_neurons, group_bits,
    _group_words_per_neuron, group_cluster_ids_flat, group_cluster_counts, group_memory_offsets,
    group_conn_offsets, num_examples, total_input_bits, num_clusters, empty_value=0.0))]
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
    empty_value: f32,
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
            empty_value,
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
#[pyo3(signature = (
    genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
    num_genomes, num_clusters,
    train_input_bits, train_targets, train_negatives, num_train, num_negatives,
    eval_input_bits, eval_targets, num_eval,
    total_input_bits, empty_value,
    neuron_sample_rate=1.0, rng_seed=0
))]
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
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, num_clusters)?;
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

    // Convert to owned data. PyO3 numpy uint8 bytes are treated as logical bools
    // (0 ⇒ false, non-zero ⇒ true) and packed directly into PackedBits — no
    // Vec<bool> intermediate.
    let train_targets_vec: Vec<i64> = train_targets_slice.to_vec();
    let train_negatives_vec: Vec<i64> = train_negatives_slice.to_vec();
    let eval_targets_vec: Vec<i64> = eval_targets_slice.to_vec();

    let train_input_packed = packed_bits::PackedBits::from_bool_bytes(train_input_slice, total_input_bits);
    let eval_input_packed = packed_bits::PackedBits::from_bool_bytes(eval_input_slice, total_input_bits);

    py.allow_threads(|| {
        let fitness = adaptive::evaluate_genomes_parallel(
            &genomes_bits_flat,
            &genomes_neurons_flat,
            &genomes_connections_flat,
            num_genomes,
            num_clusters,
            &train_input_packed,
            &train_targets_vec,
            &train_negatives_vec,
            num_train,
            num_negatives,
            &eval_input_packed,
            &eval_targets_vec,
            num_eval,
            total_input_bits,
            neuron_memory::EvalSettings { empty_value, ..Default::default() },
            neuron_sample_rate,
            rng_seed,
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
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> PyResult<Vec<(f64, f64, f64, f64, f64, u32)>> {
    validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, num_clusters)?;
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

    // Numpy uint8 bytes ⇒ PackedBits directly (no Vec<bool> intermediate).
    let train_targets_vec: Vec<i64> = train_targets_slice.to_vec();
    let train_negatives_vec: Vec<i64> = train_negatives_slice.to_vec();
    let eval_targets_vec: Vec<i64> = eval_targets_slice.to_vec();

    let train_input_packed = packed_bits::PackedBits::from_bool_bytes(train_input_slice, total_input_bits);
    let eval_input_packed = packed_bits::PackedBits::from_bool_bytes(eval_input_slice, total_input_bits);

    py.allow_threads(|| {
        let fitness = adaptive::evaluate_genomes_parallel_hybrid(
            &genomes_bits_flat,
            &genomes_neurons_flat,
            &genomes_connections_flat,
            num_genomes,
            num_clusters,
            &train_input_packed,
            &train_targets_vec,
            &train_negatives_vec,
            num_train,
            num_negatives,
            &eval_input_packed,
            &eval_targets_vec,
            num_eval,
            total_input_bits,
            neuron_memory::EvalSettings { empty_value, ..Default::default() },
            neuron_sample_rate,
            rng_seed,
            None, // class_weights: direct PyO3 call doesn't use class balancing
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
    experiment_id: Option<i64>,
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
            experiment_id: None,
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
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.vocab_size())?;
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
                neuron_sample_rate,
                rng_seed,
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
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.vocab_size())?;
        py.allow_threads(|| {
            Ok(token_cache::evaluate_genomes_cached_full(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                empty_value,
                neuron_sample_rate,
                rng_seed,
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
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.vocab_size())?;
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
                neuron_sample_rate,
                rng_seed,
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
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.vocab_size())?;
        py.allow_threads(|| {
            Ok(token_cache::evaluate_genomes_cached_full_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                empty_value,
                neuron_sample_rate,
                rng_seed,
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

    /// Set experiment context for live progress reporting.
    fn set_experiment_context(&mut self, experiment_id: i64) {
        self.experiment_id = Some(experiment_id);
    }

    /// Get current live progress from active search (if any).
    ///
    /// Returns None if no search is in progress, otherwise returns a dict
    /// with progress fields. Called by the Python observer thread.
    fn get_live_progress(&self, py: Python<'_>) -> PyResult<Option<pyo3::PyObject>> {
        let guard = self.inner.live_progress.read()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e)))?;
        match &*guard {
            None => Ok(None),
            Some(lp) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("experiment_id", lp.experiment_id)?;
                dict.set_item("generation", lp.generation)?;
                dict.set_item("total_generations", lp.total_generations)?;
                dict.set_item("phase", &lp.phase)?;
                dict.set_item("evaluated", lp.evaluated)?;
                dict.set_item("target_count", lp.target_count)?;
                match lp.viable {
                    Some(v) => dict.set_item("viable", v)?,
                    None => dict.set_item("viable", py.None())?,
                }
                dict.set_item("best_ce", lp.best_ce)?;
                dict.set_item("best_acc", lp.best_acc)?;
                dict.set_item("elapsed_secs", lp.elapsed_secs)?;
                dict.set_item("updated_at", lp.updated_at)?;
                Ok(Some(dict.into()))
            }
        }
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
        mutable_clusters = None,
        phase_type = 0
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
        phase_type: u8,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>> {
        let num_clusters = base_neurons.len();
        let total_input_bits = self.inner.total_input_bits();
        let phase = match phase_type { 1 => neighbor_search::PhaseType::Bits, 2 => neighbor_search::PhaseType::Connections, 3 => neighbor_search::PhaseType::Cluster, _ => neighbor_search::PhaseType::Neurons };

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
            phase_type: phase,
        };

        // Set up live progress for observer thread
        let lp_arc = self.inner.live_progress.clone();
        let exp_id = self.experiment_id.unwrap_or(0);
        if let Ok(mut guard) = lp_arc.write() {
            *guard = Some(neighbor_search::LiveProgress {
                experiment_id: exp_id,
                generation: generation.map(|g| g as i32 + 1).unwrap_or(1),
                total_generations: total_generations.map(|g| g as i32).unwrap_or(100),
                phase: "ts_neighbors".into(),
                evaluated: 0, target_count, viable: Some(0),
                best_ce: f64::MAX, best_acc: 0.0, elapsed_secs: 0.0,
                updated_at: neighbor_search::LiveProgress::now_unix(),
            });
        }

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            // Closure captures the token cache and eval params
            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                crate::token_cache::evaluate_genomes_cached_hybrid(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, eval_subset_idx, empty_value,
                    1.0, 0, // no neuron sampling for neighbor search
                )
            };

            let lp_ref = Some(&lp_arc);

            let candidates = if return_best_n {
                neighbor_search::search_neighbors_best_n(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                )
            } else {
                let (passed, _) = neighbor_search::search_neighbors_with_threshold(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                );
                passed
            };

            Ok(candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect())
        });

        // Clear live progress after search completes
        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
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
        mutable_clusters = None,
        phase_type = 0,
        cluster_crossover_ratio = 0.0,
        pool_shuffle_ratio = 0.0,
        assortative_mating_ratio = 0.0
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
        phase_type: u8,
        cluster_crossover_ratio: f64,
        pool_shuffle_ratio: f64,
        assortative_mating_ratio: f64,
    ) -> PyResult<(Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>, usize, usize)> {
        // Returns: (candidates, evaluated, viable)
        let num_clusters = if !population.is_empty() {
            population[0].1.len()  // neurons_per_cluster length = num_clusters
        } else {
            return Ok((Vec::new(), 0, 0));
        };
        let total_input_bits = self.inner.total_input_bits();
        let phase = match phase_type { 1 => neighbor_search::PhaseType::Bits, 2 => neighbor_search::PhaseType::Connections, 3 => neighbor_search::PhaseType::Cluster, _ => neighbor_search::PhaseType::Neurons };

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
            phase_type: phase,
            cluster_crossover_ratio,
            pool_shuffle_ratio,
            assortative_mating_ratio,
        };

        // Set up live progress for observer thread
        let lp_arc = self.inner.live_progress.clone();
        let exp_id = self.experiment_id.unwrap_or(0);
        if let Ok(mut guard) = lp_arc.write() {
            *guard = Some(neighbor_search::LiveProgress {
                experiment_id: exp_id,
                generation: generation.map(|g| g as i32 + 1).unwrap_or(1),
                total_generations: total_generations.map(|g| g as i32).unwrap_or(100),
                phase: "ga_offspring".into(),
                evaluated: 0, target_count, viable: Some(0),
                best_ce: f64::MAX, best_acc: 0.0, elapsed_secs: 0.0,
                updated_at: neighbor_search::LiveProgress::now_unix(),
            });
        }

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                crate::token_cache::evaluate_genomes_cached_hybrid(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, eval_subset_idx, empty_value,
                    1.0, 0, // no neuron sampling for GA offspring search
                )
            };

            let lp_ref = Some(&lp_arc);

            let result = neighbor_search::search_offspring(
                &population, target_count, max_attempts, accuracy_threshold,
                &ga_config, &eval_fn, seed, log_path_ref,
                generation, total_generations, return_best_n, lp_ref,
            );

            let candidates: Vec<_> = result.candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect();
            Ok((candidates, result.evaluated, result.viable))
        });

        // Clear live progress after search completes
        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
    }
}

// =============================================================================
// IDSCache Python Wrapper
// =============================================================================

/// Python wrapper for IDS (Intrusion Detection System) cache.
///
/// Holds pre-encoded binary features for classification tasks.
/// Parallel to TokenCacheWrapper but for tabular binary data.
#[pyclass]
struct IDSCacheWrapper {
    inner: ids_cache::IDSCache,
}

#[pymethods]
impl IDSCacheWrapper {
    /// Create a new IDS cache with stratified partitioning.
    ///
    /// NOTE: This constructor takes `Vec<bool>` which requires a Python list.
    /// For large datasets (e.g. 46M examples × 1000+ input bits), Python list
    /// materialization explodes memory to ~8 bytes per element (pointer size)
    /// totalling hundreds of gigabytes. Prefer `new_from_numpy` for those cases.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (train_features, train_labels, eval_features, eval_labels, num_classes, total_features, num_parts, num_negatives, seed, balance_classes=false, single_cluster=false, undersample_majority=false, class_weight_multiplier=1.0))]
    fn new(
        train_features: Vec<bool>,
        train_labels: Vec<i64>,
        eval_features: Vec<bool>,
        eval_labels: Vec<i64>,
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
    ) -> Self {
        // Pack Python-list bools into PackedBits. This constructor is the
        // small-data backwards-compat path (tests, demos); large datasets
        // should use new_from_numpy with np.packbits.
        let train_packed = packed_bits::PackedBits::from_bool_slice(&train_features, total_features);
        let eval_packed = packed_bits::PackedBits::from_bool_slice(&eval_features, total_features);
        Self {
            inner: ids_cache::IDSCache::new(
                train_packed,
                train_labels,
                eval_packed,
                eval_labels,
                num_classes,
                total_features,
                num_parts,
                num_negatives,
                seed,
                balance_classes,
                single_cluster,
                undersample_majority,
                class_weight_multiplier,
            ),
        }
    }

    /// Create a new IDS cache from bit-packed numpy arrays (Phase 2 packed boundary).
    ///
    /// `train_features` / `eval_features` are uint8 arrays produced by
    /// `np.packbits(bool_matrix, axis=1, bitorder='little')`. Each row of the
    /// logical bool matrix is stored as `ceil(total_features/8)` bytes,
    /// LSB-first within each byte.
    ///
    /// For a 46M × 96-bit CIC-IoT-2023 training set this is ~552 MB (was ~4.4 GB
    /// as Vec<bool>, or ~35 GB as a Python list). No bool materialization
    /// happens on the Rust side.
    ///
    /// Args:
    ///   train_features: numpy uint8 array, packed bytes (np.packbits output).
    ///                   Shape: (num_train * bytes_per_row,) where
    ///                   bytes_per_row = ceil(total_features / 8).
    ///   eval_features:  numpy uint8 array, same packed layout.
    ///   total_features: logical bit-width per row (used for stride math).
    ///   All other args match `new()`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (train_features, train_labels, eval_features, eval_labels, num_classes, total_features, num_parts, num_negatives, seed, balance_classes=false, single_cluster=false, undersample_majority=false, class_weight_multiplier=1.0))]
    fn new_from_numpy<'py>(
        train_features: PyReadonlyArray1<'py, u8>,
        train_labels: Vec<i64>,
        eval_features: PyReadonlyArray1<'py, u8>,
        eval_labels: Vec<i64>,
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
    ) -> PyResult<Self> {
        // Zero-copy views into the numpy packed-byte arrays.
        let train_slice = train_features.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("train_features array not contiguous: {}", e),
            )
        })?;
        let eval_slice = eval_features.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("eval_features array not contiguous: {}", e),
            )
        })?;

        // Interpret incoming uint8 as packed bytes (np.packbits output).
        let train_packed = packed_bits::PackedBits::from_packed_bytes(
            train_slice.to_vec(), total_features,
        );
        let eval_packed = packed_bits::PackedBits::from_packed_bytes(
            eval_slice.to_vec(), total_features,
        );

        Ok(Self {
            inner: ids_cache::IDSCache::new(
                train_packed,
                train_labels,
                eval_packed,
                eval_labels,
                num_classes,
                total_features,
                num_parts,
                num_negatives,
                seed,
                balance_classes,
                single_cluster,
                undersample_majority,
                class_weight_multiplier,
            ),
        })
    }

    /// Get the next train subset index (advances rotator).
    fn next_train_idx(&mut self) -> usize {
        self.inner.next_train_idx()
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

    /// Get number of classes.
    fn num_classes(&self) -> usize {
        self.inner.num_classes()
    }

    /// Get total features (input bits).
    fn total_features(&self) -> usize {
        self.inner.total_features()
    }

    /// Set which class is "normal" for FPR computation.
    /// Call with 1 when flip_labels is active so FPR measures false alarms
    /// on the original benign class (which is class 1 after flipping).
    fn set_normal_class(&mut self, normal_class: usize) {
        self.inner.normal_class = normal_class;
    }

    /// Set fitness weights for threshold optimization.
    /// When set, threshold sweep maximizes fitness instead of F1.
    fn set_fitness_weights(&mut self, w_ce: f32, w_f1: f32, w_fpr: f32, w_acc: f32) {
        self.inner.fitness_weights = Some((w_ce, w_f1, w_fpr, w_acc));
    }

    /// Evaluate genomes using hybrid CPU+GPU with a specific train subset.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64, u32)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_genomes_ids_cached_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                train_subset_idx,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Evaluate genomes using full data with hybrid CPU+GPU.
    #[pyo3(signature = (genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat, num_genomes, empty_value, neuron_sample_rate, rng_seed, override_threshold=None))]
    fn evaluate_genomes_full_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        override_threshold: Option<f64>,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64, u32)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_genomes_ids_cached_full_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                empty_value,
                neuron_sample_rate,
                rng_seed,
                override_threshold,
            ))
        })
    }

    /// Evaluate genomes using K-fold cross-validation with hybrid CPU+GPU.
    ///
    /// Merges all subsets except `held_out_fold` for training, uses the
    /// held-out fold as the eval set. Rotates the held-out fold each
    /// generation for more robust F1/FPR estimates.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes_kfold_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        held_out_fold: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64, u32)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_genomes_ids_kfold_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                held_out_fold,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Evaluate genomes with training-time adaptation (*genesis).
    ///
    /// Returns (ce, acc, f1, adapted_bits, adapted_neurons, adapted_conns,
    ///          pruned, grown, added, removed, rewired) per genome.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
        num_genomes, train_subset_idx, empty_value, neuron_sample_rate, rng_seed,
        synaptogenesis_enabled, neurogenesis_enabled, axonogenesis_enabled = false,
        min_bits = 4, max_bits = 24,
        warmup_generations = 10, total_generations = 250, generation = 0,
        total_input_bits = 336, stats_sample_size = 10000, passes_per_eval = 1,
        prune_entropy_ratio = 0.3, grow_fill_utilization = 0.5, grow_error_baseline = 0.35,
        min_neurons = 3, max_neurons_per_pass = 3, max_growth_ratio = 1.5,
        cooldown_iterations = 5, stabilize_fraction = 0.25,
        axon_entropy_threshold = 0.3, axon_improvement_factor = 1.2, axon_rewire_count = 2
    ))]
    fn evaluate_genomes_hybrid_adaptive(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        synaptogenesis_enabled: bool,
        neurogenesis_enabled: bool,
        axonogenesis_enabled: bool,
        min_bits: usize,
        max_bits: usize,
        warmup_generations: usize,
        total_generations: usize,
        generation: usize,
        total_input_bits: usize,
        stats_sample_size: usize,
        passes_per_eval: usize,
        prune_entropy_ratio: f32,
        grow_fill_utilization: f32,
        grow_error_baseline: f32,
        min_neurons: usize,
        max_neurons_per_pass: usize,
        max_growth_ratio: f32,
        cooldown_iterations: usize,
        stabilize_fraction: f32,
        axon_entropy_threshold: f32,
        axon_improvement_factor: f32,
        axon_rewire_count: usize,
    ) -> PyResult<Vec<(f64, f64, f64, f64, Vec<usize>, Vec<usize>, Vec<i64>, usize, usize, usize, usize, usize)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            let adapt_config = adaptation::AdaptationConfig {
                synaptogenesis_enabled,
                neurogenesis_enabled,
                axonogenesis_enabled,
                min_bits,
                max_bits,
                warmup_generations,
                total_generations,
                total_input_bits,
                stats_sample_size,
                passes_per_eval,
                neuron_sample_rate,
                prune_entropy_ratio,
                grow_fill_utilization,
                grow_error_baseline,
                min_neurons,
                max_neurons_per_pass,
                max_growth_ratio,
                cooldown_iterations,
                stabilize_fraction,
                axon_entropy_threshold,
                axon_improvement_factor,
                axon_rewire_count,
                ..Default::default()
            };
            let results = ids_cache::evaluate_genomes_ids_cached_hybrid_adaptive(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                train_subset_idx,
                empty_value,
                neuron_sample_rate,
                rng_seed,
                &adapt_config,
                generation,
            );
            Ok(results.into_iter().map(|r| (
                r.ce, r.accuracy, r.f1_macro, r.fpr,
                r.adapted_bits, r.adapted_neurons, r.adapted_connections,
                r.pruned, r.grown, r.added, r.removed, r.rewired,
            )).collect())
        })
    }

    /// Train a single genome on full training data and return per-example predictions.
    ///
    /// Returns Vec<i64> of predicted class indices for each eval example.
    /// Used by the bitwise ECOC classifier for per-bit predictions.
    fn predict_examples(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<i64>> {
        py.allow_threads(|| {
            Ok(ids_cache::predict_examples_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Train a single genome and return per-example RAW SCORES (not thresholded).
    ///
    /// Returns Vec<f64> of raw scores for each eval example.
    /// Used for Platt scaling calibration.
    fn score_examples(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            Ok(ids_cache::score_examples_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Score TRAINING examples for calibration fitting.
    /// Trains on full train, returns scores for train (not eval).
    fn score_train_examples(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            Ok(ids_cache::score_train_examples_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Train a single genome ONCE and evaluate at multiple thresholds.
    ///
    /// Returns a tuple (eval_scores, train_scores, metrics) where
    /// `metrics[i] = (ce, acc, f1, fpr, resolved_threshold)` for thresholds[i].
    /// A threshold of -1.0 is the oracle sentinel — sweeps eval scores for the
    /// optimal F1 threshold and uses that.
    ///
    /// Replaces the validation-phase pattern of 7 evaluate_batch_full + 1
    /// score_examples + 1 score_train_examples (9 trainings) with a single
    /// training pass.
    fn evaluate_at_thresholds(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        thresholds: Vec<f64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<(f64, f64, f64, f64, f64)>)> {
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_at_thresholds_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                &thresholds,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Search for neighbors above accuracy threshold, all in Rust.
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
        empty_value,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0
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
        empty_value: f32,
        seed: u64,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
        phase_type: u8,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>> {
        let num_clusters = base_neurons.len();
        let total_input_bits = self.inner.total_features();
        let phase = match phase_type {
            1 => neighbor_search::PhaseType::Bits,
            2 => neighbor_search::PhaseType::Connections,
            _ => neighbor_search::PhaseType::Neurons,
        };

        let config = neighbor_search::MutationConfig {
            num_clusters,
            mutable_clusters,
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            total_input_bits,
            phase_type: phase,
        };

        let lp_arc = self.inner.live_progress.clone();

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                ids_cache::evaluate_genomes_ids_cached_hybrid(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, empty_value,
                    1.0, 0,
                ).into_iter().map(|(ce, acc, f1, fpr, _, _)| (ce, acc, f1, fpr)).collect()
            };

            let lp_ref = Some(&lp_arc);

            let candidates = if return_best_n {
                neighbor_search::search_neighbors_best_n(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                )
            } else {
                let (passed, _) = neighbor_search::search_neighbors_with_threshold(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                );
                passed
            };

            Ok(candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect())
        });

        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
    }

    /// Search for GA offspring above accuracy threshold, all in Rust.
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
        empty_value,
        neuron_sample_rate,
        seed,
        fitness_scores = None,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0,
        cluster_crossover_ratio = 0.0,
        pool_shuffle_ratio = 0.0,
        assortative_mating_ratio = 0.0
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
        empty_value: f32,
        neuron_sample_rate: f32,
        seed: u64,
        // Per-genome harmonic-rank fitness from Python's fitness_calculator. When
        // provided, this REPLACES the population tuple's 4th element (which
        // Python builds as raw CE — see architecture_strategies.py:958) so
        // Rust's tournament_select picks parents by the actual fitness the user
        // configured (weights ce/acc/f1/fpr), not by CE alone. Before this fix,
        // every IDS GA run since 18662b5f (2026-03-09) used CE for parent
        // selection regardless of fitness_calculator weights — weights only
        // affected elite preservation + reporting, not GA exploration direction.
        fitness_scores: Option<Vec<f64>>,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
        phase_type: u8,
        cluster_crossover_ratio: f64,
        pool_shuffle_ratio: f64,
        assortative_mating_ratio: f64,
    ) -> PyResult<(Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>, usize, usize)> {
        let num_clusters = if !population.is_empty() {
            population[0].1.len()
        } else {
            return Ok((Vec::new(), 0, 0));
        };
        let total_input_bits = self.inner.total_features();
        let phase = match phase_type {
            1 => neighbor_search::PhaseType::Bits,
            2 => neighbor_search::PhaseType::Connections,
            _ => neighbor_search::PhaseType::Neurons,
        };

        let ga_config = neighbor_search::GAConfig {
            num_clusters,
            mutable_clusters,
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            crossover_rate,
            tournament_size,
            total_input_bits,
            phase_type: phase,
            cluster_crossover_ratio,
            pool_shuffle_ratio,
            assortative_mating_ratio,
        };

        let lp_arc = self.inner.live_progress.clone();

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            // Use the SAME evaluation path as elites (kfold_hybrid on the
            // held-out fold of train) AND the SAME neuron_sample_rate the
            // cache was built with — otherwise offspring metrics are computed
            // on a different dataset partition (the 20% held-out full_eval, a
            // methodology violation per CLAUDE.md) at a different sample_rate
            // (1.0 vs 0.25 production), making fitness comparison apples-to-
            // oranges. The bug was silent for CIC-IoT random splits
            // (train ≈ test distribution) but surfaced on UNSW-temporal as
            // "offspring collapse" (offspring CE 0.26 vs elite CE 0.13 for
            // the SAME genome). See the GA debug agent's investigation report
            // in the session JSONL for the smoking-gun analysis.
            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                ids_cache::evaluate_genomes_ids_kfold_hybrid(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, empty_value,
                    neuron_sample_rate, 0,
                ).into_iter().map(|(ce, acc, f1, fpr, _, _)| (ce, acc, f1, fpr)).collect()
            };

            let lp_ref = Some(&lp_arc);

            // Replace the population tuple's 4th element with harmonic-rank
            // fitness when provided (matches the fitness used for elite
            // selection in Python; without this, tournament_select uses raw
            // CE which makes weight variations meaningless for parent
            // selection — see fitness_scores doc on the param above).
            let population_for_search = if let Some(scores) = fitness_scores {
                if scores.len() != population.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "fitness_scores length {} != population length {}",
                        scores.len(),
                        population.len()
                    )));
                }
                population
                    .into_iter()
                    .zip(scores.into_iter())
                    .map(|((b, n, c, _ce), s)| (b, n, c, s))
                    .collect()
            } else {
                population
            };

            let result = neighbor_search::search_offspring(
                &population_for_search, target_count, max_attempts, accuracy_threshold,
                &ga_config, &eval_fn, seed, log_path_ref,
                generation, total_generations, return_best_n, lp_ref,
            );

            let candidates: Vec<_> = result.candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect();
            Ok((candidates, result.evaluated, result.viable))
        });

        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
    }
}

// =============================================================================
// IDSCacheBuilder — Phase 5 F-prep chunked accumulator
// =============================================================================

/// Chunked builder for IDSCache. Lets Python feed train/eval data in slabs
/// and finalize into the same IDSCache structure that `IDSCacheWrapper.new_from_numpy`
/// produces.
///
/// Usage (Python):
///     b = IDSCacheBuilder(num_classes, total_features, num_parts, num_negatives,
///                         seed, balance_classes, single_cluster,
///                         undersample_majority, class_weight_multiplier)
///     for chunk_packed, chunk_labels in encoder.iter_chunks(df_train, chunk_size=1_000_000):
///         b.add_train_chunk(chunk_packed.ravel(), chunk_labels)
///     for chunk_packed, chunk_labels in encoder.iter_chunks(df_test, chunk_size=1_000_000):
///         b.add_eval_chunk(chunk_packed.ravel(), chunk_labels)
///     cache = b.finalize()
///
/// For Phase 2-onward (single-chunk case) prefer `IDSCacheWrapper.new_from_numpy`.
/// The builder is the entry point for Option F's streaming/memmap data path.
#[pyclass]
struct IDSCacheBuilderWrapper {
    /// Option<...> so finalize() can take ownership and consume the state.
    inner: Option<ids_cache::PartialFitState>,
}

#[pymethods]
impl IDSCacheBuilderWrapper {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (num_classes, total_features, num_parts, num_negatives, seed, balance_classes=false, single_cluster=false, undersample_majority=false, class_weight_multiplier=1.0))]
    fn new(
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
    ) -> Self {
        Self {
            inner: Some(ids_cache::PartialFitState::new(
                num_classes,
                total_features,
                num_parts,
                num_negatives,
                seed,
                balance_classes,
                single_cluster,
                undersample_majority,
                class_weight_multiplier,
            )),
        }
    }

    /// Append a chunk of training data.
    ///
    /// Args:
    ///   chunk_packed: numpy uint8 (np.packbits output, LSB-first within byte),
    ///                 shape (chunk_rows * bytes_per_row,) flattened.
    ///   chunk_labels: numpy int64 of length chunk_rows.
    fn add_train_chunk<'py>(
        &mut self,
        chunk_packed: PyReadonlyArray1<'py, u8>,
        chunk_labels: Vec<i64>,
    ) -> PyResult<()> {
        let state = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSCacheBuilder already finalized — cannot add more chunks",
            )
        })?;
        let slice = chunk_packed.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("chunk_packed not contiguous: {}", e))
        })?;
        let pb = packed_bits::PackedBits::from_packed_bytes(slice.to_vec(), state.total_features());
        state.add_train_chunk(&pb, &chunk_labels);
        Ok(())
    }

    /// Append a chunk of evaluation data.
    fn add_eval_chunk<'py>(
        &mut self,
        chunk_packed: PyReadonlyArray1<'py, u8>,
        chunk_labels: Vec<i64>,
    ) -> PyResult<()> {
        let state = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSCacheBuilder already finalized — cannot add more chunks",
            )
        })?;
        let slice = chunk_packed.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("chunk_packed not contiguous: {}", e))
        })?;
        let pb = packed_bits::PackedBits::from_packed_bytes(slice.to_vec(), state.total_features());
        state.add_eval_chunk(&pb, &chunk_labels);
        Ok(())
    }

    /// Number of training rows accumulated so far.
    fn num_train(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|s| s.num_train())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("IDSCacheBuilder already finalized")
            })
    }

    /// Number of eval rows accumulated so far.
    fn num_eval(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|s| s.num_eval())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("IDSCacheBuilder already finalized")
            })
    }

    /// Consume the builder and produce an IDSCacheWrapper. After this call,
    /// no more chunks may be added (the builder is one-shot).
    fn finalize(&mut self) -> PyResult<IDSCacheWrapper> {
        let state = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSCacheBuilder already finalized — cannot finalize twice",
            )
        })?;
        Ok(IDSCacheWrapper {
            inner: state.finalize(),
        })
    }
}

// =============================================================================
// IDSGenomeStreamer — true streaming genome evaluator (Option F)
// =============================================================================

/// Python wrapper for `ids_streaming::IDSGenomeStreamer`.
///
/// Per-genome streaming evaluator: holds only memory cells + a small score
/// buffer across chunks. Peak memory bounded by one chunk regardless of N.
/// See `ids_streaming.rs` for the full design.
///
/// Lifecycle (Python):
///     s = IDSGenomeStreamer(bits_flat, neurons_flat, connections, ...)
///     for packed_chunk, labels in train_stream:
///         s.train_chunk(packed_chunk.ravel(), labels)
///     s.seal_for_scoring()
///     for packed_chunk, labels in eval_stream:
///         s.score_chunk(packed_chunk.ravel(), labels)
///     ce, acc, f1, fpr, threshold = s.finalize_metrics()
#[pyclass]
struct IDSGenomeStreamerWrapper {
    inner: Option<ids_streaming::IDSGenomeStreamer>,
}

#[pymethods]
impl IDSGenomeStreamerWrapper {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (bits_flat, neurons_flat, connections, num_classes, num_negatives, single_cluster, total_features, empty_value=0.5, neuron_sample_rate=1.0, rng_seed=42, normal_class=0, class_weights=None))]
    fn new(
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections: Vec<i64>,
        num_classes: usize,
        num_negatives: usize,
        single_cluster: bool,
        total_features: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        normal_class: usize,
        class_weights: Option<Vec<u32>>,
    ) -> Self {
        let _ = total_features; // used by Python-side validation; not needed in Rust ctor
        Self {
            inner: Some(ids_streaming::IDSGenomeStreamer::new(
                bits_flat,
                neurons_flat,
                connections,
                num_classes,
                num_negatives,
                single_cluster,
                normal_class,
                empty_value,
                neuron_sample_rate,
                rng_seed,
                class_weights,
            )),
        }
    }

    /// Train on a single chunk. Writes to memory cells in-place.
    fn train_chunk<'py>(
        &mut self,
        chunk_packed: PyReadonlyArray1<'py, u8>,
        labels: Vec<i64>,
        total_features: usize,
    ) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSGenomeStreamer already finalized",
            )
        })?;
        let slice = chunk_packed.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("chunk not contiguous: {}", e))
        })?;
        let pb = packed_bits::PackedBits::from_packed_bytes(slice.to_vec(), total_features);
        inner.train_chunk(&pb, &labels);
        Ok(())
    }

    /// Finalize training; build the GPU-ready genome export. After this call,
    /// `train_chunk` raises and `score_chunk` becomes valid.
    fn seal_for_scoring(&mut self) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSGenomeStreamer already finalized",
            )
        })?;
        inner.seal_for_scoring();
        Ok(())
    }

    /// Score a single eval chunk. Accumulates per-row scores.
    fn score_chunk<'py>(
        &mut self,
        chunk_packed: PyReadonlyArray1<'py, u8>,
        labels: Vec<i64>,
        total_features: usize,
    ) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSGenomeStreamer already finalized",
            )
        })?;
        let slice = chunk_packed.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("chunk not contiguous: {}", e))
        })?;
        let pb = packed_bits::PackedBits::from_packed_bytes(slice.to_vec(), total_features);
        inner.score_chunk(&pb, &labels);
        Ok(())
    }

    /// Compute final metrics. Consumes the inner state (one-shot).
    /// Returns (ce, acc, f1, fpr, threshold).
    fn finalize_metrics(&mut self) -> PyResult<(f64, f64, f64, f64, f64)> {
        let inner = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSGenomeStreamer already finalized — cannot finalize twice",
            )
        })?;
        Ok(inner.finalize_metrics())
    }

    fn train_seen(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|i| i.train_seen())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
            })
    }

    fn eval_scored(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|i| i.eval_scored())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
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
    fn forward_batch<'py>(&self, py: Python<'py>, input_bits_flat: PyReadonlyArray1<'py, u8>, batch_size: usize) -> Vec<f32> {
        let total_bits = self.inner.config().total_input_bits;
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let packed = packed_bits::PackedBits::from_bool_bytes(input_slice, total_bits);
        py.allow_threads(|| self.inner.forward_batch(&packed, batch_size))
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
    fn train_batch<'py>(
        &self,
        py: Python<'py>,
        input_bits_flat: PyReadonlyArray1<'py, u8>,
        target_gates_flat: PyReadonlyArray1<'py, u8>,
        batch_size: usize,
        allow_override: bool,
    ) -> usize {
        let total_bits = self.inner.config().total_input_bits;
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let packed = packed_bits::PackedBits::from_bool_bytes(input_slice, total_bits);
        let target_bools: Vec<bool> = target_gates_flat.as_slice()
            .expect("target_gates_flat must be contiguous")
            .iter().map(|&b| b != 0).collect();
        py.allow_threads(|| {
            self.inner.train_batch(&packed, &target_bools, batch_size, allow_override)
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
    fn forward_batch_metal<'py>(&self, py: Python<'py>, input_bits_flat: PyReadonlyArray1<'py, u8>, batch_size: usize) -> PyResult<Vec<f32>> {
        let config = self.inner.config();
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let pb = packed_bits::PackedBits::from_bool_bytes(input_slice, config.total_input_bits);
        py.allow_threads(|| {
            let (packed, wpe) = neuron_memory::pack_packed_to_u64(&pb);
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;
            evaluator.forward_batch(&self.inner, &packed, batch_size, wpe)
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
    fn forward_batch_hybrid<'py>(&self, py: Python<'py>, input_bits_flat: PyReadonlyArray1<'py, u8>, batch_size: usize, cpu_fraction: f32) -> PyResult<Vec<f32>> {
        let total_bits = self.inner.config().total_input_bits;
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let pb = packed_bits::PackedBits::from_bool_bytes(input_slice, total_bits);
        py.allow_threads(|| {
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;
            metal_gating::forward_batch_hybrid(&self.inner, evaluator, &pb, batch_size, cpu_fraction)
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
    fn train_batch_metal<'py>(
        &self,
        py: Python<'py>,
        input_bits_flat: PyReadonlyArray1<'py, u8>,
        target_gates_flat: PyReadonlyArray1<'py, u8>,
        batch_size: usize,
    ) -> PyResult<usize> {
        let input_bools: Vec<bool> = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous")
            .iter().map(|&b| b != 0).collect();
        let target_bools: Vec<bool> = target_gates_flat.as_slice()
            .expect("target_gates_flat must be contiguous")
            .iter().map(|&b| b != 0).collect();
        py.allow_threads(|| {
            let config = self.inner.config();
            let (packed, wpe) = neuron_memory::pack_bools_to_u64(&input_bools, batch_size, config.total_input_bits);
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;

            // Train on GPU and get updated memory
            let updated_memory = evaluator.train_batch(&self.inner, &packed, &target_bools, batch_size, wpe)
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
fn compute_target_gates<'py>(py: Python<'py>, targets: Vec<i64>, num_clusters: usize) -> pyo3::Bound<'py, numpy::PyArray1<u8>> {
    let gates = gating::compute_target_gates(&targets, num_clusters);
    let bytes: Vec<u8> = gates.iter().map(|&b| b as u8).collect();
    numpy::PyArray1::from_vec(py, bytes)
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
    /// Optional override: None = auto-compute per genome based on budget.
    sparse_threshold_override: Option<usize>,
    experiment_id: Option<i64>,
}

#[pymethods]
impl BitwiseCacheWrapper {
    #[new]
    #[pyo3(signature = (train_tokens, eval_tokens, vocab_size, context_size, num_parts, num_eval_parts, pad_token_id, sparse_threshold=None))]
    fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        num_parts: usize,
        num_eval_parts: usize,
        pad_token_id: u32,
        sparse_threshold: Option<usize>,
    ) -> Self {
        Self {
            inner: bitwise_ramlm::BitwiseTokenCache::new(
                train_tokens, eval_tokens, vocab_size, context_size,
                num_parts, num_eval_parts, pad_token_id,
            ),
            sparse_threshold_override: sparse_threshold,
            experiment_id: None,
        }
    }

    /// Evaluate genomes with per-neuron heterogeneous configs (subset training + eval).
    ///
    /// bits_per_neuron_flat: variable total (sum of total_neurons per genome)
    /// neurons_per_cluster_flat: [num_genomes * num_clusters]
    /// connections_flat: variable total (sum of all genomes' connections)
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes(
        &self,
        py: Python<'_>,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        memory_mode: u8,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        let override_val = self.sparse_threshold_override;
        py.allow_threads(|| {
            Ok(bitwise_ramlm::evaluate_genomes(
                &self.inner, &bits_per_neuron_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes, train_subset_idx, eval_subset_idx,
                memory_mode, empty_value, neuron_sample_rate, rng_seed, override_val,
            ))
        })
    }

    /// Evaluate genomes with per-neuron heterogeneous configs (full training + full eval).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes_full(
        &self,
        py: Python<'_>,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        memory_mode: u8,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        let override_val = self.sparse_threshold_override;
        py.allow_threads(|| {
            Ok(bitwise_ramlm::evaluate_genomes_full(
                &self.inner, &bits_per_neuron_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes,
                memory_mode, empty_value, neuron_sample_rate, rng_seed, override_val,
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

    /// Set experiment context for live progress reporting.
    fn set_experiment_context(&mut self, experiment_id: i64) {
        self.experiment_id = Some(experiment_id);
    }

    /// Get current live progress from active search (if any).
    fn get_live_progress(&self, py: Python<'_>) -> PyResult<Option<pyo3::PyObject>> {
        let guard = self.inner.live_progress.read()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e)))?;
        match &*guard {
            None => Ok(None),
            Some(lp) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("experiment_id", lp.experiment_id)?;
                dict.set_item("generation", lp.generation)?;
                dict.set_item("total_generations", lp.total_generations)?;
                dict.set_item("phase", &lp.phase)?;
                dict.set_item("evaluated", lp.evaluated)?;
                dict.set_item("target_count", lp.target_count)?;
                match lp.viable {
                    Some(v) => dict.set_item("viable", v)?,
                    None => dict.set_item("viable", py.None())?,
                }
                dict.set_item("best_ce", lp.best_ce)?;
                dict.set_item("best_acc", lp.best_acc)?;
                dict.set_item("elapsed_secs", lp.elapsed_secs)?;
                dict.set_item("updated_at", lp.updated_at)?;
                Ok(Some(dict.into()))
            }
        }
    }

    /// Search for neighbors above accuracy threshold (bitwise eval backend).
    ///
    /// Same interface as TokenCacheWrapper::search_neighbors but uses the
    /// bitwise evaluation path (heterogeneous per-neuron configs).
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
        memory_mode,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0
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
        memory_mode: u8,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        seed: u64,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
        phase_type: u8,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>> {
        let num_clusters = base_neurons.len();
        let total_input_bits = self.inner.total_input_bits;
        let phase = match phase_type { 1 => neighbor_search::PhaseType::Bits, 2 => neighbor_search::PhaseType::Connections, 3 => neighbor_search::PhaseType::Cluster, _ => neighbor_search::PhaseType::Neurons };

        let config = neighbor_search::MutationConfig {
            num_clusters,
            mutable_clusters,
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            total_input_bits,
            phase_type: phase,
        };

        let override_val = self.sparse_threshold_override;

        // Set up live progress for observer thread
        let lp_arc = self.inner.live_progress.clone();
        let exp_id = self.experiment_id.unwrap_or(0);
        if let Ok(mut guard) = lp_arc.write() {
            *guard = Some(neighbor_search::LiveProgress {
                experiment_id: exp_id,
                generation: generation.map(|g| g as i32 + 1).unwrap_or(1),
                total_generations: total_generations.map(|g| g as i32).unwrap_or(100),
                phase: "ts_neighbors".into(),
                evaluated: 0, target_count, viable: Some(0),
                best_ce: f64::MAX, best_acc: 0.0, elapsed_secs: 0.0,
                updated_at: neighbor_search::LiveProgress::now_unix(),
            });
        }

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                bitwise_ramlm::evaluate_genomes(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, eval_subset_idx,
                    memory_mode, empty_value, neuron_sample_rate, rng_seed, override_val,
                )
            };

            let lp_ref = Some(&lp_arc);

            let candidates = if return_best_n {
                neighbor_search::search_neighbors_best_n(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                )
            } else {
                let (passed, _) = neighbor_search::search_neighbors_with_threshold(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                );
                passed
            };

            Ok(candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect())
        });

        // Clear live progress after search completes
        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
    }

    /// Search for GA offspring above accuracy threshold (bitwise eval backend).
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
        memory_mode,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0,
        cluster_crossover_ratio = 0.0,
        pool_shuffle_ratio = 0.0,
        assortative_mating_ratio = 0.0
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
        memory_mode: u8,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        seed: u64,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
        phase_type: u8,
        cluster_crossover_ratio: f64,
        pool_shuffle_ratio: f64,
        assortative_mating_ratio: f64,
    ) -> PyResult<(Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>, usize, usize)> {
        let num_clusters = if !population.is_empty() {
            population[0].1.len()
        } else {
            return Ok((Vec::new(), 0, 0));
        };
        let total_input_bits = self.inner.total_input_bits;
        let phase = match phase_type { 1 => neighbor_search::PhaseType::Bits, 2 => neighbor_search::PhaseType::Connections, 3 => neighbor_search::PhaseType::Cluster, _ => neighbor_search::PhaseType::Neurons };

        let ga_config = neighbor_search::GAConfig {
            num_clusters,
            mutable_clusters,
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            crossover_rate,
            tournament_size,
            total_input_bits,
            phase_type: phase,
            cluster_crossover_ratio,
            pool_shuffle_ratio,
            assortative_mating_ratio,
        };

        let override_val = self.sparse_threshold_override;

        // Set up live progress for observer thread
        let lp_arc = self.inner.live_progress.clone();
        let exp_id = self.experiment_id.unwrap_or(0);
        if let Ok(mut guard) = lp_arc.write() {
            *guard = Some(neighbor_search::LiveProgress {
                experiment_id: exp_id,
                generation: generation.map(|g| g as i32 + 1).unwrap_or(1),
                total_generations: total_generations.map(|g| g as i32).unwrap_or(100),
                phase: "ga_offspring".into(),
                evaluated: 0, target_count, viable: Some(0),
                best_ce: f64::MAX, best_acc: 0.0, elapsed_secs: 0.0,
                updated_at: neighbor_search::LiveProgress::now_unix(),
            });
        }

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                bitwise_ramlm::evaluate_genomes(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, eval_subset_idx,
                    memory_mode, empty_value, neuron_sample_rate, rng_seed, override_val,
                )
            };

            let lp_ref = Some(&lp_arc);

            let result = neighbor_search::search_offspring(
                &population, target_count, max_attempts, accuracy_threshold,
                &ga_config, &eval_fn, seed, log_path_ref,
                generation, total_generations, return_best_n, lp_ref,
            );

            let candidates: Vec<_> = result.candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect();
            Ok((candidates, result.evaluated, result.viable))
        });

        // Clear live progress after search completes
        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
    }

    /// Evaluate multiple genomes with per-genome adaptation (Baldwin effect).
    ///
    /// Each genome is adapted (synaptogenesis/neurogenesis) during evaluation,
    /// so GA/TS sees adapted fitness. Returns adapted architecture per genome.
    fn evaluate_genomes_adaptive(
        &self,
        py: Python<'_>,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        memory_mode: u8,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        generation: usize,
        // Adaptation config fields (relative thresholds)
        synaptogenesis_enabled: bool,
        neurogenesis_enabled: bool,
        axonogenesis_enabled: bool,
        prune_entropy_ratio: f32,
        grow_fill_utilization: f32,
        grow_error_baseline: f32,
        min_bits: usize,
        max_bits: usize,
        cluster_error_factor: f32,
        cluster_fill_utilization: f32,
        neuron_prune_percentile: f32,
        neuron_removal_factor: f32,
        max_growth_ratio: f32,
        min_neurons: usize,
        max_neurons_per_pass: usize,
        axon_entropy_threshold: f32,
        axon_improvement_factor: f32,
        axon_rewire_count: usize,
        warmup_generations: usize,
        cooldown_iterations: usize,
        stabilize_fraction: f32,
        total_generations: usize,
        passes_per_eval: usize,
        stats_sample_size: usize,
    ) -> PyResult<Vec<(
        f64, f64, f64,                      // ce, acc, bit_acc
        Vec<usize>, Vec<usize>, Vec<i64>,   // adapted bits, neurons, connections
        usize, usize, usize, usize, usize,  // pruned, grown, added, removed, rewired
    )>> {
        let override_val = self.sparse_threshold_override;
        let total_input_bits = self.inner.total_input_bits;

        let config = adaptation::AdaptationConfig {
            synaptogenesis_enabled,
            neurogenesis_enabled,
            axonogenesis_enabled,
            axon_entropy_threshold,
            axon_improvement_factor,
            axon_rewire_count,
            prune_entropy_ratio,
            grow_fill_utilization,
            grow_error_baseline,
            min_bits,
            max_bits,
            cluster_error_factor,
            cluster_fill_utilization,
            neuron_prune_percentile,
            neuron_removal_factor,
            max_growth_ratio,
            min_neurons,
            max_neurons_per_pass,
            warmup_generations,
            cooldown_iterations,
            stabilize_fraction,
            total_generations,
            passes_per_eval,
            total_input_bits,
            stats_sample_size,
            neuron_sample_rate,
        };

        py.allow_threads(|| {
            let cache = &self.inner;
            let train_subset = &cache.train_subsets[train_subset_idx % cache.num_parts];
            let eval_subset = &cache.eval_subsets[eval_subset_idx % cache.num_eval_parts];

            let results = bitwise_ramlm::evaluate_genomes_adaptive(
                cache,
                &bits_per_neuron_flat,
                &neurons_per_cluster_flat,
                &connections_flat,
                num_genomes,
                train_subset,
                eval_subset,
                memory_mode,
                empty_value,
                neuron_sample_rate,
                rng_seed,
                override_val,
                &config,
                generation,
            );

            Ok(results.into_iter().map(|r| (
                r.ce, r.acc, r.bit_acc,
                r.adapted_bits, r.adapted_neurons, r.adapted_connections,
                r.pruned, r.grown, r.added, r.removed, r.rewired,
            )).collect())
        })
    }
}

// =============================================================================
// Multi-Stage Token Cache — PyO3 wrapper for stage-agnostic RAM evaluation
// =============================================================================

#[pyclass]
struct MultiStageCacheWrapper {
    inner: multistage::MultiStageTokenCache,
    sparse_threshold_override: Option<usize>,
    reweight_rounds: usize,
    reweight_max_boost: usize,
    live_progress: Arc<RwLock<Option<neighbor_search::LiveProgress>>>,
    experiment_id: i64,
    progress_generation: i32,
    progress_total_generations: i32,
    progress_phase: String,
}

#[pymethods]
impl MultiStageCacheWrapper {
    #[new]
    #[pyo3(signature = (train_tokens, eval_tokens, vocab_size, context_size, k, num_parts, num_eval_parts, pad_token_id, sparse_threshold=None, stage_cluster_types=None, custom_cluster_of=None, stage_context_sizes=None, reweight_rounds=None, reweight_max_boost=None))]
    fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        k: usize,
        num_parts: usize,
        num_eval_parts: usize,
        pad_token_id: u32,
        sparse_threshold: Option<usize>,
        stage_cluster_types: Option<Vec<String>>,
        custom_cluster_of: Option<Vec<u16>>,
        stage_context_sizes: Option<Vec<usize>>,
        reweight_rounds: Option<usize>,
        reweight_max_boost: Option<usize>,
    ) -> Self {
        let rw_rounds = reweight_rounds.unwrap_or(0);
        let rw_max_boost = reweight_max_boost.unwrap_or(4);
        let mut cache = multistage::MultiStageTokenCache::new(
            train_tokens, eval_tokens, vocab_size, context_size,
            k, num_parts, num_eval_parts, pad_token_id,
            stage_cluster_types, custom_cluster_of,
            stage_context_sizes,
        );
        cache.reweight_rounds = rw_rounds;
        cache.reweight_max_boost = rw_max_boost;
        Self {
            inner: cache,
            sparse_threshold_override: sparse_threshold,
            reweight_rounds: rw_rounds,
            reweight_max_boost: rw_max_boost,
            live_progress: Arc::new(RwLock::new(None)),
            experiment_id: 0,
            progress_generation: 0,
            progress_total_generations: 0,
            progress_phase: "evaluate_batch".into(),
        }
    }

    fn set_experiment_context(&mut self, experiment_id: i64) {
        self.experiment_id = experiment_id;
    }

    /// Set progress context (generation, total, phase) so Rust sub-batch
    /// updates report correct values to the observer thread.
    /// Pre-seeds the LiveProgress Arc so Rust only needs to update
    /// evaluated/target_count/elapsed_secs in place.
    fn set_progress_context(&mut self, generation: i32, total_generations: i32, phase: String) {
        self.progress_generation = generation;
        self.progress_total_generations = total_generations;
        self.progress_phase = phase.clone();
        // Pre-seed the Arc so Rust sub-batch updates preserve these fields
        if let Ok(mut guard) = self.live_progress.write() {
            *guard = Some(neighbor_search::LiveProgress {
                experiment_id: self.experiment_id,
                generation,
                total_generations,
                phase,
                evaluated: 0,
                target_count: 0,
                viable: None,
                best_ce: 0.0,
                best_acc: 0.0,
                elapsed_secs: 0.0,
                updated_at: neighbor_search::LiveProgress::now_unix(),
            });
        }
    }

    fn get_live_progress(&self, py: Python<'_>) -> PyResult<Option<pyo3::PyObject>> {
        let guard = self.live_progress.read()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("lock: {e}")))?;
        match &*guard {
            None => Ok(None),
            Some(lp) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("experiment_id", lp.experiment_id)?;
                dict.set_item("generation", lp.generation)?;
                dict.set_item("total_generations", lp.total_generations)?;
                dict.set_item("phase", &lp.phase)?;
                dict.set_item("evaluated", lp.evaluated)?;
                dict.set_item("target_count", lp.target_count)?;
                match lp.viable {
                    Some(v) => dict.set_item("viable", v)?,
                    None => dict.set_item("viable", py.None())?,
                }
                dict.set_item("best_ce", lp.best_ce)?;
                dict.set_item("best_acc", lp.best_acc)?;
                dict.set_item("elapsed_secs", lp.elapsed_secs)?;
                Ok(Some(dict.into()))
            }
        }
    }

    // ── Bitwise evaluation (stage-agnostic) ─────────────────────────

    /// Evaluate bitwise genomes for any stage with subset rotation.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_bitwise_genomes(
        &self,
        py: Python<'_>,
        stage: usize,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        let override_val = self.sparse_threshold_override;
        let lp_arc = self.live_progress.clone();
        let exp_id = self.experiment_id;
        py.allow_threads(|| {
            Ok(multistage::evaluate_bitwise_genomes(
                &self.inner, stage, &bits_per_neuron_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes, train_subset_idx, eval_subset_idx,
                memory_mode, neuron_sample_rate, rng_seed, override_val,
                Some(&lp_arc), exp_id,
            ))
        })
    }

    /// Evaluate bitwise genomes with full (non-rotated) data.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_bitwise_genomes_full(
        &self,
        py: Python<'_>,
        stage: usize,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        let override_val = self.sparse_threshold_override;
        py.allow_threads(|| {
            Ok(multistage::evaluate_bitwise_genomes_full(
                &self.inner, stage, &bits_per_neuron_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes,
                memory_mode, neuron_sample_rate, rng_seed, override_val,
            ))
        })
    }

    /// Evaluate bitwise genomes for one group (selector mode).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_bitwise_selector_genomes(
        &self,
        py: Python<'_>,
        stage: usize,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        group_id: usize,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        let override_val = self.sparse_threshold_override;
        py.allow_threads(|| {
            Ok(multistage::evaluate_bitwise_selector_genomes(
                &self.inner, stage, &bits_per_neuron_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes, group_id,
                memory_mode, neuron_sample_rate, rng_seed, override_val,
            ))
        })
    }

    /// Get the number of eval examples per selector group for a given stage.
    fn selector_eval_counts(&self, stage: usize) -> Vec<usize> {
        if stage < self.inner.bitwise_selector_eval.len() {
            self.inner.bitwise_selector_eval[stage]
                .iter()
                .map(|s| s.num_examples)
                .collect()
        } else {
            Vec::new()
        }
    }

    // ── Rotation ────────────────────────────────────────────────────

    fn next_train_idx(&self) -> usize {
        self.inner.next_train_idx()
    }

    fn next_eval_idx(&self) -> usize {
        self.inner.next_eval_idx()
    }

    fn reset(&self) {
        self.inner.reset();
    }

    // ── Clustering info (stage-agnostic) ────────────────────────────

    fn k(&self) -> usize { self.inner.k }
    fn vocab_size(&self) -> usize { self.inner.vocab_size }
    fn context_size(&self) -> usize { self.inner.max_context_size }
    fn stage_context_sizes(&self) -> Vec<usize> { self.inner.stage_context_sizes.clone() }
    fn max_cluster_size(&self) -> usize { self.inner.max_cluster_size }
    fn num_parts(&self) -> usize { self.inner.num_parts }
    fn num_eval_parts(&self) -> usize { self.inner.num_eval_parts }

    fn cluster_sizes(&self) -> Vec<usize> {
        self.inner.cluster_sizes.clone()
    }

    /// Get total input bits for a given stage.
    fn stage_input_bits(&self, stage: usize) -> usize {
        self.inner.stage_input_bits.get(stage).copied().unwrap_or(0)
    }

    /// Get output bits (target bit count) for a given bitwise stage.
    fn bitwise_output_bits(&self, stage: usize) -> usize {
        self.inner.bitwise_output_bits.get(stage).copied().unwrap_or(0)
    }

    // ── Combined CE computation ─────────────────────────────────────

    /// Compute combined multi-stage CE from per-stage genome params.
    ///
    /// Takes flat concatenated arrays for all stages, plus stage_num_clusters
    /// to partition them.
    ///
    /// Returns: (combined_ce, combined_accuracy, stage0_ce, stage1_ce)
    #[allow(clippy::too_many_arguments)]
    fn evaluate_combined_ce(
        &self,
        py: Python<'_>,
        all_bits_per_neuron: Vec<usize>,
        all_neurons_per_cluster: Vec<usize>,
        all_connections: Vec<i64>,
        stage_num_clusters: Vec<usize>,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
        sparse_threshold: usize,
        label_smoothing: f64,
        unigram_lambda: f64,
        bigram_lambda: f64,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        py.allow_threads(|| {
            // Partition flat arrays by stage
            let num_stages = stage_num_clusters.len();
            let mut stage_bits: Vec<&[usize]> = Vec::with_capacity(num_stages);
            let mut stage_neurons: Vec<&[usize]> = Vec::with_capacity(num_stages);
            let mut stage_conns: Vec<&[i64]> = Vec::with_capacity(num_stages);

            let mut neuron_offset = 0usize;
            let mut cluster_offset = 0usize;
            let mut conn_offset = 0usize;

            for s in 0..num_stages {
                let n_clusters = stage_num_clusters[s];
                let neurons_slice = &all_neurons_per_cluster[cluster_offset..cluster_offset + n_clusters];
                let total_neurons: usize = neurons_slice.iter().sum();
                let bits_slice = &all_bits_per_neuron[neuron_offset..neuron_offset + total_neurons];
                let total_conns: usize = bits_slice.iter().sum();
                let conns_slice = &all_connections[conn_offset..conn_offset + total_conns];

                stage_bits.push(bits_slice);
                stage_neurons.push(neurons_slice);
                stage_conns.push(conns_slice);

                neuron_offset += total_neurons;
                cluster_offset += n_clusters;
                conn_offset += total_conns;
            }

            Ok(multistage::compute_combined_ce(
                &self.inner,
                &stage_bits, &stage_neurons, &stage_conns,
                memory_mode, neuron_sample_rate, rng_seed, sparse_threshold,
                label_smoothing, unigram_lambda, bigram_lambda,
            ))
        })
    }

    /// Compute combined CE for SELECTOR mode.
    ///
    /// S0 is evaluated normally (bitwise or tiered).
    /// S1 is evaluated per-group using selector data (K sub-models).
    ///
    /// When `invalid_mode` is true (Phase C), S1 groups train on ALL examples
    /// with an "invalid" target for out-of-group data, enabling self-correction
    /// of S0 mistakes. `top_m` limits the number of groups each example trains
    /// on (0 = all groups).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_combined_ce_selector(
        &self,
        py: Python<'_>,
        all_bits_per_neuron: Vec<usize>,
        all_neurons_per_cluster: Vec<usize>,
        all_connections: Vec<i64>,
        stage_num_clusters: Vec<usize>,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
        sparse_threshold: usize,
        label_smoothing: f64,
        invalid_mode: bool,
        top_m: usize,
        unigram_lambda: f64,
        bigram_lambda: f64,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        py.allow_threads(|| {
            // Partition flat arrays by stage (same as evaluate_combined_ce)
            let num_stages = stage_num_clusters.len();
            let mut stage_bits: Vec<&[usize]> = Vec::with_capacity(num_stages);
            let mut stage_neurons: Vec<&[usize]> = Vec::with_capacity(num_stages);
            let mut stage_conns: Vec<&[i64]> = Vec::with_capacity(num_stages);

            let mut neuron_offset = 0usize;
            let mut cluster_offset = 0usize;
            let mut conn_offset = 0usize;

            for s in 0..num_stages {
                let n_clusters = stage_num_clusters[s];
                let neurons_slice = &all_neurons_per_cluster[cluster_offset..cluster_offset + n_clusters];
                let total_neurons: usize = neurons_slice.iter().sum();
                let bits_slice = &all_bits_per_neuron[neuron_offset..neuron_offset + total_neurons];
                let total_conns: usize = bits_slice.iter().sum();
                let conns_slice = &all_connections[conn_offset..conn_offset + total_conns];

                stage_bits.push(bits_slice);
                stage_neurons.push(neurons_slice);
                stage_conns.push(conns_slice);

                neuron_offset += total_neurons;
                cluster_offset += n_clusters;
                conn_offset += total_conns;
            }

            Ok(multistage::compute_combined_ce_selector(
                &self.inner,
                &stage_bits, &stage_neurons, &stage_conns,
                memory_mode, neuron_sample_rate, rng_seed, sparse_threshold,
                label_smoothing, invalid_mode, top_m, unigram_lambda, bigram_lambda,
            ))
        })
    }

    // ── Tiered stage methods ────────────────────────────────────────

    fn is_stage_tiered(&self, stage: usize) -> bool {
        self.inner.stage_is_tiered.get(stage).copied().unwrap_or(false)
    }

    fn stage_num_output_clusters(&self, stage: usize) -> usize {
        self.inner.stage_num_output_clusters.get(stage).copied().unwrap_or(0)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_tiered_genomes(
        &self,
        py: Python<'_>,
        stage: usize,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        eval_subset_idx: usize,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        py.allow_threads(|| {
            Ok(multistage::evaluate_tiered_genomes(
                &self.inner, stage,
                &bits_per_neuron_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes,
                train_subset_idx, eval_subset_idx,
                memory_mode, neuron_sample_rate, rng_seed,
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_tiered_genomes_full(
        &self,
        py: Python<'_>,
        stage: usize,
        bits_per_neuron_flat: Vec<usize>,
        neurons_per_cluster_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        num_genomes: usize,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        py.allow_threads(|| {
            Ok(multistage::evaluate_tiered_genomes_full(
                &self.inner, stage,
                &bits_per_neuron_flat, &neurons_per_cluster_flat,
                &connections_flat, num_genomes,
                memory_mode, neuron_sample_rate, rng_seed,
            ))
        })
    }

    /// Re-encode data for `target_stage` using frozen previous stage's actual predictions.
    ///
    /// Trains the frozen stage on full train data, gets predictions for both train and
    /// eval, then re-encodes all `target_stage` data with those predictions instead of
    /// teacher forcing.
    ///
    /// Returns: (train_accuracy, eval_accuracy) — prediction accuracy for the frozen stage.
    #[allow(clippy::too_many_arguments)]
    fn recompute_stage_with_predictions(
        &mut self,
        py: Python<'_>,
        frozen_stage: usize,
        target_stage: usize,
        bits_per_neuron: Vec<usize>,
        neurons_per_cluster: Vec<usize>,
        connections: Vec<i64>,
        memory_mode: u8,
        neuron_sample_rate: f32,
        rng_seed: u64,
        sparse_threshold: usize,
    ) -> PyResult<(f64, f64)> {
        let rw_rounds = self.reweight_rounds;
        let rw_max_boost = self.reweight_max_boost;
        py.allow_threads(|| {
            let (train_preds, eval_preds, train_correct, eval_correct) =
                multistage::predict_stage_clusters(
                    &self.inner,
                    frozen_stage,
                    &bits_per_neuron,
                    &neurons_per_cluster,
                    &connections,
                    memory_mode,
                    neuron_sample_rate,
                    rng_seed,
                    sparse_threshold,
                    rw_rounds,
                    rw_max_boost,
                );

            let num_train = self.inner.bitwise_full_train[frozen_stage].num_examples;
            let num_eval = self.inner.bitwise_full_eval[frozen_stage].num_examples;

            self.inner.recompute_stage_data(target_stage, &train_preds, &eval_preds);

            let train_acc = train_correct as f64 / num_train.max(1) as f64;
            let eval_acc = eval_correct as f64 / num_eval.max(1) as f64;

            Ok((train_acc, eval_acc))
        })
    }
}

// =============================================================================
// Standalone utility: random connection generation (Rust-accelerated)
// =============================================================================

/// Generate random connections for a genome entirely in Rust.
///
/// Args:
///   bits_per_neuron: List of bit counts per neuron (flat, [total_neurons])
///   total_input_bits: Number of input bits to choose from
///   seed: RNG seed for reproducibility
///
/// Returns: List of random connections in [0, total_input_bits), length = sum(bits_per_neuron)
#[pyfunction]
fn generate_random_connections(
    bits_per_neuron: Vec<usize>,
    total_input_bits: usize,
    seed: u64,
) -> Vec<i64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    neighbor_search::generate_random_connections(&bits_per_neuron, total_input_bits, &mut rng)
}

/// Python module definition
/// ABI version of the accelerator's Python surface. Bump on any breaking
/// change to an exported signature; wnn/accel.py asserts it at import so a
/// stale build fails loudly instead of silently mis-marshalling.
pub const ABI_VERSION: u32 = 2;

#[pymodule]
fn ram_accelerator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI_VERSION", ABI_VERSION)?;
    m.add_function(wrap_pyfunction!(metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(reset_metal_evaluators, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_cores, m)?)?;
    // New batch prediction functions
    // Exact probs acceleration (bit-encoded - deprecated, slow due to export)
    // Exact probs acceleration (word-based - FAST, no export overhead)
    // RAMLM acceleration (proper RAM WNN architecture)
    m.add_function(wrap_pyfunction!(ramlm_train_batch_numpy, m)?)?;  // FAST numpy-based training
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_batch_numpy, m)?)?;  // Bitwise multi-label training (dense)
    m.add_function(wrap_pyfunction!(ramlm_train_batch_tiered_numpy, m)?)?;  // FAST tiered training (all tiers in one call)
    // RAMLM Metal GPU acceleration
    m.add_function(wrap_pyfunction!(ramlm_metal_available, m)?)?;
    // RAMLM NumPy-based acceleration (FAST - zero-copy)
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_metal_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_hybrid_numpy, m)?)?;
    // RAMLM Cached Metal (avoids shader recompilation)
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_metal_cached, m)?)?;
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
    // Parallel GA/TS for TIERED architectures (16 cores parallel)
    // Parallel GA/TS HYBRID CPU+GPU (memory-adaptive, pipelined)
    // Hybrid with explicit memory budget
    // Memory estimation utilities
    // Option C foundation — atomic CAS microbench (CPU+GPU coherence test)
    m.add_function(wrap_pyfunction!(run_atomic_cas_microbench, m)?)?;
    // Option B B0 — MarkerHashTable unit tests
    m.add_function(wrap_pyfunction!(run_marker_hashtable_tests, m)?)?;
    // Option B B2 — Metal marker-train kernel parity test
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(run_marker_train_parity_test, m)?)?;
    // Option B B4-batched — multi-genome Metal kernel parity test
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(run_marker_train_batched_parity_test, m)?)?;
    // Option B B5 — multi-cluster Metal kernel parity test
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(run_marker_train_multicluster_parity_test, m)?)?;
    // Per-cluster optimization (Rust-accelerated discriminative optimization)
    // Global CE with caching (true global softmax over all 50K clusters)
    // Group-based optimization for iterative refinement
    // EMPTY cell value configuration (affects PPL calculation)
    // Adaptive architecture (per-cluster variable bits/neurons)
    m.add_function(wrap_pyfunction!(adaptive_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_train_batch, m)?)?;
    // Parallel genome evaluation (KEY for GA optimization)
    m.add_function(wrap_pyfunction!(evaluate_genomes_parallel, m)?)?;
    // Multi-subset parallel genome evaluation (for per-iteration rotation)
    // Parallel hybrid CPU+GPU genome evaluation (4-8x speedup)
    m.add_function(wrap_pyfunction!(evaluate_genomes_parallel_hybrid, m)?)?;
    // FPGA export: train one genome + return per-neuron sparse (keys, values)
    // Token cache for persistent token storage with subset rotation
    m.add_class::<TokenCacheWrapper>()?;
    // IDS cache for intrusion detection classification
    m.add_class::<IDSCacheWrapper>()?;
    m.add_class::<IDSCacheBuilderWrapper>()?;
    m.add_class::<IDSGenomeStreamerWrapper>()?;
    // RAM-based gating (weightless per-cluster gating with majority voting)
    m.add_class::<RAMGatingWrapper>()?;
    m.add_function(wrap_pyfunction!(compute_target_gates, m)?)?;
    m.add_function(wrap_pyfunction!(gating_metal_available, m)?)?;
    // Bitwise RAMLM evaluation (full Rust+Metal pipeline)
    m.add_class::<BitwiseCacheWrapper>()?;
    // Multi-stage RAMLM evaluation (group prediction + within-group)
    m.add_class::<MultiStageCacheWrapper>()?;
    // Bitwise RAMLM — nudge training + quad forward
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_batch_nudge_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_and_eval_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_quad_binary_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_quad_weighted_numpy, m)?)?;
    // Utility: Rust-accelerated random connection generation
    m.add_function(wrap_pyfunction!(generate_random_connections, m)?)?;
    m.add_function(wrap_pyfunction!(find_optimal_threshold_fitness_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_platt_scaling_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_beta_calibration_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_empirical_threshold_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_binary_metrics_at_threshold_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_optimal_threshold_f1_py, m)?)?;

    // Drone-controller hot-path (paper #1). See controller.rs and
    // project_drone_controller_paper1.md.
    m.add_class::<controller::AttitudeSim>()?;
    m.add_class::<controller::WnnController>()?;
    m.add_class::<controller::AttitudePidRs>()?;
    // dagger_train scaffold (B.2 in progress) — see dagger_train.rs.
    m.add_class::<dagger_train::RewardGatedConfigPacked>()?;
    m.add_class::<dagger_train::TrainStats>()?;
    m.add_function(wrap_pyfunction!(dagger_train::dagger_train_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(dagger_train::dagger_train_batch_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(controller::strategy_5_qsr_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(controller::strategy_1_count_true, m)?)?;
    m.add_function(wrap_pyfunction!(controller::monotonicity_violations, m)?)?;
    m.add_function(wrap_pyfunction!(controller::compute_reward, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::score_controllers_metal, m)?)?;
    // Cooperative cancellation for the controller + IDS evaluators. Python's
    // SIGTERM handler calls set_cancel_flag(), Rust callsites poll at safe
    // boundaries (between genomes / episodes / GPU dispatch chunks) and return
    // partial results.
    m.add_function(wrap_pyfunction!(cancel::set_cancel_flag, m)?)?;
    m.add_function(wrap_pyfunction!(cancel::reset_cancel_flag, m)?)?;
    m.add_function(wrap_pyfunction!(cancel::is_cancelled, m)?)?;

    // EDRA constraint solver (Rust port of Memory._solve_partial_connectivity).
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_trinary_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_qsr_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_trinary_reachable_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_qsr_reachable_py, m)?)?;

    Ok(())
}
