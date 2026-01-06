//! RAM Accelerator - High-performance RAM neuron evaluation for Apple Silicon
//!
//! Provides GPU-accelerated evaluation of RAM neuron connectivity patterns
//! using Metal compute shaders on M-series Macs.

use pyo3::prelude::*;
use pyo3::types::PyModule;
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
/// # Performance
/// - Python loop: ~10+ min for 287k tokens
/// - Rust parallel: ~seconds for 287k tokens
#[pyfunction]
fn compute_exact_probs_words(
    py: Python<'_>,
    exact_rams: Vec<(
        usize,
        std::collections::HashMap<Vec<String>, std::collections::HashMap<String, u32>>,
    )>,
    tokens: Vec<String>,
) -> PyResult<Vec<Option<f64>>> {
    py.allow_threads(|| {
        let results = ram::compute_exact_probs_words(exact_rams, &tokens);
        Ok(results)
    })
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
    Ok(())
}
