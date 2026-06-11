//! Per-cluster optimization (Rust-accelerated discriminative optimization).
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

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
pub(crate) fn adaptive_forward_batch<'py>(
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
pub(crate) fn adaptive_train_batch<'py>(
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
pub(crate) fn evaluate_genomes_parallel<'py>(
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
pub(crate) fn evaluate_genomes_parallel_hybrid<'py>(
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
