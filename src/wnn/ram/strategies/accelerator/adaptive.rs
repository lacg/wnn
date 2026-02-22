//! Adaptive Architecture Accelerator
//!
//! High-performance training and forward pass for AdaptiveClusteredRAM
//! where each cluster can have its own (bits, neurons) configuration.
//!
//! Key optimization: Clusters are grouped by their config to enable
//! efficient batch processing within each group.
//!
//! Memory strategy:
//! - Dense memory (bit-packed Vec) for bits <= SPARSE_THRESHOLD
//! - Sparse memory (DashMap) for bits > SPARSE_THRESHOLD
//! This enables up to 30 bits without memory explosion.
//!
//! Metal GPU Acceleration:
//! - Dense groups can be evaluated on Metal GPU (40 cores on M4 Max)
//! - Sparse groups stay on CPU (hash lookups not GPU-friendly)
//! - Hybrid approach: Metal for dense, CPU for sparse

use dashmap::DashMap;
#[cfg(target_os = "macos")]
use metal;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// Re-export from eval_worker module for backward compatibility
pub use crate::eval_worker::{EvalData, get_eval_worker};

// =============================================================================
// Resettable Metal Evaluators
// =============================================================================
//
// Metal evaluators can accumulate driver-level state over long runs, causing
// slowdowns. These use Arc + RwLock to allow periodic reset.
// Call reset_metal_evaluators() every N generations to recreate fresh evaluators.

// Global counter incremented on each reset
static RESET_GENERATION: AtomicU64 = AtomicU64::new(0);

// Storage for resettable evaluators - uses Arc so callers can hold references
static METAL_EVALUATOR: RwLock<Option<Arc<crate::metal_ramlm::MetalRAMLMEvaluator>>> = RwLock::new(None);
static SPARSE_METAL_EVALUATOR: RwLock<Option<Arc<crate::metal_ramlm::MetalSparseEvaluator>>> = RwLock::new(None);
static GROUP_EVALUATOR: RwLock<Option<Arc<crate::metal_ramlm::MetalGroupEvaluator>>> = RwLock::new(None);

/// Get or initialize the Metal evaluator (resettable, thread-safe)
/// Returns an Arc that can be held across lock boundaries
/// Set WNN_NO_METAL=1 to disable Metal and use CPU-only evaluation (for diagnostics)
pub fn get_metal_evaluator() -> Option<Arc<crate::metal_ramlm::MetalRAMLMEvaluator>> {
    // Check for Metal disable flag (for diagnostics)
    if std::env::var("WNN_NO_METAL").is_ok() {
        return None;
    }

    // Fast path: check if initialized
    {
        let guard = METAL_EVALUATOR.read().unwrap();
        if let Some(ref arc) = *guard {
            return Some(Arc::clone(arc));
        }
    }

    // Slow path: need to initialize
    let mut guard = METAL_EVALUATOR.write().unwrap();
    if guard.is_none() {
        if let Ok(eval) = crate::metal_ramlm::MetalRAMLMEvaluator::new() {
            *guard = Some(Arc::new(eval));
        }
    }
    guard.as_ref().map(Arc::clone)
}

/// Get or initialize the sparse Metal evaluator (resettable, thread-safe)
/// Set WNN_NO_METAL=1 to disable Metal and use CPU-only evaluation (for diagnostics)
pub fn get_sparse_metal_evaluator() -> Option<Arc<crate::metal_ramlm::MetalSparseEvaluator>> {
    // Check for Metal disable flag (for diagnostics)
    if std::env::var("WNN_NO_METAL").is_ok() {
        return None;
    }

    // Fast path: check if initialized
    {
        let guard = SPARSE_METAL_EVALUATOR.read().unwrap();
        if let Some(ref arc) = *guard {
            return Some(Arc::clone(arc));
        }
    }

    // Slow path: need to initialize
    let mut guard = SPARSE_METAL_EVALUATOR.write().unwrap();
    if guard.is_none() {
        if let Ok(eval) = crate::metal_ramlm::MetalSparseEvaluator::new() {
            *guard = Some(Arc::new(eval));
        }
    }
    guard.as_ref().map(Arc::clone)
}

/// Get or initialize the group evaluator (resettable, thread-safe)
/// Set WNN_NO_METAL=1 to disable Metal and use CPU-only evaluation (for diagnostics)
fn get_group_evaluator() -> Option<Arc<crate::metal_ramlm::MetalGroupEvaluator>> {
    // Check for Metal disable flag (for diagnostics)
    if std::env::var("WNN_NO_METAL").is_ok() {
        return None;
    }

    // Fast path: check if initialized
    {
        let guard = GROUP_EVALUATOR.read().unwrap();
        if let Some(ref arc) = *guard {
            return Some(Arc::clone(arc));
        }
    }

    // Slow path: need to initialize
    let mut guard = GROUP_EVALUATOR.write().unwrap();
    if guard.is_none() {
        if let Ok(eval) = crate::metal_ramlm::MetalGroupEvaluator::new() {
            *guard = Some(Arc::new(eval));
        }
    }
    guard.as_ref().map(Arc::clone)
}

/// Reset all Metal evaluators to free accumulated driver state.
///
/// Call this periodically (e.g., every 50 generations) to prevent slowdown
/// from Metal driver state accumulation during long optimization runs.
///
/// The evaluators will be lazily re-initialized on next use.
/// Existing Arc references will continue to work until dropped.
pub fn reset_metal_evaluators() {
    // Increment generation counter (for scores/input buffer cache in evaluate_genome_hybrid)
    RESET_GENERATION.fetch_add(1, Ordering::SeqCst);

    // Also reset the sparse buffer cache (for per-group buffers in eval_sparse_to_buffer)
    crate::metal_ramlm::reset_sparse_buffer_cache();

    // Clear all evaluators - existing Arc holders keep their reference
    // until dropped, then the evaluator is truly freed
    if let Ok(mut guard) = METAL_EVALUATOR.write() {
        *guard = None;
    }
    if let Ok(mut guard) = SPARSE_METAL_EVALUATOR.write() {
        *guard = None;
    }
    if let Ok(mut guard) = GROUP_EVALUATOR.write() {
        *guard = None;
    }
}

/// Threshold for switching to sparse memory (2^12 = 4K addresses)
const SPARSE_THRESHOLD: usize = 12;

use crate::neuron_memory::{
    FALSE, TRUE, EMPTY, BITS_PER_CELL, CELLS_PER_WORD, CELL_MASK,
    compute_address, NeuronTrainMeta,
};

/// Get the EMPTY cell value from the unified global setting
fn get_empty_value() -> f32 {
    crate::neuron_memory::get_empty_value()
}

/// Check if group coalescing is enabled (set WNN_COALESCE_GROUPS=1)
fn use_coalesced_groups() -> bool {
    std::env::var("WNN_COALESCE_GROUPS").is_ok()
}

/// Build config groups with optional coalescing based on environment variable
/// When WNN_COALESCE_GROUPS is set, similar neuron counts are bucketed together
/// to reduce GPU dispatch overhead while preserving accuracy through masking
pub fn build_groups(bits_per_cluster: &[usize], neurons_per_cluster: &[usize]) -> Vec<ConfigGroup> {
    if use_coalesced_groups() {
        build_config_groups_coalesced(bits_per_cluster, neurons_per_cluster)
    } else {
        build_config_groups(bits_per_cluster, neurons_per_cluster)
    }
}

/// Reorganize connections from Python's cluster-order layout to coalesced group layout
///
/// Python generates connections in cluster ID order:
///   [cluster_0_conns, cluster_1_conns, ..., cluster_N_conns]
///   where cluster_i has neurons_per_cluster[i] * bits_per_cluster[i] connections
///
/// Coalesced groups expect connections organized by group with padding:
///   [group_0_cluster_conns, group_1_cluster_conns, ...]
///   where each cluster in group has group.neurons (MAX) * group.bits connections
///   and actual connections are followed by padding (-1) to reach MAX neurons
///
/// Returns: padded connections in group order, ready for coalesced evaluation
pub fn reorganize_connections_for_coalescing(
    original_connections: &[i64],
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
    groups: &[ConfigGroup],
) -> Vec<i64> {
    let num_clusters = bits_per_cluster.len();

    // Build mapping: cluster_id -> offset in original_connections
    let mut cluster_offsets = vec![0usize; num_clusters];
    let mut offset = 0;
    for cluster_id in 0..num_clusters {
        cluster_offsets[cluster_id] = offset;
        offset += neurons_per_cluster[cluster_id] * bits_per_cluster[cluster_id];
    }

    // Total size needed for coalesced layout
    let total_size: usize = groups.iter().map(|g| g.conn_size()).sum();
    let mut result = vec![-1i64; total_size];  // Initialize with padding value

    // For each group, copy connections for each cluster (with padding)
    let mut write_offset = 0;
    for group in groups {
        let max_neurons = group.neurons;
        let bits = group.bits;

        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_idx] as usize
            } else {
                max_neurons  // Uniform case
            };

            // Source: original connections for this cluster
            let src_offset = cluster_offsets[cluster_id];
            let src_size = actual_neurons * bits;

            // Destination: position in coalesced layout
            // Each cluster in group gets max_neurons * bits slots
            let dst_offset = write_offset + local_idx * max_neurons * bits;

            // Copy actual connections
            result[dst_offset..dst_offset + src_size]
                .copy_from_slice(&original_connections[src_offset..src_offset + src_size]);

            // Remaining slots (dst_offset + src_size .. dst_offset + max_neurons * bits)
            // are already -1 (padding)
        }

        write_offset += group.cluster_ids.len() * max_neurons * bits;
    }

    result
}

/// Convert per-neuron bits to per-cluster max bits (for `build_groups`).
///
/// `bits_per_neuron` has length `sum(neurons_per_cluster)` — one entry per neuron.
/// Returns one entry per cluster: the maximum bits among that cluster's neurons.
/// Pattern from `bitwise_ramlm.rs:1391-1400`.
pub(crate) fn per_cluster_max_bits(bits_per_neuron: &[usize], neurons_per_cluster: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(neurons_per_cluster.len());
    let mut offset = 0;
    for &nc in neurons_per_cluster {
        let max_b = bits_per_neuron[offset..offset + nc].iter().copied().max().unwrap_or(0);
        result.push(max_b);
        offset += nc;
    }
    result
}

/// Build per-neuron offset tables for heterogeneous-bits training.
///
/// Returns `(cluster_neuron_starts, neuron_conn_offsets)`:
/// - `cluster_neuron_starts[c]` = first neuron index for cluster `c`
/// - `neuron_conn_offsets[n]` = connection start offset for neuron `n` (cumulative sum of bits)
///
/// Pattern from `bitwise_ramlm.rs:683-704` (`compute_genome_layout`).
pub(crate) fn build_neuron_metadata(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let num_clusters = neurons_per_cluster.len();
    let total_neurons: usize = neurons_per_cluster.iter().sum();

    // cluster_neuron_starts[c] = index of first neuron in cluster c
    let mut cluster_neuron_starts = Vec::with_capacity(num_clusters);
    let mut cumul = 0usize;
    for &nc in neurons_per_cluster {
        cluster_neuron_starts.push(cumul);
        cumul += nc;
    }

    // neuron_conn_offsets[n] = start offset in connections array for neuron n
    let mut neuron_conn_offsets = Vec::with_capacity(total_neurons);
    let mut conn_off = 0usize;
    for &b in bits_per_neuron {
        neuron_conn_offsets.push(conn_off);
        conn_off += b;
    }

    (cluster_neuron_starts, neuron_conn_offsets)
}

/// Pad per-neuron connections to group layout for GPU dispatch.
///
/// Each neuron's `n_bits` connections are padded to `group.bits` (= cluster max_bits) with
/// connection index 0 (harmless padding). Same pattern as `bitwise_ramlm.rs:804-820`.
///
/// This replaces `reorganize_connections_for_coalescing` when per-neuron bits are heterogeneous.
pub(crate) fn reorganize_connections_for_gpu(
    original_connections: &[i64],
    per_neuron_bits: &[usize],
    neurons_per_cluster: &[usize],
    groups: &[ConfigGroup],
) -> Vec<i64> {
    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);

    // Total size needed for group layout
    let total_size: usize = groups.iter().map(|g| g.conn_size()).sum();
    // Initialize with -1 (skipped by GPU shader's `if conn_idx >= 0` check)
    let mut result = vec![-1i64; total_size];

    for group in groups {
        let max_neurons = group.neurons;
        let max_bits = group.bits;

        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_idx] as usize
            } else {
                max_neurons
            };

            let neuron_start = cluster_neuron_starts[cluster_id];

            for n in 0..actual_neurons {
                let global_n = neuron_start + n;
                let n_bits = per_neuron_bits[global_n];
                let conn_start = neuron_conn_offsets[global_n];

                // Destination in group layout: PREFIX-pad with -1, real connections at END.
                // GPU shader computes address bit i as (max_bits-1-i), so real connections
                // at the end match training's bit positions (actual_bits-1-i).
                let dst = group.conn_offset + local_idx * max_neurons * max_bits + n * max_bits;
                let pad_size = max_bits - n_bits;
                // Prefix is already -1 from initialization; copy real connections after it
                result[dst + pad_size..dst + pad_size + n_bits]
                    .copy_from_slice(&original_connections[conn_start..conn_start + n_bits]);
            }
        }
    }

    result
}

/// Configuration group - clusters sharing the same (neurons, bits) config
/// For coalesced groups, neurons is the MAX neurons and actual_neurons stores per-cluster values
#[derive(Clone, Debug)]
pub struct ConfigGroup {
    pub neurons: usize,                   // Max neurons (for memory layout)
    pub bits: usize,
    pub words_per_neuron: usize,
    pub cluster_ids: Vec<usize>,          // Global cluster IDs in this group
    pub actual_neurons: Option<Vec<u32>>, // Per-cluster actual neurons (None = all same as neurons)
    pub memory_offset: usize,             // Offset into flattened memory
    pub conn_offset: usize,               // Offset into flattened connections
}

impl ConfigGroup {
    pub fn new(neurons: usize, bits: usize, cluster_ids: Vec<usize>) -> Self {
        let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
        Self {
            neurons,
            bits,
            words_per_neuron,
            cluster_ids,
            actual_neurons: None,  // Uniform: all clusters have same neurons
            memory_offset: 0,
            conn_offset: 0,
        }
    }

    /// Create a coalesced group where clusters may have different actual neuron counts
    /// neurons = max neurons for memory allocation
    /// actual_neurons[i] = actual neuron count for cluster_ids[i]
    pub fn new_coalesced(neurons: usize, bits: usize, cluster_ids: Vec<usize>, actual_neurons: Vec<u32>) -> Self {
        let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
        Self {
            neurons,
            bits,
            words_per_neuron,
            cluster_ids,
            actual_neurons: Some(actual_neurons),
            memory_offset: 0,
            conn_offset: 0,
        }
    }

    pub fn cluster_count(&self) -> usize {
        self.cluster_ids.len()
    }

    pub fn total_neurons(&self) -> usize {
        self.cluster_count() * self.neurons
    }

    /// True total neurons (sum of actual neurons if coalesced)
    pub fn true_total_neurons(&self) -> usize {
        if let Some(ref actual) = self.actual_neurons {
            actual.iter().map(|&n| n as usize).sum()
        } else {
            self.total_neurons()
        }
    }

    pub fn memory_size(&self) -> usize {
        self.total_neurons() * self.words_per_neuron
    }

    pub fn conn_size(&self) -> usize {
        self.total_neurons() * self.bits
    }

    /// Is this a coalesced group with per-cluster masking?
    pub fn is_coalesced(&self) -> bool {
        self.actual_neurons.is_some()
    }
}

/// Maximum GPU output size: 256M addresses = 1GB output buffer.
/// Beyond this, CPU fallback is used to avoid Metal allocation hangs.
const MAX_GPU_ADDRESSES: usize = 256_000_000;

/// Try to compute training addresses on GPU for adaptive training path.
/// Returns None if GPU is unavailable, disabled, or the problem is too large.
pub(crate) fn try_gpu_addresses_adaptive(
    packed_input: &[u64],
    words_per_example: usize,
    per_neuron_bits: &[usize],
    neuron_conn_offsets: &[usize],
    connections: &[i64],
    num_train: usize,
) -> Option<Vec<u32>> {
    let total_neurons = per_neuron_bits.len();
    if total_neurons < 100 {
        return None;
    }
    // Guard against massive allocations (e.g. 251K neurons × 16K examples = 4B addresses = 16GB)
    if total_neurons.saturating_mul(num_train) > MAX_GPU_ADDRESSES {
        return None;
    }

    let trainer_mutex = crate::get_cached_metal_trainer().ok()?;
    let mut guard = trainer_mutex.lock().ok()?;
    let trainer = guard.as_mut()?;

    let neuron_meta: Vec<NeuronTrainMeta> = (0..total_neurons)
        .map(|n| NeuronTrainMeta {
            bits: per_neuron_bits[n] as u32,
            conn_offset: neuron_conn_offsets[n] as u32,
        })
        .collect();

    trainer.compute_addresses(
        packed_input,
        connections,
        &neuron_meta,
        num_train,
        words_per_example,
    ).ok()
}

/// Read a memory cell value
#[inline]
fn read_cell(memory_words: &[i64], neuron_idx: usize, address: usize, words_per_neuron: usize) -> i64 {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = neuron_idx * words_per_neuron + word_idx;
    let word = memory_words[word_offset];
    (word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
}

/// Write a memory cell value (atomic, for parallel writes)
#[inline]
fn write_cell_atomic(
    memory_words: &[AtomicI64],
    neuron_idx: usize,
    address: usize,
    value: i64,
    words_per_neuron: usize,
    allow_override: bool,
) -> bool {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = neuron_idx * words_per_neuron + word_idx;
    let shift = cell_idx * BITS_PER_CELL;
    let mask = CELL_MASK << shift;
    let new_bits = value << shift;

    loop {
        let old_word = memory_words[word_offset].load(Ordering::Acquire);
        let old_cell = (old_word >> shift) & CELL_MASK;

        if !allow_override && old_cell != EMPTY {
            return false;
        }
        if old_cell == value {
            return false;
        }

        let new_word = (old_word & !mask) | new_bits;
        match memory_words[word_offset].compare_exchange(
            old_word, new_word, Ordering::AcqRel, Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(_) => continue,
        }
    }
}

/// Build config groups from per-cluster configuration
///
/// Groups clusters by their (neurons, bits) to enable efficient batch processing.
pub fn build_config_groups(
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
) -> Vec<ConfigGroup> {
    use std::collections::HashMap;

    let num_clusters = bits_per_cluster.len();
    let mut config_to_clusters: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for cluster_id in 0..num_clusters {
        let key = (neurons_per_cluster[cluster_id], bits_per_cluster[cluster_id]);
        config_to_clusters.entry(key).or_default().push(cluster_id);
    }

    let mut groups: Vec<ConfigGroup> = config_to_clusters
        .into_iter()
        .map(|((neurons, bits), cluster_ids)| ConfigGroup::new(neurons, bits, cluster_ids))
        .collect();

    // Sort by (neurons, bits) for deterministic ordering
    groups.sort_by_key(|g| (g.neurons, g.bits));

    // Compute offsets
    let mut memory_offset = 0;
    let mut conn_offset = 0;
    for group in &mut groups {
        group.memory_offset = memory_offset;
        group.conn_offset = conn_offset;
        memory_offset += group.memory_size();
        conn_offset += group.conn_size();
    }

    // Log group diversity if enabled (helps diagnose slowdown from too many groups)
    if std::env::var("WNN_GROUP_LOG").is_ok() {
        let sparse_count = groups.iter().filter(|g| g.bits > 12).count();
        let dense_count = groups.len() - sparse_count;
        eprintln!(
            "[CONFIG_GROUPS] total={} sparse={} dense={} configs={:?}",
            groups.len(),
            sparse_count,
            dense_count,
            groups.iter().map(|g| (g.neurons, g.bits, g.cluster_ids.len())).collect::<Vec<_>>()
        );
    }

    groups
}

/// Bucket neurons into ranges to reduce group diversity
/// Returns the max neurons for the bucket
fn bucket_neurons(neurons: usize) -> usize {
    // Buckets: 1-5→5, 6-10→10, 11-15→15, 16-20→20, 21-25→25, etc.
    // This gives ~5x fewer unique neuron values
    ((neurons + 4) / 5) * 5
}

/// Build config groups with coalescing - buckets similar neuron counts together
/// This reduces the number of GPU dispatches while preserving accuracy through masking.
///
/// Example: If clusters have neurons [5, 6, 7, 8], they bucket into:
///   - 5→5 (bucket 5), 6-10→10 (bucket for 6,7,8)
///   - Instead of 4 groups, we have 2 groups
///
/// For each coalesced group:
///   - neurons = max in bucket (for memory allocation)
///   - actual_neurons[i] = true neuron count for cluster i (for scoring)
pub fn build_config_groups_coalesced(
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
) -> Vec<ConfigGroup> {
    use std::collections::HashMap;

    let num_clusters = bits_per_cluster.len();

    // Key: (bucket_max, bits) -> list of (cluster_id, actual_neurons)
    let mut bucket_to_clusters: HashMap<(usize, usize), Vec<(usize, u32)>> = HashMap::new();

    for cluster_id in 0..num_clusters {
        let actual = neurons_per_cluster[cluster_id];
        let bucket_max = bucket_neurons(actual);
        let bits = bits_per_cluster[cluster_id];
        let key = (bucket_max, bits);
        bucket_to_clusters.entry(key).or_default().push((cluster_id, actual as u32));
    }

    let mut groups: Vec<ConfigGroup> = bucket_to_clusters
        .into_iter()
        .map(|((max_neurons, bits), entries)| {
            let cluster_ids: Vec<usize> = entries.iter().map(|(id, _)| *id).collect();
            let actual_neurons: Vec<u32> = entries.iter().map(|(_, n)| *n).collect();

            // Check if all actual neurons are the same as max (can use uniform mode)
            let all_same = actual_neurons.iter().all(|&n| n as usize == max_neurons);
            if all_same {
                ConfigGroup::new(max_neurons, bits, cluster_ids)
            } else {
                ConfigGroup::new_coalesced(max_neurons, bits, cluster_ids, actual_neurons)
            }
        })
        .collect();

    // Sort by (neurons, bits) for deterministic ordering
    groups.sort_by_key(|g| (g.neurons, g.bits));

    // Compute offsets
    let mut memory_offset = 0;
    let mut conn_offset = 0;
    for group in &mut groups {
        group.memory_offset = memory_offset;
        group.conn_offset = conn_offset;
        memory_offset += group.memory_size();
        conn_offset += group.conn_size();
    }

    // Log group diversity if enabled
    if std::env::var("WNN_GROUP_LOG").is_ok() {
        let sparse_count = groups.iter().filter(|g| g.bits > 12).count();
        let dense_count = groups.len() - sparse_count;
        let coalesced_count = groups.iter().filter(|g| g.is_coalesced()).count();
        eprintln!(
            "[CONFIG_GROUPS_COALESCED] total={} sparse={} dense={} coalesced={} configs={:?}",
            groups.len(),
            sparse_count,
            dense_count,
            coalesced_count,
            groups.iter().map(|g| (g.neurons, g.bits, g.cluster_ids.len(), g.is_coalesced())).collect::<Vec<_>>()
        );
    }

    groups
}

/// Forward pass for adaptive architecture
///
/// Processes each config group efficiently, then scatters results to output.
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits]
///   connections_flat: All groups' connections concatenated
///   memory_words: All groups' memory concatenated
///   groups: Config groups with cluster assignments
///   num_examples: Number of input examples
///   total_input_bits: Total input bits per example
///   num_clusters: Total number of clusters (vocabulary size)
///
/// Returns: [num_examples * num_clusters] probabilities
pub fn forward_batch_adaptive(
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    memory_words: &[i64],
    groups: &[ConfigGroup],
    num_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
) -> Vec<f32> {
    let empty_value = get_empty_value();
    let mut probs = vec![0.0f32; num_examples * num_clusters];

    // Build reverse mapping: global_cluster_id -> (group_idx, local_cluster_idx)
    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    // Process all examples in parallel
    probs.par_chunks_mut(num_clusters).enumerate().for_each(|(ex_idx, ex_probs)| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        // Process each config group
        for group in groups {
            let neurons = group.neurons;
            let bits = group.bits;
            let words_per_neuron = group.words_per_neuron;
            let group_memory = &memory_words[group.memory_offset..];
            let group_conns = &connections_flat[group.conn_offset..];

            // For each cluster in this group
            for (local_idx, &global_cluster_id) in group.cluster_ids.iter().enumerate() {
                // Use actual neurons if coalesced, otherwise MAX (uniform case)
                let actual_neurons = if let Some(ref an) = group.actual_neurons {
                    an[local_idx] as usize
                } else {
                    neurons
                };

                let start_neuron = local_idx * neurons;  // Use MAX for memory layout
                let mut count_true = 0u32;
                let mut count_empty = 0u32;

                for neuron_offset in 0..actual_neurons {  // Only iterate actual neurons
                    let local_neuron = start_neuron + neuron_offset;
                    let conn_start = local_neuron * bits;
                    let connections = &group_conns[conn_start..conn_start + bits];

                    let address = compute_address(input_bits, connections, bits);
                    let cell_value = read_cell(group_memory, local_neuron, address, words_per_neuron);

                    if cell_value == TRUE {
                        count_true += 1;
                    } else if cell_value == EMPTY {
                        count_empty += 1;
                    }
                }

                // Divide by actual neurons for correct probability
                ex_probs[global_cluster_id] =
                    (count_true as f32 + empty_value * count_empty as f32) / actual_neurons as f32;
            }
        }
    });

    probs
}

/// Training for adaptive architecture
///
/// Two-phase training: TRUE first, then FALSE (to ensure TRUE priority).
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits]
///   true_clusters: [num_examples] global cluster indices
///   false_clusters_flat: [num_examples * num_negatives] global cluster indices
///   connections_flat: All groups' connections concatenated
///   memory_words: All groups' memory concatenated (mutable)
///   groups: Config groups with cluster assignments
///
/// Returns: Number of cells modified
pub fn train_batch_adaptive(
    input_bits_flat: &[bool],
    true_clusters: &[i64],
    false_clusters_flat: &[i64],
    connections_flat: &[i64],
    memory_words: &mut [i64],
    groups: &[ConfigGroup],
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
    num_clusters: usize,
    allow_override: bool,
) -> usize {
    // Build reverse mapping: global_cluster_id -> (group_idx, local_cluster_idx)
    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    // Convert memory to atomic for thread-safe writes
    let atomic_memory: &[AtomicI64] = unsafe {
        std::slice::from_raw_parts(
            memory_words.as_ptr() as *const AtomicI64,
            memory_words.len(),
        )
    };

    // Phase 1: Write all TRUEs
    let true_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let true_cluster = true_clusters[ex_idx] as usize;
        let (group_idx, local_cluster) = cluster_to_group[true_cluster];
        let group = &groups[group_idx];

        let neurons = group.neurons;  // MAX for memory layout
        let bits = group.bits;
        let words_per_neuron = group.words_per_neuron;
        let start_neuron = local_cluster * neurons;
        let group_conns = &connections_flat[group.conn_offset..];

        // Use actual neurons if coalesced, otherwise MAX
        let actual_neurons = if let Some(ref an) = group.actual_neurons {
            an[local_cluster] as usize
        } else {
            neurons
        };

        let mut modified = 0usize;
        for neuron_offset in 0..actual_neurons {  // Only iterate actual neurons
            let local_neuron = start_neuron + neuron_offset;
            let conn_start = local_neuron * bits;
            let connections = &group_conns[conn_start..conn_start + bits];

            let address = compute_address(input_bits, connections, bits);
            let _global_neuron_offset = group.memory_offset / words_per_neuron + local_neuron;

            if write_cell_atomic(
                &atomic_memory[group.memory_offset..],
                local_neuron, address, TRUE, words_per_neuron, allow_override,
            ) {
                modified += 1;
            }
        }
        modified
    }).sum();

    // Phase 2: Write all FALSEs (skip if already TRUE)
    let false_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];
        let true_cluster = true_clusters[ex_idx] as usize;

        let false_start = ex_idx * num_negatives;
        let mut modified = 0usize;

        for neg_idx in 0..num_negatives {
            let false_cluster = false_clusters_flat[false_start + neg_idx] as usize;
            if false_cluster == true_cluster {
                continue;
            }

            let (group_idx, local_cluster) = cluster_to_group[false_cluster];
            let group = &groups[group_idx];

            let neurons = group.neurons;  // MAX for memory layout
            let bits = group.bits;
            let words_per_neuron = group.words_per_neuron;
            let start_neuron = local_cluster * neurons;
            let group_conns = &connections_flat[group.conn_offset..];

            // Use actual neurons if coalesced, otherwise MAX
            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                neurons
            };

            for neuron_offset in 0..actual_neurons {  // Only iterate actual neurons
                let local_neuron = start_neuron + neuron_offset;
                let conn_start = local_neuron * bits;
                let connections = &group_conns[conn_start..conn_start + bits];

                let address = compute_address(input_bits, connections, bits);

                if write_cell_atomic(
                    &atomic_memory[group.memory_offset..],
                    local_neuron, address, FALSE, words_per_neuron, false, // Never override TRUE
                ) {
                    modified += 1;
                }
            }
        }
        modified
    }).sum();

    true_modified + false_modified
}

/// Dense memory for a config group (bit-packed, fast for bits <= 12)
/// Uses atomic operations for thread-safe concurrent writes.
pub(crate) struct GroupDenseMemory {
    /// Bit-packed memory words [total_neurons * words_per_neuron]
    words: Vec<AtomicI64>,
    words_per_neuron: usize,
}

impl GroupDenseMemory {
    fn new(num_neurons: usize, bits: usize) -> Self {
        let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
        let total_words = num_neurons * words_per_neuron;
        // Initialize all cells to EMPTY (pack 31 EMPTY values per word)
        let empty_word: i64 = (0..31).fold(0i64, |acc, i| acc | (EMPTY << (i * 2)));
        Self {
            words: (0..total_words).map(|_| AtomicI64::new(empty_word)).collect(),
            words_per_neuron,
        }
    }

    /// Export memory words for Metal GPU (read-only snapshot)
    fn export_for_metal(&self) -> Vec<i64> {
        self.words.iter().map(|w| w.load(Ordering::Relaxed)).collect()
    }

    #[inline]
    fn read(&self, neuron_idx: usize, address: usize) -> i64 {
        let word_idx = address / CELLS_PER_WORD;
        let cell_idx = address % CELLS_PER_WORD;
        let word_offset = neuron_idx * self.words_per_neuron + word_idx;
        let word = self.words[word_offset].load(Ordering::Relaxed);
        (word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
    }

    /// Thread-safe atomic write using compare-and-swap
    ///
    /// TRUE-wins-over-FALSE semantics:
    /// - TRUE can be written over EMPTY or FALSE
    /// - FALSE can only be written over EMPTY
    /// - TRUE cannot be overwritten by FALSE
    #[inline]
    fn write(&self, neuron_idx: usize, address: usize, value: i64, allow_override: bool) -> bool {
        let word_idx = address / CELLS_PER_WORD;
        let cell_idx = address % CELLS_PER_WORD;
        let word_offset = neuron_idx * self.words_per_neuron + word_idx;
        let shift = cell_idx * BITS_PER_CELL;
        let mask = CELL_MASK << shift;

        loop {
            let old_word = self.words[word_offset].load(Ordering::Relaxed);
            let old_cell = (old_word >> shift) & CELL_MASK;

            // No change needed if same value
            if old_cell == value {
                return false;
            }

            // TRUE wins over FALSE: don't overwrite TRUE with FALSE
            if old_cell == TRUE && value == FALSE {
                return false;
            }

            // If not allow_override:
            // - TRUE can overwrite EMPTY or FALSE (TRUE wins)
            // - FALSE can only overwrite EMPTY
            if !allow_override && value == FALSE && old_cell != EMPTY {
                return false;
            }

            let new_word = (old_word & !mask) | (value << shift);
            if self.words[word_offset]
                .compare_exchange_weak(old_word, new_word, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
            // CAS failed, retry
        }
    }
}

/// GPU-compatible sparse memory export (sorted arrays for binary search)
#[derive(Clone)]
pub struct SparseGpuExport {
    /// Sorted keys for all neurons, concatenated
    pub keys: Vec<u64>,
    /// Values corresponding to keys (0=FALSE, 1=TRUE)
    pub values: Vec<u8>,
    /// Start offset for each neuron in keys array
    pub offsets: Vec<u32>,
    /// Number of entries for each neuron
    pub counts: Vec<u32>,
    /// Total number of neurons
    pub num_neurons: usize,
}

impl SparseGpuExport {
    /// CPU binary search lookup
    #[inline]
    pub fn lookup(&self, neuron_idx: usize, address: u64) -> u8 {
        let start = self.offsets[neuron_idx] as usize;
        let count = self.counts[neuron_idx] as usize;

        if count == 0 {
            return EMPTY as u8;
        }

        let end = start + count;
        let keys_slice = &self.keys[start..end];

        match keys_slice.binary_search(&address) {
            Ok(idx) => self.values[start + idx],
            Err(_) => EMPTY as u8,
        }
    }

    /// Total memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4 + self.counts.len() * 4
    }
}

/// Sparse memory for a config group (concurrent hash-based, for bits > 12)
/// Uses DashMap for thread-safe concurrent access during parallel training.
pub(crate) struct GroupSparseMemory {
    /// Per-neuron concurrent hash maps: address -> cell value (0=FALSE, 1=TRUE, 2=EMPTY default)
    neurons: Vec<DashMap<u64, u8>>,
}

impl GroupSparseMemory {
    fn new(num_neurons: usize) -> Self {
        Self {
            neurons: (0..num_neurons).map(|_| DashMap::new()).collect(),
        }
    }

    /// Export to GPU-compatible sorted array format for binary search evaluation
    fn export_for_gpu(&self) -> SparseGpuExport {
        let mut keys: Vec<u64> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(self.neurons.len());
        let mut counts: Vec<u32> = Vec::with_capacity(self.neurons.len());

        for neuron_map in &self.neurons {
            let offset = keys.len() as u32;
            offsets.push(offset);

            // Collect and sort entries for this neuron
            let mut entries: Vec<(u64, u8)> = neuron_map.iter()
                .map(|entry| (*entry.key(), *entry.value()))
                .collect();
            entries.sort_by_key(|(k, _)| *k);

            counts.push(entries.len() as u32);

            for (key, value) in entries {
                keys.push(key);
                values.push(value);
            }
        }

        SparseGpuExport {
            keys,
            values,
            offsets,
            counts,
            num_neurons: self.neurons.len(),
        }
    }

    #[inline]
    fn read(&self, neuron_idx: usize, address: u64) -> u8 {
        *self.neurons[neuron_idx].get(&address).map(|v| *v).as_ref().unwrap_or(&2) // EMPTY
    }

    /// Thread-safe write using DashMap
    ///
    /// TRUE-wins-over-FALSE semantics (values: 0=FALSE, 1=TRUE, 2=EMPTY):
    /// - TRUE can be written over EMPTY or FALSE
    /// - FALSE can only be written over EMPTY
    /// - TRUE cannot be overwritten by FALSE
    #[inline]
    fn write(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool {
        let map = &self.neurons[neuron_idx];
        match map.entry(address) {
            dashmap::mapref::entry::Entry::Occupied(mut e) => {
                let current = *e.get();

                // No change needed if same value
                if current == value {
                    return false;
                }

                // TRUE wins over FALSE: don't overwrite TRUE with FALSE
                if current == 1 && value == 0 {
                    return false;
                }

                // If not allow_override:
                // - TRUE (1) can overwrite EMPTY (2) or FALSE (0) (TRUE wins)
                // - FALSE (0) can only overwrite EMPTY (2)
                if !allow_override && value == 0 && current != 2 {
                    return false;
                }

                // Allow TRUE to overwrite FALSE (TRUE wins) or write to EMPTY
                if allow_override || current == 2 || (value == 1 && current == 0) {
                    e.insert(value);
                    return true;
                }
                false
            }
            dashmap::mapref::entry::Entry::Vacant(e) => {
                e.insert(value);
                true
            }
        }
    }
}

/// Hybrid memory - Dense for low bits, Sparse for high bits
/// Both variants support thread-safe concurrent access for parallel training.
pub(crate) enum GroupMemory {
    Dense(GroupDenseMemory),
    Sparse(GroupSparseMemory),
}

impl GroupMemory {
    pub(crate) fn new(num_neurons: usize, bits: usize) -> Self {
        if bits <= SPARSE_THRESHOLD {
            GroupMemory::Dense(GroupDenseMemory::new(num_neurons, bits))
        } else {
            GroupMemory::Sparse(GroupSparseMemory::new(num_neurons))
        }
    }

    /// Check if this is dense memory (can be accelerated with Metal)
    pub(crate) fn is_dense(&self) -> bool {
        matches!(self, GroupMemory::Dense(_))
    }

    /// Export for Metal GPU (only works for Dense, returns None for Sparse)
    pub(crate) fn export_for_metal(&self) -> Option<Vec<i64>> {
        match self {
            GroupMemory::Dense(m) => Some(m.export_for_metal()),
            GroupMemory::Sparse(_) => None,
        }
    }

    /// Export sparse memory for GPU binary search (returns None for Dense)
    pub(crate) fn export_for_gpu_sparse(&self) -> Option<SparseGpuExport> {
        match self {
            GroupMemory::Dense(_) => None,
            GroupMemory::Sparse(m) => Some(m.export_for_gpu()),
        }
    }

    /// Check if this is sparse memory
    pub(crate) fn is_sparse(&self) -> bool {
        matches!(self, GroupMemory::Sparse(_))
    }

    #[inline]
    pub(crate) fn read(&self, neuron_idx: usize, address: usize) -> i64 {
        match self {
            GroupMemory::Dense(m) => m.read(neuron_idx, address),
            GroupMemory::Sparse(m) => m.read(neuron_idx, address as u64) as i64,
        }
    }

    /// Thread-safe write (both variants support concurrent access)
    #[inline]
    pub(crate) fn write(&self, neuron_idx: usize, address: usize, value: i64, allow_override: bool) -> bool {
        match self {
            GroupMemory::Dense(m) => m.write(neuron_idx, address, value, allow_override),
            GroupMemory::Sparse(m) => m.write(neuron_idx, address as u64, value as u8, allow_override),
        }
    }
}

/// Evaluate a dense config group using Metal GPU.
///
/// Returns scores for [num_examples × num_clusters_in_group] as f32.
/// The scores are in group-local cluster order (need scattering to global order).
pub(crate) fn evaluate_group_metal(
    metal: &crate::metal_ramlm::MetalRAMLMEvaluator,
    packed_eval: &[u64],
    connections_flat: &[i64],
    memory_words: &[i64],
    group: &ConfigGroup,
    num_eval: usize,
    words_per_example: usize,
    memory_mode: u8,
) -> Result<Vec<f32>, String> {
    let num_clusters = group.cluster_count();
    let num_neurons = group.total_neurons();

    // Extract connections for this group (they're stored contiguously at conn_offset)
    let conn_size = group.conn_size();
    let group_connections = &connections_flat[group.conn_offset..group.conn_offset + conn_size];

    metal.forward_batch(
        packed_eval,
        group_connections,
        memory_words,
        num_eval,
        words_per_example,
        num_neurons,
        group.bits,
        group.neurons,
        num_clusters,
        group.words_per_neuron,
        memory_mode,
    )
}

/// Evaluate a sparse config group using Metal GPU with binary search.
///
/// Returns scores for [num_examples × num_clusters_in_group] as f32.
/// The scores are in group-local cluster order (need scattering to global order).
pub(crate) fn evaluate_group_sparse_gpu(
    sparse_evaluator: &crate::metal_ramlm::MetalSparseEvaluator,
    packed_eval: &[u64],
    connections_flat: &[i64],
    export: &SparseGpuExport,
    group: &ConfigGroup,
    num_eval: usize,
    words_per_example: usize,
    memory_mode: u8,
) -> Result<Vec<f32>, String> {
    let num_clusters = group.cluster_count();

    // Extract connections for this group
    let conn_size = group.conn_size();
    let group_connections = &connections_flat[group.conn_offset..group.conn_offset + conn_size];

    sparse_evaluator.forward_batch_sparse(
        packed_eval,
        group_connections,
        &export.keys,
        &export.values,
        &export.offsets,
        &export.counts,
        num_eval,
        words_per_example,
        export.num_neurons,
        group.bits,
        group.neurons,
        num_clusters,
        memory_mode,
    )
}

/// Evaluate multiple genomes SEQUENTIALLY with METAL GPU ACCELERATION.
///
/// This is the KEY acceleration function for GA optimization.
/// Each genome is evaluated independently with its own memory.
///
/// Hybrid acceleration strategy:
/// - Training: CPU with rayon parallelism (random writes)
/// - Evaluation: Metal GPU for dense groups (40 cores on M4 Max)
///               CPU for sparse groups (hash lookups not GPU-friendly)
///
/// Performance: Metal accelerates evaluation by ~10-20x for dense groups,
/// which typically contain 80%+ of clusters (those with bits <= 12).
///
/// Args:
///   genomes_bits_flat: [num_genomes * num_clusters] bits per cluster for each genome
///   genomes_neurons_flat: [num_genomes * num_clusters] neurons per cluster for each genome
///   genomes_connections_flat: [num_genomes * total_connections] flattened connection indices, or empty for random
///   num_genomes: Number of genomes to evaluate
///   num_clusters: Number of clusters (vocab size)
///   train_input_bits: [num_train * total_input_bits] training contexts
///   train_targets: [num_train] target cluster for each training example
///   train_negatives: [num_train * num_negatives] negative clusters
///   num_train: Number of training examples
///   num_negatives: Number of negative samples per example
///   eval_input_bits: [num_eval * total_input_bits] evaluation contexts
///   eval_targets: [num_eval] target cluster for each eval example
///   num_eval: Number of evaluation examples
///   total_input_bits: Input bits per example
///   empty_value: Value for EMPTY cells (0.0 recommended)
///
/// Returns: [num_genomes] cross-entropy values (lower is better)
/// Evaluate multiple genomes SEQUENTIALLY, returning (CE, accuracy) for each.
///
/// Genomes are evaluated one at a time to:
/// 1. Prevent memory explosion (only 1 genome's memory allocated)
/// 2. Allow full CPU utilization for each genome's training/eval (16 cores)
/// 3. Avoid thread pool contention from nested parallelism
///
/// Each genome's training (200K examples) and evaluation (50K examples)
/// use full rayon parallelism internally.
///
/// IMPORTANT: Connections must be provided for proper evolutionary search.
/// If genomes_connections_flat is empty, random connections are generated.
///
/// Returns Vec of (cross_entropy, accuracy) tuples - one per genome.
pub fn evaluate_genomes_parallel(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> Vec<(f64, f64)> {
    use rand::prelude::*;
    use rand::SeedableRng;

    // Check if connections are provided
    let use_provided_connections = !genomes_connections_flat.is_empty();

    // Pre-compute genome_bpn_offsets: genomes_bits_flat has total_neurons entries per genome
    // (per-neuron bits), NOT num_clusters entries.
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    debug_assert_eq!(
        genomes_bits_flat.len(),
        *genome_bpn_offsets.last().unwrap(),
        "genomes_bits_flat length ({}) != expected total neurons ({})",
        genomes_bits_flat.len(),
        genome_bpn_offsets.last().unwrap(),
    );

    // Pre-compute per-genome connection offsets: conn_size = sum of per-neuron bits
    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    // Pack input bits to u64 once (shared across all genomes for GPU address computation)
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_bools_to_u64(train_input_bits, num_train, total_input_bits);

    // Check if progress logging is enabled via env var
    let progress_log = std::env::var("WNN_PROGRESS_LOG").map(|v| v == "1").unwrap_or(false);
    let log_path = std::env::var("WNN_LOG_PATH").ok();
    // Get generation info from env vars (set by Python before calling)
    let current_gen: usize = std::env::var("WNN_PROGRESS_GEN")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let total_gens: usize = std::env::var("WNN_PROGRESS_TOTAL_GENS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    // Log type: Init, New, Nbr, CE, Acc (default Init)
    let log_type = std::env::var("WNN_PROGRESS_TYPE").unwrap_or_else(|_| "Init".to_string());
    // Offset for batch position (e.g., batch starting at genome 11 in a 50-genome set)
    let batch_offset: usize = std::env::var("WNN_PROGRESS_OFFSET")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    // Total count (for showing X/50 instead of X/batch_size)
    let total_count: usize = std::env::var("WNN_PROGRESS_TOTAL")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(num_genomes);
    let _start_time = std::time::Instant::now();

    // SEQUENTIAL genome evaluation - each genome gets full thread pool for token parallelism
    // Parallel genome eval causes contention: 10 genomes × nested token parallelism = thrashing
    // Sequential is faster: ~6s/genome vs ~10s/genome with parallel outer loop
    let results: Vec<(f64, f64)> = (0..num_genomes).map(|genome_idx| {
        let genome_start = std::time::Instant::now();
        // Extract this genome's per-neuron bits and per-cluster neurons
        let genome_offset = genome_idx * num_clusters;
        let neurons_per_cluster: Vec<usize> = genomes_neurons_flat[genome_offset..genome_offset + num_clusters].to_vec();

        // Extract per-neuron bits for this genome
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let per_neuron_bits: Vec<usize> = genomes_bits_flat[bpn_start..bpn_end].to_vec();

        // Compute per-cluster max bits (for build_groups and GPU dispatch)
        let bits_per_cluster = per_cluster_max_bits(&per_neuron_bits, &neurons_per_cluster);

        // Build neuron metadata for per-neuron training and CPU eval
        let (cluster_neuron_starts, neuron_conn_offsets) =
            build_neuron_metadata(&per_neuron_bits, &neurons_per_cluster);

        // Build config groups for this genome (using per-cluster max bits)
        let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);

        // Create hybrid memory for each config group
        let group_memories: Vec<GroupMemory> = groups.iter()
            .map(|g| GroupMemory::new(g.total_neurons(), g.bits))
            .collect();

        // Get original per-neuron connections for this genome
        let original_connections: Vec<i64> = if use_provided_connections {
            let conn_offset = conn_offsets[genome_idx];
            let conn_size = conn_sizes[genome_idx];
            genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec()
        } else {
            // Generate random per-neuron connections (legacy fallback)
            let total_conn: usize = per_neuron_bits.iter().sum();
            let mut rng = rand::rngs::SmallRng::from_entropy();
            let mut conns: Vec<i64> = Vec::with_capacity(total_conn);
            for _ in 0..total_conn {
                conns.push(rng.gen_range(0..total_input_bits as i64));
            }
            conns
        };

        // Build GPU-padded connections (per-neuron → group layout with padding)
        let gpu_connections = reorganize_connections_for_gpu(
            &original_connections,
            &per_neuron_bits,
            &neurons_per_cluster,
            &groups,
        );

        // Build cluster-to-group mapping
        let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
        for (group_idx, group) in groups.iter().enumerate() {
            for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                cluster_to_group[cluster_id] = (group_idx, local_idx);
            }
        }

        // Compute training addresses on GPU (falls back to CPU if unavailable)
        let gpu_addresses = try_gpu_addresses_adaptive(
            &packed_train_input,
            words_per_example,
            &per_neuron_bits,
            &neuron_conn_offsets,
            &original_connections,
            num_train,
        );

        // Train this genome using per-neuron bits (PARALLEL across examples)
        train_genome_in_slot(
            &group_memories,
            &groups,
            &original_connections,
            &per_neuron_bits,
            &cluster_neuron_starts,
            &neuron_conn_offsets,
            &cluster_to_group,
            train_input_bits,
            train_targets,
            train_negatives,
            num_train,
            num_negatives,
            total_input_bits,
            gpu_addresses.as_deref(),
        );

        // Evaluate this genome - HYBRID Metal/CPU acceleration
        // - Dense groups (bits <= 12): Metal GPU (all examples at once)
        // - Sparse groups (bits > 12): CPU (hash lookups not GPU-friendly)
        let epsilon = 1e-10f64;

        // Pre-compute scores for all examples × clusters
        // Shape: [num_eval][num_clusters]
        let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];

        // Get Metal evaluators (lazy init, thread-safe)
        // These are Arc<T> so we can clone and hold references across the loop
        let metal = get_metal_evaluator();
        let sparse_metal = get_sparse_metal_evaluator();

        // Pack eval input bits to u64 for GPU (pack once, reuse for all groups)
        let (packed_eval, words_per_example) = crate::neuron_memory::pack_bools_to_u64(
            eval_input_bits, num_eval, total_input_bits
        );

        // Process each group - Metal for dense, GPU sparse for sparse, CPU fallback
        for (group_idx, group) in groups.iter().enumerate() {
            let memory = &group_memories[group_idx];

            if let (Some(ref metal_eval), true) = (&metal, memory.is_dense()) {
                // Metal path: evaluate all examples at once for this dense group
                // GPU uses padded connections (group layout with max_bits per neuron)
                if let Some(memory_words) = memory.export_for_metal() {
                    match evaluate_group_metal(
                        metal_eval.as_ref(),
                        &packed_eval,
                        &gpu_connections,
                        &memory_words,
                        group,
                        num_eval,
                        words_per_example,
                        crate::neuron_memory::MODE_TERNARY,
                    ) {
                        Ok(group_scores) => {
                            for ex_idx in 0..num_eval {
                                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                    let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                    all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                                }
                            }
                            continue;
                        }
                        Err(_) => {}
                    }
                }
            }

            // GPU sparse path: evaluate sparse groups using binary search on GPU
            if let (Some(ref sparse_eval), true) = (&sparse_metal, memory.is_sparse()) {
                if let Some(export) = memory.export_for_gpu_sparse() {
                    match evaluate_group_sparse_gpu(
                        sparse_eval.as_ref(),
                        &packed_eval,
                        &gpu_connections,
                        &export,
                        group,
                        num_eval,
                        words_per_example,
                        crate::neuron_memory::MODE_TERNARY,
                    ) {
                        Ok(group_scores) => {
                            for ex_idx in 0..num_eval {
                                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                    let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                    all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                                }
                            }
                            continue;
                        }
                        Err(_) => {}
                    }
                }
            }

            // CPU path: evaluate examples in parallel using per-neuron bits
            all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                let input_start = ex_idx * total_input_bits;
                let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                    let actual_neurons = if let Some(ref an) = group.actual_neurons {
                        an[local_cluster] as usize
                    } else {
                        group.neurons
                    };

                    let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

                    let mut sum = 0.0f32;
                    for n in 0..actual_neurons {
                        let global_n = cluster_neuron_starts[cluster_id] + n;
                        let n_bits = per_neuron_bits[global_n];
                        let conn_start = neuron_conn_offsets[global_n];
                        let address = compute_address(input_bits, &original_connections[conn_start..], n_bits);
                        let cell = memory.read(neuron_base + n, address);
                        sum += match cell {
                            FALSE => 0.0,
                            TRUE => 1.0,
                            _ => empty_value,
                        };
                    }

                    scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                }
            });
        }

        // Compute CE and accuracy from pre-computed scores (parallel across examples)
        let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
            let target_idx = eval_targets[ex_idx] as usize;

            // Find prediction (argmax) for accuracy
            let predicted = scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let correct: u64 = if predicted == target_idx { 1 } else { 0 };

            // Softmax and cross-entropy for this example
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            let target_prob = exp_scores[target_idx] / sum_exp;
            let ce = -(target_prob + epsilon).ln();

            (ce, correct)
        }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

        let avg_ce = total_ce / num_eval as f64;
        let accuracy = total_correct as f64 / num_eval as f64;

        // Progress logging (to log file if WNN_LOG_PATH set, otherwise stderr)
        // Format matches Python's format_genome_log for consistency
        if progress_log {
            use std::io::Write;
            let genome_elapsed = genome_start.elapsed().as_secs_f64();
            let now = chrono::Local::now();

            // Calculate overall position using batch offset
            let overall_position = batch_offset + genome_idx + 1;

            // Calculate padding widths based on totals
            let gen_width = total_gens.to_string().len();
            let pos_width = total_count.to_string().len();

            // Pad type indicator to 4 chars (Init, New , Nbr , CE  , Acc )
            let type_padded = format!("{:<4}", &log_type[..log_type.len().min(4)]);

            // Format: [Gen 001/100] Genome 01/50 (Init): CE=10.6588, Acc=0.0100% (8.7s)
            let msg = format!(
                "{} | [Gen {:0gen_width$}/{:0gen_width$}] Genome {:0pos_width$}/{} ({}): CE={:.4}, Acc={:.4}% ({:.1}s)\n",
                now.format("%H:%M:%S"),
                current_gen, total_gens,
                overall_position, total_count,
                type_padded,
                avg_ce, accuracy * 100.0,
                genome_elapsed,
                gen_width = gen_width,
                pos_width = pos_width,
            );
            if let Some(ref path) = log_path {
                use fs2::FileExt;
                if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
                    // Lock file to prevent interleaved writes with Python
                    if file.lock_exclusive().is_ok() {
                        let _ = file.write_all(msg.as_bytes());
                        let _ = file.flush();
                        let _ = file.unlock();
                    }
                }
            } else {
                eprint!("{}", msg);
            }
        }

        (avg_ce, accuracy)
    }).collect();

    results
}

/// Evaluate genomes with multi-subset rotation support.
///
/// All token subsets are pre-encoded and passed at once. The train_subset_idx
/// and eval_subset_idx select which subset to use for this batch of evaluations.
///
/// This enables per-generation/iteration rotation of data subsets, acting as
/// a regularizer that forces genomes to generalize across all subsets.
///
/// Args:
///   genomes_bits_flat: [num_genomes * num_clusters] bits per cluster
///   genomes_neurons_flat: [num_genomes * num_clusters] neurons per cluster
///   genomes_connections_flat: Flattened connections (empty = random)
///   num_genomes: Number of genomes to evaluate
///   num_clusters: Vocabulary size
///   train_subsets_flat: [sum(num_train_per_subset) * total_input_bits] all train input bits concatenated
///   train_targets_flat: [sum(num_train_per_subset)] all train targets concatenated
///   train_negatives_flat: [sum(num_train_per_subset) * num_negatives] all train negatives concatenated
///   train_subset_counts: [num_subsets] number of examples in each train subset
///   eval_subsets_flat: [sum(num_eval_per_subset) * total_input_bits] all eval input bits concatenated
///   eval_targets_flat: [sum(num_eval_per_subset)] all eval targets concatenated
///   eval_subset_counts: [num_subsets] number of examples in each eval subset
///   train_subset_idx: Which train subset to use (0-indexed)
///   eval_subset_idx: Which eval subset to use (0-indexed)
///   num_negatives: Number of negative samples per example
///   total_input_bits: Input bits per example
///   empty_value: Value for EMPTY cells (0.0 recommended)
///
/// Returns: Vec of (cross_entropy, accuracy) tuples - one per genome
pub fn evaluate_genomes_parallel_multisubset(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    // Train data - all subsets concatenated
    train_subsets_flat: &[bool],
    train_targets_flat: &[i64],
    train_negatives_flat: &[i64],
    train_subset_counts: &[usize],  // [num_subsets] examples per subset
    // Eval data - all subsets concatenated
    eval_subsets_flat: &[bool],
    eval_targets_flat: &[i64],
    eval_subset_counts: &[usize],  // [num_subsets] examples per subset
    // Subset selection
    train_subset_idx: usize,
    eval_subset_idx: usize,
    // Other params
    num_negatives: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> Vec<(f64, f64)> {
    // Compute offsets for train subsets
    let mut train_offsets: Vec<usize> = Vec::with_capacity(train_subset_counts.len() + 1);
    let mut train_target_offsets: Vec<usize> = Vec::with_capacity(train_subset_counts.len() + 1);
    train_offsets.push(0);
    train_target_offsets.push(0);
    for &count in train_subset_counts {
        let last_offset = *train_offsets.last().unwrap();
        train_offsets.push(last_offset + count * total_input_bits);
        let last_target_offset = *train_target_offsets.last().unwrap();
        train_target_offsets.push(last_target_offset + count);
    }

    // Compute offsets for eval subsets
    let mut eval_offsets: Vec<usize> = Vec::with_capacity(eval_subset_counts.len() + 1);
    let mut eval_target_offsets: Vec<usize> = Vec::with_capacity(eval_subset_counts.len() + 1);
    eval_offsets.push(0);
    eval_target_offsets.push(0);
    for &count in eval_subset_counts {
        let last_offset = *eval_offsets.last().unwrap();
        eval_offsets.push(last_offset + count * total_input_bits);
        let last_target_offset = *eval_target_offsets.last().unwrap();
        eval_target_offsets.push(last_target_offset + count);
    }

    // Extract the selected train subset
    let train_start = train_offsets[train_subset_idx];
    let train_end = train_offsets[train_subset_idx + 1];
    let train_input_bits = &train_subsets_flat[train_start..train_end];

    let train_target_start = train_target_offsets[train_subset_idx];
    let train_target_end = train_target_offsets[train_subset_idx + 1];
    let train_targets = &train_targets_flat[train_target_start..train_target_end];

    let num_train = train_subset_counts[train_subset_idx];
    let train_neg_start = train_target_start * num_negatives;
    let train_neg_end = train_target_end * num_negatives;
    let train_negatives = &train_negatives_flat[train_neg_start..train_neg_end];

    // Extract the selected eval subset
    let eval_start = eval_offsets[eval_subset_idx];
    let eval_end = eval_offsets[eval_subset_idx + 1];
    let eval_input_bits = &eval_subsets_flat[eval_start..eval_end];

    let eval_target_start = eval_target_offsets[eval_subset_idx];
    let eval_target_end = eval_target_offsets[eval_subset_idx + 1];
    let eval_targets = &eval_targets_flat[eval_target_start..eval_target_end];

    let num_eval = eval_subset_counts[eval_subset_idx];

    // Now delegate to the existing single-subset function
    evaluate_genomes_parallel(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        num_clusters,
        train_input_bits,
        train_targets,
        train_negatives,
        num_train,
        num_negatives,
        eval_input_bits,
        eval_targets,
        num_eval,
        total_input_bits,
        empty_value,
    )
}

// =============================================================================
// PARALLEL HYBRID CPU+GPU EVALUATION
// =============================================================================

/// Export data for a single genome (used in batched GPU evaluation)
#[derive(Clone)]
pub struct GenomeExport {
    /// Connections for all groups, flattened
    pub connections: Vec<i64>,
    /// For each group: (is_sparse, group_idx, cluster_ids)
    pub group_info: Vec<(bool, usize, Vec<usize>)>,
    /// Dense group exports: memory words
    pub dense_exports: Vec<Vec<i64>>,
    /// Sparse group exports: sorted arrays for binary search
    pub sparse_exports: Vec<SparseGpuExport>,
    /// Config groups for this genome
    pub groups: Vec<ConfigGroup>,
}

/// Memory pool for reusable genome memory
struct GenomeMemoryPool {
    /// Memory sets for each pool slot
    memories: Vec<Vec<GroupMemory>>,
    /// Config groups template (same for all genomes with same config)
    groups_template: Vec<ConfigGroup>,
}

impl GenomeMemoryPool {
    /// Create a pool with the given number of slots
    fn new(
        pool_size: usize,
        bits_per_cluster: &[usize],
        neurons_per_cluster: &[usize],
    ) -> Self {
        let groups_template = build_groups(bits_per_cluster, neurons_per_cluster);

        let memories = (0..pool_size)
            .map(|_| {
                groups_template.iter()
                    .map(|g| GroupMemory::new(g.total_neurons(), g.bits))
                    .collect()
            })
            .collect();

        Self {
            memories,
            groups_template,
        }
    }

    /// Reset all memory in a pool slot (clear for reuse)
    fn reset_slot(&self, slot: usize) {
        for memory in &self.memories[slot] {
            match memory {
                GroupMemory::Dense(m) => {
                    // Reset dense memory to EMPTY
                    for word in &m.words {
                        word.store(0x5555555555555555i64, std::sync::atomic::Ordering::Relaxed); // All EMPTY
                    }
                }
                GroupMemory::Sparse(m) => {
                    // Clear sparse memory
                    for neuron_map in &m.neurons {
                        neuron_map.clear();
                    }
                }
            }
        }
    }
}

/// Calculate optimal pool and batch sizes based on memory budget
fn calculate_pool_size(
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
    _num_clusters: usize,
    budget_gb: f64,
    cpu_cores: usize,
) -> (usize, usize) {
    // Estimate memory per genome (use same grouping strategy as actual training)
    let groups = build_groups(bits_per_cluster, neurons_per_cluster);
    let mut bytes_per_genome = 0usize;

    for group in &groups {
        if group.bits <= SPARSE_THRESHOLD {
            // Dense: 2 bits per cell, 2^bits cells per neuron
            let cells_per_neuron = 1 << group.bits;
            let words_per_neuron = (cells_per_neuron + 30) / 31; // 31 cells per word
            bytes_per_genome += group.total_neurons() * words_per_neuron * 8;
        } else {
            // Sparse: Based on measured data from actual training
            // With 100K training examples + 5 negatives = 600K writes, but many collide
            // Measured: ~1.2K unique entries per neuron on average (8.9M / 7500 neurons)
            // Use 3K as conservative estimate to leave headroom
            // Memory per entry: key(8) + value(1) + DashMap overhead (~24 bytes)
            bytes_per_genome += group.total_neurons() * 3_000 * 32;
        }
    }

    let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let max_pool_size = (budget_bytes / bytes_per_genome).max(1);

    // Pool size = min(max_pool, cpu_cores) to avoid over-allocation
    // Use WNN_BATCH_SIZE env var to override for testing
    let pool_size = max_pool_size.min(cpu_cores).max(1);

    // Batch size = pool size (process one batch at a time)
    let batch_size = pool_size;

    (pool_size, batch_size)
}

/// Get available memory in GB (macOS specific)
fn get_available_memory_gb() -> f64 {
    // Try to read from sysctl
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = mem_str.trim().parse::<u64>() {
                    return bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                }
            }
        }
    }
    // Fallback: assume 64GB (M4 Max typical)
    64.0
}

/// Train a genome using the given memory slot.
/// When `gpu_addresses` is Some, uses pre-computed GPU addresses instead of CPU compute_address().
/// GPU address layout: addresses[global_neuron_idx * num_train + example_idx].
pub(crate) fn train_genome_in_slot(
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    original_connections: &[i64],    // Per-neuron layout (NOT group layout)
    per_neuron_bits: &[usize],       // Bits per neuron
    cluster_neuron_starts: &[usize], // First neuron idx per cluster
    neuron_conn_offsets: &[usize],   // Conn offset per neuron
    cluster_to_group: &[(usize, usize)],
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    total_input_bits: usize,
    gpu_addresses: Option<&[u32]>,
) {
    // Use chunked parallel processing to balance parallelism vs overhead
    let chunk_size = 10_000.max(num_train / 20);

    (0..num_train).into_par_iter()
        .with_min_len(chunk_size)
        .for_each(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &train_input_bits[input_start..input_start + total_input_bits];

        let true_cluster = train_targets[ex_idx] as usize;

        // Train positive example
        {
            let (group_idx, local_cluster) = cluster_to_group[true_cluster];
            let group = &groups[group_idx];
            let memory = &memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

            for n in 0..actual_neurons {
                let global_n = cluster_neuron_starts[true_cluster] + n;
                let address = if let Some(addrs) = gpu_addresses {
                    addrs[global_n * num_train + ex_idx] as usize
                } else {
                    let n_bits = per_neuron_bits[global_n];
                    let conn_start = neuron_conn_offsets[global_n];
                    compute_address(input_bits, &original_connections[conn_start..], n_bits)
                };
                memory.write(neuron_base + n, address, TRUE, false);
            }
        }

        // Train negative examples
        let neg_start = ex_idx * num_negatives;
        for k in 0..num_negatives {
            let false_cluster = train_negatives[neg_start + k] as usize;
            if false_cluster == true_cluster {
                continue;
            }

            let (group_idx, local_cluster) = cluster_to_group[false_cluster];
            let group = &groups[group_idx];
            let memory = &memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

            for n in 0..actual_neurons {
                let global_n = cluster_neuron_starts[false_cluster] + n;
                let address = if let Some(addrs) = gpu_addresses {
                    addrs[global_n * num_train + ex_idx] as usize
                } else {
                    let n_bits = per_neuron_bits[global_n];
                    let conn_start = neuron_conn_offsets[global_n];
                    compute_address(input_bits, &original_connections[conn_start..], n_bits)
                };
                memory.write(neuron_base + n, address, FALSE, false);
            }
        }
    });
}

/// Export trained memory to GPU-compatible format
fn export_genome_for_gpu(
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    connections_flat: &[i64],
) -> GenomeExport {
    let mut dense_exports = Vec::new();
    let mut sparse_exports = Vec::new();
    let mut group_info = Vec::new();

    for (group_idx, (group, memory)) in groups.iter().zip(memories.iter()).enumerate() {
        let is_sparse = memory.is_sparse();
        group_info.push((is_sparse, group_idx, group.cluster_ids.clone()));

        if is_sparse {
            if let Some(export) = memory.export_for_gpu_sparse() {
                sparse_exports.push(export);
            } else {
                // Fallback: empty export
                sparse_exports.push(SparseGpuExport {
                    keys: vec![],
                    values: vec![],
                    offsets: vec![0; group.total_neurons()],
                    counts: vec![0; group.total_neurons()],
                    num_neurons: group.total_neurons(),
                });
            }
        } else {
            if let Some(words) = memory.export_for_metal() {
                dense_exports.push(words);
            } else {
                dense_exports.push(vec![]);
            }
        }
    }

    GenomeExport {
        connections: connections_flat.to_vec(),
        group_info,
        dense_exports,
        sparse_exports,
        groups: groups.to_vec(),
    }
}

// Thread-local cache for GPU buffers to avoid expensive 10GB buffer allocation per evaluation
// The scores buffer is ~10GB (50K examples × 50K clusters × 4 bytes), so reusing it is critical.
// The cache includes the reset generation to invalidate on Metal reset.
#[cfg(target_os = "macos")]
thread_local! {
    // (reset_gen, num_eval, num_clusters, buffer)
    static CACHED_SCORES_BUFFER: std::cell::RefCell<Option<(u64, usize, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
    // (reset_gen, size, buffer)
    static CACHED_INPUT_BUFFER: std::cell::RefCell<Option<(u64, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
}

/// Evaluate a genome export using CPU+GPU hybrid
/// Returns (cross_entropy, accuracy)
pub fn evaluate_genome_hybrid(
    export: &GenomeExport,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    empty_value: f32,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&crate::metal_ramlm::MetalSparseEvaluator>,
) -> (f64, f64) {
    let epsilon = 1e-10f64;

    // Detailed timing (enabled via WNN_GROUP_TIMING env var)
    let timing_enabled = std::env::var("WNN_GROUP_TIMING").is_ok();
    let eval_start = std::time::Instant::now();
    let mut gpu_time_ms = 0u128;
    let mut cpu_time_ms = 0u128;
    let mut scatter_time_ms = 0u128;
    let mut gpu_calls = 0usize;
    let mut cpu_calls = 0usize;

    // Pack eval input bits to u64 for GPU (pack once, reuse for all GPU paths)
    let (packed_eval, words_per_example) = crate::neuron_memory::pack_bools_to_u64(
        eval_input_bits, num_eval, total_input_bits
    );

    // FAST PATH: If single sparse group covering all clusters, use direct CE computation
    // This avoids the 10GB GPU→CPU transfer by computing CE on GPU
    #[cfg(target_os = "macos")]
    if export.group_info.len() == 1 && export.sparse_exports.len() == 1 {
        let (is_sparse, group_idx, cluster_ids) = &export.group_info[0];
        if *is_sparse && cluster_ids.len() == num_clusters {
            // Check if clusters are contiguous 0..num_clusters (identity mapping)
            let is_contiguous = cluster_ids.iter().enumerate().all(|(i, &c)| c == i);
            if is_contiguous {
                // Use MetalSparseCEEvaluator for direct CE computation
                static CE_EVALUATOR: std::sync::OnceLock<Option<crate::metal_ramlm::MetalSparseCEEvaluator>> = std::sync::OnceLock::new();
                let ce_eval = CE_EVALUATOR.get_or_init(|| {
                    crate::metal_ramlm::MetalSparseCEEvaluator::new().ok()
                });

                if let Some(evaluator) = ce_eval {
                    let group = &export.groups[*group_idx];
                    let sparse_export = &export.sparse_exports[0];

                    let call_start = std::time::Instant::now();
                    let result = evaluator.compute_ce(
                        &packed_eval,
                        &export.connections,
                        &sparse_export.keys,
                        &sparse_export.values,
                        &sparse_export.offsets,
                        &sparse_export.counts,
                        eval_targets,
                        num_eval,
                        words_per_example,
                        group.neurons * num_clusters,  // total neurons
                        group.bits,
                        group.neurons,
                        num_clusters,
                        empty_value,
                    );

                    if let Ok((ce, acc)) = result {
                        if timing_enabled {
                            let elapsed = call_start.elapsed().as_millis();
                            let total_ms = eval_start.elapsed().as_millis();
                            eprintln!(
                                "[EVAL_HYBRID] FAST_PATH total={}ms gpu_ce={}ms (no scatter!)",
                                total_ms, elapsed
                            );
                        }
                        return (ce, acc);
                    }
                    // Fall through to standard path if CE evaluator fails
                }
            }
        }
    }

    // FULL GPU PATH: Write scores directly to shared GPU buffer, compute CE on GPU
    // This avoids the GPU→CPU→GPU round-trip that was slowing down tiered evaluation
    // Full GPU path enabled by default (3x faster). Disable with WNN_GPU_CE=0
    #[cfg(target_os = "macos")]
    {
    let use_full_gpu = std::env::var("WNN_GPU_CE").map(|v| v != "0").unwrap_or(true);

    if use_full_gpu {
        if let Some(group_eval) = get_group_evaluator() {
            let gpu_start = std::time::Instant::now();

            // Get current reset generation to detect when Metal evaluators were reset
            let current_reset_gen = RESET_GENERATION.load(Ordering::SeqCst);

            // Phase timing for detailed analysis (enabled via WNN_PHASE_TIMING env var)
            let phase_timing = std::env::var("WNN_PHASE_TIMING").is_ok();
            let phase_start = std::time::Instant::now();

            // Get or create cached scores buffer (avoids expensive 10GB allocation per eval)
            // The scores buffer is zeroed efficiently using memset instead of creating a new Vec
            // Invalidate cache if Metal was reset (reset_gen changed)
            let scores_buffer = CACHED_SCORES_BUFFER.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some((cached_gen, cached_eval, cached_clusters, ref buffer)) = *cache {
                    if cached_gen == current_reset_gen && cached_eval == num_eval && cached_clusters == num_clusters {
                        // Reuse existing buffer - just zero it
                        group_eval.zero_scores_buffer(buffer, num_eval, num_clusters);
                        return buffer.clone();
                    }
                }
                // Create new buffer and cache it (invalidate on reset_gen mismatch)
                let buffer = group_eval.create_scores_buffer(num_eval, num_clusters);
                *cache = Some((current_reset_gen, num_eval, num_clusters, buffer.clone()));
                buffer
            });

            let zero_time_ms = if phase_timing { phase_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };
            let phase_start = std::time::Instant::now();

            // Get or create cached input buffer (update contents efficiently)
            // Invalidate cache if Metal was reset (reset_gen changed)
            // Uses packed u64 input for GPU
            let input_buffer = CACHED_INPUT_BUFFER.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some((cached_gen, cached_size, ref buffer)) = *cache {
                    if cached_gen == current_reset_gen && cached_size == packed_eval.len() {
                        // Reuse existing buffer - update contents
                        group_eval.update_input_buffer(buffer, &packed_eval);
                        return buffer.clone();
                    }
                }
                // Create new buffer and cache it (invalidate on reset_gen mismatch)
                let buffer = group_eval.create_input_buffer(&packed_eval);
                *cache = Some((current_reset_gen, packed_eval.len(), buffer.clone()));
                buffer
            });

            let input_time_ms = if phase_timing { phase_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };
            let _phase_start = std::time::Instant::now();

            let mut dense_idx: usize;
            let mut sparse_idx = 0usize;
            let all_groups_success = true;
            let mut sparse_time_ms = 0.0f64;
            let mut dense_time_ms = 0.0f64;
            let sparse_call_count;

            // Collect all sparse groups for batched evaluation (single command buffer)
            // This eliminates ~0.5ms overhead per group from separate commit+wait cycles
            let mut sparse_groups: Vec<crate::metal_ramlm::SparseGroupData> = Vec::new();
            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                if *is_sparse {
                    let group = &export.groups[*group_idx];
                    let sparse_export = &export.sparse_exports[sparse_idx];
                    sparse_idx += 1;

                    sparse_groups.push(crate::metal_ramlm::SparseGroupData {
                        connections: &export.connections[group.conn_offset..group.conn_offset + group.conn_size()],
                        keys: &sparse_export.keys,
                        values: &sparse_export.values,
                        offsets: &sparse_export.offsets,
                        counts: &sparse_export.counts,
                        cluster_ids,
                        bits_per_neuron: group.bits,
                        neurons_per_cluster: group.neurons,
                        actual_neurons_per_cluster: group.actual_neurons.as_deref(),
                    });
                }
            }

            // Evaluate all sparse groups in a single batched call
            sparse_call_count = sparse_groups.len();
            if !sparse_groups.is_empty() {
                let sparse_start = std::time::Instant::now();
                group_eval.eval_sparse_groups_batched(
                    &input_buffer,
                    &scores_buffer,
                    &sparse_groups,
                    num_eval,
                    words_per_example,
                    num_clusters,
                    empty_value,
                    crate::neuron_memory::MODE_TERNARY
                );
                if phase_timing {
                    sparse_time_ms = sparse_start.elapsed().as_micros() as f64 / 1000.0;
                }
            }

            // Evaluate dense groups individually (uses DENSE_BUFFER_CACHE)
            dense_idx = 0;
            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                if !*is_sparse {
                    let group = &export.groups[*group_idx];
                    let dense_words = &export.dense_exports[dense_idx];
                    dense_idx += 1;

                    let dense_start = std::time::Instant::now();
                    group_eval.eval_dense_to_buffer(
                        &input_buffer,
                        &scores_buffer,
                        &export.connections[group.conn_offset..group.conn_offset + group.conn_size()],
                        dense_words,
                        cluster_ids,
                        num_eval,
                        words_per_example,
                        group.bits,
                        group.neurons,
                        num_clusters,
                        group.words_per_neuron,
                        empty_value,
                        crate::neuron_memory::MODE_TERNARY
                    );

                    if phase_timing {
                        dense_time_ms += dense_start.elapsed().as_micros() as f64 / 1000.0;
                    }
                }
            }

            let ce_start = std::time::Instant::now();

            if all_groups_success {
                // Compute CE directly from GPU buffer
                let result = group_eval.compute_ce_from_buffer(
                    &scores_buffer,
                    eval_targets,
                    num_eval,
                    num_clusters,
                );

                let ce_time_ms = if phase_timing { ce_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };

                if let Ok((ce, acc)) = result {
                    if timing_enabled {
                        let elapsed = gpu_start.elapsed().as_millis();
                        if phase_timing {
                            eprintln!(
                                "[EVAL_PHASE] zero={:.1}ms input={:.1}ms sparse={:.1}ms({} calls) dense={:.1}ms ce={:.1}ms total={}ms",
                                zero_time_ms, input_time_ms, sparse_time_ms, sparse_call_count, dense_time_ms, ce_time_ms, elapsed
                            );
                        } else {
                            eprintln!(
                                "[EVAL_HYBRID] FULL_GPU_PATH total={}ms (no CPU scatter!)",
                                elapsed
                            );
                        }
                    }
                    return (ce, acc);
                }
            }
            // Fall through to CPU path if full GPU fails
        }
    }
    } // cfg(target_os = "macos")

    // Legacy GPU CE PATH: Disabled - kept for reference, macOS only
    #[cfg(target_os = "macos")]
    {
        static CE_REDUCE_EVALUATOR: std::sync::OnceLock<Option<crate::metal_ramlm::MetalCEReduceEvaluator>> = std::sync::OnceLock::new();
        let _ce_reduce = CE_REDUCE_EVALUATOR.get_or_init(|| {
            crate::metal_ramlm::MetalCEReduceEvaluator::new().ok()
        });

        // Disabled: Legacy GPU CE path with CPU→GPU scatter is slower
        if false {
            let ce_eval = _ce_reduce.as_ref().unwrap();
            let scores_buffer = ce_eval.create_scores_buffer(num_eval, num_clusters);

            let mut dense_idx = 0usize;
            let mut sparse_idx = 0usize;
            let mut all_gpu_success = true;

            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                let group = &export.groups[*group_idx];

                let group_scores_result = if *is_sparse {
                    let sparse_export = &export.sparse_exports[sparse_idx];
                    sparse_idx += 1;

                    let call_start = std::time::Instant::now();
                    let result = if let Some(sparse_eval) = sparse_metal {
                        evaluate_group_sparse_gpu(
                            sparse_eval,
                            &packed_eval,
                            &export.connections,
                            sparse_export,
                            group,
                            num_eval,
                            words_per_example,
                            crate::neuron_memory::MODE_TERNARY,
                        )
                    } else {
                        Err("No sparse evaluator".to_string())
                    };
                    if timing_enabled && result.is_ok() {
                        gpu_time_ms += call_start.elapsed().as_millis();
                        gpu_calls += 1;
                    }
                    result
                } else {
                    let dense_words = &export.dense_exports[dense_idx];
                    dense_idx += 1;

                    let call_start = std::time::Instant::now();
                    let result = if let Some(metal_eval) = metal {
                        evaluate_group_metal(
                            metal_eval,
                            &packed_eval,
                            &export.connections,
                            dense_words,
                            group,
                            num_eval,
                            words_per_example,
                            crate::neuron_memory::MODE_TERNARY,
                        )
                    } else {
                        Err("No metal evaluator".to_string())
                    };
                    if timing_enabled && result.is_ok() {
                        gpu_time_ms += call_start.elapsed().as_millis();
                        gpu_calls += 1;
                    }
                    result
                };

                match group_scores_result {
                    Ok(group_scores) => {
                        let scatter_start = std::time::Instant::now();
                        ce_eval.scatter_to_buffer(
                            &group_scores,
                            &scores_buffer,
                            cluster_ids,
                            num_eval,
                            num_clusters,
                        );
                        if timing_enabled {
                            scatter_time_ms += scatter_start.elapsed().as_millis();
                        }
                    }
                    Err(_) => {
                        all_gpu_success = false;
                        break;
                    }
                }
            }

            if all_gpu_success {
                let ce_start = std::time::Instant::now();
                let result = ce_eval.compute_ce_from_buffer(
                    &scores_buffer,
                    eval_targets,
                    num_eval,
                    num_clusters,
                );
                if timing_enabled {
                    let ce_time = ce_start.elapsed().as_millis();
                    let total_ms = eval_start.elapsed().as_millis();
                    eprintln!(
                        "[EVAL_HYBRID] GPU_CE_PATH total={}ms gpu={}ms({}calls) scatter={}ms ce={}ms",
                        total_ms, gpu_time_ms, gpu_calls, scatter_time_ms, ce_time
                    );
                }
                if let Ok((ce, acc)) = result {
                    return (ce, acc);
                }
            }
        }
    } // cfg(target_os = "macos") Legacy GPU CE PATH

    // CPU FALLBACK PATH: Process groups separately, accumulate scores, compute CE on CPU
    // Pre-compute scores for all examples × clusters
    let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];

    let mut dense_idx = 0usize;
    let mut sparse_idx = 0usize;

    for (is_sparse, group_idx, cluster_ids) in &export.group_info {
        let group = &export.groups[*group_idx];

        if *is_sparse {
            // Try GPU sparse evaluation
            let sparse_export = &export.sparse_exports[sparse_idx];
            sparse_idx += 1;

            let call_start = std::time::Instant::now();
            let gpu_success = if let Some(sparse_eval) = sparse_metal {
                match evaluate_group_sparse_gpu(
                    sparse_eval,
                    &packed_eval,
                    &export.connections,
                    sparse_export,
                    group,
                    num_eval,
                    words_per_example,
                    crate::neuron_memory::MODE_TERNARY,
                ) {
                    Ok(group_scores) => {
                        if timing_enabled {
                            gpu_time_ms += call_start.elapsed().as_millis();
                            gpu_calls += 1;
                        }
                        // Parallel scatter using rayon - each thread handles different examples
                        let scatter_start = std::time::Instant::now();
                        let num_group_clusters = group.cluster_count();
                        all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                            for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * num_group_clusters + local_cluster;
                                scores[cluster_id] = group_scores[score_idx] as f64;
                            }
                        });
                        if timing_enabled {
                            scatter_time_ms += scatter_start.elapsed().as_millis();
                        }
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };

            if !gpu_success {
                // CPU fallback using binary search
                let cpu_start = std::time::Instant::now();
                all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                    let input_start = ex_idx * total_input_bits;
                    let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

                    for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                        // Use actual neurons if coalesced, otherwise MAX
                        let actual_neurons = if let Some(ref an) = group.actual_neurons {
                            an[local_cluster] as usize
                        } else {
                            group.neurons
                        };

                        let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout
                        let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                        let mut sum = 0.0f32;
                        for n in 0..actual_neurons {  // Only iterate actual neurons
                            let conn_start = conn_base + n * group.bits;
                            let address = compute_address(input_bits, &export.connections[conn_start..], group.bits);
                            let cell = sparse_export.lookup(neuron_base + n, address as u64);
                            sum += match cell as i64 {
                                FALSE => 0.0,
                                TRUE => 1.0,
                                _ => empty_value,
                            };
                        }

                        scores[cluster_id] = (sum / actual_neurons as f32) as f64;  // Divide by actual
                    }
                });
                if timing_enabled {
                    cpu_time_ms += cpu_start.elapsed().as_millis();
                    cpu_calls += 1;
                }
            }
        } else {
            // Try GPU dense evaluation
            let call_start = std::time::Instant::now();
            let dense_words = &export.dense_exports[dense_idx];
            dense_idx += 1;

            let gpu_success = if let Some(metal_eval) = metal {
                match evaluate_group_metal(
                    metal_eval,
                    &packed_eval,
                    &export.connections,
                    dense_words,
                    group,
                    num_eval,
                    words_per_example,
                    crate::neuron_memory::MODE_TERNARY,
                ) {
                    Ok(group_scores) => {
                        if timing_enabled {
                            gpu_time_ms += call_start.elapsed().as_millis();
                            gpu_calls += 1;
                        }
                        // Parallel scatter using rayon - each thread handles different examples
                        let scatter_start = std::time::Instant::now();
                        let num_group_clusters = group.cluster_count();
                        all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                            for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * num_group_clusters + local_cluster;
                                scores[cluster_id] = group_scores[score_idx] as f64;
                            }
                        });
                        if timing_enabled {
                            scatter_time_ms += scatter_start.elapsed().as_millis();
                        }
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };

            if !gpu_success {
                // CPU fallback for dense groups: read cells from exported memory words
                let cpu_start = std::time::Instant::now();
                all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                    let input_start = ex_idx * total_input_bits;
                    let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

                    for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                        let actual_neurons = if let Some(ref an) = group.actual_neurons {
                            an[local_cluster] as usize
                        } else {
                            group.neurons
                        };

                        let neuron_base = local_cluster * group.neurons;
                        let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                        let mut sum = 0.0f32;
                        for n in 0..actual_neurons {
                            let conn_start = conn_base + n * group.bits;
                            let address = compute_address(input_bits, &export.connections[conn_start..], group.bits);
                            let cell = read_cell(dense_words, neuron_base + n, address, group.words_per_neuron);
                            sum += match cell {
                                FALSE => 0.0,
                                TRUE => 1.0,
                                _ => empty_value,
                            };
                        }

                        scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                    }
                });
                if timing_enabled {
                    cpu_time_ms += cpu_start.elapsed().as_millis();
                    cpu_calls += 1;
                }
            }
        }
    }

    // Print timing summary if enabled
    if timing_enabled {
        let total_ms = eval_start.elapsed().as_millis();
        eprintln!(
            "[EVAL_HYBRID] total={}ms gpu={}ms({}calls) cpu={}ms({}calls) scatter={}ms",
            total_ms, gpu_time_ms, gpu_calls, cpu_time_ms, cpu_calls, scatter_time_ms
        );
    }

    // Compute CE and accuracy from pre-computed scores
    let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
        let target_idx = eval_targets[ex_idx] as usize;

        let predicted = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let correct: u64 = if predicted == target_idx { 1 } else { 0 };

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let target_prob = exp_scores[target_idx] / sum_exp;
        let ce = -(target_prob + epsilon).ln();

        (ce, correct)
    }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

    let avg_ce = total_ce / num_eval as f64;
    let accuracy = total_correct as f64 / num_eval as f64;

    (avg_ce, accuracy)
}

/// Evaluate genomes in PARALLEL with CPU+GPU HYBRID evaluation and PIPELINING.
///
/// Strategy:
/// 1. Parallel genome training on CPU (batch of genomes at once)
/// 2. Export to GPU-compatible format
/// 3. GPU batch evaluation (while CPU trains next batch)
/// 4. CPU+GPU hybrid: GPU evaluates, CPU assists with fallback
///
/// Performance benefits:
/// - Parallel training: N genomes trained simultaneously (vs sequential)
/// - GPU acceleration: Both dense and sparse groups on GPU
/// - Persistent worker: Eval thread stays alive across calls (eliminates spawn overhead)
/// - Pipelining: CPU trains batch N+1 while GPU evaluates batch N
///
/// Returns: Vec of (cross_entropy, accuracy) tuples - one per genome
pub fn evaluate_genomes_parallel_hybrid(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> Vec<(f64, f64)> {
    if num_genomes == 0 {
        return vec![];
    }

    // Pre-compute genome_bpn_offsets: genomes_bits_flat has total_neurons entries per genome
    // (per-neuron bits), NOT num_clusters entries. This offset table maps each genome to its
    // slice in genomes_bits_flat.
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    debug_assert_eq!(
        genomes_bits_flat.len(),
        *genome_bpn_offsets.last().unwrap(),
        "genomes_bits_flat length ({}) != expected total neurons ({})",
        genomes_bits_flat.len(),
        genome_bpn_offsets.last().unwrap(),
    );

    // Get first genome's config to determine pool sizing (use per-cluster max bits)
    let first_neurons = &genomes_neurons_flat[0..num_clusters];
    let first_per_neuron_bits = &genomes_bits_flat[0..genome_bpn_offsets[1]];
    let first_bits_per_cluster = per_cluster_max_bits(first_per_neuron_bits, first_neurons);

    // Calculate memory budget and pool size
    let batch_size = std::env::var("WNN_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let budget_gb = get_available_memory_gb() * 0.6;
            let cpu_cores = rayon::current_num_threads();
            let (_, computed_batch) = calculate_pool_size(
                &first_bits_per_cluster,
                first_neurons,
                num_clusters,
                budget_gb,
                cpu_cores,
            );
            computed_batch
        });

    // Pre-compute connection offsets and sizes for each genome (handles variable configs)
    let use_provided_connections = !genomes_connections_flat.is_empty();

    // Compute per-genome connection offsets: conn_size = sum of per-neuron bits
    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    // Create shared eval data (Arc for zero-copy sharing with persistent worker)
    let eval_data = Arc::new(EvalData {
        eval_input_bits: eval_input_bits.to_vec(),
        eval_targets: eval_targets.to_vec(),
        num_eval,
        num_clusters,
        total_input_bits,
        empty_value,
    });

    // Get persistent eval worker (initialized once, stays alive for session)
    let eval_worker = get_eval_worker();

    // Collect all results
    let mut all_results: Vec<(usize, f64, f64)> = Vec::with_capacity(num_genomes);

    // Process genomes in batches
    let num_batches = (num_genomes + batch_size - 1) / batch_size;

    // Log batch configuration if WNN_GROUP_LOG is set
    if std::env::var("WNN_GROUP_LOG").is_ok() && num_batches > 1 {
        eprintln!(
            "[BATCH_CONFIG] genomes={} batch_size={} num_batches={}",
            num_genomes, batch_size, num_batches
        );
    }

    // Timing instrumentation (enabled via WNN_TIMING env var)
    let timing_enabled = std::env::var("WNN_TIMING").is_ok();
    let mut total_train_ms = 0u128;
    let mut total_eval_ms = 0u128;
    let mut total_sparse_keys = 0usize;

    // Pack input bits to u64 once (shared across all genomes for GPU address computation)
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_bools_to_u64(train_input_bits, num_train, total_input_bits);

    // Progress logging for parallel batch (logs training completion, not final results)
    let progress_log = std::env::var("WNN_PROGRESS_LOG").map(|v| v == "1").unwrap_or(false);
    let log_path = std::env::var("WNN_LOG_PATH").ok();
    let current_gen: usize = std::env::var("WNN_PROGRESS_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let total_gens: usize = std::env::var("WNN_PROGRESS_TOTAL_GENS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let log_type = std::env::var("WNN_PROGRESS_TYPE").unwrap_or_else(|_| "Init".to_string());
    let batch_offset: usize = std::env::var("WNN_PROGRESS_OFFSET").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let total_count: usize = std::env::var("WNN_PROGRESS_TOTAL").ok().and_then(|v| v.parse().ok()).unwrap_or(num_genomes);

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_genomes);
        let current_batch_size = batch_end - batch_start;

        let train_start = std::time::Instant::now();

        // Train batch in parallel - each genome builds its own config (handles variable architectures)
        let batch_exports: Vec<(usize, GenomeExport)> = (0..current_batch_size)
            .into_par_iter()
            .map(|local_idx| {
                let genome_idx = batch_start + local_idx;

                // Get this genome's config (per-neuron bits + per-cluster neurons)
                let genome_offset = genome_idx * num_clusters;
                let neurons_per_cluster = &genomes_neurons_flat[genome_offset..genome_offset + num_clusters];

                // Extract per-neuron bits for this genome
                let bpn_start = genome_bpn_offsets[genome_idx];
                let bpn_end = genome_bpn_offsets[genome_idx + 1];
                let per_neuron_bits = &genomes_bits_flat[bpn_start..bpn_end];

                // Compute per-cluster max bits (for build_groups and GPU dispatch)
                let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

                // Build neuron metadata for per-neuron training
                let (cluster_neuron_starts, neuron_conn_offsets) =
                    build_neuron_metadata(per_neuron_bits, neurons_per_cluster);

                // Build config groups for THIS genome (using per-cluster max bits)
                let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

                // Build cluster-to-group mapping for THIS genome
                let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
                for (group_idx, group) in groups.iter().enumerate() {
                    for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                        cluster_to_group[cluster_id] = (group_idx, local_idx);
                    }
                }

                // Create memory for THIS genome
                let memories: Vec<GroupMemory> = groups.iter()
                    .map(|g| GroupMemory::new(g.total_neurons(), g.bits))
                    .collect();

                // Get original per-neuron connections for this genome
                let original_connections: Vec<i64> = if use_provided_connections {
                    let conn_offset = conn_offsets[genome_idx];
                    let conn_size = conn_sizes[genome_idx];
                    genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec()
                } else {
                    // Generate random per-neuron connections
                    use rand::{Rng, SeedableRng};
                    let mut rng = rand::rngs::SmallRng::seed_from_u64((genome_idx * 12345) as u64);
                    let total_conn: usize = per_neuron_bits.iter().sum();
                    let mut conns = Vec::with_capacity(total_conn);
                    for _ in 0..total_conn {
                        conns.push(rng.gen_range(0..total_input_bits as i64));
                    }
                    conns
                };

                // Compute training addresses on GPU (falls back to CPU if unavailable)
                let gpu_addresses = try_gpu_addresses_adaptive(
                    &packed_train_input,
                    words_per_example,
                    per_neuron_bits,
                    &neuron_conn_offsets,
                    &original_connections,
                    num_train,
                );

                // Train this genome using per-neuron bits
                train_genome_in_slot(
                    &memories,
                    &groups,
                    &original_connections,
                    per_neuron_bits,
                    &cluster_neuron_starts,
                    &neuron_conn_offsets,
                    &cluster_to_group,
                    train_input_bits,
                    train_targets,
                    train_negatives,
                    num_train,
                    num_negatives,
                    total_input_bits,
                    gpu_addresses.as_deref(),
                );

                // Build GPU-padded connections (per-neuron → group layout with padding)
                let gpu_connections = reorganize_connections_for_gpu(
                    &original_connections,
                    per_neuron_bits,
                    neurons_per_cluster,
                    &groups,
                );

                // Export for GPU using THIS genome's groups
                let export = export_genome_for_gpu(&memories, &groups, &gpu_connections);

                (genome_idx, export)
            })
            .collect();

        let train_elapsed = train_start.elapsed();

        // Track sparse export sizes for timing diagnostics
        let sparse_keys_total: usize = if timing_enabled {
            batch_exports.iter()
                .map(|(_, export)| export.sparse_exports.iter().map(|se| se.keys.len()).sum::<usize>())
                .sum()
        } else { 0 };

        let eval_start = std::time::Instant::now();

        // Send to persistent eval worker and get results
        let batch_results = eval_worker.evaluate(batch_exports, Arc::clone(&eval_data));

        let eval_elapsed_secs = eval_start.elapsed().as_secs_f64();
        let batch_total_secs = train_elapsed.as_secs_f64() + eval_elapsed_secs;

        // Log results with CE/Acc after batch completes
        if progress_log {
            use std::io::Write;
            let now = chrono::Local::now();
            let gen_width = total_gens.to_string().len();
            let pos_width = total_count.to_string().len();
            let type_padded = format!("{:<4}", &log_type[..log_type.len().min(4)]);

            for (genome_idx, ce, acc) in &batch_results {
                let overall_position = batch_offset + genome_idx + 1;
                let msg = format!(
                    "{} | [Gen {:0gen_width$}/{:0gen_width$}] Genome {:0pos_width$}/{} ({}): CE={:.4}, Acc={:.4}% ({:.1}s)\n",
                    now.format("%H:%M:%S"),
                    current_gen, total_gens,
                    overall_position, total_count,
                    type_padded,
                    ce, acc * 100.0,
                    batch_total_secs,
                    gen_width = gen_width,
                    pos_width = pos_width,
                );
                if let Some(ref path) = log_path {
                    if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
                        let _ = file.write_all(msg.as_bytes());
                        let _ = file.flush();
                    }
                } else {
                    eprint!("{}", msg);
                }
            }
        }

        all_results.extend(batch_results);

        if timing_enabled {
            total_train_ms += train_elapsed.as_millis();
            total_eval_ms += (eval_elapsed_secs * 1000.0) as u128;
            total_sparse_keys += sparse_keys_total;
        }
    }

    // Print timing summary if enabled
    if timing_enabled && num_genomes > 0 {
        let train_per_genome = total_train_ms as f64 / num_genomes as f64;
        let eval_per_genome = total_eval_ms as f64 / num_genomes as f64;
        let sparse_per_genome = total_sparse_keys as f64 / num_genomes as f64;
        eprintln!(
            "[TIMING] batch_size={}, genomes={}: train={:.0}ms/genome, eval={:.0}ms/genome, total={:.0}ms/genome, sparse_keys={:.0}/genome",
            batch_size, num_genomes, train_per_genome, eval_per_genome, train_per_genome + eval_per_genome, sparse_per_genome
        );
    }

    // Sort results by genome index and return
    let mut results: Vec<(f64, f64)> = vec![(0.0, 0.0); num_genomes];
    for (genome_idx, ce, acc) in all_results {
        results[genome_idx] = (ce, acc);
    }

    results
}

/// Evaluate a SINGLE genome WITH gating, returning both gated and non-gated metrics.
///
/// This function:
/// 1. Trains base RAM on training data
/// 2. Trains gating model on training data (target gate = true only for target cluster)
/// 3. Evaluates WITHOUT gating → (ce, acc)
/// 4. Evaluates WITH gating → (gated_ce, gated_acc)
///
/// # Returns
/// (ce, accuracy, gated_ce, gated_accuracy)
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genome_with_gating(
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neurons_per_gate: usize,
    bits_per_gate_neuron: usize,
    vote_threshold_frac: f32,
    gating_seed: u64,
) -> (f64, f64, f64, f64) {
    use crate::gating::RAMGating;

    let epsilon = 1e-10f64;

    // ========================================================================
    // Step 1: Train base RAM (same as existing evaluation)
    // ========================================================================

    // Build config groups for this genome
    let bits_per_cluster: Vec<usize> = bits_flat.to_vec();
    let neurons_per_cluster: Vec<usize> = neurons_flat.to_vec();
    let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);

    // Create hybrid memory for each config group
    let group_memories: Vec<GroupMemory> = groups.iter()
        .map(|g| GroupMemory::new(g.total_neurons(), g.bits))
        .collect();

    // Reorganize connections for coalescing if needed
    let connections: Vec<i64> = if use_coalesced_groups() {
        reorganize_connections_for_coalescing(
            connections_flat,
            &bits_per_cluster,
            &neurons_per_cluster,
            &groups,
        )
    } else {
        connections_flat.to_vec()
    };

    // Build cluster-to-group mapping
    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    // Train: iterate over training examples (parallel)
    (0..num_train).into_par_iter().for_each(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &train_input_bits[input_start..input_start + total_input_bits];

        let true_cluster = train_targets[ex_idx] as usize;

        // Train positive example
        {
            let (group_idx, local_cluster) = cluster_to_group[true_cluster];
            let group = &groups[group_idx];
            let memory = &group_memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;
            let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

            for n in 0..actual_neurons {
                let conn_start = conn_base + n * group.bits;
                let address = compute_address(input_bits, &connections[conn_start..], group.bits);
                memory.write(neuron_base + n, address, TRUE, false);
            }
        }

        // Train negative examples
        let neg_start = ex_idx * num_negatives;
        for k in 0..num_negatives {
            let false_cluster = train_negatives[neg_start + k] as usize;
            if false_cluster == true_cluster {
                continue;
            }

            let (group_idx, local_cluster) = cluster_to_group[false_cluster];
            let group = &groups[group_idx];
            let memory = &group_memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;
            let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

            for n in 0..actual_neurons {
                let conn_start = conn_base + n * group.bits;
                let address = compute_address(input_bits, &connections[conn_start..], group.bits);
                memory.write(neuron_base + n, address, FALSE, false);
            }
        }
    });

    // ========================================================================
    // Step 2: Train gating model (parallel)
    // ========================================================================

    let gating = RAMGating::new(
        num_clusters,
        neurons_per_gate,
        bits_per_gate_neuron,
        total_input_bits,
        vote_threshold_frac,
        Some(gating_seed),
    );

    // Build target gates in parallel: for each example, target_gate[target] = true
    let target_gates_flat: Vec<bool> = (0..num_train)
        .into_par_iter()
        .flat_map(|ex_idx| {
            let target = train_targets[ex_idx] as usize;
            let mut gates = vec![false; num_clusters];
            if target < num_clusters {
                gates[target] = true;
            }
            gates
        })
        .collect();

    // Train gating using parallel batch training
    gating.train_batch(train_input_bits, &target_gates_flat, num_train, false);

    // ========================================================================
    // Step 3: Evaluate WITHOUT gating - pre-compute all scores
    // ========================================================================

    let all_scores: Vec<Vec<f64>> = (0..num_eval).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

        let mut scores = vec![0.0f64; num_clusters];

        for (group_idx, group) in groups.iter().enumerate() {
            let memory = &group_memories[group_idx];

            for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                let actual_neurons = if let Some(ref an) = group.actual_neurons {
                    an[local_cluster] as usize
                } else {
                    group.neurons
                };

                let neuron_base = local_cluster * group.neurons;
                let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                let mut sum = 0.0f32;
                for n in 0..actual_neurons {
                    let conn_start = conn_base + n * group.bits;
                    let address = compute_address(input_bits, &connections[conn_start..], group.bits);
                    let cell = memory.read(neuron_base + n, address);
                    sum += match cell {
                        FALSE => 0.0,
                        TRUE => 1.0,
                        _ => empty_value,
                    };
                }

                scores[cluster_id] = (sum / actual_neurons as f32) as f64;
            }
        }

        scores
    }).collect();

    // Compute CE and accuracy without gating
    let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
        let target_idx = eval_targets[ex_idx] as usize;

        // Find prediction (argmax) for accuracy
        let predicted = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let correct: u64 = if predicted == target_idx { 1 } else { 0 };

        // Softmax and cross-entropy
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let target_prob = exp_scores[target_idx] / sum_exp;
        let ce = -(target_prob + epsilon).ln();

        (ce, correct)
    }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

    let ce = total_ce / num_eval as f64;
    let accuracy = total_correct as f64 / num_eval as f64;

    // ========================================================================
    // Step 4: Evaluate WITH gating
    // ========================================================================

    let (total_gated_ce, total_gated_correct): (f64, u64) = (0..num_eval).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];
        let target_idx = eval_targets[ex_idx] as usize;

        // Get scores for this example
        let scores = &all_scores[ex_idx];

        // Compute gates for this input
        let gates = gating.forward_single(input_bits);

        // Apply gates to scores (multiply)
        let gated_scores: Vec<f64> = scores.iter().zip(gates.iter())
            .map(|(&s, &g)| s * g as f64)
            .collect();

        // Check if any gates are open
        let any_open: bool = gates.iter().any(|&g| g > 0.0);

        if !any_open {
            // No gates open - use original scores (fallback)
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let target_prob = exp_scores[target_idx] / sum_exp;
            let ce = -(target_prob + epsilon).ln();

            let predicted = scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let correct: u64 = if predicted == target_idx { 1 } else { 0 };

            (ce, correct)
        } else {
            // Gated evaluation
            let predicted = gated_scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let correct: u64 = if predicted == target_idx { 1 } else { 0 };

            // Softmax on gated scores
            let max_score = gated_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = gated_scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            let target_prob = exp_scores[target_idx] / sum_exp;
            let ce = -(target_prob + epsilon).ln();

            (ce, correct)
        }
    }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

    let gated_ce = total_gated_ce / num_eval as f64;
    let gated_accuracy = total_gated_correct as f64 / num_eval as f64;

    (ce, accuracy, gated_ce, gated_accuracy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_config_groups() {
        // 5 clusters with 3 different configs
        let bits = vec![8, 8, 10, 10, 8];
        let neurons = vec![5, 5, 3, 3, 5];

        let groups = build_config_groups(&bits, &neurons);

        assert_eq!(groups.len(), 2); // (5,8) and (3,10)

        // Find the (5,8) group
        let group_5_8 = groups.iter().find(|g| g.neurons == 5 && g.bits == 8).unwrap();
        assert_eq!(group_5_8.cluster_ids, vec![0, 1, 4]);

        // Find the (3,10) group
        let group_3_10 = groups.iter().find(|g| g.neurons == 3 && g.bits == 10).unwrap();
        assert_eq!(group_3_10.cluster_ids, vec![2, 3]);
    }
}
