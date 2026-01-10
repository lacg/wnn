//! Sparse Memory Backend for RAM Neurons
//!
//! Uses DashMap (lock-free concurrent HashMap) for sparse storage, optimal for
//! neurons with >10 bits per neuron. Memory usage is O(written cells) instead
//! of O(2^n_bits), making it feasible for larger bit configurations (15-30 bits).
//!
//! Trade-offs vs Dense:
//! - Memory: O(cells_written) vs O(2^n_bits * neurons) - MUCH better for large bits
//! - Lookup: O(1) average but with hash overhead vs O(1) direct indexing
//! - Best for: >10 bits per neuron where 2^n becomes impractically large
//!
//! Performance improvements over RwLock<FxHashMap>:
//! - Lock-free concurrent writes: ~4-8x faster with high contention
//! - Fine-grained sharding: different addresses can be written simultaneously
//! - No global lock contention on hot neurons (Tier 0)
//!
//! Cell values (matches dense backend):
//! - FALSE = 0
//! - TRUE = 1
//! - EMPTY = 2 (default for unwritten cells)

use dashmap::DashMap;
use rayon::prelude::*;
use std::hash::BuildHasherDefault;
use rustc_hash::FxHasher;

/// Memory cell values (matches dense backend)
pub const FALSE: u8 = 0;
pub const TRUE: u8 = 1;
pub const EMPTY: u8 = 2;

/// Fast hasher type for DashMap
type FxBuildHasher = BuildHasherDefault<FxHasher>;

// =============================================================================
// SOFTMAX CROSS-ENTROPY COMPUTATION
// =============================================================================

/// Compute cross-entropy with softmax normalization (matches Python's behavior)
///
/// CRITICAL: Raw scores from forward pass are NOT probabilities (don't sum to 1).
/// We must apply softmax to get proper probabilities before computing CE.
///
/// CE = -mean(log(softmax(scores)[target]))
///
/// For numerical stability, we use the log-sum-exp trick:
/// log(softmax(x)[i]) = x[i] - log(sum(exp(x)))
///                    = x[i] - max(x) - log(sum(exp(x - max(x))))
fn compute_ce_with_softmax(
    scores: &[f32],
    targets: &[i64],
    num_examples: usize,
    num_clusters: usize,
) -> f64 {
    let mut total_ce = 0.0f64;

    for ex_idx in 0..num_examples {
        let score_start = ex_idx * num_clusters;
        let target = targets[ex_idx] as usize;

        // Find max for numerical stability
        let mut max_score = f32::NEG_INFINITY;
        for c in 0..num_clusters {
            let s = scores[score_start + c];
            if s > max_score {
                max_score = s;
            }
        }

        // Compute log-sum-exp: log(sum(exp(s - max)))
        let mut sum_exp = 0.0f64;
        for c in 0..num_clusters {
            let s = scores[score_start + c] as f64;
            sum_exp += (s - max_score as f64).exp();
        }
        let log_sum_exp = sum_exp.ln();

        // log(softmax(target)) = target_score - max - log_sum_exp
        let target_score = scores[score_start + target] as f64;
        let log_prob = target_score - max_score as f64 - log_sum_exp;

        // CE contribution: -log(prob)
        total_ce -= log_prob;
    }

    total_ce / num_examples as f64
}

/// Sparse memory storage for all neurons in a layer
/// Uses DashMap for lock-free concurrent access (much faster than RwLock)
pub struct SparseLayerMemory {
    /// Per-neuron concurrent hash maps: address -> cell value
    /// Using FxHasher for fast hashing
    neurons: Vec<DashMap<u64, u8, FxBuildHasher>>,
    pub num_neurons: usize,
    pub bits_per_neuron: usize,
}

impl SparseLayerMemory {
    /// Create new sparse layer with given number of neurons
    pub fn new(num_neurons: usize, bits_per_neuron: usize) -> Self {
        let neurons: Vec<_> = (0..num_neurons)
            .map(|_| DashMap::with_hasher(FxBuildHasher::default()))
            .collect();

        Self {
            neurons,
            num_neurons,
            bits_per_neuron,
        }
    }

    /// Read cell value for a specific neuron and address
    /// Returns EMPTY for unwritten cells
    #[inline]
    pub fn read_cell(&self, neuron_idx: usize, address: u64) -> u8 {
        self.neurons[neuron_idx]
            .get(&address)
            .map(|v| *v)
            .unwrap_or(EMPTY)
    }

    /// Write cell value for a specific neuron and address
    /// Returns true if the cell was modified
    #[inline]
    pub fn write_cell(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool {
        let map = &self.neurons[neuron_idx];

        // Use entry API for atomic read-modify-write
        use dashmap::mapref::entry::Entry;

        match map.entry(address) {
            Entry::Occupied(mut entry) => {
                let current = *entry.get();
                if !allow_override || current == value {
                    return false;
                }
                entry.insert(value);
                true
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
                true
            }
        }
    }

    /// Get total number of written cells across all neurons (for stats)
    pub fn total_cells(&self) -> usize {
        self.neurons.iter()
            .map(|n| n.len())
            .sum()
    }

    /// Get per-neuron cell counts
    pub fn cell_counts(&self) -> Vec<usize> {
        self.neurons.iter()
            .map(|n: &DashMap<u64, u8, FxBuildHasher>| n.len())
            .collect()
    }

    /// Export to flat representation: Vec<(neuron_idx, address, value)>
    pub fn export(&self) -> Vec<(usize, u64, u8)> {
        let mut cells: Vec<(usize, u64, u8)> = Vec::new();

        for (neuron_idx, neuron_map) in self.neurons.iter().enumerate() {
            for entry in neuron_map.iter() {
                let key: u64 = *entry.key();
                let val: u8 = *entry.value();
                cells.push((neuron_idx, key, val));
            }
        }

        cells
    }

    /// Import from flat representation
    pub fn import(&self, cells: &[(usize, u64, u8)]) {
        for &(neuron_idx, address, value) in cells {
            if neuron_idx < self.num_neurons {
                self.neurons[neuron_idx].insert(address, value);
            }
        }
    }

    /// Reset all memory to empty
    pub fn reset(&self) {
        for neuron in &self.neurons {
            neuron.clear();
        }
    }

    /// Clone the memory (for parallel candidate evaluation)
    pub fn clone_memory(&self) -> Self {
        let neurons: Vec<DashMap<u64, u8, FxBuildHasher>> = self.neurons.iter()
            .map(|n: &DashMap<u64, u8, FxBuildHasher>| {
                let new_map: DashMap<u64, u8, FxBuildHasher> = DashMap::with_hasher(FxBuildHasher::default());
                for entry in n.iter() {
                    let key: u64 = *entry.key();
                    let val: u8 = *entry.value();
                    new_map.insert(key, val);
                }
                new_map
            })
            .collect();

        Self {
            neurons,
            num_neurons: self.num_neurons,
            bits_per_neuron: self.bits_per_neuron,
        }
    }

    /// Export to GPU-compatible sorted array format
    /// Returns (keys_flat, values_flat, offsets, counts) where:
    /// - keys_flat: All keys concatenated, sorted per neuron
    /// - values_flat: Corresponding values
    /// - offsets: Start offset for each neuron in keys_flat
    /// - counts: Number of entries for each neuron
    ///
    /// GPU can binary search within [offsets[n], offsets[n]+counts[n])
    pub fn export_for_gpu(&self) -> SparseGpuExport {
        let mut keys_flat: Vec<u64> = Vec::new();
        let mut values_flat: Vec<u8> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(self.num_neurons);
        let mut counts: Vec<u32> = Vec::with_capacity(self.num_neurons);

        for neuron_map in &self.neurons {
            let offset = keys_flat.len() as u32;
            offsets.push(offset);

            // Collect and sort entries for this neuron
            let mut entries: Vec<(u64, u8)> = neuron_map.iter()
                .map(|entry| (*entry.key(), *entry.value()))
                .collect();
            entries.sort_by_key(|(k, _)| *k);

            counts.push(entries.len() as u32);

            for (key, value) in entries {
                keys_flat.push(key);
                values_flat.push(value);
            }
        }

        SparseGpuExport {
            keys: keys_flat,
            values: values_flat,
            offsets,
            counts,
            num_neurons: self.num_neurons,
        }
    }
}

/// GPU-compatible sparse memory export
/// Format optimized for binary search on GPU
#[derive(Clone)]
pub struct SparseGpuExport {
    /// Sorted keys for all neurons, concatenated
    pub keys: Vec<u64>,
    /// Values corresponding to keys
    pub values: Vec<u8>,
    /// Start offset for each neuron in keys array
    pub offsets: Vec<u32>,
    /// Number of entries for each neuron
    pub counts: Vec<u32>,
    /// Total number of neurons
    pub num_neurons: usize,
}

impl SparseGpuExport {
    /// CPU binary search lookup (for verification)
    #[inline]
    pub fn lookup(&self, neuron_idx: usize, address: u64) -> u8 {
        let start = self.offsets[neuron_idx] as usize;
        let count = self.counts[neuron_idx] as usize;

        if count == 0 {
            return EMPTY;
        }

        let end = start + count;
        let keys_slice = &self.keys[start..end];

        // Binary search
        match keys_slice.binary_search(&address) {
            Ok(idx) => self.values[start + idx],
            Err(_) => EMPTY,
        }
    }

    /// Total memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4 + self.counts.len() * 4
    }

    /// Total number of entries
    pub fn total_entries(&self) -> usize {
        self.keys.len()
    }
}

/// Compute address from input bits and connections (MSB-first, matches dense backend)
#[inline]
pub fn compute_address(
    input_bits: &[bool],
    connections: &[i64],
    bits_per_neuron: usize,
) -> u64 {
    let mut address: u64 = 0;
    for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
        if input_bits[conn_idx as usize] {
            // MSB-first addressing (matches Python and dense Rust)
            address |= 1 << (bits_per_neuron - 1 - i);
        }
    }
    address
}

/// Batch training for sparse memory backend
///
/// Same interface as ramlm::train_batch but uses sparse storage.
/// Two-phase training: TRUEs first (with override), then FALSEs (no override)
pub fn train_batch_sparse(
    memory: &SparseLayerMemory,
    input_bits_flat: &[bool],
    true_clusters: &[i64],
    false_clusters_flat: &[i64],
    connections_flat: &[i64],
    num_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_negatives: usize,
    _allow_override: bool,
) -> usize {
    // Phase 1: Write ALL TRUEs first (with override to ensure TRUE takes priority)
    let true_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let mut ex_modified = 0usize;

        let true_cluster = true_clusters[ex_idx] as usize;
        let true_start_neuron = true_cluster * neurons_per_cluster;

        for neuron_offset in 0..neurons_per_cluster {
            let neuron_idx = true_start_neuron + neuron_offset;
            let conn_start = neuron_idx * bits_per_neuron;
            let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];

            let address = compute_address(input_bits, connections, bits_per_neuron);

            // TRUE writes always override (priority over FALSE)
            if memory.write_cell(neuron_idx, address, TRUE, true) {
                ex_modified += 1;
            }
        }

        ex_modified
    }).sum();

    // Phase 2: Write ALL FALSEs (only to EMPTY cells)
    let false_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let mut ex_modified = 0usize;

        let false_start = ex_idx * num_negatives;
        for neg_idx in 0..num_negatives {
            let false_cluster = false_clusters_flat[false_start + neg_idx] as usize;
            let false_start_neuron = false_cluster * neurons_per_cluster;

            for neuron_offset in 0..neurons_per_cluster {
                let neuron_idx = false_start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];

                let address = compute_address(input_bits, connections, bits_per_neuron);

                // FALSE writes only to EMPTY cells
                if memory.write_cell(neuron_idx, address, FALSE, false) {
                    ex_modified += 1;
                }
            }
        }

        ex_modified
    }).sum();

    true_modified + false_modified
}

/// Batch forward pass for sparse memory backend
///
/// Same interface as ramlm::forward_batch but uses sparse storage.
pub fn forward_batch_sparse(
    memory: &SparseLayerMemory,
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    num_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
) -> Vec<f32> {
    let mut probs = vec![0.0f32; num_examples * num_clusters];

    // Process all examples in parallel
    probs.par_chunks_mut(num_clusters).enumerate().for_each(|(ex_idx, ex_probs)| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        // For each cluster, count TRUE and EMPTY
        for cluster_idx in 0..num_clusters {
            let start_neuron = cluster_idx * neurons_per_cluster;
            let mut count_true = 0u32;
            let mut count_empty = 0u32;

            for neuron_offset in 0..neurons_per_cluster {
                let neuron_idx = start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];

                let address = compute_address(input_bits, connections, bits_per_neuron);
                let cell_value = memory.read_cell(neuron_idx, address);

                if cell_value == TRUE {
                    count_true += 1;
                } else if cell_value == EMPTY {
                    count_empty += 1;
                }
            }

            // Probability = (count_true + 0.5 * count_empty) / neurons_per_cluster
            ex_probs[cluster_idx] = (count_true as f32 + 0.5 * count_empty as f32) / neurons_per_cluster as f32;
        }
    });

    probs
}

// =============================================================================
// PARALLEL GA CANDIDATE EVALUATION
// =============================================================================

/// Train and evaluate a single connectivity pattern
/// Returns cross-entropy loss
fn evaluate_single_connectivity(
    connections_flat: &[i64],
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> f64 {
    // Create fresh memory for this candidate
    let num_neurons = num_clusters * neurons_per_cluster;
    let memory = SparseLayerMemory::new(num_neurons, bits_per_neuron);

    // Train
    train_batch_sparse(
        &memory,
        train_input_bits,
        train_true_clusters,
        train_false_clusters,
        connections_flat,
        num_train_examples,
        total_input_bits,
        bits_per_neuron,
        neurons_per_cluster,
        num_negatives,
        false,
    );

    // Evaluate: compute cross-entropy
    let probs = forward_batch_sparse(
        &memory,
        eval_input_bits,
        connections_flat,
        num_eval_examples,
        total_input_bits,
        bits_per_neuron,
        neurons_per_cluster,
        num_clusters,
    );

    // Compute cross-entropy with softmax normalization (matches Python)
    compute_ce_with_softmax(&probs, eval_targets, num_eval_examples, num_clusters)
}

/// Evaluate multiple connectivity patterns in parallel
/// This is the key function for parallel GA optimization
///
/// Args:
///   candidates_flat: Flattened connectivity patterns [num_candidates, num_neurons * bits_per_neuron]
///   train_input_bits: Training input bits [num_train * total_input_bits]
///   train_true_clusters: Target clusters for training [num_train]
///   train_false_clusters: Negative clusters for training [num_train * num_negatives]
///   eval_input_bits: Evaluation input bits [num_eval * total_input_bits]
///   eval_targets: Target clusters for evaluation [num_eval]
///
/// Returns: Cross-entropy for each candidate [num_candidates]
pub fn evaluate_candidates_parallel(
    candidates_flat: &[i64],
    num_candidates: usize,
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> Vec<f64> {
    let num_neurons = num_clusters * neurons_per_cluster;
    let conn_size = num_neurons * bits_per_neuron;

    // Evaluate all candidates in parallel
    (0..num_candidates).into_par_iter().map(|cand_idx| {
        let conn_start = cand_idx * conn_size;
        let connections = &candidates_flat[conn_start..conn_start + conn_size];

        evaluate_single_connectivity(
            connections,
            train_input_bits,
            train_true_clusters,
            train_false_clusters,
            eval_input_bits,
            eval_targets,
            num_train_examples,
            num_eval_examples,
            total_input_bits,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            num_negatives,
        )
    }).collect()
}

// =============================================================================
// TIERED PARALLEL GA CANDIDATE EVALUATION
// =============================================================================

/// Configuration for a single tier
#[derive(Clone)]
pub struct TierConfig {
    pub start_cluster: usize,
    pub end_cluster: usize,     // exclusive
    pub neurons_per_cluster: usize,
    pub bits_per_neuron: usize,
    pub start_neuron: usize,    // first neuron index for this tier
}

/// Tiered sparse memory - multiple tiers with different configurations
pub struct TieredSparseMemory {
    tiers: Vec<SparseLayerMemory>,
    tier_configs: Vec<TierConfig>,
    num_clusters: usize,
}

impl TieredSparseMemory {
    /// Create tiered memory from tier configurations
    /// tier_configs: Vec of (end_cluster, neurons_per_cluster, bits_per_neuron)
    /// where end_cluster is exclusive. Tiers must be consecutive starting from 0.
    pub fn new(tier_configs: &[(usize, usize, usize)], num_clusters: usize) -> Self {
        let mut tiers = Vec::new();
        let mut configs = Vec::new();
        let mut start_cluster = 0;
        let mut start_neuron = 0;

        for &(end_cluster, neurons_per_cluster, bits_per_neuron) in tier_configs {
            let num_tier_clusters = end_cluster - start_cluster;
            let num_tier_neurons = num_tier_clusters * neurons_per_cluster;

            tiers.push(SparseLayerMemory::new(num_tier_neurons, bits_per_neuron));
            configs.push(TierConfig {
                start_cluster,
                end_cluster,
                neurons_per_cluster,
                bits_per_neuron,
                start_neuron,
            });

            start_neuron += num_tier_neurons;
            start_cluster = end_cluster;
        }

        Self {
            tiers,
            tier_configs: configs,
            num_clusters,
        }
    }

    /// Get tier index for a cluster
    #[inline]
    fn get_tier(&self, cluster: usize) -> usize {
        for (i, config) in self.tier_configs.iter().enumerate() {
            if cluster < config.end_cluster {
                return i;
            }
        }
        self.tier_configs.len() - 1  // last tier for overflow
    }

    /// Get tier config for a cluster
    #[inline]
    fn get_tier_config(&self, cluster: usize) -> &TierConfig {
        &self.tier_configs[self.get_tier(cluster)]
    }

    /// Reset all tiers
    pub fn reset(&self) {
        for tier in &self.tiers {
            tier.reset();
        }
    }
}

/// Train batch on tiered sparse memory (PARALLEL version)
///
/// Uses rayon to process examples in parallel across CPU cores.
/// DashMap handles concurrent writes with lock-free operations.
///
/// Speedup: ~8-12x on 16 cores (limited by memory bandwidth and hot neuron contention)
pub fn train_batch_tiered(
    memory: &TieredSparseMemory,
    input_bits_flat: &[bool],
    true_clusters: &[i64],
    false_clusters_flat: &[i64],
    connections_flat: &[i64],  // flattened connections for ALL neurons
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
) -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let total_modified = AtomicUsize::new(0);

    // Phase 1: Write TRUEs (with override) - PARALLEL
    (0..num_examples).into_par_iter().for_each(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let true_cluster = true_clusters[ex_idx] as usize;
        let tier_idx = memory.get_tier(true_cluster);
        let config = &memory.tier_configs[tier_idx];
        let tier = &memory.tiers[tier_idx];

        let local_cluster = true_cluster - config.start_cluster;
        let local_start_neuron = local_cluster * config.neurons_per_cluster;
        let global_start_neuron = config.start_neuron + local_start_neuron;

        let mut local_modified = 0usize;
        for neuron_offset in 0..config.neurons_per_cluster {
            let local_neuron_idx = local_start_neuron + neuron_offset;
            let global_neuron_idx = global_start_neuron + neuron_offset;

            // Get connections for this neuron
            let conn_start = global_neuron_idx * config.bits_per_neuron;
            let connections = &connections_flat[conn_start..conn_start + config.bits_per_neuron];

            let address = compute_address(input_bits, connections, config.bits_per_neuron);

            if tier.write_cell(local_neuron_idx, address, TRUE, true) {
                local_modified += 1;
            }
        }
        total_modified.fetch_add(local_modified, Ordering::Relaxed);
    });

    // Phase 2: Write FALSEs (no override) - PARALLEL
    (0..num_examples).into_par_iter().for_each(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let false_start = ex_idx * num_negatives;
        let mut local_modified = 0usize;

        for neg_idx in 0..num_negatives {
            let false_cluster = false_clusters_flat[false_start + neg_idx] as usize;
            let tier_idx = memory.get_tier(false_cluster);
            let config = &memory.tier_configs[tier_idx];
            let tier = &memory.tiers[tier_idx];

            let local_cluster = false_cluster - config.start_cluster;
            let local_start_neuron = local_cluster * config.neurons_per_cluster;
            let global_start_neuron = config.start_neuron + local_start_neuron;

            for neuron_offset in 0..config.neurons_per_cluster {
                let local_neuron_idx = local_start_neuron + neuron_offset;
                let global_neuron_idx = global_start_neuron + neuron_offset;

                let conn_start = global_neuron_idx * config.bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + config.bits_per_neuron];

                let address = compute_address(input_bits, connections, config.bits_per_neuron);

                if tier.write_cell(local_neuron_idx, address, FALSE, false) {
                    local_modified += 1;
                }
            }
        }
        total_modified.fetch_add(local_modified, Ordering::Relaxed);
    });

    total_modified.load(Ordering::Relaxed)
}

/// Forward pass for tiered sparse memory
pub fn forward_batch_tiered(
    memory: &TieredSparseMemory,
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    num_examples: usize,
    total_input_bits: usize,
) -> Vec<f32> {
    let num_clusters = memory.num_clusters;
    let mut probs = vec![0.0f32; num_examples * num_clusters];

    // Process examples in parallel
    probs.par_chunks_mut(num_clusters).enumerate().for_each(|(ex_idx, ex_probs)| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        for cluster_idx in 0..num_clusters {
            let tier_idx = memory.get_tier(cluster_idx);
            let config = &memory.tier_configs[tier_idx];
            let tier = &memory.tiers[tier_idx];

            let local_cluster = cluster_idx - config.start_cluster;
            let local_start_neuron = local_cluster * config.neurons_per_cluster;
            let global_start_neuron = config.start_neuron + local_start_neuron;

            let mut count_true = 0u32;
            let mut count_empty = 0u32;

            for neuron_offset in 0..config.neurons_per_cluster {
                let local_neuron_idx = local_start_neuron + neuron_offset;
                let global_neuron_idx = global_start_neuron + neuron_offset;

                let conn_start = global_neuron_idx * config.bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + config.bits_per_neuron];

                let address = compute_address(input_bits, connections, config.bits_per_neuron);
                let cell_value = tier.read_cell(local_neuron_idx, address);

                if cell_value == TRUE {
                    count_true += 1;
                } else if cell_value == EMPTY {
                    count_empty += 1;
                }
            }

            ex_probs[cluster_idx] = (count_true as f32 + 0.5 * count_empty as f32)
                / config.neurons_per_cluster as f32;
        }
    });

    probs
}

/// Evaluate a single tiered connectivity pattern
fn evaluate_single_tiered(
    connections_flat: &[i64],
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    tier_configs: &[(usize, usize, usize)],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> f64 {
    // Create fresh tiered memory
    let memory = TieredSparseMemory::new(tier_configs, num_clusters);

    // Train
    train_batch_tiered(
        &memory,
        train_input_bits,
        train_true_clusters,
        train_false_clusters,
        connections_flat,
        num_train_examples,
        total_input_bits,
        num_negatives,
    );

    // Evaluate
    let probs = forward_batch_tiered(
        &memory,
        eval_input_bits,
        connections_flat,
        num_eval_examples,
        total_input_bits,
    );

    // Compute cross-entropy with softmax normalization (matches Python)
    compute_ce_with_softmax(&probs, eval_targets, num_eval_examples, num_clusters)
}

/// Evaluate multiple tiered connectivity patterns in parallel
/// This is the KEY function for accelerating GA/TS with tiered architectures.
///
/// Args:
///   candidates_flat: Flattened connectivity patterns [num_candidates * total_neurons * max_bits]
///                    NOTE: Each candidate has same total size, padded if needed
///   tier_configs: Vec of (end_cluster, neurons_per_cluster, bits_per_neuron)
///   conn_size_per_candidate: Total connections size per candidate
///
/// Returns: Cross-entropy for each candidate
pub fn evaluate_candidates_parallel_tiered(
    candidates_flat: &[i64],
    num_candidates: usize,
    conn_size_per_candidate: usize,
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    tier_configs: &[(usize, usize, usize)],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> Vec<f64> {
    // Evaluate all candidates in parallel using rayon
    (0..num_candidates).into_par_iter().map(|cand_idx| {
        let conn_start = cand_idx * conn_size_per_candidate;
        let connections = &candidates_flat[conn_start..conn_start + conn_size_per_candidate];

        evaluate_single_tiered(
            connections,
            train_input_bits,
            train_true_clusters,
            train_false_clusters,
            eval_input_bits,
            eval_targets,
            tier_configs,
            num_train_examples,
            num_eval_examples,
            total_input_bits,
            num_clusters,
            num_negatives,
        )
    }).collect()
}

// =============================================================================
// MEMORY-ADAPTIVE BATCH EVALUATION
// =============================================================================

use crate::metal_ramlm::MetalSparseEvaluator;
use std::sync::mpsc;
use std::thread;

/// Estimate memory usage (in bytes) for a single TieredSparseMemory when populated
///
/// Memory comes from:
/// 1. DashMap structural overhead per neuron (~200 bytes base)
/// 2. Entry storage when populated during training (~33 bytes per entry)
/// 3. We estimate entries based on training size and tier distribution
pub fn estimate_memory_per_tiered_sparse(
    tier_configs: &[(usize, usize, usize)],
    num_clusters: usize,
    num_train_examples: usize,
    num_negatives: usize,
) -> usize {
    // Calculate total neurons
    let mut total_neurons = 0usize;
    let mut start_cluster = 0usize;
    for &(end_cluster, neurons_per_cluster, _bits) in tier_configs {
        let num_tier_clusters = end_cluster.min(num_clusters) - start_cluster;
        total_neurons += num_tier_clusters * neurons_per_cluster;
        start_cluster = end_cluster.min(num_clusters);
    }

    // Base overhead: ~200 bytes per DashMap (shards, locks, metadata)
    let base_overhead = total_neurons * 200;

    // Estimate entries written during training:
    // - Each example writes to neurons_per_cluster neurons (TRUE)
    // - Each example writes to num_negatives * neurons_per_cluster neurons (FALSE)
    // - Many addresses overlap, so actual unique entries is much less
    // - Estimate ~10-20% unique addresses (conservative)
    let writes_per_example = (1 + num_negatives) * 8; // avg neurons_per_cluster ~8
    let total_writes = num_train_examples * writes_per_example;
    let estimated_unique_entries = total_writes / 5; // ~20% unique (overlap)

    // Entry size: key (8) + value (1) + DashMap entry overhead (~24) = ~33 bytes
    let entry_overhead = estimated_unique_entries * 33;

    // Add buffer for DashMap growth (capacity is usually 2x entries)
    let total_estimate = base_overhead + entry_overhead * 2;

    total_estimate
}

/// Calculate optimal pool size based on memory budget and hardware
///
/// Returns (pool_size, batch_size) tuple
pub fn calculate_adaptive_pool_size(
    tier_configs: &[(usize, usize, usize)],
    num_clusters: usize,
    num_train_examples: usize,
    num_negatives: usize,
    memory_budget_gb: f64,
    cpu_cores: usize,
) -> (usize, usize) {
    let bytes_per_memory = estimate_memory_per_tiered_sparse(
        tier_configs, num_clusters, num_train_examples, num_negatives
    );

    let budget_bytes = (memory_budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;

    // Reserve 30% for exports and other overhead
    let available_for_pool = (budget_bytes as f64 * 0.7) as usize;

    // Calculate max pool size that fits in memory
    let max_by_memory = if bytes_per_memory > 0 {
        (available_for_pool / bytes_per_memory).max(1)
    } else {
        cpu_cores
    };

    // Pool size = min(memory limit, cpu cores), at least 1
    let pool_size = max_by_memory.min(cpu_cores).max(1);

    // Batch size matches pool size for optimal parallelism
    let batch_size = pool_size;

    (pool_size, batch_size)
}

/// Get available system memory in GB (approximate)
/// Falls back to 16GB if detection fails
pub fn get_available_memory_gb() -> f64 {
    // Try to read from sysctl on macOS
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.memsize").output() {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = mem_str.trim().parse::<u64>() {
                    return bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                }
            }
        }
    }

    // Fallback: assume 16GB
    16.0
}

/// Export data for a single candidate (used in batched evaluation)
struct CandidateExport {
    connections: Vec<i64>,
    keys: Vec<u64>,
    values: Vec<u8>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
}

/// Evaluate candidates with memory-adaptive batching and CPU/GPU pipelining
///
/// Strategy:
/// 1. Calculate optimal pool/batch size based on memory budget
/// 2. Process candidates in batches
/// 3. Pipeline: CPU trains batch N+1 while GPU evaluates batch N
/// 4. Batch GPU evaluation: single dispatch for all candidates in batch
pub fn evaluate_gpu_batch_adaptive(
    candidates_flat: &[i64],
    num_candidates: usize,
    conn_size_per_candidate: usize,
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    tier_configs: &[(usize, usize, usize)],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
    memory_budget_gb: Option<f64>,
) -> Vec<f64> {
    if num_candidates == 0 {
        return vec![];
    }

    let gpu_evaluator = match MetalSparseEvaluator::new() {
        Ok(e) => e,
        Err(_) => {
            // Fallback to CPU if GPU init fails
            return evaluate_candidates_parallel_tiered(
                candidates_flat,
                num_candidates,
                conn_size_per_candidate,
                train_input_bits,
                train_true_clusters,
                train_false_clusters,
                eval_input_bits,
                eval_targets,
                tier_configs,
                num_train_examples,
                num_eval_examples,
                total_input_bits,
                num_clusters,
                num_negatives,
            );
        }
    };

    // Determine memory budget
    let budget_gb = memory_budget_gb.unwrap_or_else(|| {
        // Auto-detect: use 60% of available memory
        get_available_memory_gb() * 0.6
    });

    let cpu_cores = rayon::current_num_threads();

    // Calculate adaptive pool and batch size
    let (pool_size, batch_size) = calculate_adaptive_pool_size(
        tier_configs,
        num_clusters,
        num_train_examples,
        num_negatives,
        budget_gb,
        cpu_cores,
    );

    // Pre-allocate memory pool (reused across batches via reset())
    let memory_pool: Vec<TieredSparseMemory> = (0..pool_size)
        .map(|_| TieredSparseMemory::new(tier_configs, num_clusters))
        .collect();

    // Build tier config for GPU (done once, reused for all batches)
    let gpu_tier_configs: Vec<(usize, usize, usize, usize)> = {
        let mut configs = Vec::new();
        let mut start_neuron = 0usize;
        let mut start_cluster = 0usize;
        for &(end_cluster, neurons_per_cluster, bits_per_neuron) in tier_configs {
            let num_tier_clusters = end_cluster - start_cluster;
            configs.push((end_cluster, neurons_per_cluster, bits_per_neuron, start_neuron));
            start_neuron += num_tier_clusters * neurons_per_cluster;
            start_cluster = end_cluster;
        }
        configs
    };

    // Channel for pipelining: CPU sends exports, GPU thread receives and evaluates
    let (tx, rx) = mpsc::channel::<(usize, Vec<CandidateExport>)>();

    // Clone data needed by GPU thread
    let eval_input_bits_owned = eval_input_bits.to_vec();
    let eval_targets_owned = eval_targets.to_vec();
    let gpu_tier_configs_clone = gpu_tier_configs.clone();

    // Spawn GPU evaluation thread
    let gpu_handle = thread::spawn(move || {
        let mut batch_results: Vec<(usize, Vec<f64>)> = Vec::new();

        while let Ok((batch_idx, exports)) = rx.recv() {
            let mut batch_ces = Vec::with_capacity(exports.len());

            // Evaluate each candidate in this batch on GPU
            for export in exports {
                let probs = gpu_evaluator.forward_batch_tiered(
                    &eval_input_bits_owned,
                    &export.connections,
                    &export.keys,
                    &export.values,
                    &export.offsets,
                    &export.counts,
                    &gpu_tier_configs_clone,
                    num_eval_examples,
                    total_input_bits,
                    num_clusters,
                ).unwrap_or_else(|_| {
                    // Fallback: return uniform distribution (error case)
                    vec![1.0 / num_clusters as f32; num_eval_examples * num_clusters]
                });

                // Compute cross-entropy with softmax normalization (matches Python)
                let ce = compute_ce_with_softmax(&probs, &eval_targets_owned, num_eval_examples, num_clusters);
                batch_ces.push(ce);
            }

            batch_results.push((batch_idx, batch_ces));
        }

        batch_results
    });

    // Process candidates in batches on CPU, send exports to GPU thread
    let num_batches = (num_candidates + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_candidates);
        let current_batch_size = batch_end - batch_start;

        // Thread-local index for memory pool assignment
        use std::sync::atomic::{AtomicUsize, Ordering};
        let thread_counter = AtomicUsize::new(0);

        // Train this batch in parallel, collecting exports
        let batch_exports: Vec<CandidateExport> = (0..current_batch_size)
            .into_par_iter()
            .map(|local_idx| {
                let cand_idx = batch_start + local_idx;

                // Assign to memory pool slot (round-robin)
                let pool_idx = thread_counter.fetch_add(1, Ordering::Relaxed) % pool_size;
                let memory = &memory_pool[pool_idx];
                memory.reset(); // Clear previous data (keeps allocated buckets)

                let conn_start = cand_idx * conn_size_per_candidate;
                let connections = &candidates_flat[conn_start..conn_start + conn_size_per_candidate];

                // Train on this memory
                train_batch_tiered(
                    memory,
                    train_input_bits,
                    train_true_clusters,
                    train_false_clusters,
                    connections,
                    num_train_examples,
                    total_input_bits,
                    num_negatives,
                );

                // Export for GPU
                let mut keys: Vec<u64> = Vec::new();
                let mut values: Vec<u8> = Vec::new();
                let mut offsets: Vec<u32> = Vec::new();
                let mut counts: Vec<u32> = Vec::new();

                for tier in &memory.tiers {
                    let export = tier.export_for_gpu();
                    let base = keys.len() as u32;
                    for &off in &export.offsets {
                        offsets.push(base + off);
                    }
                    counts.extend(&export.counts);
                    keys.extend(&export.keys);
                    values.extend(&export.values);
                }

                CandidateExport {
                    connections: connections.to_vec(),
                    keys,
                    values,
                    offsets,
                    counts,
                }
            })
            .collect();

        // Send batch exports to GPU thread for evaluation
        // (This happens while CPU can start preparing next batch)
        tx.send((batch_idx, batch_exports)).expect("GPU thread disconnected");
    }

    // Close channel to signal GPU thread we're done
    drop(tx);

    // Wait for GPU thread and collect results
    let batch_results = gpu_handle.join().expect("GPU thread panicked");

    // Reassemble results in correct order
    let mut ordered_results = vec![0.0f64; num_candidates];
    for (batch_idx, batch_ces) in batch_results {
        let batch_start = batch_idx * batch_size;
        for (local_idx, ce) in batch_ces.into_iter().enumerate() {
            ordered_results[batch_start + local_idx] = ce;
        }
    }

    ordered_results
}

// =============================================================================
// HYBRID CPU+GPU EVALUATION (LEGACY - uses evaluate_gpu_batch_adaptive internally)
// =============================================================================

/// Evaluate a single tiered connectivity using CPU training + GPU evaluation
/// - Training: CPU with DashMap (fast parallel writes)
/// - Evaluation: GPU with binary search (massive parallelism)
fn evaluate_single_tiered_hybrid(
    connections_flat: &[i64],
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    tier_configs: &[(usize, usize, usize)],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
    gpu_evaluator: &MetalSparseEvaluator,
) -> f64 {
    // Create fresh tiered memory
    let memory = TieredSparseMemory::new(tier_configs, num_clusters);

    // PHASE 1: Train on CPU (DashMap is optimal for parallel writes)
    train_batch_tiered(
        &memory,
        train_input_bits,
        train_true_clusters,
        train_false_clusters,
        connections_flat,
        num_train_examples,
        total_input_bits,
        num_negatives,
    );

    // PHASE 2: Export sparse memory to GPU-compatible format
    let mut all_keys: Vec<u64> = Vec::new();
    let mut all_values: Vec<u8> = Vec::new();
    let mut all_offsets: Vec<u32> = Vec::new();
    let mut all_counts: Vec<u32> = Vec::new();

    // Build tier info for GPU
    let mut gpu_tier_configs: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut start_cluster = 0;

    for (tier_idx, &(end_cluster, neurons_per_cluster, bits_per_neuron)) in tier_configs.iter().enumerate() {
        let tier_export = memory.tiers[tier_idx].export_for_gpu();

        // Adjust offsets for global array
        let base_offset = all_keys.len() as u32;
        for &offset in &tier_export.offsets {
            all_offsets.push(base_offset + offset);
        }
        all_counts.extend(&tier_export.counts);
        all_keys.extend(&tier_export.keys);
        all_values.extend(&tier_export.values);

        // Record tier config with start_neuron
        let start_neuron = if tier_idx == 0 {
            0
        } else {
            gpu_tier_configs.iter().map(|&(_, npc, _, _)| {
                let prev_end = gpu_tier_configs.last().map(|&(ec, _, _, _)| ec).unwrap_or(0);
                (prev_end - start_cluster) * npc
            }).sum::<usize>()
        };

        gpu_tier_configs.push((end_cluster, neurons_per_cluster, bits_per_neuron, memory.tier_configs[tier_idx].start_neuron));
        start_cluster = end_cluster;
    }

    // PHASE 3: Evaluate on GPU (massive parallelism with binary search)
    let probs = gpu_evaluator.forward_batch_tiered(
        eval_input_bits,
        connections_flat,
        &all_keys,
        &all_values,
        &all_offsets,
        &all_counts,
        &gpu_tier_configs,
        num_eval_examples,
        total_input_bits,
        num_clusters,
    ).unwrap_or_else(|_| {
        // Fallback to CPU if GPU fails
        forward_batch_tiered(
            &memory,
            eval_input_bits,
            connections_flat,
            num_eval_examples,
            total_input_bits,
        )
    });

    // Compute cross-entropy with softmax normalization (matches Python)
    compute_ce_with_softmax(&probs, eval_targets, num_eval_examples, num_clusters)
}

/// Memory-adaptive hybrid CPU+GPU parallel candidate evaluation
///
/// Strategy for efficient use of all hardware with controlled memory:
/// 1. Auto-calculate optimal pool/batch size based on memory budget
/// 2. Pipeline: CPU trains batch N+1 while GPU evaluates batch N
/// 3. No data duplication - shared references where possible
///
/// Memory benefits over old approach:
/// - Old: All candidates' exports accumulated before GPU evaluation (~30-50GB)
/// - New: Only 1-2 batches of exports in memory at once (~2-4GB)
///
/// Performance benefits:
/// - Pipelining hides latency (CPU and GPU work simultaneously)
/// - Adaptive batch size balances parallelism vs memory
pub fn evaluate_candidates_parallel_hybrid(
    candidates_flat: &[i64],
    num_candidates: usize,
    conn_size_per_candidate: usize,
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    tier_configs: &[(usize, usize, usize)],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> Vec<f64> {
    // Use the new memory-adaptive version with auto-detected memory budget
    evaluate_gpu_batch_adaptive(
        candidates_flat,
        num_candidates,
        conn_size_per_candidate,
        train_input_bits,
        train_true_clusters,
        train_false_clusters,
        eval_input_bits,
        eval_targets,
        tier_configs,
        num_train_examples,
        num_eval_examples,
        total_input_bits,
        num_clusters,
        num_negatives,
        None, // Auto-detect memory budget
    )
}

/// Hybrid evaluation with explicit memory budget
///
/// Use this when you want to control memory usage explicitly.
/// memory_budget_gb: Maximum memory to use for sparse memory pool (in GB)
pub fn evaluate_candidates_parallel_hybrid_with_budget(
    candidates_flat: &[i64],
    num_candidates: usize,
    conn_size_per_candidate: usize,
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    tier_configs: &[(usize, usize, usize)],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
    memory_budget_gb: f64,
) -> Vec<f64> {
    evaluate_gpu_batch_adaptive(
        candidates_flat,
        num_candidates,
        conn_size_per_candidate,
        train_input_bits,
        train_true_clusters,
        train_false_clusters,
        eval_input_bits,
        eval_targets,
        tier_configs,
        num_train_examples,
        num_eval_examples,
        total_input_bits,
        num_clusters,
        num_negatives,
        Some(memory_budget_gb),
    )
}

/// Evaluate a batch of candidates: train on CPU, batch-evaluate on GPU
/// (Legacy wrapper - now uses memory-adaptive evaluation internally)
#[allow(dead_code)]
fn evaluate_gpu_batch(
    candidates_flat: &[i64],
    num_candidates: usize,
    conn_size_per_candidate: usize,
    train_input_bits: &[bool],
    train_true_clusters: &[i64],
    train_false_clusters: &[i64],
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    tier_configs: &[(usize, usize, usize)],
    num_train_examples: usize,
    num_eval_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
    num_negatives: usize,
) -> Vec<f64> {
    // Delegate to memory-adaptive version
    evaluate_gpu_batch_adaptive(
        candidates_flat,
        num_candidates,
        conn_size_per_candidate,
        train_input_bits,
        train_true_clusters,
        train_false_clusters,
        eval_input_bits,
        eval_targets,
        tier_configs,
        num_train_examples,
        num_eval_examples,
        total_input_bits,
        num_clusters,
        num_negatives,
        None, // Auto-detect memory budget
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_basic() {
        let layer = SparseLayerMemory::new(10, 20);

        // Read unwritten cell should return EMPTY
        assert_eq!(layer.read_cell(0, 0), EMPTY);
        assert_eq!(layer.read_cell(5, 1000), EMPTY);

        // Write TRUE
        assert!(layer.write_cell(0, 0, TRUE, false));
        assert_eq!(layer.read_cell(0, 0), TRUE);

        // Write FALSE (different address)
        assert!(layer.write_cell(0, 1, FALSE, false));
        assert_eq!(layer.read_cell(0, 1), FALSE);

        // Cannot overwrite without allow_override
        assert!(!layer.write_cell(0, 0, FALSE, false));
        assert_eq!(layer.read_cell(0, 0), TRUE);

        // Can overwrite with allow_override
        assert!(layer.write_cell(0, 0, FALSE, true));
        assert_eq!(layer.read_cell(0, 0), FALSE);

        assert_eq!(layer.total_cells(), 2);
    }

    #[test]
    fn test_sparse_parallel() {
        let layer = SparseLayerMemory::new(100, 20);

        // Write to different neurons in parallel
        (0..100usize).into_par_iter().for_each(|i| {
            layer.write_cell(i, i as u64, TRUE, false);
        });

        // Verify all writes
        for i in 0..100 {
            assert_eq!(layer.read_cell(i, i as u64), TRUE);
        }

        assert_eq!(layer.total_cells(), 100);
    }

    #[test]
    fn test_export_import() {
        let layer1 = SparseLayerMemory::new(10, 20);
        layer1.write_cell(0, 100, TRUE, false);
        layer1.write_cell(5, 200, FALSE, false);
        layer1.write_cell(9, 300, TRUE, false);

        let exported = layer1.export();
        assert_eq!(exported.len(), 3);

        let layer2 = SparseLayerMemory::new(10, 20);
        layer2.import(&exported);

        assert_eq!(layer2.read_cell(0, 100), TRUE);
        assert_eq!(layer2.read_cell(5, 200), FALSE);
        assert_eq!(layer2.read_cell(9, 300), TRUE);
    }

    #[test]
    fn test_parallel_candidate_evaluation() {
        // Simple test with 2 candidates, 2 neurons, 2 bits
        let num_candidates = 2;
        let num_neurons = 2;
        let bits_per_neuron = 2;
        let num_clusters = 2;
        let neurons_per_cluster = 1;
        let total_input_bits = 4;
        let num_negatives = 1;

        // Two different connectivity patterns
        let candidates_flat: Vec<i64> = vec![
            0, 1,  // neuron 0: bits 0,1
            2, 3,  // neuron 1: bits 2,3
            1, 2,  // neuron 0: bits 1,2 (different pattern)
            0, 3,  // neuron 1: bits 0,3
        ];

        // Training data
        let train_input_bits: Vec<bool> = vec![
            true, false, true, false,  // example 0
            false, true, false, true,  // example 1
        ];
        let train_true_clusters: Vec<i64> = vec![0, 1];
        let train_false_clusters: Vec<i64> = vec![1, 0];

        // Eval data
        let eval_input_bits: Vec<bool> = vec![
            true, false, true, false,
        ];
        let eval_targets: Vec<i64> = vec![0];

        let results = evaluate_candidates_parallel(
            &candidates_flat,
            num_candidates,
            &train_input_bits,
            &train_true_clusters,
            &train_false_clusters,
            &eval_input_bits,
            &eval_targets,
            2,  // num_train
            1,  // num_eval
            total_input_bits,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            num_negatives,
        );

        assert_eq!(results.len(), 2);
        // Both should return valid cross-entropy values
        assert!(results[0] >= 0.0);
        assert!(results[1] >= 0.0);
    }
}
