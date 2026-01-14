//! Adaptive Architecture Accelerator
//!
//! High-performance training and forward pass for AdaptiveClusteredRAM
//! where each cluster can have its own (bits, neurons) configuration.
//!
//! Key optimization: Clusters are grouped by their config to enable
//! efficient batch processing within each group.

use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

/// Memory cell values (2 bits each)
const FALSE: i64 = 0;
const TRUE: i64 = 1;
const EMPTY: i64 = 2;

/// Bit-packing constants
const BITS_PER_CELL: usize = 2;
const CELLS_PER_WORD: usize = 31;
const CELL_MASK: i64 = 0b11;

/// Get the EMPTY cell value from the global setting
fn get_empty_value() -> f32 {
    crate::ramlm::get_empty_value()
}

/// Configuration group - clusters sharing the same (neurons, bits) config
#[derive(Clone, Debug)]
pub struct ConfigGroup {
    pub neurons: usize,
    pub bits: usize,
    pub words_per_neuron: usize,
    pub cluster_ids: Vec<usize>,      // Global cluster IDs in this group
    pub memory_offset: usize,          // Offset into flattened memory
    pub conn_offset: usize,            // Offset into flattened connections
}

impl ConfigGroup {
    pub fn new(neurons: usize, bits: usize, cluster_ids: Vec<usize>) -> Self {
        let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
        Self {
            neurons,
            bits,
            words_per_neuron,
            cluster_ids,
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

    pub fn memory_size(&self) -> usize {
        self.total_neurons() * self.words_per_neuron
    }

    pub fn conn_size(&self) -> usize {
        self.total_neurons() * self.bits
    }
}

/// Compute memory address for a single neuron given input bits
#[inline]
fn compute_address(input_bits: &[bool], connections: &[i64], bits: usize) -> usize {
    let mut address: usize = 0;
    for (i, &conn_idx) in connections.iter().take(bits).enumerate() {
        if input_bits[conn_idx as usize] {
            address |= 1 << (bits - 1 - i);
        }
    }
    address
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
                let start_neuron = local_idx * neurons;
                let mut count_true = 0u32;
                let mut count_empty = 0u32;

                for neuron_offset in 0..neurons {
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

                ex_probs[global_cluster_id] =
                    (count_true as f32 + empty_value * count_empty as f32) / neurons as f32;
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

        let neurons = group.neurons;
        let bits = group.bits;
        let words_per_neuron = group.words_per_neuron;
        let start_neuron = local_cluster * neurons;
        let group_conns = &connections_flat[group.conn_offset..];

        let mut modified = 0usize;
        for neuron_offset in 0..neurons {
            let local_neuron = start_neuron + neuron_offset;
            let conn_start = local_neuron * bits;
            let connections = &group_conns[conn_start..conn_start + bits];

            let address = compute_address(input_bits, connections, bits);
            let global_neuron_offset = group.memory_offset / words_per_neuron + local_neuron;

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

            let neurons = group.neurons;
            let bits = group.bits;
            let words_per_neuron = group.words_per_neuron;
            let start_neuron = local_cluster * neurons;
            let group_conns = &connections_flat[group.conn_offset..];

            for neuron_offset in 0..neurons {
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

/// Evaluate multiple genomes in parallel using rayon.
///
/// This is the KEY acceleration function for GA optimization.
/// Each genome is evaluated independently with its own memory,
/// all running in parallel across CPU cores.
///
/// Args:
///   genomes_bits_flat: [num_genomes * num_clusters] bits per cluster for each genome
///   genomes_neurons_flat: [num_genomes * num_clusters] neurons per cluster for each genome
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
pub fn evaluate_genomes_parallel(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
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
) -> Vec<f64> {
    use rand::prelude::*;
    use rand::SeedableRng;

    // Evaluate each genome in parallel
    (0..num_genomes).into_par_iter().map(|genome_idx| {
        // Extract this genome's configuration
        let genome_offset = genome_idx * num_clusters;
        let bits_per_cluster: Vec<usize> = genomes_bits_flat[genome_offset..genome_offset + num_clusters].to_vec();
        let neurons_per_cluster: Vec<usize> = genomes_neurons_flat[genome_offset..genome_offset + num_clusters].to_vec();

        // Build config groups for this genome
        let groups = build_config_groups(&bits_per_cluster, &neurons_per_cluster);

        // Calculate total memory size needed
        let total_memory_words: usize = groups.iter().map(|g| g.memory_size()).sum();
        let total_conn_size: usize = groups.iter().map(|g| g.conn_size()).sum();

        // Allocate memory for this genome (initialized to EMPTY = 2)
        let mut memory_words: Vec<i64> = vec![0; total_memory_words];
        // Initialize all cells to EMPTY (pack 31 EMPTY values per word)
        let empty_word: i64 = (0..31).fold(0i64, |acc, i| acc | (EMPTY << (i * 2)));
        for word in &mut memory_words {
            *word = empty_word;
        }

        // Generate random connections for this genome
        let mut rng = rand::rngs::SmallRng::seed_from_u64(genome_idx as u64 + 12345);
        let mut connections_flat: Vec<i64> = Vec::with_capacity(total_conn_size);

        for group in &groups {
            let total_neurons = group.total_neurons();
            for _ in 0..total_neurons {
                for _ in 0..group.bits {
                    connections_flat.push(rng.gen_range(0..total_input_bits as i64));
                }
            }
        }

        // Train this genome
        train_batch_adaptive(
            train_input_bits,
            train_targets,
            train_negatives,
            &connections_flat,
            &mut memory_words,
            &groups,
            num_train,
            total_input_bits,
            num_negatives,
            num_clusters,
            false, // don't allow override
        );

        // Evaluate this genome
        let probs = forward_batch_adaptive(
            eval_input_bits,
            &connections_flat,
            &memory_words,
            &groups,
            num_eval,
            total_input_bits,
            num_clusters,
        );

        // Compute cross-entropy
        let mut total_ce = 0.0f64;
        let epsilon = 1e-10f64;

        for (ex_idx, &target) in eval_targets.iter().enumerate() {
            let target_idx = target as usize;
            let prob_start = ex_idx * num_clusters;

            // Get raw scores and apply softmax
            let scores: Vec<f64> = probs[prob_start..prob_start + num_clusters]
                .iter()
                .map(|&p| p as f64)
                .collect();

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            let target_prob = exp_scores[target_idx] / sum_exp;
            total_ce -= (target_prob + epsilon).ln();
        }

        total_ce / num_eval as f64
    }).collect()
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
