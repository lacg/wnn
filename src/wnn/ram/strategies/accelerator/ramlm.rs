//! RAMLM Accelerator - High-performance training and forward pass for RAM Language Models
//!
//! This module provides Rust acceleration for the core RAMLM operations:
//! - Batch training (write TRUE/FALSE to memory)
//! - Batch forward pass (read memory, compute probabilities)
//!
//! Memory format (matches Python):
//! - memory_words: [num_neurons, words_per_neuron] i64
//! - 31 cells per word (62 bits, 2 bits per cell)
//! - Cell values: FALSE=0, TRUE=1, EMPTY=2

use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering, fence};

/// Memory cell values (2 bits each)
const FALSE: i64 = 0;
const TRUE: i64 = 1;
const EMPTY: i64 = 2;

/// Bit-packing constants
const BITS_PER_CELL: usize = 2;
const CELLS_PER_WORD: usize = 31;  // 62 bits / 2 = 31 cells
const CELL_MASK: i64 = 0b11;  // 2-bit mask

/// Compute memory address for a single neuron given input bits
#[inline]
fn compute_address(
    input_bits: &[bool],
    connections: &[i64],
    bits_per_neuron: usize,
) -> usize {
    let mut address: usize = 0;
    for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
        if input_bits[conn_idx as usize] {
            // MSB-first addressing (matches Python)
            address |= 1 << (bits_per_neuron - 1 - i);
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

/// Write a memory cell value (only if EMPTY or allow_override)
#[inline]
fn write_cell(
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
        // Use Acquire ordering to ensure we see writes from other threads
        let old_word = memory_words[word_offset].load(Ordering::Acquire);
        let old_cell = (old_word >> shift) & CELL_MASK;

        // Only write if EMPTY or allow_override
        if !allow_override && old_cell != EMPTY {
            return false;
        }

        // If value already matches, no need to write
        if old_cell == value {
            return false;
        }

        let new_word = (old_word & !mask) | new_bits;

        match memory_words[word_offset].compare_exchange(
            old_word,
            new_word,
            Ordering::AcqRel,  // Acquire on success, Release semantics for the write
            Ordering::Acquire, // Acquire on failure to see latest value
        ) {
            Ok(_) => return true,
            Err(_) => continue,  // Retry on conflict
        }
    }
}

/// Batch training for RAMLM
///
/// Uses two-phase training to match PyTorch semantics:
/// 1. Phase 1: Write ALL TRUEs (correct answers) in parallel
/// 2. Phase 2: Write ALL FALSEs (negatives) in parallel, skip if already TRUE
///
/// This ensures TRUE values always take priority over FALSE, regardless of
/// which example they came from. This is critical for model quality.
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits] flattened bool array
///   true_clusters: [num_examples] target cluster indices
///   false_clusters_flat: [num_examples * num_negatives] flattened negative cluster indices
///   connections_flat: [num_neurons * bits_per_neuron] flattened connection indices
///   memory_words: [num_neurons * words_per_neuron] flattened memory (mutable)
///
/// Returns: number of cells modified
pub fn train_batch(
    input_bits_flat: &[bool],
    true_clusters: &[i64],
    false_clusters_flat: &[i64],
    connections_flat: &[i64],
    memory_words: &mut [i64],
    num_examples: usize,
    total_input_bits: usize,
    _num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_negatives: usize,
    words_per_neuron: usize,
    allow_override: bool,
) -> usize {
    // Convert memory to atomic for thread-safe writes
    // SAFETY: AtomicI64 has same layout as i64
    let atomic_memory: &[AtomicI64] = unsafe {
        std::slice::from_raw_parts(
            memory_words.as_ptr() as *const AtomicI64,
            memory_words.len(),
        )
    };

    // ========== PHASE 1: Write ALL TRUEs first ==========
    // TRUE writes always succeed (can overwrite FALSE from previous batches)
    // This ensures correct answers take priority over negatives across batches
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

            // TRUE writes use allow_override=true to overwrite FALSE from previous batches
            // This ensures TRUE always takes priority, even across batch boundaries
            if write_cell(atomic_memory, neuron_idx, address, TRUE, words_per_neuron, true) {
                ex_modified += 1;
            }
        }

        ex_modified
    }).sum();

    // Memory fence to ensure ALL Phase 1 writes are visible before Phase 2 starts
    // This is critical on ARM (Apple Silicon) where relaxed ordering can cause issues
    fence(Ordering::SeqCst);

    // ========== PHASE 2: Write ALL FALSEs second ==========
    // FALSE only writes to EMPTY cells (TRUE cells from Phase 1 are preserved)
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

                // write_cell will skip if cell is already TRUE (not EMPTY)
                if write_cell(atomic_memory, neuron_idx, address, FALSE, words_per_neuron, allow_override) {
                    ex_modified += 1;
                }
            }
        }

        ex_modified
    }).sum();

    true_modified + false_modified
}

/// Batch forward pass for RAMLM
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits] flattened bool array
///   connections_flat: [num_neurons * bits_per_neuron] flattened connection indices
///   memory_words: [num_neurons * words_per_neuron] flattened memory
///
/// Returns: [num_examples * num_clusters] flattened probabilities
pub fn forward_batch(
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    memory_words: &[i64],
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
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
                let cell_value = read_cell(memory_words, neuron_idx, address, words_per_neuron);

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

/// Get memory values for specific neurons and addresses (for debugging)
pub fn get_memory_values(
    memory_words: &[i64],
    neuron_indices: &[i64],
    addresses: &[i64],
    words_per_neuron: usize,
) -> Vec<i64> {
    neuron_indices.iter().zip(addresses.iter()).map(|(&neuron_idx, &address)| {
        read_cell(memory_words, neuron_idx as usize, address as usize, words_per_neuron)
    }).collect()
}
