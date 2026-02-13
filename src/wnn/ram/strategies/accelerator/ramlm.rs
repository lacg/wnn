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
//!
//! EMPTY_VALUE configuration:
//! - Set RAMLM_EMPTY_VALUE env var to control EMPTY cell contribution
//! - Default: 0.0 (EMPTY cells don't add to probability)
//! - Old default: 0.5 (EMPTY cells add uncertainty)

use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering, fence};

use crate::neuron_memory::{
    FALSE, TRUE, EMPTY, QUAD_FALSE, QUAD_WEAK_TRUE, QUAD_TRUE, QUAD_WEIGHTS,
    BITS_PER_CELL, CELLS_PER_WORD, CELL_MASK,
};

/// Inline xorshift32 PRNG — returns a float in [0, 1)
#[inline]
fn xorshift32(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state >> 8) as f32 / 16777216.0
}

/// Forward to neuron_memory's unified empty value.
pub fn get_empty_value() -> f32 {
    crate::neuron_memory::get_empty_value()
}

/// Forward to neuron_memory's unified empty value.
pub fn set_empty_value(value: f32) {
    crate::neuron_memory::set_empty_value(value);
}

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

/// Bitwise batch training for BitwiseRAMLM
///
/// Multi-label training: each example trains ALL clusters (one per output bit).
/// For each example, target_bits[ex, cluster] = 1 means TRUE, 0 means FALSE.
///
/// Uses two-phase training (same as train_batch):
/// 1. Phase 1: Write TRUE where target_bit=1 (with override priority)
/// 2. Phase 2: Write FALSE where target_bit=0 (EMPTY cells only)
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits] flattened bool array
///   target_bits_flat: [num_examples * num_clusters] flattened u8 array (0 or 1)
///   connections_flat: [num_neurons * bits_per_neuron] flattened connection indices
///   memory_words: [num_neurons * words_per_neuron] flattened memory (mutable)
///
/// Returns: number of cells modified
pub fn bitwise_train_batch(
    input_bits_flat: &[bool],
    target_bits_flat: &[u8],
    connections_flat: &[i64],
    memory_words: &mut [i64],
    num_examples: usize,
    total_input_bits: usize,
    _num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    _allow_override: bool,
) -> usize {
    // Convert memory to atomic for thread-safe writes
    let atomic_memory: &[AtomicI64] = unsafe {
        std::slice::from_raw_parts(
            memory_words.as_ptr() as *const AtomicI64,
            memory_words.len(),
        )
    };

    let address_space = 1usize << bits_per_neuron;
    let mut total_modified = 0usize;

    // ========== MAJORITY VOTE TRAINING ==========
    // For each (neuron, address), count +1 for TRUE, -1 for FALSE.
    // Write the majority winner to memory. This produces the optimal
    // Bayes classifier per cell, instead of the old two-phase protocol
    // which biased all cells toward TRUE.
    //
    // Process one cluster at a time to keep vote buffer small (~4MB).
    for cluster in 0..num_clusters {
        // Vote array: [neurons_per_cluster × address_space]
        let votes: Vec<AtomicI32> = (0..neurons_per_cluster * address_space)
            .map(|_| AtomicI32::new(0))
            .collect();

        // Accumulate votes in parallel over examples
        (0..num_examples).into_par_iter().for_each(|ex_idx| {
            let input_start = ex_idx * total_input_bits;
            let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];
            let target_bit = target_bits_flat[ex_idx * num_clusters + cluster];
            let vote: i32 = if target_bit == 1 { 1 } else { -1 };

            let start_neuron = cluster * neurons_per_cluster;
            for neuron_offset in 0..neurons_per_cluster {
                let neuron_idx = start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];
                let address = compute_address(input_bits, connections, bits_per_neuron);
                votes[neuron_offset * address_space + address].fetch_add(vote, Ordering::Relaxed);
            }
        });

        // Convert votes to memory cells
        let start_neuron = cluster * neurons_per_cluster;
        for neuron_offset in 0..neurons_per_cluster {
            let neuron_idx = start_neuron + neuron_offset;
            for addr in 0..address_space {
                let v = votes[neuron_offset * address_space + addr].load(Ordering::Relaxed);
                if v > 0 {
                    if write_cell(atomic_memory, neuron_idx, addr, TRUE, words_per_neuron, true) {
                        total_modified += 1;
                    }
                } else if v < 0 {
                    if write_cell(atomic_memory, neuron_idx, addr, FALSE, words_per_neuron, true) {
                        total_modified += 1;
                    }
                }
                // v == 0: leave EMPTY (tied or unvisited)
            }
        }
    }

    total_modified
}

/// Nudge a memory cell via CAS (atomic). Moves cell one step toward target.
///
/// target_true: cell = min(cell + 1, 3)
/// target_false: cell = max(cell - 1, 0)
#[inline]
fn nudge_cell(
    memory_words: &[AtomicI64],
    neuron_idx: usize,
    address: usize,
    target_true: bool,
    words_per_neuron: usize,
) -> bool {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = neuron_idx * words_per_neuron + word_idx;
    let shift = cell_idx * BITS_PER_CELL;
    let mask = CELL_MASK << shift;

    loop {
        let old_word = memory_words[word_offset].load(Ordering::Acquire);
        let old_cell = (old_word >> shift) & CELL_MASK;

        let new_cell = if target_true {
            (old_cell + 1).min(QUAD_TRUE)
        } else {
            (old_cell - 1).max(QUAD_FALSE)
        };

        if new_cell == old_cell {
            return false;
        }

        let new_word = (old_word & !mask) | (new_cell << shift);
        match memory_words[word_offset].compare_exchange(
            old_word, new_word, Ordering::AcqRel, Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(_) => continue,
        }
    }
}

/// Bitwise batch training with 4-state nudging (QUAD modes).
///
/// Each training example nudges cells one step toward the target:
///   bit=1 → cell = min(cell + 1, 3)
///   bit=0 → cell = max(cell - 1, 0)
///
/// Processes one cluster at a time. Uses CAS for thread-safe nudging.
///
/// Args:
///   neuron_sample_rate: 0.0-1.0, probability of training each neuron per example
///   rng_seed: seed for deterministic PRNG
pub fn bitwise_train_batch_nudge(
    input_bits_flat: &[bool],
    target_bits_flat: &[u8],
    connections_flat: &[i64],
    memory_words: &mut [i64],
    num_examples: usize,
    total_input_bits: usize,
    _num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> usize {
    let atomic_memory: &[AtomicI64] = unsafe {
        std::slice::from_raw_parts(
            memory_words.as_ptr() as *const AtomicI64,
            memory_words.len(),
        )
    };

    let mut total_modified = 0usize;

    for cluster in 0..num_clusters {
        // Process examples in parallel, each nudging cells
        let cluster_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
            let input_start = ex_idx * total_input_bits;
            let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];
            let target_bit = target_bits_flat[ex_idx * num_clusters + cluster];
            let target_true = target_bit == 1;

            // Per-thread PRNG seeded from (rng_seed, cluster, ex_idx)
            let mut rng_state = (rng_seed as u32)
                .wrapping_add(cluster as u32 * 1000003)
                .wrapping_add(ex_idx as u32 * 999983);
            if rng_state == 0 { rng_state = 1; }

            let start_neuron = cluster * neurons_per_cluster;
            let mut ex_modified = 0usize;

            for neuron_offset in 0..neurons_per_cluster {
                // Probabilistic sampling
                if neuron_sample_rate < 1.0 {
                    if xorshift32(&mut rng_state) >= neuron_sample_rate {
                        continue;
                    }
                }

                let neuron_idx = start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];
                let address = compute_address(input_bits, connections, bits_per_neuron);

                if nudge_cell(atomic_memory, neuron_idx, address, target_true, words_per_neuron) {
                    ex_modified += 1;
                }
            }

            ex_modified
        }).sum();

        total_modified += cluster_modified;
    }

    total_modified
}

/// Non-atomic cell read (for neuron-parallel where each thread owns its memory region)
#[inline]
fn read_cell_direct(memory_words: &[i64], word_offset: usize, cell_idx: usize) -> i64 {
    let shift = cell_idx * BITS_PER_CELL;
    (memory_words[word_offset] >> shift) & CELL_MASK
}

/// Non-atomic cell write (for neuron-parallel)
#[inline]
fn write_cell_direct(memory_words: &mut [i64], word_offset: usize, cell_idx: usize, value: i64) {
    let shift = cell_idx * BITS_PER_CELL;
    let mask = CELL_MASK << shift;
    memory_words[word_offset] = (memory_words[word_offset] & !mask) | (value << shift);
}

/// Neuron-parallel training — parallelizes over neurons, no atomics needed.
///
/// Much faster than example-parallel (bitwise_train_batch_nudge) because:
/// - No CAS atomic overhead
/// - No contention between threads
/// - Better cache locality (each thread works on one neuron's memory)
///
/// Supports all three memory modes:
/// - mode 0 (TERNARY): majority vote per (neuron, address)
/// - mode 1/2 (QUAD): sequential nudging per (neuron, address)
///
/// Returns: number of cells modified
pub fn bitwise_train_neuron_parallel(
    input_bits_flat: &[bool],
    target_bits_flat: &[u8],
    connections_flat: &[i64],
    memory_words: &mut [i64],
    num_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> usize {
    let address_space = 1usize << bits_per_neuron;

    let total_modified: usize = memory_words
        .par_chunks_mut(words_per_neuron)
        .enumerate()
        .map(|(neuron_idx, neuron_mem)| {
        let cluster = neuron_idx / neurons_per_cluster;

        // Per-neuron PRNG for sampling
        let mut rng_state = (rng_seed as u32)
            .wrapping_add(neuron_idx as u32 * 1000003);
        if rng_state == 0 { rng_state = 1; }

        let conn_start = neuron_idx * bits_per_neuron;
        let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];

        let mut modified = 0usize;

        match memory_mode {
            0 => {
                // TERNARY: accumulate votes, then write winners
                let mut votes = vec![0i32; address_space];

                for ex_idx in 0..num_examples {
                    if neuron_sample_rate < 1.0 {
                        if xorshift32(&mut rng_state) >= neuron_sample_rate {
                            continue;
                        }
                    }
                    let input_start = ex_idx * total_input_bits;
                    let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];
                    let target_bit = target_bits_flat[ex_idx * num_clusters + cluster];

                    let addr = compute_address(input_bits, connections, bits_per_neuron);
                    votes[addr] += if target_bit == 1 { 1 } else { -1 };
                }

                // Write winners to memory
                for addr in 0..address_space {
                    let v = votes[addr];
                    if v != 0 {
                        let word_idx = addr / CELLS_PER_WORD;
                        let cell_idx = addr % CELLS_PER_WORD;
                        let value = if v > 0 { TRUE } else { FALSE };
                        write_cell_direct(neuron_mem, word_idx, cell_idx, value);
                        modified += 1;
                    }
                }
            }
            1 | 2 => {
                // QUAD modes: sequential nudging
                for ex_idx in 0..num_examples {
                    if neuron_sample_rate < 1.0 {
                        if xorshift32(&mut rng_state) >= neuron_sample_rate {
                            continue;
                        }
                    }
                    let input_start = ex_idx * total_input_bits;
                    let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];
                    let target_bit = target_bits_flat[ex_idx * num_clusters + cluster];
                    let target_true = target_bit == 1;

                    let addr = compute_address(input_bits, connections, bits_per_neuron);
                    let word_idx = addr / CELLS_PER_WORD;
                    let cell_idx = addr % CELLS_PER_WORD;
                    let old_cell = read_cell_direct(neuron_mem, word_idx, cell_idx);

                    let new_cell = if target_true {
                        (old_cell + 1).min(QUAD_TRUE)
                    } else {
                        (old_cell - 1).max(QUAD_FALSE)
                    };

                    if new_cell != old_cell {
                        write_cell_direct(neuron_mem, word_idx, cell_idx, new_cell);
                        modified += 1;
                    }
                }
            }
            _ => {}
        }

        modified
    }).sum();

    total_modified
}

/// Complete bitwise train + forward + Metal CE evaluation in one Rust call.
///
/// Eliminates all PyTorch overhead:
/// 1. Training: neuron-parallel CPU (rayon, no atomics)
/// 2. Forward: example-parallel CPU (rayon)
/// 3. Reconstruction + CE: Metal GPU
///
/// Returns: (ce, accuracy, per_bit_accuracy_flat)
pub fn bitwise_train_and_eval_full(
    train_input_bits: &[bool],
    train_target_bits: &[u8],
    eval_input_bits: &[bool],
    eval_targets: &[u32],
    token_bits: &[u8],
    connections_flat: &[i64],
    memory_words: &mut [i64],
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
) -> (f64, f64, Vec<f32>) {
    use crate::bitwise_ramlm;

    // 1. Train (neuron-parallel CPU)
    let _modified = bitwise_train_neuron_parallel(
        train_input_bits, train_target_bits, connections_flat, memory_words,
        num_train, total_input_bits, bits_per_neuron, neurons_per_cluster,
        num_clusters, words_per_neuron, memory_mode, neuron_sample_rate, rng_seed,
    );

    // 2. Forward pass (example-parallel CPU)
    let bit_scores = match memory_mode {
        0 => forward_batch(
            eval_input_bits, connections_flat, memory_words,
            num_eval, total_input_bits, 0, bits_per_neuron,
            neurons_per_cluster, num_clusters, words_per_neuron,
        ),
        1 => forward_batch_quad_binary(
            eval_input_bits, connections_flat, memory_words,
            num_eval, total_input_bits, 0, bits_per_neuron,
            neurons_per_cluster, num_clusters, words_per_neuron,
        ),
        _ => forward_batch_quad_weighted(
            eval_input_bits, connections_flat, memory_words,
            num_eval, total_input_bits, 0, bits_per_neuron,
            neurons_per_cluster, num_clusters, words_per_neuron,
        ),
    };

    // 3. Per-bit accuracy (CPU, cheap)
    let mut per_bit_correct = vec![0u64; num_clusters];
    for ex in 0..num_eval {
        let target_id = eval_targets[ex] as usize;
        for b in 0..num_clusters {
            let predicted = bit_scores[ex * num_clusters + b] > 0.5;
            let actual = token_bits[target_id * num_clusters + b] == 1;
            if predicted == actual {
                per_bit_correct[b] += 1;
            }
        }
    }
    let per_bit_accuracy: Vec<f32> = per_bit_correct.iter()
        .map(|&c| c as f32 / num_eval as f32)
        .collect();

    // 4. Reconstruction + CE (Metal GPU)
    #[cfg(target_os = "macos")]
    if let Some(metal_ce) = bitwise_ramlm::get_metal_bitwise_ce() {
        match metal_ce.compute_ce_batch(
            &bit_scores, token_bits, eval_targets,
            1, num_eval, num_clusters, vocab_size,
        ) {
            Ok(results) => {
                let (ce, acc) = results[0];
                return (ce, acc, per_bit_accuracy);
            }
            Err(e) => {
                eprintln!("[bitwise_train_and_eval_full] Metal CE failed: {e}, falling back to CPU");
            }
        }
    }

    // CPU fallback for reconstruction + CE
    let mut total_ce = 0.0f64;
    let mut total_correct = 0u64;
    let eps: f64 = 1e-7;

    for ex in 0..num_eval {
        let scores = &bit_scores[ex * num_clusters..(ex + 1) * num_clusters];
        let target_id = eval_targets[ex] as usize;

        // Reconstruct log-probs for all vocab tokens
        let mut max_lp = f64::NEG_INFINITY;
        let mut target_lp = f64::NEG_INFINITY;
        let mut best_token = 0usize;
        let mut log_sum = 0.0f64;

        // Two-pass: find max log-prob, then compute log-sum-exp
        let mut log_probs_buf = vec![0.0f64; vocab_size];
        for t in 0..vocab_size {
            let mut lp = 0.0f64;
            for b in 0..num_clusters {
                let p = (scores[b] as f64).clamp(eps, 1.0 - eps);
                let bit = token_bits[t * num_clusters + b];
                lp += if bit == 1 { p.ln() } else { (1.0 - p).ln() };
            }
            log_probs_buf[t] = lp;
            if lp > max_lp {
                max_lp = lp;
                best_token = t;
            }
            if t == target_id {
                target_lp = lp;
            }
        }

        // Log-sum-exp for CE
        for t in 0..vocab_size {
            log_sum += (log_probs_buf[t] - max_lp).exp();
        }
        let ce = (max_lp + log_sum.ln()) - target_lp;
        total_ce += ce;
        if best_token == target_id {
            total_correct += 1;
        }
    }

    (total_ce / num_eval as f64, total_correct as f64 / num_eval as f64, per_bit_accuracy)
}

/// Forward pass for QUAD_BINARY mode: P = count(cell >= 2) / neurons_per_cluster
pub fn forward_batch_quad_binary(
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    memory_words: &[i64],
    num_examples: usize,
    total_input_bits: usize,
    _num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
) -> Vec<f32> {
    let mut probs = vec![0.0f32; num_examples * num_clusters];

    probs.par_chunks_mut(num_clusters).enumerate().for_each(|(ex_idx, ex_probs)| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        for cluster_idx in 0..num_clusters {
            let start_neuron = cluster_idx * neurons_per_cluster;
            let mut count_true = 0u32;

            for neuron_offset in 0..neurons_per_cluster {
                let neuron_idx = start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];
                let address = compute_address(input_bits, connections, bits_per_neuron);
                let cell_value = read_cell(memory_words, neuron_idx, address, words_per_neuron);

                if cell_value >= QUAD_WEAK_TRUE {
                    count_true += 1;
                }
            }

            ex_probs[cluster_idx] = count_true as f32 / neurons_per_cluster as f32;
        }
    });

    probs
}

/// Forward pass for QUAD_WEIGHTED mode: P = sum(weight[cell]) / neurons_per_cluster
pub fn forward_batch_quad_weighted(
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    memory_words: &[i64],
    num_examples: usize,
    total_input_bits: usize,
    _num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
) -> Vec<f32> {
    let mut probs = vec![0.0f32; num_examples * num_clusters];

    probs.par_chunks_mut(num_clusters).enumerate().for_each(|(ex_idx, ex_probs)| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        for cluster_idx in 0..num_clusters {
            let start_neuron = cluster_idx * neurons_per_cluster;
            let mut weighted_sum = 0.0f32;

            for neuron_offset in 0..neurons_per_cluster {
                let neuron_idx = start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];
                let address = compute_address(input_bits, connections, bits_per_neuron);
                let cell_value = read_cell(memory_words, neuron_idx, address, words_per_neuron);

                weighted_sum += QUAD_WEIGHTS[cell_value as usize];
            }

            ex_probs[cluster_idx] = weighted_sum / neurons_per_cluster as f32;
        }
    });

    probs
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
    _num_neurons: usize,
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

            // Probability = (count_true + EMPTY_VALUE * count_empty) / neurons_per_cluster
            let empty_value = get_empty_value();
            ex_probs[cluster_idx] = (count_true as f32 + empty_value * count_empty as f32) / neurons_per_cluster as f32;
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

/// Tier configuration for tiered training
#[derive(Clone, Copy)]
pub struct TierConfig {
    pub cluster_start: usize,      // First global cluster index for this tier
    pub cluster_end: usize,        // Last+1 global cluster index for this tier
    pub neurons_per_cluster: usize,
    pub bits_per_neuron: usize,
    pub words_per_neuron: usize,
    pub memory_offset: usize,      // Offset into flattened memory array
    pub conn_offset: usize,        // Offset into flattened connections array
}

/// Batch training for tiered RAMLM - processes ALL tiers in a single Rust call
///
/// This eliminates Python loop overhead by handling tier assignment and training
/// entirely in Rust with full parallelization.
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits] flattened bool array
///   true_clusters: [num_examples] global cluster indices
///   false_clusters_flat: [num_examples * num_negatives] global cluster indices
///   connections_flat: All tiers' connections concatenated [tier0_conns..., tier1_conns..., ...]
///   memory_words_flat: All tiers' memory concatenated [tier0_mem..., tier1_mem..., ...]
///   tier_configs: [(cluster_start, cluster_end, neurons_per_cluster, bits_per_neuron,
///                   words_per_neuron, memory_offset, conn_offset), ...]
///
/// Returns: (total_modified, updated_memory_words_flat)
pub fn train_batch_tiered(
    input_bits_flat: &[bool],
    true_clusters: &[i64],
    false_clusters_flat: &[i64],
    connections_flat: &[i64],
    memory_words: &mut [i64],
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
    tier_configs: &[TierConfig],
    allow_override: bool,
) -> usize {
    let _num_tiers = tier_configs.len();

    // Build lookup table: global_cluster -> (tier_idx, local_cluster)
    let max_cluster = tier_configs.iter().map(|t| t.cluster_end).max().unwrap_or(0);
    let mut cluster_to_tier: Vec<(usize, usize)> = vec![(0, 0); max_cluster];
    for (tier_idx, tc) in tier_configs.iter().enumerate() {
        for global_cluster in tc.cluster_start..tc.cluster_end {
            let local_cluster = global_cluster - tc.cluster_start;
            cluster_to_tier[global_cluster] = (tier_idx, local_cluster);
        }
    }

    // Convert memory to atomic for thread-safe writes
    let atomic_memory: &[AtomicI64] = unsafe {
        std::slice::from_raw_parts(
            memory_words.as_ptr() as *const AtomicI64,
            memory_words.len(),
        )
    };

    // ========== PHASE 1: Write ALL TRUEs first ==========
    let true_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let mut ex_modified = 0usize;

        let true_cluster = true_clusters[ex_idx] as usize;
        if true_cluster >= max_cluster {
            return 0;
        }

        let (tier_idx, local_cluster) = cluster_to_tier[true_cluster];
        let tc = &tier_configs[tier_idx];

        let true_start_neuron = local_cluster * tc.neurons_per_cluster;

        for neuron_offset in 0..tc.neurons_per_cluster {
            let local_neuron_idx = true_start_neuron + neuron_offset;
            let conn_start = tc.conn_offset + local_neuron_idx * tc.bits_per_neuron;
            let connections = &connections_flat[conn_start..conn_start + tc.bits_per_neuron];

            let address = compute_address(input_bits, connections, tc.bits_per_neuron);

            // Compute global memory position
            let mem_neuron_offset = tc.memory_offset + local_neuron_idx * tc.words_per_neuron;

            if write_cell_offset(atomic_memory, mem_neuron_offset, address, TRUE, tc.words_per_neuron, true) {
                ex_modified += 1;
            }
        }

        ex_modified
    }).sum();

    fence(Ordering::SeqCst);

    // ========== PHASE 2: Write ALL FALSEs second ==========
    let false_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let mut ex_modified = 0usize;

        let false_start = ex_idx * num_negatives;
        for neg_idx in 0..num_negatives {
            let false_cluster = false_clusters_flat[false_start + neg_idx] as usize;
            if false_cluster >= max_cluster {
                continue;
            }

            let (tier_idx, local_cluster) = cluster_to_tier[false_cluster];
            let tc = &tier_configs[tier_idx];

            let false_start_neuron = local_cluster * tc.neurons_per_cluster;

            for neuron_offset in 0..tc.neurons_per_cluster {
                let local_neuron_idx = false_start_neuron + neuron_offset;
                let conn_start = tc.conn_offset + local_neuron_idx * tc.bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + tc.bits_per_neuron];

                let address = compute_address(input_bits, connections, tc.bits_per_neuron);

                let mem_neuron_offset = tc.memory_offset + local_neuron_idx * tc.words_per_neuron;

                if write_cell_offset(atomic_memory, mem_neuron_offset, address, FALSE, tc.words_per_neuron, allow_override) {
                    ex_modified += 1;
                }
            }
        }

        ex_modified
    }).sum();

    true_modified + false_modified
}

/// Write cell with pre-computed memory offset (for tiered training)
#[inline]
fn write_cell_offset(
    memory_words: &[AtomicI64],
    mem_neuron_offset: usize,
    address: usize,
    value: i64,
    _words_per_neuron: usize,
    allow_override: bool,
) -> bool {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = mem_neuron_offset + word_idx;

    if word_offset >= memory_words.len() {
        return false;
    }

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
            old_word,
            new_word,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(_) => continue,
        }
    }
}
