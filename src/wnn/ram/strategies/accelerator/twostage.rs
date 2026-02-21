//! Two-Stage Token Cache — Pre-encoded data for two-stage RAM evaluation.
//!
//! Splits token prediction into:
//!   Stage 1: Predict token-group (cluster_id) from context
//!   Stage 2: Predict token within group from context (+ optional cluster info)
//!
//! P(token) = P(group | context) × P(token_in_group | context, group)
//!
//! Stage 1 reuses the bitwise evaluation pipeline (train_into + forward_eval_into + CE)
//! with cluster_id as the output instead of token_id.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bitwise_ramlm::{
    bits_needed, pack_input_bits,
    BitwiseSubset, BitwiseEvalSubset,
    evaluate_genomes_with_params,
};

/// Two-stage token cache with pre-encoded data for both stages.
///
/// Created once per experiment configuration. Holds:
/// - Cluster mapping (token → group)
/// - Stage 1 data: context → cluster_id bits (bitwise prediction)
/// - Stage 2 data: added in Sprint 2
pub struct TwoStageTokenCache {
    // ── Cluster mapping ──────────────────────────────────────────────
    pub k: usize,                          // Number of token-groups
    pub cluster_of: Vec<u16>,              // [vocab_size] → group_id (0..K-1)
    pub index_in_cluster: Vec<u16>,        // [vocab_size] → within-group index
    pub cluster_sizes: Vec<usize>,         // [K] → tokens per group
    pub max_cluster_size: usize,

    // ── Bit dimensions ───────────────────────────────────────────────
    pub bits_per_cluster_id: usize,        // ceil(log2(K))
    pub bits_per_within_index: usize,      // ceil(log2(max_cluster_size))
    pub context_size: usize,
    pub bits_per_token: usize,             // 16 (for context encoding)
    pub context_input_bits: usize,         // context_size * bits_per_token

    // ── Stage 1: cluster prediction (bitwise) ────────────────────────
    /// Bit patterns for all cluster IDs: [K * bits_per_cluster_id] (MSB-first)
    pub stage1_cluster_bits: Vec<u8>,
    pub stage1_train_subsets: Vec<BitwiseSubset>,
    pub stage1_eval_subsets: Vec<BitwiseEvalSubset>,
    pub stage1_full_train: BitwiseSubset,
    pub stage1_full_eval: BitwiseEvalSubset,

    // ── Vocab-level (for combined CE in Sprint 3) ────────────────────
    pub vocab_size: usize,

    // ── Rotation state ───────────────────────────────────────────────
    pub num_parts: usize,
    pub num_eval_parts: usize,
    current_train_idx: AtomicUsize,
    current_eval_idx: AtomicUsize,
}

impl TwoStageTokenCache {
    pub fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        k: usize,
        num_parts: usize,
        num_eval_parts: usize,
        _pad_token_id: u32,
    ) -> Self {
        let bits_per_token = bits_needed(vocab_size);
        let context_input_bits = context_size * bits_per_token;
        let bits_per_cluster_id = bits_needed(k);

        // ── Build cluster mapping (round-robin by token ID) ──────────
        let mut cluster_of = vec![0u16; vocab_size];
        let mut index_in_cluster = vec![0u16; vocab_size];
        let mut cluster_tokens: Vec<Vec<u32>> = vec![Vec::new(); k];

        for t in 0..vocab_size {
            let group = t % k;
            cluster_of[t] = group as u16;
            index_in_cluster[t] = cluster_tokens[group].len() as u16;
            cluster_tokens[group].push(t as u32);
        }

        let cluster_sizes: Vec<usize> = cluster_tokens.iter().map(|g| g.len()).collect();
        let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0);
        let bits_per_within_index = bits_needed(max_cluster_size);

        // ── Build Stage 1 cluster_bits table: [K * bits_per_cluster_id] (MSB-first) ──
        let mut stage1_cluster_bits = vec![0u8; k * bits_per_cluster_id];
        for c in 0..k {
            for b in 0..bits_per_cluster_id {
                stage1_cluster_bits[c * bits_per_cluster_id + b] =
                    ((c >> (bits_per_cluster_id - 1 - b)) & 1) as u8;
            }
        }

        // ── Encode Stage 1 training data ─────────────────────────────
        let (full_input_bits, _full_targets, full_target_bits, full_n) =
            Self::encode_stage1_sequence(
                &train_tokens, context_size, bits_per_token,
                context_input_bits, bits_per_cluster_id, &cluster_of,
            );
        let (full_packed, full_wpe) = pack_input_bits(&full_input_bits, full_n, context_input_bits);
        drop(full_input_bits);
        let stage1_full_train = BitwiseSubset {
            input_bits: Vec::new(),
            packed_input: full_packed,
            target_bits: full_target_bits,
            num_examples: full_n,
            words_per_example: full_wpe,
        };

        let stage1_train_subsets = Self::split_and_encode_stage1_train(
            &train_tokens, context_size, bits_per_token,
            context_input_bits, bits_per_cluster_id, &cluster_of, num_parts,
        );

        // ── Encode Stage 1 eval data ─────────────────────────────────
        let (full_eval_input_bits, full_eval_targets, _, full_eval_n) =
            Self::encode_stage1_sequence(
                &eval_tokens, context_size, bits_per_token,
                context_input_bits, bits_per_cluster_id, &cluster_of,
            );
        let (full_eval_packed, full_eval_wpe) =
            pack_input_bits(&full_eval_input_bits, full_eval_n, context_input_bits);
        drop(full_eval_input_bits);
        let stage1_full_eval = BitwiseEvalSubset {
            packed_input: full_eval_packed,
            targets: full_eval_targets,
            num_examples: full_eval_n,
            words_per_example: full_eval_wpe,
        };

        let stage1_eval_subsets = Self::split_and_encode_stage1_eval(
            &eval_tokens, context_size, bits_per_token,
            context_input_bits, bits_per_cluster_id, &cluster_of, num_eval_parts,
        );

        let eval_subset_n: usize = stage1_eval_subsets.iter().map(|s| s.num_examples).sum();
        eprintln!(
            "[TwoStageCache] K={k}, vocab={vocab_size}, ctx={context_size}, \
             s1_bits={bits_per_cluster_id}, s2_bits={bits_per_within_index}, \
             max_group={max_cluster_size}, \
             train={full_n} ({num_parts} parts), eval={full_eval_n} ({num_eval_parts} parts, {eval_subset_n} total)"
        );

        Self {
            k,
            cluster_of,
            index_in_cluster,
            cluster_sizes,
            max_cluster_size,
            bits_per_cluster_id,
            bits_per_within_index,
            context_size,
            bits_per_token,
            context_input_bits,
            stage1_cluster_bits,
            stage1_train_subsets,
            stage1_eval_subsets,
            stage1_full_train,
            stage1_full_eval,
            vocab_size,
            num_parts,
            num_eval_parts,
            current_train_idx: AtomicUsize::new(0),
            current_eval_idx: AtomicUsize::new(0),
        }
    }

    // ── Stage 1 encoding ─────────────────────────────────────────────

    /// Encode token sequence for Stage 1: context → cluster_id.
    ///
    /// Same context encoding as BitwiseTokenCache, but:
    /// - target_bits = cluster_id bits (not token bits)
    /// - targets = cluster_id (not token_id)
    fn encode_stage1_sequence(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        bits_per_cluster_id: usize,
        cluster_of: &[u16],
    ) -> (Vec<bool>, Vec<u32>, Vec<u8>, usize) {
        if tokens.len() <= context_size {
            return (vec![], vec![], vec![], 0);
        }

        let num_ex = tokens.len() - context_size;
        let mut input_bits = vec![false; num_ex * total_input_bits];
        let mut targets = vec![0u32; num_ex];
        let mut target_bits = vec![0u8; num_ex * bits_per_cluster_id];

        input_bits
            .par_chunks_mut(total_input_bits)
            .zip(targets.par_iter_mut())
            .zip(target_bits.par_chunks_mut(bits_per_cluster_id))
            .enumerate()
            .for_each(|(i, ((inp, tgt), tb))| {
                // Encode context tokens (identical to BitwiseTokenCache)
                for ctx in 0..context_size {
                    let token = tokens[i + ctx] as usize;
                    let off = ctx * bits_per_token;
                    for b in 0..bits_per_token {
                        inp[off + b] = ((token >> (bits_per_token - 1 - b)) & 1) == 1;
                    }
                }
                // Target = cluster_id of the next token
                let target_token = tokens[i + context_size] as usize;
                let cluster_id = cluster_of[target_token] as usize;
                *tgt = cluster_id as u32;
                // Encode cluster_id bits (MSB-first, matching token_bits encoding)
                for b in 0..bits_per_cluster_id {
                    tb[b] = ((cluster_id >> (bits_per_cluster_id - 1 - b)) & 1) as u8;
                }
            });

        (input_bits, targets, target_bits, num_ex)
    }

    /// Split training tokens and encode each part for Stage 1.
    fn split_and_encode_stage1_train(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        bits_per_cluster_id: usize,
        cluster_of: &[u16],
        num_parts: usize,
    ) -> Vec<BitwiseSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;
        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let (input_bits, _, target_bits, num_ex) =
                    Self::encode_stage1_sequence(
                        part, context_size, bits_per_token,
                        total_input_bits, bits_per_cluster_id, cluster_of,
                    );
                let (packed, wpe) = pack_input_bits(&input_bits, num_ex, total_input_bits);
                BitwiseSubset {
                    input_bits: Vec::new(),
                    packed_input: packed,
                    target_bits,
                    num_examples: num_ex,
                    words_per_example: wpe,
                }
            })
            .collect()
    }

    /// Split eval tokens and encode each part for Stage 1.
    fn split_and_encode_stage1_eval(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        bits_per_cluster_id: usize,
        cluster_of: &[u16],
        num_parts: usize,
    ) -> Vec<BitwiseEvalSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;
        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let (input_bits, targets, _, num_ex) =
                    Self::encode_stage1_sequence(
                        part, context_size, bits_per_token,
                        total_input_bits, bits_per_cluster_id, cluster_of,
                    );
                let (packed, wpe) = pack_input_bits(&input_bits, num_ex, total_input_bits);
                BitwiseEvalSubset {
                    packed_input: packed,
                    targets,
                    num_examples: num_ex,
                    words_per_example: wpe,
                }
            })
            .collect()
    }

    // ── Rotation ─────────────────────────────────────────────────────

    pub fn next_train_idx(&self) -> usize {
        self.current_train_idx.fetch_add(1, Ordering::Relaxed) % self.num_parts
    }

    pub fn next_eval_idx(&self) -> usize {
        self.current_eval_idx.fetch_add(1, Ordering::Relaxed) % self.num_eval_parts
    }

    pub fn reset(&self) {
        self.current_train_idx.store(0, Ordering::Relaxed);
        self.current_eval_idx.store(0, Ordering::Relaxed);
    }
}

// ── Stage 1 evaluation ──────────────────────────────────────────────────

/// Evaluate Stage 1 genomes (cluster prediction).
///
/// Delegates to the generic bitwise evaluation pipeline with:
/// - num_clusters = bits_per_cluster_id (e.g., 8 for K=256)
/// - token_bits = stage1_cluster_bits (cluster-id bit patterns)
/// - vocab_size = K (number of groups)
///
/// Returns: Vec<(cluster_ce, cluster_accuracy, bit_acc)> per genome.
pub fn evaluate_stage1_genomes(
    cache: &TwoStageTokenCache,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold_override: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    let train_subset = &cache.stage1_train_subsets[train_subset_idx % cache.num_parts];
    let eval_subset = &cache.stage1_eval_subsets[eval_subset_idx % cache.num_eval_parts];
    evaluate_genomes_with_params(
        cache.bits_per_cluster_id,
        &cache.stage1_cluster_bits,
        cache.k,
        bits_per_neuron_flat,
        neurons_per_cluster_flat,
        connections_flat,
        num_genomes,
        train_subset,
        eval_subset,
        memory_mode,
        neuron_sample_rate,
        rng_seed,
        sparse_threshold_override,
    )
}

/// Evaluate Stage 1 genomes with full (non-rotated) data.
pub fn evaluate_stage1_genomes_full(
    cache: &TwoStageTokenCache,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold_override: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    evaluate_genomes_with_params(
        cache.bits_per_cluster_id,
        &cache.stage1_cluster_bits,
        cache.k,
        bits_per_neuron_flat,
        neurons_per_cluster_flat,
        connections_flat,
        num_genomes,
        &cache.stage1_full_train,
        &cache.stage1_full_eval,
        memory_mode,
        neuron_sample_rate,
        rng_seed,
        sparse_threshold_override,
    )
}
