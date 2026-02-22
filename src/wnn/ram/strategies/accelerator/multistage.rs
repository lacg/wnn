//! Multi-Stage Token Cache — Pre-encoded data for multi-stage RAM evaluation.
//!
//! Splits token prediction into:
//!   Stage 0: Predict token-group (cluster_id) from context
//!   Stage 1: Predict token within group from context (+ optional cluster info)
//!
//! P(token) = P(group | context) × P(token_in_group | context, group)
//!
//! Stage 0 reuses the bitwise evaluation pipeline (train_into + forward_eval_into + CE)
//! with cluster_id as the output instead of token_id.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bitwise_ramlm::{
    bits_needed, pack_input_bits,
    BitwiseSubset, BitwiseEvalSubset,
    evaluate_genomes_with_params,
    train_and_get_scores,
};

/// Two-stage token cache with pre-encoded data for both stages.
///
/// Created once per experiment configuration. Holds:
/// - Cluster mapping (token → group)
/// - Stage 1 data: context → cluster_id bits (bitwise prediction)
/// - Stage 2 selector data: K separate per-group datasets (context-only input)
/// - Stage 2 concat data: single dataset with wider input (context + cluster_id bits)
pub struct MultiStageTokenCache {
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

    // ── Stage 2: within-group prediction ─────────────────────────────
    /// Bit patterns for within-group indices: [max_cluster_size * bits_per_within_index] (MSB-first)
    pub stage2_within_bits: Vec<u8>,

    // Stage 2 — selector mode: K separate datasets (context-only input)
    pub stage2_selector_train: Vec<BitwiseSubset>,     // [K] one per group
    pub stage2_selector_eval: Vec<BitwiseEvalSubset>,  // [K] one per group

    // Stage 2 — input_concat mode: single dataset, wider input (context + cluster_id bits)
    pub stage2_concat_input_bits: usize,               // context_input_bits + bits_per_cluster_id
    pub stage2_concat_full_train: BitwiseSubset,
    pub stage2_concat_full_eval: BitwiseEvalSubset,
    pub stage2_concat_train_subsets: Vec<BitwiseSubset>,
    pub stage2_concat_eval_subsets: Vec<BitwiseEvalSubset>,

    // ── Vocab-level (for combined CE in Sprint 3) ────────────────────
    pub vocab_size: usize,
    /// Original token_ids for full eval examples (needed for combined CE lookups).
    pub eval_target_token_ids: Vec<u32>,

    // ── Rotation state ───────────────────────────────────────────────
    pub num_parts: usize,
    pub num_eval_parts: usize,
    current_train_idx: AtomicUsize,
    current_eval_idx: AtomicUsize,
}

impl MultiStageTokenCache {
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

        // ── Build Stage 2 within_bits table: [max_cluster_size * bits_per_within_index] ──
        let mut stage2_within_bits = vec![0u8; max_cluster_size * bits_per_within_index];
        for idx in 0..max_cluster_size {
            for b in 0..bits_per_within_index {
                stage2_within_bits[idx * bits_per_within_index + b] =
                    ((idx >> (bits_per_within_index - 1 - b)) & 1) as u8;
            }
        }

        // ── Encode Stage 2 selector data (K per-group datasets) ──────
        let stage2_selector_train: Vec<BitwiseSubset> = (0..k)
            .map(|g| {
                Self::encode_stage2_selector(
                    &train_tokens, context_size, bits_per_token,
                    context_input_bits, bits_per_within_index,
                    &cluster_of, &index_in_cluster, g,
                    true, // is_train
                ).0
            })
            .collect();

        let stage2_selector_eval: Vec<BitwiseEvalSubset> = (0..k)
            .map(|g| {
                Self::encode_stage2_selector(
                    &eval_tokens, context_size, bits_per_token,
                    context_input_bits, bits_per_within_index,
                    &cluster_of, &index_in_cluster, g,
                    false, // is_eval
                ).1
            })
            .collect();

        // ── Encode Stage 2 concat data (single wider-input dataset) ──
        let stage2_concat_input_bits = context_input_bits + bits_per_cluster_id;

        let (concat_full_input, _concat_full_targets, concat_full_tb, concat_full_n) =
            Self::encode_stage2_concat_sequence(
                &train_tokens, context_size, bits_per_token,
                context_input_bits, bits_per_cluster_id, bits_per_within_index,
                &cluster_of, &index_in_cluster,
            );
        let (concat_full_packed, concat_full_wpe) =
            pack_input_bits(&concat_full_input, concat_full_n, stage2_concat_input_bits);
        drop(concat_full_input);
        let stage2_concat_full_train = BitwiseSubset {
            input_bits: Vec::new(),
            packed_input: concat_full_packed,
            target_bits: concat_full_tb,
            num_examples: concat_full_n,
            words_per_example: concat_full_wpe,
        };

        let stage2_concat_train_subsets = Self::split_and_encode_stage2_concat(
            &train_tokens, context_size, bits_per_token,
            context_input_bits, bits_per_cluster_id, bits_per_within_index,
            &cluster_of, &index_in_cluster, num_parts, true,
        );

        let (concat_eval_input, concat_eval_targets, _, concat_eval_n) =
            Self::encode_stage2_concat_sequence(
                &eval_tokens, context_size, bits_per_token,
                context_input_bits, bits_per_cluster_id, bits_per_within_index,
                &cluster_of, &index_in_cluster,
            );
        let (concat_eval_packed, concat_eval_wpe) =
            pack_input_bits(&concat_eval_input, concat_eval_n, stage2_concat_input_bits);
        drop(concat_eval_input);
        let stage2_concat_full_eval = BitwiseEvalSubset {
            packed_input: concat_eval_packed,
            targets: concat_eval_targets,
            num_examples: concat_eval_n,
            words_per_example: concat_eval_wpe,
        };

        let stage2_concat_eval_subsets = Self::split_and_encode_stage2_concat_eval(
            &eval_tokens, context_size, bits_per_token,
            context_input_bits, bits_per_cluster_id, bits_per_within_index,
            &cluster_of, &index_in_cluster, num_eval_parts,
        );

        // ── Build eval target token_ids for combined CE ──────────────
        let eval_target_token_ids: Vec<u32> = (0..full_eval_n)
            .map(|i| eval_tokens[i + context_size])
            .collect();

        // ── Logging ──────────────────────────────────────────────────
        let s1_eval_n: usize = stage1_eval_subsets.iter().map(|s| s.num_examples).sum();
        let sel_train_n: usize = stage2_selector_train.iter().map(|s| s.num_examples).sum();
        let sel_eval_n: usize = stage2_selector_eval.iter().map(|s| s.num_examples).sum();
        eprintln!(
            "[MultiStageCache] K={k}, vocab={vocab_size}, ctx={context_size}, \
             s1_bits={bits_per_cluster_id}, s2_bits={bits_per_within_index}, \
             max_group={max_cluster_size}, concat_input={stage2_concat_input_bits}b"
        );
        eprintln!(
            "  Stage1: train={full_n} ({num_parts} parts), eval={full_eval_n} ({num_eval_parts} parts, {s1_eval_n} total)"
        );
        eprintln!(
            "  Stage2 selector: {k} groups, train={sel_train_n} total, eval={sel_eval_n} total"
        );
        eprintln!(
            "  Stage2 concat: train={concat_full_n} ({num_parts} parts), eval={concat_eval_n} ({num_eval_parts} parts)"
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
            stage2_within_bits,
            stage2_selector_train,
            stage2_selector_eval,
            stage2_concat_input_bits,
            stage2_concat_full_train,
            stage2_concat_full_eval,
            stage2_concat_train_subsets,
            stage2_concat_eval_subsets,
            vocab_size,
            eval_target_token_ids,
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

    // ── Stage 2 selector encoding ───────────────────────────────────

    /// Encode a per-group dataset for selector mode.
    ///
    /// Filters examples where cluster_of[target] == group_id.
    /// Input = context bits only (same width as Stage 1).
    /// Target = index_in_cluster[target].
    ///
    /// Returns (train_subset, eval_subset) — caller picks one based on is_train.
    fn encode_stage2_selector(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_within_index: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        group_id: usize,
        is_train: bool,
    ) -> (BitwiseSubset, BitwiseEvalSubset) {
        if tokens.len() <= context_size {
            let empty_train = BitwiseSubset {
                input_bits: Vec::new(), packed_input: Vec::new(),
                target_bits: Vec::new(), num_examples: 0, words_per_example: 0,
            };
            let empty_eval = BitwiseEvalSubset {
                packed_input: Vec::new(), targets: Vec::new(),
                num_examples: 0, words_per_example: 0,
            };
            return (empty_train, empty_eval);
        }

        // Collect indices of examples belonging to this group
        let total_ex = tokens.len() - context_size;
        let group_indices: Vec<usize> = (0..total_ex)
            .filter(|&i| cluster_of[tokens[i + context_size] as usize] as usize == group_id)
            .collect();
        let num_ex = group_indices.len();

        if num_ex == 0 {
            let empty_train = BitwiseSubset {
                input_bits: Vec::new(), packed_input: Vec::new(),
                target_bits: Vec::new(), num_examples: 0, words_per_example: 0,
            };
            let empty_eval = BitwiseEvalSubset {
                packed_input: Vec::new(), targets: Vec::new(),
                num_examples: 0, words_per_example: 0,
            };
            return (empty_train, empty_eval);
        }

        let mut input_bits = vec![false; num_ex * context_input_bits];
        let mut targets = vec![0u32; num_ex];
        let mut target_bits = vec![0u8; num_ex * bits_per_within_index];

        // Encode filtered examples (parallel over group examples)
        input_bits
            .par_chunks_mut(context_input_bits)
            .zip(targets.par_iter_mut())
            .zip(target_bits.par_chunks_mut(bits_per_within_index))
            .zip(group_indices.par_iter())
            .for_each(|(((inp, tgt), tb), &orig_idx)| {
                // Encode context tokens
                for ctx in 0..context_size {
                    let token = tokens[orig_idx + ctx] as usize;
                    let off = ctx * bits_per_token;
                    for b in 0..bits_per_token {
                        inp[off + b] = ((token >> (bits_per_token - 1 - b)) & 1) == 1;
                    }
                }
                // Target = within-group index
                let target_token = tokens[orig_idx + context_size] as usize;
                let within_idx = index_in_cluster[target_token] as usize;
                *tgt = within_idx as u32;
                for b in 0..bits_per_within_index {
                    tb[b] = ((within_idx >> (bits_per_within_index - 1 - b)) & 1) as u8;
                }
            });

        let (packed, wpe) = pack_input_bits(&input_bits, num_ex, context_input_bits);

        let train = if is_train {
            BitwiseSubset {
                input_bits: Vec::new(),
                packed_input: packed.clone(),
                target_bits,
                num_examples: num_ex,
                words_per_example: wpe,
            }
        } else {
            BitwiseSubset {
                input_bits: Vec::new(), packed_input: Vec::new(),
                target_bits: Vec::new(), num_examples: 0, words_per_example: 0,
            }
        };

        let eval = if !is_train {
            BitwiseEvalSubset {
                packed_input: packed,
                targets,
                num_examples: num_ex,
                words_per_example: wpe,
            }
        } else {
            BitwiseEvalSubset {
                packed_input: Vec::new(), targets: Vec::new(),
                num_examples: 0, words_per_example: 0,
            }
        };

        (train, eval)
    }

    // ── Stage 2 concat encoding ──────────────────────────────────────

    /// Encode ALL examples for input_concat mode.
    ///
    /// Input = context bits + cluster_id bits of TRUE target (teacher forcing).
    /// Target = index_in_cluster[target].
    fn encode_stage2_concat_sequence(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_cluster_id: usize,
        bits_per_within_index: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
    ) -> (Vec<bool>, Vec<u32>, Vec<u8>, usize) {
        let total_input_bits = context_input_bits + bits_per_cluster_id;

        if tokens.len() <= context_size {
            return (vec![], vec![], vec![], 0);
        }

        let num_ex = tokens.len() - context_size;
        let mut input_bits = vec![false; num_ex * total_input_bits];
        let mut targets = vec![0u32; num_ex];
        let mut target_bits = vec![0u8; num_ex * bits_per_within_index];

        input_bits
            .par_chunks_mut(total_input_bits)
            .zip(targets.par_iter_mut())
            .zip(target_bits.par_chunks_mut(bits_per_within_index))
            .enumerate()
            .for_each(|(i, ((inp, tgt), tb))| {
                // Encode context tokens (same as Stage 1)
                for ctx in 0..context_size {
                    let token = tokens[i + ctx] as usize;
                    let off = ctx * bits_per_token;
                    for b in 0..bits_per_token {
                        inp[off + b] = ((token >> (bits_per_token - 1 - b)) & 1) == 1;
                    }
                }
                // Append cluster_id bits of TRUE target (teacher forcing)
                let target_token = tokens[i + context_size] as usize;
                let cluster_id = cluster_of[target_token] as usize;
                for b in 0..bits_per_cluster_id {
                    inp[context_input_bits + b] =
                        ((cluster_id >> (bits_per_cluster_id - 1 - b)) & 1) == 1;
                }
                // Target = within-group index
                let within_idx = index_in_cluster[target_token] as usize;
                *tgt = within_idx as u32;
                for b in 0..bits_per_within_index {
                    tb[b] = ((within_idx >> (bits_per_within_index - 1 - b)) & 1) as u8;
                }
            });

        (input_bits, targets, target_bits, num_ex)
    }

    /// Split and encode Stage 2 concat subsets (train or eval).
    fn split_and_encode_stage2_concat(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_cluster_id: usize,
        bits_per_within_index: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        num_parts: usize,
        is_train: bool,
    ) -> Vec<BitwiseSubset> where Vec<BitwiseSubset>: Sized {
        // This generic return works for both — but we need separate impls
        // for BitwiseSubset (train) and BitwiseEvalSubset (eval).
        // Use a single function that returns both and pick the one needed.
        let n = tokens.len();
        let part_size = n / num_parts;
        let total_input_bits = context_input_bits + bits_per_cluster_id;

        if is_train {
            (0..num_parts)
                .map(|i| {
                    let start = i * part_size;
                    let end = if i < num_parts - 1 { start + part_size } else { n };
                    let part = &tokens[start..end];
                    let (input_bits, _, target_bits, num_ex) =
                        Self::encode_stage2_concat_sequence(
                            part, context_size, bits_per_token,
                            context_input_bits, bits_per_cluster_id, bits_per_within_index,
                            cluster_of, index_in_cluster,
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
        } else {
            // Return empty vec — caller should use split_and_encode_stage2_concat_eval
            Vec::new()
        }
    }

    /// Split and encode Stage 2 concat eval subsets.
    fn split_and_encode_stage2_concat_eval(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_cluster_id: usize,
        bits_per_within_index: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        num_parts: usize,
    ) -> Vec<BitwiseEvalSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;
        let total_input_bits = context_input_bits + bits_per_cluster_id;

        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let (input_bits, targets, _, num_ex) =
                    Self::encode_stage2_concat_sequence(
                        part, context_size, bits_per_token,
                        context_input_bits, bits_per_cluster_id, bits_per_within_index,
                        cluster_of, index_in_cluster,
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
    cache: &MultiStageTokenCache,
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
    cache: &MultiStageTokenCache,
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

// ── Stage 2 evaluation — selector mode ──────────────────────────────────

/// Evaluate Stage 2 genomes for a single group (selector mode).
///
/// Uses per-group filtered data: only examples where target ∈ group_id.
/// Input = context bits only (same width as Stage 1).
/// CE reconstruction over cluster_sizes[group_id] possible within-group tokens.
///
/// Returns: Vec<(within_ce, within_accuracy, bit_acc)> per genome.
pub fn evaluate_stage2_selector_genomes(
    cache: &MultiStageTokenCache,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    group_id: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold_override: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    let train_subset = &cache.stage2_selector_train[group_id];
    let eval_subset = &cache.stage2_selector_eval[group_id];

    if eval_subset.num_examples == 0 {
        return vec![(0.0, 0.0, 0.0); num_genomes];
    }

    evaluate_genomes_with_params(
        cache.bits_per_within_index,
        &cache.stage2_within_bits,
        cache.cluster_sizes[group_id],
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

// ── Stage 2 evaluation — input_concat mode ──────────────────────────────

/// Evaluate Stage 2 genomes with input_concat mode (subset rotation).
///
/// Uses all examples with wider input: context bits + cluster_id bits.
/// CE reconstruction over max_cluster_size possible within-group indices.
///
/// Returns: Vec<(within_ce, within_accuracy, bit_acc)> per genome.
pub fn evaluate_stage2_concat_genomes(
    cache: &MultiStageTokenCache,
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
    let train_subset = &cache.stage2_concat_train_subsets[train_subset_idx % cache.num_parts];
    let eval_subset = &cache.stage2_concat_eval_subsets[eval_subset_idx % cache.num_eval_parts];
    evaluate_genomes_with_params(
        cache.bits_per_within_index,
        &cache.stage2_within_bits,
        cache.max_cluster_size,
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

/// Evaluate Stage 2 genomes with input_concat mode — full data.
pub fn evaluate_stage2_concat_genomes_full(
    cache: &MultiStageTokenCache,
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
        cache.bits_per_within_index,
        &cache.stage2_within_bits,
        cache.max_cluster_size,
        bits_per_neuron_flat,
        neurons_per_cluster_flat,
        connections_flat,
        num_genomes,
        &cache.stage2_concat_full_train,
        &cache.stage2_concat_full_eval,
        memory_mode,
        neuron_sample_rate,
        rng_seed,
        sparse_threshold_override,
    )
}

// ── Combined CE computation ─────────────────────────────────────────────

/// Compute combined two-stage CE from raw per-bit probability scores.
///
/// Trains both stages, forwards on eval data, then reconstructs the joint distribution:
///   P(token_t) = P(group_g | context) × P(token_t | group_g, context)
///
/// For each eval example:
///   1. Reconstruct log P(group_k) for all K groups from Stage 1 scores
///   2. Normalize via logsumexp → log P_norm(group_k)
///   3. Reconstruct log P(within_j | true_group) for all tokens in the true group
///   4. Normalize via logsumexp → log P_norm(within_j | true_group)
///   5. CE = -log P_norm(true_group) - log P_norm(true_within)
///
/// Uses teacher-forced Stage 2 scores (conditioned on true cluster_id in input).
/// Combined CE = CE_s1 + CE_s2 exactly (by chain rule with per-stage normalization).
///
/// Returns: (combined_ce, combined_accuracy, stage1_ce, stage2_ce)
pub fn compute_combined_ce(
    cache: &MultiStageTokenCache,
    // Stage 1 genome params (single genome)
    s1_bits_per_neuron: &[usize],
    s1_neurons_per_cluster: &[usize],
    s1_connections: &[i64],
    // Stage 2 genome params — input_concat mode (single genome)
    s2_bits_per_neuron: &[usize],
    s2_neurons_per_cluster: &[usize],
    s2_connections: &[i64],
    // Training params
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold: usize,
) -> (f64, f64, f64, f64) {
    let eps = 1e-7f64;

    // Train + forward Stage 1 → raw per-bit probabilities
    let s1_scores = train_and_get_scores(
        s1_connections, s1_bits_per_neuron, s1_neurons_per_cluster,
        cache.bits_per_cluster_id,
        &cache.stage1_full_train, &cache.stage1_full_eval,
        memory_mode, neuron_sample_rate, rng_seed, sparse_threshold,
    );

    // Train + forward Stage 2 (input_concat) → raw per-bit probabilities
    let s2_scores = train_and_get_scores(
        s2_connections, s2_bits_per_neuron, s2_neurons_per_cluster,
        cache.bits_per_within_index,
        &cache.stage2_concat_full_train, &cache.stage2_concat_full_eval,
        memory_mode, neuron_sample_rate, rng_seed.wrapping_add(1), sparse_threshold,
    );

    let num_eval = cache.stage1_full_eval.num_examples;
    let s1_bits = cache.bits_per_cluster_id;
    let s2_bits = cache.bits_per_within_index;
    let k = cache.k;

    // Reconstruct combined CE per example (parallel over examples)
    let results: Vec<(f64, f64, f64, u32)> = (0..num_eval)
        .into_par_iter()
        .map(|ex| {
            // Use original token_id to look up cluster membership
            let token_id = cache.eval_target_token_ids[ex] as usize;
            let true_group = cache.cluster_of[token_id] as usize;
            let true_within = cache.index_in_cluster[token_id] as usize;

            // ── Stage 1: log P(group_k) for all K groups ──
            let s1_base = ex * s1_bits;

            // Precompute log(p_b) and log(1-p_b) for this example's Stage 1 bits
            let mut s1_log_p = vec![0.0f64; s1_bits];
            let mut s1_log_1mp = vec![0.0f64; s1_bits];
            for b in 0..s1_bits {
                let p = (s1_scores[s1_base + b] as f64).clamp(eps, 1.0 - eps);
                s1_log_p[b] = p.ln();
                s1_log_1mp[b] = (1.0 - p).ln();
            }

            // Reconstruct log P_raw(group_k) with online logsumexp
            let mut max_s1 = f64::NEG_INFINITY;
            let mut sum_exp_s1 = 0.0f64;
            let mut log_p_true_group_raw = 0.0f64;
            let mut s1_predicted = 0usize;
            let mut s1_predicted_lp = f64::NEG_INFINITY;

            for gk in 0..k {
                let bit_base = gk * s1_bits;
                let mut log_p = 0.0f64;
                for b in 0..s1_bits {
                    if cache.stage1_cluster_bits[bit_base + b] == 1 {
                        log_p += s1_log_p[b];
                    } else {
                        log_p += s1_log_1mp[b];
                    }
                }

                if gk == true_group {
                    log_p_true_group_raw = log_p;
                }
                if log_p > s1_predicted_lp {
                    s1_predicted_lp = log_p;
                    s1_predicted = gk;
                }

                // Online logsumexp accumulation
                if log_p > max_s1 {
                    sum_exp_s1 = sum_exp_s1 * (max_s1 - log_p).exp() + 1.0;
                    max_s1 = log_p;
                } else {
                    sum_exp_s1 += (log_p - max_s1).exp();
                }
            }

            let log_z1 = max_s1 + sum_exp_s1.ln();
            let log_p_true_group = log_p_true_group_raw - log_z1;

            // ── Stage 2: log P(within_j | true_group) ──
            let s2_base = ex * s2_bits;
            let group_size = cache.cluster_sizes[true_group];

            // Precompute log(p_b) and log(1-p_b) for this example's Stage 2 bits
            let mut s2_log_p = vec![0.0f64; s2_bits];
            let mut s2_log_1mp = vec![0.0f64; s2_bits];
            for b in 0..s2_bits {
                let p = (s2_scores[s2_base + b] as f64).clamp(eps, 1.0 - eps);
                s2_log_p[b] = p.ln();
                s2_log_1mp[b] = (1.0 - p).ln();
            }

            // Reconstruct log P_raw(within_j) with online logsumexp
            let mut max_s2 = f64::NEG_INFINITY;
            let mut sum_exp_s2 = 0.0f64;
            let mut log_p_true_within_raw = 0.0f64;
            let mut s2_predicted = 0usize;
            let mut s2_predicted_lp = f64::NEG_INFINITY;

            for j in 0..group_size {
                let bit_base = j * s2_bits;
                let mut log_p = 0.0f64;
                for b in 0..s2_bits {
                    if cache.stage2_within_bits[bit_base + b] == 1 {
                        log_p += s2_log_p[b];
                    } else {
                        log_p += s2_log_1mp[b];
                    }
                }

                if j == true_within {
                    log_p_true_within_raw = log_p;
                }
                if log_p > s2_predicted_lp {
                    s2_predicted_lp = log_p;
                    s2_predicted = j;
                }

                // Online logsumexp accumulation
                if log_p > max_s2 {
                    sum_exp_s2 = sum_exp_s2 * (max_s2 - log_p).exp() + 1.0;
                    max_s2 = log_p;
                } else {
                    sum_exp_s2 += (log_p - max_s2).exp();
                }
            }

            let log_z2 = max_s2 + sum_exp_s2.ln();
            let log_p_true_within = log_p_true_within_raw - log_z2;

            // ── Combine ──
            let s1_ce = -log_p_true_group;
            let s2_ce = -log_p_true_within;
            let combined_ce = s1_ce + s2_ce;

            let correct = if s1_predicted == true_group && s2_predicted == true_within {
                1u32
            } else {
                0u32
            };

            (combined_ce, s1_ce, s2_ce, correct)
        })
        .collect();

    let total_ce: f64 = results.iter().map(|r| r.0).sum();
    let total_s1_ce: f64 = results.iter().map(|r| r.1).sum();
    let total_s2_ce: f64 = results.iter().map(|r| r.2).sum();
    let total_correct: u32 = results.iter().map(|r| r.3).sum();

    (
        total_ce / num_eval as f64,
        total_correct as f64 / num_eval as f64,
        total_s1_ce / num_eval as f64,
        total_s2_ce / num_eval as f64,
    )
}
