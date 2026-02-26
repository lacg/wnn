//! Multi-Stage Token Cache — Pre-encoded data for multi-stage RAM evaluation.
//!
//! Splits token prediction into N stages:
//!   Stage 0: Predict token-group (cluster_id) from context
//!   Stage 1+: Predict sub-group/token from context + previous stage outputs
//!
//! P(token) = P(group | context) × P(token_in_group | context, group)
//!
//! Each bitwise stage reuses the bitwise evaluation pipeline (train_into + forward_eval_into + CE).
//! Stage-agnostic: all methods take a `stage` index instead of hardcoded stage numbers.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bitwise_ramlm::{
    bits_needed, pack_input_bits,
    BitwiseSubset, BitwiseEvalSubset,
    evaluate_genomes_with_params,
    train_and_get_scores, train_and_get_scores_with_model,
    forward_scores_on_subset, forward_scores_on_eval, train_only,
    compute_genome_layout, train_into, forward_eval_into,
};
use crate::neuron_memory::{
    compute_address, pack_bools_to_u64,
    ClusterStorage,
    TRUE, FALSE,
};
use crate::adaptive::{
    build_groups, build_neuron_metadata, per_cluster_max_bits,
    reorganize_connections_for_gpu, try_gpu_addresses_adaptive,
    train_genome_in_slot, evaluate_group_metal, evaluate_group_sparse_gpu,
    get_metal_evaluator, get_sparse_metal_evaluator,
    GroupMemory,
};

/// Convert memory_mode to the appropriate empty_value for tiered evaluation.
///
/// TERNARY (mode 0): EMPTY cells abstain → 0.0
/// QUAD_WEIGHTED (mode 2): EMPTY cells get baseline weight → 0.25
fn empty_value_for_mode(memory_mode: u8) -> f32 {
    match memory_mode {
        crate::neuron_memory::MODE_QUAD_WEIGHTED => crate::neuron_memory::QUAD_WEIGHTS[1], // 0.25
        crate::neuron_memory::MODE_QUAD_BINARY => 0.0, // QUAD_BINARY: only >=2 counts
        _ => 0.0, // TERNARY: EMPTY abstains
    }
}

/// Training/evaluation data for a tiered stage (direct K-class prediction).
///
/// Unlike BitwiseSubset which stores target_bits (one-hot per bit), this stores
/// direct class indices + random negatives for target+negatives training.
pub struct TieredSubset {
    pub input_bits: Vec<bool>,      // [N × total_input_bits] — for CPU training
    pub packed_input: Vec<u64>,     // [N × words_per_example] — for GPU
    pub targets: Vec<i64>,          // [N] target class index
    pub negatives: Vec<i64>,        // [N × num_negatives]
    pub num_examples: usize,
    pub words_per_example: usize,
    pub num_negatives: usize,
}

/// Evaluation data for a tiered stage.
pub struct TieredEvalSubset {
    pub input_bits: Vec<bool>,
    pub packed_input: Vec<u64>,
    pub targets: Vec<i64>,
    pub num_examples: usize,
    pub words_per_example: usize,
}

/// Multi-stage token cache with pre-encoded data for all stages.
///
/// Created once per experiment configuration. Holds:
/// - Cluster mapping (token → group)
/// - Per-stage bitwise data: token_bits, train/eval subsets
/// - Per-stage tiered data (for tiered stages)
/// - Selector data (per-group datasets, for stages with selector mode)
pub struct MultiStageTokenCache {
    // ── Cluster mapping ──────────────────────────────────────────────
    pub k: usize,                          // Number of token-groups
    pub cluster_of: Vec<u16>,              // [vocab_size] → group_id (0..K-1)
    pub index_in_cluster: Vec<u16>,        // [vocab_size] → within-group index
    pub cluster_sizes: Vec<usize>,         // [K] → tokens per group
    pub max_cluster_size: usize,

    // ── Bit dimensions (stage-agnostic) ──────────────────────────────
    pub context_size: usize,
    pub bits_per_token: usize,             // 16 (for context encoding)
    pub context_input_bits: usize,         // context_size * bits_per_token
    pub bitwise_output_bits: Vec<usize>,   // [num_stages] — target bits per stage
    pub stage_input_bits: Vec<usize>,      // [num_stages] — total input bits per stage
    pub bitwise_vocab_size: Vec<usize>,    // [num_stages] — vocab for CE reconstruction

    // ── Bitwise data per stage ───────────────────────────────────────
    /// Bit patterns for each stage's classes: [num_stages][num_classes × bits]
    pub bitwise_token_bits: Vec<Vec<u8>>,
    pub bitwise_train_subsets: Vec<Vec<BitwiseSubset>>,
    pub bitwise_eval_subsets: Vec<Vec<BitwiseEvalSubset>>,
    pub bitwise_full_train: Vec<BitwiseSubset>,
    pub bitwise_full_eval: Vec<BitwiseEvalSubset>,

    // ── Bitwise selector data (per-group, for stages with selector mode) ──
    /// [num_stages][K] — empty inner Vec for stages without selector
    pub bitwise_selector_train: Vec<Vec<BitwiseSubset>>,
    pub bitwise_selector_eval: Vec<Vec<BitwiseEvalSubset>>,

    // ── Vocab-level ──────────────────────────────────────────────────
    pub vocab_size: usize,
    /// Original token_ids for full eval examples (needed for combined CE lookups).
    pub eval_target_token_ids: Vec<u32>,

    // ── Per-stage cluster type ─────────────────────────────────────
    /// [num_stages] — true if tiered (direct K-class), false if bitwise
    pub stage_is_tiered: Vec<bool>,
    /// [num_stages] — K_s if tiered, ceil(log2(K_s)) if bitwise
    pub stage_num_output_clusters: Vec<usize>,

    // ── Tiered data per stage (only populated for tiered stages) ───
    /// Vec<Option<...>> indexed by stage — None for bitwise stages
    pub tiered_train_subsets: Vec<Option<Vec<TieredSubset>>>,
    pub tiered_eval_subsets: Vec<Option<Vec<TieredEvalSubset>>>,
    pub tiered_full_train: Vec<Option<TieredSubset>>,
    pub tiered_full_eval: Vec<Option<TieredEvalSubset>>,
    /// [num_stages] — per-stage negatives count (0 for bitwise stages)
    pub tiered_num_negatives: Vec<usize>,

    // ── Raw tokens (kept for re-encoding with predicted clusters) ────
    pub train_tokens: Vec<u32>,
    pub eval_tokens: Vec<u32>,

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
        stage_cluster_types: Option<Vec<String>>,
    ) -> Self {
        let bits_per_token = bits_needed(vocab_size);
        let context_input_bits = context_size * bits_per_token;

        // Keep raw tokens for later re-encoding with predicted clusters
        let stored_train_tokens = train_tokens.clone();
        let stored_eval_tokens = eval_tokens.clone();

        // Parse stage cluster types (default: all bitwise)
        let num_stages = 2; // Currently fixed at 2
        let stage_types = stage_cluster_types.unwrap_or_else(|| vec!["bitwise".to_string(); num_stages]);
        let stage_is_tiered: Vec<bool> = stage_types.iter()
            .map(|t| t.to_lowercase() == "tiered")
            .collect();
        let any_tiered = stage_is_tiered.iter().any(|&t| t);

        // ── Build cluster mapping ────────────────────────────────────
        // Tiered: frequency-interleaved (sort by training freq, rank % K)
        // Bitwise: round-robin (token_id % K)
        let mut cluster_of = vec![0u16; vocab_size];
        let mut index_in_cluster = vec![0u16; vocab_size];
        let mut cluster_tokens: Vec<Vec<u32>> = vec![Vec::new(); k];

        if any_tiered {
            // Frequency-interleaved: sort tokens by training frequency, assign by rank
            let mut token_freq = vec![0u64; vocab_size];
            for &t in &train_tokens {
                if (t as usize) < vocab_size {
                    token_freq[t as usize] += 1;
                }
            }
            let mut sorted_tokens: Vec<usize> = (0..vocab_size).collect();
            sorted_tokens.sort_by(|&a, &b| {
                token_freq[b].cmp(&token_freq[a])
                    .then_with(|| a.cmp(&b))
            });
            for (rank, &token_id) in sorted_tokens.iter().enumerate() {
                let group = rank % k;
                cluster_of[token_id] = group as u16;
                index_in_cluster[token_id] = cluster_tokens[group].len() as u16;
                cluster_tokens[group].push(token_id as u32);
            }
        } else {
            for t in 0..vocab_size {
                let group = t % k;
                cluster_of[t] = group as u16;
                index_in_cluster[t] = cluster_tokens[group].len() as u16;
                cluster_tokens[group].push(t as u32);
            }
        }

        let cluster_sizes: Vec<usize> = cluster_tokens.iter().map(|g| g.len()).collect();
        let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0);

        // ── Compute per-stage dimensions ─────────────────────────────
        // Stage 0: predicts cluster_id, output_bits = ceil(log2(K))
        // Stage 1: predicts within_index, output_bits = ceil(log2(max_cluster_size))
        let stage_output_bits_raw = vec![bits_needed(k), bits_needed(max_cluster_size)];
        let stage_vocab = vec![k, max_cluster_size];

        // Input bits: stage 0 = context only, stage s>0 = context + sum(output_bits[0..s])
        let mut stage_input_bits_raw = vec![0usize; num_stages];
        stage_input_bits_raw[0] = context_input_bits;
        for s in 1..num_stages {
            stage_input_bits_raw[s] = context_input_bits + stage_output_bits_raw[..s].iter().sum::<usize>();
        }

        let stage_num_output_clusters: Vec<usize> = (0..num_stages)
            .map(|s| {
                if stage_is_tiered[s] { stage_vocab[s] } else { stage_output_bits_raw[s] }
            })
            .collect();

        // ── Build bit-pattern tables per stage ──────────────────────
        let mut bitwise_token_bits = Vec::with_capacity(num_stages);
        for s in 0..num_stages {
            let n_classes = stage_vocab[s];
            let n_bits = stage_output_bits_raw[s];
            let mut bits = vec![0u8; n_classes * n_bits];
            for c in 0..n_classes {
                for b in 0..n_bits {
                    bits[c * n_bits + b] = ((c >> (n_bits - 1 - b)) & 1) as u8;
                }
            }
            bitwise_token_bits.push(bits);
        }

        // ── Encode bitwise data per stage ────────────────────────────
        let mut bitwise_train_subsets = Vec::with_capacity(num_stages);
        let mut bitwise_eval_subsets = Vec::with_capacity(num_stages);
        let mut bitwise_full_train = Vec::with_capacity(num_stages);
        let mut bitwise_full_eval = Vec::with_capacity(num_stages);

        for s in 0..num_stages {
            let total_input = stage_input_bits_raw[s];
            let target_bits_count = stage_output_bits_raw[s];
            let prev_output_bits = &stage_output_bits_raw[..s];

            // Full train
            let (input_bits, _targets, target_bits, n) =
                Self::encode_bitwise_stage_sequence(
                    &train_tokens, context_size, bits_per_token,
                    context_input_bits, s, total_input, target_bits_count,
                    prev_output_bits, &cluster_of, &index_in_cluster,
                    None,
                );
            let (packed, wpe) = pack_input_bits(&input_bits, n, total_input);
            drop(input_bits);
            bitwise_full_train.push(BitwiseSubset {
                input_bits: Vec::new(),
                packed_input: packed,
                target_bits,
                num_examples: n,
                words_per_example: wpe,
                weights: Vec::new(),
            });

            // Train subsets
            bitwise_train_subsets.push(Self::split_and_encode_bitwise_train(
                &train_tokens, context_size, bits_per_token,
                context_input_bits, s, total_input, target_bits_count,
                prev_output_bits, &cluster_of, &index_in_cluster, num_parts,
                None,
            ));

            // Full eval
            let (eval_input, eval_targets, _, eval_n) =
                Self::encode_bitwise_stage_sequence(
                    &eval_tokens, context_size, bits_per_token,
                    context_input_bits, s, total_input, target_bits_count,
                    prev_output_bits, &cluster_of, &index_in_cluster,
                    None,
                );
            let (eval_packed, eval_wpe) = pack_input_bits(&eval_input, eval_n, total_input);
            drop(eval_input);
            bitwise_full_eval.push(BitwiseEvalSubset {
                packed_input: eval_packed,
                targets: eval_targets,
                num_examples: eval_n,
                words_per_example: eval_wpe,
            });

            // Eval subsets
            bitwise_eval_subsets.push(Self::split_and_encode_bitwise_eval(
                &eval_tokens, context_size, bits_per_token,
                context_input_bits, s, total_input, target_bits_count,
                prev_output_bits, &cluster_of, &index_in_cluster, num_eval_parts,
                None,
            ));

            eprintln!(
                "  Stage{s} bitwise: bits={target_bits_count}, vocab={}, train={n}, eval={eval_n}, input={total_input}b",
                stage_vocab[s]
            );
        }

        // ── Encode selector data (per-group datasets for stages with selector mode) ──
        let mut bitwise_selector_train: Vec<Vec<BitwiseSubset>> = Vec::with_capacity(num_stages);
        let mut bitwise_selector_eval: Vec<Vec<BitwiseEvalSubset>> = Vec::with_capacity(num_stages);

        for s in 0..num_stages {
            if s == 0 {
                // Stage 0 has no selector mode
                bitwise_selector_train.push(Vec::new());
                bitwise_selector_eval.push(Vec::new());
            } else {
                // Stage 1+: create K per-group datasets
                let sel_train: Vec<BitwiseSubset> = (0..k)
                    .map(|g| {
                        Self::encode_bitwise_selector(
                            &train_tokens, context_size, bits_per_token,
                            context_input_bits, stage_output_bits_raw[s],
                            &cluster_of, &index_in_cluster, g,
                            true, None,
                        ).0
                    })
                    .collect();
                let sel_eval: Vec<BitwiseEvalSubset> = (0..k)
                    .map(|g| {
                        Self::encode_bitwise_selector(
                            &eval_tokens, context_size, bits_per_token,
                            context_input_bits, stage_output_bits_raw[s],
                            &cluster_of, &index_in_cluster, g,
                            false, None,
                        ).1
                    })
                    .collect();
                let sel_train_n: usize = sel_train.iter().map(|s| s.num_examples).sum();
                let sel_eval_n: usize = sel_eval.iter().map(|s| s.num_examples).sum();
                eprintln!(
                    "  Stage{s} selector: {k} groups, train={sel_train_n} total, eval={sel_eval_n} total"
                );
                bitwise_selector_train.push(sel_train);
                bitwise_selector_eval.push(sel_eval);
            }
        }

        // ── Build eval target token_ids for combined CE ──────────────
        let eval_n_total = bitwise_full_eval[0].num_examples;
        let eval_target_token_ids: Vec<u32> = (0..eval_n_total)
            .map(|i| eval_tokens[i + context_size])
            .collect();

        // ── Encode tiered data per stage ─────────────────────────────
        let default_num_negatives = 10usize;
        let mut tiered_train_subsets: Vec<Option<Vec<TieredSubset>>> = (0..num_stages).map(|_| None).collect();
        let mut tiered_eval_subsets: Vec<Option<Vec<TieredEvalSubset>>> = (0..num_stages).map(|_| None).collect();
        let mut tiered_full_train: Vec<Option<TieredSubset>> = (0..num_stages).map(|_| None).collect();
        let mut tiered_full_eval: Vec<Option<TieredEvalSubset>> = (0..num_stages).map(|_| None).collect();
        let mut tiered_num_negatives = vec![0usize; num_stages];

        for stage in 0..num_stages {
            if !stage_is_tiered[stage] {
                continue;
            }

            let num_neg = default_num_negatives;
            tiered_num_negatives[stage] = num_neg;

            let stage_input = stage_input_bits_raw[stage];
            let stage_num_classes = stage_vocab[stage];

            // Encode full train
            let (train_input, train_targets_i64, train_negs, train_n) =
                Self::encode_tiered_stage_sequence(
                    &train_tokens, context_size, bits_per_token,
                    context_input_bits, stage_output_bits_raw[0],
                    &cluster_of, &index_in_cluster,
                    stage, stage_num_classes, num_neg,
                    None,
                );
            let (train_packed, train_wpe) = pack_bools_to_u64(&train_input, train_n, stage_input);
            tiered_full_train[stage] = Some(TieredSubset {
                input_bits: train_input,
                packed_input: train_packed,
                targets: train_targets_i64,
                negatives: train_negs,
                num_examples: train_n,
                words_per_example: train_wpe,
                num_negatives: num_neg,
            });

            // Encode full eval
            let (eval_input, eval_targets_i64, _, eval_n) =
                Self::encode_tiered_stage_sequence(
                    &eval_tokens, context_size, bits_per_token,
                    context_input_bits, stage_output_bits_raw[0],
                    &cluster_of, &index_in_cluster,
                    stage, stage_num_classes, 0,
                    None,
                );
            let (eval_packed, eval_wpe) = pack_bools_to_u64(&eval_input, eval_n, stage_input);
            tiered_full_eval[stage] = Some(TieredEvalSubset {
                input_bits: eval_input,
                packed_input: eval_packed,
                targets: eval_targets_i64,
                num_examples: eval_n,
                words_per_example: eval_wpe,
            });

            // Encode train subsets
            let train_subs = Self::split_and_encode_tiered_train(
                &train_tokens, context_size, bits_per_token,
                context_input_bits, stage_output_bits_raw[0],
                &cluster_of, &index_in_cluster,
                stage, stage_input, stage_num_classes, num_neg, num_parts,
                None,
            );
            tiered_train_subsets[stage] = Some(train_subs);

            // Encode eval subsets
            let eval_subs = Self::split_and_encode_tiered_eval(
                &eval_tokens, context_size, bits_per_token,
                context_input_bits, stage_output_bits_raw[0],
                &cluster_of, &index_in_cluster,
                stage, stage_input, stage_num_classes, num_eval_parts,
                None,
            );
            tiered_eval_subsets[stage] = Some(eval_subs);

            eprintln!(
                "  Stage{stage} tiered: K={stage_num_classes}, train={train_n}, eval={eval_n}, negatives={num_neg}"
            );
        }

        // ── Logging ──────────────────────────────────────────────────
        let mapping_type = if any_tiered { "freq-interleaved" } else { "round-robin" };
        let stage_labels: Vec<String> = (0..num_stages)
            .map(|s| if stage_is_tiered[s] { "tiered".to_string() } else { "bitwise".to_string() })
            .collect();
        eprintln!(
            "[MultiStageCache] K={k}, vocab={vocab_size}, ctx={context_size}, \
             stages={}, mapping={mapping_type}",
            stage_labels.join("+")
        );
        for s in 0..num_stages {
            eprintln!(
                "  Stage{s}: output_bits={}, input_bits={}, vocab={}",
                stage_output_bits_raw[s], stage_input_bits_raw[s], stage_vocab[s]
            );
        }

        Self {
            k,
            cluster_of,
            index_in_cluster,
            cluster_sizes,
            max_cluster_size,
            context_size,
            bits_per_token,
            context_input_bits,
            bitwise_output_bits: stage_output_bits_raw,
            stage_input_bits: stage_input_bits_raw,
            bitwise_vocab_size: stage_vocab,
            bitwise_token_bits,
            bitwise_train_subsets,
            bitwise_eval_subsets,
            bitwise_full_train,
            bitwise_full_eval,
            bitwise_selector_train,
            bitwise_selector_eval,
            vocab_size,
            eval_target_token_ids,
            stage_is_tiered,
            stage_num_output_clusters,
            tiered_train_subsets,
            tiered_eval_subsets,
            tiered_full_train,
            tiered_full_eval,
            tiered_num_negatives,
            train_tokens: stored_train_tokens,
            eval_tokens: stored_eval_tokens,
            num_parts,
            num_eval_parts,
            current_train_idx: AtomicUsize::new(0),
            current_eval_idx: AtomicUsize::new(0),
        }
    }

    // ── Generic bitwise encoding ─────────────────────────────────────

    /// Encode token sequence for any bitwise stage.
    ///
    /// Stage 0: input = context bits, target = cluster_id
    /// Stage s>0: input = context bits + previous stage output bits (teacher-forced),
    ///            target = within-group index
    fn encode_bitwise_stage_sequence(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        stage: usize,
        total_input_bits: usize,
        target_bits_count: usize,
        previous_output_bits: &[usize],
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        predicted_clusters: Option<&[u32]>,
    ) -> (Vec<bool>, Vec<u32>, Vec<u8>, usize) {
        if tokens.len() <= context_size {
            return (vec![], vec![], vec![], 0);
        }

        let num_ex = tokens.len() - context_size;
        let mut input_bits = vec![false; num_ex * total_input_bits];
        let mut targets = vec![0u32; num_ex];
        let mut target_bits = vec![0u8; num_ex * target_bits_count];

        // Precompute the offset where previous stage outputs start in the input
        // Stage 0: no previous outputs
        // Stage 1: cluster_id bits start at context_input_bits
        // Stage 2+: would accumulate previous stage output widths
        let bits_per_cluster_id = if !previous_output_bits.is_empty() {
            previous_output_bits[0]
        } else {
            0
        };

        input_bits
            .par_chunks_mut(total_input_bits)
            .zip(targets.par_iter_mut())
            .zip(target_bits.par_chunks_mut(target_bits_count))
            .enumerate()
            .for_each(|(i, ((inp, tgt), tb))| {
                // Encode context tokens (same for all stages)
                for ctx in 0..context_size {
                    let token = tokens[i + ctx] as usize;
                    let off = ctx * bits_per_token;
                    for b in 0..bits_per_token {
                        inp[off + b] = ((token >> (bits_per_token - 1 - b)) & 1) == 1;
                    }
                }

                let target_token = tokens[i + context_size] as usize;

                if stage == 0 {
                    // Stage 0: target = cluster_id
                    let cluster_id = cluster_of[target_token] as usize;
                    *tgt = cluster_id as u32;
                    for b in 0..target_bits_count {
                        tb[b] = ((cluster_id >> (target_bits_count - 1 - b)) & 1) as u8;
                    }
                } else {
                    // Stage 1+: append cluster_id bits to input
                    let cluster_id = match predicted_clusters {
                        Some(preds) => preds[i] as usize,  // stage 0's prediction
                        None => cluster_of[target_token] as usize,  // teacher forcing
                    };
                    for b in 0..bits_per_cluster_id {
                        inp[context_input_bits + b] =
                            ((cluster_id >> (bits_per_cluster_id - 1 - b)) & 1) == 1;
                    }
                    // Target = within-group index (always ground truth)
                    let within_idx = index_in_cluster[target_token] as usize;
                    *tgt = within_idx as u32;
                    for b in 0..target_bits_count {
                        tb[b] = ((within_idx >> (target_bits_count - 1 - b)) & 1) as u8;
                    }
                }
            });

        (input_bits, targets, target_bits, num_ex)
    }

    /// Split training tokens and encode each part for a bitwise stage.
    fn split_and_encode_bitwise_train(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        stage: usize,
        total_input_bits: usize,
        target_bits_count: usize,
        previous_output_bits: &[usize],
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        num_parts: usize,
        predicted_clusters: Option<&[u32]>,
    ) -> Vec<BitwiseSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;
        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                // Slice predictions to match this part's examples.
                // Part tokens[start..end] produces num_ex = (end-start)-context_size examples.
                // Part example j corresponds to full example (start + j).
                let part_preds = predicted_clusters.map(|p| {
                    let num_ex = if end > start + context_size { end - start - context_size } else { 0 };
                    &p[start..start + num_ex]
                });
                let (input_bits, _, target_bits, num_ex) =
                    Self::encode_bitwise_stage_sequence(
                        part, context_size, bits_per_token,
                        context_input_bits, stage, total_input_bits, target_bits_count,
                        previous_output_bits, cluster_of, index_in_cluster,
                        part_preds,
                    );
                let (packed, wpe) = pack_input_bits(&input_bits, num_ex, total_input_bits);
                BitwiseSubset {
                    input_bits: Vec::new(),
                    packed_input: packed,
                    target_bits,
                    num_examples: num_ex,
                    words_per_example: wpe,
                    weights: Vec::new(),
                }
            })
            .collect()
    }

    /// Split eval tokens and encode each part for a bitwise stage.
    fn split_and_encode_bitwise_eval(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        stage: usize,
        total_input_bits: usize,
        target_bits_count: usize,
        previous_output_bits: &[usize],
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        num_parts: usize,
        predicted_clusters: Option<&[u32]>,
    ) -> Vec<BitwiseEvalSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;
        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let part_preds = predicted_clusters.map(|p| {
                    let num_ex = if end > start + context_size { end - start - context_size } else { 0 };
                    &p[start..start + num_ex]
                });
                let (input_bits, targets, _, num_ex) =
                    Self::encode_bitwise_stage_sequence(
                        part, context_size, bits_per_token,
                        context_input_bits, stage, total_input_bits, target_bits_count,
                        previous_output_bits, cluster_of, index_in_cluster,
                        part_preds,
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

    // ── Bitwise selector encoding ───────────────────────────────────

    /// Encode a per-group dataset for selector mode.
    ///
    /// Filters examples where cluster_of[target] == group_id.
    /// Input = context bits only (same width as Stage 0).
    /// Target = index_in_cluster[target].
    ///
    /// Returns (train_subset, eval_subset) — caller picks one based on is_train.
    fn encode_bitwise_selector(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_within_index: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        group_id: usize,
        is_train: bool,
        predicted_clusters: Option<&[u32]>,
    ) -> (BitwiseSubset, BitwiseEvalSubset) {
        if tokens.len() <= context_size {
            let empty_train = BitwiseSubset {
                input_bits: Vec::new(), packed_input: Vec::new(),
                target_bits: Vec::new(), num_examples: 0, words_per_example: 0,
                weights: Vec::new(),
            };
            let empty_eval = BitwiseEvalSubset {
                packed_input: Vec::new(), targets: Vec::new(),
                num_examples: 0, words_per_example: 0,
            };
            return (empty_train, empty_eval);
        }

        // Collect indices of examples belonging to this group
        // With predictions: filter by predicted cluster; without: filter by true cluster
        let total_ex = tokens.len() - context_size;
        let group_indices: Vec<usize> = (0..total_ex)
            .filter(|&i| {
                match predicted_clusters {
                    Some(preds) => preds[i] as usize == group_id,
                    None => cluster_of[tokens[i + context_size] as usize] as usize == group_id,
                }
            })
            .collect();
        let num_ex = group_indices.len();

        if num_ex == 0 {
            let empty_train = BitwiseSubset {
                input_bits: Vec::new(), packed_input: Vec::new(),
                target_bits: Vec::new(), num_examples: 0, words_per_example: 0,
                weights: Vec::new(),
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

        input_bits
            .par_chunks_mut(context_input_bits)
            .zip(targets.par_iter_mut())
            .zip(target_bits.par_chunks_mut(bits_per_within_index))
            .zip(group_indices.par_iter())
            .for_each(|(((inp, tgt), tb), &orig_idx)| {
                for ctx in 0..context_size {
                    let token = tokens[orig_idx + ctx] as usize;
                    let off = ctx * bits_per_token;
                    for b in 0..bits_per_token {
                        inp[off + b] = ((token >> (bits_per_token - 1 - b)) & 1) == 1;
                    }
                }
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
                weights: Vec::new(),
            }
        } else {
            BitwiseSubset {
                input_bits: Vec::new(), packed_input: Vec::new(),
                target_bits: Vec::new(), num_examples: 0, words_per_example: 0,
                weights: Vec::new(),
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

    // ── Tiered encoding ────────────────────────────────────────────

    /// Encode a token sequence for a tiered stage.
    ///
    /// Stage 0: input = context bits, target = cluster_id, num_classes = K
    /// Stage 1+: input = context bits + cluster_id bits, target = within-index, num_classes = max_cluster_size
    ///
    /// Returns (input_bits, targets, negatives, num_examples).
    fn encode_tiered_stage_sequence(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_cluster_id: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        stage: usize,
        num_classes: usize,
        num_negatives: usize,
        predicted_clusters: Option<&[u32]>,
    ) -> (Vec<bool>, Vec<i64>, Vec<i64>, usize) {
        if tokens.len() <= context_size {
            return (vec![], vec![], vec![], 0);
        }

        let num_ex = tokens.len() - context_size;
        let total_input_bits = match stage {
            0 => context_input_bits,
            _ => context_input_bits + bits_per_cluster_id,
        };

        let mut input_bits = vec![false; num_ex * total_input_bits];
        let mut targets = vec![0i64; num_ex];
        // Allocate at least 1 element per example so par_chunks_mut produces
        // the correct number of chunks even when num_negatives == 0.
        let neg_stride = num_negatives.max(1);
        let mut negatives = vec![0i64; num_ex * neg_stride];

        input_bits
            .par_chunks_mut(total_input_bits)
            .zip(targets.par_iter_mut())
            .zip(negatives.par_chunks_mut(neg_stride))
            .enumerate()
            .for_each(|(i, ((inp, tgt), negs))| {
                for ctx in 0..context_size {
                    let token = tokens[i + ctx] as usize;
                    let off = ctx * bits_per_token;
                    for b in 0..bits_per_token {
                        inp[off + b] = ((token >> (bits_per_token - 1 - b)) & 1) == 1;
                    }
                }

                let target_token = tokens[i + context_size] as usize;

                match stage {
                    0 => {
                        *tgt = cluster_of[target_token] as i64;
                    }
                    _ => {
                        let cluster_id = match predicted_clusters {
                            Some(preds) => preds[i] as usize,
                            None => cluster_of[target_token] as usize,
                        };
                        for b in 0..bits_per_cluster_id {
                            inp[context_input_bits + b] =
                                ((cluster_id >> (bits_per_cluster_id - 1 - b)) & 1) == 1;
                        }
                        *tgt = index_in_cluster[target_token] as i64;
                    }
                }

                if num_negatives > 0 {
                    let true_class = *tgt as usize;
                    let mut rng = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
                    for k in 0..num_negatives {
                        loop {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(k as u64 + 1);
                            let neg = (rng >> 33) as usize % num_classes;
                            if neg != true_class {
                                negs[k] = neg as i64;
                                break;
                            }
                        }
                    }
                }
            });

        // Trim negatives to actual size (we may have over-allocated for zip alignment)
        if num_negatives == 0 {
            negatives.clear();
        }
        (input_bits, targets, negatives, num_ex)
    }

    /// Split and encode tiered training subsets.
    fn split_and_encode_tiered_train(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_cluster_id: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        stage: usize,
        total_input_bits: usize,
        num_classes: usize,
        num_negatives: usize,
        num_parts: usize,
        predicted_clusters: Option<&[u32]>,
    ) -> Vec<TieredSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;

        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let part_preds = predicted_clusters.map(|p| {
                    let num_ex = if end > start + context_size { end - start - context_size } else { 0 };
                    &p[start..start + num_ex]
                });
                let (input_bits, targets, negatives, num_ex) =
                    Self::encode_tiered_stage_sequence(
                        part, context_size, bits_per_token,
                        context_input_bits, bits_per_cluster_id,
                        cluster_of, index_in_cluster,
                        stage, num_classes, num_negatives,
                        part_preds,
                    );
                let (packed, wpe) = pack_bools_to_u64(&input_bits, num_ex, total_input_bits);
                TieredSubset {
                    input_bits,
                    packed_input: packed,
                    targets,
                    negatives,
                    num_examples: num_ex,
                    words_per_example: wpe,
                    num_negatives: num_negatives,
                }
            })
            .collect()
    }

    /// Split and encode tiered eval subsets.
    fn split_and_encode_tiered_eval(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        context_input_bits: usize,
        bits_per_cluster_id: usize,
        cluster_of: &[u16],
        index_in_cluster: &[u16],
        stage: usize,
        total_input_bits: usize,
        num_classes: usize,
        num_eval_parts: usize,
        predicted_clusters: Option<&[u32]>,
    ) -> Vec<TieredEvalSubset> {
        let n = tokens.len();
        let part_size = n / num_eval_parts;

        (0..num_eval_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_eval_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let part_preds = predicted_clusters.map(|p| {
                    let num_ex = if end > start + context_size { end - start - context_size } else { 0 };
                    &p[start..start + num_ex]
                });
                let (input_bits, targets, _, num_ex) =
                    Self::encode_tiered_stage_sequence(
                        part, context_size, bits_per_token,
                        context_input_bits, bits_per_cluster_id,
                        cluster_of, index_in_cluster,
                        stage, num_classes, 0,
                        part_preds,
                    );
                let (packed, wpe) = pack_bools_to_u64(&input_bits, num_ex, total_input_bits);
                TieredEvalSubset {
                    input_bits,
                    packed_input: packed,
                    targets,
                    num_examples: num_ex,
                    words_per_example: wpe,
                }
            })
            .collect()
    }

    // ── Re-encode stage data with predicted clusters ──────────────────

    /// Re-encode all data for `target_stage` using predicted clusters from the
    /// previous stage instead of teacher forcing.
    ///
    /// `target_stage` must be > 0 (stage 0 has no previous stage to predict from).
    pub fn recompute_stage_data(
        &mut self,
        target_stage: usize,
        train_predictions: &[u32],
        eval_predictions: &[u32],
    ) {
        assert!(target_stage > 0, "Cannot recompute stage 0 — it has no previous stage");
        let s = target_stage;
        let total_input = self.stage_input_bits[s];
        let target_bits_count = self.bitwise_output_bits[s];
        let prev_output_bits_slice = &self.bitwise_output_bits[..s];

        // ── Bitwise full train ──
        let (input_bits, _targets, target_bits, n) =
            Self::encode_bitwise_stage_sequence(
                &self.train_tokens, self.context_size, self.bits_per_token,
                self.context_input_bits, s, total_input, target_bits_count,
                prev_output_bits_slice, &self.cluster_of, &self.index_in_cluster,
                Some(train_predictions),
            );
        let (packed, wpe) = pack_input_bits(&input_bits, n, total_input);
        drop(input_bits);
        self.bitwise_full_train[s] = BitwiseSubset {
            input_bits: Vec::new(),
            packed_input: packed,
            target_bits,
            num_examples: n,
            words_per_example: wpe,
            weights: Vec::new(),
        };

        // ── Bitwise full eval ──
        let (eval_input, eval_targets, _, eval_n) =
            Self::encode_bitwise_stage_sequence(
                &self.eval_tokens, self.context_size, self.bits_per_token,
                self.context_input_bits, s, total_input, target_bits_count,
                prev_output_bits_slice, &self.cluster_of, &self.index_in_cluster,
                Some(eval_predictions),
            );
        let (eval_packed, eval_wpe) = pack_input_bits(&eval_input, eval_n, total_input);
        drop(eval_input);
        self.bitwise_full_eval[s] = BitwiseEvalSubset {
            packed_input: eval_packed,
            targets: eval_targets,
            num_examples: eval_n,
            words_per_example: eval_wpe,
        };

        // ── Bitwise train subsets ──
        self.bitwise_train_subsets[s] = Self::split_and_encode_bitwise_train(
            &self.train_tokens, self.context_size, self.bits_per_token,
            self.context_input_bits, s, total_input, target_bits_count,
            prev_output_bits_slice, &self.cluster_of, &self.index_in_cluster,
            self.num_parts, Some(train_predictions),
        );

        // ── Bitwise eval subsets ──
        self.bitwise_eval_subsets[s] = Self::split_and_encode_bitwise_eval(
            &self.eval_tokens, self.context_size, self.bits_per_token,
            self.context_input_bits, s, total_input, target_bits_count,
            prev_output_bits_slice, &self.cluster_of, &self.index_in_cluster,
            self.num_eval_parts, Some(eval_predictions),
        );

        // ── Bitwise selector data (per-group) ──
        if s < self.bitwise_selector_train.len() && !self.bitwise_selector_train[s].is_empty() {
            let k = self.k;
            self.bitwise_selector_train[s] = (0..k)
                .map(|g| {
                    Self::encode_bitwise_selector(
                        &self.train_tokens, self.context_size, self.bits_per_token,
                        self.context_input_bits, self.bitwise_output_bits[s],
                        &self.cluster_of, &self.index_in_cluster, g,
                        true, Some(train_predictions),
                    ).0
                })
                .collect();
            self.bitwise_selector_eval[s] = (0..k)
                .map(|g| {
                    Self::encode_bitwise_selector(
                        &self.eval_tokens, self.context_size, self.bits_per_token,
                        self.context_input_bits, self.bitwise_output_bits[s],
                        &self.cluster_of, &self.index_in_cluster, g,
                        false, Some(eval_predictions),
                    ).1
                })
                .collect();
        }

        // ── Tiered data (if this stage is tiered) ──
        if self.stage_is_tiered[s] {
            let bits_per_cluster_id = self.bitwise_output_bits[s - 1];
            let stage_input = self.stage_input_bits[s];
            let stage_num_classes = self.bitwise_vocab_size[s];
            let num_neg = self.tiered_num_negatives[s];

            // Full train
            let (train_input, train_targets_i64, train_negs, train_n) =
                Self::encode_tiered_stage_sequence(
                    &self.train_tokens, self.context_size, self.bits_per_token,
                    self.context_input_bits, bits_per_cluster_id,
                    &self.cluster_of, &self.index_in_cluster,
                    s, stage_num_classes, num_neg,
                    Some(train_predictions),
                );
            let (train_packed, train_wpe) = pack_bools_to_u64(&train_input, train_n, stage_input);
            self.tiered_full_train[s] = Some(TieredSubset {
                input_bits: train_input,
                packed_input: train_packed,
                targets: train_targets_i64,
                negatives: train_negs,
                num_examples: train_n,
                words_per_example: train_wpe,
                num_negatives: num_neg,
            });

            // Full eval
            let (eval_input_t, eval_targets_i64, _, eval_n_t) =
                Self::encode_tiered_stage_sequence(
                    &self.eval_tokens, self.context_size, self.bits_per_token,
                    self.context_input_bits, bits_per_cluster_id,
                    &self.cluster_of, &self.index_in_cluster,
                    s, stage_num_classes, 0,
                    Some(eval_predictions),
                );
            let (eval_packed_t, eval_wpe_t) = pack_bools_to_u64(&eval_input_t, eval_n_t, stage_input);
            self.tiered_full_eval[s] = Some(TieredEvalSubset {
                input_bits: eval_input_t,
                packed_input: eval_packed_t,
                targets: eval_targets_i64,
                num_examples: eval_n_t,
                words_per_example: eval_wpe_t,
            });

            // Train subsets
            self.tiered_train_subsets[s] = Some(Self::split_and_encode_tiered_train(
                &self.train_tokens, self.context_size, self.bits_per_token,
                self.context_input_bits, bits_per_cluster_id,
                &self.cluster_of, &self.index_in_cluster,
                s, stage_input, stage_num_classes, num_neg, self.num_parts,
                Some(train_predictions),
            ));

            // Eval subsets
            self.tiered_eval_subsets[s] = Some(Self::split_and_encode_tiered_eval(
                &self.eval_tokens, self.context_size, self.bits_per_token,
                self.context_input_bits, bits_per_cluster_id,
                &self.cluster_of, &self.index_in_cluster,
                s, stage_input, stage_num_classes, self.num_eval_parts,
                Some(eval_predictions),
            ));
        }

        eprintln!(
            "[recompute_stage_data] Re-encoded S{s}: bitwise train={n}, eval={eval_n}"
        );
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

// ── Stage prediction ────────────────────────────────────────────────────

/// Train a frozen stage and predict its output class for every example in train and eval data.
///
/// Trains once on full train data, then forwards on both eval and train data
/// (reusing the same trained memory). Extracts argmax predictions from the
/// bitwise bit-product reconstruction.
///
/// `stage`: which stage to predict (0, 1, ...).
///
/// Returns: (train_predictions, eval_predictions, train_correct, eval_correct)
pub fn predict_stage_clusters(
    cache: &MultiStageTokenCache,
    stage: usize,
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold: usize,
) -> (Vec<u32>, Vec<u32>, usize, usize) {
    let vocab_size = cache.bitwise_vocab_size[stage]; // K for stage 0, max_cluster_size for stage 1, etc.
    let out_bits = cache.bitwise_output_bits[stage];
    let num_clusters = out_bits; // bitwise: num_clusters = number of output bit positions
    let eps = 1e-7f64;

    let num_train = cache.bitwise_full_train[stage].num_examples;
    let num_eval = cache.bitwise_full_eval[stage].num_examples;

    // ── Train once, forward twice ──────────────────────────────────
    let empty_word: i64 = crate::neuron_memory::empty_word_for_mode(memory_mode);
    let layout = compute_genome_layout(
        bits_per_neuron, neurons_per_cluster, sparse_threshold, num_train,
    );

    let mut cluster_storage: Vec<ClusterStorage> = (0..num_clusters)
        .map(|c| ClusterStorage::new(
            neurons_per_cluster[c], layout.cluster_max_bits[c],
            sparse_threshold, empty_word, memory_mode,
        ))
        .collect();

    // Train on full train data
    train_into(
        connections, bits_per_neuron, neurons_per_cluster, num_clusters,
        &layout, &cache.bitwise_full_train[stage], &mut cluster_storage,
        memory_mode, neuron_sample_rate, rng_seed,
    );

    // Forward on eval data
    let mut eval_scores = vec![0.0f32; num_eval * num_clusters];
    forward_eval_into(
        connections, bits_per_neuron, neurons_per_cluster, num_clusters,
        &layout, &cache.bitwise_full_eval[stage], &cluster_storage,
        &mut eval_scores, memory_mode,
    );

    // Forward on train data (reuse trained memory — no second training)
    let train_as_eval = BitwiseEvalSubset {
        packed_input: cache.bitwise_full_train[stage].packed_input.clone(),
        targets: vec![0u32; num_train], // unused for forward
        num_examples: num_train,
        words_per_example: cache.bitwise_full_train[stage].words_per_example,
    };
    let mut train_scores = vec![0.0f32; num_train * num_clusters];
    forward_eval_into(
        connections, bits_per_neuron, neurons_per_cluster, num_clusters,
        &layout, &train_as_eval, &cluster_storage,
        &mut train_scores, memory_mode,
    );

    // ── Extract argmax predictions ─────────────────────────────────
    let extract_predictions = |scores: &[f32], num_examples: usize| -> Vec<u32> {
        let mut predictions = vec![0u32; num_examples];

        predictions.par_iter_mut().enumerate().for_each(|(ex, pred)| {
            // Bit-product reconstruction: log P(class_k) = sum over bits of log(p_b or 1-p_b)
            let base = ex * out_bits;
            let mut log_p = vec![0.0f64; out_bits];
            let mut log_1mp = vec![0.0f64; out_bits];
            for b in 0..out_bits {
                let p = (scores[base + b] as f64).clamp(eps, 1.0 - eps);
                log_p[b] = p.ln();
                log_1mp[b] = (1.0 - p).ln();
            }

            let mut best_k = 0usize;
            let mut best_lp = f64::NEG_INFINITY;
            for gk in 0..vocab_size {
                let bit_base = gk * out_bits;
                let mut lp = 0.0f64;
                for b in 0..out_bits {
                    if cache.bitwise_token_bits[stage][bit_base + b] == 1 {
                        lp += log_p[b];
                    } else {
                        lp += log_1mp[b];
                    }
                }
                if lp > best_lp {
                    best_lp = lp;
                    best_k = gk;
                }
            }
            *pred = best_k as u32;
        });

        predictions
    };

    let eval_predictions = extract_predictions(&eval_scores, num_eval);
    let train_predictions = extract_predictions(&train_scores, num_train);

    // ── Count correct predictions ──────────────────────────────────
    // For stage 0: target is cluster_of[token]; for stage 1+: target is index_in_cluster[token]
    let (train_correct, eval_correct) = if stage == 0 {
        let tc: usize = (0..num_train)
            .filter(|&i| {
                let token = cache.train_tokens[i + cache.context_size] as usize;
                train_predictions[i] == cache.cluster_of[token] as u32
            })
            .count();
        let ec: usize = (0..num_eval)
            .filter(|&i| {
                let token = cache.eval_tokens[i + cache.context_size] as usize;
                eval_predictions[i] == cache.cluster_of[token] as u32
            })
            .count();
        (tc, ec)
    } else {
        let tc: usize = (0..num_train)
            .filter(|&i| {
                let token = cache.train_tokens[i + cache.context_size] as usize;
                train_predictions[i] == cache.index_in_cluster[token] as u32
            })
            .count();
        let ec: usize = (0..num_eval)
            .filter(|&i| {
                let token = cache.eval_tokens[i + cache.context_size] as usize;
                eval_predictions[i] == cache.index_in_cluster[token] as u32
            })
            .count();
        (tc, ec)
    };

    eprintln!(
        "[predict_stage_clusters] S{stage}: train={train_correct}/{num_train} ({:.2}%), eval={eval_correct}/{num_eval} ({:.2}%)",
        100.0 * train_correct as f64 / num_train.max(1) as f64,
        100.0 * eval_correct as f64 / num_eval.max(1) as f64,
    );

    (train_predictions, eval_predictions, train_correct, eval_correct)
}

// ── Bitwise evaluation (stage-agnostic) ─────────────────────────────────

/// Evaluate bitwise genomes for any stage with subset rotation.
///
/// Delegates to the generic bitwise evaluation pipeline with stage-indexed data.
///
/// Returns: Vec<(ce, accuracy, bit_acc)> per genome.
pub fn evaluate_bitwise_genomes(
    cache: &MultiStageTokenCache,
    stage: usize,
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
    let train_subset = &cache.bitwise_train_subsets[stage][train_subset_idx % cache.num_parts];
    let eval_subset = &cache.bitwise_eval_subsets[stage][eval_subset_idx % cache.num_eval_parts];
    evaluate_genomes_with_params(
        cache.bitwise_output_bits[stage],
        &cache.bitwise_token_bits[stage],
        cache.bitwise_vocab_size[stage],
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

/// Evaluate bitwise genomes with full (non-rotated) data.
pub fn evaluate_bitwise_genomes_full(
    cache: &MultiStageTokenCache,
    stage: usize,
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
        cache.bitwise_output_bits[stage],
        &cache.bitwise_token_bits[stage],
        cache.bitwise_vocab_size[stage],
        bits_per_neuron_flat,
        neurons_per_cluster_flat,
        connections_flat,
        num_genomes,
        &cache.bitwise_full_train[stage],
        &cache.bitwise_full_eval[stage],
        memory_mode,
        neuron_sample_rate,
        rng_seed,
        sparse_threshold_override,
    )
}

// ── Bitwise selector evaluation ─────────────────────────────────────────

/// Evaluate bitwise genomes for a single group (selector mode).
///
/// Uses per-group filtered data: only examples where target ∈ group_id.
/// Input = context bits only.
///
/// Returns: Vec<(ce, accuracy, bit_acc)> per genome.
pub fn evaluate_bitwise_selector_genomes(
    cache: &MultiStageTokenCache,
    stage: usize,
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
    let train_subset = &cache.bitwise_selector_train[stage][group_id];
    let eval_subset = &cache.bitwise_selector_eval[stage][group_id];

    if eval_subset.num_examples == 0 {
        return vec![(0.0, 0.0, 0.0); num_genomes];
    }

    evaluate_genomes_with_params(
        cache.bitwise_output_bits[stage],
        &cache.bitwise_token_bits[stage],
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

// ── Combined CE computation ─────────────────────────────────────────────

/// Compute combined multi-stage CE from per-stage genome params.
///
/// Dispatches each stage to the appropriate scorer:
/// - Bitwise: `train_and_get_scores()` → bit-product reconstruction
/// - Tiered: `train_and_get_tiered_scores()` → softmax normalization
///
/// Trains all stages, forwards on eval data, then reconstructs the joint distribution:
///   P(token_t) = P(group_g | context) × P(token_t | group_g, context)
///
/// Returns: (combined_ce, combined_accuracy, stage0_ce, stage1_ce)
pub fn compute_combined_ce(
    cache: &MultiStageTokenCache,
    // Per-stage genome params (slices for each stage)
    stage_bits_per_neuron: &[&[usize]],
    stage_neurons_per_cluster: &[&[usize]],
    stage_connections: &[&[i64]],
    // Training params
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold: usize,
    label_smoothing: f64,
) -> (f64, f64, f64, f64) {
    let eps = 1e-7f64;
    let epsilon = 1e-10f64;
    let num_eval = cache.bitwise_full_eval[0].num_examples;
    let num_stages = cache.stage_is_tiered.len();
    let k = cache.k;

    // ── Get per-stage scores ──
    // For each stage: either tiered scores [num_eval × num_classes] or
    // bitwise scores [num_eval × output_bits]
    let mut stage_tiered_scores: Vec<Vec<f64>> = vec![Vec::new(); num_stages];
    let mut stage_bitwise_scores: Vec<Vec<f32>> = vec![Vec::new(); num_stages];

    for s in 0..num_stages {
        if cache.stage_is_tiered[s] {
            let train = cache.tiered_full_train[s].as_ref().unwrap();
            let eval = cache.tiered_full_eval[s].as_ref().unwrap();
            stage_tiered_scores[s] = train_and_get_tiered_scores(
                stage_connections[s], stage_bits_per_neuron[s], stage_neurons_per_cluster[s],
                cache.bitwise_vocab_size[s], train, eval, cache.stage_input_bits[s],
                memory_mode, neuron_sample_rate, rng_seed.wrapping_add(s as u64),
            );
        } else {
            stage_bitwise_scores[s] = train_and_get_scores(
                stage_connections[s], stage_bits_per_neuron[s], stage_neurons_per_cluster[s],
                cache.bitwise_output_bits[s],
                &cache.bitwise_full_train[s], &cache.bitwise_full_eval[s],
                memory_mode, neuron_sample_rate, rng_seed.wrapping_add(s as u64), sparse_threshold,
            );
        }
    }

    // ── Reconstruct combined CE per example ──
    let results: Vec<(f64, f64, f64, u32)> = (0..num_eval)
        .into_par_iter()
        .map(|ex| {
            let token_id = cache.eval_target_token_ids[ex] as usize;
            let true_group = cache.cluster_of[token_id] as usize;
            let true_within = cache.index_in_cluster[token_id] as usize;

            // ── Stage 0: compute log P(true_group) ──
            let (log_p_true_group, s0_predicted) = if cache.stage_is_tiered[0] {
                // Tiered: scores are [K] raw neuron outputs, apply softmax
                let base = ex * k;
                let scores = &stage_tiered_scores[0][base..base + k];
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let log_p = (exp_scores[true_group] / sum_exp + epsilon).ln();
                let predicted = scores.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx).unwrap_or(0);
                (log_p, predicted)
            } else {
                // Bitwise: bit-product reconstruction
                let s0_bits = cache.bitwise_output_bits[0];
                let s0_base = ex * s0_bits;
                let mut s0_log_p = vec![0.0f64; s0_bits];
                let mut s0_log_1mp = vec![0.0f64; s0_bits];
                for b in 0..s0_bits {
                    let p = (stage_bitwise_scores[0][s0_base + b] as f64).clamp(eps, 1.0 - eps);
                    s0_log_p[b] = p.ln();
                    s0_log_1mp[b] = (1.0 - p).ln();
                }

                let mut max_s0 = f64::NEG_INFINITY;
                let mut sum_exp_s0 = 0.0f64;
                let mut log_p_true_raw = 0.0f64;
                let mut predicted = 0usize;
                let mut predicted_lp = f64::NEG_INFINITY;

                for gk in 0..k {
                    let bit_base = gk * s0_bits;
                    let mut log_p = 0.0f64;
                    for b in 0..s0_bits {
                        if cache.bitwise_token_bits[0][bit_base + b] == 1 {
                            log_p += s0_log_p[b];
                        } else {
                            log_p += s0_log_1mp[b];
                        }
                    }
                    if gk == true_group { log_p_true_raw = log_p; }
                    if log_p > predicted_lp { predicted_lp = log_p; predicted = gk; }
                    if log_p > max_s0 {
                        sum_exp_s0 = sum_exp_s0 * (max_s0 - log_p).exp() + 1.0;
                        max_s0 = log_p;
                    } else {
                        sum_exp_s0 += (log_p - max_s0).exp();
                    }
                }
                let log_z0 = max_s0 + sum_exp_s0.ln();
                (log_p_true_raw - log_z0, predicted)
            };

            // ── Stage 1: compute log P(true_within | true_group) ──
            let group_size = cache.cluster_sizes[true_group];
            let (log_p_true_within, s1_predicted) = if cache.stage_is_tiered[1] {
                let base = ex * cache.max_cluster_size;
                let scores = &stage_tiered_scores[1][base..base + group_size];
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let log_p = (exp_scores[true_within] / sum_exp + epsilon).ln();
                let predicted = scores.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx).unwrap_or(0);
                (log_p, predicted)
            } else {
                let s1_bits = cache.bitwise_output_bits[1];
                let s1_base = ex * s1_bits;
                let mut s1_log_p = vec![0.0f64; s1_bits];
                let mut s1_log_1mp = vec![0.0f64; s1_bits];
                for b in 0..s1_bits {
                    let p = (stage_bitwise_scores[1][s1_base + b] as f64).clamp(eps, 1.0 - eps);
                    s1_log_p[b] = p.ln();
                    s1_log_1mp[b] = (1.0 - p).ln();
                }

                let mut max_s1 = f64::NEG_INFINITY;
                let mut sum_exp_s1 = 0.0f64;
                let mut log_p_true_raw = 0.0f64;
                let mut predicted = 0usize;
                let mut predicted_lp = f64::NEG_INFINITY;

                for j in 0..group_size {
                    let bit_base = j * s1_bits;
                    let mut log_p = 0.0f64;
                    for b in 0..s1_bits {
                        if cache.bitwise_token_bits[1][bit_base + b] == 1 {
                            log_p += s1_log_p[b];
                        } else {
                            log_p += s1_log_1mp[b];
                        }
                    }
                    if j == true_within { log_p_true_raw = log_p; }
                    if log_p > predicted_lp { predicted_lp = log_p; predicted = j; }
                    if log_p > max_s1 {
                        sum_exp_s1 = sum_exp_s1 * (max_s1 - log_p).exp() + 1.0;
                        max_s1 = log_p;
                    } else {
                        sum_exp_s1 += (log_p - max_s1).exp();
                    }
                }
                let log_z1 = max_s1 + sum_exp_s1.ln();
                (log_p_true_raw - log_z1, predicted)
            };

            let s0_ce = -log_p_true_group;
            let s1_ce = -log_p_true_within;
            let combined_ce = if label_smoothing > 0.0 {
                let log_p_hier = log_p_true_group + log_p_true_within;
                let p_hier = log_p_hier.exp();
                let p_smooth = (1.0 - label_smoothing) * p_hier
                             + label_smoothing / cache.vocab_size as f64;
                -p_smooth.ln()
            } else {
                s0_ce + s1_ce
            };
            let correct = if s0_predicted == true_group && s1_predicted == true_within { 1u32 } else { 0u32 };

            (combined_ce, s0_ce, s1_ce, correct)
        })
        .collect();

    let total_ce: f64 = results.iter().map(|r| r.0).sum();
    let total_s0_ce: f64 = results.iter().map(|r| r.1).sum();
    let total_s1_ce: f64 = results.iter().map(|r| r.2).sum();
    let total_correct: u32 = results.iter().map(|r| r.3).sum();

    (
        total_ce / num_eval as f64,
        total_correct as f64 / num_eval as f64,
        total_s0_ce / num_eval as f64,
        total_s1_ce / num_eval as f64,
    )
}

// ── SELECTOR combined CE ────────────────────────────────────────────────

/// Check if group g is in the top-M groups by probability for a given example.
fn is_in_top_m(probs: &[f32], ex: usize, k: usize, g: usize, top_m: usize) -> bool {
    let base = ex * k;
    let p_g = probs[base + g];
    let higher = (0..k).filter(|&gk| probs[base + gk] > p_g).count();
    higher < top_m
}

/// Build augmented training subsets for invalid token mode.
///
/// For each group g, creates a training subset containing:
/// - Examples where target ∈ group g: target = within-group index
/// - Examples where target ∉ group g: target = cluster_sizes[g] (invalid code)
/// - Weight = P(g|context) from S0
///
/// When top_m > 0, only includes out-group examples where g is in the
/// example's top-M S0 predictions (limits training cost from K×N to ~M×N).
fn build_augmented_selector_subsets(
    cache: &MultiStageTokenCache,
    s1_stage: usize,
    train_group_probs: &[f32],  // [num_train × K]
    augmented_output_bits: usize,
    top_m: usize,
) -> Vec<BitwiseSubset> {
    let k = cache.k;
    let num_train = cache.bitwise_full_train[s1_stage].num_examples;
    let context_size = cache.context_size;
    let wpe = cache.bitwise_full_train[s1_stage].words_per_example;

    (0..k).map(|g| {
        let invalid_code = cache.cluster_sizes[g];

        // Collect (global_idx, is_in_group) for all examples to include
        let mut included: Vec<(usize, bool)> = Vec::new();
        for ex in 0..num_train {
            let target_token = cache.train_tokens[ex + context_size] as usize;
            let true_group = cache.cluster_of[target_token] as usize;
            let is_in_group = true_group == g;

            if is_in_group {
                included.push((ex, true));
            } else if top_m == 0 || is_in_top_m(train_group_probs, ex, k, g, top_m) {
                included.push((ex, false));
            }
        }

        let num_aug = included.len();
        let mut packed_input = vec![0u64; num_aug * wpe];
        let mut target_bits = vec![0u8; num_aug * augmented_output_bits];
        let mut weights = vec![0.0f32; num_aug];

        for (i, &(global_idx, is_in_group)) in included.iter().enumerate() {
            // Copy packed input from S1 stage train data (includes context + cluster_id)
            let src_start = global_idx * wpe;
            packed_input[i * wpe..(i + 1) * wpe]
                .copy_from_slice(&cache.bitwise_full_train[s1_stage].packed_input[src_start..src_start + wpe]);

            // Encode target: within-group index for in-group, invalid_code for out-group
            let target = if is_in_group {
                let target_token = cache.train_tokens[global_idx + context_size] as usize;
                cache.index_in_cluster[target_token] as usize
            } else {
                invalid_code
            };

            for b in 0..augmented_output_bits {
                target_bits[i * augmented_output_bits + b] =
                    ((target >> (augmented_output_bits - 1 - b)) & 1) as u8;
            }

            weights[i] = train_group_probs[global_idx * k + g];
        }

        BitwiseSubset {
            input_bits: Vec::new(),
            packed_input,
            target_bits,
            num_examples: num_aug,
            words_per_example: wpe,
            weights,
        }
    }).collect()
}

/// Compute CE using filtered mixture evaluation for invalid token mode.
///
/// For each eval example:
/// 1. S0 gives P(g|context) for all K groups
/// 2. Each S1_g gives per-class scores including "invalid" class
/// 3. Filtered gate: gate(g) = P(g|S0) × (1 - P(invalid|S1_g))
/// 4. CE = -log P_filt(true_group) - log P(true_within | S1_true, valid)
fn eval_filtered_mixture(
    cache: &MultiStageTokenCache,
    stage_bitwise_scores_0: &[f32],
    stage_tiered_scores_0: &[f64],
    all_group_eval_scores: &[Vec<f32>],  // [K][num_eval × augmented_output_bits]
    augmented_output_bits: usize,
    label_smoothing: f64,
) -> (f64, f64, f64, f64) {
    let eps = 1e-7f64;
    let epsilon = 1e-10f64;
    let num_eval = cache.bitwise_full_eval[0].num_examples;
    let k = cache.k;

    // Compute S0 group probs for eval data (reuse existing helper for bitwise)
    let eval_group_probs: Vec<f32> = if !stage_bitwise_scores_0.is_empty() {
        compute_group_softmax_bitwise(
            stage_bitwise_scores_0, num_eval,
            cache.bitwise_output_bits[0], k, &cache.bitwise_token_bits[0],
        )
    } else {
        // Tiered S0: direct softmax over K scores
        let mut probs = vec![0.0f32; num_eval * k];
        for ex in 0..num_eval {
            let base = ex * k;
            let scores = &stage_tiered_scores_0[base..base + k];
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = scores.iter().map(|&s| (s - max_s).exp()).sum();
            for gk in 0..k {
                probs[base + gk] = ((scores[gk] - max_s).exp() / sum_exp) as f32;
            }
        }
        probs
    };

    let results: Vec<(f64, f64, f64, u32)> = (0..num_eval)
        .into_par_iter()
        .map(|ex| {
            let token_id = cache.eval_target_token_ids[ex] as usize;
            let true_group = cache.cluster_of[token_id] as usize;
            let true_within = cache.index_in_cluster[token_id] as usize;

            // S0: log P(true_group) from eval group probs
            let p_true_group = (eval_group_probs[ex * k + true_group] as f64).max(epsilon);
            let log_p_s0 = p_true_group.ln();

            // For each group g: compute P(invalid|S1_g) and gate
            let mut gates = vec![0.0f64; k];
            let mut s1_log_p_within_valid = 0.0f64; // log P(true_within | S1_true_group, valid)
            let mut best_predicted_g = 0usize;
            let mut best_predicted_within = 0usize;
            let mut best_predicted_score = f64::NEG_INFINITY;

            for g in 0..k {
                let group_size = cache.cluster_sizes[g];
                let invalid_code = group_size; // "invalid" is at index group_size
                let total_classes = group_size + 1;
                let p_s0_g = eval_group_probs[ex * k + g] as f64;

                if all_group_eval_scores[g].is_empty() || p_s0_g < epsilon {
                    gates[g] = 0.0;
                    continue;
                }

                // Compute per-class log-probs from S1_g bitwise scores
                let s1_base = ex * augmented_output_bits;
                let mut s1_log_p = vec![0.0f64; augmented_output_bits];
                let mut s1_log_1mp = vec![0.0f64; augmented_output_bits];
                for b in 0..augmented_output_bits {
                    let p = (all_group_eval_scores[g][s1_base + b] as f64).clamp(eps, 1.0 - eps);
                    s1_log_p[b] = p.ln();
                    s1_log_1mp[b] = (1.0 - p).ln();
                }

                // Softmax over total_classes (valid classes + invalid)
                let mut max_lp = f64::NEG_INFINITY;
                let mut class_log_probs = vec![0.0f64; total_classes];
                for j in 0..total_classes {
                    let mut lp = 0.0f64;
                    for b in 0..augmented_output_bits {
                        if ((j >> (augmented_output_bits - 1 - b)) & 1) == 1 {
                            lp += s1_log_p[b];
                        } else {
                            lp += s1_log_1mp[b];
                        }
                    }
                    class_log_probs[j] = lp;
                    if lp > max_lp { max_lp = lp; }
                }
                let sum_exp: f64 = class_log_probs.iter().map(|&lp| (lp - max_lp).exp()).sum();
                let log_z = max_lp + sum_exp.ln();

                // P(invalid|S1_g) and P(valid|S1_g)
                let p_invalid = (class_log_probs[invalid_code] - log_z).exp();
                let p_valid = (1.0 - p_invalid).max(epsilon);

                // Filtered gate
                gates[g] = p_s0_g * p_valid;

                // If this is the true group, compute S1 contribution
                if g == true_group {
                    let log_p_true_class = class_log_probs[true_within] - log_z;
                    s1_log_p_within_valid = log_p_true_class - p_valid.ln();
                }

                // Track best prediction for accuracy
                let best_j = class_log_probs[..group_size].iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let score = (p_s0_g * p_valid).ln() + class_log_probs[best_j] - log_z - p_valid.ln();
                if score > best_predicted_score {
                    best_predicted_score = score;
                    best_predicted_g = g;
                    best_predicted_within = best_j;
                }
            }

            // Renormalize gates
            let gate_sum: f64 = gates.iter().sum();
            let log_p_filt_true = if gate_sum > epsilon {
                (gates[true_group] / gate_sum).max(epsilon).ln()
            } else {
                -(k as f64).ln() // uniform fallback
            };

            let s0_ce = -log_p_s0;
            let s1_ce = -s1_log_p_within_valid;
            let combined_ce = if label_smoothing > 0.0 {
                let log_p_hier = log_p_filt_true + s1_log_p_within_valid;
                let p_hier = log_p_hier.exp();
                let p_smooth = (1.0 - label_smoothing) * p_hier
                             + label_smoothing / cache.vocab_size as f64;
                -p_smooth.ln()
            } else {
                -log_p_filt_true - s1_log_p_within_valid
            };
            let correct = if best_predicted_g == true_group && best_predicted_within == true_within { 1u32 } else { 0u32 };

            (combined_ce, s0_ce, s1_ce, correct)
        })
        .collect();

    let total_ce: f64 = results.iter().map(|r| r.0).sum();
    let total_s0_ce: f64 = results.iter().map(|r| r.1).sum();
    let total_s1_ce: f64 = results.iter().map(|r| r.2).sum();
    let total_correct: u32 = results.iter().map(|r| r.3).sum();

    (
        total_ce / num_eval as f64,
        total_correct as f64 / num_eval as f64,
        total_s0_ce / num_eval as f64,
        total_s1_ce / num_eval as f64,
    )
}

/// Compute softmax P(group | context) for each example from bitwise per-bit scores.
///
/// `scores`: `[num_examples × num_bits]` — per-bit probabilities from S0 forward
/// `token_bits`: `[K × num_bits]` — bit patterns for each group
///
/// Returns: `[num_examples × K]` — P(group | context) per example
fn compute_group_softmax_bitwise(
    scores: &[f32],
    num_examples: usize,
    num_bits: usize,
    k: usize,
    token_bits: &[u8],
) -> Vec<f32> {
    let eps = 1e-7f64;
    let mut result = vec![0.0f32; num_examples * k];

    for ex in 0..num_examples {
        let mut log_p = vec![0.0f64; num_bits];
        let mut log_1mp = vec![0.0f64; num_bits];
        for b in 0..num_bits {
            let p = (scores[ex * num_bits + b] as f64).clamp(eps, 1.0 - eps);
            log_p[b] = p.ln();
            log_1mp[b] = (1.0 - p).ln();
        }

        let mut max_lp = f64::NEG_INFINITY;
        let mut log_probs = vec![0.0f64; k];
        for gk in 0..k {
            let mut lp = 0.0f64;
            for b in 0..num_bits {
                if token_bits[gk * num_bits + b] == 1 {
                    lp += log_p[b];
                } else {
                    lp += log_1mp[b];
                }
            }
            log_probs[gk] = lp;
            if lp > max_lp { max_lp = lp; }
        }

        let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_lp).exp()).sum();
        let base = ex * k;
        for gk in 0..k {
            result[base + gk] = ((log_probs[gk] - max_lp).exp() / sum_exp) as f32;
        }
    }

    result
}

/// Compute combined CE for SELECTOR mode.
///
/// S0 is evaluated normally (bitwise or tiered) on full eval data.
/// S1 trains K separate sub-models (one per group) using selector data,
/// then combines: P(token) = P(group|ctx) × P(within|group,ctx).
///
/// **Soft-weighted S1 training (Phase B):** When S0 is bitwise, the trained S0 model
/// is re-forwarded on training data to compute P(group|context) per train example.
/// Each S1 group's training subset is weighted by the S0 confidence for that group,
/// giving stronger signal to examples where S0 is more confident.
///
/// Returns: (combined_ce, combined_accuracy, stage0_ce, stage1_ce)
pub fn compute_combined_ce_selector(
    cache: &MultiStageTokenCache,
    stage_bits_per_neuron: &[&[usize]],
    stage_neurons_per_cluster: &[&[usize]],
    stage_connections: &[&[i64]],
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold: usize,
    label_smoothing: f64,
    invalid_mode: bool,
    top_m: usize,
) -> (f64, f64, f64, f64) {
    let eps = 1e-7f64;
    let epsilon = 1e-10f64;
    let num_eval = cache.bitwise_full_eval[0].num_examples;
    let k = cache.k;

    // ── S0: train + forward on eval, optionally keep model for re-forwarding ──
    let mut stage_tiered_scores_0: Vec<f64> = Vec::new();
    let mut stage_bitwise_scores_0: Vec<f32> = Vec::new();
    // S0 model kept for soft-weighting S1 (bitwise S0 only)
    let mut s0_model: Option<(Vec<ClusterStorage>, crate::bitwise_ramlm::GenomeLayout)> = None;

    if cache.stage_is_tiered[0] {
        let train = cache.tiered_full_train[0].as_ref().unwrap();
        let eval = cache.tiered_full_eval[0].as_ref().unwrap();
        stage_tiered_scores_0 = train_and_get_tiered_scores(
            stage_connections[0], stage_bits_per_neuron[0], stage_neurons_per_cluster[0],
            cache.bitwise_vocab_size[0], train, eval, cache.stage_input_bits[0],
            memory_mode, neuron_sample_rate, rng_seed,
        );
    } else {
        let (scores, clusters, layout) = train_and_get_scores_with_model(
            stage_connections[0], stage_bits_per_neuron[0], stage_neurons_per_cluster[0],
            cache.bitwise_output_bits[0],
            &cache.bitwise_full_train[0], &cache.bitwise_full_eval[0],
            memory_mode, neuron_sample_rate, rng_seed, sparse_threshold,
        );
        stage_bitwise_scores_0 = scores;
        s0_model = Some((clusters, layout));
    }

    // ── S1 SELECTOR: train K sub-models, get per-group scores ──
    let s1_stage = 1;
    let s1_output_bits = cache.bitwise_output_bits[s1_stage];
    let selector_eval = &cache.bitwise_selector_eval[s1_stage];

    // Compute P(group|context) per train example from S0 model (shared by Phase B + C)
    let train_group_probs: Option<Vec<f32>> = s0_model.as_ref().map(|(clusters, layout)| {
        let s0_bits = cache.bitwise_output_bits[0];
        let s0_train_scores = forward_scores_on_subset(
            stage_connections[0], stage_bits_per_neuron[0], stage_neurons_per_cluster[0],
            s0_bits, layout,
            &cache.bitwise_full_train[0], clusters, memory_mode,
        );
        let num_train = cache.bitwise_full_train[0].num_examples;
        compute_group_softmax_bitwise(&s0_train_scores, num_train, s0_bits, k, &cache.bitwise_token_bits[0])
    });

    // ── Invalid token mode (Phase C): augmented S1 training + filtered mixture eval ──
    if invalid_mode {
        debug_assert!(
            cache.max_cluster_size < (1usize << s1_output_bits),
            "Invalid mode requires max_cluster_size ({}) < 2^output_bits ({}). \
             Power-of-2 cluster sizes need an extra output bit.",
            cache.max_cluster_size, 1usize << s1_output_bits,
        );

        let num_train = cache.bitwise_full_train[0].num_examples;
        // Fall back to uniform probs if S0 is tiered (no model to re-forward)
        let tgp = train_group_probs.unwrap_or_else(|| vec![1.0 / k as f32; num_train * k]);

        let augmented_subsets = build_augmented_selector_subsets(
            cache, s1_stage, &tgp, s1_output_bits, top_m,
        );

        // Train each group on augmented data, then forward on ALL eval examples
        let all_group_eval_scores: Vec<Vec<f32>> = (0..k)
            .map(|g| {
                if augmented_subsets[g].num_examples == 0 { return Vec::new(); }
                let (clusters, layout) = train_only(
                    stage_connections[s1_stage], stage_bits_per_neuron[s1_stage],
                    stage_neurons_per_cluster[s1_stage], s1_output_bits,
                    &augmented_subsets[g], memory_mode, neuron_sample_rate,
                    rng_seed.wrapping_add(1 + g as u64), sparse_threshold,
                );
                forward_scores_on_eval(
                    stage_connections[s1_stage], stage_bits_per_neuron[s1_stage],
                    stage_neurons_per_cluster[s1_stage], s1_output_bits,
                    &layout, &cache.bitwise_full_eval[s1_stage], &clusters, memory_mode,
                )
            })
            .collect();

        return eval_filtered_mixture(
            cache, &stage_bitwise_scores_0, &stage_tiered_scores_0,
            &all_group_eval_scores, s1_output_bits, label_smoothing,
        );
    }

    // ── Normal mode: weighted S1 training (Phase B) ──
    let weighted_selector_train: Option<Vec<BitwiseSubset>> = train_group_probs.map(|tgp| {
        let context_size = cache.context_size;
        let num_train = cache.bitwise_full_train[0].num_examples;
        let selector_train = &cache.bitwise_selector_train[s1_stage];
        let mut group_weights: Vec<Vec<f32>> = (0..k)
            .map(|g| vec![0.0f32; selector_train[g].num_examples])
            .collect();
        let mut group_counters = vec![0usize; k];

        for orig_idx in 0..num_train {
            let target_token = cache.train_tokens[orig_idx + context_size] as usize;
            let g = cache.cluster_of[target_token] as usize;
            let pos = group_counters[g];
            if pos < group_weights[g].len() {
                group_weights[g][pos] = tgp[orig_idx * k + g];
            }
            group_counters[g] += 1;
        }

        (0..k)
            .map(|g| {
                let orig = &selector_train[g];
                BitwiseSubset {
                    input_bits: Vec::new(),
                    packed_input: orig.packed_input.clone(),
                    target_bits: orig.target_bits.clone(),
                    num_examples: orig.num_examples,
                    words_per_example: orig.words_per_example,
                    weights: std::mem::take(&mut group_weights[g]),
                }
            })
            .collect()
    });

    // Pick which training subsets to use (weighted if available, original otherwise)
    let selector_train_ref: &[BitwiseSubset] = match &weighted_selector_train {
        Some(w) => w,
        None => &cache.bitwise_selector_train[s1_stage],
    };

    // Train each group's sub-model and get scores
    let group_scores: Vec<Vec<f32>> = (0..k)
        .map(|g| {
            if selector_eval[g].num_examples == 0 {
                return Vec::new();
            }
            train_and_get_scores(
                stage_connections[s1_stage],
                stage_bits_per_neuron[s1_stage],
                stage_neurons_per_cluster[s1_stage],
                s1_output_bits,
                &selector_train_ref[g], &selector_eval[g],
                memory_mode, neuron_sample_rate,
                rng_seed.wrapping_add(1 + g as u64), sparse_threshold,
            )
        })
        .collect();

    // ── Build mapping: global eval index → (group, position_in_group) ──
    let mut group_pos_counter = vec![0usize; k];
    let eval_to_group_pos: Vec<(usize, usize)> = (0..num_eval)
        .map(|ex| {
            let token_id = cache.eval_target_token_ids[ex] as usize;
            let group = cache.cluster_of[token_id] as usize;
            let pos = group_pos_counter[group];
            group_pos_counter[group] += 1;
            (group, pos)
        })
        .collect();

    // ── Combine per-example ──
    let results: Vec<(f64, f64, f64, u32)> = (0..num_eval)
        .into_par_iter()
        .map(|ex| {
            let token_id = cache.eval_target_token_ids[ex] as usize;
            let true_group = cache.cluster_of[token_id] as usize;
            let true_within = cache.index_in_cluster[token_id] as usize;

            // ── S0: compute log P(true_group) ──
            let (log_p_true_group, s0_predicted) = if cache.stage_is_tiered[0] {
                let base = ex * k;
                let scores = &stage_tiered_scores_0[base..base + k];
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let log_p = (exp_scores[true_group] / sum_exp + epsilon).ln();
                let predicted = scores.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx).unwrap_or(0);
                (log_p, predicted)
            } else {
                let s0_bits = cache.bitwise_output_bits[0];
                let s0_base = ex * s0_bits;
                let mut s0_log_p = vec![0.0f64; s0_bits];
                let mut s0_log_1mp = vec![0.0f64; s0_bits];
                for b in 0..s0_bits {
                    let p = (stage_bitwise_scores_0[s0_base + b] as f64).clamp(eps, 1.0 - eps);
                    s0_log_p[b] = p.ln();
                    s0_log_1mp[b] = (1.0 - p).ln();
                }

                let mut max_s0 = f64::NEG_INFINITY;
                let mut sum_exp_s0 = 0.0f64;
                let mut log_p_true_raw = 0.0f64;
                let mut predicted = 0usize;
                let mut predicted_lp = f64::NEG_INFINITY;

                for gk in 0..k {
                    let bit_base = gk * s0_bits;
                    let mut log_p = 0.0f64;
                    for b in 0..s0_bits {
                        if cache.bitwise_token_bits[0][bit_base + b] == 1 {
                            log_p += s0_log_p[b];
                        } else {
                            log_p += s0_log_1mp[b];
                        }
                    }
                    if gk == true_group { log_p_true_raw = log_p; }
                    if log_p > predicted_lp { predicted_lp = log_p; predicted = gk; }
                    if log_p > max_s0 {
                        sum_exp_s0 = sum_exp_s0 * (max_s0 - log_p).exp() + 1.0;
                        max_s0 = log_p;
                    } else {
                        sum_exp_s0 += (log_p - max_s0).exp();
                    }
                }
                let log_z0 = max_s0 + sum_exp_s0.ln();
                (log_p_true_raw - log_z0, predicted)
            };

            // ── S1 SELECTOR: compute log P(true_within | true_group) ──
            let (group, pos_in_group) = eval_to_group_pos[ex];
            let group_size = cache.cluster_sizes[true_group];

            let (log_p_true_within, s1_predicted) = if group_scores[group].is_empty() {
                let log_uniform = -(group_size as f64).ln();
                (log_uniform, 0)
            } else {
                let s1_base = pos_in_group * s1_output_bits;
                let mut s1_log_p = vec![0.0f64; s1_output_bits];
                let mut s1_log_1mp = vec![0.0f64; s1_output_bits];
                for b in 0..s1_output_bits {
                    let p = (group_scores[group][s1_base + b] as f64).clamp(eps, 1.0 - eps);
                    s1_log_p[b] = p.ln();
                    s1_log_1mp[b] = (1.0 - p).ln();
                }

                let mut max_s1 = f64::NEG_INFINITY;
                let mut sum_exp_s1 = 0.0f64;
                let mut log_p_true_raw = 0.0f64;
                let mut predicted = 0usize;
                let mut predicted_lp = f64::NEG_INFINITY;

                for j in 0..group_size {
                    let bit_base = j * s1_output_bits;
                    let mut log_p = 0.0f64;
                    for b in 0..s1_output_bits {
                        if cache.bitwise_token_bits[s1_stage][bit_base + b] == 1 {
                            log_p += s1_log_p[b];
                        } else {
                            log_p += s1_log_1mp[b];
                        }
                    }
                    if j == true_within { log_p_true_raw = log_p; }
                    if log_p > predicted_lp { predicted_lp = log_p; predicted = j; }
                    if log_p > max_s1 {
                        sum_exp_s1 = sum_exp_s1 * (max_s1 - log_p).exp() + 1.0;
                        max_s1 = log_p;
                    } else {
                        sum_exp_s1 += (log_p - max_s1).exp();
                    }
                }
                let log_z1 = max_s1 + sum_exp_s1.ln();
                (log_p_true_raw - log_z1, predicted)
            };

            let s0_ce = -log_p_true_group;
            let s1_ce = -log_p_true_within;
            let combined_ce = if label_smoothing > 0.0 {
                let log_p_hier = log_p_true_group + log_p_true_within;
                let p_hier = log_p_hier.exp();
                let p_smooth = (1.0 - label_smoothing) * p_hier
                             + label_smoothing / cache.vocab_size as f64;
                -p_smooth.ln()
            } else {
                s0_ce + s1_ce
            };
            let correct = if s0_predicted == true_group && s1_predicted == true_within { 1u32 } else { 0u32 };

            (combined_ce, s0_ce, s1_ce, correct)
        })
        .collect();

    let total_ce: f64 = results.iter().map(|r| r.0).sum();
    let total_s0_ce: f64 = results.iter().map(|r| r.1).sum();
    let total_s1_ce: f64 = results.iter().map(|r| r.2).sum();
    let total_correct: u32 = results.iter().map(|r| r.3).sum();

    (
        total_ce / num_eval as f64,
        total_correct as f64 / num_eval as f64,
        total_s0_ce / num_eval as f64,
        total_s1_ce / num_eval as f64,
    )
}

// ── Tiered evaluation ───────────────────────────────────────────────────

/// Evaluate tiered genomes for a given stage using the adaptive evaluation pipeline.
///
/// Delegates to `adaptive::evaluate_genomes_parallel_hybrid()` which uses Metal GPU + rayon CPU
/// with target+negatives training and softmax CE.
///
/// Returns: Vec<(ce, accuracy)> per genome.
pub fn evaluate_tiered_genomes(
    cache: &MultiStageTokenCache,
    stage: usize,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64)> {
    let train_subs = match &cache.tiered_train_subsets[stage] {
        Some(subs) => subs,
        None => return vec![(f64::MAX, 0.0); num_genomes],
    };
    let eval_subs = match &cache.tiered_eval_subsets[stage] {
        Some(subs) => subs,
        None => return vec![(f64::MAX, 0.0); num_genomes],
    };

    let train = &train_subs[train_subset_idx % train_subs.len()];
    let eval = &eval_subs[eval_subset_idx % eval_subs.len()];

    let total_input_bits = cache.stage_input_bits[stage];
    let num_clusters = cache.stage_num_output_clusters[stage];

    let empty_value = empty_value_for_mode(memory_mode);
    // Sync globals so GPU shaders use the correct mode and empty_value
    crate::neuron_memory::set_empty_value(empty_value);
    crate::neuron_memory::set_memory_mode(memory_mode);

    crate::adaptive::evaluate_genomes_parallel_hybrid(
        bits_per_neuron_flat,
        neurons_per_cluster_flat,
        connections_flat,
        num_genomes,
        num_clusters,
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        train.num_negatives,
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        total_input_bits,
        empty_value,
        neuron_sample_rate,
        rng_seed,
    )
}

/// Evaluate tiered genomes with full (non-rotated) data.
pub fn evaluate_tiered_genomes_full(
    cache: &MultiStageTokenCache,
    stage: usize,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64)> {
    let train = match &cache.tiered_full_train[stage] {
        Some(t) => t,
        None => return vec![(f64::MAX, 0.0); num_genomes],
    };
    let eval = match &cache.tiered_full_eval[stage] {
        Some(e) => e,
        None => return vec![(f64::MAX, 0.0); num_genomes],
    };

    let total_input_bits = cache.stage_input_bits[stage];
    let num_clusters = cache.stage_num_output_clusters[stage];

    let empty_value = empty_value_for_mode(memory_mode);
    crate::neuron_memory::set_empty_value(empty_value);
    crate::neuron_memory::set_memory_mode(memory_mode);

    crate::adaptive::evaluate_genomes_parallel(
        bits_per_neuron_flat,
        neurons_per_cluster_flat,
        connections_flat,
        num_genomes,
        num_clusters,
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        train.num_negatives,
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        total_input_bits,
        empty_value,
        neuron_sample_rate,
        rng_seed,
    )
}

/// Train a single genome and return raw per-example scores [num_eval × num_clusters].
///
/// This extracts the train+forward core from `evaluate_genomes_parallel()` for
/// a single genome, returning raw scores instead of CE. Needed by `compute_combined_ce()`
/// to mix tiered and bitwise stage scores.
pub fn train_and_get_tiered_scores(
    connections: &[i64],
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    num_clusters: usize,
    train_data: &TieredSubset,
    eval_data: &TieredEvalSubset,
    total_input_bits: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<f64> {
    let empty_value = empty_value_for_mode(memory_mode);
    crate::neuron_memory::set_empty_value(empty_value);
    crate::neuron_memory::set_memory_mode(memory_mode);

    // Build cluster metadata
    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(bits_per_neuron, neurons_per_cluster);
    let per_cluster_bits = per_cluster_max_bits(bits_per_neuron, neurons_per_cluster);
    let groups = build_groups(&per_cluster_bits, neurons_per_cluster);

    // Build cluster → group mapping
    let mut cluster_to_group = vec![(0usize, 0usize); num_clusters];
    for (gi, group) in groups.iter().enumerate() {
        for (local_idx, &cid) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cid] = (gi, local_idx);
        }
    }

    // Create memory
    let memories: Vec<GroupMemory> = groups.iter()
        .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
        .collect();

    // Try GPU address computation
    let gpu_addresses = try_gpu_addresses_adaptive(
        &train_data.packed_input,
        train_data.words_per_example,
        bits_per_neuron,
        &neuron_conn_offsets,
        connections,
        train_data.num_examples,
    );

    // Train
    train_genome_in_slot(
        &memories,
        &groups,
        connections,
        bits_per_neuron,
        &cluster_neuron_starts,
        &neuron_conn_offsets,
        &cluster_to_group,
        &train_data.input_bits,
        &train_data.targets,
        &train_data.negatives,
        train_data.num_examples,
        train_data.num_negatives,
        total_input_bits,
        gpu_addresses.as_deref(),
        neuron_sample_rate,
        rng_seed,
        memory_mode,
    );

    // Evaluate — build per-example scores
    let num_eval = eval_data.num_examples;
    let mut all_scores = vec![vec![0.0f64; num_clusters]; num_eval];

    // Reorganize connections for GPU
    let connections_flat = reorganize_connections_for_gpu(
        connections, bits_per_neuron, neurons_per_cluster, &groups,
    );
    let (packed_eval, words_per_example) =
        pack_bools_to_u64(&eval_data.input_bits, num_eval, total_input_bits);

    for (group_idx, (group, memory)) in groups.iter().zip(memories.iter()).enumerate() {
        // Try Metal GPU (dense groups)
        if memory.is_dense() {
            if let Some(metal) = get_metal_evaluator() {
                if let Some(mem_words) = memory.export_for_metal() {
                    if let Ok(group_scores) = evaluate_group_metal(
                        &metal, &packed_eval, &connections_flat, &mem_words,
                        group, num_eval, words_per_example,
                        memory_mode,
                    ) {
                        for ex_idx in 0..num_eval {
                            for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                            }
                        }
                        continue;
                    }
                }
            }
        }

        // Try GPU sparse (sparse groups)
        if memory.is_sparse() {
            if let Some(sparse_eval) = get_sparse_metal_evaluator() {
                if let Some(export) = memory.export_for_gpu_sparse() {
                    if let Ok(group_scores) = evaluate_group_sparse_gpu(
                        &sparse_eval, &packed_eval, &connections_flat, &export,
                        group, num_eval, words_per_example,
                        memory_mode,
                    ) {
                        for ex_idx in 0..num_eval {
                            for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                            }
                        }
                        continue;
                    }
                }
            }
        }

        // CPU fallback
        let _ = group_idx; // suppress unused warning
        all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
            let input_start = ex_idx * total_input_bits;
            let input_bits = &eval_data.input_bits[input_start..input_start + total_input_bits];

            for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                let actual_neurons = if let Some(ref an) = group.actual_neurons {
                    an[local_cluster] as usize
                } else {
                    group.neurons
                };

                let neuron_base = local_cluster * group.neurons;

                let mut sum = 0.0f32;
                for n in 0..actual_neurons {
                    let global_n = cluster_neuron_starts[cluster_id] + n;
                    let n_bits = bits_per_neuron[global_n];
                    let conn_start = neuron_conn_offsets[global_n];
                    let address = compute_address(input_bits, &connections[conn_start..], n_bits);
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

    // Flatten scores: [num_eval × num_clusters]
    let mut flat_scores = Vec::with_capacity(num_eval * num_clusters);
    for ex_scores in &all_scores {
        flat_scores.extend_from_slice(ex_scores);
    }
    flat_scores
}
