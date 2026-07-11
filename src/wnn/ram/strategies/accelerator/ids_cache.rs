//! IDS Cache - Pre-encoded binary features for intrusion detection classification.
//!
//! Parallel to TokenCache but for tabular binary classification (IDS).
//! Key differences from TokenCache:
//! - Input: flat binary features (already encoded), not token sequences
//! - No context windows, no token encoding
//! - Stratified partitioning (preserves class distribution in subsets)
//! - Exhaustive negatives (all other classes, not sampled from top-K)

use rand::prelude::*;
use rand::SeedableRng;
use std::sync::{Arc, RwLock};

use crate::neighbor_search::LiveProgress;
use ram_core::packed_bits::PackedBits;
use crate::token_cache::SubsetRotator;

/// Pre-computed IDS subset with all data needed for evaluation.
///
/// Storage is bit-packed (8 bits per byte) — 8x less memory than the legacy
/// Vec<bool> layout. Consumers read via `input_bits.bit(i, j)` or migrate
/// to packed-byte access via `input_bits.packed_row(i)`.
#[derive(Clone)]
pub struct IDSSubset {
    /// Encoded input bits, `num_examples × total_features` bit-packed.
    pub input_bits: PackedBits,
    /// Target class indices: [num_examples]
    pub targets: Vec<i64>,
    /// Negative class indices: [num_examples * num_negatives]
    pub negatives: Vec<i64>,
    /// Number of examples in this subset
    pub num_examples: usize,
}

/// Persistent IDS cache holding all features for the session.
///
/// Created once at session start, then used for all evaluations.
/// Provides zero-copy subset selection via pre-computed indices.
pub struct IDSCache {
    // Configuration
    num_classes: usize,
    total_features: usize,
    num_negatives: usize,
    num_parts: usize,

    // Pre-computed subsets for train (stratified)
    train_subsets: Vec<IDSSubset>,

    // Full datasets for final evaluation
    full_train: IDSSubset,
    full_eval: IDSSubset,

    // Optional validation partition (Protocol v2, HF `_3way` splits):
    // threshold calibrations fit on val scores, test stays report-only.
    full_val: Option<IDSSubset>,

    // Rotator for train subsets
    train_rotator: SubsetRotator,

    // Class weights for balanced training (None = unweighted)
    class_weights: Option<Vec<u32>>,

    // Single-cluster discriminator mode: 1 cluster, threshold at 0.5
    num_genome_clusters: usize,

    // Which class index represents "normal/benign" for FPR computation.
    // Default 0; set to 1 when flip_labels is active so FPR is always
    // the false alarm rate on original benign traffic.
    pub normal_class: usize,

    // Fitness weights for threshold optimization.
    // When set, threshold sweep maximizes fitness instead of F1.
    pub fitness_weights: Option<(f32, f32, f32, f32)>,  // (w_ce, w_f1, w_fpr, w_acc)

    // Live progress for observer thread
    pub live_progress: Arc<RwLock<Option<LiveProgress>>>,
}

/// Accumulator for chunked IDS cache construction (Phase 5 F-prep).
///
/// Use case: incremental data ingestion (Option F's streaming path) where
/// the full train/eval matrices aren't materialized in one go. The builder
/// extends `PackedBits` row-by-row across multiple chunks and finally
/// hands off to `IDSCache::new` once `finalize()` is called.
///
/// For Phase 2-onward, callers continue to use `IDSCache::new` directly
/// (single-chunk path). The builder is dormant infrastructure that Option F
/// will light up.
pub struct PartialFitState {
    train_features: PackedBits,
    train_labels: Vec<i64>,
    eval_features: PackedBits,
    eval_labels: Vec<i64>,

    // Frozen build params (set in `new`, used in `finalize`).
    num_classes: usize,
    total_features: usize,
    num_parts: usize,
    num_negatives: usize,
    seed: u64,
    balance_classes: bool,
    single_cluster: bool,
    undersample_majority: bool,
    class_weight_multiplier: f32,
}

impl PartialFitState {
    /// Construct an empty accumulator with frozen build parameters.
    pub fn new(
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
    ) -> Self {
        Self {
            train_features: PackedBits::empty(total_features),
            train_labels: Vec::new(),
            eval_features: PackedBits::empty(total_features),
            eval_labels: Vec::new(),
            num_classes,
            total_features,
            num_parts,
            num_negatives,
            seed,
            balance_classes,
            single_cluster,
            undersample_majority,
            class_weight_multiplier,
        }
    }

    /// Append a chunk of train data (packed bytes + labels).
    pub fn add_train_chunk(&mut self, chunk: &PackedBits, labels: &[i64]) {
        assert_eq!(
            chunk.num_rows(),
            labels.len(),
            "add_train_chunk: chunk rows ({}) != labels ({})",
            chunk.num_rows(),
            labels.len()
        );
        assert_eq!(
            chunk.total_bits(),
            self.total_features,
            "add_train_chunk: chunk total_bits ({}) != PartialFitState total_features ({})",
            chunk.total_bits(),
            self.total_features
        );
        self.train_features.extend_from(chunk);
        self.train_labels.extend_from_slice(labels);
    }

    /// Append a chunk of eval data (packed bytes + labels).
    pub fn add_eval_chunk(&mut self, chunk: &PackedBits, labels: &[i64]) {
        assert_eq!(chunk.num_rows(), labels.len());
        assert_eq!(chunk.total_bits(), self.total_features);
        self.eval_features.extend_from(chunk);
        self.eval_labels.extend_from_slice(labels);
    }

    /// Number of train rows accumulated so far.
    pub fn num_train(&self) -> usize {
        self.train_labels.len()
    }

    /// Number of eval rows accumulated so far.
    pub fn num_eval(&self) -> usize {
        self.eval_labels.len()
    }

    /// Logical bit width per row (frozen at construction).
    pub fn total_features(&self) -> usize {
        self.total_features
    }

    /// Consume the accumulator and build the IDSCache via the standard
    /// `IDSCache::new` path. Equivalent to the one-shot constructor when
    /// all data was added in a single chunk.
    pub fn finalize(self) -> IDSCache {
        IDSCache::new(
            self.train_features,
            self.train_labels,
            self.eval_features,
            self.eval_labels,
            // Protocol v2: the builder carries no val partition (no Python
            // consumer). Streaming flows score val via IDSGenomeStreamer
            // multi-set passes instead — see ids_evaluator's
            // _evaluate_at_thresholds_streaming.
            None,
            None,
            self.num_classes,
            self.total_features,
            self.num_parts,
            self.num_negatives,
            self.seed,
            self.balance_classes,
            self.single_cluster,
            self.undersample_majority,
            self.class_weight_multiplier,
        )
    }
}

impl IDSCache {
    /// Create a new IDS cache with stratified partitioning.
    ///
    /// # Arguments
    /// * `train_features` - Flattened training features [num_train * total_features] as bools
    /// * `train_labels` - Training labels [num_train] as class indices
    /// * `eval_features` - Flattened eval features [num_eval * total_features]
    /// * `eval_labels` - Eval labels [num_eval]
    /// * `num_classes` - Number of output classes (2 for binary, 10 for multi)
    /// * `total_features` - Number of binary features per example
    /// * `num_parts` - Number of train subsets for rotation
    /// * `num_negatives` - Negatives per example (typically num_classes - 1)
    /// * `seed` - Random seed for partitioning and rotation
    /// * `val_features`/`val_labels` - Optional validation partition (Protocol
    ///   v2, `_3way` splits). Built into an IDSSubset like full_eval; None for
    ///   legacy 2-way datasets.
    pub fn new(
        train_features: PackedBits,
        train_labels: Vec<i64>,
        eval_features: PackedBits,
        eval_labels: Vec<i64>,
        val_features: Option<PackedBits>,
        val_labels: Option<Vec<i64>>,
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
    ) -> Self {
        // Optionally undersample majority class to match minority count
        let (train_features, train_labels) = if undersample_majority {
            Self::undersample_to_balance(train_features, train_labels, num_classes, seed)
        } else {
            (train_features, train_labels)
        };

        let num_train = train_labels.len();
        let num_eval = eval_labels.len();

        assert_eq!(train_features.num_rows(), num_train,
            "train_features rows mismatch: {} != {}",
            train_features.num_rows(), num_train);
        assert_eq!(train_features.total_bits(), total_features,
            "train_features total_bits mismatch: {} != {}",
            train_features.total_bits(), total_features);
        assert_eq!(eval_features.num_rows(), num_eval,
            "eval_features rows mismatch: {} != {}",
            eval_features.num_rows(), num_eval);
        assert_eq!(eval_features.total_bits(), total_features,
            "eval_features total_bits mismatch: {} != {}",
            eval_features.total_bits(), total_features);

        // Single-cluster mode: 1 genome cluster, 0 negatives
        let num_genome_clusters = if single_cluster { 1 } else { num_classes };
        let actual_negatives = if single_cluster { 0 } else { num_negatives };

        // Build full train subset
        let full_train = Self::build_subset(
            &train_features, &train_labels, total_features, num_classes, actual_negatives, seed,
        );

        // Build full eval subset
        let full_eval = Self::build_subset(
            &eval_features, &eval_labels, total_features, num_classes, actual_negatives, seed + 1,
        );

        // Build full val subset (Protocol v2) — same construction as full_eval
        let full_val = match (val_features, val_labels) {
            (Some(vf), Some(vl)) => {
                assert_eq!(vf.num_rows(), vl.len(),
                    "val_features rows mismatch: {} != {}",
                    vf.num_rows(), vl.len());
                assert_eq!(vf.total_bits(), total_features,
                    "val_features total_bits mismatch: {} != {}",
                    vf.total_bits(), total_features);
                Some(Self::build_subset(
                    &vf, &vl, total_features, num_classes, actual_negatives, seed + 2,
                ))
            }
            (None, None) => None,
            _ => panic!("val_features and val_labels must both be Some or both be None"),
        };

        // Stratified partitioning of training data
        let train_subsets = Self::create_stratified_subsets(
            &train_features, &train_labels, num_classes,
            actual_negatives, num_parts, seed,
        );

        // Compute class weights: max_count / count per class * multiplier (upweights minority)
        let class_weights = if balance_classes {
            Some(crate::adaptive::compute_class_weights_with_multiplier(&train_labels, num_classes, class_weight_multiplier))
        } else {
            None
        };

        Self {
            num_classes,
            total_features,
            num_negatives: actual_negatives,
            num_parts,
            train_subsets,
            full_train,
            full_eval,
            full_val,
            class_weights,
            num_genome_clusters,
            train_rotator: SubsetRotator::new(num_parts, seed + 100),
            live_progress: Arc::new(RwLock::new(None)),
            normal_class: 0,
            fitness_weights: None,
        }
    }

    /// Build an IDSSubset from features and labels.
    ///
    /// Negatives: for each example, all classes except the target.
    /// Undersample the majority class to match the minority class count.
    /// Returns balanced (features, labels) with equal representation per class.
    fn undersample_to_balance(
        features: PackedBits,
        labels: Vec<i64>,
        num_classes: usize,
        seed: u64,
    ) -> (PackedBits, Vec<i64>) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let num_examples = labels.len();
        if num_examples == 0 {
            return (features, labels);
        }

        // Group example indices by class
        let mut class_indices: Vec<Vec<usize>> = vec![vec![]; num_classes];
        for (i, &label) in labels.iter().enumerate() {
            let c = label as usize;
            if c < num_classes {
                class_indices[c].push(i);
            }
        }

        // Find minimum class count
        let min_count = class_indices.iter()
            .filter(|v| !v.is_empty())
            .map(|v| v.len())
            .min()
            .unwrap_or(0);

        if min_count == 0 {
            return (features, labels);
        }

        // Shuffle and take min_count from each class
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 999);
        let mut selected_indices: Vec<usize> = Vec::with_capacity(min_count * num_classes);
        for indices in &mut class_indices {
            if !indices.is_empty() {
                indices.shuffle(&mut rng);
                selected_indices.extend_from_slice(&indices[..min_count.min(indices.len())]);
            }
        }
        selected_indices.sort_unstable(); // Preserve original order

        let original_count = num_examples;
        let balanced_count = selected_indices.len();

        // Build balanced features and labels
        let new_features = features.select_rows(&selected_indices);
        let mut new_labels = Vec::with_capacity(balanced_count);
        for &idx in &selected_indices {
            new_labels.push(labels[idx]);
        }

        eprintln!(
            "[IDS_CACHE] Undersampled majority: {} → {} examples ({} per class, {} classes)",
            original_count, balanced_count, min_count, num_classes
        );

        (new_features, new_labels)
    }

    /// If num_negatives < num_classes - 1, randomly sample from non-target classes.
    fn build_subset(
        features: &PackedBits,
        labels: &[i64],
        total_features: usize,
        num_classes: usize,
        num_negatives: usize,
        seed: u64,
    ) -> IDSSubset {
        let num_examples = labels.len();
        if num_examples == 0 {
            return IDSSubset {
                input_bits: PackedBits::empty(total_features),
                targets: vec![],
                negatives: vec![],
                num_examples: 0,
            };
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Input bits are already packed — clone the PackedBits
        let input_bits = features.clone();
        let targets = labels.to_vec();

        // Build negatives: all other classes for each example
        let mut negatives = vec![0i64; num_examples * num_negatives];
        let all_negatives_needed = num_classes - 1;

        for ex in 0..num_examples {
            let target = labels[ex];
            let neg_offset = ex * num_negatives;

            if num_negatives >= all_negatives_needed {
                // Exhaustive: all classes except target
                let mut neg_idx = 0;
                for c in 0..num_classes as i64 {
                    if c != target {
                        negatives[neg_offset + neg_idx] = c;
                        neg_idx += 1;
                    }
                }
                // If num_negatives > num_classes - 1, fill remaining with random
                while neg_idx < num_negatives {
                    let mut c = rng.gen_range(0..num_classes) as i64;
                    while c == target {
                        c = rng.gen_range(0..num_classes) as i64;
                    }
                    negatives[neg_offset + neg_idx] = c;
                    neg_idx += 1;
                }
            } else {
                // Sample num_negatives from non-target classes
                let mut available: Vec<i64> = (0..num_classes as i64)
                    .filter(|&c| c != target)
                    .collect();
                available.shuffle(&mut rng);
                for k in 0..num_negatives {
                    negatives[neg_offset + k] = available[k % available.len()];
                }
            }
        }

        IDSSubset {
            input_bits,
            targets,
            negatives,
            num_examples,
        }
    }

    /// Create stratified subsets by distributing examples round-robin by class.
    ///
    /// This preserves class distribution in each subset, which is critical
    /// for imbalanced IDS datasets (e.g., Worms has only 44 examples).
    fn create_stratified_subsets(
        features: &PackedBits,
        labels: &[i64],
        num_classes: usize,
        num_negatives: usize,
        num_parts: usize,
        seed: u64,
    ) -> Vec<IDSSubset> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 50);
        let total_features = features.total_bits();

        // Group example indices by class
        let mut class_indices: Vec<Vec<usize>> = vec![vec![]; num_classes];
        for (i, &label) in labels.iter().enumerate() {
            let class_idx = label as usize;
            if class_idx < num_classes {
                class_indices[class_idx].push(i);
            }
        }

        // Shuffle within each class for randomization
        for indices in &mut class_indices {
            indices.shuffle(&mut rng);
        }

        // Distribute round-robin to subsets
        let mut subset_indices: Vec<Vec<usize>> = vec![vec![]; num_parts];
        for class_idx_list in &class_indices {
            for (i, &ex_idx) in class_idx_list.iter().enumerate() {
                subset_indices[i % num_parts].push(ex_idx);
            }
        }

        // Shuffle within each subset to mix classes
        for indices in &mut subset_indices {
            indices.shuffle(&mut rng);
        }

        // Build subsets from indices
        subset_indices.iter().enumerate().map(|(part_idx, indices)| {
            let sub_features = features.select_rows(indices);
            let sub_labels: Vec<i64> = indices.iter().map(|&idx| labels[idx]).collect();

            Self::build_subset(
                &sub_features, &sub_labels, total_features,
                num_classes, num_negatives, seed + (part_idx as u64 + 1) * 1000,
            )
        }).collect()
    }

    // ── Accessors ──────────────────────────────────────────────────────

    pub fn next_train_idx(&mut self) -> usize {
        self.train_rotator.next()
    }

    pub fn train_subset(&self, idx: usize) -> &IDSSubset {
        &self.train_subsets[idx]
    }

    pub fn full_train(&self) -> &IDSSubset {
        &self.full_train
    }

    pub fn full_eval(&self) -> &IDSSubset {
        &self.full_eval
    }

    pub fn full_val(&self) -> Option<&IDSSubset> {
        self.full_val.as_ref()
    }

    pub fn total_features(&self) -> usize {
        self.total_features
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn num_negatives(&self) -> usize {
        self.num_negatives
    }

    pub fn num_genome_clusters(&self) -> usize {
        self.num_genome_clusters
    }

    pub fn num_train_subsets(&self) -> usize {
        self.train_subsets.len()
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.train_rotator.reset(seed);
    }

    pub fn num_parts(&self) -> usize {
        self.num_parts
    }
}

// ── Evaluation functions (delegate to adaptive.rs) ─────────────────────

/// Evaluate genomes using cached IDS data with a specific train subset.
pub fn evaluate_genomes_ids_cached_hybrid(
    cache: &IDSCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    train_subset_idx: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64, f64, f64, f64, u32)> {
    let train = cache.train_subset(train_subset_idx);
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genomes_parallel_hybrid(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    )
}

/// Merge all subsets except the excluded one into a single IDSSubset.
///
/// Used for K-fold cross-validation: K-1 folds become the training set,
/// the excluded fold becomes the eval set.
fn merge_subsets_except(subsets: &[&IDSSubset], exclude_idx: usize) -> IDSSubset {
    let total_features = if subsets.is_empty() {
        0
    } else {
        subsets[0].input_bits.total_bits()
    };

    let mut merged_input_bits = PackedBits::empty(total_features);
    let mut merged_targets = Vec::new();
    let mut merged_negatives = Vec::new();
    let mut merged_num_examples = 0;

    let num_negatives_per_example = if subsets.is_empty() || subsets[0].num_examples == 0 {
        0
    } else {
        subsets[0].negatives.len() / subsets[0].num_examples
    };

    for (i, subset) in subsets.iter().enumerate() {
        if i == exclude_idx {
            continue;
        }
        merged_input_bits.extend_from(&subset.input_bits);
        merged_targets.extend_from_slice(&subset.targets);
        merged_negatives.extend_from_slice(&subset.negatives);
        merged_num_examples += subset.num_examples;
    }

    // Sanity checks
    if merged_num_examples > 0 {
        assert_eq!(
            merged_input_bits.num_rows(),
            merged_num_examples,
            "merge_subsets_except: input_bits rows mismatch"
        );
        assert_eq!(
            merged_negatives.len(),
            merged_num_examples * num_negatives_per_example,
            "merge_subsets_except: negatives length mismatch"
        );
    }

    IDSSubset {
        input_bits: merged_input_bits,
        targets: merged_targets,
        negatives: merged_negatives,
        num_examples: merged_num_examples,
    }
}

/// Evaluate genomes using K-fold cross-validation.
///
/// Merges all subsets except `held_out_fold` for training, uses the held-out
/// fold as the eval set. This gives a more robust estimate of F1/FPR since
/// every example eventually serves as both train and eval.
pub fn evaluate_genomes_ids_kfold_hybrid(
    cache: &IDSCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    held_out_fold: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64, f64, f64, f64, u32)> {
    assert!(
        held_out_fold < cache.num_parts(),
        "held_out_fold {} >= num_parts {}",
        held_out_fold,
        cache.num_parts(),
    );

    // Build train set from all folds except the held-out one
    // Borrow the subsets directly — merge_subsets_except only reads them, so the
    // old `owned_subsets = all_subsets.clone()` (a full copy of the entire packed
    // train set, ~8.6 GB, allocated on EVERY fold eval) was pure waste.
    let all_subsets: Vec<&IDSSubset> = (0..cache.num_parts())
        .map(|i| cache.train_subset(i))
        .collect();
    let train = merge_subsets_except(&all_subsets, held_out_fold);

    // Use the held-out fold as eval
    let eval = cache.train_subset(held_out_fold);

    crate::adaptive::evaluate_genomes_parallel_hybrid(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    )
}

/// Evaluate genomes using full cached IDS data (for final evaluation).
pub fn evaluate_genomes_ids_cached_full_hybrid(
    cache: &IDSCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    override_threshold: Option<f64>,
) -> Vec<(f64, f64, f64, f64, f64, u32)> {
    let train = cache.full_train();
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genomes_parallel_hybrid_with_override(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
        override_threshold,
    )
}

/// Train a single genome on full training data and return per-example predictions on eval set.
///
/// Used by the bitwise ECOC classifier for per-bit predictions that are
/// combined via nearest-codeword decoding.
pub fn predict_examples_ids_cached(
    cache: &IDSCache,
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<i64> {
    let train = cache.full_train();
    let eval = cache.full_eval();

    crate::adaptive::train_and_predict_single(
        bits_flat,
        neurons_flat,
        connections_flat,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        eval.num_examples,
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    )
}

/// Train a single genome on full training data and return per-example RAW SCORES.
///
/// Returns Vec<f64> of raw scores (not thresholded) for each eval example.
/// Used for Platt scaling calibration.
pub fn score_examples_ids_cached(
    cache: &IDSCache,
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<f64> {
    let train = cache.full_train();
    let eval = cache.full_eval();

    crate::adaptive::train_and_score_single(
        bits_flat,
        neurons_flat,
        connections_flat,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        eval.num_examples,
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    )
}

/// Train ONCE and evaluate at multiple thresholds, returning eval/train scores
/// and per-threshold metrics.
///
/// Replaces the validation-phase pattern of calling `evaluate_genomes_ids_cached_full_hybrid`
/// once per threshold (each with its own training pass) — instead, trains the
/// genome a single time, scores both eval and train sets from the trained memory,
/// then derives (CE, accuracy, F1, FPR) at each requested threshold.
///
/// Threshold sentinels:
///   - `t >= 0.0`: fixed threshold
///   - `t == -1.0`: oracle — find optimal F1 threshold on EVAL scores
///                  (matches evaluate_genomes_parallel_hybrid_with_override semantics)
///
/// Returns `(eval_scores, train_scores, val_scores, metrics_per_threshold)` where
/// `metrics_per_threshold[i] = (ce, acc, f1, fpr, resolved_threshold)` and
/// `val_scores` is Some when the cache holds a val partition (Protocol v2),
/// None otherwise. The -1.0 oracle sentinel stays computed on EVAL scores —
/// Python decides what to do with val_scores.
pub fn evaluate_at_thresholds_ids_cached(
    cache: &IDSCache,
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    thresholds: &[f64],
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Vec<(f64, f64, f64, f64, f64)>) {
    let train = cache.full_train();
    let eval = cache.full_eval();
    let val = cache.full_val();

    // Train ONCE + score eval, train (and val, when present) sets
    let (eval_scores, train_scores, val_scores) = crate::adaptive::train_and_score_eval_and_train(
        bits_flat,
        neurons_flat,
        connections_flat,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        eval.num_examples,
        val.map(|v| &v.input_bits),
        val.map_or(0, |v| v.num_examples),
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    );

    // Derive metrics at each threshold (handle -1.0 oracle sentinel)
    let normal_class = cache.normal_class;
    let metrics: Vec<(f64, f64, f64, f64, f64)> = thresholds.iter().map(|&t| {
        let resolved = if t < 0.0 {
            // Oracle: sweep eval scores for optimal F1
            let (best_t, _f1, _fpr) =
                crate::adaptive::find_optimal_threshold_f1(&eval_scores, &eval.targets);
            best_t
        } else {
            t
        };
        let (ce, acc, f1, fpr) = crate::adaptive::compute_binary_metrics_at_threshold(
            &eval_scores, &eval.targets, resolved, normal_class,
        );
        (ce, acc, f1, fpr, resolved)
    }).collect();

    (eval_scores, train_scores, val_scores, metrics)
}

/// Multiclass (K-cluster) analogue of `evaluate_at_thresholds_ids_cached`.
///
/// Trains ONCE, scores eval + train (+ val when the cache holds a Protocol-v2
/// partition) as full K-vectors from the same trained memory, then computes
/// per-mode metrics — ALWAYS on the EVAL set — for the decode modes of
/// docs/MULTICLASS_DESIGN.md §3:
///   - `argmax`           — plain argmax over the K cluster scores (τ = NaN)
///   - `margin_fixed0`    — benign-margin decode at fixed τ = 0.0
///   - `margin_train_cal` — τ swept on TRAIN margins (macro-F1-optimal)
///   - `margin_val_cal`   — τ swept on VAL margins (only when val present)
///
/// Returns `(num_classes, mode_results)`; each result carries the full metric
/// set (K-class CE, confusion matrix, per-class P/R/F1, macro-/weighted-F1,
/// accuracy, benign-FPR). The benign class is `cache.normal_class` (0 for all
/// multiclass IDS datasets — benign = index 0 by construction).
pub fn evaluate_multiclass_at_thresholds_ids_cached(
    cache: &IDSCache,
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> (usize, Vec<crate::multiclass_metrics::MulticlassModeResult>) {
    use crate::multiclass_metrics as mm;

    let num_classes = cache.num_classes();
    assert_eq!(
        cache.num_genome_clusters(), num_classes,
        "evaluate_multiclass_at_thresholds requires a K-cluster cache \
         (single_cluster=false); got {} genome clusters for {} classes",
        cache.num_genome_clusters(), num_classes,
    );

    let train = cache.full_train();
    let eval = cache.full_eval();
    let val = cache.full_val();

    // Train ONCE + flattened K-vector scores for eval/train(/val)
    let (eval_scores, train_scores, val_scores) = crate::adaptive::train_and_score_sets_flat(
        bits_flat,
        neurons_flat,
        connections_flat,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        eval.num_examples,
        val.map(|v| &v.input_bits),
        val.map_or(0, |v| v.num_examples),
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    );

    // Decode-mode metrics from the flat scores — shared with the streaming
    // path (multiclass_metrics::modes_from_scores).
    let modes = mm::modes_from_scores(
        &eval_scores,
        &eval.targets,
        &train_scores,
        &train.targets,
        val_scores.as_deref(),
        val.map(|v| v.targets.as_slice()),
        num_classes,
        cache.normal_class,
    );

    (num_classes, modes)
}

/// Score TRAINING examples (for calibration fitting on training data).
/// Trains on full_train, returns scores for full_train (same data).
pub fn score_train_examples_ids_cached(
    cache: &IDSCache,
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<f64> {
    let train = cache.full_train();

    crate::adaptive::train_and_score_single(
        bits_flat,
        neurons_flat,
        connections_flat,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &train.input_bits,  // score training data, not eval
        train.num_examples,
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    )
}

/// Evaluate genomes with training-time adaptation (synaptogenesis + neurogenesis).
///
/// Returns adapted genome parameters alongside scores, enabling the Baldwin effect:
/// genomes are structurally modified during evaluation, and the adapted architecture
/// is returned for use in subsequent GA/TS generations.
pub fn evaluate_genomes_ids_cached_hybrid_adaptive(
    cache: &IDSCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    train_subset_idx: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    adapt_config: &crate::adaptation::AdaptationConfig,
    generation: usize,
) -> Vec<crate::adaptive::AdaptiveGenomeResult> {
    let train = cache.train_subset(train_subset_idx);
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genomes_parallel_hybrid_adaptive(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.num_genome_clusters(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        ram_core::neuron_memory::EvalSettings {
            empty_value,
            normal_class: cache.normal_class,
            fitness_weights: cache.fitness_weights,
            ..Default::default()
        },
        neuron_sample_rate,
        rng_seed,
        adapt_config,
        generation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ids_cache_basic() {
        // 10 examples, 8 features, 2 classes
        let total_features = 8;
        let num_classes = 2;
        let num_examples = 10;

        let mut features = vec![false; num_examples * total_features];
        let mut labels = vec![0i64; num_examples];

        // Fill with some pattern
        for i in 0..num_examples {
            labels[i] = (i % num_classes) as i64;
            for j in 0..total_features {
                features[i * total_features + j] = ((i + j) % 2) == 0;
            }
        }

        let train_packed = PackedBits::from_bool_slice(&features, total_features);
        let eval_packed = PackedBits::from_bool_slice(&features, total_features);
        let cache = IDSCache::new(
            train_packed,
            labels.clone(),
            eval_packed,
            labels.clone(),
            None,
            None,
            num_classes,
            total_features,
            2,  // num_parts
            1,  // num_negatives (binary: 1 negative per example)
            42,
            false,  // balance_classes
            false,  // single_cluster
            false,  // undersample_majority
            1.0,    // class_weight_multiplier
        );

        assert_eq!(cache.num_classes(), 2);
        assert_eq!(cache.total_features(), 8);
        assert_eq!(cache.num_train_subsets(), 2);
        assert_eq!(cache.full_train().num_examples, 10);
        assert_eq!(cache.full_eval().num_examples, 10);

        // Each subset should have ~5 examples (stratified)
        let s0 = cache.train_subset(0);
        let s1 = cache.train_subset(1);
        assert_eq!(s0.num_examples + s1.num_examples, 10);
    }

    #[test]
    fn test_ids_negatives_binary() {
        // Binary classification: each example should have 1 negative (the other class)
        let total_features = 4;
        let features = vec![true, false, true, false, false, true, false, true];
        let labels = vec![0i64, 1i64];
        let packed = PackedBits::from_bool_slice(&features, total_features);

        let subset = IDSCache::build_subset(&packed, &labels, total_features, 2, 1, 42);

        assert_eq!(subset.negatives.len(), 2);  // 2 examples * 1 negative
        assert_eq!(subset.negatives[0], 1);     // example 0 (class 0) → negative is class 1
        assert_eq!(subset.negatives[1], 0);     // example 1 (class 1) → negative is class 0
    }

    #[test]
    fn test_ids_negatives_multiclass() {
        // 3-class: each example should have 2 negatives (all other classes)
        let total_features = 4;
        let features = vec![true; 3 * total_features];
        let labels = vec![0i64, 1i64, 2i64];
        let packed = PackedBits::from_bool_slice(&features, total_features);

        let subset = IDSCache::build_subset(&packed, &labels, total_features, 3, 2, 42);

        assert_eq!(subset.negatives.len(), 6);  // 3 examples * 2 negatives

        // Example 0 (class 0): negatives should be 1 and 2
        let neg0: Vec<i64> = subset.negatives[0..2].to_vec();
        assert!(!neg0.contains(&0));
        assert!(neg0.contains(&1));
        assert!(neg0.contains(&2));
    }

    #[test]
    fn test_stratified_partitioning() {
        // Ensure stratified split preserves class distribution
        let total_features = 4;
        let num_examples = 100;
        let num_classes = 2;

        // 70 class 0, 30 class 1
        let mut labels = vec![0i64; 70];
        labels.extend(vec![1i64; 30]);
        let features = vec![false; num_examples * total_features];

        let packed = PackedBits::from_bool_slice(&features, total_features);
        let subsets = IDSCache::create_stratified_subsets(
            &packed, &labels, num_classes, 1, 3, 42,
        );

        // Each of 3 subsets should have roughly 70/3 ≈ 23 class-0 and 30/3 = 10 class-1
        for subset in &subsets {
            let class0_count = subset.targets.iter().filter(|&&t| t == 0).count();
            let class1_count = subset.targets.iter().filter(|&&t| t == 1).count();
            // Allow some variance due to rounding
            assert!(class0_count >= 20 && class0_count <= 27,
                "class0: {} not in [20,27]", class0_count);
            assert!(class1_count >= 8 && class1_count <= 12,
                "class1: {} not in [8,12]", class1_count);
        }
    }

    #[test]
    fn test_merge_subsets_except() {
        // 3 subsets with known sizes
        let total_features = 4;
        let num_classes = 2;
        let num_negatives = 1;

        let s0 = IDSCache::build_subset(
            &PackedBits::from_bool_slice(&vec![true; 3 * total_features], total_features),
            &vec![0, 1, 0],
            total_features, num_classes, num_negatives, 42,
        );
        let s1 = IDSCache::build_subset(
            &PackedBits::from_bool_slice(&vec![false; 2 * total_features], total_features),
            &vec![1, 0],
            total_features, num_classes, num_negatives, 43,
        );
        let s2 = IDSCache::build_subset(
            &PackedBits::from_bool_slice(&vec![true; 4 * total_features], total_features),
            &vec![0, 1, 1, 0],
            total_features, num_classes, num_negatives, 44,
        );

        let subsets = vec![&s0, &s1, &s2];

        // Exclude fold 0: merge s1 + s2 = 2 + 4 = 6 examples
        let merged = merge_subsets_except(&subsets, 0);
        assert_eq!(merged.num_examples, 6);
        assert_eq!(merged.input_bits.num_rows(), 6);
        assert_eq!(merged.input_bits.total_bits(), total_features);
        assert_eq!(merged.targets.len(), 6);
        assert_eq!(merged.negatives.len(), 6 * num_negatives);

        // Exclude fold 1: merge s0 + s2 = 3 + 4 = 7 examples
        let merged = merge_subsets_except(&subsets, 1);
        assert_eq!(merged.num_examples, 7);

        // Exclude fold 2: merge s0 + s1 = 3 + 2 = 5 examples
        let merged = merge_subsets_except(&subsets, 2);
        assert_eq!(merged.num_examples, 5);
    }

    #[test]
    fn test_kfold_uses_all_data() {
        // Verify that K-fold uses training data (not eval data) for both train and eval
        let total_features = 4;
        let num_classes = 2;
        let num_parts = 3;
        let num_train = 12;

        let mut features = vec![false; num_train * total_features];
        let mut labels = vec![0i64; num_train];
        for i in 0..num_train {
            labels[i] = (i % num_classes) as i64;
            for j in 0..total_features {
                features[i * total_features + j] = ((i + j) % 2) == 0;
            }
        }

        let cache = IDSCache::new(
            PackedBits::from_bool_slice(&features, total_features),
            labels.clone(),
            PackedBits::from_bool_slice(&vec![false; 4 * total_features], total_features),  // small eval set (shouldn't be used in kfold)
            vec![0i64; 4],
            None,
            None,
            num_classes,
            total_features,
            num_parts,
            1,
            42,
            false,  // balance_classes
            false,  // single_cluster
            false,  // undersample_majority
            1.0,    // class_weight_multiplier
        );

        // Each fold should have ~4 examples (12/3)
        // Merging 2 folds should give ~8 examples
        let all_subsets: Vec<&IDSSubset> = (0..num_parts)
            .map(|i| cache.train_subset(i))
            .collect();

        let total_in_subsets: usize = all_subsets.iter().map(|s| s.num_examples).sum();
        assert_eq!(total_in_subsets, num_train);

        for held_out in 0..num_parts {
            let train_merged = merge_subsets_except(&all_subsets, held_out);
            let eval_fold = &all_subsets[held_out];
            assert_eq!(
                train_merged.num_examples + eval_fold.num_examples,
                num_train,
                "fold {}: train + eval should equal total", held_out
            );
        }
    }
}
