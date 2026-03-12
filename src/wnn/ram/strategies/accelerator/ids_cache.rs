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
use crate::token_cache::SubsetRotator;

/// Pre-computed IDS subset with all data needed for evaluation.
#[derive(Clone)]
pub struct IDSSubset {
    /// Encoded input bits: [num_examples * total_features]
    pub input_bits: Vec<bool>,
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

    // Rotator for train subsets
    train_rotator: SubsetRotator,

    // Class weights for balanced training (None = unweighted)
    class_weights: Option<Vec<u32>>,

    // Live progress for observer thread
    pub live_progress: Arc<RwLock<Option<LiveProgress>>>,
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
    pub fn new(
        train_features: Vec<bool>,
        train_labels: Vec<i64>,
        eval_features: Vec<bool>,
        eval_labels: Vec<i64>,
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
    ) -> Self {
        let num_train = train_labels.len();
        let num_eval = eval_labels.len();

        assert_eq!(train_features.len(), num_train * total_features,
            "train_features length mismatch: {} != {} * {}",
            train_features.len(), num_train, total_features);
        assert_eq!(eval_features.len(), num_eval * total_features,
            "eval_features length mismatch: {} != {} * {}",
            eval_features.len(), num_eval, total_features);

        // Build full train subset
        let full_train = Self::build_subset(
            &train_features, &train_labels, total_features, num_classes, num_negatives, seed,
        );

        // Build full eval subset
        let full_eval = Self::build_subset(
            &eval_features, &eval_labels, total_features, num_classes, num_negatives, seed + 1,
        );

        // Stratified partitioning of training data
        let train_subsets = Self::create_stratified_subsets(
            &train_features, &train_labels, total_features, num_classes,
            num_negatives, num_parts, seed,
        );

        // Compute class weights: max_count / count per class (upweights minority)
        let class_weights = if balance_classes {
            Some(crate::adaptive::compute_class_weights(&train_labels, num_classes))
        } else {
            None
        };

        Self {
            num_classes,
            total_features,
            num_negatives,
            num_parts,
            train_subsets,
            full_train,
            full_eval,
            class_weights,
            train_rotator: SubsetRotator::new(num_parts, seed + 100),
            live_progress: Arc::new(RwLock::new(None)),
        }
    }

    /// Build an IDSSubset from features and labels.
    ///
    /// Negatives: for each example, all classes except the target.
    /// If num_negatives < num_classes - 1, randomly sample from non-target classes.
    fn build_subset(
        features: &[bool],
        labels: &[i64],
        _total_features: usize,
        num_classes: usize,
        num_negatives: usize,
        seed: u64,
    ) -> IDSSubset {
        let num_examples = labels.len();
        if num_examples == 0 {
            return IDSSubset {
                input_bits: vec![],
                targets: vec![],
                negatives: vec![],
                num_examples: 0,
            };
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Input bits are already encoded — just clone
        let input_bits = features.to_vec();
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
        features: &[bool],
        labels: &[i64],
        total_features: usize,
        num_classes: usize,
        num_negatives: usize,
        num_parts: usize,
        seed: u64,
    ) -> Vec<IDSSubset> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 50);

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
            let n = indices.len();
            let mut sub_features = vec![false; n * total_features];
            let mut sub_labels = vec![0i64; n];

            for (new_idx, &orig_idx) in indices.iter().enumerate() {
                // Copy features
                let src_start = orig_idx * total_features;
                let dst_start = new_idx * total_features;
                sub_features[dst_start..dst_start + total_features]
                    .copy_from_slice(&features[src_start..src_start + total_features]);
                // Copy label
                sub_labels[new_idx] = labels[orig_idx];
            }

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

    pub fn total_features(&self) -> usize {
        self.total_features
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn num_negatives(&self) -> usize {
        self.num_negatives
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
) -> Vec<(f64, f64, f64, f64)> {
    let train = cache.train_subset(train_subset_idx);
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genomes_parallel_hybrid(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.num_classes(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        empty_value,
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
    )
}

/// Merge all subsets except the excluded one into a single IDSSubset.
///
/// Used for K-fold cross-validation: K-1 folds become the training set,
/// the excluded fold becomes the eval set.
fn merge_subsets_except(subsets: &[IDSSubset], exclude_idx: usize) -> IDSSubset {
    let total_features = if subsets.is_empty() || subsets[0].num_examples == 0 {
        0
    } else {
        subsets[0].input_bits.len() / subsets[0].num_examples
    };

    let mut merged_input_bits = Vec::new();
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
        merged_input_bits.extend_from_slice(&subset.input_bits);
        merged_targets.extend_from_slice(&subset.targets);
        merged_negatives.extend_from_slice(&subset.negatives);
        merged_num_examples += subset.num_examples;
    }

    // Sanity checks
    if merged_num_examples > 0 {
        assert_eq!(
            merged_input_bits.len(),
            merged_num_examples * total_features,
            "merge_subsets_except: input_bits length mismatch"
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
) -> Vec<(f64, f64, f64, f64)> {
    assert!(
        held_out_fold < cache.num_parts(),
        "held_out_fold {} >= num_parts {}",
        held_out_fold,
        cache.num_parts(),
    );

    // Build train set from all folds except the held-out one
    let all_subsets: Vec<&IDSSubset> = (0..cache.num_parts())
        .map(|i| cache.train_subset(i))
        .collect();
    let owned_subsets: Vec<IDSSubset> = all_subsets.iter().map(|s| (*s).clone()).collect();
    let train = merge_subsets_except(&owned_subsets, held_out_fold);

    // Use the held-out fold as eval
    let eval = cache.train_subset(held_out_fold);

    crate::adaptive::evaluate_genomes_parallel_hybrid(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.num_classes(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        empty_value,
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
) -> Vec<(f64, f64, f64, f64)> {
    let train = cache.full_train();
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genomes_parallel_hybrid(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.num_classes(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        empty_value,
        neuron_sample_rate,
        rng_seed,
        cache.class_weights.as_deref(),
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
        cache.num_classes(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        eval.num_examples,
        cache.total_features(),
        empty_value,
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
        cache.num_classes(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_features(),
        empty_value,
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

        let cache = IDSCache::new(
            features.clone(),
            labels.clone(),
            features.clone(),
            labels.clone(),
            num_classes,
            total_features,
            2,  // num_parts
            1,  // num_negatives (binary: 1 negative per example)
            42,
            false,  // balance_classes
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

        let subset = IDSCache::build_subset(&features, &labels, total_features, 2, 1, 42);

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

        let subset = IDSCache::build_subset(&features, &labels, total_features, 3, 2, 42);

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

        let subsets = IDSCache::create_stratified_subsets(
            &features, &labels, total_features, num_classes, 1, 3, 42,
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
            &vec![true; 3 * total_features], &vec![0, 1, 0],
            total_features, num_classes, num_negatives, 42,
        );
        let s1 = IDSCache::build_subset(
            &vec![false; 2 * total_features], &vec![1, 0],
            total_features, num_classes, num_negatives, 43,
        );
        let s2 = IDSCache::build_subset(
            &vec![true; 4 * total_features], &vec![0, 1, 1, 0],
            total_features, num_classes, num_negatives, 44,
        );

        let subsets = vec![s0.clone(), s1.clone(), s2.clone()];

        // Exclude fold 0: merge s1 + s2 = 2 + 4 = 6 examples
        let merged = merge_subsets_except(&subsets, 0);
        assert_eq!(merged.num_examples, 6);
        assert_eq!(merged.input_bits.len(), 6 * total_features);
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
            features.clone(),
            labels.clone(),
            vec![false; 4 * total_features],  // small eval set (shouldn't be used in kfold)
            vec![0i64; 4],
            num_classes,
            total_features,
            num_parts,
            1,
            42,
            false,  // balance_classes
        );

        // Each fold should have ~4 examples (12/3)
        // Merging 2 folds should give ~8 examples
        let all_subsets: Vec<IDSSubset> = (0..num_parts)
            .map(|i| cache.train_subset(i).clone())
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
