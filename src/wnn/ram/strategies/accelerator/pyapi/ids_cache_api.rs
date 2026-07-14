//! IDSCache wrapper.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

/// Flat Python tuple form of one multiclass decode-mode result:
/// (mode_name, tau, ce, acc, macro_f1, weighted_f1, benign_fpr,
///  confusion, precision, recall, f1, support).
pub(crate) type MulticlassModeTuple = (
    String, f64, f64, f64, f64, f64, f64, Vec<u64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u64>,
);

/// Convert `MulticlassModeResult`s into the flat tuple form shared by
/// `IDSCacheWrapper.evaluate_multiclass_at_thresholds` (in-memory) and
/// `multiclass_modes_from_scores` (streaming).
pub(crate) fn multiclass_modes_to_py(
    modes: Vec<multiclass_metrics::MulticlassModeResult>,
) -> Vec<MulticlassModeTuple> {
    modes.into_iter().map(|m| (
        m.mode,
        m.tau,
        m.metrics.ce,
        m.metrics.accuracy,
        m.metrics.macro_f1,
        m.metrics.weighted_f1,
        m.metrics.benign_fpr,
        m.metrics.confusion,
        m.metrics.precision,
        m.metrics.recall,
        m.metrics.f1,
        m.metrics.support,
    )).collect()
}

// =============================================================================
// IDSCache Python Wrapper
// =============================================================================

/// Python wrapper for IDS (Intrusion Detection System) cache.
///
/// Holds pre-encoded binary features for classification tasks.
/// Parallel to TokenCacheWrapper but for tabular binary data.
#[pyclass]
pub(crate) struct IDSCacheWrapper {
    pub(crate) inner: ids_cache::IDSCache,
}

#[pymethods]
impl IDSCacheWrapper {
    /// Create a new IDS cache with stratified partitioning.
    ///
    /// NOTE: This constructor takes `Vec<bool>` which requires a Python list.
    /// For large datasets (e.g. 46M examples × 1000+ input bits), Python list
    /// materialization explodes memory to ~8 bytes per element (pointer size)
    /// totalling hundreds of gigabytes. Prefer `new_from_numpy` for those cases.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (train_features, train_labels, eval_features, eval_labels, num_classes, total_features, num_parts, num_negatives, seed, balance_classes=false, single_cluster=false, undersample_majority=false, class_weight_multiplier=1.0, val_features=None, val_labels=None))]
    fn new(
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
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
        val_features: Option<Vec<bool>>,
        val_labels: Option<Vec<i64>>,
    ) -> Self {
        // Pack Python-list bools into PackedBits. This constructor is the
        // small-data backwards-compat path (tests, demos); large datasets
        // should use new_from_numpy with np.packbits.
        let train_packed = ram_core::packed_bits::PackedBits::from_bool_slice(&train_features, total_features);
        let eval_packed = ram_core::packed_bits::PackedBits::from_bool_slice(&eval_features, total_features);
        // Protocol v2: optional val partition, packed like the eval set.
        let val_packed = val_features.map(|vf| {
            ram_core::packed_bits::PackedBits::from_bool_slice(&vf, total_features)
        });
        Self {
            inner: ids_cache::IDSCache::new(
                train_packed,
                train_labels,
                eval_packed,
                eval_labels,
                val_packed,
                val_labels,
                num_classes,
                total_features,
                num_parts,
                num_negatives,
                seed,
                balance_classes,
                single_cluster,
                undersample_majority,
                class_weight_multiplier,
            ),
        }
    }

    /// Create a new IDS cache from bit-packed numpy arrays (Phase 2 packed boundary).
    ///
    /// `train_features` / `eval_features` are uint8 arrays produced by
    /// `np.packbits(bool_matrix, axis=1, bitorder='little')`. Each row of the
    /// logical bool matrix is stored as `ceil(total_features/8)` bytes,
    /// LSB-first within each byte.
    ///
    /// For a 46M × 96-bit CIC-IoT-2023 training set this is ~552 MB (was ~4.4 GB
    /// as Vec<bool>, or ~35 GB as a Python list). No bool materialization
    /// happens on the Rust side.
    ///
    /// Args:
    ///   train_features: numpy uint8 array, packed bytes (np.packbits output).
    ///                   Shape: (num_train * bytes_per_row,) where
    ///                   bytes_per_row = ceil(total_features / 8).
    ///   eval_features:  numpy uint8 array, same packed layout.
    ///   total_features: logical bit-width per row (used for stride math).
    ///   All other args match `new()`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (train_features, train_labels, eval_features, eval_labels, num_classes, total_features, num_parts, num_negatives, seed, balance_classes=false, single_cluster=false, undersample_majority=false, class_weight_multiplier=1.0, val_features=None, val_labels=None))]
    fn new_from_numpy<'py>(
        train_features: PyReadonlyArray1<'py, u8>,
        train_labels: Vec<i64>,
        eval_features: PyReadonlyArray1<'py, u8>,
        eval_labels: Vec<i64>,
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
        val_features: Option<PyReadonlyArray1<'py, u8>>,
        val_labels: Option<Vec<i64>>,
    ) -> PyResult<Self> {
        // Zero-copy views into the numpy packed-byte arrays.
        let train_slice = train_features.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("train_features array not contiguous: {}", e),
            )
        })?;
        let eval_slice = eval_features.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("eval_features array not contiguous: {}", e),
            )
        })?;

        // Interpret incoming uint8 as packed bytes (np.packbits output).
        let train_packed = ram_core::packed_bits::PackedBits::from_packed_bytes(
            train_slice.to_vec(), total_features,
        );
        let eval_packed = ram_core::packed_bits::PackedBits::from_packed_bytes(
            eval_slice.to_vec(), total_features,
        );

        // Protocol v2: optional val partition, same packed-byte layout as eval.
        let val_packed = match &val_features {
            Some(vf) => {
                let val_slice = vf.as_slice().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("val_features array not contiguous: {}", e),
                    )
                })?;
                Some(ram_core::packed_bits::PackedBits::from_packed_bytes(
                    val_slice.to_vec(), total_features,
                ))
            }
            None => None,
        };

        Ok(Self {
            inner: ids_cache::IDSCache::new(
                train_packed,
                train_labels,
                eval_packed,
                eval_labels,
                val_packed,
                val_labels,
                num_classes,
                total_features,
                num_parts,
                num_negatives,
                seed,
                balance_classes,
                single_cluster,
                undersample_majority,
                class_weight_multiplier,
            ),
        })
    }

    /// Get the next train subset index (advances rotator).
    fn next_train_idx(&mut self) -> usize {
        self.inner.next_train_idx()
    }

    /// Reset rotators with optional new seed.
    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) {
        self.inner.reset(seed);
    }

    /// Get number of train subsets.
    fn num_train_subsets(&self) -> usize {
        self.inner.num_train_subsets()
    }

    /// Get number of classes.
    fn num_classes(&self) -> usize {
        self.inner.num_classes()
    }

    /// Get total features (input bits).
    fn total_features(&self) -> usize {
        self.inner.total_features()
    }

    /// Set which class is "normal" for FPR computation.
    /// Call with 1 when flip_labels is active so FPR measures false alarms
    /// on the original benign class (which is class 1 after flipping).
    fn set_normal_class(&mut self, normal_class: usize) {
        self.inner.normal_class = normal_class;
    }

    /// Set fitness weights for threshold optimization.
    /// When set, threshold sweep maximizes fitness instead of F1.
    fn set_fitness_weights(&mut self, w_ce: f32, w_f1: f32, w_fpr: f32, w_acc: f32) {
        self.inner.fitness_weights = Some((w_ce, w_f1, w_fpr, w_acc));
    }

    /// Set the memory-cell semantics (TERNARY/QUAD_BINARY/QUAD_WEIGHTED/BINARY/
    /// QSR/PLN) for BOTH train and score, plus the QSR/PLN per-run coin seed.
    /// Must be called after construction — before this the IDS cache defaulted
    /// to QUAD_WEIGHTED regardless of the flow's memory_mode param. run_seed is
    /// ignored by the deterministic modes.
    fn set_memory_mode(&mut self, memory_mode: u8, run_seed: u64) {
        self.inner.memory_mode = memory_mode;
        self.inner.run_seed = run_seed;
    }

    /// Evaluate genomes using hybrid CPU+GPU with a specific train subset.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64, u32)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_genomes_ids_cached_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                train_subset_idx,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Evaluate genomes using full data with hybrid CPU+GPU.
    #[pyo3(signature = (genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat, num_genomes, empty_value, neuron_sample_rate, rng_seed, override_threshold=None))]
    fn evaluate_genomes_full_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        override_threshold: Option<f64>,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64, u32)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_genomes_ids_cached_full_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                empty_value,
                neuron_sample_rate,
                rng_seed,
                override_threshold,
            ))
        })
    }

    /// Evaluate genomes using K-fold cross-validation with hybrid CPU+GPU.
    ///
    /// Merges all subsets except `held_out_fold` for training, uses the
    /// held-out fold as the eval set. Rotates the held-out fold each
    /// generation for more robust F1/FPR estimates.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_genomes_kfold_hybrid(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        held_out_fold: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64, u32)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_genomes_ids_kfold_hybrid(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                held_out_fold,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Evaluate genomes with training-time adaptation (*genesis).
    ///
    /// Returns (ce, acc, f1, adapted_bits, adapted_neurons, adapted_conns,
    ///          pruned, grown, added, removed, rewired) per genome.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
        num_genomes, train_subset_idx, empty_value, neuron_sample_rate, rng_seed,
        synaptogenesis_enabled, neurogenesis_enabled, axonogenesis_enabled = false,
        min_bits = 4, max_bits = 24,
        warmup_generations = 10, total_generations = 250, generation = 0,
        total_input_bits = 336, stats_sample_size = 10000, passes_per_eval = 1,
        prune_entropy_ratio = 0.3, grow_fill_utilization = 0.5, grow_error_baseline = 0.35,
        min_neurons = 3, max_neurons_per_pass = 3, max_growth_ratio = 1.5,
        cooldown_iterations = 5, stabilize_fraction = 0.25,
        axon_entropy_threshold = 0.3, axon_improvement_factor = 1.2, axon_rewire_count = 2
    ))]
    fn evaluate_genomes_hybrid_adaptive(
        &self,
        py: Python<'_>,
        genomes_bits_flat: Vec<usize>,
        genomes_neurons_flat: Vec<usize>,
        genomes_connections_flat: Vec<i64>,
        num_genomes: usize,
        train_subset_idx: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        synaptogenesis_enabled: bool,
        neurogenesis_enabled: bool,
        axonogenesis_enabled: bool,
        min_bits: usize,
        max_bits: usize,
        warmup_generations: usize,
        total_generations: usize,
        generation: usize,
        total_input_bits: usize,
        stats_sample_size: usize,
        passes_per_eval: usize,
        prune_entropy_ratio: f32,
        grow_fill_utilization: f32,
        grow_error_baseline: f32,
        min_neurons: usize,
        max_neurons_per_pass: usize,
        max_growth_ratio: f32,
        cooldown_iterations: usize,
        stabilize_fraction: f32,
        axon_entropy_threshold: f32,
        axon_improvement_factor: f32,
        axon_rewire_count: usize,
    ) -> PyResult<Vec<(f64, f64, f64, f64, Vec<usize>, Vec<usize>, Vec<i64>, usize, usize, usize, usize, usize)>> {
        validate_flat_genomes_py(&genomes_bits_flat, &genomes_neurons_flat, &genomes_connections_flat, num_genomes, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            let adapt_config = adaptation::AdaptationConfig {
                synaptogenesis_enabled,
                neurogenesis_enabled,
                axonogenesis_enabled,
                min_bits,
                max_bits,
                warmup_generations,
                total_generations,
                total_input_bits,
                stats_sample_size,
                passes_per_eval,
                neuron_sample_rate,
                prune_entropy_ratio,
                grow_fill_utilization,
                grow_error_baseline,
                min_neurons,
                max_neurons_per_pass,
                max_growth_ratio,
                cooldown_iterations,
                stabilize_fraction,
                axon_entropy_threshold,
                axon_improvement_factor,
                axon_rewire_count,
                ..Default::default()
            };
            let results = ids_cache::evaluate_genomes_ids_cached_hybrid_adaptive(
                &self.inner,
                &genomes_bits_flat,
                &genomes_neurons_flat,
                &genomes_connections_flat,
                num_genomes,
                train_subset_idx,
                empty_value,
                neuron_sample_rate,
                rng_seed,
                &adapt_config,
                generation,
            );
            Ok(results.into_iter().map(|r| (
                r.ce, r.accuracy, r.f1_macro, r.fpr,
                r.adapted_bits, r.adapted_neurons, r.adapted_connections,
                r.pruned, r.grown, r.added, r.removed, r.rewired,
            )).collect())
        })
    }

    /// Train a single genome on full training data and return per-example predictions.
    ///
    /// Returns Vec<i64> of predicted class indices for each eval example.
    /// Used by the bitwise ECOC classifier for per-bit predictions.
    fn predict_examples(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<i64>> {
        py.allow_threads(|| {
            Ok(ids_cache::predict_examples_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Train a single genome and return per-example RAW SCORES (not thresholded).
    ///
    /// Returns Vec<f64> of raw scores for each eval example.
    /// Used for Platt scaling calibration.
    fn score_examples(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            Ok(ids_cache::score_examples_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Score TRAINING examples for calibration fitting.
    /// Trains on full train, returns scores for train (not eval).
    fn score_train_examples(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            Ok(ids_cache::score_train_examples_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Train a single genome ONCE and evaluate at multiple thresholds.
    ///
    /// Returns a tuple (eval_scores, train_scores, val_scores, metrics) where
    /// `metrics[i] = (ce, acc, f1, fpr, resolved_threshold)` for thresholds[i].
    /// `val_scores` is a list when the cache was built with a val partition
    /// (Protocol v2, `_3way` splits), None otherwise. A threshold of -1.0 is
    /// the oracle sentinel — sweeps EVAL scores for the optimal F1 threshold
    /// and uses that (unchanged; Python decides what to do with val_scores).
    ///
    /// Replaces the validation-phase pattern of 7 evaluate_batch_full + 1
    /// score_examples + 1 score_train_examples (9 trainings) with a single
    /// training pass.
    fn evaluate_at_thresholds(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        thresholds: Vec<f64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Option<Vec<f64>>, Vec<(f64, f64, f64, f64, f64)>)> {
        py.allow_threads(|| {
            Ok(ids_cache::evaluate_at_thresholds_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                &thresholds,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            ))
        })
    }

    /// Multiclass (K-cluster) analogue of `evaluate_at_thresholds`.
    ///
    /// Trains a single genome ONCE, scores eval + train (+ val when the cache
    /// holds a Protocol-v2 partition), and returns `(num_classes, modes)`
    /// where each mode is a flat tuple:
    ///   (mode_name, tau, ce, acc, macro_f1, weighted_f1, benign_fpr,
    ///    confusion, precision, recall, f1, support)
    /// with `confusion` a K*K row-major Vec<u64> (row = true class) and the
    /// per-class vectors of length K. Modes: "argmax" (tau = NaN),
    /// "margin_fixed0" (tau = 0.0), "margin_train_cal" (tau swept on train
    /// margins), and "margin_val_cal" (tau swept on val margins) when the
    /// cache carries a val partition. Metrics are ALWAYS computed on the
    /// EVAL set. See docs/MULTICLASS_DESIGN.md §3-4.
    fn evaluate_multiclass_at_thresholds(
        &self,
        py: Python<'_>,
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections_flat: Vec<i64>,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
    ) -> PyResult<(usize, Vec<MulticlassModeTuple>)> {
        if self.inner.num_genome_clusters() != self.inner.num_classes()
            || self.inner.num_classes() < 2
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "evaluate_multiclass_at_thresholds requires a K-cluster cache \
                 (single_cluster=false, K>=2); got {} genome clusters / {} classes",
                self.inner.num_genome_clusters(),
                self.inner.num_classes(),
            )));
        }
        validate_flat_genomes_py(&bits_flat, &neurons_flat, &connections_flat, 1, self.inner.num_genome_clusters())?;
        py.allow_threads(|| {
            let (num_classes, modes) = ids_cache::evaluate_multiclass_at_thresholds_ids_cached(
                &self.inner,
                &bits_flat,
                &neurons_flat,
                &connections_flat,
                empty_value,
                neuron_sample_rate,
                rng_seed,
            );
            Ok((num_classes, multiclass_modes_to_py(modes)))
        })
    }

    /// Search for neighbors above accuracy threshold, all in Rust.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        base_bits,
        base_neurons,
        base_connections,
        target_count,
        max_attempts,
        accuracy_threshold,
        min_bits,
        max_bits,
        min_neurons,
        max_neurons,
        bits_mutation_rate,
        neurons_mutation_rate,
        train_subset_idx,
        empty_value,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0
    ))]
    fn search_neighbors(
        &self,
        py: Python<'_>,
        base_bits: Vec<usize>,
        base_neurons: Vec<usize>,
        base_connections: Vec<i64>,
        target_count: usize,
        max_attempts: usize,
        accuracy_threshold: f64,
        min_bits: usize,
        max_bits: usize,
        min_neurons: usize,
        max_neurons: usize,
        bits_mutation_rate: f64,
        neurons_mutation_rate: f64,
        train_subset_idx: usize,
        empty_value: f32,
        seed: u64,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
        phase_type: u8,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>> {
        let num_clusters = base_neurons.len();
        let total_input_bits = self.inner.total_features();
        let phase = match phase_type {
            1 => neighbor_search::PhaseType::Bits,
            2 => neighbor_search::PhaseType::Connections,
            _ => neighbor_search::PhaseType::Neurons,
        };

        let config = neighbor_search::MutationConfig {
            num_clusters,
            mutable_clusters,
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            total_input_bits,
            phase_type: phase,
        };

        let lp_arc = self.inner.live_progress.clone();

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                ids_cache::evaluate_genomes_ids_cached_hybrid(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, empty_value,
                    1.0, 0,
                ).into_iter().map(|(ce, acc, f1, fpr, _, _)| (ce, acc, f1, fpr)).collect()
            };

            let lp_ref = Some(&lp_arc);

            let candidates = if return_best_n {
                neighbor_search::search_neighbors_best_n(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                )
            } else {
                let (passed, _) = neighbor_search::search_neighbors_with_threshold(
                    &base_bits, &base_neurons, &base_connections,
                    target_count, max_attempts, accuracy_threshold,
                    &config, &eval_fn, seed, log_path_ref,
                    generation, total_generations, lp_ref,
                );
                passed
            };

            Ok(candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect())
        });

        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
    }

    /// Search for GA offspring above accuracy threshold, all in Rust.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        population,
        target_count,
        max_attempts,
        accuracy_threshold,
        min_bits,
        max_bits,
        min_neurons,
        max_neurons,
        bits_mutation_rate,
        neurons_mutation_rate,
        crossover_rate,
        tournament_size,
        train_subset_idx,
        empty_value,
        neuron_sample_rate,
        seed,
        fitness_scores = None,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0,
        cluster_crossover_ratio = 0.0,
        pool_shuffle_ratio = 0.0,
        assortative_mating_ratio = 0.0
    ))]
    fn search_offspring(
        &self,
        py: Python<'_>,
        population: Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64)>,
        target_count: usize,
        max_attempts: usize,
        accuracy_threshold: f64,
        min_bits: usize,
        max_bits: usize,
        min_neurons: usize,
        max_neurons: usize,
        bits_mutation_rate: f64,
        neurons_mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
        train_subset_idx: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        seed: u64,
        // Per-genome harmonic-rank fitness from Python's fitness_calculator. When
        // provided, this REPLACES the population tuple's 4th element (which
        // Python builds as raw CE — see architecture_strategies.py:958) so
        // Rust's tournament_select picks parents by the actual fitness the user
        // configured (weights ce/acc/f1/fpr), not by CE alone. Before this fix,
        // every IDS GA run since 18662b5f (2026-03-09) used CE for parent
        // selection regardless of fitness_calculator weights — weights only
        // affected elite preservation + reporting, not GA exploration direction.
        fitness_scores: Option<Vec<f64>>,
        log_path: Option<String>,
        generation: Option<usize>,
        total_generations: Option<usize>,
        return_best_n: bool,
        mutable_clusters: Option<Vec<usize>>,
        phase_type: u8,
        cluster_crossover_ratio: f64,
        pool_shuffle_ratio: f64,
        assortative_mating_ratio: f64,
    ) -> PyResult<(Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>, usize, usize)> {
        let num_clusters = if !population.is_empty() {
            population[0].1.len()
        } else {
            return Ok((Vec::new(), 0, 0));
        };
        let total_input_bits = self.inner.total_features();
        let phase = match phase_type {
            1 => neighbor_search::PhaseType::Bits,
            2 => neighbor_search::PhaseType::Connections,
            _ => neighbor_search::PhaseType::Neurons,
        };

        let ga_config = neighbor_search::GAConfig {
            num_clusters,
            mutable_clusters,
            min_bits,
            max_bits,
            min_neurons,
            max_neurons,
            bits_mutation_rate,
            neurons_mutation_rate,
            crossover_rate,
            tournament_size,
            total_input_bits,
            phase_type: phase,
            cluster_crossover_ratio,
            pool_shuffle_ratio,
            assortative_mating_ratio,
        };

        let lp_arc = self.inner.live_progress.clone();

        let result = py.allow_threads(|| {
            let log_path_ref = log_path.as_deref();
            let cache = &self.inner;

            // Use the SAME evaluation path as elites (kfold_hybrid on the
            // held-out fold of train) AND the SAME neuron_sample_rate the
            // cache was built with — otherwise offspring metrics are computed
            // on a different dataset partition (the 20% held-out full_eval, a
            // methodology violation per CLAUDE.md) at a different sample_rate
            // (1.0 vs 0.25 production), making fitness comparison apples-to-
            // oranges. The bug was silent for CIC-IoT random splits
            // (train ≈ test distribution) but surfaced on UNSW-temporal as
            // "offspring collapse" (offspring CE 0.26 vs elite CE 0.13 for
            // the SAME genome). See the GA debug agent's investigation report
            // in the session JSONL for the smoking-gun analysis.
            let eval_fn = |bits: &[usize], neurons: &[usize], conns: &[i64], count: usize| -> Vec<(f64, f64, f64, f64)> {
                ids_cache::evaluate_genomes_ids_kfold_hybrid(
                    cache, bits, neurons, conns, count,
                    train_subset_idx, empty_value,
                    neuron_sample_rate, 0,
                ).into_iter().map(|(ce, acc, f1, fpr, _, _)| (ce, acc, f1, fpr)).collect()
            };

            let lp_ref = Some(&lp_arc);

            // Replace the population tuple's 4th element with harmonic-rank
            // fitness when provided (matches the fitness used for elite
            // selection in Python; without this, tournament_select uses raw
            // CE which makes weight variations meaningless for parent
            // selection — see fitness_scores doc on the param above).
            let population_for_search = if let Some(scores) = fitness_scores {
                if scores.len() != population.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "fitness_scores length {} != population length {}",
                        scores.len(),
                        population.len()
                    )));
                }
                population
                    .into_iter()
                    .zip(scores.into_iter())
                    .map(|((b, n, c, _ce), s)| (b, n, c, s))
                    .collect()
            } else {
                population
            };

            let result = neighbor_search::search_offspring(
                &population_for_search, target_count, max_attempts, accuracy_threshold,
                &ga_config, &eval_fn, seed, log_path_ref,
                generation, total_generations, return_best_n, lp_ref,
            );

            let candidates: Vec<_> = result.candidates
                .into_iter()
                .map(|c| (
                    c.bits_per_neuron,
                    c.neurons_per_cluster,
                    c.connections,
                    c.cross_entropy,
                    c.accuracy,
                    c.f1_macro,
                    c.fpr,
                ))
                .collect();
            Ok((candidates, result.evaluated, result.viable))
        });

        if let Ok(mut guard) = lp_arc.write() {
            *guard = None;
        }
        result
    }
}
