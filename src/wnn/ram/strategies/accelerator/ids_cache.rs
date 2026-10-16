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
use ram_core::packed_bits::PackedBits;

/// Pre-computed IDS subset with all data needed for evaluation.
///
/// Storage is bit-packed (8 bits per byte) — 8x less memory than the legacy
/// Vec<bool> layout. Consumers read via `input_bits.bit(i, j)` or migrate
/// to packed-byte access via `input_bits.packed_row(i)`.
#[derive(Clone)]
pub struct IDSSubset
{
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
pub struct IDSCache
{
	// Configuration
	num_classes: usize,
	total_features: usize,
	num_negatives: usize,
	num_parts: usize,

	// Stratified train folds as ROW INDICES into full_train (13/10/2026):
	// the packed rows exist once, in full_train; a fold's rows are gathered on
	// demand (byte-identical to the old per-fold copy, which held the whole
	// train set a second time — 8.6 GB at 46M rows).
	train_folds: Vec<crate::ids_fold::IDSFold>,

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
	pub fitness_weights: Option<(f32, f32, f32, f32)>, // (w_ce, w_f1, w_fpr, w_acc)

	// Memory-cell semantics for train+score, threaded from the flow's
	// memory_mode param via set_memory_mode(); default QUAD_WEIGHTED(2). Before
	// this (14/07/2026) IDS eval hardcoded EvalSettings::default()→QUAD, so
	// TERNARY/BINARY/QSR/PLN flows silently trained+scored as QUAD (the param
	// only flipped empty_value, which the QUAD branch ignores).
	pub memory_mode: u8,

	// QSR/PLN per-run seed for the stochastic coin (ignored by other modes).
	pub run_seed: u64,
	/// Coverage-aware scoring: a sparse miss scores 0.0 ("no evidence") instead
	/// of the mode's empty cell. See docs/COVERAGE_AWARE_SCORER_SPEC.md.
	/// Default false = bit-exact today's behaviour.
	pub coverage_aware: bool,
	/// Per-flow MAJORITY-VOTE write rule (`ids_write_rule="vote"`, via
	/// set_write_rule): BINARY/TERNARY/PLN train through `WriteRule::Vote`.
	/// Default false = bit-exact today's Set / TRUE-wins lattice.
	pub write_vote: bool,

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
pub struct PartialFitState
{
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

impl PartialFitState
{
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
	) -> Self
	{
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
	pub fn add_train_chunk(&mut self, chunk: &PackedBits, labels: &[i64])
	{
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
	pub fn add_eval_chunk(&mut self, chunk: &PackedBits, labels: &[i64])
	{
		assert_eq!(chunk.num_rows(), labels.len());
		assert_eq!(chunk.total_bits(), self.total_features);
		self.eval_features.extend_from(chunk);
		self.eval_labels.extend_from_slice(labels);
	}

	/// Number of train rows accumulated so far.
	pub fn num_train(&self) -> usize
	{
		self.train_labels.len()
	}

	/// Number of eval rows accumulated so far.
	pub fn num_eval(&self) -> usize
	{
		self.eval_labels.len()
	}

	/// Logical bit width per row (frozen at construction).
	pub fn total_features(&self) -> usize
	{
		self.total_features
	}

	/// Consume the accumulator and build the IDSCache via the standard
	/// `IDSCache::new` path. Equivalent to the one-shot constructor when
	/// all data was added in a single chunk.
	pub fn finalize(self) -> IDSCache
	{
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

impl IDSCache
{
	/// The per-call scoring settings every IDS path trains/scores with (one
	/// definition; was copied at nine call sites).
	pub fn eval_settings(&self, empty_value: f32) -> ram_core::neuron_memory::EvalSettings
	{
		ram_core::neuron_memory::EvalSettings {
			empty_value,
			normal_class: self.normal_class,
			fitness_weights: self.fitness_weights,
			memory_mode: self.memory_mode,
			run_seed: self.run_seed,
			coverage_aware: self.coverage_aware,
			write_vote: self.write_vote,
		}
	}

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
	) -> Self
	{
		// Optionally undersample majority class to match minority count
		let (train_features, train_labels) = if undersample_majority
		{
			Self::undersample_to_balance(train_features, train_labels, num_classes, seed)
		}
		else
		{
			(train_features, train_labels)
		};

		let num_train = train_labels.len();
		let num_eval = eval_labels.len();

		assert_eq!(
			train_features.num_rows(),
			num_train,
			"train_features rows mismatch: {} != {}",
			train_features.num_rows(),
			num_train
		);
		assert_eq!(
			train_features.total_bits(),
			total_features,
			"train_features total_bits mismatch: {} != {}",
			train_features.total_bits(),
			total_features
		);
		assert_eq!(
			eval_features.num_rows(),
			num_eval,
			"eval_features rows mismatch: {} != {}",
			eval_features.num_rows(),
			num_eval
		);
		assert_eq!(
			eval_features.total_bits(),
			total_features,
			"eval_features total_bits mismatch: {} != {}",
			eval_features.total_bits(),
			total_features
		);

		// Single-cluster mode: 1 genome cluster, 0 negatives
		let num_genome_clusters = if single_cluster { 1 } else { num_classes };
		let actual_negatives = if single_cluster { 0 } else { num_negatives };

		// Stratified partitioning of training data (labels only — the folds
		// index full_train's rows instead of copying them).
		let train_folds = Self::create_stratified_folds(&train_labels, num_classes, actual_negatives, num_parts, seed);

		// Build full train subset (takes the matrix — no clone)
		let full_train = Self::build_subset(
			train_features,
			&train_labels,
			total_features,
			num_classes,
			actual_negatives,
			seed,
		);

		// Build full eval subset
		let full_eval = Self::build_subset(
			eval_features,
			&eval_labels,
			total_features,
			num_classes,
			actual_negatives,
			seed + 1,
		);

		// Build full val subset (Protocol v2) — same construction as full_eval
		let full_val = match (val_features, val_labels)
		{
			(Some(vf), Some(vl)) =>
			{
				assert_eq!(
					vf.num_rows(),
					vl.len(),
					"val_features rows mismatch: {} != {}",
					vf.num_rows(),
					vl.len()
				);
				assert_eq!(
					vf.total_bits(),
					total_features,
					"val_features total_bits mismatch: {} != {}",
					vf.total_bits(),
					total_features
				);
				Some(Self::build_subset(
					vf,
					&vl,
					total_features,
					num_classes,
					actual_negatives,
					seed + 2,
				))
			}
			(None, None) => None,
			_ => panic!("val_features and val_labels must both be Some or both be None"),
		};

		// Compute class weights: max_count / count per class * multiplier (upweights minority)
		let class_weights = if balance_classes
		{
			Some(crate::adaptive::compute_class_weights_with_multiplier(
				&train_labels,
				num_classes,
				class_weight_multiplier,
			))
		}
		else
		{
			None
		};

		Self {
			num_classes,
			total_features,
			num_negatives: actual_negatives,
			num_parts,
			train_folds,
			full_train,
			full_eval,
			full_val,
			class_weights,
			num_genome_clusters,
			train_rotator: SubsetRotator::new(num_parts, seed + 100),
			live_progress: Arc::new(RwLock::new(None)),
			normal_class: 0,
			fitness_weights: None,
			memory_mode: ram_core::neuron_memory::QUAD_WEIGHTED,
			run_seed: 0,
			coverage_aware: false,
			write_vote: false,
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
	) -> (PackedBits, Vec<i64>)
	{
		use rand::seq::SliceRandom;
		use rand::SeedableRng;

		let num_examples = labels.len();
		if num_examples == 0
		{
			return (features, labels);
		}

		// Group example indices by class
		let mut class_indices: Vec<Vec<usize>> = vec![vec![]; num_classes];
		for (i, &label) in labels.iter().enumerate()
		{
			let c = label as usize;
			if c < num_classes
			{
				class_indices[c].push(i);
			}
		}

		// Find minimum class count
		let min_count = class_indices
			.iter()
			.filter(|v| !v.is_empty())
			.map(|v| v.len())
			.min()
			.unwrap_or(0);

		if min_count == 0
		{
			return (features, labels);
		}

		// Shuffle and take min_count from each class
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 999);
		let mut selected_indices: Vec<usize> = Vec::with_capacity(min_count * num_classes);
		for indices in &mut class_indices
		{
			if !indices.is_empty()
			{
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
		for &idx in &selected_indices
		{
			new_labels.push(labels[idx]);
		}

		eprintln!(
			"[IDS_CACHE] Undersampled majority: {} → {} examples ({} per class, {} classes)",
			original_count, balanced_count, min_count, num_classes
		);

		(new_features, new_labels)
	}

	/// Build an IDSSubset from (owned) features and labels.
	fn build_subset(
		features: PackedBits,
		labels: &[i64],
		total_features: usize,
		num_classes: usize,
		num_negatives: usize,
		seed: u64,
	) -> IDSSubset
	{
		let num_examples = labels.len();
		if num_examples == 0
		{
			return IDSSubset {
				input_bits: PackedBits::empty(total_features),
				targets: vec![],
				negatives: vec![],
				num_examples: 0,
			};
		}
		IDSSubset {
			input_bits: features,
			targets: labels.to_vec(),
			negatives: Self::build_negatives(labels, num_classes, num_negatives, seed),
			num_examples,
		}
	}

	/// Negatives: for each example, all classes except the target. If
	/// num_negatives < num_classes - 1, randomly sample from non-target classes.
	/// The ONLY consumer of the `seed` RNG (so a subset's negatives depend on
	/// its labels + seed alone, never on its input rows).
	fn build_negatives(labels: &[i64], num_classes: usize, num_negatives: usize, seed: u64) -> Vec<i64>
	{
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
		let mut negatives = vec![0i64; labels.len() * num_negatives];
		for (ex, &target) in labels.iter().enumerate()
		{
			let row = &mut negatives[ex * num_negatives..(ex + 1) * num_negatives];
			Self::fill_example_negatives(row, target, num_classes, &mut rng);
		}
		negatives
	}

	/// One example's negatives (verbatim body of the old per-example loop).
	fn fill_example_negatives(row: &mut [i64], target: i64, num_classes: usize, rng: &mut rand::rngs::StdRng)
	{
		let num_negatives = row.len();
		if num_negatives >= num_classes - 1
		{
			// Exhaustive: all classes except target
			let mut neg_idx = 0;
			for c in (0..num_classes as i64).filter(|&c| c != target)
			{
				row[neg_idx] = c;
				neg_idx += 1;
			}
			// If num_negatives > num_classes - 1, fill remaining with random
			while neg_idx < num_negatives
			{
				let mut c = rng.gen_range(0..num_classes) as i64;
				while c == target
				{
					c = rng.gen_range(0..num_classes) as i64;
				}
				row[neg_idx] = c;
				neg_idx += 1;
			}
		}
		else
		{
			// Sample num_negatives from non-target classes
			let mut available: Vec<i64> = (0..num_classes as i64).filter(|&c| c != target).collect();
			available.shuffle(rng);
			for k in 0..num_negatives
			{
				row[k] = available[k % available.len()];
			}
		}
	}

	/// Create stratified folds by distributing examples round-robin by class.
	///
	/// This preserves class distribution in each fold, which is critical
	/// for imbalanced IDS datasets (e.g., Worms has only 44 examples).
	/// Same RNG stream, row order and per-fold negative seeds as the old
	/// `create_stratified_subsets` — only the rows are indices now.
	fn create_stratified_folds(
		labels: &[i64],
		num_classes: usize,
		num_negatives: usize,
		num_parts: usize,
		seed: u64,
	) -> Vec<crate::ids_fold::IDSFold>
	{
		assert!(labels.len() <= u32::MAX as usize, "IDS fold rows are u32: {} train rows", labels.len());
		Self::stratified_fold_rows(labels, num_classes, num_parts, seed)
			.into_iter()
			.enumerate()
			.map(|(part_idx, rows)| {
				let fold_labels: Vec<i64> = rows.iter().map(|&idx| labels[idx]).collect();
				let fold_seed = seed + (part_idx as u64 + 1) * 1000;
				crate::ids_fold::IDSFold {
					negatives: Self::build_negatives(&fold_labels, num_classes, num_negatives, fold_seed),
					rows: rows.iter().map(|&r| r as u32).collect(),
					targets: fold_labels,
				}
			})
			.collect()
	}

	/// The per-fold row lists: shuffle within class, deal round-robin, shuffle
	/// within fold (one `seed + 50` stream, in that order).
	fn stratified_fold_rows(labels: &[i64], num_classes: usize, num_parts: usize, seed: u64) -> Vec<Vec<usize>>
	{
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 50);
		let mut class_indices: Vec<Vec<usize>> = vec![vec![]; num_classes];
		for (i, &label) in labels.iter().enumerate()
		{
			if (label as usize) < num_classes
			{
				class_indices[label as usize].push(i);
			}
		}
		for indices in &mut class_indices
		{
			indices.shuffle(&mut rng);
		}
		let mut subset_indices: Vec<Vec<usize>> = vec![vec![]; num_parts];
		for class_idx_list in &class_indices
		{
			for (i, &ex_idx) in class_idx_list.iter().enumerate()
			{
				subset_indices[i % num_parts].push(ex_idx);
			}
		}
		for indices in &mut subset_indices
		{
			indices.shuffle(&mut rng);
		}
		subset_indices
	}

	// ── Accessors ──────────────────────────────────────────────────────

	pub fn next_train_idx(&mut self) -> usize
	{
		self.train_rotator.next()
	}

	/// Fold `idx` materialised as an IDSSubset (rows gathered from
	/// full_train — transient, freed by the caller).
	pub fn train_subset(&self, idx: usize) -> IDSSubset
	{
		self.subset_of(&self.train_folds[idx])
	}

	/// The k-fold training set: every fold except `held_out`, concatenated in
	/// fold order and gathered straight into the GPU word layout (one copy;
	/// was a merge copy + a u64 re-pack).
	pub fn kfold_train_set(&self, held_out: usize) -> IDSSubset
	{
		let others = self.train_folds.iter().enumerate().filter(move |(i, _)| *i != held_out).map(|(_, f)| f);
		self.subset_of(&crate::ids_fold::IDSFold::concat(others))
	}

	/// A fold-shaped row list as a materialised IDSSubset.
	fn subset_of(&self, fold: &crate::ids_fold::IDSFold) -> IDSSubset
	{
		IDSSubset {
			input_bits: fold.gather_input(&self.full_train.input_bits),
			targets: fold.targets.clone(),
			negatives: fold.negatives.clone(),
			num_examples: fold.num_examples(),
		}
	}

	pub fn full_train(&self) -> &IDSSubset
	{
		&self.full_train
	}

	pub fn full_eval(&self) -> &IDSSubset
	{
		&self.full_eval
	}

	pub fn full_val(&self) -> Option<&IDSSubset>
	{
		self.full_val.as_ref()
	}

	pub fn total_features(&self) -> usize
	{
		self.total_features
	}

	pub fn num_classes(&self) -> usize
	{
		self.num_classes
	}

	pub fn num_negatives(&self) -> usize
	{
		self.num_negatives
	}

	pub fn num_genome_clusters(&self) -> usize
	{
		self.num_genome_clusters
	}

	pub fn num_train_subsets(&self) -> usize
	{
		self.train_folds.len()
	}

	pub fn reset(&mut self, seed: Option<u64>)
	{
		self.train_rotator.reset(seed);
	}

	pub fn num_parts(&self) -> usize
	{
		self.num_parts
	}

	/// The absolute desirability CE half-anchor for this cache's task, derived
	/// from the FULL train partition's own label distribution.
	///
	/// Reads the labels already resident in this cache — nothing crosses the
	/// FFI boundary and nothing is materialised Python-side. The cache also
	/// already knows `num_classes`, so the binary and multiclass arms of one
	/// cohort each derive their own scale without the caller tracking which
	/// classification mode it asked for.
	pub fn desirability_ce_anchor(&self) -> Result<f64, String>
	{
		crate::base_rate_entropy::desirability_ce_anchor(
			&self.full_train.targets,
			self.num_classes,
			crate::base_rate_entropy::NORMALIZED_CE_ANCHOR,
		)
	}

	/// H(p) in nats over the full train partition — the reference scale itself,
	/// exposed for logging and for the anchor-provenance line in run output.
	pub fn base_rate_entropy(&self) -> Result<f64, String>
	{
		crate::base_rate_entropy::base_rate_entropy(&self.full_train.targets, self.num_classes)
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
) -> Vec<(f64, f64, f64, f64, f64, u32)>
{
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
		cache.eval_settings(empty_value),
		neuron_sample_rate,
		rng_seed,
		cache.class_weights.as_deref(),
	)
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
) -> Vec<(f64, f64, f64, f64, f64, u32)>
{
	assert!(
		held_out_fold < cache.num_parts(),
		"held_out_fold {} >= num_parts {}",
		held_out_fold,
		cache.num_parts(),
	);

	// Train set: all folds except the held-out one, gathered once.
	let train = cache.kfold_train_set(held_out_fold);

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
		cache.eval_settings(empty_value),
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
) -> Vec<(f64, f64, f64, f64, f64, u32)>
{
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
		cache.eval_settings(empty_value),
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
) -> Vec<i64>
{
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
		cache.eval_settings(empty_value),
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
) -> Vec<f64>
{
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
		cache.eval_settings(empty_value),
		neuron_sample_rate,
		rng_seed,
		cache.class_weights.as_deref(),
	)
}

/// `evaluate_at_thresholds_ids_cached` result: eval/train/val scores, per-threshold
/// `(ce, acc, f1, fpr, resolved_threshold)` and the trained genome's footprint.
pub type ThresholdEvaluation = (
	Vec<f64>,
	Vec<f64>,
	Option<Vec<f64>>,
	Vec<(f64, f64, f64, f64, f64)>,
	ram_core::footprint::Footprint,
);

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
/// Returns `(eval_scores, train_scores, val_scores, metrics_per_threshold, footprint)` where
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
) -> ThresholdEvaluation
{
	let train = cache.full_train();
	let eval = cache.full_eval();
	let val = cache.full_val();

	// Train ONCE + score eval, train (and val, when present) sets
	let (eval_scores, train_scores, val_scores, footprint) = crate::adaptive::train_and_score_eval_and_train(
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
		cache.eval_settings(empty_value),
		neuron_sample_rate,
		rng_seed,
		cache.class_weights.as_deref(),
	);

	// Derive metrics at each threshold (handle -1.0 oracle sentinel)
	let normal_class = cache.normal_class;
	let metrics: Vec<(f64, f64, f64, f64, f64)> = thresholds
		.iter()
		.map(|&t| {
			let resolved = if t < 0.0
			{
				// Oracle: sweep eval scores for optimal F1
				let (best_t, _f1, _fpr) =
					crate::adaptive::find_optimal_threshold_f1(&eval_scores, &eval.targets);
				best_t
			}
			else
			{
				t
			};
			let (ce, acc, f1, fpr) = crate::adaptive::compute_binary_metrics_at_threshold(
				&eval_scores,
				&eval.targets,
				resolved,
				normal_class,
			);
			(ce, acc, f1, fpr, resolved)
		})
		.collect();

	(eval_scores, train_scores, val_scores, metrics, footprint)
}

/// Train ONE genome exactly as `evaluate_at_thresholds_ids_cached` does (full
/// train set, same settings, `train_genome_export`) and return it as physical
/// unit tables plus its eval (+ val) scores — the FPGA export's source.
pub fn export_trained_ids_cached(
	cache: &IDSCache,
	bits_flat: &[usize],
	neurons_flat: &[usize],
	connections_flat: &[i64],
	empty_value: f32,
	neuron_sample_rate: f32,
	rng_seed: u64,
) -> crate::trained_layer::TrainedLayer
{
	let (train, eval, val) = (cache.full_train(), cache.full_eval(), cache.full_val());
	let settings = cache.eval_settings(empty_value);
	let clusters = cache.num_genome_clusters();
	let export = crate::adaptive::train_genome_export(
		bits_flat,
		neurons_flat,
		connections_flat,
		clusters,
		&train.input_bits,
		&train.targets,
		&train.negatives,
		train.num_examples,
		cache.num_negatives(),
		cache.total_features(),
		settings,
		neuron_sample_rate,
		rng_seed,
		cache.class_weights.as_deref(),
	);
	let score = |bits: &PackedBits, n: usize| {
		crate::adaptive::score_examples_binary(&export, bits, n, clusters, cache.total_features(), settings)
	};
	let eval_scores = score(&eval.input_bits, eval.num_examples);
	let val_scores = val.map(|v| score(&v.input_bits, v.num_examples));
	let logical = ram_core::super_genome::FlatGenome {
		bits: bits_flat,
		neurons_per_cluster: neurons_flat,
		connections: connections_flat,
	};
	crate::trained_layer::TrainedLayer::from_export(&export, logical, settings, eval_scores, val_scores)
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
/// Returns `(num_classes, mode_results, footprint)`; each result carries the full metric
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
) -> (usize, Vec<crate::multiclass_metrics::MulticlassModeResult>, ram_core::footprint::Footprint)
{
	use crate::multiclass_metrics as mm;

	let num_classes = cache.num_classes();
	assert_eq!(
		cache.num_genome_clusters(),
		num_classes,
		"evaluate_multiclass_at_thresholds requires a K-cluster cache \
         (single_cluster=false); got {} genome clusters for {} classes",
		cache.num_genome_clusters(),
		num_classes,
	);

	let train = cache.full_train();
	let eval = cache.full_eval();
	let val = cache.full_val();

	// Train ONCE + flattened K-vector scores for eval/train(/val)
	let (eval_scores, train_scores, val_scores, footprint) = crate::adaptive::train_and_score_sets_flat(
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
		cache.eval_settings(empty_value),
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

	(num_classes, modes, footprint)
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
) -> Vec<f64>
{
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
		&train.input_bits, // score training data, not eval
		train.num_examples,
		cache.total_features(),
		cache.eval_settings(empty_value),
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
) -> Vec<crate::adaptive::AdaptiveGenomeResult>
{
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
		cache.eval_settings(empty_value),
		neuron_sample_rate,
		rng_seed,
		adapt_config,
		generation,
	)
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn test_ids_cache_basic()
	{
		// 10 examples, 8 features, 2 classes
		let total_features = 8;
		let num_classes = 2;
		let num_examples = 10;

		let mut features = vec![false; num_examples * total_features];
		let mut labels = vec![0i64; num_examples];

		// Fill with some pattern
		for i in 0..num_examples
		{
			labels[i] = (i % num_classes) as i64;
			for j in 0..total_features
			{
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
			2, // num_parts
			1, // num_negatives (binary: 1 negative per example)
			42,
			false, // balance_classes
			false, // single_cluster
			false, // undersample_majority
			1.0,   // class_weight_multiplier
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
	fn test_ids_negatives_binary()
	{
		// Binary classification: each example should have 1 negative (the other class)
		let total_features = 4;
		let features = vec![true, false, true, false, false, true, false, true];
		let labels = vec![0i64, 1i64];
		let packed = PackedBits::from_bool_slice(&features, total_features);

		let subset = IDSCache::build_subset(packed, &labels, total_features, 2, 1, 42);

		assert_eq!(subset.negatives.len(), 2); // 2 examples * 1 negative
		assert_eq!(subset.negatives[0], 1); // example 0 (class 0) → negative is class 1
		assert_eq!(subset.negatives[1], 0); // example 1 (class 1) → negative is class 0
	}

	#[test]
	fn test_ids_negatives_multiclass()
	{
		// 3-class: each example should have 2 negatives (all other classes)
		let total_features = 4;
		let features = vec![true; 3 * total_features];
		let labels = vec![0i64, 1i64, 2i64];
		let packed = PackedBits::from_bool_slice(&features, total_features);

		let subset = IDSCache::build_subset(packed, &labels, total_features, 3, 2, 42);

		assert_eq!(subset.negatives.len(), 6); // 3 examples * 2 negatives

		// Example 0 (class 0): negatives should be 1 and 2
		let neg0: Vec<i64> = subset.negatives[0..2].to_vec();
		assert!(!neg0.contains(&0));
		assert!(neg0.contains(&1));
		assert!(neg0.contains(&2));
	}

	#[test]
	fn test_stratified_partitioning()
	{
		// Ensure stratified split preserves class distribution
		let num_classes = 2;

		// 70 class 0, 30 class 1
		let mut labels = vec![0i64; 70];
		labels.extend(vec![1i64; 30]);

		let folds = IDSCache::create_stratified_folds(&labels, num_classes, 1, 3, 42);

		// Each of 3 folds should have roughly 70/3 ≈ 23 class-0 and 30/3 = 10 class-1
		for fold in &folds
		{
			let class0_count = fold.targets.iter().filter(|&&t| t == 0).count();
			let class1_count = fold.targets.iter().filter(|&&t| t == 1).count();
			assert!((20..=27).contains(&class0_count), "class0: {} not in [20,27]", class0_count);
			assert!((8..=12).contains(&class1_count), "class1: {} not in [8,12]", class1_count);
		}
	}

	/// The pre-13/10/2026 fold construction, verbatim: every fold MATERIALISED
	/// (select_rows on the train matrix) with negatives from the old
	/// build_subset loop. The oracle the index-based folds must reproduce.
	fn legacy_stratified_subsets(
		features: &PackedBits,
		labels: &[i64],
		num_classes: usize,
		num_negatives: usize,
		num_parts: usize,
		seed: u64,
	) -> Vec<IDSSubset>
	{
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 50);
		let mut class_indices: Vec<Vec<usize>> = vec![vec![]; num_classes];
		for (i, &label) in labels.iter().enumerate()
		{
			let class_idx = label as usize;
			if class_idx < num_classes
			{
				class_indices[class_idx].push(i);
			}
		}
		for indices in &mut class_indices
		{
			indices.shuffle(&mut rng);
		}
		let mut subset_indices: Vec<Vec<usize>> = vec![vec![]; num_parts];
		for class_idx_list in &class_indices
		{
			for (i, &ex_idx) in class_idx_list.iter().enumerate()
			{
				subset_indices[i % num_parts].push(ex_idx);
			}
		}
		for indices in &mut subset_indices
		{
			indices.shuffle(&mut rng);
		}
		subset_indices
			.iter()
			.enumerate()
			.map(|(part_idx, indices)| {
				let sub_labels: Vec<i64> = indices.iter().map(|&idx| labels[idx]).collect();
				legacy_build_subset(
					&features.select_rows(indices),
					&sub_labels,
					num_classes,
					num_negatives,
					seed + (part_idx as u64 + 1) * 1000,
				)
			})
			.collect()
	}

	/// The pre-13/10/2026 build_subset negatives loop, verbatim.
	fn legacy_build_subset(features: &PackedBits, labels: &[i64], num_classes: usize, num_negatives: usize, seed: u64) -> IDSSubset
	{
		let num_examples = labels.len();
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
		let mut negatives = vec![0i64; num_examples * num_negatives];
		if num_examples > 0
		{
			let all_negatives_needed = num_classes - 1;
			for ex in 0..num_examples
			{
				let target = labels[ex];
				let neg_offset = ex * num_negatives;
				if num_negatives >= all_negatives_needed
				{
					let mut neg_idx = 0;
					for c in 0..num_classes as i64
					{
						if c != target
						{
							negatives[neg_offset + neg_idx] = c;
							neg_idx += 1;
						}
					}
					while neg_idx < num_negatives
					{
						let mut c = rng.gen_range(0..num_classes) as i64;
						while c == target
						{
							c = rng.gen_range(0..num_classes) as i64;
						}
						negatives[neg_offset + neg_idx] = c;
						neg_idx += 1;
					}
				}
				else
				{
					let mut available: Vec<i64> = (0..num_classes as i64).filter(|&c| c != target).collect();
					available.shuffle(&mut rng);
					for k in 0..num_negatives
					{
						negatives[neg_offset + k] = available[k % available.len()];
					}
				}
			}
		}
		IDSSubset {
			input_bits: features.clone(),
			targets: labels.to_vec(),
			negatives,
			num_examples,
		}
	}

	/// The pre-13/10/2026 k-fold merge (sequential extend), verbatim.
	fn legacy_merge_except(subsets: &[IDSSubset], exclude_idx: usize, total_features: usize) -> IDSSubset
	{
		let mut merged = IDSSubset {
			input_bits: PackedBits::empty(total_features),
			targets: vec![],
			negatives: vec![],
			num_examples: 0,
		};
		for (i, subset) in subsets.iter().enumerate().filter(|(i, _)| *i != exclude_idx)
		{
			let _ = i;
			merged.input_bits.extend_from(&subset.input_bits);
			merged.targets.extend_from_slice(&subset.targets);
			merged.negatives.extend_from_slice(&subset.negatives);
			merged.num_examples += subset.num_examples;
		}
		merged
	}

	fn assert_same_subset(a: &IDSSubset, b: &IDSSubset, what: &str)
	{
		assert_eq!(a.num_examples, b.num_examples, "{}: num_examples", what);
		assert_eq!(a.targets, b.targets, "{}: targets", what);
		assert_eq!(a.negatives, b.negatives, "{}: negatives", what);
		assert_eq!(a.input_bits.num_rows(), b.input_bits.num_rows(), "{}: rows", what);
		assert_eq!(a.input_bits.as_u64_words(), b.input_bits.as_u64_words(), "{}: input words", what);
	}

	/// Random (features, labels) with every class present.
	fn random_train(n: usize, total_features: usize, num_classes: usize, seed: u64) -> (Vec<bool>, Vec<i64>)
	{
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
		let features = (0..n * total_features).map(|_| rng.gen::<bool>()).collect();
		let labels = (0..n).map(|i| if i < num_classes { i as i64 } else { rng.gen_range(0..num_classes) as i64 }).collect();
		(features, labels)
	}

	/// Folds as row indices + on-demand gather reproduce the old materialised
	/// folds, the old k-fold merge, and the old full_train — byte for byte, for
	/// binary single-cluster (0 negatives), binary 1-negative, and multiclass
	/// with SAMPLED negatives (the per-fold RNG path), at a row width that
	/// needs word padding (1870 bits -> 234 B -> 30 words).
	#[test]
	fn index_folds_match_legacy_materialised_folds()
	{
		for &(num_classes, num_negatives, single_cluster, total_features) in
			&[(2usize, 1usize, true, 1870usize), (2, 1, false, 70), (5, 2, false, 13), (4, 3, false, 64)]
		{
			let (features, labels) = random_train(203, total_features, num_classes, 9 + num_classes as u64);
			let packed = PackedBits::from_bool_slice(&features, total_features);
			let cache = IDSCache::new(
				packed.clone(),
				labels.clone(),
				PackedBits::from_bool_slice(&features[..10 * total_features], total_features),
				labels[..10].to_vec(),
				None,
				None,
				num_classes,
				total_features,
				5,
				num_negatives,
				42,
				false,
				single_cluster,
				false,
				1.0,
			);
			let negs = if single_cluster { 0 } else { num_negatives };
			let legacy = legacy_stratified_subsets(&packed, &labels, num_classes, negs, 5, 42);
			let full_legacy = legacy_build_subset(&packed, &labels, num_classes, negs, 42);
			assert_same_subset(cache.full_train(), &full_legacy, "full_train");
			for k in 0..5
			{
				assert_same_subset(&cache.train_subset(k), &legacy[k], &format!("fold {}", k));
				let merged = legacy_merge_except(&legacy, k, total_features);
				assert_same_subset(&cache.kfold_train_set(k), &merged, &format!("kfold train {}", k));
			}
		}
	}

	#[test]
	fn test_kfold_uses_all_data()
	{
		// Verify that K-fold uses training data (not eval data) for both train and eval
		let total_features = 4;
		let num_classes = 2;
		let num_parts = 3;
		let num_train = 12;

		let mut features = vec![false; num_train * total_features];
		let mut labels = vec![0i64; num_train];
		for i in 0..num_train
		{
			labels[i] = (i % num_classes) as i64;
			for j in 0..total_features
			{
				features[i * total_features + j] = ((i + j) % 2) == 0;
			}
		}

		let cache = IDSCache::new(
			PackedBits::from_bool_slice(&features, total_features),
			labels.clone(),
			PackedBits::from_bool_slice(&vec![false; 4 * total_features], total_features), // small eval set (shouldn't be used in kfold)
			vec![0i64; 4],
			None,
			None,
			num_classes,
			total_features,
			num_parts,
			1,
			42,
			false, // balance_classes
			false, // single_cluster
			false, // undersample_majority
			1.0,   // class_weight_multiplier
		);

		let total_in_subsets: usize = (0..num_parts).map(|i| cache.train_subset(i).num_examples).sum();
		assert_eq!(total_in_subsets, num_train);

		for held_out in 0..num_parts
		{
			let train_merged = cache.kfold_train_set(held_out);
			let eval_fold = cache.train_subset(held_out);
			assert_eq!(
				train_merged.num_examples + eval_fold.num_examples,
				num_train,
				"fold {}: train + eval should equal total",
				held_out
			);
		}
	}
}
