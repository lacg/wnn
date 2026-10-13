//! Streaming IDS genome evaluator (Option F).
//!
//! Unlike `IDSCache` (which holds the full train/eval matrices in memory),
//! this evaluator holds **only the genome state** — memory cells and
//! per-cluster config — and processes data in chunks. Peak memory is one
//! chunk regardless of dataset size, enabling evaluation on datasets that
//! don't fit in RAM or even on disk.
//!
//! ## Lifecycle (per genome)
//!
//! 1. `IDSGenomeStreamer::new(genome, config)` — initializes empty memory cells.
//! 2. Train phase: stream the training source chunk-by-chunk, calling
//!    `train_chunk(packed, labels)` each time. The chunk's per-row data
//!    drives memory cell writes via the existing `train_genome_in_slot`
//!    kernel; the chunk is dropped after the call.
//! 3. `seal_for_scoring()` — finalizes training. Builds the
//!    `GenomeExport` (memory cell snapshot in GPU-ready form) once.
//! 4. Score phase: stream the eval source chunk-by-chunk, calling
//!    `score_chunk(packed, labels)`. Per-row scores are accumulated in
//!    a small `eval_scores: Vec<f64>` buffer.
//! 5. `finalize_metrics() → (ce, acc, f1, fpr, threshold)` — runs the
//!    same threshold/metric logic as `evaluate_genome_hybrid`'s
//!    single-cluster path on the accumulated scores.
//!
//! ## What this doesn't (yet) support
//!
//! - **undersample_majority**: needs full-dataset class statistics AND
//!   row-level rejection of majority-class samples in the stream. v1
//!   omits this (callers pass `undersample_majority=False`). Adding it
//!   would mean: materialize per-class row indices, build a Bernoulli
//!   accept mask per class, gate train_chunk rows on that mask.
//!
//! Note: **balance_classes** IS supported via class_weights — the caller
//! computes weights from the materialized label array (already in RAM at
//! `IDSDataset.y_train_binary`) and passes them to `new()`. No streaming
//! pre-pass needed — labels are tiny relative to features and materialized
//! once during dataset construction.
//! - **GPU train-side address pre-computation**: `train_genome_in_slot`
//!   accepts an optional `gpu_addresses` slice computed up-front from
//!   the full packed dataset. For streaming, addresses are computed
//!   per-row on the CPU. Eval scoring still uses GPU dispatch via the
//!   already-existing per-chunk path in `compute_per_example_scores`.
//!
//! Multi-cluster (K-class) mode IS supported (since 11/07/2026): with
//! `single_cluster=false`, `score_chunk` accumulates all K per-cluster
//! scores per row (flat row-major) and `finalize_metrics` returns the
//! K-class search metrics (softmax CE, argmax accuracy, macro-F1,
//! benign-FPR) mirroring `evaluate_genome_hybrid`'s K-cluster path.
//! `take_scores` drains the flat K-vector buffer for the Protocol-v2
//! decode-mode evaluation (`multiclass_metrics::modes_from_scores`).
//! Labels passed to train_chunk/score_chunk must then be K-class indices
//! (the Python caller slices the materialized `y_*_multi` arrays by
//! chunk offset — stream order == materialization order by construction).

use crate::adaptive::{
	build_groups, build_neuron_metadata, compute_f1_fpr_with_normal_class,
	compute_per_example_scores, export_genome_for_gpu, find_optimal_threshold_auto,
	get_metal_evaluator, get_sparse_metal_evaluator, per_cluster_max_bits,
	reorganize_connections_for_gpu, train_genome_in_slot_range, ConfigGroup, GenomeExport, GroupMemory,
};
use ram_core::neuron_memory::pack_packed_to_u64;
use ram_core::packed_bits::PackedBits;
use ram_core::physical_genome::PhysicalGenome;

/// Streaming evaluator state for a single genome.
///
/// One instance per (genome, evaluation pass). The training and scoring
/// phases consume chunks sequentially; only the memory cells (genome state)
/// and the small `eval_scores` accumulator persist across chunks.
pub struct IDSGenomeStreamer
{
	// ── Frozen config ────────────────────────────────────────────────────
	num_classes: usize,
	num_negatives: usize,
	num_genome_clusters: usize,
	normal_class: usize,
	/// (w_ce, w_f1, w_fpr, w_acc) — when Some, threshold sweep maximizes fitness.
	pub fitness_weights: Option<(f32, f32, f32, f32)>,
	empty_value: f32,
	neuron_sample_rate: f32,
	rng_seed: u64,
	memory_mode: u8,
	/// Per-class repetition weights for balance_classes=True. None = no balancing.
	/// Computed by the caller (Python) from materialized y_train labels.
	class_weights: Option<Vec<u32>>,

	// ── Genome (the TRAINED one: physical when the multiplicity layer applies)
	bits_flat: Vec<usize>,
	neurons_flat: Vec<usize>,
	original_connections: Vec<i64>,
	/// The LOGICAL genome + its grouping when duplicates were collapsed
	/// (IDS-37 stage 8): `seal_for_scoring` rebuilds the logical export from
	/// the trained units (`GenomeExport::from_physical`). None = plain layer.
	logical: Option<PhysicalGenomeOf>,

	// ── Derived once at construction ─────────────────────────────────────
	groups: Vec<ConfigGroup>,
	cluster_to_group: Vec<(usize, usize)>,
	cluster_neuron_starts: Vec<usize>,
	neuron_conn_offsets: Vec<usize>,

	// ── Training phase state ────────────────────────────────────────────
	memories: Vec<GroupMemory>,
	/// Order-independent training: the OI counters are opened ONCE in `new`
	/// and committed ONCE in `seal_for_scoring`, so every chunk's evidence
	/// tallies together (as the in-memory one-shot trainer does). They used
	/// to be opened and committed per `train_chunk`, so a later chunk
	/// overwrote the cells it touched with ITS tallies only.
	oi: bool,
	train_seen: usize, // count of rows seen during training (for stats)

	// ── Scoring phase state ─────────────────────────────────────────────
	export: Option<GenomeExport>,
	eval_scores: Vec<f64>,
	eval_labels: Vec<i64>,
}

impl IDSGenomeStreamer
{
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		bits_flat: Vec<usize>,
		neurons_flat: Vec<usize>,
		connections: Vec<i64>,
		num_classes: usize,
		num_negatives: usize,
		single_cluster: bool,
		normal_class: usize,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
		class_weights: Option<Vec<u32>>,
	) -> Self
	{
		// Streamer is QUAD-only (Option F shipped after the QUAD mandate).
		let memory_mode = ram_core::neuron_memory::QUAD_WEIGHTED;
		let (bits_flat, neurons_flat, connections, logical) =
			PhysicalGenomeOf::trained_genome(bits_flat, neurons_flat, connections, neuron_sample_rate, memory_mode);
		let bits_per_cluster = per_cluster_max_bits(&bits_flat, &neurons_flat);
		let groups = build_groups(&bits_per_cluster, &neurons_flat);
		let (cluster_neuron_starts, neuron_conn_offsets) =
			build_neuron_metadata(&bits_flat, &neurons_flat);

		let num_clusters = neurons_flat.len();
		let num_genome_clusters = if single_cluster { 1 } else { num_classes };
		let actual_negatives = if single_cluster { 0 } else { num_negatives };

		let mut cluster_to_group = vec![(0usize, 0usize); num_clusters];
		for (group_idx, group) in groups.iter().enumerate()
		{
			for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate()
			{
				cluster_to_group[cluster_id] = (group_idx, local_idx);
			}
		}

		let mut memories: Vec<GroupMemory> = groups
			.iter()
			.map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
			.collect();
		let oi = ram_core::neuron_memory::order_independent_training_active(memory_mode);
		if oi
		{
			memories.iter_mut().for_each(GroupMemory::init_oi_counters);
		}

		Self {
			num_classes,
			num_negatives: actual_negatives,
			num_genome_clusters,
			normal_class,
			fitness_weights: None,
			empty_value,
			neuron_sample_rate,
			rng_seed,
			memory_mode,
			class_weights,
			bits_flat,
			neurons_flat,
			original_connections: connections,
			logical,
			groups,
			cluster_to_group,
			cluster_neuron_starts,
			neuron_conn_offsets,
			memories,
			oi,
			train_seen: 0,
			export: None,
			eval_scores: Vec::new(),
			eval_labels: Vec::new(),
		}
	}

	/// Train on a single chunk. Writes to memory cells in-place.
	///
	/// `packed`: bit-packed features for this chunk.
	/// `labels`: class index per row (len == packed.num_rows()).
	pub fn train_chunk(&mut self, packed: &PackedBits, labels: &[i64])
	{
		assert_eq!(
			packed.num_rows(),
			labels.len(),
			"train_chunk: feature rows ({}) != label count ({})",
			packed.num_rows(),
			labels.len()
		);
		if self.export.is_some()
		{
			panic!("train_chunk called after seal_for_scoring()");
		}

		let num_train = packed.num_rows();
		let total_input_bits = packed.total_bits();

		// Negatives per chunk: for each example, the K class indices that are
		// NOT the target. For single_cluster mode (num_negatives=0), this is
		// empty. For multi-class, exhaustive enumeration: all classes except target.
		let negatives = compute_chunk_negatives(labels, self.num_classes, self.num_negatives);

		train_genome_in_slot_range(
			&self.memories,
			&self.groups,
			&self.original_connections,
			&self.bits_flat,
			&self.cluster_neuron_starts,
			&self.neuron_conn_offsets,
			&self.cluster_to_group,
			packed,
			labels,
			&negatives,
			num_train,
			self.num_negatives,
			total_input_bits,
			None, // gpu_addresses: per-row CPU compute (streaming v1)
			0..num_train,
			num_train,
			self.neuron_sample_rate,
			self.rng_seed.wrapping_add(self.train_seen as u64),
			self.memory_mode,
			self.class_weights.as_deref(),
			true, // parallel
		);

		self.train_seen += num_train;
	}

	/// Finalize training. Builds the GPU-ready genome export so subsequent
	/// `score_chunk` calls can dispatch to Metal where dense.
	pub fn seal_for_scoring(&mut self)
	{
		if self.export.is_some()
		{
			panic!("seal_for_scoring called twice");
		}
		if self.oi
		{
			self.memories.iter_mut().for_each(GroupMemory::commit_oi);
		}
		let gpu_connections = reorganize_connections_for_gpu(
			&self.original_connections,
			&self.bits_flat,
			&self.neurons_flat,
			&self.groups,
		);
		let trained = export_genome_for_gpu(&self.memories, &self.groups, &gpu_connections);
		self.export = Some(match &self.logical
		{
			None => trained,
			Some(logical) => logical.logical_export(&trained, &self.neurons_flat, self.memory_mode),
		});
	}

	/// Logical/physical footprint of the sealed memory (None before
	/// `seal_for_scoring`). Physical < logical when the multiplicity layer
	/// trained the distinct neurons only.
	pub fn footprint(&self) -> Option<ram_core::footprint::Footprint>
	{
		self.export.as_ref().map(GenomeExport::footprint)
	}

	/// Score a single eval chunk. Accumulates per-row scores into the
	/// `eval_scores` buffer (small — one f64 per eval row).
	pub fn score_chunk(&mut self, packed: &PackedBits, labels: &[i64])
	{
		assert_eq!(
			packed.num_rows(),
			labels.len(),
			"score_chunk: feature rows ({}) != label count ({})",
			packed.num_rows(),
			labels.len()
		);
		let export = self
			.export
			.as_ref()
			.expect("score_chunk called before seal_for_scoring — finalize training first");

		let num_eval = packed.num_rows();
		let total_input_bits = packed.total_bits();
		let (packed_eval_u64, words_per_example) = pack_packed_to_u64(packed);

		let metal_arc = get_metal_evaluator();
		let sparse_metal_arc = get_sparse_metal_evaluator();
		let metal = metal_arc.as_ref().map(|a| a.as_ref());
		let sparse_metal = sparse_metal_arc.as_ref().map(|a| a.as_ref());

		let chunk_scores = compute_per_example_scores(
			export,
			packed,
			&packed_eval_u64,
			words_per_example,
			num_eval,
			self.num_genome_clusters,
			total_input_bits,
			self.empty_value,
			self.memory_mode,
			0, // run_seed: streaming QSR/PLN not yet wired (46M path, not the abl cohorts)
			false, // coverage_aware: streaming not wired (MCST uses ids_encoded_storage=memory)
			metal,
			sparse_metal,
		);

		// Single-cluster: score[ex][0] is the attack probability.
		// Multi-cluster (K-class): accumulate ALL K per-cluster scores per
		// row, flat row-major (scores_flat[ex*K + c] — the
		// multiclass_metrics convention).
		if self.num_genome_clusters > 1
		{
			for scores in chunk_scores
			{
				self.eval_scores.extend_from_slice(&scores);
			}
		}
		else
		{
			for scores in chunk_scores
			{
				self.eval_scores.push(scores[0]);
			}
		}
		self.eval_labels.extend_from_slice(labels);
	}

	/// Compute final metrics from the accumulated eval scores.
	///
	/// Returns (ce, acc, f1, fpr, threshold). Single-cluster: threshold is
	/// auto-selected on the eval data (matches `evaluate_genome_hybrid`'s
	/// single-cluster path when override_threshold=None). Multi-cluster
	/// (K-class): argmax decode + softmax CE + macro-F1/benign-FPR —
	/// mirrors `evaluate_genome_hybrid`'s K-cluster CPU path exactly
	/// (search-comparable; threshold is the 0.5 placeholder). NOTE: this
	/// softmax search CE is NOT numerically comparable with the
	/// sum-normalized validation CE of `multiclass_metrics` — see that
	/// module's header.
	pub fn finalize_metrics(&self) -> (f64, f64, f64, f64, f64)
	{
		if self.num_genome_clusters > 1
		{
			return self.finalize_metrics_multiclass();
		}
		let epsilon = 1e-10f64;
		let num_eval = self.eval_scores.len();
		assert!(num_eval > 0, "finalize_metrics: no scores accumulated");

		// BCE loss (threshold-independent)
		let mut total_ce = 0.0f64;
		for (i, &s) in self.eval_scores.iter().enumerate()
		{
			let s = s.clamp(epsilon, 1.0 - epsilon);
			let y = self.eval_labels[i] as f64;
			total_ce += -(y * s.ln() + (1.0 - y) * (1.0 - s).ln());
		}
		let ce = total_ce / num_eval as f64;

		// Auto-find threshold + compute predictions/metrics at it
		let (threshold, _f1_at_thr, _fpr_at_thr) =
			find_optimal_threshold_auto(&self.eval_scores, &self.eval_labels, self.fitness_weights);

		let mut correct = 0u64;
		let mut predictions = Vec::with_capacity(num_eval);
		for (i, &s) in self.eval_scores.iter().enumerate()
		{
			let pred = if s >= threshold { 1u32 } else { 0u32 };
			predictions.push(pred);
			if pred as i64 == self.eval_labels[i]
			{
				correct += 1;
			}
		}
		let acc = correct as f64 / num_eval as f64;
		let (f1, fpr) =
			compute_f1_fpr_with_normal_class(&predictions, &self.eval_labels, 2, self.normal_class);

		(ce, acc, f1, fpr, threshold)
	}

	/// K-cluster finalize: argmax predictions + softmax CE + accuracy +
	/// macro-F1/benign-FPR via `compute_f1_fpr_with_normal_class`. Mirrors
	/// the K-cluster CPU path of `evaluate_genome_hybrid` (eval_single.rs),
	/// including its tie-breaking (last of equal maxima, max_by semantics).
	fn finalize_metrics_multiclass(&self) -> (f64, f64, f64, f64, f64)
	{
		let epsilon = 1e-10f64;
		let k = self.num_genome_clusters;
		let num_eval = self.eval_scores.len() / k;
		assert!(num_eval > 0, "finalize_metrics: no scores accumulated");
		assert_eq!(
			self.eval_labels.len(),
			num_eval,
			"finalize_metrics: {} labels for {} score rows (K={})",
			self.eval_labels.len(),
			num_eval,
			k,
		);

		let mut predictions = Vec::with_capacity(num_eval);
		let mut total_ce = 0.0f64;
		let mut correct = 0u64;
		for ex in 0..num_eval
		{
			let scores = &self.eval_scores[ex * k..(ex + 1) * k];
			let (mut best_c, mut max_score) = (0usize, f64::NEG_INFINITY);
			for (c, &s) in scores.iter().enumerate()
			{
				if s >= max_score
				{
					max_score = s;
					best_c = c;
				}
			}
			predictions.push(best_c as u32);

			let target = self.eval_labels[ex] as usize;
			if best_c == target
			{
				correct += 1;
			}
			let sum_exp: f64 = scores.iter().map(|&s| (s - max_score).exp()).sum();
			let target_prob = if target < k
			{
				(scores[target] - max_score).exp() / sum_exp
			}
			else
			{
				0.0 // out-of-range label: maximally wrong, don't panic
			};
			total_ce += -(target_prob + epsilon).ln();
		}

		let ce = total_ce / num_eval as f64;
		let acc = correct as f64 / num_eval as f64;
		let (f1, fpr) =
			compute_f1_fpr_with_normal_class(&predictions, &self.eval_labels, k, self.normal_class);
		(ce, acc, f1, fpr, 0.5)
	}

	/// Drain the accumulated per-row scores, resetting the buffer so another
	/// scoring pass can run against the same sealed export.
	///
	/// Protocol v2: lets one trained genome score the eval, train, and val
	/// sets in sequence — score_chunk over a set, take_scores, repeat. The
	/// labels accumulated alongside are cleared too (the caller holds its own
	/// label arrays for calibration/metrics).
	pub fn take_scores(&mut self) -> Vec<f64>
	{
		self.eval_labels.clear();
		std::mem::take(&mut self.eval_scores)
	}

	/// Number of training rows seen so far.
	pub fn train_seen(&self) -> usize
	{
		self.train_seen
	}

	/// Number of eval rows scored so far (multi-cluster stores K scores
	/// per row, so divide the flat buffer length back to rows).
	pub fn eval_scored(&self) -> usize
	{
		self.eval_scores.len() / self.num_genome_clusters.max(1)
	}
}

/// The logical genome a streamer collapsed (IDS-37 stage 8).
struct PhysicalGenomeOf
{
	bits: Vec<usize>,
	neurons_per_cluster: Vec<usize>,
	connections: Vec<i64>,
	physical: PhysicalGenome,
}

impl PhysicalGenomeOf
{
	/// The genome the streamer trains: the physical one (duplicates collapsed)
	/// exactly when `multiplicity_applies` (copies then train identically —
	/// the CPU slot trainer gives each neuron the same writes), else the
	/// logical one unchanged.
	#[allow(clippy::type_complexity)]
	fn trained_genome(
		bits: Vec<usize>,
		npc: Vec<usize>,
		conns: Vec<i64>,
		neuron_sample_rate: f32,
		memory_mode: u8,
	) -> (Vec<usize>, Vec<usize>, Vec<i64>, Option<Self>)
	{
		if conns.is_empty() || !ram_core::neuron_memory::multiplicity_applies(neuron_sample_rate, memory_mode)
		{
			return (bits, npc, conns, None);
		}
		let physical = PhysicalGenome::collapse(&bits, &npc, &conns);
		if physical.grouping.is_identity()
		{
			return (bits, npc, conns, None);
		}
		let trained = (physical.bits.clone(), physical.neurons_per_cluster.clone(), physical.connections.clone());
		(trained.0, trained.1, trained.2, Some(Self { bits, neurons_per_cluster: npc, connections: conns, physical }))
	}

	/// The logical scoring export over the trained units.
	fn logical_export(&self, trained: &GenomeExport, trained_npc: &[usize], memory_mode: u8) -> GenomeExport
	{
		let units = crate::adaptive::PhysicalUnits::new(trained, trained_npc);
		let unit_of: Vec<u32> = (0..self.bits.len()).map(|l| self.physical.grouping.index_of(l)).collect();
		let logical = ram_core::super_genome::FlatGenome {
			bits: &self.bits,
			neurons_per_cluster: &self.neurons_per_cluster,
			connections: &self.connections,
		};
		GenomeExport::from_physical(&units, &unit_of, self.physical.grouping.clone(), logical, memory_mode)
			.expect("streamer: logical export from its own physical genome")
	}

}

#[cfg(test)]
mod tests
{
	use super::*;

	/// K=3 streamer with a dummy genome config (metrics tests inject scores
	/// directly; the genome only has to pass construction).
	fn dummy_streamer_k3() -> IDSGenomeStreamer
	{
		let neurons_flat = vec![2usize; 3];
		let bits_flat = vec![6usize; 6]; // per-neuron
		let conns: Vec<i64> = (0..6).flat_map(|_| 0..6i64).collect();
		IDSGenomeStreamer::new(
			bits_flat,
			neurons_flat,
			conns,
			3,     // num_classes
			2,     // num_negatives (exhaustive for K=3)
			false, // single_cluster
			0,     // normal_class
			0.5,
			1.0,
			42,
			None,
		)
	}

	#[test]
	fn multiclass_finalize_hand_crafted()
	{
		let mut s = dummy_streamer_k3();
		// 4 examples × 3 classes; ex3 is true-benign predicted as class 1.
		s.eval_scores = vec![
			0.9, 0.1, 0.1, // → 0 (true 0) ✓
			0.2, 0.7, 0.1, // → 1 (true 1) ✓
			0.3, 0.2, 0.8, // → 2 (true 2) ✓
			0.1, 0.6, 0.3, // → 1 (true 0) ✗ = benign false alarm
		];
		s.eval_labels = vec![0, 1, 2, 0];
		let (ce, acc, f1, fpr, threshold) = s.finalize_metrics();
		assert!((acc - 0.75).abs() < 1e-12, "acc {acc}");
		// per-class F1: c0 (p=1, r=.5) = 2/3; c1 (p=.5, r=1) = 2/3; c2 = 1
		assert!(
			(f1 - (2.0 / 3.0 + 2.0 / 3.0 + 1.0) / 3.0).abs() < 1e-9,
			"macro f1 {f1}"
		);
		assert!((fpr - 0.5).abs() < 1e-12, "benign fpr {fpr}");
		assert_eq!(threshold, 0.5);
		// Softmax CE recomputed independently.
		let mut expect_ce = 0.0f64;
		for (ex, &t) in s.eval_labels.iter().enumerate()
		{
			let row = &s.eval_scores[ex * 3..(ex + 1) * 3];
			let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
			let sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
			expect_ce += -(((row[t as usize] - max).exp() / sum) + 1e-10).ln();
		}
		expect_ce /= 4.0;
		assert!((ce - expect_ce).abs() < 1e-12, "ce {ce} vs {expect_ce}");
	}

	/// Per-class bit patterns → 6 separable rows (2 per class), bool-byte form.
	fn separable_rows_k3() -> (Vec<u8>, Vec<i64>)
	{
		let mut bytes = Vec::new();
		let mut labels = Vec::new();
		for rep in 0..2
		{
			for c in 0..3usize
			{
				for b in 0..6
				{
					// class c sets bits {2c, 2c+1}; rep 1 also sets no extras
					// (identical pattern) so both rows per class agree.
					let _ = rep;
					bytes.push(u8::from(b == 2 * c || b == 2 * c + 1));
				}
				labels.push(c as i64);
			}
		}
		(bytes, labels)
	}

	#[test]
	fn multiclass_chunked_scoring_matches_single_pass_and_separates()
	{
		let (bytes, labels) = separable_rows_k3();
		let all = ram_core::packed_bits::PackedBits::from_bool_bytes(&bytes, 6);

		let mut s = dummy_streamer_k3();
		// Train in 2 chunks of 3 rows.
		let first = ram_core::packed_bits::PackedBits::from_bool_bytes(&bytes[..18], 6);
		let second = ram_core::packed_bits::PackedBits::from_bool_bytes(&bytes[18..], 6);
		s.train_chunk(&first, &labels[..3]);
		s.train_chunk(&second, &labels[3..]);
		s.seal_for_scoring();

		// Score pass 1: single chunk. K scores per row, flat.
		s.score_chunk(&all, &labels);
		assert_eq!(s.eval_scored(), 6);
		let single = s.take_scores();
		assert_eq!(single.len(), 6 * 3);

		// Score pass 2: two chunks (4 + 2 rows) — must accumulate identically.
		let head = ram_core::packed_bits::PackedBits::from_bool_bytes(&bytes[..24], 6);
		let tail = ram_core::packed_bits::PackedBits::from_bool_bytes(&bytes[24..], 6);
		s.score_chunk(&head, &labels[..4]);
		s.score_chunk(&tail, &labels[4..]);
		let chunked = s.take_scores();
		for (i, (a, b)) in single.iter().zip(chunked.iter()).enumerate()
		{
			assert!((a - b).abs() < 1e-12, "score {i}: {a} vs {b}");
		}

		// Score pass 3 + finalize: perfectly separable ⇒ acc = macro-F1 = 1.
		s.score_chunk(&all, &labels);
		let (_ce, acc, f1, fpr, _t) = s.finalize_metrics();
		assert_eq!(acc, 1.0, "separable data must decode perfectly");
		assert_eq!(f1, 1.0);
		assert_eq!(fpr, 0.0);
	}

	/// Single-cluster genome: 3 distinct 14-bit base neurons over 24 input
	/// bits; `twin = false` repeats them (0,1,0,2,0,1); `twin = true` gives
	/// copy k of a base its connections ROTATED by k — an address bijection,
	/// so every neuron reads the same cells as its base, but no two are equal.
	fn twin_genome(twin: bool, bits: usize) -> (Vec<usize>, Vec<usize>, Vec<i64>)
	{
		let bases: Vec<Vec<i64>> = (0..3).map(|b| (0..bits).map(|k| ((b * 5 + k * 7) % 24) as i64).collect()).collect();
		let plan = [0usize, 1, 0, 2, 0, 1];
		let mut seen = [0usize; 3];
		let mut conns = Vec::new();
		for &b in &plan
		{
			let mut row = bases[b].clone();
			if twin
			{
				row.rotate_left(seen[b]);
			}
			seen[b] += 1;
			conns.extend(row);
		}
		(vec![bits; plan.len()], vec![plan.len()], conns)
	}

	fn rows(n: usize, seed: u64) -> (PackedBits, Vec<i64>)
	{
		use rand::{Rng, SeedableRng};
		let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
		let bools: Vec<bool> = (0..n * 24).map(|_| rng.gen_bool(0.5)).collect();
		(PackedBits::from_bool_slice(&bools, 24), (0..n).map(|_| rng.gen_range(0..2)).collect())
	}

	fn streamed_scores(twin: bool) -> (Vec<f64>, ram_core::footprint::Footprint)
	{
		let (bits, npc, conns) = twin_genome(twin, 14);
		let mut s = IDSGenomeStreamer::new(bits, npc, conns, 2, 1, true, 0, 0.0, 1.0, 7, None);
		for seed in [1u64, 2]
		{
			let (x, y) = rows(150, seed);
			s.train_chunk(&x, &y);
		}
		s.seal_for_scoring();
		let (x, y) = rows(120, 3);
		s.score_chunk(&x, &y);
		(s.take_scores(), s.footprint().expect("sealed"))
	}

	/// Stage 8: the streamer trains the physical genome and scores through the
	/// logical export — scores bit-identical to the twin genome (same
	/// function, nothing to collapse), physical < logical only when the
	/// multiplicity layer applies.
	#[test]
	fn streamer_multiplicity_matches_rotated_twin()
	{
		let (dup, dup_fp) = streamed_scores(false);
		let (twin, twin_fp) = streamed_scores(true);
		assert!(twin_fp.is_plain() && twin_fp.logical_neurons == 6);
		let applies = ram_core::neuron_memory::multiplicity_applies(1.0, ram_core::neuron_memory::QUAD_WEIGHTED);
		assert_eq!(dup_fp.logical_neurons, 6);
		assert_eq!(dup_fp.physical_neurons, if applies { 3 } else { 6 }, "{dup_fp:?}");
		if applies
		{
			let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
			assert_eq!(bits(&dup), bits(&twin), "multiplicity streamer must score as the plain twin");
			assert!(dup_fp.physical_cells < dup_fp.logical_cells);
		}
	}

	/// Chunking must not change what the streamer learns: 3 chunks vs one
	/// chunk of the same rows, scores bit-identical (rate 1.0: no sampling).
	/// 6-bit neurons so every cell is hit in every chunk. Failed under OI
	/// before the counters spanned all chunks (a later chunk overwrote).
	#[test]
	fn streamer_chunking_is_invisible()
	{
		let (bits, npc, conns) = twin_genome(true, 6);
		let (x, y) = rows(300, 9);
		let score = |chunks: usize| {
			let mut s = IDSGenomeStreamer::new(bits.clone(), npc.clone(), conns.clone(), 2, 1, true, 0, 0.0, 1.0, 7, None);
			let per = 300 / chunks;
			for c in 0..chunks
			{
				let mut part = PackedBits::empty(24);
				(c * per..(c + 1) * per).for_each(|r| part.extend_row_from(&x, r));
				s.train_chunk(&part, &y[c * per..(c + 1) * per]);
			}
			s.seal_for_scoring();
			let (ex, ey) = rows(100, 4);
			s.score_chunk(&ex, &ey);
			s.take_scores().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
		};
		assert_eq!(score(1), score(3));
	}
}

/// For each label, fill in `num_negatives` class indices that are NOT the
/// target. For exhaustive enumeration (`num_negatives >= num_classes - 1`),
/// emits all non-target classes. For fewer, picks the first num_negatives.
/// For single_cluster mode (num_negatives=0), returns an empty Vec.
fn compute_chunk_negatives(labels: &[i64], num_classes: usize, num_negatives: usize) -> Vec<i64>
{
	if num_negatives == 0
	{
		return Vec::new();
	}
	let n = labels.len();
	let mut negatives = vec![0i64; n * num_negatives];
	for (ex, &target) in labels.iter().enumerate()
	{
		let mut k = 0;
		for c in 0..num_classes as i64
		{
			if c != target && k < num_negatives
			{
				negatives[ex * num_negatives + k] = c;
				k += 1;
			}
		}
		// Fill remainder by cycling through non-target classes (matches
		// ids_cache::build_subset behavior when num_negatives > num_classes - 1).
		while k < num_negatives
		{
			for c in 0..num_classes as i64
			{
				if c != target && k < num_negatives
				{
					negatives[ex * num_negatives + k] = c;
					k += 1;
				}
			}
		}
	}
	negatives
}
