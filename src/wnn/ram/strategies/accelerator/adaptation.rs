//! Training-time architecture adaptation: synaptogenesis, neurogenesis, and axonogenesis.
//!
//! These mechanisms run within a single genome evaluation cycle, after initial
//! training and before final evaluation. They adapt the architecture based on
//! observed training statistics.
//!
//! - **Synaptogenesis** (connection level): prune/grow connections per-neuron
//! - **Neurogenesis** (cluster level): add/remove neurons per-cluster
//! - **Axonogenesis** (connection rewiring): replace low-value with high-MI connections
//!
//! ## Design Principles (Literature-Informed)
//!
//! - **Percentile-based pruning** (SET, RigL): prune bottom X% by metric, not absolute
//! - **Relative to expected capacity**: fill thresholds scale with `min(1, examples/2^bits)`
//! - **Competition-gated removal** (JAK2/STAT1): only prune if peers perform better
//! - **Cosine annealing schedule** (RigL): aggressive early, frozen last 25% post-warmup
//! - **Contribution = uniqueness × accuracy** (Fisher Information): combined metric

use rand::prelude::*;
use rayon::prelude::*;
use crate::neuron_memory::{self, ClusterStorage, CELLS_PER_WORD};

/// Per-neuron statistics collected during training.
pub struct NeuronStats {
	pub fill_rate: f32,
	pub error_rate: f32,
	pub connection_entropy: Vec<f32>,
}

/// Per-cluster statistics collected during training.
pub struct ClusterStats {
	pub error_rate: f32,
	pub mean_fill_rate: f32,
	pub neuron_uniqueness: Vec<f32>,
	pub neuron_accuracy: Vec<f32>,
}

/// Configuration for training-time adaptation.
///
/// Uses **relative** thresholds that scale with architecture size, replacing
/// the original absolute thresholds that failed for large architectures.
#[derive(Clone)]
pub struct AdaptationConfig {
	// Synaptogenesis (connection level)
	pub synaptogenesis_enabled: bool,
	/// Prune if entropy < median * ratio (SET: 30% replacement)
	pub prune_entropy_ratio: f32,
	/// Grow if fill > expected_fill * factor (50% of expected)
	pub grow_fill_utilization: f32,
	/// Grow if error > this baseline (below random 0.5, above trained ~0.2)
	pub grow_error_baseline: f32,
	pub min_bits: usize,
	pub max_bits: usize,

	// Neurogenesis (cluster level)
	pub neurogenesis_enabled: bool,
	/// Add neuron if error > 0.5 * factor (high error = underfitting)
	pub cluster_error_factor: f32,
	/// Add if mean_fill > expected * factor
	pub cluster_fill_utilization: f32,
	/// Bottom X% candidates for removal (Lottery Ticket: 80-96% prunable)
	pub neuron_prune_percentile: f32,
	/// Only remove if score < cluster_mean * factor (competition gate)
	pub neuron_removal_factor: f32,
	/// Max neurons = initial * ratio (biology: overproduction 1.5-3x)
	pub max_growth_ratio: f32,
	pub min_neurons: usize,
	pub max_neurons_per_pass: usize,

	// Axonogenesis (connection rewiring)
	pub axonogenesis_enabled: bool,
	/// Rewire if worst entropy < median * ratio (reuses prune_entropy_ratio threshold)
	pub axon_entropy_threshold: f32,
	/// Only rewire if candidate entropy > old * factor (must be this much better)
	pub axon_improvement_factor: f32,
	/// Max connections to rewire per neuron per pass
	pub axon_rewire_count: usize,

	// Schedule (cosine annealing, warmup-excluded)
	pub warmup_generations: usize,
	pub cooldown_iterations: usize,
	/// Last fraction of post-warmup: frozen (RigL: stops at 75%)
	pub stabilize_fraction: f32,
	pub total_generations: usize,

	// Shared
	pub passes_per_eval: usize,
	pub total_input_bits: usize,
	/// Max examples to sample for stats computation (0 = use all).
	pub stats_sample_size: usize,
}

impl Default for AdaptationConfig {
	fn default() -> Self {
		Self {
			synaptogenesis_enabled: false,
			prune_entropy_ratio: 0.3,
			grow_fill_utilization: 0.5,
			grow_error_baseline: 0.35,
			min_bits: 4,
			max_bits: 24,
			neurogenesis_enabled: false,
			axonogenesis_enabled: false,
			axon_entropy_threshold: 0.3,
			axon_improvement_factor: 1.5,
			axon_rewire_count: 2,
			cluster_error_factor: 0.7,
			cluster_fill_utilization: 0.5,
			neuron_prune_percentile: 0.1,
			neuron_removal_factor: 0.5,
			max_growth_ratio: 1.5,
			min_neurons: 3,
			max_neurons_per_pass: 3,
			warmup_generations: 10,
			cooldown_iterations: 5,
			stabilize_fraction: 0.25,
			total_generations: 250,
			passes_per_eval: 1,
			total_input_bits: 64,
			stats_sample_size: 10_000,
		}
	}
}

// =============================================================================
// Cosine Schedule + Helpers
// =============================================================================

/// Warmup-aware cosine annealing schedule for adaptation rate.
///
/// - Before warmup: rate = 0.0
/// - Active window: cosine decay from 1.0 to 0.0
/// - After stabilization: rate = 0.0 (frozen)
///
/// The warmup period does NOT count against the active schedule.
pub fn adaptation_rate(generation: usize, config: &AdaptationConfig) -> f32 {
	let warmup = config.warmup_generations;
	if generation < warmup {
		return 0.0;
	}
	let post_warmup = config.total_generations.saturating_sub(warmup);
	if post_warmup == 0 {
		return 1.0;
	}
	let active_window = ((1.0 - config.stabilize_fraction) * post_warmup as f32) as usize;
	if active_window == 0 {
		return 0.0;
	}
	let active_end = warmup + active_window;
	if generation >= active_end {
		return 0.0;
	}
	let progress = (generation - warmup) as f32 / active_window as f32;
	0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
}

/// Expected fill rate for a neuron with `bits` connection bits given `num_examples` training examples.
///
/// Based on address space math: each example writes at most one address, so
/// expected fill = min(1.0, num_examples / 2^bits).
pub fn expected_fill_rate(bits: usize, num_examples: usize) -> f32 {
	if bits >= 32 {
		return (num_examples as f64 / (1u64 << bits.min(63)) as f64).min(1.0) as f32;
	}
	(num_examples as f32 / (1u32 << bits) as f32).min(1.0)
}

/// Compute median of a float slice.
pub fn median_of(values: &[f32]) -> f32 {
	if values.is_empty() {
		return 0.0;
	}
	let mut sorted = values.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let mid = sorted.len() / 2;
	if sorted.len() % 2 == 0 {
		(sorted[mid - 1] + sorted[mid]) / 2.0
	} else {
		sorted[mid]
	}
}

/// Compute percentile (0-100) of a float slice.
pub fn percentile_of(values: &[f32], pct: f32) -> f32 {
	if values.is_empty() {
		return 0.0;
	}
	let mut sorted = values.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let idx = ((pct / 100.0) * (sorted.len() - 1) as f32).round() as usize;
	sorted[idx.min(sorted.len() - 1)]
}

/// Compute mean of a float slice.
pub fn mean_of(values: &[f32]) -> f32 {
	if values.is_empty() {
		return 0.0;
	}
	values.iter().sum::<f32>() / values.len() as f32
}

// =============================================================================
// Sampling + Bit Stats
// =============================================================================

/// Build a deterministic sample of example indices.
/// If `sample_size == 0` or `sample_size >= num_examples`, returns all indices.
fn build_sample_indices(num_examples: usize, sample_size: usize, seed: u64) -> Vec<usize> {
	if sample_size == 0 || sample_size >= num_examples {
		return (0..num_examples).collect();
	}
	let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
	let mut indices: Vec<usize> = (0..num_examples).collect();
	// Fisher-Yates partial shuffle — only shuffle the first sample_size elements
	for i in 0..sample_size {
		let j = rng.gen_range(i..num_examples);
		indices.swap(i, j);
	}
	indices.truncate(sample_size);
	indices
}

/// Pre-compute ones-count for each input bit across sampled examples.
///
/// Returns `bit_ones[bit_idx]` = number of sampled examples where that bit is 1.
/// This avoids redundant counting when multiple neurons share connections.
fn precompute_bit_ones(
	packed_input: &[u64],
	words_per_example: usize,
	sample_indices: &[usize],
	total_input_bits: usize,
) -> Vec<u32> {
	let mut bit_ones = vec![0u32; total_input_bits];
	for &ex in sample_indices {
		let row_start = ex * words_per_example;
		for (word_i, &word) in packed_input[row_start..row_start + words_per_example].iter().enumerate() {
			if word == 0 { continue; }
			let base_bit = word_i * 64;
			let mut w = word;
			while w != 0 {
				let tz = w.trailing_zeros() as usize;
				let bit_idx = base_bit + tz;
				if bit_idx < total_input_bits {
					bit_ones[bit_idx] += 1;
				}
				w &= w - 1; // clear lowest set bit
			}
		}
	}
	bit_ones
}

// =============================================================================
// Stats Computation (CPU)
// =============================================================================

/// Compute per-neuron statistics from training data and memory.
///
/// Uses the **bitwise multi-label** model: each cluster is an independent binary classifier.
/// `target_bits` is `[num_examples * num_clusters]` where each value is 0 or 1.
///
/// For each neuron, computes:
/// - fill_rate: fraction of address space with non-empty cells
/// - error_rate: fraction of training examples where neuron voted incorrectly
/// - connection_entropy: Shannon entropy of each input bit across training examples
///
/// Parallelized with rayon across neurons; bit counts pre-computed and shared.
pub fn compute_neuron_stats(
	bits_per_neuron: &[usize],
	neurons_per_cluster: &[usize],
	connections: &[i64],
	storages: &[ClusterStorage],
	packed_input: &[u64],
	words_per_example: usize,
	target_bits: &[u8],
	num_examples: usize,
	num_clusters: usize,
	config: &AdaptationConfig,
) -> Vec<NeuronStats> {
	let total_neurons: usize = neurons_per_cluster.iter().sum();

	// Sample examples for stats — full scan is O(neurons*bits*examples) which is
	// prohibitively slow (e.g. 200 neurons × 18 bits × 600K examples = 2B ops).
	let sample_indices = build_sample_indices(num_examples, config.stats_sample_size, 42);
	let n_sample = sample_indices.len();

	// Pre-compute bit ones counts — shared across all neurons
	let bit_ones = precompute_bit_ones(packed_input, words_per_example, &sample_indices, config.total_input_bits);

	// Build connection offsets
	let mut conn_offsets = vec![0usize];
	for &b in bits_per_neuron {
		conn_offsets.push(conn_offsets.last().unwrap() + b);
	}

	// Build flat (global_n, cluster, local_n) tuples for parallel iteration
	let mut neuron_info: Vec<(usize, usize, usize)> = Vec::with_capacity(total_neurons);
	let mut global_n = 0;
	for cluster in 0..num_clusters {
		for local_n in 0..neurons_per_cluster[cluster] {
			neuron_info.push((global_n, cluster, local_n));
			global_n += 1;
		}
	}

	// Parallel per-neuron stats computation
	neuron_info.par_iter().map(|&(global_n, cluster, local_n)| {
		let n_bits = bits_per_neuron[global_n];
		let conn_start = conn_offsets[global_n];
		let neuron_conns = &connections[conn_start..conn_start + n_bits];

		// Connection entropy from pre-computed bit counts
		let conn_entropy: Vec<f32> = neuron_conns.iter().map(|&conn_idx| {
			let ones = bit_ones[conn_idx as usize];
			let p = ones as f32 / n_sample.max(1) as f32;
			if p > 0.0 && p < 1.0 {
				-(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
			} else {
				0.0
			}
		}).collect();

		// Fill rate from storage (reads from memory, not examples)
		let fill_rate = storage_fill_rate(&storages[cluster], local_n, n_bits);

		// Error rate: multi-label — every example has a target for this cluster
		let mut errors = 0u32;
		for &ex in &sample_indices {
			let target = target_bits[ex * num_clusters + cluster];
			let mut addr = 0u32;
			for (bit_i, &conn_idx) in neuron_conns.iter().enumerate() {
				let idx = conn_idx as usize;
				let word_idx = ex * words_per_example + idx / 64;
				let bit_idx = idx % 64;
				if packed_input[word_idx] >> bit_idx & 1 == 1 {
					addr |= 1 << (n_bits - 1 - bit_i);
				}
			}
			let vote = storage_read(&storages[cluster], local_n, addr);
			let predicted_true = vote >= 0.5;
			let target_true = target != 0;
			if predicted_true != target_true {
				errors += 1;
			}
		}
		let error_rate = if n_sample > 0 { errors as f32 / n_sample as f32 } else { 0.0 };

		NeuronStats { fill_rate, error_rate, connection_entropy: conn_entropy }
	}).collect()
}

/// Compute per-cluster statistics from neuron predictions.
///
/// Uses **bitwise multi-label** model: `target_bits` is `[num_examples * num_clusters]`.
/// For each cluster, ALL examples have a target (0 or 1).
///
/// Parallelized with rayon across clusters.
pub fn compute_cluster_stats(
	neuron_stats: &[NeuronStats],
	neurons_per_cluster: &[usize],
	bits_per_neuron: &[usize],
	connections: &[i64],
	storages: &[ClusterStorage],
	packed_input: &[u64],
	words_per_example: usize,
	target_bits: &[u8],
	num_examples: usize,
	num_clusters: usize,
	config: &AdaptationConfig,
) -> Vec<ClusterStats> {
	let mut conn_offsets = vec![0usize];
	for &b in bits_per_neuron {
		conn_offsets.push(conn_offsets.last().unwrap() + b);
	}

	let mut neuron_offsets = vec![0usize];
	for &n in neurons_per_cluster {
		neuron_offsets.push(neuron_offsets.last().unwrap() + n);
	}

	// Sample examples
	let sample_indices = build_sample_indices(num_examples, config.stats_sample_size, 43);
	let n_sample = sample_indices.len();

	(0..num_clusters).into_par_iter().map(|cluster| {
		let n_neurons = neurons_per_cluster[cluster];
		let neuron_base = neuron_offsets[cluster];

		// Mean fill rate
		let mean_fill = if n_neurons > 0 {
			(0..n_neurons).map(|i| neuron_stats[neuron_base + i].fill_rate).sum::<f32>() / n_neurons as f32
		} else {
			0.0
		};

		// Multi-label: compute votes for sampled examples
		let mut cluster_errors = 0u32;
		let mut neuron_votes: Vec<Vec<bool>> = vec![Vec::with_capacity(n_sample); n_neurons];

		for &ex in &sample_indices {
			let target = target_bits[ex * num_clusters + cluster];
			let target_true = target != 0;
			let mut votes_true = 0u32;

			for local_n in 0..n_neurons {
				let global_n = neuron_base + local_n;
				let n_bits = bits_per_neuron[global_n];
				let conn_start = conn_offsets[global_n];
				let neuron_conns = &connections[conn_start..conn_start + n_bits];

				let mut addr = 0u32;
				for (bit_i, &conn_idx) in neuron_conns.iter().enumerate() {
					let idx = conn_idx as usize;
					let word_idx = ex * words_per_example + idx / 64;
					let bit_idx = idx % 64;
					if packed_input[word_idx] >> bit_idx & 1 == 1 {
						addr |= 1 << (n_bits - 1 - bit_i);
					}
				}
				let vote = storage_read(&storages[cluster], local_n, addr);
				let is_true = vote >= 0.5;
				if is_true { votes_true += 1; }
				neuron_votes[local_n].push(is_true);
			}

			// Majority vote for this cluster
			let majority_true = votes_true > (n_neurons as u32 / 2);
			if majority_true != target_true { cluster_errors += 1; }
		}

		let cluster_error_rate = if n_sample > 0 {
			cluster_errors as f32 / n_sample as f32
		} else {
			0.0
		};

		// Precompute majority vote per sample
		let majority_votes: Vec<bool> = (0..n_sample).map(|s| {
			let votes_true: u32 = (0..n_neurons).map(|j| neuron_votes[j][s] as u32).sum();
			votes_true > (n_neurons as u32 / 2)
		}).collect();

		// Neuron uniqueness + accuracy
		let mut uniqueness = vec![0.0f32; n_neurons];
		let mut accuracy = vec![0.0f32; n_neurons];

		if n_sample > 0 {
			for local_n in 0..n_neurons {
				let mut disagree = 0u32;
				let mut correct = 0u32;
				for s in 0..n_sample {
					let ex = sample_indices[s];
					let my_vote = neuron_votes[local_n][s];
					let target_true = target_bits[ex * num_clusters + cluster] != 0;

					if my_vote != majority_votes[s] { disagree += 1; }
					if my_vote == target_true { correct += 1; }
				}
				uniqueness[local_n] = disagree as f32 / n_sample as f32;
				accuracy[local_n] = correct as f32 / n_sample as f32;
			}
		}

		ClusterStats {
			error_rate: cluster_error_rate,
			mean_fill_rate: mean_fill,
			neuron_uniqueness: uniqueness,
			neuron_accuracy: accuracy,
		}
	}).collect()
}

// =============================================================================
// Synaptogenesis (Connection-Level Adaptation)
// =============================================================================

/// Synaptogenesis pass: prune low-entropy connections, grow where underfitting.
///
/// Uses **relative** thresholds:
/// - Prune: entropy < median_entropy * prune_entropy_ratio (percentile-based)
/// - Grow: fill > expected_fill * grow_fill_utilization AND error > grow_error_baseline
///
/// Stochastic: each action gated by `rng.gen::<f32>() < rate` (cosine schedule).
///
/// Returns (pruned_count, grown_count).
pub fn synaptogenesis_pass(
	bits_per_neuron: &mut Vec<usize>,
	connections: &mut Vec<i64>,
	neuron_stats: &[NeuronStats],
	config: &AdaptationConfig,
	packed_input: &[u64],
	words_per_example: usize,
	num_examples: usize,
	rate: f32,
	rng: &mut impl Rng,
) -> (usize, usize) {
	let total_neurons = bits_per_neuron.len();
	let mut pruned = 0usize;
	let mut grown = 0usize;

	// Build connection offsets
	let mut conn_offsets = vec![0usize];
	for &b in bits_per_neuron.iter() {
		conn_offsets.push(conn_offsets.last().unwrap() + b);
	}

	// Pre-compute bit entropy for all input bits (used by grow path)
	let sample_indices = build_sample_indices(num_examples, config.stats_sample_size, 44);
	let bit_ones = precompute_bit_ones(packed_input, words_per_example, &sample_indices, config.total_input_bits);
	let n_sample = sample_indices.len();
	let bit_entropy: Vec<f32> = bit_ones.iter().map(|&ones| {
		let p = ones as f32 / n_sample.max(1) as f32;
		if p > 0.0 && p < 1.0 {
			-(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
		} else {
			0.0
		}
	}).collect();

	// Process each neuron
	let mut new_bits = bits_per_neuron.clone();
	let mut modifications: Vec<(usize, Modification)> = Vec::new();

	for n in 0..total_neurons {
		let n_bits = bits_per_neuron[n];
		let stats = &neuron_stats[n];

		// Prune: remove lowest-entropy connection if below relative threshold
		if n_bits > config.min_bits {
			let median_ent = median_of(&stats.connection_entropy);
			let prune_threshold = median_ent * config.prune_entropy_ratio;

			if let Some((min_idx, &min_entropy)) = stats.connection_entropy.iter()
				.enumerate()
				.min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
			{
				if min_entropy < prune_threshold && rng.gen::<f32>() < rate {
					new_bits[n] = n_bits - 1;
					modifications.push((n, Modification::Prune(min_idx)));
					pruned += 1;
					continue; // Don't both prune and grow same neuron
				}
			}
		}

		// Grow: add highest-entropy unconnected bit if underfitting
		let exp_fill = expected_fill_rate(n_bits, num_examples);
		if n_bits < config.max_bits
			&& stats.fill_rate > exp_fill * config.grow_fill_utilization
			&& stats.error_rate > config.grow_error_baseline
			&& rng.gen::<f32>() < rate
		{
			// Find highest-entropy unconnected input bit
			let conn_start = conn_offsets[n];
			let used: std::collections::HashSet<i64> = connections[conn_start..conn_start + n_bits]
				.iter().copied().collect();

			let mut best_bit: Option<(i64, f32)> = None;
			for bit_idx in 0..config.total_input_bits {
				if used.contains(&(bit_idx as i64)) { continue; }
				let h = bit_entropy[bit_idx];
				if best_bit.is_none() || h > best_bit.unwrap().1 {
					best_bit = Some((bit_idx as i64, h));
				}
			}

			if let Some((new_conn, _)) = best_bit {
				new_bits[n] = n_bits + 1;
				modifications.push((n, Modification::Grow(new_conn)));
				grown += 1;
			}
		}
	}

	// Apply modifications — rebuild connections
	if !modifications.is_empty() {
		let mut new_connections = Vec::with_capacity(new_bits.iter().sum());
		let mut old_conn_offsets = vec![0usize];
		for &b in bits_per_neuron.iter() {
			old_conn_offsets.push(old_conn_offsets.last().unwrap() + b);
		}

		let mods: std::collections::HashMap<usize, &Modification> = modifications.iter()
			.map(|(n, m)| (*n, m)).collect();

		for n in 0..total_neurons {
			let old_start = old_conn_offsets[n];
			let old_bits = bits_per_neuron[n];

			match mods.get(&n) {
				Some(Modification::Prune(remove_idx)) => {
					for i in 0..old_bits {
						if i != *remove_idx {
							new_connections.push(connections[old_start + i]);
						}
					}
				}
				Some(Modification::Grow(new_conn)) => {
					new_connections.extend_from_slice(&connections[old_start..old_start + old_bits]);
					new_connections.push(*new_conn);
				}
				None => {
					new_connections.extend_from_slice(&connections[old_start..old_start + old_bits]);
				}
			}
		}

		*bits_per_neuron = new_bits;
		*connections = new_connections;
	}

	if pruned > 0 || grown > 0 {
		eprintln!("[Adapt] Synaptogenesis: pruned={}, grown={} (rate={:.2})", pruned, grown, rate);
	}

	(pruned, grown)
}

enum Modification {
	Prune(usize),  // index of connection to remove
	Grow(i64),     // new connection index to add
}

// =============================================================================
// Axonogenesis (MI-Guided Connection Rewiring)
// =============================================================================

/// Axonogenesis pass: rewire low-value connections to high-information input bits.
///
/// Unlike synaptogenesis (which changes the number of connections), axonogenesis
/// keeps the same bit count but REPLACES which input bits each neuron observes.
///
/// Hybrid MI algorithm (3 stages):
///
/// Stage 1 — Entropy filter (FREE, uses pre-computed bit_ones):
///   Identify weak connections (entropy < median * threshold) and filter candidate
///   replacement bits by marginal entropy (skip near-constant bits). Eliminates ~90%.
///
/// Stage 2 — Accuracy delta (moderate, uses trained memory):
///   For each weak connection × filtered candidate, compute per-neuron accuracy change
///   by address flipping: swap one bit in the address, look up both old and new memory
///   values, count net correct predictions. Only O(K × M × N_samples) memory lookups.
///
/// Stage 3 — Redundancy penalty (cheap, bit co-occurrence):
///   Penalize candidates whose values are highly correlated with an existing connection.
///   Uses Jaccard similarity on sampled bit values. Prevents connecting to redundant info.
///
/// Returns rewired_count.
pub fn axonogenesis_pass(
	bits_per_neuron: &[usize],
	neurons_per_cluster: &[usize],
	connections: &mut Vec<i64>,
	neuron_stats: &[NeuronStats],
	storages: &[ClusterStorage],
	config: &AdaptationConfig,
	packed_input: &[u64],
	words_per_example: usize,
	target_bits: &[u8],
	num_examples: usize,
	num_clusters: usize,
	rate: f32,
	rng: &mut impl Rng,
) -> usize {
	let total_neurons = bits_per_neuron.len();
	let mut rewired = 0usize;
	let max_candidates = 20usize;  // Top-M candidates by entropy per weak connection

	// Build connection offsets (per-neuron)
	let mut conn_offsets = vec![0usize];
	for &b in bits_per_neuron.iter() {
		conn_offsets.push(conn_offsets.last().unwrap() + b);
	}

	// Build neuron-to-cluster mapping
	let mut neuron_cluster = Vec::with_capacity(total_neurons);
	let mut neuron_local_idx = Vec::with_capacity(total_neurons);
	for (c, &nc) in neurons_per_cluster.iter().enumerate() {
		for local in 0..nc {
			neuron_cluster.push(c);
			neuron_local_idx.push(local);
		}
	}

	// Stage 1: Pre-compute marginal entropy for all input bits
	let sample_indices = build_sample_indices(num_examples, config.stats_sample_size, 77);
	let n_sample = sample_indices.len();
	let bit_ones = precompute_bit_ones(packed_input, words_per_example, &sample_indices, config.total_input_bits);
	let bit_entropy: Vec<f32> = bit_ones.iter().map(|&ones| {
		let p = ones as f32 / n_sample.max(1) as f32;
		if p > 0.0 && p < 1.0 {
			-(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
		} else {
			0.0
		}
	}).collect();

	// Minimum entropy for a candidate to be considered (skip near-constant bits)
	let entropy_floor = 0.1f32;

	// Pre-compute bit values for sampled examples (for redundancy computation).
	// bit_vals[bit_idx][sample_i] = 0 or 1. Only computed for bits above entropy floor.
	// We store as packed bytes for memory efficiency.
	let mut bit_vals: Vec<Option<Vec<u8>>> = vec![None; config.total_input_bits];
	for bit_idx in 0..config.total_input_bits {
		if bit_entropy[bit_idx] >= entropy_floor {
			let mut vals = Vec::with_capacity(n_sample);
			for &ex in &sample_indices {
				let row_start = ex * words_per_example;
				let word_idx = bit_idx / 64;
				let bit_pos = bit_idx % 64;
				let bit = ((packed_input[row_start + word_idx] >> bit_pos) & 1) as u8;
				vals.push(bit);
			}
			bit_vals[bit_idx] = Some(vals);
		}
	}

	// Process each neuron
	for n in 0..total_neurons {
		let n_bits = bits_per_neuron[n];
		if n_bits < 2 { continue; }

		let stats = &neuron_stats[n];
		let conn_start = conn_offsets[n];
		let cluster = neuron_cluster[n];
		let local_n = neuron_local_idx[n];

		// Stage 1a: Find weak connections (entropy < median * threshold)
		let median_ent = median_of(&stats.connection_entropy);
		let rewire_threshold = median_ent * config.axon_entropy_threshold;

		let mut weak_conns: Vec<(usize, f32)> = stats.connection_entropy.iter()
			.enumerate()
			.filter(|(_, &ent)| ent < rewire_threshold)
			.map(|(idx, &ent)| (idx, ent))
			.collect();
		weak_conns.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
		weak_conns.truncate(config.axon_rewire_count);

		if weak_conns.is_empty() { continue; }

		// Stage 1b: Get candidate unused bits sorted by marginal entropy (descending)
		let used: std::collections::HashSet<i64> = connections[conn_start..conn_start + n_bits]
			.iter().copied().collect();

		let mut candidates: Vec<(i64, f32)> = (0..config.total_input_bits)
			.filter(|&b| !used.contains(&(b as i64)) && bit_entropy[b] >= entropy_floor)
			.map(|b| (b as i64, bit_entropy[b]))
			.collect();
		candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
		candidates.truncate(max_candidates);

		if candidates.is_empty() { continue; }

		// Pre-compute base addresses for this neuron on sampled examples
		let neuron_conns = &connections[conn_start..conn_start + n_bits];
		let mut base_addresses: Vec<u32> = Vec::with_capacity(n_sample);
		for &ex in &sample_indices {
			let row_start = ex * words_per_example;
			let mut addr = 0u32;
			for (i, &conn_idx) in neuron_conns.iter().enumerate() {
				let idx = conn_idx as usize;
				let bit = (packed_input[row_start + idx / 64] >> (idx % 64)) & 1;
				addr |= (bit as u32) << (n_bits - 1 - i);
			}
			base_addresses.push(addr);
		}

		// Stage 3 prep: Pre-compute bit values of existing connections (for redundancy)
		let existing_bit_vals: Vec<Option<&Vec<u8>>> = (0..n_bits)
			.map(|i| {
				let conn = neuron_conns[i] as usize;
				if conn < config.total_input_bits { bit_vals[conn].as_ref() } else { None }
			})
			.collect();

		// For each weak connection, find the best candidate using accuracy delta + redundancy
		for &(local_idx, _old_entropy) in &weak_conns {
			if rng.gen::<f32>() >= rate { continue; }

			let old_conn = connections[conn_start + local_idx];
			let mut best_candidate: Option<(i64, f32)> = None;  // (conn, adjusted_score)

			for &(cand_conn, _cand_entropy) in &candidates {
				// Skip if already used by this neuron (may have been rewired earlier in this loop)
				if connections[conn_start..conn_start + n_bits].contains(&cand_conn) { continue; }

				// Stage 2: Accuracy delta — measure per-neuron accuracy change from this swap
				let mut delta = 0i32;
				for (si, &ex) in sample_indices.iter().enumerate() {
					let old_addr = base_addresses[si];
					let target = target_bits[ex * num_clusters + cluster];

					// Old prediction
					let old_val = storage_read(&storages[cluster], local_n, old_addr);
					let old_correct = (old_val >= 0.5) == (target > 0);

					// Flip the address bit at local_idx if old/new input bits differ
					let row_start = ex * words_per_example;
					let old_bit = (packed_input[row_start + old_conn as usize / 64]
						>> (old_conn as usize % 64)) & 1;
					let new_bit = (packed_input[row_start + cand_conn as usize / 64]
						>> (cand_conn as usize % 64)) & 1;
					let new_addr = if old_bit != new_bit {
						old_addr ^ (1u32 << (n_bits - 1 - local_idx))
					} else {
						old_addr
					};

					// New prediction
					let new_val = storage_read(&storages[cluster], local_n, new_addr);
					let new_correct = (new_val >= 0.5) == (target > 0);

					delta += new_correct as i32 - old_correct as i32;
				}

				if delta <= 0 { continue; }  // No improvement — skip

				// Stage 3: Redundancy penalty — penalize high correlation with existing connections
				let mut max_jaccard = 0.0f32;
				if let Some(cand_vals) = &bit_vals[cand_conn as usize] {
					for (i, existing) in existing_bit_vals.iter().enumerate() {
						if i == local_idx { continue; }  // Skip the connection being replaced
						if let Some(existing_vals) = existing {
							let mut both = 0u32;
							let mut either = 0u32;
							for si in 0..n_sample {
								let a = cand_vals[si];
								let b = existing_vals[si];
								both += (a & b) as u32;
								either += (a | b) as u32;
							}
							let jaccard = if either > 0 { both as f32 / either as f32 } else { 0.0 };
							if jaccard > max_jaccard { max_jaccard = jaccard; }
						}
					}
				}

				// Adjusted score: accuracy delta scaled down by redundancy
				// redundancy_weight 0.5 means a Jaccard=1.0 (identical) halves the score
				let redundancy_weight = 0.5f32;
				let adjusted_score = delta as f32 * (1.0 - redundancy_weight * max_jaccard);

				if best_candidate.is_none() || adjusted_score > best_candidate.unwrap().1 {
					best_candidate = Some((cand_conn, adjusted_score));
				}
			}

			// Rewire if we found a positive-scoring candidate
			if let Some((new_conn, score)) = best_candidate {
				if score > 0.0 {
					connections[conn_start + local_idx] = new_conn;
					rewired += 1;
				}
			}
		}
	}

	if rewired > 0 {
		eprintln!("[Adapt] Axonogenesis: rewired={} (rate={:.2})", rewired, rate);
	}

	rewired
}

// =============================================================================
// Neurogenesis (Cluster-Level Adaptation)
// =============================================================================

/// Neurogenesis pass: add/remove neurons from clusters.
///
/// Uses **relative** thresholds + competition gate:
/// - Add: error > 0.5 * cluster_error_factor AND fill > expected * cluster_fill_utilization
/// - Remove: score in bottom neuron_prune_percentile AND score < cluster_mean * neuron_removal_factor
/// - Growth capped at initial_neurons * max_growth_ratio
///
/// Stochastic: each action gated by `rng.gen::<f32>() < rate` (cosine schedule).
///
/// Returns (added_count, removed_count).
pub fn neurogenesis_pass(
	bits_per_neuron: &mut Vec<usize>,
	neurons_per_cluster: &mut Vec<usize>,
	connections: &mut Vec<i64>,
	cluster_stats: &[ClusterStats],
	config: &AdaptationConfig,
	generation: usize,
	cooldowns: &mut Vec<usize>,
	initial_neurons: &[usize],
	num_examples: usize,
	rate: f32,
	rng: &mut impl Rng,
) -> (usize, usize) {
	if generation < config.warmup_generations {
		return (0, 0);
	}

	let num_clusters = neurons_per_cluster.len();
	let mut added = 0usize;
	let mut removed = 0usize;

	// Build neuron offsets
	let mut neuron_offsets = vec![0usize];
	for &n in neurons_per_cluster.iter() {
		neuron_offsets.push(neuron_offsets.last().unwrap() + n);
	}

	// Build connection offsets
	let mut conn_offsets = vec![0usize];
	for &b in bits_per_neuron.iter() {
		conn_offsets.push(conn_offsets.last().unwrap() + b);
	}

	// Ensure cooldowns vector is sized correctly
	if cooldowns.len() < num_clusters {
		cooldowns.resize(num_clusters, 0);
	}

	// Collect modifications first, apply later
	struct ClusterMod {
		cluster: usize,
		action: ClusterAction,
	}
	enum ClusterAction {
		AddNeuron { template_local: usize },
		RemoveNeuron { local_idx: usize },
	}

	let mut mods: Vec<ClusterMod> = Vec::new();

	for cluster in 0..num_clusters {
		// Check cooldown
		if cooldowns[cluster] > 0 {
			cooldowns[cluster] -= 1;
			continue;
		}

		let stats = &cluster_stats[cluster];
		let n_neurons = neurons_per_cluster[cluster];

		// Compute expected fill for this cluster's neurons
		let neuron_base = neuron_offsets[cluster];
		let avg_bits = if n_neurons > 0 {
			(0..n_neurons).map(|i| bits_per_neuron[neuron_base + i]).sum::<usize>() / n_neurons
		} else {
			0
		};
		let exp_fill = expected_fill_rate(avg_bits, num_examples);

		// Growth cap: initial * max_growth_ratio
		let max_neurons = (initial_neurons.get(cluster).copied().unwrap_or(n_neurons) as f32
			* config.max_growth_ratio) as usize;

		// Growth: high error + high fill + room to grow
		if stats.error_rate > 0.5 * config.cluster_error_factor
			&& stats.mean_fill_rate > exp_fill * config.cluster_fill_utilization
			&& n_neurons < max_neurons
			&& rng.gen::<f32>() < rate
		{
			// Find best-performing neuron to clone
			let best_local = stats.neuron_accuracy.iter()
				.enumerate()
				.max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
				.map(|(i, _)| i)
				.unwrap_or(0);

			let neurons_to_add = 1.min(max_neurons - n_neurons)
				.min(config.max_neurons_per_pass);
			for _ in 0..neurons_to_add {
				mods.push(ClusterMod {
					cluster,
					action: ClusterAction::AddNeuron { template_local: best_local },
				});
				added += 1;
			}
			cooldowns[cluster] = config.cooldown_iterations;
			continue;
		}

		// Pruning: remove neurons with low contribution (percentile + competition gate)
		if n_neurons > config.min_neurons && rng.gen::<f32>() < rate {
			// Score = uniqueness × accuracy
			let scores: Vec<f32> = (0..n_neurons)
				.map(|i| stats.neuron_uniqueness[i] * stats.neuron_accuracy[i])
				.collect();
			let cluster_mean = mean_of(&scores);
			let threshold = percentile_of(&scores, config.neuron_prune_percentile * 100.0);

			// Find worst neuron below both thresholds (competition gate)
			let mut worst_local = None;
			let mut worst_score = f32::MAX;
			for i in 0..n_neurons {
				if scores[i] <= threshold
					&& scores[i] < cluster_mean * config.neuron_removal_factor
					&& scores[i] < worst_score
				{
					worst_score = scores[i];
					worst_local = Some(i);
				}
			}

			if let Some(local_idx) = worst_local {
				mods.push(ClusterMod {
					cluster,
					action: ClusterAction::RemoveNeuron { local_idx },
				});
				removed += 1;
				cooldowns[cluster] = config.cooldown_iterations;
			}
		}
	}

	// Apply modifications — rebuild all arrays
	if !mods.is_empty() {
		let old_bits = bits_per_neuron.clone();
		let old_neurons = neurons_per_cluster.clone();
		let old_connections = connections.clone();

		let mut new_bits_per_neuron = Vec::new();
		let mut new_neurons = neurons_per_cluster.clone();
		let mut new_connections = Vec::new();

		// Group mods by cluster
		let mut cluster_adds: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
		let mut cluster_removes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
		for m in &mods {
			match &m.action {
				ClusterAction::AddNeuron { template_local } => {
					cluster_adds.entry(m.cluster).or_default().push(*template_local);
				}
				ClusterAction::RemoveNeuron { local_idx } => {
					cluster_removes.entry(m.cluster).or_default().push(*local_idx);
				}
			}
		}

		for cluster in 0..num_clusters {
			let n_neurons = old_neurons[cluster];
			let neuron_base = neuron_offsets[cluster];
			let removes = cluster_removes.get(&cluster);
			let adds = cluster_adds.get(&cluster);

			// Copy existing neurons (skip removed ones)
			let mut kept = 0;
			for local_n in 0..n_neurons {
				if let Some(remove_list) = removes {
					if remove_list.contains(&local_n) {
						continue;
					}
				}
				let global_n = neuron_base + local_n;
				new_bits_per_neuron.push(old_bits[global_n]);
				let conn_start = conn_offsets[global_n];
				let conn_end = conn_start + old_bits[global_n];
				new_connections.extend_from_slice(&old_connections[conn_start..conn_end]);
				kept += 1;
			}

			// Add new neurons (cloned from template + mutated)
			if let Some(add_list) = adds {
				for &template_local in add_list {
					let template_global = neuron_base + template_local;
					let template_bits = old_bits[template_global];
					let template_conn_start = conn_offsets[template_global];

					// Clone bits with small mutation
					let new_n_bits = template_bits;
					new_bits_per_neuron.push(new_n_bits);

					// Clone connections with perturbation
					for i in 0..new_n_bits {
						let old_conn = old_connections[template_conn_start + i];
						let delta = *[-2i64, -1, 0, 0, 0, 1, 2].choose(rng).unwrap();
						let new_conn = (old_conn + delta).max(0).min(config.total_input_bits as i64 - 1);
						new_connections.push(new_conn);
					}
					kept += 1;
				}
			}

			new_neurons[cluster] = kept;
		}

		*bits_per_neuron = new_bits_per_neuron;
		*neurons_per_cluster = new_neurons;
		*connections = new_connections;
	}

	if added > 0 || removed > 0 {
		eprintln!("[Adapt] Neurogenesis: added={}, removed={} (rate={:.2})", added, removed, rate);
	}

	(added, removed)
}

// =============================================================================
// Storage Helpers
// =============================================================================

/// Read a cell value from ClusterStorage as a float vote.
/// Bitwise path uses quad mode: [0.0, 0.25, 0.75, 1.0] for raw values 0-3.
fn storage_read(storage: &ClusterStorage, neuron_idx: usize, address: u32) -> f32 {
	let raw = storage.read_cell(neuron_idx, address as usize);
	neuron_memory::QUAD_WEIGHTS[raw.clamp(0, 3) as usize]
}

/// Compute fill rate for a neuron in its storage.
fn storage_fill_rate(storage: &ClusterStorage, neuron_idx: usize, bits: usize) -> f32 {
	let total_cells = 1usize << bits;
	let filled = match storage {
		ClusterStorage::Dense { words, words_per_neuron, empty_word, .. } => {
			let wpn = *words_per_neuron;
			let word_offset = neuron_idx * wpn;
			// Extract the per-cell empty value from the empty_word
			let empty_cell = *empty_word & 0x3;
			let mut count = 0u32;
			for w in 0..wpn {
				let word = words[word_offset + w];
				for c in 0..CELLS_PER_WORD {
					let cell = (word >> (c * 2)) & 0x3;
					if cell != empty_cell { count += 1; }
				}
			}
			count as usize
		}
		ClusterStorage::Sparse { neurons: maps, .. } => {
			if neuron_idx >= maps.len() { 0 } else { maps[neuron_idx].len() }
		}
	};
	filled.min(total_cells) as f32 / total_cells.max(1) as f32
}

// =============================================================================
// Combined Stats (CPU fallback for non-macOS or when no GPU is available)
// =============================================================================

/// Compute combined neuron + cluster stats using CPU.
/// On macOS, prefer `compute_neuron_and_cluster_stats_gpu()` for ~100x speedup.
pub fn compute_combined_stats(
	bits_per_neuron: &[usize],
	neurons_per_cluster: &[usize],
	connections: &[i64],
	storages: &[ClusterStorage],
	packed_input: &[u64],
	words_per_example: usize,
	target_bits: &[u8],
	num_examples: usize,
	num_clusters: usize,
	config: &AdaptationConfig,
) -> (Vec<NeuronStats>, Vec<ClusterStats>) {
	let neuron_stats = compute_neuron_stats(
		bits_per_neuron, neurons_per_cluster, connections,
		storages, packed_input, words_per_example, target_bits,
		num_examples, num_clusters, config,
	);
	let cluster_stats = compute_cluster_stats(
		&neuron_stats, neurons_per_cluster, bits_per_neuron,
		connections, storages, packed_input, words_per_example,
		target_bits, num_examples, num_clusters, config,
	);
	(neuron_stats, cluster_stats)
}

// =============================================================================
// GPU Stats
// =============================================================================

/// Compute neuron + cluster stats on GPU via Metal.
///
/// Returns `(Vec<NeuronStats>, Vec<ClusterStats>)` matching the CPU versions.
/// Connection entropy is still computed on CPU (trivially fast from precomputed bit_ones).
#[cfg(target_os = "macos")]
pub fn compute_neuron_and_cluster_stats_gpu(
	metal: &crate::metal_stats::MetalStatsComputer,
	bits_per_neuron: &[usize],
	neurons_per_cluster: &[usize],
	connections: &[i64],
	storages: &[ClusterStorage],
	packed_input: &[u64],
	words_per_example: usize,
	target_bits: &[u8],
	num_examples: usize,
	num_clusters: usize,
	config: &AdaptationConfig,
	memory_mode: u8,
) -> (Vec<NeuronStats>, Vec<ClusterStats>) {
	let total_neurons: usize = neurons_per_cluster.iter().sum();

	// Build sample indices (same seed as CPU path for consistency)
	let sample_indices_usize = build_sample_indices(num_examples, config.stats_sample_size, 42);
	let n_sample = sample_indices_usize.len();
	let sample_indices_u32: Vec<u32> = sample_indices_usize.iter().map(|&i| i as u32).collect();

	// Connection entropy on CPU (cheap — just mapping precomputed bit_ones)
	let bit_ones = precompute_bit_ones(packed_input, words_per_example, &sample_indices_usize, config.total_input_bits);

	let mut conn_offsets = vec![0usize];
	for &b in bits_per_neuron {
		conn_offsets.push(conn_offsets.last().unwrap() + b);
	}

	let conn_entropies: Vec<Vec<f32>> = (0..total_neurons).map(|n| {
		let n_bits = bits_per_neuron[n];
		let conn_start = conn_offsets[n];
		connections[conn_start..conn_start + n_bits].iter().map(|&conn_idx| {
			let ones = bit_ones[conn_idx as usize];
			let p = ones as f32 / n_sample.max(1) as f32;
			if p > 0.0 && p < 1.0 {
				-(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
			} else {
				0.0
			}
		}).collect()
	}).collect();

	// GPU computation
	let (error_counts, filled_counts, cluster_errors, uniqueness_counts, accuracy_counts) =
		metal.compute_all_stats(
			packed_input, words_per_example,
			connections, bits_per_neuron, neurons_per_cluster,
			storages, target_bits, &sample_indices_u32,
			num_clusters, memory_mode,
		);

	// Build NeuronStats
	let neuron_stats: Vec<NeuronStats> = (0..total_neurons).map(|n| {
		let n_bits = bits_per_neuron[n];
		let total_cells = 1usize << n_bits;
		let fill_rate = filled_counts[n].min(total_cells as u32) as f32 / total_cells.max(1) as f32;
		let error_rate = if n_sample > 0 {
			error_counts[n] as f32 / n_sample as f32
		} else {
			0.0
		};
		NeuronStats {
			fill_rate,
			error_rate,
			connection_entropy: conn_entropies[n].clone(),
		}
	}).collect();

	// Build ClusterStats
	let mut neuron_offsets = vec![0usize];
	for &nc in neurons_per_cluster {
		neuron_offsets.push(neuron_offsets.last().unwrap() + nc);
	}

	let cluster_sample_size = n_sample;

	let cluster_stats: Vec<ClusterStats> = (0..num_clusters).map(|c| {
		let n_neurons = neurons_per_cluster[c];
		let neuron_base = neuron_offsets[c];

		let mean_fill = if n_neurons > 0 {
			(0..n_neurons).map(|i| neuron_stats[neuron_base + i].fill_rate).sum::<f32>() / n_neurons as f32
		} else {
			0.0
		};

		let error_rate = if cluster_sample_size > 0 {
			cluster_errors[c] as f32 / cluster_sample_size as f32
		} else {
			0.0
		};

		let uniqueness: Vec<f32> = (0..n_neurons).map(|i| {
			if cluster_sample_size > 0 {
				uniqueness_counts[neuron_base + i] as f32 / cluster_sample_size as f32
			} else {
				0.0
			}
		}).collect();

		let accuracy: Vec<f32> = (0..n_neurons).map(|i| {
			if cluster_sample_size > 0 {
				accuracy_counts[neuron_base + i] as f32 / cluster_sample_size as f32
			} else {
				0.0
			}
		}).collect();

		ClusterStats {
			error_rate,
			mean_fill_rate: mean_fill,
			neuron_uniqueness: uniqueness,
			neuron_accuracy: accuracy,
		}
	}).collect();

	(neuron_stats, cluster_stats)
}

// =============================================================================
// Batched Multi-Genome GPU Stats
// =============================================================================

/// Per-genome data for batched stats computation.
pub struct GenomeStatsInput<'a> {
	pub bits_per_neuron: &'a [usize],
	pub neurons_per_cluster: &'a [usize],
	pub connections: &'a [i64],
	pub storages: &'a [ClusterStorage],
}

/// Compute stats for ALL genomes in a single GPU dispatch.
///
/// ~50x faster than per-genome dispatches:
///   - Single GPU dispatch overhead instead of 50+
///   - Shared input buffers (60-80% of GPU data)
///   - GPU saturated with 10K+ neurons vs 200 per genome
///
/// Connection entropy is computed on CPU (trivially fast from precomputed bit_ones).
#[cfg(target_os = "macos")]
pub fn compute_batch_stats_gpu(
	metal: &crate::metal_stats::MetalStatsComputer,
	genomes: &[GenomeStatsInput],
	packed_input: &[u64],
	words_per_example: usize,
	target_bits: &[u8],
	num_examples: usize,
	num_clusters_per_genome: usize,
	config: &AdaptationConfig,
	memory_mode: u8,
) -> Vec<(Vec<NeuronStats>, Vec<ClusterStats>)> {
	use crate::metal_stats::{BatchedNeuronMeta, BatchedClusterMeta};
	use crate::neuron_memory::{MODE_QUAD_BINARY, MODE_QUAD_WEIGHTED, EMPTY_U8};

	let num_genomes = genomes.len();
	if num_genomes == 0 {
		return Vec::new();
	}

	// Build sample indices (shared across all genomes)
	let sample_indices_usize = build_sample_indices(num_examples, config.stats_sample_size, 42);
	let n_sample = sample_indices_usize.len();
	let sample_indices_u32: Vec<u32> = sample_indices_usize.iter().map(|&i| i as u32).collect();

	// Precompute bit_ones for connection entropy (shared across all genomes)
	let bit_ones = precompute_bit_ones(packed_input, words_per_example, &sample_indices_usize, config.total_input_bits);

	let empty_cell = match memory_mode {
		MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => 1u32,
		_ => EMPTY_U8 as u32,
	};

	// =========================================================================
	// Marshal all genomes into concatenated arrays
	// =========================================================================

	let mut all_neuron_metas: Vec<BatchedNeuronMeta> = Vec::new();
	let mut all_cluster_metas: Vec<BatchedClusterMeta> = Vec::new();
	let mut all_connections_i32: Vec<i32> = Vec::new();
	let mut all_dense_words: Vec<i64> = Vec::new();
	let mut all_sparse_keys: Vec<u64> = Vec::new();
	let mut all_sparse_values: Vec<u8> = Vec::new();
	let mut all_sparse_offsets: Vec<u32> = Vec::new();
	let mut all_sparse_counts: Vec<u32> = Vec::new();

	// Per-genome offsets for result partitioning
	let mut genome_neuron_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
	let mut genome_cluster_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
	genome_neuron_offsets.push(0);
	genome_cluster_offsets.push(0);

	// Per-genome connection entropy (computed on CPU)
	let mut all_conn_entropies: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_genomes);

	let mut global_neuron_cursor = 0u32;
	let mut dense_word_cursor = 0u32;
	let mut sparse_neuron_cursor = 0u32;
	let mut global_conn_offset = 0u32;

	for genome in genomes {
		let bpn = genome.bits_per_neuron;
		let npc = genome.neurons_per_cluster;
		let connections = genome.connections;
		let storages = genome.storages;
		let total_neurons: usize = npc.iter().sum();

		// Connection entropy for this genome (CPU)
		let mut conn_offsets = vec![0usize];
		for &b in bpn {
			conn_offsets.push(conn_offsets.last().unwrap() + b);
		}
		let conn_entropies: Vec<Vec<f32>> = (0..total_neurons).map(|n| {
			let n_bits = bpn[n];
			let conn_start = conn_offsets[n];
			connections[conn_start..conn_start + n_bits].iter().map(|&conn_idx| {
				let ones = bit_ones[conn_idx as usize];
				let p = ones as f32 / n_sample.max(1) as f32;
				if p > 0.0 && p < 1.0 {
					-(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
				} else {
					0.0
				}
			}).collect()
		}).collect();
		all_conn_entropies.push(conn_entropies);

		// Append connections as i32
		all_connections_i32.extend(connections.iter().map(|&c| c as i32));

		// Marshal per-cluster storage and neuron metadata
		let mut local_n = 0usize;
		for cluster in 0..num_clusters_per_genome {
			let n_neurons = npc[cluster];
			all_cluster_metas.push(BatchedClusterMeta {
				first_neuron: global_neuron_cursor,
				num_neurons: n_neurons as u32,
			});

			match &storages[cluster] {
				ClusterStorage::Dense { words, words_per_neuron, .. } => {
					let wpn = *words_per_neuron;
					all_dense_words.extend_from_slice(words);
					for local_neuron in 0..n_neurons {
						all_neuron_metas.push(BatchedNeuronMeta {
							cluster_id: (genome_cluster_offsets.last().unwrap() + cluster) as u32,
							bits: bpn[local_n] as u32,
							conn_offset: global_conn_offset + conn_offsets[local_n] as u32,
							is_sparse: 0,
							dense_word_offset: dense_word_cursor + (local_neuron * wpn) as u32,
							words_per_neuron: wpn as u32,
							sparse_neuron_idx: 0,
						});
						local_n += 1;
					}
					dense_word_cursor += words.len() as u32;
				}
				ClusterStorage::Sparse { neurons: maps, .. } => {
					for local_neuron in 0..n_neurons {
						let map = &maps[local_neuron];
						let sp_offset = all_sparse_keys.len() as u32;
						let mut entries: Vec<(u32, u8)> = map.iter().map(|(&k, &v)| (k, v)).collect();
						entries.sort_unstable_by_key(|(k, _)| *k);
						let count = entries.len() as u32;
						for (k, v) in entries {
							all_sparse_keys.push(k as u64);
							all_sparse_values.push(v);
						}
						all_sparse_offsets.push(sp_offset);
						all_sparse_counts.push(count);
						all_neuron_metas.push(BatchedNeuronMeta {
							cluster_id: (genome_cluster_offsets.last().unwrap() + cluster) as u32,
							bits: bpn[local_n] as u32,
							conn_offset: global_conn_offset + conn_offsets[local_n] as u32,
							is_sparse: 1,
							dense_word_offset: 0,
							words_per_neuron: 0,
							sparse_neuron_idx: sparse_neuron_cursor,
						});
						sparse_neuron_cursor += 1;
						local_n += 1;
					}
				}
			}
			global_neuron_cursor += n_neurons as u32;
		}

		global_conn_offset += connections.len() as u32;
		genome_neuron_offsets.push(global_neuron_cursor as usize);
		genome_cluster_offsets.push(genome_cluster_offsets.last().unwrap() + num_clusters_per_genome);
	}

	let total_neurons = global_neuron_cursor as usize;
	let total_clusters = num_genomes * num_clusters_per_genome;

	// =========================================================================
	// Single GPU dispatch for all genomes
	// =========================================================================

	let result = metal.compute_all_stats_batched(
		packed_input, words_per_example,
		target_bits, &sample_indices_u32,
		&all_connections_i32,
		&all_neuron_metas,
		&all_dense_words,
		&all_sparse_keys,
		&all_sparse_values,
		&all_sparse_offsets,
		&all_sparse_counts,
		&all_cluster_metas,
		total_neurons,
		total_clusters,
		empty_cell,
	);

	// =========================================================================
	// Partition results back into per-genome (NeuronStats, ClusterStats)
	// =========================================================================

	(0..num_genomes).map(|g| {
		let n_start = genome_neuron_offsets[g];
		let n_end = genome_neuron_offsets[g + 1];
		let c_start = genome_cluster_offsets[g];
		let genome = &genomes[g];
		let bpn = genome.bits_per_neuron;
		let npc = genome.neurons_per_cluster;

		// Build NeuronStats
		let neuron_stats: Vec<NeuronStats> = (0..(n_end - n_start)).map(|local_n| {
			let global_n = n_start + local_n;
			let n_bits = bpn[local_n];
			let total_cells = 1usize << n_bits;
			let fill_rate = result.neuron_filled[global_n].min(total_cells as u32) as f32 / total_cells.max(1) as f32;
			let error_rate = if n_sample > 0 {
				result.neuron_errors[global_n] as f32 / n_sample as f32
			} else {
				0.0
			};
			NeuronStats {
				fill_rate,
				error_rate,
				connection_entropy: all_conn_entropies[g][local_n].clone(),
			}
		}).collect();

		// Build ClusterStats
		let mut neuron_offset = 0usize;
		let cluster_stats: Vec<ClusterStats> = (0..num_clusters_per_genome).map(|c| {
			let n_neurons = npc[c];
			let global_c = c_start + c;
			let mean_fill = if n_neurons > 0 {
				(0..n_neurons).map(|i| neuron_stats[neuron_offset + i].fill_rate).sum::<f32>() / n_neurons as f32
			} else {
				0.0
			};
			let error_rate = if n_sample > 0 {
				result.cluster_errors[global_c] as f32 / n_sample as f32
			} else {
				0.0
			};
			let uniqueness: Vec<f32> = (0..n_neurons).map(|i| {
				if n_sample > 0 {
					result.uniqueness_counts[n_start + neuron_offset + i] as f32 / n_sample as f32
				} else {
					0.0
				}
			}).collect();
			let accuracy: Vec<f32> = (0..n_neurons).map(|i| {
				if n_sample > 0 {
					result.accuracy_counts[n_start + neuron_offset + i] as f32 / n_sample as f32
				} else {
					0.0
				}
			}).collect();
			neuron_offset += n_neurons;
			ClusterStats {
				error_rate,
				mean_fill_rate: mean_fill,
				neuron_uniqueness: uniqueness,
				neuron_accuracy: accuracy,
			}
		}).collect();

		(neuron_stats, cluster_stats)
	}).collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_adaptation_config_default() {
		let config = AdaptationConfig::default();
		assert!(!config.synaptogenesis_enabled);
		assert!(!config.neurogenesis_enabled);
		assert_eq!(config.min_bits, 4);
		assert_eq!(config.max_bits, 24);
		assert_eq!(config.warmup_generations, 10);
		assert!((config.prune_entropy_ratio - 0.3).abs() < 0.001);
		assert!((config.grow_fill_utilization - 0.5).abs() < 0.001);
		assert!((config.max_growth_ratio - 1.5).abs() < 0.001);
		assert!((config.stabilize_fraction - 0.25).abs() < 0.001);
	}

	#[test]
	fn test_adaptation_rate_schedule() {
		let config = AdaptationConfig {
			warmup_generations: 10,
			total_generations: 250,
			stabilize_fraction: 0.25,
			..Default::default()
		};

		// Before warmup: rate = 0
		assert_eq!(adaptation_rate(0, &config), 0.0);
		assert_eq!(adaptation_rate(5, &config), 0.0);
		assert_eq!(adaptation_rate(9, &config), 0.0);

		// First active gen: rate ≈ 1.0
		let rate_10 = adaptation_rate(10, &config);
		assert!(rate_10 > 0.99, "rate at gen 10 should be ~1.0, got {}", rate_10);

		// Mid-point: rate ≈ 0.5
		// active_window = 240 * 0.75 = 180, midpoint = 10 + 90 = 100
		let rate_100 = adaptation_rate(100, &config);
		assert!(rate_100 > 0.4 && rate_100 < 0.6, "rate at gen 100 should be ~0.5, got {}", rate_100);

		// End of active window: rate = 0
		let rate_190 = adaptation_rate(190, &config);
		assert!(rate_190 < 0.01, "rate at gen 190 should be ~0, got {}", rate_190);

		// Stabilization: rate = 0
		assert_eq!(adaptation_rate(191, &config), 0.0);
		assert_eq!(adaptation_rate(250, &config), 0.0);
	}

	#[test]
	fn test_expected_fill_rate() {
		// 8-bit neuron with 1000 examples: 1000 / 256 = 3.9 → capped at 1.0
		assert!((expected_fill_rate(8, 1000) - 1.0).abs() < 0.001);

		// 20-bit neuron with 150K examples: 150000 / 1048576 ≈ 0.143
		let fill = expected_fill_rate(20, 150_000);
		assert!(fill > 0.14 && fill < 0.15, "expected fill for 20-bit/150K: {}", fill);

		// 12-bit neuron with 50K examples: 50000 / 4096 ≈ 12.2 → capped at 1.0
		assert!((expected_fill_rate(12, 50_000) - 1.0).abs() < 0.001);
	}

	#[test]
	fn test_statistical_helpers() {
		let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
		assert!((median_of(&vals) - 3.0).abs() < 0.001);
		assert!((mean_of(&vals) - 3.0).abs() < 0.001);
		assert!((percentile_of(&vals, 0.0) - 1.0).abs() < 0.001);
		assert!((percentile_of(&vals, 100.0) - 5.0).abs() < 0.001);
		assert!((percentile_of(&vals, 50.0) - 3.0).abs() < 0.001);

		// Empty
		assert_eq!(median_of(&[]), 0.0);
		assert_eq!(mean_of(&[]), 0.0);
		assert_eq!(percentile_of(&[], 50.0), 0.0);
	}

	#[test]
	fn test_neurogenesis_warmup_guard() {
		let config = AdaptationConfig {
			neurogenesis_enabled: true,
			warmup_generations: 10,
			..Default::default()
		};

		let mut bits = vec![8, 8, 10, 10];
		let mut neurons = vec![2, 2];
		let mut connections: Vec<i64> = (0..36).collect();
		let mut cooldowns = vec![0, 0];
		let initial_neurons = vec![2, 2];
		let cluster_stats = vec![
			ClusterStats { error_rate: 0.9, mean_fill_rate: 0.9, neuron_uniqueness: vec![0.5, 0.5], neuron_accuracy: vec![0.5, 0.5] },
			ClusterStats { error_rate: 0.9, mean_fill_rate: 0.9, neuron_uniqueness: vec![0.5, 0.5], neuron_accuracy: vec![0.5, 0.5] },
		];

		let mut rng = rand::rngs::StdRng::seed_from_u64(42);
		let (added, removed) = neurogenesis_pass(
			&mut bits, &mut neurons, &mut connections,
			&cluster_stats, &config, 5, &mut cooldowns,
			&initial_neurons, 1000, 1.0, &mut rng,
		);

		// Generation 5 < warmup 10 → no changes
		assert_eq!(added, 0);
		assert_eq!(removed, 0);
	}
}
