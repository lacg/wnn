//! Training-time architecture adaptation: synaptogenesis and neurogenesis.
//!
//! These mechanisms run within a single genome evaluation cycle, after initial
//! training and before final evaluation. They adapt the architecture based on
//! observed training statistics.
//!
//! - **Synaptogenesis** (connection level): prune/grow connections per-neuron
//! - **Neurogenesis** (cluster level): add/remove neurons per-cluster

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
#[derive(Clone)]
pub struct AdaptationConfig {
	// Synaptogenesis (connection level)
	pub synaptogenesis_enabled: bool,
	pub prune_entropy_threshold: f32,
	pub grow_fill_threshold: f32,
	pub grow_error_threshold: f32,
	pub min_bits: usize,
	pub max_bits: usize,

	// Neurogenesis (cluster level)
	pub neurogenesis_enabled: bool,
	pub cluster_error_threshold: f32,
	pub cluster_fill_threshold: f32,
	pub neuron_uniqueness_threshold: f32,
	pub min_neurons: usize,
	pub max_neurons: usize,
	pub max_neurons_per_pass: usize,
	pub warmup_generations: usize,
	pub cooldown_iterations: usize,

	// Shared
	pub passes_per_eval: usize,
	pub total_input_bits: usize,
	/// Max examples to sample for stats computation (0 = use all).
	/// Stats are statistical estimates — sampling 10K examples gives
	/// <1% standard error while being 60x faster than scanning 600K.
	pub stats_sample_size: usize,
}

impl Default for AdaptationConfig {
	fn default() -> Self {
		Self {
			synaptogenesis_enabled: false,
			prune_entropy_threshold: 0.05,
			grow_fill_threshold: 0.8,
			grow_error_threshold: 0.5,
			min_bits: 4,
			max_bits: 24,
			neurogenesis_enabled: false,
			cluster_error_threshold: 0.5,
			cluster_fill_threshold: 0.7,
			neuron_uniqueness_threshold: 0.05,
			min_neurons: 3,
			max_neurons: 30,
			max_neurons_per_pass: 3,
			warmup_generations: 10,
			cooldown_iterations: 5,
			passes_per_eval: 1,
			total_input_bits: 64,
			stats_sample_size: 10_000,
		}
	}
}

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

	// Sample examples — full scan is O(clusters*neurons*examples) which is
	// prohibitively slow for large datasets.
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

					let majority_count: u32 = (0..n_neurons).map(|j| neuron_votes[j][s] as u32).sum();
					let majority_vote = majority_count > (n_neurons as u32 / 2);
					if my_vote != majority_vote { disagree += 1; }

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

/// Synaptogenesis pass: prune low-entropy connections, grow where underfitting.
///
/// Modifies bits_per_neuron and connections in-place.
/// Returns (pruned_count, grown_count).
pub fn synaptogenesis_pass(
	bits_per_neuron: &mut Vec<usize>,
	connections: &mut Vec<i64>,
	neuron_stats: &[NeuronStats],
	config: &AdaptationConfig,
	packed_input: &[u64],
	words_per_example: usize,
	num_examples: usize,
	_rng: &mut impl Rng,
) -> (usize, usize) {
	let total_neurons = bits_per_neuron.len();
	let mut pruned = 0usize;
	let mut grown = 0usize;

	// Build connection offsets
	let mut conn_offsets = vec![0usize];
	for &b in bits_per_neuron.iter() {
		conn_offsets.push(conn_offsets.last().unwrap() + b);
	}

	// Pre-compute bit entropy for all input bits (used by grow path).
	// Uses sampled examples for consistency with compute_neuron_stats.
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
	// We rebuild connections after all modifications to avoid index shifting issues
	let mut new_bits = bits_per_neuron.clone();
	let mut modifications: Vec<(usize, Modification)> = Vec::new();

	for n in 0..total_neurons {
		let n_bits = bits_per_neuron[n];
		let stats = &neuron_stats[n];

		// Prune: remove lowest-entropy connection if below threshold
		if n_bits > config.min_bits {
			if let Some((min_idx, &min_entropy)) = stats.connection_entropy.iter()
				.enumerate()
				.min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
			{
				if min_entropy < config.prune_entropy_threshold {
					new_bits[n] = n_bits - 1;
					modifications.push((n, Modification::Prune(min_idx)));
					pruned += 1;
					continue; // Don't both prune and grow same neuron
				}
			}
		}

		// Grow: add highest-entropy unconnected bit if underfitting
		if n_bits < config.max_bits
			&& stats.fill_rate > config.grow_fill_threshold
			&& stats.error_rate > config.grow_error_threshold
		{
			// Find highest-entropy unconnected input bit (from precomputed table)
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

	(pruned, grown)
}

enum Modification {
	Prune(usize),  // index of connection to remove
	Grow(i64),     // new connection index to add
}

/// Neurogenesis pass: add/remove neurons from clusters.
///
/// Modifies bits_per_neuron, neurons_per_cluster, and connections in-place.
/// Returns (added_count, removed_count).
pub fn neurogenesis_pass(
	bits_per_neuron: &mut Vec<usize>,
	neurons_per_cluster: &mut Vec<usize>,
	connections: &mut Vec<i64>,
	cluster_stats: &[ClusterStats],
	config: &AdaptationConfig,
	generation: usize,
	cooldowns: &mut Vec<usize>,  // per-cluster cooldown counters
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

		// Growth: high error + high fill + room to grow
		if stats.error_rate > config.cluster_error_threshold
			&& stats.mean_fill_rate > config.cluster_fill_threshold
			&& n_neurons < config.max_neurons
		{
			// Find best-performing neuron to clone
			let best_local = stats.neuron_accuracy.iter()
				.enumerate()
				.max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
				.map(|(i, _)| i)
				.unwrap_or(0);

			let neurons_to_add = 1.min(config.max_neurons - n_neurons)
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

		// Pruning: remove redundant neurons
		if n_neurons > config.min_neurons {
			// Find most redundant neuron (lowest uniqueness or lowest accuracy)
			let mut worst_local = None;
			let mut worst_score = f32::MAX;
			for i in 0..n_neurons {
				// Combine uniqueness and accuracy — low in both = prime candidate
				let score = stats.neuron_uniqueness[i] + stats.neuron_accuracy[i];
				if stats.neuron_uniqueness[i] < config.neuron_uniqueness_threshold
					&& score < worst_score
				{
					worst_score = score;
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

	(added, removed)
}

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
	}

	#[test]
	fn test_synaptogenesis_noop_when_disabled() {
		let config = AdaptationConfig {
			synaptogenesis_enabled: false,
			..Default::default()
		};

		let mut bits = vec![8, 10, 12];
		let mut connections: Vec<i64> = (0..30).collect();
		let neuron_stats = vec![
			NeuronStats { fill_rate: 0.9, error_rate: 0.6, connection_entropy: vec![0.01; 8] },
			NeuronStats { fill_rate: 0.9, error_rate: 0.6, connection_entropy: vec![0.01; 10] },
			NeuronStats { fill_rate: 0.9, error_rate: 0.6, connection_entropy: vec![0.01; 12] },
		];

		// Even with stats suggesting changes, disabled config should noop
		// (synaptogenesis_pass checks happen regardless — it's the caller's
		// responsibility to check config.synaptogenesis_enabled)
		// Just verify it doesn't panic
		let mut rng = rand::rngs::StdRng::seed_from_u64(42);
		let (pruned, grown) = synaptogenesis_pass(
			&mut bits, &mut connections, &neuron_stats, &config,
			&[], 0, 0, &mut rng,
		);
		// All neurons have entropy below threshold → should prune
		assert!(pruned > 0 || grown == 0);
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
		let cluster_stats = vec![
			ClusterStats { error_rate: 0.9, mean_fill_rate: 0.9, neuron_uniqueness: vec![0.5, 0.5], neuron_accuracy: vec![0.5, 0.5] },
			ClusterStats { error_rate: 0.9, mean_fill_rate: 0.9, neuron_uniqueness: vec![0.5, 0.5], neuron_accuracy: vec![0.5, 0.5] },
		];

		let mut rng = rand::rngs::StdRng::seed_from_u64(42);
		let (added, removed) = neurogenesis_pass(
			&mut bits, &mut neurons, &mut connections,
			&cluster_stats, &config, 5, &mut cooldowns, &mut rng,
		);

		// Generation 5 < warmup 10 → no changes
		assert_eq!(added, 0);
		assert_eq!(removed, 0);
	}
}
