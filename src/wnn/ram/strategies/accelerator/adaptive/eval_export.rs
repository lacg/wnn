//! Shared genome training/export machinery: GenomeExport, memory-pool sizing, slot training, GPU export, per-example scoring.
//!
//! Split out of adaptive/eval.rs (D3 follow-up, 11/06/2026).

use super::*;

// =============================================================================
// PARALLEL HYBRID CPU+GPU EVALUATION
// =============================================================================

/// Export data for a single genome (used in batched GPU evaluation)
#[derive(Clone)]
pub struct GenomeExport
{
	/// Connections for all groups, flattened
	pub connections: Vec<i64>,
	/// For each group: (is_sparse, group_idx, cluster_ids)
	pub group_info: Vec<(bool, usize, Vec<usize>)>,
	/// Dense group exports: memory words
	pub dense_exports: Vec<Vec<i64>>,
	/// Sparse group exports: sorted arrays for binary search
	pub sparse_exports: Vec<SparseGpuExport>,
	/// Config groups for this genome
	pub groups: Vec<ConfigGroup>,
}

impl GenomeExport
{
	/// Sentinel: an "empty" export used as a partial-result placeholder when
	/// a per-genome rayon worker bails on cancellation (added 31/05/2026).
	/// All fields are empty Vecs; downstream code treats this as "no cells
	/// trained for this genome" and the per-genome metrics come out as
	/// the empty-memory defaults (matching how default-initialised genomes
	/// behave on day 1).
	pub fn empty() -> Self
	{
		Self {
			connections: Vec::new(),
			group_info: Vec::new(),
			dense_exports: Vec::new(),
			sparse_exports: Vec::new(),
			groups: Vec::new(),
		}
	}

	/// Total MATERIALIZED cell count for this trained genome — the convention-free
	/// footprint primitive (see docs/sparse_footprint_fix.md). Sparse groups
	/// (bits > SPARSE_THRESHOLD) materialize only the distinct trained addresses
	/// (`keys.len()`); dense groups materialize the full array (`true_neurons ×
	/// 2^bits`). NOT the dense-only `2^bits` fiction — sparse genomes never
	/// materialize their 2^34 address space. Every byte/LUT figure derives from
	/// this count. `group_info[i]` is built in lockstep with `groups[i]`.
	pub fn materialized_cells(&self) -> u64
	{
		let mut total = 0u64;
		for (i, (is_sparse, sub_idx, _)) in self.group_info.iter().enumerate()
		{
			if *is_sparse
			{
				total += self.sparse_exports[*sub_idx].keys.len() as u64;
			}
			else
			{
				let g = &self.groups[i];
				// bits ≤ SPARSE_THRESHOLD for dense, so 2^bits is small; guard anyway.
				let addrs = 1u64.checked_shl(g.bits as u32).unwrap_or(u64::MAX);
				total = total.saturating_add((g.true_total_neurons() as u64).saturating_mul(addrs));
			}
		}
		total
	}

	/// Path 2 abstraction: read a single trained-memory cell at
	/// (logical_group_idx, neuron_in_group, address). Returns the raw cell
	/// value as i64 (matches GroupMemory.read so cell_to_weight just works).
	///
	/// Used by `compute_neuron_stats_adaptive` so the IDS adaptive research
	/// variant can consume a `GenomeExport` directly instead of needing the
	/// dense `Vec<GroupMemory>` representation (which forced the legacy
	/// `train_genome_in_slot` path).
	/// `miss_default` = the cell an absent SPARSE address reads as (QUAD: 1 =
	/// WEAK_FALSE, TERNARY: 2 = EMPTY — `default_cell_for_mode(memory_mode)`).
	/// Dense groups materialize every cell, so it only affects sparse groups.
	#[inline]
	pub fn read_cell_at(
		&self,
		group_idx: usize,
		neuron_in_group: usize,
		address: u64,
		miss_default: u8,
	) -> i64
	{
		let (is_sparse, sub_idx, _) = &self.group_info[group_idx];
		if *is_sparse
		{
			self.sparse_exports[*sub_idx].lookup(neuron_in_group, address, miss_default) as i64
		}
		else
		{
			let words = &self.dense_exports[*sub_idx];
			let bits = self.groups[group_idx].bits;
			let cells_per_neuron = 1usize << bits;
			let words_per_neuron = (cells_per_neuron + ram_core::neuron_memory::CELLS_PER_WORD - 1)
				/ ram_core::neuron_memory::CELLS_PER_WORD;
			let addr = address as usize;
			let word_idx = addr / ram_core::neuron_memory::CELLS_PER_WORD;
			let cell_idx = addr % ram_core::neuron_memory::CELLS_PER_WORD;
			let word = words[neuron_in_group * words_per_neuron + word_idx];
			(word >> (cell_idx * ram_core::neuron_memory::BITS_PER_CELL))
				& ram_core::neuron_memory::CELL_MASK
		}
	}

	/// Path 2 abstraction: fraction of a neuron's addresses that are non-EMPTY.
	/// `neuron_in_group` is the neuron's position WITHIN the group (NOT global).
	/// `bits` is the number of address bits this neuron uses.
	///
	/// Mirrors `GroupMemory::neuron_fill_rate` so call-sites can be migrated
	/// without changing semantics.
	pub fn neuron_fill_rate(&self, group_idx: usize, neuron_in_group: usize, bits: usize) -> f32
	{
		let total_cells = 1usize << bits;
		let (is_sparse, sub_idx, _) = &self.group_info[group_idx];
		if *is_sparse
		{
			let s = &self.sparse_exports[*sub_idx];
			if neuron_in_group >= s.counts.len()
			{
				return 0.0;
			}
			let count = s.counts[neuron_in_group] as usize;
			count.min(total_cells) as f32 / total_cells.max(1) as f32
		}
		else
		{
			let words = &self.dense_exports[*sub_idx];
			let words_per_neuron = (total_cells + ram_core::neuron_memory::CELLS_PER_WORD - 1)
				/ ram_core::neuron_memory::CELLS_PER_WORD;
			let start = neuron_in_group * words_per_neuron;
			let mut filled = 0u32;
			for w in 0..words_per_neuron
			{
				if start + w >= words.len()
				{
					break;
				}
				let word = words[start + w];
				for c in 0..ram_core::neuron_memory::CELLS_PER_WORD
				{
					let cell = (word >> (c * ram_core::neuron_memory::BITS_PER_CELL))
						& ram_core::neuron_memory::CELL_MASK;
					if cell != ram_core::neuron_memory::EMPTY
					{
						filled += 1;
					}
				}
			}
			filled.min(total_cells as u32) as f32 / total_cells.max(1) as f32
		}
	}
}

/// Calculate optimal pool and batch sizes based on memory budget
///
/// `num_train` + `neuron_sample_rate` make the sparse estimate dataset-size
/// aware: the previous hardcoded 3K-entries/neuron constant was calibrated on
/// ~100K-row datasets and under-estimated the 46M CIC-IoT footprint by 2-3
/// orders of magnitude → batch_size 10 concurrent genomes → ~63 GB heap →
/// jetsam kill loop (root-caused 05/07/2026).
pub(crate) fn calculate_pool_size(
	bits_per_cluster: &[usize],
	neurons_per_cluster: &[usize],
	_num_clusters: usize,
	budget_gb: f64,
	cpu_cores: usize,
	num_train: usize,
	neuron_sample_rate: f32,
) -> (usize, usize)
{
	// Estimate memory per genome (use same grouping strategy as actual training)
	let groups = build_groups(bits_per_cluster, neurons_per_cluster);
	let mut bytes_per_genome = 0usize;

	// OI QUAD training holds per-address vote tallies alongside the cells
	// (tally-then-commit), roughly doubling the per-entry cost while training.
	let oi_factor: usize = if ram_core::neuron_memory::order_independent_training_enabled()
	{
		2
	}
	else
	{
		1
	};

	for group in &groups
	{
		if group.bits <= SPARSE_THRESHOLD
		{
			// Dense: 2 bits per cell, 2^bits cells per neuron
			let cells_per_neuron = 1 << group.bits;
			let words_per_neuron = (cells_per_neuron + 30) / 31; // 31 cells per word
			bytes_per_genome += group.total_neurons() * words_per_neuron * 8;
		}
		else
		{
			// Sparse: distinct trained addresses per neuron are bounded by the
			// sampled train rows (and 2^bits at moderate widths). Reuse the
			// batched path's capacity formula (marker_capacity_for_train:
			// min(num_train×sr, 2^bits) ×2 oversize, next_pow2) — deliberately
			// conservative, matching the hashtable the marker path would size.
			// Memory per entry: key(8) + value(1) + DashMap overhead (~24 bytes).
			let entries_per_neuron = crate::marker_train::genome_path::marker_capacity_for_train(
				num_train,
				group.bits,
				neuron_sample_rate,
			);
			bytes_per_genome = bytes_per_genome
				.saturating_add(group.total_neurons() * entries_per_neuron * 32 * oi_factor);
		}
	}

	let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;
	let max_pool_size = (budget_bytes / bytes_per_genome).max(1);

	// Pool size cap:
	//   - Default (baseline path): cap at `cpu_cores` (rayon-bounded per-genome
	//     parallelism — more batches than cores just queues serially).
	//   - GPU batched train: kernel dispatches all genomes in one Metal call
	//     (GPU has ~1280 SIMD lanes), so the cpu_cores cap is artificial.
	//     Use a higher cap (B9_GPU_BATCH_CAP) so we can absorb 50+ genomes
	//     per dispatch when memory allows.
	// WNN_BATCH_SIZE env var still overrides everything for testing.
	let gpu_batched = gpu_batched_train_enabled();
	const B9_GPU_BATCH_CAP: usize = 50;
	let effective_cap = if gpu_batched
	{
		cpu_cores.max(B9_GPU_BATCH_CAP)
	}
	else
	{
		cpu_cores
	};
	let pool_size = max_pool_size.min(effective_cap).max(1);

	// Batch size = pool size (process one batch at a time)
	let batch_size = pool_size;

	(pool_size, batch_size)
}

/// Get available memory in GB (macOS specific)
pub(crate) fn get_available_memory_gb() -> f64
{
	// Actually-AVAILABLE memory (free + inactive + purgeable + speculative),
	// not hw.memsize: the old total-RAM answer made the budget blind to
	// co-tenant processes (and to our own persistent allocations — dataset,
	// packed inputs, caches — which legitimately shrink what a new genome
	// batch may claim). Floor at 4 GB so a momentary low-memory reading can't
	// zero the pool (batch size is additionally clamped to ≥1 downstream).
	#[cfg(target_os = "macos")]
	{
		use std::process::Command;
		if let Ok(output) = Command::new("vm_stat").output()
		{
			if let Ok(text) = String::from_utf8(output.stdout)
			{
				let mut page_size: f64 = 16384.0;
				if let Some(first) = text.lines().next()
				{
					if let Some(ps) = first.split("page size of").nth(1)
					{
						if let Some(n) = ps
							.split_whitespace()
							.next()
							.and_then(|s| s.parse::<f64>().ok())
						{
							page_size = n;
						}
					}
				}
				let mut pages: f64 = 0.0;
				let mut parsed_any = false;
				for line in text.lines()
				{
					for key in [
						"Pages free:",
						"Pages inactive:",
						"Pages purgeable:",
						"Pages speculative:",
					]
					{
						if let Some(rest) = line.strip_prefix(key)
						{
							if let Some(n) = rest.trim().trim_end_matches('.').parse::<f64>().ok()
							{
								pages += n;
								parsed_any = true;
							}
						}
					}
				}
				if parsed_any
				{
					let gb = pages * page_size / (1024.0 * 1024.0 * 1024.0);
					return gb.max(4.0);
				}
			}
		}
		// vm_stat unavailable/unparseable: fall back to half of total RAM
		// (the old hw.memsize answer was TOTAL, which over-promised).
		if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.memsize").output()
		{
			if let Ok(mem_str) = String::from_utf8(output.stdout)
			{
				if let Ok(bytes) = mem_str.trim().parse::<u64>()
				{
					return (bytes as f64 / (1024.0 * 1024.0 * 1024.0) * 0.5).max(4.0);
				}
			}
		}
	}
	// Fallback: assume half of 64GB (M4 Max typical)
	32.0
}

pub fn compute_class_weights_with_multiplier(
	labels: &[i64],
	num_classes: usize,
	multiplier: f32,
) -> Vec<u32>
{
	let mut counts = vec![0u64; num_classes];
	for &label in labels
	{
		let c = label as usize;
		if c < num_classes
		{
			counts[c] += 1;
		}
	}
	let max_count = *counts.iter().max().unwrap_or(&1);
	counts
		.iter()
		.map(|&c| {
			if c == 0
			{
				1
			}
			else
			{
				let base = (max_count / c).max(1) as f32;
				(base * multiplier).max(1.0) as u32
			}
		})
		.collect()
}

/// Train a genome using the given memory slot.
/// When `gpu_addresses` is Some, uses pre-computed GPU addresses instead of CPU compute_address().
/// GPU address layout: addresses[global_neuron_idx * num_train + example_idx].
/// When `parallel` is true, uses rayon par_iter for example-level parallelism.
/// Set `parallel=false` when calling from within an outer par_iter to avoid nested parallelism deadlock.
pub(crate) fn train_genome_in_slot(
	memories: &mut [GroupMemory],
	groups: &[ConfigGroup],
	original_connections: &[i64],    // Per-neuron layout (NOT group layout)
	per_neuron_bits: &[usize],       // Bits per neuron
	cluster_neuron_starts: &[usize], // First neuron idx per cluster
	neuron_conn_offsets: &[usize],   // Conn offset per neuron
	cluster_to_group: &[(usize, usize)],
	train_input_bits: &ram_core::packed_bits::PackedBits,
	train_targets: &[i64],
	train_negatives: &[i64],
	num_train: usize,
	num_negatives: usize,
	_total_input_bits: usize,
	gpu_addresses: Option<&[u64]>,
	neuron_sample_rate: f32,
	rng_seed: u64,
	memory_mode: u8,
	class_weights: Option<&[u32]>,
	parallel: bool,
)
{
	// OI orchestration: init counter buffers (when enabled), train, then commit.
	let oi = ram_core::neuron_memory::order_independent_training_enabled()
		&& memory_mode == ram_core::neuron_memory::QUAD_WEIGHTED;
	if oi
	{
		for m in memories.iter_mut()
		{
			m.init_oi_counters();
		}
	}
	// Thin wrapper: full-range training with stride == num_train (existing behavior).
	train_genome_in_slot_range(
		memories,
		groups,
		original_connections,
		per_neuron_bits,
		cluster_neuron_starts,
		neuron_conn_offsets,
		cluster_to_group,
		train_input_bits,
		train_targets,
		train_negatives,
		num_train,
		num_negatives,
		_total_input_bits,
		gpu_addresses,
		0..num_train,
		num_train,
		neuron_sample_rate,
		rng_seed,
		memory_mode,
		class_weights,
		parallel,
	);
	if oi
	{
		for m in memories.iter_mut()
		{
			m.commit_oi();
		}
	}
}

/// Range-aware training: writes memory cells for examples in `example_range`.
///
/// `addr_stride` is the stride between neurons in `gpu_addresses`. For the
/// non-chunked path this equals `num_train` (passed by the wrapper); for chunked
/// GPU address compute it equals the chunk length. Address indexing:
/// `gpu_addresses[global_n * addr_stride + (ex_idx - example_range.start)]`.
pub(crate) fn train_genome_in_slot_range(
	memories: &[GroupMemory],
	groups: &[ConfigGroup],
	original_connections: &[i64],
	per_neuron_bits: &[usize],
	cluster_neuron_starts: &[usize],
	neuron_conn_offsets: &[usize],
	cluster_to_group: &[(usize, usize)],
	train_input_bits: &ram_core::packed_bits::PackedBits,
	train_targets: &[i64],
	train_negatives: &[i64],
	_num_train: usize,
	num_negatives: usize,
	_total_input_bits: usize,
	gpu_addresses: Option<&[u64]>,
	example_range: std::ops::Range<usize>,
	addr_stride: usize,
	neuron_sample_rate: f32,
	rng_seed: u64,
	memory_mode: u8,
	class_weights: Option<&[u32]>,
	parallel: bool,
)
{
	let use_sampling = neuron_sample_rate < 1.0;
	// BINARY (classical WiSARD, Luiz 12/07/2026): one-shot own-class set —
	// a positive-class visit writes TRUE; FALSE-direction and negative-class
	// visits are IGNORED (per-discriminator classical training). No nudging,
	// no OI (a set is commutative — order-independent by construction).
	let is_binary = memory_mode == ram_core::neuron_memory::BINARY;
	let use_nudge = memory_mode != ram_core::neuron_memory::TERNARY && !is_binary;
	// OI is only meaningful for QUAD_WEIGHTED (the only mode where the existing
	// clamped nudge has order-dependence to fix).
	let use_oi = ram_core::neuron_memory::order_independent_training_enabled()
		&& memory_mode == ram_core::neuron_memory::QUAD_WEIGHTED;
	let chunk_start = example_range.start;

	let train_one_example = |ex_idx: usize| {
		let input_bits = train_input_bits.packed_row(ex_idx);

		let num_clusters = cluster_to_group.len();
		// Single-cluster mode: always target cluster 0, nudge direction = label
		// Multi-cluster mode: target cluster = label, always nudge TRUE
		let true_cluster = if num_clusters == 1
		{
			0
		}
		else
		{
			train_targets[ex_idx] as usize
		};
		let nudge_direction = if num_clusters == 1
		{
			train_targets[ex_idx] == 1 // Attack=TRUE, Normal=FALSE
		}
		else
		{
			true // Always positive for target cluster in multi-cluster mode
		};

		// Train positive example
		{
			let (group_idx, local_cluster) = cluster_to_group[true_cluster];
			let group = &groups[group_idx];
			let memory = &memories[group_idx];

			let actual_neurons = if let Some(ref an) = group.actual_neurons
			{
				an[local_cluster] as usize
			}
			else
			{
				group.neurons
			};

			let neuron_base = local_cluster * group.neurons; // Keep MAX for memory layout

			for n in 0..actual_neurons
			{
				let global_n = cluster_neuron_starts[true_cluster] + n;

				// Per-(neuron, example) deterministic sampling
				// Uses hash of (rng_seed, neuron_idx, example_idx) for parallel-safe decisions
				if use_sampling
				{
					let mut rng = (rng_seed as u32)
						.wrapping_add(global_n as u32 * 1000003)
						.wrapping_add(ex_idx as u32 * 2654435761);
					if rng == 0
					{
						rng = 1;
					}
					rng ^= rng << 13;
					rng ^= rng >> 17;
					rng ^= rng << 5;
					if (rng >> 8) as f32 / 16777216.0 >= neuron_sample_rate
					{
						continue;
					}
				}

				let address = if let Some(addrs) = gpu_addresses
				{
					addrs[global_n * addr_stride + (ex_idx - chunk_start)] as usize
				}
				else
				{
					let n_bits = per_neuron_bits[global_n];
					let conn_start = neuron_conn_offsets[global_n];
					ram_core::neuron_memory::compute_address_packed_bytes(
						input_bits,
						&original_connections[conn_start..],
						n_bits,
					)
				};
				// Weight by original label for class balancing
				let weight_idx = train_targets[ex_idx] as usize;
				let repeats = class_weights.map_or(1u32, |w| w[weight_idx]);
				if is_binary
				{
					// Classical 1-bit: set TRUE on positive-direction visits;
					// benign/negative-direction visits touch nothing (class
					// weights are moot — a bit has no graduation).
					if nudge_direction
					{
						memory.write(neuron_base + n, address, TRUE, false);
					}
				}
				else if use_oi
				{
					// OI: one accumulating call per example with weight = class_weight.
					// Semantically counts this as a single observation (obs += 1)
					// regardless of weight, while the net moves by ±weight.
					memory.nudge_oi(neuron_base + n, address, nudge_direction, repeats);
				}
				else if use_nudge
				{
					for _ in 0..repeats
					{
						memory.nudge(neuron_base + n, address, nudge_direction);
					}
				}
				else
				{
					let value = if nudge_direction { TRUE } else { FALSE };
					memory.write(neuron_base + n, address, value, false);
				}
			}
		}

		// Train negative examples.
		//
		// Single-cluster (binary IDS) encodes the FALSE direction via
		// train_targets[ex_idx] == 0 + nudge_direction in the positive branch
		// above; the negative loop is multi-class only (K > 1 with explicit
		// per-example negative cluster IDs in train_negatives). Skip when
		// there's only one cluster — defense against callers that mis-form
		// the train_negatives buffer (e.g., the FPGA export wrapper bug at
		// lib.rs:7662-7667 that pre-dated this guard, where row-indices got
		// mis-interpreted as cluster-ids and panicked at the indexing below).
		if cluster_to_group.len() == 1
		{
			// Inside a rayon closure (per-example), `return` exits this
			// closure invocation cleanly — equivalent to `continue` in a
			// regular for-loop.
			return;
		}
		if is_binary
		{
			// Classical training never writes negatives (own-class only).
			return;
		}
		let neg_start = ex_idx * num_negatives;
		for k in 0..num_negatives
		{
			let false_cluster = train_negatives[neg_start + k] as usize;
			if false_cluster == true_cluster
			{
				continue;
			}

			let (group_idx, local_cluster) = cluster_to_group[false_cluster];
			let group = &groups[group_idx];
			let memory = &memories[group_idx];

			let actual_neurons = if let Some(ref an) = group.actual_neurons
			{
				an[local_cluster] as usize
			}
			else
			{
				group.neurons
			};

			let neuron_base = local_cluster * group.neurons; // Keep MAX for memory layout

			for n in 0..actual_neurons
			{
				let global_n = cluster_neuron_starts[false_cluster] + n;

				// Same per-(neuron, example) sampling for negative examples
				if use_sampling
				{
					let mut rng = (rng_seed as u32)
						.wrapping_add(global_n as u32 * 1000003)
						.wrapping_add(ex_idx as u32 * 2654435761);
					if rng == 0
					{
						rng = 1;
					}
					rng ^= rng << 13;
					rng ^= rng >> 17;
					rng ^= rng << 5;
					if (rng >> 8) as f32 / 16777216.0 >= neuron_sample_rate
					{
						continue;
					}
				}

				let address = if let Some(addrs) = gpu_addresses
				{
					addrs[global_n * addr_stride + (ex_idx - chunk_start)] as usize
				}
				else
				{
					let n_bits = per_neuron_bits[global_n];
					let conn_start = neuron_conn_offsets[global_n];
					ram_core::neuron_memory::compute_address_packed_bytes(
						input_bits,
						&original_connections[conn_start..],
						n_bits,
					)
				};
				// For negative nudges, weight by the TRUE class of the example
				// (the example "belongs to" true_cluster, so its weight applies)
				let repeats = class_weights.map_or(1u32, |w| w[true_cluster]);
				if use_oi
				{
					memory.nudge_oi(neuron_base + n, address, false, repeats);
				}
				else if use_nudge
				{
					for _ in 0..repeats
					{
						memory.nudge(neuron_base + n, address, false);
					}
				}
				else
				{
					memory.write(neuron_base + n, address, FALSE, false);
				}
			}
		}
	};

	let range_len = example_range.end - example_range.start;
	if parallel
	{
		let chunk_size = 10_000.max(range_len / 20);
		example_range
			.clone()
			.into_par_iter()
			.with_min_len(chunk_size)
			.for_each(|ex_idx| train_one_example(ex_idx));
	}
	else
	{
		for ex_idx in example_range.clone()
		{
			train_one_example(ex_idx);
		}
	}
}

/// Export trained memory to GPU-compatible format
pub(crate) fn export_genome_for_gpu(
	memories: &[GroupMemory],
	groups: &[ConfigGroup],
	connections_flat: &[i64],
) -> GenomeExport
{
	let mut dense_exports = Vec::new();
	let mut sparse_exports = Vec::new();
	let mut group_info = Vec::new();

	for (group_idx, (group, memory)) in groups.iter().zip(memories.iter()).enumerate()
	{
		let is_sparse = memory.is_sparse();
		group_info.push((is_sparse, group_idx, group.cluster_ids.clone()));

		if is_sparse
		{
			if let Some(export) = memory.export_for_gpu_sparse()
			{
				sparse_exports.push(export);
			}
			else
			{
				// Fallback: empty export
				sparse_exports.push(SparseGpuExport {
					keys: vec![],
					values: vec![],
					offsets: vec![0; group.total_neurons()],
					counts: vec![0; group.total_neurons()],
					num_neurons: group.total_neurons(),
				});
			}
		}
		else
		{
			if let Some(words) = memory.export_for_metal()
			{
				dense_exports.push(words);
			}
			else
			{
				dense_exports.push(vec![]);
			}
		}
	}

	GenomeExport {
		connections: connections_flat.to_vec(),
		group_info,
		dense_exports,
		sparse_exports,
		groups: groups.to_vec(),
	}
}

// Thread-local cache for GPU buffers to avoid expensive 10GB buffer allocation per evaluation
// The scores buffer is ~10GB (50K examples × 50K clusters × 4 bytes), so reusing it is critical.
// The cache includes the reset generation to invalidate on Metal reset.
#[cfg(target_os = "macos")]
thread_local! {
		// (reset_gen, num_eval, num_clusters, buffer)
		pub(crate) static CACHED_SCORES_BUFFER: std::cell::RefCell<Option<(u64, usize, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
		// (reset_gen, size, buffer)
		pub(crate) static CACHED_INPUT_BUFFER: std::cell::RefCell<Option<(u64, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
}

/// Evaluate a genome export using CPU+GPU hybrid
/// Returns (cross_entropy, accuracy)
/// Compute per-example, per-cluster scores from a trained genome export.
///
/// Shared by `evaluate_genome_hybrid` (CE/accuracy) and `predict_genome_hybrid` (argmax).
/// Tries GPU evaluation (sparse + dense) for each group, falling back to CPU binary search.
pub(crate) fn compute_per_example_scores(
	export: &GenomeExport,
	eval_input_bits: &ram_core::packed_bits::PackedBits,
	packed_eval: &[u64],
	words_per_example: usize,
	num_eval: usize,
	num_clusters: usize,
	_total_input_bits: usize,
	empty_value: f32,
	memory_mode: u8,
	run_seed: u64,
	metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
	sparse_metal: Option<&ram_core::metal_sparse::MetalSparseEvaluator>,
) -> Vec<Vec<f64>>
{
	let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];
	// Mode-correct miss default for sparse CPU-fallback lookups (matches the
	// GPU sparse eval's default_cell_value).
	let miss_default_cell = ram_core::metal_sparse::default_cell_for_mode(memory_mode) as u8;

	let mut dense_idx = 0usize;
	let mut sparse_idx = 0usize;

	for (is_sparse, group_idx, cluster_ids) in &export.group_info
	{
		let group = &export.groups[*group_idx];

		if *is_sparse
		{
			let sparse_export = &export.sparse_exports[sparse_idx];
			sparse_idx += 1;

			let gpu_success = if let Some(sparse_eval) = sparse_metal
			{
				match evaluate_group_sparse_gpu(
					sparse_eval,
					packed_eval,
					&export.connections,
					sparse_export,
					group,
					num_eval,
					words_per_example,
					memory_mode,
					empty_value,
					run_seed,
				)
				{
					Ok(group_scores) =>
					{
						let num_group_clusters = group.cluster_count();
						all_scores
							.par_iter_mut()
							.enumerate()
							.for_each(|(ex_idx, scores)| {
								for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate()
								{
									let score_idx = ex_idx * num_group_clusters + local_cluster;
									scores[cluster_id] = group_scores[score_idx] as f64;
								}
							});
						true
					}
					Err(_) => false,
				}
			}
			else
			{
				false
			};

			if !gpu_success
			{
				// CPU fallback using binary search
				all_scores
					.par_iter_mut()
					.enumerate()
					.for_each(|(ex_idx, scores)| {
						let input_bits = eval_input_bits.packed_row(ex_idx);

						for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate()
						{
							let actual_neurons = if let Some(ref an) = group.actual_neurons
							{
								an[local_cluster] as usize
							}
							else
							{
								group.neurons
							};

							let neuron_base = local_cluster * group.neurons;
							let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

							let mut sum = 0.0f32;
							for n in 0..actual_neurons
							{
								let conn_start = conn_base + n * group.bits;
								let address = ram_core::neuron_memory::compute_address_packed_bytes(
									input_bits,
									&export.connections[conn_start..],
									group.bits,
								);
								let cell = sparse_export.lookup(neuron_base + n, address as u64, miss_default_cell);
								// QSR/PLN: fire the seeded coin (byte-identical to
								// cell_to_weight for deterministic modes). Same
								// (neuron_base+n, address, ex_idx) key as the GPU.
								let rng = ram_core::neuron_memory::qsr_key(
									run_seed,
									(neuron_base + n) as u64,
									address as u64,
									ex_idx as u64,
								);
								sum += cell_to_weight_rng(cell as i64, memory_mode, empty_value, rng);
							}

							scores[cluster_id] = (sum / actual_neurons as f32) as f64;
						}
					});
			}
		}
		else
		{
			let dense_words = &export.dense_exports[dense_idx];
			dense_idx += 1;

			let gpu_success = if let Some(metal_eval) = metal
			{
				match evaluate_group_metal(
					metal_eval,
					packed_eval,
					&export.connections,
					dense_words,
					group,
					num_eval,
					words_per_example,
					memory_mode,
					empty_value,
					run_seed,
				)
				{
					Ok(group_scores) =>
					{
						let num_group_clusters = group.cluster_count();
						all_scores
							.par_iter_mut()
							.enumerate()
							.for_each(|(ex_idx, scores)| {
								for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate()
								{
									let score_idx = ex_idx * num_group_clusters + local_cluster;
									scores[cluster_id] = group_scores[score_idx] as f64;
								}
							});
						true
					}
					Err(_) => false,
				}
			}
			else
			{
				false
			};

			if !gpu_success
			{
				// CPU fallback for dense groups
				all_scores
					.par_iter_mut()
					.enumerate()
					.for_each(|(ex_idx, scores)| {
						let input_bits = eval_input_bits.packed_row(ex_idx);

						for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate()
						{
							let actual_neurons = if let Some(ref an) = group.actual_neurons
							{
								an[local_cluster] as usize
							}
							else
							{
								group.neurons
							};

							let neuron_base = local_cluster * group.neurons;
							let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

							let mut sum = 0.0f32;
							for n in 0..actual_neurons
							{
								let conn_start = conn_base + n * group.bits;
								let address = ram_core::neuron_memory::compute_address_packed_bytes(
									input_bits,
									&export.connections[conn_start..],
									group.bits,
								);
								let cell = read_cell(
									dense_words,
									neuron_base + n,
									address,
									group.words_per_neuron,
								);
								let rng = qsr_key(
									run_seed,
									(neuron_base + n) as u64,
									address as u64,
									ex_idx as u64,
								);
								sum += cell_to_weight_rng(cell, memory_mode, empty_value, rng);
							}

							scores[cluster_id] = (sum / actual_neurons as f32) as f64;
						}
					});
			}
		}
	}

	all_scores
}

#[cfg(test)]
mod pool_size_tests
{
	use super::*;

	// Jetsam root-cause regression (05/07/2026): the sparse per-genome
	// estimate must scale with num_train. Production 46M shape: 250 neurons
	// at up to 100 bits, sr=0.25 → the batch size MUST collapse to ~1 so we
	// never train 10 concurrent multi-GB genomes again.
	#[test]
	fn test_pool_size_46m_collapses_batch()
	{
		let bits = vec![96usize];
		let neurons = vec![250usize];
		let (_, batch) = calculate_pool_size(&bits, &neurons, 1, 23.0, 10, 37_000_000, 0.25);
		assert!(batch <= 2, "46M-scale batch must be 1-2, got {}", batch);
	}

	// Small datasets keep the old cpu_cores-bound behavior (no regression
	// for UNSW/CICIDS-scale flows on the dense/moderate-sparse estimate).
	#[test]
	fn test_pool_size_small_dataset_unchanged()
	{
		let bits = vec![16usize];
		let neurons = vec![250usize];
		let (_, batch) = calculate_pool_size(&bits, &neurons, 1, 23.0, 10, 100_000, 0.25);
		assert!(
			batch >= 4,
			"100K-scale batch should stay multi-genome, got {}",
			batch
		);
	}

	// Budget floor: even a starved memory reading must yield batch >= 1.
	#[test]
	fn test_pool_size_min_one()
	{
		let bits = vec![100usize];
		let neurons = vec![250usize];
		let (_, batch) = calculate_pool_size(&bits, &neurons, 1, 0.5, 10, 46_000_000, 1.0);
		assert_eq!(batch.max(1), batch, "batch must never be 0");
		assert!(batch >= 1);
	}
}
