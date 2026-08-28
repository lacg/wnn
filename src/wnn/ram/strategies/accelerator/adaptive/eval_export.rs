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
	/// One entry per group, IN GROUP ORDER, so a group's index is its position
	/// in this Vec. The middle field is the group's SUB-INDEX within its own
	/// export vector — `sparse_exports[sub]` or `dense_exports[sub]` — NOT the
	/// group index. The two coincide only when every group is the same kind,
	/// which is why storing group_idx here went unnoticed: the marker path made
	/// everything sparse. Mixed dense/sparse genomes (per-class bits straddling
	/// SPARSE_THRESHOLD) index the wrong export with the old convention.
	pub group_info: Vec<(bool, usize, Vec<usize>)>,
	/// Dense group exports: memory words
	pub dense_exports: Vec<Vec<i64>>,
	/// Parallel to `dense_exports`: 1 bit per (neuron, address), set iff the
	/// cell was addressed in training. A dense read otherwise cannot tell an
	/// untouched cell from a learned tie — `oi_bin_to_cell` collapses obs==0,
	/// obs==1&net<0 and obs>=2&net==0 all onto WEAK_FALSE — so this is the
	/// dense analogue of a sparse lookup miss, and what lets coverage-aware
	/// scoring work below SPARSE_THRESHOLD. Empty when not tracked.
	pub dense_coverage: Vec<Vec<u64>>,
	/// Sparse group exports: sorted arrays for binary search
	pub sparse_exports: Vec<SparseGpuExport>,
	/// Config groups for this genome
	pub groups: Vec<ConfigGroup>,
}

/// Materialise a per-neuron sparse export into DENSE packed words plus a
/// coverage bitmap.
///
/// Below `SPARSE_THRESHOLD` a hash export is the wrong shape: at b=4 it costs
/// ~152 B/neuron against the dense array's 4 B (36x) and turns an O(1) indexed
/// read into an O(log n) binary search. The crossover is almost exactly the
/// threshold — at b=34 the ratio inverts to ~2e4 the other way.
///
/// The coverage bitmap comes free from the same pass: a key present in the
/// export IS the "this cell was addressed" signal that a dense word array
/// cannot otherwise express.
pub(crate) fn densify_sparse_export(
	keys: &[u64],
	values: &[u8],
	offsets: &[u32],
	counts: &[u32],
	num_neurons: usize,
	bits: usize,
	memory_mode: u8,
) -> (Vec<i64>, Vec<u64>)
{
	use ram_core::neuron_memory::{BITS_PER_CELL, CELLS_PER_WORD, CELL_MASK};
	let addresses_per_neuron = 1usize << bits;
	let words_per_neuron = addresses_per_neuron.div_ceil(CELLS_PER_WORD);
	let empty_word = ram_core::neuron_memory::empty_word_for_mode(memory_mode);
	let mut words = vec![empty_word; num_neurons * words_per_neuron];
	let mut coverage = vec![0u64; (num_neurons * addresses_per_neuron).div_ceil(64)];

	for n in 0..num_neurons
	{
		let start = *offsets.get(n).unwrap_or(&0) as usize;
		let cnt = *counts.get(n).unwrap_or(&0) as usize;
		for i in start..start + cnt
		{
			let addr = keys[i] as usize;
			debug_assert!(addr < addresses_per_neuron, "address {addr} exceeds 2^{bits}");
			if addr >= addresses_per_neuron
			{
				continue;
			}
			let cell = values[i] as i64;
			let word_idx = addr / CELLS_PER_WORD;
			let cell_idx = addr % CELLS_PER_WORD;
			let shift = cell_idx * BITS_PER_CELL;
			let w = &mut words[n * words_per_neuron + word_idx];
			*w = (*w & !(CELL_MASK << shift)) | (cell << shift);
			let cov_idx = n * addresses_per_neuron + addr;
			coverage[cov_idx / 64] |= 1u64 << (cov_idx % 64);
		}
	}
	(words, coverage)
}

/// True iff (neuron, address) was addressed during training. An EMPTY bitmap
/// means coverage was not tracked, and every cell reads as covered so callers
/// degrade to today's behaviour instead of silently scoring everything as
/// uncovered.
#[inline]
pub(crate) fn dense_is_covered(coverage: &[u64], neuron_in_group: usize, address: usize,
                               addresses_per_neuron: usize) -> bool
{
	if coverage.is_empty()
	{
		return true;
	}
	let idx = neuron_in_group * addresses_per_neuron + address;
	coverage.get(idx / 64).is_some_and(|w| (w >> (idx % 64)) & 1 == 1)
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
			dense_coverage: Vec::new(),
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

	for (group, memory) in groups.iter().zip(memories.iter())
	{
		let is_sparse = memory.is_sparse();
		// SUB-index within this group's own export vector (see GenomeExport).
		let sub_idx = if is_sparse { sparse_exports.len() } else { dense_exports.len() };
		group_info.push((is_sparse, sub_idx, group.cluster_ids.clone()));

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
		// The classic path's GroupDenseMemory carries no coverage bitmap. It is
		// the marker path's error fallback and has not fired in production
		// (PATH2_FALLBACK=0); the guard at the top of compute_per_example_scores
		// says so loudly if it ever runs with coverage_aware on.
		dense_coverage: Vec::new(),
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
	coverage_aware: bool,
	metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
	sparse_metal: Option<&ram_core::metal_sparse::MetalSparseEvaluator>,
) -> Vec<Vec<f64>>
{
	let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];
	// Mode-correct miss default for sparse CPU-fallback lookups (matches the
	// GPU sparse eval's default_cell_value). Under `coverage_aware` a miss
	// resolves to cell 0 (weight 0.0 in every mode) instead of the mode's
	// "empty" cell, so ignorance no longer outranks a learned rejection —
	// docs/COVERAGE_AWARE_SCORER_SPEC.md. Dense groups below never consult it.
	let miss_default_cell =
		ram_core::metal_sparse::default_cell_for_coverage(memory_mode, coverage_aware) as u8;


	// A DENSE group cannot honour coverage-aware scoring: it reads the cell
	// straight out of the packed word, and `oi_bin_to_cell` collapses obs==0,
	// obs==1&net<0 and obs>=2&net==0 all onto WEAK_FALSE — so "never addressed"
	// is indistinguishable from "learned tie" and the 0.25 pedestal cannot be
	// identified at read time. On the LIVE path this never bites: the marker
	// path (Option B) exports every group sparse whatever the bit width, and
	// PATH2_FALLBACK has not fired in production. But if the dense fallback ever
	// does run with the flag on, the flag would silently do nothing and the run
	// would look like a coverage-aware result while scoring the pedestal.
	// Say so, loudly, once. See docs/COVERAGE_AWARE_SCORER_SPEC.md.
	if coverage_aware && export.group_info.iter().any(|(is_sparse, _, _)| !*is_sparse)
	{
		static WARNED: std::sync::Once = std::sync::Once::new();
		WARNED.call_once(|| {
			eprintln!(
				"[COVERAGE_AWARE] ⚠️ DENSE group present — coverage-aware scoring is \
				 INERT for it (a dense read cannot distinguish an untouched cell from \
				 a learned tie). Results mixing dense groups are NOT coverage-aware. \
				 This should not happen on the marker path; investigate PATH2_FALLBACK."
			);
		});
	}

	for (group_idx, (is_sparse, sub_idx, cluster_ids)) in export.group_info.iter().enumerate()
	{
		let group = &export.groups[group_idx];

		if *is_sparse
		{
			let sparse_export = &export.sparse_exports[*sub_idx];

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
					coverage_aware,
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
			let dense_words = &export.dense_exports[*sub_idx];
			let dense_cov: &[u64] = export
				.dense_coverage
				.get(*sub_idx)
				.map_or(&[][..], |v| v.as_slice());
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
					dense_cov,
					coverage_aware,
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

							let addresses_per_neuron = 1usize << group.bits;
							let mut sum = 0.0f32;
							for n in 0..actual_neurons
							{
								let conn_start = conn_base + n * group.bits;
								let address = ram_core::neuron_memory::compute_address_packed_bytes(
									input_bits,
									&export.connections[conn_start..],
									group.bits,
								);
								// Coverage-aware: an address this neuron never saw is NO
								// EVIDENCE, not weak evidence. Skipping the add scores it
								// 0.0 while the denominator stays actual_neurons, exactly
								// matching the sparse rule (a miss resolves to cell 0,
								// whose weight is 0.0 in every mode).
								if coverage_aware
									&& !dense_is_covered(
										dense_cov,
										neuron_base + n,
										address,
										addresses_per_neuron,
									)
								{
									continue;
								}
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

#[cfg(test)]
mod densify_tests
{
	use super::*;
	use ram_core::neuron_memory::{
		cell_to_weight, empty_word_for_mode, QUAD_WEAK_FALSE, QUAD_TRUE, QUAD_WEIGHTED,
	};

	/// Densifying a low-bits export must place every trained cell at its address,
	/// leave every other cell at the mode's empty word, and mark exactly the
	/// trained addresses as covered.
	#[test]
	fn densify_places_cells_and_records_exactly_the_trained_addresses()
	{
		let bits = 4; // 16 addresses
		let num_neurons = 2;
		// neuron 0 trained at addresses 3 and 9; neuron 1 at address 0 only.
		let keys = vec![3u64, 9, 0];
		let values = vec![QUAD_TRUE as u8, QUAD_WEAK_FALSE as u8, QUAD_TRUE as u8];
		let offsets = vec![0u32, 2];
		let counts = vec![2u32, 1];

		let (words, coverage) = densify_sparse_export(
			&keys, &values, &offsets, &counts, num_neurons, bits, QUAD_WEIGHTED,
		);

		let wpn = 16usize.div_ceil(ram_core::neuron_memory::CELLS_PER_WORD);
		assert_eq!(words.len(), num_neurons * wpn);
		assert_eq!(read_cell(&words, 0, 3, wpn), QUAD_TRUE);
		assert_eq!(read_cell(&words, 0, 9, wpn), QUAD_WEAK_FALSE);
		assert_eq!(read_cell(&words, 1, 0, wpn), QUAD_TRUE);
		// An untrained address keeps the empty word's cell.
		let empty_cell = (empty_word_for_mode(QUAD_WEIGHTED)
			& ram_core::neuron_memory::CELL_MASK) as i64;
		assert_eq!(read_cell(&words, 0, 7, wpn), empty_cell);

		// Coverage is EXACTLY the trained set — 3 of 32 cells.
		let set: u32 = coverage.iter().map(|w| w.count_ones()).sum();
		assert_eq!(set, 3, "coverage must mark exactly the trained addresses");
		for (n, a, want) in [(0, 3, true), (0, 9, true), (1, 0, true),
		                     (0, 7, false), (1, 5, false), (0, 0, false)]
		{
			assert_eq!(dense_is_covered(&coverage, n, a, 16), want, "n{n} a{a}");
		}
	}

	/// THE point of the bitmap: address 9 was trained to WEAK_FALSE and address 7
	/// was never trained, yet both read as WEAK_FALSE from the packed word — the
	/// commit lattice collapses them. Only coverage separates "learned a weak
	/// negative" from "never saw it", which is what stops ignorance outscoring a
	/// learned rejection at low bits.
	#[test]
	fn coverage_separates_a_learned_weak_false_from_an_untouched_cell()
	{
		let bits = 4;
		let keys = vec![9u64];
		let values = vec![QUAD_WEAK_FALSE as u8];
		let (words, coverage) =
			densify_sparse_export(&keys, &values, &vec![0u32], &vec![1u32], 1, bits, QUAD_WEIGHTED);
		let wpn = 16usize.div_ceil(ram_core::neuron_memory::CELLS_PER_WORD);

		// Indistinguishable by cell value...
		assert_eq!(read_cell(&words, 0, 9, wpn), read_cell(&words, 0, 7, wpn));
		// ...and both would score the 0.25 pedestal.
		assert_eq!(cell_to_weight(read_cell(&words, 0, 7, wpn), QUAD_WEIGHTED, 0.0), 0.25);
		// Coverage is the ONLY thing that tells them apart.
		assert!(dense_is_covered(&coverage, 0, 9, 16), "trained cell is covered");
		assert!(!dense_is_covered(&coverage, 0, 7, 16), "untouched cell is not");
	}

	/// An empty bitmap means "not tracked" and must read as fully covered, so a
	/// caller degrades to today's behaviour rather than scoring everything as
	/// uncovered (which would zero every genome).
	#[test]
	fn untracked_coverage_reads_as_fully_covered()
	{
		for a in 0..16
		{
			assert!(dense_is_covered(&[], 0, a, 16));
		}
	}
}
