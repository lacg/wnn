//! Config-group construction + coalescing (neuron bucketing) + cell access.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

use super::*;

/// Check if group coalescing is enabled (set WNN_COALESCE_GROUPS=1)
pub(crate) fn use_coalesced_groups() -> bool
{
	std::env::var("WNN_COALESCE_GROUPS").is_ok()
}

/// Build config groups with optional coalescing based on environment variable
/// When WNN_COALESCE_GROUPS is set, similar neuron counts are bucketed together
/// to reduce GPU dispatch overhead while preserving accuracy through masking
pub fn build_groups(bits_per_cluster: &[usize], neurons_per_cluster: &[usize]) -> Vec<ConfigGroup>
{
	if use_coalesced_groups()
	{
		build_config_groups_coalesced(bits_per_cluster, neurons_per_cluster)
	}
	else
	{
		build_config_groups(bits_per_cluster, neurons_per_cluster)
	}
}

/// Reorganize connections from Python's cluster-order layout to coalesced group layout
///
/// Python generates connections in cluster ID order:
///   [cluster_0_conns, cluster_1_conns, ..., cluster_N_conns]
///   where cluster_i has neurons_per_cluster[i] * bits_per_cluster[i] connections
///
/// Coalesced groups expect connections organized by group with padding:
///   [group_0_cluster_conns, group_1_cluster_conns, ...]
///   where each cluster in group has group.neurons (MAX) * group.bits connections
///   and actual connections are followed by padding (-1) to reach MAX neurons
///
/// Returns: padded connections in group order, ready for coalesced evaluation
pub fn reorganize_connections_for_coalescing(
	original_connections: &[i64],
	bits_per_cluster: &[usize],
	neurons_per_cluster: &[usize],
	groups: &[ConfigGroup],
) -> Vec<i64>
{
	let num_clusters = bits_per_cluster.len();

	// Build mapping: cluster_id -> offset in original_connections
	let mut cluster_offsets = vec![0usize; num_clusters];
	let mut offset = 0;
	for cluster_id in 0..num_clusters
	{
		cluster_offsets[cluster_id] = offset;
		offset += neurons_per_cluster[cluster_id] * bits_per_cluster[cluster_id];
	}

	// Total size needed for coalesced layout
	let total_size: usize = groups.iter().map(|g| g.conn_size()).sum();
	let mut result = vec![-1i64; total_size]; // Initialize with padding value

	// For each group, copy connections for each cluster (with padding)
	let mut write_offset = 0;
	for group in groups
	{
		let max_neurons = group.neurons;
		let bits = group.bits;

		for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate()
		{
			let actual_neurons = if let Some(ref an) = group.actual_neurons
			{
				an[local_idx] as usize
			}
			else
			{
				max_neurons // Uniform case
			};

			// Source: original connections for this cluster
			let src_offset = cluster_offsets[cluster_id];
			let src_size = actual_neurons * bits;

			// Destination: position in coalesced layout
			// Each cluster in group gets max_neurons * bits slots
			let dst_offset = write_offset + local_idx * max_neurons * bits;

			// Copy actual connections
			result[dst_offset..dst_offset + src_size]
				.copy_from_slice(&original_connections[src_offset..src_offset + src_size]);

			// Remaining slots (dst_offset + src_size .. dst_offset + max_neurons * bits)
			// are already -1 (padding)
		}

		write_offset += group.cluster_ids.len() * max_neurons * bits;
	}

	result
}

/// Convert per-neuron bits to per-cluster max bits (for `build_groups`).
///
/// `bits_per_neuron` has length `sum(neurons_per_cluster)` — one entry per neuron.
/// Returns one entry per cluster: the maximum bits among that cluster's neurons.
/// Pattern from `bitwise_ramlm.rs:1391-1400`.
pub(crate) fn per_cluster_max_bits(
	bits_per_neuron: &[usize],
	neurons_per_cluster: &[usize],
) -> Vec<usize>
{
	let mut result = Vec::with_capacity(neurons_per_cluster.len());
	let mut offset = 0;
	for &nc in neurons_per_cluster
	{
		let max_b = bits_per_neuron[offset..offset + nc]
			.iter()
			.copied()
			.max()
			.unwrap_or(0);
		result.push(max_b);
		offset += nc;
	}
	result
}

/// Build per-neuron offset tables for heterogeneous-bits training.
///
/// Returns `(cluster_neuron_starts, neuron_conn_offsets)`:
/// - `cluster_neuron_starts[c]` = first neuron index for cluster `c`
/// - `neuron_conn_offsets[n]` = connection start offset for neuron `n` (cumulative sum of bits)
///
/// Pattern from `bitwise_ramlm.rs:683-704` (`compute_genome_layout`).
pub(crate) fn build_neuron_metadata(
	bits_per_neuron: &[usize],
	neurons_per_cluster: &[usize],
) -> (Vec<usize>, Vec<usize>)
{
	let num_clusters = neurons_per_cluster.len();
	let total_neurons: usize = neurons_per_cluster.iter().sum();

	// cluster_neuron_starts[c] = index of first neuron in cluster c
	let mut cluster_neuron_starts = Vec::with_capacity(num_clusters);
	let mut cumul = 0usize;
	for &nc in neurons_per_cluster
	{
		cluster_neuron_starts.push(cumul);
		cumul += nc;
	}

	// neuron_conn_offsets[n] = start offset in connections array for neuron n
	let mut neuron_conn_offsets = Vec::with_capacity(total_neurons);
	let mut conn_off = 0usize;
	for &b in bits_per_neuron
	{
		neuron_conn_offsets.push(conn_off);
		conn_off += b;
	}

	(cluster_neuron_starts, neuron_conn_offsets)
}

/// Pad per-neuron connections to group layout for GPU dispatch.
///
/// Each neuron's `n_bits` connections are padded to `group.bits` (= cluster max_bits) with
/// connection index 0 (harmless padding). Same pattern as `bitwise_ramlm.rs:804-820`.
///
/// This replaces `reorganize_connections_for_coalescing` when per-neuron bits are heterogeneous.
pub(crate) fn reorganize_connections_for_gpu(
	original_connections: &[i64],
	per_neuron_bits: &[usize],
	neurons_per_cluster: &[usize],
	groups: &[ConfigGroup],
) -> Vec<i64>
{
	let (cluster_neuron_starts, neuron_conn_offsets) =
		build_neuron_metadata(per_neuron_bits, neurons_per_cluster);

	// Total size needed for group layout
	let total_size: usize = groups.iter().map(|g| g.conn_size()).sum();
	// Initialize with -1 (skipped by GPU shader's `if conn_idx >= 0` check)
	let mut result = vec![-1i64; total_size];

	for group in groups
	{
		let max_neurons = group.neurons;
		let max_bits = group.bits;

		for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate()
		{
			let actual_neurons = if let Some(ref an) = group.actual_neurons
			{
				an[local_idx] as usize
			}
			else
			{
				max_neurons
			};

			let neuron_start = cluster_neuron_starts[cluster_id];

			for n in 0..actual_neurons
			{
				let global_n = neuron_start + n;
				let n_bits = per_neuron_bits[global_n];
				let conn_start = neuron_conn_offsets[global_n];

				// Destination in group layout: PREFIX-pad with -1, real connections at END.
				// GPU shader computes address bit i as (max_bits-1-i), so real connections
				// at the end match training's bit positions (actual_bits-1-i).
				let dst = group.conn_offset + local_idx * max_neurons * max_bits + n * max_bits;
				let pad_size = max_bits - n_bits;
				// Prefix is already -1 from initialization; copy real connections after it
				result[dst + pad_size..dst + pad_size + n_bits]
					.copy_from_slice(&original_connections[conn_start..conn_start + n_bits]);
			}
		}
	}

	result
}

/// Configuration group - clusters sharing the same (neurons, bits) config
/// For coalesced groups, neurons is the MAX neurons and actual_neurons stores per-cluster values
#[derive(Clone, Debug)]
pub struct ConfigGroup
{
	pub neurons: usize, // Max neurons (for memory layout)
	pub bits: usize,
	pub words_per_neuron: usize,
	pub cluster_ids: Vec<usize>,          // Global cluster IDs in this group
	pub actual_neurons: Option<Vec<u32>>, // Per-cluster actual neurons (None = all same as neurons)
	pub memory_offset: usize,             // Offset into flattened memory
	pub conn_offset: usize,               // Offset into flattened connections
}

impl ConfigGroup
{
	pub fn new(neurons: usize, bits: usize, cluster_ids: Vec<usize>) -> Self
	{
		let words_per_neuron = Self::dense_words_per_neuron(bits);
		Self {
			neurons,
			bits,
			words_per_neuron,
			cluster_ids,
			actual_neurons: None, // Uniform: all clusters have same neurons
			memory_offset: 0,
			conn_offset: 0,
		}
	}

	/// Create a coalesced group where clusters may have different actual neuron counts
	/// neurons = max neurons for memory allocation
	/// actual_neurons[i] = actual neuron count for cluster_ids[i]
	/// Dense flat-buffer words per neuron. Sparse groups (bits above
	/// SPARSE_THRESHOLD) never materialize dense words — 0 keeps memory_size()
	/// / memory_offset honest AND avoids the 1<<bits overflow at high widths
	/// (debug panic; silent shift-masking garbage in release builds).
	fn dense_words_per_neuron(bits: usize) -> usize
	{
		if bits <= super::metal_state::SPARSE_THRESHOLD
		{
			(1usize << bits).div_ceil(CELLS_PER_WORD)
		}
		else
		{
			0
		}
	}

	pub fn new_coalesced(
		neurons: usize,
		bits: usize,
		cluster_ids: Vec<usize>,
		actual_neurons: Vec<u32>,
	) -> Self
	{
		let words_per_neuron = Self::dense_words_per_neuron(bits);
		Self {
			neurons,
			bits,
			words_per_neuron,
			cluster_ids,
			actual_neurons: Some(actual_neurons),
			memory_offset: 0,
			conn_offset: 0,
		}
	}

	pub fn cluster_count(&self) -> usize
	{
		self.cluster_ids.len()
	}

	pub fn total_neurons(&self) -> usize
	{
		self.cluster_count() * self.neurons
	}

	/// True total neurons (sum of actual neurons if coalesced)
	pub fn true_total_neurons(&self) -> usize
	{
		if let Some(ref actual) = self.actual_neurons
		{
			actual.iter().map(|&n| n as usize).sum()
		}
		else
		{
			self.total_neurons()
		}
	}

	pub fn memory_size(&self) -> usize
	{
		self.total_neurons() * self.words_per_neuron
	}

	pub fn conn_size(&self) -> usize
	{
		self.total_neurons() * self.bits
	}

	/// Is this a coalesced group with per-cluster masking?
	pub fn is_coalesced(&self) -> bool
	{
		self.actual_neurons.is_some()
	}
}

/// Maximum GPU output size: 256M addresses = 1GB output buffer.
/// Beyond this, CPU fallback is used to avoid Metal allocation hangs.
// Halved 28/08/2026 with the u32 -> u64 address widening: the element count
// drops so the PEAK BYTES stay put (256M x 4B == 128M x 8B == 1.0 GB).
pub(crate) const MAX_GPU_ADDRESSES: usize = 128_000_000;

/// Try to compute training addresses on GPU for adaptive training path.
/// Returns None if GPU is unavailable, disabled, or the problem is too large.
pub(crate) fn try_gpu_addresses_adaptive(
	packed_input: &[u64],
	words_per_example: usize,
	per_neuron_bits: &[usize],
	neuron_conn_offsets: &[usize],
	connections: &[i64],
	num_train: usize,
) -> Option<Vec<u64>>
{
	let total_neurons = per_neuron_bits.len();
	if total_neurons < 100
	{
		return None;
	}
	// NO bits guard since 28/08/2026: `compute_addresses` returns Vec<u64> and
	// train_address.metal computes the full u64 address, so train and the u64
	// sparse eval path agree at every width. (Before that a uint buffer
	// truncated bits > 32 mod 2^32 and produced sub-baseline accuracy at
	// b >= 48; the fix was to widen the buffer, not to skip the GPU.)
	// Guard against massive allocations (e.g. 251K neurons × 16K examples = 4B addresses = 16GB).
	// Callers that want larger workloads should use `try_gpu_addresses_for_chunk` in a chunked loop.
	if total_neurons.saturating_mul(num_train) > MAX_GPU_ADDRESSES
	{
		return None;
	}

	let trainer_mutex = crate::get_cached_metal_trainer().ok()?;
	let mut guard = trainer_mutex.lock().ok()?;
	let trainer = guard.as_mut()?;

	let neuron_meta: Vec<NeuronTrainMeta> = (0..total_neurons)
		.map(|n| NeuronTrainMeta {
			bits: per_neuron_bits[n] as u32,
			conn_offset: neuron_conn_offsets[n] as u32,
		})
		.collect();

	trainer
		.compute_addresses(
			packed_input,
			connections,
			&neuron_meta,
			num_train,
			words_per_example,
		)
		.ok()
}

/// Chunked GPU address computation: caller passes a packed-input slice that
/// covers exactly `chunk_num_examples` rows and is responsible for keeping the
/// product `total_neurons * chunk_num_examples` under `MAX_GPU_ADDRESSES`.
///
/// Returns a `Vec<u32>` of length `total_neurons * chunk_num_examples` laid out
/// neuron-major (`addrs[global_n * chunk_num_examples + chunk_local_ex_idx]`).
/// Returns `None` when the GPU path is unavailable or `total_neurons < 100`
/// (CPU fallback wins for small genomes).
pub(crate) fn try_gpu_addresses_for_chunk(
	packed_input_chunk: &[u64],
	words_per_example: usize,
	per_neuron_bits: &[usize],
	neuron_conn_offsets: &[usize],
	connections: &[i64],
	chunk_num_examples: usize,
) -> Option<Vec<u64>>
{
	let total_neurons = per_neuron_bits.len();
	if total_neurons < 100 || chunk_num_examples == 0
	{
		return None;
	}
	// No bits guard — the kernel returns u64 (see try_gpu_addresses_adaptive).
	debug_assert!(
		total_neurons.saturating_mul(chunk_num_examples) <= MAX_GPU_ADDRESSES,
		"try_gpu_addresses_for_chunk: chunk too large ({} * {} > {})",
		total_neurons,
		chunk_num_examples,
		MAX_GPU_ADDRESSES,
	);
	debug_assert_eq!(
		packed_input_chunk.len(),
		chunk_num_examples * words_per_example,
		"try_gpu_addresses_for_chunk: packed_input_chunk size mismatch",
	);

	let trainer_mutex = crate::get_cached_metal_trainer().ok()?;
	let mut guard = trainer_mutex.lock().ok()?;
	let trainer = guard.as_mut()?;

	let neuron_meta: Vec<NeuronTrainMeta> = (0..total_neurons)
		.map(|n| NeuronTrainMeta {
			bits: per_neuron_bits[n] as u32,
			conn_offset: neuron_conn_offsets[n] as u32,
		})
		.collect();

	trainer
		.compute_addresses(
			packed_input_chunk,
			connections,
			&neuron_meta,
			chunk_num_examples,
			words_per_example,
		)
		.ok()
}

/// Read a memory cell value
#[inline]
pub(crate) fn read_cell(
	memory_words: &[i64],
	neuron_idx: usize,
	address: usize,
	words_per_neuron: usize,
) -> i64
{
	let word_idx = address / CELLS_PER_WORD;
	let cell_idx = address % CELLS_PER_WORD;
	let word_offset = neuron_idx * words_per_neuron + word_idx;
	let word = memory_words[word_offset];
	(word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
}

/// Write a memory cell value (atomic, for parallel writes)
#[inline]
pub(crate) fn write_cell_atomic(
	memory_words: &[AtomicI64],
	neuron_idx: usize,
	address: usize,
	value: i64,
	words_per_neuron: usize,
	allow_override: bool,
) -> bool
{
	let word_idx = address / CELLS_PER_WORD;
	let cell_idx = address % CELLS_PER_WORD;
	let word_offset = neuron_idx * words_per_neuron + word_idx;
	let shift = cell_idx * BITS_PER_CELL;
	let mask = CELL_MASK << shift;
	let new_bits = value << shift;

	loop
	{
		let old_word = memory_words[word_offset].load(Ordering::Acquire);
		let old_cell = (old_word >> shift) & CELL_MASK;

		if !allow_override && old_cell != EMPTY
		{
			return false;
		}
		if old_cell == value
		{
			return false;
		}

		let new_word = (old_word & !mask) | new_bits;
		match memory_words[word_offset].compare_exchange(
			old_word,
			new_word,
			Ordering::AcqRel,
			Ordering::Acquire,
		)
		{
			Ok(_) => return true,
			Err(_) => continue,
		}
	}
}

/// Build config groups from per-cluster configuration
///
/// Groups clusters by their (neurons, bits) to enable efficient batch processing.
pub fn build_config_groups(
	bits_per_cluster: &[usize],
	neurons_per_cluster: &[usize],
) -> Vec<ConfigGroup>
{
	use std::collections::HashMap;

	let num_clusters = bits_per_cluster.len();
	let mut config_to_clusters: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

	for cluster_id in 0..num_clusters
	{
		let key = (
			neurons_per_cluster[cluster_id],
			bits_per_cluster[cluster_id],
		);
		config_to_clusters.entry(key).or_default().push(cluster_id);
	}

	let mut groups: Vec<ConfigGroup> = config_to_clusters
		.into_iter()
		.map(|((neurons, bits), cluster_ids)| ConfigGroup::new(neurons, bits, cluster_ids))
		.collect();

	// Sort by (neurons, bits) for deterministic ordering
	groups.sort_by_key(|g| (g.neurons, g.bits));

	// Compute offsets
	let mut memory_offset = 0;
	let mut conn_offset = 0;
	for group in &mut groups
	{
		group.memory_offset = memory_offset;
		group.conn_offset = conn_offset;
		memory_offset += group.memory_size();
		conn_offset += group.conn_size();
	}

	// Log group diversity if enabled (helps diagnose slowdown from too many groups)
	if std::env::var("WNN_GROUP_LOG").is_ok()
	{
		let sparse_count = groups.iter().filter(|g| g.bits > 12).count();
		let dense_count = groups.len() - sparse_count;
		eprintln!(
			"[CONFIG_GROUPS] total={} sparse={} dense={} configs={:?}",
			groups.len(),
			sparse_count,
			dense_count,
			groups
				.iter()
				.map(|g| (g.neurons, g.bits, g.cluster_ids.len()))
				.collect::<Vec<_>>()
		);
	}

	groups
}

/// Bucket neurons into ranges to reduce group diversity
/// Returns the max neurons for the bucket
fn bucket_neurons(neurons: usize) -> usize
{
	// Buckets: 1-5→5, 6-10→10, 11-15→15, 16-20→20, 21-25→25, etc.
	// This gives ~5x fewer unique neuron values
	((neurons + 4) / 5) * 5
}

/// Build config groups with coalescing - buckets similar neuron counts together
/// This reduces the number of GPU dispatches while preserving accuracy through masking.
///
/// Example: If clusters have neurons [5, 6, 7, 8], they bucket into:
///   - 5→5 (bucket 5), 6-10→10 (bucket for 6,7,8)
///   - Instead of 4 groups, we have 2 groups
///
/// For each coalesced group:
///   - neurons = max in bucket (for memory allocation)
///   - actual_neurons[i] = true neuron count for cluster i (for scoring)
pub fn build_config_groups_coalesced(
	bits_per_cluster: &[usize],
	neurons_per_cluster: &[usize],
) -> Vec<ConfigGroup>
{
	use std::collections::HashMap;

	let num_clusters = bits_per_cluster.len();

	// Key: (bucket_max, bits) -> list of (cluster_id, actual_neurons)
	let mut bucket_to_clusters: HashMap<(usize, usize), Vec<(usize, u32)>> = HashMap::new();

	for cluster_id in 0..num_clusters
	{
		let actual = neurons_per_cluster[cluster_id];
		let bucket_max = bucket_neurons(actual);
		let bits = bits_per_cluster[cluster_id];
		let key = (bucket_max, bits);
		bucket_to_clusters
			.entry(key)
			.or_default()
			.push((cluster_id, actual as u32));
	}

	let mut groups: Vec<ConfigGroup> = bucket_to_clusters
		.into_iter()
		.map(|((max_neurons, bits), entries)| {
			let cluster_ids: Vec<usize> = entries.iter().map(|(id, _)| *id).collect();
			let actual_neurons: Vec<u32> = entries.iter().map(|(_, n)| *n).collect();

			// Check if all actual neurons are the same as max (can use uniform mode)
			let all_same = actual_neurons.iter().all(|&n| n as usize == max_neurons);
			if all_same
			{
				ConfigGroup::new(max_neurons, bits, cluster_ids)
			}
			else
			{
				ConfigGroup::new_coalesced(max_neurons, bits, cluster_ids, actual_neurons)
			}
		})
		.collect();

	// Sort by (neurons, bits) for deterministic ordering
	groups.sort_by_key(|g| (g.neurons, g.bits));

	// Compute offsets
	let mut memory_offset = 0;
	let mut conn_offset = 0;
	for group in &mut groups
	{
		group.memory_offset = memory_offset;
		group.conn_offset = conn_offset;
		memory_offset += group.memory_size();
		conn_offset += group.conn_size();
	}

	// Log group diversity if enabled
	if std::env::var("WNN_GROUP_LOG").is_ok()
	{
		let sparse_count = groups.iter().filter(|g| g.bits > 12).count();
		let dense_count = groups.len() - sparse_count;
		let coalesced_count = groups.iter().filter(|g| g.is_coalesced()).count();
		eprintln!(
			"[CONFIG_GROUPS_COALESCED] total={} sparse={} dense={} coalesced={} configs={:?}",
			groups.len(),
			sparse_count,
			dense_count,
			coalesced_count,
			groups
				.iter()
				.map(|g| (g.neurons, g.bits, g.cluster_ids.len(), g.is_coalesced()))
				.collect::<Vec<_>>()
		);
	}

	groups
}

#[cfg(test)]
mod gpu_address_width_tests
{
	use super::*;

	struct Lcg(u64);
	impl Lcg
	{
		fn next(&mut self) -> u64
		{
			self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			self.0
		}
	}

	/// GPU addresses must equal CPU addresses at EVERY bit width, especially
	/// above 32 where a u32 output buffer used to truncate mod 2^32. Train
	/// wrote the truncated key while the u64 sparse eval path computed the
	/// full one, so the two disagreed silently and accuracy fell below
	/// baseline at b >= 48 (27dabcf8). The old fix skipped the GPU; the real
	/// fix widened the buffer, and this test is what keeps it honest.
	#[test]
	fn gpu_addresses_match_cpu_above_32_bits()
	{
		const TOTAL_INPUT_BITS: usize = 128;
		const WORDS: usize = TOTAL_INPUT_BITS / 64;
		const NUM_EX: usize = 64;
		// >= 100 neurons or try_gpu_addresses_adaptive declines by design
		const NEURONS: usize = 128;

		let mut rng = Lcg(0xC0FFEE_1234_5678);
		let packed_input: Vec<u64> = (0..NUM_EX * WORDS).map(|_| rng.next()).collect();

		// Widths that straddle the old u32 boundary, including 64.
		for &bits in &[16usize, 32, 33, 40, 50, 64]
		{
			let per_neuron_bits = vec![bits; NEURONS];
			let mut connections: Vec<i64> = Vec::with_capacity(NEURONS * bits);
			let mut neuron_conn_offsets = Vec::with_capacity(NEURONS);
			for _n in 0..NEURONS
			{
				neuron_conn_offsets.push(connections.len());
				for _b in 0..bits
				{
					connections.push((rng.next() % TOTAL_INPUT_BITS as u64) as i64);
				}
			}

			let gpu = match try_gpu_addresses_adaptive(
				&packed_input, WORDS, &per_neuron_bits, &neuron_conn_offsets,
				&connections, NUM_EX,
			)
			{
				Some(v) => v,
				// No Metal device in this environment — nothing to compare.
				None => return,
			};
			assert_eq!(gpu.len(), NEURONS * NUM_EX, "bits={bits}: address count");

			for n in 0..NEURONS
			{
				let conns = &connections[neuron_conn_offsets[n]..neuron_conn_offsets[n] + bits];
				for ex in 0..NUM_EX
				{
					let row = &packed_input[ex * WORDS..(ex + 1) * WORDS];
					// CPU reference: MSB-first over the same connections.
					let mut expect: u64 = 0;
					for (i, &c) in conns.iter().enumerate()
					{
						let ci = c as usize;
						if (row[ci / 64] >> (ci % 64)) & 1 == 1
						{
							expect |= 1u64 << (bits - 1 - i);
						}
					}
					let got = gpu[n * NUM_EX + ex];
					assert_eq!(
						got, expect,
						"bits={bits} neuron={n} example={ex}: GPU {got} != CPU {expect}"
					);
				}
			}

			// A width above 32 must actually produce addresses that do NOT fit
			// in u32, otherwise the test would pass even against a u32 buffer.
			if bits > 32
			{
				assert!(
					gpu.iter().any(|&a| a > u32::MAX as u64),
					"bits={bits}: no address exceeded u32::MAX — the test cannot \
					 distinguish a truncating buffer from a correct one"
				);
			}
		}
	}
}
