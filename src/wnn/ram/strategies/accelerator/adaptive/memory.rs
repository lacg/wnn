//! Group memory backends: dense bit-packed, sparse DashMap (+atomic), batch forward/train.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

use super::*;

/// Forward pass for adaptive architecture
///
/// Processes each config group efficiently, then scatters results to output.
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits]
///   connections_flat: All groups' connections concatenated
///   memory_words: All groups' memory concatenated
///   groups: Config groups with cluster assignments
///   num_examples: Number of input examples
///   total_input_bits: Total input bits per example
///   num_clusters: Total number of clusters (vocabulary size)
///
/// Returns: [num_examples * num_clusters] probabilities
pub fn forward_batch_adaptive(
	input_bits_flat: &[bool],
	connections_flat: &[i64],
	memory_words: &[i64],
	groups: &[ConfigGroup],
	num_examples: usize,
	total_input_bits: usize,
	num_clusters: usize,
	empty_value: f32,
) -> Vec<f32>
{
	let mut probs = vec![0.0f32; num_examples * num_clusters];

	// Build reverse mapping: global_cluster_id -> (group_idx, local_cluster_idx)
	let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
	for (group_idx, group) in groups.iter().enumerate()
	{
		for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate()
		{
			cluster_to_group[cluster_id] = (group_idx, local_idx);
		}
	}

	// Process all examples in parallel
	probs
		.par_chunks_mut(num_clusters)
		.enumerate()
		.for_each(|(ex_idx, ex_probs)| {
			let input_start = ex_idx * total_input_bits;
			let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

			// Process each config group
			for group in groups
			{
				let neurons = group.neurons;
				let bits = group.bits;
				let words_per_neuron = group.words_per_neuron;
				let group_memory = &memory_words[group.memory_offset..];
				let group_conns = &connections_flat[group.conn_offset..];

				// For each cluster in this group
				for (local_idx, &global_cluster_id) in group.cluster_ids.iter().enumerate()
				{
					// Use actual neurons if coalesced, otherwise MAX (uniform case)
					let actual_neurons = if let Some(ref an) = group.actual_neurons
					{
						an[local_idx] as usize
					}
					else
					{
						neurons
					};

					let start_neuron = local_idx * neurons; // Use MAX for memory layout
					let mut count_true = 0u32;
					let mut count_empty = 0u32;

					for neuron_offset in 0..actual_neurons
					{
						// Only iterate actual neurons
						let local_neuron = start_neuron + neuron_offset;
						let conn_start = local_neuron * bits;
						let connections = &group_conns[conn_start..conn_start + bits];

						let address = compute_address(input_bits, connections, bits);
						let cell_value = read_cell(group_memory, local_neuron, address, words_per_neuron);

						if cell_value == TRUE
						{
							count_true += 1;
						}
						else if cell_value == EMPTY
						{
							count_empty += 1;
						}
					}

					// Divide by actual neurons for correct probability
					ex_probs[global_cluster_id] =
						(count_true as f32 + empty_value * count_empty as f32) / actual_neurons as f32;
				}
			}
		});

	probs
}

/// Training for adaptive architecture
///
/// Two-phase training: TRUE first, then FALSE (to ensure TRUE priority).
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits]
///   true_clusters: [num_examples] global cluster indices
///   false_clusters_flat: [num_examples * num_negatives] global cluster indices
///   connections_flat: All groups' connections concatenated
///   memory_words: All groups' memory concatenated (mutable)
///   groups: Config groups with cluster assignments
///
/// Returns: Number of cells modified
pub fn train_batch_adaptive(
	input_bits_flat: &[bool],
	true_clusters: &[i64],
	false_clusters_flat: &[i64],
	connections_flat: &[i64],
	memory_words: &mut [i64],
	groups: &[ConfigGroup],
	num_examples: usize,
	total_input_bits: usize,
	num_negatives: usize,
	num_clusters: usize,
	allow_override: bool,
) -> usize
{
	// Build reverse mapping: global_cluster_id -> (group_idx, local_cluster_idx)
	let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
	for (group_idx, group) in groups.iter().enumerate()
	{
		for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate()
		{
			cluster_to_group[cluster_id] = (group_idx, local_idx);
		}
	}

	// Convert memory to atomic for thread-safe writes
	let atomic_memory: &[AtomicI64] = unsafe {
		std::slice::from_raw_parts(
			memory_words.as_ptr() as *const AtomicI64,
			memory_words.len(),
		)
	};

	// Phase 1: Write all TRUEs
	let true_modified: usize = (0..num_examples)
		.into_par_iter()
		.map(|ex_idx| {
			let input_start = ex_idx * total_input_bits;
			let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

			let true_cluster = true_clusters[ex_idx] as usize;
			let (group_idx, local_cluster) = cluster_to_group[true_cluster];
			let group = &groups[group_idx];

			let neurons = group.neurons; // MAX for memory layout
			let bits = group.bits;
			let words_per_neuron = group.words_per_neuron;
			let start_neuron = local_cluster * neurons;
			let group_conns = &connections_flat[group.conn_offset..];

			// Use actual neurons if coalesced, otherwise MAX
			let actual_neurons = if let Some(ref an) = group.actual_neurons
			{
				an[local_cluster] as usize
			}
			else
			{
				neurons
			};

			let mut modified = 0usize;
			for neuron_offset in 0..actual_neurons
			{
				// Only iterate actual neurons
				let local_neuron = start_neuron + neuron_offset;
				let conn_start = local_neuron * bits;
				let connections = &group_conns[conn_start..conn_start + bits];

				let address = compute_address(input_bits, connections, bits);
				let _global_neuron_offset = group.memory_offset / words_per_neuron + local_neuron;

				if write_cell_atomic(
					&atomic_memory[group.memory_offset..],
					local_neuron,
					address,
					TRUE,
					words_per_neuron,
					allow_override,
				)
				{
					modified += 1;
				}
			}
			modified
		})
		.sum();

	// Phase 2: Write all FALSEs (skip if already TRUE)
	let false_modified: usize = (0..num_examples)
		.into_par_iter()
		.map(|ex_idx| {
			let input_start = ex_idx * total_input_bits;
			let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];
			let true_cluster = true_clusters[ex_idx] as usize;

			let false_start = ex_idx * num_negatives;
			let mut modified = 0usize;

			for neg_idx in 0..num_negatives
			{
				let false_cluster = false_clusters_flat[false_start + neg_idx] as usize;
				if false_cluster == true_cluster
				{
					continue;
				}

				let (group_idx, local_cluster) = cluster_to_group[false_cluster];
				let group = &groups[group_idx];

				let neurons = group.neurons; // MAX for memory layout
				let bits = group.bits;
				let words_per_neuron = group.words_per_neuron;
				let start_neuron = local_cluster * neurons;
				let group_conns = &connections_flat[group.conn_offset..];

				// Use actual neurons if coalesced, otherwise MAX
				let actual_neurons = if let Some(ref an) = group.actual_neurons
				{
					an[local_cluster] as usize
				}
				else
				{
					neurons
				};

				for neuron_offset in 0..actual_neurons
				{
					// Only iterate actual neurons
					let local_neuron = start_neuron + neuron_offset;
					let conn_start = local_neuron * bits;
					let connections = &group_conns[conn_start..conn_start + bits];

					let address = compute_address(input_bits, connections, bits);

					if write_cell_atomic(
						&atomic_memory[group.memory_offset..],
						local_neuron,
						address,
						FALSE,
						words_per_neuron,
						false, // Never override TRUE
					)
					{
						modified += 1;
					}
				}
			}
			modified
		})
		.sum();

	true_modified + false_modified
}

/// Dense memory for a config group (bit-packed, fast for bits <= 12)
/// Uses atomic operations for thread-safe concurrent writes.
pub(crate) struct GroupDenseMemory
{
	/// Bit-packed memory words [total_neurons * words_per_neuron]
	words: Vec<AtomicI64>,
	words_per_neuron: usize,
	/// Number of addresses per neuron (= 1 << bits). Used for the OI counter
	/// buffer layout, which is one AtomicU32 per (neuron, address) and does
	/// not share the 31-cells-per-word packing.
	addresses_per_neuron: usize,
	/// Order-independent training counter buffer. Allocated by
	/// `init_oi_counters()`, consumed and dropped by `commit_oi()`.
	/// None outside of an OI training pass.
	counters: Option<Vec<std::sync::atomic::AtomicU32>>,
}

impl GroupDenseMemory
{
	pub(crate) fn new(num_neurons: usize, bits: usize, memory_mode: u8) -> Self
	{
		let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
		let addresses_per_neuron = 1usize << bits;
		let total_words = num_neurons * words_per_neuron;
		let empty_word = ram_core::neuron_memory::empty_word_for_mode(memory_mode);
		Self {
			words: (0..total_words)
				.map(|_| AtomicI64::new(empty_word))
				.collect(),
			words_per_neuron,
			addresses_per_neuron,
			counters: None,
		}
	}

	fn num_neurons(&self) -> usize
	{
		self.words.len() / self.words_per_neuron
	}

	/// Allocate the OI counter buffer (idempotent — no-op if already allocated).
	/// Called once before an order-independent training pass.
	pub fn init_oi_counters(&mut self)
	{
		if self.counters.is_some()
		{
			return;
		}
		let n = self.num_neurons() * self.addresses_per_neuron;
		let mut buf = Vec::with_capacity(n);
		for _ in 0..n
		{
			buf.push(std::sync::atomic::AtomicU32::new(
				ram_core::neuron_memory::OI_INITIAL,
			));
		}
		self.counters = Some(buf);
	}

	/// Order-independent nudge: accumulates ±weight into the per-cell counter.
	/// Must be called between `init_oi_counters()` and `commit_oi()`.
	#[inline]
	pub fn nudge_oi(&self, neuron_idx: usize, address: usize, target_true: bool, weight: u32)
		-> bool
	{
		let counters = self
			.counters
			.as_ref()
			.expect("nudge_oi called without init_oi_counters");
		let idx = neuron_idx * self.addresses_per_neuron + address;
		let delta: i32 = if target_true
		{
			weight as i32
		}
		else
		{
			-(weight as i32)
		};
		ram_core::neuron_memory::oi_nudge_atomic(&counters[idx], delta);
		true
	}

	/// Commit pass: bin every touched counter into its 2-bit cell, then free
	/// the counter buffer. After commit, the dense memory is identical in
	/// shape to a normally-trained memory (`forward`, exports, etc. unchanged).
	pub fn commit_oi(&mut self)
	{
		let Some(counters) = self.counters.take()
		else
		{
			return;
		};
		let n = self.num_neurons();
		for neuron_idx in 0..n
		{
			let n_base = neuron_idx * self.addresses_per_neuron;
			for address in 0..self.addresses_per_neuron
			{
				let packed = counters[n_base + address].load(Ordering::Relaxed);
				// Skip cells that were never touched: they keep their initial
				// (QUAD_WEAK_FALSE) value from `empty_word_for_mode`.
				if packed == ram_core::neuron_memory::OI_INITIAL
				{
					continue;
				}
				let cell = ram_core::neuron_memory::oi_bin_to_cell(packed);
				let word_idx = address / CELLS_PER_WORD;
				let cell_idx = address % CELLS_PER_WORD;
				let word_offset = neuron_idx * self.words_per_neuron + word_idx;
				let shift = cell_idx * BITS_PER_CELL;
				let mask = CELL_MASK << shift;
				let old = self.words[word_offset].load(Ordering::Relaxed);
				let new_word = (old & !mask) | (cell << shift);
				self.words[word_offset].store(new_word, Ordering::Relaxed);
			}
		}
	}

	/// Export memory words for Metal GPU (read-only snapshot)
	fn export_for_metal(&self) -> Vec<i64>
	{
		self
			.words
			.iter()
			.map(|w| w.load(Ordering::Relaxed))
			.collect()
	}

	#[inline]
	pub(crate) fn read(&self, neuron_idx: usize, address: usize) -> i64
	{
		let word_idx = address / CELLS_PER_WORD;
		let cell_idx = address % CELLS_PER_WORD;
		let word_offset = neuron_idx * self.words_per_neuron + word_idx;
		let word = self.words[word_offset].load(Ordering::Relaxed);
		(word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
	}

	/// Thread-safe atomic write using compare-and-swap
	///
	/// TRUE-wins-over-FALSE semantics:
	/// - TRUE can be written over EMPTY or FALSE
	/// - FALSE can only be written over EMPTY
	/// - TRUE cannot be overwritten by FALSE
	#[inline]
	fn write(&self, neuron_idx: usize, address: usize, value: i64, allow_override: bool) -> bool
	{
		let word_idx = address / CELLS_PER_WORD;
		let cell_idx = address % CELLS_PER_WORD;
		let word_offset = neuron_idx * self.words_per_neuron + word_idx;
		let shift = cell_idx * BITS_PER_CELL;
		let mask = CELL_MASK << shift;

		loop
		{
			let old_word = self.words[word_offset].load(Ordering::Relaxed);
			let old_cell = (old_word >> shift) & CELL_MASK;

			// No change needed if same value
			if old_cell == value
			{
				return false;
			}

			// TRUE wins over FALSE: don't overwrite TRUE with FALSE
			if old_cell == TRUE && value == FALSE
			{
				return false;
			}

			// If not allow_override:
			// - TRUE can overwrite EMPTY or FALSE (TRUE wins)
			// - FALSE can only overwrite EMPTY
			if !allow_override && value == FALSE && old_cell != EMPTY
			{
				return false;
			}

			let new_word = (old_word & !mask) | (value << shift);
			if self.words[word_offset]
				.compare_exchange_weak(old_word, new_word, Ordering::Relaxed, Ordering::Relaxed)
				.is_ok()
			{
				return true;
			}
			// CAS failed, retry
		}
	}

	/// Thread-safe atomic nudge for quad modes (CAS loop).
	/// Moves cell one step toward target: +1 if target_true, -1 if target_false.
	/// Clamps to [0, 3] (QUAD_FALSE..QUAD_TRUE).
	#[inline]
	fn nudge(&self, neuron_idx: usize, address: usize, target_true: bool) -> bool
	{
		let word_idx = address / CELLS_PER_WORD;
		let cell_idx = address % CELLS_PER_WORD;
		let word_offset = neuron_idx * self.words_per_neuron + word_idx;
		let shift = cell_idx * BITS_PER_CELL;
		let mask = CELL_MASK << shift;
		let delta = 2 * (target_true as i64) - 1; // +1 or -1

		loop
		{
			let old_word = self.words[word_offset].load(Ordering::Relaxed);
			let old_cell = (old_word >> shift) & CELL_MASK;

			let new_cell = (old_cell + delta).clamp(
				ram_core::neuron_memory::QUAD_FALSE,
				ram_core::neuron_memory::QUAD_TRUE,
			);
			if new_cell == old_cell
			{
				return false; // already at boundary
			}

			let new_word = (old_word & !mask) | (new_cell << shift);
			if self.words[word_offset]
				.compare_exchange_weak(old_word, new_word, Ordering::Relaxed, Ordering::Relaxed)
				.is_ok()
			{
				return true;
			}
			// CAS failed, retry
		}
	}
}

/// GPU-compatible sparse memory export (sorted arrays for binary search)
#[derive(Clone)]
pub struct SparseGpuExport
{
	/// Sorted keys for all neurons, concatenated
	pub keys: Vec<u64>,
	/// Values corresponding to keys (0=FALSE, 1=TRUE)
	pub values: Vec<u8>,
	/// Start offset for each neuron in keys array
	pub offsets: Vec<u32>,
	/// Number of entries for each neuron
	pub counts: Vec<u32>,
	/// Total number of neurons
	pub num_neurons: usize,
}

impl SparseGpuExport
{
	/// CPU binary search lookup. `miss_default` is the cell an ABSENT address
	/// reads as — memory-mode-dependent (QUAD: 1 = WEAK_FALSE, TERNARY: 2 =
	/// EMPTY; compute via `default_cell_for_mode`). Fixed 07/07/2026: this
	/// returned hardcoded TERNARY EMPTY(2) on miss, which QUAD cell_to_weight
	/// reads as WEAK_TRUE (0.75) instead of WEAK_FALSE (0.25) — diverging
	/// from the GPU sparse eval's default_cell_value semantics.
	#[inline]
	pub fn lookup(&self, neuron_idx: usize, address: u64, miss_default: u8) -> u8
	{
		let start = self.offsets[neuron_idx] as usize;
		let count = self.counts[neuron_idx] as usize;

		if count == 0
		{
			return miss_default;
		}

		let end = start + count;
		let keys_slice = &self.keys[start..end];

		match keys_slice.binary_search(&address)
		{
			Ok(idx) => self.values[start + idx],
			Err(_) => miss_default,
		}
	}

	/// Total memory size in bytes
	pub fn memory_size(&self) -> usize
	{
		self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4 + self.counts.len() * 4
	}
}

/// Sparse memory for a config group (concurrent hash-based, for bits > 12)
/// Uses DashMap for thread-safe concurrent access during parallel training.
pub(crate) struct GroupSparseMemory
{
	/// Per-neuron concurrent hash maps: address -> cell value
	pub(crate) neurons: Vec<DashMap<u64, u8>>,
	/// Default cell value for unvisited addresses (EMPTY_U8=2 for ternary, 1=QUAD_WEAK_FALSE for quad)
	pub(crate) default_empty: u8,
	/// Order-independent training: per-neuron counter maps storing packed
	/// (obs, net) per address. None outside an OI pass.
	counter_maps: Option<Vec<DashMap<u64, u32>>>,
}

impl GroupSparseMemory
{
	pub(crate) fn new(num_neurons: usize, memory_mode: u8) -> Self
	{
		let default_empty = match memory_mode
		{
			ram_core::neuron_memory::QUAD_BINARY
			| ram_core::neuron_memory::QUAD_WEIGHTED
			| ram_core::neuron_memory::QSR => 1, // QUAD_WEAK_FALSE
			ram_core::neuron_memory::BINARY => 0, // classical 1-bit: unwritten = FALSE
			_ => EMPTY as u8,                     // 2
		};
		Self {
			neurons: (0..num_neurons).map(|_| DashMap::new()).collect(),
			default_empty,
			counter_maps: None,
		}
	}

	/// Allocate OI counter maps (one DashMap per neuron). Idempotent.
	pub fn init_oi_counters(&mut self)
	{
		if self.counter_maps.is_some()
		{
			return;
		}
		self.counter_maps = Some((0..self.neurons.len()).map(|_| DashMap::new()).collect());
	}

	/// Order-independent nudge: apply ±weight to the packed counter for this
	/// (neuron, address) via DashMap entry API. Entry-API holds a bucket lock
	/// during the closure, making the read-modify-write atomic for that key.
	#[inline]
	pub fn nudge_oi(&self, neuron_idx: usize, address: u64, target_true: bool, weight: u32) -> bool
	{
		let maps = self
			.counter_maps
			.as_ref()
			.expect("nudge_oi called without init_oi_counters");
		let delta: i32 = if target_true
		{
			weight as i32
		}
		else
		{
			-(weight as i32)
		};
		let map = &maps[neuron_idx];
		match map.entry(address)
		{
			dashmap::mapref::entry::Entry::Occupied(mut e) =>
			{
				let new = ram_core::neuron_memory::oi_apply_nudge(*e.get(), delta);
				e.insert(new);
			}
			dashmap::mapref::entry::Entry::Vacant(e) =>
			{
				let new =
					ram_core::neuron_memory::oi_apply_nudge(ram_core::neuron_memory::OI_INITIAL, delta);
				e.insert(new);
			}
		}
		true
	}

	/// Commit pass: bin each counter to its 2-bit cell value, write into
	/// the cell map, then drop the counter maps. Entries that bin back to
	/// the default_empty value are not inserted (matches existing convention
	/// that absent entries == default_empty).
	pub fn commit_oi(&mut self)
	{
		let Some(counter_maps) = self.counter_maps.take()
		else
		{
			return;
		};
		for (neuron_idx, ctr_map) in counter_maps.into_iter().enumerate()
		{
			let cell_map = &self.neurons[neuron_idx];
			for entry in ctr_map.into_iter()
			{
				let (addr, packed) = entry;
				if packed == ram_core::neuron_memory::OI_INITIAL
				{
					continue;
				}
				let cell = ram_core::neuron_memory::oi_bin_to_cell(packed) as u8;
				if cell == self.default_empty
				{
					cell_map.remove(&addr);
				}
				else
				{
					cell_map.insert(addr, cell);
				}
			}
		}
	}

	/// Export to GPU-compatible sorted array format for binary search evaluation
	fn export_for_gpu(&self) -> SparseGpuExport
	{
		let mut keys: Vec<u64> = Vec::new();
		let mut values: Vec<u8> = Vec::new();
		let mut offsets: Vec<u32> = Vec::with_capacity(self.neurons.len());
		let mut counts: Vec<u32> = Vec::with_capacity(self.neurons.len());

		for neuron_map in &self.neurons
		{
			let offset = keys.len() as u32;
			offsets.push(offset);

			// Collect and sort entries for this neuron
			let mut entries: Vec<(u64, u8)> = neuron_map
				.iter()
				.map(|entry| (*entry.key(), *entry.value()))
				.collect();
			entries.sort_by_key(|(k, _)| *k);

			counts.push(entries.len() as u32);

			for (key, value) in entries
			{
				keys.push(key);
				values.push(value);
			}
		}

		SparseGpuExport {
			keys,
			values,
			offsets,
			counts,
			num_neurons: self.neurons.len(),
		}
	}

	#[inline]
	pub(crate) fn read(&self, neuron_idx: usize, address: u64) -> u8
	{
		*self.neurons[neuron_idx]
			.get(&address)
			.map(|v| *v)
			.as_ref()
			.unwrap_or(&self.default_empty)
	}

	/// Thread-safe write using DashMap
	///
	/// TRUE-wins-over-FALSE semantics (values: 0=FALSE, 1=TRUE, 2=EMPTY):
	/// - TRUE can be written over EMPTY or FALSE
	/// - FALSE can only be written over EMPTY
	/// - TRUE cannot be overwritten by FALSE
	#[inline]
	fn write(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool
	{
		let map = &self.neurons[neuron_idx];
		match map.entry(address)
		{
			dashmap::mapref::entry::Entry::Occupied(mut e) =>
			{
				let current = *e.get();

				// No change needed if same value
				if current == value
				{
					return false;
				}

				// TRUE wins over FALSE: don't overwrite TRUE with FALSE
				if current == 1 && value == 0
				{
					return false;
				}

				// If not allow_override:
				// - TRUE (1) can overwrite EMPTY (2) or FALSE (0) (TRUE wins)
				// - FALSE (0) can only overwrite EMPTY (2)
				if !allow_override && value == 0 && current != 2
				{
					return false;
				}

				// Allow TRUE to overwrite FALSE (TRUE wins) or write to EMPTY
				if allow_override || current == 2 || (value == 1 && current == 0)
				{
					e.insert(value);
					return true;
				}
				false
			}
			dashmap::mapref::entry::Entry::Vacant(e) =>
			{
				e.insert(value);
				true
			}
		}
	}

	/// Thread-safe nudge for quad modes using DashMap entry API.
	/// Moves cell one step toward target. For vacant entries, inserts one step
	/// from default (QUAD_WEAK_TRUE=2 if target_true, QUAD_WEAK_FALSE=1 stays if target_false).
	#[inline]
	fn nudge(&self, neuron_idx: usize, address: u64, target_true: bool) -> bool
	{
		let map = &self.neurons[neuron_idx];
		match map.entry(address)
		{
			dashmap::mapref::entry::Entry::Occupied(mut e) =>
			{
				let old_cell = *e.get() as i64;
				let delta = 2 * (target_true as i64) - 1;
				let new_cell = (old_cell + delta).clamp(
					ram_core::neuron_memory::QUAD_FALSE,
					ram_core::neuron_memory::QUAD_TRUE,
				) as u8;
				if new_cell == old_cell as u8
				{
					return false;
				}
				// Remove entry if it matches default_empty (saves memory)
				if new_cell == self.default_empty
				{
					e.remove();
				}
				else
				{
					e.insert(new_cell);
				}
				true
			}
			dashmap::mapref::entry::Entry::Vacant(e) =>
			{
				// Default is QUAD_WEAK_FALSE (1). Nudge toward true → insert 2, toward false → insert 0
				let default = self.default_empty as i64;
				let delta = 2 * (target_true as i64) - 1;
				let new_cell = (default + delta).clamp(
					ram_core::neuron_memory::QUAD_FALSE,
					ram_core::neuron_memory::QUAD_TRUE,
				) as u8;
				if new_cell == self.default_empty
				{
					return false; // no change from default
				}
				e.insert(new_cell);
				true
			}
		}
	}
}

/// Sparse memory backed by the new lock-free `AtomicHashTable` (per-neuron
/// flat-array hash). Drop-in replacement for `GroupSparseMemory` (DashMap) —
/// gated by the `WNN_SPARSE_BACKEND=atomic` environment variable so we can
/// A/B against the established DashMap path.
pub(crate) struct GroupSparseMemoryAtomic
{
	pub(crate) neurons: Vec<crate::atomic_hashtable::AtomicHashTable>,
	pub(crate) default_empty: u8,
}

impl GroupSparseMemoryAtomic
{
	pub(crate) fn new(num_neurons: usize, memory_mode: u8, initial_capacity: usize) -> Self
	{
		let default_empty = match memory_mode
		{
			ram_core::neuron_memory::QUAD_BINARY
			| ram_core::neuron_memory::QUAD_WEIGHTED
			| ram_core::neuron_memory::QSR => 1,
			ram_core::neuron_memory::BINARY => 0, // classical 1-bit: unwritten = FALSE
			_ => EMPTY as u8,
		};
		Self {
			neurons: (0..num_neurons)
				.map(|_| crate::atomic_hashtable::AtomicHashTable::new(initial_capacity, default_empty))
				.collect(),
			default_empty,
		}
	}

	/// Allocate OI counter buffers inside each per-neuron AtomicHashTable.
	/// Lock-free per-slot u32 counters; same hash table, parallel value array.
	pub fn init_oi_counters(&mut self)
	{
		for table in &self.neurons
		{
			table.init_oi_counters();
		}
	}

	/// Order-independent nudge: lock-free CAS on the packed counter for
	/// this (neuron, address) slot inside the AtomicHashTable.
	#[inline]
	pub fn nudge_oi(&self, neuron_idx: usize, address: u64, target_true: bool, weight: u32) -> bool
	{
		let delta: i32 = if target_true
		{
			weight as i32
		}
		else
		{
			-(weight as i32)
		};
		self.neurons[neuron_idx].nudge_oi(address, delta)
	}

	/// Commit pass: bin each per-slot counter into the 2-bit value field
	/// inside each AtomicHashTable, then drop the counter buffers. Entries
	/// with counter == OI_INITIAL are untouched. No DashMap layer needed —
	/// the AtomicHashTable provides lock-free atomic storage throughout.
	pub fn commit_oi(&mut self)
	{
		let _ = self.default_empty; // value used inside AtomicHashTable::commit_oi
		for table in &self.neurons
		{
			table.commit_oi();
		}
	}

	fn export_for_gpu(&self) -> SparseGpuExport
	{
		let mut keys: Vec<u64> = Vec::new();
		let mut values: Vec<u8> = Vec::new();
		let mut offsets: Vec<u32> = Vec::with_capacity(self.neurons.len());
		let mut counts: Vec<u32> = Vec::with_capacity(self.neurons.len());

		for table in &self.neurons
		{
			offsets.push(keys.len() as u32);
			let snap = table.snapshot_sorted();
			counts.push(snap.len() as u32);
			for (k, v) in snap
			{
				keys.push(k);
				values.push(v);
			}
		}

		SparseGpuExport {
			keys,
			values,
			offsets,
			counts,
			num_neurons: self.neurons.len(),
		}
	}

	#[inline]
	pub(crate) fn read(&self, neuron_idx: usize, address: u64) -> u8
	{
		self.neurons[neuron_idx].read(address)
	}

	#[inline]
	fn write(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool
	{
		self.neurons[neuron_idx].write(address, value, allow_override)
	}

	#[inline]
	fn nudge(&self, neuron_idx: usize, address: u64, target_true: bool) -> bool
	{
		self.neurons[neuron_idx].nudge(address, target_true)
	}
}

/// Returns true if the runtime is configured to use the atomic-hashtable
/// sparse backend (`WNN_SPARSE_BACKEND=atomic`). Default backend remains the
/// DashMap-based `GroupSparseMemory` until atomic is validated against it on
/// the cohort.
fn use_atomic_sparse_backend() -> bool
{
	std::env::var("WNN_SPARSE_BACKEND")
		.map(|v| v.eq_ignore_ascii_case("atomic"))
		.unwrap_or(false)
}

/// Hybrid memory - Dense for low bits, Sparse for high bits
/// Both variants support thread-safe concurrent access for parallel training.
pub(crate) enum GroupMemory
{
	Dense(GroupDenseMemory),
	Sparse(GroupSparseMemory),
	SparseAtomic(GroupSparseMemoryAtomic),
}

impl GroupMemory
{
	pub(crate) fn new(num_neurons: usize, bits: usize, memory_mode: u8) -> Self
	{
		if bits <= SPARSE_THRESHOLD
		{
			GroupMemory::Dense(GroupDenseMemory::new(num_neurons, bits, memory_mode))
		}
		else if use_atomic_sparse_backend()
		{
			// Initial capacity sized via heuristic on a "typical" working set;
			// the table grows 2x at 75% load so under-sizing is recoverable.
			let initial_capacity = crate::atomic_hashtable::estimate_capacity(1_000_000);
			GroupMemory::SparseAtomic(GroupSparseMemoryAtomic::new(
				num_neurons,
				memory_mode,
				initial_capacity,
			))
		}
		else
		{
			GroupMemory::Sparse(GroupSparseMemory::new(num_neurons, memory_mode))
		}
	}

	/// Check if this is dense memory (can be accelerated with Metal)
	pub(crate) fn is_dense(&self) -> bool
	{
		matches!(self, GroupMemory::Dense(_))
	}

	/// Export for Metal GPU (only works for Dense, returns None for Sparse)
	pub(crate) fn export_for_metal(&self) -> Option<Vec<i64>>
	{
		match self
		{
			GroupMemory::Dense(m) => Some(m.export_for_metal()),
			GroupMemory::Sparse(_) | GroupMemory::SparseAtomic(_) => None,
		}
	}

	/// Export sparse memory for GPU binary search (returns None for Dense)
	pub(crate) fn export_for_gpu_sparse(&self) -> Option<SparseGpuExport>
	{
		match self
		{
			GroupMemory::Dense(_) => None,
			GroupMemory::Sparse(m) => Some(m.export_for_gpu()),
			GroupMemory::SparseAtomic(m) => Some(m.export_for_gpu()),
		}
	}

	/// Check if this is sparse memory
	pub(crate) fn is_sparse(&self) -> bool
	{
		matches!(self, GroupMemory::Sparse(_) | GroupMemory::SparseAtomic(_))
	}

	#[inline]
	pub(crate) fn read(&self, neuron_idx: usize, address: usize) -> i64
	{
		match self
		{
			GroupMemory::Dense(m) => m.read(neuron_idx, address),
			GroupMemory::Sparse(m) => m.read(neuron_idx, address as u64) as i64,
			GroupMemory::SparseAtomic(m) => m.read(neuron_idx, address as u64) as i64,
		}
	}

	/// Thread-safe write (both variants support concurrent access)
	#[inline]
	pub(crate) fn write(
		&self,
		neuron_idx: usize,
		address: usize,
		value: i64,
		allow_override: bool,
	) -> bool
	{
		match self
		{
			GroupMemory::Dense(m) => m.write(neuron_idx, address, value, allow_override),
			GroupMemory::Sparse(m) => m.write(neuron_idx, address as u64, value as u8, allow_override),
			GroupMemory::SparseAtomic(m) =>
			{
				m.write(neuron_idx, address as u64, value as u8, allow_override)
			}
		}
	}

	/// Thread-safe nudge for quad modes — moves cell one step toward target.
	#[inline]
	pub(crate) fn nudge(&self, neuron_idx: usize, address: usize, target_true: bool) -> bool
	{
		match self
		{
			GroupMemory::Dense(m) => m.nudge(neuron_idx, address, target_true),
			GroupMemory::Sparse(m) => m.nudge(neuron_idx, address as u64, target_true),
			GroupMemory::SparseAtomic(m) => m.nudge(neuron_idx, address as u64, target_true),
		}
	}

	/// Order-independent training: allocate per-cell counter buffers.
	/// Must be called before any `nudge_oi` and matched by `commit_oi`.
	pub(crate) fn init_oi_counters(&mut self)
	{
		match self
		{
			GroupMemory::Dense(m) => m.init_oi_counters(),
			GroupMemory::Sparse(m) => m.init_oi_counters(),
			GroupMemory::SparseAtomic(m) => m.init_oi_counters(),
		}
	}

	/// Order-independent nudge: accumulates ±weight into the counter buffer.
	/// `init_oi_counters` must have been called first.
	#[inline]
	pub(crate) fn nudge_oi(
		&self,
		neuron_idx: usize,
		address: usize,
		target_true: bool,
		weight: u32,
	) -> bool
	{
		match self
		{
			GroupMemory::Dense(m) => m.nudge_oi(neuron_idx, address, target_true, weight),
			GroupMemory::Sparse(m) => m.nudge_oi(neuron_idx, address as u64, target_true, weight),
			GroupMemory::SparseAtomic(m) => m.nudge_oi(neuron_idx, address as u64, target_true, weight),
		}
	}

	/// Commit pass: bin counters to 2-bit cells and free counter buffers.
	pub(crate) fn commit_oi(&mut self)
	{
		match self
		{
			GroupMemory::Dense(m) => m.commit_oi(),
			GroupMemory::Sparse(m) => m.commit_oi(),
			GroupMemory::SparseAtomic(m) => m.commit_oi(),
		}
	}
}

/// Evaluate a dense config group using Metal GPU.
///
/// Returns scores for [num_examples × num_clusters_in_group] as f32.
/// The scores are in group-local cluster order (need scattering to global order).
pub(crate) fn evaluate_group_metal(
	metal: &crate::metal_ramlm::MetalRAMLMEvaluator,
	packed_eval: &[u64],
	connections_flat: &[i64],
	memory_words: &[i64],
	group: &ConfigGroup,
	num_eval: usize,
	words_per_example: usize,
	memory_mode: u8,
	empty_value: f32,
	run_seed: u64,
) -> Result<Vec<f32>, String>
{
	let num_clusters = group.cluster_count();
	let num_neurons = group.total_neurons();

	// Extract connections for this group (they're stored contiguously at conn_offset)
	let conn_size = group.conn_size();
	let group_connections = &connections_flat[group.conn_offset..group.conn_offset + conn_size];

	metal.forward_batch(
		packed_eval,
		group_connections,
		memory_words,
		num_eval,
		words_per_example,
		num_neurons,
		group.bits,
		group.neurons,
		num_clusters,
		group.words_per_neuron,
		memory_mode,
		empty_value,
		run_seed,
	)
}

/// Evaluate a sparse config group using Metal GPU with binary search.
///
/// Returns scores for [num_examples × num_clusters_in_group] as f32.
/// The scores are in group-local cluster order (need scattering to global order).
pub(crate) fn evaluate_group_sparse_gpu(
	sparse_evaluator: &ram_core::metal_sparse::MetalSparseEvaluator,
	packed_eval: &[u64],
	connections_flat: &[i64],
	export: &SparseGpuExport,
	group: &ConfigGroup,
	num_eval: usize,
	words_per_example: usize,
	coverage_aware: bool,
	memory_mode: u8,
	empty_value: f32,
	run_seed: u64,
) -> Result<Vec<f32>, String>
{
	let num_clusters = group.cluster_count();

	// Extract connections for this group
	let conn_size = group.conn_size();
	let group_connections = &connections_flat[group.conn_offset..group.conn_offset + conn_size];

	sparse_evaluator.forward_batch_sparse(
		packed_eval,
		group_connections,
		&export.keys,
		&export.values,
		&export.offsets,
		&export.counts,
		num_eval,
		words_per_example,
		export.num_neurons,
		group.bits,
		group.neurons,
		num_clusters,
		coverage_aware,
		memory_mode,
		empty_value,
		run_seed,
	)
}
