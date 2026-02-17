//! Unified Neuron Memory — single source of truth for all memory operations.
//!
//! This module consolidates constants, cell access functions, GPU export structs,
//! and memory mode definitions that were previously duplicated across:
//! - ramlm.rs (dense ternary + quad)
//! - bitwise_ramlm.rs (sequential per-genome)
//! - adaptive.rs (concurrent dense + sparse)
//! - sparse_memory.rs (DashMap sparse)
//!
//! Memory encoding (2-bit cells, 31 per 64-bit word):
//!   Ternary: FALSE=0, TRUE=1, EMPTY=2
//!   Quad:    QUAD_FALSE=0, QUAD_WEAK_FALSE=1, QUAD_WEAK_TRUE=2, QUAD_TRUE=3

use std::sync::atomic::{AtomicU32, Ordering};
use rustc_hash::FxHashMap;

// =============================================================================
// Cell Value Constants — Ternary Mode
// =============================================================================

pub const FALSE: i64 = 0;
pub const TRUE: i64 = 1;
pub const EMPTY: i64 = 2;

/// u8 variants for sparse storage (matches i64 encoding)
pub const FALSE_U8: u8 = 0;
pub const TRUE_U8: u8 = 1;
pub const EMPTY_U8: u8 = 2;

// =============================================================================
// Cell Value Constants — Quad Mode (4-state nudging)
// =============================================================================

pub const QUAD_FALSE: i64 = 0;
pub const QUAD_WEAK_FALSE: i64 = 1; // initial state for quad modes
pub const QUAD_WEAK_TRUE: i64 = 2;
pub const QUAD_TRUE: i64 = 3;

/// Weights for QUAD_WEIGHTED forward pass accumulation
pub const QUAD_WEIGHTS: [f32; 4] = [0.0, 0.25, 0.75, 1.0];

// =============================================================================
// Bit-Packing Constants
// =============================================================================

pub const BITS_PER_CELL: usize = 2;
pub const CELLS_PER_WORD: usize = 31; // 62 bits / 2 = 31 cells per i64 word
pub const CELL_MASK: i64 = 0b11;

// =============================================================================
// Memory Mode Constants
// =============================================================================

pub const MODE_TERNARY: u8 = 0;
pub const MODE_QUAD_BINARY: u8 = 1;
pub const MODE_QUAD_WEIGHTED: u8 = 2;

// =============================================================================
// Empty Value Global State
// =============================================================================
//
// Controls the contribution of EMPTY cells in ternary forward pass:
//   0.0 = EMPTY cells abstain (default, recommended)
//   0.5 = EMPTY cells add uncertainty (old default)

static EMPTY_VALUE_BITS: AtomicU32 = AtomicU32::new(0); // 0.0f32 as bits

/// Get the global EMPTY cell value for ternary forward pass.
pub fn get_empty_value() -> f32 {
	f32::from_bits(EMPTY_VALUE_BITS.load(Ordering::Relaxed))
}

/// Set the global EMPTY cell value (call from Python before evaluation).
pub fn set_empty_value(value: f32) {
	EMPTY_VALUE_BITS.store(value.to_bits(), Ordering::Relaxed);
}

// =============================================================================
// GPU Export Struct (unified from adaptive.rs + sparse_memory.rs)
// =============================================================================

/// GPU-compatible sparse memory export — sorted arrays for binary search on Metal.
///
/// Per-neuron layout:
///   keys[offsets[n]..offsets[n]+counts[n]] — sorted addresses
///   values[offsets[n]..offsets[n]+counts[n]] — corresponding cell values
#[derive(Clone)]
pub struct SparseGpuExport {
	/// Sorted keys for all neurons, concatenated
	pub keys: Vec<u64>,
	/// Values corresponding to keys (0=FALSE, 1=TRUE, 2=EMPTY for ternary;
	/// 0-3 for quad)
	pub values: Vec<u8>,
	/// Start offset for each neuron in keys array
	pub offsets: Vec<u32>,
	/// Number of entries for each neuron
	pub counts: Vec<u32>,
	/// Total number of neurons
	pub num_neurons: usize,
}

impl SparseGpuExport {
	/// CPU binary search lookup (for verification/fallback)
	#[inline]
	pub fn lookup(&self, neuron_idx: usize, address: u64) -> u8 {
		let start = self.offsets[neuron_idx] as usize;
		let count = self.counts[neuron_idx] as usize;

		if count == 0 {
			return EMPTY_U8;
		}

		let end = start + count;
		let keys_slice = &self.keys[start..end];

		match keys_slice.binary_search(&address) {
			Ok(idx) => self.values[start + idx],
			Err(_) => EMPTY_U8,
		}
	}

	/// Total memory size in bytes
	pub fn memory_size(&self) -> usize {
		self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4 + self.counts.len() * 4
	}

	/// Total number of entries across all neurons
	pub fn total_entries(&self) -> usize {
		self.keys.len()
	}
}

// =============================================================================
// Cell Access Functions — Sequential (non-atomic, single-thread per genome)
// =============================================================================

/// Read a 2-bit cell from bit-packed memory.
#[inline]
pub fn read_cell(memory_words: &[i64], neuron_idx: usize, address: usize, words_per_neuron: usize) -> i64 {
	let word_idx = address / CELLS_PER_WORD;
	let cell_idx = address % CELLS_PER_WORD;
	let word_offset = neuron_idx * words_per_neuron + word_idx;
	(memory_words[word_offset] >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
}

/// Read a cell using pre-computed memory offset (for heterogeneous configs).
#[inline]
pub fn read_cell_offset(memory: &[i64], neuron_mem_start: usize, address: usize) -> i64 {
	let word_idx = address / CELLS_PER_WORD;
	let cell_idx = address % CELLS_PER_WORD;
	let word_offset = neuron_mem_start + word_idx;
	(memory[word_offset] >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
}

/// Non-atomic cell read by (word_offset, cell_idx) — for neuron-parallel training.
#[inline]
pub fn read_cell_direct(memory_words: &[i64], word_offset: usize, cell_idx: usize) -> i64 {
	let shift = cell_idx * BITS_PER_CELL;
	(memory_words[word_offset] >> shift) & CELL_MASK
}

/// Non-atomic cell write by (word_offset, cell_idx) — for neuron-parallel training.
#[inline]
pub fn write_cell_direct(memory_words: &mut [i64], word_offset: usize, cell_idx: usize, value: i64) {
	let shift = cell_idx * BITS_PER_CELL;
	let mask = CELL_MASK << shift;
	memory_words[word_offset] = (memory_words[word_offset] & !mask) | (value << shift);
}

/// Write cell using pre-computed memory offset (sequential, no atomics).
#[inline]
pub fn write_cell_offset(
	memory: &mut [i64],
	neuron_mem_start: usize,
	address: usize,
	value: i64,
) {
	let word_idx = address / CELLS_PER_WORD;
	let cell_idx = address % CELLS_PER_WORD;
	let word_offset = neuron_mem_start + word_idx;
	let shift = cell_idx * BITS_PER_CELL;
	let mask = CELL_MASK << shift;
	memory[word_offset] = (memory[word_offset] & !mask) | (value << shift);
}

/// Nudge a cell one step toward target (sequential, branchless).
/// target_true: cell = min(cell + 1, 3)
/// target_false: cell = max(cell - 1, 0)
#[inline]
pub fn nudge_cell_offset(
	memory: &mut [i64],
	neuron_mem_start: usize,
	address: usize,
	target_true: bool,
) {
	let word_idx = address / CELLS_PER_WORD;
	let cell_idx = address % CELLS_PER_WORD;
	let word_offset = neuron_mem_start + word_idx;
	let shift = cell_idx * BITS_PER_CELL;
	let old_cell = (memory[word_offset] >> shift) & CELL_MASK;

	let delta = 2 * (target_true as i64) - 1;
	let new_cell = (old_cell + delta).clamp(QUAD_FALSE, QUAD_TRUE);

	let mask = CELL_MASK << shift;
	memory[word_offset] = (memory[word_offset] & !mask) | (new_cell << shift);
}

// =============================================================================
// GPU Training Metadata (shared with Metal shader)
// =============================================================================

/// Per-neuron metadata for GPU address computation during training.
/// Passed to Metal `train_compute_addresses` kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeuronTrainMeta {
	/// Number of address bits for this neuron
	pub bits: u32,
	/// Offset into the flat connections array for this neuron's connections
	pub conn_offset: u32,
}

/// Parameters for GPU training address computation kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TrainAddressParams {
	/// Number of training examples
	pub num_examples: u32,
	/// Words per example in packed_input (ceil(total_input_bits / 64))
	pub words_per_example: u32,
	/// Total number of neurons across all clusters
	pub total_neurons: u32,
	/// Padding for 16-byte alignment
	pub _pad: u32,
}

// =============================================================================
// Address Computation
// =============================================================================

/// Compute memory address from boolean input bits (MSB-first).
#[inline]
pub fn compute_address(input_bits: &[bool], connections: &[i64], bits_per_neuron: usize) -> usize {
	let mut address: usize = 0;
	for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
		if input_bits[conn_idx as usize] {
			address |= 1 << (bits_per_neuron - 1 - i);
		}
	}
	address
}

/// Compute memory address from packed u64 input bits (8x less memory bandwidth).
#[inline]
pub fn compute_address_packed(packed_words: &[u64], connections: &[i64], bits_per_neuron: usize) -> usize {
	let mut address: usize = 0;
	for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
		let idx = conn_idx as usize;
		let bit = (packed_words[idx / 64] >> (idx % 64)) & 1;
		address |= (bit as usize) << (bits_per_neuron - 1 - i);
	}
	address
}

/// Compute address for sparse storage (returns u64 for high-bit neurons).
#[inline]
pub fn compute_address_sparse(input_bits: &[bool], connections: &[i64], bits_per_neuron: usize) -> u64 {
	let mut address: u64 = 0;
	for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
		if input_bits[conn_idx as usize] {
			address |= 1 << (bits_per_neuron - 1 - i);
		}
	}
	address
}

// =============================================================================
// Helper: Build empty word for initialization
// =============================================================================

/// Build a 64-bit word with all 31 cells set to the given 2-bit value.
pub fn build_empty_word(cell_value: i64) -> i64 {
	(0..31i64).fold(0i64, |acc, i| acc | (cell_value << (i * 2)))
}

/// Build the empty word for a given memory mode.
pub fn empty_word_for_mode(memory_mode: u8) -> i64 {
	match memory_mode {
		MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => build_empty_word(QUAD_WEAK_FALSE),
		_ => build_empty_word(EMPTY),
	}
}

// =============================================================================
// Helper: Words per neuron
// =============================================================================

/// Compute the number of 64-bit words needed per neuron for a given bit width.
#[inline]
pub fn words_per_neuron(bits: usize) -> usize {
	let addresses = 1usize << bits;
	(addresses + CELLS_PER_WORD - 1) / CELLS_PER_WORD
}

// =============================================================================
// Packing: Bool → u64 (for GPU input)
// =============================================================================

/// Pack flat bool slice into u64 words (LSB-first, matching Metal shader bit extraction).
/// Returns (packed_data, words_per_example).
pub fn pack_bools_to_u64(bools: &[bool], num_examples: usize, total_bits: usize) -> (Vec<u64>, usize) {
	let words_per_example = (total_bits + 63) / 64;
	let mut packed = vec![0u64; num_examples * words_per_example];
	for ex in 0..num_examples {
		let bits_off = ex * total_bits;
		let pack_off = ex * words_per_example;
		for i in 0..total_bits {
			if bools[bits_off + i] {
				packed[pack_off + i / 64] |= 1u64 << (i % 64);
			}
		}
	}
	(packed, words_per_example)
}

// =============================================================================
// ClusterStorage — Per-Cluster Dense/Sparse Memory
// =============================================================================

/// Default threshold: clusters with bits > this use sparse storage.
/// Only used when auto-compute is disabled. Normally auto_sparse_threshold() picks
/// the optimal value per genome based on the memory budget.
pub const DEFAULT_SPARSE_THRESHOLD: usize = 20;

/// Auto-compute the optimal sparse threshold for a genome to fit within target_bytes.
///
/// Tries thresholds from max_bits down to 0, returns the highest (= most dense = fastest)
/// threshold where the total estimated memory fits the budget.
/// Returns usize::MAX if all-dense fits.
pub fn auto_sparse_threshold(
	max_bits_per_cluster: &[usize],
	neurons_per_cluster: &[usize],
	target_bytes: u64,
	expected_train: usize,
) -> usize {
	// Try all-dense first (fastest path)
	let all_dense: u64 = max_bits_per_cluster.iter().zip(neurons_per_cluster.iter())
		.map(|(&b, &n)| {
			let wpn = words_per_neuron(b);
			(n * wpn * 8) as u64
		})
		.sum();
	if all_dense <= target_bytes {
		return usize::MAX;
	}

	// Find max bits in this genome
	let max_bits = *max_bits_per_cluster.iter().max().unwrap_or(&0);

	// Try thresholds from max_bits-1 down to 0
	// Each step makes one more bit-width level sparse
	for threshold in (0..max_bits).rev() {
		let est: u64 = max_bits_per_cluster.iter().zip(neurons_per_cluster.iter())
			.map(|(&b, &n)| ClusterStorage::estimated_bytes(n, b, threshold, expected_train))
			.sum();
		if est <= target_bytes {
			return threshold;
		}
	}

	0 // all sparse as last resort
}

/// Per-cluster neuron memory — either dense (bit-packed) or sparse (HashMap).
///
/// Dense: bit-packed `Vec<i64>`, 31 cells per word. Fast for small address spaces.
/// Sparse: `Vec<FxHashMap<u32, u8>>`, one map per neuron. Compact for large address spaces.
///
/// Uses `FxHashMap` (non-concurrent) since `bitwise_ramlm.rs` trains sequentially per genome.
pub enum ClusterStorage {
	Dense {
		words: Vec<i64>,
		words_per_neuron: usize,
		num_neurons: usize,
		empty_word: i64,
	},
	Sparse {
		neurons: Vec<FxHashMap<u32, u8>>,
		num_neurons: usize,
		/// Default cell value for unvisited addresses (EMPTY_U8 for ternary, 1 for quad).
		empty_cell: u8,
	},
}

impl ClusterStorage {
	/// Create storage for a cluster. Uses dense if `bits <= threshold`, sparse otherwise.
	pub fn new(num_neurons: usize, bits: usize, threshold: usize, empty_word: i64, memory_mode: u8) -> Self {
		if bits <= threshold {
			let wpn = words_per_neuron(bits);
			ClusterStorage::Dense {
				words: vec![empty_word; num_neurons * wpn],
				words_per_neuron: wpn,
				num_neurons,
				empty_word,
			}
		} else {
			let empty_cell = match memory_mode {
				MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => 1, // QUAD_WEAK_FALSE
				_ => EMPTY_U8,
			};
			ClusterStorage::Sparse {
				neurons: (0..num_neurons).map(|_| FxHashMap::default()).collect(),
				num_neurons,
				empty_cell,
			}
		}
	}

	/// Reset storage: refill dense with empty_word, clear all sparse maps.
	pub fn reset(&mut self) {
		match self {
			ClusterStorage::Dense { words, empty_word, .. } => {
				words.fill(*empty_word);
			}
			ClusterStorage::Sparse { neurons, .. } => {
				for map in neurons.iter_mut() {
					map.clear();
				}
			}
		}
	}

	/// Read a 2-bit cell value for a given neuron and address.
	#[inline]
	pub fn read_cell(&self, neuron_idx: usize, address: usize) -> i64 {
		match self {
			ClusterStorage::Dense { words, words_per_neuron, .. } => {
				let word_idx = address / CELLS_PER_WORD;
				let cell_idx = address % CELLS_PER_WORD;
				let word_offset = neuron_idx * words_per_neuron + word_idx;
				(words[word_offset] >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
			}
			ClusterStorage::Sparse { neurons, empty_cell, .. } => {
				*neurons[neuron_idx].get(&(address as u32)).unwrap_or(empty_cell) as i64
			}
		}
	}

	/// Write a cell value unconditionally.
	#[inline]
	pub fn write_cell(&mut self, neuron_idx: usize, address: usize, value: i64) {
		match self {
			ClusterStorage::Dense { words, words_per_neuron, .. } => {
				let word_idx = address / CELLS_PER_WORD;
				let cell_idx = address % CELLS_PER_WORD;
				let word_offset = neuron_idx * *words_per_neuron + word_idx;
				let shift = cell_idx * BITS_PER_CELL;
				let mask = CELL_MASK << shift;
				words[word_offset] = (words[word_offset] & !mask) | (value << shift);
			}
			ClusterStorage::Sparse { neurons, empty_cell, .. } => {
				if value == *empty_cell as i64 {
					neurons[neuron_idx].remove(&(address as u32));
				} else {
					neurons[neuron_idx].insert(address as u32, value as u8);
				}
			}
		}
	}

	/// Nudge a cell one step toward target (quad mode 4-state nudging).
	/// target_true: cell = min(cell + 1, 3)
	/// target_false: cell = max(cell - 1, 0)
	#[inline]
	pub fn nudge_cell(&mut self, neuron_idx: usize, address: usize, target_true: bool) {
		match self {
			ClusterStorage::Dense { words, words_per_neuron, .. } => {
				let word_idx = address / CELLS_PER_WORD;
				let cell_idx = address % CELLS_PER_WORD;
				let word_offset = neuron_idx * *words_per_neuron + word_idx;
				let shift = cell_idx * BITS_PER_CELL;
				let old_cell = (words[word_offset] >> shift) & CELL_MASK;
				let delta = 2 * (target_true as i64) - 1;
				let new_cell = (old_cell + delta).clamp(QUAD_FALSE, QUAD_TRUE);
				let mask = CELL_MASK << shift;
				words[word_offset] = (words[word_offset] & !mask) | (new_cell << shift);
			}
			ClusterStorage::Sparse { neurons, empty_cell, .. } => {
				let old_cell = *neurons[neuron_idx].get(&(address as u32)).unwrap_or(empty_cell) as i64;
				let delta = 2 * (target_true as i64) - 1;
				let new_cell = (old_cell + delta).clamp(QUAD_FALSE, QUAD_TRUE);
				if new_cell == *empty_cell as i64 {
					neurons[neuron_idx].remove(&(address as u32));
				} else {
					neurons[neuron_idx].insert(address as u32, new_cell as u8);
				}
			}
		}
	}

	#[inline]
	pub fn is_dense(&self) -> bool {
		matches!(self, ClusterStorage::Dense { .. })
	}

	#[inline]
	pub fn is_sparse(&self) -> bool {
		matches!(self, ClusterStorage::Sparse { .. })
	}

	pub fn num_neurons(&self) -> usize {
		match self {
			ClusterStorage::Dense { num_neurons, .. } => *num_neurons,
			ClusterStorage::Sparse { num_neurons, .. } => *num_neurons,
		}
	}

	pub fn wpn(&self) -> usize {
		match self {
			ClusterStorage::Dense { words_per_neuron, .. } => *words_per_neuron,
			ClusterStorage::Sparse { .. } => 0,
		}
	}

	/// Actual memory usage in bytes.
	pub fn memory_bytes(&self) -> usize {
		match self {
			ClusterStorage::Dense { words, .. } => words.len() * 8,
			ClusterStorage::Sparse { neurons, .. } => {
				// FxHashMap overhead: ~56 bytes base + 12 bytes per entry (key: 4, value: 1, hash+padding: 7)
				let base = neurons.len() * 56;
				let entries: usize = neurons.iter().map(|m| m.len()).sum();
				base + entries * 12
			}
		}
	}

	/// Estimated memory bytes for budget planning (static, before allocation).
	/// Caps sparse entries at min(expected, 2^bits) since a neuron can't have
	/// more unique addresses than its address space.
	pub fn estimated_bytes(num_neurons: usize, bits: usize, threshold: usize, expected_entries_per_neuron: usize) -> u64 {
		if bits <= threshold {
			let wpn = words_per_neuron(bits);
			(num_neurons * wpn * 8) as u64
		} else {
			let max_entries = 1usize << bits;
			let actual_entries = expected_entries_per_neuron.min(max_entries);
			(num_neurons as u64) * (56 + actual_entries as u64 * 12)
		}
	}

	/// Extract the dense memory slice for GPU (dense only).
	/// Returns the raw words slice for this cluster's neurons.
	pub fn dense_words(&self) -> &[i64] {
		match self {
			ClusterStorage::Dense { words, .. } => words,
			ClusterStorage::Sparse { .. } => panic!("dense_words() called on sparse storage"),
		}
	}

	/// Build GPU export from sparse storage (sorted arrays for binary search).
	pub fn export_sparse_gpu(&self) -> SparseGpuExport {
		match self {
			ClusterStorage::Sparse { neurons, num_neurons, .. } => {
				let mut keys = Vec::new();
				let mut values = Vec::new();
				let mut offsets = Vec::with_capacity(*num_neurons);
				let mut counts = Vec::with_capacity(*num_neurons);

				for map in neurons.iter() {
					offsets.push(keys.len() as u32);
					let mut entries: Vec<(u32, u8)> = map.iter().map(|(&k, &v)| (k, v)).collect();
					entries.sort_unstable_by_key(|(k, _)| *k);
					counts.push(entries.len() as u32);
					for (k, v) in entries {
						keys.push(k as u64);
						values.push(v);
					}
				}

				SparseGpuExport { keys, values, offsets, counts, num_neurons: *num_neurons }
			}
			ClusterStorage::Dense { .. } => panic!("export_sparse_gpu() called on dense storage"),
		}
	}
}
