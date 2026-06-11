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
// Cell → Weight Conversion (forward-pass scoring)
// =============================================================================

/// Convert a raw cell value to a forward-pass weight based on memory mode.
///
/// - TERNARY: FALSE=0.0, TRUE=1.0, EMPTY=empty_value
/// - QUAD_WEIGHTED / QUAD_BINARY: QUAD_WEIGHTS[cell] = [0.0, 0.25, 0.75, 1.0]
///   (`empty_value` is unused — WEAK_FALSE=0.25 is the initial/baseline state)
///
/// This is THE single source of truth for CPU-side cell scoring. Never
/// hardcode `FALSE => 0.0, TRUE => 1.0` at a call site: ternary and quad
/// encodings agree only on cell 0, so a raw ternary match in QUAD mode
/// scores WEAK_FALSE as 1.0 and TRUE as 0.25 — silently inverted.
#[inline(always)]
pub fn cell_to_weight(cell: i64, memory_mode: u8, empty_value: f32) -> f32 {
	match memory_mode {
		MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => QUAD_WEIGHTS[cell.clamp(0, 3) as usize],
		_ => match cell {
			FALSE => 0.0,
			TRUE => 1.0,
			_ => empty_value,
		},
	}
}

// =============================================================================
// Empty Value Global State
// =============================================================================
//
// Controls the contribution of EMPTY cells in ternary forward pass:
//   0.0 = EMPTY cells abstain (default, recommended)
//   0.5 = EMPTY cells add uncertainty (old default)

static EMPTY_VALUE_BITS: AtomicU32 = AtomicU32::new(0); // 0.0f32 as bits
static MEMORY_MODE: AtomicU32 = AtomicU32::new(2); // MODE_QUAD_WEIGHTED by default

/// Get the global EMPTY cell value for ternary forward pass.
pub fn get_empty_value() -> f32 {
	f32::from_bits(EMPTY_VALUE_BITS.load(Ordering::Relaxed))
}

/// Set the global EMPTY cell value (call from Python before evaluation).
pub fn set_empty_value(value: f32) {
	EMPTY_VALUE_BITS.store(value.to_bits(), Ordering::Relaxed);
}

/// Get the global memory mode (for GPU shader dispatch).
pub fn get_memory_mode() -> u8 {
	MEMORY_MODE.load(Ordering::Relaxed) as u8
}

/// Set the global memory mode.
pub fn set_memory_mode(mode: u8) {
	MEMORY_MODE.store(mode as u32, Ordering::Relaxed);
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
	// KEPT-API: GPU-export contract completeness (mirrors offsets/counts shape)
	#[allow(dead_code)]
	pub num_neurons: usize,
}

impl SparseGpuExport {
	/// CPU binary search lookup (for verification/fallback)
	#[inline]
	// KEPT-API: CPU verification twin of the Metal binary-search lookup (parity debugging)
	#[allow(dead_code)]
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
	// KEPT-API: export introspection (debug/telemetry symmetry)
	#[allow(dead_code)]
	pub fn memory_size(&self) -> usize {
		self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4 + self.counts.len() * 4
	}

	/// Total number of entries across all neurons
	// KEPT-API: export introspection (debug/telemetry symmetry)
	#[allow(dead_code)]
	pub fn total_entries(&self) -> usize {
		self.keys.len()
	}
}

// =============================================================================
// Cell Access Functions — Sequential (non-atomic, single-thread per genome)
// =============================================================================

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

/// Compute memory address from a bit-packed row (LSB-first within byte).
///
/// Identical semantics to `compute_address` but reads from `PackedBits::packed_row(i)`
/// output (a `&[u8]` of `bytes_per_row` bytes) instead of a `&[bool]` slice.
/// Used by adaptive.rs after the Phase 2 PackedBits migration.
#[inline]
pub fn compute_address_packed_bytes(packed_row: &[u8], connections: &[i64], bits_per_neuron: usize) -> usize {
	let mut address: usize = 0;
	for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
		let idx = conn_idx as usize;
		let bit = (unsafe { *packed_row.get_unchecked(idx >> 3) } >> (idx & 7)) & 1;
		address |= (bit as usize) << (bits_per_neuron - 1 - i);
	}
	address
}

/// Sparse variant of `compute_address_packed_bytes` (returns u64 for high-bit neurons).
#[inline]
// KEPT-API: canonical sparse twin of compute_address_packed_bytes (single source of truth API)
#[allow(dead_code)]
pub fn compute_address_packed_bytes_sparse(packed_row: &[u8], connections: &[i64], bits_per_neuron: usize) -> u64 {
	let mut address: u64 = 0;
	for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
		let idx = conn_idx as usize;
		let bit = (unsafe { *packed_row.get_unchecked(idx >> 3) } >> (idx & 7)) & 1;
		address |= (bit as u64) << (bits_per_neuron - 1 - i);
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

/// Pack PackedBits (LSB-first u8) into u64 words (LSB-first within u64).
///
/// Since both representations are little-endian within their containers, this is
/// effectively a memcpy of bytes into u64 form — much faster than the bool-by-bool
/// path. Trailing bytes in each row beyond `total_bits` are zero by PackedBits
/// invariant (`row_as_bools` never reads them).
pub fn pack_packed_to_u64(packed: &crate::packed_bits::PackedBits) -> (Vec<u64>, usize) {
	let total_bits = packed.total_bits();
	let num_examples = packed.num_rows();
	let words_per_example = (total_bits + 63) / 64;
	let bytes_per_row = packed.bytes_per_row();
	let bytes_per_word = 8usize;

	let mut out = vec![0u64; num_examples * words_per_example];
	let bytes = packed.as_bytes();

	for ex in 0..num_examples {
		let byte_off = ex * bytes_per_row;
		let word_off = ex * words_per_example;
		for w in 0..words_per_example {
			let b_start = w * bytes_per_word;
			let mut word_bytes = [0u8; 8];
			let take = bytes_per_word.min(bytes_per_row.saturating_sub(b_start));
			if take > 0 {
				word_bytes[..take].copy_from_slice(&bytes[byte_off + b_start..byte_off + b_start + take]);
			}
			out[word_off + w] = u64::from_le_bytes(word_bytes);
		}
	}
	(out, words_per_example)
}

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
		/// OI training (QUAD only): parallel u32 packed counter per (neuron, address).
		/// Length = num_neurons * addresses_per_neuron. Lives only during an OI
		/// training pass; binned into `words` and freed at commit. Sequential
		/// (no atomics) — ClusterStorage is used single-thread per genome.
		oi_counters: Option<Vec<u32>>,
		/// TERNARY training vote buffer: parallel f32 vote per (neuron, address).
		/// Lives only during a TERNARY training pass; sign-binned into `words`
		/// and freed at commit. Replaces the function-local vote storage in
		/// bitwise_ramlm so all training paths share the init/accumulate/commit
		/// shape.
		ternary_votes: Option<Vec<f32>>,
		/// Number of addresses per neuron (= 1 << bits). Needed for OI counter
		/// and ternary-vote indexing since `words_per_neuron * CELLS_PER_WORD`
		/// can have rounding padding that doesn't match the actual address space.
		addresses_per_neuron: usize,
	},
	Sparse {
		neurons: Vec<FxHashMap<u32, u8>>,
		num_neurons: usize,
		/// Default cell value for unvisited addresses (EMPTY_U8 for ternary, 1 for quad).
		empty_cell: u8,
		/// OI training (QUAD only): per-neuron counter maps storing packed (obs, net) values.
		oi_counter_maps: Option<Vec<FxHashMap<u32, u32>>>,
		/// TERNARY training vote maps: per-neuron HashMap of f32 votes.
		ternary_vote_maps: Option<Vec<FxHashMap<u32, f32>>>,
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
				oi_counters: None,
				ternary_votes: None,
				addresses_per_neuron: 1usize << bits,
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
				oi_counter_maps: None,
				ternary_vote_maps: None,
			}
		}
	}

	/// Reset storage: refill dense with empty_word, clear all sparse maps.
	pub fn reset(&mut self) {
		match self {
			ClusterStorage::Dense { words, empty_word, oi_counters, ternary_votes, .. } => {
				words.fill(*empty_word);
				*oi_counters = None;
				*ternary_votes = None;
			}
			ClusterStorage::Sparse { neurons, oi_counter_maps, ternary_vote_maps, .. } => {
				for map in neurons.iter_mut() {
					map.clear();
				}
				*oi_counter_maps = None;
				*ternary_vote_maps = None;
			}
		}
	}

	/// Allocate the OI training counter buffer. Idempotent — no-op if already
	/// allocated. Called once before an OI training pass.
	pub fn init_oi_counters(&mut self) {
		match self {
			ClusterStorage::Dense { num_neurons, addresses_per_neuron, oi_counters, .. } => {
				if oi_counters.is_some() { return; }
				*oi_counters = Some(vec![OI_INITIAL; (*num_neurons) * (*addresses_per_neuron)]);
			}
			ClusterStorage::Sparse { num_neurons, oi_counter_maps, .. } => {
				if oi_counter_maps.is_some() { return; }
				*oi_counter_maps = Some((0..*num_neurons).map(|_| FxHashMap::default()).collect());
			}
		}
	}

	/// Order-independent nudge: accumulate ±weight into the packed counter.
	/// Must be called between `init_oi_counters()` and `commit_oi()`.
	/// Sequential, no atomics — ClusterStorage is used single-thread per genome.
	#[inline]
	pub fn nudge_cell_oi(&mut self, neuron_idx: usize, address: usize, target_true: bool, weight: u32) {
		let delta: i32 = if target_true { weight as i32 } else { -(weight as i32) };
		match self {
			ClusterStorage::Dense { oi_counters, addresses_per_neuron, .. } => {
				let counters = oi_counters.as_mut()
					.expect("nudge_cell_oi called without init_oi_counters");
				let idx = neuron_idx * (*addresses_per_neuron) + address;
				counters[idx] = oi_apply_nudge(counters[idx], delta);
			}
			ClusterStorage::Sparse { oi_counter_maps, .. } => {
				let maps = oi_counter_maps.as_mut()
					.expect("nudge_cell_oi called without init_oi_counters");
				let entry = maps[neuron_idx].entry(address as u32).or_insert(OI_INITIAL);
				*entry = oi_apply_nudge(*entry, delta);
			}
		}
	}

	/// Allocate the TERNARY training vote buffer. Idempotent.
	/// Used by `bitwise_ramlm::train_into` TERNARY branch in place of the
	/// previous function-local f32 vote arrays — same algorithm, just owned
	/// by ClusterStorage so the API mirrors QUAD's `init/nudge/commit_oi`.
	pub fn init_ternary_votes(&mut self) {
		match self {
			ClusterStorage::Dense { num_neurons, addresses_per_neuron, ternary_votes, .. } => {
				if ternary_votes.is_some() { return; }
				*ternary_votes = Some(vec![0.0f32; (*num_neurons) * (*addresses_per_neuron)]);
			}
			ClusterStorage::Sparse { num_neurons, ternary_vote_maps, .. } => {
				if ternary_vote_maps.is_some() { return; }
				*ternary_vote_maps = Some((0..*num_neurons).map(|_| FxHashMap::default()).collect());
			}
		}
	}

	/// Accumulate a signed f32 vote into the ternary vote buffer.
	/// Must be called between `init_ternary_votes()` and `commit_ternary()`.
	#[inline]
	pub fn add_ternary_vote(&mut self, neuron_idx: usize, address: usize, vote: f32) {
		match self {
			ClusterStorage::Dense { ternary_votes, addresses_per_neuron, .. } => {
				let votes = ternary_votes.as_mut()
					.expect("add_ternary_vote called without init_ternary_votes");
				let idx = neuron_idx * (*addresses_per_neuron) + address;
				votes[idx] += vote;
			}
			ClusterStorage::Sparse { ternary_vote_maps, .. } => {
				let maps = ternary_vote_maps.as_mut()
					.expect("add_ternary_vote called without init_ternary_votes");
				*maps[neuron_idx].entry(address as u32).or_insert(0.0) += vote;
			}
		}
	}

	/// Commit ternary votes: write TRUE for v > 0, FALSE for v < 0, leave
	/// untouched (EMPTY/default) for v == 0 or no vote. Drops the vote buffer.
	pub fn commit_ternary(&mut self) {
		match self {
			ClusterStorage::Dense { words, words_per_neuron, num_neurons,
				addresses_per_neuron, ternary_votes, .. } => {
				let Some(votes) = ternary_votes.take() else { return; };
				for neuron_idx in 0..*num_neurons {
					let n_base = neuron_idx * (*addresses_per_neuron);
					for address in 0..*addresses_per_neuron {
						let v = votes[n_base + address];
						if v == 0.0 { continue; }
						let cell = if v > 0.0 { TRUE } else { FALSE };
						let word_idx = address / CELLS_PER_WORD;
						let cell_idx = address % CELLS_PER_WORD;
						let word_offset = neuron_idx * (*words_per_neuron) + word_idx;
						let shift = cell_idx * BITS_PER_CELL;
						let mask = CELL_MASK << shift;
						words[word_offset] = (words[word_offset] & !mask) | (cell << shift);
					}
				}
			}
			ClusterStorage::Sparse { neurons, empty_cell, ternary_vote_maps, .. } => {
				let Some(vote_maps) = ternary_vote_maps.take() else { return; };
				for (neuron_idx, vmap) in vote_maps.into_iter().enumerate() {
					let cell_map = &mut neurons[neuron_idx];
					for (addr, v) in vmap.into_iter() {
						if v == 0.0 { continue; }
						let cell = if v > 0.0 { TRUE as u8 } else { FALSE as u8 };
						if cell == *empty_cell {
							cell_map.remove(&addr);
						} else {
							cell_map.insert(addr, cell);
						}
					}
				}
			}
		}
	}

	/// Commit pass: bin every touched counter into its 2-bit cell, then drop
	/// the counter buffer. After commit, the storage layout is identical to
	/// a normally-trained cluster (eval/export paths unchanged).
	pub fn commit_oi(&mut self) {
		match self {
			ClusterStorage::Dense { words, words_per_neuron, num_neurons,
				addresses_per_neuron, oi_counters, .. } => {
				let Some(counters) = oi_counters.take() else { return; };
				for neuron_idx in 0..*num_neurons {
					let n_base = neuron_idx * (*addresses_per_neuron);
					for address in 0..*addresses_per_neuron {
						let packed = counters[n_base + address];
						if packed == OI_INITIAL { continue; }
						let cell = oi_bin_to_cell(packed);
						let word_idx = address / CELLS_PER_WORD;
						let cell_idx = address % CELLS_PER_WORD;
						let word_offset = neuron_idx * (*words_per_neuron) + word_idx;
						let shift = cell_idx * BITS_PER_CELL;
						let mask = CELL_MASK << shift;
						words[word_offset] = (words[word_offset] & !mask) | (cell << shift);
					}
				}
			}
			ClusterStorage::Sparse { neurons, empty_cell, oi_counter_maps, .. } => {
				let Some(counter_maps) = oi_counter_maps.take() else { return; };
				for (neuron_idx, ctr_map) in counter_maps.into_iter().enumerate() {
					let cell_map = &mut neurons[neuron_idx];
					for (addr, packed) in ctr_map.into_iter() {
						if packed == OI_INITIAL { continue; }
						let cell = oi_bin_to_cell(packed) as u8;
						if cell == *empty_cell {
							cell_map.remove(&addr);
						} else {
							cell_map.insert(addr, cell);
						}
					}
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
	// KEPT-API: ClusterStorage API completeness (read/write/introspection symmetry)
	#[allow(dead_code)]
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
	// KEPT-API: ClusterStorage API symmetry
	#[allow(dead_code)]
	pub fn is_sparse(&self) -> bool {
		matches!(self, ClusterStorage::Sparse { .. })
	}

	// KEPT-API: ClusterStorage API symmetry
	#[allow(dead_code)]
	pub fn num_neurons(&self) -> usize {
		match self {
			ClusterStorage::Dense { num_neurons, .. } => *num_neurons,
			ClusterStorage::Sparse { num_neurons, .. } => *num_neurons,
		}
	}

	// KEPT-API: ClusterStorage API symmetry
	#[allow(dead_code)]
	pub fn wpn(&self) -> usize {
		match self {
			ClusterStorage::Dense { words_per_neuron, .. } => *words_per_neuron,
			ClusterStorage::Sparse { .. } => 0,
		}
	}

	/// Actual memory usage in bytes.
	// KEPT-API: ClusterStorage API symmetry
	#[allow(dead_code)]
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

// =============================================================================
// Order-Independent Training (OI) — packed (obs, net) counter
// =============================================================================
//
// Background: the original `nudge_cell_offset` is a clamped random walk on the
// 2-bit cell ∈ [0, 3]. The final cell depends on the *order* of training
// examples (see project_training_clamped_random_walk memory).
//
// OI training fixes this by accumulating a signed `net` and a saturating `obs`
// counter per cell during the training pass, then binning to a 4-state cell at
// commit time. Storage = i32 per touched cell during training (vs 2 bits at
// runtime); the counter buffer is freed after commit.
//
// Packed layout in u32:
//   bit 31:    obs_ge_2     — sticky once obs reaches 2
//   bit 30:    obs_ge_1     — set on any nudge
//   bits 29:0: net (signed 30-bit, range ±2^29 ≈ ±536M, saturating)
//
// Three reachable (obs_ge_2, obs_ge_1) states:
//   (0, 0): untouched (initial)
//   (0, 1): obs == 1 (single observation)
//   (1, 1): obs >= 2 (multiple observations)
// The (1, 0) combination is unreachable by construction.

/// Mask for the signed 30-bit `net` field within the packed counter.
pub const OI_NET_MASK: u32 = 0x3FFF_FFFF;
/// Bit position for `obs_ge_1`.
pub const OI_OBS_GE_1_BIT: u32 = 30;
/// Bit position for `obs_ge_2`.
pub const OI_OBS_GE_2_BIT: u32 = 31;
/// Saturation bounds for the 30-bit signed net.
pub const OI_NET_MAX: i32 = (1 << 29) - 1;
pub const OI_NET_MIN: i32 = -(1 << 29);

/// Initial value of a fresh counter (untouched cell: obs=0, net=0).
pub const OI_INITIAL: u32 = 0;

/// Decompose a packed counter into (net, obs_ge_1, obs_ge_2).
///
/// `net` is sign-extended from the 30-bit signed field to full i32.
#[inline]
pub fn oi_unpack(word: u32) -> (i32, bool, bool) {
	let net30 = word & OI_NET_MASK;
	// Sign-extend 30-bit → 32-bit.
	let net = if net30 & (1 << 29) != 0 {
		(net30 | !OI_NET_MASK) as i32
	} else {
		net30 as i32
	};
	let obs_ge_1 = (word >> OI_OBS_GE_1_BIT) & 1 != 0;
	let obs_ge_2 = (word >> OI_OBS_GE_2_BIT) & 1 != 0;
	(net, obs_ge_1, obs_ge_2)
}

/// Compose (net, obs_ge_1, obs_ge_2) back into a packed counter.
#[inline]
pub fn oi_pack(net: i32, obs_ge_1: bool, obs_ge_2: bool) -> u32 {
	let net30 = (net as u32) & OI_NET_MASK;
	let obs1 = (obs_ge_1 as u32) << OI_OBS_GE_1_BIT;
	let obs2 = (obs_ge_2 as u32) << OI_OBS_GE_2_BIT;
	obs2 | obs1 | net30
}

/// Apply one nudge to a packed counter. Pure function — returns the new packed value.
///
/// `delta` is the signed weight (±class_weight). The result saturates the net
/// at the 30-bit boundary and updates the obs state machine:
///   (0,0) → (0,1) → (1,1) → (1,1) → ...
#[inline]
pub fn oi_apply_nudge(old: u32, delta: i32) -> u32 {
	let (old_net, old_obs1, _old_obs2) = oi_unpack(old);
	let new_net = old_net.saturating_add(delta).clamp(OI_NET_MIN, OI_NET_MAX);
	// obs_ge_1 is always set after any nudge.
	// obs_ge_2 becomes true on the second nudge: it's true if it was already true,
	// or if obs_ge_1 was already true (meaning this isn't the first nudge).
	let new_obs1 = true;
	let new_obs2 = old_obs1; // sticky once set (since obs1 stays set after any nudge)
	oi_pack(new_net, new_obs1, new_obs2)
}

/// Thread-safe nudge on an AtomicU32-packed counter (CAS loop).
///
/// Returns the previous packed value (for diagnostics; callers can ignore).
#[inline]
pub fn oi_nudge_atomic(counter: &AtomicU32, delta: i32) -> u32 {
	let mut old = counter.load(Ordering::Relaxed);
	loop {
		let new = oi_apply_nudge(old, delta);
		match counter.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
			Ok(_) => return old,
			Err(actual) => old = actual,
		}
	}
}

/// Bin a packed counter into the final 4-state cell value at commit time.
///
/// Rule (hybrid: option 1 + force-weak-at-obs=1):
///   obs == 0                  → QUAD_WEAK_FALSE (untouched, initial state)
///   obs == 1, net > 0         → QUAD_WEAK_TRUE  (single observation, positive)
///   obs == 1, net < 0         → QUAD_WEAK_FALSE (single observation, negative — symmetric)
///   obs >= 2, net <= -1       → QUAD_FALSE
///   obs >= 2, net == 0        → QUAD_WEAK_FALSE
///   obs >= 2, net == +1       → QUAD_WEAK_TRUE
///   obs >= 2, net >= +2       → QUAD_TRUE
///
/// Returns a value in {QUAD_FALSE, QUAD_WEAK_FALSE, QUAD_WEAK_TRUE, QUAD_TRUE}.
#[inline]
pub fn oi_bin_to_cell(packed: u32) -> i64 {
	let (net, obs_ge_1, obs_ge_2) = oi_unpack(packed);

	if !obs_ge_1 {
		return QUAD_WEAK_FALSE; // untouched
	}
	if !obs_ge_2 {
		// obs == 1: force WEAK based on sign of net (handles class-weighted cases).
		return if net > 0 { QUAD_WEAK_TRUE } else { QUAD_WEAK_FALSE };
	}
	// obs >= 2: option 1 thresholds.
	if net <= -1 {
		QUAD_FALSE
	} else if net == 0 {
		QUAD_WEAK_FALSE
	} else if net == 1 {
		QUAD_WEAK_TRUE
	} else {
		QUAD_TRUE
	}
}

/// Returns true iff order-independent training is enabled via env var.
///
/// Gated by `WNN_ORDER_INDEPENDENT_TRAIN=1`. Default off until the cohort delta
/// is measured (see project_training_clamped_random_walk).
pub fn order_independent_training_enabled() -> bool {
	std::env::var("WNN_ORDER_INDEPENDENT_TRAIN")
		.map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
		.unwrap_or(false)
}

#[cfg(test)]
mod oi_tests {
	use super::*;

	#[test]
	fn oi_pack_unpack_roundtrip() {
		for &net in &[-(1 << 29), -1, 0, 1, (1 << 29) - 1, 12345, -67890] {
			for &o1 in &[false, true] {
				for &o2 in &[false, true] {
					let packed = oi_pack(net, o1, o2);
					let (n, a, b) = oi_unpack(packed);
					assert_eq!(n, net, "net mismatch for ({}, {}, {})", net, o1, o2);
					assert_eq!(a, o1);
					assert_eq!(b, o2);
				}
			}
		}
	}

	#[test]
	fn oi_apply_nudge_state_machine() {
		// Untouched → first +1 nudge: obs=(0,1), net=+1
		let s1 = oi_apply_nudge(OI_INITIAL, 1);
		assert_eq!(oi_unpack(s1), (1, true, false));

		// Second nudge -1: obs=(1,1), net=0
		let s2 = oi_apply_nudge(s1, -1);
		assert_eq!(oi_unpack(s2), (0, true, true));

		// Third nudge +1: obs stays (1,1), net=+1
		let s3 = oi_apply_nudge(s2, 1);
		assert_eq!(oi_unpack(s3), (1, true, true));
	}

	#[test]
	fn oi_apply_nudge_saturating() {
		let near_max = oi_pack(OI_NET_MAX - 5, true, true);
		let saturated = oi_apply_nudge(near_max, 100);
		let (net, _, _) = oi_unpack(saturated);
		assert_eq!(net, OI_NET_MAX);

		let near_min = oi_pack(OI_NET_MIN + 5, true, true);
		let saturated = oi_apply_nudge(near_min, -100);
		let (net, _, _) = oi_unpack(saturated);
		assert_eq!(net, OI_NET_MIN);
	}

	#[test]
	fn oi_bin_to_cell_rule_table() {
		// Untouched
		assert_eq!(oi_bin_to_cell(oi_pack(0, false, false)), QUAD_WEAK_FALSE);

		// obs == 1
		assert_eq!(oi_bin_to_cell(oi_pack(1, true, false)), QUAD_WEAK_TRUE);
		assert_eq!(oi_bin_to_cell(oi_pack(-1, true, false)), QUAD_WEAK_FALSE);
		assert_eq!(oi_bin_to_cell(oi_pack(12, true, false)), QUAD_WEAK_TRUE);
		assert_eq!(oi_bin_to_cell(oi_pack(-12, true, false)), QUAD_WEAK_FALSE);

		// obs >= 2: option 1 thresholds
		assert_eq!(oi_bin_to_cell(oi_pack(-5, true, true)), QUAD_FALSE);
		assert_eq!(oi_bin_to_cell(oi_pack(-1, true, true)), QUAD_FALSE);
		assert_eq!(oi_bin_to_cell(oi_pack(0, true, true)), QUAD_WEAK_FALSE);
		assert_eq!(oi_bin_to_cell(oi_pack(1, true, true)), QUAD_WEAK_TRUE);
		assert_eq!(oi_bin_to_cell(oi_pack(2, true, true)), QUAD_TRUE);
		assert_eq!(oi_bin_to_cell(oi_pack(100, true, true)), QUAD_TRUE);
	}

	#[test]
	fn oi_atomic_nudge_concurrent() {
		use std::sync::Arc;
		use std::thread;

		let counter = Arc::new(AtomicU32::new(OI_INITIAL));
		let num_threads = 8;
		let nudges_per_thread = 1000;

		let handles: Vec<_> = (0..num_threads).map(|_| {
			let c = counter.clone();
			thread::spawn(move || {
				for _ in 0..nudges_per_thread {
					oi_nudge_atomic(&c, 1);
				}
			})
		}).collect();
		for h in handles { h.join().unwrap(); }

		let (net, o1, o2) = oi_unpack(counter.load(Ordering::Relaxed));
		assert_eq!(net, (num_threads * nudges_per_thread) as i32);
		assert!(o1 && o2);
	}

	// =========================================================================
	// ClusterStorage OI tests (LM single-threaded path)
	// =========================================================================

	fn cluster_train_oi_dense(
		nudges: &[(usize, usize, bool, u32)],
		num_neurons: usize,
		bits: usize,
	) -> Vec<i64> {
		let empty_word = empty_word_for_mode(MODE_QUAD_WEIGHTED);
		let mut storage = ClusterStorage::new(num_neurons, bits, 12, empty_word, MODE_QUAD_WEIGHTED);
		storage.init_oi_counters();
		for &(n, a, t, w) in nudges {
			storage.nudge_cell_oi(n, a, t, w);
		}
		storage.commit_oi();
		let n_addrs = 1usize << bits;
		let mut snap = Vec::with_capacity(num_neurons * n_addrs);
		for n in 0..num_neurons {
			for a in 0..n_addrs {
				snap.push(storage.read_cell(n, a));
			}
		}
		snap
	}

	#[test]
	fn oi_cluster_dense_permutation_invariance() {
		use rand::seq::SliceRandom;
		use rand::SeedableRng;
		use rand::rngs::StdRng;

		let mut nudges: Vec<(usize, usize, bool, u32)> = Vec::new();
		for a in 0..8 {
			for i in 0..(a + 1) {
				nudges.push((0, a, true, if i % 3 == 0 { 3 } else { 1 }));
			}
			for i in 0..(7 - a) {
				nudges.push((0, a, false, if i % 4 == 0 { 2 } else { 1 }));
			}
		}

		let baseline = cluster_train_oi_dense(&nudges, 1, 3);
		for seed in 0..6u64 {
			let mut rng = StdRng::seed_from_u64(seed);
			let mut shuffled = nudges.clone();
			shuffled.shuffle(&mut rng);
			let snap = cluster_train_oi_dense(&shuffled, 1, 3);
			assert_eq!(snap, baseline, "ClusterStorage Dense permutation {} differed", seed);
		}
	}

	#[test]
	fn oi_cluster_dense_bin_oracle() {
		let nudges = vec![
			(0usize, 1usize, true, 1u32),
			(0, 2, false, 5),
			(0, 3, true, 1), (0, 3, true, 1), (0, 3, true, 1), (0, 3, true, 1), (0, 3, true, 1),
			(0, 3, false, 1), (0, 3, false, 1), (0, 3, false, 1),
		];
		let snap = cluster_train_oi_dense(&nudges, 1, 2);
		assert_eq!(snap[0], QUAD_WEAK_FALSE); // untouched
		assert_eq!(snap[1], QUAD_WEAK_TRUE);  // single positive
		assert_eq!(snap[2], QUAD_WEAK_FALSE); // single negative (hybrid)
		assert_eq!(snap[3], QUAD_TRUE);       // 5+ / 3-, net=+2
	}

	fn cluster_train_oi_sparse(
		nudges: &[(usize, u32, bool, u32)],
		num_neurons: usize,
	) -> Vec<(usize, u32, u8)> {
		let empty_word = empty_word_for_mode(MODE_QUAD_WEIGHTED);
		// Force sparse via threshold=4, bits=16.
		let mut storage = ClusterStorage::new(num_neurons, 16, 4, empty_word, MODE_QUAD_WEIGHTED);
		storage.init_oi_counters();
		for &(n, a, t, w) in nudges {
			storage.nudge_cell_oi(n, a as usize, t, w);
		}
		storage.commit_oi();
		let mut snap: Vec<(usize, u32, u8)> = Vec::new();
		if let ClusterStorage::Sparse { neurons, .. } = &storage {
			for (n, map) in neurons.iter().enumerate() {
				for (&k, &v) in map.iter() {
					snap.push((n, k, v));
				}
			}
		}
		snap.sort_unstable();
		snap
	}

	#[test]
	fn oi_cluster_sparse_permutation_invariance() {
		use rand::seq::SliceRandom;
		use rand::SeedableRng;
		use rand::rngs::StdRng;

		let mut nudges: Vec<(usize, u32, bool, u32)> = Vec::new();
		for i in 0..40 {
			let addr = (i as u32) * 0x100;
			for _ in 0..(i % 4 + 1) { nudges.push((0, addr, true, 1)); }
			for _ in 0..(i % 3 + 1) { nudges.push((0, addr, false, if i % 2 == 0 { 2 } else { 1 })); }
		}

		let baseline = cluster_train_oi_sparse(&nudges, 1);
		for seed in 0..5u64 {
			let mut rng = StdRng::seed_from_u64(seed);
			let mut shuffled = nudges.clone();
			shuffled.shuffle(&mut rng);
			let snap = cluster_train_oi_sparse(&shuffled, 1);
			assert_eq!(snap, baseline, "ClusterStorage Sparse permutation {} differed", seed);
		}
	}
}

#[cfg(test)]
mod cell_weight_tests {
	use super::*;

	/// Full mapping table for QUAD modes. Regression for the multistage
	/// CPU-fallback bug (10/06/2026): a raw ternary match scored
	/// WEAK_FALSE (cell 1) as 1.0 and TRUE (cell 3) as empty_value.
	#[test]
	fn quad_weighted_mapping() {
		for mode in [MODE_QUAD_WEIGHTED, MODE_QUAD_BINARY] {
			// empty_value must be ignored in quad modes — pass a poison value.
			let poison = 99.0;
			assert_eq!(cell_to_weight(QUAD_FALSE, mode, poison), 0.0);
			assert_eq!(cell_to_weight(QUAD_WEAK_FALSE, mode, poison), 0.25);
			assert_eq!(cell_to_weight(QUAD_WEAK_TRUE, mode, poison), 0.75);
			assert_eq!(cell_to_weight(QUAD_TRUE, mode, poison), 1.0);
			// Out-of-range cells clamp instead of panicking.
			assert_eq!(cell_to_weight(-1, mode, poison), 0.0);
			assert_eq!(cell_to_weight(7, mode, poison), 1.0);
		}
	}

	#[test]
	fn ternary_mapping() {
		let empty_value = 0.5;
		assert_eq!(cell_to_weight(FALSE, MODE_TERNARY, empty_value), 0.0);
		assert_eq!(cell_to_weight(TRUE, MODE_TERNARY, empty_value), 1.0);
		assert_eq!(cell_to_weight(EMPTY, MODE_TERNARY, empty_value), 0.5);
	}

	/// The exact inversion the bug produced: under QUAD_WEIGHTED, the buggy
	/// `FALSE => 0.0, TRUE => 1.0, _ => empty` match maps cell 1 to 1.0 and
	/// cell 3 to empty_value. Assert the correct helper disagrees with it.
	#[test]
	fn quad_disagrees_with_raw_ternary_match() {
		let empty_value = 0.25;
		let buggy = |cell: i64| -> f32 {
			match cell {
				FALSE => 0.0,
				TRUE => 1.0,
				_ => empty_value,
			}
		};
		assert_ne!(cell_to_weight(QUAD_WEAK_FALSE, MODE_QUAD_WEIGHTED, empty_value), buggy(QUAD_WEAK_FALSE));
		assert_ne!(cell_to_weight(QUAD_WEAK_TRUE, MODE_QUAD_WEIGHTED, empty_value), buggy(QUAD_WEAK_TRUE));
		assert_ne!(cell_to_weight(QUAD_TRUE, MODE_QUAD_WEIGHTED, empty_value), buggy(QUAD_TRUE));
	}
}
