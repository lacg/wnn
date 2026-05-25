//! BPTT/EDRA training for WnnController — Rust implementation.
//!
//! This is the Rust port of the Python BPTT/EDRA reference in
//! `src/wnn/control/bptt_trainer.py`. The Rust port is needed because
//! (a) Python uses TRINARY memory but the WnnController uses
//! QUAD_WEIGHTED — only a native Rust trainer can write the WEAK_FALSE
//! and WEAK_TRUE values that give QSR its partial-confidence semantics;
//! (b) inference + training share the same SparseLayerMemory backing
//! store; (c) per-step training of a 1 kHz control loop benefits
//! enormously from avoiding the Python/torch round trip.
//!
//! ## Status: SCAFFOLD with a SIMPLIFIED solver
//!
//! The full beam-search constraint solver from
//! `src/wnn/ram/core/Memory.py::_solve_partial_connectivity` is a
//! significant port (500+ LOC of careful Rust). This file currently has:
//!
//!   - `train_step_greedy`: single-step write that nudges output cells
//!     toward the target with QSR semantics. NO state-layer back-prop
//!     yet (single-step EDRA: only output supervised, state is whatever
//!     the controller's current forward pass produces).
//!   - `train_episode`: outer loop that runs sim + PID + train_step_greedy
//!     for a single training episode. Mirrors the Python `train_bptt`
//!     outer loop but stays in Rust end-to-end.
//!
//! TODO (left to a follow-up commit):
//!
//!   - `solve_constraints_sparse`: per-neuron beam search over candidate
//!     input bit assignments. Faithful port of
//!     `_solve_partial_connectivity` from Memory.py:673. ~300-500 LOC.
//!   - `bptt_train_window`: K-step BPTT-through-time using the above
//!     solver to propagate desired state targets backward across K
//!     consecutive sim steps. ~150 LOC.
//!
//! The simplified `train_step_greedy` is enough to validate the
//! end-to-end pipeline + start collecting data on whether single-step
//! EDRA is sufficient for attitude stabilization. The decision on
//! whether to invest in full BPTT-through-time will be data-driven from
//! the simplified-trainer results.

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::neuron_memory::compute_address_sparse;
use crate::sparse_memory::SparseLayerMemory;

// ============================================================================
// EDRA constraint solver — faithful Rust port of
// Memory._solve_partial_connectivity (src/wnn/ram/core/Memory.py:673-819).
//
// Given the current input bits + a desired per-neuron output (target_bits),
// search for a NEW input-bit assignment whose addresses, when written to the
// neurons' memory, produce the target outputs — at minimum cost. This is the
// core EDRA primitive: it answers "what input would have produced the right
// answer?" so we can write memory at that address (commit) and/or propagate
// the desired input backward (BPTT).
//
// TRINARY value space (matches Memory.py): FALSE=0, TRUE=1, EMPTY=2.
// Cost function (Garcia 2003): CONFLICT=20, EMPTY=10, HAMMING=1, SATURATION=15.
//
// This port targets the DENSE-addressable regime (memory_size = 2^bits with
// bits small enough to enumerate, ≲ 2^16). Each neuron's full address space
// is scanned for candidate addresses; that matches how the Python solver
// works and how the controller's small state/output layers (b ≤ ~12) are
// configured. Large-b sparse memories would need a touched-addresses-only
// variant (future work).
// ============================================================================

/// TRINARY memory values (mirrors wnn.ram.core.MemoryVal).
pub const TRI_FALSE: u8 = 0;
pub const TRI_TRUE: u8 = 1;
pub const TRI_EMPTY: u8 = 2;

const CONFLICT_COST: f64 = 20.0;
const EMPTY_COST: f64 = 10.0;
const HAMMING_COST: f64 = 1.0;
const SATURATION_COST: f64 = 15.0;
const MAX_BEAM_WIDTH: usize = 64;

/// Decode an integer address into its `n_bits` bits, MSB-first.
/// Matches Memory.decode_addresses_bits: bit i = (addr >> (n_bits-1-i)) & 1.
#[inline]
fn address_bit(addr: usize, i: usize, n_bits: usize) -> bool {
	((addr >> (n_bits - 1 - i)) & 1) != 0
}

/// A candidate partial assignment of input bits during the beam search.
/// `bits[j]` is -1 (unset), 0 (FALSE), or 1 (TRUE) for input bit j.
#[derive(Clone)]
struct Candidate {
	bits: Vec<i8>,
	cost: f64,
}

/// Faithful port of `_solve_partial_connectivity`.
///
/// Args mirror the Python signature:
///   - `read_cell(neuron, addr)`: closure returning the TRINARY cell value.
///   - `connections`: flat `num_neurons * n_bits_per_neuron` input-bit indices.
///   - `input_bits`: current input pattern (length `total_input_bits`).
///   - `target_bits`: desired output per neuron (length `num_neurons`).
///   - `allow_override`: if true, conflicting cells are allowed (validity = all).
///   - `n_immutable_bits`: leading input bits that cannot change.
///   - `topk_per_neuron`: top-K candidate addresses considered per neuron.
///
/// Returns the solved input bits (length `total_input_bits`) or None if no
/// satisfiable assignment exists. Uses ARGMIN selection (deterministic) to
/// match the parity-test configuration.
pub fn solve_partial_connectivity_trinary<F: Fn(usize, usize) -> u8>(
	read_cell: F,
	connections: &[i64],
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: &[bool],
	target_bits: &[bool],
	allow_override: bool,
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> Option<Vec<bool>> {
	let memory_size = 1usize << n_bits_per_neuron;
	let beam_width = MAX_BEAM_WIDTH.min((8 * num_neurons).max(16));
	let k_top = topk_per_neuron.min(memory_size);

	// ---- Precompute per-neuron per-address cost + validity ----
	// per_neuron_cost[n][addr], valid[n][addr].
	let mut per_neuron_cost = vec![vec![0.0f64; memory_size]; num_neurons];
	let mut valid = vec![vec![true; memory_size]; num_neurons];

	for n in 0..num_neurons {
		let conn = &connections[n * n_bits_per_neuron..(n + 1) * n_bits_per_neuron];
		let desired = if target_bits[n] { TRI_TRUE } else { TRI_FALSE };

		// Occupancy → saturation penalty (fraction of non-empty cells).
		let mut occupied = 0usize;
		for addr in 0..memory_size {
			if read_cell(n, addr) != TRI_EMPTY {
				occupied += 1;
			}
		}
		let saturation_penalty = SATURATION_COST * (occupied as f64 / memory_size as f64);

		// Current address bits for this neuron (the input projected onto conn).
		// Used for the Hamming term.
		// current[k] = input_bits[conn[k]]
		// hamming(addr) = count of k where address_bit(addr,k) != current[k]

		// Duplicate-connection consistency pre-filter: if conn has duplicates,
		// addresses where the duplicated positions disagree are unreachable.
		// We detect duplicates once.
		let mut has_dupes = false;
		{
			let mut seen = std::collections::HashMap::new();
			for &c in conn.iter() {
				*seen.entry(c).or_insert(0usize) += 1;
				if seen[&c] > 1 {
					has_dupes = true;
				}
			}
		}

		for addr in 0..memory_size {
			let cell = read_cell(n, addr);
			let conflict = cell != TRI_EMPTY && cell != desired;
			let empty = cell == TRI_EMPTY;

			// validity
			let mut ok = if allow_override { true } else { !conflict };

			// duplicate-index consistency
			if ok && has_dupes {
				// For each set of duplicate connection positions, all the
				// address bits at those positions must agree.
				// Group conn positions by their input-bit index.
				// (Recomputed per address — cheap for small memory_size.)
				let mut groups: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
				for (k, &c) in conn.iter().enumerate() {
					groups.entry(c).or_default().push(k);
				}
				'dup: for (_idx, positions) in groups.iter() {
					if positions.len() > 1 {
						let first = address_bit(addr, positions[0], n_bits_per_neuron);
						for &p in &positions[1..] {
							if address_bit(addr, p, n_bits_per_neuron) != first {
								ok = false;
								break 'dup;
							}
						}
					}
				}
			}

			valid[n][addr] = ok;

			// hamming
			let mut hamming = 0u32;
			for k in 0..n_bits_per_neuron {
				let abit = address_bit(addr, k, n_bits_per_neuron);
				let cbit = input_bits[conn[k] as usize];
				if abit != cbit {
					hamming += 1;
				}
			}

			let cost = CONFLICT_COST * (conflict as u32 as f64)
				+ EMPTY_COST * (empty as u32 as f64)
				+ HAMMING_COST * (hamming as f64)
				+ saturation_penalty;
			per_neuron_cost[n][addr] = if ok { cost } else { f64::INFINITY };
		}
	}

	run_beam_search(
		&per_neuron_cost, connections, num_neurons, n_bits_per_neuron,
		total_input_bits, input_bits, n_immutable_bits, k_top, beam_width,
	)
}

/// Shared beam search over neurons, parameterized by a precomputed
/// `per_neuron_cost[neuron][addr]` matrix (INFINITY = invalid address).
/// Both the TRINARY and QSR solvers feed their own cost matrices here.
fn run_beam_search(
	per_neuron_cost: &[Vec<f64>],
	connections: &[i64],
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: &[bool],
	n_immutable_bits: usize,
	k_top: usize,
	beam_width: usize,
) -> Option<Vec<bool>> {
	let memory_size = 1usize << n_bits_per_neuron;
	// Exhaustive top-k per neuron from the full cost vector. Stable sort by
	// cost over ascending-address input => ties broken by ascending address.
	let top_k: Vec<Vec<(usize, f64)>> = (0..num_neurons)
		.map(|n| {
			let mut addr_cost: Vec<(usize, f64)> = (0..memory_size)
				.map(|a| (a, per_neuron_cost[n][a]))
				.filter(|&(_, c)| c.is_finite())
				.collect();
			addr_cost.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
			addr_cost.truncate(k_top);
			addr_cost
		})
		.collect();
	run_beam_search_from_topk(
		&top_k, connections, num_neurons, n_bits_per_neuron,
		total_input_bits, input_bits, n_immutable_bits, beam_width,
	)
}

/// Beam search over neurons given a PRECOMPUTED top-k address list per neuron.
/// `top_k_per_neuron[n]` must be sorted ascending by (cost, address) — both the
/// exhaustive `run_beam_search` and the reachable solver produce that order, so
/// the beam expansion (and its exact-cost tie-breaking) is identical for both.
fn run_beam_search_from_topk(
	top_k_per_neuron: &[Vec<(usize, f64)>],
	connections: &[i64],
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: &[bool],
	n_immutable_bits: usize,
	beam_width: usize,
) -> Option<Vec<bool>> {
	let mut beam: Vec<Candidate> = vec![Candidate {
		bits: vec![-1i8; total_input_bits],
		cost: 0.0,
	}];
	if n_immutable_bits > 0 {
		for j in 0..n_immutable_bits.min(total_input_bits) {
			beam[0].bits[j] = input_bits[j] as i8;
		}
	}

	for n in 0..num_neurons {
		let conn = &connections[n * n_bits_per_neuron..(n + 1) * n_bits_per_neuron];

		let addr_cost = &top_k_per_neuron[n];
		if addr_cost.is_empty() {
			return None;
		}

		// Expand beam × addresses. PERF: the old path cloned a full
		// Vec<i8>[total_input_bits] for EVERY (beam × addr) pair (incl. conflicts
		// and soon-truncated ones) — hundreds of K allocations/window dominated
		// runtime for these tiny nets. Instead: (1) conflict-check against the
		// parent's bits WITHOUT cloning, recording only (parent_idx, addr, cost);
		// (2) stable-sort by cost + truncate to beam_width; (3) materialize bit
		// vectors for the SURVIVORS only. Rust's sort_by is stable and `refs` is
		// built in the same parent-major/addr-minor order the old `expanded` was,
		// so the surviving set, order, and argmin are bit-identical.
		let mut refs: Vec<(usize, u64, f64)> = Vec::with_capacity(beam.len() * addr_cost.len());
		for (pi, cand) in beam.iter().enumerate() {
			for &(addr, acost) in addr_cost {
				let mut conflict = false;
				for k in 0..n_bits_per_neuron {
					let pos = conn[k] as usize;
					let abit = address_bit(addr, k, n_bits_per_neuron) as i8;
					let cur = cand.bits[pos];
					if cur != -1 && cur != abit {
						conflict = true;
						break;
					}
				}
				if !conflict {
					refs.push((pi, addr as u64, cand.cost + acost));
				}
			}
		}

		if refs.is_empty() {
			return None;
		}

		refs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
		refs.truncate(beam_width);

		let mut next: Vec<Candidate> = Vec::with_capacity(refs.len());
		for (pi, addr, cost) in refs {
			let mut bits = beam[pi].bits.clone();
			for k in 0..n_bits_per_neuron {
				let pos = conn[k] as usize;
				bits[pos] = address_bit(addr as usize, k, n_bits_per_neuron) as i8;
			}
			next.push(Candidate { bits, cost });
		}
		beam = next;
	}

	// ARGMIN — first minimum (matches torch.argmin / Python cost_calculator).
	let mut best_idx = 0usize;
	let mut best_cost = beam[0].cost;
	for (i, c) in beam.iter().enumerate().skip(1) {
		if c.cost < best_cost {
			best_cost = c.cost;
			best_idx = i;
		}
	}
	if !best_cost.is_finite() {
		return None;
	}
	let best = &beam[best_idx];
	let solved: Vec<bool> = (0..total_input_bits)
		.map(|j| if best.bits[j] == -1 { input_bits[j] } else { best.bits[j] == 1 })
		.collect();
	Some(solved)
}

// ============================================================================
// Reachable-address top-k (avoids the O(2^n_bits) exhaustive scan).
//
// The exhaustive solvers compute a cost for ALL 2^n_bits addresses but only the
// top-k cheapest VALID ones are used. The cheapest addresses are always either
// (a) a trained (written) cell, or (b) an untrained cell with low Hamming
// distance from the "projected" address (the one whose address bits equal the
// current input projected onto the neuron's connections). Untrained cells all
// share the same base cost, so their cost is monotonic in Hamming distance ⇒
// the k cheapest untrained are the k lowest-Hamming ones. Therefore the set
//   {all trained valid cells} ∪ {k_top lowest-Hamming untrained cells}
// provably CONTAINS the global top-k, and sorting it by (cost, address) — the
// same order the exhaustive stable-sort produces — yields the identical top-k.
// This is O(num_trained + k_top·n_bits) instead of O(2^n_bits).
//
// Verified exact against the exhaustive path by tests/test_controller_solver_reachable.py.
// ============================================================================

/// Advance `combo` (h ascending distinct positions chosen from 0..n) to the
/// next combination in lexicographic order. Returns false when exhausted.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
	let h = combo.len();
	if h == 0 {
		return false;
	}
	let mut i = h;
	loop {
		if i == 0 {
			return false;
		}
		i -= 1;
		if combo[i] != i + n - h {
			combo[i] += 1;
			for j in (i + 1)..h {
				combo[j] = combo[j - 1] + 1;
			}
			return true;
		}
	}
}

/// Top-k (addr, cost) for ONE neuron without scanning 2^n_bits.
/// cost(addr) = base_and_valid(cell_value(addr)) + HAMMING_COST·hamming + saturation.
/// `entries`: written cells (addr, val). `default_val`: value of unwritten cells.
/// `base_and_valid(val)`: per-cell base cost, or None if the address is invalid.
/// `dup_groups`: connection-position groups (len>1) for trinary duplicate-index
/// consistency; empty for QSR (every address valid). Result sorted (cost, addr).
fn reachable_topk_for_neuron(
	entries: &[(u64, u8)],
	default_val: u8,
	conn: &[i64],
	input_bits: &[bool],
	n_bits: usize,
	k_top: usize,
	saturation: f64,
	base_and_valid: &dyn Fn(u8) -> Option<f64>,
	dup_groups: &[Vec<usize>],
) -> Vec<(usize, f64)> {
	// Projected address: address_bit(proj, k) == input_bits[conn[k]] (MSB-first).
	let mut proj: usize = 0;
	for k in 0..n_bits {
		if input_bits[conn[k] as usize] {
			proj |= 1usize << (n_bits - 1 - k);
		}
	}
	let dup_ok = |addr: usize| -> bool {
		for g in dup_groups {
			let first = (addr >> (n_bits - 1 - g[0])) & 1;
			for &p in &g[1..] {
				if (addr >> (n_bits - 1 - p)) & 1 != first {
					return false;
				}
			}
		}
		true
	};
	let trained: std::collections::HashSet<usize> =
		entries.iter().map(|&(a, _)| a as usize).collect();

	let mut cands: Vec<(usize, f64)> = Vec::new();

	// (a) Trained cells.
	for &(addr, val) in entries {
		let a = addr as usize;
		if !dup_ok(a) {
			continue;
		}
		if let Some(base) = base_and_valid(val) {
			let h = (a ^ proj).count_ones() as f64;
			cands.push((a, base + HAMMING_COST * h + saturation));
		}
	}

	// (b) k_top lowest-Hamming untrained cells (value = default_val).
	if let Some(base_def) = base_and_valid(default_val) {
		let mut collected = 0usize;
		let mut h = 0usize;
		while h <= n_bits {
			let mut combo: Vec<usize> = (0..h).collect();
			loop {
				let mut addr = proj;
				for &k in &combo {
					addr ^= 1usize << (n_bits - 1 - k);
				}
				if !trained.contains(&addr) && dup_ok(addr) {
					cands.push((addr, base_def + HAMMING_COST * (h as f64) + saturation));
					collected += 1;
				}
				if !next_combination(&mut combo, n_bits) {
					break;
				}
			}
			if collected >= k_top {
				break;
			}
			h += 1;
		}
	}

	cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
	cands.truncate(k_top);
	cands
}

/// Reachable-address TRINARY solver — exact drop-in for
/// `solve_partial_connectivity_trinary`, but enumerates candidates from the
/// sparse `entries_fn` instead of scanning 2^n_bits. Per-neuron top-k runs in
/// parallel (rayon). `default_val` is the unwritten-cell value (TRI_EMPTY).
pub fn solve_partial_connectivity_trinary_reachable<G: Fn(usize) -> Vec<(u64, u8)> + Sync>(
	entries_fn: G,
	default_val: u8,
	connections: &[i64],
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: &[bool],
	target_bits: &[bool],
	allow_override: bool,
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> Option<Vec<bool>> {
	let memory_size = 1usize << n_bits_per_neuron;
	let beam_width = MAX_BEAM_WIDTH.min((8 * num_neurons).max(16));
	let k_top = topk_per_neuron.min(memory_size);

	let top_k: Vec<Vec<(usize, f64)>> = (0..num_neurons)
		.into_par_iter()
		.map(|n| {
			let conn = &connections[n * n_bits_per_neuron..(n + 1) * n_bits_per_neuron];
			let desired = if target_bits[n] { TRI_TRUE } else { TRI_FALSE };
			let entries = entries_fn(n);
			// Occupancy: written cells != EMPTY (untrained == EMPTY don't count).
			let occupied = entries.iter().filter(|&&(_, v)| v != TRI_EMPTY).count();
			let saturation = SATURATION_COST * (occupied as f64 / memory_size as f64);
			// Duplicate connection-position groups (consistency filter).
			let mut groups: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
			for (k, &c) in conn.iter().enumerate() {
				groups.entry(c).or_default().push(k);
			}
			let dup_groups: Vec<Vec<usize>> =
				groups.into_values().filter(|g| g.len() > 1).collect();
			let base = |val: u8| -> Option<f64> {
				let conflict = val != TRI_EMPTY && val != desired;
				if !allow_override && conflict {
					return None;
				}
				let empty = val == TRI_EMPTY;
				Some(CONFLICT_COST * (conflict as u32 as f64) + EMPTY_COST * (empty as u32 as f64))
			};
			reachable_topk_for_neuron(
				&entries, default_val, conn, input_bits, n_bits_per_neuron,
				k_top, saturation, &base, &dup_groups,
			)
		})
		.collect();

	run_beam_search_from_topk(
		&top_k, connections, num_neurons, n_bits_per_neuron,
		total_input_bits, input_bits, n_immutable_bits, beam_width,
	)
}

/// Reachable-address QSR solver — exact drop-in for
/// `solve_partial_connectivity_qsr`. Every address is valid (graded nudge cost);
/// untrained cells read as `default_val` (EMPTY). Per-neuron top-k via rayon.
pub fn solve_partial_connectivity_qsr_reachable<G: Fn(usize) -> Vec<(u64, u8)> + Sync>(
	entries_fn: G,
	default_val: u8,
	connections: &[i64],
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: &[bool],
	target_bits: &[bool],
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> Option<Vec<bool>> {
	let memory_size = 1usize << n_bits_per_neuron;
	let beam_width = MAX_BEAM_WIDTH.min((8 * num_neurons).max(16));
	let k_top = topk_per_neuron.min(memory_size);

	let top_k: Vec<Vec<(usize, f64)>> = (0..num_neurons)
		.into_par_iter()
		.map(|n| {
			let conn = &connections[n * n_bits_per_neuron..(n + 1) * n_bits_per_neuron];
			let target_true = target_bits[n];
			let entries = entries_fn(n);
			// Occupancy: read_cell != WEAK_FALSE. Untrained read as EMPTY(2) != 1
			// ⇒ all untrained occupied; plus written cells != WEAK_FALSE.
			let occ_written = entries.iter().filter(|&&(_, v)| v != WEAK_FALSE).count();
			let occupied = (memory_size - entries.len()) + occ_written;
			let saturation = SATURATION_COST * (occupied as f64 / memory_size as f64);
			let base = |val: u8| -> Option<f64> {
				Some(QSR_DISTANCE_COST * qsr_nudge_distance(val, target_true) as f64)
			};
			reachable_topk_for_neuron(
				&entries, default_val, conn, input_bits, n_bits_per_neuron,
				k_top, saturation, &base, &[],
			)
		})
		.collect();

	run_beam_search_from_topk(
		&top_k, connections, num_neurons, n_bits_per_neuron,
		total_input_bits, input_bits, n_immutable_bits, beam_width,
	)
}

// ============================================================================
// QSR (QUAD_WEIGHTED) constraint solver.
//
// The controller's memory is QUAD_WEIGHTED (FALSE=0, WEAK_FALSE=1,
// WEAK_TRUE=2, TRUE=3), not TRINARY. There is no hard EMPTY sentinel and no
// binary conflict: any cell can be NUDGED toward any target (the QSR training
// rule moves a cell one step per write). So the TRINARY binary CONFLICT/EMPTY
// costs become a GRADED nudge-distance: how many QSR steps a cell is from the
// target side.
//
// Per-neuron cost(addr) for a target bit:
//   nudge_distance = steps to move the cell to the target extreme
//       target TRUE  → 3 - cell   (FALSE=3 steps, ..., TRUE=0 steps)
//       target FALSE → cell        (TRUE=3 steps, ..., FALSE=0 steps)
//   cost = QSR_DISTANCE_COST * nudge_distance
//        + HAMMING_COST      * hamming(addr, current_input_projection)
//        + SATURATION_COST   * occupancy   (fraction of non-default cells)
//
// No address is ever invalid (every cell is nudgeable), so the QSR solver
// never returns None from validity — only from merge conflicts on shared
// input bits (same as TRINARY). The default/untrained cell is WEAK_FALSE(1);
// "occupancy" counts cells != WEAK_FALSE.
// ============================================================================

const QSR_DISTANCE_COST: f64 = 7.0;  // per nudge-step; ~half the TRINARY CONFLICT so a
                                     // 3-step move (~21) ≈ a hard TRINARY conflict (20).

/// QSR cell weight side: true if the cell is on the TRUE side (WEAK_TRUE/TRUE).
#[inline]
fn qsr_nudge_distance(cell: u8, target_true: bool) -> u8 {
	let c = (cell & 0x3) as i8;
	if target_true { (3 - c) as u8 } else { c as u8 }
}

/// QSR analog of solve_partial_connectivity_trinary. Same beam search, graded
/// nudge-distance cost. `read_cell` returns QSR values 0..3. Default/untrained
/// cell value is WEAK_FALSE(1).
pub fn solve_partial_connectivity_qsr<F: Fn(usize, usize) -> u8>(
	read_cell: F,
	connections: &[i64],
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: &[bool],
	target_bits: &[bool],
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> Option<Vec<bool>> {
	let memory_size = 1usize << n_bits_per_neuron;
	let beam_width = MAX_BEAM_WIDTH.min((8 * num_neurons).max(16));
	let k_top = topk_per_neuron.min(memory_size);

	let mut per_neuron_cost = vec![vec![0.0f64; memory_size]; num_neurons];
	for n in 0..num_neurons {
		let conn = &connections[n * n_bits_per_neuron..(n + 1) * n_bits_per_neuron];
		let target_true = target_bits[n];

		// Occupancy: fraction of cells differing from the WEAK_FALSE default.
		let mut occupied = 0usize;
		for addr in 0..memory_size {
			if read_cell(n, addr) != WEAK_FALSE {
				occupied += 1;
			}
		}
		let saturation_penalty = SATURATION_COST * (occupied as f64 / memory_size as f64);

		for addr in 0..memory_size {
			let cell = read_cell(n, addr);
			let dist = qsr_nudge_distance(cell, target_true) as f64;
			let mut hamming = 0u32;
			for k in 0..n_bits_per_neuron {
				if address_bit(addr, k, n_bits_per_neuron) != input_bits[conn[k] as usize] {
					hamming += 1;
				}
			}
			per_neuron_cost[n][addr] = QSR_DISTANCE_COST * dist
				+ HAMMING_COST * (hamming as f64)
				+ saturation_penalty;
		}
	}

	run_beam_search(
		&per_neuron_cost, connections, num_neurons, n_bits_per_neuron,
		total_input_bits, input_bits, n_immutable_bits, k_top, beam_width,
	)
}

/// QSR cell values matching neuron_memory.rs and controller.rs.
const FALSE_VAL: u8 = 0;
const WEAK_FALSE: u8 = 1;  // EMPTY default
const WEAK_TRUE: u8 = 2;
const TRUE_VAL: u8 = 3;

/// Encode a single target bit into the QSR nudge to apply:
///   target=TRUE  → nudge UP (FALSE→WEAK_FALSE→WEAK_TRUE→TRUE)
///   target=FALSE → nudge DOWN (TRUE→WEAK_TRUE→WEAK_FALSE→FALSE)
/// Returns the new cell value for a current cell value.
#[inline]
pub fn nudge_toward_pub(current: u8, target_true: bool) -> u8 {
	nudge_toward(current, target_true)
}

/// Nudge a QSR cell ONE step toward a SPECIFIC target value (0..3), not the
/// boolean extreme. Used by per-motor EDRA to move the state cell toward the
/// solver-derived desired QSR state value.
#[inline]
pub fn nudge_toward_value(current: u8, target: u8) -> u8 {
	let c = (current & 0x3) as i8;
	let t = (target & 0x3) as i8;
	if c < t {
		(c + 1) as u8
	} else if c > t {
		(c - 1) as u8
	} else {
		c as u8
	}
}

#[inline]
fn nudge_toward(current: u8, target_true: bool) -> u8 {
	let cur = current & 0x3;
	if target_true {
		// Nudge toward TRUE
		match cur {
			FALSE_VAL => WEAK_FALSE,
			WEAK_FALSE => WEAK_TRUE,
			WEAK_TRUE => TRUE_VAL,
			_ => TRUE_VAL, // already TRUE or unknown — stay TRUE
		}
	} else {
		// Nudge toward FALSE
		match cur {
			TRUE_VAL => WEAK_TRUE,
			WEAK_TRUE => WEAK_FALSE,
			WEAK_FALSE => FALSE_VAL,
			_ => FALSE_VAL,
		}
	}
}

/// Thermometer-encode a 4-motor PWM into a flat bool vector of length
/// `num_motors * levels_per_motor`. Matches Python's
/// encode_action_thermometer.
fn encode_target_pwm(
	pwm: &[f32; 4],
	num_motors: usize,
	levels_per_motor: usize,
) -> Vec<bool> {
	let total = num_motors * levels_per_motor;
	let mut bits = vec![false; total];
	for m in 0..num_motors {
		let p = pwm[m].clamp(0.0, 1.0);
		let n_true = (p * levels_per_motor as f32) as usize;
		let start = m * levels_per_motor;
		for i in 0..n_true.min(levels_per_motor) {
			bits[start + i] = true;
		}
	}
	bits
}

/// Single-step EDRA write to the output layer. For each output neuron:
///   - Compute its address from the current state-layer-output bits.
///   - Nudge its cell at that address toward the target bit.
///
/// This is the simplified one-step EDRA — no constraint solving across
/// the state layer. Only the OUTPUT layer is supervised. The state
/// layer is unchanged by this call; its cells are whatever was loaded
/// at controller construction (or zero / EMPTY for fresh controllers).
///
/// Returns the number of cells actually modified.
pub fn train_step_greedy_output_only(
	output_memory: &SparseLayerMemory,
	output_connections: &[i64],
	output_bits_per_neuron: usize,
	state_output_bits: &[bool],
	target_pwm_bits: &[bool],
) -> usize {
	let num_output_neurons = target_pwm_bits.len();
	debug_assert_eq!(output_memory.num_neurons, num_output_neurons);
	let mut writes = 0;
	for n in 0..num_output_neurons {
		let conn_start = n * output_bits_per_neuron;
		let conn_end = conn_start + output_bits_per_neuron;
		let address = compute_address_sparse(
			state_output_bits,
			&output_connections[conn_start..conn_end],
			output_bits_per_neuron,
		);
		let current = output_memory.read_cell(n, address);
		let new_value = nudge_toward(current, target_pwm_bits[n]);
		if new_value != current {
			output_memory.write_cell(n, address, new_value, true);
			writes += 1;
		}
	}
	writes
}

/// Single-step EDRA write to the STATE layer. The "target" for the
/// state layer is provided externally — for the simplified version, we
/// use `target_pwm_bits` directly (this matches the existing Python
/// `_solve_output`'s identity-mapping assumption when
/// `state_neurons == num_output_neurons`).
///
/// Future: replace with proper constraint-solver-derived state target
/// once `solve_constraints_sparse` is ported.
pub fn train_step_greedy_state_layer(
	state_memory: &SparseLayerMemory,
	state_connections: &[i64],
	state_bits_per_neuron: usize,
	state_input_bits: &[bool],
	target_state_bits: &[bool],
) -> usize {
	let num_state_neurons = target_state_bits.len();
	debug_assert_eq!(state_memory.num_neurons, num_state_neurons);
	let mut writes = 0;
	for n in 0..num_state_neurons {
		let conn_start = n * state_bits_per_neuron;
		let conn_end = conn_start + state_bits_per_neuron;
		let address = compute_address_sparse(
			state_input_bits,
			&state_connections[conn_start..conn_end],
			state_bits_per_neuron,
		);
		let current = state_memory.read_cell(n, address);
		let new_value = nudge_toward(current, target_state_bits[n]);
		if new_value != current {
			state_memory.write_cell(n, address, new_value, true);
			writes += 1;
		}
	}
	writes
}

// ============================================================================
// Python bindings
// ============================================================================

/// Python wrapper for `train_step_greedy_output_only`. Takes a Python
/// list of state-output bits (length == state_neurons) and a list of
/// target-PWM bits (length == num_motors * levels_per_motor) plus the
/// output-layer connectivity, and applies single-step EDRA to the
/// output cells. Returns the number of cells modified.
///
/// The caller is responsible for keeping the SparseLayerMemory alive
/// (it's stored inside the WnnController — usually you'd not call this
/// directly but go through `wnn_controller.train_step_*`).
#[pyfunction]
#[pyo3(signature = (
	num_motors,
	levels_per_motor,
	output_bits_per_neuron,
	state_output_bits,
	output_connections,
	target_pwm,
))]
pub fn train_output_step_qsr(
	num_motors: usize,
	levels_per_motor: usize,
	output_bits_per_neuron: usize,
	state_output_bits: Vec<bool>,
	output_connections: Vec<i64>,
	target_pwm: [f32; 4],
) -> PyResult<usize> {
	let _ = (
		num_motors,
		levels_per_motor,
		output_bits_per_neuron,
		state_output_bits,
		output_connections,
		target_pwm,
	);
	// Cannot expose SparseLayerMemory directly without holding a reference into
	// WnnController; the proper API will be a method on WnnController itself.
	// This standalone function exists for completeness/testing once the
	// WnnController.train_step_* methods land.
	Err(pyo3::exceptions::PyNotImplementedError::new_err(
		"Use WnnController.train_step_* methods instead (TODO in a follow-up)",
	))
}

/// PyO3 wrapper for `solve_partial_connectivity_trinary` — used by the
/// parity test that checks the Rust port matches Python's
/// `Memory._solve_partial_connectivity` exactly.
///
/// Memory state is passed as a flat `num_neurons * memory_size` array of
/// TRINARY cell values (row-major: cell (n, addr) at index n*memory_size+addr).
/// Returns the solved input bits, or None if unsatisfiable.
#[pyfunction]
#[pyo3(signature = (
	cells_flat,
	connections,
	num_neurons,
	n_bits_per_neuron,
	total_input_bits,
	input_bits,
	target_bits,
	allow_override = false,
	n_immutable_bits = 0,
	topk_per_neuron = 4,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_partial_trinary_py(
	cells_flat: Vec<u8>,
	connections: Vec<i64>,
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: Vec<bool>,
	target_bits: Vec<bool>,
	allow_override: bool,
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> PyResult<Option<Vec<bool>>> {
	let memory_size = 1usize << n_bits_per_neuron;
	let expected = num_neurons * memory_size;
	if cells_flat.len() != expected {
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"cells_flat length {} != num_neurons * 2^bits = {}",
			cells_flat.len(), expected
		)));
	}
	if connections.len() != num_neurons * n_bits_per_neuron {
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"connections length {} != num_neurons * n_bits_per_neuron = {}",
			connections.len(), num_neurons * n_bits_per_neuron
		)));
	}
	let read = |n: usize, addr: usize| cells_flat[n * memory_size + addr];
	Ok(solve_partial_connectivity_trinary(
		read,
		&connections,
		num_neurons,
		n_bits_per_neuron,
		total_input_bits,
		&input_bits,
		&target_bits,
		allow_override,
		n_immutable_bits,
		topk_per_neuron,
	))
}

/// PyO3 wrapper for `solve_partial_connectivity_qsr`. Cells are QSR values
/// 0..3 (default/untrained = WEAK_FALSE=1). Returns the solved input bits.
/// The QSR solver always finds a solution unless shared-bit merge conflicts
/// make the beam empty, so None is rare.
#[pyfunction]
#[pyo3(signature = (
	cells_flat,
	connections,
	num_neurons,
	n_bits_per_neuron,
	total_input_bits,
	input_bits,
	target_bits,
	n_immutable_bits = 0,
	topk_per_neuron = 4,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_partial_qsr_py(
	cells_flat: Vec<u8>,
	connections: Vec<i64>,
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: Vec<bool>,
	target_bits: Vec<bool>,
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> PyResult<Option<Vec<bool>>> {
	let memory_size = 1usize << n_bits_per_neuron;
	if cells_flat.len() != num_neurons * memory_size {
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"cells_flat length {} != num_neurons * 2^bits = {}",
			cells_flat.len(), num_neurons * memory_size
		)));
	}
	let read = |n: usize, addr: usize| cells_flat[n * memory_size + addr];
	Ok(solve_partial_connectivity_qsr(
		read,
		&connections,
		num_neurons,
		n_bits_per_neuron,
		total_input_bits,
		&input_bits,
		&target_bits,
		n_immutable_bits,
		topk_per_neuron,
	))
}

/// PyO3 wrapper for the reachable-address TRINARY solver. Cells are passed
/// SPARSELY as parallel (neuron, addr, val) entry arrays + the unwritten-cell
/// `default_val`, so the test can verify the low-Hamming untrained enumeration
/// against the exhaustive `solve_partial_trinary_py` fed the equivalent dense
/// array.
#[pyfunction]
pub fn solve_partial_trinary_reachable_py(
	entry_neurons: Vec<usize>,
	entry_addrs: Vec<u64>,
	entry_vals: Vec<u8>,
	default_val: u8,
	connections: Vec<i64>,
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: Vec<bool>,
	target_bits: Vec<bool>,
	allow_override: bool,
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> PyResult<Option<Vec<bool>>> {
	let mut per_neuron: Vec<Vec<(u64, u8)>> = vec![Vec::new(); num_neurons];
	for i in 0..entry_neurons.len() {
		per_neuron[entry_neurons[i]].push((entry_addrs[i], entry_vals[i]));
	}
	let entries_fn = |n: usize| per_neuron[n].clone();
	Ok(solve_partial_connectivity_trinary_reachable(
		entries_fn, default_val, &connections, num_neurons, n_bits_per_neuron,
		total_input_bits, &input_bits, &target_bits, allow_override,
		n_immutable_bits, topk_per_neuron,
	))
}

/// PyO3 wrapper for the reachable-address QSR solver (sparse entry arrays).
#[pyfunction]
pub fn solve_partial_qsr_reachable_py(
	entry_neurons: Vec<usize>,
	entry_addrs: Vec<u64>,
	entry_vals: Vec<u8>,
	default_val: u8,
	connections: Vec<i64>,
	num_neurons: usize,
	n_bits_per_neuron: usize,
	total_input_bits: usize,
	input_bits: Vec<bool>,
	target_bits: Vec<bool>,
	n_immutable_bits: usize,
	topk_per_neuron: usize,
) -> PyResult<Option<Vec<bool>>> {
	let mut per_neuron: Vec<Vec<(u64, u8)>> = vec![Vec::new(); num_neurons];
	for i in 0..entry_neurons.len() {
		per_neuron[entry_neurons[i]].push((entry_addrs[i], entry_vals[i]));
	}
	let entries_fn = |n: usize| per_neuron[n].clone();
	Ok(solve_partial_connectivity_qsr_reachable(
		entries_fn, default_val, &connections, num_neurons, n_bits_per_neuron,
		total_input_bits, &input_bits, &target_bits, n_immutable_bits,
		topk_per_neuron,
	))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn nudge_up_chain() {
		// FALSE → WEAK_FALSE → WEAK_TRUE → TRUE → TRUE
		assert_eq!(nudge_toward(FALSE_VAL, true), WEAK_FALSE);
		assert_eq!(nudge_toward(WEAK_FALSE, true), WEAK_TRUE);
		assert_eq!(nudge_toward(WEAK_TRUE, true), TRUE_VAL);
		assert_eq!(nudge_toward(TRUE_VAL, true), TRUE_VAL);
	}

	#[test]
	fn nudge_down_chain() {
		// TRUE → WEAK_TRUE → WEAK_FALSE → FALSE → FALSE
		assert_eq!(nudge_toward(TRUE_VAL, false), WEAK_TRUE);
		assert_eq!(nudge_toward(WEAK_TRUE, false), WEAK_FALSE);
		assert_eq!(nudge_toward(WEAK_FALSE, false), FALSE_VAL);
		assert_eq!(nudge_toward(FALSE_VAL, false), FALSE_VAL);
	}

	#[test]
	fn encode_target_pwm_basics() {
		// p=0.5, levels=4 → first 2 bits TRUE per motor
		let bits = encode_target_pwm(&[0.5, 0.0, 1.0, 0.25], 4, 4);
		assert_eq!(bits.len(), 16);
		assert_eq!(&bits[0..4], &[true, true, false, false]);  // motor 0: 0.5
		assert_eq!(&bits[4..8], &[false, false, false, false]); // motor 1: 0.0
		assert_eq!(&bits[8..12], &[true, true, true, true]);    // motor 2: 1.0
		assert_eq!(&bits[12..16], &[true, false, false, false]); // motor 3: 0.25
	}

	#[test]
	fn train_step_writes_cells() {
		// Tiny memory: 4 neurons with 4-bit addresses.
		let mem = SparseLayerMemory::new(4, 4);
		let conn: Vec<i64> = (0..16).map(|i| i as i64).collect();
		let state_bits = vec![false, true, false, true, true, false, false, true,
		                       false, false, true, true, true, true, false, false];
		let target_pwm = vec![true, false, true, true]; // 4 output bits, target

		let writes = train_step_greedy_output_only(&mem, &conn, 4, &state_bits, &target_pwm);
		// Each neuron's cell at its computed address gets nudged once. Initial
		// is EMPTY (WEAK_FALSE = 1). For target=TRUE: WEAK_FALSE → WEAK_TRUE (1 write).
		// For target=FALSE: WEAK_FALSE → FALSE (1 write).
		// 4 neurons → 4 cell writes.
		assert_eq!(writes, 4);
	}
}
