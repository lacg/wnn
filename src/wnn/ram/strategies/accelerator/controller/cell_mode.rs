//! Memory-mode-aware cell semantics for the controller (ABI 12).
//!
//! The controller substrate historically hardcoded QUAD_WEIGHTED (4-state
//! nudging cells). For the neuron-format granularity ablation (Luiz, approved
//! 12/07/2026) the controller now supports:
//!
//! - QUAD_WEIGHTED / QUAD_BINARY (2): the historical semantics —
//!   cells 0..3, weights [0, .25, .75, 1], neutral = QUAD_WEIGHTS[EMPTY_U8]
//!   = 0.75, one-step nudge training. Bit-identical to pre-ABI-12.
//! - TERNARY (0): cells FALSE(0)/TRUE(1)/EMPTY(2), weights
//!   [0, 1, TERNARY_EMPTY_VALUE=0.5] — the deterministic expected-value PLN
//!   convention (untrained cell ≡ a fair u-state coin). Neutral = 0.5.
//! - BINARY (3): classical WiSARD 1-bit cells, weight TRUE→1 else 0.
//!   Untrained decode is the FLOOR (0.0) — a single thermometer bank could
//!   only ever push one way. The controller therefore uses ANTAGONIST PAIRS
//!   (the classical RAM-network push-pull trick): each motor's `levels`
//!   output cells split into an excitatory half E (levels 0..L/2) and an
//!   inhibitory half I (levels L/2..L), decoded as
//!       decoded = 0.5 + (ΣE − ΣI) / levels ∈ [0, 1]
//!   so the EFFECTIVE neutral is 0.5 and an untrained bank decodes EXACTLY
//!   neutral. Genome layout is unchanged vs QUAD/TERNARY (same neuron count,
//!   same connection arrays) → seed-matched ablations stay clean, and at
//!   1 bit/cell vs QUAD's 2 the storage halves.
//!
//! TRAINING SEMANTICS DIFFER FROM THE IDS WORKER ON PURPOSE: the worker's
//! TERNARY is a TRUE-wins lattice and BINARY is one-shot set-TRUE (order-
//! independent batch training over a fixed dataset). The controller's DAGGER
//! loop must be able to CORRECT earlier labels as the teacher distribution
//! shifts, so TERNARY/BINARY controller training is LAST-WRITE-WINS direct
//! set (the mode-analog of QUAD's bidirectional nudge). Single-writer on both
//! CPU and GPU (thread-per-genome), so order is deterministic.
//!
//! Everything here is the single source of truth; the Metal twins live in
//! controller_rollout.metal (same mode constants via ram_core).

use ram_core::neuron_memory::{
	EMPTY_U8, FALSE_U8, TRUE_U8, BINARY, QUAD_BINARY, QUAD_WEIGHTED, QSR, PLN,
	TERNARY, QUAD_WEIGHTS,
};

/// TERNARY untrained-cell decode weight — the PLN expected-value convention
/// (Luiz rule 12/07/2026: the controller ALWAYS runs TERNARY at 0.5; with
/// 0.0 the neutral would sit at the decode floor and hover is unlearnable).
pub const TERNARY_EMPTY_VALUE: f32 = 0.5;

#[inline]
pub fn is_quad(mode: u8) -> bool {
	// QSR is QUAD in every respect except the read → shares all lattice/training
	// logic gated by is_quad (fire-bit, nudge, plant, reservoir, default).
	mode == QUAD_WEIGHTED || mode == QUAD_BINARY || mode == QSR
}

/// Validate a controller memory mode at construction — refuse unknown values
/// loudly instead of silently decoding garbage.
pub fn validate_mode(mode: u8) -> Result<(), String> {
	match mode {
		TERNARY | QUAD_BINARY | QUAD_WEIGHTED | QSR | BINARY | PLN => Ok(()),
		_ => Err(format!(
			"unknown memory_mode {mode} (TERNARY=0, QUAD_BINARY=1, QUAD_WEIGHTED=2, BINARY=3, QSR=4, PLN=5)"
		)),
	}
}

/// The untrained-cell decode value — the delta-control neutral AND the
/// residual anchor (ABI 11 generalized). QUAD: QUAD_WEIGHTS[EMPTY_U8]=0.75.
/// TERNARY: the 0.5 PLN convention. BINARY: 0.5 — NOT the raw cell floor,
/// because the antagonist decode remaps ΣE−ΣI=0 (untrained) to 0.5.
#[inline]
pub fn neutral_decode(mode: u8) -> f32 {
	match mode {
		// PLN shares TERNARY's 3-state cells; its stochastic u-coin has expected
		// value 0.5, so the neutral (and residual anchor) is the same 0.5.
		TERNARY | PLN => TERNARY_EMPTY_VALUE,
		BINARY => 0.5,
		_ => QUAD_WEIGHTS[EMPTY_U8 as usize],
	}
}

/// Cell → forward weight. Delegates to the substrate's canonical
/// cell_to_weight with the controller's fixed TERNARY empty value.
#[inline]
pub fn cell_weight(cell: u8, mode: u8) -> f32 {
	ram_core::neuron_memory::cell_to_weight(cell as i64, mode, TERNARY_EMPTY_VALUE)
}

/// True-firing probability of a cell for the STOCHASTIC modes (QSR/PLN) — the p
/// in "fire 1.0 with probability p, else 0.0". QSR: the graded QUAD weight
/// (WEAK_FALSE=.25, WEAK_TRUE=.75). PLN: FALSE→0, TRUE→1 deterministic, the u
/// state (EMPTY)→0.5 fair coin. Same probabilities as the IDS qsr_coin/pln_coin;
/// the controller draws the coin with its OWN per-timestep counter PRNG
/// (dist_uniform) so the coin is a pure function of (seed, step, motor, level)
/// and stays bit-mirrored CPU↔GPU. For deterministic modes this equals
/// cell_weight (so is_stochastic gating is the only branch a caller needs).
#[allow(dead_code)] // staged for the QSR/PLN stochastic decode wiring (decode_outputs)
#[inline]
pub fn cell_coin_prob(cell: u8, mode: u8) -> f32 {
	match mode {
		QSR => QUAD_WEIGHTS[(cell & 0x3) as usize],
		PLN => match cell {
			FALSE_U8 => 0.0,
			TRUE_U8 => 1.0,
			_ => 0.5,
		},
		_ => cell_weight(cell, mode),
	}
}

/// Whether a mode's READ is stochastic (a per-timestep seeded coin) vs a
/// deterministic decode. Only QSR (stochastic QUAD) and PLN (stochastic TERNARY).
#[allow(dead_code)] // staged for the QSR/PLN stochastic decode wiring (decode_outputs)
#[inline]
pub fn is_stochastic(mode: u8) -> bool {
	mode == QSR || mode == PLN
}

/// The CONTROLLER's canonical default cell — the value a `SparseLayerMemory`
/// should DELETE-on-write rather than store (sparse hygiene). It is the cell
/// value whose read is indistinguishable from an unwritten address for THIS
/// substrate: QUAD/TERNARY → EMPTY(2) (read_cell's own default, so deletion is
/// transparent to decode, the nudge lattice, and the fire-bit); BINARY → FALSE(0)
/// (the last-write-wins negative — decodes to 0, same as an unwritten cell via
/// the antagonist sum). NOTE: this differs from `ram_core::default_cell_for_mode`
/// (IDS QUAD default = WEAK_FALSE(1)); the controller's QUAD default is EMPTY(2).
#[inline]
pub fn canonical_default_cell(mode: u8) -> u8 {
	match mode {
		BINARY => FALSE_U8,   // 0 — delete the negative writes
		_ => EMPTY_U8,             // 2 — QUAD (WEAK_TRUE/0.75) & TERNARY (0.5)
	}
}

/// The 1-bit recurrent-state feedback ("fired or not") for a state cell.
/// QUAD: the QSR MSB ((v>>1)&1 — the side). TERNARY/BINARY: cell == TRUE(1);
/// unwritten (reads EMPTY=2) → not fired. The QUAD MSB rule would INVERT
/// ternary semantics (TRUE=1 has MSB 0; EMPTY=2 has MSB 1).
#[inline]
pub fn cell_fire_bit(cell: u8, mode: u8) -> bool {
	if is_quad(mode) { (cell >> 1) & 1 != 0 } else { cell == TRUE_U8 }
}

/// The mode-native "fully TRUE" / "fully FALSE" cell values (training targets).
#[inline]
pub fn true_cell(mode: u8) -> u8 {
	if is_quad(mode) { 3 } else { TRUE_U8 }
}
#[inline]
pub fn false_cell(mode: u8) -> u8 {
	let _ = mode;
	FALSE_U8
}

/// Split-trainer planted-cell encoding (Type-1 latches, Type-2/3 counters).
/// QUAD keeps the historical lattice: strong TRUE(3) on / soft WEAK_FALSE(1)
/// off, so one later nudge can flip a planted off but can't erase a latch.
/// TERNARY/BINARY have no soft states and train last-write-wins (any write
/// fully overrides), so plants set TRUE/FALSE directly — the strong/weak
/// asymmetry has no analog to preserve.
#[inline]
pub fn plant_cell(on: bool, mode: u8) -> u8 {
	if is_quad(mode) {
		if on { 3 } else { 1 }
	} else if on {
		TRUE_U8
	} else {
		FALSE_U8
	}
}

/// One training write toward a boolean target. QUAD: one nudge step through
/// the 4-state lattice (the historical rule). TERNARY/BINARY: last-write-wins
/// direct set (see module docs for why this differs from the IDS worker).
#[inline]
pub fn nudge_cell(current: u8, target_true: bool, mode: u8) -> u8 {
	if is_quad(mode) {
		crate::controller_training::nudge_toward_pub(current, target_true)
	} else if target_true {
		TRUE_U8
	} else {
		FALSE_U8
	}
}

/// One training write toward a SPECIFIC mode-native cell value.
#[inline]
pub fn nudge_cell_value(current: u8, target: u8, mode: u8) -> u8 {
	if is_quad(mode) {
		crate::controller_training::nudge_toward_value(current, target)
	} else {
		target
	}
}

/// Solver cost: how "far" a cell is from satisfying a boolean target
/// (the QSR graded nudge-distance, generalized). QUAD: 0..3 lattice steps.
/// TERNARY/BINARY: 0 = already the target, 1 = untrained (EMPTY read),
/// 2 = explicit opposite (a flip erases learned evidence — costlier).
#[inline]
pub fn nudge_distance(cell: u8, target_true: bool, mode: u8) -> u8 {
	if is_quad(mode) {
		let c = (cell & 0x3) as i8;
		if target_true { (3 - c) as u8 } else { c as u8 }
	} else {
		let want = if target_true { TRUE_U8 } else { FALSE_U8 };
		if cell == want { 0 } else if cell == EMPTY_U8 { 1 } else { 2 }
	}
}

/// Mode-native pseudo-random cell for the fixed state reservoir.
/// QUAD: the historical 2-bit draw (bit-identical). TERNARY/BINARY: 1-bit
/// fire/not-fire draw (EMPTY would be a no-op reservoir entry; 3 is invalid).
#[inline]
pub fn reservoir_cell(hash: u64, mode: u8) -> u8 {
	if is_quad(mode) { (hash & 0x3) as u8 } else { (hash & 0x1) as u8 }
}

/// BINARY antagonist thermometer target for output neuron level `level_idx`
/// (0..levels) of a motor whose desired RAW decode is `d_target` ∈ [0,1].
/// Inverse of `decoded = 0.5 + (ΣE − ΣI)/levels`: net = d_target − 0.5;
/// net>0 lights the first net·levels excitatory cells (levels 0..L/2),
/// net<0 the first −net·levels inhibitory cells (levels L/2..L). At the
/// neutral target everything is FALSE → untrained ≡ trained-hover ≡ 0.5.
/// Truncation matches the QUAD thermometer convention ((p·L) as usize > i).
#[inline]
pub fn binary_antagonist_target(d_target: f32, level_idx: usize, levels: usize) -> bool {
	let half = levels / 2;
	let net = d_target - 0.5;
	if level_idx < half {
		net > 0.0 && (net * levels as f32) as usize > level_idx
	} else {
		net < 0.0 && ((-net) * levels as f32) as usize > (level_idx - half)
	}
}

/// Boolean thermometer target for output neuron level `level_idx`, all modes.
/// QUAD/TERNARY: the classic cumulative thermometer. BINARY: antagonist pairs.
#[inline]
pub fn output_target_bit(d_target: f32, level_idx: usize, levels: usize, mode: u8) -> bool {
	if mode == BINARY {
		binary_antagonist_target(d_target, level_idx, levels)
	} else {
		(d_target * levels as f32) as usize > level_idx
	}
}

/// Decode one motor's output cells → raw decode value ∈ [0,1].
/// QUAD/TERNARY: mean cell weight. BINARY: antagonist 0.5 + (ΣE−ΣI)/levels.
#[inline]
pub fn decode_motor_cells(cells: &[u8], mode: u8) -> f32 {
	let levels = cells.len();
	if mode == BINARY {
		let half = levels / 2;
		let sum_e: f32 = cells[..half].iter().map(|&c| cell_weight(c, mode)).sum();
		let sum_i: f32 = cells[half..].iter().map(|&c| cell_weight(c, mode)).sum();
		(0.5 + (sum_e - sum_i) / levels as f32).clamp(0.0, 1.0)
	} else {
		let sum: f32 = cells.iter().map(|&c| cell_weight(c, mode)).sum();
		(sum / levels as f32).clamp(0.0, 1.0)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn neutral_per_mode() {
		assert_eq!(neutral_decode(QUAD_WEIGHTED), 0.75);
		assert_eq!(neutral_decode(QUAD_BINARY), 0.75);
		assert_eq!(neutral_decode(TERNARY), 0.5);
		assert_eq!(neutral_decode(BINARY), 0.5);
	}

	#[test]
	fn fire_bit_semantics() {
		// QUAD MSB rule unchanged.
		for (c, want) in [(0u8, false), (1, false), (2, true), (3, true)] {
			assert_eq!(cell_fire_bit(c, QUAD_WEIGHTED), want);
		}
		// TERNARY: TRUE fires; EMPTY (untrained) and FALSE do not — the QUAD
		// MSB rule would invert this.
		assert!(cell_fire_bit(TRUE_U8, TERNARY));
		assert!(!cell_fire_bit(FALSE_U8, TERNARY));
		assert!(!cell_fire_bit(EMPTY_U8, TERNARY));
		// BINARY: unwritten reads EMPTY → not fired.
		assert!(cell_fire_bit(TRUE_U8, BINARY));
		assert!(!cell_fire_bit(EMPTY_U8, BINARY));
	}

	#[test]
	fn untrained_decodes_exactly_neutral_all_modes() {
		// The ABI-11 invariant, per mode: a bank of unwritten cells (sparse
		// read = EMPTY) decodes exactly to neutral_decode(mode).
		for mode in [TERNARY, QUAD_WEIGHTED, BINARY] {
			let cells = vec![EMPTY_U8; 256];
			assert_eq!(decode_motor_cells(&cells, mode), neutral_decode(mode),
				"mode {mode}: untrained decode must equal neutral");
		}
	}

	#[test]
	fn binary_antagonist_roundtrip() {
		// Train-target → decode inverse: for a grid of raw targets, lighting
		// exactly the antagonist target bits must decode back to ~the target.
		let levels = 256usize;
		for k in 0..=20 {
			let d = k as f32 / 20.0;
			let cells: Vec<u8> = (0..levels)
				.map(|i| if binary_antagonist_target(d, i, levels) { TRUE_U8 } else { FALSE_U8 })
				.collect();
			let decoded = decode_motor_cells(&cells, BINARY);
			assert!((decoded - d).abs() <= 1.0 / levels as f32 + 1e-6,
				"target {d} decoded {decoded}");
		}
	}

	#[test]
	fn binary_neutral_target_writes_nothing_on() {
		for i in 0..256 {
			assert!(!binary_antagonist_target(0.5, i, 256));
		}
	}

	#[test]
	fn ternary_binary_nudge_is_direct_set() {
		for mode in [TERNARY, BINARY] {
			assert_eq!(nudge_cell(EMPTY_U8, true, mode), TRUE_U8);
			assert_eq!(nudge_cell(EMPTY_U8, false, mode), FALSE_U8);
			assert_eq!(nudge_cell(TRUE_U8, false, mode), FALSE_U8); // correctable
			assert_eq!(nudge_cell(FALSE_U8, true, mode), TRUE_U8);
		}
		// QUAD unchanged: one step at a time.
		assert_eq!(nudge_cell(1, true, QUAD_WEIGHTED), 2);
	}
}
