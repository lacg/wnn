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

// ---------------------------------------------------------------------------
// OUTPUT DECODE TOPOLOGY — orthogonal to the cell format (03/08/2026)
// ---------------------------------------------------------------------------
//
// Until now "antagonist E/I" was welded to BINARY, because BINARY *needs* it: a
// 1-bit cell reads 0 when untrained, so one thermometer bank could only ever push
// UP from the floor. The E/I split gives it a two-sided neutral.
//
// But that welded two independent things together, and it left a confound in the
// granularity ablation: BINARY beats QUAD in 4/4 dfa1l cells, and BINARY differs
// from QUAD in THREE ways at once — 1-bit vs 2-bit cells, antagonist vs cumulative
// decode, and a 0.5 vs 0.75 neutral. The ablation is labelled "granularity" but
// cannot currently distinguish which of the three is doing the work.
//
// QUAD does not NEED E/I, but that is not the same as not benefiting from it.
// QUAD's neutral is QUAD_WEIGHTS[EMPTY] = 0.75, which is not the middle: a cell can
// travel 0.75 downward (to 0.0) but only 0.25 upward (to 1.0) — a 3:1 asymmetry in
// control authority around hover, on a substrate whose whole job is symmetric
// push-pull. Under the antagonist decode an untrained QUAD bank has
// ΣE = ΣI = (L/2)·0.75, so ΣE−ΣI = 0 and the effective neutral is EXACTLY 0.5 with
// symmetric authority in both directions. The decode formula needs no change: it is
// already a weight sum, and its range stays [0,1] for any weight set in [0,1].
//
// So topology becomes a separate axis. Defaults preserve every existing result
// bit-for-bit (BINARY→antagonist, everything else→cumulative); `--output-decode`
// is what lets QUAD opt in and de-confound the ablation.

/// Cumulative thermometer: decoded = mean cell weight. The historical QUAD/TERNARY path.
pub const DECODE_CUMULATIVE: u8 = 0;
/// Antagonist E/I halves: decoded = 0.5 + (ΣE − ΣI)/levels. The historical BINARY path.
pub const DECODE_ANTAGONIST: u8 = 1;

/// The topology a mode uses when the caller does not override it. BINARY MUST be
/// antagonist (a single 1-bit bank cannot represent a two-sided neutral at all);
/// every other mode keeps the cumulative thermometer it has always used, so every
/// cohort measured before 03/08/2026 reproduces exactly.
#[inline]
pub fn default_output_decode(mode: u8) -> u8 {
	if mode == BINARY { DECODE_ANTAGONIST } else { DECODE_CUMULATIVE }
}

/// Validate an output-decode topology, and refuse the one combination that is
/// not merely unusual but incoherent: BINARY cannot use the cumulative decode,
/// because an untrained 1-bit bank reads all-FALSE → decoded 0.0, i.e. the neutral
/// would sit on the decode floor and hover becomes unlearnable.
pub fn validate_output_decode(decode: u8, mode: u8) -> Result<(), String> {
	match decode {
		DECODE_CUMULATIVE if mode == BINARY => Err(
			"output_decode=cumulative is incoherent with memory_mode=BINARY: an untrained \
			 1-bit bank decodes to 0.0, putting the neutral on the floor. BINARY requires \
			 the antagonist E/I decode.".to_string()),
		DECODE_CUMULATIVE | DECODE_ANTAGONIST => Ok(()),
		_ => Err(format!("unknown output_decode {decode} (CUMULATIVE=0, ANTAGONIST=1)")),
	}
}

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
///
/// Under the ANTAGONIST topology this is 0.5 for EVERY mode, and that is the
/// point: the two halves start at the same weight whatever that weight is, so
/// they cancel. `neutral_decode_for` is the topology-aware entry point; this
/// mode-only form is the cumulative answer and is kept for the callers that
/// legitimately have no topology in hand.
#[inline]
pub fn neutral_decode_for(mode: u8, decode: u8) -> f32 {
	if decode == DECODE_ANTAGONIST { 0.5 } else { neutral_decode(mode) }
}

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

/// Antagonist thermometer target for output neuron level `level_idx`
/// (0..levels) of a motor whose desired RAW decode is `d_target` ∈ [0,1].
/// Inverse of `decoded = 0.5 + (ΣE − ΣI)/levels`: net = d_target − 0.5;
/// net>0 lights the first net·levels excitatory cells (levels 0..L/2),
/// net<0 the first −net·levels inhibitory cells (levels L/2..L). At the
/// neutral target everything is FALSE → untrained ≡ trained-hover ≡ 0.5.
/// Truncation matches the QUAD thermometer convention ((p·L) as usize > i).
#[inline]
pub fn antagonist_target(d_target: f32, level_idx: usize, levels: usize) -> bool {
	let half = levels / 2;
	let net = d_target - 0.5;
	if level_idx < half {
		net > 0.0 && (net * levels as f32) as usize > level_idx
	} else {
		net < 0.0 && ((-net) * levels as f32) as usize > (level_idx - half)
	}
}

/// Boolean thermometer target for output neuron level `level_idx`.
/// CUMULATIVE: the classic thermometer. ANTAGONIST: E/I pairs.
///
/// `mode` is no longer consulted — topology alone decides the target layout, which
/// is exactly the separation this refactor buys. Callers that want the historical
/// behaviour pass `default_output_decode(mode)`.
#[inline]
pub fn output_target_bit(d_target: f32, level_idx: usize, levels: usize, decode: u8) -> bool {
	if decode == DECODE_ANTAGONIST {
		antagonist_target(d_target, level_idx, levels)
	} else {
		(d_target * levels as f32) as usize > level_idx
	}
}

/// TARGET-LEVELS redundancy (16/08/2026, Luiz's decoupling). The thermometer
/// target couples population size to threshold resolution: N neurons per motor
/// means N thresholds spaced 1/N apart, so growing the population for its
/// error-averaging benefit also demands finer per-neuron discrimination than a
/// narrow address function can learn (arm B's mono_viol 24-28 at b15). This map
/// breaks the coupling on the TRAINING side only: N neurons share T (<N)
/// distinct thresholds, R = N/T contiguous neurons per threshold. The sum
/// decode is untouched — it already aggregates the redundant group, so its
/// members' independent errors average out while each threshold stays as
/// learnable as a T-level code.
///
/// Returns (virtual_idx, effective_levels) to feed output_target_bit.
/// target_levels == 0 or >= levels ⇒ identity (legacy, bit-for-bit).
/// ANTAGONIST maps E-half onto the first T/2 and I-half onto the last T/2, so
/// the sign structure of the code is preserved.
///
/// The map is PROPORTIONAL (idx*T/N per half), not stride-based: the NEURONS
/// stage mutates per-genome output counts to arbitrary values (on=72 ⇒ 18
/// levels/motor in banked runs), so requiring levels % T == 0 would kill
/// mid-search offspring. When T divides N this reduces exactly to R=N/T
/// contiguous neurons per threshold; otherwise group sizes differ by ±1.
/// T even under ANTAGONIST is validated at set_target_levels, not here.
#[inline]
pub fn map_target_level(level_idx: usize, levels: usize, target_levels: usize, decode: u8) -> (usize, usize) {
	if target_levels == 0 || target_levels >= levels {
		return (level_idx, levels);
	}
	if decode == DECODE_ANTAGONIST {
		let half = levels / 2;
		let t_half = target_levels / 2;
		if level_idx < half {
			(level_idx * t_half / half, target_levels)
		} else {
			(t_half + (level_idx - half) * t_half / half, target_levels)
		}
	} else {
		(level_idx * target_levels / levels, target_levels)
	}
}

/// Decode one motor's output cells → raw decode value ∈ [0,1].
/// CUMULATIVE: mean cell weight. ANTAGONIST: 0.5 + (ΣE−ΣI)/levels.
///
/// The antagonist branch is weight-generic, not BINARY-specific: it sums
/// cell_weight() over each half, so it works for any weight set in [0,1]. For QUAD
/// an untrained bank gives ΣE = ΣI = (L/2)·0.75 → decoded exactly 0.5, and the
/// extremes still reach 0.0/1.0, so the range is unchanged.
#[inline]
pub fn decode_motor_cells(cells: &[u8], mode: u8, decode: u8) -> f32 {
	let levels = cells.len();
	if decode == DECODE_ANTAGONIST {
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

	#[test]
	fn map_target_level_semantics() {
		use super::{map_target_level, output_target_bit, DECODE_ANTAGONIST, DECODE_CUMULATIVE};
		// Identity: 0 and >= levels are bit-for-bit legacy.
		for idx in 0..8 {
			assert_eq!(map_target_level(idx, 8, 0, DECODE_ANTAGONIST), (idx, 8));
			assert_eq!(map_target_level(idx, 8, 8, DECODE_ANTAGONIST), (idx, 8));
			assert_eq!(map_target_level(idx, 8, 16, DECODE_CUMULATIVE), (idx, 8));
		}
		// CUMULATIVE N=8 T=4: contiguous pairs share a threshold, order kept.
		let virt: Vec<usize> = (0..8).map(|i| map_target_level(i, 8, 4, DECODE_CUMULATIVE).0).collect();
		assert_eq!(virt, vec![0, 0, 1, 1, 2, 2, 3, 3]);
		// ANTAGONIST N=8 T=4: E-half maps onto 0..2, I-half onto 2..4 — the sign
		// structure survives (an E neuron never lands on an I threshold).
		let virt_a: Vec<usize> = (0..8).map(|i| map_target_level(i, 8, 4, DECODE_ANTAGONIST).0).collect();
		assert_eq!(virt_a, vec![0, 0, 1, 1, 2, 2, 3, 3]);
		assert!(virt_a[..4].iter().all(|&v| v < 2) && virt_a[4..].iter().all(|&v| v >= 2));
		// NON-DIVISIBLE levels (GA offspring have arbitrary counts, e.g. on=72
		// => 18 levels/motor): proportional map stays total, ordered, and
		// half-preserving — never out of range, never an error.
		for idx in 0..18 {
			let (v, l) = map_target_level(idx, 18, 4, DECODE_ANTAGONIST);
			assert_eq!(l, 4);
			assert!(v < 4, "virt {v} out of range");
			if idx < 9 { assert!(v < 2); } else { assert!(v >= 2); }
		}
		let vs: Vec<usize> = (0..9).map(|i| map_target_level(i, 18, 4, DECODE_ANTAGONIST).0).collect();
		assert!(vs.windows(2).all(|w| w[0] <= w[1]), "E-half order broken: {vs:?}");
		// Redundancy group members get IDENTICAL targets across the pwm range,
		// and the coarse code stays monotone within each half.
		for d10 in 0..=10 {
			let d = d10 as f32 / 10.0;
			for g in 0..4 {
				let (v0, l0) = map_target_level(2 * g, 8, 4, DECODE_ANTAGONIST);
				let (v1, l1) = map_target_level(2 * g + 1, 8, 4, DECODE_ANTAGONIST);
				assert_eq!(
					output_target_bit(d, v0, l0, DECODE_ANTAGONIST),
					output_target_bit(d, v1, l1, DECODE_ANTAGONIST),
					"group {g} split at d={d}");
			}
		}
	}

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
			assert_eq!(decode_motor_cells(&cells, mode, default_output_decode(mode)),
				neutral_decode(mode), "mode {mode}: untrained decode must equal neutral");
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
				.map(|i| if antagonist_target(d, i, levels) { TRUE_U8 } else { FALSE_U8 })
				.collect();
			let decoded = decode_motor_cells(&cells, BINARY, DECODE_ANTAGONIST);
			assert!((decoded - d).abs() <= 1.0 / levels as f32 + 1e-6,
				"target {d} decoded {decoded}");
		}
	}

	#[test]
	fn binary_neutral_target_writes_nothing_on() {
		for i in 0..256 {
			assert!(!antagonist_target(0.5, i, 256));
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

	// ---- output-decode topology (03/08/2026) -------------------------------

	#[test]
	fn defaults_reproduce_every_pre_existing_cohort() {
		// The whole safety argument: with default topology, nothing moves.
		assert_eq!(default_output_decode(BINARY), DECODE_ANTAGONIST);
		for mode in [TERNARY, QUAD_WEIGHTED, QUAD_BINARY, QSR, PLN] {
			assert_eq!(default_output_decode(mode), DECODE_CUMULATIVE, "mode {mode}");
		}
		// ...and the default path decodes identically to the old mode-only branch.
		for mode in [TERNARY, QUAD_WEIGHTED, BINARY] {
			for cells in [vec![EMPTY_U8; 64], vec![TRUE_U8; 64], vec![FALSE_U8; 64]] {
				let got = decode_motor_cells(&cells, mode, default_output_decode(mode));
				let want = if mode == BINARY {
					let h = cells.len() / 2;
					let e: f32 = cells[..h].iter().map(|&c| cell_weight(c, mode)).sum();
					let i: f32 = cells[h..].iter().map(|&c| cell_weight(c, mode)).sum();
					(0.5 + (e - i) / cells.len() as f32).clamp(0.0, 1.0)
				} else {
					let s: f32 = cells.iter().map(|&c| cell_weight(c, mode)).sum();
					(s / cells.len() as f32).clamp(0.0, 1.0)
				};
				assert_eq!(got, want, "mode {mode} default decode must match the old branch");
			}
		}
	}

	#[test]
	fn quad_cumulative_neutral_is_asymmetric_and_antagonist_fixes_it() {
		// THE motivation, pinned as a test. Cumulative QUAD sits at 0.75, so it can
		// travel 0.75 down but only 0.25 up — 3:1 authority around hover.
		let n = neutral_decode(QUAD_WEIGHTED);
		assert_eq!(n, 0.75);
		let (down, up) = (n - 0.0, 1.0 - n);
		assert!((down / up - 3.0).abs() < 1e-6, "expected a 3:1 asymmetry, got {down}:{up}");

		// Under the antagonist decode an untrained QUAD bank cancels to EXACTLY 0.5,
		// with symmetric authority — and the range still spans the full [0,1].
		let levels = 64usize;
		let untrained = vec![EMPTY_U8; levels];
		assert_eq!(decode_motor_cells(&untrained, QUAD_WEIGHTED, DECODE_ANTAGONIST), 0.5);
		assert_eq!(neutral_decode_for(QUAD_WEIGHTED, DECODE_ANTAGONIST), 0.5);

		// true_cell/false_cell, NOT the raw TRUE_U8/FALSE_U8: under QUAD the raw
		// TERNARY constants index QUAD_WEIGHTS[1]=0.25 and [0]=0.0, so hardcoding
		// them measures a 0.25-wide substrate. This is the exact trap CLAUDE.md
		// records as the inverted-QUAD multistage bug.
		let (on, off) = (true_cell(QUAD_WEIGHTED), false_cell(QUAD_WEIGHTED));
		let mut full_up = vec![off; levels];
		for c in full_up[..levels / 2].iter_mut() { *c = on; }
		assert_eq!(decode_motor_cells(&full_up, QUAD_WEIGHTED, DECODE_ANTAGONIST), 1.0);
		let mut full_down = vec![off; levels];
		for c in full_down[levels / 2..].iter_mut() { *c = on; }
		assert_eq!(decode_motor_cells(&full_down, QUAD_WEIGHTED, DECODE_ANTAGONIST), 0.0);
	}

	#[test]
	fn quad_antagonist_roundtrips_like_binary_does() {
		// The inverse must hold for QUAD too, or the trainer cannot hit its target.
		let levels = 256usize;
		for k in 0..=20 {
			let d = k as f32 / 20.0;
			let cells: Vec<u8> = (0..levels)
				.map(|i| if antagonist_target(d, i, levels) {
					true_cell(QUAD_WEIGHTED) } else { false_cell(QUAD_WEIGHTED) })
				.collect();
			let decoded = decode_motor_cells(&cells, QUAD_WEIGHTED, DECODE_ANTAGONIST);
			assert!((decoded - d).abs() <= 1.0 / levels as f32 + 1e-6,
				"QUAD antagonist target {d} decoded {decoded}");
		}
	}

	#[test]
	fn target_layout_depends_on_topology_not_mode() {
		// Same topology => same bits, whatever the cell format. This is the
		// separation the refactor exists to create.
		let levels = 64usize;
		for i in 0..levels {
			for d in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
				assert_eq!(output_target_bit(d, i, levels, DECODE_ANTAGONIST),
				           antagonist_target(d, i, levels));
				assert_eq!(output_target_bit(d, i, levels, DECODE_CUMULATIVE),
				           (d * levels as f32) as usize > i);
			}
		}
	}

	#[test]
	fn binary_may_not_use_the_cumulative_decode() {
		// Not merely unusual — incoherent: an untrained 1-bit bank reads all-FALSE,
		// so the neutral would land on the decode floor and hover is unlearnable.
		assert!(validate_output_decode(DECODE_CUMULATIVE, BINARY).is_err());
		assert!(validate_output_decode(DECODE_ANTAGONIST, BINARY).is_ok());
		assert!(validate_output_decode(DECODE_CUMULATIVE, QUAD_WEIGHTED).is_ok());
		assert!(validate_output_decode(DECODE_ANTAGONIST, QUAD_WEIGHTED).is_ok());
		assert!(validate_output_decode(7, QUAD_WEIGHTED).is_err());
	}
}
