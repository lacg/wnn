//! `CellMode` — the public name for the memory-mode ladder.
//!
//! The u8 constants in `neuron_memory.rs` (TERNARY=0 … PLN=5) are the wire
//! codes every kernel, marker and checkpoint already carries; this enum is a
//! typed view over EXACTLY those codes (`#[repr(u8)]`, discriminant == code), so
//! nothing internal moves and `as_u8()`/`from_u8()` are free. Public docs and
//! the sklearn estimator name modes through here; hot paths keep the u8.
//!
//! The ladder, by information per cell:
//!
//! | mode           | code | states           | read                                   |
//! |----------------|------|------------------|----------------------------------------|
//! | `Binary`       | 3    | FALSE / TRUE     | TRUE → 1, else 0 (classical WiSARD)    |
//! | `Ternary`      | 0    | FALSE / u / TRUE | u → `empty_value`                      |
//! | `Pln`          | 5    | FALSE / u / TRUE | u fires a fair coin                    |
//! | `QuadBinary`   | 1    | 4-state nudging  | cell ≥ WEAK_TRUE → 1, else 0           |
//! | `QuadWeighted` | 2    | 4-state nudging  | graded 0 / .25 / .75 / 1               |
//! | `Qsr`          | 4    | 4-state nudging  | fires a coin with p = graded weight    |
//!
//! Every table below is the single source of truth for "which cell is the
//! untrained/miss default in mode X" — `metal_sparse::default_cell_for_mode`
//! (both platforms) delegates here. Before this file the non-macOS stub had its
//! own two-arm copy that returned WEAK_FALSE for BINARY and PLN (wrong for both).
//!
//! The per-cell WRITE primitives (lattice step, fire bit, plants, nudge
//! distance, coin probability) live in `cell_write.rs` as a second `impl
//! CellMode` block — shared by the IDS worker and the drone controller.

use crate::neuron_memory::{
	BINARY, EMPTY_U8, FALSE_U8, PLN, QSR, QUAD_BINARY, QUAD_WEAK_FALSE, QUAD_WEIGHTED, TERNARY,
};
pub use crate::unwritten_lean::UnwrittenLean;
use crate::vote_share::VoteShare;

/// How one training visit writes a cell. THE dispatch every IDS trainer keys
/// on — the CPU slot trainer (`train_genome_in_slot_range`), the GPU marker
/// kernel (`wnn_write_rule` in common.metal, its twin) and the LM gating
/// trainer — so a mode can never again reach a writer meant for another cell
/// encoding (PLN fell through `!= TERNARY` to the QUAD nudge until 12/10/2026:
/// a FALSE nudge on EMPTY(2) left 1 = PLN TRUE, i.e. inverted training).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteRule
{
	/// BINARY: own-class one-shot set TRUE; FALSE-direction visits write
	/// nothing. Idempotent, so order-independent.
	Set,
	/// TERNARY / PLN: the TRUE-wins lattice join EMPTY < FALSE < TRUE (TRUE
	/// absorbs, FALSE lands only on EMPTY), on the ternary codes FALSE=0,
	/// TRUE=1, EMPTY=2. Idempotent + commutative, so order-independent.
	Lattice,
	/// QUAD_WEIGHTED / QUAD_BINARY / QSR: the 4-state nudge (OI packed
	/// counters when active, else the order-dependent clamped nudge).
	Nudge,
	/// MAJORITY VOTE (opt-in per flow, `ids_write_rule="vote"`; BINARY /
	/// TERNARY / PLN only): every visit adds ±w to a signed per-(neuron,
	/// address) tally (+w own-class / attack, −w otherwise; w = the example's
	/// class weight), held in the OI packed counter's `net` field (the same
	/// buffers, merges and atomics as the QUAD OI path). Commit: tally > 0 →
	/// TRUE, < 0 → FALSE, 0 → the mode's untrained cell (TERNARY/PLN EMPTY,
	/// BINARY FALSE). Integer sums commute, so order-independent. Same
	/// semantics as `train::SparseTrainer`'s TERNARY votes. Fixes the
	/// TRUE-wins floor: one attack on an address no longer outvotes any
	/// number of normals.
	///
	/// IDS-56 MIN-SHARE (`ids_vote_share` θ, worker ABI 28): the payload sets
	/// the per-visit weights — +own·w on a TRUE-direction visit, −other·w on
	/// any other, own:other = (1−θ):θ exact on a 1/100 grid (`VoteShare`).
	/// `VoteShare::MAJORITY` (θ = 0.5, 1:1) is the plain vote above,
	/// bit-identical. The binning is the same sign rule for every θ.
	Vote(VoteShare),
	/// The drone CONTROLLER's DAgger direct write for BINARY / TERNARY / PLN
	/// (ram_controller cell_mode::nudge_cell, unchanged): every visit SETS the
	/// cell to its target (TRUE or FALSE), so the last visit wins. Deliberately
	/// NOT the IDS writers: the DAgger loop must be able to correct an earlier
	/// label as the teacher distribution shifts (ram_controller cell_mode.rs
	/// module doc). Order-DEPENDENT. Named here (CTRL-46) so the controller's
	/// write rules live in the same vocabulary as the IDS ones; it is produced
	/// only by `resolve_controller` and has no IDS kernel arm (`code` 4 is never
	/// sent to an IDS trainer).
	LastWrite,
}

/// How a counter-accumulated training pass bins each packed counter into a
/// 2-bit cell at commit. The counters (`neuron_memory::OI_*`) are shared by
/// the QUAD OI nudge and the vote rule; only the binning differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterBin
{
	/// QUAD OI: `oi_bin_to_cell` (obs-aware 4-state binning).
	Oi,
	/// `WriteRule::Vote`: sign of the net tally; a tie bins to `tie` (the
	/// mode's untrained cell, so a tied cell is indistinguishable from an
	/// unvisited one).
	Vote
	{
		tie: u8,
	},
}

impl CounterBin
{
	/// The committed cell for one (possibly merged) packed counter.
	#[inline]
	pub fn cell(self, packed: u32) -> u8
	{
		match self
		{
			CounterBin::Oi => crate::neuron_memory::oi_bin_to_cell(packed) as u8,
			CounterBin::Vote { tie } => crate::neuron_memory::vote_bin_to_cell(packed, tie),
		}
	}

	/// The counter buffer a training pass under (`rule`, `memory_mode`) needs:
	/// `Vote` always (its tally IS the counter), `Nudge` only when the OI
	/// trainer is active (`order_independent_training_active`), `None` for the
	/// direct writers (Set / Lattice / legacy clamped nudge). THE gate every
	/// trainer uses for init/accumulate/commit.
	pub fn for_training(rule: WriteRule, memory_mode: u8) -> Option<CounterBin>
	{
		match rule
		{
			// Vote never reaches a 4-state mode, so the lean is irrelevant here.
			WriteRule::Vote(_) => Some(CounterBin::Vote {
				tie: CellMode::from_u8(memory_mode).map_or(EMPTY_U8, |m| m.default_cell(UnwrittenLean::False)),
			}),
			WriteRule::Nudge if crate::neuron_memory::order_independent_training_active(memory_mode) =>
			{
				Some(CounterBin::Oi)
			}
			_ => None,
		}
	}
}

impl WriteRule
{
	/// The rule for a wire code. An unknown code keeps the historical
	/// fall-through (the nudge) — every entry point rejects unknown codes.
	pub fn of_code(code: u8) -> WriteRule
	{
		CellMode::from_u8(code).map_or(WriteRule::Nudge, |m| m.write_rule())
	}

	/// The rule a trainer applies for `memory_mode` under the per-flow
	/// selector: `vote == None` is EXACTLY `of_code` (bit-identical default);
	/// `vote == Some(share)` swaps the direct writers (BINARY Set, TERNARY/PLN
	/// Lattice) for `Vote(share)` (`VoteShare::MAJORITY` = the plain vote). The 4-state family keeps its nudge either way
	/// (its OI counters already tally votes; the Python facade refuses
	/// `ids_write_rule="vote"` on those modes so it never silently no-ops).
	/// GPU twin: common.metal `wnn_write_rule(memory_mode, vote)`.
	pub fn resolve(memory_mode: u8, vote: Option<VoteShare>) -> WriteRule
	{
		match (WriteRule::of_code(memory_mode), vote)
		{
			(WriteRule::Set | WriteRule::Lattice, Some(share)) => WriteRule::Vote(share),
			(rule, _) => rule,
		}
	}

	/// The rule the CONTROLLER's DAgger walk applies for `memory_mode` (CTRL-46,
	/// `--write-rule {last,vote}` + `--vote-share θ`): the 4-state family keeps its
	/// nudge (`Nudge` — its lattice already settles disagreement by tally, so a
	/// vote request is REFUSED rather than silently ignored); BINARY / TERNARY /
	/// PLN get `LastWrite` (vote None, today's writer) or `Vote(share)`. Unknown
	/// modes are refused.
	pub fn resolve_controller(memory_mode: u8, vote: Option<VoteShare>) -> Result<WriteRule, String>
	{
		let mode = CellMode::from_u8(memory_mode).ok_or_else(|| format!("unknown memory_mode {memory_mode}"))?;
		match (mode.write_rule(), vote)
		{
			(WriteRule::Nudge, None) => Ok(WriteRule::Nudge),
			(WriteRule::Nudge, Some(_)) => Err(format!(
				"write rule 'vote' needs memory_mode BINARY, TERNARY or PLN; got {} (the 4-state \
				 modes already settle same-address disagreement through their nudge lattice)",
				mode.name()
			)),
			(_, None) => Ok(WriteRule::LastWrite),
			(_, Some(share)) => Ok(WriteRule::Vote(share)),
		}
	}

	/// True for the vote writer (any share).
	#[inline]
	pub fn is_vote(self) -> bool
	{
		matches!(self, WriteRule::Vote(_))
	}

	/// The min-share weights when this is the vote writer.
	#[inline]
	pub fn vote_share(self) -> Option<VoteShare>
	{
		match self
		{
			WriteRule::Vote(share) => Some(share),
			_ => None,
		}
	}

	/// The GPU wire code (common.metal `WNN_WRITE_*`).
	pub fn code(self) -> u32
	{
		match self
		{
			WriteRule::Set => 0,
			WriteRule::Lattice => 1,
			WriteRule::Nudge => 2,
			WriteRule::Vote(_) => 3,
			WriteRule::LastWrite => 4,
		}
	}

	/// The lattice join of two ternary cells (EMPTY < FALSE < TRUE) — what
	/// the two writes leave on one cell in either order. Duplicate-key GPU
	/// slots of one address merge through this, as OI counters merge through
	/// `oi_merge`. BINARY cells (TRUE only, else FALSE) join the same way.
	#[inline]
	pub fn lattice_join(a: u8, b: u8) -> u8
	{
		use crate::neuron_memory::{FALSE_U8, TRUE_U8};
		if a == TRUE_U8 || b == TRUE_U8
		{
			TRUE_U8
		}
		else if a == FALSE_U8 || b == FALSE_U8
		{
			FALSE_U8
		}
		else
		{
			a
		}
	}
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CellMode
{
	Ternary = TERNARY,
	QuadBinary = QUAD_BINARY,
	QuadWeighted = QUAD_WEIGHTED,
	Binary = BINARY,
	Qsr = QSR,
	Pln = PLN,
}

impl CellMode
{
	/// Every mode, in ladder order (information per cell, ascending).
	pub const ALL: [CellMode; 6] = [
		CellMode::Binary,
		CellMode::Ternary,
		CellMode::Pln,
		CellMode::QuadBinary,
		CellMode::QuadWeighted,
		CellMode::Qsr,
	];

	/// The project default (CLAUDE.md: QUAD_WEIGHTED, never TERNARY for new work).
	pub const DEFAULT: CellMode = CellMode::QuadWeighted;

	#[inline]
	pub fn as_u8(self) -> u8
	{
		self as u8
	}

	/// The wire code → mode. `None` for a code no kernel knows.
	pub fn from_u8(code: u8) -> Option<CellMode>
	{
		match code
		{
			TERNARY => Some(CellMode::Ternary),
			QUAD_BINARY => Some(CellMode::QuadBinary),
			QUAD_WEIGHTED => Some(CellMode::QuadWeighted),
			BINARY => Some(CellMode::Binary),
			QSR => Some(CellMode::Qsr),
			PLN => Some(CellMode::Pln),
			_ => None,
		}
	}

	/// Stable lower-case name (marker fields, CLI flags, Python enum values).
	pub fn name(self) -> &'static str
	{
		match self
		{
			CellMode::Ternary => "ternary",
			CellMode::QuadBinary => "quad_binary",
			CellMode::QuadWeighted => "quad_weighted",
			CellMode::Binary => "binary",
			CellMode::Qsr => "qsr",
			CellMode::Pln => "pln",
		}
	}

	/// The read fires a seeded coin (QSR/PLN). Deterministic otherwise.
	#[inline]
	pub fn is_stochastic(self) -> bool
	{
		matches!(self, CellMode::Qsr | CellMode::Pln)
	}

	/// How a training visit writes this mode's cells (see `WriteRule`).
	/// PLN shares TERNARY's cells AND its writer — it differs only in the
	/// read (a fair coin on the u-state).
	#[inline]
	pub fn write_rule(self) -> WriteRule
	{
		match self
		{
			CellMode::Binary => WriteRule::Set,
			CellMode::Ternary | CellMode::Pln => WriteRule::Lattice,
			CellMode::QuadBinary | CellMode::QuadWeighted | CellMode::Qsr => WriteRule::Nudge,
		}
	}

	/// The mode trains through the 4-state nudging lattice and therefore
	/// takes the ORDER-INDEPENDENT packed (obs, net) counter path
	/// (`oi_apply_nudge` / `oi_bin_to_cell`, `OI_INITIAL` slot default) when
	/// `WNN_ORDER_INDEPENDENT_TRAIN=1`. QUAD_WEIGHTED, QUAD_BINARY and QSR
	/// share the lattice — QSR differs from QUAD_WEIGHTED ONLY in the read
	/// (a coin with p = graded weight), never in the write. BINARY
	/// (`WriteRule::Set`) and TERNARY/PLN (`WriteRule::Lattice`, the
	/// TRUE-wins join) are order-independent by construction and keep their
	/// own rules. Exactly the `WriteRule::Nudge` modes.
	///
	/// THE single source of truth for every OI gate: `SparseTrainer::new`,
	/// `neuron_memory::order_independent_training_active`, the batched Metal
	/// trainer and the CPU slot trainers all ask here. Before 16/09/2026 six
	/// sites compared `memory_mode == QUAD_WEIGHTED` literally, so QSR fell
	/// silently to the legacy clamped-nudge trainer (z=1 chunking once
	/// genomes×neurons >= 320 → ~12x slower) while its claimed slots were
	/// still zeroed to OI_INITIAL by the default-cell chooser.
	#[inline]
	pub fn uses_oi_counters(self) -> bool
	{
		matches!(self, CellMode::QuadWeighted | CellMode::QuadBinary | CellMode::Qsr)
	}

	/// Whether `empty_value` enters the read at all. Only TERNARY reads its
	/// u-state as `empty_value`; PLN fixes the coin at 0.5, QUAD's baseline is
	/// WEAK_FALSE = 0.25 by construction, BINARY has no untrained state.
	#[inline]
	pub fn uses_empty_value(self) -> bool
	{
		matches!(self, CellMode::Ternary)
	}

	/// Number of distinct cell states the mode can hold.
	#[inline]
	pub fn num_states(self) -> u8
	{
		match self
		{
			CellMode::Binary => 2,
			CellMode::Ternary | CellMode::Pln => 3,
			CellMode::QuadBinary | CellMode::QuadWeighted | CellMode::Qsr => 4,
		}
	}

	/// The cell an UNWRITTEN address reads as — the initial state of every cell
	/// and the sparse-miss default. TERNARY/PLN: EMPTY(2); BINARY: FALSE(0)
	/// ("never seen → no vote"); QUAD_*/QSR: the `lean` side of the lattice —
	/// `False` WEAK_FALSE(1) (the IDS worker), `True` WEAK_TRUE(2) = EMPTY_U8
	/// (the drone controller's hover-neutral bank). See `UnwrittenLean`.
	#[inline]
	pub fn default_cell(self, lean: UnwrittenLean) -> u8
	{
		match (self, lean)
		{
			(CellMode::Ternary | CellMode::Pln, _) => EMPTY_U8,
			(CellMode::Binary, _) => FALSE_U8,
			(_, UnwrittenLean::False) => QUAD_WEAK_FALSE as u8,
			(_, UnwrittenLean::True) => EMPTY_U8,
		}
	}

	/// The sparse-miss cell under the coverage-aware rule
	/// (docs/COVERAGE_AWARE_SCORER_SPEC.md): a miss resolves to cell 0, which
	/// weighs 0.0 in EVERY mode, so a class is penalised in proportion to how
	/// much of its memory is empty. `coverage_aware == false` is bit-exact OFF.
	#[inline]
	pub fn miss_cell(self, lean: UnwrittenLean, coverage_aware: bool) -> u8
	{
		if coverage_aware
		{
			0
		}
		else
		{
			self.default_cell(lean)
		}
	}
}

impl std::fmt::Display for CellMode
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		f.write_str(self.name())
	}
}

#[cfg(test)]
mod tests
{
	use super::*;
	use crate::neuron_memory::{build_empty_word, cell_to_weight, empty_word_for_mode};

	#[test]
	fn discriminants_are_the_wire_codes()
	{
		for mode in CellMode::ALL
		{
			assert_eq!(CellMode::from_u8(mode.as_u8()), Some(mode));
		}
		assert_eq!(CellMode::Ternary.as_u8(), TERNARY);
		assert_eq!(CellMode::QuadBinary.as_u8(), QUAD_BINARY);
		assert_eq!(CellMode::QuadWeighted.as_u8(), QUAD_WEIGHTED);
		assert_eq!(CellMode::Binary.as_u8(), BINARY);
		assert_eq!(CellMode::Qsr.as_u8(), QSR);
		assert_eq!(CellMode::Pln.as_u8(), PLN);
		assert_eq!(CellMode::from_u8(6), None);
		assert_eq!(CellMode::ALL.len(), 6);
	}

	/// `default_cell` must agree with the dense empty word (`empty_word_for_mode`)
	/// — the dense and sparse "untrained" states are the same state.
	#[test]
	fn default_cell_matches_the_dense_empty_word()
	{
		for mode in CellMode::ALL
		{
			assert_eq!(
				empty_word_for_mode(mode.as_u8()),
				build_empty_word(mode.default_cell(UnwrittenLean::False) as i64),
				"{mode}: sparse miss default and dense empty word disagree"
			);
		}
	}

	/// The three delegating entry points (macOS metal_sparse, the non-macOS stub,
	/// and this enum) must return the same table — the stub used to disagree.
	#[test]
	fn metal_sparse_default_cell_delegates_here()
	{
		use crate::metal_sparse::{default_cell_for_coverage, default_cell_for_mode};
		for mode in CellMode::ALL
		{
			assert_eq!(
				default_cell_for_mode(mode.as_u8()),
				mode.default_cell(UnwrittenLean::False) as u32,
				"{mode}"
			);
			assert_eq!(
				default_cell_for_coverage(mode.as_u8(), false),
				mode.miss_cell(UnwrittenLean::False, false) as u32
			);
			assert_eq!(default_cell_for_coverage(mode.as_u8(), true), 0);
		}
	}

	/// A default (untrained) cell weighs: TERNARY → empty_value, PLN → 0.5,
	/// QUAD_WEIGHTED/QSR → 0.25, QUAD_BINARY → 0.0, BINARY → 0.0.
	#[test]
	fn default_cell_weights_per_mode()
	{
		let w = |m: CellMode| cell_to_weight(m.default_cell(UnwrittenLean::False) as i64, m.as_u8(), 0.37);
		assert_eq!(w(CellMode::Ternary), 0.37);
		assert_eq!(w(CellMode::Pln), 0.5);
		assert_eq!(w(CellMode::QuadWeighted), 0.25);
		assert_eq!(w(CellMode::Qsr), 0.25);
		assert_eq!(w(CellMode::QuadBinary), 0.0);
		assert_eq!(w(CellMode::Binary), 0.0);
	}

	/// The OI-counter predicate is exactly the 4-state nudging family: QSR
	/// (mode 4) MUST be in it — the 16/09/2026 defect was QSR falling out of
	/// literal `== QUAD_WEIGHTED` gates.
	#[test]
	fn uses_oi_counters_is_the_quad_family()
	{
		assert!(CellMode::QuadWeighted.uses_oi_counters());
		assert!(CellMode::QuadBinary.uses_oi_counters());
		assert!(CellMode::Qsr.uses_oi_counters());
		assert!(!CellMode::Binary.uses_oi_counters());
		assert!(!CellMode::Ternary.uses_oi_counters());
		assert!(!CellMode::Pln.uses_oi_counters());
		// Every OI mode is a 4-state mode and vice versa.
		for mode in CellMode::ALL
		{
			assert_eq!(mode.uses_oi_counters(), mode.num_states() == 4, "{mode}");
		}
		// And every OI mode's untrained cell is WEAK_FALSE (the legacy default
		// the OI slot default OI_INITIAL=0 replaces — see marker_train.rs).
		for mode in CellMode::ALL.into_iter().filter(|m| m.uses_oi_counters())
		{
			assert_eq!(mode.default_cell(UnwrittenLean::False), QUAD_WEAK_FALSE as u8, "{mode}");
		}
	}

	/// PLN writes through TERNARY's lattice (never the QUAD nudge — the
	/// 12/10/2026 inverted-PLN defect), the nudge set is exactly the OI set,
	/// and unknown codes keep the historical nudge fall-through.
	#[test]
	fn write_rule_per_mode()
	{
		assert_eq!(CellMode::Pln.write_rule(), WriteRule::Lattice);
		assert_eq!(CellMode::Ternary.write_rule(), WriteRule::Lattice);
		assert_eq!(CellMode::Binary.write_rule(), WriteRule::Set);
		for mode in CellMode::ALL
		{
			assert_eq!(mode.write_rule() == WriteRule::Nudge, mode.uses_oi_counters(), "{mode}");
			assert_eq!(WriteRule::of_code(mode.as_u8()), mode.write_rule(), "{mode}");
		}
		assert_eq!(WriteRule::of_code(99), WriteRule::Nudge);
	}

	/// The join is commutative, idempotent, and TRUE-wins: it equals the
	/// writer's result for any two writes (EMPTY=2 is the identity).
	#[test]
	fn lattice_join_is_the_writer_join()
	{
		use crate::neuron_memory::TRUE_U8;
		for a in [FALSE_U8, TRUE_U8, EMPTY_U8]
		{
			assert_eq!(WriteRule::lattice_join(a, a), a);
			assert_eq!(WriteRule::lattice_join(a, EMPTY_U8), a);
			assert_eq!(WriteRule::lattice_join(a, TRUE_U8), TRUE_U8);
			for b in [FALSE_U8, TRUE_U8, EMPTY_U8]
			{
				assert_eq!(WriteRule::lattice_join(a, b), WriteRule::lattice_join(b, a));
			}
		}
		assert_eq!(WriteRule::lattice_join(FALSE_U8, EMPTY_U8), FALSE_U8);
	}

	/// The per-flow selector: unset (`vote=false`) is EXACTLY the mode's own
	/// writer (default bit-identity); `vote=true` swaps only the direct
	/// writers; the 4-state family keeps its nudge.
	#[test]
	fn resolve_default_is_of_code_and_vote_swaps_direct_writers()
	{
		for mode in CellMode::ALL
		{
			let code = mode.as_u8();
			assert_eq!(WriteRule::resolve(code, None), WriteRule::of_code(code), "{mode}");
			let share = VoteShare::from_theta(0.1).unwrap();
			for vote in [VoteShare::MAJORITY, share]
			{
				let expected = match mode.write_rule()
				{
					WriteRule::Set | WriteRule::Lattice => WriteRule::Vote(vote),
					other => other,
				};
				assert_eq!(WriteRule::resolve(code, Some(vote)), expected, "{mode}");
				assert_eq!(WriteRule::resolve(code, Some(vote)).vote_share().is_some(), expected.is_vote());
			}
		}
		assert!(crate::neuron_memory::EvalSettings::default().write_vote.is_none());
		assert_eq!(crate::neuron_memory::EvalSettings::default().write_rule(), WriteRule::Nudge);
	}

	/// CTRL-46: the controller's writer per mode — QUAD family Nudge (vote
	/// refused), the direct modes LastWrite (vote None) or Vote(share).
	#[test]
	fn resolve_controller_names_last_write_and_refuses_quad_vote()
	{
		let share = VoteShare::from_theta(0.2).unwrap();
		for mode in CellMode::ALL
		{
			let code = mode.as_u8();
			match mode.write_rule()
			{
				WriteRule::Nudge =>
				{
					assert_eq!(WriteRule::resolve_controller(code, None), Ok(WriteRule::Nudge), "{mode}");
					assert!(WriteRule::resolve_controller(code, Some(share)).is_err(), "{mode}");
				}
				_ =>
				{
					assert_eq!(WriteRule::resolve_controller(code, None), Ok(WriteRule::LastWrite), "{mode}");
					assert_eq!(WriteRule::resolve_controller(code, Some(share)), Ok(WriteRule::Vote(share)), "{mode}");
				}
			}
		}
		assert!(WriteRule::resolve_controller(250, None).is_err());
		// The IDS resolver never yields the controller writer.
		for mode in CellMode::ALL
		{
			for v in [None, Some(share)]
			{
				assert_ne!(WriteRule::resolve(mode.as_u8(), v), WriteRule::LastWrite);
			}
		}
	}

	/// Vote binning: sign of the net tally, tie → the mode's untrained cell;
	/// `for_training` always gives Vote a counter (no env gate) with that tie.
	#[test]
	fn vote_counter_bins_by_sign()
	{
		use crate::neuron_memory::{oi_apply_nudge, OI_INITIAL, TRUE_U8};
		let tally = |deltas: &[i32]| deltas.iter().fold(OI_INITIAL, |acc, &d| oi_apply_nudge(acc, d));
		for mode in [CellMode::Binary, CellMode::Ternary, CellMode::Pln]
		{
			let bin = CounterBin::for_training(WriteRule::Vote(VoteShare::MAJORITY), mode.as_u8()).expect("vote always counts");
			assert_eq!(bin, CounterBin::Vote { tie: mode.default_cell(UnwrittenLean::False) });
			let share = VoteShare::from_theta(0.1).unwrap();
			assert_eq!(CounterBin::for_training(WriteRule::Vote(share), mode.as_u8()), Some(bin), "{mode}: θ never changes the binning");
			assert_eq!(bin.cell(tally(&[1, -1, -1, -1])), FALSE_U8, "{mode}: 1 vs 3");
			assert_eq!(bin.cell(tally(&[3, -1, -1])), TRUE_U8, "{mode}: weighted 3 vs 2");
			assert_eq!(bin.cell(tally(&[1, -1])), mode.default_cell(UnwrittenLean::False), "{mode}: tie");
			assert_eq!(CounterBin::for_training(mode.write_rule(), mode.as_u8()), None, "{mode}: direct writer");
		}
		assert_eq!(CounterBin::Vote { tie: EMPTY_U8 }.cell(tally(&[1, -1])), EMPTY_U8);
	}

	/// `WriteRule::code` is the GPU wire code: the constants in common.metal
	/// (the single source of shader cell semantics) must match.
	#[test]
	fn write_rule_codes_match_common_metal()
	{
		let src = include_str!("shaders/common.metal");
		for (rule, name) in [
			(WriteRule::Set, "WNN_WRITE_SET"),
			(WriteRule::Lattice, "WNN_WRITE_LATTICE"),
			(WriteRule::Nudge, "WNN_WRITE_NUDGE"),
			(WriteRule::Vote(VoteShare::MAJORITY), "WNN_WRITE_VOTE"),
		]
		{
			let decl = format!("constant uint {name} = {};", rule.code());
			assert!(src.contains(&decl), "common.metal lacks `{decl}`");
		}
	}

	#[test]
	fn names_round_trip_and_are_unique()
	{
		let mut seen = std::collections::HashSet::new();
		for mode in CellMode::ALL
		{
			assert!(seen.insert(mode.name()), "duplicate name {}", mode.name());
			assert_eq!(format!("{mode}"), mode.name());
		}
	}
}
