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

use crate::neuron_memory::{
	BINARY, EMPTY_U8, FALSE_U8, PLN, QSR, QUAD_BINARY, QUAD_WEAK_FALSE, QUAD_WEIGHTED, TERNARY,
};

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
	/// ("never seen → no vote"); QUAD_*/QSR: WEAK_FALSE(1).
	#[inline]
	pub fn default_cell(self) -> u8
	{
		match self
		{
			CellMode::Ternary | CellMode::Pln => EMPTY_U8,
			CellMode::Binary => FALSE_U8,
			CellMode::QuadBinary | CellMode::QuadWeighted | CellMode::Qsr => QUAD_WEAK_FALSE as u8,
		}
	}

	/// The sparse-miss cell under the coverage-aware rule
	/// (docs/COVERAGE_AWARE_SCORER_SPEC.md): a miss resolves to cell 0, which
	/// weighs 0.0 in EVERY mode, so a class is penalised in proportion to how
	/// much of its memory is empty. `coverage_aware == false` is bit-exact OFF.
	#[inline]
	pub fn miss_cell(self, coverage_aware: bool) -> u8
	{
		if coverage_aware
		{
			0
		}
		else
		{
			self.default_cell()
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
				build_empty_word(mode.default_cell() as i64),
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
				mode.default_cell() as u32,
				"{mode}"
			);
			assert_eq!(
				default_cell_for_coverage(mode.as_u8(), false),
				mode.miss_cell(false) as u32
			);
			assert_eq!(default_cell_for_coverage(mode.as_u8(), true), 0);
		}
	}

	/// A default (untrained) cell weighs: TERNARY → empty_value, PLN → 0.5,
	/// QUAD_WEIGHTED/QSR → 0.25, QUAD_BINARY → 0.0, BINARY → 0.0.
	#[test]
	fn default_cell_weights_per_mode()
	{
		let w = |m: CellMode| cell_to_weight(m.default_cell() as i64, m.as_u8(), 0.37);
		assert_eq!(w(CellMode::Ternary), 0.37);
		assert_eq!(w(CellMode::Pln), 0.5);
		assert_eq!(w(CellMode::QuadWeighted), 0.25);
		assert_eq!(w(CellMode::Qsr), 0.25);
		assert_eq!(w(CellMode::QuadBinary), 0.0);
		assert_eq!(w(CellMode::Binary), 0.0);
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
