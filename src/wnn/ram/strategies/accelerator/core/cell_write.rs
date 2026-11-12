//! Per-cell WRITE semantics of every memory mode — the ONE implementation the
//! IDS worker and the drone controller share (Luiz 09/11/2026: "they should
//! reuse the types of neurons, this should not be different codes").
//!
//! Memory mode = what a cell IS (`CellMode`); write rule = how a visit writes it
//! (`WriteRule`). This file holds the cell-level primitives every trainer calls:
//! the 4-state lattice step, the mode-native TRUE/FALSE cells, the recurrent
//! fire bit, planted cells, the solver's nudge distance and the stochastic-read
//! coin probability. GPU twins: core/shaders/common.metal `wnn_*` (prepended to
//! every shader at compile time — never per-shader copies).
//!
//! Until 12/11/2026 the controller kept its own copies (controller/cell_mode.rs,
//! controller_training.rs `nudge_toward*` / `qsr_nudge_distance`,
//! controller_bptt.metal `bw_*`, controller_rollout.metal `ctrl_fire_bit` /
//! `ctrl_nudge_distance` / `slot_set_direct`), and the GPU split trainer had
//! drifted: it sent PLN through the QUAD nudge while the CPU set PLN directly.

use crate::cell_mode::{CellMode, WriteRule};
use crate::neuron_memory::{cell_to_weight, EMPTY_U8, FALSE_U8, QUAD_TRUE, QUAD_WEAK_FALSE, TRUE_U8};

/// ONE step of the 4-state lattice FALSE(0) → WEAK_FALSE(1) → WEAK_TRUE(2) →
/// TRUE(3) toward `target_true`, saturating at both ends. THE nudge of every
/// 4-state writer (IDS clamped trainers, controller DAgger, GPU `wnn_quad_step`).
/// Cells are 2-bit: only the low 2 bits are read.
#[inline(always)]
pub fn quad_step(cell: u8, target_true: bool) -> u8
{
	let c = cell & 0x3;
	if target_true
	{
		c + (c < QUAD_TRUE as u8) as u8
	}
	else
	{
		c - (c > 0) as u8
	}
}

/// ONE lattice step toward a SPECIFIC 4-state value (0..3); holds once there.
#[inline(always)]
pub fn quad_step_toward(cell: u8, target: u8) -> u8
{
	let (c, t) = (cell & 0x3, target & 0x3);
	if c == t
	{
		c
	}
	else
	{
		quad_step(c, t > c)
	}
}

impl CellMode
{
	/// The wire code → mode for a caller that VALIDATED the code at construction
	/// (the controller's `validate_mode`). Panics on an unknown code — reaching
	/// one here is a construction bug, never data.
	#[inline]
	pub fn of(code: u8) -> CellMode
	{
		match CellMode::from_u8(code)
		{
			Some(mode) => mode,
			None => panic!("unknown memory_mode {code} (validated at construction)"),
		}
	}

	/// The 4-state lattice family (QUAD_WEIGHTED / QUAD_BINARY / QSR): cells
	/// nudge one step at a time and fire on their MSB.
	#[inline]
	pub fn is_quad_family(self) -> bool
	{
		self.num_states() == 4
	}

	/// The mode-native fully-TRUE cell: 4-state TRUE(3), else the ternary TRUE(1).
	#[inline]
	pub fn true_cell(self) -> u8
	{
		if self.is_quad_family()
		{
			QUAD_TRUE as u8
		}
		else
		{
			TRUE_U8
		}
	}

	/// The mode-native fully-FALSE cell: 0 in both encodings.
	#[inline]
	pub fn false_cell(self) -> u8
	{
		FALSE_U8
	}

	/// The 1-bit "fired" read of a cell (recurrent state feedback, thermometer
	/// monotonicity). 4-state: the MSB (WEAK_TRUE / TRUE). Others: cell == TRUE(1)
	/// — the MSB rule would invert ternary (TRUE=1 has MSB 0, EMPTY=2 has MSB 1).
	#[inline]
	pub fn fire_bit(self, cell: u8) -> bool
	{
		if self.is_quad_family()
		{
			(cell >> 1) & 1 != 0
		}
		else
		{
			cell == TRUE_U8
		}
	}

	/// A PLANTED cell (split-trainer latches / counters). 4-state: strong TRUE(3)
	/// on, soft WEAK_FALSE(1) off — one later nudge can flip a planted off but not
	/// erase a latch. Others have no soft states: TRUE / FALSE directly.
	#[inline]
	pub fn plant_cell(self, on: bool) -> u8
	{
		match (self.is_quad_family(), on)
		{
			(true, true) => QUAD_TRUE as u8,
			(true, false) => QUAD_WEAK_FALSE as u8,
			(false, true) => TRUE_U8,
			(false, false) => FALSE_U8,
		}
	}

	/// The writer of a SEQUENTIAL (order-dependent, label-correcting) trainer:
	/// the 4-state family nudges; every other mode is `LastWrite`.
	#[inline]
	pub fn sequential_write_rule(self) -> WriteRule
	{
		if self.is_quad_family()
		{
			WriteRule::Nudge
		}
		else
		{
			WriteRule::LastWrite
		}
	}

	/// One sequential write toward a boolean target: 4-state `quad_step`;
	/// otherwise `WriteRule::LastWrite` (the cell becomes the target outright).
	#[inline]
	pub fn nudge_step(self, cell: u8, target_true: bool) -> u8
	{
		match self.sequential_write_rule()
		{
			WriteRule::Nudge => quad_step(cell, target_true),
			_ if target_true => self.true_cell(),
			_ => self.false_cell(),
		}
	}

	/// One sequential write toward a SPECIFIC mode-native cell value.
	#[inline]
	pub fn nudge_toward_value(self, cell: u8, target: u8) -> u8
	{
		match self.sequential_write_rule()
		{
			WriteRule::Nudge => quad_step_toward(cell, target),
			_ => target,
		}
	}

	/// Solver cost: how far `cell` is from satisfying a boolean target.
	/// 4-state: 0..3 lattice steps. Others: 0 already the target, 1 untrained
	/// (EMPTY), 2 explicit opposite (a flip erases learned evidence — dearer).
	#[inline]
	pub fn nudge_distance(self, cell: u8, target_true: bool) -> u8
	{
		if self.is_quad_family()
		{
			let c = cell & 0x3;
			return if target_true { 3 - c } else { c };
		}
		let want = if target_true { TRUE_U8 } else { FALSE_U8 };
		match cell
		{
			c if c == want => 0,
			EMPTY_U8 => 1,
			_ => 2,
		}
	}

	/// TRUE-firing probability of a stochastic read (QSR: the graded QUAD weight;
	/// PLN: FALSE 0, TRUE 1, u-state the fair 0.5). For every mode this IS the
	/// deterministic `cell_to_weight` — the read's expected value — so the
	/// deterministic modes return their weight and `is_stochastic` is the only
	/// branch a caller needs.
	#[inline]
	pub fn coin_prob(self, cell: u8, empty_value: f32) -> f32
	{
		cell_to_weight(cell as i64, self.as_u8(), empty_value)
	}
}

#[cfg(test)]
#[path = "cell_write_tests.rs"]
mod tests;
