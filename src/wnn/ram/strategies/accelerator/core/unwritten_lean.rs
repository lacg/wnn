//! `UnwrittenLean` — which side of the 4-state lattice an UNWRITTEN cell sits on.
//!
//! The 4-state family (QUAD_WEIGHTED / QUAD_BINARY / QSR) has no neutral
//! midpoint: an unwritten address must read as either WEAK_FALSE (1, weight
//! 0.25) or WEAK_TRUE (2 = EMPTY_U8, weight 0.75). That choice is a property of
//! the memory's INITIAL STATE, not of the cell encoding or the write rule, so it
//! is a parameter of `CellMode::default_cell` (Luiz 09/11/2026), not a substrate
//! name. Today's users: the IDS worker leans `False` (an unseen pattern is weak
//! evidence AGAINST the class); the drone controller leans `True` (an untrained
//! thermometer bank decodes to the 0.75 hover neutral, and a fresh cell nudges
//! from EMPTY(2) — see controller_rollout.metal `controller_train`).
//!
//! Modes with a dedicated untrained state (TERNARY / PLN → EMPTY) or none
//! (BINARY → FALSE) ignore the lean.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnwrittenLean
{
	/// 4-state unwritten cell = WEAK_FALSE (1).
	False,
	/// 4-state unwritten cell = WEAK_TRUE (2, the ternary EMPTY code).
	True,
}
