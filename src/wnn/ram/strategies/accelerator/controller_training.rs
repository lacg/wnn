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

use crate::neuron_memory::compute_address_sparse;
use crate::sparse_memory::SparseLayerMemory;

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

/// Stub for the full BPTT-through-time port. Returns NotImplemented for now;
/// see module-level docstring for the algorithm reference.
#[pyfunction]
pub fn bptt_train_window_stub() -> PyResult<usize> {
	Err(pyo3::exceptions::PyNotImplementedError::new_err(
		"bptt_train_window: full Rust port pending. See controller_training.rs \
		 module docstring + src/wnn/control/bptt_trainer.py for the algorithm spec.",
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
