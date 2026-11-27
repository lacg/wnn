//! IDS-81: the three LM CPU read sites score a cell through
//! `cell_to_weight(memory_mode, empty_value)` in EVERY mode.
//!
//! One 2-bit neuron (one cluster) holding the mode-native TRUE at address 3,
//! cell 2 at address 1 (TERNARY/PLN EMPTY, QUAD WEAK_TRUE, BINARY not-TRUE)
//! and FALSE elsewhere, read at all four addresses. Before IDS-81:
//!   - `ramlm::forward_for_mode` sent BINARY / QSR / PLN to the QUAD_WEIGHTED
//!     forward (a BINARY TRUE scored 0.25);
//!   - the bitwise CPU forward scored QSR / PLN as a constant 0.5;
//!   - the adaptation stats read every mode through QUAD_WEIGHTS (a TERNARY
//!     TRUE never fired).

use crate::bitwise_ramlm::{compute_genome_layout, cpu_forward_eval_into, BitwiseEvalSubset};
use ram_core::cell_mode::CellMode;
use ram_core::neuron_memory::{cell_to_weight, compute_address, compute_address_packed, empty_word_for_mode, ClusterStorage};

const EMPTY_VALUE: f32 = 0.3;
const CONNS: [i64; 2] = [0, 1];

/// The cell this fixture stores at `addr` under `mode`.
fn cell(mode: CellMode, addr: usize) -> i64
{
	match addr
	{
		3 => mode.true_cell() as i64,
		1 => 2,
		_ => 0,
	}
}

fn want(mode: CellMode, addr: usize) -> f32
{
	cell_to_weight(cell(mode, addr), mode.as_u8(), EMPTY_VALUE)
}

/// The four 2-bit input rows as one packed word each.
fn rows() -> Vec<u64>
{
	(0..4u64).collect()
}

fn storage(mode: CellMode) -> ClusterStorage
{
	let mut s = ClusterStorage::new(1, 2, 12, empty_word_for_mode(mode.as_u8()), mode.as_u8());
	(0..4).for_each(|a| s.write_cell(0, a, cell(mode, a)));
	s
}

#[test]
fn ramlm_forward_reads_every_mode_through_cell_to_weight()
{
	for mode in CellMode::ALL
	{
		let words = [(0..4).fold(0i64, |w, a| w | (cell(mode, a) << (a * 2)))];
		let bools: Vec<bool> = rows().iter().flat_map(|&r| [r & 1 == 1, r & 2 == 2]).collect();
		let got = crate::ramlm::forward_for_mode(&bools, &CONNS, &words, 4, 2, 2, 1, 1, 1, mode.as_u8(), EMPTY_VALUE);
		for ex in 0..4
		{
			let addr = compute_address(&bools[ex * 2..ex * 2 + 2], &CONNS, 2);
			assert_eq!(got[ex], want(mode, addr), "{mode} example {ex} (address {addr})");
		}
	}
}

#[test]
fn bitwise_cpu_forward_reads_every_mode_through_cell_to_weight()
{
	let layout = compute_genome_layout(&[2], &[1], 12, 4);
	let eval = BitwiseEvalSubset { packed_input: rows(), targets: vec![0; 4], num_examples: 4, words_per_example: 1 };
	for mode in CellMode::ALL
	{
		let mut out = vec![0.0f32; 4];
		cpu_forward_eval_into(&CONNS, &[2], &[1], 1, &layout, &eval, &[storage(mode)], &mut out, mode.as_u8(), EMPTY_VALUE);
		for (ex, &row) in rows().iter().enumerate()
		{
			let addr = compute_address_packed(&[row], &CONNS, 2);
			assert_eq!(out[ex], want(mode, addr), "{mode} example {ex} (address {addr})");
		}
	}
}

#[test]
fn adaptation_stats_read_every_mode_through_cell_to_weight()
{
	// Target bit = "fires at this address" under the mode's own read, so a
	// correct read has zero error in every mode.
	let packed: Vec<u64> = (0..32u64).map(|r| r % 4).collect();
	for mode in CellMode::ALL
	{
		let targets: Vec<u8> = packed.iter().map(|&r| (want(mode, compute_address_packed(&[r], &CONNS, 2)) >= 0.5) as u8).collect();
		let config = crate::adaptation::AdaptationConfig { total_input_bits: 2, ..Default::default() };
		let read = crate::lm_cell_read::LmCellRead { memory_mode: mode.as_u8(), empty_value: EMPTY_VALUE };
		let stats = crate::adaptation::compute_neuron_stats(
			&[2], &[1], &CONNS, &[storage(mode)], &packed, 1, &targets, packed.len(), 1, &config, read,
		);
		assert_eq!(stats[0].error_rate, 0.0, "{mode}: the stats pass must read cells as the mode does");
	}
}
