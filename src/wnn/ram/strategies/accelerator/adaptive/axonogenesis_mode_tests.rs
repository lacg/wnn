//! IDS-81: axonogenesis reads cells through `cell_to_weight(memory_mode)`.
//!
//! One 2-bit neuron reading inputs [0, 1] whose memory fires only at address
//! 0b11, on data where the class is `in2 AND in1`: swapping the weak input 0
//! for input 2 makes the neuron exact, so the pass must rewire 0 -> 2 in EVERY
//! mode. The memory holds the mode-native TRUE/FALSE cells; the pre-IDS-81
//! pass read them through `QUAD_WEIGHTS`, so a ternary TRUE (1) scored 0.25,
//! never fired, and TERNARY / PLN / BINARY never rewired.

use super::adaptive_eval::axonogenesis_pass_adaptive;
use super::*;
use ram_core::cell_mode::CellMode;

const TIB: usize = 4;

/// Every 4-bit row 8 times (one packed word each) and its class: 0 iff in2 AND in1.
fn data() -> (Vec<u64>, Vec<i64>)
{
	let rows: Vec<u64> = (0..128u64).map(|r| r % 16).collect();
	let targets = rows.iter().map(|&w| if (w >> 2) & 1 == 1 && (w >> 1) & 1 == 1 { 0 } else { 1 }).collect();
	(rows, targets)
}

/// One dense 2-bit neuron (one cluster) firing only at address 3.
fn export(mode: CellMode) -> GenomeExport
{
	let mut e = GenomeExport::empty();
	e.groups = vec![ConfigGroup::new(1, 2, vec![0])];
	e.group_info = vec![(false, 0, vec![0])];
	e.dense_exports = vec![vec![(mode.true_cell() as i64) << (3 * ram_core::neuron_memory::BITS_PER_CELL)]];
	e.dense_coverage = vec![Vec::new()];
	e
}

fn rewire(mode: CellMode) -> Vec<i64>
{
	let (rows, targets) = data();
	let mut connections = vec![0i64, 1];
	let stats = [crate::adaptation::NeuronStats { fill_rate: 0.25, error_rate: 0.25, connection_entropy: vec![0.0, 1.0] }];
	let config = crate::adaptation::AdaptationConfig { total_input_bits: TIB, ..Default::default() };
	let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(7);
	axonogenesis_pass_adaptive(
		&[2], &[1], &mut connections, &stats, &export(mode), &[(0, 0)], &[0], &config, &rows, 1, &targets,
		rows.len(), 1, mode.as_u8(), 0.0, 1.0, &mut rng,
	);
	connections
}

#[test]
fn axonogenesis_rewires_the_same_in_every_mode()
{
	for mode in CellMode::ALL
	{
		assert_eq!(rewire(mode), vec![2, 1], "{mode}: the weak input 0 must be rewired to the exact input 2");
	}
}
