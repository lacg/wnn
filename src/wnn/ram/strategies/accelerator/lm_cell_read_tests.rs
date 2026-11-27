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

/// IDS-81 follow-up: the GPU adaptation stats (neuron_stats.metal, the default
/// path on macOS) read cells through `wnn_cell_weight(memory_mode, empty_value)`
/// exactly as the CPU fallback reads them through `LmCellRead`. Before the fix
/// the shader read every mode as QUAD_WEIGHTED (a TERNARY/PLN/BINARY TRUE scored
/// 0.25 and never fired) and counted a BINARY dense FALSE as filled.
#[cfg(target_os = "macos")]
mod gpu_stats_parity
{
	use crate::adaptation::{
		compute_batch_stats_gpu, compute_combined_stats, compute_neuron_and_cluster_stats_gpu, AdaptationConfig,
		ClusterStats, GenomeStatsInput, NeuronStats,
	};
	use crate::lm_cell_read::LmCellRead;
	use ram_core::cell_mode::CellMode;
	use ram_core::neuron_memory::{empty_word_for_mode, ClusterStorage};

	const BITS: usize = 3;
	const NEURONS: usize = 3;
	const INPUT_BITS: usize = 6;
	/// Cluster 0 dense (bits <= threshold), cluster 1 sparse.
	const THRESHOLDS: [usize; 2] = [12, 2];

	type Stats = (Vec<NeuronStats>, Vec<ClusterStats>);

	fn lcg(state: &mut u64) -> u64
	{
		*state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		*state >> 33
	}

	fn storages(mode: CellMode, seed: u64) -> Vec<ClusterStorage>
	{
		let mut rng = seed;
		THRESHOLDS
			.iter()
			.map(|&threshold| {
				let mut s = ClusterStorage::new(NEURONS, BITS, threshold, empty_word_for_mode(mode.as_u8()), mode.as_u8());
				for n in 0..NEURONS
				{
					for a in 0..1usize << BITS
					{
						// Leave some addresses unwritten so the miss / fill paths run.
						if lcg(&mut rng) % 4 != 0
						{
							s.write_cell(n, a, (lcg(&mut rng) % mode.num_states() as u64) as i64);
						}
					}
				}
				s
			})
			.collect()
	}

	fn connections(seed: u64) -> Vec<i64>
	{
		let mut rng = seed;
		(0..2 * NEURONS * BITS).map(|_| (lcg(&mut rng) % INPUT_BITS as u64) as i64).collect()
	}

	fn assert_same(mode: CellMode, label: &str, cpu: &Stats, gpu: &Stats)
	{
		for (n, (c, g)) in cpu.0.iter().zip(&gpu.0).enumerate()
		{
			assert_eq!(c.error_rate, g.error_rate, "{mode} {label}: neuron {n} error_rate");
			assert_eq!(c.fill_rate, g.fill_rate, "{mode} {label}: neuron {n} fill_rate");
		}
		for (k, (c, g)) in cpu.1.iter().zip(&gpu.1).enumerate()
		{
			assert_eq!(c.error_rate, g.error_rate, "{mode} {label}: cluster {k} error_rate");
			assert_eq!(c.mean_fill_rate, g.mean_fill_rate, "{mode} {label}: cluster {k} mean_fill_rate");
			assert_eq!(c.neuron_uniqueness, g.neuron_uniqueness, "{mode} {label}: cluster {k} uniqueness");
			assert_eq!(c.neuron_accuracy, g.neuron_accuracy, "{mode} {label}: cluster {k} accuracy");
		}
	}

	#[test]
	fn adaptation_stats_cpu_fallback_matches_gpu_every_mode()
	{
		let Ok(metal) = crate::metal_stats::MetalStatsComputer::new()
		else
		{
			eprintln!("skipping: no Metal device");
			return;
		};
		// Every 6-bit row, twice; stats_sample_size 0 = every example, so the CPU
		// (sample seeds 42/43) and the GPU (seed 42) score the identical sample.
		let packed: Vec<u64> = (0..128u64).map(|r| r % 64).collect();
		let mut rng = 7u64;
		let targets: Vec<u8> = (0..packed.len() * 2).map(|_| (lcg(&mut rng) & 1) as u8).collect();
		let config = AdaptationConfig { total_input_bits: INPUT_BITS, stats_sample_size: 0, ..Default::default() };
		let bits = vec![BITS; 2 * NEURONS];
		let npc = vec![NEURONS; 2];
		for mode in CellMode::ALL
		{
			for empty_value in [0.3f32, 0.7]
			{
				let read = LmCellRead { memory_mode: mode.as_u8(), empty_value };
				let genomes: Vec<(Vec<ClusterStorage>, Vec<i64>)> =
					(0..3u64).map(|g| (storages(mode, 11 + g), connections(101 + g))).collect();
				let cpu: Vec<Stats> = genomes
					.iter()
					.map(|(st, c)| compute_combined_stats(&bits, &npc, c, st, &packed, 1, &targets, packed.len(), 2, &config, read))
					.collect();
				for (g, (st, c)) in genomes.iter().enumerate()
				{
					let gpu = compute_neuron_and_cluster_stats_gpu(&metal, &bits, &npc, c, st, &packed, 1, &targets, packed.len(), 2, &config, read);
					assert_same(mode, &format!("single g{g} empty={empty_value}"), &cpu[g], &gpu);
				}
				let inputs: Vec<GenomeStatsInput> = genomes
					.iter()
					.map(|(st, c)| GenomeStatsInput { bits_per_neuron: &bits, neurons_per_cluster: &npc, connections: c, storages: st })
					.collect();
				let batched = compute_batch_stats_gpu(&metal, &inputs, &packed, 1, &targets, packed.len(), 2, &config, read);
				for (g, b) in batched.iter().enumerate()
				{
					assert_same(mode, &format!("batched g{g} empty={empty_value}"), &cpu[g], b);
				}
			}
		}
	}
}
