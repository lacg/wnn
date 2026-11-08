//! IDS-50 end-to-end: at neuron_sample_rate 0.25 a genome's fitness must not
//! depend on the ORDER of its neurons (the mask is keyed on identity), and the
//! CPU trainer must match the GPU marker trainer under sampling.
//!
//! Before ABI 25 the mask hashed the neuron's flat index, so reversing one
//! cluster changed which examples every neuron in it trained on — and the
//! scores moved (the permutation tests below fail on that code).
//!
//! Fixture (from marker_train's multigroup parity): class c sets input bits
//! {2c, 2c+1}; every neuron reads bits 0..6, so each (neuron, address) sees one
//! class only — every cell's nudges are monotone and the training order is
//! immaterial, which makes CPU-vs-GPU comparisons exact.

use crate::adaptive::{
	build_groups, build_neuron_metadata, compute_per_example_scores, export_genome_for_gpu, per_cluster_max_bits,
	reorganize_connections_for_gpu, train_genome_in_slot, GenomeExport, GroupMemory,
};
use crate::marker_train::batched_path::{batched_train_offspring_plain, trainer_available};
use ram_core::neuron_memory::QUAD_WEIGHTED;
use ram_core::packed_bits::PackedBits;

const TIB: usize = 16;
const K: usize = 3;
const NUM_TRAIN: usize = 600;
const RATE: f32 = 0.25;
const NPC: [usize; 3] = [3, 3, 4];

struct Data
{
	packed: PackedBits,
	targets: Vec<i64>,
	negatives: Vec<i64>,
}

fn data() -> Data
{
	let mut bools = vec![false; NUM_TRAIN * TIB];
	let targets: Vec<i64> = (0..NUM_TRAIN).map(|ex| (ex % K) as i64).collect();
	for ex in 0..NUM_TRAIN
	{
		let c = ex % K;
		bools[ex * TIB + 2 * c] = true;
		bools[ex * TIB + 2 * c + 1] = true;
		for b in 6..TIB
		{
			bools[ex * TIB + b] = (ex.wrapping_mul(2654435761) >> (b - 6)) & 1 == 1;
		}
	}
	let negatives: Vec<i64> = targets.iter().flat_map(|&t| (0..K as i64).filter(move |&c| c != t)).collect();
	Data { packed: PackedBits::from_bool_slice(&bools, TIB), targets, negatives }
}

/// Neuron `i` (global, logical order): 8 bits (clusters 0/1) or 10 (cluster 2),
/// class bits 0..6 + a neuron-specific rotation of the noise bits.
fn neuron(i: usize) -> (usize, Vec<i64>)
{
	let bits = if i < NPC[0] + NPC[1] { 8 } else { 10 };
	let conns = (0..bits).map(|k| if k < 6 { k as i64 } else { (6 + (k - 6 + 3 * i) % 10) as i64 }).collect();
	(bits, conns)
}

/// The genome with neurons in `order` (a permutation that keeps each neuron in
/// its cluster).
fn genome(order: &[usize]) -> (Vec<usize>, Vec<i64>)
{
	let parts: Vec<(usize, Vec<i64>)> = order.iter().map(|&i| neuron(i)).collect();
	(parts.iter().map(|p| p.0).collect(), parts.into_iter().flat_map(|p| p.1).collect())
}

fn identity() -> Vec<usize>
{
	(0..NPC.iter().sum()).collect()
}

/// Cluster 1 reversed, cluster 2 rotated.
fn permuted() -> Vec<usize>
{
	vec![0, 1, 2, 5, 4, 3, 7, 8, 9, 6]
}

fn scores(export: &GenomeExport, d: &Data) -> Vec<Vec<f64>>
{
	compute_per_example_scores(export, &d.packed, NUM_TRAIN, K, TIB, 0.5, QUAD_WEIGHTED, 0, false, None, None)
}

fn cpu_export(bits: &[usize], conns: &[i64], d: &Data) -> GenomeExport
{
	let groups = build_groups(&per_cluster_max_bits(bits, &NPC), &NPC);
	let (starts, offsets) = build_neuron_metadata(bits, &NPC);
	let mut c2g = vec![(0usize, 0usize); K];
	for (gi, g) in groups.iter().enumerate()
	{
		g.cluster_ids.iter().enumerate().for_each(|(li, &c)| c2g[c] = (gi, li));
	}
	let mut mems: Vec<GroupMemory> = groups.iter().map(|g| GroupMemory::new(g.total_neurons(), g.bits, QUAD_WEIGHTED)).collect();
	let rule = ram_core::cell_mode::WriteRule::of_code(QUAD_WEIGHTED);
	train_genome_in_slot(
		&mut mems, &groups, conns, bits, &starts, &offsets, &c2g, &d.packed, &d.targets, &d.negatives, NUM_TRAIN, K - 1, TIB, None,
		RATE, 42, QUAD_WEIGHTED, rule, None, false,
	);
	export_genome_for_gpu(&mems, &groups, &reorganize_connections_for_gpu(conns, bits, &NPC, &groups))
}

fn gpu_export(bits: &[usize], conns: &[i64], d: &Data) -> GenomeExport
{
	let rule = ram_core::cell_mode::WriteRule::of_code(QUAD_WEIGHTED);
	batched_train_offspring_plain(
		bits, &NPC, conns, 1, K, &d.packed, &d.targets, &d.negatives, NUM_TRAIN, K - 1, TIB, 0.5, RATE, 42, None, QUAD_WEIGHTED, rule,
	)
	.expect("gpu train")
	.remove(0)
}

fn assert_same(a: &[Vec<f64>], b: &[Vec<f64>], what: &str)
{
	for (ex, (ra, rb)) in a.iter().zip(b).enumerate()
	{
		for c in 0..K
		{
			assert!((ra[c] - rb[c]).abs() < 1e-12, "{}: ex {} cluster {}: {} vs {}", what, ex, c, ra[c], rb[c]);
		}
	}
}

fn cpu_scores(order: &[usize], d: &Data) -> Vec<Vec<f64>>
{
	let (bits, conns) = genome(order);
	scores(&cpu_export(&bits, &conns, d), d)
}

fn gpu_scores(order: &[usize], d: &Data) -> Vec<Vec<f64>>
{
	let (bits, conns) = genome(order);
	scores(&gpu_export(&bits, &conns, d), d)
}

#[test]
fn cpu_fitness_is_permutation_invariant_at_quarter_rate()
{
	let d = data();
	assert_same(&cpu_scores(&identity(), &d), &cpu_scores(&permuted(), &d), "cpu permuted");
}

#[test]
fn gpu_fitness_is_permutation_invariant_at_quarter_rate()
{
	if !trainer_available()
	{
		eprintln!("[sample_key_parity] no Metal device — skipping");
		return;
	}
	let d = data();
	assert_same(&gpu_scores(&identity(), &d), &gpu_scores(&permuted(), &d), "gpu permuted");
}

#[test]
fn cpu_fallback_matches_gpu_sampled_quarter_rate()
{
	if !trainer_available()
	{
		eprintln!("[sample_key_parity] no Metal device — skipping");
		return;
	}
	let d = data();
	assert_same(&cpu_scores(&identity(), &d), &gpu_scores(&identity(), &d), "cpu vs gpu (identity)");
	assert_same(&cpu_scores(&permuted(), &d), &gpu_scores(&identity(), &d), "cpu permuted vs gpu");
}
