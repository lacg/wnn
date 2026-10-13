//! The multi-cluster GA fitness path (`evaluate_genome_hybrid`: the sparse CE
//! fast path and the full-GPU shared-buffer path) must score a genome exactly
//! as `compute_per_example_scores` + `multiclass_ce_acc` (the reference CPU/GPU
//! scorer) does — for genomes whose groups mix SPARSE and DENSE storage, with
//! coverage-aware scoring on and off, and on coalesced layouts.
//!
//! 12/10/2026: the full-GPU path looked groups up by the SUB-index in
//! `group_info` (the index within sparse_exports / dense_exports), not the
//! group index, so a genome with a sparse AND a dense group scored one group
//! with the other group's bits / connections / neuron layout.

use crate::adaptive::{compute_per_example_scores, evaluate_genome_hybrid, multiclass_ce_acc, GenomeExport};
use crate::marker_train::batched_path::{batched_train_offspring_plain, trainer_available};
use ram_core::neuron_memory::{pack_packed_to_u64, EvalSettings, QUAD_WEIGHTED, TERNARY};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 24;
const K: usize = 3;

struct Data
{
	train: PackedBits,
	targets: Vec<i64>,
	negatives: Vec<i64>,
	eval: PackedBits,
	eval_targets: Vec<i64>,
}

impl Data
{
	fn new(rng: &mut SmallRng) -> Self
	{
		let rows = |n: usize, rng: &mut SmallRng| {
			PackedBits::from_bool_slice(&(0..n * TIB).map(|_| rng.gen_bool(0.5)).collect::<Vec<_>>(), TIB)
		};
		let (train, eval) = (rows(800, rng), rows(300, rng));
		let targets: Vec<i64> = (0..800).map(|_| rng.gen_range(0..K as i64)).collect();
		let negatives = targets.iter().flat_map(|&t| (0..K as i64).filter(move |&c| c != t)).collect();
		let eval_targets = (0..300).map(|_| rng.gen_range(0..K as i64)).collect();
		Self { train, targets, negatives, eval, eval_targets }
	}
}

/// K = 3 genome: `bits[c]` per cluster (uniform within a cluster), `npc[c]` neurons.
fn train(bits: [usize; K], npc: [usize; K], data: &Data, mode: u8, rng: &mut SmallRng) -> GenomeExport
{
	let bits_flat: Vec<usize> = (0..K).flat_map(|c| std::iter::repeat(bits[c]).take(npc[c])).collect();
	let conns: Vec<i64> = bits_flat.iter().flat_map(|&b| (0..b).map(|_| rng.gen_range(0..TIB as i64)).collect::<Vec<_>>()).collect();
	let n = data.targets.len();
	batched_train_offspring_plain(
		&bits_flat, &npc, &conns, 1, K, &data.train, &data.targets, &data.negatives, n, K - 1, TIB, 0.5, 1.0, 42, None, mode,
	)
	.expect("train")
	.remove(0)
}

fn assert_fitness_matches_reference(e: &GenomeExport, data: &Data, mode: u8, coverage_aware: bool, what: &str)
{
	let settings = EvalSettings { empty_value: 0.5, memory_mode: mode, coverage_aware, ..EvalSettings::default() };
	let dense = crate::adaptive::get_metal_evaluator();
	let sparse = crate::adaptive::get_sparse_metal_evaluator();
	let n = data.eval_targets.len();
	let (ce, acc, _, _, _) = evaluate_genome_hybrid(
		e, &data.eval, &data.eval_targets, n, K, TIB, settings, dense.as_deref(), sparse.as_deref(), None,
	);
	let (packed, wpe) = pack_packed_to_u64(&data.eval);
	let scores = compute_per_example_scores(
		e, &data.eval, &packed, wpe, n, K, TIB, 0.5, mode, 0, coverage_aware, dense.as_deref(), sparse.as_deref(),
	);
	let (ref_ce, ref_acc, _) = multiclass_ce_acc(&scores, &data.eval_targets);
	assert!((ce - ref_ce).abs() < 1e-5, "{what} cov={coverage_aware}: ce {ce} vs reference {ref_ce}");
	assert_eq!(acc, ref_acc, "{what} cov={coverage_aware}: acc");
}

#[test]
fn multicluster_fitness_matches_reference_scorer()
{
	if !trainer_available()
	{
		eprintln!("[ce_path] no Metal device — skipping");
		return;
	}
	let mut rng = SmallRng::seed_from_u64(0xCE57);
	let data = Data::new(&mut rng);
	let fixtures: [([usize; K], [usize; K], &str); 4] = [
		([14, 14, 14], [4, 4, 4], "one sparse group (fast path)"),
		([10, 10, 10], [4, 4, 4], "one dense group"),
		([14, 10, 14], [4, 4, 4], "sparse + dense groups"),
		([10, 14, 10], [3, 4, 3], "dense + sparse groups"),
	];
	for mode in [QUAD_WEIGHTED, TERNARY]
	{
		for (bits, npc, what) in fixtures
		{
			let e = train(bits, npc, &data, mode, &mut rng);
			for coverage_aware in [false, true]
			{
				assert_fitness_matches_reference(&e, &data, mode, coverage_aware, &format!("{what} mode={mode}"));
			}
		}
	}
}
