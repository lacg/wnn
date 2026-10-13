//! CPU vs GPU scorer parity on MIXED-BITS genomes (12/10/2026).
//!
//! A group's connections are prefix-padded with -1 to its max bits
//! (`reorganize_connections_for_gpu`). The Metal kernels skip negative
//! connections; the CPU readers (`compute_address_packed_bytes`, used by the
//! CPU sparse and dense paths of `compute_per_example_scores`) cast -1 to
//! usize::MAX and read out of bounds. Every reader now shares one address
//! definition (`neuron_memory::address_over`) that gives a pad slot a 0 bit.
//!
//! Per memory mode: GPU-trained exports of mixed-bits genomes (one sparse
//! group, one dense group, a K = 3 multi-group genome) scored on the CPU and on
//! the GPU, coverage-aware on and off — the per-example cluster scores agree.

use crate::adaptive::{compute_per_example_scores, GenomeExport};
use crate::marker_train::batched_path::{batched_train_offspring_plain, trainer_available};
use ram_core::neuron_memory::{pack_packed_to_u64, BINARY, PLN, QSR, QUAD_BINARY, QUAD_WEIGHTED, TERNARY};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 24;
const RUN_SEED: u64 = 5;
const ALL_MODES: [u8; 6] = [QUAD_WEIGHTED, QUAD_BINARY, QSR, TERNARY, BINARY, PLN];

struct Data
{
	train: PackedBits,
	targets: Vec<i64>,
	negatives: Vec<i64>,
	eval: PackedBits,
	num_clusters: usize,
}

impl Data
{
	fn new(num_clusters: usize, rng: &mut SmallRng) -> Self
	{
		let rows = |n: usize, rng: &mut SmallRng| {
			PackedBits::from_bool_slice(&(0..n * TIB).map(|_| rng.gen_bool(0.5)).collect::<Vec<_>>(), TIB)
		};
		let (train, eval) = (rows(600, rng), rows(200, rng));
		let classes = num_clusters.max(2) as i64;
		let targets: Vec<i64> = (0..600).map(|_| rng.gen_range(0..classes)).collect();
		let negatives = if num_clusters < 2
		{
			Vec::new()
		}
		else
		{
			targets.iter().flat_map(|&t| (0..classes).filter(move |&c| c != t)).collect()
		};
		Self { train, targets, negatives, eval, num_clusters }
	}

	fn num_negatives(&self) -> usize
	{
		if self.num_clusters > 1
		{
			self.num_clusters - 1
		}
		else
		{
			0
		}
	}
}

/// (bits per neuron, neurons per cluster, connections) with random wiring.
fn genome(bits: &[usize], npc: &[usize], rng: &mut SmallRng) -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let conns = bits.iter().flat_map(|&b| (0..b).map(|_| rng.gen_range(0..TIB as i64)).collect::<Vec<_>>()).collect();
	(bits.to_vec(), npc.to_vec(), conns)
}

fn train(g: &(Vec<usize>, Vec<usize>, Vec<i64>), data: &Data, mode: u8) -> GenomeExport
{
	let n = data.targets.len();
	batched_train_offspring_plain(
		&g.0, &g.1, &g.2, 1, data.num_clusters, &data.train, &data.targets, &data.negatives, n,
		data.num_negatives(), TIB, 0.5, 1.0, 42, None, mode,
	)
	.expect("gpu train")
	.remove(0)
}

fn scores(e: &GenomeExport, data: &Data, mode: u8, coverage_aware: bool, gpu: bool) -> Vec<Vec<f64>>
{
	let dense = if gpu { crate::metal_ramlm::MetalRAMLMEvaluator::new().ok() } else { None };
	let sparse = if gpu { ram_core::metal_sparse::MetalSparseEvaluator::new().ok() } else { None };
	assert!(!gpu || (dense.is_some() && sparse.is_some()), "GPU scorers unavailable");
	let (packed, wpe) = pack_packed_to_u64(&data.eval);
	compute_per_example_scores(
		e, &data.eval, &packed, wpe, data.eval.num_rows(), data.num_clusters, TIB, 0.5, mode, RUN_SEED,
		coverage_aware, dense.as_ref(), sparse.as_ref(),
	)
}

fn assert_cpu_matches_gpu(e: &GenomeExport, data: &Data, mode: u8, what: &str)
{
	for coverage_aware in [false, true]
	{
		let (cpu, gpu) = (scores(e, data, mode, coverage_aware, false), scores(e, data, mode, coverage_aware, true));
		for (ex, (c, g)) in cpu.iter().zip(&gpu).enumerate()
		{
			for (k, (a, b)) in c.iter().zip(g).enumerate()
			{
				assert!((a - b).abs() < 1e-6, "{what} mode={mode} cov={coverage_aware} ex={ex} c={k}: cpu={a} gpu={b}");
			}
		}
	}
}

#[test]
fn cpu_scorer_matches_gpu_on_mixed_bits()
{
	if !trainer_available()
	{
		eprintln!("[mixed_bits] no Metal device — skipping");
		return;
	}
	let mut rng = SmallRng::seed_from_u64(0x0B17);
	let binary = Data::new(1, &mut rng);
	let multi = Data::new(3, &mut rng);
	let sparse = genome(&[14, 10, 14, 12, 14], &[5], &mut rng);
	let dense = genome(&[10, 6, 10, 8, 10], &[5], &mut rng);
	let k3 = genome(&[14, 13, 14, 14, 9, 14, 10, 8, 10], &[3, 3, 3], &mut rng);
	for mode in ALL_MODES
	{
		assert_cpu_matches_gpu(&train(&sparse, &binary, mode), &binary, mode, "sparse mixed");
		assert_cpu_matches_gpu(&train(&dense, &binary, mode), &binary, mode, "dense mixed");
		assert_cpu_matches_gpu(&train(&k3, &multi, mode), &multi, mode, "K=3 mixed");
	}
}
