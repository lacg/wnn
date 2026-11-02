//! `ids_calib_sample` (worker ABI 24) through the real IDS search paths.
//!
//! - absent / 0 / >= rows  => every k-fold and train-subset result is
//!   bit-identical to a cache that never heard of the setter;
//! - set => deterministic (same rows, same results, run after run), validation
//!   (`full_hybrid`) untouched, and on a synthetic single-cluster task the
//!   subsampled threshold lands close to the full-set one.
//!
//! TERNARY / BINARY only: they train deterministically on every route, so
//! "bit-identical" is a meaningful assertion without WNN_ORDER_INDEPENDENT_TRAIN
//! (which a test cannot set without racing the other tests).

use crate::ids_cache::{
	evaluate_genomes_ids_cached_full_hybrid, evaluate_genomes_ids_cached_hybrid, evaluate_genomes_ids_kfold_hybrid,
	evaluate_genomes_ids_kfold_mean_hybrid, IDSCache,
};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED, TERNARY};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 24;
const PARTS: usize = 5;
const SEED: u64 = 42;

type Row = (f64, f64, f64, f64, f64, u32);

/// Binary rows whose label is a noisy function of bits 0..3 (~25% attacks),
/// plus an attack "category" per row (0 benign, 1..=3 attack kinds, kind 3 rare).
fn split(n: usize, rng: &mut SmallRng) -> (PackedBits, Vec<i64>, Vec<i64>)
{
	let mut bools = Vec::with_capacity(n * TIB);
	let (mut labels, mut kinds) = (Vec::with_capacity(n), Vec::with_capacity(n));
	for _ in 0..n
	{
		let row: Vec<bool> = (0..TIB).map(|_| rng.gen_bool(0.5)).collect();
		let signal = row[0] && row[1] || row[2] && row[3] && row[4];
		let attack = if rng.gen_bool(0.9) { signal } else { rng.gen_bool(0.3) };
		labels.push(attack as i64);
		kinds.push(if !attack { 0 } else if rng.gen_bool(0.02) { 3 } else { 1 + rng.gen_range(0..2) });
		bools.extend(row);
	}
	(PackedBits::from_bool_slice(&bools, TIB), labels, kinds)
}

struct Data
{
	train: (PackedBits, Vec<i64>, Vec<i64>),
	eval: (PackedBits, Vec<i64>, Vec<i64>),
}

impl Data
{
	fn new(n_train: usize) -> Self
	{
		let mut rng = SmallRng::seed_from_u64(0xCA11B + n_train as u64);
		Self { train: split(n_train, &mut rng), eval: split(n_train / 3, &mut rng) }
	}

	/// A single-cluster binary cache configured like IDSEvaluator's.
	fn cache(&self, mode: u8) -> IDSCache
	{
		let mut cache = IDSCache::new(
			self.train.0.clone(),
			self.train.1.clone(),
			self.eval.0.clone(),
			self.eval.1.clone(),
			None,
			None,
			2,
			TIB,
			PARTS,
			1,
			SEED,
			false,
			true,
			false,
			1.0,
		);
		cache.memory_mode = mode;
		cache
	}
}

/// 6 neurons x 8 bits (dense groups), connections spread over the signal bits.
fn genome() -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let conns: Vec<i64> = (0..6).flat_map(|n| (0..8).map(move |k| ((n * 5 + k * 3) % TIB) as i64)).collect();
	(vec![8; 6], vec![6], conns)
}

fn bits(r: &Row) -> [u64; 5]
{
	[r.0, r.1, r.2, r.3, r.4].map(f64::to_bits)
}

/// Every search route of the cache: each k-fold, the 5-fold mean, each train subset.
fn search_results(cache: &IDSCache) -> Vec<[u64; 5]>
{
	let (b, npc, conns) = genome();
	let mut out = Vec::new();
	for fold in 0..PARTS
	{
		out.push(bits(&evaluate_genomes_ids_kfold_hybrid(cache, &b, &npc, &conns, 1, fold, 0.5, 1.0, 0)[0]));
		out.push(bits(&evaluate_genomes_ids_cached_hybrid(cache, &b, &npc, &conns, 1, fold, 0.5, 1.0, 0)[0]));
	}
	let folds: Vec<usize> = (0..PARTS).collect();
	out.push(bits(&evaluate_genomes_ids_kfold_mean_hybrid(cache, &b, &npc, &conns, 1, &folds, 0.5, 1.0, 0)[0]));
	out
}

fn full_result(cache: &IDSCache) -> [u64; 5]
{
	let (b, npc, conns) = genome();
	bits(&evaluate_genomes_ids_cached_full_hybrid(cache, &b, &npc, &conns, 1, 0.5, 1.0, 0, None)[0])
}

#[test]
fn absent_zero_or_oversized_sample_is_bit_identical()
{
	let data = Data::new(600);
	for mode in [TERNARY, BINARY]
	{
		let want = search_results(&data.cache(mode));
		for sample in [0usize, 600, 10_000]
		{
			let mut cache = data.cache(mode);
			cache.set_calib_sample(sample, SEED, Some(data.train.2.clone())).unwrap();
			assert_eq!(search_results(&cache), want, "mode {mode} sample {sample}");
		}
	}
}

#[test]
fn subsample_is_deterministic_and_leaves_validation_alone()
{
	let data = Data::new(600);
	let mut a = data.cache(TERNARY);
	a.set_calib_sample(120, SEED, Some(data.train.2.clone())).unwrap();
	let mut b = data.cache(TERNARY);
	b.set_calib_sample(120, SEED, Some(data.train.2.clone())).unwrap();
	let first = search_results(&a);
	assert_eq!(first, search_results(&a), "rerun on the same cache");
	assert_eq!(first, search_results(&b), "a fresh cache with the same seed");
	assert_eq!(full_result(&a), full_result(&data.cache(TERNARY)), "validation path never subsamples");
}

#[test]
fn subsampled_threshold_is_close_to_the_full_set_threshold()
{
	// QUAD_WEIGHTED (the production mode): its nudge tally learns the noisy
	// signal, so the threshold sweep has a real interior optimum. The 4-state
	// trainer is order-dependent without OI, hence a tolerance, not equality.
	let data = Data::new(30_000);
	let full = data.cache(QUAD_WEIGHTED);
	let mut sub = data.cache(QUAD_WEIGHTED);
	sub.set_calib_sample(4_000, SEED, Some(data.train.2.clone())).unwrap();
	let (b, npc, conns) = genome();
	for fold in 0..PARTS
	{
		let f = evaluate_genomes_ids_kfold_hybrid(&full, &b, &npc, &conns, 1, fold, 0.5, 1.0, 0)[0];
		let s = evaluate_genomes_ids_kfold_hybrid(&sub, &b, &npc, &conns, 1, fold, 0.5, 1.0, 0)[0];
		eprintln!("fold {fold}: threshold full {:.4} sub {:.4}  f1 full {:.4} sub {:.4}", f.4, s.4, f.2, s.2);
		assert!(f.4 > 0.0 && f.4 < 1.0, "fold {fold}: degenerate full-set threshold {}", f.4);
		assert!((f.4 - s.4).abs() <= 0.05, "fold {fold}: threshold {} vs {}", f.4, s.4);
		assert!((f.2 - s.2).abs() <= 0.01, "fold {fold}: F1 {} vs {}", f.2, s.2);
	}
}

#[test]
fn strata_must_cover_the_train_set()
{
	let data = Data::new(300);
	let mut cache = data.cache(TERNARY);
	assert!(cache.set_calib_sample(50, SEED, Some(vec![0; 299])).is_err());
	assert!(cache.set_calib_sample(50, SEED, Some(data.train.2.clone())).is_ok());
	assert_eq!(cache.calib_sample(), 50);
}

/// Eval-in-place (WNN_EVAL_IN_PLACE=1, live in production): the fused trainer
/// probes ONLY the calibration rows, and their scores are bit-identical to the
/// full-train probe's scores at those rows (training is unchanged; the probe
/// is per-example). OI only — skips otherwise, like the marker_train suite:
///   WNN_ORDER_INDEPENDENT_TRAIN=1 cargo test -p ram_accelerator --lib --no-default-features calib_sample
#[test]
fn fused_probe_on_the_subsample_matches_the_full_probe_at_those_rows()
{
	if !ram_core::neuron_memory::order_independent_training_enabled()
	{
		eprintln!("[calib fused] needs WNN_ORDER_INDEPENDENT_TRAIN=1 — skipping");
		return;
	}
	let data = Data::new(3_000);
	let (b, _npc, conns) = genome();
	let (train, targets, kinds) = &data.train;
	let calib = crate::adaptive::CalibSet::build(train, targets, Some(kinds), 400, SEED, 2).unwrap();
	let probe = |rows: &PackedBits| {
		crate::marker_train::train_single_genome_chunked_scored_probe(
			&b, &conns, train, targets, &[], train.num_rows(), 0, TIB, 0.5, 1.0, 0, None,
			&data.eval.0, data.eval.0.num_rows(), Some(rows), QUAD_WEIGHTED,
		)
		.unwrap()
	};
	let full = probe(train);
	let sub = probe(&calib.input_bits);
	assert_eq!(full.eval_scores, sub.eval_scores, "eval scores unchanged");
	let full_train = full.train_scores.unwrap();
	let want: Vec<u32> = calib.rows.iter().map(|&r| full_train[r].to_bits()).collect();
	let got: Vec<u32> = sub.train_scores.unwrap().iter().map(|x| x.to_bits()).collect();
	assert_eq!(got, want);
}
