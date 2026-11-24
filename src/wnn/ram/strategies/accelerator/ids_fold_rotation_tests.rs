//! `ids_fold_rotation` (IDS-76, worker ABI 30): `IDSCache::set_fold_generation`.
//!
//! - generation 0 is the construction-time partition: a fresh cache is already
//!   there (no-op), and a re-draw followed by generation 0 restores every fold
//!   (rows, targets, negatives, gathered input) and every k-fold search result
//!   bit for bit — the flag-off guarantee;
//! - generation g >= 1 holds a VALID stratified K-fold partition of the TRAIN
//!   rows: disjoint folds covering every row once, per-class counts per fold
//!   within 1, labels and negatives consistent, deterministic per (seed, g),
//!   different across g;
//! - the k-fold CV fitness follows the held partition.
//!
//! TERNARY / BINARY train deterministically on every route, so bit-identical is
//! a meaningful assertion without WNN_ORDER_INDEPENDENT_TRAIN.

use crate::ids_cache::{evaluate_genomes_ids_kfold_hybrid, evaluate_genomes_ids_kfold_mean_hybrid, IDSCache, IDSSubset};
use ram_core::neuron_memory::{BINARY, TERNARY};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 24;
const PARTS: usize = 5;
const SEED: u64 = 20421;
const N_TRAIN: usize = 503;

/// (packed rows, labels): `num_classes` classes, every class present, class 0
/// the majority, the last class rare; labels a noisy function of bits 0..4.
fn data(num_classes: usize, n: usize) -> (PackedBits, Vec<i64>)
{
	let mut rng = SmallRng::seed_from_u64(0xF01D + num_classes as u64);
	let mut bools = Vec::with_capacity(n * TIB);
	let mut labels = Vec::with_capacity(n);
	for i in 0..n
	{
		let row: Vec<bool> = (0..TIB).map(|_| rng.gen_bool(0.5)).collect();
		let signal = (row[0] && row[1]) as usize + (row[2] && row[3] && row[4]) as usize;
		let label = if i < num_classes { i } else if rng.gen_bool(0.85) { signal.min(num_classes - 1) } else { rng.gen_range(0..num_classes) };
		labels.push(label as i64);
		bools.extend(row);
	}
	(PackedBits::from_bool_slice(&bools, TIB), labels)
}

fn cache(num_classes: usize, num_negatives: usize, single_cluster: bool, mode: u8) -> IDSCache
{
	let (train, labels) = data(num_classes, N_TRAIN);
	let (eval, eval_labels) = data(num_classes, 60);
	let mut c = IDSCache::new(
		train, labels, eval, eval_labels, None, None, num_classes, TIB, PARTS, num_negatives, SEED, false,
		single_cluster, false, 1.0,
	);
	c.memory_mode = mode;
	c
}

/// The configurations the folds are built for: binary single-cluster (0
/// negatives), binary two-cluster (1 negative), multiclass with SAMPLED
/// negatives (the per-fold RNG path).
fn configs() -> [(usize, usize, bool); 3]
{
	[(2, 1, true), (2, 1, false), (4, 2, false)]
}

/// Every fold of the held partition, materialised.
fn folds(c: &IDSCache) -> Vec<IDSSubset>
{
	(0..PARTS).map(|k| c.train_subset(k)).collect()
}

fn assert_same_folds(a: &[IDSSubset], b: &[IDSSubset], what: &str)
{
	for (k, (x, y)) in a.iter().zip(b).enumerate()
	{
		assert_eq!(x.num_examples, y.num_examples, "{what} fold {k}: num_examples");
		assert_eq!(x.targets, y.targets, "{what} fold {k}: targets");
		assert_eq!(x.negatives, y.negatives, "{what} fold {k}: negatives");
		assert_eq!(x.input_bits.as_u64_words(), y.input_bits.as_u64_words(), "{what} fold {k}: input words");
	}
}

fn rows(c: &IDSCache) -> Vec<Vec<u32>>
{
	(0..PARTS).map(|k| c.fold_rows(k).to_vec()).collect()
}

/// 6 neurons x 8 bits per cluster (dense groups) over the signal bits.
fn genome(clusters: usize) -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let conns: Vec<i64> =
		(0..6 * clusters).flat_map(|n| (0..8).map(move |k| ((n * 5 + k * 3) % TIB) as i64)).collect();
	(vec![8; 6 * clusters], vec![6; clusters], conns)
}

/// Each single-fold result and the 5-fold CV mean (the GA fitness), as bits.
fn search_results(c: &IDSCache) -> Vec<[u64; 5]>
{
	let (b, npc, conns) = genome(c.num_genome_clusters());
	let bits = |r: &(f64, f64, f64, f64, f64, u32)| [r.0, r.1, r.2, r.3, r.4].map(f64::to_bits);
	let mut out: Vec<[u64; 5]> = (0..PARTS)
		.map(|f| bits(&evaluate_genomes_ids_kfold_hybrid(c, &b, &npc, &conns, 1, f, 0.5, 1.0, 0)[0]))
		.collect();
	let all: Vec<usize> = (0..PARTS).collect();
	out.push(bits(&evaluate_genomes_ids_kfold_mean_hybrid(c, &b, &npc, &conns, 1, &all, 0.5, 1.0, 0)[0]));
	out
}

#[test]
fn generation_zero_is_the_construction_partition()
{
	for (classes, negs, single) in configs()
	{
		let mut c = cache(classes, negs, single, TERNARY);
		let before = folds(&c);
		assert!(!c.set_fold_generation(0), "a fresh cache already holds P(0)");
		assert_same_folds(&folds(&c), &before, "P(0) no-op");
	}
}

#[test]
fn redraw_then_zero_restores_folds_and_fitness_bit_for_bit()
{
	for mode in [TERNARY, BINARY]
	{
		for (classes, negs, single) in configs()
		{
			let fresh = cache(classes, negs, single, mode);
			let (want_folds, want_fit) = (folds(&fresh), search_results(&fresh));
			let mut c = cache(classes, negs, single, mode);
			for g in [3u64, 1, 7]
			{
				assert!(c.set_fold_generation(g));
			}
			assert!(c.set_fold_generation(0), "back to P(0) is a change");
			assert_same_folds(&folds(&c), &want_folds, "restored P(0)");
			assert_eq!(search_results(&c), want_fit, "mode {mode} classes {classes}: restored fitness");
		}
	}
}

#[test]
fn rotated_partition_is_a_valid_stratified_cv_partition()
{
	for (classes, negs, single) in configs()
	{
		let (_, labels) = data(classes, N_TRAIN);
		let mut c = cache(classes, negs, single, TERNARY);
		let p0 = rows(&c);
		let mut seen = vec![p0.clone()];
		for g in 1..=6u64
		{
			assert!(c.set_fold_generation(g), "g={g} re-draws");
			let p = rows(&c);
			assert_partition(&p, &labels, classes, g);
			assert_fold_labels(&c, &p, &labels, classes, negs, single);
			assert!(!seen.contains(&p), "g={g}: partition repeats an earlier generation");
			seen.push(p);
		}
	}
}

/// Disjoint folds covering every TRAIN row once; per-class counts per fold within 1.
fn assert_partition(p: &[Vec<u32>], labels: &[i64], classes: usize, g: u64)
{
	let mut all: Vec<u32> = p.iter().flatten().copied().collect();
	all.sort_unstable();
	assert_eq!(all, (0..labels.len() as u32).collect::<Vec<_>>(), "g={g}: folds must cover every row exactly once");
	for class in 0..classes as i64
	{
		let counts: Vec<usize> = p.iter().map(|f| f.iter().filter(|&&r| labels[r as usize] == class).count()).collect();
		let (lo, hi) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
		assert!(hi - lo <= 1, "g={g} class {class}: per-fold counts {counts:?} not stratified");
	}
}

/// Each fold's targets are its rows' labels; negatives exclude the target.
fn assert_fold_labels(c: &IDSCache, p: &[Vec<u32>], labels: &[i64], classes: usize, negs: usize, single: bool)
{
	let per_row = if single { 0 } else { negs.min(classes - 1) };
	for (k, fold_rows) in p.iter().enumerate()
	{
		let f = c.train_subset(k);
		let want: Vec<i64> = fold_rows.iter().map(|&r| labels[r as usize]).collect();
		assert_eq!(f.targets, want, "fold {k}: targets are the rows' labels");
		assert_eq!(f.negatives.len(), want.len() * per_row, "fold {k}: negatives per row");
		for (i, &t) in want.iter().enumerate()
		{
			assert!(!f.negatives[i * per_row..(i + 1) * per_row].contains(&t), "fold {k} row {i}: negative == target");
		}
	}
}

#[test]
fn rotation_is_deterministic_per_seed_and_generation()
{
	let mut a = cache(4, 2, false, TERNARY);
	let mut b = cache(4, 2, false, TERNARY);
	for g in [5u64, 2, 9]
	{
		a.set_fold_generation(g);
		b.set_fold_generation(g);
		assert_eq!(rows(&a), rows(&b), "g={g}");
		assert_same_folds(&folds(&a), &folds(&b), "same (seed, g)");
	}
	assert!(!a.set_fold_generation(9), "the held generation again is a no-op");
}

#[test]
fn fitness_follows_the_held_partition()
{
	let mut c = cache(2, 1, true, TERNARY);
	let p0 = search_results(&c);
	c.set_fold_generation(4);
	let p4 = search_results(&c);
	assert_ne!(p0, p4, "a re-drawn partition changes the k-fold scores");
	let mut d = cache(2, 1, true, TERNARY);
	d.set_fold_generation(4);
	assert_eq!(search_results(&d), p4, "P(4) scores are a function of (seed, 4)");
}
