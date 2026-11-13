//! Per-neuron output export (IDS-57 arm E, 12/11/2026): the exported cells,
//! combined by the scorer's rule, ARE the reported scores, on the same training
//! path as `evaluate_at_thresholds` / `export_trained`: bit-exact against the
//! CPU scorer; against the Metal scorer the integer numerator is exact and the
//! score within 1 f32 ULP (Metal's division is fast-math — a pre-existing
//! CPU/GPU difference of the SCORE, see `same_numerators`). Column 0 of the
//! exported scores is bit-identical to what those two entry points return.
//!
//! Sweep: BINARY lattice, BINARY vote (majority and θ = 0.2), TERNARY, QUAD_
//! WEIGHTED, QUAD_BINARY × {single-cluster, two-cluster} × {plain, class-
//! balanced} over a genome with a dense cluster layout, a sparse one, mixed
//! widths inside a cluster and duplicate neurons (the multiplicity layer).
//! Checked against the production scorer (Metal when present) AND the CPU
//! scorer; packed == unpacked; a row subset == those rows of the full export.

use crate::adaptive::{compute_per_example_scores, train_genome_export, NeuronOutputs};
use crate::ids_cache::{
	evaluate_at_thresholds_ids_cached, export_neuron_outputs_ids_cached, export_trained_ids_cached, IDSCache,
};
use ram_core::neuron_memory::{BINARY, QSR, QUAD_BINARY, QUAD_WEIGHTED, TERNARY};
use ram_core::packed_bits::PackedBits;
use ram_core::vote_share::VoteShare;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 40;
const EMPTY_VALUE: f32 = 0.5;

fn split(n: usize, rng: &mut SmallRng) -> (PackedBits, Vec<i64>)
{
	let bools: Vec<bool> = (0..n * TIB).map(|_| rng.gen_bool(0.5)).collect();
	// Label = a noisy function of the first bits, so cells disagree and both classes fill.
	let labels = (0..n).map(|r| ((bools[r * TIB] ^ bools[r * TIB + 3]) ^ rng.gen_bool(0.15)) as i64).collect();
	(PackedBits::from_bool_slice(&bools, TIB), labels)
}

fn cache(single_cluster: bool, balance: bool, mode: u8, vote: Option<VoteShare>) -> IDSCache
{
	let mut rng = SmallRng::seed_from_u64(0x57E);
	let (train, eval, val) = (split(900, &mut rng), split(300, &mut rng), split(250, &mut rng));
	let mut cache = IDSCache::new(
		train.0, train.1, eval.0, eval.1, Some(val.0), Some(val.1), 2, TIB, 5, 1, 11, balance, single_cluster, false,
		1.0,
	);
	cache.memory_mode = mode;
	cache.write_vote = vote;
	cache
}

/// Cluster shapes: dense 8-bit, sparse 20-bit with two narrower (16-bit)
/// neurons mixed in. Neuron 1 of every cluster duplicates neuron 0.
fn genome(clusters: usize) -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let mut rng = SmallRng::seed_from_u64(0xE57);
	let shapes: [Vec<usize>; 2] = [vec![8; 7], [vec![20; 6], vec![16; 2]].concat()];
	let (mut bits, mut npc, mut conns) = (Vec::new(), Vec::new(), Vec::new());
	for c in 0..clusters
	{
		let shape = &shapes[if clusters == 1 { 1 } else { c % 2 }];
		let mut first: Vec<i64> = Vec::new();
		for (n, &b) in shape.iter().enumerate()
		{
			let own: Vec<i64> = (0..b).map(|_| rng.gen_range(0..TIB) as i64).collect();
			let own = if n == 1 && shape[0] == b { first.clone() } else { own };
			if n == 0
			{
				first = own.clone();
			}
			conns.extend(own);
			bits.push(b);
		}
		npc.push(shape.len());
	}
	(bits, npc, conns)
}

fn bits_of(v: &[f64]) -> Vec<u64>
{
	v.iter().map(|x| x.to_bits()).collect()
}

/// Bit-exact equality with a readable failure (first mismatch + count).
fn same(got: &[f64], want: &[f64], what: &str)
{
	assert_eq!(got.len(), want.len(), "{what}: length");
	let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i].to_bits() != want[i].to_bits()).collect();
	assert!(bad.is_empty(), "{what}: {}/{} differ; first at {} got {} want {}", bad.len(), got.len(), bad[0], got[bad[0]], want[bad[0]]);
}

fn export(c: &IDSCache, g: &(Vec<usize>, Vec<usize>, Vec<i64>), part: &str, rows: Option<&[usize]>, packed: bool)
	-> NeuronOutputs
{
	export_neuron_outputs_ids_cached(c, &g.0, &g.1, &g.2, EMPTY_VALUE, 1.0, 0, part, rows, packed).expect("export")
}

/// Against the PRODUCTION scorer (Metal when present). Its float division is
/// fast-math (not IEEE-rounded: 6/7 differs from the CPU's by 1 ULP), so the
/// pin is the exact part — the integer numerator: the cells' weight sum in
/// quarters equals round(score · n · 4) — plus |Δ| <= 1 f32 ULP on the score.
fn same_numerators(out: &NeuronOutputs, what: &str)
{
	let combined = out.cluster_scores().expect("deterministic mode");
	let k = out.neurons_per_cluster.len();
	for (i, (&got, &want)) in combined.iter().zip(&out.scores).enumerate()
	{
		let quarters = out.neurons_per_cluster[i % k] as f64 * 4.0;
		assert_eq!((got * quarters).round(), (want * quarters).round(), "{what}: numerator differs at {i} ({got} vs {want})");
		assert!((got - want).abs() <= f32::EPSILON as f64, "{what}: score {i} off by more than 1 ULP ({got} vs {want})");
	}
}

/// One configuration: every partition reproduces the scorer bit-exactly.
fn check(single_cluster: bool, balance: bool, mode: u8, vote: Option<VoteShare>, what: &str)
{
	let c = cache(single_cluster, balance, mode, vote);
	let g = genome(c.num_genome_clusters());
	for part in ["train", "val", "test"]
	{
		let out = export(&c, &g, part, None, false);
		let total: usize = g.1.iter().sum();
		assert_eq!(out.cells.len(), out.num_rows * total, "{what} {part}: shape");
		assert_eq!(out.scores.len(), out.num_rows * g.1.len(), "{what} {part}: K scores per row");
		assert!(out.cells.iter().any(|&v| v != out.cells[0]), "{what} {part}: constant cells (vacuous)");
		same_numerators(&out, &format!("{what} {part}"));
		let packed = export(&c, &g, part, None, true);
		assert_eq!(packed.cells.len(), out.num_rows * total.div_ceil(4), "{what} {part}: packed shape");
		for r in (0..out.num_rows).step_by(7)
		{
			for j in 0..total
			{
				assert_eq!(packed.cell(r, j), out.cell(r, j), "{what} {part}: packed != unpacked at ({r}, {j})");
			}
		}
		same(&packed.cluster_scores().unwrap(), &out.cluster_scores().unwrap(), &format!("{what} {part}: packed scores"));
	}
	check_reported_path(&c, &g, what);
	check_cpu_scorer(&c, &g, what);
	check_row_subset(&c, &g, what);
}

/// Column 0 is the score `evaluate_at_thresholds` / `export_trained` report.
fn check_reported_path(c: &IDSCache, g: &(Vec<usize>, Vec<usize>, Vec<i64>), what: &str)
{
	let k = g.1.len();
	let col0 = |o: &NeuronOutputs| -> Vec<u64> { o.scores.iter().step_by(k).map(|x| x.to_bits()).collect() };
	let layer = export_trained_ids_cached(c, &g.0, &g.1, &g.2, EMPTY_VALUE, 1.0, 0);
	assert_eq!(col0(&export(c, g, "test", None, false)), bits_of(&layer.eval_scores), "{what}: export_trained eval");
	assert_eq!(col0(&export(c, g, "val", None, false)), bits_of(&layer.val_scores.unwrap()), "{what}: export_trained val");
	if k == 1
	{
		let (eval, train, val, _, _) = evaluate_at_thresholds_ids_cached(c, &g.0, &g.1, &g.2, &[0.5], EMPTY_VALUE, 1.0, 0);
		assert_eq!(col0(&export(c, g, "test", None, false)), bits_of(&eval), "{what}: evaluate_at_thresholds eval");
		assert_eq!(col0(&export(c, g, "train", None, false)), bits_of(&train), "{what}: evaluate_at_thresholds train");
		assert_eq!(col0(&export(c, g, "val", None, false)), bits_of(&val.unwrap()), "{what}: evaluate_at_thresholds val");
	}
}

/// The same cells against the CPU scorer (no Metal evaluators).
fn check_cpu_scorer(c: &IDSCache, g: &(Vec<usize>, Vec<usize>, Vec<i64>), what: &str)
{
	let (train, eval, settings) = (c.full_train(), c.full_eval(), c.eval_settings(EMPTY_VALUE));
	let k = c.num_genome_clusters();
	let e = train_genome_export(
		&g.0, &g.1, &g.2, k, &train.input_bits, &train.targets, &train.negatives, train.num_examples,
		c.num_negatives(), TIB, settings, 1.0, 0, c.class_weights(),
	);
	let cpu: Vec<f64> = compute_per_example_scores(
		&e, &eval.input_bits, eval.num_examples, k, TIB, EMPTY_VALUE, settings.memory_mode, settings.run_seed,
		settings.coverage_aware, None, None,
	)
	.into_iter()
	.flatten()
	.collect();
	let out = export(c, g, "test", None, false);
	same(&out.cluster_scores().unwrap(), &cpu, &format!("{what}: CPU scorer"));
}

fn check_row_subset(c: &IDSCache, g: &(Vec<usize>, Vec<usize>, Vec<i64>), what: &str)
{
	let full = export(c, g, "train", None, false);
	let rows = [5usize, 0, 899, 5, 412];
	let sub = export(c, g, "train", Some(&rows), false);
	let (total, k): (usize, usize) = (g.1.iter().sum(), g.1.len());
	for (i, &r) in rows.iter().enumerate()
	{
		assert_eq!(&sub.cells[i * total..(i + 1) * total], &full.cells[r * total..(r + 1) * total], "{what}: row {r} cells");
		same(&sub.scores[i * k..(i + 1) * k], &full.scores[r * k..(r + 1) * k], &format!("{what}: row {r} scores"));
		assert_eq!(sub.labels[i], full.labels[r], "{what}: row {r} label");
	}
}

#[test]
fn neuron_outputs_reproduce_the_scores_bit_exactly()
{
	let theta = VoteShare::from_theta(0.2).unwrap();
	let rules: [(u8, Option<VoteShare>, &str); 6] = [
		(BINARY, None, "BINARY lattice"),
		(BINARY, Some(VoteShare::MAJORITY), "BINARY vote"),
		(BINARY, Some(theta), "BINARY vote θ=0.2"),
		(TERNARY, None, "TERNARY lattice"),
		(QUAD_WEIGHTED, None, "QUAD_WEIGHTED"),
		(QUAD_BINARY, None, "QUAD_BINARY"),
	];
	for (mode, vote, name) in rules
	{
		for single_cluster in [true, false]
		{
			for balance in [false, true]
			{
				check(single_cluster, balance, mode, vote, &format!("{name} single={single_cluster} balance={balance}"));
			}
		}
	}
}

/// Coverage-aware scoring (a miss / dense hole weighs 0): still exact.
#[test]
fn neuron_outputs_coverage_aware()
{
	for mode in [BINARY, QUAD_WEIGHTED]
	{
		let mut c = cache(false, false, mode, None);
		c.coverage_aware = true;
		let g = genome(2);
		let out = export(&c, &g, "test", None, false);
		same_numerators(&out, &format!("coverage-aware mode {mode}"));
		check_cpu_scorer(&c, &g, &format!("coverage-aware mode {mode}"));
	}
}

#[test]
fn neuron_outputs_refusals()
{
	let c = cache(true, false, BINARY, None);
	let g = genome(1);
	let call = |part: &str, rows: Option<&[usize]>| export_neuron_outputs_ids_cached(&c, &g.0, &g.1, &g.2, EMPTY_VALUE, 1.0, 0, part, rows, false);
	assert!(call("eval", None).is_err(), "unknown partition");
	assert!(call("test", Some(&[300])).is_err(), "row out of range");
	let mut two_way = cache(true, false, BINARY, None);
	two_way.memory_mode = QSR;
	let q = export_neuron_outputs_ids_cached(&two_way, &g.0, &g.1, &g.2, EMPTY_VALUE, 1.0, 0, "test", Some(&[1]), false);
	assert!(q.is_err(), "QSR row subset");
	let full = export_neuron_outputs_ids_cached(&two_way, &g.0, &g.1, &g.2, EMPTY_VALUE, 1.0, 0, "test", None, false).unwrap();
	assert!(full.cell_weights.is_none() && full.cluster_scores().is_none(), "QSR has no fixed cell weights");
}
