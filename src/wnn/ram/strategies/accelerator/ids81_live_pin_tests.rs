//! IDS-81 swap pin: the LIVE IDS configurations score bit-identically on the
//! pre-IDS-81 wheel and on this branch, so the worker swap changes no
//! in-flight cohort's lineage.
//!
//! Each case mirrors a queued/restarting flow's params (read-only from the DB
//! 27/11): memory mode, write rule (+ vote share), cluster layout,
//! balance_classes, neuron_sample_rate and its genome width range, on a small
//! synthetic dataset with a val partition. Every search route (each k-fold, the
//! 5-fold mean, full train->test) and the validation route (single-cluster
//! threshold scores / K-cluster decode modes) is digested; the eval-in-place
//! probe kernel (6433/6438: QUAD_WEIGHTED, OI, WNN_EVAL_IN_PLACE=1) is digested
//! directly, since the over-budget regime that routes to it needs 46M rows.
//! The test prints one `PIN <case> <digest>` line per case: run it on both
//! trees with the live env and diff the lines.
//!   WNN_ORDER_INDEPENDENT_TRAIN=1 WNN_EVAL_IN_PLACE=1 cargo test -p ram_accelerator \
//!     --lib --no-default-features ids81_live_pin -- --ignored --nocapture --test-threads=1

use crate::ids_cache::{
	evaluate_at_thresholds_ids_cached, evaluate_genomes_ids_cached_full_hybrid, evaluate_genomes_ids_kfold_hybrid,
	evaluate_genomes_ids_kfold_mean_hybrid, evaluate_multiclass_at_thresholds_ids_cached, IDSCache,
};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 96;
const PARTS: usize = 5;
const SEED: u64 = 20421;
const POP: usize = 3;
/// Connections draw from the first inputs, where the label signal lives.
const SIGNAL_INPUTS: usize = 48;

struct Split
{
	rows: PackedBits,
	labels: Vec<i64>,
}

fn split(n: usize, rng: &mut SmallRng) -> Split
{
	let mut bools = Vec::with_capacity(n * TIB);
	let mut labels = Vec::with_capacity(n);
	for _ in 0..n
	{
		let row: Vec<bool> = (0..TIB).map(|_| rng.gen_bool(0.5)).collect();
		let signal = row[0] && row[7] || row[13] && row[29] && !row[41];
		labels.push(if rng.gen_bool(0.85) { signal } else { rng.gen_bool(0.3) } as i64);
		bools.extend(row);
	}
	Split { rows: PackedBits::from_bool_slice(&bools, TIB), labels }
}

struct Data
{
	train: Split,
	eval: Split,
	val: Split,
}

fn data() -> Data
{
	let mut rng = SmallRng::seed_from_u64(0x1D581);
	Data { train: split(6000, &mut rng), eval: split(1500, &mut rng), val: split(1000, &mut rng) }
}

/// One live flow's knobs.
struct Live
{
	name: &'static str,
	mode: u8,
	vote_theta: Option<f64>,
	single_cluster: bool,
	balance: bool,
	rate: f32,
	min_bits: usize,
	max_bits: usize,
	neurons: usize,
	fitness: (f32, f32, f32, f32),
}

fn live_cases() -> Vec<Live>
{
	let (desir_80, ce20) = ((0.1, 0.35, 0.35, 0.2), (0.2, 0.3, 0.4, 0.1));
	vec![
		// 7637-7641: unswr BINARY vote 0.02, two-cluster, fixed 80 bits, rate 0.25.
		Live { name: "7638-7641_binary_vote002_b80", mode: BINARY, vote_theta: Some(0.02), single_cluster: false, balance: false, rate: 0.25, min_bits: 80, max_bits: 80, neurons: 12, fitness: desir_80 },
		// 7549: cicids BINARY vote 0.02, two-cluster, bits 4..34 (dense + sparse).
		Live { name: "7549_binary_vote002_b4-34", mode: BINARY, vote_theta: Some(0.02), single_cluster: false, balance: false, rate: 0.25, min_bits: 4, max_bits: 34, neurons: 12, fitness: ce20 },
		// 7535: cicids BINARY "lattice" (= the BINARY Set writer), balanced, bits 4..34.
		Live { name: "7535_binary_set_b4-34_bal", mode: BINARY, vote_theta: None, single_cluster: false, balance: true, rate: 0.25, min_bits: 4, max_bits: 34, neurons: 12, fitness: ce20 },
		// 6433 / 6438: ciciot46m QUAD_WEIGHTED (default) OI, single cluster, balanced.
		Live { name: "6433_quad_oi_single_r025", mode: QUAD_WEIGHTED, vote_theta: None, single_cluster: true, balance: true, rate: 0.25, min_bits: 4, max_bits: 100, neurons: 12, fitness: (0.175, 0.15, 0.15, 0.525) },
		Live { name: "6438_quad_oi_single_r1", mode: QUAD_WEIGHTED, vote_theta: None, single_cluster: true, balance: true, rate: 1.0, min_bits: 4, max_bits: 100, neurons: 12, fitness: (0.175, 0.15, 0.15, 0.525) },
	]
}

impl Live
{
	fn clusters(&self) -> usize
	{
		if self.single_cluster { 1 } else { 2 }
	}

	/// POP genomes; neuron widths spread over [min_bits, max_bits], plus one
	/// duplicated neuron per genome (exercises the multiplicity layer at rate 1.0).
	fn population(&self) -> (Vec<usize>, Vec<usize>, Vec<i64>)
	{
		let mut rng = SmallRng::seed_from_u64(SEED ^ self.max_bits as u64);
		let (mut bits, mut npc, mut conns) = (Vec::new(), Vec::new(), Vec::new());
		for _ in 0..POP * self.clusters()
		{
			npc.push(self.neurons);
			let mut neuron_bits = Vec::new();
			for n in 0..self.neurons
			{
				let b = if n == self.neurons - 1 { neuron_bits[0] } else { self.width(n, &mut rng) };
				neuron_bits.push(b);
			}
			let first: Vec<i64> = (0..neuron_bits[0]).map(|_| rng.gen_range(0..SIGNAL_INPUTS) as i64).collect();
			for (n, &b) in neuron_bits.iter().enumerate()
			{
				let c: Vec<i64> = if n == self.neurons - 1 { first.clone() } else { (0..b).map(|_| rng.gen_range(0..SIGNAL_INPUTS) as i64).collect() };
				conns.extend(if n == 0 { first.clone() } else { c });
			}
			bits.extend(neuron_bits);
		}
		(bits, npc, conns)
	}

	/// Mostly narrow widths (shared addresses, so the write rules and class
	/// balance actually bite) plus every 4th neuron anywhere in the range
	/// (sparse groups, the live genomes' wide end).
	fn width(&self, n: usize, rng: &mut SmallRng) -> usize
	{
		let narrow_top = self.max_bits.min(self.min_bits + 6);
		if n % 4 == 3 { rng.gen_range(self.min_bits..=self.max_bits) } else { rng.gen_range(self.min_bits..=narrow_top) }
	}

	fn cache(&self, d: &Data) -> IDSCache
	{
		let mut cache = IDSCache::new(
			d.train.rows.clone(), d.train.labels.clone(), d.eval.rows.clone(), d.eval.labels.clone(),
			Some(d.val.rows.clone()), Some(d.val.labels.clone()), 2, TIB, PARTS, 1, SEED, self.balance,
			self.single_cluster, false, 1.0,
		);
		cache.memory_mode = self.mode;
		cache.fitness_weights = Some(self.fitness);
		cache.write_vote = self.vote_theta.map(|t| ram_core::vote_share::VoteShare::from_theta(t).expect("theta"));
		cache
	}
}

/// FNV-1a over u64 words.
struct Digest(u64);

impl Digest
{
	fn push(&mut self, w: u64)
	{
		for b in w.to_le_bytes()
		{
			self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
		}
	}

	fn rows(&mut self, rows: &[(f64, f64, f64, f64, f64, u32)])
	{
		// .5 is wall-clock ms — never digested.
		for r in rows
		{
			for x in [r.0, r.1, r.2, r.3, r.4]
			{
				self.push(x.to_bits());
			}
		}
	}

	fn floats(&mut self, xs: &[f64])
	{
		xs.iter().for_each(|x| self.push(x.to_bits()));
	}

	fn text(&mut self, s: &str)
	{
		s.bytes().for_each(|b| self.push(b as u64));
	}
}

fn search_digest(c: &Live, cache: &IDSCache, d: &mut Digest)
{
	let (bits, npc, conns) = c.population();
	for fold in 0..PARTS
	{
		let rows = evaluate_genomes_ids_kfold_hybrid(cache, &bits, &npc, &conns, POP, fold, 0.5, c.rate, 7);
		if fold == 0
		{
			eprintln!("PIN-SAMPLE {} fold0 {:?}", c.name, &rows[0]);
		}
		d.rows(&rows);
	}
	let folds: Vec<usize> = (0..PARTS).collect();
	d.rows(&evaluate_genomes_ids_kfold_mean_hybrid(cache, &bits, &npc, &conns, POP, &folds, 0.5, c.rate, 7));
	d.rows(&evaluate_genomes_ids_cached_full_hybrid(cache, &bits, &npc, &conns, POP, 0.5, c.rate, 7, None));
}

fn validation_digest(c: &Live, cache: &IDSCache, d: &mut Digest)
{
	let (bits, npc, conns) = c.population();
	let one = |v: &[usize]| v[..v.len() / POP].to_vec();
	let (b1, n1) = (bits[..npc[..c.clusters()].iter().sum::<usize>()].to_vec(), one(&npc));
	let c1: Vec<i64> = conns[..b1.iter().sum::<usize>()].to_vec();
	if c.single_cluster
	{
		let (e, t, v, metrics, _fp) = evaluate_at_thresholds_ids_cached(cache, &b1, &n1, &c1, &[0.5, -1.0], 0.5, c.rate, 7);
		d.floats(&e);
		d.floats(&t);
		d.floats(&v.expect("val"));
		metrics.iter().for_each(|m| d.floats(&[m.0, m.1, m.2, m.3, m.4]));
		return;
	}
	let (_k, modes, _fp) = evaluate_multiclass_at_thresholds_ids_cached(cache, &b1, &n1, &c1, 0.5, c.rate, 7);
	d.text(&format!("{modes:?}"));
}

/// The eval-in-place probe (6433/6438's live route at 46M rows) on one genome.
fn probe_digest(c: &Live, data: &Data, d: &mut Digest)
{
	if !ram_core::neuron_memory::order_independent_training_enabled()
	{
		return;
	}
	let (bits, npc, conns) = c.population();
	let (b1, c1) = (&bits[..npc[0]], &conns[..bits[..npc[0]].iter().sum::<usize>()]);
	let v = crate::marker_train::train_single_genome_chunked_scored_probe(
		b1, c1, &data.train.rows, &data.train.labels, &[], data.train.labels.len(), 0, TIB, 0.5, c.rate, 7, None,
		&data.eval.rows, data.eval.labels.len(), Some(&data.train.rows), c.mode,
	)
	.expect("probe");
	v.eval_scores.iter().for_each(|x| d.push(x.to_bits() as u64));
	v.train_scores.expect("train scores").iter().for_each(|x| d.push(x.to_bits() as u64));
}

#[test]
#[ignore = "IDS-81 swap pin: run on both trees with the live env and diff the PIN lines"]
fn ids81_live_pin()
{
	let data = data();
	eprintln!(
		"PIN env oi={} eval_in_place={:?}",
		ram_core::neuron_memory::order_independent_training_enabled(),
		std::env::var("WNN_EVAL_IN_PLACE").ok()
	);
	let mut digests = std::collections::HashMap::new();
	for c in live_cases()
	{
		let cache = c.cache(&data);
		let mut d = Digest(0xcbf2_9ce4_8422_2325);
		search_digest(&c, &cache, &mut d);
		validation_digest(&c, &cache, &mut d);
		let searched = d.0;
		if c.mode == QUAD_WEIGHTED
		{
			probe_digest(&c, &data, &mut d);
		}
		eprintln!("PIN {} search+validation={:016x} with_probe={:016x}", c.name, searched, d.0);
		digests.insert(c.name, d.0);
	}
	assert_ne!(digests["7535_binary_set_b4-34_bal"], sensitivity_control(&data), "the digest must see the write rule");
}

/// NOT a live flow: 7535's knobs under the plain MAJORITY vote. Its digest must
/// differ from 7535's, or the harness would be blind to write-rule changes.
fn sensitivity_control(data: &Data) -> u64
{
	let mut c = live_cases().remove(2);
	c.vote_theta = Some(0.5);
	let cache = c.cache(data);
	let mut d = Digest(0xcbf2_9ce4_8422_2325);
	search_digest(&c, &cache, &mut d);
	validation_digest(&c, &cache, &mut d);
	eprintln!("PIN control_binary_vote_majority search+validation={:016x}", d.0);
	d.0
}
