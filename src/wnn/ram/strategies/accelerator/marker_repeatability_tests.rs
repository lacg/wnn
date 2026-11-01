//! IDS-34: the GPU marker trainer must give BIT-IDENTICAL exports when the
//! same genome is trained repeatedly in one process, at every bit width.
//!
//! Root cause (fixed 01/11/2026): when a neuron's key space fits its slot
//! region (2^bits <= slot_capacity — the capacity is capped at 2 x 2^bits once
//! the train set has more visits than addresses), the hashed find-or-claim
//! still ran with z (up to 1024) threads per neuron. A claimer that out-waits a
//! pending CLAIMED probes forward and publishes a DUPLICATE-key slot; with
//! hundreds of threads per neuron the duplicates filled the 2 x 2^bits region,
//! `find_or_claim_slot` returned table-full and the write was silently DROPPED
//! — which writes depended on GPU scheduling (b = 8 every call, some b = 12).
//! Fix: such neurons address their region DIRECTLY (slot = key): no hashing,
//! no claim race, no duplicates, no table-full.
//!
//! Two guards:
//!   - `repeat_train_bit_identical_low_bits`: b in {4, 8, 12, 16} x every
//!     order-independent writer, 5 trainings each, exports must be equal (and
//!     the probe-eval route must score them identically at b = 8 / 12).
//!   - `slot_layout_golden_wide_bits`: digests of b in {16, 20, 24} exports
//!     recorded on the PRE-FIX trainer — the direct-addressing switch must not
//!     change a single exported cell where the hashed table never overflowed.
//!
//! Run both ways (QUAD is only order-independent under the OI trainer):
//!   WNN_ORDER_INDEPENDENT_TRAIN=1 cargo test -p ram_accelerator --lib --no-default-features marker_repeatability
//!   cargo test -p ram_accelerator --lib --no-default-features marker_repeatability

use crate::adaptive::GenomeExport;
use crate::marker_train::batched_path::batched_train_offspring;
use ram_core::cell_mode::WriteRule;
use ram_core::neuron_memory::{training_is_order_independent, BINARY, PLN, QUAD_WEIGHTED, TERNARY};
use ram_core::packed_bits::PackedBits;

const TIB: usize = 64;

/// Deterministic xorshift64 stream (fixture data + connections).
struct Xs(u64);

impl Xs
{
	fn next(&mut self) -> u64
	{
		self.0 ^= self.0 << 13;
		self.0 ^= self.0 >> 7;
		self.0 ^= self.0 << 17;
		self.0
	}
}

/// Random 64-bit rows; label = bit0 XOR bit3 (learnable, both directions).
/// Multi-cluster: label = (bit0 + 2*bit3) % k, negatives = every other class.
struct Fixture
{
	packed: PackedBits,
	targets: Vec<i64>,
	negatives: Vec<i64>,
	num_train: usize,
	num_negatives: usize,
}

fn fixture(num_train: usize, k: usize) -> Fixture
{
	let mut xs = Xs(0x9E37_79B9_7F4A_7C15);
	let bools: Vec<bool> = (0..num_train * TIB).map(|_| xs.next() & 1 == 1).collect();
	let label = |ex: usize| {
		let (a, b) = (bools[ex * TIB] as usize, bools[ex * TIB + 3] as usize);
		if k == 1 { (a ^ b) as i64 } else { ((a + 2 * b) % k) as i64 }
	};
	let targets: Vec<i64> = (0..num_train).map(label).collect();
	let num_negatives = if k == 1 { 0 } else { k - 1 };
	let negatives: Vec<i64> = if k == 1
	{
		Vec::new()
	}
	else
	{
		targets.iter().flat_map(|&t| (0..k as i64).filter(move |&c| c != t)).collect()
	};
	Fixture { packed: PackedBits::from_bool_slice(&bools, TIB), targets, negatives, num_train, num_negatives }
}

/// One genome shape: per-neuron bits, neurons per cluster, connections.
struct Shape
{
	bits: Vec<usize>,
	npc: Vec<usize>,
	conns: Vec<i64>,
}

fn shape(bits: Vec<usize>, npc: Vec<usize>, seed: u64) -> Shape
{
	let mut xs = Xs(seed | 1);
	let total: usize = bits.iter().sum();
	let conns = (0..total).map(|_| (xs.next() % TIB as u64) as i64).collect();
	Shape { bits, npc, conns }
}

fn train(f: &Fixture, s: &Shape, rate: f32, mode: u8, rule: WriteRule) -> GenomeExport
{
	batched_train_offspring(
		&s.bits, &s.npc, &s.conns, 1, s.npc.len(), &f.packed, &f.targets, &f.negatives, f.num_train, f.num_negatives,
		TIB, 0.5, rate, 42, None, mode, rule,
	)
	.expect("marker train failed")
	.remove(0)
}

/// Every (mode, writer) pair whose GPU training is order-independent in this
/// process: lattice / set / vote always, QUAD under the OI trainer.
fn writers() -> Vec<(u8, WriteRule)>
{
	let mut out: Vec<(u8, WriteRule)> = [TERNARY, PLN, BINARY, QUAD_WEIGHTED]
		.into_iter()
		.filter(|&m| training_is_order_independent(m))
		.map(|m| (m, WriteRule::of_code(m)))
		.collect();
	out.push((BINARY, WriteRule::Vote));
	out
}

fn fnv(h: &mut u64, bytes: &[u8])
{
	for &b in bytes
	{
		*h ^= b as u64;
		*h = h.wrapping_mul(0x0000_0100_0000_01B3);
	}
}

/// Digest of everything a scorer reads from an export.
fn digest(e: &GenomeExport) -> u64
{
	let mut h = 0xCBF2_9CE4_8422_2325u64;
	for c in &e.connections
	{
		fnv(&mut h, &c.to_le_bytes());
	}
	for (sparse, sub, clusters) in &e.group_info
	{
		fnv(&mut h, &[*sparse as u8]);
		fnv(&mut h, &sub.to_le_bytes());
		clusters.iter().for_each(|c| fnv(&mut h, &c.to_le_bytes()));
	}
	for words in &e.dense_exports
	{
		words.iter().for_each(|w| fnv(&mut h, &w.to_le_bytes()));
	}
	for words in &e.dense_coverage
	{
		words.iter().for_each(|w| fnv(&mut h, &w.to_le_bytes()));
	}
	for s in &e.sparse_exports
	{
		s.keys.iter().for_each(|k| fnv(&mut h, &k.to_le_bytes()));
		fnv(&mut h, &s.values);
		s.offsets.iter().for_each(|o| fnv(&mut h, &o.to_le_bytes()));
		s.counts.iter().for_each(|c| fnv(&mut h, &c.to_le_bytes()));
	}
	h
}

fn metal_available() -> bool
{
	crate::marker_train::batched_path::trainer_available()
}

/// b in {4, 8, 12, 16}: five trainings of the same genome must export the
/// same memory. Pre-fix: b = 8 differed on every call (table-full drops).
#[test]
fn marker_repeatability_low_bits()
{
	if !metal_available()
	{
		eprintln!("[marker_repeatability] no Metal device — skipping");
		return;
	}
	let f = fixture(100_000, 1);
	for (n, b) in [(100usize, 4usize), (5, 8), (100, 8), (300, 8), (100, 12), (300, 12), (100, 16)]
	{
		let s = shape(vec![b; n], vec![n], (n * 131 + b) as u64);
		for &rate in &[0.25f32, 1.0]
		{
			for (mode, rule) in writers()
			{
				let reference = digest(&train(&f, &s, rate, mode, rule));
				for rep in 1..5
				{
					let again = digest(&train(&f, &s, rate, mode, rule));
					assert_eq!(reference, again, "n={n} b={b} rate={rate} mode={mode} {rule:?}: rep {rep} differs");
				}
			}
		}
	}
}

/// The golden shapes: wide-bit single- and multi-cluster genomes, uniform and
/// heterogeneous bits, both sample rates.
fn golden_cases() -> Vec<(&'static str, Shape, usize)>
{
	vec![
		("n50b16", shape(vec![16; 50], vec![50], 7), 1),
		("n40b17", shape(vec![17; 40], vec![40], 23), 1),
		("n100b20", shape(vec![20; 100], vec![100], 11), 1),
		("n30b24", shape(vec![24; 30], vec![30], 13), 1),
		("het16-24", shape((0..40).map(|i| 16 + (i % 9)).collect(), vec![40], 17), 1),
		("k3b16", shape(vec![16; 60], vec![20, 20, 20], 19), 3),
	]
}

/// Digest of every golden case x writer x rate, in a fixed order.
fn golden_digest() -> u64
{
	let fixtures = [fixture(60_000, 1), fixture(60_000, 3)];
	let mut h = 0xCBF2_9CE4_8422_2325u64;
	for (name, s, k) in golden_cases()
	{
		let f = &fixtures[(k == 3) as usize];
		for &rate in &[0.25f32, 1.0]
		{
			for (mode, rule) in writers()
			{
				let d = digest(&train(f, &s, rate, mode, rule));
				eprintln!("[marker_golden] {name} rate={rate} mode={mode} {rule:?} digest={d:#x}");
				fnv(&mut h, &d.to_le_bytes());
			}
		}
	}
	h
}

/// Recorded on the PRE-FIX trainer (hashed slots everywhere), 01/11/2026.
const GOLDEN_WITH_OI: u64 = 0xf6b5_8db9_a8c7_211b;
const GOLDEN_WITHOUT_OI: u64 = 0x7ede_e1d9_fbe9_c184;

/// b >= 16 exports are unchanged by the direct-addressing fix.
#[test]
fn marker_repeatability_golden_wide_bits()
{
	if !metal_available()
	{
		eprintln!("[marker_repeatability] no Metal device — skipping");
		return;
	}
	let oi = training_is_order_independent(QUAD_WEIGHTED);
	let got = golden_digest();
	eprintln!("[marker_golden] oi={oi} TOTAL={got:#x}");
	assert_eq!(got, if oi { GOLDEN_WITH_OI } else { GOLDEN_WITHOUT_OI });
}
