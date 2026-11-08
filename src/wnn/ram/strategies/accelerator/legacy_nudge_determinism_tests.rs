//! The legacy clamped nudge (QUAD family WITHOUT the order-independent trainer)
//! must train the same genome to the same cells on every run (08/11/2026).
//!
//! Root cause of the old nondeterminism: the nudge saturates (FALSE..TRUE), so
//! it does not commute — from WEAK_FALSE, +1,+1,+1,-1 ends at WEAK_TRUE while
//! -1,+1,+1,+1 ends at TRUE. The CPU trainer fanned EXAMPLES out over rayon, so
//! each cell received its nudges in work-stealing order; the GPU trainer ran up
//! to 8 threads per neuron over example slices. Fixes: the CPU trainer
//! parallelises such writers over neuron lanes (each lane walks the examples in
//! order), the GPU trainer runs them at z = 1. Separately, the CPU CE reductions
//! summed f64 in rayon's re-association order (ulp-level drift run to run).
//!
//! The fixture forces conflicts: 8-bit neurons over random-label rows, so most
//! cells see both nudge directions. Skipped under WNN_ORDER_INDEPENDENT_TRAIN=1
//! (QUAD then trains OI, which is commutative and covered elsewhere).

use crate::adaptive::{
	build_groups, build_neuron_metadata, compute_per_example_scores, export_genome_for_gpu, per_cluster_max_bits,
	reorganize_connections_for_gpu, train_genome_in_slot, GenomeExport, GroupMemory,
};
use crate::marker_train::batched_path::{batched_train_offspring_plain, trainer_available};
use ram_core::cell_mode::WriteRule;
use ram_core::neuron_memory::{training_is_order_independent, QUAD_WEIGHTED};
use ram_core::packed_bits::PackedBits;

const TIB: usize = 24;
const K: usize = 2;
const NUM_TRAIN: usize = 60_000;
const NPC: [usize; 2] = [12, 12];
const BITS: usize = 8;

struct Data
{
	packed: PackedBits,
	targets: Vec<i64>,
	negatives: Vec<i64>,
}

fn data() -> Data
{
	let mut x = 0x9E37_79B9_7F4A_7C15u64;
	let mut next = move || {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		x
	};
	let bools: Vec<bool> = (0..NUM_TRAIN * TIB).map(|_| next() & 1 == 1).collect();
	let targets: Vec<i64> = (0..NUM_TRAIN).map(|_| (next() % K as u64) as i64).collect();
	let negatives = targets.iter().map(|&t| 1 - t).collect();
	Data { packed: PackedBits::from_bool_slice(&bools, TIB), targets, negatives }
}

fn genome() -> (Vec<usize>, Vec<i64>)
{
	let n: usize = NPC.iter().sum();
	let conns = (0..n).flat_map(|i| (0..BITS).map(move |k| ((5 * i + 3 * k) % TIB) as i64)).collect();
	(vec![BITS; n], conns)
}

fn cpu_export(d: &Data, rate: f32, parallel: bool) -> GenomeExport
{
	let (bits, conns) = genome();
	let groups = build_groups(&per_cluster_max_bits(&bits, &NPC), &NPC);
	let (starts, offsets) = build_neuron_metadata(&bits, &NPC);
	let mut c2g = vec![(0usize, 0usize); K];
	for (gi, g) in groups.iter().enumerate()
	{
		g.cluster_ids.iter().enumerate().for_each(|(li, &c)| c2g[c] = (gi, li));
	}
	let mut mems: Vec<GroupMemory> = groups.iter().map(|g| GroupMemory::new(g.total_neurons(), g.bits, QUAD_WEIGHTED)).collect();
	train_genome_in_slot(
		&mut mems, &groups, &conns, &bits, &starts, &offsets, &c2g, &d.packed, &d.targets, &d.negatives, NUM_TRAIN, 1, TIB, None, rate,
		42, QUAD_WEIGHTED, WriteRule::Nudge, None, parallel,
	);
	export_genome_for_gpu(&mems, &groups, &reorganize_connections_for_gpu(&conns, &bits, &NPC, &groups))
}

fn gpu_export(d: &Data, rate: f32) -> GenomeExport
{
	let (bits, conns) = genome();
	batched_train_offspring_plain(&bits, &NPC, &conns, 1, K, &d.packed, &d.targets, &d.negatives, NUM_TRAIN, 1, TIB, 0.0, rate, 42, None, QUAD_WEIGHTED, WriteRule::Nudge)
		.expect("gpu train")
		.remove(0)
}

fn scores(e: &GenomeExport, d: &Data) -> Vec<Vec<f64>>
{
	compute_per_example_scores(e, &d.packed, NUM_TRAIN, K, TIB, 0.0, QUAD_WEIGHTED, 0, false, None, None)
}

fn legacy_quad_active() -> bool
{
	if training_is_order_independent(QUAD_WEIGHTED)
	{
		eprintln!("[legacy_nudge] OI trainer active — legacy nudge not exercised, skipping");
		return false;
	}
	true
}

#[test]
fn cpu_legacy_nudge_is_deterministic_and_equals_sequential()
{
	if !legacy_quad_active()
	{
		return;
	}
	let d = data();
	for rate in [1.0f32, 0.25]
	{
		let sequential = cpu_export(&d, rate, false);
		for run in 0..3
		{
			let parallel = cpu_export(&d, rate, true);
			assert_eq!(parallel.dense_exports, sequential.dense_exports, "rate {rate} run {run}: cells differ");
			assert_eq!(scores(&parallel, &d), scores(&sequential, &d), "rate {rate} run {run}: scores differ");
		}
	}
}

#[test]
fn cpu_ce_reduction_is_deterministic()
{
	let d = data();
	let s = scores(&cpu_export(&d, 1.0, false), &d);
	let first = crate::adaptive::multiclass_ce_acc(&s, &d.targets);
	for _ in 0..5
	{
		assert_eq!(crate::adaptive::multiclass_ce_acc(&s, &d.targets), first, "CE/acc moved between identical calls");
	}
}

#[test]
fn gpu_legacy_nudge_equals_cpu_sequential()
{
	if !legacy_quad_active() || !trainer_available()
	{
		return;
	}
	let d = data();
	for rate in [1.0f32, 0.25]
	{
		let cpu = scores(&cpu_export(&d, rate, false), &d);
		for run in 0..3
		{
			assert_eq!(scores(&gpu_export(&d, rate), &d), cpu, "rate {rate} run {run}: GPU legacy nudge != CPU sequential");
		}
	}
}
