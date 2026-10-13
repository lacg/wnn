//! PLN writer regression (12/10/2026). PLN = TERNARY's cells (FALSE=0 / TRUE=1
//! / u=EMPTY=2) with a fair-coin read of u. Until 12/10/2026 both IDS trainers
//! wrote PLN through the clamped QUAD nudge — CPU `use_nudge = mode != TERNARY`,
//! GPU `marker_train.metal` with no PLN branch — so a FALSE visit on EMPTY(2)
//! left 1 (= PLN TRUE) and a TRUE visit left 3 (read as u): inverted training,
//! and order-dependent. Both trainers now dispatch on `CellMode::write_rule`
//! (GPU twin `wnn_write_rule`): PLN takes TERNARY's TRUE-wins lattice.
//!
//! Pinned here: the cell a visit leaves, PLN storage == TERNARY storage, CPU
//! order independence, and GPU == CPU (storage and CPU/GPU scorers).

use crate::adaptive::{
	build_groups, build_neuron_metadata, compute_per_example_scores, export_genome_for_gpu, per_cluster_max_bits,
	reorganize_connections_for_gpu, train_genome_in_slot, GenomeExport, GroupMemory,
};
use ram_core::neuron_memory::{pack_packed_to_u64, EMPTY, FALSE, PLN, TERNARY, TRUE};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 16;
const RUN_SEED: u64 = 11;

/// Binary-IDS data whose labels CONFLICT on shared addresses (random labels
/// over a 16-bit input space), so the writer's join rule decides the cells.
struct Data
{
	train: PackedBits,
	targets: Vec<i64>,
}

impl Data
{
	fn random(num_train: usize, rng: &mut SmallRng) -> Self
	{
		let bools: Vec<bool> = (0..num_train * TIB).map(|_| rng.gen_bool(0.5)).collect();
		let targets = (0..num_train).map(|_| rng.gen_range(0..2i64)).collect();
		Self { train: PackedBits::from_bool_slice(&bools, TIB), targets }
	}

	/// The same examples in reverse order.
	fn reversed(&self) -> Self
	{
		let n = self.targets.len();
		let row = |i: usize| (0..TIB).map(move |b| (self.train.packed_row(i)[b >> 3] >> (b & 7)) & 1 == 1);
		let bools: Vec<bool> = (0..n).rev().flat_map(row).collect();
		let targets = self.targets.iter().rev().copied().collect();
		Self { train: PackedBits::from_bool_slice(&bools, TIB), targets }
	}
}

/// One single-cluster genome: `bits[i]` per neuron, random connections.
struct Genome
{
	bits: Vec<usize>,
	npc: Vec<usize>,
	conns: Vec<i64>,
}

impl Genome
{
	fn random(bits: &[usize], rng: &mut SmallRng) -> Self
	{
		let conns = bits.iter().flat_map(|&b| (0..b).map(|_| rng.gen_range(0..TIB as i64)).collect::<Vec<_>>()).collect();
		Self { bits: bits.to_vec(), npc: vec![bits.len()], conns }
	}
}

/// The CPU slot trainer (sequential) + export.
fn cpu_train(g: &Genome, data: &Data, mode: u8) -> (Vec<GroupMemory>, GenomeExport)
{
	let groups = build_groups(&per_cluster_max_bits(&g.bits, &g.npc), &g.npc);
	let (cluster_starts, conn_offsets) = build_neuron_metadata(&g.bits, &g.npc);
	let mut memories: Vec<GroupMemory> =
		groups.iter().map(|gr| GroupMemory::new(gr.total_neurons(), gr.bits, mode)).collect();
	let n = data.targets.len();
	train_genome_in_slot(
		&mut memories, &groups, &g.conns, &g.bits, &cluster_starts, &conn_offsets, &[(0, 0)], &data.train, &data.targets,
		&[], n, 0, TIB, None, 1.0, 42, mode, None, false,
	);
	let export = export_genome_for_gpu(&memories, &groups, &reorganize_connections_for_gpu(&g.conns, &g.bits, &g.npc, &groups));
	(memories, export)
}

fn assert_storage_equal(a: &GenomeExport, b: &GenomeExport, what: &str)
{
	assert_eq!(a.connections, b.connections, "{what}: connections");
	assert_eq!(a.dense_exports, b.dense_exports, "{what}: dense rows");
	assert_eq!(a.sparse_exports.len(), b.sparse_exports.len(), "{what}: sparse groups");
	for (sa, sb) in a.sparse_exports.iter().zip(&b.sparse_exports)
	{
		for n in 0..sa.num_neurons
		{
			assert_eq!(sa.neuron_range(n), sb.neuron_range(n), "{what}: sparse neuron {n}");
		}
	}
}

/// One visit on an untouched (EMPTY) cell: FALSE-direction → FALSE(0),
/// TRUE-direction → TRUE(1); a later FALSE never overwrites TRUE.
#[test]
fn pln_visit_leaves_the_lattice_cell()
{
	let g = Genome { bits: vec![4], npc: vec![1], conns: vec![0, 1, 2, 3] };
	let row = |v: u32| (0..TIB).map(move |b| b < 4 && (v >> (3 - b)) & 1 == 1);
	let bools: Vec<bool> = [5u32, 9, 9].into_iter().flat_map(row).collect();
	let data = Data { train: PackedBits::from_bool_slice(&bools, TIB), targets: vec![0, 1, 0] };
	for mode in [PLN, TERNARY]
	{
		let (mem, _) = cpu_train(&g, &data, mode);
		assert_eq!(mem[0].read(0, 5), FALSE, "mode {mode}: FALSE visit on EMPTY");
		assert_eq!(mem[0].read(0, 9), TRUE, "mode {mode}: TRUE then FALSE");
		assert_eq!(mem[0].read(0, 0), EMPTY, "mode {mode}: untouched");
	}
}

fn fixtures(rng: &mut SmallRng) -> Vec<(Genome, Data)>
{
	vec![
		(Genome::random(&[14, 14, 14, 14], rng), Data::random(2000, rng)),
		(Genome::random(&[10, 10, 10, 10], rng), Data::random(2000, rng)),
	]
}

/// PLN storage is TERNARY storage, and independent of example order.
#[test]
fn pln_cpu_trains_as_ternary_in_any_order()
{
	let mut rng = SmallRng::seed_from_u64(0x0915);
	for (i, (g, data)) in fixtures(&mut rng).iter().enumerate()
	{
		let (_, pln) = cpu_train(g, data, PLN);
		let (_, ternary) = cpu_train(g, data, TERNARY);
		let (_, reversed) = cpu_train(g, &data.reversed(), PLN);
		assert_storage_equal(&pln, &ternary, &format!("fixture {i} PLN vs TERNARY"));
		assert_storage_equal(&pln, &reversed, &format!("fixture {i} PLN forward vs reversed"));
	}
}

fn gpu_train(g: &Genome, data: &Data, mode: u8) -> GenomeExport
{
	let n = data.targets.len();
	crate::marker_train::batched_path::batched_train_offspring_plain(
		&g.bits, &g.npc, &g.conns, 1, 1, &data.train, &data.targets, &[], n, 0, TIB, 0.5, 1.0, 42, None, mode,
	)
	.expect("gpu train")
	.remove(0)
}

fn scores(e: &GenomeExport, data: &Data, gpu: bool) -> Vec<u64>
{
	let dense = if gpu { crate::metal_ramlm::MetalRAMLMEvaluator::new().ok() } else { None };
	let sparse = if gpu { ram_core::metal_sparse::MetalSparseEvaluator::new().ok() } else { None };
	let (packed, wpe) = pack_packed_to_u64(&data.train);
	let n = data.targets.len();
	let rows = compute_per_example_scores(
		e, &data.train, &packed, wpe, n, 1, TIB, 0.5, PLN, RUN_SEED, false, dense.as_ref(), sparse.as_ref(),
	);
	rows.iter().flatten().map(|s| s.to_bits()).collect()
}

/// The batched Metal trainer writes PLN exactly as the CPU slot trainer (and as
/// its own TERNARY); PLN coin scores agree on the CPU and GPU scorers.
#[test]
fn pln_gpu_matches_cpu()
{
	if !crate::marker_train::batched_path::trainer_available()
	{
		eprintln!("[pln] no Metal device — skipping");
		return;
	}
	let mut rng = SmallRng::seed_from_u64(0x0916);
	for (i, (g, data)) in fixtures(&mut rng).iter().enumerate()
	{
		let (_, cpu) = cpu_train(g, data, PLN);
		let gpu = gpu_train(g, data, PLN);
		assert_storage_equal(&cpu, &gpu, &format!("fixture {i} PLN CPU vs GPU"));
		assert_storage_equal(&gpu, &gpu_train(g, data, TERNARY), &format!("fixture {i} GPU PLN vs TERNARY"));
		assert_eq!(scores(&cpu, data, false), scores(&gpu, data, true), "fixture {i}: CPU vs GPU PLN scores");
	}
}
