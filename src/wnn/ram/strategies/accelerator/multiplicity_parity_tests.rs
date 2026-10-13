//! IDS-37 stage 2 parity sweep: the multiplicity layer (duplicate neurons
//! trained once, `GenomeExport::from_physical`) vs the plain layer (every copy
//! trained in its own slot region), bit for bit.
//!
//! Per memory mode, routes: batched G = 1 (sparse and dense single-cluster,
//! K = 3 multi-group whose logical groups split across physical groups), the
//! G = 3 per-genome loop, and the neuron-chunked over-budget trainer.
//! Assertions: per-logical-neuron storage equal; `compute_per_example_scores`
//! `to_bits` equal on CPU and GPU scorers, coverage-aware on and off;
//! `grouping.is_identity()` exactly when not selected; selected exports store
//! fewer physical keys. Plus the CPU slot-trainer route through
//! `from_physical`, and rate < 1.0 never selecting the layer.
//!
//! Run twice — the OI modes are only selected under the OI trainer:
//!   WNN_ORDER_INDEPENDENT_TRAIN=1 cargo test -p ram_accelerator --lib --no-default-features multiplicity
//!   cargo test -p ram_accelerator --lib --no-default-features multiplicity

use crate::adaptive::{
	build_groups, build_neuron_metadata, compute_per_example_scores, export_genome_for_gpu, per_cluster_max_bits,
	reorganize_connections_for_gpu, train_genome_in_slot, GenomeExport, GroupMemory,
};
use crate::marker_train::batched_path::{
	batched_train_offspring, batched_train_offspring_plain, multiplicity_selected, one_export,
	train_single_genome_neuron_chunked,
};
use crate::multiplicity_batch::MultiplicityBatch;
use ram_core::neuron_memory::{pack_packed_to_u64, BINARY, PLN, QSR, QUAD_BINARY, QUAD_WEIGHTED, TERNARY};
use ram_core::packed_bits::PackedBits;
use ram_core::physical_genome::PhysicalGenome;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 24;
const EMPTY_VALUE: f32 = 0.5;
const RUN_SEED: u64 = 7;
const ALL_MODES: [u8; 6] = [QUAD_WEIGHTED, QUAD_BINARY, QSR, TERNARY, BINARY, PLN];

/// Training + eval data for `num_clusters` classes (1 = binary IDS: 0/1 labels).
struct Data
{
	train: PackedBits,
	targets: Vec<i64>,
	negatives: Vec<i64>,
	num_negatives: usize,
	eval: PackedBits,
	num_clusters: usize,
}

impl Data
{
	fn new(num_clusters: usize, rng: &mut SmallRng) -> Self
	{
		let (num_train, num_eval) = (400usize, 160usize);
		let random_rows = |n: usize, rng: &mut SmallRng| {
			let bools: Vec<bool> = (0..n * TIB).map(|_| rng.gen_bool(0.5)).collect();
			PackedBits::from_bool_slice(&bools, TIB)
		};
		let train = random_rows(num_train, rng);
		let eval = random_rows(num_eval, rng);
		let classes = num_clusters.max(2) as i64;
		let targets: Vec<i64> = (0..num_train).map(|_| rng.gen_range(0..classes)).collect();
		let num_negatives = if num_clusters > 1 { num_clusters - 1 } else { 0 };
		let negatives = exhaustive_negatives(&targets, num_clusters);
		Self { train, targets, negatives, num_negatives, eval, num_clusters }
	}

	fn num_train(&self) -> usize
	{
		self.targets.len()
	}
}

fn exhaustive_negatives(targets: &[i64], num_clusters: usize) -> Vec<i64>
{
	if num_clusters < 2
	{
		return Vec::new();
	}
	targets
		.iter()
		.flat_map(|&t| (0..num_clusters as i64).filter(move |&c| c != t))
		.collect()
}

/// A flat genome batch (the factory's layout).
struct Batch
{
	bits: Vec<usize>,
	npc: Vec<usize>,
	conns: Vec<i64>,
	num_genomes: usize,
}

/// One genome built from distinct base neurons and a copy plan: `plan[c]` lists
/// base-neuron indices of cluster c in logical order (repeats = duplicates).
fn genome(bases: &[(usize, Vec<i64>)], plan: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let mut out = (Vec::new(), Vec::new(), Vec::new());
	for cluster in plan
	{
		out.1.push(cluster.len());
		for &b in cluster
		{
			out.0.push(bases[b].0);
			out.2.extend_from_slice(&bases[b].1);
		}
	}
	out
}

fn bases(bits: &[usize], rng: &mut SmallRng) -> Vec<(usize, Vec<i64>)>
{
	bits.iter().map(|&b| (b, (0..b).map(|_| rng.gen_range(0..TIB as i64)).collect())).collect()
}

/// Copies k = 1..5 of five bases, interleaved round-robin.
fn interleaved_plan() -> Vec<usize>
{
	let copies = [1usize, 2, 3, 4, 5];
	(0..5).flat_map(|r| (0..5).filter(move |&b| copies[b] > r)).collect()
}

fn batch_of(genomes: Vec<(Vec<usize>, Vec<usize>, Vec<i64>)>) -> Batch
{
	let num_genomes = genomes.len();
	let mut b = Batch { bits: Vec::new(), npc: Vec::new(), conns: Vec::new(), num_genomes };
	for (bits, npc, conns) in genomes
	{
		b.bits.extend(bits);
		b.npc.extend(npc);
		b.conns.extend(conns);
	}
	b
}

/// Single cluster of five base neurons (bits per base given), copies k = 1..5
/// interleaved. Max bits 14 -> one SPARSE group; max 10 -> one DENSE group
/// (coverage bitmap). Mixed bits exercise the heterogeneous (-1 prefix-pad) path.
fn single_cluster(base_bits: [usize; 5], rng: &mut SmallRng) -> Batch
{
	let b = bases(&base_bits, rng);
	batch_of(vec![genome(&b, &[interleaved_plan()])])
}

/// K = 3: c0, c1 = 4n x 14b share ONE logical group but collapse to 2 and 3
/// physical neurons (two physical groups); c2 = 5n, dense. `hetero` mixes
/// bits within c1 and c2.
fn multi_cluster(hetero: bool, rng: &mut SmallRng) -> Batch
{
	let (b13, b8) = if hetero { (13, 8) } else { (14, 10) };
	let b = bases(&[14, 14, 14, b13, 14, 10, b8, 10], rng);
	batch_of(vec![genome(&b, &[vec![0, 1, 0, 0], vec![2, 3, 4, 2], vec![5, 6, 5, 7, 6]])])
}

/// The CPU scorers (`compute_address_packed_bytes`) do not skip the -1
/// prefix-pad connections of a heterogeneous-bits group the way the GPU
/// kernels do (pre-existing; out-of-bounds read). Score those on GPU only.
fn cpu_scorable(batch: &Batch) -> bool
{
	let mut start = 0usize;
	for n in &batch.npc
	{
		let bits = &batch.bits[start..start + n];
		if bits.iter().any(|&b| b != bits[0])
		{
			return false;
		}
		start += n;
	}
	true
}

/// G = 3 uniform-shape genomes (same bits / npc), different duplicate
/// structure; genome 2 has none (identity inside a collapsing batch).
fn three_genomes(rng: &mut SmallRng) -> Batch
{
	let b = bases(&[14, 14, 14, 14, 14, 14], rng);
	batch_of(vec![
		genome(&b, &[vec![0, 1, 0, 2, 1, 0]]),
		genome(&b, &[vec![3, 3, 3, 3, 4, 3]]),
		genome(&b, &[vec![0, 1, 2, 3, 4, 5]]),
	])
}

fn metal_available() -> bool
{
	crate::marker_train::batched_path::trainer_available()
}

/// Logical (bits, npc, conns) of genome g in a batch.
fn logical_genome(batch: &Batch, num_clusters: usize, g: usize) -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let npc_all: Vec<&[usize]> = batch.npc.chunks(num_clusters).collect();
	let n0: usize = npc_all[..g].iter().map(|n| n.iter().sum::<usize>()).sum();
	let n: usize = npc_all[g].iter().sum();
	let c0: usize = batch.bits[..n0].iter().sum();
	let bits = batch.bits[n0..n0 + n].to_vec();
	let c: usize = bits.iter().sum();
	(bits, npc_all[g].to_vec(), batch.conns[c0..c0 + c].to_vec())
}

// ---------------------------------------------------------------- comparisons

/// Plain GPU training is itself run-to-run reproducible only for an
/// order-independent writer: the legacy clamped nudge (QUAD/QSR without OI,
/// and PLN always) races under GPU scheduling, so two PLAIN runs differ. For
/// those modes only the selection (identity grouping) is asserted.
fn deterministic(mode: u8) -> bool
{
	ram_core::neuron_memory::training_is_order_independent(mode)
}

/// Per-logical-neuron storage: same layout, same connections, every slot's
/// sparse range / dense row / coverage bits equal.
fn assert_storage_equal(a: &GenomeExport, b: &GenomeExport, what: &str)
{
	assert_eq!(a.connections, b.connections, "{what}: connections");
	assert_eq!(a.group_info, b.group_info, "{what}: group_info");
	assert_eq!(a.groups.len(), b.groups.len(), "{what}: groups");
	assert_eq!(a.dense_exports, b.dense_exports, "{what}: dense rows");
	assert_eq!(a.dense_coverage, b.dense_coverage, "{what}: dense coverage");
	assert_eq!(a.sparse_exports.len(), b.sparse_exports.len(), "{what}: sparse groups");
	for (s, (sa, sb)) in a.sparse_exports.iter().zip(&b.sparse_exports).enumerate()
	{
		assert_eq!(sa.num_neurons, sb.num_neurons, "{what}: sparse group {s} neurons");
		for n in 0..sa.num_neurons
		{
			assert_eq!(sa.neuron_range(n), sb.neuron_range(n), "{what}: sparse group {s} neuron {n}");
		}
	}
}

/// Scores bitwise equal on the CPU and GPU scorers, coverage-aware on and off.
fn assert_scores_equal(a: &GenomeExport, b: &GenomeExport, data: &Data, mode: u8, cpu: bool, what: &str)
{
	let dense = crate::metal_ramlm::MetalRAMLMEvaluator::new().ok();
	let sparse = ram_core::metal_sparse::MetalSparseEvaluator::new().ok();
	assert!(cpu || (dense.is_some() && sparse.is_some()), "{what}: GPU-only fixture needs both Metal scorers");
	for gpu in [false, true].into_iter().filter(|&g| g || cpu)
	{
		for coverage_aware in [false, true]
		{
			let score = |e: &GenomeExport| scores(e, data, mode, coverage_aware, gpu, &dense, &sparse);
			let (sa, sb) = (score(a), score(b));
			assert!(sa == sb, "{what}: scores differ (gpu={gpu}, coverage_aware={coverage_aware})");
		}
	}
}

fn scores(
	e: &GenomeExport,
	data: &Data,
	mode: u8,
	coverage_aware: bool,
	gpu: bool,
	dense: &Option<crate::metal_ramlm::MetalRAMLMEvaluator>,
	sparse: &Option<ram_core::metal_sparse::MetalSparseEvaluator>,
) -> Vec<u64>
{
	let (packed, wpe) = pack_packed_to_u64(&data.eval);
	let num_eval = data.eval.num_rows();
	let rows = compute_per_example_scores(
		e,
		&data.eval,
		&packed,
		wpe,
		num_eval,
		data.num_clusters,
		TIB,
		EMPTY_VALUE,
		mode,
		RUN_SEED,
		coverage_aware,
		if gpu { dense.as_ref() } else { None },
		if gpu { sparse.as_ref() } else { None },
	);
	rows.iter().flatten().map(|s| s.to_bits()).collect()
}

/// Selection is visible on the export: a collapsing grouping exactly when the
/// layer ran AND the genome has duplicates; then fewer physical keys.
fn assert_selection(plain: &GenomeExport, mult: &GenomeExport, logical: &(Vec<usize>, Vec<usize>, Vec<i64>), selected: bool)
{
	let phys = PhysicalGenome::collapse(&logical.0, &logical.1, &logical.2);
	let expect_collapse = selected && !phys.grouping.is_identity();
	assert!(plain.grouping.is_identity(), "plain export must carry the identity grouping");
	assert_eq!(!mult.grouping.is_identity(), expect_collapse, "grouping identity iff not selected");
	assert_footprint(plain, mult, &phys, expect_collapse);
	if expect_collapse
	{
		assert_eq!(mult.grouping, phys.grouping);
		let keys = |e: &GenomeExport| e.sparse_exports.iter().map(|s| s.total_entries()).sum::<usize>();
		assert!(keys(plain) == 0 || keys(mult) < keys(plain), "physical keys must shrink");
	}
}

/// Footprint (stage 3): the plain export is plain; the multiplicity export
/// reports the genome's logical neurons and its distinct units as physical;
/// for an all-sparse genome physical cells are exactly the stored keys (each
/// unit's range held once).
fn assert_footprint(plain: &GenomeExport, mult: &GenomeExport, phys: &PhysicalGenome, collapsed: bool)
{
	let (fp, fm) = (plain.footprint(), mult.footprint());
	assert!(fp.is_plain() && fp.logical_cells == fp.physical_cells, "plain footprint must be plain: {fp:?}");
	assert_eq!(fm.logical_neurons, fp.logical_neurons, "logical neurons");
	let expect_physical = if collapsed { phys.num_neurons() } else { phys.grouping.num_units() };
	assert_eq!(fm.physical_neurons as usize, expect_physical, "physical neurons");
	if mult.dense_exports.is_empty()
	{
		let stored: usize = mult.sparse_exports.iter().map(|s| s.keys.len()).sum();
		assert_eq!(fm.physical_cells as usize, stored, "physical cells = stored keys");
	}
	assert!(fm.physical_cells <= fm.logical_cells, "{fm:?}");
}

// ---------------------------------------------------------------- routes

fn train_plain_batched(batch: &Batch, data: &Data, mode: u8, rate: f32) -> Vec<GenomeExport>
{
	batched_train_offspring_plain(
		&batch.bits, &batch.npc, &batch.conns, batch.num_genomes, data.num_clusters, &data.train, &data.targets,
		&data.negatives, data.num_train(), data.num_negatives, TIB, EMPTY_VALUE, rate, 42, None, mode,
	)
	.expect("plain train")
}

fn train_factory(batch: &Batch, data: &Data, mode: u8, rate: f32) -> Vec<GenomeExport>
{
	batched_train_offspring(
		&batch.bits, &batch.npc, &batch.conns, batch.num_genomes, data.num_clusters, &data.train, &data.targets,
		&data.negatives, data.num_train(), data.num_negatives, TIB, EMPTY_VALUE, rate, 42, None, mode,
	)
	.expect("factory train")
}

/// Factory (plain or multiplicity, as it selects) vs the plain layer.
fn check_batched(batch: &Batch, data: &Data, mode: u8, what: &str)
{
	let plain = train_plain_batched(batch, data, mode, 1.0);
	let mult = train_factory(batch, data, mode, 1.0);
	let selected = multiplicity_selected(1.0, mode, &batch.conns);
	assert_eq!((plain.len(), mult.len()), (batch.num_genomes, batch.num_genomes));
	for g in 0..batch.num_genomes
	{
		let what = format!("{what} mode={mode} genome={g}");
		assert_selection(&plain[g], &mult[g], &logical_genome(batch, data.num_clusters, g), selected);
		if !deterministic(mode)
		{
			continue;
		}
		assert_storage_equal(&plain[g], &mult[g], &what);
		assert_eq!(plain[g].footprint().logical_cells, mult[g].footprint().logical_cells, "{what}: logical cells");
		assert_scores_equal(&plain[g], &mult[g], data, mode, cpu_scorable(batch), &what);
	}
}

fn chunked(bits: &[usize], conns: &[i64], data: &Data, mode: u8, budget: usize) -> GenomeExport
{
	one_export(
		train_single_genome_neuron_chunked(
			bits, conns, &data.train, &data.targets, &data.negatives, data.num_train(), data.num_negatives, TIB,
			EMPTY_VALUE, 1.0, 42, None, budget, mode,
		)
		.expect("chunked train"),
	)
	.expect("one export")
}

/// The over-budget route: the physical genome trained through the
/// neuron-chunked trainer (2 neurons per chunk) vs the logical one.
fn check_chunked(batch: &Batch, data: &Data, mode: u8)
{
	if !multiplicity_selected(1.0, mode, &batch.conns)
	{
		return;
	}
	let cap = crate::marker_train::genome_path::marker_capacity_for_train(data.num_train(), 14, 1.0);
	let budget = 2 * cap * 16;
	let plain = chunked(&batch.bits, &batch.conns, data, mode, budget);
	let mb = MultiplicityBatch::new(&batch.bits, &batch.npc, &batch.conns, 1, 1, mode);
	let mult = mb.train(|p| Ok(chunked(&p.bits, &p.connections, data, mode, budget))).expect("mult chunked");
	let what = format!("chunked mode={mode}");
	assert_selection(&plain, &mult[0], &logical_genome(batch, 1, 0), true);
	assert_storage_equal(&plain, &mult[0], &what);
	assert_scores_equal(&plain, &mult[0], data, mode, cpu_scorable(batch), &what);
}

fn run_parity(mode: u8)
{
	if !metal_available()
	{
		eprintln!("[multiplicity] no Metal device — skipping");
		return;
	}
	let mut rng = SmallRng::seed_from_u64(0x1D5_0037 ^ mode as u64);
	let binary = Data::new(1, &mut rng);
	let sparse_hetero = single_cluster([14, 10, 14, 12, 14], &mut rng);
	let sparse_uniform = single_cluster([14; 5], &mut rng);
	check_batched(&sparse_hetero, &binary, mode, "single sparse hetero");
	check_batched(&sparse_uniform, &binary, mode, "single sparse uniform");
	check_batched(&single_cluster([10, 6, 10, 8, 10], &mut rng), &binary, mode, "single dense hetero");
	check_batched(&single_cluster([10; 5], &mut rng), &binary, mode, "single dense uniform");
	check_batched(&three_genomes(&mut rng), &binary, mode, "G=3");
	let multi = Data::new(3, &mut rng);
	check_batched(&multi_cluster(true, &mut rng), &multi, mode, "K=3 multigroup hetero");
	check_batched(&multi_cluster(false, &mut rng), &multi, mode, "K=3 multigroup uniform");
	check_chunked(&sparse_hetero, &binary, mode);
	check_chunked(&sparse_uniform, &binary, mode);
}

#[test]
fn multiplicity_parity_quad_weighted()
{
	run_parity(QUAD_WEIGHTED);
}

#[test]
fn multiplicity_parity_quad_binary()
{
	run_parity(QUAD_BINARY);
}

#[test]
fn multiplicity_parity_qsr()
{
	run_parity(QSR);
}

#[test]
fn multiplicity_parity_ternary()
{
	run_parity(TERNARY);
}

#[test]
fn multiplicity_parity_binary()
{
	run_parity(BINARY);
}

#[test]
fn multiplicity_parity_pln()
{
	run_parity(PLN);
}

/// TERNARY / BINARY always selected; OI modes exactly when the OI trainer is
/// on; PLN never (its IDS writer is the order-dependent clamped nudge).
#[test]
fn multiplicity_selection_rule()
{
	let conns = [1i64];
	let oi = ram_core::neuron_memory::order_independent_training_enabled();
	for mode in ALL_MODES
	{
		let expected = match mode
		{
			TERNARY | BINARY => true,
			PLN => false,
			_ => oi,
		};
		assert_eq!(multiplicity_selected(1.0, mode, &conns), expected, "mode {mode}");
		assert!(!multiplicity_selected(1.0, mode, &[]), "random connections never select (mode {mode})");
		for rate in [0.0f32, 0.25, 0.5, 0.999]
		{
			assert!(!multiplicity_selected(rate, mode, &conns), "rate {rate} selected mode {mode}");
		}
	}
	assert!(!multiplicity_selected(1.0, 99, &conns), "unknown mode never selects");
}

/// Below rate 1.0 the factory output is the plain layer, byte for byte.
#[test]
fn multiplicity_never_below_rate_one()
{
	if !metal_available()
	{
		eprintln!("[multiplicity] no Metal device — skipping");
		return;
	}
	let mut rng = SmallRng::seed_from_u64(0x0025);
	let data = Data::new(1, &mut rng);
	let batch = single_cluster([14, 10, 14, 12, 14], &mut rng);
	for mode in ALL_MODES
	{
		for rate in [0.25f32, 0.999]
		{
			let plain = train_plain_batched(&batch, &data, mode, rate);
			let factory = train_factory(&batch, &data, mode, rate);
			assert!(factory[0].grouping.is_identity(), "rate {rate} mode {mode} selected the layer");
			if !deterministic(mode)
			{
				continue;
			}
			assert_storage_equal(&plain[0], &factory[0], &format!("rate {rate} mode {mode}"));
		}
	}
}

// ---------------------------------------------------------------- CPU route

/// Plain CPU slot trainer (sequential) + export, as the multigroup reference.
fn cpu_export(bits: &[usize], npc: &[usize], conns: &[i64], data: &Data, mode: u8) -> GenomeExport
{
	let groups = build_groups(&per_cluster_max_bits(bits, npc), npc);
	let (cluster_starts, conn_offsets) = build_neuron_metadata(bits, npc);
	let mut cluster_to_group = vec![(0usize, 0usize); npc.len()];
	for (gi, gr) in groups.iter().enumerate()
	{
		for (li, &cid) in gr.cluster_ids.iter().enumerate()
		{
			cluster_to_group[cid] = (gi, li);
		}
	}
	let mut memories: Vec<GroupMemory> =
		groups.iter().map(|gr| GroupMemory::new(gr.total_neurons(), gr.bits, mode)).collect();
	train_genome_in_slot(
		&mut memories, &groups, conns, bits, &cluster_starts, &conn_offsets, &cluster_to_group, &data.train,
		&data.targets, &data.negatives, data.num_train(), data.num_negatives, TIB, None, 1.0, 42, mode, None, false,
	);
	export_genome_for_gpu(&memories, &groups, &reorganize_connections_for_gpu(conns, bits, npc, &groups))
}

/// CPU only: the sequential slot trainer writes copies identically in EVERY
/// mode (one writer per neuron, example order fixed), so `from_physical` over
/// the physical genome's CPU export must reproduce the logical CPU export —
/// sparse ranges aliased, dense rows copied, untracked coverage kept empty.
#[test]
fn from_physical_aliases_and_copies()
{
	let mut rng = SmallRng::seed_from_u64(0xF00D);
	let cases: Vec<(Batch, Data)> = vec![
		(single_cluster([14, 10, 14, 12, 14], &mut rng), Data::new(1, &mut rng)),
		(single_cluster([14; 5], &mut rng), Data::new(1, &mut rng)),
		(single_cluster([10, 6, 10, 8, 10], &mut rng), Data::new(1, &mut rng)),
		(single_cluster([10; 5], &mut rng), Data::new(1, &mut rng)),
		(multi_cluster(true, &mut rng), Data::new(3, &mut rng)),
		(multi_cluster(false, &mut rng), Data::new(3, &mut rng)),
	];
	for mode in ALL_MODES
	{
		for (i, (batch, data)) in cases.iter().enumerate()
		{
			let p = PhysicalGenome::collapse(&batch.bits, &batch.npc, &batch.conns);
			assert!(!p.grouping.is_identity(), "fixture {i} must carry duplicates");
			let logical = cpu_export(&batch.bits, &batch.npc, &batch.conns, data, mode);
			let physical = cpu_export(&p.bits, &p.neurons_per_cluster, &p.connections, data, mode);
			let rebuilt = GenomeExport::from_physical(physical, p.grouping.clone(), &batch.bits, &batch.npc, &batch.conns, mode)
				.expect("from_physical");
			assert_storage_equal(&logical, &rebuilt, &format!("cpu fixture {i} mode {mode}"));
			let cpu = cpu_scorable(batch);
			assert_scores_equal(&logical, &rebuilt, data, mode, cpu, &format!("cpu fixture {i} mode {mode}"));
		}
	}
}
