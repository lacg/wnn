//! MAJORITY-VOTE writer (16/10/2026, `ids_write_rule="vote"`, worker ABI 19).
//!
//! Under the default writers a single attack visit pins an address TRUE for
//! good (BINARY one-shot Set; TERNARY/PLN TRUE-wins lattice), so in
//! single-cluster binary IDS any address touched by >= 1 attack reads TRUE —
//! the ~42 % FPR floor on UNSW. `WriteRule::Vote` instead keeps a signed
//! ±class-weight tally per (neuron, address) in the OI packed counter and
//! commits its sign (tie → the mode's untrained cell).
//!
//! Pinned here, for BINARY / TERNARY / PLN, single- and multi-cluster, dense
//! (bits <= SPARSE_THRESHOLD) and sparse:
//!   - the hand case (1 attack + 3 normals → FALSE under vote, TRUE under the
//!     default writer; ties; class weights);
//!   - order independence (reversed and shuffled rows → identical cells) on
//!     the CPU slot trainer and the batched Metal trainer;
//!   - CPU == GPU (storage and scores);
//!   - vote == `train::SparseTrainer`'s signed-vote TERNARY (the public
//!     package's reference semantics).
//! Default-unset bit-identity is pinned by `WriteRule::resolve(m, false) ==
//! of_code(m)` (core) plus every pre-existing parity suite, which now routes
//! through `resolve(.., false)` unchanged.

use crate::adaptive::{
	build_groups, build_neuron_metadata, compute_per_example_scores, export_genome_for_gpu, per_cluster_max_bits,
	reorganize_connections_for_gpu, train_genome_in_slot, GenomeExport, GroupMemory,
};
use ram_core::cell_mode::{CellMode, WriteRule};
use ram_core::neuron_memory::{BINARY, EMPTY, FALSE, PLN, TERNARY, TRUE};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const TIB: usize = 16;
const RUN_SEED: u64 = 7;
const VOTE_MODES: [u8; 3] = [BINARY, TERNARY, PLN];

/// Labelled rows + (multi-cluster) exhaustive negatives.
struct Data
{
	train: PackedBits,
	targets: Vec<i64>,
	negatives: Vec<i64>,
	num_clusters: usize,
}

impl Data
{
	/// Random rows over `TIB` bits; `num_clusters == 1` is binary IDS (labels
	/// 0/1, no negatives), otherwise labels 0..K with all other classes negative.
	fn random(num_train: usize, num_clusters: usize, rng: &mut SmallRng) -> Self
	{
		let bools: Vec<bool> = (0..num_train * TIB).map(|_| rng.gen_bool(0.5)).collect();
		let classes = num_clusters.max(2) as i64;
		let targets: Vec<i64> = (0..num_train).map(|_| rng.gen_range(0..classes)).collect();
		Self::from_rows(PackedBits::from_bool_slice(&bools, TIB), targets, num_clusters)
	}

	fn from_rows(train: PackedBits, targets: Vec<i64>, num_clusters: usize) -> Self
	{
		let negatives = if num_clusters == 1
		{
			Vec::new()
		}
		else
		{
			targets.iter().flat_map(|&t| (0..num_clusters as i64).filter(move |&c| c != t)).collect()
		};
		Self { train, targets, negatives, num_clusters }
	}

	fn num_negatives(&self) -> usize
	{
		if self.num_clusters == 1 { 0 } else { self.num_clusters - 1 }
	}

	fn row(&self, i: usize) -> impl Iterator<Item = bool> + '_
	{
		(0..TIB).map(move |b| (self.train.packed_row(i)[b >> 3] >> (b & 7)) & 1 == 1)
	}

	/// The same examples in `order`.
	fn permuted(&self, order: &[usize]) -> Self
	{
		let bools: Vec<bool> = order.iter().flat_map(|&i| self.row(i)).collect();
		let targets = order.iter().map(|&i| self.targets[i]).collect();
		Self::from_rows(PackedBits::from_bool_slice(&bools, TIB), targets, self.num_clusters)
	}

	fn reversed(&self) -> Self
	{
		let order: Vec<usize> = (0..self.targets.len()).rev().collect();
		self.permuted(&order)
	}

	fn shuffled(&self, rng: &mut SmallRng) -> Self
	{
		let mut order: Vec<usize> = (0..self.targets.len()).collect();
		order.shuffle(rng);
		self.permuted(&order)
	}
}

/// One genome: `bits[i]` per neuron, `npc` neurons per cluster, random connections.
struct Genome
{
	bits: Vec<usize>,
	npc: Vec<usize>,
	conns: Vec<i64>,
}

impl Genome
{
	fn random(npc: &[usize], bits: &[usize], rng: &mut SmallRng) -> Self
	{
		let conns = bits.iter().flat_map(|&b| (0..b).map(|_| rng.gen_range(0..TIB as i64)).collect::<Vec<_>>()).collect();
		Self { bits: bits.to_vec(), npc: npc.to_vec(), conns }
	}

	/// One neuron reading input bits 0..b (address = the row's top b bits).
	fn identity(b: usize) -> Self
	{
		Self { bits: vec![b], npc: vec![1], conns: (0..b as i64).collect() }
	}
}

/// The CPU slot trainer (sequential) + export, under `rule`.
fn cpu_train(g: &Genome, data: &Data, mode: u8, rule: WriteRule, cw: Option<&[u32]>)
	-> (Vec<GroupMemory>, GenomeExport)
{
	let groups = build_groups(&per_cluster_max_bits(&g.bits, &g.npc), &g.npc);
	let (cluster_starts, conn_offsets) = build_neuron_metadata(&g.bits, &g.npc);
	let mut cluster_to_group = vec![(0usize, 0usize); g.npc.len()];
	for (gi, gr) in groups.iter().enumerate()
	{
		for (li, &cid) in gr.cluster_ids.iter().enumerate()
		{
			cluster_to_group[cid] = (gi, li);
		}
	}
	let mut memories: Vec<GroupMemory> =
		groups.iter().map(|gr| GroupMemory::new(gr.total_neurons(), gr.bits, mode)).collect();
	let n = data.targets.len();
	train_genome_in_slot(
		&mut memories, &groups, &g.conns, &g.bits, &cluster_starts, &conn_offsets, &cluster_to_group, &data.train,
		&data.targets, &data.negatives, n, data.num_negatives(), TIB, None, 1.0, 42, mode, rule, cw, false,
	);
	let export = export_genome_for_gpu(&memories, &groups, &reorganize_connections_for_gpu(&g.conns, &g.bits, &g.npc, &groups));
	(memories, export)
}

/// The batched Metal trainer under `rule` (None without a Metal device).
fn gpu_train(g: &Genome, data: &Data, mode: u8, rule: WriteRule, cw: Option<&[u32]>) -> Option<GenomeExport>
{
	if !crate::marker_train::batched_path::trainer_available()
	{
		return None;
	}
	let n = data.targets.len();
	let out = crate::marker_train::batched_path::batched_train_offspring_plain(
		&g.bits, &g.npc, &g.conns, 1, g.npc.len(), &data.train, &data.targets, &data.negatives, n,
		data.num_negatives(), TIB, 0.5, 1.0, 42, cw, mode, rule,
	);
	Some(out.expect("gpu train").remove(0))
}

/// Every cell the export reads, (group, neuron, address) order — dense and
/// sparse alike, through the sparse-miss default (so a dropped default entry
/// and a stored one compare equal, exactly as every scorer sees them).
fn cells(e: &GenomeExport, mode: u8) -> Vec<i64>
{
	let miss = CellMode::from_u8(mode).unwrap().default_cell();
	let mut out = Vec::new();
	for (gi, group) in e.groups.iter().enumerate()
	{
		for n in 0..group.total_neurons()
		{
			for a in 0..(1u64 << group.bits)
			{
				out.push(e.read_cell_at(gi, n, a, miss));
			}
		}
	}
	out
}

fn scores(e: &GenomeExport, data: &Data, mode: u8, gpu: bool) -> Vec<u64>
{
	let dense = if gpu { crate::metal_ramlm::MetalRAMLMEvaluator::new().ok() } else { None };
	let sparse = if gpu { ram_core::metal_sparse::MetalSparseEvaluator::new().ok() } else { None };
	let n = data.targets.len();
	// PLN scores through its deterministic twin (TERNARY at 0.5) so the
	// comparison is exact and independent of the coin.
	let score_mode = if mode == PLN { TERNARY } else { mode };
	let rows = compute_per_example_scores(
		e, &data.train, n, g_clusters(e), TIB, 0.5, score_mode, RUN_SEED, false, dense.as_ref(), sparse.as_ref(),
	);
	rows.iter().flatten().map(|s| s.to_bits()).collect()
}

fn g_clusters(e: &GenomeExport) -> usize
{
	e.groups.iter().map(|g| g.cluster_ids.len()).sum()
}

/// Rows whose top 4 bits are `addr`, with labels.
fn rows_at(addrs_labels: &[(u32, i64)], num_clusters: usize) -> Data
{
	let row = |v: u32| (0..TIB).map(move |b| b < 4 && (v >> (3 - b)) & 1 == 1);
	let bools: Vec<bool> = addrs_labels.iter().flat_map(|&(a, _)| row(a)).collect();
	let targets = addrs_labels.iter().map(|&(_, y)| y).collect();
	Data::from_rows(PackedBits::from_bool_slice(&bools, TIB), targets, num_clusters)
}

fn vote(mode: u8) -> WriteRule
{
	WriteRule::resolve(mode, true)
}

/// 1 attack + 3 normals on address 5: TRUE under the default writer (one
/// attack pins it), FALSE under vote. Address 9: 2 + 2 → tie → the mode's
/// untrained cell (EMPTY; BINARY FALSE). Address 3: 3 attacks + 1 normal →
/// TRUE either way. Address 0 untouched. CPU and GPU.
#[test]
fn vote_hand_case_one_attack_three_normals()
{
	let data = rows_at(
		&[(5, 1), (5, 0), (5, 0), (5, 0), (9, 1), (9, 0), (9, 1), (9, 0), (3, 1), (3, 0), (3, 1), (3, 1)],
		1,
	);
	let g = Genome::identity(4);
	for mode in VOTE_MODES
	{
		let untrained = CellMode::from_u8(mode).unwrap().default_cell() as i64;
		let (lat, _) = cpu_train(&g, &data, mode, WriteRule::of_code(mode), None);
		assert_eq!(lat[0].read(0, 5), TRUE, "mode {mode}: default writer, one attack pins TRUE");
		let (mem, cexp) = cpu_train(&g, &data, mode, vote(mode), None);
		assert_eq!(mem[0].read(0, 5), FALSE, "mode {mode}: vote 1 vs 3 → FALSE");
		assert_eq!(mem[0].read(0, 9), untrained, "mode {mode}: vote tie → untrained");
		assert_eq!(mem[0].read(0, 3), TRUE, "mode {mode}: vote 3 vs 1 → TRUE");
		assert_eq!(mem[0].read(0, 0), untrained, "mode {mode}: untouched");
		if let Some(gexp) = gpu_train(&g, &data, mode, vote(mode), None)
		{
			assert_eq!(cells(&gexp, mode), cells(&cexp, mode), "mode {mode}: GPU vs CPU hand case");
			assert_eq!(gexp.read_cell_at(0, 0, 5, untrained as u8), FALSE, "mode {mode}: GPU vote → FALSE");
		}
	}
	// Ternary/PLN: the tie really is EMPTY, not FALSE.
	let (mem, _) = cpu_train(&g, &data, TERNARY, vote(TERNARY), None);
	assert_eq!(mem[0].read(0, 9), EMPTY);
}

/// balance_classes weights scale the vote like QUAD's OI path: with attack
/// weight 3, 1 attack (+3) beats 2 normals (−2).
#[test]
fn vote_honours_class_weights()
{
	let data = rows_at(&[(5, 1), (5, 0), (5, 0)], 1);
	let g = Genome::identity(4);
	let cw = [1u32, 3];
	for mode in VOTE_MODES
	{
		let (mem, cexp) = cpu_train(&g, &data, mode, vote(mode), Some(&cw));
		assert_eq!(mem[0].read(0, 5), TRUE, "mode {mode}: +3 vs −2");
		let (unweighted, _) = cpu_train(&g, &data, mode, vote(mode), None);
		assert_eq!(unweighted[0].read(0, 5), FALSE, "mode {mode}: +1 vs −2");
		if let Some(gexp) = gpu_train(&g, &data, mode, vote(mode), Some(&cw))
		{
			assert_eq!(cells(&gexp, mode), cells(&cexp, mode), "mode {mode}: weighted GPU vs CPU");
		}
	}
}

/// Multi-cluster: an example votes + into its own cluster's neurons and − into
/// every negative cluster's. Address 5 sees 1 × class 0 and 2 × class 1:
/// cluster 0 → +1 −2 = FALSE, cluster 1 → +2 −1 = TRUE (the default lattice
/// leaves BOTH TRUE). BINARY under vote writes negatives too.
#[test]
fn vote_multi_cluster_own_plus_negatives_minus()
{
	let data = rows_at(&[(5, 0), (5, 1), (5, 1)], 2);
	let g = Genome { bits: vec![4, 4], npc: vec![1, 1], conns: vec![0, 1, 2, 3, 0, 1, 2, 3] };
	for mode in VOTE_MODES
	{
		let (mem, cexp) = cpu_train(&g, &data, mode, vote(mode), None);
		// One group; cluster c's first neuron sits at group-local c × stride
		// (stride = the group's neuron slot count: 1, or the coalesced bucket
		// under WNN_COALESCE_GROUPS=1).
		let c1 = cexp.groups[0].neurons;
		assert_eq!(mem[0].read(0, 5), FALSE, "mode {mode}: cluster 0 outvoted");
		assert_eq!(mem[0].read(c1, 5), TRUE, "mode {mode}: cluster 1 wins");
		if mode != BINARY
		{
			let (lat, _) = cpu_train(&g, &data, mode, WriteRule::of_code(mode), None);
			assert_eq!((lat[0].read(0, 5), lat[0].read(c1, 5)), (TRUE, TRUE), "mode {mode}: lattice");
		}
		if let Some(gexp) = gpu_train(&g, &data, mode, vote(mode), None)
		{
			assert_eq!(cells(&gexp, mode), cells(&cexp, mode), "mode {mode}: multi-cluster GPU vs CPU");
		}
	}
}

/// Dense (<= 12 bits), sparse (> 12), mixed-bits; single- and multi-cluster.
fn fixtures(rng: &mut SmallRng) -> Vec<(Genome, Data)>
{
	vec![
		(Genome::random(&[4], &[10, 10, 10, 10], rng), Data::random(2000, 1, rng)),
		(Genome::random(&[4], &[14, 14, 14, 14], rng), Data::random(2000, 1, rng)),
		(Genome::random(&[4], &[14, 14, 13, 14], rng), Data::random(2000, 1, rng)),
		(Genome::random(&[4], &[10, 10, 8, 10], rng), Data::random(2000, 1, rng)),
		(Genome::random(&[2, 2, 3], &[8, 8, 8, 8, 10, 10, 10], rng), Data::random(1500, 3, rng)),
		(Genome::random(&[2, 2], &[14, 14, 14, 14], rng), Data::random(1500, 2, rng)),
	]
}

/// CPU: reversed and shuffled rows leave identical cells (integer tallies).
#[test]
fn vote_cpu_is_order_independent()
{
	let mut rng = SmallRng::seed_from_u64(0x1016);
	for (i, (g, data)) in fixtures(&mut rng).iter().enumerate()
	{
		for mode in VOTE_MODES
		{
			let (_, a) = cpu_train(g, data, mode, vote(mode), Some(&[1, 2, 3]));
			let (_, b) = cpu_train(g, &data.reversed(), mode, vote(mode), Some(&[1, 2, 3]));
			let (_, c) = cpu_train(g, &data.shuffled(&mut rng), mode, vote(mode), Some(&[1, 2, 3]));
			assert_eq!(cells(&a, mode), cells(&b, mode), "fixture {i} mode {mode}: reversed");
			assert_eq!(cells(&a, mode), cells(&c, mode), "fixture {i} mode {mode}: shuffled");
		}
	}
}

/// Vote must actually differ from the default writer on conflicting data
/// (else the parity tests below would pass vacuously).
#[test]
fn vote_differs_from_default_writer_on_conflicts()
{
	let mut rng = SmallRng::seed_from_u64(0x1017);
	for (i, (g, data)) in fixtures(&mut rng).iter().enumerate()
	{
		for mode in VOTE_MODES
		{
			let (_, v) = cpu_train(g, data, mode, vote(mode), None);
			let (_, d) = cpu_train(g, data, mode, WriteRule::of_code(mode), None);
			assert_ne!(cells(&v, mode), cells(&d, mode), "fixture {i} mode {mode}: vote == default?");
		}
	}
}

/// Metal == CPU (cells and scores) for every mode/fixture, and the GPU result
/// is independent of row order (high-z duplicate slots merged by oi_merge).
#[test]
fn vote_gpu_matches_cpu_and_is_order_independent()
{
	if !crate::marker_train::batched_path::trainer_available()
	{
		eprintln!("[vote] no Metal device — skipping");
		return;
	}
	let mut rng = SmallRng::seed_from_u64(0x1018);
	for (i, (g, data)) in fixtures(&mut rng).iter().enumerate()
	{
		for mode in VOTE_MODES
		{
			let cw: &[u32] = &[1, 2, 3];
			let (_, cpu) = cpu_train(g, data, mode, vote(mode), Some(cw));
			let gpu = gpu_train(g, data, mode, vote(mode), Some(cw)).unwrap();
			assert_eq!(cells(&cpu, mode), cells(&gpu, mode), "fixture {i} mode {mode}: CPU vs GPU cells");
			// Trainer parity is EXACT: both exports, the same scorer.
			assert_eq!(scores(&cpu, data, mode, false), scores(&gpu, data, mode, false), "fixture {i} mode {mode}: CPU scorer");
			assert_eq!(scores(&cpu, data, mode, true), scores(&gpu, data, mode, true), "fixture {i} mode {mode}: GPU scorer");
			// Across scorers: the Metal scorer divides with fast math, so a
			// non-dyadic normalisation (e.g. 2.5 / 3 neurons) may sit one f32 ulp
			// off the CPU's IEEE division — a scorer property, not the trainer.
			for (x, y) in scores(&cpu, data, mode, false).iter().zip(scores(&gpu, data, mode, true))
			{
				assert!((f64::from_bits(*x) - f64::from_bits(y)).abs() < 1e-6, "fixture {i} mode {mode}: CPU vs GPU scorer");
			}
			let gpu_shuffled = gpu_train(g, &data.shuffled(&mut rng), mode, vote(mode), Some(cw)).unwrap();
			assert_eq!(cells(&gpu, mode), cells(&gpu_shuffled, mode), "fixture {i} mode {mode}: GPU shuffled");
		}
	}
}

/// The IDS vote writer and the public package's `SparseTrainer` (signed-vote
/// TERNARY: sign → TRUE/FALSE, tie → EMPTY) leave identical cells for the same
/// multi-cluster data, weights and connections.
#[test]
fn vote_matches_sparse_trainer_reference()
{
	use ram_core::forward::ClusterLayout;
	use ram_core::train::{new_memory, SparseTrainer};
	let mut rng = SmallRng::seed_from_u64(0x1019);
	let (k, n, bits) = (3usize, 4usize, 10usize);
	let g = Genome::random(&vec![n; k], &vec![bits; k * n], &mut rng);
	let data = Data::random(1500, k, &mut rng);
	let cw = [2u32, 1, 3];
	let weights: Vec<u32> = data.targets.iter().map(|&y| cw[y as usize]).collect();
	for mode in [TERNARY, PLN]
	{
		let cm = CellMode::from_u8(mode).unwrap();
		let layout = ClusterLayout { num_clusters: k, neurons_per_cluster: n, bits_per_neuron: bits };
		let t = SparseTrainer::new(cm, layout);
		t.accumulate(&data.train, &g.conns, &data.targets, Some(&weights)).unwrap();
		let reference = new_memory(cm, layout);
		t.commit(&reference);
		let (_, ids) = cpu_train(&g, &data, mode, vote(mode), Some(&cw));
		assert_eq!(ids.groups.len(), 1);
		let stride = ids.groups[0].neurons; // n, or the coalesced bucket
		for neuron in 0..k * n
		{
			let local = (neuron / n) * stride + neuron % n;
			for a in 0..(1u64 << bits)
			{
				assert_eq!(
					ids.read_cell_at(0, local, a, cm.default_cell()),
					reference.read_cell_or_default(neuron, a) as i64,
					"mode {mode}: neuron {neuron} address {a}"
				);
			}
		}
	}
}
