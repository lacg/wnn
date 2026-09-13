//! `SparseTrainer` — mode-aware, ORDER-INDEPENDENT training into a
//! `SparseLayerMemory`, the write side of the public `Forward` boundary.
//!
//! One rule per mode, each the canonical one from the research paths:
//!
//! | mode           | accumulate per (neuron, address, example)            | commit                          |
//! |----------------|------------------------------------------------------|---------------------------------|
//! | BINARY         | own-class visit → TRUE; other classes IGNORED        | set (one-shot, saturating)      |
//! | TERNARY / PLN  | vote += w (own class) / −w (other class)             | sign → TRUE / FALSE; 0 → EMPTY  |
//! | QUAD_* / QSR   | OI counter: obs += 1, net ±= w (`oi_apply_nudge`)    | `oi_bin_to_cell`                |
//!
//! "Own class" is the example's label; every OTHER cluster sees the example as
//! a negative (all negatives, none sampled) — exactly `bitwise_ramlm.rs`'s
//! per-cluster loop over `target_bits`. Weights are integer VOTE weights
//! (`max(1)`), as in the QUAD path: they scale `net` (and TERNARY votes) while
//! `obs` still counts examples — so a weight of 2 is not two copies of the row
//! in the QUAD modes. Every accumulator is an integer, so the result cannot
//! depend on example order or thread scheduling: shuffle the rows, get the
//! identical memory (pinned below).
//!
//! The container is `DashMap<u64, _>` per neuron (u64 addresses — the wide
//! hashed address above 64 bits needs them; `ClusterStorage`'s sparse arm
//! keys on u32). The OI RULE itself is not re-implemented: `oi_apply_nudge`,
//! `oi_bin_to_cell`, `OI_INITIAL` are the single source in `neuron_memory.rs`.
//! Folding `ClusterStorage::Sparse` onto this container is the noted follow-up.

use crate::cell_mode::CellMode;
use crate::forward::ClusterLayout;
use crate::neuron_memory::{
	compute_address_packed, oi_apply_nudge, oi_bin_to_cell, pack_packed_to_u64, EMPTY_U8, FALSE_U8, OI_INITIAL,
	TRUE_U8,
};
use crate::packed_bits::PackedBits;
use crate::sparse_memory::SparseLayerMemory;
use dashmap::DashMap;
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;

type FxBuildHasher = BuildHasherDefault<FxHasher>;

enum Accum
{
	/// BINARY: the set of visited own-class addresses per neuron.
	Set(Vec<DashMap<u64, (), FxBuildHasher>>),
	/// TERNARY / PLN: signed integer vote per (neuron, address).
	Votes(Vec<DashMap<u64, i64, FxBuildHasher>>),
	/// QUAD_* / QSR: packed OI counter per (neuron, address).
	Oi(Vec<DashMap<u64, u32, FxBuildHasher>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError
{
	Shape(String),
}

impl std::fmt::Display for TrainError
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		match self
		{
			TrainError::Shape(s) => write!(f, "train: shape error: {s}"),
		}
	}
}

impl std::error::Error for TrainError {}

pub struct SparseTrainer
{
	mode: CellMode,
	layout: ClusterLayout,
	accum: Accum,
}

impl SparseTrainer
{
	pub fn new(mode: CellMode, layout: ClusterLayout) -> Self
	{
		let n = layout.num_neurons();
		fn maps<V>(n: usize) -> Vec<DashMap<u64, V, FxBuildHasher>>
		{
			(0..n).map(|_| DashMap::with_hasher(FxBuildHasher::default())).collect()
		}
		let accum = match mode
		{
			CellMode::Binary => Accum::Set(maps(n)),
			CellMode::Ternary | CellMode::Pln => Accum::Votes(maps(n)),
			CellMode::QuadBinary | CellMode::QuadWeighted | CellMode::Qsr => Accum::Oi(maps(n)),
		};
		Self { mode, layout, accum }
	}

	pub fn mode(&self) -> CellMode
	{
		self.mode
	}

	pub fn layout(&self) -> ClusterLayout
	{
		self.layout
	}

	fn validate(&self, input: &PackedBits, connections: &[i64], labels: &[i64], weights: Option<&[u32]>)
		-> Result<(), TrainError>
	{
		let l = self.layout;
		let shape = |m: String| Err(TrainError::Shape(m));
		if connections.len() != l.num_neurons() * l.bits_per_neuron
		{
			return shape(format!(
				"connections.len()={} but layout needs {}",
				connections.len(),
				l.num_neurons() * l.bits_per_neuron
			));
		}
		let total_bits = input.total_bits() as i64;
		if let Some(&bad) = connections.iter().find(|&&c| c < 0 || c >= total_bits)
		{
			return shape(format!("connection index {bad} outside the input width {total_bits}"));
		}
		if labels.len() != input.num_rows()
		{
			return shape(format!("labels.len()={} but input has {} rows", labels.len(), input.num_rows()));
		}
		if let Some(&bad) = labels.iter().find(|&&y| y < 0 || y as usize >= l.num_clusters)
		{
			return shape(format!("label {bad} outside 0..{}", l.num_clusters));
		}
		if let Some(w) = weights
		{
			if w.len() != labels.len()
			{
				return shape(format!("weights.len()={} but labels.len()={}", w.len(), labels.len()));
			}
		}
		Ok(())
	}

	/// Accumulate one batch. Every example touches every cluster: its own as a
	/// positive, all others as negatives (BINARY ignores the negatives by rule).
	/// `weights` are integer vote weights; `None` = 1 each; 0 is raised to 1.
	pub fn accumulate(
		&self,
		input: &PackedBits,
		connections: &[i64],
		labels: &[i64],
		weights: Option<&[u32]>,
	) -> Result<(), TrainError>
	{
		self.validate(input, connections, labels, weights)?;
		let l = self.layout;
		let (words, wpe) = pack_packed_to_u64(input);
		(0..input.num_rows()).into_par_iter().for_each(|ex| {
			let w = &words[ex * wpe..(ex + 1) * wpe];
			let own = labels[ex] as usize;
			let weight = weights.map_or(1u32, |ws| ws[ex].max(1));
			for c in 0..l.num_clusters
			{
				let target_true = c == own;
				if matches!(self.accum, Accum::Set(_)) && !target_true
				{
					continue; // BINARY: negatives are ignored by rule
				}
				for n in 0..l.neurons_per_cluster
				{
					let neuron = c * l.neurons_per_cluster + n;
					let conns = &connections[neuron * l.bits_per_neuron..(neuron + 1) * l.bits_per_neuron];
					let address = compute_address_packed(w, conns, l.bits_per_neuron) as u64;
					self.touch(neuron, address, target_true, weight);
				}
			}
		});
		Ok(())
	}

	#[inline]
	fn touch(&self, neuron: usize, address: u64, target_true: bool, weight: u32)
	{
		match &self.accum
		{
			Accum::Set(maps) =>
			{
				maps[neuron].insert(address, ());
			}
			Accum::Votes(maps) =>
			{
				let delta = if target_true { weight as i64 } else { -(weight as i64) };
				*maps[neuron].entry(address).or_insert(0) += delta;
			}
			Accum::Oi(maps) =>
			{
				let delta = if target_true { weight as i32 } else { -(weight as i32) };
				let mut e = maps[neuron].entry(address).or_insert(OI_INITIAL);
				*e = oi_apply_nudge(*e, delta);
			}
		}
	}

	/// Bin every accumulator into cells and write them. `memory` must have been
	/// built with `new_with_default(num_neurons, bits, mode.default_cell())` so a
	/// cell that bins to the default is REMOVED rather than stored (the
	/// `write_cell` contract) and the export stays minimal. Writes override
	/// whatever the memory held — a trainer is the whole truth for its batch
	/// set; call `accumulate` again before `commit` to fold more data in.
	pub fn commit(&self, memory: &SparseLayerMemory)
	{
		match &self.accum
		{
			Accum::Set(maps) => maps.par_iter().enumerate().for_each(|(neuron, map)| {
				for entry in map.iter()
				{
					memory.write_cell(neuron, *entry.key(), TRUE_U8, true);
				}
			}),
			Accum::Votes(maps) => maps.par_iter().enumerate().for_each(|(neuron, map)| {
				for entry in map.iter()
				{
					let cell = match entry.value().signum()
					{
						1 => TRUE_U8,
						-1 => FALSE_U8,
						_ => EMPTY_U8,
					};
					memory.write_cell(neuron, *entry.key(), cell, true);
				}
			}),
			Accum::Oi(maps) => maps.par_iter().enumerate().for_each(|(neuron, map)| {
				for entry in map.iter()
				{
					let cell = oi_bin_to_cell(*entry.value()) as u8;
					memory.write_cell(neuron, *entry.key(), cell, true);
				}
			}),
		}
	}

	/// Number of (neuron, address) accumulators touched so far.
	pub fn touched(&self) -> usize
	{
		match &self.accum
		{
			Accum::Set(m) => m.iter().map(|d| d.len()).sum(),
			Accum::Votes(m) => m.iter().map(|d| d.len()).sum(),
			Accum::Oi(m) => m.iter().map(|d| d.len()).sum(),
		}
	}
}

/// Convenience: a fresh memory sized and defaulted for `mode` and `layout`.
pub fn new_memory(mode: CellMode, layout: ClusterLayout) -> SparseLayerMemory
{
	SparseLayerMemory::new_with_default(layout.num_neurons(), layout.bits_per_neuron, mode.default_cell())
}

#[cfg(test)]
mod tests
{
	use super::*;
	use crate::neuron_memory::{QUAD_FALSE, QUAD_TRUE, QUAD_WEAK_FALSE, QUAD_WEAK_TRUE};

	struct Lcg(u64);
	impl Lcg
	{
		fn next(&mut self) -> u64
		{
			self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			self.0 >> 11
		}
	}

	fn layout() -> ClusterLayout
	{
		ClusterLayout {
			num_clusters: 3,
			neurons_per_cluster: 4,
			bits_per_neuron: 10,
		}
	}

	fn data(seed: u64, rows: usize) -> (PackedBits, Vec<i64>, Vec<i64>)
	{
		let l = layout();
		let total_bits = 48;
		let mut rng = Lcg(seed);
		let conns: Vec<i64> = (0..l.num_neurons() * l.bits_per_neuron)
			.map(|_| (rng.next() % total_bits as u64) as i64)
			.collect();
		// Few distinct rows so addresses collide across examples and classes.
		let pool: Vec<Vec<bool>> = (0..12)
			.map(|_| (0..total_bits).map(|_| rng.next() & 1 == 1).collect())
			.collect();
		let mut bools = Vec::new();
		let mut labels = Vec::new();
		for _ in 0..rows
		{
			let r = (rng.next() % 12) as usize;
			bools.extend_from_slice(&pool[r]);
			labels.push((rng.next() % 3) as i64);
		}
		(PackedBits::from_bool_slice(&bools, total_bits), conns, labels)
	}

	fn train(mode: CellMode, input: &PackedBits, conns: &[i64], labels: &[i64]) -> Vec<(usize, u64, u8)>
	{
		let t = SparseTrainer::new(mode, layout());
		t.accumulate(input, conns, labels, None).unwrap();
		let m = new_memory(mode, layout());
		t.commit(&m);
		let mut e = m.export();
		e.sort();
		e
	}

	fn shuffled(input: &PackedBits, labels: &[i64], seed: u64) -> (PackedBits, Vec<i64>)
	{
		let mut idx: Vec<usize> = (0..input.num_rows()).collect();
		let mut rng = Lcg(seed);
		for i in (1..idx.len()).rev()
		{
			let j = (rng.next() % (i as u64 + 1)) as usize;
			idx.swap(i, j);
		}
		let input2 = input.select_rows(&idx);
		let labels2: Vec<i64> = idx.iter().map(|&i| labels[i]).collect();
		(input2, labels2)
	}

	/// THE property: the memory is a function of the multiset of examples, not
	/// of their order — for every mode.
	#[test]
	fn every_mode_is_order_independent()
	{
		let (input, conns, labels) = data(11, 300);
		for mode in CellMode::ALL
		{
			let a = train(mode, &input, &conns, &labels);
			assert!(!a.is_empty(), "{mode}: nothing written");
			for seed in 1..4
			{
				let (i2, l2) = shuffled(&input, &labels, seed);
				let b = train(mode, &i2, &conns, &l2);
				assert_eq!(a, b, "{mode}: memory depends on example order (shuffle seed {seed})");
			}
		}
	}

	/// BINARY: a cell is TRUE iff visited by an own-class example; negatives
	/// never write anything (a memory trained on one class has entries only in
	/// that class's neurons).
	#[test]
	fn binary_ignores_negatives()
	{
		let (input, conns, _) = data(5, 100);
		let labels = vec![1i64; input.num_rows()];
		let e = train(CellMode::Binary, &input, &conns, &labels);
		let l = layout();
		assert!(!e.is_empty());
		for (neuron, _, cell) in &e
		{
			assert_eq!(*neuron / l.neurons_per_cluster, 1, "negative-class neuron {neuron} was written");
			assert_eq!(*cell, TRUE_U8);
		}
	}

	/// TERNARY / PLN: sign of the vote decides; an exact tie reads EMPTY (which
	/// is the default, so it is NOT stored).
	#[test]
	fn ternary_sign_bins_and_ties_are_empty()
	{
		let l = ClusterLayout {
			num_clusters: 2,
			neurons_per_cluster: 1,
			bits_per_neuron: 2,
		};
		// One neuron per class, both reading bits [0,1]; every row = 0b11 → address 3.
		let conns = vec![0i64, 1, 0, 1];
		let rows = |labels: &[i64]| {
			let bools: Vec<bool> = labels.iter().flat_map(|_| [true, true]).collect();
			(PackedBits::from_bool_slice(&bools, 2), labels.to_vec())
		};
		for mode in [CellMode::Ternary, CellMode::Pln]
		{
			// class 0: 2 positives, 1 negative → TRUE ; class 1: 1 positive, 2 negatives → FALSE
			let (input, labels) = rows(&[0, 0, 1]);
			let t = SparseTrainer::new(mode, l);
			t.accumulate(&input, &conns, &labels, None).unwrap();
			let m = new_memory(mode, l);
			t.commit(&m);
			assert_eq!(m.read_cell_or_default(0, 3), TRUE_U8, "{mode}");
			assert_eq!(m.read_cell_or_default(1, 3), FALSE_U8, "{mode}");
			// tie: 1 positive, 1 negative → EMPTY, not stored
			let (input, labels) = rows(&[0, 1]);
			let t = SparseTrainer::new(mode, l);
			t.accumulate(&input, &conns, &labels, None).unwrap();
			let m = new_memory(mode, l);
			t.commit(&m);
			assert_eq!(m.read_cell_or_default(0, 3), EMPTY_U8, "{mode}");
			assert_eq!(m.total_cells(), 0, "{mode}: a tie must not be stored");
		}
	}

	/// QUAD: the OI bins (obs==1 → WEAK by sign; obs>=2 → net ≤ −1 FALSE, 0
	/// WEAK_FALSE, 1 WEAK_TRUE, ≥ 2 TRUE), through the public trainer.
	#[test]
	fn quad_oi_bins_through_the_trainer()
	{
		let l = ClusterLayout {
			num_clusters: 2,
			neurons_per_cluster: 1,
			bits_per_neuron: 2,
		};
		let conns = vec![0i64, 1, 0, 1];
		let rows = |labels: &[i64]| {
			let bools: Vec<bool> = labels.iter().flat_map(|_| [true, true]).collect();
			(PackedBits::from_bool_slice(&bools, 2), labels.to_vec())
		};
		let cell = |labels: &[i64], neuron: usize| -> u8 {
			let (input, labels) = rows(labels);
			let t = SparseTrainer::new(CellMode::QuadWeighted, l);
			t.accumulate(&input, &conns, &labels, None).unwrap();
			let m = new_memory(CellMode::QuadWeighted, l);
			t.commit(&m);
			m.read_cell_or_default(neuron, 3)
		};
		assert_eq!(cell(&[0], 0), QUAD_WEAK_TRUE as u8); // obs 1, net +1
		assert_eq!(cell(&[0], 1), QUAD_WEAK_FALSE as u8); // obs 1, net −1
		assert_eq!(cell(&[0, 0], 0), QUAD_TRUE as u8); // obs 2, net +2
		assert_eq!(cell(&[0, 1], 0), QUAD_WEAK_FALSE as u8); // obs 2, net 0
		assert_eq!(cell(&[0, 0, 1], 0), QUAD_WEAK_TRUE as u8); // obs 3, net +1
		assert_eq!(cell(&[1, 1], 0), QUAD_FALSE as u8); // obs 2, net −2
	}

	#[test]
	fn weights_scale_net_not_obs()
	{
		let l = ClusterLayout {
			num_clusters: 2,
			neurons_per_cluster: 1,
			bits_per_neuron: 2,
		};
		let conns = vec![0i64, 1, 0, 1];
		let input = PackedBits::from_bool_slice(&[true, true], 2);
		let t = SparseTrainer::new(CellMode::QuadWeighted, l);
		t.accumulate(&input, &conns, &[0], Some(&[3])).unwrap();
		let m = new_memory(CellMode::QuadWeighted, l);
		t.commit(&m);
		// obs==1 with net +3 → still WEAK_TRUE (obs gates the strong bins).
		assert_eq!(m.read_cell_or_default(0, 3), QUAD_WEAK_TRUE as u8);
	}

	/// `read_cell_or_default` reads a miss as the mode's default cell (what the
	/// scorers score); plain `read_cell` keeps its EMPTY-on-miss contract, which
	/// the controller's BINARY don't-punish rule depends on.
	#[test]
	fn memory_miss_reads()
	{
		for mode in CellMode::ALL
		{
			let m = new_memory(mode, layout());
			assert_eq!(m.read_cell_or_default(0, 12345), mode.default_cell(), "{mode}");
			assert_eq!(m.read_cell(0, 12345), EMPTY_U8, "{mode}: read_cell must stay EMPTY on a miss");
		}
		assert_eq!(SparseLayerMemory::new(1, 4).read_cell_or_default(0, 0), EMPTY_U8, "NO_CANONICAL_DEFAULT keeps EMPTY");
	}

	#[test]
	fn validate_rejects_bad_shapes()
	{
		let (input, conns, mut labels) = data(3, 10);
		let t = SparseTrainer::new(CellMode::QuadWeighted, layout());
		assert!(t.accumulate(&input, &conns[1..], &labels, None).is_err());
		labels[0] = 3;
		assert!(t.accumulate(&input, &conns, &labels, None).is_err());
		labels[0] = 0;
		assert!(t.accumulate(&input, &conns, &labels[1..], None).is_err());
		assert!(t.accumulate(&input, &conns, &labels, Some(&[1u32; 3])).is_err());
	}
}
