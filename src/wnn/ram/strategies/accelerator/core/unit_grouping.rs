//! `UnitGrouping`: group identical units — the one map from a LOGICAL layout
//! (unit i, in its original order) to PHYSICAL storage (one group per distinct
//! key, in first-occurrence order).
//!
//! Generic over the key, so every "collapse identical things" site shares it
//! instead of growing its own HashSet walk (docs/ids_multiplicity_layer_design.md
//! §2.1). Consumers: the GA crossover dedupe (`neighbor_search::dedupe_fill`),
//! the IDS multiplicity layer (`GenomeExport::from_physical`), the sparse export
//! alias (`SparseGpuExport::alias`) and the per-batch distinct-fraction log.
//!
//! Layout is CSR so `members(g)` (the k logical positions of a group) is a slice.

use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Group identical units: unit i (logical order) -> group index (first-occurrence order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitGrouping
{
	/// logical unit -> group
	index_of: Vec<u32>,
	/// group -> first logical unit with that key
	representatives: Vec<u32>,
	/// CSR over groups (len = groups + 1)
	member_offsets: Vec<u32>,
	/// logical units of each group, ascending
	members: Vec<u32>,
}

impl UnitGrouping
{
	/// Every unit is its own group (the plain layer).
	pub fn identity(num_units: usize) -> Self
	{
		let iota: Vec<u32> = (0..num_units as u32).collect();
		Self {
			index_of: iota.clone(),
			representatives: iota.clone(),
			member_offsets: (0..=num_units as u32).collect(),
			members: iota,
		}
	}

	/// Group units by key. Groups are numbered in order of their key's FIRST
	/// occurrence, so the representative sequence is the order a sequential
	/// "keep unless seen" walk would keep.
	pub fn by_key<K: Eq + Hash, I: IntoIterator<Item = K>>(keys: I) -> Self
	{
		let mut first: FxHashMap<K, u32> = FxHashMap::default();
		let mut index_of: Vec<u32> = Vec::new();
		let mut representatives: Vec<u32> = Vec::new();
		for (unit, key) in keys.into_iter().enumerate()
		{
			let next = representatives.len() as u32;
			let group = *first.entry(key).or_insert(next);
			if group == next
			{
				representatives.push(unit as u32);
			}
			index_of.push(group);
		}
		let (member_offsets, members) = Self::csr(&index_of, representatives.len());
		Self { index_of, representatives, member_offsets, members }
	}

	/// Counting-sort the units into per-group member lists (ascending).
	fn csr(index_of: &[u32], num_groups: usize) -> (Vec<u32>, Vec<u32>)
	{
		let mut offsets = vec![0u32; num_groups + 1];
		for &g in index_of
		{
			offsets[g as usize + 1] += 1;
		}
		for g in 0..num_groups
		{
			offsets[g + 1] += offsets[g];
		}
		let mut cursor = offsets.clone();
		let mut members = vec![0u32; index_of.len()];
		for (unit, &g) in index_of.iter().enumerate()
		{
			members[cursor[g as usize] as usize] = unit as u32;
			cursor[g as usize] += 1;
		}
		(offsets, members)
	}

	pub fn num_units(&self) -> usize
	{
		self.index_of.len()
	}

	pub fn num_groups(&self) -> usize
	{
		self.representatives.len()
	}

	pub fn is_identity(&self) -> bool
	{
		self.num_groups() == self.num_units()
	}

	pub fn index_of(&self, unit: usize) -> u32
	{
		self.index_of[unit]
	}

	pub fn is_representative(&self, unit: usize) -> bool
	{
		self.representatives[self.index_of[unit] as usize] as usize == unit
	}

	pub fn representatives(&self) -> &[u32]
	{
		&self.representatives
	}

	pub fn count(&self, group: usize) -> u32
	{
		self.member_offsets[group + 1] - self.member_offsets[group]
	}

	/// The k logical positions of a group, ascending.
	pub fn members(&self, group: usize) -> &[u32]
	{
		&self.members[self.member_offsets[group] as usize..self.member_offsets[group + 1] as usize]
	}

	/// Logical -> group order: keep each group's representative value.
	pub fn gather<T: Clone>(&self, per_unit: &[T]) -> Vec<T>
	{
		self.representatives.iter().map(|&u| per_unit[u as usize].clone()).collect()
	}

	/// Group -> logical order: every unit gets its group's value.
	pub fn expand<T: Clone>(&self, per_group: &[T]) -> Vec<T>
	{
		self.index_of.iter().map(|&g| per_group[g as usize].clone()).collect()
	}

	/// Physical / logical. 1.0 for identity (and for zero units).
	pub fn distinct_fraction(&self) -> f64
	{
		if self.num_units() == 0
		{
			return 1.0;
		}
		self.num_groups() as f64 / self.num_units() as f64
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn unit_grouping_identity()
	{
		let g = UnitGrouping::identity(4);
		assert!(g.is_identity());
		assert_eq!(g.num_groups(), 4);
		assert_eq!(g.representatives(), &[0, 1, 2, 3]);
		assert!((0..4).all(|u| g.is_representative(u) && g.count(u) == 1));
		assert_eq!(g, UnitGrouping::by_key([7, 8, 9, 10]));
		assert_eq!(UnitGrouping::identity(0).distinct_fraction(), 1.0);
	}

	#[test]
	fn unit_grouping_first_occurrence_order()
	{
		let g = UnitGrouping::by_key(["b", "a", "b", "c", "a", "b"]);
		assert!(!g.is_identity());
		assert_eq!(g.num_units(), 6);
		assert_eq!(g.num_groups(), 3);
		assert_eq!(g.representatives(), &[0, 1, 3]);
		assert_eq!((0..6).map(|u| g.index_of(u)).collect::<Vec<_>>(), vec![0, 1, 0, 2, 1, 0]);
		assert_eq!((0..6).filter(|&u| g.is_representative(u)).collect::<Vec<_>>(), vec![0, 1, 3]);
		assert!((g.distinct_fraction() - 0.5).abs() < 1e-12);
	}

	#[test]
	fn unit_grouping_csr_members()
	{
		let g = UnitGrouping::by_key([5, 6, 5, 7, 6, 5]);
		assert_eq!(g.members(0), &[0, 2, 5]);
		assert_eq!(g.members(1), &[1, 4]);
		assert_eq!(g.members(2), &[3]);
		assert_eq!((0..3).map(|k| g.count(k)).collect::<Vec<_>>(), vec![3, 2, 1]);
	}

	#[test]
	fn unit_grouping_gather_expand_round_trip()
	{
		let keys = vec![10u64, 20, 10, 30, 20];
		let g = UnitGrouping::by_key(keys.iter().copied());
		let physical = g.gather(&keys);
		assert_eq!(physical, vec![10, 20, 30]);
		assert_eq!(g.expand(&physical), keys);
	}
}
