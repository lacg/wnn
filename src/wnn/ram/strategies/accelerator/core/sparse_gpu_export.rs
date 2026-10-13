//! `SparseGpuExport`: the sorted CSR export of a sparse layer's trained memory.
//!
//! Neuron n owns `keys[offsets[n]..offsets[n]+counts[n]]` (sorted addresses) and
//! the parallel `values` (cells). Every scorer — CPU binary search, Metal
//! `sparse_forward.metal`, the batched CE evaluators — reads exactly that range,
//! and nothing assumes the ranges are DISJOINT, so several neurons may point at
//! one range (`alias`: the multiplicity layer's shared physical storage).
//!
//! Unified 12/10/2026 (IDS-37 stage 1): the IDS worker carried a field-identical
//! copy in `adaptive/memory.rs`; it now re-exports this one, so `alias` exists
//! once. `neuron_memory::SparseGpuExport` (legacy, no IDS consumer) remains.

use crate::unit_grouping::UnitGrouping;

/// GPU-compatible sparse memory export (sorted arrays for binary search).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseGpuExport
{
	/// Sorted keys per neuron, concatenated
	pub keys: Vec<u64>,
	/// Cells parallel to `keys`
	pub values: Vec<u8>,
	/// Start of each neuron's range in `keys`
	pub offsets: Vec<u32>,
	/// Entries in each neuron's range
	pub counts: Vec<u32>,
	/// Number of neurons (= offsets.len() = counts.len())
	pub num_neurons: usize,
}

impl SparseGpuExport
{
	/// CPU binary search lookup. `miss_default` is the cell an ABSENT address
	/// reads as — memory-mode-dependent (QUAD: 1 = WEAK_FALSE, TERNARY: 2 =
	/// EMPTY; compute via `default_cell_for_mode`). A hardcoded TERNARY EMPTY
	/// miss read as WEAK_TRUE (0.75) under QUAD (fixed 07/07/2026).
	#[inline]
	pub fn lookup(&self, neuron_idx: usize, address: u64, miss_default: u8) -> u8
	{
		let (keys, values) = self.neuron_range(neuron_idx);
		match keys.binary_search(&address)
		{
			Ok(idx) => values[idx],
			Err(_) => miss_default,
		}
	}

	/// The (keys, values) range a neuron reads.
	#[inline]
	pub fn neuron_range(&self, neuron_idx: usize) -> (&[u64], &[u8])
	{
		let start = self.offsets[neuron_idx] as usize;
		let end = start + self.counts[neuron_idx] as usize;
		(&self.keys[start..end], &self.values[start..end])
	}

	/// Total memory size in bytes
	pub fn memory_size(&self) -> usize
	{
		self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4 + self.counts.len() * 4
	}

	/// Total stored entries (physical; aliased ranges count once)
	pub fn total_entries(&self) -> usize
	{
		self.keys.len()
	}

	/// Build an export with one neuron per range, copied in order.
	pub fn from_ranges<'a, I: IntoIterator<Item = (&'a [u64], &'a [u8])>>(ranges: I) -> Self
	{
		let mut out = Self { keys: Vec::new(), values: Vec::new(), offsets: Vec::new(), counts: Vec::new(), num_neurons: 0 };
		for (keys, values) in ranges
		{
			debug_assert_eq!(keys.len(), values.len());
			out.offsets.push(out.keys.len() as u32);
			out.counts.push(keys.len() as u32);
			out.keys.extend_from_slice(keys);
			out.values.extend_from_slice(values);
			out.num_neurons += 1;
		}
		out
	}

	/// Logical view over shared physical storage: `self` holds one neuron per
	/// GROUP of `grouping`; the result holds one neuron per UNIT, whose
	/// offset/count point at its group's range. Keys/values are stored once —
	/// k copies of a unit cost 8 bytes of (offset, count) each, not k ranges.
	pub fn alias(&self, grouping: &UnitGrouping) -> Self
	{
		assert_eq!(self.num_neurons, grouping.num_groups(), "alias: export neurons != grouping groups");
		Self {
			keys: self.keys.clone(),
			values: self.values.clone(),
			offsets: grouping.expand(&self.offsets),
			counts: grouping.expand(&self.counts),
			num_neurons: grouping.num_units(),
		}
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	fn physical() -> SparseGpuExport
	{
		let a: (&[u64], &[u8]) = (&[1, 5, 9], &[3, 0, 2]);
		let b: (&[u64], &[u8]) = (&[], &[]);
		let c: (&[u64], &[u8]) = (&[2, 4], &[1, 3]);
		SparseGpuExport::from_ranges([a, b, c])
	}

	#[test]
	fn sparse_alias_lookup_equals_per_neuron_lookup()
	{
		let phys = physical();
		let grouping = UnitGrouping::by_key(["a", "c", "a", "b", "c", "a"]);
		let logical = phys.alias(&grouping);
		assert_eq!(logical.num_neurons, 6);
		assert_eq!(logical.total_entries(), phys.total_entries(), "alias must not copy ranges");
		// The same logical layout trained plain: every copy owns its own range.
		let plain = SparseGpuExport::from_ranges((0..6).map(|u| phys.neuron_range(grouping.index_of(u) as usize)));
		for u in 0..6
		{
			for addr in 0..12u64
			{
				assert_eq!(logical.lookup(u, addr, 7), plain.lookup(u, addr, 7), "unit {u} addr {addr}");
			}
		}
	}

	#[test]
	fn sparse_alias_identity_is_noop()
	{
		let phys = physical();
		assert_eq!(phys.alias(&UnitGrouping::identity(3)), phys);
	}
}
