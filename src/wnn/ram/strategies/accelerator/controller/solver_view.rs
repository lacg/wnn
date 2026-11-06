//! ABI 35 (B): a solver-side VIEW of one SparseLayerMemory that lives for a whole
//! bptt window instead of being rebuilt per record.
//!
//! The reachable solver needs, per neuron, (1) the written cells to rank and (2) an
//! "is this address written?" test for the untrained Hamming hunt. Before ABI 35 the
//! walk got both by calling `neuron_entries()` (a clone of the neuron's DashMap) and
//! building a `HashSet` of it — for EVERY neuron of EVERY record, i.e. O(total cells)
//! of allocation per record on top of the O(total cells) scan the solve really needs.
//!
//! The view is built ONCE per window and kept in lock-step with the memory: every
//! commit the walk makes is mirrored by `apply_write`, which reproduces
//! `SparseLayerMemory::write_cell(.., allow_override=true)` exactly, including the
//! canonical-default DELETE (a write of the layer's default cell removes the entry).
//! So `entries(n)` is always the same SET `neuron_entries(n)` would return, and
//! `contains` is membership in it. Entry ORDER differs from the DashMap's, which is
//! harmless: the solver fully sorts its candidates by (cost, address), and addresses
//! are unique, so the top-k it keeps is order-independent.

use ram_core::sparse_memory::SparseLayerMemory;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// FxHash-style multiplicative hasher for u64 addresses (the view's index only).
#[derive(Default)]
pub(crate) struct AddrHasher(u64);

impl Hasher for AddrHasher
{
	#[inline]
	fn finish(&self) -> u64
	{
		self.0
	}
	#[inline]
	fn write(&mut self, bytes: &[u8])
	{
		for &b in bytes
		{
			self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7cc1_b727_220a_95);
		}
	}
	#[inline]
	fn write_u64(&mut self, v: u64)
	{
		self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7cc1_b727_220a_95);
	}
}

type AddrMap = HashMap<u64, u32, BuildHasherDefault<AddrHasher>>;

pub(crate) struct SolverView
{
	entries: Vec<Vec<(u64, u8)>>,
	index: Vec<AddrMap>,
	default_cell: u8,
}

impl SolverView
{
	/// Snapshot `mem` (all its neurons). `default_cell` must be the canonical default
	/// the memory was built with (cell_mode::canonical_default_cell(mode)).
	pub(crate) fn from_memory(mem: &SparseLayerMemory, default_cell: u8) -> Self
	{
		let n = mem.num_neurons;
		let mut entries = Vec::with_capacity(n);
		let mut index = Vec::with_capacity(n);
		for i in 0..n
		{
			let e = mem.neuron_entries(i);
			let mut m: AddrMap = AddrMap::with_capacity_and_hasher(e.len() * 2, Default::default());
			for (p, &(a, _)) in e.iter().enumerate()
			{
				m.insert(a, p as u32);
			}
			entries.push(e);
			index.push(m);
		}
		SolverView { entries, index, default_cell }
	}

	#[inline]
	pub(crate) fn entries(&self, n: usize) -> &[(u64, u8)]
	{
		&self.entries[n]
	}

	#[inline]
	pub(crate) fn contains(&self, n: usize, addr: u64) -> bool
	{
		self.index[n].contains_key(&addr)
	}

	/// Mirror of `write_cell(n, addr, value, allow_override = true)`.
	pub(crate) fn apply_write(&mut self, n: usize, addr: u64, value: u8)
	{
		let pos = self.index[n].get(&addr).copied();
		if value == self.default_cell
		{
			if let Some(p) = pos
			{
				let p = p as usize;
				self.index[n].remove(&addr);
				let last = self.entries[n].len() - 1;
				self.entries[n].swap_remove(p);
				if p != last
				{
					let moved = self.entries[n][p].0;
					self.index[n].insert(moved, p as u32);
				}
			}
			return;
		}
		match pos
		{
			Some(p) => self.entries[n][p as usize].1 = value,
			None =>
			{
				self.index[n].insert(addr, self.entries[n].len() as u32);
				self.entries[n].push((addr, value));
			}
		}
	}
}
