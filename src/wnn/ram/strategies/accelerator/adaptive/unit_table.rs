//! `UnitTable`: one PHYSICAL unit of a trained genome as a deployable lookup
//! table — its own connections and every (address, effective cell) it can
//! read (IDS-37 stage 7, the FPGA exporter's source of truth).
//!
//! Read straight out of the scored `GenomeExport` (the representative's slot),
//! so the table is exactly what the scorer reads: sparse = the stored keys
//! (an absent key reads `miss_cell`), dense = all 2^bits addresses with
//! coverage-aware holes resolved to cell 0 (as the scorer does).

use super::genome_export_physical::{cluster_starts, group_slots};
use super::*;
use ram_core::super_genome::FlatGenome;

/// One unit (its representative logical neuron's slot).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitTable
{
	pub cluster: u32,
	pub bits: u32,
	pub connections: Vec<i64>,
	/// Ascending addresses; dense units list all 2^bits.
	pub keys: Vec<u64>,
	pub cells: Vec<u8>,
}

impl GenomeExport
{
	/// Every unit's table, in unit order. `logical` is the genome the export
	/// was trained from (connections / bits / clusters of each neuron).
	pub fn unit_tables(&self, logical: FlatGenome, coverage_aware: bool) -> Vec<UnitTable>
	{
		let starts = cluster_starts(logical.neurons_per_cluster);
		let mut tables: Vec<Option<UnitTable>> = vec![None; self.unit_count(logical)];
		for (gi, group) in self.groups.iter().enumerate()
		{
			for (slot, l) in group_slots(group, &starts).into_iter().enumerate()
			{
				let Some(l) = l.filter(|&l| self.owns(l))
				else
				{
					continue;
				};
				tables[self.unit_index(l)] = Some(self.slot_table(gi, slot, l, logical, coverage_aware));
			}
		}
		tables
			.into_iter()
			.map(|t| t.expect("every unit has a representative slot"))
			.collect()
	}

	fn unit_count(&self, logical: FlatGenome) -> usize
	{
		if self.grouping.is_identity()
		{
			logical.bits.len()
		}
		else
		{
			self.grouping.num_groups()
		}
	}

	fn owns(&self, l: usize) -> bool
	{
		self.grouping.is_identity() || self.grouping.is_representative(l)
	}

	fn unit_index(&self, l: usize) -> usize
	{
		if self.grouping.is_identity()
		{
			l
		}
		else
		{
			self.grouping.index_of(l) as usize
		}
	}

	fn slot_table(
		&self,
		gi: usize,
		slot: usize,
		l: usize,
		logical: FlatGenome,
		coverage_aware: bool,
	) -> UnitTable
	{
		let bits = logical.bits[l];
		let conn_start: usize = logical.bits[..l].iter().sum();
		let (keys, cells) = if self.group_info[gi].0
		{
			self.sparse_slot(gi, slot)
		}
		else
		{
			self.dense_slot(gi, slot, bits, coverage_aware)
		};
		UnitTable {
			cluster: cluster_of(logical.neurons_per_cluster, l) as u32,
			bits: bits as u32,
			connections: logical.connections[conn_start..conn_start + bits].to_vec(),
			keys,
			cells,
		}
	}

	fn sparse_slot(&self, gi: usize, slot: usize) -> (Vec<u64>, Vec<u8>)
	{
		let (keys, values) = self.sparse_exports[self.group_info[gi].1].neuron_range(slot);
		(keys.to_vec(), values.to_vec())
	}

	/// All 2^bits addresses of a dense slot; uncovered cells read 0 when coverage-aware.
	fn dense_slot(
		&self,
		gi: usize,
		slot: usize,
		bits: usize,
		coverage_aware: bool,
	) -> (Vec<u64>, Vec<u8>)
	{
		let sub = self.group_info[gi].1;
		let coverage = self
			.dense_coverage
			.get(sub)
			.map_or(&[][..], |c| c.as_slice());
		let per_neuron = 1usize << self.groups[gi].bits;
		let cell = |a: usize| {
			if coverage_aware && !dense_is_covered(coverage, slot, a, per_neuron)
			{
				0
			}
			else
			{
				self.read_cell_at(gi, slot, a as u64, 0) as u8
			}
		};
		(
			(0..1u64 << bits).collect(),
			(0..1usize << bits).map(cell).collect(),
		)
	}
}

/// Cluster of genome-order neuron `l`.
fn cluster_of(npc: &[usize], l: usize) -> usize
{
	let mut acc = 0usize;
	npc
		.iter()
		.position(|&n| {
			acc += n;
			l < acc
		})
		.unwrap_or(npc.len().saturating_sub(1))
}

/// Per-cell-code weight in QUARTERS (`cell_to_weight` x 4) when every code's
/// weight is an exact quarter — the integer adder an RTL design implements.
/// None for the stochastic modes (QSR / PLN read a per-example coin).
pub fn cell_weight_quarters(memory_mode: u8, empty_value: f32) -> Option<[u32; 4]>
{
	if matches!(
		memory_mode,
		ram_core::neuron_memory::QSR | ram_core::neuron_memory::PLN
	)
	{
		return None;
	}
	let mut q = [0u32; 4];
	for (cell, w) in q.iter_mut().enumerate()
	{
		let quarters = cell_to_weight(cell as i64, memory_mode, empty_value) * 4.0;
		if quarters.fract() != 0.0 || quarters < 0.0
		{
			return None;
		}
		*w = quarters as u32;
	}
	Some(q)
}
