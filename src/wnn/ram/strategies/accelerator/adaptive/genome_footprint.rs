//! `GenomeExport::footprint()`: logical AND physical size of a trained genome
//! (docs/ids_multiplicity_layer_design.md §5). Replaces `materialized_cells`
//! (its logical_cells is that number) so there is one footprint primitive.
//!
//! Every export holds the LOGICAL layout; `grouping` says which logical neuron
//! owns each physical unit (its representative). A slot's cells: sparse = the
//! keys its range holds, dense = the 2^bits array; coalescing pads count nothing.

use super::*;
use super::genome_export_physical::{cluster_starts, group_slots};
use ram_core::footprint::Footprint;

impl GenomeExport
{
	pub fn footprint(&self) -> Footprint
	{
		let starts = cluster_starts(&self.logical_npc());
		let mut fp = Footprint::default();
		for (gi, group) in self.groups.iter().enumerate()
		{
			for (slot, logical) in group_slots(group, &starts).into_iter().enumerate()
			{
				if let Some(l) = logical
				{
					fp.add_neuron(self.slot_cells(gi, slot), self.owns_unit(l));
				}
			}
		}
		fp
	}

	/// Neurons per cluster, read back from the groups (cluster c's actual count).
	fn logical_npc(&self) -> Vec<usize>
	{
		let num_clusters = self.groups.iter().flat_map(|g| g.cluster_ids.iter()).map(|&c| c + 1).max().unwrap_or(0);
		let mut npc = vec![0usize; num_clusters];
		for group in &self.groups
		{
			for (li, &c) in group.cluster_ids.iter().enumerate()
			{
				npc[c] = group.actual_neurons.as_ref().map_or(group.neurons, |a| a[li] as usize);
			}
		}
		npc
	}

	/// Cells logical slot `slot` of group `gi` reads.
	fn slot_cells(&self, gi: usize, slot: usize) -> u64
	{
		let (is_sparse, sub, _) = &self.group_info[gi];
		if *is_sparse
		{
			return *self.sparse_exports[*sub].counts.get(slot).unwrap_or(&0) as u64;
		}
		1u64.checked_shl(self.groups[gi].bits as u32).unwrap_or(u64::MAX)
	}

	/// True iff logical neuron `l` is its unit's representative (the plain
	/// layer: every neuron, whatever size the identity grouping was built with).
	fn owns_unit(&self, l: usize) -> bool
	{
		self.grouping.is_identity() || self.grouping.is_representative(l)
	}
}
