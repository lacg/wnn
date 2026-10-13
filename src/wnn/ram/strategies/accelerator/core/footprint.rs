//! `Footprint`: what a trained layer costs, LOGICAL and PHYSICAL — always both.
//!
//! Logical = what the genome says (copies included, e.g. 500n × 34b): every
//! logical neuron with the cells its slot reads. Physical = what is actually
//! trained and held: one unit per distinct neuron (`UnitGrouping` group). For
//! the plain layer the two are equal and both are still reported
//! (docs/ids_multiplicity_layer_design.md §5).
//!
//! Cells: a sparse neuron counts its trained (materialized) keys, a dense
//! neuron its full 2^bits array — the convention-free primitive of
//! docs/sparse_footprint_fix.md (formerly `GenomeExport::materialized_cells`).

/// Logical vs physical neuron and cell counts of one trained genome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Footprint
{
	pub logical_neurons: u64,
	pub physical_neurons: u64,
	pub logical_cells: u64,
	pub physical_cells: u64,
}

impl Footprint
{
	/// Account one logical neuron holding `cells`; `is_physical` marks the one
	/// logical neuron per unit (its representative) that owns the storage.
	pub fn add_neuron(&mut self, cells: u64, is_physical: bool)
	{
		self.logical_neurons += 1;
		self.logical_cells = self.logical_cells.saturating_add(cells);
		if is_physical
		{
			self.physical_neurons += 1;
			self.physical_cells = self.physical_cells.saturating_add(cells);
		}
	}

	/// Physical storage equals the logical layout (no duplicate collapsed).
	pub fn is_plain(&self) -> bool
	{
		self.physical_neurons == self.logical_neurons
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn footprint_counts_logical_and_physical()
	{
		let mut fp = Footprint::default();
		fp.add_neuron(10, true);
		fp.add_neuron(10, false);
		fp.add_neuron(4, true);
		assert_eq!(fp, Footprint { logical_neurons: 3, physical_neurons: 2, logical_cells: 24, physical_cells: 14 });
		assert!(!fp.is_plain());
		assert!(Footprint::default().is_plain());
	}
}
