//! `MultiplicityBatch`: train a genome batch's distinct neurons once, in one
//! dispatch (IDS-37 stages 2, 4, 5).
//!
//! Selected by the GPU training factory (`marker_train::batched_train_offspring`)
//! only when `ram_core::neuron_memory::multiplicity_applies(rate, mode)` holds —
//! then same-key neurons train byte-identically. The batch collapses to one
//! `SuperGenome` (the batch's distinct neurons — shared across genomes —
//! cluster-major, any shapes),
//! which trains as ONE genome; every genome's logical export is then rebuilt
//! from the trained units by `GenomeExport::from_physical`. A super-genome over
//! the GPU budget bisects the batch; a single genome over budget takes the
//! plain trainer's neuron-chunked route. docs/ids_multiplicity_layer_design.md §3.

use crate::adaptive::{GenomeExport, PhysicalUnits};
use crate::marker_train::batched_path::is_budget_error;
use ram_core::distinct_fractions::DistinctFractions;
use ram_core::neuron_key::NeuronKey;
use ram_core::super_genome::{FlatGenome, SuperGenome};
use ram_core::unit_grouping::UnitGrouping;
use std::borrow::Cow;

/// How a super-genome trains, as ONE genome. `batch` serves a multi-genome
/// super-genome and must surface a budget overflow as the budget error (the
/// batch then bisects); `single` serves one genome and may chunk.
pub(crate) struct UnitTrainers<B, S>
where
	B: Fn(&SuperGenome) -> Result<GenomeExport, String>,
	S: Fn(&SuperGenome) -> Result<GenomeExport, String>,
{
	pub batch: B,
	pub single: S,
}

/// A flat genome batch (the factory's layout), collapsed.
pub(crate) struct MultiplicityBatch<'a>
{
	genomes: Vec<FlatGenome<'a>>,
	/// Per genome: logical neuron -> its own distinct neuron (the footprint)
	own: Vec<UnitGrouping>,
	whole: SuperGenome,
	memory_mode: u8,
}

impl<'a> MultiplicityBatch<'a>
{
	/// `connections_flat` is the unpadded genome-major concatenation.
	pub(crate) fn new(
		bits_flat: &'a [usize],
		neurons_flat: &'a [usize],
		connections_flat: &'a [i64],
		num_genomes: usize,
		num_clusters: usize,
		memory_mode: u8,
	) -> Self
	{
		let genomes = FlatGenome::split_batch(bits_flat, neurons_flat, connections_flat, num_genomes, num_clusters);
		let own = genomes
			.iter()
			.map(|g| UnitGrouping::by_key(NeuronKey::genome_keys(g.bits, g.neurons_per_cluster, g.connections)))
			.collect();
		let whole = SuperGenome::collapse(&genomes);
		Self { genomes, own, whole, memory_mode }
	}

	/// True when the plain batched dispatch is exactly the multiplicity result
	/// AND can take this batch: nothing collapses and every genome has the
	/// same per-cluster neuron counts (the plain kernel's shape constraint;
	/// the super-genome has none, so a mixed-shape batch always takes it).
	pub(crate) fn plain_is_exact(&self) -> bool
	{
		let first = self.genomes.first().map(|g| g.neurons_per_cluster);
		self.whole.grouping.is_identity() && self.genomes.iter().all(|g| Some(g.neurons_per_cluster) == first)
	}

	/// The `[MULTIPLICITY]` line: per-genome distinct fractions and the
	/// batch-wide distinct fraction — exactly the units this batch trains.
	pub(crate) fn log_line(&self) -> String
	{
		DistinctFractions::from_groupings(&self.own, &self.whole.grouping).log_line()
	}

	/// Train the batch's units and return the logical exports in genome order.
	pub(crate) fn train<B, S>(&self, trainers: &UnitTrainers<B, S>) -> Result<Vec<GenomeExport>, String>
	where
		B: Fn(&SuperGenome) -> Result<GenomeExport, String>,
		S: Fn(&SuperGenome) -> Result<GenomeExport, String>,
	{
		let all: Vec<usize> = (0..self.genomes.len()).collect();
		self.train_subset(&all, trainers)
	}

	fn train_subset<B, S>(&self, subset: &[usize], trainers: &UnitTrainers<B, S>) -> Result<Vec<GenomeExport>, String>
	where
		B: Fn(&SuperGenome) -> Result<GenomeExport, String>,
		S: Fn(&SuperGenome) -> Result<GenomeExport, String>,
	{
		let units = self.super_genome(subset);
		let trained = if subset.len() == 1 { (trainers.single)(&units) } else { (trainers.batch)(&units) };
		match trained
		{
			Ok(export) => self.logical_exports(subset, &units, &export),
			Err(msg) if subset.len() > 1 && is_budget_error(&msg) => self.bisect(subset, trainers),
			Err(msg) => Err(msg),
		}
	}

	/// Over budget: train each half on its own (the halves' shared units train twice).
	fn bisect<B, S>(&self, subset: &[usize], trainers: &UnitTrainers<B, S>) -> Result<Vec<GenomeExport>, String>
	where
		B: Fn(&SuperGenome) -> Result<GenomeExport, String>,
		S: Fn(&SuperGenome) -> Result<GenomeExport, String>,
	{
		let (left, right) = subset.split_at(subset.len() / 2);
		eprintln!("[MULTIPLICITY] super-genome over budget: bisecting {} genomes -> {} + {}", subset.len(), left.len(), right.len());
		let mut exports = self.train_subset(left, trainers)?;
		exports.extend(self.train_subset(right, trainers)?);
		Ok(exports)
	}

	fn super_genome(&self, subset: &[usize]) -> Cow<'_, SuperGenome>
	{
		if subset.len() == self.genomes.len()
		{
			return Cow::Borrowed(&self.whole);
		}
		let genomes: Vec<FlatGenome> = subset.iter().map(|&g| self.genomes[g]).collect();
		Cow::Owned(SuperGenome::collapse(&genomes))
	}

	fn logical_exports(&self, subset: &[usize], units: &SuperGenome, export: &GenomeExport) -> Result<Vec<GenomeExport>, String>
	{
		let physical = PhysicalUnits::new(export, &units.neurons_per_cluster);
		subset
			.iter()
			.enumerate()
			.map(|(i, &g)| {
				GenomeExport::from_physical(&physical, &units.unit_of[i], self.own[g].clone(), self.genomes[g], self.memory_mode)
			})
			.collect()
	}
}

/// Neurons per cluster the training factory actually trains (and holds) for
/// one genome: its distinct neurons when the multiplicity layer is selected,
/// every logical neuron otherwise. Memory sizing reads this, so a rate-1.0
/// batch is sized by its physical footprint, not its logical one.
pub(crate) fn trained_neurons_per_cluster(genome: &FlatGenome, neuron_sample_rate: f32, memory_mode: u8) -> Vec<usize>
{
	if !crate::marker_train::batched_path::multiplicity_selected(neuron_sample_rate, memory_mode, genome.connections)
	{
		return genome.neurons_per_cluster.to_vec();
	}
	let grouping = UnitGrouping::by_key(NeuronKey::genome_keys(genome.bits, genome.neurons_per_cluster, genome.connections));
	ram_core::physical_genome::PhysicalGenome::cluster_counts(&grouping, genome.neurons_per_cluster)
}

#[cfg(test)]
mod tests
{
	use super::*;
	use ram_core::neuron_memory::QUAD_WEIGHTED as QUAD_WEIGHTED_MODE;

	#[test]
	fn trained_neurons_per_cluster_is_physical_only_when_selected()
	{
		// cluster 0: a, a, b; cluster 1: c, c
		let (bits, npc, conns) = ([1usize, 1, 1, 1, 1], [3usize, 2], [5i64, 5, 6, 7, 7]);
		let g = FlatGenome { bits: &bits, neurons_per_cluster: &npc, connections: &conns };
		let selected = ram_core::neuron_memory::multiplicity_applies(1.0, QUAD_WEIGHTED_MODE);
		let expect = if selected { vec![2, 1] } else { vec![3, 2] };
		assert_eq!(trained_neurons_per_cluster(&g, 1.0, QUAD_WEIGHTED_MODE), expect);
		assert_eq!(trained_neurons_per_cluster(&g, 0.25, QUAD_WEIGHTED_MODE), vec![3, 2]);
		let random = FlatGenome { bits: &bits, neurons_per_cluster: &npc, connections: &[] };
		assert_eq!(trained_neurons_per_cluster(&random, 1.0, QUAD_WEIGHTED_MODE), vec![3, 2]);
	}

	#[test]
	fn plain_is_exact_needs_identity_and_one_shape()
	{
		let (bits, conns) = ([1usize, 1, 1], [5i64, 6, 7]);
		let same = MultiplicityBatch::new(&bits[..2], &[1, 1], &conns[..2], 2, 1, QUAD_WEIGHTED_MODE);
		assert!(same.plain_is_exact());
		let mixed = MultiplicityBatch::new(&bits, &[1, 2], &conns, 2, 1, QUAD_WEIGHTED_MODE);
		assert!(!mixed.plain_is_exact(), "mixed shapes: plain kernel cannot take the batch");
		let dup = MultiplicityBatch::new(&bits[..2], &[2], &[5, 5], 1, 1, QUAD_WEIGHTED_MODE);
		assert!(!dup.plain_is_exact());
	}
}
