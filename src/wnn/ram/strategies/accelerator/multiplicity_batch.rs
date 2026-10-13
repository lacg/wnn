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
		let genomes = split_genomes(bits_flat, neurons_flat, connections_flat, num_genomes, num_clusters);
		let own = genomes
			.iter()
			.map(|g| UnitGrouping::by_key(NeuronKey::genome_keys(g.bits, g.neurons_per_cluster, g.connections)))
			.collect();
		let whole = SuperGenome::collapse(&genomes);
		Self { genomes, own, whole, memory_mode }
	}

	/// True when nothing collapses: the plain batched dispatch is then exactly
	/// the multiplicity result, and keeps its single GPU dispatch.
	pub(crate) fn is_identity(&self) -> bool
	{
		self.whole.grouping.is_identity()
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

/// The factory's flat batch as per-genome slices.
fn split_genomes<'a>(
	bits_flat: &'a [usize],
	neurons_flat: &'a [usize],
	connections_flat: &'a [i64],
	num_genomes: usize,
	num_clusters: usize,
) -> Vec<FlatGenome<'a>>
{
	let (mut b0, mut c0) = (0usize, 0usize);
	let mut genomes = Vec::with_capacity(num_genomes);
	for g in 0..num_genomes
	{
		let npc = &neurons_flat[g * num_clusters..(g + 1) * num_clusters];
		let bits = &bits_flat[b0..b0 + npc.iter().sum::<usize>()];
		let conns = &connections_flat[c0..c0 + bits.iter().sum::<usize>()];
		genomes.push(FlatGenome { bits, neurons_per_cluster: npc, connections: conns });
		b0 += bits.len();
		c0 += conns.len();
	}
	genomes
}
