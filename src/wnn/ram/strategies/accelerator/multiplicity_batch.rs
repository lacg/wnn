//! `MultiplicityBatch`: train each genome's distinct neurons once (IDS-37 stage 2).
//!
//! Selected by the GPU training factory (`marker_train::batched_train_offspring`)
//! only when `ram_core::neuron_memory::multiplicity_applies(rate, mode)` holds —
//! then same-key neurons would train byte-identically, so each genome is
//! collapsed to its `PhysicalGenome`, the physical genome is trained through the
//! plain trainer (inheriting its budget/chunked routing), and the logical export
//! is rebuilt by `GenomeExport::from_physical`. Stage 2 trains G > 1 batches as
//! a per-genome loop; super-genome batching is stage 4.
//! docs/ids_multiplicity_layer_design.md §3.

use crate::adaptive::GenomeExport;
use ram_core::physical_genome::PhysicalGenome;

/// A flat genome batch (the factory's layout), collapsed per genome.
pub(crate) struct MultiplicityBatch<'a>
{
	bits_flat: &'a [usize],
	neurons_flat: &'a [usize],
	connections_flat: &'a [i64],
	num_clusters: usize,
	memory_mode: u8,
	physical: Vec<PhysicalGenome>,
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
		let mut batch = Self { bits_flat, neurons_flat, connections_flat, num_clusters, memory_mode, physical: Vec::new() };
		let (mut b0, mut c0) = (0usize, 0usize);
		for g in 0..num_genomes
		{
			let (bits, npc, conns) = batch.logical(g, b0, c0);
			batch.physical.push(PhysicalGenome::collapse(bits, npc, conns));
			b0 += bits.len();
			c0 += conns.len();
		}
		batch
	}

	/// Genome g's logical (bits, neurons_per_cluster, connections), given the
	/// running bits/connection offsets of genomes before it.
	fn logical(&self, g: usize, bits_start: usize, conn_start: usize) -> (&'a [usize], &'a [usize], &'a [i64])
	{
		let npc = &self.neurons_flat[g * self.num_clusters..(g + 1) * self.num_clusters];
		let bits = &self.bits_flat[bits_start..bits_start + npc.iter().sum::<usize>()];
		let conns = &self.connections_flat[conn_start..conn_start + bits.iter().sum::<usize>()];
		(bits, npc, conns)
	}

	/// True when no genome has a duplicate: the plain batched dispatch is then
	/// exactly the multiplicity result, and keeps its single GPU dispatch.
	pub(crate) fn is_identity(&self) -> bool
	{
		self.physical.iter().all(|p| p.grouping.is_identity())
	}

	/// Train every genome's physical genome with `train_physical` (the plain
	/// trainer for ONE genome) and return the logical exports in genome order.
	pub(crate) fn train<F>(&self, train_physical: F) -> Result<Vec<GenomeExport>, String>
	where
		F: Fn(&PhysicalGenome) -> Result<GenomeExport, String>,
	{
		let mut exports = Vec::with_capacity(self.physical.len());
		let (mut b0, mut c0) = (0usize, 0usize);
		for (g, phys) in self.physical.iter().enumerate()
		{
			let (bits, npc, conns) = self.logical(g, b0, c0);
			exports.push(self.train_one(phys, bits, npc, conns, &train_physical)?);
			b0 += bits.len();
			c0 += conns.len();
		}
		Ok(exports)
	}

	fn train_one<F>(
		&self,
		phys: &PhysicalGenome,
		bits: &[usize],
		npc: &[usize],
		conns: &[i64],
		train_physical: &F,
	) -> Result<GenomeExport, String>
	where
		F: Fn(&PhysicalGenome) -> Result<GenomeExport, String>,
	{
		let trained = train_physical(phys)?;
		if phys.grouping.is_identity()
		{
			return Ok(trained);
		}
		GenomeExport::from_physical(trained, phys.grouping.clone(), bits, npc, conns, self.memory_mode)
	}
}
