//! `SuperGenome`: the distinct neurons of a whole genome BATCH as one flat
//! genome, so a batch trains in a single dispatch (IDS-37 stage 4).
//!
//! The batch's neurons are walked cluster-major (cluster c of genome 0, of
//! genome 1, ...) and collapsed by `NeuronKey` within each genome, so every
//! cluster's units stay contiguous whatever the genomes' per-cluster neuron
//! counts — no uniform-shape constraint, no padding, no kernel edit.
//! `unit_of[i][l]` maps logical neuron l of the batch's i-th genome to its unit.
//! Exact wherever `neuron_memory::multiplicity_applies` holds: a unit trains
//! as each of its copies would (docs/ids_multiplicity_layer_design.md §3).

use crate::neuron_key::NeuronKey;
use crate::unit_grouping::UnitGrouping;

/// One flat genome: `connections` is unpadded, neuron i owns `bits[i]` entries.
#[derive(Clone, Copy, Debug)]
pub struct FlatGenome<'a>
{
	pub bits: &'a [usize],
	pub neurons_per_cluster: &'a [usize],
	pub connections: &'a [i64],
}

/// The collapsed batch: a flat genome of units plus each genome's logical -> unit map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuperGenome
{
	/// Bits per unit
	pub bits: Vec<usize>,
	/// Units per cluster
	pub neurons_per_cluster: Vec<usize>,
	/// Unpadded connections, unit-major
	pub connections: Vec<i64>,
	/// Per genome (batch order): logical neuron -> unit
	pub unit_of: Vec<Vec<u32>>,
	/// Position in the cluster-major walk -> unit
	pub grouping: UnitGrouping,
}

/// A batch neuron in cluster-major walk order.
struct Visit
{
	genome: usize,
	logical: usize,
}

impl SuperGenome
{
	pub fn collapse(genomes: &[FlatGenome]) -> Self
	{
		let keys: Vec<Vec<NeuronKey>> = genomes
			.iter()
			.map(|g| NeuronKey::genome_keys(g.bits, g.neurons_per_cluster, g.connections).collect())
			.collect();
		let walk = cluster_major_walk(genomes);
		let grouping = UnitGrouping::by_key(walk.iter().map(|v| (v.genome, &keys[v.genome][v.logical])));
		let mut unit_of: Vec<Vec<u32>> = genomes.iter().map(|g| vec![0u32; g.bits.len()]).collect();
		for (pos, v) in walk.iter().enumerate()
		{
			unit_of[v.genome][v.logical] = grouping.index_of(pos);
		}
		let reps: Vec<&NeuronKey> = grouping.representatives().iter().map(|&p| &keys[walk[p as usize].genome][walk[p as usize].logical]).collect();
		Self {
			bits: reps.iter().map(|k| k.bits as usize).collect(),
			neurons_per_cluster: unit_counts(&reps, genomes.first().map_or(0, |g| g.neurons_per_cluster.len())),
			connections: reps.iter().flat_map(|k| k.connections.iter().copied()).collect(),
			unit_of,
			grouping,
		}
	}

	pub fn num_units(&self) -> usize
	{
		self.bits.len()
	}

	pub fn genome(&self) -> FlatGenome<'_>
	{
		FlatGenome { bits: &self.bits, neurons_per_cluster: &self.neurons_per_cluster, connections: &self.connections }
	}
}

/// Every batch neuron, cluster-major: cluster c's neurons of genome 0, then of
/// genome 1, ...; within a genome in logical order.
fn cluster_major_walk(genomes: &[FlatGenome]) -> Vec<Visit>
{
	let num_clusters = genomes.first().map_or(0, |g| g.neurons_per_cluster.len());
	let starts: Vec<Vec<usize>> = genomes.iter().map(|g| cluster_starts(g.neurons_per_cluster)).collect();
	let mut walk = Vec::with_capacity(genomes.iter().map(|g| g.bits.len()).sum());
	for c in 0..num_clusters
	{
		for (genome, g) in genomes.iter().enumerate()
		{
			let s = starts[genome][c];
			walk.extend((s..s + g.neurons_per_cluster[c]).map(|logical| Visit { genome, logical }));
		}
	}
	walk
}

fn cluster_starts(npc: &[usize]) -> Vec<usize>
{
	npc.iter()
		.scan(0usize, |acc, &n| {
			let s = *acc;
			*acc += n;
			Some(s)
		})
		.collect()
}

fn unit_counts(reps: &[&NeuronKey], num_clusters: usize) -> Vec<usize>
{
	let mut counts = vec![0usize; num_clusters];
	for k in reps
	{
		counts[k.cluster as usize] += 1;
	}
	counts
}

#[cfg(test)]
mod tests
{
	use super::*;
	use crate::physical_genome::PhysicalGenome;

	fn flat<'a>(bits: &'a [usize], npc: &'a [usize], conns: &'a [i64]) -> FlatGenome<'a>
	{
		FlatGenome { bits, neurons_per_cluster: npc, connections: conns }
	}

	#[test]
	fn super_genome_of_one_is_the_physical_genome()
	{
		let (bits, npc, conns) = ([2usize, 1, 2, 2, 2], [3usize, 2], [1i64, 2, 3, 1, 2, 1, 2, 1, 2]);
		let s = SuperGenome::collapse(&[flat(&bits, &npc, &conns)]);
		let p = PhysicalGenome::collapse(&bits, &npc, &conns);
		assert_eq!((&s.bits, &s.neurons_per_cluster, &s.connections), (&p.bits, &p.neurons_per_cluster, &p.connections));
		assert_eq!(s.unit_of[0], (0..5).map(|l| p.grouping.index_of(l)).collect::<Vec<_>>());
	}

	#[test]
	fn super_genome_is_cluster_major_with_unequal_shapes()
	{
		// g0: c0 = [a, a], c1 = [b]; g1: c0 = [a], c1 = [b, c, b]. Per-genome
		// collapse only: g1's a and b are separate units from g0's.
		let (b0, n0, k0) = ([1usize, 1, 1], [2usize, 1], [5i64, 5, 6]);
		let (b1, n1, k1) = ([1usize, 1, 1, 1], [1usize, 3], [5i64, 6, 7, 6]);
		let s = SuperGenome::collapse(&[flat(&b0, &n0, &k0), flat(&b1, &n1, &k1)]);
		assert_eq!(s.neurons_per_cluster, vec![2, 3]);
		assert_eq!(s.connections, vec![5, 5, 6, 6, 7]);
		assert_eq!(s.unit_of, vec![vec![0, 0, 2], vec![1, 3, 4, 3]]);
		assert_eq!(s.num_units(), 5);
	}
}
