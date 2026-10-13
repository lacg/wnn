//! `PhysicalGenome`: a flat genome with duplicate neurons collapsed.
//!
//! The LOGICAL genome is what the GA evolves (copies included, e.g. 500n×34b).
//! The physical genome keeps one neuron per distinct `NeuronKey`, in
//! first-occurrence order, which keeps neurons cluster-contiguous (a key carries
//! its cluster). `grouping` maps logical neuron -> physical neuron. Training the
//! physical genome and reading it back through the grouping is exact whenever
//! `neuron_memory::multiplicity_applies` holds (docs/ids_multiplicity_layer_design.md).

use crate::neuron_key::NeuronKey;
use crate::unit_grouping::UnitGrouping;

/// A collapsed flat genome plus its logical -> physical map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalGenome
{
	/// Bits per physical neuron
	pub bits: Vec<usize>,
	/// Physical neurons per cluster (same cluster count as the logical genome)
	pub neurons_per_cluster: Vec<usize>,
	/// Unpadded connections, physical neuron-major
	pub connections: Vec<i64>,
	/// Logical neuron -> physical neuron
	pub grouping: UnitGrouping,
}

impl PhysicalGenome
{
	/// Collapse one flat genome. `connections_flat` is unpadded: neuron i owns
	/// `bits_flat[i]` consecutive entries.
	pub fn collapse(bits_flat: &[usize], neurons_per_cluster: &[usize], connections_flat: &[i64]) -> Self
	{
		let grouping = UnitGrouping::by_key(NeuronKey::genome_keys(bits_flat, neurons_per_cluster, connections_flat));
		let conn_starts = prefix_sums(bits_flat);
		let mut connections = Vec::new();
		for &rep in grouping.representatives()
		{
			let r = rep as usize;
			connections.extend_from_slice(&connections_flat[conn_starts[r]..conn_starts[r + 1]]);
		}
		Self {
			bits: grouping.gather(bits_flat),
			neurons_per_cluster: Self::cluster_counts(&grouping, neurons_per_cluster),
			connections,
			grouping,
		}
	}

	pub fn num_neurons(&self) -> usize
	{
		self.bits.len()
	}

	/// Physical neurons per cluster: the representatives falling in each cluster.
	pub fn cluster_counts(grouping: &UnitGrouping, neurons_per_cluster: &[usize]) -> Vec<usize>
	{
		let cluster_of: Vec<usize> = neurons_per_cluster
			.iter()
			.enumerate()
			.flat_map(|(c, &n)| std::iter::repeat(c).take(n))
			.collect();
		let mut counts = vec![0usize; neurons_per_cluster.len()];
		for &rep in grouping.representatives()
		{
			counts[cluster_of[rep as usize]] += 1;
		}
		counts
	}
}

/// `[0, b0, b0+b1, ...]`: start of each neuron's connection run.
fn prefix_sums(bits_flat: &[usize]) -> Vec<usize>
{
	let mut out = Vec::with_capacity(bits_flat.len() + 1);
	out.push(0);
	for &b in bits_flat
	{
		out.push(out.last().unwrap() + b);
	}
	out
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn physical_genome_collapses_within_cluster_only()
	{
		// cluster 0: n0 = [1,2] (b2), n1 = [3] (b1), n2 = [1,2] (dup of n0)
		// cluster 1: n3 = [1,2] (same conns as n0, OTHER cluster -> distinct), n4 = dup of n3
		let bits = [2usize, 1, 2, 2, 2];
		let npc = [3usize, 2];
		let conns = [1i64, 2, 3, 1, 2, 1, 2, 1, 2];
		let p = PhysicalGenome::collapse(&bits, &npc, &conns);
		assert_eq!(p.bits, vec![2, 1, 2]);
		assert_eq!(p.neurons_per_cluster, vec![2, 1]);
		assert_eq!(p.connections, vec![1, 2, 3, 1, 2]);
		assert_eq!((0..5).map(|u| p.grouping.index_of(u)).collect::<Vec<_>>(), vec![0, 1, 0, 2, 2]);
	}

	#[test]
	fn physical_genome_without_duplicates_is_the_genome()
	{
		let bits = [2usize, 2];
		let npc = [2usize];
		let conns = [1i64, 2, 2, 1];
		let p = PhysicalGenome::collapse(&bits, &npc, &conns);
		assert!(p.grouping.is_identity());
		assert_eq!((p.bits.as_slice(), p.neurons_per_cluster.as_slice(), p.connections.as_slice()), (&bits[..], &npc[..], &conns[..]));
	}
}
