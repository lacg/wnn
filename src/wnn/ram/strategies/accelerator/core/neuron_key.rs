//! `NeuronKey`: the ONE definition of "the same neuron".
//!
//! Two neurons are the same iff they sit in the same cluster, read the same
//! number of address bits and observe the same connections in the same order.
//! Same key ⇒ same address function ⇒ (at neuron_sample_rate >= 1.0, with an
//! order-independent writer) byte-identical trained memory. A permuted
//! connection list is a DIFFERENT key on purpose: the address differs.
//!
//! Previously inlined as the `(bits, conns)` tuple of the GA crossover dedupe.

/// Identity of a neuron within a genome.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NeuronKey
{
	pub cluster: u32,
	pub bits: u32,
	pub connections: Vec<i64>,
}

impl NeuronKey
{
	pub fn of(cluster: u32, bits: u32, connections: &[i64]) -> Self
	{
		Self { cluster, bits, connections: connections.to_vec() }
	}

	/// Keys of every neuron of a genome in logical (cluster-major) order.
	/// `connections_flat` is the UNPADDED layout: neuron i owns `bits_flat[i]`
	/// consecutive entries.
	pub fn genome_keys<'a>(
		bits_flat: &'a [usize],
		neurons_per_cluster: &'a [usize],
		connections_flat: &'a [i64],
	) -> impl Iterator<Item = NeuronKey> + 'a
	{
		let clusters = neurons_per_cluster
			.iter()
			.enumerate()
			.flat_map(|(c, &n)| std::iter::repeat(c as u32).take(n));
		let mut conn_start = 0usize;
		clusters.zip(bits_flat.iter()).map(move |(cluster, &bits)| {
			let key = NeuronKey::of(cluster, bits as u32, &connections_flat[conn_start..conn_start + bits]);
			conn_start += bits;
			key
		})
	}
}

#[cfg(test)]
mod tests
{
	use super::*;
	use crate::unit_grouping::UnitGrouping;

	#[test]
	fn neuron_key_cluster_and_bits_sensitive()
	{
		let base = NeuronKey::of(0, 3, &[1, 2, 3]);
		assert_eq!(base, NeuronKey::of(0, 3, &[1, 2, 3]));
		assert_ne!(base, NeuronKey::of(1, 3, &[1, 2, 3]), "cluster must separate");
		assert_ne!(base, NeuronKey::of(0, 4, &[1, 2, 3]), "bits must separate");
		assert_ne!(base, NeuronKey::of(0, 3, &[3, 2, 1]), "order must separate");
	}

	#[test]
	fn neuron_key_genome_keys_walk()
	{
		// clusters [2, 2]; neuron 2 duplicates neuron 0's conns in ANOTHER cluster,
		// neuron 3 duplicates neuron 2 within cluster 1.
		let bits = [2usize, 1, 2, 2];
		let npc = [2usize, 2];
		let conns = [4i64, 5, 9, 4, 5, 4, 5];
		let keys: Vec<NeuronKey> = NeuronKey::genome_keys(&bits, &npc, &conns).collect();
		assert_eq!(keys.len(), 4);
		assert_eq!(keys[1], NeuronKey::of(0, 1, &[9]));
		assert_eq!(keys[2], NeuronKey::of(1, 2, &[4, 5]));
		let g = UnitGrouping::by_key(keys);
		assert_eq!(g.representatives(), &[0, 1, 2]);
		assert_eq!(g.index_of(3), 2);
	}
}
