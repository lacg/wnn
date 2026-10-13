//! Distinct-neuron fractions of a genome batch: per genome (what the stage-2
//! multiplicity layer collapses) and across the whole batch (what stage-5
//! cross-genome sharing would collapse). Stage 0 of IDS-37 measured these
//! offline on 4 genomes; this puts the number on every real training batch so
//! stage 5 is sized on production data (docs/ids_multiplicity_layer_design.md).

use crate::neuron_key::NeuronKey;
use crate::unit_grouping::UnitGrouping;

/// Logical vs physical neuron counts of a flat genome batch.
#[derive(Clone, Debug, PartialEq)]
pub struct DistinctFractions
{
	/// Physical / logical neurons, one per genome
	pub per_genome: Vec<f64>,
	/// Logical neurons over the whole batch
	pub batch_logical: usize,
	/// Distinct `NeuronKey`s over the whole batch
	pub batch_distinct: usize,
}

impl DistinctFractions
{
	/// `connections_flat` is the unpadded concatenation of every genome's
	/// connections (genome-major, neuron i owns `bits_flat[i]` entries).
	pub fn of_batch(
		bits_flat: &[usize],
		neurons_flat: &[usize],
		connections_flat: &[i64],
		num_genomes: usize,
		num_clusters: usize,
	) -> Self
	{
		let mut per_genome = Vec::with_capacity(num_genomes);
		let mut all_keys: Vec<NeuronKey> = Vec::with_capacity(bits_flat.len());
		let (mut bit_start, mut conn_start) = (0usize, 0usize);
		for g in 0..num_genomes
		{
			let npc = &neurons_flat[g * num_clusters..(g + 1) * num_clusters];
			let n: usize = npc.iter().sum();
			let bits = &bits_flat[bit_start..bit_start + n];
			let conns_len: usize = bits.iter().sum();
			let keys: Vec<NeuronKey> =
				NeuronKey::genome_keys(bits, npc, &connections_flat[conn_start..conn_start + conns_len]).collect();
			per_genome.push(UnitGrouping::by_key(keys.iter()).distinct_fraction());
			all_keys.extend(keys);
			bit_start += n;
			conn_start += conns_len;
		}
		let batch_logical = all_keys.len();
		let batch_distinct = UnitGrouping::by_key(all_keys).num_groups();
		Self { per_genome, batch_logical, batch_distinct }
	}

	/// From groupings already built: each genome's own (logical -> distinct)
	/// and the batch's (every neuron -> the unit it trains as).
	pub fn from_groupings(per_genome: &[UnitGrouping], batch: &UnitGrouping) -> Self
	{
		Self {
			per_genome: per_genome.iter().map(UnitGrouping::distinct_fraction).collect(),
			batch_logical: batch.num_units(),
			batch_distinct: batch.num_groups(),
		}
	}

	/// One log line: `[MULTIPLICITY] batch genomes=G logical=N distinct/genome
	/// mean=..% min=..% max=..% | batch distinct=U (..%)`.
	pub fn log_line(&self) -> String
	{
		let pct = |f: f64| 100.0 * f;
		let n = self.per_genome.len().max(1) as f64;
		let mean = self.per_genome.iter().sum::<f64>() / n;
		let min = self.per_genome.iter().copied().fold(f64::INFINITY, f64::min);
		let max = self.per_genome.iter().copied().fold(0.0f64, f64::max);
		let batch = self.batch_distinct as f64 / self.batch_logical.max(1) as f64;
		format!(
			"[MULTIPLICITY] batch genomes={} logical={} distinct/genome mean={:.1}% min={:.1}% max={:.1}% | batch distinct={} ({:.1}%)",
			self.per_genome.len(),
			self.batch_logical,
			pct(mean),
			pct(if min.is_finite() { min } else { 1.0 }),
			pct(max),
			self.batch_distinct,
			pct(batch)
		)
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn distinct_fractions_per_genome_and_batch()
	{
		// genome 0: 4 neurons of 2 bits, neurons 0 and 2 identical -> 3/4.
		// genome 1: same as genome 0 except neuron 3 -> union adds 1 key.
		let bits = [2usize; 8];
		let npc = [4usize, 4];
		let g0 = [1i64, 2, 3, 4, 1, 2, 5, 6];
		let g1 = [1i64, 2, 3, 4, 1, 2, 7, 8];
		let conns: Vec<i64> = g0.iter().chain(g1.iter()).copied().collect();
		let d = DistinctFractions::of_batch(&bits, &npc, &conns, 2, 1);
		assert_eq!(d.per_genome, vec![0.75, 0.75]);
		assert_eq!((d.batch_logical, d.batch_distinct), (8, 4));
		assert!(d.log_line().contains("batch distinct=4 (50.0%)"), "{}", d.log_line());
	}
}
