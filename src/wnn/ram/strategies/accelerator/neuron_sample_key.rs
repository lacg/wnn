//! IDS-50 (08/11/2026, worker ABI 25): the neuron-sampling mask is keyed on the
//! neuron's IDENTITY, not its position.
//!
//! At `neuron_sample_rate < 1` every (neuron, example) pair is trained with
//! probability `rate`, decided by a deterministic xorshift hash. That hash used
//! to take the neuron's flat index (`cluster_neuron_starts[c] + n`), so a
//! neurons mutation that trimmed cluster 0, or a pool-shuffle crossover,
//! re-drew the training subset of every neuron behind it: fitness depended on
//! where a neuron sat, not on what it was.
//!
//! The key is now a hash of the neuron's identity (`ram_core::neuron_key`
//! semantics: cluster, address bits, connections in order) plus its OCCURRENCE
//! ORDINAL among identical neurons of the same genome. The ordinal keeps two
//! copies of one neuron as two independent draws (as before); the multiset of
//! keys is invariant under any permutation of a cluster's neurons and under
//! adding/removing other neurons.
//!
//! CPU (`skip_sample`) and GPU (`marker_train.metal::should_skip_sample`) share
//! the hash: `seed + key * 1000003 + ex * 2654435761`, xorshift32. At
//! rate >= 1.0 both return before reading the key — rate 1.0 is bit-identical
//! to every earlier ABI.

use std::collections::HashMap;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Builds the sampling keys of ONE genome's neurons, fed in logical
/// (cluster-major) order. One builder per genome: the duplicate ordinal is
/// genome-scoped.
pub(crate) struct SampleKeyBuilder
{
	seen: HashMap<u64, u64>,
}

impl SampleKeyBuilder
{
	pub(crate) fn new() -> Self
	{
		Self { seen: HashMap::new() }
	}

	/// Key of the next neuron: identity hash, then the k-th-copy ordinal.
	pub(crate) fn key(&mut self, cluster: u32, bits: u32, connections: impl IntoIterator<Item = i64>) -> u32
	{
		let identity = identity_hash(cluster, bits, connections);
		let ordinal = self.seen.entry(identity).or_insert(0);
		let key = finalize(identity ^ ordinal.wrapping_mul(0x9E37_79B9_7F4A_7C15));
		*ordinal += 1;
		key
	}
}

fn fnv(mut h: u64, bytes: &[u8]) -> u64
{
	for &b in bytes
	{
		h ^= b as u64;
		h = h.wrapping_mul(FNV_PRIME);
	}
	h
}

fn identity_hash(cluster: u32, bits: u32, connections: impl IntoIterator<Item = i64>) -> u64
{
	let mut h = fnv(FNV_OFFSET, &cluster.to_le_bytes());
	h = fnv(h, &bits.to_le_bytes());
	for c in connections
	{
		h = fnv(h, &c.to_le_bytes());
	}
	h
}

/// splitmix64 finaliser folded to 32 bits.
fn finalize(mut z: u64) -> u32
{
	z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
	((z ^ (z >> 31)) & 0xFFFF_FFFF) as u32
}

/// Keys of a genome given the per-neuron layout the CPU trainer uses
/// (`build_neuron_metadata`): cluster `c` owns neurons
/// `[cluster_neuron_starts[c], cluster_neuron_starts[c+1])`.
pub(crate) fn keys_from_layout(
	connections: &[i64],
	per_neuron_bits: &[usize],
	cluster_neuron_starts: &[usize],
	neuron_conn_offsets: &[usize],
) -> Vec<u32>
{
	let total = per_neuron_bits.len();
	let mut builder = SampleKeyBuilder::new();
	let mut keys = Vec::with_capacity(total);
	for (c, &start) in cluster_neuron_starts.iter().enumerate()
	{
		let end = cluster_neuron_starts.get(c + 1).copied().unwrap_or(total);
		for n in start..end
		{
			let (lo, bits) = (neuron_conn_offsets[n], per_neuron_bits[n]);
			let conns = connections.get(lo..lo + bits).unwrap_or(&[]);
			keys.push(builder.key(c as u32, bits as u32, conns.iter().copied()));
		}
	}
	keys
}

/// Keys of a flat genome (`bits_flat` per neuron, unpadded `connections`).
pub(crate) fn keys_for_genome(bits_flat: &[usize], neurons_per_cluster: &[usize], connections: &[i64]) -> Vec<u32>
{
	let (starts, offsets) = crate::adaptive::build_neuron_metadata(bits_flat, neurons_per_cluster);
	keys_from_layout(connections, bits_flat, &starts, &offsets)
}

/// CPU twin of `marker_train.metal::should_skip_sample`: true = skip this
/// (neuron, example) pair. Returns before hashing at rate >= 1.0.
#[inline]
pub(crate) fn skip_sample(key: u32, ex_idx: usize, rng_seed: u64, rate: f32) -> bool
{
	if rate >= 1.0
	{
		return false;
	}
	let mut rng = (rng_seed as u32)
		.wrapping_add(key.wrapping_mul(1000003))
		.wrapping_add((ex_idx as u32).wrapping_mul(2654435761));
	if rng == 0
	{
		rng = 1;
	}
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	(rng >> 8) as f32 / 16777216.0 >= rate
}

#[cfg(test)]
#[path = "neuron_sample_key_tests.rs"]
mod tests;
