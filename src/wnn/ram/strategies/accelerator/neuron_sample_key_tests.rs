//! IDS-50 unit tests: the sampling key is neuron IDENTITY, not position.

use super::*;

/// Three clusters of distinct neurons (cluster, bits, connections).
fn genome() -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let npc = vec![3usize, 4, 2];
	let bits: Vec<usize> = vec![4, 5, 4, 6, 4, 4, 5, 3, 4];
	let conns: Vec<i64> = bits.iter().enumerate().flat_map(|(n, &b)| (0..b).map(move |k| ((7 * n + 3 * k) % 40) as i64)).collect();
	(bits, npc, conns)
}

/// The mask a key draws over `examples` (true = trained).
fn mask(key: u32, examples: usize) -> Vec<bool>
{
	(0..examples).map(|ex| !skip_sample(key, ex, 42, 0.25)).collect()
}

/// Neuron `n`'s connections in an unpadded flat genome.
fn neuron_conns<'a>(bits: &[usize], conns: &'a [i64], n: usize) -> &'a [i64]
{
	let lo: usize = bits[..n].iter().sum();
	&conns[lo..lo + bits[n]]
}

#[test]
fn removing_a_cluster0_neuron_keeps_every_other_mask()
{
	let (bits, npc, conns) = genome();
	let before = keys_for_genome(&bits, &npc, &conns);
	// Drop the FIRST neuron of cluster 0 (the trim/crossover shift case).
	let removed_conns: Vec<i64> = conns[bits[0]..].to_vec();
	let after = keys_for_genome(&bits[1..], &[2, 4, 2], &removed_conns);
	assert_eq!(&before[1..], &after[..], "a cluster-0 removal re-keyed other neurons");
	for (b, a) in before[1..].iter().zip(&after)
	{
		assert_eq!(mask(*b, 2000), mask(*a, 2000));
	}
}

#[test]
fn permuting_within_a_cluster_moves_keys_with_their_neurons()
{
	let (bits, npc, conns) = genome();
	let before = keys_for_genome(&bits, &npc, &conns);
	// Reverse cluster 1 (neurons 3..7).
	let order: Vec<usize> = vec![0, 1, 2, 6, 5, 4, 3, 7, 8];
	let p_bits: Vec<usize> = order.iter().map(|&n| bits[n]).collect();
	let p_conns: Vec<i64> = order.iter().flat_map(|&n| neuron_conns(&bits, &conns, n).to_vec()).collect();
	let after = keys_for_genome(&p_bits, &npc, &p_conns);
	for (pos, &n) in order.iter().enumerate()
	{
		assert_eq!(after[pos], before[n], "neuron {} lost its key when moved to {}", n, pos);
	}
}

#[test]
fn duplicates_are_independent_draws_and_order_free()
{
	let mut b = SampleKeyBuilder::new();
	let k1 = b.key(0, 3, [1i64, 2, 3]);
	let k2 = b.key(0, 3, [1i64, 2, 3]);
	assert_ne!(k1, k2, "a duplicate must get its own draw (ordinal)");
	assert_ne!(mask(k1, 2000), mask(k2, 2000));
	assert_ne!(SampleKeyBuilder::new().key(1, 3, [1i64, 2, 3]), k1, "cluster must separate");
	assert_ne!(SampleKeyBuilder::new().key(0, 3, [3i64, 2, 1]), k1, "connection order is identity");
}

#[test]
fn rate_one_never_skips_and_hash_matches_the_legacy_formula()
{
	assert!((0..5000).all(|ex| !skip_sample(12345, ex, 7, 1.0)));
	// LM callers pass key = the old neuron index: same hash as before ABI 25.
	for (n, ex) in [(0usize, 0usize), (3, 17), (199, 4096)]
	{
		let mut rng = 42u32.wrapping_add((n as u32).wrapping_mul(1000003)).wrapping_add((ex as u32).wrapping_mul(2654435761));
		if rng == 0
		{
			rng = 1;
		}
		rng ^= rng << 13;
		rng ^= rng >> 17;
		rng ^= rng << 5;
		let legacy = (rng >> 8) as f32 / 16777216.0 >= 0.25;
		assert_eq!(skip_sample(n as u32, ex, 42, 0.25), legacy);
	}
}

#[test]
fn layout_keys_equal_flat_keys()
{
	let (bits, npc, conns) = genome();
	let (starts, offsets) = crate::adaptive::build_neuron_metadata(&bits, &npc);
	assert_eq!(keys_from_layout(&conns, &bits, &starts, &offsets), keys_for_genome(&bits, &npc, &conns));
}
