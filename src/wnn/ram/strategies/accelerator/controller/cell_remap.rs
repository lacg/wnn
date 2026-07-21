// ============================================================================
// Cell address remapping — Rust port of the helpers in
// src/wnn/control/recurrent_genome.py (_remap_* / _drop_* / _majority).
//
// When a genome's ARCHITECTURE changes, every trained cell's address has to be
// rewritten: growing the sampled-bit width appends LSBs, shrinking collapses
// colliders by majority vote, state neurogenesis inserts or removes prefix bit
// fields, and a connectivity resample invalidates whole neurons. The Python
// originals do this as per-cell loops building lists of tuples; at the measured
// 5-13.6M cells per genome that is the last cell operation with no Rust
// equivalent, and therefore the blocker for keeping cells inside Rust.
//
// BIT-EXACTNESS IS THE CONTRACT. These run inside the GA, so any divergence
// from the Python semantics changes genome lineage and silently invalidates
// comparisons against every prior controller result. Two subtleties carry that
// risk and are easy to get wrong:
//
//   1. OUTPUT ORDER IS FIRST-ENCOUNTER, NOT SORTED. The Python collapses use a
//      dict and then `list(buckets.keys())`, so surviving cells appear in the
//      order their bucket was first created while scanning the input. Values are
//      positional and fingerprint() hashes the raw buffers, so a plain HashMap
//      here would reorder the universe and change genome identity. We reproduce
//      dict ordering with a key->index map plus a Vec.
//
//   2. TIES GO TO THE LOWER VALUE. See `majority`. The Python form had a uint8
//      wraparound bug (fixed 21/07/2026) that inverted this; expressing the
//      tie-break on u8 keys directly makes that class of bug impossible.
//
// Addresses are computed in u128 because a grow can shift past u64 (the Python
// side raises OverflowError from MemoryPayload when that happens). We surface it
// as an explicit error rather than wrapping silently.
// ============================================================================

use std::collections::HashMap;

/// Per-neuron neutral feedback for the legacy QSR pair (recurrent_genome.NEUTRAL_PAIR).
pub const NEUTRAL_PAIR: u128 = 2;

/// Cells in column form: parallel neuron / address / value arrays.
pub type Cells = (Vec<u32>, Vec<u64>, Vec<u8>);

/// Most common value among colliders; ties resolve to the LOWER value.
///
/// Values are QSR 0..3, so a 4-slot tally beats a hash map. Iterating 0..=3
/// ascending and taking strictly-greater counts gives "ties -> lower" for free.
pub fn majority(vals: &[u8]) -> u8 {
	let mut tally = [0u32; 256];
	for &v in vals {
		tally[v as usize] += 1;
	}
	let mut best_v = 0u8;
	let mut best_c = 0u32;
	for v in 0..=255usize {
		if tally[v] > best_c {
			best_c = tally[v];
			best_v = v as u8;
		}
	}
	best_v
}

/// Address overflowed u64 during a remap (Python raises OverflowError here).
#[derive(Debug)]
pub struct AddrOverflow(pub u128);

fn fit_u64(a: u128) -> Result<u64, AddrOverflow> {
	if a > u64::MAX as u128 { Err(AddrOverflow(a)) } else { Ok(a as u64) }
}

/// Insertion-ordered bucketing: returns (keys in first-encounter order,
/// per-key collected values). This is exactly Python's dict-of-lists +
/// `list(buckets.keys())` idiom, and the ordering is load-bearing (see header).
fn bucket_ordered(
	neurons: &[u32],
	keys: &[u64],
	values: &[u8],
) -> (Vec<(u32, u64)>, Vec<Vec<u8>>) {
	let mut index: HashMap<(u32, u64), usize> = HashMap::with_capacity(neurons.len());
	let mut order: Vec<(u32, u64)> = Vec::with_capacity(neurons.len());
	let mut buckets: Vec<Vec<u8>> = Vec::with_capacity(neurons.len());
	for i in 0..neurons.len() {
		let k = (neurons[i], keys[i]);
		match index.get(&k) {
			Some(&slot) => buckets[slot].push(values[i]),
			None => {
				index.insert(k, order.len());
				order.push(k);
				buckets.push(vec![values[i]]);
			}
		}
	}
	(order, buckets)
}

fn collapse(order: Vec<(u32, u64)>, buckets: Vec<Vec<u8>>) -> Cells {
	let n = order.len();
	let mut on = Vec::with_capacity(n);
	let mut oa = Vec::with_capacity(n);
	let mut ov = Vec::with_capacity(n);
	for (i, (neuron, addr)) in order.into_iter().enumerate() {
		on.push(neuron);
		oa.push(addr);
		ov.push(majority(&buckets[i]));
	}
	(on, oa, ov)
}

/// BITS grow by `d` LSBs: A -> A*2^d + child, value REPLICATED to all 2^d
/// children (behaviour-preserving: the new low bits do not change the read).
pub fn remap_grow(neurons: &[u32], addrs: &[u64], values: &[u8], d: u32)
	-> Result<Cells, AddrOverflow>
{
	if d == 0 {
		return Ok((neurons.to_vec(), addrs.to_vec(), values.to_vec()));
	}
	let fanout = 1usize << d;
	let total = neurons.len() * fanout;
	let mut on = Vec::with_capacity(total);
	let mut oa = Vec::with_capacity(total);
	let mut ov = Vec::with_capacity(total);
	for i in 0..neurons.len() {
		let base = (addrs[i] as u128) << d;
		for child in 0..fanout as u128 {
			on.push(neurons[i]);
			oa.push(fit_u64(base | child)?);
			ov.push(values[i]);
		}
	}
	Ok((on, oa, ov))
}

/// BITS shrink by `d` LSBs: A -> A >> d, colliders resolved by majority vote.
pub fn remap_shrink(neurons: &[u32], addrs: &[u64], values: &[u8], d: u32) -> Cells {
	if d == 0 {
		return (neurons.to_vec(), addrs.to_vec(), values.to_vec());
	}
	let keys: Vec<u64> = addrs.iter().map(|&a| a >> d).collect();
	let (order, buckets) = bucket_ordered(neurons, &keys, values);
	collapse(order, buckets)
}

/// STATE neurogenesis +k: the prefix gains `pf*k` mid-bits just above the w-bit
/// suffix, defaulting to the per-neuron neutral feedback, so behaviour is
/// preserved on the neutral branch.
pub fn remap_prefix_grow(
	neurons: &[u32], addrs: &[u64], values: &[u8], k: u32, w: u32, pf: u32,
) -> Result<Cells, AddrOverflow> {
	if k == 0 {
		return Ok((neurons.to_vec(), addrs.to_vec(), values.to_vec()));
	}
	let mask: u128 = (1u128 << w) - 1;
	let per: u128 = if pf == 2 { NEUTRAL_PAIR } else { 0 };
	let mut neutral: u128 = 0;
	for j in 0..k {
		neutral |= per << (pf * j);
	}
	let shift = pf * k + w;
	let mut on = Vec::with_capacity(neurons.len());
	let mut oa = Vec::with_capacity(addrs.len());
	for i in 0..neurons.len() {
		let a = addrs[i] as u128;
		let (p, s) = (a >> w, a & mask);
		on.push(neurons[i]);
		oa.push(fit_u64((p << shift) | (neutral << w) | s)?);
	}
	Ok((on, oa, values.to_vec()))
}

/// STATE neurogenesis -k: drop the lowest `pf*k` prefix bits, majority collapse.
pub fn remap_prefix_shrink(
	neurons: &[u32], addrs: &[u64], values: &[u8], k: u32, w: u32, pf: u32,
) -> Cells {
	if k == 0 {
		return (neurons.to_vec(), addrs.to_vec(), values.to_vec());
	}
	let mask: u64 = (1u64 << w) - 1;
	let drop = pf * k + w;
	let keys: Vec<u64> = addrs.iter()
		.map(|&a| {
			let p_high = if drop >= 64 { 0 } else { a >> drop };
			(p_high << w) | (a & mask)
		})
		.collect();
	let (order, buckets) = bucket_ordered(neurons, &keys, values);
	collapse(order, buckets)
}

/// Excise `nbits` adjacent address bits starting at `p_lsb` (delete a mid-address
/// field), majority-collapsing collisions. Used by surgical state-neuron removal.
pub fn remap_delete_bit_window(
	neurons: &[u32], addrs: &[u64], values: &[u8], p_lsb: u32, nbits: u32,
) -> Cells {
	let mask_low: u64 = if p_lsb == 0 { 0 } else { (1u64 << p_lsb) - 1 };
	let hi_shift = p_lsb + nbits;
	let keys: Vec<u64> = addrs.iter()
		.map(|&a| {
			let low = a & mask_low;
			let high = if hi_shift >= 64 { 0 } else { a >> hi_shift };
			(high << p_lsb) | low
		})
		.collect();
	let (order, buckets) = bucket_ordered(neurons, &keys, values);
	collapse(order, buckets)
}

/// Drop cells whose neuron index >= `limit` (the neurons being removed).
pub fn drop_neurons_ge(neurons: &[u32], addrs: &[u64], values: &[u8], limit: u32) -> Cells {
	let mut on = Vec::new();
	let mut oa = Vec::new();
	let mut ov = Vec::new();
	for i in 0..neurons.len() {
		if neurons[i] < limit {
			on.push(neurons[i]);
			oa.push(addrs[i]);
			ov.push(values[i]);
		}
	}
	(on, oa, ov)
}

/// CONNECTIONS remap: drop cells of neurons whose sampled suffix changed (their
/// address semantics are scrambled); keep the rest verbatim.
pub fn drop_changed_neurons(
	neurons: &[u32], addrs: &[u64], values: &[u8], changed: &[u32],
) -> Cells {
	let set: std::collections::HashSet<u32> = changed.iter().copied().collect();
	let mut on = Vec::new();
	let mut oa = Vec::new();
	let mut ov = Vec::new();
	for i in 0..neurons.len() {
		if !set.contains(&neurons[i]) {
			on.push(neurons[i]);
			oa.push(addrs[i]);
			ov.push(values[i]);
		}
	}
	(on, oa, ov)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn majority_ties_go_to_lower_value() {
		assert_eq!(majority(&[3, 3, 1]), 3);
		assert_eq!(majority(&[0, 1]), 0);
		assert_eq!(majority(&[0, 1, 2, 3]), 0);
		assert_eq!(majority(&[2, 3]), 2);
		// the case the Python uint8 wraparound got wrong: 0 vs 3 tie -> 0
		assert_eq!(majority(&[0, 3]), 0);
	}

	#[test]
	fn shrink_preserves_first_encounter_order() {
		// 8>>2=2, 12>>2=3, 9>>2=2 (collides with the FIRST bucket)
		let (n, a, v) = remap_shrink(&[0, 0, 0, 1], &[8, 12, 9, 4], &[1, 2, 3, 0], 2);
		assert_eq!(n, vec![0, 0, 1]);
		assert_eq!(a, vec![2, 3, 1]);      // NOT sorted -- first-encounter
		assert_eq!(v, vec![1, 2, 0]);      // bucket {1,3} ties -> lower = 1
	}

	#[test]
	fn grow_replicates_across_children() {
		let (n, a, v) = remap_grow(&[0, 1], &[5, 0], &[3, 1], 2).unwrap();
		assert_eq!(n, vec![0, 0, 0, 0, 1, 1, 1, 1]);
		assert_eq!(a, vec![20, 21, 22, 23, 0, 1, 2, 3]);
		assert_eq!(v, vec![3, 3, 3, 3, 1, 1, 1, 1]);
	}

	#[test]
	fn grow_overflow_is_reported_not_wrapped() {
		assert!(remap_grow(&[0], &[1u64 << 63], &[3], 2).is_err());
	}

	#[test]
	fn drops_filter_by_neuron() {
		let (n, _a, v) = drop_neurons_ge(&[0, 1, 2], &[1, 2, 3], &[1, 2, 3], 2);
		assert_eq!(n, vec![0, 1]);
		assert_eq!(v, vec![1, 2]);
		let (n2, _a2, v2) = drop_changed_neurons(&[0, 1, 2], &[1, 2, 3], &[1, 2, 3], &[1]);
		assert_eq!(n2, vec![0, 2]);
		assert_eq!(v2, vec![1, 3]);
	}
}
