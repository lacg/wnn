//! Controller ARCHITECTURE operators (connectivity level), in Rust.
//!
//! Companion to memory_ops.rs, which moved the cell-VALUE operators. These are
//! the per-connection / per-neuron loops from control/recurrent_genome.py:
//!
//!   _resample_in_place   -> resample_suffix        (per connection, 8-try retry)
//!   _sample_distinct     -> sample_distinct        (k distinct, no replacement)
//!   _rebalance_features  -> rebalance_features     (data-dependent move loop)
//!   _mix_blocks / crossover_average per-position -> pick_mask
//!
//! All draws come from ram_core::counter_rng, so they are order-independent and
//! identical to the Python mirror. Two Python behaviours are reproduced exactly
//! because they are load-bearing, not incidental:
//!
//!   * resample keeps per-suffix DISTINCTNESS via a bounded 8-try retry; on
//!     failure the ORIGINAL bit is kept (the `for…else` in Python). Retry count
//!     is bounded, so the operator stays a pure function of its coordinates.
//!   * rebalance moves ONE bit per outer iteration and stops when no neuron can
//!     donate — the `moved` flag. Its iteration cap is sum(counts)*4 + 100.
//!
//! NOTE the scalar per-genome gates (grow probability, delta magnitudes, which
//! parent supplies the shape) deliberately stay in Python: they are ~4 draws per
//! genome and are orchestration decisions, not loops. Each Python operator draws
//! ONE numpy seed and hands it here, so the caller's rng chain still determines
//! the outcome.

use ram_core::counter_rng;

/// Feature group of a sampled input-bit index (window-folded, mirrors
/// recurrent_genome._feature_of).
#[inline]
fn feature_of(idx: usize, frame_bits: usize, bpf: usize) -> usize {
	(idx % frame_bits) / bpf
}

/// Per-entry resample of one sampled suffix, preserving distinctness within it.
///
/// Mirrors `_resample_in_place`: for each position, with probability `rate`,
/// try up to 8 random bits and take the first not already used; if all 8 collide,
/// keep the original. Sub-draw coordinates: 0 = the rate gate, 1..=8 = the tries.
pub fn resample_suffix(
	suffix: &mut [i64], space: usize, rate: f64,
	seed: u64, generation: u64, genome: u64, layer: u64,
) {
	if space == 0 {
		return;
	}
	let mut used: std::collections::HashSet<i64> = suffix.iter().copied().collect();
	for j in 0..suffix.len() {
		let jj = j as u64;
		if counter_rng::uniform(seed, generation, genome, layer, jj, 0) >= rate {
			continue;
		}
		used.remove(&suffix[j]);
		let mut placed = false;
		for t in 1..=8u64 {
			let cand = counter_rng::below(space as u64, seed, generation, genome, layer, jj, t) as i64;
			if !used.contains(&cand) {
				suffix[j] = cand;
				used.insert(cand);
				placed = true;
				break;
			}
		}
		if !placed {
			used.insert(suffix[j]); // all tries collided — keep the original
		}
	}
}

/// `k` distinct indices in [0, space) avoiding `exclude`, without replacement.
///
/// Mirrors `_sample_distinct`. Partial Fisher-Yates over the eligible pool: k
/// draws, each picking from the remaining tail, so the result is distinct by
/// construction and needs no rejection.
pub fn sample_distinct(
	space: usize, k: usize, exclude: &[i64],
	seed: u64, generation: u64, genome: u64, layer: u64,
) -> Vec<i64> {
	let ex: std::collections::HashSet<i64> = exclude.iter().copied().collect();
	let mut pool: Vec<i64> = (0..space as i64).filter(|b| !ex.contains(b)).collect();
	let k = k.min(pool.len());
	let mut out = Vec::with_capacity(k);
	for i in 0..k {
		let remaining = pool.len() - i;
		let pick = i + counter_rng::below(remaining as u64, seed, generation, genome, layer, i as u64, 0) as usize;
		pool.swap(i, pick);
		out.push(pool[i]);
	}
	out
}

/// Feature-balance cap. Moves sampled bits from over-represented input features
/// to under-represented ones until no feature exceeds `ratio` x the least-wired
/// feature. `sampled` is flat with `offsets` (len = n_neurons + 1).
///
/// Mirrors `_rebalance_features` including its stopping rules: one move per outer
/// iteration, give up when no neuron can donate, cap at sum(counts)*4 + 100.
#[allow(clippy::too_many_arguments)]
pub fn rebalance_features(
	sampled: &mut [i64], offsets: &[usize], space: usize,
	frame_bits: usize, bpf: usize, ratio: f64,
	seed: u64, generation: u64, genome: u64, layer: u64,
) {
	if ratio <= 1.0 || offsets.len() < 2 || frame_bits == 0 || bpf == 0 {
		return;
	}
	let nfeat = frame_bits / bpf;
	if nfeat <= 1 {
		return;
	}
	let n_neurons = offsets.len() - 1;
	let mut feat_bits: Vec<Vec<i64>> = vec![Vec::new(); nfeat];
	for b in 0..space {
		feat_bits[feature_of(b, frame_bits, bpf)].push(b as i64);
	}
	let mut counts = vec![0usize; nfeat];
	for &b in sampled.iter() {
		counts[feature_of(b as usize, frame_bits, bpf)] += 1;
	}
	let max_iter = counts.iter().sum::<usize>() * 4 + 100;

	for it in 0..max_iter {
		let hi = argmax(&counts);
		let lo = argmin(&counts);
		if counts[hi] as f64 <= ratio * counts[lo].max(1) as f64 {
			break;
		}
		if !move_one(sampled, offsets, n_neurons, &feat_bits, &mut counts, hi, lo,
		             frame_bits, bpf, seed, generation, genome, layer, it as u64) {
			break;
		}
	}
}

/// One rebalancing move: find a neuron holding a `hi`-feature bit and a free
/// `lo`-feature bit, and swap. Neuron visit order is a counter-RNG permutation
/// (the Python used rng.permutation). Returns false when nothing can move.
#[allow(clippy::too_many_arguments)]
fn move_one(
	sampled: &mut [i64], offsets: &[usize], n_neurons: usize,
	feat_bits: &[Vec<i64>], counts: &mut [usize], hi: usize, lo: usize,
	frame_bits: usize, bpf: usize,
	seed: u64, generation: u64, genome: u64, layer: u64, it: u64,
) -> bool {
	let mut order: Vec<usize> = (0..n_neurons).collect();
	for i in 0..n_neurons {
		let pick = i + counter_rng::below((n_neurons - i) as u64, seed, generation, genome, layer, it, i as u64) as usize;
		order.swap(i, pick);
	}
	for ni in order {
		let (s, e) = (offsets[ni], offsets[ni + 1]);
		let hi_pos = (s..e).find(|&k| feature_of(sampled[k] as usize, frame_bits, bpf) == hi);
		let Some(pos) = hi_pos else { continue };
		let cands: Vec<i64> = feat_bits[lo].iter().copied()
			.filter(|b| !sampled[s..e].contains(b)).collect();
		if cands.is_empty() {
			continue;
		}
		let c = counter_rng::below(cands.len() as u64, seed, generation, genome, layer, it, 0xFFFF) as usize;
		sampled[pos] = cands[c];
		counts[hi] -= 1;
		counts[lo] += 1;
		return true;
	}
	false
}

/// First-maximum index (matches numpy argmax tie-breaking).
fn argmax(v: &[usize]) -> usize {
	let mut best = 0;
	for i in 1..v.len() {
		if v[i] > v[best] { best = i; }
	}
	best
}

/// First-minimum index (matches numpy argmin tie-breaking).
fn argmin(v: &[usize]) -> usize {
	let mut best = 0;
	for i in 1..v.len() {
		if v[i] < v[best] { best = i; }
	}
	best
}

/// `n` independent fair coins — the per-position / per-block parent picks in
/// crossover_average and _mix_blocks. True = take from the first parent.
pub fn pick_mask(n: usize, seed: u64, generation: u64, genome: u64, layer: u64) -> Vec<bool> {
	(0..n)
		.map(|i| counter_rng::uniform(seed, generation, genome, layer, i as u64, 0) < 0.5)
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn resample_keeps_entries_distinct_and_in_range() {
		let mut suf: Vec<i64> = (0..32).collect();
		resample_suffix(&mut suf, 256, 1.0, 42, 0, 0, 0);
		let set: std::collections::HashSet<i64> = suf.iter().copied().collect();
		assert_eq!(set.len(), suf.len(), "resample introduced a duplicate");
		assert!(suf.iter().all(|&b| (0..256).contains(&b)), "bit out of space");
	}

	#[test]
	fn resample_rate_zero_is_a_no_op() {
		let before: Vec<i64> = (0..16).collect();
		let mut suf = before.clone();
		resample_suffix(&mut suf, 128, 0.0, 7, 0, 0, 0);
		assert_eq!(suf, before);
	}

	/// space smaller than the suffix: every try collides, so entries must be KEPT
	/// rather than duplicated (the Python for/else path).
	#[test]
	fn resample_survives_a_saturated_space() {
		let mut suf: Vec<i64> = (0..4).collect();
		resample_suffix(&mut suf, 4, 1.0, 3, 0, 0, 0);
		let set: std::collections::HashSet<i64> = suf.iter().copied().collect();
		assert_eq!(set.len(), 4, "saturated space must stay distinct");
	}

	#[test]
	fn sample_distinct_is_distinct_and_honours_exclude() {
		let out = sample_distinct(50, 10, &[0, 1, 2, 3, 4], 9, 0, 0, 0);
		assert_eq!(out.len(), 10);
		let set: std::collections::HashSet<i64> = out.iter().copied().collect();
		assert_eq!(set.len(), 10, "duplicates in sample_distinct");
		assert!(out.iter().all(|&b| b >= 5 && b < 50), "excluded bit was sampled");
	}

	#[test]
	fn sample_distinct_clamps_to_pool() {
		let out = sample_distinct(5, 99, &[], 1, 0, 0, 0);
		assert_eq!(out.len(), 5, "must clamp k to the pool size");
	}

	#[test]
	fn rebalance_reduces_the_worst_imbalance() {
		// 4 features x 8 bits per feature = 32 bits; everything wired to feature 0
		let (frame_bits, bpf) = (32usize, 8usize);
		let mut sampled: Vec<i64> = vec![0, 1, 2, 3, 4, 5, 6, 7];
		let offsets = vec![0usize, 4, 8];
		let counts_before = imbalance(&sampled, frame_bits, bpf);
		rebalance_features(&mut sampled, &offsets, 32, frame_bits, bpf, 1.5, 5, 0, 0, 0);
		let counts_after = imbalance(&sampled, frame_bits, bpf);
		assert!(counts_after < counts_before,
			"rebalance did not reduce imbalance: {counts_before} -> {counts_after}");
		// distinctness per neuron preserved
		for w in offsets.windows(2) {
			let s: std::collections::HashSet<i64> = sampled[w[0]..w[1]].iter().copied().collect();
			assert_eq!(s.len(), w[1] - w[0], "rebalance broke per-neuron distinctness");
		}
	}

	fn imbalance(sampled: &[i64], frame_bits: usize, bpf: usize) -> usize {
		let nfeat = frame_bits / bpf;
		let mut c = vec![0usize; nfeat];
		for &b in sampled {
			c[feature_of(b as usize, frame_bits, bpf)] += 1;
		}
		argmax(&c).max(1);
		c.iter().max().unwrap() - c.iter().min().unwrap()
	}

	#[test]
	fn rebalance_disabled_below_ratio_one() {
		let mut sampled: Vec<i64> = vec![0, 1, 2, 3];
		let before = sampled.clone();
		rebalance_features(&mut sampled, &[0, 4], 32, 32, 8, 1.0, 5, 0, 0, 0);
		assert_eq!(sampled, before, "ratio<=1 must disable");
	}

	#[test]
	fn pick_mask_is_balanced_and_deterministic() {
		let a = pick_mask(10_000, 11, 0, 0, 0);
		let b = pick_mask(10_000, 11, 0, 0, 0);
		assert_eq!(a, b, "pick_mask must be a pure function of its coordinates");
		let t = a.iter().filter(|&&x| x).count() as f64 / a.len() as f64;
		assert!((0.45..0.55).contains(&t), "pick_mask skewed: {t}");
	}
}
