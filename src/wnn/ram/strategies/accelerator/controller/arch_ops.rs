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
fn feature_of(idx: usize, frame_bits: usize, bpf: usize) -> usize
{
	(idx % frame_bits) / bpf
}

/// Per-entry resample of one sampled suffix, preserving distinctness within it.
///
/// Mirrors `_resample_in_place`: for each position, with probability `rate`,
/// try up to 8 random bits and take the first not already used; if all 8 collide,
/// keep the original. Sub-draw coordinates: 0 = the rate gate, 1..=8 = the tries.
/// Scope 0 (free) — bit-identical to the pre-16/08 behaviour.
pub fn resample_suffix(
	suffix: &mut [i64],
	space: usize,
	rate: f64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
)
{
	resample_suffix_scoped(
		suffix, space, rate, 0, 0, 0, seed, generation, genome, layer,
	);
}

/// Scoped axonogenesis (16/08/2026, Luiz's GA-connectivity types). Each mutated
/// connection's replacement candidate is drawn from a range determined by the
/// ORIGINAL bit and the scope:
///
///   0 FREE     — anywhere in [0, space): can leave the feature, the window,
///                everything (the legacy draw, bit-identical coordinates).
///   1 WINDOW   — the original bit's window [w·frame_bits, (w+1)·frame_bits):
///                rewiring explores features but never crosses time. At k=1
///                (frame_bits == space) this degenerates to FREE.
///   2 FEATURE  — the original bit's thermometer run [f·bpf, f·bpf+bpf) inside
///                its window: only WHERE on the feature moves; the neuron's
///                feature map (and window) is frozen at what init/grid chose.
///
/// The feature run is computed on the FLAT index, so multi-window spaces keep
/// window purity for free (bpf divides frame_bits divides space).
#[allow(clippy::too_many_arguments)]
pub fn resample_suffix_scoped(
	suffix: &mut [i64],
	space: usize,
	rate: f64,
	scope: u32,
	frame_bits: usize,
	bpf: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
)
{
	if space == 0
	{
		return;
	}
	let mut used: std::collections::HashSet<i64> = suffix.iter().copied().collect();
	for j in 0..suffix.len()
	{
		let jj = j as u64;
		if counter_rng::uniform(seed, generation, genome, layer, jj, 0) >= rate
		{
			continue;
		}
		let old = suffix[j] as usize;
		let (lo, len) = match scope
		{
			1 if frame_bits > 0 => ((old / frame_bits) * frame_bits, frame_bits),
			2 if bpf > 0 => ((old / bpf) * bpf, bpf),
			_ => (0, space),
		};
		used.remove(&suffix[j]);
		let mut placed = false;
		for t in 1..=8u64
		{
			let cand = (lo
				+ counter_rng::below(len as u64, seed, generation, genome, layer, jj, t) as usize)
				as i64;
			if !used.contains(&cand)
			{
				suffix[j] = cand;
				used.insert(cand);
				placed = true;
				break;
			}
		}
		if !placed
		{
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
	space: usize,
	k: usize,
	exclude: &[i64],
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	let ex: std::collections::HashSet<i64> = exclude.iter().copied().collect();
	let mut pool: Vec<i64> = (0..space as i64).filter(|b| !ex.contains(b)).collect();
	let k = k.min(pool.len());
	let mut out = Vec::with_capacity(k);
	for i in 0..k
	{
		let remaining = pool.len() - i;
		let pick = i
			+ counter_rng::below(
				remaining as u64,
				seed,
				generation,
				genome,
				layer,
				i as u64,
				0,
			) as usize;
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
	sampled: &mut [i64],
	offsets: &[usize],
	space: usize,
	frame_bits: usize,
	bpf: usize,
	ratio: f64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
)
{
	if ratio <= 1.0 || offsets.len() < 2 || frame_bits == 0 || bpf == 0
	{
		return;
	}
	let nfeat = frame_bits / bpf;
	if nfeat <= 1
	{
		return;
	}
	let n_neurons = offsets.len() - 1;
	let mut feat_bits: Vec<Vec<i64>> = vec![Vec::new(); nfeat];
	for b in 0..space
	{
		feat_bits[feature_of(b, frame_bits, bpf)].push(b as i64);
	}
	let mut counts = vec![0usize; nfeat];
	for &b in sampled.iter()
	{
		counts[feature_of(b as usize, frame_bits, bpf)] += 1;
	}
	let max_iter = counts.iter().sum::<usize>() * 4 + 100;

	for it in 0..max_iter
	{
		let hi = argmax(&counts);
		let lo = argmin(&counts);
		if counts[hi] as f64 <= ratio * counts[lo].max(1) as f64
		{
			break;
		}
		if !move_one(
			sampled,
			offsets,
			n_neurons,
			&feat_bits,
			&mut counts,
			hi,
			lo,
			frame_bits,
			bpf,
			seed,
			generation,
			genome,
			layer,
			it as u64,
		)
		{
			break;
		}
	}
}

/// One rebalancing move: find a neuron holding a `hi`-feature bit and a free
/// `lo`-feature bit, and swap. Neuron visit order is a counter-RNG permutation
/// (the Python used rng.permutation). Returns false when nothing can move.
#[allow(clippy::too_many_arguments)]
fn move_one(
	sampled: &mut [i64],
	offsets: &[usize],
	n_neurons: usize,
	feat_bits: &[Vec<i64>],
	counts: &mut [usize],
	hi: usize,
	lo: usize,
	frame_bits: usize,
	bpf: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
	it: u64,
) -> bool
{
	let mut order: Vec<usize> = (0..n_neurons).collect();
	for i in 0..n_neurons
	{
		let pick = i
			+ counter_rng::below(
				(n_neurons - i) as u64,
				seed,
				generation,
				genome,
				layer,
				it,
				i as u64,
			) as usize;
		order.swap(i, pick);
	}
	for ni in order
	{
		let (s, e) = (offsets[ni], offsets[ni + 1]);
		let hi_pos = (s..e).find(|&k| feature_of(sampled[k] as usize, frame_bits, bpf) == hi);
		let Some(pos) = hi_pos
		else
		{
			continue;
		};
		let cands: Vec<i64> = feat_bits[lo]
			.iter()
			.copied()
			.filter(|b| !sampled[s..e].contains(b))
			.collect();
		if cands.is_empty()
		{
			continue;
		}
		let c = counter_rng::below(
			cands.len() as u64,
			seed,
			generation,
			genome,
			layer,
			it,
			0xFFFF,
		) as usize;
		sampled[pos] = cands[c];
		counts[hi] -= 1;
		counts[lo] += 1;
		return true;
	}
	false
}

/// First-maximum index (matches numpy argmax tie-breaking).
fn argmax(v: &[usize]) -> usize
{
	let mut best = 0;
	for i in 1..v.len()
	{
		if v[i] > v[best]
		{
			best = i;
		}
	}
	best
}

/// First-minimum index (matches numpy argmin tie-breaking).
fn argmin(v: &[usize]) -> usize
{
	let mut best = 0;
	for i in 1..v.len()
	{
		if v[i] < v[best]
		{
			best = i;
		}
	}
	best
}

/// `n` independent fair coins — the per-position / per-block parent picks in
/// crossover_average and _mix_blocks. True = take from the first parent.
pub fn pick_mask(n: usize, seed: u64, generation: u64, genome: u64, layer: u64) -> Vec<bool>
{
	(0..n)
		.map(|i| counter_rng::uniform(seed, generation, genome, layer, i as u64, 0) < 0.5)
		.collect()
}

/// One fresh suffix under MIN_PER_CLUSTER(m) — the Rust home of
/// `_sample_min_per_cluster` (ported 16/08/2026 same-day per rust-first).
///
/// Choose width/m features, give each m distinct thresholds, DONATE the
/// remainder one bit each to already-chosen features (capped at bpf; never
/// opening a new feature below m). m=1 is the COVERAGE end: at width == nfeat
/// every feature gets exactly one threshold; at width == 2*nfeat exactly two.
/// Falls back to sample_distinct when the space has no feature structure or
/// the request is unsatisfiable — the fallback DECISION lives here, not in
/// Python, so there is exactly one implementation of the rule.
///
/// Sub-draw coordinates: feature permutation (i, 0x10); thresholds for chosen
/// slot j (0x2000 + j, t) — disjoint from every other operator in this module.
pub fn sample_min_per_cluster(
	space: usize,
	width: usize,
	bpf: usize,
	m: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	let nfeat = if bpf > 0 { space / bpf } else { 0 };
	if m < 1 || nfeat <= 1 || width > nfeat * bpf
	{
		return sample_distinct(space, width, &[], seed, generation, genome, layer);
	}
	let n_take = (width / m).min(nfeat);
	// Partial Fisher-Yates permutation of the features; take the first n_take.
	let mut feats: Vec<usize> = (0..nfeat).collect();
	for i in 0..n_take
	{
		let pick = i
			+ counter_rng::below(
				(nfeat - i) as u64,
				seed,
				generation,
				genome,
				layer,
				i as u64,
				0x10,
			) as usize;
		feats.swap(i, pick);
	}
	let mut counts = vec![m; n_take];
	let extras = width - m * n_take;
	for i in 0..extras
	{
		let j = i % n_take;
		if counts[j] < bpf
		{
			counts[j] += 1;
		}
	}
	let mut out = Vec::with_capacity(width);
	for (j, (&f, &c)) in feats[..n_take].iter().zip(counts.iter()).enumerate()
	{
		// c distinct thresholds of bpf via partial Fisher-Yates.
		let mut thr: Vec<usize> = (0..bpf).collect();
		for t in 0..c
		{
			let pick = t
				+ counter_rng::below(
					(bpf - t) as u64,
					seed,
					generation,
					genome,
					layer,
					0x2000 + j as u64,
					t as u64,
				) as usize;
			thr.swap(t, pick);
			out.push((f * bpf + thr[t]) as i64);
		}
	}
	out
}

/// One fresh suffix under FRAMED1 — the Rust home of `_sample_framed1`.
///
/// Each neuron picks ONE frame of the K-window and min1-covers it; the
/// POPULATION covers time. `slot < 0` = draw the frame with recency weights
/// 2^s (slot 0 oldest, k-1 current — 8:4:2:1 at k=4); `slot >= 0` = the caller
/// (genome init) supplies it from an exact quota schedule. Degenerate k<=1 is
/// min1 over the single frame; unsatisfiable widths fall back to spread.
/// Sub-draw coordinates: weighted slot draw (0xF0, 0).
#[allow(clippy::too_many_arguments)]
pub fn sample_framed1(
	space: usize,
	width: usize,
	bpf: usize,
	k: usize,
	slot: i64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	let nfeat_total = if bpf > 0 { space / bpf } else { 0 };
	if k <= 1 || nfeat_total <= 1
	{
		return sample_min_per_cluster(space, width, bpf, 1, seed, generation, genome, layer);
	}
	let frame_bits = space / k;
	let nfeat = if bpf > 0 { frame_bits / bpf } else { 0 };
	if frame_bits == 0 || nfeat <= 1 || width > nfeat * bpf
	{
		return sample_distinct(space, width, &[], seed, generation, genome, layer);
	}
	let s = if slot >= 0
	{
		(slot as usize).min(k - 1)
	}
	else
	{
		// Weighted draw: P(slot) ∝ 2^slot. Walk the cumulative mass once.
		let total = (1u64 << k) - 1; // sum of 2^0..2^(k-1)
		let u = counter_rng::uniform(seed, generation, genome, layer, 0xF0, 0) * total as f64;
		let mut acc = 0f64;
		let mut chosen = k - 1;
		for cand in 0..k
		{
			acc += (1u64 << cand) as f64;
			if u < acc
			{
				chosen = cand;
				break;
			}
		}
		chosen
	};
	let base = (s * frame_bits) as i64;
	sample_min_per_cluster(frame_bits, width, bpf, 1, seed, generation, genome, layer)
		.into_iter()
		.map(|b| base + b)
		.collect()
}

/// EXACT frame-slot quotas for a fresh framed1 population — the Rust home of
/// `_framed1_slot_schedule` (Luiz's round-2 FQ spec: 128/64/32/16 at 240n,
/// 32/16/8/4 per motor block, deterministic).
///
/// Neurons are motor-major (`quantum` = num_motors), so the quota is computed
/// PER MOTOR BLOCK (largest-remainder over weights 2^s, remainder ties toward
/// NEWER frames) and shuffled within each block so no index-keyed structure
/// leaves a motor commanding on stale state. Non-divisible n falls back to a
/// single block. Shuffle coordinates: (block_start + i, 0x30).
pub fn framed1_slot_schedule(
	n_neurons: usize,
	k: usize,
	quantum: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	if k <= 1 || n_neurons == 0
	{
		return vec![0; n_neurons];
	}
	let quantum = quantum.max(1);
	let block = if n_neurons % quantum == 0
	{
		n_neurons / quantum
	}
	else
	{
		n_neurons
	};
	let total_w = ((1u64 << k) - 1) as f64;
	let mut schedule: Vec<i64> = Vec::with_capacity(n_neurons);
	let mut start = 0usize;
	while start < n_neurons
	{
		let size = (n_neurons - start).min(block);
		// Largest-remainder apportionment of `size` over weights 2^s.
		let exact: Vec<f64> = (0..k)
			.map(|s| (1u64 << s) as f64 / total_w * size as f64)
			.collect();
		let mut counts: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
		let assigned: usize = counts.iter().sum();
		let mut order: Vec<usize> = (0..k).collect();
		// Sort by remainder desc, ties toward newer frames (higher slot).
		order.sort_by(|&a, &b| {
			let ra = exact[a] - counts[a] as f64;
			let rb = exact[b] - counts[b] as f64;
			rb.partial_cmp(&ra).unwrap().then(b.cmp(&a))
		});
		for i in 0..(size - assigned)
		{
			counts[order[i % k]] += 1;
		}
		let mut slots: Vec<i64> = Vec::with_capacity(size);
		for (s, &c) in counts.iter().enumerate()
		{
			slots.extend(std::iter::repeat(s as i64).take(c));
		}
		// Fisher-Yates within the block.
		for i in 0..size
		{
			let pick = i
				+ counter_rng::below(
					(size - i) as u64,
					seed,
					generation,
					genome,
					layer,
					(start + i) as u64,
					0x30,
				) as usize;
			slots.swap(i, pick);
		}
		schedule.extend(slots);
		start += size;
	}
	schedule
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn resample_keeps_entries_distinct_and_in_range()
	{
		let mut suf: Vec<i64> = (0..32).collect();
		resample_suffix(&mut suf, 256, 1.0, 42, 0, 0, 0);
		let set: std::collections::HashSet<i64> = suf.iter().copied().collect();
		assert_eq!(set.len(), suf.len(), "resample introduced a duplicate");
		assert!(
			suf.iter().all(|&b| (0..256).contains(&b)),
			"bit out of space"
		);
	}

	#[test]
	fn resample_rate_zero_is_a_no_op()
	{
		let before: Vec<i64> = (0..16).collect();
		let mut suf = before.clone();
		resample_suffix(&mut suf, 128, 0.0, 7, 0, 0, 0);
		assert_eq!(suf, before);
	}

	/// space smaller than the suffix: every try collides, so entries must be KEPT
	/// rather than duplicated (the Python for/else path).
	#[test]
	fn resample_survives_a_saturated_space()
	{
		let mut suf: Vec<i64> = (0..4).collect();
		resample_suffix(&mut suf, 4, 1.0, 3, 0, 0, 0);
		let set: std::collections::HashSet<i64> = suf.iter().copied().collect();
		assert_eq!(set.len(), 4, "saturated space must stay distinct");
	}

	#[test]
	fn sample_distinct_is_distinct_and_honours_exclude()
	{
		let out = sample_distinct(50, 10, &[0, 1, 2, 3, 4], 9, 0, 0, 0);
		assert_eq!(out.len(), 10);
		let set: std::collections::HashSet<i64> = out.iter().copied().collect();
		assert_eq!(set.len(), 10, "duplicates in sample_distinct");
		assert!(
			out.iter().all(|&b| b >= 5 && b < 50),
			"excluded bit was sampled"
		);
	}

	#[test]
	fn sample_distinct_clamps_to_pool()
	{
		let out = sample_distinct(5, 99, &[], 1, 0, 0, 0);
		assert_eq!(out.len(), 5, "must clamp k to the pool size");
	}

	#[test]
	fn rebalance_reduces_the_worst_imbalance()
	{
		// 4 features x 8 bits per feature = 32 bits; everything wired to feature 0
		let (frame_bits, bpf) = (32usize, 8usize);
		let mut sampled: Vec<i64> = vec![0, 1, 2, 3, 4, 5, 6, 7];
		let offsets = vec![0usize, 4, 8];
		let counts_before = imbalance(&sampled, frame_bits, bpf);
		rebalance_features(&mut sampled, &offsets, 32, frame_bits, bpf, 1.5, 5, 0, 0, 0);
		let counts_after = imbalance(&sampled, frame_bits, bpf);
		assert!(
			counts_after < counts_before,
			"rebalance did not reduce imbalance: {counts_before} -> {counts_after}"
		);
		// distinctness per neuron preserved
		for w in offsets.windows(2)
		{
			let s: std::collections::HashSet<i64> = sampled[w[0]..w[1]].iter().copied().collect();
			assert_eq!(
				s.len(),
				w[1] - w[0],
				"rebalance broke per-neuron distinctness"
			);
		}
	}

	fn imbalance(sampled: &[i64], frame_bits: usize, bpf: usize) -> usize
	{
		let nfeat = frame_bits / bpf;
		let mut c = vec![0usize; nfeat];
		for &b in sampled
		{
			c[feature_of(b as usize, frame_bits, bpf)] += 1;
		}
		argmax(&c).max(1);
		c.iter().max().unwrap() - c.iter().min().unwrap()
	}

	#[test]
	fn rebalance_disabled_below_ratio_one()
	{
		let mut sampled: Vec<i64> = vec![0, 1, 2, 3];
		let before = sampled.clone();
		rebalance_features(&mut sampled, &[0, 4], 32, 32, 8, 1.0, 5, 0, 0, 0);
		assert_eq!(sampled, before, "ratio<=1 must disable");
	}

	#[test]
	fn pick_mask_is_balanced_and_deterministic()
	{
		let a = pick_mask(10_000, 11, 0, 0, 0);
		let b = pick_mask(10_000, 11, 0, 0, 0);
		assert_eq!(a, b, "pick_mask must be a pure function of its coordinates");
		let t = a.iter().filter(|&&x| x).count() as f64 / a.len() as f64;
		assert!((0.45..0.55).contains(&t), "pick_mask skewed: {t}");
	}

	// ---- min_per_cluster / framed1 / quota schedule (16/08/2026 port) --------

	const BPF: usize = 8;
	const NFEAT: usize = 18;
	const FRAME: usize = NFEAT * BPF; // 144
	const K: usize = 4;
	const SPACE: usize = K * FRAME; // 576

	/// thresholds-per-feature histogram of one suffix within its frame
	fn coverage(
		suffix: &[i64],
		frame_bits: usize,
		bpf: usize,
	) -> std::collections::HashMap<usize, usize>
	{
		let mut per_feat = std::collections::HashMap::new();
		for &b in suffix
		{
			*per_feat
				.entry(feature_of(b as usize, frame_bits, bpf))
				.or_insert(0usize) += 1;
		}
		let mut hist = std::collections::HashMap::new();
		for (_, c) in per_feat
		{
			*hist.entry(c).or_insert(0usize) += 1;
		}
		hist
	}

	#[test]
	fn min_per_cluster_width_dose_is_exact()
	{
		// The FQ dose axis: b18 = 18x1, b36 = 18x2, b30 = 12x2 + 6x1.
		for (width, want) in [
			(18usize, vec![(1usize, 18usize)]),
			(36, vec![(2, 18)]),
			(30, vec![(2, 12), (1, 6)]),
		]
		{
			let s = sample_min_per_cluster(FRAME, width, BPF, 1, 99, 0, 0, 0);
			assert_eq!(s.len(), width);
			let set: std::collections::HashSet<i64> = s.iter().copied().collect();
			assert_eq!(set.len(), width, "duplicates at width {width}");
			let hist = coverage(&s, FRAME, BPF);
			for (c, n) in want
			{
				assert_eq!(hist.get(&c), Some(&n), "width {width}: coverage {hist:?}");
			}
		}
	}

	#[test]
	fn min_per_cluster_m2_drops_features_and_donates()
	{
		// b=30 at m=2: 15 features x 2, 3 features dropped (the C2 semantics).
		let s = sample_min_per_cluster(FRAME, 30, BPF, 2, 7, 0, 0, 0);
		assert_eq!(s.len(), 30);
		let hist = coverage(&s, FRAME, BPF);
		assert_eq!(
			hist.get(&2),
			Some(&15),
			"m=2 b=30 must be 15 features x 2: {hist:?}"
		);
	}

	#[test]
	fn min_per_cluster_unsatisfiable_falls_back_to_spread()
	{
		// width > nfeat*bpf inside one frame -> spread over the space, still distinct
		let s = sample_min_per_cluster(16, 12, 8, 1, 3, 0, 0, 0); // nfeat=2, cap 16
		assert_eq!(s.len(), 12);
		let set: std::collections::HashSet<i64> = s.iter().copied().collect();
		assert_eq!(set.len(), 12);
	}

	#[test]
	fn framed1_is_frame_pure_and_honours_slot()
	{
		for slot in 0..K as i64
		{
			let s = sample_framed1(SPACE, 18, BPF, K, slot, 5, 0, 0, 0);
			assert_eq!(s.len(), 18);
			let frames: std::collections::HashSet<usize> =
				s.iter().map(|&b| b as usize / FRAME).collect();
			assert_eq!(frames, [slot as usize].into(), "slot {slot} not honoured");
			assert_eq!(
				coverage(&s, FRAME, BPF).get(&1),
				Some(&18),
				"not min1 within frame"
			);
		}
	}

	#[test]
	fn framed1_weighted_draw_prefers_recent_frames()
	{
		// slot=-1: over many seeds the draw must be frame-pure each time and
		// newest-heavy in aggregate (2^slot weights => slot 3 ~ 53%).
		let mut per_slot = [0usize; K];
		for seed in 0..2000u64
		{
			let s = sample_framed1(SPACE, 18, BPF, K, -1, seed, 0, 0, 0);
			let frames: std::collections::HashSet<usize> =
				s.iter().map(|&b| b as usize / FRAME).collect();
			assert_eq!(frames.len(), 1, "mixed frames at seed {seed}");
			per_slot[*frames.iter().next().unwrap()] += 1;
		}
		assert!(
			per_slot[3] > per_slot[2] && per_slot[2] > per_slot[1] && per_slot[1] > per_slot[0],
			"recency ordering violated: {per_slot:?}"
		);
		let newest = per_slot[3] as f64 / 2000.0;
		assert!(
			(0.48..0.59).contains(&newest),
			"newest share {newest} far from 8/15"
		);
	}

	#[test]
	fn framed1_k1_degenerates_to_min1()
	{
		let s = sample_framed1(FRAME, 18, BPF, 1, -1, 2, 0, 0, 0);
		assert_eq!(coverage(&s, FRAME, BPF).get(&1), Some(&18));
	}

	#[test]
	fn slot_schedule_quotas_exact_global_and_per_motor()
	{
		let sched = framed1_slot_schedule(240, K, 4, 31337002, 0, 0, 0);
		assert_eq!(sched.len(), 240);
		let count = |sl: &[i64], v: i64| sl.iter().filter(|&&x| x == v).count();
		// slot 3 = newest = Luiz's window0
		assert_eq!(
			[
				count(&sched, 3),
				count(&sched, 2),
				count(&sched, 1),
				count(&sched, 0)
			],
			[128, 64, 32, 16],
			"global quota"
		);
		for m in 0..4
		{
			let blk = &sched[m * 60..(m + 1) * 60];
			assert_eq!(
				[count(blk, 3), count(blk, 2), count(blk, 1), count(blk, 0)],
				[32, 16, 8, 4],
				"motor {m} quota"
			);
		}
	}

	#[test]
	fn slot_schedule_is_deterministic_and_shuffled()
	{
		let a = framed1_slot_schedule(240, K, 4, 42, 0, 0, 0);
		let b = framed1_slot_schedule(240, K, 4, 42, 0, 0, 0);
		assert_eq!(a, b, "schedule must be a pure function of its coordinates");
		// Shuffled: the first motor block must not be sorted (16 zeros first would
		// mean an index-keyed layout — exactly the stale-motor trap).
		let first: Vec<i64> = a[..60].to_vec();
		let mut sorted = first.clone();
		sorted.sort_unstable();
		assert_ne!(first, sorted, "block not shuffled");
	}

	#[test]
	fn slot_schedule_odd_sizes_are_safe()
	{
		for n in [7usize, 61, 240]
		{
			let sc = framed1_slot_schedule(n, K, 4, 1, 0, 0, 0);
			assert_eq!(sc.len(), n);
			assert!(sc.iter().all(|&s| (0..K as i64).contains(&s)));
		}
	}

	// ---- scoped axonogenesis (16/08/2026, GA-connectivity types) -------------

	#[test]
	fn scope_free_is_bit_identical_to_legacy()
	{
		let before: Vec<i64> = (0..24).map(|i| i * 3).collect();
		let mut legacy = before.clone();
		let mut scoped = before.clone();
		resample_suffix(&mut legacy, SPACE, 0.7, 77, 1, 2, 3);
		resample_suffix_scoped(&mut scoped, SPACE, 0.7, 0, FRAME, BPF, 77, 1, 2, 3);
		assert_eq!(
			legacy, scoped,
			"scope 0 must reproduce the legacy draw exactly"
		);
		assert_ne!(legacy, before, "rate 0.7 should have moved something");
	}

	#[test]
	fn scope_window_never_crosses_time()
	{
		// bits spread over all 4 windows; after heavy mutation each bit must
		// still live in its ORIGINAL window.
		let before: Vec<i64> = (0..32).map(|i| (i % 4) * FRAME as i64 + i * 4).collect();
		let windows_before: Vec<i64> = before.iter().map(|b| b / FRAME as i64).collect();
		let mut suf = before.clone();
		resample_suffix_scoped(&mut suf, SPACE, 1.0, 1, FRAME, BPF, 13, 0, 0, 0);
		let windows_after: Vec<i64> = suf.iter().map(|b| b / FRAME as i64).collect();
		assert_eq!(windows_before, windows_after, "window scope crossed time");
		assert_ne!(suf, before, "rate 1.0 should have rewired");
		let set: std::collections::HashSet<i64> = suf.iter().copied().collect();
		assert_eq!(set.len(), suf.len(), "distinctness broken");
	}

	#[test]
	fn scope_feature_freezes_the_feature_map()
	{
		// One bit per feature across two windows; after mutation each bit keeps
		// its (window, feature) — only the threshold may move.
		let before: Vec<i64> = (0..24)
			.map(|i| ((i % 2) * FRAME + (i / 2) * BPF + 3) as i64)
			.collect();
		let runs_before: Vec<i64> = before.iter().map(|b| b / BPF as i64).collect();
		let mut suf = before.clone();
		resample_suffix_scoped(&mut suf, SPACE, 1.0, 2, FRAME, BPF, 21, 0, 0, 0);
		let runs_after: Vec<i64> = suf.iter().map(|b| b / BPF as i64).collect();
		assert_eq!(
			runs_before, runs_after,
			"feature scope changed a feature/window"
		);
		assert_ne!(suf, before, "rate 1.0 should have moved thresholds");
		let set: std::collections::HashSet<i64> = suf.iter().copied().collect();
		assert_eq!(set.len(), suf.len(), "distinctness broken");
	}

	#[test]
	fn scope_window_at_k1_degenerates_to_free()
	{
		let before: Vec<i64> = (0..16).collect();
		let mut a = before.clone();
		let mut b = before.clone();
		// k=1: frame_bits == space
		resample_suffix_scoped(&mut a, FRAME, 0.9, 1, FRAME, BPF, 5, 0, 0, 0);
		resample_suffix_scoped(&mut b, FRAME, 0.9, 0, FRAME, BPF, 5, 0, 0, 0);
		assert_eq!(a, b, "window scope at one window must equal free");
	}
}
