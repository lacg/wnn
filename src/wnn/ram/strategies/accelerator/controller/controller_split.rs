//! Conflict-driven state-splitting trainer — pure helper logic.
//!
//! Design doc: .claude/plans/controller_state_splitting_design.md
//! This module holds the data-only parts of the splitting trainer (conflict
//! scan + discriminative backward walk) as free functions, so they can be unit
//! reasoned about independently of the WnnController forward-roll (which lives
//! in controller.rs and owns the memory/connectivity).
//!
//! Phase 2 scope: the SCAN (bucket by output-layer input, flag PWM disagreement)
//! and the discriminative WALK (find the (bit, lag) that best separates a
//! conflict's high- vs low-PWM instances). The TYPE-1 (event) arm only — TYPE-2
//! (accumulative/integral) detection is reported but its counter install is
//! Phase 3.

/// A conflict: instances (indices into the flat record arrays) that share the
/// SAME output-layer input but disagree on target PWM beyond tau. The output
/// layer cannot satisfy them all → only a state distinction can.
pub struct Conflict {
	// KEPT-API: conflict diagnostics for dump inspection
	#[allow(dead_code)]
	pub out_in: Vec<bool>,
	pub instances: Vec<usize>,
	pub spread: f32,
}

/// Per-motor PWM spread of a set of instances = max over motors of (max - min).
pub(crate) fn pwm_spread(idxs: &[usize], pwms: &[[f32; 4]]) -> f32 {
	let mut spread = 0.0f32;
	for m in 0..4 {
		let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
		for &i in idxs {
			let v = pwms[i][m];
			lo = lo.min(v);
			hi = hi.max(v);
		}
		spread = spread.max(hi - lo);
	}
	spread
}

/// Bucket records by output-layer input; flag buckets whose PWM spread exceeds
/// tau. Returns conflicts sorted by descending spread (worst first). Singletons
/// and agreeing buckets are dropped.
pub fn scan_conflicts(out_ins: &[Vec<bool>], pwms: &[[f32; 4]], tau: f32) -> Vec<Conflict> {
	use std::collections::HashMap;
	let mut buckets: HashMap<&Vec<bool>, Vec<usize>> = HashMap::new();
	for (i, oi) in out_ins.iter().enumerate() {
		buckets.entry(oi).or_default().push(i);
	}
	let mut conflicts: Vec<Conflict> = buckets
		.into_iter()
		.filter(|(_, idxs)| idxs.len() >= 2)
		.filter_map(|(oi, idxs)| {
			let spread = pwm_spread(&idxs, pwms);
			(spread > tau).then(|| Conflict { out_in: oi.clone(), instances: idxs, spread })
		})
		.collect();
	conflicts.sort_by(|a, b| b.spread.partial_cmp(&a.spread).unwrap_or(std::cmp::Ordering::Equal));
	conflicts
}

/// Coarsen one record's output-layer input to a bucket key: `k` evenly-spaced
/// thermometer bits per feature (a coarse attitude region) + the FULL state.
/// Control still uses the fine encoding; only the conflict scanner coarsens.
pub(crate) fn coarse_key(oi: &[bool], k: usize, bpf: usize, num_features: usize, frame_bits: usize) -> Vec<bool> {
	let mut key = Vec::with_capacity(num_features * k + (oi.len() - frame_bits));
	for f in 0..num_features {
		let base = f * bpf;
		for j in 0..k {
			// evenly-spaced bit within the feature's bpf-bit block (k>=bpf → all)
			let idx = if k >= bpf { j } else { ((j + 1) * bpf) / (k + 1) };
			key.push(oi[base + idx.min(bpf - 1)]);
		}
	}
	key.extend_from_slice(&oi[frame_bits..]); // full state
	key
}

/// Length of a coarse key at coarseness `k` (mirrors coarse_key's output len).
#[inline]
pub(crate) fn coarse_key_len(oi_len: usize, k: usize, num_features: usize, frame_bits: usize) -> usize {
	num_features * k + (oi_len - frame_bits)
}

/// Pack one record's coarse key into `out` (a zeroed words_per_key slice).
/// Bit i of the key → word i/64, bit i%64. Same bit SEQUENCE as `coarse_key`,
/// so two records bucket together here iff their `coarse_key`s are equal.
#[inline]
fn coarse_key_packed(
	oi: &[bool], k: usize, bpf: usize, num_features: usize, frame_bits: usize, out: &mut [u64],
) {
	let mut bit = 0usize;
	let mut set = |b: bool, bit: usize, out: &mut [u64]| {
		if b {
			out[bit >> 6] |= 1u64 << (bit & 63);
		}
	};
	for f in 0..num_features {
		let base = f * bpf;
		for j in 0..k {
			let idx = if k >= bpf { j } else { ((j + 1) * bpf) / (k + 1) };
			set(oi[base + idx.min(bpf - 1)], bit, out);
			bit += 1;
		}
	}
	for &b in &oi[frame_bits..] {
		set(b, bit, out);
		bit += 1;
	}
}

/// Unpack a packed key back to the `Vec<bool>` the Conflict diagnostic carries.
#[inline]
fn unpack_key(words: &[u64], key_len: usize) -> Vec<bool> {
	(0..key_len).map(|b| (words[b >> 6] >> (b & 63)) & 1 == 1).collect()
}

/// Conflict scan over PACKED keys held in ONE flat buffer (20/07/2026).
///
/// Memory: the old path materialised a `Vec<Vec<bool>>` — one heap allocation of
/// ~key_len bytes + 24B header + allocator slack PER RECORD, rebuilt for every
/// `k` the adaptive loop tried. At production scale (24 eps × 2000 steps = 48k
/// records) that is 48k allocations and ~6 MB per genome per k, live across the
/// rayon fan-out. Packing to u64 words in a single buffer is ~8× smaller and
/// ONE allocation; bucketing borrows `&[u64]` slices out of it. Bucketing is
/// bit-for-bit the old semantics (identical key bits ⇒ identical words), so
/// conflicts and their order are unchanged.
fn scan_conflicts_packed(
	keys_flat: &[u64], words_per_key: usize, n: usize, key_len: usize,
	pwms: &[[f32; 4]], tau: f32,
) -> Vec<Conflict> {
	use std::collections::HashMap;
	let mut buckets: HashMap<&[u64], Vec<usize>> = HashMap::new();
	for i in 0..n {
		buckets.entry(&keys_flat[i * words_per_key..(i + 1) * words_per_key])
			.or_default().push(i);
	}
	let mut conflicts: Vec<Conflict> = buckets
		.into_iter()
		.filter(|(_, idxs)| idxs.len() >= 2)
		.filter_map(|(kw, idxs)| {
			let spread = pwm_spread(&idxs, pwms);
			// out_in (diagnostic) is materialised ONLY for real conflicts — a small
			// fraction of records — instead of for every record up front.
			(spread > tau).then(|| Conflict { out_in: unpack_key(kw, key_len), instances: idxs, spread })
		})
		.collect();
	conflicts.sort_by(|a, b| b.spread.partial_cmp(&a.spread).unwrap_or(std::cmp::Ordering::Equal));
	conflicts
}

/// Adaptive-coarseness conflict scan. Exact full-frame bucketing never collides
/// on real continuous-attitude trajectories (every thermometer code is unique →
/// zero conflicts). This buckets by a COARSE frame signature instead, and picks
/// the LARGEST `k` (most specific) whose conflict count still reaches
/// `target_min` — coarsening only as much as needed to surface conflicts. Returns
/// (conflicts at the chosen k, chosen_k).
pub fn scan_conflicts_coarse(
	out_ins: &[Vec<bool>],
	pwms: &[[f32; 4]],
	tau: f32,
	bpf: usize,
	num_features: usize,
	frame_bits: usize,
	target_min: usize,
) -> (Vec<Conflict>, usize) {
	if bpf == 0 || out_ins.is_empty() {
		return (Vec::new(), bpf);
	}
	// ONE reusable flat buffer, sized for the widest key (k = bpf) and re-zeroed
	// per k — so the whole adaptive loop costs a single allocation, not one Vec
	// per record per k.
	let n = out_ins.len();
	let max_len = coarse_key_len(out_ins[0].len(), bpf, num_features, frame_bits);
	let max_words = max_len.div_ceil(64);
	let mut keys_flat = vec![0u64; n * max_words];
	for k in (1..=bpf).rev() {
		let key_len = coarse_key_len(out_ins[0].len(), k, num_features, frame_bits);
		let words = key_len.div_ceil(64);
		keys_flat[..n * words].fill(0);
		for (i, oi) in out_ins.iter().enumerate() {
			coarse_key_packed(oi, k, bpf, num_features, frame_bits,
				&mut keys_flat[i * words..(i + 1) * words]);
		}
		let conflicts = scan_conflicts_packed(&keys_flat[..n * words], words, n, key_len, pwms, tau);
		if conflicts.len() >= target_min || k == 1 {
			return (conflicts, k);
		}
	}
	(Vec::new(), 1)
}

/// Result of the discriminative backward walk over one conflict.
pub struct Separator {
	pub bit: usize,   // index into the STATE-LAYER input vector
	pub lag: usize,   // steps back from the conflict step (0 = the conflict step itself)
	pub gain: f32,    // separation quality in [0,1]; 1.0 = perfect partition
	pub high_on: bool, // true if bit=1 correlates with the HIGH-PWM group
}

/// Split a conflict's instances into HIGH vs LOW groups by motor-0 PWM about the
/// bucket mean (Phase 2 event tasks are binary; richer clustering is later).
/// Returns (is_high per instance). Falls back to the motor with the largest
/// spread so the split tracks the disagreeing actuator.
pub fn label_high_low(idxs: &[usize], pwms: &[[f32; 4]]) -> Vec<bool> {
	// pick the motor with the largest spread
	let mut best_m = 0usize;
	let mut best_s = -1.0f32;
	for m in 0..4 {
		let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
		for &i in idxs {
			lo = lo.min(pwms[i][m]);
			hi = hi.max(pwms[i][m]);
		}
		if hi - lo > best_s {
			best_s = hi - lo;
			best_m = m;
		}
	}
	let mean: f32 = idxs.iter().map(|&i| pwms[i][best_m]).sum::<f32>() / idxs.len() as f32;
	idxs.iter().map(|&i| pwms[i][best_m] >= mean).collect()
}

/// Binary information-gain-ish separation score of a 0/1 feature against a 0/1
/// label, in [0,1]. 1.0 = the feature perfectly partitions the labels. Uses
/// classification purity (1 - weighted minority fraction) which is 0.5 at chance
/// and 1.0 at perfect — rescaled to [0,1] as 2*(purity-0.5).
pub fn separation_score(feature: &[bool], label: &[bool]) -> f32 {
	let n = feature.len();
	if n == 0 {
		return 0.0;
	}
	// counts: [feature][label]
	let mut c = [[0usize; 2]; 2];
	for k in 0..n {
		c[feature[k] as usize][label[k] as usize] += 1;
	}
	let mut correct = 0usize;
	for f in 0..2 {
		correct += c[f][0].max(c[f][1]); // majority label within this feature value
	}
	let purity = correct as f32 / n as f32; // 0.5..1.0
	(2.0 * (purity - 0.5)).clamp(0.0, 1.0)
}

/// Discriminative backward walk over one conflict (the utile criterion, backward
/// direction). For each candidate state-observable bit and each lag, score how
/// well "bit at (step-lag)" partitions the HIGH- vs LOW-PWM instances. Returns
/// the best separator, or None if nothing observed separates them (which is the
/// TYPE-2 / accumulative signal — handled in Phase 3, not here).
///
/// `instances` are record indices; `ep_start[ep]` is the record index where
/// episode `ep` begins (records are episode-major), so the record at lag is
/// `ep_start[ep] + (step - lag)`. `candidate_bits` are state-layer input bit
/// positions some state neuron observes (connectivity gating). `max_lag` bounds
/// how far back to look; lags where any instance has `step < lag` are skipped.
#[allow(clippy::too_many_arguments)]
pub fn discriminative_walk(
	instances: &[usize],
	labels: &[bool],
	ep_of: &[usize],
	step_of: &[usize],
	ep_start: &[usize],
	state_ins_flat: &[bool],
	state_in_len: usize,
	candidate_bits: &[usize],
	max_lag: usize,
) -> Option<Separator> {
	let mut best: Option<Separator> = None;
	// lag ≥ 1: state is about HISTORY. A lag-0 (current-frame) separator is the
	// output's job (it observes the frame) — planting state for it would be a
	// degenerate "latch" of the current bit. So state distinctions only come from
	// lag ≥ 1. (Coarse bucketing over-generates conflicts; this is the sieve that
	// keeps only the genuinely-historical ones — design §11/5d.)
	for lag in 1..=max_lag {
		// require every instance to have history at this lag
		if instances.iter().any(|&i| step_of[i] < lag) {
			continue;
		}
		for &b in candidate_bits {
			let feature: Vec<bool> = instances
				.iter()
				.map(|&i| {
					let rec = ep_start[ep_of[i]] + (step_of[i] - lag);
					state_ins_flat[rec * state_in_len + b]
				})
				.collect();
			let gain = separation_score(&feature, labels);
			// direction: does bit=1 predict the HIGH (label=1) group?
			let agree = feature.iter().zip(labels).filter(|(f, l)| f == l).count();
			let high_on = agree * 2 >= feature.len();
			let better = match &best {
				None => true,
				Some(s) => gain > s.gain || (gain == s.gain && lag < s.lag),
			};
			if better && gain > 0.0 {
				best = Some(Separator { bit: b, lag, gain, high_on });
			}
		}
	}
	best
}

/// An accumulative (TYPE-2) distinction: a feature whose WINDOW COUNT (signed
/// sum over the lookback) correlates with the disagreeing PWM, when no single
/// (bit, lag) cleanly separates the conflict. This is the integral signal.
pub struct Accumulator {
	pub bit: usize,
	pub up: bool,  // true: more count -> higher PWM
	pub corr: f32, // |Pearson| of window-count vs PWM, in [0,1]
}

/// Pearson correlation of two equal-length series; 0.0 if either has no variance.
pub(crate) fn pearson(x: &[f32], y: &[f32]) -> f32 {
	let n = x.len();
	if n == 0 {
		return 0.0;
	}
	let nf = n as f32;
	let mx = x.iter().sum::<f32>() / nf;
	let my = y.iter().sum::<f32>() / nf;
	let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
	for k in 0..n {
		let dx = x[k] - mx;
		let dy = y[k] - my;
		sxy += dx * dy;
		sxx += dx * dx;
		syy += dy * dy;
	}
	if sxx <= 1e-9 || syy <= 1e-9 {
		return 0.0;
	}
	sxy / (sxx.sqrt() * syy.sqrt())
}

/// Detect a TYPE-2 (accumulative) distinction: for each candidate feature, count
/// its occurrences over the lookback window per instance and correlate that count
/// with the per-instance PWM. Returns the best-correlated feature. The caller
/// invokes this only when the TYPE-1 walk found no clean stump — a strong result
/// here means "the conflict is explained by how much has accumulated, not by any
/// single past event" = the integral signal.
#[allow(clippy::too_many_arguments)]
pub fn detect_accumulator(
	instances: &[usize],
	pwm_scalar: &[f32],
	ep_of: &[usize],
	step_of: &[usize],
	ep_start: &[usize],
	state_ins_flat: &[bool],
	state_in_len: usize,
	candidate_bits: &[usize],
	max_lag: usize,
) -> Option<Accumulator> {
	let mut best: Option<Accumulator> = None;
	for &b in candidate_bits {
		let counts: Vec<f32> = instances
			.iter()
			.map(|&i| {
				let mut cnt = 0.0f32;
				for lag in 0..=max_lag {
					if step_of[i] >= lag {
						let rec = ep_start[ep_of[i]] + (step_of[i] - lag);
						if state_ins_flat[rec * state_in_len + b] {
							cnt += 1.0;
						}
					}
				}
				cnt
			})
			.collect();
		let corr = pearson(&counts, pwm_scalar);
		if corr.abs() > best.as_ref().map(|a| a.corr).unwrap_or(0.0) {
			best = Some(Accumulator { bit: b, up: corr >= 0.0, corr: corr.abs() });
		}
	}
	best
}

/// A BIDIRECTIONAL (signed) accumulator: a pair of features whose NET window
/// count (count(up) − count(dn)) correlates with the disagreeing PWM. This is
/// the signal that the integral must UNWIND, not just saturate — e.g. two
/// instances with the SAME up-count but different down-counts need different
/// outputs, which an increment-only counter cannot represent.
pub struct BidirAccumulator {
	pub up: usize,
	pub dn: usize,
	pub corr: f32,
}

/// Detect a bidirectional accumulator: over all ordered pairs of candidate
/// features, correlate the per-instance NET count (up − dn) with the PWM; keep
/// the best positive correlation (so `up` is the increment direction). Returns
/// None if no pair beats zero. The caller thresholds `corr`.
#[allow(clippy::too_many_arguments)]
pub fn detect_accumulator_bidir(
	instances: &[usize],
	pwm_scalar: &[f32],
	ep_of: &[usize],
	step_of: &[usize],
	ep_start: &[usize],
	state_ins_flat: &[bool],
	state_in_len: usize,
	candidate_bits: &[usize],
	max_lag: usize,
) -> Option<BidirAccumulator> {
	let m = instances.len();
	// per-candidate-bit window counts
	let counts: Vec<Vec<f32>> = candidate_bits
		.iter()
		.map(|&b| {
			instances
				.iter()
				.map(|&i| {
					let mut cnt = 0.0f32;
					for lag in 0..=max_lag {
						if step_of[i] >= lag {
							let rec = ep_start[ep_of[i]] + (step_of[i] - lag);
							if state_ins_flat[rec * state_in_len + b] {
								cnt += 1.0;
							}
						}
					}
					cnt
				})
				.collect()
		})
		.collect();
	let mut best: Option<BidirAccumulator> = None;
	for ai in 0..candidate_bits.len() {
		for bi in 0..candidate_bits.len() {
			if ai == bi {
				continue;
			}
			let net: Vec<f32> = (0..m).map(|k| counts[ai][k] - counts[bi][k]).collect();
			let corr = pearson(&net, pwm_scalar);
			if corr > best.as_ref().map(|x| x.corr).unwrap_or(0.0) {
				best = Some(BidirAccumulator { up: candidate_bits[ai], dn: candidate_bits[bi], corr });
			}
		}
	}
	best
}

#[cfg(test)]
mod packed_key_tests {
	//! Packed-key conflict scan (20/07/2026) must be a DROP-IN for the old
	//! Vec<Vec<bool>> path: same buckets, same conflicts, same order, same
	//! diagnostic out_in. These pin that equivalence against the reference
	//! implementation (kept here verbatim as the oracle).
	use super::*;
	use rand::rngs::SmallRng;
	use rand::{Rng, SeedableRng};

	/// The PRE-optimization implementation, kept as the oracle.
	fn scan_conflicts_coarse_reference(
		out_ins: &[Vec<bool>], pwms: &[[f32; 4]], tau: f32,
		bpf: usize, num_features: usize, frame_bits: usize, target_min: usize,
	) -> (Vec<Conflict>, usize) {
		if bpf == 0 || out_ins.is_empty() {
			return (Vec::new(), bpf);
		}
		for k in (1..=bpf).rev() {
			let keys: Vec<Vec<bool>> = out_ins
				.iter()
				.map(|oi| coarse_key(oi, k, bpf, num_features, frame_bits))
				.collect();
			let conflicts = scan_conflicts(&keys, pwms, tau);
			if conflicts.len() >= target_min || k == 1 {
				return (conflicts, k);
			}
		}
		(Vec::new(), 1)
	}

	/// Synthetic records with DELIBERATE coarse collisions: a small set of
	/// distinct attitude "regions" revisited many times with differing PWMs, so
	/// the scan finds real conflicts at some k (a purely random set collides
	/// only at the coarsest k and would under-test the equivalence).
	fn fixture(seed: u64, n: usize, num_features: usize, bpf: usize, n_state: usize)
		-> (Vec<Vec<bool>>, Vec<[f32; 4]>, usize) {
		let frame_bits = num_features * bpf;
		let mut rng = SmallRng::seed_from_u64(seed);
		let regions: Vec<Vec<bool>> = (0..6)
			.map(|_| (0..frame_bits + n_state).map(|_| rng.gen::<bool>()).collect())
			.collect();
		let mut out_ins = Vec::with_capacity(n);
		let mut pwms = Vec::with_capacity(n);
		for _ in 0..n {
			let mut r = regions[rng.gen_range(0..regions.len())].clone();
			// jitter a couple of fine bits so exact keys differ but coarse keys collide
			for _ in 0..2 {
				let b = rng.gen_range(0..frame_bits);
				r[b] = !r[b];
			}
			out_ins.push(r);
			let base: f32 = rng.gen_range(0.2..0.8);
			pwms.push([base, base + rng.gen_range(-0.3..0.3), base, base]);
		}
		(out_ins, pwms, frame_bits)
	}

	fn assert_same(a: &(Vec<Conflict>, usize), b: &(Vec<Conflict>, usize), what: &str) {
		assert_eq!(a.1, b.1, "{what}: chosen k differs");
		assert_eq!(a.0.len(), b.0.len(), "{what}: conflict count differs");
		for (i, (x, y)) in a.0.iter().zip(b.0.iter()).enumerate() {
			assert_eq!(x.instances, y.instances, "{what}: conflict {i} instances differ");
			assert_eq!(x.out_in, y.out_in, "{what}: conflict {i} out_in differs");
			assert!((x.spread - y.spread).abs() < 1e-6, "{what}: conflict {i} spread differs");
		}
	}

	#[test]
	fn packed_matches_reference_across_shapes() {
		// Include a key wider than one u64 word (9 feats × 8 bpf + 24 state = 96 bits)
		// so the multi-word packing path is exercised, not just the ≤64-bit case.
		for &(nf, bpf, ns, n) in &[
			(9usize, 8usize, 8usize, 400usize),
			(9, 8, 24, 500),   // 96-bit key → 2 words
			(12, 4, 16, 300),
			(9, 1, 4, 200),    // k can only be 1
		] {
			let (out_ins, pwms, frame_bits) = fixture(0xC0FFEE + nf as u64, n, nf, bpf, ns);
			for &tau in &[0.05f32, 0.2] {
				for &tmin in &[1usize, 32] {
					let got = scan_conflicts_coarse(&out_ins, &pwms, tau, bpf, nf, frame_bits, tmin);
					let want = scan_conflicts_coarse_reference(&out_ins, &pwms, tau, bpf, nf, frame_bits, tmin);
					assert_same(&got, &want, &format!("nf={nf} bpf={bpf} ns={ns} tau={tau} tmin={tmin}"));
				}
			}
		}
	}

	/// Non-vacuity: the fixture must actually produce conflicts, or the
	/// equivalence above would pass trivially on two empty vectors.
	#[test]
	fn fixture_produces_real_conflicts() {
		let (out_ins, pwms, frame_bits) = fixture(0xC0FFEE + 9, 400, 9, 8, 8);
		let (c, _k) = scan_conflicts_coarse(&out_ins, &pwms, 0.05, 8, 9, frame_bits, 1);
		assert!(!c.is_empty(), "fixture produced no conflicts — equivalence test is vacuous");
		assert!(c[0].instances.len() >= 2, "conflict must have ≥2 instances");
	}

	/// Empty input keeps the old early-out contract.
	#[test]
	fn empty_input_is_unchanged() {
		let (c, k) = scan_conflicts_coarse(&[], &[], 0.1, 8, 9, 72, 1);
		assert!(c.is_empty() && k == 8);
	}
}
