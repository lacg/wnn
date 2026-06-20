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
fn pearson(x: &[f32], y: &[f32]) -> f32 {
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
