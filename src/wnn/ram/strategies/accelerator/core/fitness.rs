//! Generic fitness combines — the results-determining math, in the wheel.
//!
//! Ported from Python 19/08/2026 (Luiz: "why is this in Python at all?"). The
//! fitness combine decides which genome every search KEEPS, yet it lived in the
//! editable Python layer — the same layer where the 18/08 six-site weight no-op
//! and the S16 recipe drift happened. Metrics are produced in Rust either way;
//! this module moves the combine next to them, ABI-gated and cargo-tested, so a
//! live run can no longer lazily import a half-edited selector.
//!
//! DOMAIN-BLIND by design: callers reduce their metrics to columns of f64 plus
//! an orientation flag. The controller (err²/stable/jerk/mono/steady/alt/pos)
//! and IDS (CE/acc/...) both reduce to exactly that, which is what makes this
//! ram_core substrate rather than either wheel's private code. The Python
//! calculators keep only the domain mapping (Metrics -> columns) and the
//! warn-once policy for unplumbed metrics; every number that ranks a genome is
//! computed here.
//!
//! Three combines, one rank helper, all lower-score-is-better:
//!   compute_ranks     fractional tie-aware ranks (the "1224.5" scheme)
//!   rank_combine      weighted rank mean — Harmonic (legacy) or Arithmetic
//!   zrank_combine     winsorized robust z — magnitude-aware (19/08 decision)

/// One metric column: the raw per-candidate values plus which direction wins.
///
/// `higher_is_better` exists so callers never pre-negate values — a silent
/// negation at one call site and not another is exactly the class of drift
/// this module exists to end.
pub struct MetricColumn<'a>
{
	pub values: &'a [f64],
	pub weight: f64,
	pub higher_is_better: bool,
}

/// How rank_combine aggregates the weighted per-metric ranks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RankAggregation
{
	/// WHM = Σw / Σ(w/rank). Dominated by a candidate's BEST weighted rank and
	/// nearly indifferent to its worst (rank 1 at w=.35 contributes .350 to the
	/// denominator; rank 9 at w=.15 contributes .017) — selects specialists.
	/// Kept for the legacy arm of A/Bs and for reproducing banked results.
	Harmonic,
	/// Σ(w·rank)/Σw. Every rank hurts in proportion to its weight — what a
	/// weight vector reads as promising. Stage-select default since 19/08/2026.
	Arithmetic,
}

/// 1-based fractional ranks; tied values share the AVERAGE of their positions.
///
/// Port of the ONE Python ranking helper (FitnessCalculator.compute_ranks),
/// including the 09/08/2026 tie fix: `enumerate(sorted(...))` handed tied
/// values distinct ranks by list position, silently favoring incumbent elites
/// on exactly the metric where genomes are indistinguishable (measured: 42% of
/// IDS populations carry an fpr tie; the controller's dominant tie is
/// stable_rate=100%). Ties group on EXACT value equality, matching Python.
///
/// `ascending=true` → lower value gets rank 1 (costs); `false` → higher wins.
pub fn compute_ranks(values: &[f64], ascending: bool) -> Vec<f64>
{
	let n = values.len();
	let mut order: Vec<usize> = (0..n).collect();
	// Total order: NaN would poison sort transitivity — callers must reject it
	// (validate_columns does); total_cmp keeps the sort lawful regardless.
	if ascending
	{
		order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
	}
	else
	{
		order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
	}
	let mut ranks = vec![0.0; n];
	let mut i = 0;
	while i < n
	{
		let mut j = i;
		while j + 1 < n && values[order[j + 1]] == values[order[i]]
		{
			j += 1;
		}
		let avg = (i + j) as f64 / 2.0 + 1.0; // positions i..j hold ranks i+1..j+1
		for k in i..=j
		{
			ranks[order[k]] = avg;
		}
		i = j + 1;
	}
	ranks
}

/// Shared validation: equal lengths, at least one candidate, positive total
/// weight, finite values. Returns candidate count.
fn validate_columns(columns: &[MetricColumn]) -> Result<usize, String>
{
	let Some(first) = columns.first()
	else
	{
		return Err("fitness combine: no metric columns".into());
	};
	let n = first.values.len();
	if n == 0
	{
		return Err("fitness combine: empty candidate list".into());
	}
	let mut w_sum = 0.0;
	for (ci, c) in columns.iter().enumerate()
	{
		if c.values.len() != n
		{
			return Err(format!(
				"fitness combine: column {} has {} values, expected {}",
				ci, c.values.len(), n));
		}
		if !c.weight.is_finite() || c.weight < 0.0
		{
			return Err(format!("fitness combine: column {} weight {} invalid", ci, c.weight));
		}
		w_sum += c.weight;
		if let Some(bad) = c.values.iter().find(|v| !v.is_finite())
		{
			// Refuse rather than skip: the Python adapter owns the "metric not
			// plumbed -> warn once and drop the COLUMN" policy. A non-finite
			// value reaching this far is a scorer bug, and ranking around it
			// would hide the bug inside a plausible ordering.
			return Err(format!("fitness combine: column {} contains non-finite value {}", ci, bad));
		}
	}
	if w_sum <= 0.0
	{
		return Err("fitness combine: total weight is zero".into());
	}
	Ok(n)
}

/// Weighted rank mean over the columns. Lower score = better candidate.
pub fn rank_combine(columns: &[MetricColumn], aggregation: RankAggregation)
	-> Result<Vec<f64>, String>
{
	let n = validate_columns(columns)?;
	if n == 1
	{
		return Ok(vec![1.0]);
	}
	let w_sum: f64 = columns.iter().map(|c| c.weight).sum();
	let per_col: Vec<(Vec<f64>, f64)> = columns.iter()
		.filter(|c| c.weight > 0.0)
		.map(|c| (compute_ranks(c.values, !c.higher_is_better), c.weight))
		.collect();
	let scores = (0..n).map(|i| match aggregation
	{
		RankAggregation::Harmonic =>
			w_sum / per_col.iter().map(|(r, w)| w / r[i]).sum::<f64>(),
		RankAggregation::Arithmetic =>
			per_col.iter().map(|(r, w)| w * r[i]).sum::<f64>() / w_sum,
	}).collect();
	Ok(scores)
}

/// Winsorized robust z combine — the magnitude-aware fitness (19/08/2026).
///
/// Per column: z = (x − median) / (1.4826·MAD), clamped to ±`clamp`, negated
/// for higher-is-better columns. Score = Σ(w·z)/Σw; lower = better.
///
/// Why THIS shape (over gap-interpolated ranks and log-ratios):
///   * magnitude counts — 1st by 13° is no longer worth the same as 1st by
///     0.1°, which is the whole point;
///   * median/MAD ignores the degenerate pool members every controller pool
///     contains (0% stable, −450k rewards), where min-max normalization lets
///     one degenerate compress all healthy candidates together;
///   * scale-free per column — metres never numerically compete with radians,
///     preserving the property the rank scheme was adopted for;
///   * the ±clamp is the λ_alt lesson encoded: no single dimension may capture
///     the score, however extreme the outlier.
///
/// MAD = 0 (a majority of candidates share the median) degenerates gracefully:
/// values at the median score z = 0, values off it score the full ±clamp —
/// exactly the limit of (x−med)/ε under the clamp, without the ε.
pub fn zrank_combine(columns: &[MetricColumn], clamp: f64) -> Result<Vec<f64>, String>
{
	if !(clamp > 0.0) || !clamp.is_finite()
	{
		return Err(format!("zrank_combine: clamp {} must be a positive finite number", clamp));
	}
	let n = validate_columns(columns)?;
	let w_sum: f64 = columns.iter().map(|c| c.weight).sum();
	let mut scores = vec![0.0; n];
	for c in columns.iter().filter(|c| c.weight > 0.0)
	{
		let med = median(c.values);
		let abs_dev: Vec<f64> = c.values.iter().map(|v| (v - med).abs()).collect();
		let scale = 1.4826 * median(&abs_dev);
		for (i, v) in c.values.iter().enumerate()
		{
			let z = if scale > 0.0
			{
				((v - med) / scale).clamp(-clamp, clamp)
			}
			else if *v > med
			{
				clamp
			}
			else if *v < med
			{
				-clamp
			}
			else
			{
				0.0
			};
			let oriented = if c.higher_is_better { -z } else { z };
			scores[i] += c.weight * oriented;
		}
	}
	for s in scores.iter_mut()
	{
		*s /= w_sum;
	}
	Ok(scores)
}

fn median(values: &[f64]) -> f64
{
	let mut sorted = values.to_vec();
	sorted.sort_by(|a, b| a.total_cmp(b));
	let n = sorted.len();
	if n % 2 == 1
	{
		sorted[n / 2]
	}
	else
	{
		(sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	fn col<'a>(values: &'a [f64], weight: f64, higher: bool) -> MetricColumn<'a>
	{
		MetricColumn { values, weight, higher_is_better: higher }
	}

	// --- compute_ranks: parity with the Python helper, ties included ---------

	#[test]
	fn ranks_basic_ascending()
	{
		assert_eq!(compute_ranks(&[3.0, 1.0, 2.0], true), vec![3.0, 1.0, 2.0]);
	}

	#[test]
	fn ranks_descending_inverts()
	{
		assert_eq!(compute_ranks(&[3.0, 1.0, 2.0], false), vec![1.0, 3.0, 2.0]);
	}

	#[test]
	fn ranks_ties_share_average_positions()
	{
		// The 09/08 fix: 20 genomes sharing the best value must ALL get the
		// average rank, not 1..20 by arrival order. Σranks is preserved.
		let ranks = compute_ranks(&[5.0, 5.0, 5.0, 7.0], true);
		assert_eq!(ranks, vec![2.0, 2.0, 2.0, 4.0]);
		let all_tied = compute_ranks(&[1.0, 1.0, 1.0], true);
		assert_eq!(all_tied, vec![2.0, 2.0, 2.0]);
	}

	// --- rank_combine: pins the arm-9 headline flip in Rust too --------------

	// Arm 9 (AW_REF_lam16) stage-select pool, verbatim from the marker:
	// stable% / reward / jerk / mono / steady for 9 candidates.
	const STABLE: [f64; 9] = [0.0, 0.0, 5.2, 22.4, 0.4, 6.0, 8.4, 16.4, 0.0];
	const REWARD: [f64; 9] = [-446218.31, -454539.90, -266045.09, -42195.02,
	                          -162713.12, -205989.16, -226324.99, -39921.60, -158382.17];
	const JERK: [f64; 9] = [0.0343, 0.0381, 0.0445, 0.0479, 0.0413, 0.0472, 0.0506, 0.0457, 0.0058];
	const MONO: [f64; 9] = [0.7, 0.6, 0.5, 0.3, 0.6, 0.4, 0.4, 0.3, 0.0];
	const STEADY: [f64; 9] = [88.62, 89.71, 38.93, 39.48, 47.63, 35.12, 25.83, 40.70, 87.88];
	const MEM0: usize = 6;   // the steady-specialist the WHM crowned
	const CONN0: usize = 3;  // the all-rounder Luiz identified by hand

	fn arm9() -> [(&'static [f64], f64, bool); 5]
	{
		// S16 weights; reward is higher-is-better, the rest are costs.
		[(&REWARD, 0.25, true), (&STABLE, 0.20, true), (&JERK, 0.15, false),
		 (&MONO, 0.05, false), (&STEADY, 0.35, false)]
	}

	fn arm9_columns() -> Vec<MetricColumn<'static>>
	{
		arm9().into_iter()
			.map(|(v, w, h)| MetricColumn { values: v, weight: w, higher_is_better: h })
			.collect()
	}

	fn argmin(scores: &[f64]) -> usize
	{
		scores.iter().enumerate()
			.min_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()
	}

	#[test]
	fn harmonic_crowns_the_specialist()
	{
		let scores = rank_combine(&arm9_columns(), RankAggregation::Harmonic).unwrap();
		assert_eq!(argmin(&scores), MEM0);
	}

	#[test]
	fn arithmetic_flips_to_the_all_rounder()
	{
		let scores = rank_combine(&arm9_columns(), RankAggregation::Arithmetic).unwrap();
		// CONN#0 and MEM#1 (index 7) tie at 3.425 under pure ranks; either way
		// the specialist MEM#0 (4.025) must NOT win.
		let best = argmin(&scores);
		assert!(best == CONN0 || best == 7, "got index {best}");
		assert!(scores[MEM0] > scores[CONN0]);
		assert!((scores[CONN0] - 3.425).abs() < 1e-9);
		assert!((scores[MEM0] - 4.025).abs() < 1e-9);
	}

	#[test]
	fn zrank_separates_what_ranks_tie()
	{
		// Magnitude-awareness is the point: CONN#0 wins OUTRIGHT (no tie with
		// MEM#1), and the specialist trails both. Values pinned from the
		// 19/08 Python session that chose this formulation.
		let scores = zrank_combine(&arm9_columns(), 3.0).unwrap();
		assert_eq!(argmin(&scores), CONN0);
		assert!(scores[CONN0] < scores[7]);
		assert!(scores[MEM0] > scores[7]);
		assert!((scores[CONN0] - -0.880).abs() < 5e-3, "got {}", scores[CONN0]);
		assert!((scores[MEM0] - -0.351).abs() < 5e-3, "got {}", scores[MEM0]);
	}

	// --- zrank properties ----------------------------------------------------

	#[test]
	fn zrank_clamp_bounds_single_metric_capture()
	{
		// The λ_alt lesson: one −450k outlier may not own the ranking. With the
		// clamp, the outlier's z on that column is exactly −clamp, not −40.
		let outlier = [1.0, 2.0, 3.0, -450000.0];
		let scores = zrank_combine(&[col(&outlier, 1.0, false)], 3.0).unwrap();
		assert_eq!(scores[3], -3.0);
		// And every score stays within ±clamp by construction.
		assert!(scores.iter().all(|s| s.abs() <= 3.0));
	}

	#[test]
	fn zrank_orientation_negates()
	{
		let v = [1.0, 2.0, 3.0];
		let cost = zrank_combine(&[col(&v, 1.0, false)], 3.0).unwrap();
		let gain = zrank_combine(&[col(&v, 1.0, true)], 3.0).unwrap();
		for (c, g) in cost.iter().zip(gain.iter())
		{
			assert!((c + g).abs() < 1e-12);
		}
	}

	#[test]
	fn zrank_mad_zero_degenerates_to_sign_times_clamp()
	{
		// Majority at the median → MAD = 0. At-median scores 0, off-median ±clamp.
		let v = [5.0, 5.0, 5.0, 9.0, 1.0];
		let scores = zrank_combine(&[col(&v, 1.0, false)], 3.0).unwrap();
		assert_eq!(&scores[..3], &[0.0, 0.0, 0.0]);
		assert_eq!(scores[3], 3.0);
		assert_eq!(scores[4], -3.0);
	}

	#[test]
	fn zrank_weights_scale_contributions()
	{
		// Two opposing columns, 3:1 weights → the heavy column wins 3:1.
		let a = [1.0, 3.0];
		let b = [3.0, 1.0];
		let scores = zrank_combine(
			&[col(&a, 0.75, false), col(&b, 0.25, false)], 3.0).unwrap();
		assert!(scores[0] < scores[1]);
		assert!((scores[0] + scores[1]).abs() < 1e-12); // symmetric pool
	}

	// --- shared validation ----------------------------------------------------

	#[test]
	fn single_candidate_is_rank_one()
	{
		let v = [42.0];
		let scores = rank_combine(&[col(&v, 1.0, false)], RankAggregation::Harmonic).unwrap();
		assert_eq!(scores, vec![1.0]);
	}

	#[test]
	fn non_finite_values_are_refused_not_ranked()
	{
		let v = [1.0, f64::NAN];
		assert!(rank_combine(&[col(&v, 1.0, false)], RankAggregation::Arithmetic).is_err());
		assert!(zrank_combine(&[col(&v, 1.0, false)], 3.0).is_err());
	}

	#[test]
	fn zero_total_weight_refused()
	{
		let v = [1.0, 2.0];
		assert!(zrank_combine(&[col(&v, 0.0, false)], 3.0).is_err());
	}

	#[test]
	fn mismatched_column_lengths_refused()
	{
		let a = [1.0, 2.0];
		let b = [1.0];
		assert!(rank_combine(
			&[col(&a, 0.5, false), col(&b, 0.5, false)],
			RankAggregation::Arithmetic).is_err());
	}

	#[test]
	fn invalid_clamp_refused()
	{
		let v = [1.0, 2.0];
		assert!(zrank_combine(&[col(&v, 1.0, false)], 0.0).is_err());
		assert!(zrank_combine(&[col(&v, 1.0, false)], f64::NAN).is_err());
	}
}
