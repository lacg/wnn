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

/// Flat-argument dispatcher for the pyo3 wrappers — ONE marshalling contract
/// shared by both wheels, so `ram_accelerator` and `ram_controller` cannot
/// drift apart in how Python reaches these combines.
///
/// `values_flat` is column-major: column c's candidate i sits at c*n + i.
/// `mode` ∈ {"harmonic", "arithmetic", "zscore"}; `clamp` is read only by
/// zscore. Errors are strings for the wrappers to raise as ValueError.
pub fn combine_flat(
	values_flat: &[f64],
	num_candidates: usize,
	weights: &[f64],
	higher_is_better: &[bool],
	mode: &str,
	clamp: f64,
) -> Result<Vec<f64>, String>
{
	let cols = weights.len();
	if higher_is_better.len() != cols
	{
		return Err(format!(
			"combine_flat: {} weights but {} orientation flags", cols, higher_is_better.len()));
	}
	if values_flat.len() != cols * num_candidates
	{
		return Err(format!(
			"combine_flat: {} values != {} columns x {} candidates",
			values_flat.len(), cols, num_candidates));
	}
	let columns: Vec<MetricColumn> = (0..cols).map(|c| MetricColumn {
		values: &values_flat[c * num_candidates..(c + 1) * num_candidates],
		weight: weights[c],
		higher_is_better: higher_is_better[c],
	}).collect();
	match mode
	{
		"harmonic" => rank_combine(&columns, RankAggregation::Harmonic),
		"arithmetic" => rank_combine(&columns, RankAggregation::Arithmetic),
		"zscore" => zrank_combine(&columns, clamp),
		other => Err(format!(
			"combine_flat: unknown mode {:?} (harmonic|arithmetic|zscore)", other)),
	}
}

/// Viability gate + base combine (21/08/2026, spec: docs/CONTROLLER_FITNESS_GATE_SPEC.md).
///
/// Splits the objective into a QUALIFYING stage and a PREFERENTIAL stage. A
/// weighted sum is compensatory by construction — arbitrarily bad performance
/// on one term can be bought back by another, which is how a tumbling genome
/// (0% stable, 86° err, jerk 0.0015, mono 0) outranked flying ones three times
/// across both aggregations. The gate makes "does it fly" non-negotiable:
///
///   feasible                       stable >= gate_stable_min AND err <= gate_err_max
///   feasible vs infeasible         feasible ALWAYS wins (Deb's rule 1)
///   feasible vs feasible           base combine, computed over the FEASIBLE
///                                  SUBSET ONLY — pool-relative normalisation
///                                  (ranks, median/MAD) must not be distorted
///                                  by members the gate has already excluded
///   infeasible vs infeasible       smaller normalised violation wins (Deb's
///                                  rule 3) — generation 0, when nothing flies,
///                                  still ranks by "how close to flying"
///
/// `gate_stable` / `gate_err` are per-candidate GATE INPUTS, not weighted
/// columns: the fitness's own columns rank reward (not err°) and may omit
/// stability entirely, so the gate reads the physical pair directly. Units are
/// the caller's — the Python adapter passes stable_rate as a FRACTION and err
/// in DEGREES, and thresholds in the same units (0.70, 8.0).
///
/// Violation = max(0, (s_min - stable)/s_min) + max(0, (err - e_max)/e_max):
/// each term is a dimensionless "fraction of the bound missed by", so a genome
/// failing one gate badly outranks nothing it shouldn't. Infeasible scores are
/// offset above the worst feasible score by 1.0 + violation, keeping the
/// combined vector totally ordered under "lower = better".
pub fn gated_combine_flat(
	values_flat: &[f64],
	num_candidates: usize,
	weights: &[f64],
	higher_is_better: &[bool],
	mode: &str,
	clamp: f64,
	gate_stable: &[f64],
	gate_err: &[f64],
	gate_stable_min: f64,
	gate_err_max: f64,
) -> Result<Vec<f64>, String>
{
	if !(gate_stable_min > 0.0) || !gate_stable_min.is_finite()
		|| !(gate_err_max > 0.0) || !gate_err_max.is_finite()
	{
		return Err(format!(
			"gated_combine_flat: thresholds must be positive finite (stable_min={}, err_max={})",
			gate_stable_min, gate_err_max));
	}
	if gate_stable.len() != num_candidates || gate_err.len() != num_candidates
	{
		return Err(format!(
			"gated_combine_flat: gate vectors ({} stable, {} err) != {} candidates",
			gate_stable.len(), gate_err.len(), num_candidates));
	}
	if let Some(bad) = gate_stable.iter().chain(gate_err.iter()).find(|v| !v.is_finite())
	{
		return Err(format!("gated_combine_flat: non-finite gate value {}", bad));
	}
	let feasible: Vec<bool> = (0..num_candidates)
		.map(|i| gate_stable[i] >= gate_stable_min && gate_err[i] <= gate_err_max)
		.collect();
	let violation = |i: usize| -> f64 {
		(gate_stable_min - gate_stable[i]).max(0.0) / gate_stable_min
			+ (gate_err[i] - gate_err_max).max(0.0) / gate_err_max
	};
	let n_feasible = feasible.iter().filter(|f| **f).count();
	if n_feasible == num_candidates
	{
		// Everything flies: the gate is inert, the base combine IS the answer.
		return combine_flat(values_flat, num_candidates, weights, higher_is_better, mode, clamp);
	}
	if n_feasible == 0
	{
		// Generation-0 regime: rank purely by distance to feasibility.
		return Ok((0..num_candidates).map(violation).collect());
	}
	// Base combine over the feasible subset only.
	let idx: Vec<usize> = (0..num_candidates).filter(|i| feasible[*i]).collect();
	let cols = weights.len();
	let mut sub_flat = Vec::with_capacity(cols * n_feasible);
	for c in 0..cols
	{
		for &i in &idx
		{
			sub_flat.push(values_flat[c * num_candidates + i]);
		}
	}
	let sub_scores = combine_flat(&sub_flat, n_feasible, weights, higher_is_better, mode, clamp)?;
	let worst = sub_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let mut out = vec![0.0; num_candidates];
	for (k, &i) in idx.iter().enumerate()
	{
		out[i] = sub_scores[k];
	}
	for i in 0..num_candidates
	{
		if !feasible[i]
		{
			out[i] = worst + 1.0 + violation(i);
		}
	}
	Ok(out)
}

/// Desirability combine (26/08/2026, spec: docs/DESIRABILITY_FITNESS_SHAPES.md).
///
/// Luiz's redesign of the two-regime gate: ONE continuous multiplicative
/// utility (Cobb-Douglas / Derringer-Suich) whose LIMIT behavior does the
/// gate's job. Per metric a utility u(x) in (eps, 1], 1 = ideal; fitness is
/// the product PROD u_i^w_i. The score returned is
///
///     -log2(fitness) = SUM w_i * h_i(x_i),   h_i = -log2(u_i)
///
/// "weighted half-lives of desirability lost" — additive, LOWER = better
/// (matches the combine contract), no float underflow. The exchange rate
/// between metrics is (w_i/w_j)*(x_j/x_i): trading a metric away gets more
/// expensive the worse it already is, so compensation vanishes as any metric
/// approaches unacceptable. A tumbling genome (stable ~ 0) carries the capped
/// H_CAP stable half-lives and cannot be bought back by smoothness — the
/// viability gate's job, now emergent instead of a branch. The weights are
/// NEVER inert (the gated design's infeasible branch ignored them; measured
/// 0/686 feasible samples across the 32n ladder, i.e. whole searches ranked
/// without the weights ever applying).
///
/// Two shapes, chosen per column by `shapes[c]`:
///   "power" — higher-is-better FRACTION in [0,1] (stable_rate, f1, recall).
///             u = x^k with k = ln(0.5)/ln(anchor)  =>  u(anchor) = 0.5.
///             h = k * (-log2(x)); x <= 0 caps at H_CAP.
///   "exp"   — lower-is-better cost >= 0 (err deg, steady deg, jerk, mono,
///             alt m, fpr, ce). u = 2^(-x/anchor)  =>  u(anchor) = 0.5.
///             h = x / anchor — the anchor IS the half-life.
///
/// eps = 2^-H_CAP floors every utility (Luiz 26/08: H_CAP = 20), keeping the
/// ordering strict and a gradient alive even among total failures (the old
/// gen-0 "distance to flying" regime falls out of the same formula).
///
/// The retained gate thresholds 0.70 / 8.0 become the half-anchors of their
/// own curves — the ABI-24 calibration survives; its role changes from cliff
/// to concern point.
pub const DESIRABILITY_H_CAP: f64 = 20.0;

pub fn desirability_combine_flat(
	values_flat: &[f64],
	num_candidates: usize,
	weights: &[f64],
	shapes: &[&str],
	half_anchors: &[f64],
) -> Result<Vec<f64>, String>
{
	let cols = weights.len();
	if shapes.len() != cols || half_anchors.len() != cols
	{
		return Err(format!(
			"desirability_combine_flat: {} weights but {} shapes / {} anchors",
			cols, shapes.len(), half_anchors.len()));
	}
	if values_flat.len() != cols * num_candidates
	{
		return Err(format!(
			"desirability_combine_flat: {} values != {} columns x {} candidates",
			values_flat.len(), cols, num_candidates));
	}
	if let Some(bad) = weights.iter().find(|w| !w.is_finite() || **w < 0.0)
	{
		return Err(format!("desirability_combine_flat: bad weight {}", bad));
	}
	// Per-column half-life exponents, validated up front so a bad anchor fails
	// loudly at call time rather than producing a silently-flat column.
	let mut k = vec![0.0_f64; cols];
	for c in 0..cols
	{
		let a = half_anchors[c];
		match shapes[c]
		{
			"power" =>
			{
				if !(a > 0.0 && a < 1.0)
				{
					return Err(format!(
						"desirability_combine_flat: power anchor {} not in (0,1) (col {})", a, c));
				}
				k[c] = 0.5_f64.ln() / a.ln(); // > 0
			}
			"exp" =>
			{
				if !(a > 0.0 && a.is_finite())
				{
					return Err(format!(
						"desirability_combine_flat: exp anchor {} not positive finite (col {})", a, c));
				}
			}
			other => return Err(format!(
				"desirability_combine_flat: unknown shape {:?} (power|exp) (col {})", other, c)),
		}
	}
	let mut out = vec![0.0_f64; num_candidates];
	for c in 0..cols
	{
		let vals = &values_flat[c * num_candidates..(c + 1) * num_candidates];
		if let Some(bad) = vals.iter().find(|v| !v.is_finite())
		{
			return Err(format!(
				"desirability_combine_flat: non-finite value {} (col {})", bad, c));
		}
		for (i, &x) in vals.iter().enumerate()
		{
			let h = match shapes[c]
			{
				// h = k * (-log2 x); x<=0 -> cap. x>1 clamps to 1 (h=0): a
				// fraction above 1 is caller error but must not go negative.
				"power" =>
				{
					if x <= 0.0 { DESIRABILITY_H_CAP }
					else { (k[c] * -(x.min(1.0)).log2()).min(DESIRABILITY_H_CAP) }
				}
				// h = x / anchor; negative cost clamps to 0 (ideal).
				_ => (x.max(0.0) / half_anchors[c]).min(DESIRABILITY_H_CAP),
			};
			out[i] += weights[c] * h;
		}
	}
	Ok(out)
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

	// --- combine_flat: the marshalling contract both wheels share -------------

	#[test]
	fn flat_matches_structured_all_modes()
	{
		let (n, cols) = (9, 5);
		let mut flat = Vec::with_capacity(cols * n);
		let mut weights = Vec::with_capacity(cols);
		let mut higher = Vec::with_capacity(cols);
		for (v, w, h) in arm9()
		{
			flat.extend_from_slice(v);
			weights.push(w);
			higher.push(h);
		}
		let structured = arm9_columns();
		for mode in ["harmonic", "arithmetic", "zscore"]
		{
			let got = combine_flat(&flat, n, &weights, &higher, mode, 3.0).unwrap();
			let want = match mode
			{
				"harmonic" => rank_combine(&structured, RankAggregation::Harmonic).unwrap(),
				"arithmetic" => rank_combine(&structured, RankAggregation::Arithmetic).unwrap(),
				_ => zrank_combine(&structured, 3.0).unwrap(),
			};
			assert_eq!(got, want, "mode {mode} diverged from the structured API");
		}
	}

	#[test]
	fn flat_rejects_shape_and_mode_errors()
	{
		let flat = [1.0, 2.0, 3.0, 4.0];
		assert!(combine_flat(&flat, 2, &[1.0], &[false, true], "zscore", 3.0).is_err());
		assert!(combine_flat(&flat, 3, &[1.0, 1.0], &[false, false], "zscore", 3.0).is_err());
		assert!(combine_flat(&flat, 2, &[0.5, 0.5], &[false, false], "geometric", 3.0).is_err());
	}

	// --- gated_combine_flat: the viability gate (21/08/2026) -----------------

	fn gate_call(vals: &[f64], n: usize, st: &[f64], er: &[f64]) -> Vec<f64>
	{
		// Two columns: reward (higher better, w .7), jerk (lower better, w .3).
		gated_combine_flat(vals, n, &[0.7, 0.3], &[true, false],
			"zscore", 3.0, st, er, 0.70, 8.0).unwrap()
	}

	#[test]
	fn gate_tumbler_ranks_last_despite_perfect_jerk()
	{
		// Candidate 2 is the measured tumbler shape: hopeless reward, PERFECT
		// jerk — under the raw combine its jerk column buys rank back; under
		// the gate it must be worse than every flyer.
		let vals = [-10.0, -12.0, -4000.0,   0.031, 0.033, 0.0015];
		let st = [0.95, 0.90, 0.00];
		let er = [2.5, 3.0, 86.3];
		let s = gate_call(&vals, 3, &st, &er);
		assert!(s[2] > s[0] && s[2] > s[1],
			"tumbler must rank below both flyers: {:?}", s);
	}

	#[test]
	fn gate_all_feasible_is_bit_identical_to_base()
	{
		let vals = [-10.0, -12.0, -9.0,   0.031, 0.033, 0.030];
		let st = [0.95, 0.90, 0.80];
		let er = [2.5, 3.0, 4.0];
		let gated = gate_call(&vals, 3, &st, &er);
		let base = combine_flat(&vals, 3, &[0.7, 0.3], &[true, false], "zscore", 3.0).unwrap();
		assert_eq!(gated, base);
	}

	#[test]
	fn gate_none_feasible_orders_by_violation()
	{
		// Nothing flies: closer-to-flying must score lower (better).
		let vals = [-100.0, -200.0,   0.02, 0.02];
		let st = [0.60, 0.10];          // both below 0.70
		let er = [9.0, 40.0];           // both above 8.0
		let s = gate_call(&vals, 2, &st, &er);
		assert!(s[0] < s[1], "less-violating candidate must win: {:?}", s);
	}

	#[test]
	fn gate_subset_normalisation_excludes_infeasible()
	{
		// The infeasible member must not distort the feasible pair's ordering:
		// scores of the two flyers must equal a 2-candidate base combine.
		let vals = [-10.0, -12.0, -4000.0,   0.031, 0.033, 0.0015];
		let st = [0.95, 0.90, 0.00];
		let er = [2.5, 3.0, 86.3];
		let gated = gate_call(&vals, 3, &st, &er);
		let sub = combine_flat(&[-10.0, -12.0, 0.031, 0.033], 2,
			&[0.7, 0.3], &[true, false], "zscore", 3.0).unwrap();
		assert_eq!(&gated[..2], &sub[..]);
	}

	#[test]
	fn gate_boundary_is_inclusive()
	{
		// stable == S_min and err == E_max both PASS (>=, <=).
		let vals = [-10.0, -12.0,   0.031, 0.033];
		let st = [0.70, 0.95];
		let er = [8.0, 2.0];
		let s = gate_call(&vals, 2, &st, &er);
		let base = combine_flat(&vals, 2, &[0.7, 0.3], &[true, false], "zscore", 3.0).unwrap();
		assert_eq!(s, base);
	}

	#[test]
	fn gate_rejects_bad_inputs()
	{
		let vals = [1.0, 2.0];
		assert!(gated_combine_flat(&vals, 2, &[1.0], &[true], "zscore", 3.0,
			&[0.9], &[1.0, 2.0], 0.7, 8.0).is_err());          // gate len mismatch
		assert!(gated_combine_flat(&vals, 2, &[1.0], &[true], "zscore", 3.0,
			&[0.9, 0.9], &[1.0, 2.0], 0.0, 8.0).is_err());     // zero threshold
		assert!(gated_combine_flat(&vals, 2, &[1.0], &[true], "zscore", 3.0,
			&[0.9, f64::NAN], &[1.0, 2.0], 0.7, 8.0).is_err()); // non-finite gate
	}

	// --- desirability: Luiz's four intuitions as executable spec (26/08) ------

	/// stable (power, anchor .70, w .7) + alt (exp, anchor 1m, w .3) —
	/// the exact cases from the design discussion. Lower score = better.
	fn desir_stable_alt(stable: f64, alt: f64) -> f64
	{
		desirability_combine_flat(
			&[stable, alt], 1, &[0.7, 0.3], &["power", "exp"], &[0.70, 1.0],
		).unwrap()[0]
	}

	#[test]
	fn desirability_near_tie_when_both_good()
	{
		// 90%/0.10m vs 85%/0.01m: "we are not sure which is better" -> close.
		// Nearness lives in FITNESS space: ratio = 2^(-|dScore|). The score is
		// a log, which compresses near ideal, so a relative score gap is the
		// wrong ruler (both scores are tiny). Contrast with the decisive
		// case 2, whose fitness ratio is far from 1.
		let a = desir_stable_alt(0.90, 0.10);
		let b = desir_stable_alt(0.85, 0.01);
		let tie_ratio = 2.0_f64.powf(-(a - b).abs());
		assert!(tie_ratio > 0.90,
			"expected near-tie (fitness ratio > .90), got {} vs {} (ratio {})", a, b, tie_ratio);
		let c = desir_stable_alt(0.90, 2.0);
		let d = desir_stable_alt(0.10, 0.0);
		let decisive_ratio = 2.0_f64.powf(-(c - d).abs());
		assert!(decisive_ratio < 0.70,
			"case 2 must be decisive (ratio < .70), got ratio {}", decisive_ratio);
	}

	#[test]
	fn desirability_sure_when_stability_dominates()
	{
		// 90%/2m vs 10%/0m: "completely sure the 90% is better".
		assert!(desir_stable_alt(0.90, 2.0) < desir_stable_alt(0.10, 0.0));
	}

	#[test]
	fn desirability_flips_at_absurd_secondary()
	{
		// 90%/100m: "well, the formula would say something" — it flips.
		assert!(desir_stable_alt(0.90, 100.0) > desir_stable_alt(0.10, 0.0));
	}

	#[test]
	fn desirability_tumbler_cannot_buy_back_with_smoothness()
	{
		// The ABI-24 motivating bug: 0% stable / 86 deg err / jerk 0.0015
		// outranked flying genomes under the weighted SUM. Columns here:
		// stable(power,.70) err(exp,8) steady(exp,8) jerk(exp,.06),
		// S16noJM-style weights + a deliberately generous jerk weight.
		let w = [0.25, 0.3125, 0.4375, 0.30];
		let shapes = ["power", "exp", "exp", "exp"];
		let anchors = [0.70, 8.0, 8.0, 0.06];
		let score = |s: f64, e: f64, d: f64, j: f64| desirability_combine_flat(
			&[s, e, d, j], 1, &w, &shapes, &anchors).unwrap()[0];
		let tumbler = score(0.0, 86.0, 86.0, 0.0015);
		let flyer = score(0.90, 3.2, 3.0, 0.055);
		assert!(flyer < tumbler,
			"flyer {} must beat tumbler {} — the gate's job, emergent", flyer, tumbler);
	}

	#[test]
	fn desirability_anchor_is_exactly_one_half_life()
	{
		// u(anchor) = 0.5 <=> h = 1.0, for both shapes, at weight 1.
		let hp = desirability_combine_flat(
			&[0.70], 1, &[1.0], &["power"], &[0.70]).unwrap()[0];
		let he = desirability_combine_flat(
			&[8.0], 1, &[1.0], &["exp"], &[8.0]).unwrap()[0];
		assert!((hp - 1.0).abs() < 1e-12, "power h at anchor = {}", hp);
		assert!((he - 1.0).abs() < 1e-12, "exp h at anchor = {}", he);
	}

	#[test]
	fn desirability_caps_and_clamps()
	{
		// stable = 0 caps at H_CAP; negative cost clamps to ideal (h = 0);
		// fraction above 1 clamps to 1 (h = 0, never negative).
		let cap = desirability_combine_flat(
			&[0.0], 1, &[1.0], &["power"], &[0.70]).unwrap()[0];
		assert_eq!(cap, DESIRABILITY_H_CAP);
		let neg = desirability_combine_flat(
			&[-3.0], 1, &[1.0], &["exp"], &[8.0]).unwrap()[0];
		assert_eq!(neg, 0.0);
		let over = desirability_combine_flat(
			&[1.5], 1, &[1.0], &["power"], &[0.70]).unwrap()[0];
		assert_eq!(over, 0.0);
	}

	#[test]
	fn desirability_monotone_per_axis()
	{
		// Strictly better on one axis, equal elsewhere -> strictly lower score.
		assert!(desir_stable_alt(0.60, 0.5) < desir_stable_alt(0.50, 0.5));
		assert!(desir_stable_alt(0.60, 0.4) < desir_stable_alt(0.60, 0.5));
	}

	#[test]
	fn desirability_rejects_bad_inputs()
	{
		assert!(desirability_combine_flat(
			&[0.5], 1, &[1.0], &["power"], &[1.5]).is_err());   // power anchor >= 1
		assert!(desirability_combine_flat(
			&[0.5], 1, &[1.0], &["exp"], &[0.0]).is_err());     // exp anchor 0
		assert!(desirability_combine_flat(
			&[0.5], 1, &[1.0], &["nope"], &[0.5]).is_err());    // unknown shape
		assert!(desirability_combine_flat(
			&[f64::NAN], 1, &[1.0], &["exp"], &[8.0]).is_err()); // non-finite value
		assert!(desirability_combine_flat(
			&[0.5, 0.5], 1, &[1.0], &["exp"], &[8.0]).is_err()); // shape mismatch
	}
}

