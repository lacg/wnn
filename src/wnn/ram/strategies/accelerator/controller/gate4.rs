//! ABI 59 (CTRL-58, Luiz 18/11/2026): the NATIVE four-input viability gate (stable, err,
//! steady, pos) for the staged-refinement fitness — replaces the Python composite that folded
//! the steady / pos excess into the err slot (err' = E·(1+V)).
//!
//! WHERE IT LIVES: the controller crate, on top of the SHARED combine. ram_core::fitness keeps
//! the two-input `gated_combine_flat` (both wheels link it; the IDS worker is untouched) and
//! the base `combine_flat`, which this gate calls for the feasible subset. With the steady and
//! pos bounds unset `gated4_combine_flat` IS `ram_core::fitness::gated_combine_flat` (same call,
//! same arguments): the two-input behaviour is bit-identical by construction.
//!
//! SEMANTICS (Deb's rules, the shared gate's, extended to N bounds):
//!   feasible  = stable >= S AND err <= E AND steady <= D AND pos <= P (inclusive)
//!   feasible vs infeasible   feasible always wins
//!   feasible vs feasible     base combine over the FEASIBLE SUBSET only
//!   infeasible vs infeasible smaller summed normalised violation wins:
//!                            V = (S-stable)+/S + (err-E)+/E + (steady-D)+/D + (pos-P)+/P
//!   infeasible score         worst feasible + 1 + V (or V alone when nothing is feasible)

/// One extra upper-bound gate input: per-candidate values and the bound (same units).
pub(crate) struct UpperBound<'a>
{
	pub(crate) values: &'a [f64],
	pub(crate) max: f64,
	pub(crate) name: &'static str,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gated4_combine_flat(
	values_flat: &[f64],
	num_candidates: usize,
	weights: &[f64],
	higher_is_better: &[bool],
	mode: &str,
	clamp: f64,
	scale_floors: Option<&[f64]>,
	gate_stable: &[f64],
	gate_err: &[f64],
	gate_stable_min: f64,
	gate_err_max: f64,
	extra: &[UpperBound],
) -> Result<Vec<f64>, String>
{
	if extra.is_empty()
	{
		return ram_core::fitness::gated_combine_flat(
			values_flat, num_candidates, weights, higher_is_better, mode, clamp, scale_floors,
			gate_stable, gate_err, gate_stable_min, gate_err_max);
	}
	for b in extra
	{
		if !(b.max > 0.0 && b.max.is_finite())
		{
			return Err(format!("gated4_combine_flat: {} bound must be positive finite (got {})", b.name, b.max));
		}
		if b.values.len() != num_candidates
		{
			return Err(format!("gated4_combine_flat: {} gate vector has {} values for {} candidates", b.name, b.values.len(), num_candidates));
		}
		if let Some(v) = b.values.iter().find(|v| !v.is_finite())
		{
			return Err(format!("gated4_combine_flat: non-finite {} gate value {}", b.name, v));
		}
	}
	// The two shared inputs, validated with the shared gate's rules.
	if !(gate_stable_min > 0.0) || !gate_stable_min.is_finite() || !(gate_err_max > 0.0) || !gate_err_max.is_finite()
	{
		return Err(format!(
			"gated4_combine_flat: thresholds must be positive finite (stable_min={gate_stable_min}, err_max={gate_err_max})"));
	}
	if gate_stable.len() != num_candidates || gate_err.len() != num_candidates
	{
		return Err(format!(
			"gated4_combine_flat: gate vectors ({} stable, {} err) != {} candidates",
			gate_stable.len(), gate_err.len(), num_candidates));
	}
	if let Some(bad) = gate_stable.iter().chain(gate_err.iter()).find(|v| !v.is_finite())
	{
		return Err(format!("gated4_combine_flat: non-finite gate value {bad}"));
	}
	let feasible: Vec<bool> = (0..num_candidates)
		.map(|i| {
			gate_stable[i] >= gate_stable_min
				&& gate_err[i] <= gate_err_max
				&& extra.iter().all(|b| b.values[i] <= b.max)
		})
		.collect();
	let violation = |i: usize| -> f64 {
		let mut v = (gate_stable_min - gate_stable[i]).max(0.0) / gate_stable_min
			+ (gate_err[i] - gate_err_max).max(0.0) / gate_err_max;
		for b in extra
		{
			v += (b.values[i] - b.max).max(0.0) / b.max;
		}
		v
	};
	let n_feasible = feasible.iter().filter(|f| **f).count();
	if n_feasible == num_candidates
	{
		return ram_core::fitness::combine_flat(
			values_flat, num_candidates, weights, higher_is_better, mode, clamp, scale_floors);
	}
	if n_feasible == 0
	{
		return Ok((0..num_candidates).map(violation).collect());
	}
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
	let sub = ram_core::fitness::combine_flat(&sub_flat, n_feasible, weights, higher_is_better, mode, clamp, scale_floors)?;
	let worst = sub.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let mut out = vec![0.0; num_candidates];
	for (k, &i) in idx.iter().enumerate()
	{
		out[i] = sub[k];
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

#[cfg(test)]
mod tests
{
	use super::*;

	fn cols(n: usize, seed: u64) -> Vec<f64>
	{
		use rand::{Rng, SeedableRng};
		let mut r = rand::rngs::SmallRng::seed_from_u64(seed);
		(0..3 * n).map(|_| r.gen_range(0.0..2.0)).collect()
	}

	fn call(vals: &[f64], n: usize, st: &[f64], er: &[f64], extra: &[UpperBound]) -> Vec<f64>
	{
		gated4_combine_flat(vals, n, &[0.5, 0.3, 0.2], &[false, false, true], "zscore", 3.0, None, st, er, 0.99, 1.5, extra).unwrap()
	}

	/// Unset steady / pos bounds ⇒ the shared two-input gate, bit for bit.
	#[test]
	fn no_extra_bounds_is_the_shared_gate()
	{
		for seed in 0..40u64
		{
			let n = 12;
			let v = cols(n, seed);
			let st: Vec<f64> = (0..n).map(|i| if (i + seed as usize) % 4 == 0 { 0.95 } else { 1.0 }).collect();
			let er: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * ((i * 7 + seed as usize) % 9) as f64).collect();
			let a = call(&v, n, &st, &er, &[]);
			let b = ram_core::fitness::gated_combine_flat(&v, n, &[0.5, 0.3, 0.2], &[false, false, true], "zscore", 3.0, None, &st, &er, 0.99, 1.5).unwrap();
			assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
		}
	}

	/// The four-bound gate: a steady or pos violation makes a candidate infeasible exactly
	/// like an err violation (bounds inclusive), infeasibles order by the summed violation,
	/// and the feasible subset is ranked by the base combine over that subset.
	#[test]
	fn four_bounds_deb_order()
	{
		let n = 6;
		let v = cols(n, 9);
		let st = [1.0; 6];
		let er = [1.0, 1.0, 1.0, 1.6, 1.0, 1.0];
		let steady = [0.5, 1.0, 1.2, 0.5, 0.5, 2.0];
		let pos = [0.3, 0.3, 0.3, 0.3, 0.41, 0.3];
		let b = [UpperBound { values: &steady, max: 1.0, name: "steady" }, UpperBound { values: &pos, max: 0.40, name: "pos" }];
		let out = call(&v, n, &st, &er, &b);
		let feasible = [true, true, false, false, false, false];
		let worst = out.iter().zip(feasible).filter(|(_, f)| *f).map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
		for i in 0..n
		{
			if !feasible[i]
			{
				assert!(out[i] > worst, "infeasible {i} must rank behind every feasible");
			}
		}
		let viol = |i: usize| (er[i] - 1.5f64).max(0.0) / 1.5 + (steady[i] - 1.0f64).max(0.0) / 1.0 + (pos[i] - 0.4f64).max(0.0) / 0.4;
		for i in 2..n
		{
			assert!((out[i] - (worst + 1.0 + viol(i))).abs() < 1e-12, "candidate {i}: {} vs worst+1+V {}", out[i], worst + 1.0 + viol(i));
		}
		let sub: Vec<f64> = (0..3).flat_map(|c| [v[c * n], v[c * n + 1]]).collect();
		let base = ram_core::fitness::combine_flat(&sub, 2, &[0.5, 0.3, 0.2], &[false, false, true], "zscore", 3.0, None).unwrap();
		assert_eq!([out[0], out[1]], [base[0], base[1]], "feasible subset = base combine over the subset");
		// nothing feasible ⇒ pure violation
		let none = call(&v, n, &[0.5; 6], &er, &b);
		assert!((none[0] - ((0.99 - 0.5) / 0.99)).abs() < 1e-12);
	}

	#[test]
	fn four_bounds_refusals()
	{
		let v = cols(2, 1);
		let s = [0.5, 0.5];
		let bad = [UpperBound { values: &s, max: 0.0, name: "steady" }];
		assert!(gated4_combine_flat(&v, 2, &[0.5, 0.3, 0.2], &[false; 3], "zscore", 3.0, None, &[1.0, 1.0], &[1.0, 1.0], 0.99, 1.5, &bad).is_err());
		let short = [0.5];
		let b2 = [UpperBound { values: &short, max: 1.0, name: "pos" }];
		assert!(gated4_combine_flat(&v, 2, &[0.5, 0.3, 0.2], &[false; 3], "zscore", 3.0, None, &[1.0, 1.0], &[1.0, 1.0], 0.99, 1.5, &b2).is_err());
		let nan = [f64::NAN, 0.5];
		let b3 = [UpperBound { values: &nan, max: 1.0, name: "steady" }];
		assert!(gated4_combine_flat(&v, 2, &[0.5, 0.3, 0.2], &[false; 3], "zscore", 3.0, None, &[1.0, 1.0], &[1.0, 1.0], 0.99, 1.5, &b3).is_err());
	}
}
