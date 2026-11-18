//! ABI 59 (CTRL-58, Luiz 18/11/2026): the GA's near-fail TAIL — the 99th percentile of a
//! genome's per-episode mean attitude error over its evaluation episodes.
//!
//! The SAME definition `--select-tail p99` uses at stage-select (select_tail.py:
//! numpy.percentile(errs, 99, method="inverted_cdf")) — Hyndman & Fan method 1: the j-th
//! order statistic, j = ceil(n·p), p = 0.99 — except that a DIVERGED episode counts as the
//! run's gate err limit (`--gate-err`, 8.0° by default), not +inf: the GA ranks the column
//! through a ZRank, which refuses non-finite values, and "worse than the viability limit"
//! is what a diverged episode means to the gate.
//!
//! The scorers (score_controllers_cpu / score_controllers_metal, `episode_errs=True`) append
//! the per-episode errors (degrees, diverged → cap) to each genome's row; the evaluator
//! pools them across its CRN pools and calls `episode_tail_p99_deg` ONCE on the union, so
//! the column is the percentile over every evaluation episode of the genome — never a mean
//! of per-pool percentiles.

use pyo3::prelude::*;

/// The percentile the tail uses (fraction).
pub(crate) const TAIL_P: f64 = 0.99;

/// numpy `inverted_cdf` at `p`: the ceil(n·p)-th smallest (1-based). Errors on empty or
/// non-finite input (a tail over nothing, or over a NaN, is not a number to rank on).
pub(crate) fn percentile_inverted_cdf(values: &[f64], p: f64) -> Result<f64, String>
{
	if values.is_empty()
	{
		return Err("episode tail: no episodes".into());
	}
	if let Some(bad) = values.iter().find(|v| !v.is_finite())
	{
		return Err(format!("episode tail: non-finite episode error {bad} (diverged episodes must be capped)"));
	}
	let mut v = values.to_vec();
	v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
	let j = ((v.len() as f64) * p).ceil() as usize;
	Ok(v[j.clamp(1, v.len()) - 1])
}

/// One episode's tail value (degrees): its mean error, or `cap_deg` if it diverged.
#[inline]
pub(crate) fn episode_value_deg(mean_err_rad: f64, diverged: bool, cap_deg: f64) -> f64
{
	if diverged || !mean_err_rad.is_finite() { cap_deg } else { mean_err_rad.to_degrees() }
}

/// The GA tail column: p99 (inverted_cdf) of the pooled per-episode errors, degrees.
#[pyfunction]
pub fn episode_tail_p99_deg(errs_deg: Vec<f64>) -> PyResult<f64>
{
	percentile_inverted_cdf(&errs_deg, TAIL_P).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn inverted_cdf_order_statistic()
	{
		let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
		assert_eq!(percentile_inverted_cdf(&v, 0.99).unwrap(), 99.0);
		let w: Vec<f64> = (1..=200).rev().map(|i| i as f64).collect();
		assert_eq!(percentile_inverted_cdf(&w, 0.99).unwrap(), 198.0);
		assert_eq!(percentile_inverted_cdf(&[3.0], 0.99).unwrap(), 3.0);
		assert!(percentile_inverted_cdf(&[], 0.99).is_err());
		assert!(percentile_inverted_cdf(&[1.0, f64::INFINITY], 0.99).is_err());
		assert_eq!(episode_value_deg(0.01, true, 8.0), 8.0);
		assert_eq!(episode_value_deg(f64::NAN, false, 8.0), 8.0);
		assert!((episode_value_deg(0.01, false, 8.0) - 0.01f64.to_degrees()).abs() < 1e-15);
	}
}

/// A rollout's per-episode tail sink: one value per episode (degrees, diverged → cap).
#[derive(Clone, Debug, Default)]
pub(crate) struct EpisodeErrSink
{
	pub(crate) errs: Vec<f64>,
	pub(crate) cap_deg: f64,
}

impl EpisodeErrSink
{
	pub(crate) fn new(cap_deg: f64) -> Self
	{
		EpisodeErrSink { errs: Vec::new(), cap_deg }
	}

	pub(crate) fn push(&mut self, mean_err_rad: f64, diverged: bool)
	{
		self.errs.push(episode_value_deg(mean_err_rad, diverged, self.cap_deg));
	}

	/// The values a scorer row carries after its metrics: [n, e_1 .. e_n].
	pub(crate) fn row_tail(&self) -> Vec<f64>
	{
		std::iter::once(self.errs.len() as f64).chain(self.errs.iter().copied()).collect()
	}
}

/// The Metal per-episode export rows ([gi, ep, stable, diverged, mean_err rad, ...]) as one
/// per-genome tail list each, in EPISODE order (the CPU sink's order). Shared by
/// score_controllers_metal(episode_errs=) and the CPU/GPU parity test.
pub(crate) fn per_genome_from_episode_rows(episodes: &[Vec<f64>], n_genomes: usize, cap_deg: f64) -> Vec<Vec<f64>>
{
	let mut per_g: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_genomes];
	for r in episodes
	{
		per_g[r[0] as usize].push((r[1] as usize, episode_value_deg(r[4], r[3] != 0.0, cap_deg)));
	}
	per_g
		.into_iter()
		.map(|mut eps| {
			eps.sort_by_key(|e| e.0);
			eps.into_iter().map(|e| e.1).collect()
		})
		.collect()
}
