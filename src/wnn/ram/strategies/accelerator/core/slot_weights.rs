//! `SlotWeights`: per-slot multiplicities of one DISTINCT-slot scored group.
//!
//! Duplicate-aware scoring (worker ABI 23): when the IDS multiplicity layer
//! trained duplicate neurons once, a cluster's m identical logical copies read
//! the same address through the same memory, so the scorer reads the unit once
//! and counts it m times — `score = sum_u m_u * w_u / N` with N the cluster's
//! LOGICAL neuron count. The kernels `sparse_forward_weighted`
//! (core/shaders/sparse_forward.metal) and `ramlm_forward_weighted`
//! (shaders/ramlm.metal) consume this; the CPU twins live in the worker's
//! `adaptive::eval_export`. Exactness is the caller's gate
//! (`adaptive/distinct_scoring.rs`).

/// Multiplicities of a group laid out at `stride` slots per cluster.
#[derive(Clone, Copy, Debug)]
pub struct SlotWeights<'a>
{
	/// Copies each slot stands for, cluster-major at `stride` (0 = pad)
	pub slot_mult: &'a [u32],
	/// Slots scored per cluster: the first `distinct[c]` of its stride
	pub distinct: &'a [u32],
	/// Logical neurons per cluster: the mean's denominator
	pub norm: &'a [u32],
}

impl SlotWeights<'_>
{
	/// Shape check against a group of `num_clusters` x `stride` slots, and
	/// `norm[c] == sum of the cluster's scored multiplicities` (the weighted
	/// mean must be over exactly the logical neurons).
	pub fn validate(&self, num_clusters: usize, stride: usize) -> Result<(), String>
	{
		if self.distinct.len() != num_clusters || self.norm.len() != num_clusters || self.slot_mult.len() != num_clusters * stride
		{
			return Err(format!(
				"SlotWeights shape: distinct {} norm {} slot_mult {} for {} clusters x {} slots",
				self.distinct.len(),
				self.norm.len(),
				self.slot_mult.len(),
				num_clusters,
				stride
			));
		}
		for c in 0..num_clusters
		{
			let d = self.distinct[c] as usize;
			if d > stride
			{
				return Err(format!("SlotWeights: cluster {c} scores {d} slots > stride {stride}"));
			}
			let total: u64 = self.slot_mult[c * stride..c * stride + d].iter().map(|&m| m as u64).sum();
			if total != self.norm[c] as u64
			{
				return Err(format!("SlotWeights: cluster {c} multiplicities sum {total} != norm {}", self.norm[c]));
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn slot_weights_validate_shape_and_norm()
	{
		let (mult, distinct, norm) = ([3u32, 2, 0, 1, 0, 0], [2u32, 1], [5u32, 1]);
		let w = SlotWeights { slot_mult: &mult, distinct: &distinct, norm: &norm };
		assert!(w.validate(2, 3).is_ok());
		assert!(w.validate(2, 2).is_err(), "stride mismatch");
		let bad_norm = [4u32, 1];
		assert!(SlotWeights { norm: &bad_norm, ..w }.validate(2, 3).is_err(), "norm != sum of multiplicities");
		let wide = [4u32, 1];
		assert!(SlotWeights { distinct: &wide, ..w }.validate(2, 3).is_err(), "distinct > stride");
	}
}
