//! Axis receptive fields (CTRL-45, docs/ctrl45_axis_receptive_fields_design.md).
//!
//! A CONSTRAINT on the output layer's connectivity map, not a new neuron type: an
//! output neuron of bank b (CTBR [T, p, q, r] — one control axis per bank) may hold
//! at most `max_cross` connections to input features OUTSIDE its bank's axis family.
//! It is ONE idempotent re-projection applied after every genome operator that can
//! create or move an output connection (init, mutate, both crossovers, genesis) —
//! the `rebalance_features` pattern — so no operator has to know about families.
//!
//! Host-side GA operator: the runtime (CPU step, Metal kernels, trainers, MCU
//! exporter) only ever sees connection INDICES, so there is no GPU twin to keep in
//! parity. Families are bitmasks (a feature may serve several axes); the masks come
//! from Python's feature_layout.py, the single name/family table.
//!
//! EXCLUDED FEATURES (CTRL-48, ABI 45, `--drop-features` / `--drop-constant-features`):
//! a feature whose mask is 0 belongs to no family and is never allowed, not even as
//! a cross connection — every connection on it is redrawn (in-family first, then any
//! allowed bit). With receptive fields off the host passes one all-axes bank and a
//! cross budget = width, so the projection only clears excluded features. The frame
//! layout (and therefore every kernel, the thresholds and the MCU export) is
//! unchanged: an excluded feature is still encoded, no neuron reads it.
//!
//! Draw coordinates (index, sub) = (0x4000 + neuron, t) — disjoint from every other
//! operator in arch_ops.rs (0x10, 0x2000+j, 0x30, 0xF0, per-position j).

use ram_core::counter_rng;

/// The fixed geometry one projection works over.
pub struct RfLayout<'a>
{
	pub space: usize,            // output input space (frames x frame_bits)
	pub frame_bits: usize,       // one frame = num_features x bpf
	pub bpf: usize,              // thermometer bits per feature
	pub levels: usize,           // output neurons per bank: bank = n / levels
	pub bank_family: &'a [u8],   // per bank
	pub feature_family: &'a [u8], // per feature (len = frame_bits / bpf)
	pub max_cross: usize,        // out-of-family connections allowed per neuron
}

/// The counter-RNG coordinates of one call.
#[derive(Clone, Copy)]
pub struct RfDraw
{
	pub seed: u64,
	pub generation: u64,
	pub genome: u64,
	pub layer: u64,
}

const RF_INDEX: u64 = 0x4000;
const RF_POOL_SUB: u64 = 0x10_0000;
const RF_ANY_SUB: u64 = 0x20_0000;

impl RfLayout<'_>
{
	fn validate(&self, n_neurons: usize) -> Result<(), String>
	{
		if self.levels == 0 || self.bpf == 0 || self.frame_bits == 0 || self.frame_bits % self.bpf != 0
		{
			return Err(format!(
				"receptive fields: bad geometry levels={} bpf={} frame_bits={}",
				self.levels, self.bpf, self.frame_bits
			));
		}
		if self.feature_family.len() != self.frame_bits / self.bpf
		{
			return Err(format!(
				"receptive fields: {} feature masks for {} features",
				self.feature_family.len(),
				self.frame_bits / self.bpf
			));
		}
		if n_neurons.div_ceil(self.levels) > self.bank_family.len()
		{
			return Err(format!(
				"receptive fields: {n_neurons} neurons at {} per bank need more than {} bank masks",
				self.levels,
				self.bank_family.len()
			));
		}
		Ok(())
	}

	#[inline]
	fn mask_of(&self, bit: i64) -> u8
	{
		self.feature_family[(bit as usize % self.frame_bits) / self.bpf]
	}

	#[inline]
	fn in_family(&self, bit: i64, family: u8) -> bool
	{
		self.mask_of(bit) & family != 0
	}
}

/// Re-project every output neuron onto its bank's family + cross budget, in place.
/// Returns the indices of the neurons whose map changed (their cells must go).
///
/// Per neuron: if it holds more than `max_cross` out-of-family connections, keep a
/// counter-RNG-chosen `max_cross` of them and redraw the rest uniformly from the
/// family's unused bits, in the same slots. Valid neurons are untouched and draw
/// nothing (idempotent). When the family pool is too small for the excess (only
/// after a large BITS grow), the remaining excess stays out-of-family — a
/// deterministic, documented fallback, as in arch_ops::sample_min_per_cluster.
pub fn project_receptive_fields(
	sampled: &mut [i64],
	offsets: &[usize],
	layout: &RfLayout,
	draw: RfDraw,
) -> Result<Vec<u32>, String>
{
	let n_neurons = offsets.len().saturating_sub(1);
	layout.validate(n_neurons)?;
	let mut changed = Vec::new();
	for n in 0..n_neurons
	{
		let family = layout.bank_family[n / layout.levels];
		if project_neuron(&mut sampled[offsets[n]..offsets[n + 1]], n, family, layout, draw)?
		{
			changed.push(n as u32);
		}
	}
	Ok(changed)
}

/// One neuron. Returns true when any connection was replaced. Excluded slots are
/// redrawn first (so they win the family pool); with no excluded feature this is the
/// ABI-44 projection draw for draw.
fn project_neuron(suffix: &mut [i64], n: usize, family: u8, layout: &RfLayout, draw: RfDraw) -> Result<bool, String>
{
	let excluded: Vec<usize> = (0..suffix.len()).filter(|&j| layout.mask_of(suffix[j]) == 0).collect();
	let mut cross: Vec<usize> = (0..suffix.len())
		.filter(|&j| layout.mask_of(suffix[j]) != 0 && !layout.in_family(suffix[j], family))
		.collect();
	if excluded.is_empty() && cross.len() <= layout.max_cross
	{
		return Ok(false);
	}
	let index = RF_INDEX + n as u64;
	let keep = layout.max_cross.min(cross.len());
	shuffle_prefix(&mut cross, keep, index, 0, draw);
	let mut redraw = excluded.clone();
	redraw.extend_from_slice(&cross[keep..]);
	let fresh = draw_family_bits(suffix, family, redraw.len(), layout, index, draw);
	for (&slot, &bit) in redraw.iter().zip(fresh.iter())
	{
		suffix[slot] = bit;
	}
	if fresh.len() < excluded.len()
	{
		fill_excluded_anywhere(suffix, &excluded[fresh.len()..], layout, index, draw)?;
	}
	Ok(!fresh.is_empty() || !excluded.is_empty())
}

/// Family pool exhausted: the remaining excluded slots take any allowed (mask != 0)
/// bit the neuron does not hold. Refuses when even that cannot clear them.
fn fill_excluded_anywhere(suffix: &mut [i64], slots: &[usize], layout: &RfLayout, index: u64, draw: RfDraw)
	-> Result<(), String>
{
	let fresh = draw_family_bits(suffix, 0xFF, slots.len(), layout, index, RfDraw { layer: draw.layer ^ RF_ANY_SUB, ..draw });
	if fresh.len() < slots.len()
	{
		return Err(format!(
			"excluded features: a {}-bit neuron cannot be wired without them ({} allowed bits left)",
			suffix.len(),
			fresh.len()
		));
	}
	for (&slot, &bit) in slots.iter().zip(fresh.iter())
	{
		suffix[slot] = bit;
	}
	Ok(())
}

/// Partial Fisher-Yates: after it, v[..k] is a uniform k-subset of v. Draw t uses
/// sub = sub0 + t.
fn shuffle_prefix<T>(v: &mut [T], k: usize, index: u64, sub0: u64, draw: RfDraw)
{
	for t in 0..k.min(v.len())
	{
		let pick = t
			+ counter_rng::below(
				(v.len() - t) as u64,
				draw.seed,
				draw.generation,
				draw.genome,
				draw.layer,
				index,
				sub0 + t as u64,
			) as usize;
		v.swap(t, pick);
	}
}

/// Up to `k` distinct in-family bits the suffix does not already hold, uniform
/// without replacement (fewer if the pool is smaller).
fn draw_family_bits(suffix: &[i64], family: u8, k: usize, layout: &RfLayout, index: u64, draw: RfDraw)
	-> Vec<i64>
{
	let held: std::collections::HashSet<i64> = suffix.iter().copied().collect();
	let mut pool: Vec<i64> = (0..layout.space as i64)
		.filter(|&b| layout.in_family(b, family) && !held.contains(&b))
		.collect();
	let k = k.min(pool.len());
	shuffle_prefix(&mut pool, k, index, RF_POOL_SUB, draw);
	pool.truncate(k);
	pool
}

#[cfg(test)]
mod tests
{
	use super::*;

	// 6 features x 4 bits, one frame; feature families R R P P Y T (bitmask).
	const BPF: usize = 4;
	const FRAME: usize = 24;
	const FEAT: [u8; 6] = [1, 1, 2, 2, 4, 8];
	const BANKS: [u8; 4] = [8, 1, 2, 4]; // T, roll, pitch, yaw

	fn layout(max_cross: usize) -> RfLayout<'static>
	{
		RfLayout {
			space: FRAME,
			frame_bits: FRAME,
			bpf: BPF,
			levels: 2,
			bank_family: &BANKS,
			feature_family: &FEAT,
			max_cross,
		}
	}

	const DRAW: RfDraw = RfDraw { seed: 9, generation: 0, genome: 0, layer: 0 };

	/// 8 neurons (2 per bank), width 6, every neuron spread over all features.
	fn genome() -> (Vec<i64>, Vec<usize>)
	{
		let flat: Vec<i64> = (0..8).flat_map(|n| (0..6).map(move |f| (f * BPF + (n % BPF)) as i64)).collect();
		(flat, (0..=8).map(|n| n * 6).collect())
	}

	fn out_of_family(flat: &[i64], off: &[usize], l: &RfLayout) -> Vec<usize>
	{
		(0..off.len() - 1)
			.map(|n| {
				let fam = l.bank_family[n / l.levels];
				flat[off[n]..off[n + 1]].iter().filter(|&&b| !l.in_family(b, fam)).count()
			})
			.collect()
	}

	#[test]
	fn invariant_holds_and_maps_stay_distinct()
	{
		let (mut flat, off) = genome();
		let l = layout(1);
		let changed = project_receptive_fields(&mut flat, &off, &l, DRAW).unwrap();
		assert_eq!(changed.len(), 8, "every spread neuron exceeds a budget of 1");
		// Roll/pitch own 2 features (8 bits): 1 cross + 5 in-family fits. T and yaw own
		// ONE feature (4 bits, 1 already held): only 3 fresh in-family bits exist for 4
		// excess slots, so the documented fallback leaves one extra out-of-family.
		assert_eq!(out_of_family(&flat, &off, &l), vec![2, 2, 1, 1, 1, 1, 2, 2]);
		for n in 0..8
		{
			let s: std::collections::HashSet<i64> = flat[off[n]..off[n + 1]].iter().copied().collect();
			assert_eq!(s.len(), 6, "neuron {n} lost distinctness");
			assert!(s.iter().all(|&b| (0..FRAME as i64).contains(&b)));
		}
	}

	#[test]
	fn idempotent_and_deterministic()
	{
		let (mut a, off) = genome();
		let (mut b, _) = genome();
		let l = layout(2);
		project_receptive_fields(&mut a, &off, &l, DRAW).unwrap();
		project_receptive_fields(&mut b, &off, &l, DRAW).unwrap();
		assert_eq!(a, b, "same coordinates must give the same projection");
		let again = a.clone();
		let changed = project_receptive_fields(&mut a, &off, &l, RfDraw { seed: 77, ..DRAW }).unwrap();
		assert!(changed.is_empty() && a == again, "a valid genome must be a fixed point");
	}

	#[test]
	fn budget_zero_is_pure_family_and_full_budget_is_a_no_op()
	{
		let (mut flat, off) = genome();
		let mut l = layout(0);
		l.feature_family = &[1, 1, 2, 2, 4, 4]; // every family owns 8 bits: room for all 6
		l.bank_family = &[1, 1, 2, 4];
		project_receptive_fields(&mut flat, &off, &l, DRAW).unwrap();
		assert_eq!(out_of_family(&flat, &off, &l), vec![0; 8], "budget 0 must leave only in-family bits");
		let (mut g, _) = genome();
		let before = g.clone();
		assert!(project_receptive_fields(&mut g, &off, &layout(6), DRAW).unwrap().is_empty());
		assert_eq!(g, before, "budget = width must not touch anything");
	}

	#[test]
	fn excluded_features_are_cleared_even_with_a_full_budget()
	{
		// RF off shape: one all-axes bank, budget = width; feature 5 (T) excluded.
		let (mut flat, off) = genome();
		let feat = [1u8, 1, 2, 2, 4, 0];
		let l = RfLayout { levels: 8, bank_family: &[0xFF], feature_family: &feat, ..layout(6) };
		let changed = project_receptive_fields(&mut flat, &off, &l, DRAW).unwrap();
		assert_eq!(changed.len(), 8, "every neuron held a bit of the excluded feature");
		assert!(flat.iter().all(|&b| (b as usize) / BPF != 5), "an excluded bit survived");
		for n in 0..8
		{
			let s: std::collections::HashSet<i64> = flat[off[n]..off[n + 1]].iter().copied().collect();
			assert_eq!(s.len(), 6, "neuron {n} lost distinctness");
		}
		let again = flat.clone();
		assert!(project_receptive_fields(&mut flat, &off, &l, DRAW).unwrap().is_empty() && flat == again);
		// With receptive fields on: excluded never survives as a cross connection.
		let (mut flat, off) = genome();
		let feat = [1u8, 1, 2, 2, 0, 8];
		let l = RfLayout { feature_family: &feat, ..layout(6) };
		project_receptive_fields(&mut flat, &off, &l, DRAW).unwrap();
		assert!(flat.iter().all(|&b| (b as usize) / BPF != 4), "excluded kept as cross");
		// No allowed bit left for a full-width neuron -> refused, never silently kept.
		let (mut flat, off) = genome();
		let feat = [0u8, 0, 0, 0, 0, 8];
		let l = RfLayout { levels: 8, bank_family: &[0xFF], feature_family: &feat, ..layout(6) };
		assert!(project_receptive_fields(&mut flat, &off, &l, DRAW).is_err());
	}

	#[test]
	fn refuses_bad_geometry()
	{
		let (mut flat, off) = genome();
		let mut l = layout(1);
		l.feature_family = &[1, 2];
		assert!(project_receptive_fields(&mut flat, &off, &l, DRAW).is_err());
		let mut l = layout(1);
		l.levels = 1; // 8 banks needed, 4 masks
		assert!(project_receptive_fields(&mut flat, &off, &l, DRAW).is_err());
	}
}
