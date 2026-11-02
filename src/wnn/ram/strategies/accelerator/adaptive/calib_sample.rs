//! Subsampled threshold calibration for the single-cluster GA fitness
//! (flow param `ids_calib_sample`, worker ABI 24).
//!
//! The single-cluster search picks each genome x fold's decision threshold by
//! scoring the fold's TRAINING rows and maximising F1 (or the flow's fitness).
//! On cicids that is ~1.81M rows per genome-fold, ~72% of the scoring time. A
//! `CalibSet` replaces those rows with a FIXED stratified subsample:
//!
//! * fixed — the rows are a pure function of (seed, fold, labels), so every
//!   genome of a fold is calibrated on the same rows and a rerun picks them
//!   again (`ram_core::counter_rng` coordinates; no stream state);
//! * stratified — rows are grouped by (target, stratum label) and every group
//!   keeps its share (largest-remainder rounding) with at least one row, so
//!   the attack/benign proportion holds and no attack class vanishes;
//! * ordered — the chosen rows stay in train-set order.
//!
//! Only the SEARCH-time calibration reads it (IDSCache k-fold / train-subset
//! evaluation). Validation thresholds (train_cal, val_cal, platt, ...) never
//! see a CalibSet. `sample == 0` or `sample >= rows` means the full set —
//! `CalibSet::build` returns None and the caller runs the unchanged path.

use ram_core::packed_bits::PackedBits;
use std::collections::BTreeMap;

/// `counter_rng` layer coordinate reserved for the calibration draw.
const CALIB_LAYER: u64 = 0xCA11_B5A3;

/// The rows a search-time threshold is calibrated on (see module docs).
pub struct CalibSet
{
	/// Chosen rows, as indices into the call's train set, ascending.
	pub rows: Vec<usize>,
	/// Their packed input rows, gathered once per call.
	pub input_bits: PackedBits,
	/// Their targets.
	pub targets: Vec<i64>,
}

impl CalibSet
{
	/// The calibration set for one train set, or None for the full set.
	/// `strata[k]` refines row k's group beyond its target (e.g. the attack
	/// class); `fold` keys the draw so each fold has its own fixed rows.
	pub fn build(
		train_bits: &PackedBits,
		targets: &[i64],
		strata: Option<&[i64]>,
		sample: usize,
		seed: u64,
		fold: u64,
	) -> Option<CalibSet>
	{
		let rows = stratified_rows(targets, strata, sample, seed, fold)?;
		Some(CalibSet {
			input_bits: train_bits.select_rows(&rows),
			targets: rows.iter().map(|&r| targets[r]).collect(),
			rows,
		})
	}

	pub fn num_examples(&self) -> usize
	{
		self.rows.len()
	}
}

/// Stratified, seeded row choice of size ~min(sample, n) (exactly that unless
/// there are more groups than `sample` — every group keeps one row). None when
/// the sample is the full set (0 or >= n).
pub fn stratified_rows(
	targets: &[i64],
	strata: Option<&[i64]>,
	sample: usize,
	seed: u64,
	fold: u64,
) -> Option<Vec<usize>>
{
	let n = targets.len();
	if sample == 0 || sample >= n
	{
		return None;
	}
	if let Some(s) = strata
	{
		assert_eq!(s.len(), n, "calib strata length {} != train rows {}", s.len(), n);
	}
	let groups = group_rows(targets, strata);
	let sizes: Vec<usize> = groups.values().map(|g| g.len()).collect();
	let quotas = allocate(&sizes, sample);
	let mut rows: Vec<usize> = Vec::with_capacity(quotas.iter().sum());
	for (gi, (group, &quota)) in groups.into_values().zip(&quotas).enumerate()
	{
		rows.extend(draw(group, quota, seed, fold, gi as u64));
	}
	rows.sort_unstable();
	Some(rows)
}

/// Row indices per (target, stratum) group, groups in key order.
fn group_rows(targets: &[i64], strata: Option<&[i64]>) -> BTreeMap<(i64, i64), Vec<usize>>
{
	let mut groups: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
	for (k, &t) in targets.iter().enumerate()
	{
		let s = strata.map_or(0, |s| s[k]);
		groups.entry((t, s)).or_default().push(k);
	}
	groups
}

/// Proportional quotas summing to `sample` (largest remainder, ties to the
/// earlier group), each group at least 1 and at most its size.
fn allocate(sizes: &[usize], sample: usize) -> Vec<usize>
{
	let n: usize = sizes.iter().sum();
	let exact: Vec<f64> = sizes.iter().map(|&c| c as f64 * sample as f64 / n as f64).collect();
	let mut quotas: Vec<usize> = exact.iter().zip(sizes).map(|(&e, &c)| (e.floor() as usize).clamp(1, c)).collect();
	let mut total: usize = quotas.iter().sum();
	// Grow: largest fractional remainder first, among groups with room.
	let mut order: Vec<usize> = (0..sizes.len()).collect();
	order.sort_by(|&a, &b| (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor())).then(a.cmp(&b)));
	let mut grew = true;
	while total < sample && grew
	{
		grew = false;
		for &g in &order
		{
			if total < sample && quotas[g] < sizes[g]
			{
				quotas[g] += 1;
				total += 1;
				grew = true;
			}
		}
	}
	// Shrink (the at-least-one floor overshot): trim the largest quotas.
	while total > sample
	{
		let g = (0..quotas.len()).filter(|&g| quotas[g] > 1).max_by_key(|&g| (quotas[g], usize::MAX - g));
		match g
		{
			Some(g) =>
			{
				quotas[g] -= 1;
				total -= 1;
			}
			None => break, // more groups than `sample`: one row each
		}
	}
	quotas
}

/// `quota` rows of `group` by a seeded partial Fisher-Yates shuffle.
fn draw(mut group: Vec<usize>, quota: usize, seed: u64, fold: u64, group_idx: u64) -> Vec<usize>
{
	let c = group.len();
	for i in 0..quota.min(c)
	{
		let j = i + ram_core::counter_rng::below((c - i) as u64, seed, fold, group_idx, CALIB_LAYER, i as u64, 0) as usize;
		group.swap(i, j);
	}
	group.truncate(quota.min(c));
	group
}

#[cfg(test)]
mod tests
{
	use super::*;

	fn labels(n_benign: usize, n_attack: usize) -> Vec<i64>
	{
		let mut t = vec![0i64; n_benign];
		t.extend(std::iter::repeat(1).take(n_attack));
		t
	}

	#[test]
	fn absent_or_oversized_sample_is_the_full_set()
	{
		let t = labels(90, 10);
		assert!(stratified_rows(&t, None, 0, 7, 0).is_none());
		assert!(stratified_rows(&t, None, 100, 7, 0).is_none());
		assert!(stratified_rows(&t, None, 1_000, 7, 0).is_none());
	}

	#[test]
	fn deterministic_per_seed_and_fold()
	{
		let t = labels(9_000, 1_000);
		let a = stratified_rows(&t, None, 500, 42, 3).unwrap();
		let b = stratified_rows(&t, None, 500, 42, 3).unwrap();
		assert_eq!(a, b);
		assert_ne!(a, stratified_rows(&t, None, 500, 42, 4).unwrap(), "fold must key the draw");
		assert_ne!(a, stratified_rows(&t, None, 500, 43, 3).unwrap(), "seed must key the draw");
	}

	#[test]
	fn stratified_proportions_and_size()
	{
		// Interleaved labels so ordering cannot fake the proportions.
		let t: Vec<i64> = (0..10_000).map(|k| if k % 10 == 3 { 1 } else { 0 }).collect();
		let rows = stratified_rows(&t, None, 1_000, 1, 0).unwrap();
		assert_eq!(rows.len(), 1_000);
		assert!(rows.windows(2).all(|w| w[0] < w[1]), "ascending, no duplicates");
		let attacks = rows.iter().filter(|&&r| t[r] == 1).count();
		assert_eq!(attacks, 100, "10% attack share preserved exactly");
	}

	#[test]
	fn every_stratum_survives_including_rare_attack_classes()
	{
		// 9_990 benign (stratum 0), 1_000 class-1 attacks, 3 class-2 attacks.
		let n = 9_990 + 1_000 + 3;
		let t: Vec<i64> = (0..n).map(|k| if k < 9_990 { 0 } else { 1 }).collect();
		let s: Vec<i64> = (0..n).map(|k| if k < 9_990 { 0 } else if k < 10_990 { 1 } else { 2 }).collect();
		let rows = stratified_rows(&t, Some(&s), 100, 5, 0).unwrap();
		assert_eq!(rows.len(), 100);
		assert!(rows.iter().any(|&r| s[r] == 2), "rare attack class kept");
		let benign = rows.iter().filter(|&&r| t[r] == 0).count();
		assert!((89..=91).contains(&benign), "benign share ~90.8%, got {}", benign);
	}

	#[test]
	fn more_groups_than_sample_keeps_one_each()
	{
		let t: Vec<i64> = (0..50).map(|k| k % 5).collect();
		let rows = stratified_rows(&t, None, 3, 0, 0).unwrap();
		assert_eq!(rows.len(), 5);
	}

	#[test]
	fn calib_set_gathers_matching_rows()
	{
		let n = 64;
		let bools: Vec<bool> = (0..n * 9).map(|i| (i * 7) % 3 == 0).collect();
		let bits = PackedBits::from_bool_slice(&bools, 9);
		let t: Vec<i64> = (0..n as i64).map(|k| k % 4 / 3).collect();
		let set = CalibSet::build(&bits, &t, None, 16, 9, 1).unwrap();
		assert_eq!(set.num_examples(), 16);
		for (k, &r) in set.rows.iter().enumerate()
		{
			assert_eq!(set.input_bits.row_as_bools(k), bits.row_as_bools(r));
			assert_eq!(set.targets[k], t[r]);
		}
	}
}
