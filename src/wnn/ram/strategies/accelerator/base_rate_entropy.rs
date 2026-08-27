//! Base-rate entropy — the cross-entropy of the trivial "predict the class
//! priors" classifier, in nats. This is the reference scale for the
//! desirability CE half-anchor.
//!
//! CE is the only column in the desirability vector with no absolute meaning:
//! it carries units and grows with both the class count and the class
//! imbalance. The frozen anchor 0.133 (median best_ce over 18,355 unsw-nb15
//! BINARY IDSZ iterations) is 14.3x too small for the SAME dataset's 10-class
//! task, which pushes the column toward the h<=20 clamp — killing its gradient
//! for weak genomes — and inflates its share of the score far above the weight
//! the operator set. Every other desirability column is either bounded (stable,
//! f1, acc) or carries an absolute domain threshold (fpr 0.10, err 8 deg), so
//! this is the one column that needs a reference scale.
//!
//! `ce / H(p)` is "log-loss as a fraction of what predicting the priors would
//! cost": 1.0 means "no better than the base rate" on every dataset and every
//! class count, so ONE normalized anchor is portable.
//!
//! THE CONSTANT IS 0.2128, AND WHICH PARTITION IT COMES FROM MATTERS. The
//! anchor scales the CE the FITNESS sees, and that is the during-search k-fold
//! CE — computed on TRAIN folds. So H must be the TRAIN partition's entropy,
//! which is what `IDSCache::desirability_ce_anchor` reads. unsw-nb15 temporal
//! is 32.0% benign in train (56,000 / 175,341) but 44.9% in test, so the two
//! entropies differ materially: 0.6264 vs 0.6880 nats. Calibrating against the
//! TEST entropy (0.1333/0.6880 = 0.1937) and then dividing the TRAIN one
//! yields 0.1213 — a silently 9% tighter anchor. The correct calibration keeps
//! both sides on train: 0.1333 / 0.6264 = 0.2128.
//!
//! CPU + rayon, not Metal, and deliberately so: this is a once-per-flow O(n)
//! integer reduction into <=K bins over labels that already live host-side in
//! `IDSCache`. At 46M rows (368 MB of i64) the fold is memory-bandwidth-bound
//! and finishes in tens of milliseconds; a Metal dispatch would have to upload
//! those labels first, so the transfer alone would cost more than the compute.

use rayon::prelude::*;

/// Chunk size for the parallel histogram fold. Large enough that the
/// per-chunk `vec![0; K]` allocation is amortised, small enough to keep every
/// core fed on a short label vector.
const FOLD_CHUNK: usize = 65_536;

/// Empirical class histogram over `labels`, folded in parallel.
///
/// Returns `Err` naming the offending value if any label falls outside
/// `[0, num_classes)`. Silently skipping such a label would change the
/// distribution the anchor is derived from, and it almost always means the
/// caller mismatched the classification mode — the exact bug class this
/// module exists to prevent.
fn class_counts(labels: &[i64], num_classes: usize) -> Result<Vec<u64>, String>
{
	labels
		.par_chunks(FOLD_CHUNK)
		.try_fold(
			|| vec![0u64; num_classes],
			|mut acc, chunk| {
				for &y in chunk
				{
					let idx = usize::try_from(y)
						.map_err(|_| format!("label {y} is negative; expected 0..{num_classes}"))?;
					if idx >= num_classes
					{
						return Err(format!("label {y} >= num_classes {num_classes}"));
					}
					acc[idx] += 1;
				}
				Ok(acc)
			},
		)
		.try_reduce(
			|| vec![0u64; num_classes],
			|mut a, b| {
				for (x, y) in a.iter_mut().zip(b)
				{
					*x += y;
				}
				Ok(a)
			},
		)
}

/// H(p) in nats over the empirical label distribution of `labels`.
///
/// Empty classes contribute nothing (0*log0 = 0). Both an empty label set and
/// a single-class label set RAISE rather than returning 0.0: a zero entropy
/// makes the derived anchor 0.0, and `2^(-ce/0)` is not a fitness function.
/// A degenerate train partition is a configuration error, not a genome result.
pub fn base_rate_entropy(labels: &[i64], num_classes: usize) -> Result<f64, String>
{
	if num_classes < 2
	{
		return Err(format!("num_classes must be >= 2, got {num_classes}"));
	}
	let counts = class_counts(labels, num_classes)?;
	let total: u64 = counts.iter().sum();
	if total == 0
	{
		return Err("base_rate_entropy: empty label set".to_string());
	}
	let total = total as f64;
	let h: f64 = -counts
		.iter()
		.filter(|&&c| c > 0)
		.map(|&c| {
			let p = c as f64 / total;
			p * p.ln()
		})
		.sum::<f64>();
	if h <= 0.0
	{
		return Err(format!(
			"base_rate_entropy: H(p) = {h} — the train partition holds a single \
			 class, so there is no base rate to normalise against"
		));
	}
	Ok(h)
}

/// The portable CE calibration: `ce / H(p)` at the concern point.
///
/// 0.1333 (the median best_ce over 18,355 unsw-nb15 binary IDSZ iterations)
/// divided by that task's TRAIN base-rate entropy 0.6264. It is a constant of
/// the fitness definition, NOT a per-run setting: every task derives its own
/// absolute anchor from its own labels, so there is nothing for a caller to
/// choose and nothing to get wrong per cohort.
pub const NORMALIZED_CE_ANCHOR: f64 = 0.2128;

/// The absolute desirability CE half-anchor for this task: `normalized * H(p)`.
///
/// Callers should prefer the no-argument form on `IDSCache`; this variant takes
/// `normalized` only so the calibration itself can be tested.
pub fn desirability_ce_anchor(
	labels: &[i64],
	num_classes: usize,
	normalized: f64,
) -> Result<f64, String>
{
	if !(normalized > 0.0) || !normalized.is_finite()
	{
		return Err(format!("normalized anchor must be finite and > 0, got {normalized}"));
	}
	Ok(normalized * base_rate_entropy(labels, num_classes)?)
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// Build a label vector with the given per-class counts.
	fn labels_from(counts: &[usize]) -> Vec<i64>
	{
		counts
			.iter()
			.enumerate()
			.flat_map(|(c, &n)| std::iter::repeat(c as i64).take(n))
			.collect()
	}

	#[test]
	fn uniform_binary_is_ln2()
	{
		let h = base_rate_entropy(&labels_from(&[500, 500]), 2).unwrap();
		assert!((h - std::f64::consts::LN_2).abs() < 1e-12, "got {h}");
	}

	#[test]
	fn uniform_k_class_is_ln_k()
	{
		let h = base_rate_entropy(&labels_from(&[100; 10]), 10).unwrap();
		assert!((h - 10f64.ln()).abs() < 1e-12, "got {h}");
	}

	/// unsw-nb15 temporal_3way TRAIN: 56,000 benign of 175,341 (the complement,
	/// 119,341, is what the cascade baseline reports as s1_train_rows). This is
	/// the partition the anchor is actually derived from.
	#[test]
	fn unsw_binary_train_base_rate_matches_measured()
	{
		let h = base_rate_entropy(&labels_from(&[56_000, 175_341 - 56_000]), 2).unwrap();
		assert!((h - 0.6264).abs() < 5e-4, "expected ~0.6264 nats, got {h}");
	}

	/// The TEST partition is a different base rate entirely (18,500 of 41,166 =
	/// 44.9% benign vs train's 32.0%). Pinned so nobody re-derives the constant
	/// against it again — doing so is what produced the 9%-tight anchor.
	#[test]
	fn train_and_test_base_rates_genuinely_differ()
	{
		let h_test = base_rate_entropy(&labels_from(&[18_500, 41_166 - 18_500]), 2).unwrap();
		let h_train = base_rate_entropy(&labels_from(&[56_000, 175_341 - 56_000]), 2).unwrap();
		assert!((h_test - 0.6880).abs() < 5e-4, "expected ~0.6880 nats, got {h_test}");
		assert!(h_test - h_train > 0.05, "the two partitions must not be treated as interchangeable");
	}

	/// THE PARITY CHECK: the portable constant must reproduce the frozen
	/// unsw-nb15 binary half-anchor 0.133 FROM THE TRAIN PARTITION, which is the
	/// one the production path reads. Calibrating on test gives 0.1937 and this
	/// assertion fails by ~9%.
	#[test]
	fn normalized_constant_reproduces_frozen_binary_anchor()
	{
		let labels = labels_from(&[56_000, 175_341 - 56_000]);
		let anchor = desirability_ce_anchor(&labels, 2, 0.2128).unwrap();
		assert!((anchor - 0.133).abs() < 5e-4, "expected ~0.133, got {anchor}");
		let miscalibrated = desirability_ce_anchor(&labels, 2, 0.1937).unwrap();
		assert!(miscalibrated < 0.125, "the test-calibrated constant is ~9% tight: {miscalibrated}");
	}

	/// The same dataset's 10-class partition. Its entropy is 2.25x the binary
	/// one, which is the whole reason a single absolute anchor cannot serve
	/// both: at anchor 0.133 the multiclass median CE 1.8998 sits at 14.3
	/// half-lives, against a clamp of 20.
	#[test]
	fn unsw_multiclass_base_rate_matches_measured()
	{
		let counts = [18_500, 9_438, 5_545, 3_000, 2_061, 1_766, 351, 283, 199, 23];
		let h = base_rate_entropy(&labels_from(&counts), 10).unwrap();
		assert!((h - 1.5478).abs() < 5e-4, "expected ~1.5478 nats, got {h}");

		let anchor = desirability_ce_anchor(&labels_from(&counts), 10, 0.2128).unwrap();
		let half_lives = 1.8998 / anchor;
		assert!(
			(4.0..9.0).contains(&half_lives),
			"multiclass median CE should land well inside the clamp, got {half_lives}"
		);
	}

	#[test]
	fn single_class_is_rejected()
	{
		let err = base_rate_entropy(&labels_from(&[1_000, 0]), 2).unwrap_err();
		assert!(err.contains("single"), "got {err}");
	}

	#[test]
	fn empty_is_rejected()
	{
		assert!(base_rate_entropy(&[], 2).is_err());
	}

	#[test]
	fn out_of_range_label_is_rejected()
	{
		let err = base_rate_entropy(&[0, 1, 7], 2).unwrap_err();
		assert!(err.contains("num_classes"), "got {err}");
		assert!(base_rate_entropy(&[0, -3], 2).is_err());
	}

	/// The shipped constant must reproduce the frozen binary anchor from the
	/// TRAIN partition. If this fails, every desirability run silently changes
	/// scale relative to the banked IDSD result.
	#[test]
	fn shipped_constant_reproduces_frozen_binary_anchor()
	{
		let labels = labels_from(&[56_000, 175_341 - 56_000]);
		let anchor = desirability_ce_anchor(&labels, 2, NORMALIZED_CE_ANCHOR).unwrap();
		assert!((anchor - 0.133).abs() < 5e-4, "expected ~0.133, got {anchor}");
	}

	#[test]
	fn bad_normalized_is_rejected()
	{
		let labels = labels_from(&[10, 10]);
		assert!(desirability_ce_anchor(&labels, 2, 0.0).is_err());
		assert!(desirability_ce_anchor(&labels, 2, f64::NAN).is_err());
	}

	/// The parallel fold must agree with a plain serial histogram on a vector
	/// long enough to span many chunks.
	#[test]
	fn parallel_fold_matches_serial()
	{
		let labels: Vec<i64> = (0..500_000i64).map(|i| i % 7).collect();
		let h = base_rate_entropy(&labels, 7).unwrap();
		assert!((h - 7f64.ln()).abs() < 1e-6, "got {h}");
	}
}
