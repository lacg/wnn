//! F1/FPR metrics, threshold search, Platt/beta/empirical calibration.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

pub fn compute_f1_fpr_with_normal_class(predictions: &[u32], targets: &[i64], num_classes: usize, normal_class: usize) -> (f64, f64) {
    if num_classes == 0 || predictions.is_empty() {
        return (0.0, 0.0);
    }

    // Build confusion matrix: confusion[true_class * K + predicted_class]
    let mut confusion = vec![0u64; num_classes * num_classes];
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        let t = *target as usize;
        let p = *pred as usize;
        if t < num_classes && p < num_classes {
            confusion[t * num_classes + p] += 1;
        }
    }

    let mut f1_sum = 0.0f64;
    let mut num_active_classes = 0usize;

    for c in 0..num_classes {
        let tp = confusion[c * num_classes + c] as f64;
        let fp: f64 = (0..num_classes)
            .filter(|&t| t != c)
            .map(|t| confusion[t * num_classes + c] as f64)
            .sum();
        let fn_count: f64 = (0..num_classes)
            .filter(|&p| p != c)
            .map(|p| confusion[c * num_classes + p] as f64)
            .sum();
        // Skip classes with no support (no true examples)
        if tp + fn_count == 0.0 {
            continue;
        }

        num_active_classes += 1;

        // F1
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        f1_sum += f1;

    }

    if num_active_classes == 0 {
        (0.0, 0.0)
    } else {
        let f1_macro = f1_sum / num_active_classes as f64;

        // IDS FPR: fraction of Normal samples misclassified as any attack class.
        // normal_class is 0 by default, but 1 when flip_labels is active, so
        // FPR always measures false alarms on the original benign traffic.
        //   FPR = (Normal predicted as non-Normal) / (all Normal samples)
        let normal_total: f64 = (0..num_classes)
            .map(|p| confusion[normal_class * num_classes + p] as f64)
            .sum();
        let normal_correct = confusion[normal_class * num_classes + normal_class] as f64;
        let fpr = if normal_total > 0.0 {
            (normal_total - normal_correct) / normal_total
        } else {
            0.0
        };

        (f1_macro, fpr)
    }
}

/// Find the optimal threshold for binary (single-cluster) classification.
///
/// Efficiently sweeps all unique score boundaries in O(n log n) time.
/// Returns (threshold, f1_macro, fpr) where f1_macro is maximized.
/// The threshold is set to the midpoint between adjacent sorted scores.
pub fn find_optimal_threshold_f1(scores: &[f64], labels: &[i64]) -> (f64, f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.5, 0.0, 0.0);
    }

    // Create (score, label) pairs and sort by score ascending
    let mut pairs: Vec<(f64, i64)> = scores.iter().copied()
        .zip(labels.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = labels.iter().filter(|&&l| l == 1).count() as u64;
    let total_neg = (n as u64) - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return (0.5, 0.0, 0.0);
    }

    // Start: threshold = -inf → everything predicted as 1 (attack)
    let mut tp = total_pos;
    let mut fp = total_neg;
    let mut fn_count = 0u64;
    let mut tn = 0u64;

    let mut best_f1 = 0.0f64;
    let mut best_threshold = 0.5f64;
    let mut best_fpr = 1.0f64;

    // Sweep threshold from low to high
    // Moving examples below threshold from "predicted 1" to "predicted 0"
    let mut i = 0;
    while i < n {
        let current_score = pairs[i].0;

        // Process all examples with the same score
        while i < n && pairs[i].0 == current_score {
            if pairs[i].1 == 1 {
                tp -= 1;
                fn_count += 1;
            } else {
                fp -= 1;
                tn += 1;
            }
            i += 1;
        }

        // Compute F1-macro at this threshold boundary
        // F1 for attack class (label=1)
        let f1_pos = if 2 * tp + fp + fn_count > 0 {
            2.0 * tp as f64 / (2.0 * tp as f64 + fp as f64 + fn_count as f64)
        } else {
            0.0
        };
        // F1 for normal class (label=0)
        let f1_neg = if 2 * tn + fn_count + fp > 0 {
            2.0 * tn as f64 / (2.0 * tn as f64 + fn_count as f64 + fp as f64)
        } else {
            0.0
        };
        let f1_macro = (f1_pos + f1_neg) / 2.0;

        if f1_macro > best_f1 {
            best_f1 = f1_macro;
            // Threshold: midpoint between current score and next, or slightly above if last
            best_threshold = if i < n {
                (current_score + pairs[i].0) / 2.0
            } else {
                current_score + 1e-6
            };
            // FPR = normal samples predicted as attack / total normal
            best_fpr = fp as f64 / total_neg as f64;
        }
    }

    (best_threshold, best_f1, best_fpr)
}

/// Find the optimal threshold maximizing weighted fitness instead of F1.
///
/// Same O(n log n) sweep as find_optimal_threshold_f1, but at each boundary
/// computes fitness = w_f1*F1 + w_fpr*(1-FPR) + w_acc*Acc + w_ce*(1-CE_approx)
/// and picks the threshold with the highest fitness.
///
/// Returns (threshold, f1_macro, fpr, accuracy, fitness).
pub fn find_optimal_threshold_fitness(
    scores: &[f64],
    labels: &[i64],
    w_ce: f32,
    w_f1: f32,
    w_fpr: f32,
    w_acc: f32,
) -> (f64, f64, f64, f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.5, 0.0, 0.0, 0.0, 0.0);
    }

    let mut pairs: Vec<(f64, i64)> = scores.iter().copied()
        .zip(labels.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = labels.iter().filter(|&&l| l == 1).count() as u64;
    let total_neg = (n as u64) - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return (0.5, 0.0, 0.0, 0.0, 0.0);
    }

    // Precompute CE for the full dataset (doesn't change with threshold)
    let ce: f64 = scores.iter().zip(labels.iter()).map(|(&s, &l)| {
        let p = s.max(1e-10).min(1.0 - 1e-10);
        -(l as f64 * p.ln() + (1.0 - l as f64) * (1.0 - p).ln())
    }).sum::<f64>() / n as f64;
    let ce_score = (1.0 - ce).max(0.0);

    // Start: threshold = -inf → everything predicted as 1 (attack)
    let mut tp = total_pos;
    let mut fp = total_neg;
    let mut fn_count = 0u64;
    let mut tn = 0u64;

    let mut best_fitness = -1.0f64;
    let mut best_threshold = 0.5f64;
    let mut best_f1 = 0.0f64;
    let mut best_fpr_val = 1.0f64;
    let mut best_acc = 0.0f64;

    let n_f64 = n as f64;

    let mut i = 0;
    while i < n {
        let current_score = pairs[i].0;

        while i < n && pairs[i].0 == current_score {
            if pairs[i].1 == 1 {
                tp -= 1;
                fn_count += 1;
            } else {
                fp -= 1;
                tn += 1;
            }
            i += 1;
        }

        // F1-macro
        let f1_pos = if 2 * tp + fp + fn_count > 0 {
            2.0 * tp as f64 / (2.0 * tp as f64 + fp as f64 + fn_count as f64)
        } else { 0.0 };
        let f1_neg = if 2 * tn + fn_count + fp > 0 {
            2.0 * tn as f64 / (2.0 * tn as f64 + fn_count as f64 + fp as f64)
        } else { 0.0 };
        let f1_macro = (f1_pos + f1_neg) / 2.0;

        let fpr = fp as f64 / total_neg as f64;
        let acc = (tp + tn) as f64 / n_f64;

        let fitness = w_f1 as f64 * f1_macro
            + w_fpr as f64 * (1.0 - fpr)
            + w_acc as f64 * acc
            + w_ce as f64 * ce_score;

        if fitness > best_fitness {
            best_fitness = fitness;
            best_threshold = if i < n {
                (current_score + pairs[i].0) / 2.0
            } else {
                current_score + 1e-6
            };
            best_f1 = f1_macro;
            best_fpr_val = fpr;
            best_acc = acc;
        }
    }

    (best_threshold, best_f1, best_fpr_val, best_acc, best_fitness)
}

/// Fit Platt scaling on scores+labels, return threshold.
/// Learns sigmoid P(attack) = 1/(1+exp(-(a*score+b))) via Newton's method.
/// Returns (threshold, a, b) where threshold = -b/a.
pub fn fit_platt_scaling(scores: &[f64], labels: &[i64]) -> (f64, f64, f64) {
    let n = scores.len();
    if n == 0 { return (0.5, 1.0, 0.0); }

    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return (0.5, 1.0, 0.0); }

    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    let mut a = 1.0f64;
    let mut b = 0.0f64;

    for _ in 0..100 {
        let mut g_a = 0.0f64;
        let mut g_b = 0.0f64;
        let mut h_aa = 0.0f64;
        let mut h_ab = 0.0f64;
        let mut h_bb = 0.0f64;

        for i in 0..n {
            let fval = a * scores[i] + b;
            let p = if fval >= 0.0 {
                1.0 / (1.0 + (-fval).exp())
            } else {
                let ef = fval.exp();
                ef / (1.0 + ef)
            };
            let p = p.max(1e-15).min(1.0 - 1e-15);

            let t = if labels[i] == 1 { t_pos } else { t_neg };
            let d = p - t;
            g_a += d * scores[i];
            g_b += d;
            let w = p * (1.0 - p);
            h_aa += w * scores[i] * scores[i];
            h_ab += w * scores[i];
            h_bb += w;
        }

        h_aa += 1e-6;
        h_bb += 1e-6;
        let det = h_aa * h_bb - h_ab * h_ab;
        if det.abs() < 1e-12 { break; }
        let da = -(h_bb * g_a - h_ab * g_b) / det;
        let db = -(h_aa * g_b - h_ab * g_a) / det;
        a += da;
        b += db;
        if da.abs() < 1e-8 && db.abs() < 1e-8 { break; }
    }

    let threshold = if a.abs() > 1e-10 { -b / a } else { 0.5 };
    (threshold, a, b)
}

/// Fit Beta calibration on scores+labels, return threshold.
/// Learns logit(P) = a*ln(s) + b*(-ln(1-s)) + c via gradient descent.
/// Returns (threshold, a, b, c).
pub fn fit_beta_calibration(scores: &[f64], labels: &[i64]) -> (f64, f64, f64, f64) {
    let n = scores.len();
    if n == 0 { return (0.5, 1.0, 1.0, 0.0); }

    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return (0.5, 1.0, 1.0, 0.0); }

    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    let eps = 1e-10f64;
    let x1: Vec<f64> = scores.iter().map(|&s| s.max(eps).ln()).collect();
    let x2: Vec<f64> = scores.iter().map(|&s| -(1.0 - s).max(eps).ln()).collect();

    let mut a = 1.0f64;
    let mut b = 1.0f64;
    let mut c = 0.0f64;

    for _ in 0..100 {
        let mut g_a = 0.0f64;
        let mut g_b = 0.0f64;
        let mut g_c = 0.0f64;
        let mut h_aa = 0.0f64;
        let mut h_bb = 0.0f64;
        let mut h_cc = 0.0f64;

        for i in 0..n {
            let fval = a * x1[i] + b * x2[i] + c;
            let p = if fval >= 0.0 {
                1.0 / (1.0 + (-fval).exp())
            } else {
                let ef = fval.exp();
                ef / (1.0 + ef)
            };
            let p = p.max(1e-15).min(1.0 - 1e-15);

            let t = if labels[i] == 1 { t_pos } else { t_neg };
            let d = p - t;
            g_a += d * x1[i];
            g_b += d * x2[i];
            g_c += d;
            let w = p * (1.0 - p);
            h_aa += w * x1[i] * x1[i];
            h_bb += w * x2[i] * x2[i];
            h_cc += w;
        }

        h_aa += 1e-6;
        h_bb += 1e-6;
        h_cc += 1e-6;
        a -= 0.1 * g_a / h_aa;
        b -= 0.1 * g_b / h_bb;
        c -= 0.1 * g_c / h_cc;
        if (g_a / h_aa.max(1e-10)).abs() < 1e-6 && (g_b / h_bb.max(1e-10)).abs() < 1e-6 {
            break;
        }
    }

    // Binary search for threshold where a*ln(s) + b*(-ln(1-s)) + c = 0
    let mut lo = 0.001f64;
    let mut hi = 0.999f64;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let val = a * mid.ln() + b * (-(1.0 - mid).ln()) + c;
        if val < 0.0 { lo = mid; } else { hi = mid; }
    }
    let threshold = (lo + hi) / 2.0;

    (threshold, a, b, c)
}

/// Fit empirical threshold: find lowest score where P(attack|score) >= 0.5,
/// **constrained to bins with at least `MIN_BIN_SIZE` examples**.
///
/// HISTORY (29/05/2026 fix): the original rule was "first bin with rate ≥ 50%
/// regardless of bin size". That picked from noisy small bins around the
/// transition zone, making the chosen threshold sensitive to small score-
/// distribution shifts (e.g., training-code drift across cohort revisions).
/// The r106 paper-replay diagnostic (`scripts/diagnose_empirical_brittleness.py`)
/// showed a noisy bin region 38-63% attack-rate where the FIRST 50%-crossing
/// bin had only ~200 examples; downstream, that meant 90.52→79.81 F1 swings
/// between training-code revisions despite the trained model being equivalent.
///
/// With `MIN_BIN_SIZE = 200`, the algorithm skips small noise-prone bins and
/// only considers transitions backed by enough samples to be statistically
/// real. On the r106 replay this changes the picked threshold from 0.0975
/// (noisy) to ~0.55 (stable) and recovers F1 ≈ 88-90 (vs paper 90.52).
///
/// IMPORTANT: existing cohort numbers stored in the DB were computed with
/// MIN_BIN_SIZE=1 (the old default). Post-fix cohort numbers will use 200.
/// Cross-revision comparisons of empirical-mode F1/FPR are NOT directly
/// valid until the older cohort is re-validated under the new rule. See
/// `[[project_empirical_brittleness_fix]]` for the cohort-comparability
/// audit trail.
///
/// Returns (threshold, n_bins).
pub fn fit_empirical_threshold(scores: &[f64], labels: &[i64]) -> (f64, usize) {
    use std::collections::BTreeMap;
    /// Minimum bin size to consider for the "first crossing" rule. 200 was
    /// chosen empirically on the UNSW-temp r106 replay; smaller values let
    /// noise through, larger values miss real transitions in small datasets.
    const MIN_BIN_SIZE: u64 = 200;

    let mut bins: BTreeMap<i64, (u64, u64)> = BTreeMap::new();   // score_key → (normal, attack)
    for (&s, &l) in scores.iter().zip(labels.iter()) {
        let key = (s * 1_000_000.0) as i64;
        let entry = bins.entry(key).or_insert((0, 0));
        if l == 1 { entry.1 += 1; } else { entry.0 += 1; }
    }
    let n_bins = bins.len();

    // Pass 1 (preferred): find the first bin with ≥ MIN_BIN_SIZE examples
    // AND attack-rate ≥ 50%. The size guard skips noise.
    let mut threshold = None;
    for (&key, &(normal, attack)) in &bins {
        let total = normal + attack;
        if total >= MIN_BIN_SIZE && (attack as f64 / total as f64) >= 0.5 {
            threshold = Some(key as f64 / 1_000_000.0);
            break;
        }
    }

    // Pass 2 (fallback for tiny datasets / extreme imbalance): if no bin
    // reaches MIN_BIN_SIZE, fall back to the original "any bin" rule so we
    // never return the 0.5 default silently when a real signal exists.
    if threshold.is_none() {
        for (&key, &(normal, attack)) in &bins {
            let total = normal + attack;
            if total > 0 && (attack as f64 / total as f64) >= 0.5 {
                threshold = Some(key as f64 / 1_000_000.0);
                break;
            }
        }
    }

    (threshold.unwrap_or(0.5), n_bins)
}
