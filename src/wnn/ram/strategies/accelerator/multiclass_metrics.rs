//! K-class metrics for multiclass IDS evaluation (docs/MULTICLASS_DESIGN.md §3-4).
//!
//! Decode rules:
//!   1. `argmax` over the K per-cluster scores (baseline decode; what the GA
//!      search fitness uses).
//!   2. `benign_margin`: m = max_{c≠benign} score_c − score_benign; predict
//!      attack (argmax over attack clusters only) iff m ≥ τ, else benign.
//!      τ is calibrated by sweeping margins for the macro-F1 optimum
//!      (`find_optimal_margin_tau`, mirroring `find_optimal_threshold_f1`).
//!
//! Metrics: K-class CE, K×K confusion matrix (row = true class), per-class
//! precision/recall/F1, macro-F1 (headline), weighted-F1 (secondary),
//! accuracy, and benign-FPR = fraction of true-benign examples predicted as
//! ANY attack class (= 1 − benign recall).
//!
//! CE normalization (softmax-free, consistent with the binary path): the
//! per-cluster scores are mean QUAD cell outputs in [0, 1] — probability-like
//! values that binary BCE consumes DIRECTLY as P(attack)
//! (`compute_binary_metrics_at_threshold`). The K-class generalization keeps
//! that linear scale: p_c = (s_c + ε) / Σ_j (s_j + ε), CE = −ln p_target.
//! Exponentiating (softmax) would distort the calibrated [0,1] scale the
//! binary metrics rely on. NOTE: the GA-search CE for K-cluster genomes
//! (`evaluate_genome_hybrid` / GPU compute_ce) is softmax-based and is left
//! unchanged — search CE and this validation CE are both valid ranking
//! signals but are not numerically comparable.
//!
//! All score inputs are flattened row-major: `scores_flat[ex * K + c]`.

const EPS: f64 = 1e-10;

/// Full multiclass metric set for one decode rule's predictions.
#[derive(Clone, Debug)]
pub struct MulticlassMetrics {
    /// K-class cross-entropy (decode-independent; sum-normalized scores).
    pub ce: f64,
    pub accuracy: f64,
    /// Macro-F1 over classes WITH support (matches
    /// `compute_f1_fpr_with_normal_class`'s skip-no-support convention).
    pub macro_f1: f64,
    /// Support-weighted F1 (literature-comparability secondary metric).
    pub weighted_f1: f64,
    /// Fraction of true-benign examples predicted as ANY attack class.
    pub benign_fpr: f64,
    /// K×K confusion matrix, row-major, row = true class, col = predicted.
    pub confusion: Vec<u64>,
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub f1: Vec<f64>,
    /// True-example count per class (confusion row sums).
    pub support: Vec<u64>,
}

/// One decode mode's result: mode name + resolved τ (NaN for argmax) + metrics.
#[derive(Clone, Debug)]
pub struct MulticlassModeResult {
    pub mode: String,
    pub tau: f64,
    pub metrics: MulticlassMetrics,
}

/// K-class cross-entropy on sum-normalized scores (see module docs for why
/// this is softmax-free): p_c = (s_c + ε) / Σ_j (s_j + ε), CE = −ln p_target.
pub fn multiclass_ce(scores_flat: &[f64], targets: &[i64], num_classes: usize) -> f64 {
    let n = targets.len();
    if n == 0 || num_classes == 0 {
        return 0.0;
    }
    debug_assert_eq!(scores_flat.len(), n * num_classes);

    let mut total = 0.0f64;
    for ex in 0..n {
        let row = &scores_flat[ex * num_classes..(ex + 1) * num_classes];
        let sum: f64 = row.iter().map(|&s| s + EPS).sum();
        let t = targets[ex] as usize;
        let p_target = if t < num_classes {
            (row[t] + EPS) / sum
        } else {
            EPS / sum // out-of-range label: maximally wrong, don't panic
        };
        total += -p_target.max(EPS).ln();
    }
    total / n as f64
}

/// Argmax decode: predicted class = argmax_c score_c. Tie-breaking matches
/// `predict_genome_hybrid` (last of equal maxima via max_by semantics).
pub fn argmax_decode(scores_flat: &[f64], num_classes: usize) -> Vec<u32> {
    let n = scores_flat.len() / num_classes.max(1);
    (0..n).map(|ex| {
        let row = &scores_flat[ex * num_classes..(ex + 1) * num_classes];
        row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }).collect()
}

/// Per-example benign margin + attack argmax.
///
/// Returns (margins, attack_argmax) where margin[ex] = max_{c≠benign}
/// score_c − score_benign and attack_argmax[ex] is the best attack class
/// (tie-breaking mirrors `argmax_decode`: last of equal maxima).
pub fn benign_margins(
    scores_flat: &[f64],
    num_classes: usize,
    benign_class: usize,
) -> (Vec<f64>, Vec<u32>) {
    let n = scores_flat.len() / num_classes.max(1);
    let mut margins = Vec::with_capacity(n);
    let mut attack_argmax = Vec::with_capacity(n);
    for ex in 0..n {
        let row = &scores_flat[ex * num_classes..(ex + 1) * num_classes];
        let mut best_c = if benign_class == 0 { 1 } else { 0 };
        let mut best_s = f64::NEG_INFINITY;
        for c in 0..num_classes {
            if c == benign_class {
                continue;
            }
            if row[c] >= best_s {
                best_s = row[c];
                best_c = c;
            }
        }
        margins.push(best_s - row[benign_class]);
        attack_argmax.push(best_c as u32);
    }
    (margins, attack_argmax)
}

/// Benign-margin decode at threshold τ: attack_argmax iff margin ≥ τ, else benign.
pub fn margin_decode(
    margins: &[f64],
    attack_argmax: &[u32],
    tau: f64,
    benign_class: usize,
) -> Vec<u32> {
    margins.iter().zip(attack_argmax.iter())
        .map(|(&m, &a)| if m >= tau { a } else { benign_class as u32 })
        .collect()
}

/// Build the K×K confusion matrix (row-major, row = true class).
/// Out-of-range labels/predictions are skipped (same guard as
/// `compute_f1_fpr_with_normal_class`).
pub fn confusion_matrix(predictions: &[u32], targets: &[i64], num_classes: usize) -> Vec<u64> {
    let mut confusion = vec![0u64; num_classes * num_classes];
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        let t = *target as usize;
        let p = *pred as usize;
        if t < num_classes && p < num_classes {
            confusion[t * num_classes + p] += 1;
        }
    }
    confusion
}

/// Compute the full multiclass metric set for a fixed set of predictions.
///
/// `scores_flat` is only consumed for the (decode-independent) K-class CE;
/// everything else derives from the confusion matrix.
pub fn metrics_from_predictions(
    scores_flat: &[f64],
    predictions: &[u32],
    targets: &[i64],
    num_classes: usize,
    benign_class: usize,
) -> MulticlassMetrics {
    let n = targets.len();
    let confusion = confusion_matrix(predictions, targets, num_classes);

    let mut precision = vec![0.0f64; num_classes];
    let mut recall = vec![0.0f64; num_classes];
    let mut f1 = vec![0.0f64; num_classes];
    let mut support = vec![0u64; num_classes];

    let mut macro_sum = 0.0f64;
    let mut active = 0usize;
    let mut weighted_sum = 0.0f64;
    let mut total_support = 0u64;
    let mut correct = 0u64;

    for c in 0..num_classes {
        let tp = confusion[c * num_classes + c];
        correct += tp;
        let row: u64 = (0..num_classes).map(|p| confusion[c * num_classes + p]).sum();
        let col: u64 = (0..num_classes).map(|t| confusion[t * num_classes + c]).sum();
        support[c] = row;
        let p = if col > 0 { tp as f64 / col as f64 } else { 0.0 };
        let r = if row > 0 { tp as f64 / row as f64 } else { 0.0 };
        let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        precision[c] = p;
        recall[c] = r;
        f1[c] = f;
        // Skip classes with no support (no true examples) — same convention
        // as compute_f1_fpr_with_normal_class.
        if row > 0 {
            active += 1;
            macro_sum += f;
            weighted_sum += row as f64 * f;
            total_support += row;
        }
    }

    let macro_f1 = if active > 0 { macro_sum / active as f64 } else { 0.0 };
    let weighted_f1 = if total_support > 0 { weighted_sum / total_support as f64 } else { 0.0 };
    let accuracy = if n > 0 { correct as f64 / n as f64 } else { 0.0 };

    // Benign-FPR: fraction of true-benign examples predicted as ANY attack class.
    let benign_total: u64 = (0..num_classes)
        .map(|p| confusion[benign_class * num_classes + p])
        .sum();
    let benign_correct = confusion[benign_class * num_classes + benign_class];
    let benign_fpr = if benign_total > 0 {
        (benign_total - benign_correct) as f64 / benign_total as f64
    } else {
        0.0
    };

    let ce = multiclass_ce(scores_flat, targets, num_classes);

    MulticlassMetrics {
        ce,
        accuracy,
        macro_f1,
        weighted_f1,
        benign_fpr,
        confusion,
        precision,
        recall,
        f1,
        support,
    }
}

/// Sweep the benign-margin threshold τ for the macro-F1 optimum.
///
/// Mirrors `find_optimal_threshold_f1`'s O(n log n) sweep style: sort by
/// margin ascending, start at τ = −inf (every example predicted as its
/// attack-argmax class), then move examples to the benign prediction one
/// margin-boundary at a time, maintaining per-class tp/fp/fn incrementally.
/// The candidate τ at each boundary is the midpoint to the next margin
/// (or last + 1e-6). Returns (tau, macro_f1_at_tau).
pub fn find_optimal_margin_tau(
    margins: &[f64],
    attack_argmax: &[u32],
    targets: &[i64],
    num_classes: usize,
    benign_class: usize,
) -> (f64, f64) {
    let n = margins.len();
    if n == 0 || num_classes == 0 {
        return (0.0, 0.0);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        margins[a].partial_cmp(&margins[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Start: τ = −inf → every example predicted as its attack-argmax class.
    let mut tp = vec![0u64; num_classes];
    let mut fp = vec![0u64; num_classes];
    let mut fn_ = vec![0u64; num_classes];
    for ex in 0..n {
        let t = targets[ex] as usize;
        let p = attack_argmax[ex] as usize;
        if t < num_classes && t == p {
            tp[t] += 1;
        } else {
            if p < num_classes {
                fp[p] += 1;
            }
            if t < num_classes {
                fn_[t] += 1;
            }
        }
    }

    // Macro-F1 from the running counters; tp[c] + fn_[c] (= support) is
    // invariant under prediction flips, so the skip-no-support rule is stable.
    let macro_f1_of = |tp: &[u64], fp: &[u64], fn_: &[u64]| -> f64 {
        let mut sum = 0.0f64;
        let mut active = 0usize;
        for c in 0..num_classes {
            if tp[c] + fn_[c] == 0 {
                continue;
            }
            active += 1;
            let denom = 2 * tp[c] + fp[c] + fn_[c];
            if denom > 0 {
                sum += 2.0 * tp[c] as f64 / denom as f64;
            }
        }
        if active == 0 { 0.0 } else { sum / active as f64 }
    };

    let mut best_f1 = -1.0f64;
    let mut best_tau = 0.0f64;

    let mut i = 0usize;
    while i < n {
        let current_margin = margins[order[i]];

        // Flip all examples at this margin from attack-argmax to benign.
        while i < n && margins[order[i]] == current_margin {
            let ex = order[i];
            let t = targets[ex] as usize;
            let p_old = attack_argmax[ex] as usize;
            // Remove the old (attack) contribution...
            if t < num_classes && t == p_old {
                tp[t] -= 1;
            } else {
                if p_old < num_classes {
                    fp[p_old] -= 1;
                }
                if t < num_classes {
                    fn_[t] -= 1;
                }
            }
            // ...and add the new (benign) one.
            if t == benign_class {
                tp[benign_class] += 1;
            } else {
                fp[benign_class] += 1;
                if t < num_classes {
                    fn_[t] += 1;
                }
            }
            i += 1;
        }

        let f1 = macro_f1_of(&tp, &fp, &fn_);
        if f1 > best_f1 {
            best_f1 = f1;
            best_tau = if i < n {
                (current_margin + margins[order[i]]) / 2.0
            } else {
                current_margin + 1e-6
            };
        }
    }

    (best_tau, best_f1.max(0.0))
}

/// Compute the full per-mode metric set from pre-computed flat K-vector
/// scores (docs/MULTICLASS_DESIGN.md §3). Shared by the in-memory cache path
/// (`evaluate_multiclass_at_thresholds_ids_cached`) and the streaming path
/// (scores accumulated chunk-wise by `IDSGenomeStreamer`, drained via
/// `take_scores`). Metrics are ALWAYS computed on the EVAL set; train/val
/// margins only calibrate τ. `margin_val_cal` is emitted only when both
/// `val_scores` and `val_targets` are present (Protocol v2, 3-way splits).
pub fn modes_from_scores(
    eval_scores: &[f64],
    eval_targets: &[i64],
    train_scores: &[f64],
    train_targets: &[i64],
    val_scores: Option<&[f64]>,
    val_targets: Option<&[i64]>,
    num_classes: usize,
    benign_class: usize,
) -> Vec<MulticlassModeResult> {
    let mut modes = Vec::with_capacity(4);

    // 1. argmax — the baseline decode (same rule as the GA-search fitness)
    let argmax_preds = argmax_decode(eval_scores, num_classes);
    modes.push(MulticlassModeResult {
        mode: "argmax".to_string(),
        tau: f64::NAN,
        metrics: metrics_from_predictions(
            eval_scores, &argmax_preds, eval_targets, num_classes, benign_class,
        ),
    });

    // Benign-margin decode: margins per set from the SAME trained memory;
    // metrics ALWAYS on the eval set.
    let (eval_margins, eval_attack) = benign_margins(eval_scores, num_classes, benign_class);
    let margin_mode = |mode: &str, tau: f64| -> MulticlassModeResult {
        let preds = margin_decode(&eval_margins, &eval_attack, tau, benign_class);
        MulticlassModeResult {
            mode: mode.to_string(),
            tau,
            metrics: metrics_from_predictions(
                eval_scores, &preds, eval_targets, num_classes, benign_class,
            ),
        }
    };

    // 2. fixed τ = 0.0 (attack wins any positive margin)
    modes.push(margin_mode("margin_fixed0", 0.0));

    // 3. train-calibrated τ (macro-F1-optimal sweep on train margins)
    let (train_margins, train_attack) = benign_margins(train_scores, num_classes, benign_class);
    let (train_tau, _train_f1) = find_optimal_margin_tau(
        &train_margins, &train_attack, train_targets, num_classes, benign_class,
    );
    modes.push(margin_mode("margin_train_cal", train_tau));

    // 4. val-calibrated τ (Protocol v2 — only when a val partition exists)
    if let (Some(vs), Some(vt)) = (val_scores, val_targets) {
        let (val_margins, val_attack) = benign_margins(vs, num_classes, benign_class);
        let (val_tau, _val_f1) = find_optimal_margin_tau(
            &val_margins, &val_attack, vt, num_classes, benign_class,
        );
        modes.push(margin_mode("margin_val_cal", val_tau));
    }

    // 5-6. Per-class calibrated argmax (the design doc's deferred v2 item,
    // promoted 12/07/2026 on Luiz order). Fit a one-vs-rest calibration map
    // g_c per class on the CALIBRATION partition (val under Protocol v2,
    // train otherwise — same convention as the binary platt/beta modes),
    // then predict argmax_c g_c(s_c) on eval. UNLIKE any single monotone map
    // of the benign margin (order-preserving ⇒ decode-identical to a τ
    // sweep), K DIFFERENT per-class maps re-weight classes against each
    // other, so this genuinely differs from raw argmax — it can recover a
    // class whose scores are informative but compressed/offset.
    let (cal_scores, cal_targets) = match (val_scores, val_targets) {
        (Some(vs), Some(vt)) => (vs, vt),
        _ => (train_scores, train_targets),
    };
    for (mode_name, kind) in [("argmax_platt", CalKind::Platt), ("argmax_beta", CalKind::Beta)] {
        let params = fit_per_class_calibration(cal_scores, cal_targets, num_classes, kind);
        let preds = calibrated_argmax_decode(eval_scores, num_classes, &params);
        modes.push(MulticlassModeResult {
            mode: mode_name.to_string(),
            tau: f64::NAN,
            metrics: metrics_from_predictions(
                eval_scores, &preds, eval_targets, num_classes, benign_class,
            ),
        });
    }

    modes
}

// ---------------------------------------------------------------------------
// Per-class calibrated argmax (argmax_platt / argmax_beta)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum CalKind {
    Platt,
    Beta,
}

/// One class's fitted calibration map g_c. Platt: p = σ(a·s + b).
/// Beta: p = σ(a·ln s̃ + b·(−ln(1−s̃)) + c), s̃ clamped to [ε, 1−ε]
/// (mirrors fit_beta_calibration's internal feature transform exactly).
#[derive(Clone, Copy)]
struct ClassCal {
    kind_beta: bool,
    a: f64,
    b: f64,
    c: f64,
}

impl ClassCal {
    #[inline]
    fn apply(&self, s: f64) -> f64 {
        let fval = if self.kind_beta {
            let eps = 1e-10;
            let sc = s.clamp(eps, 1.0 - eps);
            self.a * sc.ln() + self.b * (-(1.0 - sc).ln()) + self.c
        } else {
            self.a * s + self.b
        };
        if fval >= 0.0 {
            1.0 / (1.0 + (-fval).exp())
        } else {
            let ef = fval.exp();
            ef / (1.0 + ef)
        }
    }
}

/// Fit one-vs-rest calibration per class on the calibration partition. A class
/// with no positives (or no negatives) on the partition gets the fit fns'
/// identity-ish fallback (a=1, b=0[, c=0]) — its scores pass through a plain
/// sigmoid, preserving their ordering.
fn fit_per_class_calibration(
    cal_scores: &[f64],
    cal_targets: &[i64],
    num_classes: usize,
    kind: CalKind,
) -> Vec<ClassCal> {
    let n = cal_targets.len();
    (0..num_classes)
        .map(|c| {
            let col: Vec<f64> = (0..n).map(|ex| cal_scores[ex * num_classes + c]).collect();
            let ovr: Vec<i64> = cal_targets.iter().map(|&t| (t as usize == c) as i64).collect();
            match kind {
                CalKind::Platt => {
                    let (_thr, a, b) = crate::adaptive::fit_platt_scaling(&col, &ovr);
                    ClassCal { kind_beta: false, a, b, c: 0.0 }
                }
                CalKind::Beta => {
                    let (_thr, a, b, c) = crate::adaptive::fit_beta_calibration(&col, &ovr);
                    ClassCal { kind_beta: true, a, b, c }
                }
            }
        })
        .collect()
}

/// argmax over the per-class CALIBRATED probabilities. Tie-break: HIGHEST
/// class index wins (>= replacement) — the same rule as argmax_decode, whose
/// max_by returns the last of equal maxima.
fn calibrated_argmax_decode(
    scores_flat: &[f64],
    num_classes: usize,
    params: &[ClassCal],
) -> Vec<u32> {
    let n = scores_flat.len() / num_classes;
    (0..n)
        .map(|ex| {
            let mut best_c = 0u32;
            let mut best_p = f64::NEG_INFINITY;
            for c in 0..num_classes {
                let p = params[c].apply(scores_flat[ex * num_classes + c]);
                if p >= best_p {
                    best_p = p;
                    best_c = c as u32;
                }
            }
            best_c
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn assert_close(a: f64, b: f64, msg: &str) {
        assert!((a - b).abs() < 1e-6, "{}: {} != {}", msg, a, b);
    }

    /// Per-class calibrated argmax: on WELL-SEPARATED scores the calibration
    /// maps are monotone per class and cannot flip a clear winner — the
    /// calibrated decodes must match raw argmax prediction-for-prediction.
    #[test]
    fn calibrated_argmax_matches_argmax_when_separated() {
        let k = 3usize;
        let mut scores = Vec::new();
        let mut targets = Vec::new();
        for ex in 0..60 {
            let t = ex % k;
            for c in 0..k {
                scores.push(if c == t { 0.9 } else { 0.1 });
            }
            targets.push(t as i64);
        }
        let modes = modes_from_scores(&scores, &targets, &scores, &targets, None, None, k, 0);
        let names: Vec<&str> = modes.iter().map(|m| m.mode.as_str()).collect();
        assert!(names.contains(&"argmax_platt") && names.contains(&"argmax_beta"), "{names:?}");
        for m in &modes {
            if m.mode.starts_with("argmax") {
                assert!((m.metrics.macro_f1 - 1.0).abs() < 1e-9,
                    "{}: macro_f1 {} != 1.0", m.mode, m.metrics.macro_f1);
            }
        }
    }

    /// The point of per-class calibration: class 2's scores are informative
    /// but COMPRESSED (own-score 0.30 vs others' 0.9), so raw argmax never
    /// predicts it (macro-F1 tanks); per-class Platt re-scales class 2's
    /// column and recovers it. This is what a single monotone map on the
    /// benign margin can NEVER do (order-preserving ⇒ decode-identical).
    #[test]
    fn calibrated_argmax_recovers_compressed_class() {
        let k = 3usize;
        let mut scores = Vec::new();
        let mut targets = Vec::new();
        for ex in 0..300 {
            let t = ex % k;
            for c in 0..k {
                // Class-2 column is compressed: own-signal 0.30, noise 0.02.
                // Classes 0/1: own-signal 0.9, noise 0.4 (below 0.9, above
                // class-2's compressed signal → argmax never picks 2).
                let own = c == t;
                scores.push(match (c, own) {
                    (2, true) => 0.30,
                    (2, false) => 0.02,
                    (_, true) => 0.9,
                    (_, false) => 0.4,
                });
            }
            targets.push(t as i64);
        }
        let modes = modes_from_scores(&scores, &targets, &scores, &targets, None, None, k, 0);
        let get = |name: &str| modes.iter().find(|m| m.mode == name).unwrap();
        let raw = get("argmax");
        let platt = get("argmax_platt");
        let beta = get("argmax_beta");
        // Raw argmax cannot predict class 2 (0.30 < 0.4 noise on cols 0/1).
        assert!(raw.metrics.recall[2] < 1e-9, "raw argmax recall[2] = {}", raw.metrics.recall[2]);
        // Per-class calibration recovers it fully on this synthetic set.
        assert!((platt.metrics.macro_f1 - 1.0).abs() < 1e-6,
            "argmax_platt macro_f1 {} != 1.0", platt.metrics.macro_f1);
        assert!((beta.metrics.macro_f1 - 1.0).abs() < 1e-6,
            "argmax_beta macro_f1 {} != 1.0", beta.metrics.macro_f1);
        assert!(platt.metrics.macro_f1 > raw.metrics.macro_f1 + 0.2);
    }

    /// Protocol v2: with a val partition the calibration must fit on VAL —
    /// give train a MISLEADING class-2 mapping and val the correct one; the
    /// calibrated decode must follow val.
    #[test]
    fn calibrated_argmax_fits_on_val_when_present() {
        let k = 3usize;
        let build = |own2: f64, noise2: f64| {
            let mut scores = Vec::new();
            let mut targets = Vec::new();
            for ex in 0..300 {
                let t = ex % k;
                for c in 0..k {
                    let own = c == t;
                    scores.push(match (c, own) {
                        (2, true) => own2,
                        (2, false) => noise2,
                        (_, true) => 0.9,
                        (_, false) => 0.4,
                    });
                }
                targets.push(t as i64);
            }
            (scores, targets)
        };
        let (eval_s, eval_t) = build(0.30, 0.02);   // compressed (true behavior)
        // Train: class-2 column INVERTED (high noise, low own) — a platt fit
        // on it maps class-2 scores with NEGATIVE slope → decode breaks.
        let (train_s, train_t) = build(0.02, 0.30);
        let (val_s, val_t) = build(0.30, 0.02);     // val matches eval
        let modes = modes_from_scores(&eval_s, &eval_t, &train_s, &train_t,
                                      Some(&val_s), Some(&val_t), k, 0);
        let platt = modes.iter().find(|m| m.mode == "argmax_platt").unwrap();
        assert!((platt.metrics.macro_f1 - 1.0).abs() < 1e-6,
            "val-fit argmax_platt macro_f1 {} != 1.0 (fit on train instead of val?)",
            platt.metrics.macro_f1);
    }

    #[test]
    fn test_multiclass_ce_hand_computed() {
        // K=3: p_target = s_t / Σ s (EPS negligible at this scale)
        let scores = vec![
            0.5, 0.25, 0.25, // target 0 → p = 0.5
            0.0, 1.0, 0.0,   // target 1 → p = 1.0
        ];
        let targets = vec![0i64, 1];
        let ce = multiclass_ce(&scores, &targets, 3);
        let expected = (-(0.5f64).ln() + 0.0) / 2.0;
        assert_close(ce, expected, "K=3 CE");
    }

    #[test]
    fn test_multiclass_ce_uniform_scores() {
        // All-equal scores → p_target = 1/K → CE = ln K
        let scores = vec![0.4, 0.4, 0.4];
        let targets = vec![2i64];
        let ce = multiclass_ce(&scores, &targets, 3);
        assert_close(ce, (3.0f64).ln(), "uniform CE = ln K");
    }

    #[test]
    fn test_argmax_decode() {
        let scores = vec![
            0.9, 0.1, 0.2,
            0.2, 0.6, 0.5,
            0.3, 0.3, 0.4,
        ];
        assert_eq!(argmax_decode(&scores, 3), vec![0, 1, 2]);
    }

    #[test]
    fn test_confusion_and_per_class_metrics() {
        // Hand-computed K=3 fixture:
        //   targets = [0,0,0,1,1,2], preds = [0,0,1,1,2,2]
        //   confusion rows (true): [2,1,0], [0,1,1], [0,0,1]
        let targets = vec![0i64, 0, 0, 1, 1, 2];
        let preds = vec![0u32, 0, 1, 1, 2, 2];
        let scores = vec![0.0; targets.len() * 3]; // CE not under test here
        let m = metrics_from_predictions(&scores, &preds, &targets, 3, 0);

        assert_eq!(m.confusion, vec![2, 1, 0, 0, 1, 1, 0, 0, 1]);
        assert_eq!(m.support, vec![3, 2, 1]);

        // class 0: P=2/2=1.0, R=2/3, F1=0.8
        // class 1: P=1/2,     R=1/2, F1=0.5
        // class 2: P=1/2,     R=1/1, F1=2/3
        assert_close(m.precision[0], 1.0, "P0");
        assert_close(m.recall[0], 2.0 / 3.0, "R0");
        assert_close(m.f1[0], 0.8, "F1_0");
        assert_close(m.f1[1], 0.5, "F1_1");
        assert_close(m.f1[2], 2.0 / 3.0, "F1_2");

        assert_close(m.macro_f1, (0.8 + 0.5 + 2.0 / 3.0) / 3.0, "macro F1");
        assert_close(
            m.weighted_f1,
            (3.0 * 0.8 + 2.0 * 0.5 + 1.0 * 2.0 / 3.0) / 6.0,
            "weighted F1",
        );
        assert_close(m.accuracy, 4.0 / 6.0, "accuracy");
        // Benign-FPR: 1 of 3 true-benign predicted as an attack class.
        assert_close(m.benign_fpr, 1.0 / 3.0, "benign FPR");
    }

    #[test]
    fn test_macro_f1_skips_no_support_class() {
        // Class 2 has no true examples → excluded from the macro average.
        let targets = vec![0i64, 0, 1, 1];
        let preds = vec![0u32, 0, 1, 1];
        let scores = vec![0.0; targets.len() * 3];
        let m = metrics_from_predictions(&scores, &preds, &targets, 3, 0);
        assert_close(m.macro_f1, 1.0, "perfect on supported classes");
        assert_eq!(m.support[2], 0);
    }

    #[test]
    fn test_benign_margins_and_decode() {
        let scores = vec![
            0.9, 0.1, 0.2, // margin = 0.2 − 0.9 = −0.7, attack argmax = 2
            0.2, 0.6, 0.5, // margin = 0.6 − 0.2 = 0.4,  attack argmax = 1
            0.3, 0.3, 0.4, // margin = 0.4 − 0.3 = 0.1,  attack argmax = 2
        ];
        let (margins, attack) = benign_margins(&scores, 3, 0);
        assert!((margins[0] - (-0.7)).abs() < TOL);
        assert!((margins[1] - 0.4).abs() < TOL);
        assert!((margins[2] - 0.1).abs() < TOL);
        assert_eq!(attack, vec![2, 1, 2]);

        assert_eq!(margin_decode(&margins, &attack, 0.0, 0), vec![0, 1, 2]);
        assert_eq!(margin_decode(&margins, &attack, 0.2, 0), vec![0, 1, 0]);
    }

    #[test]
    fn test_find_optimal_margin_tau_hand_computed() {
        // targets = [0, 0, 1, 2]; margins/attack-argmax hand-built so the
        // sweep reaches perfect macro-F1 exactly between margins 0.1 and 0.3.
        let targets = vec![0i64, 0, 1, 2];
        let margins = vec![-0.5, 0.1, 0.3, 0.6];
        let attack = vec![1u32, 2, 1, 2];

        let (tau, f1) = find_optimal_margin_tau(&margins, &attack, &targets, 3, 0);
        assert_close(tau, 0.2, "optimal tau (midpoint of 0.1 and 0.3)");
        assert_close(f1, 1.0, "perfect macro F1 at optimum");

        // Sanity: decoding at the returned tau reproduces the perfect labels.
        let preds = margin_decode(&margins, &attack, tau, 0);
        assert_eq!(preds, vec![0, 0, 1, 2]);
    }

    #[test]
    fn test_find_optimal_margin_tau_empty() {
        let (tau, f1) = find_optimal_margin_tau(&[], &[], &[], 3, 0);
        assert_eq!(tau, 0.0);
        assert_eq!(f1, 0.0);
    }

    #[test]
    fn test_margin_tau_sweep_matches_brute_force() {
        // Randomized cross-check: incremental sweep == brute-force best over
        // all candidate midpoints.
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let k = 4usize;
        let n = 200usize;
        let targets: Vec<i64> = (0..n).map(|_| rng.gen_range(0..k) as i64).collect();
        let margins: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let attack: Vec<u32> = (0..n).map(|_| rng.gen_range(1..k) as u32).collect();

        let (tau, f1) = find_optimal_margin_tau(&margins, &attack, &targets, k, 0);

        // Brute force over the same candidate set (midpoints + last + 1e-6).
        let mut sorted: Vec<f64> = margins.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();
        let mut candidates: Vec<f64> = sorted.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        candidates.push(sorted[sorted.len() - 1] + 1e-6);
        let scores = vec![0.0; n * k];
        let mut best = (0.0f64, -1.0f64);
        for &c in &candidates {
            let preds = margin_decode(&margins, &attack, c, 0);
            let m = metrics_from_predictions(&scores, &preds, &targets, k, 0);
            if m.macro_f1 > best.1 {
                best = (c, m.macro_f1);
            }
        }
        assert_close(f1, best.1, "sweep macro-F1 == brute force");
        // The tau itself may differ when several boundaries tie on F1; the
        // achieved F1 is the contract.
        let preds = margin_decode(&margins, &attack, tau, 0);
        let m = metrics_from_predictions(&scores, &preds, &targets, k, 0);
        assert_close(m.macro_f1, best.1, "decoded F1 at swept tau == brute force");
    }
}
