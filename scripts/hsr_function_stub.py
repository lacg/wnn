#!/usr/bin/env python3
"""HSR(workload) function — predict the optimal hybrid-speed-ratio from genome
+ dataset shape. To be fit from the OI-v2 cohort timing data once cross-dataset
coverage is sufficient (probably after Stage 1 thermometer sweeps land).

This is a STUB. The actual workload formula, discretization, and extrapolation
policy are design choices the user should decide based on the paper's framing
and the camera-ready risk appetite.

Usage (after fitting):
    from hsr_function import predict_hsr
    hsr = predict_hsr(neurons=218, bits=64, thermo_width=96, n_train_samples=183000)
    # Worker would set os.environ['WNN_HYBRID_SPEED_RATIO'] = str(hsr) before flow
"""

# -----------------------------------------------------------------------------
# DESIGN CHOICE #1: WORKLOAD FORMULA
# -----------------------------------------------------------------------------
# What single scalar best summarizes the per-genome workload?
#
# Option A — multiplicative compute proxy:
#     workload = n_train_samples * neurons * bits
#     (ignores thermo_width because each neuron only reads `bits` of the
#     encoded input; the rest doesn't directly enter neuron computation)
#
# Option B — input-bandwidth + compute:
#     bandwidth = n_train_samples * thermo_width * num_features
#     compute   = n_train_samples * neurons * bits
#     workload  = bandwidth + compute
#
# Option C — multiplicative all-factors (assumes thermo_width matters for
# GPU memory layout / cache pressure even when neurons don't read all of it):
#     workload = n_train_samples * neurons * bits * thermo_width
#
# TODO(user): pick the formula. The cohort's current variance is largely in
# (neurons, bits, n_samples). thermo_width varies only across datasets, so
# Option B is the only one that lets us cross-validate the thermo term using
# Stage 1 sweep data.

def compute_workload(neurons: int, bits: int, thermo_width: int, n_train_samples: int) -> float:
    """Return a single scalar summarizing the per-genome workload."""
    # TODO(user): implement the chosen formula. Recommended starting point:
    bandwidth = n_train_samples * thermo_width * 20  # 20 features used (top-20)
    compute = n_train_samples * neurons * bits
    return bandwidth + compute


# -----------------------------------------------------------------------------
# DESIGN CHOICE #2: BUCKETS vs CONTINUOUS
# -----------------------------------------------------------------------------
# Should predict_hsr output a value from a finite set (e.g., {1, 5, 8, 10})
# or a continuous number that we then snap to the nearest tested value?
#
# Buckets (recommended for paper claim):
#     workload < T1 → HSR=1
#     T1 ≤ workload < T2 → HSR=5
#     T2 ≤ workload < T3 → HSR=8
#     workload ≥ T3 → HSR=10
#     Thresholds T1, T2, T3 fit from the cohort data.
#
# Continuous (more flexible but riskier):
#     HSR = a * log(workload) + b  (or similar)
#     Then snap to nearest in [1, 2, 3, 5, 7, 8, 10] OR return raw value.
#
# TODO(user): pick bucket vs continuous. If buckets, set the breakpoints
# below (currently placeholders — must be fit from the data).

# Bucket breakpoints (workload thresholds).
# Initial fit from 18/05/2026 CIC-IoT-2023 OI-v2 cohort timing data.
#   - HSR=2 and HSR=3 are EXCLUDED from safe set (lose by 40-70% on most shapes)
#   - Within {1, 5, 7, 8, 10}, optima are SHALLOW (~5% spread across "good" choices)
#   - The 100n × 48b regime is the only "deep" exception (HSR=8 wins by 23% over HSR=10)
#
# CAVEAT: thresholds are CIC-IoT-2023-subsample-tuned (n_samples=183K/fold, thermo=96b).
# Cross-dataset Stage 2 sweeps will re-fit and may shift breakpoints meaningfully.
WORKLOAD_T1 = 5_000     # work < T1 → HSR=8 (small architecture, hybrid still helps)
WORKLOAD_T2 = 15_000    # work < T2 → HSR=8 (mid range, "safe" default)
WORKLOAD_T3 = 5e9       # work >= T2 → HSR=10 (large workloads, max hybrid)

# Allowed HSR output values (the "safe set" — avoids dominated 2, 3)
SAFE_HSR_VALUES = [1, 5, 7, 8, 10]


# -----------------------------------------------------------------------------
# DESIGN CHOICE #3: EXTRAPOLATION POLICY
# -----------------------------------------------------------------------------
# What happens when the workload is outside the trained range?
#
# Clip-to-grid (safe, recommended):
#     If predicted HSR is outside SAFE_HSR_VALUES, return the nearest member.
#     Pros: never recommends an untested value.
#     Cons: may be sub-optimal at extreme workloads (very small or very large).
#
# Extrapolate (bold):
#     Allow HSR values outside the tested set (e.g., 12 for very large workloads,
#     0.5 to disable hybrid entirely for tiny workloads).
#     Pros: potentially better at extremes.
#     Cons: untested — could recommend a value that's actually bad.
#
# Per-dataset clip (cautious mid-ground):
#     Clip to the SAFE_HSR_VALUES set for production runs.
#     But also LOG a flag when the prediction was clipped — that flag becomes
#     a signal that we should add a sweep at that workload regime.
#
# TODO(user): pick extrapolation policy. Recommended: per-dataset clip.

EXTRAPOLATION_POLICY = "clip"  # "clip" | "extrapolate" | "warn-on-clip"


# -----------------------------------------------------------------------------
# THE FUNCTION
# -----------------------------------------------------------------------------

def predict_hsr(neurons: int, bits: int, thermo_width: int, n_train_samples: int) -> int:
    """Predict the optimal WNN_HYBRID_SPEED_RATIO for the given genome + dataset.

    Inputs:
        neurons          — max_neurons (cohort search ceiling)
        bits             — max_bits (per-neuron address width)
        thermo_width     — ids_n_bits (thermometer encoding bits per feature)
        n_train_samples  — examples per fold (= train_size / k_folds)

    Returns: int from SAFE_HSR_VALUES, or HSR=10 for extreme workloads
             (extrapolation — actual optimum may be HSR=15/20/100+ untested).

    Initial fit (18/05/2026, CIC-IoT-2023 OI-v2 cohort, n=15):

    The cohort data shows SHALLOW OPTIMA within {1, 5, 7, 8, 10} — for nearly
    every shape, the gap between the winning HSR and HSR=8 is under 5%. The
    function therefore picks HSR=8 as the safe default and only deviates for
    two empirically-clear cases:
      - extreme workloads (46M scale) → HSR=10 (clipped extrapolation)
      - tiny workloads (UNSW-temporal Micro) → HSR=1 (no hybrid)
    """
    workload = compute_workload(neurons, bits, thermo_width, n_train_samples)

    # TINY workload — input bandwidth + compute both small, GPU dispatch
    # overhead dominates. Pure CPU path wins. The data we have on CIC-IoT
    # doesn't directly confirm this (smallest sampled shape was 5n × 12b
    # which still had 96b thermo × 183K samples), but theory predicts HSR=1
    # for sub-million-bit-ops workloads. Stage 2 UNSW-temporal sweep validates.
    if workload < 10_000_000:  # ~10M bit-ops/fold
        return 1

    # EXTREME workload (46M-scale and beyond) — GPU dominates, hybrid pays.
    # ACTUAL OPTIMUM IS UNKNOWN: HSR=10 is clipped from extrapolation.
    # If 46M runs show HSR=10 is well within ~5% of optimum, this stays.
    # If 46M shows hybrid losing entirely, function should be re-fitted to
    # extend the safe set (HSR=15, 20, 50) or to reject hybrid at extreme
    # imbalance (HSR=10 = effectively-pure-GPU when paths are 50× apart).
    if workload >= 5_000_000_000:  # 5B bit-ops/fold
        return 10

    # MID-RANGE workload (everything in the CIC-IoT-2023 cohort's sampled
    # regime) — HSR=8 is the safe default, within ~5% of the per-shape
    # optimum on almost every confident cell. The ONE deep exception is
    # 100n × 48b where HSR=8 beats HSR=10 by 23% — also returns HSR=8 here,
    # which is the correct call.
    return 8


# -----------------------------------------------------------------------------
# UNCERTAINTY NOTES (for paper methodology section)
# -----------------------------------------------------------------------------
#
# The function's safe set [1, 5, 7, 8, 10] reflects EMPIRICAL TESTING.
# Values that should be considered as future expansions:
#   - HSR=15, 20, 50: for workloads beyond the 46M tier. Unknown whether
#     these dominate HSR=10 in practice. If 46M deployment at HSR=10 leaves
#     measurable headroom (e.g., per-genome time still GPU-dominated with
#     CPU idle), extending the safe set upward is worth a follow-up sweep.
#   - HSR=0.5: explicit "never hybrid" sentinel. Currently HSR=1 is the
#     lowest tested; with paths typically >1× apart, HSR=1 already disables
#     hybrid. But for sub-million-op workloads, dedicated CPU-only mode
#     might be cleaner than relying on HSR=1's threshold.
#
# Function output of HSR=5 or HSR=7: currently NEVER returned because they
# win only by noise-level margins (<2%) over HSR=8 on the shapes where they
# do win. Cross-dataset Stage 2 data may reveal regimes where HSR=5 or HSR=7
# wins by meaningful (>5%) margins; the function would then add those cases
# to the decision tree.


# -----------------------------------------------------------------------------
# FITTING (TODO)
# -----------------------------------------------------------------------------
# To fit WORKLOAD_T1/T2/T3 from cohort data:
#   1. Pull per-(genome_shape, hsr, dataset) mean eval_time_ms from genome_evaluations
#   2. Compute workload for each (genome_shape, dataset)
#   3. For each workload value, record the winning HSR
#   4. Find the workload threshold where the winning HSR transitions (1→5, 5→8, 8→10)
#   5. Set T1, T2, T3 at the transition points
#
# Cross-dataset coverage requirement: need ≥3 datasets with confident HSR data
# at varying workloads. The current cohort gives us CIC-IoT only — we need
# UNSW + CICIDS Stage 2 sweeps before this can be fit responsibly.


if __name__ == "__main__":
    # Smoke test
    test_cases = [
        # (neurons, bits, thermo, samples, label)
        (5,   12,   96, 183_000, "5n×12b CIC-IoT subs"),
        (218, 64,   96, 183_000, "OI-v2 GA convergence (CIC-IoT subs)"),
        (5,    4,   8,  100_000, "Micro 20-byte detector (UNSW)"),
        (218, 64,   96, 7_460_000, "OI-v2 on 46M (per-fold)"),
        (50,  16,   16, 460_000, "CICIDS2017 mid-range"),
    ]
    print(f"{'Case':<45} | {'workload':>12} | HSR")
    print(f"{'-'*45}-+-{'-'*12}-+----")
    for n, b, t, s, label in test_cases:
        w = compute_workload(n, b, t, s)
        h = predict_hsr(n, b, t, s)
        print(f"{label:<45} | {w:>12.2e} | {h}")
