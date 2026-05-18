# Paper updates pending — wait until OI-v2 cohort has n≥30

**Status as of 18/05/2026 ~02:00 UTC**: OI-v2 cohort at 9/112 completed. Hold paper edits until ≥30 NEW flows have landed (estimated ~21/05/2026 if worker runs steady). Then refresh from the data captured in `docs/ids_results.md` (regenerated via `scripts/build_oi_vs_old_report.py`).

## Headline findings to incorporate

1. **Architecture shift under OI training** (the biggest structural change):
   - OLD GA Neurons: ~109±57 neurons × 48±0 bits per neuron
   - NEW GA Neurons: ~218±35 neurons × 64±0 bits per neuron
   - Confirms the prior hypothesis (project memory `oi-cohort-in-flight.md`) that the order-dependent clamped random walk in legacy QUAD training was biasing the GA against larger architectures. With OI removing that bias, the GA reliably finds the 200n×64b regime.
   - Implication: the paper's narrative that "GA converges sharply on b=48±0 bits, multi-modal neuron counts" is true ONLY for the pre-fix training. Post-fix, the GA picks 64 bits and ~220 neurons consistently.

2. **Metric improvements** at calibrated thresholds (preliminary, n=9):
   - best_fitness train_cal: ΔF1 +0.30, ΔFPR −1.63, ΔAcc +0.11
   - best_fitness platt: ΔF1 +0.30, ΔFPR −2.40, ΔAcc +0.09
   - best_fitness val_cal: ΔF1 +0.30, ΔFPR −2.26, ΔAcc +0.09
   - Standard deviations roughly halve (NEW σ≈0.18 vs OLD σ≈0.41) → OI is more reproducible.

3. **fixed_05 threshold behaves qualitatively differently under OI**:
   - OLD fixed_05: F1≈90.87, FPR≈4.78
   - NEW fixed_05: F1≈85.31, FPR≈0.85
   - This is NOT a regression — it's the OI training distribution shift. OI produces more conservative cell saturation, so the uncalibrated 0.5 threshold predicts fewer positives (lower F1, lower FPR). Calibrated thresholds (train_cal, platt, val_cal) correct for this and show NEW > OLD.
   - **Discuss explicitly** to avoid a reader misreading the fixed_05 row as a degradation.

4. **`empirical_cumulative` column semantics changed** (commit 0b7f084d):
   - Pre-fix: pure-F1 sweep (== train_cal, redundant column).
   - Post-fix: GA-fitness-weighted sweep (genuinely different operating point).
   - For OI-v2 cohort with w_fpr=0.35: empirical_cumulative ≈ FPR 7.35 vs train_cal FPR 8.66. Lower-FPR Pareto point now reachable via this column.
   - **Mention in methodology** that the column was repurposed to give a per-flow GA-fitness-optimal threshold; this restores meaning lost in the April-28 refactor.

## Specific table updates

### Table 5 (`tab:ciot-phase`, line 795)

Current paper has 29 of 112 FIXED runs. After cohort maturity:

```
Replace "29 of 112" → state-of-data (likely "n=30+ of 112 OI-v2 cohort, post-fix").

Numeric rows (Grid Search + GA Neurons + fixed_05) will need to be:
  - Either refreshed with OLD n=63 numbers AND a new OI-v2 row added,
  - Or fully replaced with OI-v2 numbers and OLD moved to appendix.

Current OLD n=29 numbers in paper           After cohort maturity (OI-v2 n=30+):
  GA Neurons (train_cal):                       Update to OI-v2 means:
    F1 92.41±0.37                                F1 ~92.9 (likely)
    FPR 10.71±1.12                               FPR ~8.7 (likely)
    Acc 96.27±0.17                               Acc ~96.5 (likely)
  GA Neurons (fixed_05):                       Update with caveat paragraph:
    F1 90.74±0.69                                F1 ~85.3 (lower — OI distribution shift)
    FPR 5.08±0.52                                FPR ~0.85 (much lower — OI conservative)
    Acc 95.09±0.42                               Acc ~91.2 (lower)
```

**Per-genome Pareto extremes** (rows above the cohort means) will change as new OI-v2 best-FPR genomes land. Currently:
- Best F1: 93.44 / 8.28 / 96.75 (r82398 GA, best_acc, train_cal)
- Best F1 (FPR<4%): 91.68 / 3.81 / 95.61 (r54356 GA, best_fpr, fixed_05)

These are OLD-cohort extremes. The OI-v2 cohort will produce its own; re-mine after cohort completes.

### Table 7 (`tab:phase`, line 985) — Phase Progression

Current CIC-IoT-2023 row (val_cal threshold, best-fitness):
```
Grid Search:  F1 89.32±0.20  FPR 15.89±0.82  Neurons 50±0    Bits 69±15
GA Neurons:   F1 92.42±0.37  FPR 11.00±0.93  Neurons 51±4    Bits 46±4
Δ:            +3.10           −4.89           +1 (+2%)        −23
```

After OI-v2 cohort maturity, replace with OI-v2 numbers. Expected (extrapolated from n=9):
```
Grid Search:  F1 90.97±0.19  FPR ~15.8        Neurons ~160    Bits ~43
GA Neurons:   F1 92.92±0.18  FPR ~8.3         Neurons ~218    Bits 64±0
Δ:            +1.95           −7.5            +58 (+36%)       +21
```

**The Δ row's interpretation flips**: paper's CIC-IoT row currently shows GA REDUCES neuron count. Under OI, GA INCREASES neuron count by ~36%. The "neuron-count optimization as regularizer" narrative in the prose around line 967 needs revision — it holds for CICIDS2017 and UNSW-NB15 (both show pruning) but not for CIC-IoT-2023 under OI.

### Table 6 (`tab:ciot-46m`, line 902) — Full 46M

Not directly affected by OI cohort (it's a different experiment with small-genome architectures on 46M). Already flagged in caption: "Peak and Search(46M) reflect prior dataset mirrors and will be refreshed in the camera-ready with results from the 250n×100b architecture trained with canonical TOP20."

When the OI-v2 250n×100b cohort completes, the architectures (218n×64b regime) should be evaluated on full 46M to populate fresh Peak and Search(46M) rows.

### Appendix

Look for any per-seed result tables that mention specific OLD-cohort flows (r-numbers). The 4 OI-OLD flows (2671-2674) — if any of their numbers are cited — should be replaced with OI-v2 cohort numbers after maturity.

## Discussion edits

### §6.1 Connectivity as Feature Selection (line 1029)

Add paragraph noting that under OI training, the GA's neuron-count selection produces LARGER architectures (~218 vs ~109), not just feature-selection via pruning. The "neurons whose random connectivity captures discriminative input combinations survive" interpretation still holds — but the surviving set is bigger, suggesting more of the random connectivity space is genuinely informative once training bias is removed.

### §6.2 Threshold Sensitivity (line 1058)

Mention:
- Pre-April-28 train_cal accidentally fit on EVAL data (oracle-leak adjacent). The April-28 refactor (commit f04da00f) fixed this. Paper-submission numbers used the corrected train_cal.
- empirical_cumulative semantics: now reports the GA-fitness-weighted threshold (genuinely different operating point from train_cal); pre-fix it was pure-F1 (identical to train_cal, redundant column).
- The fixed_05 vs calibrated threshold gap WIDENS under OI training because OI produces more conservative cell saturation. This is a feature of training, not a bug; it just means calibration matters more under OI.

### NEW subsection: Order-Independent Training (suggested)

A 0.5–1 page subsection describing:
- The bug: legacy QUAD training was a clamped random walk on cell states, not a vote tally. Cells saturate quickly based on example order, not just net vote sign.
- The fix: order-independent vote accumulation (commit `1082e2f4` merged from path2-lm-followup branch). Gated by `WNN_ORDER_INDEPENDENT_TRAIN=1`.
- The effect: more cells stay at WEAK_FALSE / WEAK_TRUE rather than saturating to FALSE / TRUE; vote-distribution is wider; calibrated thresholds shift; GA converges to larger architectures because the order-dependent walk no longer penalizes them.
- Reproducibility wins: std halves on cohort metrics.

This is a major methodology contribution and should be visible in the paper.

## Watching for cohort maturity

- Cohort target: 112 OI-v2 flows
- Current: 9 completed, 1 running, ~103 queued
- Average flow duration: ~6h per flow (HSR-tagged) or ~1.6h (HSR=1 was fastest at ~98 min)
- Rough ETA for n=30: ~6 × 21 / 8 parallel = if-serial-only: ~5 days; realistic ~3-5 days from 18/05/2026
- Rough ETA for full n=112: ~14-20 days

To monitor:
```bash
python3 /Users/lacg/wnn/scripts/build_oi_vs_old_report.py 2>&1 | head -3
# When NEW cohort shows ≥30 in the count, time to update the paper.
```

## Cross-dataset OI validation (UNSW + CICIDS)

The CIC-IoT-2023 OI-v2 cohort reveals that the GA *grows* architecture under OI
(~109n×48b → ~218n×64b), contradicting the paper's "neuron-count optimization as
regularizer (pruning)" narrative on this dataset. Whether the same flip happens
on CICIDS2017 and UNSW-NB15 is an **open empirical question** that must be
answered before the camera-ready paper claims hold cross-dataset.

Current cohort inventory (all pre-fix, FIXED):
```
CICIDS2017    : 162 completed  → OLD cohort exists, needs renaming + OI rerun
UNSW-random   : 130 completed  → OLD cohort exists, needs renaming + OI rerun
UNSW-temporal : 184 completed  → OLD cohort exists, needs renaming + OI rerun
```

Plan (after CIC-IoT OI-v2 reaches n≥30 and frees the worker queue):

1. **Rename existing cohorts** with `-FIXED-OLD-` suffix (same pattern used for
   CIC-IoT cohort; preserves history without losing seed-pairing info).
2. **Queue OI-v2 cohorts** for each dataset:
   - CICIDS2017: 30+ flows (matches CIC-IoT cadence; 112 if compute permits)
   - UNSW-random: 30+ flows
   - UNSW-temporal: 30+ flows (paper's primary UNSW evaluation)
3. **Re-run on same architecture** the paper currently uses for each dataset
   (CICIDS2017: not 250n×100b — paper shows 16-bit thermometer; check what
   genome shape the GA actually selected). Important: don't FORCE 250n×100b if
   the existing best architecture is different — let the GA pick under OI from
   the same search space, then compare.

Per-dataset narrative questions to answer:
- **CICIDS2017** (current paper: GA prunes 343→198 neurons, F1 +0.11): does
  OI still prune? Or does it grow neurons like CIC-IoT?
- **UNSW-temporal** (current paper: GA barely changes 348→333, F1 +0.82):
  the temporal distribution shift may bias differently. Worth checking
  whether OI tightens std (which would be a methodology win regardless of
  architecture direction).
- **UNSW-random**: paper noted at submission it had only 50/112 runs; the
  130 completed reflects later additions. OI cohort should target ≥30.

Compute estimate (revised 18/05/2026 ~02:30 UTC after seeing actual durations):

OI flows hit patience-based early stopping at **60-80 generations** (not the
250 max), with wall times of **80-105 minutes** per flow:
```
HSR1  r60218: 70 gen / 85.6 min     HSR3  r72358: 80 gen / 104.5 min
HSR1  r81071: 80 gen / 98.2 min     HSR5  r10329: 80 gen / 94.4 min
HSR2  r38428: 70 gen / 79.6 min     HSR7  r49616: 60 gen / 79.6 min
HSR3  r47401: 70 gen / 89.7 min     HSR8  r16362: 70 gen / 83.4 min
HSR10 r70285: 70 gen / 80.0 min
```
Average ≈ 88 min ≈ 1.5h per flow.

**Target scope (per user, full camera-ready validation):**
- CIC-IoT-2023 OI-v2 (1.14M subsample): 112 flows  ← in progress, 9/112 done
- CICIDS2017 OI: 112 flows
- UNSW-random OI: 112 flows
- UNSW-temporal OI: 112 flows
- CIC-IoT-2023 46M (full dataset) OI: 2-4 flows (full-scale validation)

Total: ~450 flows. At 1.5h/flow serial: ~28 days. With faster datasets (smaller
than CIC-IoT) and early stopping potentially firing sooner on simpler tasks,
**realistic estimate: 2.5 weeks (~17 days)**. Still within camera-ready budget.

## Per-dataset config sweeps before the main OI cohorts

The CIC-IoT-2023 cohort's current config is the result of multiple sweeps that
established defaults: 96-bit thermometer encoding, 250n × 100b architecture
ceiling, fitness weights, etc. **Those sweeps were run only on CIC-IoT-2023.**
Before queuing 112-flow main cohorts for CICIDS2017 / UNSW-temporal / UNSW-random,
we need to validate that the same defaults hold (or tune per dataset).

### Stage 1: 2-flow pilot per dataset (apples-to-apples check)

Run 2 flows on each non-CIC-IoT dataset with the current CIC-IoT cohort config:
- 96-bit thermometer encoding (NEW — papers' published UNSW/CICIDS used 8b/16b)
- 250n × 100b architecture ceiling
- OI training enabled
- Empirical_cumulative fixed (fitness-weighted)
- HSR = current env default (will tune in Stage 2)

Compare results to:
- Per-dataset best from the existing paper (UNSW-temp F1 ~87%, UNSW-rand near-perfect,
  CICIDS2017 99.3% F1 random)
- The new CIC-IoT OI-v2 cohort numbers (F1 92.9% at calibrated thresholds)

**Decision points after pilot:**
- If 96-bit thermometer matches or exceeds the per-dataset published baseline,
  proceed with 96b for all cross-dataset OI cohorts.
- If 96b is meaningfully worse on a dataset, consider falling back to that
  dataset's published encoding (8b/16b) for the main cohort.
- 250n × 100b architecture ceiling: same check — does the GA still converge
  comfortably below the 250n cap, or does it want more headroom on some datasets?

Total Stage 1: 2 flows × 3 datasets = 6 flows.

### Stage 2: Reduced HSR sweep per dataset (24 flows each)

After pilot validates the architecture/encoding choices, run a reduced HSR
sweep to find each dataset's optimal HSR threshold. **Skip HSR=2 and HSR=3 —
both are dominated in CIC-IoT data** (HSR=3 by 4 other values with 13-26%
mean gaps; HSR=2 by HSR=5 and HSR=10). The reduced grid `[1, 5, 8, 10]` covers
the actionable range with 24 flows instead of 56.

```
Per dataset:  4 HSR values × 6 seeds = 24 flows
Total:        3 datasets × 24 = 72 flows
```

Expected dataset-specific HSR optimum (theoretical prediction; empirical test needed):
- CICIDS2017 (largest dataset, mid encoding): HSR=5 or 8 (similar regime to CIC-IoT)
- UNSW-NB15 (smallest dataset, narrow encoding): HSR=1 (pure-path, hybrid never pays)

### Stage 3: Main 112-cohort per dataset (with dataset-specific HSR)

After Stages 1 and 2 lock in encoding + architecture + HSR per dataset, queue
the canonical 112-cohort:

```
Per dataset:  ~112 OI flows  +  the 112 pre-fix FIXED-OLD already done (rename)
Total:        3 datasets × 112 OI = 336 flows
```

### Stage 3.5: Reduced 46M validation runs

Per the existing plan, also queue 2-4 OI flows on the **46M CIC-IoT-2023 full
dataset** (architectures from the OI-v2 250n×100b cohort) so the paper's
flagship full-scale row also gets the OI treatment. These take longer per-flow
than the subsample runs (rough estimate: 4-8h each at full size).

```
46M OI runs:   ~4 flows × ~6h = ~24h
```

### Total compute budget

```
Stage 1 (pilots):                6 flows × ~1.5h  = ~9h
Stage 2 (reduced HSR per ds):   72 flows × ~1.5h  = ~108h  (~4.5 days)
Stage 3 (main cohorts):        336 flows × ~1.5h  = ~504h  (~21 days)
Stage 3.5 (46M):                ~4 flows × ~6h    = ~24h   (~1 day)
---------------------------------------------------+
                                                  ~27 days serial

Realistic with patience-based early stopping (60-80 gen typical): ~17 days
Plus current CIC-IoT cohort completing:        ~5 days

Combined total before paper camera-ready:      ~22 days
```

Still within camera-ready budget (~2.5-3 weeks total from 18/05/2026).

### Open question (worth flagging in the paper methodology)

The HSR-as-function approach (HSR_opt as a function of `(neurons, bits)` per
dataset, or more ambitiously `(neurons, bits, examples)` for a fully-portable
function) is **future work**. Current pragmatic recommendation: tune per
dataset via the reduced sweep, then fix HSR for the main cohort.

The `(neurons, bits, batch_size)` formulation would let HSR transfer across
datasets without per-dataset tuning — but it'd need fitting data from at least
3 datasets to validate. Currently a 1-dataset interpolation that overfits to
CIC-IoT specifics.

## Related memories

- `project_oi_cohort_v2_rebuild.md` — the cohort setup
- `project_oi_training_shipped.md` — the OI training fix details
- `project_training_clamped_random_walk.md` — the underlying bug
- `project_raid2026_submitted.md` — paper submission state
