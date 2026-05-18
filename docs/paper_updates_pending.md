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

### Dataset sizing reference

```
Dataset                          | Train     | Test    | Per-fold (K=5)  | Per-fold work @ 96b
---------------------------------+-----------+---------+------------------+--------------------
UNSW-NB15 temporal               |   175K    |   82K   |     ~28K         | ~54M bit-ops
UNSW-NB15 random (dedup)         |  1.27M    |  317K   |    ~203K         | ~390M bit-ops
CIC-IoT-2023 1.14M subsample     |  0.91M    |  286K   |    ~146K         | ~280M bit-ops
CICIDS2017 random                |  2.30M    |  566K   |    ~368K         | ~706M bit-ops
(46M CIC-IoT-2023 full)          | 37.3M     |  9.3M   |   ~5.96M         | ~11.4B bit-ops
```

Implications for HSR per dataset (theoretical predictions, validated in Stage 2):
- **UNSW-temporal**: GPU dispatch overhead dominates → HSR=1 (no hybrid) likely
- **UNSW-random**, **CIC-IoT subs**: mid-range → HSR=5-8 likely (similar regime)
- **CICIDS2017**: largest per-fold work → HSR=8-10 likely
- **46M**: extreme → HSR=10 almost certainly

### Stage 1: Coarse thermometer sweep per dataset

The CIC-IoT-2023 thermometer sweep was fine-grained (multiple widths from 2b to
64b — Section "Encoding Resolution and Saturation" in the paper). For
UNSW/CICIDS we only need a **coarse sweep** to find which encoding width OI
prefers. Grid of 6 widths × 2 seeds × 3 datasets = 36 flows.

```
Thermometer widths to test : [8, 16, 32, 48, 64, 96] bits per feature
Seeds per (dataset, width) : 2
Datasets                   : UNSW-temporal, UNSW-random, CICIDS2017
Architecture (locked)      : 250n × 100b ceiling
Training (locked)          : OI enabled, empirical_cumulative fixed
HSR (locked)               : env default (will tune in Stage 2)
Total                      : 6 × 2 × 3 = 36 flows
```

Decision after Stage 1: pick the F1-optimal encoding per dataset (or the
smallest encoding that achieves within ~0.5pp of the best, for parsimony).

Expected encodings (based on per-dataset workload and prior published configs):
- UNSW-temporal: likely 8-16b (small dataset, narrow encoding sufficient)
- UNSW-random: 16-32b (larger dataset, more resolution helps)
- CICIDS2017: 16-32b (paper baseline was 16b)

The point of the sweep is to test whether OI shifts the optimal width.

### Stage 1b (optional): apples-to-apples 96b pilot

If Stage 1 shows that 96b is the consistent winner on UNSW/CICIDS — same as
for CIC-IoT-2023 — then no extra pilot is needed; just use 96b everywhere.
If Stage 1 picks dataset-specific encodings, optionally run **2 flows per
dataset at 96b** as well to publish a "common-encoding" comparison alongside
the per-dataset-optimal results. Cheap insurance for the methodology section.

```
Stage 1b: 2 flows × 3 datasets = 6 flows (optional)
```

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

### Timeline anchors (camera-ready deadline)

```
Today               : 18/05/2026
Current CIC-IoT ETA :  24/05/2026  (cohort hits 112/112)
Notification        : 10/07/2026  (~7 weeks budget for new cohorts)
Camera-ready        : 13/08/2026  (~12-13 weeks total)
```

The 22-day budget below is the SERIAL compute estimate — there's roughly 7
weeks of wall-clock room before notification, so the plan fits with margin
for re-runs, debugging, and Stage-N+1 iteration if something needs revising.

### Total compute budget

Per-flow times vary substantially by dataset:
- UNSW-temporal: ~30 min/flow (smallest workload)
- UNSW-random: ~75 min/flow
- CICIDS2017: ~120 min/flow (largest non-46M)
- CIC-IoT subs: ~94 min/flow (current cohort baseline)
- 46M CIC-IoT: ~6h/flow (extrapolated; needs validation)

```
Stage 1   (coarse thermo sweep, 6 widths × 2 seeds × 3 ds = 36 flows):
  UNSW-temp:   12 flows × 30 min  = ~6h
  UNSW-rand:   12 flows × 75 min  = ~15h
  CICIDS:      12 flows × 120 min = ~24h
  Subtotal                        = ~45h (~2 days)

Stage 1b  (optional 96b pilot, 2 flows × 3 ds = 6 flows):
                                    ~7h (~0.3 day)

Stage 2   (reduced HSR sweep, 4 widths × 6 seeds × 3 ds = 72 flows):
  UNSW-temp:   24 flows × 30 min  = ~12h
  UNSW-rand:   24 flows × 75 min  = ~30h
  CICIDS:      24 flows × 120 min = ~48h
  Subtotal                        = ~90h (~3.8 days)

Stage 3   (main 112-cohort × 3 datasets = 336 flows):
  UNSW-temp:  112 flows × 30 min  = ~56h
  UNSW-rand:  112 flows × 75 min  = ~140h
  CICIDS:     112 flows × 120 min = ~224h
  Subtotal                        = ~420h (~17.5 days)

Stage 3.5 (46M OI runs, ~4 flows × 6h):
                                    ~24h (~1 day)

────────────────────────────────────────────────────
TOTAL                              ~24 days serial
With patience-based early stopping (60-80 gen typical reduction): ~15-18 days
Plus current CIC-IoT cohort completing:                            ~5 days
────────────────────────────────────────────────────
Combined total before camera-ready:                                ~20-23 days
```

Still well within the camera-ready budget — 22-24 day SERIAL estimate vs ~7 weeks
of wall-clock room before notification (10/07/2026). Leaves slack for re-runs
and any Stage-N+1 iteration needed.

### Execution order (decided 18/05/2026)

1. **Finish current CIC-IoT-2023 OI-v2 cohort** to 112/112 (~5-6 days from now).
2. **Queue 4× CIC-IoT-2023 46M OI runs** (~1 week at ~6h each), plus the small
   "Search(46M)" architectures from the existing paper's Table 6 if any
   still need re-running.
3. **Then** start the cross-dataset cycle (Stages 1 → 2 → 3 → 3.5 above) for
   UNSW-temporal, UNSW-random, CICIDS2017.

This sequencing keeps the flagship CIC-IoT-2023 dataset's numbers nailed
down first (the paper's strongest single dataset story), then expands
cross-dataset validation.

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
