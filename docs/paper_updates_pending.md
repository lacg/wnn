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
2. **Queue 2× CIC-IoT-2023 46M OI runs** (each flow runs grid_search +
   ga_neurons at full 37.3M-train scale; **historically 1-2 weeks per flow**).
   Plan is intentionally 2 flows, not 4 — the per-flow time dominates.
   Total: ~2-4 weeks for both 46M flows.
3. **Then** start the cross-dataset cycle (Stages 1 → 2 → 3 → 3.5 above) for
   UNSW-temporal, UNSW-random, CICIDS2017.

This sequencing keeps the flagship CIC-IoT-2023 dataset's numbers nailed
down first (the paper's strongest single dataset story), then expands
cross-dataset validation.

### Revised wall-clock timeline (with realistic 46M estimates)

```
Today             18/05/2026  — CIC-IoT cohort at ~16/112
                                  ↓ ~5-6 days
~24/05/2026       CIC-IoT-2023 1.14M OI-v2 cohort done (112/112)
                                  ↓ ~2-4 weeks (2× 46M flows, serial)
~07/06 – 21/06    2× 46M OI runs done
                                  ↓ ~3 weeks (Stages 1-3.5)
~28/06 – 12/07    UNSW + CICIDS OI cohorts done
                                  ↓
10/07/2026        Notification deadline  ← tight if 46M lands on slow end
                                  ↓ ~5 weeks
13/08/2026        Camera-ready
```

**Risk note**: if 46M flows trend toward the 2-week-per-flow end, the
cross-dataset phase risks brushing up against notification (10/07). Mitigation
options if that happens:
- Reduce UNSW/CICIDS main cohort from n=112 to n=50 (saves ~10 days)
- Run only one UNSW split for the camera-ready (drop temporal or random)
- Push cross-dataset to post-notification revision window (5 weeks Jul-Aug)

The 46M runs are the critical path — their actual wall-clock will determine
whether the cross-dataset cycle is fully comfortable or starts to squeeze.

### Open question (worth flagging in the paper methodology)

The HSR-as-function approach (HSR_opt as a function of `(neurons, bits)` per
dataset, or more ambitiously `(neurons, bits, examples)` for a fully-portable
function) is **future work**. Current pragmatic recommendation: tune per
dataset via the reduced sweep, then fix HSR for the main cohort.

The `(neurons, bits, batch_size)` formulation would let HSR transfer across
datasets without per-dataset tuning — but it'd need fitting data from at least
3 datasets to validate. Currently a 1-dataset interpolation that overfits to
CIC-IoT specifics.

## HSR-as-function approach (preferred over per-dataset sweeps long-term)

Stub: `scripts/hsr_function_stub.py` — sketches a 4-parameter predictor
`predict_hsr(neurons, bits, thermo_width, n_train_samples) → HSR ∈ {1, 5, 7, 8, 10}`.

### Why a function beats per-dataset HSR tuning

Per-dataset HSR tuning treats each dataset as a separate experiment. A function
of *workload* captures the underlying physics (bandwidth + compute vs dispatch
overhead) and **transfers across datasets without re-tuning**. The same function
should predict HSR=10 for 46M CIC-IoT *and* HSR=1 for tiny UNSW-temporal —
because the workload itself is the input, not the dataset identity.

### Three design choices (TODOs in the stub)

1. **Workload formula**: pick one of
   - Multiplicative compute proxy (`samples × neurons × bits`)
   - Bandwidth + compute (separate terms for input bandwidth and neuron compute)
   - All-factors product (treats thermo_width as memory-pressure multiplier)

   Stub uses bandwidth + compute as starting point.

2. **Buckets vs continuous output**: currently bucketed
   (`{1, 5, 7, 8, 10}`). HSR=2 and HSR=3 are **excluded from the safe set**
   because they're dominated everywhere (40-70% slower than the winning HSR
   on every shape with sufficient data).

3. **Extrapolation policy**: stub uses **clip to safe set**. If a 46M
   workload predicts HSR=12 from a linear extrapolation, the function returns
   HSR=10 (highest tested). Safer than recommending an untested value.

### Initial fit (CIC-IoT-2023 OI-v2 cohort, n=15)

The current cohort data shows **shallow optima**: within the safe set
`{1, 5, 7, 8, 10}`, the spread is ~5% across most shapes. The big penalties
are for HSR=2 and HSR=3 (40-70% slower). One genuine deep optimum exists
at 100n × 48b where HSR=8 beats HSR=10 by 23.3%.

Initial function (encoded in stub):
```
IF (75n ≤ neurons ≤ 125n AND 40 ≤ bits ≤ 56):  return HSR=8  ← deep optimum
IF (neurons × bits < 5000 AND samples < 100K):   return HSR=1  ← tiny workload
IF (workload ≥ 5e9):                              return HSR=10 ← extreme workload
IF (neurons × bits ≥ 15000 AND bits ≥ 64):       return HSR=8  ← mid-large
DEFAULT:                                          return HSR=8  ← shallow optimum middle
```

### Validation plan

1. **Stage 1 (thermo sweeps)**: validates the `thermo_width` term. If a
   dataset's optimal HSR transitions cleanly as thermo width grows from
   8 → 96, the formula's bandwidth term is well-calibrated.

2. **Stage 2 (HSR sweeps per dataset)**: validates the `n_samples` term and
   `neurons × bits` interactions. With UNSW-temp (28K/fold), UNSW-rand (203K),
   CICIDS (368K), and CIC-IoT subs (146K), we span ~13× in per-fold workload.
   Enough to fit log-scale bucket thresholds.

3. **46M deployment**: pure extrapolation test. Function predicts HSR=10;
   if measured timing confirms (any HSR ≥ 5 within ~5%), function is trusted
   for future 46M-scale runs.

### Paper scope decision (18/05/2026)

**THIS paper (RAID 2026)**: HSR function is NOT mentioned. Used silently in
the worker because it affects only wall-clock dispatch time, not the F1/FPR/Acc
numbers in the tables. Same numbers would be produced regardless of HSR
choice; the function just picks the fastest path.

**Follow-up paper / workshop note**: write up as a methodology contribution.
Title candidate: "Workload-Aware Hybrid Dispatch for Evolutionary WNN Training
on Apple Silicon." Sections:
  1. Problem: CPU+GPU hybrid dispatch threshold is a tuning hyperparameter
  2. Measurement: 56-flow HSR sweep across 7 values × 8 seeds = empirical data
  3. Finding: shallow optima within {1, 5, 7, 8, 10}; catastrophic penalties
     for HSR=2/3; 100n × 48b deep optimum at HSR=8
  4. Function: workload bucket → HSR_opt; clipping policy for extrapolation
  5. Validation: cross-dataset (UNSW, CICIDS, 46M CIC-IoT) confirms transfer
  6. Cost-benefit: ~20h experiment cost, ~10-15 days compute saved across
     paired cohorts. ~50-100× payback ratio.

The CIC-IoT-2023 + UNSW-temp + UNSW-rand + CICIDS Stage 2 sweeps provide
the dataset diversity needed to fit + validate the function. The paper's
contribution is the *methodology* (workload formula + bucketed function +
extrapolation policy), not the specific bucket thresholds.

### Worker integration

**Already shipped** in commit `5de5f361` (18/05/2026). The worker calls
`predict_hsr_from_params()` for any flow without an explicit
`wnn_hybrid_speed_ratio` in its config. Explicit overrides (e.g., sweep
experiments) still win.

## Scope boundaries: full diff vs submission

The submission tag is `raid-2026-submitted` (commit `42d3ee58`, 17/04/2026).
Diffing against that snapshot reveals SIX changes between submission and
camera-ready prep, of varying severity:

| Change                          | Affects results?      | Type        | Treatment in RAID 2026 paper                          |
|---------------------------------|-----------------------|-------------|-------------------------------------------------------|
| **Encoding 8b → 96b** (CIC-IoT) | YES (~+11 pp F1)      | METHODOLOGY | **Needs explicit disclosure or revert**               |
| **Architecture 500n×34b → 250n×100b** | YES (different search space, GA finds new regime) | METHODOLOGY | **Needs explicit disclosure or revert** |
| OI training fix                 | Yes (+0.3 F1, std halved) | Bug fix | Reported as correction (existing methodology)         |
| empirical_cumulative repurpose  | Yes (column semantic) | Refinement  | Reported as column-semantic clarification             |
| Cache-key `_oi` suffix          | No (no cache hits)    | Bug fix     | Silent                                                |
| HSR function                    | No (wall-time only)   | New module  | Silent (defer to follow-up paper as methodology)      |

**The first two rows are the camera-ready risk.** They are real methodology
changes that affect results materially (+11 pp F1 on CIC-IoT-2023). Reviewers
reviewed the 8-bit / 500n×34b version with ~80% F1; the camera-ready would
show 96-bit / 250n×100b with ~93% F1.

Options:

1. **Disclose + defend**: add a "Post-submission methodology improvements"
   paragraph in §3, explaining (a) the thermometer-width sweep that
   identified 96-bit as optimal for CIC-IoT-2023, and (b) the architecture-
   ceiling re-tuning. Frame as discoveries made during cohort runs. Risk:
   reviewers may flag the magnitude of change.

2. **Revert to submitted methodology**: re-run camera-ready cohorts at 8b /
   500n×34b. Numbers stay close to submission (with OI fix only). Defer
   the 96b / 250n×100b improvements to a follow-up paper. Risk: lose the
   strongest single-dataset result.

3. **Hybrid**: keep 8b primary in main tables (matches submission), add
   "post-submission ablation" appendix showing 96b improvement. Most
   transparent; preserves both stories.

4. **Ask the PC chair**: many venues have explicit channels for "we want to
   include post-submission improvements; is this allowed?" Lowest risk
   path for big methodology changes — get permission rather than
   forgiveness.

**Recommendation**: option 4 first (ask), then option 1 if approved, then
option 3 if borderline, then option 2 if pushed back. Decide before
camera-ready, ideally as soon as the cohort matures (n≥30).

## Camera-ready edit plan (concrete location-by-location)

Sequence depends on the PC-chair decision above. Assuming **option 1 (disclose
+ defend)** is approved, the edits are:

### §3 (Methodology / Training) — line ~493 area

Add a "**Post-submission methodology improvements**" paragraph after the
existing training description:

> *"During the post-submission period, we identified one implementation
> issue and conducted two methodological refinements. (1) **Training-order
> correction**: the legacy QUAD-state cell update was a sequential clamped
> random walk on cell values rather than the intended order-independent
> vote tally. We corrected to single-pass accumulation followed by a fixed
> bin function ({≤−1 → FALSE, 0 → WEAK_FALSE, +1 → WEAK_TRUE, ≥+2 → TRUE}).
> Impact: F1 mean +0.3 pp at calibrated thresholds, FPR mean −1.6 pp,
> cohort std halved (≈0.4 → ≈0.2) — order independence improves
> reproducibility as expected. (2) **Thermometer width re-tuning**: a
> post-submission encoding sweep on CIC-IoT-2023 identified 96-bit
> thermometer as substantially better than the submitted 8-bit. We adopted
> 96-bit for the camera-ready CIC-IoT-2023 cohort; the wider encoding
> gives higher feature-discrimination resolution for the dataset's
> heavy-tailed feature distributions. (3) **Architecture-ceiling
> re-tuning**: the submitted cap of 500n × 34b was widened to 250n × 100b
> based on a discovery that the GA prefers higher-bits / lower-neurons
> regimes than the submission cap allowed. All CIC-IoT-2023 numbers in
> this camera-ready reflect these three refinements. UNSW-NB15 and
> CICIDS2017 use their submitted methodologies unchanged [or: are also
> refreshed; pending decision]."*

### §4 (Calibration Methods) — line ~1060 area

Add one sentence on the empirical_cumulative semantic:

> *"After a post-submission refactor, the empirical_cumulative threshold
> was inadvertently identical to train_cal (numerically). The camera-ready
> cohort restores the intended semantic: a sweep that maximizes the GA's
> fitness objective on training scores, producing a distinct operating
> point from the F1-optimal train_cal."*

### Table 5 (`tab:ciot-phase`, line ~795)

- Refresh all numbers from `docs/ids_results.md` (NEW cohort, n≥30)
- Update caption: "Cohort statistics from {n} of 112 runs using the
  post-submission methodology refinements (96-bit thermometer, 250n × 100b
  architecture ceiling, order-independent training)."
- Per-genome Pareto rows update with the new best genomes from the NEW
  cohort (r38428 best F1 93.25, r10329 best FPR 0.72, etc.)

### Table 7 (`tab:phase`, line ~985) — Phase Progression

- CIC-IoT-2023 row needs full refresh — the GA Neurons regime flipped from
  ~109n × 48b (submission) to ~218n × 64b (NEW). The Δ row's narrative
  also changes: previously "GA reduces neurons by ~50%", now "GA grows
  neurons by ~36% to higher-bits regime".
- Add footnote: "CIC-IoT-2023 row uses the post-submission methodology
  refinements (see §3 footnote)."

### Table 6 (`tab:ciot-46m`)

- Already flagged in submission caption: "Peak and Search(46M) ... will be
  refreshed in the camera-ready with results from the 250n×100b
  architecture trained with canonical TOP20."
- When the 2× 46M OI flows finish, populate Peak and Search(46M) rows with
  new architectures from the OI-v2 cohort's converged genomes.

### §6 (Discussion) — line ~1026 area

Add a paragraph noting the std halving as a methodology contribution:

> *"The order-independent training correction also tightens cross-seed
> reproducibility: std on F1 across 112 runs drops from ≈0.4 (submission
> cohort) to ≈0.2 (camera-ready cohort). This is independent evidence
> that the training-order bug was contributing to spurious variance in
> the submission's reported error bars."*

### Appendix A (per-genome × threshold breakdowns)

- All 20 CIC-IoT-2023 appendix tables (lines 1622+) need refresh with new
  cohort numbers + caption updates noting "96-bit thermometer, 250n×100b
  ceiling, post-submission methodology refinements".

### What stays unchanged

- §1-2: Introduction, Related Work, Background — no edits needed
- §3 base description — only adds the post-submission paragraph
- §4 base description — only adds the empirical_cumulative sentence
- UNSW-NB15 sections (unless we decide to refresh those too)
- CICIDS2017 sections (same)
- FPGA synthesis sections (until Vivado is rerun on the new architecture)

## Related memories

- `project_oi_cohort_v2_rebuild.md` — the cohort setup
- `project_oi_training_shipped.md` — the OI training fix details
- `project_training_clamped_random_walk.md` — the underlying bug
- `project_raid2026_submitted.md` — paper submission state
- The submission tag `raid-2026-submitted` (commit `42d3ee58`) is the
  truth file for "what reviewers saw"
