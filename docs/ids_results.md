# IDS results — canonical source of truth (S&P-2027 paper track)

Generated 13/08/2026 from `file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro`
(read-only; no flow was created, cancelled, requeued or modified to produce this file).

## Provenance — which tool produced which number

| Section | Producer | Note |
|---|---|---|
| 1. SP100 live cohort 5-tables | `scripts/build_sp_5tables.py --prefix 'SP100-%'` | NEW fallback tool (see below) |
| 2. SP ablation 5-tables | `scripts/build_sp_5tables.py --prefix 'SP-%' --exclude 'SP100-%'` | NEW fallback tool |
| 3. Per-config leaderboards | `scripts/build_ids_leaderboard.py --prefix ...` | NEW fallback tool |
| 4. best_fitness delta tables | `scripts/build_ids_leaderboard.py --fitness-delta` | NEW fallback tool |
| 5. XDS 5-tables (4 cohorts) | `scripts/build_xds_5tables.py --cohort {unsw-temporal,unsw-random,cicids,ciciot}` | EXISTING tool, unmodified |
| 6. Prior config-lock analysis (09/08) | preserved verbatim from the previous revision of this file | hand-written |
| 7. 46M single-flow manual section | preserved verbatim from the previous revision of this file | hand-written |

**Tooling gap, stated explicitly:** `scripts/build_oi_vs_old_report.py` does NOT cover the
SP cohorts. Its auto-detection keys on the `-FIXED-OLD-` rename marker and finds exactly one
cohort (`WSWEEP-T20-96b-C35-250n100b`, OLD=63 / NEW=102); its XDS aliases dispatch only to
`build_xds_5tables.py`, whose regex matches only `XDS-<cohort>-<w>b-W<x>-C35-<n>n<b>b-OI-r<seed>`.
Neither matches `SP100-unswt-quad-16bWb-r25052` or `SP-unswt-abl2s-16bWb-n10-r*`. Sections 1-4
therefore come from two NEW generators written for this report, reading the same tables and
using the same formatting as `build_xds_5tables.py`. Section 5 is the existing tool verbatim.

## Metric contract (do not violate)

- Every number below is a **HELD-OUT** value read from
  `validation_summaries.threshold_metadata` at `validation_point='final'`.
  **No `iterations.best_f1` anywhere** — that is the during-search 5-fold CV metric and
  reporting it is the train-on-eval leak fixed 28/05.
- All 7 threshold modes are reported in order: `train_cal, fixed_05, platt, beta, empirical,
  empirical_cumulative, val_cal`. Empirical modes use `min_bin_size=200`.
- `best_genomes` is used ONLY for the neurons/bits header lines (it dedups by genome_hash and
  is incomplete); all metrics come from `validation_summaries`.

## ⚠️ Protocol split — these two groups are NOT apples-to-apples

| Group | Split | val_cal meaning |
|---|---|---|
| **SP100-\*, SP-\*** (sections 1-4) | `*_3way` (80/10/10) — **Protocol v2** | threshold fitted on the VAL partition, reported on the disjoint TEST partition. Deployable. |
| **XDS-\*** (section 5) | `random` / `temporal` (80/20) — **LEGACY 2-way** | val_cal is F1-optimal **on the report set itself** = oracle. Known-optimistic, reviewer-attackable. |

Never rank an SP row against an XDS row. The leaderboards keep them in separate sections for
this reason.

## Cohort state at generation time

    SP100 (live)        : 54/502 completed | 1 running | 447 queued
    SP-* ablations      : 190/190 completed (85 bin-n30 flows still paused, not counted)
    XDS-*               : complete (legacy)


---

# =====================================================================
# SECTION 1 — SP100 LIVE COHORT (Protocol v2, _3way) — the CURRENT runs
# =====================================================================

# SP100 — live S&P-2027 cohort (Protocol v2, _3way)

    Flows : 54/502 completed | running: 1 | queued: 447 | paused (not counted in target): 0
    Total wall (completed) : 78.8h  |  Avg/run: 88m
    Latest done : 13/08/2026 11:45 UTC (13/08/2026 07:45 ET)
    ETA remaining 448 runs : 09/09/2026 17:08 UTC (09/09/2026 13:08 ET)


## SP100-cicids-quad-96bWa  (11/100 completed)

    dataset=cicids2017 split=random_3way bits=96 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 500n/34b | w(ce/acc/f1/fpr)=0.35/0.3/0.3/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  99.63% |   0.08% |  99.77% | r66758 GA best_acc     empirical_cumulative
    Best F1 (FPR<10%)   |  99.63% |   0.08% |  99.77% | r66758 GA best_acc     empirical_cumulative
    Best F1 (FPR<6%)    |  99.63% |   0.08% |  99.77% | r66758 GA best_acc     empirical_cumulative
    Best F1 (FPR<5%)    |  99.63% |   0.08% |  99.77% | r66758 GA best_acc     empirical_cumulative
    Best F1 (FPR<4%)    |  99.63% |   0.08% |  99.77% | r66758 GA best_acc     empirical_cumulative
    Best F1 (FPR<2%)    |  99.63% |   0.08% |  99.77% | r66758 GA best_acc     empirical_cumulative
    Best FPR (any F1)   |  78.59% |   0.04% |  89.43% | r37040 GS best_acc     fixed_05
    Best FPR (F1>80%)   |  97.05% |   0.04% |  98.20% | r80829 GS best_acc     empirical
    Best FPR (F1>90%)   |  97.05% |   0.04% |  98.20% | r80829 GS best_acc     empirical
    Best Acc (any FPR)  |  99.63% |   0.08% |  99.77% | r66758 GA best_f1      empirical_cumulative

### best_f1  (runs: GS 11 | GA 11)
    Grid Search : 175±89 neurons | 34±1 bits
    GA Neurons  : 139±74 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.30±0.03 99.54±0.05 | 0.28±0.03  0.13±0.05 |99.56±0.02 99.71±0.03
    fixed_05             |97.17±6.16 99.13±0.10 | 0.53±0.18  0.53±0.08 |98.48±3.00 99.45±0.07
    platt                |99.27±0.05 99.42±0.05 | 0.33±0.03  0.26±0.04 |99.54±0.03 99.63±0.03
    beta                 |99.26±0.06 99.48±0.05 | 0.29±0.07  0.20±0.04 |99.53±0.04 99.67±0.03
    empirical            |98.65±0.63 99.41±0.17 | 0.10±0.06  0.10±0.04 |99.16±0.38 99.63±0.10
    empirical_cumulative |99.30±0.03 99.54±0.05 | 0.28±0.03  0.13±0.05 |99.56±0.02 99.71±0.03
    val_cal              |99.30±0.03 99.54±0.05 | 0.29±0.03  0.14±0.05 |99.56±0.02 99.71±0.03

### best_fpr  (runs: GS 11 | GA 11)
    Grid Search : 300±141 neurons | 33±1 bits
    GA Neurons  : 141±82 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |96.14±7.92 99.47±0.05 | 1.04±2.85  0.13±0.05 |97.62±4.87 99.66±0.03
    fixed_05             |94.57±8.70 99.17±0.07 | 4.52±8.00  0.50±0.05 |95.91±7.13 99.47±0.04
    platt                |95.87±7.97 99.40±0.04 | 1.52±2.91  0.28±0.03 |97.42±4.91 99.62±0.03
    beta                 |96.02±8.22 99.42±0.03 | 0.22±0.09  0.24±0.04 |97.96±3.73 99.63±0.02
    empirical            |95.55±8.03 99.37±0.14 | 0.20±0.44  0.09±0.04 |97.66±3.62 99.61±0.09
    empirical_cumulative |96.04±8.23 99.47±0.05 | 0.17±0.06  0.15±0.06 |97.97±3.74 99.66±0.03
    val_cal              |96.14±7.92 99.46±0.05 | 1.07±2.84  0.17±0.05 |97.62±4.87 99.66±0.03

### best_acc  (runs: GS 11 | GA 11)
    Grid Search : 175±89 neurons | 34±1 bits
    GA Neurons  : 139±74 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.30±0.03 99.54±0.05 | 0.28±0.03  0.13±0.05 |99.56±0.02 99.71±0.03
    fixed_05             |97.17±6.16 99.13±0.10 | 0.53±0.18  0.53±0.08 |98.48±3.00 99.45±0.07
    platt                |99.27±0.05 99.42±0.05 | 0.33±0.03  0.26±0.04 |99.54±0.03 99.63±0.03
    beta                 |99.26±0.06 99.48±0.05 | 0.29±0.07  0.20±0.04 |99.53±0.04 99.67±0.03
    empirical            |98.65±0.63 99.41±0.17 | 0.10±0.06  0.10±0.04 |99.16±0.38 99.63±0.10
    empirical_cumulative |99.30±0.03 99.54±0.05 | 0.28±0.03  0.13±0.05 |99.56±0.02 99.71±0.03
    val_cal              |99.30±0.03 99.54±0.05 | 0.29±0.03  0.14±0.05 |99.56±0.02 99.71±0.03

### best_ce  (runs: GS 11 | GA 11)
    Grid Search : 450±71 neurons | 34±0 bits
    GA Neurons  : 192±131 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.27±0.03 99.36±0.03 | 0.29±0.05  0.27±0.04 |99.54±0.02 99.59±0.02
    fixed_05             |98.97±0.03 99.07±0.12 | 0.62±0.03  0.57±0.09 |99.35±0.02 99.41±0.08
    platt                |99.25±0.04 99.28±0.08 | 0.32±0.02  0.36±0.06 |99.53±0.03 99.54±0.05
    beta                 |99.23±0.02 99.33±0.05 | 0.28±0.02  0.28±0.03 |99.52±0.01 99.58±0.03
    empirical            |98.44±0.30 99.05±0.28 | 0.07±0.04  0.14±0.06 |99.03±0.18 99.40±0.17
    empirical_cumulative |99.27±0.03 99.35±0.03 | 0.30±0.06  0.27±0.04 |99.54±0.02 99.59±0.02
    val_cal              |99.27±0.03 99.35±0.03 | 0.32±0.05  0.28±0.04 |99.54±0.02 99.59±0.02

### best_fitness  (runs: GS 11 | GA 11)
    Grid Search : 162±92 neurons | 34±1 bits
    GA Neurons  : 164±102 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.30±0.03 99.53±0.08 | 0.28±0.03  0.14±0.07 |99.56±0.02 99.70±0.05
    fixed_05             |97.17±6.16 99.12±0.11 | 0.53±0.17  0.54±0.08 |98.48±3.00 99.44±0.07
    platt                |99.27±0.05 99.41±0.05 | 0.32±0.03  0.28±0.03 |99.54±0.03 99.63±0.03
    beta                 |99.25±0.06 99.46±0.07 | 0.30±0.07  0.22±0.05 |99.53±0.04 99.66±0.04
    empirical            |98.58±0.62 99.36±0.29 | 0.09±0.05  0.09±0.04 |99.12±0.37 99.60±0.18
    empirical_cumulative |99.30±0.03 99.52±0.08 | 0.28±0.03  0.15±0.08 |99.56±0.02 99.70±0.05
    val_cal              |99.30±0.03 99.52±0.08 | 0.29±0.03  0.15±0.07 |99.56±0.02 99.70±0.05


## SP100-ciciot-quad-96bWc  (11/100 completed)

    dataset=ciciot2023_neto_subsample split=random_3way bits=96 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 250n/100b | w(ce/acc/f1/fpr)=0.7/0.1/0.15/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  93.35% |   7.50% |  96.69% | r79803 GA best_acc     train_cal
    Best F1 (FPR<10%)   |  93.35% |   7.50% |  96.69% | r79803 GA best_acc     train_cal
    Best F1 (FPR<6%)    |  93.18% |   5.29% |  96.53% | r79803 GA best_ce      empirical_cumulative
    Best F1 (FPR<5%)    |  93.08% |   4.91% |  96.46% | r79803 GA best_acc     empirical_cumulative
    Best F1 (FPR<4%)    |  90.35% |   2.04% |  94.72% | r61231 GA best_acc     empirical
    Best F1 (FPR<2%)    |  87.86% |   0.89% |  93.04% | r79803 GA best_ce      fixed_05
    Best FPR (any F1)   |  65.99% |   0.00% |  72.84% | r24530 GA best_ce      empirical
    Best FPR (F1>80%)   |  85.18% |   0.64% |  91.12% | r61231 GA best_acc     fixed_05
    Best FPR (F1>90%)   |  90.35% |   2.04% |  94.72% | r61231 GA best_acc     empirical
    Best Acc (any FPR)  |  93.27% |   9.25% |  96.69% | r79803 GA best_f1      beta

### best_f1  (runs: GS 11 | GA 11)
    Grid Search : 125±96 neurons | 68±24 bits
    GA Neurons  : 172±65 neurons | 62±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.87±0.21 92.89±0.23 |15.46±0.89  8.63±0.72 |95.59±0.13 96.46±0.11
    fixed_05             |80.92±0.59 84.77±1.31 | 1.02±0.13  0.79±0.10 |87.81±0.49 90.81±0.97
    platt                |90.03±0.51 92.88±0.24 |11.14±0.45  8.88±0.71 |94.92±0.31 96.47±0.11
    beta                 |90.54±0.69 92.72±0.30 |17.42±3.66 10.91±1.22 |95.50±0.22 96.45±0.13
    empirical            |86.93±8.46 71.67±6.91 |13.98±7.12  0.20±0.61 |92.23±7.48 78.60±6.23
    empirical_cumulative |90.58±0.31 92.73±0.24 |12.66±2.20  6.65±1.09 |95.31±0.26 96.31±0.12
    val_cal              |90.86±0.22 92.88±0.23 |15.28±1.06  8.49±0.64 |95.58±0.15 96.46±0.11

### best_fpr  (runs: GS 11 | GA 11)
    Grid Search : 107±72 neurons | 68±11 bits
    GA Neurons  : 210±30 neurons | 63±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.70±0.98 92.68±0.35 |14.48±0.74  8.84±0.70 |94.88±0.59 96.35±0.18
    fixed_05             |81.76±0.20 85.93±1.23 | 1.09±0.34  0.88±0.11 |88.51±0.16 91.67±0.89
    platt                |89.67±1.06 92.67±0.37 |12.62±1.82  8.77±0.86 |94.78±0.54 96.35±0.19
    beta                 |89.52±1.31 92.52±0.36 |15.22±2.12 11.07±1.12 |94.79±0.94 96.34±0.17
    empirical            |76.49±9.57 68.17±1.30 | 8.05±11.28  0.01±0.01 |83.34±9.06 75.29±1.43
    empirical_cumulative |89.33±0.87 92.38±0.41 | 9.84±1.78  5.56±0.91 |94.44±0.46 96.08±0.24
    val_cal              |89.69±0.98 92.67±0.36 |14.56±1.29  8.62±0.94 |94.87±0.59 96.34±0.17

### best_acc  (runs: GS 11 | GA 11)
    Grid Search : 117±115 neurons | 64±28 bits
    GA Neurons  : 155±69 neurons | 59±11 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.84±0.17 92.84±0.31 |15.60±0.96  9.02±1.13 |95.59±0.12 96.45±0.14
    fixed_05             |80.86±0.54 84.44±1.73 | 1.02±0.13  0.80±0.12 |87.76±0.45 90.56±1.31
    platt                |89.89±0.44 92.72±0.65 |11.15±0.52  8.83±0.64 |94.84±0.27 96.37±0.36
    beta                 |90.49±0.66 92.63±0.54 |17.55±3.59 11.45±2.47 |95.48±0.21 96.42±0.21
    empirical            |87.00±8.21 74.00±8.94 |14.00±7.14  1.25±3.50 |92.32±7.22 80.67±7.80
    empirical_cumulative |90.55±0.34 92.65±0.34 |13.00±2.02  6.96±1.45 |95.31±0.26 96.28±0.15
    val_cal              |90.84±0.17 92.84±0.31 |15.65±1.06  8.86±1.18 |95.59±0.12 96.44±0.14

### best_ce  (runs: GS 11 | GA 11)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 211±29 neurons | 63±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.24±0.27 92.70±0.32 |15.38±0.96  9.09±0.88 |95.24±0.18 96.37±0.15
    fixed_05             |81.80±0.12 85.86±1.30 | 0.97±0.05  0.87±0.09 |88.53±0.10 91.62±0.94
    platt                |90.16±0.20 92.70±0.32 |12.08±0.20  8.81±0.89 |95.04±0.11 96.37±0.16
    beta                 |90.22±0.30 92.56±0.33 |15.77±0.42 11.08±1.13 |95.25±0.15 96.36±0.15
    empirical            |67.28±1.07 68.30±1.35 | 0.00±0.00  0.01±0.00 |74.29±1.19 75.42±1.50
    empirical_cumulative |89.81±0.25 92.47±0.35 | 9.74±0.75  5.91±0.69 |94.73±0.17 96.14±0.20
    val_cal              |90.25±0.27 92.71±0.31 |15.12±0.89  8.56±0.63 |95.23±0.18 96.36±0.16

### best_fitness  (runs: GS 11 | GA 11)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 211±29 neurons | 63±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.24±0.27 92.70±0.32 |15.38±0.96  9.09±0.88 |95.24±0.18 96.37±0.15
    fixed_05             |81.80±0.12 85.86±1.30 | 0.97±0.05  0.87±0.09 |88.53±0.10 91.62±0.94
    platt                |90.16±0.20 92.70±0.32 |12.08±0.20  8.81±0.89 |95.04±0.11 96.37±0.16
    beta                 |90.22±0.30 92.56±0.33 |15.77±0.42 11.08±1.13 |95.25±0.15 96.36±0.15
    empirical            |67.28±1.07 68.30±1.35 | 0.00±0.00  0.01±0.00 |74.29±1.19 75.42±1.50
    empirical_cumulative |89.81±0.25 92.47±0.35 | 9.74±0.75  5.91±0.69 |94.73±0.17 96.14±0.20
    val_cal              |90.25±0.27 92.71±0.31 |15.12±0.89  8.56±0.63 |95.23±0.18 96.36±0.16


## SP100-unswr-qsr-64bWb  (10/100 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=QSR | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  94.56% |   0.61% |  99.17% | r32732 GA best_acc     val_cal
    Best F1 (FPR<10%)   |  94.56% |   0.61% |  99.17% | r32732 GA best_acc     val_cal
    Best F1 (FPR<6%)    |  94.56% |   0.61% |  99.17% | r32732 GA best_acc     val_cal
    Best F1 (FPR<5%)    |  94.56% |   0.61% |  99.17% | r32732 GA best_acc     val_cal
    Best F1 (FPR<4%)    |  94.56% |   0.61% |  99.17% | r32732 GA best_acc     val_cal
    Best F1 (FPR<2%)    |  94.56% |   0.61% |  99.17% | r32732 GA best_acc     val_cal
    Best FPR (any F1)   |  65.37% |   0.00% |  96.92% | r10596 GS best_ce      empirical
    Best FPR (F1>80%)   |  80.06% |   0.00% |  97.87% | r22224 GA best_ce      empirical
    Best FPR (F1>90%)   |  94.06% |   0.37% |  99.15% | r32732 GA best_acc     empirical_cumulative
    Best Acc (any FPR)  |  94.53% |   0.52% |  99.18% | r22224 GA best_f1      platt

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 350±120 neurons | 34±1 bits
    GA Neurons  : 332±128 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.31±0.05 94.42±0.07 | 0.56±0.03  0.55±0.04 |99.14±0.01 99.16±0.01
    fixed_05             |93.65±0.13 93.66±0.15 | 1.06±0.05  1.05±0.06 |98.95±0.03 98.96±0.03
    platt                |94.29±0.05 94.39±0.08 | 0.53±0.02  0.52±0.02 |99.14±0.01 99.16±0.01
    beta                 |94.30±0.06 94.39±0.08 | 0.55±0.02  0.53±0.02 |99.14±0.01 99.16±0.01
    empirical            |71.78±5.45 73.84±5.93 | 0.00±0.00  0.00±0.00 |97.32±0.37 97.45±0.41
    empirical_cumulative |94.25±0.08 94.30±0.14 | 0.50±0.04  0.48±0.07 |99.15±0.01 99.16±0.01
    val_cal              |94.33±0.08 94.42±0.09 | 0.59±0.05  0.61±0.06 |99.14±0.01 99.15±0.02

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 370±142 neurons | 33±1 bits
    GA Neurons  : 312±139 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.19±0.26 94.33±0.08 | 0.59±0.19  0.55±0.04 |99.12±0.07 99.15±0.01
    fixed_05             |93.59±0.12 93.64±0.13 | 1.08±0.05  1.06±0.05 |98.94±0.03 98.95±0.03
    platt                |94.13±0.32 94.32±0.07 | 0.59±0.18  0.54±0.03 |99.11±0.08 99.15±0.01
    beta                 |94.11±0.51 94.33±0.07 | 0.60±0.16  0.55±0.03 |99.11±0.10 99.15±0.01
    empirical            |74.36±9.34 74.59±6.90 | 0.09±0.26  0.01±0.02 |97.51±0.65 97.51±0.49
    empirical_cumulative |94.12±0.24 94.24±0.11 | 0.57±0.20  0.48±0.05 |99.11±0.07 99.15±0.01
    val_cal              |94.18±0.25 94.30±0.10 | 0.65±0.18  0.56±0.07 |99.11±0.07 99.14±0.02

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 333±71 neurons | 33±1 bits
    GA Neurons  : 332±128 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.32±0.07 94.41±0.07 | 0.55±0.03  0.54±0.04 |99.15±0.01 99.16±0.01
    fixed_05             |93.62±0.13 93.66±0.14 | 1.07±0.05  1.05±0.06 |98.95±0.03 98.96±0.03
    platt                |94.31±0.07 94.39±0.08 | 0.53±0.01  0.52±0.02 |99.15±0.01 99.16±0.01
    beta                 |94.33±0.06 94.40±0.07 | 0.55±0.02  0.53±0.02 |99.15±0.01 99.16±0.01
    empirical            |71.31±4.67 74.18±5.48 | 0.00±0.00  0.00±0.00 |97.28±0.32 97.47±0.38
    empirical_cumulative |94.26±0.10 94.29±0.13 | 0.49±0.05  0.47±0.06 |99.15±0.01 99.16±0.01
    val_cal              |94.29±0.07 94.42±0.09 | 0.59±0.05  0.62±0.04 |99.13±0.02 99.15±0.01

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 290±74 neurons | 34±0 bits
    GA Neurons  : 274±130 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.29±0.05 94.35±0.08 | 0.56±0.04  0.53±0.03 |99.14±0.01 99.15±0.01
    fixed_05             |93.69±0.11 93.70±0.14 | 1.04±0.04  1.04±0.05 |98.96±0.02 98.96±0.03
    platt                |94.28±0.04 94.32±0.06 | 0.54±0.02  0.52±0.01 |99.14±0.00 99.15±0.01
    beta                 |94.31±0.05 94.35±0.06 | 0.56±0.02  0.53±0.01 |99.14±0.01 99.15±0.01
    empirical            |73.37±4.88 77.08±6.18 | 0.00±0.00  0.01±0.01 |97.41±0.31 97.68±0.44
    empirical_cumulative |94.23±0.06 94.31±0.09 | 0.49±0.04  0.51±0.07 |99.15±0.01 99.15±0.01
    val_cal              |94.31±0.04 94.36±0.07 | 0.57±0.03  0.59±0.06 |99.14±0.01 99.14±0.01

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 350±108 neurons | 34±1 bits
    GA Neurons  : 332±128 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.31±0.07 94.42±0.07 | 0.55±0.03  0.55±0.04 |99.14±0.01 99.16±0.01
    fixed_05             |93.63±0.14 93.66±0.15 | 1.07±0.05  1.05±0.06 |98.95±0.03 98.96±0.03
    platt                |94.29±0.06 94.39±0.08 | 0.53±0.02  0.52±0.02 |99.14±0.01 99.16±0.01
    beta                 |94.31±0.06 94.39±0.08 | 0.55±0.02  0.53±0.02 |99.15±0.01 99.16±0.01
    empirical            |72.73±5.12 73.84±5.93 | 0.00±0.00  0.00±0.00 |97.37±0.34 97.45±0.41
    empirical_cumulative |94.26±0.08 94.30±0.14 | 0.49±0.04  0.48±0.07 |99.15±0.01 99.16±0.01
    val_cal              |94.33±0.08 94.42±0.09 | 0.57±0.05  0.61±0.06 |99.14±0.01 99.15±0.02


## SP100-unswr-quad-64bWb  (11/100 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  93.93% |   0.72% |  99.06% | r32732 GA best_f1      train_cal
    Best F1 (FPR<10%)   |  93.93% |   0.72% |  99.06% | r32732 GA best_f1      train_cal
    Best F1 (FPR<6%)    |  93.93% |   0.72% |  99.06% | r32732 GA best_f1      train_cal
    Best F1 (FPR<5%)    |  93.93% |   0.72% |  99.06% | r32732 GA best_f1      train_cal
    Best F1 (FPR<4%)    |  93.93% |   0.72% |  99.06% | r32732 GA best_f1      train_cal
    Best F1 (FPR<2%)    |  93.93% |   0.72% |  99.06% | r32732 GA best_f1      train_cal
    Best FPR (any F1)   |  83.84% |   0.12% |  98.13% | r32732 GA best_fpr     train_cal
    Best FPR (F1>80%)   |  83.84% |   0.12% |  98.13% | r32732 GA best_fpr     train_cal
    Best FPR (F1>90%)   |  92.12% |   0.35% |  98.90% | r32732 GA best_acc     beta
    Best Acc (any FPR)  |  93.93% |   0.72% |  99.06% | r32732 GA best_f1      train_cal

### best_f1  (runs: GS 11 | GA 11)
    Grid Search : 286±168 neurons | 16±3 bits
    GA Neurons  : 179±140 neurons | 14±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.56±0.13 | 1.12±0.00  1.06±0.12 |98.92±0.00 98.94±0.04
    fixed_05             |92.49±1.11 91.83±1.62 | 1.33±0.24  1.47±0.35 |98.72±0.23 98.58±0.34
    platt                |93.29±0.04 93.31±0.09 | 1.12±0.00  1.03±0.18 |98.89±0.01 98.91±0.04
    beta                 |91.94±4.45 91.23±4.35 | 1.05±0.20  0.94±0.25 |98.76±0.42 98.69±0.42
    empirical            |89.82±2.67 91.86±1.86 | 0.92±0.15  0.98±0.12 |98.49±0.29 98.73±0.24
    empirical_cumulative |93.50±0.00 93.56±0.13 | 1.12±0.00  1.06±0.12 |98.92±0.00 98.94±0.04
    val_cal              |93.50±0.00 93.56±0.13 | 1.12±0.00  1.06±0.12 |98.92±0.00 98.94±0.04

### best_fpr  (runs: GS 11 | GA 11)
    Grid Search : 5±0 neurons | 21±11 bits
    GA Neurons  : 29±37 neurons | 14±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |91.57±1.86 88.11±3.71 | 1.35±0.43  1.15±0.79 |98.57±0.38 98.21±0.51
    fixed_05             |90.23±2.30 78.53±11.93 | 1.82±0.51  6.58±5.38 |98.24±0.49 93.59±5.22
    platt                |90.88±2.14 86.65±4.32 | 1.02±0.08  1.07±0.31 |98.59±0.27 98.06±0.53
    beta                 |90.66±2.14 87.10±4.06 | 1.01±0.10  0.54±0.29 |98.56±0.26 98.30±0.47
    empirical            |91.13±1.83 87.46±3.85 | 1.23±0.40  1.21±0.64 |98.54±0.36 98.08±0.58
    empirical_cumulative |91.47±1.99 87.62±3.90 | 1.07±0.07  0.47±0.35 |98.65±0.27 98.40±0.36
    val_cal              |91.57±1.86 88.10±3.71 | 1.34±0.43  1.14±0.79 |98.58±0.38 98.21±0.52

### best_acc  (runs: GS 11 | GA 11)
    Grid Search : 286±168 neurons | 16±3 bits
    GA Neurons  : 154±142 neurons | 15±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.46±0.12 | 1.12±0.00  1.06±0.13 |98.92±0.00 98.93±0.04
    fixed_05             |92.49±1.11 91.04±1.97 | 1.33±0.24  1.65±0.43 |98.72±0.23 98.41±0.42
    platt                |93.29±0.04 93.17±0.22 | 1.12±0.00  0.77±0.33 |98.89±0.01 98.95±0.06
    beta                 |91.94±4.45 92.53±1.08 | 1.05±0.20  0.79±0.34 |98.76±0.42 98.86±0.17
    empirical            |89.82±2.67 92.41±1.21 | 0.92±0.15  0.81±0.26 |98.49±0.29 98.83±0.19
    empirical_cumulative |93.50±0.00 93.38±0.17 | 1.12±0.00  0.83±0.30 |98.92±0.00 98.96±0.05
    val_cal              |93.50±0.00 93.41±0.18 | 1.12±0.00  0.90±0.29 |98.92±0.00 98.95±0.05

### best_ce  (runs: GS 11 | GA 11)
    Grid Search : 438±106 neurons | 12±0 bits
    GA Neurons  : 327±169 neurons | 15±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.47±0.02 93.48±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00
    fixed_05             |92.33±0.14 92.82±0.20 | 1.36±0.03  1.26±0.04 |98.69±0.03 98.79±0.04
    platt                |93.30±0.02 93.26±0.06 | 1.12±0.00  1.11±0.00 |98.89±0.00 98.88±0.01
    beta                 |93.20±0.12 93.26±0.04 | 1.11±0.01  1.12±0.00 |98.88±0.02 98.88±0.01
    empirical            |87.64±2.44 91.22±1.91 | 0.82±0.15  0.95±0.15 |98.25±0.25 98.65±0.23
    empirical_cumulative |93.47±0.02 93.42±0.19 | 1.12±0.00  1.08±0.14 |98.91±0.00 98.92±0.01
    val_cal              |93.47±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00

### best_fitness  (runs: GS 11 | GA 11)
    Grid Search : 286±168 neurons | 16±3 bits
    GA Neurons  : 118±176 neurons | 17±7 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 92.29±2.60 | 1.12±0.00  1.13±0.41 |98.92±0.00 98.75±0.38
    fixed_05             |92.49±1.11 90.35±4.33 | 1.33±0.24  1.83±1.04 |98.72±0.23 98.21±1.06
    platt                |93.29±0.04 91.56±3.60 | 1.12±0.00  1.06±0.19 |98.89±0.01 98.67±0.48
    beta                 |91.94±4.45 90.22±4.71 | 1.05±0.20  0.74±0.34 |98.76±0.42 98.62±0.46
    empirical            |89.82±2.67 90.29±3.08 | 0.92±0.15  1.02±0.34 |98.49±0.29 98.50±0.47
    empirical_cumulative |93.50±0.00 92.09±2.95 | 1.12±0.00  0.89±0.32 |98.92±0.00 98.80±0.29
    val_cal              |93.50±0.00 92.28±2.60 | 1.12±0.00  1.12±0.41 |98.92±0.00 98.75±0.38


## SP100-unswt-quad-16bWb  (11/100 completed)

    dataset=unsw-nb15 split=temporal_3way bits=16 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  89.32% |  10.60% |  89.41% | r35879 GS best_acc     val_cal
    Best F1 (FPR<10%)   |  89.25% |   9.19% |  89.32% | r88120 GA best_ce      val_cal
    Best F1 (FPR<6%)    |  88.44% |   4.60% |  88.45% | r63749 GA best_ce      empirical_cumulative
    Best F1 (FPR<5%)    |  88.44% |   4.60% |  88.45% | r63749 GA best_ce      empirical_cumulative
    Best F1 (FPR<4%)    |  87.68% |   3.34% |  87.68% | r25052 GA best_ce      empirical_cumulative
    Best F1 (FPR<2%)    |  83.40% |   1.91% |  83.44% | r77715 GA best_ce      empirical
    Best FPR (any F1)   |  74.83% |   0.06% |  75.36% | r63749 GA best_fpr     empirical_cumulative
    Best FPR (F1>80%)   |  81.45% |   0.92% |  81.55% | r25052 GA best_ce      empirical
    Best FPR (F1>90%)   |       — |       — |       — | —
    Best Acc (any FPR)  |  89.32% |  10.60% |  89.41% | r35879 GS best_acc     val_cal

### best_f1  (runs: GS 11 | GA 11)
    Grid Search : 69±79 neurons | 31±3 bits
    GA Neurons  : 101±135 neurons | 30±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.85±0.78 84.03±0.77 |29.07±2.28 28.52±2.23 |84.56±0.66 84.71±0.65
    fixed_05             |84.82±1.24 84.28±1.90 |25.92±3.60 27.55±5.29 |85.37±1.07 84.94±1.63
    platt                |86.03±1.46 86.13±1.41 |20.22±3.94 19.94±3.83 |86.35±1.35 86.43±1.30
    beta                 |85.42±2.18 85.63±2.38 |19.60±10.32 18.83±7.95 |85.81±1.90 85.95±2.18
    empirical            |84.77±2.08 85.22±2.16 |15.23±11.05 15.01±9.63 |85.05±2.01 85.46±2.08
    empirical_cumulative |85.53±4.17 85.73±2.42 | 7.09±2.43  8.75±2.91 |85.58±4.10 85.77±2.43
    val_cal              |87.16±2.00 86.96±1.51 |13.61±7.97 13.19±4.88 |87.34±1.76 87.09±1.42

### best_fpr  (runs: GS 11 | GA 11)
    Grid Search : 234±195 neurons | 28±12 bits
    GA Neurons  : 200±232 neurons | 21±15 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |78.20±7.00 77.34±7.86 |38.02±12.39 40.43±13.62 |79.68±5.98 79.10±6.58
    fixed_05             |80.03±6.19 79.45±6.07 |24.23±5.22 28.04±9.48 |80.36±6.33 80.08±5.96
    platt                |80.89±6.99 80.14±7.76 |21.04±5.87 23.93±7.52 |81.10±7.01 80.43±7.72
    beta                 |82.37±6.31 81.40±6.42 |12.55±6.25 16.05±9.30 |82.50±6.35 81.65±6.42
    empirical            |77.18±5.56 78.04±7.52 |17.74±23.87 20.59±24.08 |78.11±4.47 79.09±6.29
    empirical_cumulative |82.67±5.77 82.24±6.13 | 4.08±2.57  3.96±3.43 |82.81±5.63 82.44±5.92
    val_cal              |83.27±6.08 82.93±6.21 | 6.89±5.17  7.98±4.30 |83.40±6.01 83.03±6.16

### best_acc  (runs: GS 11 | GA 11)
    Grid Search : 37±55 neurons | 32±3 bits
    GA Neurons  : 63±53 neurons | 29±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.20±1.68 83.70±1.37 |30.66±4.50 29.26±3.47 |84.02±1.37 84.42±1.15
    fixed_05             |83.75±2.42 83.75±2.10 |27.76±4.27 28.86±5.56 |84.38±2.27 84.48±1.79
    platt                |84.72±2.84 85.39±2.13 |22.44±5.42 20.46±3.55 |85.11±2.67 85.70±2.08
    beta                 |84.00±2.93 84.30±3.27 |22.86±11.28 21.51±10.19 |84.51±2.66 84.75±2.91
    empirical            |84.37±2.35 85.03±2.12 |20.78±11.77 19.48±11.09 |84.83±1.94 85.42±1.74
    empirical_cumulative |83.75±5.04 84.70±3.29 | 7.65±2.99  8.80±3.50 |83.81±4.96 84.75±3.28
    val_cal              |85.74±3.20 86.11±2.48 |18.12±11.51 16.21±8.56 |86.11±2.73 86.35±2.15

### best_ce  (runs: GS 11 | GA 11)
    Grid Search : 383±133 neurons | 33±1 bits
    GA Neurons  : 340±131 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.20±2.08 83.96±0.57 |30.83±5.55 29.03±1.69 |84.05±1.66 84.66±0.47
    fixed_05             |84.15±2.48 84.41±0.67 |27.97±6.97 27.58±1.99 |84.86±1.98 85.03±0.57
    platt                |85.52±3.46 87.26±0.25 |22.35±9.64 18.04±0.67 |86.02±2.88 87.51±0.24
    beta                 |86.42±3.74 88.49±0.41 |18.44±11.94 12.52±1.22 |86.84±3.07 88.60±0.39
    empirical            |79.19±2.63 81.56±2.46 | 7.90±14.74  1.30±0.69 |79.52±2.34 81.67±2.38
    empirical_cumulative |84.95±5.35 87.70±0.48 | 8.73±7.91  4.21±0.65 |84.98±5.31 87.70±0.49
    val_cal              |86.91±3.82 88.89±0.32 |15.21±13.38  9.51±0.94 |87.28±3.14 88.96±0.32

### best_fitness  (runs: GS 11 | GA 11)
    Grid Search : 136±195 neurons | 32±3 bits
    GA Neurons  : 101±134 neurons | 30±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.88±0.78 84.10±0.73 |29.02±2.27 28.29±2.08 |84.58±0.66 84.76±0.62
    fixed_05             |84.78±1.21 84.17±1.97 |26.06±3.49 27.82±5.51 |85.33±1.04 84.84±1.69
    platt                |86.01±1.43 86.08±1.41 |20.28±3.90 19.96±3.84 |86.32±1.33 86.38±1.30
    beta                 |85.49±2.26 86.08±1.89 |19.24±10.51 17.30±5.36 |85.87±1.98 86.31±1.85
    empirical            |84.01±2.98 85.11±2.07 |14.88±11.47 15.73±9.69 |84.32±2.94 85.37±2.00
    empirical_cumulative |85.47±4.14 85.66±2.37 | 7.00±2.48  8.64±2.79 |85.51±4.06 85.69±2.38
    val_cal              |87.17±2.01 86.89±1.50 |13.55±8.01 13.09±4.92 |87.35±1.77 87.01±1.42


---

# =====================================================================
# SECTION 2 — SP-* MEMORY-MODE ABLATION COHORTS (Protocol v2, _3way)
# =====================================================================

# SP ablation cohorts — memory-mode ablation (Protocol v2, _3way)

    Flows : 231/231 completed | running: 0 | queued: 0 | paused (not counted in target): 85
    Total wall (completed) : 652.3h  |  Avg/run: 169m
    Latest done : 10/08/2026 03:46 UTC (09/08/2026 23:46 ET)


## SP-cicids-abl2big-96bWa-n10  (10/10 completed)

    dataset=cicids2017 split=random_3way bits=96 feats=top20 class=binary | mem=BINARY | caps 250n/100b | w(ce/acc/f1/fpr)=0.35/0.3/0.3/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  99.15% |   0.59% |  99.46% | r43932 GA best_acc     train_cal
    Best F1 (FPR<10%)   |  99.15% |   0.59% |  99.46% | r43932 GA best_acc     train_cal
    Best F1 (FPR<6%)    |  99.15% |   0.59% |  99.46% | r43932 GA best_acc     train_cal
    Best F1 (FPR<5%)    |  99.15% |   0.59% |  99.46% | r43932 GA best_acc     train_cal
    Best F1 (FPR<4%)    |  99.15% |   0.59% |  99.46% | r43932 GA best_acc     train_cal
    Best F1 (FPR<2%)    |  99.15% |   0.59% |  99.46% | r43932 GA best_acc     train_cal
    Best FPR (any F1)   |  44.54% |   0.00% |  80.32% | r26177 GS best_fpr     platt
    Best FPR (F1>80%)   |  98.91% |   0.54% |  99.31% | r80829 GA best_fpr     train_cal
    Best FPR (F1>90%)   |  98.91% |   0.54% |  99.31% | r80829 GA best_fpr     train_cal
    Best Acc (any FPR)  |  99.15% |   0.59% |  99.46% | r43932 GA best_f1      train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 120±67 neurons | 97±6 bits
    GA Neurons  : 132±73 neurons | 95±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.04±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.39±0.01 99.45±0.01
    fixed_05             |93.90±0.74 94.51±0.61 | 5.17±0.68  4.61±0.55 |95.84±0.54 96.29±0.44
    platt                |98.88±0.03 99.04±0.06 | 0.82±0.02  0.70±0.05 |99.28±0.02 99.38±0.04
    beta                 |98.97±0.07 98.88±0.37 | 0.60±0.04  0.75±0.36 |99.35±0.05 99.28±0.24
    empirical            |99.00±0.05 99.04±0.05 | 0.61±0.03  0.58±0.01 |99.36±0.03 99.39±0.03
    empirical_cumulative |99.04±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.39±0.01 99.45±0.01
    val_cal              |99.03±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.38±0.01 99.45±0.01

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 150±90 neurons | 99±2 bits
    GA Neurons  : 113±57 neurons | 93±7 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.85±13.11 99.05±0.07 | 5.61±15.87  0.58±0.02 |95.35±12.69 99.40±0.04
    fixed_05             |85.81±24.39 94.98±0.92 |14.97±29.89  4.20±0.82 |87.97±24.00 96.62±0.66
    platt                |93.42±17.17 98.95±0.07 | 0.77±0.27  0.77±0.06 |97.37±5.99 99.33±0.05
    beta                 |94.81±13.09 98.85±0.36 | 5.63±15.86  0.76±0.35 |95.32±12.68 99.27±0.24
    empirical            |90.73±26.10 99.01±0.05 |10.52±31.44  0.57±0.01 |91.38±25.19 99.37±0.03
    empirical_cumulative |93.55±17.22 99.05±0.06 | 0.54±0.20  0.60±0.05 |97.46±6.02 99.40±0.04
    val_cal              |94.85±13.11 99.05±0.06 | 5.62±15.86  0.60±0.05 |95.35±12.69 99.40±0.04

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 120±67 neurons | 97±6 bits
    GA Neurons  : 132±73 neurons | 95±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.04±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.39±0.01 99.45±0.01
    fixed_05             |93.90±0.74 94.51±0.61 | 5.17±0.68  4.61±0.55 |95.84±0.54 96.29±0.44
    platt                |98.88±0.03 99.04±0.06 | 0.82±0.02  0.70±0.05 |99.28±0.02 99.38±0.04
    beta                 |98.97±0.07 98.88±0.37 | 0.60±0.04  0.75±0.36 |99.35±0.05 99.28±0.24
    empirical            |99.00±0.05 99.04±0.05 | 0.61±0.03  0.58±0.01 |99.36±0.03 99.39±0.03
    empirical_cumulative |99.04±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.39±0.01 99.45±0.01
    val_cal              |99.03±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.38±0.01 99.45±0.01

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 205±55 neurons | 82±6 bits
    GA Neurons  : 104±62 neurons | 91±7 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |98.96±0.03 99.05±0.04 | 0.64±0.04  0.65±0.04 |99.34±0.02 99.39±0.02
    fixed_05             |94.69±0.17 96.49±0.55 | 4.45±0.16  2.88±0.47 |96.42±0.13 97.68±0.38
    platt                |98.86±0.02 98.96±0.06 | 0.84±0.01  0.77±0.04 |99.27±0.01 99.33±0.04
    beta                 |98.94±0.02 98.81±0.19 | 0.70±0.02  0.90±0.17 |99.32±0.01 99.24±0.13
    empirical            |98.93±0.05 98.98±0.03 | 0.62±0.04  0.61±0.03 |99.32±0.03 99.35±0.02
    empirical_cumulative |98.96±0.03 99.05±0.04 | 0.65±0.05  0.64±0.03 |99.34±0.02 99.39±0.03
    val_cal              |98.96±0.03 99.05±0.04 | 0.67±0.06  0.65±0.04 |99.34±0.02 99.39±0.03

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 120±67 neurons | 97±6 bits
    GA Neurons  : 132±73 neurons | 95±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.04±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.39±0.01 99.45±0.01
    fixed_05             |93.90±0.74 94.51±0.61 | 5.17±0.68  4.61±0.55 |95.84±0.54 96.29±0.44
    platt                |98.88±0.03 99.04±0.06 | 0.82±0.02  0.70±0.05 |99.28±0.02 99.38±0.04
    beta                 |98.97±0.07 98.88±0.37 | 0.60±0.04  0.75±0.36 |99.35±0.05 99.28±0.24
    empirical            |99.00±0.05 99.04±0.05 | 0.61±0.03  0.58±0.01 |99.36±0.03 99.39±0.03
    empirical_cumulative |99.04±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.39±0.01 99.45±0.01
    val_cal              |99.03±0.02 99.13±0.01 | 0.63±0.03  0.60±0.01 |99.38±0.01 99.45±0.01


## SP-cicids-abl2s-96bWa-n10  (10/10 completed)

    dataset=cicids2017 split=random_3way bits=96 feats=top20 class=binary | mem=BINARY | caps 500n/34b | w(ce/acc/f1/fpr)=0.35/0.3/0.3/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<10%)   |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<6%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<5%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<4%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<2%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best FPR (any F1)   |  98.50% |   0.05% |  99.07% | r84914 GS best_ce      empirical
    Best FPR (F1>80%)   |  98.50% |   0.05% |  99.07% | r84914 GS best_ce      empirical
    Best FPR (F1>90%)   |  98.50% |   0.05% |  99.07% | r84914 GS best_ce      empirical
    Best Acc (any FPR)  |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 370±142 neurons | 31±5 bits
    GA Neurons  : 370±149 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |98.95±0.16 99.08±0.03 | 0.64±0.19  0.62±0.03 |99.33±0.11 99.41±0.02
    fixed_05             |81.79±10.39 83.70±4.53 |18.71±12.66 15.79±5.32 |84.96±10.15 87.31±4.27
    platt                |98.44±0.36 98.78±0.10 | 1.17±0.35  0.91±0.08 |98.99±0.24 99.22±0.07
    beta                 |98.88±0.16 98.96±0.05 | 0.63±0.18  0.58±0.02 |99.29±0.10 99.34±0.03
    empirical            |98.86±0.17 99.04±0.04 | 0.63±0.24  0.61±0.03 |99.28±0.11 99.39±0.02
    empirical_cumulative |98.95±0.16 99.08±0.04 | 0.64±0.19  0.61±0.03 |99.33±0.11 99.41±0.02
    val_cal              |98.94±0.16 99.08±0.03 | 0.68±0.16  0.62±0.03 |99.33±0.10 99.41±0.02

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 388±113 neurons | 32±3 bits
    GA Neurons  : 402±137 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |96.34±8.33 99.03±0.04 | 1.48±2.87  0.60±0.03 |97.73±5.12 99.38±0.02
    fixed_05             |83.07±7.69 86.75±4.65 |16.35±7.81 12.45±5.24 |86.59±6.65 90.00±4.21
    platt                |95.90±8.18 98.68±0.18 | 1.97±2.71  0.99±0.15 |97.44±5.02 99.15±0.12
    beta                 |96.21±8.65 98.95±0.05 | 0.51±0.19  0.59±0.03 |98.09±3.91 99.33±0.03
    empirical            |96.18±8.63 99.02±0.03 | 0.50±0.24  0.60±0.03 |98.07±3.90 99.37±0.02
    empirical_cumulative |96.24±8.66 99.02±0.04 | 0.52±0.21  0.62±0.04 |98.11±3.91 99.38±0.03
    val_cal              |96.34±8.33 99.02±0.04 | 1.51±2.86  0.62±0.04 |97.73±5.12 99.38±0.03

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 390±145 neurons | 32±5 bits
    GA Neurons  : 369±148 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |98.96±0.16 99.08±0.03 | 0.63±0.19  0.62±0.03 |99.34±0.11 99.41±0.02
    fixed_05             |82.53±10.39 83.70±4.52 |17.85±12.71 15.79±5.32 |85.65±10.19 87.32±4.27
    platt                |98.44±0.36 98.77±0.10 | 1.16±0.35  0.91±0.08 |99.00±0.24 99.22±0.07
    beta                 |98.87±0.16 98.96±0.05 | 0.62±0.18  0.58±0.02 |99.28±0.10 99.34±0.03
    empirical            |98.88±0.18 99.04±0.04 | 0.62±0.23  0.61±0.03 |99.29±0.11 99.39±0.02
    empirical_cumulative |98.96±0.16 99.08±0.04 | 0.63±0.19  0.61±0.03 |99.34±0.11 99.41±0.02
    val_cal              |98.96±0.16 99.08±0.03 | 0.66±0.16  0.62±0.03 |99.33±0.11 99.41±0.02

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 288±155 neurons | 34±0 bits
    GA Neurons  : 349±179 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.03±0.17 98.99±0.07 | 0.56±0.21  0.68±0.05 |99.38±0.11 99.36±0.04
    fixed_05             |90.55±5.87 91.41±0.98 | 8.61±5.57  7.51±0.94 |93.04±4.40 93.97±0.75
    platt                |98.63±0.44 98.66±0.21 | 0.96±0.46  1.01±0.17 |99.12±0.29 99.14±0.14
    beta                 |98.97±0.18 98.93±0.03 | 0.54±0.19  0.64±0.07 |99.35±0.12 99.32±0.02
    empirical            |98.76±0.27 98.96±0.05 | 0.50±0.31  0.66±0.05 |99.22±0.16 99.34±0.03
    empirical_cumulative |99.03±0.17 98.99±0.07 | 0.57±0.20  0.68±0.05 |99.38±0.11 99.36±0.04
    val_cal              |99.03±0.17 98.99±0.07 | 0.59±0.17  0.68±0.05 |99.38±0.11 99.36±0.04

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 390±145 neurons | 32±5 bits
    GA Neurons  : 369±148 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |98.96±0.16 99.08±0.03 | 0.63±0.19  0.62±0.03 |99.34±0.11 99.41±0.02
    fixed_05             |82.53±10.39 83.70±4.52 |17.85±12.71 15.79±5.32 |85.65±10.19 87.32±4.27
    platt                |98.44±0.36 98.77±0.10 | 1.16±0.35  0.91±0.08 |99.00±0.24 99.22±0.07
    beta                 |98.87±0.16 98.96±0.05 | 0.62±0.18  0.58±0.02 |99.28±0.10 99.34±0.03
    empirical            |98.88±0.18 99.04±0.04 | 0.62±0.23  0.61±0.03 |99.29±0.11 99.39±0.02
    empirical_cumulative |98.96±0.16 99.08±0.04 | 0.63±0.19  0.61±0.03 |99.34±0.11 99.41±0.02
    val_cal              |98.96±0.16 99.08±0.03 | 0.66±0.16  0.62±0.03 |99.33±0.11 99.41±0.02


## SP-cicids-abl3s-96bWa-n10  (10/10 completed)

    dataset=cicids2017 split=random_3way bits=96 feats=top20 class=binary | mem=TERNARY | caps 500n/34b | w(ce/acc/f1/fpr)=0.35/0.3/0.3/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<10%)   |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<6%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<5%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<4%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best F1 (FPR<2%)    |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal
    Best FPR (any F1)   |  44.54% |   0.00% |  80.32% | r62982 GS best_fpr     train_cal
    Best FPR (F1>80%)   |  98.30% |   0.06% |  98.95% | r41738 GS best_ce      empirical
    Best FPR (F1>90%)   |  98.30% |   0.06% |  98.95% | r41738 GS best_ce      empirical
    Best Acc (any FPR)  |  99.29% |   0.31% |  99.55% | r65122 GS best_ce      train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 333±132 neurons | 30±4 bits
    GA Neurons  : 305±133 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.00±0.10 99.11±0.02 | 0.61±0.15  0.61±0.02 |99.37±0.07 99.43±0.02
    fixed_05             |81.20±9.36 86.69±3.94 |19.26±11.02 12.45±4.30 |84.51±8.82 90.00±3.45
    platt                |98.51±0.34 98.81±0.16 | 1.11±0.34  0.90±0.12 |99.04±0.22 99.24±0.10
    beta                 |98.95±0.12 99.00±0.03 | 0.60±0.13  0.58±0.02 |99.33±0.08 99.37±0.02
    empirical            |98.89±0.12 98.99±0.04 | 0.58±0.20  0.58±0.02 |99.30±0.07 99.36±0.03
    empirical_cumulative |99.00±0.10 99.11±0.02 | 0.60±0.15  0.61±0.02 |99.36±0.07 99.43±0.02
    val_cal              |99.00±0.10 99.11±0.02 | 0.62±0.14  0.61±0.02 |99.36±0.06 99.43±0.02

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 321±190 neurons | 28±13 bits
    GA Neurons  : 307±149 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |86.44±21.33 99.06±0.05 | 6.91±17.37  0.60±0.03 |91.44±14.68 99.40±0.03
    fixed_05             |70.16±28.92 89.24±1.61 |32.62±35.96  9.68±1.65 |73.53±28.83 92.23±1.33
    platt                |85.15±22.87 98.80±0.16 | 1.72±2.83  0.91±0.13 |93.72±8.63 99.23±0.11
    beta                 |84.08±26.71 99.00±0.04 |15.41±32.75  0.60±0.02 |86.18±25.86 99.37±0.03
    empirical            |79.81±34.09 99.00±0.05 |20.24±41.69  0.60±0.03 |82.24±32.92 99.37±0.03
    empirical_cumulative |85.38±23.16 99.06±0.05 | 0.38±0.28  0.60±0.03 |94.32±8.33 99.40±0.03
    val_cal              |86.44±21.33 99.06±0.05 | 6.91±17.37  0.60±0.03 |91.44±14.68 99.40±0.03

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 333±132 neurons | 30±4 bits
    GA Neurons  : 306±134 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.00±0.10 99.11±0.02 | 0.61±0.15  0.61±0.02 |99.37±0.07 99.43±0.02
    fixed_05             |81.20±9.36 86.69±3.94 |19.26±11.02 12.45±4.30 |84.51±8.82 90.00±3.45
    platt                |98.51±0.34 98.81±0.16 | 1.11±0.34  0.89±0.12 |99.04±0.22 99.24±0.10
    beta                 |98.95±0.12 99.00±0.03 | 0.60±0.13  0.58±0.02 |99.33±0.08 99.37±0.02
    empirical            |98.89±0.12 98.99±0.04 | 0.58±0.20  0.58±0.02 |99.30±0.07 99.36±0.03
    empirical_cumulative |99.00±0.10 99.11±0.02 | 0.60±0.15  0.61±0.02 |99.36±0.07 99.43±0.02
    val_cal              |99.00±0.10 99.11±0.02 | 0.62±0.14  0.61±0.02 |99.36±0.06 99.43±0.02

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 450±55 neurons | 34±0 bits
    GA Neurons  : 272±144 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.01±0.14 99.04±0.07 | 0.58±0.15  0.65±0.06 |99.37±0.09 99.39±0.05
    fixed_05             |88.90±5.35 91.31±0.44 |10.24±5.11  7.61±0.42 |91.74±4.04 93.89±0.34
    platt                |98.63±0.35 98.77±0.12 | 0.99±0.37  0.94±0.09 |99.12±0.23 99.21±0.08
    beta                 |98.96±0.15 99.01±0.07 | 0.56±0.15  0.63±0.06 |99.34±0.09 99.37±0.05
    empirical            |98.78±0.24 98.98±0.08 | 0.53±0.25  0.63±0.06 |99.23±0.14 99.35±0.05
    empirical_cumulative |99.02±0.14 99.04±0.07 | 0.59±0.14  0.65±0.06 |99.37±0.09 99.39±0.05
    val_cal              |99.02±0.14 99.04±0.07 | 0.61±0.15  0.65±0.06 |99.37±0.09 99.39±0.05

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 333±132 neurons | 30±4 bits
    GA Neurons  : 306±134 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.00±0.10 99.11±0.02 | 0.61±0.15  0.61±0.02 |99.37±0.07 99.43±0.02
    fixed_05             |81.20±9.36 86.69±3.94 |19.26±11.02 12.45±4.30 |84.51±8.82 90.00±3.45
    platt                |98.51±0.34 98.81±0.16 | 1.11±0.34  0.89±0.12 |99.04±0.22 99.24±0.10
    beta                 |98.95±0.12 99.00±0.03 | 0.60±0.13  0.58±0.02 |99.33±0.08 99.37±0.02
    empirical            |98.89±0.12 98.99±0.04 | 0.58±0.20  0.58±0.02 |99.30±0.07 99.36±0.03
    empirical_cumulative |99.00±0.10 99.11±0.02 | 0.60±0.15  0.61±0.02 |99.36±0.07 99.43±0.02
    val_cal              |99.00±0.10 99.11±0.02 | 0.62±0.14  0.61±0.02 |99.36±0.06 99.43±0.02


## SP-cicids-ablpln-96bWa-n10  (10/10 completed)

    dataset=cicids2017 split=random_3way bits=96 feats=top20 class=binary | mem=PLN | caps 500n/34b | w(ce/acc/f1/fpr)=0.35/0.3/0.3/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  99.35% |   0.22% |  99.59% | r38183 GA best_acc     train_cal
    Best F1 (FPR<10%)   |  99.35% |   0.22% |  99.59% | r38183 GA best_acc     train_cal
    Best F1 (FPR<6%)    |  99.35% |   0.22% |  99.59% | r38183 GA best_acc     train_cal
    Best F1 (FPR<5%)    |  99.35% |   0.22% |  99.59% | r38183 GA best_acc     train_cal
    Best F1 (FPR<4%)    |  99.35% |   0.22% |  99.59% | r38183 GA best_acc     train_cal
    Best F1 (FPR<2%)    |  99.35% |   0.22% |  99.59% | r38183 GA best_acc     train_cal
    Best FPR (any F1)   |  78.71% |   0.02% |  89.48% | r80829 GA best_fpr     fixed_05
    Best FPR (F1>80%)   |  80.90% |   0.03% |  90.33% | r43932 GA best_acc     fixed_05
    Best FPR (F1>90%)   |  97.02% |   0.04% |  98.18% | r80829 GS best_ce      empirical
    Best Acc (any FPR)  |  99.35% |   0.22% |  99.59% | r38183 GA best_f1      train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 411±136 neurons | 34±0 bits
    GA Neurons  : 367±101 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.21±0.09 99.27±0.04 | 0.31±0.14  0.28±0.04 |99.50±0.06 99.54±0.02
    fixed_05             |81.35±6.73 79.01±0.93 | 1.34±3.94  0.04±0.02 |90.45±3.15 89.59±0.36
    platt                |99.12±0.20 99.23±0.03 | 0.46±0.22  0.38±0.04 |99.44±0.13 99.51±0.02
    beta                 |99.11±0.11 99.16±0.05 | 0.47±0.10  0.46±0.05 |99.44±0.07 99.47±0.03
    empirical            |98.27±0.60 98.45±0.34 | 0.17±0.20  0.13±0.08 |98.93±0.36 99.04±0.21
    empirical_cumulative |99.21±0.09 99.27±0.03 | 0.32±0.14  0.28±0.03 |99.50±0.06 99.54±0.02
    val_cal              |99.21±0.09 99.27±0.03 | 0.32±0.14  0.28±0.04 |99.50±0.06 99.54±0.02

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 410±88 neurons | 33±1 bits
    GA Neurons  : 390±79 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.21±0.04 99.27±0.03 | 0.29±0.04  0.27±0.03 |99.50±0.02 99.54±0.02
    fixed_05             |78.13±0.89 78.33±1.23 | 0.03±0.00  0.03±0.01 |89.26±0.33 89.34±0.46
    platt                |99.16±0.05 99.23±0.03 | 0.39±0.03  0.38±0.03 |99.47±0.03 99.51±0.02
    beta                 |99.11±0.07 99.17±0.03 | 0.46±0.05  0.44±0.04 |99.43±0.04 99.48±0.02
    empirical            |98.03±0.49 98.34±0.36 | 0.09±0.06  0.11±0.06 |98.78±0.30 98.97±0.22
    empirical_cumulative |99.21±0.04 99.27±0.04 | 0.28±0.04  0.27±0.03 |99.50±0.02 99.54±0.02
    val_cal              |99.21±0.04 99.27±0.04 | 0.28±0.04  0.28±0.03 |99.50±0.02 99.54±0.02

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 411±136 neurons | 34±0 bits
    GA Neurons  : 367±101 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.21±0.09 99.27±0.04 | 0.31±0.14  0.28±0.04 |99.50±0.06 99.54±0.02
    fixed_05             |81.35±6.73 79.01±0.93 | 1.34±3.94  0.04±0.02 |90.45±3.15 89.59±0.36
    platt                |99.12±0.20 99.23±0.03 | 0.46±0.22  0.38±0.04 |99.44±0.13 99.51±0.02
    beta                 |99.11±0.11 99.16±0.05 | 0.47±0.10  0.46±0.05 |99.44±0.07 99.47±0.03
    empirical            |98.27±0.60 98.45±0.34 | 0.17±0.20  0.13±0.08 |98.93±0.36 99.04±0.21
    empirical_cumulative |99.21±0.09 99.27±0.03 | 0.32±0.14  0.28±0.03 |99.50±0.06 99.54±0.02
    val_cal              |99.21±0.09 99.27±0.03 | 0.32±0.14  0.28±0.04 |99.50±0.06 99.54±0.02

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 267±208 neurons | 34±0 bits
    GA Neurons  : 218±128 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.20±0.13 99.20±0.10 | 0.34±0.14  0.38±0.08 |99.50±0.09 99.50±0.07
    fixed_05             |92.02±9.29 80.18±1.24 | 1.53±3.57  0.09±0.05 |95.63±4.83 90.03±0.48
    platt                |99.10±0.32 99.15±0.10 | 0.46±0.32  0.48±0.09 |99.43±0.21 99.46±0.06
    beta                 |99.14±0.15 99.04±0.12 | 0.40±0.16  0.59±0.11 |99.45±0.09 99.39±0.08
    empirical            |98.40±0.60 98.95±0.18 | 0.18±0.21  0.29±0.12 |99.01±0.36 99.34±0.11
    empirical_cumulative |99.20±0.13 99.21±0.10 | 0.34±0.13  0.37±0.08 |99.50±0.09 99.50±0.07
    val_cal              |99.20±0.13 99.20±0.10 | 0.35±0.13  0.38±0.08 |99.50±0.08 99.50±0.07

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 411±136 neurons | 34±0 bits
    GA Neurons  : 367±101 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.21±0.09 99.27±0.04 | 0.31±0.14  0.28±0.04 |99.50±0.06 99.54±0.02
    fixed_05             |81.35±6.73 79.01±0.93 | 1.34±3.94  0.04±0.02 |90.45±3.15 89.59±0.36
    platt                |99.12±0.20 99.23±0.03 | 0.46±0.22  0.38±0.04 |99.44±0.13 99.51±0.02
    beta                 |99.11±0.11 99.16±0.05 | 0.47±0.10  0.46±0.05 |99.44±0.07 99.47±0.03
    empirical            |98.27±0.60 98.45±0.34 | 0.17±0.20  0.13±0.08 |98.93±0.36 99.04±0.21
    empirical_cumulative |99.21±0.09 99.27±0.03 | 0.32±0.14  0.28±0.03 |99.50±0.06 99.54±0.02
    val_cal              |99.21±0.09 99.27±0.03 | 0.32±0.14  0.28±0.04 |99.50±0.06 99.54±0.02


## SP-cicids-ablqsr-96bWa-n10  (10/10 completed)

    dataset=cicids2017 split=random_3way bits=96 feats=top20 class=binary | mem=QSR | caps 500n/34b | w(ce/acc/f1/fpr)=0.35/0.3/0.3/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  99.45% |   0.23% |  99.65% | r26177 GA best_acc     train_cal
    Best F1 (FPR<10%)   |  99.45% |   0.23% |  99.65% | r26177 GA best_acc     train_cal
    Best F1 (FPR<6%)    |  99.45% |   0.23% |  99.65% | r26177 GA best_acc     train_cal
    Best F1 (FPR<5%)    |  99.45% |   0.23% |  99.65% | r26177 GA best_acc     train_cal
    Best F1 (FPR<4%)    |  99.45% |   0.23% |  99.65% | r26177 GA best_acc     train_cal
    Best F1 (FPR<2%)    |  99.45% |   0.23% |  99.65% | r26177 GA best_acc     train_cal
    Best FPR (any F1)   |  96.69% |   0.04% |  97.99% | r80829 GA best_acc     empirical
    Best FPR (F1>80%)   |  96.69% |   0.04% |  97.99% | r80829 GA best_acc     empirical
    Best FPR (F1>90%)   |  96.69% |   0.04% |  97.99% | r80829 GA best_acc     empirical
    Best Acc (any FPR)  |  99.45% |   0.23% |  99.65% | r26177 GA best_acc     train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 289±105 neurons | 34±1 bits
    GA Neurons  : 261±168 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.24±0.03 99.34±0.06 | 0.25±0.05  0.22±0.04 |99.52±0.02 99.58±0.04
    fixed_05             |99.08±0.07 99.24±0.08 | 0.46±0.09  0.37±0.05 |99.42±0.05 99.52±0.05
    platt                |99.21±0.03 99.32±0.06 | 0.32±0.01  0.28±0.02 |99.50±0.02 99.57±0.04
    beta                 |99.22±0.02 99.33±0.06 | 0.29±0.01  0.27±0.03 |99.50±0.01 99.57±0.04
    empirical            |97.44±0.88 97.76±1.12 | 0.05±0.01  0.05±0.02 |98.43±0.52 98.63±0.67
    empirical_cumulative |99.24±0.03 99.33±0.06 | 0.24±0.05  0.24±0.05 |99.52±0.02 99.58±0.04
    val_cal              |99.24±0.04 99.33±0.06 | 0.28±0.09  0.25±0.05 |99.52±0.03 99.58±0.04

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 201±155 neurons | 32±4 bits
    GA Neurons  : 258±165 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |98.88±0.84 99.33±0.05 | 0.46±0.63  0.21±0.06 |99.28±0.55 99.58±0.03
    fixed_05             |98.68±0.84 99.24±0.07 | 0.75±0.61  0.39±0.06 |99.15±0.55 99.52±0.05
    platt                |98.85±0.83 99.30±0.05 | 0.52±0.61  0.30±0.03 |99.27±0.54 99.56±0.03
    beta                 |98.86±0.83 99.31±0.06 | 0.50±0.62  0.28±0.04 |99.28±0.54 99.56±0.04
    empirical            |97.50±1.00 97.54±1.34 | 0.32±0.69  0.05±0.02 |98.46±0.61 98.50±0.80
    empirical_cumulative |98.88±0.84 99.33±0.06 | 0.46±0.63  0.23±0.06 |99.29±0.55 99.57±0.04
    val_cal              |98.87±0.84 99.32±0.06 | 0.47±0.63  0.25±0.06 |99.28±0.55 99.57±0.04

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 300±122 neurons | 34±1 bits
    GA Neurons  : 261±168 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.23±0.03 99.34±0.06 | 0.26±0.07  0.22±0.04 |99.51±0.02 99.58±0.04
    fixed_05             |99.10±0.07 99.24±0.08 | 0.43±0.08  0.37±0.05 |99.43±0.05 99.52±0.05
    platt                |99.20±0.02 99.32±0.06 | 0.32±0.01  0.28±0.02 |99.50±0.01 99.57±0.04
    beta                 |99.22±0.02 99.33±0.06 | 0.29±0.01  0.27±0.03 |99.50±0.01 99.57±0.04
    empirical            |97.11±1.10 97.77±1.13 | 0.05±0.01  0.05±0.02 |98.24±0.65 98.63±0.68
    empirical_cumulative |99.23±0.03 99.33±0.06 | 0.25±0.07  0.24±0.05 |99.51±0.02 99.58±0.04
    val_cal              |99.23±0.04 99.33±0.06 | 0.29±0.09  0.25±0.05 |99.51±0.03 99.58±0.04

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 163±105 neurons | 34±0 bits
    GA Neurons  : 182±115 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.24±0.09 99.29±0.08 | 0.25±0.04  0.25±0.06 |99.52±0.06 99.55±0.05
    fixed_05             |99.00±0.15 99.12±0.14 | 0.53±0.13  0.50±0.11 |99.36±0.10 99.44±0.09
    platt                |99.16±0.20 99.22±0.14 | 0.37±0.17  0.36±0.07 |99.47±0.13 99.51±0.09
    beta                 |98.80±1.36 99.26±0.11 | 0.27±0.05  0.32±0.05 |99.26±0.80 99.53±0.07
    empirical            |98.19±0.87 98.42±0.73 | 0.09±0.07  0.08±0.05 |98.88±0.53 99.02±0.44
    empirical_cumulative |99.24±0.09 99.29±0.08 | 0.25±0.04  0.26±0.05 |99.52±0.06 99.55±0.05
    val_cal              |99.24±0.09 99.29±0.08 | 0.25±0.04  0.26±0.05 |99.52±0.06 99.55±0.05

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 275±104 neurons | 34±1 bits
    GA Neurons  : 261±168 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.24±0.03 99.34±0.06 | 0.25±0.05  0.22±0.04 |99.52±0.02 99.58±0.04
    fixed_05             |99.08±0.07 99.24±0.08 | 0.46±0.09  0.37±0.05 |99.42±0.05 99.52±0.05
    platt                |99.21±0.03 99.32±0.06 | 0.32±0.01  0.28±0.02 |99.50±0.02 99.57±0.04
    beta                 |99.22±0.02 99.33±0.06 | 0.29±0.01  0.27±0.03 |99.50±0.01 99.57±0.04
    empirical            |97.44±0.88 97.77±1.13 | 0.05±0.01  0.05±0.02 |98.43±0.52 98.63±0.68
    empirical_cumulative |99.24±0.03 99.33±0.06 | 0.24±0.05  0.24±0.05 |99.52±0.02 99.58±0.04
    val_cal              |99.24±0.04 99.33±0.06 | 0.28±0.09  0.25±0.05 |99.52±0.03 99.58±0.04


## SP-cicids-bin-96bWa-n30  (10/10 completed)

    dataset=cicids2017 split=random_3way bits=96 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 500n/34b | w(ce/acc/f1/fpr)=0.35/0.3/0.3/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  99.64% |   0.08% |  99.77% | r84914 GA best_acc     train_cal
    Best F1 (FPR<10%)   |  99.64% |   0.08% |  99.77% | r84914 GA best_acc     train_cal
    Best F1 (FPR<6%)    |  99.64% |   0.08% |  99.77% | r84914 GA best_acc     train_cal
    Best F1 (FPR<5%)    |  99.64% |   0.08% |  99.77% | r84914 GA best_acc     train_cal
    Best F1 (FPR<4%)    |  99.64% |   0.08% |  99.77% | r84914 GA best_acc     train_cal
    Best F1 (FPR<2%)    |  99.64% |   0.08% |  99.77% | r84914 GA best_acc     train_cal
    Best FPR (any F1)   |  98.14% |   0.04% |  98.85% | r26177 GS best_acc     empirical
    Best FPR (F1>80%)   |  98.14% |   0.04% |  98.85% | r26177 GS best_acc     empirical
    Best FPR (F1>90%)   |  98.14% |   0.04% |  98.85% | r26177 GS best_acc     empirical
    Best Acc (any FPR)  |  99.64% |   0.08% |  99.77% | r84914 GA best_acc     train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 210±129 neurons | 34±1 bits
    GA Neurons  : 137±74 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.02 99.59±0.04 | 0.29±0.03  0.09±0.02 |99.56±0.01 99.74±0.02
    fixed_05             |99.00±0.06 99.14±0.10 | 0.62±0.05  0.51±0.08 |99.36±0.04 99.46±0.06
    platt                |99.30±0.03 99.46±0.08 | 0.31±0.01  0.23±0.05 |99.55±0.02 99.66±0.05
    beta                 |99.27±0.02 99.54±0.08 | 0.27±0.01  0.14±0.07 |99.54±0.01 99.71±0.05
    empirical            |98.70±0.31 99.52±0.08 | 0.10±0.05  0.06±0.01 |99.19±0.19 99.70±0.05
    empirical_cumulative |99.31±0.02 99.59±0.04 | 0.29±0.03  0.08±0.01 |99.56±0.01 99.74±0.02
    val_cal              |99.31±0.02 99.59±0.04 | 0.30±0.02  0.09±0.02 |99.56±0.01 99.74±0.02

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 122±111 neurons | 24±10 bits
    GA Neurons  : 138±73 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |95.48±8.18 99.46±0.08 | 1.27±2.98  0.11±0.08 |97.21±5.03 99.66±0.05
    fixed_05             |93.45±8.86 99.13±0.08 | 5.42±8.17  0.53±0.06 |95.10±7.29 99.45±0.05
    platt                |95.19±8.21 99.36±0.06 | 1.78±3.00  0.31±0.04 |96.99±5.06 99.59±0.04
    beta                 |95.31±8.50 99.38±0.07 | 0.21±0.11  0.27±0.07 |97.57±3.85 99.61±0.05
    empirical            |94.96±8.32 99.39±0.14 | 0.38±0.65  0.06±0.02 |97.33±3.74 99.62±0.09
    empirical_cumulative |95.33±8.50 99.46±0.08 | 0.15±0.07  0.14±0.09 |97.59±3.86 99.66±0.05
    val_cal              |95.47±8.18 99.46±0.08 | 1.30±2.97  0.15±0.10 |97.21±5.03 99.66±0.05

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 210±129 neurons | 34±1 bits
    GA Neurons  : 137±74 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.02 99.59±0.04 | 0.29±0.03  0.09±0.02 |99.56±0.01 99.74±0.02
    fixed_05             |99.00±0.06 99.14±0.10 | 0.62±0.05  0.51±0.08 |99.36±0.04 99.45±0.06
    platt                |99.30±0.03 99.46±0.08 | 0.31±0.01  0.23±0.05 |99.55±0.02 99.66±0.05
    beta                 |99.27±0.02 99.54±0.08 | 0.27±0.01  0.14±0.07 |99.54±0.01 99.71±0.05
    empirical            |98.70±0.31 99.52±0.08 | 0.10±0.05  0.06±0.01 |99.19±0.19 99.70±0.05
    empirical_cumulative |99.31±0.02 99.59±0.04 | 0.29±0.03  0.08±0.01 |99.56±0.01 99.74±0.02
    val_cal              |99.31±0.02 99.59±0.04 | 0.30±0.02  0.09±0.02 |99.56±0.01 99.74±0.02

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 300±149 neurons | 34±0 bits
    GA Neurons  : 126±63 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.27±0.02 99.33±0.07 | 0.27±0.05  0.27±0.03 |99.54±0.01 99.58±0.05
    fixed_05             |98.97±0.04 99.07±0.11 | 0.62±0.03  0.58±0.08 |99.35±0.02 99.41±0.07
    platt                |99.24±0.04 99.26±0.07 | 0.33±0.01  0.38±0.04 |99.52±0.02 99.53±0.04
    beta                 |99.23±0.02 99.30±0.08 | 0.29±0.01  0.28±0.03 |99.52±0.01 99.56±0.05
    empirical            |98.47±0.32 99.15±0.17 | 0.08±0.05  0.22±0.12 |99.05±0.20 99.46±0.10
    empirical_cumulative |99.27±0.02 99.33±0.08 | 0.29±0.06  0.30±0.07 |99.54±0.01 99.57±0.05
    val_cal              |99.27±0.02 99.33±0.08 | 0.31±0.06  0.30±0.07 |99.54±0.01 99.57±0.05

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 250±127 neurons | 34±1 bits
    GA Neurons  : 137±74 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.30±0.02 99.59±0.04 | 0.29±0.03  0.09±0.02 |99.56±0.01 99.74±0.02
    fixed_05             |99.00±0.06 99.14±0.10 | 0.62±0.05  0.51±0.08 |99.36±0.04 99.45±0.06
    platt                |99.30±0.03 99.46±0.08 | 0.31±0.01  0.23±0.05 |99.55±0.02 99.66±0.05
    beta                 |99.26±0.02 99.54±0.08 | 0.27±0.01  0.14±0.07 |99.53±0.01 99.71±0.05
    empirical            |98.57±0.28 99.52±0.08 | 0.08±0.05  0.06±0.01 |99.11±0.17 99.70±0.05
    empirical_cumulative |99.30±0.02 99.59±0.04 | 0.31±0.02  0.08±0.01 |99.56±0.01 99.74±0.02
    val_cal              |99.30±0.02 99.59±0.04 | 0.31±0.02  0.09±0.02 |99.56±0.01 99.74±0.02


## SP-ciciot-abl2s-96bWc-n10  (10/10 completed)

    dataset=ciciot2023_neto_subsample split=random_3way bits=96 feats=top20 class=binary | mem=BINARY | caps 250n/100b | w(ce/acc/f1/fpr)=0.7/0.1/0.15/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  90.91% |  15.70% |  95.63% | r57826 GS best_f1      empirical_cumulative
    Best F1 (FPR<10%)   |  89.46% |   8.81% |  94.48% | r61231 GS best_ce      empirical_cumulative
    Best F1 (FPR<6%)    |  81.74% |   1.04% |  88.49% | r98273 GS best_ce      fixed_05
    Best F1 (FPR<5%)    |  81.74% |   1.04% |  88.49% | r98273 GS best_ce      fixed_05
    Best F1 (FPR<4%)    |  81.74% |   1.04% |  88.49% | r98273 GS best_ce      fixed_05
    Best F1 (FPR<2%)    |  81.74% |   1.04% |  88.49% | r98273 GS best_ce      fixed_05
    Best FPR (any F1)   |  12.27% |   0.00% |  13.99% | r78637 GS best_fpr     train_cal
    Best FPR (F1>80%)   |  81.70% |   1.00% |  88.45% | r61231 GS best_ce      fixed_05
    Best FPR (F1>90%)   |  90.19% |  12.11% |  95.06% | r98273 GS best_ce      platt
    Best Acc (any FPR)  |  90.78% |  17.96% |  95.65% | r57826 GS best_f1      beta

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 175±106 neurons | 16±0 bits
    GA Neurons  : 103±82 neurons | 30±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |71.83±6.83 77.69±1.49 |37.73±15.86 45.15±8.69 |84.42±5.28 90.19±1.74
    fixed_05             |50.18±10.47 53.77±11.63 |89.39±31.15 90.25±15.60 |85.93±0.44 87.14±2.11
    platt                |55.38±14.00 72.19±6.38 |84.97±27.67 60.44±12.12 |86.76±3.18 89.43±1.73
    beta                 |53.04±15.17 68.40±12.31 |85.00±31.78 61.37±23.28 |86.53±3.50 88.31±1.89
    empirical            |52.82±13.35 75.75±3.56 |89.44±24.36 53.22±7.72 |86.54±3.22 90.22±1.31
    empirical_cumulative |71.43±7.10 77.12±1.79 |32.20±19.37 39.80±12.96 |83.08±6.62 89.04±2.68
    val_cal              |71.82±6.84 77.70±1.49 |37.21±16.18 44.83±8.69 |84.32±5.38 90.15±1.74

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 5±0 neurons | 4±0 bits
    GA Neurons  : 219±29 neurons | 53±20 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |49.46±25.73 74.35±2.86 |20.27±14.11 34.34±4.74 |57.71±30.22 86.02±2.63
    fixed_05             |46.73±0.43 59.81±9.27 |99.35±0.57 81.42±13.92 |85.62±0.36 87.13±1.63
    platt                |47.15±0.83 67.72±8.14 |98.84±1.03 66.96±13.62 |85.51±0.37 87.93±2.03
    beta                 |46.24±0.00 65.25±11.34 |100.00±0.00 70.80±18.18 |86.01±0.00 88.04±1.72
    empirical            |48.82±2.05 67.51±9.86 |96.32±2.98 66.00±17.04 |84.80±0.93 87.98±1.94
    empirical_cumulative |49.08±25.48 73.40±2.69 |12.45±8.90 19.23±2.78 |55.96±29.08 83.06±2.84
    val_cal              |49.45±25.72 74.43±2.98 |19.80±13.83 35.61±6.80 |57.61±30.16 86.25±3.04

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 135±123 neurons | 12±7 bits
    GA Neurons  : 102±83 neurons | 26±11 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |54.88±11.49 71.65±9.99 |88.05±16.33 63.99±15.85 |87.11±1.50 90.30±1.78
    fixed_05             |46.24±0.00 55.18±14.28 |100.00±0.00 87.67±20.08 |86.01±0.00 87.60±2.58
    platt                |47.66±14.98 70.10±9.73 |84.11±31.43 66.64±15.43 |79.46±23.04 89.88±1.73
    beta                 |46.24±0.00 58.26±15.08 |100.00±0.00 80.21±25.92 |86.01±0.00 87.60±2.11
    empirical            |50.35±8.23 71.44±9.90 |94.62±11.45 64.10±15.96 |86.51±1.06 90.20±1.69
    empirical_cumulative |51.48±17.68 71.58±9.95 |78.05±31.64 62.82±16.97 |79.91±23.21 90.08±1.70
    val_cal              |51.48±17.68 71.66±10.00 |78.05±31.64 63.53±16.25 |79.91±23.21 90.23±1.74

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : 229±26 neurons | 63±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |72.69±9.27 76.11±1.01 |27.19±6.55 34.23±3.50 |82.89±6.62 87.52±0.94
    fixed_05             |54.99±14.10 66.92±4.54 |78.21±40.69 71.19±7.60 |86.02±1.31 88.28±1.09
    platt                |59.68±16.42 72.30±3.09 |76.19±34.12 58.49±5.68 |87.05±4.22 88.85±0.98
    beta                 |56.46±18.26 71.77±3.72 |81.20±34.86 59.84±7.17 |87.88±3.83 88.82±1.02
    empirical            |57.17±5.74 72.79±2.71 |69.96±37.21 57.04±5.23 |82.32±4.94 88.89±0.96
    empirical_cumulative |72.06±9.40 75.18±0.89 |15.66±3.38 22.25±3.41 |80.55±7.57 85.16±0.90
    val_cal              |72.68±9.26 76.12±1.01 |25.66±5.42 34.92±4.53 |82.61±6.78 87.62±1.04

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : 229±26 neurons | 63±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |72.69±9.27 76.11±1.01 |27.19±6.55 34.23±3.50 |82.89±6.62 87.52±0.94
    fixed_05             |54.99±14.10 66.92±4.54 |78.21±40.69 71.19±7.60 |86.02±1.31 88.28±1.09
    platt                |59.68±16.42 72.30±3.09 |76.19±34.12 58.49±5.68 |87.05±4.22 88.85±0.98
    beta                 |56.46±18.26 71.77±3.72 |81.20±34.86 59.84±7.17 |87.88±3.83 88.82±1.02
    empirical            |57.17±5.74 72.79±2.71 |69.96±37.21 57.04±5.23 |82.32±4.94 88.89±0.96
    empirical_cumulative |72.06±9.40 75.18±0.89 |15.66±3.38 22.25±3.41 |80.55±7.57 85.16±0.90
    val_cal              |72.68±9.26 76.12±1.01 |25.66±5.42 34.92±4.53 |82.61±6.78 87.62±1.04


## SP-ciciot-abl3s-96bWc-n10  (10/10 completed)

    dataset=ciciot2023_neto_subsample split=random_3way bits=96 feats=top20 class=binary | mem=TERNARY | caps 250n/100b | w(ce/acc/f1/fpr)=0.7/0.1/0.15/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  90.87% |  15.72% |  95.61% | r29083 GS best_f1      train_cal
    Best F1 (FPR<10%)   |  89.46% |   8.81% |  94.48% | r61231 GS best_ce      empirical_cumulative
    Best F1 (FPR<6%)    |  81.93% |   0.85% |  88.63% | r29083 GS best_f1      fixed_05
    Best F1 (FPR<5%)    |  81.93% |   0.85% |  88.63% | r29083 GS best_f1      fixed_05
    Best F1 (FPR<4%)    |  81.93% |   0.85% |  88.63% | r29083 GS best_f1      fixed_05
    Best F1 (FPR<2%)    |  81.93% |   0.85% |  88.63% | r29083 GS best_f1      fixed_05
    Best FPR (any F1)   |  66.01% |   0.00% |  72.86% | r47707 GS best_ce      empirical
    Best FPR (F1>80%)   |  81.93% |   0.85% |  88.63% | r29083 GS best_f1      fixed_05
    Best FPR (F1>90%)   |  90.42% |  10.72% |  95.13% | r29083 GS best_f1      empirical_cumulative
    Best Acc (any FPR)  |  90.86% |  15.96% |  95.61% | r29083 GS best_f1      val_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 150±0 neurons | 64±0 bits
    GA Neurons  : 138±82 neurons | 42±19 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |75.11±5.70 81.33±1.28 |30.77±7.88 36.26±6.07 |86.07±3.57 91.46±0.77
    fixed_05             |50.51±11.19 51.69±7.71 |89.37±31.16 93.82±9.02 |86.36±0.83 86.86±1.24
    platt                |60.28±12.41 78.86±3.91 |78.07±25.12 48.14±8.75 |87.15±2.91 91.45±1.08
    beta                 |57.61±13.66 73.21±14.29 |82.28±25.64 53.72±24.69 |87.25±2.96 90.19±2.43
    empirical            |55.99±7.27 78.19±6.88 |80.80±29.29 48.18±14.71 |85.36±3.00 91.33±1.48
    empirical_cumulative |74.97±5.60 80.81±1.48 |27.54±9.93 31.79±8.38 |85.54±3.78 90.64±1.28
    val_cal              |75.11±5.69 81.34±1.29 |30.79±7.83 36.31±6.04 |86.07±3.57 91.47±0.78

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : 165±96 neurons | 58±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |69.17±1.75 79.57±1.41 |30.91±5.88 35.08±7.24 |80.83±1.78 90.19±1.03
    fixed_05             |46.38±0.08 55.50±6.34 |99.86±0.09 89.75±7.67 |86.01±0.04 87.43±1.06
    platt                |53.24±2.85 77.77±2.31 |91.09±3.74 48.67±5.33 |85.47±0.33 90.77±0.80
    beta                 |48.08±2.58 75.01±10.36 |97.93±3.02 51.76±17.63 |86.00±0.08 90.26±1.69
    empirical            |52.20±3.52 77.45±3.04 |92.29±5.08 49.37±8.63 |85.49±0.42 90.76±0.78
    empirical_cumulative |68.44±1.80 78.87±1.59 |17.57±2.79 23.84±6.08 |77.91±2.18 88.36±1.45
    val_cal              |69.17±1.75 79.56±1.39 |30.48±6.09 35.66±7.11 |80.76±1.78 90.24±1.10

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 116±92 neurons | 31±33 bits
    GA Neurons  : 103±73 neurons | 33±21 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |65.37±6.09 79.78±1.56 |72.87±11.89 46.01±6.84 |88.04±1.02 91.66±0.67
    fixed_05             |46.51±0.59 51.73±9.45 |99.73±0.58 93.42±11.54 |86.04±0.07 86.91±1.57
    platt                |55.22±7.88 77.52±3.95 |89.06±10.26 52.78±9.04 |86.89±1.37 91.32±1.02
    beta                 |50.13±8.19 64.12±16.23 |90.78±19.88 70.01±28.53 |85.27±1.77 88.61±2.48
    empirical            |52.09±7.51 73.21±9.63 |92.79±10.69 60.53±17.54 |86.62±1.01 90.57±1.99
    empirical_cumulative |65.37±6.09 79.63±1.58 |72.87±11.89 44.60±9.01 |88.04±1.02 91.40±1.03
    val_cal              |65.39±6.06 79.78±1.56 |72.77±11.71 46.01±6.84 |87.95±1.22 91.66±0.67

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : 226±35 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |76.30±9.72 80.37±1.00 |26.53±7.96 33.00±2.96 |85.96±6.65 90.47±0.68
    fixed_05             |57.76±16.57 63.26±5.32 |69.29±47.13 80.05±7.34 |86.57±1.37 88.77±1.02
    platt                |67.14±16.64 79.17±1.67 |62.78±35.71 45.45±3.31 |88.55±4.52 91.18±0.63
    beta                 |65.28±18.29 79.47±1.54 |66.75±36.08 43.60±3.03 |88.94±4.31 91.13±0.62
    empirical            |60.68±5.36 79.72±1.55 |57.94±40.38 39.48±5.30 |81.79±6.16 90.80±0.69
    empirical_cumulative |75.69±9.89 79.68±1.11 |15.37±3.97 24.48±3.79 |83.85±7.73 89.05±0.94
    val_cal              |76.29±9.71 80.38±1.00 |24.72±6.34 32.65±3.80 |85.65±6.85 90.44±0.78

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : 226±35 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |76.30±9.72 80.37±1.00 |26.53±7.96 33.00±2.96 |85.96±6.65 90.47±0.68
    fixed_05             |57.76±16.57 63.26±5.32 |69.29±47.13 80.05±7.34 |86.57±1.37 88.77±1.02
    platt                |67.14±16.64 79.17±1.67 |62.78±35.71 45.45±3.31 |88.55±4.52 91.18±0.63
    beta                 |65.28±18.29 79.47±1.54 |66.75±36.08 43.60±3.03 |88.94±4.31 91.13±0.62
    empirical            |60.68±5.36 79.72±1.55 |57.94±40.38 39.48±5.30 |81.79±6.16 90.80±0.69
    empirical_cumulative |75.69±9.89 79.68±1.11 |15.37±3.97 24.48±3.79 |83.85±7.73 89.05±0.94
    val_cal              |76.29±9.71 80.38±1.00 |24.72±6.34 32.65±3.80 |85.65±6.85 90.44±0.78


## SP-ciciot-ablpln-96bWc-n10  (10/10 completed)

    dataset=ciciot2023_neto_subsample split=random_3way bits=96 feats=top20 class=binary | mem=PLN | caps 250n/100b | w(ce/acc/f1/fpr)=0.7/0.1/0.15/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  92.65% |  12.75% |  96.47% | r98273 GA best_ce      train_cal
    Best F1 (FPR<10%)   |  92.25% |   9.87% |  96.15% | r57826 GA best_f1      empirical_cumulative
    Best F1 (FPR<6%)    |  82.04% |   0.92% |  88.73% | r56462 GS best_ce      fixed_05
    Best F1 (FPR<5%)    |  82.04% |   0.92% |  88.73% | r56462 GS best_ce      fixed_05
    Best F1 (FPR<4%)    |  82.04% |   0.92% |  88.73% | r56462 GS best_ce      fixed_05
    Best F1 (FPR<2%)    |  82.04% |   0.92% |  88.73% | r56462 GS best_ce      fixed_05
    Best FPR (any F1)   |  66.01% |   0.00% |  72.86% | r47707 GS best_ce      empirical
    Best FPR (F1>80%)   |  81.93% |   0.85% |  88.63% | r29083 GS best_acc     fixed_05
    Best FPR (F1>90%)   |  92.13% |   9.09% |  96.06% | r57826 GA best_fpr     empirical_cumulative
    Best Acc (any FPR)  |  92.62% |  13.70% |  96.48% | r98273 GA best_ce      empirical

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 244±18 neurons | 64±0 bits
    GA Neurons  : 237±18 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |85.48±8.22 91.91±0.58 |19.47±6.27 13.67±1.27 |92.15±5.65 96.10±0.27
    fixed_05             |62.16±16.94 49.84±1.16 |20.48±41.04  1.20±0.22 |73.75±17.91 53.09±1.46
    platt                |82.24±13.84 91.15±0.59 |33.07±28.92 20.15±1.29 |92.82±3.83 95.95±0.26
    beta                 |81.02±15.12 90.22±0.68 |37.95±29.27 24.35±1.65 |92.85±3.58 95.65±0.28
    empirical            |73.74±12.95 91.81±0.55 |24.86±31.53 15.28±1.45 |84.25±9.90 96.10±0.26
    empirical_cumulative |85.08±8.40 91.64±0.55 |12.74±3.19 10.73±0.96 |91.23±6.66 95.84±0.29
    val_cal              |85.48±8.23 91.91±0.58 |18.43±4.56 13.64±1.20 |92.01±5.89 96.10±0.28

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 250±0 neurons | 92±13 bits
    GA Neurons  : 237±17 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |88.10±0.76 91.86±0.54 |17.25±0.82 13.62±0.90 |94.08±0.41 96.07±0.27
    fixed_05             |52.74±10.19 49.77±1.16 | 1.03±0.06  1.20±0.27 |56.25±11.33 53.00±1.47
    platt                |86.84±1.22 91.11±0.58 |27.20±5.35 20.20±1.23 |93.99±0.40 95.93±0.26
    beta                 |85.22±1.75 90.15±0.61 |34.12±6.43 24.49±1.43 |93.60±0.57 95.62±0.25
    empirical            |85.41±6.85 91.72±0.53 |19.88±7.16 15.59±1.42 |91.94±6.74 96.07±0.26
    empirical_cumulative |87.76±0.82 91.60±0.54 |12.90±1.11 10.53±0.87 |93.63±0.46 95.81±0.28
    val_cal              |88.08±0.77 91.84±0.53 |17.68±1.02 13.60±0.98 |94.10±0.41 96.06±0.27

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 225±27 neurons | 64±0 bits
    GA Neurons  : 236±17 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.40±1.04 91.92±0.54 |16.54±1.33 13.58±1.07 |94.81±0.54 96.10±0.26
    fixed_05             |65.90±16.86 49.46±1.22 | 1.07±0.14  1.19±0.23 |70.89±18.68 52.61±1.54
    platt                |88.69±1.69 91.16±0.57 |20.03±8.53 20.17±1.28 |94.64±0.50 95.95±0.25
    beta                 |87.88±2.62 90.22±0.66 |25.10±10.13 24.37±1.61 |94.51±0.85 95.65±0.27
    empirical            |78.04±10.75 91.83±0.51 |10.79±11.39 15.27±1.34 |84.68±10.30 96.11±0.25
    empirical_cumulative |89.05±0.95 91.66±0.49 |11.80±2.37 10.83±1.05 |94.37±0.47 95.86±0.25
    val_cal              |89.40±1.05 91.92±0.56 |16.26±1.42 13.72±1.36 |94.80±0.55 96.10±0.26

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 235±19 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.27±6.78 91.64±0.59 |17.77±4.04 14.03±1.25 |93.27±4.89 95.96±0.28
    fixed_05             |65.69±16.98 49.60±1.60 |10.65±30.36  1.59±0.59 |74.01±18.08 52.83±2.04
    platt                |85.07±11.97 90.79±0.65 |26.43±24.78 21.21±1.60 |93.68±3.13 95.80±0.28
    beta                 |83.99±13.45 89.87±0.70 |31.43±25.68 25.31±1.81 |93.73±2.80 95.51±0.28
    empirical            |74.11±12.70 91.53±0.66 |16.97±26.65 15.64±2.52 |82.93±10.43 95.97±0.29
    empirical_cumulative |86.85±6.87 91.43±0.62 |11.63±2.40 11.34±1.10 |92.54±5.60 95.74±0.31
    val_cal              |87.27±6.79 91.63±0.59 |17.12±3.62 13.94±1.77 |93.21±4.99 95.95±0.26

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 235±19 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.27±6.78 91.64±0.59 |17.77±4.04 14.03±1.25 |93.27±4.89 95.96±0.28
    fixed_05             |65.69±16.98 49.60±1.60 |10.65±30.36  1.59±0.59 |74.01±18.08 52.83±2.04
    platt                |85.07±11.97 90.79±0.65 |26.43±24.78 21.21±1.60 |93.68±3.13 95.80±0.28
    beta                 |83.99±13.45 89.87±0.70 |31.43±25.68 25.31±1.81 |93.73±2.80 95.51±0.28
    empirical            |74.11±12.70 91.53±0.66 |16.97±26.65 15.64±2.52 |82.93±10.43 95.97±0.29
    empirical_cumulative |86.85±6.87 91.43±0.62 |11.63±2.40 11.34±1.10 |92.54±5.60 95.74±0.31
    val_cal              |87.27±6.79 91.63±0.59 |17.12±3.62 13.94±1.77 |93.21±4.99 95.95±0.26


## SP-ciciot-ablqsr-96bWc-n10  (10/10 completed)

    dataset=ciciot2023_neto_subsample split=random_3way bits=96 feats=top20 class=binary | mem=QSR | caps 250n/100b | w(ce/acc/f1/fpr)=0.7/0.1/0.15/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  92.69% |  10.46% |  96.41% | r24530 GA best_acc     val_cal
    Best F1 (FPR<10%)   |  92.48% |   8.59% |  96.24% | r61231 GA best_acc     empirical_cumulative
    Best F1 (FPR<6%)    |  90.66% |   4.52% |  95.02% | r42704 GA best_ce      fixed_05
    Best F1 (FPR<5%)    |  90.66% |   4.52% |  95.02% | r42704 GA best_ce      fixed_05
    Best F1 (FPR<4%)    |  89.70% |   3.73% |  94.39% | r24530 GA best_fpr     fixed_05
    Best F1 (FPR<2%)    |  81.93% |   0.85% |  88.63% | r29083 GS best_acc     fixed_05
    Best FPR (any F1)   |  67.89% |   0.01% |  74.99% | r42704 GS best_acc     empirical
    Best FPR (F1>80%)   |  81.93% |   0.85% |  88.63% | r29083 GS best_acc     fixed_05
    Best FPR (F1>90%)   |  90.66% |   4.52% |  95.02% | r42704 GA best_ce      fixed_05
    Best Acc (any FPR)  |  92.59% |  12.29% |  96.42% | r24530 GA best_acc     beta

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 206±42 neurons | 70±8 bits
    GA Neurons  : 116±40 neurons | 40±12 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.31±6.73 92.04±0.36 |18.08±4.34 11.18±0.58 |93.33±4.78 96.08±0.20
    fixed_05             |80.49±11.24 89.77±0.61 |13.01±29.57  4.65±1.22 |89.94±1.89 94.47±0.42
    platt                |85.58±12.07 91.94±0.41 |23.82±24.58 13.14±1.16 |93.80±3.13 96.09±0.20
    beta                 |85.08±13.66 91.86±0.41 |26.24±25.96 14.07±1.10 |94.00±2.82 96.09±0.20
    empirical            |81.37±12.97 91.98±0.40 |23.05±25.09 12.49±1.42 |89.99±8.02 96.09±0.20
    empirical_cumulative |86.88±6.79 91.75±0.42 |12.33±2.18  8.30±0.79 |92.61±5.51 95.81±0.25
    val_cal              |87.31±6.72 92.05±0.37 |17.67±3.72 11.28±0.83 |93.29±4.89 96.09±0.20

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 110±77 neurons | 38±8 bits
    GA Neurons  : 114±42 neurons | 37±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.56±0.87 91.83±0.37 |17.06±1.21 11.01±0.73 |93.75±0.49 95.96±0.20
    fixed_05             |85.42±0.66 90.03±0.58 | 6.98±0.72  4.85±0.99 |91.71±0.43 94.64±0.39
    platt                |87.35±0.97 91.75±0.39 |21.04±2.09 13.13±0.67 |93.87±0.45 95.99±0.21
    beta                 |87.30±0.93 91.66±0.42 |21.67±1.54 14.26±0.70 |93.88±0.46 95.98±0.21
    empirical            |87.27±0.96 91.71±0.40 |21.99±1.97 13.14±1.56 |93.88±0.45 95.97±0.21
    empirical_cumulative |87.02±0.89 91.46±0.44 |11.62±0.81  7.84±0.76 |93.08±0.56 95.63±0.28
    val_cal              |87.55±0.87 91.82±0.36 |17.53±1.10 11.10±0.97 |93.77±0.51 95.95±0.20

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 206±46 neurons | 66±5 bits
    GA Neurons  : 115±40 neurons | 40±12 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.32±6.73 92.06±0.39 |18.16±4.32 11.15±0.58 |93.34±4.78 96.09±0.21
    fixed_05             |80.44±11.21 89.79±0.61 |12.96±29.59  4.65±1.22 |89.90±1.87 94.48±0.42
    platt                |85.59±12.07 91.95±0.43 |23.76±24.60 13.19±1.08 |93.80±3.13 96.10±0.21
    beta                 |85.08±13.66 91.88±0.44 |26.16±25.98 14.05±1.12 |94.00±2.82 96.09±0.21
    empirical            |81.36±12.96 92.00±0.42 |23.08±25.09 12.54±1.38 |89.99±8.02 96.10±0.21
    empirical_cumulative |86.84±6.78 91.77±0.46 |12.20±2.15  8.31±0.79 |92.58±5.50 95.82±0.27
    val_cal              |87.31±6.72 92.07±0.40 |17.89±3.67 11.18±0.88 |93.30±4.89 96.09±0.22

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 140±61 neurons | 32±0 bits
    GA Neurons  : 122±58 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |88.43±1.31 91.35±0.66 |16.98±1.19 12.57±2.25 |94.26±0.71 95.75±0.30
    fixed_05             |84.65±2.43 89.61±0.86 | 6.39±2.88  6.05±1.57 |91.04±2.08 94.43±0.55
    platt                |88.14±0.91 91.29±0.63 |18.32±3.93 14.13±1.55 |94.18±0.34 95.78±0.31
    beta                 |88.13±1.08 91.24±0.60 |21.05±1.67 14.94±1.46 |94.32±0.60 95.78±0.29
    empirical            |88.19±1.29 90.87±1.46 |21.08±2.07 13.98±4.12 |94.35±0.66 95.50±1.01
    empirical_cumulative |88.03±1.53 91.03±0.78 |12.72±1.18  9.42±1.15 |93.76±0.95 95.44±0.43
    val_cal              |88.42±1.32 91.34±0.65 |16.60±1.24 12.51±2.29 |94.23±0.73 95.74±0.29

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 140±61 neurons | 32±0 bits
    GA Neurons  : 122±58 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |88.43±1.31 91.35±0.66 |16.98±1.19 12.57±2.25 |94.26±0.71 95.75±0.30
    fixed_05             |84.65±2.43 89.61±0.86 | 6.39±2.88  6.05±1.57 |91.04±2.08 94.43±0.55
    platt                |88.14±0.91 91.29±0.63 |18.32±3.93 14.13±1.55 |94.18±0.34 95.78±0.31
    beta                 |88.13±1.08 91.24±0.60 |21.05±1.67 14.94±1.46 |94.32±0.60 95.78±0.29
    empirical            |88.19±1.29 90.87±1.46 |21.08±2.07 13.98±4.12 |94.35±0.66 95.50±1.01
    empirical_cumulative |88.03±1.53 91.03±0.78 |12.72±1.18  9.42±1.15 |93.76±0.95 95.44±0.43
    val_cal              |88.42±1.32 91.34±0.65 |16.60±1.24 12.51±2.29 |94.23±0.73 95.74±0.29


## SP-ciciot-bin-96bWc-n30  (10/10 completed)

    dataset=ciciot2023_neto_subsample split=random_3way bits=96 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 250n/100b | w(ce/acc/f1/fpr)=0.7/0.1/0.15/0.05 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  92.99% |   7.76% |  96.49% | r98273 GA best_acc     platt
    Best F1 (FPR<10%)   |  92.99% |   7.76% |  96.49% | r98273 GA best_acc     platt
    Best F1 (FPR<6%)    |  92.86% |   5.36% |  96.34% | r98273 GA best_acc     empirical_cumulative
    Best F1 (FPR<5%)    |  92.58% |   4.76% |  96.17% | r29083 GA best_fpr     empirical_cumulative
    Best F1 (FPR<4%)    |  88.25% |   0.89% |  93.30% | r98273 GA best_ce      fixed_05
    Best F1 (FPR<2%)    |  88.25% |   0.89% |  93.30% | r98273 GA best_ce      fixed_05
    Best FPR (any F1)   |  70.73% |   0.00% |  78.08% | r42704 GA best_acc     empirical
    Best FPR (F1>80%)   |  85.55% |   0.64% |  91.39% | r29083 GA best_fpr     fixed_05
    Best FPR (F1>90%)   |  92.38% |   4.51% |  96.04% | r57826 GA best_ce      empirical_cumulative
    Best Acc (any FPR)  |  92.88% |  10.01% |  96.50% | r57826 GA best_acc     train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 125±72 neurons | 46±19 bits
    GA Neurons  : 160±71 neurons | 56±9 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.89±0.20 92.82±0.11 |15.33±1.07  9.13±1.02 |95.60±0.13 96.44±0.05
    fixed_05             |80.70±0.66 84.16±1.91 | 0.98±0.15  0.78±0.11 |87.63±0.55 90.33±1.43
    platt                |89.87±0.47 92.81±0.12 |11.07±0.36  8.76±0.43 |94.83±0.29 96.43±0.06
    beta                 |90.52±0.73 92.67±0.13 |17.86±3.67 11.10±0.59 |95.52±0.24 96.43±0.05
    empirical            |86.56±8.68 73.93±10.10 |13.55±7.42  1.95±4.11 |91.92±7.66 80.42±8.68
    empirical_cumulative |90.58±0.38 92.64±0.13 |12.94±2.13  6.80±1.55 |95.32±0.29 96.27±0.10
    val_cal              |90.89±0.21 92.82±0.11 |15.40±1.19  8.66±1.03 |95.60±0.14 96.42±0.05

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 150±53 neurons | 64±0 bits
    GA Neurons  : 216±31 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.94±0.20 92.68±0.11 |14.54±0.64  9.04±0.64 |95.03±0.13 96.36±0.06
    fixed_05             |81.82±0.11 86.06±1.20 | 0.99±0.07  0.82±0.12 |88.55±0.09 91.76±0.86
    platt                |89.95±0.19 92.68±0.12 |12.19±0.24  8.72±0.54 |94.92±0.10 96.35±0.06
    beta                 |89.87±0.23 92.56±0.12 |16.07±0.35 11.01±0.62 |95.06±0.11 96.36±0.06
    empirical            |69.79±2.23 67.81±1.29 | 0.01±0.01  0.01±0.00 |77.03±2.40 74.88±1.42
    empirical_cumulative |89.46±0.22 92.36±0.19 | 8.88±0.37  5.44±0.56 |94.48±0.14 96.06±0.10
    val_cal              |89.94±0.20 92.68±0.11 |14.49±0.94  8.68±0.70 |95.03±0.12 96.35±0.06

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 120±54 neurons | 53±28 bits
    GA Neurons  : 159±70 neurons | 55±10 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.76±0.24 92.80±0.11 |15.69±1.30  9.27±1.01 |95.55±0.12 96.44±0.05
    fixed_05             |80.54±0.66 84.00±1.99 | 0.96±0.15  0.78±0.10 |87.49±0.56 90.21±1.49
    platt                |89.62±0.39 92.78±0.11 |11.29±0.50  8.82±0.43 |94.69±0.24 96.41±0.05
    beta                 |90.55±0.57 92.66±0.12 |17.51±2.57 11.23±0.56 |95.51±0.21 96.42±0.05
    empirical            |88.64±6.11 74.16±10.02 |15.54±5.86  1.86±3.92 |93.82±5.34 80.66±8.60
    empirical_cumulative |90.55±0.35 92.65±0.14 |14.02±1.54  7.22±1.66 |95.35±0.25 96.29±0.11
    val_cal              |90.76±0.25 92.79±0.09 |15.76±1.27  8.88±1.15 |95.55±0.13 96.42±0.04

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 225±35 neurons | 64±0 bits
    GA Neurons  : 216±32 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.26±0.27 92.68±0.13 |15.63±0.74  9.20±0.76 |95.26±0.17 96.36±0.07
    fixed_05             |81.80±0.13 86.22±1.28 | 0.97±0.05  0.88±0.23 |88.54±0.10 91.88±0.91
    platt                |90.18±0.20 92.67±0.13 |12.05±0.19  8.76±0.54 |95.06±0.11 96.35±0.07
    beta                 |90.25±0.29 92.56±0.13 |15.75±0.43 11.02±0.64 |95.26±0.15 96.36±0.06
    empirical            |67.16±1.21 68.17±1.14 | 0.00±0.00  0.01±0.01 |74.15±1.35 75.29±1.26
    empirical_cumulative |89.84±0.25 92.37±0.16 | 9.78±0.77  5.55±0.80 |94.75±0.17 96.07±0.09
    val_cal              |90.27±0.28 92.67±0.14 |15.19±0.89  8.89±0.79 |95.25±0.18 96.35±0.06

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 225±35 neurons | 64±0 bits
    GA Neurons  : 216±32 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.26±0.27 92.68±0.13 |15.63±0.74  9.20±0.76 |95.26±0.17 96.36±0.07
    fixed_05             |81.80±0.13 86.22±1.28 | 0.97±0.05  0.88±0.23 |88.54±0.10 91.88±0.91
    platt                |90.18±0.20 92.67±0.13 |12.05±0.19  8.76±0.54 |95.06±0.11 96.35±0.07
    beta                 |90.25±0.29 92.56±0.13 |15.75±0.43 11.02±0.64 |95.26±0.15 96.36±0.06
    empirical            |67.16±1.21 68.17±1.14 | 0.00±0.00  0.01±0.01 |74.15±1.35 75.29±1.26
    empirical_cumulative |89.84±0.25 92.37±0.16 | 9.78±0.77  5.55±0.80 |94.75±0.17 96.07±0.09
    val_cal              |90.27±0.28 92.67±0.14 |15.19±0.89  8.89±0.79 |95.25±0.18 96.35±0.06


## SP-unswr-abl2big-64bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=BINARY | caps 250n/100b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  93.50% |   1.12% |  98.92% | r49648 GA best_acc     train_cal
    Best F1 (FPR<10%)   |  93.50% |   1.12% |  98.92% | r49648 GA best_acc     train_cal
    Best F1 (FPR<6%)    |  93.50% |   1.12% |  98.92% | r49648 GA best_acc     train_cal
    Best F1 (FPR<5%)    |  93.50% |   1.12% |  98.92% | r49648 GA best_acc     train_cal
    Best F1 (FPR<4%)    |  93.50% |   1.12% |  98.92% | r49648 GA best_acc     train_cal
    Best F1 (FPR<2%)    |  93.50% |   1.12% |  98.92% | r49648 GA best_acc     train_cal
    Best FPR (any F1)   |  49.03% |   0.00% |  96.19% | r32732 GS best_fpr     train_cal
    Best FPR (F1>80%)   |  84.85% |   0.41% |  98.09% | r34524 GS best_ce      beta
    Best FPR (F1>90%)   |  90.20% |   0.80% |  98.55% | r20361 GS best_ce      empirical
    Best Acc (any FPR)  |  93.50% |   1.12% |  98.92% | r49648 GA best_acc     train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 180±67 neurons | 16±0 bits
    GA Neurons  : 168±38 neurons | 17±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.46±0.02 93.48±0.02 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00
    fixed_05             |88.65±0.09 89.89±1.54 | 2.18±0.02  1.90±0.35 |97.90±0.02 98.17±0.33
    platt                |93.34±0.02 93.34±0.02 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.00
    beta                 |92.81±0.56 92.79±0.65 | 1.08±0.03  1.08±0.04 |98.83±0.07 98.82±0.08
    empirical            |92.74±0.14 92.60±0.18 | 1.09±0.01  1.08±0.01 |98.82±0.02 98.80±0.02
    empirical_cumulative |93.44±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.01
    val_cal              |93.45±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.01

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 5±0 neurons | 43±39 bits
    GA Neurons  : 7±1 neurons | 69±16 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |73.98±21.91 73.84±21.78 | 1.08±1.57  1.13±1.57 |97.39±1.58 97.36±1.55
    fixed_05             |51.68±44.47 51.54±44.33 |46.22±49.05 46.23±49.03 |55.52±47.15 55.50±47.13
    platt                |70.57±22.71 70.69±22.83 | 0.51±0.54  0.50±0.53 |97.47±1.35 97.48±1.36
    beta                 |65.53±33.68 65.43±33.59 |27.18±37.15 27.13±37.19 |73.71±35.63 73.71±35.63
    empirical            |51.66±44.45 51.40±44.19 |46.17±49.09 46.14±49.12 |55.53±47.16 55.50±47.13
    empirical_cumulative |73.99±21.93 73.86±21.80 | 1.09±1.57  1.14±1.57 |97.39±1.58 97.36±1.55
    val_cal              |73.99±21.92 73.86±21.80 | 1.10±1.57  1.14±1.57 |97.39±1.58 97.36±1.55

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 180±67 neurons | 16±0 bits
    GA Neurons  : 168±38 neurons | 17±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.46±0.02 93.48±0.02 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00
    fixed_05             |88.65±0.09 89.89±1.54 | 2.18±0.02  1.90±0.35 |97.90±0.02 98.17±0.33
    platt                |93.34±0.02 93.34±0.02 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.00
    beta                 |92.81±0.56 92.79±0.65 | 1.08±0.03  1.08±0.04 |98.83±0.07 98.82±0.08
    empirical            |92.74±0.14 92.60±0.18 | 1.09±0.01  1.08±0.01 |98.82±0.02 98.80±0.02
    empirical_cumulative |93.44±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.01
    val_cal              |93.45±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.01

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 240±21 neurons | 72±14 bits
    GA Neurons  : 246±7 neurons | 72±16 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.40±0.02 93.38±0.05 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.01
    fixed_05             |93.26±0.03 93.26±0.05 | 1.11±0.00  1.11±0.00 |98.88±0.00 98.88±0.01
    platt                |92.59±0.08 92.60±0.12 | 1.06±0.01  1.06±0.01 |98.80±0.01 98.80±0.01
    beta                 |88.85±2.81 90.02±2.66 | 0.72±0.23  0.81±0.24 |98.43±0.25 98.55±0.25
    empirical            |89.39±0.35 89.18±0.51 | 0.74±0.03  0.71±0.04 |98.47±0.04 98.45±0.05
    empirical_cumulative |93.39±0.02 93.37±0.04 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.01
    val_cal              |93.40±0.01 93.38±0.04 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.01

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 180±67 neurons | 16±0 bits
    GA Neurons  : 168±38 neurons | 17±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.46±0.02 93.48±0.02 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00
    fixed_05             |88.65±0.09 89.89±1.54 | 2.18±0.02  1.90±0.35 |97.90±0.02 98.17±0.33
    platt                |93.34±0.02 93.34±0.02 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.00
    beta                 |92.81±0.56 92.79±0.65 | 1.08±0.03  1.08±0.04 |98.83±0.07 98.82±0.08
    empirical            |92.74±0.14 92.60±0.18 | 1.09±0.01  1.08±0.01 |98.82±0.02 98.80±0.02
    empirical_cumulative |93.44±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.01
    val_cal              |93.45±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.01


## SP-unswr-abl2s-64bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=BINARY | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05
    Best F1 (FPR<10%)   |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05
    Best F1 (FPR<6%)    |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05
    Best F1 (FPR<5%)    |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05
    Best F1 (FPR<4%)    |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05
    Best F1 (FPR<2%)    |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05
    Best FPR (any F1)   |  49.03% |   0.00% |  96.19% | r63890 GS best_fpr     platt
    Best FPR (F1>80%)   |  88.34% |   0.70% |  98.36% | r49337 GS best_ce      beta
    Best FPR (F1>90%)   |  90.68% |   0.81% |  98.61% | r32732 GS best_ce      empirical
    Best Acc (any FPR)  |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 300±150 neurons | 16±6 bits
    GA Neurons  : 223±155 neurons | 15±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.46±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    fixed_05             |87.01±10.89 76.72±20.91 | 3.87±6.58 12.40±19.42 |96.27±6.33 88.07±18.68
    platt                |93.33±0.06 93.34±0.10 | 1.12±0.00  1.11±0.00 |98.89±0.01 98.90±0.01
    beta                 |92.44±0.78 92.89±0.43 | 1.07±0.04  1.10±0.06 |98.78±0.10 98.83±0.06
    empirical            |92.55±0.26 92.76±0.32 | 1.07±0.02  1.08±0.03 |98.79±0.03 98.82±0.04
    empirical_cumulative |93.45±0.03 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    val_cal              |93.45±0.03 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 88±202 neurons | 18±15 bits
    GA Neurons  : 71±138 neurons | 22±14 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |73.94±20.91 73.55±20.53 | 3.55±6.05  3.64±6.02 |95.41±5.33 95.33±5.27
    fixed_05             |41.94±44.42 35.93±40.55 |57.39±49.11 63.44±44.97 |44.79±47.24 38.97±43.24
    platt                |70.96±23.11 70.64±22.80 | 0.56±0.59  0.63±0.69 |97.51±1.39 97.45±1.33
    beta                 |69.26±28.90 68.94±28.63 |20.85±29.96 20.92±29.91 |79.88±28.78 79.82±28.73
    empirical            |50.93±44.90 50.57±44.53 |47.45±49.59 47.50±49.54 |54.31±47.66 54.25±47.59
    empirical_cumulative |71.09±23.26 70.69±22.84 | 0.55±0.58  0.61±0.68 |97.53±1.41 97.45±1.34
    val_cal              |73.94±20.91 73.54±20.52 | 3.55±6.05  3.65±6.01 |95.41±5.33 95.33±5.27

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 300±150 neurons | 16±6 bits
    GA Neurons  : 217±147 neurons | 14±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.46±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    fixed_05             |87.01±10.89 73.14±21.69 | 3.87±6.58 14.98±19.63 |96.27±6.33 85.59±18.88
    platt                |93.33±0.06 93.34±0.10 | 1.12±0.00  1.11±0.00 |98.89±0.01 98.90±0.01
    beta                 |92.44±0.78 92.98±0.39 | 1.07±0.04  1.11±0.06 |98.78±0.10 98.84±0.05
    empirical            |92.55±0.26 92.79±0.35 | 1.07±0.02  1.08±0.03 |98.79±0.03 98.82±0.04
    empirical_cumulative |93.45±0.03 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    val_cal              |93.45±0.03 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 485±34 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.45±0.02 93.44±0.02 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    fixed_05             |93.36±0.16 93.36±0.17 | 1.14±0.03  1.13±0.03 |98.90±0.03 98.90±0.03
    platt                |93.11±0.05 93.09±0.01 | 1.11±0.00  1.11±0.00 |98.86±0.01 98.86±0.00
    beta                 |92.02±1.42 92.61±0.17 | 1.01±0.13  1.07±0.02 |98.74±0.15 98.80±0.02
    empirical            |90.90±0.23 90.77±0.30 | 0.89±0.04  0.88±0.03 |98.61±0.02 98.60±0.03
    empirical_cumulative |93.44±0.02 93.44±0.02 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    val_cal              |93.44±0.02 93.44±0.02 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 300±150 neurons | 16±6 bits
    GA Neurons  : 223±155 neurons | 15±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.46±0.02 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    fixed_05             |87.01±10.89 76.72±20.91 | 3.87±6.58 12.40±19.42 |96.27±6.33 88.07±18.68
    platt                |93.33±0.06 93.34±0.10 | 1.12±0.00  1.11±0.00 |98.89±0.01 98.90±0.01
    beta                 |92.44±0.78 92.89±0.43 | 1.07±0.04  1.10±0.06 |98.78±0.10 98.83±0.06
    empirical            |92.55±0.26 92.76±0.32 | 1.07±0.02  1.08±0.03 |98.79±0.03 98.82±0.04
    empirical_cumulative |93.45±0.03 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00
    val_cal              |93.45±0.03 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.91±0.00


## SP-unswr-abl3s-64bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=TERNARY | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  93.52% |   1.12% |  98.92% | r63890 GA best_f1      fixed_05
    Best F1 (FPR<10%)   |  93.52% |   1.12% |  98.92% | r63890 GA best_f1      fixed_05
    Best F1 (FPR<6%)    |  93.52% |   1.12% |  98.92% | r63890 GA best_f1      fixed_05
    Best F1 (FPR<5%)    |  93.52% |   1.12% |  98.92% | r63890 GA best_f1      fixed_05
    Best F1 (FPR<4%)    |  93.52% |   1.12% |  98.92% | r63890 GA best_f1      fixed_05
    Best F1 (FPR<2%)    |  93.52% |   1.12% |  98.92% | r63890 GA best_f1      fixed_05
    Best FPR (any F1)   |  49.03% |   0.00% |  96.19% | r32732 GS best_fpr     train_cal
    Best FPR (F1>80%)   |  90.52% |   0.79% |  98.59% | r46247 GA best_ce      empirical
    Best FPR (F1>90%)   |  90.52% |   0.79% |  98.59% | r46247 GA best_ce      empirical
    Best Acc (any FPR)  |  93.52% |   1.12% |  98.92% | r32732 GS best_ce      fixed_05

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 230±163 neurons | 27±3 bits
    GA Neurons  : 61±57 neurons | 30±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.02 93.50±0.02 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.04±0.22 93.25±0.25 | 1.21±0.04  1.17±0.05 |98.83±0.04 98.87±0.05
    platt                |93.32±0.05 93.30±0.06 | 1.11±0.00  1.11±0.00 |98.89±0.01 98.89±0.01
    beta                 |92.70±0.93 92.71±0.92 | 1.07±0.06  1.07±0.07 |98.82±0.11 98.82±0.11
    empirical            |92.11±0.58 92.31±0.59 | 1.01±0.06  1.03±0.05 |98.74±0.07 98.77±0.07
    empirical_cumulative |93.51±0.02 93.50±0.03 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.02 93.50±0.03 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 5±0 neurons | 29±6 bits
    GA Neurons  : 17±4 neurons | 32±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |76.44±21.93 72.85±21.57 | 2.83±6.61  4.73±8.47 |96.12±6.06 94.29±7.61
    fixed_05             |57.16±45.61 49.97±44.03 |40.60±50.70 47.80±49.38 |60.95±48.76 54.02±47.50
    platt                |75.48±22.76 70.95±23.11 | 0.66±0.57  0.57±0.60 |97.80±1.38 97.51±1.39
    beta                 |71.61±29.96 67.81±28.85 |20.12±30.54 22.03±29.79 |80.57±29.33 78.71±28.58
    empirical            |57.39±45.79 53.82±43.95 |40.45±50.82 42.29±49.57 |61.02±48.82 59.21±47.58
    empirical_cumulative |75.59±22.86 71.05±23.22 | 0.67±0.57  0.59±0.62 |97.81±1.40 97.52±1.40
    val_cal              |76.44±21.93 72.85±21.57 | 2.83±6.61  4.74±8.46 |96.12±6.06 94.29±7.61

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 220±168 neurons | 27±3 bits
    GA Neurons  : 59±59 neurons | 30±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.02 93.47±0.11 | 1.12±0.00  1.11±0.01 |98.92±0.00 98.92±0.01
    fixed_05             |93.10±0.23 93.23±0.24 | 1.20±0.05  1.18±0.05 |98.84±0.04 98.87±0.05
    platt                |93.31±0.05 93.29±0.07 | 1.11±0.00  1.11±0.01 |98.89±0.01 98.89±0.01
    beta                 |92.98±0.63 92.92±0.62 | 1.10±0.03  1.09±0.04 |98.85±0.08 98.84±0.08
    empirical            |92.12±0.61 92.30±0.64 | 1.01±0.06  1.02±0.07 |98.75±0.07 98.77±0.08
    empirical_cumulative |93.51±0.02 93.48±0.08 | 1.12±0.00  1.12±0.01 |98.92±0.00 98.92±0.01
    val_cal              |93.51±0.02 93.48±0.08 | 1.12±0.00  1.12±0.01 |98.92±0.00 98.92±0.01

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 427±150 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.48±0.03 93.50±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.40±0.19 93.43±0.16 | 1.14±0.04  1.14±0.03 |98.90±0.04 98.90±0.03
    platt                |93.20±0.06 93.23±0.02 | 1.11±0.00  1.11±0.00 |98.88±0.01 98.88±0.00
    beta                 |92.51±0.67 92.14±1.26 | 1.04±0.07  1.00±0.13 |98.80±0.08 98.76±0.14
    empirical            |90.77±0.36 90.86±0.55 | 0.85±0.04  0.83±0.05 |98.61±0.04 98.62±0.06
    empirical_cumulative |93.49±0.04 93.51±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.01 98.92±0.00
    val_cal              |93.49±0.04 93.51±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.01 98.92±0.00

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 230±163 neurons | 27±3 bits
    GA Neurons  : 61±57 neurons | 30±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.02 93.50±0.02 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.04±0.22 93.25±0.25 | 1.21±0.04  1.17±0.05 |98.83±0.04 98.87±0.05
    platt                |93.32±0.05 93.30±0.06 | 1.11±0.00  1.11±0.00 |98.89±0.01 98.89±0.01
    beta                 |92.70±0.93 92.71±0.92 | 1.07±0.06  1.07±0.07 |98.82±0.11 98.82±0.11
    empirical            |92.11±0.58 92.31±0.59 | 1.01±0.06  1.03±0.05 |98.74±0.07 98.77±0.07
    empirical_cumulative |93.51±0.02 93.50±0.03 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.02 93.50±0.03 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## SP-unswr-ablpln-64bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=PLN | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  93.71% |   1.02% |  98.97% | r63890 GS best_f1      train_cal
    Best F1 (FPR<10%)   |  93.71% |   1.02% |  98.97% | r63890 GS best_f1      train_cal
    Best F1 (FPR<6%)    |  93.71% |   1.02% |  98.97% | r63890 GS best_f1      train_cal
    Best F1 (FPR<5%)    |  93.71% |   1.02% |  98.97% | r63890 GS best_f1      train_cal
    Best F1 (FPR<4%)    |  93.71% |   1.02% |  98.97% | r63890 GS best_f1      train_cal
    Best F1 (FPR<2%)    |  93.71% |   1.02% |  98.97% | r63890 GS best_f1      train_cal
    Best FPR (any F1)   |  49.48% |   0.00% |  96.20% | r10596 GA best_fpr     beta
    Best FPR (F1>80%)   |  80.84% |   0.31% |  97.77% | r32732 GA best_f1      fixed_05
    Best FPR (F1>90%)   |  90.81% |   0.65% |  98.67% | r34524 GS best_ce      empirical
    Best Acc (any FPR)  |  93.67% |   0.95% |  98.98% | r20361 GS best_ce      platt

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 420±103 neurons | 19±3 bits
    GA Neurons  : 462±78 neurons | 25±5 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.54±0.10 93.52±0.11 | 1.09±0.04  1.03±0.05 |98.93±0.02 98.94±0.02
    fixed_05             |79.59±1.89 80.55±2.53 | 0.36±0.05  0.35±0.05 |97.64±0.14 97.74±0.18
    platt                |93.43±0.10 93.50±0.11 | 1.04±0.06  1.02±0.04 |98.92±0.02 98.94±0.02
    beta                 |93.53±0.11 93.37±0.18 | 1.10±0.04  1.14±0.05 |98.93±0.02 98.90±0.04
    empirical            |89.24±1.50 89.51±1.08 | 0.69±0.05  0.63±0.07 |98.47±0.16 98.52±0.12
    empirical_cumulative |93.49±0.14 93.47±0.12 | 1.06±0.05  1.01±0.05 |98.93±0.03 98.94±0.02
    val_cal              |93.52±0.12 93.52±0.10 | 1.08±0.05  1.04±0.04 |98.93±0.02 98.94±0.02

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 262±234 neurons | 24±9 bits
    GA Neurons  : 313±225 neurons | 24±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.48±7.73 88.08±10.58 | 1.18±0.32  1.70±1.21 |98.40±0.96 97.94±1.90
    fixed_05             |80.55±6.00 78.34±5.76 | 0.61±0.45  0.71±1.09 |97.63±0.74 97.30±1.23
    platt                |86.40±11.82 86.71±12.04 | 0.83±0.29  0.87±0.25 |98.32±0.92 98.33±1.07
    beta                 |84.62±13.61 84.96±16.60 | 0.94±0.51  1.00±0.58 |98.16±0.99 98.29±1.06
    empirical            |86.08±10.13 84.34±12.21 | 0.72±0.26  0.58±0.26 |98.23±0.68 98.16±0.78
    empirical_cumulative |89.40±7.72 88.00±10.69 | 1.07±0.23  1.23±0.93 |98.42±0.95 98.19±1.63
    val_cal              |89.47±7.72 88.09±10.59 | 1.19±0.31  1.70±1.21 |98.40±0.95 97.94±1.90

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 440±97 neurons | 23±6 bits
    GA Neurons  : 460±78 neurons | 26±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.52±0.09 93.52±0.10 | 1.07±0.04  1.03±0.04 |98.93±0.02 98.94±0.02
    fixed_05             |80.48±2.08 81.26±1.13 | 0.37±0.04  0.35±0.03 |97.72±0.16 97.79±0.09
    platt                |93.45±0.06 93.52±0.09 | 1.04±0.05  1.02±0.03 |98.93±0.02 98.94±0.02
    beta                 |93.41±0.19 93.36±0.17 | 1.13±0.05  1.14±0.05 |98.90±0.04 98.89±0.04
    empirical            |89.04±1.41 89.77±1.06 | 0.65±0.06  0.62±0.06 |98.46±0.15 98.55±0.12
    empirical_cumulative |93.44±0.11 93.48±0.13 | 1.03±0.05  1.00±0.03 |98.93±0.02 98.94±0.02
    val_cal              |93.49±0.10 93.52±0.11 | 1.07±0.04  1.04±0.04 |98.93±0.02 98.94±0.02

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 380±148 neurons | 24±0 bits
    GA Neurons  : 341±149 neurons | 24±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.47±0.10 93.54±0.08 | 1.07±0.04  1.06±0.04 |98.92±0.02 98.94±0.02
    fixed_05             |81.47±0.93 81.63±0.93 | 0.40±0.04  0.40±0.06 |97.78±0.08 97.80±0.07
    platt                |93.38±0.24 93.38±0.36 | 1.03±0.06  1.01±0.05 |98.92±0.03 98.92±0.05
    beta                 |93.44±0.10 93.47±0.09 | 1.11±0.04  1.10±0.03 |98.91±0.02 98.92±0.02
    empirical            |90.20±1.55 90.52±1.47 | 0.72±0.11  0.73±0.14 |98.58±0.17 98.62±0.15
    empirical_cumulative |93.45±0.09 93.53±0.08 | 1.04±0.06  1.04±0.05 |98.93±0.02 98.94±0.02
    val_cal              |93.48±0.10 93.54±0.08 | 1.07±0.06  1.06±0.03 |98.93±0.03 98.94±0.02

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 440±97 neurons | 22±6 bits
    GA Neurons  : 461±77 neurons | 25±5 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.52±0.09 93.52±0.10 | 1.07±0.04  1.03±0.05 |98.93±0.02 98.94±0.02
    fixed_05             |79.93±2.28 80.51±2.52 | 0.36±0.04  0.35±0.05 |97.68±0.18 97.73±0.18
    platt                |93.44±0.07 93.50±0.11 | 1.04±0.05  1.03±0.04 |98.93±0.02 98.94±0.02
    beta                 |93.42±0.19 93.37±0.18 | 1.13±0.05  1.14±0.05 |98.91±0.04 98.90±0.04
    empirical            |88.97±1.41 89.55±1.03 | 0.65±0.06  0.64±0.07 |98.45±0.15 98.52±0.11
    empirical_cumulative |93.44±0.11 93.48±0.12 | 1.03±0.06  1.01±0.05 |98.93±0.02 98.94±0.02
    val_cal              |93.49±0.10 93.52±0.10 | 1.07±0.04  1.04±0.04 |98.93±0.02 98.94±0.02


## SP-unswr-ablqsr-64bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=QSR | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  94.48% |   0.52% |  99.17% | r49648 GA best_acc     beta
    Best F1 (FPR<10%)   |  94.48% |   0.52% |  99.17% | r49648 GA best_acc     beta
    Best F1 (FPR<6%)    |  94.48% |   0.52% |  99.17% | r49648 GA best_acc     beta
    Best F1 (FPR<5%)    |  94.48% |   0.52% |  99.17% | r49648 GA best_acc     beta
    Best F1 (FPR<4%)    |  94.48% |   0.52% |  99.17% | r49648 GA best_acc     beta
    Best F1 (FPR<2%)    |  94.48% |   0.52% |  99.17% | r49648 GA best_acc     beta
    Best FPR (any F1)   |  64.35% |   0.00% |  96.87% | r10596 GA best_fpr     empirical
    Best FPR (F1>80%)   |  80.40% |   0.00% |  97.90% | r10596 GA best_ce      empirical
    Best FPR (F1>90%)   |  94.02% |   0.37% |  99.14% | r54070 GA best_fpr     empirical_cumulative
    Best Acc (any FPR)  |  94.40% |   0.45% |  99.17% | r10596 GA best_acc     train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 350±108 neurons | 33±1 bits
    GA Neurons  : 356±120 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.29±0.06 94.31±0.06 | 0.55±0.03  0.56±0.05 |99.14±0.01 99.14±0.01
    fixed_05             |93.62±0.13 93.63±0.15 | 1.07±0.05  1.06±0.06 |98.95±0.03 98.95±0.03
    platt                |94.28±0.06 94.31±0.08 | 0.54±0.02  0.53±0.01 |99.14±0.01 99.15±0.01
    beta                 |94.29±0.06 94.30±0.07 | 0.55±0.02  0.54±0.02 |99.14±0.01 99.15±0.01
    empirical            |71.49±4.79 72.33±6.48 | 0.00±0.00  0.01±0.01 |97.29±0.31 97.36±0.49
    empirical_cumulative |94.24±0.09 94.26±0.12 | 0.52±0.05  0.51±0.06 |99.14±0.01 99.14±0.01
    val_cal              |94.29±0.06 94.33±0.07 | 0.60±0.04  0.62±0.07 |99.13±0.01 99.14±0.02

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 370±134 neurons | 26±7 bits
    GA Neurons  : 378±133 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.11±0.18 94.24±0.08 | 0.57±0.06  0.55±0.05 |99.11±0.03 99.14±0.01
    fixed_05             |93.49±0.12 93.61±0.14 | 1.11±0.04  1.07±0.06 |98.92±0.02 98.94±0.03
    platt                |94.10±0.16 94.23±0.09 | 0.57±0.04  0.54±0.02 |99.11±0.03 99.14±0.01
    beta                 |94.13±0.17 94.24±0.08 | 0.59±0.05  0.56±0.03 |99.11±0.03 99.13±0.01
    empirical            |72.97±9.68 71.36±6.48 | 0.12±0.35  0.00±0.01 |97.43±0.68 97.30±0.47
    empirical_cumulative |94.02±0.20 94.18±0.10 | 0.52±0.08  0.52±0.09 |99.11±0.03 99.13±0.01
    val_cal              |94.12±0.17 94.26±0.08 | 0.59±0.07  0.62±0.05 |99.11±0.03 99.12±0.02

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 360±135 neurons | 34±1 bits
    GA Neurons  : 335±120 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.32±0.05 94.32±0.09 | 0.57±0.02  0.54±0.06 |99.14±0.01 99.15±0.02
    fixed_05             |93.59±0.13 93.63±0.15 | 1.08±0.05  1.06±0.06 |98.94±0.03 98.95±0.03
    platt                |94.29±0.05 94.32±0.08 | 0.53±0.02  0.52±0.02 |99.15±0.01 99.15±0.01
    beta                 |94.30±0.05 94.32±0.10 | 0.55±0.02  0.54±0.02 |99.14±0.01 99.15±0.01
    empirical            |72.29±6.07 73.73±6.49 | 0.00±0.01  0.00±0.01 |97.35±0.43 97.45±0.48
    empirical_cumulative |94.28±0.08 94.28±0.08 | 0.53±0.05  0.50±0.05 |99.14±0.01 99.15±0.01
    val_cal              |94.33±0.06 94.32±0.08 | 0.60±0.05  0.61±0.08 |99.14±0.02 99.14±0.02

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 380±114 neurons | 34±0 bits
    GA Neurons  : 287±137 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.31±0.05 94.32±0.03 | 0.55±0.03  0.54±0.05 |99.15±0.01 99.15±0.01
    fixed_05             |93.57±0.12 93.71±0.15 | 1.09±0.05  1.03±0.06 |98.94±0.03 98.97±0.03
    platt                |94.29±0.06 94.32±0.06 | 0.52±0.01  0.52±0.02 |99.15±0.01 99.15±0.01
    beta                 |94.30±0.05 94.33±0.06 | 0.54±0.02  0.55±0.02 |99.14±0.01 99.15±0.01
    empirical            |72.66±4.90 74.88±7.80 | 0.00±0.00  0.00±0.01 |97.37±0.33 97.54±0.55
    empirical_cumulative |94.28±0.08 94.28±0.08 | 0.53±0.06  0.51±0.07 |99.14±0.01 99.15±0.01
    val_cal              |94.33±0.03 94.34±0.07 | 0.60±0.05  0.62±0.05 |99.14±0.01 99.14±0.02

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 380±114 neurons | 33±1 bits
    GA Neurons  : 357±121 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |94.27±0.05 94.31±0.06 | 0.55±0.04  0.55±0.05 |99.14±0.01 99.14±0.01
    fixed_05             |93.57±0.12 93.63±0.15 | 1.09±0.05  1.06±0.06 |98.94±0.03 98.95±0.03
    platt                |94.28±0.07 94.31±0.08 | 0.53±0.01  0.52±0.01 |99.14±0.01 99.15±0.01
    beta                 |94.28±0.06 94.30±0.08 | 0.55±0.02  0.54±0.02 |99.14±0.01 99.15±0.01
    empirical            |71.03±4.01 72.62±6.45 | 0.00±0.00  0.01±0.01 |97.26±0.25 97.38±0.48
    empirical_cumulative |94.25±0.07 94.26±0.12 | 0.53±0.06  0.51±0.06 |99.14±0.01 99.14±0.01
    val_cal              |94.29±0.07 94.33±0.07 | 0.60±0.05  0.62±0.07 |99.13±0.02 99.13±0.02


## SP-unswr-bin-64bWb-n30  (10/10 completed)

    dataset=unsw-nb15 split=random_3way bits=64 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  93.95% |   0.74% |  99.06% | r49648 GA best_fitness train_cal
    Best F1 (FPR<10%)   |  93.95% |   0.74% |  99.06% | r49648 GA best_fitness train_cal
    Best F1 (FPR<6%)    |  93.95% |   0.74% |  99.06% | r49648 GA best_fitness train_cal
    Best F1 (FPR<5%)    |  93.95% |   0.74% |  99.06% | r49648 GA best_fitness train_cal
    Best F1 (FPR<4%)    |  93.95% |   0.74% |  99.06% | r49648 GA best_fitness train_cal
    Best F1 (FPR<2%)    |  93.95% |   0.74% |  99.06% | r49648 GA best_fitness train_cal
    Best FPR (any F1)   |  81.58% |   0.06% |  97.96% | r54070 GA best_fpr     empirical_cumulative
    Best FPR (F1>80%)   |  81.58% |   0.06% |  97.96% | r54070 GA best_fpr     empirical_cumulative
    Best FPR (F1>90%)   |  91.01% |   0.38% |  98.77% | r49648 GA best_acc     empirical
    Best Acc (any FPR)  |  93.95% |   0.74% |  99.06% | r49648 GA best_f1      train_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 240±143 neurons | 16±2 bits
    GA Neurons  : 291±162 neurons | 15±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.49±0.01 93.54±0.14 | 1.12±0.00  1.08±0.12 |98.92±0.00 98.93±0.04
    fixed_05             |92.83±0.35 92.36±1.10 | 1.26±0.07  1.36±0.24 |98.79±0.07 98.69±0.23
    platt                |93.30±0.02 93.32±0.05 | 1.12±0.00  1.07±0.15 |98.89±0.00 98.90±0.03
    beta                 |90.96±4.70 92.85±1.01 | 1.02±0.19  1.04±0.17 |98.64±0.47 98.85±0.13
    empirical            |90.84±1.92 90.91±2.99 | 0.97±0.10  0.92±0.22 |98.59±0.22 98.63±0.33
    empirical_cumulative |93.49±0.01 93.54±0.15 | 1.12±0.00  1.08±0.12 |98.92±0.00 98.93±0.04
    val_cal              |93.49±0.01 93.54±0.15 | 1.12±0.00  1.08±0.12 |98.92±0.00 98.93±0.04

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 14±30 neurons | 19±11 bits
    GA Neurons  : 44±67 neurons | 13±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.89±3.50 88.43±4.17 | 1.13±0.52  1.80±0.77 |98.58±0.39 97.97±0.86
    fixed_05             |88.53±6.62 84.27±9.84 | 2.41±2.08  4.12±4.29 |97.65±2.10 96.00±4.21
    platt                |90.35±3.44 87.03±4.77 | 0.90±0.26  1.31±0.30 |98.57±0.32 98.02±0.57
    beta                 |90.17±3.38 87.25±5.04 | 0.89±0.26  0.83±0.52 |98.55±0.31 98.24±0.47
    empirical            |90.56±3.34 87.65±5.00 | 1.01±0.44  1.55±0.98 |98.56±0.35 98.01±0.70
    empirical_cumulative |90.82±3.52 87.65±4.55 | 0.93±0.27  0.90±0.64 |98.63±0.33 98.23±0.64
    val_cal              |90.89±3.50 88.43±4.17 | 1.13±0.52  1.80±0.77 |98.58±0.39 97.97±0.86

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 240±143 neurons | 16±2 bits
    GA Neurons  : 244±176 neurons | 14±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.49±0.01 93.50±0.17 | 1.12±0.00  1.01±0.18 |98.92±0.00 98.94±0.04
    fixed_05             |92.83±0.35 92.06±1.52 | 1.26±0.07  1.42±0.33 |98.79±0.07 98.63±0.32
    platt                |93.30±0.02 93.24±0.38 | 1.12±0.00  0.99±0.21 |98.89±0.00 98.91±0.05
    beta                 |90.96±4.70 92.84±1.03 | 1.02±0.19  0.95±0.23 |98.64±0.47 98.86±0.13
    empirical            |90.84±1.92 90.90±3.03 | 0.97±0.10  0.85±0.24 |98.59±0.22 98.64±0.34
    empirical_cumulative |93.49±0.01 93.50±0.18 | 1.12±0.00  1.01±0.18 |98.92±0.00 98.94±0.04
    val_cal              |93.49±0.01 93.50±0.17 | 1.12±0.00  1.01±0.18 |98.92±0.00 98.94±0.04

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 420±132 neurons | 12±1 bits
    GA Neurons  : 404±136 neurons | 15±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.47±0.01 93.49±0.01 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00
    fixed_05             |92.41±0.23 92.82±0.21 | 1.34±0.05  1.26±0.04 |98.71±0.05 98.79±0.04
    platt                |93.26±0.14 93.27±0.05 | 1.12±0.00  1.10±0.04 |98.88±0.02 98.89±0.01
    beta                 |93.16±0.14 93.23±0.10 | 1.11±0.00  1.11±0.01 |98.87±0.02 98.88±0.01
    empirical            |87.79±2.63 90.79±2.30 | 0.82±0.15  0.84±0.26 |98.27±0.27 98.63±0.27
    empirical_cumulative |93.46±0.01 93.41±0.19 | 1.12±0.00  1.06±0.14 |98.91±0.00 98.92±0.00
    val_cal              |93.46±0.01 93.49±0.01 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 240±143 neurons | 16±2 bits
    GA Neurons  : 291±162 neurons | 15±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.49±0.01 93.54±0.14 | 1.12±0.00  1.08±0.12 |98.92±0.00 98.93±0.04
    fixed_05             |92.83±0.35 92.36±1.10 | 1.26±0.07  1.36±0.24 |98.79±0.07 98.69±0.23
    platt                |93.30±0.02 93.32±0.05 | 1.12±0.00  1.07±0.15 |98.89±0.00 98.90±0.03
    beta                 |90.96±4.70 92.85±1.01 | 1.02±0.19  1.04±0.17 |98.64±0.47 98.85±0.13
    empirical            |90.84±1.92 90.91±2.99 | 0.97±0.10  0.92±0.22 |98.59±0.22 98.63±0.33
    empirical_cumulative |93.49±0.01 93.54±0.15 | 1.12±0.00  1.08±0.12 |98.92±0.00 98.93±0.04
    val_cal              |93.49±0.01 93.54±0.15 | 1.12±0.00  1.08±0.12 |98.92±0.00 98.93±0.04


## SP-unswt-abl2big-16bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=temporal_3way bits=16 feats=top20 class=binary | mem=BINARY | caps 250n/100b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  79.21% |  42.09% |  80.95% | r88120 GA best_acc     val_cal
    Best F1 (FPR<10%)   |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best F1 (FPR<6%)    |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best F1 (FPR<5%)    |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best F1 (FPR<4%)    |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best F1 (FPR<2%)    |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best FPR (any F1)   |  31.01% |   0.00% |  44.94% | r98954 GS best_fpr     empirical_cumulative
    Best FPR (F1>80%)   |       — |       — |       — | —
    Best FPR (F1>90%)   |       — |       — |       — | —
    Best Acc (any FPR)  |  79.21% |  42.09% |  80.95% | r88120 GA best_acc     val_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.05±0.08 79.06±0.08 |41.96±0.02 41.97±0.04 |80.76±0.08 80.77±0.08
    fixed_05             |56.27±14.51 45.11±15.04 |75.71±17.10 88.65±18.24 |65.96±7.68 60.15±8.18
    platt                |78.96±0.13 78.72±0.50 |41.94±0.05 42.43±1.04 |80.67±0.14 80.48±0.40
    beta                 |78.26±1.25 65.61±18.20 |41.50±0.81 59.22±25.76 |79.87±1.42 72.06±10.62
    empirical            |78.17±0.68 78.56±0.60 |41.72±0.20 41.85±0.12 |79.82±0.74 80.24±0.65
    empirical_cumulative |79.12±0.07 79.10±0.09 |42.00±0.04 42.01±0.06 |80.84±0.07 80.82±0.10
    val_cal              |79.13±0.06 79.12±0.08 |42.09±0.11 42.07±0.10 |80.87±0.07 80.85±0.09

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 5±0 neurons | 4±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |62.22±20.43 62.23±20.45 |63.36±29.04 63.36±29.04 |70.23±12.07 70.24±12.09
    fixed_05             |60.73±21.72 60.89±21.87 |62.66±32.97 62.67±32.96 |69.38±12.38 69.55±12.54
    platt                |60.89±20.32 60.94±20.37 |51.14±33.80 51.14±33.80 |68.17±12.81 68.23±12.86
    beta                 |59.88±20.29 60.32±20.53 |54.98±40.21 57.24±38.43 |67.99±11.42 68.53±11.55
    empirical            |60.77±19.15 60.89±19.25 |56.91±36.25 56.12±36.73 |68.33±10.32 68.38±10.35
    empirical_cumulative |55.05±20.84 55.04±20.84 | 7.54±8.02  6.52±6.84 |60.73±13.82 60.75±13.84
    val_cal              |62.31±21.53 62.32±21.54 |49.80±35.98 49.80±35.98 |69.73±14.11 69.74±14.11

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.05±0.08 79.06±0.08 |41.96±0.02 41.97±0.04 |80.76±0.08 80.77±0.08
    fixed_05             |56.99±14.96 45.11±15.04 |74.66±17.76 88.65±18.24 |66.43±7.97 60.15±8.18
    platt                |78.96±0.13 78.72±0.50 |41.94±0.05 42.43±1.04 |80.67±0.14 80.48±0.40
    beta                 |78.52±1.05 65.61±18.20 |41.68±0.64 59.22±25.76 |80.17±1.19 72.06±10.62
    empirical            |78.17±0.68 78.56±0.60 |41.74±0.20 41.85±0.12 |79.82±0.73 80.24±0.65
    empirical_cumulative |79.11±0.07 79.10±0.09 |42.00±0.03 42.01±0.06 |80.83±0.07 80.82±0.10
    val_cal              |79.14±0.06 79.12±0.08 |42.12±0.12 42.07±0.10 |80.87±0.07 80.85±0.09

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |78.77±0.24 78.60±0.32 |41.84±0.10 41.82±0.11 |80.46±0.27 80.27±0.35
    fixed_05             |78.16±0.16 77.98±0.20 |41.79±0.02 41.77±0.03 |79.81±0.17 79.62±0.21
    platt                |76.86±0.16 76.81±0.20 |41.00±0.16 40.91±0.18 |78.34±0.15 78.28±0.19
    beta                 |74.37±4.38 77.65±0.25 |28.02±14.71 41.58±0.20 |75.29±5.16 79.25±0.27
    empirical            |75.44±0.34 75.62±0.47 |33.85±0.56 32.97±1.32 |76.12±0.35 76.25±0.43
    empirical_cumulative |69.81±1.12 70.51±1.53 |12.66±0.48 11.64±0.77 |70.04±1.07 70.74±1.46
    val_cal              |79.03±0.04 79.04±0.05 |41.99±0.07 41.98±0.03 |80.74±0.04 80.76±0.05

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.05±0.08 79.06±0.08 |41.96±0.02 41.97±0.04 |80.76±0.08 80.77±0.08
    fixed_05             |56.27±14.51 45.11±15.04 |75.71±17.10 88.65±18.24 |65.96±7.68 60.15±8.18
    platt                |78.96±0.13 78.72±0.50 |41.94±0.05 42.43±1.04 |80.67±0.14 80.48±0.40
    beta                 |78.26±1.25 65.61±18.20 |41.50±0.81 59.22±25.76 |79.87±1.42 72.06±10.62
    empirical            |78.17±0.68 78.56±0.60 |41.72±0.20 41.85±0.12 |79.82±0.74 80.24±0.65
    empirical_cumulative |79.12±0.07 79.10±0.09 |42.00±0.04 42.01±0.06 |80.84±0.07 80.82±0.10
    val_cal              |79.13±0.06 79.12±0.08 |42.09±0.11 42.07±0.10 |80.87±0.07 80.85±0.09


## SP-unswt-abl2s-16bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=temporal_3way bits=16 feats=top20 class=binary | mem=BINARY | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  88.76% |   8.42% |  88.81% | r72261 GS best_ce      val_cal
    Best F1 (FPR<10%)   |  88.76% |   8.42% |  88.81% | r72261 GS best_ce      val_cal
    Best F1 (FPR<6%)    |  87.47% |   5.29% |  87.48% | r72261 GS best_ce      empirical_cumulative
    Best F1 (FPR<5%)    |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best F1 (FPR<4%)    |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best F1 (FPR<2%)    |  77.59% |   0.28% |  77.90% | r35879 GS best_fpr     empirical
    Best FPR (any F1)   |  31.01% |   0.00% |  44.94% | r67145 GS best_fpr     platt
    Best FPR (F1>80%)   |  87.47% |   5.29% |  87.48% | r72261 GS best_ce      empirical_cumulative
    Best FPR (F1>90%)   |       — |       — |       — | —
    Best Acc (any FPR)  |  88.76% |   8.42% |  88.81% | r72261 GS best_ce      val_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.07±0.11 79.04±0.14 |41.97±0.02 41.97±0.02 |80.78±0.12 80.75±0.15
    fixed_05             |35.54±0.02 35.54±0.02 |99.97±0.02 99.97±0.02 |55.07±0.01 55.07±0.01
    platt                |79.08±0.06 79.05±0.05 |41.98±0.01 41.99±0.06 |80.80±0.06 80.77±0.05
    beta                 |74.20±7.66 75.82±6.84 |49.93±12.06 46.90±10.88 |77.24±5.22 78.38±4.67
    empirical            |78.11±0.32 78.32±0.34 |41.80±0.06 41.81±0.04 |79.76±0.34 79.98±0.36
    empirical_cumulative |79.20±0.03 79.19±0.10 |42.01±0.02 42.00±0.02 |80.93±0.03 80.92±0.10
    val_cal              |79.19±0.03 79.20±0.10 |42.09±0.09 42.01±0.02 |80.93±0.03 80.92±0.10

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |62.00±19.14 62.00±19.14 |64.20±27.33 64.20±27.33 |69.84±11.33 69.83±11.33
    fixed_05             |60.56±21.68 60.51±21.63 |61.64±33.86 61.76±33.78 |69.10±12.40 69.06±12.36
    platt                |61.82±19.92 61.80±19.90 |49.62±33.21 49.62±33.21 |68.67±12.89 68.65±12.87
    beta                 |60.06±21.21 59.87±21.06 |55.17±40.35 57.64±39.20 |68.28±11.62 68.36±11.67
    empirical            |61.16±18.30 61.11±18.25 |59.03±34.20 59.31±34.00 |68.70±10.10 68.68±10.08
    empirical_cumulative |57.22±22.59 57.10±22.49 |10.70±12.68 10.49±12.46 |62.90±15.50 62.77±15.40
    val_cal              |62.62±20.53 62.60±20.51 |46.60±36.41 46.65±36.41 |69.49±13.45 69.48±13.43

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.07±0.11 79.06±0.15 |41.97±0.02 41.97±0.02 |80.78±0.12 80.77±0.16
    fixed_05             |35.54±0.02 35.54±0.02 |99.97±0.02 99.97±0.02 |55.07±0.01 55.07±0.01
    platt                |79.08±0.06 79.06±0.06 |41.98±0.01 41.99±0.06 |80.80±0.06 80.77±0.06
    beta                 |74.20±7.66 75.56±6.76 |49.93±12.06 47.44±10.73 |77.24±5.22 78.18±4.60
    empirical            |78.11±0.32 78.34±0.36 |41.80±0.06 41.81±0.04 |79.76±0.34 80.00±0.38
    empirical_cumulative |79.20±0.03 79.19±0.10 |42.01±0.02 42.00±0.02 |80.93±0.03 80.92±0.10
    val_cal              |79.19±0.03 79.19±0.10 |42.09±0.09 42.01±0.02 |80.93±0.03 80.92±0.10

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.51±1.58 79.02±0.03 |40.66±4.10 41.96±0.01 |81.12±1.26 80.73±0.03
    fixed_05             |79.66±2.03 78.98±0.02 |40.21±5.59 41.95±0.01 |81.25±1.63 80.69±0.02
    platt                |79.20±2.80 78.26±0.09 |39.43±7.58 41.82±0.01 |80.72±2.36 79.92±0.10
    beta                 |77.86±4.29 77.85±1.87 |34.62±11.57 38.43±7.67 |79.00±4.47 79.23±2.52
    empirical            |75.84±0.61 75.71±0.19 |34.16±11.82 37.62±0.70 |76.83±0.38 76.75±0.23
    empirical_cumulative |74.57±4.57 73.96±0.92 |22.36±6.04 23.77±0.60 |74.64±4.56 74.04±0.95
    val_cal              |80.04±3.07 79.05±0.02 |38.73±10.65 42.12±0.12 |81.60±2.53 80.78±0.02

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.07±0.11 79.04±0.14 |41.97±0.02 41.97±0.02 |80.78±0.12 80.75±0.15
    fixed_05             |35.54±0.02 35.54±0.02 |99.97±0.02 99.97±0.02 |55.07±0.01 55.07±0.01
    platt                |79.08±0.06 79.05±0.05 |41.98±0.01 41.99±0.06 |80.80±0.06 80.77±0.05
    beta                 |74.20±7.66 75.82±6.84 |49.93±12.06 46.90±10.88 |77.24±5.22 78.38±4.67
    empirical            |78.11±0.32 78.32±0.34 |41.80±0.06 41.81±0.04 |79.76±0.34 79.98±0.36
    empirical_cumulative |79.20±0.03 79.19±0.10 |42.01±0.02 42.00±0.02 |80.93±0.03 80.92±0.10
    val_cal              |79.19±0.03 79.20±0.10 |42.09±0.09 42.01±0.02 |80.93±0.03 80.92±0.10


## SP-unswt-abl3s-16bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=temporal_3way bits=16 feats=top20 class=binary | mem=TERNARY | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  88.76% |   8.42% |  88.81% | r72261 GS best_ce      val_cal
    Best F1 (FPR<10%)   |  88.76% |   8.42% |  88.81% | r72261 GS best_ce      val_cal
    Best F1 (FPR<6%)    |  87.47% |   5.29% |  87.48% | r72261 GS best_ce      empirical_cumulative
    Best F1 (FPR<5%)    |  78.33% |   0.86% |  78.57% | r88120 GS best_fpr     empirical
    Best F1 (FPR<4%)    |  78.33% |   0.86% |  78.57% | r88120 GS best_fpr     empirical
    Best F1 (FPR<2%)    |  78.33% |   0.86% |  78.57% | r88120 GS best_fpr     empirical
    Best FPR (any F1)   |  31.01% |   0.00% |  44.94% | r98954 GS best_fpr     empirical_cumulative
    Best FPR (F1>80%)   |  87.08% |   5.05% |  87.09% | r88120 GS best_fpr     empirical_cumulative
    Best FPR (F1>90%)   |       — |       — |       — | —
    Best Acc (any FPR)  |  88.76% |   8.42% |  88.81% | r72261 GS best_ce      val_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.14±0.07 79.05±0.24 |41.98±0.01 41.86±0.33 |80.86±0.07 80.75±0.28
    fixed_05             |35.51±0.00 42.39±14.50 |100.00±0.00 91.63±17.66 |55.06±0.00 58.82±7.94
    platt                |79.15±0.04 79.01±0.25 |41.98±0.01 41.93±0.08 |80.87±0.04 80.72±0.28
    beta                 |74.28±8.37 75.02±7.31 |49.92±13.05 48.70±11.25 |77.39±5.75 77.86±4.90
    empirical            |78.26±0.33 78.41±0.29 |41.76±0.09 41.77±0.07 |79.91±0.35 80.07±0.30
    empirical_cumulative |79.28±0.03 79.28±0.04 |42.01±0.02 42.01±0.02 |81.01±0.04 81.01±0.05
    val_cal              |79.28±0.03 79.29±0.04 |42.03±0.03 42.01±0.02 |81.02±0.03 81.02±0.04

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |62.28±20.91 61.29±19.85 |62.32±30.90 64.93±28.14 |70.28±12.59 69.50±11.67
    fixed_05             |61.89±23.05 60.66±21.77 |58.13±36.82 61.71±33.82 |70.18±13.65 69.21±12.51
    platt                |62.95±22.54 61.28±20.76 |45.58±36.92 50.36±34.22 |69.90±14.93 68.52±13.34
    beta                 |63.33±23.34 60.94±20.94 |52.44±42.06 56.18±38.92 |71.29±14.15 69.02±11.74
    empirical            |60.89±19.42 60.55±19.10 |51.98±42.90 58.87±35.59 |68.62±10.68 68.39±10.46
    empirical_cumulative |59.95±25.35 57.43±22.77 | 6.74±10.24 10.26±12.13 |65.61±18.39 63.10±15.69
    val_cal              |63.86±23.26 62.00±21.31 |40.70±40.92 47.37±37.40 |70.79±15.54 69.26±13.82

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.14±0.07 79.05±0.24 |41.98±0.01 41.86±0.33 |80.86±0.07 80.75±0.28
    fixed_05             |35.51±0.00 42.39±14.50 |100.00±0.00 91.63±17.66 |55.06±0.00 58.82±7.94
    platt                |79.15±0.04 79.01±0.25 |41.98±0.01 41.93±0.08 |80.87±0.04 80.72±0.28
    beta                 |74.28±8.37 75.02±7.31 |49.92±13.05 48.70±11.25 |77.39±5.75 77.86±4.90
    empirical            |78.26±0.33 78.41±0.29 |41.76±0.09 41.77±0.07 |79.91±0.35 80.07±0.30
    empirical_cumulative |79.28±0.03 79.28±0.04 |42.01±0.02 42.01±0.02 |81.01±0.04 81.01±0.05
    val_cal              |79.28±0.03 79.29±0.04 |42.03±0.03 42.01±0.02 |81.02±0.03 81.02±0.04

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.51±1.58 79.03±0.02 |40.65±4.10 41.95±0.01 |81.12±1.26 80.74±0.03
    fixed_05             |79.88±1.95 79.30±0.01 |40.35±5.63 42.16±0.01 |81.50±1.55 81.05±0.00
    platt                |79.59±2.67 78.78±0.03 |39.45±7.59 41.85±0.01 |81.13±2.21 80.47±0.03
    beta                 |78.71±3.92 77.27±2.66 |34.46±11.07 34.95±9.50 |79.84±4.08 78.37±3.47
    empirical            |76.19±0.52 76.30±0.29 |33.14±11.46 36.02±0.43 |77.10±0.34 77.23±0.29
    empirical_cumulative |75.86±4.23 74.90±1.37 |23.33±6.40 24.78±1.09 |76.00±4.21 75.05±1.42
    val_cal              |80.24±3.00 79.33±0.01 |38.76±10.66 42.10±0.02 |81.81±2.46 81.07±0.01

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.14±0.07 79.05±0.24 |41.98±0.01 41.86±0.33 |80.86±0.07 80.75±0.28
    fixed_05             |35.51±0.00 42.39±14.50 |100.00±0.00 91.63±17.66 |55.06±0.00 58.82±7.94
    platt                |79.15±0.04 79.01±0.25 |41.98±0.01 41.93±0.08 |80.87±0.04 80.72±0.28
    beta                 |74.28±8.37 75.02±7.31 |49.92±13.05 48.70±11.25 |77.39±5.75 77.86±4.90
    empirical            |78.26±0.33 78.41±0.29 |41.76±0.09 41.77±0.07 |79.91±0.35 80.07±0.30
    empirical_cumulative |79.28±0.03 79.28±0.04 |42.01±0.02 42.01±0.02 |81.01±0.04 81.01±0.05
    val_cal              |79.28±0.03 79.29±0.04 |42.03±0.03 42.01±0.02 |81.02±0.03 81.02±0.04


## SP-unswt-ablpln-16bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=temporal_3way bits=16 feats=top20 class=binary | mem=PLN | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  88.61% |   8.84% |  88.67% | r98954 GS best_ce      val_cal
    Best F1 (FPR<10%)   |  88.61% |   8.84% |  88.67% | r98954 GS best_ce      val_cal
    Best F1 (FPR<6%)    |  87.17% |   4.81% |  87.17% | r98954 GS best_ce      empirical_cumulative
    Best F1 (FPR<5%)    |  87.17% |   4.81% |  87.17% | r98954 GS best_ce      empirical_cumulative
    Best F1 (FPR<4%)    |  78.61% |   1.05% |  78.82% | r98954 GS best_ce      empirical
    Best F1 (FPR<2%)    |  78.61% |   1.05% |  78.82% | r98954 GS best_ce      empirical
    Best FPR (any F1)   |  31.01% |   0.00% |  44.94% | r98954 GS best_fpr     empirical_cumulative
    Best FPR (F1>80%)   |  87.17% |   4.81% |  87.17% | r98954 GS best_ce      empirical_cumulative
    Best FPR (F1>90%)   |       — |       — |       — | —
    Best Acc (any FPR)  |  88.61% |   8.84% |  88.67% | r98954 GS best_ce      val_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.55±1.10 79.22±0.03 |41.31±2.80 42.30±0.06 |81.22±0.86 80.98±0.02
    fixed_05             |63.87±11.22 62.97±0.35 |24.50±11.64 22.73±0.49 |65.35±9.33 63.12±0.33
    platt                |79.94±2.76 79.25±0.03 |39.43±8.37 42.21±0.05 |81.52±2.28 81.00±0.03
    beta                 |79.89±2.55 79.17±0.03 |40.06±7.58 42.41±0.05 |81.51±2.07 80.94±0.02
    empirical            |78.76±2.85 78.27±0.13 |36.61±10.18 40.85±0.12 |80.03±2.51 79.82±0.14
    empirical_cumulative |78.77±3.88 78.95±0.24 |33.62±13.66 41.73±0.26 |79.98±4.05 80.63±0.28
    val_cal              |80.10±2.78 79.24±0.03 |39.16±9.49 42.18±0.04 |81.68±2.26 80.99±0.03

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |65.11±15.43 67.99±9.25 |52.14±25.69 44.48±11.24 |69.83±9.30 69.39±9.24
    fixed_05             |52.90±11.73 55.58±8.57 |32.30±36.68 16.64±9.46 |58.57±4.80 57.57±7.03
    platt                |65.22±15.44 66.16±13.21 |50.97±25.67 46.59±19.90 |69.73±9.34 68.98±9.45
    beta                 |60.30±16.96 61.39±14.13 |51.21±32.48 52.92±25.83 |66.82±11.67 66.16±11.13
    empirical            |64.91±15.18 64.57±14.43 |51.46±25.84 51.98±23.75 |69.51±8.88 68.68±9.34
    empirical_cumulative |60.54±18.11 61.53±15.04 |19.80±16.73 20.80±15.87 |64.42±13.82 64.43±12.40
    val_cal              |64.77±16.43 68.09±9.23 |41.43±24.02 43.76±10.53 |68.79±11.41 69.36±9.26

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.54±1.12 79.22±0.03 |41.10±3.52 42.29±0.05 |81.20±0.82 80.99±0.02
    fixed_05             |63.63±12.04 62.85±0.44 |24.39±11.66 22.61±0.63 |65.27±9.66 63.01±0.40
    platt                |79.75±2.16 79.25±0.02 |39.34±8.69 42.20±0.03 |81.31±1.67 81.00±0.02
    beta                 |79.75±2.11 79.17±0.03 |40.01±7.71 42.41±0.05 |81.37±1.62 80.94±0.02
    empirical            |78.65±2.63 78.26±0.13 |37.35±7.70 40.87±0.13 |79.92±2.35 79.80±0.14
    empirical_cumulative |78.45±3.14 78.98±0.25 |33.61±13.71 41.78±0.26 |79.65±3.46 80.66±0.29
    val_cal              |79.88±2.09 79.24±0.02 |39.42±8.71 42.17±0.03 |81.46±1.57 80.99±0.02

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 250±71 neurons | 32±0 bits
    GA Neurons  : 203±7 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |80.46±2.02 79.92±1.46 |38.96±5.51 40.48±3.79 |81.95±1.57 81.53±1.13
    fixed_05             |60.36±13.70 58.95±8.91 |18.60±9.78 18.34±9.63 |62.63±10.21 60.81±5.24
    platt                |81.63±3.82 80.88±3.42 |34.66±12.12 36.92±11.15 |82.92±3.08 82.32±2.76
    beta                 |81.54±3.84 80.69±3.23 |35.02±12.04 37.77±9.80 |82.85±3.10 82.17±2.59
    empirical            |80.07±3.58 80.13±3.88 |31.35±15.78 35.42±11.42 |81.21±3.06 81.39±3.29
    empirical_cumulative |81.11±3.32 80.50±2.92 |31.51±16.69 35.07±14.34 |82.30±2.52 81.86±2.21
    val_cal              |81.78±4.08 80.90±3.46 |33.17±14.58 36.39±12.22 |83.03±3.29 82.32±2.77

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.54±1.10 79.22±0.03 |41.33±2.80 42.30±0.06 |81.21±0.86 80.98±0.02
    fixed_05             |63.91±11.22 62.97±0.35 |24.44±11.65 22.73±0.49 |65.40±9.32 63.12±0.33
    platt                |79.94±2.76 79.25±0.03 |39.43±8.37 42.21±0.05 |81.52±2.28 81.00±0.03
    beta                 |79.89±2.55 79.17±0.03 |40.06±7.58 42.41±0.05 |81.51±2.07 80.94±0.02
    empirical            |78.75±2.85 78.27±0.13 |36.57±10.16 40.85±0.12 |80.01±2.51 79.82±0.14
    empirical_cumulative |78.78±3.88 78.95±0.24 |33.63±13.67 41.73±0.26 |79.98±4.05 80.63±0.28
    val_cal              |80.10±2.78 79.24±0.03 |39.16±9.49 42.18±0.04 |81.68±2.26 80.99±0.03


## SP-unswt-ablqsr-16bWb-n10  (10/10 completed)

    dataset=unsw-nb15 split=temporal_3way bits=16 feats=top20 class=binary | mem=QSR | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  90.20% |   5.30% |  90.22% | r25052 GA best_f1      val_cal
    Best F1 (FPR<10%)   |  90.20% |   5.30% |  90.22% | r25052 GA best_f1      val_cal
    Best F1 (FPR<6%)    |  90.20% |   5.30% |  90.22% | r25052 GA best_f1      val_cal
    Best F1 (FPR<5%)    |  90.16% |   4.29% |  90.18% | r35879 GA best_ce      val_cal
    Best F1 (FPR<4%)    |  89.79% |   3.19% |  89.79% | r35879 GA best_ce      fixed_05
    Best F1 (FPR<2%)    |  88.53% |   1.97% |  88.53% | r35879 GS best_fpr     empirical_cumulative
    Best FPR (any F1)   |  63.39% |   0.01% |  65.56% | r58983 GA best_acc     empirical
    Best FPR (F1>80%)   |  80.98% |   0.42% |  81.11% | r72261 GS best_ce      empirical
    Best FPR (F1>90%)   |  90.16% |   4.29% |  90.18% | r35879 GA best_ce      val_cal
    Best Acc (any FPR)  |  90.20% |   5.30% |  90.22% | r25052 GA best_acc     val_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 233±115 neurons | 33±1 bits
    GA Neurons  : 254±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |80.65±2.54 81.19±2.72 |38.14±6.16 36.90±6.64 |82.07±2.06 82.51±2.21
    fixed_05             |81.38±3.76 82.54±4.53 |30.04±19.15 26.24±20.26 |82.57±2.93 83.56±3.65
    platt                |81.76±5.15 82.81±5.51 |32.13±15.07 28.97±16.14 |82.92±4.40 83.82±4.71
    beta                 |80.88±5.84 82.56±5.65 |24.08±15.36 26.95±15.64 |81.61±5.70 83.46±5.14
    empirical            |77.83±4.56 76.01±5.74 |24.49±15.20 20.08±16.85 |78.53±4.33 76.82±5.22
    empirical_cumulative |79.16±6.19 80.72±7.01 | 9.51±4.09  8.62±4.36 |79.19±6.17 80.75±6.99
    val_cal              |82.21±4.89 83.30±5.36 |32.10±16.26 27.83±18.68 |83.45±4.07 84.37±4.47

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 260±89 neurons | 34±1 bits
    GA Neurons  : 242±91 neurons | 32±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.22±2.68 81.77±2.86 |34.27±6.68 35.36±7.05 |83.34±2.17 82.98±2.32
    fixed_05             |83.73±4.07 82.81±4.09 |20.19±19.92 21.98±21.12 |84.46±3.22 83.67±3.22
    platt                |84.78±5.41 83.91±5.59 |23.42±15.80 25.70±16.71 |85.50±4.61 84.77±4.76
    beta                 |84.96±5.11 84.03±5.21 |24.09±15.40 26.97±15.81 |85.71±4.29 84.94±4.36
    empirical            |77.39±5.04 77.68±4.59 |14.42±17.13 18.09±17.41 |78.01±4.81 78.32±4.38
    empirical_cumulative |83.32±6.51 81.81±6.67 | 6.92±4.60  8.92±7.78 |83.33±6.50 81.86±6.62
    val_cal              |85.25±5.22 84.24±5.35 |21.20±18.03 24.91±18.21 |85.97±4.34 85.13±4.46

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 233±115 neurons | 33±1 bits
    GA Neurons  : 254±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |80.65±2.54 81.19±2.72 |38.14±6.16 36.90±6.64 |82.07±2.06 82.51±2.21
    fixed_05             |81.38±3.76 82.54±4.53 |30.04±19.15 26.24±20.26 |82.57±2.93 83.56±3.65
    platt                |81.76±5.15 82.81±5.51 |32.13±15.07 28.97±16.14 |82.92±4.40 83.82±4.71
    beta                 |80.88±5.84 82.56±5.65 |24.08±15.36 26.95±15.64 |81.61±5.70 83.46±5.14
    empirical            |77.83±4.56 76.01±5.74 |24.49±15.20 20.08±16.85 |78.53±4.33 76.82±5.22
    empirical_cumulative |79.16±6.19 80.72±7.01 | 9.51±4.09  8.62±4.36 |79.19±6.17 80.75±6.99
    val_cal              |82.21±4.89 83.30±5.36 |32.10±16.26 27.83±18.68 |83.45±4.07 84.37±4.47

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 200±82 neurons | 34±0 bits
    GA Neurons  : 177±48 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |81.28±2.85 81.08±2.51 |36.63±6.92 37.23±6.17 |82.59±2.32 82.42±2.02
    fixed_05             |82.37±4.31 82.85±4.96 |26.16±20.35 26.29±20.19 |83.40±3.43 83.88±4.09
    platt                |82.83±5.51 82.80±5.49 |29.00±16.05 28.83±16.07 |83.84±4.72 83.80±4.70
    beta                 |82.58±5.66 82.65±5.50 |27.08±15.64 26.34±15.46 |83.48±5.15 83.47±4.92
    empirical            |78.56±4.14 79.85±4.12 |21.35±16.46 20.13±16.52 |79.16±3.94 80.33±3.81
    empirical_cumulative |80.54±7.14 81.46±6.34 | 8.23±3.74  8.88±5.32 |80.58±7.12 81.47±6.33
    val_cal              |83.26±5.28 83.37±5.38 |27.89±18.46 27.48±18.91 |84.32±4.39 84.43±4.49

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 233±115 neurons | 33±1 bits
    GA Neurons  : 254±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |80.65±2.54 81.19±2.72 |38.14±6.16 36.90±6.64 |82.07±2.06 82.51±2.21
    fixed_05             |81.38±3.76 82.54±4.53 |30.04±19.15 26.24±20.26 |82.57±2.93 83.56±3.65
    platt                |81.76±5.15 82.81±5.51 |32.13±15.07 28.97±16.14 |82.92±4.40 83.82±4.71
    beta                 |80.88±5.84 82.56±5.65 |24.08±15.36 26.95±15.64 |81.61±5.70 83.46±5.14
    empirical            |77.83±4.56 76.01±5.74 |24.49±15.20 20.08±16.85 |78.53±4.33 76.82±5.22
    empirical_cumulative |79.16±6.19 80.72±7.01 | 9.51±4.09  8.62±4.36 |79.19±6.17 80.75±6.99
    val_cal              |82.21±4.89 83.30±5.36 |32.10±16.26 27.83±18.68 |83.45±4.07 84.37±4.47


## SP-unswt-bin-16bWb-n30  (10/10 completed)

    dataset=unsw-nb15 split=temporal_3way bits=16 feats=top20 class=binary | mem=QUAD_WEIGHTED (worker default; param absent) | caps 500n/34b | w(ce/acc/f1/fpr)=0.1/0.2/0.35/0.35 | kfold=5x5 gens=250

### Best individual genomes (all phases x genome types x 7 modes)

    Metric              |      F1 |     FPR |     Acc | Source (seed phase genome_type mode)
    --------------------+---------+---------+---------+--------------------------------------
    Best F1 (any FPR)   |  89.35% |   7.59% |  89.40% | r35879 GA best_ce      val_cal
    Best F1 (FPR<10%)   |  89.35% |   7.59% |  89.40% | r35879 GA best_ce      val_cal
    Best F1 (FPR<6%)    |  88.31% |   5.75% |  88.32% | r72261 GA best_ce      empirical_cumulative
    Best F1 (FPR<5%)    |  87.92% |   4.97% |  87.93% | r63749 GA best_ce      empirical_cumulative
    Best F1 (FPR<4%)    |  87.78% |   3.88% |  87.78% | r98954 GA best_ce      empirical_cumulative
    Best F1 (FPR<2%)    |  83.71% |   1.65% |  83.75% | r25052 GA best_ce      empirical
    Best FPR (any F1)   |  75.29% |   0.03% |  75.78% | r98954 GA best_fpr     empirical_cumulative
    Best FPR (F1>80%)   |  82.60% |   0.81% |  82.67% | r35879 GA best_ce      empirical
    Best FPR (F1>90%)   |       — |       — |       — | —
    Best Acc (any FPR)  |  89.35% |   7.59% |  89.40% | r35879 GA best_ce      val_cal

### best_f1  (runs: GS 10 | GA 10)
    Grid Search : 136±173 neurons | 32±2 bits
    GA Neurons  : 49±80 neurons | 29±5 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.73±0.84 84.01±1.16 |29.02±1.96 28.61±2.90 |84.43±0.74 84.70±1.01
    fixed_05             |84.29±1.28 85.22±1.43 |27.06±3.37 24.14±4.21 |84.89±1.11 85.69±1.27
    platt                |85.91±1.53 85.98±1.73 |19.68±3.21 18.66±4.16 |86.20±1.47 86.24±1.64
    beta                 |84.95±2.31 85.06±2.43 |20.36±9.51 22.03±7.99 |85.34±2.06 85.48±2.17
    empirical            |83.93±2.16 84.68±2.08 |16.13±10.91 18.50±11.26 |84.23±2.13 85.05±1.97
    empirical_cumulative |86.01±2.48 85.12±3.39 | 9.01±2.97  8.11±2.76 |86.05±2.49 85.16±3.36
    val_cal              |86.66±1.95 86.54±1.87 |13.22±5.56 13.58±7.08 |86.78±1.83 86.70±1.73

### best_fpr  (runs: GS 10 | GA 10)
    Grid Search : 179±177 neurons | 20±15 bits
    GA Neurons  : 64±146 neurons | 17±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |77.09±7.49 76.65±7.41 |39.99±13.19 38.35±16.43 |78.77±6.31 78.35±6.26
    fixed_05             |79.93±5.68 78.85±6.39 |21.93±5.88 25.46±10.12 |80.21±5.88 79.32±6.47
    platt                |80.75±6.51 79.44±7.03 |19.01±4.94 23.23±7.88 |80.92±6.59 79.74±7.04
    beta                 |81.72±6.36 79.84±6.89 |13.22±5.98 14.81±9.82 |81.85±6.41 80.07±6.79
    empirical            |78.98±5.98 78.22±10.82 |14.90±21.26 27.77±22.86 |79.67±5.03 79.48±9.06
    empirical_cumulative |81.84±6.33 81.51±6.60 | 3.80±2.72  4.31±3.48 |82.03±6.15 81.74±6.34
    val_cal              |82.74±6.23 82.16±5.98 | 6.98±4.98  6.28±4.09 |82.86±6.17 82.30±5.89

### best_acc  (runs: GS 10 | GA 10)
    Grid Search : 83±128 neurons | 32±2 bits
    GA Neurons  : 49±80 neurons | 29±5 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.24±1.20 84.17±1.13 |30.29±3.12 28.17±2.94 |84.02±1.01 84.83±0.98
    fixed_05             |83.71±1.62 85.27±1.42 |28.58±4.42 24.02±4.20 |84.41±1.38 85.74±1.26
    platt                |84.93±2.74 86.00±1.70 |20.88±3.92 18.59±4.10 |85.24±2.68 86.25±1.62
    beta                 |84.17±2.95 85.00±2.49 |21.05±9.12 19.96±9.11 |84.56±2.79 85.37±2.28
    empirical            |84.41±2.16 84.71±2.07 |17.30±10.12 18.45±11.23 |84.71±1.99 85.08±1.97
    empirical_cumulative |84.93±3.37 85.17±3.36 | 9.40±2.97  7.96±2.77 |84.96±3.38 85.21±3.33
    val_cal              |85.84±2.55 86.57±1.80 |16.08±8.90 13.86±6.87 |86.08±2.24 86.73±1.66

### best_ce  (runs: GS 10 | GA 10)
    Grid Search : 400±105 neurons | 33±1 bits
    GA Neurons  : 336±100 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.02±0.25 84.11±0.41 |28.57±0.73 28.48±1.37 |84.70±0.21 84.78±0.33
    fixed_05             |85.24±0.22 84.63±0.31 |24.83±0.58 26.85±0.89 |85.73±0.20 85.21±0.27
    platt                |86.98±0.20 87.24±0.27 |18.16±0.44 18.01±0.56 |87.22±0.19 87.48±0.25
    beta                 |88.02±0.11 88.41±0.38 |13.19±0.27 12.48±1.16 |88.14±0.11 88.52±0.37
    empirical            |79.55±1.64 82.46±1.59 | 1.23±0.48  1.52±0.52 |79.73±1.56 82.53±1.55
    empirical_cumulative |87.30±0.36 87.51±0.49 | 5.23±0.45  4.11±0.89 |87.30±0.36 87.52±0.49
    val_cal              |88.61±0.13 88.82±0.28 | 8.96±0.61  8.81±0.99 |88.66±0.13 88.87±0.28

### best_fitness  (runs: GS 10 | GA 10)
    Grid Search : 136±173 neurons | 32±2 bits
    GA Neurons  : 49±80 neurons | 29±5 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.73±0.84 84.01±1.16 |29.02±1.96 28.61±2.90 |84.43±0.74 84.70±1.01
    fixed_05             |84.29±1.28 85.22±1.43 |27.06±3.37 24.14±4.21 |84.89±1.11 85.69±1.27
    platt                |85.91±1.53 85.98±1.73 |19.68±3.21 18.66±4.16 |86.20±1.47 86.24±1.64
    beta                 |84.95±2.31 85.06±2.43 |20.36±9.51 22.03±7.99 |85.34±2.06 85.48±2.17
    empirical            |83.93±2.16 84.68±2.08 |16.13±10.91 18.50±11.26 |84.23±2.13 85.05±1.97
    empirical_cumulative |86.01±2.48 85.12±3.39 | 9.01±2.97  8.11±2.76 |86.05±2.49 85.16±3.36
    val_cal              |86.66±1.95 86.54±1.87 |13.22±5.56 13.58±7.08 |86.78±1.83 86.70±1.73


---

# =====================================================================
# SECTION 3 — PER-CONFIG LEADERBOARD (best F1 / best FPR / best Acc as TRIPLES)
# =====================================================================

Each "best" is the single best POINT over (runs x 2 phases x 5 genome types x 7 modes)
in that config cell, printed with its other two metrics and its identity. `COHORT` is the
honest central number for the same cell (GA x best_f1 x val_cal, mean+/-std over runs);
best-minus-COHORT is the best-of-N inflation. n = distinct completed runs (seeds).
Note the search space per cell is n x 70 points, so even n=10 cells carry real best-of-N
inflation — read `COHORT` as the claim and the best row as the ceiling.

## 3A. Protocol v2 (_3way) — SP100 + SP ablations

## CIC-IoT-2023 neto-subsample random (3way, Protocol v2)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-ciciot-abl2s-96bWc-n10                  | 10 | F1        |  90.91 |  15.70 |  95.63 | r57826 GS best_f1 empirical_cumulative
                                               |    | FPR       |  65.94 |   0.00 |  72.77 | r98273 GS best_ce empirical
                                               |    | Acc       |  90.78 |  17.96 |  95.65 | r57826 GS best_f1 beta
                                               |    | F1|FPR<5  |  81.74 |   1.04 |  88.49 | r98273 GS best_ce fixed_05
                                               |    | F1|FPR<4  |  81.74 |   1.04 |  88.49 | r98273 GS best_ce fixed_05
                                               |    | F1|FPR<2  |  81.74 |   1.04 |  88.49 | r98273 GS best_ce fixed_05
                                               |    | COHORT    |  77.70 |  44.83 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±1.49, FPR ±8.69
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-ciciot-abl3s-96bWc-n10                  | 10 | F1        |  90.87 |  15.72 |  95.61 | r29083 GS best_f1 train_cal
                                               |    | FPR       |  66.01 |   0.00 |  72.86 | r47707 GS best_ce empirical
                                               |    | Acc       |  90.86 |  15.96 |  95.61 | r29083 GS best_f1 val_cal
                                               |    | F1|FPR<5  |  81.93 |   0.85 |  88.63 | r29083 GS best_f1 fixed_05
                                               |    | F1|FPR<4  |  81.93 |   0.85 |  88.63 | r29083 GS best_f1 fixed_05
                                               |    | F1|FPR<2  |  81.93 |   0.85 |  88.63 | r29083 GS best_f1 fixed_05
                                               |    | COHORT    |  81.34 |  36.31 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±1.29, FPR ±6.04
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-ciciot-ablpln-96bWc-n10                 | 10 | F1        |  92.65 |  12.75 |  96.47 | r98273 GA best_ce train_cal
                                               |    | FPR       |  66.36 |   0.00 |  73.26 | r79803 GS best_ce empirical
                                               |    | Acc       |  92.62 |  13.70 |  96.48 | r98273 GA best_ce empirical
                                               |    | F1|FPR<5  |  82.04 |   0.92 |  88.73 | r56462 GS best_ce fixed_05
                                               |    | F1|FPR<4  |  82.04 |   0.92 |  88.73 | r56462 GS best_ce fixed_05
                                               |    | F1|FPR<2  |  82.04 |   0.92 |  88.73 | r56462 GS best_ce fixed_05
                                               |    | COHORT    |  91.91 |  13.64 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.58, FPR ±1.20
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-ciciot-ablqsr-96bWc-n10                 | 10 | F1        |  92.69 |  10.46 |  96.41 | r24530 GA best_acc val_cal
                                               |    | FPR       |  69.78 |   0.01 |  77.07 | r29083 GS best_acc empirical
                                               |    | Acc       |  92.59 |  12.29 |  96.42 | r24530 GA best_acc beta
                                               |    | F1|FPR<5  |  90.66 |   4.52 |  95.02 | r42704 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  89.70 |   3.73 |  94.39 | r24530 GA best_fpr fixed_05
                                               |    | F1|FPR<2  |  81.93 |   0.85 |  88.63 | r29083 GS best_acc fixed_05
                                               |    | COHORT    |  92.05 |  11.28 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.37, FPR ±0.83
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-ciciot-bin-96bWc-n30                    | 10 | F1        |  92.99 |   7.76 |  96.49 | r98273 GA best_acc platt
                                               |    | FPR       |  70.73 |   0.00 |  78.08 | r42704 GA best_acc empirical
                                               |    | Acc       |  92.88 |  10.01 |  96.50 | r57826 GA best_acc train_cal
                                               |    | F1|FPR<5  |  92.58 |   4.76 |  96.17 | r29083 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  88.25 |   0.89 |  93.30 | r98273 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  88.25 |   0.89 |  93.30 | r98273 GA best_ce fixed_05
                                               |    | COHORT    |  92.82 |   8.66 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.11, FPR ±1.03
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP100-ciciot-quad-96bWc                    | 11 | F1        |  93.35 |   7.50 |  96.69 | r79803 GA best_acc train_cal
                                               |    | FPR       |  70.59 |   0.00 |  77.92 | r42704 GS best_acc empirical
                                               |    | Acc       |  93.27 |   9.25 |  96.69 | r79803 GA best_f1 beta
                                               |    | F1|FPR<5  |  93.08 |   4.91 |  96.46 | r79803 GA best_acc empirical_cumulative
                                               |    | F1|FPR<4  |  90.35 |   2.04 |  94.72 | r61231 GA best_acc empirical
                                               |    | F1|FPR<2  |  87.86 |   0.89 |  93.04 | r79803 GA best_ce fixed_05
                                               |    | COHORT    |  92.88 |   8.49 |    --- | GA best_f1 val_cal mean±std over n=11: F1 ±0.23, FPR ±0.64
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## CICIDS2017 random (3way, Protocol v2)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-cicids-abl2big-96bWa-n10                | 10 | F1        |  99.15 |   0.59 |  99.46 | r43932 GA best_acc train_cal
                                               |    | FPR       |  44.54 |   0.00 |  80.32 | r26177 GS best_fpr platt
                                               |    | Acc       |  99.15 |   0.59 |  99.46 | r43932 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  99.15 |   0.59 |  99.46 | r43932 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.15 |   0.59 |  99.46 | r43932 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.15 |   0.59 |  99.46 | r43932 GA best_acc train_cal
                                               |    | COHORT    |  99.13 |   0.60 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.01, FPR ±0.01
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-cicids-abl2s-96bWa-n10                  | 10 | F1        |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | FPR       |  98.50 |   0.05 |  99.07 | r84914 GS best_ce empirical
                                               |    | Acc       |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | F1|FPR<5  |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | F1|FPR<4  |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | F1|FPR<2  |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | COHORT    |  99.08 |   0.62 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.03, FPR ±0.03
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-cicids-abl3s-96bWa-n10                  | 10 | F1        |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | FPR       |  44.54 |   0.00 |  80.32 | r62982 GS best_fpr train_cal
                                               |    | Acc       |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | F1|FPR<5  |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | F1|FPR<4  |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | F1|FPR<2  |  99.29 |   0.31 |  99.55 | r65122 GS best_ce train_cal
                                               |    | COHORT    |  99.11 |   0.61 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.02, FPR ±0.02
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-cicids-ablpln-96bWa-n10                 | 10 | F1        |  99.35 |   0.22 |  99.59 | r38183 GA best_acc train_cal
                                               |    | FPR       |  78.71 |   0.02 |  89.48 | r80829 GA best_fpr fixed_05
                                               |    | Acc       |  99.35 |   0.22 |  99.59 | r38183 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  99.35 |   0.22 |  99.59 | r38183 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.35 |   0.22 |  99.59 | r38183 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.35 |   0.22 |  99.59 | r38183 GA best_acc train_cal
                                               |    | COHORT    |  99.27 |   0.28 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.03, FPR ±0.04
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-cicids-ablqsr-96bWa-n10                 | 10 | F1        |  99.45 |   0.23 |  99.65 | r26177 GA best_acc train_cal
                                               |    | FPR       |  96.69 |   0.04 |  97.99 | r80829 GA best_acc empirical
                                               |    | Acc       |  99.45 |   0.23 |  99.65 | r26177 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.45 |   0.23 |  99.65 | r26177 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.45 |   0.23 |  99.65 | r26177 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.45 |   0.23 |  99.65 | r26177 GA best_acc train_cal
                                               |    | COHORT    |  99.33 |   0.25 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.06, FPR ±0.05
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-cicids-bin-96bWa-n30                    | 10 | F1        |  99.64 |   0.08 |  99.77 | r84914 GA best_acc train_cal
                                               |    | FPR       |  98.14 |   0.04 |  98.85 | r26177 GS best_acc empirical
                                               |    | Acc       |  99.64 |   0.08 |  99.77 | r84914 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.64 |   0.08 |  99.77 | r84914 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.64 |   0.08 |  99.77 | r84914 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.64 |   0.08 |  99.77 | r84914 GA best_acc train_cal
                                               |    | COHORT    |  99.59 |   0.09 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.04, FPR ±0.02
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP100-cicids-quad-96bWa                    | 11 | F1        |  99.63 |   0.08 |  99.77 | r66758 GA best_acc empirical_cumulative
                                               |    | FPR       |  78.59 |   0.04 |  89.43 | r37040 GS best_acc fixed_05
                                               |    | Acc       |  99.63 |   0.08 |  99.77 | r66758 GA best_f1 empirical_cumulative
                                               |    | F1|FPR<5  |  99.63 |   0.08 |  99.77 | r66758 GA best_acc empirical_cumulative
                                               |    | F1|FPR<4  |  99.63 |   0.08 |  99.77 | r66758 GA best_acc empirical_cumulative
                                               |    | F1|FPR<2  |  99.63 |   0.08 |  99.77 | r66758 GA best_acc empirical_cumulative
                                               |    | COHORT    |  99.54 |   0.14 |    --- | GA best_f1 val_cal mean±std over n=11: F1 ±0.05, FPR ±0.05
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## UNSW-NB15 random (3way, Protocol v2)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswr-abl2big-64bWb-n10                 | 10 | F1        |  93.50 |   1.12 |  98.92 | r49648 GA best_acc train_cal
                                               |    | FPR       |  49.03 |   0.00 |  96.19 | r32732 GS best_fpr train_cal
                                               |    | Acc       |  93.50 |   1.12 |  98.92 | r49648 GA best_acc train_cal
                                               |    | F1|FPR<5  |  93.50 |   1.12 |  98.92 | r49648 GA best_acc train_cal
                                               |    | F1|FPR<4  |  93.50 |   1.12 |  98.92 | r49648 GA best_acc train_cal
                                               |    | F1|FPR<2  |  93.50 |   1.12 |  98.92 | r49648 GA best_acc train_cal
                                               |    | COHORT    |  93.47 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.03, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswr-abl2s-64bWb-n10                   | 10 | F1        |  93.52 |   1.12 |  98.92 | r32732 GS best_ce fixed_05
                                               |    | FPR       |  49.03 |   0.00 |  96.19 | r63890 GS best_fpr platt
                                               |    | Acc       |  93.52 |   1.12 |  98.92 | r32732 GS best_ce fixed_05
                                               |    | F1|FPR<5  |  93.52 |   1.12 |  98.92 | r32732 GS best_ce fixed_05
                                               |    | F1|FPR<4  |  93.52 |   1.12 |  98.92 | r32732 GS best_ce fixed_05
                                               |    | F1|FPR<2  |  93.52 |   1.12 |  98.92 | r32732 GS best_ce fixed_05
                                               |    | COHORT    |  93.47 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.03, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswr-abl3s-64bWb-n10                   | 10 | F1        |  93.52 |   1.12 |  98.92 | r63890 GA best_f1 fixed_05
                                               |    | FPR       |  49.03 |   0.00 |  96.19 | r32732 GS best_fpr train_cal
                                               |    | Acc       |  93.52 |   1.12 |  98.92 | r32732 GS best_ce fixed_05
                                               |    | F1|FPR<5  |  93.52 |   1.12 |  98.92 | r63890 GA best_f1 fixed_05
                                               |    | F1|FPR<4  |  93.52 |   1.12 |  98.92 | r63890 GA best_f1 fixed_05
                                               |    | F1|FPR<2  |  93.52 |   1.12 |  98.92 | r63890 GA best_f1 fixed_05
                                               |    | COHORT    |  93.50 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.03, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswr-ablpln-64bWb-n10                  | 10 | F1        |  93.71 |   1.02 |  98.97 | r63890 GS best_f1 train_cal
                                               |    | FPR       |  49.48 |   0.00 |  96.20 | r10596 GA best_fpr beta
                                               |    | Acc       |  93.67 |   0.95 |  98.98 | r20361 GS best_ce platt
                                               |    | F1|FPR<5  |  93.71 |   1.02 |  98.97 | r63890 GS best_f1 train_cal
                                               |    | F1|FPR<4  |  93.71 |   1.02 |  98.97 | r63890 GS best_f1 train_cal
                                               |    | F1|FPR<2  |  93.71 |   1.02 |  98.97 | r63890 GS best_f1 train_cal
                                               |    | COHORT    |  93.52 |   1.04 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.10, FPR ±0.04
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswr-ablqsr-64bWb-n10                  | 10 | F1        |  94.48 |   0.52 |  99.17 | r49648 GA best_acc beta
                                               |    | FPR       |  74.50 |   0.00 |  97.47 | r49337 GS best_acc empirical
                                               |    | Acc       |  94.40 |   0.45 |  99.17 | r10596 GA best_acc train_cal
                                               |    | F1|FPR<5  |  94.48 |   0.52 |  99.17 | r49648 GA best_acc beta
                                               |    | F1|FPR<4  |  94.48 |   0.52 |  99.17 | r49648 GA best_acc beta
                                               |    | F1|FPR<2  |  94.48 |   0.52 |  99.17 | r49648 GA best_acc beta
                                               |    | COHORT    |  94.33 |   0.62 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.07, FPR ±0.07
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswr-bin-64bWb-n30                     | 10 | F1        |  93.95 |   0.74 |  99.06 | r49648 GA best_fitness train_cal
                                               |    | FPR       |  81.58 |   0.06 |  97.96 | r54070 GA best_fpr empirical_cumulative
                                               |    | Acc       |  93.95 |   0.74 |  99.06 | r49648 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  93.95 |   0.74 |  99.06 | r49648 GA best_fitness train_cal
                                               |    | F1|FPR<4  |  93.95 |   0.74 |  99.06 | r49648 GA best_fitness train_cal
                                               |    | F1|FPR<2  |  93.95 |   0.74 |  99.06 | r49648 GA best_fitness train_cal
                                               |    | COHORT    |  93.54 |   1.08 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.15, FPR ±0.12
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP100-unswr-qsr-64bWb                      | 10 | F1        |  94.56 |   0.61 |  99.17 | r32732 GA best_acc val_cal
                                               |    | FPR       |  75.57 |   0.00 |  97.54 | r49648 GA best_ce empirical
                                               |    | Acc       |  94.53 |   0.52 |  99.18 | r22224 GA best_f1 platt
                                               |    | F1|FPR<5  |  94.56 |   0.61 |  99.17 | r32732 GA best_acc val_cal
                                               |    | F1|FPR<4  |  94.56 |   0.61 |  99.17 | r32732 GA best_acc val_cal
                                               |    | F1|FPR<2  |  94.56 |   0.61 |  99.17 | r32732 GA best_acc val_cal
                                               |    | COHORT    |  94.42 |   0.61 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.09, FPR ±0.06
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP100-unswr-quad-64bWb                     | 11 | F1        |  93.93 |   0.72 |  99.06 | r32732 GA best_f1 train_cal
                                               |    | FPR       |  83.84 |   0.12 |  98.13 | r32732 GA best_fpr train_cal
                                               |    | Acc       |  93.93 |   0.72 |  99.06 | r32732 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  93.93 |   0.72 |  99.06 | r32732 GA best_f1 train_cal
                                               |    | F1|FPR<4  |  93.93 |   0.72 |  99.06 | r32732 GA best_f1 train_cal
                                               |    | F1|FPR<2  |  93.93 |   0.72 |  99.06 | r32732 GA best_f1 train_cal
                                               |    | COHORT    |  93.56 |   1.06 |    --- | GA best_f1 val_cal mean±std over n=11: F1 ±0.13, FPR ±0.12
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## UNSW-NB15 temporal (3way, Protocol v2)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswt-abl2big-16bWb-n10                 | 10 | F1        |  79.21 |  42.09 |  80.95 | r88120 GA best_acc val_cal
                                               |    | FPR       |  31.01 |   0.00 |  44.94 | r98954 GS best_fpr empirical_cumulative
                                               |    | Acc       |  79.21 |  42.09 |  80.95 | r88120 GA best_acc val_cal
                                               |    | F1|FPR<5  |  77.59 |   0.28 |  77.90 | r35879 GS best_fpr empirical
                                               |    | F1|FPR<4  |  77.59 |   0.28 |  77.90 | r35879 GS best_fpr empirical
                                               |    | F1|FPR<2  |  77.59 |   0.28 |  77.90 | r35879 GS best_fpr empirical
                                               |    | COHORT    |  79.12 |  42.07 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.08, FPR ±0.10
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswt-abl2s-16bWb-n10                   | 10 | F1        |  88.76 |   8.42 |  88.81 | r72261 GS best_ce val_cal
                                               |    | FPR       |  31.01 |   0.00 |  44.94 | r67145 GS best_fpr platt
                                               |    | Acc       |  88.76 |   8.42 |  88.81 | r72261 GS best_ce val_cal
                                               |    | F1|FPR<5  |  77.59 |   0.28 |  77.90 | r35879 GS best_fpr empirical
                                               |    | F1|FPR<4  |  77.59 |   0.28 |  77.90 | r35879 GS best_fpr empirical
                                               |    | F1|FPR<2  |  77.59 |   0.28 |  77.90 | r35879 GS best_fpr empirical
                                               |    | COHORT    |  79.20 |  42.01 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.10, FPR ±0.02
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswt-abl3s-16bWb-n10                   | 10 | F1        |  88.76 |   8.42 |  88.81 | r72261 GS best_ce val_cal
                                               |    | FPR       |  31.01 |   0.00 |  44.94 | r98954 GS best_fpr empirical_cumulative
                                               |    | Acc       |  88.76 |   8.42 |  88.81 | r72261 GS best_ce val_cal
                                               |    | F1|FPR<5  |  78.33 |   0.86 |  78.57 | r88120 GS best_fpr empirical
                                               |    | F1|FPR<4  |  78.33 |   0.86 |  78.57 | r88120 GS best_fpr empirical
                                               |    | F1|FPR<2  |  78.33 |   0.86 |  78.57 | r88120 GS best_fpr empirical
                                               |    | COHORT    |  79.29 |  42.01 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.04, FPR ±0.02
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswt-ablpln-16bWb-n10                  | 10 | F1        |  88.61 |   8.84 |  88.67 | r98954 GS best_ce val_cal
                                               |    | FPR       |  31.11 |   0.00 |  44.99 | r98954 GA best_fpr empirical_cumulative
                                               |    | Acc       |  88.61 |   8.84 |  88.67 | r98954 GS best_ce val_cal
                                               |    | F1|FPR<5  |  87.17 |   4.81 |  87.17 | r98954 GS best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  78.61 |   1.05 |  78.82 | r98954 GS best_ce empirical
                                               |    | F1|FPR<2  |  78.61 |   1.05 |  78.82 | r98954 GS best_ce empirical
                                               |    | COHORT    |  79.24 |  42.18 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±0.03, FPR ±0.04
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswt-ablqsr-16bWb-n10                  | 10 | F1        |  90.20 |   5.30 |  90.22 | r25052 GA best_f1 val_cal
                                               |    | FPR       |  63.39 |   0.01 |  65.56 | r58983 GA best_acc empirical
                                               |    | Acc       |  90.20 |   5.30 |  90.22 | r25052 GA best_acc val_cal
                                               |    | F1|FPR<5  |  90.16 |   4.29 |  90.18 | r35879 GA best_ce val_cal
                                               |    | F1|FPR<4  |  89.79 |   3.19 |  89.79 | r35879 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  88.53 |   1.97 |  88.53 | r35879 GS best_fpr empirical_cumulative
                                               |    | COHORT    |  83.30 |  27.83 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±5.36, FPR ±18.68
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswt-bin-16bWb-n30                     | 10 | F1        |  89.35 |   7.59 |  89.40 | r35879 GA best_ce val_cal
                                               |    | FPR       |  75.29 |   0.03 |  75.78 | r98954 GA best_fpr empirical_cumulative
                                               |    | Acc       |  89.35 |   7.59 |  89.40 | r35879 GA best_ce val_cal
                                               |    | F1|FPR<5  |  87.92 |   4.97 |  87.93 | r63749 GA best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  87.78 |   3.88 |  87.78 | r98954 GA best_ce empirical_cumulative
                                               |    | F1|FPR<2  |  83.71 |   1.65 |  83.75 | r25052 GA best_ce empirical
                                               |    | COHORT    |  86.54 |  13.58 |    --- | GA best_f1 val_cal mean±std over n=10: F1 ±1.87, FPR ±7.08
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    SP-unswt-mcsmoke-16bWb                     |  1 | (no final validation_summaries rows)
    SP100-unswt-quad-16bWb                     | 11 | F1        |  89.32 |  10.60 |  89.41 | r35879 GS best_acc val_cal
                                               |    | FPR       |  74.83 |   0.06 |  75.36 | r63749 GA best_fpr empirical_cumulative
                                               |    | Acc       |  89.32 |  10.60 |  89.41 | r35879 GS best_acc val_cal
                                               |    | F1|FPR<5  |  88.44 |   4.60 |  88.45 | r63749 GA best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  87.68 |   3.34 |  87.68 | r25052 GA best_ce empirical_cumulative
                                               |    | F1|FPR<2  |  83.40 |   1.91 |  83.44 | r77715 GA best_ce empirical
                                               |    | COHORT    |  86.96 |  13.19 |    --- | GA best_f1 val_cal mean±std over n=11: F1 ±1.51, FPR ±4.88
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## 3B. LEGACY 2-way (XDS) — val_cal is an ORACLE here; do NOT rank against 3A

## CIC-IoT-2023 neto-subsample random (2way, LEGACY)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-16b-Wa-C35-250n100b-OI |  1 | F1        |  90.43 |  10.99 |  95.15 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  81.28 |   1.04 |  88.11 | r45211 GS best_acc fixed_05
                                               |    | Acc       |  90.33 |  14.19 |  95.23 | r45211 GA best_f1 beta
                                               |    | F1|FPR<5  |  83.94 |   1.11 |  90.22 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  83.94 |   1.11 |  90.22 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  83.94 |   1.11 |  90.22 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  90.43 |  10.99 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-16b-Wa-C35-500n34b-OI |  1 | F1        |  90.29 |  12.23 |  95.12 | r45211 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  80.75 |   0.92 |  87.67 | r45211 GA best_f1 fixed_05
                                               |    | Acc       |  90.14 |  15.16 |  95.17 | r45211 GA best_f1 beta
                                               |    | F1|FPR<5  |  80.89 |   0.97 |  87.79 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  80.89 |   0.97 |  87.79 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  80.89 |   0.97 |  87.79 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  90.28 |  12.51 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-16b-Wb-C35-250n100b-OI |  1 | F1        |  90.17 |  11.88 |  95.04 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  82.72 |   0.78 |  89.26 | r45211 GA best_fpr fixed_05
                                               |    | Acc       |  90.06 |  13.99 |  95.07 | r45211 GA best_acc beta
                                               |    | F1|FPR<5  |  88.71 |   4.87 |  93.80 | r45211 GA best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  84.65 |   2.47 |  90.85 | r45211 GS best_fpr empirical_cumulative
                                               |    | F1|FPR<2  |  83.39 |   1.08 |  89.80 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  90.17 |  11.88 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-16b-Wb-C35-500n34b-OI |  1 | F1        |  90.09 |  12.44 |  95.02 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  79.46 |   0.81 |  86.57 | r45211 GS best_acc fixed_05
                                               |    | Acc       |  89.90 |  15.73 |  95.06 | r45211 GA best_acc beta
                                               |    | F1|FPR<5  |  82.06 |   3.45 |  88.94 | r45211 GS best_fpr beta
                                               |    | F1|FPR<4  |  82.06 |   3.45 |  88.94 | r45211 GS best_fpr beta
                                               |    | F1|FPR<2  |  80.78 |   1.06 |  87.70 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  90.09 |  12.44 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-16b-Wc-C35-250n100b-OI |  1 | F1        |  90.48 |  10.58 |  95.16 | r45211 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  83.51 |   0.99 |  89.89 | r45211 GA best_fpr fixed_05
                                               |    | Acc       |  90.28 |  14.25 |  95.21 | r45211 GA best_acc beta
                                               |    | F1|FPR<5  |  84.03 |   1.03 |  90.29 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  84.03 |   1.03 |  90.29 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  84.03 |   1.03 |  90.29 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  90.45 |  10.54 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-16b-Wc-C35-500n34b-OI |  1 | F1        |  90.32 |  13.24 |  95.19 | r45211 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  79.47 |   0.93 |  86.59 | r45211 GS best_fpr fixed_05
                                               |    | Acc       |  90.29 |  13.81 |  95.20 | r45211 GA best_ce beta
                                               |    | F1|FPR<5  |  82.53 |   1.00 |  89.12 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  82.53 |   1.00 |  89.12 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  82.53 |   1.00 |  89.12 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  90.30 |  12.90 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-32b-Wa-C35-250n100b-OI |  1 | F1        |  91.39 |  11.19 |  95.71 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  82.31 |   0.89 |  88.94 | r45211 GA best_fpr fixed_05
                                               |    | Acc       |  91.37 |  11.42 |  95.71 | r45211 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  83.59 |   0.94 |  89.94 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  83.59 |   0.94 |  89.94 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  83.59 |   0.94 |  89.94 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  91.39 |  11.19 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-32b-Wa-C35-500n34b-OI |  1 | F1        |  91.28 |  10.59 |  95.63 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  80.53 |   0.83 |  87.48 | r45211 GA best_acc fixed_05
                                               |    | Acc       |  91.23 |  11.92 |  95.65 | r45211 GA best_acc train_cal
                                               |    | F1|FPR<5  |  81.24 |   1.06 |  88.09 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  81.24 |   1.06 |  88.09 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  81.24 |   1.06 |  88.09 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  91.28 |  10.59 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-32b-Wb-C35-250n100b-OI |  1 | F1        |  91.21 |  10.14 |  95.57 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  65.50 |   0.00 |  72.27 | r45211 GA best_fpr empirical_cumulative
                                               |    | Acc       |  91.03 |  13.46 |  95.60 | r45211 GA best_acc beta
                                               |    | F1|FPR<5  |  88.51 |   4.89 |  93.68 | r45211 GA best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  82.60 |   0.88 |  89.17 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  82.60 |   0.88 |  89.17 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  91.20 |  10.21 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-32b-Wb-C35-500n34b-OI |  1 | F1        |  90.94 |  11.21 |  95.46 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  71.88 |   0.24 |  79.31 | r45211 GA best_fpr empirical_cumulative
                                               |    | Acc       |  90.73 |  14.90 |  95.49 | r45211 GA best_f1 empirical
                                               |    | F1|FPR<5  |  81.59 |   1.35 |  88.39 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  81.59 |   1.35 |  88.39 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  81.59 |   1.35 |  88.39 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  90.94 |  11.21 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-32b-Wc-C35-250n100b-OI |  1 | F1        |  91.61 |  10.17 |  95.80 | r45211 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  79.73 |   0.90 |  86.81 | r45211 GS best_acc fixed_05
                                               |    | Acc       |  91.56 |  10.95 |  95.80 | r45211 GA best_ce platt
                                               |    | F1|FPR<5  |  84.83 |   0.98 |  90.89 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  84.83 |   0.98 |  90.89 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  84.83 |   0.98 |  90.89 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  91.46 |  10.63 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-32b-Wc-C35-500n34b-OI |  1 | F1        |  91.30 |  11.02 |  95.66 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  80.93 |   0.81 |  87.81 | r45211 GA best_acc fixed_05
                                               |    | Acc       |  91.22 |  12.27 |  95.66 | r45211 GA best_acc train_cal
                                               |    | F1|FPR<5  |  81.68 |   0.87 |  88.43 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  81.68 |   0.87 |  88.43 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  81.68 |   0.87 |  88.43 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  91.30 |  11.02 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-64b-Wa-C35-250n100b-OI |  3 | F1        |  92.65 |   9.01 |  96.35 | r45211 GA best_f1 train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  84.50 |   0.82 |  90.63 | r45211 GA best_acc fixed_05
                                               |    | Acc       |  92.65 |   9.30 |  96.35 | r45211 GA best_f1 platt
                                               |    | F1|FPR<5  |  85.17 |   0.89 |  91.13 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  85.17 |   0.89 |  91.13 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  85.17 |   0.89 |  91.13 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  92.61 |   8.83 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.05, FPR ±0.23
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-64b-Wa-C35-500n34b-OI |  2 | F1        |  92.45 |  10.28 |  96.27 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  81.44 |   0.68 |  88.22 | r45211 GA best_fpr fixed_05
                                               |    | Acc       |  92.45 |  10.28 |  96.27 | r45211 GA best_f1 val_cal
                                               |    | F1|FPR<5  |  82.43 |   3.77 |  89.26 | r45198 GS best_fpr beta
                                               |    | F1|FPR<4  |  82.43 |   3.77 |  89.26 | r45198 GS best_fpr beta
                                               |    | F1|FPR<2  |  81.45 |   0.70 |  88.23 | r45211 GA best_acc fixed_05
                                               |    | COHORT    |  92.34 |  10.64 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.16, FPR ±0.51
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-64b-Wb-C35-250n100b-OI |  3 | F1        |  92.52 |   7.82 |  96.23 | r45211 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  63.98 |   0.00 |  70.49 | r8198 GA best_fpr empirical_cumulative
                                               |    | Acc       |  92.45 |   9.50 |  96.25 | r45211 GA best_ce platt
                                               |    | F1|FPR<5  |  91.39 |   3.72 |  95.43 | r45211 GA best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  91.39 |   3.72 |  95.43 | r45211 GA best_ce empirical_cumulative
                                               |    | F1|FPR<2  |  85.32 |   1.75 |  91.29 | r8198 GA best_ce fixed_05
                                               |    | COHORT    |  92.35 |   9.56 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.08, FPR ±0.24
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-64b-Wb-C35-500n34b-OI |  2 | F1        |  92.18 |  10.78 |  96.14 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  50.29 |   0.40 |  53.58 | r45198 GA best_fpr empirical_cumulative
                                               |    | Acc       |  92.09 |  12.39 |  96.15 | r45211 GA best_acc empirical
                                               |    | F1|FPR<5  |  83.85 |   3.26 |  90.31 | r45211 GS best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  83.85 |   3.26 |  90.31 | r45211 GS best_fpr empirical_cumulative
                                               |    | F1|FPR<2  |  80.74 |   1.03 |  87.67 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  92.17 |  10.92 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.03, FPR ±0.20
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-64b-Wc-C35-250n100b-OI |  3 | F1        |  92.61 |   9.21 |  96.33 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  83.48 |   0.78 |  89.85 | r45198 GA best_fpr fixed_05
                                               |    | Acc       |  92.61 |   9.21 |  96.33 | r45211 GA best_acc val_cal
                                               |    | F1|FPR<5  |  92.16 |   4.61 |  95.92 | r45211 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  87.44 |   1.11 |  92.76 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  87.44 |   1.11 |  92.76 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  92.40 |   9.63 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.19, FPR ±0.41
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-64b-Wc-C35-500n34b-OI |  1 | F1        |  92.38 |  11.41 |  96.27 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  81.62 |   0.85 |  88.38 | r45211 GA best_ce fixed_05
                                               |    | Acc       |  92.38 |  11.41 |  96.27 | r45211 GA best_acc val_cal
                                               |    | F1|FPR<5  |  81.62 |   0.85 |  88.38 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  81.62 |   0.85 |  88.38 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  81.62 |   0.85 |  88.38 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  92.37 |  10.63 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-8b-Wa-C35-250n100b-OI |  1 | F1        |  87.95 |  12.82 |  93.75 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  82.11 |   1.40 |  88.82 | r45211 GA best_acc fixed_05
                                               |    | Acc       |  87.54 |  19.33 |  93.87 | r45211 GA best_f1 empirical
                                               |    | F1|FPR<5  |  82.41 |   1.51 |  89.06 | r45211 GA best_f1 fixed_05
                                               |    | F1|FPR<4  |  82.41 |   1.51 |  89.06 | r45211 GA best_f1 fixed_05
                                               |    | F1|FPR<2  |  82.41 |   1.51 |  89.06 | r45211 GA best_f1 fixed_05
                                               |    | COHORT    |  87.95 |  12.82 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-8b-Wa-C35-500n34b-OI  |  1 | F1        |  87.76 |  13.81 |  93.68 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  72.10 |   0.94 |  79.61 | r45211 GS best_ce fixed_05
                                               |    | Acc       |  87.46 |  18.88 |  93.80 | r45211 GA best_acc empirical
                                               |    | F1|FPR<5  |  80.71 |   1.06 |  87.65 | r45211 GA best_acc fixed_05
                                               |    | F1|FPR<4  |  80.71 |   1.06 |  87.65 | r45211 GA best_acc fixed_05
                                               |    | F1|FPR<2  |  80.71 |   1.06 |  87.65 | r45211 GA best_acc fixed_05
                                               |    | COHORT    |  87.64 |  14.86 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.17, FPR ±1.47
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-8b-Wb-C35-250n100b-OI |  1 | F1        |  87.66 |  13.94 |  93.63 | r45211 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  78.19 |   1.06 |  85.48 | r45211 GA best_fpr fixed_05
                                               |    | Acc       |  87.32 |  19.41 |  93.75 | r45211 GA best_ce empirical
                                               |    | F1|FPR<5  |  85.50 |   4.78 |  91.63 | r45211 GA best_acc empirical_cumulative
                                               |    | F1|FPR<4  |  82.34 |   1.68 |  89.03 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  82.34 |   1.68 |  89.03 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  87.55 |  14.52 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-8b-Wb-C35-500n34b-OI  |  1 | F1        |  87.15 |  14.81 |  93.36 | r45211 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  79.60 |   0.94 |  86.70 | r45211 GA best_acc fixed_05
                                               |    | Acc       |  86.67 |  22.20 |  93.54 | r45211 GA best_acc empirical
                                               |    | F1|FPR<5  |  80.84 |   4.01 |  88.01 | r45211 GS best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  80.74 |   3.40 |  87.87 | r45211 GS best_fpr empirical_cumulative
                                               |    | F1|FPR<2  |  79.97 |   1.15 |  87.03 | r45211 GS best_acc fixed_05
                                               |    | COHORT    |  87.15 |  14.81 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-8b-Wc-C35-250n100b-OI |  1 | F1        |  87.86 |  13.35 |  93.72 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  80.81 |   1.28 |  87.74 | r45211 GS best_acc fixed_05
                                               |    | Acc       |  87.78 |  15.83 |  93.81 | r45211 GA best_acc beta
                                               |    | F1|FPR<5  |  82.46 |   1.62 |  89.11 | r45211 GA best_acc fixed_05
                                               |    | F1|FPR<4  |  82.46 |   1.62 |  89.11 | r45211 GA best_acc fixed_05
                                               |    | F1|FPR<2  |  82.46 |   1.62 |  89.11 | r45211 GA best_acc fixed_05
                                               |    | COHORT    |  87.86 |  13.35 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-8b-Wc-C35-500n34b-OI  |  1 | F1        |  87.09 |  15.35 |  93.36 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  73.42 |   1.07 |  80.97 | r45211 GS best_fpr fixed_05
                                               |    | Acc       |  86.62 |  21.88 |  93.49 | r45211 GA best_acc empirical
                                               |    | F1|FPR<5  |  80.26 |   1.43 |  87.30 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  80.26 |   1.43 |  87.30 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  80.26 |   1.43 |  87.30 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  87.09 |  15.35 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-96b-Wa-C35-250n100b-OI |  3 | F1        |  92.96 |   8.81 |  96.51 | r45198 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  84.56 |   0.83 |  90.68 | r45211 GA best_fpr fixed_05
                                               |    | Acc       |  92.94 |   9.91 |  96.53 | r8198 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  84.89 |   0.98 |  90.93 | r45198 GA best_fpr fixed_05
                                               |    | F1|FPR<4  |  84.89 |   0.98 |  90.93 | r45198 GA best_fpr fixed_05
                                               |    | F1|FPR<2  |  84.89 |   0.98 |  90.93 | r45198 GA best_fpr fixed_05
                                               |    | COHORT    |  92.94 |   9.28 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.03, FPR ±0.57
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-96b-Wa-C35-500n34b-OI |  1 | F1        |  92.71 |  10.41 |  96.42 | r45211 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  79.60 |   0.76 |  86.68 | r45211 GS best_acc fixed_05
                                               |    | Acc       |  92.68 |  11.03 |  96.43 | r45211 GA best_acc empirical
                                               |    | F1|FPR<5  |  84.74 |   3.13 |  90.96 | r45211 GS best_fpr beta
                                               |    | F1|FPR<4  |  84.74 |   3.13 |  90.96 | r45211 GS best_fpr beta
                                               |    | F1|FPR<2  |  82.74 |   1.02 |  89.29 | r45211 GA best_fpr fixed_05
                                               |    | COHORT    |  92.71 |  10.41 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-96b-Wb-C35-250n100b-OI |  3 | F1        |  92.98 |   8.51 |  96.51 | r45198 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  66.37 |   0.08 |  73.28 | r45198 GA best_fpr empirical_cumulative
                                               |    | Acc       |  92.96 |   9.08 |  96.52 | r45198 GA best_ce train_cal
                                               |    | F1|FPR<5  |  91.83 |   4.61 |  95.72 | r45198 GA best_ce empirical_cumulative
                                               |    | F1|FPR<4  |  91.45 |   3.90 |  95.47 | r8198 GA best_ce empirical_cumulative
                                               |    | F1|FPR<2  |  86.08 |   1.03 |  91.80 | r8198 GA best_ce fixed_05
                                               |    | COHORT    |  92.56 |   9.84 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.30, FPR ±0.78
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-96b-Wb-C35-500n34b-OI |  1 | F1        |  92.63 |   9.80 |  96.36 | r45211 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  76.09 |   0.08 |  83.45 | r45211 GA best_fpr fixed_05
                                               |    | Acc       |  92.63 |   9.80 |  96.36 | r45211 GA best_f1 val_cal
                                               |    | F1|FPR<5  |  80.72 |   0.92 |  87.64 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<4  |  80.72 |   0.92 |  87.64 | r45211 GA best_ce fixed_05
                                               |    | F1|FPR<2  |  80.72 |   0.92 |  87.64 | r45211 GA best_ce fixed_05
                                               |    | COHORT    |  92.63 |   9.80 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-96b-Wc-C35-250n100b-OI | 30 | F1        |  93.17 |   8.07 |  96.60 | r18871 GA best_acc val_cal
                                               |    | FPR       |  88.27 |   0.72 |  93.31 | r39029 GA best_fpr fixed_05
                                               |    | Acc       |  93.17 |   8.37 |  96.61 | r18871 GA best_acc train_cal
                                               |    | F1|FPR<5  |  92.73 |   4.86 |  96.26 | r93825 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  88.48 |   1.06 |  93.46 | r35476 GA best_fpr fixed_05
                                               |    | F1|FPR<2  |  88.48 |   1.06 |  93.46 | r35476 GA best_fpr fixed_05
                                               |    | COHORT    |  92.93 |   8.34 |    --- | GA best_f1 val_cal mean±std over n=30: F1 ±0.15, FPR ±0.84
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-ciciot-subsample-96b-Wc-C35-500n34b-OI |  1 | F1        |  92.62 |   9.25 |  96.33 | r45211 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  80.67 |   0.87 |  87.60 | r45211 GS best_fpr fixed_05
                                               |    | Acc       |  92.60 |  10.94 |  96.38 | r45211 GA best_acc beta
                                               |    | F1|FPR<5  |  84.74 |   3.13 |  90.96 | r45211 GS best_fpr beta
                                               |    | F1|FPR<4  |  84.74 |   3.13 |  90.96 | r45211 GS best_fpr beta
                                               |    | F1|FPR<2  |  82.98 |   0.89 |  89.47 | r45211 GA best_fpr fixed_05
                                               |    | COHORT    |  92.62 |   9.25 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## CICIDS2017 random (2way, LEGACY)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-16b-Wa-C35-500n34b-OI    |  5 | F1        |  99.57 |   0.12 |  99.73 | r8188 GA best_acc val_cal
                                               |    | FPR       |  97.37 |   0.08 |  98.39 | r63504 GS best_fpr beta
                                               |    | Acc       |  99.57 |   0.12 |  99.73 | r8188 GA best_acc val_cal
                                               |    | F1|FPR<5  |  99.57 |   0.12 |  99.73 | r8188 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.57 |   0.12 |  99.73 | r8188 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.57 |   0.12 |  99.73 | r8188 GA best_acc val_cal
                                               |    | COHORT    |  99.55 |   0.12 |    --- | GA best_f1 val_cal mean±std over n=5: F1 ±0.01, FPR ±0.01
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-16b-Wb-C35-500n34b-OI    |  2 | F1        |  99.55 |   0.12 |  99.72 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  98.79 |   0.06 |  99.24 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.55 |   0.12 |  99.72 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<5  |  99.55 |   0.12 |  99.72 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.55 |   0.12 |  99.72 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.55 |   0.12 |  99.72 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  99.50 |   0.18 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.07, FPR ±0.08
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-16b-Wbu-C35-500n34b-OI   |  3 | F1        |  99.53 |   0.10 |  99.71 | r25608 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.03 |   0.07 |  99.39 | r25608 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.53 |   0.10 |  99.71 | r25608 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.53 |   0.10 |  99.71 | r25608 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.53 |   0.10 |  99.71 | r25608 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.53 |   0.10 |  99.71 | r25608 GA best_acc train_cal
                                               |    | COHORT    |  99.53 |   0.12 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.00, FPR ±0.03
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-16b-Wc-C35-500n34b-OI    |  3 | F1        |  99.55 |   0.11 |  99.72 | r25608 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.38 |   0.08 |  99.61 | r8188 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.55 |   0.11 |  99.72 | r25608 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.55 |   0.11 |  99.72 | r25608 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.55 |   0.11 |  99.72 | r25608 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.55 |   0.11 |  99.72 | r25608 GA best_acc train_cal
                                               |    | COHORT    |  99.54 |   0.11 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.01, FPR ±0.01
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-32b-Wa-C35-500n34b-OI    |  2 | F1        |  99.57 |   0.08 |  99.73 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.56 |   0.07 |  99.72 | r82096 GA best_acc empirical
                                               |    | Acc       |  99.57 |   0.08 |  99.73 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<5  |  99.57 |   0.08 |  99.73 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.57 |   0.08 |  99.73 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.57 |   0.08 |  99.73 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  99.51 |   0.15 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.08, FPR ±0.10
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-32b-Wb-C35-500n34b-OI    |  1 | F1        |  99.40 |   0.19 |  99.62 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  98.82 |   0.05 |  99.26 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.40 |   0.19 |  99.62 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.40 |   0.19 |  99.62 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.40 |   0.19 |  99.62 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.40 |   0.19 |  99.62 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  99.40 |   0.19 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-32b-Wbu-C35-500n34b-OI   |  1 | F1        |  99.54 |   0.17 |  99.71 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.26 |   0.07 |  99.54 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.54 |   0.17 |  99.71 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<5  |  99.54 |   0.17 |  99.71 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.54 |   0.17 |  99.71 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.54 |   0.17 |  99.71 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  99.54 |   0.17 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-32b-Wc-C35-500n34b-OI    |  1 | F1        |  99.47 |   0.18 |  99.67 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.41 |   0.14 |  99.63 | r82096 GA best_acc empirical
                                               |    | Acc       |  99.47 |   0.18 |  99.67 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.47 |   0.18 |  99.67 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.47 |   0.18 |  99.67 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.47 |   0.18 |  99.67 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  99.47 |   0.18 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-64b-Wa-C35-500n34b-OI    |  5 | F1        |  99.61 |   0.07 |  99.75 | r8188 GA best_acc train_cal
                                               |    | FPR       |  99.54 |   0.06 |  99.71 | r8188 GA best_fpr empirical
                                               |    | Acc       |  99.61 |   0.07 |  99.75 | r8188 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.61 |   0.07 |  99.75 | r8188 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.61 |   0.07 |  99.75 | r8188 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.61 |   0.07 |  99.75 | r8188 GA best_acc train_cal
                                               |    | COHORT    |  99.56 |   0.12 |    --- | GA best_f1 val_cal mean±std over n=5: F1 ±0.04, FPR ±0.03
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-64b-Wb-C35-500n34b-OI    |  1 | F1        |  99.40 |   0.25 |  99.62 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.14 |   0.05 |  99.46 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.40 |   0.25 |  99.62 | r82096 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  99.40 |   0.25 |  99.62 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.40 |   0.25 |  99.62 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.40 |   0.25 |  99.62 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  99.40 |   0.25 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-64b-Wbu-C35-500n34b-OI   |  2 | F1        |  99.59 |   0.07 |  99.74 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.55 |   0.06 |  99.72 | r82096 GA best_fpr val_cal
                                               |    | Acc       |  99.59 |   0.07 |  99.74 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.59 |   0.07 |  99.74 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.59 |   0.07 |  99.74 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.59 |   0.07 |  99.74 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  99.50 |   0.16 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.13, FPR ±0.12
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-64b-Wc-C35-500n34b-OI    |  2 | F1        |  99.54 |   0.10 |  99.71 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.45 |   0.07 |  99.66 | r82096 GA best_fpr train_cal
                                               |    | Acc       |  99.54 |   0.10 |  99.71 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.54 |   0.10 |  99.71 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.54 |   0.10 |  99.71 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.54 |   0.10 |  99.71 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  99.51 |   0.14 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.05, FPR ±0.06
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-8b-Wa-C35-500n34b-OI     |  1 | F1        |  99.35 |   0.25 |  99.59 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.32 |   0.22 |  99.57 | r82096 GA best_fpr train_cal
                                               |    | Acc       |  99.35 |   0.25 |  99.59 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.35 |   0.25 |  99.59 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.35 |   0.25 |  99.59 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.35 |   0.25 |  99.59 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  99.35 |   0.25 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-8b-Wb-C35-500n34b-OI     |  1 | F1        |  99.39 |   0.21 |  99.62 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  96.87 |   0.16 |  98.08 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.39 |   0.21 |  99.62 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<5  |  99.39 |   0.21 |  99.62 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.39 |   0.21 |  99.62 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.39 |   0.21 |  99.62 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  99.39 |   0.21 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-8b-Wbu-C35-500n34b-OI    |  1 | F1        |  99.39 |   0.21 |  99.62 | r82096 GA best_f1 train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  94.85 |   0.05 |  96.93 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.39 |   0.21 |  99.62 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.39 |   0.21 |  99.62 | r82096 GA best_f1 train_cal
                                               |    | F1|FPR<4  |  99.39 |   0.21 |  99.62 | r82096 GA best_f1 train_cal
                                               |    | F1|FPR<2  |  99.39 |   0.21 |  99.62 | r82096 GA best_f1 train_cal
                                               |    | COHORT    |  99.39 |   0.21 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-8b-Wc-C35-500n34b-OI     |  1 | F1        |  99.36 |   0.24 |  99.60 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.35 |   0.20 |  99.59 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  99.36 |   0.24 |  99.60 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.36 |   0.24 |  99.60 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.36 |   0.24 |  99.60 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.36 |   0.24 |  99.60 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  99.36 |   0.24 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-96b-Wa-C35-250n100b-OI   |  1 | F1        |  99.58 |   0.19 |  99.73 | r25608 GA best_acc empirical_cumulative  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.50 |   0.16 |  99.68 | r25608 GA best_acc empirical
                                               |    | Acc       |  99.58 |   0.19 |  99.73 | r25608 GA best_acc empirical_cumulative
                                               |    | F1|FPR<5  |  99.58 |   0.19 |  99.73 | r25608 GA best_acc empirical_cumulative
                                               |    | F1|FPR<4  |  99.58 |   0.19 |  99.73 | r25608 GA best_acc empirical_cumulative
                                               |    | F1|FPR<2  |  99.58 |   0.19 |  99.73 | r25608 GA best_acc empirical_cumulative
                                               |    | COHORT    |  99.58 |   0.19 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-96b-Wa-C35-500n34b-OI    | 30 | F1        |  99.64 |   0.08 |  99.77 | r95235 GA best_acc val_cal
                                               |    | FPR       |  97.68 |   0.05 |  98.57 | r48540 GS best_fpr beta
                                               |    | Acc       |  99.64 |   0.08 |  99.77 | r95235 GA best_acc val_cal
                                               |    | F1|FPR<5  |  99.64 |   0.08 |  99.77 | r95235 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.64 |   0.08 |  99.77 | r95235 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.64 |   0.08 |  99.77 | r95235 GA best_acc val_cal
                                               |    | COHORT    |  99.55 |   0.12 |    --- | GA best_f1 val_cal mean±std over n=30: F1 ±0.05, FPR ±0.05
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-96b-Wb-C35-500n34b-OI    |  3 | F1        |  99.53 |   0.10 |  99.70 | r8188 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  44.54 |   0.00 |  80.32 | r8188 GS best_fpr platt
                                               |    | Acc       |  99.53 |   0.10 |  99.70 | r8188 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.53 |   0.10 |  99.70 | r8188 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.53 |   0.10 |  99.70 | r8188 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.53 |   0.10 |  99.70 | r8188 GA best_acc train_cal
                                               |    | COHORT    |  99.52 |   0.10 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.00, FPR ±0.01
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-96b-Wbu-C35-500n34b-OI   |  1 | F1        |  99.48 |   0.18 |  99.67 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  68.48 |   0.00 |  86.03 | r82096 GA best_fpr empirical
                                               |    | Acc       |  99.48 |   0.18 |  99.67 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<5  |  99.48 |   0.18 |  99.67 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  99.48 |   0.18 |  99.67 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  99.48 |   0.18 |  99.67 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  99.48 |   0.18 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-cicids-random-96b-Wc-C35-500n34b-OI    |  3 | F1        |  99.56 |   0.07 |  99.73 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  99.56 |   0.07 |  99.72 | r82096 GA best_fpr train_cal
                                               |    | Acc       |  99.56 |   0.07 |  99.73 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  99.56 |   0.07 |  99.73 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  99.56 |   0.07 |  99.73 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  99.56 |   0.07 |  99.73 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  99.54 |   0.11 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.02, FPR ±0.04
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## UNSW-NB15 random (2way, LEGACY)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-16b-Wb-C35-250n100b-OI     |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  80.53 |   0.41 |  97.69 | r82096 GS best_f1 beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-16b-Wb-C35-500n34b-OI      |  7 | F1        |  93.86 |   0.75 |  99.05 | r25608 GA best_acc val_cal
                                               |    | FPR       |  49.03 |   0.00 |  96.19 | r25608 GA best_fpr platt
                                               |    | Acc       |  93.86 |   0.75 |  99.05 | r25608 GA best_acc val_cal
                                               |    | F1|FPR<5  |  93.86 |   0.75 |  99.05 | r25608 GA best_acc val_cal
                                               |    | F1|FPR<4  |  93.86 |   0.75 |  99.05 | r25608 GA best_acc val_cal
                                               |    | F1|FPR<2  |  93.86 |   0.75 |  99.05 | r25608 GA best_acc val_cal
                                               |    | COHORT    |  93.61 |   1.03 |    --- | GA best_f1 val_cal mean±std over n=7: F1 ±0.10, FPR ±0.09
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-16b-Wc-C35-500n34b-OI      |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  91.83 |   1.01 |  98.70 | r82096 GS best_f1 beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-32b-Wa-C35-500n34b-OI      |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  91.99 |   1.05 |  98.72 | r82096 GS best_fpr empirical
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-32b-Wb-C35-250n100b-OI     |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.56 |   0.75 |  98.03 | r82096 GA best_fpr beta
                                               |    | Acc       |  93.49 |   0.99 |  98.94 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | COHORT    |  93.50 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-32b-Wb-C35-500n34b-OI      |  3 | F1        |  93.51 |   1.12 |  98.92 | r8188 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  86.57 |   0.62 |  98.19 | r8188 GA best_acc beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r8188 GS best_acc train_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r8188 GA best_acc val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r8188 GA best_acc val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r8188 GA best_acc val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-32b-Wc-C35-500n34b-OI      |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  93.00 |   1.09 |  98.85 | r82096 GA best_ce empirical
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GA best_fpr val_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GA best_fpr val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GA best_fpr val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GA best_fpr val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-64b-Wa-C35-500n34b-OI      |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  86.35 |   0.74 |  98.11 | r82096 GS best_fpr beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GA best_f1 val_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-64b-Wb-C35-250n100b-OI     |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  86.69 |   0.84 |  98.11 | r82096 GA best_acc beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GS best_acc train_cal
                                               |    | COHORT    |  93.50 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-64b-Wb-C35-500n34b-OI      | 39 | F1        |  93.92 |   0.68 |  99.07 | r82096 GA best_acc train_cal
                                               |    | FPR       |  49.03 |   0.00 |  96.19 | r67784 GA best_fpr platt
                                               |    | Acc       |  93.92 |   0.68 |  99.07 | r82096 GA best_f1 train_cal
                                               |    | F1|FPR<5  |  93.92 |   0.68 |  99.07 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  93.92 |   0.68 |  99.07 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  93.92 |   0.68 |  99.07 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  93.56 |   1.08 |    --- | GA best_f1 val_cal mean±std over n=39: F1 ±0.11, FPR ±0.10
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-64b-Wc-C35-500n34b-OI      |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  92.19 |   1.05 |  98.75 | r82096 GA best_f1 beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-8b-Wb-C35-250n100b-OI      |  1 | F1        |  93.52 |   1.12 |  98.92 | r82096 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  83.61 |   0.09 |  98.12 | r82096 GA best_fpr empirical_cumulative
                                               |    | Acc       |  93.52 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<5  |  93.52 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<4  |  93.52 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | F1|FPR<2  |  93.52 |   1.12 |  98.92 | r82096 GA best_ce val_cal
                                               |    | COHORT    |  93.52 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-8b-Wb-C35-500n34b-OI       |  2 | F1        |  93.51 |   1.12 |  98.92 | r8188 GS best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  49.03 |   0.00 |  96.19 | r82096 GS best_fpr train_cal
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r8188 GS best_acc val_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r8188 GS best_acc val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r8188 GS best_acc val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r8188 GS best_acc val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-8b-Wc-C35-500n34b-OI       |  1 | F1        |  93.52 |   1.12 |  98.92 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  49.03 |   0.00 |  96.19 | r82096 GS best_fpr train_cal
                                               |    | Acc       |  93.52 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  93.52 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  93.52 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  93.52 |   1.12 |  98.92 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  93.52 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-96b-Wa-C35-500n34b-OI      |  1 | F1        |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  88.44 |   0.76 |  98.35 | r82096 GA best_acc beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r82096 GA best_acc val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-96b-Wb-C35-250n100b-OI     |  2 | F1        |  93.51 |   1.12 |  98.92 | r25608 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  84.52 |   0.35 |  98.09 | r25608 GA best_fpr beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r25608 GA best_ce empirical_cumulative
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r25608 GA best_acc val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r25608 GA best_acc val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r25608 GA best_acc val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-96b-Wb-C35-500n34b-OI      |  3 | F1        |  93.78 |   0.84 |  99.02 | r82096 GA best_acc train_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  92.25 |   0.42 |  98.90 | r82096 GA best_f1 beta
                                               |    | Acc       |  93.78 |   0.84 |  99.02 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<5  |  93.78 |   0.84 |  99.02 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<4  |  93.78 |   0.84 |  99.02 | r82096 GA best_acc train_cal
                                               |    | F1|FPR<2  |  93.78 |   0.84 |  99.02 | r82096 GA best_acc train_cal
                                               |    | COHORT    |  93.60 |   1.03 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.15, FPR ±0.16
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-96b-Wc-C35-250n100b-OI     |  1 | F1        |  93.51 |   1.12 |  98.92 | r69488 GA best_acc val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  84.67 |   0.42 |  98.07 | r69488 GS best_fpr beta
                                               |    | Acc       |  93.51 |   1.12 |  98.92 | r69488 GA best_acc val_cal
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r69488 GA best_acc val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r69488 GA best_acc val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r69488 GA best_acc val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=1: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-random-96b-Wc-C35-500n34b-OI      |  3 | F1        |  93.51 |   1.12 |  98.92 | r8188 GA best_f1 val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  93.14 |   0.63 |  98.97 | r8188 GA best_fpr empirical_cumulative
                                               |    | Acc       |  93.14 |   0.63 |  98.97 | r8188 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<5  |  93.51 |   1.12 |  98.92 | r8188 GA best_f1 val_cal
                                               |    | F1|FPR<4  |  93.51 |   1.12 |  98.92 | r8188 GA best_f1 val_cal
                                               |    | F1|FPR<2  |  93.51 |   1.12 |  98.92 | r8188 GA best_f1 val_cal
                                               |    | COHORT    |  93.51 |   1.12 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.00, FPR ±0.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## UNSW-NB15 temporal (2way, LEGACY)

    config                                     |  n | best      |     F1 |    FPR |    Acc | source (seed phase genome mode)
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-16b-Wa-C35-500n34b-OI    |  3 | F1        |  89.07 |   9.04 |  89.13 | r11760 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.11 |   1.65 |  85.13 | r88021 GA best_ce empirical
                                               |    | Acc       |  89.07 |   9.04 |  89.13 | r11760 GA best_fpr val_cal
                                               |    | F1|FPR<5  |  87.39 |   3.92 |  87.39 | r74627 GA best_fpr empirical
                                               |    | F1|FPR<4  |  87.39 |   3.92 |  87.39 | r74627 GA best_fpr empirical
                                               |    | F1|FPR<2  |  85.11 |   1.65 |  85.13 | r88021 GA best_ce empirical
                                               |    | COHORT    |  88.44 |  10.01 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.22, FPR ±0.87
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-16b-Wa-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  89.16 |   9.44 |  89.23 | r88021 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  89.10 |   8.07 |  89.14 | r11760 GA best_fpr val_cal
                                               |    | Acc       |  89.16 |   9.44 |  89.23 | r88021 GA best_ce val_cal
                                               |    | COHORT    |  88.68 |   9.93 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.19, FPR ±1.72
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-16b-Wb-C35-500n34b-OI    | 30 | F1        |  89.34 |   9.76 |  89.42 | r56926 GA best_ce val_cal
                                               |    | FPR       |  70.21 |   0.15 |  71.23 | r27384 GA best_fpr empirical_cumulative
                                               |    | Acc       |  89.34 |   9.76 |  89.42 | r56926 GA best_ce val_cal
                                               |    | F1|FPR<5  |  88.14 |   4.76 |  88.15 | r42823 GA best_ce empirical
                                               |    | F1|FPR<4  |  87.27 |   3.49 |  87.27 | r84446 GA best_ce empirical
                                               |    | F1|FPR<2  |  78.41 |   1.96 |  78.60 | r13710 GS best_fpr empirical_cumulative
                                               |    | COHORT    |  85.88 |  17.05 |    --- | GA best_f1 val_cal mean±std over n=30: F1 ±2.02, FPR ±7.35
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-16b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  88.81 |  11.15 |  88.91 | r11760 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  87.50 |   5.93 |  87.51 | r88021 GA best_fpr val_cal
                                               |    | Acc       |  88.81 |  11.15 |  88.91 | r11760 GA best_ce val_cal
                                               |    | COHORT    |  84.43 |  12.87 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±2.59, FPR ±8.67
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-16b-Wc-C35-500n34b-OI    |  3 | F1        |  89.72 |   8.96 |  89.78 | r11760 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.58 |   1.64 |  85.59 | r11760 GA best_fpr empirical
                                               |    | Acc       |  89.72 |   8.96 |  89.78 | r11760 GA best_fpr val_cal
                                               |    | F1|FPR<5  |  87.66 |   3.64 |  87.66 | r88021 GA best_ce empirical
                                               |    | F1|FPR<4  |  87.66 |   3.64 |  87.66 | r88021 GA best_ce empirical
                                               |    | F1|FPR<2  |  85.92 |   1.84 |  85.93 | r11760 GA best_ce empirical
                                               |    | COHORT    |  88.78 |   8.71 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.62, FPR ±0.18
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-32b-Wa-C35-250n100b-OI   |  3 | F1        |  88.31 |   9.11 |  88.36 | r52015 GS best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  88.31 |   9.11 |  88.36 | r52015 GS best_ce val_cal
                                               |    | Acc       |  88.31 |   9.11 |  88.36 | r52015 GS best_ce val_cal
                                               |    | COHORT    |  87.86 |  12.22 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.07, FPR ±0.57
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-32b-Wa-C35-500n34b-OI    |  4 | F1        |  89.46 |   8.99 |  89.53 | r11760 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.20 |   1.58 |  85.21 | r74627 GA best_ce empirical
                                               |    | Acc       |  89.46 |   8.99 |  89.53 | r11760 GA best_ce val_cal
                                               |    | F1|FPR<5  |  88.68 |   4.61 |  88.69 | r54181 GA best_ce empirical
                                               |    | F1|FPR<4  |  87.80 |   3.33 |  87.80 | r88021 GA best_ce empirical
                                               |    | F1|FPR<2  |  85.64 |   1.95 |  85.65 | r74627 GA best_fpr empirical
                                               |    | COHORT    |  88.84 |   8.94 |    --- | GA best_f1 val_cal mean±std over n=4: F1 ±0.51, FPR ±0.88
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-32b-Wa-C35-500n34b-OI-PREEMP-OLD | 15 | F1        |  89.46 |   9.21 |  89.52 | r11760 GA best_ce val_cal
                                               |    | FPR       |  89.11 |   6.68 |  89.14 | r65167 GA best_ce val_cal
                                               |    | Acc       |  89.46 |   9.21 |  89.52 | r11760 GA best_ce val_cal
                                               |    | COHORT    |  88.48 |   9.69 |    --- | GA best_f1 val_cal mean±std over n=15: F1 ±0.29, FPR ±1.12
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-32b-Wb-C35-250n100b-OI   |  3 | F1        |  88.39 |  10.36 |  88.46 | r25694 GS best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  77.10 |   2.78 |  77.33 | r52015 GS best_fpr empirical_cumulative
                                               |    | Acc       |  88.39 |  10.36 |  88.46 | r25694 GS best_ce val_cal
                                               |    | F1|FPR<5  |  77.10 |   2.78 |  77.33 | r52015 GS best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  77.10 |   2.78 |  77.33 | r52015 GS best_fpr empirical_cumulative
                                               |    | COHORT    |  87.80 |  12.95 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.15, FPR ±1.16
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-32b-Wb-C35-500n34b-OI    |  3 | F1        |  89.10 |   6.34 |  89.12 | r74627 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  71.51 |   0.25 |  72.36 | r11760 GA best_fpr empirical_cumulative
                                               |    | Acc       |  89.10 |   6.34 |  89.12 | r74627 GA best_ce val_cal
                                               |    | F1|FPR<5  |  86.12 |   4.82 |  86.12 | r88021 GA best_fpr empirical
                                               |    | F1|FPR<4  |  86.11 |   3.39 |  86.11 | r88021 GA best_ce empirical
                                               |    | F1|FPR<2  |  84.24 |   1.90 |  84.26 | r11760 GA best_ce empirical
                                               |    | COHORT    |  87.03 |  13.28 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±1.87, FPR ±6.09
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-32b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  88.95 |   9.77 |  89.02 | r88021 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  63.03 |   0.24 |  65.23 | r88021 GA best_fpr empirical_cumulative
                                               |    | Acc       |  88.95 |   9.77 |  89.02 | r88021 GA best_ce val_cal
                                               |    | F1|FPR<5  |  63.03 |   0.24 |  65.23 | r88021 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  63.03 |   0.24 |  65.23 | r88021 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<2  |  63.03 |   0.24 |  65.23 | r88021 GA best_fpr empirical_cumulative
                                               |    | COHORT    |  86.94 |  14.35 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±1.93, FPR ±8.20
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-32b-Wc-C35-500n34b-OI    |  3 | F1        |  89.58 |   9.46 |  89.65 | r11760 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  84.92 |   1.88 |  84.93 | r74627 GA best_ce empirical
                                               |    | Acc       |  89.58 |   9.46 |  89.65 | r11760 GA best_fpr val_cal
                                               |    | F1|FPR<5  |  87.99 |   4.13 |  87.99 | r11760 GA best_ce empirical
                                               |    | F1|FPR<4  |  87.59 |   3.28 |  87.59 | r11760 GA best_fpr empirical
                                               |    | F1|FPR<2  |  84.92 |   1.88 |  84.93 | r74627 GA best_ce empirical
                                               |    | COHORT    |  88.76 |   9.47 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.29, FPR ±0.63
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-64b-Wa-C35-250n100b-OI   |  3 | F1        |  88.45 |   9.42 |  88.51 | r25694 GS best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  88.45 |   9.42 |  88.51 | r25694 GS best_ce val_cal
                                               |    | Acc       |  88.45 |   9.42 |  88.51 | r25694 GS best_ce val_cal
                                               |    | COHORT    |  87.80 |  12.08 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.05, FPR ±1.79
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-64b-Wa-C35-500n34b-OI    |  3 | F1        |  89.26 |   8.50 |  89.31 | r74627 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.32 |   1.28 |  85.33 | r11760 GA best_ce empirical
                                               |    | Acc       |  89.26 |   8.50 |  89.31 | r74627 GA best_ce val_cal
                                               |    | F1|FPR<5  |  85.70 |   4.76 |  85.70 | r74627 GS best_ce empirical
                                               |    | F1|FPR<4  |  85.32 |   1.28 |  85.33 | r11760 GA best_ce empirical
                                               |    | F1|FPR<2  |  85.32 |   1.28 |  85.33 | r11760 GA best_ce empirical
                                               |    | COHORT    |  88.67 |  10.11 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.29, FPR ±1.68
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-64b-Wa-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  89.10 |   9.59 |  89.17 | r88021 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  77.32 |   6.36 |  77.42 | r74627 GS best_fpr train_cal
                                               |    | Acc       |  89.10 |   9.59 |  89.17 | r88021 GA best_ce val_cal
                                               |    | COHORT    |  88.38 |   9.34 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.04, FPR ±0.90
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-64b-Wb-C35-250n100b-OI   |  3 | F1        |  88.59 |   9.74 |  88.65 | r14675 GS best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  74.66 |   0.55 |  75.17 | r25694 GS best_fpr empirical_cumulative
                                               |    | Acc       |  88.59 |   9.74 |  88.65 | r14675 GS best_fpr val_cal
                                               |    | F1|FPR<5  |  78.49 |   2.45 |  78.66 | r14675 GA best_fpr val_cal
                                               |    | F1|FPR<4  |  78.49 |   2.45 |  78.66 | r14675 GA best_fpr val_cal
                                               |    | F1|FPR<2  |  74.66 |   0.55 |  75.17 | r25694 GS best_fpr empirical_cumulative
                                               |    | COHORT    |  87.69 |  12.50 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.17, FPR ±1.03
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-64b-Wb-C35-500n34b-OI    |  3 | F1        |  88.99 |   8.83 |  89.05 | r74627 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  64.91 |   0.69 |  66.70 | r74627 GA best_fpr empirical_cumulative
                                               |    | Acc       |  88.99 |   8.83 |  89.05 | r74627 GA best_ce val_cal
                                               |    | F1|FPR<5  |  87.31 |   3.30 |  87.31 | r74627 GA best_ce empirical
                                               |    | F1|FPR<4  |  87.31 |   3.30 |  87.31 | r74627 GA best_ce empirical
                                               |    | F1|FPR<2  |  64.91 |   0.69 |  66.70 | r74627 GA best_fpr empirical_cumulative
                                               |    | COHORT    |  86.78 |  11.53 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±2.27, FPR ±2.69
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-64b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  89.12 |   8.64 |  89.17 | r11760 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  70.25 |   0.75 |  71.22 | r74627 GA best_fpr empirical_cumulative
                                               |    | Acc       |  89.12 |   8.64 |  89.17 | r11760 GA best_ce val_cal
                                               |    | F1|FPR<5  |  70.25 |   0.75 |  71.22 | r74627 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  70.25 |   0.75 |  71.22 | r74627 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<2  |  70.25 |   0.75 |  71.22 | r74627 GA best_fpr empirical_cumulative
                                               |    | COHORT    |  86.95 |  15.78 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±1.86, FPR ±8.51
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-64b-Wc-C35-500n34b-OI    |  3 | F1        |  89.64 |   8.98 |  89.71 | r88021 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  87.65 |   2.55 |  87.65 | r88021 GA best_ce empirical
                                               |    | Acc       |  89.64 |   8.98 |  89.71 | r88021 GA best_fpr val_cal
                                               |    | F1|FPR<5  |  87.65 |   2.55 |  87.65 | r88021 GA best_ce empirical
                                               |    | F1|FPR<4  |  87.65 |   2.55 |  87.65 | r88021 GA best_ce empirical
                                               |    | COHORT    |  88.90 |   9.70 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.38, FPR ±1.26
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-8b-Wa-C35-500n34b-OI     |  3 | F1        |  89.48 |   8.68 |  89.54 | r88021 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  88.14 |   3.98 |  88.15 | r88021 GA best_ce empirical
                                               |    | Acc       |  89.48 |   8.68 |  89.54 | r88021 GA best_ce val_cal
                                               |    | F1|FPR<5  |  88.14 |   3.98 |  88.15 | r88021 GA best_ce empirical
                                               |    | F1|FPR<4  |  88.14 |   3.98 |  88.15 | r88021 GA best_ce empirical
                                               |    | COHORT    |  88.20 |  10.19 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.77, FPR ±0.49
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-8b-Wa-C35-500n34b-OI-PREEMP-OLD |  2 | F1        |  89.49 |   9.86 |  89.57 | r74627 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  88.96 |   9.28 |  89.02 | r74627 GS best_ce val_cal
                                               |    | Acc       |  89.49 |  10.32 |  89.58 | r11760 GA best_ce val_cal
                                               |    | COHORT    |  88.80 |   9.56 |    --- | GA best_f1 val_cal mean±std over n=2: F1 ±0.17, FPR ±0.37
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-8b-Wb-C35-500n34b-OI     |  3 | F1        |  89.51 |   9.50 |  89.59 | r74627 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  86.34 |   2.58 |  86.34 | r88021 GA best_ce empirical
                                               |    | Acc       |  89.51 |   9.50 |  89.59 | r74627 GA best_ce val_cal
                                               |    | F1|FPR<5  |  86.65 |   2.77 |  86.65 | r11760 GA best_ce empirical
                                               |    | F1|FPR<4  |  86.65 |   2.77 |  86.65 | r11760 GA best_ce empirical
                                               |    | COHORT    |  88.10 |  11.42 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.74, FPR ±1.21
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-8b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  89.22 |   9.16 |  89.28 | r11760 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  86.42 |   7.45 |  86.44 | r11760 GS best_fpr empirical_cumulative
                                               |    | Acc       |  89.22 |   9.16 |  89.28 | r11760 GA best_ce val_cal
                                               |    | COHORT    |  87.22 |  13.25 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±2.19, FPR ±6.97
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-8b-Wc-C35-500n34b-OI     |  3 | F1        |  89.72 |  10.37 |  89.81 | r88021 GA best_fpr val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.77 |   2.60 |  85.78 | r11760 GA best_fpr empirical
                                               |    | Acc       |  89.72 |  10.37 |  89.81 | r88021 GA best_fpr val_cal
                                               |    | F1|FPR<5  |  86.98 |   4.27 |  86.98 | r74627 GA best_acc empirical
                                               |    | F1|FPR<4  |  86.72 |   3.29 |  86.72 | r88021 GA best_acc empirical
                                               |    | COHORT    |  89.39 |   9.59 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.10, FPR ±0.51
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-96b-Wa-C35-250n100b-OI   |  3 | F1        |  88.38 |  10.06 |  88.44 | r14675 GS best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  88.38 |  10.06 |  88.44 | r14675 GS best_ce val_cal
                                               |    | Acc       |  88.38 |  10.06 |  88.44 | r14675 GS best_ce val_cal
                                               |    | COHORT    |  87.82 |  13.29 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.17, FPR ±0.48
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-96b-Wa-C35-500n34b-OI    |  3 | F1        |  88.90 |   9.26 |  88.96 | r74627 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.48 |   2.27 |  85.49 | r88021 GA best_ce empirical
                                               |    | Acc       |  88.90 |   9.26 |  88.96 | r74627 GA best_ce val_cal
                                               |    | F1|FPR<5  |  86.93 |   4.03 |  86.93 | r11760 GA best_ce empirical
                                               |    | F1|FPR<4  |  86.11 |   3.27 |  86.11 | r74627 GA best_ce empirical
                                               |    | COHORT    |  88.38 |  10.20 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.06, FPR ±0.93
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-96b-Wa-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  89.18 |   8.43 |  89.24 | r88021 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  89.18 |   8.43 |  89.24 | r88021 GA best_ce val_cal
                                               |    | Acc       |  89.18 |   8.43 |  89.24 | r88021 GA best_ce val_cal
                                               |    | COHORT    |  88.52 |  10.53 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.17, FPR ±1.55
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-96b-Wb-C35-250n100b-OI   |  4 | F1        |  89.19 |   9.55 |  89.26 | r25608 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  76.85 |   0.23 |  77.21 | r25608 GA best_fpr empirical_cumulative
                                               |    | Acc       |  89.19 |   9.55 |  89.26 | r25608 GA best_ce val_cal
                                               |    | F1|FPR<5  |  85.15 |   4.25 |  85.15 | r25608 GA best_acc empirical
                                               |    | F1|FPR<4  |  77.05 |   2.70 |  77.29 | r25608 GS best_fpr empirical_cumulative
                                               |    | F1|FPR<2  |  76.85 |   0.23 |  77.21 | r25608 GA best_fpr empirical_cumulative
                                               |    | COHORT    |  87.80 |  13.48 |    --- | GA best_f1 val_cal mean±std over n=4: F1 ±0.07, FPR ±1.00
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-96b-Wb-C35-500n34b-OI    |  3 | F1        |  89.13 |   9.40 |  89.19 | r74627 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  85.77 |   2.61 |  85.78 | r88021 GA best_ce empirical
                                               |    | Acc       |  89.13 |   9.40 |  89.19 | r74627 GA best_ce val_cal
                                               |    | F1|FPR<5  |  86.98 |   4.51 |  86.98 | r74627 GA best_fpr empirical_cumulative
                                               |    | F1|FPR<4  |  86.83 |   3.90 |  86.83 | r11760 GA best_ce empirical
                                               |    | COHORT    |  88.22 |   9.79 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.14, FPR ±0.67
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-96b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | F1        |  89.23 |   8.58 |  89.29 | r11760 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  87.25 |   4.65 |  87.25 | r88021 GA best_fpr empirical_cumulative
                                               |    | Acc       |  89.23 |   8.58 |  89.29 | r11760 GA best_ce val_cal
                                               |    | F1|FPR<5  |  87.25 |   4.65 |  87.25 | r88021 GA best_fpr empirical_cumulative
                                               |    | COHORT    |  87.22 |  11.50 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±1.73, FPR ±2.99
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------
    XDS-unsw-temporal-96b-Wc-C35-500n34b-OI    |  3 | F1        |  89.69 |   8.02 |  89.74 | r74627 GA best_ce val_cal  <-- n<5: winner's-curse territory
                                               |    | FPR       |  87.08 |   1.62 |  87.09 | r74627 GA best_fpr empirical
                                               |    | Acc       |  89.69 |   8.02 |  89.74 | r74627 GA best_ce val_cal
                                               |    | F1|FPR<5  |  88.09 |   4.24 |  88.10 | r88021 GA best_fpr empirical
                                               |    | F1|FPR<4  |  87.87 |   2.85 |  87.87 | r74627 GA best_ce empirical
                                               |    | F1|FPR<2  |  87.08 |   1.62 |  87.09 | r74627 GA best_fpr empirical
                                               |    | COHORT    |  88.93 |   9.19 |    --- | GA best_f1 val_cal mean±std over n=3: F1 ±0.42, FPR ±0.23
    -------------------------------------------+----+-----------+--------+--------+--------+---------------------------------

## 3C. Dataset roll-up — Protocol v2 (_3way): best point across ALL configs per dataset

    Best POINT per dataset across every config in that dataset. n = runs in the
    winning config. Read as a ceiling, not as the claim (see COHORT rows in 3A/3B).

    dataset                                            | best      |     F1 |    FPR |    Acc |  n | winning config / source
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    CIC-IoT-2023 neto-subsample random (3way, Protocol v2) | F1        |  93.35 |   7.50 |  96.69 | 11 | SP100-ciciot-quad-96bWc r79803 GA best_acc train_cal
                                                       | FPR       |  70.73 |   0.00 |  78.08 | 10 | SP-ciciot-bin-96bWc-n30 r42704 GA best_acc empirical
                                                       | Acc       |  93.27 |   9.25 |  96.69 | 11 | SP100-ciciot-quad-96bWc r79803 GA best_f1 beta
                                                       | F1|FPR<5  |  93.08 |   4.91 |  96.46 | 11 | SP100-ciciot-quad-96bWc r79803 GA best_acc empirical_cumulative
                                                       | F1|FPR<4  |  90.35 |   2.04 |  94.72 | 11 | SP100-ciciot-quad-96bWc r61231 GA best_acc empirical
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    CICIDS2017 random (3way, Protocol v2)              | F1        |  99.64 |   0.08 |  99.77 | 10 | SP-cicids-bin-96bWa-n30 r84914 GA best_acc train_cal
                                                       | FPR       |  44.54 |   0.00 |  80.32 | 10 | SP-cicids-abl2big-96bWa-n10 r26177 GS best_fpr platt
                                                       | Acc       |  99.64 |   0.08 |  99.77 | 10 | SP-cicids-bin-96bWa-n30 r84914 GA best_acc train_cal
                                                       | F1|FPR<5  |  99.64 |   0.08 |  99.77 | 10 | SP-cicids-bin-96bWa-n30 r84914 GA best_acc train_cal
                                                       | F1|FPR<4  |  99.64 |   0.08 |  99.77 | 10 | SP-cicids-bin-96bWa-n30 r84914 GA best_acc train_cal
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    UNSW-NB15 random (3way, Protocol v2)               | F1        |  94.56 |   0.61 |  99.17 | 10 | SP100-unswr-qsr-64bWb r32732 GA best_acc val_cal
                                                       | FPR       |  75.57 |   0.00 |  97.54 | 10 | SP100-unswr-qsr-64bWb r49648 GA best_ce empirical
                                                       | Acc       |  94.53 |   0.52 |  99.18 | 10 | SP100-unswr-qsr-64bWb r22224 GA best_f1 platt
                                                       | F1|FPR<5  |  94.56 |   0.61 |  99.17 | 10 | SP100-unswr-qsr-64bWb r32732 GA best_acc val_cal
                                                       | F1|FPR<4  |  94.56 |   0.61 |  99.17 | 10 | SP100-unswr-qsr-64bWb r32732 GA best_acc val_cal
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    UNSW-NB15 temporal (3way, Protocol v2)             | F1        |  90.20 |   5.30 |  90.22 | 10 | SP-unswt-ablqsr-16bWb-n10 r25052 GA best_f1 val_cal
                                                       | FPR       |  31.11 |   0.00 |  44.99 | 10 | SP-unswt-ablpln-16bWb-n10 r98954 GA best_fpr empirical_cumulative
                                                       | Acc       |  90.20 |   5.30 |  90.22 | 10 | SP-unswt-ablqsr-16bWb-n10 r25052 GA best_acc val_cal
                                                       | F1|FPR<5  |  90.16 |   4.29 |  90.18 | 10 | SP-unswt-ablqsr-16bWb-n10 r35879 GA best_ce val_cal
                                                       | F1|FPR<4  |  89.79 |   3.19 |  89.79 | 10 | SP-unswt-ablqsr-16bWb-n10 r35879 GA best_ce fixed_05
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------


## 3D. Dataset roll-up — LEGACY 2-way (XDS)

    Best POINT per dataset across every config in that dataset. n = runs in the
    winning config. Read as a ceiling, not as the claim (see COHORT rows in 3A/3B).

    dataset                                            | best      |     F1 |    FPR |    Acc |  n | winning config / source
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    CIC-IoT-2023 neto-subsample random (2way, LEGACY)  | F1        |  93.17 |   8.07 |  96.60 | 30 | XDS-ciciot-subsample-96b-Wc-C35-250n100b-OI r18871 GA best_acc val_cal
                                                       | FPR       |  65.50 |   0.00 |  72.27 |  1 | XDS-ciciot-subsample-32b-Wb-C35-250n100b-OI r45211 GA best_fpr empirical_cumulative
                                                       | Acc       |  93.17 |   8.37 |  96.61 | 30 | XDS-ciciot-subsample-96b-Wc-C35-250n100b-OI r18871 GA best_acc train_cal
                                                       | F1|FPR<5  |  92.73 |   4.86 |  96.26 | 30 | XDS-ciciot-subsample-96b-Wc-C35-250n100b-OI r93825 GA best_fpr empirical_cumulative
                                                       | F1|FPR<4  |  91.45 |   3.90 |  95.47 |  3 | XDS-ciciot-subsample-96b-Wb-C35-250n100b-OI r8198 GA best_ce empirical_cumulative
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    CICIDS2017 random (2way, LEGACY)                   | F1        |  99.64 |   0.08 |  99.77 | 30 | XDS-cicids-random-96b-Wa-C35-500n34b-OI r95235 GA best_acc val_cal
                                                       | FPR       |  44.54 |   0.00 |  80.32 |  3 | XDS-cicids-random-96b-Wb-C35-500n34b-OI r8188 GS best_fpr platt
                                                       | Acc       |  99.64 |   0.08 |  99.77 | 30 | XDS-cicids-random-96b-Wa-C35-500n34b-OI r95235 GA best_acc val_cal
                                                       | F1|FPR<5  |  99.64 |   0.08 |  99.77 | 30 | XDS-cicids-random-96b-Wa-C35-500n34b-OI r95235 GA best_acc val_cal
                                                       | F1|FPR<4  |  99.64 |   0.08 |  99.77 | 30 | XDS-cicids-random-96b-Wa-C35-500n34b-OI r95235 GA best_acc val_cal
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    UNSW-NB15 random (2way, LEGACY)                    | F1        |  93.92 |   0.68 |  99.07 | 39 | XDS-unsw-random-64b-Wb-C35-500n34b-OI r82096 GA best_acc train_cal
                                                       | FPR       |  49.03 |   0.00 |  96.19 |  7 | XDS-unsw-random-16b-Wb-C35-500n34b-OI r25608 GA best_fpr platt
                                                       | Acc       |  93.92 |   0.68 |  99.07 | 39 | XDS-unsw-random-64b-Wb-C35-500n34b-OI r82096 GA best_f1 train_cal
                                                       | F1|FPR<5  |  93.92 |   0.68 |  99.07 | 39 | XDS-unsw-random-64b-Wb-C35-500n34b-OI r82096 GA best_acc train_cal
                                                       | F1|FPR<4  |  93.92 |   0.68 |  99.07 | 39 | XDS-unsw-random-64b-Wb-C35-500n34b-OI r82096 GA best_acc train_cal
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------
    UNSW-NB15 temporal (2way, LEGACY)                  | F1        |  89.72 |  10.37 |  89.81 |  3 | XDS-unsw-temporal-8b-Wc-C35-500n34b-OI r88021 GA best_fpr val_cal
                                                       | FPR       |  70.21 |   0.15 |  71.23 | 30 | XDS-unsw-temporal-16b-Wb-C35-500n34b-OI r27384 GA best_fpr empirical_cumulative
                                                       | Acc       |  89.72 |  10.37 |  89.81 |  3 | XDS-unsw-temporal-8b-Wc-C35-500n34b-OI r88021 GA best_fpr val_cal
                                                       | F1|FPR<5  |  88.68 |   4.61 |  88.69 |  4 | XDS-unsw-temporal-32b-Wa-C35-500n34b-OI r54181 GA best_ce empirical
                                                       | F1|FPR<4  |  88.14 |   3.98 |  88.15 |  3 | XDS-unsw-temporal-8b-Wa-C35-500n34b-OI r88021 GA best_ce empirical
    ---------------------------------------------------+-----------+--------+--------+--------+----+------------------------


---

# =====================================================================
# SECTION 4 — best_fitness Grid-vs-GA DELTA TABLES
# =====================================================================

## 4A. Protocol v2 (_3way)

    Genome type best_fitness, threshold mode val_cal, HELD-OUT report partition.
    Delta = GA Neurons minus Grid Search (positive F1/Acc = GA better; negative FPR = GA better).

    config                                     |  n | F1 Grid    F1 GA     dF1  | FPR Grid   FPR GA    dFPR | Acc Grid   Acc GA    dAcc
    -------------------------------------------+----+---------------------------+---------------------------+--------------------------
    SP-cicids-abl2big-96bWa-n10                | 10 | 99.03±0.02 99.13±0.01  +0.10 |  0.63±0.03  0.60±0.01  -0.03 | 99.38±0.01 99.45±0.01  +0.06
    SP-cicids-abl2s-96bWa-n10                  | 10 | 98.96±0.16 99.08±0.03  +0.12 |  0.66±0.16  0.62±0.03  -0.04 | 99.33±0.11 99.41±0.02  +0.08
    SP-cicids-abl3s-96bWa-n10                  | 10 | 99.00±0.10 99.11±0.02  +0.11 |  0.62±0.14  0.61±0.02  -0.01 | 99.36±0.06 99.43±0.02  +0.07
    SP-cicids-ablpln-96bWa-n10                 | 10 | 99.21±0.09 99.27±0.03  +0.07 |  0.32±0.14  0.28±0.04  -0.04 | 99.50±0.06 99.54±0.02  +0.04
    SP-cicids-ablqsr-96bWa-n10                 | 10 | 99.24±0.04 99.33±0.06  +0.10 |  0.28±0.09  0.25±0.05  -0.03 | 99.52±0.03 99.58±0.04  +0.06
    SP-cicids-bin-96bWa-n30                    | 10 | 99.30±0.02 99.59±0.04  +0.29 |  0.31±0.02  0.09±0.02  -0.22 | 99.56±0.01 99.74±0.02  +0.18
    SP-ciciot-abl2s-96bWc-n10                  | 10 | 72.68±9.26 76.12±1.01  +3.44 | 25.66±5.42 34.92±4.53  +9.26 | 82.61±6.78 87.62±1.04  +5.01
    SP-ciciot-abl3s-96bWc-n10                  | 10 | 76.29±9.71 80.38±1.00  +4.09 | 24.72±6.34 32.65±3.80  +7.94 | 85.65±6.85 90.44±0.78  +4.79
    SP-ciciot-ablpln-96bWc-n10                 | 10 | 87.27±6.79 91.63±0.59  +4.36 | 17.12±3.62 13.94±1.77  -3.18 | 93.21±4.99 95.95±0.26  +2.74
    SP-ciciot-ablqsr-96bWc-n10                 | 10 | 88.42±1.32 91.34±0.65  +2.93 | 16.60±1.24 12.51±2.29  -4.10 | 94.23±0.73 95.74±0.29  +1.51
    SP-ciciot-bin-96bWc-n30                    | 10 | 90.27±0.28 92.67±0.14  +2.40 | 15.19±0.89  8.89±0.79  -6.30 | 95.25±0.18 96.35±0.06  +1.11
    SP-unswr-abl2big-64bWb-n10                 | 10 | 93.45±0.02 93.47±0.03  +0.02 |  1.12±0.00  1.12±0.00  +0.00 | 98.91±0.00 98.92±0.01  +0.00
    SP-unswr-abl2s-64bWb-n10                   | 10 | 93.45±0.03 93.47±0.03  +0.02 |  1.12±0.00  1.12±0.00  -0.00 | 98.91±0.00 98.91±0.00  +0.00
    SP-unswr-abl3s-64bWb-n10                   | 10 | 93.51±0.02 93.50±0.03  -0.01 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  -0.00
    SP-unswr-ablpln-64bWb-n10                  | 10 | 93.49±0.10 93.52±0.10  +0.02 |  1.07±0.04  1.04±0.04  -0.03 | 98.93±0.02 98.94±0.02  +0.01
    SP-unswr-ablqsr-64bWb-n10                  | 10 | 94.29±0.07 94.33±0.07  +0.04 |  0.60±0.05  0.62±0.07  +0.02 | 99.13±0.02 99.13±0.02  +0.00
    SP-unswr-bin-64bWb-n30                     | 10 | 93.49±0.01 93.54±0.15  +0.05 |  1.12±0.00  1.08±0.12  -0.04 | 98.92±0.00 98.93±0.04  +0.01
    SP-unswt-abl2big-16bWb-n10                 | 10 | 79.13±0.06 79.12±0.08  -0.01 | 42.09±0.11 42.07±0.10  -0.02 | 80.87±0.07 80.85±0.09  -0.01
    SP-unswt-abl2s-16bWb-n10                   | 10 | 79.19±0.03 79.20±0.10  +0.00 | 42.09±0.09 42.01±0.02  -0.08 | 80.93±0.03 80.92±0.10  -0.01
    SP-unswt-abl3s-16bWb-n10                   | 10 | 79.28±0.03 79.29±0.04  +0.00 | 42.03±0.03 42.01±0.02  -0.01 | 81.02±0.03 81.02±0.04  +0.00
    SP-unswt-ablpln-16bWb-n10                  | 10 | 80.10±2.78 79.24±0.03  -0.86 | 39.16±9.49 42.18±0.04  +3.01 | 81.68±2.26 80.99±0.03  -0.70
    SP-unswt-ablqsr-16bWb-n10                  | 10 | 82.21±4.89 83.30±5.36  +1.09 | 32.10±16.26 27.83±18.68  -4.27 | 83.45±4.07 84.37±4.47  +0.92
    SP-unswt-bin-16bWb-n30                     | 10 | 86.66±1.95 86.54±1.87  -0.11 | 13.22±5.56 13.58±7.08  +0.36 | 86.78±1.83 86.70±1.73  -0.09
    SP100-cicids-quad-96bWa                    | 11 | 99.30±0.03 99.52±0.08  +0.22 |  0.29±0.03  0.15±0.07  -0.14 | 99.56±0.02 99.70±0.05  +0.14
    SP100-ciciot-quad-96bWc                    | 11 | 90.25±0.27 92.71±0.31  +2.46 | 15.12±0.89  8.56±0.63  -6.56 | 95.23±0.18 96.36±0.16  +1.13
    SP100-unswr-qsr-64bWb                      | 10 | 94.33±0.08 94.42±0.09  +0.09 |  0.57±0.05  0.61±0.06  +0.04 | 99.14±0.01 99.15±0.02  +0.00
    SP100-unswr-quad-64bWb                     | 11 | 93.50±0.00 92.28±2.60  -1.21 |  1.12±0.00  1.12±0.41  +0.00 | 98.92±0.00 98.75±0.38  -0.17
    SP100-unswt-quad-16bWb                     | 11 | 87.17±2.01 86.89±1.50  -0.28 | 13.55±8.01 13.09±4.92  -0.46 | 87.35±1.77 87.01±1.42  -0.34

## 4B. LEGACY 2-way (XDS)

    Genome type best_fitness, threshold mode val_cal, HELD-OUT report partition.
    Delta = GA Neurons minus Grid Search (positive F1/Acc = GA better; negative FPR = GA better).

    config                                     |  n | F1 Grid    F1 GA     dF1  | FPR Grid   FPR GA    dFPR | Acc Grid   Acc GA    dAcc
    -------------------------------------------+----+---------------------------+---------------------------+--------------------------
    XDS-cicids-random-16b-Wa-C35-500n34b-OI    |  5 | 99.38±0.01 99.55±0.01  +0.18 |  0.23±0.03  0.12±0.01  -0.11 | 99.61±0.01 99.72±0.01  +0.11
    XDS-cicids-random-16b-Wb-C35-500n34b-OI    |  2 | 99.38±0.01 99.50±0.07  +0.12 |  0.20±0.02  0.18±0.08  -0.02 | 99.61±0.01 99.68±0.05  +0.07
    XDS-cicids-random-16b-Wbu-C35-500n34b-OI   |  3 | 99.38±0.01 99.53±0.00  +0.15 |  0.21±0.02  0.12±0.03  -0.09 | 99.61±0.01 99.70±0.00  +0.09
    XDS-cicids-random-16b-Wc-C35-500n34b-OI    |  3 | 99.36±0.00 99.40±0.09  +0.04 |  0.26±0.03  0.21±0.11  -0.05 | 99.59±0.00 99.62±0.06  +0.03
    XDS-cicids-random-32b-Wa-C35-500n34b-OI    |  2 | 99.27±0.00 99.51±0.08  +0.25 |  0.25±0.03  0.15±0.10  -0.11 | 99.54±0.00 99.69±0.05  +0.16
    XDS-cicids-random-32b-Wb-C35-500n34b-OI    |  1 | 99.25±0.00 99.40±0.00  +0.15 |  0.27±0.00  0.19±0.00  -0.08 | 99.53±0.00 99.62±0.00  +0.10
    XDS-cicids-random-32b-Wbu-C35-500n34b-OI   |  1 | 99.33±0.00 99.54±0.00  +0.21 |  0.28±0.00  0.17±0.00  -0.11 | 99.58±0.00 99.71±0.00  +0.14
    XDS-cicids-random-32b-Wc-C35-500n34b-OI    |  1 | 99.27±0.00 99.32±0.00  +0.05 |  0.21±0.00  0.32±0.00  +0.11 | 99.54±0.00 99.57±0.00  +0.03
    XDS-cicids-random-64b-Wa-C35-500n34b-OI    |  5 | 99.31±0.03 99.56±0.04  +0.24 |  0.31±0.02  0.12±0.03  -0.19 | 99.57±0.02 99.72±0.02  +0.15
    XDS-cicids-random-64b-Wb-C35-500n34b-OI    |  1 | 99.33±0.00 99.34±0.00  +0.01 |  0.29±0.00  0.33±0.00  +0.04 | 99.57±0.00 99.58±0.00  +0.01
    XDS-cicids-random-64b-Wbu-C35-500n34b-OI   |  2 | 99.32±0.01 99.50±0.13  +0.18 |  0.28±0.03  0.16±0.12  -0.12 | 99.57±0.00 99.68±0.08  +0.11
    XDS-cicids-random-64b-Wc-C35-500n34b-OI    |  2 | 99.28±0.00 99.33±0.01  +0.05 |  0.34±0.01  0.28±0.00  -0.06 | 99.54±0.00 99.58±0.00  +0.03
    XDS-cicids-random-8b-Wa-C35-500n34b-OI     |  1 | 99.29±0.00 99.35±0.00  +0.06 |  0.29±0.00  0.25±0.00  -0.05 | 99.55±0.00 99.59±0.00  +0.04
    XDS-cicids-random-8b-Wb-C35-500n34b-OI     |  1 | 99.26±0.00 99.39±0.00  +0.14 |  0.25±0.00  0.21±0.00  -0.04 | 99.53±0.00 99.62±0.00  +0.09
    XDS-cicids-random-8b-Wbu-C35-500n34b-OI    |  1 | 99.29±0.00 99.39±0.00  +0.11 |  0.29±0.00  0.21±0.00  -0.08 | 99.55±0.00 99.62±0.00  +0.07
    XDS-cicids-random-8b-Wc-C35-500n34b-OI     |  1 | 99.28±0.00 99.30±0.00  +0.02 |  0.31±0.00  0.30±0.00  -0.01 | 99.54±0.00 99.56±0.00  +0.01
    XDS-cicids-random-96b-Wa-C35-250n100b-OI   |  1 | 99.46±0.00 99.58±0.00  +0.12 |  0.27±0.00  0.19±0.00  -0.07 | 99.66±0.00 99.73±0.00  +0.08
    XDS-cicids-random-96b-Wa-C35-500n34b-OI    | 30 | 99.31±0.02 99.55±0.05  +0.24 |  0.29±0.03  0.12±0.05  -0.17 | 99.56±0.01 99.72±0.03  +0.15
    XDS-cicids-random-96b-Wb-C35-500n34b-OI    |  3 | 99.27±0.01 99.52±0.00  +0.26 |  0.28±0.07  0.10±0.01  -0.18 | 99.54±0.00 99.70±0.00  +0.16
    XDS-cicids-random-96b-Wbu-C35-500n34b-OI   |  1 | 99.27±0.00 99.48±0.00  +0.21 |  0.20±0.00  0.18±0.00  -0.02 | 99.54±0.00 99.67±0.00  +0.13
    XDS-cicids-random-96b-Wc-C35-500n34b-OI    |  3 | 99.28±0.04 99.35±0.04  +0.07 |  0.35±0.04  0.25±0.03  -0.09 | 99.54±0.03 99.59±0.03  +0.04
    XDS-ciciot-subsample-16b-Wa-C35-250n100b-OI |  1 | 89.93±0.00 90.15±0.00  +0.22 | 12.82±0.00 11.74±0.00  -1.08 | 94.94±0.00 95.02±0.00  +0.08
    XDS-ciciot-subsample-16b-Wa-C35-500n34b-OI |  1 | 89.93±0.00 90.24±0.00  +0.31 | 13.61±0.00 12.31±0.00  -1.30 | 94.98±0.00 95.10±0.00  +0.12
    XDS-ciciot-subsample-16b-Wb-C35-250n100b-OI |  1 | 89.47±0.00 90.17±0.00  +0.71 | 13.21±0.00 11.88±0.00  -1.33 | 94.69±0.00 95.04±0.00  +0.35
    XDS-ciciot-subsample-16b-Wb-C35-500n34b-OI |  1 | 89.65±0.00 90.09±0.00  +0.44 | 14.79±0.00 12.44±0.00  -2.35 | 94.87±0.00 95.02±0.00  +0.15
    XDS-ciciot-subsample-16b-Wc-C35-250n100b-OI |  1 | 89.94±0.00 90.48±0.00  +0.54 | 12.69±0.00 10.58±0.00  -2.11 | 94.94±0.00 95.16±0.00  +0.22
    XDS-ciciot-subsample-16b-Wc-C35-500n34b-OI |  1 | 89.93±0.00 90.32±0.00  +0.38 | 13.61±0.00 13.24±0.00  -0.37 | 94.98±0.00 95.19±0.00  +0.21
    XDS-ciciot-subsample-32b-Wa-C35-250n100b-OI |  1 | 90.66±0.00 90.71±0.00  +0.05 | 14.90±0.00 12.55±0.00  -2.35 | 95.46±0.00 95.38±0.00  -0.07
    XDS-ciciot-subsample-32b-Wa-C35-500n34b-OI |  1 | 90.71±0.00 91.17±0.00  +0.46 | 14.60±0.00 11.14±0.00  -3.46 | 95.47±0.00 95.59±0.00  +0.12
    XDS-ciciot-subsample-32b-Wb-C35-250n100b-OI |  1 | 90.74±0.00 91.20±0.00  +0.46 | 14.57±0.00 10.21±0.00  -4.36 | 95.48±0.00 95.56±0.00  +0.08
    XDS-ciciot-subsample-32b-Wb-C35-500n34b-OI |  1 | 90.43±0.00 90.94±0.00  +0.51 | 16.64±0.00 11.21±0.00  -5.42 | 95.40±0.00 95.46±0.00  +0.06
    XDS-ciciot-subsample-32b-Wc-C35-250n100b-OI |  1 | 89.91±0.00 91.61±0.00  +1.70 | 15.25±0.00 10.17±0.00  -5.08 | 95.04±0.00 95.80±0.00  +0.75
    XDS-ciciot-subsample-32b-Wc-C35-500n34b-OI |  1 | 90.60±0.00 91.16±0.00  +0.57 | 15.85±0.00 12.54±0.00  -3.30 | 95.46±0.00 95.64±0.00  +0.18
    XDS-ciciot-subsample-64b-Wa-C35-250n100b-OI |  3 | 90.94±0.13 92.56±0.07  +1.61 | 15.50±0.62  8.91±0.32  -6.59 | 95.64±0.09 96.29±0.04  +0.65
    XDS-ciciot-subsample-64b-Wa-C35-500n34b-OI |  2 | 90.79±0.36 92.28±0.07  +1.48 | 16.83±0.48 10.45±0.77  -6.38 | 95.61±0.18 96.18±0.01  +0.57
    XDS-ciciot-subsample-64b-Wb-C35-250n100b-OI |  3 | 90.63±0.66 92.17±0.28  +1.54 | 14.64±0.86  9.68±0.46  -4.96 | 95.42±0.34 96.10±0.14  +0.67
    XDS-ciciot-subsample-64b-Wb-C35-500n34b-OI |  2 | 90.81±0.09 92.17±0.03  +1.35 | 18.83±0.14 10.92±0.20  -7.91 | 95.71±0.04 96.14±0.01  +0.43
    XDS-ciciot-subsample-64b-Wc-C35-250n100b-OI |  3 | 90.41±0.12 92.25±0.19  +1.84 | 15.47±0.74  9.07±1.43  -6.40 | 95.34±0.08 96.12±0.10  +0.78
    XDS-ciciot-subsample-64b-Wc-C35-500n34b-OI |  1 | 90.34±0.00 92.33±0.00  +1.99 | 19.55±0.00  9.91±0.00  -9.64 | 95.48±0.00 96.19±0.00  +0.71
    XDS-ciciot-subsample-8b-Wa-C35-250n100b-OI |  1 | 87.44±0.00 87.94±0.00  +0.50 | 13.00±0.00 14.03±0.00  +1.03 | 93.44±0.00 93.81±0.00  +0.37
    XDS-ciciot-subsample-8b-Wa-C35-500n34b-OI  |  2 | 86.99±0.00 87.04±1.01  +0.05 | 16.17±1.00 14.60±1.11  -1.57 | 93.34±0.06 93.28±0.57  -0.07
    XDS-ciciot-subsample-8b-Wb-C35-250n100b-OI |  1 | 87.33±0.00 87.55±0.00  +0.22 | 14.12±0.00 14.52±0.00  +0.40 | 93.43±0.00 93.60±0.00  +0.16
    XDS-ciciot-subsample-8b-Wb-C35-500n34b-OI  |  1 | 86.79±0.00 87.15±0.00  +0.37 | 16.29±0.00 14.81±0.00  -1.48 | 93.23±0.00 93.36±0.00  +0.14
    XDS-ciciot-subsample-8b-Wc-C35-250n100b-OI |  1 | 87.44±0.00 87.76±0.00  +0.32 | 13.00±0.00 13.62±0.00  +0.62 | 93.44±0.00 93.67±0.00  +0.24
    XDS-ciciot-subsample-8b-Wc-C35-500n34b-OI  |  1 | 84.50±0.00 86.27±0.00  +1.77 | 19.39±0.00 13.29±0.00  -6.10 | 91.97±0.00 92.70±0.00  +0.73
    XDS-ciciot-subsample-96b-Wa-C35-250n100b-OI |  3 | 90.97±0.16 92.93±0.03  +1.97 | 15.09±1.57  9.32±0.64  -5.76 | 95.63±0.11 96.51±0.02  +0.88
    XDS-ciciot-subsample-96b-Wa-C35-500n34b-OI |  1 | 91.04±0.00 92.71±0.00  +1.66 | 14.99±0.00 10.41±0.00  -4.58 | 95.67±0.00 96.42±0.00  +0.75
    XDS-ciciot-subsample-96b-Wb-C35-250n100b-OI |  3 | 90.54±0.12 92.56±0.30  +2.02 | 17.37±1.36  9.84±0.78  -7.53 | 95.49±0.11 96.32±0.16  +0.83
    XDS-ciciot-subsample-96b-Wb-C35-500n34b-OI |  1 | 90.13±0.00 92.63±0.00  +2.50 | 16.14±0.00  9.80±0.00  -6.34 | 95.21±0.00 96.36±0.00  +1.15
    XDS-ciciot-subsample-96b-Wc-C35-250n100b-OI | 30 | 90.18±0.16 92.83±0.17  +2.64 | 14.72±0.84  8.27±0.58  -6.45 | 95.18±0.11 96.42±0.09  +1.24
    XDS-ciciot-subsample-96b-Wc-C35-500n34b-OI |  1 | 90.85±0.00 92.28±0.00  +1.42 | 18.10±0.00 10.96±0.00  -7.13 | 95.70±0.00 96.20±0.00  +0.50
    XDS-unsw-random-16b-Wb-C35-250n100b-OI     |  1 | 93.51±0.00 93.51±0.00  +0.00 |  1.12±0.00  1.12±0.00  -0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-16b-Wb-C35-500n34b-OI      |  7 | 93.47±0.09 93.61±0.10  +0.14 |  1.06±0.15  1.03±0.09  -0.03 | 98.93±0.02 98.95±0.03  +0.03
    XDS-unsw-random-16b-Wc-C35-500n34b-OI      |  1 | 93.51±0.00 93.50±0.00  -0.01 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  -0.00
    XDS-unsw-random-32b-Wa-C35-500n34b-OI      |  1 | 93.51±0.00 93.51±0.00  +0.00 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-32b-Wb-C35-250n100b-OI     |  1 | 93.50±0.00 93.50±0.00  +0.00 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-32b-Wb-C35-500n34b-OI      |  3 | 93.51±0.00 93.51±0.00  -0.00 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  -0.00
    XDS-unsw-random-32b-Wc-C35-500n34b-OI      |  1 | 93.49±0.00 93.50±0.00  +0.00 |  1.12±0.00  1.12±0.00  -0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-64b-Wa-C35-500n34b-OI      |  1 | 93.51±0.00 93.51±0.00  +0.00 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-64b-Wb-C35-250n100b-OI     |  1 | 93.51±0.00 93.50±0.00  -0.00 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  -0.00
    XDS-unsw-random-64b-Wb-C35-500n34b-OI      | 39 | 93.51±0.00 93.56±0.11  +0.05 |  1.11±0.04  1.08±0.10  -0.04 | 98.92±0.01 98.94±0.04  +0.01
    XDS-unsw-random-64b-Wc-C35-500n34b-OI      |  1 | 93.46±0.00 93.14±0.00  -0.32 |  1.12±0.00  1.12±0.00  +0.00 | 98.91±0.00 98.87±0.00  -0.05
    XDS-unsw-random-8b-Wb-C35-250n100b-OI      |  1 | 93.52±0.00 93.52±0.00  +0.00 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-8b-Wb-C35-500n34b-OI       |  2 | 93.51±0.00 93.51±0.00  -0.00 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  -0.00
    XDS-unsw-random-8b-Wc-C35-500n34b-OI       |  1 | 93.51±0.00 93.51±0.00  -0.00 |  1.12±0.00  1.12±0.00  -0.00 | 98.92±0.00 98.92±0.00  -0.00
    XDS-unsw-random-96b-Wa-C35-500n34b-OI      |  1 | 93.50±0.00 93.51±0.00  +0.01 |  1.12±0.00  1.12±0.00  +0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-96b-Wb-C35-250n100b-OI     |  2 | 93.50±0.00 93.51±0.00  +0.01 |  1.12±0.00  1.12±0.00  -0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-96b-Wb-C35-500n34b-OI      |  3 | 93.51±0.00 93.60±0.15  +0.09 |  1.12±0.00  1.03±0.16  -0.09 | 98.92±0.00 98.95±0.06  +0.03
    XDS-unsw-random-96b-Wc-C35-250n100b-OI     |  1 | 93.49±0.00 93.50±0.00  +0.01 |  1.12±0.00  1.12±0.00  -0.00 | 98.92±0.00 98.92±0.00  +0.00
    XDS-unsw-random-96b-Wc-C35-500n34b-OI      |  3 | 93.49±0.01 93.47±0.03  -0.02 |  1.12±0.00  1.12±0.00  -0.00 | 98.92±0.00 98.91±0.01  -0.00
    XDS-unsw-temporal-16b-Wa-C35-500n34b-OI    |  3 | 88.41±0.03 88.40±0.15  -0.01 |  8.94±0.32  9.78±0.98  +0.85 | 88.46±0.03 88.46±0.14  -0.00
    XDS-unsw-temporal-16b-Wa-C35-500n34b-OI-PREEMP-OLD |  3 | 88.40±0.02 88.68±0.19  +0.29 |  8.71±0.38  9.93±1.72  +1.22 | 88.44±0.01 88.75±0.22  +0.31
    XDS-unsw-temporal-16b-Wb-C35-500n34b-OI    | 30 | 86.47±1.95 85.88±2.02  -0.59 | 14.62±6.54 17.05±7.35  +2.44 | 86.64±1.83 86.13±1.82  -0.52
    XDS-unsw-temporal-16b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | 85.00±1.79 84.43±2.59  -0.57 | 20.30±9.31 12.87±8.67  -7.43 | 85.35±1.50 84.54±2.50  -0.81
    XDS-unsw-temporal-16b-Wc-C35-500n34b-OI    |  3 | 88.47±0.06 89.50±0.20  +1.03 |  9.55±0.79  9.41±1.19  -0.15 | 88.53±0.07 89.57±0.21  +1.04
    XDS-unsw-temporal-32b-Wa-C35-250n100b-OI   |  3 | 87.86±0.07 87.86±0.07  +0.00 | 12.22±0.57 12.22±0.57  +0.00 | 87.96±0.07 87.96±0.07  +0.00
    XDS-unsw-temporal-32b-Wa-C35-500n34b-OI    |  4 | 88.29±0.21 88.84±0.51  +0.56 | 10.16±0.70  8.94±0.88  -1.22 | 88.35±0.21 88.90±0.51  +0.54
    XDS-unsw-temporal-32b-Wa-C35-500n34b-OI-PREEMP-OLD | 15 | 88.30±0.15 88.48±0.29  +0.19 |  9.47±0.64  9.69±1.12  +0.22 | 88.35±0.15 88.54±0.29  +0.19
    XDS-unsw-temporal-32b-Wb-C35-250n100b-OI   |  3 | 87.80±0.15 87.80±0.15  +0.00 | 12.95±1.16 12.95±1.16  +0.00 | 87.92±0.17 87.92±0.17  +0.00
    XDS-unsw-temporal-32b-Wb-C35-500n34b-OI    |  3 | 87.47±1.16 87.03±1.87  -0.44 | 12.63±5.51 13.28±6.09  +0.66 | 87.59±1.04 87.16±1.75  -0.43
    XDS-unsw-temporal-32b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | 88.42±0.12 86.94±1.93  -1.49 |  9.54±0.53 14.35±8.20  +4.81 | 88.48±0.12 87.12±1.72  -1.37
    XDS-unsw-temporal-32b-Wc-C35-500n34b-OI    |  3 | 88.47±0.04 89.31±0.06  +0.84 |  8.76±0.15  8.44±0.47  -0.32 | 88.52±0.04 89.37±0.06  +0.85
    XDS-unsw-temporal-64b-Wa-C35-250n100b-OI   |  3 | 87.80±0.05 87.80±0.05  +0.00 | 12.08±1.79 12.08±1.79  +0.00 | 87.89±0.08 87.89±0.08  +0.00
    XDS-unsw-temporal-64b-Wa-C35-500n34b-OI    |  3 | 88.38±0.09 88.67±0.29  +0.29 |  9.74±0.62 10.11±1.68  +0.37 | 88.44±0.09 88.75±0.32  +0.30
    XDS-unsw-temporal-64b-Wa-C35-500n34b-OI-PREEMP-OLD |  3 | 88.25±0.16 88.38±0.04  +0.12 | 10.41±1.18  9.34±0.90  -1.07 | 88.33±0.14 88.43±0.02  +0.11
    XDS-unsw-temporal-64b-Wb-C35-250n100b-OI   |  3 | 87.69±0.17 87.69±0.17  +0.00 | 12.50±1.03 12.50±1.03  +0.00 | 87.79±0.17 87.79±0.17  +0.00
    XDS-unsw-temporal-64b-Wb-C35-500n34b-OI    |  3 | 86.42±2.56 86.78±2.27  +0.36 | 13.06±5.68 11.53±2.69  -1.53 | 86.53±2.47 86.85±2.26  +0.32
    XDS-unsw-temporal-64b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | 86.59±2.71 86.95±1.86  +0.35 | 13.15±5.61 15.78±8.51  +2.62 | 86.71±2.62 87.17±1.61  +0.46
    XDS-unsw-temporal-64b-Wc-C35-500n34b-OI    |  3 | 88.39±0.11 89.36±0.08  +0.97 |  9.81±0.33  8.25±0.81  -1.56 | 88.45±0.11 89.41±0.09  +0.96
    XDS-unsw-temporal-8b-Wa-C35-500n34b-OI     |  3 | 88.14±1.23 88.20±0.77  +0.06 | 12.15±3.05 10.19±0.49  -1.96 | 88.24±1.19 88.26±0.78  +0.02
    XDS-unsw-temporal-8b-Wa-C35-500n34b-OI-PREEMP-OLD |  2 | 88.60±0.50 88.80±0.17  +0.20 |  9.61±0.33  9.56±0.37  -0.05 | 88.67±0.51 88.86±0.17  +0.20
    XDS-unsw-temporal-8b-Wb-C35-500n34b-OI     |  3 | 88.31±0.61 88.10±0.74  -0.20 | 10.40±0.43 11.42±1.21  +1.01 | 88.38±0.61 88.19±0.76  -0.18
    XDS-unsw-temporal-8b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | 88.65±0.08 87.22±2.19  -1.43 |  9.48±0.59 13.25±6.97  +3.78 | 88.71±0.07 87.36±2.04  -1.35
    XDS-unsw-temporal-8b-Wc-C35-500n34b-OI     |  3 | 88.85±0.20 89.41±0.12  +0.56 |  9.66±0.66  9.92±0.80  +0.25 | 88.92±0.21 89.49±0.13  +0.57
    XDS-unsw-temporal-96b-Wa-C35-250n100b-OI   |  3 | 87.82±0.17 87.82±0.17  +0.00 | 13.29±0.48 13.29±0.48  +0.00 | 87.95±0.18 87.95±0.18  +0.00
    XDS-unsw-temporal-96b-Wa-C35-500n34b-OI    |  3 | 88.42±0.06 88.38±0.06  -0.05 | 10.15±0.95 10.20±0.93  +0.05 | 88.49±0.05 88.45±0.06  -0.05
    XDS-unsw-temporal-96b-Wa-C35-500n34b-OI-PREEMP-OLD |  3 | 88.45±0.08 88.52±0.17  +0.07 | 10.16±0.93 10.53±1.55  +0.37 | 88.52±0.07 88.60±0.17  +0.08
    XDS-unsw-temporal-96b-Wb-C35-250n100b-OI   |  4 | 87.86±0.18 87.80±0.07  -0.06 | 13.16±0.81 13.48±1.00  +0.32 | 87.98±0.17 87.93±0.08  -0.05
    XDS-unsw-temporal-96b-Wb-C35-500n34b-OI    |  3 | 85.94±3.88 88.22±0.14  +2.28 | 18.05±14.91  9.79±0.67  -8.26 | 86.35±3.28 88.28±0.15  +1.93
    XDS-unsw-temporal-96b-Wb-C35-500n34b-OI-PREEMP-OLD |  3 | 87.37±1.41 87.22±1.73  -0.15 | 11.15±3.18 11.50±2.99  +0.35 | 87.45±1.37 87.30±1.71  -0.15
    XDS-unsw-temporal-96b-Wc-C35-500n34b-OI    |  3 | 88.44±0.02 89.23±0.41  +0.79 |  9.42±0.25  9.35±1.19  -0.07 | 88.49±0.02 89.30±0.40  +0.80

---

# =====================================================================
# SECTION 5 — XDS CROSS-DATASET COHORTS (LEGACY 2-way split)
# produced by scripts/build_xds_5tables.py (existing tool, unmodified)
# =====================================================================

# XDS-unsw-temporal — width × weight cohort breakdown (92 non-OLD completed)

    Total non-OLD completed : 92  |  Total wall: 42.2h  |  Avg/run: 28m
    Latest done : 28/06/2026 21:43 UTC

    Weight schemes:
      Wa (CIC-IoT legacy, ce=0.35 acc=0.30)
      Wb (paper/PUB50, ce=0.10 acc=0.20)
      Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)
      Wc (CE-heavy NEW, ce=0.70 acc=0.10)


## XDS-unsw-temporal-8b-Wa-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.48% |   8.68% |  89.54% | r88021 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.48% |   8.68% |  89.54% | r88021 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.48% |   8.68% |  89.54% | r88021 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  88.83% |   5.74% |  88.85% | r88021 GS best_fpr       val_cal
    Best F1 (FPR<5%)         |  88.14% |   3.98% |  88.15% | r88021 GA best_ce        empirical
    Best F1 (FPR<4%)         |  88.14% |   3.98% |  88.15% | r88021 GA best_ce        empirical
    Best FPR (any F1)        |  88.14% |   3.98% |  88.15% | r88021 GA best_ce        empirical
    Best FPR (F1>80%)        |  88.14% |   3.98% |  88.15% | r88021 GA best_ce        empirical
    Best Acc (any FPR)       |  89.48% |   8.68% |  89.54% | r88021 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±141 neurons | 34±0 bits
    GA Neurons  : 167±191 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.76±0.36 84.20±0.67 |26.65±0.98 28.91±1.45 |85.34±0.31 84.90±0.59
    fixed_05             |85.62±0.60 86.09±0.28 |23.20±2.76 22.30±1.33 |86.05±0.49 86.47±0.23
    platt                |85.07±0.75 85.25±0.54 |25.16±3.13 25.24±1.63 |85.58±0.61 85.76±0.47
    beta                 |82.41±2.42 81.73±2.84 |33.30±5.81 34.97±6.76 |83.42±1.98 82.88±2.29
    empirical            |86.13±1.03 85.14±1.19 |11.92±12.00 20.36±13.29 |86.31±0.78 85.57±0.84
    empirical_cumulative |84.84±0.24 84.66±1.06 |26.38±0.58 27.16±3.15 |85.40±0.21 85.27±0.91
    val_cal              |88.14±1.23 88.20±0.77 |12.15±3.05 10.19±0.49 |88.24±1.19 88.26±0.78

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 167±191 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.40±0.70 84.20±0.67 |27.81±2.10 28.91±1.45 |85.03±0.59 84.90±0.59
    fixed_05             |85.73±0.51 86.09±0.28 |22.80±2.42 22.30±1.33 |86.13±0.42 86.47±0.23
    platt                |85.12±0.70 85.25±0.54 |24.93±2.92 25.24±1.63 |85.62±0.58 85.76±0.47
    beta                 |82.36±2.37 81.73±2.84 |33.34±5.78 34.97±6.76 |83.36±1.93 82.88±2.29
    empirical            |85.85±0.66 85.14±1.19 |11.71±12.18 20.36±13.29 |86.03±0.36 85.57±0.84
    empirical_cumulative |84.94±0.27 84.66±1.06 |26.02±0.79 27.16±3.15 |85.48±0.24 85.27±0.91
    val_cal              |88.10±1.20 88.20±0.77 |11.99±3.22 10.19±0.49 |88.20±1.15 88.26±0.78

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 202±198 neurons | 31±3 bits
    GA Neurons  : 236±202 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.69±0.80 84.10±0.86 |30.07±2.51 29.09±2.17 |84.46±0.65 84.81±0.74
    fixed_05             |85.57±0.71 85.21±1.18 |23.99±2.02 25.12±3.75 |86.03±0.63 85.73±1.00
    platt                |85.17±1.06 84.95±0.77 |25.40±3.08 26.28±2.29 |85.69±0.93 85.51±0.66
    beta                 |82.05±2.66 82.13±2.56 |34.22±6.57 34.17±6.61 |83.14±2.13 83.22±2.03
    empirical            |85.76±2.68 87.43±2.31 |15.05±15.09 15.88±10.96 |86.08±2.23 87.70±1.96
    empirical_cumulative |85.48±0.79 85.00±0.97 |24.35±2.16 25.82±3.15 |85.95±0.71 85.54±0.81
    val_cal              |88.77±0.13 88.39±1.17 | 8.96±2.83 11.37±1.93 |88.83±0.11 88.48±1.14

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±141 neurons | 34±0 bits
    GA Neurons  : 167±191 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.76±0.36 84.20±0.67 |26.65±0.98 28.91±1.45 |85.34±0.31 84.90±0.59
    fixed_05             |85.62±0.60 86.09±0.28 |23.20±2.76 22.30±1.33 |86.05±0.49 86.47±0.23
    platt                |85.07±0.75 85.25±0.54 |25.16±3.13 25.24±1.63 |85.58±0.61 85.76±0.47
    beta                 |82.41±2.42 81.73±2.84 |33.30±5.81 34.97±6.76 |83.42±1.98 82.88±2.29
    empirical            |86.13±1.03 85.14±1.19 |11.92±12.00 20.36±13.29 |86.31±0.78 85.57±0.84
    empirical_cumulative |84.84±0.24 84.66±1.06 |26.38±0.58 27.16±3.15 |85.40±0.21 85.27±0.91
    val_cal              |88.14±1.23 88.20±0.77 |12.15±3.05 10.19±0.49 |88.24±1.19 88.26±0.78

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±71 neurons | 34±0 bits
    GA Neurons  : 396±86 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.31±0.45 83.91±0.76 |28.02±1.40 29.37±2.27 |84.95±0.37 84.63±0.62
    fixed_05             |84.98±0.19 84.38±0.56 |25.60±0.45 27.80±1.79 |85.50±0.17 85.02±0.46
    platt                |84.45±0.14 84.30±0.25 |27.46±0.22 28.15±0.78 |85.07±0.13 84.95±0.21
    beta                 |83.68±0.15 83.51±0.20 |30.22±0.24 30.71±0.51 |84.45±0.13 84.31±0.17
    empirical            |86.89±0.26 88.24±0.40 | 5.27±0.53  5.29±1.77 |86.89±0.26 88.25±0.42
    empirical_cumulative |85.83±1.79 85.15±0.07 |22.53±6.54 25.28±0.43 |86.26±1.56 85.66±0.05
    val_cal              |88.74±0.12 89.36±0.13 | 9.73±0.67  9.52±0.73 |88.81±0.12 89.43±0.13


## XDS-unsw-temporal-8b-Wb-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.51% |   9.50% |  89.59% | r74627 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.51% |   9.50% |  89.59% | r74627 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.51% |   9.50% |  89.59% | r74627 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  86.65% |   2.77% |  86.65% | r11760 GA best_ce        empirical
    Best F1 (FPR<5%)         |  86.65% |   2.77% |  86.65% | r11760 GA best_ce        empirical
    Best F1 (FPR<4%)         |  86.65% |   2.77% |  86.65% | r11760 GA best_ce        empirical
    Best FPR (any F1)        |  86.34% |   2.58% |  86.34% | r88021 GA best_ce        empirical
    Best FPR (F1>80%)        |  86.34% |   2.58% |  86.34% | r88021 GA best_ce        empirical
    Best Acc (any FPR)       |  89.51% |   9.50% |  89.59% | r74627 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 202±263 neurons | 33±1 bits
    GA Neurons  : 42±59 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.12±0.14 84.95±1.16 |28.34±1.02 26.78±3.21 |84.78±0.09 85.55±1.00
    fixed_05             |84.94±0.52 86.81±0.41 |25.51±1.13 19.99±1.19 |85.46±0.48 87.11±0.38
    platt                |84.43±0.09 85.73±0.67 |27.13±0.64 23.60±2.61 |85.03±0.08 86.17±0.57
    beta                 |82.19±2.21 81.27±2.68 |33.95±5.46 36.55±6.64 |83.24±1.79 82.54±2.11
    empirical            |84.17±2.14 84.08±1.99 |23.22±15.17 29.27±5.50 |84.75±1.64 84.83±1.66
    empirical_cumulative |87.82±0.19 87.62±1.16 |13.44±2.22 16.87±3.95 |87.95±0.24 87.85±1.07
    val_cal              |88.31±0.61 88.10±0.74 |10.40±0.43 11.42±1.21 |88.38±0.61 88.19±0.76

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 202±263 neurons | 33±1 bits
    GA Neurons  : 42±59 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.12±0.14 84.95±1.16 |28.34±1.02 26.78±3.21 |84.78±0.09 85.55±1.00
    fixed_05             |84.94±0.52 86.81±0.41 |25.51±1.13 19.99±1.19 |85.46±0.48 87.11±0.38
    platt                |84.43±0.09 85.73±0.67 |27.13±0.64 23.60±2.61 |85.03±0.08 86.17±0.57
    beta                 |82.19±2.21 81.27±2.68 |33.95±5.46 36.55±6.64 |83.24±1.79 82.54±2.11
    empirical            |84.17±2.14 84.08±1.99 |23.22±15.17 29.27±5.50 |84.75±1.64 84.83±1.66
    empirical_cumulative |87.82±0.19 87.62±1.16 |13.44±2.22 16.87±3.95 |87.95±0.24 87.85±1.07
    val_cal              |88.31±0.61 88.10±0.74 |10.40±0.43 11.42±1.21 |88.38±0.61 88.19±0.76

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 202±279 neurons | 29±7 bits
    GA Neurons  : 71±110 neurons | 27±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.06±2.66 82.36±1.86 |28.35±7.39 33.17±5.11 |84.78±2.25 83.35±1.55
    fixed_05             |84.39±2.80 83.12±1.85 |27.15±7.73 30.43±4.44 |85.05±2.37 83.91±1.57
    platt                |84.15±2.68 83.16±1.65 |28.05±7.29 29.95±3.23 |84.84±2.27 83.91±1.47
    beta                 |80.77±3.27 80.84±2.22 |36.93±7.60 36.78±5.38 |82.08±2.62 82.10±1.78
    empirical            |82.44±3.79 83.74±4.26 |26.81±19.97 26.54±16.51 |83.37±2.85 84.53±3.61
    empirical_cumulative |87.82±1.24 85.85±3.98 | 9.92±2.14 14.05±10.57 |87.89±1.28 86.02±3.77
    val_cal              |87.86±1.28 86.62±2.82 | 9.27±1.76  7.29±2.16 |87.91±1.31 86.64±2.85

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 202±263 neurons | 33±1 bits
    GA Neurons  : 42±59 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.12±0.14 84.95±1.16 |28.34±1.02 26.78±3.21 |84.78±0.09 85.55±1.00
    fixed_05             |84.94±0.52 86.81±0.41 |25.51±1.13 19.99±1.19 |85.46±0.48 87.11±0.38
    platt                |84.43±0.09 85.73±0.67 |27.13±0.64 23.60±2.61 |85.03±0.08 86.17±0.57
    beta                 |82.19±2.21 81.27±2.68 |33.95±5.46 36.55±6.64 |83.24±1.79 82.54±2.11
    empirical            |84.17±2.14 84.08±1.99 |23.22±15.17 29.27±5.50 |84.75±1.64 84.83±1.66
    empirical_cumulative |87.82±0.19 87.62±1.16 |13.44±2.22 16.87±3.95 |87.95±0.24 87.85±1.07
    val_cal              |88.31±0.61 88.10±0.74 |10.40±0.43 11.42±1.21 |88.38±0.61 88.19±0.76

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 352±117 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.98±0.16 83.96±0.40 |29.04±0.48 29.01±0.99 |84.68±0.13 84.66±0.34
    fixed_05             |85.07±0.18 84.41±0.57 |25.31±0.48 27.55±1.57 |85.58±0.15 85.03±0.49
    platt                |84.46±0.13 84.12±0.27 |27.37±0.33 28.45±0.59 |85.07±0.12 84.79±0.24
    beta                 |83.62±0.07 83.35±0.33 |30.30±0.11 31.05±0.82 |84.39±0.07 84.17±0.28
    empirical            |85.99±0.82 86.34±0.31 | 4.48±1.09  9.27±11.43 |86.00±0.82 86.47±0.16
    empirical_cumulative |88.23±0.48 89.20±0.26 |13.65±2.15 10.71±0.81 |88.37±0.44 89.29±0.26
    val_cal              |88.76±0.12 89.31±0.24 | 9.28±0.80  9.69±0.60 |88.82±0.11 89.38±0.23


## XDS-unsw-temporal-8b-Wc-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.72% |  10.37% |  89.81% | r88021 GA best_fpr       val_cal
    Best F1 (FPR<14%)        |  89.72% |  10.37% |  89.81% | r88021 GA best_fpr       val_cal
    Best F1 (FPR<10%)        |  89.50% |   9.03% |  89.57% | r74627 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  88.64% |   5.43% |  88.66% | r74627 GA best_fpr       empirical
    Best F1 (FPR<5%)         |  86.98% |   4.27% |  86.98% | r74627 GA best_acc       empirical
    Best F1 (FPR<4%)         |  86.72% |   3.29% |  86.72% | r88021 GA best_acc       empirical
    Best FPR (any F1)        |  85.77% |   2.60% |  85.78% | r11760 GA best_fpr       empirical
    Best FPR (F1>80%)        |  85.77% |   2.60% |  85.78% | r11760 GA best_fpr       empirical
    Best Acc (any FPR)       |  89.72% |  10.37% |  89.81% | r88021 GA best_fpr       val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 33±1 bits
    GA Neurons  : 400±44 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.21±0.27 84.00±0.61 |28.36±0.69 28.93±1.59 |84.87±0.23 84.70±0.53
    fixed_05             |85.07±0.34 83.78±0.19 |25.36±0.95 29.64±0.87 |85.58±0.29 84.52±0.14
    platt                |84.47±0.19 84.11±0.13 |27.41±0.53 28.64±0.29 |85.08±0.17 84.78±0.12
    beta                 |83.71±0.24 83.38±0.21 |30.07±0.58 30.90±0.33 |84.47±0.21 84.19±0.19
    empirical            |86.95±0.39 86.32±0.32 |10.83±8.37  2.84±0.17 |87.07±0.22 86.32±0.32
    empirical_cumulative |86.30±0.80 86.39±0.64 |21.49±2.42 21.36±1.75 |86.66±0.72 86.75±0.58
    val_cal              |88.85±0.20 89.41±0.12 | 9.66±0.66  9.92±0.80 |88.92±0.21 89.49±0.13

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 34±0 bits
    GA Neurons  : 413±44 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.21±0.50 84.21±0.30 |28.23±1.67 28.34±0.67 |84.86±0.41 84.87±0.26
    fixed_05             |85.28±0.14 84.25±0.48 |24.65±0.39 28.18±1.56 |85.76±0.12 84.91±0.40
    platt                |84.57±0.07 84.23±0.22 |26.99±0.19 28.30±0.49 |85.16±0.06 84.89±0.20
    beta                 |83.65±0.04 83.43±0.18 |30.18±0.14 30.81±0.26 |84.42±0.03 84.23±0.17
    empirical            |86.03±0.93 86.73±0.24 | 4.33±1.16  3.49±0.70 |86.04±0.93 86.73±0.24
    empirical_cumulative |85.53±0.22 85.82±0.59 |23.88±0.85 23.03±1.61 |85.98±0.18 86.24±0.53
    val_cal              |88.77±0.13 89.39±0.10 | 9.39±0.71  9.59±0.51 |88.84±0.12 89.46±0.09

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 367±58 neurons | 33±1 bits
    GA Neurons  : 401±57 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.88±0.29 83.98±0.59 |29.42±0.98 28.96±1.54 |84.61±0.24 84.67±0.51
    fixed_05             |85.34±0.15 83.90±0.37 |24.62±0.41 29.36±1.52 |85.82±0.13 84.62±0.28
    platt                |84.60±0.09 84.13±0.02 |26.99±0.16 28.58±0.35 |85.19±0.09 84.81±0.00
    beta                 |83.54±0.05 83.55±0.17 |30.53±0.04 30.42±0.12 |84.33±0.05 84.33±0.17
    empirical            |86.64±0.57 87.62±1.60 | 5.08±0.60  4.72±1.87 |86.64±0.57 87.64±1.61
    empirical_cumulative |86.69±0.28 87.54±0.56 |20.20±1.01 18.00±1.48 |87.00±0.25 87.79±0.52
    val_cal              |88.85±0.09 89.53±0.30 | 9.49±0.22 10.23±0.24 |88.91±0.08 89.62±0.31

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 34±0 bits
    GA Neurons  : 413±44 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.10±0.68 84.21±0.30 |28.56±2.22 28.34±0.67 |84.78±0.56 84.87±0.26
    fixed_05             |85.11±0.16 84.25±0.48 |25.08±0.37 28.18±1.56 |85.60±0.14 84.91±0.40
    platt                |84.51±0.05 84.23±0.22 |27.16±0.10 28.30±0.49 |85.11±0.05 84.89±0.20
    beta                 |83.59±0.10 83.43±0.18 |30.36±0.17 30.81±0.26 |84.37±0.10 84.23±0.17
    empirical            |86.10±0.99 86.73±0.24 | 4.42±1.30  3.49±0.70 |86.10±0.98 86.73±0.24
    empirical_cumulative |85.87±0.50 85.82±0.59 |22.63±1.84 23.03±1.61 |86.27±0.43 86.24±0.53
    val_cal              |88.73±0.20 89.39±0.10 | 9.48±0.75  9.59±0.51 |88.79±0.20 89.46±0.09

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 33±1 bits
    GA Neurons  : 400±44 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.21±0.27 84.00±0.61 |28.36±0.69 28.93±1.59 |84.87±0.23 84.70±0.53
    fixed_05             |85.07±0.34 83.78±0.19 |25.36±0.95 29.64±0.87 |85.58±0.29 84.52±0.14
    platt                |84.47±0.19 84.11±0.13 |27.41±0.53 28.64±0.29 |85.08±0.17 84.78±0.12
    beta                 |83.71±0.24 83.38±0.21 |30.07±0.58 30.90±0.33 |84.47±0.21 84.19±0.19
    empirical            |86.95±0.39 86.32±0.32 |10.83±8.37  2.84±0.17 |87.07±0.22 86.32±0.32
    empirical_cumulative |86.30±0.80 86.39±0.64 |21.49±2.42 21.36±1.75 |86.66±0.72 86.75±0.58
    val_cal              |88.85±0.20 89.41±0.12 | 9.66±0.66  9.92±0.80 |88.92±0.21 89.49±0.13


## XDS-unsw-temporal-16b-Wa-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.07% |   9.04% |  89.13% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<14%)        |  89.07% |   9.04% |  89.13% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<10%)        |  89.07% |   9.04% |  89.13% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<6%)         |  87.39% |   3.92% |  87.39% | r74627 GA best_fpr       empirical
    Best F1 (FPR<5%)         |  87.39% |   3.92% |  87.39% | r74627 GA best_fpr       empirical
    Best F1 (FPR<4%)         |  87.39% |   3.92% |  87.39% | r74627 GA best_fpr       empirical
    Best FPR (any F1)        |  85.11% |   1.65% |  85.13% | r88021 GA best_ce        empirical
    Best FPR (F1>80%)        |  85.11% |   1.65% |  85.13% | r88021 GA best_ce        empirical
    Best Acc (any FPR)       |  89.07% |   9.04% |  89.13% | r11760 GA best_fpr       val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 357±207 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.03±0.17 84.09±0.17 |28.29±0.53 28.47±0.41 |84.68±0.14 84.76±0.15
    fixed_05             |85.01±0.17 85.53±0.12 |25.13±0.56 23.83±0.27 |85.51±0.14 85.97±0.10
    platt                |84.32±0.08 84.75±0.16 |27.32±0.27 26.38±0.35 |84.92±0.07 85.31±0.15
    beta                 |83.27±0.05 83.57±0.04 |30.51±0.17 29.96±0.15 |84.05±0.04 84.32±0.03
    empirical            |83.52±4.24 86.52±1.48 |17.53±22.07  5.57±2.18 |84.13±3.18 86.53±1.50
    empirical_cumulative |84.88±0.46 84.40±0.16 |25.55±1.43 27.53±0.43 |85.40±0.39 85.02±0.14
    val_cal              |88.41±0.03 88.40±0.15 | 8.94±0.32  9.78±0.98 |88.46±0.03 88.46±0.14

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 433±115 neurons | 34±0 bits
    GA Neurons  : 361±211 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.09±0.34 84.03±0.21 |28.11±0.99 28.69±0.60 |84.73±0.29 84.71±0.18
    fixed_05             |85.24±0.17 85.54±0.09 |24.48±0.55 23.77±0.21 |85.71±0.14 85.98±0.08
    platt                |84.47±0.10 84.76±0.17 |26.92±0.34 26.35±0.37 |85.05±0.09 85.32±0.15
    beta                 |83.38±0.07 83.61±0.06 |30.30±0.32 29.90±0.13 |84.15±0.06 84.36±0.06
    empirical            |83.68±4.38 85.81±2.86 |17.42±22.16  5.31±2.67 |84.29±3.33 85.83±2.86
    empirical_cumulative |84.48±0.03 84.49±0.20 |26.87±0.02 27.24±0.49 |85.06±0.03 85.09±0.17
    val_cal              |88.40±0.06 88.44±0.22 | 8.94±0.73 10.01±0.87 |88.46±0.07 88.51±0.21

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±71 neurons | 34±0 bits
    GA Neurons  : 333±142 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.91±0.09 84.14±0.12 |28.90±0.26 28.37±0.62 |84.60±0.09 84.80±0.08
    fixed_05             |85.60±1.59 84.83±0.75 |23.55±4.61 26.26±2.36 |86.05±1.42 85.39±0.64
    platt                |85.22±1.86 84.43±0.39 |24.87±5.21 27.54±1.10 |85.73±1.65 85.05±0.33
    beta                 |81.73±2.37 83.33±0.24 |34.68±6.05 30.87±0.65 |82.84±1.88 84.14±0.20
    empirical            |85.99±1.91 87.53±1.26 |13.64±13.44  4.96±1.68 |86.23±1.55 87.54±1.28
    empirical_cumulative |85.57±1.57 85.09±0.09 |23.70±4.21 25.48±0.50 |86.02±1.41 85.61±0.07
    val_cal              |88.55±0.30 88.81±0.23 | 9.50±1.81  9.40±0.53 |88.61±0.32 88.87±0.23

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 333±58 neurons | 34±0 bits
    GA Neurons  : 357±207 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.92±0.28 84.09±0.17 |28.58±0.87 28.47±0.41 |84.59±0.23 84.76±0.15
    fixed_05             |85.24±0.13 85.53±0.12 |24.38±0.43 23.83±0.27 |85.71±0.11 85.97±0.10
    platt                |84.44±0.11 84.75±0.16 |26.92±0.34 26.38±0.35 |85.03±0.09 85.31±0.15
    beta                 |83.29±0.16 83.57±0.04 |30.44±0.44 29.96±0.15 |84.07±0.13 84.32±0.03
    empirical            |85.89±0.39 86.52±1.48 | 4.82±0.18  5.57±2.18 |85.89±0.39 86.53±1.50
    empirical_cumulative |84.42±0.43 84.40±0.16 |27.07±1.39 27.53±0.43 |85.02±0.36 85.02±0.14
    val_cal              |88.44±0.08 88.40±0.15 | 8.88±0.35  9.78±0.98 |88.49±0.09 88.46±0.14

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±0 neurons | 32±0 bits
    GA Neurons  : 326±135 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.92±0.28 84.12±0.25 |28.62±0.88 28.54±0.85 |84.59±0.23 84.79±0.20
    fixed_05             |84.85±0.06 84.33±0.53 |25.75±0.23 27.90±1.38 |85.37±0.05 84.97±0.46
    platt                |84.20±0.06 84.24±0.36 |27.68±0.19 28.19±0.82 |84.82±0.05 84.89±0.32
    beta                 |83.21±0.09 83.35±0.23 |30.78±0.27 30.87±0.59 |84.01±0.07 84.16±0.19
    empirical            |83.21±4.52 86.93±1.87 |17.69±23.10  4.00±2.66 |83.87±3.38 86.94±1.88
    empirical_cumulative |84.73±0.56 84.93±0.19 |26.05±1.80 26.03±0.85 |85.27±0.48 85.47±0.15
    val_cal              |88.48±0.07 88.93±0.13 | 9.00±0.17  9.52±0.73 |88.53±0.07 89.00±0.12


## XDS-unsw-temporal-16b-Wb-500n34b  (30 flows × 2 phases, seeds: [3922, 11760, 13710, 13922, 25737, 27140, 27384, 35141, 40417, 41448, 42823, 43097, 45199, 49994, 56926, 63480, 65268, 69269, 73300, 74627, 75197, 75840, 76297, 77567, 84446, 85011, 87167, 88021, 91849, 99534])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.34% |   9.76% |  89.42% | r56926 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.34% |   9.76% |  89.42% | r56926 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.34% |   9.76% |  89.42% | r56926 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  88.34% |   5.37% |  88.36% | r65268 GA best_ce        empirical
    Best F1 (FPR<5%)         |  88.14% |   4.76% |  88.15% | r42823 GA best_ce        empirical
    Best F1 (FPR<4%)         |  87.27% |   3.49% |  87.27% | r84446 GA best_ce        empirical
    Best FPR (any F1)        |  85.39% |   2.26% |  85.40% | r63480 GA best_ce        empirical
    Best FPR (F1>80%)        |  85.39% |   2.26% |  85.40% | r63480 GA best_ce        empirical
    Best Acc (any FPR)       |  89.34% |   9.76% |  89.42% | r56926 GA best_ce        val_cal

### best_fitness  (GS: 30 runs | GA: 30 runs)
    Grid Search : 114±162 neurons | 33±2 bits
    GA Neurons  : 131±158 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.22±1.25 83.92±1.10 |30.36±3.53 28.52±3.14 |84.01±1.03 84.60±0.93
    fixed_05             |84.39±1.88 84.34±1.67 |26.25±5.61 26.63±4.78 |84.98±1.57 84.94±1.40
    platt                |83.87±1.56 84.28±1.26 |28.03±4.40 27.25±3.47 |84.54±1.30 84.90±1.08
    beta                 |79.60±3.53 79.67±3.03 |39.17±8.34 39.21±7.20 |81.15±2.80 81.19±2.40
    empirical            |83.34±2.78 83.24±2.23 |26.85±12.30 27.84±11.14 |84.10±2.30 84.02±1.79
    empirical_cumulative |85.79±1.95 85.57±1.95 |20.60±6.18 20.90±6.28 |86.14±1.72 85.93±1.73
    val_cal              |86.47±1.95 85.88±2.02 |14.62±6.54 17.05±7.35 |86.64±1.83 86.13±1.82

### best_f1  (GS: 30 runs | GA: 30 runs)
    Grid Search : 114±162 neurons | 33±2 bits
    GA Neurons  : 131±158 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.22±1.25 83.92±1.10 |30.36±3.53 28.52±3.14 |84.01±1.03 84.60±0.93
    fixed_05             |84.39±1.88 84.34±1.67 |26.25±5.61 26.63±4.78 |84.98±1.57 84.94±1.40
    platt                |83.87±1.56 84.28±1.26 |28.03±4.40 27.25±3.47 |84.54±1.30 84.90±1.08
    beta                 |79.60±3.53 79.67±3.03 |39.17±8.34 39.21±7.20 |81.15±2.80 81.19±2.40
    empirical            |83.34±2.78 83.24±2.23 |26.85±12.30 27.84±11.14 |84.10±2.30 84.02±1.79
    empirical_cumulative |85.79±1.95 85.57±1.95 |20.60±6.18 20.90±6.28 |86.14±1.72 85.93±1.73
    val_cal              |86.47±1.95 85.88±2.02 |14.62±6.54 17.05±7.35 |86.64±1.83 86.13±1.82

### best_fpr  (GS: 30 runs | GA: 30 runs)
    Grid Search : 238±167 neurons | 21±13 bits
    GA Neurons  : 165±173 neurons | 22±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.78±5.68 79.64±5.49 |35.38±11.39 35.86±11.22 |81.07±4.77 80.96±4.64
    fixed_05             |81.24±5.28 80.75±4.41 |23.95±4.06 25.99±6.06 |81.61±5.49 81.24±4.63
    platt                |80.41±5.76 79.74±5.71 |31.81±8.77 33.99±8.27 |81.27±5.33 80.74±5.27
    beta                 |77.67±7.03 77.45±6.06 |40.20±11.49 41.77±10.43 |79.36±5.83 79.29±4.87
    empirical            |80.79±7.24 78.97±8.04 |24.40±22.31 33.20±22.56 |81.92±5.79 80.68±6.33
    empirical_cumulative |84.14±5.67 83.38±5.77 | 9.34±3.37  9.22±6.80 |84.26±5.62 83.59±5.58
    val_cal              |84.47±5.46 84.28±4.97 | 7.96±2.49  6.75±3.37 |84.55±5.41 84.38±4.90

### best_acc  (GS: 30 runs | GA: 30 runs)
    Grid Search : 120±152 neurons | 33±2 bits
    GA Neurons  : 131±158 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.84±1.29 83.99±1.43 |31.19±3.45 28.33±4.03 |83.68±1.08 84.67±1.21
    fixed_05             |83.55±1.84 84.30±1.66 |28.58±5.30 26.77±4.76 |84.25±1.55 84.90±1.41
    platt                |83.19±1.45 84.13±1.30 |29.82±3.88 27.72±3.53 |83.94±1.22 84.77±1.11
    beta                 |78.88±3.39 79.60±3.08 |40.92±7.55 39.34±7.24 |80.58±2.68 81.14±2.44
    empirical            |82.85±2.87 83.05±2.38 |28.96±11.38 28.29±11.44 |83.70±2.37 83.86±1.91
    empirical_cumulative |84.90±2.21 85.48±1.98 |23.19±6.96 20.62±6.37 |85.36±1.94 85.83±1.78
    val_cal              |85.37±2.25 85.80±2.04 |18.37±8.10 16.80±7.32 |85.66±2.05 86.04±1.84

### best_ce  (GS: 30 runs | GA: 30 runs)
    Grid Search : 410±99 neurons | 34±1 bits
    GA Neurons  : 309±107 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.90±0.24 83.97±0.34 |28.76±0.75 28.71±1.04 |84.58±0.20 84.65±0.29
    fixed_05             |84.95±0.19 84.27±0.44 |25.45±0.54 27.74±1.25 |85.46±0.17 84.90±0.37
    platt                |84.31±0.14 84.12±0.24 |27.45±0.34 28.23±0.59 |84.92±0.12 84.78±0.21
    beta                 |83.29±0.11 83.26±0.21 |30.64±0.25 30.92±0.48 |84.08±0.10 84.06±0.18
    empirical            |85.74±1.62 87.18±1.14 | 5.74±7.08  5.57±4.51 |85.80±1.36 87.22±1.11
    empirical_cumulative |88.03±0.70 88.45±0.34 |12.99±2.74 12.40±1.04 |88.15±0.62 88.56±0.33
    val_cal              |88.54±0.10 88.86±0.24 | 9.12±0.58  8.78±1.18 |88.59±0.10 88.91±0.24


## XDS-unsw-temporal-16b-Wc-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.72% |   8.96% |  89.78% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<14%)        |  89.72% |   8.96% |  89.78% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<10%)        |  89.72% |   8.96% |  89.78% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<6%)         |  87.66% |   3.64% |  87.66% | r88021 GA best_ce        empirical
    Best F1 (FPR<5%)         |  87.66% |   3.64% |  87.66% | r88021 GA best_ce        empirical
    Best F1 (FPR<4%)         |  87.66% |   3.64% |  87.66% | r88021 GA best_ce        empirical
    Best FPR (any F1)        |  85.58% |   1.64% |  85.59% | r11760 GA best_fpr       empirical
    Best FPR (F1>80%)        |  85.58% |   1.64% |  85.59% | r11760 GA best_fpr       empirical
    Best Acc (any FPR)       |  89.72% |   8.96% |  89.78% | r11760 GA best_fpr       val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 433±58 neurons | 33±1 bits
    GA Neurons  : 436±65 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.86±0.17 84.38±0.05 |28.81±0.50 28.01±0.40 |84.54±0.14 85.02±0.03
    fixed_05             |84.75±0.07 84.44±0.73 |26.05±0.26 27.90±2.01 |85.29±0.06 85.09±0.62
    platt                |84.17±0.06 84.49±0.33 |27.81±0.17 27.71±0.75 |84.79±0.05 85.12±0.29
    beta                 |83.25±0.07 83.58±0.19 |30.74±0.22 30.39±0.46 |84.05±0.06 84.36±0.16
    empirical            |86.16±0.46 86.86±0.88 | 4.68±0.15  3.00±1.01 |86.16±0.46 86.87±0.88
    empirical_cumulative |85.53±0.29 86.85±0.77 |23.43±1.04 20.50±2.23 |85.95±0.25 87.18±0.69
    val_cal              |88.47±0.06 89.50±0.20 | 9.55±0.79  9.41±1.19 |88.53±0.07 89.57±0.21

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±141 neurons | 34±0 bits
    GA Neurons  : 431±102 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.99±0.18 84.39±0.11 |28.47±0.58 27.56±0.67 |84.65±0.15 85.01±0.07
    fixed_05             |85.03±0.26 85.05±0.21 |25.15±0.75 25.34±1.03 |85.53±0.23 85.56±0.16
    platt                |84.35±0.23 84.59±0.09 |27.29±0.57 26.92±0.48 |84.95±0.21 85.17±0.07
    beta                 |83.30±0.15 83.49±0.16 |30.55±0.30 30.29±0.04 |84.09±0.13 84.26±0.16
    empirical            |83.18±4.34 85.95±1.04 |17.20±22.37  3.56±1.52 |83.80±3.42 85.96±1.04
    empirical_cumulative |84.89±0.30 85.44±0.51 |25.63±1.04 24.22±1.25 |85.41±0.25 85.90±0.46
    val_cal              |88.40±0.04 88.78±0.62 | 9.14±0.45  8.71±0.18 |88.45±0.05 88.83±0.62

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 52±67 neurons | 32±0 bits
    GA Neurons  : 425±64 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.09±0.49 84.36±0.13 |28.61±1.78 28.13±0.40 |84.77±0.39 85.01±0.11
    fixed_05             |85.45±1.82 84.21±0.56 |24.24±5.57 28.49±1.64 |85.94±1.59 84.88±0.47
    platt                |85.16±1.93 84.36±0.20 |25.16±5.54 28.10±0.56 |85.69±1.71 85.01±0.17
    beta                 |80.22±2.06 83.57±0.12 |38.82±5.46 30.55±0.19 |81.66±1.59 84.36±0.11
    empirical            |84.06±3.32 86.37±0.87 |28.12±10.29  2.71±1.19 |84.81±2.77 86.37±0.86
    empirical_cumulative |87.16±0.41 87.13±1.30 |18.81±1.32 19.55±3.74 |87.42±0.38 87.44±1.19
    val_cal              |88.25±0.67 89.46±0.32 |10.10±1.57  9.41±0.41 |88.31±0.70 89.53±0.32

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 433±115 neurons | 34±0 bits
    GA Neurons  : 431±102 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.99±0.18 84.39±0.11 |28.47±0.58 27.56±0.67 |84.65±0.15 85.01±0.07
    fixed_05             |85.03±0.26 85.05±0.21 |25.15±0.75 25.34±1.03 |85.53±0.23 85.56±0.16
    platt                |84.35±0.23 84.59±0.09 |27.29±0.57 26.92±0.48 |84.95±0.21 85.17±0.07
    beta                 |83.30±0.15 83.49±0.16 |30.55±0.30 30.29±0.04 |84.09±0.13 84.26±0.16
    empirical            |83.18±4.34 85.95±1.04 |17.20±22.37  3.56±1.52 |83.80±3.42 85.96±1.04
    empirical_cumulative |84.89±0.30 85.44±0.51 |25.63±1.04 24.22±1.25 |85.41±0.25 85.90±0.46
    val_cal              |88.40±0.04 88.78±0.62 | 9.14±0.45  8.71±0.18 |88.45±0.05 88.83±0.62

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 433±58 neurons | 33±1 bits
    GA Neurons  : 436±65 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.86±0.17 84.38±0.05 |28.81±0.50 28.01±0.40 |84.54±0.14 85.02±0.03
    fixed_05             |84.75±0.07 84.44±0.73 |26.05±0.26 27.90±2.01 |85.29±0.06 85.09±0.62
    platt                |84.17±0.06 84.49±0.33 |27.81±0.17 27.71±0.75 |84.79±0.05 85.12±0.29
    beta                 |83.25±0.07 83.58±0.19 |30.74±0.22 30.39±0.46 |84.05±0.06 84.36±0.16
    empirical            |86.16±0.46 86.86±0.88 | 4.68±0.15  3.00±1.01 |86.16±0.46 86.87±0.88
    empirical_cumulative |85.53±0.29 86.85±0.77 |23.43±1.04 20.50±2.23 |85.95±0.25 87.18±0.69
    val_cal              |88.47±0.06 89.50±0.20 | 9.55±0.79  9.41±1.19 |88.53±0.07 89.57±0.21


## XDS-unsw-temporal-32b-Wa-250n100b  (3 flows × 2 phases, seeds: [14675, 25694, 52015])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  88.31% |   9.11% |  88.36% | r52015 GS best_ce        val_cal
    Best F1 (FPR<14%)        |  88.31% |   9.11% |  88.36% | r52015 GS best_ce        val_cal
    Best F1 (FPR<10%)        |  88.31% |   9.11% |  88.36% | r52015 GS best_ce        val_cal
    Best F1 (FPR<6%)         |       — |       — |       — | —
    Best F1 (FPR<5%)         |       — |       — |       — | —
    Best F1 (FPR<4%)         |       — |       — |       — | —
    Best FPR (any F1)        |  88.31% |   9.11% |  88.36% | r52015 GS best_ce        val_cal
    Best FPR (F1>80%)        |  88.31% |   9.11% |  88.36% | r52015 GS best_ce        val_cal
    Best Acc (any FPR)       |  88.31% |   9.11% |  88.36% | r52015 GS best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 183±29 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.94±0.11 84.94±0.11 |26.26±0.59 26.26±0.59 |85.50±0.09 85.50±0.09
    fixed_05             |86.87±0.08 86.87±0.08 |18.77±0.35 18.77±0.35 |87.13±0.07 87.13±0.07
    platt                |85.88±0.12 85.88±0.12 |22.85±0.21 22.85±0.21 |86.28±0.11 86.28±0.11
    beta                 |84.60±0.23 84.60±0.23 |27.46±0.46 27.46±0.46 |85.22±0.21 85.22±0.21
    empirical            |82.17±1.07 82.17±1.07 |34.65±2.87 34.65±2.87 |83.25±0.86 83.25±0.86
    empirical_cumulative |85.22±0.21 85.22±0.21 |25.21±0.45 25.21±0.45 |85.72±0.19 85.72±0.19
    val_cal              |87.86±0.07 87.86±0.07 |12.22±0.57 12.22±0.57 |87.96±0.07 87.96±0.07

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 183±29 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.94±0.11 84.94±0.11 |26.26±0.59 26.26±0.59 |85.50±0.09 85.50±0.09
    fixed_05             |86.87±0.08 86.87±0.08 |18.77±0.35 18.77±0.35 |87.13±0.07 87.13±0.07
    platt                |85.88±0.12 85.88±0.12 |22.85±0.21 22.85±0.21 |86.28±0.11 86.28±0.11
    beta                 |84.60±0.23 84.60±0.23 |27.46±0.46 27.46±0.46 |85.22±0.21 85.22±0.21
    empirical            |82.17±1.07 82.17±1.07 |34.65±2.87 34.65±2.87 |83.25±0.86 83.25±0.86
    empirical_cumulative |85.22±0.21 85.22±0.21 |25.21±0.45 25.21±0.45 |85.72±0.19 85.72±0.19
    val_cal              |87.86±0.07 87.86±0.07 |12.22±0.57 12.22±0.57 |87.96±0.07 87.96±0.07

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 167±76 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.94±0.40 84.94±0.40 |26.37±1.11 26.37±1.11 |85.50±0.35 85.50±0.35
    fixed_05             |87.14±0.23 87.14±0.23 |18.22±0.50 18.22±0.50 |87.39±0.21 87.39±0.21
    platt                |86.03±0.11 86.03±0.11 |22.64±0.23 22.64±0.23 |86.43±0.11 86.43±0.11
    beta                 |84.52±0.11 84.52±0.11 |27.64±0.25 27.64±0.25 |85.15±0.10 85.15±0.10
    empirical            |81.92±1.15 81.92±1.15 |35.43±2.66 35.43±2.66 |83.07±0.94 83.07±0.94
    empirical_cumulative |85.50±0.16 85.50±0.16 |24.49±0.53 24.49±0.53 |85.97±0.14 85.97±0.14
    val_cal              |87.98±0.15 87.98±0.15 |12.24±0.71 12.24±0.71 |88.08±0.14 88.08±0.14

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 183±29 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.94±0.11 84.94±0.11 |26.26±0.59 26.26±0.59 |85.50±0.09 85.50±0.09
    fixed_05             |86.87±0.08 86.87±0.08 |18.77±0.35 18.77±0.35 |87.13±0.07 87.13±0.07
    platt                |85.88±0.12 85.88±0.12 |22.85±0.21 22.85±0.21 |86.28±0.11 86.28±0.11
    beta                 |84.60±0.23 84.60±0.23 |27.46±0.46 27.46±0.46 |85.22±0.21 85.22±0.21
    empirical            |82.17±1.07 82.17±1.07 |34.65±2.87 34.65±2.87 |83.25±0.86 83.25±0.86
    empirical_cumulative |85.22±0.21 85.22±0.21 |25.21±0.45 25.21±0.45 |85.72±0.19 85.72±0.19
    val_cal              |87.86±0.07 87.86±0.07 |12.22±0.57 12.22±0.57 |87.96±0.07 87.96±0.07

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 217±29 neurons | 43±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.62±0.14 83.62±0.14 |29.63±0.23 29.63±0.23 |84.35±0.13 84.35±0.13
    fixed_05             |85.44±1.02 85.44±1.02 |23.74±2.98 23.74±2.98 |85.89±0.89 85.89±0.89
    platt                |84.65±0.71 84.65±0.71 |26.33±1.96 26.33±1.96 |85.20±0.62 85.20±0.62
    beta                 |83.43±0.36 83.43±0.36 |30.30±0.93 30.30±0.93 |84.20±0.30 84.20±0.30
    empirical            |79.61±0.34 79.61±0.34 |40.90±0.78 40.90±0.78 |81.22±0.26 81.22±0.26
    empirical_cumulative |84.65±0.71 84.65±0.71 |26.37±1.97 26.37±1.97 |85.21±0.62 85.21±0.62
    val_cal              |88.17±0.18 88.17±0.18 |10.29±1.44 10.29±1.44 |88.24±0.19 88.24±0.19


## XDS-unsw-temporal-32b-Wa-500n34b  (4 flows × 2 phases, seeds: [11760, 54181, 74627, 88021])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.46% |   8.99% |  89.53% | r11760 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.46% |   8.99% |  89.53% | r11760 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.46% |   8.99% |  89.53% | r11760 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  88.92% |   5.71% |  88.94% | r54181 GA best_fpr       empirical
    Best F1 (FPR<5%)         |  88.68% |   4.61% |  88.69% | r54181 GA best_ce        empirical
    Best F1 (FPR<4%)         |  87.80% |   3.33% |  87.80% | r88021 GA best_ce        empirical
    Best FPR (any F1)        |  85.20% |   1.58% |  85.21% | r74627 GA best_ce        empirical
    Best FPR (F1>80%)        |  85.20% |   1.58% |  85.21% | r74627 GA best_ce        empirical
    Best Acc (any FPR)       |  89.46% |   8.99% |  89.53% | r11760 GA best_ce        val_cal

### best_fitness  (GS: 4 runs | GA: 4 runs)
    Grid Search : 325±150 neurons | 34±0 bits
    GA Neurons  : 283±108 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.48±0.26 83.87±0.42 |30.01±0.70 29.05±1.00 |84.23±0.22 84.57±0.37
    fixed_05             |85.10±0.30 84.70±0.09 |24.76±0.87 26.31±0.65 |85.58±0.26 85.25±0.06
    platt                |84.41±0.20 84.33±0.09 |27.08±0.45 27.56±0.20 |85.00±0.18 84.95±0.10
    beta                 |83.33±0.05 83.30±0.08 |30.47±0.23 30.86±0.28 |84.11±0.04 84.11±0.08
    empirical            |85.45±1.84 86.84±0.92 | 4.29±1.63  3.88±1.73 |85.46±1.82 86.85±0.92
    empirical_cumulative |84.33±0.34 84.36±0.64 |27.28±1.10 27.48±1.74 |84.93±0.29 84.97±0.55
    val_cal              |88.29±0.21 88.84±0.51 |10.16±0.70  8.94±0.88 |88.35±0.21 88.90±0.51

### best_f1  (GS: 4 runs | GA: 4 runs)
    Grid Search : 300±115 neurons | 34±1 bits
    GA Neurons  : 283±108 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.51±0.27 83.87±0.42 |29.93±0.71 29.05±1.00 |84.26±0.23 84.57±0.37
    fixed_05             |85.15±0.27 84.70±0.09 |24.63±0.79 26.31±0.65 |85.62±0.24 85.25±0.06
    platt                |84.41±0.20 84.33±0.09 |27.07±0.44 27.56±0.20 |85.00±0.18 84.95±0.10
    beta                 |83.38±0.06 83.30±0.08 |30.32±0.27 30.86±0.28 |84.15±0.05 84.11±0.08
    empirical            |86.21±0.74 86.84±0.92 | 4.95±0.47  3.88±1.73 |86.22±0.74 86.85±0.92
    empirical_cumulative |84.16±0.22 84.36±0.64 |27.89±0.61 27.48±1.74 |84.79±0.19 84.97±0.55
    val_cal              |88.25±0.17 88.84±0.51 |10.12±0.79  8.94±0.88 |88.31±0.17 88.90±0.51

### best_fpr  (GS: 4 runs | GA: 4 runs)
    Grid Search : 126±94 neurons | 33±1 bits
    GA Neurons  : 328±114 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.06±0.85 84.31±0.18 |28.37±2.38 27.72±0.64 |84.73±0.73 84.93±0.15
    fixed_05             |85.22±0.12 84.53±0.18 |24.64±0.48 27.00±0.57 |85.70±0.10 85.12±0.16
    platt                |84.68±0.42 84.30±0.13 |26.43±1.06 27.80±0.28 |85.24±0.37 84.93±0.12
    beta                 |82.37±1.62 83.27±0.13 |33.11±4.34 31.01±0.45 |83.34±1.28 84.08±0.11
    empirical            |85.27±3.07 87.09±1.38 |14.77±15.68  3.41±1.62 |85.62±2.46 87.10±1.39
    empirical_cumulative |84.95±0.32 85.06±0.37 |25.56±0.78 25.36±1.00 |85.47±0.29 85.57±0.32
    val_cal              |88.20±0.34 89.07±0.16 | 9.66±0.89  9.54±1.18 |88.26±0.34 89.14±0.15

### best_acc  (GS: 4 runs | GA: 4 runs)
    Grid Search : 333±115 neurons | 34±0 bits
    GA Neurons  : 283±108 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.41±0.11 83.87±0.42 |30.01±0.36 29.05±1.00 |84.16±0.09 84.57±0.37
    fixed_05             |84.72±0.17 84.70±0.09 |25.61±0.37 26.31±0.65 |85.24±0.15 85.25±0.06
    platt                |84.15±0.09 84.33±0.09 |27.55±0.19 27.56±0.20 |84.76±0.09 84.95±0.10
    beta                 |83.24±0.08 83.30±0.08 |30.56±0.26 30.86±0.28 |84.02±0.06 84.11±0.08
    empirical            |84.25±3.47 86.84±0.92 |14.21±18.26  3.88±1.73 |84.68±2.70 86.85±0.92
    empirical_cumulative |83.84±0.10 84.36±0.64 |28.60±0.40 27.48±1.74 |84.51±0.08 84.97±0.55
    val_cal              |88.15±0.14 88.84±0.51 | 9.88±0.32  8.94±0.88 |88.21±0.14 88.90±0.51

### best_ce  (GS: 4 runs | GA: 4 runs)
    Grid Search : 400±141 neurons | 34±1 bits
    GA Neurons  : 330±109 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.60±0.13 84.37±0.36 |29.55±0.51 27.59±1.02 |84.32±0.10 84.99±0.31
    fixed_05             |84.78±0.31 84.31±0.27 |25.73±0.87 27.72±0.75 |85.31±0.27 84.94±0.23
    platt                |84.22±0.15 84.20±0.13 |27.58±0.40 28.10±0.27 |84.83±0.13 84.85±0.11
    beta                 |83.22±0.08 83.26±0.07 |30.73±0.28 31.06±0.33 |84.02±0.06 84.08±0.05
    empirical            |84.91±2.12 86.71±1.79 | 3.71±1.55  2.77±1.48 |84.93±2.11 86.73±1.79
    empirical_cumulative |84.33±0.33 84.97±0.29 |27.21±1.06 25.73±0.94 |84.92±0.28 85.50±0.25
    val_cal              |88.47±0.13 89.37±0.07 | 9.40±0.33  8.47±0.99 |88.53±0.13 89.43±0.08


## XDS-unsw-temporal-32b-Wb-250n100b  (3 flows × 2 phases, seeds: [14675, 25694, 52015])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  88.39% |  10.36% |  88.46% | r25694 GS best_ce        val_cal
    Best F1 (FPR<14%)        |  88.39% |  10.36% |  88.46% | r25694 GS best_ce        val_cal
    Best F1 (FPR<10%)        |       — |       — |       — | —
    Best F1 (FPR<6%)         |       — |       — |       — | —
    Best F1 (FPR<5%)         |       — |       — |       — | —
    Best F1 (FPR<4%)         |       — |       — |       — | —
    Best FPR (any F1)        |  88.39% |  10.36% |  88.46% | r25694 GS best_ce        val_cal
    Best FPR (F1>80%)        |  88.39% |  10.36% |  88.46% | r25694 GS best_ce        val_cal
    Best Acc (any FPR)       |  88.39% |  10.36% |  88.46% | r25694 GS best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 75±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.84±0.34 84.84±0.34 |26.60±1.14 26.60±1.14 |85.41±0.29 85.41±0.29
    fixed_05             |86.94±0.34 86.94±0.34 |18.65±0.84 18.65±0.84 |87.20±0.32 87.20±0.32
    platt                |85.90±0.24 85.90±0.24 |22.84±0.54 22.84±0.54 |86.30±0.22 86.30±0.22
    beta                 |84.42±0.11 84.42±0.11 |27.91±0.07 27.91±0.07 |85.06±0.11 85.06±0.11
    empirical            |81.36±0.77 81.36±0.77 |36.89±1.78 36.89±1.78 |82.61±0.63 82.61±0.63
    empirical_cumulative |87.64±0.15 87.64±0.15 |14.83±0.89 14.83±0.89 |87.79±0.16 87.79±0.16
    val_cal              |87.80±0.15 87.80±0.15 |12.95±1.16 12.95±1.16 |87.92±0.17 87.92±0.17

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 75±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.84±0.34 84.84±0.34 |26.60±1.14 26.60±1.14 |85.41±0.29 85.41±0.29
    fixed_05             |86.94±0.34 86.94±0.34 |18.65±0.84 18.65±0.84 |87.20±0.32 87.20±0.32
    platt                |85.90±0.24 85.90±0.24 |22.84±0.54 22.84±0.54 |86.30±0.22 86.30±0.22
    beta                 |84.42±0.11 84.42±0.11 |27.91±0.07 27.91±0.07 |85.06±0.11 85.06±0.11
    empirical            |81.36±0.77 81.36±0.77 |36.89±1.78 36.89±1.78 |82.61±0.63 82.61±0.63
    empirical_cumulative |87.64±0.15 87.64±0.15 |14.83±0.89 14.83±0.89 |87.79±0.16 87.79±0.16
    val_cal              |87.80±0.15 87.80±0.15 |12.95±1.16 12.95±1.16 |87.92±0.17 87.92±0.17

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 4±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |70.76±0.13 70.76±0.13 |53.63±2.65 53.63±2.65 |73.80±0.48 73.80±0.48
    fixed_05             |72.84±1.74 72.84±1.74 |28.11±9.70 28.11±9.70 |73.16±1.30 73.16±1.30
    platt                |71.27±0.84 71.27±0.84 |46.93±2.57 46.93±2.57 |73.14±0.64 73.14±0.64
    beta                 |71.04±0.83 71.04±0.83 |49.19±1.96 49.19±1.96 |73.25±0.59 73.25±0.59
    empirical            |68.47±2.02 68.47±2.02 |60.10±3.46 60.10±3.46 |72.80±1.33 72.80±1.33
    empirical_cumulative |76.73±0.44 76.73±0.44 | 4.22±1.26  4.22±1.26 |76.92±0.45 76.92±0.45
    val_cal              |76.78±0.40 76.78±0.40 | 5.65±3.94  5.65±3.94 |76.93±0.50 76.93±0.50

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 75±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.84±0.34 84.84±0.34 |26.60±1.14 26.60±1.14 |85.41±0.29 85.41±0.29
    fixed_05             |86.94±0.34 86.94±0.34 |18.65±0.84 18.65±0.84 |87.20±0.32 87.20±0.32
    platt                |85.90±0.24 85.90±0.24 |22.84±0.54 22.84±0.54 |86.30±0.22 86.30±0.22
    beta                 |84.42±0.11 84.42±0.11 |27.91±0.07 27.91±0.07 |85.06±0.11 85.06±0.11
    empirical            |81.36±0.77 81.36±0.77 |36.89±1.78 36.89±1.78 |82.61±0.63 82.61±0.63
    empirical_cumulative |87.64±0.15 87.64±0.15 |14.83±0.89 14.83±0.89 |87.79±0.16 87.79±0.16
    val_cal              |87.80±0.15 87.80±0.15 |12.95±1.16 12.95±1.16 |87.92±0.17 87.92±0.17

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 217±29 neurons | 43±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.87±0.39 83.87±0.39 |28.90±1.27 28.90±1.27 |84.56±0.32 84.56±0.32
    fixed_05             |85.66±0.82 85.66±0.82 |23.17±2.54 23.17±2.54 |86.08±0.71 86.08±0.71
    platt                |84.72±0.58 84.72±0.58 |26.08±1.70 26.08±1.70 |85.27±0.50 85.27±0.50
    beta                 |83.46±0.33 83.46±0.33 |30.09±0.70 30.09±0.70 |84.22±0.29 84.22±0.29
    empirical            |79.88±0.37 79.88±0.37 |40.29±0.89 40.29±0.89 |81.43±0.28 81.43±0.28
    empirical_cumulative |87.87±0.25 87.87±0.25 |12.97±0.71 12.97±0.71 |87.99±0.24 87.99±0.24
    val_cal              |88.04±0.33 88.04±0.33 |11.29±1.49 11.29±1.49 |88.13±0.33 88.13±0.33


## XDS-unsw-temporal-32b-Wb-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.10% |   6.34% |  89.12% | r74627 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.10% |   6.34% |  89.12% | r74627 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.10% |   6.34% |  89.12% | r74627 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  88.62% |   5.27% |  88.64% | r74627 GA best_ce        empirical
    Best F1 (FPR<5%)         |  86.12% |   4.82% |  86.12% | r88021 GA best_fpr       empirical
    Best F1 (FPR<4%)         |  86.11% |   3.39% |  86.11% | r88021 GA best_ce        empirical
    Best FPR (any F1)        |  84.24% |   1.90% |  84.26% | r11760 GA best_ce        empirical
    Best FPR (F1>80%)        |  84.24% |   1.90% |  84.26% | r11760 GA best_ce        empirical
    Best Acc (any FPR)       |  89.10% |   6.34% |  89.12% | r74627 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±0 neurons | 33±1 bits
    GA Neurons  : 272±247 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.90±0.67 83.44±1.16 |28.41±2.32 29.93±3.18 |84.56±0.55 84.20±0.97
    fixed_05             |84.79±0.01 85.05±1.07 |25.45±0.29 24.68±2.44 |85.30±0.01 85.53±0.97
    platt                |84.32±0.31 84.37±0.48 |26.99±1.08 26.79±0.73 |84.90±0.26 84.94±0.45
    beta                 |82.05±2.02 84.02±0.75 |33.69±5.20 26.59±5.49 |83.07±1.62 84.60±0.49
    empirical            |85.83±2.73 84.57±2.88 |15.42±14.12 16.03±15.32 |86.12±2.27 84.92±2.57
    empirical_cumulative |86.94±0.73 86.22±2.15 |16.67±2.54 19.23±7.33 |87.14±0.67 86.53±1.92
    val_cal              |87.47±1.16 87.03±1.87 |12.63±5.51 13.28±6.09 |87.59±1.04 87.16±1.75

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±0 neurons | 33±1 bits
    GA Neurons  : 272±247 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.90±0.67 83.44±1.16 |28.41±2.32 29.93±3.18 |84.56±0.55 84.20±0.97
    fixed_05             |84.79±0.01 85.05±1.07 |25.45±0.29 24.68±2.44 |85.30±0.01 85.53±0.97
    platt                |84.32±0.31 84.37±0.48 |26.99±1.08 26.79±0.73 |84.90±0.26 84.94±0.45
    beta                 |82.05±2.02 84.02±0.75 |33.69±5.20 26.59±5.49 |83.07±1.62 84.60±0.49
    empirical            |85.83±2.73 84.57±2.88 |15.42±14.12 16.03±15.32 |86.12±2.27 84.92±2.57
    empirical_cumulative |86.94±0.73 86.22±2.15 |16.67±2.54 19.23±7.33 |87.14±0.67 86.53±1.92
    val_cal              |87.47±1.16 87.03±1.87 |12.63±5.51 13.28±6.09 |87.59±1.04 87.16±1.75

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 209±287 neurons | 25±10 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |77.71±7.51 80.03±7.46 |42.62±15.08 28.49±4.87 |79.81±5.70 80.53±7.58
    fixed_05             |79.64±5.67 79.29±4.68 |27.13±10.70 27.37±13.32 |80.27±6.03 80.01±5.00
    platt                |77.99±7.71 80.14±7.54 |38.22±10.59 28.13±4.73 |79.30±7.09 80.62±7.64
    beta                 |77.65±7.45 74.25±11.84 |42.71±14.90 47.75±20.74 |79.75±5.65 77.30±8.63
    empirical            |77.74±8.24 77.36±14.37 |36.00±27.90 32.86±34.24 |79.76±6.16 79.74±10.58
    empirical_cumulative |80.99±9.10 82.30±9.37 | 9.51±2.32  6.44±5.42 |81.17±8.88 82.62±8.91
    val_cal              |82.59±6.61 83.52±7.42 |11.01±1.58 10.09±1.92 |82.62±6.63 83.56±7.42

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±0 neurons | 24±0 bits
    GA Neurons  : 272±247 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.13±1.94 83.44±1.16 |29.90±4.91 29.93±3.18 |83.89±1.65 84.20±0.97
    fixed_05             |83.08±1.92 85.05±1.07 |30.00±5.01 24.68±2.44 |83.84±1.63 85.53±0.97
    platt                |82.98±1.83 84.37±0.48 |30.36±4.64 26.79±0.73 |83.76±1.56 84.94±0.45
    beta                 |81.09±1.45 84.02±0.75 |35.81±3.66 26.59±5.49 |82.25±1.17 84.60±0.49
    empirical            |82.66±4.78 84.57±2.88 |29.65±16.28 16.03±15.32 |83.65±3.81 84.92±2.57
    empirical_cumulative |85.28±2.14 86.22±2.15 |21.22±5.82 19.23±7.33 |85.63±1.92 86.53±1.92
    val_cal              |85.46±2.34 87.03±1.87 |19.78±7.67 13.28±6.09 |85.77±2.09 87.16±1.75

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 433±115 neurons | 33±1 bits
    GA Neurons  : 360±104 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.93±0.32 84.24±0.27 |28.52±1.06 28.10±0.56 |84.60±0.26 84.89±0.25
    fixed_05             |84.72±0.15 84.42±0.57 |25.86±0.41 27.57±1.53 |85.25±0.13 85.05±0.49
    platt                |84.17±0.10 84.21±0.36 |27.76±0.24 28.20±0.86 |84.79±0.09 84.86±0.31
    beta                 |83.23±0.06 83.29±0.27 |30.67±0.16 30.98±0.58 |84.02±0.05 84.10±0.24
    empirical            |82.97±4.14 86.32±2.20 |17.44±22.95  3.52±1.69 |83.62±3.04 86.33±2.20
    empirical_cumulative |88.00±0.15 88.42±0.21 |12.27±0.60 12.80±1.54 |88.10±0.14 88.54±0.20
    val_cal              |88.44±0.03 88.94±0.14 | 9.17±0.52  7.86±1.39 |88.50±0.03 88.98±0.12


## XDS-unsw-temporal-32b-Wc-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.58% |   9.46% |  89.65% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<14%)        |  89.58% |   9.46% |  89.65% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<10%)        |  89.58% |   9.46% |  89.65% | r11760 GA best_fpr       val_cal
    Best F1 (FPR<6%)         |  87.99% |   4.13% |  87.99% | r11760 GA best_ce        empirical
    Best F1 (FPR<5%)         |  87.99% |   4.13% |  87.99% | r11760 GA best_ce        empirical
    Best F1 (FPR<4%)         |  87.59% |   3.28% |  87.59% | r11760 GA best_fpr       empirical
    Best FPR (any F1)        |  84.92% |   1.88% |  84.93% | r74627 GA best_ce        empirical
    Best FPR (F1>80%)        |  84.92% |   1.88% |  84.93% | r74627 GA best_ce        empirical
    Best Acc (any FPR)       |  89.58% |   9.46% |  89.65% | r11760 GA best_fpr       val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 333±58 neurons | 33±1 bits
    GA Neurons  : 362±55 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.66±0.22 84.52±0.93 |29.35±0.57 27.40±2.65 |84.37±0.19 85.14±0.80
    fixed_05             |84.65±0.25 83.93±0.23 |26.15±0.52 29.17±0.54 |85.19±0.23 84.63±0.20
    platt                |84.05±0.18 84.24±0.22 |27.98±0.32 28.23±0.44 |84.68±0.17 84.90±0.20
    beta                 |83.14±0.06 83.47±0.21 |30.98±0.18 30.49±0.44 |83.95±0.05 84.26±0.19
    empirical            |83.34±4.08 86.66±1.58 |17.35±22.07  3.14±1.15 |83.94±3.07 86.67±1.57
    empirical_cumulative |86.62±1.21 86.60±0.35 |18.72±5.59 20.95±1.52 |86.90±1.06 86.94±0.30
    val_cal              |88.47±0.04 89.31±0.06 | 8.76±0.15  8.44±0.47 |88.52±0.04 89.37±0.06

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 450±71 neurons | 33±1 bits
    GA Neurons  : 386±89 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.63±0.32 84.03±0.19 |29.46±1.03 28.58±0.39 |84.34±0.26 84.70±0.17
    fixed_05             |85.05±0.14 85.06±0.25 |24.83±0.22 25.45±0.85 |85.53±0.13 85.57±0.22
    platt                |84.33±0.10 84.47±0.18 |27.15±0.17 27.17±0.51 |84.93±0.09 85.07±0.16
    beta                 |83.38±0.05 83.55±0.12 |30.26±0.09 30.08±0.16 |84.14±0.04 84.31±0.12
    empirical            |81.31±3.71 86.04±1.46 |29.78±21.18  3.28±0.72 |82.46±2.71 86.05±1.45
    empirical_cumulative |85.80±1.19 85.79±0.77 |21.92±4.60 23.11±2.32 |86.18±1.03 86.21±0.68
    val_cal              |88.31±0.13 88.76±0.29 |10.02±0.22  9.47±0.63 |88.37±0.13 88.83±0.29

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 100±0 neurons | 33±1 bits
    GA Neurons  : 362±61 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.79±0.23 84.60±0.12 |29.08±0.57 26.90±0.39 |84.49±0.20 85.19±0.10
    fixed_05             |85.27±0.03 84.04±0.30 |24.44±0.08 28.61±0.87 |85.74±0.03 84.72±0.25
    platt                |84.41±0.11 84.13±0.14 |27.10±0.19 28.32±0.36 |85.00±0.10 84.79±0.12
    beta                 |82.92±0.10 83.47±0.11 |31.69±0.31 30.35±0.34 |83.78±0.08 84.25±0.09
    empirical            |85.29±4.73 86.85±0.91 |22.26±15.80  3.30±0.69 |85.89±3.89 86.86±0.91
    empirical_cumulative |85.56±0.68 86.81±0.49 |23.50±2.12 20.25±1.37 |85.99±0.59 87.12±0.44
    val_cal              |88.35±0.04 89.23±0.31 |10.58±0.45  8.96±0.70 |88.42±0.03 89.29±0.32

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 33±1 bits
    GA Neurons  : 381±89 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.63±0.32 84.05±0.20 |29.46±1.03 28.49±0.44 |84.34±0.26 84.72±0.18
    fixed_05             |85.05±0.14 85.03±0.20 |24.83±0.22 25.55±0.72 |85.53±0.13 85.55±0.17
    platt                |84.33±0.10 84.45±0.14 |27.15±0.17 27.23±0.40 |84.93±0.09 85.05±0.13
    beta                 |83.38±0.05 83.54±0.13 |30.26±0.09 30.10±0.16 |84.14±0.04 84.30±0.12
    empirical            |81.31±3.71 86.06±1.47 |29.78±21.18  3.32±0.78 |82.46±2.71 86.07±1.46
    empirical_cumulative |85.80±1.19 85.81±0.75 |21.92±4.60 23.07±2.28 |86.18±1.03 86.23±0.67
    val_cal              |88.31±0.13 88.77±0.28 |10.02±0.22  9.44±0.69 |88.37±0.13 88.83±0.29

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±0 neurons | 32±0 bits
    GA Neurons  : 362±55 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.66±0.22 84.52±0.93 |29.35±0.57 27.40±2.65 |84.37±0.19 85.14±0.80
    fixed_05             |84.65±0.25 83.93±0.23 |26.15±0.52 29.17±0.54 |85.19±0.23 84.63±0.20
    platt                |84.05±0.18 84.24±0.22 |27.98±0.32 28.23±0.44 |84.68±0.17 84.90±0.20
    beta                 |83.14±0.06 83.47±0.21 |30.98±0.18 30.49±0.44 |83.95±0.05 84.26±0.19
    empirical            |83.34±4.08 86.66±1.58 |17.35±22.07  3.14±1.15 |83.94±3.07 86.67±1.57
    empirical_cumulative |86.62±1.21 86.60±0.35 |18.72±5.59 20.95±1.52 |86.90±1.06 86.94±0.30
    val_cal              |88.47±0.04 89.31±0.06 | 8.76±0.15  8.44±0.47 |88.52±0.04 89.37±0.06


## XDS-unsw-temporal-64b-Wa-250n100b  (3 flows × 2 phases, seeds: [14675, 25694, 52015])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  88.45% |   9.42% |  88.51% | r25694 GS best_ce        val_cal
    Best F1 (FPR<14%)        |  88.45% |   9.42% |  88.51% | r25694 GS best_ce        val_cal
    Best F1 (FPR<10%)        |  88.45% |   9.42% |  88.51% | r25694 GS best_ce        val_cal
    Best F1 (FPR<6%)         |       — |       — |       — | —
    Best F1 (FPR<5%)         |       — |       — |       — | —
    Best F1 (FPR<4%)         |       — |       — |       — | —
    Best FPR (any F1)        |  88.45% |   9.42% |  88.51% | r25694 GS best_ce        val_cal
    Best FPR (F1>80%)        |  88.45% |   9.42% |  88.51% | r25694 GS best_ce        val_cal
    Best Acc (any FPR)       |  88.45% |   9.42% |  88.51% | r25694 GS best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±87 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.96±0.22 84.96±0.22 |26.12±0.60 26.12±0.60 |85.51±0.19 85.51±0.19
    fixed_05             |86.94±0.13 86.94±0.13 |18.62±0.47 18.62±0.47 |87.20±0.12 87.20±0.12
    platt                |85.89±0.12 85.89±0.12 |22.89±0.25 22.89±0.25 |86.30±0.11 86.30±0.11
    beta                 |84.42±0.20 84.42±0.20 |27.91±0.46 27.91±0.46 |85.06±0.17 85.06±0.17
    empirical            |81.53±1.86 81.53±1.86 |36.09±4.87 36.09±4.87 |82.73±1.48 82.73±1.48
    empirical_cumulative |85.22±0.13 85.22±0.13 |25.23±0.48 25.23±0.48 |85.73±0.11 85.73±0.11
    val_cal              |87.80±0.05 87.80±0.05 |12.08±1.79 12.08±1.79 |87.89±0.08 87.89±0.08

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±87 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.96±0.22 84.96±0.22 |26.12±0.60 26.12±0.60 |85.51±0.19 85.51±0.19
    fixed_05             |86.94±0.13 86.94±0.13 |18.62±0.47 18.62±0.47 |87.20±0.12 87.20±0.12
    platt                |85.89±0.12 85.89±0.12 |22.89±0.25 22.89±0.25 |86.30±0.11 86.30±0.11
    beta                 |84.42±0.20 84.42±0.20 |27.91±0.46 27.91±0.46 |85.06±0.17 85.06±0.17
    empirical            |81.53±1.86 81.53±1.86 |36.09±4.87 36.09±4.87 |82.73±1.48 82.73±1.48
    empirical_cumulative |85.22±0.13 85.22±0.13 |25.23±0.48 25.23±0.48 |85.73±0.11 85.73±0.11
    val_cal              |87.80±0.05 87.80±0.05 |12.08±1.79 12.08±1.79 |87.89±0.08 87.89±0.08

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 50±0 neurons | 81±18 bits
    GA Neurons  : 167±76 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.85±0.22 85.23±0.61 |26.27±0.50 25.43±1.85 |85.40±0.20 85.75±0.53
    fixed_05             |86.84±0.36 87.14±0.13 |18.82±1.25 18.01±0.21 |87.10±0.33 87.38±0.12
    platt                |85.78±0.39 86.02±0.17 |23.05±0.93 22.61±0.30 |86.20±0.35 86.41±0.16
    beta                 |84.09±0.25 84.59±0.05 |28.87±0.62 27.51±0.11 |84.78±0.22 85.21±0.05
    empirical            |83.51±0.17 81.98±1.77 |30.70±0.22 35.25±4.20 |84.31±0.16 83.12±1.43
    empirical_cumulative |85.56±0.21 85.51±0.44 |23.79±1.14 24.44±1.36 |86.01±0.15 85.98±0.38
    val_cal              |87.67±0.07 87.77±0.09 |14.19±0.65 13.11±1.89 |87.81±0.05 87.88±0.13

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±87 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.96±0.22 84.96±0.22 |26.12±0.60 26.12±0.60 |85.51±0.19 85.51±0.19
    fixed_05             |86.94±0.13 86.94±0.13 |18.62±0.47 18.62±0.47 |87.20±0.12 87.20±0.12
    platt                |85.89±0.12 85.89±0.12 |22.89±0.25 22.89±0.25 |86.30±0.11 86.30±0.11
    beta                 |84.42±0.20 84.42±0.20 |27.91±0.46 27.91±0.46 |85.06±0.17 85.06±0.17
    empirical            |81.53±1.86 81.53±1.86 |36.09±4.87 36.09±4.87 |82.73±1.48 82.73±1.48
    empirical_cumulative |85.22±0.13 85.22±0.13 |25.23±0.48 25.23±0.48 |85.73±0.11 85.73±0.11
    val_cal              |87.80±0.05 87.80±0.05 |12.08±1.79 12.08±1.79 |87.89±0.08 87.89±0.08

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 233±29 neurons | 43±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.02±0.38 84.02±0.38 |28.56±1.28 28.56±1.28 |84.69±0.31 84.69±0.31
    fixed_05             |85.72±1.15 85.72±1.15 |22.93±3.45 22.93±3.45 |86.14±1.01 86.14±1.01
    platt                |84.88±0.78 84.88±0.78 |25.77±2.20 25.77±2.20 |85.41±0.68 85.41±0.68
    beta                 |83.53±0.55 83.53±0.55 |29.98±1.29 29.98±1.29 |84.28±0.48 84.28±0.48
    empirical            |79.60±0.10 79.60±0.10 |40.92±0.26 40.92±0.26 |81.21±0.08 81.21±0.08
    empirical_cumulative |84.89±0.49 84.89±0.49 |25.75±1.20 25.75±1.20 |85.41±0.44 85.41±0.44
    val_cal              |88.26±0.18 88.26±0.18 |11.37±1.70 11.37±1.70 |88.35±0.15 88.35±0.15


## XDS-unsw-temporal-64b-Wa-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.26% |   8.50% |  89.31% | r74627 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.26% |   8.50% |  89.31% | r74627 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.26% |   8.50% |  89.31% | r74627 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  86.98% |   5.78% |  86.98% | r74627 GA best_fpr       empirical
    Best F1 (FPR<5%)         |  85.70% |   4.76% |  85.70% | r74627 GS best_ce        empirical
    Best F1 (FPR<4%)         |  85.32% |   1.28% |  85.33% | r11760 GA best_ce        empirical
    Best FPR (any F1)        |  85.32% |   1.28% |  85.33% | r11760 GA best_ce        empirical
    Best FPR (F1>80%)        |  85.32% |   1.28% |  85.33% | r11760 GA best_ce        empirical
    Best Acc (any FPR)       |  89.26% |   8.50% |  89.31% | r74627 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 140±62 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.14±0.40 84.19±0.79 |28.03±1.19 28.67±1.56 |84.78±0.34 84.88±0.71
    fixed_05             |85.20±0.24 86.36±1.09 |24.37±0.55 21.34±2.49 |85.67±0.22 86.71±1.01
    platt                |84.55±0.18 85.42±0.91 |26.77±0.36 24.62±1.83 |85.13±0.16 85.90±0.84
    beta                 |83.40±0.12 84.11±0.76 |30.32±0.24 28.92±1.31 |84.17±0.10 84.81±0.70
    empirical            |83.35±3.35 87.61±1.20 |16.53±21.30  7.91±2.86 |83.90±2.42 87.65±1.23
    empirical_cumulative |84.84±0.60 84.71±1.12 |25.69±2.27 27.07±2.60 |85.37±0.50 85.31±1.00
    val_cal              |88.38±0.09 88.67±0.29 | 9.74±0.62 10.11±1.68 |88.44±0.09 88.75±0.32

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 367±58 neurons | 34±0 bits
    GA Neurons  : 140±62 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.14±0.40 84.19±0.79 |28.03±1.19 28.67±1.56 |84.78±0.34 84.88±0.71
    fixed_05             |85.20±0.24 86.36±1.09 |24.37±0.55 21.34±2.49 |85.67±0.22 86.71±1.01
    platt                |84.55±0.18 85.42±0.91 |26.77±0.36 24.62±1.83 |85.13±0.16 85.90±0.84
    beta                 |83.40±0.12 84.11±0.76 |30.32±0.24 28.92±1.31 |84.17±0.10 84.81±0.70
    empirical            |83.35±3.35 87.61±1.20 |16.53±21.30  7.91±2.86 |83.90±2.42 87.65±1.23
    empirical_cumulative |84.84±0.60 84.71±1.12 |25.69±2.27 27.07±2.60 |85.37±0.50 85.31±1.00
    val_cal              |88.38±0.09 88.67±0.29 | 9.74±0.62 10.11±1.68 |88.44±0.09 88.75±0.32

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±71 neurons | 34±0 bits
    GA Neurons  : 229±140 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |81.65±3.77 84.26±0.74 |21.43±13.08 28.03±1.96 |82.14±4.10 84.91±0.64
    fixed_05             |81.12±7.01 86.02±0.65 |22.49±3.69 22.52±1.67 |81.43±7.29 86.42±0.59
    platt                |78.02±10.98 84.96±0.52 |35.32±14.16 25.85±1.14 |79.07±10.18 85.50±0.47
    beta                 |73.55±16.48 83.52±0.18 |46.48±26.31 30.20±0.25 |76.72±12.41 84.29±0.17
    empirical            |79.96±12.66 86.67±2.18 |23.70±24.46  6.78±4.62 |80.66±11.56 86.71±2.22
    empirical_cumulative |82.25±4.29 85.47±0.21 |19.47±11.35 24.28±0.20 |82.64±4.54 85.93±0.21
    val_cal              |84.53±6.25 88.64±0.12 | 8.81±2.16 10.58±1.78 |84.61±6.23 88.72±0.15

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±212 neurons | 32±0 bits
    GA Neurons  : 140±62 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.87±0.04 84.19±0.79 |28.83±0.25 28.67±1.56 |84.55±0.03 84.88±0.71
    fixed_05             |85.14±0.47 86.36±1.09 |24.49±1.37 21.34±2.49 |85.61±0.41 86.71±1.01
    platt                |84.51±0.28 85.42±0.91 |26.82±0.74 24.62±1.83 |85.09±0.24 85.90±0.84
    beta                 |83.23±0.19 84.11±0.76 |30.84±0.53 28.92±1.31 |84.03±0.17 84.81±0.70
    empirical            |83.91±4.28 87.61±1.20 |17.75±20.33  7.91±2.86 |84.47±3.48 87.65±1.23
    empirical_cumulative |84.19±0.20 84.71±1.12 |27.85±0.64 27.07±2.60 |84.82±0.16 85.31±1.00
    val_cal              |88.35±0.08 88.67±0.29 | 9.58±0.56 10.11±1.68 |88.41±0.07 88.75±0.32

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 433±58 neurons | 33±1 bits
    GA Neurons  : 449±48 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.11±0.55 83.87±0.56 |28.05±1.61 29.28±1.67 |84.75±0.46 84.59±0.47
    fixed_05             |85.03±0.43 84.38±0.65 |24.92±1.19 27.64±1.91 |85.52±0.38 85.01±0.55
    platt                |84.39±0.29 84.19±0.39 |27.15±0.70 28.22±1.05 |84.98±0.26 84.85±0.33
    beta                 |83.35±0.15 83.39±0.22 |30.45±0.36 30.71±0.69 |84.12±0.13 84.19±0.18
    empirical            |84.98±1.18 85.03±0.27 | 4.07±0.84  1.91±0.72 |84.98±1.17 85.05±0.27
    empirical_cumulative |84.64±0.03 84.81±0.46 |26.25±0.36 26.28±1.17 |85.19±0.01 85.36±0.40
    val_cal              |88.42±0.05 89.14±0.11 | 9.85±0.35  9.42±0.84 |88.48±0.05 89.21±0.10


## XDS-unsw-temporal-64b-Wb-250n100b  (3 flows × 2 phases, seeds: [14675, 25694, 52015])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  88.59% |   9.74% |  88.65% | r14675 GS best_fpr       val_cal
    Best F1 (FPR<14%)        |  88.59% |   9.74% |  88.65% | r14675 GS best_fpr       val_cal
    Best F1 (FPR<10%)        |  88.59% |   9.74% |  88.65% | r14675 GS best_fpr       val_cal
    Best F1 (FPR<6%)         |       — |       — |       — | —
    Best F1 (FPR<5%)         |       — |       — |       — | —
    Best F1 (FPR<4%)         |       — |       — |       — | —
    Best FPR (any F1)        |  88.33% |   8.44% |  88.38% | r25694 GS best_ce        val_cal
    Best FPR (F1>80%)        |  88.33% |   8.44% |  88.38% | r25694 GS best_ce        val_cal
    Best Acc (any FPR)       |  88.59% |   9.74% |  88.65% | r14675 GS best_fpr       val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 100±50 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.89±0.44 84.89±0.44 |26.40±1.17 26.40±1.17 |85.45±0.39 85.45±0.39
    fixed_05             |86.95±0.10 86.95±0.10 |18.65±0.32 18.65±0.32 |87.21±0.09 87.21±0.09
    platt                |85.91±0.25 85.91±0.25 |22.81±0.39 22.81±0.39 |86.32±0.23 86.32±0.23
    beta                 |84.20±0.07 84.20±0.07 |28.58±0.44 28.58±0.44 |84.88±0.05 84.88±0.05
    empirical            |83.36±0.83 83.36±0.83 |31.52±2.60 31.52±2.60 |84.23±0.67 84.23±0.67
    empirical_cumulative |87.50±0.09 87.50±0.09 |15.48±0.71 15.48±0.71 |87.68±0.10 87.68±0.10
    val_cal              |87.69±0.17 87.69±0.17 |12.50±1.03 12.50±1.03 |87.79±0.17 87.79±0.17

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 100±50 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.89±0.44 84.89±0.44 |26.40±1.17 26.40±1.17 |85.45±0.39 85.45±0.39
    fixed_05             |86.95±0.10 86.95±0.10 |18.65±0.32 18.65±0.32 |87.21±0.09 87.21±0.09
    platt                |85.91±0.25 85.91±0.25 |22.81±0.39 22.81±0.39 |86.32±0.23 86.32±0.23
    beta                 |84.20±0.07 84.20±0.07 |28.58±0.44 28.58±0.44 |84.88±0.05 84.88±0.05
    empirical            |83.36±0.83 83.36±0.83 |31.52±2.60 31.52±2.60 |84.23±0.67 84.23±0.67
    empirical_cumulative |87.50±0.09 87.50±0.09 |15.48±0.71 15.48±0.71 |87.68±0.10 87.68±0.10
    val_cal              |87.69±0.17 87.69±0.17 |12.50±1.03 12.50±1.03 |87.79±0.17 87.79±0.17

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 102±84 neurons | 13±16 bits
    GA Neurons  : 20±0 neurons | 8±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |73.26±10.09 69.45±4.86 |46.93±16.14 55.39±2.61 |75.61±8.78 72.65±4.88
    fixed_05             |73.65±10.53 69.63±5.05 |33.94±10.76 39.12±8.55 |74.23±10.34 70.46±4.96
    platt                |73.14±10.23 69.44±4.90 |44.48±15.44 50.19±7.18 |75.05±8.97 71.71±4.12
    beta                 |70.34±13.52 66.55±8.97 |52.37±23.10 59.69±14.98 |74.17±9.68 71.11±5.65
    empirical            |67.69±11.41 64.98±7.75 |60.14±17.33 64.93±10.59 |72.60±8.08 70.53±5.24
    empirical_cumulative |79.91±7.06 75.78±1.26 | 5.77±6.51  9.65±13.13 |80.19±6.90 76.07±1.13
    val_cal              |80.25±7.31 76.89±1.81 | 8.15±4.58  5.72±5.21 |80.36±7.29 77.03±1.88

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 100±50 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.89±0.44 84.89±0.44 |26.40±1.17 26.40±1.17 |85.45±0.39 85.45±0.39
    fixed_05             |86.95±0.10 86.95±0.10 |18.65±0.32 18.65±0.32 |87.21±0.09 87.21±0.09
    platt                |85.91±0.25 85.91±0.25 |22.81±0.39 22.81±0.39 |86.32±0.23 86.32±0.23
    beta                 |84.20±0.07 84.20±0.07 |28.58±0.44 28.58±0.44 |84.88±0.05 84.88±0.05
    empirical            |83.36±0.83 83.36±0.83 |31.52±2.60 31.52±2.60 |84.23±0.67 84.23±0.67
    empirical_cumulative |87.50±0.09 87.50±0.09 |15.48±0.71 15.48±0.71 |87.68±0.10 87.68±0.10
    val_cal              |87.69±0.17 87.69±0.17 |12.50±1.03 12.50±1.03 |87.79±0.17 87.79±0.17

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±0 neurons | 43±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.24±0.10 84.24±0.10 |27.73±0.27 27.73±0.27 |84.86±0.11 84.86±0.11
    fixed_05             |85.75±1.00 85.75±1.00 |22.72±2.82 22.72±2.82 |86.15±0.89 86.15±0.89
    platt                |84.80±0.65 84.80±0.65 |25.83±1.80 25.83±1.80 |85.33±0.57 85.33±0.57
    beta                 |83.53±0.43 83.53±0.43 |29.98±0.96 29.98±0.96 |84.28±0.37 84.28±0.37
    empirical            |79.56±0.25 79.56±0.25 |40.95±0.56 40.95±0.56 |81.17±0.19 81.17±0.19
    empirical_cumulative |88.08±0.17 88.08±0.17 |13.32±0.46 13.32±0.46 |88.21±0.17 88.21±0.17
    val_cal              |88.33±0.11 88.33±0.11 |10.43±1.80 10.43±1.80 |88.41±0.13 88.41±0.13


## XDS-unsw-temporal-64b-Wb-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  88.99% |   8.83% |  89.05% | r74627 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  88.99% |   8.83% |  89.05% | r74627 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  88.99% |   8.83% |  89.05% | r74627 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  87.31% |   3.30% |  87.31% | r74627 GA best_ce        empirical
    Best F1 (FPR<5%)         |  87.31% |   3.30% |  87.31% | r74627 GA best_ce        empirical
    Best F1 (FPR<4%)         |  87.31% |   3.30% |  87.31% | r74627 GA best_ce        empirical
    Best FPR (any F1)        |  83.96% |   2.57% |  83.98% | r88021 GA best_ce        empirical
    Best FPR (F1>80%)        |  83.96% |   2.57% |  83.98% | r88021 GA best_ce        empirical
    Best Acc (any FPR)       |  88.99% |   8.83% |  89.05% | r74627 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±71 neurons | 29±7 bits
    GA Neurons  : 304±270 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.92±1.10 84.35±0.45 |31.04±2.57 27.40±1.18 |83.73±0.94 84.96±0.39
    fixed_05             |84.00±0.94 85.04±1.05 |27.20±1.96 25.12±2.74 |84.59±0.87 85.55±0.93
    platt                |83.63±0.52 84.41±0.49 |28.55±1.21 27.13±1.22 |84.29±0.49 85.00±0.44
    beta                 |81.29±2.42 81.21±3.32 |35.77±5.79 35.83±7.98 |82.47±1.93 82.44±2.64
    empirical            |85.61±3.46 83.97±3.64 |20.02±13.07 17.80±18.34 |86.05±2.89 84.46±3.04
    empirical_cumulative |85.93±2.34 86.32±1.99 |20.38±6.26 19.55±7.51 |86.26±2.12 86.64±1.73
    val_cal              |86.42±2.56 86.78±2.27 |13.06±5.68 11.53±2.69 |86.53±2.47 86.85±2.26

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±71 neurons | 29±7 bits
    GA Neurons  : 304±270 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.92±1.10 84.35±0.45 |31.04±2.57 27.40±1.18 |83.73±0.94 84.96±0.39
    fixed_05             |84.00±0.94 85.04±1.05 |27.20±1.96 25.12±2.74 |84.59±0.87 85.55±0.93
    platt                |83.63±0.52 84.41±0.49 |28.55±1.21 27.13±1.22 |84.29±0.49 85.00±0.44
    beta                 |81.29±2.42 81.21±3.32 |35.77±5.79 35.83±7.98 |82.47±1.93 82.44±2.64
    empirical            |85.61±3.46 83.97±3.64 |20.02±13.07 17.80±18.34 |86.05±2.89 84.46±3.04
    empirical_cumulative |85.93±2.34 86.32±1.99 |20.38±6.26 19.55±7.51 |86.26±2.12 86.64±1.73
    val_cal              |86.42±2.56 86.78±2.27 |13.06±5.68 11.53±2.69 |86.53±2.47 86.85±2.26

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 500±0 neurons | 32±0 bits
    GA Neurons  : 162±267 neurons | 22±16 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |81.71±3.81 77.59±8.45 |21.25±12.91 32.26±2.88 |82.20±4.14 78.25±8.86
    fixed_05             |81.04±6.94 77.93±8.40 |22.84±3.90 33.23±8.68 |81.37±7.22 78.69±8.16
    platt                |78.01±10.97 77.75±8.16 |35.46±14.03 34.12±7.71 |79.08±10.18 78.56±7.97
    beta                 |73.67±16.58 70.69±16.85 |46.09±26.65 52.59±26.45 |76.82±12.50 74.95±11.87
    empirical            |77.73±10.78 72.47±18.29 |19.98±27.45 41.15±39.05 |78.40±9.66 76.22±12.85
    empirical_cumulative |84.47±6.21 78.62±12.15 |10.50±3.59  8.52±6.78 |84.58±6.21 79.27±11.19
    val_cal              |84.72±6.41 80.41±9.84 | 8.50±2.09  6.66±2.78 |84.80±6.39 80.69±9.43

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±71 neurons | 29±7 bits
    GA Neurons  : 304±270 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.92±1.10 84.25±0.62 |31.04±2.57 27.75±1.73 |83.73±0.94 84.88±0.53
    fixed_05             |84.00±0.94 84.87±1.30 |27.20±1.96 24.79±2.32 |84.59±0.87 85.35±1.22
    platt                |83.63±0.52 84.35±0.57 |28.55±1.21 27.31±1.45 |84.29±0.49 84.96±0.50
    beta                 |81.29±2.42 81.11±3.49 |35.77±5.79 36.02±8.31 |82.47±1.93 82.36±2.77
    empirical            |85.61±3.46 83.89±3.75 |20.02±13.07 17.97±18.64 |86.05±2.89 84.40±3.12
    empirical_cumulative |85.93±2.34 86.15±2.27 |20.38±6.26 19.10±6.78 |86.26±2.12 86.45±2.07
    val_cal              |86.42±2.56 86.87±2.11 |13.06±5.68 12.27±3.96 |86.53±2.47 86.96±2.06

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 359±116 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.89±0.19 84.29±0.28 |28.66±0.50 27.85±0.87 |84.56±0.16 84.92±0.24
    fixed_05             |84.72±0.26 84.54±0.26 |25.91±0.84 27.02±0.72 |85.26±0.22 85.13±0.22
    platt                |84.18±0.18 84.24±0.17 |27.73±0.50 28.00±0.34 |84.80±0.15 84.88±0.15
    beta                 |83.21±0.11 83.30±0.18 |30.77±0.25 30.88±0.44 |84.01±0.10 84.11±0.16
    empirical            |81.44±4.24 85.89±1.73 |29.85±21.52  2.99±0.38 |82.60±3.23 85.90±1.72
    empirical_cumulative |86.65±2.10 88.69±0.02 |18.40±8.11 11.92±0.41 |86.95±1.82 88.79±0.02
    val_cal              |88.54±0.05 88.90±0.09 | 9.85±0.28 10.10±1.20 |88.60±0.05 88.97±0.07


## XDS-unsw-temporal-64b-Wc-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.64% |   8.98% |  89.71% | r88021 GA best_fpr       val_cal
    Best F1 (FPR<14%)        |  89.64% |   8.98% |  89.71% | r88021 GA best_fpr       val_cal
    Best F1 (FPR<10%)        |  89.64% |   8.98% |  89.71% | r88021 GA best_fpr       val_cal
    Best F1 (FPR<6%)         |  88.79% |   5.95% |  88.81% | r11760 GA best_ce        empirical
    Best F1 (FPR<5%)         |  87.65% |   2.55% |  87.65% | r88021 GA best_ce        empirical
    Best F1 (FPR<4%)         |  87.65% |   2.55% |  87.65% | r88021 GA best_ce        empirical
    Best FPR (any F1)        |  87.65% |   2.55% |  87.65% | r88021 GA best_ce        empirical
    Best FPR (F1>80%)        |  87.65% |   2.55% |  87.65% | r88021 GA best_ce        empirical
    Best Acc (any FPR)       |  89.64% |   8.98% |  89.71% | r88021 GA best_fpr       val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 433±58 neurons | 33±1 bits
    GA Neurons  : 309±50 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.91±0.22 84.40±0.70 |28.59±0.67 27.58±1.74 |84.58±0.19 85.02±0.62
    fixed_05             |84.94±0.15 84.12±0.83 |25.19±0.48 28.43±2.16 |85.44±0.13 84.78±0.71
    platt                |84.29±0.07 84.21±0.45 |27.38±0.23 28.17±0.89 |84.89±0.06 84.87±0.41
    beta                 |83.24±0.08 83.41±0.28 |30.62±0.16 30.59±0.32 |84.03±0.07 84.20±0.26
    empirical            |83.17±3.61 87.72±1.04 |16.97±21.86  3.76±1.90 |83.75±2.61 87.72±1.05
    empirical_cumulative |86.39±1.35 87.02±0.29 |19.66±5.78 19.33±0.77 |86.70±1.19 87.30±0.27
    val_cal              |88.39±0.11 89.36±0.08 | 9.81±0.33  8.25±0.81 |88.45±0.11 89.41±0.09

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±71 neurons | 34±0 bits
    GA Neurons  : 354±62 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.69±0.56 84.71±0.66 |29.36±1.83 26.60±1.36 |84.41±0.46 85.28±0.60
    fixed_05             |85.20±0.08 84.95±1.14 |24.41±0.19 25.73±3.18 |85.66±0.07 85.48±0.99
    platt                |84.49±0.14 84.58±0.71 |26.83±0.30 26.99±1.61 |85.07±0.13 85.17±0.63
    beta                 |83.35±0.09 83.56±0.37 |30.38±0.29 30.07±0.71 |84.12±0.08 84.32±0.33
    empirical            |84.21±4.54 86.70±2.21 |17.85±20.90  4.49±1.00 |84.79±3.60 86.71±2.21
    empirical_cumulative |85.22±0.58 85.78±0.88 |24.35±2.13 23.18±2.18 |85.68±0.48 86.20±0.80
    val_cal              |88.29±0.17 88.90±0.38 |10.01±1.03  9.70±1.26 |88.35±0.17 88.97±0.38

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 167±58 neurons | 33±1 bits
    GA Neurons  : 318±57 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.71±0.21 84.47±1.38 |29.23±0.67 27.46±3.50 |84.42±0.17 85.09±1.22
    fixed_05             |85.29±0.18 84.77±1.03 |24.28±0.53 26.65±2.47 |85.75±0.16 85.35±0.92
    platt                |84.48±0.11 84.53±0.61 |26.83±0.26 27.40±1.19 |85.06±0.10 85.14±0.55
    beta                 |83.17±0.09 83.57±0.42 |30.99±0.22 30.30±0.50 |83.99±0.08 84.34±0.39
    empirical            |87.01±0.62 88.00±1.35 | 8.67±4.18  4.94±2.03 |87.06±0.69 88.01±1.36
    empirical_cumulative |85.72±0.45 87.16±0.50 |22.93±1.41 19.11±0.74 |86.13±0.39 87.44±0.48
    val_cal              |88.34±0.09 89.39±0.22 | 9.67±0.55  8.89±0.25 |88.40±0.09 89.45±0.22

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±71 neurons | 34±0 bits
    GA Neurons  : 354±62 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.69±0.56 84.71±0.66 |29.36±1.83 26.60±1.36 |84.41±0.46 85.28±0.60
    fixed_05             |85.20±0.08 84.95±1.14 |24.41±0.19 25.73±3.18 |85.66±0.07 85.48±0.99
    platt                |84.49±0.14 84.58±0.71 |26.83±0.30 26.99±1.61 |85.07±0.13 85.17±0.63
    beta                 |83.35±0.09 83.56±0.37 |30.38±0.29 30.07±0.71 |84.12±0.08 84.32±0.33
    empirical            |84.21±4.54 86.70±2.21 |17.85±20.90  4.49±1.00 |84.79±3.60 86.71±2.21
    empirical_cumulative |85.22±0.58 85.78±0.88 |24.35±2.13 23.18±2.18 |85.68±0.48 86.20±0.80
    val_cal              |88.29±0.17 88.90±0.38 |10.01±1.03  9.70±1.26 |88.35±0.17 88.97±0.38

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 309±50 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.91±0.22 84.40±0.70 |28.59±0.67 27.58±1.74 |84.58±0.19 85.02±0.62
    fixed_05             |84.94±0.15 84.12±0.83 |25.19±0.48 28.43±2.16 |85.44±0.13 84.78±0.71
    platt                |84.29±0.07 84.21±0.45 |27.38±0.23 28.17±0.89 |84.89±0.06 84.87±0.41
    beta                 |83.24±0.08 83.41±0.28 |30.62±0.16 30.59±0.32 |84.03±0.07 84.20±0.26
    empirical            |83.17±3.61 87.72±1.04 |16.97±21.86  3.76±1.90 |83.75±2.61 87.72±1.05
    empirical_cumulative |86.39±1.35 87.02±0.29 |19.66±5.78 19.33±0.77 |86.70±1.19 87.30±0.27
    val_cal              |88.39±0.11 89.36±0.08 | 9.81±0.33  8.25±0.81 |88.45±0.11 89.41±0.09


## XDS-unsw-temporal-96b-Wa-250n100b  (3 flows × 2 phases, seeds: [14675, 25694, 52015])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  88.38% |  10.06% |  88.44% | r14675 GS best_ce        val_cal
    Best F1 (FPR<14%)        |  88.38% |  10.06% |  88.44% | r14675 GS best_ce        val_cal
    Best F1 (FPR<10%)        |       — |       — |       — | —
    Best F1 (FPR<6%)         |       — |       — |       — | —
    Best F1 (FPR<5%)         |       — |       — |       — | —
    Best F1 (FPR<4%)         |       — |       — |       — | —
    Best FPR (any F1)        |  88.38% |  10.06% |  88.44% | r14675 GS best_ce        val_cal
    Best FPR (F1>80%)        |  88.38% |  10.06% |  88.44% | r14675 GS best_ce        val_cal
    Best Acc (any FPR)       |  88.38% |  10.06% |  88.44% | r14675 GS best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.81±0.28 84.81±0.28 |26.69±0.72 26.69±0.72 |85.38±0.25 85.38±0.25
    fixed_05             |87.00±0.11 87.00±0.11 |18.48±0.18 18.48±0.18 |87.26±0.11 87.26±0.11
    platt                |85.82±0.11 85.82±0.11 |23.03±0.10 23.03±0.10 |86.24±0.11 86.24±0.11
    beta                 |84.44±0.18 84.44±0.18 |27.89±0.35 27.89±0.35 |85.08±0.16 85.08±0.16
    empirical            |81.77±1.38 81.77±1.38 |35.71±3.58 35.71±3.58 |82.94±1.11 82.94±1.11
    empirical_cumulative |85.31±0.17 85.31±0.17 |24.94±0.39 24.94±0.39 |85.81±0.15 85.81±0.15
    val_cal              |87.82±0.17 87.82±0.17 |13.29±0.48 13.29±0.48 |87.95±0.18 87.95±0.18

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 64±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.81±0.28 84.81±0.28 |26.69±0.72 26.69±0.72 |85.38±0.25 85.38±0.25
    fixed_05             |87.00±0.11 87.00±0.11 |18.48±0.18 18.48±0.18 |87.26±0.11 87.26±0.11
    platt                |85.82±0.11 85.82±0.11 |23.03±0.10 23.03±0.10 |86.24±0.11 86.24±0.11
    beta                 |84.44±0.18 84.44±0.18 |27.89±0.35 27.89±0.35 |85.08±0.16 85.08±0.16
    empirical            |81.77±1.38 81.77±1.38 |35.71±3.58 35.71±3.58 |82.94±1.11 82.94±1.11
    empirical_cumulative |85.31±0.17 85.31±0.17 |24.94±0.39 24.94±0.39 |85.81±0.15 85.81±0.15
    val_cal              |87.82±0.17 87.82±0.17 |13.29±0.48 13.29±0.48 |87.95±0.18 87.95±0.18

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±0 neurons | 75±18 bits
    GA Neurons  : 183±58 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.98±0.18 84.96±0.24 |26.14±0.58 26.29±0.53 |85.53±0.15 85.52±0.22
    fixed_05             |87.02±0.20 87.13±0.19 |18.30±0.39 18.09±0.46 |87.27±0.19 87.38±0.18
    platt                |85.82±0.10 85.97±0.20 |22.97±0.25 22.74±0.47 |86.23±0.09 86.37±0.19
    beta                 |84.49±0.13 84.47±0.17 |27.71±0.29 27.80±0.32 |85.12±0.12 85.11±0.16
    empirical            |80.95±0.92 80.98±0.27 |37.76±2.12 37.77±0.62 |82.28±0.74 82.30±0.22
    empirical_cumulative |85.51±0.22 85.41±0.34 |24.20±0.59 24.82±0.85 |85.98±0.20 85.90±0.30
    val_cal              |87.77±0.23 87.93±0.14 |12.76±0.51 12.99±1.32 |87.88±0.23 88.04±0.13

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 69±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.65±0.21 84.65±0.21 |27.06±0.44 27.06±0.44 |85.24±0.19 85.24±0.19
    fixed_05             |86.85±0.22 86.85±0.22 |18.79±0.42 18.79±0.42 |87.11±0.21 87.11±0.21
    platt                |85.69±0.19 85.69±0.19 |23.19±0.30 23.19±0.30 |86.11±0.18 86.11±0.18
    beta                 |84.38±0.23 84.38±0.23 |27.99±0.43 27.99±0.43 |85.02±0.21 85.02±0.21
    empirical            |82.38±1.04 82.38±1.04 |34.25±2.74 34.25±2.74 |83.43±0.84 83.43±0.84
    empirical_cumulative |85.28±0.16 85.28±0.16 |24.81±0.45 24.81±0.45 |85.77±0.15 85.77±0.15
    val_cal              |87.66±0.13 87.66±0.13 |13.29±0.47 13.29±0.47 |87.78±0.13 87.78±0.13

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±0 neurons | 43±9 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.00±0.27 84.00±0.27 |28.54±0.87 28.54±0.87 |84.67±0.23 84.67±0.23
    fixed_05             |85.76±0.83 85.76±0.83 |22.72±2.62 22.72±2.62 |86.16±0.72 86.16±0.72
    platt                |84.84±0.59 84.84±0.59 |25.84±1.69 25.84±1.69 |85.37±0.51 85.37±0.51
    beta                 |83.55±0.32 83.55±0.32 |29.93±0.88 29.93±0.88 |84.30±0.27 84.30±0.27
    empirical            |79.23±0.55 79.23±0.55 |41.69±1.27 41.69±1.27 |80.93±0.43 80.93±0.43
    empirical_cumulative |84.76±0.33 84.76±0.33 |26.15±1.15 26.15±1.15 |85.31±0.27 85.31±0.27
    val_cal              |88.30±0.11 88.30±0.11 |10.52±0.52 10.52±0.52 |88.37±0.10 88.37±0.10


## XDS-unsw-temporal-96b-Wa-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  88.90% |   9.26% |  88.96% | r74627 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  88.90% |   9.26% |  88.96% | r74627 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  88.90% |   9.26% |  88.96% | r74627 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  86.93% |   4.03% |  86.93% | r11760 GA best_ce        empirical
    Best F1 (FPR<5%)         |  86.93% |   4.03% |  86.93% | r11760 GA best_ce        empirical
    Best F1 (FPR<4%)         |  86.11% |   3.27% |  86.11% | r74627 GA best_ce        empirical
    Best FPR (any F1)        |  85.48% |   2.27% |  85.49% | r88021 GA best_ce        empirical
    Best FPR (F1>80%)        |  85.48% |   2.27% |  85.49% | r88021 GA best_ce        empirical
    Best Acc (any FPR)       |  88.90% |   9.26% |  88.96% | r74627 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 235±174 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.73±0.25 84.13±0.43 |29.20±0.66 28.43±0.83 |84.44±0.21 84.79±0.39
    fixed_05             |85.52±0.26 86.18±0.84 |23.46±0.63 21.85±2.32 |85.95±0.23 86.55±0.75
    platt                |84.66±0.08 85.21±0.55 |26.37±0.15 25.13±1.20 |85.21±0.08 85.71±0.50
    beta                 |83.29±0.09 83.65±0.25 |30.49±0.31 29.93±0.58 |84.07±0.07 84.40±0.23
    empirical            |81.16±3.55 86.62±1.73 |29.36±21.67  6.47±3.47 |82.30±2.57 86.64±1.75
    empirical_cumulative |85.65±0.71 84.97±0.49 |23.07±2.31 25.82±1.12 |86.07±0.62 85.50±0.45
    val_cal              |88.42±0.06 88.38±0.06 |10.15±0.95 10.20±0.93 |88.49±0.05 88.45±0.06

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±141 neurons | 34±0 bits
    GA Neurons  : 235±174 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.73±0.25 84.13±0.43 |29.20±0.66 28.43±0.83 |84.44±0.21 84.79±0.39
    fixed_05             |85.52±0.26 86.18±0.84 |23.46±0.63 21.85±2.32 |85.95±0.23 86.55±0.75
    platt                |84.66±0.08 85.21±0.55 |26.37±0.15 25.13±1.20 |85.21±0.08 85.71±0.50
    beta                 |83.29±0.09 83.65±0.25 |30.49±0.31 29.93±0.58 |84.07±0.07 84.40±0.23
    empirical            |81.16±3.55 86.62±1.73 |29.36±21.67  6.47±3.47 |82.30±2.57 86.64±1.75
    empirical_cumulative |85.65±0.71 84.97±0.49 |23.07±2.31 25.82±1.12 |86.07±0.62 85.50±0.45
    val_cal              |88.42±0.06 88.38±0.06 |10.15±0.95 10.20±0.93 |88.49±0.05 88.45±0.06

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±283 neurons | 32±0 bits
    GA Neurons  : 358±162 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.78±0.72 83.99±0.41 |29.11±2.34 28.47±1.26 |84.48±0.58 84.66±0.34
    fixed_05             |85.53±0.68 85.34±0.41 |23.33±2.20 24.34±1.43 |85.96±0.59 85.81±0.35
    platt                |84.92±1.03 84.56±0.20 |25.41±3.30 26.70±0.69 |85.44±0.89 85.13±0.16
    beta                 |81.08±3.63 83.47±0.16 |35.81±8.37 30.03±0.38 |82.31±2.90 84.22±0.14
    empirical            |83.95±1.30 85.40±1.55 |19.85±14.51  3.85±1.92 |84.39±1.29 85.41±1.54
    empirical_cumulative |85.32±0.81 85.26±0.12 |24.17±2.50 24.68±0.11 |85.78±0.70 85.74±0.12
    val_cal              |88.42±0.19 88.57±0.07 |10.95±1.36 10.21±0.78 |88.50±0.18 88.64±0.06

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±141 neurons | 34±0 bits
    GA Neurons  : 235±173 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.95±0.14 84.13±0.45 |28.53±0.54 28.38±0.91 |84.62±0.11 84.80±0.40
    fixed_05             |85.47±0.33 86.18±0.84 |23.64±0.91 21.83±2.34 |85.90±0.30 86.55±0.76
    platt                |84.58±0.21 85.22±0.56 |26.58±0.44 25.09±1.25 |85.14±0.19 85.72±0.51
    beta                 |83.26±0.04 83.64±0.24 |30.59±0.19 29.96±0.56 |84.05±0.03 84.39±0.21
    empirical            |80.95±3.18 86.62±1.74 |29.36±21.67  6.43±3.40 |82.09±2.20 86.65±1.76
    empirical_cumulative |85.24±0.67 84.94±0.44 |24.44±1.96 25.92±0.97 |85.71±0.59 85.48±0.40
    val_cal              |88.42±0.05 88.38±0.06 |10.56±0.38 10.20±0.93 |88.49±0.05 88.45±0.06

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±141 neurons | 33±1 bits
    GA Neurons  : 378±100 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.85±0.35 84.14±0.33 |28.72±1.09 28.16±0.76 |84.53±0.29 84.79±0.29
    fixed_05             |84.85±0.36 84.12±0.54 |25.44±1.17 28.17±1.64 |85.36±0.31 84.77±0.45
    platt                |84.27±0.29 84.08±0.21 |27.43±0.87 28.28±0.65 |84.87±0.24 84.73±0.17
    beta                 |83.21±0.16 83.25±0.13 |30.71±0.47 30.78±0.30 |84.00±0.13 84.05±0.11
    empirical            |84.76±1.36 86.17±0.73 | 4.16±0.43  3.19±0.88 |84.77±1.35 86.18±0.72
    empirical_cumulative |84.99±1.00 85.11±0.57 |25.05±3.30 25.30±1.63 |85.49±0.86 85.62±0.50
    val_cal              |88.42±0.13 88.76±0.12 | 9.21±0.43  9.21±0.18 |88.48±0.14 88.82±0.12


## XDS-unsw-temporal-96b-Wb-250n100b  (4 flows × 2 phases, seeds: [14675, 25608, 25694, 52015])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.19% |   9.55% |  89.26% | r25608 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.19% |   9.55% |  89.26% | r25608 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.19% |   9.55% |  89.26% | r25608 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  86.08% |   5.10% |  86.09% | r25608 GS best_acc       empirical
    Best F1 (FPR<5%)         |  85.15% |   4.25% |  85.15% | r25608 GA best_acc       empirical
    Best F1 (FPR<4%)         |       — |       — |       — | —
    Best FPR (any F1)        |  85.15% |   4.25% |  85.15% | r25608 GA best_acc       empirical
    Best FPR (F1>80%)        |  85.15% |   4.25% |  85.15% | r25608 GA best_acc       empirical
    Best Acc (any FPR)       |  89.19% |   9.55% |  89.26% | r25608 GA best_ce        val_cal

### best_fitness  (GS: 4 runs | GA: 4 runs)
    Grid Search : 162±103 neurons | 68±8 bits
    GA Neurons  : 106±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.94±0.29 85.00±0.39 |26.31±0.70 26.21±0.85 |85.50±0.26 85.55±0.36
    fixed_05             |87.08±0.32 87.14±0.40 |18.25±0.79 18.08±1.02 |87.33±0.30 87.39±0.37
    platt                |86.01±0.27 86.12±0.43 |22.62±0.61 22.41±0.86 |86.40±0.25 86.51±0.40
    beta                 |84.44±0.24 84.50±0.34 |27.98±0.44 27.87±0.63 |85.09±0.22 85.14±0.32
    empirical            |82.56±2.92 82.32±2.56 |28.63±16.31 28.42±16.71 |83.49±2.22 83.26±1.88
    empirical_cumulative |87.52±0.18 87.54±0.21 |16.21±0.47 16.00±0.86 |87.71±0.17 87.72±0.19
    val_cal              |87.86±0.18 87.80±0.07 |13.16±0.81 13.48±1.00 |87.98±0.17 87.93±0.08

### best_f1  (GS: 4 runs | GA: 4 runs)
    Grid Search : 162±103 neurons | 68±8 bits
    GA Neurons  : 106±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.94±0.29 85.00±0.39 |26.31±0.70 26.21±0.85 |85.50±0.26 85.55±0.36
    fixed_05             |87.08±0.32 87.14±0.40 |18.25±0.79 18.08±1.02 |87.33±0.30 87.39±0.37
    platt                |86.01±0.27 86.12±0.43 |22.62±0.61 22.41±0.86 |86.40±0.25 86.51±0.40
    beta                 |84.44±0.24 84.50±0.34 |27.98±0.44 27.87±0.63 |85.09±0.22 85.14±0.32
    empirical            |82.56±2.92 82.32±2.56 |28.63±16.31 28.42±16.71 |83.49±2.22 83.26±1.88
    empirical_cumulative |87.52±0.18 87.54±0.21 |16.21±0.47 16.00±0.86 |87.71±0.17 87.72±0.19
    val_cal              |87.86±0.18 87.80±0.07 |13.16±0.81 13.48±1.00 |87.98±0.17 87.93±0.08

### best_fpr  (GS: 4 runs | GA: 4 runs)
    Grid Search : 51±39 neurons | 15±22 bits
    GA Neurons  : 152±138 neurons | 18±20 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |72.04±8.03 71.76±8.25 |50.50±14.69 51.27±15.38 |74.95±6.54 74.85±6.60
    fixed_05             |76.11±6.66 75.95±6.23 |25.15±9.52 25.03±10.44 |76.41±6.62 76.28±6.24
    platt                |74.30±6.98 73.53±7.13 |38.05±8.73 38.86±7.90 |75.33±6.66 74.57±6.89
    beta                 |73.56±6.44 73.16±6.66 |41.39±8.08 43.42±10.24 |74.91±6.12 74.82±6.09
    empirical            |69.67±7.08 69.32±6.59 |56.23±11.67 57.12±10.69 |73.43±5.53 73.21±5.12
    empirical_cumulative |79.39±5.79 79.36±5.87 | 6.56±4.91  5.69±5.19 |79.58±5.76 79.58±5.81
    val_cal              |79.61±5.89 79.56±5.93 | 6.40±3.38  5.77±4.36 |79.76±5.84 79.74±5.86

### best_acc  (GS: 4 runs | GA: 4 runs)
    Grid Search : 162±103 neurons | 68±8 bits
    GA Neurons  : 106±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.94±0.29 85.00±0.39 |26.31±0.70 26.21±0.85 |85.50±0.26 85.55±0.36
    fixed_05             |87.08±0.32 87.14±0.40 |18.25±0.79 18.08±1.02 |87.33±0.30 87.39±0.37
    platt                |86.01±0.27 86.12±0.43 |22.62±0.61 22.41±0.86 |86.40±0.25 86.51±0.40
    beta                 |84.44±0.24 84.50±0.34 |27.98±0.44 27.87±0.63 |85.09±0.22 85.14±0.32
    empirical            |82.56±2.92 82.32±2.56 |28.63±16.31 28.42±16.71 |83.49±2.22 83.26±1.88
    empirical_cumulative |87.52±0.18 87.54±0.21 |16.21±0.47 16.00±0.86 |87.71±0.17 87.72±0.19
    val_cal              |87.86±0.18 87.80±0.07 |13.16±0.81 13.48±1.00 |87.98±0.17 87.93±0.08

### best_ce  (GS: 4 runs | GA: 4 runs)
    Grid Search : 225±50 neurons | 40±9 bits
    GA Neurons  : 109±0 neurons | 41±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.63±0.27 83.77±0.24 |29.64±0.78 29.41±0.65 |84.36±0.23 84.49±0.21
    fixed_05             |85.48±0.64 85.65±0.37 |23.58±2.20 23.25±1.66 |85.91±0.54 86.07±0.30
    platt                |84.66±0.43 84.80±0.23 |26.27±1.40 26.04±1.03 |85.21±0.37 85.35±0.19
    beta                 |83.47±0.35 83.63±0.12 |30.17±0.96 29.89±0.49 |84.23±0.30 84.38±0.10
    empirical            |81.83±4.35 81.89±4.46 |32.47±16.55 32.40±16.70 |83.04±3.57 83.09±3.68
    empirical_cumulative |88.15±0.29 88.25±0.47 |12.89±1.13 13.16±0.69 |88.27±0.27 88.37±0.47
    val_cal              |88.35±0.35 88.45±0.52 |10.44±1.64 10.58±1.49 |88.43±0.33 88.52±0.51


## XDS-unsw-temporal-96b-Wb-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.13% |   9.40% |  89.19% | r74627 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.13% |   9.40% |  89.19% | r74627 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.13% |   9.40% |  89.19% | r74627 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  86.98% |   4.51% |  86.98% | r74627 GA best_fpr       empirical_cumulative
    Best F1 (FPR<5%)         |  86.98% |   4.51% |  86.98% | r74627 GA best_fpr       empirical_cumulative
    Best F1 (FPR<4%)         |  86.83% |   3.90% |  86.83% | r11760 GA best_ce        empirical
    Best FPR (any F1)        |  85.77% |   2.61% |  85.78% | r88021 GA best_ce        empirical
    Best FPR (F1>80%)        |  85.77% |   2.61% |  85.78% | r88021 GA best_ce        empirical
    Best Acc (any FPR)       |  89.13% |   9.40% |  89.19% | r74627 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±283 neurons | 30±3 bits
    GA Neurons  : 438±99 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.96±1.34 83.89±0.12 |31.24±3.75 28.72±0.41 |83.80±1.09 84.56±0.10
    fixed_05             |83.65±1.94 85.13±0.34 |29.08±5.46 24.79±0.86 |84.38±1.61 85.61±0.31
    platt                |83.17±1.59 84.36±0.20 |30.41±4.16 27.26±0.46 |83.96±1.32 84.96±0.18
    beta                 |81.15±3.34 83.34±0.16 |35.98±8.26 30.45±0.47 |82.40±2.60 84.12±0.14
    empirical            |83.56±4.04 84.15±2.04 |18.82±19.03  3.92±1.34 |84.12±3.43 84.17±2.02
    empirical_cumulative |85.34±3.46 87.45±0.40 |22.69±10.58 15.44±2.09 |85.82±2.96 87.62±0.36
    val_cal              |85.94±3.88 88.22±0.14 |18.05±14.91  9.79±0.67 |86.35±3.28 88.28±0.15

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±283 neurons | 30±3 bits
    GA Neurons  : 438±99 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.96±1.34 83.89±0.12 |31.24±3.75 28.72±0.41 |83.80±1.09 84.56±0.10
    fixed_05             |83.65±1.94 85.13±0.34 |29.08±5.46 24.79±0.86 |84.38±1.61 85.61±0.31
    platt                |83.17±1.59 84.36±0.20 |30.41±4.16 27.26±0.46 |83.96±1.32 84.96±0.18
    beta                 |81.15±3.34 83.34±0.16 |35.98±8.26 30.45±0.47 |82.40±2.60 84.12±0.14
    empirical            |83.56±4.04 84.15±2.04 |18.82±19.03  3.92±1.34 |84.12±3.43 84.17±2.02
    empirical_cumulative |85.34±3.46 87.45±0.40 |22.69±10.58 15.44±2.09 |85.82±2.96 87.62±0.36
    val_cal              |85.94±3.88 88.22±0.14 |18.05±14.91  9.79±0.67 |86.35±3.28 88.28±0.15

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 173±202 neurons | 30±5 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |84.34±0.43 82.63±1.97 |26.70±2.06 32.67±6.00 |84.92±0.32 83.59±1.52
    fixed_05             |84.90±0.07 83.99±1.24 |24.81±0.69 28.24±3.90 |85.38±0.09 84.66±1.01
    platt                |84.42±0.36 83.68±0.99 |26.49±1.86 29.30±2.93 |84.98±0.27 84.40±0.82
    beta                 |80.58±4.43 82.31±1.76 |36.55±9.69 33.61±5.40 |81.89±3.55 83.34±1.34
    empirical            |82.46±4.11 84.18±4.46 |27.81±19.41 18.63±20.05 |83.42±3.23 84.77±3.64
    empirical_cumulative |88.40±0.05 87.71±0.64 |10.92±0.68  9.55±4.36 |88.48±0.05 87.78±0.70
    val_cal              |88.58±0.17 87.87±0.78 | 9.70±0.37  8.11±3.13 |88.64±0.16 87.91±0.82

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±283 neurons | 30±3 bits
    GA Neurons  : 438±99 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.96±1.34 83.89±0.12 |31.24±3.75 28.72±0.41 |83.80±1.09 84.56±0.10
    fixed_05             |83.65±1.94 85.13±0.34 |29.08±5.46 24.79±0.86 |84.38±1.61 85.61±0.31
    platt                |83.17±1.59 84.36±0.20 |30.41±4.16 27.26±0.46 |83.96±1.32 84.96±0.18
    beta                 |81.15±3.34 83.34±0.16 |35.98±8.26 30.45±0.47 |82.40±2.60 84.12±0.14
    empirical            |83.56±4.04 84.15±2.04 |18.82±19.03  3.92±1.34 |84.12±3.43 84.17±2.02
    empirical_cumulative |85.34±3.46 87.45±0.40 |22.69±10.58 15.44±2.09 |85.82±2.96 87.62±0.36
    val_cal              |85.94±3.88 88.22±0.14 |18.05±14.91  9.79±0.67 |86.35±3.28 88.28±0.15

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 350±71 neurons | 33±1 bits
    GA Neurons  : 370±30 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.96±0.44 83.84±0.51 |28.45±1.30 29.10±1.40 |84.62±0.37 84.54±0.43
    fixed_05             |85.00±0.24 84.19±0.08 |25.01±0.79 27.95±0.07 |85.49±0.20 84.83±0.08
    platt                |84.34±0.22 84.00±0.16 |27.25±0.73 28.55±0.30 |84.94±0.18 84.67±0.15
    beta                 |83.30±0.08 83.25±0.15 |30.54±0.31 30.91±0.58 |84.08±0.06 84.06±0.12
    empirical            |85.10±0.64 86.36±0.54 | 4.27±0.60  3.10±0.70 |85.11±0.63 86.36±0.54
    empirical_cumulative |87.38±1.09 88.44±0.34 |15.96±4.66 12.58±0.63 |87.58±0.97 88.55±0.33
    val_cal              |88.39±0.17 88.85±0.24 | 9.62±0.46  9.94±0.83 |88.45±0.17 88.92±0.24


## XDS-unsw-temporal-96b-Wc-500n34b  (3 flows × 2 phases, seeds: [11760, 74627, 88021])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  89.69% |   8.02% |  89.74% | r74627 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  89.69% |   8.02% |  89.74% | r74627 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  89.69% |   8.02% |  89.74% | r74627 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  88.09% |   4.24% |  88.10% | r88021 GA best_fpr       empirical
    Best F1 (FPR<5%)         |  88.09% |   4.24% |  88.10% | r88021 GA best_fpr       empirical
    Best F1 (FPR<4%)         |  87.87% |   2.85% |  87.87% | r74627 GA best_ce        empirical
    Best FPR (any F1)        |  87.08% |   1.62% |  87.09% | r74627 GA best_fpr       empirical
    Best FPR (F1>80%)        |  87.08% |   1.62% |  87.09% | r74627 GA best_fpr       empirical
    Best Acc (any FPR)       |  89.69% |   8.02% |  89.74% | r74627 GA best_ce        val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 34±0 bits
    GA Neurons  : 389±15 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.67±0.30 84.30±0.75 |29.27±0.78 27.95±1.72 |84.38±0.25 84.95±0.66
    fixed_05             |85.01±0.37 84.16±0.70 |24.85±0.97 28.37±1.53 |85.49±0.33 84.83±0.62
    platt                |84.36±0.30 84.24±0.55 |27.06±0.73 28.19±1.05 |84.94±0.27 84.89±0.50
    beta                 |83.26±0.15 83.48±0.58 |30.53±0.41 30.52±0.96 |84.04±0.13 84.27±0.53
    empirical            |84.70±0.89 87.00±0.99 | 4.01±0.77  2.89±0.30 |84.70±0.88 87.01±0.99
    empirical_cumulative |85.40±0.63 86.81±0.47 |23.63±2.04 20.31±1.38 |85.83±0.55 87.13±0.43
    val_cal              |88.44±0.02 89.23±0.41 | 9.42±0.25  9.35±1.19 |88.49±0.02 89.30±0.40

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 333±115 neurons | 34±0 bits
    GA Neurons  : 389±13 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.85±0.53 84.37±0.26 |28.72±1.57 27.75±1.15 |84.53±0.44 85.00±0.20
    fixed_05             |85.12±0.17 85.13±0.41 |24.67±0.53 25.20±1.56 |85.59±0.14 85.63±0.34
    platt                |84.37±0.08 84.63±0.17 |27.06±0.22 26.87±0.61 |84.96±0.07 85.21±0.14
    beta                 |83.31±0.07 83.57±0.24 |30.43±0.11 30.10±0.26 |84.08±0.06 84.32±0.23
    empirical            |83.20±3.64 85.57±1.30 |17.25±21.07  3.18±1.10 |83.77±2.76 85.57±1.30
    empirical_cumulative |85.12±0.51 85.97±0.68 |24.58±1.64 22.40±1.83 |85.59±0.43 86.36±0.61
    val_cal              |88.30±0.05 88.93±0.42 |10.30±0.95  9.19±0.23 |88.37±0.04 88.99±0.42

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 367±231 neurons | 34±0 bits
    GA Neurons  : 395±11 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.89±0.64 84.56±0.51 |28.73±1.71 27.28±1.15 |84.57±0.55 85.17±0.46
    fixed_05             |85.18±0.40 84.22±0.81 |24.53±0.99 28.25±1.80 |85.65±0.37 84.88±0.72
    platt                |84.39±0.34 84.27±0.63 |27.07±0.70 28.10±1.24 |84.98±0.31 84.92±0.57
    beta                 |83.26±0.15 83.44±0.57 |30.54±0.33 30.56±0.96 |84.05±0.13 84.23±0.52
    empirical            |83.42±1.83 87.35±0.66 | 9.86±12.36  3.38±1.52 |83.60±2.02 87.35±0.66
    empirical_cumulative |86.23±0.31 86.91±0.49 |21.29±1.33 19.91±1.98 |86.58±0.26 87.21±0.42
    val_cal              |88.42±0.11 89.20±0.34 |10.75±0.40  9.89±1.08 |88.50±0.12 89.28±0.33

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 333±115 neurons | 34±0 bits
    GA Neurons  : 400±18 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.85±0.53 84.47±0.11 |28.72±1.57 27.26±0.34 |84.53±0.44 85.07±0.09
    fixed_05             |85.12±0.17 85.00±0.56 |24.67±0.53 25.43±1.87 |85.59±0.14 85.52±0.48
    platt                |84.37±0.08 84.51±0.24 |27.06±0.22 27.09±0.83 |84.96±0.07 85.11±0.20
    beta                 |83.31±0.07 83.48±0.11 |30.43±0.11 30.19±0.17 |84.08±0.06 84.25±0.10
    empirical            |83.20±3.64 85.44±1.11 |17.25±21.07  3.45±0.64 |83.77±2.76 85.45±1.11
    empirical_cumulative |85.12±0.51 85.79±0.44 |24.58±1.64 22.75±1.45 |85.59±0.43 86.19±0.38
    val_cal              |88.30±0.05 88.80±0.20 |10.30±0.95  9.29±0.34 |88.37±0.04 88.86±0.21

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 34±0 bits
    GA Neurons  : 389±15 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.67±0.30 84.30±0.75 |29.27±0.78 27.95±1.72 |84.38±0.25 84.95±0.66
    fixed_05             |85.01±0.37 84.16±0.70 |24.85±0.97 28.37±1.53 |85.49±0.33 84.83±0.62
    platt                |84.36±0.30 84.24±0.55 |27.06±0.73 28.19±1.05 |84.94±0.27 84.89±0.50
    beta                 |83.26±0.15 83.48±0.58 |30.53±0.41 30.52±0.96 |84.04±0.13 84.27±0.53
    empirical            |84.70±0.89 87.00±0.99 | 4.01±0.77  2.89±0.30 |84.70±0.88 87.01±0.99
    empirical_cumulative |85.40±0.63 86.81±0.47 |23.63±2.04 20.31±1.38 |85.83±0.55 87.13±0.43
    val_cal              |88.44±0.02 89.23±0.41 | 9.42±0.25  9.35±1.19 |88.49±0.02 89.30±0.40



# XDS-unsw-random — width × weight cohort breakdown (71 non-OLD completed)

    Total non-OLD completed : 71  |  Total wall: 125.5h  |  Avg/run: 106m
    Latest done : 10/07/2026 00:58 UTC

    Weight schemes:
      Wa (CIC-IoT legacy, ce=0.35 acc=0.30)
      Wb (paper/PUB50, ce=0.10 acc=0.20)
      Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)
      Wc (CE-heavy NEW, ce=0.70 acc=0.10)


## XDS-unsw-random-8b-Wb-250n100b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.52% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  93.52% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  93.52% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  93.52% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<5%)         |  93.52% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<4%)         |  93.52% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best FPR (any F1)        |  83.61% |   0.09% |  98.12% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  83.61% |   0.09% |  98.12% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  93.52% |   1.12% |  98.92% | r82096 GA best_ce        val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 16±0 bits
    GA Neurons  : 159±0 neurons | 20±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.32     93.44   |    1.16      1.14   |   98.88     98.91  
    platt                |  93.37     93.36   |    1.12      1.11   |   98.90     98.90  
    beta                 |  80.80     93.43   |    0.42      1.12   |   97.72     98.91  
    empirical            |  93.30     93.18   |    1.11      1.11   |   98.89     98.87  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.52     93.52   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 16±0 bits
    GA Neurons  : 159±0 neurons | 20±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.32     93.44   |    1.16      1.14   |   98.88     98.91  
    platt                |  93.37     93.36   |    1.12      1.11   |   98.90     98.90  
    beta                 |  80.80     93.43   |    0.42      1.12   |   97.72     98.91  
    empirical            |  93.30     93.18   |    1.11      1.11   |   98.89     98.87  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.52     93.52   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 4±0 bits
    GA Neurons  : 209±0 neurons | 4±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.24     85.29   |    1.89      2.72   |   98.08     97.19  
    fixed_05             |  87.63     77.29   |    2.43      5.80   |   97.66     94.41  
    platt                |  88.10     84.61   |    0.92      1.09   |   98.25     97.78  
    beta                 |  87.92     83.92   |    0.88      0.99   |   98.24     97.75  
    empirical            |  89.09     84.70   |    1.99      2.99   |   98.03     96.99  
    empirical_cumulative |  89.06     83.61   |    1.03      0.09   |   98.33     98.12  
    val_cal              |  89.24     85.29   |    1.89      2.72   |   98.08     97.19  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 16±0 bits
    GA Neurons  : 159±0 neurons | 20±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.32     93.44   |    1.16      1.14   |   98.88     98.91  
    platt                |  93.37     93.36   |    1.12      1.11   |   98.90     98.90  
    beta                 |  80.80     93.43   |    0.42      1.12   |   97.72     98.91  
    empirical            |  93.30     93.18   |    1.11      1.11   |   98.89     98.87  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.52     93.52   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 16±0 bits
    GA Neurons  : 249±0 neurons | 21±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.42     93.49   |    1.14      1.13   |   98.90     98.92  
    platt                |  93.32     93.33   |    1.12      1.11   |   98.89     98.90  
    beta                 |  93.34     93.30   |    1.12      1.11   |   98.90     98.89  
    empirical            |  93.12     92.98   |    1.11      1.11   |   98.87     98.85  
    empirical_cumulative |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-8b-Wb-500n34b  (2 flows × 2 phases, seeds: [8188, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       val_cal
    Best FPR (any F1)        |  83.12% |   0.24% |  98.01% | r8188 GS best_fpr       train_cal
    Best FPR (F1>80%)        |  83.12% |   0.24% |  98.01% | r8188 GS best_fpr       train_cal
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       val_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±141 neurons | 16±0 bits
    GA Neurons  : 483±0 neurons | 15±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.45±0.03 93.45±0.03 | 1.13±0.01  1.13±0.01 |98.91±0.00 98.91±0.01
    platt                |93.32±0.03 93.33±0.03 | 1.12±0.00  1.12±0.00 |98.89±0.00 98.89±0.00
    beta                 |93.37±0.02 93.32±0.09 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.89±0.01
    empirical            |93.05±0.08 93.07±0.10 | 1.11±0.00  1.11±0.00 |98.86±0.01 98.86±0.01
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±141 neurons | 16±0 bits
    GA Neurons  : 483±0 neurons | 15±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.45±0.03 93.45±0.03 | 1.13±0.01  1.13±0.01 |98.91±0.00 98.91±0.01
    platt                |93.32±0.03 93.33±0.03 | 1.12±0.00  1.12±0.00 |98.89±0.00 98.89±0.00
    beta                 |93.37±0.02 93.32±0.09 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.89±0.01
    empirical            |93.05±0.08 93.07±0.10 | 1.11±0.00  1.11±0.00 |98.86±0.01 98.86±0.01
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 5±0 neurons | 4±0 bits
    GA Neurons  : 14±0 neurons | 4±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |66.07±24.11 77.41±1.26 | 0.12±0.17  1.03±1.41 |97.10±1.29 97.10±0.69
    fixed_05             |55.95±9.79 50.72±6.68 | 7.65±10.82 30.71±12.18 |90.73±7.72 70.19±11.43
    platt                |66.07±24.11 75.69±1.18 | 0.12±0.17  0.56±0.75 |97.10±1.29 97.21±0.54
    beta                 |66.07±24.11 76.98±0.65 | 0.12±0.17  0.05±0.03 |97.10±1.29 97.62±0.03
    empirical            |66.07±24.11 76.98±0.65 | 0.12±0.17  0.05±0.03 |97.10±1.29 97.62±0.03
    empirical_cumulative |66.07±24.11 76.98±0.65 | 0.12±0.17  0.05±0.03 |97.10±1.29 97.62±0.03
    val_cal              |66.07±24.11 77.41±1.26 | 0.12±0.17  1.03±1.41 |97.10±1.29 97.10±0.69

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±141 neurons | 16±0 bits
    GA Neurons  : 483±0 neurons | 15±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.45±0.03 93.45±0.03 | 1.13±0.01  1.13±0.01 |98.91±0.00 98.91±0.01
    platt                |93.32±0.03 93.33±0.03 | 1.12±0.00  1.12±0.00 |98.89±0.00 98.89±0.00
    beta                 |93.37±0.02 93.32±0.09 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.89±0.01
    empirical            |93.05±0.08 93.07±0.10 | 1.11±0.00  1.11±0.00 |98.86±0.01 98.86±0.01
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±141 neurons | 12±0 bits
    GA Neurons  : 458±30 neurons | 15±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |90.80±1.96 93.24±0.30 | 1.70±0.43  1.17±0.06 |98.37±0.42 98.87±0.06
    platt                |93.33±0.00 93.28±0.04 | 1.12±0.00  1.11±0.00 |98.89±0.00 98.89±0.01
    beta                 |92.72±0.51 93.26±0.03 | 1.11±0.00  1.11±0.00 |98.81±0.07 98.88±0.00
    empirical            |93.01±0.11 93.03±0.00 | 1.11±0.00  1.11±0.00 |98.85±0.01 98.85±0.00
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## XDS-unsw-random-8b-Wc-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.52% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  93.52% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  93.52% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  93.52% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  93.52% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  93.52% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  91.09% |   1.05% |  98.59% | r82096 GA best_acc       beta
    Best FPR (F1>80%)        |  91.09% |   1.05% |  98.59% | r82096 GA best_acc       beta
    Best Acc (any FPR)       |  93.52% |   1.12% |  98.92% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 468±0 neurons | 13±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  89.07     93.46   |    2.08      1.13   |   98.00     98.91  
    platt                |  93.33     93.36   |    1.12      1.12   |   98.89     98.90  
    beta                 |  92.74     93.39   |    1.10      1.12   |   98.81     98.90  
    empirical            |  93.08     93.01   |    1.11      1.11   |   98.86     98.85  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 16±0 bits
    GA Neurons  : 500±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.44     90.24   |    1.13      1.81   |   98.91     98.26  
    platt                |  93.34     93.11   |    1.12      1.11   |   98.90     98.86  
    beta                 |  93.37     91.09   |    1.12      1.05   |   98.90     98.59  
    empirical            |  93.06     93.11   |    1.11      1.11   |   98.86     98.86  
    empirical_cumulative |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 486±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  49.03     93.51   |    0.00      1.12   |   96.19     98.92  
    fixed_05             |  49.03     89.91   |    0.00      1.88   |   96.19     98.19  
    platt                |  49.03     92.53   |    0.00      1.11   |   96.19     98.78  
    beta                 |  49.03     91.43   |    0.00      1.07   |   96.19     98.64  
    empirical            |  49.03     93.21   |    0.00      1.11   |   96.19     98.88  
    empirical_cumulative |  49.03     93.51   |    0.00      1.12   |   96.19     98.92  
    val_cal              |  49.03     93.52   |    0.00      1.12   |   96.19     98.92  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 16±0 bits
    GA Neurons  : 500±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.44     90.24   |    1.13      1.81   |   98.91     98.26  
    platt                |  93.34     93.11   |    1.12      1.11   |   98.90     98.86  
    beta                 |  93.37     91.09   |    1.12      1.05   |   98.90     98.59  
    empirical            |  93.06     93.11   |    1.11      1.11   |   98.86     98.86  
    empirical_cumulative |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.52   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 468±0 neurons | 13±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  89.07     93.46   |    2.08      1.13   |   98.00     98.91  
    platt                |  93.33     93.36   |    1.12      1.12   |   98.89     98.90  
    beta                 |  92.74     93.39   |    1.10      1.12   |   98.81     98.90  
    empirical            |  93.08     93.01   |    1.11      1.11   |   98.86     98.85  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-16b-Wb-250n100b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best FPR (any F1)        |  80.53% |   0.41% |  97.69% | r82096 GS best_f1        beta
    Best FPR (F1>80%)        |  80.53% |   0.41% |  97.69% | r82096 GS best_f1        beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 16±0 bits
    GA Neurons  : 246±0 neurons | 38±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.93     93.50   |    1.24      1.12   |   98.81     98.92  
    platt                |  93.34     92.89   |    1.12      1.10   |   98.90     98.83  
    beta                 |  80.53     93.11   |    0.41      1.11   |   97.69     98.86  
    empirical            |  93.11     91.70   |    1.11      1.02   |   98.86     98.68  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 16±0 bits
    GA Neurons  : 246±0 neurons | 38±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.93     93.50   |    1.24      1.12   |   98.81     98.92  
    platt                |  93.34     92.89   |    1.12      1.10   |   98.90     98.83  
    beta                 |  80.53     93.11   |    0.41      1.11   |   97.69     98.86  
    empirical            |  93.11     91.70   |    1.11      1.02   |   98.86     98.68  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 16±0 bits
    GA Neurons  : 13±0 neurons | 16±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.35     88.55   |    1.12      1.68   |   98.90     98.05  
    fixed_05             |  92.64     84.39   |    1.29      3.18   |   98.75     96.87  
    platt                |  93.28     85.36   |    1.11      1.53   |   98.89     97.68  
    beta                 |  93.20     88.38   |    1.11      0.77   |   98.88     98.33  
    empirical            |  93.20     88.54   |    1.11      1.68   |   98.88     98.05  
    empirical_cumulative |  93.35     88.42   |    1.12      0.78   |   98.90     98.34  
    val_cal              |  93.35     88.55   |    1.12      1.68   |   98.90     98.05  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 16±0 bits
    GA Neurons  : 246±0 neurons | 38±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.99     93.50   |    1.22      1.12   |   98.82     98.92  
    platt                |  93.32     92.89   |    1.12      1.10   |   98.89     98.83  
    beta                 |  93.38     93.11   |    1.12      1.11   |   98.90     98.86  
    empirical            |  92.96     91.70   |    1.11      1.02   |   98.84     98.68  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 16±0 bits
    GA Neurons  : 249±0 neurons | 39±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.94     93.50   |    1.24      1.12   |   98.81     98.92  
    platt                |  93.32     92.87   |    1.12      1.10   |   98.89     98.83  
    beta                 |  93.33     92.93   |    1.12      1.10   |   98.89     98.84  
    empirical            |  93.25     92.02   |    1.12      1.05   |   98.88     98.72  
    empirical_cumulative |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-16b-Wb-500n34b  (7 flows × 2 phases, seeds: [8188, 8627, 25608, 60123, 67673, 82096, 92774])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.86% |   0.75% |  99.05% | r25608 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.86% |   0.75% |  99.05% | r25608 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.86% |   0.75% |  99.05% | r25608 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.86% |   0.75% |  99.05% | r25608 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  93.86% |   0.75% |  99.05% | r25608 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  93.86% |   0.75% |  99.05% | r25608 GA best_acc       val_cal
    Best FPR (any F1)        |  87.59% |   0.27% |  98.42% | r8188 GA best_fpr       beta
    Best FPR (F1>80%)        |  87.59% |   0.27% |  98.42% | r8188 GA best_fpr       beta
    Best Acc (any FPR)       |  93.86% |   0.75% |  99.05% | r25608 GA best_acc       val_cal

### best_fitness  (GS: 7 runs | GA: 7 runs)
    Grid Search : 229±159 neurons | 13±2 bits
    GA Neurons  : 68±126 neurons | 13±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.47±0.09 93.61±0.10 | 1.06±0.15  1.03±0.09 |98.93±0.02 98.95±0.03
    fixed_05             |91.51±1.67 92.82±0.53 | 1.54±0.37  1.26±0.11 |98.52±0.35 98.79±0.10
    platt                |93.32±0.05 92.89±1.10 | 1.06±0.15  0.97±0.15 |98.90±0.02 98.87±0.15
    beta                 |90.94±6.23 89.22±5.79 | 0.96±0.28  0.74±0.31 |98.69±0.58 98.52±0.54
    empirical            |93.11±0.09 93.39±0.27 | 1.06±0.15  1.00±0.15 |98.88±0.04 98.93±0.06
    empirical_cumulative |93.47±0.09 93.57±0.12 | 1.06±0.15  1.00±0.15 |98.93±0.02 98.95±0.03
    val_cal              |93.47±0.09 93.61±0.10 | 1.06±0.15  1.03±0.09 |98.93±0.02 98.95±0.03

### best_f1  (GS: 7 runs | GA: 7 runs)
    Grid Search : 271±125 neurons | 13±2 bits
    GA Neurons  : 68±126 neurons | 13±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.61±0.10 | 1.12±0.00  1.03±0.09 |98.92±0.00 98.95±0.03
    fixed_05             |91.48±1.72 92.82±0.53 | 1.55±0.38  1.26±0.11 |98.51±0.36 98.79±0.10
    platt                |93.34±0.01 92.89±1.10 | 1.12±0.00  0.97±0.15 |98.90±0.00 98.87±0.15
    beta                 |90.97±6.24 89.22±5.79 | 1.02±0.26  0.74±0.31 |98.68±0.57 98.52±0.54
    empirical            |93.11±0.09 93.39±0.27 | 1.11±0.00  1.00±0.15 |98.86±0.01 98.93±0.06
    empirical_cumulative |93.51±0.00 93.57±0.12 | 1.12±0.00  1.00±0.15 |98.92±0.00 98.95±0.03
    val_cal              |93.51±0.00 93.61±0.10 | 1.12±0.00  1.03±0.09 |98.92±0.00 98.95±0.03

### best_fpr  (GS: 7 runs | GA: 7 runs)
    Grid Search : 5±0 neurons | 18±11 bits
    GA Neurons  : 14±5 neurons | 16±7 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.72±5.77 85.32±11.18 | 0.98±0.57  3.95±5.75 |98.52±0.54 96.04±5.59
    fixed_05             |84.71±11.74 81.75±11.74 | 4.35±5.15  5.65±7.01 |95.81±4.97 94.54±6.76
    platt                |89.51±5.67 82.41±14.82 | 0.82±0.34  1.22±0.62 |98.53±0.50 97.79±0.76
    beta                 |88.69±7.21 84.35±10.80 | 0.79±0.39  2.96±6.17 |98.48±0.56 96.25±5.68
    empirical            |89.63±5.77 84.78±12.06 | 0.82±0.35  4.39±7.42 |98.55±0.51 95.52±7.04
    empirical_cumulative |89.65±5.77 83.17±15.11 | 0.83±0.35  0.78±0.59 |98.55±0.51 98.05±0.84
    val_cal              |89.72±5.77 85.33±11.18 | 0.98±0.57  3.97±5.74 |98.52±0.54 96.04±5.59

### best_acc  (GS: 7 runs | GA: 7 runs)
    Grid Search : 229±159 neurons | 14±2 bits
    GA Neurons  : 68±126 neurons | 13±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.47±0.09 93.60±0.14 | 1.06±0.15  1.04±0.14 |98.93±0.02 98.95±0.05
    fixed_05             |91.64±1.76 92.27±1.24 | 1.51±0.38  1.38±0.26 |98.54±0.37 98.67±0.26
    platt                |93.32±0.05 92.91±1.12 | 1.06±0.15  0.95±0.18 |98.90±0.02 98.87±0.16
    beta                 |90.93±6.23 89.83±5.98 | 0.96±0.28  0.77±0.30 |98.68±0.57 98.59±0.56
    empirical            |93.12±0.08 93.39±0.30 | 1.06±0.15  0.98±0.18 |98.88±0.04 98.93±0.07
    empirical_cumulative |93.47±0.09 93.59±0.15 | 1.06±0.15  0.98±0.18 |98.93±0.02 98.96±0.05
    val_cal              |93.48±0.10 93.60±0.14 | 1.06±0.15  1.04±0.14 |98.93±0.02 98.95±0.05

### best_ce  (GS: 7 runs | GA: 7 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 374±142 neurons | 16±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |91.82±1.32 93.03±0.30 | 1.47±0.29  1.22±0.06 |98.58±0.28 98.83±0.06
    platt                |93.32±0.01 93.24±0.11 | 1.12±0.00  1.11±0.01 |98.89±0.00 98.88±0.01
    beta                 |93.07±0.13 93.26±0.04 | 1.11±0.01  1.11±0.00 |98.86±0.02 98.88±0.01
    empirical            |93.05±0.07 93.00±0.28 | 1.11±0.00  1.10±0.01 |98.85±0.01 98.85±0.04
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## XDS-unsw-random-16b-Wc-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  91.83% |   1.01% |  98.70% | r82096 GS best_f1        beta
    Best FPR (F1>80%)        |  91.83% |   1.01% |  98.70% | r82096 GS best_f1        beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 378±0 neurons | 17±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  89.52     93.46   |    1.97      1.13   |   98.10     98.91  
    platt                |  93.32     93.28   |    1.12      1.11   |   98.89     98.89  
    beta                 |  92.95     93.37   |    1.11      1.12   |   98.84     98.90  
    empirical            |  93.16     93.00   |    1.12      1.11   |   98.87     98.85  
    empirical_cumulative |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 24±0 bits
    GA Neurons  : 290±0 neurons | 19±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    platt                |  93.27     93.26   |    1.11      1.11   |   98.89     98.88  
    beta                 |  91.83     93.20   |    1.01      1.11   |   98.70     98.88  
    empirical            |  92.32     92.59   |    1.08      1.10   |   98.76     98.79  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 8±0 bits
    GA Neurons  : 384±0 neurons | 17±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.38     93.49   |    1.12      1.12   |   98.90     98.92  
    fixed_05             |  88.43     93.46   |    2.23      1.13   |   97.85     98.91  
    platt                |  93.27     93.27   |    1.12      1.11   |   98.88     98.89  
    beta                 |  93.11     93.38   |    1.11      1.12   |   98.86     98.90  
    empirical            |  93.23     92.75   |    1.11      1.10   |   98.88     98.81  
    empirical_cumulative |  93.38     93.49   |    1.12      1.12   |   98.90     98.92  
    val_cal              |  93.38     93.49   |    1.12      1.12   |   98.90     98.92  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 12±0 bits
    GA Neurons  : 290±0 neurons | 19±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  89.02     93.51   |    2.09      1.12   |   97.99     98.92  
    platt                |  93.35     93.26   |    1.12      1.11   |   98.90     98.88  
    beta                 |  93.38     93.20   |    1.12      1.11   |   98.90     98.88  
    empirical            |  93.17     92.59   |    1.11      1.10   |   98.87     98.79  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 378±0 neurons | 17±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  89.52     93.46   |    1.97      1.13   |   98.10     98.91  
    platt                |  93.32     93.28   |    1.12      1.11   |   98.89     98.89  
    beta                 |  92.95     93.37   |    1.11      1.12   |   98.84     98.90  
    empirical            |  93.16     93.00   |    1.12      1.11   |   98.87     98.85  
    empirical_cumulative |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-32b-Wa-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best FPR (any F1)        |  91.99% |   1.05% |  98.72% | r82096 GS best_fpr       empirical
    Best FPR (F1>80%)        |  91.99% |   1.05% |  98.72% | r82096 GS best_fpr       empirical
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 12±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.51     92.51   |    1.32      1.32   |   98.73     98.73  
    platt                |  93.35     93.35   |    1.12      1.12   |   98.90     98.90  
    beta                 |  93.34     93.34   |    1.12      1.12   |   98.90     98.90  
    empirical            |  93.29     93.29   |    1.11      1.11   |   98.89     98.89  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 12±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.51     92.51   |    1.32      1.32   |   98.73     98.73  
    platt                |  93.35     93.35   |    1.12      1.12   |   98.90     98.90  
    beta                 |  93.34     93.34   |    1.12      1.12   |   98.90     98.90  
    empirical            |  93.29     93.29   |    1.11      1.11   |   98.89     98.89  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 28±0 bits
    GA Neurons  : 500±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.48     93.50   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.08     92.48   |    1.20      1.33   |   98.84     98.72  
    platt                |  93.21     93.33   |    1.11      1.12   |   98.88     98.89  
    beta                 |  92.62     93.34   |    1.08      1.12   |   98.80     98.90  
    empirical            |  91.99     93.10   |    1.05      1.11   |   98.72     98.86  
    empirical_cumulative |  93.48     93.50   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.48     93.50   |    1.12      1.12   |   98.92     98.92  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 12±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.51     92.51   |    1.32      1.32   |   98.73     98.73  
    platt                |  93.35     93.35   |    1.12      1.12   |   98.90     98.90  
    beta                 |  93.34     93.34   |    1.12      1.12   |   98.90     98.90  
    empirical            |  93.29     93.29   |    1.11      1.11   |   98.89     98.89  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 294±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.49     93.50   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.22     92.70   |    1.38      1.28   |   98.67     98.77  
    platt                |  93.35     93.35   |    1.12      1.12   |   98.90     98.90  
    beta                 |  92.43     93.12   |    1.11      1.08   |   98.76     98.87  
    empirical            |  93.12     93.34   |    1.11      1.12   |   98.86     98.90  
    empirical_cumulative |  93.49     93.50   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.49     93.50   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-32b-Wb-250n100b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_ce        val_cal
    Best FPR (any F1)        |  85.56% |   0.75% |  98.03% | r82096 GA best_fpr       beta
    Best FPR (F1>80%)        |  85.56% |   0.75% |  98.03% | r82096 GA best_fpr       beta
    Best Acc (any FPR)       |  93.49% |   0.99% |  98.94% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 16±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.50   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.00     93.00   |    1.22      1.22   |   98.82     98.82  
    platt                |  93.33     93.33   |    1.12      1.12   |   98.89     98.89  
    beta                 |  93.27     93.27   |    1.11      1.11   |   98.89     98.89  
    empirical            |  93.14     93.14   |    1.10      1.10   |   98.87     98.87  
    empirical_cumulative |  93.50     93.50   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.50   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 16±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.50   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.00     93.00   |    1.22      1.22   |   98.82     98.82  
    platt                |  93.33     93.33   |    1.12      1.12   |   98.89     98.89  
    beta                 |  93.27     93.27   |    1.11      1.11   |   98.89     98.89  
    empirical            |  93.14     93.14   |    1.10      1.10   |   98.87     98.87  
    empirical_cumulative |  93.50     93.50   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.50   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 64±0 bits
    GA Neurons  : 9±0 neurons | 16±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.37     85.63   |    1.12      0.75   |   98.90     98.03  
    fixed_05             |  93.34     77.52   |    1.11      5.48   |   98.90     94.62  
    platt                |  92.57     84.77   |    1.01      1.53   |   98.81     97.61  
    beta                 |  90.98     85.56   |    0.94      0.75   |   98.61     98.03  
    empirical            |  93.28     85.56   |    1.11      0.75   |   98.89     98.03  
    empirical_cumulative |  93.37     85.63   |    1.12      0.75   |   98.90     98.03  
    val_cal              |  93.37     85.63   |    1.12      0.75   |   98.90     98.03  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 16±0 bits
    GA Neurons  : 8±0 neurons | 16±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.49   |    1.12      0.99   |   98.92     98.94  
    fixed_05             |  93.00     88.91   |    1.22      2.12   |   98.82     97.96  
    platt                |  93.33     91.23   |    1.12      0.88   |   98.89     98.66  
    beta                 |  93.27     86.39   |    1.11      0.77   |   98.89     98.11  
    empirical            |  93.14     93.46   |    1.10      0.99   |   98.87     98.94  
    empirical_cumulative |  93.50     93.49   |    1.12      0.99   |   98.92     98.94  
    val_cal              |  93.50     93.49   |    1.12      0.99   |   98.92     98.94  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 16±0 bits
    GA Neurons  : 141±0 neurons | 15±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.92     92.98   |    1.24      1.23   |   98.81     98.82  
    platt                |  93.33     93.32   |    1.12      1.12   |   98.90     98.89  
    beta                 |  93.33     93.32   |    1.12      1.12   |   98.89     98.89  
    empirical            |  92.92     93.11   |    1.11      1.11   |   98.84     98.86  
    empirical_cumulative |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-32b-Wb-500n34b  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r8188 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r8188 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r8188 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r8188 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r8188 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r8188 GA best_acc       val_cal
    Best FPR (any F1)        |  86.57% |   0.62% |  98.19% | r8188 GA best_acc       beta
    Best FPR (F1>80%)        |  86.57% |   0.62% |  98.19% | r8188 GA best_acc       beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r8188 GS best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 367±231 neurons | 15±2 bits
    GA Neurons  : 96±2 neurons | 15±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.81±0.31 92.88±0.11 | 1.26±0.06  1.25±0.02 |98.79±0.06 98.80±0.02
    platt                |93.34±0.00 93.28±0.10 | 1.12±0.00  1.10±0.02 |98.90±0.00 98.89±0.01
    beta                 |93.27±0.15 91.04±3.87 | 1.12±0.00  0.94±0.28 |98.89±0.02 98.66±0.41
    empirical            |93.10±0.12 93.18±0.20 | 1.11±0.00  1.11±0.00 |98.86±0.02 98.87±0.03
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 367±231 neurons | 15±2 bits
    GA Neurons  : 96±2 neurons | 15±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.81±0.31 92.88±0.11 | 1.26±0.06  1.25±0.02 |98.79±0.06 98.80±0.02
    platt                |93.34±0.00 93.28±0.10 | 1.12±0.00  1.10±0.02 |98.90±0.00 98.89±0.01
    beta                 |93.27±0.15 91.04±3.87 | 1.12±0.00  0.94±0.28 |98.89±0.02 98.66±0.41
    empirical            |93.10±0.12 93.18±0.20 | 1.11±0.00  1.11±0.00 |98.86±0.02 98.87±0.03
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 5±0 neurons | 16±8 bits
    GA Neurons  : 15±3 neurons | 17±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |92.87±0.61 91.51±2.70 | 1.10±0.04  1.39±0.35 |98.83±0.09 98.56±0.49
    fixed_05             |91.47±2.68 86.94±9.48 | 1.56±0.60  3.03±2.92 |98.50±0.57 97.09±2.80
    platt                |92.49±0.57 90.14±4.58 | 1.08±0.06  1.22±0.23 |98.78±0.08 98.43±0.67
    beta                 |91.29±2.21 90.92±3.11 | 1.41±0.61  0.93±0.17 |98.51±0.49 98.63±0.35
    empirical            |92.85±0.62 91.40±2.62 | 1.10±0.04  1.29±0.45 |98.83±0.09 98.56±0.50
    empirical_cumulative |92.87±0.61 91.06±3.19 | 1.10±0.04  0.91±0.17 |98.83±0.09 98.65±0.36
    val_cal              |92.87±0.61 91.51±2.70 | 1.10±0.04  1.39±0.35 |98.83±0.09 98.56±0.49

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 367±231 neurons | 15±2 bits
    GA Neurons  : 96±2 neurons | 15±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.81±0.31 92.88±0.11 | 1.26±0.06  1.25±0.02 |98.79±0.06 98.80±0.02
    platt                |93.34±0.00 93.28±0.10 | 1.12±0.00  1.10±0.02 |98.90±0.00 98.89±0.01
    beta                 |93.27±0.15 91.04±3.87 | 1.12±0.00  0.94±0.28 |98.89±0.02 98.66±0.41
    empirical            |93.10±0.12 93.18±0.20 | 1.11±0.00  1.11±0.00 |98.86±0.02 98.87±0.03
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±141 neurons | 14±3 bits
    GA Neurons  : 371±162 neurons | 18±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.51±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.58±0.40 93.12±0.32 | 1.31±0.08  1.20±0.06 |98.74±0.08 98.85±0.06
    platt                |93.35±0.00 93.30±0.05 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.89±0.01
    beta                 |93.03±0.52 93.28±0.03 | 1.11±0.00  1.12±0.00 |98.85±0.07 98.89±0.00
    empirical            |93.10±0.04 92.76±0.37 | 1.11±0.00  1.10±0.02 |98.86±0.01 98.81±0.05
    empirical_cumulative |93.50±0.01 93.51±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.50±0.01 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## XDS-unsw-random-32b-Wc-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GA best_fpr       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_fpr       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_fpr       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_fpr       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_fpr       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_fpr       val_cal
    Best FPR (any F1)        |  93.00% |   1.09% |  98.85% | r82096 GA best_ce        empirical
    Best FPR (F1>80%)        |  93.00% |   1.09% |  98.85% | r82096 GA best_ce        empirical
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GA best_fpr       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 418±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.49     93.48   |    1.12      1.12   |   98.92     98.91  
    fixed_05             |  92.22     92.52   |    1.38      1.32   |   98.67     98.73  
    platt                |  93.35     93.30   |    1.12      1.12   |   98.90     98.89  
    beta                 |  92.43     93.08   |    1.11      1.11   |   98.76     98.86  
    empirical            |  93.12     93.00   |    1.11      1.09   |   98.86     98.85  
    empirical_cumulative |  93.49     93.48   |    1.12      1.12   |   98.92     98.91  
    val_cal              |  93.49     93.50   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 12±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  88.75     88.75   |    2.16      2.16   |   97.93     97.93  
    platt                |  93.32     93.32   |    1.12      1.12   |   98.89     98.89  
    beta                 |  93.31     93.31   |    1.12      1.12   |   98.89     98.89  
    empirical            |  93.05     93.05   |    1.11      1.11   |   98.85     98.85  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 20±0 bits
    GA Neurons  : 500±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.48     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.04     92.27   |    1.21      1.37   |   98.83     98.68  
    platt                |  93.30     93.35   |    1.12      1.12   |   98.89     98.90  
    beta                 |  93.25     93.31   |    1.11      1.12   |   98.88     98.89  
    empirical            |  92.73     92.92   |    1.10      1.11   |   98.81     98.84  
    empirical_cumulative |  93.48     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.48     93.51   |    1.12      1.12   |   98.92     98.92  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 12±0 bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  88.75     88.75   |    2.16      2.16   |   97.93     97.93  
    platt                |  93.32     93.32   |    1.12      1.12   |   98.89     98.89  
    beta                 |  93.31     93.31   |    1.12      1.12   |   98.89     98.89  
    empirical            |  93.05     93.05   |    1.11      1.11   |   98.85     98.85  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 418±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.49     93.48   |    1.12      1.12   |   98.92     98.91  
    fixed_05             |  92.22     92.52   |    1.38      1.32   |   98.67     98.73  
    platt                |  93.35     93.30   |    1.12      1.12   |   98.90     98.89  
    beta                 |  92.43     93.08   |    1.11      1.11   |   98.76     98.86  
    empirical            |  93.12     93.00   |    1.11      1.09   |   98.86     98.85  
    empirical_cumulative |  93.49     93.48   |    1.12      1.12   |   98.92     98.91  
    val_cal              |  93.49     93.50   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-64b-Wa-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  86.35% |   0.74% |  98.11% | r82096 GS best_fpr       beta
    Best FPR (F1>80%)        |  86.35% |   0.74% |  98.11% | r82096 GS best_fpr       beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GA best_f1        val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 103±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  88.93     92.64   |    2.11      1.30   |   97.97     98.75  
    platt                |  93.26     93.44   |    1.12      1.12   |   98.88     98.91  
    beta                 |  93.41     93.40   |    1.12      1.11   |   98.91     98.91  
    empirical            |  93.26     93.38   |    1.12      1.11   |   98.88     98.90  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 103±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  88.93     92.64   |    2.11      1.30   |   97.97     98.75  
    platt                |  93.26     93.44   |    1.12      1.12   |   98.88     98.91  
    beta                 |  93.41     93.40   |    1.12      1.11   |   98.91     98.91  
    empirical            |  93.26     93.38   |    1.12      1.11   |   98.88     98.90  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 16±0 bits
    GA Neurons  : 99±0 neurons | 13±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.41     93.08   |    1.12      1.10   |   98.90     98.86  
    fixed_05             |  92.91     92.93   |    1.24      1.23   |   98.81     98.81  
    platt                |  93.29     92.92   |    1.12      1.20   |   98.89     98.82  
    beta                 |  86.35     93.02   |    0.74      1.21   |   98.11     98.83  
    empirical            |  93.15     93.07   |    1.11      1.10   |   98.87     98.86  
    empirical_cumulative |  93.41     93.08   |    1.12      1.10   |   98.90     98.86  
    val_cal              |  93.41     93.08   |    1.12      1.10   |   98.91     98.86  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 103±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  88.93     92.64   |    2.11      1.30   |   97.97     98.75  
    platt                |  93.26     93.44   |    1.12      1.12   |   98.88     98.91  
    beta                 |  93.41     93.40   |    1.12      1.11   |   98.91     98.91  
    empirical            |  93.26     93.38   |    1.12      1.11   |   98.88     98.90  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 104±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.48     93.35   |    1.12      1.12   |   98.92     98.90  
    fixed_05             |  92.04     92.89   |    1.42      1.24   |   98.63     98.80  
    platt                |  93.34     93.33   |    1.12      1.12   |   98.90     98.89  
    beta                 |  92.89     93.03   |    1.09      1.21   |   98.84     98.83  
    empirical            |  93.29     93.25   |    1.12      1.11   |   98.89     98.88  
    empirical_cumulative |  93.48     93.35   |    1.12      1.12   |   98.92     98.90  
    val_cal              |  93.48     93.35   |    1.12      1.12   |   98.92     98.90  


## XDS-unsw-random-64b-Wb-250n100b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal
    Best FPR (any F1)        |  86.69% |   0.84% |  98.11% | r82096 GA best_acc       beta
    Best FPR (F1>80%)        |  86.69% |   0.84% |  98.11% | r82096 GA best_acc       beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GS best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 16±0 bits
    GA Neurons  : 62±0 neurons | 24±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.49   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.95     93.18   |    1.23      1.19   |   98.81     98.86  
    platt                |  93.35     93.21   |    1.11      1.11   |   98.90     98.88  
    beta                 |  91.15     86.69   |    1.08      0.84   |   98.59     98.11  
    empirical            |  93.21     92.97   |    1.11      1.10   |   98.88     98.84  
    empirical_cumulative |  93.51     93.49   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 16±0 bits
    GA Neurons  : 62±0 neurons | 24±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.49   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.95     93.18   |    1.23      1.19   |   98.81     98.86  
    platt                |  93.35     93.21   |    1.11      1.11   |   98.90     98.88  
    beta                 |  91.15     86.69   |    1.08      0.84   |   98.59     98.11  
    empirical            |  93.21     92.97   |    1.11      1.10   |   98.88     98.84  
    empirical_cumulative |  93.51     93.49   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 32±0 bits
    GA Neurons  : 19±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.21     92.98   |    1.10      1.22   |   98.88     98.82  
    fixed_05             |  93.01     92.97   |    1.21      1.22   |   98.83     98.82  
    platt                |  92.86     92.67   |    1.08      1.17   |   98.83     98.79  
    beta                 |  92.54     91.75   |    1.08      1.01   |   98.79     98.69  
    empirical            |  93.21     92.70   |    1.10      1.14   |   98.88     98.80  
    empirical_cumulative |  93.21     92.96   |    1.10      1.20   |   98.88     98.82  
    val_cal              |  93.21     92.99   |    1.10      1.22   |   98.88     98.82  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 16±0 bits
    GA Neurons  : 62±0 neurons | 24±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.49   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.95     93.18   |    1.23      1.19   |   98.81     98.86  
    platt                |  93.35     93.21   |    1.11      1.11   |   98.90     98.88  
    beta                 |  91.15     86.69   |    1.08      0.84   |   98.59     98.11  
    empirical            |  93.21     92.97   |    1.11      1.10   |   98.88     98.84  
    empirical_cumulative |  93.51     93.49   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.50   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 16±0 bits
    GA Neurons  : 249±0 neurons | 26±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.47     93.50   |    1.12      1.12   |   98.91     98.92  
    fixed_05             |  92.86     93.04   |    1.25      1.21   |   98.80     98.83  
    platt                |  93.32     93.12   |    1.12      1.11   |   98.89     98.87  
    beta                 |  93.32     93.16   |    1.12      1.11   |   98.89     98.87  
    empirical            |  93.05     92.49   |    1.11      1.06   |   98.85     98.78  
    empirical_cumulative |  93.47     93.50   |    1.12      1.12   |   98.91     98.92  
    val_cal              |  93.47     93.50   |    1.12      1.12   |   98.91     98.92  


## XDS-unsw-random-64b-Wb-500n34b  (39 flows × 2 phases, seeds: [2647, 6858, 8161, 8188, 8627, 13119, 14613, 17375, 17717, 17821, 20521, 21395, 21777, 25608, 25987, 26607, 30971, 35419, 35432, 39086, 43427, 44520, 48846, 50011, 57192, 60123, 67673, 67784, 69436, 73846, 73945, 75501, 77021, 78572, 82096, 92726, 92774, 96530, 96660])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.92% |   0.68% |  99.07% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  93.92% |   0.68% |  99.07% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  93.92% |   0.68% |  99.07% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  93.92% |   0.68% |  99.07% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  93.92% |   0.68% |  99.07% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  93.92% |   0.68% |  99.07% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  83.79% |   0.07% |  98.14% | r92726 GA best_fpr       beta
    Best FPR (F1>80%)        |  83.79% |   0.07% |  98.14% | r92726 GA best_fpr       beta
    Best Acc (any FPR)       |  93.92% |   0.68% |  99.07% | r82096 GA best_f1        train_cal

### best_fitness  (GS: 39 runs | GA: 39 runs)
    Grid Search : 272±148 neurons | 14±2 bits
    GA Neurons  : 145±125 neurons | 14±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.55±0.11 | 1.11±0.04  1.08±0.10 |98.92±0.01 98.94±0.04
    fixed_05             |92.28±1.04 92.25±1.32 | 1.37±0.23  1.38±0.29 |98.68±0.22 98.67±0.28
    platt                |93.34±0.03 93.16±0.58 | 1.11±0.04  1.06±0.13 |98.90±0.01 98.88±0.08
    beta                 |91.07±4.59 91.88±3.24 | 0.99±0.23  0.99±0.21 |98.67±0.45 98.75±0.33
    empirical            |93.15±0.13 93.25±0.26 | 1.11±0.04  1.06±0.13 |98.87±0.02 98.89±0.06
    empirical_cumulative |93.50±0.00 93.55±0.11 | 1.11±0.04  1.08±0.10 |98.92±0.01 98.94±0.04
    val_cal              |93.51±0.00 93.56±0.11 | 1.11±0.04  1.08±0.10 |98.92±0.01 98.94±0.04

### best_f1  (GS: 39 runs | GA: 39 runs)
    Grid Search : 285±146 neurons | 14±2 bits
    GA Neurons  : 145±125 neurons | 14±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.55±0.11 | 1.12±0.00  1.08±0.10 |98.92±0.00 98.94±0.04
    fixed_05             |92.38±0.85 92.25±1.32 | 1.35±0.19  1.38±0.29 |98.70±0.18 98.67±0.28
    platt                |93.33±0.03 93.16±0.58 | 1.12±0.00  1.06±0.13 |98.89±0.00 98.88±0.08
    beta                 |91.07±4.59 91.88±3.24 | 1.00±0.23  0.99±0.21 |98.67±0.45 98.75±0.33
    empirical            |93.14±0.11 93.25±0.26 | 1.11±0.00  1.06±0.13 |98.87±0.02 98.89±0.06
    empirical_cumulative |93.50±0.00 93.55±0.11 | 1.12±0.00  1.08±0.10 |98.92±0.00 98.94±0.04
    val_cal              |93.50±0.00 93.56±0.11 | 1.12±0.00  1.08±0.10 |98.92±0.00 98.94±0.04

### best_fpr  (GS: 39 runs | GA: 39 runs)
    Grid Search : 27±61 neurons | 17±11 bits
    GA Neurons  : 31±44 neurons | 15±9 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.60±4.73 85.98±9.53 | 1.23±0.66  2.84±4.07 |98.50±0.66 96.87±4.08
    fixed_05             |88.42±7.64 79.51±13.91 | 2.64±3.48  7.25±8.98 |97.40±3.50 92.92±8.81
    platt                |90.19±4.93 83.68±12.53 | 0.96±0.31  1.09±0.45 |98.56±0.44 97.92±0.79
    beta                 |90.01±4.83 84.76±9.35 | 0.90±0.30  1.66±3.12 |98.56±0.40 97.30±3.15
    empirical            |90.48±4.87 85.08±11.15 | 1.10±0.52  3.23±6.42 |98.55±0.47 96.32±6.28
    empirical_cumulative |90.44±4.93 84.42±12.59 | 0.95±0.36  0.78±0.62 |98.60±0.43 98.12±0.79
    val_cal              |90.60±4.73 85.99±9.53 | 1.23±0.66  2.82±4.07 |98.50±0.66 96.87±4.08

### best_acc  (GS: 39 runs | GA: 39 runs)
    Grid Search : 244±143 neurons | 14±2 bits
    GA Neurons  : 109±119 neurons | 14±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.53±0.14 | 1.11±0.04  1.04±0.15 |98.92±0.01 98.94±0.04
    fixed_05             |92.35±1.06 91.91±1.80 | 1.36±0.23  1.46±0.41 |98.69±0.22 98.59±0.39
    platt                |93.33±0.04 93.14±0.54 | 1.11±0.04  0.99±0.18 |98.90±0.01 98.90±0.08
    beta                 |90.93±4.60 92.09±2.62 | 0.98±0.23  0.92±0.24 |98.65±0.45 98.78±0.28
    empirical            |93.15±0.13 93.27±0.27 | 1.11±0.04  1.01±0.15 |98.87±0.02 98.91±0.06
    empirical_cumulative |93.50±0.00 93.52±0.15 | 1.11±0.04  1.02±0.16 |98.92±0.01 98.94±0.04
    val_cal              |93.50±0.00 93.53±0.14 | 1.11±0.04  1.04±0.15 |98.92±0.01 98.94±0.04

### best_ce  (GS: 39 runs | GA: 39 runs)
    Grid Search : 382±135 neurons | 12±2 bits
    GA Neurons  : 312±158 neurons | 16±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.44±0.29 93.47±0.09 | 1.13±0.06  1.12±0.01 |98.91±0.06 98.91±0.01
    fixed_05             |92.38±0.24 92.92±0.22 | 1.35±0.05  1.24±0.04 |98.70±0.05 98.81±0.04
    platt                |93.23±0.68 93.25±0.13 | 1.12±0.01  1.11±0.01 |98.88±0.10 98.88±0.02
    beta                 |93.06±0.68 93.21±0.14 | 1.11±0.01  1.11±0.01 |98.86±0.09 98.88±0.02
    empirical            |93.08±0.25 93.07±0.17 | 1.12±0.06  1.10±0.05 |98.86±0.05 98.86±0.03
    empirical_cumulative |93.44±0.29 93.47±0.11 | 1.13±0.06  1.12±0.01 |98.91±0.06 98.91±0.01
    val_cal              |93.44±0.29 93.47±0.09 | 1.13±0.06  1.12±0.01 |98.91±0.06 98.91±0.01


## XDS-unsw-random-64b-Wc-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  92.19% |   1.05% |  98.75% | r82096 GA best_acc       beta
    Best FPR (F1>80%)        |  92.19% |   1.05% |  98.75% | r82096 GA best_acc       beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 242±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.46     93.14   |    1.12      1.12   |   98.91     98.87  
    fixed_05             |  92.54     92.97   |    1.32      1.23   |   98.73     98.82  
    platt                |  93.35     92.89   |    1.12      1.20   |   98.90     98.81  
    beta                 |  93.24     93.00   |    1.11      1.20   |   98.88     98.83  
    empirical            |  93.13     93.08   |    1.11      1.11   |   98.87     98.86  
    empirical_cumulative |  93.46     93.14   |    1.12      1.12   |   98.91     98.87  
    val_cal              |  93.46     93.14   |    1.12      1.12   |   98.91     98.87  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 16±0 bits
    GA Neurons  : 102±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.03     92.41   |    1.22      1.34   |   98.83     98.71  
    platt                |  93.34     93.28   |    1.12      1.12   |   98.90     98.89  
    beta                 |  93.30     92.19   |    1.12      1.05   |   98.89     98.75  
    empirical            |  93.21     93.16   |    1.11      1.12   |   98.88     98.87  
    empirical_cumulative |  93.49     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 16±0 bits
    GA Neurons  : 247±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.49     93.06   |    1.12      1.21   |   98.92     98.84  
    fixed_05             |  92.87     92.90   |    1.25      1.24   |   98.80     98.80  
    platt                |  93.32     92.88   |    1.12      1.20   |   98.89     98.81  
    beta                 |  93.34     92.97   |    1.12      1.20   |   98.90     98.82  
    empirical            |  93.00     92.94   |    1.10      1.11   |   98.85     98.84  
    empirical_cumulative |  93.49     93.02   |    1.12      1.11   |   98.92     98.85  
    val_cal              |  93.49     93.06   |    1.12      1.21   |   98.92     98.84  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 16±0 bits
    GA Neurons  : 102±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  93.03     92.41   |    1.22      1.34   |   98.83     98.71  
    platt                |  93.34     93.28   |    1.12      1.12   |   98.90     98.89  
    beta                 |  93.30     92.19   |    1.12      1.05   |   98.89     98.75  
    empirical            |  93.21     93.16   |    1.11      1.12   |   98.88     98.87  
    empirical_cumulative |  93.49     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 12±0 bits
    GA Neurons  : 242±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.46     93.14   |    1.12      1.12   |   98.91     98.87  
    fixed_05             |  92.54     92.97   |    1.32      1.23   |   98.73     98.82  
    platt                |  93.35     92.89   |    1.12      1.20   |   98.90     98.81  
    beta                 |  93.24     93.00   |    1.11      1.20   |   98.88     98.83  
    empirical            |  93.13     93.08   |    1.11      1.11   |   98.87     98.86  
    empirical_cumulative |  93.46     93.14   |    1.12      1.12   |   98.91     98.87  
    val_cal              |  93.46     93.14   |    1.12      1.12   |   98.91     98.87  


## XDS-unsw-random-96b-Wa-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  88.44% |   0.76% |  98.35% | r82096 GA best_acc       beta
    Best FPR (F1>80%)        |  88.44% |   0.76% |  98.35% | r82096 GA best_acc       beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r82096 GA best_acc       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 16±0 bits
    GA Neurons  : 201±0 neurons | 13±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.99     89.25   |    1.22      2.04   |   98.82     98.04  
    platt                |  93.36     93.09   |    1.12      1.11   |   98.90     98.86  
    beta                 |  92.61     88.44   |    1.05      0.76   |   98.81     98.35  
    empirical            |  93.16     93.38   |    1.11      1.12   |   98.87     98.90  
    empirical_cumulative |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 16±0 bits
    GA Neurons  : 201±0 neurons | 13±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.99     89.25   |    1.22      2.04   |   98.82     98.04  
    platt                |  93.36     93.09   |    1.12      1.11   |   98.90     98.86  
    beta                 |  92.61     88.44   |    1.05      0.76   |   98.81     98.35  
    empirical            |  93.16     93.38   |    1.11      1.12   |   98.87     98.90  
    empirical_cumulative |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 32±0 bits
    GA Neurons  : 209±0 neurons | 13±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.45     93.51   |    1.12      1.12   |   98.91     98.92  
    fixed_05             |  93.05     89.17   |    1.21      2.06   |   98.83     98.02  
    platt                |  93.15     93.15   |    1.11      1.11   |   98.87     98.87  
    beta                 |  89.89     89.60   |    0.89      0.85   |   98.48     98.46  
    empirical            |  92.49     93.33   |    1.07      1.12   |   98.78     98.89  
    empirical_cumulative |  93.45     93.51   |    1.12      1.12   |   98.91     98.92  
    val_cal              |  93.45     93.51   |    1.12      1.12   |   98.91     98.92  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 16±0 bits
    GA Neurons  : 201±0 neurons | 13±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.99     89.25   |    1.22      2.04   |   98.82     98.04  
    platt                |  93.36     93.09   |    1.12      1.11   |   98.90     98.86  
    beta                 |  92.61     88.44   |    1.05      0.76   |   98.81     98.35  
    empirical            |  93.16     93.38   |    1.11      1.12   |   98.87     98.90  
    empirical_cumulative |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.50     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 12±0 bits
    GA Neurons  : 189±0 neurons | 14±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.48     93.47   |    1.12      1.12   |   98.92     98.91  
    fixed_05             |  92.08     92.85   |    1.41      1.25   |   98.64     98.79  
    platt                |  93.34     93.34   |    1.12      1.12   |   98.90     98.90  
    beta                 |  92.94     93.37   |    1.11      1.12   |   98.84     98.90  
    empirical            |  93.15     93.14   |    1.11      1.11   |   98.87     98.87  
    empirical_cumulative |  93.48     93.47   |    1.12      1.12   |   98.92     98.91  
    val_cal              |  93.49     93.47   |    1.12      1.12   |   98.92     98.91  


## XDS-unsw-random-96b-Wb-250n100b  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r25608 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r25608 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r25608 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r25608 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r25608 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r25608 GA best_acc       val_cal
    Best FPR (any F1)        |  84.52% |   0.35% |  98.09% | r25608 GA best_fpr       beta
    Best FPR (F1>80%)        |  84.52% |   0.35% |  98.09% | r25608 GA best_fpr       beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r25608 GA best_ce        empirical_cumulative

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 175±35 neurons | 16±0 bits
    GA Neurons  : 242±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.89±0.03 93.19±0.45 | 1.25±0.01  1.18±0.09 |98.80±0.01 98.86±0.09
    platt                |93.33±0.01 93.14±0.27 | 1.12±0.00  1.11±0.01 |98.89±0.00 98.87±0.04
    beta                 |93.24±0.05 93.21±0.01 | 1.11±0.00  1.11±0.00 |98.88±0.01 98.88±0.00
    empirical            |93.01±0.20 92.54±0.87 | 1.11±0.00  1.07±0.06 |98.85±0.03 98.79±0.11
    empirical_cumulative |93.50±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.50±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 175±35 neurons | 16±0 bits
    GA Neurons  : 242±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.89±0.03 93.19±0.45 | 1.25±0.01  1.18±0.09 |98.80±0.01 98.86±0.09
    platt                |93.33±0.01 93.14±0.27 | 1.12±0.00  1.11±0.01 |98.89±0.00 98.87±0.04
    beta                 |93.24±0.05 93.21±0.01 | 1.11±0.00  1.11±0.00 |98.88±0.01 98.88±0.00
    empirical            |93.01±0.20 92.54±0.87 | 1.11±0.00  1.07±0.06 |98.85±0.03 98.79±0.11
    empirical_cumulative |93.50±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.50±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 5±0 neurons | 72±34 bits
    GA Neurons  : 11±3 neurons | 16±11 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.10±0.18 89.30±5.47 | 1.07±0.05  1.42±0.35 |98.87±0.01 98.25±0.86
    fixed_05             |92.91±0.03 85.87±4.55 | 1.21±0.01  2.42±0.49 |98.81±0.01 97.39±0.86
    platt                |92.60±0.31 88.30±6.25 | 1.01±0.04  1.22±0.28 |98.81±0.03 98.19±0.90
    beta                 |92.14±0.24 88.14±5.12 | 0.97±0.01  0.63±0.39 |98.76±0.03 98.41±0.45
    empirical            |93.02±0.29 88.01±6.86 | 1.14±0.05  1.77±1.04 |98.84±0.05 97.91±1.32
    empirical_cumulative |93.10±0.18 88.84±6.12 | 1.07±0.05  0.76±0.59 |98.87±0.01 98.47±0.54
    val_cal              |93.10±0.18 89.30±5.47 | 1.07±0.05  1.42±0.35 |98.87±0.01 98.25±0.86

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 100±71 neurons | 16±0 bits
    GA Neurons  : 242±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.49±0.01 93.50±0.02 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.83±0.11 93.13±0.53 | 1.26±0.02  1.20±0.11 |98.79±0.02 98.85±0.10
    platt                |93.32±0.01 93.14±0.27 | 1.12±0.00  1.11±0.01 |98.89±0.00 98.87±0.04
    beta                 |93.20±0.10 93.17±0.06 | 1.11±0.00  1.11±0.00 |98.88±0.01 98.87±0.01
    empirical            |92.97±0.14 92.50±0.81 | 1.11±0.00  1.07±0.06 |98.84±0.02 98.79±0.10
    empirical_cumulative |93.49±0.01 93.50±0.02 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.49±0.01 93.50±0.02 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 250±0 neurons | 72±11 bits
    GA Neurons  : 249±0 neurons | 46±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.46±0.00 93.50±0.01 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00
    fixed_05             |93.33±0.02 93.47±0.02 | 1.12±0.00  1.12±0.00 |98.89±0.00 98.92±0.00
    platt                |92.54±0.05 92.69±0.08 | 1.06±0.01  1.08±0.00 |98.79±0.01 98.81±0.01
    beta                 |87.84±5.24 92.89±0.11 | 0.67±0.39  1.10±0.01 |98.36±0.46 98.84±0.01
    empirical            |90.81±0.19 91.16±0.24 | 0.86±0.03  0.92±0.01 |98.61±0.02 98.64±0.03
    empirical_cumulative |93.46±0.00 93.50±0.01 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.92±0.00
    val_cal              |93.51±0.01 93.50±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## XDS-unsw-random-96b-Wb-500n34b  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  92.25% |   0.42% |  98.90% | r82096 GA best_f1        beta
    Best FPR (F1>80%)        |  92.25% |   0.42% |  98.90% | r82096 GA best_f1        beta
    Best Acc (any FPR)       |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±141 neurons | 14±3 bits
    GA Neurons  : 271±207 neurons | 11±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.60±0.16 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06
    fixed_05             |91.47±2.12 90.31±1.66 | 1.55±0.47  1.80±0.36 |98.50±0.45 98.27±0.35
    platt                |93.33±0.01 93.10±0.45 | 1.12±0.00  0.89±0.39 |98.89±0.00 98.91±0.02
    beta                 |91.70±2.83 92.86±0.57 | 1.03±0.16  0.89±0.40 |98.70±0.33 98.88±0.04
    empirical            |93.20±0.14 93.40±0.31 | 1.11±0.01  1.02±0.16 |98.88±0.02 98.92±0.08
    empirical_cumulative |93.50±0.01 93.60±0.16 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06
    val_cal              |93.51±0.00 93.60±0.15 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±141 neurons | 14±3 bits
    GA Neurons  : 271±207 neurons | 11±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.60±0.16 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06
    fixed_05             |91.47±2.12 90.31±1.66 | 1.55±0.47  1.80±0.36 |98.50±0.45 98.27±0.35
    platt                |93.33±0.01 93.10±0.45 | 1.12±0.00  0.89±0.39 |98.89±0.00 98.91±0.02
    beta                 |91.70±2.83 92.86±0.57 | 1.03±0.16  0.89±0.40 |98.70±0.33 98.88±0.04
    empirical            |93.20±0.14 93.40±0.31 | 1.11±0.01  1.02±0.16 |98.88±0.02 98.92±0.08
    empirical_cumulative |93.50±0.01 93.60±0.16 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06
    val_cal              |93.51±0.00 93.60±0.15 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 52±67 neurons | 24±11 bits
    GA Neurons  : 71±55 neurons | 17±14 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |91.90±2.43 91.07±2.64 | 1.10±0.07  1.29±0.59 |98.70±0.32 98.51±0.53
    fixed_05             |91.13±2.74 87.26±5.50 | 1.64±0.62  2.63±1.40 |98.42±0.59 97.46±1.35
    platt                |91.65±2.27 89.99±3.79 | 1.00±0.16  1.09±0.18 |98.69±0.32 98.45±0.52
    beta                 |91.38±1.99 89.63±2.83 | 0.97±0.13  0.83±0.07 |98.66±0.28 98.48±0.36
    empirical            |91.38±2.13 90.70±2.65 | 1.32±0.66  1.40±0.88 |98.55±0.49 98.41±0.64
    empirical_cumulative |91.78±2.36 90.93±2.68 | 0.99±0.15  0.80±0.28 |98.71±0.32 98.66±0.27
    val_cal              |91.90±2.44 91.07±2.64 | 1.10±0.07  1.29±0.59 |98.70±0.32 98.51±0.53

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±141 neurons | 14±3 bits
    GA Neurons  : 270±208 neurons | 11±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.60±0.16 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06
    fixed_05             |91.47±2.12 90.30±1.67 | 1.55±0.47  1.81±0.37 |98.50±0.45 98.26±0.35
    platt                |93.33±0.01 93.22±0.25 | 1.12±0.00  0.93±0.33 |98.89±0.00 98.92±0.04
    beta                 |91.70±2.83 92.98±0.38 | 1.03±0.16  0.90±0.38 |98.70±0.33 98.89±0.05
    empirical            |93.20±0.14 93.39±0.30 | 1.11±0.01  1.02±0.16 |98.88±0.02 98.92±0.07
    empirical_cumulative |93.50±0.01 93.60±0.16 | 1.12±0.00  1.03±0.16 |98.92±0.00 98.95±0.06
    val_cal              |93.51±0.00 93.60±0.15 | 1.12±0.00  1.02±0.16 |98.92±0.00 98.95±0.06

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 350±212 neurons | 12±0 bits
    GA Neurons  : 133±53 neurons | 18±7 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.47±0.01 93.37±0.10 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.90±0.01
    fixed_05             |92.43±0.20 92.92±0.09 | 1.34±0.04  1.24±0.02 |98.71±0.04 98.81±0.02
    platt                |93.34±0.01 93.19±0.12 | 1.12±0.00  1.11±0.00 |98.90±0.00 98.88±0.02
    beta                 |93.27±0.03 93.25±0.04 | 1.12±0.00  1.11±0.00 |98.89±0.00 98.88±0.01
    empirical            |93.06±0.05 92.68±0.25 | 1.11±0.01  0.92±0.25 |98.86±0.01 98.85±0.04
    empirical_cumulative |93.48±0.01 93.37±0.10 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.90±0.01
    val_cal              |93.48±0.01 93.38±0.10 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.90±0.01


## XDS-unsw-random-96b-Wc-250n100b  (1 flows × 2 phases, seeds: [69488])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r69488 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r69488 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r69488 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r69488 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r69488 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r69488 GA best_acc       val_cal
    Best FPR (any F1)        |  84.67% |   0.42% |  98.07% | r69488 GS best_fpr       beta
    Best FPR (F1>80%)        |  84.67% |   0.42% |  98.07% | r69488 GS best_fpr       beta
    Best Acc (any FPR)       |  93.51% |   1.12% |  98.92% | r69488 GA best_acc       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 246±0 neurons | 61±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.46     93.48   |    1.12      1.12   |   98.91     98.92  
    fixed_05             |  93.31     93.35   |    1.11      1.12   |   98.89     98.90  
    platt                |  92.49     92.50   |    1.06      1.05   |   98.78     98.79  
    beta                 |  86.30     92.75   |    0.53      1.08   |   98.19     98.82  
    empirical            |  90.47     90.70   |    0.84      0.81   |   98.57     98.61  
    empirical_cumulative |  93.46     93.48   |    1.12      1.12   |   98.91     98.92  
    val_cal              |  93.49     93.50   |    1.12      1.12   |   98.92     98.92  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 16±0 bits
    GA Neurons  : 205±0 neurons | 44±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.95     93.49   |    1.23      1.12   |   98.81     98.92  
    platt                |  93.38     93.09   |    1.12      1.11   |   98.90     98.86  
    beta                 |  93.29     93.22   |    1.12      1.11   |   98.89     98.88  
    empirical            |  93.23     92.28   |    1.11      1.04   |   98.88     98.76  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 248±0 neurons | 61±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.46     93.46   |    1.12      1.12   |   98.91     98.91  
    fixed_05             |  93.31     93.34   |    1.11      1.12   |   98.89     98.90  
    platt                |  92.48     92.50   |    1.06      1.05   |   98.78     98.79  
    beta                 |  84.67     92.82   |    0.42      1.09   |   98.07     98.83  
    empirical            |  90.84     90.55   |    0.88      0.83   |   98.61     98.58  
    empirical_cumulative |  93.46     93.46   |    1.12      1.12   |   98.91     98.91  
    val_cal              |  93.50     93.49   |    1.12      1.12   |   98.92     98.92  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 16±0 bits
    GA Neurons  : 205±0 neurons | 44±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    fixed_05             |  92.95     93.49   |    1.23      1.12   |   98.81     98.92  
    platt                |  93.38     93.09   |    1.12      1.11   |   98.90     98.86  
    beta                 |  93.29     93.22   |    1.12      1.11   |   98.89     98.88  
    empirical            |  93.23     92.28   |    1.11      1.04   |   98.88     98.76  
    empirical_cumulative |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  
    val_cal              |  93.51     93.51   |    1.12      1.12   |   98.92     98.92  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 246±0 neurons | 61±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  93.46     93.48   |    1.12      1.12   |   98.91     98.92  
    fixed_05             |  93.31     93.35   |    1.11      1.12   |   98.89     98.90  
    platt                |  92.49     92.50   |    1.06      1.05   |   98.78     98.79  
    beta                 |  86.30     92.75   |    0.53      1.08   |   98.19     98.82  
    empirical            |  90.47     90.70   |    0.84      0.81   |   98.57     98.61  
    empirical_cumulative |  93.46     93.48   |    1.12      1.12   |   98.91     98.92  
    val_cal              |  93.49     93.50   |    1.12      1.12   |   98.92     98.92  


## XDS-unsw-random-96b-Wc-500n34b  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.51% |   1.12% |  98.92% | r8188 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  93.51% |   1.12% |  98.92% | r8188 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  93.51% |   1.12% |  98.92% | r8188 GA best_f1        val_cal
    Best F1 (FPR<6%)         |  93.51% |   1.12% |  98.92% | r8188 GA best_f1        val_cal
    Best F1 (FPR<5%)         |  93.51% |   1.12% |  98.92% | r8188 GA best_f1        val_cal
    Best F1 (FPR<4%)         |  93.51% |   1.12% |  98.92% | r8188 GA best_f1        val_cal
    Best FPR (any F1)        |  93.14% |   0.63% |  98.97% | r8188 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  93.14% |   0.63% |  98.97% | r8188 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  93.14% |   0.63% |  98.97% | r8188 GA best_fpr       empirical_cumulative

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 267±115 neurons | 12±0 bits
    GA Neurons  : 313±151 neurons | 13±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.49±0.01 93.47±0.04 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01
    fixed_05             |91.21±2.13 92.70±0.27 | 1.61±0.47  1.28±0.05 |98.45±0.46 98.76±0.05
    platt                |93.35±0.01 93.30±0.04 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.89±0.01
    beta                 |93.27±0.09 93.32±0.04 | 1.12±0.00  1.12±0.00 |98.89±0.01 98.89±0.01
    empirical            |93.11±0.11 93.16±0.07 | 1.11±0.00  1.11±0.00 |98.86±0.02 98.87±0.01
    empirical_cumulative |93.49±0.01 93.47±0.04 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01
    val_cal              |93.49±0.01 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 233±58 neurons | 12±0 bits
    GA Neurons  : 279±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.51±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |91.28±1.93 91.27±1.92 | 1.59±0.43  1.59±0.42 |98.47±0.41 98.47±0.41
    platt                |93.35±0.02 93.34±0.01 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.00
    beta                 |91.72±2.85 91.71±2.84 | 1.03±0.16  1.03±0.16 |98.70±0.34 98.70±0.34
    empirical            |93.24±0.07 93.28±0.02 | 1.11±0.01  1.12±0.00 |98.88±0.01 98.89±0.00
    empirical_cumulative |93.50±0.00 93.51±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.50±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 233±231 neurons | 16±7 bits
    GA Neurons  : 325±156 neurons | 13±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.32±0.23 93.43±0.07 | 1.12±0.01  1.12±0.00 |98.89±0.04 98.91±0.01
    fixed_05             |91.46±2.69 92.76±0.13 | 1.56±0.60  1.27±0.03 |98.49±0.57 98.78±0.03
    platt                |93.20±0.15 93.21±0.22 | 1.12±0.01  1.02±0.18 |98.87±0.02 98.90±0.01
    beta                 |88.12±4.87 93.20±0.22 | 0.90±0.27  1.00±0.20 |98.31±0.50 98.90±0.01
    empirical            |92.98±0.11 93.09±0.11 | 1.11±0.01  1.11±0.00 |98.84±0.01 98.86±0.02
    empirical_cumulative |93.32±0.24 93.36±0.19 | 1.12±0.01  0.95±0.28 |98.89±0.04 98.93±0.03
    val_cal              |93.32±0.24 93.43±0.07 | 1.12±0.01  1.12±0.00 |98.89±0.04 98.91±0.01

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±100 neurons | 12±0 bits
    GA Neurons  : 271±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.46±0.08 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01
    fixed_05             |92.40±0.02 92.52±0.23 | 1.35±0.00  1.32±0.05 |98.70±0.00 98.73±0.05
    platt                |93.35±0.02 93.22±0.21 | 1.12±0.00  1.02±0.17 |98.90±0.00 98.90±0.01
    beta                 |91.70±2.83 91.57±2.73 | 1.03±0.16  0.92±0.17 |98.70±0.33 98.71±0.34
    empirical            |93.19±0.10 93.19±0.10 | 1.11±0.01  1.11±0.01 |98.88±0.01 98.88±0.01
    empirical_cumulative |93.51±0.00 93.39±0.21 | 1.12±0.00  0.96±0.28 |98.92±0.00 98.94±0.03
    val_cal              |93.51±0.00 93.46±0.07 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 267±115 neurons | 12±0 bits
    GA Neurons  : 313±151 neurons | 13±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.49±0.01 93.47±0.04 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01
    fixed_05             |91.21±2.13 92.70±0.27 | 1.61±0.47  1.28±0.05 |98.45±0.46 98.76±0.05
    platt                |93.35±0.01 93.30±0.04 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.89±0.01
    beta                 |93.27±0.09 93.32±0.04 | 1.12±0.00  1.12±0.00 |98.89±0.01 98.89±0.01
    empirical            |93.11±0.11 93.16±0.07 | 1.11±0.00  1.11±0.00 |98.86±0.02 98.87±0.01
    empirical_cumulative |93.49±0.01 93.47±0.04 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01
    val_cal              |93.49±0.01 93.47±0.03 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.91±0.01



# XDS-cicids — width × weight cohort breakdown (70 non-OLD completed)

    Total non-OLD completed : 70  |  Total wall: 370.2h  |  Avg/run: 317m
    Latest done : 28/06/2026 17:46 UTC

    Weight schemes:
      Wa (CIC-IoT legacy, ce=0.35 acc=0.30)
      Wb (paper/PUB50, ce=0.10 acc=0.20)
      Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)
      Wc (CE-heavy NEW, ce=0.70 acc=0.10)


## XDS-cicids-8b-Wa-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.35% |   0.25% |  99.59% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.35% |   0.25% |  99.59% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.35% |   0.25% |  99.59% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.35% |   0.25% |  99.59% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.35% |   0.25% |  99.59% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.35% |   0.25% |  99.59% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.32% |   0.22% |  99.57% | r82096 GA best_fpr       train_cal
    Best FPR (F1>80%)        |  99.32% |   0.22% |  99.57% | r82096 GA best_fpr       train_cal
    Best Acc (any FPR)       |  99.35% |   0.25% |  99.59% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 105±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.35   |    0.29      0.25   |   99.55     99.59  
    fixed_05             |  98.97     98.81   |    0.63      0.78   |   99.35     99.24  
    platt                |  99.28     99.27   |    0.31      0.36   |   99.54     99.54  
    beta                 |  99.19     99.32   |    0.27      0.28   |   99.49     99.57  
    empirical            |  99.15     99.35   |    0.47      0.25   |   99.46     99.59  
    empirical_cumulative |  99.29     99.35   |    0.29      0.25   |   99.55     99.59  
    val_cal              |  99.29     99.35   |    0.29      0.25   |   99.55     99.59  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 105±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.35   |    0.26      0.25   |   99.55     99.59  
    fixed_05             |  98.87     98.81   |    0.72      0.78   |   99.28     99.24  
    platt                |  99.27     99.27   |    0.29      0.36   |   99.54     99.54  
    beta                 |  99.20     99.32   |    0.25      0.28   |   99.50     99.57  
    empirical            |  99.18     99.35   |    0.38      0.25   |   99.48     99.59  
    empirical_cumulative |  99.29     99.35   |    0.26      0.25   |   99.55     99.59  
    val_cal              |  99.29     99.35   |    0.26      0.25   |   99.55     99.59  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 24±0 bits
    GA Neurons  : 107±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.03     99.32   |    0.23      0.22   |   99.39     99.57  
    fixed_05             |  97.87     98.73   |    1.50      0.84   |   98.63     99.19  
    platt                |  98.95     99.27   |    0.40      0.35   |   99.34     99.54  
    beta                 |  99.01     99.29   |    0.31      0.30   |   99.37     99.55  
    empirical            |  98.37     95.73   |    1.08      3.40   |   98.96     97.17  
    empirical_cumulative |  99.03     99.32   |    0.23      0.22   |   99.39     99.57  
    val_cal              |  99.04     99.32   |    0.27      0.22   |   99.39     99.57  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 105±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.35   |    0.26      0.25   |   99.55     99.59  
    fixed_05             |  98.87     98.81   |    0.72      0.78   |   99.28     99.24  
    platt                |  99.27     99.27   |    0.29      0.36   |   99.54     99.54  
    beta                 |  99.20     99.32   |    0.25      0.28   |   99.50     99.57  
    empirical            |  99.18     99.35   |    0.38      0.25   |   99.48     99.59  
    empirical_cumulative |  99.29     99.35   |    0.26      0.25   |   99.55     99.59  
    val_cal              |  99.29     99.35   |    0.26      0.25   |   99.55     99.59  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 264±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.20   |    0.29      0.25   |   99.55     99.50  
    fixed_05             |  98.97     98.73   |    0.63      0.84   |   99.35     99.19  
    platt                |  99.28     98.98   |    0.31      0.53   |   99.54     99.35  
    beta                 |  99.19     99.18   |    0.27      0.27   |   99.49     99.48  
    empirical            |  99.15     97.91   |    0.47      1.55   |   99.46     98.65  
    empirical_cumulative |  99.29     99.20   |    0.29      0.25   |   99.55     99.50  
    val_cal              |  99.29     99.20   |    0.29      0.25   |   99.55     99.50  


## XDS-cicids-8b-Wb-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  96.87% |   0.16% |  98.08% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  96.87% |   0.16% |  98.08% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 106±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.26     99.39   |    0.25      0.22   |   99.53     99.62  
    fixed_05             |  98.85     98.90   |    0.72      0.70   |   99.27     99.30  
    platt                |  99.20     99.31   |    0.36      0.34   |   99.49     99.56  
    beta                 |  99.26     99.39   |    0.25      0.22   |   99.53     99.62  
    empirical            |  99.18     99.39   |    0.39      0.21   |   99.48     99.62  
    empirical_cumulative |  99.26     99.39   |    0.25      0.21   |   99.53     99.62  
    val_cal              |  99.26     99.39   |    0.25      0.21   |   99.53     99.62  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 106±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.26     99.39   |    0.25      0.22   |   99.53     99.62  
    fixed_05             |  98.85     98.90   |    0.72      0.70   |   99.27     99.30  
    platt                |  99.20     99.31   |    0.36      0.34   |   99.49     99.56  
    beta                 |  99.26     99.39   |    0.25      0.22   |   99.53     99.62  
    empirical            |  99.18     99.39   |    0.39      0.21   |   99.48     99.62  
    empirical_cumulative |  99.26     99.39   |    0.25      0.21   |   99.53     99.62  
    val_cal              |  99.26     99.39   |    0.25      0.21   |   99.53     99.62  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 12±0 bits
    GA Neurons  : 213±0 neurons | 12±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  96.96     96.93   |    0.29      0.31   |   98.13     98.11  
    fixed_05             |  90.07     87.69   |    8.10     10.55   |   93.03     91.09  
    platt                |  96.78     96.77   |    0.52      0.56   |   98.01     98.00  
    beta                 |  96.91     96.92   |    0.34      0.34   |   98.10     98.10  
    empirical            |  96.76     94.43   |    1.14      3.51   |   97.96     96.34  
    empirical_cumulative |  96.96     96.87   |    0.29      0.16   |   98.13     98.08  
    val_cal              |  96.96     96.93   |    0.29      0.31   |   98.13     98.11  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 106±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.26     99.39   |    0.25      0.22   |   99.53     99.62  
    fixed_05             |  98.85     98.90   |    0.72      0.70   |   99.27     99.30  
    platt                |  99.20     99.31   |    0.36      0.34   |   99.49     99.56  
    beta                 |  99.26     99.39   |    0.25      0.22   |   99.53     99.62  
    empirical            |  99.18     99.39   |    0.39      0.21   |   99.48     99.62  
    empirical_cumulative |  99.26     99.39   |    0.25      0.21   |   99.53     99.62  
    val_cal              |  99.26     99.39   |    0.25      0.21   |   99.53     99.62  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 273±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.24   |    0.28      0.35   |   99.54     99.52  
    fixed_05             |  98.95     98.87   |    0.65      0.73   |   99.33     99.28  
    platt                |  99.22     99.14   |    0.36      0.45   |   99.51     99.46  
    beta                 |  99.18     99.16   |    0.27      0.34   |   99.48     99.47  
    empirical            |  99.08     99.14   |    0.53      0.49   |   99.42     99.45  
    empirical_cumulative |  99.19     99.20   |    0.21      0.23   |   99.49     99.50  
    val_cal              |  99.27     99.24   |    0.28      0.35   |   99.54     99.52  


## XDS-cicids-8b-Wbu-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.39% |   0.21% |  99.62% | r82096 GA best_f1        train_cal
    Best F1 (FPR<14%)        |  99.39% |   0.21% |  99.62% | r82096 GA best_f1        train_cal
    Best F1 (FPR<10%)        |  99.39% |   0.21% |  99.62% | r82096 GA best_f1        train_cal
    Best F1 (FPR<6%)         |  99.39% |   0.21% |  99.62% | r82096 GA best_f1        train_cal
    Best F1 (FPR<5%)         |  99.39% |   0.21% |  99.62% | r82096 GA best_f1        train_cal
    Best F1 (FPR<4%)         |  99.39% |   0.21% |  99.62% | r82096 GA best_f1        train_cal
    Best FPR (any F1)        |  94.85% |   0.05% |  96.93% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  94.85% |   0.05% |  96.93% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.39% |   0.21% |  99.62% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 193±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.39   |    0.29      0.21   |   99.55     99.62  
    fixed_05             |  98.93     99.07   |    0.66      0.56   |   99.32     99.41  
    platt                |  99.26     99.36   |    0.32      0.28   |   99.53     99.59  
    beta                 |  99.19     99.36   |    0.28      0.20   |   99.49     99.60  
    empirical            |  99.16     99.39   |    0.46      0.21   |   99.47     99.61  
    empirical_cumulative |  99.28     99.39   |    0.29      0.21   |   99.55     99.62  
    val_cal              |  99.29     99.39   |    0.29      0.21   |   99.55     99.62  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 193±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.39   |    0.29      0.21   |   99.55     99.62  
    fixed_05             |  98.93     99.07   |    0.66      0.56   |   99.32     99.41  
    platt                |  99.26     99.36   |    0.32      0.28   |   99.53     99.59  
    beta                 |  99.19     99.36   |    0.28      0.20   |   99.49     99.60  
    empirical            |  99.16     99.39   |    0.46      0.21   |   99.47     99.61  
    empirical_cumulative |  99.28     99.39   |    0.29      0.21   |   99.55     99.62  
    val_cal              |  99.29     99.39   |    0.29      0.21   |   99.55     99.62  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 8±0 bits
    GA Neurons  : 215±0 neurons | 8±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  95.76     94.90   |    0.20      0.20   |   97.44     96.95  
    fixed_05             |  80.94     80.14   |   17.66     18.35   |   85.17     84.47  
    platt                |  95.17     93.26   |    1.19      2.14   |   97.01     95.80  
    beta                 |  95.58     93.43   |    0.70      1.92   |   97.29     95.92  
    empirical            |  69.98     86.79   |   32.64     10.20   |   73.58     90.59  
    empirical_cumulative |  95.76     94.85   |    0.20      0.05   |   97.44     96.93  
    val_cal              |  95.76     94.90   |    0.20      0.20   |   97.44     96.95  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 193±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.39   |    0.29      0.21   |   99.55     99.62  
    fixed_05             |  98.93     99.07   |    0.66      0.56   |   99.32     99.41  
    platt                |  99.26     99.36   |    0.32      0.28   |   99.53     99.59  
    beta                 |  99.19     99.36   |    0.28      0.20   |   99.49     99.60  
    empirical            |  99.16     99.39   |    0.46      0.21   |   99.47     99.61  
    empirical_cumulative |  99.28     99.39   |    0.29      0.21   |   99.55     99.62  
    val_cal              |  99.29     99.39   |    0.29      0.21   |   99.55     99.62  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 196±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.32   |    0.29      0.23   |   99.55     99.57  
    fixed_05             |  98.93     99.09   |    0.66      0.54   |   99.32     99.42  
    platt                |  99.26     99.25   |    0.32      0.36   |   99.53     99.52  
    beta                 |  99.19     99.31   |    0.28      0.26   |   99.49     99.56  
    empirical            |  99.16     99.11   |    0.46      0.53   |   99.47     99.43  
    empirical_cumulative |  99.28     99.32   |    0.29      0.23   |   99.55     99.57  
    val_cal              |  99.29     99.32   |    0.29      0.23   |   99.55     99.57  


## XDS-cicids-8b-Wc-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.36% |   0.24% |  99.60% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.36% |   0.24% |  99.60% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.36% |   0.24% |  99.60% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.36% |   0.24% |  99.60% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.36% |   0.24% |  99.60% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.36% |   0.24% |  99.60% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.35% |   0.20% |  99.59% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.35% |   0.20% |  99.59% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.36% |   0.24% |  99.60% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 195±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.30   |    0.31      0.30   |   99.54     99.56  
    fixed_05             |  98.92     98.92   |    0.68      0.70   |   99.31     99.31  
    platt                |  99.20     99.21   |    0.31      0.41   |   99.49     99.50  
    beta                 |  99.18     99.27   |    0.27      0.29   |   99.48     99.54  
    empirical            |  99.13     99.30   |    0.49      0.30   |   99.45     99.55  
    empirical_cumulative |  99.28     99.30   |    0.31      0.30   |   99.54     99.56  
    val_cal              |  99.28     99.30   |    0.31      0.30   |   99.54     99.56  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 193±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.36   |    0.26      0.24   |   99.55     99.60  
    fixed_05             |  98.87     99.07   |    0.72      0.57   |   99.28     99.41  
    platt                |  99.27     99.27   |    0.29      0.35   |   99.54     99.54  
    beta                 |  99.20     99.35   |    0.25      0.25   |   99.50     99.59  
    empirical            |  99.18     99.17   |    0.38      0.47   |   99.48     99.47  
    empirical_cumulative |  99.29     99.36   |    0.26      0.24   |   99.55     99.60  
    val_cal              |  99.29     99.36   |    0.26      0.24   |   99.55     99.60  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 8±0 bits
    GA Neurons  : 193±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  95.92     99.36   |    0.25      0.20   |   97.52     99.59  
    fixed_05             |  81.65     99.06   |   16.70      0.57   |   85.87     99.40  
    platt                |  95.47     99.27   |    0.76      0.35   |   97.22     99.54  
    beta                 |  95.70     99.34   |    0.54      0.25   |   97.38     99.59  
    empirical            |  88.39     97.87   |    9.15      1.58   |   91.79     98.62  
    empirical_cumulative |  95.92     99.35   |    0.25      0.20   |   97.52     99.59  
    val_cal              |  95.92     99.36   |    0.25      0.20   |   97.52     99.59  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 193±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.36   |    0.26      0.24   |   99.55     99.60  
    fixed_05             |  98.87     99.07   |    0.72      0.57   |   99.28     99.41  
    platt                |  99.27     99.27   |    0.29      0.35   |   99.54     99.54  
    beta                 |  99.20     99.35   |    0.25      0.25   |   99.50     99.59  
    empirical            |  99.18     99.17   |    0.38      0.47   |   99.48     99.47  
    empirical_cumulative |  99.29     99.36   |    0.26      0.24   |   99.55     99.60  
    val_cal              |  99.29     99.36   |    0.26      0.24   |   99.55     99.60  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 195±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.30   |    0.31      0.30   |   99.54     99.56  
    fixed_05             |  98.92     98.92   |    0.68      0.70   |   99.31     99.31  
    platt                |  99.20     99.21   |    0.31      0.41   |   99.49     99.50  
    beta                 |  99.18     99.27   |    0.27      0.29   |   99.48     99.54  
    empirical            |  99.13     99.30   |    0.49      0.30   |   99.45     99.55  
    empirical_cumulative |  99.28     99.30   |    0.31      0.30   |   99.54     99.56  
    val_cal              |  99.28     99.30   |    0.31      0.30   |   99.54     99.56  


## XDS-cicids-16b-Wa-500n34b  (5 flows × 2 phases, seeds: [8188, 25608, 41773, 63504, 82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.57% |   0.12% |  99.73% | r8188 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.57% |   0.12% |  99.73% | r8188 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.57% |   0.12% |  99.73% | r8188 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.57% |   0.12% |  99.73% | r8188 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.57% |   0.12% |  99.73% | r8188 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.57% |   0.12% |  99.73% | r8188 GA best_acc       val_cal
    Best FPR (any F1)        |  97.37% |   0.08% |  98.39% | r63504 GS best_fpr       beta
    Best FPR (F1>80%)        |  97.37% |   0.08% |  98.39% | r63504 GS best_fpr       beta
    Best Acc (any FPR)       |  99.57% |   0.12% |  99.73% | r8188 GA best_acc       val_cal

### best_fitness  (GS: 5 runs | GA: 5 runs)
    Grid Search : 175±96 neurons | 34±0 bits
    GA Neurons  : 211±98 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.55±0.01 | 0.21±0.04  0.12±0.01 |99.61±0.01 99.72±0.01
    fixed_05             |99.02±0.07 99.19±0.10 | 0.60±0.05  0.48±0.08 |99.38±0.04 99.48±0.06
    platt                |99.34±0.02 99.45±0.06 | 0.29±0.02  0.24±0.04 |99.58±0.02 99.65±0.04
    beta                 |99.35±0.04 99.52±0.06 | 0.24±0.01  0.16±0.06 |99.59±0.02 99.69±0.04
    empirical            |99.20±0.39 99.46±0.10 | 0.39±0.37  0.16±0.13 |99.49±0.25 99.66±0.06
    empirical_cumulative |99.38±0.01 99.55±0.01 | 0.21±0.04  0.12±0.01 |99.61±0.01 99.72±0.01
    val_cal              |99.38±0.01 99.55±0.01 | 0.23±0.03  0.12±0.01 |99.61±0.01 99.72±0.01

### best_f1  (GS: 5 runs | GA: 5 runs)
    Grid Search : 175±96 neurons | 34±0 bits
    GA Neurons  : 211±98 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.55±0.01 | 0.21±0.04  0.12±0.01 |99.61±0.01 99.72±0.01
    fixed_05             |99.02±0.07 99.19±0.10 | 0.60±0.05  0.48±0.08 |99.38±0.04 99.48±0.06
    platt                |99.34±0.02 99.45±0.06 | 0.29±0.02  0.24±0.04 |99.58±0.02 99.65±0.04
    beta                 |99.35±0.04 99.52±0.06 | 0.24±0.01  0.16±0.06 |99.59±0.02 99.69±0.04
    empirical            |99.20±0.39 99.46±0.10 | 0.39±0.37  0.16±0.13 |99.49±0.25 99.66±0.06
    empirical_cumulative |99.38±0.01 99.55±0.01 | 0.21±0.04  0.12±0.01 |99.61±0.01 99.72±0.01
    val_cal              |99.38±0.01 99.55±0.01 | 0.23±0.03  0.12±0.01 |99.61±0.01 99.72±0.01

### best_fpr  (GS: 5 runs | GA: 5 runs)
    Grid Search : 162±179 neurons | 28±5 bits
    GA Neurons  : 182±110 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |98.23±1.32 99.44±0.05 | 0.17±0.08  0.14±0.04 |98.91±0.80 99.65±0.03
    fixed_05             |96.96±2.34 99.10±0.13 | 2.23±1.87  0.55±0.10 |97.99±1.58 99.43±0.09
    platt                |98.13±1.40 99.38±0.05 | 0.82±0.65  0.29±0.04 |98.81±0.89 99.61±0.03
    beta                 |98.19±1.30 99.42±0.07 | 0.22±0.11  0.23±0.06 |98.89±0.78 99.63±0.04
    empirical            |98.20±1.30 99.34±0.08 | 0.30±0.21  0.32±0.12 |98.89±0.78 99.58±0.05
    empirical_cumulative |98.23±1.32 99.44±0.05 | 0.15±0.05  0.14±0.04 |98.91±0.79 99.65±0.03
    val_cal              |98.23±1.32 99.44±0.05 | 0.17±0.08  0.15±0.06 |98.91±0.80 99.65±0.03

### best_acc  (GS: 5 runs | GA: 5 runs)
    Grid Search : 175±96 neurons | 34±0 bits
    GA Neurons  : 211±98 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.55±0.01 | 0.21±0.04  0.12±0.01 |99.61±0.01 99.72±0.01
    fixed_05             |99.02±0.07 99.19±0.10 | 0.60±0.05  0.48±0.08 |99.38±0.04 99.48±0.06
    platt                |99.34±0.02 99.45±0.06 | 0.29±0.02  0.24±0.04 |99.58±0.02 99.65±0.04
    beta                 |99.35±0.04 99.52±0.06 | 0.24±0.01  0.16±0.06 |99.59±0.02 99.69±0.04
    empirical            |99.20±0.39 99.46±0.10 | 0.39±0.37  0.16±0.13 |99.49±0.25 99.66±0.06
    empirical_cumulative |99.38±0.01 99.55±0.01 | 0.21±0.04  0.12±0.01 |99.61±0.01 99.72±0.01
    val_cal              |99.38±0.01 99.55±0.01 | 0.23±0.03  0.12±0.01 |99.61±0.01 99.72±0.01

### best_ce  (GS: 5 runs | GA: 5 runs)
    Grid Search : 400±82 neurons | 34±1 bits
    GA Neurons  : 214±176 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.33±0.03 99.40±0.04 | 0.24±0.05  0.22±0.07 |99.57±0.02 99.62±0.03
    fixed_05             |99.02±0.04 99.12±0.07 | 0.60±0.03  0.54±0.06 |99.38±0.03 99.44±0.04
    platt                |99.30±0.02 99.29±0.07 | 0.31±0.02  0.37±0.06 |99.56±0.01 99.55±0.05
    beta                 |99.30±0.03 99.37±0.04 | 0.23±0.02  0.25±0.05 |99.56±0.02 99.60±0.03
    empirical            |98.91±0.55 99.18±0.29 | 0.64±0.53  0.42±0.30 |99.30±0.36 99.48±0.19
    empirical_cumulative |99.32±0.03 99.40±0.04 | 0.22±0.03  0.22±0.07 |99.57±0.02 99.62±0.03
    val_cal              |99.33±0.03 99.40±0.04 | 0.28±0.02  0.24±0.06 |99.57±0.02 99.62±0.03


## XDS-cicids-16b-Wb-500n34b  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.55% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.55% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.55% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.55% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.55% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.55% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  98.79% |   0.06% |  99.24% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  98.79% |   0.06% |  99.24% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.55% |   0.12% |  99.72% | r82096 GA best_acc       val_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 332±71 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.50±0.07 | 0.20±0.02  0.18±0.08 |99.61±0.01 99.68±0.05
    fixed_05             |99.02±0.01 99.16±0.09 | 0.60±0.01  0.49±0.07 |99.38±0.01 99.47±0.06
    platt                |99.33±0.02 99.42±0.00 | 0.30±0.01  0.26±0.00 |99.58±0.01 99.63±0.00
    beta                 |99.36±0.02 99.49±0.08 | 0.23±0.03  0.15±0.05 |99.60±0.01 99.68±0.05
    empirical            |98.94±0.62 99.16±0.31 | 0.62±0.60  0.40±0.43 |99.33±0.40 99.47±0.20
    empirical_cumulative |99.37±0.01 99.49±0.08 | 0.19±0.00  0.15±0.05 |99.60±0.00 99.68±0.05
    val_cal              |99.38±0.01 99.50±0.07 | 0.20±0.02  0.18±0.08 |99.61±0.01 99.68±0.05

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 332±71 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.50±0.07 | 0.20±0.02  0.18±0.08 |99.61±0.01 99.68±0.05
    fixed_05             |99.02±0.01 99.16±0.09 | 0.60±0.01  0.49±0.07 |99.38±0.01 99.47±0.06
    platt                |99.33±0.02 99.42±0.00 | 0.30±0.01  0.26±0.00 |99.58±0.01 99.63±0.00
    beta                 |99.36±0.02 99.49±0.08 | 0.23±0.03  0.15±0.05 |99.60±0.01 99.68±0.05
    empirical            |98.94±0.62 99.16±0.31 | 0.62±0.60  0.40±0.43 |99.33±0.40 99.47±0.20
    empirical_cumulative |99.37±0.01 99.49±0.08 | 0.19±0.00  0.15±0.05 |99.60±0.00 99.68±0.05
    val_cal              |99.38±0.01 99.50±0.07 | 0.20±0.02  0.18±0.08 |99.61±0.01 99.68±0.05

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 214±150 neurons | 30±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.09±0.20 99.15±0.18 | 0.44±0.19  0.31±0.06 |99.42±0.13 99.46±0.11
    fixed_05             |98.83±0.14 98.81±0.20 | 0.74±0.11  0.75±0.15 |99.25±0.09 99.24±0.13
    platt                |99.04±0.26 99.11±0.18 | 0.40±0.11  0.40±0.10 |99.39±0.17 99.44±0.12
    beta                 |99.02±0.15 99.13±0.18 | 0.29±0.05  0.32±0.10 |99.38±0.09 99.45±0.11
    empirical            |99.05±0.26 99.05±0.32 | 0.50±0.27  0.49±0.32 |99.39±0.17 99.40±0.21
    empirical_cumulative |99.03±0.17 98.95±0.23 | 0.11±0.04  0.07±0.02 |99.39±0.11 99.34±0.14
    val_cal              |99.09±0.20 99.15±0.18 | 0.27±0.05  0.31±0.06 |99.43±0.13 99.46±0.11

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 332±71 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.50±0.07 | 0.20±0.02  0.18±0.08 |99.61±0.01 99.68±0.05
    fixed_05             |99.02±0.01 99.16±0.09 | 0.60±0.01  0.49±0.07 |99.38±0.01 99.47±0.06
    platt                |99.33±0.02 99.42±0.00 | 0.30±0.01  0.26±0.00 |99.58±0.01 99.63±0.00
    beta                 |99.36±0.02 99.49±0.08 | 0.23±0.03  0.15±0.05 |99.60±0.01 99.68±0.05
    empirical            |98.94±0.62 99.16±0.31 | 0.62±0.60  0.40±0.43 |99.33±0.40 99.47±0.20
    empirical_cumulative |99.37±0.01 99.49±0.08 | 0.19±0.00  0.15±0.05 |99.60±0.00 99.68±0.05
    val_cal              |99.38±0.01 99.50±0.07 | 0.20±0.02  0.18±0.08 |99.61±0.01 99.68±0.05

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 322±42 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.35±0.00 99.40±0.03 | 0.25±0.02  0.20±0.01 |99.59±0.00 99.62±0.02
    fixed_05             |99.08±0.01 99.08±0.10 | 0.55±0.00  0.58±0.07 |99.41±0.00 99.41±0.06
    platt                |99.33±0.01 99.32±0.00 | 0.29±0.01  0.34±0.00 |99.58±0.01 99.57±0.00
    beta                 |99.33±0.02 99.38±0.03 | 0.23±0.04  0.25±0.00 |99.57±0.01 99.61±0.02
    empirical            |99.34±0.01 99.32±0.10 | 0.22±0.01  0.30±0.17 |99.58±0.01 99.57±0.06
    empirical_cumulative |99.31±0.01 99.36±0.09 | 0.18±0.00  0.15±0.07 |99.56±0.01 99.60±0.05
    val_cal              |99.35±0.00 99.40±0.03 | 0.25±0.02  0.21±0.02 |99.59±0.00 99.62±0.02


## XDS-cicids-16b-Wbu-500n34b  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.53% |   0.10% |  99.71% | r25608 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.53% |   0.10% |  99.71% | r25608 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.53% |   0.10% |  99.71% | r25608 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.53% |   0.10% |  99.71% | r25608 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.53% |   0.10% |  99.71% | r25608 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.53% |   0.10% |  99.71% | r25608 GA best_acc       train_cal
    Best FPR (any F1)        |  99.03% |   0.07% |  99.39% | r25608 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.03% |   0.07% |  99.39% | r25608 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.53% |   0.10% |  99.71% | r25608 GA best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 350±71 neurons | 33±1 bits
    GA Neurons  : 159±52 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.53±0.00 | 0.19±0.02  0.12±0.03 |99.61±0.00 99.70±0.00
    fixed_05             |99.04±0.02 99.11±0.05 | 0.59±0.02  0.54±0.04 |99.39±0.01 99.43±0.03
    platt                |99.33±0.03 99.42±0.03 | 0.30±0.01  0.26±0.02 |99.58±0.02 99.63±0.02
    beta                 |99.37±0.02 99.49±0.05 | 0.23±0.02  0.19±0.05 |99.60±0.01 99.68±0.03
    empirical            |99.09±0.50 99.29±0.31 | 0.50±0.48  0.35±0.31 |99.42±0.33 99.55±0.20
    empirical_cumulative |99.38±0.01 99.53±0.00 | 0.19±0.02  0.11±0.00 |99.61±0.00 99.70±0.00
    val_cal              |99.38±0.01 99.53±0.00 | 0.21±0.02  0.12±0.03 |99.61±0.01 99.70±0.00

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 159±52 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.53±0.00 | 0.19±0.02  0.12±0.03 |99.61±0.00 99.70±0.00
    fixed_05             |99.04±0.02 99.11±0.05 | 0.59±0.02  0.54±0.04 |99.39±0.01 99.43±0.03
    platt                |99.33±0.03 99.42±0.03 | 0.30±0.01  0.26±0.02 |99.58±0.02 99.63±0.02
    beta                 |99.37±0.02 99.49±0.05 | 0.23±0.02  0.19±0.05 |99.60±0.01 99.68±0.03
    empirical            |99.09±0.50 99.29±0.31 | 0.50±0.48  0.35±0.31 |99.42±0.33 99.55±0.20
    empirical_cumulative |99.38±0.01 99.53±0.00 | 0.19±0.02  0.11±0.00 |99.61±0.00 99.70±0.00
    val_cal              |99.38±0.01 99.53±0.00 | 0.21±0.02  0.12±0.03 |99.61±0.01 99.70±0.00

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±212 neurons | 26±3 bits
    GA Neurons  : 276±161 neurons | 27±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.08±0.14 99.08±0.12 | 0.37±0.18  0.26±0.09 |99.42±0.09 99.42±0.08
    fixed_05             |98.83±0.10 98.76±0.19 | 0.73±0.08  0.79±0.15 |99.25±0.07 99.21±0.12
    platt                |99.02±0.19 99.02±0.16 | 0.38±0.08  0.39±0.07 |99.38±0.12 99.38±0.10
    beta                 |99.03±0.11 99.01±0.12 | 0.28±0.04  0.29±0.03 |99.39±0.07 99.37±0.08
    empirical            |98.90±0.31 99.07±0.14 | 0.64±0.30  0.47±0.10 |99.30±0.20 99.41±0.09
    empirical_cumulative |99.02±0.12 99.00±0.09 | 0.10±0.03  0.12±0.08 |99.39±0.08 99.37±0.05
    val_cal              |99.08±0.14 99.08±0.12 | 0.26±0.05  0.26±0.09 |99.42±0.09 99.42±0.08

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 159±52 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.53±0.00 | 0.19±0.02  0.12±0.03 |99.61±0.00 99.70±0.00
    fixed_05             |99.04±0.02 99.11±0.05 | 0.59±0.02  0.54±0.04 |99.39±0.01 99.43±0.03
    platt                |99.33±0.03 99.42±0.03 | 0.30±0.01  0.26±0.02 |99.58±0.02 99.63±0.02
    beta                 |99.37±0.02 99.49±0.05 | 0.23±0.02  0.19±0.05 |99.60±0.01 99.68±0.03
    empirical            |99.09±0.50 99.29±0.31 | 0.50±0.48  0.35±0.31 |99.42±0.33 99.55±0.20
    empirical_cumulative |99.38±0.01 99.53±0.00 | 0.19±0.02  0.11±0.00 |99.61±0.00 99.70±0.00
    val_cal              |99.38±0.01 99.53±0.00 | 0.21±0.02  0.12±0.03 |99.61±0.01 99.70±0.00

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±200 neurons | 34±0 bits
    GA Neurons  : 231±55 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.34±0.01 99.43±0.06 | 0.25±0.04  0.14±0.08 |99.58±0.01 99.64±0.04
    fixed_05             |99.06±0.04 99.08±0.15 | 0.56±0.03  0.57±0.11 |99.41±0.03 99.42±0.10
    platt                |99.32±0.01 99.35±0.05 | 0.31±0.01  0.31±0.04 |99.57±0.01 99.59±0.03
    beta                 |99.32±0.02 99.39±0.03 | 0.23±0.02  0.23±0.03 |99.57±0.02 99.61±0.02
    empirical            |99.33±0.02 99.27±0.14 | 0.26±0.02  0.36±0.17 |99.58±0.01 99.54±0.09
    empirical_cumulative |99.32±0.03 99.43±0.06 | 0.20±0.02  0.14±0.08 |99.57±0.02 99.64±0.04
    val_cal              |99.34±0.01 99.43±0.06 | 0.25±0.04  0.14±0.08 |99.58±0.01 99.64±0.04


## XDS-cicids-16b-Wc-500n34b  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best FPR (any F1)        |  99.38% |   0.08% |  99.61% | r8188 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.38% |   0.08% |  99.61% | r8188 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 333±208 neurons | 34±0 bits
    GA Neurons  : 247±142 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.35±0.01 99.40±0.09 | 0.25±0.02  0.21±0.11 |99.59±0.00 99.62±0.06
    fixed_05             |99.07±0.01 99.07±0.04 | 0.56±0.00  0.58±0.03 |99.41±0.00 99.41±0.03
    platt                |99.32±0.01 99.27±0.10 | 0.30±0.01  0.39±0.08 |99.57±0.01 99.54±0.07
    beta                 |99.33±0.02 99.37±0.06 | 0.22±0.02  0.26±0.05 |99.58±0.01 99.60±0.04
    empirical            |98.98±0.61 99.32±0.15 | 0.58±0.57  0.26±0.20 |99.35±0.40 99.57±0.10
    empirical_cumulative |99.35±0.01 99.40±0.09 | 0.22±0.02  0.21±0.11 |99.59±0.00 99.62±0.06
    val_cal              |99.36±0.00 99.40±0.09 | 0.26±0.03  0.21±0.11 |99.59±0.00 99.62±0.06

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±141 neurons | 34±0 bits
    GA Neurons  : 169±108 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.54±0.01 | 0.21±0.02  0.11±0.01 |99.61±0.01 99.71±0.01
    fixed_05             |99.04±0.02 99.17±0.07 | 0.59±0.02  0.49±0.05 |99.39±0.01 99.48±0.05
    platt                |99.33±0.02 99.44±0.07 | 0.29±0.02  0.24±0.07 |99.58±0.01 99.65±0.04
    beta                 |99.37±0.02 99.52±0.02 | 0.23±0.02  0.15±0.03 |99.60±0.02 99.69±0.01
    empirical            |98.90±0.44 99.47±0.13 | 0.69±0.42  0.20±0.15 |99.30±0.28 99.67±0.08
    empirical_cumulative |99.38±0.01 99.54±0.01 | 0.21±0.02  0.11±0.01 |99.61±0.01 99.71±0.01
    val_cal              |99.38±0.01 99.54±0.01 | 0.21±0.02  0.11±0.01 |99.61±0.01 99.71±0.01

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±100 neurons | 26±7 bits
    GA Neurons  : 151±91 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.09±0.21 99.43±0.01 | 0.27±0.15  0.16±0.06 |99.43±0.13 99.64±0.01
    fixed_05             |98.77±0.25 99.08±0.07 | 0.77±0.18  0.58±0.06 |99.22±0.16 99.41±0.05
    platt                |99.04±0.24 99.38±0.03 | 0.37±0.05  0.28±0.03 |99.39±0.15 99.61±0.02
    beta                 |99.03±0.24 99.41±0.01 | 0.30±0.06  0.23±0.01 |99.39±0.15 99.63±0.00
    empirical            |99.01±0.28 99.36±0.10 | 0.51±0.23  0.27±0.14 |99.37±0.18 99.60±0.07
    empirical_cumulative |99.08±0.21 99.42±0.03 | 0.17±0.04  0.12±0.05 |99.42±0.13 99.63±0.02
    val_cal              |99.09±0.21 99.43±0.01 | 0.30±0.11  0.16±0.06 |99.43±0.13 99.64±0.01

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±141 neurons | 34±0 bits
    GA Neurons  : 169±108 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.54±0.01 | 0.21±0.02  0.11±0.01 |99.61±0.01 99.71±0.01
    fixed_05             |99.04±0.02 99.17±0.07 | 0.59±0.02  0.49±0.05 |99.39±0.01 99.48±0.05
    platt                |99.33±0.02 99.44±0.07 | 0.29±0.02  0.24±0.07 |99.58±0.01 99.65±0.04
    beta                 |99.37±0.02 99.52±0.02 | 0.23±0.02  0.15±0.03 |99.60±0.02 99.69±0.01
    empirical            |98.90±0.44 99.47±0.13 | 0.69±0.42  0.20±0.15 |99.30±0.28 99.67±0.08
    empirical_cumulative |99.38±0.01 99.54±0.01 | 0.21±0.02  0.11±0.01 |99.61±0.01 99.71±0.01
    val_cal              |99.38±0.01 99.54±0.01 | 0.21±0.02  0.11±0.01 |99.61±0.01 99.71±0.01

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 450±71 neurons | 34±0 bits
    GA Neurons  : 247±142 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.35±0.01 99.40±0.09 | 0.25±0.02  0.21±0.11 |99.59±0.00 99.62±0.06
    fixed_05             |99.07±0.01 99.07±0.04 | 0.56±0.00  0.58±0.03 |99.41±0.00 99.41±0.03
    platt                |99.32±0.01 99.27±0.10 | 0.30±0.01  0.39±0.08 |99.57±0.01 99.54±0.07
    beta                 |99.33±0.02 99.37±0.06 | 0.22±0.02  0.26±0.05 |99.58±0.01 99.60±0.04
    empirical            |98.98±0.61 99.32±0.15 | 0.58±0.57  0.26±0.20 |99.35±0.40 99.57±0.10
    empirical_cumulative |99.35±0.01 99.40±0.09 | 0.22±0.02  0.21±0.11 |99.59±0.00 99.62±0.06
    val_cal              |99.36±0.00 99.40±0.09 | 0.26±0.03  0.21±0.11 |99.59±0.00 99.62±0.06


## XDS-cicids-32b-Wa-500n34b  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.57% |   0.08% |  99.73% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.57% |   0.08% |  99.73% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.57% |   0.08% |  99.73% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.57% |   0.08% |  99.73% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.57% |   0.08% |  99.73% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.57% |   0.08% |  99.73% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  99.56% |   0.07% |  99.72% | r82096 GA best_acc       empirical
    Best FPR (F1>80%)        |  99.56% |   0.07% |  99.72% | r82096 GA best_acc       empirical
    Best Acc (any FPR)       |  99.57% |   0.08% |  99.73% | r82096 GA best_acc       val_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 162±83 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.27±0.00 99.51±0.08 | 0.25±0.03  0.15±0.10 |99.54±0.00 99.69±0.05
    fixed_05             |98.97±0.03 99.13±0.12 | 0.62±0.03  0.52±0.08 |99.34±0.02 99.45±0.08
    platt                |99.21±0.00 99.39±0.06 | 0.34±0.00  0.26±0.07 |99.50±0.00 99.62±0.04
    beta                 |99.25±0.00 99.42±0.02 | 0.27±0.00  0.21±0.01 |99.53±0.00 99.63±0.01
    empirical            |99.18±0.04 99.47±0.13 | 0.43±0.04  0.12±0.06 |99.48±0.03 99.66±0.08
    empirical_cumulative |99.27±0.00 99.51±0.09 | 0.25±0.03  0.15±0.10 |99.54±0.00 99.69±0.06
    val_cal              |99.27±0.00 99.51±0.08 | 0.25±0.03  0.15±0.10 |99.54±0.00 99.69±0.05

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 162±83 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.28±0.02 99.51±0.08 | 0.26±0.02  0.15±0.10 |99.55±0.01 99.69±0.05
    fixed_05             |98.96±0.02 99.13±0.12 | 0.64±0.00  0.52±0.08 |99.34±0.01 99.45±0.08
    platt                |99.25±0.05 99.39±0.06 | 0.32±0.01  0.26±0.07 |99.53±0.03 99.62±0.04
    beta                 |99.25±0.00 99.42±0.02 | 0.25±0.02  0.21±0.01 |99.53±0.00 99.63±0.01
    empirical            |99.22±0.10 99.47±0.13 | 0.39±0.09  0.12±0.06 |99.50±0.06 99.66±0.08
    empirical_cumulative |99.28±0.02 99.51±0.09 | 0.26±0.02  0.15±0.10 |99.55±0.01 99.69±0.06
    val_cal              |99.28±0.02 99.51±0.08 | 0.26±0.02  0.15±0.10 |99.55±0.01 99.69±0.05

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 162±86 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |92.55±9.47 99.40±0.06 | 0.18±0.03  0.23±0.02 |95.96±5.04 99.62±0.04
    fixed_05             |85.10±19.71 99.17±0.03 |14.74±20.04  0.50±0.04 |87.43±16.90 99.47±0.02
    platt                |90.93±11.69 99.40±0.07 | 4.73±6.19  0.27±0.06 |93.95±7.83 99.62±0.04
    beta                 |92.55±9.47 99.39±0.06 | 0.22±0.10  0.24±0.05 |95.96±5.04 99.61±0.04
    empirical            |92.46±9.35 99.37±0.02 | 0.33±0.25  0.21±0.05 |95.90±4.96 99.60±0.02
    empirical_cumulative |92.55±9.47 99.39±0.05 | 0.18±0.03  0.21±0.04 |95.96±5.04 99.62±0.03
    val_cal              |92.55±9.47 99.40±0.06 | 0.18±0.04  0.26±0.05 |95.96±5.04 99.62±0.04

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 162±83 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.28±0.02 99.51±0.08 | 0.26±0.02  0.15±0.10 |99.55±0.01 99.69±0.05
    fixed_05             |98.96±0.02 99.13±0.12 | 0.64±0.00  0.52±0.08 |99.34±0.01 99.45±0.08
    platt                |99.25±0.05 99.39±0.06 | 0.32±0.01  0.26±0.07 |99.53±0.03 99.62±0.04
    beta                 |99.25±0.00 99.42±0.02 | 0.25±0.02  0.21±0.01 |99.53±0.00 99.63±0.01
    empirical            |99.22±0.10 99.47±0.13 | 0.39±0.09  0.12±0.06 |99.50±0.06 99.66±0.08
    empirical_cumulative |99.28±0.02 99.51±0.09 | 0.26±0.02  0.15±0.10 |99.55±0.01 99.69±0.06
    val_cal              |99.28±0.02 99.51±0.08 | 0.26±0.02  0.15±0.10 |99.55±0.01 99.69±0.05

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 153±79 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.25±0.02 99.37±0.00 | 0.25±0.02  0.31±0.02 |99.53±0.01 99.60±0.00
    fixed_05             |98.95±0.06 99.25±0.15 | 0.64±0.05  0.44±0.13 |99.33±0.04 99.52±0.10
    platt                |99.21±0.01 99.33±0.02 | 0.34±0.01  0.32±0.03 |99.50±0.01 99.58±0.02
    beta                 |99.24±0.02 99.33±0.01 | 0.26±0.01  0.28±0.01 |99.52±0.01 99.58±0.01
    empirical            |99.18±0.04 99.33±0.01 | 0.42±0.03  0.24±0.01 |99.48±0.02 99.58±0.01
    empirical_cumulative |99.25±0.03 99.37±0.00 | 0.23±0.01  0.31±0.02 |99.52±0.02 99.60±0.00
    val_cal              |99.25±0.02 99.37±0.01 | 0.25±0.02  0.31±0.03 |99.53±0.01 99.60±0.00


## XDS-cicids-32b-Wb-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.40% |   0.19% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.40% |   0.19% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.40% |   0.19% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.40% |   0.19% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.40% |   0.19% |  99.62% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.40% |   0.19% |  99.62% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  98.82% |   0.05% |  99.26% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  98.82% |   0.05% |  99.26% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.40% |   0.19% |  99.62% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 109±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.25     99.40   |    0.27      0.19   |   99.53     99.62  
    fixed_05             |  98.90     99.02   |    0.67      0.60   |   99.30     99.37  
    platt                |  99.21     99.29   |    0.34      0.31   |   99.50     99.55  
    beta                 |  99.23     99.39   |    0.26      0.17   |   99.51     99.61  
    empirical            |  99.09     99.40   |    0.50      0.19   |   99.42     99.62  
    empirical_cumulative |  99.24     99.39   |    0.21      0.16   |   99.52     99.61  
    val_cal              |  99.25     99.40   |    0.27      0.19   |   99.53     99.62  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 109±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.25     99.40   |    0.27      0.19   |   99.53     99.62  
    fixed_05             |  98.90     99.02   |    0.67      0.60   |   99.30     99.37  
    platt                |  99.21     99.29   |    0.34      0.31   |   99.50     99.55  
    beta                 |  99.23     99.39   |    0.26      0.17   |   99.51     99.61  
    empirical            |  99.09     99.40   |    0.50      0.19   |   99.42     99.62  
    empirical_cumulative |  99.24     99.39   |    0.21      0.16   |   99.52     99.61  
    val_cal              |  99.25     99.40   |    0.27      0.19   |   99.53     99.62  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 110±0 neurons | 31±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.11     98.98   |    0.33      0.20   |   99.44     99.36  
    fixed_05             |  98.88     98.55   |    0.68      0.92   |   99.29     99.08  
    platt                |  99.07     98.91   |    0.37      0.36   |   99.41     99.31  
    beta                 |  99.09     98.93   |    0.30      0.26   |   99.43     99.33  
    empirical            |  98.97     98.88   |    0.57      0.60   |   99.35     99.29  
    empirical_cumulative |  98.95     98.82   |    0.11      0.05   |   99.34     99.26  
    val_cal              |  99.11     98.98   |    0.33      0.20   |   99.44     99.36  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 109±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.25     99.40   |    0.27      0.19   |   99.53     99.62  
    fixed_05             |  98.90     99.02   |    0.67      0.60   |   99.30     99.37  
    platt                |  99.21     99.29   |    0.34      0.31   |   99.50     99.55  
    beta                 |  99.23     99.39   |    0.26      0.17   |   99.51     99.61  
    empirical            |  99.09     99.40   |    0.50      0.19   |   99.42     99.62  
    empirical_cumulative |  99.24     99.39   |    0.21      0.16   |   99.52     99.61  
    val_cal              |  99.25     99.40   |    0.27      0.19   |   99.53     99.62  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 293±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.25     99.25   |    0.23      0.21   |   99.53     99.53  
    fixed_05             |  98.97     99.10   |    0.62      0.52   |   99.35     99.43  
    platt                |  99.16     99.16   |    0.36      0.38   |   99.47     99.47  
    beta                 |  99.23     99.19   |    0.23      0.30   |   99.52     99.49  
    empirical            |  99.17     99.16   |    0.44      0.45   |   99.47     99.46  
    empirical_cumulative |  99.23     99.24   |    0.21      0.20   |   99.51     99.52  
    val_cal              |  99.25     99.25   |    0.23      0.21   |   99.53     99.53  


## XDS-cicids-32b-Wbu-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.54% |   0.17% |  99.71% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.54% |   0.17% |  99.71% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.54% |   0.17% |  99.71% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.54% |   0.17% |  99.71% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.54% |   0.17% |  99.71% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.54% |   0.17% |  99.71% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  99.26% |   0.07% |  99.54% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.26% |   0.07% |  99.54% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.54% |   0.17% |  99.71% | r82096 GA best_acc       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 95±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    fixed_05             |  99.00     99.17   |    0.63      0.50   |   99.36     99.47  
    platt                |  99.29     99.46   |    0.33      0.24   |   99.55     99.66  
    beta                 |  99.32     99.50   |    0.27      0.16   |   99.57     99.69  
    empirical            |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    empirical_cumulative |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    val_cal              |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 95±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    fixed_05             |  99.00     99.17   |    0.63      0.50   |   99.36     99.47  
    platt                |  99.29     99.46   |    0.33      0.24   |   99.55     99.66  
    beta                 |  99.32     99.50   |    0.27      0.16   |   99.57     99.69  
    empirical            |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    empirical_cumulative |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    val_cal              |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 118±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  98.98     99.27   |    0.42      0.12   |   99.35     99.54  
    fixed_05             |  98.49     98.92   |    0.98      0.69   |   99.04     99.31  
    platt                |  98.98     99.29   |    0.42      0.33   |   99.35     99.55  
    beta                 |  98.98     99.25   |    0.34      0.29   |   99.36     99.52  
    empirical            |  98.89     99.30   |    0.56      0.32   |   99.30     99.55  
    empirical_cumulative |  98.93     99.26   |    0.15      0.07   |   99.33     99.54  
    val_cal              |  98.98     99.30   |    0.42      0.32   |   99.36     99.55  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 95±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    fixed_05             |  99.00     99.17   |    0.63      0.50   |   99.36     99.47  
    platt                |  99.29     99.46   |    0.33      0.24   |   99.55     99.66  
    beta                 |  99.32     99.50   |    0.27      0.16   |   99.57     99.69  
    empirical            |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    empirical_cumulative |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  
    val_cal              |  99.33     99.54   |    0.28      0.17   |   99.58     99.71  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 97±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.24     99.45   |    0.27      0.19   |   99.52     99.65  
    fixed_05             |  98.90     99.09   |    0.67      0.58   |   99.30     99.42  
    platt                |  99.20     99.29   |    0.35      0.38   |   99.49     99.55  
    beta                 |  99.22     99.38   |    0.26      0.28   |   99.51     99.60  
    empirical            |  99.15     99.44   |    0.45      0.19   |   99.46     99.64  
    empirical_cumulative |  99.22     99.44   |    0.22      0.15   |   99.51     99.65  
    val_cal              |  99.24     99.45   |    0.27      0.19   |   99.52     99.65  


## XDS-cicids-32b-Wc-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.41% |   0.14% |  99.63% | r82096 GA best_acc       empirical
    Best FPR (F1>80%)        |  99.41% |   0.14% |  99.63% | r82096 GA best_acc       empirical
    Best Acc (any FPR)       |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 242±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.32   |    0.21      0.33   |   99.54     99.57  
    fixed_05             |  98.88     99.07   |    0.69      0.57   |   99.29     99.41  
    platt                |  99.20     99.26   |    0.35      0.39   |   99.49     99.53  
    beta                 |  99.25     99.28   |    0.25      0.31   |   99.53     99.54  
    empirical            |  99.03     99.30   |    0.56      0.31   |   99.38     99.55  
    empirical_cumulative |  99.27     99.32   |    0.21      0.33   |   99.54     99.57  
    val_cal              |  99.27     99.32   |    0.21      0.32   |   99.54     99.57  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 421±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.47   |    0.27      0.18   |   99.54     99.67  
    fixed_05             |  98.95     99.13   |    0.64      0.52   |   99.33     99.45  
    platt                |  99.21     99.37   |    0.33      0.29   |   99.50     99.60  
    beta                 |  99.25     99.47   |    0.26      0.19   |   99.53     99.66  
    empirical            |  99.15     99.41   |    0.46      0.14   |   99.46     99.63  
    empirical_cumulative |  99.27     99.47   |    0.27      0.18   |   99.54     99.67  
    val_cal              |  99.27     99.47   |    0.27      0.18   |   99.54     99.67  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 8±0 bits
    GA Neurons  : 420±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  85.85     99.36   |    0.15      0.24   |   92.39     99.59  
    fixed_05             |  71.16     99.19   |   28.91      0.47   |   75.48     99.49  
    platt                |  82.66     99.35   |    9.11      0.31   |   88.41     99.59  
    beta                 |  85.85     99.35   |    0.15      0.27   |   92.39     99.59  
    empirical            |  85.85     99.36   |    0.15      0.27   |   92.39     99.59  
    empirical_cumulative |  85.85     99.34   |    0.15      0.14   |   92.39     99.58  
    val_cal              |  85.85     99.36   |    0.15      0.25   |   92.39     99.60  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 421±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.47   |    0.27      0.18   |   99.54     99.67  
    fixed_05             |  98.95     99.13   |    0.64      0.52   |   99.33     99.45  
    platt                |  99.21     99.37   |    0.33      0.29   |   99.50     99.60  
    beta                 |  99.25     99.47   |    0.26      0.19   |   99.53     99.66  
    empirical            |  99.15     99.41   |    0.46      0.14   |   99.46     99.63  
    empirical_cumulative |  99.27     99.47   |    0.27      0.18   |   99.54     99.67  
    val_cal              |  99.27     99.47   |    0.27      0.18   |   99.54     99.67  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 242±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.32   |    0.21      0.33   |   99.54     99.57  
    fixed_05             |  98.88     99.07   |    0.69      0.57   |   99.29     99.41  
    platt                |  99.20     99.26   |    0.35      0.39   |   99.49     99.53  
    beta                 |  99.25     99.28   |    0.25      0.31   |   99.53     99.54  
    empirical            |  99.03     99.30   |    0.56      0.31   |   99.38     99.55  
    empirical_cumulative |  99.27     99.32   |    0.21      0.33   |   99.54     99.57  
    val_cal              |  99.27     99.32   |    0.21      0.32   |   99.54     99.57  


## XDS-cicids-64b-Wa-500n34b  (5 flows × 2 phases, seeds: [8188, 25608, 41773, 63504, 82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.61% |   0.07% |  99.75% | r8188 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.61% |   0.07% |  99.75% | r8188 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.61% |   0.07% |  99.75% | r8188 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.61% |   0.07% |  99.75% | r8188 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.61% |   0.07% |  99.75% | r8188 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.61% |   0.07% |  99.75% | r8188 GA best_acc       train_cal
    Best FPR (any F1)        |  99.54% |   0.06% |  99.71% | r8188 GA best_fpr       empirical
    Best FPR (F1>80%)        |  99.54% |   0.06% |  99.71% | r8188 GA best_fpr       empirical
    Best Acc (any FPR)       |  99.61% |   0.07% |  99.75% | r8188 GA best_acc       train_cal

### best_fitness  (GS: 5 runs | GA: 5 runs)
    Grid Search : 180±110 neurons | 34±0 bits
    GA Neurons  : 192±135 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.03 99.56±0.04 | 0.31±0.02  0.12±0.03 |99.57±0.02 99.72±0.02
    fixed_05             |99.07±0.06 99.18±0.10 | 0.56±0.04  0.49±0.08 |99.41±0.04 99.48±0.06
    platt                |99.29±0.05 99.46±0.07 | 0.32±0.01  0.23±0.04 |99.55±0.03 99.66±0.04
    beta                 |99.28±0.02 99.52±0.04 | 0.28±0.01  0.16±0.02 |99.54±0.01 99.69±0.03
    empirical            |99.29±0.04 99.52±0.07 | 0.33±0.03  0.13±0.07 |99.55±0.02 99.70±0.04
    empirical_cumulative |99.31±0.03 99.56±0.04 | 0.31±0.02  0.11±0.03 |99.57±0.02 99.72±0.02
    val_cal              |99.31±0.03 99.56±0.04 | 0.31±0.02  0.12±0.03 |99.57±0.02 99.72±0.02

### best_f1  (GS: 5 runs | GA: 5 runs)
    Grid Search : 180±110 neurons | 34±0 bits
    GA Neurons  : 192±135 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.03 99.56±0.04 | 0.31±0.02  0.12±0.03 |99.57±0.02 99.72±0.02
    fixed_05             |99.07±0.06 99.18±0.10 | 0.56±0.04  0.49±0.08 |99.41±0.04 99.48±0.06
    platt                |99.29±0.05 99.46±0.07 | 0.32±0.01  0.23±0.04 |99.55±0.03 99.66±0.04
    beta                 |99.28±0.02 99.52±0.04 | 0.28±0.01  0.16±0.02 |99.54±0.01 99.69±0.03
    empirical            |99.29±0.04 99.52±0.07 | 0.33±0.03  0.13±0.07 |99.55±0.02 99.70±0.04
    empirical_cumulative |99.31±0.03 99.56±0.04 | 0.31±0.02  0.11±0.03 |99.57±0.02 99.72±0.02
    val_cal              |99.31±0.03 99.56±0.04 | 0.31±0.02  0.12±0.03 |99.57±0.02 99.72±0.02

### best_fpr  (GS: 5 runs | GA: 5 runs)
    Grid Search : 182±203 neurons | 27±7 bits
    GA Neurons  : 195±141 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |97.59±3.22 99.48±0.07 | 0.18±0.03  0.16±0.09 |98.57±1.83 99.67±0.05
    fixed_05             |96.19±5.63 99.15±0.12 | 2.71±4.27  0.52±0.10 |97.37±4.03 99.46±0.08
    platt                |97.36±3.66 99.41±0.05 | 0.81±1.12  0.27±0.03 |98.36±2.25 99.63±0.03
    beta                 |97.57±3.21 99.42±0.06 | 0.23±0.06  0.22±0.04 |98.56±1.83 99.63±0.04
    empirical            |97.57±3.20 99.48±0.08 | 0.28±0.11  0.16±0.08 |98.56±1.82 99.67±0.05
    empirical_cumulative |97.59±3.22 99.47±0.08 | 0.18±0.03  0.09±0.03 |98.57±1.83 99.67±0.05
    val_cal              |97.59±3.22 99.48±0.07 | 0.21±0.08  0.16±0.09 |98.57±1.83 99.67±0.05

### best_acc  (GS: 5 runs | GA: 5 runs)
    Grid Search : 180±110 neurons | 34±0 bits
    GA Neurons  : 192±135 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.03 99.56±0.04 | 0.31±0.02  0.12±0.03 |99.57±0.02 99.72±0.02
    fixed_05             |99.07±0.06 99.18±0.10 | 0.56±0.04  0.49±0.08 |99.41±0.04 99.48±0.06
    platt                |99.29±0.05 99.46±0.07 | 0.32±0.01  0.23±0.04 |99.55±0.03 99.66±0.04
    beta                 |99.28±0.02 99.52±0.04 | 0.28±0.01  0.16±0.02 |99.54±0.01 99.69±0.03
    empirical            |99.29±0.04 99.52±0.07 | 0.33±0.03  0.13±0.07 |99.55±0.02 99.70±0.04
    empirical_cumulative |99.31±0.03 99.56±0.04 | 0.31±0.02  0.11±0.03 |99.57±0.02 99.72±0.02
    val_cal              |99.31±0.03 99.56±0.04 | 0.31±0.02  0.12±0.03 |99.57±0.02 99.72±0.02

### best_ce  (GS: 5 runs | GA: 5 runs)
    Grid Search : 220±130 neurons | 34±0 bits
    GA Neurons  : 172±122 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.29±0.04 99.34±0.05 | 0.30±0.03  0.19±0.07 |99.55±0.02 99.59±0.03
    fixed_05             |99.06±0.06 99.10±0.07 | 0.57±0.04  0.56±0.05 |99.40±0.04 99.43±0.04
    platt                |99.27±0.05 99.27±0.08 | 0.32±0.01  0.37±0.07 |99.54±0.03 99.53±0.05
    beta                 |99.26±0.04 99.31±0.02 | 0.29±0.01  0.24±0.04 |99.53±0.02 99.56±0.01
    empirical            |99.29±0.03 99.33±0.05 | 0.33±0.02  0.27±0.06 |99.55±0.02 99.58±0.03
    empirical_cumulative |99.29±0.04 99.34±0.05 | 0.29±0.05  0.19±0.07 |99.55±0.03 99.59±0.03
    val_cal              |99.30±0.04 99.35±0.05 | 0.32±0.04  0.23±0.06 |99.55±0.02 99.59±0.03


## XDS-cicids-64b-Wb-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.40% |   0.25% |  99.62% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.40% |   0.25% |  99.62% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.40% |   0.25% |  99.62% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.40% |   0.25% |  99.62% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.40% |   0.25% |  99.62% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.40% |   0.25% |  99.62% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.14% |   0.05% |  99.46% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.14% |   0.05% |  99.46% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.40% |   0.25% |  99.62% | r82096 GA best_f1        train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 195±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.33   |    0.29      0.31   |   99.57     99.58  
    fixed_05             |  99.03     99.05   |    0.60      0.60   |   99.38     99.39  
    platt                |  99.30     99.33   |    0.32      0.33   |   99.56     99.58  
    beta                 |  99.32     99.33   |    0.24      0.31   |   99.57     99.58  
    empirical            |  99.31     99.33   |    0.23      0.31   |   99.57     99.58  
    empirical_cumulative |  99.32     99.14   |    0.23      0.05   |   99.57     99.46  
    val_cal              |  99.33     99.34   |    0.29      0.33   |   99.57     99.58  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 290±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.40   |    0.29      0.25   |   99.57     99.62  
    fixed_05             |  99.03     99.00   |    0.60      0.64   |   99.38     99.36  
    platt                |  99.30     99.39   |    0.32      0.25   |   99.56     99.62  
    beta                 |  99.32     99.37   |    0.24      0.24   |   99.57     99.60  
    empirical            |  99.31     99.30   |    0.23      0.23   |   99.57     99.56  
    empirical_cumulative |  99.32     99.40   |    0.23      0.25   |   99.57     99.62  
    val_cal              |  99.33     99.40   |    0.29      0.25   |   99.57     99.62  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 20±0 bits
    GA Neurons  : 195±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  98.78     99.33   |    0.43      0.31   |   99.23     99.58  
    fixed_05             |  98.07     99.05   |    1.20      0.60   |   98.77     99.39  
    platt                |  98.78     99.33   |    0.43      0.33   |   99.23     99.58  
    beta                 |  98.78     99.33   |    0.35      0.31   |   99.23     99.58  
    empirical            |  98.73     99.33   |    0.53      0.31   |   99.20     99.58  
    empirical_cumulative |  98.73     99.14   |    0.14      0.05   |   99.20     99.46  
    val_cal              |  98.78     99.34   |    0.43      0.33   |   99.23     99.58  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 290±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.40   |    0.29      0.25   |   99.57     99.62  
    fixed_05             |  99.03     99.00   |    0.60      0.64   |   99.38     99.36  
    platt                |  99.30     99.39   |    0.32      0.25   |   99.56     99.62  
    beta                 |  99.32     99.37   |    0.24      0.24   |   99.57     99.60  
    empirical            |  99.31     99.30   |    0.23      0.23   |   99.57     99.56  
    empirical_cumulative |  99.32     99.40   |    0.23      0.25   |   99.57     99.62  
    val_cal              |  99.33     99.40   |    0.29      0.25   |   99.57     99.62  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 189±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.38   |    0.33      0.29   |   99.55     99.61  
    fixed_05             |  99.02     99.02   |    0.60      0.63   |   99.38     99.38  
    platt                |  99.26     99.36   |    0.31      0.32   |   99.53     99.59  
    beta                 |  99.24     99.37   |    0.29      0.29   |   99.52     99.60  
    empirical            |  99.30     99.33   |    0.33      0.25   |   99.55     99.58  
    empirical_cumulative |  99.23     99.33   |    0.18      0.25   |   99.51     99.58  
    val_cal              |  99.30     99.38   |    0.33      0.29   |   99.55     99.61  


## XDS-cicids-64b-Wbu-500n34b  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.55% |   0.06% |  99.72% | r82096 GA best_fpr       val_cal
    Best FPR (F1>80%)        |  99.55% |   0.06% |  99.72% | r82096 GA best_fpr       val_cal
    Best Acc (any FPR)       |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 244±102 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.32±0.01 99.50±0.13 | 0.28±0.03  0.16±0.12 |99.57±0.00 99.68±0.08
    fixed_05             |99.03±0.02 99.09±0.04 | 0.58±0.02  0.57±0.04 |99.39±0.01 99.42±0.02
    platt                |99.31±0.00 99.41±0.01 | 0.31±0.00  0.25±0.01 |99.56±0.00 99.63±0.01
    beta                 |99.26±0.01 99.43±0.08 | 0.26±0.02  0.22±0.03 |99.53±0.00 99.64±0.05
    empirical            |99.31±0.01 99.44±0.19 | 0.29±0.00  0.13±0.08 |99.56±0.00 99.65±0.12
    empirical_cumulative |99.28±0.07 99.50±0.13 | 0.23±0.03  0.16±0.12 |99.54±0.04 99.68±0.08
    val_cal              |99.32±0.01 99.50±0.13 | 0.28±0.03  0.16±0.12 |99.57±0.00 99.68±0.08

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 244±102 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.32±0.01 99.50±0.13 | 0.28±0.03  0.16±0.12 |99.57±0.00 99.68±0.08
    fixed_05             |99.03±0.02 99.09±0.04 | 0.58±0.02  0.57±0.04 |99.39±0.01 99.42±0.02
    platt                |99.31±0.00 99.41±0.01 | 0.31±0.00  0.25±0.01 |99.56±0.00 99.63±0.01
    beta                 |99.26±0.01 99.43±0.08 | 0.26±0.02  0.22±0.03 |99.53±0.00 99.64±0.05
    empirical            |99.31±0.01 99.44±0.19 | 0.29±0.00  0.13±0.08 |99.56±0.00 99.65±0.12
    empirical_cumulative |99.28±0.07 99.50±0.13 | 0.23±0.03  0.16±0.12 |99.54±0.04 99.68±0.08
    val_cal              |99.32±0.01 99.50±0.13 | 0.28±0.03  0.16±0.12 |99.57±0.00 99.68±0.08

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±283 neurons | 29±7 bits
    GA Neurons  : 338±229 neurons | 29±7 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.13±0.16 99.31±0.34 | 0.32±0.13  0.21±0.20 |99.45±0.10 99.56±0.22
    fixed_05             |98.88±0.13 98.88±0.29 | 0.65±0.04  0.68±0.15 |99.29±0.08 99.29±0.19
    platt                |99.07±0.13 99.20±0.23 | 0.38±0.02  0.36±0.06 |99.41±0.08 99.49±0.15
    beta                 |99.07±0.17 99.21±0.27 | 0.32±0.02  0.28±0.03 |99.42±0.10 99.50±0.17
    empirical            |98.85±0.48 99.11±0.61 | 0.66±0.38  0.46±0.50 |99.27±0.31 99.43±0.39
    empirical_cumulative |99.07±0.23 99.23±0.46 | 0.14±0.07  0.07±0.01 |99.41±0.14 99.52±0.29
    val_cal              |99.13±0.16 99.31±0.34 | 0.32±0.13  0.21±0.20 |99.45±0.10 99.56±0.22

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 244±102 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.32±0.01 99.50±0.13 | 0.28±0.03  0.16±0.12 |99.57±0.00 99.68±0.08
    fixed_05             |99.03±0.02 99.09±0.04 | 0.58±0.02  0.57±0.04 |99.39±0.01 99.42±0.02
    platt                |99.31±0.00 99.41±0.01 | 0.31±0.00  0.25±0.01 |99.56±0.00 99.63±0.01
    beta                 |99.26±0.01 99.43±0.08 | 0.26±0.02  0.22±0.03 |99.53±0.00 99.64±0.05
    empirical            |99.31±0.01 99.44±0.19 | 0.29±0.00  0.13±0.08 |99.56±0.00 99.65±0.12
    empirical_cumulative |99.28±0.07 99.50±0.13 | 0.23±0.03  0.16±0.12 |99.54±0.04 99.68±0.08
    val_cal              |99.32±0.01 99.50±0.13 | 0.28±0.03  0.16±0.12 |99.57±0.00 99.68±0.08

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 141±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.29±0.03 99.27±0.00 | 0.34±0.01  0.31±0.03 |99.55±0.02 99.54±0.00
    fixed_05             |99.02±0.04 99.10±0.12 | 0.61±0.03  0.54±0.10 |99.37±0.03 99.43±0.08
    platt                |99.28±0.03 99.24±0.01 | 0.32±0.02  0.38±0.04 |99.54±0.02 99.52±0.00
    beta                 |99.25±0.04 99.23±0.02 | 0.29±0.01  0.31±0.07 |99.52±0.03 99.51±0.01
    empirical            |99.28±0.04 99.24±0.03 | 0.33±0.00  0.37±0.07 |99.54±0.03 99.52±0.02
    empirical_cumulative |99.26±0.05 99.19±0.07 | 0.25±0.07  0.20±0.10 |99.54±0.03 99.49±0.04
    val_cal              |99.29±0.03 99.27±0.00 | 0.34±0.01  0.31±0.03 |99.55±0.02 99.54±0.00


## XDS-cicids-64b-Wc-500n34b  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.54% |   0.10% |  99.71% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.54% |   0.10% |  99.71% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.54% |   0.10% |  99.71% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.54% |   0.10% |  99.71% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.54% |   0.10% |  99.71% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.54% |   0.10% |  99.71% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.45% |   0.07% |  99.66% | r82096 GA best_fpr       train_cal
    Best FPR (F1>80%)        |  99.45% |   0.07% |  99.66% | r82096 GA best_fpr       train_cal
    Best Acc (any FPR)       |  99.54% |   0.10% |  99.71% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±212 neurons | 34±0 bits
    GA Neurons  : 282±132 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.28±0.00 99.33±0.01 | 0.34±0.01  0.28±0.00 |99.54±0.00 99.58±0.00
    fixed_05             |99.04±0.03 99.08±0.03 | 0.58±0.02  0.57±0.02 |99.39±0.02 99.42±0.02
    platt                |99.27±0.02 99.28±0.03 | 0.32±0.02  0.37±0.04 |99.54±0.01 99.54±0.02
    beta                 |99.23±0.00 99.28±0.06 | 0.28±0.01  0.32±0.08 |99.51±0.00 99.54±0.04
    empirical            |99.27±0.02 99.28±0.02 | 0.33±0.01  0.25±0.04 |99.54±0.01 99.55±0.01
    empirical_cumulative |99.28±0.00 99.33±0.01 | 0.34±0.01  0.28±0.00 |99.54±0.00 99.58±0.00
    val_cal              |99.28±0.00 99.33±0.01 | 0.34±0.01  0.28±0.00 |99.54±0.00 99.58±0.00

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±141 neurons | 34±0 bits
    GA Neurons  : 156±65 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.32±0.01 99.51±0.05 | 0.29±0.02  0.14±0.06 |99.57±0.00 99.69±0.03
    fixed_05             |99.06±0.01 99.07±0.03 | 0.57±0.01  0.57±0.04 |99.40±0.01 99.41±0.02
    platt                |99.30±0.02 99.38±0.01 | 0.32±0.02  0.28±0.00 |99.56±0.01 99.61±0.01
    beta                 |99.25±0.01 99.45±0.07 | 0.26±0.03  0.20±0.09 |99.53±0.01 99.65±0.05
    empirical            |99.30±0.03 99.48±0.07 | 0.29±0.00  0.13±0.06 |99.56±0.02 99.67±0.05
    empirical_cumulative |99.32±0.01 99.51±0.05 | 0.29±0.02  0.14±0.06 |99.57±0.00 99.69±0.03
    val_cal              |99.32±0.01 99.51±0.05 | 0.30±0.01  0.14±0.06 |99.57±0.01 99.69±0.03

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 5±0 neurons | 24±0 bits
    GA Neurons  : 162±68 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±2.31 99.40±0.07 | 1.39±1.77  0.16±0.13 |96.09±1.09 99.62±0.04
    fixed_05             |89.29±4.47 99.06±0.04 | 8.11±3.15  0.58±0.04 |92.50±3.32 99.40±0.03
    platt                |93.00±3.03 99.31±0.06 | 2.72±0.11  0.33±0.08 |95.60±1.78 99.56±0.04
    beta                 |93.39±2.14 99.36±0.12 | 0.18±0.06  0.27±0.12 |96.13±1.15 99.59±0.08
    empirical            |93.39±2.14 99.17±0.14 | 0.18±0.06  0.47±0.16 |96.13±1.15 99.47±0.09
    empirical_cumulative |93.39±2.14 99.40±0.07 | 0.18±0.06  0.16±0.13 |96.13±1.15 99.62±0.04
    val_cal              |93.51±2.31 99.40±0.07 | 1.39±1.77  0.16±0.13 |96.09±1.09 99.62±0.04

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±141 neurons | 34±0 bits
    GA Neurons  : 156±65 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.32±0.01 99.51±0.05 | 0.29±0.02  0.14±0.06 |99.57±0.00 99.69±0.03
    fixed_05             |99.06±0.01 99.07±0.03 | 0.57±0.01  0.57±0.04 |99.40±0.01 99.41±0.02
    platt                |99.30±0.02 99.38±0.01 | 0.32±0.02  0.28±0.00 |99.56±0.01 99.61±0.01
    beta                 |99.25±0.01 99.45±0.07 | 0.26±0.03  0.20±0.09 |99.53±0.01 99.65±0.05
    empirical            |99.30±0.03 99.48±0.07 | 0.29±0.00  0.13±0.06 |99.56±0.02 99.67±0.05
    empirical_cumulative |99.32±0.01 99.51±0.05 | 0.29±0.02  0.14±0.06 |99.57±0.00 99.69±0.03
    val_cal              |99.32±0.01 99.51±0.05 | 0.30±0.01  0.14±0.06 |99.57±0.01 99.69±0.03

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±212 neurons | 34±0 bits
    GA Neurons  : 282±132 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.28±0.00 99.33±0.01 | 0.34±0.01  0.28±0.00 |99.54±0.00 99.58±0.00
    fixed_05             |99.04±0.03 99.08±0.03 | 0.58±0.02  0.57±0.02 |99.39±0.02 99.42±0.02
    platt                |99.27±0.02 99.28±0.03 | 0.32±0.02  0.37±0.04 |99.54±0.01 99.54±0.02
    beta                 |99.23±0.00 99.28±0.06 | 0.28±0.01  0.32±0.08 |99.51±0.00 99.54±0.04
    empirical            |99.27±0.02 99.28±0.02 | 0.33±0.01  0.25±0.04 |99.54±0.01 99.55±0.01
    empirical_cumulative |99.28±0.00 99.33±0.01 | 0.34±0.01  0.28±0.00 |99.54±0.00 99.58±0.00
    val_cal              |99.28±0.00 99.33±0.01 | 0.34±0.01  0.28±0.00 |99.54±0.00 99.58±0.00


## XDS-cicids-96b-Wa-250n100b  (1 flows × 2 phases, seeds: [25608])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.58% |   0.19% |  99.73% | r25608 GA best_acc       empirical_cumulative
    Best F1 (FPR<14%)        |  99.58% |   0.19% |  99.73% | r25608 GA best_acc       empirical_cumulative
    Best F1 (FPR<10%)        |  99.58% |   0.19% |  99.73% | r25608 GA best_acc       empirical_cumulative
    Best F1 (FPR<6%)         |  99.58% |   0.19% |  99.73% | r25608 GA best_acc       empirical_cumulative
    Best F1 (FPR<5%)         |  99.58% |   0.19% |  99.73% | r25608 GA best_acc       empirical_cumulative
    Best F1 (FPR<4%)         |  99.58% |   0.19% |  99.73% | r25608 GA best_acc       empirical_cumulative
    Best FPR (any F1)        |  99.50% |   0.16% |  99.68% | r25608 GA best_acc       empirical
    Best FPR (F1>80%)        |  99.50% |   0.16% |  99.68% | r25608 GA best_acc       empirical
    Best Acc (any FPR)       |  99.58% |   0.19% |  99.73% | r25608 GA best_acc       empirical_cumulative

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 100±0 bits
    GA Neurons  : 134±0 neurons | 90±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.46     99.58   |    0.27      0.20   |   99.66     99.73  
    fixed_05             |  99.45     99.54   |    0.30      0.23   |   99.65     99.71  
    platt                |  99.46     99.57   |    0.27      0.19   |   99.66     99.73  
    beta                 |  99.45     99.54   |    0.26      0.18   |   99.65     99.71  
    empirical            |  99.44     99.50   |    0.26      0.16   |   99.65     99.68  
    empirical_cumulative |  99.46     99.58   |    0.27      0.19   |   99.66     99.73  
    val_cal              |  99.46     99.58   |    0.27      0.19   |   99.66     99.73  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 100±0 bits
    GA Neurons  : 134±0 neurons | 90±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.46     99.58   |    0.27      0.20   |   99.66     99.73  
    fixed_05             |  99.45     99.54   |    0.30      0.23   |   99.65     99.71  
    platt                |  99.46     99.57   |    0.27      0.19   |   99.66     99.73  
    beta                 |  99.45     99.54   |    0.26      0.18   |   99.65     99.71  
    empirical            |  99.44     99.50   |    0.26      0.16   |   99.65     99.68  
    empirical_cumulative |  99.46     99.58   |    0.27      0.19   |   99.66     99.73  
    val_cal              |  99.46     99.58   |    0.27      0.19   |   99.66     99.73  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 64±0 bits
    GA Neurons  : 132±0 neurons | 86±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.29     99.57   |    0.16      0.20   |   99.55     99.73  
    fixed_05             |  98.43     99.55   |    1.11      0.23   |   98.99     99.71  
    platt                |  99.01     99.57   |    0.55      0.20   |   99.37     99.73  
    beta                 |  99.21     99.54   |    0.16      0.18   |   99.50     99.71  
    empirical            |  99.29     99.50   |    0.16      0.16   |   99.55     99.69  
    empirical_cumulative |  99.29     99.57   |    0.16      0.20   |   99.55     99.73  
    val_cal              |  99.29     99.57   |    0.16      0.20   |   99.55     99.73  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 100±0 bits
    GA Neurons  : 134±0 neurons | 90±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.46     99.58   |    0.27      0.20   |   99.66     99.73  
    fixed_05             |  99.45     99.54   |    0.30      0.23   |   99.65     99.71  
    platt                |  99.46     99.57   |    0.27      0.19   |   99.66     99.73  
    beta                 |  99.45     99.54   |    0.26      0.18   |   99.65     99.71  
    empirical            |  99.44     99.50   |    0.26      0.16   |   99.65     99.68  
    empirical_cumulative |  99.46     99.58   |    0.27      0.19   |   99.66     99.73  
    val_cal              |  99.46     99.58   |    0.27      0.19   |   99.66     99.73  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 80±0 bits
    GA Neurons  : 243±0 neurons | 81±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.45     99.47   |    0.29      0.26   |   99.65     99.66  
    fixed_05             |  99.44     99.46   |    0.30      0.30   |   99.65     99.66  
    platt                |  99.44     99.46   |    0.28      0.27   |   99.65     99.66  
    beta                 |  99.43     99.45   |    0.27      0.25   |   99.64     99.65  
    empirical            |  99.33     99.41   |    0.19      0.22   |   99.58     99.63  
    empirical_cumulative |  99.45     99.46   |    0.29      0.23   |   99.65     99.66  
    val_cal              |  99.45     99.47   |    0.29      0.26   |   99.65     99.66  


## XDS-cicids-96b-Wa-500n34b  (30 flows × 2 phases, seeds: [234, 8188, 13996, 22866, 23147, 24523, 25608, 25932, 26511, 29347, 37708, 39805, 40455, 41773, 47767, 48540, 53439, 57009, 59822, 63165, 63504, 63823, 81594, 82096, 83390, 84488, 86686, 94345, 95235, 97725])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.64% |   0.08% |  99.77% | r95235 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.64% |   0.08% |  99.77% | r95235 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.64% |   0.08% |  99.77% | r95235 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.64% |   0.08% |  99.77% | r95235 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.64% |   0.08% |  99.77% | r95235 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.64% |   0.08% |  99.77% | r95235 GA best_acc       val_cal
    Best FPR (any F1)        |  97.68% |   0.05% |  98.57% | r48540 GS best_fpr       beta
    Best FPR (F1>80%)        |  97.68% |   0.05% |  98.57% | r48540 GS best_fpr       beta
    Best Acc (any FPR)       |  99.64% |   0.08% |  99.77% | r95235 GA best_acc       val_cal

### best_fitness  (GS: 30 runs | GA: 30 runs)
    Grid Search : 220±137 neurons | 34±1 bits
    GA Neurons  : 180±109 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.02 99.55±0.05 | 0.29±0.03  0.12±0.05 |99.56±0.01 99.72±0.03
    fixed_05             |99.02±0.06 99.17±0.12 | 0.60±0.04  0.50±0.10 |99.37±0.04 99.47±0.08
    platt                |99.29±0.03 99.45±0.07 | 0.32±0.01  0.24±0.05 |99.55±0.02 99.65±0.04
    beta                 |99.26±0.04 99.50±0.07 | 0.27±0.02  0.17±0.06 |99.53±0.02 99.68±0.05
    empirical            |99.28±0.07 99.51±0.09 | 0.32±0.07  0.14±0.08 |99.55±0.04 99.69±0.06
    empirical_cumulative |99.31±0.02 99.55±0.05 | 0.29±0.04  0.12±0.05 |99.56±0.01 99.72±0.03
    val_cal              |99.31±0.02 99.55±0.05 | 0.29±0.03  0.12±0.05 |99.56±0.01 99.72±0.03

### best_f1  (GS: 30 runs | GA: 30 runs)
    Grid Search : 220±137 neurons | 34±1 bits
    GA Neurons  : 181±109 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.02 99.55±0.05 | 0.29±0.03  0.12±0.05 |99.56±0.01 99.72±0.03
    fixed_05             |99.02±0.06 99.17±0.12 | 0.60±0.04  0.50±0.10 |99.37±0.04 99.47±0.08
    platt                |99.29±0.03 99.45±0.07 | 0.32±0.01  0.24±0.05 |99.55±0.02 99.65±0.04
    beta                 |99.26±0.04 99.50±0.07 | 0.27±0.02  0.17±0.06 |99.53±0.02 99.68±0.04
    empirical            |99.28±0.07 99.50±0.10 | 0.32±0.07  0.15±0.09 |99.55±0.04 99.68±0.06
    empirical_cumulative |99.31±0.02 99.55±0.05 | 0.29±0.04  0.12±0.05 |99.56±0.01 99.72±0.03
    val_cal              |99.31±0.02 99.55±0.05 | 0.29±0.03  0.12±0.05 |99.56±0.01 99.72±0.03

### best_fpr  (GS: 30 runs | GA: 30 runs)
    Grid Search : 148±122 neurons | 26±10 bits
    GA Neurons  : 182±108 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |96.88±5.25 99.46±0.06 | 0.45±1.45  0.13±0.06 |98.15±3.07 99.66±0.04
    fixed_05             |93.78±8.47 99.15±0.11 | 4.72±7.28  0.52±0.09 |95.48±6.49 99.46±0.07
    platt                |96.25±6.52 99.38±0.05 | 1.11±1.99  0.28±0.04 |97.77±3.61 99.61±0.03
    beta                 |96.60±5.79 99.41±0.06 | 0.36±0.54  0.23±0.05 |98.11±2.81 99.63±0.04
    empirical            |96.65±5.71 99.42±0.07 | 0.41±0.40  0.21±0.09 |98.14±2.74 99.64±0.04
    empirical_cumulative |96.77±5.73 99.46±0.06 | 0.19±0.05  0.11±0.05 |98.22±2.75 99.66±0.04
    val_cal              |96.88±5.25 99.46±0.06 | 0.46±1.45  0.15±0.08 |98.15±3.07 99.66±0.04

### best_acc  (GS: 30 runs | GA: 30 runs)
    Grid Search : 217±134 neurons | 34±1 bits
    GA Neurons  : 180±109 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.31±0.02 99.55±0.05 | 0.29±0.03  0.12±0.05 |99.56±0.01 99.72±0.03
    fixed_05             |99.02±0.06 99.17±0.12 | 0.60±0.05  0.50±0.10 |99.38±0.04 99.47±0.08
    platt                |99.29±0.03 99.45±0.07 | 0.32±0.01  0.24±0.05 |99.55±0.02 99.65±0.04
    beta                 |99.26±0.04 99.50±0.07 | 0.27±0.02  0.17±0.06 |99.54±0.02 99.68±0.05
    empirical            |99.28±0.07 99.51±0.09 | 0.32±0.07  0.14±0.08 |99.55±0.04 99.69±0.06
    empirical_cumulative |99.31±0.02 99.55±0.05 | 0.28±0.04  0.12±0.05 |99.56±0.01 99.72±0.03
    val_cal              |99.31±0.02 99.55±0.05 | 0.29±0.03  0.12±0.05 |99.56±0.01 99.72±0.03

### best_ce  (GS: 30 runs | GA: 30 runs)
    Grid Search : 307±151 neurons | 34±0 bits
    GA Neurons  : 208±127 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.26±0.03 99.35±0.06 | 0.29±0.04  0.28±0.06 |99.53±0.02 99.59±0.04
    fixed_05             |98.98±0.06 99.15±0.09 | 0.62±0.05  0.51±0.07 |99.35±0.04 99.46±0.06
    platt                |99.23±0.04 99.31±0.07 | 0.33±0.01  0.34±0.05 |99.51±0.03 99.56±0.05
    beta                 |99.23±0.02 99.30±0.09 | 0.28±0.02  0.28±0.05 |99.51±0.01 99.56±0.05
    empirical            |99.24±0.05 99.29±0.18 | 0.36±0.05  0.32±0.17 |99.52±0.03 99.55±0.11
    empirical_cumulative |99.26±0.03 99.35±0.06 | 0.27±0.04  0.27±0.06 |99.53±0.02 99.59±0.04
    val_cal              |99.26±0.03 99.35±0.06 | 0.31±0.05  0.28±0.06 |99.53±0.02 99.59±0.04


## XDS-cicids-96b-Wb-500n34b  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best FPR (any F1)        |  99.10% |   0.05% |  99.44% | r25608 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.10% |   0.05% |  99.44% | r25608 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 467±58 neurons | 33±1 bits
    GA Neurons  : 102±2 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.26±0.01 99.52±0.00 | 0.28±0.07  0.11±0.01 |99.53±0.01 99.70±0.00
    fixed_05             |99.02±0.01 99.18±0.13 | 0.59±0.01  0.49±0.11 |99.38±0.01 99.48±0.08
    platt                |99.22±0.02 99.40±0.03 | 0.33±0.01  0.25±0.03 |99.51±0.01 99.62±0.02
    beta                 |99.23±0.01 99.47±0.04 | 0.28±0.02  0.18±0.05 |99.51±0.01 99.66±0.02
    empirical            |99.25±0.02 99.45±0.06 | 0.36±0.01  0.19±0.09 |99.52±0.01 99.65±0.04
    empirical_cumulative |99.24±0.02 99.52±0.00 | 0.21±0.01  0.10±0.01 |99.52±0.01 99.70±0.00
    val_cal              |99.27±0.01 99.52±0.00 | 0.28±0.07  0.10±0.01 |99.54±0.00 99.70±0.00

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 467±58 neurons | 33±1 bits
    GA Neurons  : 102±2 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.26±0.01 99.52±0.00 | 0.28±0.07  0.11±0.01 |99.53±0.01 99.70±0.00
    fixed_05             |99.02±0.01 99.18±0.13 | 0.59±0.01  0.49±0.11 |99.38±0.01 99.48±0.08
    platt                |99.22±0.02 99.40±0.03 | 0.33±0.01  0.25±0.03 |99.51±0.01 99.62±0.02
    beta                 |99.23±0.01 99.47±0.04 | 0.28±0.02  0.18±0.05 |99.51±0.01 99.66±0.02
    empirical            |99.25±0.02 99.45±0.06 | 0.36±0.01  0.19±0.09 |99.52±0.01 99.65±0.04
    empirical_cumulative |99.24±0.02 99.52±0.00 | 0.21±0.01  0.10±0.01 |99.52±0.01 99.70±0.00
    val_cal              |99.27±0.01 99.52±0.00 | 0.28±0.07  0.10±0.01 |99.54±0.00 99.70±0.00

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 400±0 neurons | 26±3 bits
    GA Neurons  : 109±3 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.21±15.33 90.38±15.47 | 7.63±12.59  7.57±12.65 |92.43±12.09 92.53±12.18
    fixed_05             |84.29±25.20 84.49±25.37 |18.50±30.84 18.37±30.95 |85.01±24.68 85.14±24.79
    platt                |80.88±31.47 81.04±31.61 | 0.26±0.22  0.21±0.19 |93.04±11.01 93.14±11.10
    beta                 |84.39±25.29 84.60±25.47 |18.25±31.05 18.23±31.07 |85.08±24.74 85.21±24.85
    empirical            |84.37±25.27 84.60±25.47 |18.42±30.91 18.22±31.08 |85.06±24.72 85.21±24.85
    empirical_cumulative |80.76±31.36 81.00±31.57 | 0.07±0.06  0.09±0.12 |92.97±10.95 93.12±11.08
    val_cal              |90.21±15.33 90.38±15.47 | 7.63±12.59  7.57±12.65 |92.43±12.09 92.53±12.18

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 467±58 neurons | 33±1 bits
    GA Neurons  : 102±2 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.26±0.01 99.52±0.00 | 0.28±0.07  0.11±0.01 |99.53±0.01 99.70±0.00
    fixed_05             |99.02±0.01 99.18±0.13 | 0.59±0.01  0.49±0.11 |99.38±0.01 99.48±0.08
    platt                |99.22±0.02 99.40±0.03 | 0.33±0.01  0.25±0.03 |99.51±0.01 99.62±0.02
    beta                 |99.23±0.01 99.47±0.04 | 0.28±0.02  0.18±0.05 |99.51±0.01 99.66±0.02
    empirical            |99.25±0.02 99.45±0.06 | 0.36±0.01  0.19±0.09 |99.52±0.01 99.65±0.04
    empirical_cumulative |99.24±0.02 99.52±0.00 | 0.21±0.01  0.10±0.01 |99.52±0.01 99.70±0.00
    val_cal              |99.27±0.01 99.52±0.00 | 0.28±0.07  0.10±0.01 |99.54±0.00 99.70±0.00

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±212 neurons | 34±0 bits
    GA Neurons  : 99±9 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.24±0.02 99.39±0.12 | 0.27±0.04  0.25±0.12 |99.52±0.01 99.62±0.08
    fixed_05             |98.95±0.07 99.10±0.16 | 0.64±0.06  0.55±0.15 |99.33±0.05 99.43±0.10
    platt                |99.19±0.02 99.32±0.07 | 0.34±0.01  0.34±0.07 |99.49±0.01 99.57±0.04
    beta                 |99.22±0.01 99.30±0.10 | 0.29±0.02  0.26±0.02 |99.50±0.01 99.56±0.06
    empirical            |99.23±0.03 99.39±0.12 | 0.39±0.02  0.25±0.12 |99.51±0.02 99.62±0.08
    empirical_cumulative |99.22±0.02 99.31±0.18 | 0.22±0.01  0.13±0.04 |99.51±0.01 99.57±0.11
    val_cal              |99.24±0.02 99.39±0.12 | 0.29±0.07  0.25±0.12 |99.52±0.01 99.62±0.08


## XDS-cicids-96b-Wbu-500n34b  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.48% |   0.18% |  99.67% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.48% |   0.18% |  99.67% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.48% |   0.18% |  99.67% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.48% |   0.18% |  99.67% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.48% |   0.18% |  99.67% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.48% |   0.18% |  99.67% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       empirical
    Best FPR (F1>80%)        |  99.47% |   0.18% |  99.67% | r82096 GA best_acc       empirical
    Best Acc (any FPR)       |  99.48% |   0.18% |  99.67% | r82096 GA best_acc       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 212±0 neurons | 30±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.48   |    0.20      0.21   |   99.54     99.67  
    fixed_05             |  98.96     99.24   |    0.63      0.45   |   99.34     99.52  
    platt                |  99.23     99.44   |    0.33      0.25   |   99.51     99.65  
    beta                 |  99.24     99.47   |    0.27      0.21   |   99.52     99.67  
    empirical            |  99.17     99.47   |    0.43      0.18   |   99.47     99.67  
    empirical_cumulative |  99.27     99.47   |    0.20      0.18   |   99.54     99.67  
    val_cal              |  99.27     99.48   |    0.20      0.18   |   99.54     99.67  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 212±0 neurons | 30±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.48   |    0.20      0.21   |   99.54     99.67  
    fixed_05             |  98.96     99.24   |    0.63      0.45   |   99.34     99.52  
    platt                |  99.23     99.44   |    0.33      0.25   |   99.51     99.65  
    beta                 |  99.24     99.47   |    0.27      0.21   |   99.52     99.67  
    empirical            |  99.17     99.47   |    0.43      0.18   |   99.47     99.67  
    empirical_cumulative |  99.27     99.47   |    0.20      0.18   |   99.54     99.67  
    val_cal              |  99.27     99.48   |    0.20      0.18   |   99.54     99.67  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  75.48     78.06   |   18.56     14.01   |   81.45     84.40  
    fixed_05             |  75.48     70.05   |   18.56     28.23   |   81.45     74.93  
    platt                |  67.36     71.74   |    6.33      3.97   |   82.46     85.21  
    beta                 |  68.49     69.15   |    0.07      1.10   |   86.00     85.68  
    empirical            |  68.49     68.48   |    0.07      0.00   |   86.00     86.03  
    empirical_cumulative |  68.49     68.48   |    0.07      0.00   |   86.00     86.03  
    val_cal              |  75.48     78.06   |   18.56     14.01   |   81.45     84.40  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 212±0 neurons | 30±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.48   |    0.20      0.21   |   99.54     99.67  
    fixed_05             |  98.96     99.24   |    0.63      0.45   |   99.34     99.52  
    platt                |  99.23     99.44   |    0.33      0.25   |   99.51     99.65  
    beta                 |  99.24     99.47   |    0.27      0.21   |   99.52     99.67  
    empirical            |  99.17     99.47   |    0.43      0.18   |   99.47     99.67  
    empirical_cumulative |  99.27     99.47   |    0.20      0.18   |   99.54     99.67  
    val_cal              |  99.27     99.48   |    0.20      0.18   |   99.54     99.67  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 195±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.24     99.31   |    0.25      0.33   |   99.52     99.56  
    fixed_05             |  98.98     99.11   |    0.63      0.53   |   99.35     99.44  
    platt                |  99.21     99.29   |    0.34      0.32   |   99.50     99.55  
    beta                 |  99.23     99.16   |    0.25      0.30   |   99.52     99.47  
    empirical            |  99.24     99.31   |    0.35      0.33   |   99.52     99.56  
    empirical_cumulative |  99.24     99.31   |    0.25      0.33   |   99.52     99.56  
    val_cal              |  99.24     99.31   |    0.38      0.33   |   99.52     99.56  


## XDS-cicids-96b-Wc-500n34b  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.56% |   0.07% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.56% |   0.07% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.56% |   0.07% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.56% |   0.07% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.56% |   0.07% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.56% |   0.07% |  99.73% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.56% |   0.07% |  99.72% | r82096 GA best_fpr       train_cal
    Best FPR (F1>80%)        |  99.56% |   0.07% |  99.72% | r82096 GA best_fpr       train_cal
    Best Acc (any FPR)       |  99.56% |   0.07% |  99.73% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±173 neurons | 34±0 bits
    GA Neurons  : 133±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.27±0.05 99.34±0.04 | 0.28±0.03  0.22±0.04 |99.54±0.03 99.59±0.03
    fixed_05             |98.97±0.08 99.13±0.14 | 0.63±0.06  0.54±0.12 |99.34±0.05 99.45±0.09
    platt                |99.24±0.05 99.31±0.08 | 0.33±0.00  0.35±0.06 |99.52±0.03 99.56±0.05
    beta                 |99.25±0.03 99.33±0.04 | 0.26±0.01  0.28±0.02 |99.53±0.02 99.57±0.03
    empirical            |99.27±0.04 99.34±0.04 | 0.34±0.04  0.25±0.08 |99.54±0.02 99.58±0.02
    empirical_cumulative |99.27±0.05 99.34±0.04 | 0.26±0.04  0.22±0.05 |99.54±0.03 99.59±0.03
    val_cal              |99.28±0.04 99.35±0.04 | 0.35±0.04  0.25±0.03 |99.54±0.03 99.59±0.03

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 167±58 neurons | 34±0 bits
    GA Neurons  : 133±62 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.30±0.02 99.54±0.02 | 0.28±0.05  0.11±0.04 |99.56±0.01 99.71±0.01
    fixed_05             |99.04±0.01 99.16±0.06 | 0.58±0.01  0.52±0.05 |99.39±0.01 99.47±0.04
    platt                |99.28±0.01 99.38±0.06 | 0.32±0.01  0.29±0.05 |99.54±0.01 99.61±0.04
    beta                 |99.27±0.02 99.44±0.06 | 0.27±0.01  0.23±0.06 |99.54±0.01 99.65±0.04
    empirical            |99.30±0.01 99.45±0.16 | 0.31±0.02  0.19±0.18 |99.56±0.01 99.65±0.10
    empirical_cumulative |99.29±0.03 99.54±0.02 | 0.25±0.05  0.11±0.04 |99.55±0.02 99.71±0.01
    val_cal              |99.30±0.02 99.54±0.02 | 0.28±0.05  0.11±0.04 |99.56±0.01 99.71±0.01

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 235±249 neurons | 20±14 bits
    GA Neurons  : 133±63 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |91.04±14.15 99.46±0.10 | 2.86±4.58  0.10±0.04 |94.57±8.55 99.66±0.07
    fixed_05             |90.84±13.97 99.13±0.01 | 3.16±4.32  0.53±0.02 |94.44±8.43 99.45±0.00
    platt                |89.83±16.12 99.37±0.08 | 0.55±0.33  0.30±0.06 |95.11±7.52 99.60±0.05
    beta                 |89.95±15.99 99.38±0.07 | 0.24±0.07  0.27±0.07 |95.28±7.27 99.61±0.04
    empirical            |89.89±15.94 99.35±0.21 | 0.33±0.19  0.29±0.23 |95.25±7.24 99.59±0.14
    empirical_cumulative |89.97±16.01 99.46±0.10 | 0.19±0.03  0.09±0.03 |95.30±7.28 99.66±0.07
    val_cal              |91.04±14.15 99.46±0.10 | 2.86±4.58  0.10±0.04 |94.57±8.55 99.66±0.07

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 167±58 neurons | 34±0 bits
    GA Neurons  : 133±62 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.30±0.02 99.54±0.02 | 0.28±0.05  0.11±0.04 |99.56±0.01 99.71±0.01
    fixed_05             |99.04±0.01 99.16±0.06 | 0.58±0.01  0.52±0.05 |99.39±0.01 99.47±0.04
    platt                |99.28±0.01 99.38±0.06 | 0.32±0.01  0.29±0.05 |99.54±0.01 99.61±0.04
    beta                 |99.27±0.02 99.44±0.06 | 0.27±0.01  0.23±0.06 |99.54±0.01 99.65±0.04
    empirical            |99.30±0.01 99.45±0.16 | 0.31±0.02  0.19±0.18 |99.56±0.01 99.65±0.10
    empirical_cumulative |99.29±0.03 99.54±0.02 | 0.25±0.05  0.11±0.04 |99.55±0.02 99.71±0.01
    val_cal              |99.30±0.02 99.54±0.02 | 0.28±0.05  0.11±0.04 |99.56±0.01 99.71±0.01

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 300±173 neurons | 34±0 bits
    GA Neurons  : 133±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.27±0.05 99.34±0.04 | 0.28±0.03  0.22±0.04 |99.54±0.03 99.59±0.03
    fixed_05             |98.97±0.08 99.13±0.14 | 0.63±0.06  0.54±0.12 |99.34±0.05 99.45±0.09
    platt                |99.24±0.05 99.31±0.08 | 0.33±0.00  0.35±0.06 |99.52±0.03 99.56±0.05
    beta                 |99.25±0.03 99.33±0.04 | 0.26±0.01  0.28±0.02 |99.53±0.02 99.57±0.03
    empirical            |99.27±0.04 99.34±0.04 | 0.34±0.04  0.25±0.08 |99.54±0.02 99.58±0.02
    empirical_cumulative |99.27±0.05 99.34±0.04 | 0.26±0.04  0.22±0.05 |99.54±0.03 99.59±0.03
    val_cal              |99.28±0.04 99.35±0.04 | 0.35±0.04  0.25±0.03 |99.54±0.03 99.59±0.03



# XDS-ciciot — width × weight cohort breakdown (72 non-OLD completed)

    Total non-OLD completed : 72  |  Total wall: 280.1h  |  Avg/run: 233m
    Latest done : 03/07/2026 09:21 UTC

    Weight schemes:
      Wa (CIC-IoT legacy, ce=0.35 acc=0.30)
      Wb (paper/PUB50, ce=0.10 acc=0.20)
      Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)
      Wc (CE-heavy NEW, ce=0.70 acc=0.10)


## XDS-ciciot-8b-Wa-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  87.95% |  12.82% |  93.75% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  87.95% |  12.82% |  93.75% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  82.41% |   1.51% |  89.06% | r45211 GA best_f1        fixed_05
    Best F1 (FPR<6%)         |  82.41% |   1.51% |  89.06% | r45211 GA best_f1        fixed_05
    Best F1 (FPR<5%)         |  82.41% |   1.51% |  89.06% | r45211 GA best_f1        fixed_05
    Best F1 (FPR<4%)         |  82.41% |   1.51% |  89.06% | r45211 GA best_f1        fixed_05
    Best FPR (any F1)        |  82.11% |   1.40% |  88.82% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  82.11% |   1.40% |  88.82% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  87.54% |  19.33% |  93.87% | r45211 GA best_f1        empirical

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 208±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.40     87.94   |   15.87     14.15   |   93.58     93.81  
    fixed_05             |  81.69     82.11   |    1.47      1.40   |   88.49     88.82  
    platt                |  87.42     87.92   |   14.34     14.53   |   93.50     93.82  
    beta                 |  87.35     87.87   |   16.66     15.75   |   93.60     93.86  
    empirical            |  87.04     87.60   |   20.52     18.58   |   93.65     93.87  
    empirical_cumulative |  87.43     87.88   |   12.90     12.53   |   93.42     93.69  
    val_cal              |  87.44     87.94   |   13.00     14.03   |   93.44     93.81  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 208±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.40     87.94   |   15.87     14.39   |   93.58     93.83  
    fixed_05             |  81.69     82.41   |    1.47      1.51   |   88.49     89.06  
    platt                |  87.42     87.92   |   14.34     14.32   |   93.50     93.81  
    beta                 |  87.35     87.90   |   16.66     15.42   |   93.60     93.86  
    empirical            |  87.04     87.54   |   20.52     19.33   |   93.65     93.87  
    empirical_cumulative |  87.43     87.92   |   12.90     12.08   |   93.42     93.69  
    val_cal              |  87.44     87.95   |   13.00     12.82   |   93.44     93.75  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 64±0 bits
    GA Neurons  : 249±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.15     87.75   |   15.81     14.36   |   93.42     93.71  
    fixed_05             |  81.58     82.23   |    1.42      1.53   |   88.39     88.92  
    platt                |  87.14     87.70   |   14.57     14.94   |   93.34     93.71  
    beta                 |  87.03     87.62   |   17.66     16.35   |   93.46     93.75  
    empirical            |  86.53     87.45   |   22.85     18.42   |   93.49     93.76  
    empirical_cumulative |  87.14     87.73   |   12.72     12.31   |   93.23     93.58  
    val_cal              |  87.16     87.78   |   15.54     13.61   |   93.41     93.69  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 208±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.40     87.94   |   15.87     14.15   |   93.58     93.81  
    fixed_05             |  81.69     82.11   |    1.47      1.40   |   88.49     88.82  
    platt                |  87.42     87.92   |   14.34     14.53   |   93.50     93.82  
    beta                 |  87.35     87.87   |   16.66     15.75   |   93.60     93.86  
    empirical            |  87.04     87.60   |   20.52     18.58   |   93.65     93.87  
    empirical_cumulative |  87.43     87.88   |   12.90     12.53   |   93.42     93.69  
    val_cal              |  87.44     87.94   |   13.00     14.03   |   93.44     93.81  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 244±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.40     87.76   |   15.87     13.85   |   93.58     93.69  
    fixed_05             |  81.69     82.25   |    1.47      1.54   |   88.49     88.94  
    platt                |  87.42     87.70   |   14.34     14.94   |   93.50     93.71  
    beta                 |  87.35     87.63   |   16.66     16.38   |   93.60     93.75  
    empirical            |  87.04     87.34   |   20.52     19.16   |   93.65     93.74  
    empirical_cumulative |  87.43     87.71   |   12.90     12.12   |   93.42     93.56  
    val_cal              |  87.44     87.78   |   13.00     13.45   |   93.44     93.68  


## XDS-ciciot-8b-Wa-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  87.76% |  13.81% |  93.68% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  87.76% |  13.81% |  93.68% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  80.71% |   1.06% |  87.65% | r45211 GA best_acc       fixed_05
    Best F1 (FPR<6%)         |  80.71% |   1.06% |  87.65% | r45211 GA best_acc       fixed_05
    Best F1 (FPR<5%)         |  80.71% |   1.06% |  87.65% | r45211 GA best_acc       fixed_05
    Best F1 (FPR<4%)         |  80.71% |   1.06% |  87.65% | r45211 GA best_acc       fixed_05
    Best FPR (any F1)        |  80.71% |   1.06% |  87.65% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  80.71% |   1.06% |  87.65% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  87.46% |  18.88% |  93.80% | r45211 GA best_acc       empirical

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 450±71 neurons | 34±0 bits
    GA Neurons  : 443±54 neurons | 31±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |86.97±0.00 87.04±1.01 |16.20±0.62 15.03±1.23 |93.33±0.04 93.30±0.56
    fixed_05             |80.03±0.03 80.62±0.13 | 1.12±0.04  1.31±0.36 |87.08±0.03 87.59±0.08
    platt                |86.68±0.02 86.96±0.91 |12.41±0.01 13.78±2.61 |92.91±0.01 93.18±0.42
    beta                 |86.81±0.08 86.70±1.11 |19.29±0.41 19.60±1.55 |93.43±0.02 93.38±0.58
    empirical            |86.44±0.01 85.71±2.47 |22.31±0.29 25.26±9.02 |93.41±0.03 93.22±0.83
    empirical_cumulative |86.95±0.03 87.00±1.06 |15.16±0.77 13.42±0.40 |93.26±0.07 93.18±0.70
    val_cal              |86.99±0.00 87.04±1.01 |16.17±1.00 14.60±1.11 |93.34±0.06 93.28±0.57

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 450±71 neurons | 34±0 bits
    GA Neurons  : 385±28 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |86.97±0.00 87.63±0.17 |16.20±0.62 15.17±1.44 |93.33±0.04 93.68±0.02
    fixed_05             |80.03±0.03 80.48±0.32 | 1.12±0.04  1.10±0.06 |87.08±0.03 87.46±0.26
    platt                |86.68±0.02 87.45±0.22 |12.41±0.01 12.11±0.25 |92.91±0.01 93.39±0.13
    beta                 |86.81±0.08 87.38±0.14 |19.29±0.41 18.87±0.52 |93.43±0.02 93.75±0.05
    empirical            |86.44±0.01 87.35±0.16 |22.31±0.29 19.24±0.50 |93.41±0.03 93.76±0.06
    empirical_cumulative |86.95±0.03 87.59±0.23 |15.16±0.77 13.82±0.16 |93.26±0.07 93.58±0.13
    val_cal              |86.99±0.00 87.64±0.17 |16.17±1.00 14.86±1.47 |93.34±0.06 93.67±0.02

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 450±71 neurons | 12±0 bits
    GA Neurons  : 444±47 neurons | 31±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.34±0.01 87.18±0.75 |15.71±0.17 14.68±1.05 |90.89±0.02 93.37±0.41
    fixed_05             |73.80±0.03 80.48±0.03 | 1.01±0.06  1.28±0.30 |81.34±0.04 87.47±0.00
    platt                |83.06±0.04 87.11±0.65 |20.82±0.09 13.42±1.96 |91.11±0.04 93.26±0.30
    beta                 |81.99±0.20 86.89±0.73 |30.57±0.48 19.20±0.38 |91.29±0.08 93.47±0.42
    empirical            |81.93±0.20 85.61±2.43 |30.91±1.83 26.21±9.53 |91.29±0.05 93.23±0.75
    empirical_cumulative |83.20±0.10 87.15±0.73 |12.25±1.36 13.16±0.49 |90.51±0.18 93.26±0.44
    val_cal              |83.34±0.01 87.20±0.75 |15.71±0.17 14.62±0.97 |90.89±0.02 93.38±0.53

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 392±19 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |86.98±0.02 87.63±0.17 |16.32±0.78 14.91±1.07 |93.35±0.06 93.67±0.04
    fixed_05             |80.01±0.01 80.50±0.29 | 1.15±0.08  1.12±0.09 |87.07±0.01 87.48±0.24
    platt                |86.66±0.06 87.42±0.25 |12.46±0.08 12.15±0.30 |92.90±0.03 93.38±0.14
    beta                 |86.77±0.02 87.36±0.17 |19.68±0.14 18.91±0.58 |93.43±0.02 93.74±0.07
    empirical            |86.49±0.08 87.32±0.20 |21.96±0.21 19.46±0.81 |93.41±0.03 93.75±0.07
    empirical_cumulative |86.95±0.03 87.62±0.19 |15.32±0.54 13.91±0.28 |93.27±0.05 93.60±0.10
    val_cal              |86.99±0.01 87.64±0.17 |16.83±0.07 14.30±0.69 |93.39±0.00 93.64±0.07

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 10±3 bits
    GA Neurons  : 402±112 neurons | 27±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |82.92±0.77 86.18±0.20 |17.01±0.77 15.29±0.87 |90.70±0.48 92.78±0.18
    fixed_05             |72.91±1.16 80.59±0.08 | 0.95±0.01  1.62±0.09 |80.44±1.17 87.59±0.08
    platt                |82.21±1.21 86.05±0.38 |25.14±4.82 17.38±2.48 |90.93±0.38 92.83±0.08
    beta                 |81.01±1.39 85.40±0.73 |33.46±4.30 23.28±3.66 |90.96±0.46 92.84±0.18
    empirical            |80.54±0.27 82.33±2.31 |36.12±4.76 37.88±8.83 |90.94±0.35 92.28±0.49
    empirical_cumulative |82.87±0.83 86.12±0.19 |15.27±0.27 13.54±0.58 |90.51±0.61 92.62±0.09
    val_cal              |82.92±0.77 86.19±0.19 |17.00±0.76 15.27±0.16 |90.70±0.48 92.78±0.14


## XDS-ciciot-8b-Wb-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  87.66% |  13.94% |  93.63% | r45211 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  87.66% |  13.94% |  93.63% | r45211 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  85.50% |   4.78% |  91.63% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<6%)         |  85.50% |   4.78% |  91.63% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<5%)         |  85.50% |   4.78% |  91.63% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<4%)         |  82.34% |   1.68% |  89.03% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  81.66% |   1.49% |  88.46% | r45211 GS best_ce        fixed_05
    Best FPR (F1>80%)        |  81.66% |   1.49% |  88.46% | r45211 GS best_ce        fixed_05
    Best Acc (any FPR)       |  87.32% |  19.41% |  93.75% | r45211 GA best_ce        empirical

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 64±0 bits
    GA Neurons  : 58±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.33     87.52   |   14.12     13.88   |   93.43     93.54  
    fixed_05             |  81.78     82.26   |    1.51      1.77   |   88.56     88.97  
    platt                |  87.29     87.54   |   15.04     14.96   |   93.46     93.61  
    beta                 |  87.13     87.39   |   18.44     17.32   |   93.57     93.66  
    empirical            |  86.83     86.90   |   21.22     21.41   |   93.57     93.62  
    empirical_cumulative |  85.23     85.50   |    4.70      4.78   |   91.42     91.63  
    val_cal              |  87.33     87.55   |   14.12     14.52   |   93.43     93.60  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 64±0 bits
    GA Neurons  : 58±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.33     87.52   |   14.12     13.88   |   93.43     93.54  
    fixed_05             |  81.78     82.26   |    1.51      1.77   |   88.56     88.97  
    platt                |  87.29     87.54   |   15.04     14.96   |   93.46     93.61  
    beta                 |  87.13     87.39   |   18.44     17.32   |   93.57     93.66  
    empirical            |  86.83     86.90   |   21.22     21.41   |   93.57     93.62  
    empirical_cumulative |  85.23     85.50   |    4.70      4.78   |   91.42     91.63  
    val_cal              |  87.33     87.55   |   14.12     14.52   |   93.43     93.60  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 16±0 bits
    GA Neurons  : 56±0 neurons | 16±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  84.53     82.98   |   18.32     28.69   |   91.90     91.75  
    fixed_05             |  76.50     78.19   |    1.56      1.06   |   83.99     85.48  
    platt                |  84.33     82.41   |   15.02     18.06   |   91.53     90.42  
    beta                 |  77.22     81.63   |   53.46     38.10   |   91.11     91.83  
    empirical            |  83.03     81.75   |   32.79     37.71   |   92.16     91.86  
    empirical_cumulative |  80.78     79.02   |    3.69      1.35   |   87.93     86.23  
    val_cal              |  84.53     82.98   |   18.25     28.72   |   91.90     91.76  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 64±0 bits
    GA Neurons  : 58±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.33     87.52   |   14.12     13.88   |   93.43     93.54  
    fixed_05             |  81.78     82.26   |    1.51      1.77   |   88.56     88.97  
    platt                |  87.29     87.54   |   15.04     14.96   |   93.46     93.61  
    beta                 |  87.13     87.39   |   18.44     17.32   |   93.57     93.66  
    empirical            |  86.83     86.90   |   21.22     21.41   |   93.57     93.62  
    empirical_cumulative |  85.23     85.50   |    4.70      4.78   |   91.42     91.63  
    val_cal              |  87.33     87.55   |   14.12     14.52   |   93.43     93.60  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 250±0 neurons | 62±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.34     87.61   |   14.88     15.40   |   93.49     93.68  
    fixed_05             |  81.66     82.34   |    1.49      1.68   |   88.46     89.03  
    platt                |  87.31     87.62   |   14.22     14.85   |   93.43     93.65  
    beta                 |  87.30     87.51   |   16.70     16.89   |   93.57     93.71  
    empirical            |  86.80     87.32   |   21.44     19.41   |   93.56     93.75  
    empirical_cumulative |  85.06     85.17   |    4.73      4.08   |   91.30     91.34  
    val_cal              |  87.34     87.66   |   14.88     13.94   |   93.49     93.63  


## XDS-ciciot-8b-Wb-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  87.15% |  14.81% |  93.36% | r45211 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  86.99% |  12.80% |  93.14% | r45211 GA best_acc       platt
    Best F1 (FPR<10%)        |  84.47% |   6.15% |  90.97% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<6%)         |  83.91% |   5.68% |  90.53% | r45211 GS best_acc       empirical_cumulative
    Best F1 (FPR<5%)         |  80.84% |   4.01% |  88.01% | r45211 GS best_ce        empirical_cumulative
    Best F1 (FPR<4%)         |  80.74% |   3.40% |  87.87% | r45211 GS best_fpr       empirical_cumulative
    Best FPR (any F1)        |  80.74% |   3.40% |  87.87% | r45211 GS best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  80.74% |   3.40% |  87.87% | r45211 GS best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  86.67% |  22.20% |  93.54% | r45211 GA best_acc       empirical

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 116±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  86.79     87.15   |   16.29     14.81   |   93.23     93.36  
    fixed_05             |  79.97     79.60   |    1.15      0.94   |   87.03     86.70  
    platt                |  86.49     86.99   |   12.90     12.80   |   92.82     93.14  
    beta                 |  86.53     85.98   |   20.89     26.39   |   93.36     93.42  
    empirical            |  86.43     86.67   |   21.99     22.20   |   93.38     93.54  
    empirical_cumulative |  83.91     84.47   |    5.68      6.15   |   90.53     90.97  
    val_cal              |  86.79     87.15   |   16.29     14.81   |   93.23     93.36  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 116±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  86.79     87.15   |   16.29     14.81   |   93.23     93.36  
    fixed_05             |  79.97     79.60   |    1.15      0.94   |   87.03     86.70  
    platt                |  86.49     86.99   |   12.90     12.80   |   92.82     93.14  
    beta                 |  86.53     85.98   |   20.89     26.39   |   93.36     93.42  
    empirical            |  86.43     86.67   |   21.99     22.20   |   93.38     93.54  
    empirical_cumulative |  83.91     84.47   |    5.68      6.15   |   90.53     90.97  
    val_cal              |  86.79     87.15   |   16.29     14.81   |   93.23     93.36  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 32±0 bits
    GA Neurons  : 5±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  82.41     80.22   |   14.84      6.11   |   90.14     87.68  
    fixed_05             |  78.35     77.94   |    1.25      0.95   |   85.64     85.25  
    platt                |  82.41     80.16   |   14.92      7.17   |   90.15     87.72  
    beta                 |  82.32     80.16   |   15.93      7.17   |   90.17     87.72  
    empirical            |  82.32     80.16   |   15.93      7.17   |   90.17     87.72  
    empirical_cumulative |  80.74     78.00   |    3.40      1.04   |   87.87     85.30  
    val_cal              |  82.41     80.22   |   14.92      6.11   |   90.15     87.68  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 116±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  86.79     87.15   |   16.29     14.81   |   93.23     93.36  
    fixed_05             |  79.97     79.60   |    1.15      0.94   |   87.03     86.70  
    platt                |  86.49     86.99   |   12.90     12.80   |   92.82     93.14  
    beta                 |  86.53     85.98   |   20.89     26.39   |   93.36     93.42  
    empirical            |  86.43     86.67   |   21.99     22.20   |   93.38     93.54  
    empirical_cumulative |  83.91     84.47   |    5.68      6.15   |   90.53     90.97  
    val_cal              |  86.79     87.15   |   16.29     14.81   |   93.23     93.36  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 16±0 bits
    GA Neurons  : 93±0 neurons | 24±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  84.50     85.23   |   19.02     14.64   |   91.94     92.10  
    fixed_05             |  76.01     82.76   |    1.11      5.90   |   83.48     89.68  
    platt                |  84.44     83.55   |   16.64     28.41   |   91.72     92.09  
    beta                 |  83.99     83.00   |   25.92     31.42   |   92.16     92.02  
    empirical            |  79.75     75.29   |   45.63     57.78   |   91.55     90.65  
    empirical_cumulative |  80.84     83.33   |    4.01      6.76   |   88.01     90.18  
    val_cal              |  84.50     85.23   |   19.39     14.75   |   91.97     92.11  


## XDS-ciciot-8b-Wc-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  87.86% |  13.35% |  93.72% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  87.86% |  13.35% |  93.72% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  87.51% |   9.22% |  93.26% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<6%)         |  82.46% |   1.62% |  89.11% | r45211 GA best_acc       fixed_05
    Best F1 (FPR<5%)         |  82.46% |   1.62% |  89.11% | r45211 GA best_acc       fixed_05
    Best F1 (FPR<4%)         |  82.46% |   1.62% |  89.11% | r45211 GA best_acc       fixed_05
    Best FPR (any F1)        |  80.81% |   1.28% |  87.74% | r45211 GS best_acc       fixed_05
    Best FPR (F1>80%)        |  80.81% |   1.28% |  87.74% | r45211 GS best_acc       fixed_05
    Best Acc (any FPR)       |  87.78% |  15.83% |  93.81% | r45211 GA best_acc       beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 250±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.40     87.76   |   15.87     13.62   |   93.58     93.67  
    fixed_05             |  81.69     82.29   |    1.47      1.49   |   88.49     88.97  
    platt                |  87.42     87.69   |   14.34     15.57   |   93.50     93.74  
    beta                 |  87.35     87.62   |   16.66     16.57   |   93.60     93.76  
    empirical            |  87.04     87.15   |   20.52     20.49   |   93.65     93.72  
    empirical_cumulative |  86.99     87.49   |    9.60      9.79   |   92.95     93.28  
    val_cal              |  87.44     87.76   |   13.00     13.62   |   93.44     93.67  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 64±0 bits
    GA Neurons  : 242±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.44     87.85   |   15.80     13.81   |   93.60     93.74  
    fixed_05             |  81.50     82.46   |    1.33      1.62   |   88.32     89.11  
    platt                |  87.45     87.83   |   13.93     14.79   |   93.50     93.79  
    beta                 |  87.34     87.78   |   17.40     15.83   |   93.64     93.81  
    empirical            |  87.12     87.34   |   19.94     19.57   |   93.66     93.77  
    empirical_cumulative |  87.07     87.51   |   10.43      9.22   |   93.05     93.26  
    val_cal              |  87.46     87.86   |   14.39     13.35   |   93.53     93.72  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 64±0 bits
    GA Neurons  : 244±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.34     87.78   |   15.54     13.61   |   93.52     93.69  
    fixed_05             |  81.65     82.35   |    1.40      1.41   |   88.45     89.01  
    platt                |  87.33     87.65   |   14.51     15.68   |   93.46     93.73  
    beta                 |  87.28     87.55   |   16.91     16.74   |   93.57     93.72  
    empirical            |  86.95     87.33   |   20.64     19.28   |   93.60     93.75  
    empirical_cumulative |  86.75     87.48   |    8.71      9.29   |   92.74     93.24  
    val_cal              |  87.38     87.81   |   13.60     12.62   |   93.43     93.65  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 48±0 bits
    GA Neurons  : 242±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.28     87.85   |   15.90     13.81   |   93.51     93.74  
    fixed_05             |  80.81     82.46   |    1.28      1.62   |   87.74     89.11  
    platt                |  87.14     87.83   |   12.96     14.79   |   93.24     93.79  
    beta                 |  87.13     87.78   |   19.05     15.83   |   93.61     93.81  
    empirical            |  86.62     87.34   |   23.17     19.57   |   93.57     93.77  
    empirical_cumulative |  86.88     87.51   |   10.97      9.22   |   92.96     93.26  
    val_cal              |  87.29     87.86   |   16.84     13.35   |   93.57     93.72  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 250±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.40     87.76   |   15.87     13.62   |   93.58     93.67  
    fixed_05             |  81.69     82.29   |    1.47      1.49   |   88.49     88.97  
    platt                |  87.42     87.69   |   14.34     15.57   |   93.50     93.74  
    beta                 |  87.35     87.62   |   16.66     16.57   |   93.60     93.76  
    empirical            |  87.04     87.15   |   20.52     20.49   |   93.65     93.72  
    empirical_cumulative |  86.99     87.49   |    9.60      9.79   |   92.95     93.28  
    val_cal              |  87.44     87.76   |   13.00     13.62   |   93.44     93.67  


## XDS-ciciot-8b-Wc-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  87.09% |  15.35% |  93.36% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  86.79% |  12.14% |  92.97% | r45211 GA best_f1        platt
    Best F1 (FPR<10%)        |  85.32% |   9.00% |  91.78% | r45211 GA best_fpr       empirical_cumulative
    Best F1 (FPR<6%)         |  80.26% |   1.43% |  87.30% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  80.26% |   1.43% |  87.30% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  80.26% |   1.43% |  87.30% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  80.21% |   1.26% |  87.24% | r45211 GA best_f1        fixed_05
    Best FPR (F1>80%)        |  80.21% |   1.26% |  87.24% | r45211 GA best_f1        fixed_05
    Best Acc (any FPR)       |  86.62% |  21.88% |  93.49% | r45211 GA best_acc       empirical

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 16±0 bits
    GA Neurons  : 419±0 neurons | 26±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  84.50     86.23   |   19.02     14.97   |   91.94     92.79  
    fixed_05             |  76.01     80.26   |    1.11      1.43   |   83.48     87.30  
    platt                |  84.44     86.11   |   16.64     16.89   |   91.72     92.83  
    beta                 |  83.99     85.57   |   25.92     21.06   |   92.16     92.79  
    empirical            |  79.75     59.41   |   45.63     85.03   |   91.55     87.76  
    empirical_cumulative |  83.63     85.96   |   10.15     10.61   |   90.66     92.32  
    val_cal              |  84.50     86.27   |   19.39     13.29   |   91.97     92.70  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 403±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  86.79     87.08   |   16.29     16.16   |   93.23     93.40  
    fixed_05             |  79.97     80.21   |    1.15      1.26   |   87.03     87.24  
    platt                |  86.49     86.79   |   12.90     12.14   |   92.82     92.97  
    beta                 |  86.53     86.82   |   20.89     20.04   |   93.36     93.49  
    empirical            |  86.43     86.58   |   21.99     22.18   |   93.38     93.48  
    empirical_cumulative |  86.45     86.66   |   12.12     11.43   |   92.75     92.84  
    val_cal              |  86.79     87.09   |   16.29     15.35   |   93.23     93.36  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 12±0 bits
    GA Neurons  : 434±0 neurons | 26±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  83.20     85.85   |   20.23     14.28   |   91.16     92.49  
    fixed_05             |  73.42     79.84   |    1.07      1.21   |   80.97     86.93  
    platt                |  83.09     85.49   |   22.00     18.62   |   91.24     92.56  
    beta                 |  82.22     85.01   |   31.67     22.48   |   91.55     92.53  
    empirical            |  82.07     78.60   |   32.60     47.69   |   91.54     91.12  
    empirical_cumulative |  82.60     85.32   |    9.92      9.00   |   89.88     91.78  
    val_cal              |  83.24     85.88   |   19.17     13.78   |   91.10     92.47  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 422±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  86.79     87.08   |   16.29     15.90   |   93.23     93.38  
    fixed_05             |  79.97     80.18   |    1.15      1.28   |   87.03     87.22  
    platt                |  86.49     86.77   |   12.90     12.17   |   92.82     92.96  
    beta                 |  86.53     86.81   |   20.89     20.02   |   93.36     93.48  
    empirical            |  86.43     86.62   |   21.99     21.88   |   93.38     93.49  
    empirical_cumulative |  86.45     86.63   |   12.12     11.41   |   92.75     92.82  
    val_cal              |  86.79     87.08   |   16.29     15.75   |   93.23     93.38  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 419±0 neurons | 26±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  84.50     86.23   |   19.02     14.97   |   91.94     92.79  
    fixed_05             |  76.01     80.26   |    1.11      1.43   |   83.48     87.30  
    platt                |  84.44     86.11   |   16.64     16.89   |   91.72     92.83  
    beta                 |  83.99     85.57   |   25.92     21.06   |   92.16     92.79  
    empirical            |  79.75     59.41   |   45.63     85.03   |   91.55     87.76  
    empirical_cumulative |  83.63     85.96   |   10.15     10.61   |   90.66     92.32  
    val_cal              |  84.50     86.27   |   19.39     13.29   |   91.97     92.70  


## XDS-ciciot-16b-Wa-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  90.43% |  10.99% |  95.15% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  90.43% |  10.99% |  95.15% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  83.94% |   1.11% |  90.22% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<6%)         |  83.94% |   1.11% |  90.22% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  83.94% |   1.11% |  90.22% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  83.94% |   1.11% |  90.22% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  81.28% |   1.04% |  88.11% | r45211 GS best_acc       fixed_05
    Best FPR (F1>80%)        |  81.28% |   1.04% |  88.11% | r45211 GS best_acc       fixed_05
    Best Acc (any FPR)       |  90.33% |  14.19% |  95.23% | r45211 GA best_f1        beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 240±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.93     90.14   |   13.22     12.39   |   94.96     95.05  
    fixed_05             |  82.19     83.94   |    1.10      1.11   |   88.86     90.22  
    platt                |  89.93     90.13   |   12.97     12.67   |   94.95     95.05  
    beta                 |  89.69     89.95   |   16.48     15.29   |   94.98     95.07  
    empirical            |  89.69     89.96   |   16.48     15.18   |   94.98     95.07  
    empirical_cumulative |  89.84     90.09   |   10.87     10.16   |   94.80     94.91  
    val_cal              |  89.93     90.15   |   12.82     11.74   |   94.94     95.02  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 48±0 bits
    GA Neurons  : 247±0 neurons | 83±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.01     90.43   |   12.86     11.41   |   94.99     95.17  
    fixed_05             |  81.54     82.82   |    1.25      1.24   |   88.35     89.36  
    platt                |  90.01     90.42   |   11.90     11.90   |   94.94     95.19  
    beta                 |  89.92     90.33   |   14.77     14.19   |   95.03     95.23  
    empirical            |  89.79     90.10   |   15.97     16.16   |   95.01     95.19  
    empirical_cumulative |  89.98     90.40   |   11.69     10.74   |   94.92     95.13  
    val_cal              |  90.01     90.43   |   12.86     10.99   |   94.99     95.15  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 244±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.93     90.10   |   13.22     12.62   |   94.96     95.03  
    fixed_05             |  82.19     83.78   |    1.10      1.09   |   88.86     90.10  
    platt                |  89.93     90.10   |   12.97     12.72   |   94.95     95.04  
    beta                 |  89.69     89.93   |   16.48     15.24   |   94.98     95.06  
    empirical            |  89.69     89.83   |   16.48     16.26   |   94.98     95.05  
    empirical_cumulative |  89.84     90.10   |   10.87     10.17   |   94.80     94.92  
    val_cal              |  89.93     90.11   |   12.82     12.10   |   94.94     95.02  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 100±0 bits
    GA Neurons  : 236±0 neurons | 81±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.89     90.35   |   14.79     13.32   |   95.01     95.21  
    fixed_05             |  81.28     82.73   |    1.04      1.31   |   88.11     89.30  
    platt                |  89.38     90.04   |   12.98     12.27   |   94.62     94.98  
    beta                 |  89.74     90.26   |   16.43     14.61   |   95.01     95.22  
    empirical            |  89.86     90.01   |   15.59     16.52   |   95.03     95.16  
    empirical_cumulative |  89.88     90.35   |   14.69     13.32   |   95.00     95.21  
    val_cal              |  89.90     90.36   |   15.05     13.40   |   95.03     95.22  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 240±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.93     90.14   |   13.22     12.39   |   94.96     95.05  
    fixed_05             |  82.19     83.94   |    1.10      1.11   |   88.86     90.22  
    platt                |  89.93     90.13   |   12.97     12.67   |   94.95     95.05  
    beta                 |  89.69     89.95   |   16.48     15.29   |   94.98     95.07  
    empirical            |  89.69     89.96   |   16.48     15.18   |   94.98     95.07  
    empirical_cumulative |  89.84     90.09   |   10.87     10.16   |   94.80     94.91  
    val_cal              |  89.93     90.15   |   12.82     11.74   |   94.94     95.02  


## XDS-ciciot-16b-Wa-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  90.29% |  12.23% |  95.12% | r45211 GA best_fpr       val_cal
    Best F1 (FPR<14%)        |  90.29% |  12.23% |  95.12% | r45211 GA best_fpr       val_cal
    Best F1 (FPR<10%)        |  82.91% |   9.44% |  90.07% | r45211 GS best_fpr       train_cal
    Best F1 (FPR<6%)         |  80.89% |   0.97% |  87.79% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  80.89% |   0.97% |  87.79% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  80.89% |   0.97% |  87.79% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  80.75% |   0.92% |  87.67% | r45211 GA best_f1        fixed_05
    Best FPR (F1>80%)        |  80.75% |   0.92% |  87.67% | r45211 GA best_f1        fixed_05
    Best Acc (any FPR)       |  90.14% |  15.16% |  95.17% | r45211 GA best_f1        beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.92     90.22   |   13.68     12.79   |   94.98     95.11  
    fixed_05             |  80.30     80.89   |    0.98      0.97   |   87.30     87.79  
    platt                |  89.76     90.20   |   11.55     11.45   |   94.78     95.04  
    beta                 |  89.69     90.05   |   17.07     16.00   |   95.01     95.16  
    empirical            |  89.75     90.05   |   16.75     16.00   |   95.02     95.16  
    empirical_cumulative |  89.91     90.20   |   12.91     11.45   |   94.94     95.04  
    val_cal              |  89.93     90.24   |   13.61     12.31   |   94.98     95.10  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 479±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.92     90.26   |   13.68     12.97   |   94.98     95.14  
    fixed_05             |  80.30     80.75   |    0.98      0.92   |   87.30     87.67  
    platt                |  89.76     89.64   |   11.55     11.48   |   94.78     94.71  
    beta                 |  89.69     90.14   |   17.07     15.16   |   95.01     95.17  
    empirical            |  89.75     89.77   |   16.75     17.99   |   95.02     95.09  
    empirical_cumulative |  89.91     90.27   |   12.91     12.02   |   94.94     95.10  
    val_cal              |  89.93     90.28   |   13.61     12.51   |   94.98     95.13  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 20±0 bits
    GA Neurons  : 494±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  82.91     90.28   |    9.44     12.35   |   90.07     95.13  
    fixed_05             |  75.54     80.75   |    2.27      0.94   |   83.16     87.67  
    platt                |  82.90     90.20   |    9.57     11.12   |   90.07     95.02  
    beta                 |  80.14     90.11   |    4.35     15.22   |   87.46     95.16  
    empirical            |  82.90     89.86   |    9.57     17.55   |   90.07     95.12  
    empirical_cumulative |  82.91     90.21   |    9.44     11.18   |   90.07     95.03  
    val_cal              |  82.91     90.29   |    9.44     12.23   |   90.07     95.12  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.78     90.20   |   15.15     13.44   |   94.97     95.13  
    fixed_05             |  80.35     80.59   |    1.12      0.97   |   87.35     87.54  
    platt                |  88.65     89.25   |   12.00     11.77   |   94.14     94.49  
    beta                 |  89.49     90.06   |   18.22     15.85   |   94.95     95.16  
    empirical            |  89.63     89.81   |   17.14     17.68   |   94.97     95.10  
    empirical_cumulative |  89.78     90.20   |   15.04     13.30   |   94.96     95.12  
    val_cal              |  89.79     90.21   |   14.95     13.73   |   94.96     95.15  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.92     90.22   |   13.68     12.79   |   94.98     95.11  
    fixed_05             |  80.30     80.89   |    0.98      0.97   |   87.30     87.79  
    platt                |  89.76     90.20   |   11.55     11.45   |   94.78     95.04  
    beta                 |  89.69     90.05   |   17.07     16.00   |   95.01     95.16  
    empirical            |  89.75     90.05   |   16.75     16.00   |   95.02     95.16  
    empirical_cumulative |  89.91     90.20   |   12.91     11.45   |   94.94     95.04  
    val_cal              |  89.93     90.24   |   13.61     12.31   |   94.98     95.10  


## XDS-ciciot-16b-Wb-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  90.17% |  11.88% |  95.04% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  90.17% |  11.88% |  95.04% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  89.75% |   8.45% |  94.64% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<6%)         |  88.71% |   4.87% |  93.80% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<5%)         |  88.71% |   4.87% |  93.80% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<4%)         |  84.65% |   2.47% |  90.85% | r45211 GS best_fpr       empirical_cumulative
    Best FPR (any F1)        |  82.72% |   0.78% |  89.26% | r45211 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  82.72% |   0.78% |  89.26% | r45211 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  90.06% |  13.99% |  95.07% | r45211 GA best_acc       beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 32±0 bits
    GA Neurons  : 247±0 neurons | 48±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.47     90.14   |   13.21     12.50   |   94.69     95.05  
    fixed_05             |  79.84     81.89   |    0.97      1.15   |   86.91     88.62  
    platt                |  89.39     90.14   |   11.43     11.46   |   94.56     95.01  
    beta                 |  87.86     90.06   |   25.01     13.99   |   94.41     95.07  
    empirical            |  89.39     89.72   |   15.70     16.90   |   94.77     95.02  
    empirical_cumulative |  88.94     89.75   |    9.62      8.45   |   94.19     94.64  
    val_cal              |  89.47     90.17   |   13.21     11.88   |   94.69     95.04  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 32±0 bits
    GA Neurons  : 247±0 neurons | 48±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.47     90.14   |   13.21     12.50   |   94.69     95.05  
    fixed_05             |  79.84     81.89   |    0.97      1.15   |   86.91     88.62  
    platt                |  89.39     90.14   |   11.43     11.46   |   94.56     95.01  
    beta                 |  87.86     90.06   |   25.01     13.99   |   94.41     95.07  
    empirical            |  89.39     89.72   |   15.70     16.90   |   94.77     95.02  
    empirical_cumulative |  88.94     89.75   |    9.62      8.45   |   94.19     94.64  
    val_cal              |  89.47     90.17   |   13.21     11.88   |   94.69     95.04  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 32±0 bits
    GA Neurons  : 7±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  87.92     85.03   |    9.10      7.70   |   93.52     91.48  
    fixed_05             |  80.42     82.72   |    0.80      0.78   |   87.38     89.26  
    platt                |  87.92     85.02   |    9.14      7.79   |   93.52     91.48  
    beta                 |  87.85     85.02   |   10.02      7.50   |   93.53     91.47  
    empirical            |  87.92     85.03   |    9.10      7.70   |   93.52     91.48  
    empirical_cumulative |  84.65     82.75   |    2.47      0.82   |   90.85     89.28  
    val_cal              |  87.92     85.03   |    9.14      7.70   |   93.52     91.48  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 32±0 bits
    GA Neurons  : 247±0 neurons | 48±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.47     90.14   |   13.21     12.50   |   94.69     95.05  
    fixed_05             |  79.84     81.89   |    0.97      1.15   |   86.91     88.62  
    platt                |  89.39     90.14   |   11.43     11.46   |   94.56     95.01  
    beta                 |  87.86     90.06   |   25.01     13.99   |   94.41     95.07  
    empirical            |  89.39     89.72   |   15.70     16.90   |   94.77     95.02  
    empirical_cumulative |  88.94     89.75   |    9.62      8.45   |   94.19     94.64  
    val_cal              |  89.47     90.17   |   13.21     11.88   |   94.69     95.04  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 250±0 neurons | 63±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.84     90.01   |   13.03     12.80   |   94.90     94.99  
    fixed_05             |  82.17     83.39   |    1.14      1.08   |   88.85     89.80  
    platt                |  89.83     90.02   |   12.85     12.29   |   94.89     94.97  
    beta                 |  89.64     89.92   |   16.45     15.01   |   94.95     95.04  
    empirical            |  89.52     89.75   |   17.59     17.09   |   94.93     95.04  
    empirical_cumulative |  87.64     88.71   |    5.38      4.87   |   93.13     93.80  
    val_cal              |  89.84     90.04   |   12.64     11.50   |   94.88     94.95  


## XDS-ciciot-16b-Wb-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  90.09% |  12.44% |  95.02% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  90.09% |  12.44% |  95.02% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  89.06% |   7.81% |  94.18% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<6%)         |  82.06% |   3.45% |  88.94% | r45211 GS best_fpr       beta
    Best F1 (FPR<5%)         |  82.06% |   3.45% |  88.94% | r45211 GS best_fpr       beta
    Best F1 (FPR<4%)         |  82.06% |   3.45% |  88.94% | r45211 GS best_fpr       beta
    Best FPR (any F1)        |  80.78% |   1.06% |  87.70% | r45211 GA best_ce        fixed_05
    Best FPR (F1>80%)        |  80.78% |   1.06% |  87.70% | r45211 GA best_ce        fixed_05
    Best Acc (any FPR)       |  89.90% |  15.73% |  95.06% | r45211 GA best_acc       beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 494±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.65     90.07   |   13.97     12.29   |   94.83     95.00  
    fixed_05             |  79.46     80.65   |    0.81      1.13   |   86.57     87.60  
    platt                |  89.44     90.06   |   11.82     11.49   |   94.60     94.96  
    beta                 |  89.46     89.90   |   16.79     15.73   |   94.86     95.06  
    empirical            |  89.47     89.64   |   16.72     17.78   |   94.86     95.01  
    empirical_cumulative |  89.12     89.89   |   10.26     10.67   |   94.33     94.82  
    val_cal              |  89.65     90.09   |   14.79     12.44   |   94.87     95.02  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 494±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.65     90.07   |   13.97     12.29   |   94.83     95.00  
    fixed_05             |  79.46     80.65   |    0.81      1.13   |   86.57     87.60  
    platt                |  89.44     90.06   |   11.82     11.49   |   94.60     94.96  
    beta                 |  89.46     89.90   |   16.79     15.73   |   94.86     95.06  
    empirical            |  89.47     89.64   |   16.72     17.78   |   94.86     95.01  
    empirical_cumulative |  89.12     89.89   |   10.26     10.67   |   94.33     94.82  
    val_cal              |  89.65     90.09   |   14.79     12.44   |   94.87     95.02  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 28±0 bits
    GA Neurons  : 5±0 neurons | 28±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  85.86     83.07   |   11.90      9.80   |   92.34     90.22  
    fixed_05             |  78.70     79.00   |    0.93      1.53   |   85.91     86.23  
    platt                |  85.86     83.06   |   11.90     10.23   |   92.34     90.25  
    beta                 |  82.06     82.36   |    3.45      7.42   |   88.94     89.49  
    empirical            |  85.84     83.09   |   12.73     10.61   |   92.38     90.30  
    empirical_cumulative |  81.90     79.00   |    3.07      1.53   |   88.78     86.23  
    val_cal              |  85.87     83.09   |   11.80     10.61   |   92.34     90.30  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 494±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.65     90.07   |   13.97     12.29   |   94.83     95.00  
    fixed_05             |  79.46     80.65   |    0.81      1.13   |   86.57     87.60  
    platt                |  89.44     90.06   |   11.82     11.49   |   94.60     94.96  
    beta                 |  89.46     89.90   |   16.79     15.73   |   94.86     95.06  
    empirical            |  89.47     89.64   |   16.72     17.78   |   94.86     95.01  
    empirical_cumulative |  89.12     89.89   |   10.26     10.67   |   94.33     94.82  
    val_cal              |  89.65     90.09   |   14.79     12.44   |   94.87     95.02  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.81     90.01   |   15.19     13.24   |   94.99     95.01  
    fixed_05             |  80.32     80.78   |    1.08      1.06   |   87.32     87.70  
    platt                |  88.76     89.96   |   11.77     11.34   |   94.19     94.89  
    beta                 |  89.56     89.85   |   17.85     15.93   |   94.97     95.04  
    empirical            |  89.68     89.66   |   16.73     17.52   |   94.99     95.01  
    empirical_cumulative |  86.63     89.06   |    7.29      7.81   |   92.56     94.18  
    val_cal              |  89.83     90.03   |   15.04     12.01   |   94.99     94.96  


## XDS-ciciot-16b-Wc-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  90.48% |  10.58% |  95.16% | r45211 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  90.48% |  10.58% |  95.16% | r45211 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  90.14% |   7.06% |  94.81% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<6%)         |  84.03% |   1.03% |  90.29% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  84.03% |   1.03% |  90.29% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  84.03% |   1.03% |  90.29% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  83.51% |   0.99% |  89.89% | r45211 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  83.51% |   0.99% |  89.89% | r45211 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  90.28% |  14.25% |  95.21% | r45211 GA best_acc       beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 64±0 bits
    GA Neurons  : 213±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.94     90.46   |   12.84     11.18   |   94.95     95.18  
    fixed_05             |  82.16     84.03   |    1.12      1.03   |   88.84     90.29  
    platt                |  89.92     90.43   |   13.20     12.00   |   94.96     95.20  
    beta                 |  89.70     90.22   |   16.44     14.61   |   94.98     95.19  
    empirical            |  89.76     90.22   |   15.83     14.61   |   94.98     95.19  
    empirical_cumulative |  89.73     90.14   |   10.66      7.06   |   94.72     94.81  
    val_cal              |  89.94     90.48   |   12.69     10.58   |   94.94     95.16  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 64±0 bits
    GA Neurons  : 239±0 neurons | 79±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.99     90.42   |   13.34     11.54   |   95.00     95.17  
    fixed_05             |  82.29     82.59   |    1.27      1.12   |   88.95     89.18  
    platt                |  89.99     90.42   |   13.10     12.00   |   94.99     95.19  
    beta                 |  89.92     90.30   |   15.08     13.91   |   95.04     95.21  
    empirical            |  89.92     90.14   |   15.08     15.62   |   95.04     95.19  
    empirical_cumulative |  89.87     90.43   |   12.10     10.22   |   94.87     95.12  
    val_cal              |  90.01     90.45   |   13.80     10.54   |   95.03     95.15  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 212±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.93     90.35   |   13.22     11.85   |   94.96     95.14  
    fixed_05             |  82.19     83.51   |    1.10      0.99   |   88.86     89.89  
    platt                |  89.93     90.34   |   12.97     12.29   |   94.95     95.16  
    beta                 |  89.69     90.17   |   16.48     14.89   |   94.98     95.18  
    empirical            |  89.69     90.17   |   16.48     15.00   |   94.98     95.18  
    empirical_cumulative |  89.50     90.06   |    9.22      7.31   |   94.51     94.77  
    val_cal              |  89.93     90.37   |   12.82     10.05   |   94.94     95.08  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 100±0 bits
    GA Neurons  : 234±0 neurons | 79±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.94     90.42   |   14.07     12.17   |   95.01     95.20  
    fixed_05             |  81.31     82.44   |    1.07      1.12   |   88.14     89.06  
    platt                |  89.44     90.42   |   12.89     12.03   |   94.66     95.19  
    beta                 |  89.78     90.28   |   16.43     14.25   |   95.03     95.21  
    empirical            |  89.78     90.18   |   16.43     15.29   |   95.03     95.20  
    empirical_cumulative |  89.90     90.40   |   13.69     11.60   |   94.97     95.16  
    val_cal              |  89.94     90.42   |   14.07     12.11   |   95.01     95.20  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 64±0 bits
    GA Neurons  : 213±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.94     90.46   |   12.84     11.18   |   94.95     95.18  
    fixed_05             |  82.16     84.03   |    1.12      1.03   |   88.84     90.29  
    platt                |  89.92     90.43   |   13.20     12.00   |   94.96     95.20  
    beta                 |  89.70     90.22   |   16.44     14.61   |   94.98     95.19  
    empirical            |  89.76     90.22   |   15.83     14.61   |   94.98     95.19  
    empirical_cumulative |  89.73     90.14   |   10.66      7.06   |   94.72     94.81  
    val_cal              |  89.94     90.48   |   12.69     10.58   |   94.94     95.16  


## XDS-ciciot-16b-Wc-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  90.32% |  13.24% |  95.19% | r45211 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  90.32% |  13.24% |  95.19% | r45211 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  90.08% |   9.25% |  94.87% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<6%)         |  82.53% |   1.00% |  89.12% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  82.53% |   1.00% |  89.12% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  82.53% |   1.00% |  89.12% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  80.46% |   0.94% |  87.42% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  80.46% |   0.94% |  87.42% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  90.29% |  13.81% |  95.20% | r45211 GA best_ce        beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 497±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.92     90.29   |   13.68     12.30   |   94.98     95.13  
    fixed_05             |  80.30     82.53   |    0.98      1.00   |   87.30     89.12  
    platt                |  89.76     90.26   |   11.55     10.57   |   94.78     95.03  
    beta                 |  89.69     90.29   |   17.07     13.81   |   95.01     95.20  
    empirical            |  89.75     90.07   |   16.75     16.59   |   95.02     95.20  
    empirical_cumulative |  89.75     90.08   |   11.47      9.25   |   94.77     94.87  
    val_cal              |  89.93     90.32   |   13.61     13.24   |   94.98     95.19  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 442±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.88     90.28   |   14.88     13.20   |   95.01     95.17  
    fixed_05             |  80.33     80.48   |    1.03      0.98   |   87.32     87.45  
    platt                |  88.71     89.44   |   11.95     11.41   |   94.17     94.59  
    beta                 |  89.68     90.01   |   17.18     16.02   |   95.00     95.14  
    empirical            |  89.72     89.97   |   16.79     16.35   |   95.01     95.13  
    empirical_cumulative |  89.85     90.22   |   14.67     12.28   |   94.98     95.09  
    val_cal              |  89.89     90.30   |   14.97     12.90   |   95.02     95.16  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 488±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.31     90.25   |   15.87     13.11   |   94.73     95.14  
    fixed_05             |  79.47     82.49   |    0.93      0.96   |   86.59     89.09  
    platt                |  88.99     90.21   |   11.78     10.57   |   94.33     95.01  
    beta                 |  89.22     90.22   |   18.20     13.54   |   94.80     95.15  
    empirical            |  89.14     89.96   |   19.31     16.64   |   94.81     95.14  
    empirical_cumulative |  88.86     89.94   |   10.84      8.29   |   94.20     94.74  
    val_cal              |  89.31     90.27   |   15.87     11.81   |   94.73     95.09  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 447±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.88     90.25   |   14.88     13.43   |   95.01     95.16  
    fixed_05             |  80.33     80.46   |    1.03      0.94   |   87.32     87.42  
    platt                |  88.71     89.39   |   11.95     11.51   |   94.17     94.56  
    beta                 |  89.68     90.06   |   17.18     15.83   |   95.00     95.16  
    empirical            |  89.72     90.13   |   16.79     15.14   |   95.01     95.17  
    empirical_cumulative |  89.85     90.23   |   14.67     12.91   |   94.98     95.12  
    val_cal              |  89.89     90.25   |   14.97     13.43   |   95.02     95.16  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 497±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.92     90.29   |   13.68     12.30   |   94.98     95.13  
    fixed_05             |  80.30     82.53   |    0.98      1.00   |   87.30     89.12  
    platt                |  89.76     90.26   |   11.55     10.57   |   94.78     95.03  
    beta                 |  89.69     90.29   |   17.07     13.81   |   95.01     95.20  
    empirical            |  89.75     90.07   |   16.75     16.59   |   95.02     95.20  
    empirical_cumulative |  89.75     90.08   |   11.47      9.25   |   94.77     94.87  
    val_cal              |  89.93     90.32   |   13.61     13.24   |   94.98     95.19  


## XDS-ciciot-32b-Wa-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  91.39% |  11.19% |  95.71% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  91.39% |  11.19% |  95.71% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  91.27% |   9.93% |  95.60% | r45211 GA best_fpr       platt
    Best F1 (FPR<6%)         |  83.59% |   0.94% |  89.94% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  83.59% |   0.94% |  89.94% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  83.59% |   0.94% |  89.94% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  82.31% |   0.89% |  88.94% | r45211 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  82.31% |   0.89% |  88.94% | r45211 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  91.37% |  11.42% |  95.71% | r45211 GA best_f1        train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 48±0 bits
    GA Neurons  : 241±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.66     90.70   |   14.90     12.10   |   95.46     95.36  
    fixed_05             |  81.27     83.59   |    1.06      0.94   |   88.11     89.94  
    platt                |  90.07     90.70   |   11.12     12.21   |   94.94     95.36  
    beta                 |  90.65     90.55   |   15.21     14.88   |   95.46     95.39  
    empirical            |  90.61     90.37   |   15.77     16.56   |   95.46     95.36  
    empirical_cumulative |  90.66     90.58   |   14.76      9.95   |   95.45     95.20  
    val_cal              |  90.66     90.71   |   14.90     12.55   |   95.46     95.38  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 48±0 bits
    GA Neurons  : 250±0 neurons | 46±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.66     91.37   |   14.90     11.42   |   95.46     95.71  
    fixed_05             |  81.27     82.43   |    1.06      1.06   |   88.11     89.04  
    platt                |  90.07     91.34   |   11.12     10.58   |   94.94     95.66  
    beta                 |  90.65     91.25   |   15.21     13.00   |   95.46     95.71  
    empirical            |  90.61     91.16   |   15.77     13.89   |   95.46     95.69  
    empirical_cumulative |  90.66     91.35   |   14.76     10.63   |   95.45     95.67  
    val_cal              |  90.66     91.39   |   14.90     11.19   |   95.46     95.71  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 64±0 bits
    GA Neurons  : 241±0 neurons | 45±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.86     91.28   |   14.37     10.50   |   94.97     95.62  
    fixed_05             |  82.04     82.31   |    1.08      0.89   |   88.74     88.94  
    platt                |  89.88     91.27   |   12.69      9.93   |   94.91     95.60  
    beta                 |  89.82     91.23   |   15.90     11.80   |   95.02     95.65  
    empirical            |  89.61     91.01   |   18.58     14.39   |   95.03     95.63  
    empirical_cumulative |  89.84     91.25   |   11.92      9.39   |   94.85     95.56  
    val_cal              |  89.91     91.29   |   13.14     10.79   |   94.94     95.64  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 48±0 bits
    GA Neurons  : 237±0 neurons | 45±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.66     91.31   |   14.90     11.82   |   95.46     95.69  
    fixed_05             |  81.27     82.40   |    1.06      0.95   |   88.11     89.01  
    platt                |  90.07     91.10   |   11.12     10.23   |   94.94     95.51  
    beta                 |  90.65     91.28   |   15.21     12.34   |   95.46     95.70  
    empirical            |  90.61     91.13   |   15.77     13.90   |   95.46     95.67  
    empirical_cumulative |  90.66     91.31   |   14.76     11.12   |   95.45     95.67  
    val_cal              |  90.66     91.31   |   14.90     11.21   |   95.46     95.67  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 241±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.84     90.70   |   14.49     12.10   |   94.97     95.36  
    fixed_05             |  81.85     83.59   |    1.10      0.94   |   88.59     89.94  
    platt                |  89.80     90.70   |   13.12     12.21   |   94.88     95.36  
    beta                 |  89.75     90.55   |   16.66     14.88   |   95.02     95.39  
    empirical            |  89.46     90.37   |   19.62     16.56   |   95.00     95.36  
    empirical_cumulative |  89.78     90.58   |   12.67      9.95   |   94.85     95.20  
    val_cal              |  89.85     90.71   |   14.38     12.55   |   94.97     95.38  


## XDS-ciciot-32b-Wa-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  91.28% |  10.59% |  95.63% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  91.28% |  10.59% |  95.63% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  91.21% |   9.99% |  95.56% | r45211 GA best_fpr       empirical_cumulative
    Best F1 (FPR<6%)         |  81.24% |   1.06% |  88.09% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  81.24% |   1.06% |  88.09% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  81.24% |   1.06% |  88.09% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  80.53% |   0.83% |  87.48% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  80.53% |   0.83% |  87.48% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  91.23% |  11.92% |  95.65% | r45211 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.70     91.14   |   14.72     11.94   |   95.47     95.60  
    fixed_05             |  80.35     81.24   |    0.92      1.06   |   87.33     88.09  
    platt                |  89.79     90.92   |   10.96     10.38   |   94.77     95.41  
    beta                 |  90.62     90.95   |   15.81     14.62   |   95.47     95.61  
    empirical            |  90.62     90.99   |   15.90     14.33   |   95.48     95.62  
    empirical_cumulative |  90.71     91.15   |   14.60     10.95   |   95.47     95.57  
    val_cal              |  90.71     91.17   |   14.60     11.14   |   95.47     95.59  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 324±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.70     91.26   |   14.72     10.89   |   95.47     95.63  
    fixed_05             |  80.35     80.75   |    0.92      0.85   |   87.33     87.66  
    platt                |  89.79     91.27   |   10.96     10.65   |   94.77     95.62  
    beta                 |  90.62     91.09   |   15.81     13.86   |   95.47     95.65  
    empirical            |  90.62     91.10   |   15.90     13.69   |   95.48     95.65  
    empirical_cumulative |  90.71     91.27   |   14.60     10.53   |   95.47     95.62  
    val_cal              |  90.71     91.28   |   14.60     10.59   |   95.47     95.63  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 317±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.10     91.23   |   18.95     11.48   |   95.32     95.64  
    fixed_05             |  80.12     80.80   |    1.03      0.89   |   87.15     87.71  
    platt                |  89.27     91.21   |   11.47     10.49   |   94.48     95.58  
    beta                 |  89.93     91.07   |   17.29     13.69   |   95.15     95.63  
    empirical            |  90.08     90.97   |   19.09     14.56   |   95.32     95.62  
    empirical_cumulative |  90.10     91.21   |   18.95      9.99   |   95.32     95.56  
    val_cal              |  90.10     91.24   |   18.95     10.99   |   95.32     95.62  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 343±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.70     91.23   |   14.72     11.92   |   95.47     95.65  
    fixed_05             |  80.35     80.53   |    0.92      0.83   |   87.33     87.48  
    platt                |  89.79     90.89   |   10.96     10.48   |   94.77     95.40  
    beta                 |  90.62     91.00   |   15.81     14.26   |   95.47     95.62  
    empirical            |  90.62     91.04   |   15.90     13.99   |   95.48     95.63  
    empirical_cumulative |  90.71     91.22   |   14.60     11.52   |   95.47     95.63  
    val_cal              |  90.71     91.24   |   14.60     11.66   |   95.47     95.65  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.14     91.14   |   18.25     11.94   |   95.32     95.60  
    fixed_05             |  80.51     81.24   |    1.09      1.06   |   87.48     88.09  
    platt                |  89.42     90.92   |   11.18     10.38   |   94.56     95.41  
    beta                 |  90.09     90.95   |   16.56     14.62   |   95.21     95.61  
    empirical            |  90.11     90.99   |   18.47     14.33   |   95.31     95.62  
    empirical_cumulative |  90.11     91.15   |   15.46     10.95   |   95.17     95.57  
    val_cal              |  90.15     91.17   |   17.79     11.14   |   95.30     95.59  


## XDS-ciciot-32b-Wb-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  91.21% |  10.14% |  95.57% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  91.21% |  10.14% |  95.57% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  91.05% |   8.85% |  95.43% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<6%)         |  88.51% |   4.89% |  93.68% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<5%)         |  88.51% |   4.89% |  93.68% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<4%)         |  82.60% |   0.88% |  89.17% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  82.60% |   0.88% |  89.17% | r45211 GA best_ce        fixed_05
    Best FPR (F1>80%)        |  82.60% |   0.88% |  89.17% | r45211 GA best_ce        fixed_05
    Best Acc (any FPR)       |  91.03% |  13.46% |  95.60% | r45211 GA best_acc       beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 48±0 bits
    GA Neurons  : 250±0 neurons | 48±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.74     91.17   |   14.43     10.54   |   95.48     95.56  
    fixed_05             |  81.31     81.93   |    1.04      1.10   |   88.14     88.65  
    platt                |  90.34     91.17   |   11.45     10.97   |   95.12     95.58  
    beta                 |  90.66     90.98   |   15.47     13.60   |   95.48     95.58  
    empirical            |  90.61     91.01   |   15.87     13.30   |   95.47     95.59  
    empirical_cumulative |  88.90     90.83   |    7.88      8.08   |   94.08     95.26  
    val_cal              |  90.74     91.20   |   14.57     10.21   |   95.48     95.56  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 48±0 bits
    GA Neurons  : 250±0 neurons | 48±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.74     91.17   |   14.43     10.54   |   95.48     95.56  
    fixed_05             |  81.31     81.93   |    1.04      1.10   |   88.14     88.65  
    platt                |  90.34     91.17   |   11.45     10.97   |   95.12     95.58  
    beta                 |  90.66     90.98   |   15.47     13.60   |   95.48     95.58  
    empirical            |  90.61     91.01   |   15.87     13.30   |   95.47     95.59  
    empirical_cumulative |  88.90     90.83   |    7.88      8.08   |   94.08     95.26  
    val_cal              |  90.74     91.20   |   14.57     10.21   |   95.48     95.56  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : — neurons | — bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  71.60     74.34   |   28.39     28.55   |   82.78     85.27  
    fixed_05             |  60.92     70.79   |    2.93     28.55   |   67.18     82.03  
    platt                |  69.71     73.22   |   61.44     47.04   |   87.57     87.28  
    beta                 |  69.71     74.34   |   61.44     28.55   |   87.57     85.27  
    empirical            |  69.71     73.22   |   61.44     47.04   |   87.57     87.28  
    empirical_cumulative |  60.92     65.50   |    2.93      0.00   |   67.18     72.27  
    val_cal              |  71.60     74.34   |   28.39     28.55   |   82.78     85.27  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 32±0 bits
    GA Neurons  : 250±0 neurons | 48±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.71     91.19   |   13.69     10.71   |   95.43     95.58  
    fixed_05             |  80.29     82.03   |    0.98      1.16   |   87.29     88.73  
    platt                |  89.76     91.18   |   11.11     10.97   |   94.76     95.58  
    beta                 |  90.57     91.03   |   15.64     13.46   |   95.44     95.60  
    empirical            |  90.60     90.96   |   15.45     13.97   |   95.45     95.59  
    empirical_cumulative |  88.75     91.05   |    8.62      8.85   |   94.02     95.43  
    val_cal              |  90.71     91.21   |   13.99     10.14   |   95.44     95.57  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 250±0 neurons | 62±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.77     90.42   |   15.52     12.45   |   94.98     95.21  
    fixed_05             |  81.92     82.60   |    1.15      0.88   |   88.64     89.17  
    platt                |  89.72     90.41   |   13.06     12.37   |   94.83     95.20  
    beta                 |  89.78     90.24   |   16.64     15.28   |   95.03     95.24  
    empirical            |  89.39     89.92   |   20.05     17.99   |   94.99     95.18  
    empirical_cumulative |  87.39     88.51   |    5.68      4.89   |   92.98     93.68  
    val_cal              |  89.80     90.42   |   15.70     12.45   |   95.00     95.21  


## XDS-ciciot-32b-Wb-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  90.94% |  11.21% |  95.46% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  90.94% |  11.21% |  95.46% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  88.71% |   9.15% |  94.02% | r45211 GS best_acc       empirical_cumulative
    Best F1 (FPR<6%)         |  81.59% |   1.35% |  88.39% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  81.59% |   1.35% |  88.39% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  81.59% |   1.35% |  88.39% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  80.27% |   0.90% |  87.27% | r45211 GA best_f1        fixed_05
    Best FPR (F1>80%)        |  80.27% |   0.90% |  87.27% | r45211 GA best_f1        fixed_05
    Best Acc (any FPR)       |  90.73% |  14.90% |  95.49% | r45211 GA best_f1        empirical

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 424±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.43     90.91   |   16.64     12.07   |   95.40     95.48  
    fixed_05             |  80.05     80.27   |    0.98      0.90   |   87.09     87.27  
    platt                |  89.58     90.93   |   11.05     10.74   |   94.65     95.43  
    beta                 |  90.39     90.68   |   17.12     15.31   |   95.40     95.48  
    empirical            |  90.37     90.73   |   17.26     14.90   |   95.40     95.49  
    empirical_cumulative |  88.71     90.85   |    9.15     10.29   |   94.02     95.37  
    val_cal              |  90.43     90.94   |   16.64     11.21   |   95.40     95.46  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 424±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.43     90.91   |   16.64     12.07   |   95.40     95.48  
    fixed_05             |  80.05     80.27   |    0.98      0.90   |   87.09     87.27  
    platt                |  89.58     90.93   |   11.05     10.74   |   94.65     95.43  
    beta                 |  90.39     90.68   |   17.12     15.31   |   95.40     95.48  
    empirical            |  90.37     90.73   |   17.26     14.90   |   95.40     95.49  
    empirical_cumulative |  88.71     90.85   |    9.15     10.29   |   94.02     95.37  
    val_cal              |  90.43     90.94   |   16.64     11.21   |   95.40     95.46  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 28±0 bits
    GA Neurons  : 8±0 neurons | 16±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  82.58     82.42   |   12.83     22.73   |   90.10     90.84  
    fixed_05             |  77.93     77.24   |    0.83     12.46   |   85.22     85.80  
    platt                |  82.56     82.41   |   12.95     22.71   |   90.10     90.84  
    beta                 |  82.52     79.31   |   14.55     13.28   |   90.21     87.62  
    empirical            |  82.52     82.41   |   14.55     22.70   |   90.21     90.84  
    empirical_cumulative |  79.61     71.88   |    2.53      0.24   |   86.85     79.31  
    val_cal              |  82.58     82.42   |   12.83     22.73   |   90.10     90.84  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 433±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.43     90.88   |   16.64     12.31   |   95.40     95.47  
    fixed_05             |  80.05     80.30   |    0.98      0.92   |   87.09     87.29  
    platt                |  89.58     90.89   |   11.05     10.78   |   94.65     95.41  
    beta                 |  90.39     90.63   |   17.12     15.61   |   95.40     95.47  
    empirical            |  90.37     90.64   |   17.26     15.44   |   95.40     95.47  
    empirical_cumulative |  88.71     90.81   |    9.15     10.55   |   94.02     95.36  
    val_cal              |  90.43     90.93   |   16.64     11.26   |   95.40     95.46  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 361±0 neurons | 28±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.13     89.80   |   18.86     12.86   |   95.34     94.87  
    fixed_05             |  80.44     81.59   |    1.08      1.35   |   87.42     88.39  
    platt                |  89.45     89.73   |   11.32     12.00   |   94.59     94.79  
    beta                 |  90.00     89.71   |   17.11     14.20   |   95.18     94.88  
    empirical            |  90.13     70.43   |   18.97     67.95   |   95.34     89.71  
    empirical_cumulative |  87.21     87.09   |    7.40      6.46   |   92.96     92.82  
    val_cal              |  90.13     89.80   |   18.97     12.73   |   95.34     94.86  


## XDS-ciciot-32b-Wc-250n100b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  91.61% |  10.17% |  95.80% | r45211 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  91.61% |  10.17% |  95.80% | r45211 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  91.59% |   9.49% |  95.77% | r45211 GA best_ce        train_cal
    Best F1 (FPR<6%)         |  84.83% |   0.98% |  90.89% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  84.83% |   0.98% |  90.89% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  84.83% |   0.98% |  90.89% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  84.79% |   0.92% |  90.85% | r45211 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  84.79% |   0.92% |  90.85% | r45211 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  91.56% |  10.95% |  95.80% | r45211 GA best_ce        platt

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 189±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.90     91.59   |   14.99      9.49   |   95.03     95.77  
    fixed_05             |  81.96     84.83   |    1.13      0.98   |   88.67     90.89  
    platt                |  89.83     91.56   |   12.98     10.95   |   94.89     95.80  
    beta                 |  89.88     91.37   |   16.46     13.13   |   95.08     95.78  
    empirical            |  89.59     91.39   |   19.53     13.00   |   95.07     95.78  
    empirical_cumulative |  89.54     91.40   |   10.29      7.13   |   94.59     95.56  
    val_cal              |  89.91     91.61   |   15.25     10.17   |   95.04     95.80  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 32±0 bits
    GA Neurons  : 197±0 neurons | 50±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.51     91.44   |   14.27     11.11   |   95.34     95.74  
    fixed_05             |  79.73     82.99   |    0.90      0.98   |   86.81     89.48  
    platt                |  89.84     91.45   |   11.83     10.72   |   94.85     95.73  
    beta                 |  89.03     91.31   |   23.29     13.18   |   94.96     95.75  
    empirical            |  90.51     91.22   |   14.27     14.06   |   95.34     95.73  
    empirical_cumulative |  90.53     91.26   |   13.91      8.94   |   95.34     95.55  
    val_cal              |  90.53     91.46   |   13.91     10.63   |   95.34     95.73  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 150±0 neurons | 64±0 bits
    GA Neurons  : 195±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.81     91.42   |   14.12     10.11   |   94.93     95.69  
    fixed_05             |  82.04     84.79   |    1.20      0.92   |   88.74     90.85  
    platt                |  89.79     91.43   |   12.94     10.72   |   94.87     95.72  
    beta                 |  89.71     91.27   |   16.54     12.85   |   94.99     95.71  
    empirical            |  89.47     91.16   |   19.85     13.88   |   95.02     95.69  
    empirical_cumulative |  89.47     91.18   |    9.90      6.29   |   94.53     95.40  
    val_cal              |  89.81     91.44   |   14.29      9.11   |   94.94     95.66  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 50±0 neurons | 32±0 bits
    GA Neurons  : 68±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.51     91.07   |   14.27     12.89   |   95.34     95.60  
    fixed_05             |  79.73     81.40   |    0.90      1.33   |   86.81     88.24  
    platt                |  89.84     90.61   |   11.83     10.93   |   94.85     95.26  
    beta                 |  89.03     90.59   |   23.29     16.71   |   94.96     95.49  
    empirical            |  90.51     91.02   |   14.27     13.50   |   95.34     95.60  
    empirical_cumulative |  90.53     91.07   |   13.91     12.89   |   95.34     95.60  
    val_cal              |  90.53     91.07   |   13.91     12.89   |   95.34     95.60  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 64±0 bits
    GA Neurons  : 189±0 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.90     91.59   |   14.99      9.49   |   95.03     95.77  
    fixed_05             |  81.96     84.83   |    1.13      0.98   |   88.67     90.89  
    platt                |  89.83     91.56   |   12.98     10.95   |   94.89     95.80  
    beta                 |  89.88     91.37   |   16.46     13.13   |   95.08     95.78  
    empirical            |  89.59     91.39   |   19.53     13.00   |   95.07     95.78  
    empirical_cumulative |  89.54     91.40   |   10.29      7.13   |   94.59     95.56  
    val_cal              |  89.91     91.61   |   15.25     10.17   |   95.04     95.80  


## XDS-ciciot-32b-Wc-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  91.30% |  11.02% |  95.66% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  91.30% |  11.02% |  95.66% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  91.24% |   9.98% |  95.58% | r45211 GA best_f1        empirical_cumulative
    Best F1 (FPR<6%)         |  81.68% |   0.87% |  88.43% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  81.68% |   0.87% |  88.43% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  81.68% |   0.87% |  88.43% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  80.93% |   0.81% |  87.81% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  80.93% |   0.81% |  87.81% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  91.22% |  12.27% |  95.66% | r45211 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 489±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.59     91.14   |   15.76     12.25   |   95.45     95.61  
    fixed_05             |  80.22     81.68   |    0.93      0.87   |   87.22     88.43  
    platt                |  89.68     90.87   |   11.04     10.44   |   94.71     95.39  
    beta                 |  90.55     90.95   |   16.33     14.59   |   95.45     95.60  
    empirical            |  90.55     91.13   |   16.26     11.96   |   95.45     95.60  
    empirical_cumulative |  89.70     91.11   |   11.17     11.89   |   94.73     95.58  
    val_cal              |  90.60     91.16   |   15.85     12.54   |   95.46     95.64  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 497±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.35     91.29   |   14.30     11.24   |   95.25     95.66  
    fixed_05             |  80.43     81.15   |    1.06      0.89   |   87.41     88.00  
    platt                |  89.71     91.29   |   11.26     10.55   |   94.74     95.63  
    beta                 |  90.26     91.01   |   16.24     14.61   |   95.29     95.64  
    empirical            |  90.18     91.23   |   18.79      9.82   |   95.36     95.57  
    empirical_cumulative |  90.27     91.24   |   12.88      9.98   |   95.14     95.58  
    val_cal              |  90.36     91.30   |   14.08     11.02   |   95.25     95.66  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 16±0 bits
    GA Neurons  : 497±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  79.64     91.15   |   13.62     12.43   |   87.91     95.62  
    fixed_05             |  76.12     81.64   |    1.16      0.86   |   83.60     88.39  
    platt                |  79.64     90.81   |   13.62     10.41   |   87.91     95.35  
    beta                 |  78.27     90.93   |    2.96     14.61   |   85.73     95.60  
    empirical            |  79.64     91.06   |   13.62     13.61   |   87.91     95.63  
    empirical_cumulative |  78.27     90.36   |    2.96      8.24   |   85.73     94.99  
    val_cal              |  79.64     91.15   |   13.62     12.38   |   87.91     95.63  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 488±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.43     91.22   |   16.64     12.27   |   95.40     95.66  
    fixed_05             |  80.05     80.93   |    0.98      0.81   |   87.09     87.81  
    platt                |  89.58     90.91   |   11.05     10.66   |   94.65     95.42  
    beta                 |  90.39     90.97   |   17.12     14.67   |   95.40     95.62  
    empirical            |  90.37     91.21   |   17.26     11.91   |   95.40     95.64  
    empirical_cumulative |  90.26     91.22   |   13.90     11.95   |   95.18     95.65  
    val_cal              |  90.43     91.23   |   16.64     12.18   |   95.40     95.66  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 489±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.59     91.14   |   15.76     12.25   |   95.45     95.61  
    fixed_05             |  80.22     81.68   |    0.93      0.87   |   87.22     88.43  
    platt                |  89.68     90.87   |   11.04     10.44   |   94.71     95.39  
    beta                 |  90.55     90.95   |   16.33     14.59   |   95.45     95.60  
    empirical            |  90.55     91.13   |   16.26     11.96   |   95.45     95.60  
    empirical_cumulative |  89.70     91.11   |   11.17     11.89   |   94.73     95.58  
    val_cal              |  90.60     91.16   |   15.85     12.54   |   95.46     95.64  


## XDS-ciciot-64b-Wa-250n100b  (3 flows × 2 phases, seeds: [8198, 45198, 45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.65% |   9.01% |  96.35% | r45211 GA best_f1        train_cal
    Best F1 (FPR<14%)        |  92.65% |   9.01% |  96.35% | r45211 GA best_f1        train_cal
    Best F1 (FPR<10%)        |  92.65% |   9.01% |  96.35% | r45211 GA best_f1        train_cal
    Best F1 (FPR<6%)         |  85.17% |   0.89% |  91.13% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  85.17% |   0.89% |  91.13% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  85.17% |   0.89% |  91.13% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  84.50% |   0.82% |  90.63% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  84.50% |   0.82% |  90.63% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  92.65% |   9.30% |  96.35% | r45211 GA best_f1        platt

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 83±58 neurons | 59±33 bits
    GA Neurons  : 236±4 neurons | 66±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.93±0.14 92.54±0.09 |15.64±0.40  9.08±0.15 |95.64±0.09 96.28±0.05
    fixed_05             |80.88±0.81 84.58±0.47 | 1.08±0.08  0.88±0.06 |87.78±0.68 90.69±0.35
    platt                |89.74±0.48 92.54±0.08 |11.43±0.21  9.29±0.06 |94.76±0.29 96.29±0.04
    beta                 |90.55±0.42 92.39±0.08 |16.71±3.60 11.46±0.21 |95.47±0.08 96.28±0.05
    empirical            |90.83±0.03 92.43±0.10 |17.03±1.75 10.84±0.11 |95.64±0.09 96.28±0.05
    empirical_cumulative |90.94±0.13 92.52±0.10 |15.50±0.62  8.13±0.54 |95.64±0.09 96.24±0.07
    val_cal              |90.94±0.13 92.56±0.07 |15.50±0.62  8.91±0.32 |95.64±0.09 96.29±0.04

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 83±58 neurons | 59±33 bits
    GA Neurons  : 171±100 neurons | 68±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.93±0.14 92.60±0.05 |15.64±0.40  9.06±0.19 |95.64±0.09 96.32±0.03
    fixed_05             |80.88±0.81 84.51±0.54 | 1.08±0.08  0.96±0.16 |87.78±0.68 90.64±0.41
    platt                |89.74±0.48 92.60±0.05 |11.43±0.21  9.16±0.22 |94.76±0.29 96.32±0.03
    beta                 |90.55±0.42 92.49±0.08 |16.71±3.60 10.74±1.38 |95.47±0.08 96.31±0.03
    empirical            |90.83±0.03 92.48±0.05 |17.03±1.75 11.01±0.40 |95.64±0.09 96.32±0.03
    empirical_cumulative |90.94±0.13 92.57±0.06 |15.50±0.62  7.81±0.15 |95.64±0.09 96.26±0.03
    val_cal              |90.94±0.13 92.61±0.05 |15.50±0.62  8.83±0.23 |95.64±0.09 96.32±0.03

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 133±29 neurons | 64±0 bits
    GA Neurons  : 233±10 neurons | 65±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.89±0.06 92.53±0.06 |14.25±0.59  8.98±0.14 |94.98±0.04 96.28±0.03
    fixed_05             |82.04±0.07 84.45±0.56 | 1.11±0.06  0.88±0.06 |88.74±0.05 90.59±0.42
    platt                |89.88±0.05 92.51±0.06 |12.88±0.27  9.29±0.02 |94.92±0.02 96.28±0.04
    beta                 |89.82±0.06 92.35±0.04 |16.20±0.10 11.46±0.24 |95.04±0.03 96.26±0.03
    empirical            |89.70±0.08 92.35±0.08 |17.56±0.60 11.53±0.26 |95.04±0.03 96.26±0.03
    empirical_cumulative |89.86±0.05 92.51±0.10 |12.22±0.26  7.66±0.45 |94.87±0.04 96.22±0.07
    val_cal              |89.90±0.04 92.55±0.06 |14.26±0.10  8.46±0.43 |94.99±0.03 96.27±0.03

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 83±58 neurons | 59±33 bits
    GA Neurons  : 170±99 neurons | 67±7 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.93±0.14 92.57±0.08 |15.64±0.40  9.24±0.02 |95.64±0.09 96.31±0.04
    fixed_05             |80.88±0.81 84.47±0.37 | 1.08±0.08  0.97±0.18 |87.78±0.68 90.62±0.28
    platt                |89.74±0.48 92.58±0.07 |11.43±0.21  9.18±0.14 |94.76±0.29 96.31±0.05
    beta                 |90.55±0.42 92.45±0.05 |16.71±3.60 10.90±1.18 |95.47±0.08 96.30±0.03
    empirical            |90.83±0.03 92.44±0.08 |17.03±1.75 11.06±0.54 |95.64±0.09 96.30±0.04
    empirical_cumulative |90.94±0.13 92.55±0.06 |15.50±0.62  8.48±0.38 |95.64±0.09 96.27±0.03
    val_cal              |90.94±0.13 92.58±0.08 |15.50±0.62  8.97±0.14 |95.64±0.09 96.30±0.04

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 233±29 neurons | 64±0 bits
    GA Neurons  : 236±4 neurons | 65±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.11±0.27 92.52±0.07 |15.15±0.82  9.02±0.07 |95.15±0.17 96.27±0.04
    fixed_05             |81.95±0.12 84.80±0.56 | 1.19±0.05  0.91±0.03 |88.67±0.10 90.86±0.41
    platt                |90.00±0.21 92.53±0.06 |12.79±0.14  9.25±0.12 |94.98±0.12 96.28±0.03
    beta                 |90.11±0.28 92.38±0.06 |16.03±0.25 11.38±0.25 |95.20±0.15 96.27±0.03
    empirical            |89.87±0.31 92.39±0.05 |19.17±0.40 11.10±0.36 |95.21±0.19 96.27±0.03
    empirical_cumulative |90.03±0.26 92.52±0.09 |13.05±0.83  7.87±0.33 |95.01±0.17 96.23±0.06
    val_cal              |90.13±0.27 92.54±0.06 |15.54±0.71  8.87±0.29 |95.18±0.19 96.28±0.03


## XDS-ciciot-64b-Wa-500n34b  (2 flows × 2 phases, seeds: [45198, 45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.45% |  10.28% |  96.27% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  92.45% |  10.28% |  96.27% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  92.44% |   9.44% |  96.24% | r45211 GA best_f1        platt
    Best F1 (FPR<6%)         |  82.43% |   3.77% |  89.26% | r45198 GS best_fpr       beta
    Best F1 (FPR<5%)         |  82.43% |   3.77% |  89.26% | r45198 GS best_fpr       beta
    Best F1 (FPR<4%)         |  82.43% |   3.77% |  89.26% | r45198 GS best_fpr       beta
    Best FPR (any F1)        |  81.44% |   0.68% |  88.22% | r45211 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  81.44% |   0.68% |  88.22% | r45211 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  92.45% |  10.28% |  96.27% | r45211 GA best_f1        val_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 100±0 neurons | 31±4 bits
    GA Neurons  : 306±274 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.79±0.36 92.27±0.07 |16.83±0.48 10.59±0.75 |95.61±0.18 96.19±0.01
    fixed_05             |79.82±0.27 80.98±0.66 | 0.93±0.14  0.80±0.17 |86.88±0.22 87.84±0.53
    platt                |89.33±1.23 92.18±0.16 |11.35±0.41  9.84±0.75 |94.51±0.72 96.11±0.06
    beta                 |90.19±1.17 92.09±0.10 |20.63±5.44 13.07±0.87 |95.47±0.38 96.17±0.02
    empirical            |90.70±0.49 92.11±0.04 |18.02±2.16 10.29±3.22 |95.61±0.17 96.09±0.14
    empirical_cumulative |90.79±0.36 92.25±0.04 |16.83±0.48  9.99±1.59 |95.61±0.18 96.15±0.04
    val_cal              |90.79±0.36 92.28±0.07 |16.83±0.48 10.45±0.77 |95.61±0.18 96.18±0.01

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 100±0 neurons | 31±4 bits
    GA Neurons  : 300±264 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.79±0.36 92.32±0.14 |16.83±0.48 10.83±0.40 |95.61±0.18 96.22±0.06
    fixed_05             |79.82±0.27 80.88±0.52 | 0.93±0.14  0.81±0.14 |86.88±0.22 87.76±0.42
    platt                |89.33±1.23 92.25±0.26 |11.35±0.41  9.90±0.65 |94.51±0.72 96.15±0.12
    beta                 |90.19±1.17 92.15±0.18 |20.63±5.44 13.01±0.95 |95.47±0.38 96.20±0.07
    empirical            |90.70±0.49 92.23±0.13 |18.02±2.16 10.68±2.67 |95.61±0.17 96.17±0.02
    empirical_cumulative |90.79±0.36 92.32±0.13 |16.83±0.48 10.17±1.34 |95.61±0.18 96.20±0.03
    val_cal              |90.79±0.36 92.34±0.16 |16.83±0.48 10.64±0.51 |95.61±0.18 96.22±0.07

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 102±138 neurons | 31±4 bits
    GA Neurons  : 306±272 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.39±3.70 92.20±0.15 |14.73±5.02 10.26±0.27 |93.39±2.61 96.14±0.07
    fixed_05             |79.76±0.68 80.91±0.76 | 1.14±0.35  0.73±0.08 |86.85±0.55 87.78±0.62
    platt                |87.11±3.30 92.20±0.14 |11.24±0.04  9.61±0.45 |93.07±2.14 96.11±0.06
    beta                 |86.22±5.37 92.05±0.15 |10.09±8.94 12.84±0.57 |92.21±4.18 96.14±0.06
    empirical            |87.10±3.32 92.07±0.01 |17.35±7.79 10.07±2.94 |93.37±2.53 96.06±0.10
    empirical_cumulative |87.31±3.58 92.16±0.15 |12.50±1.86  9.11±0.30 |93.24±2.39 96.08±0.08
    val_cal              |87.40±3.71 92.21±0.16 |13.89±3.79 10.03±0.19 |93.36±2.56 96.13±0.08

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 100±0 neurons | 31±4 bits
    GA Neurons  : 296±255 neurons | 32±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.79±0.36 92.28±0.12 |16.83±0.48 11.38±0.51 |95.61±0.18 96.22±0.05
    fixed_05             |79.82±0.27 80.96±0.70 | 0.93±0.14  0.80±0.14 |86.88±0.22 87.83±0.57
    platt                |89.33±1.23 91.77±0.17 |11.35±0.41  9.71±0.47 |94.51±0.72 95.88±0.11
    beta                 |90.19±1.17 92.14±0.16 |20.63±5.44 12.91±0.77 |95.47±0.38 96.19±0.06
    empirical            |90.70±0.49 92.24±0.14 |18.02±2.16 11.68±1.12 |95.61±0.17 96.21±0.04
    empirical_cumulative |90.79±0.36 92.28±0.11 |16.83±0.48 11.31±0.41 |95.61±0.18 96.22±0.05
    val_cal              |90.79±0.36 92.28±0.11 |16.83±0.48 11.39±0.29 |95.61±0.18 96.22±0.05

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.88±0.18 92.23±0.12 |17.12±1.57 11.44±1.95 |95.12±0.17 96.19±0.00
    fixed_05             |80.31±0.15 81.19±0.36 | 1.04±0.02  0.82±0.19 |87.31±0.12 88.02±0.28
    platt                |89.21±0.07 91.78±0.72 |11.43±0.01  9.82±0.73 |94.45±0.04 95.89±0.38
    beta                 |89.83±0.11 92.06±0.13 |17.06±0.04 13.41±1.35 |95.08±0.06 96.17±0.02
    empirical            |88.40±0.94 92.12±0.04 |28.59±3.22 10.45±3.44 |94.92±0.31 96.09±0.15
    empirical_cumulative |89.79±0.05 92.21±0.09 |14.90±0.06 10.81±2.75 |94.96±0.03 96.16±0.04
    val_cal              |89.89±0.17 92.24±0.13 |16.81±2.00 11.46±2.20 |95.11±0.19 96.20±0.01


## XDS-ciciot-64b-Wb-250n100b  (3 flows × 2 phases, seeds: [8198, 45198, 45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.52% |   7.82% |  96.23% | r45211 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  92.52% |   7.82% |  96.23% | r45211 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  92.52% |   7.82% |  96.23% | r45211 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  91.90% |   5.57% |  95.80% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<5%)         |  91.39% |   3.72% |  95.43% | r45211 GA best_ce        empirical_cumulative
    Best F1 (FPR<4%)         |  91.39% |   3.72% |  95.43% | r45211 GA best_ce        empirical_cumulative
    Best FPR (any F1)        |  83.97% |   0.81% |  90.23% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  83.97% |   0.81% |  90.23% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  92.45% |   9.50% |  96.25% | r45211 GA best_ce        platt

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 167±104 neurons | 48±0 bits
    GA Neurons  : 90±48 neurons | 50±9 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.61±0.67 92.15±0.27 |15.28±0.98  9.72±1.18 |95.44±0.34 96.09±0.13
    fixed_05             |81.49±0.02 82.58±1.30 | 1.09±0.05  0.96±0.14 |88.29±0.02 89.14±1.01
    platt                |90.18±0.44 92.14±0.29 |11.17±0.48 10.21±0.74 |95.02±0.24 96.10±0.14
    beta                 |90.52±0.52 91.93±0.32 |14.63±1.40 12.72±1.21 |95.37±0.24 96.08±0.13
    empirical            |90.40±0.86 91.97±0.32 |18.13±3.34 12.36±1.34 |95.46±0.33 96.08±0.13
    empirical_cumulative |88.56±0.25 91.86±0.21 | 6.92±0.86  7.43±1.65 |93.81±0.19 95.84±0.11
    val_cal              |90.63±0.66 92.17±0.28 |14.64±0.86  9.68±0.46 |95.42±0.34 96.10±0.14

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 167±104 neurons | 48±0 bits
    GA Neurons  : 91±48 neurons | 53±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.61±0.67 92.33±0.09 |15.28±0.98  9.52±1.10 |95.44±0.34 96.18±0.03
    fixed_05             |81.49±0.02 83.08±0.82 | 1.09±0.05  0.95±0.14 |88.29±0.02 89.54±0.63
    platt                |90.18±0.44 92.34±0.09 |11.17±0.48  9.79±0.04 |95.02±0.24 96.20±0.05
    beta                 |90.52±0.52 92.15±0.09 |14.63±1.40 12.09±0.36 |95.37±0.24 96.17±0.04
    empirical            |90.40±0.86 92.20±0.13 |18.13±3.34 11.64±0.71 |95.46±0.33 96.18±0.05
    empirical_cumulative |88.56±0.25 91.89±0.16 | 6.92±0.86  6.73±1.22 |93.81±0.19 95.84±0.12
    val_cal              |90.63±0.66 92.35±0.08 |14.64±0.86  9.56±0.24 |95.42±0.34 96.19±0.05

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 5±0 neurons | 18±20 bits
    GA Neurons  : 5±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |78.46±8.35 75.44±8.10 | 9.94±8.32 36.11±21.97 |86.03±6.73 87.60±3.43
    fixed_05             |70.88±10.24 72.50±8.55 | 0.96±0.11 27.05±22.21 |77.67±11.15 83.58±4.77
    platt                |64.69±21.34 72.20±10.85 |64.85±46.61 50.44±33.48 |88.83±4.29 88.46±2.67
    beta                 |73.67±14.16 72.68±9.59 |37.18±39.40 39.52±32.81 |87.71±5.63 86.67±4.17
    empirical            |64.70±21.35 72.33±10.79 |64.78±46.73 50.20±33.90 |88.83±4.29 88.56±2.62
    empirical_cumulative |74.32±9.48 67.22±13.83 | 1.62±1.07  0.59±0.89 |81.24±9.29 73.13±14.78
    val_cal              |78.46±8.35 75.44±8.10 | 9.94±8.32 36.11±21.97 |86.03±6.73 87.60±3.43

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 167±104 neurons | 48±0 bits
    GA Neurons  : 90±48 neurons | 50±9 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.61±0.67 92.15±0.27 |15.28±0.98  9.72±1.18 |95.44±0.34 96.09±0.13
    fixed_05             |81.49±0.02 82.58±1.30 | 1.09±0.05  0.96±0.14 |88.29±0.02 89.14±1.01
    platt                |90.18±0.44 92.14±0.29 |11.17±0.48 10.21±0.74 |95.02±0.24 96.10±0.14
    beta                 |90.52±0.52 91.93±0.32 |14.63±1.40 12.72±1.21 |95.37±0.24 96.08±0.13
    empirical            |90.40±0.86 91.97±0.32 |18.13±3.34 12.36±1.34 |95.46±0.33 96.08±0.13
    empirical_cumulative |88.56±0.25 91.86±0.21 | 6.92±0.86  7.43±1.65 |93.81±0.19 95.84±0.11
    val_cal              |90.63±0.66 92.17±0.28 |14.64±0.86  9.68±0.46 |95.42±0.34 96.10±0.14

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±71 neurons | 64±0 bits
    GA Neurons  : 152±94 neurons | 54±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.02±0.11 92.27±0.29 |15.10±0.44  9.66±0.92 |95.10±0.05 96.15±0.16
    fixed_05             |81.91±0.00 84.61±0.97 | 1.12±0.13  1.16±0.51 |88.64±0.01 90.72±0.75
    platt                |89.99±0.14 92.27±0.28 |12.88±0.17  9.63±0.12 |94.98±0.08 96.15±0.16
    beta                 |90.01±0.10 92.12±0.21 |16.15±0.22 11.36±0.93 |95.14±0.05 96.13±0.15
    empirical            |89.72±0.14 92.10±0.24 |19.30±0.67 11.65±0.60 |95.13±0.06 96.13±0.15
    empirical_cumulative |87.55±0.33 91.26±0.25 | 5.41±0.11  4.55±0.74 |93.07±0.22 95.38±0.15
    val_cal              |90.04±0.12 92.29±0.31 |14.65±0.49  9.07±1.08 |95.09±0.05 96.15±0.15


## XDS-ciciot-64b-Wb-500n34b  (2 flows × 2 phases, seeds: [45198, 45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.18% |  10.78% |  96.14% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  92.18% |  10.78% |  96.14% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  92.10% |   9.55% |  96.06% | r45198 GA best_f1        platt
    Best F1 (FPR<6%)         |  83.85% |   3.26% |  90.31% | r45211 GS best_fpr       empirical_cumulative
    Best F1 (FPR<5%)         |  83.85% |   3.26% |  90.31% | r45211 GS best_fpr       empirical_cumulative
    Best F1 (FPR<4%)         |  83.85% |   3.26% |  90.31% | r45211 GS best_fpr       empirical_cumulative
    Best FPR (any F1)        |  80.50% |   0.79% |  87.45% | r45211 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  80.50% |   0.79% |  87.45% | r45211 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  92.09% |  12.39% |  96.15% | r45211 GA best_acc       empirical

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±212 neurons | 33±1 bits
    GA Neurons  : 352±207 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.81±0.09 92.16±0.02 |18.83±0.14 10.86±0.25 |95.71±0.04 96.13±0.00
    fixed_05             |80.17±0.01 80.45±0.01 | 0.96±0.00  0.93±0.03 |87.19±0.01 87.42±0.01
    platt                |89.76±0.24 91.91±0.28 |10.85±0.13  9.61±0.08 |94.75±0.13 95.95±0.15
    beta                 |90.62±0.15 91.93±0.01 |16.29±0.22 13.65±0.22 |95.49±0.10 96.11±0.00
    empirical            |90.81±0.09 92.03±0.08 |18.83±0.14 12.80±0.59 |95.71±0.04 96.13±0.02
    empirical_cumulative |89.04±0.02 92.02±0.03 | 9.25±0.28  9.48±0.56 |94.24±0.00 96.01±0.04
    val_cal              |90.81±0.09 92.17±0.03 |18.83±0.14 10.92±0.20 |95.71±0.04 96.14±0.01

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±212 neurons | 33±1 bits
    GA Neurons  : 352±207 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.81±0.09 92.16±0.02 |18.83±0.14 10.86±0.25 |95.71±0.04 96.13±0.00
    fixed_05             |80.17±0.01 80.45±0.01 | 0.96±0.00  0.93±0.03 |87.19±0.01 87.42±0.01
    platt                |89.76±0.24 91.91±0.28 |10.85±0.13  9.61±0.08 |94.75±0.13 95.95±0.15
    beta                 |90.62±0.15 91.93±0.01 |16.29±0.22 13.65±0.22 |95.49±0.10 96.11±0.00
    empirical            |90.81±0.09 92.03±0.08 |18.83±0.14 12.80±0.59 |95.71±0.04 96.13±0.02
    empirical_cumulative |89.04±0.02 92.02±0.03 | 9.25±0.28  9.48±0.56 |94.24±0.00 96.01±0.04
    val_cal              |90.81±0.09 92.17±0.03 |18.83±0.14 10.92±0.20 |95.71±0.04 96.14±0.01

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 5±0 neurons | 28±0 bits
    GA Neurons  : 5±0 neurons | 28±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |80.21±9.99 76.37±11.93 |13.56±5.14 36.69±38.04 |87.90±7.43 89.09±3.37
    fixed_05             |69.67±14.52 69.43±15.66 | 1.10±0.22 16.94±22.84 |76.05±15.52 78.13±13.19
    platt                |66.76±29.02 65.52±27.27 |54.99±63.65 54.94±63.72 |89.58±5.05 88.74±3.86
    beta                 |80.17±9.94 76.37±11.94 |14.01±4.51 36.87±37.78 |87.90±7.43 89.11±3.39
    empirical            |66.76±29.02 76.37±11.94 |54.96±63.69 36.87±37.78 |89.58±5.05 89.11±3.39
    empirical_cumulative |74.38±13.38 65.44±21.42 | 2.22±1.48  0.63±0.33 |81.02±13.13 70.55±24.01
    val_cal              |80.21±9.99 76.37±11.94 |13.59±5.10 36.87±37.78 |87.90±7.43 89.11±3.39

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±212 neurons | 33±1 bits
    GA Neurons  : 352±207 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.81±0.09 92.12±0.07 |18.83±0.14 11.03±0.48 |95.71±0.04 96.12±0.02
    fixed_05             |80.17±0.01 80.46±0.00 | 0.96±0.00  0.99±0.04 |87.19±0.01 87.43±0.00
    platt                |89.76±0.24 91.64±0.11 |10.85±0.13  9.74±0.11 |94.75±0.13 95.80±0.06
    beta                 |90.62±0.15 91.87±0.08 |16.29±0.22 14.03±0.32 |95.49±0.10 96.09±0.03
    empirical            |90.81±0.09 92.03±0.09 |18.83±0.14 12.65±0.37 |95.71±0.04 96.12±0.04
    empirical_cumulative |89.04±0.02 92.03±0.03 | 9.25±0.28 10.11±0.33 |94.24±0.00 96.03±0.00
    val_cal              |90.81±0.09 92.13±0.07 |18.83±0.14 10.93±0.21 |95.71±0.04 96.12±0.03

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 496±6 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.98±0.04 91.78±0.11 |18.11±0.16 12.97±0.78 |95.22±0.03 96.00±0.03
    fixed_05             |80.32±0.16 80.69±0.08 | 1.07±0.07  1.02±0.01 |87.32±0.14 87.62±0.07
    platt                |89.25±0.12 91.02±0.35 |11.34±0.13 10.37±0.09 |94.46±0.06 95.47±0.20
    beta                 |89.92±0.02 91.72±0.12 |17.02±0.02 14.05±1.29 |95.13±0.01 96.01±0.02
    empirical            |88.87±0.28 91.64±0.07 |26.88±0.80 15.28±0.47 |95.07±0.10 96.01±0.02
    empirical_cumulative |86.86±0.85 90.02±0.33 | 6.99±1.01  8.53±0.99 |92.69±0.63 94.80±0.24
    val_cal              |89.99±0.03 91.79±0.10 |17.95±0.38 12.60±0.33 |95.22±0.04 95.99±0.04


## XDS-ciciot-64b-Wc-250n100b  (3 flows × 2 phases, seeds: [8198, 45198, 45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.61% |   9.21% |  96.33% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  92.61% |   9.21% |  96.33% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  92.61% |   9.21% |  96.33% | r45211 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  92.16% |   4.61% |  95.92% | r45211 GA best_fpr       empirical_cumulative
    Best F1 (FPR<5%)         |  92.16% |   4.61% |  95.92% | r45211 GA best_fpr       empirical_cumulative
    Best F1 (FPR<4%)         |  87.44% |   1.11% |  92.76% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  83.48% |   0.78% |  89.85% | r45198 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  83.48% |   0.78% |  89.85% | r45198 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  92.61% |   9.21% |  96.33% | r45211 GA best_acc       val_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 233±29 neurons | 64±0 bits
    GA Neurons  : 237±17 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.40±0.12 92.23±0.19 |15.43±0.50  9.75±1.13 |95.33±0.05 96.13±0.08
    fixed_05             |81.89±0.06 85.11±2.02 | 1.10±0.04  0.97±0.13 |88.62±0.05 91.06±1.47
    platt                |90.24±0.12 92.22±0.20 |12.59±0.22  9.82±1.22 |95.11±0.06 96.13±0.08
    beta                 |90.39±0.12 92.10±0.27 |15.80±0.23 11.88±1.54 |95.34±0.06 96.14±0.10
    empirical            |90.04±0.17 91.98±0.14 |20.15±0.48 12.92±0.55 |95.35±0.08 96.11±0.07
    empirical_cumulative |89.96±0.14 91.98±0.15 |10.61±0.46  6.62±1.83 |94.86±0.10 95.88±0.08
    val_cal              |90.41±0.12 92.25±0.19 |15.47±0.74  9.07±1.43 |95.34±0.08 96.12±0.10

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 50±0 neurons | 53±24 bits
    GA Neurons  : 184±111 neurons | 55±10 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.73±0.58 92.39±0.18 |14.68±1.26  9.51±0.35 |95.48±0.30 96.22±0.09
    fixed_05             |81.12±1.05 83.47±1.04 | 1.08±0.23  0.89±0.06 |87.98±0.90 89.84±0.80
    platt                |90.24±0.38 92.39±0.17 |11.71±1.13  9.85±0.25 |95.08±0.19 96.22±0.09
    beta                 |90.54±0.44 92.24±0.12 |15.74±1.83 11.76±0.71 |95.42±0.32 96.21±0.08
    empirical            |90.39±0.74 92.27±0.18 |18.86±3.69 11.50±0.52 |95.49±0.29 96.22±0.09
    empirical_cumulative |90.43±0.77 92.23±0.18 |11.89±0.07  7.89±0.44 |95.19±0.44 96.07±0.10
    val_cal              |90.73±0.58 92.40±0.19 |14.48±0.94  9.63±0.41 |95.48±0.30 96.23±0.09

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 100±71 neurons | 64±0 bits
    GA Neurons  : 164±102 neurons | 56±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.81±0.42 92.22±0.26 |14.56±0.29  8.64±0.84 |94.95±0.25 96.09±0.15
    fixed_05             |81.91±0.06 84.37±2.41 | 1.07±0.09  0.90±0.13 |88.63±0.05 90.49±1.79
    platt                |89.81±0.41 92.21±0.25 |13.07±0.68  9.20±0.76 |94.88±0.21 96.11±0.13
    beta                 |89.76±0.43 92.05±0.28 |15.87±0.26 11.35±1.01 |94.98±0.25 96.09±0.14
    empirical            |89.64±0.35 92.02±0.09 |17.60±1.11 11.64±1.04 |95.00±0.24 96.09±0.09
    empirical_cumulative |89.43±0.48 91.98±0.19 | 9.72±0.41  5.76±1.00 |94.50±0.31 95.85±0.09
    val_cal              |89.82±0.43 92.24±0.26 |14.10±0.26  8.67±0.50 |94.94±0.26 96.10±0.13

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 67±29 neurons | 48±28 bits
    GA Neurons  : 185±109 neurons | 56±9 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.70±0.61 92.39±0.18 |15.54±2.76  9.86±0.79 |95.51±0.29 96.23±0.08
    fixed_05             |80.44±1.10 83.59±0.84 | 1.00±0.21  0.88±0.04 |87.41±0.93 89.93±0.65
    platt                |89.87±0.97 92.32±0.26 |11.61±1.13  9.86±0.23 |94.84±0.57 96.19±0.15
    beta                 |90.53±0.44 92.27±0.12 |17.47±2.04 11.77±0.70 |95.49±0.26 96.22±0.06
    empirical            |90.60±0.50 92.28±0.17 |17.40±1.17 11.71±0.49 |95.53±0.26 96.23±0.08
    empirical_cumulative |90.40±0.80 92.25±0.17 |12.88±1.69  8.46±1.41 |95.22±0.42 96.10±0.10
    val_cal              |90.70±0.61 92.40±0.18 |15.54±2.76 10.07±0.89 |95.51±0.29 96.24±0.07

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 250±0 neurons | 64±0 bits
    GA Neurons  : 237±17 neurons | 64±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.40±0.12 92.23±0.19 |15.43±0.50  9.75±1.13 |95.33±0.05 96.13±0.08
    fixed_05             |81.89±0.06 85.11±2.02 | 1.10±0.04  0.97±0.13 |88.62±0.05 91.06±1.47
    platt                |90.24±0.12 92.22±0.20 |12.59±0.22  9.82±1.22 |95.11±0.06 96.13±0.08
    beta                 |90.39±0.12 92.10±0.27 |15.80±0.23 11.88±1.54 |95.34±0.06 96.14±0.10
    empirical            |90.04±0.17 91.98±0.14 |20.15±0.48 12.92±0.55 |95.35±0.08 96.11±0.07
    empirical_cumulative |89.96±0.14 91.98±0.15 |10.61±0.46  6.62±1.83 |94.86±0.10 95.88±0.08
    val_cal              |90.41±0.12 92.25±0.19 |15.47±0.74  9.07±1.43 |95.34±0.08 96.12±0.10


## XDS-ciciot-64b-Wc-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.38% |  11.41% |  96.27% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  92.38% |  11.41% |  96.27% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  92.33% |   9.91% |  96.19% | r45211 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  81.62% |   0.85% |  88.38% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  81.62% |   0.85% |  88.38% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  81.62% |   0.85% |  88.38% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  81.62% |   0.85% |  88.38% | r45211 GA best_ce        fixed_05
    Best FPR (F1>80%)        |  81.62% |   0.85% |  88.38% | r45211 GA best_ce        fixed_05
    Best Acc (any FPR)       |  92.38% |  11.41% |  96.27% | r45211 GA best_acc       val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 490±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.30     92.29   |   19.09     10.60   |   95.44     96.20  
    fixed_05             |  80.28     81.62   |    0.95      0.85   |   87.28     88.38  
    platt                |  89.51     92.27   |   11.08      9.51   |   94.61     96.15  
    beta                 |  90.29     92.10   |   16.55     13.38   |   95.32     96.19  
    empirical            |  90.27     92.25   |   21.43      9.46   |   95.54     96.13  
    empirical_cumulative |  89.90     92.30   |   12.60      9.61   |   94.91     96.17  
    val_cal              |  90.34     92.33   |   19.55      9.91   |   95.48     96.19  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 470±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.48     92.35   |   19.61     11.08   |   95.56     96.25  
    fixed_05             |  79.66     80.57   |    0.93      0.88   |   86.75     87.52  
    platt                |  88.93     92.15   |   11.32      9.78   |   94.27     96.09  
    beta                 |  90.27     92.11   |   17.50     14.01   |   95.35     96.22  
    empirical            |  90.48     92.31   |   19.76     11.94   |   95.57     96.26  
    empirical_cumulative |  89.95     92.33   |   13.43      9.96   |   94.98     96.19  
    val_cal              |  90.48     92.37   |   19.76     10.63   |   95.57     96.24  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 477±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  89.66     92.23   |   16.93     10.46   |   94.98     96.16  
    fixed_05             |  80.59     81.24   |    1.18      0.91   |   87.56     88.07  
    platt                |  89.25     92.20   |   11.06      9.37   |   94.45     96.10  
    beta                 |  89.58     92.12   |   16.22     12.98   |   94.90     96.19  
    empirical            |  89.44     91.88   |   19.01     14.68   |   94.96     96.12  
    empirical_cumulative |  89.28     92.13   |   11.21      8.48   |   94.48     96.04  
    val_cal              |  89.66     92.25   |   16.93     10.79   |   94.98     96.18  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 28±0 bits
    GA Neurons  : 452±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.48     92.37   |   19.61     11.38   |   95.56     96.27  
    fixed_05             |  79.66     80.77   |    0.93      0.90   |   86.75     87.68  
    platt                |  88.93     91.73   |   11.32      9.85   |   94.27     95.86  
    beta                 |  90.27     92.10   |   17.50     14.06   |   95.35     96.22  
    empirical            |  90.48     92.05   |   19.76     14.46   |   95.57     96.20  
    empirical_cumulative |  89.95     92.35   |   13.43     10.98   |   94.98     96.24  
    val_cal              |  90.48     92.38   |   19.76     11.41   |   95.57     96.27  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 490±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.30     92.29   |   19.09     10.60   |   95.44     96.20  
    fixed_05             |  80.28     81.62   |    0.95      0.85   |   87.28     88.38  
    platt                |  89.51     92.27   |   11.08      9.51   |   94.61     96.15  
    beta                 |  90.29     92.10   |   16.55     13.38   |   95.32     96.19  
    empirical            |  90.27     92.25   |   21.43      9.46   |   95.54     96.13  
    empirical_cumulative |  89.90     92.30   |   12.60      9.61   |   94.91     96.17  
    val_cal              |  90.34     92.33   |   19.55      9.91   |   95.48     96.19  


## XDS-ciciot-96b-Wa-250n100b  (3 flows × 2 phases, seeds: [8198, 45198, 45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.96% |   8.81% |  96.51% | r45198 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  92.96% |   8.81% |  96.51% | r45198 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  92.96% |   8.81% |  96.51% | r45198 GA best_f1        val_cal
    Best F1 (FPR<6%)         |  84.89% |   0.98% |  90.93% | r45198 GA best_fpr       fixed_05
    Best F1 (FPR<5%)         |  84.89% |   0.98% |  90.93% | r45198 GA best_fpr       fixed_05
    Best F1 (FPR<4%)         |  84.89% |   0.98% |  90.93% | r45198 GA best_fpr       fixed_05
    Best FPR (any F1)        |  84.56% |   0.83% |  90.68% | r45211 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  84.56% |   0.83% |  90.68% | r45211 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  92.94% |   9.91% |  96.53% | r8198 GA best_f1        train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 133±104 neurons | 53±24 bits
    GA Neurons  : 166±101 neurons | 47±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.96±0.17 92.93±0.03 |15.49±1.37  9.24±0.70 |95.65±0.10 96.51±0.03
    fixed_05             |81.12±0.86 82.65±1.39 | 1.12±0.20  1.03±0.21 |87.98±0.72 89.21±1.11
    platt                |90.19±0.61 92.83±0.16 |11.62±0.50  8.93±0.22 |95.04±0.38 96.44±0.09
    beta                 |90.87±0.25 92.79±0.02 |15.43±1.22 11.17±0.53 |95.59±0.19 96.49±0.02
    empirical            |90.85±0.23 92.85±0.00 |16.94±1.13 10.67±0.62 |95.65±0.11 96.50±0.02
    empirical_cumulative |90.96±0.16 92.92±0.03 |15.01±1.69  8.87±1.03 |95.63±0.11 96.49±0.05
    val_cal              |90.97±0.16 92.93±0.03 |15.09±1.57  9.32±0.64 |95.63±0.11 96.51±0.02

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 67±29 neurons | 53±24 bits
    GA Neurons  : 165±99 neurons | 47±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.99±0.20 92.93±0.03 |14.79±0.71  9.20±0.63 |95.63±0.09 96.51±0.03
    fixed_05             |81.28±0.57 82.66±1.38 | 1.20±0.14  1.03±0.22 |88.13±0.47 89.21±1.10
    platt                |90.17±0.64 92.83±0.16 |11.59±0.53  8.92±0.22 |95.02±0.40 96.44±0.09
    beta                 |90.73±0.23 92.79±0.02 |16.47±2.95 11.15±0.49 |95.56±0.15 96.49±0.03
    empirical            |90.84±0.23 92.85±0.00 |17.00±1.17 10.63±0.55 |95.65±0.11 96.50±0.02
    empirical_cumulative |90.99±0.19 92.92±0.04 |14.36±0.92  8.82±0.95 |95.62±0.09 96.49±0.05
    val_cal              |90.99±0.19 92.94±0.03 |14.44±0.78  9.28±0.57 |95.62±0.09 96.51±0.02

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 150±50 neurons | 64±0 bits
    GA Neurons  : 226±42 neurons | 65±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.82±0.17 92.64±0.13 |14.32±0.30  9.25±0.58 |94.95±0.09 96.35±0.05
    fixed_05             |81.96±0.02 84.49±0.44 | 1.12±0.05  0.96±0.12 |88.67±0.01 90.63±0.33
    platt                |89.81±0.15 92.63±0.12 |12.71±0.16  9.46±0.62 |94.87±0.09 96.35±0.06
    beta                 |89.75±0.16 92.47±0.11 |16.25±0.34 11.61±0.65 |95.00±0.08 96.33±0.05
    empirical            |89.46±0.09 92.48±0.14 |19.12±0.46 11.50±0.50 |94.98±0.06 96.33±0.06
    empirical_cumulative |89.80±0.15 92.60±0.14 |12.35±0.63  7.44±0.28 |94.85±0.07 96.26±0.08
    val_cal              |89.83±0.17 92.66±0.12 |13.75±1.14  8.76±0.80 |94.93±0.11 96.34±0.06

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 233±29 neurons | 32±0 bits
    GA Neurons  : 168±102 neurons | 47±13 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.82±0.20 92.90±0.03 |17.35±0.48  9.52±0.55 |95.65±0.09 96.50±0.03
    fixed_05             |80.05±0.18 82.59±1.31 | 0.98±0.04  1.06±0.19 |87.09±0.16 89.16±1.06
    platt                |89.49±0.19 92.81±0.13 |11.21±0.09  9.00±0.34 |94.61±0.11 96.43±0.07
    beta                 |90.74±0.32 92.75±0.06 |17.15±0.40 11.34±0.56 |95.60±0.16 96.47±0.03
    empirical            |90.79±0.19 92.81±0.07 |17.81±0.34 10.91±0.68 |95.65±0.09 96.49±0.04
    empirical_cumulative |90.82±0.21 92.89±0.04 |17.18±0.48  9.12±0.97 |95.64±0.09 96.48±0.05
    val_cal              |90.82±0.20 92.90±0.03 |17.23±0.55  9.48±0.50 |95.65±0.09 96.50±0.03

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 217±58 neurons | 64±0 bits
    GA Neurons  : 223±38 neurons | 65±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.21±0.15 92.56±0.14 |15.86±0.86  9.48±0.91 |95.24±0.10 96.31±0.06
    fixed_05             |82.00±0.09 84.43±0.17 | 1.09±0.07  0.96±0.10 |88.71±0.07 90.59±0.12
    platt                |90.16±0.11 92.58±0.11 |12.59±0.20  9.59±0.64 |95.07±0.06 96.33±0.04
    beta                 |90.22±0.16 92.45±0.12 |15.77±0.18 11.66±0.62 |95.24±0.08 96.32±0.05
    empirical            |89.94±0.21 92.43±0.10 |19.60±0.29 11.67±0.24 |95.27±0.10 96.31±0.05
    empirical_cumulative |90.18±0.10 92.52±0.14 |12.85±0.12  7.37±0.30 |95.09±0.06 96.22±0.09
    val_cal              |90.25±0.15 92.60±0.11 |14.80±0.71  8.61±0.81 |95.21±0.12 96.30±0.05


## XDS-ciciot-96b-Wa-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.71% |  10.41% |  96.42% | r45211 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  92.71% |  10.41% |  96.42% | r45211 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  92.62% |   9.35% |  96.34% | r45211 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  84.74% |   3.13% |  90.96% | r45211 GS best_fpr       beta
    Best F1 (FPR<5%)         |  84.74% |   3.13% |  90.96% | r45211 GS best_fpr       beta
    Best F1 (FPR<4%)         |  84.74% |   3.13% |  90.96% | r45211 GS best_fpr       beta
    Best FPR (any F1)        |  81.45% |   0.87% |  88.24% | r45211 GA best_ce        fixed_05
    Best FPR (F1>80%)        |  81.45% |   0.87% |  88.24% | r45211 GA best_ce        fixed_05
    Best Acc (any FPR)       |  92.68% |  11.03% |  96.43% | r45211 GA best_acc       empirical

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 112±0 neurons | 31±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  
    fixed_05             |  79.60     81.66   |    0.76      0.88   |   86.68     88.41  
    platt                |  89.87     92.43   |   11.40      8.99   |   94.84     96.22  
    beta                 |  90.83     92.59   |   17.50     12.06   |   95.66     96.41  
    empirical            |  90.97     92.68   |   16.61     11.03   |   95.70     96.43  
    empirical_cumulative |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  
    val_cal              |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 112±0 neurons | 31±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  
    fixed_05             |  79.60     81.66   |    0.76      0.88   |   86.68     88.41  
    platt                |  89.87     92.43   |   11.40      8.99   |   94.84     96.22  
    beta                 |  90.83     92.59   |   17.50     12.06   |   95.66     96.41  
    empirical            |  90.97     92.68   |   16.61     11.03   |   95.70     96.43  
    empirical_cumulative |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  
    val_cal              |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 28±0 bits
    GA Neurons  : 102±0 neurons | 31±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  86.89     92.58   |   10.42     10.10   |   92.93     96.34  
    fixed_05             |  80.67     82.74   |    0.87      1.02   |   87.60     89.29  
    platt                |  86.87     92.53   |   10.55      8.54   |   92.92     96.26  
    beta                 |  84.74     92.50   |    3.13     11.50   |   90.96     96.34  
    empirical            |  86.89     92.55   |   10.42     10.88   |   92.93     96.35  
    empirical_cumulative |  86.94     92.47   |    9.55      7.64   |   92.91     96.20  
    val_cal              |  86.94     92.58   |    9.55      9.63   |   92.91     96.33  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 112±0 neurons | 31±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  
    fixed_05             |  79.60     81.66   |    0.76      0.88   |   86.68     88.41  
    platt                |  89.87     92.43   |   11.40      8.99   |   94.84     96.22  
    beta                 |  90.83     92.59   |   17.50     12.06   |   95.66     96.41  
    empirical            |  90.97     92.68   |   16.61     11.03   |   95.70     96.43  
    empirical_cumulative |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  
    val_cal              |  91.04     92.71   |   14.99     10.41   |   95.67     96.42  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 495±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.02     92.62   |   18.99      9.39   |   95.28     96.34  
    fixed_05             |  80.40     81.45   |    1.04      0.87   |   87.38     88.24  
    platt                |  89.42     92.61   |   11.15      9.44   |   94.56     96.34  
    beta                 |  89.94     92.38   |   16.73     13.08   |   95.13     96.33  
    empirical            |  89.80     92.56   |   23.43      8.57   |   95.38     96.28  
    empirical_cumulative |  89.91     92.61   |   14.56      9.11   |   95.01     96.32  
    val_cal              |  90.02     92.62   |   18.84      9.35   |   95.28     96.34  


## XDS-ciciot-96b-Wb-250n100b  (3 flows × 2 phases, seeds: [8198, 45198, 45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.98% |   8.51% |  96.51% | r45198 GA best_ce        val_cal
    Best F1 (FPR<14%)        |  92.98% |   8.51% |  96.51% | r45198 GA best_ce        val_cal
    Best F1 (FPR<10%)        |  92.98% |   8.51% |  96.51% | r45198 GA best_ce        val_cal
    Best F1 (FPR<6%)         |  91.83% |   4.61% |  95.72% | r45198 GA best_ce        empirical_cumulative
    Best F1 (FPR<5%)         |  91.83% |   4.61% |  95.72% | r45198 GA best_ce        empirical_cumulative
    Best F1 (FPR<4%)         |  91.45% |   3.90% |  95.47% | r8198 GA best_ce        empirical_cumulative
    Best FPR (any F1)        |  80.21% |   0.84% |  87.21% | r45211 GS best_fpr       fixed_05
    Best FPR (F1>80%)        |  80.21% |   0.84% |  87.21% | r45211 GS best_fpr       fixed_05
    Best Acc (any FPR)       |  92.96% |   9.08% |  96.52% | r45198 GA best_ce        train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±50 neurons | 48±0 bits
    GA Neurons  : 195±44 neurons | 61±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.54±0.12 92.54±0.32 |17.14±0.90 10.16±0.78 |95.48±0.10 96.32±0.16
    fixed_05             |81.39±0.15 83.67±1.27 | 1.10±0.02  0.97±0.07 |88.21±0.12 90.00±0.97
    platt                |90.04±0.06 92.53±0.29 |11.49±0.04  9.94±0.55 |94.95±0.03 96.31±0.14
    beta                 |90.47±0.07 92.40±0.35 |15.45±0.30 12.03±0.92 |95.37±0.04 96.30±0.16
    empirical            |90.44±0.16 92.42±0.34 |19.43±1.23 11.79±0.92 |95.53±0.03 96.31±0.15
    empirical_cumulative |88.55±0.17 91.58±0.55 | 7.45±0.17  5.89±1.77 |93.84±0.11 95.62±0.37
    val_cal              |90.54±0.12 92.56±0.30 |17.37±1.36  9.84±0.78 |95.49±0.11 96.32±0.16

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±50 neurons | 48±0 bits
    GA Neurons  : 195±44 neurons | 61±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.54±0.12 92.54±0.32 |17.14±0.90 10.16±0.78 |95.48±0.10 96.32±0.16
    fixed_05             |81.39±0.15 83.67±1.27 | 1.10±0.02  0.97±0.07 |88.21±0.12 90.00±0.97
    platt                |90.04±0.06 92.53±0.29 |11.49±0.04  9.94±0.55 |94.95±0.03 96.31±0.14
    beta                 |90.47±0.07 92.40±0.35 |15.45±0.30 12.03±0.92 |95.37±0.04 96.30±0.16
    empirical            |90.44±0.16 92.42±0.34 |19.43±1.23 11.79±0.92 |95.53±0.03 96.31±0.15
    empirical_cumulative |88.55±0.17 91.58±0.55 | 7.45±0.17  5.89±1.77 |93.84±0.11 95.62±0.37
    val_cal              |90.54±0.12 92.56±0.30 |17.37±1.36  9.84±0.78 |95.49±0.11 96.32±0.16

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 5±0 neurons | 32±0 bits
    GA Neurons  : 5±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |83.80±6.89 78.35±5.51 |13.69±5.41 14.08±14.00 |90.82±4.65 86.96±3.06
    fixed_05             |74.74±9.81 76.35±4.60 | 1.11±0.45  4.54±5.54 |81.58±10.10 84.06±3.73
    platt                |73.91±23.98 75.79±9.93 |40.43±51.60 28.69±38.32 |90.99±4.36 88.05±1.27
    beta                 |83.75±6.85 75.45±9.67 |13.89±5.19 27.51±39.32 |90.80±4.63 87.67±1.21
    empirical            |73.91±23.98 75.79±9.93 |40.40±51.62 28.69±38.32 |90.99±4.35 88.05±1.27
    empirical_cumulative |79.35±8.31 75.58±8.18 | 3.21±0.71  2.11±2.32 |86.15±7.59 82.66±8.28
    val_cal              |83.81±6.89 78.35±5.51 |13.53±5.58 14.23±13.92 |90.81±4.65 86.98±3.07

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 200±50 neurons | 48±0 bits
    GA Neurons  : 216±50 neurons | 55±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.54±0.12 92.21±0.60 |17.14±0.90 11.47±1.60 |95.48±0.10 96.19±0.28
    fixed_05             |81.39±0.15 82.45±0.84 | 1.10±0.02  0.98±0.09 |88.21±0.12 89.05±0.67
    platt                |90.04±0.06 92.18±0.61 |11.49±0.04 10.32±0.57 |94.95±0.03 96.12±0.33
    beta                 |90.47±0.07 92.12±0.58 |15.45±0.30 12.82±1.32 |95.37±0.04 96.18±0.27
    empirical            |90.44±0.16 91.98±0.80 |19.43±1.23 14.17±3.84 |95.53±0.03 96.16±0.29
    empirical_cumulative |88.55±0.17 91.60±0.53 | 7.45±0.17  7.72±1.74 |93.84±0.11 95.70±0.31
    val_cal              |90.54±0.12 92.24±0.58 |17.37±1.36 10.87±1.00 |95.49±0.11 96.18±0.29

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 233±29 neurons | 64±0 bits
    GA Neurons  : 198±49 neurons | 61±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.25±0.11 92.66±0.29 |15.88±1.11  9.51±0.81 |95.27±0.11 96.36±0.14
    fixed_05             |81.88±0.10 84.63±1.33 | 1.12±0.01  0.95±0.07 |88.61±0.08 90.72±0.99
    platt                |90.16±0.08 92.66±0.30 |12.70±0.23  9.49±0.64 |95.07±0.04 96.36±0.15
    beta                 |90.24±0.15 92.54±0.28 |15.97±0.21 11.47±1.00 |95.26±0.08 96.36±0.13
    empirical            |90.03±0.20 92.53±0.31 |19.53±0.43 11.61±1.15 |95.31±0.11 96.36±0.13
    empirical_cumulative |87.92±0.53 91.44±0.39 | 5.64±0.60  4.40±0.43 |93.33±0.37 95.48±0.23
    val_cal              |90.28±0.11 92.67±0.29 |14.77±1.20  9.39±1.02 |95.23±0.11 96.37±0.13


## XDS-ciciot-96b-Wb-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.63% |   9.80% |  96.36% | r45211 GA best_f1        val_cal
    Best F1 (FPR<14%)        |  92.63% |   9.80% |  96.36% | r45211 GA best_f1        val_cal
    Best F1 (FPR<10%)        |  92.63% |   9.80% |  96.36% | r45211 GA best_f1        val_cal
    Best F1 (FPR<6%)         |  80.72% |   0.92% |  87.64% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<5%)         |  80.72% |   0.92% |  87.64% | r45211 GA best_ce        fixed_05
    Best F1 (FPR<4%)         |  80.72% |   0.92% |  87.64% | r45211 GA best_ce        fixed_05
    Best FPR (any F1)        |  80.00% |   0.81% |  87.03% | r45211 GA best_acc       fixed_05
    Best FPR (F1>80%)        |  80.00% |   0.81% |  87.03% | r45211 GA best_acc       fixed_05
    Best Acc (any FPR)       |  92.63% |   9.80% |  96.36% | r45211 GA best_f1        val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 32±0 bits
    GA Neurons  : 331±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.10     92.63   |   15.63      9.58   |   95.17     96.35  
    fixed_05             |  79.82     80.16   |    0.92      0.85   |   86.88     87.17  
    platt                |  89.53     92.57   |   11.06      9.06   |   94.62     96.30  
    beta                 |  90.08     92.41   |   17.02     12.17   |   95.22     96.32  
    empirical            |  89.56     92.48   |   24.12     11.58   |   95.29     96.33  
    empirical_cumulative |  88.67     92.49   |    9.28      8.69   |   94.01     96.24  
    val_cal              |  90.13     92.63   |   16.14      9.80   |   95.21     96.36  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 32±0 bits
    GA Neurons  : 331±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.10     92.63   |   15.63      9.58   |   95.17     96.35  
    fixed_05             |  79.82     80.16   |    0.92      0.85   |   86.88     87.17  
    platt                |  89.53     92.57   |   11.06      9.06   |   94.62     96.30  
    beta                 |  90.08     92.41   |   17.02     12.17   |   95.22     96.32  
    empirical            |  89.56     92.48   |   24.12     11.58   |   95.29     96.33  
    empirical_cumulative |  88.67     92.49   |    9.28      8.69   |   94.01     96.24  
    val_cal              |  90.13     92.63   |   16.14      9.80   |   95.21     96.36  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 16±0 bits
    GA Neurons  : 5±0 neurons | 16±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  84.44     79.80   |   12.65      6.38   |   91.42     87.35  
    fixed_05             |  75.61     76.09   |    0.79      0.08   |   83.08     83.45  
    platt                |  84.43     79.79   |   12.71      6.40   |   91.42     87.35  
    beta                 |  84.44     79.79   |   12.65      6.40   |   91.42     87.35  
    empirical            |  84.43     79.79   |   12.77      6.32   |   91.42     87.34  
    empirical_cumulative |  79.99     79.79   |    2.22      6.32   |   87.14     87.34  
    val_cal              |  84.44     79.80   |   12.65      6.38   |   91.42     87.35  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 32±0 bits
    GA Neurons  : 304±0 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.10     92.53   |   15.63     10.03   |   95.17     96.31  
    fixed_05             |  79.82     80.00   |    0.92      0.81   |   86.88     87.03  
    platt                |  89.53     91.86   |   11.06      9.29   |   94.62     95.91  
    beta                 |  90.08     92.22   |   17.02     13.19   |   95.22     96.25  
    empirical            |  89.56     92.36   |   24.12     12.30   |   95.29     96.30  
    empirical_cumulative |  88.67     92.44   |    9.28      9.43   |   94.01     96.24  
    val_cal              |  90.13     92.53   |   16.14     10.28   |   95.21     96.32  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 486±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.12     92.09   |   17.32     13.12   |   95.26     96.18  
    fixed_05             |  80.34     80.72   |    0.98      0.92   |   87.33     87.64  
    platt                |  89.54     90.76   |   11.17     10.12   |   94.63     95.31  
    beta                 |  90.09     91.95   |   16.58     14.38   |   95.21     96.15  
    empirical            |  89.95     90.89   |   22.81     20.83   |   95.43     95.84  
    empirical_cumulative |  87.77     89.20   |    7.54      7.29   |   93.33     94.24  
    val_cal              |  90.13     92.09   |   17.19     13.04   |   95.26     96.17  


## XDS-ciciot-96b-Wc-250n100b  (30 flows × 2 phases, seeds: [4239, 4484, 8198, 9697, 17350, 18871, 22408, 29576, 30294, 35241, 35476, 39029, 42887, 45198, 45211, 51567, 51785, 57218, 58977, 60723, 64557, 69769, 70873, 75452, 79203, 80563, 81710, 88923, 93175, 93825])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 250n100b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.17% |   8.07% |  96.60% | r18871 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  93.17% |   8.07% |  96.60% | r18871 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  93.17% |   8.07% |  96.60% | r18871 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  93.01% |   5.89% |  96.45% | r45211 GA best_acc       empirical_cumulative
    Best F1 (FPR<5%)         |  92.73% |   4.86% |  96.26% | r93825 GA best_fpr       empirical_cumulative
    Best F1 (FPR<4%)         |  88.48% |   1.06% |  93.46% | r35476 GA best_fpr       fixed_05
    Best FPR (any F1)        |  88.27% |   0.72% |  93.31% | r39029 GA best_fpr       fixed_05
    Best FPR (F1>80%)        |  88.27% |   0.72% |  93.31% | r39029 GA best_fpr       fixed_05
    Best Acc (any FPR)       |  93.17% |   8.37% |  96.61% | r18871 GA best_acc       train_cal

### best_fitness  (GS: 30 runs | GA: 30 runs)
    Grid Search : 233±24 neurons | 64±0 bits
    GA Neurons  : 228±18 neurons | 64±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.16±0.16 92.81±0.17 |15.29±0.67  8.57±0.67 |95.19±0.11 96.42±0.08
    fixed_05             |81.96±0.10 86.43±0.88 | 1.12±0.04  0.96±0.11 |88.67±0.08 92.04±0.62
    platt                |90.11±0.13 92.80±0.17 |12.71±0.20  8.76±0.46 |95.04±0.07 96.42±0.09
    beta                 |90.14±0.17 92.67±0.16 |16.01±0.23 10.68±0.56 |95.21±0.09 96.41±0.08
    empirical            |89.84±0.24 92.63±0.19 |20.17±0.67 11.04±0.79 |95.24±0.11 96.40±0.08
    empirical_cumulative |89.78±0.15 92.56±0.21 |10.17±0.32  5.57±0.58 |94.73±0.10 96.18±0.12
    val_cal              |90.18±0.16 92.83±0.17 |14.72±0.84  8.27±0.58 |95.18±0.11 96.42±0.09

### best_f1  (GS: 30 runs | GA: 30 runs)
    Grid Search : 98±66 neurons | 42±17 bits
    GA Neurons  : 200±60 neurons | 60±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.94±0.25 92.92±0.15 |15.50±1.36  8.63±0.65 |95.64±0.11 96.48±0.07
    fixed_05             |80.63±0.70 85.01±1.64 | 1.05±0.15  0.87±0.11 |87.57±0.59 90.98±1.22
    platt                |89.88±0.64 92.90±0.15 |11.27±0.40  8.90±0.62 |94.83±0.38 96.48±0.07
    beta                 |90.43±1.02 92.76±0.18 |18.23±4.76 10.84±0.93 |95.49±0.31 96.46±0.08
    empirical            |90.82±0.25 92.79±0.15 |17.21±1.26 10.56±0.77 |95.64±0.11 96.47±0.07
    empirical_cumulative |90.75±0.36 92.74±0.15 |13.68±1.86  6.35±1.18 |95.45±0.22 96.31±0.09
    val_cal              |90.95±0.25 92.93±0.15 |15.33±1.39  8.34±0.84 |95.63±0.11 96.48±0.07

### best_fpr  (GS: 30 runs | GA: 30 runs)
    Grid Search : 124±77 neurons | 67±9 bits
    GA Neurons  : 222±28 neurons | 64±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |89.60±0.97 92.83±0.18 |14.61±0.96  8.51±0.64 |94.83±0.55 96.43±0.09
    fixed_05             |82.00±0.30 86.31±0.94 | 1.25±0.46  0.92±0.10 |88.72±0.25 91.95±0.67
    platt                |89.61±0.97 92.82±0.17 |13.08±1.22  8.68±0.50 |94.76±0.52 96.43±0.09
    beta                 |89.29±1.76 92.69±0.18 |17.22±4.38 10.58±0.61 |94.80±0.75 96.42±0.09
    empirical            |89.27±0.92 92.66±0.19 |19.13±1.31 10.89±0.76 |94.86±0.56 96.41±0.09
    empirical_cumulative |89.16±1.25 92.56±0.20 | 9.34±1.05  5.37±0.54 |94.29±0.87 96.17±0.11
    val_cal              |89.63±0.95 92.84±0.17 |13.63±0.98  8.05±0.81 |94.80±0.55 96.42±0.09

### best_acc  (GS: 30 runs | GA: 30 runs)
    Grid Search : 103±72 neurons | 42±17 bits
    GA Neurons  : 195±63 neurons | 59±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.91±0.28 92.89±0.18 |15.54±1.41  8.89±0.89 |95.62±0.12 96.47±0.08
    fixed_05             |80.64±0.70 84.63±1.80 | 1.06±0.14  0.87±0.09 |87.58±0.58 90.70±1.37
    platt                |89.81±0.67 92.85±0.27 |11.31±0.40  8.97±0.61 |94.80±0.40 96.45±0.14
    beta                 |90.51±0.71 92.73±0.20 |17.89±3.77 11.06±1.05 |95.51±0.22 96.45±0.08
    empirical            |90.81±0.27 92.77±0.17 |17.21±1.26 10.67±0.74 |95.64±0.12 96.47±0.07
    empirical_cumulative |90.75±0.36 92.73±0.16 |14.01±2.00  6.84±1.59 |95.46±0.22 96.32±0.08
    val_cal              |90.92±0.28 92.90±0.18 |15.43±1.45  8.68±1.03 |95.62±0.12 96.47±0.08

### best_ce  (GS: 30 runs | GA: 30 runs)
    Grid Search : 233±24 neurons | 64±0 bits
    GA Neurons  : 228±18 neurons | 64±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.16±0.16 92.81±0.17 |15.29±0.67  8.57±0.67 |95.19±0.11 96.42±0.08
    fixed_05             |81.96±0.10 86.43±0.88 | 1.12±0.04  0.96±0.11 |88.67±0.08 92.04±0.62
    platt                |90.11±0.13 92.80±0.17 |12.71±0.20  8.76±0.46 |95.04±0.07 96.42±0.09
    beta                 |90.14±0.17 92.67±0.16 |16.01±0.23 10.68±0.56 |95.21±0.09 96.41±0.08
    empirical            |89.84±0.24 92.63±0.19 |20.17±0.67 11.04±0.79 |95.24±0.11 96.40±0.08
    empirical_cumulative |89.78±0.15 92.56±0.21 |10.17±0.32  5.57±0.58 |94.73±0.10 96.18±0.12
    val_cal              |90.18±0.16 92.83±0.17 |14.72±0.84  8.27±0.58 |95.18±0.11 96.42±0.09


## XDS-ciciot-96b-Wc-500n34b  (1 flows × 2 phases, seeds: [45211])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)  |  Arch : 500n34b

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  92.62% |   9.25% |  96.33% | r45211 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  92.62% |   9.25% |  96.33% | r45211 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  92.62% |   9.25% |  96.33% | r45211 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  84.74% |   3.13% |  90.96% | r45211 GS best_fpr       beta
    Best F1 (FPR<5%)         |  84.74% |   3.13% |  90.96% | r45211 GS best_fpr       beta
    Best F1 (FPR<4%)         |  84.74% |   3.13% |  90.96% | r45211 GS best_fpr       beta
    Best FPR (any F1)        |  80.67% |   0.87% |  87.60% | r45211 GS best_fpr       fixed_05
    Best FPR (F1>80%)        |  80.67% |   0.87% |  87.60% | r45211 GS best_fpr       fixed_05
    Best Acc (any FPR)       |  92.60% |  10.94% |  96.38% | r45211 GA best_acc       beta

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.85     92.25   |   18.10     11.71   |   95.70     96.21  
    fixed_05             |  80.44     82.05   |    1.00      0.94   |   87.41     88.73  
    platt                |  89.66     92.12   |   11.08      9.25   |   94.70     96.06  
    beta                 |  90.51     92.24   |   16.40     12.61   |   95.44     96.24  
    empirical            |  90.67     92.18   |   19.83     14.32   |   95.68     96.27  
    empirical_cumulative |  89.86     92.08   |   12.09      8.89   |   94.87     96.02  
    val_cal              |  90.85     92.28   |   18.10     10.96   |   95.70     96.20  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 98±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.86     92.61   |   16.66     10.04   |   95.64     96.35  
    fixed_05             |  79.52     82.67   |    0.90      0.88   |   86.63     89.23  
    platt                |  89.48     92.44   |   11.32      8.79   |   94.61     96.22  
    beta                 |  90.69     92.60   |   19.18     10.94   |   95.66     96.38  
    empirical            |  90.75     92.56   |   18.75     11.51   |   95.67     96.38  
    empirical_cumulative |  90.89     92.60   |   14.58      9.16   |   95.57     96.32  
    val_cal              |  90.89     92.62   |   14.58      9.25   |   95.57     96.33  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 96±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  86.89     92.48   |   10.42      9.71   |   92.93     96.27  
    fixed_05             |  80.67     82.98   |    0.87      0.89   |   87.60     89.47  
    platt                |  86.87     92.45   |   10.55      8.28   |   92.92     96.21  
    beta                 |  84.74     92.49   |    3.13     10.51   |   90.96     96.30  
    empirical            |  86.89     92.48   |   10.42     11.20   |   92.93     96.32  
    empirical_cumulative |  86.94     92.29   |    9.55      6.66   |   92.91     96.06  
    val_cal              |  86.94     92.50   |    9.55      9.88   |   92.91     96.29  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 98±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.86     92.61   |   16.66     10.04   |   95.64     96.35  
    fixed_05             |  79.52     82.67   |    0.90      0.88   |   86.63     89.23  
    platt                |  89.48     92.44   |   11.32      8.79   |   94.61     96.22  
    beta                 |  90.69     92.60   |   19.18     10.94   |   95.66     96.38  
    empirical            |  90.75     92.56   |   18.75     11.51   |   95.67     96.38  
    empirical_cumulative |  90.89     92.60   |   14.58      9.16   |   95.57     96.32  
    val_cal              |  90.89     92.62   |   14.58      9.25   |   95.57     96.33  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 500±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  90.85     92.25   |   18.10     11.71   |   95.70     96.21  
    fixed_05             |  80.44     82.05   |    1.00      0.94   |   87.41     88.73  
    platt                |  89.66     92.12   |   11.08      9.25   |   94.70     96.06  
    beta                 |  90.51     92.24   |   16.40     12.61   |   95.44     96.24  
    empirical            |  90.67     92.18   |   19.83     14.32   |   95.68     96.27  
    empirical_cumulative |  89.86     92.08   |   12.09      8.89   |   94.87     96.02  
    val_cal              |  90.85     92.28   |   18.10     10.96   |   95.70     96.20  


---

# =====================================================================
# SECTION 6 — PRESERVED: config-lock analysis written 09/08/2026
# (hand-written; kept verbatim. Its ciciot n=9 cells are now n=10 —
#  the 3 outstanding abl runs completed; Sections 2-4 supersede those cells.)
# =====================================================================

# Config ranking — held-out GA best_f1 val_cal (mean±std over runs)

## unswt — UNSW-NB15 temporal_3way (16b Wb)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    quad     | 10 |  86.54± 1.87 |  13.58± 7.08 |  86.70± 1.73
    ablqsr   | 10 |  83.30± 5.36 |  27.83±18.68 |  84.37± 4.47
    abl3s    | 10 |  79.29± 0.04 |  42.01± 0.02 |  81.02± 0.04
    ablpln   | 10 |  79.24± 0.03 |  42.18± 0.04 |  80.99± 0.03
    abl2s    | 10 |  79.20± 0.10 |  42.01± 0.02 |  80.92± 0.10
    abl2big  | 10 |  79.12± 0.08 |  42.07± 0.10 |  80.85± 0.09

## unswr — UNSW-NB15 random_3way (64b Wb)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    ablqsr   | 10 |  94.33± 0.07 |   0.62± 0.07 |  99.14± 0.02
    quad     | 10 |  93.54± 0.15 |   1.08± 0.12 |  98.93± 0.04
    ablpln   | 10 |  93.52± 0.10 |   1.04± 0.04 |  98.94± 0.02
    abl3s    | 10 |  93.50± 0.03 |   1.12± 0.00 |  98.92± 0.00
    abl2big  | 10 |  93.47± 0.03 |   1.12± 0.00 |  98.92± 0.01
    abl2s    | 10 |  93.47± 0.03 |   1.12± 0.00 |  98.91± 0.00

## cicids — CICIDS2017 random_3way (96b Wa)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    quad     | 10 |  99.59± 0.04 |   0.09± 0.02 |  99.74± 0.02
    ablqsr   | 10 |  99.33± 0.06 |   0.25± 0.05 |  99.58± 0.04
    ablpln   | 10 |  99.27± 0.03 |   0.28± 0.04 |  99.54± 0.02
    abl2big  | 10 |  99.13± 0.01 |   0.60± 0.01 |  99.45± 0.01
    abl3s    | 10 |  99.11± 0.02 |   0.61± 0.02 |  99.43± 0.02
    abl2s    | 10 |  99.08± 0.03 |   0.62± 0.03 |  99.41± 0.02

## ciciot — CIC-IoT-2023 neto_subsample random_3way (96b Wc)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    quad     | 10 |  92.82± 0.11 |   8.66± 1.03 |  96.42± 0.05
    ablqsr   |  9 |  92.07± 0.38 |  11.16± 0.78 |  96.09± 0.21
    ablpln   |  9 |  92.02± 0.50 |  13.42± 1.02 |  96.15± 0.25
    abl3s    |  9 |  81.38± 1.36 |  36.64± 6.31 |  91.54± 0.80
    abl2s    | 10 |  77.70± 1.49 |  44.83± 8.69 |  90.15± 1.74


# Decision — config lock for the 100-run SP cohorts (90 new + existing 10 per dataset)

## QUAD-run identification (veto if wrong)

The "existing 10" per dataset are the completed SP-*-bin-*-n30 flows. They carry NO
memory_mode param, and worker.py resolves `params.get("memory_mode", "QUAD_WEIGHTED")`,
so they ran QUAD_WEIGHTED. The only config difference vs abl2s is memory_mode
(verified key-by-key diff; seeds overlap with the abl groups). Counted flows:

    unswt : 4404 4409 4414 4419 4424 4429 4434 4439 4444 4449
    unswr : 4405 4410 4415 4420 4425 4430 4435 4440 4445 4450
    cicids: 4406 4411 4416 4421 4426 4431 4436 4441 4446 4451
    ciciot: 4407 4412 4417 4422 4427 4432 4437 4442 4447 4452

(4539 SP-unswt-mcsmoke is a multiclass smoke test — excluded. Flows 4454-4538
bin-n30/ciciot46m are paused, not counted.)

## Per-config cost (avg minutes per completed run)

    Config   |  unswt |  unswr | cicids | ciciot
    ---------+--------+--------+--------+--------
    quad     |    3.9 |   22.2 |   71.6 |   66.1
    abl2s    |   17.0 |   24.6 |  177.7 |  166.5
    abl2big  |   13.3 |   19.2 |  119.1 |      —
    abl3s    |   24.0 |  159.8 |  512.4 |  195.5
    ablpln   |   46.5 |  377.4 |  601.8 |  306.1
    ablqsr   |   44.8 |  342.0 |  485.2 |  118.9

## Ranking verdict per dataset (GA best_f1, val_cal held-out, n=10 unless noted)

    unswt  : QUAD wins outright — 86.54±1.87 F1 / 13.58±7.08 FPR. Every non-QUAD mode
             except QSR COLLAPSES to a saturated detector (F1 ~79.2, FPR ~42, std <0.15;
             fixed_05 shows FPR ~100% => memory saturates to "attack" at 16b temporal).
             QSR is bimodal (83.30±5.36, FPR 27.8±18.7) — some seeds escape, some don't.
             Difference QUAD vs best non-QUAD is >> within-config std. DECISIVE.
    unswr  : QSR statistically beats QUAD — 94.33±0.07 vs 93.54±0.15 F1 (+0.79pp),
             FPR 0.62±0.07 vs 1.08±0.12 (-0.46pp). At n=10 the gap is >5x the larger
             within-config std — a real effect. Everything else ties QUAD within ~0.07pp.
    cicids : QUAD wins — 99.59±0.04 vs QSR 99.33±0.06 F1, FPR 0.09±0.02 vs 0.25±0.05.
             Margins >> std. DECISIVE.
    ciciot : QUAD wins — 92.82±0.11 vs QSR 92.07±0.38 (n=9) F1, FPR 8.66±1.03 vs
             11.16±0.78. Margins >> std. DECISIVE. QUAD's val_cal point (92.82/8.66/96.42)
             sits right at the standing WNN-vs-XGBoost reference (93.34/8.37/96.71).

## Recommendation

LOCK QUAD_WEIGHTED for all four dataset tracks.

  - QUAD wins 3/4 tracks decisively and is the ONLY mode that does not collapse on
    unswt; it is also the cheapest mode on every dataset (90 new QUAD runs cost
    ~6h/33h/107h/99h for unswt/unswr/cicids/ciciot vs 8-27x more for PLN/QSR).
  - Only QUAD lets the existing 10 runs count toward the 100 — any other lock means
    100 NEW runs for that track, not 90.
  - The one honest caveat: on unswr, QSR beat QUAD by +0.79pp F1 / -0.46pp FPR
    (significant at n=10). If the paper wants the best unswr headline, a separate
    100-run QSR-unswr cohort (est. ~570h at 342m/run) is the price; as an ablation
    finding ("QSR helps only on the easiest split, at 15x cost") the n=10 result
    already carries the point. Recommend NOT switching the main cohort.

Suggested new-flow naming: SP-{ds}-quad-{width}W{x}-n90-r{seed} (or continue bin-n30
resume-style naming); memory_mode may be set explicitly to "QUAD_WEIGHTED" to make the
cohort self-documenting — it is behaviorally identical to omitting the key.

## Data-quality flags

  1. ciciot has NO abl2big group (never created) — 19 config groups, not 20.
  2. 3 abl runs outstanding, all ciciot: 1 running (abl3s), 2 queued (ablpln, ablqsr)
     — those groups report n=9. Untouched per worker discipline.
  3. unswt non-QUAD collapse (above) is itself a paper-worthy ablation finding:
     graduated QUAD nudging prevents write-saturation where BINARY/TERNARY/PLN commit.
  4. Architecture (neurons/bits) headers are unresolvable for several unswt-abl and
     ciciot GS groups — the winners' genome_hash was never persisted to `genomes`
     (annotated "arch resolvable for X/N" inline). Metrics coverage is 100%.
  5. best_fpr genomes are frequently degenerate (FPR ~0 with F1 ~31-75, or FPR ~100%)
     — known pattern; Pareto mining uses best_ce/best_acc/best_f1 points instead.
  6. All numbers are held-out report-set values (Protocol v2, val-calibrated modes);
     no iterations.best_f1 anywhere in this report.
  7. Base-rate vigilance (ciciot): accuracy runs ~3.5pp above F1 — rank on F1/FPR,
     as done above.



---

# ======================================================================
# MANUAL SECTION — 46M single-flow results (not produced by build_xds_5tables.py;
# a full regen of this file may drop this section — re-append from git if so)
# ======================================================================

## XDS-ciciot-46M-96b-Wc-C35-250n100b-OI-r63432  (flow 4299, completed 09/07/2026 ~15:45 UTC)

Single seed (r63432), 96b thermometer × Wc (ce=0.70) × top20, 46M CIC-IoT-2023
(neto_full, random split, K-fold 5×5, OI/QUAD). Held-out = 20% val (9,337,316 rows,
2.35% benign). Early-stopped Gen 100/250 (plateau best=1.3559 from ~Gen 46).
GA winners ~245-250n × ~60-100b (chunked-wheel run; MARKER_CHUNK 8×32n dispatches).

### Final validation — all genome types × all threshold modes (F1% | FPR% | Acc%)

    Genome        Threshold            |   F1    |  FPR    |  Acc
    --------------------------------------------------------------
    best_f1/acc   train_cal            |  92.28  |   6.16  |  99.22
    best_f1/acc   fixed_05             |  87.28  |   0.05  |  98.46
    best_f1/acc   platt                |  91.78  |   3.49  |  99.14
    best_f1/acc   beta                 |  91.06  |   2.09  |  99.03
    best_f1/acc   empirical            |  92.00  |  11.30  |  99.23
    best_f1/acc   empirical_cumulative |  92.12  |   4.35  |  99.18
    best_f1/acc   val_cal              |  92.28  |   6.16  |  99.22
    best_fpr      train_cal            |  92.20  |   7.64  |  99.22
    best_fpr      fixed_05             |  87.73  |   0.04  |  98.53
    best_fpr      platt                |  91.72  |   3.95  |  99.13
    best_fpr      empirical_cumulative |  91.96  |   4.77  |  99.17
    best_ce/fit   train_cal            |  92.20  |   7.73  |  99.22
    best_ce/fit   fixed_05             |  87.66  |   0.04  |  98.52
    best_ce/fit   platt                |  91.78  |   3.87  |  99.14
    best_ce/fit   empirical_cumulative |  91.91  |   4.30  |  99.16

### vs published paper table (250n × 60-64b cohort bests)

    Paper row       Paper (F1/FPR/Acc)      f4299 point                  f4299 (F1/FPR/Acc)   Delta
    ----------------------------------------------------------------------------------------------------
    Best F1         92.18 /  6.73 / 99.21   best_f1 x train_cal(=val)    92.28 /  6.16 / 99.22   dominates all 3
    Best FPR        88.34 /  0.71 / 98.64   best_fpr x fixed_05          87.73 /  0.04 / 98.53   FPR 18x lower, -0.61pp F1
    Matched FPR(b)  89.58 /  1.65 / 98.83   best_f1 x beta               91.06 /  2.09 / 99.03   +1.48pp F1, +0.44pp FPR
    (bonus)         --                      best_f1 x emp_cumulative     92.12 /  4.35 / 99.18   ~paper-best F1 at -2.4pp FPR

Notes:
- train_cal == val_cal for best_f1 (92.28/6.16/99.22): threshold fitted on train
  transfers exactly to held-out val — DEPLOYABLE, not an oracle artifact.
- The 0.04% FPR point is fixed_05 (raw 0.5 threshold, calibration-free):
  ~88 false alarms / 219,639 benign flows.
- n=1 seed vs paper best-of-cohort: needs sibling 96b-Wc seeds before any paper
  table swap. Data source: validation_summaries flow_id=4299, validation_point='final'
  (latest batch), threshold_metadata JSON.
