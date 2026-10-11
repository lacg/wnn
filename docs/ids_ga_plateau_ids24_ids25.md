# IDS-23 spike: GA plateau — IDS-24 (Grid vs GA, held-out) and IDS-25 (objective mismatch)

Date: 10/10/2026. Read-only on the DB. Scripts: `scripts/ids_plateau/`.
Question (IDS-23): GA Neurons "normalizes" to a plateau and does not keep rare grid genomes.
Is that (1) a dataset+decode CEILING, (2) SEARCH COLLAPSE (IDS-26, separate), or
(3) OBJECTIVE MISMATCH (k-fold search fitness does not rank genomes like held-out)?

## Cohorts and protocol

```
All: QUAD_WEIGHTED, Protocol v2 (_3way 80/10/10; thresholds calibrated on VAL, TEST report-only),
     completed flows, FINAL validation rows, own-flow rows only.
SP100 : SP100-{cicids,ciciot,unswr,unswt}-quad-*      production config, 29 completed per dataset
SWEEP : IDSXD-{ds}-quad-* (B34-CTRL excluded: min_bits=34 pins bits; r20401-04 are pre-fix hybrids)
        IDSAGG-/IDSAGGX-{cicids,ciciot}-quad-*
Excluded: zINVALID*, all QSR, SP-*-bin (same seeds/config as SP100 -> double count).
Own rows: cache_key NOT NULL (re-validated) or not listed as inherited in the 10/10 audit pickle.
  341 inherited, not re-validated rows excluded -> they are almost all GRID-phase rows
  (the grid evaluates the same points in every flow, so the pre-fix cache matched them across flows).
  A run enters View A only if BOTH phases have all 5 genome types x 7 modes own:
    unswr 43 (SP100 19 + SWEEP 24)   unswt 29 (21 + 8)   cicids 32 (19 + 13)   ciciot 41 (19 + 22)
```

Generators: `scripts/ids_plateau/ids24_grid_vs_ga.py` (IDS-24),
`ids25_rescore_grid.py` + `ids25_guarded.sh` + `ids25_select.py` + `ids25_analyze.py` (IDS-25).

## IDS-24 — Grid best vs GA best on held-out TEST, paired per run

FPR floors (stated rule: dataset's median GA best_f1 val_cal F1 minus 0.5 pp, the UNSW-random 93.0 analog):

```
    Dataset | median GA best_f1 val_cal F1 | FPR floor | grid-reachable floor (min(floor, median Grid best-F1 - 0.5))
    --------+------------------------------+-----------+--------------------------------------------------------------
    unswr   |            93.49             |   93.0    |   93.0
    unswt   |            87.59             |   87.1    |   87.1
    cicids  |            99.54             |   99.0    |   98.8
    ciciot  |            92.90             |   92.4    |   90.3   (no Grid cell in 41 runs reaches 92.4)
```

### Headline: F1 max over the 35 cells, delta GA - Grid (pp), by cohort family

```
    Scope  | Dataset | N  | Grid best    | GA best      | delta GA-Grid        | 95% boot CI       | GA>Grid / < / tie | Wilcoxon p
    -------+---------+----+--------------+--------------+----------------------+-------------------+-------------------+-----------
    ALL    | unswr   | 43 | 93.50±0.01   | 93.49±0.02   | -0.007±0.023         | [-0.014,-0.000]   | 15 / 19 / 9       | 0.078
    ALL    | unswt   | 29 | 88.69±0.12   | 89.10±0.29   | +0.406±0.263         | [+0.313,+0.504]   | 28 /  1 / 0       | 7.5e-09
    ALL    | cicids  | 32 | 99.32±0.02   | 99.52±0.08   | +0.202±0.075         | [+0.176,+0.227]   | 32 /  0 / 0       | 4.7e-10
    ALL    | ciciot  | 41 | 90.78±0.38   | 92.93±0.14   | +2.146±0.425         | [+2.016,+2.276]   | 41 /  0 / 0       | 9.1e-13
    SP100  | unswr   | 19 | 93.49±0.01   | 93.50±0.02   | +0.008±0.017         | [+0.003,+0.017]   | 11 /  0 / 8       | 0.003
    SP100  | unswt   | 21 | 88.68±0.12   | 89.03±0.25   | +0.357±0.236         | [+0.263,+0.458]   | 20 /  1 / 0       | 1.9e-06
    SP100  | cicids  | 19 | 99.31±0.02   | 99.55±0.06   | +0.232±0.056         | [+0.207,+0.255]   | 19 /  0 / 0       | 3.8e-06
    SP100  | ciciot  | 19 | 90.98±0.28   | 92.96±0.13   | +1.981±0.330         | [+1.837,+2.128]   | 19 /  0 / 0       | 3.8e-06
    SWEEP  | unswr   | 24 | 93.50±0.01   | 93.48±0.02   | -0.019±0.021         | [-0.027,-0.010]   |  4 / 19 / 1       | 0.00053
    SWEEP  | unswt   |  8 | 88.73±0.10   | 89.27±0.35   | +0.534±0.302         | [+0.346,+0.735]   |  8 /  0 / 0       | 0.0078
    SWEEP  | cicids  | 13 | 99.32±0.03   | 99.47±0.08   | +0.158±0.079         | [+0.116,+0.198]   | 13 /  0 / 0       | 0.00024
    SWEEP  | ciciot  | 22 | 90.61±0.37   | 92.90±0.14   | +2.288±0.453         | [+2.102,+2.465]   | 22 /  0 / 0       | 4.8e-07
```

Luiz's early look ("best F1 cell from Grid in 50/57") reproduces ONLY on UNSW-random SWEEP
(Grid holds the run's best F1 cell in 19/24) — and there the whole gap is -0.019 pp, inside a
93.47-93.51 band where the Grid best has SD 0.01 pp across 24 runs. On every other dataset the GA
holds the best F1 cell in essentially every run (unswt 28/29, cicids 32/32, ciciot 41/41).

### View A (scope ALL) — best-everywhere cell per phase, all three metrics

```
    ================ VIEW A — paired per run, scope=ALL ================
    ### unswr  (N=43 runs; FPR floor F1>=93.0, N paired=43)
    Metric            | Grid best      | GA best        | delta GA-Grid (pp), sign, test
    F1 max            |  93.50±0.01   |  93.49±0.02   | -0.007±0.023  CI[-0.014,-0.000]  GA-better 15/43 worse 19 tie 9  Wilcoxon p=0.078
    FPR min @F1>=93.0  |   1.11±0.01   |   1.03±0.15   | -0.077±0.149  CI[-0.125,-0.035]  GA-better 31/43 worse 9 tie 3  Wilcoxon p=0.00024  (N paired=43)
    Acc max           |  98.92±0.00   |  98.93±0.03   | +0.012±0.025  CI[+0.005,+0.020]  GA-better 25/43 worse 11 tie 7  Wilcoxon p=0.027
    holder of run's best F1 cell: {'GS': 19, 'GA': 15, 'tie': 9}  | best FPR@floor cell: {'GS': 9, 'GA': 31, 'tie': 3} | @grid-reachable floor: {'GS': 9, 'GA': 31, 'tie': 3}
    runs where GA reaches F1>=93.0 but Grid never does (any of 35 cells): 0/43
    ### unswt  (N=29 runs; FPR floor F1>=87.1, N paired=29)
    Metric            | Grid best      | GA best        | delta GA-Grid (pp), sign, test
    F1 max            |  88.69±0.12   |  89.10±0.29   | +0.406±0.263  CI[+0.313,+0.504]  GA-better 28/29 worse 1 tie 0  Wilcoxon p=7.5e-09
    FPR min @F1>=87.1  |   5.88±1.23   |   5.35±2.21   | -0.530±2.344  CI[-1.356,+0.320]  GA-better 21/29 worse 8 tie 0  Wilcoxon p=0.11  (N paired=29)
    Acc max           |  88.74±0.11   |  89.16±0.30   | +0.417±0.267  CI[+0.324,+0.515]  GA-better 28/29 worse 1 tie 0  Wilcoxon p=2.8e-06
    holder of run's best F1 cell: {'GS': 1, 'GA': 28, 'tie': 0}  | best FPR@floor cell: {'GS': 8, 'GA': 21, 'tie': 0} | @grid-reachable floor: {'GS': 8, 'GA': 21, 'tie': 0}
    runs where GA reaches F1>=87.1 but Grid never does (any of 35 cells): 0/29
    ### cicids  (N=32 runs; FPR floor F1>=99.0, N paired=32)
    Metric            | Grid best      | GA best        | delta GA-Grid (pp), sign, test
    F1 max            |  99.32±0.02   |  99.52±0.08   | +0.202±0.075  CI[+0.176,+0.227]  GA-better 32/32 worse 0 tie 0  Wilcoxon p=4.7e-10
    FPR min @F1>=99.0  |   0.19±0.04   |   0.10±0.08   | -0.092±0.086  CI[-0.120,-0.062]  GA-better 26/32 worse 5 tie 1  Wilcoxon p=2.5e-05  (N paired=32)
    FPR min @F1>=98.8  |   0.16±0.05   |   0.09±0.06   | -0.067±0.072  CI[-0.091,-0.042]  GA-better 26/32 worse 5 tie 1  Wilcoxon p=9.6e-05  (N paired=32)
    Acc max           |  99.57±0.01   |  99.69±0.05   | +0.128±0.048  CI[+0.111,+0.144]  GA-better 32/32 worse 0 tie 0  Wilcoxon p=7.9e-07
    holder of run's best F1 cell: {'GS': 0, 'GA': 32, 'tie': 0}  | best FPR@floor cell: {'GS': 5, 'GA': 26, 'tie': 1} | @grid-reachable floor: {'GS': 5, 'GA': 26, 'tie': 1}
    runs where GA reaches F1>=99.0 but Grid never does (any of 35 cells): 0/32
    ### ciciot  (N=41 runs; FPR floor F1>=92.4, N paired=0)
    Metric            | Grid best      | GA best        | delta GA-Grid (pp), sign, test
    F1 max            |  90.78±0.38   |  92.93±0.14   | +2.146±0.425  CI[+2.016,+2.276]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=9.1e-13
    FPR min @F1>=92.4  | n/a   | n/a   | n/a  (N paired=0)
    FPR min @F1>=90.3  |  13.18±1.86   |   5.48±1.09   | -7.707±2.173  CI[-8.426,-7.005]  GA-better 35/35 worse 0 tie 0  Wilcoxon p=5.8e-11  (N paired=35)
    Acc max           |  95.57±0.21   |  96.48±0.07   | +0.909±0.215  CI[+0.845,+0.975]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=2.4e-08
    holder of run's best F1 cell: {'GS': 0, 'GA': 41, 'tie': 0}  | best FPR@floor cell: {'GS': 0, 'GA': 0, 'tie': 0} | @grid-reachable floor: {'GS': 0, 'GA': 35, 'tie': 0}
    runs where GA reaches F1>=92.4 but Grid never does (any of 35 cells): 41/41
    
    ---- Conventional cell (val_cal), paired, scope=ALL ----
      unswr  best_f1      N= 43 | F1 GS  93.49±0.01 GA  93.48±0.03 d -0.019±0.035  CI[-0.030,-0.010]  GA-better 7/43 worse 22 tie 14  Wilcoxon p=0.00026
                             | FPR GS   1.12±0.00 GA   1.09±0.09 d -0.025±0.094  CI[-0.056,-0.000]  GA-better 16/43 worse 3 tie 24  Wilcoxon p=0.0039
                             | Acc GS  98.92±0.00 GA  98.92±0.02 d +0.002±0.017  CI[-0.002,+0.008]  GA-better 9/43 worse 17 tie 17  Wilcoxon p=0.084
      unswr  best_fitness N= 43 | F1 GS  93.48±0.03 GA  92.80±1.93 d -0.672±1.937  CI[-1.290,-0.166]  GA-better 12/43 worse 19 tie 12  Wilcoxon p=0.15
                             | FPR GS   1.12±0.00 GA   1.16±0.25 d +0.045±0.254  CI[-0.024,+0.125]  GA-better 11/43 worse 16 tie 16  Wilcoxon p=0.26
                             | Acc GS  98.92±0.00 GA  98.81±0.32 d -0.109±0.324  CI[-0.213,-0.024]  GA-better 12/43 worse 17 tie 14  Wilcoxon p=0.4
      unswt  best_f1      N= 29 | F1 GS  87.31±1.82 GA  86.87±2.31 d -0.433±2.101  CI[-1.206,+0.295]  GA-better 17/29 worse 12 tie 0  Wilcoxon p=0.88
                             | FPR GS  12.37±6.84 GA  14.18±7.26 d +1.809±8.059  CI[-0.989,+4.751]  GA-better 11/29 worse 18 tie 0  Wilcoxon p=0.25
                             | Acc GS  87.45±1.62 GA  87.06±2.16 d -0.396±1.908  CI[-1.097,+0.266]  GA-better 18/29 worse 11 tie 0  Wilcoxon p=1
      unswt  best_fitness N= 29 | F1 GS  87.33±1.83 GA  87.13±2.30 d -0.194±1.970  CI[-0.937,+0.455]  GA-better 19/29 worse 10 tie 0  Wilcoxon p=0.45
                             | FPR GS  12.23±6.91 GA  12.93±6.99 d +0.698±9.762  CI[-2.966,+4.156]  GA-better 13/29 worse 16 tie 0  Wilcoxon p=0.42
                             | Acc GS  87.47±1.63 GA  87.29±2.17 d -0.181±1.780  CI[-0.860,+0.408]  GA-better 20/29 worse 9 tie 0  Wilcoxon p=0.36
      cicids best_f1      N= 32 | F1 GS  99.31±0.02 GA  99.52±0.08 d +0.202±0.076  CI[+0.176,+0.227]  GA-better 32/32 worse 0 tie 0  Wilcoxon p=4.7e-10
                             | FPR GS   0.29±0.03 GA   0.16±0.07 d -0.139±0.072  CI[-0.163,-0.114]  GA-better 31/32 worse 1 tie 0  Wilcoxon p=1.4e-09
                             | Acc GS  99.57±0.01 GA  99.69±0.05 d +0.128±0.048  CI[+0.112,+0.144]  GA-better 32/32 worse 0 tie 0  Wilcoxon p=4.7e-10
      cicids best_fitness N= 32 | F1 GS  99.30±0.03 GA  99.51±0.08 d +0.209±0.071  CI[+0.185,+0.233]  GA-better 32/32 worse 0 tie 0  Wilcoxon p=4.7e-10
                             | FPR GS   0.30±0.03 GA   0.16±0.08 d -0.140±0.072  CI[-0.165,-0.115]  GA-better 32/32 worse 0 tie 0  Wilcoxon p=4.7e-10
                             | Acc GS  99.56±0.02 GA  99.69±0.05 d +0.133±0.045  CI[+0.117,+0.149]  GA-better 32/32 worse 0 tie 0  Wilcoxon p=4.7e-10
      ciciot best_f1      N= 41 | F1 GS  90.75±0.40 GA  92.90±0.13 d +2.152±0.447  CI[+2.020,+2.286]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=9.1e-13
                             | FPR GS  15.74±1.34 GA   8.14±0.95 d -7.602±1.681  CI[-8.108,-7.092]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=9.1e-13
                             | Acc GS  95.54±0.23 GA  96.46±0.07 d +0.916±0.230  CI[+0.848,+0.984]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=9.1e-13
      ciciot best_fitness N= 41 | F1 GS  90.29±0.33 GA  92.81±0.17 d +2.525±0.407  CI[+2.395,+2.640]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=9.1e-13
                             | FPR GS  14.78±1.37 GA   8.11±1.04 d -6.671±1.453  CI[-7.115,-6.234]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=9.1e-13
                             | Acc GS  95.24±0.22 GA  96.40±0.08 d +1.168±0.235  CI[+1.095,+1.235]  GA-better 41/41 worse 0 tie 0  Wilcoxon p=9.1e-13
    
```

FPR on UNSW-random is the one place the GA does move something: at F1 >= 93.0 it finds a lower-FPR
cell than the grid in 31/43 runs (-0.077±0.149 pp, p=0.0002) — consistent with "best FPR cell from GA
in 41/57". It is a Pareto move inside the F1 ceiling, not an F1 gain.

### Conventional cell (val_cal), Grid vs GA paired, scope ALL

```
    Dataset | genome      | N  | F1 Grid -> GA             d (GA>Grid/</tie)       | FPR Grid -> GA            d (GA better/worse/tie)
    --------+-------------+----+---------------------------------------------------+------------------------------------------------
    unswr   | best_f1     | 43 | 93.49±0.01 -> 93.48±0.03  -0.019±0.035 (7/22/14)  |  1.12±0.00 ->  1.09±0.09  -0.025±0.094 (16/3/24)
    unswr   | best_fitness| 43 | 93.48±0.03 -> 92.80±1.93  -0.672±1.937 (12/19/12) |  1.12±0.00 ->  1.16±0.25  +0.045±0.254 (11/16/16)
    unswt   | best_f1     | 29 | 87.31±1.82 -> 86.87±2.31  -0.433±2.101 (17/12/0)  | 12.37±6.84 -> 14.18±7.26  +1.809±8.059 (11/18/0)
    unswt   | best_fitness| 29 | 87.33±1.83 -> 87.13±2.30  -0.194±1.970 (19/10/0)  | 12.23±6.91 -> 12.93±6.99  +0.698±9.762 (13/16/0)
    cicids  | best_f1     | 32 | 99.31±0.02 -> 99.52±0.08  +0.202±0.076 (32/0/0)   |  0.29±0.03 ->  0.16±0.07  -0.139±0.072 (31/1/0)
    cicids  | best_fitness| 32 | 99.30±0.03 -> 99.51±0.08  +0.209±0.071 (32/0/0)   |  0.30±0.03 ->  0.16±0.08  -0.140±0.072 (32/0/0)
    ciciot  | best_f1     | 41 | 90.75±0.40 -> 92.90±0.13  +2.152±0.447 (41/0/0)   | 15.74±1.34 ->  8.14±0.95  -7.602±1.681 (41/0/0)
    ciciot  | best_fitness| 41 | 90.29±0.33 -> 92.81±0.17  +2.525±0.407 (41/0/0)   | 14.78±1.37 ->  8.11±1.04  -6.671±1.453 (41/0/0)
```

Side finding (not the plateau): UNSW-random SP100 GA **best_fitness** loses -1.53±2.72 pp F1
(6/19 worse, 12 ties; e.g. r85002 477n x 4b -> 85.02 F1, r49337 11n -> 88.18). SP100 uses the
harmonic-rank calculator; on a population tied at the ceiling, ranks are noise and WHM crowns
tiny specialists. The SWEEP arms (desirability) do not show it (+0.004±0.041 pp, 11 better / 13 worse of 24). best_f1 is
unaffected. UNSW-temporal val_cal is noisy (SD 2 pp F1, 7 pp FPR) and shows no Grid/GA difference.

### View B — best-of-best single cell per dataset per phase (n=1, selected on TEST — descriptive only)

```
    ================ VIEW B — best-of-best single cell per dataset per phase (n=1, ALL scope) ================
    ### unswr  (pool: 43 runs x 35 cells per phase; FPR floor F1>=93.0)
    Phase | Metric              |    F1   |   FPR   |   Acc   | Source (run / genome_type / mode / size)
    GS    | Best F1 (any FPR)   |  93.510 |   1.119 |  98.921 | IDSXD-unswr-quad-64b-B05-AC-r20405 / best_f1 / train_cal / 200n x 16.0b
    GS    | Best FPR (F1>=93.0) |  93.045 |   1.068 |  98.863 | SP100-unswr-quad-64bWb-r18612 / best_fpr / platt / 5n x 34.0b
    GS    | Best Acc            |  93.510 |   1.119 |  98.921 | IDSXD-unswr-quad-64b-B05-AC-r20405 / best_f1 / train_cal / 200n x 16.0b
    GA    | Best F1 (any FPR)   |  93.566 |   0.659 |  99.022 | SP100-unswr-quad-64bWb-r85002 / best_acc / train_cal / 304n x 13.8b
    GA    | Best FPR (F1>=93.0) |  93.204 |   0.614 |  98.983 | SP100-unswr-quad-64bWb-r56815 / best_acc / platt / 194n x 24.2b
    GA    | Best Acc            |  93.566 |   0.659 |  99.022 | SP100-unswr-quad-64bWb-r85002 / best_acc / train_cal / 304n x 13.8b
    ### unswt  (pool: 29 runs x 35 cells per phase; FPR floor F1>=87.1)
    Phase | Metric              |    F1   |   FPR   |   Acc   | Source (run / genome_type / mode / size)
    GS    | Best F1 (any FPR)   |  89.034 |   6.438 |  89.061 | SP100-unswt-quad-16bWb-r55758 / best_fpr / empirical_cumulative / 5n x 32.0b
    GS    | Best FPR (F1>=87.1) |  87.169 |   4.476 |  87.171 | SP100-unswt-quad-16bWb-r66833 / best_ce / empirical_cumulative / 300n x 32.0b
    GS    | Best Acc            |  89.034 |   6.438 |  89.061 | SP100-unswt-quad-16bWb-r55758 / best_fpr / empirical_cumulative / 5n x 32.0b
    GA    | Best F1 (any FPR)   |  89.725 |   9.400 |  89.797 | IDSXD-unswt-quad-16b-B10-CE-r20403 / best_fpr / val_cal / 411n x 34.0b
    GA    | Best FPR (F1>=87.1) |  87.125 |   2.676 |  87.125 | SP100-unswt-quad-16bWb-r47498 / best_ce / empirical_cumulative / 382n x 34.0b
    GA    | Best Acc            |  89.725 |   9.400 |  89.797 | IDSXD-unswt-quad-16b-B10-CE-r20403 / best_fpr / val_cal / 411n x 34.0b
    ### cicids  (pool: 32 runs x 35 cells per phase; FPR floor F1>=99.0)
    Phase | Metric              |    F1   |   FPR   |   Acc   | Source (run / genome_type / mode / size)
    GS    | Best F1 (any FPR)   |  99.356 |   0.249 |  99.593 | SP100-cicids-quad-96bWa-r45134 / best_f1 / val_cal / 100n x 32.0b
    GS    | Best FPR (F1>=99.0) |  99.141 |   0.084 |  99.461 | SP100-cicids-quad-96bWa-r66758 / best_f1 / empirical / 100n x 32.0b
    GS    | Best FPR (F1>=98.8) |  98.905 |   0.074 |  99.315 | SP100-cicids-quad-96bWa-r44834 / best_f1 / empirical / 300n x 34.0b
    GS    | Best Acc            |  99.356 |   0.249 |  99.593 | SP100-cicids-quad-96bWa-r45134 / best_f1 / val_cal / 100n x 32.0b
    GA    | Best F1 (any FPR)   |  99.630 |   0.075 |  99.767 | SP100-cicids-quad-96bWa-r66758 / best_f1 / empirical_cumulative / 107n x 32.5b
    GA    | Best FPR (F1>=99.0) |  99.127 |   0.048 |  99.453 | SP100-cicids-quad-96bWa-r25437 / best_fpr / empirical / 214n x 34.0b
    GA    | Best FPR (F1>=98.8) |  98.927 |   0.046 |  99.329 | SP100-cicids-quad-96bWa-r45134 / best_f1 / empirical / 105n x 32.0b
    GA    | Best Acc            |  99.630 |   0.075 |  99.767 | SP100-cicids-quad-96bWa-r66758 / best_f1 / empirical_cumulative / 107n x 32.5b
    ### ciciot  (pool: 41 runs x 35 cells per phase; FPR floor F1>=92.4)
    Phase | Metric              |    F1   |   FPR   |   Acc   | Source (run / genome_type / mode / size)
    GS    | Best F1 (any FPR)   |  91.461 |  14.205 |  95.871 | SP100-ciciot-quad-96bWc-r49735 / best_f1 / train_cal / 200n x 48.0b
    GS    | Best FPR (F1>=92.4) |   —     |   —     |   —     | no cell reaches the floor
    GS    | Best FPR (F1>=90.3) |  90.356 |  10.330 |  95.081 | SP100-ciciot-quad-96bWc-r73307 / best_acc / platt / 50n x 48.0b
    GS    | Best Acc            |  91.461 |  14.205 |  95.871 | SP100-ciciot-quad-96bWc-r49735 / best_acc / train_cal / 200n x 48.0b
    GA    | Best F1 (any FPR)   |  93.177 |   6.910 |  96.571 | IDSXD-ciciot-quad-96b-B05-AC-r20405 / best_f1 / train_cal / 174n x 64.0b
    GA    | Best FPR (F1>=92.4) |  92.577 |   4.235 |  96.148 | IDSXD-ciciot-quad-96b-B10-CE-r20405 / best_fitness / empirical_cumulative / 194n x 80.0b
    GA    | Best FPR (F1>=90.3) |  91.192 |   2.725 |  95.272 | IDSXD-ciciot-quad-96b-CE20-r20403 / best_f1 / empirical_cumulative / 145n x 80.0b
    GA    | Best Acc            |  93.155 |   8.340 |  96.602 | SP100-ciciot-quad-96bWc-r54841 / best_f1 / platt / 141n x 57.2b
    
    ================ Phase best_f1 genome size (mean±SD over runs) ================
      unswr  GS: 288.37±117.94 neurons |  14.42±1.98 bits  (N=43)
      unswr  GA: 348.14±108.68 neurons |  13.53±1.90 bits  (N=43)
      unswt  GS: 130.54±154.34 neurons |  30.50±5.01 bits  (N=28)
      unswt  GA: 162.71±180.12 neurons |  29.34±5.46 bits  (N=28)
      cicids GS: 178.12±112.84 neurons |  33.62±0.79 bits  (N=32)
      cicids GA: 185.47±114.70 neurons |  33.75±0.55 bits  (N=32)
      ciciot GS: 147.56±69.78 neurons |  53.85±16.69 bits  (N=41)
      ciciot GA: 182.37±52.18 neurons |  64.66±11.14 bits  (N=41)
```

Genome sizes are from the genomes table by hash (cross-checked against the GA checkpoint for
r20403 Wb-CTRL: 373n x 12.1b both ways). They do not support "~497n x 34b in every arm" for the
best_f1 genome on UNSW-random (GA 348±109n x 13.5±1.9 mean bits); that figure was not reconciled.

### IDS-24 verdict

**Supported (ceiling, not GA) — on the held-out TEST partition, N = 43/29/32/41 paired runs.**
The GA beats the grid on every dataset that has headroom: CIC-IoT +2.15 pp F1 (41/41), CICIDS
+0.20 (32/32), UNSW-temporal +0.41 (28/29), all with CIs excluding 0 and the same sign in both
cohort families. On UNSW-random it does not, because there is nothing to beat: grid and GA both sit
at 93.48-93.51 F1 / ~1.1 FPR with between-run SD 0.01-0.02 pp; the -0.007 pp delta is below any
meaningful resolution (indistinguishable at N=43). Rule from the brief: GA beats grid on CICIDS and
CIC-IoT but not UNSW-random -> ceiling.

## IDS-25 — does k-fold search fitness rank genomes like held-out?

Where the numbers live:
- Per-genome k-fold metrics: `genome_evaluations` (ce/accuracy/f1_macro/fpr/fitness_score per
  iteration; the grid writes one iteration per grid point — 94 on UNSW — plus one population
  iteration of 50 with fitness_score) and the grid checkpoint
  `checkpoints/<flow>/exp_00/grid_search_(neurons_x_bits).json.gz`
  (`phase_result.final_population` = top-50 grid points by fitness WITH connections,
  `population_metrics` = their k-fold ce/acc/f1/fpr). The `genomes` table has no connections for
  grid genomes, so only the checkpoint's 50 can be re-scored.
- Held-out: re-scored through the worker's own path (`FlowWorker._build_flow` ->
  `Experiment.revalidate_genome` -> `_validate_one_genome`, single training pass, 7 Protocol-v2
  modes). **No source edits** (experiment.py untouched — the worker spawns a fresh flow_runner per
  flow, so an edit would go live): threshold_metadata is captured by a stand-in dashboard client;
  the worker's client is wrapped read-only (get_*/list_* only), no SqliteTracker; nothing written.
- Parity: re-scored min-k-fold-CE genome == stored own DB best_ce row to 4 decimals on all 7 modes
  checked (flows 4786, 4787, 5965).

Subset (full set ~10 h > 2 h cap): stratified, `ids25_select.py` — per dataset first K SP100 runs by
flow id + K SWEEP runs one per arm: cicids 5+5, ciciot 5+5, unswr 8+8, unswt 8+8 = 52 flows x 50
genomes = 2600 held-out re-scores. Compute: 1.50 h re-score wall (cicids ~4.5 min/flow, ciciot ~3,
unswr ~1, unswt ~0.3), RAYON_NUM_THREADS=4, memory-guarded (>=16 GiB to start, kill <12 GiB; never
triggered).

Metrics: Spearman per run over the 50 genomes, oriented so rho>0 = search agrees with held-out.
fit_* = search fitness_score (lower=better) vs held-out TEST val_cal F1 / FPR and CE;
k*_* = same-metric k-fold vs held-out. Regret = best held-out val_cal F1 in the population minus
held-out F1 of the fitness-best genome (what the GA's choice costs). Pooled = mean of per-run rho,
bootstrap 95% CI over runs.

```
    Dataset | runs | fit vs F1             | fit vs FPR            | fit vs CE             | kF1 vs F1             | kFPR vs FPR           | kCE vs CE             | regret F1 pp | fit-best ho rank (median of 50)
    --------+------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+--------------+--------------------------------
    unswr   |  16  | +0.018 [-0.042,+0.085]| +0.029 [-0.013,+0.070]| +0.544 [+0.324,+0.767]| +0.884 [+0.852,+0.914]| +0.440 [+0.362,+0.521]| +0.992 [+0.990,+0.994]| 0.011±0.009  | 12
    unswt   |  16  | +0.378 [+0.279,+0.473]| +0.038 [-0.018,+0.093]| +0.335 [+0.220,+0.447]| +0.186 [+0.078,+0.295]| +0.121 [+0.019,+0.226]| +0.704 [+0.637,+0.771]| 1.173±1.790  | 25
    cicids  |  10  | +0.829 [+0.802,+0.860]| +0.013 [-0.086,+0.126]| +0.867 [+0.792,+0.929]| +0.931 [+0.906,+0.952]| +0.486 [+0.386,+0.588]| +0.994 [+0.992,+0.996]| 0.018±0.023  | 2
    ciciot  |  10  | +0.417 [+0.315,+0.542]| -0.312 [-0.398,-0.230]| +0.467 [+0.240,+0.696]| +0.668 [+0.547,+0.767]| +0.303 [+0.013,+0.583]| +0.738 [+0.629,+0.842]| 0.298±0.229  | 4
```

Per-family split worth knowing (from the per-run table below): on UNSW-random the SWEEP
(desirability) fitness tracks held-out CE at rho 0.985-0.994 while SP100 (harmonic rank) tracks it
at -0.05..+0.18 — the aggregation, not the k-fold estimate, is what decorrelates. Neither tracks
held-out F1 (|rho| <= 0.34), yet the regret is 0.011 pp because the 50 genomes span ~0.35 pp of
held-out F1 (median) — a flat ceiling where ordering cannot cost anything.

Per-run detail and pooled output (`python scripts/ids_plateau/ids25_analyze.py`):

```
    Per-run Spearman (rho>0 = search agrees with held-out); held-out = TEST val_cal F1/FPR, CE
    run                                        |   N |   fit_F1 |  fit_FPR |   fit_CE |   kF1_F1 | kFPR_FPR |   kCE_CE | kF1 spread | hoF1 spread | regret F1 pp | ho rank | wall s
    IDSAGG-cicids-quad-96b-CE20-desir-r20414   |  50 |   +0.813 |   -0.117 |   +0.987 |   +0.870 |   +0.302 |   +0.997 |      0.279 |       0.251 |        0.012 |       2 |    198
    IDSAGG-cicids-quad-96b-CE20-zs-r20414      |  50 |   +0.833 |   +0.363 |   +0.607 |   +0.857 |   +0.222 |   +0.995 |      0.355 |       0.333 |        0.071 |      21 |    294
    IDSAGG-cicids-quad-96b-Wa-CTRL-desir-r20414 |  50 |   +0.788 |   -0.077 |   +0.993 |   +0.963 |   +0.753 |   +0.994 |      0.604 |       0.628 |        0.012 |       2 |    240
    IDSAGG-cicids-quad-96b-Wa-CTRL-zs-r20414   |  50 |   +0.938 |   -0.172 |   +0.905 |   +0.967 |   +0.688 |   +0.996 |      0.168 |       0.181 |        0.018 |       3 |    291
    IDSXD-cicids-quad-96b-B05-AC-r20403        |  50 |   +0.766 |   -0.084 |   +0.990 |   +0.962 |   +0.406 |   +0.991 |      0.260 |       0.251 |        0.000 |       1 |    269
    SP100-cicids-quad-96bWa-r26177             |  50 |   +0.777 |   +0.258 |   +0.860 |   +0.935 |   +0.570 |   +0.993 |      4.306 |       4.413 |        0.000 |       1 |    204
    SP100-cicids-quad-96bWa-r37040             |  50 |   +0.837 |   +0.063 |   +0.837 |   +0.904 |   +0.632 |   +0.996 |      0.190 |       0.203 |        0.046 |      14 |    250
    SP100-cicids-quad-96bWa-r38183             |  50 |   +0.873 |   -0.102 |   +0.832 |   +0.953 |   +0.483 |   +0.993 |      0.242 |       0.234 |        0.017 |       2 |    265
    SP100-cicids-quad-96bWa-r41738             |  50 |   +0.838 |   -0.112 |   +0.837 |   +0.957 |   +0.335 |   +0.987 |     27.521 |      26.667 |        0.000 |       1 |    277
    SP100-cicids-quad-96bWa-r43932             |  50 |   +0.827 |   +0.117 |   +0.820 |   +0.940 |   +0.472 |   +0.997 |      0.198 |       0.186 |        0.000 |       1 |    291
    IDSAGG-ciciot-quad-96b-B15-AC-desir-r20414 |  50 |   +0.236 |   -0.317 |   +0.965 |   +0.222 |   +0.738 |   +0.901 |      1.203 |       0.921 |        0.259 |       9 |    209
    IDSAGG-ciciot-quad-96b-B15-AC-zs-r20414    |  50 |   +0.849 |   -0.198 |   +0.050 |   +0.858 |   +0.680 |   +0.989 |      1.992 |       1.694 |        0.000 |       1 |    124
    IDSAGG-ciciot-quad-96b-Wc-CTRL-desir-r20414 |  50 |   +0.363 |   -0.314 |   +0.872 |   +0.541 |   +0.671 |   +0.846 |      1.385 |       1.342 |        0.278 |       3 |    172
    IDSAGG-ciciot-quad-96b-Wc-CTRL-zs-r20414   |  50 |   +0.277 |   -0.115 |   +0.823 |   +0.697 |   +0.812 |   +0.896 |      1.487 |       1.336 |        0.260 |       3 |    168
    IDSAGGX-ciciot-quad-96b-B05-AC-desir-r20414 |  50 |   +0.248 |   -0.290 |   +0.893 |   +0.565 |   +0.806 |   +0.844 |      0.713 |       0.848 |        0.101 |       3 |    184
    SP100-ciciot-quad-96bWc-r24530             |  50 |   +0.467 |   -0.367 |   +0.239 |   +0.775 |   +0.062 |   +0.606 |      1.813 |       1.580 |        0.470 |       4 |    136
    SP100-ciciot-quad-96bWc-r29083             |  50 |   +0.628 |   -0.517 |   +0.336 |   +0.650 |   -0.269 |   +0.579 |      1.680 |       3.476 |        0.000 |       1 |    149
    SP100-ciciot-quad-96bWc-r42704             |  50 |   +0.387 |   -0.133 |   +0.266 |   +0.748 |   +0.051 |   +0.634 |      0.868 |       1.176 |        0.516 |       6 |    159
    SP100-ciciot-quad-96bWc-r47707             |  50 |   +0.292 |   -0.524 |   -0.106 |   +0.813 |   -0.217 |   +0.422 |      1.377 |       1.427 |        0.704 |      12 |    197
    SP100-ciciot-quad-96bWc-r56462             |  50 |   +0.422 |   -0.342 |   +0.328 |   +0.813 |   -0.300 |   +0.666 |      1.576 |       1.470 |        0.391 |       4 |    176
    IDSXD-unswr-quad-64b-B05-AC-r20403         |  50 |   -0.076 |   -0.026 |   +0.985 |   +0.937 |   +0.362 |   +0.985 |      0.336 |       0.348 |        0.020 |      14 |     43
    IDSXD-unswr-quad-64b-B05-CE-r20403         |  50 |   -0.149 |   -0.049 |   +0.985 |   +0.938 |   +0.304 |   +0.985 |      0.439 |       0.345 |        0.020 |      13 |     43
    IDSXD-unswr-quad-64b-B10-AC-r20403         |  50 |   -0.098 |   -0.043 |   +0.991 |   +0.932 |   +0.268 |   +0.991 |      0.504 |       0.356 |        0.020 |      14 |     44
    IDSXD-unswr-quad-64b-B10-CE-r20403         |  50 |   -0.095 |   -0.110 |   +0.990 |   +0.929 |   +0.376 |   +0.990 |      0.321 |       0.273 |        0.020 |      14 |     43
    IDSXD-unswr-quad-64b-B15-AC-r20403         |  50 |   -0.048 |   -0.052 |   +0.986 |   +0.929 |   +0.259 |   +0.986 |      0.316 |       0.278 |        0.017 |      13 |     46
    IDSXD-unswr-quad-64b-B15-CE-r20403         |  50 |   -0.092 |   +0.074 |   +0.992 |   +0.937 |   +0.432 |   +0.992 |      0.341 |       0.266 |        0.014 |      12 |     43
    IDSXD-unswr-quad-64b-CE20-r20403           |  50 |   -0.066 |   +0.111 |   +0.990 |   +0.950 |   +0.279 |   +0.990 |      0.512 |       0.348 |        0.020 |      14 |     59
    IDSXD-unswr-quad-64b-Wb-CTRL-r20403        |  50 |   -0.061 |   +0.103 |   +0.994 |   +0.928 |   +0.132 |   +0.994 |      0.330 |       0.341 |        0.014 |      12 |     50
    SP100-unswr-quad-64bWb-r10596              |  50 |   +0.045 |   +0.129 |   +0.128 |   +0.804 |   +0.735 |   +0.995 |      0.429 |       0.352 |        0.000 |       1 |     44
    SP100-unswr-quad-64bWb-r20361              |  50 |   +0.020 |   +0.078 |   -0.051 |   +0.741 |   +0.424 |   +0.992 |      0.372 |       0.301 |        0.000 |       1 |     86
    SP100-unswr-quad-64bWb-r22224              |  50 |   +0.339 |   +0.127 |   +0.184 |   +0.882 |   +0.521 |   +0.994 |      5.065 |       4.482 |        0.001 |       2 |     57
    SP100-unswr-quad-64bWb-r32732              |  50 |   +0.164 |   +0.066 |   +0.168 |   +0.859 |   +0.645 |   +0.995 |      5.369 |       4.727 |        0.004 |       2 |     50
    SP100-unswr-quad-64bWb-r34524              |  50 |   +0.084 |   -0.042 |   +0.077 |   +0.827 |   +0.594 |   +0.993 |      0.392 |       0.355 |        0.000 |       1 |     59
    SP100-unswr-quad-64bWb-r46247              |  50 |   +0.190 |   +0.039 |   +0.181 |   +0.943 |   +0.570 |   +0.995 |      3.305 |       3.485 |        0.002 |       4 |     56
    SP100-unswr-quad-64bWb-r49337              |  50 |   +0.130 |   +0.154 |   +0.052 |   +0.833 |   +0.524 |   +0.997 |      1.423 |       1.339 |        0.000 |       1 |     65
    SP100-unswr-quad-64bWb-r49648              |  50 |   +0.004 |   -0.098 |   +0.050 |   +0.779 |   +0.622 |   +0.994 |      0.233 |       0.221 |        0.017 |      18 |     65
    IDSXD-unswt-quad-16b-B05-AC-r20403         |  50 |   +0.314 |   -0.109 |   +0.565 |   +0.374 |   -0.243 |   +0.469 |      0.306 |       0.651 |        0.317 |      20 |     25
    IDSXD-unswt-quad-16b-B05-CE-r20403         |  50 |   +0.453 |   +0.119 |   +0.665 |   +0.361 |   -0.079 |   +0.663 |      0.353 |       0.594 |        0.052 |       4 |     16
    IDSXD-unswt-quad-16b-B10-AC-r20403         |  50 |   +0.517 |   +0.034 |   +0.569 |   +0.111 |   +0.024 |   +0.597 |      0.322 |       0.716 |        0.296 |      17 |     18
    IDSXD-unswt-quad-16b-B10-CE-r20403         |  50 |   +0.500 |   +0.052 |   +0.630 |   +0.308 |   -0.020 |   +0.610 |      0.553 |       0.555 |        0.231 |      30 |     20
    IDSXD-unswt-quad-16b-B15-AC-r20403         |  50 |   +0.684 |   +0.140 |   +0.516 |   +0.309 |   -0.120 |   +0.610 |      0.457 |       0.562 |        0.065 |       2 |     17
    IDSXD-unswt-quad-16b-B15-CE-r20403         |  50 |   +0.673 |   +0.016 |   +0.596 |   +0.109 |   -0.055 |   +0.636 |      0.679 |       0.855 |        0.000 |       1 |     33
    IDSXD-unswt-quad-16b-CE20-r20403           |  50 |   +0.629 |   +0.162 |   +0.470 |   +0.627 |   +0.112 |   +0.538 |      1.146 |       0.828 |        0.334 |      30 |     24
    IDSXD-unswt-quad-16b-Wb-CTRL-r20403        |  50 |   +0.527 |   +0.008 |   +0.437 |   +0.235 |   +0.005 |   +0.533 |      0.958 |       0.607 |        0.273 |      20 |     23
    SP100-unswt-quad-16bWb-r25052              |  50 |   +0.085 |   -0.161 |   +0.007 |   -0.255 |   +0.073 |   +0.823 |      1.389 |       4.674 |        4.674 |      50 |     13
    SP100-unswt-quad-16bWb-r34952              |  50 |   +0.069 |   +0.126 |   +0.146 |   -0.141 |   +0.169 |   +0.892 |      1.677 |       6.706 |        2.417 |      46 |     16
    SP100-unswt-quad-16bWb-r35879              |  50 |   +0.258 |   +0.244 |   +0.102 |   +0.517 |   +0.476 |   +0.733 |     21.491 |      11.730 |        0.000 |       1 |     20
    SP100-unswt-quad-16bWb-r58983              |  50 |   +0.240 |   -0.036 |   +0.159 |   +0.066 |   +0.329 |   +0.868 |      1.376 |       3.563 |        2.163 |      45 |     17
    SP100-unswt-quad-16bWb-r63749              |  50 |   +0.105 |   -0.114 |   -0.007 |   -0.005 |   +0.320 |   +0.727 |     20.709 |      11.810 |        5.967 |      48 |     15
    SP100-unswt-quad-16bWb-r67145              |  50 |   +0.282 |   +0.162 |   +0.209 |   +0.155 |   +0.509 |   +0.831 |     20.482 |      12.469 |        0.593 |      34 |     15
    SP100-unswt-quad-16bWb-r72261              |  50 |   +0.395 |   +0.070 |   +0.067 |   +0.032 |   +0.081 |   +0.888 |      1.173 |       8.285 |        0.266 |       5 |     14
    SP100-unswt-quad-16bWb-r77715              |  50 |   +0.308 |   -0.102 |   +0.228 |   +0.176 |   +0.351 |   +0.843 |     22.281 |      14.059 |        1.128 |      45 |     20
    
    Pooled per dataset: mean per-run rho [bootstrap 95% CI over runs] (N runs)
    ### unswr (runs=16, genomes/run=50)
    fit_F1    +0.018 [-0.042,+0.085] (N=16)  min -0.149 max +0.339
    fit_FPR   +0.029 [-0.013,+0.070] (N=16)  min -0.110 max +0.154
    fit_CE    +0.544 [+0.324,+0.767] (N=16)  min -0.051 max +0.994
    kF1_F1    +0.884 [+0.852,+0.914] (N=16)  min +0.741 max +0.950
    kFPR_FPR  +0.440 [+0.362,+0.521] (N=16)  min +0.132 max +0.735
    kCE_CE    +0.992 [+0.990,+0.994] (N=16)  min +0.985 max +0.997
    regret_F1   0.011±0.009 pp (N=16)
    regret_FPR  0.019±0.027 pp (N=16)
    spread_hoF1 1.114±1.586 pp (N=16)
    spread_kF1  1.230±1.732 pp (N=16)
    held-out F1 rank of the fitness-best genome: median 12 of ~50 (ranks [1, 1, 1, 1, 2, 2, 4, 12, 12, 13, 13, 14, 14, 14, 14, 18])
    ### unswt (runs=16, genomes/run=50)
    fit_F1    +0.378 [+0.279,+0.473] (N=16)  min +0.069 max +0.684
    fit_FPR   +0.038 [-0.018,+0.093] (N=16)  min -0.161 max +0.244
    fit_CE    +0.335 [+0.220,+0.447] (N=16)  min -0.007 max +0.665
    kF1_F1    +0.186 [+0.078,+0.295] (N=16)  min -0.255 max +0.627
    kFPR_FPR  +0.121 [+0.019,+0.226] (N=16)  min -0.243 max +0.509
    kCE_CE    +0.704 [+0.637,+0.771] (N=16)  min +0.469 max +0.892
    regret_F1   1.173±1.790 pp (N=16)
    regret_FPR  6.029±8.036 pp (N=16)
    spread_hoF1 4.917±5.133 pp (N=16)
    spread_kF1  5.960±9.129 pp (N=16)
    held-out F1 rank of the fitness-best genome: median 25 of ~50 (ranks [1, 1, 2, 4, 5, 17, 20, 20, 30, 30, 34, 45, 45, 46, 48, 50])
    ### cicids (runs=10, genomes/run=50)
    fit_F1    +0.829 [+0.802,+0.860] (N=10)  min +0.766 max +0.938
    fit_FPR   +0.013 [-0.086,+0.126] (N=10)  min -0.172 max +0.363
    fit_CE    +0.867 [+0.792,+0.929] (N=10)  min +0.607 max +0.993
    kF1_F1    +0.931 [+0.906,+0.952] (N=10)  min +0.857 max +0.967
    kFPR_FPR  +0.486 [+0.386,+0.588] (N=10)  min +0.222 max +0.753
    kCE_CE    +0.994 [+0.992,+0.996] (N=10)  min +0.987 max +0.997
    regret_F1   0.018±0.023 pp (N=10)
    regret_FPR  0.106±0.050 pp (N=10)
    spread_hoF1 3.335±8.301 pp (N=10)
    spread_kF1  3.412±8.565 pp (N=10)
    held-out F1 rank of the fitness-best genome: median 2 of ~50 (ranks [1, 1, 1, 1, 2, 2, 2, 3, 14, 21])
    ### ciciot (runs=10, genomes/run=50)
    fit_F1    +0.417 [+0.315,+0.542] (N=10)  min +0.236 max +0.849
    fit_FPR   -0.312 [-0.398,-0.230] (N=10)  min -0.524 max -0.115
    fit_CE    +0.467 [+0.240,+0.696] (N=10)  min -0.106 max +0.965
    kF1_F1    +0.668 [+0.547,+0.767] (N=10)  min +0.222 max +0.858
    kFPR_FPR  +0.303 [+0.013,+0.583] (N=10)  min -0.300 max +0.812
    kCE_CE    +0.738 [+0.629,+0.842] (N=10)  min +0.422 max +0.989
    regret_F1   0.298±0.229 pp (N=10)
    regret_FPR  3.027±1.316 pp (N=10)
    spread_hoF1 1.527±0.735 pp (N=10)
    spread_kF1  1.409±0.399 pp (N=10)
    held-out F1 rank of the fitness-best genome: median 4 of ~50 (ranks [1, 1, 3, 3, 3, 4, 4, 6, 9, 12])
    
    compute: 52 flows, 1.50 h re-score wall
```

### IDS-25 verdict

**Not supported as the cause of the UNSW-random plateau — provisional (grid top-50 only, one
population per run, 16 runs).** On UNSW-random the k-fold estimate itself ranks held-out almost
perfectly (CE rho 0.992, F1 rho 0.884); the composite fitness does not rank held-out F1 (rho
+0.02), but the cost of that is 0.011±0.009 pp F1 because every top-50 genome is already at the
ceiling. CICIDS shows no mismatch (F1 rho 0.83, regret 0.018 pp).

Where mismatch IS real:
- **UNSW-temporal**: k-fold F1 barely predicts held-out F1 (rho 0.19 [0.08,0.30]); the fitness-best
  grid genome is held-out rank >=45/50 in 5/16 runs, regret up to 6.0 pp (SP100 runs). This is the
  temporal split's train->test shift (k-fold partitions the past; the test is the future), i.e. a
  dataset property, not a GA defect — but it means search fitness on UNSW-temporal is a weak proxy.
- **CIC-IoT FPR**: fitness is anti-correlated with held-out val_cal FPR (rho -0.31 [-0.40,-0.23]);
  the fitness-best genome costs 3.0±1.3 pp FPR vs the best-FPR genome. F1 regret is 0.30 pp.
  Note val_cal picks the F1-optimal threshold, so its FPR is a by-product, not what the genome was
  selected for — read with the Pareto view, not as an F1 failure.

## Caveats

- **Selection on test.** View A's best-of-35 and View B's best-of-pool pick cells by TEST metrics.
  Both phases get the same selection, so the paired deltas are fair, but absolute levels are
  optimistic. The val_cal conventional cell is the honest per-genome number and agrees in sign.
- **N.** SP100 has 29/100 completed per dataset (71 cancelled); View A loses runs whose grid rows
  were inherited and not re-validated (unswr 10, unswt 24, cicids 41, ciciot 42 dropped). Re-validating
  those grid rows would roughly double CICIDS/CIC-IoT N; given 32/32 and 41/41 sign agreement, it
  cannot change the verdict.
- **IDS-25 range restriction.** Only the grid's top-50 (of ~94) carry connections, so rho is
  measured among already-good genomes — the relevant range for GA selection, but it understates
  agreement over the full grid. GA-phase populations were not re-scored (the GA's within-run
  mismatch is the IDS-26/next step if wanted: same script on `exp_01/ga_neurons.json.gz`).
- **Stratified subset** (52 of ~260 eligible flows), chosen by flow id / arm order, not by outcome.
- IDSXD-cicids/ciciot and CIC-IoT era confounds (memory: IDSXD unpaired) do not touch a within-run
  Grid-vs-GA pairing.
