# =====================================================================
# SECTION 12 — IDSXD AC/CE COHORT, cicids2017 CELL (COMPLETE, 24/24)
# =====================================================================

Appended 29/08/2026. Cohort `IDSXD-cicids-quad-96b-*`, flows 5862-5885,
**24/24 completed, 0 failed** — the first of three dataset cells in IDSXD
(ciciot 96b and unswr 64b are still queued). 8 arms x 3 seeds (20403/04/05).

Config, identical across all 24 except the fitness weights and the seed
(verified by diffing `flows.config_json.params`):

```
dataset cicids2017 · split random_3way · binary · QUAD · ids_n_bits 96 · top20
K-fold 5 on the 80% train, ids_kfold_per_gen 5 · held-out = 10% TEST (Protocol v2)
population 50 · ga_generations 250 · patience 5 · max_bits 34 · min_bits 4
fitness_aggregation = DESIRABILITY · fitness_calculator = harmonic_rank
fitness_ce_anchor_normalized = 0.2128 · fitness_percentile 0.75 · zrank_clamp 3.0
total 37.7 h · avg 94.4 min/run · last done 29/08/2026 09:22 UTC
```

The eight arms (the only thing that varies, besides seed):

```
arm       w_ce   w_acc   w_f1  w_fpr
B05-AC   0.225   0.675   0.05   0.05
B05-CE   0.675   0.225   0.05   0.05
B10-AC   0.200   0.600   0.10   0.10
B10-CE   0.600   0.200   0.10   0.10
B15-AC   0.175   0.525   0.15   0.15
B15-CE   0.525   0.175   0.15   0.15
CE20     0.200   0.100   0.30   0.40
Wa-CTRL  0.350   0.300   0.30   0.05
```

AC/CE pairs swap `w_ce` <-> `w_acc` at three balance levels (B05/B10/B15 =
the f1+fpr weight held at 0.05/0.10/0.15). That gives **9 matched pairs**.

## 12A. THE HEADLINE — this cell is SATURATED and cannot discriminate

```
best_f1 / val_cal held-out / GA Neurons, mean±SD over 3 seeds
arm       w_ce/w_acc/w_f1/w_fpr           F1%             FPR%             Acc%
B05-AC    0.225/0.675/0.05/0.05   99.430±0.107     0.232±0.101     99.639±0.068
B05-CE    0.675/0.225/0.05/0.05   99.424±0.096     0.249±0.096     99.636±0.061
B10-AC    0.2/0.6/0.1/0.1         99.469±0.069     0.206±0.084     99.664±0.044
B10-CE    0.6/0.2/0.1/0.1         99.453±0.059     0.169±0.108     99.655±0.038
B15-AC    0.175/0.525/0.15/0.15   99.481±0.057     0.154±0.055     99.672±0.036
B15-CE    0.525/0.175/0.15/0.15   99.435±0.084     0.232±0.074     99.643±0.054
CE20      0.2/0.1/0.3/0.4         99.519±0.033     0.118±0.008     99.697±0.020
Wa-CTRL   0.35/0.3/0.3/0.05       99.474±0.093     0.208±0.081     99.667±0.059
```

**F1 spans 99.424-99.519 across all EIGHT arms — a 0.095 pp total range, smaller
than most arms' own seed SD.** FPR spans 0.118-0.249%. There is no headroom on
this dataset/split for a fitness-weight change to move, so every comparison below
is reported for completeness and none of it licenses a claim.

## 12B. PRE-REGISTERED READ — AC vs CE, paired by (band, seed), n=9

delta = CE minus AC. GA Neurons / best_f1 / val_cal held-out. Exact sign-flip.

```
  F1     band       s20403      s20404      s20405   AC mean   CE mean    delta
         B05        -0.012      -0.029      +0.026    99.430    99.424   -0.005
         B10        +0.000      +0.008      -0.058    99.469    99.453   -0.016
         B15        +0.106      -0.162      -0.083    99.481    99.435   -0.046
         POOLED n=9:  -0.023pp   SD 0.075   CE better 4/9   p=0.4102

  FPR    B05        +0.019      -0.008      +0.042     0.232     0.249   +0.018
         B10        -0.011      -0.021      -0.079     0.206     0.169   -0.037
         B15        -0.059      +0.168      +0.123     0.154     0.232   +0.077
         POOLED n=9:  +0.019pp   SD 0.081   CE better 5/9   p=0.5000

  Acc    B05        -0.008      -0.018      +0.016    99.639    99.636   -0.003
         B10        +0.000      +0.005      -0.035    99.664    99.655   -0.010
         B15        +0.067      -0.103      -0.053    99.672    99.643   -0.030
         POOLED n=9:  -0.014pp   SD 0.048   CE better 4/9   p=0.4102
```

**NULL on all three.** 4/9, 5/9, 4/9 — coin flips — with every pooled delta an
order of magnitude below its own SD.

⚠️ **This does NOT confirm the banked "cicids REVERSES the AC/CE prediction"
note.** It is null, not reversed. The earlier reading was n=2 and had no power to
distinguish the two. Update any text carrying the "reverses" claim for cicids.

## 12C. Full threshold-mode breakdown — 5 genome types x 7 modes

Grid Search : 300±162 neurons | 34.0±0.0 bits
GA Neurons  : 145±89 neurons  | 34.0±0.1 bits
(bits pinned at `max_bits`=34 in all 24 runs, SD 0.0-0.1 — nothing here speaks
to bits)


best_f1  (runs: 24/24)
```
  mode                      F1 Grid      F1 GA |    FPR Grid     FPR GA |    Acc Grid     Acc GA
  ----------------------------------------------------------------------------------------------------
  train_cal              99.30±0.02 99.46±0.07 |   0.31±0.02  0.19±0.08 |  99.56±0.01 99.66±0.04
  fixed_05               99.04±0.05 99.14±0.12 |   0.58±0.04  0.52±0.09 |  99.39±0.03 99.45±0.08
  platt                  99.28±0.04 99.35±0.09 |   0.31±0.02  0.32±0.07 |  99.54±0.03 99.59±0.05
  beta                   99.26±0.03 99.39±0.07 |   0.28±0.02  0.24±0.06 |  99.53±0.02 99.62±0.05
  empirical              98.83±0.20 99.16±0.28 |   0.14±0.04  0.12±0.06 |  99.27±0.13 99.47±0.17
  empirical_cumulative   99.30±0.02 99.46±0.07 |   0.30±0.02  0.19±0.07 |  99.56±0.01 99.66±0.05
  val_cal                99.30±0.02 99.46±0.07 |   0.31±0.03  0.20±0.08 |  99.56±0.01 99.66±0.05
```

best_fpr  (runs: 24/24)
```
  mode                      F1 Grid      F1 GA |    FPR Grid     FPR GA |    Acc Grid     Acc GA
  ----------------------------------------------------------------------------------------------------
  train_cal              99.21±0.07 99.43±0.07 |   0.25±0.06  0.19±0.07 |  99.50±0.04 99.64±0.05
  fixed_05               98.91±0.05 99.14±0.11 |   0.67±0.04  0.52±0.08 |  99.30±0.03 99.45±0.07
  platt                  99.17±0.05 99.33±0.08 |   0.35±0.01  0.33±0.06 |  99.47±0.03 99.57±0.05
  beta                   99.18±0.07 99.38±0.08 |   0.28±0.01  0.25±0.07 |  99.48±0.05 99.61±0.05
  empirical              98.30±0.18 99.13±0.26 |   0.05±0.02  0.13±0.07 |  98.95±0.11 99.46±0.16
  empirical_cumulative   99.19±0.10 99.43±0.08 |   0.20±0.01  0.18±0.08 |  99.49±0.06 99.64±0.05
  val_cal                99.21±0.07 99.43±0.07 |   0.27±0.07  0.19±0.08 |  99.50±0.04 99.64±0.05
```

best_acc  (runs: 24/24)
```
  mode                      F1 Grid      F1 GA |    FPR Grid     FPR GA |    Acc Grid     Acc GA
  ----------------------------------------------------------------------------------------------------
  train_cal              99.30±0.02 99.46±0.07 |   0.31±0.02  0.19±0.08 |  99.56±0.01 99.66±0.04
  fixed_05               99.05±0.05 99.14±0.12 |   0.57±0.04  0.52±0.09 |  99.39±0.03 99.45±0.08
  platt                  99.28±0.04 99.35±0.09 |   0.31±0.02  0.32±0.07 |  99.54±0.03 99.59±0.05
  beta                   99.26±0.03 99.39±0.07 |   0.28±0.02  0.24±0.06 |  99.53±0.02 99.62±0.05
  empirical              98.84±0.20 99.16±0.28 |   0.13±0.04  0.12±0.06 |  99.27±0.12 99.47±0.17
  empirical_cumulative   99.31±0.02 99.46±0.07 |   0.30±0.02  0.19±0.07 |  99.56±0.01 99.66±0.05
  val_cal                99.30±0.02 99.46±0.07 |   0.31±0.03  0.20±0.08 |  99.56±0.01 99.66±0.05
```

best_ce  (runs: 24/24)
```
  mode                      F1 Grid      F1 GA |    FPR Grid     FPR GA |    Acc Grid     Acc GA
  ----------------------------------------------------------------------------------------------------
  train_cal              99.27±0.02 99.44±0.08 |   0.33±0.03  0.20±0.08 |  99.54±0.01 99.65±0.05
  fixed_05               99.05±0.06 99.15±0.12 |   0.57±0.04  0.51±0.09 |  99.39±0.04 99.46±0.08
  platt                  99.23±0.02 99.31±0.09 |   0.32±0.01  0.35±0.07 |  99.51±0.01 99.56±0.06
  beta                   99.24±0.03 99.40±0.08 |   0.28±0.02  0.23±0.07 |  99.52±0.02 99.62±0.05
  empirical              98.69±0.26 99.10±0.26 |   0.11±0.05  0.11±0.05 |  99.18±0.16 99.43±0.16
  empirical_cumulative   99.27±0.03 99.44±0.08 |   0.31±0.03  0.20±0.08 |  99.54±0.02 99.65±0.05
  val_cal                99.27±0.02 99.44±0.08 |   0.33±0.03  0.20±0.08 |  99.54±0.01 99.65±0.05
```

best_fitness  (runs: 24/24)
```
  mode                      F1 Grid      F1 GA |    FPR Grid     FPR GA |    Acc Grid     Acc GA
  ----------------------------------------------------------------------------------------------------
  train_cal              99.27±0.02 99.45±0.07 |   0.32±0.04  0.19±0.08 |  99.54±0.01 99.65±0.05
  fixed_05               99.04±0.06 99.16±0.12 |   0.57±0.04  0.51±0.09 |  99.39±0.04 99.46±0.08
  platt                  99.23±0.03 99.32±0.09 |   0.32±0.01  0.34±0.07 |  99.51±0.02 99.57±0.06
  beta                   99.24±0.03 99.39±0.07 |   0.28±0.02  0.23±0.07 |  99.52±0.02 99.61±0.05
  empirical              98.62±0.33 99.10±0.28 |   0.10±0.05  0.11±0.05 |  99.14±0.20 99.43±0.17
  empirical_cumulative   99.27±0.03 99.45±0.07 |   0.30±0.04  0.19±0.08 |  99.54±0.02 99.65±0.05
  val_cal                99.27±0.02 99.45±0.07 |   0.32±0.04  0.19±0.08 |  99.54±0.02 99.65±0.05
```

## 12D. What this cell DOES support

**GA earns its keep, and gets smaller doing it.** GA Neurons beats Grid Search on
every genome_type x mode combination — +0.16 pp F1 and -0.12 pp FPR on
best_f1/val_cal — while using **145±89 neurons against the grid's 300±162**.
Half the genome, better numbers, on 24/24 runs.

**`empirical` is a distinct operating point, not a better one.** It gives the
lowest FPR of the seven modes (0.05±0.02% on best_fpr/Grid) and the worst F1
(98.30±0.18%). Same shape as the QSR lesson: an FPR win bought by shedding
predictions, not by detecting better.

**CE20 leads all three columns with the tightest variance** (SD 0.033 F1,
0.008 FPR vs ~0.1 elsewhere) — but ⚠️ **its name is misleading.** Its vector is
`0.2/0.1/0.3/0.4`, i.e. **f1+fpr-weighted, not CE-weighted**. Whatever advantage
it shows here is attributable to the 0.40 FPR weight, not to cross-entropy. Do
NOT carry the IDSZ "CE20" framing onto this arm without re-reading its weights.

## 12E. What this cell does NOT support

- Any AC/CE claim in either direction — all three metrics are null (12B).
- The banked "cicids reverses" note (12B) — retire or restate it.
- Anything about bits — the band was pinned at 34 in all 24 runs.
- Any cross-dataset generalisation. This is 1 of 3 IDSXD cells; ciciot 96b and
  unswr 64b have not run. **Do not pool these 24 with the other cells** until
  they land, and expect the saturation to differ per dataset.
- Any ranking of the eight arms. A 0.095 pp spread across eight arms on a
  saturated dataset is arm-ordering noise, not a Pareto front.

## 12F. Methodological note — this cohort runs DESIRABILITY

All 106 IDSXD runs (this cell included) use `fitness_aggregation = desirability`
with `fitness_calculator = harmonic_rank`, NOT zscore. That is the setting the
IDSD A/B selected (desirability beat zscore on F1 5-0, +0.281 pp, on
unsw-nb15 temporal_3way 16b) and it has been the IDS default since.

⚠️ There is **no viability-gate arm on the IDS side.** `--gate-stable` /
`--gate-err` are controller-only (attitude feasibility, Deb's rules); the IDS
fitness has no analogous hard constraint. So the controller's GATE-vs-DESIR
ladder has **no IDS counterpart** — the IDS A/B was aggregation-only
(desirability vs zscore). Do not read the two comparisons as the same question.
