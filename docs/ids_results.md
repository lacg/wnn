# PUB50 8b CIC-IoT Random — FINAL Results (Generated 12/04/2026 08:36)

    Completed : 112/112  ✅ BATCH COMPLETE

### Best individual genomes — CIC-IoT-2023 1.3M Random 8b

    Metric                   |      F1 |     FPR |     Acc | Phase Genome        Threshold       |    N
    -------------------------+---------+---------+---------+-----------------------------------+-----
    Best F1 (any FPR)        |  83.04% |  20.18% |  90.62% |  GA best_f1       empirical_cumulative |  198
    Best F1 (FPR<14%)        |  82.76% |  13.54% |  89.84% |  GA best_ce       empirical_cumulative |  374
    Best F1 (FPR<10%)        |  82.45% |   9.86% |  89.29% |  GA best_acc      fixed_05        |  245
    Best F1 (FPR<5%)         |  81.14% |   4.15% |  87.75% |  GA best_ce       train_cal       |  299
    Best FPR (F1>70%)        |  70.34% |   0.03% |  76.93% |  GS best_fpr      train_cal       |    5
    Best FPR (F1>80%)        |  80.12% |   2.73% |  86.75% |  GA best_fpr      val_cal         |  170
    Best Acc (any FPR)       |  81.89% |  33.10% |  91.05% |  GA best_ce       platt           |  496
    Best Acc (FPR<14%)       |  82.76% |  13.54% |  89.84% |  GA best_ce       empirical_cumulative |  374
    --- baselines ---
    RF (top-20, raw)         |  85.53% |  25.18% |  92.71% |  sklearn predict()                |    —
    XGBoost (top-20, raw)    |  84.13% |  28.34% |  92.07% |  sklearn predict()                |    —
    RF (all 39, raw)         |  86.08% |  24.00% |  92.96% |  sklearn predict()                |    —
    Neto RF (46M)            |  96.53% |     —   |  99.68% |  StandardScaler, F1 avg unspec    |    —

## best_fitness  (GS: 111 runs | GA: 112 runs)
    Grid Search : 397±96 neurons | 34±1 bits
    GA Neurons  : 322±124 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.55±0.43 80.08±0.48 | 4.51±1.65  4.26±0.49 |86.42±0.49 86.85±0.42
    fixed_05             |81.08±0.17 81.65±0.43 | 9.04±0.96  9.01±1.06 |88.13±0.20 88.58±0.38
    platt                |80.98±0.58 81.41±0.52 |33.73±3.19 33.15±3.11 |90.51±0.14 90.74±0.18
    beta                 |80.82±0.66 81.12±0.64 |34.83±3.76 34.88±3.54 |90.53±0.13 90.74±0.15
    empirical            |68.40±13.35 68.46±14.59 |64.72±22.20 63.14±25.16 |88.71±2.18 88.78±2.43
    empirical_cumulative |81.76±0.70 82.29±0.67 |22.61±6.46 20.99±6.40 |89.94±0.35 90.17±0.38
    val_cal              |79.31±0.39 79.93±0.47 | 4.09±1.59  3.95±0.52 |86.17±0.46 86.70±0.42

## best_f1  (GS: 111 runs | GA: 112 runs)
    Grid Search : 397±96 neurons | 34±1 bits
    GA Neurons  : 322±124 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.56±0.43 80.08±0.48 | 4.51±1.65  4.26±0.49 |86.42±0.49 86.85±0.42
    fixed_05             |81.08±0.17 81.65±0.43 | 9.02±0.96  9.01±1.06 |88.12±0.20 88.58±0.38
    platt                |80.98±0.58 81.41±0.52 |33.73±3.19 33.15±3.11 |90.51±0.14 90.74±0.18
    beta                 |80.81±0.65 81.12±0.64 |34.86±3.74 34.88±3.54 |90.53±0.13 90.74±0.15
    empirical            |68.39±13.35 68.46±14.59 |64.73±22.20 63.14±25.16 |88.71±2.18 88.78±2.43
    empirical_cumulative |81.76±0.70 82.29±0.67 |22.63±6.45 20.99±6.40 |89.94±0.35 90.17±0.38
    val_cal              |79.31±0.39 79.93±0.47 | 4.08±1.59  3.95±0.52 |86.17±0.46 86.70±0.42

## best_fpr  (GS: 111 runs | GA: 112 runs)
    Grid Search : 148±165 neurons | 14±10 bits
    GA Neurons  : 282±174 neurons | 24±11 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |74.50±1.82 77.65±2.40 | 3.26±1.11  3.78±0.77 |81.57±1.96 84.59±2.30
    fixed_05             |74.64±6.29 79.02±3.80 |28.05±20.81 14.09±12.23 |85.66±1.59 87.09±1.58
    platt                |71.17±8.15 77.52±6.31 |48.10±22.17 38.87±14.89 |86.44±1.98 89.07±2.06
    beta                 |67.24±9.57 75.93±8.35 |61.24±23.04 44.07±19.11 |86.67±1.47 89.07±2.11
    empirical            |61.39±12.87 64.80±13.59 |72.46±25.47 69.70±23.49 |86.40±1.40 87.73±2.16
    empirical_cumulative |73.34±8.91 78.92±5.40 |27.64±25.15 24.15±15.44 |84.76±5.94 88.19±2.51
    val_cal              |74.51±1.54 77.61±2.27 | 3.38±2.25  3.63±0.69 |81.60±1.46 84.54±2.16

## best_acc  (GS: 111 runs | GA: 112 runs)
    Grid Search : 392±98 neurons | 34±1 bits
    GA Neurons  : 341±122 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.56±0.43 79.96±0.53 | 4.53±1.66  4.23±0.46 |86.43±0.49 86.74±0.47
    fixed_05             |81.07±0.16 81.55±0.49 | 9.00±0.95  9.06±1.23 |88.11±0.19 88.51±0.43
    platt                |80.97±0.57 81.33±0.49 |33.78±3.21 33.32±2.67 |90.51±0.14 90.70±0.19
    beta                 |80.82±0.65 81.07±0.55 |34.75±3.73 34.88±3.06 |90.52±0.14 90.70±0.18
    empirical            |68.96±13.28 67.91±14.56 |63.57±22.14 64.34±24.99 |88.80±2.17 88.68±2.42
    empirical_cumulative |81.77±0.70 82.19±0.69 |22.65±6.36 21.24±6.32 |89.95±0.35 90.13±0.38
    val_cal              |79.31±0.39 79.78±0.54 | 4.08±1.59  3.90±0.51 |86.17±0.46 86.56±0.48

## best_ce  (GS: 112 runs | GA: 112 runs)
    Grid Search : 383±114 neurons | 33±1 bits
    GA Neurons  : 324±130 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.42±0.43 80.00±0.58 | 4.46±1.65  4.19±0.51 |86.30±0.49 86.78±0.52
    fixed_05             |81.05±0.21 81.63±0.51 | 9.25±1.04  9.12±1.58 |88.12±0.24 88.58±0.48
    platt                |80.86±0.57 81.34±0.78 |34.25±3.27 33.09±3.84 |90.49±0.15 90.69±0.20
    beta                 |80.71±0.66 81.11±0.83 |35.27±3.51 34.59±4.24 |90.50±0.15 90.70±0.21
    empirical            |67.87±13.98 67.62±14.99 |65.07±23.41 64.47±25.66 |88.62±2.28 88.63±2.48
    empirical_cumulative |81.70±0.76 82.21±0.73 |22.62±7.21 21.20±6.86 |89.91±0.36 90.14±0.41
    val_cal              |79.19±0.44 79.78±0.59 | 4.02±1.61  3.83±0.50 |86.06±0.50 86.56±0.53


## CIC-IoT-2023 Baseline Comparison (F1-macro vs F1-weighted)

    All models use binary classification (normal vs attack).
    F1-macro treats both classes equally; F1-weighted favours the 97% attack majority.

### Full 46M dataset

    Source               | F1-macro | F1-weighted |    FPR |    Acc | Data / Features
    ---------------------|----------|-------------|--------|--------|----------------
    Neto RF              |  96.53%† |      —      |     —  | 99.68% | 46M, all, StandardScaler
    Our RF (all 39)      |  92.64%  |   99.18%    | 13.48% | 99.18% | 46M, all, no scaler
    Our XGBoost (all 39) |  91.67%  |   99.07%    | 14.80% | 99.06% | 46M, all, no scaler
    WNN 245n×32b         |  84.79%  |   97.99%    |  1.59% | 97.68% | 46M, top-20, 8b thermo
    WNN 400n×8b          |  83.13%  |   97.72%    |  2.06% | 97.33% | 46M, top-20, 8b thermo
    WNN 6n×16b           |  78.89%  |      —      |  1.94% | 96.23% | 46M, top-20, 8b thermo, FILTER

    † Neto et al. F1 averaging method unspecified; StandardScaler + different 80/20 split.

### 1.3M subsample (random 80/20) — FINAL 112 runs

    Source               | F1-macro | F1-weighted |    FPR |    Acc | Data / Features
    ---------------------|----------|-------------|--------|--------|----------------
    Our RF (all 39)      |  86.08%  |   92.95%    | 24.00% | 92.96% | 1.3M, all, no scaler
    Our RF (top-20)      |  85.53%  |   92.68%    | 25.18% | 92.71% | 1.3M, top-20, no scaler
    Our XGBoost (top-20) |  84.13%  |   92.07%    | 28.34% | 92.07% | 1.3M, top-20, no scaler
    WNN avg (train_cal)  |  80.08%  |      —      |  4.26% | 86.85% | 1.3M, top-20, n=112 FINAL
    WNN avg (fixed_05)   |  81.65%  |      —      |  9.01% | 88.58% | 1.3M, top-20, n=112 FINAL


## 46M 6n×16b First-Stage Filter — VALIDATED

    1.3M (PUB50 r020):  F1=73.70%  FPR=1.09%  Acc=80.56%  (train_cal)
    46M  (flow 1231):   F1=78.89%  FPR=1.94%  Acc=96.23%  (train_cal)
    Delta (46M vs 1.3M): F1 +5.19pp, FPR +0.85pp, Acc +15.67pp

    The Pareto point validates at scale: 46M training data lifts F1 by 5pp
    while FPR stays under 2%. Viable first-stage filter for cascade deployment.


## UNSW-NB15 Temporal Baselines + WNN Comparison

    Split: temporal (175K train / 82K test, official UNSW split)
    HF column names FIXED on 12/04/2026 — all configs now use 20/20 features.

### ML Baselines (raw features, FIXED 20/20)

    Source                    | F1-macro |    FPR |    Acc | Features
    --------------------------|----------|--------|--------|----------
    RF (all 42 features)      |  86.69%  | 26.58% | 87.30% | all
    XGBoost (all 42 features) |  86.71%  | 26.70% | 87.32% | all
    RF (top-20, FIXED)        |  86.41%  | 25.22% | 86.94% | top-20 (20/20)
    XGBoost (top-20, FIXED)   |  85.62%  | 27.29% | 86.24% | top-20 (20/20)
    Zoghi et al. 2024         | 85-90%   |    —   |    —   | literature range

    Note: 20/20 features improved RF by +1.27pp F1 over the old 13/20
    (85.14% → 86.41%), confirming the feature fix matters.

### OLD WNN Results (12 runs, 8b, OLD weights, 13/20 features)

    Threshold            |    F1    |    FPR    |   Acc
    ---------------------|----------|-----------|--------
    train_cal            |  82.53%  |  32.65%   | 83.55%
    fixed_05             |  82.57%  |  27.35%   | 83.58%
    val_cal (oracle)     |  87.96%  |  10.57%   | 88.03%

### NEW WNN Temporal Mini-Sweep (in progress, 20/20 features, NEW weights)

    Per-flow time: ~13 min (vs ~75 min on random). 175K temporal dataset.

### Full UNSW Temporal Comparison Table

    Source                         |    F1    |    FPR   |    Acc   | Notes
    -------------------------------|----------|----------|----------|------
    RF (top-20, raw)               |  86.41%  |  25.22%  |  86.94%  | baseline
    XGBoost (top-20, raw)          |  85.62%  |  27.29%  |  86.24%  | baseline
    RF (all 42, raw)               |  86.69%  |  26.58%  |  87.30%  | all features
    XGBoost (all 42, raw)          |  86.71%  |  26.70%  |  87.32%  | all features
    Zoghi & Serpen 2024            |  85-90%  |    —     |    —     | literature
                                   |          |          |          |
    OLD WNN val_cal (12r, 13ft)    |  87.96%  |  10.57%  |  88.03%  | oracle, OLD weights
    OLD WNN train_cal (12r, 13ft)  |  82.53%  |  32.65%  |  83.55%  | OLD weights
                                   |          |          |          |
    NEW 64b r001 GS best_fpr tc   |  88.31%  |  12.45%  |  88.42%  | train_cal, beats old oracle F1!
    NEW 64b r002 GS best_ce beta  |  87.86%  |   8.44%  |  87.90%  | low FPR
    NEW 64b r001 GA best_ce tc    |  87.91%  |  10.22%  |  87.97%  | GA didn't help much on 64b
    NEW 32b r002 GA best_fpr f05  |  89.07%  |   6.59%  |  89.10%  | ★ HIGHEST F1, beats everything
    NEW 32b r002 GA best_fpr beta |  82.80%  |   2.04%  |  82.85%  | ultra-low FPR mode
    NEW 32b r002 GA best_fpr platt|  71.08%  |   0.02%  |  72.00%  | near-zero FPR (1 in 5000)
    NEW 32b r002 GA best_fpr orac |  81.85%  |   3.66%  |  81.89%  |

    ★ f1241 (32b r002) GA best_fpr genome under fixed_05:
      89.07% F1 beats RF by +2.66pp AND old oracle by +1.11pp
      with FPR 3.8× lower than RF (6.59% vs 25.22%)

# UNSW-NB15 TEMPORAL 8b — 5-Table Results (Generated 13/04/2026 12:12)

    NEW weights (0.1/0.35/0.35/0.2), 20/20 features, fitness-aligned threshold
    Completed : 43/112  |  Remaining: 69
    Avg/run   : 8.0 min
    Latest    : 13/04/2026 12:10:46 ET
    ETA       : 13/04/2026 21:22:46 ET

### Best individual genomes — UNSW-NB15 TEMPORAL 8b

    Metric                   |      F1 |     FPR |     Acc | Phase Genome        Threshold       |    N
    -------------------------+---------+---------+---------+-----------------------------------+-----
    Best F1 (any FPR)        |  90.52% |   4.13% |  90.54% |  GS best_f1       empirical       |  100
    Best F1 (FPR<14%)        |  90.52% |   4.13% |  90.54% |  GS best_f1       empirical       |  100
    Best F1 (FPR<10%)        |  90.52% |   4.13% |  90.54% |  GS best_f1       empirical       |  100
    Best F1 (FPR<5%)         |  90.52% |   4.13% |  90.54% |  GS best_f1       empirical       |  100
    Best FPR (F1>70%)        |  70.71% |   0.00% |  71.68% |  GS best_acc      beta            |  500
    Best FPR (F1>80%)        |  81.80% |   0.22% |  81.91% |  GA best_f1       empirical_cumulative |  500
    Best Acc (any FPR)       |  90.52% |   4.13% |  90.54% |  GS best_f1       empirical       |  100
    Best Acc (FPR<14%)       |  90.52% |   4.13% |  90.54% |  GS best_f1       empirical       |  100
    --- baselines ---
    RF (top-20, raw)         |  86.41% |  25.22% |  86.94% |  sklearn predict()                |    —
    XGBoost (top-20, raw)    |  85.62% |  27.29% |  86.24% |  sklearn predict()                |    —
    RF (all 42, raw)         |  86.69% |  26.58% |  87.30% |  sklearn predict()                |    —
    XGBoost (all 42, raw)    |  86.71% |  26.70% |  87.32% |  sklearn predict()                |    —
    Zoghi & Serpen 2024      | 85-90%  |     —   |     —   |  literature range                 |    —

## best_fitness  (GS: 43 runs | GA: 43 runs)
    Grid Search : 351±133 neurons | 32±1 bits
    GA Neurons  : 323±139 neurons | 31±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.49±2.84 86.98±2.50 |14.22±5.91 13.46±4.42 |87.66±2.75 87.11±2.47
    fixed_05             |84.53±3.63 83.93±8.75 |21.06±16.31 15.80±14.49 |85.18±2.96 84.65±7.30
    platt                |84.21±4.91 83.11±4.84 |20.58±15.13 22.57±15.02 |84.85±4.37 83.84±4.41
    beta                 |83.22±4.87 82.70±5.62 |20.83±16.13 25.56±17.37 |83.92±4.44 83.67±4.58
    empirical            |76.71±12.93 75.22±13.48 |41.54±23.09 44.45±23.33 |79.67±8.63 78.57±9.05
    empirical_cumulative |82.12±4.57 81.94±6.00 |28.07±16.87 29.54±17.89 |83.17±3.88 83.16±4.74
    val_cal              |86.35±3.06 87.59±1.65 | 5.01±3.27  5.29±4.18 |86.37±3.03 87.62±1.63

## best_f1  (GS: 43 runs | GA: 43 runs)
    Grid Search : 351±133 neurons | 32±1 bits
    GA Neurons  : 323±139 neurons | 31±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.49±2.84 86.98±2.50 |14.22±5.91 13.46±4.42 |87.66±2.75 87.11±2.47
    fixed_05             |84.53±3.63 83.93±8.75 |21.06±16.31 15.80±14.49 |85.18±2.96 84.65±7.30
    platt                |84.21±4.91 83.11±4.84 |20.58±15.13 22.57±15.02 |84.85±4.37 83.84±4.41
    beta                 |83.22±4.87 82.70±5.62 |20.83±16.13 25.56±17.37 |83.92±4.44 83.67±4.58
    empirical            |76.71±12.93 75.22±13.48 |41.54±23.09 44.45±23.33 |79.67±8.63 78.57±9.05
    empirical_cumulative |82.12±4.57 81.94±6.00 |28.07±16.87 29.54±17.89 |83.17±3.88 83.16±4.74
    val_cal              |86.35±3.06 87.59±1.65 | 5.01±3.27  5.29±4.18 |86.37±3.03 87.62±1.63

## best_fpr  (GS: 43 runs | GA: 43 runs)
    Grid Search : 277±146 neurons | 30±5 bits
    GA Neurons  : 260±154 neurons | 29±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |86.45±2.61 84.99±9.07 |15.26±5.10 14.75±6.13 |86.62±2.53 85.48±7.13
    fixed_05             |83.28±8.27 82.23±8.41 |20.40±20.42 22.60±18.05 |84.29±5.48 83.28±6.57
    platt                |83.41±5.46 80.22±12.55 |21.01±17.56 22.19±16.91 |84.18±4.74 81.71±9.72
    beta                 |81.27±9.37 80.18±9.33 |26.99±21.62 30.32±17.22 |82.73±6.56 81.72±7.15
    empirical            |74.97±14.00 72.45±13.71 |44.10±25.05 50.45±22.16 |78.48±9.38 76.59±9.05
    empirical_cumulative |81.92±5.50 80.46±9.28 |30.57±16.13 29.42±17.87 |83.12±4.40 81.97±7.15
    val_cal              |85.46±8.83 84.79±4.54 | 6.60±6.01  7.91±6.83 |85.82±6.80 84.88±4.36

## best_acc  (GS: 43 runs | GA: 43 runs)
    Grid Search : 356±128 neurons | 32±1 bits
    GA Neurons  : 297±133 neurons | 31±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.45±2.85 86.48±3.11 |14.44±5.72 14.89±6.27 |87.62±2.76 86.67±2.96
    fixed_05             |84.60±3.64 82.10±10.88 |20.40±16.51 16.54±17.60 |85.23±2.97 83.14±8.96
    platt                |84.21±4.91 82.12±7.26 |20.34±15.35 22.47±17.94 |84.85±4.37 83.07±5.86
    beta                 |83.41±4.90 82.84±5.76 |20.02±16.14 24.54±17.01 |84.08±4.47 83.74±4.76
    empirical            |76.43±12.75 73.91±14.95 |42.47±22.34 44.06±24.09 |79.44±8.46 77.59±10.31
    empirical_cumulative |82.46±4.53 81.48±6.11 |28.36±16.48 30.15±18.12 |83.50±3.77 82.76±4.90
    val_cal              |86.31±3.04 87.17±2.23 | 5.02±3.27  5.46±4.36 |86.34±3.01 87.20±2.20

## best_ce  (GS: 43 runs | GA: 43 runs)
    Grid Search : 279±142 neurons | 32±0 bits
    GA Neurons  : 260±159 neurons | 32±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |87.27±2.06 87.36±1.90 |13.42±3.16 13.27±3.05 |87.39±2.06 87.49±1.91
    fixed_05             |84.55±3.44 84.18±3.69 |20.09±16.07 23.41±15.76 |85.15±2.83 84.90±3.05
    platt                |82.84±4.25 84.13±4.92 |23.79±16.68 21.53±15.07 |83.66±3.71 84.81±4.35
    beta                 |82.14±5.37 82.21±5.51 |23.75±16.42 26.59±15.77 |83.02±4.81 83.19±4.76
    empirical            |78.31±11.41 76.89±10.00 |38.34±22.58 42.61±19.60 |80.77±8.10 79.52±7.01
    empirical_cumulative |83.71±4.71 80.69±5.95 |24.17±15.65 29.36±17.60 |84.50±3.98 81.94±5.11
    val_cal              |86.38±3.69 85.75±3.82 | 4.77±3.16  5.02±3.30 |86.42±3.61 85.78±3.75

