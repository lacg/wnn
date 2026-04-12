# PUB50 8b CIC-IoT Random — Results (Generated 11/04/2026 21:28)

    Completed : 102/112  |  Remaining: 10
    Avg/run   : 76.9 min
    Latest    : 12/04/2026 01:12 UTC
    ETA       : 12/04/2026 14:01 UTC

## best_fitness  (GS: 101 runs | GA: 102 runs)
    Grid Search : 404±94 neurons | 34±1 bits
    GA Neurons  : 325±125 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.54±0.45 80.07±0.47 | 4.51±1.73  4.26±0.48 |86.41±0.51 86.84±0.42
    fixed_05             |81.08±0.16 81.62±0.41 | 9.06±0.95  9.01±1.04 |88.13±0.20 88.56±0.36
    platt                |80.98±0.57 81.39±0.54 |33.75±3.13 33.13±3.23 |90.51±0.14 90.73±0.17
    beta                 |80.79±0.66 81.14±0.63 |34.95±3.67 34.68±3.50 |90.53±0.13 90.73±0.15
    empirical            |68.36±13.26 68.42±14.57 |64.92±21.98 63.18±25.22 |88.70±2.17 88.76±2.42
    empirical_cumulative |81.73±0.73 82.26±0.68 |22.90±6.61 21.08±6.65 |89.96±0.35 90.16±0.38
    val_cal              |79.31±0.40 79.92±0.46 | 4.11±1.66  3.94±0.54 |86.17±0.47 86.69±0.42

## best_f1  (GS: 101 runs | GA: 102 runs)
    Grid Search : 404±94 neurons | 34±1 bits
    GA Neurons  : 325±125 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.55±0.44 80.07±0.47 | 4.52±1.73  4.26±0.48 |86.42±0.51 86.84±0.42
    fixed_05             |81.08±0.16 81.62±0.41 | 9.04±0.95  9.01±1.04 |88.13±0.20 88.56±0.36
    platt                |80.98±0.57 81.39±0.54 |33.75±3.13 33.13±3.23 |90.51±0.14 90.73±0.17
    beta                 |80.79±0.65 81.14±0.63 |34.98±3.65 34.68±3.50 |90.52±0.13 90.73±0.15
    empirical            |68.35±13.26 68.42±14.57 |64.92±21.99 63.18±25.22 |88.70±2.17 88.76±2.42
    empirical_cumulative |81.73±0.73 82.26±0.68 |22.92±6.60 21.08±6.65 |89.96±0.35 90.16±0.38
    val_cal              |79.31±0.41 79.92±0.46 | 4.10±1.67  3.94±0.54 |86.17±0.47 86.69±0.42

## best_fpr  (GS: 101 runs | GA: 102 runs)
    Grid Search : 149±168 neurons | 14±10 bits
    GA Neurons  : 280±179 neurons | 24±11 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |74.48±1.87 77.58±2.36 | 3.25±1.14  3.77±0.79 |81.55±2.01 84.52±2.26
    fixed_05             |74.42±6.52 78.92±3.88 |28.80±21.42 14.15±12.50 |85.65±1.62 87.03±1.58
    platt                |71.44±7.62 77.32±6.50 |47.57±21.31 39.39±15.25 |86.42±1.97 89.02±2.08
    beta                 |67.09±9.74 75.76±8.53 |61.64±22.86 44.26±19.51 |86.68±1.35 89.01±2.15
    empirical            |61.41±12.82 65.90±13.25 |72.59±25.15 67.80±23.09 |86.41±1.41 87.87±2.13
    empirical_cumulative |73.60±8.48 78.76±5.55 |26.87±24.11 24.61±15.74 |84.69±6.18 88.14±2.56
    val_cal              |74.50±1.57 77.54±2.22 | 3.48±2.32  3.62±0.68 |81.61±1.48 84.48±2.12

## best_acc  (GS: 101 runs | GA: 102 runs)
    Grid Search : 398±96 neurons | 34±1 bits
    GA Neurons  : 345±122 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.54±0.44 79.93±0.52 | 4.53±1.73  4.23±0.45 |86.41±0.51 86.72±0.46
    fixed_05             |81.07±0.16 81.52±0.47 | 9.03±0.93  9.07±1.23 |88.12±0.19 88.48±0.42
    platt                |80.98±0.56 81.31±0.50 |33.72±3.18 33.31±2.77 |90.51±0.14 90.69±0.19
    beta                 |80.79±0.65 81.08±0.53 |34.92±3.62 34.68±2.97 |90.52±0.14 90.68±0.18
    empirical            |68.98±13.18 67.82±14.55 |63.66±21.93 64.49±25.03 |88.80±2.15 88.66±2.41
    empirical_cumulative |81.74±0.73 82.15±0.70 |22.94±6.50 21.35±6.55 |89.97±0.34 90.11±0.38
    val_cal              |79.32±0.40 79.76±0.53 | 4.10±1.67  3.90±0.53 |86.18±0.47 86.54±0.48

## best_ce  (GS: 102 runs | GA: 102 runs)
    Grid Search : 380±115 neurons | 33±1 bits
    GA Neurons  : 326±131 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.44±0.44 79.96±0.59 | 4.49±1.72  4.19±0.52 |86.32±0.50 86.75±0.53
    fixed_05             |81.05±0.22 81.61±0.52 | 9.28±1.07  9.16±1.57 |88.12±0.25 88.56±0.49
    platt                |80.82±0.55 81.33±0.80 |34.42±3.19 33.07±3.88 |90.48±0.15 90.68±0.20
    beta                 |80.68±0.68 81.06±0.84 |35.40±3.61 34.79±4.21 |90.50±0.15 90.69±0.21
    empirical            |68.16±13.84 67.77±14.89 |64.56±23.26 64.24±25.54 |88.66±2.26 88.65±2.46
    empirical_cumulative |81.67±0.79 82.17±0.75 |22.72±7.53 21.50±7.04 |89.91±0.37 90.14±0.41
    val_cal              |79.20±0.45 79.76±0.60 | 4.05±1.68  3.84±0.50 |86.07±0.52 86.54±0.54


## CIC-IoT-2023 Baseline Comparison (F1-macro vs F1-weighted)

    All models use binary classification (normal vs attack).
    F1-macro treats both classes equally; F1-weighted favours the 97% attack majority.

### Full 46M dataset

    Source               | F1-macro | F1-weighted |    FPR |    Acc | Data / Features
    ---------------------|----------|-------------|--------|--------|----------------
    Neto RF              |  96.53%† |      —      |     —  | 99.68% | 46M, all, StandardScaler
    Our RF (all 39)      |  92.64%  |   99.18%    | 13.48% | 99.18% | 46M, all, no scaler
    Our XGBoost (all 39) |  91.67%  |   99.07%    | 14.80% | 99.06% | 46M, all, no scaler
    Our RF (top-20)      |  92.45%  |   99.16%    | 13.73% | 99.15% | 46M, top-20, no scaler
    WNN 245n×32b         |  84.79%  |   97.99%    |  1.59% | 97.68% | 46M, top-20, 8b thermo
    WNN 400n×8b          |  83.13%  |   97.72%    |  2.06% | 97.33% | 46M, top-20, 8b thermo
    WNN 96n×32b          |  84.36%  |   97.93%    |  1.61% | 97.59% | 46M, top-20, 8b thermo

    † Neto et al. F1 averaging method unspecified; StandardScaler + different 80/20 split.

### 1.3M subsample (random 80/20)

    Source               | F1-macro | F1-weighted |    FPR |    Acc | Data / Features
    ---------------------|----------|-------------|--------|--------|----------------
    Our RF (all 39)      |  86.08%  |   92.95%    | 24.00% | 92.96% | 1.3M, all, no scaler
    Our RF (top-20)      |  85.53%  |   92.68%    | 25.18% | 92.71% | 1.3M, top-20, no scaler
    Our XGBoost (top-20) |  84.13%  |   92.07%    | 28.34% | 92.07% | 1.3M, top-20, no scaler
    WNN avg (train_cal)  |  80.07%  |      —      |  4.26% | 86.84% | 1.3M, top-20, n=102
    WNN avg (fixed_05)   |  81.62%  |      —      |  9.01% | 88.56% | 1.3M, top-20, n=102


## Observation: 6n × 16b best_fpr Pareto Point (flow r020)

    During the PUB50 8b batch, flow PUB50-ciciot-random-r020 (experiment 4964)
    produced a genuinely interesting best_fpr genome:

    Genome        | Size           | train_cal F1 | train_cal FPR | train_cal Acc
    --------------|----------------|--------------|---------------|---------------
    best_fitness  | 387n × 32b     |    80.21%    |    4.20%      |    86.96%
    best_f1       | 387n × 32b     |    80.21%    |    4.20%      |    86.96%
    best_ce       | 350n × 32b     |    79.89%    |    3.81%      |    86.65%
    best_acc      | 500n × 34b     |    79.10%    |    4.09%      |    85.99%
    best_fpr      | **6n × 16b**   |  **73.70%**  |  **1.09%**    |    80.56%

    Threshold sensitivity on 6n × 16b:

    Threshold mode        |    F1    |   FPR
    ----------------------|----------|--------
    train_cal             |  73.70%  |   1.09%
    val_cal (oracle)      |  74.75%  |   1.73%
    fixed_05              |  77.26%  |  30.76%

    Pending: 46M validation (flow 1231, queued after PUB50 finishes).
