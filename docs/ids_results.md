# PUB50 8b CIC-IoT Random — Results (Generated 11/04/2026 09:47)

    Completed : 93/112  |  Remaining: 19
    Avg/run   : 76.4 min
    Latest    : 11/04/2026 12:57 UTC
    ETA       : 12/04/2026 13:08 UTC

## best_fitness  (GS: 92 runs | GA: 93 runs)
    Grid Search : 404±94 neurons | 34±1 bits
    GA Neurons  : 324±125 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.54±0.43 80.05±0.47 | 4.53±1.80  4.24±0.46 |86.41±0.50 86.83±0.41
    fixed_05             |81.08±0.16 81.61±0.41 | 9.07±0.89  9.05±0.97 |88.13±0.19 88.55±0.36
    platt                |81.00±0.58 81.38±0.53 |33.59±3.14 33.17±3.18 |90.51±0.15 90.72±0.18
    beta                 |80.83±0.63 81.12±0.64 |34.75±3.56 34.77±3.53 |90.53±0.13 90.73±0.15
    empirical            |67.77±13.45 68.36±14.53 |65.98±22.21 63.38±25.10 |88.61±2.20 88.75±2.41
    empirical_cumulative |81.72±0.76 82.23±0.70 |22.97±6.89 21.36±6.82 |89.96±0.35 90.17±0.39
    val_cal              |79.30±0.40 79.90±0.47 | 4.11±1.74  3.93±0.54 |86.16±0.48 86.67±0.43

## best_f1  (GS: 92 runs | GA: 93 runs)
    Grid Search : 404±94 neurons | 34±1 bits
    GA Neurons  : 324±125 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.55±0.43 80.05±0.47 | 4.53±1.80  4.24±0.46 |86.42±0.50 86.83±0.41
    fixed_05             |81.08±0.16 81.61±0.41 | 9.06±0.89  9.05±0.97 |88.13±0.19 88.55±0.36
    platt                |81.00±0.58 81.38±0.53 |33.59±3.14 33.17±3.18 |90.51±0.15 90.72±0.18
    beta                 |80.82±0.62 81.12±0.64 |34.78±3.53 34.77±3.53 |90.52±0.13 90.73±0.15
    empirical            |67.77±13.46 68.36±14.53 |65.98±22.22 63.38±25.10 |88.60±2.20 88.75±2.41
    empirical_cumulative |81.72±0.76 82.23±0.70 |22.99±6.88 21.36±6.82 |89.96±0.35 90.17±0.39
    val_cal              |79.30±0.41 79.90±0.47 | 4.11±1.74  3.93±0.54 |86.16±0.48 86.67±0.43

## best_fpr  (GS: 92 runs | GA: 93 runs)
    Grid Search : 142±164 neurons | 14±10 bits
    GA Neurons  : 289±177 neurons | 25±11 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |74.49±1.93 77.72±2.35 | 3.27±1.17  3.81±0.79 |81.55±2.08 84.66±2.25
    fixed_05             |74.70±6.05 79.29±2.92 |28.13±20.05 13.01±9.72 |85.65±1.64 87.10±1.61
    platt                |71.85±6.95 77.41±6.71 |46.43±20.61 39.49±15.49 |86.39±2.01 89.14±2.05
    beta                 |67.26±9.57 76.34±7.97 |61.14±22.77 42.95±18.51 |86.65±1.35 89.12±2.14
    empirical            |61.61±12.84 66.04±13.27 |72.15±25.37 67.83±22.99 |86.43±1.43 87.97±2.14
    empirical_cumulative |73.43±8.82 78.90±5.65 |26.43±24.73 24.32±15.84 |84.55±6.43 88.22±2.60
    val_cal              |74.50±1.60 77.66±2.22 | 3.52±2.39  3.65±0.66 |81.61±1.49 84.60±2.11

## best_acc  (GS: 92 runs | GA: 93 runs)
    Grid Search : 396±99 neurons | 34±1 bits
    GA Neurons  : 339±123 neurons | 33±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.54±0.43 79.92±0.52 | 4.54±1.81  4.21±0.46 |86.41±0.51 86.71±0.46
    fixed_05             |81.07±0.15 81.51±0.46 | 9.05±0.86  9.10±1.24 |88.12±0.18 88.48±0.41
    platt                |81.00±0.57 81.31±0.50 |33.55±3.18 33.28±2.79 |90.51±0.14 90.69±0.19
    beta                 |80.82±0.62 81.07±0.53 |34.74±3.54 34.71±3.02 |90.52±0.14 90.68±0.18
    empirical            |68.46±13.40 68.47±14.27 |64.57±22.24 63.40±24.66 |88.72±2.19 88.76±2.37
    empirical_cumulative |81.72±0.76 82.13±0.72 |22.99±6.79 21.60±6.74 |89.96±0.35 90.12±0.38
    val_cal              |79.30±0.40 79.75±0.52 | 4.10±1.74  3.89±0.54 |86.16±0.48 86.54±0.47

## best_ce  (GS: 93 runs | GA: 93 runs)
    Grid Search : 379±115 neurons | 33±1 bits
    GA Neurons  : 325±132 neurons | 32±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |79.45±0.43 79.94±0.59 | 4.52±1.79  4.15±0.50 |86.34±0.50 86.72±0.53
    fixed_05             |81.05±0.22 81.60±0.53 | 9.25±1.07  9.21±1.60 |88.12±0.25 88.55±0.51
    platt                |80.82±0.54 81.35±0.82 |34.42±3.17 32.86±3.93 |90.48±0.14 90.68±0.20
    beta                 |80.68±0.68 81.07±0.85 |35.40±3.58 34.64±4.28 |90.50±0.15 90.69±0.22
    empirical            |68.55±13.66 68.21±14.80 |63.92±23.03 63.36±25.53 |88.72±2.22 88.72±2.44
    empirical_cumulative |81.64±0.81 82.14±0.77 |23.04±7.73 21.64±7.34 |89.92±0.38 90.14±0.42
    val_cal              |79.19±0.46 79.75±0.61 | 4.06±1.76  3.83±0.50 |86.06±0.53 86.53±0.55


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
    WNN avg (train_cal)  |  80.05%  |      —      |  4.24% | 86.83% | 1.3M, top-20, n=93
    WNN avg (fixed_05)   |  81.61%  |      —      |  9.05% | 88.55% | 1.3M, top-20, n=93

## Observation: 6n × 16b best_fpr Pareto Point (flow r020)

During the PUB50 8b batch, flow `PUB50-ciciot-random-r020` (experiment 4964)
produced a genuinely interesting best_fpr genome:

    Genome        | Size           | train_cal F1 | train_cal FPR | train_cal Acc
    --------------|----------------|--------------|---------------|---------------
    best_fitness  | 387n × 32b     |    80.21%    |    4.20%      |    86.96%
    best_f1       | 387n × 32b     |    80.21%    |    4.20%      |    86.96%
    best_ce       | 350n × 32b     |    79.89%    |    3.81%      |    86.65%
    best_acc      | 500n × 34b     |    79.10%    |    4.09%      |    85.99%
    best_fpr      | **6n × 16b**   |  **73.70%**  |  **1.09%**    |    80.56%

Four of the five genome types converged on the same architectural region
(350-500 neurons × 32-34 bits). The best_fpr genome is the outlier:
**65× fewer neurons and half the address width**, yet achieves **1.09% FPR**
— the lowest of any genome in the batch at any threshold.

### Why this works

With only 6 neurons voting, the score distribution is coarse (0/6, 1/6, ..., 6/6).
A well-calibrated threshold (train_cal) catches clean attack cases while
rejecting almost all normal traffic. The penalty: missed borderline attacks,
lower F1 (73.70%). But the FPR is dramatically lower because a false alarm
requires almost unanimous (incorrect) agreement across all 6 neurons.

Threshold sensitivity is extreme for small-neuron-count genomes:

    Threshold mode        |    F1    |   FPR
    ----------------------|----------|--------
    train_cal             |  73.70%  |   1.09%  ← calibrated sweet spot
    val_cal (oracle)      |  74.75%  |   1.73%
    fixed_05              |  77.26%  |  30.76%  ← FPR disaster at 0.5
    empirical             |  77.08%  |  36.81%
    platt                 |  72.30%  |  51.64%
    beta                  |  75.05%  |  48.94%
    empirical_cumulative  |  76.88%  |  18.73%

The low FPR only emerges under calibrated thresholds. This is a strong
argument for the threshold-aware fitness function.

### Deployment implication

A **first-stage filter** use case:
- Stage 1: 6n × 16b WNN filters obvious attacks at very low FPR (~1%)
- Stage 2: larger WNN or ML model handles the uncertain cases

At ~600-2000 bytes of sparse memory footprint, this could fit as a
pre-filter in front of the 25 KB 400n × 8b peak genome, reducing load
on the larger classifier.

### Validation pending

This is a **single-seed 1.3M result**. Pareto point needs 46M validation
before it's paper-ready. Added to the roadmap as a follow-up 46M run
after PUB50 8b completes.
