# XDS-unsw-random — width × weight cohort breakdown (60 non-OLD completed)

    Total non-OLD completed : 60  |  Total wall: 109.8h  |  Avg/run: 110m
    Latest done : 06/06/2026 00:55 UTC

    Weight schemes:
      Wa (CIC-IoT legacy, ce=0.35 acc=0.30)
      Wb (paper/PUB50, ce=0.10 acc=0.20)
      Wc (CE-heavy NEW, ce=0.70 acc=0.10)


## XDS-unsw-random-8b-Wb  (2 flows × 2 phases, seeds: [8188, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 317±176 neurons | 16±0 bits
    GA Neurons  : 321±229 neurons | 18±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.40±0.07 93.44±0.02 | 1.14±0.01  1.13±0.00 |98.90±0.01 98.91±0.00
    platt                |93.34±0.03 93.34±0.03 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.00
    beta                 |89.18±7.26 93.36±0.09 | 0.88±0.40  1.12±0.00 |98.51±0.68 98.90±0.01
    empirical            |93.13±0.15 93.11±0.10 | 1.11±0.00  1.11±0.00 |98.87±0.02 98.86±0.01
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.52±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 317±176 neurons | 16±0 bits
    GA Neurons  : 321±229 neurons | 18±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.40±0.07 93.44±0.02 | 1.14±0.01  1.13±0.00 |98.90±0.01 98.91±0.00
    platt                |93.34±0.03 93.34±0.03 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.00
    beta                 |89.18±7.26 93.36±0.09 | 0.88±0.40  1.12±0.00 |98.51±0.68 98.90±0.01
    empirical            |93.13±0.15 93.11±0.10 | 1.11±0.00  1.11±0.00 |98.87±0.02 98.86±0.01
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.52±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 102±138 neurons | 4±0 bits
    GA Neurons  : 79±113 neurons | 4±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |73.80±21.67 80.04±4.63 | 0.71±1.03  1.59±1.40 |97.43±1.07 97.13±0.49
    fixed_05             |66.51±19.56 59.58±16.05 | 5.91±8.22 22.41±16.77 |93.04±6.76 78.27±16.15
    platt                |73.42±21.27 78.66±5.21 | 0.38±0.48  0.73±0.61 |97.48±1.13 97.40±0.51
    beta                 |73.36±21.20 79.30±4.03 | 0.37±0.46  0.36±0.54 |97.48±1.12 97.66±0.08
    empirical            |73.75±21.61 79.55±4.48 | 0.74±1.09  1.03±1.70 |97.41±1.06 97.41±0.36
    empirical_cumulative |73.74±21.60 79.19±3.85 | 0.42±0.54  0.06±0.03 |97.51±1.15 97.78±0.29
    val_cal              |73.80±21.67 80.04±4.63 | 0.71±1.03  1.59±1.40 |97.43±1.07 97.13±0.49

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 317±176 neurons | 16±0 bits
    GA Neurons  : 321±229 neurons | 18±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |93.40±0.07 93.44±0.02 | 1.14±0.01  1.13±0.00 |98.90±0.01 98.91±0.00
    platt                |93.34±0.03 93.34±0.03 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.90±0.00
    beta                 |89.18±7.26 93.36±0.09 | 0.88±0.40  1.12±0.00 |98.51±0.68 98.90±0.01
    empirical            |93.13±0.15 93.11±0.10 | 1.11±0.00  1.11±0.00 |98.87±0.02 98.86±0.01
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.52±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 333±153 neurons | 13±2 bits
    GA Neurons  : 389±123 neurons | 17±3 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |91.67±2.05 93.33±0.25 | 1.51±0.44  1.16±0.05 |98.55±0.43 98.89±0.05
    platt                |93.33±0.01 93.30±0.04 | 1.12±0.00  1.11±0.00 |98.89±0.00 98.89±0.01
    beta                 |92.93±0.51 93.27±0.03 | 1.11±0.00  1.11±0.00 |98.84±0.07 98.89±0.00
    empirical            |93.05±0.10 93.01±0.03 | 1.11±0.00  1.11±0.00 |98.85±0.01 98.85±0.00
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## XDS-unsw-random-8b-Wc  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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


## XDS-unsw-random-16b-Wb  (7 flows × 2 phases, seeds: [8188, 8627, 25608, 60123, 67673, 82096, 92774])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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
    Grid Search : 226±148 neurons | 14±2 bits
    GA Neurons  : 97±134 neurons | 18±10 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.48±0.09 93.60±0.10 | 1.07±0.14  1.04±0.09 |98.93±0.02 98.95±0.03
    fixed_05             |91.69±1.63 92.90±0.55 | 1.50±0.35  1.24±0.11 |98.55±0.34 98.80±0.11
    platt                |93.32±0.05 92.89±1.02 | 1.07±0.14  0.99±0.15 |98.90±0.02 98.86±0.14
    beta                 |89.64±6.85 89.71±5.53 | 0.89±0.32  0.79±0.32 |98.56±0.64 98.56±0.51
    empirical            |93.11±0.08 93.17±0.65 | 1.06±0.14  1.00±0.14 |98.87±0.03 98.90±0.10
    empirical_cumulative |93.48±0.09 93.56±0.12 | 1.07±0.14  1.02±0.15 |98.93±0.02 98.95±0.03
    val_cal              |93.48±0.09 93.60±0.10 | 1.07±0.14  1.04±0.09 |98.93±0.02 98.95±0.03

### best_f1  (GS: 7 runs | GA: 7 runs)
    Grid Search : 262±119 neurons | 14±2 bits
    GA Neurons  : 97±134 neurons | 18±10 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.60±0.10 | 1.12±0.00  1.04±0.09 |98.92±0.00 98.95±0.03
    fixed_05             |91.66±1.67 92.90±0.55 | 1.51±0.37  1.24±0.11 |98.55±0.35 98.80±0.11
    platt                |93.34±0.01 92.89±1.02 | 1.12±0.00  0.99±0.15 |98.90±0.00 98.86±0.14
    beta                 |89.67±6.86 89.71±5.53 | 0.94±0.32  0.79±0.32 |98.56±0.63 98.56±0.51
    empirical            |93.11±0.08 93.17±0.65 | 1.11±0.00  1.00±0.14 |98.86±0.01 98.90±0.10
    empirical_cumulative |93.51±0.00 93.56±0.12 | 1.12±0.00  1.02±0.15 |98.92±0.00 98.95±0.03
    val_cal              |93.51±0.00 93.60±0.10 | 1.12±0.00  1.04±0.09 |98.92±0.00 98.95±0.03

### best_fpr  (GS: 7 runs | GA: 7 runs)
    Grid Search : 5±0 neurons | 18±10 bits
    GA Neurons  : 14±5 neurons | 16±6 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |90.17±5.49 85.73±10.41 | 1.00±0.53  3.67±5.38 |98.56±0.52 96.29±5.22
    fixed_05             |85.70±11.22 82.08±10.91 | 3.97±4.89  5.34±6.55 |96.17±4.72 94.83±6.31
    platt                |89.98±5.42 82.78±13.76 | 0.86±0.34  1.26±0.58 |98.58±0.48 97.78±0.71
    beta                 |89.26±6.86 84.85±10.10 | 0.83±0.38  2.69±5.77 |98.53±0.54 96.51±5.31
    empirical            |90.08±5.48 85.25±11.25 | 0.86±0.34  4.05±6.94 |98.59±0.49 95.84±6.58
    empirical_cumulative |90.12±5.50 83.82±14.12 | 0.86±0.34  0.78±0.55 |98.60±0.49 98.08±0.78
    val_cal              |90.17±5.49 85.73±10.42 | 1.00±0.53  3.68±5.38 |98.56±0.52 96.29±5.22

### best_acc  (GS: 7 runs | GA: 7 runs)
    Grid Search : 232±148 neurons | 14±2 bits
    GA Neurons  : 97±134 neurons | 18±10 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.48±0.09 93.59±0.13 | 1.07±0.14  1.05±0.13 |98.93±0.02 98.95±0.04
    fixed_05             |91.81±1.69 92.42±1.23 | 1.48±0.37  1.34±0.26 |98.58±0.36 98.70±0.25
    platt                |93.32±0.04 92.91±1.04 | 1.07±0.14  0.97±0.17 |98.90±0.02 98.87±0.14
    beta                 |91.24±5.83 90.24±5.66 | 0.98±0.26  0.81±0.30 |98.71±0.54 98.62±0.53
    empirical            |93.10±0.09 93.18±0.66 | 1.06±0.14  0.98±0.16 |98.87±0.04 98.90±0.11
    empirical_cumulative |93.48±0.09 93.58±0.14 | 1.07±0.14  1.00±0.17 |98.93±0.02 98.96±0.05
    val_cal              |93.48±0.09 93.59±0.13 | 1.07±0.14  1.05±0.13 |98.93±0.02 98.95±0.04

### best_ce  (GS: 7 runs | GA: 7 runs)
    Grid Search : 450±141 neurons | 12±1 bits
    GA Neurons  : 358±138 neurons | 19±8 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |91.96±1.28 93.08±0.33 | 1.44±0.28  1.21±0.07 |98.61±0.27 98.84±0.06
    platt                |93.32±0.01 93.19±0.16 | 1.12±0.00  1.11±0.01 |98.89±0.00 98.88±0.02
    beta                 |93.10±0.15 93.22±0.12 | 1.11±0.01  1.11±0.00 |98.86±0.02 98.88±0.02
    empirical            |93.08±0.10 92.88±0.43 | 1.11±0.00  1.10±0.02 |98.86±0.01 98.83±0.06
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## XDS-unsw-random-16b-Wc  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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


## XDS-unsw-random-32b-Wa  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

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


## XDS-unsw-random-32b-Wb  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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
    Best Acc (any FPR)       |  93.49% |   0.99% |  98.94% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 312±217 neurons | 15±2 bits
    GA Neurons  : 96±2 neurons | 15±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.86±0.27 92.91±0.11 | 1.25±0.05  1.24±0.02 |98.80±0.05 98.81±0.02
    platt                |93.34±0.01 93.29±0.09 | 1.12±0.00  1.11±0.02 |98.90±0.00 98.89±0.01
    beta                 |93.27±0.12 91.60±3.35 | 1.11±0.00  0.98±0.24 |98.89±0.02 98.71±0.35
    empirical            |93.11±0.10 93.17±0.16 | 1.11±0.00  1.11±0.01 |98.86±0.01 98.87±0.02
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 312±217 neurons | 15±2 bits
    GA Neurons  : 96±2 neurons | 15±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.86±0.27 92.91±0.11 | 1.25±0.05  1.24±0.02 |98.80±0.05 98.81±0.02
    platt                |93.34±0.01 93.29±0.09 | 1.12±0.00  1.11±0.02 |98.90±0.00 98.89±0.01
    beta                 |93.27±0.12 91.60±3.35 | 1.11±0.00  0.98±0.24 |98.89±0.02 98.71±0.35
    empirical            |93.11±0.10 93.17±0.16 | 1.11±0.00  1.11±0.01 |98.86±0.01 98.87±0.02
    empirical_cumulative |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.51±0.00 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 5±0 neurons | 28±25 bits
    GA Neurons  : 14±4 neurons | 17±5 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.00±0.56 90.04±3.68 | 1.11±0.03  1.23±0.43 |98.85±0.08 98.43±0.48
    fixed_05             |91.94±2.38 84.58±9.06 | 1.45±0.54  3.64±2.68 |98.60±0.51 96.47±2.60
    platt                |92.51±0.47 88.80±4.60 | 1.07±0.06  1.30±0.25 |98.79±0.06 98.23±0.69
    beta                 |91.21±1.81 89.58±3.69 | 1.29±0.55  0.88±0.17 |98.54±0.40 98.48±0.41
    empirical            |92.96±0.55 89.94±3.62 | 1.10±0.03  1.16±0.46 |98.84±0.08 98.43±0.49
    empirical_cumulative |93.00±0.56 89.70±3.77 | 1.11±0.03  0.87±0.16 |98.85±0.08 98.49±0.43
    val_cal              |93.00±0.56 90.04±3.68 | 1.11±0.03  1.23±0.43 |98.85±0.08 98.43±0.48

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 312±217 neurons | 15±2 bits
    GA Neurons  : 67±51 neurons | 16±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.51±0.00 93.50±0.01 | 1.12±0.00  1.09±0.06 |98.92±0.00 98.93±0.01
    fixed_05             |92.86±0.27 91.89±1.99 | 1.25±0.05  1.47±0.44 |98.80±0.05 98.59±0.42
    platt                |93.34±0.01 92.77±1.03 | 1.12±0.00  1.05±0.11 |98.90±0.00 98.83±0.12
    beta                 |93.27±0.12 89.88±3.92 | 1.11±0.00  0.90±0.24 |98.89±0.02 98.52±0.43
    empirical            |93.11±0.10 93.25±0.22 | 1.11±0.00  1.08±0.06 |98.86±0.01 98.89±0.04
    empirical_cumulative |93.51±0.00 93.50±0.01 | 1.12±0.00  1.09±0.06 |98.92±0.00 98.93±0.01
    val_cal              |93.51±0.00 93.51±0.01 | 1.12±0.00  1.09±0.06 |98.92±0.00 98.93±0.01

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 350±132 neurons | 15±2 bits
    GA Neurons  : 313±175 neurons | 17±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    fixed_05             |92.66±0.37 93.09±0.27 | 1.29±0.08  1.20±0.05 |98.76±0.07 98.84±0.05
    platt                |93.35±0.01 93.31±0.04 | 1.12±0.00  1.12±0.00 |98.90±0.00 98.89±0.01
    beta                 |93.10±0.45 93.29±0.03 | 1.11±0.00  1.12±0.00 |98.86±0.06 98.89±0.00
    empirical            |93.06±0.09 92.84±0.35 | 1.11±0.00  1.10±0.02 |98.86±0.01 98.83±0.05
    empirical_cumulative |93.50±0.01 93.51±0.01 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00
    val_cal              |93.50±0.01 93.51±0.00 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.92±0.00


## XDS-unsw-random-32b-Wc  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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


## XDS-unsw-random-64b-Wa  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

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


## XDS-unsw-random-64b-Wb  (30 flows × 2 phases, seeds: [2647, 6858, 8161, 8188, 8627, 14613, 17375, 17821, 21395, 21777, 25608, 25987, 26607, 30971, 35432, 39086, 43427, 44520, 48846, 50011, 60123, 67673, 67784, 69436, 78572, 82096, 92726, 92774, 96530, 96660])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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

### best_fitness  (GS: 30 runs | GA: 30 runs)
    Grid Search : 273±157 neurons | 14±2 bits
    GA Neurons  : 132±122 neurons | 15±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.56±0.12 | 1.11±0.05  1.07±0.11 |98.92±0.01 98.94±0.04
    fixed_05             |92.21±1.15 92.13±1.44 | 1.39±0.25  1.41±0.32 |98.66±0.24 98.64±0.30
    platt                |93.33±0.04 93.10±0.64 | 1.11±0.05  1.04±0.14 |98.90±0.01 98.88±0.09
    beta                 |91.28±4.29 92.14±2.31 | 1.00±0.22  0.99±0.20 |98.69±0.42 98.77±0.26
    empirical            |93.15±0.13 93.26±0.28 | 1.10±0.05  1.05±0.14 |98.87±0.03 98.90±0.06
    empirical_cumulative |93.50±0.00 93.56±0.12 | 1.11±0.05  1.07±0.11 |98.92±0.01 98.94±0.04
    val_cal              |93.51±0.00 93.56±0.12 | 1.11±0.05  1.07±0.11 |98.92±0.01 98.94±0.04

### best_f1  (GS: 30 runs | GA: 30 runs)
    Grid Search : 289±154 neurons | 14±2 bits
    GA Neurons  : 132±122 neurons | 15±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.56±0.12 | 1.12±0.00  1.07±0.11 |98.92±0.00 98.94±0.04
    fixed_05             |92.34±0.94 92.13±1.44 | 1.36±0.20  1.41±0.32 |98.69±0.20 98.64±0.30
    platt                |93.33±0.03 93.10±0.64 | 1.12±0.00  1.04±0.14 |98.89±0.00 98.88±0.09
    beta                 |91.27±4.29 92.14±2.31 | 1.01±0.22  0.99±0.20 |98.68±0.42 98.77±0.26
    empirical            |93.14±0.11 93.26±0.28 | 1.11±0.00  1.05±0.14 |98.87±0.02 98.90±0.06
    empirical_cumulative |93.50±0.00 93.56±0.12 | 1.12±0.00  1.07±0.11 |98.92±0.00 98.94±0.04
    val_cal              |93.50±0.00 93.56±0.12 | 1.12±0.00  1.07±0.11 |98.92±0.00 98.94±0.04

### best_fpr  (GS: 30 runs | GA: 30 runs)
    Grid Search : 14±29 neurons | 18±11 bits
    GA Neurons  : 23±29 neurons | 15±9 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |91.24±3.28 86.91±7.49 | 1.25±0.67  2.16±2.40 |98.55±0.66 97.49±2.52
    fixed_05             |89.20±6.04 79.72±13.06 | 2.23±2.29  6.74±7.58 |97.80±2.31 93.42±7.46
    platt                |90.87±3.60 84.79±10.32 | 0.97±0.29  1.16±0.42 |98.62±0.38 97.94±0.68
    beta                 |90.77±3.53 85.78±7.28 | 0.91±0.28  1.28±2.58 |98.63±0.33 97.67±2.54
    empirical            |91.14±3.56 86.14±9.16 | 1.08±0.49  2.76±5.79 |98.62±0.41 96.82±5.62
    empirical_cumulative |91.08±3.64 85.57±10.31 | 0.92±0.29  0.73±0.61 |98.67±0.35 98.19±0.67
    val_cal              |91.24±3.28 86.91±7.49 | 1.25±0.67  2.15±2.40 |98.55±0.66 97.50±2.52

### best_acc  (GS: 30 runs | GA: 30 runs)
    Grid Search : 237±149 neurons | 14±2 bits
    GA Neurons  : 109±119 neurons | 15±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.00 93.54±0.14 | 1.11±0.05  1.04±0.14 |98.92±0.01 98.94±0.04
    fixed_05             |92.30±1.17 91.83±1.89 | 1.37±0.26  1.48±0.43 |98.68±0.25 98.58±0.42
    platt                |93.33±0.04 93.09±0.59 | 1.11±0.05  0.98±0.19 |98.90±0.01 98.89±0.09
    beta                 |91.10±4.32 92.03±2.33 | 0.99±0.22  0.93±0.23 |98.67±0.42 98.77±0.26
    empirical            |93.15±0.13 93.28±0.30 | 1.10±0.05  1.02±0.15 |98.87±0.03 98.91±0.07
    empirical_cumulative |93.50±0.01 93.54±0.16 | 1.11±0.05  1.02±0.15 |98.92±0.01 98.95±0.04
    val_cal              |93.50±0.00 93.55±0.14 | 1.11±0.05  1.04±0.14 |98.92±0.01 98.94±0.04

### best_ce  (GS: 30 runs | GA: 30 runs)
    Grid Search : 377±131 neurons | 13±1 bits
    GA Neurons  : 321±148 neurons | 17±4 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.49±0.01 93.47±0.11 | 1.12±0.00  1.12±0.01 |98.92±0.00 98.91±0.02
    fixed_05             |92.42±0.25 92.89±0.23 | 1.34±0.05  1.24±0.05 |98.71±0.05 98.80±0.04
    platt                |93.33±0.02 93.25±0.14 | 1.12±0.00  1.11±0.01 |98.89±0.00 98.88±0.02
    beta                 |93.19±0.17 93.20±0.15 | 1.11±0.01  1.11±0.01 |98.87±0.02 98.88±0.02
    empirical            |93.11±0.10 93.06±0.20 | 1.11±0.00  1.10±0.04 |98.86±0.01 98.86±0.03
    empirical_cumulative |93.49±0.02 93.46±0.12 | 1.12±0.00  1.12±0.01 |98.92±0.00 98.91±0.02
    val_cal              |93.49±0.01 93.47±0.10 | 1.12±0.00  1.12±0.01 |98.92±0.00 98.91±0.02


## XDS-unsw-random-64b-Wc  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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


## XDS-unsw-random-96b-Wa  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

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


## XDS-unsw-random-96b-Wb  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  84.14% |   0.39% |  98.03% | r82096 GS best_ce        beta
    Best FPR (F1>80%)        |  84.14% |   0.39% |  98.03% | r82096 GS best_ce        beta
    Best Acc (any FPR)       |  93.78% |   0.84% |  99.02% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 3 runs | GA: 3 runs)
    Grid Search : 267±115 neurons | 15±2 bits
    GA Neurons  : 271±207 neurons | 11±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.57±0.13 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05
    fixed_05             |91.82±1.87 90.95±1.86 | 1.48±0.41  1.66±0.41 |98.58±0.39 98.40±0.39
    platt                |93.33±0.01 93.16±0.39 | 1.12±0.00  0.95±0.33 |98.89±0.00 98.91±0.02
    beta                 |92.07±2.43 92.94±0.49 | 1.05±0.14  0.94±0.35 |98.75±0.29 98.88±0.03
    empirical            |93.19±0.12 93.34±0.28 | 1.11±0.01  1.04±0.14 |98.87±0.02 98.91±0.07
    empirical_cumulative |93.50±0.01 93.57±0.13 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05
    val_cal              |93.51±0.00 93.58±0.13 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05

### best_f1  (GS: 3 runs | GA: 3 runs)
    Grid Search : 267±115 neurons | 15±2 bits
    GA Neurons  : 271±207 neurons | 11±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.57±0.13 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05
    fixed_05             |91.82±1.87 90.95±1.86 | 1.48±0.41  1.66±0.41 |98.58±0.39 98.40±0.39
    platt                |93.33±0.01 93.16±0.39 | 1.12±0.00  0.95±0.33 |98.89±0.00 98.91±0.02
    beta                 |92.07±2.43 92.94±0.49 | 1.05±0.14  0.94±0.35 |98.75±0.29 98.88±0.03
    empirical            |93.19±0.12 93.34±0.28 | 1.11±0.01  1.04±0.14 |98.87±0.02 98.91±0.07
    empirical_cumulative |93.50±0.01 93.57±0.13 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05
    val_cal              |93.51±0.00 93.58±0.13 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05

### best_fpr  (GS: 3 runs | GA: 3 runs)
    Grid Search : 37±55 neurons | 32±16 bits
    GA Neurons  : 55±55 neurons | 19±12 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |92.16±2.06 91.60±2.39 | 1.08±0.07  1.26±0.48 |98.74±0.27 98.60±0.46
    fixed_05             |91.57±2.41 87.72±4.58 | 1.53±0.55  2.49±1.18 |98.52±0.52 97.60±1.14
    platt                |91.83±1.89 90.67±3.38 | 0.99±0.13  1.08±0.15 |98.71±0.27 98.54±0.46
    beta                 |91.61±1.69 90.16±2.54 | 0.97±0.11  0.85±0.07 |98.69±0.24 98.54±0.32
    empirical            |91.74±1.88 91.24±2.42 | 1.28±0.54  1.31±0.74 |98.61±0.42 98.52±0.57
    empirical_cumulative |92.08±2.01 91.49±2.46 | 1.00±0.13  0.89±0.30 |98.75±0.27 98.71±0.24
    val_cal              |92.16±2.06 91.60±2.39 | 1.08±0.07  1.26±0.48 |98.74±0.27 98.60±0.46

### best_acc  (GS: 3 runs | GA: 3 runs)
    Grid Search : 217±176 neurons | 15±2 bits
    GA Neurons  : 270±208 neurons | 11±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.50±0.01 93.57±0.14 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05
    fixed_05             |91.79±1.85 90.91±1.84 | 1.48±0.41  1.67±0.40 |98.57±0.39 98.39±0.39
    platt                |93.33±0.01 93.25±0.21 | 1.12±0.00  0.97±0.29 |98.89±0.00 98.91±0.03
    beta                 |92.05±2.42 93.01±0.32 | 1.05±0.14  0.95±0.33 |98.74±0.28 98.89±0.04
    empirical            |93.17±0.13 93.31±0.29 | 1.11±0.01  1.04±0.14 |98.87±0.02 98.91±0.07
    empirical_cumulative |93.50±0.01 93.57±0.14 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05
    val_cal              |93.50±0.01 93.57±0.14 | 1.12±0.00  1.05±0.14 |98.92±0.00 98.94±0.05

### best_ce  (GS: 3 runs | GA: 3 runs)
    Grid Search : 317±161 neurons | 35±39 bits
    GA Neurons  : 162±73 neurons | 25±16 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |93.47±0.01 93.41±0.10 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.90±0.01
    fixed_05             |92.66±0.48 93.05±0.28 | 1.28±0.12  1.21±0.06 |98.76±0.10 98.83±0.05
    platt                |93.15±0.39 93.05±0.30 | 1.11±0.02  1.10±0.02 |98.87±0.05 98.86±0.04
    beta                 |90.99±4.57 93.18±0.14 | 0.94±0.36  1.11±0.01 |98.67±0.43 98.87±0.02
    empirical            |92.53±1.06 92.26±0.87 | 1.05±0.12  0.92±0.21 |98.80±0.12 98.79±0.12
    empirical_cumulative |93.47±0.01 93.41±0.10 | 1.12±0.00  1.12±0.00 |98.91±0.00 98.90±0.01
    val_cal              |93.48±0.02 93.41±0.10 | 1.12±0.00  1.12±0.00 |98.92±0.00 98.90±0.02


## XDS-unsw-random-96b-Wc  (3 flows × 2 phases, seeds: [8188, 25608, 82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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

