# XDS-cicids — width × weight cohort breakdown (26 non-OLD completed)

    Total non-OLD completed : 26  |  Total wall: 122.3h  |  Avg/run: 282m
    Latest done : 11/06/2026 00:55 UTC

    Weight schemes:
      Wa (CIC-IoT legacy, ce=0.35 acc=0.30)
      Wb (paper/PUB50, ce=0.10 acc=0.20)
      Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)
      Wc (CE-heavy NEW, ce=0.70 acc=0.10)


## XDS-cicids-8b-Wa  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

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


## XDS-cicids-8b-Wb  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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


## XDS-cicids-8b-Wbu  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)

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


## XDS-cicids-8b-Wc  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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


## XDS-cicids-16b-Wa  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.56% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.56% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.56% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.56% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.56% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.56% |   0.12% |  99.72% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  99.41% |   0.09% |  99.63% | r25608 GA best_fpr       train_cal
    Best FPR (F1>80%)        |  99.41% |   0.09% |  99.63% | r25608 GA best_fpr       train_cal
    Best Acc (any FPR)       |  99.56% |   0.12% |  99.72% | r82096 GA best_f1        val_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 262±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.56±0.00 | 0.18±0.01  0.11±0.00 |99.61±0.01 99.72±0.00
    fixed_05             |99.04±0.03 99.19±0.09 | 0.59±0.03  0.47±0.07 |99.39±0.02 99.49±0.06
    platt                |99.34±0.03 99.43±0.04 | 0.30±0.02  0.24±0.04 |99.58±0.02 99.64±0.02
    beta                 |99.37±0.03 99.49±0.09 | 0.24±0.01  0.18±0.10 |99.60±0.02 99.68±0.06
    empirical            |98.95±0.63 99.49±0.09 | 0.64±0.58  0.10±0.01 |99.33±0.41 99.68±0.06
    empirical_cumulative |99.38±0.01 99.56±0.00 | 0.18±0.01  0.11±0.00 |99.61±0.01 99.72±0.00
    val_cal              |99.38±0.01 99.56±0.00 | 0.21±0.03  0.12±0.00 |99.61±0.01 99.72±0.00

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 262±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.56±0.00 | 0.18±0.01  0.11±0.00 |99.61±0.01 99.72±0.00
    fixed_05             |99.04±0.03 99.19±0.09 | 0.59±0.03  0.47±0.07 |99.39±0.02 99.49±0.06
    platt                |99.34±0.03 99.43±0.04 | 0.30±0.02  0.24±0.04 |99.58±0.02 99.64±0.02
    beta                 |99.37±0.03 99.49±0.09 | 0.24±0.01  0.18±0.10 |99.60±0.02 99.68±0.06
    empirical            |98.95±0.63 99.49±0.09 | 0.64±0.58  0.10±0.01 |99.33±0.41 99.68±0.06
    empirical_cumulative |99.38±0.01 99.56±0.00 | 0.18±0.01  0.11±0.00 |99.61±0.01 99.72±0.00
    val_cal              |99.38±0.01 99.56±0.00 | 0.21±0.03  0.12±0.00 |99.61±0.01 99.72±0.00

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±71 neurons | 26±8 bits
    GA Neurons  : 200±151 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.11±0.24 99.43±0.03 | 0.21±0.12  0.14±0.07 |99.44±0.15 99.64±0.02
    fixed_05             |98.50±0.65 99.02±0.10 | 0.99±0.49  0.62±0.08 |99.04±0.42 99.38±0.06
    platt                |99.05±0.32 99.39±0.05 | 0.36±0.10  0.28±0.05 |99.40±0.20 99.61±0.03
    beta                 |99.03±0.32 99.41±0.06 | 0.30±0.10  0.24±0.07 |99.39±0.20 99.63±0.04
    empirical            |99.07±0.29 99.42±0.02 | 0.46±0.24  0.20±0.04 |99.41±0.19 99.63±0.01
    empirical_cumulative |99.10±0.23 99.43±0.03 | 0.17±0.06  0.14±0.07 |99.44±0.14 99.64±0.02
    val_cal              |99.11±0.24 99.43±0.03 | 0.21±0.12  0.14±0.08 |99.44±0.15 99.64±0.02

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 262±66 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.56±0.00 | 0.18±0.01  0.11±0.00 |99.61±0.01 99.72±0.00
    fixed_05             |99.04±0.03 99.19±0.09 | 0.59±0.03  0.47±0.07 |99.39±0.02 99.49±0.06
    platt                |99.34±0.03 99.43±0.04 | 0.30±0.02  0.24±0.04 |99.58±0.02 99.64±0.02
    beta                 |99.37±0.03 99.49±0.09 | 0.24±0.01  0.18±0.10 |99.60±0.02 99.68±0.06
    empirical            |98.95±0.63 99.49±0.09 | 0.64±0.58  0.10±0.01 |99.33±0.41 99.68±0.06
    empirical_cumulative |99.38±0.01 99.56±0.00 | 0.18±0.01  0.11±0.00 |99.61±0.01 99.72±0.00
    val_cal              |99.38±0.01 99.56±0.00 | 0.21±0.03  0.12±0.00 |99.61±0.01 99.72±0.00

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 102±7 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.32±0.05 99.40±0.06 | 0.28±0.02  0.22±0.07 |99.57±0.03 99.62±0.04
    fixed_05             |99.01±0.07 99.06±0.05 | 0.60±0.06  0.59±0.04 |99.37±0.05 99.40±0.03
    platt                |99.30±0.04 99.23±0.05 | 0.30±0.00  0.43±0.03 |99.56±0.02 99.51±0.03
    beta                 |99.28±0.04 99.39±0.05 | 0.22±0.02  0.23±0.09 |99.55±0.03 99.61±0.03
    empirical            |99.30±0.04 98.92±0.32 | 0.27±0.03  0.70±0.29 |99.56±0.02 99.31±0.21
    empirical_cumulative |99.30±0.05 99.40±0.06 | 0.21±0.00  0.22±0.07 |99.56±0.03 99.62±0.04
    val_cal              |99.32±0.05 99.40±0.06 | 0.28±0.01  0.22±0.07 |99.57±0.03 99.62±0.04


## XDS-cicids-16b-Wb  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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


## XDS-cicids-16b-Wbu  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)

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

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 138±50 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.53±0.00 | 0.18±0.01  0.10±0.00 |99.61±0.01 99.70±0.00
    fixed_05             |99.04±0.03 99.13±0.03 | 0.59±0.03  0.52±0.03 |99.39±0.02 99.45±0.02
    platt                |99.34±0.03 99.42±0.03 | 0.30±0.02  0.26±0.03 |99.58±0.02 99.63±0.02
    beta                 |99.37±0.03 99.47±0.06 | 0.24±0.01  0.20±0.05 |99.60±0.02 99.67±0.04
    empirical            |98.95±0.63 99.18±0.36 | 0.64±0.58  0.47±0.33 |99.33±0.41 99.48±0.23
    empirical_cumulative |99.38±0.01 99.53±0.00 | 0.18±0.01  0.10±0.00 |99.61±0.01 99.70±0.00
    val_cal              |99.38±0.01 99.53±0.00 | 0.21±0.03  0.10±0.00 |99.61±0.01 99.70±0.00

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 138±50 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.53±0.00 | 0.18±0.01  0.10±0.00 |99.61±0.01 99.70±0.00
    fixed_05             |99.04±0.03 99.13±0.03 | 0.59±0.03  0.52±0.03 |99.39±0.02 99.45±0.02
    platt                |99.34±0.03 99.42±0.03 | 0.30±0.02  0.26±0.03 |99.58±0.02 99.63±0.02
    beta                 |99.37±0.03 99.47±0.06 | 0.24±0.01  0.20±0.05 |99.60±0.02 99.67±0.04
    empirical            |98.95±0.63 99.18±0.36 | 0.64±0.58  0.47±0.33 |99.33±0.41 99.48±0.23
    empirical_cumulative |99.38±0.01 99.53±0.00 | 0.18±0.01  0.10±0.00 |99.61±0.01 99.70±0.00
    val_cal              |99.38±0.01 99.53±0.00 | 0.21±0.03  0.10±0.00 |99.61±0.01 99.70±0.00

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 202±135 neurons | 28±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.09±0.20 99.09±0.17 | 0.44±0.19  0.29±0.11 |99.42±0.13 99.43±0.11
    fixed_05             |98.83±0.14 98.74±0.27 | 0.74±0.11  0.82±0.20 |99.25±0.09 99.19±0.17
    platt                |99.04±0.26 99.04±0.22 | 0.40±0.11  0.41±0.09 |99.39±0.17 99.39±0.14
    beta                 |99.02±0.15 99.00±0.17 | 0.29±0.05  0.29±0.04 |99.38±0.09 99.37±0.11
    empirical            |99.05±0.26 99.08±0.19 | 0.50±0.27  0.46±0.14 |99.39±0.17 99.41±0.12
    empirical_cumulative |99.03±0.17 98.97±0.08 | 0.11±0.04  0.08±0.00 |99.39±0.11 99.35±0.05
    val_cal              |99.09±0.20 99.09±0.17 | 0.27±0.05  0.29±0.11 |99.43±0.13 99.43±0.11

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 138±50 neurons | 34±1 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.38±0.01 99.53±0.00 | 0.18±0.01  0.10±0.00 |99.61±0.01 99.70±0.00
    fixed_05             |99.04±0.03 99.13±0.03 | 0.59±0.03  0.52±0.03 |99.39±0.02 99.45±0.02
    platt                |99.34±0.03 99.42±0.03 | 0.30±0.02  0.26±0.03 |99.58±0.02 99.63±0.02
    beta                 |99.37±0.03 99.47±0.06 | 0.24±0.01  0.20±0.05 |99.60±0.02 99.67±0.04
    empirical            |98.95±0.63 99.18±0.36 | 0.64±0.58  0.47±0.33 |99.33±0.41 99.48±0.23
    empirical_cumulative |99.38±0.01 99.53±0.00 | 0.18±0.01  0.10±0.00 |99.61±0.01 99.70±0.00
    val_cal              |99.38±0.01 99.53±0.00 | 0.21±0.03  0.10±0.00 |99.61±0.01 99.70±0.00

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 200±141 neurons | 34±0 bits
    GA Neurons  : 240±74 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.34±0.01 99.47±0.03 | 0.26±0.05  0.10±0.01 |99.58±0.01 99.66±0.02
    fixed_05             |99.09±0.02 99.17±0.02 | 0.54±0.02  0.50±0.00 |99.42±0.01 99.47±0.01
    platt                |99.32±0.01 99.37±0.01 | 0.30±0.01  0.29±0.01 |99.57±0.01 99.60±0.00
    beta                 |99.33±0.03 99.41±0.01 | 0.23±0.01  0.22±0.03 |99.58±0.02 99.62±0.01
    empirical            |99.33±0.02 99.25±0.18 | 0.27±0.01  0.43±0.17 |99.57±0.01 99.52±0.12
    empirical_cumulative |99.33±0.03 99.47±0.03 | 0.21±0.02  0.10±0.01 |99.58±0.02 99.66±0.02
    val_cal              |99.34±0.01 99.47±0.03 | 0.26±0.05  0.10±0.01 |99.59±0.01 99.66±0.02


## XDS-cicids-16b-Wc  (2 flows × 2 phases, seeds: [25608, 82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal
    Best FPR (any F1)        |  99.45% |   0.09% |  99.65% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.45% |   0.09% |  99.65% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.55% |   0.11% |  99.72% | r25608 GA best_acc       train_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 300±283 neurons | 34±0 bits
    GA Neurons  : 182±124 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.35±0.00 99.40±0.13 | 0.25±0.03  0.22±0.16 |99.59±0.00 99.62±0.08
    fixed_05             |99.07±0.01 99.08±0.06 | 0.56±0.00  0.58±0.04 |99.41±0.00 99.41±0.04
    platt                |99.32±0.01 99.23±0.11 | 0.31±0.01  0.43±0.08 |99.57±0.01 99.51±0.07
    beta                 |99.33±0.03 99.36±0.09 | 0.22±0.02  0.27±0.07 |99.58±0.02 99.60±0.06
    empirical            |99.34±0.01 99.31±0.21 | 0.26±0.01  0.29±0.27 |99.58±0.01 99.56±0.14
    empirical_cumulative |99.35±0.01 99.40±0.13 | 0.22±0.01  0.22±0.16 |99.59±0.01 99.62±0.08
    val_cal              |99.35±0.00 99.40±0.13 | 0.25±0.03  0.22±0.16 |99.59±0.00 99.62±0.08

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±141 neurons | 34±0 bits
    GA Neurons  : 198±135 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.39±0.02 99.55±0.01 | 0.20±0.02  0.10±0.01 |99.61±0.01 99.71±0.01
    fixed_05             |99.03±0.02 99.21±0.05 | 0.60±0.02  0.47±0.03 |99.38±0.01 99.50±0.03
    platt                |99.33±0.02 99.40±0.03 | 0.30±0.01  0.28±0.02 |99.57±0.01 99.62±0.02
    beta                 |99.37±0.03 99.50±0.01 | 0.23±0.03  0.16±0.01 |99.60±0.02 99.69±0.00
    empirical            |98.67±0.23 99.44±0.16 | 0.92±0.19  0.24±0.19 |99.15±0.15 99.65±0.10
    empirical_cumulative |99.39±0.02 99.55±0.01 | 0.20±0.02  0.10±0.01 |99.61±0.01 99.71±0.01
    val_cal              |99.39±0.02 99.55±0.01 | 0.20±0.02  0.10±0.01 |99.61±0.01 99.71±0.01

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±71 neurons | 27±10 bits
    GA Neurons  : 174±115 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.12±0.28 99.44±0.01 | 0.18±0.04  0.14±0.07 |99.45±0.18 99.65±0.01
    fixed_05             |98.81±0.34 99.04±0.01 | 0.74±0.24  0.61±0.00 |99.24±0.22 99.39±0.01
    platt                |99.06±0.33 99.36±0.03 | 0.36±0.08  0.30±0.00 |99.41±0.21 99.60±0.02
    beta                 |99.05±0.34 99.41±0.01 | 0.31±0.09  0.23±0.01 |99.40±0.21 99.63±0.01
    empirical            |99.06±0.37 99.33±0.13 | 0.47±0.31  0.31±0.18 |99.41±0.24 99.58±0.08
    empirical_cumulative |99.12±0.28 99.44±0.01 | 0.18±0.04  0.13±0.06 |99.45±0.18 99.65±0.01
    val_cal              |99.12±0.29 99.44±0.01 | 0.23±0.03  0.14±0.07 |99.45±0.18 99.65±0.01

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±141 neurons | 34±0 bits
    GA Neurons  : 198±135 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.39±0.02 99.55±0.01 | 0.20±0.02  0.10±0.01 |99.61±0.01 99.71±0.01
    fixed_05             |99.03±0.02 99.21±0.05 | 0.60±0.02  0.47±0.03 |99.38±0.01 99.50±0.03
    platt                |99.33±0.02 99.40±0.03 | 0.30±0.01  0.28±0.02 |99.57±0.01 99.62±0.02
    beta                 |99.37±0.03 99.50±0.01 | 0.23±0.03  0.16±0.01 |99.60±0.02 99.69±0.00
    empirical            |98.67±0.23 99.44±0.16 | 0.92±0.19  0.24±0.19 |99.15±0.15 99.65±0.10
    empirical_cumulative |99.39±0.02 99.55±0.01 | 0.20±0.02  0.10±0.01 |99.61±0.01 99.71±0.01
    val_cal              |99.39±0.02 99.55±0.01 | 0.20±0.02  0.10±0.01 |99.61±0.01 99.71±0.01

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 182±124 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.35±0.00 99.40±0.13 | 0.25±0.03  0.22±0.16 |99.59±0.00 99.62±0.08
    fixed_05             |99.07±0.01 99.08±0.06 | 0.56±0.00  0.58±0.04 |99.41±0.00 99.41±0.04
    platt                |99.32±0.01 99.23±0.11 | 0.31±0.01  0.43±0.08 |99.57±0.01 99.51±0.07
    beta                 |99.33±0.03 99.36±0.09 | 0.22±0.02  0.27±0.07 |99.58±0.02 99.60±0.06
    empirical            |99.34±0.01 99.31±0.21 | 0.26±0.01  0.29±0.27 |99.58±0.01 99.56±0.14
    empirical_cumulative |99.35±0.01 99.40±0.13 | 0.22±0.01  0.22±0.16 |99.59±0.01 99.62±0.08
    val_cal              |99.35±0.00 99.40±0.13 | 0.25±0.03  0.22±0.16 |99.59±0.00 99.62±0.08


## XDS-cicids-32b-Wa  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

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

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 104±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  
    fixed_05             |  98.95     99.04   |    0.64      0.58   |   99.33     99.39  
    platt                |  99.21     99.35   |    0.33      0.30   |   99.50     99.59  
    beta                 |  99.25     99.43   |    0.26      0.22   |   99.53     99.64  
    empirical            |  99.15     99.56   |    0.46      0.07   |   99.46     99.72  
    empirical_cumulative |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  
    val_cal              |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 104±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  
    fixed_05             |  98.95     99.04   |    0.64      0.58   |   99.33     99.39  
    platt                |  99.21     99.35   |    0.33      0.30   |   99.50     99.59  
    beta                 |  99.25     99.43   |    0.26      0.22   |   99.53     99.64  
    empirical            |  99.15     99.56   |    0.46      0.07   |   99.46     99.72  
    empirical_cumulative |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  
    val_cal              |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 101±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  85.85     99.36   |    0.15      0.24   |   92.39     99.59  
    fixed_05             |  71.16     99.19   |   28.91      0.47   |   75.48     99.48  
    platt                |  82.66     99.35   |    9.11      0.31   |   88.41     99.59  
    beta                 |  85.85     99.35   |    0.15      0.28   |   92.39     99.59  
    empirical            |  85.85     99.36   |    0.15      0.24   |   92.39     99.59  
    empirical_cumulative |  85.85     99.36   |    0.15      0.24   |   92.39     99.59  
    val_cal              |  85.85     99.36   |    0.15      0.30   |   92.39     99.59  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 104±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  
    fixed_05             |  98.95     99.04   |    0.64      0.58   |   99.33     99.39  
    platt                |  99.21     99.35   |    0.33      0.30   |   99.50     99.59  
    beta                 |  99.25     99.43   |    0.26      0.22   |   99.53     99.64  
    empirical            |  99.15     99.56   |    0.46      0.07   |   99.46     99.72  
    empirical_cumulative |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  
    val_cal              |  99.27     99.57   |    0.27      0.08   |   99.54     99.73  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 97±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.24     99.36   |    0.27      0.29   |   99.52     99.60  
    fixed_05             |  98.90     99.14   |    0.67      0.54   |   99.30     99.45  
    platt                |  99.20     99.32   |    0.35      0.35   |   99.49     99.57  
    beta                 |  99.22     99.32   |    0.26      0.28   |   99.51     99.57  
    empirical            |  99.15     99.32   |    0.45      0.25   |   99.46     99.57  
    empirical_cumulative |  99.22     99.36   |    0.22      0.29   |   99.51     99.60  
    val_cal              |  99.24     99.36   |    0.27      0.29   |   99.52     99.60  


## XDS-cicids-32b-Wb  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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


## XDS-cicids-32b-Wbu  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)

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


## XDS-cicids-32b-Wc  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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


## XDS-cicids-64b-Wa  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.56% |   0.09% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.56% |   0.09% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.56% |   0.09% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.56% |   0.09% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.56% |   0.09% |  99.73% | r82096 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.56% |   0.09% |  99.73% | r82096 GA best_acc       train_cal
    Best FPR (any F1)        |  99.56% |   0.09% |  99.72% | r82096 GA best_f1        empirical
    Best FPR (F1>80%)        |  99.56% |   0.09% |  99.72% | r82096 GA best_f1        empirical
    Best Acc (any FPR)       |  99.56% |   0.09% |  99.73% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 105±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.31     99.56   |    0.30      0.09   |   99.57     99.73  
    fixed_05             |  99.04     99.02   |    0.57      0.62   |   99.39     99.38  
    platt                |  99.31     99.39   |    0.31      0.28   |   99.56     99.62  
    beta                 |  99.28     99.53   |    0.29      0.15   |   99.54     99.70  
    empirical            |  99.28     99.56   |    0.35      0.09   |   99.54     99.72  
    empirical_cumulative |  99.31     99.56   |    0.30      0.09   |   99.57     99.73  
    val_cal              |  99.31     99.56   |    0.31      0.09   |   99.57     99.73  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 105±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.31     99.56   |    0.30      0.09   |   99.57     99.73  
    fixed_05             |  99.04     99.02   |    0.57      0.62   |   99.39     99.38  
    platt                |  99.31     99.39   |    0.31      0.28   |   99.56     99.62  
    beta                 |  99.28     99.53   |    0.29      0.15   |   99.54     99.70  
    empirical            |  99.28     99.56   |    0.35      0.09   |   99.54     99.72  
    empirical_cumulative |  99.31     99.56   |    0.30      0.09   |   99.57     99.73  
    val_cal              |  99.31     99.56   |    0.31      0.09   |   99.57     99.73  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 32±0 bits
    GA Neurons  : 112±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.25     99.52   |    0.21      0.13   |   99.53     99.70  
    fixed_05             |  98.97     98.99   |    0.62      0.64   |   99.35     99.36  
    platt                |  99.19     99.41   |    0.34      0.26   |   99.49     99.62  
    beta                 |  99.21     99.49   |    0.26      0.16   |   99.50     99.68  
    empirical            |  99.18     99.52   |    0.41      0.13   |   99.48     99.70  
    empirical_cumulative |  99.25     99.52   |    0.21      0.13   |   99.53     99.70  
    val_cal              |  99.25     99.52   |    0.21      0.13   |   99.53     99.70  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 105±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.31     99.56   |    0.30      0.09   |   99.57     99.73  
    fixed_05             |  99.04     99.02   |    0.57      0.62   |   99.39     99.38  
    platt                |  99.31     99.39   |    0.31      0.28   |   99.56     99.62  
    beta                 |  99.28     99.53   |    0.29      0.15   |   99.54     99.70  
    empirical            |  99.28     99.56   |    0.35      0.09   |   99.54     99.72  
    empirical_cumulative |  99.31     99.56   |    0.30      0.09   |   99.57     99.73  
    val_cal              |  99.31     99.56   |    0.31      0.09   |   99.57     99.73  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 94±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.25     99.27   |    0.26      0.20   |   99.53     99.54  
    fixed_05             |  99.02     98.98   |    0.59      0.66   |   99.38     99.35  
    platt                |  99.23     99.15   |    0.34      0.48   |   99.51     99.46  
    beta                 |  99.24     99.27   |    0.28      0.20   |   99.52     99.54  
    empirical            |  99.25     99.26   |    0.36      0.20   |   99.52     99.53  
    empirical_cumulative |  99.25     99.27   |    0.26      0.20   |   99.53     99.54  
    val_cal              |  99.25     99.27   |    0.25      0.20   |   99.53     99.54  


## XDS-cicids-64b-Wb  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

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


## XDS-cicids-64b-Wbu  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)

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

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 172±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.32     99.59   |    0.30      0.07   |   99.57     99.74  
    fixed_05             |  99.04     99.06   |    0.57      0.59   |   99.39     99.40  
    platt                |  99.31     99.42   |    0.31      0.25   |   99.56     99.63  
    beta                 |  99.25     99.48   |    0.28      0.19   |   99.53     99.67  
    empirical            |  99.30     99.57   |    0.29      0.07   |   99.56     99.73  
    empirical_cumulative |  99.23     99.59   |    0.20      0.07   |   99.52     99.74  
    val_cal              |  99.32     99.59   |    0.30      0.07   |   99.57     99.74  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 172±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.32     99.59   |    0.30      0.07   |   99.57     99.74  
    fixed_05             |  99.04     99.06   |    0.57      0.59   |   99.39     99.40  
    platt                |  99.31     99.42   |    0.31      0.25   |   99.56     99.63  
    beta                 |  99.25     99.48   |    0.28      0.19   |   99.53     99.67  
    empirical            |  99.30     99.57   |    0.29      0.07   |   99.56     99.73  
    empirical_cumulative |  99.23     99.59   |    0.20      0.07   |   99.52     99.74  
    val_cal              |  99.32     99.59   |    0.30      0.07   |   99.57     99.74  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 176±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.24     99.55   |    0.22      0.06   |   99.52     99.72  
    fixed_05             |  98.97     99.09   |    0.62      0.57   |   99.35     99.42  
    platt                |  99.17     99.36   |    0.37      0.31   |   99.47     99.60  
    beta                 |  99.19     99.40   |    0.31      0.26   |   99.49     99.62  
    empirical            |  99.19     99.54   |    0.39      0.11   |   99.49     99.71  
    empirical_cumulative |  99.23     99.55   |    0.20      0.06   |   99.52     99.72  
    val_cal              |  99.24     99.55   |    0.22      0.06   |   99.52     99.72  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 172±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.32     99.59   |    0.30      0.07   |   99.57     99.74  
    fixed_05             |  99.04     99.06   |    0.57      0.59   |   99.39     99.40  
    platt                |  99.31     99.42   |    0.31      0.25   |   99.56     99.63  
    beta                 |  99.25     99.48   |    0.28      0.19   |   99.53     99.67  
    empirical            |  99.30     99.57   |    0.29      0.07   |   99.56     99.73  
    empirical_cumulative |  99.23     99.59   |    0.20      0.07   |   99.52     99.74  
    val_cal              |  99.32     99.59   |    0.30      0.07   |   99.57     99.74  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 188±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.26     99.27   |    0.34      0.29   |   99.53     99.54  
    fixed_05             |  98.99     99.19   |    0.63      0.47   |   99.35     99.48  
    platt                |  99.25     99.24   |    0.34      0.35   |   99.53     99.52  
    beta                 |  99.22     99.21   |    0.29      0.26   |   99.51     99.50  
    empirical            |  99.24     99.21   |    0.33      0.42   |   99.52     99.50  
    empirical_cumulative |  99.23     99.15   |    0.21      0.13   |   99.51     99.46  
    val_cal              |  99.26     99.27   |    0.34      0.29   |   99.53     99.54  


## XDS-cicids-64b-Wc  (1 flows × 2 phases, seeds: [82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 376±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.34   |    0.33      0.28   |   99.55     99.58  
    fixed_05             |  99.02     99.10   |    0.59      0.56   |   99.38     99.43  
    platt                |  99.28     99.30   |    0.33      0.34   |   99.54     99.56  
    beta                 |  99.23     99.32   |    0.27      0.26   |   99.51     99.57  
    empirical            |  99.26     99.27   |    0.32      0.22   |   99.53     99.54  
    empirical_cumulative |  99.28     99.34   |    0.33      0.28   |   99.55     99.58  
    val_cal              |  99.28     99.34   |    0.33      0.28   |   99.55     99.58  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 202±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.54   |    0.28      0.10   |   99.57     99.71  
    fixed_05             |  99.05     99.09   |    0.58      0.54   |   99.40     99.42  
    platt                |  99.32     99.37   |    0.30      0.29   |   99.57     99.60  
    beta                 |  99.25     99.50   |    0.24      0.14   |   99.52     99.69  
    empirical            |  99.33     99.53   |    0.28      0.09   |   99.57     99.70  
    empirical_cumulative |  99.33     99.54   |    0.28      0.09   |   99.57     99.71  
    val_cal              |  99.33     99.54   |    0.29      0.10   |   99.58     99.71  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 24±0 bits
    GA Neurons  : 210±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  95.14     99.45   |    2.64      0.07   |   96.86     99.66  
    fixed_05             |  92.45     99.09   |    5.88      0.55   |   94.85     99.42  
    platt                |  95.14     99.36   |    2.64      0.27   |   96.86     99.59  
    beta                 |  94.90     99.44   |    0.22      0.19   |   96.95     99.65  
    empirical            |  94.90     99.07   |    0.22      0.58   |   96.95     99.41  
    empirical_cumulative |  94.91     99.45   |    0.22      0.07   |   96.95     99.66  
    val_cal              |  95.14     99.45   |    2.64      0.07   |   96.86     99.66  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 202±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.54   |    0.28      0.10   |   99.57     99.71  
    fixed_05             |  99.05     99.09   |    0.58      0.54   |   99.40     99.42  
    platt                |  99.32     99.37   |    0.30      0.29   |   99.57     99.60  
    beta                 |  99.25     99.50   |    0.24      0.14   |   99.52     99.69  
    empirical            |  99.33     99.53   |    0.28      0.09   |   99.57     99.70  
    empirical_cumulative |  99.33     99.54   |    0.28      0.09   |   99.57     99.71  
    val_cal              |  99.33     99.54   |    0.29      0.10   |   99.58     99.71  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 376±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.34   |    0.33      0.28   |   99.55     99.58  
    fixed_05             |  99.02     99.10   |    0.59      0.56   |   99.38     99.43  
    platt                |  99.28     99.30   |    0.33      0.34   |   99.54     99.56  
    beta                 |  99.23     99.32   |    0.27      0.26   |   99.51     99.57  
    empirical            |  99.26     99.27   |    0.32      0.22   |   99.53     99.54  
    empirical_cumulative |  99.28     99.34   |    0.33      0.28   |   99.55     99.58  
    val_cal              |  99.28     99.34   |    0.33      0.28   |   99.55     99.58  


## XDS-cicids-96b-Wa  (1 flows × 2 phases, seeds: [82096])

    Weight : Wa (CIC-IoT legacy, ce=0.35 acc=0.30)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       val_cal
    Best F1 (FPR<14%)        |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       val_cal
    Best F1 (FPR<10%)        |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       val_cal
    Best F1 (FPR<6%)         |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       val_cal
    Best F1 (FPR<5%)         |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       val_cal
    Best F1 (FPR<4%)         |  99.59% |   0.07% |  99.74% | r82096 GA best_acc       val_cal
    Best FPR (any F1)        |  99.58% |   0.06% |  99.73% | r82096 GA best_acc       empirical
    Best FPR (F1>80%)        |  99.58% |   0.06% |  99.73% | r82096 GA best_acc       empirical
    Best Acc (any FPR)       |  99.59% |   0.07% |  99.74% | r82096 GA best_f1        train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 107±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.59   |    0.24      0.07   |   99.55     99.74  
    fixed_05             |  98.97     99.22   |    0.64      0.45   |   99.34     99.51  
    platt                |  99.27     99.40   |    0.35      0.30   |   99.54     99.62  
    beta                 |  99.30     99.48   |    0.30      0.21   |   99.56     99.67  
    empirical            |  99.27     99.58   |    0.35      0.06   |   99.54     99.73  
    empirical_cumulative |  99.28     99.59   |    0.24      0.07   |   99.55     99.74  
    val_cal              |  99.30     99.59   |    0.30      0.07   |   99.56     99.74  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 107±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.59   |    0.24      0.07   |   99.55     99.74  
    fixed_05             |  98.97     99.22   |    0.64      0.45   |   99.34     99.51  
    platt                |  99.27     99.40   |    0.35      0.30   |   99.54     99.62  
    beta                 |  99.30     99.48   |    0.30      0.21   |   99.56     99.67  
    empirical            |  99.27     99.58   |    0.35      0.06   |   99.54     99.73  
    empirical_cumulative |  99.28     99.59   |    0.24      0.07   |   99.55     99.74  
    val_cal              |  99.30     99.59   |    0.30      0.07   |   99.56     99.74  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 88±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  74.71     99.51   |    8.15      0.08   |   84.70     99.69  
    fixed_05             |  74.71     99.24   |    8.15      0.45   |   84.70     99.52  
    platt                |  71.21     99.39   |    0.94      0.31   |   86.43     99.61  
    beta                 |  71.49     99.40   |    0.16      0.27   |   86.89     99.62  
    empirical            |  71.49     99.33   |    0.16      0.38   |   86.89     99.58  
    empirical_cumulative |  71.49     99.51   |    0.16      0.08   |   86.89     99.69  
    val_cal              |  74.71     99.51   |    8.15      0.08   |   84.70     99.69  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 107±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.28     99.59   |    0.24      0.07   |   99.55     99.74  
    fixed_05             |  98.97     99.22   |    0.64      0.45   |   99.34     99.51  
    platt                |  99.27     99.40   |    0.35      0.30   |   99.54     99.62  
    beta                 |  99.30     99.48   |    0.30      0.21   |   99.56     99.67  
    empirical            |  99.27     99.58   |    0.35      0.06   |   99.54     99.73  
    empirical_cumulative |  99.28     99.59   |    0.24      0.07   |   99.55     99.74  
    val_cal              |  99.30     99.59   |    0.30      0.07   |   99.56     99.74  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 79±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.24     99.39   |    0.25      0.23   |   99.52     99.62  
    fixed_05             |  98.98     99.19   |    0.63      0.48   |   99.35     99.49  
    platt                |  99.21     99.23   |    0.34      0.44   |   99.50     99.51  
    beta                 |  99.23     99.21   |    0.25      0.46   |   99.52     99.50  
    empirical            |  99.24     99.35   |    0.35      0.30   |   99.52     99.59  
    empirical_cumulative |  99.24     99.39   |    0.25      0.23   |   99.52     99.62  
    val_cal              |  99.24     99.39   |    0.38      0.23   |   99.52     99.62  


## XDS-cicids-96b-Wb  (2 flows × 2 phases, seeds: [8188, 82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<14%)        |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<10%)        |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<6%)         |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<5%)         |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best F1 (FPR<4%)         |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal
    Best FPR (any F1)        |  99.52% |   0.10% |  99.70% | r8188 GA best_acc       empirical
    Best FPR (F1>80%)        |  99.52% |   0.10% |  99.70% | r8188 GA best_acc       empirical
    Best Acc (any FPR)       |  99.53% |   0.10% |  99.70% | r8188 GA best_acc       train_cal

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 450±71 neurons | 33±1 bits
    GA Neurons  : 101±1 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.26±0.01 99.52±0.00 | 0.29±0.10  0.11±0.01 |99.53±0.01 99.70±0.00
    fixed_05             |99.02±0.01 99.14±0.15 | 0.59±0.01  0.53±0.11 |99.38±0.01 99.45±0.10
    platt                |99.22±0.03 99.39±0.03 | 0.33±0.01  0.27±0.00 |99.51±0.02 99.61±0.02
    beta                 |99.23±0.01 99.48±0.03 | 0.27±0.02  0.16±0.05 |99.51±0.00 99.67±0.02
    empirical            |99.25±0.02 99.46±0.08 | 0.35±0.00  0.19±0.12 |99.52±0.01 99.66±0.05
    empirical_cumulative |99.23±0.01 99.52±0.00 | 0.22±0.01  0.11±0.01 |99.52±0.01 99.70±0.00
    val_cal              |99.26±0.01 99.52±0.00 | 0.30±0.10  0.11±0.01 |99.53±0.00 99.70±0.00

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 450±71 neurons | 33±1 bits
    GA Neurons  : 101±1 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.26±0.01 99.52±0.00 | 0.29±0.10  0.11±0.01 |99.53±0.01 99.70±0.00
    fixed_05             |99.02±0.01 99.14±0.15 | 0.59±0.01  0.53±0.11 |99.38±0.01 99.45±0.10
    platt                |99.22±0.03 99.39±0.03 | 0.33±0.01  0.27±0.00 |99.51±0.02 99.61±0.02
    beta                 |99.23±0.01 99.48±0.03 | 0.27±0.02  0.16±0.05 |99.51±0.00 99.67±0.02
    empirical            |99.25±0.02 99.46±0.08 | 0.35±0.00  0.19±0.12 |99.52±0.01 99.66±0.05
    empirical_cumulative |99.23±0.01 99.52±0.00 | 0.22±0.01  0.11±0.01 |99.52±0.01 99.70±0.00
    val_cal              |99.26±0.01 99.52±0.00 | 0.30±0.10  0.11±0.01 |99.53±0.00 99.70±0.00

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 400±0 neurons | 28±0 bits
    GA Neurons  : 107±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |85.81±18.80 85.94±18.98 |11.26±15.44 11.21±15.52 |88.95±14.82 89.03±14.94
    fixed_05             |77.05±30.91 77.17±31.09 |27.38±37.79 27.30±37.90 |77.91±30.26 77.99±30.37
    platt                |71.81±38.57 71.94±38.75 | 0.18±0.26  0.15±0.21 |89.87±13.51 89.95±13.62
    beta                 |77.12±31.02 77.27±31.22 |27.21±38.05 27.19±38.07 |77.96±30.33 78.05±30.46
    empirical            |77.09±30.97 77.27±31.23 |27.34±37.86 27.19±38.07 |77.94±30.29 78.05±30.46
    empirical_cumulative |71.76±38.49 71.95±38.75 | 0.06±0.08  0.11±0.16 |89.84±13.46 89.95±13.63
    val_cal              |85.81±18.80 85.94±18.98 |11.26±15.44 11.21±15.52 |88.95±14.82 89.03±14.94

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 450±71 neurons | 33±1 bits
    GA Neurons  : 101±1 neurons | 33±2 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.26±0.01 99.52±0.00 | 0.29±0.10  0.11±0.01 |99.53±0.01 99.70±0.00
    fixed_05             |99.02±0.01 99.14±0.15 | 0.59±0.01  0.53±0.11 |99.38±0.01 99.45±0.10
    platt                |99.22±0.03 99.39±0.03 | 0.33±0.01  0.27±0.00 |99.51±0.02 99.61±0.02
    beta                 |99.23±0.01 99.48±0.03 | 0.27±0.02  0.16±0.05 |99.51±0.00 99.67±0.02
    empirical            |99.25±0.02 99.46±0.08 | 0.35±0.00  0.19±0.12 |99.52±0.01 99.66±0.05
    empirical_cumulative |99.23±0.01 99.52±0.00 | 0.22±0.01  0.11±0.01 |99.52±0.01 99.70±0.00
    val_cal              |99.26±0.01 99.52±0.00 | 0.30±0.10  0.11±0.01 |99.53±0.00 99.70±0.00

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 94±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.23±0.02 99.40±0.17 | 0.29±0.00  0.24±0.17 |99.52±0.01 99.62±0.11
    fixed_05             |98.91±0.03 99.01±0.05 | 0.67±0.03  0.64±0.05 |99.31±0.02 99.37±0.03
    platt                |99.19±0.03 99.29±0.07 | 0.34±0.01  0.37±0.06 |99.49±0.02 99.55±0.05
    beta                 |99.22±0.01 99.30±0.14 | 0.28±0.02  0.27±0.03 |99.51±0.00 99.56±0.09
    empirical            |99.23±0.03 99.40±0.17 | 0.38±0.03  0.24±0.17 |99.51±0.02 99.62±0.11
    empirical_cumulative |99.21±0.03 99.34±0.25 | 0.23±0.01  0.15±0.04 |99.50±0.02 99.59±0.15
    val_cal              |99.24±0.03 99.40±0.17 | 0.33±0.05  0.24±0.17 |99.52±0.02 99.62±0.11


## XDS-cicids-96b-Wbu  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)

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


## XDS-cicids-96b-Wc  (2 flows × 2 phases, seeds: [8188, 82096])

    Weight : Wc (CE-heavy NEW, ce=0.70 acc=0.10)

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

### best_fitness  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±212 neurons | 34±0 bits
    GA Neurons  : 96±2 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.24±0.01 99.33±0.05 | 0.27±0.03  0.24±0.03 |99.52±0.01 99.58±0.03
    fixed_05             |98.93±0.06 99.07±0.14 | 0.66±0.05  0.59±0.12 |99.32±0.04 99.41±0.09
    platt                |99.21±0.00 99.30±0.10 | 0.34±0.00  0.37±0.07 |99.50±0.00 99.55±0.06
    beta                 |99.23±0.01 99.31±0.05 | 0.26±0.01  0.28±0.02 |99.51±0.00 99.57±0.03
    empirical            |99.25±0.01 99.33±0.05 | 0.36±0.00  0.29±0.04 |99.52±0.01 99.58±0.03
    empirical_cumulative |99.23±0.00 99.33±0.05 | 0.24±0.02  0.24±0.03 |99.52±0.00 99.58±0.03
    val_cal              |99.25±0.01 99.33±0.05 | 0.37±0.01  0.24±0.03 |99.53±0.01 99.58±0.03

### best_f1  (GS: 2 runs | GA: 2 runs)
    Grid Search : 150±71 neurons | 34±0 bits
    GA Neurons  : 153±74 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.29±0.01 99.54±0.03 | 0.28±0.07  0.09±0.03 |99.55±0.00 99.71±0.02
    fixed_05             |99.03±0.01 99.18±0.08 | 0.59±0.00  0.50±0.06 |99.38±0.00 99.48±0.05
    platt                |99.28±0.00 99.41±0.06 | 0.31±0.01  0.27±0.05 |99.54±0.00 99.62±0.04
    beta                 |99.26±0.02 99.42±0.05 | 0.27±0.02  0.25±0.07 |99.53±0.01 99.63±0.03
    empirical            |99.29±0.01 99.54±0.03 | 0.32±0.01  0.09±0.03 |99.55±0.00 99.71±0.02
    empirical_cumulative |99.28±0.01 99.54±0.03 | 0.22±0.01  0.09±0.03 |99.54±0.01 99.71±0.02
    val_cal              |99.29±0.01 99.54±0.03 | 0.28±0.07  0.09±0.03 |99.55±0.00 99.71±0.02

### best_fpr  (GS: 2 runs | GA: 2 runs)
    Grid Search : 252±350 neurons | 16±17 bits
    GA Neurons  : 152±76 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |86.96±17.33 99.46±0.15 | 4.18±5.61  0.08±0.01 |92.10±10.47 99.66±0.09
    fixed_05             |86.81±17.11 99.13±0.01 | 4.41±5.29  0.54±0.01 |92.00±10.33 99.45±0.01
    platt                |85.16±19.73 99.40±0.08 | 0.66±0.39  0.27±0.06 |92.93±9.20 99.62±0.05
    beta                 |85.33±19.58 99.40±0.08 | 0.22±0.09  0.25±0.08 |93.18±8.90 99.62±0.05
    empirical            |85.25±19.47 99.45±0.15 | 0.35±0.26  0.17±0.15 |93.13±8.83 99.65±0.10
    empirical_cumulative |85.35±19.60 99.46±0.15 | 0.19±0.04  0.08±0.01 |93.19±8.92 99.66±0.09
    val_cal              |86.96±17.33 99.46±0.15 | 4.18±5.61  0.08±0.01 |92.10±10.47 99.66±0.09

### best_acc  (GS: 2 runs | GA: 2 runs)
    Grid Search : 150±71 neurons | 34±0 bits
    GA Neurons  : 153±74 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.29±0.01 99.54±0.03 | 0.28±0.07  0.09±0.03 |99.55±0.00 99.71±0.02
    fixed_05             |99.03±0.01 99.18±0.08 | 0.59±0.00  0.50±0.06 |99.38±0.00 99.48±0.05
    platt                |99.28±0.00 99.41±0.06 | 0.31±0.01  0.27±0.05 |99.54±0.00 99.62±0.04
    beta                 |99.26±0.02 99.42±0.05 | 0.27±0.02  0.25±0.07 |99.53±0.01 99.63±0.03
    empirical            |99.29±0.01 99.54±0.03 | 0.32±0.01  0.09±0.03 |99.55±0.00 99.71±0.02
    empirical_cumulative |99.28±0.01 99.54±0.03 | 0.22±0.01  0.09±0.03 |99.54±0.01 99.71±0.02
    val_cal              |99.29±0.01 99.54±0.03 | 0.28±0.07  0.09±0.03 |99.55±0.00 99.71±0.02

### best_ce  (GS: 2 runs | GA: 2 runs)
    Grid Search : 350±212 neurons | 34±0 bits
    GA Neurons  : 96±2 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |99.24±0.01 99.33±0.05 | 0.27±0.03  0.24±0.03 |99.52±0.01 99.58±0.03
    fixed_05             |98.93±0.06 99.07±0.14 | 0.66±0.05  0.59±0.12 |99.32±0.04 99.41±0.09
    platt                |99.21±0.00 99.30±0.10 | 0.34±0.00  0.37±0.07 |99.50±0.00 99.55±0.06
    beta                 |99.23±0.01 99.31±0.05 | 0.26±0.01  0.28±0.02 |99.51±0.00 99.57±0.03
    empirical            |99.25±0.01 99.33±0.05 | 0.36±0.00  0.29±0.04 |99.52±0.01 99.58±0.03
    empirical_cumulative |99.23±0.00 99.33±0.05 | 0.24±0.02  0.24±0.03 |99.52±0.00 99.58±0.03
    val_cal              |99.25±0.01 99.33±0.05 | 0.37±0.01  0.24±0.03 |99.53±0.01 99.58±0.03

