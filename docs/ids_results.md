# XDS-cicids — width × weight cohort breakdown (16 non-OLD completed)

    Total non-OLD completed : 16  |  Total wall: 76.4h  |  Avg/run: 286m
    Latest done : 09/06/2026 01:45 UTC

    Weight schemes:
      Wa (CIC-IoT legacy, ce=0.35 acc=0.30)
      Wb (paper/PUB50, ce=0.10 acc=0.20)
      Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)
      Wc (CE-heavy NEW, ce=0.70 acc=0.10)


## XDS-cicids-16b-Wa  (1 flows × 2 phases, seeds: [82096])

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
    Best FPR (any F1)        |  99.43% |   0.09% |  99.64% | r82096 GA best_acc       empirical
    Best FPR (F1>80%)        |  99.43% |   0.09% |  99.64% | r82096 GA best_acc       empirical
    Best Acc (any FPR)       |  99.56% |   0.12% |  99.72% | r82096 GA best_f1        val_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 309±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  
    fixed_05             |  99.02     99.26   |    0.61      0.42   |   99.37     99.53  
    platt                |  99.31     99.46   |    0.31      0.21   |   99.57     99.66  
    beta                 |  99.35     99.56   |    0.25      0.11   |   99.59     99.72  
    empirical            |  98.51     99.43   |    1.05      0.09   |   99.04     99.64  
    empirical_cumulative |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  
    val_cal              |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 309±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  
    fixed_05             |  99.02     99.26   |    0.61      0.42   |   99.37     99.53  
    platt                |  99.31     99.46   |    0.31      0.21   |   99.57     99.66  
    beta                 |  99.35     99.56   |    0.25      0.11   |   99.59     99.72  
    empirical            |  98.51     99.43   |    1.05      0.09   |   99.04     99.64  
    empirical_cumulative |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  
    val_cal              |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 20±0 bits
    GA Neurons  : 307±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  98.94     99.45   |    0.13      0.19   |   99.33     99.66  
    fixed_05             |  98.05     99.09   |    1.34      0.56   |   98.74     99.42  
    platt                |  98.82     99.43   |    0.43      0.24   |   99.25     99.64  
    beta                 |  98.81     99.45   |    0.37      0.19   |   99.25     99.65  
    empirical            |  98.87     99.43   |    0.63      0.18   |   99.28     99.64  
    empirical_cumulative |  98.94     99.45   |    0.13      0.19   |   99.33     99.66  
    val_cal              |  98.94     99.45   |    0.13      0.20   |   99.33     99.66  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 309±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  
    fixed_05             |  99.02     99.26   |    0.61      0.42   |   99.37     99.53  
    platt                |  99.31     99.46   |    0.31      0.21   |   99.57     99.66  
    beta                 |  99.35     99.56   |    0.25      0.11   |   99.59     99.72  
    empirical            |  98.51     99.43   |    1.05      0.09   |   99.04     99.64  
    empirical_cumulative |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  
    val_cal              |  99.37     99.56   |    0.19      0.12   |   99.61     99.72  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 107±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.36     99.44   |    0.27      0.17   |   99.59     99.64  
    fixed_05             |  99.07     99.03   |    0.56      0.62   |   99.41     99.38  
    platt                |  99.33     99.19   |    0.30      0.45   |   99.57     99.49  
    beta                 |  99.31     99.42   |    0.21      0.17   |   99.57     99.64  
    empirical            |  99.33     99.15   |    0.25      0.50   |   99.58     99.46  
    empirical_cumulative |  99.34     99.44   |    0.21      0.17   |   99.58     99.64  
    val_cal              |  99.36     99.44   |    0.27      0.17   |   99.59     99.64  


## XDS-cicids-16b-Wb  (1 flows × 2 phases, seeds: [82096])

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

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 282±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.55   |    0.19      0.12   |   99.61     99.71  
    fixed_05             |  99.02     99.23   |    0.61      0.44   |   99.37     99.51  
    platt                |  99.31     99.42   |    0.31      0.26   |   99.57     99.63  
    beta                 |  99.35     99.55   |    0.25      0.12   |   99.59     99.72  
    empirical            |  98.51     99.38   |    1.05      0.09   |   99.04     99.61  
    empirical_cumulative |  99.37     99.55   |    0.19      0.12   |   99.61     99.71  
    val_cal              |  99.37     99.55   |    0.19      0.12   |   99.61     99.72  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 282±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.55   |    0.19      0.12   |   99.61     99.71  
    fixed_05             |  99.02     99.23   |    0.61      0.44   |   99.37     99.51  
    platt                |  99.31     99.42   |    0.31      0.26   |   99.57     99.63  
    beta                 |  99.35     99.55   |    0.25      0.12   |   99.59     99.72  
    empirical            |  98.51     99.38   |    1.05      0.09   |   99.04     99.61  
    empirical_cumulative |  99.37     99.55   |    0.19      0.12   |   99.61     99.71  
    val_cal              |  99.37     99.55   |    0.19      0.12   |   99.61     99.72  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 108±0 neurons | 28±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  98.94     99.03   |    0.57      0.35   |   99.33     99.39  
    fixed_05             |  98.73     98.67   |    0.82      0.86   |   99.19     99.15  
    platt                |  98.85     98.98   |    0.47      0.47   |   99.27     99.36  
    beta                 |  98.92     99.00   |    0.32      0.39   |   99.32     99.37  
    empirical            |  98.86     98.82   |    0.69      0.72   |   99.28     99.25  
    empirical_cumulative |  98.91     98.79   |    0.14      0.06   |   99.32     99.24  
    val_cal              |  98.94     99.03   |    0.24      0.35   |   99.34     99.39  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 282±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.55   |    0.19      0.12   |   99.61     99.71  
    fixed_05             |  99.02     99.23   |    0.61      0.44   |   99.37     99.51  
    platt                |  99.31     99.42   |    0.31      0.26   |   99.57     99.63  
    beta                 |  99.35     99.55   |    0.25      0.12   |   99.59     99.72  
    empirical            |  98.51     99.38   |    1.05      0.09   |   99.04     99.61  
    empirical_cumulative |  99.37     99.55   |    0.19      0.12   |   99.61     99.71  
    val_cal              |  99.37     99.55   |    0.19      0.12   |   99.61     99.72  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 34±0 bits
    GA Neurons  : 293±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.35     99.42   |    0.26      0.20   |   99.59     99.64  
    fixed_05             |  99.07     99.01   |    0.56      0.63   |   99.41     99.37  
    platt                |  99.34     99.33   |    0.29      0.34   |   99.58     99.57  
    beta                 |  99.31     99.40   |    0.21      0.25   |   99.56     99.62  
    empirical            |  99.33     99.39   |    0.21      0.18   |   99.58     99.62  
    empirical_cumulative |  99.30     99.42   |    0.18      0.20   |   99.56     99.64  
    val_cal              |  99.35     99.42   |    0.26      0.20   |   99.59     99.64  


## XDS-cicids-16b-Wbu  (1 flows × 2 phases, seeds: [82096])

    Weight : Wbu (uniform Wb across datasets, ce=0.10 acc=0.20 f1=0.35 fpr=0.35)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.53% |   0.11% |  99.70% | r82096 GA best_f1        train_cal
    Best F1 (FPR<14%)        |  99.53% |   0.11% |  99.70% | r82096 GA best_f1        train_cal
    Best F1 (FPR<10%)        |  99.53% |   0.11% |  99.70% | r82096 GA best_f1        train_cal
    Best F1 (FPR<6%)         |  99.53% |   0.11% |  99.70% | r82096 GA best_f1        train_cal
    Best F1 (FPR<5%)         |  99.53% |   0.11% |  99.70% | r82096 GA best_f1        train_cal
    Best F1 (FPR<4%)         |  99.53% |   0.11% |  99.70% | r82096 GA best_f1        train_cal
    Best FPR (any F1)        |  98.91% |   0.08% |  99.32% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  98.91% |   0.08% |  99.32% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.53% |   0.11% |  99.70% | r82096 GA best_acc       train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 102±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  
    fixed_05             |  99.02     99.11   |    0.61      0.54   |   99.37     99.43  
    platt                |  99.31     99.39   |    0.31      0.28   |   99.57     99.62  
    beta                 |  99.35     99.43   |    0.25      0.24   |   99.59     99.64  
    empirical            |  98.51     99.43   |    1.05      0.23   |   99.04     99.64  
    empirical_cumulative |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  
    val_cal              |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 102±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  
    fixed_05             |  99.02     99.11   |    0.61      0.54   |   99.37     99.43  
    platt                |  99.31     99.39   |    0.31      0.28   |   99.57     99.62  
    beta                 |  99.35     99.43   |    0.25      0.24   |   99.59     99.64  
    empirical            |  98.51     99.43   |    1.05      0.23   |   99.04     99.64  
    empirical_cumulative |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  
    val_cal              |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 28±0 bits
    GA Neurons  : 106±0 neurons | 28±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  98.94     98.97   |    0.57      0.21   |   99.33     99.35  
    fixed_05             |  98.73     98.55   |    0.82      0.96   |   99.19     99.07  
    platt                |  98.85     98.88   |    0.47      0.47   |   99.27     99.29  
    beta                 |  98.92     98.88   |    0.32      0.32   |   99.32     99.29  
    empirical            |  98.86     98.94   |    0.69      0.56   |   99.28     99.33  
    empirical_cumulative |  98.91     98.91   |    0.14      0.08   |   99.32     99.32  
    val_cal              |  98.94     98.97   |    0.24      0.21   |   99.34     99.35  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : — neurons | — bits
    GA Neurons  : 102±0 neurons | 33±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  
    fixed_05             |  99.02     99.11   |    0.61      0.54   |   99.37     99.43  
    platt                |  99.31     99.39   |    0.31      0.28   |   99.57     99.62  
    beta                 |  99.35     99.43   |    0.25      0.24   |   99.59     99.64  
    empirical            |  98.51     99.43   |    1.05      0.23   |   99.04     99.64  
    empirical_cumulative |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  
    val_cal              |  99.37     99.53   |    0.19      0.11   |   99.61     99.70  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 293±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.33     99.45   |    0.29      0.10   |   99.58     99.65  
    fixed_05             |  99.10     99.18   |    0.53      0.50   |   99.43     99.48  
    platt                |  99.33     99.37   |    0.30      0.30   |   99.58     99.60  
    beta                 |  99.31     99.42   |    0.24      0.20   |   99.56     99.63  
    empirical            |  99.31     99.38   |    0.28      0.31   |   99.56     99.60  
    empirical_cumulative |  99.31     99.45   |    0.20      0.10   |   99.57     99.65  
    val_cal              |  99.34     99.45   |    0.29      0.10   |   99.58     99.65  


## XDS-cicids-16b-Wc  (1 flows × 2 phases, seeds: [82096])

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
    Best FPR (any F1)        |  99.45% |   0.09% |  99.65% | r82096 GA best_fpr       empirical_cumulative
    Best FPR (F1>80%)        |  99.45% |   0.09% |  99.65% | r82096 GA best_fpr       empirical_cumulative
    Best Acc (any FPR)       |  99.54% |   0.10% |  99.71% | r82096 GA best_f1        train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 270±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.36     99.49   |    0.27      0.11   |   99.59     99.68  
    fixed_05             |  99.07     99.12   |    0.56      0.55   |   99.41     99.44  
    platt                |  99.33     99.31   |    0.30      0.37   |   99.57     99.56  
    beta                 |  99.31     99.42   |    0.21      0.22   |   99.57     99.64  
    empirical            |  99.33     99.46   |    0.25      0.10   |   99.58     99.66  
    empirical_cumulative |  99.34     99.49   |    0.21      0.11   |   99.58     99.68  
    val_cal              |  99.36     99.49   |    0.27      0.11   |   99.59     99.68  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 293±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.54   |    0.19      0.10   |   99.61     99.71  
    fixed_05             |  99.02     99.24   |    0.61      0.45   |   99.37     99.52  
    platt                |  99.31     99.38   |    0.31      0.29   |   99.57     99.61  
    beta                 |  99.35     99.51   |    0.25      0.16   |   99.59     99.69  
    empirical            |  98.51     99.33   |    1.05      0.37   |   99.04     99.57  
    empirical_cumulative |  99.37     99.54   |    0.19      0.10   |   99.61     99.71  
    val_cal              |  99.37     99.54   |    0.19      0.10   |   99.61     99.71  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 20±0 bits
    GA Neurons  : 256±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  98.92     99.45   |    0.21      0.09   |   99.32     99.65  
    fixed_05             |  98.57     99.04   |    0.91      0.61   |   99.09     99.39  
    platt                |  98.83     99.39   |    0.42      0.29   |   99.26     99.61  
    beta                 |  98.81     99.41   |    0.37      0.23   |   99.25     99.62  
    empirical            |  98.80     99.24   |    0.69      0.44   |   99.24     99.52  
    empirical_cumulative |  98.92     99.45   |    0.21      0.09   |   99.32     99.65  
    val_cal              |  98.92     99.45   |    0.21      0.09   |   99.32     99.65  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 300±0 neurons | 34±0 bits
    GA Neurons  : 293±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.37     99.54   |    0.19      0.10   |   99.61     99.71  
    fixed_05             |  99.02     99.24   |    0.61      0.45   |   99.37     99.52  
    platt                |  99.31     99.38   |    0.31      0.29   |   99.57     99.61  
    beta                 |  99.35     99.51   |    0.25      0.16   |   99.59     99.69  
    empirical            |  98.51     99.33   |    1.05      0.37   |   99.04     99.57  
    empirical_cumulative |  99.37     99.54   |    0.19      0.10   |   99.61     99.71  
    val_cal              |  99.37     99.54   |    0.19      0.10   |   99.61     99.71  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 270±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.36     99.49   |    0.27      0.11   |   99.59     99.68  
    fixed_05             |  99.07     99.12   |    0.56      0.55   |   99.41     99.44  
    platt                |  99.33     99.31   |    0.30      0.37   |   99.57     99.56  
    beta                 |  99.31     99.42   |    0.21      0.22   |   99.57     99.64  
    empirical            |  99.33     99.46   |    0.25      0.10   |   99.58     99.66  
    empirical_cumulative |  99.34     99.49   |    0.21      0.11   |   99.58     99.68  
    val_cal              |  99.36     99.49   |    0.27      0.11   |   99.59     99.68  


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


## XDS-cicids-96b-Wb  (1 flows × 2 phases, seeds: [82096])

    Weight : Wb (paper/PUB50, ce=0.10 acc=0.20)

### Best individual genomes

    Metric                   |      F1 |     FPR |     Acc | Source
    -------------------------+---------+---------+---------+-----------------------------------
    Best F1 (any FPR)        |  99.52% |   0.12% |  99.70% | r82096 GA best_ce        train_cal
    Best F1 (FPR<14%)        |  99.52% |   0.12% |  99.70% | r82096 GA best_ce        train_cal
    Best F1 (FPR<10%)        |  99.52% |   0.12% |  99.70% | r82096 GA best_ce        train_cal
    Best F1 (FPR<6%)         |  99.52% |   0.12% |  99.70% | r82096 GA best_ce        train_cal
    Best F1 (FPR<5%)         |  99.52% |   0.12% |  99.70% | r82096 GA best_ce        train_cal
    Best F1 (FPR<4%)         |  99.52% |   0.12% |  99.70% | r82096 GA best_ce        train_cal
    Best FPR (any F1)        |  99.52% |   0.12% |  99.70% | r82096 GA best_acc       val_cal
    Best FPR (F1>80%)        |  99.52% |   0.12% |  99.70% | r82096 GA best_acc       val_cal
    Best Acc (any FPR)       |  99.52% |   0.12% |  99.70% | r82096 GA best_ce        train_cal

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 100±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.52   |    0.36      0.12   |   99.54     99.70  
    fixed_05             |  99.01     99.25   |    0.60      0.45   |   99.37     99.52  
    platt                |  99.24     99.40   |    0.32      0.26   |   99.52     99.62  
    beta                 |  99.23     99.46   |    0.29      0.20   |   99.51     99.66  
    empirical            |  99.26     99.40   |    0.35      0.27   |   99.53     99.62  
    empirical_cumulative |  99.22     99.52   |    0.23      0.12   |   99.51     99.70  
    val_cal              |  99.27     99.52   |    0.37      0.12   |   99.54     99.70  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 100±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.52   |    0.36      0.12   |   99.54     99.70  
    fixed_05             |  99.01     99.25   |    0.60      0.45   |   99.37     99.52  
    platt                |  99.24     99.40   |    0.32      0.26   |   99.52     99.62  
    beta                 |  99.23     99.46   |    0.29      0.20   |   99.51     99.66  
    empirical            |  99.26     99.40   |    0.35      0.27   |   99.53     99.62  
    empirical_cumulative |  99.22     99.52   |    0.23      0.12   |   99.51     99.70  
    val_cal              |  99.27     99.52   |    0.37      0.12   |   99.54     99.70  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 400±0 neurons | 28±0 bits
    GA Neurons  : 107±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.10     99.36   |    0.34      0.23   |   99.43     99.60  
    fixed_05             |  98.91     99.16   |    0.66      0.50   |   99.30     99.46  
    platt                |  99.08     99.34   |    0.37      0.30   |   99.42     99.58  
    beta                 |  99.05     99.35   |    0.30      0.27   |   99.40     99.59  
    empirical            |  98.99     99.35   |    0.56      0.27   |   99.36     99.59  
    empirical_cumulative |  98.98     99.35   |    0.12      0.22   |   99.36     99.59  
    val_cal              |  99.10     99.36   |    0.34      0.23   |   99.43     99.60  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 500±0 neurons | 34±0 bits
    GA Neurons  : 100±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.27     99.52   |    0.36      0.12   |   99.54     99.70  
    fixed_05             |  99.01     99.25   |    0.60      0.45   |   99.37     99.52  
    platt                |  99.24     99.40   |    0.32      0.26   |   99.52     99.62  
    beta                 |  99.23     99.46   |    0.29      0.20   |   99.51     99.66  
    empirical            |  99.26     99.40   |    0.35      0.27   |   99.53     99.62  
    empirical_cumulative |  99.22     99.52   |    0.23      0.12   |   99.51     99.70  
    val_cal              |  99.27     99.52   |    0.37      0.12   |   99.54     99.70  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 100±0 neurons | 34±0 bits
    GA Neurons  : 94±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.22     99.52   |    0.29      0.12   |   99.51     99.70  
    fixed_05             |  98.94     99.05   |    0.65      0.60   |   99.32     99.39  
    platt                |  99.17     99.35   |    0.35      0.32   |   99.48     99.59  
    beta                 |  99.22     99.40   |    0.29      0.25   |   99.51     99.62  
    empirical            |  99.21     99.52   |    0.40      0.12   |   99.50     99.70  
    empirical_cumulative |  99.20     99.52   |    0.23      0.12   |   99.49     99.70  
    val_cal              |  99.22     99.52   |    0.29      0.12   |   99.51     99.70  


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


## XDS-cicids-96b-Wc  (1 flows × 2 phases, seeds: [82096])

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

### best_fitness  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 97±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.24     99.37   |    0.25      0.22   |   99.52     99.60  
    fixed_05             |  98.98     99.17   |    0.63      0.51   |   99.35     99.47  
    platt                |  99.21     99.36   |    0.34      0.32   |   99.50     99.60  
    beta                 |  99.23     99.35   |    0.25      0.26   |   99.52     99.59  
    empirical            |  99.24     99.36   |    0.35      0.32   |   99.52     99.60  
    empirical_cumulative |  99.24     99.37   |    0.25      0.22   |   99.52     99.60  
    val_cal              |  99.24     99.37   |    0.38      0.22   |   99.52     99.60  

### best_f1  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 205±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.30     99.56   |    0.33      0.07   |   99.56     99.73  
    fixed_05             |  99.03     99.12   |    0.58      0.54   |   99.39     99.44  
    platt                |  99.28     99.45   |    0.32      0.23   |   99.54     99.65  
    beta                 |  99.25     99.46   |    0.29      0.20   |   99.53     99.66  
    empirical            |  99.30     99.56   |    0.33      0.07   |   99.55     99.73  
    empirical_cumulative |  99.27     99.56   |    0.23      0.07   |   99.54     99.73  
    val_cal              |  99.30     99.56   |    0.33      0.07   |   99.56     99.73  

### best_fpr  (GS: 1 runs | GA: 1 runs)
    Grid Search : 5±0 neurons | 4±0 bits
    GA Neurons  : 206±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  74.71     99.56   |    8.15      0.07   |   84.70     99.72  
    fixed_05             |  74.71     99.12   |    8.15      0.54   |   84.70     99.44  
    platt                |  71.21     99.46   |    0.94      0.23   |   86.43     99.66  
    beta                 |  71.49     99.46   |    0.16      0.20   |   86.89     99.66  
    empirical            |  71.49     99.56   |    0.16      0.07   |   86.89     99.72  
    empirical_cumulative |  71.49     99.56   |    0.16      0.07   |   86.89     99.72  
    val_cal              |  74.71     99.56   |    8.15      0.07   |   84.70     99.72  

### best_acc  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 205±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.30     99.56   |    0.33      0.07   |   99.56     99.73  
    fixed_05             |  99.03     99.12   |    0.58      0.54   |   99.39     99.44  
    platt                |  99.28     99.45   |    0.32      0.23   |   99.54     99.65  
    beta                 |  99.25     99.46   |    0.29      0.20   |   99.53     99.66  
    empirical            |  99.30     99.56   |    0.33      0.07   |   99.55     99.73  
    empirical_cumulative |  99.27     99.56   |    0.23      0.07   |   99.54     99.73  
    val_cal              |  99.30     99.56   |    0.33      0.07   |   99.56     99.73  

### best_ce  (GS: 1 runs | GA: 1 runs)
    Grid Search : 200±0 neurons | 34±0 bits
    GA Neurons  : 97±0 neurons | 34±0 bits

    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA
    ---------------------+---------------------+---------------------+--------------------
    train_cal            |  99.24     99.37   |    0.25      0.22   |   99.52     99.60  
    fixed_05             |  98.98     99.17   |    0.63      0.51   |   99.35     99.47  
    platt                |  99.21     99.36   |    0.34      0.32   |   99.50     99.60  
    beta                 |  99.23     99.35   |    0.25      0.26   |   99.52     99.59  
    empirical            |  99.24     99.36   |    0.35      0.32   |   99.52     99.60  
    empirical_cumulative |  99.24     99.37   |    0.25      0.22   |   99.52     99.60  
    val_cal              |  99.24     99.37   |    0.38      0.22   |   99.52     99.60  

