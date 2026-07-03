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

