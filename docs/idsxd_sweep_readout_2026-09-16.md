# IDSXD desirability fitness-weight sweep — readout 16/09/2026

Generated 16/09/2026 20:52 UTC from `file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro` (read-only; no flow created, cancelled, requeued or modified).
Every number is HELD-OUT: `validation_summaries.threshold_metadata` at `validation_point='final'`, val_cal mode unless stated.
Protocol v2 (`random_3way`): thresholds calibrated on the 10% VAL partition, the 10% TEST partition is report-only. No `iterations.best_f1` anywhere.
Arm weights are (w_ce / w_acc / w_f1 / w_fpr), `fitness_aggregation=desirability`, CE anchor normalized 0.2128. `unswt` (temporal) is queued with 0 completed and is skipped.

```
Runs      : 106/108 completed | running: 1 | queued: 1   (unswt excluded)
Total wall: 467.3 h over 106 completed runs | avg/run 4.41 h
Latest done: 16/09/2026 19:31 UTC

dataset           | arms | done | total wall h | avg h/run |  min h |  max h | latest done (UTC)
----------------------------------------------------------------------------------------------------
cicids quad 96b   |    8 |   24 |         37.7 |      1.57 |   0.87 |   3.26 | 29/08/2026 09:22
ciciot quad 96b   |    8 |   24 |         95.6 |      3.98 |   1.65 |   7.01 | 03/09/2026 08:03
unswr qsr 64b     |    9 |   29 |        308.9 |     10.65 |   2.72 |  27.33 | 15/09/2026 18:30
unswr quad 64b    |    9 |   29 |         25.0 |      0.86 |   0.38 |   1.71 | 16/09/2026 19:31
  not counted: IDSXD-ciciot-quad-96b-B05-AC-r20405-w64fix status=running started=2026-09-16T19:31:54.417533+00:00
  not counted: IDSXD-ciciot-quad-96b-B05-CE-r20404-w64fix status=queued started=—
```

## cicids quad 96b — CICIDS2017 random_3way, decode=QUAD, input 96 bits, caps 500n/34b

```
Arm weights (w_ce/w_acc/w_f1/w_fpr):
  B05-AC    0.225/0.675/0.05/0.05   seeds: 20403,20404,20405
  B05-CE    0.675/0.225/0.05/0.05   seeds: 20403,20404,20405
  B10-AC    0.2/0.6/0.1/0.1   seeds: 20403,20404,20405
  B10-CE    0.6/0.2/0.1/0.1   seeds: 20403,20404,20405
  B15-AC    0.175/0.525/0.15/0.15   seeds: 20403,20404,20405
  B15-CE    0.525/0.175/0.15/0.15   seeds: 20403,20404,20405
  CE20      0.2/0.1/0.3/0.4   seeds: 20403,20404,20405
  Wa-CTRL   0.35/0.3/0.3/0.05   seeds: 20403,20404,20405
```

### Table 1 — arm comparison, GA Neurons phase (final), val_cal threshold, held-out TEST, mean±SD %

```
arm      | n |  best_f1 F1  | best_f1 FPR | best_f1 Acc || best_fpr F1 | best_fpr FPR || best_fit F1 | best_fit FPR ||    gens     |   wall h    | winner n x bits (best_f1)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CE20     | 3 | 99.52±0.03  |  0.12±0.01  | 99.70±0.02  || 99.48±0.02  |  0.11±0.01  || 99.51±0.03  |  0.12±0.00  ||  80.0±10.0  |  1.25±0.31  | 127n x 34.0b
B15-AC   | 3 | 99.48±0.06  |  0.15±0.05  | 99.67±0.04  || 99.45±0.03  |  0.17±0.07  || 99.47±0.05  |  0.16±0.05  || 106.7±25.2  |  1.83±0.95  | 125n x 34.0b
Wa-CTRL  | 3 | 99.47±0.09  |  0.21±0.08  | 99.67±0.06  || 99.47±0.09  |  0.18±0.07  || 99.48±0.09  |  0.18±0.07  || 116.7±15.3  |  1.77±0.75  | 126n x 33.8b
B10-AC   | 3 | 99.47±0.07  |  0.21±0.08  | 99.66±0.04  || 99.45±0.07  |  0.19±0.09  || 99.45±0.08  |  0.21±0.10  ||  80.0±17.3  |  1.32±0.14  | 130n x 34.0b
B10-CE   | 3 | 99.45±0.06  |  0.17±0.11  | 99.65±0.04  || 99.41±0.09  |  0.19±0.08  || 99.43±0.03  |  0.17±0.09  ||  83.3±20.8  |  1.24±0.21  | 103n x 34.0b
B15-CE   | 3 | 99.44±0.08  |  0.23±0.07  | 99.64±0.05  || 99.39±0.12  |  0.21±0.07  || 99.41±0.11  |  0.23±0.08  ||  73.3±15.3  |  1.07±0.31  | 99n x 34.0b
B05-AC   | 3 | 99.43±0.11  |  0.23±0.10  | 99.64±0.07  || 99.39±0.07  |  0.26±0.07  || 99.42±0.11  |  0.25±0.11  ||  90.0±43.6  |  1.89±0.74  | 242n x 34.0b
B05-CE   | 3 | 99.42±0.10  |  0.25±0.10  | 99.64±0.06  || 99.41±0.08  |  0.24±0.11  || 99.41±0.08  |  0.23±0.09  ||  93.3±5.8   |  2.20±1.01  | 212n x 34.0b
```

### Per-seed matrix — GA best_f1 val_cal: F1 / FPR (held-out %), paired by seed

```
arm      |      r20403       |      r20404       |      r20405       
---------------------------------------------------------------------
CE20     |   99.56 /  0.11   |   99.50 /  0.12   |   99.50 /  0.12   
B15-AC   |   99.43 /  0.21   |   99.54 /  0.10   |   99.48 /  0.16   
Wa-CTRL  |   99.50 /  0.15   |   99.37 /  0.30   |   99.55 /  0.17   
B10-AC   |   99.51 /  0.17   |   99.39 /  0.30   |   99.50 /  0.15   
B10-CE   |   99.52 /  0.16   |   99.40 /  0.28   |   99.45 /  0.07   
B15-CE   |   99.53 /  0.15   |   99.38 /  0.26   |   99.40 /  0.28   
B05-AC   |   99.55 /  0.12   |   99.40 /  0.32   |   99.34 /  0.26   
B05-CE   |   99.53 /  0.14   |   99.37 /  0.31   |   99.36 /  0.30   
```

### Best individual genomes — all 1680 (run x phase x genome_type x mode) points

```
criterion          |   F1    |   FPR   |   Acc   | source (arm, seed, phase, genome_type, mode)
----------------------------------------------------------------------------------------------------
Best F1 (any FPR)  |  99.56% |   0.11% |  99.72% | CE20 r20403 GA best_f1 train_cal
Best F1 (FPR<5%)   |  99.56% |   0.11% |  99.72% | CE20 r20403 GA best_f1 train_cal
Best F1 (FPR<2%)   |  99.56% |   0.11% |  99.72% | CE20 r20403 GA best_f1 train_cal
Best F1 (FPR<1%)   |  99.56% |   0.11% |  99.72% | CE20 r20403 GA best_f1 train_cal
Best FPR (F1>80%)  |  97.97% |   0.04% |  98.74% | B05-CE r20404 GS best_fpr empirical
Best FPR (F1>90%)  |  97.97% |   0.04% |  98.74% | B05-CE r20404 GS best_fpr empirical
Best Acc (any FPR) |  99.56% |   0.11% |  99.72% | CE20 r20403 GA best_f1 train_cal
```

## ciciot quad 96b — CIC-IoT-2023 neto_subsample random_3way, decode=QUAD, input 96 bits, caps 250n/100b

```
Arm weights (w_ce/w_acc/w_f1/w_fpr):
  B05-AC    0.225/0.675/0.05/0.05   seeds: 20403,20404,20405
  B05-CE    0.675/0.225/0.05/0.05   seeds: 20403,20404,20405
  B10-AC    0.2/0.6/0.1/0.1   seeds: 20403,20404,20405
  B10-CE    0.6/0.2/0.1/0.1   seeds: 20403,20404,20405
  B15-AC    0.175/0.525/0.15/0.15   seeds: 20403,20404,20405
  B15-CE    0.525/0.175/0.15/0.15   seeds: 20403,20404,20405
  CE20      0.2/0.1/0.3/0.4   seeds: 20403,20404,20405
  Wc-CTRL   0.7/0.1/0.15/0.05   seeds: 20403,20404,20405
```

### Table 1 — arm comparison, GA Neurons phase (final), val_cal threshold, held-out TEST, mean±SD %

```
arm      | n |  best_f1 F1  | best_f1 FPR | best_f1 Acc || best_fpr F1 | best_fpr FPR || best_fit F1 | best_fit FPR ||    gens     |   wall h    | winner n x bits (best_f1)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
B15-AC   | 3 | 92.98±0.08  |  6.79±0.76  | 96.46±0.07  || 92.96±0.07  |  7.88±0.47  || 92.99±0.05  |  7.49±0.85  || 150.0±26.5  |  4.90±0.52  | 198n x 80.0b
B10-AC   | 3 | 92.97±0.05  |  7.14±1.19  | 96.46±0.02  || 92.96±0.05  |  7.19±1.18  || 92.96±0.06  |  6.96±1.07  || 146.7±5.8   |  3.14±1.51  | 227n x 69.3b
B05-AC   | 3 | 92.96±0.18  |  7.91±0.51  | 96.48±0.08  || 92.96±0.16  |  7.92±0.40  || 92.94±0.16  |  8.06±0.48  || 150.0±26.5  |  1.82±0.26  | 168n x 64.0b
B10-CE   | 3 | 92.91±0.06  |  6.75±0.35  | 96.42±0.04  || 92.90±0.07  |  7.03±1.03  || 92.92±0.08  |  6.98±0.87  || 156.7±25.2  |  5.59±1.56  | 214n x 80.0b
Wc-CTRL  | 3 | 92.85±0.05  |  7.93±0.13  | 96.42±0.03  || 92.87±0.02  |  7.73±0.23  || 92.87±0.02  |  7.97±0.26  || 140.0±10.0  |  5.22±0.39  | 218n x 80.0b
B05-CE   | 3 | 92.80±0.10  |  7.83±0.15  | 96.39±0.05  || 92.79±0.10  |  7.94±0.81  || 92.80±0.12  |  7.80±0.56  || 136.7±15.3  |  2.15±0.26  | 242n x 64.0b
CE20     | 3 | 92.71±0.27  |  7.90±1.42  | 96.34±0.11  || 92.68±0.28  |  7.80±1.00  || 92.70±0.29  |  7.85±1.34  || 136.7±30.6  |  3.90±2.02  | 149n x 80.0b
B15-CE   | 3 | 92.70±0.14  |  7.52±1.02  | 96.33±0.05  || 92.71±0.17  |  7.72±0.86  || 92.73±0.11  |  7.06±0.79  || 136.7±5.8   |  5.15±1.15  | 234n x 74.7b
```

### Per-seed matrix — GA best_f1 val_cal: F1 / FPR (held-out %), paired by seed   [pre/post = bits>64 OR-fold fix, cut 30/08/2026 02:27 UTC]

```
arm      |         r20403          |         r20404          |         r20405          
---------------------------------------------------------------------------------------
B15-AC   |    93.05 /  7.55 post   |    92.90 /  6.04 post   |    93.00 /  6.79 post   
B10-AC   |    92.91 /  7.78 PRE    |    92.98 /  7.87 PRE    |    93.01 /  5.77 post   
B05-AC   |    92.79 /  8.49 PRE    |    92.96 /  7.63 PRE    |    93.14 /  7.59 PRE    
B10-CE   |    92.84 /  6.58 post   |    92.92 /  7.16 post   |    92.97 /  6.53 post   
Wc-CTRL  |    92.79 /  7.93 post   |    92.88 /  8.06 post   |    92.88 /  7.81 post   
B05-CE   |    92.69 /  8.00 PRE    |    92.81 /  7.70 PRE    |    92.90 /  7.79 PRE    
CE20     |    92.84 /  6.88 post   |    92.89 /  7.29 post   |    92.39 /  9.52 post   
B15-CE   |    92.86 /  6.42 post   |    92.65 /  7.72 post   |    92.60 /  8.43 post   
```

Era per run (completed_at UTC): B05-AC r20403=29/08 11:03 PRE; B05-AC r20404=29/08 13:11 PRE; B05-AC r20405=29/08 14:50 PRE; B05-CE r20403=29/08 16:42 PRE; B05-CE r20404=29/08 19:05 PRE; B05-CE r20405=29/08 21:18 PRE; B10-AC r20403=29/08 23:38 PRE; B10-AC r20404=30/08 01:50 PRE; B10-AC r20405=30/08 10:41 post; B10-CE r20403=30/08 17:42 post; B10-CE r20404=31/08 17:46 post; B10-CE r20405=31/08 22:32 post; B15-AC r20403=01/09 04:02 post; B15-AC r20404=01/09 08:34 post; B15-AC r20405=01/09 13:15 post; B15-CE r20403=01/09 19:07 post; B15-CE r20404=01/09 22:56 post; B15-CE r20405=02/09 04:41 post; CE20 r20403=02/09 08:31 post; CE20 r20404=02/09 14:28 post; CE20 r20405=02/09 16:23 post; Wc-CTRL r20403=02/09 21:09 post; Wc-CTRL r20404=03/09 02:40 post; Wc-CTRL r20405=03/09 08:03 post

### Best individual genomes — all 1680 (run x phase x genome_type x mode) points

```
criterion          |   F1    |   FPR   |   Acc   | source (arm, seed, phase, genome_type, mode)
----------------------------------------------------------------------------------------------------
Best F1 (any FPR)  |  93.18% |   6.91% |  96.57% | B05-AC r20405 GA best_f1 train_cal
Best F1 (FPR<5%)   |  92.90% |   4.84% |  96.35% | B10-AC r20405 GA best_f1 empirical_cumulative
Best F1 (FPR<2%)   |  88.33% |   0.81% |  93.35% | B05-CE r20405 GA best_fitness fixed_05
Best F1 (FPR<1%)   |  88.33% |   0.81% |  93.35% | B05-CE r20405 GA best_fitness fixed_05
Best FPR (F1>80%)  |  84.41% |   0.62% |  90.54% | CE20 r20405 GA best_fpr fixed_05
Best FPR (F1>90%)  |  91.30% |   2.71% |  95.33% | CE20 r20404 GA best_f1 empirical_cumulative
Best Acc (any FPR) |  93.14% |   7.59% |  96.57% | B05-AC r20405 GA best_f1 val_cal
```

## unswr qsr 64b — UNSW-NB15 random_3way, decode=QSR, input 64 bits, caps 500n/34b

```
Arm weights (w_ce/w_acc/w_f1/w_fpr):
  B05-AC    0.225/0.675/0.05/0.05   seeds: 20403,20404,20405
  B05-CE    0.675/0.225/0.05/0.05   seeds: 20403,20404,20405
  B10-AC    0.2/0.6/0.1/0.1   seeds: 20403,20404,20405
  B10-CE    0.6/0.2/0.1/0.1   seeds: 20403,20404,20405
  B15-AC    0.175/0.525/0.15/0.15   seeds: 20403,20404,20405
  B15-CE    0.525/0.175/0.15/0.15   seeds: 20403,20404,20405
  B34-CTRL  0.1/0.2/0.35/0.35   seeds: 20401,20402,20403,20404,20405
  CE20      0.2/0.1/0.3/0.4   seeds: 20403,20404,20405
  Wb-CTRL   0.1/0.2/0.35/0.35   seeds: 20403,20404,20405
```

### Table 1 — arm comparison, GA Neurons phase (final), val_cal threshold, held-out TEST, mean±SD %

```
arm      | n |  best_f1 F1  | best_f1 FPR | best_f1 Acc || best_fpr F1 | best_fpr FPR || best_fit F1 | best_fit FPR ||    gens     |   wall h    | winner n x bits (best_f1)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
B34-CTRL | 5 | 94.38±0.06  |  0.59±0.03  | 99.15±0.01  || 94.33±0.12  |  0.65±0.06  || 94.35±0.07  |  0.61±0.04  ||  68.0±17.9  |  7.49±4.35  | 367n x 34.0b
B15-CE   | 3 | 94.36±0.13  |  0.65±0.04  | 99.13±0.01  || 94.38±0.07  |  0.68±0.03  || 94.37±0.14  |  0.59±0.02  ||  70.0±10.0  | 13.87±1.85  | 428n x 34.0b
B15-AC   | 3 | 94.35±0.08  |  0.61±0.05  | 99.14±0.01  || 94.35±0.10  |  0.60±0.05  || 94.33±0.07  |  0.62±0.07  ||  80.0±20.0  | 15.71±3.89  | 467n x 34.0b
B05-AC   | 3 | 94.34±0.12  |  0.59±0.06  | 99.14±0.03  || 94.34±0.14  |  0.60±0.03  || 94.30±0.17  |  0.59±0.04  ||  93.3±5.8   |  8.85±8.16  | 290n x 33.9b
B05-CE   | 3 | 94.32±0.01  |  0.53±0.09  | 99.15±0.02  || 94.37±0.04  |  0.56±0.07  || 94.36±0.03  |  0.56±0.09  ||  93.3±20.8  | 10.96±4.47  | 361n x 33.9b
Wb-CTRL  | 3 | 94.32±0.05  |  0.61±0.06  | 99.14±0.01  || 94.30±0.06  |  0.61±0.04  || 94.30±0.07  |  0.60±0.05  || 100.0±26.5  |  4.24±1.46  | 233n x 33.3b
B10-CE   | 3 | 94.31±0.05  |  0.58±0.07  | 99.14±0.02  || 94.28±0.06  |  0.58±0.07  || 94.29±0.09  |  0.58±0.07  ||  96.7±40.4  | 14.83±11.35 | 371n x 33.5b
B10-AC   | 3 | 94.29±0.07  |  0.59±0.08  | 99.14±0.01  || 94.36±0.11  |  0.58±0.08  || 94.33±0.11  |  0.56±0.06  ||  86.7±5.8   |  8.93±8.16  | 369n x 34.0b
CE20     | 3 | 94.25±0.07  |  0.59±0.10  | 99.13±0.02  || 94.27±0.05  |  0.60±0.07  || 94.26±0.05  |  0.60±0.10  ||  63.3±5.8   | 13.09±0.99  | 480n x 34.0b
```

### Per-seed matrix — GA best_f1 val_cal: F1 / FPR (held-out %), paired by seed

```
arm      |      r20401       |      r20402       |      r20403       |      r20404       |      r20405       
-------------------------------------------------------------------------------------------------------------
B34-CTRL |   94.41 /  0.61   |   94.43 /  0.60   |   94.27 /  0.61   |   94.41 /  0.58   |   94.35 /  0.54   
B15-CE   |         —         |         —         |   94.49 /  0.69   |   94.23 /  0.62   |   94.36 /  0.65   
B15-AC   |         —         |         —         |   94.31 /  0.55   |   94.29 /  0.64   |   94.44 /  0.64   
B05-AC   |         —         |         —         |   94.35 /  0.58   |   94.21 /  0.65   |   94.45 /  0.53   
B05-CE   |         —         |         —         |   94.32 /  0.63   |   94.33 /  0.45   |   94.31 /  0.51   
Wb-CTRL  |         —         |         —         |   94.37 /  0.65   |   94.30 /  0.63   |   94.29 /  0.55   
B10-CE   |         —         |         —         |   94.30 /  0.67   |   94.36 /  0.55   |   94.25 /  0.54   
B10-AC   |         —         |         —         |   94.35 /  0.62   |   94.22 /  0.50   |   94.32 /  0.65   
CE20     |         —         |         —         |   94.33 /  0.58   |   94.22 /  0.50   |   94.20 /  0.70   
```

### Best individual genomes — all 2030 (run x phase x genome_type x mode) points

```
criterion          |   F1    |   FPR   |   Acc   | source (arm, seed, phase, genome_type, mode)
----------------------------------------------------------------------------------------------------
Best F1 (any FPR)  |  94.53% |   0.60% |  99.17% | B15-CE r20403 GA best_fitness val_cal
Best F1 (FPR<5%)   |  94.53% |   0.60% |  99.17% | B15-CE r20403 GA best_fitness val_cal
Best F1 (FPR<2%)   |  94.53% |   0.60% |  99.17% | B15-CE r20403 GA best_fitness val_cal
Best F1 (FPR<1%)   |  94.53% |   0.60% |  99.17% | B15-CE r20403 GA best_fitness val_cal
Best FPR (F1>80%)  |  80.59% |   0.01% |  97.91% | Wb-CTRL r20404 GS best_ce empirical
Best FPR (F1>90%)  |  93.95% |   0.37% |  99.13% | B05-CE r20403 GS best_ce empirical_cumulative
Best Acc (any FPR) |  94.49% |   0.46% |  99.19% | B15-CE r20403 GA best_fitness platt
```

## unswr quad 64b — UNSW-NB15 random_3way, decode=QUAD, input 64 bits, caps 500n/34b

```
Arm weights (w_ce/w_acc/w_f1/w_fpr):
  B05-AC    0.225/0.675/0.05/0.05   seeds: 20403,20404,20405
  B05-CE    0.675/0.225/0.05/0.05   seeds: 20403,20404,20405
  B10-AC    0.2/0.6/0.1/0.1   seeds: 20403,20404,20405
  B10-CE    0.6/0.2/0.1/0.1   seeds: 20403,20404,20405
  B15-AC    0.175/0.525/0.15/0.15   seeds: 20403,20404,20405
  B15-CE    0.525/0.175/0.15/0.15   seeds: 20403,20404,20405
  B34-CTRL  0.1/0.2/0.35/0.35   seeds: 20401,20402,20403,20404,20405
  CE20      0.2/0.1/0.3/0.4   seeds: 20403,20404,20405
  Wb-CTRL   0.1/0.2/0.35/0.35   seeds: 20403,20404,20405
```

### Table 1 — arm comparison, GA Neurons phase (final), val_cal threshold, held-out TEST, mean±SD %

```
arm      | n |  best_f1 F1  | best_f1 FPR | best_f1 Acc || best_fpr F1 | best_fpr FPR || best_fit F1 | best_fit FPR ||    gens     |   wall h    | winner n x bits (best_f1)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
B15-CE   | 3 | 93.49±0.01  |  1.12±0.00  | 98.92±0.00  || 93.48±0.03  |  1.12±0.00  || 93.49±0.01  |  1.12±0.00  ||  60.0±0.0   |  0.64±0.02  | 404n x 13.5b
B34-CTRL | 5 | 93.49±0.01  |  1.12±0.00  | 98.92±0.00  || 93.47±0.04  |  1.12±0.00  || 93.48±0.01  |  1.12±0.00  ||  60.0±0.0   |  1.42±0.04  | 481n x 29.4b
B10-CE   | 3 | 93.48±0.02  |  1.12±0.00  | 98.92±0.00  || 93.46±0.03  |  1.12±0.00  || 93.48±0.01  |  1.12±0.00  ||  60.0±0.0   |  0.52±0.12  | 330n x 12.0b
B15-AC   | 3 | 93.48±0.02  |  1.12±0.00  | 98.92±0.00  || 93.47±0.02  |  1.12±0.00  || 93.48±0.01  |  1.12±0.00  ||  60.0±0.0   |  0.65±0.08  | 395n x 14.2b
B05-CE   | 3 | 93.47±0.02  |  1.12±0.00  | 98.92±0.00  || 93.48±0.01  |  1.12±0.00  || 93.49±0.00  |  1.12±0.00  ||  80.0±17.3  |  0.67±0.31  | 411n x 12.3b
B10-AC   | 3 | 93.47±0.02  |  1.02±0.16  | 98.93±0.03  || 93.45±0.03  |  1.12±0.00  || 93.47±0.02  |  1.02±0.18  ||  70.0±17.3  |  0.72±0.24  | 355n x 13.6b
B05-AC   | 3 | 93.45±0.05  |  1.12±0.00  | 98.91±0.01  || 93.45±0.03  |  1.12±0.01  || 93.45±0.04  |  1.12±0.00  ||  70.0±17.3  |  0.74±0.22  | 349n x 12.4b
Wb-CTRL  | 3 | 93.44±0.04  |  0.99±0.23  | 98.94±0.04  || 93.41±0.09  |  0.98±0.24  || 93.44±0.04  |  0.99±0.23  || 103.3±40.4  |  1.07±0.42  | 392n x 12.9b
CE20     | 3 | 93.41±0.07  |  0.98±0.23  | 98.93±0.04  || 93.44±0.03  |  1.12±0.00  || 93.45±0.03  |  1.12±0.00  || 110.0±34.6  |  0.96±0.65  | 385n x 12.7b
```

### Per-seed matrix — GA best_f1 val_cal: F1 / FPR (held-out %), paired by seed

```
arm      |      r20401       |      r20402       |      r20403       |      r20404       |      r20405       
-------------------------------------------------------------------------------------------------------------
B15-CE   |         —         |         —         |   93.50 /  1.12   |   93.49 /  1.12   |   93.50 /  1.12   
B34-CTRL |   93.49 /  1.12   |   93.46 /  1.12   |   93.48 /  1.12   |   93.50 /  1.12   |   93.50 /  1.12   
B10-CE   |         —         |         —         |   93.48 /  1.12   |   93.46 /  1.12   |   93.50 /  1.12   
B15-AC   |         —         |         —         |   93.45 /  1.12   |   93.49 /  1.12   |   93.49 /  1.12   
B05-CE   |         —         |         —         |   93.48 /  1.12   |   93.45 /  1.12   |   93.49 /  1.12   
B10-AC   |         —         |         —         |   93.45 /  0.83   |   93.47 /  1.12   |   93.49 /  1.12   
B05-AC   |         —         |         —         |   93.45 /  1.12   |   93.50 /  1.12   |   93.40 /  1.12   
Wb-CTRL  |         —         |         —         |   93.41 /  0.73   |   93.49 /  1.12   |   93.43 /  1.12   
CE20     |         —         |         —         |   93.34 /  0.72   |   93.42 /  1.12   |   93.48 /  1.12   
```

### Best individual genomes — all 2030 (run x phase x genome_type x mode) points

```
criterion          |   F1    |   FPR   |   Acc   | source (arm, seed, phase, genome_type, mode)
----------------------------------------------------------------------------------------------------
Best F1 (any FPR)  |  94.27% |   0.52% |  99.15% | B34-CTRL r20402 GS best_ce beta
Best F1 (FPR<5%)   |  94.27% |   0.52% |  99.15% | B34-CTRL r20402 GS best_ce beta
Best F1 (FPR<2%)   |  94.27% |   0.52% |  99.15% | B34-CTRL r20402 GS best_ce beta
Best F1 (FPR<1%)   |  94.27% |   0.52% |  99.15% | B34-CTRL r20402 GS best_ce beta
Best FPR (F1>80%)  |  90.55% |   0.33% |  98.73% | B15-CE r20403 GA best_f1 empirical
Best FPR (F1>90%)  |  90.55% |   0.33% |  98.73% | B15-CE r20403 GA best_f1 empirical
Best Acc (any FPR) |  94.22% |   0.46% |  99.15% | B34-CTRL r20402 GS best_ce empirical_cumulative
```


## Notes for the reader (observations only — no winner is called here; adjudication belongs to experiment-design)

- **Metric provenance.** GA phase = the final `ga_neurons` experiment of the 2-phase `ids-binary-2-phase` template; "gens" = `experiments.current_iteration` of that phase; wall h = `flows.completed_at - flows.started_at` (includes the grid phase). Winner n x bits = `best_genomes` (metric `f1_macro`, mode `val_cal`, GA phase) joined to `genomes.tiers_json.bits_per_neuron`, mean over neurons; a non-integer bits figure means a heterogeneous-bits genome.
- **B34-CTRL vs Wb-CTRL (unswr, both decodes) have IDENTICAL weights (0.1/0.2/0.35/0.35)** and differ ONLY in `min_bits` (34 vs 4). They are a bits-floor control pair, not two weight arms. B34-CTRL has n=5 (seeds 20401-20405), every other arm n=3 (20403-20405).
- **CIC-IoT code era.** The bits>64 OR-fold fix landed 30/08/2026 02:27 UTC. On ciciot (caps 250n/100b, so bits>64 IS reachable): B05-AC (3/3) and B05-CE (3/3) completed PRE-fix, B10-AC is 2 PRE + 1 post (r20405 post), all other arms post. The two `-w64fix` re-runs (B05-AC r20405 running since 16/09 19:31 UTC, B05-CE r20404 queued) are not counted. The `-w64fix` re-run params are byte-identical to the originals (verified: empty param diff). cicids (caps 34b) and unswr (caps 34b) never exceed 64 bits and are unaffected regardless of completion date.
- **unswr quad 64b FPR is pinned at 1.12% on 25/29 runs** (the known QUAD-decode FPR floor on UNSW random; only QSR is tunable there). The few 0.72-0.83 FPR points are seed 20403 on B10-AC / Wb-CTRL / CE20. 6 of 9 arms stopped at exactly 60 GA generations. The best individual genome on this dataset comes from the GRID phase (B34-CTRL r20402 GS best_ce beta, 94.27/0.52), above every GA final (<= 93.50).
- **min_bits floor not honoured by the GA on unswr quad B34-CTRL:** GA winners carry 59-116 neurons at 8 bits alongside 34-bit neurons (e.g. r20401: 116 x 8b + 359 x 34b; r20403: 59 x 8b + 425 x 34b), although `min_bits=max_bits=34`. Grid-phase winners are 100% 34b. On unswr qsr the same arm stays at 100% 34b. Not diagnosed here; flagged for whoever owns the GA neuron-add operator.
- **Wall-clock.** unswr qsr is the expensive dataset (10.65 h/run avg, 2.7-27.3 h) — QSR decode cost; unswr quad is 0.86 h/run. ciciot 3.98 h/run, cicids 1.57 h/run.
- **Effect sizes vs spread.** On cicids the arm means span 99.42-99.52 F1 (0.10pp) with per-arm SD 0.03-0.11; on ciciot 92.70-92.98 (0.28pp) with SD 0.05-0.27; on unswr qsr 94.25-94.38 (0.13pp) with SD 0.01-0.13; on unswr quad 93.41-93.49 (0.08pp) with SD 0.01-0.07. Per-seed columns above are the paired data; seed 20404 is the low seed on cicids for 7 of 8 arms.
