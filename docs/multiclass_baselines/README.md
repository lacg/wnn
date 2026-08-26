# RF / XGBoost multiclass baselines — the bar for the MCS arms

**Generated** by `scripts/build_multiclass_baseline_table.py` from the
`*.json` legs in this directory. Do NOT hand-edit — re-run the script.
Measurements produced by `scripts/run_multiclass_baselines.py` on the same
3-way splits and `top20` feature selection the WNN screening arms use.

## Why this exists

Reviewer A asked for multiclass; it had been deferred as future work. These
are the no-box numbers a WNN multiclass arm has to be read against. **The bar
is per-dataset macro-F1, not the binary F1** — binary looks near-solved on
every dataset here while macro-F1 drops as far as 51 points below it.

## At a glance

| dataset                        |  K |   test rows | RF macro-F1 | XGB macro-F1 | RF benFPR | XGB benFPR |
|--------------------------------|----|-------------|-------------|--------------|-----------|------------|
| unsw-nb15 [temporal]           | 10 |      41,166 |       51.45 |        52.25 |     22.58 |      23.10 |
| cicids2017 [random]            | 15 |     282,788 |       81.36 |        64.76 |      0.07 |       0.38 |
| ciciot2023_neto_subsample [random] |  8 |     142,975 |       89.37 |        88.58 |      4.96 |       5.54 |
| ciciot2023_neto_full [random]  |  8 |   4,668,657 |       88.91 |        73.01 |      2.11 |       4.55 |

## Per dataset

### unsw-nb15 — `temporal_3way`

10 classes · train 175,341 · test 41,166 · features `top20`

```
model    binF1   binFPR   macroF1  weightF1     acc   benFPR    fit_s
---------------------------------------------------------------------
RF       89.18    24.95     51.45     78.35   76.49    22.58      4.9
XGB      89.07    25.44     52.25     78.23   77.58    23.10      2.1
```

```
class                           support   RF rec  XGB rec
---------------------------------------------------------
Worms                                23     43.5     43.5
Shellcode                           199     59.8     64.3
Backdoor                            283      7.4     14.5
Analysis                            351      0.9      4.3
Reconnaissance                    1,766     79.4     80.8
DoS                               2,061     17.1     11.5
Fuzzers                           3,000     56.0     56.5
Exploits                          5,545     79.7     90.0
Generic                           9,438     97.0     97.1
Normal                           18,500     77.4     76.9
```

### cicids2017 — `random_3way`

15 classes · train 2,262,300 · test 282,788 · features `top20`

```
model    binF1   binFPR   macroF1  weightF1     acc   benFPR    fit_s
---------------------------------------------------------------------
RF       99.72     0.07     81.36     99.86   99.86     0.07    192.7
XGB      99.69     0.07     64.76     99.43   99.43     0.38    104.6
```

```
class                           support   RF rec  XGB rec
---------------------------------------------------------
Heartbleed                            0      0.0      0.0
Infiltration                          2     50.0      0.0
Web Attack - SQL Injection            2      0.0      0.0
Web Attack - XSS                     62     38.7      6.5
Web Attack - Brute Force            147     72.8     20.4
Bot                                 196     76.0     73.5
DoS Slowhttptest                    547     99.5     91.6
DoS slowloris                       557     99.5     94.8
SSH-Patator                         587     99.7     98.5
FTP-Patator                         860     99.9     99.1
DoS GoldenEye                     1,019     99.6     93.8
DDoS                             12,678    100.0     99.2
PortScan                         15,887     99.5     99.4
DoS Hulk                         23,112     99.9     99.2
BENIGN                          227,132     99.9     99.6
```

### ciciot2023_neto_subsample — `random_3way`

8 classes · train 1,143,802 · test 142,975 · features `top20`

```
model    binF1   binFPR   macroF1  weightF1     acc   benFPR    fit_s
---------------------------------------------------------------------
RF       98.75     7.12     89.37     95.86   95.95     4.96    127.7
XGB      98.61     7.50     88.58     95.57   95.65     5.54     36.6
```

```
class                           support   RF rec  XGB rec
---------------------------------------------------------
BruteForce                        1,307     60.2     60.7
Web-based                         2,457     61.9     63.5
Spoofing                          9,937     82.3     80.6
Mirai                            15,014    100.0    100.0
Recon                            18,999     91.7     90.8
Benign                           20,000     95.0     94.5
DoS                              20,121    100.0    100.0
DDoS                             55,140    100.0    100.0
```

### ciciot2023_neto_full — `random_3way`

8 classes · train 37,349,263 · test 4,668,657 · features `top20`

```
model    binF1   binFPR   macroF1  weightF1     acc   benFPR    fit_s
---------------------------------------------------------------------
RF       99.89     3.41     88.91     99.65   99.66     2.11   2398.4
XGB      99.69    13.21     73.01     99.32   99.34     4.55    494.7
```

```
class                           support   RF rec  XGB rec
---------------------------------------------------------
BruteForce                        1,376     53.8     14.6
Web-based                         2,448     53.4     18.1
Recon                            35,428     85.4     70.8
Spoofing                         48,612     87.1     76.5
Benign                          109,819     97.9     95.4
Mirai                           263,573    100.0    100.0
DoS                             808,413    100.0     99.9
DDoS                          3,398,988    100.0    100.0
```

