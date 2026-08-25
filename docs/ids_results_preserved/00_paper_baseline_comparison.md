# =====================================================================
# SECTION 0 — PAPER / BASELINE COMPARISON
# =====================================================================

## 0A. Cohort-level claim (defensible) — GA Neurons, `best_fitness`, val_cal, mean±SD

This is what the paper should claim: a mean over all completed seeds of one pre-declared
config, not a mined maximum.

```
dataset / config                    |  n | WNN F1        WNN FPR      | RF (raw)      XGB (raw)     | dF1 vs RF   dFPR vs RF
------------------------------------+----+----------------------------+-----------------------------+----------------------
UNSW-NB15 temporal                  | 22 | 86.21±1.89   14.39±7.15   | 85.18/27.73   84.93/28.68  |  +1.03      -13.34  WIN
  SP100-unswt-quad-16bWb            |    |                            |                             |
UNSW-NB15 random                    | 21 | 94.39±0.09    0.61±0.06   | 96.06/ 0.30   95.54/ 0.34  |  -1.67       +0.31  loss
  SP100-unswr-qsr-64bWb             |    |                            |                             |
CICIDS2017 random                   | 22 | 99.53±0.07    0.15±0.07   | 99.74/ 0.08   99.65/ 0.12  |  -0.21       +0.07  ~tie
  SP100-cicids-quad-96bWa           |    |                            |                             |
CIC-IoT-2023 neto-sub random        | 22 | 92.75±0.24    8.40±0.77   |    n/a        93.36/11.65  |  -0.61*     -3.25*  mixed
  SP100-ciciot-quad-96bWc           |    |                            |  (* vs XGB)                 |
```

## 0B. Best individual genome (CEILING, not the claim)

Mined across every genome_type × all 7 threshold modes. **Best-of-N inflates** — a maximum
over ~22 seeds × 5 genome types × 7 modes is not an estimate of expected performance. Quote
these only as "the best architecture we found", never as the cohort result.

```
dataset                             | WNN best F1/FPR/Acc      | baseline (raw)        | verdict
------------------------------------+--------------------------+-----------------------+------------------
UNSW-NB15 temporal                  | 90.20 /  5.30 / 90.22    | RF  85.18/27.73       | +5.02 F1, -22.43 FPR  STRICT WIN
UNSW-NB15 random                    | 94.56 /  0.61 / 99.17    | RF  96.06/ 0.30       | -1.50 F1, +0.31 FPR   loss
CICIDS2017 random                   | 99.64 /  0.08 / 99.77    | RF  99.74/ 0.08       | -0.10 F1, tie FPR     ~tie
CIC-IoT-2023 neto-sub random        | 93.35 /  7.50 / 96.69    | XGB 93.36/11.65       | -0.01 F1, -4.15 FPR   FPR WIN
  sub-5% FPR point                  | 93.08 /  4.91 / 96.46    |                       |
  sub-4% FPR point                  | 90.35 /  2.04 / 94.72    |                       |
```

## 0C. Reading — where the paper's claim actually is

1. **UNSW-NB15 temporal is the strongest result and it is a cohort-level win**, not a
   best-of-N artifact: +1.03 pp F1 and **−13.34 pp FPR** vs raw RF at n=22, and the mined
   best widens it to +5.02 / −22.43. Temporal is the hard, realistic split (train past →
   test future), which is exactly where the trees degrade.
2. **CICIDS2017 is solved by classical ML** (RF 99.74 F1 / 0.08 FPR) and the WNN lands
   ~0.2 pp behind. This must be framed as **efficiency parity** (10⁴–10⁶× smaller model),
   never as accuracy superiority — the pre-registered warning from the baseline measurement
   holds.
3. **UNSW random is a genuine loss** (−1.67 pp F1). Random splits leak temporal structure,
   so trees do very well; the honest reading is that the WNN's advantage is specific to the
   temporal/shift regime.
4. **CIC-IoT-2023 is an FPR story**: F1 is a statistical tie with XGBoost while FPR is
   3–4 pp lower, and the cohort holds a sub-5% FPR point at 93.08 F1.
5. **Baselines are RAW numeric top-20 on the WNN's own split.** Thermo-encoded RF is
   sometimes *stronger* (temporal 86.05 vs raw 85.18), so the paper reports both per table;
   the raw column is used here.


## Cohort state at generation time

```
TOTAL flows: 3740  completed: 2268  running: 1  queued: 392
SP100 cicids: 22/100
SP100 ciciot: 22/100
SP100 ciciot46m: 0/2
SP100 unswr: 43/200
SP100 unswt: 22/100
```

