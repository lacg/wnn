# ======================================================================
# MANUAL SECTION — 46M single-flow results (not produced by build_xds_5tables.py;
# a full regen of this file may drop this section — re-append from git if so)
# ======================================================================

## XDS-ciciot-46M-96b-Wc-C35-250n100b-OI-r63432  (flow 4299, completed 09/07/2026 ~15:45 UTC)

Single seed (r63432), 96b thermometer × Wc (ce=0.70) × top20, 46M CIC-IoT-2023
(neto_full, random split, K-fold 5×5, OI/QUAD). Held-out = 20% val (9,337,316 rows,
2.35% benign). Early-stopped Gen 100/250 (plateau best=1.3559 from ~Gen 46).
GA winners ~245-250n × ~60-100b (chunked-wheel run; MARKER_CHUNK 8×32n dispatches).

### Final validation — all genome types × all threshold modes (F1% | FPR% | Acc%)

    Genome        Threshold            |   F1    |  FPR    |  Acc
    --------------------------------------------------------------
    best_f1/acc   train_cal            |  92.28  |   6.16  |  99.22
    best_f1/acc   fixed_05             |  87.28  |   0.05  |  98.46
    best_f1/acc   platt                |  91.78  |   3.49  |  99.14
    best_f1/acc   beta                 |  91.06  |   2.09  |  99.03
    best_f1/acc   empirical            |  92.00  |  11.30  |  99.23
    best_f1/acc   empirical_cumulative |  92.12  |   4.35  |  99.18
    best_f1/acc   val_cal              |  92.28  |   6.16  |  99.22
    best_fpr      train_cal            |  92.20  |   7.64  |  99.22
    best_fpr      fixed_05             |  87.73  |   0.04  |  98.53
    best_fpr      platt                |  91.72  |   3.95  |  99.13
    best_fpr      empirical_cumulative |  91.96  |   4.77  |  99.17
    best_ce/fit   train_cal            |  92.20  |   7.73  |  99.22
    best_ce/fit   fixed_05             |  87.66  |   0.04  |  98.52
    best_ce/fit   platt                |  91.78  |   3.87  |  99.14
    best_ce/fit   empirical_cumulative |  91.91  |   4.30  |  99.16

### vs published paper table (250n × 60-64b cohort bests)

    Paper row       Paper (F1/FPR/Acc)      f4299 point                  f4299 (F1/FPR/Acc)   Delta
    ----------------------------------------------------------------------------------------------------
    Best F1         92.18 /  6.73 / 99.21   best_f1 x train_cal(=val)    92.28 /  6.16 / 99.22   dominates all 3
    Best FPR        88.34 /  0.71 / 98.64   best_fpr x fixed_05          87.73 /  0.04 / 98.53   FPR 18x lower, -0.61pp F1
    Matched FPR(b)  89.58 /  1.65 / 98.83   best_f1 x beta               91.06 /  2.09 / 99.03   +1.48pp F1, +0.44pp FPR
    (bonus)         --                      best_f1 x emp_cumulative     92.12 /  4.35 / 99.18   ~paper-best F1 at -2.4pp FPR

Notes:
- train_cal == val_cal for best_f1 (92.28/6.16/99.22): threshold fitted on train
  transfers exactly to held-out val — DEPLOYABLE, not an oracle artifact.
- The 0.04% FPR point is fixed_05 (raw 0.5 threshold, calibration-free):
  ~88 false alarms / 219,639 benign flows.
- n=1 seed vs paper best-of-cohort: needs sibling 96b-Wc seeds before any paper
  table swap. Data source: validation_summaries flow_id=4299, validation_point='final'
  (latest batch), threshold_metadata JSON.
