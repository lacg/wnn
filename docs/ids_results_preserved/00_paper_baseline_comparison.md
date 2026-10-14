# =====================================================================
# SECTION 0 — PAPER / BASELINE COMPARISON
# =====================================================================

## 0A. Cohort-level claim (defensible) — GA Neurons, `best_fitness`, val_cal, mean±SD

This is what the paper should claim: a mean over all completed seeds of one pre-declared
config, not a mined maximum.

REFRESHED 31/08/2026 — n is now 29 seeds per cohort (was 21-22). ALL THREE metrics:
F1 AND FPR AND ACC. Baseline Acc is an em dash where it was never measured, never inferred.

SCOPE OF THE MEAN (this is the whole claim): GA-NEURONS phase only, genome_type
best_fitness, val_cal, held-out final. It must NOT be pooled across phases — a flow emits
one validation_summary per phase, so an unfiltered query returns 2 rows per seed and mixes
the GRID result into the GA mean. Doing that on 31/08 dragged CIC-IoT to 91.92±1.03 /
11.94±3.60 and looked like "the old SDs were understated". They were not; the pooling was
wrong. Grid is ~2.9pp F1 and ~8.7pp FPR worse than GA on this dataset, so pooling shifts
the mean and inflates the SD roughly 4x. Filter on experiments.phase_type='ga_neurons'.

```
dataset / config                | n  | WNN F1       WNN FPR      WNN Acc     | RF (raw)              XGB (raw)            | dF1 vs RF  dFPR vs RF
--------------------------------+----+--------------------------------------+--------------------------------------------+----------------------
UNSW-NB15 temporal              | 29 | 86.41±1.98  14.17±7.22  86.59±1.85  | 85.37/27.39/86.00     85.08/28.42/85.76    |  +1.04     -13.22  WIN
  SP100-unswt-quad-16bWb        |    |                                      |                                            |
UNSW-NB15 random                | 29 | 94.37±0.08   0.62±0.05  99.14±0.02  | 95.91/ 0.30/99.40     95.46/ 0.33/99.34    |  -1.54      +0.32  loss
  SP100-unswr-qsr-64bWb         |    |                                      |                                            |
CICIDS2017 random               | 29 | 99.53±0.06   0.14±0.07  99.71±0.04  | 99.75/ 0.08/99.84     99.66/ 0.11/99.79    |  -0.22      +0.06  ~tie
  SP100-cicids-quad-96bWa       |    |                                      |                                            |
CIC-IoT-2023 neto-sub random    | 29 | 92.74±0.24   8.39±0.76  96.38±0.12  | 94.42/ 9.81/97.32     93.38/11.59/96.82    |  -1.68      -1.42  mixed
  SP100-ciciot-quad-96bWc       |    |                                      |  (measured 31/08, random_3way TEST split)  |
```

WHAT CHANGED
* n 21-22 -> 29. The means and SDs are otherwise unchanged to two decimals — the prior
  table was correct. No significance claim needs revisiting.
* Acc column added for the WNN on all four datasets.
* CIC-IoT now shows RANDOM FOREST, which the old row recorded as "n/a". RF was measured on
  28/04 and is the STRONGER baseline: against it the WNN loses F1 by 1.68pp while winning
  FPR by 1.42pp. The "-0.61 F1 / -3.25 FPR vs XGB" framing flattered the result by
  comparing against the weaker of the two classical models.
* The CIC-IoT baselines are re-measured on the random_3way TEST partition — the same 10%
  the WNN is reported on. The earlier run merged test+validation, a 2-way evaluation on a
  3-way dataset. The merge proved numerically harmless here (<0.1pp on every metric, as a
  random split should be) but it was not apples-to-apples
  (scripts/run_neto_subsample_baselines.py --eval-split test).
* UNSW/CICIDS baselines RE-MEASURED 31/08 on their own _3way TEST partitions, emitting all
  three metrics (scripts/run_all_baselines.py --split ... --eval-split test). That script
  had the same test+val merge AND did not expose --split at all, so UNSW temporal could not
  previously be run through it. The re-measured values land within 0.2pp of the banked ones
  on every metric, so the old F1/FPR figures were sound; only Acc was missing.
* ⚠️ **VAL-CALIBRATED RF/XGB on UNSW temporal_3way (13/09/2026, Plane IDS-9) — the trees
  catch up once they get the WNN's threshold rule.** Every banked "WNN −18pp FPR vs RF/XGB"
  read compared the WNN's `val_cal` row against trees scored at a FIXED 0.5 — not
  like-for-like. Re-measured with `scripts/verify_ids_baselines.py --dataset unsw --split
  temporal_3way [--n-bits 16]` (top-20, thermo-encoded exactly as the WNN sees it; the
  F1-optimal threshold fit on the 10% VAL, scored on the 10% TEST; logs in
  `logs/baselines_unsw/rf_xgb_temporal_3way_{8,16}b.log`):

      encoding  method   mode       thr      F1     FPR     Acc     size
      8b        RF       fixed_05   0.50   85.95   25.82   86.50   84.06 MB
      8b        RF       val_cal    0.89   91.16    6.01   91.20
      8b        XGBoost  fixed_05   0.50   84.67   28.95   85.38    0.25 MB
      8b        XGBoost  val_cal    0.89   91.85    2.66   91.87
      16b       RF       fixed_05   0.50   86.12   25.35   86.65  138.04 MB
      16b       RF       val_cal    0.88   90.66    7.61   90.71
      16b       XGBoost  fixed_05   0.50   84.96   28.42   85.64    0.27 MB
      16b       XGBoost  val_cal    0.87   92.09    3.64   92.11

  The `fixed_05` rows reproduce the banked 2-way numbers (RF 85.83/25.99, XGB 84.62/29.18)
  within 0.2pp, so the split change is immaterial; the CALIBRATION is not. XGBoost val_cal
  at 16b (92.09 / 3.64) beats the legacy WNN 16b-Wb row (88.86 / 8.78) on F1 AND FPR. Until
  the IDSXD-unswt cohort (flows 6183-6236) lands a Protocol-v2 WNN row, the honest UNSW
  temporal statement is "trees lead under equal calibration"; the size argument
  (0.27 MB XGB vs ~1 KB WNN) is the surviving differentiator. Every paper table must pair
  the WNN's mode with the SAME mode on the comparator — never val_cal vs fixed_05.
* **SWEEP-WINNER vs BEST-TREE-COLUMN table (13/09/2026, experiment-design readout; script
  `scripts/idsxd_best_column.py`, held-out `validation_summaries` final rows, mean±SD over seeds).**
  WNN column = the IDSXD column (arm × mode × phase × genome_type × thr_mode) with the best mean
  held-out F1 per dataset, reported in `val_cal`; comparator = best tree column in the same mode
  (`scripts/verify_ids_baselines.py`, RF/XGB/AdaBoost, raw and thermo, `logs/baselines_*/`).

      dataset (random_3way)  WNN best column (val_cal)                                    best RAW tree      best THERMO tree     WNN−raw          WNN−thermo
      UNSW-NB15              B34-CTRL qsr GA best_f1     94.38±0.06/0.59±0.03/99.15  n=5   RF 95.82/0.34      RF16b 95.30/0.36     −1.4 F1 +0.25 FPR  −0.9 F1 +0.23 FPR
      CICIDS2017             CE20 quad GA best_f1        99.52±0.03/0.12±0.01/99.70  n=3   RF 99.85/0.10      RF8b  99.52/0.21      −0.3 F1 +0.02 FPR   0.0 F1 −0.09 FPR
      CIC-IoT subsample      B15-AC quad GA best_fitness 92.99±0.05/7.49±0.85/96.48  n=3   RF 95.53/7.38      XGB8b 88.85/15.10    −2.5 F1 +0.11 FPR  +4.1 F1 −7.6 FPR
      Pareto WNN: unswr B05-CE GA best_ce empirical_cumulative 94.33±0.07/0.50±0.06 (n=3);
                  ciciot B15-AC GA best_f1 val_cal 92.98±0.08/6.79±0.76 (n=3, FPR SD 0.8 — noise).

  Caveats from the readout: unswr is a FLOOR (102 columns / all 7 done arms within 0.07pp; the
  quad half 0/30 and CE20/Wb-CTRL qsr unfinished); cicids CE20 leads Wa-CTRL by 0.04pp vs Wa's SD
  0.09 (provisional, era-clean); ciciot B15-AC wins with or without the pre-fix arms but the arm
  gap (0.02-0.03pp) is below the seed SD, IDSXD2 0/12 done; best_f1 and best_acc share a
  genome_hash per seed (duplicate rows = one genome under several thresholds). Under equal
  calibration the WNN trails the best RAW tree on F1 everywhere at ~equal FPR; on the SAME
  thermometer input it ties CICIDS, trails UNSW by 0.9pp and beats CIC-IoT by +4.1pp F1 / −7.6pp FPR.
  The old ciciot WNN row (93.34/8.37, 250n×100b) is superseded by the sweep's 92.99/7.49.
  **SUPERSEDED 26/09/2026 for cicids/ciciot by "IDS-2 FINAL (26/09/2026)" below** (IDSXD2 n=5 landed; ciciot call re-derived).
* **46M CIC-IoT-2023 (neto_full) classical comparators — Protocol v2, raw only (measured 13/09/2026,
  Plane IDS-13; table added 14/10/2026).** `scripts/verify_ids_baselines.py --dataset ciciot46m --split
  random_3way --raw`; dataset `lacg030175/CIC-IoT-2023-neto-full` (Kaggle/Neto copy, 46,686,580 rows),
  RAW top-20 canonical features (`TOP20_RF_FEATURES`), Inf/NaN median-imputed; RF 100 trees, XGBoost
  100/depth 6/lr 0.1, AdaBoost 100 stumps, rs 42; `val_cal` = F1-optimal threshold fit on the 10% VAL
  (the WNN's rule); scored on the 10% TEST (4,668,657 rows, 2.4% benign). Train time RF 231 s, XGB 19 s,
  AdaBoost 2602 s. Log: `logs/baselines_ciciot46m/rf_xgb_ada_random_3way_raw.log`. RAW only — 46.7M
  rows are never thermo-encoded for the trees.

      input  method         mode       thr      F1     FPR     Acc     size
      raw    RF             fixed_05   0.50   97.63    3.57   99.78   1008.55 MB
      raw    RF             val_cal    0.53   97.63    3.11   99.78
      raw    XGBoost        fixed_05   0.50   96.04    4.98   99.63      0.39 MB
      raw    XGBoost        val_cal    0.51   96.04    4.73   99.62
      raw    AdaBoost-100   fixed_05   0.50   94.94    7.58   99.52      0.05 MB
      raw    AdaBoost-100   val_cal    0.50   94.94    7.58   99.52
      raw    MLP (128-64)   —          —         —       —       —      PENDING (Plane IDS-39, blocked by IDS-6)

  Reading: F1/FPR are the columns — Acc 99.5-99.8 is the 97.6%-attack base-rate artifact. Calibration
  moves FPR only (RF 3.57→3.11, XGB 4.98→4.73; thr 0.50-0.53 because train/test share the base rate).
  ~0.7pp F1 above the 28/04 read on the same copy (RF 96.98/4.24, XGB 95.28/5.98, AdaBoost-50
  93.04/11.84), which merged test+val and scored at 0.5. Neto's own published rows (RF 96.53,
  AdaBoost 95.63) agree with this copy. The RAID-2026 46M rows (RF 92.64/13.48, XGB 91.67/14.80) are
  bencorn's re-merge — NOT comparable (see the audit bullet below).
  **WNN row: none yet.** The Protocol-v2 46M WNN row is IDS-6 (`IDS6-ciciot46m-B15AC-desir-r<seed>`,
  n=5; first flow 6438 in GA Neurons since 13/10). Its GA iteration numbers (e.g. F1 92.19 / FPR 1.34
  at gen 2, 14/10) are the during-search K-fold score, NOT comparable to this table — the comparable
  row is its held-out `val_cal` on the same 10% TEST. The legacy 2-way 46M WNN rows (§7) are not
  like-for-like.
* **WHY THE RAID-2026 TABLES DISAGREE WITH THE ROWS ABOVE (13/09/2026 audit, Plane IDS-14).**
  (1) 46M "RF (ours) 92.64/13.48, XGB 91.67/14.80" = the 23/04 run on bencorn's re-merge
  (`canonical-neto`, 45.0M rows, 39 features, fabricated Time_To_Live; logs/canonical_baselines_
  20260423T023819Z.log). The Kaggle/Neto copy (46,686,580 rows, 46 features) reads 96.98/4.24 and
  95.28/5.98 on 28/04 with the OLD 19/20 features and 97.63/3.57, 96.04/4.98 today: the DATASET COPY
  is +4.3pp/−9pp FPR, the feature list +0.65pp, calibration FPR-only. Neto's own rows (RF 96.53,
  Ada 95.63) agree with the Kaggle copy. (2) 1.3M subsample "RF 85.53/25.18, XGB 84.13/28.34" = the
  bencorn 1.07M/268K mirror (13/20 canonical features); reproduced today raw 83.46/29.62, 83.08/29.43;
  on neto-subsample thermo 88.46/19.98 · raw 95.53/7.38. (3) UNSW temporal "RF 86.41/25.22" = 2-way
  test at 0.5; temporal_3way TEST at 0.5 is 85.95/25.82 (partition/encoder noise); the +6pp is
  VAL-fitted calibration the paper never gave the trees (Elkan 2001; Sheng & Ling 2006; Lipton et
  al. 2014; sklearn TunedThresholdClassifierCV) — legitimate, VAL-only, same rule as the WNN's val_cal;
  inert on UNSW-random/CICIDS (thr 0.43-0.56, ≤0.3pp) = prior correction, not VAL overfit.
  Every CIC-IoT row in the next paper comes from the Kaggle/Neto copies; every table pairs modes.

## IDS-16 READOUT + CLOSED (09/10/2026) — SP-*-ablqsr memory-mode ablation on the fixed OI trainer (worker ABI 13)
> **Re-audit 10/10/2026: STANDS.** The cross-flow cache leak touched 1 of the 40 QSR cells this readout uses
> (flow 6313, unswr r46247, GA best_f1 = a copied QUAD row); the other 9 unswr runs sit at 93.48-93.51 / 1.12, so the
> unswr column (a tie) and every decision above are unchanged.

QSR-abi13 (n=10/dataset) vs two QUAD arms (SP-*-bin-n30, SP100-*-quad; config diff = memory_mode only, verified
key-by-key; ciciot bin with 4 -w64fix substitutes). GA best_f1 val_cal, held-out TEST. Pre-fix QSR rows are VOID.
```
dataset (shape)   QSR F1/FPR      QUAD bin        QUAD SP100      paired delta (QSR-QUAD)                    h/run QSR|QUAD
unswt 16bWb       85.46 / 17.59   86.54 / 13.58   86.86 / 13.79   dF1 -1.08 / -1.40 (n.s.), dFPR +4.0 (n.s.)  0.10 | 0.06
unswr 64bWb       93.50 /  1.12   93.54 /  1.08   93.56 /  1.05   tie; QSR ON the FP floor 10/10              0.51 | 0.36
cicids 96bWa      99.54 /  0.14   99.59 /  0.09   99.52 /  0.14   tie (QUAD-vs-QUAD era drift 10/10, p=0.002)  2.34 | 1.15
ciciot 96bWc      92.69 /  9.50   92.84 /  8.53   92.89 /  8.56   dFPR +0.97 / +0.93, 8/10, p<=0.03 (QSR worse) 3.07 | 1.3
```
- "QSR = the only tunable decode (+0.87 pp F1 / -0.5 pp FPR)" is REFUTED: it was the legacy trainer, not the decode.
- Cost after the fix: QSR 1.4-2.1x QUAD per run (was 7-15x).
- No pre-registered rule existed for this ablation — these are descriptive readings, not a rule firing.
- Decisions: IDS-3 gets NO QSR arm (the "only if competitive on unswt-16b" condition fails); IDS-4/5 memory mode
  stays QUAD (HOLD released). Caveats: n=10; cross-era (QUAD Jul/Aug vs QSR Oct); ablation at Wa/Wc/Wb with default
  aggregation, not CE20-zs / B15-AC-desir; cicids era drift as large as the QSR delta (no cicids memory-mode claim).
- The 36 remaining UNSW-random QSR reruns (IDSXD-unswr-qsr 9-arm sweep + SP100-unswr-qsr) continue under IDS-2.
- Readout: ids-security agent; scripts in the session scratchpad (abl.py / run2.py / stats.py).

### IDS-2 DECIDED (10/10/2026) — Wb-CTRL QUAD (rule applied on full 70 own cells after re-validation)
260 cache-inherited rows re-validated on each run's own model (`scripts/ids2/revalidate_inherited.py`, b66c0eb7; 546 s).
The 94.274 B34-CTRL QSR outlier re-scores to 93.466. Raw per-metric winners all fall inside 1 pooled seed-SD of
Wb-CTRL QUAD (F1 B15-AC QSR +0.007/SD 0.008; FPR Wb-CTRL QSR −0.012/SD 0.116; Acc CE20 QUAD +0.007/SD 0.015) →
fallback (b) on all three → **single winner Wb-CTRL QUAD** = IDS-7 config. Every arm sits on the same FP floor
(F1 93.49-93.50 mean, 18/18 arms share the r20405 grid best cell 93.510/1.119/98.921); QSR ≈ QUAD on UNSW-random.
Readout: `experiments/ids2_readout_post_revalidation.txt`.

### CACHE-LEAK RE-VALIDATION SCOPE (10/10/2026, Luiz)
Fix (a) DEPLOYED 10/10 12:32 EDT (f4e14295; dashboard + worker restarted; DB backup `db/wnn.db.pre-valcache-20261010`).
Re-score ONLY: **A — IDS-2** (`IDSXD-unswr-*`, 68 genomes / 121 grid rows / 30 flows; gates IDS-7 and the 405-flow tandem) and
**B — IDS-16** (`SP-*-abl*`, 167 genomes / 296 rows / 114 flows; the closed ablation readout is under re-audit).
SKIPPED, annotate only: SP100 (old n=100, superseded by the IDS-3..7 tandem), MC ablations (multiclass is post-paper),
XDS, legacy CIC-IoT 8b explorations, zINVALID. Retrain-and-score stored genomes; no search re-runs.

## IDS-20 FINAL (06/10/2026) — IDSAGG paired aggregation A/B (zscore vs desirability), CICIDS2017 + CIC-IoT-2023 subsample

Pre-registered rule: `experiments/ids_aggregation_ab_rule.json` (registered 26/09, before any IDSAGG flow existed;
B05-AC addendum 04/10 added before any IDSAGG metric was read). Read-only DB readout. Every number is HELD-OUT TEST
val_cal from `validation_summaries.threshold_metadata` (`validation_point='final'`), GA-neurons phase unless marked
Grid. Never `iterations.best_f1`.

```
Header (06/10/2026 ~11:00 UTC)
  IDSAGG  cicids quad-96b  2 arms x 2 aggr x {20414..20418}  20/20 completed | 66.8 h total | 3.34 h/run | last done 05/10/2026 14:10 UTC
  IDSAGG  ciciot quad-96b  2 arms x 2 aggr x {20414..20418}  20/20 completed | 61.8 h total | 3.09 h/run | last done 06/10/2026 02:20 UTC
  IDSAGGX ciciot quad-96b  B05-AC x 2 aggr x {20414..20418}   5/10 completed | 7.7 h total  | 1.54 h/run | last done 06/10/2026 10:02 UTC
          (6425 desir r20416 running, 6426-6429 queued) ETA rule (latest + 5 x avg) 06/10/2026 17:45 UTC = 13:45 EDT;
          per-aggr avg (zs 1.00 h, desir 2.36 h) gives ~06/10/2026 19:10 UTC = 15:10 EDT
Config check (all 45 completed + running/queued pairs, key-by-key): zs vs desir differ ONLY in fitness_aggregation and
  fitness_ce_anchor_normalized (desir 0.2128), for every arm. Constant: QUAD, 96b thermo, top20, binary, random_3way,
  k_folds=5 / kfold_per_gen=5, pop 50, patience 5, ga_generations 250. cicids bits 4-34 / cap 500; ciciot bits 4-100 / cap 250.
```

### Primary (F1) and secondary (FPR) — exactly as pre-registered

Per seed: mean over the 2 pre-registered arms of (desirability − zscore), GA best_f1 val_cal; two-sided paired t over
5 seeds; Holm across the 2 datasets; fires iff Holm p < 0.05 AND ≥ 4/5 seeds share the sign.

```
test       | dataset | r20414 | r20415 | r20416 | r20417 | r20418 | mean desir-zs | SD    | CI95             | t     | p raw  | Holm   | signs  | rule
-----------+---------+--------+--------+--------+--------+--------+---------------+-------+------------------+-------+--------+--------+--------+-------------
F1  (pp)   | cicids  | +0.023 | -0.155 | -0.054 | -0.153 | -0.073 | -0.082        | 0.075 | [-0.175, +0.010] | -2.47 | 0.0687 | 0.1375 | +1/-4  | did NOT fire
F1  (pp)   | ciciot  | +0.123 | +0.097 | +0.155 | +0.019 | -0.086 | +0.062        | 0.097 | [-0.059, +0.182] | +1.42 | 0.2275 | 0.2275 | +4/-1  | did NOT fire
FPR (pp)   | cicids  | +0.005 | +0.076 | -0.010 | +0.150 | +0.024 | +0.049        | 0.065 | [-0.032, +0.130] | +1.68 | 0.1679 | 0.1679 | +4/-1  | did NOT fire
FPR (pp)   | ciciot  | -1.165 | -0.870 | -1.845 | -0.983 | -0.445 | -1.061        | 0.512 | [-1.697, -0.426] | -4.64 | 0.0097 | 0.0195 | +0/-5  | FIRED (desirability lower FPR)
```

**DECISION (Luiz, 06/10/2026) — supersedes the rule's default for IDS-4:**
- **IDS-4 (CICIDS2017 n=100) = CE20 under ZSCORE.** CE20-zs 99.52±0.07 F1 / 0.11±0.04 FPR vs CE20-desir 99.45±0.08 /
  0.10±0.04 (zs F1 ahead 4/5, FPR tied); best single genome 99.602/0.072/99.749 is CE20-zs (flow 6414); every Pareto
  point with F1>=99.4 is CE20-zs; 2.2 vs 4.7 h/run (~222 vs ~472 h at n=100). Weights f1/fpr/ce/acc .30/.40/.20/.10.
- **IDS-5 (CIC-IoT n=100) = B15-AC under DESIRABILITY (Luiz 07/10/2026).** Weights f1/fpr/ce/acc .15/.15/.175/.525.
  vs B05-AC-desir (same seeds): mean F1 93.00 vs 93.04 (tie), FPR 6.92 vs 7.35 (B15 lower 4/5; under the 0.50 pp
  pre-registered flag, so the pick is Luiz's call, not the rule's), Acc 96.47 vs 96.51 (tie); best genome 93.225/6.71/96.59
  (B15); FPR<5% at F1>=92.5 on 4/5 runs vs 0/5; only WNN mean FPR below raw RF's 7.38. Cost ~5.35 vs 3.47 h/run.
- Both cohorts queue AFTER the 94 QSR -abi13 reruns.
- **COHORT SPEC (Luiz 07/10/2026)**: **100 FRESH seeds** 20419-20518 per cohort, seed-major,
  `IDS4-cicids-CE20-zs-r<seed>` / `IDS5-ciciot-B15AC-desir-r<seed>`; spec `experiments/ids45_n100_cohort_spec.json`
  (dry-run 200 clean, NOT created). The IDS-20 seeds 20414-20418 are NOT pooled: they selected the config, so they
  stay the selection evidence (the aggregation ablation), never part of the confirmatory n=100.
- **HOLD — 4-COHORT TANDEM (Luiz 09/10/2026).** The memory-mode HOLD is RELEASED (IDS-16 readout above: no reason
  to reopen QUAD). The cohorts now wait for IDS-2 (UNSW-random winner) so IDS-7 can be specified, then ALL FOUR
  n=100 cohorts (IDS-3 unswt, IDS-4 cicids, IDS-5 ciciot, IDS-7 unswr) are queued in ONE seed-major interleave on the
  same seeds 20419-20518 (seed s: IDS3, IDS4, IDS5, IDS7, then s+1 ...), so every cohort walks to n=100 in tandem and
  any partial readout is a complete 4-dataset set at equal n.
- **IDS-3 SPEC DRAFTED (09/10/2026):** clones of `IDSXD-unswt-quad-16b-B15-CE-r20403` (flow 6242) with only `seed`
  changed, `IDS3-unswt-B15CE-desir-r<seed>`, seeds 20419-20518; spec `experiments/ids3_n100_cohort_spec.json` (dry-run
  100 clean, NOT created). Config: B15-CE .15/.15/.525/.175, desirability (anchor 0.2128), QUAD_WEIGHTED, 16b
  thermometer top20, bits 4-34, neurons <=500, temporal_3way, k-fold 5x5, patience 5. NO QSR arm (IDS-16).
  The final combined 4-way queue file is built from the three spec files + IDS-7's once IDS-2 names its winner.
- **IDS-6 JOINS THE TANDEM (Luiz 10/10/2026): CIC-IoT-2023 46M (neto_full), n=5 (was n=2 in the story),**
  interleaved in the FIRST 5 seed rounds only (seeds 20419-20423): round order IDS3, IDS4, IDS5, IDS6, IDS7; rounds
  6-100 are IDS3/4/5/7. Config = IDS-5 (B15-AC desirability, `IDSAGG-ciciot-quad-96b-B15-AC-desir-r20414`) with ONLY
  `ids_dataset=ciciot2023_neto_full` and `ids_encoded_storage=memmap` (required at 46M: the in-memory encode is ~150 GB
  of transient allocations — the 03/07 OOM loop; memmap bounds it to one chunk). Names `IDS6-ciciot46m-B15AC-desir-r<seed>`,
  spec `experiments/ids6_46m_cohort_spec.json` (dry-run 5 clean, NOT created). Total tandem = 405 flows. Each 46M run
  is >1 day, so rounds 1-5 are ~2 days each. The 5 legacy paused 46M Wc flows (4534-4538) were CANCELLED 10/10.
- **OLDER ABLATIONS RE-SCORED (10/10/2026):** all 320 cache-inherited rows in `SP-*-abl*` (123 flows) re-validated on
  their own model (`experiments/ablations_inherited_rows.json`; parts in `experiments/ablations_reval_parts/`; old values in
  `db/wnn.db.pre-valcache-20261010`). Conventional cell (GA best_f1 val_cal) UNCHANGED in every cohort (max |dF1| 0.005).
  Best-everywhere moved only DOWN, only for the weak modes that had been borrowing QUAD rows: ciciot 2s −3.92 / 3s −3.66,
  unswt 3s −1.86 / 2s −0.97 / PLN −0.94, cicids 2s −0.08 / 3s −0.05 pp. The memory-mode ablation's ordering (QUAD ≥ QSR >
  PLN > ternary/binary) is unchanged and the gaps widen. Pre-ABI-13 `ablqsr` rows remain VOID (IDS-16).
- **405-FLOW TANDEM QUEUED (Luiz go, 10/10/2026 16:39 EDT): flows 6430-6834**, seed-major IDS3/IDS4/IDS5/[IDS6 rounds 1-5]/IDS7, 2 experiments each, verify failures 0; first flow 6430 started on the fixed cache (`[VALCACHE] abi=14 ... mm=QUAD_WEIGHTED seed=20419`). 46M memory: Luiz chose to let the controller mem-watchdog (v7, pause-first) handle contention. Readout rule DRAFT `experiments/ids_n100_readout_rule.json` — criterion A/B still open (does not gate the runs). Next: re-score the older summer ablations (abl3s/2s/2big/pln).
- **IDS-7 SPEC DRAFTED (Luiz 10/10/2026): Wb-CTRL QUAD** (IDS-2 winner) — clones of `IDSXD-unswr-quad-64b-Wb-CTRL-r20403`
  (flow 5965; the 3 sources differ only in seed) with only `seed` changed, `IDS7-unswr-WbCTRL-desir-r<seed>`, seeds
  20419-20518; Wb .10/.35/.35/.20 desirability, QUAD, 64b top20 random_3way, bits 4-34, neurons <=500, OI, k-fold 5x5,
  patience 5. Spec `experiments/ids7_n100_cohort_spec.json` (dry-run 100 clean). **TANDEM FILE BUILT:**
  `experiments/ids_tandem_405_spec.json` = seed-major IDS3, IDS4, IDS5, [IDS6 rounds 1-5], IDS7 (dry-run 405 clean,
  NOT created).


**Decision the rule dictates:**
- **CICIDS2017: desirability stands.** Primary did not fire (Holm 0.138); FPR secondary did not fire (Holm 0.168), so
  the "unless FPR favours zscore" clause is not triggered. F1 DIRECTION favoured zscore on 4/5 seeds (−0.08 pp, at the
  ~99.5 ceiling; CI touches 0) — disclose, do not claim. IDS-4 = CE20 under desirability needs no zscore re-check per the rule.
- **CIC-IoT-2023: desirability stands.** Primary did not fire (Holm 0.228). The FPR secondary FIRED in favour of
  desirability (−1.06 pp, 5/5 seeds, Holm 0.0195) — it reinforces the default, it does not trigger the zscore clause.
  IDS-5 = B05-AC under desirability stands. The FPR effect is concentrated in B15-AC (per-arm table below); B05-AC's own
  contrast is the addendum and is interim.
- IDS-3 (UNSW temporal, B15-CE) is outside this cohort; the earlier paired IDSD test on unswt already ruled for desirability.

Paper wording: "the pre-registered aggregation test did not fire on F1 on either dataset; on CIC-IoT-2023 the FPR
secondary fired for desirability (−1.06 pp, 5/5 seeds, Holm p = 0.02)". Never "desirability is better" on F1.

### Per-arm descriptive (GA best_f1 val_cal, mean±SD over 5 seeds; NOT tested — arm × aggregation is descriptive)

```
dataset | arm      | aggr  | n | F1 %         | FPR %       | Acc %        | genome (GA best_f1)       | GA gens        | wall h
--------+----------+-------+---+--------------+-------------+--------------+---------------------------+----------------+-----------
cicids  | Wa-CTRL  | zs    | 5 | 99.52±0.07   | 0.15±0.06   | 99.70±0.05   | 221±170 n x 33.95±0.07 b  | 60.0±0.0       | 2.37±1.53
cicids  | Wa-CTRL  | desir | 5 | 99.42±0.06   | 0.25±0.05   | 99.63±0.04   | 286±115 n x 34.00±0.00 b  | 88.0±31.1      | 4.05±1.10
cicids  | CE20     | zs    | 5 | 99.52±0.07   | 0.11±0.04   | 99.70±0.04   | 176±80 n x 33.46±0.83 b   | 66.0±5.5       | 2.22±0.91
cicids  | CE20     | desir | 5 | 99.45±0.08   | 0.10±0.04   | 99.66±0.05   | 242±102 n x 33.93±0.15 b  | 98.0±21.7      | 4.72±2.29
ciciot  | Wc-CTRL  | zs    | 5 | 92.93±0.06   | 7.69±0.85   | 96.46±0.05   | 212±40 n x 67.2±7.2 b     | 72.0±16.4      | 2.19±0.56
ciciot  | Wc-CTRL  | desir | 5 | 92.79±0.12   | 7.64±0.94   | 96.38±0.08   | 181±34 n x 76.8±7.2 b     | 116.0±15.2     | 3.71±0.61
ciciot  | B15-AC   | zs    | 5 | 92.74±0.10   | 9.00±0.81   | 96.39±0.03   | 134±69 n x 50.6±3.2 b     | 64.0±5.5       | 1.10±0.51
ciciot  | B15-AC   | desir | 5 | 93.00±0.11   | 6.92±0.74   | 96.47±0.06   | 216±29 n x 80.0±0.0 b     | 146.0±11.4     | 5.35±0.55
```
Pre-declared best_fitness column (GA val_cal): cicids Wa zs 99.53±0.07/0.16±0.06, desir 99.42±0.06/0.27±0.06; CE20 zs
99.51±0.08/0.09±0.04, desir 99.46±0.08/0.11±0.02; ciciot Wc zs 92.88±0.05/7.53±0.76, desir 92.81±0.13/7.72±0.82; B15-AC zs
92.75±0.11/9.38±0.40, desir 93.04±0.14/7.04±0.41.

Per-seed desir − zs (GA best_f1 val_cal; zs value, desir value, delta):
```
dataset | arm     | seed  | F1 zs / desir / d        | FPR zs / desir / d       | Acc zs / desir / d       | size zs -> desir   | gens zs/desir | wall h zs/desir | flows
--------+---------+-------+--------------------------+--------------------------+--------------------------+--------------------+---------------+-----------------+----------
cicids  | Wa-CTRL | 20414 | 99.47 / 99.42 / -0.054   | 0.18 / 0.26 / +0.083     | 99.66 / 99.63 / -0.035   | 369n/34b -> 340n/34b | 60 / 90     | 3.90 / 5.57     | 6380,6381
cicids  | Wa-CTRL | 20415 | 99.56 / 99.42 / -0.147   | 0.11 / 0.26 / +0.149     | 99.72 / 99.63 / -0.094   | 98n/34b -> 264n/34b  | 60 / 80     | 1.21 / 3.99     | 6388,6389
cicids  | Wa-CTRL | 20416 | 99.58 / 99.53 / -0.056   | 0.12 / 0.17 / +0.057     | 99.74 / 99.70 / -0.036   | 97n/34b -> 94n/34b   | 60 / 140    | 1.24 / 2.49     | 6396,6397
cicids  | Wa-CTRL | 20417 | 99.58 / 99.39 / -0.192   | 0.11 / 0.32 / +0.209     | 99.73 / 99.61 / -0.123   | 100n/34b -> 362n/34b | 60 / 70     | 1.32 / 4.28     | 6404,6405
cicids  | Wa-CTRL | 20418 | 99.43 / 99.37 / -0.057   | 0.24 / 0.24 / +0.006     | 99.64 / 99.60 / -0.036   | 441n/34b -> 371n/34b | 60 / 60     | 4.20 / 3.93     | 6412,6413
cicids  | CE20    | 20414 | 99.47 / 99.57 / +0.100   | 0.14 / 0.06 / -0.073     | 99.67 / 99.73 / +0.063   | 102n/32b -> 328n/34b | 70 / 90     | 1.58 / 5.53     | 6382,6383
cicids  | CE20    | 20415 | 99.57 / 99.41 / -0.162   | 0.13 / 0.13 / +0.004     | 99.73 / 99.63 / -0.101   | 99n/34b -> 264n/34b  | 70 / 120    | 1.47 / 6.18     | 6390,6391
cicids  | CE20    | 20416 | 99.45 / 99.40 / -0.053   | 0.14 / 0.07 / -0.077     | 99.65 / 99.62 / -0.032   | 291n/34b -> 94n/34b  | 70 / 70     | 3.74 / 1.63     | 6398,6399
cicids  | CE20    | 20417 | 99.49 / 99.38 / -0.114   | 0.06 / 0.15 / +0.090     | 99.68 / 99.61 / -0.072   | 203n/34b -> 335n/34b | 60 / 120    | 2.11 / 7.15     | 6406,6407
cicids  | CE20    | 20418 | 99.60 / 99.51 / -0.088   | 0.07 / 0.11 / +0.042     | 99.75 / 99.69 / -0.056   | 186n/34b -> 188n/34b | 60 / 90     | 2.20 / 3.10     | 6414,6415
ciciot  | Wc-CTRL | 20414 | 92.99 / 92.76 / -0.232   | 7.35 / 6.98 / -0.365     | 96.48 / 96.34 / -0.141   | 210n/64b -> 159n/80b | 60 / 110    | 1.87 / 3.36     | 6384,6385
ciciot  | Wc-CTRL | 20415 | 92.98 / 92.95 / -0.023   | 8.53 / 8.69 / +0.160     | 96.51 / 96.50 / -0.008   | 208n/64b -> 220n/64b | 60 / 100    | 1.81 / 3.07     | 6392,6393
ciciot  | Wc-CTRL | 20416 | 92.84 / 92.86 / +0.022   | 8.03 / 6.65 / -1.370     | 96.42 / 96.39 / -0.031   | 150n/80b -> 149n/80b | 60 / 140    | 1.76 / 4.25     | 6400,6401
ciciot  | Wc-CTRL | 20417 | 92.90 / 92.65 / -0.247   | 6.37 / 7.27 / +0.905     | 96.40 / 96.29 / -0.111   | 244n/64b -> 216n/80b | 90 / 120    | 3.06 / 4.48     | 6408,6409
ciciot  | Wc-CTRL | 20418 | 92.94 / 92.74 / -0.202   | 8.18 / 8.61 / +0.425     | 96.48 / 96.38 / -0.099   | 250n/64b -> 161n/80b | 90 / 110    | 2.43 / 3.42     | 6416,6417
ciciot  | B15-AC  | 20414 | 92.61 / 93.09 / +0.478   | 9.88 / 7.92 / -1.965     | 96.35 / 96.55 / +0.202   | 63n/49b -> 247n/80b  | 70 / 150    | 0.55 / 5.96     | 6386,6387
ciciot  | B15-AC  | 20415 | 92.78 / 92.99 / +0.218   | 9.06 / 7.17 / -1.900     | 96.41 / 96.48 / +0.062   | 170n/48b -> 191n/80b | 70 / 140    | 1.42 / 4.75     | 6394,6395
ciciot  | B15-AC  | 20416 | 92.84 / 93.13 / +0.288   | 8.17 / 5.85 / -2.320     | 96.42 / 96.51 / +0.090   | 219n/55b -> 180n/80b | 60 / 160    | 1.69 / 5.21     | 6402,6403
ciciot  | B15-AC  | 20417 | 92.66 / 92.95 / +0.286   | 9.71 / 6.83 / -2.870     | 96.37 / 96.44 / +0.068   | 156n/48b -> 222n/80b | 60 / 150    | 1.27 / 5.89     | 6410,6411
ciciot  | B15-AC  | 20418 | 92.83 / 92.86 / +0.029   | 8.18 / 6.86 / -1.315     | 96.42 / 96.39 / -0.025   | 63n/52b -> 239n/80b  | 60 / 130    | 0.59 / 4.94     | 6418,6419
```
Per-arm mean desir − zs (descriptive): cicids Wa-CTRL F1 −0.101 (0/5 positive), FPR +0.101 (5/5 higher); CE20 F1 −0.063
(1/5 positive), FPR −0.003 (2 lower / 3 higher). ciciot Wc-CTRL F1 −0.136 (1/5 positive), FPR −0.049 (2 lower / 3 higher);
B15-AC F1 +0.260 (5/5), FPR −2.074 (5/5). The ciciot pooled FPR firing is carried by B15-AC; on the control the
aggregation effect is mixed in sign. Arm x aggregation interaction is visible but descriptive only.

Descriptive-only items (rule): **generations** — desirability ran longer in every arm (+28 to +82 GA generations;
mediator under the IDSZ 24/08 ruling, disclosed, not penalised). **Wall time** — desirability 1.6x (Wc) to 4.9x
(B15-AC) zscore. **Genome size** — no uniform direction: desirability picked MORE neurons on cicids (both arms) and on
B15-AC (216 vs 134), FEWER on Wc-CTRL (181 vs 212); on ciciot it pushed bits up (B15-AC 80b every seed vs ~51b;
Wc 77b vs 67b). The unswt "~21% smaller genomes" observation does not transfer.

### B05-AC extension (IDSAGGX, addendum 04/10) — INTERIM, DESCRIPTIVE ONLY, 5/10 flows

Not part of the primary or secondary. Completed: zs r20414/15/16, desir r20414/15. No inference.
```
aggr  | n | F1 %         | FPR %        | Acc %        | genome (GA best_f1)     | GA gens      | wall h     | seeds
------+---+--------------+--------------+--------------+-------------------------+--------------+------------+------------------
zs    | 3 | 92.89±0.27   | 9.23±0.87    | 96.49±0.13   | 111±79 n x 44.5±2.1 b   | 83.3±5.8     | 1.00±0.58  | 20414,20415,20416
desir | 2 | 93.01±0.03   | 7.21±1.05    | 96.49±0.05   | 110±62 n x 80.0±0.0 b   | 115.0±21.2   | 2.36±1.85  | 20414,20415

paired desir - zs | seed  | F1 zs / desir / d      | FPR zs / desir / d      | Acc zs / desir / d     | flows
------------------+-------+------------------------+-------------------------+------------------------+----------
B05-AC            | 20414 | 92.58 / 93.00 / +0.414 | 10.01 / 6.47 / -3.535   | 96.34 / 96.46 / +0.118 | 6420,6421
B05-AC            | 20415 | 93.04 / 93.03 / -0.003 | 8.29 / 7.95 / -0.340    | 96.54 / 96.52 / -0.012 | 6422,6423

B05-AC vs B15-AC (same seed, same aggregation; F1/FPR/Acc)
aggr  | seed  | B05-AC              | B15-AC              | d F1 / d FPR / d Acc
------+-------+---------------------+---------------------+-----------------------
zs    | 20414 | 92.58/10.01/96.34   | 92.61/9.88/96.35    | -0.029 / +0.125 / -0.012
zs    | 20415 | 93.04/8.29/96.54    | 92.78/9.06/96.41    | +0.262 / -0.775 / +0.120
zs    | 20416 | 93.06/9.40/96.58    | 92.84/8.17/96.42    | +0.220 / +1.230 / +0.159
desir | 20414 | 93.00/6.47/96.46    | 93.09/7.92/96.55    | -0.093 / -1.445 / -0.096
desir | 20415 | 93.03/7.95/96.52    | 92.99/7.17/96.48    | +0.041 / +0.785 / +0.047
```
Direction on 2 seeds matches B15-AC (desirability lower FPR), but seed 20415 is near zero. Re-read at 10/10.

### Pareto mining (GA, all 5 genome types x 7 modes x 2 arms x 2 aggr, seed means; non-dominated on F1 up / FPR down)

```
## cicids non-dominated (GA, mean F1 up / mean FPR down) over 2 arms x 2 aggs x 5 gt x 7 modes = 140 cells
  Wa-CTRL  zs    best_fitness train_cal            F1 99.53±0.07 FPR 0.16±0.06 Acc 99.70±0.04
  Wa-CTRL  zs    best_f1      train_cal            F1 99.52±0.07 FPR 0.15±0.06 Acc 99.70±0.05
  Wa-CTRL  zs    best_acc     train_cal            F1 99.52±0.07 FPR 0.15±0.06 Acc 99.70±0.05
  CE20     zs    best_f1      train_cal            F1 99.52±0.06 FPR 0.09±0.04 Acc 99.70±0.04
  CE20     zs    best_acc     train_cal            F1 99.52±0.06 FPR 0.09±0.04 Acc 99.70±0.04
  CE20     zs    best_f1      empirical_cumulative F1 99.52±0.06 FPR 0.07±0.04 Acc 99.70±0.04
  CE20     zs    best_fitness empirical_cumulative F1 99.51±0.08 FPR 0.07±0.02 Acc 99.69±0.05
  CE20     zs    best_acc     empirical            F1 99.45±0.08 FPR 0.07±0.03 Acc 99.65±0.05
  CE20     zs    best_fitness empirical            F1 99.41±0.14 FPR 0.06±0.02 Acc 99.63±0.09
  CE20     zs    best_fpr     empirical            F1 99.37±0.19 FPR 0.06±0.02 Acc 99.61±0.12
  Wa-CTRL  zs    best_fitness empirical            F1 99.08±0.38 FPR 0.06±0.01 Acc 99.42±0.24
## ciciot non-dominated (GA, mean F1 up / mean FPR down) over 2 arms x 2 aggs x 5 gt x 7 modes = 140 cells
  B15-AC   desir best_fitness val_cal              F1 93.04±0.14 FPR 7.04±0.41 Acc 96.50±0.08
  B15-AC   desir best_fpr     val_cal              F1 93.03±0.16 FPR 7.00±0.38 Acc 96.49±0.09
  B15-AC   desir best_f1      val_cal              F1 93.00±0.11 FPR 6.92±0.74 Acc 96.47±0.06
  B15-AC   desir best_acc     val_cal              F1 93.00±0.11 FPR 6.92±0.74 Acc 96.47±0.06
  Wc-CTRL  zs    best_acc     empirical_cumulative F1 92.67±0.12 FPR 5.33±0.58 Acc 96.24±0.08
  Wc-CTRL  zs    best_f1      empirical_cumulative F1 92.66±0.11 FPR 5.30±0.52 Acc 96.23±0.07
  B15-AC   desir best_fpr     empirical_cumulative F1 92.63±0.25 FPR 4.40±0.43 Acc 96.18±0.15
  B15-AC   desir best_fitness empirical_cumulative F1 92.62±0.24 FPR 4.34±0.38 Acc 96.17±0.15
  Wc-CTRL  zs    best_fpr     fixed_05             F1 86.51±1.16 FPR 0.86±0.09 Acc 92.08±0.82
  Wc-CTRL  zs    best_fitness fixed_05             F1 86.41±1.27 FPR 0.86±0.10 Acc 92.01±0.90
  Wc-CTRL  zs    best_f1      fixed_05             F1 86.40±1.28 FPR 0.85±0.09 Acc 92.01±0.92
  B15-AC   desir best_f1      fixed_05             F1 85.90±0.74 FPR 0.76±0.08 Acc 91.65±0.54
  B15-AC   desir best_acc     fixed_05             F1 85.90±0.74 FPR 0.76±0.08 Acc 91.65±0.54
  B15-AC   desir best_fpr     fixed_05             F1 85.87±0.77 FPR 0.76±0.09 Acc 91.63±0.56
  Wc-CTRL  zs    best_f1      empirical            F1 68.30±0.46 FPR 0.01±0.01 Acc 75.45±0.51
  Wc-CTRL  zs    best_ce      empirical            F1 68.18±0.35 FPR 0.01±0.01 Acc 75.31±0.39
  Wc-CTRL  desir best_ce      empirical            F1 67.83±1.84 FPR 0.01±0.00 Acc 74.90±2.04
  Wc-CTRL  desir best_fitness empirical            F1 67.83±1.84 FPR 0.01±0.00 Acc 74.90±2.04
  B15-AC   desir best_ce      empirical            F1 66.08±0.90 FPR 0.01±0.01 Acc 72.93±1.04
  B15-AC   desir best_fitness empirical            F1 65.92±1.03 FPR 0.00±0.00 Acc 72.74±1.18
  B15-AC   desir best_f1      empirical            F1 65.76±0.78 FPR 0.00±0.00 Acc 72.57±0.90
  B15-AC   desir best_acc     empirical            F1 65.76±0.78 FPR 0.00±0.00 Acc 72.57±0.90
```
Post-hoc best-of-140 cells per dataset: operating-point context, not claims. Every non-dominated cicids cell is zscore;
on ciciot the val_cal top and the low-FPR empirical_cumulative points are B15-AC desirability, with Wc-CTRL zscore
between them. train_cal is fit on train; platt/beta/empirical/empirical_cumulative/val_cal on VAL (Protocol v2); all TEST-reported.

### Rule-7 full breakdown (IDSAGG, all 5 genome types x 7 modes, Grid vs GA, mean±SD %)

<details><summary>40 tables (2 datasets x 2 arms x 2 aggregations x 5 genome types)</summary>

```
### cicids Wa-CTRL zs  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 200.00±173.21 neurons | 33.60±0.89 bits
  GA Neurons  : 221.00±169.89 neurons | 33.95±0.07 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.33±0.01  | 99.52±0.07  | 0.28±0.02   | 0.15±0.06   | 99.58±0.01  | 99.70±0.05 
  fixed_05             | 98.98±0.03  | 99.18±0.05  | 0.63±0.02   | 0.49±0.05   | 99.35±0.02  | 99.48±0.03 
  platt                | 99.31±0.02  | 99.40±0.04  | 0.31±0.01   | 0.27±0.03   | 99.56±0.01  | 99.62±0.02 
  beta                 | 99.30±0.03  | 99.45±0.08  | 0.26±0.02   | 0.21±0.06   | 99.56±0.02  | 99.65±0.05 
  empirical            | 98.86±0.30  | 99.04±0.40  | 0.14±0.05   | 0.06±0.02   | 99.29±0.18  | 99.40±0.25 
  empirical_cumulative | 99.33±0.01  | 99.52±0.07  | 0.28±0.02   | 0.15±0.06   | 99.58±0.01  | 99.70±0.05 
  val_cal              | 99.33±0.01  | 99.52±0.07  | 0.28±0.02   | 0.15±0.06   | 99.58±0.01  | 99.70±0.05 
best_fpr  (runs: 5/5)
  Grid Search : 380.00±83.67 neurons | 31.60±2.19 bits
  GA Neurons  : 220.20±171.04 neurons | 33.94±0.08 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.21±0.08  | 99.51±0.07  | 0.24±0.08   | 0.16±0.05   | 99.50±0.05  | 99.69±0.04 
  fixed_05             | 98.92±0.06  | 99.15±0.05  | 0.65±0.04   | 0.51±0.04   | 99.31±0.04  | 99.46±0.03 
  platt                | 99.16±0.06  | 99.40±0.04  | 0.36±0.02   | 0.27±0.04   | 99.47±0.04  | 99.62±0.03 
  beta                 | 99.18±0.07  | 99.46±0.05  | 0.29±0.02   | 0.19±0.05   | 99.48±0.04  | 99.66±0.03 
  empirical            | 98.52±0.26  | 99.13±0.43  | 0.05±0.01   | 0.06±0.01   | 99.08±0.16  | 99.46±0.26 
  empirical_cumulative | 99.21±0.08  | 99.51±0.07  | 0.20±0.03   | 0.14±0.05   | 99.50±0.05  | 99.69±0.05 
  val_cal              | 99.21±0.08  | 99.51±0.07  | 0.24±0.07   | 0.16±0.05   | 99.50±0.05  | 99.69±0.04 
best_acc  (runs: 5/5)
  Grid Search : 200.00±173.21 neurons | 33.60±0.89 bits
  GA Neurons  : 221.00±169.89 neurons | 33.95±0.07 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.33±0.01  | 99.52±0.07  | 0.28±0.02   | 0.15±0.06   | 99.58±0.01  | 99.70±0.05 
  fixed_05             | 98.98±0.03  | 99.18±0.05  | 0.63±0.02   | 0.49±0.05   | 99.35±0.02  | 99.48±0.03 
  platt                | 99.31±0.02  | 99.40±0.04  | 0.31±0.01   | 0.27±0.03   | 99.56±0.01  | 99.62±0.02 
  beta                 | 99.30±0.03  | 99.45±0.08  | 0.26±0.02   | 0.21±0.06   | 99.56±0.02  | 99.65±0.05 
  empirical            | 98.86±0.30  | 99.04±0.40  | 0.14±0.05   | 0.06±0.02   | 99.29±0.18  | 99.40±0.25 
  empirical_cumulative | 99.33±0.01  | 99.52±0.07  | 0.28±0.02   | 0.15±0.06   | 99.58±0.01  | 99.70±0.05 
  val_cal              | 99.33±0.01  | 99.52±0.07  | 0.28±0.02   | 0.15±0.06   | 99.58±0.01  | 99.70±0.05 
best_ce  (runs: 5/5)
  Grid Search : 220.00±109.54 neurons | 34.00±0.00 bits
  GA Neurons  : 217.20±172.32 neurons | 33.94±0.09 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.26±0.03  | 99.52±0.07  | 0.28±0.04   | 0.15±0.06   | 99.53±0.02  | 99.70±0.05 
  fixed_05             | 98.97±0.03  | 99.13±0.10  | 0.62±0.02   | 0.53±0.08   | 99.35±0.02  | 99.45±0.06 
  platt                | 99.23±0.05  | 99.41±0.06  | 0.33±0.01   | 0.27±0.05   | 99.52±0.03  | 99.63±0.04 
  beta                 | 99.23±0.04  | 99.46±0.06  | 0.28±0.01   | 0.20±0.05   | 99.52±0.03  | 99.66±0.04 
  empirical            | 98.77±0.39  | 99.08±0.44  | 0.12±0.08   | 0.07±0.02   | 99.23±0.24  | 99.42±0.27 
  empirical_cumulative | 99.25±0.04  | 99.52±0.07  | 0.28±0.04   | 0.15±0.06   | 99.53±0.02  | 99.70±0.05 
  val_cal              | 99.25±0.04  | 99.52±0.08  | 0.32±0.05   | 0.15±0.06   | 99.53±0.03  | 99.70±0.05 
best_fitness  (runs: 5/5)
  Grid Search : 320.00±204.94 neurons | 34.00±0.00 bits
  GA Neurons  : 222.80±173.11 neurons | 33.95±0.07 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.32±0.01  | 99.53±0.07  | 0.29±0.01   | 0.16±0.06   | 99.57±0.01  | 99.70±0.04 
  fixed_05             | 99.01±0.03  | 99.19±0.05  | 0.61±0.03   | 0.48±0.04   | 99.37±0.02  | 99.48±0.03 
  platt                | 99.31±0.02  | 99.41±0.05  | 0.31±0.01   | 0.27±0.04   | 99.56±0.01  | 99.63±0.03 
  beta                 | 99.27±0.04  | 99.47±0.07  | 0.28±0.01   | 0.20±0.06   | 99.54±0.02  | 99.67±0.05 
  empirical            | 98.53±0.45  | 99.08±0.38  | 0.10±0.06   | 0.06±0.01   | 99.09±0.27  | 99.42±0.24 
  empirical_cumulative | 99.32±0.01  | 99.53±0.07  | 0.30±0.01   | 0.16±0.06   | 99.57±0.01  | 99.70±0.04 
  val_cal              | 99.32±0.01  | 99.53±0.07  | 0.30±0.01   | 0.16±0.06   | 99.57±0.01  | 99.70±0.04 
### cicids CE20 zs  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 160.00±89.44 neurons | 33.60±0.89 bits
  GA Neurons  : 176.20±79.78 neurons | 33.46±0.83 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.30±0.03  | 99.52±0.06  | 0.25±0.03   | 0.09±0.04   | 99.55±0.02  | 99.70±0.04 
  fixed_05             | 99.02±0.06  | 99.16±0.06  | 0.60±0.05   | 0.51±0.04   | 99.37±0.04  | 99.46±0.04 
  platt                | 99.26±0.04  | 99.38±0.07  | 0.33±0.01   | 0.29±0.04   | 99.53±0.02  | 99.61±0.04 
  beta                 | 99.28±0.05  | 99.41±0.09  | 0.28±0.02   | 0.25±0.06   | 99.54±0.03  | 99.63±0.06 
  empirical            | 98.97±0.10  | 99.45±0.08  | 0.13±0.04   | 0.07±0.03   | 99.36±0.06  | 99.65±0.05 
  empirical_cumulative | 99.28±0.04  | 99.52±0.06  | 0.23±0.04   | 0.07±0.04   | 99.55±0.03  | 99.70±0.04 
  val_cal              | 99.30±0.03  | 99.52±0.07  | 0.26±0.02   | 0.11±0.04   | 99.56±0.02  | 99.70±0.04 
best_fpr  (runs: 5/5)
  Grid Search : 300.00±158.11 neurons | 28.00±5.48 bits
  GA Neurons  : 179.80±85.74 neurons | 33.49±0.84 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.11±0.13  | 99.49±0.09  | 0.33±0.10   | 0.12±0.05   | 99.44±0.08  | 99.68±0.06 
  fixed_05             | 98.84±0.22  | 99.11±0.08  | 0.69±0.15   | 0.54±0.06   | 99.26±0.14  | 99.44±0.05 
  platt                | 99.08±0.12  | 99.36±0.07  | 0.38±0.04   | 0.29±0.04   | 99.42±0.08  | 99.60±0.04 
  beta                 | 99.06±0.14  | 99.38±0.07  | 0.32±0.01   | 0.27±0.04   | 99.41±0.09  | 99.61±0.04 
  empirical            | 98.69±0.37  | 99.37±0.19  | 0.09±0.04   | 0.06±0.02   | 99.18±0.23  | 99.61±0.12 
  empirical_cumulative | 99.03±0.20  | 99.48±0.09  | 0.17±0.05   | 0.07±0.02   | 99.39±0.12  | 99.67±0.06 
  val_cal              | 99.11±0.12  | 99.49±0.09  | 0.32±0.08   | 0.12±0.05   | 99.44±0.08  | 99.68±0.06 
best_acc  (runs: 5/5)
  Grid Search : 160.00±89.44 neurons | 33.60±0.89 bits
  GA Neurons  : 176.20±79.78 neurons | 33.46±0.83 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.30±0.03  | 99.52±0.06  | 0.25±0.03   | 0.09±0.04   | 99.55±0.02  | 99.70±0.04 
  fixed_05             | 99.02±0.06  | 99.16±0.06  | 0.60±0.05   | 0.51±0.04   | 99.37±0.04  | 99.46±0.04 
  platt                | 99.26±0.04  | 99.38±0.07  | 0.33±0.01   | 0.29±0.04   | 99.53±0.02  | 99.61±0.04 
  beta                 | 99.28±0.05  | 99.41±0.09  | 0.28±0.02   | 0.25±0.06   | 99.54±0.03  | 99.63±0.06 
  empirical            | 98.97±0.10  | 99.45±0.08  | 0.13±0.04   | 0.07±0.03   | 99.36±0.06  | 99.65±0.05 
  empirical_cumulative | 99.28±0.04  | 99.52±0.06  | 0.23±0.04   | 0.07±0.04   | 99.55±0.03  | 99.70±0.04 
  val_cal              | 99.30±0.03  | 99.52±0.07  | 0.26±0.02   | 0.11±0.04   | 99.56±0.02  | 99.70±0.04 
best_ce  (runs: 5/5)
  Grid Search : 320.00±130.38 neurons | 34.00±0.00 bits
  GA Neurons  : 178.00±81.61 neurons | 33.50±0.85 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.25±0.03  | 99.51±0.07  | 0.28±0.03   | 0.09±0.03   | 99.53±0.02  | 99.69±0.04 
  fixed_05             | 99.00±0.04  | 99.15±0.06  | 0.60±0.02   | 0.51±0.04   | 99.36±0.02  | 99.46±0.04 
  platt                | 99.22±0.02  | 99.37±0.07  | 0.33±0.01   | 0.28±0.05   | 99.51±0.01  | 99.60±0.04 
  beta                 | 99.24±0.02  | 99.40±0.08  | 0.28±0.01   | 0.26±0.05   | 99.52±0.01  | 99.62±0.05 
  empirical            | 98.55±0.24  | 99.45±0.09  | 0.08±0.03   | 0.07±0.03   | 99.10±0.15  | 99.65±0.06 
  empirical_cumulative | 99.24±0.03  | 99.51±0.07  | 0.26±0.05   | 0.09±0.03   | 99.52±0.02  | 99.69±0.04 
  val_cal              | 99.25±0.03  | 99.51±0.07  | 0.30±0.05   | 0.09±0.03   | 99.53±0.02  | 99.69±0.04 
best_fitness  (runs: 5/5)
  Grid Search : 200.00±141.42 neurons | 33.60±0.89 bits
  GA Neurons  : 178.00±83.01 neurons | 33.49±0.84 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.27±0.02  | 99.52±0.07  | 0.24±0.02   | 0.08±0.02   | 99.54±0.01  | 99.70±0.05 
  fixed_05             | 99.05±0.05  | 99.13±0.09  | 0.57±0.04   | 0.52±0.06   | 99.40±0.03  | 99.45±0.06 
  platt                | 99.22±0.03  | 99.37±0.07  | 0.34±0.01   | 0.29±0.04   | 99.51±0.02  | 99.60±0.04 
  beta                 | 99.25±0.04  | 99.42±0.07  | 0.29±0.02   | 0.24±0.04   | 99.53±0.02  | 99.63±0.04 
  empirical            | 98.76±0.48  | 99.41±0.14  | 0.10±0.06   | 0.06±0.02   | 99.22±0.29  | 99.63±0.09 
  empirical_cumulative | 99.25±0.01  | 99.51±0.08  | 0.20±0.02   | 0.07±0.02   | 99.53±0.01  | 99.69±0.05 
  val_cal              | 99.27±0.02  | 99.51±0.08  | 0.29±0.06   | 0.09±0.04   | 99.54±0.02  | 99.69±0.05 
### cicids Wa-CTRL desir  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 340.00±134.16 neurons | 34.00±0.00 bits
  GA Neurons  : 286.20±115.40 neurons | 34.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.33±0.01  | 99.42±0.06  | 0.30±0.01   | 0.25±0.05   | 99.58±0.01  | 99.63±0.04 
  fixed_05             | 99.04±0.04  | 99.18±0.10  | 0.59±0.02   | 0.50±0.07   | 99.39±0.02  | 99.48±0.06 
  platt                | 99.33±0.01  | 99.36±0.02  | 0.30±0.01   | 0.32±0.02   | 99.57±0.01  | 99.60±0.01 
  beta                 | 99.26±0.02  | 99.39±0.04  | 0.28±0.01   | 0.26±0.02   | 99.53±0.01  | 99.61±0.03 
  empirical            | 98.44±0.16  | 98.88±0.32  | 0.06±0.02   | 0.12±0.03   | 99.03±0.10  | 99.30±0.20 
  empirical_cumulative | 99.33±0.01  | 99.42±0.06  | 0.30±0.01   | 0.25±0.05   | 99.58±0.01  | 99.63±0.04 
  val_cal              | 99.33±0.01  | 99.42±0.06  | 0.30±0.01   | 0.25±0.05   | 99.57±0.01  | 99.63±0.04 
best_fpr  (runs: 5/5)
  Grid Search : 280.00±83.67 neurons | 31.60±2.19 bits
  GA Neurons  : 287.20±116.64 neurons | 34.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.20±0.08  | 99.40±0.07  | 0.20±0.04   | 0.24±0.06   | 99.50±0.05  | 99.62±0.04 
  fixed_05             | 98.91±0.10  | 99.17±0.08  | 0.67±0.08   | 0.51±0.06   | 99.31±0.07  | 99.47±0.05 
  platt                | 99.14±0.10  | 99.35±0.06  | 0.36±0.03   | 0.33±0.05   | 99.46±0.06  | 99.59±0.04 
  beta                 | 99.17±0.10  | 99.35±0.04  | 0.29±0.02   | 0.28±0.02   | 99.47±0.06  | 99.59±0.03 
  empirical            | 98.62±0.29  | 98.91±0.33  | 0.06±0.02   | 0.12±0.03   | 99.14±0.18  | 99.32±0.20 
  empirical_cumulative | 99.20±0.08  | 99.40±0.07  | 0.23±0.09   | 0.24±0.07   | 99.50±0.05  | 99.62±0.04 
  val_cal              | 99.20±0.08  | 99.40±0.07  | 0.23±0.09   | 0.27±0.07   | 99.50±0.05  | 99.62±0.04 
best_acc  (runs: 5/5)
  Grid Search : 340.00±134.16 neurons | 34.00±0.00 bits
  GA Neurons  : 286.20±115.40 neurons | 34.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.33±0.01  | 99.42±0.06  | 0.30±0.01   | 0.25±0.05   | 99.58±0.01  | 99.63±0.04 
  fixed_05             | 99.04±0.04  | 99.18±0.10  | 0.59±0.02   | 0.50±0.07   | 99.39±0.02  | 99.48±0.06 
  platt                | 99.33±0.01  | 99.36±0.02  | 0.30±0.01   | 0.32±0.02   | 99.57±0.01  | 99.60±0.01 
  beta                 | 99.26±0.02  | 99.39±0.04  | 0.28±0.01   | 0.26±0.02   | 99.53±0.01  | 99.61±0.03 
  empirical            | 98.44±0.16  | 98.88±0.32  | 0.06±0.02   | 0.12±0.03   | 99.03±0.10  | 99.30±0.20 
  empirical_cumulative | 99.33±0.01  | 99.42±0.06  | 0.30±0.01   | 0.25±0.05   | 99.58±0.01  | 99.63±0.04 
  val_cal              | 99.33±0.01  | 99.42±0.06  | 0.30±0.01   | 0.25±0.05   | 99.57±0.01  | 99.63±0.04 
best_ce  (runs: 5/5)
  Grid Search : 300.00±141.42 neurons | 34.00±0.00 bits
  GA Neurons  : 288.20±117.28 neurons | 34.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.27±0.05  | 99.41±0.07  | 0.30±0.02   | 0.26±0.06   | 99.54±0.03  | 99.63±0.04 
  fixed_05             | 99.01±0.06  | 99.19±0.09  | 0.59±0.05   | 0.49±0.07   | 99.37±0.04  | 99.49±0.06 
  platt                | 99.24±0.05  | 99.36±0.02  | 0.32±0.02   | 0.32±0.02   | 99.52±0.03  | 99.60±0.01 
  beta                 | 99.24±0.03  | 99.37±0.05  | 0.28±0.01   | 0.26±0.02   | 99.52±0.02  | 99.60±0.03 
  empirical            | 98.52±0.18  | 98.68±0.23  | 0.08±0.03   | 0.08±0.02   | 99.08±0.11  | 99.18±0.14 
  empirical_cumulative | 99.27±0.05  | 99.41±0.06  | 0.29±0.03   | 0.27±0.06   | 99.54±0.03  | 99.63±0.04 
  val_cal              | 99.27±0.05  | 99.41±0.06  | 0.30±0.02   | 0.27±0.06   | 99.54±0.03  | 99.63±0.04 
best_fitness  (runs: 5/5)
  Grid Search : 300.00±141.42 neurons | 34.00±0.00 bits
  GA Neurons  : 288.40±117.30 neurons | 34.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.27±0.05  | 99.42±0.06  | 0.30±0.02   | 0.26±0.06   | 99.54±0.03  | 99.63±0.04 
  fixed_05             | 99.01±0.06  | 99.19±0.10  | 0.59±0.05   | 0.49±0.07   | 99.37±0.04  | 99.49±0.06 
  platt                | 99.24±0.05  | 99.37±0.02  | 0.32±0.02   | 0.31±0.02   | 99.52±0.03  | 99.60±0.02 
  beta                 | 99.24±0.03  | 99.37±0.05  | 0.28±0.01   | 0.27±0.02   | 99.52±0.02  | 99.60±0.03 
  empirical            | 98.52±0.18  | 98.76±0.18  | 0.08±0.03   | 0.09±0.01   | 99.08±0.11  | 99.23±0.11 
  empirical_cumulative | 99.27±0.05  | 99.42±0.06  | 0.29±0.03   | 0.27±0.06   | 99.54±0.03  | 99.63±0.04 
  val_cal              | 99.27±0.05  | 99.42±0.06  | 0.30±0.02   | 0.27±0.06   | 99.54±0.03  | 99.63±0.04 
### cicids CE20 desir  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 260.00±114.02 neurons | 34.00±0.00 bits
  GA Neurons  : 241.80±101.67 neurons | 33.93±0.15 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.29±0.03  | 99.46±0.08  | 0.25±0.05   | 0.10±0.04   | 99.55±0.02  | 99.66±0.05 
  fixed_05             | 99.00±0.04  | 99.12±0.11  | 0.61±0.03   | 0.55±0.10   | 99.36±0.02  | 99.44±0.07 
  platt                | 99.25±0.06  | 99.32±0.08  | 0.33±0.02   | 0.34±0.05   | 99.53±0.04  | 99.57±0.05 
  beta                 | 99.25±0.03  | 99.38±0.06  | 0.27±0.03   | 0.20±0.08   | 99.53±0.02  | 99.61±0.04 
  empirical            | 98.60±0.07  | 99.15±0.21  | 0.09±0.04   | 0.08±0.03   | 99.12±0.04  | 99.46±0.13 
  empirical_cumulative | 99.29±0.02  | 99.45±0.08  | 0.25±0.05   | 0.10±0.04   | 99.55±0.01  | 99.66±0.05 
  val_cal              | 99.29±0.02  | 99.45±0.08  | 0.26±0.05   | 0.10±0.04   | 99.55±0.01  | 99.66±0.05 
best_fpr  (runs: 5/5)
  Grid Search : 220.00±83.67 neurons | 30.00±2.83 bits
  GA Neurons  : 242.00±99.85 neurons | 33.93±0.15 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.17±0.09  | 99.44±0.07  | 0.26±0.07   | 0.13±0.07   | 99.47±0.05  | 99.65±0.04 
  fixed_05             | 98.92±0.03  | 99.09±0.11  | 0.65±0.02   | 0.57±0.10   | 99.32±0.02  | 99.42±0.07 
  platt                | 99.10±0.11  | 99.30±0.07  | 0.38±0.04   | 0.35±0.03   | 99.43±0.07  | 99.56±0.05 
  beta                 | 99.15±0.11  | 99.36±0.04  | 0.30±0.02   | 0.22±0.09   | 99.46±0.07  | 99.59±0.02 
  empirical            | 98.84±0.11  | 99.10±0.21  | 0.08±0.01   | 0.11±0.06   | 99.28±0.07  | 99.44±0.13 
  empirical_cumulative | 99.11±0.11  | 99.43±0.08  | 0.16±0.03   | 0.10±0.03   | 99.44±0.07  | 99.64±0.05 
  val_cal              | 99.17±0.09  | 99.44±0.07  | 0.26±0.08   | 0.13±0.07   | 99.47±0.06  | 99.65±0.04 
best_acc  (runs: 5/5)
  Grid Search : 260.00±114.02 neurons | 34.00±0.00 bits
  GA Neurons  : 241.80±101.67 neurons | 33.93±0.15 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.29±0.03  | 99.46±0.08  | 0.25±0.05   | 0.10±0.04   | 99.55±0.02  | 99.66±0.05 
  fixed_05             | 99.00±0.04  | 99.12±0.11  | 0.61±0.03   | 0.55±0.10   | 99.36±0.02  | 99.44±0.07 
  platt                | 99.25±0.06  | 99.32±0.08  | 0.33±0.02   | 0.34±0.05   | 99.53±0.04  | 99.57±0.05 
  beta                 | 99.25±0.03  | 99.38±0.06  | 0.27±0.03   | 0.20±0.08   | 99.53±0.02  | 99.61±0.04 
  empirical            | 98.60±0.07  | 99.15±0.21  | 0.09±0.04   | 0.08±0.03   | 99.12±0.04  | 99.46±0.13 
  empirical_cumulative | 99.29±0.02  | 99.45±0.08  | 0.25±0.05   | 0.10±0.04   | 99.55±0.01  | 99.66±0.05 
  val_cal              | 99.29±0.02  | 99.45±0.08  | 0.26±0.05   | 0.10±0.04   | 99.55±0.01  | 99.66±0.05 
best_ce  (runs: 5/5)
  Grid Search : 440.00±54.77 neurons | 34.00±0.00 bits
  GA Neurons  : 247.40±106.21 neurons | 33.94±0.14 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.27±0.04  | 99.42±0.07  | 0.28±0.04   | 0.16±0.08   | 99.54±0.03  | 99.64±0.05 
  fixed_05             | 99.00±0.03  | 99.11±0.11  | 0.61±0.03   | 0.55±0.09   | 99.36±0.02  | 99.43±0.07 
  platt                | 99.24±0.05  | 99.32±0.07  | 0.33±0.01   | 0.33±0.02   | 99.52±0.03  | 99.57±0.04 
  beta                 | 99.23±0.02  | 99.37±0.05  | 0.28±0.01   | 0.21±0.08   | 99.52±0.01  | 99.60±0.03 
  empirical            | 98.33±0.30  | 98.96±0.18  | 0.06±0.01   | 0.07±0.02   | 98.96±0.18  | 99.35±0.11 
  empirical_cumulative | 99.26±0.05  | 99.42±0.08  | 0.28±0.04   | 0.10±0.04   | 99.53±0.03  | 99.63±0.05 
  val_cal              | 99.26±0.04  | 99.42±0.07  | 0.28±0.04   | 0.16±0.08   | 99.53±0.02  | 99.64±0.05 
best_fitness  (runs: 5/5)
  Grid Search : 440.00±54.77 neurons | 34.00±0.00 bits
  GA Neurons  : 247.40±106.21 neurons | 33.94±0.14 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 99.27±0.04  | 99.46±0.08  | 0.28±0.04   | 0.11±0.03   | 99.54±0.03  | 99.66±0.05 
  fixed_05             | 99.00±0.03  | 99.11±0.10  | 0.61±0.03   | 0.55±0.09   | 99.36±0.02  | 99.43±0.06 
  platt                | 99.24±0.05  | 99.31±0.07  | 0.33±0.01   | 0.34±0.04   | 99.52±0.03  | 99.56±0.04 
  beta                 | 99.23±0.02  | 99.38±0.07  | 0.28±0.01   | 0.20±0.09   | 99.52±0.01  | 99.61±0.05 
  empirical            | 98.33±0.30  | 99.05±0.24  | 0.06±0.01   | 0.13±0.11   | 98.96±0.18  | 99.40±0.15 
  empirical_cumulative | 99.26±0.05  | 99.46±0.08  | 0.28±0.04   | 0.10±0.03   | 99.53±0.03  | 99.66±0.05 
  val_cal              | 99.26±0.04  | 99.46±0.08  | 0.28±0.04   | 0.11±0.02   | 99.53±0.02  | 99.66±0.05 
### ciciot Wc-CTRL zs  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 200.00±61.24 neurons | 54.40±8.76 bits
  GA Neurons  : 212.40±39.78 neurons | 67.20±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.60±0.38  | 92.94±0.06  | 16.72±1.21  | 8.18±0.73   | 95.50±0.23  | 96.48±0.05 
  fixed_05             | 81.49±0.25  | 86.40±1.28  | 1.00±0.05   | 0.85±0.09   | 88.28±0.20  | 92.01±0.92 
  platt                | 90.05±0.20  | 92.90±0.14  | 11.31±0.62  | 8.45±0.94   | 94.94±0.12  | 96.46±0.05 
  beta                 | 90.52±0.32  | 92.72±0.19  | 15.47±0.26  | 10.63±0.97  | 95.40±0.17  | 96.44±0.07 
  empirical            | 70.13±1.66  | 68.30±0.46  | 0.00±0.00   | 0.01±0.01   | 77.42±1.80  | 75.45±0.51 
  empirical_cumulative | 89.96±0.19  | 92.66±0.11  | 10.74±0.97  | 5.30±0.52   | 94.86±0.14  | 96.23±0.07 
  val_cal              | 90.59±0.37  | 92.93±0.06  | 16.78±1.27  | 7.69±0.85   | 95.50±0.22  | 96.46±0.05 
best_fpr  (runs: 5/5)
  Grid Search : 200.00±50.00 neurons | 97.60±2.19 bits
  GA Neurons  : 211.60±39.61 neurons | 67.20±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.10±0.13  | 92.85±0.04  | 10.31±0.16  | 7.98±1.14   | 94.93±0.08  | 96.42±0.05 
  fixed_05             | 79.69±0.29  | 86.51±1.16  | 0.74±0.04   | 0.86±0.09   | 86.75±0.25  | 92.08±0.82 
  platt                | 89.95±0.22  | 92.83±0.07  | 16.22±0.48  | 8.40±0.81   | 95.11±0.12  | 96.42±0.02 
  beta                 | 89.62±0.25  | 92.66±0.17  | 18.98±0.35  | 10.71±1.00  | 95.06±0.13  | 96.40±0.06 
  empirical            | 62.25±2.73  | 68.15±0.25  | 0.01±0.01   | 0.01±0.01   | 68.40±3.28  | 75.28±0.28 
  empirical_cumulative | 89.89±0.11  | 92.63±0.05  | 9.14±0.32   | 5.60±0.72   | 94.75±0.07  | 96.22±0.05 
  val_cal              | 90.18±0.18  | 92.85±0.04  | 12.78±0.76  | 7.84±0.73   | 95.08±0.11  | 96.42±0.04 
best_acc  (runs: 5/5)
  Grid Search : 200.00±61.24 neurons | 54.40±8.76 bits
  GA Neurons  : 213.20±39.71 neurons | 67.20±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.60±0.38  | 92.93±0.06  | 16.72±1.21  | 8.14±0.70   | 95.50±0.23  | 96.47±0.05 
  fixed_05             | 81.49±0.25  | 86.40±1.28  | 1.00±0.05   | 0.85±0.09   | 88.28±0.20  | 92.00±0.92 
  platt                | 90.05±0.20  | 92.90±0.14  | 11.31±0.62  | 8.46±0.94   | 94.94±0.12  | 96.46±0.05 
  beta                 | 90.52±0.32  | 92.73±0.19  | 15.47±0.26  | 10.61±0.96  | 95.40±0.17  | 96.44±0.07 
  empirical            | 70.13±1.66  | 68.27±0.43  | 0.00±0.00   | 0.01±0.01   | 77.42±1.80  | 75.41±0.48 
  empirical_cumulative | 89.96±0.19  | 92.67±0.12  | 10.74±0.97  | 5.33±0.58   | 94.86±0.14  | 96.24±0.08 
  val_cal              | 90.59±0.37  | 92.93±0.06  | 16.78±1.27  | 7.70±0.86   | 95.50±0.22  | 96.46±0.05 
best_ce  (runs: 5/5)
  Grid Search : 190.00±41.83 neurons | 70.40±8.76 bits
  GA Neurons  : 210.60±41.79 neurons | 67.20±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.27±0.20  | 92.82±0.04  | 15.08±0.89  | 8.37±0.91   | 95.24±0.14  | 96.42±0.04 
  fixed_05             | 81.59±0.33  | 86.43±1.26  | 0.95±0.04   | 0.87±0.10   | 88.36±0.27  | 92.03±0.89 
  platt                | 90.23±0.18  | 92.80±0.07  | 12.80±1.00  | 8.51±0.81   | 95.11±0.09  | 96.41±0.03 
  beta                 | 90.24±0.31  | 92.66±0.16  | 16.12±0.94  | 10.69±0.97  | 95.27±0.14  | 96.40±0.06 
  empirical            | 67.27±0.67  | 68.18±0.35  | 0.01±0.01   | 0.01±0.01   | 74.29±0.76  | 75.31±0.39 
  empirical_cumulative | 89.88±0.14  | 92.54±0.06  | 9.48±0.74   | 5.34±0.70   | 94.76±0.11  | 96.16±0.05 
  val_cal              | 90.30±0.22  | 92.83±0.05  | 14.46±1.24  | 8.02±0.38   | 95.23±0.17  | 96.41±0.03 
best_fitness  (runs: 5/5)
  Grid Search : 210.00±41.83 neurons | 67.20±7.16 bits
  GA Neurons  : 212.60±41.34 neurons | 67.20±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.29±0.16  | 92.88±0.07  | 15.34±0.49  | 8.22±0.91   | 95.26±0.10  | 96.45±0.05 
  fixed_05             | 81.68±0.26  | 86.41±1.27  | 0.96±0.04   | 0.86±0.10   | 88.44±0.21  | 92.01±0.90 
  platt                | 90.24±0.15  | 92.85±0.10  | 12.41±0.73  | 8.45±0.84   | 95.11±0.10  | 96.44±0.04 
  beta                 | 90.31±0.19  | 92.69±0.17  | 15.80±0.57  | 10.64±0.99  | 95.30±0.10  | 96.42±0.06 
  empirical            | 66.99±0.62  | 68.16±0.38  | 0.00±0.00   | 0.01±0.01   | 73.97±0.70  | 75.28±0.42 
  empirical_cumulative | 89.87±0.15  | 92.61±0.11  | 9.61±0.63   | 5.29±0.51   | 94.76±0.10  | 96.20±0.07 
  val_cal              | 90.32±0.20  | 92.88±0.05  | 14.97±0.70  | 7.53±0.76   | 95.26±0.12  | 96.42±0.03 
### ciciot B15-AC zs  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 120.00±83.67 neurons | 44.80±7.16 bits
  GA Neurons  : 134.20±69.08 neurons | 50.56±3.15 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.99±0.13  | 92.73±0.10  | 16.26±1.82  | 9.41±0.58   | 95.70±0.07  | 96.40±0.04 
  fixed_05             | 80.99±0.67  | 83.38±0.82  | 0.97±0.07   | 0.86±0.12   | 87.87±0.57  | 89.77±0.63 
  platt                | 90.05±0.27  | 92.73±0.11  | 10.87±0.18  | 9.09±0.29   | 94.92±0.16  | 96.39±0.06 
  beta                 | 90.68±0.63  | 92.63±0.09  | 17.00±4.21  | 11.12±0.74  | 95.57±0.16  | 96.40±0.04 
  empirical            | 86.71±9.40  | 79.60±11.98 | 12.61±7.35  | 3.33±4.56   | 91.95±8.25  | 85.47±9.96 
  empirical_cumulative | 90.52±0.41  | 92.62±0.21  | 11.91±0.92  | 7.90±0.33   | 95.24±0.27  | 96.29±0.12 
  val_cal              | 90.99±0.13  | 92.74±0.10  | 16.09±1.87  | 9.00±0.81   | 95.69±0.07  | 96.39±0.03 
best_fpr  (runs: 5/5)
  Grid Search : 190.00±54.77 neurons | 89.60±8.76 bits
  GA Neurons  : 131.80±66.13 neurons | 50.38±3.08 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.06±0.20  | 92.71±0.17  | 11.55±1.55  | 9.65±0.93   | 94.96±0.05  | 96.40±0.06 
  fixed_05             | 80.40±0.72  | 83.16±0.70  | 0.79±0.10   | 0.83±0.10   | 87.36±0.61  | 89.60±0.54 
  platt                | 89.96±0.13  | 92.68±0.24  | 15.19±0.88  | 9.03±0.35   | 95.07±0.12  | 96.36±0.13 
  beta                 | 89.66±0.12  | 92.62±0.13  | 18.24±0.59  | 11.04±0.56  | 95.05±0.09  | 96.40±0.07 
  empirical            | 63.95±3.71  | 83.32±10.85 | 0.01±0.01   | 4.44±4.94   | 70.38±4.40  | 88.81±9.04 
  empirical_cumulative | 89.56±0.40  | 92.62±0.22  | 8.29±0.68   | 7.97±0.94   | 94.51±0.28  | 96.29±0.11 
  val_cal              | 90.09±0.23  | 92.71±0.18  | 12.43±1.14  | 9.21±1.14   | 95.02±0.11  | 96.38±0.06 
best_acc  (runs: 5/5)
  Grid Search : 120.00±83.67 neurons | 44.80±7.16 bits
  GA Neurons  : 133.40±69.77 neurons | 50.19±2.56 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.99±0.13  | 92.72±0.09  | 16.26±1.82  | 9.70±0.31   | 95.70±0.07  | 96.41±0.05 
  fixed_05             | 80.99±0.67  | 83.19±0.65  | 0.97±0.07   | 0.87±0.12   | 87.87±0.57  | 89.63±0.51 
  platt                | 90.05±0.27  | 92.72±0.11  | 10.87±0.18  | 9.21±0.30   | 94.92±0.16  | 96.39±0.06 
  beta                 | 90.68±0.63  | 92.59±0.07  | 17.00±4.21  | 11.32±0.61  | 95.57±0.16  | 96.39±0.03 
  empirical            | 86.71±9.40  | 84.51±11.11 | 12.61±7.35  | 5.25±4.77   | 91.95±8.25  | 89.70±9.10 
  empirical_cumulative | 90.52±0.41  | 92.61±0.19  | 11.91±0.92  | 8.31±0.55   | 95.24±0.27  | 96.30±0.11 
  val_cal              | 90.99±0.13  | 92.73±0.10  | 16.09±1.87  | 9.34±0.54   | 95.69±0.07  | 96.40±0.04 
best_ce  (runs: 5/5)
  Grid Search : 220.00±44.72 neurons | 70.40±8.76 bits
  GA Neurons  : 135.00±67.21 neurons | 50.05±2.39 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.18±0.18  | 92.74±0.14  | 14.33±1.06  | 9.49±0.35   | 95.15±0.14  | 96.41±0.07 
  fixed_05             | 81.59±0.29  | 83.22±0.56  | 0.91±0.08   | 0.86±0.11   | 88.35±0.24  | 89.65±0.44 
  platt                | 90.15±0.13  | 92.74±0.14  | 12.88±1.00  | 9.10±0.28   | 95.07±0.07  | 96.39±0.07 
  beta                 | 90.08±0.26  | 92.62±0.10  | 16.32±0.79  | 11.24±0.66  | 95.19±0.12  | 96.40±0.06 
  empirical            | 65.95±2.20  | 78.65±12.90 | 0.00±0.00   | 3.46±4.75   | 72.75±2.53  | 84.44±11.01
  empirical_cumulative | 89.62±0.22  | 92.68±0.12  | 8.78±1.09   | 8.23±0.66   | 94.57±0.18  | 96.34±0.06 
  val_cal              | 90.17±0.15  | 92.75±0.14  | 14.45±1.44  | 9.23±0.85   | 95.16±0.15  | 96.41±0.05 
best_fitness  (runs: 5/5)
  Grid Search : 120.00±83.67 neurons | 48.00±0.00 bits
  GA Neurons  : 134.20±69.26 neurons | 50.15±2.51 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.99±0.14  | 92.73±0.10  | 15.82±1.84  | 9.71±0.29   | 95.68±0.07  | 96.41±0.05 
  fixed_05             | 81.38±0.12  | 83.23±0.62  | 1.00±0.03   | 0.86±0.12   | 88.20±0.10  | 89.66±0.49 
  platt                | 90.29±0.20  | 92.73±0.11  | 11.04±0.29  | 9.14±0.25   | 95.07±0.13  | 96.39±0.06 
  beta                 | 90.94±0.17  | 92.61±0.07  | 14.82±0.55  | 11.23±0.72  | 95.61±0.09  | 96.40±0.03 
  empirical            | 82.78±11.20 | 79.94±11.72 | 9.48±8.70   | 3.45±4.70   | 88.53±9.77  | 85.82±9.73 
  empirical_cumulative | 90.36±0.51  | 92.62±0.19  | 11.10±1.27  | 8.07±0.33   | 95.11±0.36  | 96.30±0.11 
  val_cal              | 90.98±0.14  | 92.75±0.11  | 15.66±1.84  | 9.38±0.40   | 95.67±0.06  | 96.41±0.05 
### ciciot Wc-CTRL desir  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 180.00±57.01 neurons | 64.00±0.00 bits
  GA Neurons  : 181.00±34.11 neurons | 76.80±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.39±0.16  | 92.77±0.11  | 15.69±0.69  | 7.10±1.15   | 95.34±0.11  | 96.35±0.09 
  fixed_05             | 81.69±0.08  | 85.46±0.90  | 0.98±0.03   | 0.89±0.12   | 88.45±0.07  | 91.34±0.66 
  platt                | 90.21±0.07  | 92.65±0.19  | 11.98±0.12  | 9.65±0.78   | 95.07±0.04  | 96.36±0.09 
  beta                 | 90.38±0.17  | 92.39±0.27  | 15.59±0.32  | 12.02±0.84  | 95.33±0.08  | 96.30±0.13 
  empirical            | 68.51±2.17  | 67.96±1.75  | 0.00±0.00   | 0.01±0.00   | 75.64±2.37  | 75.04±1.92 
  empirical_cumulative | 89.80±0.16  | 92.54±0.15  | 9.46±0.65   | 5.55±0.67   | 94.71±0.12  | 96.17±0.10 
  val_cal              | 90.39±0.14  | 92.79±0.12  | 15.21±0.68  | 7.64±0.94   | 95.32±0.10  | 96.38±0.08 
best_fpr  (runs: 5/5)
  Grid Search : 200.00±50.00 neurons | 97.60±2.19 bits
  GA Neurons  : 180.20±34.52 neurons | 76.80±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 89.98±0.13  | 92.76±0.12  | 10.24±0.46  | 7.19±1.23   | 94.85±0.07  | 96.35±0.10 
  fixed_05             | 79.58±0.21  | 85.47±0.91  | 0.74±0.04   | 0.88±0.12   | 86.66±0.18  | 91.34±0.67 
  platt                | 89.81±0.12  | 92.65±0.17  | 16.46±0.09  | 9.64±0.75   | 95.04±0.07  | 96.36±0.07 
  beta                 | 89.46±0.10  | 92.39±0.28  | 19.14±0.10  | 12.02±0.84  | 94.98±0.06  | 96.30±0.13 
  empirical            | 62.07±3.10  | 68.08±1.75  | 0.01±0.01   | 0.01±0.00   | 68.16±3.72  | 75.17±1.93 
  empirical_cumulative | 89.71±0.20  | 92.56±0.14  | 8.87±0.46   | 5.61±0.59   | 94.63±0.13  | 96.18±0.09 
  val_cal              | 90.07±0.12  | 92.78±0.09  | 11.68±0.53  | 7.51±0.79   | 94.97±0.07  | 96.37±0.06 
best_acc  (runs: 5/5)
  Grid Search : 190.00±41.83 neurons | 64.00±0.00 bits
  GA Neurons  : 181.00±34.11 neurons | 76.80±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.37±0.15  | 92.77±0.11  | 15.77±0.82  | 7.10±1.15   | 95.33±0.10  | 96.35±0.09 
  fixed_05             | 81.72±0.06  | 85.46±0.90  | 0.99±0.04   | 0.89±0.12   | 88.47±0.05  | 91.34±0.66 
  platt                | 90.19±0.08  | 92.65±0.19  | 12.03±0.11  | 9.65±0.78   | 95.06±0.05  | 96.36±0.09 
  beta                 | 90.35±0.16  | 92.39±0.27  | 15.65±0.32  | 12.02±0.84  | 95.32±0.08  | 96.30±0.13 
  empirical            | 68.06±1.42  | 67.96±1.75  | 0.00±0.00   | 0.01±0.00   | 75.16±1.58  | 75.04±1.92 
  empirical_cumulative | 89.76±0.16  | 92.54±0.15  | 9.44±0.62   | 5.55±0.67   | 94.68±0.12  | 96.17±0.10 
  val_cal              | 90.37±0.12  | 92.79±0.12  | 15.30±0.86  | 7.64±0.94   | 95.31±0.09  | 96.38±0.08 
best_ce  (runs: 5/5)
  Grid Search : 180.00±27.39 neurons | 76.80±7.16 bits
  GA Neurons  : 182.00±36.38 neurons | 76.80±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.20±0.18  | 92.77±0.14  | 13.89±0.70  | 7.08±1.13   | 95.15±0.12  | 96.35±0.10 
  fixed_05             | 81.40±0.21  | 85.46±0.93  | 0.89±0.08   | 0.89±0.12   | 88.20±0.18  | 91.34±0.68 
  platt                | 90.15±0.12  | 92.67±0.17  | 13.52±0.82  | 9.64±0.74   | 95.10±0.06  | 96.37±0.08 
  beta                 | 90.05±0.25  | 92.38±0.28  | 16.65±0.71  | 12.10±0.86  | 95.19±0.11  | 96.30±0.13 
  empirical            | 66.56±1.23  | 67.83±1.84  | 0.01±0.00   | 0.01±0.00   | 73.47±1.40  | 74.90±2.04 
  empirical_cumulative | 89.76±0.23  | 92.54±0.14  | 8.98±0.66   | 5.58±0.55   | 94.66±0.17  | 96.17±0.09 
  val_cal              | 90.19±0.20  | 92.81±0.13  | 14.46±0.91  | 7.72±0.82   | 95.17±0.12  | 96.39±0.08 
best_fitness  (runs: 5/5)
  Grid Search : 190.00±41.83 neurons | 76.80±7.16 bits
  GA Neurons  : 182.00±36.38 neurons | 76.80±7.16 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.19±0.20  | 92.77±0.14  | 13.48±0.90  | 7.08±1.13   | 95.12±0.15  | 96.35±0.10 
  fixed_05             | 81.40±0.21  | 85.46±0.93  | 0.88±0.08   | 0.89±0.12   | 88.20±0.18  | 91.34±0.68 
  platt                | 90.15±0.14  | 92.67±0.17  | 13.55±0.85  | 9.64±0.74   | 95.10±0.07  | 96.37±0.08 
  beta                 | 90.04±0.27  | 92.38±0.28  | 16.69±0.75  | 12.10±0.86  | 95.18±0.12  | 96.30±0.13 
  empirical            | 66.02±2.07  | 67.83±1.84  | 0.01±0.00   | 0.01±0.00   | 72.84±2.40  | 74.90±2.04 
  empirical_cumulative | 89.75±0.25  | 92.54±0.14  | 8.89±0.78   | 5.58±0.55   | 94.65±0.19  | 96.17±0.09 
  val_cal              | 90.19±0.20  | 92.81±0.13  | 14.10±0.85  | 7.72±0.82   | 95.15±0.14  | 96.39±0.08 
### ciciot B15-AC desir  (runs: 5/5)
best_f1  (runs: 5/5)
  Grid Search : 190.00±65.19 neurons | 64.00±0.00 bits
  GA Neurons  : 215.80±29.35 neurons | 80.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.30±0.17  | 93.03±0.16  | 15.51±0.92  | 7.39±0.46   | 95.28±0.13  | 96.51±0.07 
  fixed_05             | 81.69±0.11  | 85.90±0.74  | 1.00±0.04   | 0.76±0.08   | 88.45±0.09  | 91.65±0.54 
  platt                | 90.15±0.08  | 92.95±0.11  | 12.04±0.13  | 8.59±0.23   | 95.03±0.04  | 96.50±0.06 
  beta                 | 90.28±0.17  | 92.68±0.12  | 15.70±0.17  | 10.85±0.31  | 95.27±0.09  | 96.42±0.06 
  empirical            | 68.31±2.46  | 65.76±0.78  | 0.00±0.00   | 0.00±0.00   | 75.41±2.70  | 72.57±0.90 
  empirical_cumulative | 89.72±0.14  | 92.62±0.26  | 9.55±0.49   | 4.38±0.39   | 94.67±0.11  | 96.18±0.16 
  val_cal              | 90.29±0.16  | 93.00±0.11  | 14.69±1.18  | 6.92±0.74   | 95.23±0.14  | 96.47±0.06 
best_fpr  (runs: 5/5)
  Grid Search : 170.00±67.08 neurons | 97.60±2.19 bits
  GA Neurons  : 218.20±29.51 neurons | 80.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.05±0.10  | 93.03±0.16  | 10.49±0.38  | 7.36±0.49   | 94.91±0.04  | 96.50±0.07 
  fixed_05             | 79.59±0.26  | 85.87±0.77  | 0.72±0.04   | 0.76±0.09   | 86.67±0.23  | 91.63±0.56 
  platt                | 89.92±0.12  | 92.95±0.12  | 16.31±0.31  | 8.57±0.17   | 95.10±0.07  | 96.50±0.06 
  beta                 | 89.58±0.13  | 92.68±0.12  | 18.97±0.39  | 10.85±0.30  | 95.04±0.06  | 96.42±0.06 
  empirical            | 72.87±15.83 | 65.68±0.85  | 4.53±6.19   | 0.00±0.00   | 78.39±15.23 | 72.47±0.98 
  empirical_cumulative | 89.61±0.17  | 92.63±0.25  | 8.39±0.25   | 4.40±0.43   | 94.54±0.11  | 96.18±0.15 
  val_cal              | 90.13±0.18  | 93.03±0.16  | 12.64±1.34  | 7.00±0.38   | 95.05±0.05  | 96.49±0.09 
best_acc  (runs: 5/5)
  Grid Search : 190.00±65.19 neurons | 64.00±0.00 bits
  GA Neurons  : 215.80±29.35 neurons | 80.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.30±0.17  | 93.03±0.16  | 15.51±0.92  | 7.39±0.46   | 95.28±0.13  | 96.51±0.07 
  fixed_05             | 81.69±0.11  | 85.90±0.74  | 1.00±0.04   | 0.76±0.08   | 88.45±0.09  | 91.65±0.54 
  platt                | 90.15±0.08  | 92.95±0.11  | 12.04±0.13  | 8.59±0.23   | 95.03±0.04  | 96.50±0.06 
  beta                 | 90.28±0.17  | 92.68±0.12  | 15.70±0.17  | 10.85±0.31  | 95.27±0.09  | 96.42±0.06 
  empirical            | 68.31±2.46  | 65.76±0.78  | 0.00±0.00   | 0.00±0.00   | 75.41±2.70  | 72.57±0.90 
  empirical_cumulative | 89.72±0.14  | 92.62±0.26  | 9.55±0.49   | 4.38±0.39   | 94.67±0.11  | 96.18±0.16 
  val_cal              | 90.29±0.16  | 93.00±0.11  | 14.69±1.18  | 6.92±0.74   | 95.23±0.14  | 96.47±0.06 
best_ce  (runs: 5/5)
  Grid Search : 220.00±44.72 neurons | 76.80±7.16 bits
  GA Neurons  : 219.40±29.36 neurons | 80.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.15±0.15  | 93.02±0.15  | 13.78±0.51  | 7.61±0.49   | 95.11±0.08  | 96.51±0.07 
  fixed_05             | 81.38±0.26  | 85.85±0.76  | 0.90±0.07   | 0.76±0.09   | 88.18±0.22  | 91.62±0.55 
  platt                | 90.12±0.12  | 92.94±0.11  | 13.54±0.86  | 8.62±0.23   | 95.08±0.07  | 96.49±0.06 
  beta                 | 89.99±0.18  | 92.69±0.11  | 16.78±0.68  | 10.86±0.33  | 95.16±0.07  | 96.42±0.06 
  empirical            | 64.94±2.07  | 66.08±0.90  | 0.00±0.00   | 0.01±0.01   | 71.59±2.41  | 72.93±1.04 
  empirical_cumulative | 89.63±0.20  | 92.62±0.27  | 8.67±0.74   | 4.38±0.36   | 94.57±0.15  | 96.18±0.16 
  val_cal              | 90.13±0.15  | 93.03±0.13  | 14.48±0.82  | 7.02±0.45   | 95.13±0.07  | 96.49±0.07 
best_fitness  (runs: 5/5)
  Grid Search : 210.00±65.19 neurons | 80.00±0.00 bits
  GA Neurons  : 217.00±29.23 neurons | 80.00±0.00 bits
  mode                 | F1 Grid     | F1 GA       | FPR Grid    | FPR GA      | Acc Grid    | Acc GA     
  ---------------------+-------------+-------------+-------------+-------------+-------------+------------
  train_cal            | 90.13±0.15  | 93.03±0.16  | 12.99±0.33  | 7.37±0.40   | 95.07±0.10  | 96.50±0.07 
  fixed_05             | 81.29±0.13  | 85.88±0.75  | 0.85±0.04   | 0.76±0.08   | 88.11±0.11  | 91.64±0.54 
  platt                | 90.12±0.15  | 92.95±0.12  | 13.81±0.10  | 8.56±0.23   | 95.10±0.08  | 96.50±0.07 
  beta                 | 89.93±0.22  | 92.69±0.12  | 16.89±0.18  | 10.81±0.32  | 95.14±0.12  | 96.42±0.06 
  empirical            | 64.91±3.21  | 65.92±1.03  | 0.01±0.01   | 0.00±0.00   | 71.52±3.64  | 72.74±1.18 
  empirical_cumulative | 89.52±0.09  | 92.62±0.24  | 7.86±0.26   | 4.34±0.38   | 94.46±0.06  | 96.17±0.15 
  val_cal              | 90.11±0.15  | 93.04±0.14  | 13.30±1.05  | 7.04±0.41   | 95.07±0.13  | 96.50±0.08 
```

</details>

Reproduce: read-only script (session scratchpad `agg.py`, NOT committed — task was read-only/no code, subcommands diff/hdr/primary/arms/seeds/b05vb15/pareto/rule7), same SQL pattern as scripts/ids2/load.py restricted to `IDSAGG%`.

### IDS-20 design check (06/10/2026, experiment-design agent) — interaction, mechanism (stopping-'defect' note superseded by Luiz 06/10)

All POST-HOC except where marked. Same metric (held-out TEST val_cal, GA best_f1), IDSAGG- prefix only, n=5 paired seeds.

**1. CIC-IoT arm x aggregation interaction** (per seed: (B15-AC desir−zs) − (Wc-CTRL desir−zs)):
```
metric | r20414 | r20415 | r20416 | r20417 | r20418 | mean   | SD    | CI95             | t     | p raw  | signs
-------+--------+--------+--------+--------+--------+--------+-------+------------------+-------+--------+-------
FPR pp | -1.600 | -2.060 | -0.950 | -3.775 | -1.740 | -2.025 | 1.058 | [-3.339, -0.711] | -4.28 | 0.0129 | +0/-5
F1  pp | +0.710 | +0.241 | +0.266 | +0.533 | +0.231 | +0.396 | 0.215 | [+0.128, +0.664] | +4.11 | 0.0147 | +5/-0
```
Per-arm paired desir−zs, post-hoc family = 2 datasets x 2 arms x {F1,FPR} = 8 tests, Holm over the 8
(exact sign test at n=5 floors at p=0.0625 two-sided, so 5/5 alone never reaches 0.05):
```
dataset | arm     | metric | mean   | SD    | CI95             | p raw  | Holm(8) | signs
--------+---------+--------+--------+-------+------------------+--------+---------+------
cicids  | Wa-CTRL | F1     | -0.101 | 0.064 | [-0.181, -0.021] | 0.0245 | 0.160   | +0/-5
cicids  | Wa-CTRL | FPR    | +0.101 | 0.080 | [+0.002, +0.200] | 0.0472 | 0.236   | +5/-0
cicids  | CE20    | F1     | -0.063 | 0.100 | [-0.187, +0.060] | 0.2271 | 0.681   | +1/-4
cicids  | CE20    | FPR    | -0.003 | 0.073 | [-0.093, +0.087] | 0.9331 | 1.000   | +3/-2
ciciot  | Wc-CTRL | F1     | -0.136 | 0.126 | [-0.293, +0.020] | 0.0728 | 0.291   | +1/-4
ciciot  | Wc-CTRL | FPR    | -0.049 | 0.869 | [-1.128, +1.030] | 0.9058 | 1.000   | +3/-2
ciciot  | B15-AC  | F1     | +0.260 | 0.161 | [+0.059, +0.460] | 0.0228 | 0.160   | +5/-0
ciciot  | B15-AC  | FPR    | -2.074 | 0.573 | [-2.785, -1.363] | 0.0013 | 0.010   | +0/-5
```
Verdict: the CIC-IoT FPR firing is B15-AC-specific. Only B15-AC FPR survives Holm(8). On the production control
(Wc-CTRL) the effect is −0.05 pp with CI ±1.1 pp: we cannot detect an effect there, and that is not evidence that
none exists. Selection asymmetry: B15-AC was picked in the DESIRABILITY
sweep (IDSXD), partly for its low FPR, so its weights were tuned in desirability's fitness geometry. Fresh seeds
remove seed-level winner's curse but not that. The pooled "mean over arms" secondary hid an interaction this size.

**2. Mechanism (descriptive).** Matched-generation check uses the during-search pool[0] FPR at GA generation 60
(`iterations.best_fpr`, k-fold on train — mechanism evidence only, NOT a result):
```
arm     | gen-60 FPR desir (5 seeds)      | gen-60 FPR zs (5 seeds)         | gap desir-zs (per seed)              | desir gen60->last
--------+---------------------------------+---------------------------------+--------------------------------------+------------------
B15-AC  | 4.12 4.26 4.02 4.21 3.62 (4.05) | 6.31 6.09 6.76 7.10 5.37 (6.33) | -2.19 -1.83 -2.74 -2.89 -1.75 (-2.28) | -0.26 mean
Wc-CTRL | 4.77 5.78 4.96 4.99 4.94 (5.09) | 5.10 5.76 4.36 4.63 4.98 (4.97) | -0.33 +0.02 +0.60 +0.36 -0.04 (+0.12) | -0.23 mean
```
The B15-AC FPR gap is already there at EQUAL generations. The extra 70-100 desirability generations add only
~0.26 pp more, so the extra GENERATIONS do not explain the gap. Across the 10 ciciot pairs, held-out ΔFPR vs Δgens
has Spearman −0.95 (−0.90 in Wc, −0.87 in B15). That is a post-treatment correlation (runs that keep improving keep
running), not a mechanism. BITS: on B15-AC every desirability run lands at 80b and no zscore run does
(48-55b), so within that arm bits and aggregation are perfectly collinear and cannot be separated. The one partial
hint against bits-as-cause is Wc-CTRL: desirability moved it to 80b on 4/5 seeds with no FPR gain, and the single
zscore run at 80b (r20416, FPR 8.03) is not low. That is n=1, so it is not evidence. Within-arm ρ(ΔFPR, Δbits) is
+0.58 (Wc) and −0.15 (B15). The pooled −0.66 is driven by the arm difference.

**zscore stopping — NOT a defect (Luiz ruling 06/10/2026; supersedes the "stopping defect" text first written here).**
Observed: under zscore the IDS magnitude-patience watches the incumbent `best`, which zscore rarely replaces (18/20
zscore GA phases never changed it; 13/20 stopped at the 60-gen floor vs 1/20 desirability). Patience is BY DESIGN:
it stops a search that is not improving its incumbent. The root cause is zscore's scoring, which does not separate
the population enough to produce improvements, NOT the stopper. So the early stop is part of the zscore treatment,
exactly as the 24/08 ruling says (generations is a MEDIATOR of the aggregation, not a confounder). Patience-off /
budget-matched re-flies are NOT warranted: patience-off was tested when patience was built (~2025-12/2026-02), and
the cost/benefit does not justify it. No cohort audit (IDSZ/IDSX/MCS) is warranted on this ground.
- IDS-20 stands as the pre-registered rule read it: desirability stays on both datasets.
- The CIC-IoT FPR secondary is real AND arm-specific (interaction above, post-hoc): report it with that caveat.

**3. B05-AC extension criterion** pre-registered in `experiments/ids_aggregation_b05ac_ext_criterion.json`
(commit 4d5f8105, BEFORE flows 6425-6429 were read). Its 06/10 amendment (stopping 'defect') is superseded by Luiz's 06/10 ruling; thresholds unchanged.

**4. Paper-safe wording.**
- CICIDS2017: "Replacing z-score with desirability aggregation did not improve detection on CICIDS2017 (paired
  desirability − z-score F1 −0.08 pp, 95% CI [−0.18, +0.01], FPR +0.05 pp, n = 5 seeds x 2 weight settings;
  F1 is at ceiling, ~99.5%)."
- CIC-IoT-2023 (the arm-dependence is post-hoc; say so): "On CIC-IoT-2023 the aggregation effect
  depended on the weight setting. Desirability lowered held-out FPR by 2.1 pp (5/5 seeds) under the weights
  selected for it, and had no detectable effect under the default weights (−0.05 pp, 95% CI ±1.1 pp). We do not
  claim a general FPR benefit." Never "desirability lowers FPR on CIC-IoT" unqualified.

### B05-AC extension 10/10 (final read 06/10/2026, all 10 IDSAGGX flows 6420-6429 completed; last done 07/10/2026 00:26 UTC)

Criterion applied exactly as pre-registered in `experiments/ids_aggregation_b05ac_ext_criterion.json` (thresholds
unchanged; Luiz's 06/10 ruling supersedes the stopping amendment, so this is read as plain desirability vs zscore).
Metric: held-out TEST val_cal, GA phase, `validation_summaries` final. DESCRIPTIVE; the t/CI are for information only.
Disclosure: 6425 (desir r20416) and 6426 (zs r20417) were read early. Blind at this read: r20417 desir, r20418 zs+desir.

- **S (keep B05-AC + desirability): SUPPORTED.** S1 mean dF1 +0.215 >= -0.10, 1/5 seeds < 0. S2 mean dFPR -2.122 <= +0.50, 0/5 seeds > 0.
- **R (reopen): not fired.** R1: dF1 is positive. R2: dFPR is negative on all 5 seeds.
- **Shares the B15-AC FPR benefit: YES** (mean dFPR -2.12 <= -1.0, 5/5 < 0). B15-AC's IDSAGG value was -2.07, 5/5.
- **Fresh pairs r20416-18:** dF1 +0.220 (+3/-0), dFPR -2.245 (0/3 opposite to the n=5 mean), so the FPR result is NOT carried by the pairs known at registration.
- **Blind seeds:** r20418 (fully blind) dF1 +0.263 / dFPR -1.395. r20417 (desir blind) dF1 +0.382 / dFPR -2.775. Same sign as the n=5 mean.
- **Arm-choice side read (B15-AC desir vs B05-AC desir): flag NOT fired.** B15 does not beat B05 on mean F1 (B05-B15 = +0.041). B15 is ahead on FPR by 0.43 pp, 4/5 seeds, which is below the 0.50 pp bar.

```
B05-AC, desir - zs | seed  | F1 zs / desir / d        | FPR zs / desir / d        | Acc zs / desir / d       | flows     | read
-------------------+-------+--------------------------+---------------------------+--------------------------+-----------+------------------
GA best_f1         | 20414 | 92.581 / 92.995 / +0.414 | 10.005 / 6.470 / -3.535   | 96.338 / 96.456 / +0.118 | 6420,6421 | known at reg
GA best_f1         | 20415 | 93.037 / 93.035 / -0.003 |  8.290 / 7.950 / -0.340   | 96.535 / 96.523 / -0.012 | 6422,6423 | known at reg
GA best_f1         | 20416 | 93.062 / 93.078 / +0.016 |  9.400 / 6.835 / -2.565   | 96.583 / 96.513 / -0.069 | 6424,6425 | fresh (6425 early)
GA best_f1         | 20417 | 92.680 / 93.062 / +0.382 | 10.130 / 7.355 / -2.775   | 96.397 / 96.520 / +0.124 | 6426,6427 | fresh (desir blind)
GA best_f1         | 20418 | 92.785 / 93.048 / +0.263 |  9.535 / 8.140 / -1.395   | 96.435 / 96.536 / +0.101 | 6428,6429 | fresh, blind
n=5   d mean±SD    |       | +0.215±0.198 CI[-0.031,+0.460] +4/-1 | -2.122±1.257 CI[-3.683,-0.561] +0/-5 | +0.052±0.088 +3/-2
fresh d mean±SD    |       | +0.220±0.187 +3/-0       | -2.245±0.744 +0/-3        | +0.052±0.106 +2/-1
GA best_fitness n=5|       | +0.227±0.156 CI[+0.034,+0.420] +5/-0 | -2.485±0.937 CI[-3.648,-1.322] +0/-5 | +0.048±0.072 +4/-1

arm-choice side read (desir) | F1 %         | FPR %       | Acc %        | GA best_f1 genome    | GA gens      | wall h
-----------------------------+--------------+-------------+--------------+----------------------+--------------+-----------
B05-AC (IDSAGGX 6421..6429)  | 93.04±0.03   | 7.35±0.71   | 96.51±0.03   | 149±51 n x 80 b      | 128.0±26.8   | 3.47±1.44
B15-AC (IDSAGG  6387..6419)  | 93.00±0.11   | 6.92±0.74   | 96.47±0.06   | 216±29 n x 80 b      | 146.0±11.4   | 5.35±0.55
d B05-B15 (paired, n=5)      | +0.041±0.117 +3/-2 | +0.425±1.082 +4/-1 | +0.036±0.091 +4/-1 |               |              |
```
Best-of-best (B05-AC desir vs B15-AC desir, 700 single-genome cells = 5 gt x 7 modes x GS/GA x 10 flows; this is SELECTION, not a claim):
best F1 anywhere: B15 6403 r20416 GA best_fpr val_cal 93.225/6.710/96.592 (183n x 80b) vs B05 6425 r20416 GA best_fpr val_cal
93.105/7.210/96.540 (217n x 80b); top-10 F1 is 10/10 B15. Lowest FPR at F1>=92.5: B15 6411 r20417 GA best_fitness empirical_cumulative
92.527/4.090 (228n x 80b) vs B05 6427 r20417 GA best_acc empirical_cumulative 92.970/5.725 (162n x 80b). Per-run lowest FPR@92.5: B05
6.08±0.28 (0/5 sub-5%) vs B15 4.92±1.08 (4/5 sub-5%). Seed-mean GA best_f1 empirical_cumulative: B05 93.01±0.04/6.39±0.38 vs B15
92.62±0.26/4.38±0.39. That is a trade along the curve, not dominance. The Pareto front over 700 cells holds 32 B15 and 14 B05 points.

Note for experiment-design (not ruled here): B05-AC was also selected in the desirability sweep (IDSXD/IDSXD2, IDS-2). So the selection-asymmetry
caveat attached to B15-AC applies to B05-AC too. The "no detectable effect under default weights (Wc-CTRL)" wording stays as written.

## IDS-2 FINAL (26/09/2026) — IDSXD / IDSXD2 weight-sweep winners (cicids, ciciot; unswr quad PROVISIONAL)

Supersedes the 13/09 interim bullet in §0A for cicids and ciciot. Read-only DB readout; every
number is HELD-OUT TEST (`validation_summaries.threshold_metadata`, `validation_point='final'`,
random_3way, Protocol v2: thresholds fit on the 10% VAL, reported on the 10% TEST). No
`iterations.best_f1` anywhere. Statistical support (is the lead real) is the experiment-design
agent's call, made separately from this readout.

```
Header (26/09/2026 17:30 UTC)
  IDSXD  cicids  quad-96b  8 arms x {20403,20404,20405}          24/24 completed | 37.7 h total | 1.57 h/run | last done 29/08/2026 09:22 UTC
  IDSXD  ciciot  quad-96b  8 arms x {20403,20404,20405}          24/24 completed | 95.6 h total | 3.98 h/run | last done 03/09/2026 08:03 UTC
         + 2 targeted -w64fix repairs (6050 B05-AC s20405, 6051 B05-CE s20404)  2/2 | 8.2 h | 4.10 h/run | 17/09/2026 03:43 UTC
  IDSXD2 ciciot  quad-96b  6 arms x {20406,20407}                12/12 completed | 45.2 h total | 3.77 h/run | last done 26/09/2026 12:07 UTC
  IDSXD  unswr   quad-64b  9 arms (B34-CTRL n=5)                 29/29 completed | 25.0 h total | 0.86 h/run | 16/09/2026 19:31 UTC
  IDSXD  unswr   qsr-64b   pre-ABI-13 29 completed = VOID (IDS-16); 29 -abi13 reruns (6264-6356) 0/29, QUEUED
  ETA cicids/ciciot: complete. ETA unswr qsr: not computable (0 of 94 ABI-13 reruns done, no measured duration).
Constant params (all 62 cicids/ciciot flows, verified from flows.config_json): fitness_aggregation=desirability,
  fitness_ce_anchor_normalized=0.2128, fitness_calculator=harmonic_rank, memory mode QUAD_WEIGHTED (default; no
  memory_mode key), ids_n_bits=96 (thermometer), top20, binary, random_3way, ids_k_folds=5 / kfold_per_gen=5,
  ga_generations=250, patience=5, population 50, min_bits=4, min_neurons=5, neuron_sample_rate=0.25, OI training.
  Varying: weights, dataset, max_bits (cicids 34 / ciciot 100), max_neurons (cicids 500 / ciciot 250), seed.
  IDSXD2 flows differ from their IDSXD arm ONLY in `seed` (diffed key-by-key).
Weights (f1/fpr/ce/acc): B05-AC .05/.05/.225/.675  B05-CE .05/.05/.675/.225  B10-AC .1/.1/.2/.6  B10-CE .1/.1/.6/.2
  B15-AC .15/.15/.175/.525  B15-CE .15/.15/.525/.175  CE20 .3/.4/.2/.1  Wa-CTRL .3/.05/.35/.3  Wc-CTRL .15/.05/.7/.1
```

### DECISION (Luiz, 02/10/2026) — CICIDS2017 = **CE20**, CIC-IoT-2023 subsample = **B05-AC**

> **IDS-20 resolved 06/10/2026** (§ IDS-20 FINAL above): the aggregation rule did NOT fire on F1 on either dataset;
> the CIC-IoT FPR secondary fired FOR desirability (−1.06 pp, 5/5, Holm 0.0195). Desirability stands on both → CE20 and
> B05-AC stand as chosen; no zscore re-check required by the rule.

Config decisions, NOT statistically supported wins (n=3/5 sweep; IDS-18 confirmation Holm p=0.109). Both arms are
desirability-only, so re-read if IDS-20 (IDSAGG, zscore vs desirability, ETA ~06/10) favours zscore. These feed
IDS-4 (CICIDS n=100) and IDS-5 (CIC-IoT n=100). Means are GA val_cal held-out TEST; best-found is the best SINGLE
genome anywhere (5 genome types x 7 modes x GS/GA; `python3 scripts/ids2/best_anywhere.py <ds>`), always quoted
beside the mean.

```
CICIDS2017 -> CE20   weights f1/fpr/ce/acc .30/.40/.20/.10, desirability, QUAD, 96b, bits 4-34, cap 500
  mean (n=3)  best_f1      99.52±0.03 / 0.12±0.01 / 99.70±0.02   best on F1, FPR and Acc; tightest SD
              best_fitness 99.51±0.03 / 0.12±0.00 / 99.69±0.02
  control     Wa-CTRL      99.48±0.09 / 0.18±0.07 / 99.67±0.06   (gap +0.04 F1, -0.06 FPR: stable, not large)
  best found  99.56 / 0.11 / 99.72  GA best_f1 train_cal r20403 flow 5880 (top of the sweep)

CIC-IoT-2023 -> B05-AC   weights .05/.05/.225/.675, desirability, QUAD, 96b, bits 4-100, cap 250
  mean (n=5)  best_f1      93.09±0.21 / 7.52±0.64 / 96.54±0.10   best mean F1 and Acc
  runner-up   B15-AC       93.04±0.11 / 6.53±0.66 / 96.48±0.07   ~1 pp lower sweep FPR for -0.05 F1
  control     Wc-CTRL      92.97±0.18 / 7.63±0.46 / 96.48±0.09
  IDS-18 (6 fresh seeds)   B05-AC vs Wc-CTRL FPR -1.51 pp, F1 +0.26 | B15-AC -1.39 pp, +0.21 (Holm .109, NOT confirmed)
  best found  93.30 / 6.92 / 96.64  GA best_f1 val_cal IDSXD2 r20406 flow 6117 (top F1)
              93.27 / 8.12 / 96.66  GA best_f1 train_cal IDSXD3 r20408 flow 6359 (top Acc)
              93.00 / 5.36 / 96.43  GA best_ce emp_cumulative r20405 flow 6050 (lowest FPR at F1 >= 93)
```
Why B05-AC over B15-AC: best mean F1/Acc and both top single genomes; B15-AC's sweep FPR edge did not
replicate in IDS-18, where B05-AC had the (slightly) larger FPR reduction. Paper wording: "direction replicated,
not significant" — never "significantly lower FPR". Supersedes the 26/09 provisional ciciot call (B15-AC) below.

### Winner calls (original, 26/09/2026)

```
dataset | call                     | weights f1/fpr/ce/acc | aggr.        | per-neuron bits (band) | neurons (cap) | mem  | genome / thr        | n | held-out F1 / FPR / Acc (mean±SD)
--------+--------------------------+-----------------------+--------------+------------------------+---------------+------+---------------------+---+-----------------------------------
cicids  | CE20                     | .30/.40/.20/.10       | desirability | 34 (4-34), 96b thermo  | 127±54 (500)  | QUAD | GA best_f1  val_cal | 3 | 99.52±0.03 / 0.12±0.01 / 99.70±0.02
        |   pre-declared column    |                       |              |                        |               |      | GA best_fitness val | 3 | 99.51±0.03 / 0.12±0.00 / 99.69±0.02
ciciot  | B15-AC (era-clean)       | .15/.15/.175/.525     | desirability | 80 (4-100), 96b thermo | 177±54 (250)  | QUAD | GA best_f1  val_cal | 5 | 93.04±0.11 / 6.53±0.66 / 96.48±0.07
        |   pre-declared column    |                       |              |                        |               |      | GA best_fitness val | 5 | 93.06±0.12 / 7.13±0.83 / 96.51±0.07
        |   alt. operating point   |                       |              |                        |               |      | GA best_fitness emp_cum | 5 | 92.85±0.17 / 4.96±0.43 / 96.32±0.10
unswr   | PROVISIONAL, no winner   | quad half only; every arm on the val_cal FP floor (FPR 1.12); QSR half blocked on IDS-16
```

Runner-ups within one seed-SD of the winner (GA val_cal, same genome type):
- **cicids** (CE20 best_f1 99.52±0.03): B15-AC 99.48±0.06 / 0.15±0.05; Wa-CTRL 99.47±0.09 / 0.21±0.08
  (best_fitness 99.48±0.09 / 0.18±0.07); B10-AC 99.47±0.07 / 0.21±0.08. Each is within ITS OWN SD, not CE20's.
  CE20 has the lowest FPR in the sweep's top group AND the smallest SD. Per seed (best_fitness val_cal), CE20 minus Wa-CTRL =
  +0.02 / +0.11 / −0.06 F1, and minus B15-AC best_f1 = +0.13 / −0.04 / +0.02.
- **ciciot, era-clean n=5** (B15-AC best_fitness 93.06±0.12): Wc-CTRL 92.97±0.15 / 7.77±0.34 (per seed
  B15-AC minus Wc = +0.15 / +0.05 / +0.17 / +0.09 / −0.03); B10-CE 92.90±0.08 / 7.18±0.70.
- **ciciot co-leader, NOT era-clean:** B05-AC n=5 best_f1 93.09±0.21 / 7.52±0.64 / 96.54±0.10 (3 pre-fix + 2 post-fix).
  **IDSXD2 changes the interim picture.** On its two post-fix seeds, B05-AC is the TOP arm: best_f1 93.27±0.03 / 6.94±0.03
  vs B15-AC 93.14±0.09 / 6.13±0.24 and Wc-CTRL 93.16±0.09 / 7.18±0.36 on the same seeds (20406/07). So "B05-AC's lead is
  the pre-fix wheel" is NOT supported. B05-AC buys F1 at higher FPR, and B15-AC is the lower-FPR point. They are co-leaders,
  and the pick of B15-AC rests on era-cleanliness plus FPR, not on F1.
  B10-AC (n=3, 2 pre-fix) 92.97±0.05 / 7.14±1.19 stays unrankable.

### Caveats that travel with these calls
1. **n=3 on cicids**, n=5 on the 6 extended ciciot arms, n=3 on ciciot B05-CE/B10-AC. Never rank an n=5 arm against an
   n=3 arm on means alone. On the common seeds 20403-05 the ciciot order is B15-AC 92.99±0.05, B10-AC 92.96, B05-AC 92.94,
   B10-CE 92.92, Wc 92.87, B05-CE 92.80, B15-CE 92.73, CE20 92.70 (best_fitness val_cal).
2. **Seed block = code era for IDSXD2.** Seeds 20406/07 ran 24-26/09: after the 17/09 immigrant fix (129f8533; immigrants
   now take the population's mode bits instead of 8b) and on worker ABI 13 (a QSR-only trainer change; the quad path is
   unchanged by design, not re-verified here). Seeds 20403-05 ran 27/08-03/09 before both changes. The two eras sit on
   different seeds, so they cannot be separated. They ARE balanced across the 6 extended arms: 4 of those arms gained (GA best_f1 val_cal, mean of 20406-07 minus mean of 20403-05)
   +0.14 to +0.31pp F1 on the new block, while B10-CE (−0.05) and CE20 (+0.04) did not.
3. **OR-fold era (ciciot only).** B05-AC s20403-05 and B05-CE s20403-05 are pre-fix, and so are B10-AC s20403/04. Their GA
   winners are stored at 64 b/neuron, while EVERY post-fix ciciot GA winner is at 80 b/neuron. So the fix moved the bits
   the grid selected, which makes the pre-fix seeds a different architecture outcome and not only folded addresses.
   Rows from -w64fix are a TARGETED repair set, picked by winner width. They are not seeds, are not pooled, and are
   listed for reference only: 6050 B05-AC s20405 best_f1 93.09/6.85/96.52, 6051 B05-CE s20404 92.89/6.05/96.38
   (both 80b). cicids is immune (max_bits 34).
4. **Bits/neuron source:** `genomes.tiers_json` for the validated hash. Before 17/09, tracked rows stored a PER-CLUSTER
   ROUNDED MEAN (IDS-17 census caveat), and IDS runs are single-cluster, so pre-17/09 widths are cluster means, not
   guaranteed exact per-neuron widths. The GA phase does not optimize bits, so the GA width is the grid winner's width.
5. **UNPAIRED with IDSX** (zscore): zero shared seeds. Never pool IDSX with IDSXD. This readout says nothing about
   desirability vs zscore.
6. **Aggregation is not an arm here.** Every flow is desirability with the same normalized anchor 0.2128. "Winner" means
   winning weights under desirability, not "desirability wins".
7. **cicids is a ceiling dataset.** The whole top group spans 99.42-99.52 F1 (per-arm GA best_f1 val_cal). Measured RF raw on the same TEST split
   (§0A 13/09) is 99.85 / 0.10 val_cal. This is efficiency parity at best, never superiority.
8b. **ciciot MLP comparator (measured 07/10/2026, `verify_ids_baselines.py --models mlp`, scaler + 128-64 ReLU, early
   stopping; same random_3way TEST split, top-20; logs/baselines_ciciot/mlp_random_3way_{raw,8b}.log):** raw val_cal
   90.91 / 13.41 / 95.53 (0.14 MB); 8b thermo val_cal 89.57 / 14.63 / 94.82 (0.71 MB). The WNN (B15-AC desir 93.00 / 6.92 /
   96.47) beats the MLP on F1, FPR and Acc on both inputs.
8. **ciciot comparators (measured, §0A 13/09):** RF raw 95.53 / 7.38 / 97.84 (the WNN trails by 2.5pp F1 at equal FPR),
   thermo XGB8b 88.85 / 15.10 (the WNN leads on the same input). Pair modes: val_cal against val_cal.
9. **empirical / empirical_cumulative** are VAL-fit (Protocol v2), so they are legitimate held-out operating points.
   Plain `empirical` collapses on ciciot, though (F1 67, FPR 0.01: a degenerate high threshold), so use empirical_cumulative only.
10. **unswr quad (PROVISIONAL):** all 9 arms sit at GA val_cal 93.41-93.49 F1 / FPR 1.12 (the known UNSW-random val_cal
    FP floor). val_cal cannot discriminate them. The best Pareto point off val_cal is Wb-CTRL GA best_ce val_cal 93.42±0.10 / 0.98±0.24,
    and empirical_cumulative gives CE20 93.06±0.24 / 0.63±0.08 (n=3). B34-CTRL quad r20401-04 GA rows are 34/8-bit
    immigrant HYBRIDS (IDS-17), invalid as a fixed-34b control. The QSR half (29 runs) is VOID pending the ABI-13 reruns
    (IDS-16), so there is no unswr winner and nothing to hand to an n=100 cohort yet.

### Pareto scan (all arms x GS/GA x 5 genome types x 7 modes; mean over seeds; non-dominated on mean F1 vs mean FPR)
```
cicids (560 columns, n=3)                        F1           FPR          Acc
  CE20 GA best_f1/best_acc   train_cal = val_cal  99.52±0.03   0.12±0.01   99.70±0.02
  CE20 GA best_fpr           val_cal/emp_cum      99.48±0.02   0.11±0.01   99.67±0.01
  CE20 GA best_f1            empirical            99.28±0.07   0.10±0.01   99.55±0.05
  B10-CE GA best_fitness     empirical            99.14±0.14   0.09±0.03   99.46±0.09
  B15-AC GS best_fpr         empirical            98.36±0.09   0.05±0.01   98.98±0.06
  (CE20 owns the whole F1 end of the front; every point below 99.3 F1 is an `empirical` threshold point)
ciciot, n=5 columns only (6 arms; B05-AC mixed-era)
  B05-AC GA best_f1          val_cal              93.09±0.21   7.52±0.64   96.54±0.10   (mixed era)
  B05-AC GA best_f1          emp_cum              93.07±0.16   6.30±0.63   96.49±0.08   (mixed era)
  B15-AC GA best_f1/best_acc emp_cum              92.85±0.17   4.99±0.49   96.33±0.11   era-clean
  B10-CE GA best_f1          emp_cum              92.40±0.14   4.43±0.51   96.05±0.09   era-clean
  B15-CE GA best_f1          emp_cum              91.95±0.40   3.78±0.22   95.77±0.24   era-clean
  CE20   GA best_f1          emp_cum              91.17±0.33   2.90±0.18   95.27±0.20   era-clean
  (the ciciot front is an F1-for-FPR trade driven by the ACC weight: AC arms hold F1, CE20/B15-CE buy FPR ~3-4%)
```

### Rule-7 5-tables for the called configs (GA vs Grid, all 7 modes, mean±SD %, held-out TEST)
```

##### RULE-7 5-tables: IDSXD(+2)-cicids-quad-CE20 (Grid vs GA, held-out TEST, mean±SD %) #####

best_f1  (runs: GS 3 | GA 3)
Grid Search : 167±115 neurons | 33.3±1.2 bits (mean per-neuron)
GA Neurons  : 127±54 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  99.29±0.02 |  99.52±0.03 |   0.26±0.03 |   0.12±0.01 |  99.55±0.01 |  99.70±0.02
fixed_05             |  99.03±0.05 |  99.11±0.09 |   0.59±0.06 |   0.55±0.08 |  99.38±0.03 |  99.43±0.06
platt                |  99.24±0.07 |  99.28±0.08 |   0.34±0.04 |   0.38±0.06 |  99.52±0.04 |  99.54±0.05
beta                 |  99.25±0.04 |  99.35±0.01 |   0.29±0.02 |   0.29±0.00 |  99.53±0.03 |  99.59±0.00
empirical            |  98.75±0.41 |  99.28±0.07 |   0.12±0.08 |   0.10±0.01 |  99.22±0.25 |  99.55±0.05
empirical_cumulative |  99.29±0.02 |  99.52±0.03 |   0.26±0.03 |   0.12±0.01 |  99.55±0.01 |  99.70±0.02
val_cal              |  99.29±0.02 |  99.52±0.03 |   0.28±0.06 |   0.12±0.01 |  99.55±0.01 |  99.70±0.02

best_fpr  (runs: GS 3 | GA 3)
Grid Search : 400±173 neurons | 29.3±2.3 bits (mean per-neuron)
GA Neurons  : 127±54 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  99.13±0.03 |  99.48±0.02 |   0.32±0.03 |   0.11±0.01 |  99.45±0.02 |  99.67±0.01
fixed_05             |  98.86±0.05 |  99.14±0.05 |   0.70±0.04 |   0.52±0.05 |  99.27±0.03 |  99.45±0.03
platt                |  99.11±0.03 |  99.27±0.08 |   0.36±0.01 |   0.39±0.06 |  99.44±0.02 |  99.54±0.05
beta                 |  99.09±0.04 |  99.33±0.03 |   0.29±0.02 |   0.31±0.03 |  99.43±0.03 |  99.58±0.02
empirical            |  98.44±0.28 |  99.29±0.18 |   0.07±0.04 |   0.10±0.02 |  99.03±0.17 |  99.55±0.11
empirical_cumulative |  99.05±0.03 |  99.48±0.02 |   0.18±0.03 |   0.11±0.01 |  99.40±0.02 |  99.67±0.01
val_cal              |  99.13±0.03 |  99.48±0.02 |   0.32±0.03 |   0.11±0.01 |  99.45±0.02 |  99.67±0.01

best_acc  (runs: GS 3 | GA 3)
Grid Search : 167±115 neurons | 33.3±1.2 bits (mean per-neuron)
GA Neurons  : 127±54 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  99.29±0.02 |  99.52±0.03 |   0.26±0.03 |   0.12±0.01 |  99.55±0.01 |  99.70±0.02
fixed_05             |  99.03±0.05 |  99.11±0.09 |   0.59±0.06 |   0.55±0.08 |  99.38±0.03 |  99.43±0.06
platt                |  99.24±0.07 |  99.28±0.08 |   0.34±0.04 |   0.38±0.06 |  99.52±0.04 |  99.54±0.05
beta                 |  99.25±0.04 |  99.35±0.01 |   0.29±0.02 |   0.29±0.00 |  99.53±0.03 |  99.59±0.00
empirical            |  98.75±0.41 |  99.28±0.07 |   0.12±0.08 |   0.10±0.01 |  99.22±0.25 |  99.55±0.05
empirical_cumulative |  99.29±0.02 |  99.52±0.03 |   0.26±0.03 |   0.12±0.01 |  99.55±0.01 |  99.70±0.02
val_cal              |  99.29±0.02 |  99.52±0.03 |   0.28±0.06 |   0.12±0.01 |  99.55±0.01 |  99.70±0.02

best_ce  (runs: GS 3 | GA 3)
Grid Search : 233±231 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 126±55 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  99.27±0.03 |  99.48±0.04 |   0.34±0.03 |   0.14±0.03 |  99.54±0.02 |  99.67±0.02
fixed_05             |  99.05±0.07 |  99.12±0.06 |   0.57±0.05 |   0.54±0.05 |  99.40±0.04 |  99.44±0.04
platt                |  99.23±0.02 |  99.26±0.07 |   0.32±0.01 |   0.39±0.06 |  99.51±0.01 |  99.53±0.05
beta                 |  99.25±0.03 |  99.41±0.09 |   0.28±0.02 |   0.24±0.10 |  99.52±0.02 |  99.62±0.06
empirical            |  98.69±0.32 |  99.21±0.19 |   0.11±0.06 |   0.10±0.04 |  99.18±0.19 |  99.51±0.11
empirical_cumulative |  99.27±0.04 |  99.47±0.03 |   0.31±0.03 |   0.14±0.03 |  99.54±0.03 |  99.67±0.02
val_cal              |  99.27±0.03 |  99.47±0.03 |   0.34±0.03 |   0.14±0.03 |  99.54±0.02 |  99.67±0.02

best_fitness  (runs: GS 3 | GA 3)
Grid Search : 367±231 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 126±53 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  99.27±0.03 |  99.51±0.03 |   0.30±0.09 |   0.12±0.00 |  99.54±0.02 |  99.69±0.02
fixed_05             |  99.03±0.08 |  99.12±0.09 |   0.58±0.06 |   0.53±0.07 |  99.38±0.05 |  99.44±0.06
platt                |  99.22±0.02 |  99.27±0.08 |   0.33±0.01 |   0.39±0.06 |  99.50±0.01 |  99.53±0.05
beta                 |  99.24±0.04 |  99.33±0.02 |   0.27±0.02 |   0.31±0.01 |  99.52±0.02 |  99.57±0.01
empirical            |  98.37±0.43 |  99.25±0.25 |   0.07±0.04 |   0.10±0.02 |  98.99±0.26 |  99.53±0.15
empirical_cumulative |  99.26±0.04 |  99.50±0.02 |   0.27±0.06 |   0.12±0.01 |  99.53±0.02 |  99.68±0.01
val_cal              |  99.27±0.03 |  99.51±0.03 |   0.30±0.09 |   0.12±0.00 |  99.54±0.02 |  99.69±0.02

##### RULE-7 5-tables: IDSXD(+2)-ciciot-quad-B15-AC (Grid vs GA, held-out TEST, mean±SD %) #####

best_f1  (runs: GS 5 | GA 5)
Grid Search : 140±65 neurons | 67.2±7.2 bits (mean per-neuron)
GA Neurons  : 177±54 neurons | 80.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.59±0.06 |  93.05±0.11 |  15.93±1.41 |   6.87±0.69 |  95.46±0.07 |  96.50±0.05
fixed_05             |  81.60±0.14 |  85.95±0.40 |   0.98±0.09 |   0.74±0.05 |  88.37±0.12 |  91.69±0.29
platt                |  90.35±0.15 |  92.95±0.11 |  12.32±0.66 |   8.95±0.27 |  95.16±0.11 |  96.51±0.06
beta                 |  90.55±0.10 |  92.72±0.09 |  15.89±0.36 |  11.09±0.50 |  95.43±0.05 |  96.45±0.04
empirical            |  70.23±2.37 |  67.16±2.69 |   0.02±0.03 |   0.01±0.02 |  77.51±2.58 |  74.11±3.02
empirical_cumulative |  89.96±0.66 |  92.85±0.17 |  10.94±3.90 |   4.99±0.49 |  94.86±0.56 |  96.33±0.11
val_cal              |  90.58±0.08 |  93.04±0.11 |  15.84±1.22 |   6.53±0.66 |  95.45±0.05 |  96.48±0.07

best_fpr  (runs: GS 5 | GA 5)
Grid Search : 200±50 neurons | 97.6±2.2 bits (mean per-neuron)
GA Neurons  : 174±53 neurons | 80.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.04±0.14 |  93.03±0.11 |  10.18±0.46 |   6.82±0.59 |  94.89±0.10 |  96.49±0.05
fixed_05             |  79.66±0.31 |  85.95±0.44 |   0.75±0.04 |   0.75±0.04 |  86.73±0.26 |  91.69±0.31
platt                |  89.88±0.20 |  92.94±0.10 |  16.30±0.70 |   9.00±0.30 |  95.08±0.08 |  96.51±0.06
beta                 |  89.53±0.25 |  92.70±0.10 |  19.04±0.74 |  11.15±0.53 |  95.01±0.11 |  96.44±0.05
empirical            |  62.11±2.60 |  67.37±2.65 |   0.01±0.00 |   0.02±0.02 |  68.23±3.14 |  74.35±2.98
empirical_cumulative |  89.63±0.09 |  92.77±0.24 |   8.33±0.57 |   4.87±0.49 |  94.55±0.06 |  96.28±0.15
val_cal              |  90.13±0.10 |  93.04±0.13 |  12.68±0.67 |   7.37±0.83 |  95.05±0.09 |  96.51±0.07

best_acc  (runs: GS 5 | GA 5)
Grid Search : 160±65 neurons | 64.0±0.0 bits (mean per-neuron)
GA Neurons  : 177±54 neurons | 80.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.46±0.20 |  93.05±0.11 |  16.55±0.33 |   6.87±0.69 |  95.41±0.12 |  96.50±0.05
fixed_05             |  81.64±0.07 |  85.95±0.40 |   1.01±0.04 |   0.74±0.05 |  88.41±0.05 |  91.69±0.29
platt                |  90.22±0.10 |  92.95±0.11 |  12.15±0.17 |   8.95±0.27 |  95.08±0.05 |  96.51±0.06
beta                 |  90.44±0.20 |  92.72±0.09 |  15.94±0.28 |  11.09±0.50 |  95.38±0.10 |  96.45±0.04
empirical            |  69.65±2.56 |  67.16±2.69 |   0.01±0.01 |   0.01±0.02 |  76.87±2.78 |  74.11±3.02
empirical_cumulative |  89.67±0.56 |  92.85±0.17 |  10.04±3.05 |   4.99±0.49 |  94.65±0.47 |  96.33±0.11
val_cal              |  90.45±0.18 |  93.04±0.11 |  15.99±1.26 |   6.53±0.66 |  95.39±0.15 |  96.48±0.07

best_ce  (runs: GS 5 | GA 5)
Grid Search : 210±65 neurons | 73.6±8.8 bits (mean per-neuron)
GA Neurons  : 175±53 neurons | 80.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.28±0.29 |  93.05±0.10 |  14.37±1.80 |   6.93±0.53 |  95.21±0.21 |  96.50±0.05
fixed_05             |  81.46±0.26 |  86.02±0.42 |   0.93±0.10 |   0.77±0.07 |  88.26±0.22 |  91.74±0.30
platt                |  90.20±0.25 |  92.97±0.10 |  13.14±0.85 |   8.96±0.31 |  95.11±0.14 |  96.52±0.06
beta                 |  90.15±0.39 |  92.71±0.09 |  16.55±0.62 |  11.15±0.45 |  95.24±0.19 |  96.44±0.04
empirical            |  66.59±3.56 |  67.09±2.70 |   0.02±0.03 |   0.01±0.02 |  73.43±4.01 |  74.03±3.03
empirical_cumulative |  89.87±0.48 |  92.84±0.18 |  10.53±3.39 |   5.03±0.46 |  94.80±0.43 |  96.33±0.11
val_cal              |  90.27±0.29 |  93.05±0.12 |  14.79±1.38 |   7.06±0.77 |  95.23±0.21 |  96.51±0.06

best_fitness  (runs: GS 5 | GA 5)
Grid Search : 160±65 neurons | 80.0±0.0 bits (mean per-neuron)
GA Neurons  : 176±54 neurons | 80.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.13±0.25 |  93.05±0.10 |  13.08±0.35 |   6.85±0.59 |  95.07±0.16 |  96.50±0.04
fixed_05             |  81.32±0.11 |  86.03±0.40 |   0.86±0.03 |   0.76±0.05 |  88.14±0.09 |  91.75±0.29
platt                |  90.12±0.26 |  92.97±0.09 |  13.82±0.24 |   8.92±0.32 |  95.10±0.14 |  96.51±0.06
beta                 |  89.96±0.30 |  92.72±0.09 |  16.82±0.39 |  11.10±0.43 |  95.15±0.16 |  96.45±0.05
empirical            |  67.86±3.85 |  67.40±2.65 |   0.03±0.03 |   0.02±0.02 |  74.85±4.30 |  74.39±2.98
empirical_cumulative |  89.65±0.26 |  92.85±0.17 |   8.89±2.17 |   4.96±0.43 |  94.59±0.24 |  96.32±0.10
val_cal              |  90.12±0.25 |  93.06±0.12 |  13.72±0.55 |   7.13±0.83 |  95.10±0.16 |  96.51±0.07

##### RULE-7 5-tables: IDSXD(+2)-ciciot-quad-B05-AC (Grid vs GA, held-out TEST, mean±SD %) #####

best_f1  (runs: GS 5 | GA 5)
Grid Search : 150±35 neurons | 64.0±19.6 bits (mean per-neuron)
GA Neurons  : 167±31 neurons | 70.4±8.8 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.75±0.11 |  93.07±0.20 |  16.73±1.23 |   7.18±1.05 |  95.58±0.10 |  96.52±0.08
fixed_05             |  81.43±0.32 |  86.16±0.55 |   1.00±0.07 |   0.80±0.12 |  88.23±0.27 |  91.84±0.39
platt                |  90.17±0.27 |  93.01±0.16 |  11.44±0.54 |   8.72±0.42 |  95.02±0.17 |  96.53±0.10
beta                 |  90.68±0.06 |  92.75±0.05 |  15.60±0.36 |  10.95±0.57 |  95.50±0.04 |  96.46±0.04
empirical            |  74.48±9.25 |  68.93±1.86 |   3.48±7.76 |   0.01±0.00 |  81.29±8.13 |  76.10±2.00
empirical_cumulative |  90.73±0.15 |  93.07±0.16 |  15.75±2.33 |   6.30±0.63 |  95.53±0.17 |  96.49±0.08
val_cal              |  90.73±0.12 |  93.09±0.21 |  16.56±1.51 |   7.52±0.64 |  95.57±0.11 |  96.54±0.10

best_fpr  (runs: GS 5 | GA 5)
Grid Search : 180±84 neurons | 78.4±19.7 bits (mean per-neuron)
GA Neurons  : 164±32 neurons | 70.4±8.8 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.03±0.10 |  93.09±0.19 |  12.92±2.56 |   7.11±0.79 |  95.01±0.10 |  96.53±0.08
fixed_05             |  80.90±1.25 |  86.10±0.54 |   0.90±0.14 |   0.80±0.13 |  87.77±1.05 |  91.80±0.39
platt                |  89.94±0.11 |  93.00±0.15 |  13.90±2.46 |   8.66±0.45 |  95.00±0.10 |  96.52±0.09
beta                 |  89.80±0.26 |  92.74±0.06 |  17.07±1.92 |  10.95±0.49 |  95.07±0.07 |  96.45±0.04
empirical            |  69.53±11.67 |  69.13±1.50 |   4.04±9.02 |   0.01±0.00 |  75.86±11.39 |  76.34±1.63
empirical_cumulative |  90.09±0.12 |  93.05±0.17 |  12.11±1.14 |   6.41±0.71 |  95.00±0.07 |  96.49±0.08
val_cal              |  90.08±0.14 |  93.07±0.19 |  13.65±0.60 |   7.75±0.44 |  95.07±0.07 |  96.54±0.09

best_acc  (runs: GS 5 | GA 5)
Grid Search : 150±35 neurons | 64.0±19.6 bits (mean per-neuron)
GA Neurons  : 167±31 neurons | 70.4±8.8 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.75±0.11 |  93.07±0.20 |  16.73±1.23 |   7.17±1.04 |  95.58±0.10 |  96.52±0.08
fixed_05             |  81.43±0.32 |  86.15±0.54 |   1.00±0.07 |   0.79±0.13 |  88.23±0.27 |  91.83±0.39
platt                |  90.17±0.27 |  93.01±0.16 |  11.44±0.54 |   8.72±0.42 |  95.02±0.17 |  96.53±0.10
beta                 |  90.68±0.06 |  92.75±0.05 |  15.60±0.36 |  10.95±0.56 |  95.50±0.04 |  96.46±0.04
empirical            |  74.48±9.25 |  69.07±1.86 |   3.48±7.76 |   0.01±0.00 |  81.29±8.13 |  76.26±2.01
empirical_cumulative |  90.73±0.15 |  93.08±0.17 |  15.75±2.33 |   6.42±0.52 |  95.53±0.17 |  96.50±0.08
val_cal              |  90.73±0.12 |  93.07±0.20 |  16.56±1.51 |   7.66±0.57 |  95.57±0.11 |  96.53±0.10

best_ce  (runs: GS 5 | GA 5)
Grid Search : 230±45 neurons | 70.4±8.8 bits (mean per-neuron)
GA Neurons  : 166±31 neurons | 70.4±8.8 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.19±0.24 |  93.05±0.21 |  15.05±1.64 |   7.24±1.03 |  95.19±0.18 |  96.51±0.08
fixed_05             |  81.58±0.26 |  86.17±0.58 |   0.93±0.04 |   0.80±0.14 |  88.35±0.21 |  91.85±0.42
platt                |  90.14±0.11 |  92.99±0.15 |  12.79±0.91 |   8.71±0.41 |  95.06±0.07 |  96.52±0.09
beta                 |  90.13±0.29 |  92.74±0.08 |  16.35±0.69 |  10.96±0.50 |  95.22±0.14 |  96.45±0.05
empirical            |  65.95±1.44 |  68.62±1.99 |   0.00±0.01 |   0.00±0.00 |  72.78±1.67 |  75.77±2.19
empirical_cumulative |  90.19±0.25 |  93.06±0.18 |  13.11±1.33 |   6.54±0.60 |  95.11±0.20 |  96.49±0.09
val_cal              |  90.21±0.22 |  93.05±0.20 |  14.84±1.08 |   7.72±0.40 |  95.20±0.17 |  96.52±0.10

best_fitness  (runs: GS 5 | GA 5)
Grid Search : 180±67 neurons | 70.4±8.8 bits (mean per-neuron)
GA Neurons  : 166±30 neurons | 70.4±8.8 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  90.28±0.25 |  93.07±0.19 |  15.02±1.59 |   7.08±0.80 |  95.25±0.19 |  96.52±0.08
fixed_05             |  81.61±0.22 |  86.20±0.59 |   0.94±0.04 |   0.77±0.11 |  88.38±0.18 |  91.87±0.43
platt                |  90.21±0.15 |  93.00±0.14 |  12.78±0.96 |   8.68±0.45 |  95.11±0.07 |  96.52±0.09
beta                 |  90.23±0.32 |  92.75±0.06 |  16.31±0.67 |  10.91±0.53 |  95.27±0.16 |  96.46±0.04
empirical            |  67.95±2.33 |  68.60±1.80 |   0.01±0.01 |   0.01±0.01 |  75.01±2.54 |  75.75±1.98
empirical_cumulative |  90.30±0.25 |  93.07±0.17 |  13.37±1.32 |   6.53±0.52 |  95.18±0.20 |  96.50±0.08
val_cal              |  90.29±0.24 |  93.05±0.20 |  15.35±1.29 |   7.78±0.53 |  95.27±0.19 |  96.53±0.10
```

### Reproduce
- Supplementary script (read-only): `python ids2_final.py {arms <ds> <dec> [seeds]|seeds <ds> <dec> <arm> <gt>|pareto <ds> <dec> <seeds|all> [fmin]|rule7 <ds> <dec> <arm>|extra}`.
  It was run from the agent scratchpad and needs promoting to `scripts/ids2_final_readout.py` (not committed by this readout).
- Core query (every number above is aggregated from exactly this):
  `SELECT f.id, f.name, f.started_at, e.phase_type, vs.genome_type, vs.genome_hash, vs.threshold_metadata FROM validation_summaries vs JOIN flows f ON f.id=vs.flow_id JOIN experiments e ON e.id=vs.experiment_id WHERE (f.name LIKE 'IDSXD-%' OR f.name LIKE 'IDSXD2-%') AND f.name NOT LIKE 'IDSXD-unswt-%' AND f.status='completed' AND vs.validation_point='final'`
  Parse the seed from the name suffix `-r(\d+)$` (this excludes `-w64fix` / `-abi13`). The phase is GA when `phase_type='ga_neurons'`.
  Metrics are `json(threshold_metadata)[mode].{f1,fpr,acc}` x 100, mean±sample SD over seeds.
- `scripts/idsxd_best_column.py` (the interim tool) still runs and agrees on every cell. Its column ranking is a max
  over ~70 columns per arm (best-of-N), and it labels every cicids seed "pre" (harmless, since cicids is fold-immune but
  the label is misleading). Use it for the per-arm max view only, not for the call.

## IDS-8 FINAL (02/10/2026) — UNSW temporal_3way quad-16b sweep

Read-only DB readout of `IDSXD-unswt-quad-16b-*` (flows 6237-6263). Every number is HELD-OUT TEST
(`validation_summaries.threshold_metadata`, `validation_point='final'`, temporal_3way, Protocol v2:
thresholds fit on the 10% VAL, reported on the 10% TEST). No `iterations.best_f1` anywhere.
Producer: `scripts/ids8/ids8_final_readout.py {audit|arms|paired|stats|seeds|pareto|rule7|refs}`.

```
Header (02/10/2026)
  IDSXD unswt quad-16b  9 arms x {20403,20404,20405}  27/27 completed, 0 failed | 54/54 experiments (GS + GA per flow)
  total 10.8 h | 24.1 min/run | first start 26/09/2026 12:07 UTC | last done 26/09/2026 23:00 UTC | ETA: complete
Constant params (verified from flows.config_json): unsw-nb15, temporal_3way, ids_n_bits=16 (thermometer), top20, binary,
  memory QUAD_WEIGHTED (worker default; no memory_mode key), fitness_aggregation=desirability, ce_anchor_normalized=0.2128,
  harmonic_rank, ids_k_folds=5 / kfold_per_gen=5, ga_generations=250, patience=5, population 50, neurons 5-500,
  neuron_sample_rate=0.25, OI training. Varying: weights, min_bits (B34-CTRL 34, all others 4; max_bits 34 everywhere), seed.
Weights (f1/fpr/ce/acc): B05-AC .05/.05/.225/.675  B05-CE .05/.05/.675/.225  B10-AC .10/.10/.20/.60  B10-CE .10/.10/.60/.20
  B15-AC .15/.15/.175/.525  B15-CE .15/.15/.525/.175  CE20 .30/.40/.20/.10  Wb-CTRL .35/.35/.10/.20
  B34-CTRL .35/.35/.10/.20 with bits pinned 34-34 (the production 500n x 34b control)
```

### DECISION (Luiz, 02/10/2026) — CHOSEN config for IDS-3 = **B15-CE** (supersedes the CE20 pick below)

Still NO statistically supported winner; this is a config decision. B15-CE = weights f1/fpr/ce/acc
.15/.15/.525/.175, desirability (anchor 0.2128), QUAD_WEIGHTED, 16b thermometer, bits 4-34, neuron cap 500.

```
GA best_fitness val_cal (n=3)   F1 89.74±0.22 | FPR  8.84±0.59 | Acc 89.80±0.22 | 270±57n x 33.0b
  vs B34-CTRL (paired)          dF1 +0.40±0.48 [2/3] | dFPR -0.63±1.07 [2/3]
  vs Wb-CTRL  (paired)          dF1 +0.59±0.34 [3/3] | dFPR -0.30±1.04 [1/3]
  low-FPR (empirical_cumulative) 88.49±0.44 / 4.47±0.60
Best found (single genome, anywhere — best-of-210, report beside the mean):
  top F1      89.87 / 8.18 / 89.93  GA best_fitness val_cal              r20403 flow 6242  205n x 32.0b
  low FPR     88.07 / 3.80 / 88.07  GA best_fitness empirical_cumulative r20404 flow 6251  289n x 33.2b
```
Why B15-CE over CE20 / B15-AC: highest mean F1 and Acc of the sweep, with a mean FPR below both controls
and B15-AC (10.28); its best single genome (89.87) ties the sweep top (B15-AC 89.90) at 1.3 pp lower FPR; and
it reaches that with the SMALLEST genomes of the top arms (~270n vs CE20 ~396n, B10-CE ~397n, B34-CTRL ~445n),
which matters for the footprint claim. What it gives up: CE20's mean FPR is ~1.0 pp lower (7.84) and CE20 was the
only arm 3/3 better on FPR vs both controls; B15-CE's FPR edge over Wb-CTRL is 1/3 seeds. Every gap is under
the n=3 MDD (~0.45 pp F1 / ~1.23 pp FPR). Desirability-only — re-read if IDS-20 favours zscore.
Best-anywhere scan: `python3 scripts/ids8/ids8_final_readout.py best any <f1_floor>`.

### Verdict (original, 02/10/2026 agent readout) — NO supported winner; CHOSEN config for IDS-3 = CE20 (desirability)

No arm separates from either control beyond seed noise. Two-way (arm x seed) residual sigma = 0.26 pp F1 /
0.71 pp FPR (df 16), so the paired n=3 MDD is ~0.45 pp F1 / ~1.23 pp FPR. After Holm over the 8 contrasts per
control, the ONLY Holm-significant contrast in the sweep is a HARM (B05-CE FPR +1.30 pp vs Wb-CTRL, best_fitness, Holm p 0.01).
No F1 contrast survives (best: B15-CE +0.59 vs Wb-CTRL, 3/3 seeds, raw p 0.095).

```
CHOSEN (UNSW temporal_3way, for IDS-3 n=100):
  weights f1/fpr/ce/acc = .30/.40/.20/.10 (CE20) | aggregation desirability (anchor_normalized 0.2128) | harmonic_rank
  bits/neuron band 4-34 (GA winners 33.5±0.9 mean b/neuron) | neuron cap 500 (GA winners 395±153) | min_neurons 5
  memory QUAD_WEIGHTED | ids_n_bits 16 thermometer | top20 | binary | k-fold 5x5 | gens 250 | patience 5
  held-out val_cal, GA best_fitness (pre-declared column) : F1 89.10±0.28 / FPR 7.84±0.24 / Acc 89.14±0.28  (n=3)
  held-out val_cal, GA best_f1                            : F1 89.16±0.31 / FPR 8.14±0.29 / Acc 89.21±0.32  (n=3)
  low-FPR operating point, GA best_f1 empirical_cumulative: F1 87.10±0.48 / FPR 3.44±1.18 / Acc 87.10±0.48  (VAL-fit, legit)
```
Why CE20 (a decision, not a claim):
1. It is the only arm whose direction holds 3/3 seeds against BOTH controls on the deployment axis, on both
   genome types: FPR −1.62 / −1.29 pp (best_fitness, vs B34 / Wb), −1.25 / −1.66 pp (best_f1). Magnitude is at
   or above the 1.23 pp MDD; raw p 0.010-0.031, Holm 0.08-0.25, so NOT significant.
2. Its F1 cost is inside noise: −0.24 / −0.05 pp (best_fitness), −0.15 / −0.02 pp (best_f1); MDD 0.45 pp.
3. Independent prior on this exact dataset/bits: CE20 was the ONLY supported winner of IDSZ (zscore, n=5,
   +0.95 pp F1 vs Wb-CTRL, p 0.0006, survived budget matching) with FPR also lower (−0.70 pp). Here, under
   desirability, the F1 half did NOT replicate (−0.05 pp) and the FPR half did, larger.
4. Same weight vector as the IDS-2 cicids pick, so the S&P cohorts carry one fewer config.
Alternatives (both legitimate, neither supported):
- **B10-CE** (.10/.10/.60/.20) is the only arm better than BOTH controls on BOTH axes in 3/3 seeds, on both genome
  types (best_fitness: F1 +0.30 / +0.48, FPR −0.64 / −0.31 vs B34 / Wb). Every magnitude is below the MDD. In IDSZ,
  its F1 lead over Wb-CTRL (+0.73, 5/5) collapsed under budget matching, and here it again ran the longest GA
  (127 gens mean vs 90-93 for the controls, 113 for CE20).
- **B15-CE** (.15/.15/.525/.175) has the highest F1 in the sweep, 89.74±0.22 / 8.84±0.59 (best_fitness): +0.59 vs Wb (3/3),
  +0.40 vs B34 (2/3). Its FPR is not better than the controls'. Pick it only if IDS-3 is meant to maximise F1.
The aggregation half of the config (desirability vs zscore) is IDS-20's call. If IDSAGG flips it, re-read this choice.
IDSZ zscore CE20 at 89.83±0.24 / 8.64±0.77 is UNPAIRED (different seeds, pre-WNN-1 / pre-immigrant-fix code era), so it
cannot be attributed to the aggregation.

### Per-arm table — headline GA val_cal, paired by seed vs B34-CTRL and Wb-CTRL (n=3 each)
```
GA best_fitness val_cal, n=3 seeds each; delta = arm − control, paired by seed (pp); [k/3] = seeds where the arm is better
arm      | F1           | FPR          | Acc          | dF1 vs B34       | dFPR vs B34      | dF1 vs Wb        | dFPR vs Wb       | GA shape
---------+--------------+--------------+--------------+------------------+------------------+------------------+------------------+---------------
B05-AC   |  89.01±0.24 |   8.61±0.62 |  89.07±0.25 | -0.32±0.35 [0/3] | -0.86±0.98 [3/3] | -0.14±0.23 [1/3] | -0.53±1.04 [2/3] | 312n x 33.6b
B05-CE   |  89.38±0.26 |  10.43±0.45 |  89.46±0.25 | +0.04±0.15 [2/3] | +0.97±0.34 [0/3] | +0.23±0.13 [3/3] | +1.30±0.10 [0/3] | 343n x 33.3b
B10-AC   |  88.95±0.22 |   9.09±1.60 |  89.02±0.22 | -0.38±0.06 [0/3] | -0.37±1.68 [1/3] | -0.20±0.10 [0/3] | -0.04±1.43 [1/3] | 300n x 33.7b
B10-CE   |  89.63±0.21 |   8.82±0.57 |  89.70±0.21 | +0.30±0.28 [3/3] | -0.64±0.28 [3/3] | +0.48±0.25 [3/3] | -0.31±0.12 [3/3] | 397n x 34.0b
B15-AC   |  89.34±0.44 |  10.28±0.58 |  89.42±0.44 | +0.01±0.53 [2/3] | +0.82±0.99 [0/3] | +0.19±0.43 [2/3] | +1.15±0.83 [0/3] | 369n x 34.0b
B15-CE   |  89.74±0.22 |   8.84±0.59 |  89.80±0.22 | +0.40±0.48 [2/3] | -0.63±1.07 [2/3] | +0.59±0.34 [3/3] | -0.30±1.04 [1/3] | 270n x 33.0b
B34-CTRL |  89.34±0.27 |   9.46±0.49 |  89.40±0.26 |        —         |        —         | +0.18±0.14 [3/3] | +0.33±0.24 [0/3] | 445n x 34.0b
CE20     |  89.10±0.28 |   7.84±0.24 |  89.14±0.28 | -0.24±0.54 [1/3] | -1.62±0.50 [3/3] | -0.05±0.41 [1/3] | -1.29±0.35 [3/3] | 396n x 33.5b
Wb-CTRL  |  89.15±0.14 |   9.14±0.45 |  89.21±0.14 | -0.18±0.14 [0/3] | -0.33±0.24 [3/3] |        —         |        —         | 318n x 33.6b

GA best_f1 val_cal, n=3 seeds each; delta = arm − control, paired by seed (pp); [k/3] = seeds where the arm is better
arm      | F1           | FPR          | Acc          | dF1 vs B34       | dFPR vs B34      | dF1 vs Wb        | dFPR vs Wb       | GA shape
---------+--------------+--------------+--------------+------------------+------------------+------------------+------------------+---------------
B05-AC   |  89.02±0.24 |   8.59±0.64 |  89.07±0.25 | -0.30±0.34 [0/3] | -0.79±0.84 [3/3] | -0.17±0.21 [1/3] | -1.21±0.74 [3/3] | 311n x 33.6b
B05-CE   |  89.37±0.29 |   9.94±1.14 |  89.45±0.28 | +0.06±0.18 [2/3] | +0.56±0.78 [1/3] | +0.19±0.12 [3/3] | +0.14±1.19 [1/3] | 350n x 33.3b
B10-AC   |  88.93±0.22 |   8.90±1.31 |  88.99±0.21 | -0.39±0.09 [0/3] | -0.48±1.35 [1/3] | -0.25±0.15 [0/3] | -0.90±1.20 [2/3] | 301n x 33.7b
B10-CE   |  89.61±0.25 |   8.89±0.59 |  89.68±0.26 | +0.30±0.33 [3/3] | -0.49±0.49 [3/3] | +0.43±0.35 [3/3] | -0.91±0.56 [3/3] | 398n x 34.0b
B15-AC   |  89.33±0.45 |  10.36±0.55 |  89.42±0.44 | +0.02±0.52 [2/3] | +0.98±0.88 [0/3] | +0.15±0.40 [2/3] | +0.56±0.40 [0/3] | 368n x 34.0b
B15-CE   |  89.74±0.20 |   8.76±0.52 |  89.80±0.20 | +0.42±0.46 [2/3] | -0.62±0.87 [2/3] | +0.56±0.34 [3/3] | -1.04±0.54 [3/3] | 268n x 33.0b
B34-CTRL |  89.32±0.27 |   9.38±0.40 |  89.38±0.27 |        —         |        —         | +0.13±0.14 [3/3] | -0.42±0.50 [3/3] | 457n x 34.0b
CE20     |  89.16±0.31 |   8.14±0.29 |  89.21±0.32 | -0.15±0.57 [2/3] | -1.25±0.22 [3/3] | -0.02±0.44 [2/3] | -1.66±0.43 [3/3] | 396n x 33.5b
Wb-CTRL  |  89.18±0.18 |   9.80±0.15 |  89.25±0.18 | -0.13±0.14 [0/3] | +0.42±0.50 [0/3] |        —         |        —         | 319n x 33.6b

Seed noise (two-way arm x seed residual, df 16): best_fitness 0.259 pp F1 / 0.711 pp FPR; best_f1 0.270 / 0.726.
Paired n=3 MDD (alpha .05): ~0.45 pp F1 / ~1.23 pp FPR. Paired t, df 2, Holm over 8 arms per control:
  F1  : nothing below raw p 0.07 except B10-AC −0.38 vs B34 (p 0.007, Holm 0.06, WORSE).
  FPR : CE20 vs B34 p 0.031 / vs Wb p 0.023 (best_fitness), 0.010 / 0.021 (best_f1); Holm 0.08-0.25.
        B05-CE vs Wb +1.30 (WORSE) p 0.002, Holm 0.01: the one Holm-significant contrast, and it is a harm.
GA generations used (patience 5): B05-AC 83, B10-AC 93, Wb-CTRL 93, B34-CTRL 90, CE20 113, B05-CE 120, B15-CE 120,
  B15-AC 123, B10-CE 127 (mean of 3). Gens are a MEDIATOR (§8.3), not adjusted for.
GA gain over grid (best_fitness val_cal): F1 +0.40..+1.05 pp on every arm; FPR −1.37 (CE20), −0.75 (Wb), −0.12 (B10-CE),
  +0.06..+1.91 elsewhere — CE20 is the only arm whose GA phase buys FPR as well as F1.
```

### Validity checks
1. **27/27 completed, 0 failed, 2 experiments each** (grid_search + ga_neurons), 270 final + 135 init validation rows.
2. **Immigrant min_bits (IDS-17): NO hybrids.** Every flow was created 16/09 21:47 UTC but started 26/09 12:07-22:48 UTC,
   all after the 17/09 01:57 UTC fix (129f8533). Census over all 7,729 tracked genomes in the 54 experiments (both
   formats, expanded per neuron): 0 neurons outside [min_bits, max_bits]. B34-CTRL min = max = 34 on every
   neuron of all 3 seeds (42,132 / 39,428 / 44,907 neurons), so it IS a clean fixed-34b control here, unlike
   IDSXD-unswr-quad B34-CTRL r20401-04.
3. **WNN-1 era (counter-RNG offspring seeds, worker ABI 14; swap at ~19:09 UTC 26/09).** Wall-clock-seeded: 17 flows
   (all of s20403, all of s20404 except Wb-CTRL). Counter-RNG: 10 flows (all of s20405, plus Wb-CTRL s20404 6254,
   which was re-queued after the swap race (WNN-2) and ran 19:31 UTC). The seed block is therefore nearly the era.
   There is no visible shift: arm-mean GA best_fitness val_cal per seed block is 89.20 / 89.39 / 89.29 F1 and
   9.39 / 8.88 / 9.23 FPR. Wb-CTRL s20404 is the one cross-era pair in the paired table. The pre-swap 17 flows are not
   seed-reproducible (wall-clock offspring), but they are valid samples.
4. **Leak hygiene:** GA fitness = 5-fold CV on the 80% train. Thresholds fit on VAL, numbers on TEST. `empirical`
   and `empirical_cumulative` are VAL-fit, so they are legitimate operating points. Plain `empirical` collapses
   (F1 81-84), so use `empirical_cumulative` for the low-FPR point.
5. **Aggregation is not an arm.** All 27 runs use desirability, so "CE20" here means CE20 weights under desirability.

### Reference comparison (UNSW temporal_3way, 16b, held-out TEST; pair modes like-for-like)
```
source                                         | n   | mode     | F1           | FPR          | Acc          | note
-----------------------------------------------+-----+----------+--------------+--------------+--------------+------------------------------
IDS-8 CE20 (CHOSEN) GA best_fitness            |   3 | val_cal  | 89.10±0.28   |  7.84±0.24   | 89.14±0.28   | desirability, this readout
IDS-8 CE20 GA best_f1                          |   3 | emp_cum  | 87.10±0.48   |  3.44±1.18   | 87.10±0.48   | low-FPR operating point
IDS-8 B15-CE GA best_fitness (F1 max)          |   3 | val_cal  | 89.74±0.22   |  8.84±0.59   | 89.80±0.22   |
IDS-8 B34-CTRL GA best_fitness (prod. 34b)     |   3 | val_cal  | 89.34±0.27   |  9.46±0.49   | 89.40±0.26   |
IDS-8 Wb-CTRL GA best_fitness                  |   3 | val_cal  | 89.15±0.14   |  9.14±0.45   | 89.21±0.14   |
IDSZ CE20 GA best_fitness (zscore)             |   5 | val_cal  | 89.83±0.24   |  8.64±0.77   | 89.89±0.23   | UNPAIRED, Aug code era
IDSZ Wb-CTRL GA best_fitness (zscore)          |   5 | val_cal  | 88.84±0.15   |  9.29±1.05   | 88.91±0.15   | UNPAIRED, Aug code era
IDSZ CE20 best_fpr (zscore, banked)            |   5 | emp_cum  | 87.59±0.45   |  2.87±0.10   |      —       | memory project_idsz_n5_complete
SP100-unswt-quad-16bWb GA best_fitness         |  29 | val_cal  | 86.41±1.98   | 14.17±7.22   | 86.59±1.85   | pre-19/08 code era, NOT a control
XGBoost 16b thermo (IDS-9, measured 13/09)     |   — | val_cal  | 92.09        |  3.64        | 92.11        | 0.27 MB
RF 16b thermo (IDS-9, measured 13/09)          |   — | val_cal  | 90.66        |  7.61        | 90.71        | 138 MB
XGBoost / RF 16b                               |   — | fixed_05 | 84.96 / 86.12| 28.42 / 25.35| 85.64 / 86.65|
IDS-8 CE20 GA best_fitness                     |   3 | fixed_05 | 85.27±0.08   | 24.94±0.20   | 85.77±0.07   |
```
Where CE20 sits: under equal val-calibration it TRAILS XGBoost by −2.99 pp F1 and +4.20 pp FPR, and RF by −1.56 pp F1
at equal FPR (+0.23). Even CE20's low-FPR point (87.10 / 3.44) is dominated by XGBoost val_cal (92.09 / 3.64).
The fixed_05 rows favour the WNN over XGBoost (+0.31 F1 / −3.48 FPR), but that is not the deployment mode. **On UNSW temporal,
the trees lead under equal calibration. This sweep did not change the IDS-9 conclusion.** The surviving argument is size,
but IDS-9's "~1 KB WNN" figure was for small genomes. The chosen GA winners are ~400 neurons x ~34 b (sparse), so the
footprint must be MEASURED for these genomes before it is set against XGB's 0.27 MB. Against the prior WNN rows, this sweep
is a clear step up from SP100 (+2.69 F1, −6.33 FPR; different code era, not a control) and sits inside the IDSZ arm range
(88.7-89.8 F1, unpaired).

### Pareto scan (9 arms x GS/GA x 5 genome types x 7 modes = 630 columns; non-dominated on mean F1 vs mean FPR)
```
B15-CE    GA best_ce      val_cal              n=3 F1  89.75±0.21 FPR   8.89±0.47 Acc  89.82±0.20
B15-CE    GA best_f1      val_cal              n=3 F1  89.74±0.20 FPR   8.76±0.52 Acc  89.80±0.20
B15-CE    GA best_acc     val_cal              n=3 F1  89.74±0.20 FPR   8.76±0.52 Acc  89.80±0.20
B15-AC    GA best_fpr     empirical_cumulative n=3 F1  89.23±0.58 FPR   8.74±0.64 Acc  89.28±0.60
B05-CE    GA best_f1      empirical_cumulative n=3 F1  89.18±0.51 FPR   7.59±0.89 Acc  89.22±0.52
B05-CE    GA best_acc     empirical_cumulative n=3 F1  89.18±0.51 FPR   7.59±0.89 Acc  89.22±0.52
B05-CE    GA best_ce      empirical_cumulative n=3 F1  89.01±0.39 FPR   6.90±0.55 Acc  89.04±0.40
B10-CE    GA best_ce      empirical_cumulative n=3 F1  88.98±0.24 FPR   5.78±0.83 Acc  89.00±0.25
B10-CE    GA best_fitness empirical_cumulative n=3 F1  88.92±0.08 FPR   5.60±0.35 Acc  88.94±0.08
B10-CE    GA best_f1      empirical_cumulative n=3 F1  88.86±0.06 FPR   5.52±0.36 Acc  88.88±0.06
B10-CE    GA best_acc     empirical_cumulative n=3 F1  88.86±0.06 FPR   5.52±0.36 Acc  88.88±0.06
B15-CE    GA best_ce      empirical_cumulative n=3 F1  88.51±0.45 FPR   4.37±0.49 Acc  88.51±0.45
Wb-CTRL   GA best_f1      empirical_cumulative n=3 F1  87.26±0.65 FPR   4.22±0.32 Acc  87.26±0.65
Wb-CTRL   GA best_acc     empirical_cumulative n=3 F1  87.26±0.65 FPR   4.22±0.32 Acc  87.26±0.65
CE20      GA best_f1      empirical_cumulative n=3 F1  87.10±0.48 FPR   3.44±1.18 Acc  87.10±0.48
CE20      GA best_acc     empirical_cumulative n=3 F1  87.10±0.48 FPR   3.44±1.18 Acc  87.10±0.48
CE20      GA best_fpr     empirical_cumulative n=3 F1  86.92±0.41 FPR   3.33±1.02 Acc  86.93±0.41
CE20      GA best_ce      empirical_cumulative n=3 F1  86.86±0.47 FPR   3.26±0.93 Acc  86.86±0.47
(val_cal front = B15-CE alone at the F1 end. Below 9% FPR the front is all empirical_cumulative: B05-CE → B10-CE
 (88.92/5.60) → B15-CE best_ce (88.51/4.37) → Wb-CTRL (87.26/4.22) → CE20 (87.10/3.44, ~3.3 for best_fpr/best_ce).
 The SDs on these emp_cum points are 0.4-1.2 pp FPR, so the ordering below 6% FPR is not resolved at n=3.)
```

### Rule-7 5-tables (Grid vs GA, all 7 modes, mean±SD %, held-out TEST) — CE20 (chosen), B15-CE, B34-CTRL, Wb-CTRL
```

##### RULE-7 5-tables: IDSXD-unswt-quad-16b-CE20 (Grid vs GA, held-out TEST, mean±SD %) #####

best_f1  (runs: GS 3 | GA 3)
Grid Search : 167±58 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 396±154 neurons | 33.5±0.9 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.17±0.20 |  84.27±0.15 |  28.16±0.81 |  28.17±0.67 |  84.82±0.16 |  84.92±0.12
fixed_05             |  85.36±0.25 |  85.25±0.13 |  24.39±0.63 |  25.01±0.26 |  85.83±0.23 |  85.74±0.12
platt                |  87.19±0.16 |  87.38±0.32 |  17.52±0.38 |  17.55±0.51 |  87.42±0.15 |  87.61±0.31
beta                 |  88.12±0.31 |  88.49±0.31 |  13.57±1.66 |  12.27±0.78 |  88.25±0.28 |  88.60±0.30
empirical            |  83.34±1.47 |  81.57±2.31 |   3.11±1.02 |   1.21±0.73 |  83.37±1.45 |  81.68±2.23
empirical_cumulative |  87.26±1.58 |  87.10±0.48 |   6.09±1.95 |   3.44±1.18 |  87.28±1.60 |  87.10±0.48
val_cal              |  88.62±0.05 |  89.16±0.31 |   9.27±0.74 |   8.14±0.29 |  88.68±0.04 |  89.21±0.32

best_fpr  (runs: GS 3 | GA 3)
Grid Search : 400±0 neurons | 31.3±3.1 bits (mean per-neuron)
GA Neurons  : 395±153 neurons | 33.5±0.9 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  83.93±0.14 |  84.22±0.20 |  28.73±0.40 |  28.35±0.59 |  84.61±0.12 |  84.88±0.17
fixed_05             |  85.02±0.07 |  85.28±0.08 |  25.40±0.24 |  24.90±0.17 |  85.53±0.06 |  85.77±0.08
platt                |  86.85±0.02 |  87.40±0.32 |  18.49±0.08 |  17.51±0.49 |  87.10±0.02 |  87.63±0.31
beta                 |  87.77±0.08 |  88.47±0.31 |  13.99±0.24 |  12.30±0.74 |  87.91±0.08 |  88.58±0.30
empirical            |  78.77±0.48 |  81.72±2.49 |   1.20±0.33 |   1.25±0.77 |  78.97±0.45 |  81.82±2.40
empirical_cumulative |  87.08±0.85 |  86.92±0.41 |   5.54±0.91 |   3.33±1.02 |  87.09±0.85 |  86.93±0.41
val_cal              |  88.47±0.02 |  89.10±0.28 |   9.30±0.82 |   7.84±0.24 |  88.53±0.01 |  89.14±0.28

best_acc  (runs: GS 3 | GA 3)
Grid Search : 167±58 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 396±154 neurons | 33.5±0.9 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.17±0.20 |  84.27±0.15 |  28.16±0.81 |  28.17±0.67 |  84.82±0.16 |  84.92±0.12
fixed_05             |  85.36±0.25 |  85.25±0.13 |  24.39±0.63 |  25.01±0.26 |  85.83±0.23 |  85.74±0.12
platt                |  87.19±0.16 |  87.38±0.32 |  17.52±0.38 |  17.55±0.51 |  87.42±0.15 |  87.61±0.31
beta                 |  88.12±0.31 |  88.49±0.31 |  13.57±1.66 |  12.27±0.78 |  88.25±0.28 |  88.60±0.30
empirical            |  83.34±1.47 |  81.57±2.31 |   3.11±1.02 |   1.21±0.73 |  83.37±1.45 |  81.68±2.23
empirical_cumulative |  87.26±1.58 |  87.10±0.48 |   6.09±1.95 |   3.44±1.18 |  87.28±1.60 |  87.10±0.48
val_cal              |  88.62±0.05 |  89.16±0.31 |   9.27±0.74 |   8.14±0.29 |  88.68±0.04 |  89.21±0.32

best_ce  (runs: GS 3 | GA 3)
Grid Search : 367±231 neurons | 33.3±1.2 bits (mean per-neuron)
GA Neurons  : 398±155 neurons | 33.5±0.9 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.30±0.55 |  84.21±0.20 |  27.72±1.60 |  28.36±0.68 |  84.93±0.47 |  84.87±0.17
fixed_05             |  85.14±0.22 |  85.24±0.12 |  25.12±0.75 |  25.05±0.24 |  85.64±0.19 |  85.73±0.11
platt                |  87.06±0.08 |  87.39±0.31 |  18.15±0.30 |  17.54±0.47 |  87.30±0.07 |  87.62±0.30
beta                 |  87.96±0.12 |  88.48±0.30 |  13.38±0.36 |  12.30±0.75 |  88.09±0.12 |  88.59±0.29
empirical            |  81.70±4.77 |  81.71±2.25 |   2.37±2.18 |   1.25±0.75 |  81.84±4.66 |  81.81±2.17
empirical_cumulative |  87.36±0.86 |  86.86±0.47 |   5.21±0.84 |   3.26±0.93 |  87.37±0.87 |  86.86±0.47
val_cal              |  88.60±0.10 |  89.16±0.32 |   8.34±0.75 |   8.36±0.27 |  88.64±0.09 |  89.21±0.32

best_fitness  (runs: GS 3 | GA 3)
Grid Search : 367±153 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 396±154 neurons | 33.5±0.9 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  83.76±0.41 |  84.22±0.20 |  29.28±1.29 |  28.35±0.59 |  84.47±0.34 |  84.88±0.17
fixed_05             |  85.27±0.25 |  85.27±0.08 |  24.59±0.85 |  24.94±0.20 |  85.74±0.21 |  85.77±0.07
platt                |  87.05±0.21 |  87.40±0.31 |  18.02±0.57 |  17.55±0.46 |  87.29±0.19 |  87.63±0.30
beta                 |  88.02±0.17 |  88.47±0.30 |  13.30±0.54 |  12.32±0.71 |  88.14±0.16 |  88.58±0.29
empirical            |  79.56±1.31 |  81.71±2.49 |   1.27±0.37 |   1.26±0.76 |  79.73±1.25 |  81.82±2.41
empirical_cumulative |  87.19±0.20 |  86.91±0.43 |   5.11±0.23 |   3.33±1.01 |  87.20±0.20 |  86.91±0.43
val_cal              |  88.62±0.17 |  89.10±0.28 |   9.21±0.92 |   7.84±0.24 |  88.68±0.16 |  89.14±0.28

##### RULE-7 5-tables: IDSXD-unswt-quad-16b-B15-CE (Grid vs GA, held-out TEST, mean±SD %) #####

best_f1  (runs: GS 3 | GA 3)
Grid Search : 167±115 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 268±56 neurons | 33.0±1.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  83.90±0.36 |  84.74±0.33 |  28.76±0.86 |  27.04±0.94 |  84.58±0.31 |  85.33±0.28
fixed_05             |  85.30±0.24 |  84.91±0.40 |  24.41±0.62 |  26.52±1.11 |  85.76±0.22 |  85.48±0.35
platt                |  87.11±0.29 |  88.12±0.32 |  17.66±0.65 |  16.15±0.90 |  87.34±0.28 |  88.32±0.30
beta                 |  87.82±0.53 |  89.56±0.26 |  14.19±2.05 |  10.59±0.34 |  87.97±0.49 |  89.65±0.26
empirical            |  84.39±2.54 |  83.85±2.38 |   3.11±1.52 |   1.27±0.49 |  84.42±2.49 |  83.90±2.32
empirical_cumulative |  87.83±0.12 |  88.50±0.46 |   5.99±0.42 |   4.51±0.58 |  87.84±0.13 |  88.50±0.47
val_cal              |  88.47±0.33 |  89.74±0.20 |   8.86±0.43 |   8.76±0.52 |  88.52±0.34 |  89.80±0.20

best_fpr  (runs: GS 3 | GA 3)
Grid Search : 300±100 neurons | 32.7±1.2 bits (mean per-neuron)
GA Neurons  : 269±60 neurons | 33.0±1.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.00±0.34 |  84.74±0.42 |  28.68±0.91 |  27.05±1.24 |  84.68±0.30 |  85.33±0.36
fixed_05             |  85.15±0.43 |  84.84±0.22 |  25.02±1.23 |  26.70±0.64 |  85.65±0.38 |  85.42±0.19
platt                |  86.85±0.32 |  88.07±0.26 |  18.41±0.77 |  16.27±0.72 |  87.09±0.30 |  88.28±0.24
beta                 |  87.91±0.27 |  89.55±0.24 |  13.88±0.67 |  10.64±0.37 |  88.05±0.26 |  89.64±0.25
empirical            |  80.83±1.03 |  83.82±1.90 |   1.76±0.33 |   1.25±0.44 |  80.94±0.99 |  83.86±1.87
empirical_cumulative |  88.17±0.56 |  88.49±0.45 |   6.87±0.86 |   4.45±0.53 |  88.20±0.58 |  88.50±0.45
val_cal              |  88.60±0.08 |  89.74±0.22 |   8.75±0.27 |   8.94±0.54 |  88.65±0.09 |  89.81±0.22

best_acc  (runs: GS 3 | GA 3)
Grid Search : 167±115 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 268±56 neurons | 33.0±1.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  83.90±0.36 |  84.74±0.33 |  28.76±0.86 |  27.04±0.94 |  84.58±0.31 |  85.33±0.28
fixed_05             |  85.30±0.24 |  84.91±0.40 |  24.41±0.62 |  26.52±1.11 |  85.76±0.22 |  85.48±0.35
platt                |  87.11±0.29 |  88.12±0.32 |  17.66±0.65 |  16.15±0.90 |  87.34±0.28 |  88.32±0.30
beta                 |  87.82±0.53 |  89.56±0.26 |  14.19±2.05 |  10.59±0.34 |  87.97±0.49 |  89.65±0.26
empirical            |  84.39±2.54 |  83.85±2.38 |   3.11±1.52 |   1.27±0.49 |  84.42±2.49 |  83.90±2.32
empirical_cumulative |  87.83±0.12 |  88.50±0.46 |   5.99±0.42 |   4.51±0.58 |  87.84±0.13 |  88.50±0.47
val_cal              |  88.47±0.33 |  89.74±0.20 |   8.86±0.43 |   8.76±0.52 |  88.52±0.34 |  89.80±0.20

best_ce  (runs: GS 3 | GA 3)
Grid Search : 433±58 neurons | 33.3±1.2 bits (mean per-neuron)
GA Neurons  : 268±57 neurons | 33.0±1.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.21±0.36 |  84.76±0.39 |  28.05±1.01 |  26.95±1.12 |  84.85±0.31 |  85.36±0.33
fixed_05             |  85.45±0.05 |  84.83±0.23 |  24.29±0.13 |  26.73±0.66 |  85.91±0.04 |  85.41±0.20
platt                |  87.16±0.16 |  88.08±0.24 |  17.72±0.33 |  16.25±0.68 |  87.39±0.15 |  88.28±0.22
beta                 |  88.23±0.07 |  89.56±0.23 |  12.75±0.40 |  10.60±0.39 |  88.35±0.06 |  89.65±0.23
empirical            |  79.25±0.75 |  83.87±1.87 |   1.14±0.15 |   1.25±0.44 |  79.44±0.71 |  83.92±1.84
empirical_cumulative |  87.68±0.95 |  88.51±0.45 |   5.74±1.63 |   4.37±0.49 |  87.70±0.97 |  88.51±0.45
val_cal              |  88.74±0.04 |  89.75±0.21 |   8.55±1.06 |   8.89±0.47 |  88.79±0.03 |  89.82±0.20

best_fitness  (runs: GS 3 | GA 3)
Grid Search : 333±115 neurons | 32.7±1.2 bits (mean per-neuron)
GA Neurons  : 270±57 neurons | 33.0±1.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.19±0.37 |  84.74±0.29 |  28.12±1.06 |  26.98±0.75 |  84.84±0.32 |  85.33±0.25
fixed_05             |  85.44±0.05 |  84.89±0.37 |  24.27±0.15 |  26.57±1.01 |  85.90±0.04 |  85.47±0.32
platt                |  87.17±0.18 |  88.12±0.32 |  17.64±0.45 |  16.14±0.89 |  87.40±0.17 |  88.32±0.30
beta                 |  88.32±0.16 |  89.55±0.24 |  12.52±0.28 |  10.62±0.26 |  88.44±0.16 |  89.64±0.24
empirical            |  80.47±1.83 |  84.14±1.97 |   1.50±0.59 |   1.31±0.45 |  80.60±1.75 |  84.18±1.93
empirical_cumulative |  87.68±0.95 |  88.49±0.44 |   5.76±1.62 |   4.47±0.60 |  87.70±0.97 |  88.50±0.45
val_cal              |  88.85±0.15 |  89.74±0.22 |   8.33±0.70 |   8.84±0.59 |  88.89±0.16 |  89.80±0.22

##### RULE-7 5-tables: IDSXD-unswt-quad-16b-B34-CTRL (Grid vs GA, held-out TEST, mean±SD %) #####

best_f1  (runs: GS 3 | GA 3)
Grid Search : 70±113 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 457±31 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.53±0.55 |  84.02±0.31 |  27.19±1.65 |  29.03±0.81 |  85.13±0.47 |  84.73±0.27
fixed_05             |  86.22±1.50 |  85.92±0.20 |  21.64±4.31 |  23.16±0.70 |  86.60±1.35 |  86.34±0.17
platt                |  87.70±0.19 |  87.91±0.15 |  16.03±0.83 |  16.33±0.44 |  87.89±0.17 |  88.11±0.14
beta                 |  86.42±1.50 |  89.01±0.12 |   7.39±5.21 |  11.39±0.23 |  86.46±1.56 |  89.11±0.12
empirical            |  86.11±3.29 |  79.57±2.28 |   9.50±6.66 |   0.99±0.61 |  86.21±3.33 |  79.76±2.16
empirical_cumulative |  87.13±0.65 |  88.30±0.55 |   4.78±0.21 |   6.19±0.85 |  87.13±0.65 |  88.32±0.56
val_cal              |  88.16±0.42 |  89.32±0.27 |   8.26±3.22 |   9.38±0.40 |  88.21±0.45 |  89.38±0.27

best_fpr  (runs: GS 3 | GA 3)
Grid Search : 233±153 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 444±42 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.16±0.22 |  84.14±0.36 |  28.12±0.74 |  28.80±0.97 |  84.80±0.19 |  84.83±0.31
fixed_05             |  85.33±0.09 |  85.85±0.20 |  24.49±0.23 |  23.32±0.54 |  85.80±0.09 |  86.28±0.18
platt                |  86.98±0.39 |  87.95±0.11 |  18.05±0.85 |  16.30±0.25 |  87.22±0.37 |  88.15±0.11
beta                 |  87.85±0.17 |  88.90±0.19 |  14.21±1.20 |  11.79±0.22 |  88.00±0.15 |  89.00±0.19
empirical            |  83.44±2.83 |  80.03±2.30 |   2.89±1.39 |   1.23±0.46 |  83.49±2.76 |  80.19±2.19
empirical_cumulative |  87.57±0.27 |  87.86±0.84 |   5.61±0.46 |   5.45±0.97 |  87.58±0.27 |  87.87±0.85
val_cal              |  88.69±0.06 |  89.35±0.30 |   8.31±0.70 |   9.44±0.58 |  88.74±0.07 |  89.42±0.30

best_acc  (runs: GS 3 | GA 3)
Grid Search : 70±113 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 457±31 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.53±0.55 |  84.02±0.31 |  27.19±1.65 |  29.03±0.81 |  85.13±0.47 |  84.73±0.27
fixed_05             |  86.22±1.50 |  85.92±0.20 |  21.64±4.31 |  23.16±0.70 |  86.60±1.35 |  86.34±0.17
platt                |  87.70±0.19 |  87.91±0.15 |  16.03±0.83 |  16.33±0.44 |  87.89±0.17 |  88.11±0.14
beta                 |  86.42±1.50 |  89.01±0.12 |   7.39±5.21 |  11.39±0.23 |  86.46±1.56 |  89.11±0.12
empirical            |  86.11±3.29 |  79.57±2.28 |   9.50±6.66 |   0.99±0.61 |  86.21±3.33 |  79.76±2.16
empirical_cumulative |  87.13±0.65 |  88.30±0.55 |   4.78±0.21 |   6.19±0.85 |  87.13±0.65 |  88.32±0.56
val_cal              |  88.16±0.42 |  89.32±0.27 |   8.26±3.22 |   9.38±0.40 |  88.21±0.45 |  89.38±0.27

best_ce  (runs: GS 3 | GA 3)
Grid Search : 367±58 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 449±39 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.01±0.26 |  83.77±0.34 |  28.61±0.89 |  29.90±1.15 |  84.68±0.21 |  84.52±0.27
fixed_05             |  85.19±0.43 |  85.81±0.27 |  24.99±1.14 |  23.44±0.88 |  85.68±0.38 |  86.24±0.23
platt                |  87.05±0.34 |  87.95±0.13 |  18.06±0.90 |  16.26±0.40 |  87.29±0.32 |  88.15±0.12
beta                 |  88.09±0.36 |  89.01±0.14 |  13.07±0.97 |  11.44±0.28 |  88.21±0.34 |  89.11±0.14
empirical            |  78.93±1.93 |  79.54±2.31 |   0.85±0.36 |   1.01±0.61 |  79.15±1.82 |  79.73±2.19
empirical_cumulative |  87.15±0.50 |  88.20±0.84 |   4.64±0.29 |   5.90±1.42 |  87.15±0.51 |  88.22±0.85
val_cal              |  88.61±0.21 |  89.31±0.23 |   8.94±0.31 |   9.34±0.73 |  88.66±0.21 |  89.37±0.22

best_fitness  (runs: GS 3 | GA 3)
Grid Search : 367±153 neurons | 34.0±0.0 bits (mean per-neuron)
GA Neurons  : 445±39 neurons | 34.0±0.0 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  83.91±0.05 |  83.67±0.49 |  29.01±0.15 |  30.16±1.58 |  84.61±0.05 |  84.44±0.39
fixed_05             |  85.39±0.14 |  85.80±0.27 |  24.37±0.33 |  23.48±0.89 |  85.86±0.13 |  86.23±0.24
platt                |  87.13±0.43 |  87.95±0.14 |  17.82±0.75 |  16.27±0.43 |  87.36±0.42 |  88.15±0.13
beta                 |  88.09±0.27 |  89.00±0.13 |  13.16±0.56 |  11.53±0.25 |  88.21±0.26 |  89.10±0.13
empirical            |  81.00±2.95 |  79.36±2.04 |   1.74±1.27 |   0.95±0.48 |  81.12±2.84 |  79.56±1.94
empirical_cumulative |  87.46±0.23 |  88.21±0.85 |   5.38±0.17 |   5.88±1.40 |  87.46±0.23 |  88.23±0.86
val_cal              |  88.71±0.08 |  89.34±0.27 |   8.26±0.55 |   9.46±0.49 |  88.75±0.08 |  89.40±0.26

##### RULE-7 5-tables: IDSXD-unswt-quad-16b-Wb-CTRL (Grid vs GA, held-out TEST, mean±SD %) #####

best_f1  (runs: GS 3 | GA 3)
Grid Search : 300±173 neurons | 32.0±0.0 bits (mean per-neuron)
GA Neurons  : 319±157 neurons | 33.6±0.7 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.34±0.44 |  84.04±0.24 |  27.65±1.39 |  28.76±0.61 |  84.96±0.37 |  84.73±0.21
fixed_05             |  85.17±0.09 |  85.58±0.25 |  25.01±0.26 |  24.10±0.70 |  85.66±0.08 |  86.04±0.22
platt                |  87.10±0.04 |  87.98±0.19 |  18.14±0.08 |  16.18±0.33 |  87.34±0.04 |  88.18±0.18
beta                 |  87.87±0.05 |  88.87±0.14 |  14.01±0.49 |  11.75±0.25 |  88.01±0.05 |  88.98±0.14
empirical            |  80.72±0.53 |  81.32±1.75 |   1.71±0.16 |   1.68±0.76 |  80.83±0.50 |  81.42±1.68
empirical_cumulative |  87.62±0.83 |  87.26±0.65 |   6.36±0.99 |   4.22±0.32 |  87.64±0.84 |  87.26±0.65
val_cal              |  88.47±0.15 |  89.18±0.18 |   9.10±0.91 |   9.80±0.15 |  88.52±0.14 |  89.25±0.18

best_fpr  (runs: GS 3 | GA 3)
Grid Search : 367±58 neurons | 33.3±1.2 bits (mean per-neuron)
GA Neurons  : 316±157 neurons | 33.6±0.7 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  83.84±0.28 |  84.15±0.53 |  28.98±0.82 |  28.41±1.73 |  84.54±0.24 |  84.82±0.44
fixed_05             |  85.32±0.21 |  85.45±0.17 |  24.44±0.57 |  24.40±0.51 |  85.79±0.19 |  85.92±0.15
platt                |  87.01±0.22 |  87.88±0.21 |  18.13±0.35 |  16.34±0.56 |  87.25±0.21 |  88.09±0.20
beta                 |  87.95±0.23 |  88.68±0.03 |  13.48±0.56 |  12.04±0.30 |  88.08±0.22 |  88.79±0.03
empirical            |  78.48±1.90 |  80.92±1.29 |   0.95±0.54 |   1.55±0.66 |  78.73±1.79 |  81.03±1.24
empirical_cumulative |  87.31±0.16 |  87.75±0.58 |   5.32±0.20 |   5.04±0.91 |  87.32±0.16 |  87.76±0.59
val_cal              |  88.65±0.23 |  89.16±0.13 |   8.87±0.52 |   9.05±0.18 |  88.70±0.23 |  89.22±0.13

best_acc  (runs: GS 3 | GA 3)
Grid Search : 300±173 neurons | 32.0±0.0 bits (mean per-neuron)
GA Neurons  : 319±157 neurons | 33.6±0.7 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.34±0.44 |  84.04±0.24 |  27.65±1.39 |  28.76±0.61 |  84.96±0.37 |  84.73±0.21
fixed_05             |  85.17±0.09 |  85.58±0.25 |  25.01±0.26 |  24.10±0.70 |  85.66±0.08 |  86.04±0.22
platt                |  87.10±0.04 |  87.98±0.19 |  18.14±0.08 |  16.18±0.33 |  87.34±0.04 |  88.18±0.18
beta                 |  87.87±0.05 |  88.87±0.14 |  14.01±0.49 |  11.75±0.25 |  88.01±0.05 |  88.98±0.14
empirical            |  80.72±0.53 |  81.32±1.75 |   1.71±0.16 |   1.68±0.76 |  80.83±0.50 |  81.42±1.68
empirical_cumulative |  87.62±0.83 |  87.26±0.65 |   6.36±0.99 |   4.22±0.32 |  87.64±0.84 |  87.26±0.65
val_cal              |  88.47±0.15 |  89.18±0.18 |   9.10±0.91 |   9.80±0.15 |  88.52±0.14 |  89.25±0.18

best_ce  (runs: GS 3 | GA 3)
Grid Search : 500±0 neurons | 32.7±1.2 bits (mean per-neuron)
GA Neurons  : 320±155 neurons | 33.6±0.7 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.25±0.30 |  84.07±0.13 |  27.94±0.98 |  28.72±0.38 |  84.89±0.25 |  84.75±0.11
fixed_05             |  85.28±0.18 |  85.53±0.20 |  24.81±0.52 |  24.22±0.61 |  85.77±0.16 |  85.99±0.17
platt                |  87.14±0.16 |  87.94±0.20 |  17.87±0.39 |  16.27±0.45 |  87.37±0.15 |  88.14±0.20
beta                 |  88.08±0.19 |  88.82±0.10 |  13.15±0.36 |  11.83±0.14 |  88.21±0.18 |  88.92±0.10
empirical            |  79.39±0.73 |  81.09±1.84 |   1.15±0.38 |   1.62±0.84 |  79.57±0.69 |  81.20±1.77
empirical_cumulative |  88.00±0.73 |  87.50±0.36 |   6.35±1.34 |   4.47±0.14 |  88.02±0.74 |  87.50±0.36
val_cal              |  88.64±0.11 |  89.16±0.15 |   8.91±0.72 |   9.48±0.37 |  88.70±0.11 |  89.23±0.16

best_fitness  (runs: GS 3 | GA 3)
Grid Search : 367±115 neurons | 33.3±1.2 bits (mean per-neuron)
GA Neurons  : 318±158 neurons | 33.6±0.7 bits (mean per-neuron)
mode                 | F1 Grid      | F1 GA        | FPR Grid     | FPR GA       | Acc Grid     | Acc GA
---------------------+--------------+--------------+--------------+--------------+--------------+-------------
train_cal            |  84.17±0.13 |  84.15±0.49 |  28.27±0.32 |  28.39±1.60 |  84.82±0.11 |  84.82±0.40
fixed_05             |  85.38±0.37 |  85.49±0.17 |  24.37±1.09 |  24.28±0.51 |  85.85±0.33 |  85.95±0.14
platt                |  87.26±0.25 |  87.87±0.19 |  17.54±0.38 |  16.37±0.52 |  87.49±0.25 |  88.07±0.18
beta                 |  88.21±0.17 |  88.69±0.04 |  12.76±0.25 |  12.02±0.24 |  88.33±0.17 |  88.80±0.04
empirical            |  80.59±0.95 |  80.92±1.29 |   1.52±0.47 |   1.54±0.64 |  80.71±0.91 |  81.03±1.24
empirical_cumulative |  87.75±0.69 |  87.49±0.39 |   5.97±1.58 |   4.71±0.28 |  87.76±0.71 |  87.50±0.40
val_cal              |  88.62±0.08 |  89.15±0.14 |   9.89±1.21 |   9.14±0.45 |  88.68±0.08 |  89.21±0.14
```

### Reproduce
`python scripts/ids8/ids8_final_readout.py audit | armtable <gt> | stats <gt> | seeds <arm> <gt> | pareto [fmin] | rule7 <arm> | refs`
(read-only `?mode=ro`). Core query: `validation_summaries` JOIN `flows`/`experiments` WHERE `f.id BETWEEN 6237 AND 6263`
AND `validation_point='final'`; GA = `phase_type='ga_neurons'`. Metrics = `json(threshold_metadata)[mode].{f1,fpr,acc}` x 100,
mean±sample SD over seeds 20403-20405. Genome shape is from `genomes.tiers_json` for the validated hash.

## 0B. Best individual genome (CEILING, not the claim)

Mined across every genome_type × all 7 threshold modes. **Best-of-N inflates** — a maximum
over 29 seeds × 5 genome types × 7 modes is not an estimate of expected performance. Quote
these only as "the best architecture we found", never as the cohort result.

```
dataset                             | WNN best F1/FPR/Acc      | baseline (raw)        | verdict
------------------------------------+--------------------------+-----------------------+------------------
UNSW-NB15 temporal                  | 90.20 /  5.30 / 90.22    | RF  85.18/27.73       | +5.02 F1, -22.43 FPR  STRICT WIN
UNSW-NB15 random                    | 94.56 /  0.61 / 99.17    | RF  96.06/ 0.30       | -1.50 F1, +0.31 FPR   loss
CICIDS2017 random                   | 99.64 /  0.08 / 99.77    | RF  99.74/ 0.08       | -0.10 F1, tie FPR     ~tie
CIC-IoT-2023 neto-sub random        | 93.35 /  7.50 / 96.69    | RF  94.42/ 9.81/97.32 | -1.07 F1, -2.31 FPR   FPR WIN, F1 loss
    (re-mined 31/08 at n=29: UNCHANGED) |                        | XGB 93.38/11.59/96.82 | -0.03 F1, -4.09 FPR   FPR WIN
  sub-5% FPR point                  | 93.08 /  4.91 / 96.46    |                       |
  sub-4% FPR point                  | 90.35 /  2.04 / 94.72    |                       |
```

### 0B-IDSXD. Best individual genome per dataset from the IDSXD sweep vs RF raw (13/09/2026)

Each WNN row = the MAX over ~1,600 non-independent (seed x genome x threshold) points (completed
flows x 2 phases x 5 genome_types x 7 modes); RF = ONE deterministic fit (rs=42, raw top-20, TEST).
Selection inflates the WNN side, so the true gap to RF is at least what is shown. Script:
`scripts/idsxd_best_column.py` (second block). Mean±SD columns (§0A) are what a table may carry.

    UNSW-NB15 random_3way (qsr 64b; max over 1610)                        F1     FPR    Acc
    WNN best-F1  B15-CE qsr GA best_fitness val_cal   flow 5925 s20403   94.53   0.60  99.17  (= best-Acc genome)
    WNN best-FPR B05-CE qsr GS best_ce emp_cum        flow 5913 s20403   93.95   0.37  99.13  @val_cal 94.14/0.78/99.08
    WNN best-Acc B15-CE qsr GA best_fitness platt     flow 5925 s20403   94.49   0.46  99.19  @val_cal 94.53/0.60/99.17
    RF raw fixed_05 / val_cal(0.47)                                      95.80/95.82  0.31/0.34  99.39/99.38

    CICIDS2017 random_3way (quad 96b; max over 1680; era-immune)          F1     FPR    Acc
    WNN best-F1  CE20 quad GA best_f1 train_cal       flow 5880 s20403   99.56   0.11  99.72  (= @val_cal; = best-Acc genome)
    WNN best-FPR B05-CE quad GS best_fpr empirical    flow 5866 s20404   97.97   0.04  98.74  @val_cal 99.26/0.20/99.53
    RF raw fixed_05 / val_cal(0.32)                                      99.83/99.85  0.07/0.10  99.89/99.90

    CIC-IoT subsample random_3way (quad 96b; max over 1680; IDSXD2 0/12)  F1     FPR    Acc
    WNN best-F1  B05-AC quad GA best_f1 train_cal     flow 5888 s20405   93.18   6.91  96.57  PRE-FIX wheel — do not cite
      era-clean  B15-AC quad GA best_ce train_cal     flow 5900 s20405   93.06   7.56  96.53  @val_cal 93.04/7.08/96.50
    WNN best-FPR CE20 quad GA best_acc emp_cum        flow 5905 s20404   91.30   2.71  95.33  @val_cal 92.89/7.29/96.42
    WNN best-Acc B05-AC quad GA best_f1 val_cal       flow 5888 s20405   93.14   7.59  96.57  PRE-FIX
    RF raw fixed_05 = val_cal(0.50)                                      95.53   7.38  97.84

Reading: the F1/Acc ceilings are all GA-neurons genomes (never the grid) — CE-weighted arms on
UNSW/CICIDS, AC-weighted on CIC-IoT — and best-F1/best-Acc share one genome hash per dataset. The
FPR ceilings are empirical/empirical_cumulative threshold points that do NOT survive at val_cal
(0.37→0.78, 0.04→0.20, 2.71→7.29): threshold-mode artifacts, not operating points. One RF raw fit
beats the WNN ceiling on F1 and Acc on all three datasets (+1.3 / +0.27 / +2.35 pp F1) and ties or
beats it on FPR wherever the WNN point is a real val_cal one.

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
