# =====================================================================
# SECTION 9 — MCSD-auto MULTICLASS COHORT (Protocol v2, _3way)
# =====================================================================

Appended 27/08/2026 from `file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro`
(read-only). Producer: `ids-security` agent readout of flows **5980-5994**, cross-checked
against `docs/multiclass_baselines/unsw-nb15_temporal_3way.json` and `/tmp/wnn_worker.log`.

**Cohort:** `MCSD-unswt-quad-16b-{binary,hier,multi}-s{20401..20405}-auto`, 15/15 complete.
UNSW-NB15 `temporal_3way`, top20, QUAD, `ids_n_bits=16`, production Wb weights
(w_f1=.35 w_fpr=.35 w_acc=.20 w_ce=.10, desirability), pop=50, gens<=250, patience=5,
k_folds=5. Train 175,341 / TEST 41,166 (report-only) — the same partition the RF/XGB
baseline file was fit and scored on.

**CE anchor: AUTOMATIC per task** — binary 0.133298, hier-S1 9-class 0.348766, flat-multi
10-class 0.370682 (= 0.2128 x that task's own train-label base-rate entropy). No run here
used the retired TEST-fitted 0.1937. **Older MCS/MCSD generations are NOT comparable and
must NOT be merged into this section.**

**Pre-registered read:** macro-F1 + benign-FPR are the PRIMARIES. weighted-F1 is SECONDARY
/ descriptive and must never be promoted to the headline.

## 9A. Arm-level table — held-out TEST, mean±SD over 5 seeds

⚠️ Read the granularity column before the numbers. The three arms do not emit the same
prediction, so their macro-F1 columns are NOT on one scale.

```
arm / decode mode                   granularity | macroF1(PRIM) | benignFPR(PRIM) | weightedF1(2nd) |  accuracy  |  n | search budget
------------------------------------------------+---------------+-----------------+-----------------+------------+----+---------------------------
BINARY  val_cal         (Protocol-v2)  2-class  |  89.07±0.04   |   9.34±0.17     |  89.16±0.04     | 89.13±0.04 |  5 | 2 phases (grid+GA), 1 model
BINARY  beta                           2-class  |  88.61±0.24   |  12.27±0.84     |  88.72±0.23     | 88.72±0.22 |  5 |
BINARY  empirical_cumulative           2-class  |  87.25±0.35   |   4.80±0.42     |  87.28±0.35     | 87.26±0.35 |  5 |
BINARY  platt                          2-class  |  87.63±0.21   |  16.82±0.43     |  87.79±0.21     | 87.84±0.20 |  5 |
BINARY  fixed_05                       2-class  |  85.71±0.40   |  23.64±1.20     |  85.97±0.38     | 86.15±0.35 |  5 |
BINARY  train_cal                      2-class  |  84.29±0.06   |  28.17±0.24     |  84.61±0.05     | 84.94±0.05 |  5 |
BINARY  empirical                      2-class  |  81.29±2.85   |   1.54±0.82     |  81.15±2.93     | 81.41±2.77 |  5 |
------------------------------------------------+---------------+-----------------+-----------------+------------+----+---------------------------
HIER    cascade  (ONLY operating pt)  10-class  |  37.26±1.75   |  13.54±1.07     |  74.37±0.95     | 71.42±1.39 |  5 | 4 phases (2 grids + 2 GAs),
                                                |               |                 |                 |            |    | 2 models = 2x TOTAL budget
------------------------------------------------+---------------+-----------------+-----------------+------------+----+---------------------------
MULTI   margin_val_cal  (Protocol-v2) 10-class  |  39.34±0.82   |  41.22±2.05     |  69.93±0.84     | 64.05±0.94 |  5 | 2 phases (grid+GA), 1 model
MULTI   argmax           (GA fitness) 10-class  |  39.30±0.80   |  41.20±2.04     |  69.87±0.86     | 64.04±0.95 |  5 |
MULTI   margin_train_cal              10-class  |  39.29±0.80   |  41.27±2.08     |  69.92±0.84     | 64.03±0.96 |  5 |
MULTI   margin_fixed0                 10-class  |  39.30±0.80   |  41.20±2.04     |  69.87±0.86     | 64.04±0.95 |  5 |
MULTI   argmax_platt                  10-class  |  36.19±1.38   |  13.95±3.01     |  68.80±0.21     | 71.77±0.71 |  5 |
MULTI   argmax_beta                   10-class  |  27.23±12.30  |  20.94±21.05    |  57.20±13.82    | 61.63±7.50 |  5 | UNSTABLE — report or drop deliberately
```

Genome selection: `best_f1` at the final GA checkpoint for BINARY/MULTI. The HIER cascade row
is NOT metric-selectable (the pipeline freezes one S0 genome and pairs it with the S1
population best). Binary `f1` was verified by hand-recomputation to be macro-F1 over 2 classes
(seed 20401 val_cal: 0.89909 attack-F1, 0.88240 benign-F1 -> 0.89075 = stored value).

### Against the bar

```
model                        macroF1  benignFPR  weightedF1  accuracy   source
--------------------------------------------------------------------------------------------
RF  multi                     51.45     22.58      78.35      76.49    multiclass_baselines/*.json
XGB multi                     52.25     23.10      78.23      77.58    (same)
RF  cascade                   51.73     24.95      77.71      75.56    (same)
XGB cascade                   50.78     25.44      77.73      76.63    (same)
--------------------------------------------------------------------------------------------
WNN hier cascade   (n=5)      37.26     13.54      74.37      71.42    THIS COHORT
WNN flat multi val_cal (n=5)  39.34     41.22      69.93      64.05    THIS COHORT
WNN flat multi platt   (n=5)  36.19     13.95      68.80      71.77    THIS COHORT
--------------------------------------------------------------------------------------------
banked WNN flat-multi (n=1)   40.51     41.91        -          -      prior investigation, DESCRIPTIVE ONLY
banked WNN cascade  flow 5975 35.02     13.85        -        70.92    TEST-anchor era, NOT merged
```

**Both WNN 10-class configurations sit 12-15 pp of macro-F1 BELOW the RF/XGB bar**, below all
four tree comparators including the tree cascades. They win benign-FPR by 9-11 pp. The trade
is ~14 pp macro-F1 for ~11 pp FPR and must be stated in that order.

## 9B. MANDATORY per-class recall — 10 classes x 3 arms, held-out TEST, mean±SD, n=5

```
class            support |   BINARY*   |  HIER casc  | MULTI valcal | MULTI platt |  RF multi | XGB multi | RF casc | XGB casc
-------------------------+-------------+-------------+--------------+-------------+-----------+-----------+---------+---------
Normal             18500 |  90.7± 0.2  |  86.5± 1.1  |   58.8± 2.0  |  86.1± 3.0  |    77.4   |    76.9   |   75.1  |   74.6
Analysis             351 |  89.6± 0.7  |   3.7± 2.6  |    6.5± 3.1  |   0.0± 0.0  |     0.9   |     4.3   |    2.8  |    2.6
Backdoor             283 |  94.6± 0.8  |   8.3± 7.7  |   14.6± 6.3  |   0.0± 0.0  |     7.4   |    14.5   |    8.8  |   11.3
DoS                 2061 |  93.3± 0.1  |   2.8± 1.4  |   20.9± 5.1  |   0.6± 0.2  |    17.1   |    11.5   |   16.6  |   14.2
Exploits            5545 |  92.5± 0.3  |  39.8± 4.4  |   43.9± 3.0  |  44.1± 2.1  |    79.7   |    90.0   |   80.0  |   88.6
Fuzzers             3000 |  40.2± 0.4  |  27.7± 3.0  |   71.5± 8.5  |  30.2± 9.2  |    56.0   |    56.5   |   57.1  |   59.4
Generic             9438 |  98.0± 0.1  |  95.0± 0.6  |   94.7± 0.5  |  94.6± 0.8  |    97.0   |    97.1   |   97.1  |   97.1
Reconnaissance      1766 |  92.6± 1.1  |  66.6±13.9  |   76.1± 2.0  |  72.3± 5.2  |    79.4   |    80.8   |   79.6  |   80.3
Shellcode            199 |  87.8± 0.4  |  53.8± 6.2  |   62.4± 6.6  |  24.6± 7.3  |    59.8   |    64.3   |   59.8  |   66.8
Worms                 23 |  93.0± 2.4  |  87.0± 3.1  |   68.7±29.1  |   1.7± 2.4  |    43.5   |    43.5   |   47.8  |   30.4
-------------------------+-------------+-------------+--------------+-------------+-----------+-----------+---------+---------
mean attack recall (9)   |    86.9     |    42.7     |     51.0     |    29.8     |    49.0   |    51.4   |   50.0  |   50.1
```

**\* The BINARY column is a DIFFERENT QUANTITY.** The binary arm never emits a 10-class label.
Its entry is the *attack-detection rate* (fraction of that true class flagged as attack), a
strict UPPER BOUND on any 10-class recall; `Normal` there is `1 - FPR`. Never plot it beside
the others without this caveat.

**What the table exposes (the QSR lesson applied):**

- MULTI's +2.08 pp macro-F1 lead over HIER at the pre-registered `val_cal` decode is **not
  better detection**. MULTI wins 7/9 attack classes on recall by collapsing `Normal` recall
  86.5 -> 58.8, i.e. a 41.2% benign-FPR. It buys attack recall with a 3x worse FPR, and
  macro-F1 barely registers the cost because 9 of 10 classes are attacks.
- At the FPR-matched decode (`argmax_platt`, 13.95% vs hier 13.54%), MULTI **loses** recall on
  6/9 attack classes and drops Analysis / Backdoor / DoS / Worms to ~zero (0.0 / 0.0 / 0.6 / 1.7).
- **Rare-class collapse is the DATASET, not the substrate** — RF is 0.9 / 7.4 / 17.1 and XGB
  4.3 / 14.5 / 11.5 on Analysis / Backdoor / DoS. The real WNN deficit is `Exploits`
  (39.8-44.1 vs 79.7-90.0) plus Fuzzers / Recon — large, well-supported classes. That is where
  the 14 pp macro-F1 gap lives. Do not blame the tail.

## 9C. Gate-FPR identity — structurally 5/5, persisted rows only 2/5

**(a) Structural identity HOLDS 5/5, exactly.** A benign row is a false positive of the
10-class cascade iff the S0 gate routes it to S1 (S1 emits only attack labels). Recomputing
the gate from each stored cascade confusion matrix reproduces `benign_fpr` to full double
precision and the derived routed count matches `routed_to_s1` exactly:

```
flow   derived gate acc   derived gate FPR   cascade benignFPR   derived routed   stored routed_to_s1
5985      0.893747           0.141027            0.141027           23510              23510   OK
5986      0.891537           0.145622            0.145622           23589              23589   OK
5987      0.899334           0.117730            0.117730           22878              22878   OK
5988      0.898849           0.134054            0.134054           23462              23462   OK
5989      0.897075           0.138432            0.138432           23551              23551   OK
```

**(b) Identity against the persisted `best_*` S0 rows HOLDS ONLY 2/5.**

```
flow   cascade FPR   matching S0 best_* row              S0 best_* FPRs on file
5985     0.141027    best_f1 / best_acc      EXACT      ce .141946  acc/f1 .141027  fpr/fit .140811
5986     0.145622    NONE                    MISS       ce .146432  acc/f1 .143946  fpr/fit .125135
5987     0.117730    best_fitness            EXACT      ce/acc/f1 .122541  fpr .097081  fit .117730
5988     0.134054    NONE                    MISS       ce .138649  acc/f1 .139081  fpr .129351  fit .133351
5989     0.138432    NONE                    MISS       ce .113946  acc/f1 .130324  fpr .106649  fit .112811
```

This is **`best_genome != pop[0]` again**. The cascade uses the *frozen* S0 genome carried
across the stage boundary, which for 3/5 seeds is none of the five persisted `best_*` genomes.
**`validation_summaries` contains no row describing the gate that actually produced the
published cascade for seeds 20402 / 20404 / 20405.** Reconstructing "the S0 gate row" from
`best_f1` or `best_fpr` would publish a gate that was never deployed.
**FIX BEFORE ANY TABLE SHIPS: persist the frozen S0 genome as its own `genome_type`.**

**Protocol-v2 gap on the gate:** `threshold_metadata` is NULL on all 50 S0 rows (5/5 seeds).
The decode sweep in `src/wnn/ram/experiments/experiment.py` (~L1251-1310) fires only when the
evaluator is `_single_cluster` (7 binary modes) or `_classification == 'multi'` (6 multiclass
modes). The hier S0 gate is a 2-cluster argmax gate (`ids_single_cluster=False`) and is
neither, so it gets no sweep and no calibration. **The hierarchical arm therefore has exactly
ONE operating point** — no val-calibrated threshold, no Pareto mining — versus 7 (binary) and
6 (multi).

## 9D. The hier duration split — genome size, NOT early stopping

All five hier seeds completed all four stage-tagged phases and emitted a real cascade, with
comparable generation counts. No seed early-stopped prematurely.

```
flow  seed    S0:Grid  S0:GA          S1:Grid  S1:GA          total   frozen S0 gate
                       iters  s/gen            iters  s/gen   (min)
5985  20401     1      140    12.068     1     110    0.966   30.60   2 clusters, 608 neurons, 34 bits
5986  20402     1      120     0.476     1     100    0.396    2.14   2 clusters,  12 neurons, 34 bits
5987  20403     1      100     0.444     1     120    0.388    1.98   2 clusters,  22 neurons, 34 bits
5988  20404     1      110     0.378     1     110    0.393    1.80   2 clusters,  26 neurons, 34 bits
5989  20405     1      110     0.327     1     110    0.927    2.82   2 clusters,  12 neurons, 34 bits
```

The cause is decided at the GRID, not the GA: seed 20401's S0 grid winner was a 600-neuron
gate (grown to 608), the other four were 10-neuron gates (grown to 12-26). ~50x the size
costs ~30x per generation (12.07 s/gen vs 0.33-0.48). **n=5 stands for the hier arm.**

Two results worth carrying:

1. **Gate capacity saturates.** The 608-neuron gate is the WORST of the five on held-out
   accuracy (0.8937) and among the worst on FPR (0.1410); the 12- and 22-neuron gates reach
   0.8971 and 0.8993. A 50x larger gate bought nothing.
2. Report hier runtime as **1.8-30.6 min, median 2.14** — never as a bare mean of 7.9.

## 9E. Budget asymmetry — disclose in the table

HIER runs **4 phases across 2 models** vs **2 phases / 1 model** for both flat arms. Per-stage
budget is matched; TOTAL budget is ~2x. Measured GA generations:

```
arm     GA generations per run (5 seeds)              mean
binary  100, 120,  80,  70,  60                        86
multi    90, 100,  90,  90, 100                        94
hier    250, 220, 220, 220, 220  (S0+S1 combined)     226   <- ~2.4x
```

Partly offset: the S1 model trains only on attack rows (~119k of 175,341; the S1 grid line
shows 71,606 for its fixed 3-of-5 subset) and S1 generations are 5-30x cheaper in wall-clock.
On generation count the hier arm is **not** budget-matched.

## 9F. Statistical read — paired by seed, n=5, exact sign-flip permutation

```
HIER cascade - MULTI margin_val_cal   (pre-registered decode)
  macroF1   pp: -5.68 -3.31 +0.13 -1.42 -0.10    mean  -2.08  SD 2.44  p = 0.1875   NOT significant
  benignFPR pp: -28.03 -27.49 -30.58 -28.61 -23.72  mean -27.69  SD 2.50  p = 0.0625, 5/5 same sign

HIER cascade - MULTI argmax_platt     (FPR-matched decode)
  macroF1   pp: -0.71 -1.23 +2.05 +1.48 +3.78    mean  +1.07  SD 2.05  p = 0.3125   NOT significant
  benignFPR pp: -2.71 -0.95 -3.21 -0.01 +4.81    mean  -0.41  SD 3.19  p = 0.8125   NOT significant
```

p=0.0625 is the SMALLEST attainable p for a two-sided n=5 sign-flip test — "as significant as
n=5 permits", no more. n=5 at SD ~2 pp cannot resolve a 1-2 pp macro-F1 effect.

## 9G. Per-class genome shape — the GA differentiates, but not by class support

Read from `genomes.tiers_json` (final GA genome per flow). Cluster order is assumed to follow
the class order of section 9B; **that mapping is NOT independently verified** — treat the
class attribution as provisional, the neuron counts themselves are exact.

```
MULTI  s20402   10 clusters, 1419 total neurons
  per-cluster neurons: 96, 101, 104, 194, 104, 203, 98, 205, 205, 109      (range 96-205, ~2.1x)

HIER-S1 s20401   9 clusters, 48 total neurons    per-cluster: 6,5,6,5,5,5,6,5,5   (bits 24)
HIER-S1 s20402   9 clusters, 45 total neurons    per-cluster: 5,5,5,5,5,5,5,5,5   (bits 28-34)
HIER-S1 s20403   9 clusters, 48 total neurons    per-cluster: 5,6,5,5,5,6,5,5,6   (bits 32-34)
HIER-S1 s20404   9 clusters, 48 total neurons    per-cluster: 5,5,5,5,5,6,6,6,5   (bits 32)
HIER-S1 s20405   9 clusters, 46 total neurons    per-cluster: 5,5,5,6,5,5,5,5,5   (bits 32-34)
```

Two findings:

1. **The mechanism exists and fires.** `ClusterGenome` carries per-cluster `neurons_per_cluster`
   and per-neuron `bits_per_neuron`, and `_mutate_neurons` mutates each cluster INDEPENDENTLY
   (`for c in range(len(new_neurons))`, own `rng.random() < mutation_rate` draw). The MULTI arm
   really did diverge, 96 -> 205 neurons across classes.
2. **But it is not steered by class support, and the HIER S1 collapsed to the floor.** Every
   S1 cluster sits at 5-6 neurons — `config.min_neurons` is 5 — so the 9-class attack model is
   45-48 neurons TOTAL. Both arms are seeded by `ClusterGenome.create_uniform` (identical
   neurons and bits for every class) and bounded by GLOBAL `min_neurons`/`max_neurons`
   (5..500) and `min_bits`/`max_bits` (4..34); there is no per-class size prior.

**Open hypothesis (Luiz, 27/08) — support-tiered per-class sizing.** Class support here spans
23 (Worms) to 18,500 (Normal), ~800x, while the seeded architecture is uniform. A 250n x 34b
cluster for a 23-row class is absurdly sparse; a 5n x 10b cluster may be near-ideal for it and
hopeless for Normal. The `tier_config` mechanism already exists (`flow.py` L2840-2860,
`(count, neurons, bits)` triples, `experiments.tier_config` column, LM-era) and is NOT plumbed
to the IDS path. Proposal: size each class's cluster by support band
(<100 / <1k / <10k / <100k / >=100k rows). **UNTESTED — no run has varied this.**

## 9H. What this cohort DOES and DOES NOT support

**Supported:**

1. **The BINARY WNN detector matches the tree bar on F1 and crushes it on FPR** —
   89.07±0.04 F1 / 9.34±0.17 FPR vs RF-binary 89.18 / 24.95 and XGB-binary 89.07 / 25.44:
   a tie on F1 (-0.11 pp vs RF, +0.00 vs XGB) with **-15.6 / -16.1 pp FPR**, same partition,
   SD 0.04 over 5 seeds. This is the cohort's strongest result, and it is the BINARY arm.
2. Neither WNN 10-class arm reaches the RF/XGB multiclass bar (37.26 / 39.34 vs 50.78-52.25).
3. At matched benign-FPR (~13.5-14%), hierarchical and flat-multi are statistically
   indistinguishable on macro-F1 (+1.07 pp, p=0.31). The cascade is **not-worse** at ~1/6 the
   median wall-clock (2.14 min vs 49.0 min).
4. The gate-FPR identity is structurally exact, 5/5 — extending 8/8 on the comparators.
5. Rare-class collapse is a dataset property; RF and XGB fail on the same three classes.

**NOT supported — do not write these:**

1. "The hierarchical cascade beats flat multiclass." It LOSES macro-F1 by 2.08 pp at the
   pre-registered decode; the +1.07 pp at matched FPR is p=0.31.
2. "WNN detects attacks better." The recall table refutes it at every operating point.
3. Any macro-F1 comparison putting BINARY's 89.07 beside HIER's 37.26 — different label space.
4. Any claim resting on MULTI `argmax_platt` without disclosing it was selected BECAUSE it
   matched hier's FPR. That is best-of-6-modes mining.
5. Any claim about hier threshold behaviour or a hier Pareto front. It has ONE point.

**Reviewer-attackable, ranked:**

1. The published hier cascade's gate is absent from the DB for 3/5 seeds (9C).
2. **`ids_single_cluster` co-varies with the arm** (`True` binary, `False` hier+multi), so
   "the ONLY variable is `ids_classification`" is **false as stated**. Arguably entailed, but
   the hier S0 gate is a 2-cluster argmax model while the binary arm is 1-cluster thresholded
   — "hier gate vs flat binary" is not a clean matched comparison.
3. Unequal mining surface: 7 modes (binary) / 6 (multi) / 1 (hier) structurally favours flat.
4. The three arms optimize three different CE anchors (0.133 / 0.349 / 0.371). Correct by
   design (per-task base-rate normalization) but the arms are not searching one objective.
5. Protocol-v2 coverage incomplete on hier S1 — 5 decode modes, missing `margin_val_cal`.
6. n=5 at SD ~2 pp cannot resolve the effects discussed.
7. `argmax_beta` is unstable (27.23±12.30) — report or drop deliberately, never silently.

**Numbers NOT found (not estimated):** the S0 gate's decision threshold for any hier seed
(`threshold_metadata` NULL 5/5; the log's `threshold N%` lines are the GA VIABILITY threshold,
not a decision threshold). Per-class metrics for `rf_binary` / `xgb_binary` are absent from
the baseline file, so the binary detection-rate column has no tree comparator.
