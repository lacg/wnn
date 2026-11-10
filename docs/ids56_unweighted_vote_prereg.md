# IDS-56 U: pre-registration of the UNWEIGHTED min-share vote on unswr

Written 10/11/2026 (ids-security agent), on Luiz's approval of "recommendation 2, n=3 first" from
`docs/ids56_unswr_tiny_genomes.md` (commit 52b8246c). It is the basis for this doc.

**Status: REVIEWED (experiment-design, 10/11/2026): GO-WITH-CHANGES, changes applied below (§3, script). Nothing queued.**
- No flow was queued, paused, cancelled, resumed or reordered. The worker (ABI 28) was live throughout.
- Probes ran at `nice -n 10` through the worker's own construction path (`IdsEvaluatorFactory` → `IDSCacheWrapper`,
  `set_write_rule`, Rust `evaluate_multiclass_at_thresholds`). There is no Python reimplementation.
- The scratch probe is `/tmp/ids56r/probe5.py` and is not committed.
- Queue script: `scripts/queue_ids56_unweighted_vote.py`. Run it only with Luiz's go.

## 1. Class-weight audit: is `balance_classes=false` a clean, vote-only switch?

**Verdict: CLEAN. No code change and no ABI bump are needed.**
- `balance_classes` becomes one per-class `u32` weight vector.
- The only consumer of that vector is the per-visit training write.
- Lattice and set writers ignore it on both CPU and GPU. Only the vote tally (and the QUAD nudge, not used here) reads it.
- It does not touch folds, negatives, fitness, CE, threshold calibration, sampling or grid scoring.

Against the lattice control, the arm therefore differs in exactly one component: the writer. The lattice writer is
replaced by the unweighted min-share tally.

| # | Site (file:line) | What it does with the flag/weights | Affects the arm? |
|---|---|---|---|
| 1 | `src/wnn/ram/experiments/IdsEvaluatorSettings.py:94` | reads `balance_classes` from flow params | plumbing |
| 2 | `src/wnn/ram/experiments/IdsEvaluatorFactory.py:129,140` | log lines only | no |
| 3 | `src/wnn/ram/experiments/IdsEvaluatorFactory.py:162,232` | forwards to `IDSEvaluator` (flat + hierarchical) | plumbing |
| 4 | `src/wnn/ram/architecture/ids_evaluator.py:320-327,1080` | streaming-mode weights (`_compute_class_weights`) | no: unswr is in-memory, not streaming |
| 5 | `src/wnn/ram/experiments/validation_cache_key.py:164` | `("bal", …)` in the validation cache key | bookkeeping only (expected: no cache reuse with lattice flows) |
| 6 | `src/wnn/ram/strategies/accelerator/pyapi/ids_cache_api.rs:78-123,148-222` | PyO3 ctor arg | plumbing |
| 7 | `ids_cache.rs:468-479` | `class_weights = compute_class_weights_with_multiplier(train_labels)` if set, else `None` | **source** |
| 8 | `adaptive/eval_export.rs:349-378` | `max_count / count × multiplier` (unswr: benign 1, attack 25) | source |
| 9 | `ids_cache.rs:413`, `ids_cache.rs:614` | stratified folds and negatives: labels + seed only, **no weights** | no |
| 10 | `ids_cache.rs:896, 951, 1040, 1078, 1115, 1181, 1243, 1318, 1367` | every eval/score entry point passes `class_weights` → it reaches only the train calls (row 11) | via row 11 only |
| 11 | `adaptive/eval_hybrid.rs:414, 459, 513, 652, 1138, 1230, 1346`; `adaptive/eval_single.rs:562, 640, 693, 805, 858, 953, 1004, 1169` | the arg is consumed ONLY by `train_genome_in_slot` / `train_single_via_marker` / `marker_train::batched_train_offspring` / `train_single_genome_chunked_scored_probe`, which are all trainers | via row 12 only |
| 12 | `adaptive/eval_export.rs:602` (own-cluster visit), `:711` (negative visits) | `repeats = class_weights[label]` | yes, see the next three rows |
| 12a | `adaptive/eval_export.rs:630, 725` | **lattice** write: `memory.write(…, TRUE/FALSE)`, `repeats` unused | **no** (lattice is weight-invariant) |
| 12b | `adaptive/eval_export.rs:518, 618, 714` → `core/vote_share.rs:84` → `core/neuron_memory.rs:1457` (`oi_apply_nudge`) | **vote** tally: `±repeats × own/other` | **yes: this is the treatment** |
| 13 | `shaders/marker_train.metal:275-330` | GPU twin: `weight = class_weights[target]`; SET/LATTICE "weights moot"; VOTE `±weight·own/other` | same as 12a/12b |
| 14 | `ids_streaming.rs:113, 306` | streaming trainer weights | no (not streaming) |
| 15 | fitness / CE / thresholds (`core/fitness.rs`, `multiclass_metrics.rs`, `base_rate_entropy.rs`, `ids_evaluator.py:716-721` calib strata) | no weight input. Calib strata keys on `undersample_majority`, not `balance_classes` | no |

**Measured corroboration:** lattice is bit-identical with and without `balance_classes` on all 5 genomes probed
(§2, rows `lattice_w` = `lattice_u`, VAL and TEST).

**Caveat (attribution, not confounding):** the treatment is a package, "vote θ=0.02 + unweighted tally", measured
against lattice. The weighted ms01 flows (7503/7507/7511) are the third point. They separate "vote" from "weighting"
only loosely, because θ differs (0.1 vs 0.02).

## 2. Offline probe on more seeds

What is available in the DB at 10/11 21:30 UTC:
- r20419: 7501 is complete.
- r20420: 7505 has finished its grid. Its GA is running (gen 62), and GA elites have no stored connections.
- r20421: 7509 is still queued.

So **r20421 has no genome yet**, and r20420 has only its three grid finals.

Probe setup:
- Columns are VAL F1 / FPR / TPR in %. τ is fit on VAL and scored on VAL; this is the choosing read.
- Δ is relative to lattice on the same genome.
- TEST (`margin_val_cal`) is a check only.
- `w` = weighted tally (as flown); `u` = `balance_classes=false`.
- **θ=0.005 is not representable.** `VoteShare::from_theta` and `validate_vote_share` refuse values off the 1/100
  grid. Supporting it would need a ram_core change and a worker swap, so it was not pursued.

```
r20419  7501 GA best_f1 (492n, [250,242]) — the only GA (point-B) genome
rule      | VAL F1 / FPR / TPR      | dF1   dFPR  | TEST F1 / FPR
----------+-------------------------+-------------+--------------
lattice   | 94.63 / 0.18 / 84.94    |  0.00  0.00 | 94.17 / 0.19
w θ=0.10  | 93.51 / 1.12 / 99.98    | -1.11 +0.94 | 93.51 / 1.12
u θ=0.01  | 94.90 / 0.20 / 86.15    | +0.27 +0.01 | 94.48 / 0.21
u θ=0.02  | 95.03 / 0.22 / 87.04    | +0.40 +0.04 | 94.67 / 0.22
u θ=0.04  | 95.05 / 0.31 / 89.08    | +0.42 +0.13 | 94.78 / 0.32
u θ=0.05  | 95.08 / 0.30 / 88.91    | +0.45 +0.12 | 94.71 / 0.30

r20419  7501 grid best_f1 (400n, [200,200])
lattice   | 94.19 / 0.89 / 97.96    |  0.00  0.00 | 94.01 / 0.90
w θ=0.10  | 93.51 / 1.12 / 99.98    | -0.68 +0.23 | 93.52 / 1.12
u θ=0.01  | 94.72 / 0.19 / 85.42    | +0.53 -0.70 | 94.16 / 0.21
u θ=0.02  | 94.93 / 0.23 / 86.96    | +0.74 -0.66 | 94.52 / 0.23
u θ=0.04  | 95.12 / 0.32 / 89.59    | +0.93 -0.57 | 94.70 / 0.34
u θ=0.05  | 95.12 / 0.37 / 90.54    | +0.93 -0.52 | 94.75 / 0.38

r20420  7505 grid best_f1 (500n, [250,250])
lattice   | 93.98 / 0.92 / 97.76    |  0.00  0.00 | 93.84 / 0.93
w θ=0.10  | 93.51 / 1.12 / 99.93    | -0.47 +0.20 | 93.50 / 1.12
u θ=0.01  | 94.68 / 0.22 / 85.97    | +0.70 -0.70 | 94.23 / 0.22
u θ=0.02  | 94.86 / 0.26 / 87.32    | +0.88 -0.66 | 94.45 / 0.26
u θ=0.04  | 95.10 / 0.34 / 89.91    | +1.11 -0.58 | 94.73 / 0.35
u θ=0.05  | 95.13 / 0.38 / 90.85    | +1.14 -0.54 | 94.71 / 0.39

r20420  7505 grid best_fpr (400n, [200,200])
lattice   | 93.88 / 0.90 / 97.03    |  0.00  0.00 | 93.77 / 0.90
w θ=0.10  | 93.51 / 1.12 / 99.97    | -0.37 +0.22 | 93.50 / 1.12
u θ=0.01  | 93.87 / 0.91 / 97.07    | -0.01 +0.00 | 93.77 / 0.91
u θ=0.02  | 93.96 / 0.21 / 83.43    | +0.08 -0.69 | 93.47 / 0.22
u θ=0.04  | 94.37 / 0.29 / 86.41    | +0.49 -0.61 | 93.97 / 0.30
u θ=0.05  | 94.57 / 0.34 / 88.10    | +0.69 -0.56 | 94.20 / 0.35

r20420  7505 grid best_fitness (500n, [250,250])
lattice   | 94.10 / 0.90 / 97.68    |  0.00  0.00 | 93.93 / 0.90
w θ=0.10  | 93.51 / 1.12 / 99.98    | -0.58 +0.23 | 93.52 / 1.12
u θ=0.01  | 94.89 / 0.21 / 86.41    | +0.80 -0.69 | 94.39 / 0.22
u θ=0.02  | 95.01 / 0.24 / 87.36    | +0.91 -0.66 | 94.48 / 0.26
u θ=0.04  | 95.12 / 0.35 / 90.07    | +1.03 -0.55 | 94.69 / 0.36
u θ=0.05  | 95.03 / 0.28 / 88.45    | +0.93 -0.61 | 94.62 / 0.30
```

How to read the probe:
- **Grid genomes are weak evidence for the FPR bound.** Lattice grid genomes sit on attractor A (FPR ≈ 0.9), so any
  θ that escapes A "wins" FPR trivially. The bound only binds on a point-B genome (lattice GA, FPR 0.18), and there is
  one of those (r20419).
- **Seed reproduction on r20420 (grid genomes only):**
  - The weighted θ=0.1 attractor (93.51 / 1.12) reproduces exactly on all three genomes.
  - Unweighted θ 0.01–0.02 escapes it at 2/3 and 3/3 genomes respectively, with FPR 0.21–0.26.
  - On grid genomes, the unweighted vote beats lattice on F1 and FPR together.
- **r20420 and r20421 GA genomes are not available.** Whether θ≈0.02 keeps a lattice-GA genome's 0.18 FPR within
  +0.05 pp is shown on **one** genome (r20419, +0.04 VAL / +0.03 TEST). That is the margin the adoption rule will test.

### θ choice (rule fixed before reading the table above; TEST not consulted)

**The rule:**
- Candidates are θ ∈ {0.01, 0.02, 0.04}. 0.005 is off-grid, and 0.05 was a continuity column only.
- A θ is *eligible* on a genome if VAL ΔF1 > 0 AND VAL ΔFPR ≤ +0.05 pp versus lattice on that genome.
- Pick the θ eligible on the most genomes.
- Break ties by the larger mean VAL ΔF1, then by the smaller θ.

**The result:**

| θ | Eligible genomes | Where it failed |
|---|---|---|
| 0.01 | 4/5 | ΔF1 −0.01 on 7505 best_fpr |
| 0.02 | **5/5** | none |
| 0.04 | 4/5 | ΔFPR +0.13 on the r20419 GA genome |

**→ θ = 0.02** (flow tag `nb02`). It is the only candidate that clears both conditions on every genome, including the
single point-B genome.

## 3. The pre-registered arm

**Arm `nb02`:**
- Clone **7501** (unswr lattice encoff r20419).
- Override **only** these four params: `ids_write_rule="vote"`, `ids_vote_share=0.02`, `balance_classes=false`,
  `ids_encoding_dedupe="off"` (the last is unchanged from 7501, but set explicitly).
- Everything else is inherited from 7501 verbatim. The script asserts the paper-critical keys: `random_3way`,
  BINARY 2-cluster, `max_neurons` 250, `max_bits` 34, `ids_k_folds=5`, `ids_kfold_per_gen=5`, desirability fitness,
  rate 0.25, legacy ranking, receptive fields off.

**Why encoff only:**
- Lattice encoff beats encon on r20419 (TEST 94.17/0.19 vs 93.89/0.19).
- Weighted-ms01 encon was the noisy arm (grid draw spread ±1.5 pp).
- Every offline evidence point is encoff.
- An encon arm would double cost on no evidence. Revisit only if the encoff arm adopts.

**Control:** the existing IDS-56 lattice encoff flows on the same seeds (changed in review, see §5):
- **7509** (r20421)
- **7513** (r20422)
- **7517** (r20423)
- extension seeds: 7501 (r20419) and 7505 (r20420), the θ-tuning seeds, disclosed as such

Each control has the same base params and seed as its arm (offspring seeding no longer reads the wall clock since
worker ABI 14), so the control is **not** re-flown.

**Seeds:** r20421–23 first (standing rule n=3). These are the seeds θ was NOT chosen on. Extend with r20419–20
only per the rule below; the n=5 readout must also print the clean-seed (r20421–23) subset on its own.
All three controls have ids below the arm's, so under FIFO they finish before the arm's first flow starts: the seed
change costs no queue time.

**Metric:**
- Taken from `validation_summaries.threshold_metadata` for the GA phase, `genome_type=best_f1`, `validation_point=final`.
- Mode is `margin_val_cal`: τ calibrated on VAL, scored on the 10% TEST, which is report-only.
- Pairing is by seed.
- All stored threshold modes (this binary-margin pipeline stores 10: argmax, margin_fixed0, margin_train_cal,
  margin_val_cal, margin_train_f1, margin_val_f1, argmax_platt, argmax_beta, argmax_classnorm, margin_classnorm)
  × 5 genome types are reported in Rule-7 layout. Only the line above decides adoption.
- F1 here is **macro-F1**. On unswr (benign 96%) benign-F1 sits near 99.6 and barely moves, so ΔmacroF1 ≈
  ½·Δattack-F1. Report `attack_f1` and TPR (from `confusion`) beside every macro-F1 cell.
- Readout guard: `validation_summaries.worker_abi` must be equal for each arm flow and its paired control. If the worker
  ABI changes before the arm runs, the comparison is cross-build: stop and ask Luiz whether to re-fly the controls.
- Never use `iterations.best_f1`.

### Adoption rule (revised in review, §5)

Definitions, per seed, on GA `best_f1` final `margin_val_cal` TEST: ΔF1 = arm − control (macro-F1, pp);
ΔFPR = arm − control (pp). A seed is a **win** if ΔF1 ≥ +0.05 pp, a **tie** if |ΔF1| < 0.05 pp, a **loss** otherwise.

**At n=3 (r20421–23).** Apply in this order; the first match decides.
1. **REJECT** if wins ≤ 1/3 (ties count as non-wins), OR mean ΔFPR > +0.15 pp.
2. **ADOPT** if wins = 3/3 AND mean ΔF1 ≥ +0.15 pp AND mean ΔFPR ≤ +0.05 pp AND every seed ΔFPR ≤ +0.10 pp.
3. **AMBIGUOUS** otherwise → extend to n=5 with r20419–20. Examples: 2 wins + 1 tie or loss; 3/3 wins with mean ΔF1
   < +0.15; 3/3 wins with mean ΔFPR in (+0.05, +0.15] or one seed above +0.10.

**At n=5 (after extension).**
- **ADOPT** if wins ≥ 4/5 AND mean ΔF1 ≥ +0.15 pp AND mean ΔFPR ≤ +0.05 pp AND every seed ΔFPR ≤ +0.10 pp.
- Otherwise **REJECT**. There is no further extension.

**Why these bounds (measured, not assumed).** Lattice GA `best_f1` finals on unswr at this recipe (IDS-47, 9 flows =
3 seeds × 3 crossovers, plus 7501) give a seed-to-seed SD of **0.016 pp FPR** (≈ 24 benign TEST rows) and
**0.15–0.19 pp macro-F1**. The +0.05 pp mean bound is therefore ≈ 3 seed-SDs, a cost tolerance and not a noise
threshold. The +0.15 pp mean-ΔF1 floor is ≈ 1 seed-SD of F1.

**Scope.** ADOPT means "becomes the unswr write rule for the IDS-56 line". It is not a paper claim: at n=3 it is
provisional, and a paper claim needs the clean-seed subset and a stated N.

**No θ shopping.** Only θ=0.02 is flown. If it rejects, any other θ is a new pre-registration, not a rescue.

**Stop condition (futility gate, revised in review).** When each of 7509/7513/7517 lands, probe5 re-trains its GA
`best_f1` final under θ=0.02 unweighted and reads **VAL only** (do not print or read the TEST columns of this probe
before the arm reads out). Withdraw the arm if, on ≥ 2 of the 3 genomes, VAL ΔF1 ≤ 0 OR VAL ΔFPR > +0.15 pp. The gate
can only withdraw; θ is **not** re-chosen on that data. Queued arm flows are cancelled only on Luiz's named order. All
three controls finish before the arm's first flow under FIFO, so the gate is always decidable in time.

### Estimated duration

| Flow | Measured duration |
|---|---|
| 7501 lattice encoff (r20419) | 5 h 16 min |
| 7502 lattice encon | 4 h 01 min |
| 7503 weighted ms01 encoff | 1 h 01 min (stalled on attractor A at the patience floor) |
| 7505 | still running at 3.6 h (GA gen 62) |

- The unweighted vote leaves attractor A, so the GA should keep climbing like lattice: **~4.5–6 h per flow**.
- The n=3 total is **~14–18 h**. The extension adds ~10–12 h.
- Under FIFO (lowest id first) the arm runs after 7505–7542, which is roughly 2–3 days of queue.
- The arm's flows are queued, not held, so they run ahead of the held backlog.

### Exact params (per flow; only `seed` differs)

`scripts/queue_ids56_unweighted_vote.py --dry-run` prints the full dict. These are the deltas from 7501:

```
seed                  20421 | 20422 | 20423   (ext: 20419 | 20420)
ids_write_rule        "lattice" -> "vote"
ids_vote_share        null      -> 0.02
balance_classes       true      -> false
ids_encoding_dedupe   "off"     (unchanged)
```

The flow names are `IDS56U-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-encoff-r{seed}`.
Each flow carries 2 experiments (grid_search + ga_neurons). The script verifies the experiment count in the DB before
it PATCHes the flow to queued.

## 4. Open issues for `experiment-design` (before queueing)

1. **The +0.05 pp FPR bound is tight.** The r20419 point-B probe sits at +0.04 VAL / +0.03 TEST. +0.05 pp is about
   76 benign TEST rows, and per-seed FPR noise of lattice GA finals on unswr is not yet known (n=1 complete). The bound
   may reject on noise. Decide whether the bound applies to the mean (as written) or per seed.
2. **The GA under the vote will select its own genome.** The offline probe re-trains lattice-evolved genomes. The
   desirability fitness (FPR weight 0.35) may land somewhere else on the TPR/FPR curve.
3. **θ was chosen on VAL of genomes from the same seeds that will be tested** (r20419–20). That is VAL, not TEST, and
   the rule was stated first, but the choice is not independent of r20419/20.

## 5. Review (experiment-design, 10/11/2026)

**Verdict: GO-WITH-CHANGES.** The changes are applied in §3 and in `scripts/queue_ids56_unweighted_vote.py`
(still dry-run only). With them, the design is ready to queue on Luiz's go. Nothing was queued or touched.

### Data this review rests on (DB reads, 10/11 21:16 UTC)

7505's GA is still running (GA gen 64; 7501 stopped at gen 90), so no second IDS-56 lattice GA final exists yet.
The nearest measured proxy is IDS-47: unswr lattice, BINARY, desirability, max250, `random_3way`, 3 seeds × 3
crossovers. It differs from 7501 only in the crossover and in params later made explicit (dedupe, ranking, RF). It
also ran on an older build, so it serves as a noise estimate, not as a control.

```
GA best_f1 final, margin_val_cal (tau on VAL, scored on TEST), macro-F1 / FPR %, N=1 per row
flow  | crossover      | seed   | F1     | FPR    | attack-F1
------+----------------+--------+--------+--------+----------
7224  | inherit        | r20419 | 93.92  | 0.224  | 88.27
7225  | blend_neuron   | r20419 | 94.22  | 0.244  | 88.87
7226  | blend_cluster  | r20419 | 94.36  | 0.188  | 89.13
7234  | inherit        | r20420 | 94.16  | 0.217  | 88.74
7235  | blend_neuron   | r20420 | 94.11  | 0.213  | 88.64
7236  | blend_cluster  | r20420 | 94.32  | 0.217  | 89.05
7246  | inherit        | r20421 | 94.09  | 0.213  | 88.60
7247  | blend_neuron   | r20421 | 94.01  | 0.199  | 88.46
7248  | blend_cluster  | r20421 | 94.02  | 0.211  | 88.46
7501  | blend_cluster  | r20419 | 94.17  | 0.194  | 88.75   (IDS-56 control, ABI 28)

all 9 IDS-47      : F1 94.13±0.15  FPR 0.214±0.016  (N=9)
blend_cluster only: F1 94.23±0.19  FPR 0.205±0.015  (N=3)
TEST benign support 152,390 -> 0.01 pp = 15 rows; attack support 6,036
```

All 10 lattice GA finals sit on point B (FPR 0.19–0.24). None has fallen into attractor A (1.12). Older QUAD/BIN
unswr cohorts do sit on A, so the FPR distribution across recipes is bimodal. Within lattice-B it is tight.

### Q1: the FPR bound. Keep +0.05 pp on the MEAN, add a per-seed ceiling of +0.10 pp

- The concern was that +0.05 pp would reject on noise. It will not: the measured seed SD of lattice FPR is 0.016 pp,
  so +0.05 pp is about 3 seed-SDs. Under a normal model with these SDs (arm noise = control noise, **an assumption**:
  the vote could be noisier), a true ΔFPR of +0.03 passes the mean bound about 98% of the time when the F1 effect is
  real.
- So the bound is a **cost tolerance**: about 76 extra benign alarms per 152k, traded for roughly 2 pp of TPR in the
  r20419 probe. That is a Pareto move along the curve, not dominance. The doc now says so. Whether +0.05 is the right
  price is Luiz's policy call. The statistics support it as written.
- **Mean or per seed:** both. The mean decides, and a per-seed ceiling of +0.10 pp (about 6 seed-SDs) blocks ADOPT.
  Reason: a bimodal metric can hide a regression in a mean. One seed falling to A adds about +0.9 pp and is caught by
  the mean. But a control seed falling to A (−0.9 for the arm) could mask a +0.10 regression on the other two. The
  ceiling closes that hole at no power cost.

### Q2: the θ selection seeds. A real dependency → test on r20421–23 instead

- On data, there is no leak: θ was picked on VAL, and TEST was untouched by the rule.
- On genomes, there is a measured dependency. **Grid connectivity is seed-determined:** 7501 (lattice, r20419) and
  7503 (ms01, r20419) store a byte-identical 500n grid genome (connections md5 `018e5854bc`), and 7505 (r20420)
  does not share it. So the arm's grid stage on r20419/20 would re-evaluate the exact connectivities θ was tuned on.
  The GA starts from those grid elites.
- Every IDS-56 seed sees the same `random_3way` partition. Changing seeds therefore removes the genome overlap but
  does not change the data. That is the right fix, and the only one available.
- **Change:** primary seeds are r20421–23, with controls 7509/7513/7517. All are already queued with ids below the
  arm's, so under FIFO they finish first: **zero added queue time**.
- The extension, if triggered, uses r20419/20 (controls 7501/7505 exist). It is disclosed as the tuning seeds, and the
  clean subset is printed on its own. A paper claim may cite only the clean subset or a fresh-seed replication.

### Q3: the stop condition. Sound as a futility gate; tightened

- A gate that can only withdraw, never tune, is conservative: it can only lower the false-adoption rate. Keep it.
- **"ΔF1 ≤ 0 on both" was too lax:** one bad genome out of two passed. The gate also ignored FPR.
- **Changed to:** with the new seeds, probe the GA `best_f1` finals of 7509/7513/7517. Withdraw if, on ≥ 2/3,
  VAL ΔF1 ≤ 0 OR VAL ΔFPR > +0.15 pp.
- Read **VAL only**. The probe prints TEST, and reading it on the test seeds' control genomes before the readout
  would be a peek.
- The gate checks a necessary condition, not a sufficient one. It retrains lattice-evolved genomes, and the arm's GA
  will pick its own (author's issue 2). The arm's flown result is the only evidence that counts.

### Q4: other findings

1. **The rule was internally inconsistent.** "Wins ≤ 1/3 → REJECT" and "any |ΔF1| < 0.05 → AMBIGUOUS" both fired on
   1 win + 2 ties. The fix: wins and ties are now defined, and the rules are applied in order (REJECT, then ADOPT,
   then AMBIGUOUS).
2. **3/3 sign wins alone is a weak bar.** Under no true effect, P(3/3) = 1/8. Sequentially, through the n=5
   extension, the original rule adopts about **21%** of the time on a null F1 effect. That figure is simulated from
   the measured SDs; it is a model, not a measurement. Adding **mean ΔF1 ≥ +0.15 pp** (about one seed-SD) cuts this to
   about **8.5%**. Power at the probe's effect size (+0.5 pp, ΔFPR +0.03) stays at about 98%. Power falls from 0.84
   to 0.73 only if the true effect is as small as +0.25 pp, which the floor is meant to screen out.
3. **Metric.** The decision stays on macro-F1 (the project convention). On a 96%-benign binary task, ΔmacroF1 ≈
   ½·Δattack-F1, so attack-F1 and TPR are now reported beside it.
4. **"7 threshold modes" was wrong for this pipeline.** It stores 10 binary-margin modes, and the reporting line now
   lists them.
5. **Pairing is real but partial.** Same seed means the same folds and the same grid connectivity (verified above),
   but the GA diverges from generation 1. Treat the pairs as matched starting points. Do not read the paired SD as a
   within-genome SD.
6. **ABI drift.** The controls are ABI 28, and the arm runs 2–3 days later. `MIN_WORKER_ABI` is a floor (`<`), not an
   equality. A readout guard (equal `worker_abi` per pair, else ask Luiz about re-flying controls) is now in §3.
7. **Multiplicity.** Only one θ is flown, so there is no multiplicity at the test stage. The 5-candidate VAL search is
   disclosed. A θ=0.02 rejection does not license flying θ=0.01/0.04 as a rescue; that would need a new
   pre-registration (now stated).
8. **Reporting.** n=3 is labelled provisional. Cells are `mean±std (N)`. Show the Pareto view (best F1 at FPR < 0.25,
   etc.) beside the 5 tables. The weighted ms01 encoff flows on the same seeds (7511/7515/7519; ext 7503/7507) go in as the third point for
   vote-vs-weighting attribution, with the θ mismatch (0.1 vs 0.02) stated.

### Ready to queue?

Yes, on Luiz's go: `scripts/queue_ids56_unweighted_vote.py` (no flags) queues r20421–23. The dry run was verified
10/11 and prints exactly those three names, with params unchanged from §3 apart from the seed. One action remains
before the arm's first flow starts: the futility-gate probe on 7509/7513/7517 as each lands.
