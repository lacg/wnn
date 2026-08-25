# ======================================================================
# SECTION 8 — IDSX AC/CE MATCHED-PAIR COHORT (INTERIM, n=2 — NOT PUBLISHABLE)
# ======================================================================

**Appended 24/08/2026. NOTHING in this section is a paper number.** Two of four datasets
have reached n=2 seeds; two are at n=1. It is recorded here because it CORRECTS three
banked claims that ARE in the paper track, and because a reader of the sections above
would otherwise carry the retracted versions forward.

Metric contract unchanged from the top of this file: held-out `validation_summaries.
threshold_metadata` at `validation_point='final'`. No `iterations.best_f1` anywhere.

## 8.0 Cohort state

```
IDSX-%          completed 49/170   queued 120   running 1   failed 0
compute booked  70.2 h over 49 runs; avg 85.9 min/run
per-dataset     unswr-quad 30.0 | ciciot 59.8 | cicids 128.4 | unswr-qsr 202.1  min/run
ETA (dataset-weighted, 229.4 h remaining)  03/09/2026 02:28 UTC

dataset            r20401   r20402   n
unswr-quad-64b       8/8      8/8    2
ciciot-quad-96b      8/8      8/8    2
cicids-quad-96b      8/8      1/8    1
unswr-qsr-64b        8/8      0/8    1
```
Design: 3 matched AC/CE pairs (identical weight STRUCTURE, only `w_acc` <-> `w_ce` swapped)
+ CE20 + each dataset's OWN production control (Wb / Wc / Wa), 5 seeds, `zscore`,
patience 5, gens 250, k-folds 5, `random_3way`. Plus a 9th arm on both unswr datasets,
`B34-CTRL` (min_bits 4 -> 34), which is a separate bits-scope control, 0/10 completed.

## 8.1 CORRECTION 1 — CE20 beats production, and it SURVIVES budget-matching

This one is a **strengthening**, not a retraction. Paired by seed, held-out val_cal F1,
`best_f1`, GA Neurons, against **Wb-CTRL** absolute 88.831+-0.092 on IDSZ-unswt-quad-16b:

```
arm      | per-seed deltas (pp)                | mean   | sd    |   t   |   p    | dFPR
---------+-------------------------------------+--------+-------+-------+--------+-------
CE20     | +0.863 +1.059 +0.630 +1.197 +1.006  | +0.951 | 0.216 | +9.86 | 0.0006 | -0.698
B05-CE   | +0.968 +1.082 +0.943 +0.554 +0.181  | +0.746 | 0.373 | +4.47 | 0.0111 | -1.363
B10-CE   | +0.417 +0.520 +1.282 +0.862 +0.578  | +0.732 | 0.349 | +4.69 | 0.0094 | -0.959
CE40     | +0.997 +0.748 +0.749 +0.647 +0.464  | +0.721 | 0.193 | +8.34 | 0.0011 | -0.395
B15-CE   | +0.318 +0.262 +0.095 +0.669 +0.704  | +0.410 | 0.266 | +3.44 | 0.0262 | -0.066
B05-AC   | +0.205 -0.007 +0.073 -0.151 -0.110  | +0.002 | 0.143 | +0.03 | 0.9758 | -0.183
B10-AC   | -0.045 -0.046 -0.403 +0.192 -0.135  | -0.087 | 0.214 | -0.91 | 0.4130 | +0.098
B15-AC   | +0.119 -0.030 -0.359 -0.221 -0.168  | -0.132 | 0.183 | -1.61 | 0.1834 | -0.749
C35-CTRL | +0.379 -0.236 -0.127 -0.274 -0.315  | -0.115 | 0.285 | -0.90 | 0.4184 | -0.411
```
**Every arm with `w_ce > w_acc` beats production 5/5 seeds, above the 0.409pp bar, with FPR
directionally better too — a Pareto move, not a trade. All three AC arms and C35-CTRL are
flat.** A family with a consistent sign is not one lucky arm out of 23, which is why the
selection-bias objection does not sink it. vs C35-CTRL: CE20 +1.066 (p=0.0049).

**Budget is NOT the explanation for CE20:** CE20 consumed 62.0 generations, Wb-CTRL 62.0 —
identical. On the 3 budget-EXACT seeds CE20 is +0.944pp vs +0.951pp unrestricted.

**It does NOT transfer.** CE20 vs its own control on the four IDSX datasets:
unswr-quad -0.010 / ciciot +0.101 / cicids -0.053 / unswr-qsr +0.061 pp — all inside noise.
**Scope the claim to UNSW temporal_3way QUAD 16b.**

## 8.2 CORRECTION 2 — the +0.70pp AC/CE effect does NOT generalize

18 matched pairs across 4 datasets, held-out val_cal F1, `best_f1`, GA:
```
dataset          n_pairs  mean dF1 (CE-AC)  CE wins   note
unswr-quad-64b       6        -0.017          0/6     SATURATED - see 8.4
ciciot-quad-96b      6        -0.015          3/6     MDD 0.30pp; adequately powered, reports ~0
cicids-quad-96b      3        -0.203          0/3     REVERSES - AC wins
unswr-qsr-64b        3        +0.110          3/3     the ONLY dataset supporting it (n=1 seed)
-------------------------------------------------------------------------------------
pooled              18        -0.026          6/18
```
Full 35-column surface (cells favouring CE): IDSZ 29/35 · qsr 35/35 · ciciot 19/35 ·
unswr-quad 13/35 · cicids 4/35.

**Revised claim, safe to publish:** *"On UNSW temporal_3way QUAD 16b, `w_ce > w_acc` beats
`w_acc > w_ce` by +0.70pp held-out val_cal F1 (n=5, 3/3 pairs, 29/35 surface cells). It does
NOT generalize: of four other dataset/substrate combinations, 1 reproduces the sign, 1
reverses, 2 are null."*

**PARETO NOTE that must travel with it:** on ciciot the CE arms win FPR while tying F1 —
B10-CE 7.27+-0.33 vs B10-AC 8.75+-0.71; B15-CE 7.45+-0.16 vs B15-AC 8.98+-0.30. That is the
one positive signal in the two n=2 datasets, and it is on the axis an IDS deploys against.

## 8.3 CORRECTION 3 — generations is a MEDIATOR, not a confounder

The banked `partial r(w_ce, F1 | gens) = -0.009` is real and reproduces (-0.013), but reading
it as *"the weight effect is entirely mediated by budget"* is WRONG on two counts.

1. **It is about `w_ce` only.** Conditioning on generations kills that half
   (+0.420 -> -0.123) and leaves `w_acc` untouched (-0.499 -> -0.487). The
   budget-independent mechanism is **"weighting accuracy hurts"**, not "weighting CE helps".
2. **Conditioning on generations is over-adjustment.** Patience terminates when an arm stops
   improving, so generations-consumed lies ON the causal path from weight vector to outcome.
   It is a mediator. Blocking it estimates the DIRECT effect ("how much does CE help other
   than by sustaining productive search"), which is not what gets deployed.

```
                                   estimand                      pairs (B05/B10/B15)
unrestricted, patience running    TOTAL effect  <- DEPLOY THIS   +0.74 / +0.82 / +0.54 (mean +0.70)
budget-exact subset               DIRECT effect <- over-adjusted +0.18 / +0.52 / +0.10 (mean ~+0.37)
```
**Report +0.70pp. The +0.37pp figure is an over-adjusted estimate and must not be quoted as
"the honest number".** The mediator reading is licensed because the extra search buys genuine
held-out gain: at FIXED weights, within-arm r(gens, held-out F1) = **0.309**.

## 8.4 Per-dataset power — the 0.409pp bar is IDSZ's, not universal

```
dataset                 | runs | sigma_seed(F1) | df | MDD 80% power (n=5)
------------------------+------+----------------+----+--------------------
IDSZ unswt-quad-16b     |  115 |     0.2282     | 92 |      0.4041   <- the banked bar
ciciot-quad-96b         |   16 |     0.1686     |  8 |      0.2985   <- the workhorse
cicids-quad-96b         |    9 |     0.0441     |  1 |      0.0782   (df=1, treat as unmeasured)
unswr-quad-64b          |   16 |     0.0063     |  8 |      0.0112   SATURATED - see below
unswr-qsr-64b           |    8 |      n/a       |  0 |       n/a
```
**`unswr-quad-64b` is saturated and cannot host a weight effect at all.** GA gain over
grid_search is **+0.004 pp** held-out F1 across all 16 runs; the entire spread over 8 weight
vectors x 2 seeds is 0.038 pp F1 and 0.001 pp FPR. There is no search trajectory for a
fitness weight to steer. It is a powerful falsifier (a +0.70pp claim dies at ~60 sigma) and
useless for estimating an effect. GA gain by dataset: ciciot +1.973 · IDSZ +0.470 ·
cicids +0.235 · unswr-qsr +0.058 · unswr-quad +0.004 pp.

## 8.5 Methodological notes for whoever writes this up

- **IDSX is NOT budget-matched.** Verified on all 170 experiments: `max_iterations=250`,
  `patience=5`. The documented budget-match recipe was never applied, so IDSX reproduces the
  budget asymmetry it was meant to remove (ciciot: CE arms ~85 gens vs AC ~65). It also has
  **no unswt / no 16b / no temporal cell**, so it was only ever a GENERALIZATION test, never
  a replication of IDSZ's condition.
- **`best_f1` vs `best_fitness` is a genuine fork.** They select different genomes in 73% of
  IDSX runs and 74% of IDSZ runs. Going forward: name **`best_fitness`** primary (it is the
  deployment read and what the arm actually produces), `best_f1` the fixed-selector secondary,
  and state that the choice was made AFTER seeing 49 runs. It is not load-bearing here — both
  reads fail to replicate +0.70pp equally (-0.026 vs -0.002 pooled).
- **SP100 cannot be pooled in as extra control power.** Same Wb weights, same dataset/split/
  bits, but a pre-19/08 code era: SP100 86.180+-1.906 F1 / 15.430+-7.263 FPR (29 runs) vs
  in-cohort Wb-CTRL 88.831+-0.103 / 9.529+-1.027 (5 runs) — **dF1 +2.651pp, Welch t=+7.43,
  and 18.5x the seed variance.** Config identity is not experimental identity. (The variance
  collapse is itself notable: the 19/08 change is what made 0.4pp effects resolvable at all.)
- **n=2 is not a result.** Do not rank arms, declare winners, or move any paper table on the
  strength of Section 8. Revisit at n=5.
