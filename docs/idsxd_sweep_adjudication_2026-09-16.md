# IDSXD desirability fitness-weight sweep — statistical adjudication

Date: 16/09/2026 (DB read ~21:00 UTC). Author: experiment-design agent. READ-ONLY; no flow was touched.
Source of truth: `validation_summaries` rows of the **GA Neurons (final) phase**, `threshold_metadata.val_cal`
(Protocol v2 on `random_3way`: threshold F1-optimal on the 10% VAL, metrics reported on the 10% TEST partition).
`iterations.best_f1` (during-search K-fold) was NOT used anywhere. Every number below is an actual DB read;
nothing is estimated. Script: `/private/tmp/claude-501/-Users-lacg-wnn/1ca3ecd9-4d49-4bed-87a0-ddb2c56c50a7/scratchpad/idsxd_adj.py`.

Statistics: paired-by-seed differences (arm − control, same seed), 95% t-CI with df = n−1 (t = 4.303 at n=3).
MDD = minimum detectable paired difference at 80% power, two-sided α = 0.05, exact noncentral-t, using the
OBSERVED paired SD_d. Caveat that applies to every MDD in this document: an SD estimated at df = 2 has a 95% CI
of [0.52×, 6.28×] its point value, so the MDDs are order-of-magnitude planning numbers, not guarantees.
"Separated" (for tie sets) = pairwise paired CI excludes 0 OR |gap| > 2·SD_d; otherwise the two arms are tied.

## Executive verdict

```
    Dataset         | Verdict              | Basis (best_f1 genome, GA final, held-out val_cal, n=3 paired)
    ----------------+----------------------+-------------------------------------------------------------------------------------
    cicids-quad-96b | TIED{all 8 arms}     | no arm's dF1 CI vs Wa-CTRL excludes 0; largest |gap| 0.049 pp vs MDD(n=3) 0.35 pp.
                    |   = NULL at ceiling  | F1 99.42-99.52, FPR 0.12-0.25. Nothing resolvable at n<=10 (pooled MDD n=10 = 0.107).
    ciciot-quad-96b | TIED{B15-AC,B10-CE,  | era-clean: NO arm's dF1 CI excludes 0. B10-CE FPR -1.18 [-1.78,-0.57] on best_f1 only.
                    |  Wc-CTRL}(+B10-AC*)  | *B10-AC is the only nominal F1 winner (+0.117 [+0.073,+0.160]) but is era-MIXED (2 pre-fix
                    |                      |  seeds at bits=64, 1 post-fix at bits=80) -> PROVISIONAL until r20403/r20404 -w64fix rerun.
    unswr-qsr-64b   | TIED{all 9 arms}     | no dF1 CI excludes 0 vs Wb-CTRL or vs B34-CTRL; largest positive gap +0.044 pp; CE20 is
                    |   = NULL             | borderline WORSE (-0.068 [-0.135,+0.000]). B34-CTRL vs Wb-CTRL +0.028+-0.109: no effect.
    unswr-quad-64b  | SATURATED            | 25/29 runs at FPR 1.118-1.121 and F1 93.40-93.50 regardless of arm or bits (12-16 free,
                    |                      | 34 forced). The 3 off-basin runs are all seed 20403 in 3 different arms -> seed, not arm.
```

**No arm beats its dataset control on era-clean data at n=3, on any of the four datasets.** The banked IDSZ prior
(CE20 beats production on unswt-16b/zscore) does NOT transfer: CE20 is the top mean only on cicids (+0.045, n.s.)
and is at or below control on ciciot (−0.144), unswr-qsr (−0.068, borderline) and unswr-quad (−0.027).
The one consistent-sign structural finding is **acc-weighted > ce-weighted at every bias level on ciciot**
(B05: −0.162±0.076, B10: −0.058±0.014, B15: −0.278±0.105 pp F1 for CE−AC; all three pairwise-separated; the B15
pair is era-clean with CI [−0.539, −0.017]) and the same sign, not significant, on cicids (3/3 levels). That is the
opposite direction to IDSZ's "every w_ce > w_acc arm won" on unswt.

## Inventory and cost check

Seeds are shared (20403/20404/20405 on every arm; B34-CTRL additionally 20401/20402), so pairing by seed is valid.
Actual wall time per run differs materially from the figures supplied for unswr:

```
    Dataset      | runs | arms | actual h/run mean+-SD (min..max) | supplied | note
    -------------+------+------+----------------------------------+----------+----------------------------------
    cicids-quad  |   24 |    8 |  1.57+-0.66 (0.87..3.26)         |   1.6    | ok
    ciciot-quad  |   24 |    8 |  3.98+-1.70 (1.65..7.01)         |   4.0    | ok
    unswr-qsr    |   29 |    9 | 10.65+-6.13 (2.72..27.33)        |   5.8    | ~1.8x MORE expensive than supplied
    unswr-quad   |   29 |    9 |  0.86+-0.39 (0.38..1.71)         |   5.8    | ~7x CHEAPER than supplied
```
Cost figures below use the ACTUAL per-dataset means.

Pending/related flows (not in any table): 6050 `IDSXD-ciciot-quad-96b-B05-AC-r20405-w64fix` RUNNING (GA gen 33 at
20:54 UTC; no GA-final row yet); 6051 `IDSXD-ciciot-quad-96b-B05-CE-r20404-w64fix` QUEUED; 55 unswt flows QUEUED
(0 completed — unswt skipped as instructed).

## Q1 — per-arm n, mean±SD (held-out val_cal, GA final)

Full tables for best_f1, best_fpr and best_fitness are in Detail §1-§4 below. Compact best_f1 view:

```
    Arm       | cicids F1 / FPR / Acc          | ciciot F1 / FPR / Acc          | unswr-qsr F1 / FPR / Acc       | unswr-quad F1 / FPR / Acc
    ----------+--------------------------------+--------------------------------+--------------------------------+--------------------------------
    B05-AC    | 99.43+-0.11  0.23+-0.10  99.64 | 92.96+-0.18  7.91+-0.51  96.48 | 94.34+-0.12  0.59+-0.06  99.14 | 93.45+-0.05  1.12+-0.00  98.91
    B05-CE    | 99.42+-0.10  0.25+-0.10  99.64 | 92.80+-0.10  7.83+-0.15  96.39 | 94.32+-0.01  0.53+-0.09  99.15 | 93.47+-0.02  1.12+-0.00  98.92
    B10-AC    | 99.47+-0.07  0.21+-0.08  99.66 | 92.97+-0.05  7.14+-1.19  96.46 | 94.29+-0.07  0.59+-0.08  99.14 | 93.47+-0.02  1.02+-0.16  98.93
    B10-CE    | 99.45+-0.06  0.17+-0.11  99.65 | 92.91+-0.06  6.75+-0.35  96.42 | 94.31+-0.05  0.58+-0.07  99.14 | 93.48+-0.02  1.12+-0.00  98.92
    B15-AC    | 99.48+-0.06  0.15+-0.05  99.67 | 92.98+-0.08  6.79+-0.76  96.46 | 94.35+-0.08  0.61+-0.05  99.14 | 93.48+-0.02  1.12+-0.00  98.92
    B15-CE    | 99.44+-0.08  0.23+-0.07  99.64 | 92.70+-0.14  7.52+-1.02  96.33 | 94.36+-0.13  0.65+-0.04  99.13 | 93.49+-0.01  1.12+-0.00  98.92
    CE20      | 99.52+-0.03  0.12+-0.01  99.70 | 92.71+-0.27  7.90+-1.42  96.34 | 94.25+-0.07  0.59+-0.10  99.13 | 93.41+-0.07  0.98+-0.23  98.93
    CTRL      | 99.47+-0.09  0.21+-0.08  99.67 | 92.85+-0.05  7.93+-0.13  96.42 | 94.32+-0.05  0.61+-0.06  99.14 | 93.44+-0.04  0.99+-0.23  98.94
    B34-CTRL  |              --                |              --                | 94.38+-0.06  0.59+-0.03  99.15 | 93.49+-0.01  1.12+-0.00  98.92
    (n=3 everywhere; B34-CTRL n=5. CTRL = Wa (cicids), Wc (ciciot), Wb (unswr).)
```

## Q2 — paired deltas vs control, which CIs exclude 0, genome-type agreement

Full paired tables (dF1/dFPR/dAcc × 3 genome types, plus vs B34-CTRL on unswr) are in Detail §1-§5.
Every CI that excludes 0 anywhere in the sweep (all 4 datasets × 8-9 arms × 3 genome types × 3 metrics):

```
    Dataset    | Arm    | genome       | metric | delta mean+-SD  [95% CI]        | corroborated by other genome types?
    -----------+--------+--------------+--------+---------------------------------+------------------------------------------------
    cicids     | B05-AC | best_fpr     | dFPR   | +0.081+-0.007 [+0.063,+0.098]   | NO: best_f1 +0.024 [-0.121,+0.168] (worse FPR anyway)
    ciciot     | B10-AC | best_f1      | dF1    | +0.117+-0.017 [+0.073,+0.160]   | YES sign (+++), best_fpr CI also excludes 0:
    ciciot     | B10-AC | best_fpr     | dF1    | +0.098+-0.029 [+0.026,+0.170]   |   best_fitness +0.093 [-0.011,+0.196] just misses. ERA-MIXED.
    ciciot     | B10-AC | best_fitness | dAcc   | +0.021+-0.004 [+0.010,+0.031]   | best_f1 +0.041 [-0.025,+0.106]
    ciciot     | B10-CE | best_f1      | dFPR   | -1.178+-0.244 [-1.784,-0.572]   | sign YES (---), magnitude NO: best_fpr -0.698+-0.804, best_fit -0.988+-0.718
    unswr-qsr  | B05-CE | best_fpr     | dF1    | +0.068+-0.021 [+0.015,+0.121]   | NO: best_f1 +0.001 [-0.116,+0.119]
    unswr-qsr  | B10-AC | best_fpr     | dAcc   | +0.015+-0.006 [+0.001,+0.029]   | trivial magnitude (0.015 pp)
    unswr-quad | B10-CE | best_fpr     | dF1 vs B34 | -0.016+-0.006 [-0.032,-0.001] | trivial magnitude (0.016 pp), one genome type
```
Pareto rule: an arm counts as directionally better only if best_f1, best_fpr AND best_fitness agree on sign for F1
and for FPR. Arms that pass that (point estimates only, no CI support): cicids CE20 (F1 +++ / FPR −−−);
ciciot B10-AC and B10-CE (F1 +++ / FPR −−−); unswr-qsr B05-AC and B05-CE (F1 +++ / FPR −−−). Everything else either
has mixed signs across genome types or is a Pareto MOVE (e.g. ciciot B15-CE: F1 down, FPR down). unswr-quad's
uniform "+++/+++" pattern is the saturation artifact described under Q6, not a Pareto move.

## Q3 — can a winner be named at n=3?

```
    Dataset    | Winner rule met? | Top arm by mean F1 | NOT separated from                                    | Tied set
    -----------+------------------+--------------------+-------------------------------------------------------+------------------------------------------
    cicids     | NO               | CE20 99.519        | B05-AC,B05-CE,B10-AC,B15-AC,B15-CE,Wa-CTRL (sep only from B10-CE) | ALL 8
    ciciot ALL | B10-AC only      | B15-AC 92.983      | B05-AC,B05-CE,B10-AC,B10-CE,CE20,Wc-CTRL              | {B10-AC,B15-AC,B05-AC} top; B10-CE close
               |  (era-mixed)     |                    | B10-AC separated from Wc-CTRL,B05-CE,B10-CE but NOT from B05-AC,B15-AC |
    ciciot POST| NO               | B15-AC 92.983      | B10-CE, CE20, Wc-CTRL (sep from B15-CE)               | {B15-AC,B10-CE,Wc-CTRL} (+CE20 by its own noise)
    unswr-qsr  | NO               | B34-CTRL 94.376    | everyone                                              | ALL 9 (CE20 borderline worse)
    unswr-quad | NO (saturated)   | B15-CE 93.494      | everyone                                              | ALL 9
```

## Q4 — what n would resolve each tied set (paired t, 80% power, observed SD_d)

```
    Dataset     | Tied set (candidates)      | pooled SD_d | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10 | largest gap in set   | n needed for that gap
    ------------+----------------------------+-------------+---------+---------+---------+----------+----------------------+-----------------------
    cicids      | all 8                      |   0.107     |  0.350  |  0.180  |  0.136  |  0.107   | CE20 +0.045 (SD 0.088)| 32
    ciciot ALL  | B10-AC/B15-AC/B05-AC/B10-CE|   0.152     |  0.495  |  0.255  |  0.193  |  0.151   | B15-AC +0.131 (0.120) | 9   (B10-AC +0.117/0.017 -> 3; B10-CE +0.059/0.028 -> 5)
    ciciot POST | B15-AC/B10-CE/Wc-CTRL      |   0.186     |  0.607  |  0.313  |  0.237  |  0.185   | B15-AC +0.131 (0.120) | 9   (B10-CE -> 5)
    unswr-qsr   | all 9                      |   0.087     |  0.284  |  0.146  |  0.111  |  0.087   | B15-CE +0.044 (0.097) | 40  (CE20 -0.068/0.027 -> 4, negative)
    unswr-quad  | all 9 (saturated)          |   0.049     |  0.160  |  0.082  |  0.062  |  0.049   | B15-CE +0.053 (0.048) | 9   (meaningless: same basin)
```
FPR on ciciot is far noisier: paired SD_d 0.68-1.13 pp → MDD n=5 1.15-1.90 pp, n=10 0.68-1.12 pp. Only B10-CE's
−1.18 pp (best_f1 genome, SD_d 0.244) is inside reach; on the best_fpr genome the same arm has SD_d 0.804 (MDD n=10 = 0.80).

Marginal cost to extend (seeds × arms; B34-CTRL already n=5; ciciot B10-AC also needs r20403 and r20404 re-run
post-fix to become era-clean, +2 runs):

```
    Dataset     | Set to extend                     | to n=5           | to n=7            | to n=10
    ------------+-----------------------------------+------------------+-------------------+-------------------
    cicids      | all 8 (nothing to resolve)        | 16 runs,  25 h   | 32 runs,   50 h   | 56 runs,   88 h
    ciciot      | B15-AC, B10-CE, B10-AC, Wc-CTRL   | 8+2 = 10, 40 h   | 16+2 = 18, 72 h   | 28+2 = 30, 119 h
    unswr-qsr   | all 8 free arms                   | 16 runs, 170 h   | 32 runs,  341 h   | 56 runs,  596 h
    unswr-quad  | none (saturated)                  |  --              |  --               |  --
```
Reading: at n=5 ciciot can confirm B10-CE's small F1 gain (+0.059, needs 5) and make B10-AC era-clean; it CANNOT
resolve B15-AC vs control (needs ~9) nor any ciciot FPR claim except B10-CE's. Nothing on cicids or unswr-qsr is
resolvable at n<=10 because the observed gaps (<=0.05 pp positive) are below MDD(n=10). n=7 buys little over n=5
on ciciot (MDD 0.24 vs 0.31, gaps 0.06-0.13); n=10 is what B15-AC would need, if its SD_d holds.

## Q5 — ciciot restricted to the 5 post-fix arms; do the pre-fix arms change anything?

Era split by `started_at` vs the 30/08/2026 02:27 UTC fix: B05-AC 3/3 pre, B05-CE 3/3 pre, B10-AC 2 pre + 1 post
(r20405 started 30/08 05:48), B10-CE/B15-AC/B15-CE/CE20/Wc-CTRL 3/3 post — matches the stated context.

The era confound is architectural, not just numerical: the final GA genome of **every pre-fix run has bits = 64
(8/8)**, while **14 of 15 post-fix runs have bits = 80** (the exception is B15-CE r20404 at 64). With max_bits = 100
on ciciot, the OR-fold bug made bits > 64 unusable, so the pre-fix GA could not climb past 64; post-fix it does.
Pre-fix arms therefore measured "this weight vector under a 64-bit cap". They are not comparable to post-fix arms.

Post-fix-only results (Detail §5): no arm's dF1 CI excludes 0 (B10-CE +0.059 [−0.010,+0.127]; B15-AC +0.131
[−0.168,+0.430]; B15-CE −0.147; CE20 −0.144). B10-CE dFPR −1.178 [−1.784,−0.572] excludes 0 on best_f1 only.
Tied set {B15-AC, B10-CE, Wc-CTRL}; B15-AC is separated only from B15-CE (+0.278±0.105).

Do the pre-fix arms change the conclusion? **Yes, in one direction only:** with all arms included, B10-AC is the
single nominal F1 winner (+0.117 [+0.073,+0.160], separated from Wc-CTRL); era-clean there is no winner. Its three
per-seed deltas are +0.120 (pre, 64b), +0.097 (pre, 64b), +0.132 (post, 80b) — the post-fix seed is not an outlier
on F1, but its FPR (5.77) is 2 pp below the pre-fix seeds (7.78, 7.87), so era moves FPR much more than F1. The
pre-fix arms B05-AC/B05-CE do not win in either analysis, so their inclusion/exclusion changes no other verdict.
The pre-fix arms DO affect the AC>CE finding's evidence: the B05 pair is pre-fix (both 64b, still paired fairly
within era), the B10 pair is mixed, the B15 pair is era-clean and separated on its own.

B05-AC r20405-w64fix (flow 6050): NOT complete (GA at gen 33, 20:54 UTC). Its after-grid checkpoint (best_f1
val_cal) is F1 90.602 / FPR 16.500 / Acc 95.492 vs the pre-fix r20405 grid checkpoint 90.661 / 17.045 / 95.548 —
indistinguishable at the grid stage. The GA-final comparison (pre-fix 93.143 / 7.590 / 96.572, bits 64) is pending.

## Q6 — sanity: n=3 SDs vs the older sigmas

```
    Dataset     | older sigma (pp) | this sweep: pooled within-arm SD (RMS) / median | read
    ------------+------------------+-------------------------------------------------+------------------------------------------------------
    cicids      | 0.044            | 0.078 / 0.077  (8 arms, df=16)                  | LARGER (~1.8x). df=16 CI on 0.078 is [0.058, 0.119]; 0.044 lies below it.
                |                  |                                                 | Likely architecture variance: final neuron counts range 94..442 within an arm.
    ciciot      | 0.169            | 0.137 / 0.090                                   | consistent (CE20 alone contributes SD 0.273)
    unswr-quad  | 0.006            | 0.035 / 0.020                                   | larger but inflated by the 3 off-basin seed-20403 runs; in-basin spread ~0.02
    unswr-qsr   | unmeasured       | 0.079 / 0.068                                   | first measurement; MDD(n=5) ~0.15 pp
```
Is any dataset decisive at n=3? **cicids is decisive for the null**: MDD(n=3) = 0.35 pp and every observed gap is
<= 0.05 pp; with the SD 1.8x larger than the old sigma the arms are even less distinguishable than planned. It is
NOT decisive for a winner. unswr-quad is decisive that there is nothing to find (saturation). Neither ciciot nor
unswr-qsr is decisive either way.

## Q7 — one cross-dataset arm, or per-dataset winners?

There is no arm that wins anywhere on era-clean data, so there is nothing to adopt as a cross-dataset weight vector
on the evidence of this sweep; the honest per-dataset winner on cicids, unswr-qsr and unswr-quad is the existing
control (no change), and on ciciot it is an unresolved tie in which the control is a member. Two "never worse"
candidates exist by sign only: B15-AC has dF1 >= 0 on 4/4 datasets (+0.007, +0.131, +0.029, +0.035; sign-test
p = 0.125, no CI excludes 0), and B10-CE has dFPR <= 0 on 3/4 (−0.039, −1.178*, −0.024; the 4th, unswr-quad +0.131,
is the seed-20403 basin artifact) with dF1 within ±0.06 everywhere. CE20, the banked unswt winner, does not
transfer (top mean only on cicids, below control on the other three). The one structural, consistent-sign result is
AC-heavy > CE-heavy at every bias level on ciciot (3/3 pairwise-separated, Acc up, FPR not worse) with the same
sign on cicids (3/3, n.s.) — which argues that the weight direction is dataset-dependent (IDSZ found the opposite
on unswt), so "per-dataset winner" is the right frame, and at n=3 that winner is the control everywhere except
possibly ciciot. If one arm had to be carried into n=100 cohorts on all datasets, B15-AC is the only one with no
negative sign anywhere, and that is a provisional, point-estimate statement.

## Recommendation

1. cicids: do not extend. Keep Wa-CTRL. (If an FPR operating point matters, CE20 has the lowest FPR 0.12±0.01 and
   the lowest variance, but its −0.090 pp FPR delta has CI [−0.281, +0.102].)
2. unswr-quad: do not extend; SATURATED. The weight sweep cannot act on a substrate pinned at FPR 1.118.
3. unswr-qsr: do not extend for a winner (nothing positive within reach at n<=10; 10.65 h/run). Optional 2 extra
   seeds of CE20 + Wb-CTRL (4 runs, ~43 h) would confirm CE20 is worse — low value.
4. ciciot: extend {B15-AC, B10-CE, B10-AC, Wc-CTRL} to n=5 with 2 new seeds (interleaved: one seed of each arm,
   then the second) PLUS post-fix reruns of B10-AC r20403 and r20404 — 10 runs, ~40 h. Decision rule set in
   advance: B10-CE is adopted only if at n=5 its paired dF1 CI excludes 0 AND dFPR is <= 0 on all three genome
   types; B15-AC is adopted only if its paired dF1 CI excludes 0 (expected to need ~n=9; if n=5 shows SD_d <= 0.06,
   go to n=10 = +20 runs). Otherwise Wc-CTRL stays.
5. Wait for 6050 (B05-AC r20405-w64fix) before quoting any B05-AC/B05-CE ciciot number; label all pre-fix ciciot
   rows "64-bit-capped era" in any table.

## What n=3 cannot support (explicit)

- Any ordering among arms whose F1 means differ by less than ~0.35 pp (cicids), ~0.5-0.6 pp (ciciot), ~0.28 pp
  (unswr-qsr), ~0.16 pp (unswr-quad). Every positive gap observed is below those thresholds.
- Any FPR claim on ciciot below ~2.6-3.7 pp except B10-CE's −1.18 pp, and that only on one genome type.
- Any statement about the SD itself: at df = 2 the SD is known to within a factor of ~6 upward, so the MDDs and
  "n needed" values above can move by that factor.
- "B10-AC wins on ciciot": era-mixed, 2/3 seeds measured under the 64-bit cap. PROVISIONAL.
- "CE20 beats production" from IDSZ: not reproduced on any of these four datasets under desirability.

---

# Detail tables (generated from the DB; percentages; mean±SD; n stated)

## 0. Cohort inventory (completed, non-unswt)

```
    Dataset      | runs | arms | seeds            | actual h/run mean±SD (min..max) | given h/run
    -------------+------+------+------------------+---------------------------------+------------
    cicids-quad  |   24 |    8 | 20403,20404,20405 |  1.57±0.66 (0.87.. 3.26)        | 1.6
    ciciot-quad  |   24 |    8 | 20403,20404,20405 |  3.98±1.70 (1.65.. 7.01)        | 4.0
    unswr-qsr    |   29 |    9 | 20401,20402,20403,20404,20405 | 10.65±6.13 (2.72..27.33)        | 5.8
    unswr-quad   |   29 |    9 | 20401,20402,20403,20404,20405 |  0.86±0.39 (0.38.. 1.71)        | 5.8
```

Weight vectors (w_ce, w_acc, w_f1, w_fpr) and bits range per arm (identical across datasets except where noted):
```
    Arm       | w_ce  w_acc  w_f1  w_fpr | bits    | note
    ----------+--------------------------+---------+----------------------------------
    B05-AC    | 0.225 0.675  0.05  0.05  |  4..34  | 
    B05-AC    | 0.225 0.675  0.05  0.05  |  4..100 | ciciot max_bits=100
    B05-CE    | 0.675 0.225  0.05  0.05  |  4..34  | 
    B05-CE    | 0.675 0.225  0.05  0.05  |  4..100 | ciciot max_bits=100
    B10-AC    | 0.2   0.6    0.1   0.1   |  4..34  | 
    B10-AC    | 0.2   0.6    0.1   0.1   |  4..100 | ciciot max_bits=100
    B10-CE    | 0.6   0.2    0.1   0.1   |  4..34  | 
    B10-CE    | 0.6   0.2    0.1   0.1   |  4..100 | ciciot max_bits=100
    B15-AC    | 0.175 0.525  0.15  0.15  |  4..34  | 
    B15-AC    | 0.175 0.525  0.15  0.15  |  4..100 | ciciot max_bits=100
    B15-CE    | 0.525 0.175  0.15  0.15  |  4..34  | 
    B15-CE    | 0.525 0.175  0.15  0.15  |  4..100 | ciciot max_bits=100
    CE20      | 0.2   0.1    0.3   0.4   |  4..34  | 
    CE20      | 0.2   0.1    0.3   0.4   |  4..100 | ciciot max_bits=100
    Wa-CTRL   | 0.35  0.3    0.3   0.05  |  4..34  | cicids production control
    Wc-CTRL   | 0.7   0.1    0.15  0.05  |  4..100 | ciciot production control
    Wb-CTRL   | 0.1   0.2    0.35  0.35  |  4..34  | unsw production control
    B34-CTRL  | 0.1   0.2    0.35  0.35  | 34..34  | FIXED bits=34 (architecture control, Wb weights)
```

## 1. cicids-quad-96b (control Wa-CTRL; all 24 runs same code era — max_bits=34, immune to OR-fold bug)

### cicids-quad — best_f1 genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  99.43±0.11 |   0.23±0.10 |  99.64±0.07 |  90.00±43.59 | 03p 04p 05p
    B05-CE    | 3 |  99.42±0.10 |   0.25±0.10 |  99.64±0.06 |  93.33±5.77 | 03p 04p 05p
    B10-AC    | 3 |  99.47±0.07 |   0.21±0.08 |  99.66±0.04 |  80.00±17.32 | 03p 04p 05p
    B10-CE    | 3 |  99.45±0.06 |   0.17±0.11 |  99.65±0.04 |  83.33±20.82 | 03p 04p 05p
    B15-AC    | 3 |  99.48±0.06 |   0.15±0.05 |  99.67±0.04 | 106.67±25.17 | 03p 04p 05p
    B15-CE    | 3 |  99.44±0.08 |   0.23±0.07 |  99.64±0.05 |  73.33±15.28 | 03p 04p 05p
    CE20      | 3 |  99.52±0.03 |   0.12±0.01 |  99.70±0.02 |  80.00±10.00 | 03p 04p 05p
    Wa-CTRL   | 3 |  99.47±0.09 |   0.21±0.08 |  99.67±0.06 | 116.67±15.28 | 03p 04p 05p
```

### cicids-quad — best_fpr genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  99.39±0.07 |   0.26±0.07 |  99.61±0.05 |  90.00±43.59 | 03p 04p 05p
    B05-CE    | 3 |  99.41±0.08 |   0.24±0.11 |  99.63±0.05 |  93.33±5.77 | 03p 04p 05p
    B10-AC    | 3 |  99.45±0.07 |   0.19±0.09 |  99.65±0.04 |  80.00±17.32 | 03p 04p 05p
    B10-CE    | 3 |  99.41±0.09 |   0.19±0.08 |  99.62±0.06 |  83.33±20.82 | 03p 04p 05p
    B15-AC    | 3 |  99.45±0.03 |   0.17±0.07 |  99.65±0.02 | 106.67±25.17 | 03p 04p 05p
    B15-CE    | 3 |  99.39±0.12 |   0.21±0.07 |  99.61±0.07 |  73.33±15.28 | 03p 04p 05p
    CE20      | 3 |  99.48±0.02 |   0.11±0.01 |  99.67±0.01 |  80.00±10.00 | 03p 04p 05p
    Wa-CTRL   | 3 |  99.47±0.09 |   0.18±0.07 |  99.67±0.06 | 116.67±15.28 | 03p 04p 05p
```

### cicids-quad — best_fitness genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  99.42±0.11 |   0.25±0.11 |  99.63±0.07 |  90.00±43.59 | 03p 04p 05p
    B05-CE    | 3 |  99.41±0.08 |   0.23±0.09 |  99.63±0.05 |  93.33±5.77 | 03p 04p 05p
    B10-AC    | 3 |  99.45±0.08 |   0.21±0.10 |  99.65±0.05 |  80.00±17.32 | 03p 04p 05p
    B10-CE    | 3 |  99.43±0.03 |   0.17±0.09 |  99.64±0.02 |  83.33±20.82 | 03p 04p 05p
    B15-AC    | 3 |  99.47±0.05 |   0.16±0.05 |  99.67±0.03 | 106.67±25.17 | 03p 04p 05p
    B15-CE    | 3 |  99.41±0.11 |   0.23±0.08 |  99.62±0.07 |  73.33±15.28 | 03p 04p 05p
    CE20      | 3 |  99.51±0.03 |   0.12±0.00 |  99.69±0.02 |  80.00±10.00 | 03p 04p 05p
    Wa-CTRL   | 3 |  99.48±0.09 |   0.18±0.07 |  99.67±0.06 | 116.67±15.28 | 03p 04p 05p
```

### cicids-quad — paired-by-seed delta vs Wa-CTRL (arm − control, pp), 95% t-CI df=n−1, HELD-OUT val_cal
```
    Arm       | genome       | n | dF1  mean±SD   [95% CI]          | ex0 | dFPR mean±SD   [95% CI]          | ex0 | dAcc mean±SD   [95% CI]          | ex0
    ----------+--------------+---+----------------------------------+-----+----------------------------------+-----+----------------------------------+----
    B05-AC    | best_f1      | 3 | -0.044±0.146 [ -0.408, +0.319] | no  | +0.024±0.058 [ -0.121, +0.168] | no  | -0.028±0.092 [ -0.258, +0.202] | no 
    B05-AC    | best_fpr     | 3 | -0.086±0.107 [ -0.352, +0.181] | no  | +0.081±0.007 [ +0.063, +0.098] | YES | -0.055±0.067 [ -0.222, +0.113] | no 
    B05-AC    | best_fitness | 3 | -0.058±0.141 [ -0.408, +0.291] | no  | +0.064±0.057 [ -0.077, +0.205] | no  | -0.037±0.089 [ -0.258, +0.184] | no 
    B05-CE    | best_f1      | 3 | -0.049±0.120 [ -0.348, +0.249] | no  | +0.041±0.075 [ -0.146, +0.228] | no  | -0.031±0.076 [ -0.221, +0.158] | no 
    B05-CE    | best_fpr     | 3 | -0.064±0.095 [ -0.301, +0.173] | no  | +0.062±0.053 [ -0.069, +0.193] | no  | -0.041±0.060 [ -0.191, +0.109] | no 
    B05-CE    | best_fitness | 3 | -0.068±0.093 [ -0.300, +0.163] | no  | +0.048±0.023 [ -0.009, +0.105] | no  | -0.043±0.059 [ -0.190, +0.103] | no 
    B10-AC    | best_f1      | 3 | -0.005±0.038 [ -0.099, +0.090] | no  | -0.002±0.024 [ -0.062, +0.058] | no  | -0.003±0.024 [ -0.062, +0.056] | no 
    B10-AC    | best_fpr     | 3 | -0.027±0.057 [ -0.168, +0.114] | no  | +0.005±0.055 [ -0.132, +0.143] | no  | -0.017±0.035 [ -0.104, +0.070] | no 
    B10-AC    | best_fitness | 3 | -0.029±0.053 [ -0.160, +0.103] | no  | +0.023±0.060 [ -0.127, +0.174] | no  | -0.018±0.033 [ -0.099, +0.063] | no 
    B10-CE    | best_f1      | 3 | -0.021±0.074 [ -0.205, +0.163] | no  | -0.039±0.060 [ -0.189, +0.112] | no  | -0.013±0.046 [ -0.127, +0.101] | no 
    B10-CE    | best_fpr     | 3 | -0.067±0.126 [ -0.381, +0.247] | no  | +0.010±0.011 [ -0.016, +0.037] | no  | -0.042±0.079 [ -0.239, +0.155] | no 
    B10-CE    | best_fitness | 3 | -0.054±0.067 [ -0.219, +0.112] | no  | -0.010±0.069 [ -0.180, +0.160] | no  | -0.034±0.041 [ -0.136, +0.069] | no 
    B15-AC    | best_f1      | 3 | +0.007±0.140 [ -0.340, +0.355] | no  | -0.054±0.135 [ -0.389, +0.281] | no  | +0.005±0.089 [ -0.216, +0.227] | no 
    B15-AC    | best_fpr     | 3 | -0.026±0.111 [ -0.302, +0.249] | no  | -0.011±0.133 [ -0.341, +0.320] | no  | -0.016±0.071 [ -0.193, +0.160] | no 
    B15-AC    | best_fitness | 3 | -0.012±0.136 [ -0.350, +0.326] | no  | -0.026±0.126 [ -0.338, +0.285] | no  | -0.007±0.087 [ -0.223, +0.208] | no 
    B15-CE    | best_f1      | 3 | -0.039±0.102 [ -0.292, +0.214] | no  | +0.024±0.078 [ -0.170, +0.218] | no  | -0.025±0.065 [ -0.186, +0.136] | no 
    B15-CE    | best_fpr     | 3 | -0.087±0.062 [ -0.240, +0.066] | no  | +0.026±0.024 [ -0.034, +0.086] | no  | -0.055±0.039 [ -0.151, +0.042] | no 
    B15-CE    | best_fitness | 3 | -0.075±0.086 [ -0.289, +0.138] | no  | +0.049±0.055 [ -0.089, +0.187] | no  | -0.048±0.055 [ -0.183, +0.088] | no 
    CE20      | best_f1      | 3 | +0.045±0.088 [ -0.174, +0.265] | no  | -0.090±0.077 [ -0.281, +0.102] | no  | +0.029±0.056 [ -0.110, +0.169] | no 
    CE20      | best_fpr     | 3 | +0.006±0.070 [ -0.167, +0.179] | no  | -0.069±0.068 [ -0.238, +0.100] | no  | +0.004±0.044 [ -0.106, +0.115] | no 
    CE20      | best_fitness | 3 | +0.027±0.080 [ -0.171, +0.225] | no  | -0.060±0.077 [ -0.251, +0.130] | no  | +0.017±0.050 [ -0.108, +0.143] | no 
```

    Sign agreement best_f1 vs best_fpr vs best_fitness on dF1 / dFPR vs Wa-CTRL (+ = arm higher):
```
    Arm       | dF1 sign (f1/fpr/fit) | dFPR sign (f1/fpr/fit) | Pareto read
    ----------+-----------------------+------------------------+------------------------------------------
    B05-AC    |          −−−          |          +++           | dominated by control (F1 down, FPR up)
    B05-CE    |          −−−          |          +++           | dominated by control (F1 down, FPR up)
    B10-AC    |          −−−          |          −++           | genome types DISAGREE on sign — no direction
    B10-CE    |          −−−          |          −+−           | genome types DISAGREE on sign — no direction
    B15-AC    |          +−−          |          −−−           | genome types DISAGREE on sign — no direction
    B15-CE    |          −−−          |          +++           | dominated by control (F1 down, FPR up)
    CE20      |          +++          |          −−−           | dominates control (F1 up, FPR down) — point estimates only
```

### cicids-quad — pairwise paired dF1 (row − col, pp), best_f1 val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wa-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | +0.005±0.028    3 | -0.040±0.109    3 | -0.023±0.074    3 | -0.052±0.149    3 | -0.005±0.045    3 | -0.090±0.078    3 | -0.044±0.146    3
    B05-CE    | -0.005±0.028    3 |                     | -0.045±0.083    3 | -0.028±0.050    3 | -0.057±0.146    3 | -0.011±0.018    3 | -0.095±0.063    3 | -0.049±0.120    3
    B10-AC    | +0.040±0.109    3 | +0.045±0.083    3 |                     | +0.016±0.036    3 | -0.012±0.123    3 | +0.034±0.065    3 | -0.050±0.054    3 | -0.005±0.038    3
    B10-CE    | +0.023±0.074    3 | +0.028±0.050    3 | -0.016±0.036    3 |                     | -0.028±0.115    3 | +0.018±0.033    3 | -0.066±0.030 SEP3 | -0.021±0.074    3
    B15-AC    | +0.052±0.149    3 | +0.057±0.146    3 | +0.012±0.123    3 | +0.028±0.115    3 |                     | +0.046±0.138    3 | -0.038±0.087    3 | +0.007±0.140    3
    B15-CE    | +0.005±0.045    3 | +0.011±0.018    3 | -0.034±0.065    3 | -0.018±0.033    3 | -0.046±0.138    3 |                     | -0.084±0.052    3 | -0.039±0.102    3
    CE20      | +0.090±0.078    3 | +0.095±0.063    3 | +0.050±0.054    3 | +0.066±0.030 SEP3 | +0.038±0.087    3 | +0.084±0.052    3 |                     | +0.045±0.088    3
    Wa-CTRL   | +0.044±0.146    3 | +0.049±0.120    3 | +0.005±0.038    3 | +0.021±0.074    3 | -0.007±0.140    3 | +0.039±0.102    3 | -0.045±0.088    3 |                    
```

    Arms whose paired dF1 vs Wa-CTRL CI EXCLUDES 0 upward: NONE
    Arms whose paired dF1 vs Wa-CTRL CI EXCLUDES 0 downward: NONE
    Top arm by mean F1 (best_f1): CE20 (99.519); NOT separated from: ['B05-AC', 'B05-CE', 'B10-AC', 'B15-AC', 'B15-CE', 'Wa-CTRL']

### cicids-quad — pairwise paired dF1 (row − col, pp), best_fitness val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wa-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | +0.010±0.047    3 | -0.030±0.089    3 | -0.005±0.093    3 | -0.046±0.152    3 | +0.017±0.075    3 | -0.085±0.083    3 | -0.058±0.141    3
    B05-CE    | -0.010±0.047    3 |                     | -0.040±0.042    3 | -0.015±0.053    3 | -0.057±0.132    3 | +0.007±0.041    3 | -0.095±0.048    3 | -0.068±0.093    3
    B10-AC    | +0.030±0.089    3 | +0.040±0.042    3 |                     | +0.025±0.046    3 | -0.017±0.135    3 | +0.047±0.038    3 | -0.055±0.053    3 | -0.029±0.053    3
    B10-CE    | +0.005±0.093    3 | +0.015±0.053    3 | -0.025±0.046    3 |                     | -0.042±0.088    3 | +0.022±0.078    3 | -0.081±0.013 SEP3 | -0.054±0.067    3
    B15-AC    | +0.046±0.152    3 | +0.057±0.132    3 | +0.017±0.135    3 | +0.042±0.088    3 |                     | +0.063±0.165    3 | -0.039±0.085    3 | -0.012±0.136    3
    B15-CE    | -0.017±0.075    3 | -0.007±0.041    3 | -0.047±0.038    3 | -0.022±0.078    3 | -0.063±0.165    3 |                     | -0.102±0.080    3 | -0.075±0.086    3
    CE20      | +0.085±0.083    3 | +0.095±0.048    3 | +0.055±0.053    3 | +0.081±0.013 SEP3 | +0.039±0.085    3 | +0.102±0.080    3 |                     | +0.027±0.080    3
    Wa-CTRL   | +0.058±0.141    3 | +0.068±0.093    3 | +0.029±0.053    3 | +0.054±0.067    3 | +0.012±0.136    3 | +0.075±0.086    3 | -0.027±0.080    3 |                    
```
### cicids-quad — paired SD_d (vs Wa-CTRL, best_f1 val_cal F1) and MDD at 80% power, two-sided α=0.05, paired t
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10 | n needed for |gap|
    ----------+---+-------------+--------+---------+---------+---------+----------+-------------------
    B05-AC    | 3 |      -0.044 |  0.146 |   0.478 |   0.246 |   0.186 |    0.146 | 88
    B05-CE    | 3 |      -0.049 |  0.120 |   0.392 |   0.202 |   0.153 |    0.120 | 49
    B10-AC    | 3 |      -0.005 |  0.038 |   0.124 |   0.064 |   0.048 |    0.038 | >200
    B10-CE    | 3 |      -0.021 |  0.074 |   0.241 |   0.124 |   0.094 |    0.074 | 100
    B15-AC    | 3 |      +0.007 |  0.140 |   0.457 |   0.235 |   0.178 |    0.139 | >200
    B15-CE    | 3 |      -0.039 |  0.102 |   0.333 |   0.172 |   0.130 |    0.102 | 57
    CE20      | 3 |      +0.045 |  0.088 |   0.288 |   0.149 |   0.112 |    0.088 | 32
    pooled    |   |             |  0.107 |   0.350 |   0.180 |   0.136 |    0.107 | (RMS of paired SDs)
```
    FPR version (paired SD_d of dFPR vs Wa-CTRL, best_fpr genome val_cal):
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10
    ----------+---+-------------+--------+---------+---------+---------+---------
    B05-AC    | 3 |      +0.081 |  0.007 |   0.023 |   0.012 |   0.009 |    0.007
    B05-CE    | 3 |      +0.062 |  0.053 |   0.172 |   0.089 |   0.067 |    0.052
    B10-AC    | 3 |      +0.005 |  0.055 |   0.180 |   0.093 |   0.070 |    0.055
    B10-CE    | 3 |      +0.010 |  0.011 |   0.035 |   0.018 |   0.013 |    0.011
    B15-AC    | 3 |      -0.011 |  0.133 |   0.435 |   0.224 |   0.169 |    0.133
    B15-CE    | 3 |      +0.026 |  0.024 |   0.079 |   0.041 |   0.031 |    0.024
    CE20      | 3 |      -0.069 |  0.068 |   0.222 |   0.115 |   0.087 |    0.068
```
    Within-arm SD of F1 (best_f1 val_cal), per arm and pooled:
```
    B05-AC    | n=3 | SD= 0.107 | values:  99.547  99.404  99.338
    B05-CE    | n=3 | SD= 0.096 | values:  99.535  99.374  99.365
    B10-AC    | n=3 | SD= 0.069 | values:  99.515  99.390  99.503
    B10-CE    | n=3 | SD= 0.059 | values:  99.515  99.398  99.445
    B15-AC    | n=3 | SD= 0.057 | values:  99.426  99.539  99.479
    B15-CE    | n=3 | SD= 0.084 | values:  99.532  99.377  99.396
    CE20      | n=3 | SD= 0.033 | values:  99.557  99.498  99.503
    Wa-CTRL   | n=3 | SD= 0.093 | values:  99.500  99.370  99.551
    POOLED within-arm SD (RMS) = 0.078 pp ; median = 0.077
```

## 2. ciciot-quad-96b — ALL completed runs (control Wc-CTRL) — era-MIXED, see §5

### ciciot-quad — best_f1 genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  92.96±0.18 |   7.91±0.51 |  96.48±0.08 | 150.00±26.46 | 03p 04p 05p
    B05-CE    | 3 |  92.80±0.10 |   7.83±0.15 |  96.39±0.05 | 136.67±15.28 | 03p 04p 05p
    B10-AC    | 3 |  92.97±0.05 |   7.14±1.19 |  96.46±0.02 | 146.67±5.77 | 03p 04p 05P
    B10-CE    | 3 |  92.91±0.06 |   6.75±0.35 |  96.42±0.04 | 156.67±25.17 | 03P 04P 05P
    B15-AC    | 3 |  92.98±0.08 |   6.79±0.76 |  96.46±0.07 | 150.00±26.46 | 03P 04P 05P
    B15-CE    | 3 |  92.70±0.14 |   7.52±1.02 |  96.33±0.05 | 136.67±5.77 | 03P 04P 05P
    CE20      | 3 |  92.71±0.27 |   7.90±1.42 |  96.34±0.11 | 136.67±30.55 | 03P 04P 05P
    Wc-CTRL   | 3 |  92.85±0.05 |   7.93±0.13 |  96.42±0.03 | 140.00±10.00 | 03P 04P 05P
```

### ciciot-quad — best_fpr genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  92.96±0.16 |   7.92±0.40 |  96.48±0.07 | 150.00±26.46 | 03p 04p 05p
    B05-CE    | 3 |  92.79±0.10 |   7.94±0.81 |  96.39±0.05 | 136.67±15.28 | 03p 04p 05p
    B10-AC    | 3 |  92.96±0.05 |   7.19±1.18 |  96.46±0.02 | 146.67±5.77 | 03p 04p 05P
    B10-CE    | 3 |  92.90±0.07 |   7.03±1.03 |  96.42±0.06 | 156.67±25.17 | 03P 04P 05P
    B15-AC    | 3 |  92.96±0.07 |   7.88±0.47 |  96.48±0.05 | 150.00±26.46 | 03P 04P 05P
    B15-CE    | 3 |  92.71±0.17 |   7.72±0.86 |  96.33±0.07 | 136.67±5.77 | 03P 04P 05P
    CE20      | 3 |  92.68±0.28 |   7.80±1.00 |  96.32±0.13 | 136.67±30.55 | 03P 04P 05P
    Wc-CTRL   | 3 |  92.87±0.02 |   7.73±0.23 |  96.42±0.02 | 140.00±10.00 | 03P 04P 05P
```

### ciciot-quad — best_fitness genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  92.94±0.16 |   8.06±0.48 |  96.47±0.08 | 150.00±26.46 | 03p 04p 05p
    B05-CE    | 3 |  92.80±0.12 |   7.80±0.56 |  96.39±0.07 | 136.67±15.28 | 03p 04p 05p
    B10-AC    | 3 |  92.96±0.06 |   6.96±1.07 |  96.45±0.01 | 146.67±5.77 | 03p 04p 05P
    B10-CE    | 3 |  92.92±0.08 |   6.98±0.87 |  96.43±0.06 | 156.67±25.17 | 03P 04P 05P
    B15-AC    | 3 |  92.99±0.05 |   7.49±0.85 |  96.48±0.04 | 150.00±26.46 | 03P 04P 05P
    B15-CE    | 3 |  92.73±0.11 |   7.06±0.79 |  96.33±0.04 | 136.67±5.77 | 03P 04P 05P
    CE20      | 3 |  92.70±0.29 |   7.85±1.34 |  96.33±0.12 | 136.67±30.55 | 03P 04P 05P
    Wc-CTRL   | 3 |  92.87±0.02 |   7.97±0.26 |  96.43±0.02 | 140.00±10.00 | 03P 04P 05P
```

### ciciot-quad — paired-by-seed delta vs Wc-CTRL (arm − control, pp), 95% t-CI df=n−1, HELD-OUT val_cal
```
    Arm       | genome       | n | dF1  mean±SD   [95% CI]          | ex0 | dFPR mean±SD   [95% CI]          | ex0 | dAcc mean±SD   [95% CI]          | ex0
    ----------+--------------+---+----------------------------------+-----+----------------------------------+-----+----------------------------------+----
    B05-AC    | best_f1      | 3 | +0.111±0.137 [ -0.228, +0.450] | no  | -0.028±0.524 [ -1.331, +1.274] | no  | +0.061±0.068 [ -0.107, +0.229] | no 
    B05-AC    | best_fpr     | 3 | +0.092±0.139 [ -0.253, +0.438] | no  | +0.192±0.558 [ -1.195, +1.578] | no  | +0.057±0.066 [ -0.108, +0.223] | no 
    B05-AC    | best_fitness | 3 | +0.069±0.153 [ -0.310, +0.447] | no  | +0.095±0.274 [ -0.586, +0.776] | no  | +0.041±0.084 [ -0.168, +0.250] | no 
    B05-CE    | best_f1      | 3 | -0.051±0.060 [ -0.201, +0.098] | no  | -0.105±0.226 [ -0.666, +0.456] | no  | -0.032±0.034 [ -0.117, +0.054] | no 
    B05-CE    | best_fpr     | 3 | -0.077±0.088 [ -0.295, +0.141] | no  | +0.210±1.034 [ -2.358, +2.778] | no  | -0.036±0.053 [ -0.168, +0.096] | no 
    B05-CE    | best_fitness | 3 | -0.063±0.110 [ -0.336, +0.210] | no  | -0.167±0.746 [ -2.020, +1.686] | no  | -0.040±0.072 [ -0.219, +0.138] | no 
    B10-AC    | best_f1      | 3 | +0.117±0.017 [ +0.073, +0.160] | YES | -0.795±1.078 [ -3.474, +1.884] | no  | +0.041±0.026 [ -0.025, +0.106] | no 
    B10-AC    | best_fpr     | 3 | +0.098±0.029 [ +0.026, +0.170] | YES | -0.543±1.129 [ -3.347, +2.261] | no  | +0.038±0.018 [ -0.008, +0.084] | no 
    B10-AC    | best_fitness | 3 | +0.093±0.042 [ -0.011, +0.196] | no  | -1.008±0.852 [ -3.124, +1.107] | no  | +0.021±0.004 [ +0.010, +0.031] | YES
    B10-CE    | best_f1      | 3 | +0.059±0.028 [ -0.010, +0.127] | no  | -1.178±0.244 [ -1.784, -0.572] | YES | -0.004±0.013 [ -0.036, +0.028] | no 
    B10-CE    | best_fpr     | 3 | +0.039±0.053 [ -0.093, +0.171] | no  | -0.698±0.804 [ -2.696, +1.300] | no  | +0.000±0.038 [ -0.095, +0.095] | no 
    B10-CE    | best_fitness | 3 | +0.054±0.062 [ -0.099, +0.207] | no  | -0.988±0.718 [ -2.773, +0.796] | no  | -0.000±0.046 [ -0.115, +0.114] | no 
    B15-AC    | best_f1      | 3 | +0.131±0.120 [ -0.168, +0.430] | no  | -1.140±0.830 [ -3.201, +0.921] | no  | +0.038±0.092 [ -0.192, +0.267] | no 
    B15-AC    | best_fpr     | 3 | +0.090±0.084 [ -0.118, +0.298] | no  | +0.147±0.683 [ -1.550, +1.843] | no  | +0.055±0.066 [ -0.110, +0.219] | no 
    B15-AC    | best_fitness | 3 | +0.123±0.063 [ -0.034, +0.281] | no  | -0.473±0.916 [ -2.749, +1.803] | no  | +0.054±0.061 [ -0.097, +0.205] | no 
    B15-CE    | best_f1      | 3 | -0.147±0.186 [ -0.608, +0.315] | no  | -0.410±1.067 [ -3.060, +2.240] | no  | -0.095±0.073 [ -0.276, +0.086] | no 
    B15-CE    | best_fpr     | 3 | -0.157±0.194 [ -0.638, +0.324] | no  | -0.007±0.802 [ -1.999, +1.986] | no  | -0.088±0.088 [ -0.306, +0.130] | no 
    B15-CE    | best_fitness | 3 | -0.133±0.134 [ -0.466, +0.199] | no  | -0.910±0.686 [ -2.614, +0.794] | no  | -0.103±0.055 [ -0.239, +0.032] | no 
    CE20      | best_f1      | 3 | -0.144±0.298 [ -0.884, +0.596] | no  | -0.037±1.515 [ -3.800, +3.727] | no  | -0.081±0.117 [ -0.372, +0.211] | no 
    CE20      | best_fpr     | 3 | -0.184±0.280 [ -0.879, +0.511] | no  | +0.072±1.022 [ -2.467, +2.610] | no  | -0.100±0.124 [ -0.408, +0.208] | no 
    CE20      | best_fitness | 3 | -0.167±0.289 [ -0.884, +0.551] | no  | -0.117±1.579 [ -4.038, +3.805] | no  | -0.096±0.111 [ -0.372, +0.179] | no 
```

    Sign agreement best_f1 vs best_fpr vs best_fitness on dF1 / dFPR vs Wc-CTRL (+ = arm higher):
```
    Arm       | dF1 sign (f1/fpr/fit) | dFPR sign (f1/fpr/fit) | Pareto read
    ----------+-----------------------+------------------------+------------------------------------------
    B05-AC    |          +++          |          −++           | genome types DISAGREE on sign — no direction
    B05-CE    |          −−−          |          −+−           | genome types DISAGREE on sign — no direction
    B10-AC    |          +++          |          −−−           | dominates control (F1 up, FPR down) — point estimates only
    B10-CE    |          +++          |          −−−           | dominates control (F1 up, FPR down) — point estimates only
    B15-AC    |          +++          |          −+−           | genome types DISAGREE on sign — no direction
    B15-CE    |          −−−          |          −−−           | Pareto MOVE: F1 down but FPR down
    CE20      |          −−−          |          −+−           | genome types DISAGREE on sign — no direction
```

### ciciot-quad — pairwise paired dF1 (row − col, pp), best_f1 val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wc-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | +0.162±0.076 SEP3 | -0.006±0.128    3 | +0.052±0.114    3 | -0.020±0.214    3 | +0.258±0.307    3 | +0.255±0.431    3 | +0.111±0.137    3
    B05-CE    | -0.162±0.076 SEP3 |                     | -0.168±0.052 SEP3 | -0.110±0.039 SEP3 | -0.182±0.153    3 | +0.095±0.237    3 | +0.093±0.356    3 | -0.051±0.060    3
    B10-AC    | +0.006±0.128    3 | +0.168±0.052 SEP3 |                     | +0.058±0.014 SEP3 | -0.014±0.112    3 | +0.263±0.186    3 | +0.261±0.311    3 | +0.117±0.017 SEP3
    B10-CE    | -0.052±0.114    3 | +0.110±0.039 SEP3 | -0.058±0.014 SEP3 |                     | -0.072±0.120    3 | +0.205±0.198    3 | +0.203±0.324    3 | +0.059±0.028 SEP3
    B15-AC    | +0.020±0.214    3 | +0.182±0.153    3 | +0.014±0.112    3 | +0.072±0.120    3 |                     | +0.278±0.105 SEP3 | +0.275±0.305    3 | +0.131±0.120    3
    B15-CE    | -0.258±0.307    3 | -0.095±0.237    3 | -0.263±0.186    3 | -0.205±0.198    3 | -0.278±0.105 SEP3 |                     | -0.003±0.227    3 | -0.147±0.186    3
    CE20      | -0.255±0.431    3 | -0.093±0.356    3 | -0.261±0.311    3 | -0.203±0.324    3 | -0.275±0.305    3 | +0.003±0.227    3 |                     | -0.144±0.298    3
    Wc-CTRL   | -0.111±0.137    3 | +0.051±0.060    3 | -0.117±0.017 SEP3 | -0.059±0.028 SEP3 | -0.131±0.120    3 | +0.147±0.186    3 | +0.144±0.298    3 |                    
```

    Arms whose paired dF1 vs Wc-CTRL CI EXCLUDES 0 upward: ['B10-AC']
    Arms whose paired dF1 vs Wc-CTRL CI EXCLUDES 0 downward: NONE
    Top arm by mean F1 (best_f1): B15-AC (92.983); NOT separated from: ['B05-AC', 'B05-CE', 'B10-AC', 'B10-CE', 'CE20', 'Wc-CTRL']

### ciciot-quad — pairwise paired dF1 (row − col, pp), best_fitness val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wc-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | +0.131±0.045 SEP3 | -0.024±0.111    3 | +0.014±0.094    3 | -0.055±0.130    3 | +0.202±0.246    3 | +0.235±0.439    3 | +0.069±0.153    3
    B05-CE    | -0.131±0.045 SEP3 |                     | -0.155±0.068 SEP3 | -0.117±0.049 SEP3 | -0.186±0.101    3 | +0.071±0.217    3 | +0.104±0.395    3 | -0.063±0.110    3
    B10-AC    | +0.024±0.111    3 | +0.155±0.068 SEP3 |                     | +0.038±0.021    3 | -0.031±0.058    3 | +0.226±0.161    3 | +0.259±0.329    3 | +0.093±0.042 SEP3
    B10-CE    | -0.014±0.094    3 | +0.117±0.049 SEP3 | -0.038±0.021    3 |                     | -0.069±0.073    3 | +0.188±0.181    3 | +0.221±0.345    3 | +0.054±0.062    3
    B15-AC    | +0.055±0.130    3 | +0.186±0.101    3 | +0.031±0.058    3 | +0.069±0.073    3 |                     | +0.257±0.117 SEP3 | +0.290±0.336    3 | +0.123±0.063    3
    B15-CE    | -0.202±0.246    3 | -0.071±0.217    3 | -0.226±0.161    3 | -0.188±0.181    3 | -0.257±0.117 SEP3 |                     | +0.033±0.281    3 | -0.133±0.134    3
    CE20      | -0.235±0.439    3 | -0.104±0.395    3 | -0.259±0.329    3 | -0.221±0.345    3 | -0.290±0.336    3 | -0.033±0.281    3 |                     | -0.167±0.289    3
    Wc-CTRL   | -0.069±0.153    3 | +0.063±0.110    3 | -0.093±0.042 SEP3 | -0.054±0.062    3 | -0.123±0.063    3 | +0.133±0.134    3 | +0.167±0.289    3 |                    
```
### ciciot-quad — paired SD_d (vs Wc-CTRL, best_f1 val_cal F1) and MDD at 80% power, two-sided α=0.05, paired t
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10 | n needed for |gap|
    ----------+---+-------------+--------+---------+---------+---------+----------+-------------------
    B05-AC    | 3 |      +0.111 |  0.137 |   0.446 |   0.230 |   0.174 |    0.136 | 14
    B05-CE    | 3 |      -0.051 |  0.060 |   0.196 |   0.101 |   0.076 |    0.060 | 13
    B10-AC    | 3 |      +0.117 |  0.017 |   0.057 |   0.029 |   0.022 |    0.017 | 3
    B10-CE    | 3 |      +0.059 |  0.028 |   0.090 |   0.047 |   0.035 |    0.028 | 5
    B15-AC    | 3 |      +0.131 |  0.120 |   0.393 |   0.203 |   0.153 |    0.120 | 9
    B15-CE    | 3 |      -0.147 |  0.186 |   0.606 |   0.312 |   0.236 |    0.185 | 15
    CE20      | 3 |      -0.144 |  0.298 |   0.972 |   0.501 |   0.379 |    0.297 | 36
    pooled    |   |             |  0.152 |   0.495 |   0.255 |   0.193 |    0.151 | (RMS of paired SDs)
```
    FPR version (paired SD_d of dFPR vs Wc-CTRL, best_fpr genome val_cal):
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10
    ----------+---+-------------+--------+---------+---------+---------+---------
    B05-AC    | 3 |      +0.192 |  0.558 |   1.822 |   0.939 |   0.710 |    0.556
    B05-CE    | 3 |      +0.210 |  1.034 |   3.374 |   1.739 |   1.315 |    1.029
    B10-AC    | 3 |      -0.543 |  1.129 |   3.684 |   1.899 |   1.436 |    1.124
    B10-CE    | 3 |      -0.698 |  0.804 |   2.625 |   1.353 |   1.023 |    0.801
    B15-AC    | 3 |      +0.147 |  0.683 |   2.230 |   1.149 |   0.869 |    0.680
    B15-CE    | 3 |      -0.007 |  0.802 |   2.618 |   1.349 |   1.021 |    0.799
    CE20      | 3 |      +0.072 |  1.022 |   3.335 |   1.719 |   1.300 |    1.018
```
    Within-arm SD of F1 (best_f1 val_cal), per arm and pooled:
```
    B05-AC    | n=3 | SD= 0.177 | values:  92.789  92.955  93.143
    B05-CE    | n=3 | SD= 0.102 | values:  92.692  92.812  92.896
    B10-AC    | n=3 | SD= 0.050 | values:  92.914  92.977  93.013
    B10-CE    | n=3 | SD= 0.064 | values:  92.844  92.916  92.971
    B15-AC    | n=3 | SD= 0.078 | values:  93.052  92.897  92.999
    B15-CE    | n=3 | SD= 0.136 | values:  92.860  92.651  92.604
    CE20      | n=3 | SD= 0.273 | values:  92.836  92.894  92.394
    Wc-CTRL   | n=3 | SD= 0.050 | values:  92.794  92.880  92.881
    POOLED within-arm SD (RMS) = 0.137 pp ; median = 0.090
```

## 5. ciciot-quad-96b — POST-FIX arms only (started ≥ 30/08/2026 02:27 UTC), control Wc-CTRL

### ciciot-quad — best_f1 genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B10-CE    | 3 |  92.91±0.06 |   6.75±0.35 |  96.42±0.04 | 156.67±25.17 | 03P 04P 05P
    B15-AC    | 3 |  92.98±0.08 |   6.79±0.76 |  96.46±0.07 | 150.00±26.46 | 03P 04P 05P
    B15-CE    | 3 |  92.70±0.14 |   7.52±1.02 |  96.33±0.05 | 136.67±5.77 | 03P 04P 05P
    CE20      | 3 |  92.71±0.27 |   7.90±1.42 |  96.34±0.11 | 136.67±30.55 | 03P 04P 05P
    Wc-CTRL   | 3 |  92.85±0.05 |   7.93±0.13 |  96.42±0.03 | 140.00±10.00 | 03P 04P 05P
```

### ciciot-quad — best_fpr genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B10-CE    | 3 |  92.90±0.07 |   7.03±1.03 |  96.42±0.06 | 156.67±25.17 | 03P 04P 05P
    B15-AC    | 3 |  92.96±0.07 |   7.88±0.47 |  96.48±0.05 | 150.00±26.46 | 03P 04P 05P
    B15-CE    | 3 |  92.71±0.17 |   7.72±0.86 |  96.33±0.07 | 136.67±5.77 | 03P 04P 05P
    CE20      | 3 |  92.68±0.28 |   7.80±1.00 |  96.32±0.13 | 136.67±30.55 | 03P 04P 05P
    Wc-CTRL   | 3 |  92.87±0.02 |   7.73±0.23 |  96.42±0.02 | 140.00±10.00 | 03P 04P 05P
```

### ciciot-quad — best_fitness genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B10-CE    | 3 |  92.92±0.08 |   6.98±0.87 |  96.43±0.06 | 156.67±25.17 | 03P 04P 05P
    B15-AC    | 3 |  92.99±0.05 |   7.49±0.85 |  96.48±0.04 | 150.00±26.46 | 03P 04P 05P
    B15-CE    | 3 |  92.73±0.11 |   7.06±0.79 |  96.33±0.04 | 136.67±5.77 | 03P 04P 05P
    CE20      | 3 |  92.70±0.29 |   7.85±1.34 |  96.33±0.12 | 136.67±30.55 | 03P 04P 05P
    Wc-CTRL   | 3 |  92.87±0.02 |   7.97±0.26 |  96.43±0.02 | 140.00±10.00 | 03P 04P 05P
```

### ciciot-quad — paired-by-seed delta vs Wc-CTRL (arm − control, pp), 95% t-CI df=n−1, HELD-OUT val_cal
```
    Arm       | genome       | n | dF1  mean±SD   [95% CI]          | ex0 | dFPR mean±SD   [95% CI]          | ex0 | dAcc mean±SD   [95% CI]          | ex0
    ----------+--------------+---+----------------------------------+-----+----------------------------------+-----+----------------------------------+----
    B10-CE    | best_f1      | 3 | +0.059±0.028 [ -0.010, +0.127] | no  | -1.178±0.244 [ -1.784, -0.572] | YES | -0.004±0.013 [ -0.036, +0.028] | no 
    B10-CE    | best_fpr     | 3 | +0.039±0.053 [ -0.093, +0.171] | no  | -0.698±0.804 [ -2.696, +1.300] | no  | +0.000±0.038 [ -0.095, +0.095] | no 
    B10-CE    | best_fitness | 3 | +0.054±0.062 [ -0.099, +0.207] | no  | -0.988±0.718 [ -2.773, +0.796] | no  | -0.000±0.046 [ -0.115, +0.114] | no 
    B15-AC    | best_f1      | 3 | +0.131±0.120 [ -0.168, +0.430] | no  | -1.140±0.830 [ -3.201, +0.921] | no  | +0.038±0.092 [ -0.192, +0.267] | no 
    B15-AC    | best_fpr     | 3 | +0.090±0.084 [ -0.118, +0.298] | no  | +0.147±0.683 [ -1.550, +1.843] | no  | +0.055±0.066 [ -0.110, +0.219] | no 
    B15-AC    | best_fitness | 3 | +0.123±0.063 [ -0.034, +0.281] | no  | -0.473±0.916 [ -2.749, +1.803] | no  | +0.054±0.061 [ -0.097, +0.205] | no 
    B15-CE    | best_f1      | 3 | -0.147±0.186 [ -0.608, +0.315] | no  | -0.410±1.067 [ -3.060, +2.240] | no  | -0.095±0.073 [ -0.276, +0.086] | no 
    B15-CE    | best_fpr     | 3 | -0.157±0.194 [ -0.638, +0.324] | no  | -0.007±0.802 [ -1.999, +1.986] | no  | -0.088±0.088 [ -0.306, +0.130] | no 
    B15-CE    | best_fitness | 3 | -0.133±0.134 [ -0.466, +0.199] | no  | -0.910±0.686 [ -2.614, +0.794] | no  | -0.103±0.055 [ -0.239, +0.032] | no 
    CE20      | best_f1      | 3 | -0.144±0.298 [ -0.884, +0.596] | no  | -0.037±1.515 [ -3.800, +3.727] | no  | -0.081±0.117 [ -0.372, +0.211] | no 
    CE20      | best_fpr     | 3 | -0.184±0.280 [ -0.879, +0.511] | no  | +0.072±1.022 [ -2.467, +2.610] | no  | -0.100±0.124 [ -0.408, +0.208] | no 
    CE20      | best_fitness | 3 | -0.167±0.289 [ -0.884, +0.551] | no  | -0.117±1.579 [ -4.038, +3.805] | no  | -0.096±0.111 [ -0.372, +0.179] | no 
```

    Sign agreement best_f1 vs best_fpr vs best_fitness on dF1 / dFPR vs Wc-CTRL (+ = arm higher):
```
    Arm       | dF1 sign (f1/fpr/fit) | dFPR sign (f1/fpr/fit) | Pareto read
    ----------+-----------------------+------------------------+------------------------------------------
    B10-CE    |          +++          |          −−−           | dominates control (F1 up, FPR down) — point estimates only
    B15-AC    |          +++          |          −+−           | genome types DISAGREE on sign — no direction
    B15-CE    |          −−−          |          −−−           | Pareto MOVE: F1 down but FPR down
    CE20      |          −−−          |          −+−           | genome types DISAGREE on sign — no direction
```

### ciciot-quad — pairwise paired dF1 (row − col, pp), best_f1 val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wc-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------
    B10-CE    |                     | -0.072±0.120    3 | +0.205±0.198    3 | +0.203±0.324    3 | +0.059±0.028 SEP3
    B15-AC    | +0.072±0.120    3 |                     | +0.278±0.105 SEP3 | +0.275±0.305    3 | +0.131±0.120    3
    B15-CE    | -0.205±0.198    3 | -0.278±0.105 SEP3 |                     | -0.003±0.227    3 | -0.147±0.186    3
    CE20      | -0.203±0.324    3 | -0.275±0.305    3 | +0.003±0.227    3 |                     | -0.144±0.298    3
    Wc-CTRL   | -0.059±0.028 SEP3 | -0.131±0.120    3 | +0.147±0.186    3 | +0.144±0.298    3 |                    
```

    Arms whose paired dF1 vs Wc-CTRL CI EXCLUDES 0 upward: NONE
    Arms whose paired dF1 vs Wc-CTRL CI EXCLUDES 0 downward: NONE
    Top arm by mean F1 (best_f1): B15-AC (92.983); NOT separated from: ['B10-CE', 'CE20', 'Wc-CTRL']

### ciciot-quad — pairwise paired dF1 (row − col, pp), best_fitness val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wc-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------
    B10-CE    |                     | -0.069±0.073    3 | +0.188±0.181    3 | +0.221±0.345    3 | +0.054±0.062    3
    B15-AC    | +0.069±0.073    3 |                     | +0.257±0.117 SEP3 | +0.290±0.336    3 | +0.123±0.063    3
    B15-CE    | -0.188±0.181    3 | -0.257±0.117 SEP3 |                     | +0.033±0.281    3 | -0.133±0.134    3
    CE20      | -0.221±0.345    3 | -0.290±0.336    3 | -0.033±0.281    3 |                     | -0.167±0.289    3
    Wc-CTRL   | -0.054±0.062    3 | -0.123±0.063    3 | +0.133±0.134    3 | +0.167±0.289    3 |                    
```
### ciciot-quad — paired SD_d (vs Wc-CTRL, best_f1 val_cal F1) and MDD at 80% power, two-sided α=0.05, paired t
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10 | n needed for |gap|
    ----------+---+-------------+--------+---------+---------+---------+----------+-------------------
    B10-CE    | 3 |      +0.059 |  0.028 |   0.090 |   0.047 |   0.035 |    0.028 | 5
    B15-AC    | 3 |      +0.131 |  0.120 |   0.393 |   0.203 |   0.153 |    0.120 | 9
    B15-CE    | 3 |      -0.147 |  0.186 |   0.606 |   0.312 |   0.236 |    0.185 | 15
    CE20      | 3 |      -0.144 |  0.298 |   0.972 |   0.501 |   0.379 |    0.297 | 36
    pooled    |   |             |  0.186 |   0.607 |   0.313 |   0.237 |    0.185 | (RMS of paired SDs)
```
    FPR version (paired SD_d of dFPR vs Wc-CTRL, best_fpr genome val_cal):
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10
    ----------+---+-------------+--------+---------+---------+---------+---------
    B10-CE    | 3 |      -0.698 |  0.804 |   2.625 |   1.353 |   1.023 |    0.801
    B15-AC    | 3 |      +0.147 |  0.683 |   2.230 |   1.149 |   0.869 |    0.680
    B15-CE    | 3 |      -0.007 |  0.802 |   2.618 |   1.349 |   1.021 |    0.799
    CE20      | 3 |      +0.072 |  1.022 |   3.335 |   1.719 |   1.300 |    1.018
```
    Within-arm SD of F1 (best_f1 val_cal), per arm and pooled:
```
    B10-CE    | n=3 | SD= 0.064 | values:  92.844  92.916  92.971
    B15-AC    | n=3 | SD= 0.078 | values:  93.052  92.897  92.999
    B15-CE    | n=3 | SD= 0.136 | values:  92.860  92.651  92.604
    CE20      | n=3 | SD= 0.273 | values:  92.836  92.894  92.394
    Wc-CTRL   | n=3 | SD= 0.050 | values:  92.794  92.880  92.881
    POOLED within-arm SD (RMS) = 0.146 pp ; median = 0.078
```

## 3. unswr-qsr-64b (control Wb-CTRL; also vs B34-CTRL fixed-bits control)

### unswr-qsr — best_f1 genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  94.34±0.12 |   0.59±0.06 |  99.14±0.03 |  93.33±5.77 | 03P 04P 05P
    B05-CE    | 3 |  94.32±0.01 |   0.53±0.09 |  99.15±0.02 |  93.33±20.82 | 03P 04P 05P
    B10-AC    | 3 |  94.29±0.07 |   0.59±0.08 |  99.14±0.01 |  86.67±5.77 | 03P 04P 05P
    B10-CE    | 3 |  94.31±0.05 |   0.58±0.07 |  99.14±0.02 |  96.67±40.41 | 03P 04P 05P
    B15-AC    | 3 |  94.35±0.08 |   0.61±0.05 |  99.14±0.01 |  80.00±20.00 | 03P 04P 05P
    B15-CE    | 3 |  94.36±0.13 |   0.65±0.04 |  99.13±0.01 |  70.00±10.00 | 03P 04P 05P
    CE20      | 3 |  94.25±0.07 |   0.59±0.10 |  99.13±0.02 |  63.33±5.77 | 03P 04P 05P
    Wb-CTRL   | 3 |  94.32±0.05 |   0.61±0.06 |  99.14±0.01 | 100.00±26.46 | 03P 04P 05P
    B34-CTRL  | 5 |  94.38±0.06 |   0.59±0.03 |  99.15±0.01 |  68.00±17.89 | 01P 02P 03P 04P 05P
```

### unswr-qsr — best_fpr genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  94.34±0.14 |   0.60±0.03 |  99.14±0.02 |  93.33±5.77 | 03P 04P 05P
    B05-CE    | 3 |  94.37±0.04 |   0.56±0.07 |  99.15±0.01 |  93.33±20.82 | 03P 04P 05P
    B10-AC    | 3 |  94.36±0.11 |   0.58±0.08 |  99.15±0.01 |  86.67±5.77 | 03P 04P 05P
    B10-CE    | 3 |  94.28±0.06 |   0.58±0.07 |  99.14±0.02 |  96.67±40.41 | 03P 04P 05P
    B15-AC    | 3 |  94.35±0.10 |   0.60±0.05 |  99.14±0.01 |  80.00±20.00 | 03P 04P 05P
    B15-CE    | 3 |  94.38±0.07 |   0.68±0.03 |  99.13±0.01 |  70.00±10.00 | 03P 04P 05P
    CE20      | 3 |  94.27±0.05 |   0.60±0.07 |  99.13±0.02 |  63.33±5.77 | 03P 04P 05P
    Wb-CTRL   | 3 |  94.30±0.06 |   0.61±0.04 |  99.13±0.01 | 100.00±26.46 | 03P 04P 05P
    B34-CTRL  | 5 |  94.33±0.12 |   0.65±0.06 |  99.13±0.02 |  68.00±17.89 | 01P 02P 03P 04P 05P
```

### unswr-qsr — best_fitness genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  94.30±0.17 |   0.59±0.04 |  99.14±0.03 |  93.33±5.77 | 03P 04P 05P
    B05-CE    | 3 |  94.36±0.03 |   0.56±0.09 |  99.15±0.01 |  93.33±20.82 | 03P 04P 05P
    B10-AC    | 3 |  94.33±0.11 |   0.56±0.06 |  99.14±0.01 |  86.67±5.77 | 03P 04P 05P
    B10-CE    | 3 |  94.29±0.09 |   0.58±0.07 |  99.14±0.01 |  96.67±40.41 | 03P 04P 05P
    B15-AC    | 3 |  94.33±0.07 |   0.62±0.07 |  99.13±0.01 |  80.00±20.00 | 03P 04P 05P
    B15-CE    | 3 |  94.37±0.14 |   0.59±0.02 |  99.15±0.02 |  70.00±10.00 | 03P 04P 05P
    CE20      | 3 |  94.26±0.05 |   0.60±0.10 |  99.13±0.02 |  63.33±5.77 | 03P 04P 05P
    Wb-CTRL   | 3 |  94.30±0.07 |   0.60±0.05 |  99.13±0.01 | 100.00±26.46 | 03P 04P 05P
    B34-CTRL  | 5 |  94.35±0.07 |   0.61±0.04 |  99.14±0.01 |  68.00±17.89 | 01P 02P 03P 04P 05P
```

### unswr-qsr — paired-by-seed delta vs Wb-CTRL (arm − control, pp), 95% t-CI df=n−1, HELD-OUT val_cal
```
    Arm       | genome       | n | dF1  mean±SD   [95% CI]          | ex0 | dFPR mean±SD   [95% CI]          | ex0 | dAcc mean±SD   [95% CI]          | ex0
    ----------+--------------+---+----------------------------------+-----+----------------------------------+-----+----------------------------------+----
    B05-AC    | best_f1      | 3 | +0.020±0.129 [ -0.300, +0.341] | no  | -0.020±0.051 [ -0.146, +0.105] | no  | +0.006±0.021 [ -0.047, +0.060] | no 
    B05-AC    | best_fpr     | 3 | +0.038±0.106 [ -0.226, +0.301] | no  | -0.012±0.065 [ -0.174, +0.149] | no  | +0.007±0.016 [ -0.032, +0.047] | no 
    B05-AC    | best_fitness | 3 | +0.001±0.196 [ -0.486, +0.488] | no  | -0.009±0.069 [ -0.180, +0.162] | no  | +0.002±0.020 [ -0.048, +0.051] | no 
    B05-CE    | best_f1      | 3 | +0.001±0.047 [ -0.116, +0.119] | no  | -0.077±0.083 [ -0.282, +0.128] | no  | +0.014±0.020 [ -0.034, +0.063] | no 
    B05-CE    | best_fpr     | 3 | +0.068±0.021 [ +0.015, +0.121] | YES | -0.052±0.061 [ -0.203, +0.100] | no  | +0.019±0.014 [ -0.016, +0.053] | no 
    B05-CE    | best_fitness | 3 | +0.067±0.067 [ -0.099, +0.234] | no  | -0.033±0.104 [ -0.292, +0.227] | no  | +0.015±0.021 [ -0.037, +0.067] | no 
    B10-AC    | best_f1      | 3 | -0.022±0.054 [ -0.157, +0.113] | no  | -0.020±0.117 [ -0.310, +0.270] | no  | +0.001±0.014 [ -0.034, +0.036] | no 
    B10-AC    | best_fpr     | 3 | +0.062±0.054 [ -0.071, +0.195] | no  | -0.033±0.055 [ -0.169, +0.103] | no  | +0.015±0.006 [ +0.001, +0.029] | YES
    B10-AC    | best_fitness | 3 | +0.030±0.090 [ -0.195, +0.254] | no  | -0.037±0.054 [ -0.171, +0.096] | no  | +0.011±0.007 [ -0.006, +0.028] | no 
    B10-CE    | best_f1      | 3 | -0.012±0.071 [ -0.188, +0.164] | no  | -0.024±0.048 [ -0.144, +0.096] | no  | +0.003±0.019 [ -0.043, +0.049] | no 
    B10-CE    | best_fpr     | 3 | -0.017±0.116 [ -0.305, +0.272] | no  | -0.030±0.034 [ -0.113, +0.053] | no  | +0.003±0.018 [ -0.042, +0.048] | no 
    B10-CE    | best_fitness | 3 | -0.001±0.124 [ -0.308, +0.306] | no  | -0.021±0.024 [ -0.082, +0.039] | no  | +0.004±0.020 [ -0.047, +0.054] | no 
    B15-AC    | best_f1      | 3 | +0.029±0.109 [ -0.242, +0.300] | no  | +0.001±0.102 [ -0.253, +0.255] | no  | +0.004±0.007 [ -0.014, +0.022] | no 
    B15-AC    | best_fpr     | 3 | +0.054±0.082 [ -0.151, +0.258] | no  | -0.008±0.091 [ -0.234, +0.219] | no  | +0.009±0.008 [ -0.012, +0.029] | no 
    B15-AC    | best_fitness | 3 | +0.030±0.109 [ -0.240, +0.300] | no  | +0.018±0.127 [ -0.298, +0.335] | no  | +0.001±0.008 [ -0.020, +0.021] | no 
    B15-CE    | best_f1      | 3 | +0.044±0.097 [ -0.196, +0.285] | no  | +0.047±0.054 [ -0.088, +0.182] | no  | -0.002±0.010 [ -0.028, +0.023] | no 
    B15-CE    | best_fpr     | 3 | +0.081±0.045 [ -0.031, +0.193] | no  | +0.065±0.040 [ -0.036, +0.165] | no  | -0.000±0.013 [ -0.033, +0.033] | no 
    B15-CE    | best_fitness | 3 | +0.072±0.083 [ -0.134, +0.278] | no  | -0.009±0.041 [ -0.110, +0.092] | no  | +0.012±0.019 [ -0.035, +0.058] | no 
    CE20      | best_f1      | 3 | -0.068±0.027 [ -0.135, +0.000] | no  | -0.013±0.147 [ -0.379, +0.353] | no  | -0.007±0.029 [ -0.079, +0.065] | no 
    CE20      | best_fpr     | 3 | -0.035±0.110 [ -0.308, +0.238] | no  | -0.007±0.100 [ -0.257, +0.242] | no  | -0.004±0.028 [ -0.073, +0.066] | no 
    CE20      | best_fitness | 3 | -0.039±0.069 [ -0.210, +0.131] | no  | +0.005±0.145 [ -0.355, +0.365] | no  | -0.007±0.032 [ -0.085, +0.072] | no 
    B34-CTRL  | best_f1      | 3 | +0.028±0.109 [ -0.243, +0.298] | no  | -0.029±0.024 [ -0.088, +0.029] | no  | +0.009±0.014 [ -0.026, +0.044] | no 
    B34-CTRL  | best_fpr     | 3 | -0.029±0.171 [ -0.452, +0.395] | no  | +0.044±0.087 [ -0.172, +0.260] | no  | -0.012±0.035 [ -0.100, +0.076] | no 
    B34-CTRL  | best_fitness | 3 | +0.043±0.146 [ -0.320, +0.407] | no  | -0.012±0.017 [ -0.055, +0.032] | no  | +0.008±0.018 [ -0.037, +0.053] | no 
```

    Sign agreement best_f1 vs best_fpr vs best_fitness on dF1 / dFPR vs Wb-CTRL (+ = arm higher):
```
    Arm       | dF1 sign (f1/fpr/fit) | dFPR sign (f1/fpr/fit) | Pareto read
    ----------+-----------------------+------------------------+------------------------------------------
    B05-AC    |          +++          |          −−−           | dominates control (F1 up, FPR down) — point estimates only
    B05-CE    |          +++          |          −−−           | dominates control (F1 up, FPR down) — point estimates only
    B10-AC    |          −++          |          −−−           | genome types DISAGREE on sign — no direction
    B10-CE    |          −−−          |          −−−           | Pareto MOVE: F1 down but FPR down
    B15-AC    |          +++          |          +−+           | genome types DISAGREE on sign — no direction
    B15-CE    |          +++          |          ++−           | genome types DISAGREE on sign — no direction
    CE20      |          −−−          |          −−+           | genome types DISAGREE on sign — no direction
    B34-CTRL  |          +−+          |          −+−           | genome types DISAGREE on sign — no direction
```

### unswr-qsr — paired-by-seed delta vs B34-CTRL (arm − control, pp), 95% t-CI df=n−1, HELD-OUT val_cal
```
    Arm       | genome       | n | dF1  mean±SD   [95% CI]          | ex0 | dFPR mean±SD   [95% CI]          | ex0 | dAcc mean±SD   [95% CI]          | ex0
    ----------+--------------+---+----------------------------------+-----+----------------------------------+-----+----------------------------------+----
    B05-AC    | best_f1      | 3 | -0.008±0.168 [ -0.424, +0.409] | no  | +0.009±0.053 [ -0.123, +0.141] | no  | -0.003±0.033 [ -0.084, +0.078] | no 
    B05-AC    | best_fpr     | 3 | +0.066±0.249 [ -0.551, +0.684] | no  | -0.056±0.091 [ -0.282, +0.169] | no  | +0.020±0.051 [ -0.107, +0.146] | no 
    B05-AC    | best_fitness | 3 | -0.043±0.172 [ -0.470, +0.385] | no  | +0.003±0.056 [ -0.136, +0.141] | no  | -0.006±0.027 [ -0.075, +0.062] | no 
    B05-CE    | best_f1      | 3 | -0.027±0.064 [ -0.185, +0.132] | no  | -0.048±0.077 [ -0.238, +0.142] | no  | +0.005±0.006 [ -0.010, +0.020] | no 
    B05-CE    | best_fpr     | 3 | +0.097±0.149 [ -0.274, +0.467] | no  | -0.096±0.060 [ -0.244, +0.053] | no  | +0.031±0.022 [ -0.024, +0.086] | no 
    B05-CE    | best_fitness | 3 | +0.024±0.091 [ -0.202, +0.249] | no  | -0.021±0.098 [ -0.264, +0.222] | no  | +0.007±0.011 [ -0.021, +0.035] | no 
    B10-AC    | best_f1      | 3 | -0.050±0.134 [ -0.384, +0.284] | no  | +0.009±0.096 [ -0.230, +0.249] | no  | -0.008±0.016 [ -0.048, +0.031] | no 
    B10-AC    | best_fpr     | 3 | +0.091±0.223 [ -0.463, +0.645] | no  | -0.077±0.090 [ -0.302, +0.147] | no  | +0.027±0.036 [ -0.062, +0.115] | no 
    B10-AC    | best_fitness | 3 | -0.014±0.165 [ -0.424, +0.397] | no  | -0.026±0.050 [ -0.149, +0.097] | no  | +0.003±0.014 [ -0.031, +0.037] | no 
    B10-CE    | best_f1      | 3 | -0.039±0.059 [ -0.186, +0.107] | no  | +0.005±0.049 [ -0.117, +0.127] | no  | -0.006±0.006 [ -0.021, +0.008] | no 
    B10-CE    | best_fpr     | 3 | +0.012±0.055 [ -0.124, +0.148] | no  | -0.074±0.111 [ -0.351, +0.202] | no  | +0.015±0.024 [ -0.044, +0.075] | no 
    B10-CE    | best_fitness | 3 | -0.044±0.102 [ -0.297, +0.208] | no  | -0.010±0.038 [ -0.104, +0.085] | no  | -0.004±0.009 [ -0.026, +0.018] | no 
    B15-AC    | best_f1      | 3 | +0.001±0.111 [ -0.274, +0.276] | no  | +0.030±0.083 [ -0.176, +0.236] | no  | -0.005±0.021 [ -0.059, +0.048] | no 
    B15-AC    | best_fpr     | 3 | +0.082±0.199 [ -0.412, +0.577] | no  | -0.052±0.082 [ -0.255, +0.152] | no  | +0.021±0.041 [ -0.080, +0.122] | no 
    B15-AC    | best_fitness | 3 | -0.013±0.096 [ -0.252, +0.226] | no  | +0.030±0.110 [ -0.244, +0.304] | no  | -0.007±0.024 [ -0.067, +0.052] | no 
    B15-CE    | best_f1      | 3 | +0.017±0.197 [ -0.472, +0.505] | no  | +0.076±0.035 [ -0.011, +0.164] | no  | -0.011±0.024 [ -0.070, +0.047] | no 
    B15-CE    | best_fpr     | 3 | +0.110±0.168 [ -0.307, +0.527] | no  | +0.020±0.047 [ -0.096, +0.136] | no  | +0.012±0.028 [ -0.057, +0.081] | no 
    B15-CE    | best_fitness | 3 | +0.029±0.210 [ -0.493, +0.550] | no  | +0.002±0.024 [ -0.056, +0.061] | no  | +0.004±0.033 [ -0.079, +0.086] | no 
    CE20      | best_f1      | 3 | -0.095±0.135 [ -0.430, +0.239] | no  | +0.016±0.125 [ -0.294, +0.326] | no  | -0.016±0.031 [ -0.093, +0.061] | no 
    CE20      | best_fpr     | 3 | -0.006±0.061 [ -0.159, +0.146] | no  | -0.051±0.032 [ -0.131, +0.028] | no  | +0.009±0.013 [ -0.025, +0.042] | no 
    CE20      | best_fitness | 3 | -0.083±0.108 [ -0.350, +0.185] | no  | +0.017±0.129 [ -0.303, +0.336] | no  | -0.015±0.036 [ -0.103, +0.074] | no 
    Wb-CTRL   | best_f1      | 3 | -0.028±0.109 [ -0.298, +0.243] | no  | +0.029±0.024 [ -0.029, +0.088] | no  | -0.009±0.014 [ -0.044, +0.026] | no 
    Wb-CTRL   | best_fpr     | 3 | +0.029±0.171 [ -0.395, +0.452] | no  | -0.044±0.087 [ -0.260, +0.172] | no  | +0.012±0.035 [ -0.076, +0.100] | no 
    Wb-CTRL   | best_fitness | 3 | -0.043±0.146 [ -0.407, +0.320] | no  | +0.012±0.017 [ -0.032, +0.055] | no  | -0.008±0.018 [ -0.053, +0.037] | no 
```

    Sign agreement best_f1 vs best_fpr vs best_fitness on dF1 / dFPR vs B34-CTRL (+ = arm higher):
```
    Arm       | dF1 sign (f1/fpr/fit) | dFPR sign (f1/fpr/fit) | Pareto read
    ----------+-----------------------+------------------------+------------------------------------------
    B05-AC    |          −+−          |          +−+           | genome types DISAGREE on sign — no direction
    B05-CE    |          −++          |          −−−           | genome types DISAGREE on sign — no direction
    B10-AC    |          −+−          |          +−−           | genome types DISAGREE on sign — no direction
    B10-CE    |          −+−          |          +−−           | genome types DISAGREE on sign — no direction
    B15-AC    |          ++−          |          +−+           | genome types DISAGREE on sign — no direction
    B15-CE    |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
    CE20      |          −−−          |          +−+           | genome types DISAGREE on sign — no direction
    Wb-CTRL   |          −+−          |          +−+           | genome types DISAGREE on sign — no direction
```

### unswr-qsr — pairwise paired dF1 (row − col, pp), best_f1 val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wb-CTRL       |      B34-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | +0.019±0.126    3 | +0.042±0.078    3 | +0.032±0.175    3 | -0.009±0.061    3 | -0.024±0.111    3 | +0.088±0.142    3 | +0.020±0.129    3 | -0.008±0.168    3
    B05-CE    | -0.019±0.126    3 |                     | +0.023±0.074    3 | +0.013±0.048    3 | -0.028±0.085    3 | -0.043±0.134    3 | +0.069±0.074    3 | +0.001±0.047    3 | -0.027±0.064    3
    B10-AC    | -0.042±0.078    3 | -0.023±0.074    3 |                     | -0.010±0.116    3 | -0.051±0.080    3 | -0.066±0.065    3 | +0.046±0.063    3 | -0.022±0.054    3 | -0.050±0.134    3
    B10-CE    | -0.032±0.175    3 | -0.013±0.048    3 | +0.010±0.116    3 |                     | -0.041±0.131    3 | -0.056±0.167    3 | +0.056±0.090    3 | -0.012±0.071    3 | -0.039±0.059    3
    B15-AC    | +0.009±0.061    3 | +0.028±0.085    3 | +0.051±0.080    3 | +0.041±0.131    3 |                     | -0.015±0.139    3 | +0.096±0.131    3 | +0.029±0.109    3 | +0.001±0.111    3
    B15-CE    | +0.024±0.111    3 | +0.043±0.134    3 | +0.066±0.065    3 | +0.056±0.167    3 | +0.015±0.139    3 |                     | +0.112±0.084    3 | +0.044±0.097    3 | +0.017±0.197    3
    CE20      | -0.088±0.142    3 | -0.069±0.074    3 | -0.046±0.063    3 | -0.056±0.090    3 | -0.096±0.131    3 | -0.112±0.084    3 |                     | -0.068±0.027 SEP3 | -0.095±0.135    3
    Wb-CTRL   | -0.020±0.129    3 | -0.001±0.047    3 | +0.022±0.054    3 | +0.012±0.071    3 | -0.029±0.109    3 | -0.044±0.097    3 | +0.068±0.027 SEP3 |                     | -0.028±0.109    3
    B34-CTRL  | +0.008±0.168    3 | +0.027±0.064    3 | +0.050±0.134    3 | +0.039±0.059    3 | -0.001±0.111    3 | -0.017±0.197    3 | +0.095±0.135    3 | +0.028±0.109    3 |                    
```

    Arms whose paired dF1 vs Wb-CTRL CI EXCLUDES 0 upward: NONE
    Arms whose paired dF1 vs Wb-CTRL CI EXCLUDES 0 downward: NONE
    Top arm by mean F1 (best_f1): B34-CTRL (94.376); NOT separated from: ['B05-AC', 'B05-CE', 'B10-AC', 'B10-CE', 'B15-AC', 'B15-CE', 'CE20', 'Wb-CTRL']

### unswr-qsr — pairwise paired dF1 (row − col, pp), best_fitness val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wb-CTRL       |      B34-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | -0.066±0.144    3 | -0.029±0.122    3 | +0.002±0.251    3 | -0.029±0.097    3 | -0.071±0.278    3 | +0.040±0.221    3 | +0.001±0.196    3 | -0.043±0.172    3
    B05-CE    | +0.066±0.144    3 |                     | +0.038±0.080    3 | +0.068±0.118    3 | +0.037±0.048    3 | -0.005±0.147    3 | +0.107±0.077    3 | +0.067±0.067    3 | +0.024±0.091    3
    B10-AC    | +0.029±0.122    3 | -0.038±0.080    3 |                     | +0.031±0.193    3 | -0.000±0.076    3 | -0.042±0.165    3 | +0.069±0.144    3 | +0.030±0.090    3 | -0.014±0.165    3
    B10-CE    | -0.002±0.251    3 | -0.068±0.118    3 | -0.031±0.193    3 |                     | -0.031±0.156    3 | -0.073±0.146    3 | +0.038±0.055    3 | -0.001±0.124    3 | -0.044±0.102    3
    B15-AC    | +0.029±0.097    3 | -0.037±0.048    3 | +0.000±0.076    3 | +0.031±0.156    3 |                     | -0.042±0.191    3 | +0.070±0.123    3 | +0.030±0.109    3 | -0.013±0.096    3
    B15-CE    | +0.071±0.278    3 | +0.005±0.147    3 | +0.042±0.165    3 | +0.073±0.146    3 | +0.042±0.191    3 |                     | +0.112±0.105    3 | +0.072±0.083    3 | +0.029±0.210    3
    CE20      | -0.040±0.221    3 | -0.107±0.077    3 | -0.069±0.144    3 | -0.038±0.055    3 | -0.070±0.123    3 | -0.112±0.105    3 |                     | -0.039±0.069    3 | -0.083±0.108    3
    Wb-CTRL   | -0.001±0.196    3 | -0.067±0.067    3 | -0.030±0.090    3 | +0.001±0.124    3 | -0.030±0.109    3 | -0.072±0.083    3 | +0.039±0.069    3 |                     | -0.043±0.146    3
    B34-CTRL  | +0.043±0.172    3 | -0.024±0.091    3 | +0.014±0.165    3 | +0.044±0.102    3 | +0.013±0.096    3 | -0.029±0.210    3 | +0.083±0.108    3 | +0.043±0.146    3 |                    
```
### unswr-qsr — paired SD_d (vs Wb-CTRL, best_f1 val_cal F1) and MDD at 80% power, two-sided α=0.05, paired t
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10 | n needed for |gap|
    ----------+---+-------------+--------+---------+---------+---------+----------+-------------------
    B05-AC    | 3 |      +0.020 |  0.129 |   0.421 |   0.217 |   0.164 |    0.128 | >200
    B05-CE    | 3 |      +0.001 |  0.047 |   0.154 |   0.080 |   0.060 |    0.047 | >200
    B10-AC    | 3 |      -0.022 |  0.054 |   0.178 |   0.092 |   0.069 |    0.054 | 50
    B10-CE    | 3 |      -0.012 |  0.071 |   0.231 |   0.119 |   0.090 |    0.070 | >200
    B15-AC    | 3 |      +0.029 |  0.109 |   0.356 |   0.183 |   0.139 |    0.109 | 114
    B15-CE    | 3 |      +0.044 |  0.097 |   0.316 |   0.163 |   0.123 |    0.096 | 40
    CE20      | 3 |      -0.068 |  0.027 |   0.089 |   0.046 |   0.035 |    0.027 | 4
    B34-CTRL  | 3 |      +0.028 |  0.109 |   0.355 |   0.183 |   0.139 |    0.108 | 123
    pooled    |   |             |  0.087 |   0.284 |   0.146 |   0.111 |    0.087 | (RMS of paired SDs)
```
    FPR version (paired SD_d of dFPR vs Wb-CTRL, best_fpr genome val_cal):
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10
    ----------+---+-------------+--------+---------+---------+---------+---------
    B05-AC    | 3 |      -0.012 |  0.065 |   0.212 |   0.109 |   0.083 |    0.065
    B05-CE    | 3 |      -0.052 |  0.061 |   0.199 |   0.102 |   0.077 |    0.061
    B10-AC    | 3 |      -0.033 |  0.055 |   0.179 |   0.092 |   0.070 |    0.054
    B10-CE    | 3 |      -0.030 |  0.034 |   0.109 |   0.056 |   0.043 |    0.033
    B15-AC    | 3 |      -0.008 |  0.091 |   0.298 |   0.153 |   0.116 |    0.091
    B15-CE    | 3 |      +0.065 |  0.040 |   0.132 |   0.068 |   0.051 |    0.040
    CE20      | 3 |      -0.007 |  0.100 |   0.328 |   0.169 |   0.128 |    0.100
    B34-CTRL  | 3 |      +0.044 |  0.087 |   0.284 |   0.146 |   0.111 |    0.087
```
    Within-arm SD of F1 (best_f1 val_cal), per arm and pooled:
```
    B05-AC    | n=3 | SD= 0.120 | values:  94.351  94.211  94.450
    B05-CE    | n=3 | SD= 0.006 | values:  94.315  94.325  94.314
    B10-AC    | n=3 | SD= 0.068 | values:  94.349  94.219  94.317
    B10-CE    | n=3 | SD= 0.054 | values:  94.298  94.363  94.255
    B15-AC    | n=3 | SD= 0.080 | values:  94.312  94.288  94.437
    B15-CE    | n=3 | SD= 0.128 | values:  94.488  94.231  94.364
    CE20      | n=3 | SD= 0.072 | values:  94.332  94.217  94.199
    Wb-CTRL   | n=3 | SD= 0.045 | values:  94.369  94.296  94.286
    B34-CTRL  | n=5 | SD= 0.065 | values:  94.415  94.429  94.274  94.411  94.348
    POOLED within-arm SD (RMS) = 0.079 pp ; median = 0.068
```

## 4. unswr-quad-64b (control Wb-CTRL; also vs B34-CTRL fixed-bits control)

### unswr-quad — best_f1 genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  93.45±0.05 |   1.12±0.00 |  98.91±0.01 |  70.00±17.32 | 03P 04P 05P
    B05-CE    | 3 |  93.47±0.02 |   1.12±0.00 |  98.92±0.00 |  80.00±17.32 | 03P 04P 05P
    B10-AC    | 3 |  93.47±0.02 |   1.02±0.16 |  98.93±0.03 |  70.00±17.32 | 03P 04P 05P
    B10-CE    | 3 |  93.48±0.02 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    B15-AC    | 3 |  93.48±0.02 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    B15-CE    | 3 |  93.49±0.01 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    CE20      | 3 |  93.41±0.07 |   0.98±0.23 |  98.93±0.04 | 110.00±34.64 | 03P 04P 05P
    Wb-CTRL   | 3 |  93.44±0.04 |   0.99±0.23 |  98.94±0.04 | 103.33±40.41 | 03P 04P 05P
    B34-CTRL  | 5 |  93.49±0.01 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 01P 02P 03P 04P 05P
```

### unswr-quad — best_fpr genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  93.45±0.03 |   1.12±0.01 |  98.91±0.01 |  70.00±17.32 | 03P 04P 05P
    B05-CE    | 3 |  93.48±0.01 |   1.12±0.00 |  98.92±0.00 |  80.00±17.32 | 03P 04P 05P
    B10-AC    | 3 |  93.45±0.03 |   1.12±0.00 |  98.91±0.00 |  70.00±17.32 | 03P 04P 05P
    B10-CE    | 3 |  93.46±0.03 |   1.12±0.00 |  98.91±0.00 |  60.00±0.00 | 03P 04P 05P
    B15-AC    | 3 |  93.47±0.02 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    B15-CE    | 3 |  93.48±0.03 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    CE20      | 3 |  93.44±0.03 |   1.12±0.00 |  98.91±0.01 | 110.00±34.64 | 03P 04P 05P
    Wb-CTRL   | 3 |  93.41±0.09 |   0.98±0.24 |  98.93±0.04 | 103.33±40.41 | 03P 04P 05P
    B34-CTRL  | 5 |  93.47±0.04 |   1.12±0.00 |  98.91±0.01 |  60.00±0.00 | 01P 02P 03P 04P 05P
```

### unswr-quad — best_fitness genome, GA final, HELD-OUT val_cal (mean±SD, %)
```
    Arm       | n | F1            | FPR           | Acc           | GA gens      | seeds (era)
    ----------+---+---------------+---------------+---------------+--------------+------------------
    B05-AC    | 3 |  93.45±0.04 |   1.12±0.00 |  98.91±0.01 |  70.00±17.32 | 03P 04P 05P
    B05-CE    | 3 |  93.49±0.00 |   1.12±0.00 |  98.92±0.00 |  80.00±17.32 | 03P 04P 05P
    B10-AC    | 3 |  93.47±0.02 |   1.02±0.18 |  98.93±0.04 |  70.00±17.32 | 03P 04P 05P
    B10-CE    | 3 |  93.48±0.01 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    B15-AC    | 3 |  93.48±0.01 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    B15-CE    | 3 |  93.49±0.01 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 03P 04P 05P
    CE20      | 3 |  93.45±0.03 |   1.12±0.00 |  98.91±0.01 | 110.00±34.64 | 03P 04P 05P
    Wb-CTRL   | 3 |  93.44±0.04 |   0.99±0.23 |  98.94±0.04 | 103.33±40.41 | 03P 04P 05P
    B34-CTRL  | 5 |  93.48±0.01 |   1.12±0.00 |  98.92±0.00 |  60.00±0.00 | 01P 02P 03P 04P 05P
```

### unswr-quad — paired-by-seed delta vs Wb-CTRL (arm − control, pp), 95% t-CI df=n−1, HELD-OUT val_cal
```
    Arm       | genome       | n | dF1  mean±SD   [95% CI]          | ex0 | dFPR mean±SD   [95% CI]          | ex0 | dAcc mean±SD   [95% CI]          | ex0
    ----------+--------------+---+----------------------------------+-----+----------------------------------+-----+----------------------------------+----
    B05-AC    | best_f1      | 3 | +0.008±0.035 [ -0.078, +0.095] | no  | +0.131±0.227 [ -0.433, +0.695] | no  | -0.026±0.043 [ -0.132, +0.080] | no 
    B05-AC    | best_fpr     | 3 | +0.037±0.096 [ -0.202, +0.275] | no  | +0.140±0.235 [ -0.445, +0.724] | no  | -0.024±0.036 [ -0.113, +0.065] | no 
    B05-AC    | best_fitness | 3 | +0.005±0.037 [ -0.087, +0.096] | no  | +0.131±0.227 [ -0.433, +0.695] | no  | -0.027±0.042 [ -0.131, +0.078] | no 
    B05-CE    | best_f1      | 3 | +0.033±0.059 [ -0.113, +0.179] | no  | +0.131±0.228 [ -0.436, +0.698] | no  | -0.022±0.043 [ -0.129, +0.085] | no 
    B05-CE    | best_fpr     | 3 | +0.072±0.090 [ -0.150, +0.295] | no  | +0.138±0.238 [ -0.453, +0.729] | no  | -0.018±0.037 [ -0.111, +0.074] | no 
    B05-CE    | best_fitness | 3 | +0.041±0.035 [ -0.046, +0.128] | no  | +0.132±0.227 [ -0.433, +0.697] | no  | -0.021±0.043 [ -0.128, +0.085] | no 
    B10-AC    | best_f1      | 3 | +0.029±0.039 [ -0.068, +0.125] | no  | +0.036±0.063 [ -0.121, +0.193] | no  | -0.003±0.013 [ -0.036, +0.029] | no 
    B10-AC    | best_fpr     | 3 | +0.041±0.073 [ -0.141, +0.223] | no  | +0.137±0.237 [ -0.452, +0.726] | no  | -0.023±0.041 [ -0.124, +0.079] | no 
    B10-AC    | best_fitness | 3 | +0.023±0.058 [ -0.122, +0.168] | no  | +0.030±0.052 [ -0.099, +0.160] | no  | -0.003±0.008 [ -0.024, +0.018] | no 
    B10-CE    | best_f1      | 3 | +0.040±0.058 [ -0.103, +0.183] | no  | +0.131±0.228 [ -0.434, +0.696] | no  | -0.021±0.043 [ -0.129, +0.087] | no 
    B10-CE    | best_fpr     | 3 | +0.051±0.101 [ -0.200, +0.303] | no  | +0.137±0.238 [ -0.454, +0.729] | no  | -0.021±0.035 [ -0.109, +0.066] | no 
    B10-CE    | best_fitness | 3 | +0.041±0.037 [ -0.052, +0.133] | no  | +0.131±0.228 [ -0.434, +0.697] | no  | -0.021±0.044 [ -0.129, +0.087] | no 
    B15-AC    | best_f1      | 3 | +0.035±0.030 [ -0.039, +0.110] | no  | +0.132±0.227 [ -0.432, +0.695] | no  | -0.022±0.046 [ -0.135, +0.091] | no 
    B15-AC    | best_fpr     | 3 | +0.064±0.078 [ -0.129, +0.256] | no  | +0.138±0.237 [ -0.452, +0.728] | no  | -0.020±0.039 [ -0.116, +0.077] | no 
    B15-AC    | best_fitness | 3 | +0.034±0.029 [ -0.039, +0.107] | no  | +0.132±0.227 [ -0.432, +0.695] | no  | -0.022±0.044 [ -0.132, +0.088] | no 
    B15-CE    | best_f1      | 3 | +0.053±0.048 [ -0.066, +0.171] | no  | +0.132±0.227 [ -0.431, +0.695] | no  | -0.020±0.042 [ -0.125, +0.086] | no 
    B15-CE    | best_fpr     | 3 | +0.066±0.108 [ -0.202, +0.333] | no  | +0.138±0.238 [ -0.452, +0.729] | no  | -0.019±0.034 [ -0.104, +0.065] | no 
    B15-CE    | best_fitness | 3 | +0.046±0.046 [ -0.068, +0.159] | no  | +0.133±0.226 [ -0.429, +0.695] | no  | -0.021±0.042 [ -0.125, +0.083] | no 
    CE20      | best_f1      | 3 | -0.027±0.071 [ -0.203, +0.150] | no  | -0.003±0.005 [ -0.014, +0.009] | no  | -0.003±0.010 [ -0.028, +0.021] | no 
    CE20      | best_fpr     | 3 | +0.029±0.106 [ -0.233, +0.292] | no  | +0.142±0.242 [ -0.458, +0.742] | no  | -0.025±0.038 [ -0.119, +0.068] | no 
    CE20      | best_fitness | 3 | +0.009±0.063 [ -0.147, +0.165] | no  | +0.134±0.232 [ -0.443, +0.711] | no  | -0.027±0.044 [ -0.137, +0.084] | no 
    B34-CTRL  | best_f1      | 3 | +0.051±0.038 [ -0.042, +0.145] | no  | +0.131±0.228 [ -0.434, +0.696] | no  | -0.020±0.044 [ -0.129, +0.090] | no 
    B34-CTRL  | best_fpr     | 3 | +0.068±0.098 [ -0.175, +0.311] | no  | +0.139±0.237 [ -0.450, +0.727] | no  | -0.019±0.035 [ -0.107, +0.069] | no 
    B34-CTRL  | best_fitness | 3 | +0.035±0.034 [ -0.050, +0.120] | no  | +0.133±0.226 [ -0.429, +0.694] | no  | -0.022±0.042 [ -0.126, +0.082] | no 
```

    Sign agreement best_f1 vs best_fpr vs best_fitness on dF1 / dFPR vs Wb-CTRL (+ = arm higher):
```
    Arm       | dF1 sign (f1/fpr/fit) | dFPR sign (f1/fpr/fit) | Pareto read
    ----------+-----------------------+------------------------+------------------------------------------
    B05-AC    |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
    B05-CE    |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
    B10-AC    |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
    B10-CE    |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
    B15-AC    |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
    B15-CE    |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
    CE20      |          −++          |          −++           | genome types DISAGREE on sign — no direction
    B34-CTRL  |          +++          |          +++           | Pareto MOVE: F1 up but FPR up
```

### unswr-quad — paired-by-seed delta vs B34-CTRL (arm − control, pp), 95% t-CI df=n−1, HELD-OUT val_cal
```
    Arm       | genome       | n | dF1  mean±SD   [95% CI]          | ex0 | dFPR mean±SD   [95% CI]          | ex0 | dAcc mean±SD   [95% CI]          | ex0
    ----------+--------------+---+----------------------------------+-----+----------------------------------+-----+----------------------------------+----
    B05-AC    | best_f1      | 3 | -0.043±0.047 [ -0.160, +0.075] | no  | -0.000±0.000 [ -0.001, +0.001] | no  | -0.006±0.007 [ -0.024, +0.011] | no 
    B05-AC    | best_fpr     | 3 | -0.031±0.013 [ -0.062, +0.000] | no  | +0.001±0.002 [ -0.005, +0.007] | no  | -0.005±0.002 [ -0.010, +0.001] | no 
    B05-AC    | best_fitness | 3 | -0.031±0.021 [ -0.084, +0.022] | no  | -0.002±0.002 [ -0.006, +0.003] | no  | -0.004±0.003 [ -0.011, +0.003] | no 
    B05-CE    | best_f1      | 3 | -0.018±0.022 [ -0.072, +0.036] | no  | +0.000±0.001 [ -0.002, +0.002] | no  | -0.003±0.003 [ -0.010, +0.005] | no 
    B05-CE    | best_fpr     | 3 | +0.005±0.028 [ -0.065, +0.075] | no  | -0.001±0.003 [ -0.008, +0.006] | no  | +0.001±0.005 [ -0.011, +0.013] | no 
    B05-CE    | best_fitness | 3 | +0.006±0.013 [ -0.027, +0.039] | no  | -0.001±0.002 [ -0.006, +0.004] | no  | +0.001±0.002 [ -0.005, +0.007] | no 
    B10-AC    | best_f1      | 3 | -0.023±0.016 [ -0.062, +0.017] | no  | -0.095±0.164 [ -0.503, +0.314] | no  | +0.016±0.032 [ -0.063, +0.095] | no 
    B10-AC    | best_fpr     | 3 | -0.027±0.053 [ -0.158, +0.105] | no  | -0.002±0.001 [ -0.005, +0.002] | no  | -0.004±0.008 [ -0.024, +0.017] | no 
    B10-AC    | best_fitness | 3 | -0.013±0.038 [ -0.106, +0.081] | no  | -0.102±0.174 [ -0.534, +0.330] | no  | +0.019±0.037 [ -0.072, +0.111] | no 
    B10-CE    | best_f1      | 3 | -0.011±0.021 [ -0.063, +0.041] | no  | +0.000±0.001 [ -0.002, +0.002] | no  | -0.002±0.003 [ -0.009, +0.006] | no 
    B10-CE    | best_fpr     | 3 | -0.016±0.006 [ -0.032, -0.001] | YES | -0.002±0.003 [ -0.008, +0.005] | no  | -0.002±0.000 [ -0.003, -0.001] | YES
    B10-CE    | best_fitness | 3 | +0.005±0.021 [ -0.047, +0.058] | no  | -0.001±0.002 [ -0.006, +0.004] | no  | +0.001±0.003 [ -0.008, +0.010] | no 
    B15-AC    | best_f1      | 3 | -0.016±0.012 [ -0.046, +0.014] | no  | +0.001±0.001 [ -0.001, +0.002] | no  | -0.003±0.002 [ -0.007, +0.002] | no 
    B15-AC    | best_fpr     | 3 | -0.004±0.033 [ -0.085, +0.077] | no  | -0.001±0.003 [ -0.008, +0.007] | no  | -0.000±0.005 [ -0.014, +0.013] | no 
    B15-AC    | best_fitness | 3 | -0.002±0.020 [ -0.051, +0.048] | no  | -0.001±0.002 [ -0.006, +0.004] | no  | -0.000±0.003 [ -0.008, +0.008] | no 
    B15-CE    | best_f1      | 3 | +0.002±0.011 [ -0.026, +0.029] | no  | +0.001±0.001 [ -0.002, +0.004] | no  | +0.000±0.002 [ -0.005, +0.005] | no 
    B15-CE    | best_fpr     | 3 | -0.002±0.010 [ -0.027, +0.023] | no  | -0.000±0.004 [ -0.011, +0.010] | no  | -0.000±0.002 [ -0.004, +0.004] | no 
    B15-CE    | best_fitness | 3 | +0.010±0.021 [ -0.042, +0.063] | no  | +0.000±0.003 [ -0.008, +0.008] | no  | +0.001±0.004 [ -0.008, +0.011] | no 
    CE20      | best_f1      | 3 | -0.078±0.067 [ -0.244, +0.088] | no  | -0.134±0.232 [ -0.710, +0.443] | no  | +0.016±0.040 [ -0.083, +0.116] | no 
    CE20      | best_fpr     | 3 | -0.038±0.055 [ -0.175, +0.098] | no  | +0.003±0.007 [ -0.014, +0.020] | no  | -0.006±0.009 [ -0.029, +0.017] | no 
    CE20      | best_fitness | 3 | -0.027±0.049 [ -0.148, +0.095] | no  | +0.001±0.006 [ -0.015, +0.017] | no  | -0.004±0.008 [ -0.024, +0.015] | no 
    Wb-CTRL   | best_f1      | 3 | -0.051±0.038 [ -0.145, +0.042] | no  | -0.131±0.228 [ -0.696, +0.434] | no  | +0.020±0.044 [ -0.090, +0.129] | no 
    Wb-CTRL   | best_fpr     | 3 | -0.068±0.098 [ -0.311, +0.175] | no  | -0.139±0.237 [ -0.727, +0.450] | no  | +0.019±0.035 [ -0.069, +0.107] | no 
    Wb-CTRL   | best_fitness | 3 | -0.035±0.034 [ -0.120, +0.050] | no  | -0.133±0.226 [ -0.694, +0.429] | no  | +0.022±0.042 [ -0.082, +0.126] | no 
```

    Sign agreement best_f1 vs best_fpr vs best_fitness on dF1 / dFPR vs B34-CTRL (+ = arm higher):
```
    Arm       | dF1 sign (f1/fpr/fit) | dFPR sign (f1/fpr/fit) | Pareto read
    ----------+-----------------------+------------------------+------------------------------------------
    B05-AC    |          −−−          |          −+−           | genome types DISAGREE on sign — no direction
    B05-CE    |          −++          |          −−−           | genome types DISAGREE on sign — no direction
    B10-AC    |          −−−          |          −−−           | Pareto MOVE: F1 down but FPR down
    B10-CE    |          −−+          |          −−−           | genome types DISAGREE on sign — no direction
    B15-AC    |          −−−          |          +−−           | genome types DISAGREE on sign — no direction
    B15-CE    |          +−+          |          +−+           | genome types DISAGREE on sign — no direction
    CE20      |          −−−          |          −++           | genome types DISAGREE on sign — no direction
    Wb-CTRL   |          −−−          |          −−−           | Pareto MOVE: F1 down but FPR down
```

### unswr-quad — pairwise paired dF1 (row − col, pp), best_f1 val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wb-CTRL       |      B34-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | -0.024±0.066    3 | -0.020±0.059    3 | -0.032±0.067    3 | -0.027±0.049    3 | -0.044±0.052    3 | +0.035±0.103    3 | +0.008±0.035    3 | -0.043±0.047    3
    B05-CE    | +0.024±0.066    3 |                     | +0.004±0.026    3 | -0.007±0.003 SEP3 | -0.002±0.030    3 | -0.020±0.015    3 | +0.060±0.069    3 | +0.033±0.059    3 | -0.018±0.022    3
    B10-AC    | +0.020±0.059    3 | -0.004±0.026    3 |                     | -0.011±0.023    3 | -0.007±0.010    3 | -0.024±0.023    3 | +0.055±0.051    3 | +0.029±0.039    3 | -0.023±0.016    3
    B10-CE    | +0.032±0.067    3 | +0.007±0.003 SEP3 | +0.011±0.023    3 |                     | +0.005±0.028    3 | -0.013±0.015    3 | +0.067±0.066    3 | +0.040±0.058    3 | -0.011±0.021    3
    B15-AC    | +0.027±0.049    3 | +0.002±0.030    3 | +0.007±0.010    3 | -0.005±0.028    3 |                     | -0.018±0.023    3 | +0.062±0.058    3 | +0.035±0.030    3 | -0.016±0.012    3
    B15-CE    | +0.044±0.052    3 | +0.020±0.015    3 | +0.024±0.023    3 | +0.013±0.015    3 | +0.018±0.023    3 |                     | +0.080±0.074    3 | +0.053±0.048    3 | +0.002±0.011    3
    CE20      | -0.035±0.103    3 | -0.060±0.069    3 | -0.055±0.051    3 | -0.067±0.066    3 | -0.062±0.058    3 | -0.080±0.074    3 |                     | -0.027±0.071    3 | -0.078±0.067    3
    Wb-CTRL   | -0.008±0.035    3 | -0.033±0.059    3 | -0.029±0.039    3 | -0.040±0.058    3 | -0.035±0.030    3 | -0.053±0.048    3 | +0.027±0.071    3 |                     | -0.051±0.038    3
    B34-CTRL  | +0.043±0.047    3 | +0.018±0.022    3 | +0.023±0.016    3 | +0.011±0.021    3 | +0.016±0.012    3 | -0.002±0.011    3 | +0.078±0.067    3 | +0.051±0.038    3 |                    
```

    Arms whose paired dF1 vs Wb-CTRL CI EXCLUDES 0 upward: NONE
    Arms whose paired dF1 vs Wb-CTRL CI EXCLUDES 0 downward: NONE
    Top arm by mean F1 (best_f1): B15-CE (93.494); NOT separated from: ['B05-AC', 'B05-CE', 'B10-AC', 'B10-CE', 'B15-AC', 'CE20', 'Wb-CTRL', 'B34-CTRL']

### unswr-quad — pairwise paired dF1 (row − col, pp), best_fitness val_cal; cell = mean±SD_d [sep?]  sep = CI excludes 0 OR |gap|>2·SD_d
```
              |       B05-AC        |       B05-CE        |       B10-AC        |       B10-CE        |       B15-AC        |       B15-CE        |        CE20         |       Wb-CTRL       |      B34-CTRL      
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    |                     | -0.037±0.034    3 | -0.018±0.059    3 | -0.036±0.042    3 | -0.029±0.039    3 | -0.041±0.042    3 | -0.004±0.070    3 | +0.005±0.037    3 | -0.031±0.021    3
    B05-CE    | +0.037±0.034    3 |                     | +0.019±0.026    3 | +0.001±0.008    3 | +0.007±0.009    3 | -0.004±0.011    3 | +0.032±0.036    3 | +0.041±0.035    3 | +0.006±0.013    3
    B10-AC    | +0.018±0.059    3 | -0.019±0.026    3 |                     | -0.018±0.021    3 | -0.011±0.029    3 | -0.023±0.016    3 | +0.014±0.014    3 | +0.023±0.058    3 | -0.013±0.038    3
    B10-CE    | +0.036±0.042    3 | -0.001±0.008    3 | +0.018±0.021    3 |                     | +0.007±0.008    3 | -0.005±0.010    3 | +0.032±0.029    3 | +0.041±0.037    3 | +0.005±0.021    3
    B15-AC    | +0.029±0.039    3 | -0.007±0.009    3 | +0.011±0.029    3 | -0.007±0.008    3 |                     | -0.012±0.018    3 | +0.025±0.035    3 | +0.034±0.029    3 | -0.002±0.020    3
    B15-CE    | +0.041±0.042    3 | +0.004±0.011    3 | +0.023±0.016    3 | +0.005±0.010    3 | +0.012±0.018    3 |                     | +0.037±0.029    3 | +0.046±0.046    3 | +0.010±0.021    3
    CE20      | +0.004±0.070    3 | -0.032±0.036    3 | -0.014±0.014    3 | -0.032±0.029    3 | -0.025±0.035    3 | -0.037±0.029    3 |                     | +0.009±0.063    3 | -0.027±0.049    3
    Wb-CTRL   | -0.005±0.037    3 | -0.041±0.035    3 | -0.023±0.058    3 | -0.041±0.037    3 | -0.034±0.029    3 | -0.046±0.046    3 | -0.009±0.063    3 |                     | -0.035±0.034    3
    B34-CTRL  | +0.031±0.021    3 | -0.006±0.013    3 | +0.013±0.038    3 | -0.005±0.021    3 | +0.002±0.020    3 | -0.010±0.021    3 | +0.027±0.049    3 | +0.035±0.034    3 |                    
```
### unswr-quad — paired SD_d (vs Wb-CTRL, best_f1 val_cal F1) and MDD at 80% power, two-sided α=0.05, paired t
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10 | n needed for |gap|
    ----------+---+-------------+--------+---------+---------+---------+----------+-------------------
    B05-AC    | 3 |      +0.008 |  0.035 |   0.114 |   0.059 |   0.044 |    0.035 | 136
    B05-CE    | 3 |      +0.033 |  0.059 |   0.192 |   0.099 |   0.075 |    0.059 | 28
    B10-AC    | 3 |      +0.029 |  0.039 |   0.127 |   0.065 |   0.049 |    0.039 | 17
    B10-CE    | 3 |      +0.040 |  0.058 |   0.188 |   0.097 |   0.073 |    0.057 | 19
    B15-AC    | 3 |      +0.035 |  0.030 |   0.098 |   0.050 |   0.038 |    0.030 | 8
    B15-CE    | 3 |      +0.053 |  0.048 |   0.156 |   0.080 |   0.061 |    0.048 | 9
    CE20      | 3 |      -0.027 |  0.071 |   0.232 |   0.119 |   0.090 |    0.071 | 58
    B34-CTRL  | 3 |      +0.051 |  0.038 |   0.122 |   0.063 |   0.048 |    0.037 | 7
    pooled    |   |             |  0.049 |   0.160 |   0.082 |   0.062 |    0.049 | (RMS of paired SDs)
```
    FPR version (paired SD_d of dFPR vs Wb-CTRL, best_fpr genome val_cal):
```
    Arm       | n | gap vs ctrl | SD_d   | MDD n=3 | MDD n=5 | MDD n=7 | MDD n=10
    ----------+---+-------------+--------+---------+---------+---------+---------
    B05-AC    | 3 |      +0.140 |  0.235 |   0.768 |   0.396 |   0.299 |    0.234
    B05-CE    | 3 |      +0.138 |  0.238 |   0.777 |   0.400 |   0.303 |    0.237
    B10-AC    | 3 |      +0.137 |  0.237 |   0.774 |   0.399 |   0.302 |    0.236
    B10-CE    | 3 |      +0.137 |  0.238 |   0.777 |   0.401 |   0.303 |    0.237
    B15-AC    | 3 |      +0.138 |  0.237 |   0.775 |   0.399 |   0.302 |    0.236
    B15-CE    | 3 |      +0.138 |  0.238 |   0.776 |   0.400 |   0.303 |    0.237
    CE20      | 3 |      +0.142 |  0.242 |   0.788 |   0.406 |   0.307 |    0.241
    B34-CTRL  | 3 |      +0.139 |  0.237 |   0.773 |   0.398 |   0.301 |    0.236
```
    Within-arm SD of F1 (best_f1 val_cal), per arm and pooled:
```
    B05-AC    | n=3 | SD= 0.047 | values:  93.449  93.497  93.402
    B05-CE    | n=3 | SD= 0.019 | values:  93.477  93.453  93.492
    B10-AC    | n=3 | SD= 0.022 | values:  93.446  93.474  93.489
    B10-CE    | n=3 | SD= 0.020 | values:  93.480  93.462  93.501
    B15-AC    | n=3 | SD= 0.019 | values:  93.454  93.489  93.485
    B15-CE    | n=3 | SD= 0.005 | values:  93.497  93.488  93.497
    CE20      | n=3 | SD= 0.073 | values:  93.338  93.420  93.484
    Wb-CTRL   | n=3 | SD= 0.043 | values:  93.406  93.488  93.429
    B34-CTRL  | n=5 | SD= 0.015 | values:  93.494  93.462  93.484  93.497  93.496
    POOLED within-arm SD (RMS) = 0.035 pp ; median = 0.020
```

### ciciot era detail per run (p = pre-fix start, P = post-fix start; W = -w64fix rerun)
```
    Arm       | seed  | flow | started (UTC)     | era  | best_f1 val_cal F1 | FPR    | Acc    | gens
    ----------+-------+------+-------------------+------+--------------------+--------+--------+-----
    B05-AC    | 20403 | 5886 | 29/08/2026 09:22  | pre  |             92.789 |  8.375 | 96.404 | 180
    B05-AC    | 20404 | 5887 | 29/08/2026 11:03  | pre  |             92.955 |  7.780 | 96.469 | 140
    B05-AC    | 20405 | 5888 | 29/08/2026 13:11  | pre  |             93.143 |  7.610 | 96.572 | 130
    B05-CE    | 20403 | 5889 | 29/08/2026 14:50  | pre  |             92.692 |  8.760 | 96.334 | 120
    B05-CE    | 20404 | 5890 | 29/08/2026 16:42  | pre  |             92.812 |  7.135 | 96.392 | 150
    B05-CE    | 20405 | 5891 | 29/08/2026 19:05  | pre  |             92.896 |  7.925 | 96.441 | 140
    B10-AC    | 20403 | 5892 | 29/08/2026 21:18  | pre  |             92.914 |  7.825 | 96.451 | 150
    B10-AC    | 20404 | 5893 | 29/08/2026 23:38  | pre  |             92.977 |  7.905 | 96.489 | 150
    B10-AC    | 20405 | 5894 | 30/08/2026 05:48  | POST |             93.013 |  5.830 | 96.445 | 140
    B10-CE    | 20403 | 5895 | 30/08/2026 10:41  | POST |             92.844 |  6.405 | 96.374 | 180
    B10-CE    | 20404 | 5896 | 31/08/2026 11:55  | POST |             92.916 |  8.215 | 96.433 | 160
    B10-CE    | 20405 | 5897 | 31/08/2026 18:36  | POST |             92.971 |  6.475 | 96.444 | 130
    B15-AC    | 20403 | 5898 | 31/08/2026 22:32  | POST |             93.052 |  8.390 | 96.520 | 160
    B15-AC    | 20404 | 5899 | 01/09/2026 04:02  | POST |             92.897 |  7.465 | 96.387 | 120
    B15-AC    | 20405 | 5900 | 01/09/2026 08:34  | POST |             92.999 |  7.775 | 96.468 | 170
    B15-CE    | 20403 | 5901 | 01/09/2026 13:15  | POST |             92.860 |  6.805 | 96.378 | 140
    B15-CE    | 20404 | 5902 | 01/09/2026 19:07  | POST |             92.651 |  7.855 | 96.302 | 130
    B15-CE    | 20405 | 5903 | 01/09/2026 22:56  | POST |             92.604 |  8.510 | 96.299 | 140
    CE20      | 20403 | 5904 | 02/09/2026 04:42  | POST |             92.836 |  6.975 | 96.379 | 130
    CE20      | 20404 | 5905 | 02/09/2026 08:32  | POST |             92.894 |  7.520 | 96.425 | 170
    CE20      | 20405 | 5906 | 02/09/2026 14:28  | POST |             92.394 |  8.910 | 96.218 | 110
    Wc-CTRL   | 20403 | 5907 | 02/09/2026 16:23  | POST |             92.794 |  7.545 | 96.389 | 130
    Wc-CTRL   | 20404 | 5908 | 02/09/2026 21:10  | POST |             92.880 |  7.985 | 96.441 | 150
    Wc-CTRL   | 20405 | 5909 | 03/09/2026 02:40  | POST |             92.881 |  7.660 | 96.434 | 140
```

## 7. Cross-dataset sign matrix — paired dF1 / dFPR vs the dataset control, best_f1 val_cal (mean pp; * = 95% CI excludes 0)
```
    Arm       |     cicids-quad     |     ciciot-quad     |  ciciot-quad-post   |      unswr-qsr      |     unswr-quad     
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    | F1 -0.04  FPR +0.02  | F1 +0.11  FPR -0.03  |          —          | F1 +0.02  FPR -0.02  | F1 +0.01  FPR +0.13 
    B05-CE    | F1 -0.05  FPR +0.04  | F1 -0.05  FPR -0.11  |          —          | F1 +0.00  FPR -0.08  | F1 +0.03  FPR +0.13 
    B10-AC    | F1 -0.00  FPR -0.00  | F1 +0.12* FPR -0.80  |          —          | F1 -0.02  FPR -0.02  | F1 +0.03  FPR +0.04 
    B10-CE    | F1 -0.02  FPR -0.04  | F1 +0.06  FPR -1.18* | F1 +0.06  FPR -1.18* | F1 -0.01  FPR -0.02  | F1 +0.04  FPR +0.13 
    B15-AC    | F1 +0.01  FPR -0.05  | F1 +0.13  FPR -1.14  | F1 +0.13  FPR -1.14  | F1 +0.03  FPR +0.00  | F1 +0.04  FPR +0.13 
    B15-CE    | F1 -0.04  FPR +0.02  | F1 -0.15  FPR -0.41  | F1 -0.15  FPR -0.41  | F1 +0.04  FPR +0.05  | F1 +0.05  FPR +0.13 
    CE20      | F1 +0.05  FPR -0.09  | F1 -0.14  FPR -0.04  | F1 -0.14  FPR -0.04  | F1 -0.07  FPR -0.01  | F1 -0.03  FPR -0.00 
    B34-CTRL  |          —          |          —          |          —          | F1 +0.03  FPR -0.03  | F1 +0.05  FPR +0.13 
```

Same matrix on best_fpr genome (FPR is the primary metric for that genome):
```
    Arm       |     cicids-quad     |     ciciot-quad     |  ciciot-quad-post   |      unswr-qsr      |     unswr-quad     
    ----------+---------------------+---------------------+---------------------+---------------------+---------------------
    B05-AC    | F1 -0.09  FPR +0.08* | F1 +0.09  FPR +0.19  |          —          | F1 +0.04  FPR -0.01  | F1 +0.04  FPR +0.14 
    B05-CE    | F1 -0.06  FPR +0.06  | F1 -0.08  FPR +0.21  |          —          | F1 +0.07* FPR -0.05  | F1 +0.07  FPR +0.14 
    B10-AC    | F1 -0.03  FPR +0.01  | F1 +0.10* FPR -0.54  |          —          | F1 +0.06  FPR -0.03  | F1 +0.04  FPR +0.14 
    B10-CE    | F1 -0.07  FPR +0.01  | F1 +0.04  FPR -0.70  | F1 +0.04  FPR -0.70  | F1 -0.02  FPR -0.03  | F1 +0.05  FPR +0.14 
    B15-AC    | F1 -0.03  FPR -0.01  | F1 +0.09  FPR +0.15  | F1 +0.09  FPR +0.15  | F1 +0.05  FPR -0.01  | F1 +0.06  FPR +0.14 
    B15-CE    | F1 -0.09  FPR +0.03  | F1 -0.16  FPR -0.01  | F1 -0.16  FPR -0.01  | F1 +0.08  FPR +0.06  | F1 +0.07  FPR +0.14 
    CE20      | F1 +0.01  FPR -0.07  | F1 -0.18  FPR +0.07  | F1 -0.18  FPR +0.07  | F1 -0.04  FPR -0.01  | F1 +0.03  FPR +0.14 
    B34-CTRL  |          —          |          —          |          —          | F1 -0.03  FPR +0.04  | F1 +0.07  FPR +0.14 
```

## Appendix A — per-run held-out val_cal values (GA final), all three genome types
```
    flow | name                                        | gens | bf1: F1     FPR    Acc    | bfpr: F1    FPR    Acc    | bfit: F1    FPR    Acc
    -----+---------------------------------------------+------+---------------------------+---------------------------+--------------------------
    5862 | cicids-quad-96b-B05-AC-r20403               |  140 | 99.547  0.120 99.714    | 99.467  0.185 99.663    | 99.540  0.121 99.710
    5863 | cicids-quad-96b-B05-AC-r20404               |   70 | 99.404  0.316 99.622    | 99.366  0.329 99.598    | 99.404  0.316 99.622
    5864 | cicids-quad-96b-B05-AC-r20405               |   60 | 99.338  0.258 99.582    | 99.330  0.273 99.576    | 99.325  0.302 99.573
    5865 | cicids-quad-96b-B05-CE-r20403               |  100 | 99.535  0.139 99.706    | 99.498  0.123 99.684    | 99.505  0.136 99.687
    5866 | cicids-quad-96b-B05-CE-r20404               |   90 | 99.374  0.308 99.604    | 99.367  0.307 99.599    | 99.364  0.305 99.597
    5867 | cicids-quad-96b-B05-CE-r20405               |   90 | 99.365  0.301 99.598    | 99.362  0.302 99.596    | 99.370  0.251 99.601
    5868 | cicids-quad-96b-B10-AC-r20403               |   90 | 99.515  0.171 99.693    | 99.517  0.143 99.695    | 99.531  0.171 99.703
    5869 | cicids-quad-96b-B10-AC-r20404               |   60 | 99.390  0.302 99.613    | 99.378  0.292 99.606    | 99.370  0.314 99.601
    5870 | cicids-quad-96b-B10-AC-r20405               |   90 | 99.503  0.146 99.686    | 99.444  0.126 99.649    | 99.457  0.133 99.657
    5871 | cicids-quad-96b-B10-CE-r20403               |   90 | 99.515  0.160 99.694    | 99.499  0.115 99.684    | 99.461  0.157 99.659
    5872 | cicids-quad-96b-B10-CE-r20404               |   60 | 99.398  0.281 99.619    | 99.394  0.273 99.616    | 99.393  0.268 99.616
    5873 | cicids-quad-96b-B10-CE-r20405               |  100 | 99.445  0.066 99.651    | 99.325  0.188 99.574    | 99.429  0.093 99.640
    5874 | cicids-quad-96b-B15-AC-r20403               |   80 | 99.426  0.205 99.637    | 99.408  0.213 99.626    | 99.412  0.201 99.629
    5875 | cicids-quad-96b-B15-AC-r20404               |  110 | 99.539  0.096 99.709    | 99.467  0.094 99.664    | 99.521  0.099 99.698
    5876 | cicids-quad-96b-B15-AC-r20405               |  130 | 99.479  0.162 99.671    | 99.466  0.206 99.663    | 99.475  0.168 99.668
    5877 | cicids-quad-96b-B15-CE-r20403               |   90 | 99.532  0.147 99.704    | 99.498  0.125 99.683    | 99.527  0.145 99.701
    5878 | cicids-quad-96b-B15-CE-r20404               |   70 | 99.377  0.264 99.606    | 99.264  0.260 99.535    | 99.310  0.261 99.564
    5879 | cicids-quad-96b-B15-CE-r20405               |   60 | 99.396  0.285 99.617    | 99.397  0.238 99.618    | 99.381  0.290 99.608
    5880 | cicids-quad-96b-CE20-r20403                 |   80 | 99.557  0.110 99.720    | 99.490  0.108 99.678    | 99.544  0.127 99.712
    5881 | cicids-quad-96b-CE20-r20404                 |   70 | 99.498  0.122 99.683    | 99.453  0.113 99.655    | 99.485  0.121 99.675
    5882 | cicids-quad-96b-CE20-r20405                 |   90 | 99.503  0.123 99.687    | 99.495  0.119 99.681    | 99.495  0.119 99.681
    5883 | cicids-quad-96b-Wa-CTRL-r20403              |  130 | 99.500  0.150 99.684    | 99.517  0.110 99.695    | 99.521  0.111 99.698
    5884 | cicids-quad-96b-Wa-CTRL-r20404              |  100 | 99.370  0.301 99.601    | 99.367  0.251 99.600    | 99.377  0.258 99.606
    5885 | cicids-quad-96b-Wa-CTRL-r20405              |  120 | 99.551  0.173 99.716    | 99.536  0.184 99.706    | 99.546  0.179 99.713
    5886 | ciciot-quad-96b-B05-AC-r20403               |  180 | 92.789  8.495 96.404    | 92.805  8.375 96.409    | 92.797  7.920 96.390
    5887 | ciciot-quad-96b-B05-AC-r20404               |  140 | 92.955  7.630 96.469    | 92.952  7.780 96.472    | 92.896  8.595 96.467
    5888 | ciciot-quad-96b-B05-AC-r20405               |  130 | 93.143  7.590 96.572    | 93.116  7.610 96.558    | 93.114  7.670 96.559
    5889 | ciciot-quad-96b-B05-CE-r20403               |  120 | 92.692  8.000 96.334    | 92.691  8.760 96.358    | 92.686  8.235 96.339
    5890 | ciciot-quad-96b-B05-CE-r20404               |  150 | 92.812  7.700 96.392    | 92.779  7.135 96.355    | 92.796  7.165 96.366
    5891 | ciciot-quad-96b-B05-CE-r20405               |  140 | 92.896  7.785 96.441    | 92.896  7.925 96.446    | 92.931  8.000 96.467
    5892 | ciciot-quad-96b-B10-AC-r20403               |  150 | 92.914  7.780 96.451    | 92.913  7.825 96.452    | 92.900  7.810 96.444
    5893 | ciciot-quad-96b-B10-AC-r20404               |  150 | 92.977  7.865 96.489    | 92.973  7.905 96.487    | 92.968  7.305 96.467
    5894 | ciciot-quad-96b-B10-AC-r20405               |  140 | 93.013  5.770 96.445    | 93.005  5.830 96.442    | 93.011  5.760 96.443
    5895 | ciciot-quad-96b-B10-CE-r20403               |  180 | 92.844  6.575 96.374    | 92.825  6.405 96.358    | 92.839  6.400 96.366
    5896 | ciciot-quad-96b-B10-CE-r20404               |  160 | 92.916  7.160 96.433    | 92.921  8.215 96.469    | 92.935  7.975 96.469
    5897 | ciciot-quad-96b-B10-CE-r20405               |  130 | 92.971  6.530 96.444    | 92.968  6.475 96.441    | 92.992  6.560 96.457
    5898 | ciciot-quad-96b-B15-AC-r20403               |  160 | 93.052  7.550 96.520    | 92.985  8.390 96.509    | 92.987  8.410 96.511
    5899 | ciciot-quad-96b-B15-AC-r20404               |  120 | 92.897  6.035 96.387    | 92.878  7.465 96.421    | 92.939  6.745 96.433
    5900 | ciciot-quad-96b-B15-AC-r20405               |  170 | 92.999  6.795 96.468    | 93.005  7.775 96.501    | 93.045  7.325 96.510
    5901 | ciciot-quad-96b-B15-CE-r20403               |  140 | 92.860  6.420 96.378    | 92.903  6.805 96.415    | 92.860  6.420 96.378
    5902 | ciciot-quad-96b-B15-CE-r20404               |  130 | 92.651  7.720 96.302    | 92.635  7.855 96.297    | 92.651  7.945 96.309
    5903 | ciciot-quad-96b-B15-CE-r20405               |  140 | 92.604  8.430 96.299    | 92.586  8.510 96.292    | 92.691  6.805 96.295
    5904 | ciciot-quad-96b-CE20-r20403                 |  130 | 92.836  6.880 96.379    | 92.788  6.975 96.355    | 92.836  6.880 96.379
    5905 | ciciot-quad-96b-CE20-r20404                 |  170 | 92.894  7.295 96.425    | 92.887  7.520 96.428    | 92.894  7.295 96.425
    5906 | ciciot-quad-96b-CE20-r20405                 |  110 | 92.394  9.515 96.218    | 92.369  8.910 96.183    | 92.372  9.375 96.201
    5907 | ciciot-quad-96b-Wc-CTRL-r20403              |  130 | 92.794  7.930 96.389    | 92.838  7.545 96.401    | 92.842  8.035 96.419
    5908 | ciciot-quad-96b-Wc-CTRL-r20404              |  150 | 92.880  8.060 96.441    | 92.884  7.985 96.441    | 92.887  8.190 96.449
    5909 | ciciot-quad-96b-Wc-CTRL-r20405              |  140 | 92.881  7.810 96.434    | 92.875  7.660 96.425    | 92.872  7.675 96.425
    5910 | unswr-qsr-64b-B05-AC-r20403                 |  100 | 94.351  0.580 99.145    | 94.306  0.571 99.140    | 94.184  0.566 99.125
    5911 | unswr-qsr-64b-B05-AC-r20404                 |   90 | 94.211  0.654 99.112    | 94.214  0.636 99.116    | 94.214  0.636 99.116
    5912 | unswr-qsr-64b-B05-AC-r20405                 |   90 | 94.450  0.529 99.167    | 94.496  0.593 99.162    | 94.491  0.564 99.167
    5913 | unswr-qsr-64b-B05-CE-r20403                 |   70 | 94.315  0.629 99.131    | 94.393  0.616 99.144    | 94.359  0.621 99.138
    5914 | unswr-qsr-64b-B05-CE-r20404                 |  100 | 94.325  0.453 99.164    | 94.324  0.477 99.160    | 94.335  0.457 99.165
    5915 | unswr-qsr-64b-B05-CE-r20405                 |  110 | 94.314  0.511 99.152    | 94.390  0.589 99.148    | 94.395  0.617 99.144
    5916 | unswr-qsr-64b-B10-AC-r20403                 |   90 | 94.349  0.620 99.137    | 94.406  0.664 99.137    | 94.349  0.620 99.137
    5917 | unswr-qsr-64b-B10-AC-r20404                 |   90 | 94.219  0.496 99.142    | 94.237  0.499 99.144    | 94.207  0.501 99.140
    5918 | unswr-qsr-64b-B10-AC-r20405                 |   80 | 94.317  0.648 99.128    | 94.447  0.573 99.159    | 94.420  0.559 99.158
    5919 | unswr-qsr-64b-B10-CE-r20403                 |   60 | 94.298  0.668 99.121    | 94.279  0.664 99.119    | 94.311  0.658 99.125
    5920 | unswr-qsr-64b-B10-CE-r20404                 |   90 | 94.363  0.547 99.152    | 94.346  0.549 99.150    | 94.373  0.551 99.153
    5921 | unswr-qsr-64b-B10-CE-r20405                 |  140 | 94.255  0.536 99.140    | 94.228  0.533 99.137    | 94.200  0.520 99.135
    5922 | unswr-qsr-64b-B15-AC-r20403                 |  100 | 94.312  0.547 99.145    | 94.306  0.543 99.145    | 94.279  0.533 99.143
    5923 | unswr-qsr-64b-B15-AC-r20404                 |   80 | 94.288  0.637 99.126    | 94.289  0.637 99.126    | 94.289  0.637 99.126
    5924 | unswr-qsr-64b-B15-AC-r20405                 |   60 | 94.437  0.643 99.145    | 94.470  0.633 99.152    | 94.409  0.677 99.135
    5925 | unswr-qsr-64b-B15-CE-r20403                 |   70 | 94.488  0.694 99.143    | 94.461  0.694 99.140    | 94.527  0.599 99.166
    5926 | unswr-qsr-64b-B15-CE-r20404                 |   60 | 94.231  0.621 99.121    | 94.314  0.639 99.129    | 94.297  0.595 99.135
    5927 | unswr-qsr-64b-B15-CE-r20405                 |   80 | 94.364  0.650 99.134    | 94.371  0.697 99.126    | 94.279  0.571 99.137
    5933 | unswr-qsr-64b-CE20-r20403                   |   70 | 94.332  0.583 99.142    | 94.265  0.557 99.137    | 94.298  0.541 99.145
    5934 | unswr-qsr-64b-CE20-r20404                   |   60 | 94.217  0.501 99.141    | 94.320  0.571 99.142    | 94.271  0.555 99.138
    5935 | unswr-qsr-64b-CE20-r20405                   |   60 | 94.199  0.700 99.102    | 94.213  0.686 99.106    | 94.198  0.713 99.099
    5936 | unswr-qsr-64b-Wb-CTRL-r20403                |   70 | 94.369  0.653 99.134    | 94.336  0.656 99.129    | 94.369  0.653 99.134
    5937 | unswr-qsr-64b-Wb-CTRL-r20404                |  120 | 94.296  0.625 99.129    | 94.232  0.595 99.126    | 94.232  0.595 99.126
    5938 | unswr-qsr-64b-Wb-CTRL-r20405                |  110 | 94.286  0.545 99.142    | 94.336  0.586 99.142    | 94.286  0.545 99.142
    5928 | unswr-qsr-64b-B34-CTRL-r20401               |  100 | 94.415  0.608 99.148    | 94.446  0.685 99.139    | 94.429  0.619 99.148
    5929 | unswr-qsr-64b-B34-CTRL-r20402               |   60 | 94.429  0.598 99.152    | 94.410  0.604 99.148    | 94.324  0.667 99.125
    5930 | unswr-qsr-64b-B34-CTRL-r20403               |   60 | 94.274  0.609 99.129    | 94.238  0.644 99.118    | 94.257  0.623 99.124
    5931 | unswr-qsr-64b-B34-CTRL-r20404               |   60 | 94.411  0.584 99.152    | 94.397  0.595 99.148    | 94.411  0.584 99.152
    5932 | unswr-qsr-64b-B34-CTRL-r20405               |   60 | 94.348  0.543 99.151    | 94.181  0.730 99.094    | 94.349  0.551 99.150
    5939 | unswr-quad-64b-B05-AC-r20403                |   60 | 93.449  1.118 98.912    | 93.449  1.118 98.912    | 93.449  1.118 98.912
    5940 | unswr-quad-64b-B05-AC-r20404                |   60 | 93.497  1.119 98.919    | 93.479  1.118 98.916    | 93.484  1.119 98.917
    5941 | unswr-quad-64b-B05-AC-r20405                |   90 | 93.402  1.118 98.905    | 93.411  1.127 98.904    | 93.412  1.119 98.906
    5942 | unswr-quad-64b-B05-CE-r20403                |   60 | 93.477  1.119 98.916    | 93.477  1.119 98.916    | 93.481  1.119 98.916
    5943 | unswr-quad-64b-B05-CE-r20404                |   90 | 93.453  1.118 98.912    | 93.481  1.119 98.916    | 93.489  1.119 98.917
    5944 | unswr-quad-64b-B05-CE-r20405                |   90 | 93.492  1.118 98.918    | 93.489  1.119 98.917    | 93.485  1.119 98.917
    5945 | unswr-quad-64b-B10-AC-r20403                |   90 | 93.446  0.834 98.970    | 93.419  1.118 98.907    | 93.473  0.816 98.977
    5946 | unswr-quad-64b-B10-AC-r20404                |   60 | 93.474  1.118 98.916    | 93.448  1.118 98.912    | 93.441  1.118 98.911
    5947 | unswr-quad-64b-B10-AC-r20405                |   60 | 93.489  1.119 98.917    | 93.486  1.120 98.917    | 93.486  1.120 98.917
    5948 | unswr-quad-64b-B10-CE-r20403                |   60 | 93.480  1.119 98.916    | 93.471  1.119 98.915    | 93.477  1.119 98.916
    5949 | unswr-quad-64b-B10-CE-r20404                |   60 | 93.462  1.118 98.914    | 93.483  1.118 98.917    | 93.483  1.118 98.917
    5950 | unswr-quad-64b-B10-CE-r20405                |   60 | 93.501  1.119 98.919    | 93.429  1.119 98.909    | 93.494  1.119 98.918
    5951 | unswr-quad-64b-B15-AC-r20403                |   60 | 93.454  1.119 98.912    | 93.454  1.119 98.912    | 93.463  1.119 98.914
    5952 | unswr-quad-64b-B15-AC-r20404                |   60 | 93.489  1.119 98.917    | 93.482  1.120 98.916    | 93.485  1.119 98.917
    5953 | unswr-quad-64b-B15-AC-r20405                |   60 | 93.485  1.119 98.917    | 93.485  1.119 98.917    | 93.485  1.119 98.917
    5954 | unswr-quad-64b-B15-CE-r20403                |   60 | 93.497  1.119 98.919    | 93.494  1.119 98.918    | 93.494  1.119 98.918
    5955 | unswr-quad-64b-B15-CE-r20404                |   60 | 93.488  1.121 98.917    | 93.488  1.121 98.917    | 93.481  1.123 98.916
    5956 | unswr-quad-64b-B15-CE-r20405                |   60 | 93.497  1.119 98.919    | 93.445  1.118 98.911    | 93.494  1.119 98.918
    5962 | unswr-quad-64b-CE20-r20403                  |   90 | 93.338  0.717 98.979    | 93.441  1.127 98.909    | 93.449  1.127 98.910
    5963 | unswr-quad-64b-CE20-r20404                  |   90 | 93.420  1.119 98.907    | 93.405  1.123 98.904    | 93.420  1.119 98.907
    5964 | unswr-quad-64b-CE20-r20405                  |  150 | 93.484  1.119 98.917    | 93.471  1.119 98.915    | 93.488  1.119 98.917
    5965 | unswr-quad-64b-Wb-CTRL-r20403               |  140 | 93.406  0.725 98.987    | 93.304  0.707 98.977    | 93.406  0.725 98.987
    5966 | unswr-quad-64b-Wb-CTRL-r20404               |   60 | 93.488  1.119 98.917    | 93.480  1.119 98.916    | 93.484  1.119 98.917
    5967 | unswr-quad-64b-Wb-CTRL-r20405               |  110 | 93.429  1.119 98.909    | 93.446  1.119 98.911    | 93.442  1.119 98.911
    5957 | unswr-quad-64b-B34-CTRL-r20401              |   60 | 93.494  1.119 98.918    | 93.498  1.119 98.919    | 93.489  1.119 98.917
    5958 | unswr-quad-64b-B34-CTRL-r20402              |   60 | 93.462  1.118 98.914    | 93.409  1.117 98.906    | 93.470  1.118 98.915
    5959 | unswr-quad-64b-B34-CTRL-r20403              |   60 | 93.484  1.119 98.917    | 93.484  1.119 98.917    | 93.480  1.119 98.916
    5960 | unswr-quad-64b-B34-CTRL-r20404              |   60 | 93.497  1.119 98.919    | 93.496  1.118 98.919    | 93.494  1.119 98.918
    5961 | unswr-quad-64b-B34-CTRL-r20405              |   60 | 93.496  1.118 98.919    | 93.452  1.123 98.911    | 93.464  1.123 98.913
```
