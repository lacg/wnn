# IDS-56 unswr: why min-share vote (`ms01`) settles on tiny genomes

Written 10/11/2026 (ids-security agent). Diagnosis only: no flow was queued, paused, cancelled, resumed or
reordered. The worker (ABI 28) was live throughout. Probes ran at `nice -n 10` through the worker's own construction path
(`IdsEvaluatorFactory` -> `IDSCacheWrapper`). The rule was switched with `set_write_rule`. Training and scoring used
the Rust `evaluate_multiclass_at_thresholds` and `export_trained`. There is no Python reimplementation of any network
logic. Scratch scripts are in `/tmp/ids56r/` (probe3.py, probe4.py, conflict.py, grid.py) and are not committed.

**Fidelity check:** every production genome retrained offline reproduces its DB `margin_val_cal` TEST number:
- 7501 lattice: 94.17 / 0.19
- 7503 ms01: 93.78 / 1.02
- 7504 ms01: 93.50 / 1.12

**Evidence base:**
- Only seed r20419 has finished (flows 7501–7504). 7505 has finished its grid, and its GA is running.
- Every claim below is n=1 seed. The offline probes cover 20 genomes per encoding: 4 production genomes, plus nested
  subsets of the lattice GA genome.

**Reading convention:**
- **VAL** = threshold fit on the 10% VAL partition and scored on VAL. This is the read used to choose anything.
- **TEST** = production `margin_val_cal` (τ fit on VAL, scored on TEST). It is report-only.
- Grid and GA landscapes use the k-fold CV search metric from `genome_evaluations`. That is what the optimizer saw. It
  is never reported as a result.

## 0. Data facts (measured)

```
dataset | train rows | attack share | class weights (balance_classes, max//count) | input bits
--------+------------+--------------+---------------------------------------------+---------------------
unswr   | 1,267,407  | 3.81%        | benign 1 : attack 25                         | 1104 encoff / 559 encon
unswt   |   175,341  | 68.06%       | benign 2 : attack 1                          | 288 encoff / 161 encon
```

The vote tally is **class-weighted**: an own-class visit adds +(1−θ)·w and an other-class visit −θ·w. A benign-cluster
cell is therefore cleared when the RAW attack share of its visits reaches the value below. This conversion is
arithmetic on the rule (`vote_share.rs`) and the measured weights.

```
θ     | unswr: benign cell cleared at raw attack share ≥ | unswt: same
------+--------------------------------------------------+------------
0.02  | 66.2%                                            | 99.0%
0.04  | 49.0%                                            | 98.0%
0.10  | 26.5%                                            | 94.7%
0.50  |  3.8%                                            | 66.7%
```

So the same θ=0.1 means very different things on the two datasets:
- On unswt it clears benign cells only in addresses that are about 95% attack.
- On unswr it clears benign cells in addresses where **benign is still the 73% majority**.

To match unswt's raw cut, unswr would need θ ≈ 0.002, which is below the 1/100 grid.

## 1. Q1: grid landscape. Larger is NOT worse under the vote; it is FLAT.

Grid CV `f1_macro` by total neurons, bits ≥ 28, distinct grid genomes. Mean [min–max] is in %.

```
flow (arm)              | 10n                 | 100n          | 200n          | 300n          | 400n          | 500n
------------------------+---------------------+---------------+---------------+---------------+---------------+--------------
7501 lattice encoff     | 93.15 [92.97-93.29] | 93.56         | 93.68         | 93.77         | 93.87         | 93.91 [93.74-94.02]
7505 lattice encoff r20 | 93.27               | 93.56         | 93.67         | 93.78         | 93.83         | 93.89
7502 lattice encon      | 92.39 [89.30-93.31] | 93.45         | 93.55         | 93.70         | 93.76         | 93.80
7503 ms01 encoff        | 93.00 [92.61-93.30] | 93.36         | 93.41         | 93.40         | 93.37         | 93.38 [93.19-93.44]
7504 ms01 encon         | 90.27 [88.24-92.36] | 90.70         | 90.97         | 91.04         | 90.62         | 90.74 [88.90-91.69]
```

- **Lattice** improves steadily with size: +0.75 pp from 10n to 500n, with the same shape on both seeds.
- **ms01-encoff** is flat at about 93.4 from 100n upward.
- **ms01-encon** is flat at about 90.3–91.0 but noisy: one genome draw varies by about ±1.5 pp.
- The 10n winner of 7504 (5n/cluster × 32b) is the single best draw, 92.36. That is a winner's-curse pick from a high-variance
  cell of the grid, not a size advantage. ms01-encon's grid CV sits about 2.7 pp below its own held-out (93.06 TEST for
  that genome). That gap was not investigated.

## 2. Q2: mechanism. The hypothesis is refuted on attack cells and confirmed on benign cells.

### 2a. MEASURED: the vote clears BENIGN cells, almost never attack cells

These are the TRUE cells cleared relative to lattice on the same genome: the 7501 lattice-GA genome and its nested
subsets (first k neurons per cluster), encoff, trained under the weighted vote.

```
genome (total n)    | θ=0.02 ben / att | θ=0.04 ben / att | θ=0.10 ben / att | θ=0.20 ben / att | θ=0.50 ben / att
--------------------+------------------+------------------+------------------+------------------+-----------------
first5    (10n)     | 2.2% / 0.06%     | 3.5% / 0.06%     | 4.7% / 0.06%     | 4.9% / 0.17%     | 5.0% / 0.28%
first25   (50n)     | 2.1% / 0.03%     | 3.4% / 0.03%     | 4.4% / 0.05%     | 4.7% / 0.10%     | 4.8% / 0.40%
first100 (200n)     | 2.4% / 0.03%     | 4.0% / 0.03%     | 5.3% / 0.05%     | 5.6% / 0.08%     | 5.8% / 0.41%
full GA  (492n)     | 2.5% / 0.03%     | 4.1% / 0.03%     | 5.5% / 0.05%     | 5.8% / 0.10%     | 6.0% / 0.42%
```

- Attack-cluster cells survive because each attack visit carries weight 25. A cell would need more than 225 benign
  visits per attack visit to be cleared at θ=0.1. So "attack recall is lost as networks grow" is **refuted**: TPR goes
  UP under the vote (§2c).
- The benign fraction cleared grows only slightly with size (4.7% → 5.5%).
- Class-weighted share distribution versus unswt, θ=0.1 on lattice genomes:
  - unswt clears benign 5.0% / attack 1.5% (encoff) and 4.1% / 1.0% (encon).
  - unswr clears benign 5.5% / attack 0.05%.
  - The *fraction* of benign cells cleared is similar on both datasets. The *raw* address mix behind it is not (§0):
    on unswr the cleared cells are benign-majority addresses.

### 2b. MEASURED: a size-invariant attractor at F1 93.5 / FPR 1.12 / TPR 99.9

VAL read (τ fit on VAL), encoff, F1 / FPR / TPR in %:

```
genome (total n)     | lattice              | θ=0.02               | θ=0.04               | θ=0.10               | θ=0.50
---------------------+----------------------+----------------------+----------------------+----------------------+---------------------
first5    (10n)      | 93.38 / 1.14 / 99.9  | 94.36 / 0.57 / 92.0  | 94.72 / 0.66 / 95.2  | 93.64 / 1.03 / 98.8  | 93.49 / 1.12 / 99.9
first10   (20n)      | 93.45 / 1.10 / 99.2  | 94.67 / 0.43 / 90.3  | 94.73 / 0.71 / 96.3  | 93.51 / 1.12 / 99.9  | 93.51 / 1.12 / 99.9
first25   (50n)      | 93.52 / 1.05 / 98.7  | 94.55 / 0.40 / 89.2  | 94.69 / 0.52 / 92.2  | 93.50 / 1.12 / 99.9  | 93.50 / 1.12 / 99.9
first50  (100n)      | 93.60 / 1.02 / 98.2  | 94.62 / 0.38 / 89.1  | 94.71 / 0.62 / 94.4  | 93.75 / 0.70 / 92.5  | 93.51 / 1.12 / 99.9
first100 (200n)      | 93.74 / 1.00 / 98.4  | 94.67 / 0.42 / 90.1  | 94.76 / 0.52 / 92.5  | 93.64 / 0.70 / 92.1  | 93.51 / 1.12 / 99.9
first175 (350n)      | 94.60 / 0.16 / 84.5  | 94.74 / 0.43 / 90.5  | 94.78 / 0.48 / 91.7  | 93.62 / 0.72 / 92.4  | 93.51 / 1.12 / 99.9
full GA  (492n)      | 94.63 / 0.18 / 84.9  | 94.73 / 0.43 / 90.5  | 94.80 / 0.51 / 92.3  | 93.51 / 1.12 / 100.0 | 93.51 / 1.12 / 100.0
```

encon shows the same shape. Its lattice-GA genome is [245 benign, 5 attack]:
- lattice reads 93.34–93.49 below 180n and 93.83/0.19 at 180n and 250n
- θ=0.1 reads 93.42–93.45 / 1.11–1.13 at every size
- θ=0.02 reads 94.26–94.79

- Two operating points exist:
  - **A** = high recall: TPR ≈ 99.9, FPR ≈ 1.1, F1 ≈ 93.5.
  - **B** = low FPR: TPR ≈ 85, FPR ≈ 0.18, F1 ≈ 94.6.
- Under **lattice**, B appears only with a big benign cluster: between 200n and 350n (encoff), and at ≥ 175 benign
  neurons (encon). Below that, lattice also sits on A. That is why lattice's grid and GA reward size.
- Under **θ ≥ 0.1**, every size from 10n to 492n sits on A. Size cannot move the score, so the landscape is flat.
- At the 3.8% base rate, F1 is dominated by FPR. One extra 0.1 pp FPR is about 150 benign rows, which equals 2.5% of the
  attack count.
- **The vote's benign clearing destroys point B.** The benign rows that share an address with ≥ 26.5% attack visits
  lose benign support at every neuron. They then score like attacks no matter how many neurons vote.
- Full-vector duplicates with mixed labels explain only part of the 1.1%. Measured with numpy on the encoded rows:
  0.18% of VAL benign rows sit in a train vector whose θ=0.1 outcome is "attack only". The rest of the 1.1% are
  per-neuron address collisions (**inferred**; per-address visit counts are not exported).

### 2c. MEASURED: is θ=0.1 simply too high? Partly. The real lever is the class weighting.

- **Lower θ (still weighted):** θ=0.02–0.04 escapes the attractor at every size, giving 94.4–94.8 VAL. But it buys F1
  with recall, not FPR. FPR stays at 0.4–0.7, versus lattice's 0.18 at full size. On this genome, θ 0.02–0.04 roughly
  ties lattice on F1 (+0.1–0.2 VAL) at 2–4× the FPR.
- **Unweighted tally:** the same genomes with `balance_classes=false`, which affects only the vote tally. Lattice is
  bit-identical: 94.63/0.18 under both settings.

The unweighted read (VAL F1 / FPR / TPR in %; TEST F1 / FPR in brackets):

```
genome (n)        | lattice                       | nobal θ=0.01                  | nobal θ=0.02                  | nobal θ=0.05                  | nobal θ=0.10                  | nobal θ=0.50
------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------
7501 GA (492n)    | 94.63/0.18/84.9 [94.17/0.19]  | 94.90/0.20/86.2 [94.48/0.21]  | 95.03/0.22/87.0 [94.67/0.22]  | 95.08/0.30/88.9 [94.71/0.30]  | 95.11/0.36/90.3 [94.69/0.37]  | 94.82/0.56/93.6 [94.63/0.58]
first100 (200n)   | 93.74/1.00/98.4 [93.74/0.99]  | 94.55/0.16/84.2 [93.99/0.17]  | 94.80/0.18/85.5 [94.32/0.19]  | 94.83/0.31/88.4 [94.54/0.31]  | 94.85/0.45/91.3 [94.58/0.45]  | 94.73/0.58/93.5 [94.62/0.58]
first10   (20n)   | 93.45/1.10/99.2 [93.50/1.08]  | 93.46/1.10/99.3               | 93.46/1.10/99.3               | 93.88/0.12/81.3 [93.46/0.12]  | 94.07/0.21/83.6 [93.80/0.20]  | 94.51/0.55/92.2 [94.34/0.55]
7503 ms01 GA (98n)| 93.46/1.12/99.8               | 93.46/1.12/99.8               | 93.46/1.06/98.6               | 93.46/1.06/98.6               | 93.73/0.27/83.7 [93.37/0.24]  | 94.70/0.59/93.7 [94.64/0.58]
```

- The unweighted vote at θ=0.01–0.02 keeps lattice's FPR (0.20–0.22 vs 0.18) and adds recall (+1.3 to +2.1 pp TPR).
  VAL F1 is 94.90–95.03 vs 94.63; TEST is 94.48–94.67 vs 94.17.
- At this read it is a **Pareto gain over lattice** on both VAL and TEST, for one genome and one seed.
- Size matters again: 20n stays on attractor A until θ=0.05, while 200n and 492n leave it at θ=0.01.
- The original hypothesis does bite in this regime, but only mildly. Unweighted θ=0.5 clears 13–15% of attack cells.
  θ ≤ 0.1 clears less than 1%.
- ⚠️ θ and the weighting were chosen by looking at VAL across 6 values on 3–4 genomes of ONE seed. This is a hypothesis,
  not a result, until a pre-registered production arm runs (§5).

## 3. Q3: is the FPR loss the small genome or the rule? Both are sufficient on their own.

TEST `margin_val_cal`, same genome retrained under the other rule:

```
genome                         | under lattice     | under ms01 (θ=0.1 weighted)
-------------------------------+-------------------+----------------------------
7501 lattice GA 492n (encoff)  | 94.17 / 0.19      | 93.51 / 1.12
7502 lattice GA 250n (encon)   | 93.89 / 0.19      | 93.40 / 1.11
7503 ms01 GA 98n (encoff)      | 93.48 / 1.12      | 93.78 / 1.02
7504 ms01 GA 16n (encon)       | 93.54 / 1.11      | 93.50 / 1.12
```

- The **rule alone** costs the FPR: a large lattice genome under the weighted vote lands at 1.1.
- **Small size alone** also costs it: an ms01 tiny genome under lattice is still at 1.1.
- The low-FPR point needs **both** a large benign cluster **and** no aggressive benign clearing.
- So the tiny genomes are a consequence of the rule, not the cause of the FPR. Forcing them larger (a neuron floor) would
  not recover 0.19 (§2b: 492n under θ=0.1 = 1.12).

## 4. Q4: fitness and search

- Fitness for unswr (WbCTRL-desir): `fitness_calculator=harmonic_rank` with `fitness_aggregation=desirability`.
  Weights are CE 0.10, F1 0.35, FPR 0.35, Acc 0.20, and `fitness_ce_anchor_normalized=0.2128`. The desirability
  combine (`core/fitness.rs desirability_combine_flat`) has **no size or memory term**. Size is a by-product.
- The ms01 GAs are **stuck from generation 1**. Elite CV values per generation (`genome_evaluations`, role=elite):

```
flow | gen 1                     | gen 5                     | gen 10                   | final
-----+---------------------------+---------------------------+--------------------------+---------------------------
7503 | F1 93.44, 100-500n (269)  | 93.44, 98-101n            | 93.44, 100-102n          | g60: 93.44-93.45, 94-97n
7504 | F1 92.52 max, 10-252n     | 93.10 max, 10-211n        | 93.31, 11-12n            | g60: 93.44, 15-16n
7501 | F1 94.04, 394-500n        | 94.35, 487-499n           | 94.62, 493-496n          | g90: 94.92, 491-499n
7502 | F1 93.79, 10-400n         | 93.37, 10-250n            | 93.39, 250n              | g110: 94.45, 250n
```

- In 7503, every elite had the same F1, FPR and accuracy (93.44 / 1.13 / 98.90), so CE (weight 0.10) was the only
  term that differed. CE moved from 0.3255 to 0.3241 and happened to be lowest on the ~100n genome. Elites collapsed
  500n → 100n within 5 generations.
- In 7504, the small encon genomes had both the best draws and the lowest CE.
- The best CV F1 moved +0.01 pp (7503) and +0.9 pp (7504, all of it before gen 10). Patience then stopped both at 60
  generations, versus 90 and 110 for lattice, which kept climbing.
- **The search is not "rewarding small". It is on a plateau (attractor A) where size is invisible to F1/FPR/Acc, and a
  0.001 CE tie-break picks the survivor.**

## 5. Verdict

On unswr the min-share vote clears **benign** cells, not attack cells. `balance_classes` weights the 3.8% attacks
25:1 inside the vote tally, so θ=0.1 clears a benign cell wherever attacks are just 26.5% of its raw visits. On unswt
the same θ needs 95% (§0). The benign rows behind those addresses lose benign support at every neuron. Every genome,
from 10 to 492 neurons, collapses onto one high-recall operating point (TPR 99.9 / FPR 1.12 / F1 93.5). On the
low-base-rate set, F1 is FPR-dominated, so that point is worse than lattice's low-FPR point (TPR 85 / FPR 0.18 /
F1 94.6). Lattice reaches its point only with a large benign cluster. Because the vote makes size invisible to
F1/FPR/Acc, the grid picks noise and the GA's CE tie-break drifts to tiny genomes and stalls at the patience floor. The
FPR loss is caused by the rule (a large genome under the vote = 1.12), and the tiny genomes are its symptom. The vote
itself is not unsuitable for low-base-rate data: the **class-weighted tally** is. With an unweighted tally at θ 0.01–0.02,
the lattice 492n genome keeps FPR 0.20–0.22 and gains 1.3–2.1 pp TPR (VAL F1 94.90–95.03 vs 94.63; one seed, chosen on
VAL, hypothesis).

## 6. Recommendations (nothing queued; each with a cheap test)

1. **Do not adopt weighted ms01 on heavily imbalanced data (unswr).** Any set with a large class-weight ratio needs its
   own check. On neto_full benign is the 2.35% minority, so the effect would mirror onto attack cells (inferred, not
   measured). The pre-registered A/B (7501–7520) will most likely confirm what r20419 shows: F1 about −0.4 pp and FPR
   +0.8 to +0.9 pp. Let it finish; it is the honest record.
   *Test:* none needed. Read the remaining 4 seeds as they land.
2. **Unweighted vote tally on unswr** (`ids_write_rule=vote, ids_vote_share=0.1, balance_classes=false`). In a BINARY
   flow, lattice is bit-identical with or without `balance_classes` (measured), so the param acts on the vote tally.
   It also changes the test cache key, which is expected. Before queueing, check its other consumers for
   side effects: grep `class_weights` in the worker crate. If that is clean, this needs no code change and no wheel
   swap, only a flow param.
   *Cheap tests, in order:*
   (a) Offline, about 10 min: repeat probe4 on the 7505/7509 lattice GA genomes as they land. This checks the gain
       across seeds without production cost.
   (b) Pre-registered production arm `ms01nobal` and `ms002nobal` × encoff, r20419–23, cloned from 7501. Adopt only if F1
       is paired ≥ 4/5 AND FPR does not rise by more than 0.05 pp.
   (c) Offline sanity on unswt: weights there are 2:1, so the no-balance effect should be small. Run probe4 against
       7481/7482 genomes before generalising the param.
   Cost of (b) is about 10 × 1–5 h. The grid/GA will not stall on attractor A, so expect lattice-like run times.
3. **Base-rate-scaled θ in the weighted tally (θ 0.02–0.04): only as a fallback.** It escapes the attractor at every size
   (VAL 94.4–94.8) but FPR is 0.4–0.7, 2–4× lattice. It is dominated by recommendation 2 on this evidence. A cleaner design
   if weighting must stay is to tally on raw counts and keep `balance_classes` for the trainer only. That is a Rust
   change and a worker swap, so do not do it before the paper.
   *Test:* already measured offline (§2b). No production arm unless recommendation 2 fails.
4. **No min-neurons floor.** A 492n genome under weighted θ=0.1 lands at the same 93.51 / 1.12 as a 10n one (§2b). A floor
   only adds cost.
   *Test:* already measured (§2b, §3).
5. **Process note.** When a rule makes F1/FPR/Acc size-invariant, the desirability CE term (0.10) silently decides
   genome size. A one-line watch for this:
   - flag any GA whose elite best-F1 moves less than 0.05 pp over its first 20 generations
   - 7503 moved +0.00 and 7504 +0.84 then flat
   *Test:* a SQL check on `iterations` (no code).

Hand-off: whether the §2c unweighted gain is real at n=5, and how to pre-register θ for recommendation 2b, goes to
`experiment-design` before anything is queued.
