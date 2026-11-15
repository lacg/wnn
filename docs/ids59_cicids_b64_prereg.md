# IDS59: pre-registration of A1 (64 bits) on cicids

**FROZEN 15/11/2026 by ids-security, on Luiz's approval of the same day** (Plane IDS-58 comment 22f7e9d7: "IDS59-cicids-b64
approved (3 flows, clones of 7531/7533/7535 with max_bits 64; rule frozen before data)"). This file was written and
committed **before any IDS59 flow existed** and **without reading the TEST rows of 7533 / 7535**. 7533 was running and
7535 queued at the time of writing. Nothing in §3-§5 may change after the first IDS59 flow is created.

Vocabulary (`docs/terminology_train_val_test.md`): **train** fits and drives the search (5-fold CV); **VAL** tunes
(τ calibration, gates); **TEST** is the report.

## 1. Question, hypothesis, falsifier

- **Question.** Does the unswr accuracy configuration A1 (lattice, `max_bits` 34 → 64; IDS-57, ADOPT) carry over to
  CICIDS2017?
- **Why this is a fair test.** All cicids lattice GA genomes on file sit at the 34-bit cap (IDS-47 `best_f1` finals:
  28-34 bits). The lever A1 pulled is binding here too. ciciot is not tested: its recipe already runs `max_bits` 100
  and its GA genomes use 69-100 bits.
- **H1.** Widening the cap to 64 does not make cicids worse on any seed, and improves it on average.
- **What "improve" means on cicids.** TEST errors on cicids are almost all false alarms (IDS-47: ≈1,500 FP vs ≈65 FN,
  TPR 99.88%; corr(F1, FPR) = −1.00 over the 9 IDS-47 finals). Macro-F1 and FPR are one measurement here, and an
  F1 gain essentially *is* an FPR reduction. The rule below therefore counts a seed as a win if **either** metric
  improves by its threshold, and as a loss if **either** worsens by its threshold.
- **Falsifier.** REJECT under §4.

## 2. Arms, controls, seeds, prior evidence

- **Arm (3 new flows).** Clone the lattice control of the same seed and change exactly one key, `max_bits: 34 → 64`.
  Name: `IDS59-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-lattice-b64-encon-r<seed>`.
- **Controls (not re-flown):** 7531 (r20419, completed, ABI 29), 7533 (r20420), 7535 (r20421) = IDS56C cicids lattice
  encon carry-over (CE20 zscore, `blend_neuron`, `random_3way`, BINARY 2×250, 96 input bits, dedupe on, `train_only`
  ranking, rate 0.25, k-fold 5×5, 250 generations, patience 5, population 50).
- **Pairing.** Same seed = same folds and starting point. The grid changes with the cap (part of the treatment); the
  GA diverges from generation 1.
- **Prior evidence (VAL only, not a result).** 15/11, `/tmp/ids59/cicids_ladder.py`, settings of 7531, random 2×250
  genomes, 5 draws, lattice, τ fitted and scored on VAL: 64 vs 34 bits dF1 **+0.129 ± 0.063** (5/5 > 0), dFPR
  **−0.095**; 48 bits +0.107; 80 bits +0.167. Cells 0.64M → 2.59M. On unswr, A1's TEST effect was about 0.64× its
  VAL gate, so a TEST effect of roughly +0.08 F1 / −0.06 FPR is the working expectation.
- **Disclosure.** 7531's TEST row (99.434 / 0.419) was read on 15/11 during the IDS58 follow-up, before this rule was
  written. 7533 / 7535 TEST rows were not read. The cap value 64 is inherited from unswr, not tuned on cicids.

## 3. Metric and per-seed definitions

- **Source.** `test_summaries.threshold_metadata`, GA phase (`ga_neurons`), `genome_type='best_f1'`,
  `test_point='final'`, mode `margin_val_cal` (keys `macro_f1`, `fpr`), stored precision, in pp. Never
  `iterations.best_f1`. "F1" = macro-F1.
- **Noise scale.** IDS-47 cicids lattice finals (N=9, IDS56UC prereg §3.1): σF1 = 0.057, σFPR = 0.046.
  TEST support 227,132 benign / 55,656 attack; 0.01 pp FPR = 23 benign rows.
- Per seed, arm − control: dF1, dFPR.
- **Loss:** dF1 ≤ −0.02 **or** dFPR ≥ +0.02. (0.35 σF1 and 0.43 σFPR; +0.02 pp FPR = 45 benign rows.)
- **Win:** not a loss, **and** (dF1 ≥ +0.02 **or** dFPR ≤ −0.02).
- **Tie:** neither.
- **Severe loss:** dF1 ≤ −0.06 **or** dFPR ≥ +0.05 (≈1 σ).

## 4. Decision rule (n=3: r20419-21; first match decides)

Non-inferiority in the spirit of Luiz's IDS58 ruling of 15/11 ("not worse: a tie or better"), with one extra
requirement because this change costs about 4× the cells:

1. **REJECT** if any seed is a severe loss, **or** mean dF1 ≤ 0, **or** mean dFPR ≥ 0.
2. **ADOPT** if **no seed loses**, **and** wins ≥ 2/3, **and** mean dF1 > 0, **and** mean dFPR < 0.
3. **AMBIGUOUS** otherwise. An extension to r20422-23 needs 4 new flows (controls do not exist) and Luiz's go; at
   n=5 the same rule applies with "wins ≥ 4/5", else REJECT. No further extension.

**Label on ADOPT** (reported, does not change the verdict):
- **clear:** wins 3/3 and mean dF1 ≥ +0.06 (the IDS56UC cicids ADOPT bar, 1.02 σF1);
- **non-inferior:** otherwise. Wording: "on cicids, 64 bits are not worse on any seed and lower false alarms on
  average"; never "beats".

**Sensitivity line (printed, never deciding):** the unswr A1 price rule rescaled to cicids noise (win ≥ +0.15 → +0.06,
ADOPT mean ≥ +0.40 → +0.155, i.e. 2.7 σF1).

**What ADOPT means.** A1 becomes the cicids accuracy row (as on unswr); the 34-bit lattice stays the cicids small-model
row. The cross-dataset sentence "the 64-bit cap carries over" covers unswr + cicids only; ciciot is out of scope.

## 5. Guards

- **ABI.** Print `worker_abi` for every row. Each arm must share its control's ABI (7531 is ABI 29). If a pair
  straddles an ABI change, stop and ask Luiz (no automatic re-fly).
- **Width.** Print the arm's per-cluster bit widths. If every arm GA neuron is ≤ 34 bits, the arm did not test H1:
  label **INCONCLUSIVE (cap not used)**, whatever the numbers.
- **Cost.** Alert only: if an arm's `ga_neurons` is still running 20 h after its start, alert Luiz. Never pause or
  cancel a running flow.
- **Partial reads.** An arm's TEST row may be seen when its pair completes; it decides nothing until all three pairs
  are final.

## 6. Operating characteristics (simulated, a model, not a measurement)

Model: dF1 per seed ~ N(μ, √2·0.057), dFPR = −(0.046/0.057)·dF1 (corr −1), n=3, 200k draws, seed 59.

```
true dF1 | P(ADOPT) this rule | no-loss only (IDS58-style) | wins 3/3 required
---------+--------------------+----------------------------+------------------
 0.00    |  0.16              |  0.21                      |  0.07
+0.03    |  0.33              |  0.39                      |  0.17
+0.06    |  0.54              |  0.59                      |  0.33
+0.08    |  0.68              |  0.71                      |  0.46
+0.13    |  0.90              |  0.91                      |  0.77
```

- The null false-adopt rate (16%) is higher than IDS56U's 7%. That is the price of a non-inferiority rule at n=3 and
  is disclosed with the verdict. The "clear" label (§4) is what carries the stronger claim.
- The σ used is the 34-bit lattice spread. At 64 bits unswr A1 was far quieter (SD 0.003); if cicids behaves the same,
  power is higher than shown.

## 7. Readout obligations

Per seed and mean ± SD: macro-F1, FPR, attack-F1, TPR, confusion counts, cells, neurons, bit widths, GA generations
and hours, `worker_abi`; wins / ties / losses; the verdict, label and sensitivity line; the Rule-7 tables (all stored
modes × 5 genome types, Grid vs GA) and the FPR-banded Pareto view. Only the §3 cell decides. Never quote the best of
the 50 cells.

## 8. Flows

`scripts/queue_ids59_cicids_b64.py` asserts the pinned recipe on each control, the one-key diff, and 2 experiments per
flow, and creates the flows QUEUED (Luiz 15/11; FIFO after the IDS58 extension 7570-7573). Flow ids are recorded on
the Plane story.

**Created 15/11/2026 (after the freeze commit e0f012cb):** 7574 (r20419, control 7531), 7575 (r20420, control 7533),
7576 (r20421, control 7535); QUEUED, 2 experiments each verified, `max_bits` 64 the only diff. Plane story IDS-59.
