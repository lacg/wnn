# IDS60: pre-registration of CE20 fitness weights on unswt

Written 16/11/2026 ~21:20 EST (ids-security) on Luiz's approval of 16/11 ~21:10 EST. Written before any IDS60 flow
exists. Plane story IDS-60. Queue script: `scripts/queue_ids60_unswt_ce20.py`.

Vocabulary: **train** fits and drives the search (5-fold CV). **VAL** tunes (threshold calibration). **TEST** is the
report.

## 1. Question and hypothesis

- **Question.** The adopted unswt recipe (IDS-56 ms01-encon: BINARY 2-cluster, weighted vote θ=0.1, dedupe on,
  16-bit thermometer, `temporal_3way`) still searches with the old Wb weights (ce/acc/f1/fpr 0.10/0.20/0.35/0.35,
  `fitness_calculator=harmonic_rank`, default aggregation). Does the IDSZ CE20 vector do better?
- **H1.** Leaning on CE (0.20) and FPR (0.40) raises unswt macro-F1 without an FPR cost.
- **Basis.** IDSZ (21/08, QUAD era, unswt, n=5): CE20 beat Wb-CTRL by +0.95 pp F1 (5/5). IDS-23: unswt is where the
  GA adds (+0.41 over grid) and where k-fold fitness and held-out disagree (temporal shift), so the weights have
  something to steer. Current gap to XGBoost on the same 16-bit bits (TEST, measured 16/11): −0.26 F1 / +0.22 FPR.
- **Expectation (not a result).** +0.2 to +0.4 pp F1, FPR flat. Smaller than IDSZ, because:
  1. IDSZ combined the weights by z-score. This control uses the default aggregation, and the arm keeps it, so only
     the weights change.
  2. IDSZ ran on QUAD, not on the BINARY vote.

## 2. Arms and seeds

- **Arm** (5 flows): `IDS60-unswt-BINARY-2m-max250-xo_blend_neuron-16bCE20-rate0.25-ms01-encon-r<seed>`. Each is a
  verbatim clone of its control's `params`, with exactly four keys changed: `fitness_weight_ce` 0.10→0.20,
  `fitness_weight_acc` 0.20→0.10, `fitness_weight_f1` 0.35→0.30, `fitness_weight_fpr` 0.35→0.40. The script asserts
  that the diff is exactly these four keys, that every key is in `KNOWN_PARAMS`, and that the worker ABI is 29.
- **Controls** (existing, not re-flown): 7484 / 7488 / 7492 / 7496 / 7500 = r20419 / 20420 / 20421 / 20422 / 20423.
- **Cross-ABI.** The controls ran on ABI 28 and the arm runs on ABI 29. Certified 16/11 21:17 EST:
  `scripts/ids57_abi_equivalence.py --retrain-finals 7484 7488 7492 7496 7500` on ABI 29 reproduces every stored
  TEST confusion matrix exactly (`all_exact=True`).

## 3. Metric (the reporting cell, fixed)

GA phase (`ga_neurons`), `genome_type=best_f1`, `test_point='final'`, mode `margin_val_cal` (τ fitted on VAL, scored
on TEST). Source: `test_summaries.threshold_metadata`, keys `macro_f1` and `fpr`. Never `iterations.best_f1`, and
never the best of the 50 TEST cells. Per seed: dF1 = arm − control, dFPR = arm − control (pp).

## 4. Decision rule (n=5, read once when all 5 arm flows are terminal)

Applied in order:
1. **REJECT** if any of: wins (dF1 ≥ +0.05) ≤ 2/5; mean dF1 ≤ 0; mean dFPR > +0.25 pp.
2. **ADOPT** if all of:
   - mean dF1 ≥ +0.15;
   - wins ≥ 4/5;
   - no seed loses (dF1 ≤ −0.05);
   - mean dFPR ≤ +0.05.
3. **NON-INFERIOR** (Luiz's style) if not ADOPT but all of:
   - no seed loses;
   - mean dF1 > 0;
   - mean dFPR ≤ +0.05.
   Luiz rules on this case; it is reported, not auto-adopted.
4. Otherwise **AMBIGUOUS** → REJECT. There is no n extension.

The FPR tolerance is +0.05 pp, which is 9 benign rows of 18,500. unswt FPR has a seed SD of 0.63 pp, so the FPR
clauses mainly guard against a systematic shift. Report attack F1, TPR, cells, GA generations and flow hours beside
the rule.

## 5. Guards

- `_3way` split, 5-fold CV on train (`ids_k_folds=5`, `ids_kfold_per_gen=5`) inherited and asserted. VAL calibrates.
  TEST is touched only at each flow's own final report.
- The weights come from IDSZ (an earlier cohort), not from any TEST read of these seeds.
- Flows via POST /api/flows with the 2-experiment array, count verified, then PATCH queued. Paused flows
  (7535, 7574-7576, the IDS56C/56UC sets) are not touched.
- Budget: patience-driven generations are a mediator (IDSZ). Report them; do not adjust for them.

## 6. Cost

About 0.72 h per flow (controls 0.69-0.73 h), so about 4 h for 5 flows.
