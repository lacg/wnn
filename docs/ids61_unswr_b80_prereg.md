# IDS61: pre-registration of max_bits 80 on the unswr configuration C

Written 16/11/2026 ~21:20 EST (ids-security) on Luiz's approval of 16/11 ~21:10 EST ("afterwards", gate first). Written
before any IDS61 flow exists. Plane story IDS-61. Queue script: `scripts/queue_ids61_unswr_b80.py`.

## 1. Question and hypothesis

- **Question.** C (IDS58: vote θ=0.02 unweighted + `max_bits` 64, adopted 16/11) is −0.52 pp macro-F1 against XGBoost
  on the same 64-bit thermometer bits (TEST, 95.01±0.08 vs 95.53; seeded baselines measured 16/11). Does raising the
  cap to 80 bits close part of that?
- **H1.** The 64-bit cap still binds. Wider addresses split more of the mixed benign/Fuzzers cells.
- **Expectation (not a result).** +0.07 to +0.15 pp TEST F1, FPR flat, about 1.3-1.5× cells.
- **Why the size.** The VAL gate below gives +0.147. On A1 the VAL gate overstated the TEST gain by about 1.6×. On
  random genomes, IDS-57's VAL ladder gave +0.24 for 80 over 64. Addresses above 64 bits are hashed, not identity
  (FPGA story, `project_bits_above_64_or_fold`).

## 2. VAL gate (run 16/11 ~21:18-21:20 EST, rule fixed at 21:18 before results)

C's GA `best_f1` final genomes (7572, 7573, 7567, 7568, 7569) were retrained through the worker's construction path
(Rust; report slot = VAL, TEST never scored). Each was scored as flown, then widened to 80 bits by appending uniform
random input connections, with 2 draws per genome. Metric: `margin_val_cal` on VAL (τ fitted on VAL, scored on VAL).
It is a tuning read, so only differences are shown.

Rule: PASS iff mean dF1 ≥ +0.10 AND ≥ 4/5 genomes dF1 > 0 AND mean dFPR ≤ +0.05.

```
genome (seed)   as-flown bits | dF1 @64 (widen to cap) | dF1 @80 | dFPR @80
7572 (r20419)   60-61         | −0.034                 | +0.157  | +0.026
7573 (r20420)   61-63         | +0.009                 | +0.135  | −0.016
7567 (r20421)   64-64         |  0.000                 | +0.201  | −0.004
7568 (r20422)   53-54         | +0.115                 | +0.147  | −0.024
7569 (r20423)   62-63         | +0.013                 | +0.093  | +0.002
mean                          | +0.020                 | +0.147  | −0.003   → PASS (5/5 positive)
```
The @64 column is the control for "any widening helps": filling up to the existing cap adds about 0.

## 3. Arms

- **Arm** (3 flows, r20421 / 20422 / 20423):
  `IDS61-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-b80-encoff-r<seed>`. Each is a
  verbatim clone of the C flow with exactly one key changed: `max_bits` 64 → 80 (asserted). Worker ABI 29, same build
  as C.
- **Controls** (existing): C 7567 / 7568 / 7569. The pairing is the same as IDS58's primary.

## 4. Metric

GA `ga_neurons`, `best_f1`, `test_point='final'`, `margin_val_cal`, TEST, `test_summaries.threshold_metadata` keys
`macro_f1` / `fpr`. Paired by seed: dF1, dFPR in pp. Report attack F1, TPR, cells, neuron widths (the arm must have
neurons above 64 bits, else it did not test H1), GA generations and hours.

## 5. Decision rule (n=3, read once when all 3 are terminal)

Win: dF1 ≥ +0.05. Cap: arm FPR > 0.42 % (XGBoost's) on any seed.
1. **REJECT** if wins ≤ 1/3, OR mean dF1 < +0.05, OR any seed over the cap.
2. **ADOPT** if all of:
   - wins 3/3;
   - mean dF1 ≥ +0.10;
   - mean dFPR ≤ +0.05, and every seed ≤ +0.10.
   Then b80 becomes the unswr max-accuracy row, with its own (larger) cell count. C keeps its row, and "80 beats 64"
   is not claimed at equal cells.
3. Otherwise **AMBIGUOUS** → Luiz decides. There is no automatic extension, and C stays.

Width sanity: if no arm neuron exceeds 64 bits, the run is void for H1 (report it, no verdict).

## 6. Guards and cost

- `_3way`, 5-fold CV on train, VAL calibration, all inherited and asserted. The value 80 came from the IDS-57 VAL ladder
  and this VAL gate, not from TEST. FIFO places the arm behind IDS60 (7577-7581). Paused flows stay paused.
- **Cost.** C flows ran 6.7-10.2 h (mean 8.3 h). At 80 bits, expect about 1.3× per generation, so about 10-13 h per
  flow and about 30-40 h for the three.
