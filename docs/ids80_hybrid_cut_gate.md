# IDS-80: hybrid cut set (E + B timing cuts), offline VAL gate

Written 24/11/2026 (EST) by ids-security, on Luiz's request of 24/11 (Plane IDS-80). This file fixes the
gate BEFORE any number is computed. No flow is queued, the worker and the DB are not written, no wheel
changes. Script: `logs/ids/ids80/gate_hybrid.py`; raw records: `logs/ids/ids80/gate_hybrid.jsonl`.

## Why (from the IDS-70 E vs B readout, 24/11)

- B (`xgb_splits`) spends ~39 % of a neuron's connections on the four timing features (dur, Sintpkt, Sjit,
  Dintpkt); E (`info_gain_mc`) spends ~18 %. B's neurons see 2x the distinct TRAIN addresses, and fewer attack
  / Fuzzers rows share an address with benign traffic (Fuzzers in mixed addresses 78-80 % vs 87-88 %).
- At matched VAL FPR, B alone was NOT better than E (recall -0.27 pp mean at FPR <= 0.25 %), but the E and B
  errors overlap only ~61 % (FN). Question: does adding B's timing cuts to E's encoding help one model?

## Hybrid definition (arm H)

- Every feature keeps E's `info_gain_mc` cuts, fitted on TRAIN exactly as in IDS-70 (budget 1104 bits,
  multiclass label; proto/service/state as rank thermometers).
- For dur, Sintpkt, Sjit, Dintpkt only: add B's `xgb_splits` cuts (TRAIN-fitted XGB, 100 trees, depth 6,
  lr 0.1, binary label, the IDS-70 code path, seed 20421).
- Merge per timing feature: the sorted union of E's and B's thresholds. A B cut is dropped if it induces the
  same TRAIN partition as a cut already in the set (no finite TRAIN value v with min(a,b) <= v < max(a,b)),
  which includes exact duplicates. Thermometer bit = (x >= cut), as for every supervised cut.
- Bits: 1104 (E) + the B timing cuts that survive the merge. B has 58 + 44 + 40 + 37 = 179 timing cuts; 3 dur
  cuts are exact duplicates of E's, so the upper bound is 1280 bits. The exact count is printed by the script
  (`[CUTS]` line) before any genome is scored, and reported with the verdict.
- Implementation: offline only. The script wraps `SupervisedCutFitter.fit` in-process so that the normal
  `IdsEvaluatorFactory.load` path (flow 7591 params, `info_gain_mc`) builds the hybrid encoder. No repo code
  changes.

## Arms

- **E**: cut E encoding (1104 bits), flow 7591 params. Control.
- **H**: hybrid encoding (above), same params.
- **B** (reference only, not part of the rule): cut B encoding (457 bits), flow 7594 params.

## Genome protocol (as IDS-77 G1-G4)

- Random unsearched genomes: 2 clusters x 250 neurons x 80 bits, BINARY, vote theta 0.02, connections drawn
  uniformly per bit with replacement over the arm's whole bit pool.
- 10 draws, RNG seeds 1000-1009. Pairing is by draw seed: each draw uses the same uniform stream
  `u ~ U[0,1)` per connection, mapped to bit `floor(u * total_bits)` of the arm. The bit pools differ in size,
  so the draws are paired, not identical. The per-bit draw moves connection share toward the timing features
  in H (that is part of the treatment).
- Rust trains each genome on the full TRAIN partition; scoring on VAL only.

## Partitions

- Cuts and genome training: TRAIN rows and TRAIN labels only.
- Scoring: VAL only. The shared loader reads the `random_3way` split and encodes all three partitions; the
  script replaces the report slot with VAL (`val_as_report`) before any evaluator is built and never scores
  TEST rows. No TEST number appears in the result.

## Readout per genome (VAL)

- `F1`, `FPR`, `recall` at the F1-optimal VAL threshold (the `margin_val_cal` rule: sweep every distinct
  margin).
- `recall@0.25`: recall at the best threshold with VAL FPR <= 0.25 %; `fuzz@0.25`: Fuzzers recall at that
  same threshold.
- Cells: logical cells per neuron of the draw-1000 genome per arm.

## PASS rule (fixed now)

Paired deltas d = H - E over the 10 draws. **PASS** only if all three hold:

1. dF1 (F1-optimal point) > 0 on >= 8/10 draws, and mean dF1 > 0;
2. d recall@0.25 > 0 on >= 8/10 draws, and mean > 0;
3. mean dFPR (F1-optimal point) <= 0.000 pp (FPR not worse).

Otherwise **FAIL**. 8/10 under a fair coin is p = 0.055 one-sided. Fuzzers recall at matched FPR and cells
are reported beside the rule, not part of it.

If PASS: a 3-flow cohort vs IDS-70 E (7591-7593) is outlined, not queued (it needs a `CutMethod` in code).
If FAIL: the hybrid is closed.
