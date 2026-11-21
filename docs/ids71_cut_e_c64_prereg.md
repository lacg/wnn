# IDS-71: supervised cut points (arm E) on unswr config C (≤64b) — prereg (Luiz 21/11/2026 ~07:40 EST)

## Why
IDS-70 arm E (`ids_cut_method=info_gain_mc`) s1 on the fixed-80b recipe (flow 7591 vs IDS-68 7585, same seed):
F1 95.37 vs 95.15 (+0.21 pp), FPR 0.278 vs 0.296, acc 99.33 vs 99.30, occupied cells 1.31M vs 2.89M (−55%) at 493
neurons. Luiz: "can we test with the 64b as well?" At ≤64-bit neurons the cell budget is what decides FPGA fit
(Plane IDS-66), so a cut method that keeps F1 with fewer cells matters most there.

## What E changes
Only the input encoder. The 20 top features stay thermometer-coded with the same 1104-bit global budget, but the
cut points are chosen best-first by information gain against the 10 UNSW attack classes, fitted on TRAIN only.
proto / service / state become thermometers over their TRAIN attack-rate rank. The baseline is unsupervised quantile
cuts. The WNN, GA, fitness, folds and threshold calibration are unchanged.

## Arm and control
- **Arm** = config C flows 7567 / 7568 / 7569 (r20421-23; `max_bits` 64, the GA picks widths, 53–64 observed)
  cloned VERBATIM plus `ids_cut_method=info_gain_mc`. Bodies in `docs/ids71_flows.json`, queued by
  `scripts/queue_ids71_cut_e_c64.py` at the END of the IDS queue (after IDS-70 arm B).
- **Control** = the same-seed C flow, not re-flown.

## Read
GA `best_f1` · `margin_val_cal` · TEST (`test_summaries`), per seed, paired vs C: F1, FPR, acc, logical cells /
neurons, bits, hours. Adopt reading (Luiz): F1 up consistently, not worse elsewhere. Report the cell change next to it.
