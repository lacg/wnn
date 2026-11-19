# IDS63: fixed 96-bit width on unswr configuration C (does widening past 80 keep paying?)

Written 18/11/2026 ~20:45 EST on Luiz's instruction of 18/11 ~20:30 EST, before any IDS63 flow exists and before
IDS62's third seed (7587) has finished. Queue script: `scripts/queue_ids_unswr_fixed_width.py --ids 63 --width 96`.
Follows `docs/ids62_unswr_fixed80_prereg.md` (same configuration, forcing and validity check).

## 1. Why

IDS62 (fixed 80 bits) is on track to ADOPT at 2/3: dF1 +0.164 / +0.178 over C, FPR 0.296 % / 0.244 %. Luiz's target
is the classical rows on unswr (`random_3way`, same TEST):

```
XGBoost, same 64b bits   95.53          0.417          6K nodes
RF, same 64b bits        95.32±0.07     0.380±0.064    1.78M nodes
XGBoost, raw top-20      95.60          0.427          6K nodes
RF, raw top-20           95.81±0.07     0.288±0.027    1.16M nodes
```

80b sits at macro-F1 ~95.13 with FPR at or below the lowest classical FPR. The question: does 96 bits move F1 toward
95.81 while keeping FPR there?

## 2. Forcing and arms

- **Override.** `max_bits` 64→96, `min_bits` 4→96, `bits_grid` [96]. Nothing else; the queue script asserts the
  diff is exactly these three keys. Same reason as IDS62 for fixing the width rather than a band: the CV fitness
  pulls width down to the band floor.
- **Arm.** `IDS63-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb96-encoff-r<seed>`,
  seeds r20421 / 20422 / 20423 (s1-s3, the IDS62 seeds).
- **Primary pair: IDS62 80b** (7585 / 7586 / 7587), same seed. 80b is the incumbent; this asks whether 96 beats it.
- **Secondary (descriptive): C** (7567 / 7568 / 7569).
- Addresses above 64 bits use `compute_address_wide` (splitmix64 chain), as at 80. No wheel or ABI change.

## 3. Validity check (HARD; applied before any F1 is read)

1. Every neuron of each arm's GA `best_f1` final genome is exactly 96 bits. A seed that fails is void.
2. Fewer than 3 valid seeds: no verdict, report and stop.
3. Also reported: GA `best_fitness` final width, grid `best_f1` width.

## 4. Metric and decision rule (96 vs 80, paired by seed)

Cell: GA `ga_neurons`, `best_f1`, `test_point='final'`, `margin_val_cal`, TEST, from
`test_summaries.threshold_metadata` (`macro_f1`, `fpr`). A win is dF1(96−80) ≥ +0.05. Cap: arm FPR above 0.42 % on
any seed.

1. **REJECT** if wins ≤ 1/3, OR mean dF1 < 0, OR any seed is over the cap. 80 stays.
2. **ADOPT 96** if wins 3/3, AND mean dF1 ≥ +0.05, AND mean dFPR ≤ +0.05 with every seed ≤ +0.10. 96 becomes the unswr
   max-accuracy row, with its own cell count.
3. Otherwise **AMBIGUOUS** → Luiz decides. No automatic extension.

Report beside the rule: the C pair, attack F1, TPR, cells, GA generations, hours, and the gap to each classical row
(labelled TEST, both sides).

**Expectation (not a result).** Weak and possibly flat: there is **no VAL gate at 96** (the IDS-57 / 16-17/11 VAL
ladder stopped at 80: 72 +0.12, 76 +0.11, 80 +0.15 over 64, i.e. flattening). 96 was chosen by Luiz to probe the
trend, not from a VAL read. Closing the full ~0.68 pp gap from 80b (~95.13) to RF top-20 (95.81) with width alone is
not expected; a +0.05..+0.15 step is the plausible range. CE rises with width (more unfilled addresses), so the
grid/GA may pick more neurons.

## 5. Guards and cost

- `_3way`, 5-fold CV on train, VAL calibration, all inherited and asserted.
- Queued only after the ram_core worker swap (waiter `logs/ids/queue_ids63_after_swap.sh`), so the swap is not
  blocked; FIFO puts IDS63 ahead of every held (paused) flow, which stay paused.
- **Cost.** IDS62 ran 12.6 h per flow at 80 bits; expect ~13-16 h per flow at 96, ~40-48 h for three.
