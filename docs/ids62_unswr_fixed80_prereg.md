# IDS62: fixed 80-bit width on unswr configuration C (redo of the VOID IDS61)

Written 17/11/2026 ~15:30 EST (ids-security) on Luiz's instruction of 17/11 ~14:40 EST, before any IDS62 flow exists.
Plane: IDS-61 (updated with this redo). Queue script: `scripts/queue_ids_unswr_fixed_width.py --ids 62 --width 80` (was `queue_ids62_unswr_fixed80.py`). Supersedes
`docs/ids61_unswr_b80_prereg.md`; its decision rule is kept, and a hard validity check is added.

## 1. Why IDS61 is VOID: a search property, not a clamp

IDS61 cloned C with only `max_bits` 64→80. Its GA `best_f1` finals came back at 60-63 bits (7582) and 60 bits (7583),
so the prereg's void clause fired. 7584 was paused before its GA finished.

Evidence (DB `genome_evaluations` by generation, and the flow logs):

1. **The search did propose more than 64 bits.**
   - The grid (bits grid `[4, 12, 24, 36, 48, 60, 72, 80]`) trained and scored 72- and 80-bit configs.
   - The grid's `best_f1` genome was 80 bits on all three flows.
   - The GA was seeded from 50 grid genomes. At generation 1, 38/40 offspring (7582) and 27/40 (7583) were wider than
     64. 7582's rank-1 elite at generation 1 was 73-80 bits.
2. **Selection removed them.** Population width (max per genome, mean / count above 64):
   ```
   7582 gen 1  elite 70.6 (8/10 >64)  offspring 75.0 (38/40)    7583 gen 1  elite 70.3 (8/10)  offspring 70.7 (27/40)
        gen 6  elite 63.0 (1/10)      offspring 62.1 (3/40)          gen 6  elite 60.0 (0/10)  offspring 62.0 (7/40)
        gen 16+ every genome 62-64 → 63, 0 above 64                  gen 16+ every genome 60, 0 above 64
   ```
3. **The fitness, not a clamp.**
   - The GA Neurons phase never mutates widths (`optimize_bits=False`). Widths come only from the seeds and from
     `blend_cluster` (BLX-0: a child width drawn from U[min, max] of its parents), so a width can only survive by
     selection.
   - The 5-fold-CV desirability fitness ranks narrower genomes first. Grid ranking on 7582 at n=250: b=60 fitness
     1.0645 (CE 0.3272) ahead of b=80 1.0666 (CE 0.3288). CE rises with width (more unfilled addresses) while accuracy
     moves only 99.18→99.20.
   - On a full-train VAL read the same widening is positive. C's genomes widened on VAL give: 64 +0.02, 68 +0.07,
     72 +0.12, 76 +0.11, 80 +0.15. So the search objective and VAL disagree on width.
4. **No 64-bit hard limit remains.**
   - Addresses above 64 bits use `compute_address_wide` (a splitmix64 chain; identity at ≤64) in
     `core/neuron_memory.rs`, with a Metal twin. The OR-fold bug was fixed on 29/08 (worker ABI 12).
   - No clamp to 64 exists in `architecture_ga.py` or `crossover_bits.rs`.
   - Footprints at 80 bits are real: the grid logged 500n × 80b at 3.17 Mcells.

Code fix: **none needed, no wheel, no ABI change.** The remedy is the flow configuration.

## 2. Forcing chosen: width fixed at 80

- **Override.** `max_bits` 64→80, `min_bits` 4→80, `bits_grid` [80]. All three are registered params, and the queue
  script asserts that the diff is exactly these three keys. The grid then picks neuron counts at 80 bits, and the GA
  optimizes neurons and connections at 80 bits.
- **Why not a band (e.g. 65-80).** The fitness pushes width down, so any band's floor becomes the attractor. A 65-80
  band would end near 65-68, where the VAL ladder gives only about +0.07. The prereg hypothesis and the gate are
  about 80.
- **Honest scope.** This tests "an 80-bit architecture, with neurons and connections chosen by the GA". It does
  **not** test "the GA would choose 80 bits": IDS61 showed that, under this fitness, it does not.
- **Smoke (17/11 ~15:20 EST).** One real `ArchitectureGAStrategy` generation on the K-fold optimizer, using a
  VAL-as-report dataset (TEST never scored), with C's params and min=max=80. Every parent and all 9 offspring had
  every neuron at exactly 80 bits (`neurons_le_64=0`, `neurons_gt_80=0`).
- **Smoke with a 65-80 band, for reference.** The grid rule gives `[65, 66, 68, ..., 78, 80]`. Offspring widths were
  66-80, with none at or below 64.

## 3. Arms

- **Arm.** `IDS62-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff-r<seed>`,
  seeds r20421 / 20422 / 20423.
- **Controls.** C 7567 / 7568 / 7569 (same build, ABI 29).

## 4. Validity check (HARD; applied before any F1 is read)

1. Every neuron of each arm's GA `best_f1` final genome is exactly 80 bits. If any seed fails, that seed is void.
2. If fewer than 3 seeds are valid, there is no verdict: report and stop.
3. Also reported: the GA `best_fitness` final width and the grid `best_f1` width.

## 5. Metric and decision rule (IDS61's, unchanged)

The cell is GA `ga_neurons`, `best_f1`, `test_point='final'`, `margin_val_cal`, TEST, from
`test_summaries.threshold_metadata` (`macro_f1`, `fpr`), paired by seed. A win is dF1 ≥ +0.05. The cap is an arm FPR
above 0.42 % on any seed.

1. **REJECT** if wins ≤ 1/3, OR mean dF1 < +0.05, OR any seed is over the cap.
2. **ADOPT** if wins 3/3, AND mean dF1 ≥ +0.10, AND mean dFPR ≤ +0.05 with every seed ≤ +0.10. Then fixed-80 becomes
   the unswr max-accuracy row, with its own cell count (about 1.5× C). C keeps its row.
3. Otherwise **AMBIGUOUS** → Luiz decides. There is no automatic extension, and C stays.

Report beside the rule: attack F1, TPR, cells, GA generations, hours.

**Expectation (not a result).** +0.07 to +0.15 TEST, FPR flat. Basis: the VAL widening gate gives +0.15 at 80, and the
GA re-optimizes at that width. On A1, VAL overstated TEST by about 1.6×.

## 6. Guards and cost

- `_3way`, 5-fold CV on train, VAL calibration, all inherited and asserted.
- 80 was chosen from VAL gates (IDS-57 ladder, and the C widening gates of 16/11 and 17/11), never from TEST. IDS61's
  TEST rows (7582/7583) were not used to choose anything. They are void and not reported.
- Every other flow stays paused (7535, IDS59 7574-7576, IDS56C/56UC). 7584 is cancelled once these flows are
  verified.
- **Cost.** IDS61 generations with an 80-bit-heavy population ran about 400 s; GA phases ran 60-70 generations.
  Expect about 7-9 h per flow, about 22-27 h for three.
