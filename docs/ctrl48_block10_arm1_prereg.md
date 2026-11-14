# CTRL-48 block 10, arm 1 — pre-registration: `both` + L1 vs `both` (DRAFT for Luiz, 14/11/2026)

Written BEFORE any block-10 run exists. Spec: docs/ctrl48_block10_levers_spec.md (merge 28288c7b, controller ABI 51).
Driver: logs/controller/ctrl48/ab_block10_arm1.sh (NOT launched until Luiz approves this document).

## Why
Block-9 gap analysis (docs/ctrl_gap_after_block9.md) + the L1 VAL pre-check: one tilt-reference error input carries
78-83% of the hold roll/pitch label (today's roll_err/pitch_err 8-29%); the student answers that offset with 5-10% of
its teacher's stiffness. Hypothesis: giving the student that input directly lowers steady (roll/pitch hold error).

## Design
- ONE arm, `both_L1` = the block-9 `both` recipe (each seed's own flown [argv]) + `--obs-tilt-ref-err`
  (ladder `dense`, the default). Nothing else changes; rule P kept.
- Seeds s1-s10 (base seeds 31337001-10), seed order s1..s10. Test = 35 CF2.1 units x 100 episodes (99990101-35).
- **Control = block-9 `both` s1-s10 as flown (ABI 50), NOT re-flown.** Admissible: ABI 51 with all block-10 flags
  off reproduced block-9 `both` s3 exactly (same pick NEURONS#0; 100.0 / 1.26±0.23 / 0.68±0.14 / 0.358±0.017 /
  0/3500; logs/controller/ctrl48/smoke_abi51_s3.out, 14/11 ~23:55 UTC).
- Cost ≈ 10 × ~80 min ≈ 13-14 h.

## Decision rule (s1-s10, paired by seed), applied in order
1. Safety gate: pooled fails of `both_L1` <= 0.05% (<= 17/35000) AND not significantly worse than `both`
   (6/35000 on the same units; exact one-sided conditional binomial, p < 0.05).
2. If it passes: ADOPT L1 only if mean steady is better than `both` by MORE than 0.03° (tie band) AND mean err is not
   worse by more than 0.03°. Otherwise `both` stays the base (ties go to the simpler recipe).
3. Gate fails: report to Luiz, no adoption.
If adopted, `both_L1` is the base of the block-10 2×2 (L1 / +L2y / +L3 / +L2y+L3, L3 with re-flown baselines).

## Expected (stated in advance; not part of the rule)
steady 0.79° → 0.5-0.65° (block-9 bound 0.49°); err roughly unchanged; pos unchanged (L1 is attitude-only).

## Reported, descriptive only
Per seed and mean: stable / err / steady / pos / fails, paired deltas (Δ = L1 − both), seeds won on steady out of 10;
pos_hold / dz_hold (new columns; the control has no flown value for them — re-scored from its checkpoint only if Luiz
asks). Gaps quoted vs the equal-information MPCOF COMPARATOR row (100.0 / 0.69 / 0.17 / 0.357 / 0).
Nothing above may change after the first block-10 END line.
