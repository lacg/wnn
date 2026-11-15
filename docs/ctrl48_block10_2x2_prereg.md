# CTRL-48 block 10, 2×2 on top of L1 — pre-registration (DRAFT for Luiz, 15/11/2026)

Written BEFORE any 2×2 run exists. Spec docs/ctrl48_block10_levers_spec.md (ABI 51). Base = `both_L1` (arm 1 ADOPTED,
docs/ctrl48_block10_arm1_result.md). Driver logs/controller/ctrl48/ab_block10_2x2.sh (NOT launched until Luiz approves).

## Arms (each = that seed's flown `both_L1` [argv] + the flags below; nothing else changes; rule P kept)
| arm | extra flags | hypothesis |
|---|---|---|
| L1 (control) | — | arm-1 runs AS FLOWN (same ABI 51, same code), not re-flown |
| L1+L2y | `--obs-yaw-trim-err` | trim-referenced yaw error cuts the yaw part of err/steady (err and steady are the full SO(3) angle to a yaw-0 tilt reference, so yaw shows in both) |
| L1+L3 | `--alt-integral pole3` | altitude integrator removes the hold droop → pos / pos_hold / dz_hold down |
| L1+L2y+L3 | both | the two act on different channels → effects add |

- Seeds s1-s10 (31337001-10), **seed-major** (s1: L2y, L3, L2y+L3; then s2; …) so a partial block stays balanced.
  30 runs × ~65 min ≈ **33 h**. Test = 35 CF2.1 units × 100 episodes (99990101-35).
- **L3 reference rows:** L3 changes the teacher's labels, so the three MPCOF reference rows are re-scored ONCE with
  `--alt-integral pole3` on the same 35 test units (rival scorer, report-only) and shown above the L3 arms. The WNN
  comparison is always arm vs the L1 control on the same seed.

## Decision rule (s1-s10, paired by seed, each arm vs L1), applied in order
1. **Safety gate (every arm):** pooled fails ≤ 17/35000 AND not significantly worse than L1 (5/35000; exact one-sided
   conditional binomial, p < 0.05). A gate failure = not adopted.
2. **L2y (attitude lever):** adopt if mean steady better by > 0.03° AND mean err not worse by > 0.03°.
3. **L3 (position lever):** adopt if mean pos better by > 0.010 m AND mean steady and err each not worse by > 0.03°.
4. **New base:**
   - neither adopted → L1 stays;
   - exactly one adopted → that arm;
   - both adopted → L1+L2y+L3 IF it passes the gate and is not worse than L1+L2y by > 0.03° on steady and not worse than
     L1+L3 by > 0.010 m on pos; otherwise the better single arm (by its own target's margin over L1).

## Expected (stated in advance; not part of the rule)
- L2y: a small steady/err gain, smaller than L1's (pre-check: it carries 70-85% of the missing proportional yaw term,
  but the yaw label is trim-dominated); pos unchanged.
- L3: pos 0.338 → ~0.29-0.30 m, pos_hold 0.27 → ~0.20 m, dz_hold 0.13 → ~0.05 m IF the student learns the new law;
  attitude roughly unchanged. Risk: the student's `alt_err` input changes meaning, so it may not learn it.

## Reported, descriptive only
Per seed and mean, compact pair tables per arm (A = L1 · B = arm); pos_hold / dz_hold; the 2×2 interaction
(L2y+L3 − L2y − L3 + L1) on steady and pos; gaps vs the equal-info COMPARATOR row (pole3 version for L3 arms).
Nothing above may change after the first 2×2 END line.
