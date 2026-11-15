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

## Amendment 1 (Luiz 15/11, before any 2×2 run): interim looks at n = 3, 5, 7, then 10
The driver is already seed-major (all three arms of seed k finish before seed k+1 starts), so after every 3rd, 5th and
7th seed all arms have the same n. At each interim look (n = 3, 5, 7), per arm, using the paired per-seed differences
vs L1 (B − A) and a **99 % two-sided t interval** (stricter than 95 % because we look repeatedly; t(2) = 9.92,
t(4) = 4.60, t(6) = 3.71):
- **Early ADOPT:** the target's interval lies entirely on the good side of its threshold (L2y: steady upper bound
  < −0.03°; L3: pos upper bound < −0.010 m) AND each guard's interval upper bound is < +0.03° (steady and err for L3,
  err for L2y) AND the arm's pooled fails are ≤ 0.05 % of its episodes so far AND not significantly worse than L1 on
  the same seeds (exact one-sided conditional binomial, p < 0.05).
- **Early NOT-ADOPT (futility):** the target's interval lies entirely on the bad side (L2y: steady lower bound
  > −0.03°; L3: pos lower bound > −0.010 m), OR a gate failure is already certain (pooled fails > 17 out of the full 35,000).
- Otherwise the arm is **undecided** and the block continues.
- When EVERY arm has an early verdict, the main session recommends stopping; **Luiz decides** whether to stop the driver.
  The base then follows step 4 of the rule using the means at that look. If any arm is undecided at n = 7, run to n = 10
  and apply the original rule (point estimates) to all arms.

## L3 reference rows (15/11/2026; report-only context, does not change the rule)

The three MPCOF reference rows re-scored ONCE on the 35 standard TEST units (99990101-135 × 100 episodes, the same episodes as the published rows). The rows were scored with `--alt-integral` off and then with `pole3`, with nothing tuned.
- Scorer: `rival_probe_tests::test_units_reference_rows_alt_integral_pole3` (cargo `#[ignore]`, test-only). The fixture is `scripts/ctrl_indi_val_fixture.py --units test` on the block-9 s1 run.
- **Gate passed:** with the integrator off, all three rows reproduce `docs/ctrl_rival_row_ruling.md` rows C / B / A at their printed 0.001 precision.

```
row (35 TEST units x 100 eps)                 | alt     | stable % | err °  | steady ° | pos m | pos_hold m | dz_hold m | fails
----------------------------------------------+---------+----------+--------+----------+-------+------------+-----------+------
oracle-fed (label law; a bound)               | off     | 100.00   | 0.690  | 0.163    | 0.357 | 0.309      | 0.195     | 0/3500
                                              | pole3   | 100.00   | 0.699  | 0.148    | 0.313 | 0.222      | 0.044     | 0/3500
est + bias comp (equal-info COMPARATOR)       | off     | 100.00   | 0.691  | 0.165    | 0.357 | 0.309      | 0.195     | 0/3500
                                              | pole3   | 100.00   | 0.700  | 0.150    | 0.313 | 0.222      | 0.044     | 0/3500
est-fed, no bias comp                         | off     | 100.00   | 1.352  | 0.684    | 0.341 | 0.280      | 0.195     | 0/3500
                                              | pole3   |  99.97   | 1.360  | 0.669    | 0.293 | 0.184      | 0.044     | 1/3500
```
- For the L3 arms, gaps are quoted against the **pole3 COMPARATOR row: 100.00 / 0.700 / 0.150 / 0.313 m, pos_hold 0.222 m, dz_hold 0.044 m, 0/3500.**
- pole3 changes only the vertical channel: hold |dz| 0.195 → 0.044 m, and the horizontal hold |dx| / |dy| are identical. pos falls by 0.044-0.048 m and pos_hold by 0.087-0.096 m. Attitude moves by at most +0.009° err and −0.015° steady.
- The no-bias-comp row picks up 1 fail in 3,500 under pole3, against 0 with it off. That is one episode, not a rate (the 300-unit stress test put this row at 5/30,000 with it off).
