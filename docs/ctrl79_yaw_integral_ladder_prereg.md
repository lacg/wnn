# CTRL-79 — yaw-integral ladder on the G+T base — pre-registration (25/11/2026)

Luiz 25/11 evening: "prepare option 4 for flight". Written before any flight. Nothing flies until Luiz says so.

## Why (VAL diagnosis only; G+T winners, never TEST)

The WNN observes a leaky integral of its yaw error (`yaw_err_i`, `--obs-yaw-err-i`, leak 0.9995 per ms) through 8
thresholds = a 9-level code. Today's fitted cuts are ±3.49 / ±8.73 / ±20.57 / ±40.46.

- **The integral carries the trim.** On 3 VAL units × 100 episodes (G+T s1), the per-episode hold-mean integral
  correlates **+0.97** with the yaw-rate trim the WNN ends up holding (units: trim +0.180 / −0.140 / +0.347 rad/s ↔
  integral +4.35 / −3.57 / +7.08).
- **The ladder does not resolve it.** Over 20 VAL units × 100 episodes, G+T s1 / s2 / s3, hold window:
  - code occupancy (9 bins, pooled): 0 / 0 / 1 / 30 / 42 / 23 / 4 / 0 / 0 % — entropy 1.79 of 3.17 bits;
  - the two outermost cuts (±40.46) are never reached in an episode (whole-episode |I| max 22–32); beyond ±20.57 is
    0.2–0.5 % of ticks;
  - pooled hold |I| quantiles 20 / 40 / 60 / 80 %: 2.73 / 3.42 / 4.12 / 6.33 — the integral sits AT the ±3.49 cut, and
    its code changes during the hold in 58 % of episodes (3-unit trace). The trim is being duty-cycled across one cut.
- This is the same trim L1 (CTRL-74) tried to hold in the output accumulator, where it cost a ~195 ms lag. Reading it
  from an input adds no output lag.

## Arm (model-free: an input threshold ladder on the WNN's own yaw error; no observer, no MPCOF part)

`yil` = G+T + `--yaw-integral-ladder dense`: fixed symmetric cuts on `yaw_err_i`

    −12.0  −6.5  −3.5  −1.5  +1.5  +3.5  +6.5  +12.0

Every other threshold, flag and seed is the G+T control's. Chosen on the VAL traces above (candidates compared on
pooled occupancy): hold occupancy 0 / 8 / 23 / 16 / 7 / 19 / 15 / 10 / 1 % (entropy 2.76), whole episode 3.01 bits vs
2.10 today; it keeps the ±3.5 cut the loop sits on today and still covers 95 % of whole-episode |I| (12.6).

- Control = the flown G+T sN run (`C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_sN`), never re-flown.
- Tags `C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_yil_full_sN`, ABI 63 (no wheel change).
- New threshold lineage (thresholds are the address function); not pooled with another lineage.
- Code: branch `ctrl79-yaw-integral-ladder`; flag-off pinned byte-identical (thresholds + calibration token) by
  `tests/controller_ctrl79_yaw_integral_ladder.py`. Merged into the flying tree only between controller flights.

## Hypothesis

- **H-yil.** With the trim range resolved (cuts at ±1.5 / ±3.5 / ±6.5 instead of one cut at ±3.49), the WNN maps the
  integral code to the trim directly: the held yaw-rate setpoint stops being duty-cycled, and the standing yaw offset
  that sustains the trim shrinks.
  - **Expected vs G+T: steady lower** (hold yaw 0.219° vs MPCOF 0.078°), **err lower** (hold part plus the
    wear-dependent slope). stable / fails / pos: no change expected.
  - **Risks (stated).** (1) The distribution was measured in closed loop under today's cuts and will move once the
    cuts change. (2) The ±20.57 cut marked the large early-transient integral (0.2–0.5 % of ticks); merging it into the
    ±12 code could cost early yaw. That would show as whole yaw / err up with steady down.

## Adoption rule

Luiz's usual rule, as CTRL-77: better only if **err AND steady are lower than G+T, consistently across seeds**, and
not worse on stable / fails / pos. No magnitude bar. n=3 (s1–s3) is a screen:

- **Drop at n=3** if (A) not better than G+T on a majority of seeds on err nor on steady, or (B) fails sum > G+T's + 2,
  or (C) more fails than G+T on a majority of seeds.
- A survivor goes to n=5, n=7, n=10 on Luiz's word; adoption is decided at n=10.
- Each END: paired vs G+T same seed (stable / err / steady / pos / fails, 35 CF2.1 TEST units), as VALUES.
- After s1: a VAL trace of the winner (hold code occupancy, held-setpoint SD, hold yaw) as diagnosis, never a result.
- cells / minutes are tie-breakers only.

## Amendment 1 (Luiz 26/11/2026 ~22:10 EST): yil to n=5 + stacking arm l2byil

- **yil OVERRIDES the n=3 drop.** By the letter yil stopped on rule (C) (2/3 seeds with one more fail than G+T, both
  u123 ep16). Luiz: "Yil is not bad, it deserves n=5". yil flies s4–s5, paired vs the flown G+T s4/s5 controls
  (`…_a62_…_n0g3gz_full_s{4,5}`), same readout; adoption still decided at n=10.
- **New arm l2byil = G+T + `--gyro-xy-ladder dense_hold` + `--yaw-integral-ladder dense`** (l2b + yil; tag
  `…_a63_…_gt_l2byil_full_sN`). Question: do the two promoted levers add? l2b acts on roll/pitch rate coverage
  (pre-hold roll/pitch), yil on the hold-yaw trim (steady, wear offset) — separate mechanisms, but the CTRL-77
  factorial showed l2b + l3 was not additive on yaw, so stacking is tested, not assumed. Screen at n=3 (s1–s3), paired
  vs G+T same seed (primary, same drop rule A/B/C as above), plus l2b and yil same seed as descriptive references.
- **Order** (one controller at a time, all ABI 63): CTRL-78 (flying) → CTRL-77 n=5 (l2b, l2bl3 × s4–s5) → yil s4–s5 →
  l2byil s1–s3 → CTRL-80 (ABI 64 install).
- Driver: same `logs/controller/ctrl63/ab_yil79.sh` (arm l2byil added; digest skip adds `gyro_xy_ladder`; token
  must be `…,gyroxy:dense_hold,yawi:dense`); dry runs 2/2 (yil s4–s5) and 3/3 (l2byil s1–s3) validated 26/11.
