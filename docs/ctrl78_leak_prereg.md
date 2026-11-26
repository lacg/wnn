# CTRL-78 — intermediate yaw-bank leak on the G+T base — pre-registration (25/11/2026)

Luiz 25/11 evening: "queue 2" (the intermediate leak), behind the CTRL-77 factorial. Written before any flight.

## Why

CTRL-74 (L1, `--delta-leak-yaw 0.95`, gain-matched) held the yaw trim as designed (hold setpoint SD 0.162 → 0.077 rad/s
on the 3 VAL units) but lost on TEST s1 (err 1.29 vs 1.01, steady 0.42 vs 0.27). VAL diagnosis (posted on CTRL-74): the
leak also sets the yaw accumulator's time constant, 0.7 → ~28 ms, 0.95 → ~195 ms; the WNN does not observe the
accumulator, so the setpoint builds late and is not unwound at the yaw zero crossing (overshoot ×2, hold damping lost).

The leak is therefore a trade dial between trim chatter (low leak) and output lag (high leak). L1 measured one end.

## Arms (model-free; output decoding only; no observer, no MPCOF part)

Control = the flown G+T sN run (`C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_sN`, never re-flown).
Each arm argv = that run's argv + `--delta-leak-yaw λ` + the in-place swap of `--delta-max-yaw` + the save paths.
Gain match: delta_max_yaw / (1 − λ) = 1.667 (today's 0.5 / 0.3), to 3 decimals.

```
arm    --delta-leak-yaw   --delta-max-yaw   gain     time constant −10 ms / ln λ
G+T    (0.7, global)      0.5               1.667    28 ms
lk85   0.85               0.25              1.667    62 ms
lk90   0.9                0.16667           1.667    95 ms
l1     0.95               0.08333           1.667    195 ms   (CTRL-74 / CTRL-77, already flying)
```

Tags `C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_<arm>_full_sN`, ABI 63 (no wheel change), seeds s1–s3,
seed-major, 6 flights (~6 h). Driver `logs/controller/ctrl63/ab_leak78.sh` ('L78' lines in `ctrl48/runs.log`), started by
`logs/controller/ctrl63/leak78_after_f77.sh` when the n=3 factorial logs 'F77 DONE' (stands down on 'F77 ABORT').

## Hypothesis

- **H-lk.** At 62–95 ms the trim still persists across several actions (sustaining delta 0.15·s / 0.10·s instead of
  0.30·s), so the hold chatter falls, while the lag stays short enough for the yaw transient.
  - **Expected vs G+T: steady lower, err not higher.** stable / fails / pos: no change expected.
  - **Risk (stated):** the dial may be monotone, i.e. both arms land between G+T and L1 on err and steady and neither
    wins. That outcome closes the leak dial as a lever on its own (a leak then needs the accumulator observed).

## Adoption rule

Luiz's usual rule, same as CTRL-77: an arm is better only if **err AND steady are lower than G+T, consistently across
seeds**, and it is not worse on stable / fails / pos. No magnitude bar. n=3 is a screen:

- **Drop at n=3** if (A) not better than G+T on a majority of seeds on err nor on steady, or (B) fails sum > G+T's + 2,
  or (C) more fails than G+T on a majority of seeds.
- A survivor goes to n=5 (s4, s5), then n=7, n=10 on Luiz's word. Adoption is decided at n=10.
- Reported per END: paired vs G+T same seed (stable / err / steady / pos / fails over the 35 CF2.1 TEST units), plus
  the leak dose-response row set (G+T, lk85, lk90, l1) as VALUES. VAL traces are diagnosis only, never a result.
- cells / minutes are tie-breakers only.
