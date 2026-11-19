# CTRL-48 yaw trim feed-forward block: prereg (Luiz go 17/11/2026 ~23:10 EST)

## Hypothesis
The remaining yaw steady error is the student's own yaw LAW (yaw_floor, 17/11: on the adopted base
drift is 0.018°, law 0.30° VAL against the teacher's 0.075°).

Part of that law is the TRIM the yaw rate loop rides on. Through a 2 s episode the yaw rate integrator
stays short of its rest value, and the MPCOF teacher's plan carries K_I·(I − I_ss) = K_I·L·cresid_z on
top of its proportional yaw term.

The student has to reproduce that term inside the yaw bank's table, from separately quantised inputs.
`--yaw-trim-ff` (ABI 58) adds it in floating point instead:
- sp_r = clamp(bank_r + cT·cresid_z, ±range_r), with cT = K_I·L = −0.39927 rad/s per unit cresid_z
  (cf21, n_o = 10), derived from the teacher's own gain row;
- the bank's DAgger label is recorded as sp_T − FF, so the table learns only the remainder;
- the bank's own value (the delta-law accumulator) never contains the FF.

## Arms
Both arms are rebuilt from each seed's arm-1 both_L1 `.out` and flown on ABI 58.
- **B:** the base, re-flown on ABI 58.
- **F:** B + `--yaw-trim-ff`.

**Base** = what `logs/controller/ctrl48/l2yp_investigation/ff_precheck_verdict.txt` recommends:
- **M:** both_L1 + `--zrank-pos-floor auto` + `--obs-yaw-from-att-est`, `--delta-max 0.6`;
- **Y:** M + `--delta-max-yaw 0.5`.

If the verdict has no `BASE:` line, the base is M. The chain records the base it used in the launch
addendum below.

**Seeds:** s1-s5 (base 31337001-005), flown seed by seed in the order B, F. That is 10 runs, about 8-9 h.
Tags: `C-S2-d_wposx2_a58_rf_c200_dc_biasin_ms01_P_both_L1_ff{B,F}_full_sN`.
Driver: `logs/controller/ctrl48/ab_yawff.sh`. END lines go to `runs.log`, which ends with `FF DONE`.

**Launch condition (automatic, `logs/controller/ctrl48/chain_abi58_to_ff.sh`):**
- `DMY DONE`, and the CTRL-56 watcher has finished;
- verdict line 1 is `GO`, with no unapplied `SPEC-DELTA`;
- the merged branch passes `cargo test`;
- the determinism gate PASSes for flag-off ABI 58 ×2 vs ABI 57;
- a flag-on smoke run exits 0;
- ctrl48-abi58 is merged and pushed.

## Expected (stated in advance)
The expectation is the one the FF pre-check's verdict file states. The chain freezes it verbatim in the
launch addendum below before the first `FF START`.

The direction is: F improves steady through the yaw axis, with err, pos, alt and dz_hold unchanged, and
stable and fails on par.

## Rule (TEST, 35 CF2.1 units per seed, paired by seed, F − B)
Luiz judges the WHOLE vector. There is no single-coordinate guard, because pos is the 3-D error including
altitude.

F is ADOPTED if all of these hold:
- mean steady better by > 0.03°;
- mean err not worse by > 0.03°;
- mean pos not worse by > 0.010 m;
- pooled fails ≤ 8/17500 and not significantly worse than B (exact one-sided conditional binomial, p < 0.05).

**Expectation-consistency (Luiz 16/11).** If F keeps moving the wrong way against the expectation, it is a
not-adopt candidate at any point. Luiz decides on stopping.

## Reported
- compact pair tables (A = B · B = F), with alt, pos_hold and dz_hold as descriptive columns;
- the per-axis split of each headline (ABI 56 `axis_out`, `logs/controller/ctrl48/dmax_yaw/dmy_axis.py
  --genome WINNER`), VAL and TEST: the FF must show up on the yaw axis;
- each headline's stage, plus the fit-best candidate per stage on VAL. Y s1 (17/11) showed that an n=1
  headline gap can be the stage-select draw (rule P's p99 tail), not the lever;
- B's re-fly next to the same base's previous-ABI run on the same seed, as a determinism check.

Nothing above may change after the first FF END line.

## Launch addendum (written by chain_abi58_to_ff.sh, 18/11/2026 04:06 EST, before the first FF START)
- Base: **Y** (from the verdict file's BASE line; absent ⇒ M).
- Determinism gate: PASS (flag-off ABI 58 ×2 vs ABI 57; logs/controller/detgate/ff_abi5*).
- Expectation, frozen verbatim from /Users/lacg/wnn/logs/controller/ctrl48/l2yp_investigation/ff_precheck_verdict.txt:
```
PREDICTED_HOLD_YAW_OFFSET_DEG (calibrated model, M s1-s5 genomes, 6 VAL units x 100 eps each, mean): base_M_G2 0.288 | Y_G1.67 0.235 (measured) | FF_G2 0.206 | FF+Y_G1.67 0.206
FF+Y_VS_BEST_NONFF_PER_SEED: -16% +11% -9% -34% -21%  (mean -12%, 4/5 better)
FF_G2_VS_BEST_NONFF_PER_SEED: -35% +3% +4% -15% -20%  (mean -12%, 3/5 better)
EMULATION_UNCALIBRATED (block-9 traces, heading=true yaw, M s1-s3 wiring, vote-0.1): FF cuts the predicted offset 60-74% vs no-FF at the same G (relative only; absolute level over-reads 3-5x)
WHAT_THE_BANK_MUST_DELIVER_AT_HOLD: |sp_need| 0.139-0.172 rad/s -> with FF |sp_need - cT*cz| 0.015-0.021 rad/s (-88%)
VALIDATION: decode identity sp = G*B exact (fitted 1.97-2.01 at G2, 1.66-1.67 at G1.67); offset ~ need-part R2 0.35-0.69 (slope k 0.10-0.32, used as per-seed calibration); same-genome G 2->1.67: mean predicted -0.049 vs measured -0.054 deg, sign right 3/5 seeds (s3, s4 not predicted)
NOT_PREDICTED_WORSE: err (FF delivers the teacher's own trim term exactly; no transient model); pos (yaw-only lever); clamp margin max |ff| 0.51 vs range 2.0 rad/s; ff noise at hold 0.025 rad/s sd vs held-sp sd 0.09-0.13
```

## Addendum 2 — re-launch on base D (Luiz 18/11/2026 ~21:00 EST, before any ffD run)
- **Base = D** (both_L1 + floor + `--obs-yaw-from-att-est`, `--delta-max 0.5` on all banks). Luiz PRE-APPROVED D as the
  Y/D winner for tonight; the n=10 fitness-only verdict (docs/ctrl48_n10_ymd_prereg.md) is computed overnight and Luiz
  reviews both in the morning. He may stop this block then; runs already flown are kept.
- **Only arm F flies** (`--yaw-trim-ff` on base D), seeds **s1-s10**, seed-major. **B is not re-flown:** the rebuilt
  base-D argv is identical (except checkpoint paths) to the flown D runs — verified for s6 against n10 D s6 — and every
  ABI step since 56 passed the flag-off determinism gate (58 vs 57, 59 vs 58 tonight). So B = the existing D rows:
  dm D s1-s5 (ABI 56) and n10 D s6-s10 (ABI 58). If tonight's ABI 59 gate does NOT pass, F flies on ABI 58 instead.
- Tags: `C-S2-d_wposx2_a<ABI>_rf_c200_dc_biasin_ms01_P_both_L1_ffD_F_full_sN`. The killed base-Y F s1 is not a result.
- **Rule unchanged** (F − B, paired by seed, TEST, 35 units): steady better by > 0.03°, err not worse by > 0.03°, pos not
  worse by > 0.010 m, fails ≤ 16/35000 pooled (the n=5 bound 8/17500 scaled) and not significantly worse than B.
  Read at n=10; the morning read at ~n=5-7 is a direction only. Expectation-consistency (Luiz 16/11) applies.
- Launched automatically by `logs/controller/ctrl48/overnight_d_ff.sh` after the CTRL-58 chain ends.
