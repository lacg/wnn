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
