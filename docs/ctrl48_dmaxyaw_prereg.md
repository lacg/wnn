# CTRL-48 yaw-only delta-gain block: prereg (Luiz go 17/11/2026 ~14:10 EST)

## Hypothesis
The trim-proportional standing yaw offset comes from the yaw bank's delta-law DC gain,
G = delta_max / (1 − leak) = 2. Lowering delta_max on the yaw bank ONLY, with `--delta-max-yaw 0.5` (G = 1.67),
should keep the attitude gain of the all-banks arm D without touching the collective.

The collective gain is what moved alt/pos in D.

**Evidence** (Plane CTRL-55, 17/11):
- **Delta-gain block D, all banks 0.5, n=2 TEST:** steady −0.10°, err −0.095°, pos +0.008, alt +0.013.
- **Zero-retrain pre-check** (VAL tuning numbers, M s1-s3), ΔM:

  | variant | err | steady | alt | pos | dz_hold |
  |---|---|---|---|---|---|
  | yaw-only 0.5 | −0.067 | −0.065 | 0.000 | 0.000 | 0.000 |
  | all banks 0.5 | −0.049 | −0.064 | +0.021 | +0.015 | +0.039 |

## Arms
Both arms are rebuilt from each seed's arm-1 both_L1 `.out` and flown on ABI 57.
- **M:** the adopted base: both_L1 + `--zrank-pos-floor auto` + `--obs-yaw-from-att-est`, `--delta-max 0.6`.
  It is re-flown on ABI 57 so both arms come from the same build.
- **Y:** M + `--delta-max-yaw 0.5`. The yaw bank's decode and DAgger label use 0.5; the other banks keep 0.6.

**Seeds:** s1-s5 (base 31337001-005), flown seed by seed in the order M, Y. That is 10 runs, about 10-12 h.
Tags: `C-S2-d_wposx2_a57_rf_c200_dc_biasin_ms01_P_both_L1_dmy{M,Y}_full_sN`.
Driver: `logs/controller/ctrl48/ab_dmaxyaw.sh`. END lines go to `runs.log`, which ends with `DMY DONE`.

**Launch condition (automatic):**
- `DMX DONE`;
- the ABI-57 determinism gate PASS (flag-off ABI 57 against ABI 56);
- ctrl48-abi57 merged and pushed.

Chain script: `logs/controller/ctrl48/chain_abi57_to_dmy.sh`.

## Expected (stated in advance)
Y improves on M:
- steady better by about 0.05-0.10°;
- err better (the pre-check gives about −0.07);
- pos, alt and dz_hold unchanged (in contrast to D);
- stable and fails on par.

## Rule (TEST, 35 CF2.1 units per seed, paired by seed, Y − M)
Luiz judges the WHOLE vector (Amendment 1 of the delta-gain prereg). There is no single-coordinate guard, because pos
is the 3-D error including altitude.

Y is ADOPTED if all of these hold:
- mean steady better by > 0.03°;
- mean err not worse by > 0.03°;
- mean pos not worse by > 0.010 m;
- pooled fails ≤ 8/17500 and not significantly worse than M (exact one-sided conditional binomial, p < 0.05).

**Expectation-consistency (Luiz 16/11).** If Y keeps moving the wrong way against the expectation, it is a
not-adopt candidate at any point. Luiz decides on stopping.

## Reported
- compact pair tables (A = M · B = Y), with alt, pos_hold and dz_hold as descriptive columns;
- Y next to D from the delta-gain block on the same seed (all-banks vs yaw-only, descriptive);
- M's re-fly next to dm M (ABI 56) on the same seed, as a determinism check.

Nothing above may change after the first DMY END line.
