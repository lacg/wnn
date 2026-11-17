# CTRL-48 delta-gain block: prereg (Luiz's go 17/11/2026 ~09:30 EST)

## Hypothesis
The CTBR delta law trains each bank on the teacher's ABSOLUTE command, but the student emits that command as an
increment, `delta_max × command`, and holds it with leak 0.7. Its DC gain on the label is therefore
G = delta_max / (1 − leak) = 0.6 / 0.3 = 2.

To deliver the exact yaw trim the airframe needs, the yaw angle must settle at a standing offset of about
trim·(1 − 1/G)/|K_θ|. On the new base (M), the per-episode yaw bias is α × teacher trim, with α ≈ 0.31 measured
against 0.5 predicted, and R² 0.51-0.70 against cresid_z. Coverage was ruled out (Plane CTRL-48, 17/11).

Lowering `delta_max` to 0.5 (G = 1.67) should shrink that offset.

**Zero-retrain pre-check** (existing M genomes re-scored, VAL, 5 seeds; a tuning number, not a report number):
- steady 0.406 → 0.342 (5/5 better, −0.04 to −0.10);
- err 5/5 better (−0.01 to −0.09);
- yaw hold 0.303 → 0.232;
- stable 100%, roll/pitch hold unchanged;
- pos/alt not measured.

## Arms
Both arms are rebuilt from each seed's arm-1 both_L1 `.out`, like every CTRL-48 block, and flown on ABI 56.
- **M:** the adopted base: both_L1 + `--zrank-pos-floor auto` + `--obs-yaw-from-att-est`, with `--delta-max 0.6`.
  It is re-flown on ABI 56 so both arms come from the same build. This is the s2 lesson: an ABI-52 vs ABI-55 fork
  on 1 of 5 seeds.
- **D:** M with `--delta-max 0.5`. The driver substitutes the value in place and asserts it; leak stays 0.7.

**Seeds:** s1-s5 (base 31337001-005), flown seed by seed in the order M, D. That is 10 runs, about 10-12 h.
Tags: `C-S2-d_wposx2_a56_rf_c200_dc_biasin_ms01_P_both_L1_dm{M,D}_full_sN`.
Driver: `logs/controller/ctrl48/ab_dmax.sh`. END lines go to `runs.log`, which ends with `DMX DONE`.

## Expected (stated in advance)
D improves on M:
- steady better by 0.04-0.10°;
- err not worse (the pre-check says better);
- yaw-axis hold better (the `axis_out` split is reported as a descriptive check);
- stable, fails, pos and alt on par.

Retraining under the new gain may move these numbers relative to the zero-retrain pre-check.

## Rule (TEST, 35 CF2.1 units per seed, paired by seed, D − M)
D is ADOPTED if all of these hold:
- mean steady is better by > 0.03°;
- mean err is not worse by > 0.03°;
- mean pos is not worse by > 0.010 m;
- mean alt is not worse by > 0.010 m;
- pooled fails ≤ 8/17500 and not significantly worse than M (exact one-sided conditional binomial, p < 0.05).

**Expectation-consistency (Luiz 16/11).** If D keeps moving the wrong way against the expectation (steady flat or
worse on most seeds, or err, fails, pos or alt worse), it is a not-adopt candidate at any point. Luiz decides on
stopping.

## Reported
- compact pair tables (A = M · B = D) with pos_hold and dz_hold;
- M's re-fly next to the ABI-55 yfe M runs on the same seed, as a descriptive determinism check;
- the per-axis split from `axis_out` (CPU) for both arms on VAL, as a descriptive check.

Nothing above may change after the first DMX END line.
