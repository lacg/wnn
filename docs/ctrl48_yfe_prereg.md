# CTRL-48 heading block — prereg (Luiz go 16/11/2026 ~19:20 EST)

**Hypothesis.** The student computes its yaw heading as `yaw_heading += gyro_z·dt`. Once the vehicle tilts, the body
z rate is not the world yaw rate, so the heading picks up a kinematic drift of about 0.17°. The DAgger yaw label is
built on the TRUE yaw. The mismatch between the two produces the per-episode yaw-command bias that sets C's ~0.3°
standing yaw offset.

`--obs-yaw-from-att-est` (ABI 55) takes the heading from the Mahony filter the student already runs. That filter reads
measured gyro and accelerometer only, starts from the same q0 anchor, and drifts about 0.018°.

Evidence (all on Plane CTRL-48, 16/11):
- yaw floor: MPCOF yaw steady is 0.071° fed the true state and 0.075° fed the estimate; C is at 0.403°;
- lever 1: the yaw error splits about 80% yaw law and 20% drift; swapping the heading into the existing genomes,
  without retraining, gives TEST steady −0.036° (5/5 seeds);
- lever 2: the residual yaw bias is explained 98-99% once TRUE yaw is added; the trim feed-forward was NO-GO.

**Arms** (both arms identical except for the flag; argv rebuilt from each seed's arm-1 both_L1 `.out`, as in 10b):
- **C:** both_L1 + `--zrank-pos-floor auto`. This is the 10b C recipe re-flown on ABI 55, so both arms come from the
  same build.
- **M:** C + `--obs-yaw-from-att-est`.

Seeds s1-s5 (base 31337001-005), flown seed by seed in the order C, M. That is 10 runs, about 10-12 h.
Tags: `C-S2-d_wposx2_a55_rf_c200_dc_biasin_ms01_P_both_L1_yfe{C,M}_full_sN`.
Driver: `logs/controller/ctrl48/ab_yfe.sh`. END lines go to `logs/controller/ctrl48/runs.log`; the driver ends with
`YFE DONE`.

**Expected (stated in advance).** M improves on C:
- steady better by at least about 0.04° (the no-retrain swap gave −0.036 on TEST), and possibly more, since retraining
  also removes the label/heading mismatch behind the yaw bias;
- err not worse (the no-retrain swap gave −0.030);
- stable, fails and pos on par.

**Rule** (TEST, 35 CF2.1 units per seed, paired by seed, M − C).

M is adopted if all of these hold:
- mean steady better by > 0.03°;
- mean err not worse by > 0.03°;
- mean pos not worse by > 0.010 m;
- pooled fails ≤ 8/17500 (the 10b gate rate of 17/35000) and not significantly worse than C (exact one-sided conditional binomial, p < 0.05).

**Expectation-consistency (Luiz 16/11).** If M keeps moving the wrong way against the expectation, it is a
not-adopt candidate at any point, without waiting for the intervals to close: steady flat or worse on most seeds, or
err or fails worse. Luiz decides on stopping.

**Reported.** Compact pair tables (A = C · B = M) with pos_hold and dz_hold. Per-seed values are also shown against
the 10b C row, as a descriptive check that the ABI 55 re-fly reproduces it. Nothing above may change after the first
YFE END line.
