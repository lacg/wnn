# CTRL-48 block 6: why mlab lost, why biasin won, and where the last 0.17° is (12/11/2026)

Investigation only (flight-dynamics). Nothing was trained, no code or wheel changed. Scratch: /tmp/steady_block6/ (probe.py, ana.py, thr.py, *.npz, *.log).

**Bottom line.**
- **The remaining student-vs-teacher steady gap is in YAW, not in roll/pitch.**
  - On roll/pitch, biasin already beats the MPCOF teacher: 0.62° vs 0.67°.
  - The student parks its heading at a per-episode yaw offset of about 0.48°. The teacher's is 0.08°.
  - Yaw is +0.22° of the +0.17° gap; roll/pitch is −0.05°.
- **mlab lost because estimate-built labels teach the student to regulate its Mahony estimate, as the teacher does.**
  - The student cannot regulate that estimate as tightly as the teacher, so it ends up with both the bias and its own estimate-frame error.
  - Roll/pitch rose by +0.12° on 3/3 seeds.
- **biasin won because the bias became observable.** The predictor matches the student's actual bias with R² 0.98-0.99 and slope 1.00, and the student acts on it.

## 0. Method and validation
- **Genomes.** Each run's published HEADLINE genome, loaded from its stage checkpoint via `_load_stage_genome`, scored on the 35 test units (99990101-135) × 100 eps with train-seed thresholds. This is `_holdout_report`'s path through `ControllerEvaluator.score_genomes`.
- **Probe.** A wrapper around `score_controllers_cpu` captures the exact scorer kwargs. probe.py then re-flies the same episodes from Python in `cpu_score::rollout_one`'s order: reset, estimator seed, decode seed, plant/disturbance draw, vertical/horizontal obs, `step`, `sim.step`, then the tilt-reference error.
- **What it records** in the steady window (t ≥ 1.6 s), all as Euler errors against the tilt reference:
  - the true attitude error;
  - the Mahony-estimate attitude error;
  - the bias b = estimate − true (roll/pitch);
  - true yaw;
  - the `abp_r` / `abp_p` features.
- **Validation (gate).**
  - The Python replay reproduces the scorer's steady to within 0.002° per seed, and to the 2 decimals in runs.log for all 12 runs. Examples: dc s1-s3 0.842 / 1.098 / 0.908; biasin 0.896 / 0.913 / 0.744; mlab 1.043 / 1.135 / 0.928.
  - Fail counts reproduce too: biasin s2/s3 3/0, dc 5/1/1, mlab 10/12/3.
  - Side note: the CPU scorer now matches the GPU-published per-seed rows (biasin s3 seed 99990101: 0.9409 vs the logged 0.94). The CPU≠GPU mismatch noted in docs/ctrl_steady_checks.md does not reproduce at ABI 49.
- **Teacher numbers** use the same decomposition on the 09/11 MPCOF>rate(h10) est-fed trace (/tmp/steady_probe/dec_est.npy, docs/ctrl_steady_gap_teacher.md).
- **Frames.**
  - Roll/pitch errors are ZYX-Euler of the vehicle vs the position loop's tilt reference. The tilt reference is built from world-frame position and velocity, but roll/pitch are rotations about body axes (yaw ≤ 5°, so the frames coincide to cos 5° = 0.996).
  - Yaw is the absolute heading vs the scorer's yaw-0 reference.
  - `steady` is the geodesic angle, which includes yaw. The small-angle magnitude √(r²+p²+y²) matches it to 0.001° on every run, so the components below add up correctly.

## 1. Decomposition (test, 35 units × 100 eps, mean over seeds s1/s2/s3)
| controller | steady (scored) | roll/pitch only (yaw = 0) | steady with yaw at teacher level | mean \|yaw\| | mean \|roll/pitch true err\| | mean \|est-frame err\| | mean \|bias b\| | compensation slope k (r/p) | uncompensated bias share of true-err var (r/p) |
|---|---|---|---|---|---|---|---|---|---|
| MPCOF teacher (est-fed) | **0.684** | 0.673 | 0.684 | **0.078** | .44 / .42 | .14 / .12 | .36 / .34 | −.21 / −.20 | .90 / .91 |
| dc (control) | 0.950 | 0.724 | 0.734 | 0.491 | .50 / .41 | .41 / .34 | .21 / .18 | −.34 / −.24 | .40 / .36 |
| dc_mlab | 1.036 | **0.844** | 0.852 | 0.466 | .56 / .50 | .42 / .37 | .24 / .24 | −.47 / −.36 | .54 / .55 |
| dc_biasin | **0.852** | **0.620** | **0.631** | 0.476 | .40 / .39 | .39 / .36 | .15 / .13 | +.40 / +.20 | .06 / .12 |
| dc_leak | 0.869 | 0.671 | — | 0.427 | .44 / .40 | — | — | −.32 / −.23 | — |

- **How to read k.** Regress the estimate-frame error on the bias, e_est = k·b + r, so the true error is e_true = (k − 1)·b + r.
  - k = 0 means the controller regulates its estimate and inherits the full bias.
  - k = 1 means it fully cancels the bias.
  - k < 0 means its estimate-frame error even leans against the bias. This is dynamic lag, because the bias and the tilt reference both follow the horizontal acceleration.

Per seed (s1 / s2 / s3):

| | steady | roll/pitch only | \|yaw\| |
|---|---|---|---|
| dc | .842 / 1.098 / .908 | .694 / .724 / .755 | .379 / .707 / .388 |
| dc_mlab | 1.043 / 1.135 / .928 | .904 / .827 / .800 | .398 / .649 / .350 |
| dc_biasin | .896 / .913 / .744 | .657 / .604 / .599 | .513 / .577 / .340 |
| dc_leak | .813 / 1.000 / .795 | .688 / .670 / .655 | .331 / .600 / .349 |

### biasin vs teacher, attributed
| | gap | from yaw | from roll/pitch |
|---|---|---|---|
| mean | **+0.168** | **+0.221** | **−0.053** |
| s1 | +0.213 | +0.230 | −0.017 |
| s2 | +0.230 | +0.298 | −0.068 |
| s3 (best) | +0.061 | +0.133 | −0.072 |

- Method: "from yaw" = scored − (same flight with yaw scaled to the teacher's 0.078°), and "from roll/pitch" = that counterfactual − 0.684.
- **With teacher-level yaw, biasin would read about 0.63°, below the teacher's 0.68°.** On roll/pitch it is already better than the teacher, which regulates its estimate (k ≈ −0.2) and so carries 90% of the 0.52° Mahony bias (teacher doc §2).
- This also explains docs/ctrl_steady_checks.md §1:
  - The student's excess there did not shrink at xy 0; it grew from +0.35 to +0.48.
  - Yaw parking does not depend on translation. The roll/pitch bias does, and it disappears at xy 0.

## 2. Mechanism verdict: mlab vs biasin, and the 09/11 lead hypothesis
**mlab (`--dagger-label-attitude estimate`): worse roll/pitch on 3/3 seeds (+0.21 / +0.10 / +0.05°) and 2.5× the soft fails (25 vs 7). Yaw is unchanged (0.466 vs 0.491).**
- Estimate-built labels make the target "hold the estimate on the reference", which is the teacher's own policy.
  - The teacher can afford that because it holds its estimate to 0.13°.
  - The student holds its estimate-frame error to only 0.37-0.42°, the same as dc.
  - So mlab gets the teacher's bias exposure (k −0.47 / −0.36, uncompensated share 0.54) on top of the student's sloppier estimate-frame error.
  - It also flies a slightly more accelerating hold (|b| 0.24 vs 0.19-0.21).
- True-quaternion labels (dc) carry the term −K·b. The student sees no direct input for it, but the term still pulls against the bias: dc's uncompensated share is 0.36-0.40 vs 0.54.
- **The bias term in the label was not harmful noise. It was a correction signal that the student partly used.**
- **Fails.** All soft (0 divergences). They occur in episodes whose Mahony bias peaks at 3-9° (all-episode p99 2.3-2.6°). mlab labels teach the student to trust the estimate exactly when it is most wrong.

**biasin (`--obs-att-bias-pred`): the label term becomes observable.**
- In the student's own flights the predictor tracks the true bias with R² 0.986-0.993 and slope 0.998-1.003 on both axes and all seeds. That is better than its unit test's > 0.8 gate.
- The student uses it:
  - k turns positive (r +0.51 / +0.24 / +0.45, p −0.13 / +0.72 / +0.03);
  - the uncompensated share falls to 0.01-0.18;
  - the estimate-frame error follows abp with the same slope as k.
- It also flies a gentler hold (|b| 0.13-0.15). Net roll/pitch: 0.620 vs dc 0.724 (−0.10, 3/3 seeds).

**What this says about the lead hypothesis (docs/ctrl_steady_gap_teacher.md §4.1):**
1. The mechanism is right: the true-q label carries a bias term the student cannot see.
2. The proposed fix had the wrong sign. Make the term observable (biasin); do not delete it (mlab).
3. It was aimed at the wrong axis for what is left. After biasin, roll/pitch is better than the teacher, and the whole remaining gap is yaw.

## 3. The yaw offset (what is left)
- **It is not a yaw-observability problem.**
  - The Mahony yaw estimate is within 0.017° of true yaw in the window (biasin s1).
  - The yaw-anchored `yaw_err` is seeded from the true initial yaw and dt-integrated gyro, so it is equally clean over 2 s.
- **It is a frozen per-episode offset.**
  - 76-90% of the yaw mean-square is the episode-mean offset (within-episode sd only 0.17°). |yaw| is the same at t = 1.6, 1.8 and 2.0 s, so it does not decay.
  - 23-64% of it is per-unit (motor-wear yaw-torque trim).
  - |yaw| tracks the training seed, not the arm: s2 is the worst seed in dc, mlab, biasin and leak (0.58-0.71). That points to the seed's yaw ladder or cells.
- **It is invisible at hold scale.** The `yaw_err` thermometer is a quantile fit to the calibration distribution (±2.8°). Its centre bin is 0.68-0.78° wide and not centred on 0:

| seed | centre bin of `yaw_err` |
|---|---|
| s1 | [−0.25, +0.43]° |
| s2 | [−0.22, +0.56]° |
| s3 | [−0.61, +0.10]° |

  - 30-63% of hold samples sit inside that one bin.
  - s2's mean signed `yaw_err` is −0.32 to −0.38°, just outside its offset bin. That is where the trim parks it.
- **No real integrator.** `yaw_err_i` is a per-step leaky sum with leak 0.99 at 1 kHz, a 0.1 s memory. It is a scaled low-pass copy of `yaw_err`, not an integrator over the 2 s episode.
  - So the yaw channel is effectively P-only against a per-unit torque trim, behind a ~0.7° dead zone. A steady offset is the textbook result.
  - The teacher removes the trim through its offset-free d̂ / rate integrator. Its hold yaw command is 99.9% explained by Ĩ (teacher doc §1).
- **Leak pre-check (dc_leak, `--integral-leak 0.999`).**
  - |yaw| fell on 3/3 seeds: 0.379/0.707/0.388 → 0.331/0.600/0.349 (−13% on average).
  - Roll/pitch also improved: 0.724 → 0.671.
  - But leak alone stops at 0.43°: the 1 s memory is still requantised by a transient-dominated quantile ladder.
  - Cost: s2 had 18 soft fails (vs 1). The leak is global and also lengthens roll_i/pitch_i.

## 4. Proposed arms (paired vs the same seed's a49 `dc_biasin`; none queued)
**1. `dc_biasin_leak` = dc_biasin + `--integral-leak 0.999` (flag-level).**
- Hypothesis: biasin (roll/pitch bias) and leak (yaw offset, roll/pitch integral) act on orthogonal error components, so they stack.
- Expect steady ≈ 0.80. The yaw share falls about 13% and roll/pitch holds about 0.62.
- Watch item: the s2 soft fails that leak alone produced (18/3500).
- Pre-check (done here): dc_leak lowers |yaw| on 3/3 seeds and roll/pitch on 3/3, so the direction is established. After the run, re-run probe.py on the 3 winners and check the yaw column moves without k collapsing.

**2. Yaw-family centred ladder (minimal code change): `--calib-ladder centred` applied only to `yaw_err`, `yaw_err_i` (and `cresid_z`), e.g. a `centred-yaw` choice in `CALIB_LADDERS`.**
- Hypothesis: the parking is the off-centre ~0.7° quantile centre bin. A cut at 0 plus near-zero density (±0.1°, ±0.25°, …) makes the sign and size of a hold yaw error addressable.
- Expect |yaw| from about 0.48 to ≤ 0.2°, so steady ≈ 0.65, i.e. teacher parity.
- Why yaw-only: the global centred ladder lost in block 3 (a45 centred 1.12/1.08/1.01 vs rf_c200). It re-quantises roll/pitch/translation features that do not need it.
- Pre-check (offline, no training): fit `_centred_thresholds` on the run's calibration samples for the two yaw features. Then count, on the recorded hold yaw from probe.py, the share of samples whose bin determines the sign (today 37-70% are sign-resolved outside the centre bin; target > 90%).
- Cost: thresholds are part of the address function, so this is a new lineage and needs fresh training.

**3. Yaw-only true integrator (minimal code change): a separate leak for `yaw_err_i`, e.g. `--yaw-integral-leak 0.9995` (2 s memory), leaving roll_i/pitch_i at 0.99.**
- Hypothesis: the offset is P-only steady-state error against per-unit yaw trim (23-64% of yaw MS is per-unit). An integrator with memory ≥ the episode removes it, which is the student-side analogue of the teacher's Ĩ.
- Isolating it to yaw avoids the roll/pitch side effect behind dc_leak's s2 fails.
- Expect |yaw| to fall from about 0.48 to ≤ 0.25 (steady ≈ 0.70), more if combined with arm 2.
- Pre-check: on the probe traces, integrate the recorded hold `yaw_err` with leak 0.9995 and confirm that a 0.3-0.5° offset drives the feature past its first fitted threshold within 0.5 s. With 0.99 it saturates at 100·e, inside the centre bin of the 0.99 ladder.

**Not proposed:** mlab in any combination. It is the wrong sign (§2), and stacking it with biasin would re-teach estimate regulation while biasin is teaching bias cancellation.

Variance and significance (n = 3 seeds, paired) are `experiment-design`'s call. Run mechanics are `controller`'s.
