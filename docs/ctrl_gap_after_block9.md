# CTRL: where the student's remaining gap is, and where the 35 cm of position comes from (14/11/2026)

Investigation only (flight-dynamics). Requested by Luiz 14/11/2026 (N5 of `docs/ctrl_indi_teacher_spec.md` §11.4, plus the position question). No wheel was built, no ABI changed, nothing was trained. New code is opt-in / test-only; the full suite stays green (section D).

**Every number in this document is a VAL number** (tuning-grade): the 20 stage-select val units 163226054..163226073 of the "both" s1 run, 100 episodes each, the block-8/9 regime (cf21_brushless, L4C, translation, xy offset 0.5 m, tilt reference, motor lag 0.20 s, rotor drag 0.78, specific-force IMU). No test unit was scored here. The only test numbers quoted are ones already published.

Subject: the best student, "both" (dc_biasin_ms01 + yaw-centred ladder + yaw-integral leak, rule P), HEADLINE genomes of `C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_both_full_s1/s2/s3` (NEURONS#1 256 neurons, MEMORY#2 480, NEURONS#0 264). Caveat: the s1 genome was selected by rule P on these same val units; s2 and s3 were not. s1 is not the best of the three here, so the optimism is small, but it is there.

## Bottom line

1. **The student's gap to its label law is 0.63° of steady on val (0.80° vs 0.17°), and it is neither the teacher, nor the attitude observation, nor the action alphabet.**
2. **Roll/pitch: the student does not track the tilt reference in the hold.** It is never given that reference. It has to rebuild it from position and velocity inputs that are quantised to 8 cuts each, and at the hold scale (reference sd 0.40-0.47° across episodes) one bin of position is 0.5° of reference. Measured: the true attitude follows only +0.15 to +0.34 of the reference; the per-episode error is −0.8 to −0.96 × reference (R² 0.75-0.84 with the bias term).
3. **Yaw: still parked** at 0.38-0.48° (label law 0.075°). The first positive cut of the centred yaw ladder is at +0.42 to +0.52°, so everything inside is one bin.
4. **These frozen per-episode offsets are 73-93% of the roll/pitch error and 66-88% of the yaw error.** Removing them would leave 0.27° steady, near the label law's 0.17°.
5. **Position: 0.34-0.36 m is the task, not the controller.** The position loop is a fixed critically damped law at ω_n = 1 rad/s and the episode is 2 s, so the vehicle is 41% of the way from converged when the episode ends. On top of that the altitude PD has no integrator and holds a 0.20 m droop under the mass and thrust draws. Any controller that tracks the tilt reference gets about 0.34-0.36 m.

---

## A. What was measured, and how

| tool | what it does | path |
|---|---|---|
| rival probe (Rust, `#[ignore]` cargo test) | the label law (MPCOF>rate(h10)) on the val fixture, with opt-in what-ifs: student output alphabet, bias compensation, position windows, task changes | `rival_probe.rs`, `rival_probe_tests.rs::val_gap_and_position` |
| student tracer (Python, installed wheel, read-only) | scores each genome with the production scorer, replays the same episodes step by step, records hold samples and a per-tick trace | `scripts/ctrl_gap_student_trace.py` |
| decomposition | arithmetic on the recorded samples, block-6 conventions | `scripts/ctrl_gap_student_ana.py` |
| label replay (Rust, `#[ignore]` cargo test) | the teacher's label along the STUDENT's recorded flight, in the DAgger order (observe → sync to the real rate loop → label from the true quaternion) | `label_replay_tests.rs` |
| ladders | the students' thermometer cuts | `scripts/ctrl_gap_ladders.py` |

Gates that passed:
- The tracer's replay reproduces the production scorer's steady on every unit (asserted to 0.005°; typical 0.000-0.001°).
- The rival probe is bit-identical to the unprobed scorer with its options off, and reproduces the installed wheel's row.
- The label replay rebuilds the vehicle's inner rate loop from the student's recorded setpoints and gyro and reproduces the recorded motor command on every tick (max gap 3.1e-8, all three genomes). So the inner-loop state the teacher plans from is the vehicle's.

Frames: roll and pitch errors are TRUE Euler angles minus the tilt reference (rotations about BODY x / y); yaw is the absolute TRUE heading; `steady` is the geodesic angle to the tilt reference over t ≥ 1.6 s and includes yaw. Rate setpoints are BODY-frame rad/s. Position is WORLD frame, target = origin at the target altitude.

Scratch (not committed): `/tmp/ctrl_gap/` (npz, traces, logs), `/tmp/ctrl_indi_probe/` (fixture, logs).

---

## B. The student's gap to its label source

### B.0 The two ends (VAL)

```
controller (VAL, 20 units x 100 eps)                 | stable % | err °  | steady ° | pos m | hold roll ° | hold pitch ° | hold yaw °
-----------------------------------------------------+----------+--------+----------+-------+-------------+--------------+-----------
label law, oracle-fed (what the labels encode)       | 100.00   | 0.689  | 0.168    | 0.355 | 0.095       | 0.097        | 0.073
label law, est + bias comp (student's attitude info) | 100.00   | 0.693  | 0.171    | 0.355 | 0.095       | 0.098        | 0.075
student "both" s1                                    | 100.00   | 1.415  | 0.816    | 0.358 | 0.444       | 0.361        | 0.403
student "both" s2                                    | 100.00   | 1.443  | 0.837    | 0.359 | 0.373       | 0.362        | 0.484
student "both" s3                                    | 100.00   | 1.290  | 0.745    | 0.354 | 0.374       | 0.340        | 0.380
student mean                                         | 100.00   | 1.383  | 0.799    | 0.357 | 0.397       | 0.354        | 0.422
```
Gap to close: **steady +0.63°, err +0.69°.**

### B.1 Observation

| question | measured (VAL) |
|---|---|
| Does the student's attitude information suffice? | **Yes.** The label law fed the Mahony estimate plus the student's own bias predictor flies 0.171°, against 0.168° on the true quaternion: **0.003°**. |
| How much Mahony bias can the student not see? | mean \|bias − prediction\| = **0.017-0.019°** per axis in the student's own flights; the predictor's R² against the true bias is 0.978-0.991. |
| Does the student see the reference it is scored against? | **No.** Its target features `tgt_r`, `tgt_p` are the constant 0 (all 8 cuts at 0.0) and `roll_err` / `pitch_err` are the estimated angle against LEVEL. The tilt reference exists only inside the teacher's label and the scorer. |
| Can the student rebuild the reference from what it has? | Only coarsely. The reference is (ω_n²·e − 2ζω_n·v)/g. `e_x`, `e_y` have 8 cuts, with central bins 0.07-0.11 m wide: **0.4-0.65° of reference per bin**. `v_x`, `v_y` central bins are 0.02-0.03 m/s: 0.25-0.35° per bin. The hold reference has sd 0.40-0.47° across episodes. |

What that does in flight (per-episode hold means, each genome, roll then pitch):

```
genome | e_true ~ a*ref + b*abp          | R²          | true attitude ~ c*ref | corr(ref, abp)
-------+---------------------------------+-------------+-----------------------+---------------
s1     | -0.96 ref -1.49 abp / -0.81 ref -1.53 abp | 0.80 / 0.84 | +0.15 / +0.34 | -0.17 / -0.30
s2     | -0.78 ref -1.46 abp / -0.82 ref -1.48 abp | 0.80 / 0.82 | +0.29 / +0.31 | -0.11 / -0.26
s3     | -0.86 ref -1.33 abp / -0.78 ref -1.25 abp | 0.82 / 0.75 | +0.18 / +0.26 | -0.07 / -0.09
```
- `ref` is the tilt reference, `abp` the predicted Mahony bias (estimate − truth). A controller that tracked the reference and cancelled the bias would have a = 0, b = 0.
- **a ≈ −0.8 to −0.96: in the hold the student flies close to level and leaves 80-96% of the (small) tilt reference as error.** The true attitude follows 15-34% of the reference.
- **b ≈ −1.3 to −1.5: the bias term is over-corrected by 30-50%.** With abp sd 0.15-0.21° this is about 0.06-0.09° and is the smaller term.
- Together these two terms explain 75-84% of the per-episode roll/pitch offset.
- So "observation" splits in two. Attitude: not a limit (0.003°). **Reference: the limit for roll/pitch.** The information is present in the raw observation and is lost at the input ladder, so it is an encoding problem rather than a missing sensor.

Yaw:
- The yaw estimate is not the problem (block 6: within 0.017° of truth).
- `yaw_err` ladder (s1, 8 cuts): −2.57°, −1.18°, −0.42°, **0, 0**, +0.42°, +1.18°, +2.57°. The sign is resolved, but **nothing between 0 and ±0.42° (s1), ±0.52° (s2), ±0.48° (s3) is**, and one of the eight cuts is spent twice on 0. The student's mean \|yaw\| is 0.40 / 0.48 / 0.38°: it parks inside the first bin, at about the bin's width.
- `yaw_err_i` has the same shape (0 twice, then ±3.2 to ±3.7 in its own units).

### B.2 Quantisation of the action (alphabet and 100 Hz hold)

The label law, flown through the student's own output law: per bank, setpoint ← clamp(0.7·previous + δ, ±range), δ on the γ-shaped alphabet of 0.6·range (ranges 0.25 pwm / 4.2 / 4.2 / 2.0 rad/s, γ 1 / 2 / 2 / 2), refreshed every 10 ticks. The encode and decode are the student's own functions. The 100 Hz hold is already in every label-law row (h10).

```
arm (VAL)                                   | stable % | err °  | steady ° | pos m | fails/2000 | Δ steady ° | hold roll ° | hold pitch ° | hold yaw °
--------------------------------------------+----------+--------+----------+-------+------------+------------+-------------+--------------+-----------
label law, est + bias comp                  | 100.00   | 0.693  | 0.171    | 0.355 | 0          |  0.000     | 0.095       | 0.098        | 0.075
  + student alphabet, 128 cells/bank        | 100.00   | 0.693  | 0.171    | 0.355 | 0          | +0.000     | 0.095       | 0.098        | 0.075
  + student alphabet, 64 cells/bank         | 100.00   | 0.693  | 0.172    | 0.355 | 0          | +0.001     | 0.096       | 0.099        | 0.075
  + student alphabet, 32 cells/bank         | 100.00   | 0.694  | 0.174    | 0.355 | 0          | +0.003     | 0.096       | 0.100        | 0.075
  + student alphabet, 16 cells/bank         | 100.00   | 0.704  | 0.193    | 0.355 | 0          | +0.022     | 0.109       | 0.111        | 0.077
label law, oracle-fed                       | 100.00   | 0.689  | 0.168    | 0.355 | 0          | -0.003     | 0.095       | 0.097        | 0.073
  + student alphabet, 64 cells/bank         | 100.00   | 0.689  | 0.169    | 0.356 | 0          | -0.002     | 0.095       | 0.098        | 0.074
```
- **The action alphabet costs 0.001° of steady and 0.000° of err** at the students' sizes (64-120 cells per bank). It would start to matter only below about 16 cells per bank.
- The motor-wear run of the hold over the rate loop is not separable from the label law here; outer hold 10 vs 1 was 0.01° in `docs/ctrl_steady_gap_teacher.md` §2.
- Caveat: with the alphabet on, the teacher's shadow rate loop is not re-synchronised to the quantised setpoints (the rival scorer never does). At these step sizes that is far below the numbers shown.

### B.3 Representation: does the table reproduce its label where it flies?

Label replay on the student's visited states, 400 episodes per genome (20 per unit), decision ticks only. "Slope" and R² regress the student's setpoint on the label.

```
genome | bank | window        | label RMS | student RMS | (student - label) RMS | slope | R²    | mean |episode-mean hold error|
-------+------+---------------+-----------+-------------+-----------------------+-------+-------+-------------------------------
s1     | p    | first 0.4 s   | 0.358     | 0.513       | 0.433                 | 0.80  | 0.31  | -
s1     | p    | hold          | 0.089     | 0.034       | 0.086                 | 0.13  | 0.10  | 0.062 rad/s
s1     | q    | hold          | 0.069     | 0.041       | 0.070                 | 0.16  | 0.07  | 0.046
s1     | r    | hold          | 0.212     | 0.278       | 0.205                 | 0.90  | 0.47  | 0.060
s1     | coll | hold          | 0.034     | 0.051       | 0.025                 | 1.37  | 0.83  | 0.019 pwm
s2     | p    | first 0.4 s   | 0.357     | 0.517       | 0.454                 | 0.74  | 0.26  | -
s2     | p    | hold          | 0.070     | 0.029       | 0.068                 | 0.13  | 0.09  | 0.047
s2     | q    | hold          | 0.072     | 0.045       | 0.073                 | 0.17  | 0.08  | 0.046
s2     | r    | hold          | 0.227     | 0.260       | 0.200                 | 0.77  | 0.45  | 0.073
s2     | coll | hold          | 0.030     | 0.051       | 0.025                 | 1.68  | 0.91  | 0.020
s3     | p    | first 0.4 s   | 0.361     | 0.483       | 0.376                 | 0.85  | 0.41  | -
s3     | p    | hold          | 0.067     | 0.028       | 0.066                 | 0.11  | 0.07  | 0.047
s3     | q    | hold          | 0.063     | 0.027       | 0.061                 | 0.12  | 0.07  | 0.044
s3     | r    | hold          | 0.185     | 0.261       | 0.181                 | 1.02  | 0.52  | 0.063
s3     | coll | hold          | 0.030     | 0.051       | 0.025                 | 1.60  | 0.88  | 0.020
```
(rate banks in rad/s, collective in pwm; whole-episode rows are in `/tmp/ctrl_gap/label_replay_s*.log`.)

- **In the hold the student reproduces about 10% of the roll/pitch label** (slope 0.11-0.17, R² 0.07-0.10). Its own output is small (RMS 0.03-0.045 rad/s against a label of 0.063-0.089).
- **Most of that label error is a per-episode constant**: 0.044-0.062 rad/s. That is exactly what the measured attitude offset implies through the teacher's fitted gain of 8.7 rad/s per rad (0.35° × 8.7 = 0.053 rad/s). The label keeps asking for the correction of an offset the student does not act on, because the offset is against a reference the student cannot resolve (B.1). **B.1 and B.3 are the same finding seen from the two sides.**
- **In the transient the student's gain is about right but its output is noisy**: slope 0.74-0.90, R² 0.26-0.43, student RMS 1.35-1.45× the label's. How much of the +0.69° err this costs in closed loop is not measurable without flying a de-noised student; it is the natural suspect for the transient half of the gap.
- **Yaw is imitated much better** (slope 0.77-1.02, R² 0.45-0.52), and yet yaw is parked. Consistent with B.1: the error is inside one input bin, so a faithful imitation of an input it cannot resolve still parks.
- **Collective is over-driven** (slope 1.37-1.68 against the altitude PD). That is why the student's altitude error is smaller than the label law's (0.13 m vs 0.16 m, section C.3).

Closed-loop cost of the representation term, by subtraction: the label law with the student's attitude information, alphabet and hold flies 0.172°; the student flies 0.799°. **0.63° is what the table does not reproduce in closed loop.** The counterfactuals below split it.

### B.4 Per axis, and what removing the frozen offsets would buy

```
quantity (VAL, hold window)                          | s1    | s2    | s3    | mean  | label law
-----------------------------------------------------+-------+-------+-------+-------+----------
steady, scored                                       | 0.816 | 0.837 | 0.745 | 0.799 | 0.171
roll/pitch only (yaw set to 0)                       | 0.637 | 0.579 | 0.564 | 0.593 | -
yaw only: mean |yaw|                                 | 0.403 | 0.484 | 0.380 | 0.422 | 0.075
mean |roll error| / |pitch error|                    | .44/.36 | .37/.36 | .37/.34 | .40/.35 | .095/.098
share of roll MS in the per-episode offset           | 0.93  | 0.91  | 0.93  | 0.92  | -
share of pitch MS in the per-episode offset          | 0.85  | 0.73  | 0.90  | 0.83  | -
share of yaw MS in the per-episode offset            | 0.66  | 0.76  | 0.88  | 0.77  | -
within-episode sd, roll / pitch / yaw                | .15/.18/.31 | .14/.27/.35 | .13/.14/.19 | - | -
share of the offset that is per-UNIT, roll/pitch/yaw | .31/.15/.38 | .15/.11/.28 | .21/.16/.65 | - | -
steady if roll/pitch offsets were removed            | 0.477 | 0.548 | 0.439 | 0.488 | -
steady if the yaw offset were removed                | 0.710 | 0.654 | 0.598 | 0.654 | -
steady if every per-episode offset were removed      | 0.313 | 0.287 | 0.220 | 0.273 | -
```
- The components combine in quadrature per sample, so the rows do not add linearly.
- **Roll/pitch is the larger term** (removing its offsets: −0.31°), **yaw the smaller** (−0.15°). Both together: −0.53°, leaving 0.27°.
- The roll/pitch offsets are mostly per EPISODE, not per unit (11-31% per unit). That rules out motor-wear trim as their cause and fits the reference, which is set by where each episode starts.
- The yaw offset is 28-65% per unit: trim is a real part of it, as block 6 found.
- What is left after the offsets (0.27° against the label law's 0.17°) is within-episode motion: 0.13-0.27° sd on roll/pitch, 0.19-0.35° on yaw.

### B.5 Budget

| term | measured (VAL) | status |
|---|---|---|
| teacher / label law | 0.168° oracle-fed | not the limit (INDI route closed) |
| attitude observation (Mahony + bias predictor) | +0.003° | not the limit |
| action alphabet | +0.001° | not the limit |
| 100 Hz hold | inside the 0.168°; 0.01° per `ctrl_steady_gap_teacher.md` | not the limit |
| **roll/pitch: tilt reference not resolvable from the inputs, plus bias over-correction** | **−0.31° if removed** | the largest term |
| **yaw: parked inside the first ladder bin, plus per-unit trim** | **−0.15° if removed** | the second term |
| within-episode residual | 0.27° left vs 0.17° | third; not decomposed further |
| err (transient) | +0.69°; noisy output measured (R² 0.26-0.43), closed-loop cost not measurable without a de-noised student | open |

### B.6 Ranked levers for block 10

Expected gains are bounded by the counterfactuals of B.4; I give the bound and what I would expect to realise. Every one of these changes the address function or the feature layout, so each is a new lineage that needs fresh training and its own paired control. Significance and n are `experiment-design`'s call.

1. **Give the student the reference: attitude error against the tilt reference, bias-corrected, as one centred feature per axis.**
   - Feature: (estimated angle − abp) − tilt_ref(e, v), computed in floating point BEFORE the ladder, with a centred ladder dense near 0 (cuts around ±0.05°, ±0.15°, ±0.4°).
   - Legitimacy: it uses only signals the student already reads (position error, velocity, the estimate, abp) and the fixed, disclosed position law. No new information; it removes a subtraction the table cannot do across quantised inputs. The same construction made the bias observable in block 7.
   - Addresses both measured terms (a ≈ −0.85 and b ≈ −1.4).
   - Bound: steady 0.80° → 0.49°. Expected: **−0.15 to −0.30°**. Should also tighten position slightly (C.3).
   - Minimal variant: feed the tilt reference through the existing `tgt_r` / `tgt_p` inputs (today constant 0), so `roll_err` / `pitch_err` become errors against it. Cheaper, but leaves the bias as a separate input.
   - Feature-layout change: audit every consumer (Hard Rule 2: `feature_layout.py`, `cascade_obs.rs`, the kernel feature block, `arch_shape_from_spec`, receptive-field families).
2. **Yaw ladder dense near zero.** The centred yaw ladder has its first cuts at ±0.4-0.5° and a duplicated cut at 0; put cuts at about ±0.1° and ±0.25° on `yaw_err` (the duplicate alone frees one), and scale `yaw_err_i` so a 0.2° offset held for 0.5 s crosses a cut.
   - Bound: 0.80° → 0.65°. Expected: **−0.08 to −0.15°**. Stacks with 1 (different axis).
   - Pre-check, no training: on the recorded hold yaw (`/tmp/ctrl_gap/*.npz`), count the share of samples whose bin fixes the sign and magnitude band under the proposed cuts.
3. **Integral of the feature in 1** (a per-axis leaky integral with memory ≥ the episode, as the yaw leak): turns the residual per-episode offset into something the table can see grow.
   - Only after 1; on its own it would integrate an error against the wrong reference. Expected: **−0.05 to −0.10°** on top of 1.
4. **De-noise the transient output** (targets err, not steady): the student's rate setpoint has 1.4× the label's RMS at R² 0.3-0.4.
   - Candidates already in the codebase: a larger vote share, or more cells per bank (the alphabet is not the limit, the vote margin may be).
   - Expected gain on err: not estimable from these measurements. Pre-check: label replay on a vote-share sweep before any long run.
5. **Not worth a block:** a stronger teacher (closed), a finer action alphabet (0.001°), more attitude observation (0.003°).

1 + 2 together are bounded at 0.27° steady. If they land at half their bounds the student would be near 0.5°.

---

## C. Where the 35 cm of position comes from

### C.1 Definitions (as implemented)

- **`pos`** (`dagger_train.rs::score_classical_baseline`, and the WNN scorer's twin): per episode, the mean over ALL steps (t = 0..2 s, no window) of the Euclidean 3-D distance to the target, √(x² + y² + dz²); then the mean over episodes. Target = the origin at the target altitude. It is a time-average of a transient, not a final or steady error.
- **`alt`** is the mean \|dz\| and `dx`, `dy` the mean \|x\|, \|y\| over the same window.
- **Regimen** (`wnn/control/phased_ga.py`, `training.py`): start at rest, displaced x, y ~ U(−0.5, +0.5) m each, z ~ U(−0.3, +0.3) m, vz ~ U(−0.2, +0.2) m/s; mass ±15%, commanded collective ±10%; L4C (motor asymmetry ±20%, torque scale ±20%); no wind, no gust. **Episode length 2 s.** The mean 3-D start error is 0.42 m.
- **Position loop** (`position_loop.rs`, `PositionLoop::from_plant`): a fixed, classical, disclosed law, a_des = ω_n²·e − 2ζω_n·v with **ω_n = 1 rad/s, ζ = 1**, tilt reference = a_des/g, limit 30°. It lives in two places: inside the teacher's label (the tilt reference MPCOF tracks) and inside the scorer (the reference `err` and `steady` are measured against, `att_err_ref.rs`). The student has no explicit position law; it sees e and v as inputs.
- **Altitude loop** (`altitude_pd.rs`): a PD at ω = 2 rad/s, ζ = 1, built for the NOMINAL mass, **no integrator**, output clamped to ±0.25 pwm. It is the collective label.

### C.2 Budget (VAL, label law)

```
arm (VAL)                          | pos m | at start | 0-0.4 s | 0.4-0.8 | 0.8-1.2 | 1.2-1.6 | 1.6-2.0 (hold) | at end | mean |dx| / |dy| / |dz| | hold |dx| / |dy| / |dz| | hold speed m/s
-----------------------------------+-------+----------+---------+---------+---------+---------+----------------+--------+------------------------+------------------------+---------------
MPCOF est-fed SHIPPED              | 0.339 | 0.422    | 0.413   | 0.372   | 0.331   | 0.300   | 0.282          | 0.277  | 0.179 / 0.178 / 0.158  | 0.112 / 0.112 / 0.197  | 0.087
MPCOF est-fed + bias comp          | 0.355 | 0.422    | 0.413   | 0.377   | 0.348   | 0.328   | 0.311          | 0.303  | 0.192 / 0.191 / 0.158  | 0.138 / 0.137 / 0.197  | 0.091
MPCOF oracle-fed                   | 0.355 | 0.422    | 0.413   | 0.377   | 0.348   | 0.328   | 0.311          | 0.303  | 0.192 / 0.191 / 0.157  | 0.138 / 0.137 / 0.197  | 0.091
student "both" (mean s1-s3)        | 0.357 | -        | 0.413   | 0.382   | 0.358   | 0.329   | 0.301          | -      | 0.206 / 0.202 / 0.132  | 0.160 / 0.153 / 0.144  | -
```

**Horizontal: a transient that the episode ends in the middle of.**
- A critically damped loop at ω_n = 1 rad/s leaves e(t)/e(0) = (1 + t)·e^(−t). At t = 2 s that is **0.41**: the vehicle has covered 59% of the way. The mean of that curve over 0-2 s is **0.73**.
- With \|x(0)\| averaging 0.25 m, perfect tracking of the reference gives a mean \|dx\| of 0.25 × 0.73 = **0.182 m** and a hold-window mean of 0.116 m. Measured: 0.179-0.192 m and 0.112-0.138 m. The measurement sits on the analytic value.
- The vehicle is still moving at 0.09 m/s when the episode ends. There is no steady horizontal error: in 8 s episodes the hold \|dx\| falls to 0.019 m (C.4).
- x and y are symmetric (0.179 / 0.178).
- The oracle-fed law is 1.3 cm slower than the ideal curve (attitude loop and rotor lag), the shipped est-fed law 0.3 cm faster. That is the whole of the 0.016 m "position cost" of bias compensation: the Mahony bias happens to push the biased vehicle toward the target slightly faster than the reference asks.

**Vertical: a steady error that does not converge.**
- Mean \|dz\| 0.158 m, and it GROWS to 0.197 m in the hold window and 0.22 m in 4 s and 8 s episodes.
- Cause: the altitude PD has no integrator and is built for the nominal mass. A thrust-to-weight error ε leaves a droop of ε·g/ω² = ε × 2.45 m. Mass is ±15% and the summed motor thrust varies with the ±20% wear draws; a combined mean \|ε\| of about 0.08 gives about 0.20 m. That matches.
- This is the only part of `pos` that is a hold error rather than a transient.

### C.3 The floor for any controller on this regimen

- `err` and `steady` are measured against the position loop's own tilt reference. A controller that returns faster than the loop asks is charged attitude error for it. So within this task the horizontal trajectory is prescribed, and its mean error is fixed by ω_n, ζ, the start distribution and the 2 s window: about **0.18 m per horizontal axis**.
- The vertical part is set by the altitude PD's droop: about **0.16 m mean, 0.20 m in the hold**.
- Together: **about 0.34-0.36 m.** The published rows sit on it: MPCOF 0.341 m, bias-compensated and oracle-fed MPCOF 0.357 m, PID 0.352 m, the student 0.36 m. That is why `pos` barely moved across CTRL-48 while err and steady did.
- The student differs from the label law in two small, opposite ways: it is slower horizontally (mean \|dx\| 0.20 m vs 0.19 m, because it under-tracks the tilt reference, B.1) and better vertically (\|dz\| 0.13 m vs 0.16 m, because it over-drives the collective, B.3). Lever 1 of B.6 should recover the horizontal part, about 1 cm.
- The only part a controller can legitimately reduce WITHOUT changing the task is the vertical droop: the collective is the controller's own output and is not charged in `err`.

### C.4 What-ifs measured on the label law (VAL; est-fed + bias compensation)

```
change (VAL)                 | stable % | err °  | steady ° | pos m | fails/2000 | hold |dx| / |dy| / |dz| m | hold speed m/s
-----------------------------+----------+--------+----------+-------+------------+--------------------------+---------------
as shipped (ω_n 1, 2 s)      | 100.00   | 0.693  | 0.171    | 0.355 | 0          | 0.138 / 0.137 / 0.197    | 0.091
ω_n = 1.5 rad/s              | 100.00   | 0.886  | 0.166    | 0.309 | 0          | 0.081 / 0.080 / 0.197    | 0.092
ω_n = 2 rad/s                |  99.95   | 1.306  | 0.167    | 0.274 | 1          | 0.044 / 0.044 / 0.197    | 0.072
ω_n = 3 rad/s                |  85.05   | 3.413  | 0.239    | 0.233 | 299        | 0.012 / 0.012 / 0.199    | 0.034
4 s episodes                 | 100.00   | 0.425  | 0.155    | 0.313 | 0          | 0.064 / 0.064 / 0.220    | 0.042
8 s episodes                 | 100.00   | 0.291  | 0.157    | 0.272 | 0          | 0.019 / 0.020 / 0.222    | 0.009
```
- A stiffer position loop buys position and costs err (a larger manoeuvre to track): ω_n 1.5 is −0.046 m for +0.19° err with no fails; ω_n 2 is −0.081 m for +0.61° and the first fail; ω_n 3 is too fast for the attitude loop and the 50 ms rotor lag (85% stable).
- Longer episodes dilute the transient and expose the floor: at 8 s the horizontal error is gone and `pos` is 0.27 m, almost all of it the altitude droop (0.22 m). They also lower err for the same reason (0.29°), so they change every headline number.
- In every row the hold \|dz\| stays at 0.20-0.22 m. None of these touches the vertical term.

### C.5 Levers

| lever | expected effect on `pos` | changes the task (new comparison lineage)? |
|---|---|---|
| **Altitude integrator, or a hover-thrust estimate, in the collective law** | removes the 0.20 m droop. At 2 s I estimate `pos` 0.355 → about 0.30 m (not measured; the transient part of dz remains). At 8 s it would be most of the remaining 0.27 m. | **No.** Controller and label only. The tilt reference, the scorer and the episode set are unchanged. It does change the collective label, so students are a new training lineage, and the classical rivals should get the same law to stay comparable. |
| Position-loop ω_n 1 → 1.5 | −0.046 m measured on the label law, +0.19° err, 0 fails | **Yes.** The reference that err and steady are scored against changes; nothing before it is comparable. |
| Position-loop ω_n → 2 | −0.081 m, +0.61° err, 1 fail in 2000 | Yes, and it is at the edge of the attitude loop's bandwidth. |
| Longer episodes (4 s, 8 s) | −0.04 m, −0.08 m, by averaging a longer converged tail | **Yes.** Every headline number moves (err falls too). |
| Report position on the hold window, or at the end, next to the mean | no change to the vehicle; separates transient from hold error (0.28-0.31 m hold, of which 0.20 m is altitude) | No to the task; it is an added column. |
| Integral on horizontal position | none. There is no steady horizontal error: no wind, no gust, and rotor drag vanishes at rest. | Would change the reference. Not recommended on this regimen. |
| Feed-forward | none. The target is a fixed point; there is no trajectory to feed forward. | - |
| Velocity observation | already there (`--obs-vel-xy`, and the position loop reads velocity). | - |
| Student lever 1 of B.6 (track the tilt reference) | about −0.01 m (closes the student's small horizontal lag) | No. |

Recommendation for "position as the next frontier":
1. **First fix the altitude droop** (integrator or hover-thrust estimate in the collective law, same law for teacher and rivals). It is the only persistent error in `pos`, it is the only lever that leaves the task and every published attitude number untouched, and it is the term that will dominate as soon as episodes get longer.
2. **Then decide the task.** If position is to be a headline, the 2 s / ω_n = 1 regimen measures mostly "how far the prescribed transient has got", which no controller can change. A longer episode or a stiffer loop makes `pos` informative, at the price of a new lineage. That is Luiz's decision, and `experiment-design` should weigh in on which change keeps the most of the existing evidence.
3. Add the hold-window position column now; it costs nothing and makes the split visible in every table.

---

## D. Code added (all opt-in, default off) and checks

- `controller/rival_probe.rs`: student-alphabet quantiser, position windows (in addition to the INDI-spec probe options). `dagger_train.rs`: two more probe hooks behind `Option<&mut RivalProbe>`; the pyfunction still passes `None`.
- `controller/rival_probe_tests.rs`: `val_gap_and_position` (`#[ignore]`).
- `controller/label_replay_tests.rs`: `label_replay_on_student_states` (`#[ignore]`, test-only module).
- `scripts/ctrl_gap_student_trace.py`, `ctrl_gap_student_ana.py`, `ctrl_gap_ladders.py`.
- No ABI change, no wheel built or installed. The student flights used the installed ABI-50 wheel read-only.
- Full suite after the change: `cargo test -p ram_controller --lib --no-default-features` = 359 passed, 0 failed, 24 ignored.

## E. Not done / open

- The closed-loop cost of the noisy transient output on err (B.3). Needs a de-noised student, i.e. a run.
- Why the student over-corrects the bias (b ≈ −1.4). Lever 1 removes the need for the table to combine the two inputs, so I did not chase it.
- The within-episode residual (0.27° vs 0.17°).
- The `pos` estimate for the altitude-integrator lever is an estimate, not a measurement: the rival scorer has no such law to switch on.
- Only s1-s3 were decomposed; s4-s9 exist and can be run through the same three commands in about 1 minute each.
