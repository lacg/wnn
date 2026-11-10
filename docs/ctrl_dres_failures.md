# rf_c200_dres: the 4 lost held-out episodes (diagnosis, 09/11/2026)

Arm `C-S2-d_wposx2_a45_rf_c200_dres_full_s{1,2,3}` loses 4 of 10,500 held-out episodes
(35 CF2.1 units x 100 episodes x 3 seeds) on the FIXED scorer (controller ABI 47,
`/tmp/ctrl_fix_site`). The MPCOF cascade teacher (`MPCOF>rate(h10)[est]`) loses 0 on the
same episodes. Diagnosis only: no code changed, nothing queued.

## Definitions checked in the scorer (not assumed)

- **Stable** (`cpu_score.rs::rollout_one`, Metal twin): the episode is stable if it is
  NOT diverged (`|omega| > 50 rad/s` or non-finite) AND its **episode-mean** attitude error is
  <= 5.0 deg. The episode lasts 2000 steps x 1 ms = 2 s, and the initial transient counts.
- **Attitude error** with `--att-err-ref tilt` (G5) is measured against the position loop's
  tilt reference: a clamped PD with omega_n=1, zeta=1, max 30 deg (`position_loop.rs`). So an
  episode that builds lateral speed is scored against a braking tilt of about 2*v/g rad
  (0.45 m/s gives about 5.3 deg) that the vehicle must actually fly.
- **L4C has no gusts, no sensor dropout and no observation delay.** Its stochastic terms are:
  - +/-20% per-episode, per-axis torque-scale jitter;
  - +/-20% per-unit motor wear, one draw per report seed;
  - ADIS16448 gyro/accel noise and gyro bias walk.
  So "gust/dropout/delay" triggers cannot occur in this recipe.

## Method (all measured)

1. **GPU per-episode rows.** Each headline genome was re-scored as flown
   (`scripts/ctrl_rescore_abi47.py` path) with the GPU scorer's per-episode sink armed.
2. **Capture and CPU re-score.** The exact scorer inputs (controller object, q0/omega0, every
   kwarg) were captured by patching `wnn.control._accel.score_controllers_metal`. The same
   inputs were then re-scored on the CPU scorer.
3. **Step-level trace.** The same Rust objects (`AttitudeSim`, `WnnController`) were driven
   from a scratch script that mirrors `rollout_one` step for step (stage-1/2, quad path). It was
   accepted only because it reproduces the scorer per episode. Scratch scripts are in
   `/tmp/dres_diag/` and are not committed.
4. **Teacher per episode.** The MPCOF cascade was scored with the exact kwargs
   `cascade_rival_metrics` builds, per episode by prefix-differencing
   `score_classical_baseline`, which keeps each episode's index and therefore its disturbance
   seed. Its time profile came from truncating `steps`. There is no per-step teacher trace
   for translation (`trace_classical_baseline` is attitude-only), so teacher command
   magnitudes were NOT measured.
5. **Same episodes everywhere.** WNN and teacher share q0, x0/y0, mass, `dist_seed` and the
   motor-wear vector (verified equal on all 3 seeds).

## The 4 failures

Notes on the table:
- Peak = largest 25 ms-window mean error vs the tilt reference.
- Stall = the 0.25-1.0 s window.
- Siblings = the other dres seeds on the same episode.

| genome | unit (report seed, index) / ep | err (bound 5.0) | trigger | peak err, when | stall err | siblings err | MPCOF cascade err (rank/100), peak |
|---|---|---|---|---|---|---|---|
| dres s2 MEMORY#1 | 99990112 (#11) / 86 | 5.11 | under-correction of the start kick on pitch (early q 1.1 vs 2.1 rad/s in s3) | 13.9 @0.12 s | 5.9 | 2.59 / 2.49 | 3.78 (1st, its hardest), 13.7 @0.125 s |
| dres s2 MEMORY#1 | 99990123 (#22) / 32 | 5.32 | under-correction on roll (early p 0.70 vs 2.20) | 11.1 @0.27 s | 7.2 | 2.73 / 2.45 | 2.55 (11th), 6.7 @0.10 s |
| dres s2 MEMORY#1 | 99990123 (#22) / 70 | 5.97 | under-correction on roll (0.90 vs 2.69) | 12.7 @0.26 s | 8.0 | 1.90 / 1.88 | 1.92 (51st), 6.4 @0.10 s |
| dres s3 MEMORY#2 | 99990113 (#12) / 35 | 5.12 | **over**-correction on pitch (q -2.7 rad/s, overshoot to -9.7 deg) | 8.3 @0.23 s | 6.6 | 1.51 / 1.64 | 1.50 (13th), 7.1 @0.025 s |

None of the four diverged. All are soft fails, 0.11-0.97 deg over the mean bound.

### Initial conditions

Units:
- roll/pitch/yaw in deg; omega in rad/s; z0 in m; vz0 in m/s; mass in g; x0/y0 in m.
- coll = collective fraction.
- tq = D7 torque scale (roll, pitch, yaw), measured from the sim.
- wear = motor asymmetry.

| ep | roll, pitch, yaw | omega | z0, vz0 | mass, coll | x0, y0 | tq | wear |
|---|---|---|---|---|---|---|---|
| 112/86 | -2.61, +2.52, -2.3 | (-0.40, +0.42, +0.08) | +0.29, -0.07 | 44.9, +0.100 | +0.46, +0.32 | 1.04/1.19/1.03 | 1.165/0.920/0.802/0.945 |
| 123/32 | -2.82, -1.42, +4.0 | (+0.01, -0.02, +0.10) | +0.02, +0.05 | 44.5, -0.045 | +0.40, -0.48 | 0.85/1.14/1.16 | 1.062/1.151/1.161/0.802 |
| 123/70 | -3.11, -2.09, +1.2 | (-0.17, -0.34, +0.13) | -0.20, -0.10 | 43.6, +0.016 | +0.19, -0.15 | 1.15/0.82/1.20 | (same unit) |
| 113/35 | -0.45, +4.15, +1.2 | (-0.31, +0.31, -0.08) | -0.17, +0.10 | 34.2, -0.051 | +0.44, +0.37 | 0.89/0.80/0.83 | 0.886/0.947/0.941/1.083 |

Percentiles within each seed:
- **Attitude tilt** is unremarkable (33rd-54th).
- **xy start offset** is large in 3 of 4: 87th, 98th and 91st. The exception is 123/70 at the 14th.
- **112/86** is extreme on collective (99th) and z0 (94th). It is also the teacher's hardest
  episode on that unit.

## What happens: one shared mechanism with two triggers

The time course is the same in all four:

1. **Start kick (0-0.15 s).** Every controller, the teacher included, takes a shared kick of
   6-14 deg of error. The plant has 0.2 s motor lag, a worn motor, and tilt-reference error
   from the xy start.
2. **Wrong-sized correction.**
   - Three cases (s2) under-correct: commanded rate 0.70-1.10 rad/s, against 1.6-2.7 for the
     passing siblings.
   - One case (s3, 113/35) over-corrects.
   - Either way, the vehicle spends 0.15-0.3 s at 8-10 deg off-reference on one axis.
3. **Speed builds.** Lateral speed reaches 0.45-0.51 m/s. Passing siblings on the same episode
   peak at 0.12-0.30.
4. **Stall (about 0.25-1.0 s).** The tilt reference now demands about 5-6 deg of braking tilt.
   The failing genome holds a 6-8 deg error with almost no corrective command:
   - commanded rate per degree of error is 0.017-0.071 rad/s/deg in the failing genome;
   - it is 0.13-0.27 rad/s/deg in the passing siblings on the same episode.

   The error is a slow, single-axis excursion (residual SD 3-5 deg, 1-2 zero crossings), not a
   limit cycle. PWM is never saturated (0.0-0.1% of steps).
5. **Recovery and tail.** The vehicle recovers late and often overshoots, which lifts the
   1.6-2 s tail as well. About 0.7 s at 6-8 deg adds about 2.5 deg to the 2 s mean, enough to
   cross 5.0.

### Thermometer cut-points vs what these episodes reach

Thermometer cut-point spans, fit at the train seed (`_report_thresholds`, 200 calibration
episodes flown by the training cascade), compared with what these episodes reach. Measured, and
identical across the 3 genomes to within a few percent:

| feature | lowest..highest cut | reached |
|---|---|---|
| roll_err, pitch_err (vs LEVEL) | +/-0.015 rad = **+/-0.86 deg** | up to 0.24 rad (13.6 deg) |
| v_x, v_y | **+/-0.12 m/s** | 0.45-0.50 (failing genome) |
| e_x, e_y | +/-0.3 m | +/-0.5 m start (`--xy-offset 0.5`) |
| cresid_x / cresid_y (d-hat residual) | +/-0.1 | -0.6..-0.8 on unit 123, for every genome |
| coll_cmd | 0.63..0.77 | 0.615-0.816 (pinned in 3/4) |

Beyond the outermost cut, a feature's thermometer code is constant. In the stall, the failing
genome has its velocity feature pinned for 100% of the window in all 4 cases, and the attitude
error is pinned too. So "0.15 m/s and 1 deg" and "0.5 m/s and 8 deg" present the same input
bits. The response cannot scale with the error, and what the genome emits there is whatever its
memory holds at those corner addresses. In 112/86 and 123/32 the dhat s2 genome also fails;
it shares train seed 31337002 and therefore the same calibration draw.

### Population check (measured): is this the tail mechanism or just these 4 episodes?

All 3 x 3500 report episodes were replayed on CPU (stable flag = GPU flag on 10,500/10,500).
Results bucketed by peak lateral speed, pooled over the 3 genomes:

| peak lateral speed | episodes | err > 4 deg | fails |
|---|---|---|---|
| < 0.4 m/s | 10,486 | 3 | 0 |
| >= 0.4 m/s | 14 | 12 | **4 (all of them)** |

Per genome, episodes reaching 0.4 m/s are 4 (s1), 9 (s2) and 1 (s3). That ordering matches
their fail counts: 0, 3 and 1.

Caveat (inference): the high-speed regime is partly a consequence of the bad correction, so this
proves co-location, not direction. What the traces do show directly is that in that regime the
inputs are pinned and the command stops scaling with the error.

## Is the episode hard for everyone?

- **112/86** is hard for everyone. It is the teacher's worst episode on that unit (3.78 deg)
  and is in the worst 10 for 4 of the 6 rf_c200 genomes. Even so, dres s1 and s3 beat the
  teacher there (2.59 / 2.49).
- **123/32, 123/70 and 113/35** are ordinary for the teacher (rank 11, 51, 13) and for the
  sibling genomes (1.5-2.7 deg). They are genome-specific.

The teacher's margin to the bound on the four episodes is 1.2-3.5 deg. Its 0-0.25 s transient
peak is about the same as the failing WNN's. The difference is that it is back near its
reference by 0.5 s (0.5-1 s window: 1.0-2.4 deg, against 5.5-7.2 for the failing genome).
Teacher command magnitudes: not measured (no translation trace export exists).

## CPU vs GPU reproducibility (measured)

| comparison | result |
|---|---|
| Stable fraction, GPU vs CPU scorer, 18 (genome, seed) pairs (dres + dhat s1-s3 on the three seeds) | identical; mean err agrees to <= 0.003 deg |
| Stable flag, step-mirrored CPU replay vs GPU per-episode rows, 10,500 episodes | 0 mismatches |
| Per-episode mean err on the 4 failing episodes | agrees to <= 0.0006 deg |
| Per-episode mean err, worst episode anywhere | 0.30 deg (float32 GPU vs CPU) |

The 4 fails are not a scorer artifact. Note that the as-flown ABI-45 GPU scorer reported
0/3500 for s2: its translation-start indexing bug moved these episodes.

## Fix hypotheses, ranked

1. **Widen the outer thermometer range for v_xy, att-err and e_xy, so the stall region is not
   one pinned address.** This is the most direct match to the evidence: the failing behaviour
   lives exactly where these features are pinned, and the spans (+/-0.86 deg, +/-0.12 m/s) are
   far inside the reachable states. `--threshold-outer-quantile` is the existing lever; it now
   reaches every evaluator since 8542f4c1, and the oq02 s1 re-fly is already planned. Cheap
   tests:
   - Before any training, on the train seed: refit the thresholds with the outer quantile and
     count the fraction of report-episode steps pinned (this replay harness gives it in about
     1 min per genome).
   - After the oq02 re-fly: count its episodes with peak speed >= 0.4 m/s and their err, using
     the same replay.

   Note: thresholds cannot be swapped onto existing genomes, because the cells were written
   at the old addresses (threshold misalignment).
2. **Calibrate the thresholds on student-visited states, not only the teacher cascade.**
   Inference: the narrow spans are the teacher's own visitation, which almost never leaves
   +/-0.12 m/s. The student does leave it, and then reads pinned codes. Cheap test: fit
   quantiles on a mix of teacher and current-student rollouts at the train seed, and compare
   the pinned fraction on report episodes (no GA needed).
3. **Selection cannot see a 1-in-1000 tail.** The fails are genome-specific: s1 0, s2 3,
   s3 1; dhat-s2 shares the s2 failures. The 200-episode memory evaluation and stage-select
   cannot rank a 0.1% event. Cheap test: re-rank the saved final populations by
   count(peak speed >= 0.4) or per-episode p99 err on the val seeds, and see whether the
   headline would change. This is a selection-side fix and does not change the substrate.
4. **More DAgger coverage of high-speed recovery states.** For example, wider `--xy-offset` or
   initial-velocity starts in training only. The traces suggest the corner addresses are
   undertrained rather than mis-trained (the command is weak but has the right sign).
   Test: count written cells at the pinned-corner addresses of the failing genome vs its
   siblings (`cell_fill_counts` / `last_output_addresses` during the stall).
5. **Not a fix, but context for the claim.** Under the G5 reference with a mean-over-2-s bound,
   the teacher's own margin on these episodes is only 1.2-3.5 deg. A WNN tail that is 1-4 deg
   worse than the teacher on a handful of high-speed episodes is enough to cross it.

Hand-offs:
- Why the 0.2 s-lag, worn-motor start kick has this size, and whether the G5 reference
  braking gain is physically right: `flight-dynamics`.
- Whether 4/10,500 vs 0/10,500 is a meaningful difference: `experiment-design`. Note that at
  n=4 events the teacher-vs-arm gap has a wide CI.
