# CTRL-44 — cascade d̂ input: computed right, encoded wrong (08/11/2026)

Diagnosis by the flight-dynamics agent (no source changes; probes in the job scratch dir, ABI 43 wheel, CPU).
Recipe: posx2 cascade, stage-2 plant, motor lag 0.20, L4C (±20 % per-motor asymmetry + torque-scale jitter;
`gust_sigma = 0`, so no gusts), `--obs-cascade-dhat`. Axes are BODY axes (roll x, pitch y, yaw z), rad/s².

## Measured

**d̂ is exact.** Against the true per-axis asymmetry acceleration (plant torque law on the lagged command),
35 units × 6 episodes, t > 300 ms: slope 1.000 / corr 1.0000 / sign 100 % on all axes; RMS error 0.09 (roll,
pitch) and 0.014 (yaw) vs truth SD 36.9 / 32.1 / 2.2; within 10 % of truth after ~60 ms (roll/pitch) and ~30 ms
(yaw). Teacher and student run the same `cascade_dhat.rs` on the same (gyro, applied command), so they are
identical by construction. Clamp 235/235/43 never binds. Reset per episode, action-repeat ticks, DAgger record
and replay, and the Metal f32 twin all check out.

**The thermometer is the defect.** Thresholds are quantile-fitted on 10 calibration episodes
(`evaluator.py` `_cascade_columns_from_training_cascade`, `num_episodes=10`). d̂ is near-constant within an
episode, so the 8 thresholds land on those 10 airframes' plateaus. The ladder is a fresh random draw per seed:

| | s2 thresholds | s3 thresholds |
|---|---|---|
| roll | −47.7, −37.9, −5.6, −4.2, 2.1, 8.8, 12.7, 17.8 | −72.3, −18.3, −15.8, 1.4, 27.1, 30.8, 46.5, 63.7 |
| pitch | −31.2, −30.3, −3.4, −0.2, 7.0, 20.8, 32.7, 44.6 | −48.1, −42.4, −22.4, −9.9, −7.0, −3.3, 12.7, 25.8 |
| yaw | −2.71, −0.39, 0.08, 0.67, 1.00, 1.27, 1.58, 1.82 | −1.31, −1.00, 0.77, 1.35, 1.70, 2.70, 3.44, 3.86 |

- In flight one bin holds the whole episode (median dominant-code share 1.000 roll/pitch, 0.88–1.00 yaw).
- Saturated codes take 46 % of flight time (s2 roll) and 50 % (s3 yaw).
- s3 yaw: 17 of 35 units sit in code 0, so −1.3 … −5.4 rad/s² look identical.
- The failing units (yaw truth −3.3 … −4.9) all sit in yaw code 0 on both seeds.
- Failures correlate with |yaw truth| (steady tilt r 0.57–0.60; |yaw rate| in the last second r 0.81), not pitch.

**Integrator authority** (`ki × i_limit × c`): roll 95.7, pitch 95.2, **yaw 3.65** rad/s². The rate
integrator absorbs roll/pitch disturbance within ~1–2 s, so the unabsorbed fraction (deficit) drops from 0.69 to
0.00. Yaw (τ = kp/ki ≈ 7.2 s) keeps 0.75–0.99 unabsorbed all episode. The teacher's plan acts only on the deficit
(`ctbr_axis_mpc.rs:152`: `sp_ss = 0`, `I_ss = −d/(c·ki)`).

**Connections:** trained genomes wire to the d̂ bits at the uniform rate (≈246 per feature; 164–175 of 256 neurons
read ≥1 d̂ bit). There's no selection either way, and the connection stage was skipped.

## Inferred

- The steady penalty comes from roll/pitch d̂: it stays constant through the hold, where the right correction is
  0, and it splits each address's training data by airframe class.
- Yaw is where d̂ pays, and the random ladder throws away exactly the |d| > 2 range up to the 3.65 authority edge.
- The ±3 pp seed swing fits a fresh random yaw ladder per seed.

## Ranked fixes

1. Fit the d̂ ladder on the airframe DISTRIBUTION, not 10 draws: closed form (hover thrust × U(±0.2) mixed
   through c gives roll/pitch triangular on about ±91–104 and yaw roughly normal with SD 2.4 on ±8.3), or ≥200
   calibration episodes using per-episode post-300 ms means. Seed-invariant. **The same applies to rate-I**,
   which is also near-constant per airframe and fitted on the same 10 episodes.
2. Feed the UNABSORBED disturbance `d̂ + c·ki·I` (≡ `I − I_ss(d̂)`, normalised by `c·ki·i_limit`). This decays to
   0 as the rate loop absorbs d, which removes the hold bias.
3. Cheap A/B: yaw-only d̂.
4. A centred symmetric yaw ladder of about ±6, finest near ±3.65.
5. Hold the feature at its neutral code for the first ~100 ms.
6. Leave the observer gain and clamp alone.
