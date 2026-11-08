# CTRL-44b — thermometer encoding audit, all features (08/11/2026)

Flight-dynamics agent, read-only probes on ABI 43:
- production calibration sampler, 200 eps at 5° and 100 eps at 30°, s2/s3/s4;
- 9 winners flown over the 35 CF2.1 units;
- `WNN_IMIT_PROBE` ridge R² of the MPCOF label on held-out teacher states, 51 runs.
Scratch probes: job tmp p5–p10.

## Findings (measured)

**1. Too few independent draws.** Calibration is `fit_thresholds_from_pid_rollouts` → Rust
`sample_calibration_features`: 10 episodes × 2000 steps, firmware PID pilot, 5° tilt, quantiles 1/9…8/9 on pooled
steps. `num_episodes=10` is hard-coded at phased_ga.py:891/1162/1286/1736 and controller_grid_search.py:117.

Features dominated by between-episode variance get about 10 independent values. Their thresholds miss their nominal
quantile by about one bin and move 0.25–0.6 IQR per seed:

| feature | between-episode share |
|---|---|
| yaw_err, yaw_err_i | 0.975 |
| coll_cmd | 1.00 |
| alt_err | 0.86 |
| vz | 0.80 |
| e_x, e_y | 0.96 |
| rate-I | 0.87–0.98 |
| d̂ | 0.99–1.00 |
| roll/pitch_i at leak 0.999 | 0.50 |

200 episodes cuts the seed-to-seed threshold movement 3–10×. Episodes 0–9 of the 200-episode stream equal today's fit.

**2. Wrong regime.** The ladder is fitted at the 5° scorer tilt, but DAgger trains at 8–30°. On 30° teacher states
the production ladder saturates 27–59% of samples, and 72–96% of gyro / roll-pitch error / integral samples in
the first 300 ms. Example: the roll_err ladder spans ±0.9° while training states reach ±10°.

**3. Dead features (bug).** tgt_r/p/y are always 0, so every threshold is 0.0 and the code never changes. That is
24 of 200 frame bits, 12% of every random 24-bit draw.

**4. Noise features.** Gyro gaps near 0 (~0.006 rad/s) are below the gyro σ (0.0107). Every acc_x/acc_y gap is below
the accel σ (0.126); with the specific-force IMU these features carry only about 0.035 m/s² of drag. Both cycle
through all codes during the hold.

**5. Per-feature thermometer coding loses about half the roll/pitch label information.** Ridge R² of the MPCOF
label on held-out 30° teacher states, mean of s2–s4:

| ladder | bits/feature | p | q | r | T |
|---|---|---|---|---|---|
| production | 8 | 0.340 | 0.422 | 0.694 | 0.934 |
| 200-episode quantile | 8 | 0.395 | 0.446 | 0.765 | 0.944 |
| centred, fine near 0 | 8 | 0.405 | 0.464 | 0.765 | 0.944 |
| 30°-fitted | 8 | 0.440 | 0.515 | 0.779 | 0.944 |
| 30°-fitted | 16 | 0.534 | 0.608 | 0.785 | 0.969 |
| per-episode features blanked | 8 | 0.234 | 0.300 | −0.142 | 0.665 |
| raw features, linear | — | 0.917 | 0.929 | 0.806 | 1.000 |

- Ladder placement gains +5 to +10 pp on p/q and +6 to +9 pp on r. Training regime plus 16 bits gains up to +19 pp.
- p/q stay about 0.32–0.38 below raw even then.
- The per-episode-constant features carry real label information (blanking them costs yaw rate −84 pp), so they are
  not merely overfitting airframe IDs.
- Caveat: an additive regression is only an information proxy for the RAM, which can capture interactions within
  a neuron's 24 bits.
- The 30°-fitted ladder coarsens the 5° hold the scorer measures, so it needs a closed-loop A/B before adoption.

## Recommendation

**Generic fix: one `ThresholdCalibration` typed config on args.** It goes into `fit_thresholds_from_pid_rollouts`
and `_cascade_columns_from_training_cascade` and replaces the five hard-coded `num_episodes=10`.

| field | default | recommended / options |
|---|---|---|
| `episodes` | 10 | 200 |
| `regime` | `flown` | `train_mix`: the flown tilt pooled with the curriculum tilts 8–30° |
| `ladder` | `quantile` | `centred`: symmetric ±Q_\|x\|(p), p on a ^1.5 curve, finer near 0; for zero-referenced features only |
| `bits` | 8 | separate axis, leave at 8 for now |

- Defaults reproduce today's thresholds exactly.
- Any non-default value is a lineage break and needs a paired A/B.
- Add all four fields to provenance.

**Targeted fixes (separate changes):**
1. Constant-feature guard: drop or skip zero-spread features (tgt_r/p/y), which frees 12% of each neuron's bits.
2. acc_x/acc_y drop A/B (noise under the specific-force IMU).
3. d̂: fit on the airframe distribution, or feed the unabsorbed residual (see ctrl44_dhat_investigation.md).
4. The remaining roll/pitch gap is a representation question (joint/derived features vs more bits per neuron).
   It needs a design discussion; don't just build it.
