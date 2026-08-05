# Disturbance parameters: literature grounding (05/08/2026)

Purpose: every value in the controller's disturbance ladder must cite a paper or a
datasheet, or it is dropped. This table is the review artifact BEFORE touching
`training.py` presets (L1-L3, L2D/L3D at `training.py:170-227`). Sim dt = 0.001 s
(1 kHz), `max_torque = 0.075 * 2.4 = 0.18 N·m`.

## Sources read in full

| # | Source | What it grounds |
|---|--------|-----------------|
| S1 | Molchanov et al. 2019, arXiv:1903.04628 (8 pp) | plant DR ranges, motor lag, OU motor noise, DR-magnitude limits |
| S2 | RotorS `component_snippets.xacro` (ethz-asl/rotors_simulator, ADIS16448 defaults) — the file Molchanov's sensor model delegates to (Furrer et al.) | gyro/accel noise densities, bias walks, turn-on biases |
| S3 | Panerati et al. 2021 "Learning to Fly" (gym-pybullet-drones), arXiv:2103.02142 (8 pp) | aero effects (drag/ground effect/downwash) from system ID; what a canonical sim does NOT model |
| S4 | Dryden MIL-F-8785C via MathWorks Aerospace Blockset doc | wind severity ladder, σ_w = 0.1·W20; light/mod/severe = 15/30/45 kt at 20 ft |
| S5 | arXiv:2603.02114 (thermal-inertial odometry) | the only quantified sensor-outage figure found: thermal NUC dropout "250 ms to 1 s" |

## The headline survey result

**No surveyed simulator or DR paper models observation delay, sensor dropout, or a
constant torque bias.** S1 models motor lag instead of obs delay (a physical
actuator property, T = 0.15 s settling — ~100x slower than our 4 ms obs delay) and
found modeling it "has a very small impact" on transfer. S3's observations are
ground-truth kinematics — no sensor pathology at all; its realism budget goes to
aerodynamics (drag, ground effect, downwash), each fitted to real Crazyflie data.
S2 is datasheet-scale noise. Our D5/D6 block and tau_bias have no literature
counterpart.

## Parameter-by-parameter verdict

| Ours (L2D / L3D) | Literature | Ratio | Verdict |
|---|---|---|---|
| `gyro_sigma` 0.030 / 0.080 rad/s per 1 kHz step → implied density ~9.5e-4 / 2.5e-3 rad/s/√Hz | S2: `gyroscope_noise_density = 0.0003394 rad/s/√Hz` (ADIS16448) | **~2.8x / ~7.5x** datasheet | RESCALE toward S2. NOTE: density→per-step conversion (σ=N·√f_s) is OUR derivation; and vibration on a real frame raises the effective figure — no source for the multiplier yet, so any headroom above datasheet must be declared as assumption. CODE-CHECK first: confirm the sim applies gyro_sigma as white per-step noise at 1 kHz. |
| `gyro_bias_walk` 0.003 / 0.008 | S2: `gyroscope_random_walk = 3.8785e-5 rad/s/s/√Hz`; `turn_on_bias_sigma = 0.0087 rad/s` | unit check needed | CODE-CHECK our units, then match S2's two-component model (turn-on bias + walk). Our single number conflates them. |
| `accel_sigma` 0.30 / 0.80 m/s² | S2: `accelerometer_noise_density = 0.004 m/s²/√Hz` → ~0.13 m/s² at 1 kHz | ~2.3x / ~6.2x | RESCALE toward S2, same caveats as gyro. |
| `gust_sigma` 0.121 / 0.241 (OU, slaved to tau_bias by `σ = bias/√(τ_c/2)`) | S4: Dryden σ_w = 0.1·W20; severity = wind speed 15/30/45 kt at 20 ft | not comparable (ours is torque-units, slaved) | REPLACE: decouple from tau_bias; adopt Dryden severity as the WEATHER axis (light/moderate/severe). Needs a torque-coupling model (wind→torque via drag geometry) — S3's drag model (D = -k_D·Σω_prop·ẋ, system-identified) is the sourced bridge. |
| `tau_bias` 0.027 / 0.054 N·m (15% / 30% of authority, CONSTANT) | none — S1 covers trim implicitly via t2w U(1.8,2.5), t2t U(0.005,0.02) (~±16-20% plant randomization) | unsourced | DROP as a separate knob. Trim error folds into per-episode plant randomization at S1's magnitudes. (This also removes the lever that made LQI>LQR — that finding stays true OF THE OLD MODEL only.) |
| `torque_scale_jitter` ±15% / ±25% per episode | S1 Table IV: 10-20% randomization helps, **30% hurts**; "works best if fairly small (20%)" | L3D sits in the measured-harmful band | CAP at 20% (S1). Keep per-episode draw (matches S1's per-trajectory sampling). |
| `motor_asym` ±10% / ±15% per motor | S1 randomizes whole-vehicle t2w/t2t, not per-motor; per-motor asymmetry unsourced directly | partial | KEEP magnitude ≤ S1's ~±16% envelope, declare per-motor as our extension (it is the fault-tolerance literature's territory). |
| motor lag — ABSENT from our sim | S1: first-order low-pass, T = 0.15 s settling, U(0.1, 0.2); QuadSwarm: same structure | we lack a sourced effect | ADD (sourced, and cheap). This is the literature's replacement for our obs_delay. |
| `obs_delay_steps` 2 / 4 ms | none — S1's real stack runs EKF+control at 500 Hz but does NOT model delay in training; no surveyed sim models it | unsourced | DROP from the ladder (motor lag supersedes it as the sourced latency mechanism). |
| `dropout` (TOTAL obs freeze) 3.8% / 16.7% duty, 20/40 ms | S5: thermal NUC outage 250 ms-1 s — but (a) thermal-specific, (b) a VISION outage does not freeze the IMU: gyro keeps reporting. For an ATTITUDE controller on gyro-derived features, camera dropout does not freeze the observation vector. | unsourced as modeled | DROP from attitude-controller experiments. If we ever do position/velocity control from VIO, reintroduce as a POSITION-CHANNEL outage of 250 ms-1 s citing S5 — not a full-state freeze. |

## Consequences to acknowledge

1. **The freeze was the strongest argument for recurrent state** (frozen obs vs
   holding still are indistinguishable without memory). Dropping D5 removes that
   argument. The surviving sourced case for state is yaw dead-reckoning
   (student-centric conflict measurement, 12.8% under L2) — the state question
   must be re-justified on the new ladder, not assumed.
2. **Nothing measured on L2D/L3D is submission-grade** (already recorded in the
   venue section of the roadmap). The A-probe relative findings (pidmix > tilt,
   decode topology >> alphabet) are expected to transfer but must be spot-checked.
3. **LQI-vs-LQR teacher ranking was driven by tau_bias** — with tau_bias dropped,
   re-screen teachers on the new ladder before assuming lqi.
4. Proposed axes for the new presets (design TBD with Luiz): HARDWARE
   (S2-datasheet sensors + S1 plant DR at ≤20%) x WEATHER (Dryden
   light/moderate/severe) — the "quality airframe in serious wind" cell the old
   parameterization could not express.

## Open items before presets are written

- [ ] CODE-CHECK: how the sim applies gyro/accel sigma (per-step white at 1 kHz?)
      and the units of gyro_bias_walk — determines the rescale arithmetic.
- [ ] Wind→torque coupling: adopt S3's identified drag model or derive from our
      airframe geometry; document either way.
- [ ] Vibration multiplier above datasheet noise: find a source or declare as a
      stated assumption with a sensitivity check.
- [ ] Re-screen teachers (pid/lqr/lqi/mpc/mpcof) on the new ladder.
