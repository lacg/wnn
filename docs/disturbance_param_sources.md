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

## CODE-CHECK RESOLVED (05/08/2026) — controller.rs

Read `controller.rs:410-414` and `:600-610`. The sim applies the four stochastic
terms with TWO different conventions:

```rust
g[a]  += d.gyro_sigma  * dist_gauss(...)            // :411  NO dt scaling
a2[a] += d.accel_sigma * dist_gauss(...)            // :414  NO dt scaling
self.gust[a]      += -gust/tau_c*dt + d.gust_sigma      * sqrt_dt * xi;  // :602 OU
self.gyro_bias[a] +=                  d.gyro_bias_walk  * sqrt_dt * xi;  // :608 Brownian
```

So **`gyro_sigma`/`accel_sigma` ARE the per-sample standard deviations at 1 kHz**
(white, unscaled), while gust and bias-walk carry proper `sqrt_dt` continuous-time
scaling. That settles the rescale arithmetic:

| | ours L2D / L3D (per-sample) | RotorS ADIS16448 density at 1 kHz | ratio |
|---|---|---|---|
| gyro | 0.030 / 0.080 rad/s | N=3.394e-4 → ~0.0076-0.0107 rad/s * | **~3-4x / ~7-10x** |
| accel | 0.30 / 0.80 m/s² | N=0.004 → ~0.089-0.126 m/s² * | **~2.4-3.4x / ~6.3-9x** |

\* OUR DERIVATION, and the residual ambiguity is the discretization convention:
`sigma = N*sqrt(f_s/2)` (bandwidth) vs `sigma = N/sqrt(dt) = N*sqrt(f_s)` (the form
Gazebo/RotorS plugins typically use) differ by sqrt(2). **TODO: read
`gazebo_imu_plugin.cpp` to fix the convention before writing numbers.** Ranges above
bracket both.

`gyro_bias_walk` (0.003/0.008, sqrt_dt-scaled) is structurally the SAME quantity as
S2's `gyroscope_random_walk = 3.8785e-5 rad/s/s/sqrt(Hz)` — directly comparable once
the convention is fixed. But our model has NO turn-on bias; S2 has
`gyroscope_turn_on_bias_sigma = 0.0087 rad/s` (a per-power-cycle constant offset).
Adding it is cheap, sourced, and is the physically-honest home for the trim error we
are removing from `tau_bias`.

## MEASURED: plain-L3 classical baselines (05/08/2026)

`experiments/dfa1l_markers/baselines_L3.json` — 5 held-out seeds, tilt 5.0, 100 ep x
2000 steps, sim_seed 911, fold 0. Generated because the WNN's plain-L3 result was
uninterpretable without it (only L2D and L3D baselines existed):

```
ctrl          stable%          err deg     steady deg
PID        4.0+- 5.6        8.46+-1.29    9.06+-1.35
LQR       83.6+-12.8        3.91+-1.13    4.03+-1.19
MPC       81.8+-13.9        4.06+-1.14    4.22+-1.20
LQI       89.4+- 8.3        3.33+-0.97    2.96+-0.84
MPCOF     91.6+- 7.3        2.71+-1.19    2.49+-1.26
--- WNN (P3_control_l3, sn=0, teacher=lqi, n=1 seed) ---
WNN       72.8+- 9.4        4.19+-0.45    4.10+-0.58
```

Reading: the WNN sits **~11pp below LQR/MPC on stability but INSIDE their error band**
(4.19 deg vs 3.91/4.06; its steady 4.10 deg actually beats MPC's 4.22). It trails its
own teacher LQI by **16.6pp** — that is the imitation gap at L3, versus a 66pp gap at
L3D (4.6 vs 70.6). The D-fields were destroying imitation, not the ladder magnitudes.
PID collapses at L3 (4.0%), so the WNN beats it by ~69pp.

NOTE the teacher ranking INVERTS between rungs: MPCOF is BEST at L3 (91.6%) and WORST
at L3D (47.8%); LQI is best at L3D and 2nd at L3. Any teacher choice must be
re-screened on the ladder actually being used — see open items.

## VIBRATION-MULTIPLIER HUNT — NEGATIVE RESULT (05/08/2026)

Luiz's instruction: find a source or do not invent it. **No citable in-flight
vibration multiplier was found.** Reporting the negative rather than a number.

What was checked:
- **DIDO (Zhang et al., RA-L 2022, arXiv:2203.03149) — READ IN FULL (8 pp).** A web
  search summary attributed "propeller-induced noise exceeding 5 m/s^2" to this
  paper. **That claim is NOT in the paper.** Do not cite it. What DIDO actually does
  (Eq. 1-2): models IMU noise as plain additive Gaussian white + random-walk bias,
  `n_w ~ N(0, Sigma_w^2)`, `b_dot_w ~ N(0, Sigma_bw^2)` — **no vibration term, no
  inflated sigma**. Platform: Xsens MTi-300. Its stated remedy for prop noise is
  filtering, not sigma inflation: "In practice, we low-pass filter the omega-tilde,
  a-hat to reduce noise" (Sec. III-B).
- Tangram Vision IMU-modeling series: states qualitatively that an operating
  quadrotor's noise floor exceeds bench Allan-variance figures (which are taken on a
  vibration-isolated table). **Directional only — no factor given.**
- INSANE UAV dataset (arXiv:2210.09114): PDF exceeds the fetch size limit; not read.
  Remains the best unexplored candidate for measured in-flight IMU noise.

**Conclusion: the literature does not model prop vibration as an inflated sensor
sigma at all.** It uses datasheet-grade noise and handles vibration downstream
(low-pass filtering, learned de-biasing). Inflating sigma would be OUR invention and
is therefore rejected.

**Replacement for the "vibration multiplier" idea — a SOURCED hardware axis.** Span
the hardware quality axis by IMU GRADE instead, both endpoints from datasheets:

| grade | gyro noise density | source | ratio |
|---|---|---|---|
| research | 3.394e-4 rad/s/sqrt(Hz) | ADIS16448 via RotorS S2 | 1.0x |
| consumer | 1.3e-3 rad/s/sqrt(Hz) | MPU-9250 (VERIFY against datasheet before use) | ~3.8x |

That is a ~3.8x span meaning "which IMU you bought" — citable at both ends, and it
answers the original question (can better hardware fix this?) with a real axis
rather than an invented coefficient.

## S6 — MPU-9250 DATASHEET, READ (05/08/2026). It INVERTS the assumption.

`PS-MPU-9250A-01 rev 1.1`, Tables 1 & 2, read directly. Verbatim:

| parameter | value as printed | SI (ours) |
|---|---|---|
| Gyro Rate Noise Spectral Density | `0.01 deg/s/sqrt(Hz)` | **1.745e-4 rad/s/sqrt(Hz)** |
| Gyro Total RMS Noise (DLPFCFG=2, 92 Hz) | `0.1 deg/s-rms` | **1.745e-3 rad/s rms** |
| Accel Noise Power Spectral Density (low-noise) | `300 ug/sqrt(Hz)` | **2.942e-3 m/s^2/sqrt(Hz)** |
| Accel Total RMS Noise (DLPFCFG=2, 94 Hz) | `8 mg-rms` | **7.85e-2 m/s^2 rms** |

**Two corrections this forces:**

1. The 1.3e-3 rad/s/sqrt(Hz) figure previously attributed to the MPU-9250 is WRONG —
   it is **7.5x the datasheet**. It traced to a SIMULATION MODEL in a third-party
   paper, not to InvenSense. Do not cite it. (Caught only by reading the datasheet;
   the search summary presented it as a spec.)
2. **The "IMU grade" axis DOES NOT EXIST as proposed.** The hobby-grade MPU-9250 is
   *quieter on paper* than the research-grade ADIS16448:

   | | gyro density | accel density |
   |---|---|---|
   | ADIS16448 (S2) | 3.394e-4 | 4.0e-3 |
   | MPU-9250 (S6) | **1.745e-4** | **2.942e-3** |
   | ratio | 0.51x | 0.74x |

   Modern MEMS IMUs sit within ~2x of each other and NOT in the assumed direction.
   There is no sourced sensor-noise span to build a hardware-quality axis from.
   **Axis withdrawn.**

**The cleanest sourced comparison — filtered RMS.** Real flight controllers low-pass
the gyro (DIDO does exactly this). At the datasheet's own 92 Hz DLPF setting the
MPU-9250 reads `0.1 deg/s-rms` = 1.745e-3 rad/s. Our ladder:

| | per-sample gyro sigma | vs MPU-9250 filtered RMS |
|---|---|---|
| ours L2D | 0.030 rad/s (1.72 deg/s) | **17x** |
| ours L3D | 0.080 rad/s (4.58 deg/s) | **46x** |

Sensor noise was never the physical story it looked like.

**INSANE (arXiv:2210.09114): CLOSED, unread.** PDF exceeds the WebFetch size limit;
`arxiv.org/html/2210.09114v2` returns 404; the abs page confirms only "a dedicated
high-rate IMU captures vibration dynamics" with no numbers. Not pursued further — S6
plus DIDO's filter-don't-inflate finding already settle the question.

## UNITS: SI EVERYWHERE (Luiz, 05/08/2026 — hard rule)

All parameters, presets, code and paper tables use SI. Sources may publish in
imperial/knots; convert AT INGEST and record the conversion. Notably S4 (Dryden,
MIL-F-8785C) is imperial:

| S4 as published | SI (what we store) |
|---|---|
| W20 measured at 20 ft | 6.096 m |
| light 15 kt | 7.72 m/s |
| moderate 30 kt | 15.43 m/s |
| severe 45 kt | 23.15 m/s |

`sigma_w = 0.1 * W20` is dimensionless-scaled, so it carries over unchanged once W20
is in m/s. No knots, no feet, no degrees-per-second anywhere in the ladder: rad/s,
m/s^2, N.m, m/s, s.

## WIND->TORQUE COUPLING — BLOCKED, and the blocker is deeper than wind (05/08/2026)

Attempted the derivation Luiz approved ("Dryden for the field, derive the coupling").
It cannot be done honestly, for a reason worth recording.

**1. Our sim cannot host a drag-based coupling.** `AttitudeSim::new` defaults
(controller.rs:620-626) are the WHOLE plant:

```
dt 0.001 | arm_length 0.075 m | k_thrust 2.4 N/pwm^2 | k_drag 0.05
inertia [0.0023, 0.0023, 0.0046] kg.m^2 | gravity 9.81
```

There is **no mass, no body geometry, no translational state**. It integrates
`tau = I * omega_dot` and nothing else. So the S3 (gym-pybullet-drones) drag model
`D = -k_D * (sum 2*pi*P_i/60) * x_dot` is inapplicable — there is no `x_dot`. Wind can
only enter as a torque, which is what `gust_sigma` already is. The open question was
never the mechanism, only the MAGNITUDE. (`k_drag = 0.05` is the rotor-spin yaw-torque
coefficient, not body aerodynamic drag — not usable here.)

**2. No usable published anchor found.** Two candidates read:
- Barcelos, Haleem & Bramesfeld, CASI AERO 21, "Experimental study of the aerodynamic
  loads on the airframe of a multirotor UAV" — READ (6 pp). Wind-tunnel loads on a
  **DJI Matrice 210 RTK** airframe (17-inch rotors, ~4.8 kg class) **with the rotors
  removed**, results presented **figure-only** (no tables). Wrong scale by ~20x in
  mass and the wrong mechanism (bare airframe, no rotor H-force). NOT usable.
- Otsuka, Sasaki & Nagatani 2018 (head-up pitching moment, small quad): SAGE returns
  HTTP 403. Unread. Still the best remaining candidate if we get access.

**3. THE REAL BLOCKER: our airframe is not a real airframe.** To couple a wind speed
to a torque we need drag area, centre-of-pressure offset, or rotor-plane height — all
properties of a PHYSICAL vehicle. Ours is a synthetic parameter set: implied mass
~0.245 kg (from `4 * k_thrust * 0.5^2 = 2.4 N` at the codebase's hover PWM 0.5, over
g), 0.075 m arm, so roughly a 150 mm-class quad — plausible, but matching no published
vehicle. There is nothing to look the coefficients UP for.

So the plant itself has the same provenance problem the disturbances had. Chasing the
wind coupling alone would fix the smaller half.

### Options (decision needed, not assumed)

**A. Adopt a published airframe — Crazyflie 2.x.** S1 (Molchanov), S3
(gym-pybullet-drones, whose default model it IS) and QuadSwarm all use it; mass,
inertia, arm length and thrust constants are published and system-identified, and S3
additionally publishes an **experimentally identified drag model** for that exact
vehicle. This makes the plant citable, unblocks the coupling, and maximises
comparability — the priority Luiz set. COST: every controller number re-runs on a new
plant; L4's sensor/plant rungs survive unchanged.

**B. Ship L4 without a weather axis.** Defensible as-is: L4 tests sensor noise +
plant uncertainty, and many attitude-stabilisation papers model no wind at all. State
plainly in the paper that wind is out of scope. COST: loses the "quality airframe in
serious wind" cell that motivated the split, and a reviewer may ask why.

**C. Get Otsuka 2018** and, if it reports moments in SI for a small quad, scale to our
airframe with a stated assumption. Weakest: cross-airframe scaling of an aerodynamic
moment is itself an invention unless the vehicles match.

Recommendation: **A**, because it fixes the plant's provenance and the coupling with
one decision, and B remains available as the interim (it is what L4 ships today).

## Open items before presets are written
- [ ] Wind→torque coupling: adopt S3's identified drag model or derive from our
      airframe geometry; document either way.
- [x] Vibration multiplier: HUNTED, no source exists — REJECTED as an invention.
      Replaced by a datasheet-grade IMU axis (see negative-result section).
- [x] MPU-9250 VERIFIED against the datasheet (S6) — the search figure was 7.5x too
      high, and the IMU-grade axis is WITHDRAWN (hobby IMU is quieter than research
      IMU on paper).
- [x] INSANE: closed unread (PDF over fetch limit, HTML 404). Not needed — S6 + DIDO
      settle it.
- [ ] Re-screen teachers (pid/lqr/lqi/mpc/mpcof) on the new ladder.
