# CTRL: INDI teacher for the cascade controller — DESIGN SPEC (14/11/2026)

Status: **DESIGN ONLY.** No source changed, no wheel built, nothing launched. Start of design approved by Luiz 14/11/2026; implementation waits for his decisions in section 9.
Author: flight-dynamics agent. Code references are to `src/wnn/ram/strategies/accelerator/controller/` unless stated. Vocabulary: val = tuning, test = report.

## 0. Bottom line

1. **Which loop.** INDI replaces the **body-rate loop only** (today `pid_firmware::RateStageFw`, "pidfw_rate@500Hz"). The attitude → rate-setpoint stage above it stays. INDI is an inner-loop law; it has no attitude-level content of its own (the firmware's INDI outer stage is a plain P gain).
2. **On this plant INDI is not a new mechanism. It is the disturbance observer MPCOF already has, moved into the rate loop.** `cascade_dhat.rs` computes d̂ = LPF(ω' − c·ua): gyro-differentiated angular acceleration minus the modelled lagged rotor moment. That is exactly INDI's inner quantity. INDI's command is F = (ν − d̂_f)/c with a second-order filter. Derivation in section 2.3. The time scales are also close: MPCOF's observer has τ = 50 ms; firmware INDI's filter has 28 ms group delay in front of the same 50 ms rotor lag.
3. **Observability: yes.** Angular acceleration is recoverable from the L4C gyro well enough (0.07 rad/s² rms at the firmware's 8 Hz cutoff, against 119-1307 rad/s² per unit of axis command). The actuator state is not measured but is modelled exactly in τ and to ±20% in gain, which is inside INDI's tolerance. Section 3.
4. **What INDI cannot touch.** 0.52° of the MPCOF rival's 0.68° steady is Mahony specific-force bias (`docs/ctrl_steady_gap_teacher.md` §2). That is an attitude-reference error in the outer loop. INDI cancels moments; it cannot see or remove it. The addressable part is the wear residual: **0.16° (oracle-fed) / 0.17° (est-fed)**. Best case for an INDI teacher is therefore about **0.51° est-fed steady**, not 0.
5. **The premise needs one correction.** The yaw offset in `docs/ctrl_steady_gap_block6.md` is the STUDENT's (0.48° vs the teacher's 0.08°), caused by the student's yaw ladder and missing integrator, and block 8 addressed it on the student side. The teacher's yaw is already 0.08°. An INDI teacher does not target that term.
6. **Recommended shape.** Keep the student exactly as flown (same 4 CTBR banks, same observation, same Metal kernel, same firmware rate PID inner loop). The INDI teacher's moment command is expressed as the rate setpoint that makes the student's own inner loop produce it ("label transport", section 2.5). Stage 0 measures the ceiling of that with a CPU-only rival row before any training.
7. **Acceptance bar (proposed, Luiz to fix):** on the 35 test units, the transported-label row must reach 100% stable / 0 fails, steady ≤ 0.60° est-fed (MPCOF 0.68°) and ≤ 0.10° oracle-fed (MPCOF 0.16°), err ≤ 1.35°, pos ≤ 0.346 m. Section 6.
8. **Effort.** Recommended path about 6-8 working days of implementation plus one paired 3-seed block. The alternative (INDI as the flown inner loop) is about 14-18 days and starts a new lineage. Section 8.

## 1. Where we are (as implemented)

- **Flown architecture** (`[provenance]` of `logs/controller/ctrl48/C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_both_full_s3.out`): `arch=cascade:teacher=mpcof:inner=pidfw_rate@500Hz:outer_hz=100`.
  - Position loop (`position_loop.rs`) → tilt reference. Classical, disclosed.
  - OUTER (100 Hz, action_repeat 10): the WNN emits 4 CTBR banks [collective deviation (pwm), p_sp, q_sp, r_sp (rad/s, BODY frame)] (`ctbr.rs`).
  - INNER (500 Hz): firmware rate PID on the measured gyro, sourced cf21bl gains (`pid_firmware.rs::RateStageFw`), mixed to 4 pwm through the square-law mixer.
- **Teacher** (`ctbr_mpcof.rs`, `ctbr_axis_mpc.rs`): per-axis condensed QP on the closed inner loop (8 states), horizon 20, offset-free through `CascadeDhat` (gain 0.02 per 1 kHz tick). It decides a body-rate setpoint every 10 ticks.
- **Labels.** DAgger labels are built from the TRUE quaternion (`label=true` in provenance). The est-fed "MPCOF cascade" row is the RIVAL, not the label source. The label source is the oracle-fed law, whose steady is 0.163° (teacher doc §2). The student's distance to its label source is set by what the student can observe, not by the rival's 0.68°.
- **Motor lag.** `motor_lag=0.2s:explicit` is Molchanov's 2% SETTLING time T. The plant filter is first order with coefficient 4·dt/T per 1 kHz tick (`controller.rs::apply_motor_lag`, `actuator_lag.rs::motor_lag_alpha`), i.e. **τ = T/4 = 50 ms** (Gräfe et al. 2026 measured T = 50 ms on the real brushless; `docs/papers/grafe2026_model_crazyflie_brushless.md`). It filters pwm, then thrust = k_thrust·pwm². It is the same for every unit; wear does not change τ.
- **Disturbance level L4C** (`wnn/control/training.py`): motor asymmetry ±20% per motor (per unit on test, per episode in training), torque-scale jitter ±20% per axis, gyro white noise σ = 0.01073 rad/s per 1 kHz sample, gyro bias random walk 3.88e-5, accel σ = 0.126 m/s², rotor drag 0.78. **No gust (gust_sigma = 0), no constant torque bias, no dropout, no delay.**
- **Reference numbers** (test, 35 CF2.1 units × 100 episodes, `docs/ctrl48_fixed_scorer_table.md`, `docs/ctrl48_block8_result.md`):

| row | stable % | err ° | steady ° | pos m | fails |
|---|---|---|---|---|---|
| MPCOF cascade, est-fed (rival C4) | 100.0 | 1.35 | 0.68 | 0.341 | 0 |
| PID, est-fed | 91.1 | 3.33 | 2.76 | 0.352 | 310 |
| WNN "both" (mean of s1-s3) | 100.0 | 1.37 | 0.77 | 0.36 | 2 / 10500 |

## 2. The control law

### 2.1 Sources
- Smeur, Chu, de Croon (2016), "Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of Micro Air Vehicles", J. Guidance, Control, and Dynamics 39(3). The law, the synchronisation-filter argument, the LMS adaptation of G.
- Smeur, de Croon, Chu (2018), "Cascaded incremental nonlinear dynamic inversion for MAV disturbance rejection", Control Engineering Practice 73. The outer (linear-acceleration) INDI loop. Out of scope here; noted for later.
- Tal, Karaman (2021), "Accurate Tracking of Aggressive Quadrotor Trajectories Using INDI and Differential Flatness", IEEE TCST. Motor-lag handling. Cited for the fallback only.
- **bitcraze/crazyflie-firmware `controller_indi.c` / `controller_indi.h` @master** (authors Smeur, Paraense). Read on 14/11/2026. This is the sourced structure and the sourced gains, in the same spirit as the firmware PID. Verbatim:
  - `rateDesired.roll = err_p*(attitudeDesired.roll - radians(state->attitude.roll))`
  - `attitude_error_p = rateDesired.roll - body_rates.p; angular_accel_ref.p = rate_p * attitude_error_p`
  - `rate_d[i] = (filter[i].o[0] - filter[i].o[1]) * ATTITUDE_RATE` on a 2nd-order Butterworth of the gyro
  - `du.p = 1/g1.p * (angular_accel_ref.p - rate_d[0]); u_in.p = u[0].o[0] + du.p` where `u[]` is the SAME Butterworth applied to the actuator-model output
  - `u_act_dyn.p += act_dyn.p * (u_in.p - u_act_dyn.p)`
  - Defaults: `REF_ERR_P/Q/R 5.0`, `REF_RATE_P/Q/R 24.0`, `FILT_CUTOFF 8.0` Hz (yaw the same), `ACT_DYN 0.03149`, `G1_P 0.0066146`, `G1_Q 0.0052125`, `G1_R 0.001497`, `G2_R 0.000043475`. Runs at `ATTITUDE_RATE` (500 Hz).
  - Not yet checked: whether `platform_defaults_cf21bl.h` overrides any of these for the brushless. That is a stage-0 paper-scout item (digest to `docs/papers/`).

### 2.2 The law in our coordinates (all SI, BODY axes, per axis a ∈ {roll, pitch, yaw})
Runs on the rate loop's update ticks (500 Hz), holds between, exactly as `RateStageFw` does.

1. **Angular acceleration.** ω_f = H(ω_gyro), H = 2nd-order Butterworth low-pass (reuse `pid_firmware::Lpf2p`). ω'_f = (ω_f[k] − ω_f[k−1])·f_rate.
2. **Actuator model.** F_act = the rotors' modelled differential thrust (N) in the rate stage's axis-force coordinates: the pwm the plant was TOLD to apply, clamped to [0, 1], through the plant's lag coefficient, then k_thrust·pwm² per motor, then the '+' mixer rows. This is `CascadeDhat::track_lag` + `lagged_force` verbatim, including the prime/seed protocol. Working in thrust (not pwm) coordinates makes the square law exact, so G is constant.
3. **Synchronisation filter.** F_f = H(F_act) with the SAME H and the same sample instants as step 1. This is the requirement Smeur 2016 states: the measured ω' and the actuator signal it is compared with must carry identical delay, otherwise the increment acts on a phase-shifted difference.
4. **Virtual control.** ν = K_ω·(ω_sp − ω_gyro) (rad/s²). ω_sp comes from the outer stage (section 2.4).
5. **Increment.** F_cmd = F_f + (ν − ω'_f)/c, with c = `cascade_dhat::torque_gains` = [arm/Ixx, arm/Iyy, 4·k_drag/Izz] (rad/s² per N). Clamp to the rate stage's ±out_limit_n.
6. **Mix.** `RateStageFw::mix` (hover thrust + axis forces → clamp → sqrt → pwm) plus the uniform collective deviation. One definition of the mixer; INDI does not get its own.

Notes:
- **G2 = 0.** The plant has no rotor-inertia yaw reaction (`controller.rs` models yaw torque as k_drag·thrust only), so the firmware's g2 term has nothing to invert.
- **No integrator, no anti-windup needed.** The increment is taken from the modelled ACTUAL actuator state, not from the last command, so saturation cannot wind up.
- **Start of episode.** The plant seeds its lag at the first command (no spin-up transient). The INDI filters must be seeded at their first sample the same way; the first decision has no ω' and must fall back to F_cmd = ν/c. This is the `ActuatorLag` prime/seed bug class; reuse the protocol, do not re-derive it.

### 2.3 Why this is the existing observer (the central physics finding)
Define d = ω' − c·F_act, the angular acceleration the nominal rotor moment does not explain (wear, torque-scale jitter, rotor drag).
- `CascadeDhat`: d̂ ← d̂ + 0.02·(d − d̂) at 1 kHz. A first-order low-pass of d, τ = 50 ms (3.2 Hz).
- INDI: F_cmd = F_f + (ν − ω'_f)/c = (ν − (ω'_f − c·F_f))/c = **(ν − H(d))/c**.

So INDI is nominal dynamic inversion minus a disturbance estimate that is the SAME signal `cascade_dhat.rs` already builds, with a different filter, applied as a feed-forward moment inside the rate loop at 500 Hz. MPCOF applies it through the QP's steady-state target x_ss(d̂) at 100 Hz, with the rate integrator carrying the trim.

Consequences:
- The two differ in (a) filter order and bandwidth, (b) where the cancellation enters (inner feed-forward vs outer plan), (c) INDI has no model of the rate PID to maintain.
- At the firmware's 8 Hz cutoff, H has a low-frequency group delay of √2/(2π·8) = 28 ms. With the 50 ms rotor lag after it, INDI's rejection time scale is about 80 ms. MPCOF's is 50 ms (observer) plus the plan and the same rotor lag. **Comparable, not an order of magnitude apart.**
- L4C has no gust. Every moment disturbance in the test regime is quasi-static per unit (a thrust-proportional asymmetry). Fast rejection, INDI's main advantage in the literature, has little to act on here.

### 2.4 The outer loop INDI sits under
Three candidates. All produce a BODY-frame rate setpoint from the Euler attitude error (`optimal::state_error`, the firmware attitude stage's convention).

| outer | law | role |
|---|---|---|
| O1 firmware INDI attitude stage | ω_sp = K_η·e_att, K_η = 5 s⁻¹ (REF_ERR) | the sourced "INDI as shipped" rival row |
| O2 MPC on the INDI inner loop | condensed QP like `ctbr_axis_mpc.rs`, model = INDI rate loop + lag; no offset-free term needed | the strong candidate, only under variant A |
| O3 MPCOF as today | unchanged; INDI supplies only the moment increment | the hybrid fallback (section 6.4) |

Expectation, stated before measuring: O1 is softer than what we fly. K_η = 5 s⁻¹ is below the firmware PID's attitude kp 6 and well below MPCOF's fitted 8.7 rad/s per rad. With K_ω·τ = 24·0.05 = 1.2 the rate loop's damping ratio is about 0.46 before the filter delay. **INDI + O1 will likely lose to MPCOF on err (transient) even if it wins on wear rejection.**

### 2.5 What the student imitates — two variants

**Variant B (recommended): label transport, student unchanged.**
- The student, its observation, its 4 CTBR banks, its firmware rate PID inner loop and the Metal kernel stay exactly as flown.
- The INDI teacher computes its desired axis force F*. The label is the rate setpoint that makes the student's REAL inner loop output F* at its next update:
  sp* = ω_gyro + (F* − I − D) / (kp + ki·dt_rate), clamped to the CTBR rate box (4.2 / 4.2 / 2.0 rad/s).
  I and D are read from the inner loop that is actually flying, through the existing `Teacher::ctbr_sync_inner` hand-off (MPCOF already uses it).
- Interface: same label coordinates as MPCOF's. So the two teachers can be mixed by the existing `teacher_schedule` / `teacher_blend` selector, and a later per-state arbitration or a committee is well defined (section 7.3).
- Loss: sp* is held for 10 ticks (5 rate updates), so the match is exact only at the decision tick. Stage 0 measures what survives (row R2 in section 6.1).
- Thesis fit: the flown cost is unchanged. Only the labels change.

**Variant A: INDI becomes the flown inner loop.**
- `inner=indi@500Hz` for teacher AND student. The student still emits rate setpoints; the interface width is unchanged, but the meaning of a setpoint changes.
- Needs: a Metal twin of the INDI stage in `ctbr_inner_tick`, new thread-local state, every host that runs the inner tick (scorer, DAgger, BPTT walk CPU/GPU, split trainer, calibration, recorder, probes), the MPC re-condensed on the INDI loop (O2), and a redefinition of the `obs_rate_i` and `dhat_resid` feature blocks (there is no rate integrator under INDI, so (d̂ + c·ki·I) has no meaning).
- New lineage: no result is comparable with CTRL-48 without re-flying controls.
- Thesis fit: weaker. It is a stronger classical inner loop, not a stronger teacher. The flown cost changes (small: two biquads and four lag states per axis set).
- Paper note: the inner loop would no longer be "the gains the vehicle ships with by default"; the firmware does ship INDI as a selectable controller.

Recommendation: B first. Escalate to A only if stage 0 shows the INDI gain exists at full rate (row R1) and is lost in transport (row R2).

## 3. Observability on our plant and sensor model

| quantity | available? | how, and how good |
|---|---|---|
| body rate ω (BODY) | yes, measured | gyro every 1 kHz tick, white σ = 0.01073 rad/s, bias walk 3.88e-5 |
| angular acceleration ω' | yes, derived | finite difference of the Butterworth-filtered gyro. Derivative-noise rms: 0.07 rad/s² at 8 Hz, 0.18 at 15 Hz, 0.52 at 30 Hz, 1.1 at 50 Hz (white-noise integral 4π²·S·1.11·fc³, S = σ²/500 Hz) |
| rotor moment actually applied | no, modelled | from the commanded pwm through the plant's own lag coefficient. τ exact for every unit; gain wrong by the unit's wear |
| control effectiveness G | modelled | c·(thrust slope). Roll/pitch 1307 rad/s² per unit of axis pwm at hover (pwm 0.694); yaw 119 |
| moment disturbance d | yes | ω' − c·F_act, the `CascadeDhat` signal. This is what makes INDI work |
| attitude (roll/pitch) | biased | Mahony on a specific-force accelerometer (ABI 32): bias sd 0.44 / 0.41° under translation. **Outer-loop quantity; INDI does not see it** |
| yaw heading | drifts, small | gyro-integrated; within 0.017° of truth over a 2 s episode |
| rotor speed | no | not in the observation, and not logged on a stock CF-BL |

- **Noise is not the limiter.** At 8 Hz the derivative noise is 6e-5 of a unit roll command and 6e-4 of a unit yaw command. Yaw authority is 11× weaker than roll/pitch, so yaw is where a wider filter would cost first.
- **Gyro bias** differentiates to nothing, so the INDI inner loop is immune to it. It enters only through ν as a rate offset of about 5.5e-5 rad/s after 2 s. Negligible.
- **Filter delay vs lag.** 28 ms of filter delay in front of a 50 ms actuator is acceptable (the firmware flies 8 Hz with a modelled lag of 62 ms; my conversion of `ACT_DYN 0.03149` per 500 Hz step, τ = −2 ms / ln(1 − 0.03149)).
- **Wear (±20% per motor) and torque-scale jitter (±20% per axis).**
  - As a moment bias (asymmetry × collective thrust): measured through ω' and cancelled. This is INDI's intended use.
  - As a gain error κ = G_true/G_model: per axis κ lies in about [0.64, 1.44] at the corners (0.8·0.8 to 1.2·1.2). The incremental loop's gain scales by κ; with synchronised filters INDI tolerates this (stability needs 0 < κ < 2 for the synchronised first-order case). The corners are 28% from the limit, so the fail tail must be checked on the worst units, not assumed.
  - Collective coupling: `RateStageFw::mix` adds the collective as a pwm offset after the square root, so the true differential force differs from the commanded one by a thrust-slope factor. INDI sees the true lagged force through pwm², so this is one more bounded gain error, not a bias.
- **Inertia.** The brushless inertia in `wnn/control/airframe.py` is DERIVED (scaled from the measured 2.x): (3.00, 3.02, 5.30)e-5 kg·m². Gräfe 2026 Table II gives (3.3, 3.6, 5.9)e-5 with guards at 44 g. That is a 10-19% G error on the real vehicle, inside the same tolerance. Worth stating in the paper.
- **Student-side observability (Hard Rule 1).** An INDI label depends on H(d). The student sees d only through `--obs-cascade-dhat-residual`, which is the 3.2 Hz first-order d̂. If the label's useful content sits between 3.2 Hz and 8 Hz, the student cannot address on it, and the gain will not transfer. If stage 0 passes, stage 2 must check this offline (regress the transported label on the student's features in the hold, as teacher doc §1 did) before training. A matched-filter feature would be a feature-layout change: every consumer of the layout has to be audited (`feature_layout.py`, `cascade_obs.rs`, the kernel's feature block, `arch_shape_from_spec`).

## 4. Gains and parameters

Rule: sourced or derived from the airframe model. Nothing is tuned on the test units.

| parameter | value | origin |
|---|---|---|
| structure, rate, filter type | section 2.2, 500 Hz, 2nd-order Butterworth | `controller_indi.c` |
| K_η (attitude → rate) | 5 s⁻¹, all axes | `STABILIZATION_INDI_REF_ERR_*`, verbatim |
| K_ω (rate → ang. accel.) | 24 s⁻¹, all axes | `STABILIZATION_INDI_REF_RATE_*`, verbatim |
| filter cutoff | 8 Hz, all axes | `STABILIZATION_INDI_FILT_CUTOFF`, verbatim |
| G1 | c = [arm/Ixx, arm/Iyy, 4·k_drag/Izz] in force coordinates | DERIVED from the airframe (`torque_gains`). The header's G1 numbers are in the brushed vehicle's actuator counts and are NOT used |
| G2 | 0 | plant has no rotor-inertia term |
| actuator coefficient | the plant's `motor_lag_alpha` composed over the 2 ticks of a rate period | DERIVED from `--motor-lag-s`. The header's 0.03149 is NOT used |
| output limit, hover, mixer | the rate stage's own | `RateStageFw` |

- Caveat to disclose: the three verbatim gains are the firmware's brushed CF2.x defaults. They are used unchanged on the brushless, as a sourced starting point, exactly as a user selecting the INDI controller on that vehicle would get them (subject to the cf21bl header check in 2.1).
- Home for the Python side: a small `IndiGains` dataclass with `source` strings in its own file under `wnn/control/` (one class per file), registered next to `PidGains` in `airframe.py`. The SI boundary stays in Python, as for the PID.
- **If any knob is opened** (cutoff, K_η, K_ω): candidates are compared on the 20 val units (`val_units.py::val_unit_seeds`) only, the choice is committed with its hash, and the test units are scored once with the frozen value. No knob is opened by default.
- **Adaptive INDI (Smeur 2016 LMS on G): not in v1.** Episodes are 2 s; the LMS would not converge within one, and the gain error is inside the fixed-G tolerance. Revisit only if stage 0 fails concentrate on the extreme-wear units.

## 5. Where it plugs in

### 5.1 Rust (ram_controller only; swap-free rebuild)
- **New `measured_actuator.rs`**: extract `CascadeDhat::track_lag` + `lagged_force` into one type used by both `CascadeDhat` and INDI. One definition of the measured actuator. `ctrl38_pins` must stay bit-identical across the extraction.
- **New `indi_rate.rs`**: `IndiRateStage` with the `RateStageFw` surface (`reset`, `step(rate_sp, gyro, coll_dev) -> [f64; 4]`, `phase`, plus `observe(applied_pwm)` and `axis_force()`).
- **New `indi_teacher.rs`**: `AttitudeIndiRs` = attitude stage O1 + `IndiRateStage`; `step_rs` (motor-level), `ctbr_rate` (the transported setpoint of section 2.5), `observe`, `reset`.
- **`optimal.rs`**: `Teacher::Indi`, id **5**, in `from_id*`, `ctbr_label`, `ctbr_sync_inner`, `observe`, `reset`, `integrals`, `hover`, `diag_ctbr_hidden`. Every `match self` on `Teacher` must be exhaustive with no `_` arm, so the compiler lists the sites.
- **`dagger_train.rs`**: `TEACHER_IDS` 5 → 6; `AirframeRs::teacher` builds id 5 only when the airframe supplies the firmware cascade (it needs the rate stage's mixer); `score_classical_baseline` and `position_score` accept id 5, at full rate and with `ctbr_outer_hold`.
- **`lib.rs`**: `ABI_VERSION` 50 → 51. Student paths are bit-identical (pins), but the teacher id space and the baseline signature change, and the facade must refuse a stale wheel.

### 5.2 The TeacherBank guard (the 05/08 bug must not recur)
The bug: the bank was `[_; 3]`, ids > 2 silently became PID, and the baselines stayed correct because they bypass the bank. Guards, all in `cargo test`:
1. **No literal to forget.** Build the bank with `std::array::from_fn(|_| None)` so its length is `TEACHER_IDS` by construction.
2. **Kind map.** Extend the existing test: ids 0..6 report `["pidfw", "lqr", "mpc", "lqi", "mpcof", "indi"]` on the cf21 cascade config.
3. **Distinct output.** On one fixed L4C episode, the id-5 label stream differs from the id-0 and id-4 streams on at least half of the decision ticks, through the BANK. This is the symptom that exposed the original bug (two cells bit-identical).
4. **Flown = labelled.** Label-replay identity, the `abi38_tests` pattern: the setpoints `score_classical_baseline(5, ctbr_outer_hold = 10)` flies equal, bit for bit, the labels the DAgger recorder writes on an expert-driven episode of the same seed.
5. **Bank = baseline.** For every id, the teacher the bank returns and the one `AirframeRs::teacher` returns produce identical commands on one episode. This closes the "baselines bypass the bank" gap for all ids, not just INDI.
6. **Unknown id is an error.** Proposed: `TeacherBank::get_mut` and `Teacher::from_id` return `Err` for an id they cannot build, instead of PID. One existing test asserts the PID fallback and would be updated. Luiz's call (D6).
7. **Provenance from the wheel, not from argv.** A `teacher_kind(cfg) -> String` pyfunction reports what the bank actually built; the run aborts at start if it differs from `--teacher`, and the `[provenance]` line prints the wheel's answer.

### 5.3 CPU and Metal
- Checked: `shaders/controller_rollout.metal` contains the student's inner loop (`pidfw_rate_half`, `ctbr_inner_tick`), the in-kernel firmware PID (residual baseline and the C2 identity test) and the student's d̂ feature. It contains **no MPCOF**. Teachers label on the host and rivals are CPU-scored (`ctrl48_fixed_scorer_table.md`: "Rivals CPU-scored").
- **Variant B therefore needs no kernel change.** The parity obligation is that nothing moved: the full suite, `cargo test -p ram_controller --lib --no-default-features` (all CPU/GPU parity sweeps), passes unchanged, plus the `measured_actuator` extraction pin against the kernel's d̂ twin (`cascade_blocks_cpu_scorer_matches_gpu`, `residual_cpu_scorer_matches_gpu`).
- **Variant A** needs the kernel twin, `Params` fields appended at the END (layout must match `RolloutParams`), and a new parity sweep `indi_inner_cpu_scorer_matches_gpu` plus trainer and split-trainer sweeps, all in `cargo test`. No "run parity from Python".

### 5.4 Python
- `evaluator.py::_TEACHER_IDS` gains `"indi": 5`. Five scripts carry private copies of this map (`score_classical_baselines.py`, `score_position_teachers.py`, `teacher_step_histogram.py`, `transient_decomposition.py`, `teacher_wiring_smoke.py`); point them at the one map rather than adding a sixth entry five times.
- `phased_ga.py`: `--teacher` choices, the CTBR allow-list (line 447, today `{"pid", "mpcof"}`), the schedule validator (line 901).
- `classical_baseline.py`: `cascade_rival_metrics` name map gains `5: "INDI"`; rows `INDI[est]`, `INDI[oracle]`, `INDI>rate(h10)[est]`.
- `provenance.py::arch_token`: `teacher=indi`; under variant A also `inner=indi@500Hz`.
- All through `wnn/control/_accel.py`. No Python reimplementation of the law: the stage-0 breakdown probe reads Rust through a diagnostic export, as `diag_ctbr_hidden` does today.

## 6. Acceptance tests before any student is trained

### 6.1 Rows (test: the 35 units 99990101-135 × 100 episodes, the ARGS_35 regime, CPU scorer)
| row | what it is | why |
|---|---|---|
| R0 | MPCOF>rate(h10), est and oracle | control; must reproduce 1.352 / 0.684 / 0.341 and 0.163 oracle steady exactly |
| R1 | INDI, full 500 Hz, motor-level, est and oracle | does INDI's mechanism help on this plant at all |
| R2 | INDI→sp>rate(h10), est and oracle | the transported setpoints through the firmware rate PID at the student's 100 Hz hold. **This is what a perfect variant-B student would fly; the bar applies here** |
| R3 | PID[est], PID>rate(h10)[est] | context |

Each row reports the triple plus pos and fails: stable % / err ° / steady ° / pos m / fails.

### 6.2 Breakdown (the `ctrl_steady_gap_block6.md` format, hold window t ≥ 1.6 s)
For R0, R1, R2: steady scored; roll/pitch only; mean |yaw|; mean |true roll/pitch error|; mean |estimate-frame error|; mean |Mahony bias|; compensation slope k; and the per-unit table against roll/pitch asymmetry. Frames as in that document: roll/pitch are ZYX-Euler errors of the vehicle against the tilt reference; yaw is absolute heading against the yaw-0 reference; `steady` is the geodesic angle and includes yaw.
Plus the error-budget rows of teacher doc §2 for INDI: wear off, all disturbances off, noise off, lag 0. These say where INDI's number comes from.

### 6.3 Pre-registered bar: "INDI is worth teaching from"
All of the following on **R2**:
1. stable 100.0%, 0 fails in 3500 episodes;
2. steady ≤ **0.60°** est-fed (MPCOF 0.684°; floor from the budget ≈ 0.51°), i.e. at least half of the addressable 0.17°;
3. steady ≤ **0.10°** oracle-fed (MPCOF 0.163°). This is the label source, so this is the number that says the labels got better;
4. err ≤ 1.35°, pos ≤ 0.346 m (no trade against transient or position);
5. mean |yaw| in the hold ≤ 0.08° (no regression on the axis the teacher already holds).

- Why 0.60: the best student tracks the rival to +0.09° at n = 3. A teacher improvement much smaller than that cannot be told apart after imitation. The exact threshold and the n needed to read a student-level difference are `experiment-design`'s call.
- R1 passing while R2 fails means the gain is real but lost in transport: that is the case for variant A.
- The numbers 0.60 / 0.10 are a proposal. Luiz fixes them before stage 0 scores anything (D3).

### 6.4 If INDI cannot beat MPCOF here
Meaning: on this plant the moment-disturbance channel is already closed by MPCOF's offset-free observer, as section 2.3 predicts. The teacher's ceiling is then the estimator, not the inner loop. Fallbacks in order of expected value:
1. **Bias-compensated MPCOF (F1).** Correct the teacher's Mahony attitude with the predictor the student already uses (`att_bias_pred.rs`; R² 0.98-0.99 in flight, onboard signals only). This targets the 0.52° term, three times INDI's addressable 0.17°. It is the same lever that put the biasin student below the rival on roll/pitch (0.62° vs 0.67°). It is an est-fed feed change on the CPU scorer, so it can be scored in stage 0 at almost no cost (D5).
2. **MPCOF + INDI-bandwidth observer (F2).** Keep MPCOF; give `CascadeDhat` INDI's second-order filter or a higher gain. This is INDI's content with no new interface. Constraint: the student's d̂ feature shares that law and its kernel twin, so teacher and student filters must be separated explicitly or changed together.
3. **MPCOF over an INDI inner loop (F3 = variant A with O2).** Only if R1 passes and R2 does not.
4. **NMPC + disturbance observer (F4).** Buys coupled axes, the square-law thrust and saturation handling. At 5° tilt these are small; expect transient (err) gains, little steady. Highest cost. Last.

### 6.5 A cheaper falsifier first (optional stage −1)
Section 2.3 says INDI ≈ a faster d̂. That can be tested without writing INDI: an `#[ignore]` cargo test that scores the existing MPCOF cascade on the 20 VAL units with the observer gain at 0.02 (control), 0.05, 0.1 and 0.2. If steady does not move, a faster disturbance estimate has nothing to give on this plant and INDI's prior drops sharply. Pure Rust, no wheel, about half a day. It is a val-unit read; it sets expectations, it does not replace section 6.3.

## 7. Risks, unknowns, extensibility

### 7.1 Risks
- **Small addressable gain.** At most 0.17° est-fed, and section 2.3 argues MPCOF already takes most of it. This is the main risk, and it is why stages −1 and 0 come before any training.
- **Soft outer stage.** K_η = 5 s⁻¹ likely costs err against MPCOF (section 2.4).
- **Gain-error tail.** Corner units sit at κ ≈ 0.64 and 1.44. Expect any INDI fails there; read fails per unit.
- **Transport loss.** The 100 Hz hold may erase a gain that exists at 500 Hz (R1 vs R2).
- **Student cannot see it.** Section 3, last item.
- **Seeding and phase.** Filter seeding at episode start, the 500 Hz phase against the 100 Hz decision tick, and the observe() protocol (gyro just read, pwm applied last tick) are where the DOB bug of 12/08 and the lag-estimate bugs lived.
- **Extraction risk.** Moving `track_lag` out of `CascadeDhat` touches a law the Metal kernel twins.

### 7.2 Unknowns
- Why wear leaves 0.16° under an offset-free controller (oracle 0.163° vs 0.002° with all disturbances off). If it is the thrust-proportional asymmetry moving with the collective during translation, a faster d̂ helps; if it is yaw trim riding in the setpoint (teacher doc §1), it does not. The budget rows of 6.2 answer this.
- Whether the cf21bl platform header overrides the INDI defaults.
- Whether the oracle-fed label improvement, if any, survives into an est-observing student.

### 7.3 Designed-in for later (not built now)
- **Mixing by episode or round** works at id level today (`teacher_id_for(round, episode)`, `teacher_schedule`, `teacher_blend`), and under variant B both teachers label in the same coordinates.
- **Per-state arbitration** needs every teacher in the bank kept warm: `observe()` on every tick for every built teacher, not only the selected one, because d̂ and the INDI filters are stateful. Stage 2 should make the bank's observe fan out, with a test that a teacher selected mid-episode gives the label it would have given had it been selected from t = 0.
- **A committee of students** needs nothing from this spec beyond distinct teacher ids in provenance.
- Variant A breaks the shared label coordinates. That is a second reason to prefer B.

## 8. Staged plan and effort

| stage | content | gate to pass | effort |
|---|---|---|---|
| −1 (optional) | observer-gain sweep on val units, `#[ignore]` cargo test | none; informs priors | 0.5 day |
| 0 | CPU-only: `measured_actuator.rs`, `indi_rate.rs`, `indi_teacher.rs`, `Teacher::Indi`, bank guards 1-7, ABI 51, Python ids/flags/rival rows; paper-scout digest of the firmware INDI; score R0-R3 and the 6.2 breakdown; optionally F1 | section 6.3 bar on R2 | 3-4 days |
| 1 | Metal parity. Variant B: full `cargo test` suite unchanged + extraction pin. Variant A: kernel twin + new sweeps | all sweeps green | B: 0.5 day. A: 5-8 days |
| 2 | DAgger labels: transported label under the real inner loop (`ctbr_sync_inner`), label-replay identity, offline label-vs-feature regression, bank observe fan-out | identity bit-exact; label explained by student features in the hold | B: 1-2 days. A: +2-3 days (MPC on the INDI loop) |
| 3 | student run: the "both" recipe with `--teacher indi`, 3 seeds paired against the same-seed "both" control, rule P | the 11/11 failure gate, then steady, then err | about 1 day of box time per 3-seed arm; mechanics are `controller`'s |

- Total, recommended path (B): about 6-8 working days plus one block. Variant A: about 14-18 days plus re-flown controls.
- Files touched in stage 0: new `measured_actuator.rs`, `indi_rate.rs`, `indi_teacher.rs`, `indi_tests.rs`; edited `cascade_dhat.rs`, `optimal.rs`, `dagger_train.rs`, `position_score.rs`, `lib.rs`; Python `evaluator.py`, `phased_ga.py`, `classical_baseline.py`, `provenance.py`, `airframe.py`, new `indi_gains.py`, five scripts. No `ram_core` change, so only `ram_controller` is rebuilt and no worker swap is needed.
- Nothing in stages −1 to 2 needs GPU time. Stage 0 scoring is 4 rows × 2 feeds on the CPU scorer.

## 9. Decisions for Luiz

- **D1. Student interface.** Variant B (label transport, student and kernel unchanged; recommended) or variant A (INDI as the flown inner loop; new lineage, Metal work).
- **D2. Gain sourcing.** Firmware `controller_indi` structure with K_η 5, K_ω 24, 8 Hz verbatim, G and actuator coefficient derived from the airframe, no tuning (recommended). Or allow a val-unit choice of the filter cutoff.
- **D3. The bar.** Confirm or change: 100% / 0 fails; steady ≤ 0.60° est-fed and ≤ 0.10° oracle-fed; err ≤ 1.35°; pos ≤ 0.346 m; |yaw| ≤ 0.08°; applied to R2.
- **D4. Stage −1.** Run the half-day observer-gain falsifier first (recommended).
- **D5. Score F1 in stage 0.** Bias-compensated MPCOF addresses 0.52° where INDI addresses 0.17°. Recommended to score alongside, and to consider as the first "stronger teacher" if INDI misses the bar.
- **D6. Unknown teacher id becomes a hard error** instead of a silent PID (recommended).
- **D7. Adaptive INDI.** Out of v1 (recommended).

---

## 10. Stage −1 + bias-compensated MPCOF: what is run and what each outcome means (PRE-REGISTRATION, 14/11/2026)

Approved by Luiz 14/11/2026 (D4 and D5). Evaluation only. INDI itself is not approved and is not built.
Written and committed BEFORE any number in section 11 was scored.

### 10.1 Common protocol
- **Units: VAL only.** The 20 stage-select val units of the "both" s1 run (base seed 31337001): seeds **163226054..163226073**, taken through `val_units.py::checked_val_unit_seeds` (so the disjointness from the test units and from the train/test pool seeds is asserted, not assumed). The fixture script additionally refuses any seed in 99990101..99990400. No test unit is scored.
- **Episodes.** 100 per unit (2000 total), 2000 steps, the K = 5 fold-0 pool seed of each unit, exactly as `classical_baseline.cascade_rival_metrics` draws a rival row.
- **Regime.** The run's own `[argv]` (cf21_brushless, L4C, translation, xy offset 0.5, tilt reference, motor lag 0.20 s, rotor drag 0.78, specific-force IMU). Episode inputs are produced by the shipped Python samplers and written to a fixture; the scoring is an `#[ignore]` cargo test on the shipped Rust scorer with default-off probe parameters. No wheel is built.
- **Row.** MPCOF>rate(h10): the MPCOF outer decision held 10 ticks over the firmware rate PID. Est-fed (Mahony) and oracle-fed (true quaternion) are reported separately and labelled.
- **Metrics.** stable % / err ° / steady ° / pos m / fails (episodes not stable, out of 2000), plus the hold-window (t ≥ 1.6 s) per-axis mean |error|: roll, pitch (TRUE Euler angle minus the tilt reference) and yaw (absolute TRUE heading). All val numbers: tuning-grade, never to be placed next to a test number without saying so.
- **Self-check before any reading.** The probe with every option off must reproduce the unprobed scorer bit for bit on the same unit.
- **One pass.** The settings below are fixed here. Anything added after seeing numbers is labelled post hoc.

### 10.2 Stage −1: observer speed
- **Arms.** The cascade observer gain l of `cascade_dhat.rs` (per 1 kHz tick; τ = 1 ms / l): 0.005, 0.01, **0.02 (shipped)**, 0.05, 0.1, 0.2, 0.5, i.e. τ = 200, 100, 50, 20, 10, 5, 2 ms. Only the TEACHER's observer instance is changed; nothing else in the law moves.
- **Question.** Does a faster (INDI-like) disturbance estimate improve the hold on this plant?
- **Outcomes.**
  - No setting improves est-fed steady by more than **0.03°** against the shipped 0.02 without adding fails: **INDI's inner-loop route is closed** on this plant. The spec's stage 0 for INDI is not built.
  - Some faster setting improves est-fed steady by more than 0.03° with fails not above the shipped row and err not worse by more than 0.03°: the route **stays open**, and the size of the gain says whether INDI (or simply that gain, fallback F2) is worth a stage 0.
  - The oracle-fed column is read the same way and reported as the label-source view. A gain that appears only oracle-fed means the estimator masks it for a rival but the labels would still carry it; that is reported as such, not as "open".
  - Yaw is reported per setting. A yaw gain alone, without a steady gain above 0.03°, does not open the route.

### 10.3 Bias-compensated MPCOF (fallback F1)
- **What it is.** The shipped est-fed MPCOF cascade, with the attitude reference handed to the teacher shifted by the predicted Mahony bias: target' = tilt reference + [abp_r, abp_p, 0], where [abp_r, abp_p] is `att_bias_pred.rs::AttBiasPredictor` (the student's `--obs-att-bias-pred` law, unchanged), i.e. the predicted (estimate − truth) of roll and pitch. Yaw is untouched.
- **Information it uses.** The Mahony estimate quaternion; the horizontal velocity (the same sim signal the student reads as `--obs-vel-xy` and the rival's own position loop already reads; noise-free in sim, as for them); the Mahony gains, dt and nominal g. **It never reads the true quaternion.** A student with `--obs-att-bias-pred` observes the same two numbers.
- **Outcomes** (Δ against the shipped est-fed row on the same 20 units).
  - Est-fed steady improves by more than **0.10°**, with 0 extra fails, err not worse by more than 0.03° and pos not worse by more than 0.005 m: it **becomes the stage-0 teacher candidate**, to be confirmed once on the 35 test units, only on Luiz's go.
  - Improves by 0.03° to 0.10°: a direction, not adopted as built.
  - Improves by less than 0.03°, or adds fails: does not work as built.
- **Not claimed either way.** Whether labels from it help a student. The DAgger labels already come from the true quaternion, so this row is a rival-side and est-label-side quantity; that question is stage 2's.
