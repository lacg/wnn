# CTRL-86 — X500 (500 mm quad) airframe: step-2 implementation plan + prereg

Status: PLAN, not implemented. Written 27/11/2026 (flight-dynamics). Nothing here has been
run. Plane story CTRL-86 is not updated by this document.

Tags used throughout:
- **[S]** sourced: the number is read from a pinned file or datasheet, cited inline.
- **[D]** derived: computed from [S] numbers by a formula stated here.
- **[A]** assumed: no source found. It is named as an assumption and goes in the paper's limitations.
- **[V]** to verify in the step that implements it. Recalled from the PX4 tree; the line was not re-read for this plan.

Luiz's decisions of 27/11 that this plan implements:
1. PX4 SITL numbers now, measured X500 V2 numbers in 2028.
2. IMU noise from the Pixhawk 6X IMU datasheet (ICM-42688-P class).
3. Rate loop at 1 kHz.
4. Wind is separate and later. Out of scope here.
5. Model the real positioning system. Choice proposed in §6, flagged for Luiz.
6. Battery with a sag model at 4S. 6S stays open, see §7.
7. Asymmetric motor lag (separate up and down τ) in Rust and Metal, with CPU/GPU parity.

Constraints carried in: PID is the firmware cascade with sourced gains and is never re-derived
(project_pid_not_airframe_retuned). No observer and no MPCOF part in any WNN lever
(feedback_no_observer_no_mpcof_levers). MPCOF stays a rival and a teacher only. A
robustness claim needs at least 30 units (project_airframe_tail_report_seeds). Every
config-pack site must receive the airframe (project_trainer_airframe_gap).

---

## 0. Frames (named once; every later claim uses them)

- **Sim world frame:** z up. Gravity `(0,0,-g)` in WORLD. The accelerometer reads specific
  force `= -R_wb·g_world`, i.e. +g on body z at hover.
- **Sim body frame:** FLU (x forward, y left, z up). The '+' mixer in
  `controller.rs::body_torque` uses motor 0 front (+x), 1 right (-y), 2 rear (-x), 3 left (+y).
  Its torques are roll `L(-T1+T3)`, pitch `L(-T0+T2)`, yaw `k(T0-T1+T2-T3)`.
- **PX4:** body FRD, world NED. Body vectors (rates, torques) map as
  `v_FRD = diag(1,-1,-1)·v_FLU`, so roll keeps its sign and pitch and yaw flip.
- **The PX4 port lives in the sim's FLU frame.**
  - The attitude law `eq = 2·imag(q⁻¹ q_d)` is rotation-covariant. It needs no frame change,
    provided the "thrust axis" in the yaw-weight reduction is body +z (FLU) and not −z (FRD).
  - The only place a sign map is applied is the effectiveness matrix: PX4 builds it in FRD.
    Rotate the rotor positions and axes into FLU before the cross product. It is pinned by the
    PX4 reference test in §3, step S3.

---

## 1. Parameter table

PX4 pin: **PX4-Autopilot tag `v1.16.0`** (annotated tag 439b31fd…, commit
`6ea3539157ca358c70a515878b77077af7d4611d`). The gz models are a submodule pinned at
**PX4-gazebo-models `e05f4312d3f28aa621157610584a4870406cb6d3`**. All files below were
fetched from those exact refs on 27/11/2026.

### 1a. Plant (from `models/x500_base/model.sdf` + `models/x500/model.sdf`)

| Quantity | Value | Tag | Source / derivation |
|---|---|---|---|
| body mass | 2.0 kg | S | x500_base `base_link <mass>` |
| rotor mass ×4 | 0.016077 kg each | S | `rotor_i <mass>` |
| total mass | **2.06431 kg** | D | 2.0 + 4×0.016077 |
| body inertia | 0.021667 / 0.021667 / 0.040 kg·m² | S | `base_link <inertia>`, products 0 |
| rotor position | (±0.174, ±0.174, +0.06) m | S | `rotor_i <pose>`, relative to the base_link origin |
| aggregate inertia | **Ixx 0.023839, Iyy 0.023942, Izz 0.044000 kg·m²** | D | parallel axis about the combined CoM (z_c = 1.87 mm above the base origin), plus rotor self-inertia |
| motor radius r | 0.24607 m | D | hypot(0.174, 0.174) |
| mixer arm `arm_length` | **0.348 m** | D | `axis_arm_from_radius(r) = r·√2`, the same '+'-equivalence airframe.py uses for cf21 |
| motorConstant kf | 8.54858e-6 N/(rad/s)² | S | x500 `<motorConstant>` |
| maxRotVelocity | 1000 rad/s | S | `<maxRotVelocity>` |
| `k_thrust` (pwm=1) | **8.54858 N per motor** | D | kf·ω_max². Sim pwm ≡ ω/ω_max, thrust = k·pwm² |
| T/W | **1.6885** | D | 4k/(mg) |
| hover pwm | **0.76956** (ω_h 769.6 rad/s) | D | √(mg/4k) |
| momentConstant → `k_drag` | **0.016 m** | S | `<momentConstant>`. Yaw torque = 0.016·T per rotor |
| motor τ up / down | 0.0125 s / 0.025 s | S | `<timeConstantUp/Down>`. The gz filter acts on rotor VELOCITY, our filter acts on pwm ∝ ω, so they are equivalent |
| lag settling T_up / T_down | **0.050 s / 0.100 s** | D | the codebase convention T = 4τ (Molchanov eq. 7, `apply_motor_lag`) |
| rotorDragCoefficient | 8.06428e-5 per rotor | S | `<rotorDragCoefficient>` (gz: F = −ω·c·v⊥ per rotor) |
| linear rotor drag k_d | **0.120 1/s** | D | 4·ω_h·c/m, evaluated at hover. Same model form as `DEFAULT_ROTOR_DRAG` (Faessler 2018) |
| rollingMomentCoefficient | 1e-6 | S | ignored [A]: the sim has no rolling-moment term |
| motor idle (armed) | ω_min 150 rad/s → **pwm floor 0.15** | S | 4001_gz_x500 `SIM_GZ_EC_MIN 150`, `SIM_GZ_EC_MAX 1000` |
| PX4 actuator u → ω | ω = 150 + 850·u, so pwm = 0.15 + 0.85·u | S/D | gz_bridge `GZMixingInterfaceESC::updateOutputs` publishes the output value as the rotor velocity. Linear min..max map [V] |
| gravity | 9.81 m/s² | A | gz world default is 9.8. We keep 9.81 for all airframes, so the difference is 0.1 % |
| roll angular authority | **124.8 rad/s²** (cf21 470.8) | D | L·k/Ixx |

Roll and pitch are treated as identical (Iyy/Ixx = 1.0043). The plant keeps its diagonal I.
`Ixy = 0` in the SDF, so nothing is dropped.

### 1b. PX4 controller (module defaults; 4001_gz_x500 overrides none of them)

`ROMFS/px4fmu_common/init.d-posix/airframes/4001_gz_x500` sets only CA geometry, gz ESC
ranges, `MPC_THR_HOVER 0.60` and `NAV_DLL_ACT`. Every gain below is therefore the module
default.

| Param | Default | Tag | File @ v1.16.0 |
|---|---|---|---|
| MC_ROLL_P / MC_PITCH_P | 6.5 1/s | S | `src/modules/mc_att_control/mc_att_control_params.c` |
| MC_YAW_P | 2.8 1/s | S | same |
| MC_YAW_WEIGHT | 0.4 | S | same. `AttitudeControl::setProportionalGain` divides yaw P by it |
| MC_ROLLRATE_MAX / PITCH | 220 °/s = 3.840 rad/s | S | same |
| MC_YAWRATE_MAX | 200 °/s = 3.491 rad/s | S | same |
| MC_ROLLRATE_P / I / D / FF / K | 0.15 / 0.2 / 0.003 / 0 / 1.0 | S | `src/modules/mc_rate_control/mc_rate_control_params.c` |
| MC_PITCHRATE_P / I / D / FF / K | 0.15 / 0.2 / 0.003 / 0 / 1.0 | S | same |
| MC_YAWRATE_P / I / D / FF / K | 0.2 / 0.1 / 0 / 0 / 1.0 | S | same |
| MC_RR / PR / YR _INT_LIM | 0.30 each | S | same |
| MC_YAW_TQ_CUTOFF | 2 Hz, 1-pole `AlphaFilter` on yaw torque | S | same + `MulticopterRateControl.{hpp:133,cpp:223}` |
| MC_BAT_SCALE_EN | 0 (no voltage compensation) | S | same |
| IMU_GYRO_CUTOFF | 40 Hz, `LowPassFilter2p` on the control gyro | S | `src/modules/sensors/vehicle_angular_velocity/imu_gyro_parameters.c` + `VehicleAngularVelocity.hpp:132` |
| IMU_DGYRO_CUTOFF | 30 Hz, 1-pole `AlphaFilter` on the differentiated gyro (D term) | S | same + `.hpp:172`, `.cpp:772-781` (the tap point relative to the 40 Hz LPF is [V]) |
| IMU_GYRO_RATEMAX | 400 default, 800 via `rc.mc_defaults` | S | **we fly 1000 Hz** (Luiz decision 3; 1000 is a legal PX4 value) |
| CA_ROTORi_CT / KM | 6.5 / ±0.05 | S | `control_allocator/module.yaml`, 4001_gz_x500 |
| CA_ROTOR geometry | (0.13, ±0.22) / (−0.13, ±0.20) | S | 4001_gz_x500. **Not the SDF geometry (±0.174)**: PX4's allocation model is deliberately mismatched to the physics |
| CA_METHOD | 2 = auto → sequential desaturation for MC | S/V | `module.yaml`. The selection is in `ControlAllocator.cpp:791-803` [V] |
| THR_MDL_FAC | 0 expected (thrust treated as linear in u) | V | `lib/mixer_module` |

The rate law is `RateControl::update` (`src/lib/rate_control/rate_control.cpp`):
`τ = P·e + I_int − D·ω̇_filt + FF·sp`. The gains are pre-multiplied by K. The integrator:
- is scaled by `i_factor = max(0, 1 − (e/400°/s)²)`;
- is clamped to ±INT_LIM;
- is frozen in the saturated direction using the allocator's unallocated-torque flags.

The attitude law is `AttitudeControl::update`, a reduced/full quaternion split with yaw
weight, `eq = 2·imag(canonical(q⁻¹q_d))` and `rate_sp = P∘eq` clamped to RATE_MAX.

The allocator is `ActuatorEffectivenessRotors::computeEffectivenessMatrix`
(moment = ct·p×a − ct·km·a, thrust = ct·a). Its pseudo-inverse is normalized by
`ControlAllocationPseudoInverse::updateControlAllocationMatrixScale`:
- roll and pitch share one scale `sqrt(‖col‖² / (n_nonzero/2))`;
- yaw is scaled by the column max;
- thrust is scaled by the mean |entry|.

### 1c. Normalized torque → N·m (derived, for teachers' linear models and the CTBR spec)

The port does **not** need a torque constant. Flown, it runs the real nonlinear chain:

> normalized τ → allocator → u ∈ [0,1] → pwm = 0.15 + 0.85u → plant thrust k·pwm².

A linearization is needed only where a linear model or a range is required:
- With the allocator built on the sim's '+'-equivalent geometry (§3 S3, choice A), PX4's
  normalization gives roll/pitch mix entries ±1/√2 on the two active motors and yaw entries ±1
  on all four.
- About hover, dT/du = 2·kf·ω_h·850 = **11.18 N** per unit u [D].
- Roll: 2·(1/√2)·11.18·0.348 = **5.50 N·m per unit normalized torque**. That is 231 rad/s²,
  so MC_ROLLRATE_P 0.15 gives an inner-loop crossover of about 35 rad/s (≈5.5 Hz), a plausible
  value for a 2 kg quad [D].
- Yaw: 4·11.18·0.016 = **0.716 N·m per unit**, 16.3 rad/s² [D].
- Equivalence check [D]: the X-geometry pseudo-inverse with the same normalization gives entries
  ±1/√2 on all four motors at arm r/√2, i.e. the same 2r·(1/√2)·… small-signal gain. **PX4's
  normalization makes the '+'-equivalent and the true X agree in small signal.** They differ
  only at saturation.

### 1d. IMU (Pixhawk 6X)

| Quantity | Value | Tag | Source |
|---|---|---|---|
| IMU part | ICM-42688-P on Pixhawk 6X Rev 3/4 (with ICM-42670-P + ICM-20649/BMI088) | S | docs.px4.io `flight_controller/pixhawk6x`. **Rev 8 (currently shipping) has 3× ICM-45686** |
| gyro rate noise density | 0.0028 °/s/√Hz @10 Hz = 4.887e-5 rad/s/√Hz | S | TDK DS-000347 rev 1.6, Table 1 |
| gyro σ per 1 kHz sample | **1.545e-3 rad/s** | D | density·√(1/dt), the RotorS convention already used for L4 (`training.py _L4_GYRO_SIGMA`) |
| gyro ZRO initial / vs temp | ±0.5 °/s / ±0.005 °/s/°C | S | DS-000347 Table 1. Removed by PX4 boot calibration; the IMU is heater-controlled |
| gyro bias instability | **not published** | A | absent from DS-000347; the TDK forum confirms it is not in the datasheet. See the choice in §9 |
| accel noise density | 65 µg/√Hz (x,y), 70 µg/√Hz (z) | S | DS-000347 Table 2 |
| accel σ per 1 kHz sample | **2.016e-2 m/s² (x,y), 2.171e-2 (z)** | D | density·g·√1000. The sim has one scalar: use 2.171e-2 (conservative) or split per axis [choice] |
| ODR | gyro 12.5 Hz–32 kHz | S | DS-000347 |

For comparison, L4 today uses ADIS16448 at gyro σ 1.073e-2 and accel 1.265e-1. The x500 IMU
is **6.9× quieter (gyro) and 5.8× quieter (accel)**. That is a real plant difference and it is
reported as such, never hidden in the "level".

---

## 2. What changes for cf21: nothing, at every step

Every new field defaults to "off" with the legacy path skipped, not run at unity, following the
`motor_settling_time_s` idiom. Every step ships three things:
- **(i) Pin:** a cf21 flag-off pin, a fixed cf21_brushless L4C l2b-config rollout hash on CPU
  and GPU, bit-identical before and after. It goes in a new `controller/ctrl86_flag_off_pins.rs`,
  alongside the existing `abi38_pins.rs` / `ctrl17_pins.rs` pattern.
- **(ii) Cargo suite:** the full `cargo test -p ram_controller --lib --no-default-features`
  suite, including all CPU/GPU parity sweeps.
- **(iii) Python suite:** the airframe/trainer-pack Python tests
  (`controller_trainer_airframe.py`-style).

**Wheels:**
- Only `ram_controller` is rebuilt (`maturin develop --release -m controller/Cargo.toml`),
  and the rebuild is swap-free.
- No file in `ram_core` or in the worker crate is touched, so the **IDS worker wheel
  (`ram_accelerator`) is untouched** and no worker swap is needed.
- The controller ABI is 64 today (`controller/lib.rs:657`). Each Rust step takes the next free
  number at merge time; the numbers below assume no other merge in between.

---

## 3. Work breakdown (by crate and layer), in merge order

Root for Rust paths: `src/wnn/ram/strategies/accelerator/controller/`. Python:
`src/wnn/control/`.

### S0 — airframe data entry (data-only, no ABI) — 0.5 d

- **Files:**
  - `airframe.py`: new `_X500_SITL = Airframe(name="x500_sitl", …)` with every §1a number and
    its source string, pinned to v1.16.0 / e05f4312.
  - `phased_ga.py`: `AIRFRAME_MOTOR_LAG_S` gets an entry form that can carry (up, down).
    Until S1 lands it is refused: "x500_sitl needs asymmetric lag (ABI ≥ 65)".
- **Flyability:** the airframe is not flyable yet. Every flight path refuses it with "no inner
  loop registered" until S4, so it can never fall back silently to the legacy single loop. This
  is the cf2x_urdf lesson.
- **Tests:** Python unit tests asserting the values below to 1e-6.
  - mass 2.06431;
  - I (0.023839, 0.023942, 0.044000);
  - arm 0.348;
  - k_thrust 8.54858;
  - T/W 1.6885;
  - hover 0.76956;
  - authority 124.8.

### S1 — asymmetric motor lag (plant + teachers, Rust + Metal) — ABI 65 — 2 d

- **Plant, CPU.** `controller.rs::apply_motor_lag` gets a new field `motor_settling_time_down_s`.
  - Value 0 means symmetric: the `down` value is taken from `up`, so behaviour is
    bit-identical.
  - Per motor per step: `T = (cmd > filt) ? T_up : T_down`, then the existing
    `α = 4dt/max(T,4dt)` update.
- **Plant, Metal.** `shaders/controller_rollout.metal` changes at three sites:
  - the struct field, appended at the END in lockstep with `RolloutParams` in
    `metal_controller.rs`;
  - the plant filter at ~L2327-2335;
  - the twins at ~L1180 and ~L2320.
- **Teachers' twin.** `actuator_lag.rs` currently filters **per axis after unmixing** and
  justifies that by linearity. Up/down switching breaks linearity: the comment's equality
  "filter-then-unmix = unmix-then-filter" no longer holds.
  - Change it to filter per motor and then unmix.
  - Keep the per-axis path for the symmetric case so cf21 stays bit-identical.
  - Same audit for `cascade_dhat.rs`, `cascade_obs.rs`, `cpu_score.rs`, `dagger_train.rs`,
    `position_score.rs`, `calib_sampler.rs`, `imitation_probe.rs` and `ctbr_mpcof.rs`: every
    consumer of `motor_settling_time_s` that grep finds (21 Rust/Metal files).
- **Python.** `resolve_motor_lag` returns (T_up, T_down, src). The new CLI flag
  `--motor-lag-down-s` defaults to the airframe value. `provenance.py` stamps both values.
- **Tests:**
  - Hand-computed up/down step: from 0.5, cmd 1.0 for 10 steps at dt 1 ms with T_up 0.05 gives
    0.5 + 0.5·(1−(1−0.08)^10). Then cmd 0.0 gives the down coefficient 0.04.
  - A new CPU/GPU parity sweep case `asym_lag`.
  - The cf21 pin, with `down=0` and with `down=up`, both bit-identical.

### S2 — motor idle floor (plant, Rust + Metal) — ABI 66 (may fold into 65) — 0.5 d

- **Change.** A new plant field `pwm_min`, default 0 = no clamp, bit-identical. For x500 it is
  0.15 [S]. It is applied where the plant consumes pwm, before lag, in CPU and Metal. A raw-pwm
  rival (MPCOF, LQR, LQI) then cannot command a motor below armed idle, which a real ESC refuses.
- **Tests:** parity case `pwm_floor` and the cf21 pin.

### S3 — PX4 cascade port, CPU core (new code) — ABI 67 — 3–4 d

**New files.**
- `controller/pid_px4.rs`:
  - `AttStagePx4`: reduced/full quaternion error, MC_YAW_WEIGHT, the yaw-P/weight
    compensation, RATE_MAX clamp.
  - `RateStagePx4`: P / I with i_factor and saturation freeze / D on the filtered derivative /
    FF, K-premultiplied gains, the yaw-torque AlphaFilter.
  - The gyro `LowPassFilter2p` at IMU_GYRO_CUTOFF and the D-term `AlphaFilter` at
    IMU_DGYRO_CUTOFF.
  - A `rate_loop_hz` parameter, 1000 for x500, so decimation is 1 at dt = 1 ms.
- `controller/px4_alloc.rs`: effectiveness matrix, pseudo-inverse (reuse
  `overactuated.rs::allocation_pinv` with λ = 0 for the 4×4 case, or a local 4×4 inverse),
  PX4's normalization, sequential desaturation (airmode off), and the u→pwm output map
  `0.15 + 0.85u`.
- The PyO3 class `AttitudePidPx4Rs` is registered in `controller/lib.rs`. **No Python twin.**
  The Python per-step loops refuse x500 (house rule "no Python shortcuts"; those loops are the
  attitude-only legacy anyway).

**OPEN CHOICE A vs B: which geometry does the allocator invert?**

The sim plant is '+'-mixed (§0). x500 is an X frame.
- **A (recommended for the first lineage).** Build the allocator on the sim's '+'-equivalent
  geometry (rotors at ±0.348 m on the axes, km = 0.016) through PX4's own
  `computeEffectivenessMatrix` + normalization code path.
  - Small-signal identical to the X (§1c).
  - Consistent with how cf21 (also a real X) is already modelled.
  - Zero change to the plant mixer, the Metal kernel or the teachers' `unmix_plus`.
  - Deviations, each disclosed:
    - (1) CA_ROTOR's mismatched geometry (0.13/0.22) is not reproduced;
    - (2) X shared-motor saturation coupling is absent;
    - (3) ±20 % per-motor wear projects onto axes as '+' instead of X.
- **B (follow-up story).** A true X plant.
  - Generalize `body_torque` to `RotorGeometry` (`overactuated.rs` already has it, with the
    quad-plus oracle).
  - Touches Metal, every teacher's mix/unmix, and `actuator_lag.rs`.
  - Probably 4–6 d.
- **Recommendation:** A now, B as CTRL-86b before a hardware-transfer claim.

**Tests (hand-computed and PX4 reference behaviour):**
- (a) Port `ActuatorEffectivenessRotorsTest.cpp::AllZeroCase` verbatim. The quad-wide geometry
  must give the expected 6×4 matrix exactly, in FRD. Then the same geometry mapped to FLU gives
  the sign-flipped rows (pitch and yaw rows negated).
- (b) Normalization: on the '+' geometry, roll/pitch entries are ±1/√2 and yaw entries ±1.
- (c) Attitude law on hand quaternions:
  - pure 10° roll error → rate_sp = 6.5·2·sin(5°) = 1.1331 rad/s;
  - pure 30° yaw error with yaw weight 0.4 → rate_sp_z = (2.8/0.4)·2·sin(0.4·15°) = 1.4635 rad/s;
  - clamp at 3.840 rad/s.
- (d) Rate law: e = 1 rad/s step, one tick:
  - τ = 0.15 + I·dt;
  - i_factor at e = 200 °/s is exactly 0.75;
  - integrator clamp 0.30;
  - the saturation flag freezes growth in one direction only.
- (e) Filters: the LowPassFilter2p coefficients at fs 1000 / fc 40 against PX4's formula. The
  existing `Lpf2p` (port of crazyflie `filter.c`) is expected to be the same Butterworth [V] —
  the test decides whether it is reused. AlphaFilter α at fc 30 Hz / 2 Hz.
- (f) At-rest identity: with zero error the cascade outputs hover u on all motors and pwm 0.76956.
- (g) A closed-loop sanity check on the clean x500 plant (L4A, no noise): 10° roll step, no
  limit cycle (tail swing < 0.1°), settles. The numbers are recorded as the PX4-port
  characteristic, not as a gate.

**Optional S3-ref (2–3 d, needs a PX4 toolchain, NOT on this machine during a flight).** Record a
PX4 SITL gz x500 rate-step log at v1.16.0 and replay its setpoints through the port. This is the
strongest fidelity test; it is not a blocker.

### S4 — wire the PX4 cascade as PID teacher / rival and as the CTBR inner loop (CPU) — ABI 68 — 2–3 d

- **Rust:**
  - `optimal.rs` Teacher enum: a `Px4Cascade` arm next to the `PidFirmware` arm (~L2061).
  - `ctbr.rs`: the inner rate stage becomes `enum InnerRate { Crazyflie(RateStageFw), Px4(RateStagePx4+alloc) }`.
  - `dagger_train.rs`, `cpu_score.rs`, `position_score.rs`, `calib_sampler.rs`,
    `record_ops.rs`: accept the new `af_px4_*` kwargs.
- **Python:**
  - `airframe.py`: a `Px4Gains` dataclass bound to `x500_sitl` (the same `airframe=` guard as
    `PidGains`).
  - `training.py`: `_pid_cascade_kwargs` dispatches on the gains type; `cascade_kwargs()`
    likewise.
  - `evaluator.py::ctbr_spec_dict`: the refusal text "e.g. --airframe cf21_brushless" becomes
    "an airframe with a registered inner loop (cf21_brushless, cf2x_firmware, x500_sitl)".
  - `phased_ga.py`: the cascade help text (~L3796) is generalized the same way.
  - The cf2x_urdf refusal in `_pid_cascade_kwargs` is kept.
- **Pack-site audit:** a test asserting the x500 plant, lag, idle and PX4 kwargs reach EVERY
  config-pack site:
  - evaluator `_plant_train_kwargs` (both trainer sites);
  - classical_baseline;
  - ga_memory recorder;
  - calib sampler;
  - the thermometer-calibration sims;
  - rival/rescore scripts via `episode_config_from_args`.

  This is the 19/09 trainer-airframe-gap class of bug.
- **Tests:**
  - PID teacher label replay: a CTBR setpoint stream equal to the PX4 teacher's labels
    reproduces the PX4 PID teacher bit for bit (same identity as abi38_tests).
  - The cf21 pin.

### S5 — per-airframe presets (Python data, no ABI unless noted) — 1 d

- **CTBR rate range:**
  - `CTBR_DEFAULT_RATE_MAX` becomes a per-airframe dict.
  - cf21 keeps (4.2, 4.2, 2.0).
  - x500 = (3.840, 3.840, 3.491) [S] from MC_*RATE_MAX. A label beyond PX4's own clamp can
    never be produced by the PX4 teacher.
  - γ stays at the default.
- **IMU preset:** bound to the airframe.
  - The L4 rungs keep their plant-uncertainty axis.
  - Sensor σ comes from `airframe.imu`: cf21 = ADIS16448 (unchanged, bit-identical); x500 =
    ICM-42688-P (§1d).
  - `DisturbanceConfig.preset` takes the airframe. `provenance.py` stamps the IMU part.
- **Rotor drag default:** `DEFAULT_ROTOR_DRAG` becomes per-airframe. cf21 0.78 [S, Folk 2026];
  x500 0.120 [D].
- **Altitude PD bound and collective range:**
  - hover 0.770 leaves +0.230 / −0.620 pwm of headroom (−0.620 is the drop to the 0.15 floor).
  - `alt_pd_max_delta` 0.25 would saturate upward.
  - Per-airframe value: x500 0.20 [A]. That keeps the upward motor headroom for attitude
    torque at ≥0.03 even at max collective. PX4's sequential desaturation then decides the
    rest. Flagged.
  - The PD gains themselves re-derive from the plant (`AltitudePd::from_plant`), and so does the
    position loop (gravity only).
- **Episode default:** see §5.

### S6 — teacher re-derivation audit (Rust tests; code only if needed) — 1 d

- **Re-derivation:** LQR/LQI/MPC/MPCOF re-derive from the airframe automatically through
  `attitude_linear_model` / `attitude_lag_model*` in `optimal.rs`.
  - x500 B per axis ≈ 192 rad/s² per unit u [D], against cf21's 654.
  - The DARE inputs are well within the range the solver already handles.
- **Tests:**
  - At x500 (dt 1 ms, lag on/off, integral on/off) assert: DARE residual below 1e-9 relative;
    all closed-loop eigenvalues of (A−BK) strictly inside the unit circle with margin; `P`
    symmetric positive-definite; condition number logged and bounded.
  - MPC box-QP feasible at hover with the 0.15 floor.
- **The one design decision: the asymmetric lag in a linear model.**
  - The linear teacher models need ONE τ. Proposal [A]: design at T_eff = (T_up+T_down)/2 = 0.075 s.
  - The plant flies the true switching filter, and the per-motor twin (S1) tracks it exactly
    for the observers.
  - Pinned by a test; open for Luiz to override (alternatives: T_down, conservative).

### S7 — Metal port of the PX4 cascade (GPU CTBR inner loop + GPU PID rival) — ABI 69 — 3–4 d

- **Shader:** `shaders/controller_rollout.metal` gets a `px4_*` uniform block next to
  `pidfw_*` (~L222), an inner-stage switch in the CTBR path (~L277-292), and an allocator plus
  sequential desaturation in f32.
- **Host:** `metal_controller.rs` packs it, appended at the END of both structs in lockstep.
- **Tests:** new parity sweeps:
  - `px4_ctbr_rollout`: CPU `cpu_score` vs GPU, bit-exact on the established tolerance policy
    of the existing 14 suites;
  - `px4_pid_rival`.
  - Plus the cf21 pin.
- **Until S7 lands:** x500 scoring is CPU-only. The GPU path refuses x500 loudly.

### S8 — battery sag at 4S (plant, Rust + Metal) — ABI 70 — 2 d

The model is in §7. Plant fields: `bat_cells`, `bat_v_ref`, `bat_r_ohm`, `bat_i_hover`, and the
per-episode `bat_soc`. 0 cells means off and is bit-identical. Thrust per motor =
`k·s(t)·pwm²`, where `s = (V_term/V_ref)²`, computed each step on CPU and Metal. The SOC draw
comes from the episode counter RNG; the per-unit R spread is resolved once per scoring pass,
like motor wear. Tests:
- hand-computed s at SOC 1.0 / 0.6 / 0.2: 1.159 / 1.057 / 0.959;
- parity case `bat_sag`;
- cf21 pin.

### S9 — position-sensing model (plant/obs, Rust + Metal) — ABI 71 — 2–3 d, blocked on Luiz's choice (§6)

- **Mechanism.** The measured position/velocity that feeds `--obs-pos-err-xy`, `--obs-vel-xy`,
  `--obs-alt-err` and `--obs-vz` becomes:
  - a sample-and-hold at `f_pos`;
  - a delay ring of `lat_pos` (the existing D6 ring idiom; it needs > 8 slots at 1 kHz for
    110 ms, so a separate ring);
  - white noise σ_h / σ_v / σ_vel.
- **Defaults and tests.** All 0 means truth, bit-identical. Tests: hold/latency hand traces,
  parity case `pos_sensor`, cf21 pin.
- **Observability.**
  - Velocity comes from the receiver's own Doppler velocity, a raw sensor output. No EKF, no
    observer.
  - Altitude comes from GNSS vertical (or mocap z). Baro + EKF fusion would be an observer and
    is out of scope.
  - With 10 Hz + 110 ms, the 100 Hz WNN sees position 11–21 ticks stale. That is the physical
    point of the model. The fast attitude loop is unaffected because it rides on the gyro.
- **Teacher vs rival rule (proposed, Luiz to confirm).** DAgger teachers may label from truth
  (a privileged teacher, disclosed). Rivals scored for the comparison read the same measured
  signals as the WNN.

---

## 4. Step order and gates

| # | Step | Kind | ABI | Effort | Gate after the step |
|---|---|---|---|---|---|
| S0 | airframe data | data | — | 0.5 d | Python unit tests |
| S1 | asymmetric lag | code, Rust+Metal | 65 | 2 d | parity + cf21 pin |
| S2 | idle floor | code, Rust+Metal | 66 | 0.5 d | parity + cf21 pin |
| S3 | PX4 cascade core | code, Rust | 67 | 3–4 d | PX4 ref + hand tests |
| S4 | wire teacher/rival/CTBR | code, Rust+Py | 68 | 2–3 d | pack-site audit, label replay |
| S5 | per-airframe presets | data, Py | — | 1 d | Python tests, cf21 pin |
| S6 | teacher audit | tests (+small) | (68) | 1 d | DARE/eig tests |
| — | **G0/G1 rival gate, CPU, SITL plant** | run (controller agent) | — | ~0.5 d compute | §5 G0, G1 |
| S7 | Metal PX4 cascade | code, Metal | 69 | 3–4 d | 2 new parity sweeps |
| S8 | battery sag | code, Rust+Metal | 70 | 2 d | parity + cf21 pin |
| S9 | position sensing | code, Rust+Metal | 71 | 2–3 d | parity + cf21 pin |
| — | **G2 rival gate, full plant** | run | — | ~0.5 d | §5 G2 |
| L | x500 lineage: smoke → rivals → n=5 | runs | — | see §5 | prereg |

- **Total code:** about 17–21 agent-days, plus 4–6 d if option B (true X plant) is taken.
- **Ordering rule:** S3 precedes S7 so that the CPU port is the oracle the GPU must match.
- **Parallel work:** S8 and S9 are independent of S7 and can run in parallel with it.
- **Why G1 comes before battery and positioning:** a PID failure on the SITL plant is a port or
  plant bug, attributable before more physics is layered on.

---

## 5. Prereg — first x500 lineage (CTRL-86 L1)

### 5a. Recipe (derived from the flown l2b argv)

Base: `[argv]` of
`logs/controller/ctrl48/C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s1.out`.
That cf21 run scored TEST over 35 units: stable 100.0 %, err 0.97°, steady 0.26°, pos 0.342 m.
It is the cf21 reference only, never compared in one mean with x500.

Changes, and nothing else:

| l2b token | x500 value | Why |
|---|---|---|
| `--airframe cf21_brushless` | `--airframe x500_sitl` | the story |
| `--motor-lag-s 0.20` | **remove** (airframe-bound 0.05 up / 0.10 down) | 0.20 is cf21's Grafe 2026 value; passed explicitly it would override |
| `--rotor-drag 0.78` | **remove** (airframe default 0.120) | 0.78 is cf21's Folk 2026 value; same trap |
| `--steps 2000` | `--steps N_x500` (§5c) | episode scale |
| CTBR rate range (default) | airframe default (3.840, 3.840, 3.491) | MC_*RATE_MAX |
| IMU (via L4C) | airframe-bound ICM-42688-P | Luiz decision 2 |
| (none) | battery 4S, position sensor per §6 | from S8/S9; G1 runs without them |
| `--save-*` paths | `logs/controller/ctrl86/…` | |
| marker name | `X5-l2b_a<ABI>_s{1..5}` | airframe and ABI in the name |

Kept verbatim, everything else, including:
- `--disturbance L4C` (±20 % motor wear and torque jitter, the same Molchanov axis);
- `--teacher mpcof --teacher-hover derived`;
- `--output-mode ctbr --action-repeat 10`;
- the delta/yaw/gyro ladders, the receptive fields, `--calib-episodes 200`, all fitness weights,
  `--gate-stable 0.70 --gate-err 8.0`;
- `--tilt 5.0 --xy-offset 0.5`;
- `--num-eval-folds 5`;
- `--val-units 20 --select-tail p99`;
- the 35 `--test-seeds 99990101…135`;
- `--base-seed 31337001…005`.

Grep checklist before launch: no token in the final argv may carry a cf21-specific number.

### 5b. Recalibrated per airframe (automatically, or by S5 data)

- **Automatic:**
  - the plant (mass, I, arm, k, k_drag) and the derived hover;
  - the altitude-PD and position-loop gains (from the plant);
  - LQR/LQI/MPC/MPCOF (DARE on the x500 model);
  - the WNN thermometer and ladders, re-fit from x500 calibration rollouts (`--calib-episodes 200`);
  - the PX4 PID cascade (sourced, never re-derived).
- **By S5 data:** CTBR rate ranges, IMU σ, rotor drag, motor lag (up/down), idle floor,
  collective bound.
- **Not recalibrated, on purpose:** fitness weights, gates, the delta/yaw ladder shapes, task
  geometry (tilt 5°, xy offset 0.5 m), and the disturbance ladder.

### 5c. Episode length

- **Rule (prereg):** keep the number of closed-loop response times per episode equal to cf21's.
  Measure, on the clean plant (L4A, 35 units, PID rival = each airframe's firmware cascade), the
  time for the mean tilt error to fall below 10 % of its initial 5° value: `t10_cf21` and
  `t10_x500`. Then set `N_x500 = 2000·t10_x500/t10_cf21`, rounded up to the next 500 steps.
- **Expectation [A]:**
  - angular authority is 3.8× lower and large-signal time scales roughly with √(1/α), so about
    1.9× longer;
  - **N ≈ 4000** (4 s);
  - the position loop is gravity-only (ωn 1.0), so position convergence is airframe-independent
    once attitude tracks.
- **Fixed before the smoke run:** N is measured at G0 and frozen in this file before the smoke
  run. Steady and pos-hold windows are fractions (`steady_window_frac` 0.20), so they scale with
  N without edits.
- **Cost:** about 2× per generation. The cf21 l2b s1 stage wall was 63 + 1042 + 478 s.

### 5d. Gates — "the airframe works" before any WNN training

- **G0, physics sanity (deterministic, L4A, noise off, SITL plant):**
  - Zero-command hover holds: |att| < 1e-4 rad, |z drift| < 1 mm over 4 s.
  - The PX4 cascade 10° roll step has no limit cycle (tail swing < 0.1°).
  - Measure `t10` for §5c.
- **G1, rivals on the SITL plant** (L4C, 35 units × 100 eps, CPU; no sag, truth position):
  - **PX4-PID stable ≥ 90 %** (cf21's firmware PID was 91.1 % over 35 units).
  - **MPCOF stable ≥ 95 %** (cf21 100 %).
  - LQI, LQR and MPC reported, not gated.
  - **If PID < 90 %:** stop and audit the port. PX4 defaults fly this vehicle in SITL, so a
    failure means our port or plant is wrong, not the vehicle.
  - **If MPCOF < 95 %:** stop and audit the teacher model (S6).
- **G2, rivals on the full plant** (+ battery sag + position sensor):
  - MPCOF ≥ 95 % required, because it is the DAgger teacher. A teacher that cannot hold the
    plant cannot teach it.
  - PID reported with its delta vs G1 attributed to sag and positioning.
  - Proposed thresholds, for Luiz to confirm or change.

### 5e. Sequence

1. **G0 → G1 → G2** (rivals only, no WNN).
2. **Smoke:** 1 seed (`s1`), full recipe, reduced budget (`--memory-gens 4 --neurons-gens 3`).
   Pass = rc 0, every stage line printed, the TEST block printed for 35 units, the provenance
   line names x500_sitl / ICM-42688-P / lag 0.05/0.10 / 4S / the positioning model.
3. **Rivals on the same units:** MPCOF, PX4-PID, LQI on the exact 35 TEST units and seeds the
   WNN is scored on. These rows form the comparison table.
4. **n = 5:** seeds s1–s5, full budget. Headline = TEST over 35 x500 units, reported as the
   whole vector (stable, err, steady, pos, fails).
   - No adoption claim; this is a new-airframe baseline.
   - The significance read goes to `experiment-design`; run mechanics to `controller`.

### 5f. Reporting, per airframe

- **Separation:**
  - Every table carries an `airframe` column.
  - x500 rows and cf21 rows are never averaged together, and never share a Δ column.
  - A cross-airframe view is side-by-side columns labelled by airframe.
  - Per-seed tables use the standard `ctrl-seed-table` format on `logs/controller/ctrl86/runs.log`.
  - MPCOF / PID / LQI reference rows come from the SAME x500 units.
- **Meaning of a unit:** on x500 a TEST "unit" is one x500 with its own ±20 % motor wear and,
  from S8, its own battery resistance. That is stated in every caption.
- **Disclosure line, in every report and the paper:**
  - PX4 v1.16.0 SITL plant;
  - '+'-equivalent mixing (choice A);
  - IMU = ICM-42688-P datasheet;
  - gyro bias walk assumption;
  - battery model assumptions;
  - positioning model.

---

## 6. Positioning system (Luiz to choose)

Requirement: a 500 mm, 2 kg vehicle, outdoors or in a large hall, on a 0.5 m position task. To
stay small next to the task, sensor error should be ≤ 0.05 m (10 % of 0.5 m).

| Option | Accuracy | Rate | Latency | Fits? |
|---|---|---|---|---|
| **RTK GNSS, u-blox ZED-F9P** (Holybro H-RTK F9P class) | H 0.01 m + 1 ppm CEP; V 0.01 m + 1 ppm median; vel 0.05 m/s [S: UBX-21044850 R05, Tables 1, 3] | RTK 7 Hz (GPS+GLO+GAL+BDS) … 20 Hz (GPS only); **10 Hz for GPS+GLO+GAL** [S, Table 2] | **110 ms**, the PX4 `EKF2_GPS_DELAY` default [S: ekf2/params_gnss.yaml]. A tuning default, not a measurement [A] | **Yes, outdoors.** Recommended. |
| Motion capture (OptiTrack/Vicon class) | sub-mm | 100–360 Hz | ~5 ms | Yes, large hall only. Volume-limited, highest fidelity. Vendor numbers not pinned [V] |
| UWB (Qorvo DWM1001 class) | ~0.1 m | ~10 Hz | tens of ms | Marginal: 20 % of the task. Not recommended. Numbers not pinned [V] |
| Stock X500 V2 kit GPS (M8N) | metre-level | — | — | **No.** The Holybro kit ships M8N only [S: Holybro X500 V2 page] |

**Recommendation: RTK GNSS (ZED-F9P), modelled as follows.**
- σ_h = 0.01/1.1774 = **8.5 mm** per axis [D: 2-D Gaussian CEP → σ].
- σ_v = 0.01/0.6745 = **14.8 mm** [D: median → σ].
- σ_vel = 0.05 m/s per axis [S; 1σ assumed, A].
- 10 Hz sample-and-hold, 110 ms latency.
- The 1 ppm term is ignored at sub-km baselines [D: 1 mm at 1 km].
- **Known simplification [A]:** RTK error is time-correlated (multipath), not white. The first
  model is white at 10 Hz. A Gauss-Markov term can follow when measured data exist (2028).
- Choose mocap instead if the 2028 flights will be indoors. The model has the same shape with
  different numbers (fast, low-latency).

---

## 7. Battery: 4S with sag (6S open)

- **Why 4S.** Holybro's X500 V2 kit specifies "4S 2000–5000 mAh LiPo", 2216 KV920 motors and
  BLHeli S 20 A ESCs "compatible with 4S" [S: Holybro X500 V2 page].
- **What 6S would mean.** 6S is outside the stock power train's rating, so it means different
  motors or ESCs: a new kf and ω_max, i.e. a different airframe entry, not a battery parameter.
  Recommendation: 4S; 6S becomes a separate airframe if ever wanted.
- **Consistency check [D/A].** KV920 × 14.8 V = 1426 rad/s no-load. The SITL ω_max of 1000 rad/s
  is 70 % of that, plausible under prop load at nominal voltage. This is why V_ref = 14.8 V
  (4 × 3.7 V) is taken as the voltage the SITL k_thrust corresponds to. That choice is an
  assumption.

**Minimal model:**
- **Open-circuit voltage.** OCV_cell(SOC) = 3.6 + 0.45·SOC V. These are PX4's own battery model
  endpoints, `BAT_V_EMPTY` 3.6 / `BAT_V_CHARGED` 4.05 [S: src/lib/battery/module.yaml]. Using
  PX4's estimator line as the plant curve assumes a linear mid-range [A]. A measured OCV curve
  replaces it in 2028.
- **Terminal voltage.** V_term = 4·OCV(SOC) − I·R_pack.
- **Current.** I = I_h·Σpwm³/(4·pwm_h³). Rotor aero power ∝ ω³ (momentum theory) [D].
  - I_h = 5.0 Ah × 0.8 usable / 0.30 h = **13.3 A** [D from the vendor's "~18 min hover, tested
    with 5000 mAh"; the 80 % usable fraction is A].
- **R_pack = 0.020 Ω** nominal [A: 4 cells at about 4 mΩ plus wiring]. PX4's `BAT_R_INTERNAL`
  defaults to −1 (auto), so there is no source. Measure it in 2028.
- **Thrust.** ω = pwm·ω_max·V_term/V_ref, so the thrust scale is s = (V_term/V_ref)², applied to
  k_thrust per step. PX4 does not compensate by default (`MC_BAT_SCALE_EN 0` [S]); the PX4 rival
  flies uncompensated, faithfully.
- **Magnitudes [D]:**

  | SOC | s | hover pwm |
  |---|---|---|
  | 1.0 | 1.159 | 0.715 |
  | 0.6 | 1.057 | 0.749 |
  | 0.2 | 0.959 | 0.786 |

  That is a thrust spread of about 20 % across SOC, comparable to L4B motor wear: not small.
- **Per-unit spread and SOC draw:**
  - SOC is drawn U[0.2, 1.0] per episode, in training and in scoring.
  - R_pack per TEST unit is drawn ±30 % around nominal [A], resolved once per scoring pass like
    motor wear.
  - SOC is held constant within an episode [D: 13 A × 4 s ≈ 15 mAh = 0.3 % of 5000 mAh].
- **Teachers.** Teachers derive at s = 1 (V_ref). The sag appears to them as an unmodelled thrust
  gain. MPCOF's offset-free action absorbs part of it, which is fine for a rival.
- **Observability.** The WNN sees sag only indirectly (through the collective/altitude
  features), the way it sees motor wear. An optional raw `battery voltage` observation is a raw
  sensor, so it is model-free-admissible. It is a later lever, not part of L1.

---

## 8. Risks

1. **'+'-equivalent mixer (choice A).** Shared-motor saturation and the X projection of motor
   wear are absent. Small signal is exact (§1c); saturation is not. This matters most at
   max-collective + max-torque, which x500's 0.23 upward headroom makes more frequent than on
   cf21. Mitigation: disclose it; plan B.
2. **Port fidelity without a SITL log.** The unit tests pin the algebra, not the integrated
   behaviour. Mitigation: the G1 PID ≥ 90 % gate, plus optional S3-ref.
3. **Collective headroom.** Hover 0.77 (up to 0.79 at low SOC) leaves ≤ 0.23 pwm upward.
   Altitude and attitude compete for it. Expect more saturation-driven failures than on cf21.
   The alt-PD bound 0.20 is an assumption.
4. **Trainer/pack-site gap recurring** (19/09 class). Mitigation: the S4 audit test enumerates
   every site.
5. **Asymmetric lag breaks `actuator_lag.rs`'s linearity argument.** S1 moves the twin to
   per-motor filtering. A missed consumer would make MPCOF read the lag as a disturbance.
6. **IMU:**
   - The Pixhawk 6X shipping today (Rev 8) uses ICM-45686, not ICM-42688-P.
   - ICM-42688-P bias instability is unpublished.
   - The RotorS σ = density·√f_s convention at 1 kHz is about √2 conservative against PX4's
     FIFO-averaged 8 kHz → 1 kHz pipeline [D]. It is kept for consistency with L4.
7. **Lag discretization.** The gz filter is exponential, ours is Euler: dt/τ = 0.08 vs
   1−e^(−0.08) = 0.0769 at τ_up, about 4 % on effective τ [D]. We keep Euler (same as cf21) and
   disclose it.
8. **Positioning.** The 110 ms latency is PX4's tuning default. RTK noise is coloured. Truth
   altitude is replaced by GNSS vertical, which is noisier than z truth by 15 mm σ.
9. **Rotor drag.** gz drag is ∝ ω·v per rotor, ours is a constant linear k_d at hover. The error
   is second order at the 0.5 m task's speeds.
10. **Compute.** About 2× steps per episode means about 2× wall per generation and more GPU
    memory per rollout chunk (`--eval-mem-budget auto` must be re-read at the smoke run).
11. **Thermometer contamination.** The WNN encoder is fit from rollouts. A wrong PX4 port
    contaminates the encoder, not just the comparator. That is why G1 precedes any WNN run.

---

## 9. Open choices for Luiz

1. **Positioning:** RTK GNSS ZED-F9P at 10 Hz / 110 ms (recommended, outdoors), or mocap (large
   hall). UWB and the stock M8N are not adequate.
2. **4S vs 6S:** 4S recommended; it is the kit's rated power train. 6S = a different airframe
   (motors or ESCs), so it gets its own entry if wanted.
3. **Mixer geometry:** A ('+'-equivalent, recommended now) vs B (true X plant, +4–6 d), before
   any hardware-transfer claim.
4. **Gyro bias walk for ICM-42688-P** (unpublished): keep L4's ADIS walk 3.8785e-5 (recommended,
   conservative, labelled A), set it to 0 (post-calibration, heated IMU), or wait for the 2028
   Allan measurement.
5. **IMU part:** ICM-42688-P (Rev 3/4, as decided) vs ICM-45686 (Rev 8, what ships now).
6. **Design τ for the linear teachers** under asymmetric lag: the mean 0.075 s (proposed) or
   T_down 0.10 s.
7. **Truth vs measured position for DAgger teachers:** privileged teacher on truth (proposed),
   rivals on measured.
8. **G1/G2 thresholds** (PID ≥ 90 %, MPCOF ≥ 95 %) and the episode-length rule (§5c).
9. **Accel σ:** one scalar at the z figure (conservative) or split per axis (needs a small plant
   change).
