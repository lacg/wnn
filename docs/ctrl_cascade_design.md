# CTRL cascade: WNN outer attitude loop over a classical rate loop (CTBR)

> **DECISIONS 06/11/2026 (Luiz)** — supersede the defaults below where they differ:
> 1. Inner loop = the firmware rate PID (as drafted).
> 2. **Outer-loop TEACHER = MPCOF** (its attitude decision expressed as rate setpoints) — the default,
>    not an ablation. The firmware attitude stage becomes the C2 rival / ablation teacher.
> 3. Build starts as soon as the motor-lag diagnosis (CTRL-26) reports, whatever its verdict. Plane: CTRL-27.
> 4. Next step after v1 flies: Neural-Fly-style disturbance learner on top of the cascade (CTRL-28).
> 5. **sn > 0 IS IN v1 SCOPE** (Luiz 06/11: "not in a hurry … take your time to not leave some tech
>    debt"). The EDRA-BPTT walk (CPU + GPU-resident, ABI 35), the split trainer and every sn>0 path must
>    support the CTBR banks (collective + p, q, r) with the same CPU/GPU parity and bit-exact tests as the
>    motor-bank path. R9 below ("sn>0 unsupported in v1") is SUPERSEDED. No `unsupported in v1` asserts.
> 6. **Train = score plant (CTRL-32, part of v1):** the trainer draws the motor-asymmetry δ per training
>    episode with the scorer's distribution (L4C ±20%), CPU + Metal, every trainer path, motor-level AND
>    cascade. Own commit(s) before the cascade commits; provenance `train_asym=per_episode`.

Status: DESIGN ONLY. No code is changed by this document and no run is launched.
Approved for exploration by Luiz on 06/11/2026. Author: flight-dynamics agent.
Code references are to `src/wnn/ram/strategies/accelerator/controller/` unless stated.

---

## 0. Why: the measurement that motivates this

On the cf21_brushless airframe, the real rotor lag is first order with τ = 50 ms
(Gräfe et al. 2026; `--motor-lag-s 0.20`, which is Molchanov's settling time T = 4τ;
the sim realises τ_eff ≈ 49.5 ms, `actuator_lag.rs::motor_lag_alpha`). Measured on
06/11/2026, grid stage, pop 50, seed 31337002, `_op31` recipe (`logs/controller/op31_lag/runs.log`):

| arm | lag | output law | stable | err | steady |
|---|---|---|---|---|---|
| R0 | 0 | leak 0.95 / dmax 0.1 | 96.6% | 1.60° | 1.01° |
| R1 | 0.20 | leak 0.95 / dmax 0.1 | 0.0% | 33.69° | 52.00° |
| R2 | 0.20 | leak 0.7 / dmax 0.6 | 70.0% | 4.55° | 5.22° |

The PID (firmware cascade) and MPCOF teachers fly the same lag at 100% stable.
`logs/controller/abi36_fix/` holds the output-law screens at pop 6 (leak 0..0.9, dmax
0.05..0.8). None of them beat about 61-65% stable at lag 0.20.

**Mechanism.** The student closes the FASTEST loop of the vehicle. Body-rate damping is
a motor-level loop, and the student runs it through a quantised, leaky delta
accumulator with no model of the actuator. With a 50 ms pole inside that loop, the
learned rate damping runs out of phase. A faster output law (R2) buys back phase by
reacting faster, but it is a patch on the wrong layer. The classical teachers survive
because their rate loop is a linear compensator with phase lead (the rate-D term) and
an integrator that the lag does not destabilise (section 1.4).

**Proposal.** Hand the fast loop back to classical control and keep the WNN at the
attitude level. This is the standard autopilot split and the "collective thrust + body
rates" (CTBR) action abstraction used in learned-quadrotor work.

---

## 1. Architecture

### 1.1 Diagram

```
                 (unchanged, classical, disclosed)
 pos err, vel ──► PositionLoop (position_loop.rs) ──► tilt ref (roll_ref, pitch_ref)
 alt err, vz  ──► [teacher only] AltitudePd ──► collective δ           │
                                                                        ▼
 ┌──────────────────────── OUTER LOOP: WNN (sn=0 RAM), 100 Hz ───────────────────────┐
 │ obs: gyro(3) accel(3) target(3) roll/pitch err(2) roll/pitch ∫(2) yaw err, ∫ (2)  │
 │      collective_cmd, alt_err, vz (3)  [+ pos_err_xy(2), vel_xy(2)]                │
 │ out: 4 banks = [ T_coll , p_sp , q_sp , r_sp ]   (ABSOLUTE decode, held 10 ticks) │
 └────────────────────────────────────┬──────────────────────────────────────────────┘
                                      │ body-frame rate setpoint (rad/s) + collective (pwm)
 ┌──────────────── INNER LOOP: firmware rate PID (pid_firmware.rs), 500 Hz ──────────┐
 │ e_ω = ω_sp − gyro (BODY frame, measured gyro incl. noise + bias walk)            │
 │ per-axis PID (kp, ki, kd + 30 Hz lpf2p on D) → axis force (N) → '+' mixer        │
 │ → per-motor thrust → pwm = sqrt(t / k_thrust), + uniform collective offset       │
 └────────────────────────────────────┬──────────────────────────────────────────────┘
                                      ▼ 4 pwm, 1 kHz plant tick
                     motor lag (τ = 50 ms) ──► AttitudeSim rigid body (controller.rs)
```

The loop is closed through the plant and the gyro. The inner loop reads the same IMU gyro the
student sees, as the real firmware does.

### 1.2 Outer-loop inputs: what stays, what goes

Feature layout is `controller.rs::compute_features`, mirrored by
`evaluator.py::ControllerSpec.num_features`.

| feature | today (`_op31`) | cascade | reason |
|---|---|---|---|
| gyro p,q,r (body) | yes | KEEP | the yaw attitude D-term (kd 0.35) and the outer rate shaping need it |
| accel (specific force, body) | yes | KEEP | the gravity-referenced roll/pitch source |
| target r,p,y | yes | KEEP | unchanged |
| peraxis_p roll/pitch err | yes | KEEP | the attitude-P label is `kp·err`, so this is the dominant input |
| peraxis_i roll/pitch | yes | KEEP, retune leak (section 6, R4) | the attitude-I label needs it |
| yaw_err, yaw_err_i | yes | KEEP | the yaw label; dead-reckoned, same as today |
| obs_pwm (delivered actuator, ABI 36) | yes (4) | **DROP** | it existed for two reasons: the hidden delta accumulator and the lagged actuator. Absolute decode removes the first and the inner loop owns the second. The outer loop's action no longer depends on motor state. |
| d̂ observer (3) | off | stays OFF | it estimates a torque disturbance, which is now the inner integrator's job |
| collective_cmd, alt_err, vz | yes | KEEP | the thrust bank's label is the altitude PD, a function of alt_err and vz |
| pos_err_xy, vel_xy | ABI-37 recipe | KEEP when `--translation` + horizontal | unchanged |

Feature count: 22 → 18 for the `_op31` set, and 26 → 22 for the ABI-37 recipe with pos/vel.
**Hard rule 2:** this changes `prefix_factor`. Every consumer of `num_features` must be
audited together. The list is in section 5.

### 1.3 Outer-loop outputs: ranges and levels

All outputs are in the BODY frame, the same frame as the gyro.

| bank | quantity | range | neutral (untrained) | source of the range |
|---|---|---|---|---|
| 0 | collective, pwm | anchor ± `alt_pd_max_delta` | `collective_anchor` (episode hover) | the altitude PD clamp (`altitude_pd.rs`) |
| 1,2 | p_sp, q_sp (rad/s) | ±4.2 | 0 | attitude kp 6 × 30° (0.52 rad, the curriculum ceiling) = 3.1, plus ki 3 × i_limit 20°·s (0.35 rad·s) = 1.05 |
| 3 | r_sp (rad/s) | ±1.6 (provisional) | 0 | set from the smoke's label histogram (section 6, R6) |

- **Decode.** Use the existing piecewise-linear/γ map (`decoded_to_delta`, here with
  `delta_max` meaning "range"). The neutral sits at the untrained decode point, as today.
- **Levels.** With 256 output neurons there are 64 levels per bank (4 banks), the grid winner's
  `levels=64`. With **γ = 2** the smallest rate step is (1/32)²·4.2 = 0.0041 rad/s. Through
  attitude kp 6 that is a 0.04° attitude dead band. With γ = 1 it would be 0.13 rad/s, which is
  a **1.25° dead band**, the same size as the steady-error budget. So γ = 2 is the default for
  the rate banks, and the thrust bank keeps γ = 1.
- **No saturation surprises.** Rate setpoints beyond the range are clipped at label time, as
  the decode does today for out-of-range teacher pwm.

### 1.4 Loop rates

- **Inner: 500 Hz**, held across the 1 kHz plant tick. This is the firmware's `ATTITUDE_RATE`
  and the existing `pidfw` decimation 2. It is not tunable: it is the firmware's own rate.
- **Outer: 100 Hz default** (`--action-repeat 10`). Ablate at 250 and 500 Hz.

Justification. The numbers come from a linearised loop: firmware gains converted to SI exactly as
`_SiGains.from_firmware` does, I_xx = 3.00e-5 and I_zz = 5.30e-5 kg·m² (from `airframe.py`,
DERIVED not sourced), the 30 Hz lpf2p on rate-D, and 1.5 ms of hold delay:

| loop (roll/pitch) | τ_motor | crossover | phase margin | closed-loop BW | peak |T| |
|---|---|---|---|---|---|
| firmware rate PID | 0 | 216 rad/s | 64° | 355 rad/s | 1.02 |
| firmware rate PID | 49.5 ms | **44 rad/s** | **43°** | **76 rad/s** | 1.42 |
| yaw rate PID | 49.5 ms | 8.3 rad/s | 66° | 13 rad/s | 1.02 |

The attitude loop (kp 6 s⁻¹) crosses over at about 6 rad/s. At 6 rad/s the closed inner loop
under full lag has |T| = 1.04 and a phase of **−3.7°** (−4.0° with no lag). **To the outer loop,
the 50 ms rotor lag is essentially invisible.** That is the whole point of the cascade. A
100 Hz zero-order hold adds 5 ms of delay, which is 1.7° at 6 rad/s, so 100 Hz is ~100× the
outer bandwidth. Choosing 100 Hz rather than 500 Hz has three advantages:
- it is 10× fewer RAM lookups per second on the MCU/FPGA;
- each decision's consequence is larger than the gyro noise floor, so labels are less
  aliased by noise;
- `action_repeat` already implements it: `compute_features` keeps ticking physical-time
  integrators on hold steps (`controller.rs` ~L3198 and L4985).

The cost is 10× fewer DAgger records per episode (section 6, R7).

Yaw caveat: this is a firmware property, and it is disclosed. The yaw rate loop is slow (BW
~9-13 rad/s) relative to yaw attitude kp 6, a separation of only ~1.5-2×. Yaw is also
unobservable in absolute terms (no magnetometer). Expect yaw to remain the weakest channel, as it is today.

### 1.5 Inner-loop gains: firmware, never re-derived

These are `airframe.py::_GAINS["cf21_brushless"]` (`platform_defaults_cf21bl.h`), converted
only by `_SiGains.from_firmware`. Rate loop: roll/pitch kp 200, ki 400, kd 2.5, i_limit 33.3;
yaw kp 120, ki 16.7, kd 0, i_limit 166.7; int16 output saturation; rate-D lpf2p at 30 Hz.

The 30 Hz filter is our documented deviation (`pid_firmware.rs::Lpf2p`): the firmware ships
it disabled because its gyro is pre-filtered. These are the SAME numbers the PID teacher and
the PID rival already fly. The cascade introduces no new gains.

---

## 2. Teacher and labels

### 2.1 Outer-loop teacher (default): the firmware ATTITUDE stage

`AttitudePidFirmwareRs::update_cascade` already computes `rate_sp = att[i].update(target, rpy)`
and feeds it straight to the rate stage. The label for banks 1-3 is exactly that `rate_sp`:
attitude kp 6 / ki 3 (yaw ki 1, kd 0.35, wrapped), evaluated on the teacher's ORACLE attitude
(`quat_to_euler(q)` of the true quaternion). The label for bank 0 is
`hover_pwm + AltitudePd::delta(alt_err, vz)`, the same composition as
`Teacher::step_with_collective`. Under `--translation` with the horizontal channel, the target is the
`PositionLoop::tilt_ref`, unchanged (`Teacher::step_full_state`).

**Frame note.** The firmware feeds Euler-angle-error-derived rates directly as BODY rates
(a small-angle identification). We inherit this for teacher fidelity. At the 30° curriculum
ceiling the yaw Euler rate leaks into q/r by sin 30° = 0.5. This is a property of the teacher being
imitated, not a bug to fix, but any "error in degrees" claim must say Euler roll/pitch about the
level reference (G5 `att_err_ref`).

**Label-replay identity.** This is the design invariant and the first test. If the student's
decoded output equals the teacher label every 2 ms, the student is bit-identical to the
`PidFw` teacher flying alone, because the inner stage is the teacher's own rate stage plus mixer.
So the teacher becomes "a perfect outer-loop imitator over the inner loop", and the gap between the
WNN and its teacher is pure imitation error. The inner stage must therefore compose the collective
EXACTLY like the teacher: `mix(axis_N)` at `hover_thrust_n`, then a uniform pwm offset
`T_coll − hover_pwm`, then clamp. It must not convert the collective to newtons first, or the
identity breaks in the 4th digit.

### 2.2 Outer teacher: MPCOF on the closed inner loop (ABI 39, CTRL-33 — supersedes the ABI-38 label)

**Superseded ABI-38 label.** The motor-level MPCOF's planned body rate N_o steps ahead. It planned
TORQUE as if it commanded the rotors, never modelled the rate PID, and under lag counted the motor
lag twice. Rival bar at lag 0.20: MPCOF>rate(h10) 18.8% stable / 36.9° vs PID>rate 100% / 1.73°.

**Derivation (code: `ctbr_axis_mpc.rs`, `ctbr_mpcof.rs`).** Each body axis is its own problem (the
'+' mixer and the diagonal inertia decouple the axes). Linear model at the plant tick dt (1 kHz),
state `s = [θ, ω, ua, I, f1, f2, pm, Fh]`:
- θ is the Euler-angle error. This is the firmware attitude stage's own convention (`state_error`):
  each Euler error drives that axis's BODY rate, and θ' = ω for small angles.
- ω is the body rate (gyro).
- ua is the lagged axis force (N).
- I, f1/f2, pm and Fh are the rate PID's internals: integrator, 30 Hz lpf2p delay line, last
  measured rate and held output.

On a rate-update tick (every `decimation` ticks, dt_r = 1/attitude_hz) the rate PID runs as in pid.c:
`e = sp − ω; I += e·dt_r; D = lpf2p(−(ω − pm)/dt_r); Fh = kp·e + ki·I + kd·D`. Every tick:
`ua += α(Fh − ua)` (α = `motor_lag_alpha` of the airframe's resolved lag, 1 at T = 0), then
`ω += dt(c·ua + d)` and `θ += dt·ω + ½dt²(c·ua + d)`. The torque gains are c = L/Jx, L/Jy and
4·k_drag/Jz (`body_torque` through the firmware mixer).

The OUTER model (A, B) is the exact composition of action_repeat such ticks from the loop's current
phase, built column by column from the tick function. The gains come from the airframe's firmware
cascade (the same `RateStageFw` every CTBR inner loop uses) and are never re-derived. Cost per outer
step: 100·θ² + 0.5·ω² + 1·sp². The terminal cost is the DARE solution of the same model. The
setpoint is boxed at the bank rate range (4.2, 4.2, 2.0 rad/s). The horizon is 20 outer steps,
solved exactly (`box_qp.rs`).

**Offset-free.** The observer is `d̂ += 0.02·(ω'_gyro − c·ua_measured − d̂)` every tick. The measured
ua is the applied pwm, passed through the plant's own lag (prime/seed protocol) and converted to
thrust kt·pwm². d̂ is therefore the torque the nominal rotors do not explain: asymmetry, jitter,
rotor drag and gusts. For a constant d the closed loop rests at θ = ω = sp = 0 with
Fh = ua = −d/c and I = −d/(c·ki). The rate integrator carries the trim, and it measurably absorbs the
±20% asymmetry. The plan regulates x − x_ss(d̂), so a nonzero d̂ never biases it.

**Inner-state estimate.** The teacher runs a SHADOW firmware rate stage on the measured gyro and on its
own held setpoints. When the teacher flies (C4 rival, expert_drives, calibration, recorder), the
shadow IS the inner loop. Under DAgger the student's setpoints drive the real loop, so before every
decision the rollout copies the student's rate stage (as it stood before this tick) into the shadow
(`Teacher::ctbr_sync_inner`, fix of 06/11/2026). The earlier rule — keep the shadow on the teacher's own
unflown setpoints and let d̂ absorb the mismatch — labelled states the vehicle was not in: along the
student's own states |I_shadow − I_real| was 0.16-0.40 against |I_real| 0.13-0.19, and the fix took the
asymmetry-scored probe from 47% to 76% (`_op31`) and from 20% to 35% (stage 2), 3/3 seeds
(`logs/controller/ctbr_probe/runs.log`).

**Collective.** Not planned: the altitude PD's deviation passes through as for the PID teacher. The
mixer adds it uniformly, so the planned axis forces are unchanged.

**Timing.** One decision per action_repeat ticks, held in between: the stream the student must
reproduce.

The label-replay identity still does NOT hold (the MPCOF setpoints are tracked by the rate PID), so the
"MPCOF > rate" rival is flown and reported alongside.

### 2.3 DAgger labels

The protocol is unchanged in shape. Teacher and student both see the start-of-step
snapshot, and the sim propagates on the student's action. Two things change:
- **Records only at decision steps**, which already happens under `action_repeat > 1`
  (`controller.rs` ~L5253).
- **The label is absolute in setpoint space.** `output_decode_target` receives the 4-vector
  `[T, p_sp, q_sp, r_sp]` instead of 4 pwm values. The bank-wise antagonist/threshold
  machinery is the same one `decouple_outputs` (H3, Option A) already uses for `[T, τr, τp, τy]`.

### 2.4 Is the delta output law still needed? No, so default OFF for the cascade

The delta law was introduced for two reasons. Both disappear.

1. *Untrained bootstrap.* Absolute pwm tumbles, while delta 0 holds throttle. Under CTBR the
   untrained decode is ω_sp = 0, T = anchor: "hold zero body rate at hover thrust". The inner
   loop makes that stable by construction. The vehicle does not level itself, but it does not tumble.
2. *Integral off-load.* The accumulator was the integrator. Under CTBR the torque-bias
   integral lives in the inner rate PID (ki 400), and the attitude integral is an observed
   feature (peraxis_i). The delta accumulator would add a hidden state with no purpose, which
   is the hidden-state problem obs_pwm was invented to patch.

Keep `delta_control` available as an ablation arm (`--delta-control` with CTBR), but it is
not the default. Also off: `dagger_label_delta`, `delta_label_scale` (n/a in absolute mode).

---

## 3. Inner-loop choice

| | firmware rate PID (DEFAULT) | MPCOF at rate level (ABLATION) | INDI (FUTURE) |
|---|---|---|---|
| exists in repo | yes: `pid_firmware.rs` + Metal `pidfw_step` | partly: lag-aware MPC + observer exist at attitude level, and need a rate-tracking reformulation | no |
| compute | 3 PID channels + lpf2p at 500 Hz, ~100 flops/tick; trivial on STM32 or a few DSP slices | box-QP over a horizon at 500-1000 Hz; heavy on MCU, feasible on FPGA | Δu = G⁻¹(ν − ω̇_f), plus filtered ω̇ and an actuator state (rpm or lag model); light |
| model dependence | none (sourced gains) | b (from calibrated inertia, which is an ASSUMPTION per `airframe.py`) and τ | control effectiveness G and the actuator model |
| lag handling | robust, measured (section 1.4: PM 43° at τ 50 ms) | explicit (augmented model) | needs rotor feedback or a lag model; CF2.1 BL rpm telemetry is unverified |
| deployability | stock Crazyflie firmware rate loop. The commander's attitude-rate setpoint mode (roll/pitch `modeVelocity`) feeds it directly; **verify against `controller_pid.c` before claiming** | requires custom firmware | Crazyflie firmware has `controller_indi.c`, but its cf21bl parameters are not sourced here |
| honest paper claim | "WNN replaces the attitude stage; the shipped rate loop is untouched" | "WNN over a model-based inner loop" (weaker: two learned/modelled parts) | out of scope for paper #1 |

**Recommendation.** Use the firmware rate PID as THE inner loop. It is model-free, already
CPU/Metal parity-tested, and deployable without touching the firmware. It also keeps the comparison clean, because the rival
uses the identical inner loop. MPCOF-at-rate-level is one ablation row, and INDI is deferred.

---

## 4. What the WNN is responsible for, and where it can win

**Responsibility.** Map the observed attitude/vertical/horizontal error state to
(collective, body-rate setpoint). This is the attitude PID's 9 gains plus the altitude PD,
as one learned lookup.

Be honest about the headroom. The cascade gives the classical rival the same lag robustness and
the same inner-loop disturbance rejection. Torque bias (L4C `tau_bias`), ±20% motor
asymmetry (`torque_scale_jitter`) and the torque part of the OU gust are rejected mostly
by the inner integrator FOR BOTH. Where the WNN can still differ:

1. **Privileged-teacher compensation of estimator error.** The teacher sees the oracle
   quaternion. The student sees the inclinometer/Mahony attitude, which is biased by specific
   force under horizontal acceleration (the accelerometer reads −g_body plus the kinematic
   acceleration). The student also sees vel_xy and pos_err_xy, so it can learn the
   correction. The PID rival flies on the same estimate (`PID[est]`). Headroom is bounded
   by PID[oracle] − PID[est]. In the ABI-36 smoke rows it is small, ~0.04-0.2° err, so do
   not pre-promise it.
2. **Nonlinear attitude shaping** at large tilt (the 8°→30° curriculum), where a fixed-gain P
   on Euler error is not optimal. Imitating PID caps this. Imitating MPCOF (2.2) is the way to
   inherit a better outer law.
3. **Hardware.** 256 RAM reads per decision at 100 Hz is ~25.6k reads/s for the outer loop, with no
   multiplier. The inner loop is the stock firmware. This supports the fanless/cheap-silicon story on
   the part of the stack the WNN actually owns.

**Measurable claims and their bars.** Report the triple stable% / err° / steady° plus alt and pos, judged as a whole.

| claim | arm | bar (rival over the SAME inner loop) |
|---|---|---|
| C1 lag robustness | CTBR-WNN at lag 0.20 vs CTBR-WNN at lag 0 | non-inferior. Contrast: direct-pwm 96.6% → 0% (R0/R1), and R2's 70%/4.55° |
| C2 parity | CTBR-WNN vs firmware attitude PID over firmware rate PID = the full `PidFw` cascade on `PID[est]` | non-inferior on the triple |
| C3 advantage (only if C2 holds) | L4C + translation, est attitude | beats `PID[est]` on steady°/err° |
| C4 outer law | CTBR-WNN(MPCOF teacher) vs MPCOF-attitude-over-fw-rate rival | within its teacher |

Significance, seed counts and margins: defer to `experiment-design`.

**Framing change, for Luiz to decide.** Today's comparison is the WNN as the *monolithic*
controller against classical *cascades* (`position_loop.rs` header). Under CTBR the WNN is one stage of a
cascade. That is a different paper claim and must be stated as such.

---

## 5. Implementation plan

### 5.1 Rust (`ram_controller` only; no `ram_core` change, so no worker swap)

1. **`pid_firmware.rs`**: split `AttitudePidFirmwareRs` into `att_stage(q, target) → rate_sp[3]`
   and `RateStageFw::step(rate_sp, gyro, coll_pwm) → pwm[4]`, holding the rate PIDs, lpf2p,
   decimation/hold and `mix`. `step_rs` becomes their composition.
   **Pin test:** the composition is bit-identical to the pre-split `step_rs` over a random
   trajectory.
2. **`controller.rs`** (`WnnController`):
   - New flag `ctbr: bool` plus `ctbr_rate_max[3]`, `ctbr_coll_max`, with γ per bank. Mutually
     exclusive with `decouple_outputs` and `obs_pwm` (hard assert). `delta_control` is allowed as an ablation.
   - Own a `RateStageFw` (built from the same SI arrays `from_si_arrays` takes).
   - `step()` returns pwm as today, so **every rollout host is unchanged**: dagger_train,
     cpu_score, position_score, calib_sampler, record_ops. On a decision step it decodes
     `[T, p, q, r]` and stores it as `held_sp`. On every physical step, including action-repeat holds,
     it returns `rate_stage.step(held_sp, gyro, T)` instead of `last_pwm`.
   - `reset()` resets the rate stage.
   - `bank_anchor`: bank 0 → collective_anchor, banks 1-3 → 0 (same rule as decouple).
   - `output_decode_target` accepts the setpoint 4-vector when `ctbr`.
3. **`optimal.rs` `Teacher`**: add `ctbr_label(q, gyro, target, outer) → [f64; 4]`.
   - PidFw: `att_stage` + `hover + pd.delta`.
   - MpcOf: the closed-inner-loop MPC's rate setpoint (2.2, ABI 39).
   - Others: return `Err` (unsupported, loud).
   - `observe()` is unchanged.
4. **`episode_regime.rs::OuterLoops`**: an `expert_ctbr` twin of `expert` that routes
   position → tilt ref → `ctbr_label`.
5. **`dagger_train.rs`**: at the label call site, pick `expert_ctbr` when `cfg.ctbr`. The sn=0 BPTT/
   direct-write path writes 4 banks, as decouple already does. The split trainer and sn>0 assert
   unsupported in v1.
6. **Rivals (`cpu_score.rs` / `position_score.rs`)**: the C2 rival is the existing `PidFw` on the
   estimator, so no new code. The C4 rival is a new composition, MpcOf-label → `RateStageFw`.
7. **Metal (`metal_controller.rs`, `shaders/controller_rollout.metal`)**: split
   `pidfw_step` into an attitude half and a rate half. This adds `uint ctbr_on`,
   `float ctbr_range[4]`, `float ctbr_gamma[4]` to `Params`, plus per-episode rate-stage
   state (3 integ + 3 prev + 6 lpf = 12 floats). The hold branch (~L1400) calls the rate half on
   the held setpoint. Host-side coefficients come from the Rust object (`rate_lpf_coeffs`,
   `decimation`) and are never copied by hand.
   `controller_bptt.metal`: sn=0 decode only.
8. **Parity (Hard rule 4).** Add `ctbr_cpu_matches_gpu` to the `cargo test -p ram_controller --lib
   --no-default-features` sweeps, taking them from 14 to 15 suites. Cover lag 0 and lag 0.20, action_repeat 1 and 10.
   Add the label-replay identity test (2.1) on CPU AND GPU. The quad-plus mixer oracle in
   `overactuated.rs` is untouched, but run it.
9. **ABI 37 → 38** (`lib.rs::ABI_VERSION`). Provenance line: `output=ctbr inner=pidfw_rate
   outer_hz=100 rate_max=… gamma=…`.

### 5.2 Python

- `phased_ga.py`: add `--output-mode {pwm,ctbr}` (default pwm, so the legacy lineage stays bit-identical),
  `--ctbr-rate-max R,P,Y`, `--ctbr-coll-max`, `--ctbr-teacher-level {pidfw,mpcof}`, and
  `--ctbr-gamma`. The outer rate uses the existing `--action-repeat`. Hard-fail `ctbr + --obs-pwm`.
- `evaluator.py::ControllerSpec`: new fields. Update `num_features` (obs_pwm is already conditional).
  **Shape consumers to audit together:**
  - `evaluator.py::arch_shape_from_spec` / `prefix_factor`
  - `controller_grid_search.py` (L213-260)
  - `arch_adaptation.py`
  - `recurrent_genome.RecurrentArchShape`
  - threshold fitting (feature thresholds are fit on the flown distribution and must refit
    for the CTBR closed loop)
  - checkpoint loaders that assume the layout
- `wnn/control/_accel.py`: ABI 38 assert.
- `wnn/ram/experiments/params.py::KNOWN_PARAMS`: add `controller_output_mode`,
  `controller_ctbr_rate_max`, `controller_ctbr_coll_max`, `controller_ctbr_teacher_level`, and
  `controller_ctbr_gamma`, so the grid-orchestrator flow path can carry them.

### 5.3 Effort (agent wall-clock)

| step | estimate |
|---|---|
| Rust CPU path (1-6) + pin/identity tests | 2.5-3.5 h |
| Metal split + parity sweep (7-8) | 3-4 h |
| Python plumbing + shape audit | 1-1.5 h |
| MPCOF rate label + C4 rival | 1-1.5 h |
| pre-deploy wheel review + rebuild (`maturin develop --release -m controller/Cargo.toml`, swap-free) | 0.5 h |
| **total before the first smoke** | **~8-10.5 h** |

Smoke: ~5 min. Each pop-50 grid: ~10-30 min (R0-R2 grid winners took 547-1015 s, and lag
does not slow the plant). One controller run at a time.

### 5.4 Lineage and comparability

This is a new lineage with suffix `_ct38`, and no CTBR marker is pooled with a direct-pwm marker. Comparisons
across lineages go only through elements both lineages share:
- the same scorer (G5, L4C, translation, test seeds 99990101-05);
- the same rivals (`PID[est]` / `PID[oracle]`, printed in both);
- the R0/R1/R2 rows above as the direct-pwm reference.

Old markers stay valid within their own lineage.

---

## 6. Risks and open questions

- **R1. Does lag still matter at the outer loop?** By the linear analysis (1.4), barely: about
  4° of phase at the attitude crossover. The collective channel is NOT inside a closed inner loop:
  the 20 rad/s lag pole sits under the altitude PD (ωn 2), about 6° of phase. Expect a small alt
  degradation and no attitude collapse. This is checked by C1, and by the classical rival's own lag-0 vs lag-0.20 delta.
- **R2. Imitation ceiling = teacher.** Over the same inner loop, a perfect imitator IS the
  rival (2.1). Expect C2 parity at best on the first lineage. C3 headroom is the
  oracle-vs-est gap, which is small. If C3 fails, the paper claim is robustness plus hardware, not superiority.
- **R3. Quantisation dead band.** At γ = 1 the dead band is 1.25°. γ = 2 is mandatory (1.3). Watch for
  limit cycles in steady°. This is the same SOFT-failure class as the known ~5.6° offset.
- **R4. Integral mismatch.** The label's attitude-I is a clamped pure integrator (i_limit
  20°·s). `peraxis_i` is leaky at `integral_leak` 0.99 per PHYSICAL step, a 0.1 s memory, so
  the ki·∫ part of the label is only partly observable. Set `--integral-leak 0.999` (1 s) for
  the CTBR arm, or bound the error by noting that ki/kp = 0.5 s⁻¹ makes the I contribution slow and small.
- **R5. Gyro bias walk enters the inner loop.** The rate loop tracks ω_sp + bias. Roll/pitch
  drift is corrected by the outer loop through accel. Yaw drift is unobservable, exactly as in the
  firmware. This is not a new weakness, but yaw err will look the same as today.
- **R6. Yaw range/separation.** Set r_sp's range from the smoke label histogram. The yaw
  rate-loop BW is only ~1.5-2× the yaw attitude loop.
- **R7. Data volume.** 100 Hz means 10× fewer DAgger records per 2 s episode (200 vs 2000). If the
  grid underfits, try in order: 250 Hz, then more episodes (`--eval-episodes`), and do not raise `--steps`.
- **R8. Euler-vs-body identification** at large tilt is inherited from the firmware (2.1).
- **R9. sn > 0 unsupported in v1.** The ongoing sn=4 diagnosis (R3) does not transfer
  directly. Revisit only if sn=0 CTBR plateaus.
- **Sim-to-real.** This is the strongest argument for the design. The learned component moves from motor dynamics
  (rotor lag, thrust curve, inertia, of which inertia is DERIVED, not sourced) to attitude
  kinematics plus estimator behaviour, the most faithfully simulated part. On hardware the
  real lag, real ESCs and real inertia are handled by the vehicle's own firmware rate loop. Its gains are
  the ones we simulate, so the inner loop transfers by construction. The residual gap is the
  inner closed-loop response (BW 76 rad/s in sim), which a hardware step test measures in minutes.

### Validation sequence (NOT launched; the box is running a diagnosis)

1. **Tests.** Pin test (5.1.1), label-replay identity on CPU and GPU (2.1), and `ctbr_cpu_matches_gpu`.
   The identity rollout must reproduce `PidFw` stable/err to the last digit at lag 0 and lag 0.20.
2. **Smoke, pop 6, plumbing only.** Same argv as `abi36_fix/smoke_T020.out` with these changes:
   `--output-mode ctbr`, no `--obs-pwm`, `--action-repeat 10`, `--motor-lag-s 0.20`. Read the label
   histograms for R6. No decisions are made from it.
3. **Pop-50 grid A/B**, `_op31` recipe minus `--obs-pwm`, plus CTBR, seed 31337002, grid only
   (as R0-R2), at **lag 0.20 and lag 0**. Pass: C1 non-inferior, and lag 0.20 clearly above R2
   (70%/4.55°). Read C2 against the `PID[est]` row in the same log.
4. **Then, one at a time:** outer 250/500 Hz; MPCOF teacher (C4); delta-law-on ablation;
   `experiment-design` sets the seed count before any claim.
