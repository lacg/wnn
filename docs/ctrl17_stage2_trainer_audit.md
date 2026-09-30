# CTRL-17: stage-2 trainer audit (lateral half)

Date: 26/09/2026. This was a read-only code and physics audit: no code was changed and nothing was run. It unblocks CTRL-8.
Scope: can the DAgger training rollout (`dagger_train.rs`: `sim_default` → `rollout_and_label_rs`) and every
other rollout that shapes the student's address function (threshold fitter, MEMORY recorder, per-round
checkpoint eval) see and command the lateral channel that the scorer flies? Stage 2 is the 4-flag bundle
`--translation --xy-offset >0 --obs-pos-err-xy --obs-vel-xy` (plus `--fit-weight-pos`).

Paths below are relative to `src/wnn/ram/strategies/accelerator/controller/` unless prefixed `control/`
(= `src/wnn/control/`).

## Status (26/09/2026, branch `stage2-trainer-fixes`, controller ABI 31)

G1-G4 and G11 are FIXED, vertical half included (Luiz 26/09: new lineage for every
translation run; attitude-only runs are byte-identical, pinned by `ctrl17_pins`). One shared
episode definition (`controller/episode_regime.rs`) now feeds the training rollout, the
per-round checkpoint eval (G3), the calibration sampler (`calib_sampler.rs`, G1) and the
MEMORY recorder's cascade driver (G2); the gate and checkpoint rank on
`episode_regime::step_reward` with λ_alt/λ_pos (G4; default rule in
`wnn/control/gate_lambdas.py`). G13 (found here, FIXED 27/09): under `expert_drives` + D0
re-base + translation the sim flew the re-based LABEL, not the teacher's pwm (cf21 fell
~1.2 m in 2 s); it now flies the raw teacher pwm. Same landing: FFD sub-batch packing in
ControllerEvaluator (bit-identical to unbatched/contiguous) and `--rg-easy-tilt-deg` /
`--rg-full-tilt-deg` (defaults 8/30 unchanged). G5-G10, G12 stay open.

## Verdict: GAPS

The trainer's **core DAgger loop is wired for lateral motion**: teacher, plant, features, replay (CPU and Metal) and
deploy all agree. This is not a repeat of stage 1. But the **three side rollouts around that loop fly an
attitude-only (or stale) world**, and **two selection mechanisms rank on attitude-only reward**. Stage 2
would therefore train on a real lateral teacher while encoding the lateral features as sign bits,
recording a degenerate MEMORY universe, and keeping the DAgger round that learned the least lateral
control. G1–G3 must be fixed before a CTRL-8 marker. G4–G5 are decisions to make before the pre-registered design.

## 1. Teacher: YES, the training rollout commands lateral tilt

- `pos_loop_for` (dagger_train.rs:1501-1516) builds a `PositionLoop` when `translation && xy_offset > 0`.
- `rollout_and_label_rs` (dagger_train.rs:1706-1727) calls `teacher.step_full_state(q, gyro, target[2], pl, pd,
  −x, vx, −y, vy, alt_err, vz)` whenever both the position loop and the altitude PD exist.
- `Teacher::step_full_state` (optimal.rs:1543-1561) takes `PositionLoop::tilt_ref` (position_loop.rs:99-104),
  uses it to **replace** roll/pitch in `target_rpy`, and then applies `step_with_collective`. This is an enum
  method, so all six ids that `TeacherBank::get_mut` (dagger_train.rs:1313-1321) can return (PID, PidFw, LQR,
  MPC, LQI, MPCOF) gain position control. mpcof's `observe()` is unaffected.
- The label is the output of `teacher_label_f32(expert_pwm, rebase)` (dagger_train.rs:1741; D0 rebase at 1168-1209).
- `position_loop.rs` is **not** scorer-only. The same loop drives the rivals (`score_classical_baseline`
  dagger_train.rs:3198-3212, parameters forwarded from `control/classical_baseline.py:153-155`) and the teacher-bar
  scorer (position_score.rs:129, 270).
- Sign check in the WORLD frame, done against the plant. `step_translation` (controller.rs:2501-2521) accelerates at
  (T/m)·(r13, r23), with r13 = 2(qx qz + qw qy) and r23 = 2(qy qz − qw qx). At yaw 0: +pitch → +x and +roll → −y.
  `tilt_ref` uses the same signs, the mixer (controller.rs:7064-7072) and `state_error` (optimal.rs:299-310) close
  the loop negatively, and the unit tests pin it (position_loop.rs:120-176, controller.rs:8486).

## 2. Plant: YES, the training sim carries xy and the same disturbances

- Training: `set_translation_core` zeroes x/y/vx/vy (controller.rs:552-580). The xy initial condition U(−xy_offset, +xy_offset)
  per axis, at rest, is drawn from the episode rng (dagger_train.rs:1579-1584). `reset()` zeroes them per episode
  (controller.rs:963-968).
- Scorers: CPU cpu_score.rs:193-197 (initial condition) and 271-275 (observation). Metal shader controller_rollout.metal:1016-1017
  (buffer-30 slots 4/5), 1218-1221 (observation), and 1594-1604 (dynamics, with the component-order trap handled). Pinned by
  `parity_stage2_horizontal_channel` (metal_controller.rs:10044).
- Disturbances: the trainer uses `apply_cfg_disturbance` (dagger_train.rs:1592). This is the same D1–D7 set as the scorer
  (controller.rs:1135-1245; Metal twin). All of them are **torque-only** in both paths: there is no lateral wind force
  anywhere. The WORLD-frame lateral error can therefore only come from the initial condition or from tilt. Motor lag comes in
  through `AirframeRs::from_cfg`.
- Python forwards `xy_offset` and `lambda_pos` (control/evaluator.py:919-922, via `_plant_train_kwargs` at 1583 and 1713). It does not
  forward `pos_omega/pos_zeta/pos_max_tilt_rad`. The Rust defaults (1.0 / 1.0 / 0.5236) equal the rivals' defaults today, so there
  is no divergence now. Forward them anyway (G10).

## 3. Features: training replay and deploy are wired; the fitter, recorder and eval are NOT

| Consumer | Status | Evidence |
|---|---|---|
| Rollout observation before student step | OK | dagger_train.rs:1712-1714 (`set_horizontal_obs(−x,−y,vx,vy)`) |
| Recorded per step | OK | dagger_train.rs:1782-1783 (`traj.horiz_obs`) |
| CPU replay (BPTT + split) | OK | `ReplayObs::slice` dagger_train.rs:1905; `apply_replay_obs` controller.rs:1797-1810; fail-loud guard 1762-1790 |
| GPU split replay | OK | `flatten_gated` dagger_train.rs:2146 → buffers 24/21 → shader 1974-1977, 2173-2176 |
| Deploy CPU / Metal | OK | cpu_score.rs:271-275; shader 1218-1221 |
| Feature layout | OK | controller.rs:2027 (+2 +2), order e_x,e_y,v_x,v_y at 5737-5748; control/evaluator.py:331; `arch_shape_from_spec` uses `spec.num_features()` (control/evaluator.py:1073) |
| **Threshold fitter** | **GAP G1** | control/evaluator.py:491-800 |
| **MEMORY address recorder** | **GAP G2** | record_ops.rs:103-153; lib.rs:664-672 |
| **Per-round checkpoint eval** | **GAP G3** | dagger_train.rs:1944-2080 |
| Student-refit sampler | GAP G9 (pre-existing) | control/evaluator.py:829-895 |

(`control/genome.py:57-60` `_Layout` still hardcodes `NUM_FEATURES` and `2·n_state`. It is only reached through
`FiniteStateGenome`/`ga_strategy`, which is not on phased_ga's RecurrentArchGenome path, so it does not touch stage 2. It is legacy.)

## 4. Observability (frames named)

- **Position error and velocity are in the WORLD frame** (controller.rs:2825-2826; target = origin, so e = −p). The student acts
  in the BODY frame through motor differentials. The map from world error to body tilt is a rotation by heading ψ.
  **Every episode commands yaw_ref = 0**: `target` is [0,0,0] in training, scoring and rivals. The teacher drives ψ → 0,
  so after the yaw transient the world frame equals the heading frame, and the student can learn the direct map
  e_x → pitch, e_y → −roll with no product term. **The lateral error is observable from the new features.** It is not aliased with the
  attitude channels: the bits are separate, and the address only has to co-sample pos bits with tilt bits.
- Yaw coupling, a transient only (G8). `tilt_ref` does not rotate by ψ, so the realized WORLD acceleration is Rz(ψ)·a_des.
  For the ζ = 1, ωn = 1 loop rotated by ψ, the slowest pole's real part is −1.00 at ψ = 0°, −0.59 at 30° and −0.50 at 60°,
  and the loop is lost at about 76°. `max_initial_yaw_rad` = 0.5236 (30°), so the teacher only spirals slightly during the
  yaw transient. The teacher, rivals and student all use the same WORLD frame, so **there is no train/deploy mismatch**.
  It would start to matter only with yaw_ref ≠ 0, or with yaw drift from `gyro_bias_walk` on a yaw-blind student (small over 2 s).
- **Accelerometer fidelity (G12, disclose or fix).** `imu_base` (controller.rs:598-606; Metal shader 1120) returns
  −g_body. This is an ideal inclinometer **even while the vehicle accelerates laterally**. A real IMU reads specific
  force R^T(a_world − g_world) ≈ (0, 0, T/m) during coordinated lateral acceleration, so its x/y components carry **no tilt**.
  In stage 2 the student, and the Mahony-fed rivals, therefore get a tilt oracle through accel_x/y that hardware would not provide.
  Train and score share it, so this is not a trainer gap, but it inflates stage-2 observability. The same applies to accel_z
  under stage-1 vertical acceleration.
- Thermometer scale, after G1 is fixed. The quantile ladder spans e ∈ [−xy_offset, +xy_offset], and v spans the teacher's
  fly-back speeds (up to about 0.37·x0 m/s at t = 1 s for ωn = 1). As things stand the ladder is degenerate (G1).

## 5. Gaps (severity, minimal fix, wheel)

**G1: HIGH. The threshold fitter never flies the lateral channel (stage-1's 13/08 degenerate-ladder bug, one channel up).**
`fit_thresholds_from_pid_rollouts` (control/evaluator.py:491-800) handles translation by setting the vertical state and
feeding `feat_ctl.set_vertical_obs` (700-704). It never calls `sim.set_horizontal_state` or
`feat_ctl.set_horizontal_obs`, and its driver is the attitude-only firmware PID with target (0,0,0), which has no position loop.
All four xy features are sampled as the constructor's 0.0 (horiz_obs default, controller.rs:2196), so the quantile ladder is
**all thresholds = 0.0**. The result is a sign-only code: 8 bits per feature carrying 1 bit, i.e. 32 address bits for 4 bits
of information. Drawing x0/y0 alone is not enough. Under an attitude-only PID, e stays at x0 and v ≈ 0, so the velocity ladder
would still be degenerate. The fitter has to fly the full-state cascade.
Minimal fix: expose a Rust sampler, for example `sample_calibration_features(...)` in ram_controller, that flies
`AirframeRs::teacher(id).step_full_state` plus the xy/z initial-condition draws on the calibration plant and returns per-feature samples
from the controller's own `compute_features`. The fitter consumes that sampler when `max_initial_xy_offset_m > 0` (Rust-first; no
Python PositionLoop). This touches **ram_controller** (a new export, so an ABI bump) and Python. It is swap-free, and stage-1 and
attitude runs stay byte-identical because the sampler is gated.

**G2: HIGH. The MEMORY recorder drops the horizontal draws, and the guard gives a false green.**
`record_address_universe` puts `s2_init_x/y` into `Stage1Cfg` (lib.rs:664-672, whose own comment says "not yet threaded"). But
`run_episode` (record_ops.rs:103-153) never calls `set_horizontal_state` or `set_horizontal_obs`, so x/y/e/v stay 0.
The Python guard (control/ga_memory.py:192-198) only checks that `s2_init_x` reached the binding, so it passes. Also, the driver
is `AttitudePidRs::new_default()` at hover **0.5** (controller.rs:7256) with no altitude or position loop, so the vehicle
under translation does not hover at all. That half is a pre-existing stage-1 recorder issue.
Minimal fix: in `run_episode`, when `cfg.has_horizontal()`, call `set_horizontal_state(x0,y0,0,0)` after `set_vertical_state` and
`set_horizontal_obs(−x,−y,vx,vy)` at each step's snapshot, and drive with `Teacher::step_full_state` at `nominal_hover_pwm`.
This touches **ram_controller** only. Gate on `has_horizontal()` so stage-1 universes stay byte-identical; switching stage 1 to the
cascade driver is a separate lineage decision. Until this lands, stage-2 runs must skip the MEMORY stage.

**G3: HIGH. The per-round checkpoint eval flies stale observations and ranks on attitude only, which selects AGAINST lateral learning.**
`eval_closed_loop_rs` (dagger_train.rs:1944-2080) never calls `set_vertical_obs` or `set_horizontal_obs`, and
`controller.reset` does not clear them (controller.rs:3216ff). Every eval episode therefore addresses on the previous
rollout's **last-step (e, v, alt_err, vz)**, held constant, while the sim starts at x = y = 0 with no xy draw. Its fitness
is `compute_reward(attitude_err)` only (2061), and with `keep_best_checkpoint = True` (control/reward_gated.py:157;
restore at dagger_train.rs:2474-2486) the returned memory is the best round on that eval. A round that learned to tilt
toward a lateral error will tilt toward the phantom constant offset, lose attitude reward, and be discarded.
The vertical half has been stale since stage 1, so this is pre-existing.
Minimal fix: have the eval copy the rollout's per-episode setup (translation mass draw, vertical and horizontal initial conditions,
collective anchor, per-step `set_vertical_obs`/`set_horizontal_obs`) and score with `compute_reward_stage2`. This touches
**ram_controller**. Gate the horizontal draw on `xy_offset > 0` so stage-1 eval rng stays byte-identical. The stage-1 vertical
half changes the eval rng, which is a lineage break for Luiz to decide.

**G4: MEDIUM-HIGH. The DAgger gate is attitude-only.** The rollout's cumulative reward is `compute_reward(attitude_err…)`
(dagger_train.rs:1811). `cfg.lambda_pos` is plumbed (316) but **never read**, and `lambda_alt` is not in the packed cfg at
all. The default `gate_mode = "improvement"` (control/reward_gated.py:108) admits a trajectory only if its attitude reward
beats the history. Episodes with larger |x0| need more tilt, so they are systematically gated OUT of training.
Fix: use `compute_reward_stage2` in the rollout (add `lambda_alt` to `RewardGatedConfigPacked`) or run stage 2 ungated.
This touches **ram_controller**. Choosing λ is a design decision (see `docs/scope_c_stage2_lambda_pos_sweep.md`).

**G5: MEDIUM (fitness definition, not code). Attitude metrics penalize the tilt that translation requires.**
err/stable/steady are measured against LEVEL (cpu_score.rs:391; Metal shader 1636-1639). The WNN is monolithic, so it has no
tilt reference to be measured against. The initial required tilt is θ = ωn²·x0/g, which is 2.9° for 0.5 m, on a
stable threshold of 5° mean error. Options: measure stage-2 attitude error against the cascade's `tilt_ref(e, v)`, which the
scorer can compute, or accept the trade and say so. This belongs to controller/experiment-design.

**G6: MEDIUM (measure first). Lateral content in the labels sits mostly in the dead zone.** Take the LQR teacher,
k1 = √12 = 3.46 pwm/rad through `mix_to_motors_f64`, and the live label's ±1/16 grid (project_live_dagger_label_dead_zone).
The initial pitch/roll kick crosses one level only when |θ_ref| ≳ 0.018 rad, i.e. |x0| ≳ 0.18 m at rest. After that, tracking a
slow (ωn = 1) tilt reference keeps the tracking error inside the dead zone, so most lateral information arrives in
the first about 0.2 s of each episode. Run `scripts/teacher_step_histogram.py` with xy on and off before deciding anything.

**G7: MEDIUM (teacher bar). Horizon and missing integral.** (a) At `--steps 2000` (2 s), a critically damped ωn = 1
loop leaves (1+t)e^{−t} = **40.6 % of x0** at t = 2 s. The position metric is therefore transient-dominated, and chunk-B's "0.006 m settled" bar does
not apply. (b) `PositionLoop` is PD with no integral. Any inner-loop steady tilt offset δ parks the vehicle at
e_ss = g·δ/ωn², about **0.17 m per degree**. Teachers that are not offset-free (PID/LQR/MPC under L2D τ-bias) will sit off-target.
Measure `score_position_teacher` per teacher under the run's disturbance at 2 s and at ≥ 6 s before fixing the WNN bar.

**G8: LOW.** The yaw rotation is missing from `tilt_ref` (§4). No action while yaw_ref ≡ 0. If that changes, rotate (e, v) by −ψ in the teacher
and feed heading-frame features, which is a ram_controller change and a feature-semantics lineage break.

**G9: LOW (pre-existing).** `collect_student_feature_samples` (control/evaluator.py:829-895) sets no translation and no
vertical or horizontal observations, so `--threshold-refit-from-student` appends zeros for stage-1/2 features. The fix is on the Python side, calling the
same Rust sampler as G1.

**G10: LOW.** (a) The guard at control/phased_ga.py:3234-3237 tests `--reward-lambda-pos`, but its message names `--fit-weight-pos`, and
`--fit-weight-pos` itself is unguarded. (b) `mean_position_error_m` is **3-D** (cpu_score.rs:418-422; Metal 1655-1659), so a
pos rank double-counts altitude alongside `--fit-weight-alt`. Report the horizontal radial error separately. (c) `aggregation =
desirability` refuses `weight_pos > 0` (ram/fitness/FitnessCalculatorControllerHarmonic.py:130), so the stage-2 recipe must be
zscore or harmonic. (d) Forward `pos_omega/pos_zeta/pos_max_tilt_rad` in `_stage1_train_kwargs` so the trainer and the rivals have one owner.

**G11: LOW.** No test pins the TRAINER's lateral teacher. Rivals are covered (dagger_train.rs:3812), the loop is covered
(position_loop.rs), the sim is covered (controller.rs:8486), and the Metal parity is covered (metal_controller.rs:10044), but `rollout_and_label_rs`
with `xy_offset > 0` is never exercised (tests at 4003 and 4529 use 0.0). Add a cargo test: sign(pitch differential) = −sign(x0)
over the first 50 steps, `horiz_obs` non-zero, and x → 0 under `expert_drives`.

**G12: MEDIUM for paper claims.** The accelerometer is an inclinometer (§4). The fix is specific force from the translational
acceleration, applied only when translation is on, on CPU and Metal together. It is parity-gated and a lineage break for every translation run.
Otherwise, disclose it.

Wheel summary: G1–G4 and G11 touch **ram_controller only** (swap-free; land at an idle window and smoke ONE before any
chain). G1 adds an export, so the ABI bumps. G5, G6, G7 and G10 are Python or design work. No `ram_core` change, no worker wheel.

## 6. Smoke plan (in order; the first three need no box time)

1. `cargo test -p ram_controller --lib --no-default-features` with the new G11 test, plus the existing
   `parity_stage2_horizontal_channel`.
2. Teacher bar: `score_position_teacher` for ids 0-5 at xy 0.5 m under the recipe plant and disturbance (cf21, L2D, lag),
   `steps` 2000 and 6000. Record the mean and final radial error. This sets G7 and the WNN bar.
3. Label content: `teacher_step_histogram.py` with xy 0 vs 0.5 m, giving the fraction of pitch/roll motor-steps at |level| ≥ 1
   (G6).
4. After G1–G3 land: a pop-6 stage-2 launch, one seed, grid → neurons, **MEMORY skipped**
   (`--translation --xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --fit-weight-pos 0.10`, zscore aggregation).
   Pass criteria:
   (a) no guard fires (dagger_train.rs:2267, phased_ga.py:3229);
   (b) the fitter prints **non-degenerate spans** for all four xy features (today: span 0.0, so FAIL);
   (c) the gen line carries the position metric;
   (d) held-out horizontal radial error is below the start mean (about 0.77·xy = 0.38 m for U(−0.5, 0.5)²) and below a
   `--no-obs-pos-err-xy` twin;
   (e) the rival column starts off-origin.
   Re-enable MEMORY only after G2, with a check that the recorded universe has non-zero horizontal bits.
