# CTRL-48 block 10 levers: SPEC (14/11/2026)

Approved by Luiz 14/11/2026: levers 1 and 2 of `docs/ctrl_gap_after_block9.md` §B.6 and the altitude integrator of §C.5, plus a hold-window position column. Branch `ctrl-block10-levers` (worktree `.claude/worktrees/ctrl-block10-levers`). Controller ABI 50 → 51. The wheel is BUILT, not installed; the main checkout stays on ABI 50 while block 9 and the stress test fly.

Rule for everything below: **each lever is its own default-OFF flag, and with every flag off every existing path is bit-identical** (pinned by a flags-off test against the ABI-50 pins). Vocabulary: val = tuning, test = report. Pre-checks are VAL only.

## L1. Attitude error against the tilt reference, bias-corrected (`--obs-tilt-ref-err`)

**What changes.** The VALUE of the two existing inputs `roll_err`, `pitch_err` (`--obs-peraxis-p`) and therefore of their integrals `roll_i`, `pitch_i`. No input is added, removed or moved: the feature count and every layout offset are unchanged.

**Definition** (BODY roll about x, pitch about y; all f32, the student's own signals):
- (roll_ref, pitch_ref) = `PositionLoop::tilt_ref(e_x, v_x, e_y, v_y)`, the fixed disclosed position law (b_xy = g, ω_n, ζ, max tilt), evaluated on the horizontal observation the student already reads (`horiz_obs` = [e_x, e_y, v_x, v_y], WORLD frame).
- (abp_r, abp_p) = this step's `AttBiasPredictor` output (predicted estimate − truth), the value the trailing `abp_r`, `abp_p` inputs carry on the same step.
- today: `roll_err = target_r − roll_est`. With the flag: **`roll_err = (target_r + roll_ref + abp_r) − roll_est`**, and the same for pitch. That is minus the bias-corrected attitude error against the reference the scorer measures.
- Yaw, `tilt`, the raw target inputs (features 6-8) and the `abp_*` inputs are untouched.

**Why inside the existing inputs.** The gap analysis found the student cannot form (estimate − bias − reference) across separately quantised inputs. Computing it in floating point before the ladder is the fix; putting it in the existing slots avoids a layout change (Hard Rule 2) and makes the existing integrals integrate the right error.

**Information.** Position error, velocity, Mahony estimate, bias prediction: all already student inputs. The position law is the one the label and the scorer use. No new information.

**Requires** (refused at launch otherwise): `--obs-peraxis-p`, `--obs-att-bias-pred` (so also the estimator and `--obs-vel-xy`), `--obs-pos-err-xy`, translation with xy offset > 0.

**Order of evaluation.** The bias predictor is stepped once per physical step BEFORE the attitude block (today it is stepped at the end of `compute_features`); its value is cached for the trailing block. Live and replay.

**Replay.** Nothing new is recorded. A replay recomputes the inputs from streams it already re-applies: `horiz`, `att_up` (the estimate), `att_bias` (the prediction).

**Ladder** (`--tilt-err-ladder {quantile, dense}`, default `dense` when L1 is on): see L2; inner cuts ±0.08°, ±0.20°, ±0.45°, outer pair at the calibration distribution's outer quantile.
- Why not quantile: the calibration mass is the transient (the current `roll_err` cuts run to ±0.8°), so quantile cuts land where the samples are, not where the hold needs resolution. The offsets to resolve are 0.1-0.5°.
- Why no cut at 0: with 8 cuts a cut at 0 costs a pair (today's yaw ladder spends two cuts on 0). A dead band of ±0.08° is below the within-episode motion (0.13-0.27° sd).

## L2. Dense-near-zero centred ladder (`--yaw-centred-ladder dense`)

- `--yaw-centred-ladder` takes an optional value: no value = `quantile` = today's ladder, bit-identical. `dense` = the new one.
- **Dense ladder for a zero-referenced input with b cuts:** symmetric pairs, no cut at 0 when b is even (one at 0 when odd). The inner pairs are FIXED physical cuts; the remaining pairs are |x| quantiles of the calibration samples above the last fixed cut, the outermost at the quantile ladder's outer mass.
- Yaw (`yaw_err`): fixed ±0.10°, ±0.25°, then quantile pairs. `yaw_err_i`: the same four angles scaled by the integral's steady-state gain `integral_scale / (1 − yaw_integral_leak)` (so a held offset of 0.10° settles on the first cut), then quantile pairs.
- Python only (`evaluator.py` threshold fitting, `threshold_calibration.py`, `feature_layout.py` families). Thresholds reach Rust as data; no kernel change.

## L3. Altitude integrator (`--alt-integral {off, pi, pole3}`)

**Law.** One integrator of the altitude error per episode, I ← I + e·dt (e = target − z, m; I in m·s), reset at every episode start. It reshapes the inputs of the UNCHANGED altitude PD (`AltitudePd::delta(alt_err, vz)`, which is linear before its clamp):
- `pi`: e_eff = e + (k_i/ω²)·I, vz_eff = vz, with **k_i = ω³/4**. PD poles untouched; Routh limit is k_i < 2ζω³, so this is 1/8 of it at ζ = 1.
- `pole3`: e_eff = 3e + ω·I, vz_eff = 1.5·vz. That is k_p = 3ω², k_d = 3ω, k_i = ω³: all three closed-loop poles at −ω. Defined for ζ = 1 only (refused otherwise).
- ω, ζ are the altitude loop's own (`--alt-pd-omega`, default 2 rad/s). Nothing is tuned on test units. Which of `pi` / `pole3` flies is chosen on VAL (pre-check).

**Anti-windup.** (a) I is clamped so the integral acceleration k_i·|I| ≤ 0.4·g (covers a thrust-to-weight error of 40%; the draws reach about 35%). (b) I does not advance on a step where the PD's unclamped command is at its limit (`|az_des| ≥ b_z·max_delta`).

**Where it applies, identically** (one type, `AltitudeIntegrator`, own file):
- every classical rival: `score_classical_baseline` (motor-level and cascade branches: PID, PID>rate, MPCOF, MPCOF>rate), `trace_classical_baseline`, `score_position_teacher`;
- the label and every teacher-flown rollout: `episode_regime::OuterLoops` (`expert`, `expert_ctbr`: DAgger labels, calibration, recorder, expert_drives);
- **the student's `alt_err` input becomes e_eff** (CPU `feed_live_obs` / `rollout_one`, and the Metal rollout kernel's `vert_alt_err`). The label is then a memoryless function of inputs the student has (e_eff, vz); without this the label would depend on a hidden integral (Hard Rule 1). Replays re-apply the recorded `vert` stream, so no replay kernel changes.

**Untouched (the task definition):** episode length, start distributions, the position loop and its tilt reference, the scorer's err / steady / stable / pos / alt definitions (all on the TRUE state), the disturbance level. L3 changes a controller law and one observation, not the task.

**Consequence.** With L3 on, labels and all rival rows change. The `both` control and every rival row must be re-flown with the same `--alt-integral` for a block-10 comparison (decision for Luiz).

## Hold-window position column (no behaviour change)

- Rival scorers return two more values after the existing eight: mean 3-D position error over the hold window (t ≥ 0.8·steps, the `steady` window) and mean |dz| over it.
- The WNN scorer (CPU and Metal) gains the same two metrics as appended rows.
- Printed as `pos_hold=` and `dz_hold=` next to `pos=`. Existing values and their order are unchanged.

## Plumbing

| layer | L1 | L3 | hold column |
|---|---|---|---|
| CPU | `controller.rs::compute_features` (+ enable method carrying the position-loop shape), `cpu_score.rs` | new `altitude_integrator.rs`; `episode_regime.rs`, `dagger_train.rs` scorers, `position_score.rs`, `cpu_score.rs` | `dagger_train.rs`, `position_score.rs`, `cpu_score.rs` |
| Metal | `derive_features` call sites (rollout, replay/split, walk) get the effective target; uniforms appended at the END of each Params struct | rollout kernel `vert_alt_err`; uniforms appended at END | rollout kernel metric rows |
| DAgger label / recorder | features only (labels unchanged) | `OuterLoops` | - |
| Python | `ControllerSpec.obs_tilt_ref_err`, `--obs-tilt-ref-err`, `--tilt-err-ladder` | `EpisodeConfig.alt_integral`, `--alt-integral`, rival kwargs | metric dict keys, print lines |
| `[provenance]` | `obs=...,tilt_ref_err` and `calib=...,tilt:dense` | `alt=pi` / `alt=pole3` (absent when off) | - |
| facade | `src/wnn/control/_accel.py` `EXPECTED_ABI = 51` on this branch only | | |

## Tests (all in `cargo test -p ram_controller --lib --no-default-features`)

- **Flags off = ABI 50:** every existing pin and parity sweep unchanged, plus an explicit test that a controller / scorer / trainer built with the new arguments at their defaults is bit-identical to one built without them.
- **L1:** the input equals the definition on a hand-built state; replay = live (train = deploy); CPU scorer = Metal scorer; CPU = GPU on the trainer walk and the split trainer; the predictor's trailing inputs are unchanged by the reordering.
- **L3:** the integrator removes a constant thrust-to-weight error on the stage-1 plant (droop → 0) for both modes; clamp and conditional integration; every rival host and `OuterLoops` produce the same collective on the same state history; CPU scorer = Metal scorer with the reshaped `alt_err`; reset per episode.
- **Hold column:** the two new values equal a direct recomputation; the first eight values are bit-identical with and without.

## VAL-only pre-checks (20 val units 163226054..163226073; never 99990101-99990400)

- **L1, offline on the block-9 traces:** (a) share of the hold label's variance (p, q banks) carried by ONE 9-bin input: today's `roll_err` / `pitch_err` ladders vs the L1 input on the dense ladder; (b) share of hold samples and of per-episode offsets that fall in distinct bins (resolution of the 0.1-0.5° band).
- **L2, offline:** the same two reads for the yaw label (r bank) and the yaw offset, today's centred ladder vs dense.
- **L3, on the rival scorer:** label law (MPCOF>rate, est + bias comp and oracle-fed) and the PID cascade, `off` vs `pi` vs `pole3`: pos, hold pos, hold dz, err, steady, stable, fails.
