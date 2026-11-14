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

---

## RESULTS (14/11/2026): implementation, verification, VAL-only pre-checks

**Every number below is VAL** (20 val units 163226054..163226073, the "both" s1-s3 headline genomes where a student is involved). No test unit was touched. Nothing here is a block-10 result: nothing was trained except a wiring smoke run.

### What was built

| lever | flag | default | where |
|---|---|---|---|
| L1 | `--obs-tilt-ref-err` | off | `tilt_ref_err.rs`, `controller.rs`, three kernel call sites, `cpu_score.rs`; Python `ControllerSpec.obs_tilt_ref_err` |
| L1 ladder | `--tilt-err-ladder {quantile,dense}` | dense (only with L1) | Python thresholds |
| L2 | `--yaw-centred-ladder dense` | no value = quantile = ARM 2, byte-identical | Python thresholds |
| L3 | `--alt-integral {off,pi,pole3}` | off | `altitude_integrator.rs`, rival scorers, `OuterLoops`, the student's `alt_err` input on CPU and in the Metal rollout kernel; Python `EpisodeConfig.alt_integral` |
| hold column | (always on, additive) | - | rival scorers return 10 values (9th hold pos, 10th hold \|dz\|), WNN rows 17 → 19; printed as `pos_hold=`, `dz_hold=` |

- `[provenance]`: `abp=on+tilt_ref_err`; `calib=...,yaw:dense,tilt:dense`; `alt=pole3` (the `alt=` token is absent when off, so older lines are unchanged).
- Controller ABI 51. The facade's expected ABI is 51 on this branch only.

### Verification
- `cargo test -p ram_controller --lib --no-default-features`: 387 passed, 0 failed, 25 ignored (final, after the last cosmetic change).
- Flags off: every existing pin and parity sweep unchanged; the pin digests hash the pre-ABI-51 metric prefix, pinned values untouched; explicit default-off tests added.
- L1: definition on a hand-built state; replay = live; CPU = Metal scorer (BINARY); split trainer parity; trainer walk parity in QUAD_WEIGHTED **and BINARY** (added).
- L3: the law, clamp, conditional integration, per-episode reset; rival scorer = `OuterLoops` on the same episode for the firmware PID **and for MPCOF, motor-level and cascade, all three modes** (added); the student's input is the law's input; CPU = Metal scorer.
- Python, against the built wheel unpacked in a scratch directory (never installed): flags off reproduce the block-9 ladders; flags on give the dense ladders, the reshaped `alt_err` ladder and the 19-row parse; a tiny end-to-end `phased_ga` run (pop 4, 8 episodes, non-report seeds 777001/777002) completes with all three flags on. That run is a wiring check, not a result.
- Installed wheel after all of this: still ABI 50.

Not verified / known limits:
- The hold-window convention differs when episodes DIVERGE: rival scorers average over episodes that reached the window, WNN rows over all episodes. Identical whenever nothing diverges (true of every MPCOF and student row so far; fails there are soft). Not aligned yet; see decisions.
- On Metal a mixed population (some controllers with L1, some without) is not refused; the kernel reads the first controller's setting. Every run builds its population from one spec, so this cannot occur in a run.
- `yaw_err_i` dense cuts land where the old ones were (first cut 3.49 vs 3.17 in its units), so L2 changes `yaw_err` only in practice.

### Pre-check L1 (VAL, offline on the block-9 student flights; share of variance one quantised input can carry)

```
quantity (VAL)                                              | s1    | s2    | s3
------------------------------------------------------------+-------+-------+------
ROLL hold label carried by today's roll_err (9 bins)        | 0.290 | 0.104 | 0.130
  ... by roll_err x abp (81 cells)                          | 0.345 | 0.198 | 0.236
  ... additive in roll_err, abp, e, v (4 inputs)            | 0.732 | 0.721 | 0.713
  ... by the L1 input, dense ladder (9 bins)                | 0.806 | 0.816 | 0.826
  ... by the L1 input on quantile-style cuts                | 0.816 | 0.810 | 0.834
  ... by the L1 input, unquantised                          | 0.871 | 0.862 | 0.871
PITCH hold label carried by today's pitch_err               | 0.083 | 0.138 | 0.104
  ... additive in 4 inputs                                  | 0.663 | 0.698 | 0.701
  ... by the L1 input, dense ladder                         | 0.785 | 0.778 | 0.807
ROLL per-episode TRUE offset carried by today's roll_err    | 0.290 | 0.085 | 0.138
  ... by today's 4 inputs, additive                         | 0.813 | 0.810 | 0.818
  ... by the L1 input, dense ladder                         | 0.905 | 0.937 | 0.935
PITCH per-episode offset by today's 4 inputs / by L1 dense  | .78/.95 | .81/.91 | .79/.95
mean |offset| deg, roll / pitch                             | .43/.33 | .36/.34 | .36/.32
mean |offset| left inside one dense bin, deg, roll / pitch  | .10/.07 | .08/.08 | .08/.07
episodes inside the dense dead band (+-0.08 deg), roll/pitch| .11/.16 | .14/.16 | .16/.17
```
- **L1 is supported.** One L1 input carries 78-83% of the hold roll/pitch label; today's `roll_err` / `pitch_err` carry 8-29%, and today's four relevant inputs reach 66-73% only as a four-way additive combination.
- It leaves 0.07-0.10° of the 0.32-0.43° per-episode offset unresolved inside a bin.
- **The ladder choice is second-order for L1**: dense and quantile-style cuts carry the same share of the label (within 0.01). The dense ladder's case is the guaranteed ±0.08° / ±0.20° / ±0.45° cuts when the calibration distribution changes; the offline read cannot separate them.
- Per-episode check on the same flights: the label is −7.1 to −8.3 rad/s per rad of offset (R² 0.87-0.90); the student's output is −0.3 to −0.8. The student's stiffness against the tilt-reference error is 5-10% of its teacher's.
- **Correction to `docs/ctrl_gap_after_block9.md` §B.1.** "Cannot rebuild the reference" was too strong. The four inputs do carry 78-82% of the offset additively; the student does not use that combination. The mechanism and the lever stand.

### Pre-check L2 (VAL, offline) — NOT supported as specified

```
quantity (VAL)                                              | s1    | s2    | s3
------------------------------------------------------------+-------+-------+------
YAW hold label carried by yaw_err, today's centred ladder   | 0.047 | 0.066 | 0.084
  ... by yaw_err on the dense ladder                        | 0.048 | 0.085 | 0.074
YAW hold label carried by yaw_err_i (today)                 | 0.37  | 0.13  | 0.15
YAW hold label carried by cresid_z, the trim input (today)  | 0.70  | 0.68  | 0.77
per-episode TRUE yaw offset carried by yaw_err, today       | 0.884 | 0.920 | 0.902
  ... on the dense ladder                                   | 0.864 | 0.901 | 0.872
mean |yaw offset| left inside one bin, deg, today / dense   | .115/.105 | .141/.131 | .124/.117
mean |yaw offset| deg                                       | 0.330 | 0.424 | 0.343
episode-mean yaw label vs student output, correlation       | 0.93  | 0.87  | 0.92
episode-mean (student - label) ~ k * yaw offset: k / R²     | +7.5/.54 | +8.1/.77 | +9.0/.81
```
- **Dense cuts on `yaw_err` change nothing** (offset left in a bin 0.12-0.14° → 0.11-0.13°; label carried unchanged). Today's ladder already tells the student its yaw offset to that resolution.
- **Correction to `docs/ctrl_gap_after_block9.md` §B.1 (yaw).** The parking is not an input-resolution problem of `yaw_err`.
- What the data say instead: the yaw label is trim plus a proportional term. The trim part is carried by `cresid_z` (68-77%) and the student reproduces it (correlation 0.87-0.93). What it misses is the proportional term, about −8 rad/s per rad of yaw offset. Same shape as roll/pitch: a small proportional correction riding on a larger term the table has to add it to.
- The yaw analogue of L1 would be one input that already combines the two (yaw error plus the trim it must ride on, or the trim-corrected yaw error). That is a new lever, not approved, not built.

### Pre-check L3 (VAL, rival scorer, 20 units × 100 episodes)

```
arm (VAL)                                  | stable % | err °  | steady ° | pos m | fails/2000 | hold pos m | pos at end m | mean |dz| m | hold |dz| m
-------------------------------------------+----------+--------+----------+-------+------------+------------+--------------+-------------+------------
MPCOF>rate(h10) est + bias comp, off       | 100.00   | 0.693  | 0.171    | 0.355 | 0          | 0.311      | 0.303        | 0.158       | 0.197
MPCOF>rate(h10) est + bias comp, pi        | 100.00   | 0.697  | 0.179    | 0.347 | 0          | 0.288      | 0.272        | 0.145       | 0.166
MPCOF>rate(h10) est + bias comp, pole3     | 100.00   | 0.702  | 0.155    | 0.309 | 0          | 0.220      | 0.201        | 0.070       | 0.046
MPCOF>rate(h10) oracle-fed, off            | 100.00   | 0.689  | 0.168    | 0.355 | 0          | 0.311      | 0.303        | 0.157       | 0.197
MPCOF>rate(h10) oracle-fed, pi             | 100.00   | 0.693  | 0.178    | 0.347 | 0          | 0.288      | 0.272        | 0.145       | 0.166
MPCOF>rate(h10) oracle-fed, pole3          | 100.00   | 0.700  | 0.152    | 0.309 | 0          | 0.220      | 0.201        | 0.070       | 0.046
PID>rate(h10) est-fed, off                 |  92.25   | 3.347  | 2.774    | 0.350 | 155        | 0.290      | 0.280        | 0.158       | 0.197
PID>rate(h10) est-fed, pi                  |  92.35   | 3.347  | 2.774    | 0.341 | 153        | 0.266      | 0.247        | 0.145       | 0.166
PID>rate(h10) est-fed, pole3               |  91.25   | 3.366  | 2.792    | 0.302 | 175        | 0.192      | 0.169        | 0.070       | 0.046
```
- **`pole3` removes the droop inside the 2 s episode**: hold \|dz\| 0.197 → 0.046 m, pos 0.355 → 0.309 m, hold pos 0.311 → 0.220 m, for the label law in both feeds. Steady is 0.016° better, err 0.009-0.011° worse, 0 fails.
- **`pi` is too slow for a 2 s episode**: hold \|dz\| 0.197 → 0.166 m, pos −0.008 m. Its slow pole has a time constant of about 2 s.
- The horizontal part is untouched in every row (\|dx\|, \|dy\| identical), as it must be: the position loop and the task are unchanged.
- **The PID cascade row gets 20 more fails under `pole3`** (155 → 175 of 2000; stable 92.25 → 91.25%). MPCOF rows are unaffected. The stiffer collective costs the weaker rival a little attitude authority. This is a val read on a rival, to be stated when the rival rows are re-flown.

### Wheel (built, NOT installed)
- `/Users/lacg/wnn/.claude/worktrees/ctrl-block10-levers/dist-abi51/ram_controller-2026.212.37-cp311-abi3-macosx_11_0_arm64.whl`
- sha256 `4a42cbfda8b800098424ac0de8184d6510662833c3e78c3c752f9ad5b1414fde`
- The Python smoke tests ran against the build before a final cosmetic change (a dead-code lint attribute on the probe types); the Rust behaviour is identical and the full suite was rerun on the final build.
- Install only when block 9 and the stress test have finished: `pip install --force-reinstall <wheel>` in the run venv, and merge this branch so the facade expects 51.

### Decisions for Luiz before block 10 is pre-registered
1. **L2.** Drop it from block 10 (recommended: the pre-check shows no effect), fly it anyway as a cheap arm, or replace it with the yaw analogue of L1 (new design, not built).
2. **L3 mode.** `pole3` (recommended: it does the job in 2 s) or `pi`.
3. **L3 means re-flying the control and the rivals.** With `--alt-integral` on, labels, the student's `alt_err` input and every rival row change. A block-10 comparison needs the `both` control and all three reference rows re-flown with the same setting; or L3 is run as its own arm against a matching control, separate from L1.
4. **Arm structure.** L1 and L3 act on different channels (attitude input vs collective law). Either a 2×2 (control, L1, L3, L1+L3) or L1 alone first with L3 held for a position-focused block. L1 alone is comparable with every existing rival row; anything with L3 is not.
5. **L1 ladder.** `dense` (default) or `quantile`; the offline read does not separate them.
6. **Hold-window convention** for diverged episodes: align the WNN rows to the rival scorers' ("episodes that reached the window") before block 10, or leave it. It matters only for rows with divergences.
7. **Expected gains to pre-register** (from the block-9 bounds, unchanged): L1 steady 0.80° → 0.5-0.65° (bound 0.49°); L3 pos 0.36 m → about 0.31 m and hold \|dz\| about 0.05 m for a student that learns the law.
