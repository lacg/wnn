# Controller experiments inspired by Sajus Quadcopter-AI — attacking the 5° gap

**Goal:** close the WNN controller's 5° stability gap (currently ~80–83% stable /
~3.85° err vs PID ~98% / ~1.3°), targeting 100%-stable with better err.
**Source of ideas:** A. Sajus, *RL for the Control of Quadcopters*
(github.com/AlexandreSajus/Quadcopter-AI). His task ≠ ours (2D waypoint nav via
SAC/RL vs our 3D attitude-stabilization via DAGGER+GA), but his **wins came from
observation-engineering + action-shaping, not the policy class** — learner-agnostic,
so they apply to our imitation+GA approach.
**Status:** plan / not started. Testing waits for the active controller run to free
up (one-controller-at-a-time, XDS priority). See [[project-controller-quadcopter-inspired-hypotheses]].

**PID-reference provenance (traced 05/08/2026):** the "PID ~98% / ~1.3°" figure above is
`AttitudePIDConfig` (hand-tuned, unsourced) on the legacy synthetic plant (arm 0.075,
k_thrust 2.4, inertia [0.0023, 0.0023, 0.0046]) — gains and plant are at least matched
to each other, so the gap this document sets out to close is real *on that plant*. Both
the gains and the plant are unsourced; the plant matches no published vehicle. If any
hypothesis here is tested on a sourced airframe (`cf21_brushless`), the PID baseline must
be re-derived first — it does not retune itself the way LQR/MPC do. See
`controller_horizon_findings.md` §"PID-reference provenance" and
`l4_teacher_screen_results.md` §"PID-teacher tuning currency".

---

## The reframing
Our 5° gap is diagnosed as *precision-not-robustness* (0% divergence; soft fails
settle at a ~5.6° steady-state offset = missing integral, `state_universe=0`).
Sajus offers a second lens: **on this class of problem the ceiling is usually set by
obs/action shaping, not the learner.** His Run 2→3 was **+26% from un-clamping the
differential-thrust range** (action1 was saturating) — pure action-shaping. So before
(or alongside) growing the network/recurrence, test whether our gap is "can't act"
(saturation) or "can't perceive the error" (raw observations).

## Code grounding (verified 17/06)
- **Observations (`controller.rs`, NUM_FEATURES=9):** `gyro[0..3]` (raw body-frame
  angular vel), `accel[3..6]` (raw body-frame specific force), `target-RPY[6..9]`
  (raw target attitude). **No attitude-error feature, no integral-of-error.** The net
  must derive "distance to target attitude" from accel-vs-target itself.
  (NB `controller.rs:465` mentions an "integrator … offloads PID's integral term" —
  verify what already exists before adding.)
- **Output authority (`controller.rs`):** `decoded_to_delta(decoded, delta_max)` maps the
  net output → `[−delta_max, +delta_max]` (clamped), piecewise-linear, neutral at
  `NEUTRAL_DECODE`; then `pwm = 0.5 + delta_leak·(pwm−0.5) + delta`. So `delta_max` is the
  per-step correction authority and `levels_per_motor` the output resolution.

---

## H1 — Output saturation (cheapest; TEST FIRST, no GA) — ✅ DONE 17/06: NOT saturation → H2
**RESULT (`scripts/h1_saturation_check.py` on the magnitude-aware winner, delta_max=0.1):** in the
residual 1°–5° band the per-step delta pins at ±delta_max **0%** of the time (mean |delta|/delta_max
= **0.44**); even hard 9–11° episodes never saturate. The controller is **NOT authority-limited** —
the 5° gap is a perception/integral problem. → skip the H1-fix, **proceed to H2.** (Caveat: the
standalone rollout reproduced 46%/6.72°, not the 83%/3.67° held-out, because its IC sampling differs
from the evaluator's — the saturation signal is robust regardless. See the memory note.)


**Hypothesis:** the controller can't reach <5° because `delta_max` (per-step authority)
is too small, or the decode is too coarse near neutral to make fine corrections — Sajus's
saturating-action1, not a learning failure.
**Test (light, single rollout — no GA):**
1. Load a trained winner (`seed1@50/100 winner.yaml.gz`, or the magnitude-aware winner
   once done) via `checkpoint_io.load_controller_checkpoint`.
2. Run N≈10 held-out episodes; per step log `decoded` (net output), `delta`, and
   `|delta| == delta_max?` AND the live attitude error.
3. **Signal:** if `|delta|` pins at `delta_max` during the residual ~5° steady-state error
   → authority-limited (do H1-fix). If `delta` sits well inside the range while error
   persists → NOT saturation; the bottleneck is perception/integral (→ H2).
**Fix if saturating:** raise `delta_max`; and/or raise `levels_per_motor` for finer
near-neutral resolution (so small errors → small non-zero corrections instead of snapping).
Re-train a small probe and re-measure.
**Cost:** the test is a few seconds of rollout (instrumentation script, no GA).

## H2 — Error / integral observation features (highest payoff; H1 ruled out saturation)
**Hypothesis:** feeding the net the *error* directly (P) and a leaky integral of it (I) removes
the steady-state offset — Sajus's "angle to up" gravity-reference + error-relative features, and
our own missing-integral diagnosis. CONFIRMED-relevant by H1 (17/06): not authority-limited, so the
gap is perception/integral. NB the existing delta-control accumulator (controller.rs:465) is a leaky
integrator on the OUTPUT (throttle), not the ERROR — with delta_leak=0.95 it bounds steady-state
offset to 20·delta and pulls back toward hover; it does NOT zero attitude error.

### Design: CONFIGURABLE observation-feature set (user call 18/06 — mirror the fit-weights pattern)
Make the extra features a TOGGLEABLE set so ONE build supports an ablation study (P / I / P+I /
P+I+yaw), switched by config — not a hardcoded NUM_FEATURES=11.

**`ObsFeaturesConfig`** threaded Python→Rust like the C10 fitness weights:

| toggle | adds | stateful | #feat |
|--------|------|----------|-------|
| `tilt_p`    | tilt-to-vertical error (gravity ref, accel-only — no attitude estimator) | no  | +1 |
| `tilt_i`    | leaky integral of tilt error | yes | +1 |
| `peraxis_p` | roll/pitch/yaw error (yaw needs gyro-integrated heading) | yaw | +3 |
| `peraxis_i` | leaky integrals of the 3-axis error | yes | +3 |
| consts      | `integral_leak` (~0.99, DISTINCT from delta_leak), `integral_scale` | | |

Implementation contract:
- **NUM_FEATURES: `const` → dynamic `self.num_features` = 9 + enabled count** (max 17). The const is
  used in ~7 frame-sizing spots (controller.rs:518,872,994,1027,1224,1249) → each becomes the field.
  Hot-path; gets a cpu/gpu parity test.
- **Canonical append order fixed** `[tilt_p, tilt_i, roll_p, pitch_p, yaw_p, roll_i, pitch_i, yaw_i]`
  (present iff enabled) → deterministic bit-layout per config.
- **Integral state** = `Vec<f32>` sized to enabled stateful feats, zeroed in `reset()`; per step
  `acc = integral_leak·acc + err` (the I term as a SENSOR, not a control law).
- **Rust getter `get_last_feature_vector() -> Vec<f32>`** (len = num_features) feeds
  `fit_thresholds_from_pid_rollouts` (evaluator.py:123) so the quantile thermometer auto-calibrates
  the new bits — single source of truth, no Python re-derivation of the stateful integral. (Fitter
  drives an untrained WnnController's step()+getter during PID rollouts to gather distributions.)
- **Python**: flags `--obs-tilt-p --obs-tilt-i --obs-peraxis-p --obs-peraxis-i --obs-integral-leak`
  → ControllerSpec.obs_features (mirror fit-weights threading through phased_ga).

**Ablation matrix** (each = one ~33h retrain; A/B vs 83%/3.67° baseline on report-seed 99990101):
```
config            #feat  tests
(none) baseline    9     the current 83%/3.67° control
tilt_p            10     pure perception (P only) — "could it just not SEE the error?"
tilt_i            10     pure integral   (I only)
tilt_p + tilt_i   11     PI front-end  ← RUN FIRST
+ peraxis_p/_i  ≤17     full 3-axis incl. yaw
```
P-vs-I decomposition is the scientific payoff: if tilt_p alone closes it → perception; if it needs
tilt_i → genuine steady-state/integral.

**Sequencing constraint:** Rust change → `maturin develop --release` → FORBIDDEN while worker pid
2262 runs (shared wheel). Implement + rebuild + parity-test at a WORKER-IDLE WINDOW, batched onto the
same rebuild as the queued sparse-footprint steps 3/4/5 + H1 getter ([[project-resume-17jun]] IDLE-
WINDOW BATCH). Code edits can be written anytime; do NOT leave unbuildable Rust staged (proof-first).
To set integral_leak/scale + normalization, pull Sajus's `droneEnv` source (his dist/500, angle/180·π).
**Cost:** ~half-day Rust+Python for the configurable refactor, then retrains (serial, controller-free,
XDS priority) dominate wall-clock.

## H3 — Decouple outputs (common vs differential), per axis
Sajus's action0 (common thrust) + action1 (differential) split gives two orthogonal, simple
controls. Ours is per-motor PWM (4 motors). Test whether decoding common/differential per
axis (and giving the net those as separate output groups) simplifies learning. More invasive;
lower priority than H1/H2.

## H4 — Reward layering / single-axis curriculum
Sajus: dense survival bonus (+1/60) + shaped distance penalty (−dist/6000) + sparse task
(+100) + big terminal crash (−1000), "survive before precise." We're already 0% divergence so
survival-layering matters least. The transferable bit is the *single-axis curriculum* (master
one attitude axis → then 3) and the survive→precise ordering in the GA fitness schedule.

---

## Sequencing
1. **(now, safe)** plan + memory + the H1 instrumentation script (code only).
2. **(controller-free, ~2-3h out)** run H1 saturation check first → it routes the rest:
   - saturating → H1-fix (raise delta_max / levels), small probe.
   - not saturating → straight to H2 (error/integral obs), the likelier 5°-gap fix.
3. H2 retrain (pull droneEnv source for normalization) → H3/H4 as warranted.
**Constraints:** one controller at a time; don't disturb the running MEMORY stage; XDS
worker keeps priority. Report each probe's HELD-OUT (not gen-line) per
[[feedback_holdout_not_kfold_metric]] / [[project_controller_eval_variance]].

---

## Appendix — Sajus exact reward + observations (from `src/quadai/SAC/env_SAC.py`, 17/06)
For H2 normalization parity (his features are ERROR-relative + normalized — the lesson):
- **Reward (accumulated over 5 frames):** `+1/60` survive · `−dist/(100·60)` distance ·
  `+100` on target reached (`dist<50`, respawn) · `−1000` + done on `dist>1000`.
- **Observation (7, all relative/normalized):** `angle_to_up = a/180·π` · `velocity =
  √(xd²+yd²)` · `angle_velocity = ad` · `distance_to_target = dist/500` ·
  `angle_to_target = atan2(yt−y, xt−x)` · `angle_target_vs_velocity` · (distance dup).
- **Action→thrust:** `Tl = 0.04 + a0·0.04 + a1·0.003`, `Tr = 0.04 + a0·0.04 − a1·0.003`
  (a∈[−1,1]; the `0.003` differential is the Run-3 widened value that gave +26%).
**Takeaway for H2:** normalize our error feature (e.g. err/π) and integral feature
(leaky-sum/scale) before thermometer encoding, mirroring his dist/500, angle/180·π.
