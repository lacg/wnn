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

## H1 — Output saturation (cheapest; TEST FIRST, no GA)
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

## H2 — Error / integral observation features (highest payoff for the 5° gap)
**Hypothesis:** feeding the net the *error* directly (and a leaky integral of it) removes
the steady-state offset — Sajus's "angle to up" gravity-reference + error-relative features,
and our own missing-integral diagnosis.
**Variants:**
- **H2a — attitude-ERROR feature:** add the geodesic attitude error (current attitude from
  accel/gyro vs target) as explicit feature(s). The net stops having to derive it.
- **H2b — integral-of-error feature:** add a leaky accumulator of the error as an observation
  (a manual integrator-as-input) → kills steady-state offset WITHOUT growing recurrence.
  Cheaper/safer than growing `state_universe`; complements it.
**Change:** add feature(s) in `controller.rs` (NUM_FEATURES grows; thresholds/encoder follow)
→ retrain (DAGGER + GA). **To set the integral leak/scale + normalization sanely, pull Sajus's
`droneEnv` source** for his exact obs normalization (distance/100, angle conventions).
**Cost:** Rust change + a controller GA run → needs controller-free.

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
