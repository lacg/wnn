# Hold-floor levers — L1 (d̂ feature) and L2 (residual on the firmware cascade)

Follow-up to `l4_teacher_screen_results.md` §"Where the error lives". The measured
mechanism: every student is teacher-grade in RECOVERY (0.88–1.21×) and hits an absolute
hold-attitude floor (steady 0.57–0.87°, cruise ~1°) independent of teacher quality. The
teachers that beat the floor carry integral action or an explicit disturbance observer.
These two levers give the student that instrument — L1 as an input feature, L2
architecturally.

Both are gated on the box: one controller at a time; the levels ablation finishes first.
L1 changes the `WnnController` ctor → **ABI bump → source+wheel must land atomically at
driver-idle** (the standing skew rule).

---

## L1 — d̂ disturbance-estimate input feature (`--obs-dhat`)

### Design

Port the mpcof teacher's observer INTO the student's feature extractor. The law
(`optimal.rs::update_dhat`, per axis):

```
u_applied = mixer⁻¹(pwm_prev)          # '+' mixer: u_roll=(m3−m1)/2, u_pitch=(m2−m0)/2,
                                       #            u_yaw=((m0+m2)−(m1+m3))/4
rate_dot  = (gyro − last_gyro) / dt
d̂        += l_gain · (rate_dot − b·u_applied − d̂)
```

Everything it needs is already inside `WnnController.step()`: gyro (input), its own
`pwm_prev` (state), `dt` (spec). New controller state: `last_gyro: Option<[f64;3]>`,
`dhat: [f64;3]`. New constants: `b` (from `calibrate_control_gains_rs` — same call the
teachers make), `l_gain` (default 0.05 = the teacher's; NOT searched initially).
`reset()` zeroes both.

Feature vector: **+3 features** (d̂ roll/pitch/yaw, rate-accel units), thermometer-
encoded like every other extra. Thresholds calibrate through the existing extras path
(`fit_thresholds_from_pid_rollouts` → `feat_ctl` reads `get_last_feature_vector`, the
single source of truth — no Python re-derivation).

Spec/flag: `obs_dhat: bool` on `ControllerSpec` + `--obs-dhat` on phased_ga, default
OFF (untouched recipes bit-identical). Touchpoints mirror `obs_yaw_err` exactly:
`controller.rs` (ctor + state + compute_features), `dagger_train.rs`, and the **Metal
kernel** (`controller_rollout.metal` computes features on-GPU — the d̂ recurrence is
per-episode-sequential, same as the integral features, so it fits the rollout kernel's
per-step loop); Python: spec/genome/evaluator/dagger/grid/phased_ga/reward_gated/
arch_adaptation. CPU/GPU parity: extend the existing feature-parity sweep (`cargo test
-p ram_controller`), mutation-test by flipping `l_gain`.

Why d̂ and not error-integral inputs: integral-error features measured DEAD against
blind S16 (project memory `integral-INPUT dead`). d̂ is a physically different signal —
estimated **external torque** from the model residual, exactly the term the student
cannot otherwise observe, and exactly what separates mpcof (steady 0.00°) from mpc
(0.65°) — that pair differs ONLY by the observer.

### Experiment + success criteria

- Arms: mpcof teacher × 2 seeds × {`--obs-dhat`, control} — the control arm is the
  finished screen (mpcof students: err 1.21/1.58, stable 100/100, steady 0.64/0.95).
  2 new runs, closed-form budget (~3 h each).
- Primary metric: MEMORY held-out **steady°** and the CRUISE phase from
  `scripts/transient_decomposition.py` (cruise is the floor's cleanest expression;
  err° carries the recovery term which is already teacher-grade).
- Success: steady drops below ~0.35° (the lqi-teacher classical hold) on both seeds,
  i.e. clears the 0.57–0.87 floor band by more than the observed seed spread; cruise
  ratio vs teacher falls from 3.3–4.3× toward ≤2×. err° below the 1.29° band is the
  headline if it follows.
- Refutation: steady stays in the floor band on both seeds → the floor is not (only)
  observability at the input; move attention to L4/L3 (credit assignment/actuation).

### Cost

Feature-flag pattern is well-worn (obs_yaw_err precedent): ~0.5 day including parity
tests and the ABI bump, plus 2×3 h runs.

---

## L2 — residual on the firmware cascade (existing task: "migrate the residual path")

### Design

The E5 residual hybrid already exists for the overactuated path (alloc-LQR baseline:
`pwm = clamp(base + clamp((wnn − neutral)·scale, ±rclamp), 0..1)` in `rollout_one`, the
Metal kernel, and `dagger.py`). The quad residual baseline, however, still calls the
LEGACY single-loop `pid_step`, and `_refuse_cascade_on_residual` (dagger.py:218) blocks
it on any airframe carrying cascade gains — deliberately, until the baseline speaks the
cascade.

Work items (the guard's own removal conditions):
1. `dagger.py` residual baseline → `AttitudePidFirmware` (Python side; the class and
   its 1e-12 golden tests exist since the 05/08 port).
2. Rust training/scoring residual baseline → `Teacher::PidFw` when the airframe carries
   cascade gains (`Teacher::pid_for_airframe` already implements the selection).
3. Metal residual branch → `pidfw_step` (the kernel function exists since the port;
   the residual branch just doesn't call it yet).
4. Plumb the cascade config from the airframe into the residual config (the af_pid_*
   fields already flow to every scorer — this is wiring, not new surface).
5. Delete `_refuse_cascade_on_residual`. The guard, not parity, is what gates this.

No ABI-visible ctor change expected (residual params already exist) → likely NO worker
impact and possibly no ABI bump; confirm at implementation.

### Experiment + success criteria

- Arms: residual student on the cf21_brushless cascade × 2 seeds, L4C, same budget.
- Baselines to beat, both required for a win:
  (a) the cascade alone: err 1.78° / stable 100% / steady 1.03° — the residual must not
      degrade its own baseline;
  (b) the direct-student band: err 1.29° / steady ~0.7°.
- Honest caveat, stated up front: the cascade's own hold (1.03°) is WORSE than the
  direct student's floor (0.57–0.87°) — the bet is not "the baseline holds better" but
  that integral trim + learned transient correction COMBINE: the cascade's I-term
  absorbs the sustained bias (removing the term the student cannot express) while the
  student supplies the recovery speed the cascade lacks. If the combination lands
  between the two parents instead of below both, that is a real and reportable negative.
- Deployment argument (independent of the win): the cascade is the shipped Crazyflie
  firmware loop — a residual on it is the only variant flyable on hardware without
  replacing the stock controller.

### Cost

Items 1–5: ~1 day (the Metal function and the golden-tested Python/Rust cascade all
exist; this is routing + guard removal + parity check), plus 2×3 h runs.

---

---

## L1b — S16 weights on delta × d̂ on/off (2×2), queued after L1 and L2

### Why this exists

L1 as launched ranks genomes by **C10** — `err² .40 / stable .30 / jerk .20 / mono .10`
— which contains **no steady term** (`--fit-weight-steady` defaults to `0.0` and must be
passed explicitly, `phased_ga.py:1925`). That is not an oversight: C10 is the winner of
the 18-combo, 3-round fitness-weight sweep concluded 13/06/2026 **on the delta
substrate**, which is what these runs are (`delta_control=True`, confirmed by the
06/08 rollout trace in `l4_teacher_screen_results.md`).

But the sweeps asked *"which weights minimise **err**?"*. L1 asks *"can d̂ break the
**hold** floor?"* — and a ranking with no steady term cannot preferentially retain a
steady-improving genome even if d̂ creates one. A null from L1 alone therefore cannot
separate "d̂ does not help hold" from "the search never looked for hold".

The other weight set on record is **S16** — `err .25 / steady .35 / stable .20 /
jerk .15 / mono .05` — winner of the 18-combo **absolute**-substrate sweep concluded
25/06/2026, where steady carries the LARGEST weight. It has never been tested on delta,
and that sweep's own finding was that **substrate dominates weights** (+14.2 pp
absolute-vs-delta, against only ~2.7 pp of spread across weight sets), so S16 does not
transfer for free.

### Design — 2×2, but only 4 NEW runs

|            | no d̂                                   | d̂                          |
|------------|-----------------------------------------|-----------------------------|
| **C10**    | ✅ flown — the L4 screen's mpcof arm     | ⏳ L1 (in flight)           |
| **S16**    | 🆕 isolates the weight change            | 🆕 the predicted cell       |

Two cells already exist, so this costs 4 runs × 2 seeds ≈ 4×3 h. The `S16 + no d̂` cell
is what makes the 2×2 worth running rather than a lone `S16 + d̂`: without it, a gain
cannot be attributed between the weighting and the feature.

Control-arm numbers (C10 + no d̂, MEMORY held-out):
`s31337002 err 1.21 / stable 100.0 / steady 0.64` · `s31337003 err 1.58 / 100.0 / 0.95`

Flags for the S16 cells (the rest copied from `scripts/l1_dhat_chain.sh`, including the
5-generation NEURONS cap that keeps every arm budget-matched):

```
--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 \
--fit-weight-jerk 0.15 --fit-weight-mono 0.05
```

### Reading it

- **Steady is the primary**, not err — err is ~80% recovery term and recovery is already
  teacher-grade (0.88–1.21× per D2), so a hold fix can move err by at most the ~20% the
  steady window carries.
- **Expect S16 to cost some err.** Trading err for steady is exactly the bargain the
  25/06 sweep accepted on the absolute substrate; on delta it is untested. Report the
  FULL TRIPLE (err°/stable%/steady°) for every cell and do not declare a winner on one
  metric.
- Two questions get answered at once: (a) does the S16 steady weighting transfer from
  absolute to delta, and (b) does d̂ help once the ranking actually rewards hold.

---

## Order

L1 first (pure feature flag, control arm already flown, sharpest test of the
observability claim), L2 second (bigger architectural payoff + the deployability
argument, but its success criterion is more entangled), then L1b (the 2×2 above, which
needs both L1 and L2 finished for the box). L3 (`delta_leak`/`delta_max` search) and L4
(magnitude-weighted DAgger) stay deferred until those verdicts.
