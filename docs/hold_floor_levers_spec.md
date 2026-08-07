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

## L3 — actuation granularity: the (`delta_max`, `delta_leak`) pair — **NEXT**

### Why this exists (and why it is now the leading candidate)

L1, L1b and L2 are all flown and all refuted (`docs/l4_teacher_screen_results.md`
§"Hold-floor levers"): making the disturbance observable made hold WORSE in 4/4
comparisons, handing it to a controller that provably cancels it roughly DOUBLED hold
error, and ranking explicitly on steady did not reliably move it. Neither more input, nor
a better substrate, nor a hold-targeted objective touches the floor ⇒ the floor is
**structural**, and the remaining structure is how the student ACTUATES.

The deployed students run **delta control**:

```
pwm ← 0.5 + delta_leak·(pwm − 0.5) + Δ        delta_control=True, delta_max=0.1, delta_leak=0.95
```

with the 17-value alphabet quantizing **Δ**, not the throttle (step = `delta_max/8` =
0.0125). Hold a constant Δ and the accumulator settles at

```
sustained offset = Δ / (1 − delta_leak)
```

so the **smallest non-zero sustained throttle offset** the student can express is

```
(delta_max/8) / (1 − delta_leak) = 0.0125 / 0.05 = 0.25 pwm
```

That is enormous relative to the corrections a 0.5° hold needs. The rollout trace shows
how the student evades it: in the steady tail **70–82% of steps emit a non-zero Δ with
mixed signs**, i.e. it duty-cycles the increment and the leaky accumulator averages it —
a ΔΣ-modulator. That buys effectively continuous actuation **at the price of a
limit-cycle ripple**, and the hypothesis is that **the ripple IS the hold floor**.

Both parameters set that ripple, and **neither has ever been searched**. Worse:

> ⚠️ **`--delta-max` does not exist.** `--delta-leak` is a `phased_ga` flag
> (`phased_ga.py:1596`), but `delta_max` is only a `ControllerSpec` field
> (`evaluator.py:122`, default 0.1) that `phased_ga` never passes — the plumbing
> `ControllerSpec → WnnController` is already there (`dagger.py`), only the CLI → spec hop
> is missing. **Step 0 of L3 is to add `--delta-max`**, a Python-only change. Until then
> `delta_max` has not merely gone unsearched, it has been unreachable.

### Design — two routes to the SAME granularity (the discriminating part)

Granularity falls if you shrink `delta_max` **or** if you make the accumulator leakier.
Pick the two arms so they land on the *identical* predicted granularity, 4× finer than
the control:

| arm | delta_max | delta_leak | smallest sustained offset | what it costs |
|---|---|---|---|---|
| **control** (already flown) | 0.1 | 0.95 | **0.25 pwm** | — |
| **A — finer step** | **0.025** | 0.95 | 0.003125 / 0.05 = **0.0625 pwm** | slew authority: max Δ per step drops 4× |
| **B — leakier accumulator** | 0.1 | **0.80** | 0.0125 / 0.20 = **0.0625 pwm** | integrator memory: offsets decay 4× faster |

**This is the point of the pairing.** A and B reach the same granularity by opposite
means and pay opposite prices:

- **Both improve steady, and by a similar amount** ⇒ granularity is the mechanism, and
  the hold floor is an actuation-resolution limit. That is the finding.
- **Only A improves** ⇒ it is not granularity but *slew authority* — the student was
  over-actuating in the hold window.
- **Only B improves** ⇒ it is not granularity but *integrator memory* — the accumulator
  was holding stale offsets through the steady window.
- **Neither improves** ⇒ the floor is not actuation either, which leaves **L4**
  (credit assignment) as the last standing candidate.

The control arm is the L4 screen's mpcof run — the SAME cell as L1b's `C10 + no d̂` — so
this costs **4 new runs** (2 arms × 2 seeds), not 6.

Control-arm numbers (MEMORY held-out, err / stable / steady):
`s31337002 1.21 / 100.0 / 0.64` · `s31337003 1.58 / 100.0 / 0.95`

Every other flag is COPIED from `scripts/l1_dhat_chain.sh`, including the 5-generation
NEURONS cap and C10 weights, so the only difference across the comparison is the delta
pair. Runs INTERLEAVED (both arms at seed 31337002, then both at 31337003) so the first
two already answer "did granularity move steady at all" and a dead arm can be culled
before the second seed is spent.

```
A:  --delta-max 0.025 --delta-leak 0.95
B:  --delta-max 0.1   --delta-leak 0.80
```

### Reading it

- **Steady is the primary**, as always — err is ~80% recovery term and recovery is
  already teacher-grade (0.88–1.21× per D1/D2).
- **Expect A and B to COST err.** Both reduce actuation authority in the transient; that
  is the bargain being tested. Report the FULL TRIPLE (err°/stable%/steady°) for every run
  and never declare a winner on one metric.
- **Watch `stable` on arm B specifically.** A leakier accumulator may fail to hold against
  the L4C sustained bias at all; a stability collapse there is informative (it says the
  integrator memory is load-bearing), not a failed run.
- **n=1 seed ranks nothing.** Say so until both seeds of an arm have landed.

### Success / refutation (pre-registered, same bar as L1)

- **SUCCESS:** steady drops below **~0.35°** on BOTH seeds for at least one arm — i.e.
  clears the 0.57–0.87 floor band by more than the seed spread.
- **REFUTATION:** steady stays inside the floor band on both seeds for BOTH arms ⇒ the
  floor is not actuation granularity, and L4 becomes the only remaining candidate.
- Either way, the A-vs-B contrast is reportable: it separates granularity from authority
  from integrator memory, which no previous experiment has done.

### Cost

4 runs × ~2.3–2.9 h ≈ **10–12 h**, plus the small `--delta-max` plumbing change and a
rebuild-free restart (Python only — no wheel, no worker swap).

---

## Order

L1 first (pure feature flag, control arm already flown, sharpest test of the
observability claim), L2 second (bigger architectural payoff + the deployability
argument, but its success criterion is more entangled), then L1b (the 2×2, which needs
both L1 and L2 finished for the box). **All three are now COMPLETE and REFUTED (07/08/2026)
— see `docs/l4_teacher_screen_results.md`.**

**L3 (above) is next**, and is the better-motivated of the two survivors: it targets the
one mechanism that directly sets hold RESOLUTION, and one of its two parameters has never
been reachable from a run at all.

**L4 (after L3): magnitude-weighted DAgger conflict writes.** During DAgger the teacher's
labels are written into RAM cells and address collisions are settled by vote tally (QUAD
nudging) — today every write counts equally. The state distribution is heavily imbalanced
toward near-hover, so in any collision the majority of near-identical low-|err| samples
outvotes the rare large correction that actually mattered. L4 weights each write by |err|.
The supporting signal is L2's DAgger trace: best iterations are 4–5 (mean_err 2.63°,
2.33°) but as β anneals to 0.008 and the student takes over, iteration 8 — the scored one
— degrades to 3.16°/2.82°. **Getting worse as it gains control is a credit-assignment
signature, not a capacity one.** L4 needs a Rust change (the write path is in
`dagger_train.rs`), so it is second on cost as well as on evidence.
