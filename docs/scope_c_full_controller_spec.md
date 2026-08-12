# SCOPE C — from an attitude inner loop to a Molchanov-comparable controller

**Decision (Luiz, 12/08/2026): do C.** And the framing correction that goes with it:
the WNN was always meant to REPLACE the controller (PID / LQR / MPC), not to be a
torque-only inner stage inside someone else's cascade. What exists today is the latter.
That is a course correction, not a failure — the inner-loop results stand — but the
target is a full flight controller. Not a staged retreat to "attitude-only is
standard scope" — that defence is true of classical cascaded autopilots and FALSE of
the learned-controller literature this project has chosen to be measured against.

## Why the scope has to change (the fact that settles it)

Molchanov et al. 2019 (arXiv:1903.04628) §III, verbatim:

> "We aim to find a policy that **directly maps the current quadrotor state to rotor
> thrusts**. The quadrotor state is described by the tuple (e_p, e_v, R, e_ω), where
> e_p ∈ R³ is the position error, e_v ∈ R³ is the linear velocity error … R ∈ SO(3)
> is the rotation matrix"

and §II: *"we do not use auxiliary pre-tuned PD controller in the learned policy."*
Their headline metric is **Euclidean position error in metres** (0.11 / 0.19 / 0.21 /
0.24 m across configurations).

Ours today: `(gyro, accel-derived tilt) → 4 motor pwm`. A strict SUBSET of their input
space, no position, no velocity, and we cannot emit their headline number. Task #5
("published learned baseline") is not a comparison until the scopes match.

**What stays true:** every attitude result already banked remains valid AS an inner-loop
result. C does not invalidate them; it demotes them from "the paper" to "a section".

## The end state

```
(e_p, e_v, R or tilt features, e_ω)  →  4 rotor commands
reported as: position error (m) + the project triple (stable% / err° / steady°)
```

## Three chunks, each independently testable

| # | chunk | contents | passes when |
|---|---|---|---|
| **A** | sim gains translation | position p, velocity v, mass; total thrust force F = R·[0,0,ΣTᵢ]; RK4 over 13 states (q, ω, p, v) not 7; Metal twin | a classical full-state controller hovers and holds position in it; drop test falls at g |
| **B** | teacher gains position | full-state LQR/MPC (today's teachers are attitude-only); the DAgger expert must be able to hover | teacher achieves position error comparable to Molchanov's classical baseline |
| **C** | WNN gains scope | +6 features (e_p, e_v) → address space grows; reward gains position; episodes gain position ICs; metrics report metres | the actual experiment |

### Training sim vs evaluation venue — the distinction (corrected 12/08)

An earlier draft said "we are not rebuilding AttitudeSim into 6-DOF". That was
overstated and wrong as written: chunk A REQUIRES adding translation, so our sim WILL
become 4-DOF then 6-DOF for TRAINING. DOF = degrees of freedom: 3 translational
(x,y,z) + 3 rotational (roll,pitch,yaw); today's AttitudeSim is 3-DOF (rotation only).

What is actually being avoided is investing in our simulator as a **validated,
general-purpose 6-DOF simulator serving as the EVALUATION VENUE**:

| | training sim (ours) | evaluation venue (gym-pybullet-drones) |
|---|---|---|
| must be | fast: GPU-parallel, 1 kHz, ~50 genomes x 5 folds x 100 episodes per generation | credible, standard, replicable by others |
| may be | approximate, provided it approximates the RIGHT things | slow — it runs once per winner |
| answers | "can the GA learn this?" | "would a reviewer believe it?" |

This is Molchanov's own pattern: a fast custom simulator for training, a neutral venue
(real hardware, for them) for the claim. Without a fast training sim the GA is
unaffordable; with our sim as the referee, "you wrote your own simulator" stands
unanswered. So: add the DOF training needs, keep the referee external.

### Mass and gravity are PLANT parameters, not features (Luiz, 12/08)

They go in the sim and are RANDOMIZED; they are never inputs. A controller does not
observe its own mass — it observes that it is sinking and pushes harder. The
thrust→Δz mapping depends on mass, battery sag, air density and wind, all unobservable
and all varying, so the controller must be ROBUST to them rather than INFORMED of them.
Molchanov does exactly this (thrust-to-weight randomized ~U(1.8, 2.5), never an input).

Consequence for stage 1: features are the ERRORS only (obs_alt_err, obs_vz) plus the
commanded collective. Mass joins the L4-style randomization axes so the learned policy
cannot silently depend on one value.

## Stage 1 — COMMANDED COLLECTIVE (the first slice of C, spec'd here)

### Why this first

The machinery already exists and is unexercised. `decode_outputs` has a
`decouple_outputs` mode that decodes the four banks as CONTROLS `[T, τ_roll, τ_pitch,
τ_yaw]`, and `mix_controls_to_motors` builds motors from a real collective `T`:

```rust
fn mix_controls_to_motors(&self) -> Vec<f32> {
    let (t, tr, tp, ty) = (self.pwm[0], self.pwm[1], self.pwm[2], self.pwm[3]);
    vec![(t - tp + ty), (t - tr - ty), (t + tp + ty), (t + tr - ty)]  // clamped
}
```

So the controller CAN command collective. What is missing is that **nothing ever judges
whether T is right**: thrust is quadratic, so T changes control AUTHORITY (and the GA
feels that), but with no gravity to fall against, T is never penalised for being wrong.
Our controller therefore has a hardcoded implicit hover of 0.5 and no idea what throttle
it is flying at — which is not what a real inner loop is either. A real inner loop is
HANDED a collective from above and adds torque around it.

Stage 1 is the smallest change that makes collective meaningful, and every later stage
depends on it.

### The feature budget (read from compute_features, 12/08)

Base is always 9 (gyro 3, accel 3, target 3); everything else is a toggle.

| # | feature | count | flag | in today's pidmix |
|---|---|---|---|---|
| 1-3 | gyro x,y,z | 3 | always on | yes |
| 4-6 | accel x,y,z | 3 | always on | yes |
| 7-9 | target r,p,y | 3 | always on | yes |
| - | tilt (angle-to-up) | 1 | obs_tilt_p | no |
| - | integral of tilt | 1 | obs_tilt_i | no |
| 10-11 | roll_err, pitch_err | 2 | obs_peraxis_p | yes |
| - | yaw_err (per-axis) | 1 | obs_peraxis_yaw | no (off) |
| 12-13 | integral roll, pitch | 2 | obs_peraxis_i | yes |
| - | pwm accumulator x4 | 4 | obs_pwm | no |
| 14 | yaw_err (clean scalar) | 1 | obs_yaw_err | yes |
| 15 | integral yaw_err | 1 | obs_yaw_err_i | yes |
| - | d-hat roll,pitch,yaw | 3 | obs_dhat | no |

**Today = 15.** Stage 1 adds obs_collective_cmd + obs_alt_err + obs_vz → **18**.
Stage 2 adds e_p(x,y) + e_v(x,y) → **22**. Each feature costs `bits_per_feature`
(8) input bits, so scope is paid for in address space — and "what does scope cost a
weightless controller?" is itself a reportable finding.

### The change

1. **Sim (chunk A, minimal):** add `z`, `vz`, `mass`. Vertical dynamics only:
   `v̇z = (ΣTᵢ·cos θ)/m − g`, where θ is tilt from vertical (so a tilted vehicle loses
   lift, which is the coupling that makes collective interesting). Position/velocity in
   x,y deferred to stage 2.
   - `mass` default = the cf21 airframe's, so hover T is a derivable constant, not a
     magic 0.5.
   - MUST be default-inert: `enable_translation = false` ⇒ bit-identical to today, same
     discipline as motor lag (assert on bit patterns, not epsilon).
2. **Feature:** `obs_collective_cmd` — the commanded collective from the outer loop,
   1 feature. This is what makes the controller COMPOSABLE: it can be driven by any
   outer loop, including a classical one, including pybullet's.
3. **Feature:** `obs_alt_err` (+ optionally `obs_vz`) — altitude error, so the
   controller can act on it.
4. **Episodes:** vary the commanded collective across episodes (a controller that only
   ever sees hover has not learned to work at other throttles), and vary initial
   altitude offset.
5. **Reward:** add an altitude-error term. Weight to be set by the same sweep discipline
   as the C10/S16 weights — NOT guessed.
6. **Teacher:** the attitude teachers need a collective channel. Cheapest correct
   version: an altitude PD on top of the existing attitude teacher, which is exactly
   the cascade the classical rivals use, and is honest as long as it is disclosed.
7. **Metal twin + parity test** for every one of the above, two-sided (matches AND
   differs-when-on), per the motor-lag precedent.

### Pre-registered read for stage 1

Bar: with `obs_collective_cmd` and a varying commanded T, the controller holds altitude
within X m while retaining its attitude triple (stable% / err° / steady°) within noise
of the fixed-collective baseline. X to be set from the classical teacher's own altitude
error on the same episodes — i.e. the bar is "does not lose to its own teacher by more
than the attitude arms already do", not an invented number.

**Disclosure that must ride along:** stage 1 is still not Molchanov-comparable (no x,y).
It must be reported as "altitude + attitude", never as "position control".

## Cost and consequence

- Stage 1: sim + 2-3 features + teacher collective + reward term + episodes + Metal twin
  and parity. Real work; days, not hours.
- Stages 2-3 (x,y, full-state teacher, metres-comparable protocol): the main programme.
- **Address space:** each feature costs `bits_per_feature` input bits. 15 → 21 features
  is a genuine cost for a LUT substrate and may itself become a finding worth reporting
  ("what does scope cost a weightless controller?").
- **Paper:** the claim moves from "a weightless inner-loop controller that fits on an
  FPGA" to "a weightless flight controller". Larger claim, more surface to defend, and
  the sn>0 footprint result (6-37x, docs/l4_teacher_screen_results.md) already says the
  hardware demonstration must use an sn=0 winner.
- **Deadline: NOT a forcing function (Luiz, 12/08).** IROS 1 Mar 2027 if it fits; if
  not, a later venue (IROS 2028 / L4DC / RA-L). The instruction is explicit: build the
  right thing first, then choose the venue. Do not trade scope for a date.

## Order of work

1. ~~motor lag~~ (done 12/08, awaiting install at the chain boundary)
2. **stage 1 commanded-T** (this spec)
3. pybullet harness bridges verified + first transfer numbers (task #4)
4. stage 2 x,y + full-state teacher
5. learned baseline in pybullet (task #5), now scope-comparable
