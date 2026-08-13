# SCOPE C stage 2, chunk B — the full-state teacher

Design record for the DAgger expert stage 2 needs. Companion to
`docs/scope_c_full_controller_spec.md`; that spec sets the goal, this records the
decisions taken while building it and WHY, so the choices are auditable later.

Status 13/08/2026: design fixed, implementation starting with the sim.

## What is missing today

| piece | today | stage 2 needs |
|---|---|---|
| sim translation | z, vz only (stage 1) | x, y, z + velocities |
| teacher | attitude-only (5 rivals) + stage-1 altitude PD | position → attitude reference |
| features | 18 (9 base + 6 + 3 vertical) | 22 (+ e_p x,y, e_v x,y) |
| metric | attitude triple | + position error in METRES |

The teachers today take `(q, gyro, target_rpy)` and emit 4 PWMs. They cannot
hover to a POINT because nothing in that signature carries where the vehicle is.

## Decision 1 — CASCADE, not a monolithic 12-state LQR

The teacher is an outer position loop handing an attitude reference + collective
to the existing inner attitude teacher.

**Why not monolithic.** `dare()` in `optimal.rs` is general, so a 12-state
Riccati is genuinely available and would be the "textbook" full-state answer. It
loses on three counts that matter here:

1. **It collapses the rival table to one.** The cascade keeps PID / LQR / MPC /
   LQI / MPCOF as five distinct rivals at stage 2, because each remains a valid
   inner loop. A monolithic LQR is one controller, and the comparison section
   gets thinner exactly where the claim gets larger.
2. **It is not what the rivals ARE.** Real autopilots — and the classical
   baselines this work is measured against — are cascaded. A monolithic LQR
   would be a rival we invented rather than the one the field runs.
3. **Near hover the two nearly coincide**, so the extra risk buys little. The
   horizontal dynamics linearize to a double integrator driven by tilt, which is
   precisely what the cascade assumes.

**The honesty requirement (unchanged from stage 1 item 6):** the teacher must be
DISCLOSED as a cascade wherever it is reported, never sold as monolithic. The
WNN is the monolithic one — that is the whole point of the comparison, and it is
also exactly Molchanov's framing (their POLICY is state → rotor thrusts; their
BASELINE is a classical cascade).

## Decision 2 — extend semi-implicit Euler to x, y; do NOT fold p,v into the RK4

The spec (chunk A) says "RK4 over 13 states (q, ω, p, v) not 7". The implemented
reality of stage 1 chose differently and documented why: attitude runs RK4 on
(ω, q) and stage 1 added (z, vz) as semi-implicit Euler OUTSIDE it, because the
z-dynamics is far slower than the 1 kHz step.

That reasoning holds *more* strongly for x, y — horizontal motion is driven by
tilt, so it is slower still than the vertical channel. And the coupling is
genuinely ONE-WAY: with only thrust and gravity modelled (no aerodynamic drag),
attitude dynamics depend on torques alone, never on p or v. Position is a pure
downstream integration of the attitude solution.

So folding p,v into the RK4 would change nothing about correctness, and would
force touching the attitude integrator — a LINEAGE BREAK for every banked
attitude result, in exchange for accuracy on the slowest states in the system.

**Decided: extend the stage-1 Euler pattern to x, y.** Attitude RK4 untouched;
stage-1 vertical behaviour bit-identical; the new horizontal states are new, so
they break no lineage. Revisit only if a measured integration error shows up —
and if drag is ever modelled, the one-way coupling argument dies and this must be
re-opened.

## Decision 3 — gains by SHAPE, reusing the altitude-PD derivation

`altitude_pd.rs` already sets its gains by choosing a closed-loop shape (ωn, ζ)
rather than raw numbers, then dividing out the plant's control effectiveness.
The horizontal loop uses the identical construction, which is why it needs no new
justification — only a new b.

Linearize horizontal motion about hover. With total thrust T ≈ mg supporting the
vehicle, a small tilt θ tips that thrust sideways:

    ẍ ≈ (T/m)·sin θ_pitch ≈ g·θ_pitch
    ÿ ≈ −(T/m)·sin φ_roll ≈ −g·φ_roll

so the control effectiveness of TILT on horizontal acceleration is simply

    b_xy = g          (m/s² per radian)

— no thrust coefficient, no mass, because the supporting thrust already equals
mg. Choosing the loop by shape,

    a_des = ωn²·e_p − 2ζωn·v

and inverting b gives the tilt reference

    θ_pitch_ref =  a_x_des / g
    φ_roll_ref  = −a_y_des / g

**Loop separation.** Attitude is the fast loop; altitude sits an order of
magnitude below it (ωn = 2.0, ζ = 1.0). Position must be slower still or the
two outer loops argue with each other, so the horizontal default is ωn = 1.0,
ζ = 1.0 (critically damped). One decade of separation per cascade level is the
standard rule and it is what keeps this stack from needing joint tuning.

**Tilt clamp.** θ_ref/φ_ref are clamped to a maximum tilt (default 30°). Real
autopilots all do this: the small-angle inversion above stops being valid as tilt
grows, and an unclamped position error would command a flip. The clamp is what
makes large position errors converge instead of diverge.

## Decision 4 — yaw stays a commanded reference, not a position product

The horizontal loop produces roll/pitch only. Yaw is independent of position for
a symmetric quad — you can hold a point at any heading — so yaw_ref stays whatever
the episode commands. This also keeps the yaw dead-reckoning story unchanged (see
`project_estimator_gap_scales_with_horizon`), rather than entangling it with a new
loop.

## Bar (pre-registered, from the spec)

Chunk B passes when the teacher achieves position error comparable to
Molchanov's classical baseline: 0.11 / 0.19 / 0.21 / 0.24 m across their
configurations. That is the TEACHER's bar, not the WNN's — the WNN's bar is set
against the teacher afterwards, exactly as the attitude arms are.

## Order of work

1. Sim: x, y, vx, vy — Euler, default-inert, Metal twin + two-sided parity.
2. Teacher: position outer loop → (roll_ref, pitch_ref) + collective; wraps ALL
   five inner teachers.
3. Measure the teacher alone against the bar above.
4. Only then: WNN features (+4), reward, episodes, metres in the metrics.
