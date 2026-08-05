# L4 teacher screen — Crazyflie 2.1 Brushless, L4C

## STATUS: student numbers WITHDRAWN, rerun pending (05/08/2026)

The 05/08 screen (6 runs: lqr/lqi/pid × 2 seeds) and the MPC-family arm that followed it
were all flown on a plant with a **wrong moment arm**. `Airframe.arm_length` was fed the
published motor RADIUS where our '+'-config mixer needs the PER-AXIS moment arm
`L = 2a = radius·√2`, so roll/pitch authority was **0.7071×** the real vehicle's. Fixed
05/08/2026 (`axis_arm_from_radius`); derivation and the numeric check are in
`disturbance_param_sources.md` §"MOTOR GEOMETRY".

**Every student number from that screen has been deleted from this document** rather
than annotated, on Luiz's instruction: two sets of numbers for one experiment only
creates confusion. What they said is preserved in git history (commit b15bec91 and the
markers under `experiments/l4teach_markers/`) if it is ever needed. The MPC-family arm
was stopped ~40 min into its first run and produced no markers.

The defect was plant-level, so it hit every teacher identically and did **not** bias the
teacher ranking — but no number measured on it was faithful to the real brushless, which
is why the screen is being re-flown rather than rescued.

## Classical reference on the FIXED plant

Recomputed with `scripts/compute_baselines.py --airframe cf21_brushless --disturbance
L4C`, same protocol as before (tilt 5°, 5 report seeds × 100 episodes × 2000 steps,
sim-seed 911, fold 0). Scoring only, no GA — minutes. Canonical file:
`experiments/l4teach_markers/baselines_L4C_cf21bl.json`.

```
ctrl     stable%        err°         steady°
PID      100.0± 0.0   1.64±0.16    1.35±0.20     <- legacy tuning, see below
LQR      100.0± 0.0   0.93±0.04    0.42±0.07
MPC      100.0± 0.0   1.09±0.08    0.65±0.12
LQI      100.0± 0.0   0.81±0.03    0.36±0.03
MPCOF    100.0± 0.0   0.69±0.01    0.00±0.00
```

⚠️ **The PID row is "PID at legacy tuning".** LQR/LQI/MPC/MPCOF re-derive their gains
from the airframe via `calibrate_control_gains_rs` and are fully sourced. PID still runs
`AttitudePidRs::new_default()` — the retired hand-tuned constants — because the
firmware-sourced cascade (`wnn.control.pid_firmware`, gains from
`platform_defaults_cf21bl.h`) is not yet ported to Rust. Until it is, do not describe
this row as a tuned or sourced PID.

### The plant fix barely moved the classical baselines

Worth recording, because it sets expectations for the rerun. Pre-fix → post-fix err°:
PID 1.63646→1.63568, LQR 0.93258→0.93184, LQI 0.81337→0.81254, MPCOF 0.69274→0.69194,
MPC 1.03717→1.08800. Nothing is bit-identical (the plant genuinely changed), but four of
five moved by under 0.001° despite a 41% authority increase.

That is consistent with the pole analysis in §"PID-teacher tuning currency": in the
overdamped limit PID's dominant slow pole tends to `kp/kd` independently of loop gain, and
at 5° tilt under L4C the residual error is set by disturbance rejection rather than by
authority. MPC moved most (+0.051°) because it re-solves its QP against the new plant.

**Do not extrapolate this to the students.** A WNN student is far more sensitive to plant
detail than a closed-form controller — its thermometer encoding sees the state
distribution, and DAgger labels shift with the teacher's trajectories. The rerun is
justified; the expected delta is simply not known to be large.

## Rerun plan

1. ✅ Classical baselines on the fixed plant (above).
2. Port the firmware PID to Rust + Metal with CPU/GPU parity, so the PID arm is sourced
   rather than legacy-tuned. **Gate: no experiment uses `pid_firmware` before this.**
3. Re-measure the PID baseline row once (2) lands.
4. Closed-form arm: lqr / lqi / pid × 2 seeds.
5. MPC family LAST: mpcof / mpc × 2 seeds, `L4_NEURONS_GENS=5`.

Order set by Luiz (05/08): MPC family at the end. The 5-generation NEURONS cap makes that
arm **not budget-matched** to the closed-form arm (8-14 generations), so when its numbers
land: a capped MPC student that BEATS the closed-form mean is conclusive, one that loses
is ambiguous between teacher quality and search budget and must be reported that way.

## PID-teacher tuning currency — an uncontrolled variable, quantified (05/08/2026)

Kept because it explains why the PID arm needs the Rust port before it is re-flown, and
because its pole analysis predicted the baseline invariance above.

**LQR/LQI/MPC/MPCOF re-derive their gains from the airframe** — they call
`calibrate_control_gains_rs(dt, arm, k_thrust, k_drag, inertia, gravity, hover, 0.05)`,
rebuild the B matrix and re-solve for K. **PID does not.** Its gains are literal
constants in `AttitudePidRs::new_default()`, hand-tuned against the RETIRED synthetic
plant (arm 0.075, k_thrust 2.4, inertia [0.0023, 0.0023, 0.0046]) and unsourced.
`dagger_train.rs:589-595` flags this; the magnitude follows.

The sim integrates `tau = I·omega_dot` and thrust ∝ pwm², so the small-signal roll/pitch
loop gain about hover `p` is `G = 4·arm·k_thrust·p / Ixx`. With `u = kp·err − kd·rate`:

```
plant                        G (rad/s² per u)   omega_n    zeta   slow pole      tau
legacy (tuned-on)                 156.5         13.7      1.71    4.42 rad/s    227 ms
cf21_brushless (pre-fix arm)      665.8         28.3      3.53    4.08 rad/s    245 ms
cf21_brushless (fixed arm)        941.6         33.6      4.20    4.05 rad/s    247 ms
```

**The loop gain is 6.0x stale on the fixed plant — yet the dominant dynamics barely
move.** In the overdamped limit the slow pole tends to `kp/kd = 4.0 rad/s` regardless of
`G`, so the response the vehicle actually shows goes 4.42 → 4.05 rad/s. Only the fast
pole and zeta shift, both toward more damping. Separately, everything scaling with
`arm·k_thrust` is **exactly scale-invariant**: the I-term's trim torque
`ki·i_clamp·4·arm·k_thrust·p` and the L4C motor-asymmetry torque `asym·arm·k_thrust·p²`
hold a ratio of 1.00 on every registered airframe, and the authority clamp saturates at
the same 19.1° / 76.4°/s.

**So "PID lost because it was mistuned" is NOT supported** — its dominant mode is intact,
and the steady° column says the gap is a plant-model effect (PID 1.350° vs LQR 0.424 /
LQI 0.359 / **MPCOF 0.000**). What remains true is that PID is the only teacher not
re-derived for this airframe, so "weak teacher" and "teacher at another airframe's
tuning" are not yet separated. Step (2) of the rerun plan closes that.

Analysis script: `scripts/pid_provenance_check.py`.

## Provenance note

Run on the corrected `TeacherBank` (see `disturbance_param_sources.md` — pre-05/08 the
bank clamped teacher ids > 2 to PID, silently training lqi/mpcof as PID).

Airframe/gain provenance: `cf21_brushless` plant from Bitcraze firmware with the moment
arm corrected 05/08; LQ*/MPC* gains derived from it; PID gains still carried over from
the retired synthetic plant. The other controller docs
(`controller_horizon_findings.md`, `controller_quadcopter_inspired_experiments.md`,
`dfa1l_aligned_study.md`, `ceiling_pipeline_results.md`) all measure PID on that legacy
plant, where gains and plant *are* matched to each other — this screen is the only place
the two lineages meet.
