# Horizon-bounded drift and committee drift-suppression in evolved weightless controllers

_Paper-feed document for controller paper #1 (02/07/2026). All numbers measured this
investigation; every winner passes the fresh-seed protocol before being believed.
Source runs: StateIntegral_20260701, E2Reliability_20260702, FrameFix*/LowEdge/BitSweep
winners; scoring via scripts/e4_best_of_k.py (Rust `eval_ensemble_closed_loop`, commit
468b0b3d) and scripts/pid_ki_ablation.py._

## Setting

3D attitude stabilization (rigid-body rotational dynamics, RK4 @1 kHz, '+'-quad motor
model, NO disturbances/noise — deliberately PID-favorable). Controller = two-layer
recurrent RAM WNN (thermometer-encoded gyro/accel/target, QSR-weighted PWM decode),
architecture + cells evolved by phased GA (grid → neurons → memory, Lamarckian,
DAGGER-trained against an AttitudePID teacher). Episode: random initial attitude
(tilt ≤5°, yaw ≤5°, body-rate ≤0.5), target level; **stable** = not diverged AND mean
attitude error ≤5°; **steady°** = mean error over the last 20% of steps (the
drift-sensitive metric). Held-out protocol: 4 report seeds × 100 episodes; **fresh-seed
protocol** (mandatory truth serum): 4 further seeds never used in training, selection,
OR reporting.

## Finding 0 — the I-term red herring (motivating negative result)

The WNN's ~15pp stability gap to PID was long attributed to a "missing integrator"
(soft-fails settle at a flat offset). Four independent integrator fixes failed
(delta-accumulator; observed error-integral features; pidmix PID-feature inputs;
direct recurrent-state integral target: B 83.6±4.0 vs control A 84.3±4.4). The
ablation that killed the theory: **the PID's own I-term is worth ≤0.06° at every
horizon** in this disturbance-free plant — PD-only holds steady error 0.00°:

| horizon (steps @1 kHz) | PID (P+I+D) stable / err / steady | PD-only (ki=0) |
|---|---|---|
| 500 (0.5 s) | 100% / 2.28° / 0.89° | 100% / 2.30° / 0.90° |
| 2000 (2 s) | 100% / 0.72° / 0.15° | 100% / 0.66° / 0.00° |
| 5000 (5 s) | 100% / 0.32° / 0.07° | 100% / 0.26° / 0.00° |
| 10000 (10 s) | 100% / 0.17° / 0.04° | 100% / 0.13° / 0.00° |

With nothing to integrate away, the WNN gap re-anchors as: PD-approximation quality
+ yaw observability (the WNN reads raw IMU; the PID reads the true quaternion) +
search reliability. This sets up Finding 1.

## Finding 1 — evolved lookup controllers are horizon-bounded (they drift)

Controllers trained on 500-step episodes look stable (84% held-out) but **never
learned to hold attitude**: past the training horizon, un-penalized slow drift modes
take over and the craft slowly tumbles. Training on 2000-step episodes produces a
genuine equilibrium — which itself decays past ~2.5× ITS horizon. The train×eval
matrix (fresh seeds, stable% / steady°):

| trained @ | eval @500 | @2000 | @5000 | @10000 |
|---|---|---|---|---|
| 500 (anchor A_ctrl) | 84.2 / 3.12° | 65.8 / 8.20° | **28.0 / 29.13°** | — |
| 500 (best single, pidmix) | 89.2 / 3.12° | 77.5 / 4.14° | 63.8 / 5.06° | — |
| 2000 (LONG) | 80.2 / 3.20° | 88.2 / 2.77° | 88.5 / 3.26° | **78.5 / 4.40°** |
| PID reference | 100 / 0.89° | 100 / 0.15° | 100 / 0.07° | 100 / 0.04° |

Mechanism: at 0.5 s, episodes end before drift is ever penalized, so the GA cannot
select against it — "stable" @500 measures transient recovery, not hovering. At 2 s,
drift bites inside the episode and selection eliminates it (LONG's steady = 2.77°,
tightest of any single controller) — but the immunity is finite: LONG decays at 5×
its horizon. There is also a symmetric specialization cost: LONG loses ~4pp on the
500-step transient metric (80.2 vs 84.2). **Every single-controller number reported
on one horizon is silent about the others; results must state the horizon triplet.**

## Finding 2 — committees are (approximately) horizon-free

A mean-PWM committee (each member's 4-motor command averaged per step, a few adders
on FPGA) of family-diverse members SUPPRESSES drift structurally: member drift
vectors are uncorrelated across observation families, so the vote cancels them —
while every individual member tumbles.

Committee ladder (fresh seeds; members all trained @500 except LONG @2000):

| config | @500 | @2000 | @5000 | @10000 |
|---|---|---|---|---|
| best single member | 89.2±5.4 | 84.2 (drifting) | 70.5 (drifting) | — |
| 5-member (all @500) | 90.5±3.7 | 93.8±2.3 | 92.5±2.6 | — |
| **6-member (+LONG)** | — | **95.2±2.6 / err 2.65°** | **93.0±1.2** | **92.0±2.5 / err 2.65°** |
| PID | 100 | 100 | 100 | 100 |

- At 10,000 steps (20× the @500 members' training horizon) the committee holds
  **92.0±2.5%** while its best member manages 78.5 and its @500 members are at
  28-70%. Decay ≈1pp per horizon-doubling; err FLAT at 2.65°.
- Aggregation rule: **mean beats median** (90.5 vs 89.8 @500) — no member ever goes
  wild (0% divergence since the June diagnosis); failures are precision, which
  averaging smooths and median cannot. Coherent with Finding 0's re-anchor.
- Membership: diversity beats strength. Two ~84.5% members IMPROVED the 3-member
  90.0 to 90.5 @500; LONG added +1.0pp and HALVED seed-SD @5000 (±2.6→±1.2). But a
  catastrophically bimodal member dilutes: the yaw-anchored ANCH (solo @2000:
  67.5±39.6, per-seed 0/90/100/80) dropped the committee 94.8→93.8.
- 7 members ≈ 5-6 (plateau): residual failure is common-mode (hard-IC pockets +
  yaw unobservability), which votes cannot fix.

**Headline claim: individual evolved WNN controllers are horizon-bounded; committees
of family-diverse WNN controllers are approximately horizon-free — drift suppression
is structural (uncorrelated-drift cancellation), not trained.** The full 6-member
committee remains KB-scale and needs only per-motor adders at inference.

## Supporting results

- **Fresh-seed protocol is load-bearing**: report-seed winners routinely fail it
  (pidmix_pwm family 87.2→76.5±16.8; ANCH 91.0→81.2±15.6 with a 55% crater seed;
  pidmix_s10's 90.0→89.2 reproduces — that's how we know which is real).
- **Lean extreme**: a 4-input-bit / 20-total-state-bit / 60K-cell s16 controller
  matches the full input budget @500 (85.0±4.3) and is the most drift-resistant
  @500-trained single (70.5 @5000) — committee member at negligible cost.
- **Random immigrants (0.15) alone are a −2.7pp tax** (81.6±4.7 pooled vs 84.3±4.4
  anchor): diversity of the search was not the binding constraint.
- **Action-repeat (N=5)**: no stability lift (80.2±6.6) but trains ~4× faster
  (24 min vs 73-93 min cells) — a compute lever, not a control lever.
- **Difficulty curriculum**: anchor parity (83.8±4.9) at 1/3 the state neurons
  (sn=13 vs 30-47) — an architecture-lean lever.

## Threats to validity / open items

- Single plant, clean sim (no wind/noise/motor asymmetry — disturbances are the
  planned follow-up and would give integral action real work for the first time).
- Winners are n=1 per recipe cell (2 seeds per arm; fresh-seed protocol guards
  selection luck but not recipe-level seed lottery).
- Committee members share the thermometer encoding — correlated-drift risk at
  horizons ≫10⁴ steps untested.
- C2K (in flight): pool of @2000-trained members across four families (incl.
  yaw-anchored ANCH2K) → committee of non-drifters; expectation 96%+ horizon-free.

## Provenance

ki=0: scripts/pid_ki_ablation.py (PID_STEPS ∈ {500,2000,5000,10000}). Matrices +
committees: scripts/e4_best_of_k.py (E4_STEPS/E4_ONLY/E4_ENSEMBLE_TOP/E4_SKIP_SOLO;
Rust hot loop `ram_controller.eval_ensemble_closed_loop`, ICs injected from the
numpy chain for exact fresh-seed reproduction). Winners: FrameFixVal/Bits_20260627,
LowEdge_20260701, StateIntegral_20260701, E2Reliability_20260702. Commits: c3a60914
(ki=0), 0882b19d (horizon drift), 468b0b3d (Rust committee eval).
