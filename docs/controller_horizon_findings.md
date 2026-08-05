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

### PID-reference provenance (traced 05/08/2026)

Every PID number in this document comes from `AttitudePIDConfig` — roll/pitch
kp 1.2 / ki 0.05 / kd 0.30, yaw kp 0.6 / ki 0.02 / kd 0.20, hover 0.5, axis authority
0.4 — mirrored in Rust by `AttitudePidRs::new_default()`. **These gains are hand-tuned
and cite no source.** What makes them usable here is that they were hand-tuned against
*this document's* plant (`AttitudeSim::new` defaults: arm 0.075 m, k_thrust 2.4 N/pwm²,
inertia [0.0023, 0.0023, 0.0046]), so the reference is at least matched to the vehicle
it flies. The provenance defect is therefore **the plant, not the gains**: that
parameter set matches no published vehicle (implied mass ~0.245 kg, 150 mm-class) — see
`disturbance_param_sources.md` §"THE REAL BLOCKER". WNN-vs-PID comparisons in this
document are internally valid; what they are not is *citable against a real airframe*.

⚠️ **Do not carry these PID numbers onto the sourced airframes.** Unlike
LQR/LQI/MPC/MPCOF — which re-derive their gains from the airframe via
`calibrate_control_gains_rs` — the PID's gains are literal constants and do **not**
retune. See `l4_teacher_screen_results.md` §"PID-teacher tuning currency" for the
quantified consequence on `cf21_brushless`.

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

## Finding 3 — the drift IS a yaw random-walk (mechanism, 02/07 W1 analysis)

Per-axis decomposition of drifting trajectories (scripts/w1_drift_analysis.py,
8 eps × 10,000 steps, fresh-seed ICs, 10 buckets):

| subject | roll / pitch (late window) | yaw walk | late yaw share |
|---|---|---|---|
| A_ctrl (t@500) | **0.15° / 0.36° — FLAT all 10k steps** | 1.9°→17.3° (~1.7°/1000) | **96.7%** |
| LONG (t@2000) | 0.67° / 0.15° — flat | 2.3°→4.2° (~0.2°/1000, 8× slower) | 82.5% |

Implications: (a) WNN roll/pitch control is ALREADY horizon-free — PID-class holding
on the observable axes; the "tumble" is unobservable yaw crossing the 5° stable line.
(b) Horizon training selects a slower yaw-walk RATE, not zero — hence immunity ∝ H,
never absolute. (c) Committee immunity is mechanistic: per-member yaw-walk directions
are uncorrelated → the mean vote cancels them. (d) The WNN-vs-PID comparison is
information-asymmetric: PID reads the true quaternion; absolute yaw is unrecoverable
from gyro+accel in principle. (e) **Registered prediction (before the cell runs):
ANCH2K (yaw-anchored obs + 2000-step training, in the C2K pool) yields a genuinely
horizon-free single controller.** _(Outcome: REFUTED — Finding 6.)_ Metric note: a roll/pitch-only stable% (or a
yaw-referenced task) would show the WNN as horizon-free today — the paper must state
which axes the metric charges for and what reference each controller receives.

## Finding 4 — search-reliability levers do not move the @500 ceiling (E2, 12/12)

Six isolate arms on the s16 base (2 recipe seeds × 4 held-out seeds, pooled n=8),
beat-me line = the no-immigrant anchor A_ctrl 84.3±4.4 @500:

| arm (lever) | pooled stable | Δ anchor | call |
|---|---|---|---|
| ANCH (yaw-anchor obs) | 87.9±6.1 | +3.6pp | only positive mean shift; NOT beyond SD. Per-seed 91.0 / 84.8 — no @500 bimodal collapse, but the 91.0 was the tail |
| LONG (2000-step train) | 88.4±3.7 *@2000-ruler* | n/a @500 (solo@500: 80.2/83.8) | wins its own ruler decisively (Finding 1) |
| GAMMA (hover-dense thresholds) | 82.2±3.7 | −2.1pp | refuted |
| IMM (immigrants 0.15) | 81.6±4.7 | −2.7pp | refuted (a tax, not a lever) |
| CURR (difficulty-adaptive) | 81.0±5.2 | −3.3pp | refuted for stability (arch-lean: sn=13) |
| REP (action-repeat N=5) | 80.6±5.7 | −3.7pp | refuted for stability (3-8× faster cells) |

**Verdict: no lever beats the anchor beyond SD on the @500 ruler, and no single
@500-trained arm breaks 90% pooled** — the 90+ single controller remains
unachieved at short horizon; only committees (90.5-95.5) have crossed it. All four
GA-reliability levers (immigrants, curriculum, threshold shaping, action-repeat)
are refuted as isolates. The two levers with real signal are **observation**
(yaw-anchor, +3.6pp mean) and **horizon** (2000-step training, Finding 1) — they
are mechanistically complementary (ANCH makes yaw observable; horizon makes drift
selectable-against), which is exactly the ANCH2K combination registered in
Finding 3(e) — and refuted when C2K ran it (Finding 6).

## Finding 5 — the input-bit floor is substrate-dependent (low-edge lean sweep)

Fixed grid sn=8, INPUT-bits swept {4,8,12,16} (grid_bits=input+8), folds=5,
2 seeds, pooled n=8 per point; sn floats free in the neurons GA. Bit-sweep anchor:
input-16 → 83.5±5.5.

| input-bits | s16 (raw states) | pidmix_pwm (19 PID features) |
|---|---|---|
| 16 | 82.6±5.4 | 71.0±12.0 |
| 12 | 84.8±3.1 | **79.8±8.0 — its peak** |
| 8 | 77.7±7.3 | 57.0±16.3 |
| 4 | **86.1±5.9 (best point)** | 52-60 — **collapsed** (seed10 52.0±16.4) |

**Raw-state s16 has NO input-budget floor down to 4 bits** — the curve is
flat-to-rising as input shrinks (the in8 dip is one weak seed cell, not a cliff),
and the in4 seed10 winner needed just **2,227 cells at 12 total state bits**. The
PID-feature substrate cliffs below ~12 input bits (57-60%, huge SD): derived
features must survive sampling to be useful, raw states are individually
informative. pidmix_pwm's sweet spot is NARROW — it peaks at input-12 and is
worse at BOTH in16 (71.0) and in8 (57.0). At exactly 12 input bits it can
occasionally reach s16 parity — its in12 seed10 cell (84.5±3.4, sn=22, 363K
cells) is the family's best of the sweep — but with the family's trademark seed
spread (seed09 twin 75.0; pool ±8.0 vs s16-in12's ±3.1), and at ~100× the memory
of the lean s16 point. (Sweep complete 03/07, 16/16 cells, both substrates.) Lean-FPGA implication: the input budget is nearly free on the raw
substrate; spend bits on state neurons, not observation width.

## Finding 6 — @2000 training generalizes ONLY the pwm observable; ANCH2K refuted (C2K, 8/8, 04/07)

All four families re-trained AT the 2000-step horizon (2 recipe seeds × 4 held-out
seeds, pooled n=8 per arm; ruler = 2000-step held-out; beat-me line = LONG@2000
88.2 / 2.77°, LONG_s09 on this ruler: 88.8±3.3):

| arm (obs family @2000) | seed09 | seed10 | pooled (n=8) | Δ ruler | call |
|---|---|---|---|---|---|
| **PWM2K** (pwm accumulator) | 91.8±0.8 / 2.8° | 89.2±3.3 / 3.1° | **90.5±2.7 / 2.8°** | **+1.7pp** | **only arm above the ruler; first single-controller recipe to pool above 90%** |
| LEAN2K (input-4 lean grid) | 93.5±1.1 / 2.9° | 82.5±7.2 / 3.6° | 88.0±7.5 / 3.2° | −0.8pp | ties the ruler; the 93.5 single record is seed lottery (see below) |
| TILT2K (lumped tilt) | 89.2±3.5 / 3.2° | 74.0±6.8 / 4.1° | 81.6±9.3 / 3.6° | −7.2pp | refuted — the seed09 "horizon rehabilitation" (tilt was −14pp @500) does not reproduce |
| ANCH2K (yaw-anchor) | 60.0±34.7 / 4.7° | 87.0±12.2 / 3.3° | 73.5±29.3 / 4.0° | −15.3pp | **registered prediction (Finding 3e) REFUTED** |

- **ANCH2K post-mortem**: the bimodality survives the horizon. Solo ANCH @2000 was
  67.5±39.6; ANCH2K pooled is 73.5±29.3 with the seeds split 60 / 87 — training at
  the horizon does not fix the yaw-anchor's coin-flip failure mode, it inherits it.
  The mechanistic complementarity argued in Finding 4 (observability + selectability)
  is NOT sufficient; whatever makes anchored recipes bimodal dominates. Anchor's
  remaining shot is as a committee member (ANCH_s09-@500 audition in the E4 assembly).
- **The 93.5 lesson (single-cell records are seed lottery)**: LEAN2K_s09 (93.5±1.1,
  108K cells) was briefly the best single controller ever measured here; its seed10
  twin collapsed to 82.5 on a near-degenerate 1,103-cell winner. Recipe-level claims
  need the 2-seed pooled number; per-cell bests belong in supporting material only.
- **MEM-stage regression is universal**: in every C2K cell the final ho-mem ≤ the
  ho-neur interim — from −1.8pp (PWM2K_s10, 91.0→89.2) to −30pp (ANCH2K_s09,
  89.8→60.0). The memory-refinement stage overfits the in-search episode pool in
  proportion to the recipe's variance; the interim ho-neur is a cheap early-warning
  signal.
- **Heavy elites lose**: every underperforming cell carried outsized endgame genomes
  (ANCH both seeds 0.9-1.1M cells mid-search; TILT2K_s10 peaked 1.21M), while both
  PWM2K winners stayed mid-size (254-284K) and the GA repeatedly re-trimmed them.
  Cell-count bloat during NEURONS is a live overfit indicator.

**Verdict: horizon training generalizes the pwm-observable recipe and nothing else —
PWM2K is the first recipe whose POOLED single-controller number (90.5±2.7) crosses
90% — but no single controller approaches the 6-member committee (95.2±2.6 @2000,
Finding 2). The @2000 story confirms the @500 story: observation family and horizon
are the only levers with signal, and the committee remains the only horizon-free
construction.**

## Supporting results

- **Fresh-seed protocol is load-bearing**: report-seed winners routinely fail it
  (pidmix_pwm family 87.2→76.5±16.8; ANCH 91.0→81.2±15.6 with a 55% crater seed;
  pidmix_s10's 90.0→89.2 reproduces — that's how we know which is real).
- **Lean extreme**: a 4-input-bit / 20-total-state-bit / 60K-cell s16 controller
  matches the full input budget @500 (85.0±4.3) and is the most drift-resistant
  @500-trained single (70.5 @5000) — committee member at negligible cost. Now
  pooled across 2 seeds: 86.1±5.9, the best point on the lean curve (Finding 5);
  the seed10 twin is even smaller (2,227 cells @ 12 total state bits).
- **Random immigrants (0.15) alone are a −2.7pp tax** (81.6±4.7 pooled vs 84.3±4.4
  anchor): diversity of the search was not the binding constraint.
- **Action-repeat (N=5)**: no stability lift (80.2±6.6) but trains ~4× faster
  (24 min vs 73-93 min cells) — a compute lever, not a control lever.
- **Difficulty curriculum**: anchor parity (83.8±4.9) at 1/3 the state neurons
  (sn=13 vs 30-47) — an architecture-lean lever.

## Finding 7 — the horizon surface: own-ruler band, monotone training gains, record committees (W1 + E4 chain, 05-06/07)

**Setup.** W1 completed the training-horizon surface (H ∈ {500,1000,2000,4000} × 2 seeds,
recipe s16+immigrants, own-horizon rulers, 4-seed report held-outs), then the E4 chain
re-measured everything on honest common rulers: (A) each surface winner solo at
{0.5,1,2.5,5,10,20}× its own H on FRESH seeds; (B) fresh-seed truth serum @2000/@5000
over the C2K pool + W1 winners; (C) six mean-PWM committee panels @2000/5000/10000.

**7a — the own-ruler surface is a band; the common ruler says training-H is monotone.**
Own-ruler ho-mem: 81.6±4.7 → 84.5±3.7 → **88.4±3.7 (peak @2000)** → 84.0±7.1 — a
sweet-spot band. But on the FIXED 2000-step ruler the same winners rank
H500 ~80.5 < H1000 ~84.0 < H2000 ~88.2 < **H4000 91.4 (93.8/89.0)** — training horizon
helps MONOTONICALLY; the "band" is the eval ruler getting harder with H, not the
training value declining. Two separable laws where W1 alone saw one. (Also: Finding 6's
"MEM-stage regression is universal" is C2K-@2000-specific — both W1 H1000 cells and
H4000_s10 GAINED in the memory stage.)

**7b — solo decay curves: hold to ~2.5× trained H, then cliff (with a sublinear top).**
All 8 winners plateau through 2.5×H and cliff by 5×H (H500 cliff ≈ 2.5k abs steps,
H1000 ≈ 5k, H2000 ≈ 10k — cliff ≈ 5×H); the H4000 pair cliffs at 2.5× (≈10k abs, the
same absolute ceiling as H2000 — endurance payoff saturates ≈10⁴ steps). Failure
texture is seed-determined and binary: h4000_s09 keeps one fresh seed flying at 87%
@80k steps while another dies at 0 from 5×; h4000_s10 decays uniformly (all seeds
together). Drift rate is a per-genome property drawn at training time.

**7c — truth serum: first fresh single >90; PWM2K confirms; LEAN/ANCH formally dead solo.**
**w1_h4000_s09 @2000 = 93.8±2.9 (per-seed 91/95/98/91) — best single ever on the honest
protocol**, beating the old 5-member committee (90.5±3.9); still #1 @5000 (89.0±9.2).
PWM2K reproduces (89.0±4.6 / 88.0±1.4 — no crater). LEAN2K craters fresh (93.5→82.2,
82.5→66.0 — the seed-lottery verdict now includes fresh-seed non-reproduction). ANCH2K
stays bimodal on fresh seeds (72.2±40.2 with per-seed 3/99/99/88; 66.0±24.9) — dead as
a deployable.

**7d — committees: two records; heterogeneity is the anti-inversion lever; duplication is poison.**

| panel (mean-PWM) | @2000 | @5000 | @10000 |
|---|---|---|---|
| C6_prod (5-member + LONG_s09) | 95.2±2.6 | 94.5±1.1 | **92.0±2.5** |
| C7_long (+LONG_s10, dup recipe) | 95.5±1.8 | 93.8±1.1 | 87.2±7.0 (inversion) |
| C7_pwm2k (+pwm2k_s09) | 95.0±3.0 | 95.0±1.2 | 90.8±3.6 |
| C7_anch (+ANCH_s09 audition) | **96.0±1.2 ⭐record** | 94.8±1.1 | 90.2±3.5 |
| C8_pwm2k_w1 (+pwm2k_s09+w1_h4000_s10) | 95.0±2.5 | **96.5±0.5 ⭐⭐record** | 91.5±4.4 |
| C8_2xpwm2k (+both PWM2Ks) | 94.8±2.5 | 93.2±2.2 | 86.0±9.4 (worst) |

- **ANCH audition PASSED**: the catastrophically-bimodal solo is the best 7th vote
  (record 96.0@2000, no dilution anywhere) — "differently-blind members" beat
  "more of the best member".
- **C8_pwm2k_w1 = 96.5±0.5 @5000, the program's all-time best number at any horizon**
  (near-PID consistency); the panel carrying a W1 H4000-trained member excels exactly
  at long horizons (7a's law composing into committees).
- Same-recipe duplication inverts at @10000 every time (C7_long 87.2, C8_2xpwm2k 86.0
  vs C6_prod 92.0) — correlated walks, now shown within the strongest family too.
- Plain C6_prod remains the @10000 champion: adding members buys mid-horizon, not
  far-horizon.

**Consequences.** (1) --steps 2000 stays the training default for single-recipe
cost/benefit, but trained-@4000 genomes are the committee's long-horizon organ;
(2) deployment menu: single = w1_h4000_s09 (93.8@2000), ≤5 s missions = C7_anch
(96.0@2000), 5-10 s = C8_pwm2k_w1 (96.5@5000), unbounded = C6_prod (92.0@10000);
(3) next stable-point candidates: a panel including w1_h4000_s09 itself (the 93.8
single was in NO panel — all were designed before its result landed), and W2
disturbances for the first honest PID-vs-PD separation.

**7e — addendum (06/07 afternoon): the w1_h4000_s09 panels — decorrelation beats quality.**
Three follow-up panels @2000/5000/10000 (W20Panels_20260706): ADDING the 93.8 single
is fine (C7_w1s09 91.8±0.8 @10000 — tightest long-horizon committee ever, statistically
tied with C6_prod) and C8_w1s09_pwm2k ties the @10000 record (92.0±3.6); but SWAPPING
it in for the humble A_ctrl (84.3 solo) CRATERS the committee at @10000 (82.5±9.6,
−9.5pp vs C6_prod) — removing the only structurally-plain member leaves three
s16-family horizon-trained members whose walks correlate. The duplication lesson
operates at the FAMILY level: a member's committee value is decorrelation first,
solo skill second. No panel beat the 95-96 @2000/@5000 saturation — the mean-PWM
construction appears ceiling-limited there; remaining gains are variance reduction.

## Finding 8 (seed) — W2.0 disturbance calibration: the first honest PID-vs-PD separation

Torque-domain weather (D1 bias / D2 OU gusts / D3 motor asymmetry / D4 IMU noise;
commit 59c824c6, Rust+Metal bit-exact hash RNG) calibrated on fresh seeds
(W2Calibrate_20260706, two rounds): steady-state offsets scale LINEARLY with bias
(plumbing verified), and the **stock PID integrator is nearly cosmetic** — ki=0.05
with i_clamp=0.5 trims only ~26% of a constant-torque offset (max I contribution
0.025 vs the ~0.06 the bias demands). With a working integrator (PID+ = ki×4,
clamp×4) the v2 ladder ({5,15,30}% of max control torque) hits all three targets
@2000: **L2 = PID+ 99.8 / stock PID 97.0 / PD 84.0 (+15.8pp)** — the integrator's
value finally measurable on a stability ruler; L3 = PID+ 27.0 / PD 2.2 (stress
tail). These are the new anchors for every weather table.

**W2.2 brittleness audit (06/07, W22Brittleness_20260706) — clean-trained WNNs are
NOT PD-like; they are catastrophically off-distribution.** Clean → L1 → L2 @2000,
fresh seeds: w1_h4000_s09 93.8→72.0→0.0; pwm2k_s09 89.0→80.2→0.2; pwm2k_s10
88.0→25.2→0.0 (the tightest clean performer collapses hardest — its ±1.4 was a
narrow attractor, not robustness); e2_long_s09 88.2→49.0→0.0; C6_prod 95.2→86.5→0.0;
C7_anch 96.0→84.0→0.0. Where PID/PD/PID+ all hold 100% (L1) the WNNs lose 9-63pp;
where memoryless PD still flies 84% (L2) every WNN is at ZERO — a learned lookup
policy exists only on its training distribution, while an analytic law is valid
everywhere. Committees soften L1 (members 25-80 → panel 84-86.5: averaging cancels
uncorrelated weather response) but cannot fix L2 (no member has integral action —
common-mode blind spot). No evidence of implicit integral action anywhere.
**Consequence: W2.3 train-under-weather is the load-bearing experiment** (gate: beat
clean-trained-under-L1's 80.2 without losing the clean score; running as of 06/07
14:21Z — PWM2K recipe @2000 + L1, 2 seeds, --disturbance CLI in phased_ga 30045566).

**W2.3 L1 arm — GATE MET (07/07 02:27Z, W23Weather_20260706).** Training WITH L1 in
every rollout (PWM2K recipe @2000, 2 seeds): MEMORY 4-seed ho UNDER L1 = s09
**93.5±2.7 / 2.92±0.17°**, s10 89.2±2.5 / 3.26±0.13° (pooled ≈91.4±3.4) — both
seeds individually beat the 80.2 gate; s09 lands at the clean-era champion's CLEAN
level while flying in weather. Stage anatomy: s09's NEURONS sat flat at the 60% grid
elite for all 15 gens and MEMORY delivered the entire 60→93.5 lift (the clean-derived
architecture was already adequate — W2.2's "brittleness" was a training-DISTRIBUTION
problem, not a capacity problem); s10's NEURONS did find signal (60→88, three
improvements) before MEMORY consolidated — arch-search-under-weather is
seed-dependent, memory-rewriting is not. Fresh-seed verification matrix @2000
(e4, dist printed from live config):

| candidate | trained in | clean | L1 | L2 |
|---|---|---|---|---|
| w23_pwm2k_L1_s09 | L1 weather | 86.2±4.8 | **90.2±4.2 / 2.97°** | **57.2±6.1** |
| w23_pwm2k_L1_s10 | L1 weather | 58.5±16.6 | 78.0±9.1 | 1.0±1.2 |
| mean-PWM ensemble (2) | L1 weather | 87.8±8.5 | 90.2±6.4 | 33.8±11.6 |
| pwm2k_s09 (C2K twin — same recipe+seed) | clean | 89.0 | 80.2±5.1 | 0.2 |
| w1_h4000_s09 (clean-era champion) | clean | 93.8 | 72.0±11.6 | 0.0 |
| PD (memoryless analytic) | — | 100 | 100 | 84.0 |
| PID+ (working integrator) | — | 100 | 100 | 99.8 |

(Clean-trained rows from the W2.2 audit above; analytic rows from the W2.0 v2
ladder. The one-view story vs the clean-trained twin: pay −2.8pp in calm air,
gain +10pp at L1 and +57pp at L2.)

Three results: (1) s09 fresh under L1 = 90.2 vs clean-trained best 80.2 (+10pp) at a
~3pp clean cost (86.2 vs pwm2k_s09's 89.0 — within 1σ) → **gate met on both halves**;
(2) s09 flies BETTER in its training weather than in calm air (90.2 L1 > 86.2 clean)
— the distribution-match signature, strong evidence the collapse mechanism in W2.2
was distributional; (3) **L1-training transfers PARTIALLY to L2: 57.2 where every
clean-trained WNN is 0.0** — bias-compensation machinery is learnable and
extrapolates to 3× the trained bias, but stays under memoryless PD's 84. PID err
ruler: s09 2.97° vs PID ~0.95-1.4° at L1 — stability gap closed, precision gap
remains. → **L2 training arm launched 07/07 02:53Z** (W23WeatherL2_20260707, same
recipe, --disturbance L2, 2 seeds): does a distribution that DEMANDS integral action
make the GA discover it (beat PD's 84 @L2)? This is E5's go/no-go.

**W2.3 L2 arm — DECISIVE NEGATIVE (07/07 15:25Z, W23WeatherL2_20260707). From-scratch
L2 training does NOT discover integral action, and is COUNTERPRODUCTIVE vs training one
level down.** MEMORY 4-seed ho UNDER L2: s09 2.8±2.2, s10 16.5±2.3 (pooled ~9.6).
Fresh-seed matrix @2000:

| L2-trained | trained in | clean (OFF) | L1 | L2 (own level) |
|---|---|---|---|---|
| w23_pwm2k_L2_s09 | L2 weather | 11.5±4.3 | 28.8±16.6 | 1.0±0.7 |
| w23_pwm2k_L2_s10 | L2 weather | 55.0±26.1 | 63.0±8.1 | **19.5±7.8** |

The killer comparison: **L1-trained @ L2 = 57.2 beats L2-trained @ L2 = 19.5 (best seed) by 3×**
— the controller trained ONE LEVEL DOWN is the better L2 controller. And both L2-trained
winners fly BETTER in milder weather than they trained in (s10: OFF 55 / L1 63 / L2 19.5),
the exact inverse of the L1 arm's "flies best in its own weather" signature. Mechanism:
under L2 the initial population barely flies (in-search 0-9% stable, both seeds plateau at
the grid elite through all 15 NEURONS gens; s09 MEMORY stuck 4→5%, s10 MEMORY lifted 9→25
in-search but only 16.5 ho) — the GA/Lamarckian search only ever sees FAILURE trajectories,
so there is no gradient toward integral action. L1's flyable population (60-88%) is what let
the search find control structure that then partially transfers UP. **Registered answer to
the headline question: NO — a distribution that demands integral action does not, by itself,
make from-scratch GA+memory discover it. You need a learnable curriculum.** This is a GO
signal for E5 (residual hybrid / L1→L2 curriculum fine-tune), not a kill: it pinpoints
exactly where the pure-WNN-from-scratch approach hits its wall (the memoryless-PD ceiling
of 84 in the integrator zone) and why a residual analytic-integral term or a curriculum is
the right architectural answer. W2.3 verdict overall: L1 ✅ gate met (distribution-match
robustness is learnable and beats clean-trained by +10pp@L1/+57pp@L2); L2 ❌ from-scratch
fails (integral action is NOT emergent under harsh-only training). Provenance:
W23WeatherL2_20260707 (2 cells, marker 15:25Z), rescore/ matrix via e4_best_of_k.py
(w23_pwm2k_L2_s09/s10 registry entries).

## Finding 9 — E5 L1→L2 curriculum: transfer is architecture-conditional, none reaches PD

E5 fine-tunes each L1 winner under L2 (--seed-winner warm-starts arch+cells+FULL population).
Three recipes × 2 seeds, all @L2 held-out (from-scratch=19.5, L1-transfer≈57.2, PD=84, PID+=99.8):

| Recipe (arch freedom) | s09 | s10 | note |
|---|---|---|---|
| raw L1→L2 transfer (frozen, no train) | 48.2 | **0.2** | s09's L1 arch transfers, s10's does NOT |
| frozen-arch = CONNECTIONS (rewire+DAGGER-retrain, neurons+bits frozen) | 37.2 | **56.2** | **s10 RESCUED 0.2→56.2 by re-wiring alone** |
| neurons+memory (full arch re-search, FRESH-eval) | 29.0 | 63.2 | pooled 46.1 |

**Headline: curriculum value is CONDITIONAL on architectural transferability.** s10's frozen L1
architecture is non-transferable to L2 (0.2%) but recovers to 56.2% under CONNECTIONS
(synaptogenesis: neuron-count + bit-width frozen, only connectivity mutated + Lamarckian cell
retrain) — it needed **re-wiring, not new neurons**. s09's L1 arch already transfers (48.2%) and
ANY retraining under L2 **degrades** it (37.2 connections, 23.8 after the value-GA MEMORY stage).

**Two methodological caveats surfaced:** (1) the MEMORY stage is a value-GA on `score_genomes`
(NO DAGGER — cells ARE the genome) so it CANNOT adapt a frozen population to a new disturbance;
a literal "--seed-winner-stage memory" run was a verified NO-OP (winner cells byte-identical to
the L1 seed, MD5-confirmed) and only measured raw transfer. The L2 cell-adaptation lives in the
Lamarckian ARCH phases (NEURONS/CONNECTIONS via _lamarckian_evaluate_batch). (2) During-search
held-out is optimistic vs fresh rescore (E5.2 neurons+memory 56.2 during-search → 46.1 fresh;
s09 −17.8pp), so trust fresh-seed rescores for cross-recipe claims.

**Verdict: E5 curriculum beats from-scratch (19.5) and MLP (26.7) but none of the three recipes
clears the memoryless-PD ceiling (84) — E5-residual-hybrid is GO** (analytic PD baseline +
learned WNN residual supplying the integral action; plan .claude/plans/e5_residual_hybrid.md,
both-baseline ablation PD+stock-PID, learn-the-clamp per-axis). Provenance: E5Curriculum_20260707
(neurons+memory, 2 cells, marker 08/07 00:11Z), E5MemOnly_20260707 (no-op transfer probe),
E5FrozenArch_20260707 (CONNECTIONS→MEMORY, 2 cells, marker 08/07 07:00Z); --seed-winner-stage
flag commits 786300df + 95aba01e; rescore/ via e4_best_of_k.py (curric_L2_s09/s10 entries).

## Finding 10 — E5 residual hybrid: Rust-native scoring, transient metrics, and minimal authority (08/07)

The residual hybrid — `action = clip01(PID_baseline + clamp((WNN−0.5)·scale))`, WNN
trained via residual-DAGGER to reproduce `clamp(PID+ − baseline)` — is now scored
**entirely inside the Rust rollout**, and three questions it raised are answered.

**Mechanism (paths collapsed).** Phase 1 moved the transient-speed metrics
(rise-time, settling-time abs±2° + rel±5%, ITAE/IAE/ISE) into `controller_rollout.metal`
— the same GPU pass that already produced reward/err/stable/jerk/mono/steady — so
`score_controllers_metal` returns a 12-metric row (`Vec<Vec<f64>>`). Phase 2 ported
the PID baseline (`AttitudePidRs::step_rs` → in-kernel `pid_step`, quat→euler + clamped
per-axis I-term in registers + '+' mixing) and the residual composition into the same
kernel, so ONE Rust rollout scores both pure-WNN and PID+WNN-hybrid controllers. The
Python `run_episode` / `make_residual_action_fn` are now parity oracles only: GPU vs
Python agree to 0.000° err / 0.01% ITAE clean, and within one held-out episode (5%)
under L2 weather (the residual per-episode-seed derivation differs GPU-vs-CPU).

**Ablation reproduces native.** 4 cells (baseline ∈ {pd, stock_pid} × seed ∈
{20260609, 20260610}) @L2 held-out, scored through BOTH paths. In every cell HYBRID
≥ BASE and reaches/matches PID+ (rust): pd/s09 85→95→100, pd/s10 85→95→95, stock/s09
90→100→100, stock/s10 95→95→95. The Phase-0 finding (residual clears the memoryless-PD
ceiling toward PID+) holds on the production Rust scorer.

**Transient metrics: ITAE discriminates where stable_rate saturates.** Under L2 the
steady-state floor (~3.75° even for PID+) sits ABOVE the 2° settle band, so rise/settle
pin at the full-duration sentinel — but ITAE (continuous, time-weighted) cleanly orders
PD 0.147 > stock 0.143 > PID+ 0.132 where stable_rate ties at 90%. The threshold metrics
(rise/settle) only become meaningful at a lighter regime (L0/L1) where controllers
actually settle inside the band.

**Regime sweep (the "faster reaction?" test).** Sweeping {OFF, L1, L2} on the pd
baseline (each hybrid trained AND tested in-regime), rise/settle finally discriminate:
- OFF: HYBRID is WORSE than PD (err 0.64→1.21°, rise 594→849ms, settle 230→686ms) —
  with no bias to reject, the WNN's imperfect residual is just noise on an already
  near-optimal PD.
- L1: HYBRID has the FASTEST rise (1484ms vs PID+ 1639, PD 1784) but SLOWER settle
  (1382ms vs 825-1063ms) — an aggressive early corrector that overshoots near the band.
- L2: HYBRID best on err (3.69°) + ITAE (0.133); rise/settle still sentinel-pinned.

So "faster reaction" is a partial win: faster *initial* response (rise), worse
*settling*. And the residual is **disturbance-proportional** — it HURTS at OFF, is
mixed at L1, and clearly helps at L2. The hybrid is a disturbance-rejection
specialist, not a universal upgrade; its value is conditional on the operating regime.
Provenance: scripts/e5_transient_regime.sh (marker /tmp/wnn_e5transient_done.json,
08/07 18:27Z), scripts/e5_residual_proof.py (dist level = argv[4]).

**Minimal authority: the residual is a GENTLE correction.** Scalar clamp sweep
(pd @L2, retrain-per-value): the clamp binds only below ~0.08 — HYBRID stable 0.01→90%,
0.02→95%, 0.05→100%; ITAE monotone 0.149→0.133, plateaus at 0.08; identical thereafter
to 0.4. So the WNN needs **~0.05-0.08 (5-8%) motor authority** to capture the full benefit
— the proof's 0.4 default was 5× more than necessary. FPGA-relevant: small residual PWM
swing → tight DAC range, simple hardware, and it confirms the integral action PD lacks is
LOW-magnitude. A per-axis clamp is not yet warranted (the scalar binding curve is clean;
no axis-difference signal), so the `[f32;4]` plumbing was deliberately NOT built.

## Finding 11 — optimal-control teachers (LQR/MPC): a stronger teacher helps, but the WNN is capacity- not authority-limited (08/07)

The residual-DAGGER teacher was generalized from PID+ to optimal control (src/wnn/
control/optimal.py: LQRController + MPCController on the 6-state attitude double-
integrator, control→accel gains calibrated by stepping the sim; both expose the
AttitudePID step/reset interface, so they drop into make_expert + the dagger expert
slot). residual_expert ∈ {pid_plus, lqr, mpc}.

**Optimal control decisively beats PID+ (L2 held-out).** LQR err 1.57° / settle 637ms
vs PID+ 3.75° / never-settles-within-2° (2.4× lower error). MPC ≈ LQR (err 1.52°) —
the ±0.4 authority box rarely binds at 5° tilts, so constrained MPC collapses to the
unconstrained optimum, as theory predicts. (MPC needed the discrete-LQR cost-to-go as
terminal cost + dt_mpc matched to the sim step to reach LQR — the classic short-horizon
pitfalls.) Caveat: our PID+ is "stock × cranked integral," not a carefully-tuned strong
PID, so part of the gap is tuning; but LQR is a strictly stronger teacher regardless.

**A stronger teacher yields a better student — but only partially.** PD + WNN-imitating-
LQR reached err 3.16° / ITAE 0.115 — better than the PD + WNN-imitating-PID+ hybrid
(3.62° / 0.133), and the collapsed Rust path reproduces it (3.08°). But it captures only
~⅓ of the PD→LQR gap (PD 4.04° → hybrid 3.16° → LQR 1.57°): it matches the LQR's
STABILITY (100%) but not its error/settling.

**The bottleneck is WNN capacity, not authority.** An LQR-teacher clamp sweep
{0.4, 0.6, 0.8, 1.0} gave a BYTE-IDENTICAL hybrid (3.16° / 0.115) at every clamp — the
clamp never binds, so clamp(LQR−PD) < 0.4/motor: the residual is SMALL. The shortfall is
that the 16-neuron fixed-random-connectivity WNN can't faithfully reproduce the LQR's
state-feedback residual (small per-step control errors integrate into large trajectory
error over 2000 steps). So the lever to close the gap to LQR is ARCHITECTURE — more
state neurons/bits + GA-optimized connectivity (the phased-GA stack) — not more authority.
This is the concrete next experiment for "LQR performance at RAM-lookup cost."

Provenance: scripts/e5_residual_proof.py (expert=argv[5]), scripts/e5_lqr_clamp_sweep.sh
(marker /tmp/wnn_e5lqrclamp_done.json, 08/07 21:34Z), src/wnn/control/optimal.py.
Commits: LQR/MPC teachers + wiring, LQR-clamp-sweep driver.

## Threats to validity / open items

- Single plant, clean sim (no wind/noise/motor asymmetry — disturbances are the
  planned follow-up and would give integral action real work for the first time).
- Winners are n=1 per recipe cell (2 seeds per arm; fresh-seed protocol guards
  selection luck but not recipe-level seed lottery).
- Committee members share the thermometer encoding — correlated-drift risk at
  horizons ≫10⁴ steps untested.
- E4 assembly + W1 surface: DONE (Finding 7) — the 96%+ expectation was met twice
  (96.0@2000, 96.5@5000). Open: a panel including w1_h4000_s09 itself; committees
  at horizons ≫10⁴ under member-count vs heterogeneity tradeoffs.

## Provenance

ki=0: scripts/pid_ki_ablation.py (PID_STEPS ∈ {500,2000,5000,10000}). Matrices +
committees: scripts/e4_best_of_k.py (E4_STEPS/E4_ONLY/E4_ENSEMBLE_TOP/E4_SKIP_SOLO;
Rust hot loop `ram_controller.eval_ensemble_closed_loop`, ICs injected from the
numpy chain for exact fresh-seed reproduction). Winners: FrameFixVal/Bits_20260627,
LowEdge_20260701, StateIntegral_20260701, E2Reliability_20260702, C2K_20260702
(8 cells, marker 04/07 23:45 UTC; per-cell tables via scripts/c2k_status.py),
W1Surface_20260702 (4 cells, marker 05/07 23:10 UTC). Finding 7: E4 chain
logs/controller/E4Chain_20260706/ (42 cells: leg_a decay matrix via
scripts/w1_common_ruler.py, leg_b truth serum + leg_c panels via
scripts/e4_best_of_k.py; driver scripts/e4_chain_driver.sh, marker 06/07 11:36 UTC).
Commits: c3a60914 (ki=0), 0882b19d (horizon drift), 468b0b3d (Rust committee eval),
aa05c717 (E4 chain tooling).
