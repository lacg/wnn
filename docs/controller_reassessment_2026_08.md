# Controller programme reassessment — 09/08/2026

Written while the seed-31337004 tie-break cohort flies. Purpose: pin the paper claim,
audit the one gap the committee did not close (the observer gap), pre-register the two
successor moves (alphabet probe — QUEUED; sn>0/state programme — SPEC'D), and set the
frame for ranking the remaining paper tasks once the in-flight results land.

Companion docs: `docs/l4_teacher_screen_results.md` (L4 + committee cohort, the primary
record), memories `project_l3_refuted_l4_committee_armed`, `project_controller_820_instructions_step`,
`project_controller_lut_footprint_b30`.

## 1. Where the programme stands (banked, do not re-derive)

All triples `stable% / err° / steady°`, held-out report seeds 99990101–05, L4C,
cf21_brushless, 100 eps × 2000 steps.

- **Single-run hold-floor programme CLOSED.** L1 (d̂-as-input), L1b, L2, L3
  (granularity), L4 (write-order arms) — every lever refuted or weak. L4 verdict:
  keep `--write-priority-err` (~5% only-A), drop `--write-err-floor`. The ~0.5°
  steady floor of a solo student is structural under this recipe.
- **Committee bar PASSED** (first mechanism to move the floor): FULL4 median
  100.0/1.07/**0.26** on s31337002 and 100.0/1.39/**0.64** on s31337003 — through the
  0.35° bar every single-run lever failed. FULL5 mean (fully a-priori) passes too
  (0.35/0.68). k=3 matches or beats k=4/5; pid drags the vote at every size (8/8).
- **Classical rivals, same engine/episodes:** pid 100.0/2.82/1.97 · mpc 100.0/1.18/0.76
  · lqr 100.0/1.03/0.55 · lqi 100.0/0.90/0.44 · **mpcof 100.0/0.72/0.00**. The
  committee beats PID and MPC outright, brackets lqr/lqi. The only rival not beaten at
  all is mpcof, and specifically its **0.00±0.00° steady**.
- **Deployment cost banked:** 820 instructions/step on a Cortex-M4 (measured,
  equivalence-verified; PID 394, MLP 34.4k); b=30 memory FITS a Zynq-7020 as sparse
  keys in BRAM (52–84%, 13 cycles). A committee of k members is k lookups + a vote.
- **In flight:** base seed 31337004 (5th teacher run + auto-scoring chained) to answer
  whether base-seed variance swamps committee choice. Early signal: every teacher
  lands ~0.5–0.7° worse on 004 — if that holds, the honest report is the a-priori
  FULL4/FULL5 across 3 seeds, and we decline to name a single winning committee.

## 2. The paper claim (IROS 2027, deadline 01/03/2027)

> **A committee of RAM-WNN lookup-table students, each distilled from a different
> classical teacher, holds attitude under sustained disturbance to a precision that
> beats firmware PID and receding-horizon MPC outright and brackets the linear-optimal
> controllers — at ~820 instructions per control step, no multiply-accumulate hardware,
> no per-step optimization, on memory that fits the smallest Zynq part.**

What the claim does NOT need: beating mpcof. mpcof runs a per-step optimization with a
disturbance observer — it is the asymptote the committee approximates at ~1/40th the
compute of even an MLP, not a rival on the same cost curve. The observer gap
(0.26–0.64 vs 0.00) is the *successor* result; closing it strengthens paper 2 (or a
camera-ready delta), it does not gate paper 1.

## 3. The observer gap — evidence audit

Why the gap exists: to hold 0.00° against the L4C bias the controller must park at an
equilibrium input u\* and emit exactly the correction the disturbance demands. Delta
control already gives the loop an accumulator for free (u += δ each step), which is
precisely why the information-route levers all refuted:

| Route | Test | Verdict |
|---|---|---|
| Hand the student d̂ as an input | L1 / L1b (2×2) | REFUTED, 4/4 worse |
| Hand it explicit error-integral features | integral-INPUT probe | DEAD — lost to blind S16 |
| More context (bigger window, dfa) | dfa1l cost model | Same info-route + ~106h/run QUAD — vetoed on cost AND on the two refutations above |

The loop exists and the information is already inert as an input ⇒ what remains is
**(a) the resolution of the increment** and **(b) structure that carries state**:

- (a) The antagonist decode quantizes the correction: decoded = 0.5 + (ΣE−ΣI)/levels,
  mapped piecewise-linear to ±delta_max. At levels=16, delta_max=0.1 the smallest
  nonzero per-step correction is 0.0125 PWM — the controller can only orbit u\* in a
  limit cycle of that quantum. mpcof's correction is continuous. **This is a grid
  axis, not an architecture** → Probe 1, queued.
- (b) A memoryless sn=0 LUT cannot express an observer's internal state (bias
  estimate, integral memory). The stateful teachers (lqi, mpcof) distill into
  memoryless students that floor at ~0.5–0.8° — the state is exactly what
  distillation drops. → Programme 2, spec'd below.

## 4. Probe 1 — increment-alphabet resolution (QUEUED 09/08/2026)

`scripts/alphabet_probe_chain.sh`, armed behind the 31337004 scoring chain (gate
PID 9732), box-idle guarded, one controller at a time.

- **Arms:** levels ∈ {32, 64} × base seeds {31337002, 31337003}, teacher lqi
  (cheapest at ~20–25 min/run; best solo on s002). Identical to the committee control
  shape except `--levels` and the implied `--max-output-neurons` (128 / 256).
  Quantum: 0.0125 (control) → 0.00625 (L32) → 0.003125 (L64).
- **Controls (already flown, levels=16):** lqi s002 99.8/1.11/0.53 · lqi s003
  100.0/1.58/0.81.
- **Bar (pre-registered):** an arm beats the same-seed control's steady on BOTH seeds
  without losing stable. One-seed-only = suggestive, re-fly first.
- **Refutation:** neither arm beats either control ⇒ increment quantization is not the
  binding floor ⇒ Programme 2 is promoted on merit.
- **Cost:** 4 runs ≈ 2 h. If L32/L64 pass, the follow-up is re-flying the committee
  members at the winning resolution (~25 h, the full cohort) — that decision waits
  for the probe result.
- **Risk note:** levels=64 quadruples output cells vs control; `--max-cells 180000
  --max-cells-strict` and the watchdog remain armed; the arm-lib retries once memory
  recovers on a watchdog kill.

## 5. Programme 2 — sn>0 / state neurons (SPEC, not launched)

**Hypothesis.** The observer gap is *state*, not information: give the substrate a few
recurrent state neurons and let training discover an integrator/bias-estimator, the
thing L1 proved cannot be injected as an input feature.

**Why it might win now when sn=0 beat sn>0 before:** the single-layer promotion ranked
on the harmonic fitness (err²/stable/jerk/mono) — **steady is not a fitness
component**. sn>0 may well lose fitness while winning steady; nobody has looked. The
conflict-driven split trainer (+20pp when it landed) and the state-pressure
measurement (yaw dead-reckoning: 12.8% conflicts for a yaw-blind student) both say the
substrate *can* use state when the task demands it — and holding a bias is exactly a
dead-reckoning task.

**Design (2×2 + control, budget-matched to the committee recipe):**

- Teacher: **mpcof** (the 0.00° teacher — if state buys the observer loop anywhere,
  it is here), with lqi as the cheap second teacher if round 1 moves.
- Arms: sn ∈ {4, 8} × base seeds {31337002, 31337003}; `--grid-state-neurons sn
  --max-state-neurons sn`, split trainer on (the conflict-driven path is the state
  writer), everything else the committee control shape.
- Controls: the flown sn=0 members (mpcof 0.72/0.74, lqi 0.53/0.81 steady).
- **Bar:** an sn>0 arm beats its same-teacher, same-seed sn=0 control's steady on BOTH
  seeds without losing stable. Secondary read-out: does the winner's advantage grow
  with episode length (an integrator needs time to converge — a pure quantization
  winner would not show length-dependence)?
- **Refutation:** sn>0 cannot beat sn=0 on steady even under the observer teacher ⇒
  recurrent state in this substrate does not buy the observer loop; the gap is then
  either resolution (probe 1) or genuinely needs the optimizer — either way paper 1's
  framing ("approximates the observer at 1/40th the compute") stands unchanged.
- **Cost:** mpcof runs were 2.7–3.2 h each → 4 arms ≈ 12 h, +2 lqi arms ≈ 1 h if
  promoted. Launch only when it wins the queue on merit — i.e. after the 004 scoring
  and the alphabet probe have both landed and been read.

**Deployment honesty:** sn>0 adds a recurrent read-modify-write to the 820-instr/step
loop (state features re-address the memory each step). The MCU harness must re-measure
before any sn>0 number enters a paper table next to the compute claim.

## 6. Task ranking — frame now, finalize after 004 + probe

Criterion: **reviewer-risk retired per hour**, against the §2 claim. The provisional
reading (to be finalized once the 004 scoring and probe 1 land):

| Task | What it retires | Cost feel | Provisional rank |
|---|---|---|---|
| #2 Motor lag (T=0.15 s) | "Your plant is too easy" — the single most attackable idealization; sensor noise already on | Small (sim change + one re-fly) | **1** |
| #5 Learned baseline (Molchanov/Eschmann) | "Beats PID" is table stakes; reviewers will ask for a learned rival | Medium (integration + training) | **2** |
| #3 Setpoint tracking | "Regulation-only" scope narrowness; changes the task definition, so it must precede any final cohort | Medium | **3** |
| #4 gym-pybullet-drones port | Engine-specificity — strongest credibility buy per claim, but only worth doing on the FROZEN recipe | Medium-large | **4** (after #2/#3 freeze the recipe) |
| #6 Crazyflie hardware | The headline demo; gated by everything above | Large | **5** |

Ordering logic: #2 and #3 change the *plant/task* and therefore must come before the
recipe freezes; #4 and #6 validate the frozen recipe and are wasted effort if run
before it freezes. #5 is independent of the freeze and can interleave. The alphabet
probe result feeds this: if resolution moves steady, the freeze includes a levels
change and #4/#6 wait for it.

## 7. Decision log

- 09/08/2026 — reassessment drafted; alphabet probe armed behind the 004 scoring
  chain; sn>0 programme spec'd but NOT queued (launches on merit after probe + 004
  read-out). Order chosen by Luiz: probe first, then spec, then ranking finalized on
  the results.
