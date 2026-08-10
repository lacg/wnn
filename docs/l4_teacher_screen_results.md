# L4 teacher screen — Crazyflie 2.1 Brushless, L4C

## STATUS: screen COMPLETE (06/08) · ALL single-run hold-floor levers REFUTED — L1/L1b/L2/L3 refuted, L4 only-A-weak (08/08) · COMMITTEE cohort 10/10 + scoring PASSED the pre-registered bar (09/08) — FULL4 median 0.26/0.64° steady, the first mechanism to move the floor · ⚠️ **THE THERMOMETER WAS MIS-CALIBRATED (10/08)** — every number above was measured through a ~6× mis-scaled encoder; see §"The thermometer was calibrated on the wrong distribution"

The re-flown screen finished 06/08/2026 ~03:56 EDT: 10/10 markers on the fixed plant
with the firmware-sourced PID cascade. Headline: **no student beat its teacher on any
of the ten runs, and four teachers spanning 0.69–1.09° of classical quality produced
students in a 1.27–1.40° band** — teacher quality does not propagate. The mechanism was
then isolated by a per-step decomposition (§"Where the error lives"): every student is
teacher-grade in recovery and hits an absolute hold-attitude floor.

**07/08/2026 — the floor is STRUCTURAL.** All three follow-ups are flown and all three
fail (§"Hold-floor levers"): making the disturbance observable (L1, d̂ features) made hold
WORSE in 4/4 comparisons; making it irrelevant (L2, residual on the firmware cascade)
roughly DOUBLED hold error on both seeds and both scoring paths; and optimizing for hold
explicitly (L1b, S16 weights on delta) did not reliably move it. Neither more input, nor a
better substrate, nor a hold-targeted objective touches the floor ⇒ the limit sits in the
LEARNING / CREDIT-ASSIGNMENT path, which promotes **L3** (`delta_leak`/`delta_max`, never
searched) and **L4** (magnitude-weighted DAgger conflicts) to the only live candidates.

## Historical: the withdrawn 05/08 screen

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
PID      100.0± 0.0   1.78±0.34    1.03±0.36     <- firmware cascade (sourced), see note
LQR      100.0± 0.0   0.93±0.04    0.42±0.07
MPC      100.0± 0.0   1.09±0.08    0.65±0.12
LQI      100.0± 0.0   0.81±0.03    0.36±0.03
MPCOF    100.0± 0.0   0.69±0.01    0.00±0.00
```

**The PID row is the firmware-sourced cascade** (`wnn.control.pid_firmware` +
`controller/pid_firmware.rs` + `pidfw_step` in `controller_rollout.metal`, gains from
`platform_defaults_cf21bl.h`; golden-tested 1e-12 Python↔Rust, GPU parity
mutation-verified). The earlier legacy-tuned row (1.64±0.16 / 1.35±0.20) is preserved in
git history; every number in this document now uses the sourced cascade.

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
2. ✅ Firmware PID ported to Rust + Metal with CPU/GPU parity (05/08, 111 tests green).
3. ✅ PID baseline row re-measured on the cascade (1.78±0.34 / 100.0 / 1.03±0.36).
4. ✅ Closed-form arm: lqr / lqi / pid × 2 seeds (6 markers, 05–06/08).
5. ✅ MPC family LAST: mpcof / mpc × 2 seeds, NEURONS capped at 5 gens (4 markers, 06/08).

Order set by Luiz (05/08): MPC family at the end. The 5-generation NEURONS cap makes that
arm **not budget-matched** to the closed-form arm (8-14 generations), so when its numbers
land: a capped MPC student that BEATS the closed-form mean is conclusive, one that loses
is ambiguous between teacher quality and search budget and must be reported that way.

## Screen results (06/08/2026, 10/10 markers)

Recipe per run: grid (b∈{24,30}, sn=0) → NEURONS GA (early-stop, 5-gen cap on the MPC
family) → MEMORY GA (120-gen cap, patience 2, magnitude-aware), pop 50, K=5
fold-accumulate, `--max-cells 180000 --max-cells-strict`, BINARY antagonist decode,
levels=16, tilt 5°, L4C, 5 report seeds × 100 episodes. Markers:
`experiments/l4teach_markers/L4T_*.json`.

**MULTI-SEED held-out (5 report seeds), full triple err°/stable%/steady°, both stages:**

```
teacher  seed        NEURONS held-out                MEMORY held-out
lqr      31337002    1.12±0.05 / 100.0±0.0 / 0.63±0.13    1.19±0.11 /  99.4±0.8 / 0.66±0.11
lqr      31337003    1.39±0.20 /  99.0±1.5 / 0.84±0.19    1.39±0.11 /  99.8±0.4 / 0.69±0.12
lqi      31337002    1.31±0.15 /  99.6±0.8 / 0.67±0.13    1.18±0.09 / 100.0±0.0 / 0.69±0.10
lqi      31337003    1.95±0.33 / 100.0±0.0 / 1.57±0.49    1.39±0.21 /  99.6±0.5 / 0.80±0.15
pid      31337002    2.43±0.54 /  97.0±3.1 / 2.25±0.93    2.80±0.84 /  94.2±5.4 / 2.72±1.43
pid      31337003    2.24±0.43 /  97.2±5.1 / 1.75±0.82    2.60±0.31 /  96.4±2.0 / 2.01±0.41
mpcof    31337002    1.20±0.04 / 100.0±0.0 / 0.63±0.08    1.21±0.05 / 100.0±0.0 / 0.64±0.12
mpcof    31337003    1.46±0.13 / 100.0±0.0 / 0.89±0.11    1.58±0.22 / 100.0±0.0 / 0.95±0.30
mpc      31337002    1.31±0.07 /  99.8±0.4 / 0.78±0.11    1.27±0.08 /  99.8±0.4 / 0.68±0.11
mpc      31337003    1.58±0.11 / 100.0±0.0 / 0.83±0.10    1.30±0.05 / 100.0±0.0 / 0.81±0.08
```

**Student vs its own teacher (MEMORY err°, n=2 training seeds):**

```
teacher  classical      s...002   s...003   student mean   seed spread
mpcof    0.69±0.01       1.21      1.58        1.40           0.37
lqi      0.81±0.03       1.18      1.39        1.29           0.21
lqr      0.93±0.04       1.19      1.39        1.29           0.20
mpc      1.09±0.08       1.27      1.30        1.29           0.03
pid      1.78±0.34       2.80      2.60        2.70           0.20
```

Findings:

1. **No student beat its teacher on any of the ten runs.**
2. **Teacher quality does not propagate.** Four teachers spanning 0.69→1.09° (1.6×)
   produced student means of 1.29/1.29/1.29/1.40 — a 0.11° total spread, smaller than
   the seed-to-seed spread of three of the four arms individually. The mpc arm's two
   seeds landed 0.03° apart (the tightest estimator in the screen) exactly on 1.29°.
3. **Only pid separates, and it separates upward** (2.70° from a 1.78° teacher): a bad
   teacher caps the student, a good one does not lift it. Directionally the ordering
   inside the band even runs backwards (best teacher mpcof → worst non-pid student),
   though at n=2 with a 0.37° spread that is not significant.
4. **MPC-family caveat:** capped at 5 NEURONS gens, not budget-matched. Both MPC-family
   students lost to their teachers, which is ambiguous under the cap — but the
   closed-form arms with a full budget landed at the *same* 1.29°, so the cap is not
   what holds the MPC arms back.
5. **`--max-cells-strict` bounds the population MEAN only, and not reliably.** From gen
   lines (markers record last-gen only, ~2× understated): lqi peaked μ263k
   (max-in-population 492,650) and mpc μ223k (max 286,234) against the 180k budget
   during MEMORY.
6. The MEMORY stage moved err *backwards* on three arms (mpc s002 within noise, mpcof
   s003, pid both) — consistent with the floor mechanism below: once a genome's error
   is dominated by the hold floor, memory refinement optimizes a non-binding term.

## The decode is delta-mode — the static quantization ceiling does not apply (06/08)

A rollout-trace verification (production Rust sim + controller, thresholds refit on the
run's DERIVED train seed) showed the deployed students run **delta control**
(`delta_control=True, delta_max=0.1, delta_leak=0.95`):
`pwm = 0.5 + 0.95·(pwm−0.5) + Δ`, with the 17-value alphabet quantizing **Δ (step
0.0125), not the throttle** — one episode produced 7,985 distinct absolute pwm values.
In the steady tail 70–82% of steps emit a nonzero Δ with mixed signs and fractional
per-motor means: the leaky integrator converts duty-cycled increments into effectively
continuous actuation (the ΔΣ-modulator construction), at the price of a limit-cycle
ripple.

Consequently `scripts/decode_quantization_ceiling.py` — which quantizes **absolute** pwm
onto the alphabet (1.941° at L=16 for a 0.405° controller, predictions L=32→1.055°,
L=64→0.567°) — models a different mechanism from the deployed decode, and its numbers
must not be cited as the substrate's floor. The levels ablation below is the empirical
test and confirms this.

⚠️ Reproduction gotcha (cost a day of inflated numbers): winner filenames carry the
BASE seed; score-only rebuilds must fit thresholds on the **derived train seed**
(`resolve_seed_set`: 31337002 → 3072558954). Fitting on the base misaligns the address
function — the s31337003 mpcof student reads 18°/0% stable instead of 1.58°/100%.

## Levels ablation (output-alphabet test; round 1 complete, round 2 in flight)

mpcof only, L∈{32,64} × 2 seeds, cell budget scaled 180000·levels/16. Same-training-seed
MEMORY held-out triples (s31337002):

```
levels   err°          stable%       steady°
L=16     1.21±0.05     100.0±0.0     0.64±0.12
L=32     1.24±0.16      99.8±0.4     0.59±0.13
L=64     1.30±0.16      99.6±0.5     0.70±0.15
```

Quadrupling the alphabet moved nothing — err drifts mildly the wrong way (within
spread), steady wobbles 0.64→0.59→0.70 (noise, not a resolution effect), and the L=64
result is 2.3× the quantization prediction (0.567°). **The output-alphabet hypothesis
fails its own ablation at this seed**, as the delta-mode analysis predicts.

⚠️ **STOPPED at 2/4 markers (06/08/2026), so this is n=1 PER LEVEL** — one training
seed (s31337002). Round 2 (the s31337003 pairs) was not flown: the D2 decomposition
below had already identified the mechanism, and continuing would have spent ~7 h
re-testing a hypothesis the trace had superseded. Judged by this project's own
standard the null is therefore **suggestive, not conclusive**: the L=16 mpcof
seed spread is 0.37°, wider than the 0.09° spread across all three levels, so a
single seed cannot formally separate them. What makes the null credible anyway is
that it is *predicted* — the delta-mode trace says the alphabet quantizes the
increment, so raising `levels` was never attacking the err term. The load-bearing
evidence for the floor is the decomposition, not this ablation.

## Where the error lives — per-step decomposition (D1/D2, 06/08)

`scripts/transient_decomposition.py` traces every winner student AND its own classical
teacher per-step on the exact held-out pools (`trace_controller_cpu` shares the
production CPU scorer's `rollout_one`; `trace_classical_baseline` is the asserted-equal
trace twin of `score_classical_baseline`). 500 episodes per run. Phases: RECOVERY
(steps 0–20%), CRUISE (20–80%), STEADY (80–100%).

```
run               RECOV  CRUISE  STEADY      student err/stable/steady      teacher err/stable/steady
lqi   s31337002   1.07x   2.09x   1.89x      1.18°/100.0%/0.68°             0.81°/100.0%/0.36°
lqi   s31337003   1.21x   2.53x   2.23x      1.39°/ 99.6%/0.80°             0.81°/100.0%/0.36°
lqr   s31337002   1.01x   1.45x   1.49x      1.12°/100.0%/0.63°             0.93°/100.0%/0.42°
lqr   s31337003   1.17x   1.94x   1.67x      1.38°/100.0%/0.71°             0.93°/100.0%/0.42°
mpc   s31337002   1.06x   1.33x   1.03x      1.26°/ 99.8%/0.66°             1.09°/100.0%/0.65°
mpc   s31337003   1.03x   1.46x   1.23x      1.33°/100.0%/0.79°             1.09°/100.0%/0.65°
mpcof s31337002   1.09x   3.29x    →∞        1.24°/100.0%/0.57°             0.69°/100.0%/0.00°
mpcof s31337003   1.09x   4.27x    →∞        1.47°/100.0%/0.87°             0.69°/100.0%/0.00°
pid   s31337002   0.88x   1.51x   1.99x      2.34°/ 96.4%/2.05°             1.78°/100.0%/1.03°
pid   s31337003   0.99x   1.72x   1.93x      2.59°/ 96.4%/1.99°             1.78°/100.0%/1.03°
```

**The mechanism of the universal 1.2–1.4° band, measured:**

1. **RECOVERY is teacher-grade everywhere** (0.88–1.21× across all five teachers; the
   pid students even beat their sluggish teacher). Imitating the transient is solved.
2. **The hold floor is absolute, not relative.** Every non-pid student lands at steady
   0.57–0.87° whether its teacher holds 0.00° (mpcof) or 0.65° (mpc). The cruise ratios
   (1.33→4.27×) are one fixed student floor divided by five teacher denominators.
3. **mpc is the natural control:** its own steady (0.65°) sits AT the student floor, and
   its student matches it at ratio ~1.0. The deficit only materializes where the teacher
   goes *below* the floor — which is why the screen sees "teacher quality doesn't
   propagate": quality differences among good teachers live entirely below the floor,
   where the student cannot express them.
4. So err ≈ 20% teacher-grade recovery + 80% hold-at-the-floor, and the floor is a
   **disturbance-observability limit** (the teachers that beat it carry integral action
   or an explicit disturbance observer d̂), not a learning, output-resolution, or
   teacher-quality limit.

## Hold-floor levers — ALL RUN, ALL REFUTED (07/08/2026)

**Both levers and the L1b control experiment are complete. None of them moved the hold
floor. Taken together they locate the limit in the LEARNING / CREDIT-ASSIGNMENT path, and
promote L3 and L4 from "deferred" to "the only remaining candidates."**

All numbers below are MEMORY-stage multi-seed held-out (5 report seeds), reported as the
full triple **err° / stable% / steady°**. Read **steady** as the primary: err is ~80%
recovery term and recovery is already teacher-grade (0.88–1.21× per D1/D2), so a hold fix
can move err by at most the ~20% the steady window carries.

### L1 — `--obs-dhat`, the disturbance observer as 3 input features — REFUTED

Give the student the mpcof observer's d̂ so the disturbance becomes *observable*.

| seed | control (C10, no d̂) | L1 (C10 + d̂) | Δsteady |
|---|---|---|---|
| 31337002 | 1.21 / 100.0 / **0.64** | 1.44±0.20 / 99.8±0.4 / **0.66±0.17** | +0.02 (flat) |
| 31337003 | 1.58 / 100.0 / **0.95** | 2.00±0.27 / 99.8±0.4 / **1.45±0.32** | **+0.50 (worse)** |

Pre-registered success was steady < ~0.35° on BOTH seeds. Neither clears it; seed 31337003
degrades beyond its own ±0.32 spread, on all three metrics.

### L1b — the 2×2 that rules out the ranking as the explanation — REFUTED

L1 ranks by **C10** (`err² .40 / stable .30 / jerk .20 / mono .10`), which has **no steady
term** (`--fit-weight-steady` defaults to 0.0). So an L1 null could not distinguish "d̂ does
not help hold" from "the search never looked for hold." **S16** (`err .25 / **steady .35** /
stable .20 / jerk .15 / mono .05`) puts the largest weight on steady; it won the 25/06
ABSOLUTE-substrate sweep and had never been flown on delta. The 2×2 separates the WEIGHTING
from the FEATURE. Chain: `scripts/l1b_s16_dhat_chain.sh`, 4 runs, all rc=0.

| | **no d̂** | **d̂** |
|---|---|---|
| **C10** | s02 1.21/100.0/**0.64**<br>s03 1.58/100.0/**0.95** | s02 1.44±0.20/99.8±0.4/**0.66±0.17**<br>s03 2.00±0.27/99.8±0.4/**1.45±0.32** |
| **S16** | s02 1.23±0.19/99.2±1.0/**0.45±0.11**<br>s03 1.74±0.24/99.8±0.4/**1.23±0.39** | s02 1.63±0.25/99.8±0.4/**1.04±0.31**<br>s03 2.49±0.49/95.2±6.0/**2.19±0.61** |

**d̂ makes hold worse in 4 out of 4 comparisons**, holding weighting and seed fixed:

| Δsteady from adding d̂ | C10 | S16 |
|---|---|---|
| s31337002 | +0.02 | **+0.59** |
| s31337003 | +0.50 | **+0.96** |

and the penalty is **largest under S16** — the ranking that weights hold most heavily. That
is the opposite of the L1b hypothesis: nothing was hidden by C10. The most likely mechanism
is that +3 features widen the input space against an unchanged grid budget, and the search
pays for that in genome quality more than the observer information is worth.

**The S16 weighting itself has no reliable effect on delta**: without d̂ it helped one seed
(0.64 → 0.45, the best hold measured anywhere in this programme) and hurt the other
(0.95 → 1.23); with d̂ it hurt both. Split at n=2 ⇒ ranks nothing. This is consistent with
the 25/06 sweep's own finding that SUBSTRATE dominates weights (+14.2 pp vs ~2.7 pp) — S16
won on ABSOLUTE and does not transfer to delta.

⇒ **The floor survives a ranking that explicitly optimizes for it.** That was the
pre-registered third branch, and it is the strongest evidence yet that the floor is
STRUCTURAL — not input observability, not a ranking artifact.

### L2 — WNN residual on the firmware PID cascade — REFUTED

Make the disturbance *irrelevant*: let the shipped Crazyflie cascade's integral absorb the
sustained bias and leave the student the transient it is already good at. This is also the
deployability variant — the only one flyable on stock firmware without replacing the
controller. `scripts/e5_residual_proof.py`, airframe cf21_brushless, expert mpcof,
baseline stock_pid, 20 held-out episodes.

Re-run 07/08/2026 **after** the disturbance fixes below, so both columns are finally the
same experiment:

| seed / path | BASE (cascade alone) | HYBRID (base+residual) | EXPERT (mpcof) |
|---|---|---|---|
| 31337002 python | 1.24 / 100.0 / **0.52** | 2.60 / 100.0 / **2.18** | 0.91 / 100.0 / 0.46 |
| 31337002 rust | 1.15 / 100.0 / **0.52** | 2.29 / 100.0 / **1.61** | n/a |
| 31337003 python | 1.77 / 100.0 / **0.80** | 2.70 / 100.0 / **2.07** | 1.27 / 100.0 / 0.91 |
| 31337003 rust | 1.72 / 100.0 / **0.82** | 2.48 / 100.0 / **1.79** | n/a |

**BASE steady now agrees across paths** (0.52 / 0.52 and 0.80 / 0.82) — that agreement is
the check that the two columns are comparable at all, and it did not hold before the fixes.

**The residual makes hold 2.6–4.2× WORSE** in all four measurements and never improves err.
It lands worse than *both* parents. That is the "between the two parents" outcome the spec
pre-registered as a reportable negative, except it is below both.

⚠️ **The spec's premise shifted, the conclusion did not.** The spec justified L2 as
"the cascade's own hold (1.03°) is WORSE than the student floor (0.57–0.87), so the bet is
COMBINATION." Under the corrected disturbance semantics the cascade holds at **0.52–0.80°**
— at or slightly better than the student floor. So the cascade was never the weak partner;
adding the WNN residual to it is what destroys the hold.

⚠️ **`stable` no longer discriminates at all** — it is 100.0% for every arm on both paths,
which is why the script's own verdict now reads `≈ BASE (no lift)`. The steady delta
(`+1.67°` and `+1.27°` vs BASE) is the only line carrying the verdict. Do not read the
stable-based label.

<details><summary>Superseded first-run numbers (old per-episode-asym semantics, kept for provenance)</summary>

| seed / path | BASE | HYBRID | EXPERT |
|---|---|---|---|
| 31337002 python | 2.25 / 100.0 / 1.40 | 2.93 / 95.0 / 2.87 | 1.05 / 100.0 / 0.63 |
| 31337002 rust | 1.40 / 100.0 / 0.76 | 2.05 / 100.0 / 1.59 | n/a |
| 31337003 python | 2.62 / 100.0 / 1.64 | 3.46 / 95.0 / 3.47 | 1.21 / 100.0 / 0.87 |
| 31337003 rust | 1.40 / 100.0 / 0.78 | 2.36 / 100.0 / 2.25 | n/a |

Not reproducible under current code: the Python column averaged over a *distribution* of
airframes (per-episode asym redraw) while the kernel flew one, and the rust column used a
fixed `dist_seed` so it read err=1.40 on BOTH seeds. Direction was already correct.
</details>

The DAgger trace shows this is not noise: the best iterations are 4–5 (mean_err 2.63°,
2.33°), but β anneals to 0.008 and iteration 8 — the one that is scored — degrades to
3.16°/2.82°. The student gets worse as it is handed control, which is a credit-assignment
signature, not a capacity one.

#### ⚠️ The python and rust columns are NOT the same experiment — do not compare them across paths

Two defects had to be fixed before this table could be read at all, and one asymmetry
remains by design of the harness:

1. **FIXED (`score_gpu` plant omission).** `score_gpu` passed `cascade_kwargs()` but not
   `sim_kwargs()`, so the kernel kept its *signature defaults* for the vehicle
   (`arm_length` 0.075, `k_thrust` 2.4, `inertia` 2.3e-3) while being handed
   cf21_brushless's gains and hover force — `k_thrust` 12× and inertia ~77× wrong. Right
   controller, wrong aircraft. That is what produced the earlier
   `BASE (gpu) stable=15.0% err=6.96°`. The shader cascade was never at fault
   (`gpu_pidfw_cascade_matches_cpu_twin` passes, mutation-verified, because that test
   supplies the plant explicitly). A CPU/GPU parity suite proves the KERNEL, never the
   CALLER.
2. **FIXED (steady never printed).** The script reported stable/err/rise/settle/ITAE and
   never steady, on either path — so L2 was structurally unable to answer its own question,
   since steady IS the hold term. `dagger.eval_closed_loop_reset` now returns
   `mean_steady_error_deg` (the value was always computed in
   `EpisodeResult.mean_steady_error_rad`, just never collected), and `score_gpu` now
   unpacks the rust row's index 5.
3. **FIXED (fixed `dist_seed`, then per-episode asym).** Both are closed as of the 07/08
   re-run above; the detail below is kept because it explains what the superseded numbers
   were measuring. The two paths did not fly the same weather:
   - python derives the per-episode disturbance seed from the **episode rng**, which is
     seeded from the held-out seed (`apply_disturbance`), so its weather varies per episode
     AND per seed;
   - the rust path is handed `dist_seed=911` **fixed** (`_dist_args`), so it flies the
     SAME weather on both held-out seeds — which is exactly why its BASE err is `1.40` on
     both, while python's moves 2.25 → 2.62;
   - python redraws the **motor asymmetry per episode** (`resolved_motor_asym(rng)` inside
     `apply_disturbance`); `_dist_args` draws it **once** from seed 911 and bakes that one
     vector into all 20 episodes.

   **Both are now fixed.** `e5_residual_proof` calls the canonical
   `evaluator.disturbance_stream(dist, score_seed)` instead of hand-rolling the derivation
   (the rust column's seed spread went 0.000° → 0.566°), and SCORING binds one resolved
   asymmetry per pass via `DisturbanceConfig.resolved_asym`, matching the kernel's
   per-airframe-wear semantics — Python BASE fell 2.25° → 1.24° onto rust's 1.15°, with
   steady identical. TRAINING deliberately keeps the per-episode redraw; there the variety
   is the point (domain randomization over airframe wear).

   The mechanism of the old gap, for the record: err is convex in disturbance magnitude, so
   averaging over a *spread* of airframes (python) sat systematically above evaluating at a
   *single* one (rust) — Jensen, not noise. That is why it did not wash out with more
   episodes, and why switching the disturbance off made the two agree to 0.10°.

⚠️ **Do not quote the script's `rust path: does NOT reproduce ❌`.** Its test is
`rust_HYBRID_stable > rust_BASE_stable + 2`, and both saturate at 100.0%, so it can never
pass regardless of the result. It is vacuous now that stable no longer discriminates; the
verdict is the steady column.

### What survives, and what is next

The two levers fail in the *same direction* by opposite means, which is what makes the
negative informative rather than merely disappointing:

- L1 gave the student **more information** about the disturbance → hold got worse.
- L2 handed the disturbance to a controller that **provably cancels it** (the cascade's
  integral) → hold got worse.
- L1b asked the search to **optimize hold explicitly** → hold did not reliably move.

Neither more input, nor a better substrate, nor a hold-targeted objective moves the floor.
That triangulates the limit into the learning / credit-assignment path:

- **L3 (now promoted): `delta_leak` / `delta_max` search.** Sustained-offset granularity is
  Δstep/(1−leak) = 0.25 pwm and has NEVER been searched — the same blind spot the levels
  ablation had. If the floor is structural, this is the structure most likely to set it.
- **L4 (now promoted): magnitude-weighted DAgger conflict writes.** The imitation gap
  triples with |err|, and L2's β-annealing degradation is a credit-assignment signature.

### L3 — actuation granularity, both routes — REFUTED (08/08/2026, 4/4 markers)

Two arms reach the IDENTICAL 4×-finer sustained-offset granularity (0.0625 pwm) by
opposite means — `dstep` shrinks the alphabet step (`--delta-max 0.025`), `dleak` shortens
the integrator memory (`--delta-leak 0.80`) — so a granularity-limited floor must move
under at least one of them. It moved under neither (MEMORY-stage multi-seed held-out,
err°/stable%/steady° vs the within-seed control):

```
seed 31337002   control 1.21/100.0/0.64   dstep 1.97±0.95/98.0±3.1/1.44±1.20   dleak 1.93±1.16/99.0±2.0/1.51±1.72
seed 31337003   control 1.58/100.0/0.95   dstep 2.62±0.65/94.4±4.7/2.02±0.80   dleak 1.93±0.29/99.2±1.0/1.29±0.39
```

NEURONS-stage tells the same story (best cell anywhere: dstep s02 0.60±0.15 — a 0.04°
move vs control inside a ±0.15 spread; dstep s03 collapses to 2.79±0.59 with stable
93.0±4.0 — the first outright stability loss in the programme). No cell approaches the
pre-registered 0.35° bar; every other cell is WORSE than control. The signature is
variance inflation, not mean shift: steady SDs balloon to ±1.20–1.72 while means hover at
or above control — the levers perturb the search, not the control law.

**Verdict: the floor is not actuation granularity.** With input (L1/L1b), substrate (L2),
objective (L1b) and actuation (L3) all ruled out, the write rule itself (L4) is the last
candidate standing. Markers: `experiments/l3delta_markers/` (4/4, rc=0, 07-08/08).

### L4 — magnitude-priority output writes — VERDICT: only-A, WEAK (4/4 markers, 08/08/2026)

The spec'd "conflict writes" target was DEAD CODE for this programme —
`use_split` requires `state_neurons > 0` (`dagger_train.rs:1266`) and every hold-floor
run is sn=0 — so L4 was re-derived against the LIVE path: the section-(d) direct write
(`controller.rs:2621`), where BINARY is last-writer-wins and the legacy backward walk
hands every contested cell to the window's EARLIEST record, arbitrary w.r.t. error
magnitude. Two default-off flags (ram_controller ABI 22, parity-gated bit-identical when
off, 3 new unit tests + full 115-test suite green):

- **arm A `worder` `--write-priority-err`** — commits ascend by |err|; the highest-error
  record writes last and owns contested cells. No coverage price.
- **arm B `wfloor` `--write-err-floor 0.5`** — records under 0.5° never commit; the
  hover mass cannot overwrite corrections. Price: settled-hover coverage (a stable%
  collapse here is informative, not a failed run).

2×2 with the same seeds/budget/control as L3; chain `scripts/l4_write_priority_chain.sh`,
markers `experiments/l4writes_markers/`. Both improve ⇒ magnitude-blind credit assignment
is the floor. Only A ⇒ collision ordering. Only B ⇒ hover-mass dilution. Neither ⇒ all
four lever families are exhausted and the floor sits deeper than this programme reaches.

#### Results, 4/4 markers — compared STAGE-AGAINST-SAME-STAGE

⚠️ **Comparison discipline (correction, 08/08).** An earlier read of marker 3 compared
arm A's NEURONS steady (0.85°) against the control's **MEMORY** steady (0.95°) and called
it −10%. That is a cross-stage comparison and it overstates the gain. The control has
BOTH stages measured (screen table above): `s31337002 NEURONS 0.63 / MEMORY 0.64`,
`s31337003 NEURONS 0.89 / MEMORY 0.95`. Like-for-like, s31337003's gain is −4.5%, not
−10%. Every L4 number below is same-stage.

```
MULTI-SEED held-out, arm vs control WITHIN stage and WITHIN seed — full triple
stable% / err° / steady°
                      NEURONS arm       NEURONS ctrl      MEMORY arm        MEMORY ctrl
worder s31337002   99.6 / 1.22 / 0.60  100.0 / 1.20 / 0.63   99.8 / 0.98 / 0.35  100.0 / 1.21 / 0.64
worder s31337003  100.0 / 1.38 / 0.85  100.0 / 1.46 / 0.89   99.8 / 1.39 / 0.91  100.0 / 1.58 / 0.95
wfloor s31337002  100.0 / 1.61 / 0.91  100.0 / 1.20 / 0.63  100.0 / 1.33 / 0.66  100.0 / 1.21 / 0.64
wfloor s31337003  100.0 / 1.55 / 0.99  100.0 / 1.46 / 0.89   99.8 / 1.62 / 1.12  100.0 / 1.58 / 0.95
```

**Verdict (4/4): only-A, and weak — collision ordering is the mechanism, but the bar is
not met.** Arm A improves all 4 stage-cells (err improves too); the gain is a consistent
~5% with ONE outlying cell (MEMORY s31337002, 0.35 vs 0.64, −45%). Arm B worsens all 4
cells and never lost stability, so hover-mass dilution is NOT the mechanism — removing
sub-0.5° hover demonstrations costs nothing in holding and buys nothing in precision
(stable holds ~100%, the pre-registered informative failure did not occur). The
pre-registered SUCCESS bar (steady < ~0.35° on BOTH seeds) is NOT met: s31337003 sits at
0.85–0.91° under arm A. **Recipe consequence: keep `--write-priority-err`; drop
`--write-err-floor`.** With L1/L1b/L2/L3 refuted and L4 weak, the single-run hold-floor
programme is CLOSED — the floor sits deeper than any single-run lever reaches. (The
committee section below is what finally moved it.)

### Stage reporting — publish ALL stages, headline the val-selected one (from 08/08/2026)

Prompted by the L4 numbers above: reporting a single hardcoded stage (MEMORY, `[-1]`)
was never defensible, and picking the best stage *after seeing the report seeds* is
best-of-N inflation. Both problems are fixed by one rule, landing from the committee
cohort onward:

- **Every stage is measured and published** — GRID, NEURONS, MEMORY, each with its full
  report-seed triple. Nothing is hidden or selected away.
- **The headline is the stage chosen on `seeds.val`** — a draw disjoint from both the
  search folds and the report seeds. Selection never touches the published metric, so
  the headline number is unbiased.
- **Rationale:** a run generates several deployable genomes; the honest question is which
  one you would actually fly, answered on unseen data, with every candidate's numbers
  still on the page for the reader to check.

Note the control is re-scored under the same rule (best-stage control = 0.63 / 0.89, NOT
0.64 / 0.95) — scoring only the new arms this way while leaving the control on MEMORY
would manufacture a gain.

## Committee cohort — the bar is PASSED (09/08/2026; the first mechanism to move the hold floor)

The 5-teacher committee cohort (`scripts/committee_teacher_chain.sh`, 10 runs ×
5 teachers × 2 base seeds, control shape, per-stage checkpoints) re-tested the 11/07
committee mechanism (+7.5pp stable at half the σ — but won at an operating point
today's solo control already beats outright) at the 100%-stable operating point where
steady is the only currency left.

**Pre-registered bar: a committee must beat its best SOLO member's steady on BOTH base
seeds without losing stable. PASSED** — by `mpcof+lqr`, by the a-priori FULL-5 mean
vote, and (best of all) by the FULL-4 vote without pid.

### Solo screen (members = each run's val-selected headline pop[0])

All rows `stable% / err° / steady°`, mean±SD over report seeds 99990101–05, L4C,
cf21_brushless, steps 2000 × 100 episodes. Full per-stage tables live in the markers
(`experiments/committee_markers/`); solos below reproduce those marker rows EXACTLY
(the 10/10 validation that gates everything else — see "harness provenance").

```
              seed 31337002  (control 100.0 / 1.21 / 0.64)     seed 31337003  (control 100.0 / 1.58 / 0.95)
SOLO lqi       99.8± 0.4 / 1.11±0.06 / 0.53±0.05  <- best      100.0± 0.0 / 1.58±0.08 / 0.81±0.13
SOLO mpcof     99.8± 0.4 / 1.35±0.07 / 0.72±0.13               100.0± 0.0 / 1.34±0.08 / 0.74±0.10  <- best
SOLO lqr       99.8± 0.4 / 1.27±0.08 / 0.73±0.05               100.0± 0.0 / 1.67±0.23 / 1.16±0.35
SOLO mpc      100.0± 0.0 / 1.47±0.14 / 0.91±0.29               100.0± 0.0 / 1.36±0.08 / 0.94±0.09
SOLO pid       94.0± 4.3 / 2.56±0.49 / 2.16±0.89               96.2± 2.9 / 2.52±0.57 / 2.02±0.89
```

Solo verdict: **only pid separates at n=2** — worst on all three components of the
triple on both seeds, the only member losing stability, and the model-based teachers'
students span 0.53–1.16° steady with orderings that FLIP between seeds (lqi best on
002, mpcof on 003) inside a 0.31° control-to-control seed gap. Cost note: lqi ≈ 1,100–
1,450 s per run vs mpcof/mpc ≈ 9,900–11,500 s — 10× cheaper for differences inside
reproduction noise.

### Committees (PWM vote, same protocol; pairs exploratory, FULL rows a-priori)

```
              seed 31337002                                    seed 31337003
PAIR lqi+mpc      100.0± 0.0 / 1.09±0.06 / 0.26±0.06           99.6± 0.5 / 1.39±0.09 / 0.69±0.06
PAIR lqi+lqr      100.0± 0.0 / 0.98±0.08 / 0.28±0.04           99.4± 0.5 / 1.51±0.07 / 0.73±0.11
PAIR mpcof+lqr    100.0± 0.0 / 1.12±0.08 / 0.41±0.05          100.0± 0.0 / 1.39±0.08 / 0.67±0.11
PAIR mpcof+mpc    100.0± 0.0 / 1.29±0.16 / 0.55±0.18          100.0± 0.0 / 1.25±0.04 / 0.58±0.03
PAIR lqr+mpc      100.0± 0.0 / 1.30±0.11 / 0.57±0.11          100.0± 0.0 / 1.41±0.17 / 0.73±0.15
PAIR mpcof+lqi    100.0± 0.0 / 1.44±0.10 / 0.78±0.27          100.0± 0.0 / 1.42±0.06 / 0.59±0.04
PAIR lqi+pid       99.6± 0.8 / 1.62±0.30 / 0.90±0.52           99.8± 0.4 / 1.59±0.16 / 0.79±0.21
PAIR lqr+pid       99.6± 0.8 / 1.38±0.10 / 0.72±0.10           98.8± 1.5 / 1.97±0.16 / 1.61±0.21
PAIR mpc+pid       99.8± 0.4 / 1.64±0.24 / 1.14±0.40          100.0± 0.0 / 1.34±0.05 / 0.79±0.09
PAIR mpcof+pid     98.4± 3.2 / 1.73±0.24 / 1.19±0.45          100.0± 0.0 / 1.30±0.08 / 0.63±0.08
FULL4 mean        100.0± 0.0 / 1.16±0.08 / 0.29±0.09          100.0± 0.0 / 1.47±0.08 / 0.64±0.11
FULL4 median      100.0± 0.0 / 1.07±0.06 / 0.26±0.06          100.0± 0.0 / 1.39±0.10 / 0.64±0.11
FULL5 mean        100.0± 0.0 / 1.26±0.09 / 0.35±0.13          100.0± 0.0 / 1.50±0.08 / 0.68±0.13
FULL5 median      100.0± 0.0 / 1.19±0.11 / 0.40±0.18          100.0± 0.0 / 1.38±0.12 / 0.72±0.14
```

Against the bar (best solo steady: 0.53 on s002, 0.74 on s003):

- **FULL4 (mpcof+lqi+lqr+mpc), median vote: 0.26 / 0.64 at 100.0% stable on both seeds
  — the strongest passer.** Beats the best solo by 51% / 14% and the control cells by
  59% / 33%. Disclosure: dropping pid is a rule ("exclude the member that failed its
  own solo screen"), and pid's failure was visible in the markers before any committee
  was scored — but the markers ARE report-seed data, so FULL4's a-priori standing is
  one notch below FULL5's. Both pass; the paper can carry both.
- **FULL5 mean (the fully parameter-free vote): 0.35 / 0.68 at 100.0% — passes.** The
  mean vote absorbs pid at k=5; every +pid PAIR is worse than its non-pid counterpart
  (the July "weak member drags the vote" lesson holds at k=2, not at k=5-mean).
- **mpcof+lqr: 0.41 / 0.67 at 100.0% — passes.** Deep single-seed pairs (lqi+mpc 0.26,
  lqi+lqr 0.28 on s002) shed 0.4–0.6pp stable on s003 and, being ranked on the
  published partition, stay exploratory.

**The hold-floor connection: FULL4 posts 0.26–0.29° on seed 31337002 — through the
0.35° bar that L1/L1b/L2/L3/L4 all failed to reach — and 0.64° on the harder seed.
Vote diversity is the first mechanism in this programme that moved the floor at all.**

### Committee SIZE sweep (k=3 and k=4, persisted 09/08/2026)

The trio/quad sweep originally went to stdout only; it was re-run to disk on
09/08 — `logs/controller/committee_scoring/CMTSCORE_s3133700{2,3}_combos34.out`
(same protocol, `--pairs --combo-sizes 3,4 --agg both`). Every previously-quoted row
reproduces exactly, including FULL4-median 1.07/0.26 and FULL5. Best per size, by
worst-case and by average steady across the two base seeds:

```
k=1   worst 0.74  [mpcof]                       avg 0.670  [lqi]
k=2   worst 0.58  [mpcof+mpc mean]              avg 0.475  [lqi+mpc mean]
k=3   worst 0.57  [mpcof+lqi+mpc mean]   <-best avg 0.445  [lqi+lqr+mpc mean]  <-best
k=4   worst 0.64  [FULL4 median]                avg 0.450  [FULL4 median]
k=5   worst 0.68  [FULL5 mean]                  avg 0.515  [FULL5 mean]
```

Top trios (all 100.0% stable on both seeds), `s002 steady / s003 steady`:
`mpcof+lqi+mpc` mean 0.42/0.57 · median 0.45/0.59 · `lqi+lqr+mpc` median 0.30/0.60,
mean 0.25/0.64 · `mpcof+lqr+mpc` median 0.39/0.65 · `mpcof+lqi+lqr` mean 0.43/0.65.

1. **k=3 matches or beats k=4/5 on both criteria** (its average edge over k=4,
   0.445 vs 0.450, is a tie; the worst-case edge 0.57 vs 0.64 is real). Three lookup
   tables instead of four is a 25% inference cut at ~820 instr/step each.
2. **pid drags the vote in 16/16 paired trio→quad comparisons** (0 better, 0 same;
   Δsteady +0.03 to +0.21°). An earlier note said 8/8 — that counted committee SHAPES;
   the paired count per shape × seed is 16/16. The eight pid-free trios rank strictly
   above all twelve pid-bearing trios: a clean, gap-free split.
3. **The aggregator is coupled to k, not a free knob.** At k=3 median is markedly worse
   than mean on pid trios (mpcof+lqi+pid 1.17 vs 0.87 worst-case): with three members a
   bad member IS the median whenever the two good ones straddle. At k=5 the ordering
   reverses (FULL5 mean 0.35 < median 0.40 on s002) — pid can no longer reach the middle,
   so median becomes an outlier-rejector. **Median = veto at small odd k, filter at large
   odd k.** Prefer MEAN at k=3.

These trios are ranked on the published report seeds and therefore stay EXPLORATORY;
FULL4/FULL5 remain the a-priori rows. `mpcof+lqi+mpc` clears the bar on both seeds
(0.42/0.57 vs 0.53/0.74), which makes "k=3 suffices" a strong hypothesis pending
seed 31337004.

### Classical rivals, same engine, same episodes (CRN with the committee rows)

`score_classical_baseline` (one physics engine for WNN and rivals; firmware PID cascade
via `af_pid_*`), same 5 report-seed pools, L4C, cf21_brushless. Teachers are
training-free, so one table serves both base seeds:

```
CLASSICAL pid     100.0± 0.0 / 2.82±0.97 / 1.97±1.01
CLASSICAL mpc     100.0± 0.0 / 1.18±0.19 / 0.76±0.27
CLASSICAL lqr     100.0± 0.0 / 1.03±0.10 / 0.55±0.15
CLASSICAL lqi     100.0± 0.0 / 0.90±0.08 / 0.44±0.08
CLASSICAL mpcof   100.0± 0.0 / 0.72±0.01 / 0.00±0.00
```

Read against the WNN rows: the FULL4 committee (0.26/0.64) **beats classical PID
(1.97) outright, beats MPC (0.76) on both seeds, and brackets the linear-optimal
teachers** (beats lqr/lqi on s002, loses to them on s003). The un-closed gap is
**mpcof at 0.00±0.00 steady**: a receding-horizon optimizer with a disturbance
observer trims the L4C bias exactly. That is the remaining mechanism — consistent with
L1's refutation (handing the student d̂ as an INPUT did nothing; the teacher's
advantage is the integral/observer LOOP, not the signal) and with the yaw
dead-reckoning finding. The compute framing belongs next to it: mpcof spends a
per-step optimization; the committee approximating it is k lookup tables (mcu/: 820
instructions/step solo on a Cortex-M4, measured).

### Harness provenance (why these numbers can be trusted)

The first scoring pass (08/08, preserved under
`logs/controller/committee_scoring/INVALID_20260808_thresholdbug/`) reported solos
20–40× worse than their own runs' held-out. Its own built-in control caught it: a SOLO
row must reproduce that member's marker triple. Seven defects were found and fixed
across two commits (`3a9be6f6`, `6d77e697`, `ad3ea430`):
thresholds refit on report seeds (the address function shifted), base seed passed
where the DERIVED train seed belongs, no disturbance, no airframe (the L2
wrong-aircraft bug in a new script), `num_eval_folds` 1 vs 5, members read from
`best_genome` (a lucky-fold snapshot — `final_population[0]` is the published result;
they differ in 5/8 checkpoints), and a `winner_only.yaml.gz` cache COLLIDING between
stage1/stage4 in one `_stages` dir. After all seven: **all 10 solos reproduce their
marker rows exactly, mean and SD.** Residual caveat: solos score on the evaluator's
fold-pool ICs, committees on `draw_ics(report_seed)` — same distribution, not CRN
across that one comparison; the gaps exceed the per-seed SDs by an order of magnitude.

Next steps queued from this result: NEURONS-stage member pool (all 10 runs) and GRID
members (runs 8–10, `stage0_grid` fix) for cross-stage committees; a same-pool
1-member row if a reviewer asks; and the observer gap (mpcof's 0.00) as the successor
programme to the closed single-run hold-floor levers.

Secondary observations worth carrying forward:

- **Seed 31337003 is simply the harder seed** — every arm lands worse on it (control 0.95 vs
  31337002's 0.64). Always compare WITHIN seed.
- **MEMORY sometimes ends worse than NEURONS on seed 31337003** (S16-plain 1.23 vs 0.94;
  L1's C10+d̂ 1.45 vs 1.27) — twice, across different weightings, so it looks like a
  property of that seed's MEMORY stage rather than of either lever. Worth a look before
  that seed is reused.

### Build provenance (both levers shipped before they ran)

- **L1 (ABI 22, `c0a105e0`):** the mpcof observer runs inside `WnnController.step()` from
  the student's own throttle accumulator and gyro finite-difference, adding 3 features. `b`
  comes from the exposed `calibrate_control_gains`; `--obs-dhat` requires `--airframe`.
  GPU/CPU parity test `gpu_dhat_feature_matches_cpu_closed_loop` is mutation-verified.
- **L2 (`c7fa0a2d`):** the guard is lifted. `score_controllers_metal` takes `af_pid_*` and
  builds the cascade from `AttitudePidFirmwareRs` (filter coefficients + decimation from the
  Rust cascade itself); `dagger.py::make_residual_baseline` returns `AttitudePidFirmware` on
  a cascade airframe (`pd` stays legacy — it is the Ki=0 ablation floor);
  `EpisodeConfig.cascade_kwargs()` hands the GPU the CPU's controller.
- **L1b (`scripts/l1b_s16_dhat_chain.sh`):** runs INTERLEAVED (both variants at seed
  31337002, then both at 31337003) so the first two runs already answer "did the weighting
  move steady at all"; every non-weight flag is copied from `l1_dhat_chain.sh`, including
  the 5-generation NEURONS cap, so the only differences across the 2×2 are the fitness
  weights and the presence of `--obs-dhat`.

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

---

## Alphabet probe — REFUTED at its bar (09/08/2026)

Pre-registered in `docs/controller_reassessment_2026_08.md` §4: the delta decode
quantizes the correction (`decoded = 0.5 + (ΣE−ΣI)/levels` → ±delta_max), so at
levels=16 / delta_max=0.1 the smallest nonzero per-step correction is 0.0125 PWM.
mpcof's correction is continuous and posts 0.00° steady. Arms: levels ∈ {32, 64} ×
seeds {31337002, 31337003}, teacher lqi, otherwise the committee control shape.
**Bar: beat the same-seed levels=16 control's steady on BOTH seeds without losing
stable.**

```
              s31337002                       s31337003
L16 control   99.8±0.4 / 1.11±0.06 / 0.53    100.0±0.0 / 1.58±0.08 / 0.81
L32           99.8±0.4 / 1.22±0.14 / 0.52     99.8±0.4 / 1.60±0.18 / 0.99
L64          100.0±0.0 / 0.86±0.02 / 0.31    100.0±0.0 / 1.43±0.15 / 0.91
```

**Verdict: REFUTED.** L32 is a null on s002 (0.52 vs 0.53) and worse on s003
(+0.18). L64 breaks the floor spectacularly on s002 — **0.31° at 100.0% stable with
the best err of any lqi run (0.86)** — and loses on s003 (+0.10). One seed each way
fails the bar. Increment resolution is not a reliable hold-floor lever.

Two caveats that keep the s002 number honest rather than promoting it: L64's GRID
stage is catastrophic on both seeds (15.0% and 36.8% stable) — the fine alphabet is
entirely GA-rescued, not a better starting point; and cell counts triple
(Σ36M vs Σ11M), which matters for the FPGA/MCU footprint claim.

⇒ Per the pre-registration, this promotes the STRUCTURAL route (sn>0 / state) over
the resolution route.

## Committee — 3-seed closure: the bar holds, the winner does not (09/08/2026)

Seed 31337004 was flown to break the 002-vs-003 tie. It did not break it — it
dissolved it.

```
                  s31337002              s31337003              s31337004
best SOLO         lqi   0.53             mpcof 0.74             mpcof 1.22
best k=2          0.26 lqi+mpc           0.58 mpcof+mpc         1.17 mpcof+lqr
best k=3          0.25 lqi+lqr+mpc       0.57 mpcof+lqi+mpc     1.24 mpcof+lqi+lqr
best k=4          0.26 FULL4             0.63 mpcof+lqi+mpc+pid 1.30 FULL4
best k=5          0.35 FULL5             0.68 FULL5             1.38 FULL5
FULL4 median      100.0/1.07/0.26        100.0/1.39/0.64        100.0/1.75/1.30
FULL5 mean        100.0/1.26/0.35        100.0/1.50/0.68        100.0/1.82/1.38
```

1. **The winner changes on every seed** — top-1 is `lqi+lqr+mpc`, `mpcof+lqi+mpc`,
   `mpcof+lqr`. Three seeds, three committees.
2. **Cross-seed rank agreement is near-nil.** Kendall-τ over the 52 committees
   common to all three seeds: 002↔003 **+0.14**, 003↔004 **+0.14**, 002↔004 +0.48.
3. **The seed effect matches the entire committee effect.** Mean-committee steady per
   seed 0.58 / 0.74 / 1.48 — a 0.91° between-seed range against within-seed spreads
   of 0.67–1.04°.
4. **The mechanism survives regardless**: `mpcof+lqr` beats its best solo on all three
   seeds at 100% stable (0.41/0.67/1.17 vs 0.53/0.74/1.22), as do FULL4 and FULL5.

**⇒ Report the a-priori FULL4 and FULL5 rows across three seeds and DECLINE to name a
best committee.** The k=3 finding survives as a DEPLOYMENT claim (a trio matches k=4/5
at 25% less inference), not as an endorsement of any particular trio. On s004 the size
ordering even inverts (k=2 best, monotone degradation with k) — when every member
carries a large bias error the vote has uncorrelated noise to cancel, and it has none.

## Tie-fix A/B — underpowered, weakly negative (09/08/2026)

`ALP2_lqi_L16` (post-8b839a30: fractional tie-aware ranks + top-3 stage selection) vs
`CMT_lqi` (pre-fix), same recipe byte-for-byte, same base seeds.

```
                  OLD (CMT_lqi)              NEW (ALP2)
s31337002 GRID    85.0±29.0/2.34/1.39        85.0±29.0/2.34/1.39   <- IDENTICAL
          NEURONS 100.0±0.0/1.08/0.48        99.8±0.4/1.21/0.67
          MEMORY   99.8±0.4/1.11/0.53        99.6±0.5/1.21/0.60
          HEADLINE MEMORY  99.8/1.11/0.53    NEURONS#2 99.6/1.22/0.57   (+0.04)
s31337003 GRID    72.6±36.5/6.55/6.53        72.6±36.5/6.55/6.53   <- IDENTICAL
          NEURONS  99.2±0.7/1.66/0.91       100.0±0.0/1.63/0.97
          MEMORY  100.0±0.0/1.58/0.81        99.6±0.5/2.23/2.28   <- collapsed
          HEADLINE MEMORY 100.0/1.58/0.81    NEURONS#0 100.0/1.63/0.97   (+0.16)
```

The GRID rows are bit-identical on both seeds — same pools, deterministic up to the
GA — so every difference below GRID is attributable to the changed code path.
**New is worse on headline steady 2/2 (+0.04, +0.16).** Read honestly: n=1 per seed,
deltas at or inside the ±0.11–0.13 run-to-run SDs, and the comparison is CONFOUNDED
(ALP2 carries the tie fix AND top-3 selection). The s003 gap is mostly selection: that
run's MEMORY stage collapsed (2.28) and the new selector correctly refused it. The
correctness case for the fix is independent of this (positional tie-breaking is
indefensible whichever way the noise falls) — but the claim "no performance impact" is
NOT supported; it is UNMEASURED with a weak negative signal at n=2. The powered
version of this test is free on the IDS side: each SP100 cohort's first 10 seeds pair
against the old QUAD baselines (n=10 per dataset).

Live confirmation that the top-3 machinery is doing real work: 3 of the first 4 runs
under it headlined a RUNNER-UP (MEMORY#2, MEMORY#2, NEURONS#2), and each time the
selected genome's OWN report-seed triple was scored for the headline.

---

## The thermometer was calibrated on the wrong distribution (10/08/2026) — PAPER-CRITICAL

Every controller number in this document, in `controller_reassessment_2026_08.md`, and in
the L1–L4 / committee / alphabet-probe programmes was measured through an input encoder
fitted on a state distribution the controller never visits. Three independent defects,
found in sequence, all instances of one class: **the calibration distribution did not
match the flown distribution.**

`fit_thresholds_from_pid_rollouts` is a *learned* encoder — it quantile-fits each
feature's thermometer ladder to an empirical sample of that feature. So mis-fitting it is
a train/serve skew in the INPUT REPRESENTATION, upstream of everything. No amount of GA
search over connectivity or memory can repair a feature the encoder has already collapsed
to a constant. That is why the defect survived four refuted lever programmes: **every
lever was tested *through* the broken encoder**, so all four measured a system whose
binding constraint was elsewhere.

### The three defects

| # | Defect | Fixed in | Status |
|---|--------|----------|--------|
| 1 | **Wrong regime.** Rollout config hardcoded `max_initial_tilt_rad = 30°` while every production recipe flies `--tilt 5.0`. | `4514d5c9` | fixed, `episode_config` threaded through all 4 phased_ga sites + grid search |
| 2 | **Wrong plant.** PID was rolled out on a bare `AttitudeSim()` with NO disturbance, while every run flies L4C. | `8077d176` | fixed, mirrors `training.py` per-episode arming |
| 3 | **Wrong policy.** The fitter rolls out PID — a *better* controller than the student — so the ladder never covers the student's own excursions. DAgger covariate shift, in the input representation. | `094cefd2` + `a1a6d83a` + `5f3d113c` | built, `--threshold-refit-from-student`, default OFF, **NOT YET FLOWN** |

Defects 1 and 2 interact, and the interaction is why this went unnoticed for so long:

```
ladder span, L4C armed vs clean plant (15 feats × 8 bits, production spec):
  at 30° ICs   1.02×   <- the omission is invisible
  at  5° ICs   1.86×   <- the omission dominates
```

**The hardcoded 30° was MASKING the missing disturbance.** Wide initial conditions produce
a transient so large it dominates the sample spread, so leaving the sustained bias out
changed the fitted ladder by 2%. Fixing defect 1 alone (narrowing to the flown 5°) removed
the mask without fixing what it hid — which is exactly why `calib=5` came back **2.9×
worse on steady and degraded even GRID (1.39 → 5.34)**, the stage before any GA search
runs. A fix that makes things worse was the signal that a second defect was present.

### Span is not coverage — the metric that actually orders hold

The natural reading of defect 1 is "narrower fit ⇒ finer near-zero bins ⇒ better hold."
That reading is **wrong**, and it is worth stating plainly because it cost a day.

The quantity that orders the flown results is the **fraction of the flown distribution
that falls OUTSIDE the ladder** — saturating to an all-0 or all-1 thermometer code, at
which point the feature carries no information at all. Measured on the per-axis
**integral** channels specifically (they carry the sustained bias that the `steady` metric
scores):

```
  outside%   headline steady   fit
    56.5%         1.53°        calib 5.0,  clean-plant fit
    25.1%         0.80°        calib 2.5,  disturbance fit
    20.1%         0.53°        calib 30,   the legacy control
```

A 90° fit has ~3× the span of a `tilt5 q=.005` fit **and saturates more**, because wide
ICs spend the finite threshold budget on transient states while the tails of the actually-
flown distribution stay uncovered. Span and coverage are different quantities and only
coverage tracks the metric.

The second lever on coverage is the **outer quantile position**. The default ladder sits
at percentiles `1/(b+1) … b/(b+1)`; at `b=8` that is `0.111 … 0.889`, so **~22% of the
operating distribution falls outside the ladder by construction**, before any
mis-calibration. `--outer-quantile` reaches further into the tails (`0.02` spans
`[0.02, 0.98]`).

Scanning calibration tilt 1°…90° against `outside%`:

```
  best pure tilt        ≈ 59°     8.5% outside
  tilt 30 + q=0.005               4.3% outside
  tilt  5 + q=0.002               5.3% outside
```

Two things follow. The legacy 30 **was not even the best accidental value** — a pure-tilt
search would land near 59. And **every outer-quantile candidate beats every tilt-only
one**, which says the quantile position is the stronger of the two levers.

⚠️ **`outside%` is a PROXY.** It is validated on three flown points, two of which are
confounded (below). The live arm exists precisely to test whether it is causal or merely
correlated. Do not report `outside%` as a result; report it as the hypothesis being tested.

### ⛔ RETRACTED: the calib 2.5 vs calib 5.0 comparison

The `c2.5`-vs-`c5.0` cells are **CONFOUNDED and must not be reported as a tilt
comparison.** The defect-2 fix (`8077d176`, disturbance-armed fitting) landed at 13:11 UTC
on 10/08, *between* cells 2 and 3 of that sweep. The chain launches each run from source
at run start, so the later cells ran different code from the earlier ones. The two arms
differ in calibration tilt AND in whether the fitter saw the disturbance.

This is the fourth instance in one day of the same operational failure and it is now a
hard rule (`feedback_never_deploy_while_chain_armed`): **never edit Python or install a
wheel while a chain is armed.**

### The live test — the outer-quantile arm

Six runs, uniform current code, disturbance-armed fitting live throughout:
`q ∈ {none (= the 30° control), 0.02, 0.005} × seeds {31337002, 31337003}`, teacher `lqi`,
otherwise the committee control shape. **Controls are RE-FLOWN** — the legacy `CMT_lqi`
rows predate both fixes and are not a valid baseline.

**Bar (pre-registered): beat BOTH `q=none` control runs' headline steady without losing
stable.**

Controls, held-out over 5 report seeds, as stable%/err°/steady°:

```
                    s31337002                     s31337003
  GRID        96.8 / 1.61 / 1.28            97.4±1.0 / 2.33±0.57 / 2.11±0.74
  NEURONS    100.0 / 1.02 / 0.63           100.0±0.0 / 0.89±0.03 / 0.36±0.02
  MEMORY     100.0 / 1.01 / 0.54           100.0±0.0 / 0.92±0.03 / 0.37±0.03
  HEADLINE   NEURONS#0 100.0/1.02/0.63     MEMORY#1 100.0/0.88/0.36
```

The control band is **0.36–0.63° headline steady at 100% stable**. Cell 1 reproduces the
legacy control (0.53) closely enough to say the arm is calibrated against the historical
record; its GRID spread is ~14× tighter than the legacy runs'.

Two cautions for scoring the `q` cells:

1. **The headline stage differs between the two controls** (NEURONS#0 vs MEMORY#1).
   Compare headline-to-headline, never NEURONS-to-NEURONS.
2. **GRID moves opposite to the trained stages** across the two control seeds (+0.83 vs
   −0.27 on steady), with a much fatter spread (±0.74 vs ±0.02). A `q` effect visible only
   in GRID is an encoder artifact, not a control win.

### What this does and does not invalidate

**Does not invalidate the refutations.** L1/L1b/L2/L3 and the L4 verdict were all
*within-arm* comparisons — control and treatment shared the same encoder, so the
comparison is internally valid. They remain refuted *at the encoder they were run on*.

**Does reframe the alphabet probe.** L64 refined the ACTION alphabet and moved one seed
but not the other — the signature of a second binding constraint. Perception was that
constraint. The two limits are multiplicative, so the probe's verdict ("increment
resolution is not a reliable lever") should be read as "not a reliable lever *while
perception is the tighter constraint*", not as a closed question.

**Does put every ABSOLUTE number on notice.** The hold floor, the teacher-quality-does-not-
propagate claim, and the committee bar were all measured through the mis-fitted encoder. If
the outer-quantile arm moves the floor, the absolute figures need re-flying before
publication; the relative/structural claims survive either way.

### E1 — the coverage 2×2 (PRE-REGISTERED 10/08/2026, not yet armed)

Written before any cell ran. `scripts/e1_coverage_2x2_chain.sh`.

**Design.** `enc ∈ {c30, q} × refit ∈ {off, on} × seeds {31337002, 31337003, 31337004}`
= 12 runs, ~6h, interleaved (all four combos on 002, then 003, then 004).

**Why a 2×2 rather than stacking refit on the outer-q winner.** Both factors attack the
same defect — `outside%`. The encoder package moves the ladder; the refit fixes the
sample the ladder is fitted on. Stacking two coverage fixes can **over-correct**: the
8-bit budget is finite, so buying tail coverage twice coarsens the near-zero region,
which is precisely the failure that made `calib=5` 2.9× worse. A greedy chain reads a
joint over-correction as "refit does not work". **The interaction term is the point.**

**Bar.** One combination must beat the `c30/refit-off` control's HEADLINE steady on
**all three seeds** without losing stable. Refutation: none does ⇒ coverage does not
order the hold floor, `outside%` is correlational only, and the structural route
(sn>0 / state neurons, reassessment §5) is what remains.

**Scoring rules, fixed in advance:**
- Compare **headline-to-headline**, never NEURONS-to-NEURONS. The outer-q controls
  headlined NEURONS#0 and MEMORY#1 respectively; a fixed-stage comparison compares two
  different objects.
- **GRID is not the read-out.** Across the outer-q controls GRID moved opposite to the
  trained stages (+0.83 vs −0.27) with a ~35× fatter spread. A GRID-only effect is an
  encoder artifact.
- Report both main effects **and** the interaction, not a winner.

⚠️ **Factor A is a PACKAGE.** `c30` = `--threshold-calib-tilt 30`; `q` =
`--threshold-calib-tilt 5.0 --threshold-outer-quantile Q`. These differ in calibration
tilt AND quantile position — the same conflation the in-flight outer-q arm carries. It
is deliberate (it asks the question that gates the recipe freeze) but **must never be
written up as "the outer quantile is the lever"**. Isolating quantile from tilt needs
its own arm.

**Two properties that make the A/B clean**, both verified before the arm was designed:
- The refit's regrid is `stage0_grid(args, ec, seed, thresholds_override=thr2)` — same
  seed, same grid points. If the refit were a no-op the second grid would reproduce the
  first bit-for-bit, so refit-on gets no extra draw; the only asymmetry is wall-clock.
- The chain's pre-flight aborts unless it finds `[thr-refit] … REGRIDDING` in the smoke
  run's output. The refit degrades to a silent placebo when the student's samples are
  swamped by the teacher pool (measured: 2 samples/feature moved the ladder 1.00×), and
  a placebo that "looks implemented" is the failure mode worth spending a guard on.

### E2 — `--delta-gamma`, queued behind E1 (PRE-REGISTERED 10/08/2026)

`γ ∈ {1.0 control, 2.0} × 3 seeds`, flown on **E1's winning encoder**. Same bar.

This is the L64 alphabet question re-asked at **1/3 the footprint**: γ=2 makes the
finest increment ~8× finer at `levels=16` with the same cell count, where `levels=64`
cost 3× cells for a gain that held on one seed and not the other. It flies *after* E1
because L64's verdict is confounded by the encoder defect — action resolution can only
be read cleanly once perception is settled, the two limits being multiplicative.
