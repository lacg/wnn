# L4 teacher screen — Crazyflie 2.1 Brushless, L4C

## STATUS: screen COMPLETE (06/08) + all three hold-floor levers RUN and REFUTED (07/08)

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
