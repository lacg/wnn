# Controller fitness: viability gate + fixed-scale weights

**Status:** SPEC — no code written. Awaiting Luiz's sign-off.
**Date:** 21/08/2026
**Supersedes in practice:** the C10/S16 weight-sweep programme (see §7)

---

## 1. The problem, in one paragraph

The controller fitness is a **weighted sum over pool-relative normalised ranks**
of four metrics (err² .40 / stable .30 / jerk .20 / mono .10). A weighted sum is
*compensatory by construction*: arbitrarily bad performance on one term can be
bought back by good enough performance on another. Two of the four terms turn out
to measure the **representation** rather than the flight, and both have degenerate
optima that a non-flying genome attains perfectly. The result, measured three
times across both aggregations, is that stage-select crowns genomes that are worse
on both primary metrics.

## 2. Evidence

### 2.1 The degenerate members of every pool

From the 10 fitness-A/B runs, 90 candidates (top-3 of each stage, val rows):

```
agg        stage    stable      err°     jerk     mono
harmonic   MEMORY     0.0%     93.57    0.0004    0.0
harmonic   MEMORY     0.0%     86.30    0.0015    0.0
harmonic   MEMORY     0.0%     88.73    0.0024    0.0
```

A tumbling aircraft scores **jerk 10-30x better than a healthy genome** (healthy
pool median 0.031) and **mono perfectly** (0.0). Both "smoothness" terms are
maximised by not flying.

### 2.2 Why: jerk measures duty cycle, not smoothness

With `levels_per_motor=16`, `delta_max=0.1`, delta-control on, one level step per
motor is `2(0.1)/15 = 0.01333`. Euclidean `||dpwm||` for 1/2/3/4 motors stepping
together is `0.01333 / 0.01886 / 0.02309 / 0.02667`. The observed WNN range
0.017-0.037 **is that lattice**. Jerk is therefore

```
mean jerk  =  quantum  x  transition RATE
```

The quantum is fixed by the alphabet; only the *rate* is under a genome's control.
So minimising jerk means minimising how often the controller acts, and the global
argmin is a controller that never actuates.

Measured reference (PID, estimator-fed, cf21_brushless/L4C, 5 report seeds x 100
episodes, `scripts/calibrate_fitness_scales.py`):

```
stable 100.0% +- 0.00 | err 1.80° +- 0.22 | steady 1.07° +- 0.36 | jerk 0.00163 +- 0.00001
```

The PID sits at **0.122 quanta** — it emits continuous commands. A delta-coded WNN
cannot match that per-transition, only by transitioning less.

### 2.3 Why mono is not a flight metric

`monotonicity_violations` (`controller/controller.rs:6363`) counts **thermometer-
pattern violations in the WNN's output encoding** — "a 0->1 transition after any 1
has been observed". It is a property of the output codification. A classical
teacher has no thermometer bank, so mono is *undefined* for it and cannot be
calibrated from any rollout.

### 2.4 The plant does not model what jerk is a proxy for

`cf21_brushless` carries mass, arm_length, k_thrust, k_drag, inertia, gravity —
and **no motor time constant, no slew limit, no low-pass**. PWM maps to thrust
instantly. There is no simulated mechanism by which jerk degrades flight. Any
`J_max` is an assertion about hardware this simulator does not represent.

### 2.5 The consequence, measured

Stage-select crowned a genome **dominated on both primaries on its own selection
data** in 2 of 10 runs (both zscore arms, seeds 31337002 and 31337006). Correcting
selection with a dominance filter moved the A/B verdict from `stable 4/5, err 3/5`
to `stable 5/5, err 4/5` in zscore's favour.

## 3. Design

Three changes, in order of importance.

### 3.1 Viability gate (the actual fix)

Split the objective into a **qualifying** stage and a **preferential** stage:

```
STAGE 1  gate, NON-compensatory : does it FLY?   stable >= S_min  AND  err <= E_max
STAGE 2  weights, compensatory  : among flyers, rank on err/stable/jerk/mono
```

A tumbler never reaches stage 2, so its jerk 0.0004 and mono 0.0 are worth nothing.
A flyer with mono 5 competes normally. **This is the change that removes the
pathology, and it needs no hardware number.**

### 3.2 Feasibility ranking (Deb's rules) — so the gate never kills the search

A hard gate gives no gradient in generation 0, when nothing flies. Standard
constrained-GA handling restores it:

```
feasible beats infeasible                       always
two feasible    -> better weighted objective    wins
two infeasible  -> smaller constraint VIOLATION wins
```

Violation = `max(0, S_min - stable)/S_min + max(0, err - E_max)/E_max`, i.e.
normalised and summed. Generation 0 ranks by "how close to flying"; the moment any
genome flies, flyers dominate. No cliff, no dead search.

### 3.3 Fixed practical scales (replaces pool-relative normalisation)

Today both combines normalise each metric by the **pool's own spread** (rank
position, or median/MAD for zscore). Consequences: a fitness value is meaningless
outside its pool (`fit=1.0000` in one run, `-1.86` in another); and a metric with a
small practical spread has its noise amplified to look like a large advantage.

Replace with absolute per-metric scales, so one unit means the same thing in every
run:

```
err     1 unit = 1.0°           (against a 5.0° stability threshold)
stable  1 unit = 5 pp
jerk    1 unit = 0.01333        ONE QUANTUM — the only natural unit the metric has
mono    1 unit = 1 violation
```

`z_i = (x_i - ref_i) / scale_i`, weighted sum, lower better. Side benefits: fitness
becomes comparable across generations/runs/seeds; the tick's `(=)` becomes
interpretable; cross-run fitness plots become legal.

## 4. Gate thresholds

Admission over the 90 banked A/B candidates:

```
 S_min   E_max   admitted     %    per-stage (GRID/NEURONS/MEMORY)
     0     999         90   100%   30 / 30 / 30
    50      15         69    77%   12 / 30 / 27
    60      10         64    71%    8 / 29 / 27
    70       8         63    70%    7 / 29 / 27
    75       6         59    66%    4 / 29 / 26
    80       5         55    61%    2 / 27 / 26
    85       5         42    47%    1 / 20 / 21
    90       4         27    30%    0 / 11 / 16
```

**RECOMMENDED: `S_min = 70`, `E_max = 8.0°`.**

Rationale: it excludes every degenerate member (all tumblers are 0-16% stable,
18-94° err) while admitting 70% of candidates and 29/30 NEURONS + 27/30 MEMORY —
so the gate culls weak GRID points and nothing else. Tighter gates (85/5) start
discarding healthy MEMORY genomes, which is selection pressure the *weights* should
apply, not the gate. The gate's job is to exclude the unflyable, not to pick winners.

`E_max = 8.0°` also sits comfortably above the 5.0° `stable_deg` threshold, so a
genome that is stable-by-definition is never gated out on error.

## 5. Where it lives

Three consumption sites share `ram_core::fitness` via `combine_flat`:

| Site | Entry | Effect of changing it |
|---|---|---|
| **Search fitness** (GA elites, tournament, patience) | `ga_strategy.search_aggregation()` -> `FitnessCalculatorControllerHarmonic` -> `wnn.control._accel.fitness_combine` -> `ram_core::fitness::combine_flat` | Changes the SEARCH TRAJECTORY. Banked results become non-comparable; everything re-runs. |
| **Stage-select** (headline from 9 candidates on val seeds) | `ga_strategy.select_aggregation()`, `phased_ga.py:1614` | Pure post-processing. Re-derivable from banked val tables, no re-runs. |
| **Grid winner** | `controller_grid_search.py:136` | Same combine; picks the grid seed for stage 1. |

**Implementation lands in `ram_core::fitness`** (one place, both wheels see it), so
`ram_accelerator` and `ram_controller` both rebuild. Only the controller uses the
gate; IDS keeps the current path unless we opt it in separately.

## 6. Proposed API

```rust
/// Viability gate + fixed-scale weighted combine. Lower score = better.
///
/// Candidates failing the gate are ranked BELOW every feasible candidate, and
/// among themselves by normalised constraint violation (Deb). Feasible
/// candidates are ranked by the weighted sum of (x - ref)/scale per column.
pub fn gated_combine(
    columns:      &[MetricColumn],   // as today
    scales:       &[f64],            // fixed practical scale per column
    refs:         &[f64],            // reference point per column
    gate_stable:  f64,               // S_min, e.g. 70.0
    gate_err:     f64,               // E_max, e.g. 8.0
    stable_col:   usize,             // which column is stable
    err_col:      usize,             // which column is err
) -> Result<Vec<f64>, String>
```

Existing `rank_combine` / `zrank_combine` stay untouched so IDS and every banked
recipe remain bit-identical. New behaviour is opt-in via a new
`--fit-aggregation gated` mode plus `--gate-stable` / `--gate-err` flags.

## 7. Impact on the programme

| Artefact | Fate |
|---|---|
| The 10 A/B runs (raw rollouts) | **Survive** — unchanged measurements |
| The A/B *verdict* | Already re-derived with a dominance filter: zscore 5/5 stable, 4/5 err |
| 9 alt-weight arms (task #4) | **Re-scope.** Under a gate + fixed scales the weight simplex is a different space |
| C10/S16 weight sweeps, bits ladder (task #6) | **Re-scope.** All decided under rank-WHM with the pathology live |
| Sweep ladder (task #2) | Should start *after* this lands, not before |

The weight-sweep programme largely dissolves: with viability gated and scales
fixed, what remains is a single ratio between err and stable, plus two bounds you
can justify from intent rather than search.

## 8. What is NOT in this spec

- **`J_max` as a hard bound.** Dropped for now — §2.4 shows it cannot be validated
  in a plant with no motor model, and §3.1 shows it is not load-bearing once the
  gate exists. Revisit as a hardware safety bound when a bench measurement exists.
- **Removing `mono`.** With the gate in place, leaving mono weighted is harmless.
  Whether an encoding-hygiene metric belongs in a flight objective at all is a
  separate decision.
- **Any change to IDS.** Same structural argument applies there (optimise the
  score, constrain the decision), but it is a separate spec.

## 9. Testing

1. **Rust unit tests in `core/fitness.rs`** alongside the existing `combine_flat`
   tests: gate admits/excludes correctly; infeasible ranked below feasible;
   violation ordering among infeasible; scale invariance (doubling a scale halves
   that column's contribution).
2. **Regression oracle:** re-score the 90 banked A/B candidates under
   `gated_combine`. Assert that all three tumblers rank last, and that neither of
   the two dominated headlines (seeds 31337002/31337006 zscore) is selected.
3. **`cargo test -p ram_controller --lib --no-default-features`** — the 94 existing
   tests must stay green, including all 14 CPU/GPU parity sweeps.
4. **No behaviour change when the mode is not selected:** a run with
   `--fit-aggregation harmonic` must be bit-identical to today.

## 10. Open questions for Luiz

1. **`S_min = 70`, `E_max = 8.0°`** — accept, or prefer a different admission point
   from the §4 table?
2. **Gate at which sites?** Recommendation: *all three* (search, stage-select,
   grid), because a gate at selection only would still let the search spend its
   whole budget optimising toward tumblers.
3. **Keep `mono` weighted** (your stated preference, harmless with the gate), or
   drop it from the flight objective?
4. **Scales** — `err` 1.0°, `stable` 5 pp, `jerk` one quantum (0.01333), `mono` 1
   violation. Accept, or set differently?
5. **Re-run scope** once it lands: the full sweep ladder, or a smaller
   gate-vs-no-gate A/B first to measure what the change is worth?
