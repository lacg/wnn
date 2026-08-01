# The held-out threshold refit is an evaluation bug, not a calibration choice

Found 01/08/2026 while putting the WNN on the classical baselines' error-bar axis.

**One-line version:** `_holdout_report` refits the input thermometer thresholds on
the *report* seed, but the genome's cells were written at addresses computed under
the *train* seed's thresholds. Refitting re-quantizes the inputs, so the same
physical state maps to a different RAM address and the trained memory is read where
nothing was ever written. Measured cost: up to **+38.8 pp stable** and a collapse of
σ from ±13.8 to ±1.7.

## What a "threshold" is here

Not a decision threshold on the controller's output. It is the **input encoding** —
`NUM_FEATURES × bits_per_feature` floats that are the cut-points of the thermometer
encoder turning continuous sensor readings (angles, rates) into bits.

`src/wnn/control/evaluator.py:10-16` is explicit about the consequence:

> `state_connections, output_connections`: **which input bits each neuron addresses**…
> `thresholds`: per-feature thermometer thresholds

So **connections + thresholds together compute the RAM address.** Thresholds are part
of the address function.

They are fit by `fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=X)`:
run **PID** (the teacher — never the WNN) for 10 episodes under seed `X`, collect the
empirical sensor distributions, place thresholds at uniformly-spaced quantiles. The
WNN's own score never enters, which is why this was not previously suspected of being
a leakage-class problem. It isn't one. It is an *alignment* problem.

## Where it goes wrong

| site | seed used | correct? |
|---|---|---|
| `phased_ga.py:546` (stage 2-3 search) | `train_seed` | yes — training and eval share them |
| `phased_ga.py:887` (memory stage) | `train_seed` | yes |
| `phased_ga.py:785` (`_shell_holdout_compact`) | **`rs` = report seed** | **no**, for score-only genomes |
| `phased_ga.py:1124` (`_holdout_report`) | **`report_seed`** | **no**, for score-only genomes |

The two held-out paths refit per report seed. For a genome that carries trained
cells this is wrong: it is scored through a different address function than the one
its memory was built under.

**It is only wrong for the score-only path.** `_holdout_report` branches:

```python
use_score = (best_genome.cells is not None or ec.geometry is not None)
m = (ev.score_genomes([g]) if use_score else ev.evaluate_batch([g]))[0]
```

- `score_genomes` — pre-trained cells, no training. Thresholds MUST match training.
- `evaluate_batch` — trains fresh at eval time under whatever thresholds are supplied,
  so refitting on the report seed is legitimate there.

## Measurement

Frozen winners replayed on 5 report seeds (`scripts/rescore_winners.py`), both variants:

```
  cell                            per-report-seed         train-seed thresholds
  dfa_9feat_BINARY_s31337002      84.8± 9.4  3.4±0.4      89.6±2.4  3.2±0.1
  dfa_9feat_BINARY_s31337003      67.0± 6.2  4.4±0.4      87.6±1.9  3.4±0.1
  1layer_9feat_BINARY_s31337002   78.0± 7.8  3.7±0.4      86.2±1.2  3.4±0.1
  1layer_9feat_BINARY_s31337003   48.0±13.8  5.4±1.1      86.8±1.7  3.2±0.1
  1layer_9feat_BINARY_s31337004   63.8±11.7  4.5±0.7      90.0±2.6  3.2±0.1
  phaseA_9feat_BINARY_s31337002   85.6± 3.3  3.3±0.1      89.4±2.6  3.2±0.1
  dfa_9feat_QUAD_s31337002        36.4± 5.7  6.8±0.7      27.2±5.6  7.7±0.6   <- REVERSES
```

Five of six BINARY winners land **86.2–90.0% with σ 1.2–2.6**.

> **CORRECTION (01/08/2026).** This section originally read "every one above PID's
> **85.0% / 3.96°**". That comparator was wrong — 85.0 came from `phased_ga._pid_baseline`,
> which flew PID on episodes drawn from the RAW report seed while every WNN row is
> pool-seeded (`fold_pool_seed`). `scripts/compute_baselines.py` had documented this on
> 29/07 and been fixed; `_pid_baseline` had not. **The comparator is `baselines.json`:
> PID = 90.4±7.5% / 3.96±0.38° / 4.00±0.46° over the same 5 report seeds**
> (per-seed stable `[100.0, 98.0, 81.0, 84.0, 89.0]`).
>
> Against the correct number, these 9feat BINARY cells do **NOT** beat PID — they sit at
> or just below it. What does beat PID is the **10feat BINARY block**, all five cells,
> across both substrates (see the 18-cell table below). Fixed in
> `wnn/control/classical_baseline.py`, now the single implementation used by both
> `_pid_baseline` and `compute_baselines.py`.

### The full 18-cell picture, against PID 90.4±7.5

```
  10feat BINARY  (5 cells)  98.4 – 100.0   err 1.88–2.56   σ ≤1.6   BEATS PID
   9feat BINARY  (5 cells)  86.0 –  90.0   err 3.17–3.39            at/below PID
   QUAD         (8 cells)  26.8 –  84.0   err 4.12–7.71            well below
```

Best cell: `1layer_10feat_BINARY_s31337003` at **100.0±0.0%, err 1.88°** — it matches
LQR/MPC/LQI/MPCOF on stability (all 100.0±0.0) and approaches LQR's 1.60±0.13° error,
but does not reach it, and is far from MPCOF's 0.79±0.01°. The honest summary: **beats
PID on both stability and precision; reaches classical stability; does not reach
classical precision.** Note also that PID's own σ is 7.5 across seeds while these cells
run σ ≤1.6 — the WNN is the more consistent controller, not merely the better-scoring one.

### Why this is not leakage

The train-seed fit has **zero contact with the test draw** — it is strictly the more
conservative variant, yet it scores higher. And leakage does not simultaneously
**shrink σ from ±13.8 to ±1.7**. Variance collapsing while the mean rises is the
signature of a mismatch being removed, not of information leaking in.

### QUAD reverses, and is unexplained

`dfa_9feat_QUAD_s31337002` goes the other way (27.2 vs 36.4). No explanation offered.
Possibly the 4-state cells absorb address mismatch differently, possibly something
else. Flagged, not explained away.

## Consequences

1. **The dfa1l study table understates the substrate.** All 18 completed cells were
   scored through the per-seed refit. `1layer 9feat BINARY` reads `58.0±20.8`
   (per-seed 87/39/48); on the aligned axis its three seeds are 86.2/86.8/90.0. The
   headline "seed bimodality" that drove much of this campaign's reasoning — and the
   motivation for trying a committee — is largely this artifact.
2. **The ceiling pipeline's phase A result dissolves.** Phase A was seeded from
   `dfa_9feat_BINARY_s31337002`, which itself rescores **89.6±2.4** against phase A's
   **89.4±2.6**. The 42-generation memory-GA added nothing measurable. And "89% beats
   PID" was never true at all: against the correct comparator (90.4±7.5, not 85.0)
   phase A does **not** beat PID — see the correction above.
3. **Phase S was tuning a knob smaller than the artifact.** It chased +7–18 pp of
   suffix effect while this was worth up to +38.8 pp on the same cells.

## Fix

`--holdout-fixed-thresholds` (see `phased_ga.py`), **default off** so the in-flight
dfa1l campaign stays internally consistent. When set, the two held-out paths reuse
the train-seed thresholds for score-only genomes and leave the `evaluate_batch` path
refitting as before.

The completed table should be regenerated via `scripts/rescore_winners.py` rather
than re-run — the rescore puts every cell on the aligned axis for minutes of compute,
where re-running is hours per cell.

## Tooling gaps found

- `rescore_winners.py` has **no `--motor-fault`**, so the phase-C winner cannot be put
  on the fault-plant baselines' axis.
- Its tag regex `^(sub)_(feat)_(mode)_s(seed)$` rejects the ceiling winners' names;
  worked around with a non-destructive conforming copy.
