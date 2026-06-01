# Controller curriculum — weight-sweep results

_Generated 01/06/2026 17:49:05 EDT from `/Users/lacg/wnn/logs/controller/curriculum/ic_sweep_20260601_140443.log`._

Sweep config: Stage A only (250-step / 5° tilt / body-rate 0.5), pop=50, gens=30, patience=3, kfold-eval=5, Rust DAGGER (jerk/mono active). The auto-full winner then runs the 5-stage IC curriculum at pop=200 / 500 steps.

**search** = GA's own last-gen fitness (optimistic); **held-out** = stage-summary re-eval on a fresh draw (honest). W1 dropped 71%→54% on re-eval.


## All 18 combos (weights err²/stable/jerk/mono)

| combo | weights | search stable | search err | held-out stable | held-out err | reward | gens | total | per-gen | status |
|---|---|---|---|---|---|---|---|---|---|---|
| W1 | 0.50/0.40/0.05/0.05 | 71.0% | 4.15° | 54.0% | 4.89° | -2.08 | 30/30 | 123m | 246s | done |
| W2 | 0.40/0.50/0.05/0.05 | 75.0% | 3.92° | — | — | — | 26/30 | — | 216s | running |
| W3 | 0.60/0.30/0.05/0.05 | — | — | — | — | — | — | — | — | pending |
| W4 | 0.45/0.35/0.10/0.10 | — | — | — | — | — | — | — | — | pending |
| C1 | 0.20/0.40/0.20/0.20 | — | — | — | — | — | — | — | — | pending |
| C2 | 0.20/0.50/0.10/0.20 | — | — | — | — | — | — | — | — | pending |
| C3 | 0.20/0.50/0.20/0.10 | — | — | — | — | — | — | — | — | pending |
| C4 | 0.30/0.30/0.20/0.20 | — | — | — | — | — | — | — | — | pending |
| C5 | 0.30/0.40/0.10/0.20 | — | — | — | — | — | — | — | — | pending |
| C6 | 0.30/0.40/0.20/0.10 | — | — | — | — | — | — | — | — | pending |
| C7 | 0.30/0.50/0.10/0.10 | — | — | — | — | — | — | — | — | pending |
| C8 | 0.40/0.20/0.20/0.20 | — | — | — | — | — | — | — | — | pending |
| C9 | 0.40/0.30/0.10/0.20 | — | — | — | — | — | — | — | — | pending |
| C10 | 0.40/0.30/0.20/0.10 | — | — | — | — | — | — | — | — | pending |
| C11 | 0.40/0.40/0.10/0.10 | — | — | — | — | — | — | — | — | pending |
| C12 | 0.50/0.20/0.10/0.20 | — | — | — | — | — | — | — | — | pending |
| C13 | 0.50/0.20/0.20/0.10 | — | — | — | — | — | — | — | — | pending |
| C14 | 0.50/0.30/0.10/0.10 | — | — | — | — | — | — | — | — | pending |

## Ranking so far (completed combos — by held-out stable, then err)

| # | combo | weights | held-out stable | held-out err | reward | per-gen | total |
|---|---|---|---|---|---|---|---|
| 1 | W1 | 0.50/0.40/0.05/0.05 | 54.0% | 4.89° | -2.08 | 246s | 123m |

_1/18 combos complete._

