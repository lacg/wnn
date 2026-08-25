# Controller curriculum — weight-sweep results

_Generated 04/06/2026 09:52:54 EDT from `/Users/lacg/wnn/logs/controller/curriculum/ic_sweep_20260601_140443.log`._

Sweep config: Stage A only (250-step / 5° tilt / body-rate 0.5), pop=50, gens=30, patience=3, kfold-eval=5, Rust DAGGER (jerk/mono active). The auto-full winner then runs the 5-stage IC curriculum at pop=200 / 500 steps.

**search** = GA's own last-gen fitness (optimistic); **held-out** = stage-summary re-eval on a fresh draw (honest). W1 dropped 71%→54% on re-eval.


## All 18 combos (weights err²/stable/jerk/mono)

| combo | weights | search stable | search err | held-out stable | held-out err | reward | gens | total | per-gen | status |
|---|---|---|---|---|---|---|---|---|---|---|
| W1 | 0.50/0.40/0.05/0.05 | 71.0% | 4.15° | 54.0% | 4.89° | -2.08 | 30/30 | 123m | 246s | done |
| W2 | 0.40/0.50/0.05/0.05 | 75.0% | 3.92° | 65.0% | 4.40° | -1.75 | 30/30 | 114m | 227s | done |
| W3 | 0.60/0.30/0.05/0.05 | 71.0% | 4.15° | 60.0% | 4.67° | -1.85 | 30/30 | 128m | 255s | done |
| W4 | 0.45/0.35/0.10/0.10 | 75.0% | 3.93° | 63.0% | 4.44° | -1.72 | 30/30 | 104m | 207s | done |
| C1 | 0.20/0.40/0.20/0.20 | 77.0% | 3.99° | 55.0% | 4.58° | -1.82 | 30/30 | 98m | 197s | done |
| C2 | 0.20/0.50/0.10/0.20 | 75.0% | 3.97° | 63.0% | 4.20° | -1.58 | 30/30 | 112m | 223s | done |
| C3 | 0.20/0.50/0.20/0.10 | 80.0% | 3.84° | 47.0% | 5.22° | -2.42 | 30/30 | 118m | 237s | done |
| C4 | 0.30/0.30/0.20/0.20 | 74.0% | 3.90° | 59.0% | 4.54° | -1.74 | 30/30 | 107m | 214s | done |
| C5 | 0.30/0.40/0.10/0.20 | 75.0% | 3.96° | 58.0% | 4.50° | -1.75 | 30/30 | 105m | 211s | done |
| C6 | 0.30/0.40/0.20/0.10 | 76.0% | 3.80° | 40.0% | 5.45° | -2.78 | 30/30 | 93m | 185s | done |
| C7 | 0.30/0.50/0.10/0.10 | 79.0% | 3.88° | 47.0% | 5.16° | -2.29 | 30/30 | 136m | 271s | done |
| C8 | 0.40/0.20/0.20/0.20 | 71.0% | 4.15° | 54.0% | 4.90° | -2.06 | 30/30 | 114m | 229s | done |
| C9 | 0.40/0.30/0.10/0.20 | 79.0% | 3.74° | 62.0% | 4.48° | -1.77 | 30/30 | 105m | 209s | done |
| C10 | 0.40/0.30/0.20/0.10 | 74.0% | 4.06° | 50.0% | 4.96° | -2.10 | 30/30 | 118m | 236s | done |
| C11 | 0.40/0.40/0.10/0.10 | 71.0% | 4.15° | 63.0% | 4.34° | -1.67 | 30/30 | 124m | 248s | done |
| C12 | 0.50/0.20/0.10/0.20 | 79.0% | 3.84° | 58.0% | 4.59° | -1.83 | 30/30 | 91m | 181s | done |
| C13 | 0.50/0.20/0.20/0.10 | 74.0% | 4.19° | 71.0% | 4.19° | -1.50 | 30/30 | 120m | 240s | done |
| C14 | 0.50/0.30/0.10/0.10 | 81.0% | 3.93° | 61.0% | 4.58° | -1.86 | 30/30 | 140m | 281s | done |

## Ranking so far (completed combos — by held-out stable, then err)

| # | combo | weights | held-out stable | held-out err | reward | per-gen | total |
|---|---|---|---|---|---|---|---|
| 1 | C13 | 0.50/0.20/0.20/0.10 | 71.0% | 4.19° | -1.50 | 240s | 120m |
| 2 | W2 | 0.40/0.50/0.05/0.05 | 65.0% | 4.40° | -1.75 | 227s | 114m |
| 3 | C2 | 0.20/0.50/0.10/0.20 | 63.0% | 4.20° | -1.58 | 223s | 112m |
| 4 | C11 | 0.40/0.40/0.10/0.10 | 63.0% | 4.34° | -1.67 | 248s | 124m |
| 5 | W4 | 0.45/0.35/0.10/0.10 | 63.0% | 4.44° | -1.72 | 207s | 104m |
| 6 | C9 | 0.40/0.30/0.10/0.20 | 62.0% | 4.48° | -1.77 | 209s | 105m |
| 7 | C14 | 0.50/0.30/0.10/0.10 | 61.0% | 4.58° | -1.86 | 281s | 140m |
| 8 | W3 | 0.60/0.30/0.05/0.05 | 60.0% | 4.67° | -1.85 | 255s | 128m |
| 9 | C4 | 0.30/0.30/0.20/0.20 | 59.0% | 4.54° | -1.74 | 214s | 107m |
| 10 | C5 | 0.30/0.40/0.10/0.20 | 58.0% | 4.50° | -1.75 | 211s | 105m |
| 11 | C12 | 0.50/0.20/0.10/0.20 | 58.0% | 4.59° | -1.83 | 181s | 91m |
| 12 | C1 | 0.20/0.40/0.20/0.20 | 55.0% | 4.58° | -1.82 | 197s | 98m |
| 13 | W1 | 0.50/0.40/0.05/0.05 | 54.0% | 4.89° | -2.08 | 246s | 123m |
| 14 | C8 | 0.40/0.20/0.20/0.20 | 54.0% | 4.90° | -2.06 | 229s | 114m |
| 15 | C10 | 0.40/0.30/0.20/0.10 | 50.0% | 4.96° | -2.10 | 236s | 118m |
| 16 | C7 | 0.30/0.50/0.10/0.10 | 47.0% | 5.16° | -2.29 | 271s | 136m |
| 17 | C3 | 0.20/0.50/0.20/0.10 | 47.0% | 5.22° | -2.42 | 237s | 118m |
| 18 | C6 | 0.30/0.40/0.20/0.10 | 40.0% | 5.45° | -2.78 | 185s | 93m |

## Rounds 2-3 confirmation set (stable=0.50 family ∪ top-8 held-out)

- stable=0.50 family (fixed): **W2, C2, C3, C7**
- top-8 by held-out: C13, W2, C2, C11, W4, C9, C14, W3
- **`--combos W2,C2,C3,C7,C13,C11,W4,C9,C14,W3`** → 10 combos × 2 fresh seeds (rounds 2-3)

## Multi-seed rounds — held-out stable % across 3 seeds

Round 1 = base seed 42; rounds 2-3 = fresh seeds (confirmation set only). Mean±std over completed rounds. Watch for combos that crash at a fresh seed (overfit) vs hold steady (robust).

| combo | weights | R1 | R2 | R3 | mean±std | rounds |
|---|---|---|---|---|---|---|
| C13 | 0.50/0.20/0.20/0.10 | 71% | 69% | 37% | 59.0±15.6 | 3 |
| C9 | 0.40/0.30/0.10/0.20 | 62% | 63% | 52% | 59.0±5.0 | 3 |
| C4 | 0.30/0.30/0.20/0.20 | 59% | · | · | 59.0 | 1 |
| W4 | 0.45/0.35/0.10/0.10 | 63% | 62% | 50% | 58.3±5.9 | 3 |
| C5 | 0.30/0.40/0.10/0.20 | 58% | · | · | 58.0 | 1 |
| C12 | 0.50/0.20/0.10/0.20 | 58% | · | · | 58.0 | 1 |
| C1 | 0.20/0.40/0.20/0.20 | 55% | · | · | 55.0 | 1 |
| W3 | 0.60/0.30/0.05/0.05 | 60% | 49% | · | 54.5±5.5 | 2 |
| C2 | 0.20/0.50/0.10/0.20 | 63% | 59% | 40% | 54.0±10.0 | 3 |
| W1 | 0.50/0.40/0.05/0.05 | 54% | · | · | 54.0 | 1 |
| C8 | 0.40/0.20/0.20/0.20 | 54% | · | · | 54.0 | 1 |
| C11 | 0.40/0.40/0.10/0.10 | 63% | 61% | 36% | 53.3±12.3 | 3 |
| W2 | 0.40/0.50/0.05/0.05 | 65% | 61% | 27% | 51.0±17.0 | 3 |
| C14 | 0.50/0.30/0.10/0.10 | 61% | 59% | 30% | 50.0±14.2 | 3 |
| C10 | 0.40/0.30/0.20/0.10 | 50% | · | · | 50.0 | 1 |
| C3 | 0.20/0.50/0.20/0.10 | 47% | 47% | · | 47.0±0.0 | 2 |
| C7 | 0.30/0.50/0.10/0.10 | 47% | 46% | · | 46.5±0.5 | 2 |
| C6 | 0.30/0.40/0.20/0.10 | 40% | · | · | 40.0 | 1 |

_18/18 combos complete (round 1)._


---

# Gated weight sweep — 25/08/2026 (SUPERSEDES the June sweep above for arm selection)

_Generated from `experiments/gatedwsweep_markers/` (29 markers, all rc=0). Chain:
`scripts/gated_weight_sweep_chain.sh`; log `/private/tmp/gated_wsweep.log`; outdir
`logs/controller/gated_wsweep`._

The June sweep above ranked 18 combos in ONE population under a harmonic combine with no
viability gate. This one re-asks the question under the current regime and, critically, tests
whether the **jerk and mono terms earn their weight at all**.

**Config, identical for all six arms:** `--fit-aggregation zscore --zrank-clamp 3.0`,
viability gate `--gate-stable 0.70 --gate-err 8.0`, λ_alt = 0, alt RANK weight 0.0,
128 output neurons, grid-bits 24+30, levels 16, 18 features (`FEAT_STAGE1`, one shared
variable — no arm can differ), mpcof teacher, cf21_brushless, L4C, `--translation`,
5 eval folds, pop 50. Seed-major interleave over seeds 31337002…31337006.

**Headline = val-selected held-out**, reported as `stable% / err° / steady°`. Stage-select
ranks the union of the top-3 of every stage on 5 val seeds and never touches the 5 report
seeds (99990101…05). The `noJM` arms are their parent's vector **renormalized** after dropping
jerk and mono — not zero-padded, so copying the parent's raw numbers gives a different arm.

## All six arms × five seeds
| arm | weights | s…2 | s…3 | s…4 | s…5 | s…6 | mean (n) |
|---|---|---|---|---|---|---|---|
| **C10** | err² .40 / stable .30 / jerk .20 / mono .10 | 89.4 / 2.95 / 2.71 | 93.8 / 2.57 / 2.43 | 93.0 / 2.36 / 1.99 | 94.4 / 2.31 / 1.68 | 89.0 / 3.49 / 3.65 | **91.92 / 2.74 / 2.49** (n=5) |
| **C10noJM** | err² .57 / stable .43 | 97.0 / 2.22 / 2.06 | 94.2 / 2.04 / 1.63 | 88.6 / 2.91 / 2.83 | 97.4 / 1.80 / 1.25 | 85.4 / 2.92 / 2.75 | **92.52 / 2.38 / 2.10** (n=5) |
| **S16** | err² .25 / steady .35 / stable .20 / jerk .15 / mono .05 | 91.0 / 2.65 / 2.68 | 92.8 / 2.32 / 1.68 | 90.6 / 2.64 / 2.11 | 93.2 / 2.36 / 1.71 | 93.0 / 2.40 / 2.16 | **92.12 / 2.47 / 2.07** (n=5) |
| **S16noJM** | err² .3125 / stable .25 / steady .4375 | 93.8 / 2.12 / 1.72 | 95.8 / 2.10 / 1.76 | 90.8 / 2.69 / 2.16 | 98.0 / 2.11 / 1.59 | 92.2 / 2.72 / 2.84 | **94.12 / 2.35 / 2.01** (n=5) |
| **E50S50** | err² .50 / stable .50 | 91.8 / 2.68 / 2.08 | 92.2 / 2.76 / 2.62 | 86.8 / 3.32 / 2.88 | 97.2 / 1.99 / 1.68 | — not flown | **92.00 / 2.69 / 2.31** (n=4) |
| **STEADY40** | err² .30 / stable .30 / steady .40 | 94.2 / 2.06 / 1.55 | 91.8 / 2.66 / 2.56 | 93.2 / 2.65 / 2.19 | 93.4 / 2.25 / 1.73 | 86.2 / 3.34 / 2.69 | **91.76 / 2.59 / 2.14** (n=5) |

| comparison | stable | err | verdict |
|---|---|---|---|
| C10noJM vs C10 | 3-2 | 4-1 | **C10noJM** (pre-registered pair) |
| S16noJM vs S16 | 4-1 | 3-2 | **S16noJM** (pre-registered pair) |
| S16noJM vs C10noJM | 4-1 | 3-2 | **S16noJM** (runoff — NOT pre-registered) |
| S16noJM vs STEADY40 | 3-2 | 3-2 | **S16noJM** (runoff — NOT pre-registered) |

Headline stage chosen per run (val-selected, never on the report seeds):

| arm | s…2 | s…3 | s…4 | s…5 | s…6 |
|---|---|---|---|---|---|
| C10 | MEMORY#2 | MEMORY#2 | MEMORY#0 | MEMORY#2 | MEMORY#0 |
| C10noJM | NEURONS#0 | MEMORY#1 | NEURONS#1 | MEMORY#0 | NEURONS#2 |
| S16 | NEURONS#0 | NEURONS#2 | NEURONS#2 | NEURONS#2 | NEURONS#0 |
| S16noJM | MEMORY#2 | NEURONS#1 | NEURONS#1 | MEMORY#0 | NEURONS#1 |
| E50S50 | MEMORY#1 | NEURONS#1 | NEURONS#0 | MEMORY#0 | — |
| STEADY40 | NEURONS#0 | NEURONS#0 | MEMORY#2 | MEMORY#1 | MEMORY#0 |

_29 runs, all rc=0, 98 h of box time (3.37 h/run)._

## Rulings

**Both pre-registered pairs went to the noJM variant.** Deleting the jerk and mono terms
improves *each* parent — the direct confirmation of the saturation finding: .20 jerk + .10
mono outvote .40 err + .30 stable once attitude saturates. The noJM arms also post **lower**
jerk than their jerk-weighted parents at most seeds, so weighting jerk does not buy jerk. If
mono ever matters it belongs in the viability gate as a CONSTRAINT, not as a scored term.

**S16noJM is the winner** and is now `LADDER_WEIGHTS` in `scripts/sweep_ladder_chain.sh`. It
takes both primaries against C10noJM (4-1, 3-2) and against STEADY40 (3-2, 3-2), and it wins
mean steady (2.01°), which is the sweep ladder's own ranking criterion — fixed 17/08, before
this comparison existed, so not circular.

## What these numbers do NOT support

- **This is a decision rule, not a significance claim.** Paired diffs between the two noJM
  arms are ~0.2 pp stable / ~0.09° err against per-seed noise of ~3.0 pp / ~0.14°; resolving
  them statistically needs on the order of **1700 seeds (stable) / 21 (err)**. What may be
  said is "S16noJM won the registered 5-seed paired majority", not "S16noJM is better".
- **The two runoffs were not pre-registered.** They became necessary only because both pairs
  resolved, which is a forking path. Each was read ONCE at a fixed n=5. A look-then-extend
  rule ("3-of-5, else 4-of-7, else 6-of-10") reaches a majority with probability 1 even
  between identical arms.
- **STEADY40's tiebreak was adjudicated by a rule fixed before it flew:** descriptive only,
  overriding S16noJM solely on both primaries 3-2 AND matched-seed means. It cleared neither.
- **E50S50 is n=4, every other arm is n=5.** Its fifth seed was cancelled once the comparison
  was arithmetically sealed (0-4 stable / 1-3 err at n=4 — one seed cannot reach a majority).
  Never compare its 4-seed mean against a 5-seed mean; the unequal n moved STEADY40's mean
  steady by 0.13° in an earlier reading and briefly made a clear loss look like a tie.
- **No PID win.** A run's `.out` holds ~60 `vs PID[est] (RIVAL)` lines spanning 1.24°–3.86°,
  one per seed block. Compare the headline's report seed against *that same seed's* PID row.
  The best run of the programme (C10noJM s31337005, 97.4% / 1.80° / 1.25°) loses every column
  to PID on its matched seed (100% / 1.24° / 0.55°).
