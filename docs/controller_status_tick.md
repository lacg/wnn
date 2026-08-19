# Controller status tick — the six-line format

The half-hourly controller tick is driven by a **session-only cron** that dies when the
CLI exits. This file is the durable copy of its *format* so it can be re-armed verbatim.
The volatile half — which chain is live, which arms have landed, the current results
table — belongs in the cron prompt and `.claude/context-snapshot.md`, not here.

## The six lines

```
[ctl DD/MM/YYYY HH:MM:SS EDT]  lever <name> · markers <n>/<total>
  run R/T  <tag>  (what this point tests, one clause)
  <STAGE> gen G/TOTAL  <X>s/gen  elapsed <T>
  elite: fit <best> · stable <S>% · err <E>° · steady <D>° · alt <A>m (best so far, during-search)
  gen:   stable <S>% · err <E>° · steady <D>° · alt <A>m (this gen's leader, during-search)
  box: <k> controller · watchdog <n> kills · avail <G> GiB · IDS <done>/<run>/<queued>
```

Rules that are not negotiable, each with the incident that produced it:

- **The STAGE gets its own line** (Luiz, 16/08/2026). Long tags and named hypotheses
  buried it mid-line when it shared line 2.
- **Name the arm's hypothesis**, not just its tag — line 2 says what the point *tests*.
- **`alt` prints even at 0.00** on the fitness label. A suppressed zero makes a
  deliberate control indistinguishable from a run whose weight never arrived, which is
  exactly the 18/08 plumbing bug this line failed to expose.
- **Never invent a missing value, and never print `0.000` for "not measured"** — a zero
  in the altitude column reads as a vehicle holding altitude perfectly, the opposite of
  the truth. Use an em dash.

## `elite:` and `gen:` are two different genomes

Since `b872ba57` the `.out` gen line carries both blocks, split by a `|`:

```
best=1.8737 (=), stable=55.00%, err=13.89°, steady=17.17°, alt=1.230m
  | gen: stable=0.00%, err=26.38°, steady=30.48°, alt=0.407m
```

- `elite:` — the **incumbent**, what this stage has banked. Read the fields *before* the
  `|`. Carries a fitness value.
- `gen:` — **this generation's leader**. Read the fields *after* the `|`. Four fields,
  never five: the gen leader has no fitness of its own.

They disagree on any `(=)` generation because they are different genomes. Metrics are
frozen per genome — elites carry `pop_metrics` forward and are never re-scored — so a
disagreement between the blocks *is* the evidence of two genomes, not one genome
re-measured. Before `b872ba57`, `steady`/`alt` were read from the population while their
neighbours came from the incumbent, so one line silently mixed both. **Do not re-merge
them.**

Why both are worth the width: `(=)` does *not* mean the search is idle. The fitness is a
rank-WHM computed over the *current* pool, so the cross-generation comparison is not on a
fixed scale — `(=)` means "this generation's leader did not out-rank a number computed
over a pool that no longer exists". Watching `gen:` regress is how population collapse
becomes visible before `elite:` ever moves.

**Neither is what gets published.** Stage-select ranks the union of the top-3 of *every*
stage on the val seeds, so the headline is drawn from nine candidates and is frequently
neither — arm 2 of the alt-weight sweep headlined `CONNECTIONS#2`, arm 5 `CONNECTIONS#1`.

## When a value legitimately does not exist

- No `| gen:` block on the line (any run started before `b872ba57`) → `gen: — (pre-b872ba57 run)`.
- Before the first GA gen line: read `fit`/`stable`/`err`/`steady` off the `GRID WINNER`
  line and tag `elite:` as `(grid winner, during-search)`. That line prints the fitness
  *function* but no *value*, so `fit —`; and there is no gen leader yet → `gen: —`.
- `GRID WINNER` lines from runs before `a71bf340` have no `alt=` → `alt —`.

## Everything on these lines is during-search

`(during-search)` is a provenance tag, not decoration. These metrics were measured on the
same episode folds the GA trained and selected on, so they are optimistic by construction.
Measured on arm 1 of the alt-weight sweep:

| | stable | err | steady |
|---|---|---|---|
| during-search (CONNECTIONS, gen 5/5) | 50.0% | 18.43° | 23.06° |
| held-out (5 report seeds) | 36.4 ±7.2% | 20.93 ±6.83° | 24.14 ±8.30° |

The search view claimed 50% stability; the honest number was 36.4%. **Never put a
gen-line number in a results table** — markers, arm tables and `docs/` take held-out
values only.
