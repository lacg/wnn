# Scope C stage 2 — the λ_pos sweep (plan, 14/08/2026)

Companion to the stage-1 λ_alt sweep (`scripts/stage1_lambda_alt_sweep.sh`) and to
chunk B's teacher bar (`scope_c_stage2_chunk_b_teacher.md`). Nothing here is armed
yet; this file exists so the ladder is a DERIVATION with a pre-registered read,
not a guess made on the night it launches.

## Why the ladder must not be chosen independently of λ_alt

λ_alt and λ_pos are the SAME KIND of number. Both convert a squared distance in
**metres²** into the same reward currency the attitude term supplies in
**radians²**. So whatever conversion the λ_alt sweep lands on is direct evidence
for λ_pos — this is not two independent tuning problems, and treating it as two
would throw away the 10 runs stage 1 is paying for.

The magnitudes that set the scale:

| quantity | target | squared |
|---|---|---|
| attitude error | ~2° = 0.0349 rad | 0.0012 rad² |
| altitude error | ~0.15 m | 0.0225 m² |
| position error (Molchanov et al. 2019) | 0.11–0.24 m | 0.012–0.058 m² |

At a 0.15 m target, λ = 0.053 makes the position term EQUAL the attitude term;
λ = 0.53 makes it 10×; λ = 5.3 makes it 100×. Note where that puts the stage-1
result: λ_alt = 16 (the current leader at n=1) is a ~300× ratio. If that
replicates on seed 31337003, then the honest reading is that this controller
WANTS the translation term to dominate, and a sub-1 λ_pos ladder would be
centred in the wrong place entirely.

**Decision rule for the ladder — resolve AFTER the λ_alt re-fly lands 10 markers:**

- λ_alt winner replicates at 16 (or higher) → centre λ_pos on it:
  `{0, 4, 16, 64, 256}`, same geometric spacing stage 1 used.
- λ_alt winner replicates LOW (0–4), i.e. λ=16 was seed luck → the units argument
  wins and the ladder is `{0, 0.05, 0.5, 5, 50}`, spanning "position term equals
  attitude" through "position dominates 100×".
- λ_alt SPLITS across seeds (no replication) → do not pick a λ_pos centre from a
  coin flip. Fly `{0, 0.5, 4, 32}` — deliberately wide, 4 arms not 5 — and treat
  round 1 as bracketing rather than selection.

One adjustment regardless of branch: horizontal starts are displaced ~1 m against
altitude's ±0.3 m, so the same λ buys a ~10× larger term early in the episode.
That argues for reading the TRANSIENT and the settled error separately (below)
rather than shifting the ladder — shifting it would confound "what weight" with
"which part of the episode".

## What is measured

Per λ_pos, per seed, the held-out row now carries all four numbers, because
chunk D (`8c05a64d`) made the rival scorer fly the same displaced episodes:

- the attitude **triple** — stable% / err° / steady°,
- **mean_altitude_error_m** — must not regress vs the stage-1 winner,
- **mean_position_error_m** — the Euclidean 3-D error, Molchanov-comparable,
- the same four for all five classical rivals on the SAME episode set.

## Pre-registered read

1. Rank by held-out **position error**. This is the axis the sweep is buying.
2. Require the attitude triple within ~**1 SD** of the λ_pos = 0 control — the
   whole programme is still measured on attitude, and a controller that buys
   position by thrashing attitude has not solved the problem.
3. Require **altitude** not degraded beyond ~1 SD of the stage-1 λ_alt winner.
   Stage 2 adds an axis; it must not silently spend the one stage 1 bought.
4. Report the WHOLE table — one row per λ per seed (Rule 5/7), never just the
   winner, and never a mean across seeds that hides a split.

λ_pos = 0 is a real arm, not filler: with the horizontal features on, displaced
starts, and NO position reward, it measures what the horizontal channel costs
before it is asked to pay for itself — the stage-2 twin of the scope-cost arm.

## Budget and gating

~5 λ × 2 seeds = 10 runs. Stage 1 ran ~3h30m/run at 18 features; stage 2 carries
22 (15 pidmix + 3 vertical + 4 horizontal), so budget ~4h/run ⇒ **~40 h**, one
controller at a time, interleaved (round 1 = one run of every λ, so an early cull
is possible and a crash at hour 3 still leaves a complete round-1 comparison).

Gates, in order — do NOT arm until all are green:

1. λ_alt re-fly complete: 10 `stage1lambda` markers, ladder branch chosen above.
2. Chunk D wheel INSTALLED from `dist_staged/` and smoked (one pop-6), and the
   `_unpack` transitional shim in `classical_baseline.py` DELETED once no flying
   process predates it.
3. The three already-queued chains (calibab → scopecost → bitsaxis) drained, or
   an explicit decision to jump the queue.
4. A stage-2 pop-6 smoke with `--obs-pos-err-xy --obs-vel-xy --xy-offset 1.0`
   showing **0** `FELL BACK` lines (the 14/08 batch-trainer trap).

## Not in scope for this sweep

`--xy-offset` itself (the plant's difficulty knob) is NOT swept here — one axis
at a time. Fix it at 1.0 m, matching chunk B's ~0.97 m displaced starts, so the
teacher bar and the WNN's task are the same task.
