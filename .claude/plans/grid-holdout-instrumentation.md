# Stage instrumentation + three-way stage selection — ready to land

**Status:** WRITTEN, NOT APPLIED. Land only when the L3 chain is idle (4/4 markers).
**Author:** Andrew Martin, 07/08/2026.

Four changes, all additive, landing together:

| # | change | unlocks |
|---|---|---|
| 1 | GRID gets the 5-report-seed held-out | "does the GA earn its 7000s over a 1500s grid?" |
| 2 | persist EVERY stage winner per run | NEURONS-vs-MEMORY on identical draws; committee members |
| 3 | three-way max-over-stages, **selected on val** | ship the best stage without inflating the reported number |
| 4 | marker carries the GRID block + the selected stage | the results doc can read all of it |

## Why

`docs/l4_teacher_screen_results.md` compares stages that are not measured the same way.
Every GA stage (NEURONS, MEMORY) gets a 5-report-seed held-out block; **Stage 0 GRID gets
none**. Verified on `logs/controller/l3delta/L3D_dstep_..._s31337002.out`: the only
`RESULT — during-search winner (held-out)` blocks are at lines 196-232 (NEURONS) and
351-387 (MEMORY). Line 72 is the grid winner, during-search numbers only.

So the natural question — *does the GA earn its ~7000s over a ~1500s grid?* — is
unanswerable today. Eyeballing the two numbers points opposite ways depending on the run:

```
dstep s31337002:  grid steady 1.06 (during-search) → NEURONS 0.60 (held-out)
dleak s31337002:  grid steady 0.76 (during-search) → NEURONS 1.40 (held-out)
```

These are different measurements, so neither direction means anything. The same log shows
`Neurons gen 5/5 steady=5.84°` during-search against `0.60°` held-out on the *same*
winner — a 10× instrumentation gap.

Secondary payoff: it also gives the honest three-way GRID / NEURONS / MEMORY table needed
to decide the open "should MEMORY be the reported stage?" question (MEMORY is worse than
NEURONS in 6/8 paired runs, mean Δsteady +0.21°, but sign test p≈0.29 — not significant).

## Cost

~30s on a ~8465s run (**+0.35%**). Arithmetic from the dstep run:
`8465 total − 1577 grid − 6662 neurons − 146 memory = 80s` covers BOTH existing held-out
blocks *plus* the PID baseline, save-winner and FPGA count. A third block is ~30s.

## Change 1 — `src/wnn/control/phased_ga.py` (~line 1439)

Reuses `_maybe_holdout` verbatim; no new measurement code. `stage0_grid` already returns
`outcome.seed_population` whose `[0]` is documented (line 445-446) as the fitness-best
genome, which is exactly the `res.best_genome` / `res.final_population` shape
`_maybe_holdout` expects.

```python
	if resume_state is None:
		winner_spec, seed_pop0, m0, dt0, _thr = stage0_grid(args, ec, seed)
		stage_results = [("Grid", winner_spec, m0, dt0, grid_point_count(args))]
		# GRID held-out (REPORT ONLY, never feeds selection — same contract as every
		# other stage). The grid winner is what you would ship if you stopped before
		# the GA, so it needs the SAME 5-report-seed measurement the GA stages get.
		# Without it the Grid row carries during-search numbers while every later row
		# carries held-out ones, and "does the GA earn its keep?" cannot be asked.
		from types import SimpleNamespace
		grid_res = SimpleNamespace(best_genome=(seed_pop0[0] if seed_pop0 else None),
		                           final_population=seed_pop0)
		grid_ho = _maybe_holdout(args, ec, winner_spec, grid_res, seeds, "GRID")
		if stage_holdouts is not None and grid_ho is not None:
			stage_holdouts["GRID"] = grid_ho
```

Notes:
- `_maybe_holdout` already returns `None` when `best_genome is None`, so an empty
  `seed_pop0` is handled — no extra guard needed.
- `SimpleNamespace` over a new class: this is a 2-line adapter, and `_maybe_holdout`
  itself already builds its return value the same way (line 1231). A dedicated class file
  would be ceremony for an adapter with no behaviour.
- `seeds` is in scope at this point (used at line 1456 to build the orchestrator).
- Resume path (`resume_state is not None`) deliberately untouched — grid is skipped there.

## Change 2 — `scripts/controller_arm_lib.sh` (~line 107)

```bash
	held_gm=$(grep -E "GRID MULTI-SEED held-out" "$out" | tail -1)
```

and add one field to the marker `printf` (append after `held_memory_multiseed`; the
line-119 comment confirms field order only ever mattered for byte-parity with pre-migration
markers, and readers go through `json.load`):

```
"held_grid_multiseed":"%s",
```
```bash
		"$(echo "$held_gm" | tr -d '"' | sed 's/  */ /g')" \
```

**Do NOT extend the R3 guard** (line 114) to require a GRID block — resume runs legitimately
have no grid stage, and a missing GRID line must not suppress an otherwise-valid marker.

## Change 1b — persist every stage winner (`scripts/controller_arm_lib.sh`)

**The gap:** no recipe passes `--save-stage-checkpoints`, so stage checkpoints fall to
`_stage_emergency_path` (`phased_ga.py:96-99`) which builds ONE fixed filename with no
per-run tag:

```
/tmp/wnn-phased-ga-emergency/emergency_stage1_neurons.yaml.gz
```

Every run overwrites it. Only the MEMORY winner survives per run (`--save-winner`, `[-1]`).
That is why the NEURONS winners of L3 runs 1-2 are unrecoverable, and it is the SAME gap
that blocks the cross-teacher committee (see `docs/teacher_committee_plan.md`): a committee
wants GRID/NEURONS/MEMORY winners from each teacher, and today 2 of 3 are discarded.

**Fix:** add to the arm lib's phased_ga invocation:

```bash
--save-stage-checkpoints "logs/controller/<lever>/${tag}_stages"
```

One flag, no code change. Costs ~26 MB × 3 stages per run — trim old lever dirs when a
programme closes.

**Preserved by hand, 07/08:** `experiments/stage_winners_preserved/run3_dstep_s31337003_stage1_neurons.yaml.gz`
— L3 run 3's NEURONS winner, copied out of the emergency dir before run 4 overwrote it.
With run 3's MEMORY winner this gives ONE run where the identical-draw comparison is
possible without re-flying.

## Change 3 — three-way stage selection on val

Ship the best of {GRID, NEURONS, MEMORY} rather than always MEMORY (`[-1]`). Two facts
force the design:

- Across 8 paired runs MEMORY beats NEURONS in only 2/8 (mean Δsteady +0.21°, sign test
  p≈0.29 — a lean, not a result). Always-MEMORY is not defensible; neither is always-NEURONS.
- Elitism does NOT make the stages self-correcting. `generic_ga.py:837-840` skips
  re-evaluation of anything already holding `Metrics`, and `Re-eval streaming` fires
  exactly once per stage (empirically: log lines 81 and 245 only). So the incoming winner
  gets ONE noisy re-score on the next stage's stick and can be eliminated permanently by
  that single draw. A stage winner is a max over never-refreshed noisy estimates.

**Selector: `seeds.val`, NOT the report seeds.** Selecting on the report seeds is not a
training leak — nothing is fit to them — but `E[max(A,B,C)] > max(E[A],E[B],E[C])`, so the
published number would be biased upward by the selection itself. That is the best-of-N
inflation family, and it is the first thing a reviewer pokes at. `seeds.val` is already in
scope (`phased_ga.py:1484`, used by `_pid_baseline`). Cost: 3 extra evaluations per run.

```
search folds  → train
seeds.val     → picks GRID | NEURONS | MEMORY
report seeds  → the published number, unbiased
```

⚠️ **Report it as an ADDITIONAL column; do NOT switch the headline mid-programme.**
L1/L1b/L2/L3 all report the MEMORY stage. Changing the headline before L4 would break the
one comparison chain the hold-floor programme rests on. So: emit the val-selected stage and
its report-seed triple alongside the existing MEMORY headline, accumulate evidence across
L4, and switch the headline only at a programme boundary — pre-registered, justified by the
winner's-curse mechanism above rather than by which scored better.

## Backward-compat checks (done)

- **Marker awk anchors are safe.** `held_n` / `held_m` anchor on `/STAGE 1 (NEURONS) done/`
  and `/STAGE 4 (MEMORY) done/` before matching `RESULT —`. GRID's new RESULT lines are
  emitted *before* the STAGE 1 header, so they fall outside both scan windows. This is the
  exact bug class the line-95-99 comment documents; the anchoring already immunises it.
- **`flow_runner.py` is unaffected.** It reads `stage_holdouts.get(_CONTROLLER_STAGES[idx].upper())`
  over `["Grid", "Neurons", "Bits", "Connections", "Memory"]` — a `"GRID"` key now
  resolves where it previously returned `None`, which is the desired behaviour and not a
  break.
- **Old markers** simply lack `held_grid_multiseed`; readers use `.get()`.
  `scripts/backfill_marker_stage_fields.py` can gain a `multiseed(lines, "GRID")` line, but
  it will find nothing in pre-change logs — optional, low value.

## MUST VERIFY before trusting the first result

**Does adding the GRID held-out perturb the downstream search?** `_holdout_report` runs
rollouts between the grid and Stage 1. If it touches any global/numpy/Metal RNG state that
Stage 1 later draws from, run-to-run comparability with L1/L1b/L2/L3 breaks and every
cross-lever comparison in `docs/l4_teacher_screen_results.md` is invalidated.

Test: re-run one already-flown control (`s31337002`, no other flag changes) with the patch
and confirm the NEURONS and MEMORY multi-seed blocks come back **bit-identical** to the
banked marker. If they differ, the held-out must be moved behind an explicit RNG
save/restore before the change can land.

This is the gate. Do not fold any GRID number into the results doc until it passes.

## Landing order

1. Wait for L3 4/4 markers (runs 3+4 must stay comparable to 1+2 — do not land mid-chain).
2. Apply Change 1 + Change 2.
3. Run the RNG-parity control above.
4. If parity holds → commit + push; the next lever (L4) flies with three-way instrumentation.
5. If parity fails → fix RNG isolation first; do not report GRID numbers.

## What this does NOT decide

Switching the reported stage from MEMORY to NEURONS on the strength of the 6/8 lean would
be post-hoc stage selection. If we change the headline stage it must be **pre-registered
before the next lever flies**, justified by mechanism (MEMORY early-stops at gen 6/120, so
it is not meaningfully searching) rather than by score.
