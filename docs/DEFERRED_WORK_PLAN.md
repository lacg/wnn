# Deferred-Work Plan — options for approval

Companion to `docs/ARCHITECTURE_REVIEW_2026-06.md`. These are the items the
10/06/2026 cleanup deliberately deferred (too risky to restructure
unsupervised). Each item below has options with trade-offs and a
recommendation. **To approve: tick one option per item** (or write an
alternative next to it) — implementation then proceeds item by item, one
commit each, tests green between steps.

**Suggested order if all recommendations are approved: D1 → D5 → D2 → D4 → D3**
(bug-risk reduction first, mechanical churn last).

---

## D1. PhasedSearchOrchestrator extraction + checkpoint schema unification

**Context.** `control/phased_ga.py` (~960 ln) and `experiments/phased_search.py`
(~1770 ln) duplicate ~500 lines of stage orchestration. The 10/06 batch already
ported the two *behavioral* divergences that mattered (population-cascade guard,
explicit resume contract). What remains is structural: two checkpoint formats
(controller = pickle with `generation`+`patience`; experiments = json.gz
without them) and no shared orchestrator. Production IDS resume is unaffected
(it uses `CheckpointManager`, which already does true gen+patience resume).

- [ ] **Option A — full extraction.** New `PhasedSearchOrchestrator` generic base
  (phase loop, carry, checkpoint, resume); both strands become thin subclasses;
  one checkpoint schema (json.gz with `generation`/`patience`/`population`,
  versioned `schema: 2`; loader accepts both old formats for one release).
  *Effort: 6-10h. Risk: HIGH — touches both production pipelines; needs one
  supervised end-to-end run per strand before trusting.*
- [ ] **Option B — schema-only unification (recommended).** Keep the two
  orchestrators; give `phased_search` checkpoints the controller's missing
  fields (`generation`, `patience`, format-versioned), port the SIGTERM
  emergency-dump pattern to `phased_search`, and add a 2-way checkpoint
  converter script. Extraction itself waits until the next time either file
  needs real feature work ("boy-scout rule" instead of big-bang).
  *Effort: 3-4h. Risk: MEDIUM-LOW — additive fields, loader stays
  backward-compatible.*
- [ ] **Option C — leave as-is.** The behavioral gaps are already fixed; accept
  the structural duplication. *Effort: 0. Risk: divergence creeps back over time.*

**Recommendation: B.** A's payoff is mostly aesthetic now that behavior is
aligned; B captures the remaining practical value (true resume + crash dumps
for LM phased runs) at a fraction of the risk.

---

## D2. Per-call globals folding (`set_empty_value` / `MEMORY_MODE` / `NORMAL_CLASS` / `FITNESS_WEIGHTS`)

**Context.** These are process-global atomics mutated per evaluation call
(`lib.rs` wrappers call `set_empty_value(...)` each time). Two evaluations with
different settings cannot safely overlap. Today the worker serializes flows per
process, so this is a *latent* hazard, not an active bug — but it blocks any
future in-process parallel evaluation, and it is invisible state.

- [ ] **Option A — full fold.** Introduce `EvalSettings { empty_value, memory_mode,
  normal_class, fitness_weights }` threaded as a parameter through every
  train/eval function (~40-60 signatures across adaptive/multistage/ids_cache/
  bitwise). Globals deleted. *Effort: 8-12h mechanical + parity re-verification.
  Risk: MEDIUM — huge diff, but compiler-driven; parity tests catch semantic
  drift.*
- [ ] **Option B — guard-rail only (recommended).** Keep the globals but make the
  hazard impossible to hit silently: a process-wide `EVAL_IN_PROGRESS` atomic
  guard — if a second evaluation starts with *different* settings while one is
  running, log loudly (or error). Plus document the constraint at the setters.
  *Effort: 1-2h. Risk: LOW.*
- [ ] **Option C — fold only the per-call mutation.** Wrappers stop calling
  `set_empty_value` per call; instead the value rides the existing `EvalEnv`
  log + a checked "set once per process unless identical" rule. Halfway house.
  *Effort: 3-4h. Risk: LOW-MEDIUM.*

**Recommendation: B now, A later only if/when concurrent in-process evaluation
becomes a real goal** (e.g. dashboard runs overlapping flows in one process —
today it doesn't).

---

## D3. God-file splits

**Context.** Remaining giants: `lib.rs` (~6.1K after deletions), `adaptive.rs`
(~7.1K), `generic_strategies.py` (~3.1K incl. new SA), `architecture_strategies.py`
(~2.8K), `flow.py` `run()` (630-line method), `worker.py` `_execute_flow()`
(446-line method), `adaptive_cluster.py` (~2.5K). All splits are along seams
the review already mapped (class boundaries / responsibility blocks). Pure
mechanical churn — zero behavior change intended — but big diffs that collide
with any in-flight branch.

- [ ] **Option A — split everything in one pass.** One PR per file, re-exports
  preserved so no caller changes. *Effort: 12-16h total. Risk: LOW per edit but
  HIGH merge-conflict cost if other work is in flight.*
- [ ] **Option B — split only the two Python method-monoliths (recommended).**
  `flow.run()` → `_init_flow / _run_one_experiment / _finalize / _handle_exception`,
  and `worker._execute_flow()` → `_setup_env / _build_evaluators / _parse_config /
  _handle_result`. These are *methods*, so splitting is private-API only — no
  import churn at all, and these two files are where bugs have historically
  concentrated. Rust splits + strategies-file splits wait for a quiet window.
  *Effort: 4-5h. Risk: LOW.*
- [ ] **Option C — defer all splits** until a natural quiet point (e.g. right
  after the next paper submission), then do Option A as a dedicated "no other
  branches open" day. *Effort: 0 now.*

**Recommendation: B** — method extraction is conflict-cheap and pays
immediately in debuggability; file-level splits are better batched for a quiet
window (C for the rest).

---

## D4. `Vec<bool>` / `.tolist()` → numpy migration + connections-resend fix

**Context.** ~14 surviving exports take `Vec<bool>` (PyO3 iterates a Python
list of PyBool objects — ~28 bytes/bit on the wire); `RAMClusterLayer`
train/forward paths do `.flatten().bool().tolist()` per call AND re-send the
layer's immutable `connections` every call. Modern evaluators (caches) already
do this right; the affected paths are the core-layer ones (curriculum /
controller-adjacent LM experiments).

- [ ] **Option A — migrate live call sites to the existing `_numpy` twins
  (recommended).** `RAMClusterLayer`/`TieredRAMClusterLayer` train+forward move
  to `PyReadonlyArray1<u8>` variants (most already exist Rust-side); cache
  `connections` as a numpy array once per layer. Delete the `Vec<bool>` exports
  whose last caller migrates (rest get `#[deprecated]`-style doc note).
  *Effort: 3-4h. Risk: LOW-MEDIUM — needs one quick timing sanity check
  (expect strictly faster) + existing parity tests.*
- [ ] **Option B — also store connections Rust-side** (upload once at layer init
  via a small cache wrapper, like `IDSCacheWrapper` does). Bigger win, bigger
  surface. *Effort: +3h over A. Risk: MEDIUM.*
- [ ] **Option C — leave it** (these paths aren't on the hot IDS loop).
  *Effort: 0. Cost: every curriculum/LM-layer call stays ~10-30× heavier on
  marshalling than needed.*

**Recommendation: A.** B only if profiling shows the per-call connections
upload still dominates after A (it usually won't for ≤100K connections).

---

## D5. 80 surfaced dead-code warnings burn-down

**Context.** Removing the crate-wide `#![allow(dead_code)]` exposed 80 warnings
across 20 files (top: neuron_memory 12, sparse_memory 10, metal_ramlm 10,
adaptive 10). Each is either (a) genuinely dead → delete, (b) test-only → move
into `#[cfg(test)]`, or (c) kept-on-purpose API (e.g. `SparseGpuCache.lookup`
used for CPU verification) → targeted `#[allow(dead_code)]` with a one-line
justification comment.

- [ ] **Option A — full triage in one pass (recommended).** Walk all 80, classify
  a/b/c, delete the dead (expect ~50-60%), annotate the kept. End state:
  `cargo check` is warning-clean, and any future warning means *new* dead code.
  *Effort: 3-4h. Risk: LOW — compiler-verified, deletions are of provably
  unreferenced items.*
- [ ] **Option B — top-4 files only** (42/80 warnings), rest later.
  *Effort: 2h. Risk: LOW. Leaves the build noisy.*
- [ ] **Option C — re-add per-module allows** to silence. *Not recommended —
  re-hides exactly what we just surfaced.*

**Recommendation: A** — it's the cheapest item with a permanent payoff
(warning-clean build = free dead-code detection forever).

---

## D6. Small leftovers (grouped — approve as a single batch?)

- [ ] **D6a. Import-cycle break**: extract framework primitives
  (`OptimizationConfig`, `OptimizerResult`, `EarlyStopping*`, logger, scaler)
  from `generic_strategies.py` into `connectivity/framework.py`;
  `optimization_template` imports from it → cycle gone, and the late-import
  wart at `generic_strategies.py:~1135` disappears. Re-exports keep every
  caller working. *Effort: 2-3h. Risk: LOW (import-order only — verified by
  the existing import sweep).*
- [ ] **D6b. `MemoryVal` WEAK states (Python QUAD parity)**: add WEAK_FALSE/
  WEAK_TRUE to the Python `Memory`/`MemoryVal` + a `cell_to_weight` equivalent
  in `forward_counts`, with a parity test against `ram_accelerator`. Makes the
  Python path comparable to Rust instead of "documented trap". *Effort: 3-4h.
  Risk: LOW-MEDIUM (touches EDRA-era core; the trap doc covers us meanwhile).*
- [ ] **D6c. `ClusterGenome` → neutral module**: move from
  `strategies/connectivity/adaptive_cluster.py` to `wnn/ram/genome.py`
  (re-export kept) — fixes the core→strategies layering inversion at its root.
  *Effort: 1-2h. Risk: LOW.*
- [ ] **D6d. Factory kwargs cleanup**: `OptimizerStrategyFactory.create(type,
  config=GAConfig(...)/TSConfig(...)/SAConfig(...), **overrides)` accepting the
  typed configs that already exist; current ~70-kwarg signature kept as a
  deprecated shim for one release. *Effort: 3-4h. Risk: MEDIUM (worker
  param-forwarding path — needs one end-to-end flow test).*

**Recommendation: approve D6a + D6c now (cheap, structural), D6b when the next
Lamarckian/cell-export work makes Python-side QUAD actually needed, D6d
together with D3-Option-B since both touch the worker config path.**

---

## Approval summary (tick to approve)

| Item | Recommended | Alt | Decision |
|------|-------------|-----|----------|
| D1 orchestrator/checkpoints | **B** schema-unify + SIGTERM port | A full / C none | ☐ |
| D2 per-call globals | **B** guard-rail | A full fold / C halfway | ☐ |
| D3 god-files | **B** method-monoliths only | A all / C defer | ☐ |
| D4 Vec<bool>→numpy | **A** migrate live sites | B +Rust-side conns / C none | ☐ |
| D5 dead-code warnings | **A** full triage | B top-4 files | ☐ |
| D6 smalls | **a+c now**, b+d later | any subset | ☐ |

Estimated total for the recommended set: **~15-20h** across 8-10 commits, each
independently revertable, tests green between every step.
