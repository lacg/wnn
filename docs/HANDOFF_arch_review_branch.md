# HANDOFF: branch `worktree-arch-review-tier1` — verify, merge, rebuild

**Audience:** the Claude session working in the main `/Users/lacg/wnn` checkout.
**Context:** this branch executes the full plan in `docs/ARCHITECTURE_REVIEW_2026-06.md`
(tiers 1–3 + a tier-4 batch), produced in a separate worktree session on 10/06/2026.
The review doc's execution log is the source of truth for what is done vs deferred.

---

## 1. What's on the branch (12 commits, `04e50735..HEAD`)

| Commit | What |
|--------|------|
| `04e50735` | **FIX (active bug):** inverted QUAD weights in multistage CPU-fallback scoring + warn-once tripwire + GPU-vs-CPU parity test |
| `8f33a4ea` | docs tick |
| `fd3f466a` | **FIX (active bug):** `evaluate_genomes_parallel` arity mismatch (pyo3 signature defaults + stale 15-arg call site) |
| `c2e3a8d6` | release-mode flat-genome validation (`PyValueError`) at all 11 PyO3 entry points (debug_asserts were compiled out) |
| `5b085e86` | **`GenericSAStrategy` + `ArchitectureSAStrategy`** (Garcia-2003 SA on the live framework, `ARCHITECTURE_SA` in factory) + **DELETE legacy LM-era optimizer stack** (~7.1K lines Python + 3 test scripts) |
| `58694a28` | **DELETE 39 dead PyO3 exports + ram.rs/per_cluster.rs/database.rs** (+rusqlite dep); crate-wide `#![allow(dead_code)]` removed (80 warnings now visible = cleanup inventory) |
| `265d2e6c` | **`shaders/common.metal`** single source of truth (QUAD weights, MSB-first address, cell read) prepended to 8 shaders; Rust address dupes collapsed onto `neuron_memory.rs`; dense+sparse parity + shader-compile smoke tests |
| `99e9383c` | **`wnn/accel.py` facade**: `ABI_VERSION` check, `flatten_genomes` marshaller (replaces 6 copies), silent PyTorch fallbacks → fail-loud |
| `ab557e5e` | `set_var(WNN_OVERRIDE_THRESHOLD)` hack eliminated (real param) + `[EVAL-ENV]` resolved-flags log |
| (3 more) | params registry + ingestion validation; explicit `restore_resume_state()` contract + phased_search population-cascade guard; tier-4 batch (eval_time_ms fix, ternary docs, calc caching, `sample_ics_flat` dedup) |

Verification state when handed off: `cargo test` **107/107**, `tests/generic_sa_smoke.py`
PASS, 17 production modules import clean against the branch.

## 2. ⚠️ Behavioral changes you MUST know before deploying

1. **ABI gate (the big one).** `lib.rs` exports `ABI_VERSION = 1`; `wnn/accel.py`
   asserts it at import. The currently-installed accelerator (pre-branch) has no
   ABI constant → it is treated as **stale and refused** by accel-gated paths
   (core layers, gating probes, controller cancel checks). Therefore:

   **Deploy order is mandatory: merge → `maturin develop --release` → restart worker.**
   Never start the worker between merge and rebuild. Build one-liner:
   ```bash
   cd /Users/lacg/wnn && unset CONDA_PREFIX && source wnn/bin/activate && \
     cd src/wnn/ram/strategies/accelerator && maturin develop --release
   ```

2. **Fail-loud fallbacks.** Silent PyTorch fallbacks now raise unless
   `WNN_ALLOW_PY_FALLBACK=1` (then they warn once). If something starts raising
   `ImportError: ram_accelerator unavailable…`, that is the design working —
   check the build, don't suppress.

3. **`OptimizerStrategyType` enum ints shifted** (CONNECTIVITY_* deleted;
   ARCHITECTURE_GA is now 1, ARCHITECTURE_SA = 3). Verified: nothing persists
   these by integer value. If you see code comparing raw ints, that's a bug.

4. **Deleted Rust exports** (39, e.g. `predict_all_batch*`, `evaluate_batch_cpu/metal`,
   `evaluate_candidates_parallel*`, all `per_cluster_*`): grep-verified zero
   callers in src/scripts/tests. If some external/uncommitted script calls one,
   it fails with AttributeError — restore from git or migrate it.

5. **New log markers to expect** in worker/flow logs:
   - `[EVAL-ENV] memory_mode=2 resolved flags: …` once per process (reproducibility record)
   - `[PARAMS] flow=N …` at ingestion; `⚠️ UNKNOWN PARAM 'x' — did you mean 'y'?`
     means a typo'd dashboard param that previously vanished silently
   - `[MULTISTAGE] WARNING: GPU evaluation unavailable…` = CPU fallback engaged (investigate)
   - `[WNN.ACCEL] WARNING…` = Python fallback escape hatch active (never report those results)

## 3. Suggested verification checklist (before merging)

```bash
git fetch && git checkout worktree-arch-review-tier1
# 1. Rust
cd src/wnn/ram/strategies/accelerator && unset CONDA_PREFIX && cargo test   # expect 107/107
# 2. Python (no rebuild needed for these)
cd /Users/lacg/wnn && PYTHONPATH=src wnn/bin/python tests/generic_sa_smoke.py   # ALL PASS
PYTHONPATH=src wnn/bin/python -c "import wnn.ram.experiments.worker, wnn.control.evaluator; print('ok')"
# 3. Review the two active-bug fixes by eye (small diffs):
git show 04e50735 -- src/wnn/ram/strategies/accelerator/multistage.rs
git show fd3f466a
```

Merge when the worker is idle (restart-cancels-running-flow rule), rebuild,
then run one small IDS flow end-to-end and check the new log markers appear
and results look sane before queueing real cohorts.

## 4. New capability worth knowing

`OptimizerStrategyType.ARCHITECTURE_SA` — Simulated Annealing (Garcia 2003:
T₀=1.0, cooling 0.95, Metropolis) on the modern framework. Parallel chains, one
batch eval per iteration, full population carry, `population_size` → chains.
Available to flows exactly like GA/TS once deployed.

## 5. Deferred work (inventoried, not half-done)

See the execution log + Tier-4 section of `docs/ARCHITECTURE_REVIEW_2026-06.md`:
PhasedSearchOrchestrator extraction + checkpoint schema unification, per-call
globals folding, god-file splits, Vec<bool>→numpy migration, MemoryVal WEAK
states, factory kwargs cleanup, and the 80 surfaced dead-code warnings.
