# HANDOFF: branch `worktree-arch-review-tier1` — merge, rebuild, deploy, then one epilogue

**Audience:** the Claude session working in the main `/Users/lacg/wnn` checkout.
**State:** the ENTIRE plan in `docs/ARCHITECTURE_REVIEW_2026-06.md` is complete —
tiers 1–4 AND the approved D-series (D1–D6 max scope, `docs/DEFERRED_WORK_PLAN.md`),
finishing with the D3 god-file splits on 11/06/2026. The review doc's execution log
(on this branch) is the source of truth. Nothing is half-done; one optional epilogue
remains (§5).

Verification state at handoff: `cargo test` **102/102**, both Rust builds
**warning-clean (0)**, `tests/generic_sa_smoke.py` ALL PASS,
`tests/phased_orchestrator_smoke.py` ALL PASS (incl. real SIGTERM),
production imports verified against the worktree, `Flow`/`run_all_phases`
dry-run tested (fresh + resume + TS-fitness carry).

---

## 1. What's on the branch (~25 commits, `04e50735..HEAD`)

Tier 1–3 + tier-4 batch (active-bug fixes, SA strategy, legacy-stack deletion,
shader preamble, accel facade) — see the previous revision of this file in git
history for the long table. Since then, the D-series:

| Group | What |
|-------|------|
| D5 | 80 dead-code warnings → 0 on BOTH builds; ~700 more dead lines deleted |
| D2 | **BREAKING:** all four Rust global families deleted (`EMPTY_VALUE_BITS`, `MEMORY_MODE`, `NORMAL_CLASS`, `FITNESS_*`); `EvalSettings` threaded per call from PyO3; `set/get_empty_value` pyfns DELETED → **`ABI_VERSION = 2`** |
| D6a+c | `connectivity/framework/` package breaks the import cycle; `ClusterGenome` → `wnn/ram/genome.py` (core layering fixed) |
| D6d | factory kwargs DEAD — `create(strategy_type, opt_config, arch_config, …)` typed configs; unreachable LM runner stack deleted |
| D1 | `wnn/ram/strategies/phased/` (CarryState, schema-2 store, EmergencyDump, orchestrator skeleton); **yaml.gz canonical checkpoints** (legacy json.gz/pickle still load); zero pickle debt |
| D4 | zero `Vec<bool>` params on the Python surface (numpy everywhere); connections cached per layer |
| D6b | `core/cell_semantics.py` + `forward_quad_scores` — Python QUAD parity trap closed (parity test self-activates after the ABI-2 rebuild) |
| D3 (7 commits) | god-files: `flow.run()` 630→46; `worker._execute_flow()` 452→57; `generic_strategies` → `generic_ga/ts/sa` + shim; `architecture_strategies` → 9 modules + shim; `run_all_phases` 644→54 on PhasedOrchestrator (`_SearchOrchestrator` adapter); `adaptive.rs` 6912 → `adaptive/` (12 submodules after the eval follow-up); `lib.rs` 6119 → 358-line crate root + `pyapi/` (15 domain modules) |
| D3.5 bugfix | `get_resume_phase()` + resume validation now recognize canonical `.yaml.gz` checkpoints — latent since the YAML upgrade (it silently returned None = no resume) |

All splits keep backward-compatible re-export shims — **zero caller changes**;
the Python surface of the accelerator is unchanged except the D2 deletions.

## 2. ⚠️ Deploy order (mandatory)

1. Merge `worktree-arch-review-tier1` → `main` (worker idle — restart cancels running flows).
2. Rebuild — the installed accelerator is pre-ABI and will be **refused loudly** by design:
   ```bash
   cd /Users/lacg/wnn && unset CONDA_PREFIX && source wnn/bin/activate && \
     cd src/wnn/ram/strategies/accelerator && maturin develop --release
   python -c "import ram_accelerator as r; print(r.ABI_VERSION)"   # must print 2
   ```
3. Start the worker. Run ONE small IDS flow end-to-end; check `[EVAL-ENV]`,
   `[PARAMS]` markers appear and results look sane before queueing cohorts.
4. The worktree at `.claude/worktrees/arch-review-tier1` can be deleted after
   merge (branch is pushed).

Behavioral notes that survive from the earlier handoff: fail-loud fallbacks
(`WNN_ALLOW_PY_FALLBACK=1` escape, never report those results);
`OptimizerStrategyType` enum ints shifted (nothing persists them by value);
new log markers (`[EVAL-ENV]`, `[PARAMS]`, `[WNN.ACCEL] WARNING`, `[CANCEL-GUARD]`).

## 3. Quick verification before merging

```bash
git fetch && git checkout worktree-arch-review-tier1
cd src/wnn/ram/strategies/accelerator && unset CONDA_PREFIX && cargo test   # 102/102
cd /Users/lacg/wnn && PYTHONPATH=src wnn/bin/python tests/generic_sa_smoke.py          # ALL PASS
PYTHONPATH=src wnn/bin/python tests/phased_orchestrator_smoke.py                        # ALL PASS
PYTHONPATH=src wnn/bin/python -c "import wnn.ram.experiments.worker, wnn.ids.train; print('ok')"
```

## 4. Map of the new module layout (for navigation)

- `accelerator/lib.rs` — thin crate root: singletons, mod decls, `ABI_VERSION`, `#[pymodule]`.
- `accelerator/pyapi/` — ALL `#[pyfunction]`/`#[pyclass]` wrappers, by domain.
- `accelerator/adaptive/` — `metal_state, validation, thresholds, groups, memory,
  eval_parallel, eval_export, eval_single, eval_hybrid, adaptive_eval, gating_eval, tests`.
- `wnn/ram/strategies/connectivity/` — `generic_ga/ts/sa.py`, `architecture_ga/ts/sa.py`,
  `architecture_mixin/config`, `checkpoint_manager`, `grid_search`, `adaptation`,
  `live_progress`, `genome_tracking`; `generic_strategies.py` / `architecture_strategies.py`
  are re-export shims only.
- `wnn/ram/strategies/phased/` — shared orchestration; `phased_search.py` has the
  `_SearchOrchestrator` adapter; `flow.py` has `_RunState`.

## 5. THE ONE REMAINING EPILOGUE (post-merge, optional but planned)

**Decompose `evaluate_genomes_parallel_hybrid_impl`** —
`accelerator/adaptive/eval_hybrid.rs`, ~939 lines, the GA hot path. It was
deliberately NOT split during D3: it threads ~15 shared mutable locals
(timing accumulators, memory pool, packed inputs, `all_results`, progress-log
state) through one pipelined batch loop. Mechanical extraction would mean a
context struct + borrow renegotiation = behavior risk with no way to validate
end-to-end from the worktree (stale-ABI venv).

Recipe (do AFTER merge + rebuild, when real flows can validate):

1. **Golden test FIRST, refactor SECOND.** Add a Rust test in
   `adaptive/tests.rs`: small synthetic dataset (e.g. 2 genomes × 4 clusters,
   fixed connections, fixed seed), call `evaluate_genomes_parallel_hybrid`
   with `RAYON_NUM_THREADS=1` semantics if possible and capture the exact
   `(ce, acc, f1, fpr, threshold)` outputs as the golden expectation.
   Note: non-OI training is order-dependent under multi-thread rayon — for a
   bit-exact golden, either pin threads or set order-independent mode.
2. Introduce two structs in `eval_hybrid.rs`: a loop-invariant
   `HybridBatchConfig` (offsets tables, batch_size, eval_data Arc, env flags)
   and a mutable `HybridBatchAccum` (results, timing totals). Extract along
   the existing comment seams: setup/offset precompute → per-batch train →
   GPU dispatch → CPU scoring fallback → result assembly → timing/progress log.
3. After each extraction step: `cargo test` (golden must stay bit-identical) +
   both builds warning-clean.
4. Final validation: re-run a previously-completed small IDS flow config and
   compare per-genome metrics + `[TIMING]` (`WNN_TIMING=1`) against the
   pre-refactor run — same fitness numbers, no throughput regression
   (Engineering Priority #1 is performance; this function IS the throughput).

Everything else in the review is done. After this, the architecture-review
effort is fully closed.
