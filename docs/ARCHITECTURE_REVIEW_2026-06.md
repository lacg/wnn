# Architecture Review — Python + Rust (10/06/2026)

Multi-agent architectural review of `src/wnn/` (Python, ~84K lines) and the Rust accelerator
(`src/wnn/ram/strategies/accelerator/`, ~28K lines in the 7 largest files). Seven parallel
deep-dives: Python core, strategies, orchestration (flow/worker), evaluator family,
controller-vs-experiments duplication, Rust accelerator, PyO3 boundary.

**Headline:** the inheritance hierarchies are real (generic vs architecture strategies = 3.3%
file similarity, genuine subclassing), the GA core is genuinely shared between IDS and
controller strands, and `Memory` encapsulation is clean. But the review found **2 active
correctness bugs**, a systemic single-source-of-truth gap behind the historical bug class
(MSB-first kernel, u32 truncation), and ~3,000 lines of dead code.

Total program ≈ 50–70h. Tier 1 (~3h) captures the correctness value.

---

## Tier 1 — Fix now (correctness, ~3h)

### 1.1 ⚠️ ACTIVE BUG: QUAD weights inverted in multistage CPU fallback
- **Where:** `multistage.rs:3193-3198` inside `train_and_get_tiered_scores` (entry ~3051).
- **What:** hardcodes `FALSE => 0.0, TRUE => 1.0, _ => empty_value` on the **raw cell value**.
  `TRUE=1` is the *ternary* constant; in QUAD mode (the mandated default) cell `1` is
  WEAK_FALSE and cell `3` is TRUE. The fallback therefore scores **WEAK_FALSE as 1.0 and
  TRUE as 0.25** — inverted, not just degraded.
- **Trigger:** only when both Metal dense + sparse eval fail (e.g. `WNN_NO_METAL`, evaluator
  init failure) → silent drift, same shape as the Metal-scorer fallback warn-onced in 25cf1c3a.
- **Root cause:** the correct helper `cell_to_weight` (`adaptive.rs:284`) is private, so
  `multistage.rs` couldn't reuse it.
- **Fix:** move/promote `cell_to_weight` to `pub fn` in `neuron_memory.rs` next to
  `QUAD_WEIGHTS`; use it in multistage fallback + the adaptive.rs internal copy; unit test
  asserting QUAD mapping on the CPU fallback. **Effort: ~1h.**

### 1.2 ⚠️ ACTIVE BUG: arity mismatch — `adaptive_cluster.py:1532` cannot call Rust
- **Where:** Python `adaptive_cluster.py:1532` passes 15 args; Rust `evaluate_genomes_parallel`
  (`lib.rs:4114`, no `#[pyo3(signature)]`) requires 17 (`neuron_sample_rate`, `rng_seed`).
  Invoking this path raises `TypeError` immediately — yet it carries F1=0.49 cancel-guard
  patches (treated as live while uncallable).
- **Fix:** add `#[pyo3(signature=(..., neuron_sample_rate=1.0, rng_seed=0))]` defaults to the
  raw entry points AND fix-or-delete the stale call site. Decide whether the LM-era
  `RustParallelEvaluator` path is live at all. **Effort: ~1-2h.**

### 1.3 Length validation compiled out in release builds
- **Where:** flat-array invariants guarded only by `debug_assert_eq!` (`adaptive.rs:4794`,
  also 1160, 2494); `ids_cache.rs` has zero length validation.
- **What:** `maturin develop --release` (the only supported build) compiles the asserts out →
  a misaligned `connections_flat` (e.g. the `if g.connections is not None` asymmetry in the
  six Python flatteners) produces silently wrong results, not an error.
- **Fix:** promote to real `if … return Err(PyValueError)` at the PyO3 boundary. O(num_genomes)
  cost, negligible vs training. **Effort: ~1h.**

---

## Tier 2 — This week (~12h)

### 2.1 Single source of truth for cell semantics (CPU/Metal/fallback)
- `QUAD_WEIGHTS` exists in **10 places**: `neuron_memory.rs:40` (canonical),
  `controller.rs:38` (`QSR_WEIGHTS`), hand-rolled matches in adaptive/multistage, and 6 Metal
  shader copies (`ramlm.metal:34`, `sparse_forward.metal:31`, `batched_sparse_forward.metal:35`,
  `sparse_ce.metal:36`, `neuron_stats.metal:21`, `controller_rollout.metal:23`) — enforcement
  is a comment.
- `compute_address` has **~14 independent Rust implementations** + per-shader copies;
  `marker_train.metal:108-116` documents the LSB-first drift incident inline.
- **Fix:** (a) shared MSL preamble (`shaders/common.metal`) injected via
  `concat!(include_str!(…))` — all shaders already load via `include_str!`; (b) collapse Rust
  address computation onto the `neuron_memory.rs` variants; (c) one GPU-vs-CPU parity test per
  kernel (extend the `run_marker_train_parity_test` pattern). **Effort: ~8h.**
  Highest leverage against the next "all prior results may be bug-biased" event.

### 2.2 `wnn/accel.py` facade + kill silent fallbacks
- 36 files import `ram_accelerator` raw; **32 of 116 exports (~28%) are dead** (incl. all three
  `predict_all_batch*` advertised in CLAUDE.md "Key Functions").
- Genome flattening duplicated in ≥6 places (`ids_evaluator.py:373`,
  `multistage_evaluator.py:286`, `tiered_evaluator.py:237/288/381`, `adaptive_cluster.py:1503`,
  `bitwise_evaluator.py:148`).
- Silent PyTorch fallbacks violating No-Python-Shortcuts: `RAMClusterLayer.py:392-399`,
  `TieredRAMClusterLayer.py:588-596`, `AdaptiveClusteredRAM.py:250,606`, `gating.py:691-693,935`,
  `control/evaluator.py:929-932,1074-1077` (cancel check degrades to "never cancelled").
- **Fix:** thin `src/wnn/accel.py`: one import point, `ABI_VERSION` constant exported from
  lib.rs and asserted, fail-loud policy (`WNN_ALLOW_PY_FALLBACK=1` escape hatch), one
  `GenomeMarshaller`, typed wrappers for the ~25 live functions. **Effort: ~4-6h.**

### 2.3 Quarantine the vestigial LM-era optimizer stack (~2,700 lines)
- `connectivity/genetic_algorithm.py`, `tabu_search.py`, `simulated_annealing.py`,
  `accelerated.py`, `model_optimizer.py`, `per_cluster.py` + `CONNECTIVITY_*` factory values
  (`factory.py:256-258`) — referenced only by two LM-era test scripts.
- Also remove `#![allow(dead_code)]` (`lib.rs:5`) so rustc reports Rust-side dead code; delete
  or `#[cfg(feature="bench")]`-gate dead exports + in-module test harnesses.
- **Fix:** move to `legacy/` (or delete; git preserves), update the two test scripts.
  **Effort: ~2-3h.**

---

## Tier 3 — Next (~15h)

### 3.1 `EvalConfig`: retire env-vars-as-parameter-passing
- 35 unique `WNN_*` flags, 56+ read sites (38 in adaptive.rs alone, many mid-hot-loop).
- Worst: `adaptive.rs:5723-5737` passes an argument to its callee via
  `std::env::set_var("WNN_OVERRIDE_THRESHOLD")` + `remove_var` — non-reentrant under rayon,
  `unsafe` in Rust 2024, invisible to reproducibility. Progress metadata travels the same way
  (`WNN_PROGRESS_*`).
- Per-call global mutation: `set_empty_value()` per evaluation (`lib.rs:4336`), `MEMORY_MODE`,
  `NORMAL_CLASS`, `FITNESS_WEIGHTS` statics → two concurrent evaluations cannot safely coexist.
- **Fix:** one `EvalConfig` struct resolved once at the PyO3 boundary (env = defaults only),
  passed by reference, logged once per call. **Effort: ~6-8h.**

### 3.2 Param validation at ingestion (the Rule-6 bug class, structurally)
- Typed configs (`ExperimentConfig`, `FlowConfig`) degrade to raw dicts at the dashboard JSON
  boundary; 50+ scattered `.get(key, default)` reads.
- Hazards: `ids_k_folds` default **5** (worker.py:786,1172) vs **1** (`ids_evaluator.py:102`);
  `min_bits` default 10 (bitwise) vs 4 (multistage/IDS) per branch; typo'd
  `wnn_order_independent_train` silently ignored → order-dependent training;
  `neuron_sample_rate` read in 6 places; chained fallbacks
  (`fitness_weight_f1` → `ids_fitness_weight_f1` → 0.0); **no unknown-key detection anywhere**.
- **Fix:** validate at flow ingestion: one defaults table per param (single source of truth),
  reject unknown keys, log the resolved config. **Effort: ~4h.**

### 3.3 Shared phased-search orchestrator (controller ↔ experiments)
- ~500-600 duplicated orchestration lines (`control/phased_ga.py:752-958` vs
  `experiments/phased_search.py:1162-1763`). Controller version is strictly better:
  full-population-carry at runtime (experiments relies on checkpoint loading), generation
  counter + patience in checkpoints (true resume — experiments restarts phases at gen 0,
  patience reset), SIGTERM emergency dump. Formats incompatible (pickle vs json.gz).
- **Fix:** extract `PhasedSearchOrchestrator`; port controller resume semantics + emergency
  dump to experiments; unify checkpoint schema. **Effort: ~8-10h.**

---

## Tier 4 — Opportunistic / background

| Item | Where | Effort |
|------|-------|--------|
| Break `optimization_template` ↔ `generic_strategies` import cycle; make resume state (`_resume_start_gen`, `_resume_patience` via `getattr`) an explicit contract | `generic_strategies.py:1111,1316,1325` | 4-6h |
| God-file splits along class boundaries (lib.rs → per-domain PyO3 submodules; adaptive.rs → calibration/metrics/group_memory; flow.py `run()` 630 ln; worker.py `_execute_flow()` 446 ln; the strategies trio) | various | 15-20h |
| Layering inversion: `core/` imports `strategies/` ×6; `PerplexityCalculator` constructed **per forward call** (`RAMClusterLayer.py:350`) | core/ | 3-4h |
| Python `Memory.py` still ternary-semantic: `MemoryVal` lacks WEAK states; `core/__init__.py:174` documents TERNARY as default (contradicts CLAUDE.md); Python/Rust use different bit encodings for the same states (Lamarckian export trap). Minimum: flip docs + runtime warning | core/ | 1-6h |
| Evaluator family: extract `_flatten_genomes` / k-fold accumulation / metrics construction template methods into `BaseEvaluator`; fix missing `eval_time_ms` in `_evaluate_batch_streaming_kfold` (`ids_evaluator.py:729-775`) | architecture/ | 3-4h |
| Data transfer: 14 exports still take `Vec<bool>` via `.tolist()` (e.g. `RAMClusterLayer.py:1241-1258` re-sends immutable connections per call) — migrate live sites to `_numpy` twins | boundary | 3-4h |
| Controller IC-sampling flattening duplicated (`control/evaluator.py:846` vs `ga_memory.py:198`) — CPU/GPU parity by convention only | control/ | 1h |
| Factory sprawl: `OptimizerStrategyFactory.create` has ~70 kwargs; accept typed configs instead | factory.py:297-618 | 3-5h |
| Update CLAUDE.md "Key Functions" table (lists 3 dead functions) | docs | 0.5h |

---

## Execution log

- [ ] 1.1 multistage QUAD fallback fix + `pub cell_to_weight`
- [ ] 1.2 `evaluate_genomes_parallel` signature defaults + stale call site
- [ ] 1.3 release-mode length validation at PyO3 boundary
- [ ] 2.1 shared shader preamble + address-computation collapse + parity tests
- [ ] 2.2 `wnn/accel.py` facade + ABI version + fail-loud fallbacks
- [ ] 2.3 legacy stack quarantine + remove `#![allow(dead_code)]`
- [ ] 3.1 `EvalConfig`
- [ ] 3.2 param validation at ingestion
- [ ] 3.3 shared phased-search orchestrator
- [ ] Tier 4 items (opportunistic)
