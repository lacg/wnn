# Sparse footprint fix — measure & store the real materialized-cell count

**Status:** design / approved decisions, NOT yet implemented (16/06/2026).
**Decisions (Luiz):** (1) **new column**, (2) **backfill `best_genomes` first**, (3) **store the COUNT** (convention-free primitive, not a byte figure).

---

## 1. Problem

`genomes.total_memory_bytes` is computed as the **dense theoretical** size and is
meaningless for our sparse memories. Two identical code sites:
- `src/wnn/ram/experiments/tracker.py:103` (`GenomeConfig.total_memory_bytes`)
- `src/wnn/ram/experiments/data_layer.py:133` (same)

both do `Σ clusters × neurons × 2^bits` (1 byte/address), capping at
`SQLITE_INT_MAX = 2^63−1 = 9223372036854775807` on overflow / `bits ≥ 63`.

A third writer — the Rust dashboard — hardcodes **`0`**:
- `dashboard/src/db/best_genomes.rs:202` → `VALUES (?, …, 0, …)` for every genome it inserts.

**Measured spread (db/wnn.db, 16/06):**

| arch | count | `total_memory_bytes` | cause |
|---|---|---|---|
| `ids` (all IDS work) | 10,339 | **uniformly 0** | dashboard hardcodes 0 |
| `tiered` high-bit (LM) | 19,027 | **i64::MAX** | dense `2^bits` overflow → cap |
| `tiered` low-bit (LM) | 260,150 | dense "real" | computes, but overcounts vs sparse |

The real sparse footprint was **never stored anywhere** — there is no lost column
value to recover; the number was simply never computed. It IS, however,
deterministically reproducible (connections in DB + flow seed + dataset).

## 2. The asset (already computed at eval-time)

After training, each genome's `GenomeExport` (`adaptive/eval_export.rs:13`) holds:
- `dense_exports: Vec<Vec<i64>>` — dense memory words (bits ≤ SPARSE_THRESHOLD).
- `sparse_exports: Vec<SparseGpuExport>` where `keys: Vec<u64>` = the **distinct
  trained addresses** (bits > SPARSE_THRESHOLD).

The materialized-cell count is therefore free to compute from the export:
```
materialized_cells =
    Σ dense clusters:  num_neurons × 2^bits         (full dense array)
  + Σ sparse clusters: keys.len()                   (distinct trained addresses)
```
This is the **convention-free primitive**. Every byte figure derives from it:
- runtime FxHashMap bytes = `ClusterStorage::memory_bytes()` (`neuron_memory.rs:758`): `neurons×56 + entries×12`
- packed/edge bytes ≈ `entries × (⌈bits/8⌉ key + ¼ byte QUAD cell)`
- FPGA = LUTs (see `feedback_sparse_fpga_size` memory) — not bytes at all

Storing the **count** lets us report any convention later without re-measuring.

## 3. Design

### 3a. New column
```sql
ALTER TABLE genomes ADD COLUMN materialized_cells INTEGER;  -- NULL = not measured
```
- Authoritative size primitive. NULL until measured.
- `total_memory_bytes` is **deprecated** — leave in place (don't break dashboards
  reading it), stop trusting it; optionally stop the silent i64::MAX cap later.
- (Naming: `materialized_cells` covers BOTH dense and sparse regimes — it's "cells
  actually stored." Rename to `sparse_entries` if we decide to only ever store the
  sparse-regime count; current choice is the unified one.)

### 3b. Rust — compute + surface
1. `GenomeExport::materialized_cells(&self) -> u64` — the formula in §2. Pure.
2. Surface it out of eval. The export is in scope in
   `evaluate_genomes_parallel_hybrid_impl` (eval_hybrid.rs) right where
   `batch_exports` is consumed — compute the count there and carry it as a new
   field on the per-genome result tuple (currently
   `(idx, ce, acc, f1, fpr, threshold, ms)` → add `materialized_cells`).
   Ripple targets (tuple unpackers): `evaluate_genomes_parallel_hybrid` +
   `_with_override` + `_adaptive`, the PyO3 wrapper, `ids_cache.rs` callers,
   `eval_parallel.rs` (LM 4-tuple — drops the field), `token_cache.rs`.

### 3c. Write path (one unified writer)
The **worker (Python)** owns the eval call and the `genome_id`. After eval:
```sql
UPDATE genomes SET materialized_cells = ? WHERE id = ?;   -- idempotent
```
The fill is **constant** for a given (genome, train-set), so set-once / overwrite
is safe and path-agnostic (works whether the row came from the Python `data_layer`
or the Rust dashboard insert). Wire into the existing
`tracker.record_genome_evaluations_batch` flow which already has `genome_id`.

### 3d. Deprecate the dense formula
Stop relying on `GenomeConfig.total_memory_bytes` (tracker.py / data_layer.py) and
the dashboard hardcoded `0`. Keep the column; new code reads `materialized_cells`.

## 4. Backfill (best_genomes first)

The fill is reproducible. One-shot, idempotent:
`scripts/backfill_sparse_memory.py`
1. Target set = distinct `genome_id` in `best_genomes` (dedupe; paper-relevant).
   Skip rows where `materialized_cells IS NOT NULL`.
2. **Group by flow/dataset** → load each IDS dataset ONCE (amortize the heavy step;
   reuse `IDSCacheWrapper`, zero re-upload).
3. Per genome: read `connections_json` + `tiers_json` from DB; train against the
   cached data via a NEW Rust primitive
   `IDSCacheWrapper.measure_genome_memory(genome) -> u64` (trains, returns the
   export's `materialized_cells`; **no Python reimpl**, per the no-shortcuts rule);
   `UPDATE genomes SET materialized_cells = ?`.
4. Resumable (idempotent skip), batchable.

**Scope:** best_genomes only now (dedupes to ~hundreds–low-thousands unique ids).
All-ids (10,339) and tiered-LM (279k) are opt-in follow-ups.

**Cost:** dominated by the per-dataset load, not the per-genome train — grouping by
dataset keeps it tractable as a background job.

## 5. Implementation sequence (each step independently verifiable)

**STATUS 16/06: steps 1, 2, 6 DONE + pushed (source-only, live-wheel-safe, NO live
DB/process touched). Steps 3, 4, 5 + the migration-apply + the backfill RUN are
staged for the worker-idle window — see the IDLE-WINDOW BATCH note below.**

1. ✅ **DONE (`d3c9a21b`)** Schema — `materialized_cells` column in BOTH schemas
   (data_layer.py CREATE+migration; dashboard schema.rs + migrations.rs) + the
   `Genome` model field. The live `ALTER` runs on the next dashboard/worker restart
   (idle migration) — NOT applied to db/wnn.db yet. `total_memory_bytes` marked DEPRECATED.
2. ✅ **DONE (`32171d0f`)** Rust compute — `GenomeExport::materialized_cells()` +
   unit test (dense 32 + sparse 5 = 37, empty=0). cargo 105/105 bit-clean.
3. ⏳ **IDLE** Rust surface — add the field to the result tuples + PyO3 + unpackers.
   Verify: existing tuple consumers compile; a golden-style assert that the count
   matches a hand-computed tiny case. `cargo test` + `maturin develop` (at idle —
   this is an accelerator rebuild, same deploy window as the eval_hybrid validation).
4. ⏳ **IDLE** Worker write — `UPDATE … materialized_cells` in
   `record_genome_evaluations_batch` (consumes step-3's new tuple field).
   Verify: a fresh IDS flow populates `materialized_cells` (non-NULL, plausible).
5. ⏳ **IDLE** Backfill primitive — Rust `IDSCacheWrapper.measure_genome_memory`.
   Verify: returns the same count as a live eval for a known genome (e.g. 838008).
6. ✅ **SKELETON DONE (`8bf56b36`)** `scripts/backfill_sparse_memory.py`,
   best_genomes scope. `--dry-run` works on the live DB (read-only) and reports the
   real target set: **10,091 genomes across 101 dataset groups**. The per-genome
   measure is a guarded `TODO(step 5)`. After 5 + migration: run for real (verify on
   838008/838009 first → sub-GiB in any byte convention), then the full set.

### ⭐ IDLE-WINDOW BATCH — one `maturin develop --release` does TRIPLE DUTY
The worker-idle window (worker pid draining XDS, ~days out) runs ALL of these on a
SINGLE accelerator rebuild + restart, because they share the same deploy gate:
1. **eval_hybrid GPU-dispatch validation** — the deferred 6th-seam refactor's only
   behavioral check: queue ONE cicids-random 500n×34b flow, compare F1/FPR/CE to the
   XDS-cicids 96b-Wa cohort (flows 4151–4175). See [[arch-review-merge-plan]].
2. **First live `materialized_cells` measurement** — once step 3 lands, THAT SAME
   cicids validation flow populates `materialized_cells` for its genomes at eval-time,
   for free (500n×34b → sparse path → real count).
3. **Backfill the historical 10,091** — run `scripts/backfill_sparse_memory.py`
   (no `--dry-run`) once step 5 + the migration are in.

**Why they share:** the migration auto-applies on the restart; steps 3+5 are both
accelerator-rebuild changes; and storing the COUNT (not a byte convention) means the
validation flow's eval-time measurement and the backfill produce the SAME primitive —
no re-measuring, no convention mismatch. Order at idle: pause queue → let running flow
finish → merge/rebuild (eval_hybrid seam was on main already) → restart (migration
fires) → validation flow (also first live measurement) → backfill run → unpause.

## 6. Open / deferred
- Final column name (`materialized_cells` vs `sparse_entries`).
- Whether to also store a derived `packed_bytes` (paper convention) or compute on
  read. (Count is primitive; derive on read is cleaner.)
- Backfill of tiered-LM genomes (fix the i64::MAX) — separate pass, lower priority.
- Dashboard UI: surface `materialized_cells` (+ packed-byte derivation) on the
  genome/experiment pages; deprecate the `total_memory_bytes` display.
