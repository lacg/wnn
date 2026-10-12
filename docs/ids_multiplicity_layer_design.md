# IDS Multiplicity Layer: Design

Status: DESIGN ONLY, nothing implemented. Grounded in the live branch
`controller-grid-orchestrator-unification` @ `05fe9cad` (worker ABI 16). All
file:line references are to that commit. The worktree branch was originally cut
from a July `main` (ABI 6); it was fast-moved to the live tip before the survey.

Owner requirement (Luiz): a LAYER (not a neuron type) that holds the unique
neurons plus a multiplicity count for each one. It trains each unique neuron once
and scores as a count-weighted vote. It must be transparent to the GA, the genome
format, the evaluator API and the results. It is selected at construction when
`neuron_sample_rate >= 1.0` and is never selected below that. It must exist in
BOTH Rust and Python, and it must save training time AND memory.

---

## 0. Why this is exact (the invariant the whole design rests on)

At `neuron_sample_rate >= 1.0`, `should_skip_sample` (shaders/marker_train.metal:89,
CPU twin in `adaptive/eval_export.rs` `train_genome_in_slot`) returns false before
it reads the neuron index. That skip hash is the ONLY place where training depends
on neuron position or on the RNG seed (verified by grepping marker_train.metal and
marker_slots.metal). Each neuron's slot region is written only by that neuron's own
thread(s). So two neurons with the same (cluster, bits, connections) receive the
same writes, and they end in identical memory **iff the writes commute**:

- QUAD_WEIGHTED / QUAD_BINARY / QSR: commutative only on the OI counter path
  (`order_independent_training_active`, core/neuron_memory.rs:1611). The legacy
  clamped nudge is order-dependent, and with z-axis example chunking
  (marker_train.rs:1690) two copies diverge.
- TERNARY / PLN (TRUE-wins lattice) and BINARY (one-shot set): order-independent
  by construction (`CellMode::uses_oi_counters` doc, core/cell_mode.rs:98-117).

Therefore the selection gate is exactly:

```
applies(rate, mode) = rate >= 1.0  &&  training_is_order_independent(mode)
training_is_order_independent(mode) = !CellMode(mode).uses_oi_counters()
                                      || order_independent_training_active(mode)
```

Production SP cohorts run OI + QUAD_WEIGHTED, so SR1 qualifies. A legacy-QUAD flow
at rate 1.0 keeps the plain layer. That is correct: its copies are not identical
even in the plain layer.

**Scoring is kept bit-identical by construction, not by float argument.** The
multiplicity layer does NOT change any scorer. It hands every scorer a
`GenomeExport` that has the same logical neurons, groups, connections and
per-neuron cell contents as the plain layer. The only difference is that the
sparse key/value storage is shared between copies (aliased `offsets`/`counts`).
Every scorer still loops over the logical neurons in the same order, so the f32
sums are identical in value and in order. That also holds for QSR/PLN, whose read
coin is keyed on the logical neuron index (`qsr_key`, core/neuron_memory.rs:244),
and for coverage-aware reads. There is no float-order caveat in the shipped
design. Section 3 explains why a count-weighted kernel was rejected.

---

## 1. The layer abstraction on the IDS path (what exists)

There is no `Layer` object on the IDS path in either language. Today the trained
layer is the **`GenomeExport`** (adaptive/eval_export.rs:13): connections, groups
(`ConfigGroup`, adaptive/groups.rs:226), and sparse (`SparseGpuExport`,
adaptive/memory.rs:504) or dense words plus coverage, one per group. Every scoring
path consumes it:

- `compute_per_example_scores` (eval_export.rs:871): CPU dense, CPU sparse, and the
  dispatch to `evaluate_group_metal` / `evaluate_group_sparse_gpu`
  (adaptive/memory.rs)
- `evaluate_genome_hybrid` (eval_single.rs:7), including the
  `MetalSparseCEEvaluator` / `MetalGroupEvaluator` multiclass fast paths
  (metal_genome_eval.rs)
- `train_and_score_*` / `predict` (eval_single.rs:599-1300), thresholds,
  multiclass modes, k-fold, and validation

The constructor seam is **`batched_train_offspring`** (marker_train.rs:540). Every
GPU-trained `GenomeExport` comes out of it:

- GA eval routes: B14 shape-group (eval_hybrid.rs:944), Option-B (:1046) and B12
  hybrid GPU arm (:1172)
- the per-genome route `train_one_genome_cpu` → `train_single_via_marker`
  (eval_hybrid.rs:195/270 → eval_single.rs:545/565)
- all `train_and_score_*` / `predict` entry points, k-fold, full-train validation
  and multiclass (ids_cache.rs → eval_single.rs:643/808/965/1154)
- the adaptive/genesis path (adaptive_eval.rs:897/1152)
- the over-budget single-genome neuron-chunked route
  (`train_single_genome_neuron_chunked`, marker_train.rs:634), which is reached
  from inside it

**Decision: extend `GenomeExport`.** It is the existing layer type. It gets a
multiplicity-backed construction mode and a field that records the multiplicity.
`batched_train_offspring` is the factory that selects the mode. Rust has no
inheritance, so "extend" means a new constructor plus a documented storage
invariant ("neuron ranges may alias") on the same type. It does not mean a parallel
type, which would be the forbidden second way.

The neuron identity rule already exists in ram_core: `push_distinct` in
core/neighbor_search.rs:1328, key `(bits, connections)` per cluster, first
occurrence kept (IDS-26 crossover dedupe). The multiplicity layer MUST use the same
identity. Under the no-duplicates rule that key is promoted to one shared
definition (section 2.1).

The other training substrates do not produce their exports through the marker
trainer:

- `IDSGenomeStreamer` (ids_streaming.rs:69/110): CPU `GroupMemory` +
  `train_genome_in_slot` + `export_genome_for_gpu`
- hierarchical `multistage.rs` (:4130, :4418)
- the `PATH2_FALLBACK` dense CPU fallbacks (eval_hybrid.rs:309, eval_single.rs:690,
  etc.)

They stay on the plain layer in Stage 1. The section 6 build order adds streaming
in Stage 3 through the same `GenomeExport::from_unique`.

---

## 2. Blueprint

### 2.1 ram_core (shared substrate)

**NEW `core/neuron_multiplicity.rs`**: one struct, `NeuronMultiplicity`.

```rust
/// Logical genome → its distinct neurons + how many logical copies each has.
/// Identity = (cluster, bits, connections), first occurrence is the
/// representative (the SAME rule as neighbor_search's IDS-26 dedupe).
#[derive(Clone, Debug)]
pub struct NeuronMultiplicity
{
	unique_of: Vec<u32>,       // logical neuron (genome order) -> unique index (unique-genome order)
	representative: Vec<u32>,  // unique index -> first logical neuron with that key
	counts: Vec<u32>,          // unique index -> number of logical copies (sum == logical N)
	unique_npc: Vec<usize>,    // per cluster: distinct neurons (cluster order preserved)
}

impl NeuronMultiplicity
{
	pub fn applies(neuron_sample_rate: f32, memory_mode: u8) -> bool;
	pub fn collapse(bits_per_neuron: &[usize], neurons_per_cluster: &[usize], connections: &[i64]) -> Self;
	pub fn is_trivial(&self) -> bool;                 // every count == 1 -> caller uses the plain path
	pub fn num_unique(&self) -> usize;
	pub fn unique_of(&self) -> &[u32];
	pub fn counts(&self) -> &[u32];
	pub fn unique_neurons_per_cluster(&self) -> &[usize];
	pub fn gather_bits(&self, bits_per_neuron: &[usize]) -> Vec<usize>;
	pub fn gather_connections(&self, bits_per_neuron: &[usize], connections: &[i64]) -> Vec<i64>;
}

/// The one neuron-identity key (promoted; neighbor_search uses it too).
pub fn neuron_key(bits: usize, connections: &[i64]) -> (usize, Vec<i64>);
```

Within each cluster the unique order is first-occurrence order, which makes the
collapse deterministic. Each method stays at or under about 10 lines; `collapse`
splits into `collapse_cluster`.

**MODIFY**
- `core/lib.rs`: `pub mod neuron_multiplicity;`
- `core/neuron_memory.rs:1611`: add `pub fn training_is_order_independent(memory_mode: u8) -> bool`
  next to `order_independent_training_active` (the single gate, same doc rule).
- `core/neighbor_search.rs:1328`: `push_distinct` keys on `neuron_multiplicity::neuron_key`.
  This changes no behavior; the golden digests in `neighbor_search_golden.rs` must
  stay identical.

### 2.2 Worker crate (`ram_accelerator`)

**MODIFY `adaptive/eval_export.rs`: `GenomeExport`, the layer**
- New field `pub multiplicity: Option<ram_core::neuron_multiplicity::NeuronMultiplicity>`
  (None = plain layer). Set it to None at the 4 existing literals:
  eval_export.rs:119 (`empty`), eval_export.rs:840 (`export_genome_for_gpu`),
  marker_train.rs:770 and marker_train.rs:1989, plus test literal
  adaptive/tests.rs:753.
- New constructor:

```rust
impl GenomeExport
{
	/// Multiplicity-backed layer: `unique` was trained on the collapsed genome.
	/// Builds the LOGICAL layout (groups, connections) from the logical genome,
	/// sparse ranges ALIASED to the unique storage (keys/values held once),
	/// dense words + coverage copied per logical neuron (≤ 1 KB/neuron at b ≤ 12).
	pub fn from_unique(
		unique: GenomeExport,
		multiplicity: NeuronMultiplicity,
		logical_bits: &[usize],
		logical_npc: &[usize],
		logical_connections: &[i64],
	) -> GenomeExport;
}
```

Submethods: `logical_groups`, `expand_sparse_group`, `expand_dense_group`,
`unique_neuron_view(cluster, u_local)`.

Correctness note: collapsing removes copies only, so the per-cluster max bits are
unchanged. Logical and unique groups therefore have the same `bits` and
`words_per_neuron`, and dense words copy verbatim. Group *composition* can differ
when coalescing buckets differ. That is why sparse ranges are gathered per logical
group: a straight move for single-cluster IDS, and a copy-once of each unique range
for multi-group K-class.

**MODIFY `marker_train.rs:540` `batched_train_offspring` (the factory)**
- Rename the existing body to `batched_train_offspring_plain` (pub(crate), used by
  the parity tests).
- New body: if `NeuronMultiplicity::applies(rate, mode) && !conns.is_empty()`,
  call `MultiplicityBatch::train(...)`. Otherwise call `batched_train_offspring_plain(...)`.
  Genomes where `is_trivial()` holds go through plain as well, so they carry zero
  overhead.

**NEW `multiplicity_batch.rs` (crate root; `mod` in lib.rs): `MultiplicityBatch`**
- Stage 1 (G = 1, and G > 1 as a loop over single genomes): collapse, run
  `batched_train_offspring_plain` on the collapsed genome (this inherits the
  neuron-chunked over-budget route for free), then `GenomeExport::from_unique`.
- Stage 2 (super-genome batching, which keeps one Metal dispatch per batch):
  - Concatenate every genome's unique neurons per cluster into ONE super genome
    (num_genomes = 1, cluster order kept, per-(g, c) base offsets recorded) and run
    `train_batch_to_table` once.
  - Export each genome's unique layer from the shared table with
    `export_genome_from_table` (below), then `from_unique`.
  - On `BUDGET_ERR_PREFIX`, bisect the genome list. A single genome over budget
    falls to the Stage-1 single-genome route, which ends in the chunked trainer.
  - This also removes the "uniform `neurons_per_cluster` across the batch"
    constraint (train_batch_to_table), which collapsed genomes could not meet
    without padding.

**MODIFY `marker_train.rs:1816` `export_trained_batch` (Stage 2)**
- Extract the per-genome body into
  `fn export_genome_from_table(tb, bits, npc, conns_unpadded, cluster_slot_base: &[u32]) -> GenomeExport`.
- The plain loop passes `g*slots_per_genome + cluster_offset_in_genome[c]` as today.
  The super-genome path passes `cluster_offset_in_genome[c] + base[g][c]*cap[c]`.
- The output is byte-identical for plain; the existing multigroup / oi_z parity
  tests pin this.

**MODIFY `adaptive/eval_hybrid.rs:429` `try_eval_in_place_batch`**
- Add one gate: `if NeuronMultiplicity::applies(cfg.neuron_sample_rate, cfg.memory_mode) { return None; }`.
- The export path then runs the multiplicity layer. The collapsed genome usually
  fits the batch budget, so the chunked regime is rarer. The env var is default
  OFF anyway.
- A count-weighted vote probe (integer `votes += count*q`, exact) is deferred to
  Stage 5.

**MODIFY `pyapi/general.rs` + `lib.rs`**
- Add `#[pyfunction] neuron_multiplicity(bits_flat, neurons_flat, conns_flat) -> (Vec<u32>, Vec<u32>)`
  returning (unique_of, counts).
- Add `#[pyfunction] neuron_multiplicity_applies(rate: f32, memory_mode: u8) -> bool`.
- Register both. Bump `ABI_VERSION` 16 → 17 at lib.rs:330 with a doc entry:
  "additive; training path collapses duplicate neurons at rate ≥ 1.0 under
  order-independent writes; scores bit-identical to ABI 16".

**Unchanged by design (the point of the alias):**
- `compute_per_example_scores` (eval_export.rs:871)
- `evaluate_group_metal` / `evaluate_group_sparse_gpu` (adaptive/memory.rs)
- `core/metal_sparse.rs` + `core/shaders/sparse_forward.metal` (both kernels)
- dense `ramlm.metal`
- `metal_genome_eval.rs` (CE and batched-group kernels)
- `thresholds.rs`, `multiclass_metrics.rs`, every `ids_cache.rs` entry point, the
  k-fold merge, and `actual_neurons` divisors (logical counts, unchanged)

Kernels already take offsets/counts per neuron and binary-search inside
`[offset, offset+count)`. Nothing in the tree assumes disjoint or monotone ranges.
The only `counts.iter().sum()` uses are in tests (marker_train.rs:2389/2589/3175).

### 2.3 Data flow (rate 1.0, OI, QUAD)

```
Python IDSEvaluator.evaluate_batch (genome format unchanged)
  -> IDSCacheWrapper.evaluate_genomes_*  (flat bits/neurons/conns, rate=1.0)
  -> adaptive::evaluate_genomes_parallel_hybrid -> train_batch (routes unchanged)
  -> marker_train::batched_train_offspring            <- FACTORY: applies()?
       yes -> MultiplicityBatch: collapse each genome (ram_core NeuronMultiplicity)
              -> train ΣU unique neurons (super genome, one dispatch)
              -> export_genome_from_table (unique) -> GenomeExport::from_unique (logical, aliased)
       no  -> batched_train_offspring_plain (today's code, byte-identical)
  -> evaluate_genome_hybrid / compute_per_example_scores (UNCHANGED) -> metrics
```

### 2.4 Python

There is no Python object for a trained IDS layer: Python only orchestrates, and
Rust trains and scores. The honest Python counterpart is therefore the
multiplicity descriptor, built by Rust and never computed in Python.

**NEW `src/wnn/ram/core/NeuronMultiplicity.py`**: class `NeuronMultiplicity`
(frozen `@dataclass`, no base class).
- Fields: `unique_of: tuple[int, ...]`, `counts: tuple[int, ...]`.
- Properties: `num_unique`, `num_logical`, `distinct_fraction`.
- `@classmethod of(cls, genome: ClusterGenome) -> NeuronMultiplicity`: calls
  `require_accel().neuron_multiplicity(*flatten_genomes([genome]))`.
- `@staticmethod applies(neuron_sample_rate: float, memory_mode: int) -> bool`:
  calls `require_accel().neuron_multiplicity_applies(...)`. This is the single
  gate; Python never re-derives it.
- Export from `src/wnn/ram/core/__init__.py`.

**MODIFY `src/wnn/ram/architecture/ids_evaluator.py:92` (`__init__`)**: log one
line when `NeuronMultiplicity.applies(self._neuron_sample_rate, self._memory_mode)`
holds, for example `[MULTIPLICITY] active: duplicate neurons train once
(rate=1.0, OI)`. The worker sets the OI env at worker.py:615, before evaluators
are built at :740, so the gate reads the right env.

**MODIFY `src/wnn/accel.py:68`**: `EXPECTED_ABI = 17`.

**Factory / constructor selection**: in Rust, at the `GenomeExport` factory
(`batched_train_offspring`). Python passes `neuron_sample_rate` exactly as today
and does not branch.

**KNOWN_PARAMS impact: none.** No new flow param, no `.get()` read, and no Svelte
change. The full-stack parameter path is unchanged: `neuron_sample_rate` already
travels flow params → worker.py:776/808 → IDSEvaluator → IDSCacheWrapper →
`batched_train_offspring`. The only new hop is the log line. Per-genome
`eval_time_ms` drops, which is the intended visible effect.

**Exporter.** `fpga/scripts/export_genome.py:197` calls
`ram_accelerator.train_genome_export_fpga`, which **does not exist in the live
crate**. That is pre-existing breakage, flagged here and out of scope. The exporter
also trains at `neuron_sample_rate=1.0` (line 206) for every cohort. That is a
pre-existing train/deploy mismatch for 0.25 cohorts, also flagged.

When the exporter is repaired, it can emit unique BRAMs + `NeuronMultiplicity.counts`
so that the hardware stores each distinct neuron once and weights its vote by k in
the popcount/adder tree. That is an RTL change in `fpga/rtl`, about 3x less BRAM at
25-35% distinct. **It needs Luiz's decision** (hardware architecture). There is no
`mcu/` tree in the repo.

---

## 3. Duplicate collapse point and how multiplicity reaches every scoring path

**Collapse happens once per genome per training call**, inside
`batched_train_offspring`, before any table is allocated. It is a pure function of
the flat genome, so there is no caching or state.

**Choke point: one function, not N scorer edits.** Multiplicity reaches every
scorer through the data it receives, because the logical export lays out each
logical neuron's cells exactly as the plain layer would:

| Scoring path | Mechanism | Edit? |
|---|---|---|
| CPU sparse (eval_export.rs:871 fallback) | `SparseGpuExport::lookup` on aliased range | none |
| CPU dense | dense words copied per logical neuron | none |
| GPU sparse `sparse_forward.metal` (both kernels) | offsets/counts per logical neuron | none |
| GPU dense `ramlm.metal` (evaluate_group_metal) | words per logical neuron | none |
| Multiclass CE / batched groups (metal_genome_eval.rs) | same buffers | none |
| Coalesced groups (`actual_neurons` divisor) | logical groups rebuilt from the logical genome; divisor is the logical count = Σ multiplicities | none |
| Thresholds / score_examples / evaluate_at_thresholds / val (Protocol v2) | consume the scores above | none |
| K-fold (`evaluate_genomes_kfold_hybrid`) | per-fold training goes through the factory | none |
| Multiclass (`train_and_score_sets_flat`) | same | none |
| Adaptive / genesis (adaptive_eval.rs:897/1152) | factory; `read_cell_at` / fill rates read aliased identical data | none (parity-tested) |
| Eval-in-place (eval_hybrid.rs:429) | gated off when `applies` | 1 line |
| Streaming (ids_streaming.rs:110/124) | plain in Stage 1; Stage 3: streamer built on the collapsed genome, `seal_for_scoring` → `from_unique` | Stage 3 |
| Hierarchical S0/S1 (multistage.rs:4130/4418) | plain (correct, no savings) | deferred |
| PATH2 dense CPU fallbacks | plain (error path) | none |

No kernel needs a per-neuron weight buffer.

**Rejected: count-weighted kernels** (weight buffer + Σcounts divisor). That design
would touch the six scorer families above. Its exactness argument holds only for
dyadic weights: QUAD and BINARY sums of quarters are exact in f32 below 2^22. It is
NOT exact for QSR/PLN, whose coins are keyed on the neuron index, nor for TERNARY
with a non-dyadic `empty_value`. Supporting those modes would mean keeping a
logical loop alongside the weighted one, which is two ways. It also targets eval
time, and the measured 2.2x cost of SR1 is training. Revisit only if Stage 0
profiling shows eval above about 25% of SR1 wall time.

---

## 4. Exactness contract

1. **Selected only when `NeuronMultiplicity::applies(rate, mode)`.** That is
   `rate >= 1.0f32` (the same comparison as the shader) AND order-independent
   writes. Below 1.0 the plain function runs unchanged, with byte-identical exports
   (pinned by the existing golden/parity tests).
2. **At selection, for every logical neuron n**, the multiplicity export equals the
   plain export (`batched_train_offspring_plain` on the uncollapsed genome) in:
   - `(keys, values)[offsets[n]..+counts[n]]`
   - dense words, coverage bits, groups and connections

   Consequently every score from `compute_per_example_scores` (CPU, and GPU when
   present, with coverage_aware on and off) is bitwise equal (`f64::to_bits`), and
   so are every derived metric and threshold.
3. **No float-order caveat**: the scorers' summation order and operands are
   unchanged. If Stage 5 ever adds a count-weighted path, it is restricted to dyadic
   modes and pinned by a `to_bits` parity sweep.
4. **Residual risk**: a marker-slot overflow under concurrent duplicate-key claims
   would make even the plain layer order-dependent. Capacity is 2× the per-neuron
   upper bound (`marker_capacity_for_train`), so this does not happen at current
   sizes. The parity tests include the existing collision fixture
   (`oi_z_parity_with_collisions`, marker_train.rs:2466).

Pinning: Rust parity sweeps (section 5) compare `to_bits` of every score, mode ×
shape × route. The Python end-to-end test pins the full stack.

---

## 5. Tests

**cargo (ram_core).** No Metal needed.

```
cd src/wnn/ram/strategies/accelerator
PYO3_PYTHON=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \
  cargo test -p ram_core --lib --no-default-features neuron_multiplicity neighbor_search_golden
```

The `neuron_multiplicity` tests cover:
- the collapse is per-cluster scoped, bits-sensitive (same conns with different
  bits are distinct), keeps first occurrence in order, and `Σcounts == N`
- `is_trivial`
- `applies` is false for rate 0.25 and 0.99999994
- `applies` is false for QUAD with OI off, true for QUAD with OI on, and true for
  TERNARY/BINARY/PLN at 1.0 either way

`neighbor_search_golden` (existing) shows that the `push_distinct` refactor is
digest-neutral.

**cargo (worker).** Metal-gated; skips without a device or OI env, following the
existing pattern.

```
WNN_ORDER_INDEPENDENT_TRAIN=1 PYO3_PYTHON=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \
  cargo test -p ram_accelerator --lib --no-default-features multiplicity -- --test-threads=1
# and once WITHOUT the env var: lattice modes must still pass, OI modes must not be selected
```

- `from_unique_aliases_and_copies`: pure CPU, no Metal. A synthetic unique export
  plus multiplicity, checked for aliased sparse ranges, dense/coverage copies, and
  logical groups/connections that equal `reorganize_connections_for_gpu` output.
- `multiplicity_parity_{quad_weighted,quad_binary,qsr,ternary,binary,pln}`, next to
  `run_multigroup_parity` in marker_train.rs chunked_tests. Each compares plain vs
  multiplicity across these routes:
  - G = 1
  - G = 4 super-genome
  - forced bisection (a budget-parameterized inner, same trick as
    `train_single_genome_chunked_scored_with_budget`)
  - neuron-chunked (shrunken CHUNK budget)

  The fixtures cover single-cluster with heterogeneous bits, K = 3 multi-group, and
  planted duplicates (k = 1..5 copies, interleaved). The assertions:
  - (a) per-logical-neuron storage equality
  - (b) `compute_per_example_scores` `to_bits` equality on CPU (metal = None)
  - (c) the same on GPU sparse + dense evaluators
  - (d) coverage_aware on and off
  - (e) `export.multiplicity.is_some()` exactly when selected
- `multiplicity_never_below_rate_one`: at rate 0.25 the result is byte-identical
  to `_plain` and `multiplicity.is_none()`.
- `eval_in_place_gated_under_multiplicity`.
- Stage 2+: `adaptive_genesis_parity_under_multiplicity`, where the adapted genome
  and metrics must be identical.

**Python: NEW `tests/test_neuron_multiplicity.py`**
- `NeuronMultiplicity.of` on a hand genome gives the expected unique_of and counts;
  `applies(0.25, 2)` is False.
- **Permutation-twin end-to-end test, which needs no plain/multiplicity knob.** Two
  genomes run through `IDSEvaluator.evaluate_at_thresholds` on a tiny synthetic
  dataset (fixture pattern from `tests/test_ids_cache_builder.py`), QUAD_WEIGHTED,
  `monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")`, rate 1.0:
  - genome A has planted duplicates, so the multiplicity layer collapses it;
  - genome B is A with every duplicate copy's connection list REVERSED, so it has
    no exact duplicates and trains plain.

  Reversing the connections permutes the address bits consistently in training and
  scoring, so each copy computes the same function. eval/train/val scores and every
  metric must be exactly equal. Use neurons with bits ≥ 2 and no palindromic
  connection lists.
- Before deployment, run it in a scratch venv that has the built wheel installed
  (never in the live venv). Rerun it in the live venv after the swap.

---

## 6. Build order, deploy and effort

| Stage | Content | Effort |
|---|---|---|
| 0 | Measure, don't build: replay one SR1 batch offline with `WNN_TIMING=1` and `WNN_GPU_BATCHED_TRACE=1` (scratch copy, not the live worker) to get the train-vs-eval split and the distinct fraction per genome. This sets the expected gain. | 0.5 d |
| 1 | Smallest exact slice: ram_core `NeuronMultiplicity` + `training_is_order_independent` + shared `neuron_key`; `GenomeExport.multiplicity` + `from_unique`; factory branch in `batched_train_offspring` (G>1 handled as a per-genome loop); eval-in-place gate; PyO3 functions + ABI 17; Python class + log + ABI; all section 5 tests except the super-genome route. | 2 d |
| 2 | Super-genome batching: `MultiplicityBatch` packing/bisection + `export_genome_from_table` extraction. Restores one dispatch per batch and lifts the uniform-shape constraint. | 1 d |
| 3 | Streaming: `IDSGenomeStreamer` built on the collapsed genome, then `seal_for_scoring` → `from_unique`. | 0.5 d |
| 4 (opt.) | Cross-genome dedup within a batch: GA children share most neurons with their parents and elites. Requires a per-neuron slot index list instead of per-cluster bases in `export_genome_from_table`. Exactness is unchanged because every genome in one call trains on the same data and class weights. **Discuss first.** | 0.5 d |
| 5 (opt.) | Weighted eval-in-place probe; FPGA unique-BRAM + weighted adder tree. **Needs Luiz's decision.** | n/a |

Ship stages 1+2 in ONE worker wheel (one swap). Stage 3 can ride the next swap.

**Deploy (worker LIVE, controller LIVE; nothing is installed or edited in
/Users/lacg/wnn):**
1. In this worktree: `maturin build --release` to build only (no `develop`). Then
   run the `pre-deploy-wheel-review` skill and the cargo tests above.
2. Smoke ONE: a single SR1 genome replay against the built wheel in a scratch venv.
   Metrics must be identical to the recorded SR1 row, with lower `eval_time_ms`.
3. At worker idle, run `python scripts/detach_launch.py -- python scripts/worker_swap.py
   --auto-detect-running --install-wheel <ram_accelerator-…whl> --rayon-threads N`.
   The Python changes (accel.py ABI 17, NeuronMultiplicity.py, core `__init__`,
   ids_evaluator log) must be merged into the worker's tree at the SAME moment
   (stage the Python with the wheel). Otherwise the ABI assert refuses the start,
   which is the intended loud failure. After the swap, check the interrupted flow's
   status (known requeue race, WNN-2).
4. ram_core changes, but the **controller wheel is NOT rebuilt or installed.** The
   controller does not call the new code, and its existing binary keeps the
   ram_core it statically linked. The same applies to the public `weightless`
   wheel. Controller ABI is unchanged.

Expected gain (to be confirmed by Stage 0): training kernel work and marker-table
memory both scale with ΣU instead of ΣN, about 3x less at 25-35% distinct. The
16 GB batch budget fits about 3x more genomes, and the sparse export holds unique
keys only. Eval is unchanged.

---

## 7. Risks

1. **Exactness depends on per-neuron training independence.** Any future training
   coupling across neurons (a neuron-indexed RNG used at rate 1.0, per-neuron class
   weights) breaks it. Mitigation: the single `applies()` gate, a documented
   invariant at `should_skip_sample`, and parity sweeps in cargo.
2. **Legacy (non-OI) QUAD at rate 1.0 gets no speedup.** This is by design.
3. **Routing heuristics** (B11 affinity, eval_hybrid.rs `decide_batch_routing`)
   still read the logical neuron count. The effect is on performance only; retune
   after Stage 0/2.
4. **Footprint semantics.** `GenomeExport::materialized_cells` would count unique
   sparse keys but logical dense arrays. It is not wired to Python or the DB yet
   (data_layer.py:272 column). Decide whether it means physical or logical before
   wiring it. Luiz to decide.
5. **Identity-rule promotion** touches the IDS-26 crossover code. It must be
   digest-neutral (golden tests).
6. **Stale-base hazard.** This worktree was cut from a July `main`. Implementation
   must branch from the live branch (`05fe9cad` or later), or the line numbers here
   are wrong.

## 8. Rejected alternatives (one line each)

- Count-weighted scoring kernels: N scorer edits, not exact for QSR/PLN/TERNARY,
  and they speed up eval, which is not where the cost is (section 3).
- Python-side dedupe before calling Rust: violates no-Python-shortcuts and changes
  the genome Rust sees, so it is not transparent.
- Train once and copy memories: saves time but not memory (fails the requirement).
- New RAM neuron type / parallel layer type: forbidden (Hard Rule 1); it would be
  a second way.
- Subclassing Python `RAMClusterLayer`: the LM-era torch layer, not on the IDS
  path, so the subclass would be dead code. **If Luiz wants a Python layer class
  regardless, that needs discussion.**
- Extending the marker kernel to per-genome neuron counts: an edit to a validated
  hot shader. The super genome achieves the same with zero kernel changes.
- Padding collapsed genomes to a uniform batch shape: trains padding neurons and
  wastes table memory.
- Env-var or flow-param kill switch: a process global plus a KNOWN_PARAMS entry,
  for an exact transformation. Rollback is a wheel swap.
- Dedup of permutation-equivalent neurons (same connection set in a different
  order): it would require permuting memory keys. These are rare and outside the
  measured 25-35% figure.

## 9. Needs discussion (not decided here)

- Python "layer class": this design uses the `NeuronMultiplicity` descriptor
  because no Python IDS layer object exists. Confirm, or name the Python layer
  Luiz wants.
- Stage 4 cross-genome dedup.
- FPGA unique-BRAM + weighted adder tree, and the stale `train_genome_export_fpga`
  exporter.
- `materialized_cells` semantics (physical vs logical) before it is wired.
