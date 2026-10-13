# Multiplicity Layer (IDS + core): Design, revision 2

Status: DESIGN ONLY. Nothing is implemented.
Base: branch `controller-grid-orchestrator-unification` @ `05fe9cad` (worker ABI 16).
All file:line references point at that commit.
Revision 2 applies Luiz's review of revision 1:
- a Python layer extending `RAMClusterLayer`
- no one-off code
- QSR/PLN included
- cross-genome sharing as a planned stage
- logical + physical footprint, always reported
- corrected FPGA framing
- the FPGA exporter fix

---

## 0. The idea in one paragraph

A RAM layer has **logical neurons**: what the genome says, copies included, e.g.
500n × 34b. It also has **physical storage**: the memories actually trained and
held. Today the two are the same thing.

The multiplicity layer separates them with a single reusable map, a
`UnitGrouping`: logical neuron → physical unit, with the representative and the
count of each unit.

- Training writes each physical unit once.
- Scoring reads the logical layout. In each logical neuron's slot it finds a
  pointer to (sparse), or a copy of (dense), its unit's memory.
- Scorers, kernels, GA, genome format, evaluator API and results are unchanged.
- The same grouping also serves three other users: the GA crossover dedupe, the
  FPGA exporter (one logic instance per unit plus a count-weighted adder input),
  and footprint reporting (logical AND physical).

---

## 1. Exactness: when copies are identical, and why the design is exact

### 1.1 When it applies

At `neuron_sample_rate >= 1.0`, `should_skip_sample` (shaders/marker_train.metal:89;
CPU twin in `train_genome_in_slot`, adaptive/eval_export.rs:446) returns false
before it reads the neuron index. That skip is the only place training depends on
a neuron's position or on the RNG seed (checked in marker_train.metal and
marker_slots.metal).

Each neuron's slot region is written only by its own thread(s). So same-key
neurons, meaning same (cluster, bits, connections), receive identical writes.
They end byte-identical when the writes commute:

| Mode | Training write | Commutes? |
|---|---|---|
| QUAD_WEIGHTED, QUAD_BINARY, **QSR** | OI packed counters (`CellMode::uses_oi_counters`, core/cell_mode.rs:115) | yes, when OI is on (`order_independent_training_active`, core/neuron_memory.rs:1611); no on the legacy clamped nudge |
| TERNARY, **PLN** | TRUE-wins lattice | yes, always |
| BINARY | one-shot set | yes, always |

The single gate is:

```
applies(rate, mode) = rate >= 1.0f32 && training_is_order_independent(mode)
training_is_order_independent(mode) = !uses_oi_counters(mode) || order_independent_training_active(mode)
```

### 1.2 Why the design is exact, including QSR/PLN, with no special casing

The scoring export keeps the **logical layout**: the same groups, connections and
neuron indices as the plain layer. Each logical neuron's sparse range points at
its unit's storage; dense words are copied per logical neuron. Every scorer still
iterates logical neurons in the same order and reads the same cells.

The stochastic read modes key their coin on the **logical** neuron index:
- `qsr_key(run_seed, neuron_idx, address, example)` (core/neuron_memory.rs:244)
- `pln_coin`, same key, core/neuron_memory.rs:201
- the Metal twin in sparse_forward.metal:111 uses `neuron_idx = start_neuron + n`

So QSR and PLN draw the **same coins on the same cells**, and every score is
bitwise identical. Both modes get the full training and memory savings with no
mode-specific code. Their gates:
- QSR is selected under OI, which is how it is trained in production since the
  ABI-13 fix.
- PLN is selected always.

There is no float-order caveat, because no summation changes. The optional
count-weighted scoring stage (section 7) is analysed separately.

**Residual risk.** A marker-slot overflow under concurrent duplicate-key claims
would make even the plain layer order-dependent. Capacity is 2× the per-neuron
bound (`marker_capacity_for_train`), so it does not occur at current sizes. The
parity sweeps include the collision fixture (marker_train.rs:2466).

---

## 2. Reusable building blocks (no one-off code)

Revision 1's standalone `NeuronMultiplicity` descriptor is **dropped**. In its
place are two generic patterns and one key. Every consumer is listed.

### 2.1 `UnitGrouping`: group identical units

**NEW `ram_core/core/unit_grouping.rs`**, one struct, generic over the key.

```rust
/// Group identical units: unit i (logical order) -> group index (first-occurrence order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitGrouping
{
	index_of: Vec<u32>,        // logical unit -> group
	representatives: Vec<u32>, // group -> first logical unit with that key
	member_offsets: Vec<u32>,  // CSR over groups (len = groups + 1)
	members: Vec<u32>,         // logical units of each group, ascending
}

impl UnitGrouping
{
	pub fn identity(num_units: usize) -> Self;
	pub fn by_key<K: Eq + std::hash::Hash, I: IntoIterator<Item = K>>(keys: I) -> Self;
	pub fn num_units(&self) -> usize;
	pub fn num_groups(&self) -> usize;
	pub fn is_identity(&self) -> bool;
	pub fn index_of(&self, unit: usize) -> u32;
	pub fn is_representative(&self, unit: usize) -> bool;
	pub fn representatives(&self) -> &[u32];
	pub fn count(&self, group: usize) -> u32;
	pub fn members(&self, group: usize) -> &[u32];                 // the k logical positions
	pub fn gather<T: Clone>(&self, per_unit: &[T]) -> Vec<T>;     // logical -> group order
	pub fn expand<T: Clone>(&self, per_group: &[T]) -> Vec<T>;    // group -> logical order
}
```

**NEW `ram_core/core/neuron_key.rs`**: `NeuronKey { cluster: u32, bits: u32, connections: Vec<i64> }`
with `NeuronKey::of(cluster, bits, conns)` and `NeuronKey::genome_keys(bits_flat, npc, conns_flat) -> impl Iterator`.
This is the one definition of "the same neuron"; today it lives inline in
`push_distinct`.

Where `UnitGrouping` + `NeuronKey` are reused:

| # | Consumer | Use |
|---|---|---|
| 1 | GA crossover dedupe: `dedupe_fill` / `push_distinct` (core/neighbor_search.rs:1307/1328) | `representatives()` of each cluster's survivors. The `seen` set becomes the group keys. Digest-neutral (golden tests). |
| 2 | IDS Rust layer: `GenomeExport` construction (section 3) | collapse per genome |
| 3 | IDS cross-genome batch sharing (stage 5) | one grouping over the keys of all genomes in the batch |
| 4 | Python `MultiplicityRAMClusterLayer` (section 4) | via PyO3 `group_neurons(...)`; Python never re-derives it |
| 5 | Core sparse train/forward loops (`ram_core::sparse_memory`), used by the Python layer's sparse backend | the write/read seams in 2.2 |
| 6 | FPGA exporter + RTL generators (section 6) | emit one logic instance per group, adder input × `count` |
| 7 | Footprint reporting (section 5) | `num_units` = logical, `num_groups` = physical |
| 8 | Optional count-weighted scoring (section 7) | `members()` = the k logical positions for QSR/PLN coins |

### 2.2 Shared storage: logical view over physical storage

The pattern is the same everywhere. Physical storage holds `num_groups` units. The
logical view resolves unit i to `index_of(i)`. Writes go only through
representatives.

| Storage | Logical view | Where |
|---|---|---|
| Sorted sparse export (`SparseGpuExport`: keys/values/offsets/counts) | `SparseGpuExport::alias(&self, grouping) -> SparseGpuExport`: logical offsets/counts point into the shared keys/values (no key copy). The format is already CSR and every scorer binary-searches `[offset, offset+count)`; nothing assumes disjoint ranges (only test code sums counts: marker_train.rs:2389/2589/3175). | ram_core (after unification, below) |
| Dense packed words + coverage | row gather `grouping.expand(rows)` (≤ 1 KB/neuron at b ≤ 12) | `GenomeExport::from_physical` (worker) |
| `SparseLayerMemory` (DashMap) training/forward loops | write: `if !g.is_representative(n) { continue } mem.write(g.index_of(n), …)`; read: `mem.read(g.index_of(n), …)` | `ram_core::sparse_memory::{train_batch_sparse (:436), forward_batch_sparse (:618)}` + bitwise variant; identity grouping = today's loop |
| Python `Memory` (torch) | read: `raw_physical.index_select(1, index_of)`; write: representatives only | `MultiplicityRAMClusterLayer` (section 4) |

**Prerequisite unification (promote to base).** The tree has three
`SparseGpuExport` structs:
- `adaptive/memory.rs:504`, used by IDS
- `core/sparse_memory.rs:369`, used by core forward/public
- `core/neuron_memory.rs:318`, legacy

The IDS one moves onto the ram_core type. Fields are identical; `lookup(miss_default)`
and `memory_size` move over with it. That way `alias` exists exactly once, and the
public `weightless` wheel and the Python sparse backend get it for free.
`neuron_memory.rs:318` is left for a later cleanup; it has no IDS consumer.

---

## 3. IDS Rust layer: `GenomeExport` extended

`GenomeExport` (adaptive/eval_export.rs:13) is the IDS layer: what every scorer
consumes. It is extended, not duplicated.

- **Field** `pub grouping: UnitGrouping`. Identity for the plain layer. It replaces
  revision 1's `Option<NeuronMultiplicity>`, so plain and multiplicity are the same
  type with the same field.
  Literals to update (identity):
  - eval_export.rs:119 (`empty`)
  - eval_export.rs:840 (`export_genome_for_gpu`)
  - marker_train.rs:770 and :1989
  - adaptive/tests.rs:753
- **Constructor** `GenomeExport::from_physical(physical: GenomeExport, grouping: UnitGrouping, logical_bits, logical_npc, logical_conns) -> GenomeExport`.
  - Builds the logical groups and connections from the logical genome (same
    `build_groups` / `reorganize_connections_for_gpu` calls as the plain export).
  - Sparse groups go through `SparseGpuExport::alias`; dense groups go through the
    row gather.
  - Collapsing never removes a cluster's max-bits neuron, so per-cluster max bits
    are unchanged. Logical and physical groups therefore have the same `bits` and
    `words_per_neuron`, and dense rows copy verbatim.
  - Group composition can differ (coalescing buckets), so for multi-group genomes
    sparse ranges are gathered per logical group.
- **Footprint** `GenomeExport::footprint() -> Footprint` (section 5).
- **Factory = `batched_train_offspring`** (marker_train.rs:540). Every GPU-trained
  export is built here: GA batches (eval_hybrid.rs:944/1046/1172), the per-genome
  route (eval_hybrid.rs:195/270 → eval_single.rs:545), k-fold, thresholds,
  validation, multiclass (eval_single.rs:643/808/965/1154), adaptive
  (adaptive_eval.rs:897/1152) and the neuron-chunked over-budget route (:634).
  The new body:

```rust
if !(applies(neuron_sample_rate, memory_mode) && !connections.is_empty())
{
	return batched_train_offspring_plain(...);   // renamed existing body, byte-identical
}
MultiplicityBatch::train(...)                    // NEW multiplicity_batch.rs
```

- **`MultiplicityBatch`** (NEW `multiplicity_batch.rs`, crate root).
  - Stage 2: group each genome, train the physical genome through
    `batched_train_offspring_plain` (inheriting the chunked fallback), then
    `from_physical`.
  - Stage 4: the super-genome. Concatenate the physical units of all genomes per
    cluster into one dispatch (`train_batch_to_table`, 1 genome). This removes the
    uniform-shape constraint without padding or kernel edits. On `BUDGET_ERR_PREFIX`
    it bisects; a single genome over budget takes the chunked route.
  - Exports use `export_genome_from_table(tb, bits, npc, conns, slot_index: &[u32])`,
    extracted from `export_trained_batch` (:1816). The plain layer passes today's
    slots, so its output is byte-identical (multigroup/oi_z parity tests).
- **Unchanged scorers.** These need no edit:
  - `compute_per_example_scores` (eval_export.rs:871)
  - `evaluate_group_metal` / `evaluate_group_sparse_gpu` (adaptive/memory.rs)
  - core/metal_sparse.rs + core/shaders/sparse_forward.metal
  - ramlm.metal
  - metal_genome_eval.rs (CE + batched groups)
  - thresholds.rs, multiclass_metrics.rs, every ids_cache.rs entry point, the
    k-fold merge
  - `actual_neurons` divisors (logical = Σ counts)
- **Eval-in-place** (eval_hybrid.rs:429, env default OFF): gated off when
  `applies`. The export path then runs the multiplicity layer. Section 7 gives it a
  weighted integer probe.
- **Streaming** (`IDSGenomeStreamer`, ids_streaming.rs:110/124/240): built on the
  physical genome; `seal_for_scoring` → `from_physical`. Stage 8.
- **Hierarchical S0/S1** (multistage.rs:4130/4418) and the `PATH2_FALLBACK` dense
  CPU paths stay plain. They are correct, with no savings.

---

## 4. Python layer: `MultiplicityRAMClusterLayer(RAMClusterLayer)`

Luiz wants the Python layer regardless of current IDS use. It is designed to the
same contract.

### 4.1 Base seams (promote-to-base, behaviour-neutral) in `src/wnn/ram/core/RAMClusterLayer.py`

`RAMClusterLayer` today sizes `Memory` / `SparseMemory` to `total_neurons`
(:157-232). Every read goes through `self.memory.get_memories_for_bits` (:378, :393,
:472). Writes index `cluster*npc + arange(npc)` (train_pattern / train_batch /
train_multi_examples, :503-720). The Rust paths pass `memory_words` + `connections`
+ uniform layout (:423, :913, :963, :1017, :1095, :1211, :1271).

Add five small protected methods. Each is the identity in the base, and every
existing path is routed through them:

| Seam | Base | Used by |
|---|---|---|
| `_storage_neurons() -> int` | `total_neurons` | `__init__` sizing of `Memory` / `SparseMemory` |
| `_storage_connections(conns) -> Tensor` | `conns` | `__init__` |
| `_logical_cells(input_bits) -> Tensor` | `memory.get_memories_for_bits` | forward_counts, forward_quad_scores, forward_raw |
| `_storage_writes(logical_idx) -> Tensor` | `logical_idx` | train_pattern, train_batch, train_multi_examples(+_rust/_numpy), explore paths |
| `_unit_grouping() -> UnitGroupingView` | identity | Rust paths: `forward_rust` / `forward_metal` / `forward_hybrid` / `_forward_metal_cached` pass logical arrays (`_logical_memory_words()`, `_logical_connections()` = gather); `sparse_*` / LSH pass the grouping's `index_of` + representative mask to the Rust seams of 2.2 |

### 4.2 NEW `src/wnn/ram/core/MultiplicityRAMClusterLayer.py`

One class: `class MultiplicityRAMClusterLayer(RAMClusterLayer)`.

- **Constructor**: same explicit, typed signature as the base (no kwargs).
  - First computes `self._grouping = UnitGroupingView.of_layer(connections, num_clusters, neurons_per_cluster, bits_per_neuron)`,
    via the PyO3 `group_neurons`, which is ram_core `UnitGrouping` + `NeuronKey`.
  - Then calls `super().__init__`, which sizes storage through the seams to
    `num_groups` physical neurons.
- **Overrides only the 5 seams.**
  - Reads: `raw_physical.index_select(1, index_of)`.
  - Writes: `unique` representatives of the logical indices, in first-occurrence
    order. Each physical neuron is written once per example, exactly like each
    plain copy.
- **Connectivity changes** (`connections` setter :288, `set_cluster_connectivity`
  :739) regroup and reset storage. A connectivity change invalidates trained memory
  anyway; the plain layer keeps meaningless stale cells. Documented.
- **Footprint**: `logical_neurons` = `total_neurons`, `physical_neurons` =
  `num_groups`, plus logical/physical cells, the same `Footprint` shape as Rust.
- `get_config` / `from_config` / `save` / `load`: the config stores logical
  connections. Rebuild goes through the factory, so the grouping is recomputed and
  never serialized twice.
- Gating is unchanged; it acts on cluster scores.

**NEW `src/wnn/ram/core/UnitGroupingView.py`**: one class. A thin, immutable
holder of `index_of` / `counts` / `members` tensors returned by Rust. It is shared
with the FPGA exporter and with footprint reporting.

**NEW `src/wnn/ram/core/ClusterLayerConfig.py`** (typed dataclass: total_input_bits,
num_clusters, neurons_per_cluster, bits_per_neuron, connections, backend,
neuron_sample_rate = 1.0, rng, use_hashing, hash_size, gating_model) and
**NEW `src/wnn/ram/core/ClusterLayerFactory.py`**:

```python
class ClusterLayerFactory:
	@staticmethod
	def create(config: ClusterLayerConfig) -> RAMClusterLayer:
		# rate >= 1.0 AND the grouping collapses something -> multiplicity layer
		# rate < 1.0, random connections, or no duplicates -> plain RAMClusterLayer
```

Python cluster layers train every neuron on every example (rate 1.0 semantics),
and their writes are sequential per neuron. So the default config selects the
multiplicity layer whenever duplicates exist, and copies are provably identical.
The `neuron_sample_rate` field keeps the rule identical to Rust if Python training
ever subsamples.

Callers switch construction to the factory (one line each, transparent):
- `models/ramlm.py:162`
- `routing.py:111` and `routing.py:449`

Export both classes plus the factory and config from `src/wnn/ram/core/__init__.py`.

---

## 5. Footprint: LOGICAL and PHYSICAL, always both

```rust
// ram_core/core/footprint.rs
pub struct Footprint { pub logical_neurons: u64, pub physical_neurons: u64,
                       pub logical_cells: u64,   pub physical_cells: u64 }
```

- Logical cells: Σ over logical neurons. Sparse = the unit's key count per logical
  neuron; dense = 2^bits.
- Physical cells: Σ over physical units only. This is the storage actually trained
  and held, and what a deployment needs.
- `GenomeExport::materialized_cells()` (eval_export.rs:50) is **replaced** by
  `footprint()`, so there are not two primitives. `materialized_cells` callers are
  tests only today.

Bits stay per neuron as in the genome. Reports render as
`500n × 34b (logical) | 137n · 1.21 Mcells (physical)`. When plain, physical equals
logical, and both are still printed.

Full-stack path, hop by hop:
1. Rust `GenomeExport::footprint()`.
2. `evaluate_at_thresholds_ids_cached` / `evaluate_multiclass_at_thresholds_ids_cached`
   (ids_cache.rs) return `Footprint` as an extra tuple element. The streaming path
   uses `IDSGenomeStreamer::footprint()`.
3. pyapi `ids_cache_api.rs` (`evaluate_at_thresholds` / `evaluate_multiclass_at_thresholds`)
   and `ids_streamer_api.rs` return a dict.
4. `IDSEvaluator.evaluate_at_thresholds` / `evaluate_multiclass_at_thresholds` /
   streaming variants (ids_evaluator.py) return it.
5. `experiment.py:1470` gets it and passes it to
   `dashboard_client.create_validation_summary` (dashboard_client.py:676) as a typed
   `Footprint` param, plus the log line.
6. Dashboard API `create_validation_summary` (dashboard/src/api/mod.rs:81 →
   experiments.rs) upserts the genome's footprint.
7. `genomes` columns: reuse `materialized_cells` as **logical_cells** (it is NULL
   everywhere today; the comment is updated). ADD `physical_neurons`,
   `physical_cells`. `total_neurons` already is logical neurons.
   Files: dashboard/src/db/schema.rs:136-138, migrations.rs:86 (+2 ALTERs),
   data_layer.py:270-343 (mirror schema), models/mod.rs:376-382.
8. Frontend `lib/types.ts:234/259/438` and the genome stats views
   (routes/checkpoints/+page.svelte:195): show both, 1rem font.

`KNOWN_PARAMS`: no new flow param. `eval_time_ms` drops; that is the visible effect.

---

## 6. FPGA: corrected framing and the exporter fix (approved)

### 6.1 Framing

The deployed IDS designs are **combinational logic**: dense LUT ROM neurons
(`wnn_neuron_dense.sv`, `generate_rtl_dense.py`) feeding a `weighted_sum` adder
(generate_rtl_dense.py:104-123, generate_rtl.py:192-215). BRAM measured 0. The
large-bit sparse variants keep only search FSM logic on chip (`make_logic_only.py`).

At rate 1.0, identical copies are **identical logic**: the same ROM contents on
the same address wires. So the exporter emits one neuron instance per unit, and
the adder input is `count_u × result_u` (a constant shift-add, or the wire
replicated k times into the adder tree). The two are equivalent.

The LUT saving is roughly the neuron-logic share × (1 − U/N). Vivado may already
merge some identical logic, so **no saving is claimed until re-synthesis** (an EC2
batch, separate). The certain wins are smaller RTL and shorter synth runs. The
revision-1 "3x less BRAM" claim is withdrawn. At rate < 1.0 the grouping is
identity, and the output is identical to today.

### 6.2 Exporter fix

Defects in `fpga/scripts/export_genome.py`:
- It calls `ram_accelerator.train_genome_export_fpga` (:197), which is not in the
  live crate.
- It hard-codes `neuron_sample_rate=1.0` (:206) even for 0.25 cohorts. That is a
  train/deploy mismatch.
- It re-implements dataset loading (`load_dataset`, :23) instead of using the
  worker's evaluator construction.
- `export_genome_dense.py:20/116` imports the same broken function.

Fix (live training path, real run parameters, shared multiplicity layer):
1. **Shared evaluator construction.** Extract `worker._create_ids_evaluators`
   (worker.py:1257) and the OI env override (`_apply_oi_override`, :642) into
   `src/wnn/ram/experiments/IdsEvaluatorFactory.py` (one class, typed, no kwargs).
   The worker and the exporter both call it, and `load_dataset` in the exporter is
   deleted. Dataset/split/thermometer/feature selection, `neuron_sample_rate`
   (flow param, default 0.25 as the worker), `memory_mode`, OI, class weights and
   `run_seed` all come from the flow's real params.
2. **NEW PyO3** `IDSCacheWrapper.export_trained(bits, neurons, conns, empty_value, neuron_sample_rate, rng_seed) -> TrainedLayer`
   (pyapi/ids_cache_api.rs). It trains exactly like `evaluate_at_thresholds`
   (full_train, rng_seed 0, `train_single_via_marker` → the factory, so the
   multiplicity layer at rate ≥ 1.0) and returns:
   - grouping (`index_of`, `counts`, `members`)
   - PHYSICAL per-unit storage (sparse keys/values, or dense words)
   - connections per unit
   - `Footprint`
   - eval/val scores

   `IDSEvaluator.export_trained_layer(genome)` wraps it.
3. **Fidelity gate.** The exporter recomputes held-out metrics from the returned
   scores at the stored threshold modes and requires equality with the DB
   `validation_summaries` row for that genome/point. On mismatch it refuses to
   write files.
4. **Outputs.**
   - `metadata.json` gains `logical_neurons`, `physical_neurons`, `index_of`,
     `counts`, `logical_cells`, `physical_cells`, `neuron_sample_rate`,
     `memory_mode`.
   - Memory files are per **unit**.
   - `generate_rtl.py` / `generate_rtl_dense.py` instantiate one neuron per unit
     and weight the adder input by `counts[u]`. ACC_BITS stays sized from the
     logical N.
   - `export_test_vectors.py` + `run_sim.sh` give the HW-vs-Rust score parity
     check.
   - The "Z-7020 BRAM fits" summary (:368-372) is replaced by the footprint line.

---

## 7. Optional later stage: count-weighted scoring (saves scoring work)

Score over physical units: one address computation and one memory read per unit,
then a per-mode extension. It replaces the logical loop; the plain layer is the
identity grouping, so it is still one code path.

| Mode | Extension | Exact? |
|---|---|---|
| QUAD_WEIGHTED / QUAD_BINARY / BINARY | `sum += count × w(cell)` | **Yes, bitwise.** Weights are quarters, every partial sum is a multiple of 0.25 ≤ N, and that is exact in f32 for N < 2^22. Same divisor. |
| QSR / PLN | one read of the cell, then **k coins** keyed on the unit's original positions `members(u)`: `Σ coin(cell, qsr_key(seed, p_i, addr, ex))` | **Yes, bitwise.** Same keys and cells as the plain layer; coins are 0/1, so the integer sum is order-free. |
| TERNARY | integer `(count_true, count_empty)` accumulated with weight `count`, then `(ct + ev×ce)/N` | Exact vs the GPU sparse kernel, which already uses integer counts (sparse_forward.metal:130-139). The CPU path accumulates f32 today, so non-dyadic `empty_value` gives ULP-level differences until the CPU twin moves to counts. TERNARY is not used for new work. |
| Eval-in-place probe | `votes += count × quarter_value` | Yes (integer). |

Cost:
- Kernel + CPU twin edits in sparse_forward.metal (2 kernels), ramlm.metal (dense),
  metal_genome_eval.rs (CE + batched groups), `compute_per_example_scores`
  (CPU dense/sparse) and the marker probe.
- A members CSR buffer (4 B per logical neuron).
- Full CPU/GPU `to_bits` parity sweeps per mode.
- A rate-0.25 regression bench, because the plain layer now pays one indirection.

Gain: scoring lookups ∝ U instead of N (about 3x at 25-35% distinct). Coin hashing
stays k per unit, but it is cheap ALU work. Worth it only if Stage 0 shows scoring
is a material share of SR1 wall time.

---

## 8. Tests

**ram_core.** No Metal; `--no-default-features`.

```
cd src/wnn/ram/strategies/accelerator
PYO3_PYTHON=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \
  cargo test -p ram_core --lib --no-default-features unit_grouping neuron_key footprint sparse_alias neighbor_search_golden
```

- `unit_grouping`:
  - identity
  - first-occurrence order
  - CSR members
  - gather/expand round trip
- `neuron_key`: cluster- and bits-sensitive.
- `sparse_alias`: aliased lookup equals per-neuron lookup.
- `train_batch_sparse` / `forward_batch_sparse`: identity grouping is byte-equal to
  the old loops.
- Golden digests unchanged after the `push_distinct` refactor.

**Worker.** Metal-gated, following the existing skip pattern.

```
WNN_ORDER_INDEPENDENT_TRAIN=1 PYO3_PYTHON=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \
  cargo test -p ram_accelerator --lib --no-default-features multiplicity -- --test-threads=1
# again WITHOUT the env: lattice modes still pass; OI modes must not be selected
```

- `from_physical_aliases_and_copies` (CPU only).
- `multiplicity_parity_{quad_weighted,quad_binary,qsr,ternary,binary,pln}`.
  - Routes: G = 1 / super-genome G = 4 / bisection / neuron-chunked /
    cross-genome sharing.
  - Fixtures: single-cluster with heterogeneous bits, K = 3 multi-group, planted
    duplicates (k = 1..5, interleaved).
  - Assertions:
    - per-logical-neuron storage equal
    - `compute_per_example_scores` `to_bits` equal on CPU and GPU, coverage-aware
      on and off
    - `grouping.is_identity()` exactly when not selected
    - footprint logical == plain, physical == Σ unit storage
- `never_below_rate_one` (byte-identical to `_plain`).
- `eval_in_place_gated`.
- `adaptive_genesis_parity`.
- Controller crate must still pass `cargo test -p ram_controller --lib --no-default-features`
  (94 tests). The ram_core seams are identity-neutral.

**Python.**

- `tests/test_multiplicity_ram_cluster_layer.py`.
  - Setup: plain `RAMClusterLayer` vs `MultiplicityRAMClusterLayer` with identical
    connections and planted duplicates, across DENSE / SPARSE / LSH backends, with
    and without gating.
  - Identical training sequences through train_pattern / train_batch /
    train_multi_examples / `_rust_numpy` / train_sparse.
  - `torch.equal` on forward, forward_quad_scores, forward_counts, forward_raw,
    forward_rust, forward_metal, forward_sparse.
  - Physical neurons == groups.
  - Factory selection: rate 1.0 with duplicates → subclass; 0.25 or no duplicates →
    base.
  - Connectivity setter regroups.
- `tests/test_ids_multiplicity_e2e.py` (permutation-twin, no knob).
  - Genome A has planted duplicates and collapses. Genome B is A with each copy's
    connection list reversed: equivalent function, no exact duplicates, so it
    trains plain.
  - Through `IDSEvaluator.evaluate_at_thresholds` at rate 1.0, QUAD_WEIGHTED with
    OI, and again with QSR, the scores and metrics must be exactly equal. Footprint
    logical must be equal, physical(A) < physical(B).
  - Not valid for QSR/PLN coin-key equality across A vs B, because their keys
    include the address. For those modes the Rust parity sweep is the pin, and the
    e2e run uses QUAD only.
- `tests/test_fpga_export_fidelity.py`: export a small cohort genome at its real
  rate (0.25 and 1.0); the fidelity gate passes; at 1.0 `physical_neurons` <
  `logical_neurons` and RTL sim scores match Rust.

Python tests run in a scratch venv with the built wheel before the swap, never in
the live venv. Rerun them in the live venv after the swap.

---

## 9. Staged order and effort

| Stage | Content | Effort |
|---|---|---|
| 0 | **Measure** (offline replay, not the live worker): SR1 train-vs-score split (`WNN_TIMING`), per-genome distinct fraction, and **batch-level** distinct fraction (sizes stage 5) | 0.5 d |
| 1 | ram_core: `UnitGrouping`, `NeuronKey`, `Footprint`, `SparseGpuExport` unification + `alias`, identity-grouping seams in `sparse_memory` train/forward, `push_distinct` refactor, `training_is_order_independent` | 1.5 d |
| 2 | IDS Rust layer (smallest exact slice): `GenomeExport.grouping` + `from_physical` + `footprint`; factory branch (G > 1 as a per-genome loop); eval-in-place gate; tests | 1.5 d |
| 3 | Footprint reporting full stack (Rust → pyapi → IDSEvaluator → experiment.py → dashboard API/schema/migrations/models → Svelte) | 1 d |
| 4 | Super-genome batching (`MultiplicityBatch` + `export_genome_from_table` + bisection) | 1 d |
| 5 | **Cross-genome sharing within a batch**: one grouping over all genomes' keys (every genome in one call trains on the same data, class weights and mode, so it stays exact). Training work and table memory ∝ the batch union. Per-genome exports get their units' ranges via `slot_index`. Cross-genome sharing of *scoring* storage (Arc'd exports) is a follow-up only if stage 0 shows export memory matters. | 1.5 d |
| 6 | Python: base seams, `MultiplicityRAMClusterLayer`, `UnitGroupingView`, `ClusterLayerConfig`, `ClusterLayerFactory`, caller switch, PyO3 `group_neurons`, parity tests | 2 d |
| 7 | FPGA exporter fix: `IdsEvaluatorFactory` extraction, `export_trained`, fidelity gate, per-unit emission, RTL generator counts, test (re-synth on EC2 scheduled separately) | 1.5 d |
| 8 | Streaming (`IDSGenomeStreamer` on physical genome) | 0.5 d |
| 9 (opt.) | Count-weighted scoring (section 7), including QSR/PLN k-coin and TERNARY integer-count extensions | 3.5 d |

Required stages 0-8: about **11 d**. With stage 9: about **14.5 d**.

ABI: worker 16 → **17**. `evaluate_at_thresholds` now returns `Footprint` (a
breaking tuple change) and there are additive PyO3 functions
(`group_neurons`, `export_trained`). `EXPECTED_ABI` changes at `src/wnn/accel.py:68`.

**Deploy (worker and controller LIVE; nothing is installed or edited in
/Users/lacg/wnn).**

1. Bundle stages 1-5 plus the Rust part of 6/7 into **one worker wheel / one swap**.
   - `maturin build --release` in the worktree (build only).
   - Run the `pre-deploy-wheel-review` skill.
   - Run all cargo tests.
   - Smoke one SR1 genome in a scratch venv: metrics identical to the recorded row,
     lower `eval_time_ms`, footprint logical = genome, physical < logical.
2. At worker idle, run
   `python scripts/detach_launch.py -- python scripts/worker_swap.py --auto-detect-running --install-wheel <whl> --rayon-threads N`.
   - The Python changes (accel ABI 17, the ids_evaluator / experiment.py footprint
     plumbing, `IdsEvaluatorFactory` extraction, core classes) merge into the
     worker's tree **at the same moment**.
   - The dashboard binary is rebuilt in its CARGO_TARGET_DIR with the additive
     migrations, and restarted per the dashboard procedure.
   - Check the interrupted flow's status after the swap (WNN-2 requeue race).
3. ram_core changes, but the **controller and public `weightless` wheels are not
   rebuilt or installed now**. Their binaries keep the ram_core they statically
   linked. Their cargo tests must pass so the next scheduled rebuild is neutral.
4. Stages 8 and 9 ride later swaps.

---

## 10. Risks

1. **Exactness rests on per-neuron training independence at rate 1.0.** One gate
   (`applies`), a documented invariant at `should_skip_sample`, and cargo parity
   sweeps guard it.
2. **Legacy QUAD/QSR without OI never selects the layer.** By design: their copies
   differ even in the plain layer.
3. **The `SparseGpuExport` unification touches the public `weightless` and core
   forward paths.** Mitigation: field-identical move, run all three crates' tests.
4. **The worker evaluator-factory extraction edits live worker code.** It lands
   only at the swap, never edited under a running worker.
5. **Routing heuristics** (B11 affinity, `decide_batch_routing`) still read the
   logical N. Performance only; retune after stage 0/4.
6. **`evaluate_at_thresholds` tuple change.** Every caller must unpack the footprint:
   `experiment.py:1470` and the streaming path. The ABI bump makes a stale tree
   fail loudly.
7. **The Python layer adds `index_select` per forward** when duplicates exist. The
   factory returns the base class when the grouping is trivial.

## 11. Rejected alternatives (one line each)

- Standalone `NeuronMultiplicity` descriptor: one-off; replaced by generic
  `UnitGrouping`.
- Count-weighted kernels as the *first* step: many scorer edits for a scoring-time
  win; kept as the optional stage 9, now with exact QSR/PLN extensions.
- Python-side dedupe before Rust: violates no-Python-shortcuts and is not
  transparent.
- Train once and copy memories: saves time but not memory.
- New RAM neuron type / parallel layer type: Hard Rule 1; it would be a second way.
- Per-genome neuron counts in the marker kernel: a hot-shader edit; the
  super-genome achieves it with none.
- Padding collapsed genomes to a uniform batch shape: trains padding, wastes table
  memory.
- Kill-switch env var or flow param: a global plus a KNOWN_PARAMS entry for an exact
  transformation; rollback is a wheel swap.
- Dedup of permutation-equivalent neurons: would need key permutation; rare and
  outside the measured 25-35%.
- Claiming a BRAM or LUT saving before re-synthesis: the designs are combinational
  (BRAM 0), and synthesis may already merge identical logic.

## 12. Still open (for Luiz)

- Stage 9 go/no-go after the stage 0 measurement.
- Whether cross-genome sharing should extend to scoring storage (Arc'd exports)
  after stage 5's measurement.
- The re-synthesis batch (EC2) for unit-emitted RTL, to quantify LUT savings.

---

## Stage 0 — MEASURED (12/10/2026, worker idle)

**Flow-level split (worker log, MARKER_TRAIN_BATCHED = GPU training kernel time):**

| cohort | runs | total | GPU train kernel | everything else |
|---|---|---|---|---|
| GAF (rate 0.25) | 11 | 23.1 min | 8.8 min | 14.3 min |
| SR1 (rate 1.0) | 5 | 56.8 min | 14.9 min | 41.9 min |

**Same genomes, k-fold `evaluate_batch` (the GA hot path), WNN_TIMING=1** (4 GA genomes of SR1 r20419, 482 neurons, 24.7% distinct):

| rate | hybrid batch_size | "train" (train + sparse build/export) per genome | eval per genome | sparse keys per genome | wall |
|---|---|---|---|---|---|
| 0.25 | 10 | ~650-670 ms | ~12 ms | ~310k | 13.6 s |
| 1.0 | **2** | ~940-1190 ms | ~12 ms | ~420k | 24.0 s |

Readings:
1. **Scoring is negligible** (~12 ms/genome at both rates). Count-weighted scoring (stage 9) has no time payoff → recommend NOT doing it.
2. The rate-1.0 cost is the CPU-side train/sparse-build/export + **lost parallelism**: rate 1.0 writes ~35% more sparse keys per genome and the hybrid batch shrinks 10 → 2.
3. Stage 2 (train + store unique neurons only) cuts keys per genome ~4x (24.7% distinct) — below the rate-0.25 footprint — which should restore the batch size. This is exactly the cost the design targets.
4. Cross-genome sharing (indicative only — full populations are not checkpointed): 131 distinct of 1929 neurons across 4 related GA genomes = 6.8%. Stage 1's UnitGrouping must log the per-batch distinct fraction so stage 5 is sized on real batches.

---

## Stages 3-5 — IMPLEMENTED (12/10/2026, worktree branch, not deployed)

- **Stage 3 (footprint, worker ABI 17).** `ram_core::footprint::Footprint`;
  `GenomeExport::footprint()` replaces `materialized_cells`. Threaded through
  `evaluate_at_thresholds` (5th tuple element) / `evaluate_multiclass_at_thresholds`
  (3rd) / `IDSGenomeStreamer.footprint()` → `wnn.ram.footprint.Footprint` →
  `IDSEvaluator` → `experiment.py` (log line + summary) → dashboard.
  **Deviation (§5 step 7):** stored on `validation_summaries`
  (`logical_neurons, physical_neurons, logical_cells, physical_cells`), not on
  `genomes` — a `genomes` row exists only for leaderboard submissions, written
  after the summary and only at `final`, so the upsert would miss most rows.
  The validation-cache hit carries the footprint back.
- **Stage 4 (super-genome).** `ram_core::super_genome::SuperGenome`: batch
  neurons walked cluster-major, collapsed, trained as ONE genome through
  `batched_train_core`; budget overflow bisects; a single genome takes the plain
  trainer (chunked fallback). `GenomeExport::from_physical` reads
  `PhysicalUnits`; a unit may sit in a wider physical group than its logical
  group (dense slots re-pack / densify — exact). No `export_genome_from_table`
  extraction was needed: the logical export is rebuilt from the super-genome's
  ordinary export.
- **Stage 5 (cross-genome sharing).** The super-genome key is `NeuronKey` alone,
  so neurons shared by genomes of one call train once. Each genome keeps its own
  grouping for the footprint.
- **Known pre-existing defect:** with `WNN_COALESCE_GROUPS=1` the PLAIN
  neuron-chunked trainer mis-places heterogeneous-bits neurons
  (`chunked_tests::neuron_chunked_matches_unchunked` fails). The parity sweep
  skips only its chunked comparison under coalescing.

---

## Batch sizing + stages 6-8 — IMPLEMENTED (12-13/10/2026, worktree branch, not deployed)

### A. Hybrid batches sized and routed by the TRAINED footprint — MEASURED

`budget_batch_size` (adaptive/eval_export.rs) sizes every genome by
`trained_neurons_per_cluster` (multiplicity_batch.rs): its distinct neurons
when the factory will train it through the multiplicity layer, its logical
neurons otherwise; the most restrictive genome wins (adaptive_eval.rs used to
size off genome 0 only — same OOM class as the old 96b×250n bug; its seeds are
per genome_idx, so this is result-neutral there). `decide_batch_routing`:
under the multiplicity layer a batch is ONE dispatch (`one_dispatch`) — no B14
per-shape groups, no B12 hybrid split — so the super-genome shares units
across the whole batch. `MultiplicityBatch::plain_is_exact` sends an identity
batch of MIXED shapes to the super-genome (the plain kernel errors on them).

SR1-unswt-CTRL-r20419, 50 genomes (the 4 checkpointed GA genomes + 46
children from the GA's own crossover/NEURONS mutation, seeded; 444-500
neurons, ~23.5% distinct per genome, ~5.4% distinct per 5-genome batch),
k-fold `evaluate_batch` (5 folds), WNN_ORDER_INDEPENDENT_TRAIN=1, WNN_TIMING=1,
live IDS flow co-tenant (batch sizes are memory-dependent; all runs measured
back to back under the same load):

| wheel | rate | batch_size | train ms/genome/fold | eval ms/genome/fold | sparse keys/genome | wall ms/genome (5 folds) | metrics vs plain wheel |
|---|---|---|---|---|---|---|---|
| plain (live ABI 16) | 0.25 | 5 | ~235 | ~19 | ~278k | 1385 | — |
| plain (live ABI 16) | 1.0 | 1 | ~488 | ~22 | ~386k | 2553 | — |
| stage 5 (logical sizing) | 1.0 | 1 | ~158 | ~18 | ~86k | 892 | bit-identical (50/50) |
| **A (physical sizing + one dispatch)** | 1.0 | **5-6** | ~126 | ~24 | ~86k | **736** | bit-identical (50/50) |
| A | 0.25 | 5 | ~207 | ~18 | ~278k | 1137 | bit-identical (50/50) |
| HEAD (A + 6-8 + seed fix), worktree Python | 1.0 | 5 | ~86 | ~19 | ~86k | **529** | bit-identical (50/50) |
| HEAD, worktree Python | 0.25 | 3-6 (varies with free RAM per fold) | ~200 | ~21 | ~278k | 1113 | 45/50 differ — the seed fix below, by design |

(The plain / stage-5 / A rows ran the main tree's Python, read-only, against
each wheel; the k-fold path is Rust-orchestrated, the HEAD rows use the
worktree's Python.)

Rate 1.0 is now 3.5x faster than the plain wheel at rate 1.0 and 1.9x faster
than the plain wheel at rate 0.25.

### Bugs found and fixed on the way (each with a regression test)

- **Option B / B12 hybrid trained RANDOM connections** whenever a same-shape
  batch's connection totals differed (`&[]` = kernel-random). Now the
  contiguous genome-major slice (`HybridBatchConfig::batch_connections`).
- **`Memory.explore_batch` lost writes**: every written word was rebuilt from
  the pre-batch word, so two cells of one 31-cell word in one batch kept only
  the last (tests/test_memory_explore_batch.py).
- **`RAMClusterLayer.reset_memory` left the SPARSE / LSH stores trained.**
- **Dashboard validation cache drifted metrics by an ULP**: serde_json's
  default float parse is not correctly rounded, and a cache hit re-stores the
  served metadata as the final row (196/18500 recorded as ...596 instead of
  ...595). `float_roundtrip` enabled; dashboard test
  `cached_threshold_metadata_floats_round_trip_exactly`.
- **RESULTS-CHANGING at rate < 1.0 (own commit cb924ce0, droppable):** a
  genome's sampling mask was seeded from its batch position (rng_seed +
  batch_start / + genome_idx) and the batch size comes from free memory, so
  GA fitness moved with the machine's RAM (measured: batch 3-6 across the 5
  folds of one call). Every eval_hybrid route now trains with the call's seed.
  Test `sampled_results_do_not_depend_on_batch_size`.
- **Streamer under OI committed the vote counters per chunk**: later chunks
  overwrote earlier evidence; streamed != in-memory (CE 0.305 vs 0.139 in
  test_streaming_evaluator_evaluate_batch_full under OI). Counters now span
  all chunks. Test `streamer_chunking_is_invisible`. Changes production
  streaming results under OI.
- The FPGA exporter leaked WNN_ORDER_INDEPENDENT_TRAIN into the process.
- FPGA scripts put `src/wnn` (not `src`) on sys.path, so they imported
  whatever `wnn` the venv's editable install pointed at.
- **The RTL adder summed raw cell CODES (0..3)**, not the scorer's weights
  (QUAD 0, 0.25, 0.75, 1 = quarters 0,1,3,4): the deployed classifier ranked
  examples differently from the validated model. Now a per-mode LUT from
  the accelerator.

### Stage 6 — Python layer

As designed (§4): five identity seams on `RAMClusterLayer` (`_storage_neurons`,
`_storage_connections`, `_logical_columns` — generalises `_logical_cells` to
addresses too —, `_storage_writes`, `_unit_grouping` returning None for the
identity), `UnitGroupingView`, `MultiplicityRAMClusterLayer`,
`ClusterLayerConfig`, `ClusterLayerFactory`, callers switched (ramlm.py,
routing.py router + experts; the routed expert cache reads logical rows).
Sparse backend: ram_core `train_batch_sparse` / `forward_batch_sparse` take
`Option<&UnitGrouping>`; PyO3 `group_neurons`, `layer_footprint`, optional
`index_of` on the sparse calls. **Deviation:** write return counts on the
sparse/LSH backends are storage-level (copies are not written).

### Stage 7 — FPGA exporter

`IdsEvaluatorSettings` + `IdsEvaluatorFactory` (src/wnn/ram/experiments/) are
the one evaluator construction; the worker delegates to them. Rust
`train_genome_export` (shared with `train_and_score_eval_and_train`, so export
training IS validation training), `GenomeExport::unit_tables`, `TrainedLayer`,
`IDSCacheWrapper.export_trained`, Python `TrainedLayerExport`. fpga/scripts:
`FlowGenomeSource` (DB read-only), `FlowLayerTrainer`, `FidelityGate`
(every recorded threshold mode reproduced; MAX_ULPS=4 only for the legacy
serde drift — a real model difference moves a metric by >= 1/n), per-unit
sparse/dense files, `rtl_adder` (count x quarters LUT, ACC_BITS from logical
N), generated testbench + parametric run_sim.sh (expected = Rust score x 4N).
Python reimplementations deleted (`train_and_extract_sparse_py`,
`python_classify`, `load_dataset`). Measured on real rows: SR1 r20419
best_f1 (rate 1.0) 462n -> 115 units, 381,118 -> 91,219 cells, 7/7 modes
reproduced; GAF-CTRL r20423 best_f1 (rate 0.25) 489 -> 489, 7/7 reproduced;
the RTL adder model equals Rust score x 4N on 400 held-out vectors for both.
Stochastic modes (QSR/PLN) refuse RTL generation (no combinational adder
reproduces a per-example coin). Re-synthesis (LUT counts) still EC2-pending.

### Stage 8 — streaming

`IDSGenomeStreamer::new` trains the physical genome when
`multiplicity_applies` (CPU slot trainer: copies receive identical writes);
`seal_for_scoring` rebuilds the logical export with `from_physical`, so
`footprint()` reports physical < logical. Test: the streamer with planted
duplicates scores bit-identically to its rotated-connection twin.
