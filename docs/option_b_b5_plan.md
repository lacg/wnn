# Option B B5 — Multi-cluster support plan

**Status:** PLANNING — awaiting user decision on negative-sampling policy

**Scope:** Generalize the marker-FSM Metal kernel from single-cluster (binary IDS) to K-class. Unlocks multi-class IDS (CIC-IoT-2023 8-class), per-class ensembles, and eventually language-model output clustering.

## Why this matters

Currently `batched_train_offspring` errors out if `num_clusters != 1`. That blocks Option B from being used for:
- Multi-class IDS workflows
- LM output clustering (the project's original architecture)
- Any per-class ensemble strategy

After B5, Option B becomes generally applicable across the codebase.

## Architectural baseline

**Current single-cluster path** (B4b, commit `e5724eaa`):
- `params.single_cluster = 1` → kernel uses `nudge_true = (target == 1)` (binary IDS)
- One marker hashtable region per (genome, neuron); slot_offset = global flat index
- All neurons see all examples; nudge direction is determined by binary target

**Generalization for K clusters:**
- Neurons partition into K groups within a genome (`neurons_per_cluster[c]` neurons in cluster c)
- For example with `target = c`:
  - Neurons in cluster c nudge based on a positive-pattern rule
  - Neurons in other clusters nudge based on a negative-pattern rule
- The exact rules depend on the negative-sampling policy (decision below)

## Phase breakdown

### B5a — NeuronTrainMeta extension (~1-2h)
Add `cluster_idx: u32` to `NeuronTrainMeta`. Update `batched_train_offspring` to set it correctly: iterate `neurons_per_cluster[c]` and assign incrementing cluster_idx per neuron range.

### B5b — Kernel update (~1-2h) — DECISION NEEDED
See "Negative-sampling policy" section below. Once chosen, update `marker_train.metal` accordingly.

### B5c — CPU parity reference (~2-3h)
Implement (or reuse if existing) a multi-cluster CPU trainer that mirrors the GPU path exactly. Likely lives alongside the existing `train_genome_in_slot` for binary IDS.

### B5d — Parity test (~1h)
Mirror `run_marker_train_batched_parity_test` for K∈{2, 8}, ng∈{4, 16}, n=100, e=10K. Expect exact match.

### B5e — Wire into `evaluate_genomes_parallel_hybrid` (~30min)
Remove the `num_clusters == 1` guard at adaptive.rs:4075. Replace with "uniform K across batch" check (per-genome num_clusters must match, since the kernel dispatches a single shape).

### B5f — Real-data flow test (~1-2h)
Run a multi-class flow (e.g., CIC-IoT-2023 8-class) with WNN_OPTION_B=1 vs baseline. Verify CE/Acc parity. Compare wall time.

### B5g — Per-cluster capacity sizing (~1h)
Currently capacity is uniform per neuron. For multi-class with imbalanced data, minority-class neurons need less capacity, majority-class needs more. Either:
- Use `num_train` worst-case for all (current, wasteful)
- Use per-cluster `num_examples_in_class` for adaptive sizing

## Negative-sampling policy — DECISION NEEDED

### Approach 1: "All clusters nudge per example"
Every neuron processes every example. Cluster `target` nudges TRUE; others nudge FALSE.

```metal
bool nudge_true = (target == long(cluster_idx));
// no skipping; nudge always called
```

**Pros:** simple kernel; deterministic; exact mirror of naive multi-class trainers.
**Cons:** minority-class neurons see K-1 FALSE nudges per example — saturates toward FALSE; majority sees K-1 unrelated examples.
**Performance:** every neuron does full e×K work.

### Approach 2: "Per-example negative-sampling"
Each neuron processes examples where it's either the positive class OR explicitly selected as a negative for this example.

```metal
bool is_positive = (target == long(cluster_idx));
bool is_negative = lookup_in_negatives_for(cluster_idx, example_idx);
if (!is_positive && !is_negative) return;
bool nudge_true = is_positive;
```

**Pros:** balanced per-cluster nudges; ~K× less work; matches `train_negatives` API that's already in the kernel signature but unused for single-cluster.
**Cons:** requires per-cluster negative-index lookup; needs careful flat-array encoding of `train_negatives[K][num_negatives_per_class]`.
**Performance:** each neuron does ~e × (1 + neg_ratio) work — much faster for K > 2.

### Approach 3 (hybrid): "All-cluster nudge with weight"
Every neuron sees every example, but class_weights are used to dampen the FALSE nudges for non-positive neurons.

**Pros:** doesn't require negative-sampling API; supports class-rebalancing already exposed via `class_weights` in TrainParams.
**Cons:** still O(e×K) work per neuron; deviates slightly from binary path's semantics.

## Recommendation — UPDATED after reading existing CPU code

**Approach 2 is correct** — it matches the existing CPU path at `adaptive.rs:2837-2940`. The CPU trainer:
1. For each example: nudge TRUE for target cluster's neurons
2. For each `train_negatives[ex_idx][k]`: nudge FALSE for that negative cluster's neurons
3. All other clusters: skip

GPU kernel must mirror this exactly for parity. The unused `train_negatives` buffer in the current kernel signature has been waiting for this — B5 activates a dormant path rather than designing new semantics.

Approach 1 was based on a misread; ignore it.

## Files that will change

- `src/wnn/ram/strategies/accelerator/marker_train.rs` — NeuronTrainMeta + batched_train_offspring multi-cluster path
- `src/wnn/ram/strategies/accelerator/shaders/marker_train.metal` — kernel cluster logic
- `src/wnn/ram/strategies/accelerator/adaptive.rs` — remove num_clusters==1 guard
- `src/wnn/ram/strategies/accelerator/lib.rs` — add multi-cluster parity test entry point
- `tests/option_b_multicluster_parity.py` (new) — Python-driven parity validation

## Open questions

1. **Negative-sampling policy** (the big one, above)
2. **Heterogeneous neuron counts per cluster across genomes** — uniform-K is easy, uniform-K-with-different-neurons-per-cluster needs more careful layout
3. **Per-cluster slot_capacity** — adaptive sizing (B5g) is a nice-to-have; uniform sizing is fine for V1
4. **Whether to land B6 (export_per_neuron parallelization) before or after B5** — both are independent; B6 has bigger immediate wall-time impact

## Why this isn't urgent

The current production cohort (TOP20 250n×100b, the 46M small-genome sweep, and your current best results including exp 8654) is all **single-cluster binary IDS**. Single-cluster Option B (after today's fix) handles this case fully. Multi-cluster is a forward-compatibility feature, not unblocking current research.
