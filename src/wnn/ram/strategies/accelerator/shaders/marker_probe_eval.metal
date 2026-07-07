// =============================================================================
// marker_probe_eval.metal — eval-in-place probe kernel (07/07/2026)
// =============================================================================
//
// Scores examples DIRECTLY against a trained MarkerHashTable while it is
// still resident on the GPU, skipping the sorted sparse export for
// fitness-only evaluations (the over-budget neuron-chunked regime: 46M-flow
// 250n × 48-64b genomes).
//
// Compiled with core/shaders/common.metal + core/shaders/marker_slots.metal
// prepended (same include pattern as marker_train.metal) — slot_hash,
// MARKER_* constants and the OI_* counter constants come from there.
//
// Per (eval_example, neuron-in-chunk):
//   1. Compute the address with wnn_compute_address_u64 — the SAME helper and
//      connections layout the training kernel used (NeuronTrainMeta.conn_offset
//      / .bits are reused verbatim; single-genome dispatch → genome conn base 0).
//   2. Probe the hash table: slot_hash + linear probe. Do NOT stop at the
//      first key match — duplicate-key slots exist by design (concurrent
//      find_or_claim races publish the same key into different slots) — walk
//      the chain until MARKER_EMPTY and oi_merge ALL matching slots' packed
//      counters inline (exact: OI counters are commutative).
//   3. Bin the merged counter with an inline port of oi_bin_to_cell, map the
//      cell to an integer weight quantized ×4 (F=0, wF=1, wT=3, T=4 — quarters
//      of the QUAD forward weights 0.0/0.25/0.75/1.0). A MISS (chain hit EMPTY
//      with no match) contributes the QUAD default cell WEAK_FALSE = 1,
//      matching default_cell_for_mode on the export-eval path (which is also
//      why the export wF-filter is a no-op for scoring).
//   4. Atomically accumulate into votes[example * num_clusters + cluster].
//
// OI mode only — legacy slot values are clamped cells, not packed counters,
// and the legacy path keeps the sorted export.
//
// Sampling: NONE. neuron_sample_rate thins TRAINING writes only; evaluation
// probes every (example, neuron) pair, exactly like the export-path eval.

// Mirrors marker_train.metal's NeuronTrainMeta (separate compile unit).
struct NeuronTrainMeta {
    uint bits;           // address bit count for this neuron
    uint conn_offset;    // offset into connections array (genome-relative)
    uint slot_offset;    // start of this neuron's slots within markers/keys/values
    uint slot_capacity;  // slot count (power of 2)
    uint cluster_idx;    // which cluster this neuron belongs to (0 for single-cluster)
    uint _pad;           // alignment / reserved
};

struct ProbeParams {
    uint num_examples;          // total examples in the probe set
    uint num_neurons;           // neurons in this chunk's table
    uint words_per_example;     // packed u64 stride of the probe-set input
    uint num_clusters;          // votes stride (1 for the single-cluster path)
    uint example_offset;        // host-chunked dispatch start (cancel polling)
    uint examples_in_dispatch;  // examples this dispatch covers
};

// QUAD cell → integer weight quantized ×4. votes/4.0 reproduces the
// WNN_QUAD_WEIGHTS sum exactly (all quarters are exact in f32).
constant uint WNN_QUAD_VOTES_X4[4] = {0u, 1u, 3u, 4u};

// Sign-extend the 30-bit net field of a packed OI counter (mirrors
// neuron_memory.rs oi_unpack's net handling).
inline int oi_net_signed(uint word) {
    uint net30 = word & OI_NET_MASK;
    if ((net30 & (1u << 29)) != 0u) {
        return int(net30 | (~OI_NET_MASK));
    }
    return int(net30);
}

// Inline port of neuron_memory.rs::oi_merge — merges two packed counters that
// accumulated nudges for the SAME address in separate slots. Exactly like
// oi_apply_nudge_inline mirrors oi_apply_nudge:
//   net    — sums (saturating at the 30-bit bound; operands are within ±2^29
//            each, so the int32 add cannot overflow before the clamp)
//   obs≥1  — either side observed
//   obs≥2  — either side saw ≥2, or both sides saw ≥1
inline uint oi_merge_inline(uint a, uint b) {
    int net = clamp(oi_net_signed(a) + oi_net_signed(b), OI_NET_MIN_INT, OI_NET_MAX_INT);
    bool obs1_a = ((a >> OI_OBS_GE_1_BIT) & 1u) != 0u;
    bool obs2_a = ((a >> OI_OBS_GE_2_BIT) & 1u) != 0u;
    bool obs1_b = ((b >> OI_OBS_GE_1_BIT) & 1u) != 0u;
    bool obs2_b = ((b >> OI_OBS_GE_2_BIT) & 1u) != 0u;
    bool obs1 = obs1_a || obs1_b;
    bool obs2 = obs2_a || obs2_b || (obs1_a && obs1_b);
    uint o1 = uint(obs1) << OI_OBS_GE_1_BIT;
    uint o2 = uint(obs2) << OI_OBS_GE_2_BIT;
    return o2 | o1 | (uint(net) & OI_NET_MASK);
}

// Inline port of neuron_memory.rs::oi_bin_to_cell. Returns a QUAD cell in
// {0=FALSE, 1=WEAK_FALSE, 2=WEAK_TRUE, 3=TRUE}.
inline uint oi_bin_to_cell_inline(uint packed) {
    int net = oi_net_signed(packed);
    bool obs1 = ((packed >> OI_OBS_GE_1_BIT) & 1u) != 0u;
    bool obs2 = ((packed >> OI_OBS_GE_2_BIT) & 1u) != 0u;
    if (!obs1) {
        return 1u;  // untouched → QUAD_WEAK_FALSE
    }
    if (!obs2) {
        // obs == 1: force WEAK based on sign of net.
        return (net > 0) ? 2u : 1u;
    }
    // obs >= 2: option 1 thresholds.
    if (net <= -1) return 0u;
    if (net == 0)  return 1u;
    if (net == 1)  return 2u;
    return 3u;
}

// Grid: X = examples (SIMD-coalesced — adjacent threads probe the SAME
// neuron's table region for adjacent examples, mirroring
// sparse_forward_to_buffer's axis choice), Y = neurons-in-chunk. The example
// axis is huge (millions), so occupancy is trivially good — no z-chunking.
kernel void marker_probe_eval(
    device const ulong* packed_input          [[buffer(0)]],
    device const int* connections             [[buffer(1)]],
    device const NeuronTrainMeta* neuron_meta [[buffer(2)]],
    constant ProbeParams& params              [[buffer(3)]],
    device const uint* slot_markers           [[buffer(4)]],
    device const ulong* slot_keys             [[buffer(5)]],
    device const uint* slot_values            [[buffer(6)]],
    device atomic_uint* votes                 [[buffer(7)]],
    uint2 tid                                 [[thread_position_in_grid]]
) {
    uint local_ex = tid.x;
    uint neuron_idx = tid.y;
    if (local_ex >= params.examples_in_dispatch || neuron_idx >= params.num_neurons) return;
    uint example_idx = params.example_offset + local_ex;
    if (example_idx >= params.num_examples) return;

    NeuronTrainMeta meta = neuron_meta[neuron_idx];

    // Single-genome dispatch → genome connection base is 0; meta.conn_offset
    // is already the absolute offset (identical to training's
    // genome_idx * conn_stride + meta.conn_offset with genome_idx = 0).
    ulong addr = wnn_compute_address_u64(
        packed_input + (example_idx * params.words_per_example),
        connections + meta.conn_offset,
        meta.bits);

    // Probe chain: continue past matches (duplicates by design), stop at
    // MARKER_EMPTY. Post-training the table is quiescent (every claim was
    // published FINAL before the training command buffer completed), so plain
    // loads are safe; a FINAL slot's key is fully written by the FSM contract.
    uint mask = meta.slot_capacity - 1u;
    uint idx = slot_hash(addr, mask);
    uint acc = 0u;
    bool found = false;
    for (uint probe = 0; probe < meta.slot_capacity; probe++) {
        uint slot = meta.slot_offset + idx;
        uint m = slot_markers[slot];
        if (m == MARKER_EMPTY) break;
        if (m == MARKER_FINAL && slot_keys[slot] == addr) {
            uint v = slot_values[slot];
            acc = found ? oi_merge_inline(acc, v) : v;
            found = true;
        }
        idx = (idx + 1u) & mask;
    }

    uint cell = found ? oi_bin_to_cell_inline(acc) : 1u;  // miss → WEAK_FALSE
    atomic_fetch_add_explicit(
        &votes[example_idx * params.num_clusters + meta.cluster_idx],
        WNN_QUAD_VOTES_X4[cell],
        memory_order_relaxed);
}
