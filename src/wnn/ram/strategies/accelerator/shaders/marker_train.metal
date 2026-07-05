#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Option B-marker GPU training kernel
// =============================================================================
//
// Per-genome training entirely on GPU. Each thread handles one (neuron,
// example) pair. Writes go through the 3-state marker FSM:
//   MARKER_EMPTY (0) → MARKER_CLAIMED (1) → MARKER_FINAL (0xFFFFFFFF)
//
// Only 32-bit atomic CAS is used (Metal's _valid_compare_exchange_type
// list — atomic_uint). 64-bit keys are stored non-atomically once the
// writer holds the CLAIMED marker.
//
// Buffers (must match Rust MarkerHashTable's Metal storage layout):
//   slot_markers : device atomic_uint*  [capacity]
//   slot_keys    : device ulong*        [capacity]  (non-atomic; FSM-protected)
//   slot_values  : device atomic_uint*  [capacity]  (low 8 bits hold cell value)
//
// Per-genome scope (one neuron group per dispatch — multi-cluster handled
// by Python orchestration):
//   packed_input        : device ulong* [num_examples × words_per_example]
//   connections         : device int*   [sum of per-neuron bits]
//   neuron_meta         : device NeuronMeta* [num_neurons]
//   train_targets       : device long*  [num_examples]
//   train_negatives     : device long*  [num_examples × num_negatives]
//   class_weights       : device uint*  [num_classes]  (None → all 1s)
//
// Address space: each neuron has its own (markers, keys, values) sub-region
// within the larger flat buffer. neuron_meta[n].slot_offset is the start
// index for that neuron's slots; neuron_meta[n].slot_capacity is the
// number of slots (must be power of two — same as Rust MarkerInner).

struct NeuronTrainMeta {
    uint bits;           // address bit count for this neuron
    uint conn_offset;    // offset into connections array (genome-relative
                         // for batched dispatch — see below)
    uint slot_offset;    // start of this neuron's slots within markers/keys/values
    uint slot_capacity;  // slot count (power of 2)
    uint cluster_idx;    // which cluster this neuron belongs to (0 for single-cluster)
    uint _pad;           // alignment / reserved
};

struct TrainParams {
    uint num_examples;
    uint num_negatives;
    uint num_neurons;        // per-genome neuron count (uniform within batch)
    uint num_genomes;        // batched dispatch: how many genomes in this call
    uint words_per_example;
    uint num_classes;
    uint memory_mode;        // 2 = QUAD_WEIGHTED (default); 0 = ternary
    uint single_cluster;     // 1 = binary IDS path (true_cluster always 0)
    uint normal_class;       // 0 = benign; used for IDS multi-cluster path
    uint conn_stride;        // per-genome connection count (sum of bits per
                             // neuron); each genome's connections start at
                             // (genome_idx * conn_stride) in the flat array
    float neuron_sample_rate; // 0.0-1.0; <1.0 enables per-(neuron, ex) skip
    uint rng_seed;            // seed for sampling hash (matches CPU)
    uint num_example_chunks;  // B10: threads along example axis per (g, n)
    uint oi_mode;             // 1 = order-independent training (packed counter);
                              // 0 = legacy clamped-nudge. Slot values are
                              // interpreted accordingly. A separate host-side
                              // commit pass bins counters → 2-bit cells when
                              // oi_mode=1.
    uint example_offset;        // 31/05/2026: host-chunked dispatch start
                                // index in the global examples array.
                                // Defaults to 0 for backwards-compatible
                                // single-dispatch behaviour.
    uint examples_in_dispatch;  // 31/05/2026: number of examples this
                                // kernel call should process. The kernel
                                // reads the slice [example_offset, ex_off
                                // + examples_in_dispatch) of the global
                                // arrays. With examples_in_dispatch ==
                                // num_examples this matches the original
                                // single-dispatch behaviour exactly.
    uint neuron_index_offset;   // 05/07/2026: neuron-axis chunking. Added to
                                // neuron_idx in should_skip_sample so a
                                // chunked dispatch samples with the GLOBAL
                                // neuron index (bit-exact vs unchunked and
                                // vs CPU). 0 for unchunked dispatches.
};

// xorshift32-based per-(neuron, example) sampling decision. Matches CPU
// path at adaptive.rs:2867-2880 byte-for-byte.
// Returns true if this (neuron_idx, ex_idx) pair should be SKIPPED.
inline bool should_skip_sample(uint neuron_idx, uint ex_idx, uint rng_seed, float sample_rate) {
    if (sample_rate >= 1.0f) return false;
    uint rng = rng_seed + neuron_idx * 1000003u + ex_idx * 2654435761u;
    if (rng == 0u) rng = 1u;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    float r = float(rng >> 8) / 16777216.0f;
    return r >= sample_rate;
}

// MARKER_* constants + slot_hash moved to core/shaders/marker_slots.metal.

// Compute the address for (neuron, example) — matches CPU
// `compute_address_packed_bytes` and ramlm.metal::compute_ram_address.
//
// MSB-first ordering: bit i in the connection list lands at position
// (bits - 1 - i) in the address. The earlier LSB-first version
// (`addr |= bit << i`) was the bug behind the 15/05/2026 production
// divergence — synthetic parity tests passed because the CPU reference
// also used LSB-first, but production uses MSB-first throughout.
inline ulong compute_address(
    device const ulong* packed_input,
    device const int* connections,
    uint conn_offset,
    uint bits,
    uint example_idx,
    uint words_per_example
) {
    // Thin wrapper over the canonical helper in common.metal (MSB-first).
    // The earlier LSB-first version of this function caused the 15/05/2026
    // trivial-baseline bug — address semantics now live in exactly one place.
    return wnn_compute_address_u64(
        packed_input + (example_idx * words_per_example),
        connections + conn_offset,
        bits);
}

// find_or_claim_slot / slot_nudge / slot_nudge_oi (+ OI counter) moved to
// core/shaders/marker_slots.metal (prepended before this shader).


// Main training kernel — ONE THREAD PER (genome, neuron) PAIR.
//
// 2D grid: x = neuron_idx within genome, y = genome_idx. Each thread
// owns one (genome, neuron) cell's slot region exclusively and processes
// all examples sequentially. No atomic contention between threads because
// each (genome, neuron) pair's slot region is disjoint.
//
// Parallelism: num_neurons × num_genomes threads. For typical batch
// (16 genomes × 100 neurons = 1600 threads) this fills ~50 simdgroups
// on a 40-core GPU (~125% — fully saturated with some queueing).
//
// Connections layout: flat per-genome. Genome g's neuron n's connections
// start at (g * conn_stride + neuron_meta[g*num_neurons + n].conn_offset).
//
// Slot layout: flat per-(genome, neuron). The neuron_meta's slot_offset
// already encodes the (genome, neuron) position into the flat buffer
// (set by host-side dispatcher).
kernel void marker_train(
    device const ulong* packed_input          [[buffer(0)]],
    device const int* connections             [[buffer(1)]],
    device const NeuronTrainMeta* neuron_meta [[buffer(2)]],
    device const long* train_targets          [[buffer(3)]],
    device const long* train_negatives        [[buffer(4)]],
    device const uint* class_weights          [[buffer(5)]],
    constant TrainParams& params              [[buffer(6)]],
    device atomic_uint* slot_markers          [[buffer(7)]],
    device ulong* slot_keys                   [[buffer(8)]],
    device atomic_uint* slot_values           [[buffer(9)]],
    uint3 tid                                 [[thread_position_in_grid]]
) {
    uint neuron_idx = tid.x;
    uint genome_idx = tid.y;
    uint chunk_idx = tid.z;
    uint num_chunks = max(params.num_example_chunks, 1u);
    if (neuron_idx >= params.num_neurons || genome_idx >= params.num_genomes
        || chunk_idx >= num_chunks) return;

    uint meta_idx = genome_idx * params.num_neurons + neuron_idx;
    NeuronTrainMeta meta = neuron_meta[meta_idx];

    // Per-genome connection base offset. meta.conn_offset is genome-relative
    // (set by host); we add the genome's start offset to get the absolute
    // index into the flat connections buffer.
    uint conn_genome_base = genome_idx * params.conn_stride;
    uint conn_abs_offset = conn_genome_base + meta.conn_offset;

    // 31/05/2026: host-chunked example range for cooperative cancellation.
    // Process only [params.example_offset, params.example_offset +
    // params.examples_in_dispatch). For backwards-compatibility, callers
    // that don't chunk set example_offset=0 and examples_in_dispatch=
    // num_examples, restoring the original "process all examples in one
    // dispatch" behaviour. The B10 num_example_chunks Z-axis still works
    // within this host chunk.
    uint host_chunk_count = max(params.examples_in_dispatch, 1u);
    uint chunk_size = (host_chunk_count + num_chunks - 1u) / num_chunks;
    uint ex_start = params.example_offset + chunk_idx * chunk_size;
    uint ex_end = min(ex_start + chunk_size,
                       params.example_offset + host_chunk_count);
    // Don't run off the end of the global examples array.
    ex_end = min(ex_end, params.num_examples);

    for (uint example_idx = ex_start; example_idx < ex_end; example_idx++) {
        // Sampling skip: same xorshift hash as CPU path. Applied uniformly
        // before the cluster/negative checks so the kernel skips this
        // (neuron, example) pair entirely when sample_rate < 1.0.
        if (should_skip_sample(neuron_idx + params.neuron_index_offset, example_idx, params.rng_seed, params.neuron_sample_rate)) {
            continue;
        }

        long target = train_targets[example_idx];

        // Decide whether this neuron should participate in this example, and
        // if so with what nudge direction. Mirrors CPU semantics at
        // adaptive.rs:2837-2940.
        //
        //   single_cluster (binary IDS): all neurons participate; direction
        //   = (target == 1). cluster_idx == 0 for all neurons.
        //
        //   multi-cluster: neurons in cluster == target nudge TRUE.
        //   Neurons in any cluster c where c appears in
        //   train_negatives[example_idx][0..num_negatives] nudge FALSE.
        //   Other neurons skip this example.
        bool participates;
        bool nudge_true;
        if (params.single_cluster != 0u) {
            participates = true;
            nudge_true = (target == 1);
        } else {
            uint cid = meta.cluster_idx;
            if (uint(target) == cid) {
                participates = true;
                nudge_true = true;
            } else {
                participates = false;
                nudge_true = false;
                // Scan negatives. num_negatives is small in practice (≤20).
                uint neg_base = example_idx * params.num_negatives;
                for (uint k = 0; k < params.num_negatives; k++) {
                    long false_cluster = train_negatives[neg_base + k];
                    // CPU path also skips if false_cluster == true_cluster.
                    if (false_cluster == target) continue;
                    if (uint(false_cluster) == cid) {
                        participates = true;
                        nudge_true = false;
                        break;
                    }
                }
            }
        }
        if (!participates) continue;

        ulong addr = compute_address(
            packed_input, connections, conn_abs_offset, meta.bits,
            example_idx, params.words_per_example
        );

        uint weight = 1u;
        if (params.num_classes > 0u) {
            uint wi = (uint(target) < params.num_classes) ? uint(target) : 0u;
            weight = class_weights[wi];
            if (weight == 0u) weight = 1u;
        }

        uint slot = find_or_claim_slot(
            slot_markers, slot_keys, meta.slot_offset, meta.slot_capacity, addr
        );
        if (slot != 0xFFFFFFFFu) {
            if (params.oi_mode == 1u) {
                // Single atomic update per example: ±weight in one fetch.
                int delta = nudge_true ? int(weight) : -int(weight);
                slot_nudge_oi(slot_values, slot, delta);
            } else {
                for (uint r = 0; r < weight; r++) {
                    slot_nudge(slot_values, slot, nudge_true);
                }
            }
        }
    }
}
