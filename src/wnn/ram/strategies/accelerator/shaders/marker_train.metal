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

constant uint MARKER_EMPTY = 0u;
constant uint MARKER_CLAIMED = 1u;
constant uint MARKER_FINAL = 0xFFFFFFFFu;

// Murmur3 finalizer — identical mixer to Rust MarkerInner::hash.
inline uint slot_hash(ulong key, uint mask) {
    ulong x = key;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdul;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ul;
    x ^= x >> 33;
    return uint(x) & mask;
}

// Compute the address for (neuron, example) — identical to compute_addresses
// kernel and CPU `compute_address_packed_bytes`.
inline ulong compute_address(
    device const ulong* packed_input,
    device const int* connections,
    uint conn_offset,
    uint bits,
    uint example_idx,
    uint words_per_example
) {
    ulong addr = 0;
    device const ulong* ex_words = packed_input + (example_idx * words_per_example);
    for (uint i = 0; i < bits; i++) {
        int conn_idx = connections[conn_offset + i];
        if (conn_idx < 0) continue;
        uint cu = uint(conn_idx);
        uint word_idx = cu >> 6;       // /64
        uint bit_idx = cu & 63u;
        ulong word = ex_words[word_idx];
        ulong bit = (word >> bit_idx) & 1ul;
        addr |= bit << i;
    }
    return addr;
}

// Find-or-claim slot for (neuron, key). Returns slot index, or 0xFFFFFFFF on
// table-full. Caller must check the return value before writing.
//
// Mirrors Rust MarkerInner::find_or_claim_slot but adapted for GPU:
//   - No spin_loop() hint (use bounded retry counter to avoid warp deadlock)
//   - On CLAIMED slot, probe forward instead of waiting (loses some slot
//     coalescing but avoids deadlock; load factor accommodates)
inline uint find_or_claim_slot(
    device atomic_uint* slot_markers,
    device ulong* slot_keys,
    uint slot_offset,
    uint slot_capacity,
    ulong key
) {
    uint mask = slot_capacity - 1;
    uint idx = slot_hash(key, mask);
    for (uint probe = 0; probe < slot_capacity; probe++) {
        uint slot = slot_offset + idx;
        uint m = atomic_load_explicit(&slot_markers[slot], memory_order_relaxed);
        if (m == MARKER_FINAL) {
            if (slot_keys[slot] == key) return slot;
            // Different key — probe forward.
        } else if (m == MARKER_EMPTY) {
            uint expected = MARKER_EMPTY;
            bool won = false;
            // Weak CAS retry to handle spurious failures.
            for (uint retry = 0; retry < 4; retry++) {
                if (atomic_compare_exchange_weak_explicit(
                    &slot_markers[slot], &expected, MARKER_CLAIMED,
                    memory_order_relaxed, memory_order_relaxed
                )) {
                    won = true;
                    break;
                }
                if (expected != MARKER_EMPTY) break;  // someone else won
            }
            if (won) {
                // Exclusive access — write key, then publish FINAL.
                slot_keys[slot] = key;
                atomic_store_explicit(&slot_markers[slot], MARKER_FINAL, memory_order_relaxed);
                return slot;
            }
            // Lost the race; re-examine
            if (expected == MARKER_FINAL && slot_keys[slot] == key) return slot;
            // If still CLAIMED, fall through to wait below.
            m = expected;
        }
        if (m != MARKER_EMPTY && m != MARKER_FINAL) {
            // CLAIMED — bounded wait to let the writer resolve. Avoids
            // most same-key duplicate inserts that would happen if we just
            // probed forward. Bounded to prevent warp deadlock.
            uint resolved = m;
            for (uint w = 0; w < 64; w++) {
                resolved = atomic_load_explicit(&slot_markers[slot], memory_order_relaxed);
                if (resolved == MARKER_FINAL || resolved == MARKER_EMPTY) break;
            }
            if (resolved == MARKER_FINAL && slot_keys[slot] == key) return slot;
            // If resolved == EMPTY (writer aborted? shouldn't happen) or
            // FINAL with different key, fall through to probe forward.
        }
        // Probe forward
        idx = (idx + 1) & mask;
    }
    return 0xFFFFFFFFu;  // table full
}

// Apply nudge (clamped delta) to the value field. Mirrors CPU MarkerInner::nudge.
inline void slot_nudge(device atomic_uint* slot_values, uint slot, bool target_true) {
    int delta = target_true ? 1 : -1;
    for (uint retry = 0; retry < 8; retry++) {
        uint current = atomic_load_explicit(&slot_values[slot], memory_order_relaxed);
        int new_cell = clamp(int(current) + delta, 0, 3);
        if (uint(new_cell) == current) return;  // saturated, no change
        uint exp = current;
        if (atomic_compare_exchange_weak_explicit(
            &slot_values[slot], &exp, uint(new_cell),
            memory_order_relaxed, memory_order_relaxed
        )) return;
    }
}

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
    uint2 tid                                 [[thread_position_in_grid]]
) {
    uint neuron_idx = tid.x;
    uint genome_idx = tid.y;
    if (neuron_idx >= params.num_neurons || genome_idx >= params.num_genomes) return;

    uint meta_idx = genome_idx * params.num_neurons + neuron_idx;
    NeuronTrainMeta meta = neuron_meta[meta_idx];

    // Per-genome connection base offset. meta.conn_offset is genome-relative
    // (set by host); we add the genome's start offset to get the absolute
    // index into the flat connections buffer.
    uint conn_genome_base = genome_idx * params.conn_stride;
    uint conn_abs_offset = conn_genome_base + meta.conn_offset;

    // Sequential over all examples for this (genome, neuron) cell.
    for (uint example_idx = 0; example_idx < params.num_examples; example_idx++) {
        // Sampling skip: same xorshift hash as CPU path. Applied uniformly
        // before the cluster/negative checks so the kernel skips this
        // (neuron, example) pair entirely when sample_rate < 1.0.
        if (should_skip_sample(neuron_idx, example_idx, params.rng_seed, params.neuron_sample_rate)) {
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
            for (uint r = 0; r < weight; r++) {
                slot_nudge(slot_values, slot, nudge_true);
            }
        }
    }
}
