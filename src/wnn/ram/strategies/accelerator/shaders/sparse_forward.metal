//
// Metal Compute Shader for Sparse RAMLM Forward Pass
//
// GPU-accelerated forward pass using binary search on sorted sparse memory.
// This enables GPU acceleration for high-bit architectures (11-30+ bits)
// where dense memory is infeasible (2^n cells would exceed memory).
//
// Memory format:
// - keys_flat: Sorted u64 addresses per neuron, concatenated
// - values_flat: Corresponding cell values (0=FALSE, 1=TRUE, 2=EMPTY)
// - offsets: Start index in keys_flat for each neuron
// - counts: Number of entries for each neuron
//
// Lookup: Binary search in sorted keys → O(log n) per neuron
//

#include <metal_stdlib>
using namespace metal;

// Memory cell constants
constant uint CELL_FALSE = 0;
constant uint CELL_TRUE = 1;
constant uint CELL_EMPTY = 2;

// Memory mode constants
constant uint MEM_MODE_TERNARY = 0;
constant uint MEM_MODE_QUAD_BINARY = 1;
constant uint MEM_MODE_QUAD_WEIGHTED = 2;

// Quad weights for QUAD_WEIGHTED mode
constant float QUAD_WEIGHTS[4] = {0.0f, 0.25f, 0.75f, 1.0f};

// Parameters for sparse forward pass
struct SparseParams {
    uint num_examples;
    uint words_per_example;  // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_neurons;
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_clusters;
    float empty_value;  // Value for EMPTY cells (0.0 = abstain, 0.5 = uncertain)
    uint memory_mode;   // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
    uint default_cell_value;  // Default for missing cells (TERNARY:2=EMPTY, QUAD:1=WEAK_FALSE)
};

// Compute memory address for a neuron given packed u64 input bits
// Uses MSB-first addressing (matches Python/Rust)
inline ulong compute_address_sparse(
    device const ulong* packed_input,
    device const int* connections,
    uint bits_per_neuron
) {
    ulong address = 0;
    for (uint i = 0; i < bits_per_neuron; i++) {
        int conn_idx = connections[i];
        if (conn_idx >= 0) {
            uint word_idx = uint(conn_idx) / 64;
            uint bit_idx = uint(conn_idx) % 64;
            if ((packed_input[word_idx] >> bit_idx) & 1uL) {
                address |= (1uL << (bits_per_neuron - 1 - i));
            }
        }
    }
    return address;
}

// Binary search for address in sorted keys array
// Returns value if found, default_value if not found
inline uint binary_search_lookup(
    device const ulong* keys_flat,
    device const uchar* values_flat,
    uint start,
    uint count,
    ulong address,
    uint default_value = CELL_EMPTY
) {
    if (count == 0) {
        return default_value;
    }

    uint left = 0;
    uint right = count;

    // Binary search
    while (left < right) {
        uint mid = left + (right - left) / 2;
        ulong key = keys_flat[start + mid];

        if (key == address) {
            return values_flat[start + mid];
        } else if (key < address) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return default_value;  // Not found
}

//
// Mode-switched accumulation for sparse memory.
// Returns probability based on memory_mode.
//
inline float accumulate_sparse(
    device const ulong* packed_input,
    device const int* connections_flat,
    device const ulong* keys_flat,
    device const uchar* values_flat,
    device const uint* offsets,
    device const uint* counts,
    uint start_neuron,
    uint neurons_per_cluster,
    uint bits_per_neuron,
    uint memory_mode,
    float empty_value,
    uint default_cell_value
) {
    if (memory_mode == MEM_MODE_QUAD_WEIGHTED) {
        float weighted_sum = 0.0f;
        for (uint n = 0; n < neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* connections = connections_flat + neuron_idx * bits_per_neuron;
            ulong address = compute_address_sparse(packed_input, connections, bits_per_neuron);
            uint cell = binary_search_lookup(keys_flat, values_flat,
                offsets[neuron_idx], counts[neuron_idx], address, default_cell_value);
            weighted_sum += QUAD_WEIGHTS[cell];
        }
        return weighted_sum / float(neurons_per_cluster);
    } else if (memory_mode == MEM_MODE_QUAD_BINARY) {
        uint count = 0;
        for (uint n = 0; n < neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* connections = connections_flat + neuron_idx * bits_per_neuron;
            ulong address = compute_address_sparse(packed_input, connections, bits_per_neuron);
            uint cell = binary_search_lookup(keys_flat, values_flat,
                offsets[neuron_idx], counts[neuron_idx], address, default_cell_value);
            if (cell >= 2) count++;
        }
        return float(count) / float(neurons_per_cluster);
    } else {
        // TERNARY (default)
        uint count_true = 0;
        uint count_empty = 0;
        for (uint n = 0; n < neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* connections = connections_flat + neuron_idx * bits_per_neuron;
            ulong address = compute_address_sparse(packed_input, connections, bits_per_neuron);
            uint cell = binary_search_lookup(keys_flat, values_flat,
                offsets[neuron_idx], counts[neuron_idx], address, default_cell_value);
            if (cell == CELL_TRUE) count_true++;
            else if (cell == CELL_EMPTY) count_empty++;
        }
        return (float(count_true) + empty_value * float(count_empty)) / float(neurons_per_cluster);
    }
}

//
// Sparse Forward Pass Kernel
//
// Each thread computes the probability for one (example, cluster) pair.
// Grid: (num_clusters, num_examples)
//
kernel void sparse_forward_pass(
    device const ulong* packed_input_flat [[buffer(0)]],  // [num_examples * words_per_example]
    device const int* connections_flat [[buffer(1)]],      // [num_neurons * bits_per_neuron]
    device const ulong* keys_flat [[buffer(2)]],           // Sorted keys, all neurons concatenated
    device const uchar* values_flat [[buffer(3)]],         // Values corresponding to keys
    device const uint* offsets [[buffer(4)]],              // [num_neurons] start offset per neuron
    device const uint* counts [[buffer(5)]],               // [num_neurons] entry count per neuron
    constant SparseParams& params [[buffer(6)]],
    device float* probs_out [[buffer(7)]],                 // [num_examples * num_clusters]
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (cluster_idx >= params.num_clusters) return;
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    uint start_neuron = cluster_idx * params.neurons_per_cluster;

    float prob = accumulate_sparse(
        packed_input, connections_flat, keys_flat, values_flat, offsets, counts,
        start_neuron, params.neurons_per_cluster, params.bits_per_neuron,
        params.memory_mode, params.empty_value, params.default_cell_value
    );

    uint output_idx = example_idx * params.num_clusters + cluster_idx;
    probs_out[output_idx] = prob;
}

//
// Sparse Forward Pass Per-Example Kernel
//
// Each thread computes ALL cluster probabilities for one example.
// Better memory coalescing for small neurons_per_cluster.
// Grid: (num_examples)
//
kernel void sparse_forward_pass_per_example(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const ulong* keys_flat [[buffer(2)]],
    device const uchar* values_flat [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device const uint* counts [[buffer(5)]],
    constant SparseParams& params [[buffer(6)]],
    device float* probs_out [[buffer(7)]],
    uint example_idx [[thread_position_in_grid]]
) {
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    device float* example_probs = probs_out + example_idx * params.num_clusters;

    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint start_neuron = cluster_idx * params.neurons_per_cluster;

        example_probs[cluster_idx] = accumulate_sparse(
            packed_input, connections_flat, keys_flat, values_flat, offsets, counts,
            start_neuron, params.neurons_per_cluster, params.bits_per_neuron,
            params.memory_mode, params.empty_value, params.default_cell_value
        );
    }
}

//
// Tiered Sparse Forward Pass Kernel
//
// Supports tiered architecture with different bits/neurons per tier.
// Each thread handles one (example, cluster).
// Tier configuration is passed via tier_config buffer.
//
struct TierInfo {
    uint end_cluster;         // Exclusive end cluster for this tier
    uint neurons_per_cluster; // Neurons per cluster in this tier
    uint bits_per_neuron;     // Bits per neuron in this tier
    uint start_neuron;        // Global start neuron index for this tier
};

struct TieredParams {
    uint num_examples;
    uint words_per_example;    // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_clusters;
    uint num_tiers;
    float empty_value;
    uint memory_mode;          // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
    uint default_cell_value;   // Default for missing cells
};

//
// Sparse Forward Pass to Shared Buffer Kernel
//
// Writes scores directly to a shared buffer at specified cluster offsets.
// This avoids GPU→CPU→GPU transfer when computing CE across multiple groups.
//
// The cluster_ids buffer maps local cluster index → global cluster ID.
// Output is written to: shared_buffer[example_idx * total_clusters + cluster_ids[local_cluster]]
//
struct SparseToBufferParams {
    uint num_examples;
    uint words_per_example;     // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_neurons;           // Neurons in this group
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_group_clusters;    // Clusters in this group
    uint total_clusters;        // Total clusters across all groups
    float empty_value;
    uint memory_mode;           // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
    uint default_cell_value;    // Default for missing cells
};

// Parameters for sparse forward to buffer with per-cluster masking
// Used when clusters are coalesced by neuron bucket (e.g., 5-7 neurons → max 7)
// Each cluster has its actual neuron count for correct scoring
struct SparseToBufferMaskedParams {
    uint num_examples;
    uint words_per_example;     // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_neurons;           // Total neurons in this group (sum of max_neurons * num_clusters)
    uint bits_per_neuron;
    uint max_neurons_per_cluster;  // Max neurons (for memory layout)
    uint num_group_clusters;    // Clusters in this group
    uint total_clusters;        // Total clusters across all groups
    float empty_value;
    uint memory_mode;           // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
    uint default_cell_value;    // Default for missing cells
};

kernel void sparse_forward_to_buffer(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const ulong* keys_flat [[buffer(2)]],
    device const uchar* values_flat [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device const uint* counts [[buffer(5)]],
    device const uint* cluster_ids [[buffer(6)]],      // Maps local → global cluster ID
    constant SparseToBufferParams& params [[buffer(7)]],
    device float* shared_buffer [[buffer(8)]],         // Shared across all groups
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint local_cluster = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (local_cluster >= params.num_group_clusters) return;
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    uint start_neuron = local_cluster * params.neurons_per_cluster;

    float prob = accumulate_sparse(
        packed_input, connections_flat, keys_flat, values_flat, offsets, counts,
        start_neuron, params.neurons_per_cluster, params.bits_per_neuron,
        params.memory_mode, params.empty_value, params.default_cell_value
    );

    uint global_cluster = cluster_ids[local_cluster];
    uint output_idx = example_idx * params.total_clusters + global_cluster;
    shared_buffer[output_idx] = prob;
}

//
// Sparse Forward Pass to Buffer with Per-Cluster Masking Kernel
//
// Similar to sparse_forward_to_buffer but supports coalesced groups where
// clusters have different actual neuron counts. The max_neurons_per_cluster
// is used for memory layout, but actual_neurons is used for scoring.
//
// This enables grouping similar neuron counts (e.g., 5-7 neurons all in one group
// with max=7) while preserving exact scoring accuracy through per-cluster masking.
//
kernel void sparse_forward_to_buffer_masked(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const ulong* keys_flat [[buffer(2)]],
    device const uchar* values_flat [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device const uint* counts [[buffer(5)]],
    device const uint* cluster_ids [[buffer(6)]],           // Maps local → global cluster ID
    device const uint* actual_neurons [[buffer(7)]],        // Actual neurons per local cluster
    constant SparseToBufferMaskedParams& params [[buffer(8)]],
    device float* shared_buffer [[buffer(9)]],              // Shared across all groups
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint local_cluster = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (local_cluster >= params.num_group_clusters) return;
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    uint actual_neuron_count = actual_neurons[local_cluster];
    uint start_neuron = local_cluster * params.max_neurons_per_cluster;

    float prob = accumulate_sparse(
        packed_input, connections_flat, keys_flat, values_flat, offsets, counts,
        start_neuron, actual_neuron_count, params.bits_per_neuron,
        params.memory_mode, params.empty_value, params.default_cell_value
    );

    uint global_cluster = cluster_ids[local_cluster];
    uint output_idx = example_idx * params.total_clusters + global_cluster;
    shared_buffer[output_idx] = prob;
}

//
// Tiered Sparse Forward Pass Kernel
//
// Supports tiered architecture with different bits/neurons per tier.
// Each thread handles one (example, cluster).
// Tier configuration is passed via tier_config buffer.
//
kernel void tiered_sparse_forward_pass(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const ulong* keys_flat [[buffer(2)]],
    device const uchar* values_flat [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device const uint* counts [[buffer(5)]],
    constant TieredParams& params [[buffer(6)]],
    device const TierInfo* tiers [[buffer(7)]],
    device float* probs_out [[buffer(8)]],
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (cluster_idx >= params.num_clusters) return;
    if (example_idx >= params.num_examples) return;

    // Find which tier this cluster belongs to
    uint tier_idx = 0;
    for (uint t = 0; t < params.num_tiers; t++) {
        if (cluster_idx < tiers[t].end_cluster) {
            tier_idx = t;
            break;
        }
    }

    TierInfo tier = tiers[tier_idx];
    uint start_cluster_for_tier = (tier_idx == 0) ? 0 : tiers[tier_idx - 1].end_cluster;
    uint local_cluster = cluster_idx - start_cluster_for_tier;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;

    uint local_start_neuron = local_cluster * tier.neurons_per_cluster;
    uint global_start_neuron = tier.start_neuron + local_start_neuron;

    float prob = accumulate_sparse(
        packed_input, connections_flat, keys_flat, values_flat, offsets, counts,
        global_start_neuron, tier.neurons_per_cluster, tier.bits_per_neuron,
        params.memory_mode, params.empty_value, params.default_cell_value
    );

    uint output_idx = example_idx * params.num_clusters + cluster_idx;
    probs_out[output_idx] = prob;
}

//
// General Sparse Forward Pass Kernel (Per-Cluster Metadata)
//
// Unified kernel for both tiered and adaptive architectures.
// Each cluster has its own ClusterInfo with (neurons, bits, start_neuron, connection_offset).
// No tier lookup needed — each thread reads its cluster's metadata directly.
//
// This fixes the connection indexing bug in tiered_sparse_forward_pass where
// connections_flat was indexed by global_neuron_idx * tier.bits_per_neuron,
// which is wrong when tiers have different bits_per_neuron.
//

struct ClusterInfo {
    uint neurons_per_cluster;  // How many neurons this cluster has
    uint bits_per_neuron;      // Address bits per neuron
    uint start_neuron;         // Index into offsets/counts arrays
    uint connection_offset;    // Offset into connections_flat
};

struct GeneralParams {
    uint num_examples;
    uint words_per_example;    // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_clusters;
    float empty_value;
    uint memory_mode;          // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
    uint default_cell_value;   // Default for missing cells
};

kernel void general_sparse_forward_pass(
    device const ulong* packed_input_flat [[buffer(0)]],    // [num_examples * words_per_example]
    device const int* connections_flat [[buffer(1)]],        // Variable-stride, indexed by ClusterInfo
    device const ulong* keys_flat [[buffer(2)]],            // Sorted keys for all neurons
    device const uchar* values_flat [[buffer(3)]],          // Cell values for all neurons
    device const uint* offsets [[buffer(4)]],               // [total_neurons] start in keys per neuron
    device const uint* counts [[buffer(5)]],                // [total_neurons] entries per neuron
    device const ClusterInfo* cluster_infos [[buffer(6)]],  // [num_clusters] per-cluster metadata
    constant GeneralParams& params [[buffer(7)]],
    device float* probs_out [[buffer(8)]],                  // [num_examples * num_clusters]
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (cluster_idx >= params.num_clusters) return;
    if (example_idx >= params.num_examples) return;

    ClusterInfo info = cluster_infos[cluster_idx];
    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;

    // Note: general kernel uses per-cluster connection_offset, not uniform indexing.
    // We can't use accumulate_sparse directly since it uses uniform start_neuron * bits_per_neuron
    // for connection indexing. Inline the mode-switched logic here.
    if (params.memory_mode == MEM_MODE_QUAD_WEIGHTED) {
        float weighted_sum = 0.0f;
        for (uint n = 0; n < info.neurons_per_cluster; n++) {
            uint neuron_idx = info.start_neuron + n;
            device const int* connections = connections_flat + info.connection_offset + n * info.bits_per_neuron;
            ulong address = compute_address_sparse(packed_input, connections, info.bits_per_neuron);
            uint cell = binary_search_lookup(keys_flat, values_flat,
                offsets[neuron_idx], counts[neuron_idx], address, params.default_cell_value);
            weighted_sum += QUAD_WEIGHTS[cell];
        }
        probs_out[example_idx * params.num_clusters + cluster_idx] = weighted_sum / float(info.neurons_per_cluster);
    } else if (params.memory_mode == MEM_MODE_QUAD_BINARY) {
        uint count = 0;
        for (uint n = 0; n < info.neurons_per_cluster; n++) {
            uint neuron_idx = info.start_neuron + n;
            device const int* connections = connections_flat + info.connection_offset + n * info.bits_per_neuron;
            ulong address = compute_address_sparse(packed_input, connections, info.bits_per_neuron);
            uint cell = binary_search_lookup(keys_flat, values_flat,
                offsets[neuron_idx], counts[neuron_idx], address, params.default_cell_value);
            if (cell >= 2) count++;
        }
        probs_out[example_idx * params.num_clusters + cluster_idx] = float(count) / float(info.neurons_per_cluster);
    } else {
        uint count_true = 0;
        uint count_empty = 0;
        for (uint n = 0; n < info.neurons_per_cluster; n++) {
            uint neuron_idx = info.start_neuron + n;
            device const int* connections = connections_flat + info.connection_offset + n * info.bits_per_neuron;
            ulong address = compute_address_sparse(packed_input, connections, info.bits_per_neuron);
            uint cell = binary_search_lookup(keys_flat, values_flat,
                offsets[neuron_idx], counts[neuron_idx], address, params.default_cell_value);
            if (cell == CELL_TRUE) count_true++;
            else if (cell == CELL_EMPTY) count_empty++;
        }
        probs_out[example_idx * params.num_clusters + cluster_idx] =
            (float(count_true) + params.empty_value * float(count_empty)) / float(info.neurons_per_cluster);
    }
}
