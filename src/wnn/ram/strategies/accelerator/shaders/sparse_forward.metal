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

// Parameters for sparse forward pass
struct SparseParams {
    uint num_examples;
    uint total_input_bits;
    uint num_neurons;
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_clusters;
    float empty_value;  // Value for EMPTY cells (0.0 = abstain, 0.5 = uncertain)
};

// Compute memory address for a neuron given input bits
// Uses MSB-first addressing (matches Python/Rust)
inline ulong compute_address_sparse(
    device const uchar* input_bits,
    device const int* connections,
    uint bits_per_neuron
) {
    ulong address = 0;
    for (uint i = 0; i < bits_per_neuron; i++) {
        int conn_idx = connections[i];
        if (conn_idx >= 0 && input_bits[conn_idx]) {
            address |= (1uL << (bits_per_neuron - 1 - i));
        }
    }
    return address;
}

// Binary search for address in sorted keys array
// Returns value if found, CELL_EMPTY if not found
inline uint binary_search_lookup(
    device const ulong* keys_flat,
    device const uchar* values_flat,
    uint start,
    uint count,
    ulong address
) {
    if (count == 0) {
        return CELL_EMPTY;
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

    return CELL_EMPTY;  // Not found
}

//
// Sparse Forward Pass Kernel
//
// Each thread computes the probability for one (example, cluster) pair.
// Grid: (num_clusters, num_examples)
//
kernel void sparse_forward_pass(
    device const uchar* input_bits_flat [[buffer(0)]],   // [num_examples * total_input_bits]
    device const int* connections_flat [[buffer(1)]],     // [num_neurons * bits_per_neuron]
    device const ulong* keys_flat [[buffer(2)]],          // Sorted keys, all neurons concatenated
    device const uchar* values_flat [[buffer(3)]],        // Values corresponding to keys
    device const uint* offsets [[buffer(4)]],             // [num_neurons] start offset per neuron
    device const uint* counts [[buffer(5)]],              // [num_neurons] entry count per neuron
    constant SparseParams& params [[buffer(6)]],
    device float* probs_out [[buffer(7)]],                // [num_examples * num_clusters]
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (cluster_idx >= params.num_clusters) return;
    if (example_idx >= params.num_examples) return;

    device const uchar* input_bits = input_bits_flat + example_idx * params.total_input_bits;

    uint count_true = 0;
    uint count_empty = 0;
    uint start_neuron = cluster_idx * params.neurons_per_cluster;

    for (uint neuron_offset = 0; neuron_offset < params.neurons_per_cluster; neuron_offset++) {
        uint neuron_idx = start_neuron + neuron_offset;
        device const int* connections = connections_flat + neuron_idx * params.bits_per_neuron;

        // Compute address from input bits
        ulong address = compute_address_sparse(input_bits, connections, params.bits_per_neuron);

        // Binary search lookup in sparse memory
        uint start = offsets[neuron_idx];
        uint cnt = counts[neuron_idx];
        uint cell_value = binary_search_lookup(keys_flat, values_flat, start, cnt, address);

        if (cell_value == CELL_TRUE) {
            count_true++;
        } else if (cell_value == CELL_EMPTY) {
            count_empty++;
        }
    }

    float prob = (float(count_true) + params.empty_value * float(count_empty)) / float(params.neurons_per_cluster);
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
    device const uchar* input_bits_flat [[buffer(0)]],
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

    device const uchar* input_bits = input_bits_flat + example_idx * params.total_input_bits;
    device float* example_probs = probs_out + example_idx * params.num_clusters;

    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint count_true = 0;
        uint count_empty = 0;
        uint start_neuron = cluster_idx * params.neurons_per_cluster;

        for (uint neuron_offset = 0; neuron_offset < params.neurons_per_cluster; neuron_offset++) {
            uint neuron_idx = start_neuron + neuron_offset;
            device const int* connections = connections_flat + neuron_idx * params.bits_per_neuron;

            ulong address = compute_address_sparse(input_bits, connections, params.bits_per_neuron);

            uint start = offsets[neuron_idx];
            uint cnt = counts[neuron_idx];
            uint cell_value = binary_search_lookup(keys_flat, values_flat, start, cnt, address);

            if (cell_value == CELL_TRUE) {
                count_true++;
            } else if (cell_value == CELL_EMPTY) {
                count_empty++;
            }
        }

        example_probs[cluster_idx] = (float(count_true) + params.empty_value * float(count_empty)) / float(params.neurons_per_cluster);
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
    uint total_input_bits;
    uint num_clusters;
    uint num_tiers;
    float empty_value;  // Value for EMPTY cells (0.0 = abstain, 0.5 = uncertain)
};

kernel void tiered_sparse_forward_pass(
    device const uchar* input_bits_flat [[buffer(0)]],
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

    device const uchar* input_bits = input_bits_flat + example_idx * params.total_input_bits;

    uint count_true = 0;
    uint count_empty = 0;

    uint local_start_neuron = local_cluster * tier.neurons_per_cluster;
    uint global_start_neuron = tier.start_neuron + local_start_neuron;

    for (uint neuron_offset = 0; neuron_offset < tier.neurons_per_cluster; neuron_offset++) {
        uint global_neuron_idx = global_start_neuron + neuron_offset;
        device const int* connections = connections_flat + global_neuron_idx * tier.bits_per_neuron;

        ulong address = compute_address_sparse(input_bits, connections, tier.bits_per_neuron);

        uint start = offsets[global_neuron_idx];
        uint cnt = counts[global_neuron_idx];
        uint cell_value = binary_search_lookup(keys_flat, values_flat, start, cnt, address);

        if (cell_value == CELL_TRUE) {
            count_true++;
        } else if (cell_value == CELL_EMPTY) {
            count_empty++;
        }
    }

    float prob = (float(count_true) + params.empty_value * float(count_empty)) / float(tier.neurons_per_cluster);
    uint output_idx = example_idx * params.num_clusters + cluster_idx;
    probs_out[output_idx] = prob;
}
