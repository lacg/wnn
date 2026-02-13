//
// Metal Compute Shader for BATCHED Sparse RAMLM Forward Pass
//
// Evaluates MULTIPLE GENOMES in a single GPU dispatch.
// This avoids kernel launch overhead and allows better GPU utilization.
//
// Key insight: Input bits are SHARED across genomes (same eval data),
// only the connections and sparse memory differ per genome.
//
// Memory layout:
// - input_bits_flat: [num_examples * total_input_bits] - SHARED
// - connections_flat: [num_genomes * num_neurons * bits_per_neuron] - per genome
// - keys_flat: All genomes' keys concatenated
// - values_flat: All genomes' values concatenated
// - offsets: [num_genomes * num_neurons] - per genome, per neuron
// - counts: [num_genomes * num_neurons] - per genome, per neuron
// - genome_memory_offsets: [num_genomes] - start offset in keys/values per genome
//
// Grid: (num_clusters, num_examples, num_genomes)
//

#include <metal_stdlib>
using namespace metal;

constant uint CELL_FALSE = 0;
constant uint CELL_TRUE = 1;
constant uint CELL_EMPTY = 2;

// Memory mode constants
constant uint MEM_MODE_TERNARY = 0;
constant uint MEM_MODE_QUAD_BINARY = 1;
constant uint MEM_MODE_QUAD_WEIGHTED = 2;

// Quad weights
constant float QUAD_WEIGHTS[4] = {0.0f, 0.25f, 0.75f, 1.0f};

struct BatchedSparseParams {
    uint num_examples;
    uint total_input_bits;
    uint num_neurons;          // Per genome
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_clusters;
    uint num_genomes;
    float empty_value;
    uint memory_mode;          // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
    uint default_cell_value;   // Default for missing cells
};

// Compute memory address (same as before)
inline ulong compute_address_batched(
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

// Binary search (same as before, with configurable default)
inline uint binary_search_batched(
    device const ulong* keys_flat,
    device const uchar* values_flat,
    uint start,
    uint count,
    ulong address,
    uint default_value = CELL_EMPTY
) {
    if (count == 0) return default_value;

    uint left = 0;
    uint right = count;

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
    return default_value;
}

// Mode-switched probability computation from cell value
inline float compute_prob_batched(
    uint count_true, uint count_empty,
    float weighted_sum,
    uint neurons_per_cluster,
    uint memory_mode,
    float empty_value
) {
    if (memory_mode == MEM_MODE_QUAD_WEIGHTED) {
        return weighted_sum / float(neurons_per_cluster);
    } else if (memory_mode == MEM_MODE_QUAD_BINARY) {
        return float(count_true) / float(neurons_per_cluster);
    } else {
        return (float(count_true) + empty_value * float(count_empty)) / float(neurons_per_cluster);
    }
}

//
// Batched Sparse Forward Pass Kernel
//
// Grid: (num_clusters, num_examples, num_genomes)
// Each thread computes probability for one (genome, example, cluster) triple.
//
kernel void batched_sparse_forward_pass(
    device const uchar* input_bits_flat [[buffer(0)]],      // SHARED: [num_examples * total_input_bits]
    device const int* connections_flat [[buffer(1)]],        // [num_genomes * num_neurons * bits_per_neuron]
    device const ulong* keys_flat [[buffer(2)]],             // All genomes concatenated
    device const uchar* values_flat [[buffer(3)]],           // All genomes concatenated
    device const uint* offsets_flat [[buffer(4)]],           // [num_genomes * num_neurons]
    device const uint* counts_flat [[buffer(5)]],            // [num_genomes * num_neurons]
    device const uint* genome_key_offsets [[buffer(6)]],     // [num_genomes] offset into keys/values
    constant BatchedSparseParams& params [[buffer(7)]],
    device float* probs_out [[buffer(8)]],                   // [num_genomes * num_examples * num_clusters]
    uint3 thread_pos [[thread_position_in_grid]]
) {
    uint cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;
    uint genome_idx = thread_pos.z;

    if (cluster_idx >= params.num_clusters) return;
    if (example_idx >= params.num_examples) return;
    if (genome_idx >= params.num_genomes) return;

    // Input bits are SHARED across all genomes
    device const uchar* input_bits = input_bits_flat + example_idx * params.total_input_bits;

    // Per-genome offsets
    uint genome_neuron_offset = genome_idx * params.num_neurons;
    uint genome_conn_offset = genome_idx * params.num_neurons * params.bits_per_neuron;
    uint genome_key_offset = genome_key_offsets[genome_idx];

    uint count_true = 0;
    uint count_empty = 0;
    float weighted_sum = 0.0f;
    uint start_neuron = cluster_idx * params.neurons_per_cluster;

    for (uint neuron_offset = 0; neuron_offset < params.neurons_per_cluster; neuron_offset++) {
        uint local_neuron_idx = start_neuron + neuron_offset;
        uint global_neuron_idx = genome_neuron_offset + local_neuron_idx;

        device const int* connections = connections_flat + genome_conn_offset + local_neuron_idx * params.bits_per_neuron;

        ulong address = compute_address_batched(input_bits, connections, params.bits_per_neuron);

        uint mem_start = offsets_flat[global_neuron_idx] + genome_key_offset;
        uint mem_count = counts_flat[global_neuron_idx];

        uint cell_value = binary_search_batched(keys_flat, values_flat, mem_start, mem_count, address, params.default_cell_value);

        if (params.memory_mode == MEM_MODE_QUAD_WEIGHTED) {
            weighted_sum += QUAD_WEIGHTS[cell_value];
        } else if (params.memory_mode == MEM_MODE_QUAD_BINARY) {
            if (cell_value >= 2) count_true++;
        } else {
            if (cell_value == CELL_TRUE) count_true++;
            else if (cell_value == CELL_EMPTY) count_empty++;
        }
    }

    float prob = compute_prob_batched(count_true, count_empty, weighted_sum,
        params.neurons_per_cluster, params.memory_mode, params.empty_value);

    uint output_idx = genome_idx * params.num_examples * params.num_clusters
                    + example_idx * params.num_clusters
                    + cluster_idx;
    probs_out[output_idx] = prob;
}

//
// Batched Per-Example Kernel (alternative)
//
// Grid: (num_examples, num_genomes)
// Each thread computes ALL clusters for one (genome, example) pair.
// Better for small num_clusters or high memory latency.
//
kernel void batched_sparse_forward_per_example(
    device const uchar* input_bits_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const ulong* keys_flat [[buffer(2)]],
    device const uchar* values_flat [[buffer(3)]],
    device const uint* offsets_flat [[buffer(4)]],
    device const uint* counts_flat [[buffer(5)]],
    device const uint* genome_key_offsets [[buffer(6)]],
    constant BatchedSparseParams& params [[buffer(7)]],
    device float* probs_out [[buffer(8)]],
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint example_idx = thread_pos.x;
    uint genome_idx = thread_pos.y;

    if (example_idx >= params.num_examples) return;
    if (genome_idx >= params.num_genomes) return;

    device const uchar* input_bits = input_bits_flat + example_idx * params.total_input_bits;

    uint genome_neuron_offset = genome_idx * params.num_neurons;
    uint genome_conn_offset = genome_idx * params.num_neurons * params.bits_per_neuron;
    uint genome_key_offset = genome_key_offsets[genome_idx];

    uint output_base = genome_idx * params.num_examples * params.num_clusters
                     + example_idx * params.num_clusters;

    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint count_true = 0;
        uint count_empty = 0;
        float weighted_sum = 0.0f;
        uint start_neuron = cluster_idx * params.neurons_per_cluster;

        for (uint neuron_offset = 0; neuron_offset < params.neurons_per_cluster; neuron_offset++) {
            uint local_neuron_idx = start_neuron + neuron_offset;
            uint global_neuron_idx = genome_neuron_offset + local_neuron_idx;

            device const int* connections = connections_flat + genome_conn_offset + local_neuron_idx * params.bits_per_neuron;

            ulong address = compute_address_batched(input_bits, connections, params.bits_per_neuron);

            uint mem_start = offsets_flat[global_neuron_idx] + genome_key_offset;
            uint mem_count = counts_flat[global_neuron_idx];

            uint cell_value = binary_search_batched(keys_flat, values_flat, mem_start, mem_count, address, params.default_cell_value);

            if (params.memory_mode == MEM_MODE_QUAD_WEIGHTED) {
                weighted_sum += QUAD_WEIGHTS[cell_value];
            } else if (params.memory_mode == MEM_MODE_QUAD_BINARY) {
                if (cell_value >= 2) count_true++;
            } else {
                if (cell_value == CELL_TRUE) count_true++;
                else if (cell_value == CELL_EMPTY) count_empty++;
            }
        }

        probs_out[output_base + cluster_idx] = compute_prob_batched(count_true, count_empty, weighted_sum,
            params.neurons_per_cluster, params.memory_mode, params.empty_value);
    }
}
