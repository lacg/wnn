//
// Metal Compute Shader for RAMLM Forward Pass
//
// GPU-accelerated forward pass for RAM Language Models.
// Each thread computes probabilities for one (example, cluster) pair.
//
// Memory format:
// - 2-bit packed cells: FALSE=0, TRUE=1, EMPTY=2
// - 31 cells per 64-bit word (62 bits used)
//
// This achieves massive parallelism: num_examples * num_clusters threads
// can run simultaneously on Apple Silicon GPUs (40+ cores on M4 Max).
//

#include <metal_stdlib>
using namespace metal;

// Memory cell constants (2-bit values)
constant uint CELL_FALSE = 0;
constant uint CELL_TRUE = 1;
constant uint CELL_EMPTY = 2;

// Bit-packing constants
constant uint BITS_PER_CELL = 2;
constant uint CELLS_PER_WORD = 31;  // 62 bits / 2 = 31 cells
constant uint CELL_MASK = 0x3;      // 2-bit mask

// Memory mode constants
constant uint MEM_MODE_TERNARY = 0;
constant uint MEM_MODE_QUAD_BINARY = 1;
constant uint MEM_MODE_QUAD_WEIGHTED = 2;

// Quad weights for QUAD_WEIGHTED mode (lookup table avoids branching)
constant float QUAD_WEIGHTS[4] = {0.0f, 0.25f, 0.75f, 1.0f};

// Parameters passed from CPU
struct RAMLMParams {
    uint num_examples;
    uint words_per_example;  // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_neurons;
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_clusters;
    uint words_per_neuron;
    float empty_value;  // Value for EMPTY cells (0.0 = abstain, 0.5 = uncertain)
    uint memory_mode;   // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
};

// Compute memory address for a neuron given packed u64 input bits
// Uses MSB-first addressing (matches Python)
inline uint compute_address(
    device const ulong* packed_input,  // [words_per_example] packed u64 words for this example
    device const int* connections,     // [bits_per_neuron] for this neuron
    uint bits_per_neuron
) {
    uint address = 0;
    for (uint i = 0; i < bits_per_neuron; i++) {
        int conn_idx = connections[i];
        if (conn_idx >= 0) {
            uint word_idx = uint(conn_idx) / 64;
            uint bit_idx = uint(conn_idx) % 64;
            if ((packed_input[word_idx] >> bit_idx) & 1uL) {
                // MSB-first: bit 0 is most significant
                address |= (1u << (bits_per_neuron - 1 - i));
            }
        }
    }
    return address;
}

// Read a 2-bit cell from packed memory
inline uint read_cell(
    device const long* memory_words,
    uint neuron_idx,
    uint address,
    uint words_per_neuron
) {
    uint word_idx = address / CELLS_PER_WORD;
    uint cell_idx = address % CELLS_PER_WORD;
    uint word_offset = neuron_idx * words_per_neuron + word_idx;
    long word = memory_words[word_offset];
    // Shift in 64-bit first, then truncate (uint(word) would drop upper 32 bits)
    return uint(word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK;
}

//
// Mode-switched accumulation over neurons in a cluster.
// Returns probability based on memory_mode:
//   TERNARY: P = (count_TRUE + empty_value * count_EMPTY) / neurons
//   QUAD_BINARY: P = count(cell >= 2) / neurons
//   QUAD_WEIGHTED: P = sum(QUAD_WEIGHTS[cell]) / neurons
//
inline float accumulate_dense(
    device const ulong* packed_input,
    device const int* connections_flat,
    device const long* memory_words,
    uint start_neuron,
    uint neurons_per_cluster,
    uint bits_per_neuron,
    uint words_per_neuron,
    uint memory_mode,
    float empty_value
) {
    if (memory_mode == MEM_MODE_QUAD_WEIGHTED) {
        float weighted_sum = 0.0f;
        for (uint n = 0; n < neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* connections = connections_flat + neuron_idx * bits_per_neuron;
            uint address = compute_address(packed_input, connections, bits_per_neuron);
            uint cell = read_cell(memory_words, neuron_idx, address, words_per_neuron);
            weighted_sum += QUAD_WEIGHTS[cell];
        }
        return weighted_sum / float(neurons_per_cluster);
    } else if (memory_mode == MEM_MODE_QUAD_BINARY) {
        uint count = 0;
        for (uint n = 0; n < neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* connections = connections_flat + neuron_idx * bits_per_neuron;
            uint address = compute_address(packed_input, connections, bits_per_neuron);
            uint cell = read_cell(memory_words, neuron_idx, address, words_per_neuron);
            if (cell >= 2) count++;  // QUAD_WEAK_TRUE or QUAD_TRUE
        }
        return float(count) / float(neurons_per_cluster);
    } else {
        // TERNARY (default)
        uint count_true = 0;
        uint count_empty = 0;
        for (uint n = 0; n < neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* connections = connections_flat + neuron_idx * bits_per_neuron;
            uint address = compute_address(packed_input, connections, bits_per_neuron);
            uint cell = read_cell(memory_words, neuron_idx, address, words_per_neuron);
            if (cell == CELL_TRUE) count_true++;
            else if (cell == CELL_EMPTY) count_empty++;
        }
        return (float(count_true) + empty_value * float(count_empty)) / float(neurons_per_cluster);
    }
}

//
// Forward Pass Kernel
//
// Each thread computes the probability for one (example, cluster) pair.
// Grid: (num_clusters, num_examples)
//
kernel void ramlm_forward_pass(
    device const ulong* packed_input_flat [[buffer(0)]],  // [num_examples * words_per_example]
    device const int* connections_flat [[buffer(1)]],      // [num_neurons * bits_per_neuron]
    device const long* memory_words [[buffer(2)]],         // [num_neurons * words_per_neuron]
    constant RAMLMParams& params [[buffer(3)]],
    device float* probs_out [[buffer(4)]],                 // [num_examples * num_clusters]
    uint2 thread_pos [[thread_position_in_grid]]           // (cluster_idx, example_idx)
) {
    uint cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (cluster_idx >= params.num_clusters) return;
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    uint start_neuron = cluster_idx * params.neurons_per_cluster;

    float prob = accumulate_dense(
        packed_input, connections_flat, memory_words,
        start_neuron, params.neurons_per_cluster,
        params.bits_per_neuron, params.words_per_neuron,
        params.memory_mode, params.empty_value
    );

    uint output_idx = example_idx * params.num_clusters + cluster_idx;
    probs_out[output_idx] = prob;
}

//
// Batch Forward Pass Kernel (alternative - one thread per example)
//
// Each thread computes probabilities for ALL clusters for one example.
// Better for memory coalescing when neurons_per_cluster is small.
// Grid: (num_examples)
//
kernel void ramlm_forward_pass_per_example(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const long* memory_words [[buffer(2)]],
    constant RAMLMParams& params [[buffer(3)]],
    device float* probs_out [[buffer(4)]],
    uint example_idx [[thread_position_in_grid]]
) {
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    device float* example_probs = probs_out + example_idx * params.num_clusters;

    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint start_neuron = cluster_idx * params.neurons_per_cluster;

        example_probs[cluster_idx] = accumulate_dense(
            packed_input, connections_flat, memory_words,
            start_neuron, params.neurons_per_cluster,
            params.bits_per_neuron, params.words_per_neuron,
            params.memory_mode, params.empty_value
        );
    }
}

//
// Dense Forward Pass to Shared Buffer Kernel
//
// Writes scores directly to a shared buffer at specified cluster offsets.
// This avoids GPU→CPU→GPU transfer when computing CE across multiple groups.
//
// The cluster_ids buffer maps local cluster index → global cluster ID.
// Output is written to: shared_buffer[example_idx * total_clusters + cluster_ids[local_cluster]]
//
struct DenseToBufferParams {
    uint num_examples;
    uint words_per_example;     // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_neurons;           // Neurons in this group
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_group_clusters;    // Clusters in this group
    uint total_clusters;        // Total clusters across all groups
    uint words_per_neuron;
    float empty_value;
    uint memory_mode;           // 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
};

kernel void ramlm_forward_to_buffer(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const long* memory_words [[buffer(2)]],
    device const uint* cluster_ids [[buffer(3)]],      // Maps local → global cluster ID
    constant DenseToBufferParams& params [[buffer(4)]],
    device float* shared_buffer [[buffer(5)]],         // Shared across all groups
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint local_cluster = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (local_cluster >= params.num_group_clusters) return;
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;

    uint start_neuron = local_cluster * params.neurons_per_cluster;

    float prob = accumulate_dense(
        packed_input, connections_flat, memory_words,
        start_neuron, params.neurons_per_cluster,
        params.bits_per_neuron, params.words_per_neuron,
        params.memory_mode, params.empty_value
    );

    // Write to global position in shared buffer
    uint global_cluster = cluster_ids[local_cluster];
    uint output_idx = example_idx * params.total_clusters + global_cluster;
    shared_buffer[output_idx] = prob;
}

//
// Top-K Selection Kernel (optional optimization)
//
// After computing all probabilities, find top-k clusters.
// Uses parallel reduction for efficiency.
// Grid: (num_examples)
//
kernel void ramlm_topk_selection(
    device const float* probs [[buffer(0)]],           // [num_examples * num_clusters]
    constant RAMLMParams& params [[buffer(1)]],
    constant uint& top_k [[buffer(2)]],
    device uint* topk_indices [[buffer(3)]],           // [num_examples * top_k]
    device float* topk_probs [[buffer(4)]],            // [num_examples * top_k]
    uint example_idx [[thread_position_in_grid]]
) {
    if (example_idx >= params.num_examples) return;

    device const float* example_probs = probs + example_idx * params.num_clusters;
    device uint* example_topk_idx = topk_indices + example_idx * top_k;
    device float* example_topk_prob = topk_probs + example_idx * top_k;

    // Simple O(k*n) selection (good for small k)
    for (uint k = 0; k < top_k; k++) {
        float max_prob = -1.0f;
        uint max_idx = 0;

        for (uint c = 0; c < params.num_clusters; c++) {
            float prob = example_probs[c];

            // Check if already selected
            bool already_selected = false;
            for (uint prev_k = 0; prev_k < k; prev_k++) {
                if (example_topk_idx[prev_k] == c) {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && prob > max_prob) {
                max_prob = prob;
                max_idx = c;
            }
        }

        example_topk_idx[k] = max_idx;
        example_topk_prob[k] = max_prob;
    }
}
