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

// Parameters passed from CPU
struct RAMLMParams {
    uint num_examples;
    uint total_input_bits;
    uint num_neurons;
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_clusters;
    uint words_per_neuron;
    float empty_value;  // Value for EMPTY cells (0.0 = abstain, 0.5 = uncertain)
};

// Compute memory address for a neuron given input bits
// Uses MSB-first addressing (matches Python)
inline uint compute_address(
    device const uchar* input_bits,    // [total_input_bits] for this example
    device const int* connections,     // [bits_per_neuron] for this neuron
    uint bits_per_neuron
) {
    uint address = 0;
    for (uint i = 0; i < bits_per_neuron; i++) {
        int conn_idx = connections[i];
        if (conn_idx >= 0 && input_bits[conn_idx]) {
            // MSB-first: bit 0 is most significant
            address |= (1u << (bits_per_neuron - 1 - i));
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
    return (uint(word) >> (cell_idx * BITS_PER_CELL)) & CELL_MASK;
}

//
// Forward Pass Kernel
//
// Each thread computes the probability for one (example, cluster) pair.
// Grid: (num_clusters, num_examples)
//
kernel void ramlm_forward_pass(
    device const uchar* input_bits_flat [[buffer(0)]],   // [num_examples * total_input_bits]
    device const int* connections_flat [[buffer(1)]],     // [num_neurons * bits_per_neuron]
    device const long* memory_words [[buffer(2)]],        // [num_neurons * words_per_neuron]
    constant RAMLMParams& params [[buffer(3)]],
    device float* probs_out [[buffer(4)]],                // [num_examples * num_clusters]
    uint2 thread_pos [[thread_position_in_grid]]          // (cluster_idx, example_idx)
) {
    uint cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    // Bounds check
    if (cluster_idx >= params.num_clusters) return;
    if (example_idx >= params.num_examples) return;

    // Get input bits for this example
    device const uchar* input_bits = input_bits_flat + example_idx * params.total_input_bits;

    // Count TRUE and EMPTY cells for this cluster
    uint count_true = 0;
    uint count_empty = 0;

    uint start_neuron = cluster_idx * params.neurons_per_cluster;

    for (uint neuron_offset = 0; neuron_offset < params.neurons_per_cluster; neuron_offset++) {
        uint neuron_idx = start_neuron + neuron_offset;

        // Get connections for this neuron
        device const int* connections = connections_flat + neuron_idx * params.bits_per_neuron;

        // Compute address
        uint address = compute_address(input_bits, connections, params.bits_per_neuron);

        // Read cell value
        uint cell_value = read_cell(memory_words, neuron_idx, address, params.words_per_neuron);

        if (cell_value == CELL_TRUE) {
            count_true++;
        } else if (cell_value == CELL_EMPTY) {
            count_empty++;
        }
    }

    // Probability = (count_true + empty_value * count_empty) / neurons_per_cluster
    float prob = (float(count_true) + params.empty_value * float(count_empty)) / float(params.neurons_per_cluster);

    // Output index: [example_idx, cluster_idx]
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
    device const uchar* input_bits_flat [[buffer(0)]],
    device const int* connections_flat [[buffer(1)]],
    device const long* memory_words [[buffer(2)]],
    constant RAMLMParams& params [[buffer(3)]],
    device float* probs_out [[buffer(4)]],
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

            uint address = compute_address(input_bits, connections, params.bits_per_neuron);
            uint cell_value = read_cell(memory_words, neuron_idx, address, params.words_per_neuron);

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
