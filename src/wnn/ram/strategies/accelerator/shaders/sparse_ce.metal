//
// Metal Compute Shader for Sparse RAMLM Forward Pass with CE/Accuracy Computation
//
// Instead of returning all probabilities (10GB for 50K×50K), this kernel computes
// cross-entropy and accuracy DIRECTLY on the GPU and returns just 2 values.
//
// Key optimization: Avoids massive CPU←GPU data transfer by reducing on GPU.
//
// Memory layout:
// - packed_input: [num_examples * words_per_example] (packed u64)
// - connections: [num_neurons * bits_per_neuron]
// - keys: Sorted addresses for all neurons
// - values: Corresponding cell values
// - offsets: [num_neurons] - start index per neuron
// - counts: [num_neurons] - count of entries per neuron
// - targets: [num_examples] - target cluster for each example
// - ce_out: [num_examples] - cross-entropy per example (reduced later)
// - correct_out: [num_examples] - 1 if correct, 0 otherwise
//

#include <metal_stdlib>
using namespace metal;

constant uint CELL_FALSE = 0;
constant uint CELL_TRUE = 1;
constant uint CELL_EMPTY = 2;

struct SparseCEParams {
    uint num_examples;
    uint words_per_example;     // ceil(total_input_bits / 64) — stride for packed u64 input
    uint num_neurons;           // Total neurons (all clusters)
    uint bits_per_neuron;
    uint neurons_per_cluster;
    uint num_clusters;
    float empty_value;
};

// Compute memory address (same as sparse_forward.metal)
inline ulong compute_address_ce(
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

// Binary search (same as sparse_forward.metal)
inline uint binary_search_ce(
    device const ulong* keys,
    device const uchar* values,
    uint start,
    uint count,
    ulong address
) {
    if (count == 0) return CELL_EMPTY;

    uint left = 0;
    uint right = count;

    while (left < right) {
        uint mid = left + (right - left) / 2;
        ulong key = keys[start + mid];

        if (key == address) {
            return values[start + mid];
        } else if (key < address) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return CELL_EMPTY;
}

//
// Sparse Forward Pass with CE Computation Kernel
//
// Each thread processes ONE EXAMPLE:
// 1. Computes probability for ALL clusters
// 2. Applies softmax normalization
// 3. Computes cross-entropy for target cluster
// 4. Determines if prediction is correct
//
// Grid: (num_examples)
// Output: ce_out[example], correct_out[example]
//
kernel void sparse_forward_with_ce(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections [[buffer(1)]],
    device const ulong* keys [[buffer(2)]],
    device const uchar* values [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device const uint* counts [[buffer(5)]],
    device const int* targets [[buffer(6)]],
    constant SparseCEParams& params [[buffer(7)]],
    device float* ce_out [[buffer(8)]],
    device uint* correct_out [[buffer(9)]],
    uint example_idx [[thread_position_in_grid]]
) {
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    int target_cluster = targets[example_idx];

    // Compute raw scores for all clusters
    // Using threadgroup memory would be better but this is simpler
    float max_score = -1e10f;
    float target_score = 0.0f;
    uint predicted_cluster = 0;
    float predicted_score = -1e10f;

    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint start_neuron = cluster_idx * params.neurons_per_cluster;

        uint count_true = 0;
        uint count_empty = 0;

        for (uint n = 0; n < params.neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* conn = connections + neuron_idx * params.bits_per_neuron;

            ulong address = compute_address_ce(packed_input, conn, params.bits_per_neuron);

            uint mem_start = offsets[neuron_idx];
            uint mem_count = counts[neuron_idx];

            uint cell_value = binary_search_ce(keys, values, mem_start, mem_count, address);

            if (cell_value == CELL_TRUE) {
                count_true++;
            } else if (cell_value == CELL_EMPTY) {
                count_empty++;
            }
        }

        float score = (float(count_true) + params.empty_value * float(count_empty)) / float(params.neurons_per_cluster);

        if (score > max_score) {
            max_score = score;
        }
        if (score > predicted_score) {
            predicted_score = score;
            predicted_cluster = cluster_idx;
        }
        if (cluster_idx == uint(target_cluster)) {
            target_score = score;
        }
    }

    // Softmax computation: exp(score - max) / sum(exp(score - max))
    float sum_exp = 0.0f;
    float target_exp = exp(target_score - max_score);

    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint start_neuron = cluster_idx * params.neurons_per_cluster;

        uint count_true = 0;
        uint count_empty = 0;

        for (uint n = 0; n < params.neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* conn = connections + neuron_idx * params.bits_per_neuron;

            ulong address = compute_address_ce(packed_input, conn, params.bits_per_neuron);

            uint mem_start = offsets[neuron_idx];
            uint mem_count = counts[neuron_idx];

            uint cell_value = binary_search_ce(keys, values, mem_start, mem_count, address);

            if (cell_value == CELL_TRUE) {
                count_true++;
            } else if (cell_value == CELL_EMPTY) {
                count_empty++;
            }
        }

        float score = (float(count_true) + params.empty_value * float(count_empty)) / float(params.neurons_per_cluster);
        sum_exp += exp(score - max_score);
    }

    // Cross-entropy: -log(target_prob)
    float target_prob = target_exp / sum_exp;
    float ce = -log(target_prob + 1e-10f);

    ce_out[example_idx] = ce;
    correct_out[example_idx] = (predicted_cluster == uint(target_cluster)) ? 1 : 0;
}

//
// Optimized version using ONLINE SOFTMAX
// Computes softmax normalization in a SINGLE PASS using numerically stable algorithm
//
// Online softmax algorithm:
//   max_so_far = -inf
//   sum_exp_so_far = 0
//   for each score:
//     if score > max_so_far:
//       sum_exp_so_far *= exp(max_so_far - score)  // rescale
//       max_so_far = score
//     sum_exp_so_far += exp(score - max_so_far)
//
// This computes the same result as two-pass softmax but in ONE pass!
//
kernel void sparse_forward_with_ce_online(
    device const ulong* packed_input_flat [[buffer(0)]],
    device const int* connections [[buffer(1)]],
    device const ulong* keys [[buffer(2)]],
    device const uchar* values [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device const uint* counts [[buffer(5)]],
    device const int* targets [[buffer(6)]],
    constant SparseCEParams& params [[buffer(7)]],
    device float* ce_out [[buffer(8)]],
    device uint* correct_out [[buffer(9)]],
    uint example_idx [[thread_position_in_grid]]
) {
    if (example_idx >= params.num_examples) return;

    device const ulong* packed_input = packed_input_flat + example_idx * params.words_per_example;
    int target_cluster = targets[example_idx];

    // Online softmax state
    float max_score = -1e10f;
    float sum_exp = 0.0f;

    // Track target and predicted
    float target_score = 0.0f;
    uint predicted_cluster = 0;
    float predicted_score = -1e10f;

    // Single pass over all clusters
    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint start_neuron = cluster_idx * params.neurons_per_cluster;

        uint count_true = 0;
        uint count_empty = 0;

        for (uint n = 0; n < params.neurons_per_cluster; n++) {
            uint neuron_idx = start_neuron + n;
            device const int* conn = connections + neuron_idx * params.bits_per_neuron;

            ulong address = compute_address_ce(packed_input, conn, params.bits_per_neuron);

            uint mem_start = offsets[neuron_idx];
            uint mem_count = counts[neuron_idx];

            uint cell_value = binary_search_ce(keys, values, mem_start, mem_count, address);

            if (cell_value == CELL_TRUE) {
                count_true++;
            } else if (cell_value == CELL_EMPTY) {
                count_empty++;
            }
        }

        float score = (float(count_true) + params.empty_value * float(count_empty)) / float(params.neurons_per_cluster);

        // Track predicted (argmax)
        if (score > predicted_score) {
            predicted_score = score;
            predicted_cluster = cluster_idx;
        }

        // Track target score
        if (cluster_idx == uint(target_cluster)) {
            target_score = score;
        }

        // Online softmax update
        if (score > max_score) {
            // New max - rescale previous sum
            sum_exp = sum_exp * exp(max_score - score);
            max_score = score;
        }
        sum_exp += exp(score - max_score);
    }

    // Compute cross-entropy: -log(exp(target - max) / sum_exp)
    float target_exp = exp(target_score - max_score);
    float target_prob = target_exp / sum_exp;
    float ce = -log(target_prob + 1e-10f);

    ce_out[example_idx] = ce;
    correct_out[example_idx] = (predicted_cluster == uint(target_cluster)) ? 1 : 0;
}

//
// Parallel reduction kernel to sum CE values across examples
// Grid: (num_workgroups)
// Each workgroup reduces a chunk of examples
//
kernel void reduce_ce_sum(
    device const float* ce_in [[buffer(0)]],
    device const uint* correct_in [[buffer(1)]],
    device atomic_float* total_ce [[buffer(2)]],
    device atomic_uint* total_correct [[buffer(3)]],
    constant uint& num_examples [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Each thread sums a strided chunk
    float local_ce = 0.0f;
    uint local_correct = 0;

    for (uint i = tid; i < num_examples; i += group_size * 1024) {
        local_ce += ce_in[i];
        local_correct += correct_in[i];
    }

    // Atomic add to global totals
    atomic_fetch_add_explicit(total_ce, local_ce, memory_order_relaxed);
    atomic_fetch_add_explicit(total_correct, local_correct, memory_order_relaxed);
}
