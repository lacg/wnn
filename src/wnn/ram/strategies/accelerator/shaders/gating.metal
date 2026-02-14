//
// Metal Compute Shader for RAM-based Gating Forward Pass
//
// Evaluates binary gates using RAM neurons with majority voting.
// Each cluster has `neurons_per_gate` neurons, and the gate opens (1.0)
// if at least `vote_threshold` neurons output TRUE.
//
// Memory layout:
// - Connections: [total_neurons * bits_per_neuron] i32 indices
// - Memory: [total_neurons * address_space_size] u8 cells (EMPTY=0, FALSE=1, TRUE=2)
// - Packed input: [batch_size * words_per_example] packed u64 (LSB-first)
// - Output gates: [batch_size * num_clusters] f32 (0.0 or 1.0)
//

#include <metal_stdlib>
using namespace metal;

// Cell values (must match Rust constants)
constant uint8_t CELL_EMPTY = 0;
constant uint8_t CELL_FALSE = 1;
constant uint8_t CELL_TRUE = 2;

// Parameters struct (must match Rust repr(C))
struct GatingParams {
    uint num_clusters;
    uint neurons_per_gate;
    uint bits_per_neuron;
    uint words_per_example;  // ceil(total_input_bits / 64) — stride for packed u64 input
    uint vote_threshold;
    uint address_space_size;
    uint batch_size;
    uint _padding;  // Align to 8 bytes
};

// Compute RAM address from packed u64 input bits using connectivity
inline uint compute_address(
    device const int* connections,  // Pointer to this neuron's connectivity
    uint bits_per_neuron,
    device const ulong* packed_input
) {
    uint address = 0;
    for (uint b = 0; b < bits_per_neuron; b++) {
        int conn_idx = connections[b];
        if (conn_idx >= 0) {
            uint word_idx = uint(conn_idx) / 64;
            uint bit_idx = uint(conn_idx) % 64;
            if ((packed_input[word_idx] >> bit_idx) & 1uL) {
                address |= (1u << b);
            }
        }
    }
    return address;
}

//
// Kernel: One thread per (batch_example, cluster) pair
// Computes the gate value for one cluster on one example
//
kernel void gating_forward(
    device const int* connections [[buffer(0)]],       // [total_neurons * bits_per_neuron]
    device const uint8_t* memory [[buffer(1)]],        // [total_neurons * address_space_size]
    device const ulong* packed_input_flat [[buffer(2)]],    // [batch_size * words_per_example]
    constant GatingParams& params [[buffer(3)]],
    device float* output_gates [[buffer(4)]],          // [batch_size * num_clusters]
    uint2 thread_pos [[thread_position_in_grid]]       // (cluster_idx, batch_idx)
) {
    uint cluster_idx = thread_pos.x;
    uint batch_idx = thread_pos.y;

    // Bounds check
    if (cluster_idx >= params.num_clusters) return;
    if (batch_idx >= params.batch_size) return;

    // Get packed input for this example
    device const ulong* packed_input = packed_input_flat + batch_idx * params.words_per_example;

    // First neuron index for this cluster
    uint neuron_start = cluster_idx * params.neurons_per_gate;

    // Count TRUE neurons
    uint true_count = 0;

    for (uint n = 0; n < params.neurons_per_gate; n++) {
        uint neuron_idx = neuron_start + n;

        // Get connectivity for this neuron
        device const int* neuron_conn = connections + neuron_idx * params.bits_per_neuron;

        // Compute address
        uint address = compute_address(
            neuron_conn,
            params.bits_per_neuron,
            packed_input
        );

        // Read memory cell
        uint mem_idx = neuron_idx * params.address_space_size + address;
        uint8_t cell = memory[mem_idx];

        // TRUE counts toward majority
        if (cell == CELL_TRUE) {
            true_count++;
        }
    }

    // Majority vote: gate = 1 if true_count >= threshold
    float gate = (true_count >= params.vote_threshold) ? 1.0f : 0.0f;

    // Write output
    uint output_idx = batch_idx * params.num_clusters + cluster_idx;
    output_gates[output_idx] = gate;
}

//
// Kernel: Per-example variant (one thread per batch example)
// Better for large num_clusters when memory bandwidth matters
//
kernel void gating_forward_per_example(
    device const int* connections [[buffer(0)]],       // [total_neurons * bits_per_neuron]
    device const uint8_t* memory [[buffer(1)]],        // [total_neurons * address_space_size]
    device const ulong* packed_input_flat [[buffer(2)]],    // [batch_size * words_per_example]
    constant GatingParams& params [[buffer(3)]],
    device float* output_gates [[buffer(4)]],          // [batch_size * num_clusters]
    uint batch_idx [[thread_position_in_grid]]
) {
    // Bounds check
    if (batch_idx >= params.batch_size) return;

    // Get packed input for this example
    device const ulong* packed_input = packed_input_flat + batch_idx * params.words_per_example;

    // Output pointer for this example
    device float* example_output = output_gates + batch_idx * params.num_clusters;

    // Process all clusters for this example
    for (uint cluster_idx = 0; cluster_idx < params.num_clusters; cluster_idx++) {
        uint neuron_start = cluster_idx * params.neurons_per_gate;
        uint true_count = 0;

        for (uint n = 0; n < params.neurons_per_gate; n++) {
            uint neuron_idx = neuron_start + n;

            // Get connectivity for this neuron
            device const int* neuron_conn = connections + neuron_idx * params.bits_per_neuron;

            // Compute address
            uint address = compute_address(
                neuron_conn,
                params.bits_per_neuron,
                packed_input
            );

            // Read memory cell
            uint mem_idx = neuron_idx * params.address_space_size + address;
            uint8_t cell = memory[mem_idx];

            if (cell == CELL_TRUE) {
                true_count++;
            }
        }

        // Majority vote
        example_output[cluster_idx] = (true_count >= params.vote_threshold) ? 1.0f : 0.0f;
    }
}

//
// Kernel: Apply gates to scores (element-wise multiplication)
// Useful when gating and base evaluation run on separate GPU passes
//
kernel void apply_gates_gpu(
    device const float* scores [[buffer(0)]],         // [batch_size * num_clusters]
    device const float* gates [[buffer(1)]],          // [batch_size * num_clusters]
    device float* output [[buffer(2)]],               // [batch_size * num_clusters]
    constant uint& total_elements [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= total_elements) return;
    output[idx] = scores[idx] * gates[idx];
}

//
// Kernel: Train gating neurons
// One thread per (batch_example, cluster) pair
// Writes to memory using atomic operations to handle conflicts
//
// For each cluster:
//   - If target_gate[cluster] == 1 (true), write CELL_TRUE to all neurons
//   - If target_gate[cluster] == 0 (false), write CELL_FALSE to all neurons
//
// Note: We use atomic operations because multiple examples may map to the same
// memory address. The final value will be determined by whichever example
// maps there last, which is acceptable for RAM semantics.
//
kernel void gating_train(
    device const int* connections [[buffer(0)]],       // [total_neurons * bits_per_neuron]
    device atomic_uint* memory [[buffer(1)]],          // [total_neurons * address_space_size / 4] (packed u8s as u32)
    device const ulong* packed_input_flat [[buffer(2)]],    // [batch_size * words_per_example]
    device const uint8_t* target_gates [[buffer(3)]],  // [batch_size * num_clusters] (0 or 1)
    constant GatingParams& params [[buffer(4)]],
    uint2 thread_pos [[thread_position_in_grid]]       // (cluster_idx, batch_idx)
) {
    uint cluster_idx = thread_pos.x;
    uint batch_idx = thread_pos.y;

    // Bounds check
    if (cluster_idx >= params.num_clusters) return;
    if (batch_idx >= params.batch_size) return;

    // Get packed input for this example
    device const ulong* packed_input = packed_input_flat + batch_idx * params.words_per_example;

    // Get target gate for this cluster
    uint target_idx = batch_idx * params.num_clusters + cluster_idx;
    uint8_t target = target_gates[target_idx];
    uint8_t target_cell = (target != 0) ? CELL_TRUE : CELL_FALSE;

    // First neuron index for this cluster
    uint neuron_start = cluster_idx * params.neurons_per_gate;

    // Train all neurons for this cluster
    for (uint n = 0; n < params.neurons_per_gate; n++) {
        uint neuron_idx = neuron_start + n;

        // Get connectivity for this neuron
        device const int* neuron_conn = connections + neuron_idx * params.bits_per_neuron;

        // Compute address
        uint address = compute_address(
            neuron_conn,
            params.bits_per_neuron,
            packed_input
        );

        // Memory index (byte offset)
        uint mem_byte_idx = neuron_idx * params.address_space_size + address;

        // For atomic access, we work with 4-byte (u32) chunks
        // Each u32 contains 4 u8 cells
        uint mem_word_idx = mem_byte_idx / 4;
        uint byte_offset = mem_byte_idx % 4;
        uint shift = byte_offset * 8;
        uint mask = 0xFFu << shift;
        uint new_value = uint(target_cell) << shift;

        // Atomic read-modify-write to update just our byte
        uint old_val = atomic_load_explicit(&memory[mem_word_idx], memory_order_relaxed);
        uint updated;
        do {
            updated = (old_val & ~mask) | new_value;
        } while (!atomic_compare_exchange_weak_explicit(
            &memory[mem_word_idx],
            &old_val,
            updated,
            memory_order_relaxed,
            memory_order_relaxed
        ));
    }
}
