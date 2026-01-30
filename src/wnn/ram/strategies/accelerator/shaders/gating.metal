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
// - Input bits: [batch_size * total_input_bits] u8 (0 or 1)
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
    uint total_input_bits;
    uint vote_threshold;
    uint address_space_size;
    uint batch_size;
    uint _padding;  // Align to 8 bytes
};

// Compute RAM address from input bits using connectivity
inline uint compute_address(
    device const int* connections,  // Pointer to this neuron's connectivity
    uint bits_per_neuron,
    device const uint8_t* input_bits,
    uint total_input_bits
) {
    uint address = 0;
    for (uint b = 0; b < bits_per_neuron; b++) {
        int conn_idx = connections[b];
        if (conn_idx >= 0 && uint(conn_idx) < total_input_bits) {
            if (input_bits[conn_idx] != 0) {
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
    device const uint8_t* input_bits [[buffer(2)]],    // [batch_size * total_input_bits]
    constant GatingParams& params [[buffer(3)]],
    device float* output_gates [[buffer(4)]],          // [batch_size * num_clusters]
    uint2 thread_pos [[thread_position_in_grid]]       // (cluster_idx, batch_idx)
) {
    uint cluster_idx = thread_pos.x;
    uint batch_idx = thread_pos.y;

    // Bounds check
    if (cluster_idx >= params.num_clusters) return;
    if (batch_idx >= params.batch_size) return;

    // Get input bits for this example
    device const uint8_t* example_input = input_bits + batch_idx * params.total_input_bits;

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
            example_input,
            params.total_input_bits
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
    device const uint8_t* input_bits [[buffer(2)]],    // [batch_size * total_input_bits]
    constant GatingParams& params [[buffer(3)]],
    device float* output_gates [[buffer(4)]],          // [batch_size * num_clusters]
    uint batch_idx [[thread_position_in_grid]]
) {
    // Bounds check
    if (batch_idx >= params.batch_size) return;

    // Get input bits for this example
    device const uint8_t* example_input = input_bits + batch_idx * params.total_input_bits;

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
                example_input,
                params.total_input_bits
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
