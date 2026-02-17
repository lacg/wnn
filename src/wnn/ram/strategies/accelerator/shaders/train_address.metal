#include <metal_stdlib>
using namespace metal;

// =============================================================================
// GPU Training: Address Computation Kernel
// =============================================================================
//
// Computes memory addresses for all (neuron, example) pairs in parallel.
// The CPU then uses these pre-computed addresses for vote accumulation
// (TERNARY) or sequential nudging (QUAD modes).
//
// This offloads the expensive part (random memory lookups into packed input)
// to the GPU, while keeping the sequential write logic on the CPU.

// Per-neuron metadata (matches Rust NeuronTrainMeta)
struct NeuronTrainMeta {
    uint bits;          // Number of address bits for this neuron
    uint conn_offset;   // Offset into connections array
};

// Kernel parameters (matches Rust TrainAddressParams)
struct TrainAddressParams {
    uint num_examples;
    uint words_per_example;
    uint total_neurons;
    uint _pad;
};

// Compute memory address from packed u64 input bits (MSB-first addressing).
// Identical logic to compute_address() in ramlm.metal.
inline uint compute_address(
    device const ulong* packed_input,  // [words_per_example] for this example
    device const int* connections,     // [bits] connections for this neuron
    uint bits
) {
    uint address = 0;
    for (uint i = 0; i < bits; i++) {
        int conn_idx = connections[i];
        if (conn_idx >= 0) {
            uint word_idx = uint(conn_idx) / 64;
            uint bit_idx = uint(conn_idx) % 64;
            // Shift in 64-bit first to avoid truncation bug (see MEMORY.md)
            if ((packed_input[word_idx] >> bit_idx) & 1uL) {
                address |= (1u << (bits - 1 - i));
            }
        }
    }
    return address;
}

// Grid: (total_neurons, num_examples, 1)
// Output: address_buffer[neuron_idx * num_examples + example_idx] = u32 address
kernel void train_compute_addresses(
    device const ulong* packed_input       [[buffer(0)]],  // [num_examples * words_per_example]
    device const int* connections          [[buffer(1)]],  // flattened connections for all neurons
    device const NeuronTrainMeta* neurons  [[buffer(2)]],  // [total_neurons]
    constant TrainAddressParams& params    [[buffer(3)]],
    device uint* address_buffer            [[buffer(4)]],  // [total_neurons * num_examples]
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint neuron_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (neuron_idx >= params.total_neurons || example_idx >= params.num_examples) {
        return;
    }

    // Get per-neuron metadata
    uint bits = neurons[neuron_idx].bits;
    uint conn_offset = neurons[neuron_idx].conn_offset;

    // Pointer to this example's packed input
    device const ulong* example_input = packed_input + example_idx * params.words_per_example;

    // Pointer to this neuron's connections
    device const int* neuron_conns = connections + conn_offset;

    // Compute and store address
    uint addr = compute_address(example_input, neuron_conns, bits);
    address_buffer[neuron_idx * params.num_examples + example_idx] = addr;
}
