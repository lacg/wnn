// WNNInference.metal - GPU compute shader for WNN neuron evaluation
//
// Each thread evaluates one neuron:
// 1. Read connection indices to find which input bits this neuron observes
// 2. Build a memory address from those bits
// 3. Look up the trained value in the memory table

#include <metal_stdlib>
using namespace metal;

kernel void wnn_evaluate_neurons(
	device const uint*    input_words      [[buffer(0)]],   // Packed input bits (UInt32)
	device const long*    connections      [[buffer(1)]],   // Flat connection indices per neuron
	device const float*   memory_tables    [[buffer(2)]],   // Flat memory values
	device const uint*    bits_per_neuron  [[buffer(3)]],   // Bits count per neuron
	device const uint*    memory_offsets   [[buffer(4)]],   // Offset into memory_tables per neuron
	device const uint*    conn_offsets     [[buffer(5)]],   // Offset into connections per neuron
	device float*         output           [[buffer(6)]],   // Per-neuron output value
	uint                  tid              [[thread_position_in_grid]]
) {
	uint bits = bits_per_neuron[tid];
	uint conn_start = conn_offsets[tid];
	uint mem_offset = memory_offsets[tid];

	// Build address from input bits via connections
	uint address = 0;
	for (uint b = 0; b < bits; b++) {
		long bit_idx = connections[conn_start + b];
		if (bit_idx >= 0) {
			uint word_idx = uint(bit_idx) / 32;
			uint bit_pos = uint(bit_idx) % 32;
			if ((input_words[word_idx] >> bit_pos) & 1) {
				address |= (1u << (bits - 1 - b));
			}
		}
	}

	// Look up trained memory value
	output[tid] = memory_tables[mem_offset + address];
}
