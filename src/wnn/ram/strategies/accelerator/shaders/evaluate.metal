//
// Metal Compute Shader for RAM Connectivity Evaluation (v2 - Two-Pass)
//
// This implementation uses a two-pass strategy:
// 1. Training pass: Build hash tables in device memory using atomics
// 2. Evaluation pass: Use trained tables to make predictions
//
// Memory layout for hash tables:
// - Each (pattern, neuron, address) maps to (true_count, total_count)
// - Uses open addressing with linear probing for collisions
//

#include <metal_stdlib>
using namespace metal;

// Configuration constants
constant uint HASH_TABLE_SIZE = 256;  // Per-neuron hash table size (reduced for memory)
constant uint MAX_NEURONS = 256;      // Maximum neurons supported
constant uint CONTEXT_SIZE = 4;       // n-gram context size (4 words)
constant uint MAX_CONTEXT_BITS = 128; // Maximum input bits (4 words * 32 bits)

// Parameters passed from CPU
struct Params {
    uint n_patterns;
    uint n_neurons;
    uint bits_per_neuron;
    uint train_len;
    uint test_len;
    uint eval_subset;
};

// Hash table entry: packed (true_count:16, total_count:16) in uint32
// Using packing to halve memory usage
inline uint pack_counts(uint true_count, uint total_count) {
    return (min(true_count, 0xFFFFu) << 16) | min(total_count, 0xFFFFu);
}

inline uint2 unpack_counts(uint packed) {
    return uint2(packed >> 16, packed & 0xFFFF);
}

// Compute RAM address from context using connectivity pattern
inline uint compute_ram_address(
    device const int* connectivity,  // Pointer to this neuron's connectivity
    uint bits_per_neuron,
    thread const uint* context_words  // 4 context words
) {
    uint address = 0;
    for (uint b = 0; b < bits_per_neuron; b++) {
        int bit_idx = connectivity[b];
        if (bit_idx >= 0 && uint(bit_idx) < MAX_CONTEXT_BITS) {
            uint word_idx = bit_idx / 32;
            uint bit_pos = bit_idx % 32;
            if (word_idx < CONTEXT_SIZE) {
                uint bit = (context_words[word_idx] >> bit_pos) & 1;
                address |= (bit << b);
            }
        }
    }
    return address;
}

// Hash function for address -> table index
inline uint hash_address(uint address) {
    // FNV-1a inspired hash
    uint hash = address;
    hash ^= hash >> 16;
    hash *= 0x85ebca6bu;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35u;
    hash ^= hash >> 16;
    return hash % HASH_TABLE_SIZE;
}

//
// PASS 1: Training Kernel
// Each thread processes one (pattern, train_token) pair
// Updates hash tables using atomic operations
//
kernel void train_rams_pass1(
    device const int* connectivities [[buffer(0)]],     // [n_patterns, n_neurons, bits_per_neuron]
    device const uint* train_clusters [[buffer(1)]],    // [train_len]
    constant Params& params [[buffer(2)]],
    device atomic_uint* hash_tables [[buffer(3)]],      // [n_patterns, n_neurons, HASH_TABLE_SIZE]
    uint2 thread_pos [[thread_position_in_grid]]        // (pattern_idx, token_idx)
) {
    uint pattern_idx = thread_pos.x;
    uint token_idx = thread_pos.y;

    // Bounds check
    if (pattern_idx >= params.n_patterns) return;
    if (token_idx + CONTEXT_SIZE >= params.train_len) return;

    // Load context (4 consecutive tokens as cluster IDs)
    uint context[CONTEXT_SIZE];
    for (uint i = 0; i < CONTEXT_SIZE; i++) {
        context[i] = train_clusters[token_idx + i];
    }
    uint target = train_clusters[token_idx + CONTEXT_SIZE];

    // Get this pattern's connectivity
    device const int* my_conn = connectivities + pattern_idx * params.n_neurons * params.bits_per_neuron;

    // Train each neuron
    uint n_neurons_capped = min(params.n_neurons, (uint)MAX_NEURONS);
    for (uint n = 0; n < n_neurons_capped; n++) {
        // Compute RAM address for this neuron
        device const int* neuron_conn = my_conn + n * params.bits_per_neuron;
        uint ram_addr = compute_ram_address(neuron_conn, params.bits_per_neuron, context);

        // Hash to table index
        uint table_idx = hash_address(ram_addr);

        // Compute table offset: pattern * (n_neurons * HASH_TABLE_SIZE) + neuron * HASH_TABLE_SIZE + idx
        uint table_offset = (pattern_idx * params.n_neurons + n) * HASH_TABLE_SIZE + table_idx;

        // Get target bit for this neuron
        uint target_bit = (target >> n) & 1;

        // Atomic update: pack increment (true_count delta, total_count delta)
        // We add (1<<16) + 1 if target_bit=1, else just 1
        uint delta = target_bit ? 0x00010001 : 0x00000001;
        atomic_fetch_add_explicit(&hash_tables[table_offset], delta, memory_order_relaxed);
    }
}

//
// PASS 2: Evaluation Kernel
// Each thread processes one (pattern, test_token) pair
// Reads from hash tables to make predictions
//
kernel void evaluate_rams_pass2(
    device const int* connectivities [[buffer(0)]],     // [n_patterns, n_neurons, bits_per_neuron]
    device const uint* test_clusters [[buffer(1)]],     // [test_len]
    constant Params& params [[buffer(2)]],
    device const uint* hash_tables [[buffer(3)]],       // [n_patterns, n_neurons, HASH_TABLE_SIZE] (non-atomic read)
    device atomic_uint* correct_counts [[buffer(4)]],   // [n_patterns]
    device atomic_uint* covered_counts [[buffer(5)]],   // [n_patterns]
    uint2 thread_pos [[thread_position_in_grid]]        // (pattern_idx, test_token_idx)
) {
    uint pattern_idx = thread_pos.x;
    uint token_idx = thread_pos.y;

    // Bounds check
    if (pattern_idx >= params.n_patterns) return;
    if (token_idx >= params.eval_subset) return;
    if (token_idx + CONTEXT_SIZE >= params.test_len) return;

    // Load context
    uint context[CONTEXT_SIZE];
    for (uint i = 0; i < CONTEXT_SIZE; i++) {
        context[i] = test_clusters[token_idx + i];
    }
    uint target = test_clusters[token_idx + CONTEXT_SIZE];

    // Get this pattern's connectivity
    device const int* my_conn = connectivities + pattern_idx * params.n_neurons * params.bits_per_neuron;

    // Predict each bit and assemble prediction
    uint predicted = 0;
    uint neurons_with_data = 0;

    uint n_neurons_capped = min(params.n_neurons, (uint)MAX_NEURONS);
    for (uint n = 0; n < n_neurons_capped; n++) {
        // Compute RAM address
        device const int* neuron_conn = my_conn + n * params.bits_per_neuron;
        uint ram_addr = compute_ram_address(neuron_conn, params.bits_per_neuron, context);

        // Hash to table index
        uint table_idx = hash_address(ram_addr);

        // Read from hash table
        uint table_offset = (pattern_idx * params.n_neurons + n) * HASH_TABLE_SIZE + table_idx;
        uint packed = hash_tables[table_offset];
        uint2 counts = unpack_counts(packed);
        uint true_count = counts.x;
        uint total_count = counts.y;

        if (total_count > 0) {
            neurons_with_data++;
            // Predict 1 if true_count > half of total
            if (true_count * 2 > total_count) {
                predicted |= (1u << n);
            }
        }
    }

    // Check if prediction is correct
    bool is_covered = neurons_with_data >= n_neurons_capped / 2;  // At least half neurons have data
    bool is_correct = predicted == target;

    // Update counts atomically
    if (is_covered) {
        atomic_fetch_add_explicit(&covered_counts[pattern_idx], 1, memory_order_relaxed);
        if (is_correct) {
            atomic_fetch_add_explicit(&correct_counts[pattern_idx], 1, memory_order_relaxed);
        }
    }
}

//
// PASS 3: Finalize Results
// Each thread computes final error for one pattern
//
kernel void finalize_results(
    device const uint* correct_counts [[buffer(0)]],    // [n_patterns]
    device const uint* covered_counts [[buffer(1)]],    // [n_patterns]
    constant Params& params [[buffer(2)]],
    device float* results [[buffer(3)]],                // [n_patterns] output
    uint pattern_idx [[thread_position_in_grid]]
) {
    if (pattern_idx >= params.n_patterns) return;

    uint correct = correct_counts[pattern_idx];
    uint covered = covered_counts[pattern_idx];
    uint total = params.eval_subset;

    // Error = 1 - (accuracy * coverage)
    float accuracy = covered > 0 ? float(correct) / float(covered) : 0.0f;
    float coverage = total > 0 ? float(covered) / float(total) : 0.0f;

    results[pattern_idx] = 1.0f - (accuracy * coverage);
}

//
// Legacy single-pass kernel (kept for compatibility, but not accurate)
//
kernel void evaluate_connectivity(
    device const int* connectivities [[buffer(0)]],
    device const uint* train_clusters [[buffer(1)]],
    device const uint* test_clusters [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    device float* results [[buffer(4)]],
    uint pattern_idx [[thread_position_in_grid]]
) {
    // Redirect to placeholder - use two-pass kernels for accurate results
    if (pattern_idx >= params.n_patterns) return;
    results[pattern_idx] = 0.5f;  // Placeholder: 50% error
}
