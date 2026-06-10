// =============================================================================
// common.metal — single source of truth for cell semantics on the GPU.
//
// PREPENDED to every main-memory shader at compile time (see the concat!
// shader-load sites in metal_ramlm.rs / metal_stats.rs / marker_train.rs /
// metal_train.rs / metal_controller.rs). Mirrors neuron_memory.rs — if you
// change a constant or helper here, change it there too. Parity tests:
//   - cargo test cpu_fallback_matches_gpu   (dense + sparse scoring)
//   - cargo test metal_shaders_compile      (all shaders build with common)
//   - run_marker_train_parity_test          (training writes, from Python)
//
// History: per-shader copies of QUAD_WEIGHTS and compute_address caused the
// LSB-first marker_train bug (15/05/2026) and the u32-truncation bug
// (14/05/2026). Do NOT re-introduce local copies.
//
// NOTE: gating.metal / gating.rs deliberately use an LSB-first address
// convention (an internally-consistent pair, never mixed with main memory)
// — gating is NOT part of this family.
// =============================================================================
#include <metal_stdlib>
using namespace metal;

// --- Ternary cell values ---
constant uint WNN_CELL_FALSE = 0;
constant uint WNN_CELL_TRUE = 1;
constant uint WNN_CELL_EMPTY = 2;

// --- Cell encoding (2-bit cells packed into i64 words) ---
constant uint WNN_BITS_PER_CELL = 2;
constant uint WNN_CELLS_PER_WORD = 31;
constant uint WNN_CELL_MASK = 0x3;

// --- Memory modes (must match neuron_memory.rs MODE_*) ---
constant uint WNN_MODE_TERNARY = 0;
constant uint WNN_MODE_QUAD_BINARY = 1;
constant uint WNN_MODE_QUAD_WEIGHTED = 2;

// --- QUAD_WEIGHTED forward weights: FALSE, WEAK_FALSE, WEAK_TRUE, TRUE ---
// (must match neuron_memory.rs QUAD_WEIGHTS)
constant float WNN_QUAD_WEIGHTS[4] = {0.0f, 0.25f, 0.75f, 1.0f};

// MSB-first address computation: connection i contributes bit (bits-1-i) of
// the address. 64-bit accumulator end-to-end — uint accumulation caused the
// u32-truncation bug for bits > 32.
inline ulong wnn_compute_address_u64(
    device const ulong* packed_input,   // [words_per_example] for this example
    device const int* connections,      // [bits] for this neuron
    uint bits
) {
    ulong address = 0;
    for (uint i = 0; i < bits; i++) {
        int conn_idx = connections[i];
        if (conn_idx >= 0) {
            uint word_idx = uint(conn_idx) / 64;
            uint bit_idx = uint(conn_idx) % 64;
            if ((packed_input[word_idx] >> bit_idx) & 1uL) {
                address |= (1uL << (bits - 1u - i));
            }
        }
    }
    return address;
}

// Dense variant — safe to truncate: dense memory caps bits at the sparse
// threshold (well below 32).
inline uint wnn_compute_address(
    device const ulong* packed_input,
    device const int* connections,
    uint bits
) {
    return uint(wnn_compute_address_u64(packed_input, connections, bits));
}

// Read a 2-bit cell from dense packed memory words.
inline uint wnn_read_cell(
    device const long* memory_words,
    uint neuron_idx,
    uint address,
    uint words_per_neuron
) {
    uint word_idx = address / WNN_CELLS_PER_WORD;
    uint cell_idx = address % WNN_CELLS_PER_WORD;
    uint word_offset = neuron_idx * words_per_neuron + word_idx;
    long word = memory_words[word_offset];
    // Shift in 64-bit first, then truncate (uint(word) would drop upper bits)
    return uint(word >> (cell_idx * WNN_BITS_PER_CELL)) & WNN_CELL_MASK;
}

// Cell → forward-pass weight (mirrors neuron_memory.rs cell_to_weight).
// TERNARY: FALSE=0, TRUE=1, EMPTY=empty_value.
// QUAD modes: table lookup — ternary and quad encodings agree only on cell 0.
inline float wnn_cell_weight(uint cell, uint memory_mode, float empty_value) {
    if (memory_mode == WNN_MODE_TERNARY) {
        if (cell == 0u) { return 0.0f; }
        if (cell == 1u) { return 1.0f; }
        return empty_value;
    }
    return WNN_QUAD_WEIGHTS[min(cell, 3u)];
}
