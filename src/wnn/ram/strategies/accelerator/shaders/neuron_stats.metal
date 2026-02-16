//
// Metal Compute Shader for Adaptation Stats (Neuron + Cluster)
//
// Two-pass GPU computation for adaptation statistics:
//   Pass 1: Per-neuron forward pass + error/fill stats
//   Pass 2: Per-cluster majority vote + uniqueness/accuracy
//
// This replaces the CPU-bound compute_neuron_stats + compute_cluster_stats
// in adaptation.rs, achieving ~10x speedup on M4 Max (40 GPU cores).
//

#include <metal_stdlib>
using namespace metal;

// Bit-packing constants (must match neuron_memory.rs)
constant uint BITS_PER_CELL = 2;
constant uint CELLS_PER_WORD = 31;  // 62 bits / 2 = 31 cells per word
constant uint CELL_MASK = 0x3;

// Quad weights for vote computation (QUAD_WEIGHTED mode)
constant float QUAD_WEIGHTS[4] = {0.0f, 0.25f, 0.75f, 1.0f};

// Per-neuron metadata — describes storage layout for heterogeneous configs
struct NeuronMeta {
	uint cluster_id;        // Which cluster this neuron belongs to
	uint bits;              // Number of connection bits for this neuron
	uint conn_offset;       // Offset into connections[] array
	uint is_sparse;         // 0=dense, 1=sparse
	uint dense_word_offset; // Offset into dense_memory[] (dense only)
	uint words_per_neuron;  // Number of 64-bit words per neuron (dense only)
	uint sparse_neuron_idx; // Index into sparse arrays (sparse only)
};

// Shared params for both passes
struct StatsParams {
	uint num_neurons;       // Total neurons across all clusters
	uint sample_size;       // Number of sampled examples
	uint words_per_example; // Packed u64 words per example
	uint num_clusters;      // Number of clusters
	uint empty_cell_value;  // Empty cell value (2 for ternary, 1 for quad)
};

// Per-cluster metadata for Pass 2
struct ClusterMeta {
	uint first_neuron;      // Global neuron index of first neuron in cluster
	uint num_neurons;       // Number of neurons in this cluster
};

// =============================================================================
// Helper: Compute address from packed input + connections (MSB-first)
// =============================================================================

inline uint compute_address(
	device const ulong* packed_row,   // Packed u64 words for one example
	device const int*   connections,  // Connection indices for this neuron
	uint bits
) {
	uint address = 0;
	for (uint i = 0; i < bits; i++) {
		int conn_idx = connections[i];
		if (conn_idx >= 0) {
			uint word_idx = uint(conn_idx) / 64;
			uint bit_idx = uint(conn_idx) % 64;
			if ((packed_row[word_idx] >> bit_idx) & 1uL) {
				address |= (1u << (bits - 1 - i));
			}
		}
	}
	return address;
}

// =============================================================================
// Helper: Read dense cell
// =============================================================================

inline uint read_dense_cell(
	device const long* dense_memory,
	uint word_offset,       // Base offset for this neuron in dense_memory
	uint address
) {
	uint word_idx = address / CELLS_PER_WORD;
	uint cell_idx = address % CELLS_PER_WORD;
	long word = dense_memory[word_offset + word_idx];
	return uint(word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK;
}

// =============================================================================
// Helper: Binary search in sparse keys
// =============================================================================

inline uint sparse_lookup(
	device const ulong* sparse_keys,
	device const uchar* sparse_values,
	uint start,
	uint count,
	ulong address,
	uint default_value
) {
	if (count == 0) return default_value;

	uint lo = 0;
	uint hi = count;
	while (lo < hi) {
		uint mid = (lo + hi) / 2;
		ulong key = sparse_keys[start + mid];
		if (key < address) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	if (lo < count && sparse_keys[start + lo] == address) {
		return uint(sparse_values[start + lo]);
	}
	return default_value;
}

// =============================================================================
// Pass 1: Per-Neuron Stats Kernel
// =============================================================================
//
// Grid: (total_neurons, 1, 1) — one thread per neuron
//
// Each thread loops over sampled examples, computes forward pass,
// records vote (for Pass 2) and counts errors + filled cells.
//

kernel void compute_neuron_stats_kernel(
	device const ulong*       packed_input     [[buffer(0)]],   // [num_examples * words_per_example]
	device const int*         connections      [[buffer(1)]],   // Flattened connections
	device const uchar*       target_bits      [[buffer(2)]],   // [num_examples * num_clusters]
	device const uint*        sample_indices   [[buffer(3)]],   // [sample_size]
	device const long*        dense_memory     [[buffer(4)]],   // Flattened dense memory
	device const ulong*       sparse_keys      [[buffer(5)]],   // Sparse sorted keys
	device const uchar*       sparse_values    [[buffer(6)]],   // Sparse values
	device const uint*        sparse_offsets   [[buffer(7)]],   // Per-neuron offsets into sparse
	device const uint*        sparse_counts    [[buffer(8)]],   // Per-neuron counts in sparse
	device const NeuronMeta*  neuron_meta      [[buffer(9)]],   // Per-neuron metadata
	constant StatsParams&     params           [[buffer(10)]],
	device uint*              error_counts     [[buffer(11)]],  // [total_neurons] output
	device uint*              filled_counts    [[buffer(12)]],  // [total_neurons] output
	device uchar*             votes            [[buffer(13)]],  // [total_neurons * sample_size] output
	uint neuron_idx [[thread_position_in_grid]]
) {
	if (neuron_idx >= params.num_neurons) return;

	NeuronMeta meta = neuron_meta[neuron_idx];
	device const int* my_conns = connections + meta.conn_offset;

	uint errors = 0;
	uint filled = 0;

	// Loop over sampled examples
	for (uint s = 0; s < params.sample_size; s++) {
		uint ex = sample_indices[s];
		device const ulong* packed_row = packed_input + ex * params.words_per_example;

		// Compute address
		uint address = compute_address(packed_row, my_conns, meta.bits);

		// Read cell value
		uint cell;
		if (meta.is_sparse == 0) {
			// Dense path
			cell = read_dense_cell(dense_memory, meta.dense_word_offset, address);
		} else {
			// Sparse path: binary search
			uint sp_idx = meta.sparse_neuron_idx;
			cell = sparse_lookup(
				sparse_keys, sparse_values,
				sparse_offsets[sp_idx], sparse_counts[sp_idx],
				ulong(address), params.empty_cell_value
			);
		}

		// Compute vote using quad weights (always QUAD_WEIGHTED in bitwise path)
		float vote_val = QUAD_WEIGHTS[min(cell, 3u)];
		uchar vote = (vote_val >= 0.5f) ? 1 : 0;

		// Store vote for Pass 2
		votes[neuron_idx * params.sample_size + s] = vote;

		// Compare with target
		uchar target = target_bits[ex * params.num_clusters + meta.cluster_id];
		if (vote != target) {
			errors++;
		}
	}

	// Count filled cells from storage
	if (meta.is_sparse == 0) {
		// Dense: scan all words for non-empty cells
		uint total_cells = 1u << meta.bits;
		uint n_words = meta.words_per_neuron;
		for (uint w = 0; w < n_words; w++) {
			long word = dense_memory[meta.dense_word_offset + w];
			for (uint c = 0; c < CELLS_PER_WORD; c++) {
				uint cell_val = uint(word >> (c * BITS_PER_CELL)) & CELL_MASK;
				if (cell_val != params.empty_cell_value && filled < total_cells) {
					filled++;
				}
			}
		}
	} else {
		// Sparse: count is directly available
		filled = sparse_counts[meta.sparse_neuron_idx];
	}

	error_counts[neuron_idx] = errors;
	filled_counts[neuron_idx] = filled;
}

// =============================================================================
// Pass 2: Cluster Stats Reduction Kernel
// =============================================================================
//
// Grid: (num_clusters, sample_size) — one thread per (cluster, sample)
//
// Each thread reads all neuron votes for its cluster, computes majority,
// then updates cluster_errors, uniqueness_counts, accuracy_counts via atomics.
//

kernel void compute_cluster_stats_kernel(
	device const uchar*        votes            [[buffer(0)]],   // [total_neurons * sample_size]
	device const uchar*        target_bits      [[buffer(1)]],   // [num_examples * num_clusters]
	device const uint*         sample_indices   [[buffer(2)]],   // [sample_size]
	device const ClusterMeta*  cluster_meta     [[buffer(3)]],   // [num_clusters]
	constant StatsParams&      params           [[buffer(4)]],
	device atomic_uint*        cluster_errors   [[buffer(5)]],   // [num_clusters] output
	device atomic_uint*        uniqueness_counts [[buffer(6)]],  // [total_neurons] output
	device atomic_uint*        accuracy_counts  [[buffer(7)]],   // [total_neurons] output
	uint2 pos [[thread_position_in_grid]]                        // (cluster_idx, sample_idx)
) {
	uint cluster_idx = pos.x;
	uint sample_idx = pos.y;

	if (cluster_idx >= params.num_clusters || sample_idx >= params.sample_size) return;

	ClusterMeta cm = cluster_meta[cluster_idx];
	uint ex = sample_indices[sample_idx];
	uchar target = target_bits[ex * params.num_clusters + cluster_idx];
	bool target_true = (target != 0);

	// Count votes_true across neurons in this cluster
	uint votes_true = 0;
	for (uint n = 0; n < cm.num_neurons; n++) {
		uint global_n = cm.first_neuron + n;
		uchar v = votes[global_n * params.sample_size + sample_idx];
		if (v != 0) votes_true++;
	}

	// Majority vote
	bool majority_true = (votes_true > cm.num_neurons / 2);

	// Cluster error
	if (majority_true != target_true) {
		atomic_fetch_add_explicit(&cluster_errors[cluster_idx], 1u, memory_order_relaxed);
	}

	// Per-neuron uniqueness and accuracy
	for (uint n = 0; n < cm.num_neurons; n++) {
		uint global_n = cm.first_neuron + n;
		uchar v = votes[global_n * params.sample_size + sample_idx];
		bool neuron_true = (v != 0);

		// Uniqueness: disagrees with majority
		if (neuron_true != majority_true) {
			atomic_fetch_add_explicit(&uniqueness_counts[global_n], 1u, memory_order_relaxed);
		}

		// Accuracy: agrees with target
		if (neuron_true == target_true) {
			atomic_fetch_add_explicit(&accuracy_counts[global_n], 1u, memory_order_relaxed);
		}
	}
}
