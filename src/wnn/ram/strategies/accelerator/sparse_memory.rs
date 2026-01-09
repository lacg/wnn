//! Sparse Memory Backend for RAM Neurons
//!
//! Uses FxHashMap for sparse storage, optimal for neurons with >10 bits per neuron.
//! Memory usage is O(written cells) instead of O(2^n_bits), making it feasible
//! for larger bit configurations (15-30 bits).
//!
//! Trade-offs vs Dense:
//! - Memory: O(cells_written) vs O(2^n_bits * neurons) - MUCH better for large bits
//! - Lookup: O(1) average but with hash overhead vs O(1) direct indexing
//! - Best for: >10 bits per neuron where 2^n becomes impractically large
//!
//! Cell values (matches dense backend):
//! - FALSE = 0
//! - TRUE = 1
//! - EMPTY = 2 (default for unwritten cells)

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::RwLock;

/// Memory cell values (matches dense backend)
const FALSE: u8 = 0;
const TRUE: u8 = 1;
const EMPTY: u8 = 2;

/// Sparse memory storage for all neurons in a layer
/// Uses per-neuron RwLock for thread-safe parallel access
pub struct SparseLayerMemory {
    /// Per-neuron hash maps: address -> cell value
    neurons: Vec<RwLock<FxHashMap<u64, u8>>>,
    num_neurons: usize,
    bits_per_neuron: usize,
}

impl SparseLayerMemory {
    /// Create new sparse layer with given number of neurons
    pub fn new(num_neurons: usize, bits_per_neuron: usize) -> Self {
        let neurons: Vec<_> = (0..num_neurons)
            .map(|_| RwLock::new(FxHashMap::default()))
            .collect();

        Self {
            neurons,
            num_neurons,
            bits_per_neuron,
        }
    }

    /// Read cell value for a specific neuron and address
    /// Returns EMPTY for unwritten cells
    #[inline]
    pub fn read_cell(&self, neuron_idx: usize, address: u64) -> u8 {
        self.neurons[neuron_idx]
            .read()
            .unwrap()
            .get(&address)
            .copied()
            .unwrap_or(EMPTY)
    }

    /// Write cell value for a specific neuron and address
    /// Returns true if the cell was modified
    #[inline]
    pub fn write_cell(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool {
        let mut map = self.neurons[neuron_idx].write().unwrap();
        let current = map.get(&address).copied().unwrap_or(EMPTY);

        // Only write if EMPTY or allow_override
        if !allow_override && current != EMPTY {
            return false;
        }

        // If value already matches, no need to write
        if current == value {
            return false;
        }

        map.insert(address, value);
        true
    }

    /// Get total number of written cells across all neurons (for stats)
    pub fn total_cells(&self) -> usize {
        self.neurons.iter()
            .map(|n| n.read().unwrap().len())
            .sum()
    }

    /// Get per-neuron cell counts
    pub fn cell_counts(&self) -> Vec<usize> {
        self.neurons.iter()
            .map(|n| n.read().unwrap().len())
            .collect()
    }

    /// Export to flat representation: Vec<(neuron_idx, address, value)>
    pub fn export(&self) -> Vec<(usize, u64, u8)> {
        let mut cells: Vec<(usize, u64, u8)> = Vec::new();

        for (neuron_idx, neuron_lock) in self.neurons.iter().enumerate() {
            let neuron = neuron_lock.read().unwrap();
            for (&address, &value) in neuron.iter() {
                cells.push((neuron_idx, address, value));
            }
        }

        cells
    }

    /// Import from flat representation
    pub fn import(&self, cells: &[(usize, u64, u8)]) {
        for &(neuron_idx, address, value) in cells {
            if neuron_idx < self.num_neurons {
                self.neurons[neuron_idx].write().unwrap().insert(address, value);
            }
        }
    }

    /// Reset all memory to empty
    pub fn reset(&self) {
        for neuron in &self.neurons {
            neuron.write().unwrap().clear();
        }
    }
}

/// Compute address from input bits and connections (MSB-first, matches dense backend)
#[inline]
fn compute_address(
    input_bits: &[bool],
    connections: &[i64],
    bits_per_neuron: usize,
) -> u64 {
    let mut address: u64 = 0;
    for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
        if input_bits[conn_idx as usize] {
            // MSB-first addressing (matches Python and dense Rust)
            address |= 1 << (bits_per_neuron - 1 - i);
        }
    }
    address
}

/// Batch training for sparse memory backend
///
/// Same interface as ramlm::train_batch but uses sparse storage.
/// Two-phase training: TRUEs first (with override), then FALSEs (no override)
pub fn train_batch_sparse(
    memory: &SparseLayerMemory,
    input_bits_flat: &[bool],
    true_clusters: &[i64],
    false_clusters_flat: &[i64],
    connections_flat: &[i64],
    num_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_negatives: usize,
    _allow_override: bool,
) -> usize {
    // Phase 1: Write ALL TRUEs first (with override to ensure TRUE takes priority)
    let true_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let mut ex_modified = 0usize;

        let true_cluster = true_clusters[ex_idx] as usize;
        let true_start_neuron = true_cluster * neurons_per_cluster;

        for neuron_offset in 0..neurons_per_cluster {
            let neuron_idx = true_start_neuron + neuron_offset;
            let conn_start = neuron_idx * bits_per_neuron;
            let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];

            let address = compute_address(input_bits, connections, bits_per_neuron);

            // TRUE writes always override (priority over FALSE)
            if memory.write_cell(neuron_idx, address, TRUE, true) {
                ex_modified += 1;
            }
        }

        ex_modified
    }).sum();

    // Phase 2: Write ALL FALSEs (only to EMPTY cells)
    let false_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let mut ex_modified = 0usize;

        let false_start = ex_idx * num_negatives;
        for neg_idx in 0..num_negatives {
            let false_cluster = false_clusters_flat[false_start + neg_idx] as usize;
            let false_start_neuron = false_cluster * neurons_per_cluster;

            for neuron_offset in 0..neurons_per_cluster {
                let neuron_idx = false_start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];

                let address = compute_address(input_bits, connections, bits_per_neuron);

                // FALSE writes only to EMPTY cells
                if memory.write_cell(neuron_idx, address, FALSE, false) {
                    ex_modified += 1;
                }
            }
        }

        ex_modified
    }).sum();

    true_modified + false_modified
}

/// Batch forward pass for sparse memory backend
///
/// Same interface as ramlm::forward_batch but uses sparse storage.
pub fn forward_batch_sparse(
    memory: &SparseLayerMemory,
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    num_examples: usize,
    total_input_bits: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
) -> Vec<f32> {
    let mut probs = vec![0.0f32; num_examples * num_clusters];

    // Process all examples in parallel
    probs.par_chunks_mut(num_clusters).enumerate().for_each(|(ex_idx, ex_probs)| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        // For each cluster, count TRUE and EMPTY
        for cluster_idx in 0..num_clusters {
            let start_neuron = cluster_idx * neurons_per_cluster;
            let mut count_true = 0u32;
            let mut count_empty = 0u32;

            for neuron_offset in 0..neurons_per_cluster {
                let neuron_idx = start_neuron + neuron_offset;
                let conn_start = neuron_idx * bits_per_neuron;
                let connections = &connections_flat[conn_start..conn_start + bits_per_neuron];

                let address = compute_address(input_bits, connections, bits_per_neuron);
                let cell_value = memory.read_cell(neuron_idx, address);

                if cell_value == TRUE {
                    count_true += 1;
                } else if cell_value == EMPTY {
                    count_empty += 1;
                }
            }

            // Probability = (count_true + 0.5 * count_empty) / neurons_per_cluster
            ex_probs[cluster_idx] = (count_true as f32 + 0.5 * count_empty as f32) / neurons_per_cluster as f32;
        }
    });

    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_basic() {
        let layer = SparseLayerMemory::new(10, 20);

        // Read unwritten cell should return EMPTY
        assert_eq!(layer.read_cell(0, 0), EMPTY);
        assert_eq!(layer.read_cell(5, 1000), EMPTY);

        // Write TRUE
        assert!(layer.write_cell(0, 0, TRUE, false));
        assert_eq!(layer.read_cell(0, 0), TRUE);

        // Write FALSE (different address)
        assert!(layer.write_cell(0, 1, FALSE, false));
        assert_eq!(layer.read_cell(0, 1), FALSE);

        // Cannot overwrite without allow_override
        assert!(!layer.write_cell(0, 0, FALSE, false));
        assert_eq!(layer.read_cell(0, 0), TRUE);

        // Can overwrite with allow_override
        assert!(layer.write_cell(0, 0, FALSE, true));
        assert_eq!(layer.read_cell(0, 0), FALSE);

        assert_eq!(layer.total_cells(), 2);
    }

    #[test]
    fn test_sparse_parallel() {
        let layer = SparseLayerMemory::new(100, 20);

        // Write to different neurons in parallel
        (0..100usize).into_par_iter().for_each(|i| {
            layer.write_cell(i, i as u64, TRUE, false);
        });

        // Verify all writes
        for i in 0..100 {
            assert_eq!(layer.read_cell(i, i as u64), TRUE);
        }

        assert_eq!(layer.total_cells(), 100);
    }

    #[test]
    fn test_export_import() {
        let layer1 = SparseLayerMemory::new(10, 20);
        layer1.write_cell(0, 100, TRUE, false);
        layer1.write_cell(5, 200, FALSE, false);
        layer1.write_cell(9, 300, TRUE, false);

        let exported = layer1.export();
        assert_eq!(exported.len(), 3);

        let layer2 = SparseLayerMemory::new(10, 20);
        layer2.import(&exported);

        assert_eq!(layer2.read_cell(0, 100), TRUE);
        assert_eq!(layer2.read_cell(5, 200), FALSE);
        assert_eq!(layer2.read_cell(9, 300), TRUE);
    }
}
